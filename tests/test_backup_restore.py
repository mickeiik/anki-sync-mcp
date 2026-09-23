from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from anki._backend import RustBackend
from anki.collection import Collection
from anki.sync import SyncAuth
from starlette.testclient import TestClient

from anki_mcp.app import create_app
from anki_mcp.collection import (
    AnkiCollectionService,
    CollectionAdapter,
    RestoreFailedError,
    SyncLoginRequiredError,
)
from anki_mcp.config import Settings


def _seed_collection(tmp_path: Path) -> tuple[str, int, int]:
    path = tmp_path / "collection.anki2"
    collection = Collection(str(path))
    try:
        model = collection.models.by_name("Basic")
        assert model is not None
        note_type_id = int(model["id"])
        deck_id = int(collection.decks.id("Restore"))
        note = collection.new_note(model)
        note["Front"] = "A"
        note["Back"] = "a"
        collection.add_note(note, deck_id)
    finally:
        collection.close()
    return str(path), note_type_id, deck_id


def _fronts(payload: dict[str, Any]) -> set[str]:
    return {str(item["first_field"]) for item in payload["items"]}


def _rename_backup(created: dict[str, Any], filename: str = "target.colpkg") -> str:
    """Give a created backup a stable name.

    Native backup filenames have one-second resolution, so back-to-back
    ``create_backup`` calls would otherwise overwrite each other (including via the
    pre-restore backup that guarded mutations create).
    """
    source = Path(str(created["path"]))
    target = source.parent / filename
    target.unlink(missing_ok=True)
    source.rename(target)
    return filename


def _free_loopback_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


@pytest.fixture
def official_sync_server(tmp_path: Path) -> Iterator[str]:
    """Run the sync server shipped by the pinned official Anki package."""
    port = _free_loopback_port()
    sync_base = tmp_path / "sync-server"
    sync_base.mkdir()
    environment = {
        **os.environ,
        "SYNC_BASE": str(sync_base),
        "SYNC_HOST": "127.0.0.1",
        "SYNC_PORT": str(port),
        "SYNC_USER1": "phase1-user:phase1-password",
        "RUST_LOG": "anki=warn",
    }
    process = subprocess.Popen(  # noqa: S603 - fixed interpreter/module, test-only environment
        [sys.executable, "-m", "anki.syncserver"],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            pytest.fail(f"official sync server exited during startup: {output}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        process.terminate()
        pytest.fail("official sync server did not become ready")

    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_list_and_preview_backups(tmp_path: Path) -> None:
    path, _, _ = _seed_collection(tmp_path)

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            empty = await service.list_backups(0, 10)
            assert empty["items"] == []
            assert empty["total"] == 0
            created = await service.create_backup()
            assert created["created"] is True
            listing = await service.list_backups(0, 10)
            assert listing["total"] == 1
            item = listing["items"][0]
            assert str(item["filename"]).endswith(".colpkg")
            assert int(item["size_bytes"]) > 0
            assert float(item["mtime"]) > 0
            preview = await service.preview_backup_restore(str(item["filename"]))
            assert preview["validated"] is True
            assert preview["current"]["notes"] == 1
            assert preview["current"]["cards"] == 1
            assert preview["post_restore_modes"] == ["upload_now", "local_only"]

    asyncio.run(scenario())


def test_backup_filename_and_content_validation(tmp_path: Path) -> None:
    path, _, _ = _seed_collection(tmp_path)
    backup_folder = Path(path).parent / "backups"

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            created = await service.create_backup()
            good = Path(str(created["path"])).name
            (backup_folder / "corrupt.colpkg").write_bytes(b"this is not a zip archive")
            with pytest.raises(ValueError, match="not a valid Anki collection package"):
                await service.preview_backup_restore("corrupt.colpkg")
            for bad in ("../evil.colpkg", "nested/evil.colpkg", "evil.txt"):
                with pytest.raises(ValueError):
                    await service.preview_backup_restore(bad)
            with pytest.raises(LookupError):
                await service.preview_backup_restore("missing.colpkg")
            (backup_folder / "link.colpkg").symlink_to(backup_folder / good)
            with pytest.raises(ValueError, match="symbolic link"):
                await service.preview_backup_restore("link.colpkg")

    asyncio.run(scenario())


def test_local_only_restore_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, note_type_id, deck_id = _seed_collection(tmp_path)
    calls = {"count": 0}

    def flaky_sync(self: CollectionAdapter, sync_media: bool) -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RuntimeError("post-sync failed")
        return {"required": "NO_CHANGES"}

    monkeypatch.setattr(CollectionAdapter, "_sync_or_raise_full_sync", flaky_sync)

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100, sync_on_write=True) as service:
            created = await service.create_backup()
            name = _rename_backup(created)
            receipt = await service.coordinated_mutation(
                "anki_notes_create",
                "note-b-key",
                {"front": "B"},
                lambda adapter: adapter.create_note(
                    deck_id, note_type_id, {"Front": "B", "Back": "b"}, []
                ),
            )
            assert receipt["remote_synced"] is False
            assert receipt["retryable"] is True

            result = await service.restore_backup(name, "local_only")
            assert result["restored"] is True
            assert result["mode"] == "local_only"
            assert result["remote_replaced"] is False
            assert result["target_overwritten"] is False
            assert isinstance(result["warning"], str) and result["warning"]
            # No sync login in this scenario, so the warning must not claim
            # knowledge of the server's state.
            assert "sync login" in result["warning"]

            status = await service.status()
            assert status["post_restore_upload_pending"] is True
            listing = await service.coordinated_read(
                lambda adapter: adapter.search_notes("", 0, 10)
            )
            assert _fronts(listing) == {"A"}

            operation = await service.get_operation("note-b-key")
            assert operation["receipt"]["state"] == "discarded_by_restore"
            assert operation["receipt"]["local_committed"] is False

    asyncio.run(scenario())


def test_restore_uses_previewed_content_when_target_is_replaced_during_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, note_type_id, deck_id = _seed_collection(tmp_path)
    backup_folder = Path(path).parent / "backups"
    original_create_backup = CollectionAdapter.create_backup

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            target_backup = await service.create_backup()
            target_name = _rename_backup(target_backup, "target.colpkg")
            target_path = backup_folder / target_name

            await service.executor.run(
                lambda adapter: adapter.create_note(
                    deck_id, note_type_id, {"Front": "B", "Back": "b"}, []
                )
            )
            current_backup = await service.create_backup()
            current_path = Path(str(current_backup["path"]))

            def clobbering_create_backup(self: CollectionAdapter) -> dict[str, Any]:
                # Emulate the native same-second overwrite of the target filename
                # that a pre-restore backup can cause.
                shutil.copy2(current_path, target_path)
                return original_create_backup(self)

            monkeypatch.setattr(CollectionAdapter, "create_backup", clobbering_create_backup)

            result = await service.restore_backup(target_name, "local_only")
            assert result["restored"] is True
            assert result["target_overwritten"] is True
            assert Path(str(result["backup"]["path"])).is_file()

            listing = await service.coordinated_read(
                lambda adapter: adapter.search_notes("", 0, 10)
            )
            assert _fronts(listing) == {"A"}

    asyncio.run(scenario())


@pytest.fixture
def restore_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Settings, int, int]:
    path, note_type_id, deck_id = _seed_collection(tmp_path)
    monkeypatch.setenv("MCP_AUTH_TOKEN", "restore-token")
    monkeypatch.setenv("ANKI_COLLECTION_PATH", path)
    monkeypatch.setenv("MCP_SCOPES", "read,write,admin,destructive")
    monkeypatch.setenv("ANKI_ALLOW_DESTRUCTIVE", "true")
    monkeypatch.setenv("ANKI_ALLOW_FULL_SYNC", "true")
    monkeypatch.setenv("ANKI_ALLOW_RESTORE", "true")
    monkeypatch.setenv("ANKI_SYNC_ON_WRITE", "false")
    monkeypatch.setenv("ANKI_SYNC_HOST", "https://sync.example.test/")
    return Settings(_env_file=None), note_type_id, deck_id


def _session(
    client: TestClient,
) -> tuple[dict[str, str], Any, Any]:
    headers = {
        "Authorization": "Bearer restore-token",
        "Accept": "application/json, text/event-stream",
    }
    initialized = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1"},
            },
        },
    )
    headers["Mcp-Session-Id"] = initialized.headers["mcp-session-id"]
    request_id = 2

    def call(name: str, arguments: dict[str, object]) -> dict[str, Any]:
        nonlocal request_id
        response = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        request_id += 1
        result = response.json()["result"]
        assert result.get("isError") is not True, result
        parsed = json.loads(result["content"][0]["text"])
        assert isinstance(parsed, dict)
        return parsed

    def call_error(name: str, arguments: dict[str, object]) -> dict[str, Any]:
        nonlocal request_id
        response = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        request_id += 1
        result = response.json()["result"]
        assert result["isError"] is True, result
        text = result["content"][0]["text"]
        return json.loads(text[text.index("{") :])

    return headers, call, call_error


def test_restore_requires_matching_preview_token(
    restore_settings: tuple[Settings, int, int],
) -> None:
    settings, note_type_id, deck_id = restore_settings
    with TestClient(create_app(settings)) as client:
        _, call, call_error = _session(client)
        created = call("anki_backup_create", {})
        name = _rename_backup(created)
        call(
            "anki_notes_create",
            {
                "deck_id": deck_id,
                "note_type_id": note_type_id,
                "fields": {"Front": "B", "Back": "b"},
                "idempotency_key": "b-key",
            },
        )
        preview = call("anki_backup_restore_preview", {"filename": name})
        token = preview["confirmation_token"]

        omitted = call_error(
            "anki_backup_restore",
            {"filename": name, "mode": "local_only", "idempotency_key": "omitted"},
        )
        assert omitted["code"] == "INVALID_ARGUMENT"

        wrong = call_error(
            "anki_backup_restore",
            {
                "filename": name,
                "mode": "local_only",
                "confirmation_token": "not-a-real-token",
                "idempotency_key": "wrong",
            },
        )
        assert wrong["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"

        call(
            "anki_notes_create",
            {
                "deck_id": deck_id,
                "note_type_id": note_type_id,
                "fields": {"Front": "C", "Back": "c"},
                "idempotency_key": "c-key",
            },
        )
        stale = call_error(
            "anki_backup_restore",
            {
                "filename": name,
                "mode": "local_only",
                "confirmation_token": token,
                "idempotency_key": "stale",
            },
        )
        assert stale["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"

        fresh = call("anki_backup_restore_preview", {"filename": name})
        applied = call(
            "anki_backup_restore",
            {
                "filename": name,
                "mode": "local_only",
                "confirmation_token": fresh["confirmation_token"],
                "idempotency_key": "applied",
            },
        )
        assert applied["local_committed"] is True
        assert applied["remote_synced"] is None
        assert applied["result"]["restored"] is True
        assert Path(str(applied["result"]["backup"]["path"])).is_file()
        search = call("anki_notes_search", {})
        assert _fronts(search) == {"A"}


def test_restore_replay_returns_receipt_without_reapplying(
    restore_settings: tuple[Settings, int, int],
) -> None:
    settings, note_type_id, deck_id = restore_settings
    with TestClient(create_app(settings)) as client:
        _, call, _ = _session(client)
        created = call("anki_backup_create", {})
        name = _rename_backup(created)
        preview = call("anki_backup_restore_preview", {"filename": name})
        applied = call(
            "anki_backup_restore",
            {
                "filename": name,
                "mode": "local_only",
                "confirmation_token": preview["confirmation_token"],
                "idempotency_key": "replay-key",
            },
        )
        assert applied["result"]["restored"] is True

        call(
            "anki_notes_create",
            {
                "deck_id": deck_id,
                "note_type_id": note_type_id,
                "fields": {"Front": "SURVIVOR", "Back": "s"},
                "idempotency_key": "survivor-key",
            },
        )
        replayed = call(
            "anki_backup_restore",
            {
                "filename": name,
                "mode": "local_only",
                "confirmation_token": preview["confirmation_token"],
                "idempotency_key": "replay-key",
            },
        )
        assert replayed == applied
        search = call("anki_notes_search", {})
        assert "SURVIVOR" in _fronts(search)


def test_forced_upload_after_local_only_restore(
    official_sync_server: str, tmp_path: Path
) -> None:
    path, note_type_id, deck_id = _seed_collection(tmp_path)

    async def establish_and_restore() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            created = await service.create_backup()
            name = _rename_backup(created)
            await service.coordinated_mutation(
                "anki_notes_create",
                "b-key",
                {"front": "B"},
                lambda adapter: adapter.create_note(
                    deck_id, note_type_id, {"Front": "B", "Back": "b"}, []
                ),
            )
            await service.sync_login("phase1-user", "phase1-password", official_sync_server)
            required = (await service.sync(sync_media=False))["required"]
            assert required in {"FULL_SYNC", "FULL_UPLOAD"}
            await service.full_sync(upload=True)

            restored = await service.restore_backup(name, "local_only")
            assert restored["remote_replaced"] is False

            with pytest.raises(ValueError, match="full sync was not requested"):
                await service.full_sync(upload=True)
            forced = await service.full_sync(upload=True, force=True)
            assert forced["direction"] == "upload"
            assert (await service.status())["post_restore_upload_pending"] is False

    asyncio.run(establish_and_restore())

    second_path = tmp_path / "client-b" / "collection.anki2"
    second_path.parent.mkdir()
    Collection(str(second_path)).close()

    async def verify_download() -> None:
        async with AnkiCollectionService(str(second_path), max_page_size=100) as second:
            await second.sync_login("phase1-user", "phase1-password", official_sync_server)
            required = (await second.sync(sync_media=False))["required"]
            assert required == "FULL_DOWNLOAD"
            await second.full_sync(upload=False)
            listing = await second.coordinated_read(
                lambda adapter: adapter.search_notes("", 0, 10)
            )
            assert _fronts(listing) == {"A"}

    asyncio.run(verify_download())


def test_upload_now_replaces_server_in_one_call(
    official_sync_server: str, tmp_path: Path
) -> None:
    path, note_type_id, deck_id = _seed_collection(tmp_path)

    async def restore_and_upload() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            created = await service.create_backup()
            name = _rename_backup(created)
            await service.coordinated_mutation(
                "anki_notes_create",
                "b-key",
                {"front": "B"},
                lambda adapter: adapter.create_note(
                    deck_id, note_type_id, {"Front": "B", "Back": "b"}, []
                ),
            )
            await service.sync_login("phase1-user", "phase1-password", official_sync_server)
            result = await service.restore_backup(name, "upload_now")
            assert result["restored"] is True
            assert result["remote_replaced"] is True
            assert result["media_sync_required"] is True
            assert isinstance(result["warning"], str)
            assert "anki_sync(sync_media=true)" in result["warning"]
            assert (await service.status())["post_restore_upload_pending"] is False

    asyncio.run(restore_and_upload())

    second_path = tmp_path / "client-b" / "collection.anki2"
    second_path.parent.mkdir()
    Collection(str(second_path)).close()

    async def verify_download() -> None:
        async with AnkiCollectionService(str(second_path), max_page_size=100) as second:
            await second.sync_login("phase1-user", "phase1-password", official_sync_server)
            required = (await second.sync(sync_media=False))["required"]
            assert required == "FULL_DOWNLOAD"
            await second.full_sync(upload=False)
            listing = await second.coordinated_read(
                lambda adapter: adapter.search_notes("", 0, 10)
            )
            assert _fronts(listing) == {"A"}

    asyncio.run(verify_download())


class _StubSyncOutput:
    """Minimal stand-in for the protobuf fields ``CollectionAdapter.sync`` reads."""

    def __init__(self, required: int) -> None:
        self.required = required
        self.server_media_usn = None
        self.new_endpoint = ""
        self.server_message = ""
        self.host_number = 0


class _StubSyncStatus:
    def __init__(self, required: int) -> None:
        self.required = required


def _post_restore_status(path: str) -> bool:
    status = json.loads(
        (Path(path).parent / "state" / "operation-status.json").read_text(encoding="utf-8")
    )
    return bool(status["post_restore_upload"])


def _tool_names_for(settings: Settings) -> set[str]:
    headers = {
        "Authorization": "Bearer restore-token",
        "Accept": "application/json, text/event-stream",
    }
    with TestClient(create_app(settings)) as client:
        initialized = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1"},
                },
            },
        )
        headers["Mcp-Session-Id"] = initialized.headers["mcp-session-id"]
        listed = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
    return {tool["name"] for tool in listed.json()["result"]["tools"]}


def test_restore_tools_require_both_restore_and_full_sync_gates(
    restore_settings: tuple[Settings, int, int],
) -> None:
    settings, _, _ = restore_settings
    restore_tools = {"anki_backup_restore_preview", "anki_backup_restore"}
    assert restore_tools <= _tool_names_for(settings)
    restore_only = settings.model_copy(update={"allow_restore": True, "allow_full_sync": False})
    assert not (restore_tools & _tool_names_for(restore_only))
    full_only = settings.model_copy(update={"allow_restore": False, "allow_full_sync": True})
    assert not (restore_tools & _tool_names_for(full_only))


def test_force_gating_for_full_upload(official_sync_server: str, tmp_path: Path) -> None:
    path, _, _ = _seed_collection(tmp_path)

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            await service.sync_login("phase1-user", "phase1-password", official_sync_server)
            with pytest.raises(ValueError, match="no post-restore upload is pending"):
                await service.full_sync(upload=True, force=True)
            with pytest.raises(ValueError, match="force is only supported for a full upload"):
                await service.full_sync(upload=False, force=True)

    asyncio.run(scenario())


def test_post_restore_window_persists_across_service_restart(
    official_sync_server: str, tmp_path: Path
) -> None:
    path, _, _ = _seed_collection(tmp_path)

    async def restore_on_first_service() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            created = await service.create_backup()
            name = _rename_backup(created)
            await service.sync_login("phase1-user", "phase1-password", official_sync_server)
            result = await service.restore_backup(name, "local_only")
            assert result["restored"] is True
            assert (await service.status())["post_restore_upload_pending"] is True

    asyncio.run(restore_on_first_service())

    async def verify_on_second_service() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            assert (await service.status())["post_restore_upload_pending"] is True
            forced = await service.full_sync(upload=True, force=True)
            assert forced["direction"] == "upload"
            assert (await service.status())["post_restore_upload_pending"] is False

    asyncio.run(verify_on_second_service())


def test_restore_clears_pending_full_sync(official_sync_server: str, tmp_path: Path) -> None:
    path, _, _ = _seed_collection(tmp_path)

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            created = await service.create_backup()
            name = _rename_backup(created)
            await service.sync_login("phase1-user", "phase1-password", official_sync_server)
            required = (await service.sync(sync_media=False))["required"]
            assert required in {"FULL_SYNC", "FULL_UPLOAD"}
            assert (await service.status())["pending_full_sync"] == required
            await service.restore_backup(name, "local_only")
            assert (await service.status())["pending_full_sync"] is None

    asyncio.run(scenario())


def test_post_restore_window_clears_on_normal_sync_and_non_forced_full_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, _ = _seed_collection(tmp_path)

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            created = await service.create_backup()
            name = _rename_backup(created)
            monkeypatch.setattr(Collection, "sync_status", lambda self, auth: _StubSyncStatus(0))
            await service.executor.run(
                lambda adapter: setattr(adapter, "_sync_auth", SyncAuth(hkey="stub"))
            )
            monkeypatch.setattr(
                Collection, "sync_collection", lambda self, auth, sync_media: _StubSyncOutput(0)
            )

            assert (await service.restore_backup(name, "local_only"))["restored"] is True
            assert (await service.status())["post_restore_upload_pending"] is True
            await service.sync(sync_media=False)
            assert (await service.status())["post_restore_upload_pending"] is False
            assert _post_restore_status(path) is False

            # FIX-3: a non-forced full sync must clear the window too.
            assert (await service.restore_backup(name, "local_only"))["restored"] is True
            assert (await service.status())["post_restore_upload_pending"] is True
            monkeypatch.setattr(
                Collection, "sync_collection", lambda self, auth, sync_media: _StubSyncOutput(3)
            )
            await service.sync(sync_media=False)
            assert (await service.status())["pending_full_sync"] == "FULL_DOWNLOAD"
            assert (await service.status())["post_restore_upload_pending"] is True
            monkeypatch.setattr(Collection, "full_upload_or_download", lambda self, **kwargs: None)
            result = await service.full_sync(upload=False, force=False)
            assert result["direction"] == "download"
            assert (await service.status())["post_restore_upload_pending"] is False
            assert _post_restore_status(path) is False

    asyncio.run(scenario())


def test_upload_now_reconciles_pending_receipts(
    official_sync_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, note_type_id, deck_id = _seed_collection(tmp_path)
    calls = {"count": 0}

    def flaky_sync(self: CollectionAdapter, sync_media: bool) -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RuntimeError("post-sync failed")
        return {"required": "NO_CHANGES"}

    monkeypatch.setattr(CollectionAdapter, "_sync_or_raise_full_sync", flaky_sync)

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100, sync_on_write=True) as service:
            created = await service.create_backup()
            name = _rename_backup(created)
            receipt = await service.coordinated_mutation(
                "anki_notes_create",
                "pending-key",
                {"front": "B"},
                lambda adapter: adapter.create_note(
                    deck_id, note_type_id, {"Front": "B", "Back": "b"}, []
                ),
                sync_media=True,
            )
            assert receipt["remote_synced"] is False
            assert receipt["retryable"] is True
            await service.sync_login("phase1-user", "phase1-password", official_sync_server)
            result = await service.restore_backup(name, "upload_now")
            assert result["remote_replaced"] is True
            operation = await service.get_operation("pending-key")
            assert operation["receipt"]["remote_synced"] is True
            assert operation["receipt"]["state"] == "committed"

    asyncio.run(scenario())


def test_backup_listing_returns_newest_first(tmp_path: Path) -> None:
    path, note_type_id, deck_id = _seed_collection(tmp_path)
    backup_folder = Path(path).parent / "backups"

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            _rename_backup(await service.create_backup(), "aaa-older.colpkg")
            await service.executor.run(
                lambda adapter: adapter.create_note(
                    deck_id, note_type_id, {"Front": "B", "Back": "b"}, []
                )
            )
            _rename_backup(await service.create_backup(), "zzz-newer.colpkg")
            os.utime(backup_folder / "aaa-older.colpkg", (1_000.0, 1_000.0))
            os.utime(backup_folder / "zzz-newer.colpkg", (2_000.0, 2_000.0))
            listing = await service.list_backups(0, 10)
            assert listing["total"] == 2
            # Filename order and mtime order disagree, so this pins the mtime sort.
            assert [item["filename"] for item in listing["items"]] == [
                "zzz-newer.colpkg",
                "aaa-older.colpkg",
            ]

    asyncio.run(scenario())


def test_failed_restore_raises_restore_failed_and_allows_same_key_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, note_type_id, deck_id = _seed_collection(tmp_path)
    backup_folder = Path(path).parent / "backups"
    live_path = Path(path)
    original_import = RustBackend.import_collection_package

    def failing_import(
        self: RustBackend,
        *,
        col_path: str,
        backup_path: str,
        media_folder: str,
        media_db: str,
    ) -> Any:
        if Path(col_path) == live_path:
            raise RuntimeError("injected import failure")
        return original_import(
            self,
            col_path=col_path,
            backup_path=backup_path,
            media_folder=media_folder,
            media_db=media_db,
        )

    monkeypatch.setattr(RustBackend, "import_collection_package", failing_import)

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            name = _rename_backup(await service.create_backup())
            await service.executor.run(
                lambda adapter: adapter.create_note(
                    deck_id, note_type_id, {"Front": "B", "Back": "b"}, []
                )
            )
            before = set(backup_folder.glob("*.colpkg"))
            with pytest.raises(RestoreFailedError) as first:
                await service.coordinated_mutation(
                    "anki_backup_restore",
                    "restore-key",
                    {"filename": name, "mode": "local_only"},
                    lambda adapter: adapter.restore_backup(name, "local_only"),
                    sync_after=False,
                )
            new_backups = set(backup_folder.glob("*.colpkg")) - before
            assert new_backups
            assert any(str(candidate) in str(first.value) for candidate in new_backups)
            # The recovery reopen succeeds here, so no restart warning is warranted.
            assert "must be restarted" not in str(first.value)
            # FIX-2: the failed receipt was deleted, so the same key re-executes
            # instead of replaying a stale ``outcome_unknown`` receipt as success.
            with pytest.raises(RestoreFailedError):
                await service.coordinated_mutation(
                    "anki_backup_restore",
                    "restore-key",
                    {"filename": name, "mode": "local_only"},
                    lambda adapter: adapter.restore_backup(name, "local_only"),
                    sync_after=False,
                )

    asyncio.run(scenario())


def test_failed_restore_with_failed_reopen_warns_to_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, _ = _seed_collection(tmp_path)
    live_path = Path(path)
    original_import = RustBackend.import_collection_package

    def failing_import(
        self: RustBackend,
        *,
        col_path: str,
        backup_path: str,
        media_folder: str,
        media_db: str,
    ) -> Any:
        if Path(col_path) == live_path:
            raise RuntimeError("injected import failure")
        return original_import(
            self,
            col_path=col_path,
            backup_path=backup_path,
            media_folder=media_folder,
            media_db=media_db,
        )

    def failing_reopen(self: Collection, after_full_sync: bool = False) -> None:
        raise RuntimeError("injected reopen failure")

    monkeypatch.setattr(RustBackend, "import_collection_package", failing_import)

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            name = _rename_backup(await service.create_backup())
            # Collection.__init__ calls reopen(), so patch only once the service is open.
            monkeypatch.setattr(Collection, "reopen", failing_reopen)
            with pytest.raises(RestoreFailedError) as failure:
                await service.coordinated_mutation(
                    "anki_backup_restore",
                    "restart-key",
                    {"filename": name, "mode": "local_only"},
                    lambda adapter: adapter.restore_backup(name, "local_only"),
                    sync_after=False,
                )
            assert "must be restarted" in str(failure.value)

    asyncio.run(scenario())


def test_upload_now_without_login_does_not_leave_a_replayable_receipt(
    tmp_path: Path,
) -> None:
    path, _, _ = _seed_collection(tmp_path)

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            name = _rename_backup(await service.create_backup())
            request = {"filename": name, "mode": "upload_now"}
            with pytest.raises(SyncLoginRequiredError):
                await service.coordinated_mutation(
                    "anki_backup_restore",
                    "no-login-key",
                    request,
                    lambda adapter: adapter.restore_backup(name, "upload_now"),
                    sync_after=False,
                )
            # The predictable failure deletes its receipt, so a same-key replay
            # executes and fails loudly instead of returning a stale success.
            with pytest.raises(LookupError):
                await service.get_operation("no-login-key")
            with pytest.raises(SyncLoginRequiredError):
                await service.coordinated_mutation(
                    "anki_backup_restore",
                    "no-login-key",
                    request,
                    lambda adapter: adapter.restore_backup(name, "upload_now"),
                    sync_after=False,
                )

    asyncio.run(scenario())
