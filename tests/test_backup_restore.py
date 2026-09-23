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
from anki.collection import Collection
from starlette.testclient import TestClient

from anki_mcp.app import create_app
from anki_mcp.collection import AnkiCollectionService, CollectionAdapter
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
            assert isinstance(result["warning"], str) and result["warning"]

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
