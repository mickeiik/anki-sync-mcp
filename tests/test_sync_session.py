from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from anki.collection import Collection
from anki.errors import NetworkError, SyncError, SyncErrorKind
from anki.sync import SyncAuth, SyncOutput
from anki.sync_pb2 import SyncStatusResponse

from anki_mcp.collection import AnkiCollectionService, FullSyncRequiredError

PASSWORD = "phase1-password"
USERNAME = "phase1-user"


@pytest.fixture
def collection_path(tmp_path: Path) -> Iterator[str]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        deck_id = collection.decks.id("Study")
        note = collection.new_note(collection.models.current())
        note["Front"] = "existing"
        note["Back"] = "answer"
        collection.add_note(note, deck_id)
    finally:
        collection.close()
    yield path


def _stub_login(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(
            hkey="session-key", endpoint=endpoint or ""
        ),
    )


@pytest.mark.anyio
async def test_status_reports_default_sync_session_state(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        status = await service.status()
    assert status["next_sync_required"] is None
    assert status["sync_session_valid"] is None
    assert status["sync_session_checked_at"] is None
    assert status["last_sync_error"] is None


@pytest.mark.anyio
async def test_status_check_server_reports_a_full_sync_requirement(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    monkeypatch.setattr(
        Collection, "sync_status", lambda self, auth: SyncStatusResponse(required=2)
    )
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        status = await service.status(check_server=True)
    assert status["next_sync_required"] == "FULL_SYNC"
    assert status["pending_full_sync"] == "FULL_SYNC"
    assert status["sync_session_valid"] is True
    assert status["sync_session_checked_at"]
    assert status["ready"] is False
    assert status["readiness_reason"] == "full_sync_required"


@pytest.mark.anyio
async def test_probe_network_failure_is_recorded_without_raising(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)

    def fail(self: Collection, auth: SyncAuth) -> SyncStatusResponse:
        raise NetworkError("server unreachable", None, None, None)

    monkeypatch.setattr(Collection, "sync_status", fail)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        status = await service.status(check_server=True)
    assert status["sync_session_valid"] is None
    assert status["authenticated"] is True
    assert status["last_sync_error"]["kind"] == "NETWORK"


@pytest.mark.anyio
async def test_probe_auth_failure_invalidates_the_session(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)

    def fail(self: Collection, auth: SyncAuth) -> SyncStatusResponse:
        raise SyncError("auth rejected", None, None, None, SyncErrorKind.AUTH)

    monkeypatch.setattr(Collection, "sync_status", fail)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        status = await service.status(check_server=True)
    assert status["sync_session_valid"] is False
    assert status["authenticated"] is False
    assert status["last_sync_error"]["kind"] == "AUTH"


@pytest.mark.anyio
async def test_invalidating_auth_preserves_a_known_full_sync_requirement(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=2, server_media_usn=5),
    )
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        await service.sync(sync_media=False)
        assert (await service.status())["pending_full_sync"] == "FULL_SYNC"
        await service.executor.run(lambda adapter: adapter._invalidate_sync_auth())
        status = await service.status()
    assert status["pending_full_sync"] == "FULL_SYNC"
    assert status["authenticated"] is False


@pytest.mark.anyio
async def test_transient_post_sync_failure_records_a_sync_error(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    calls = {"count": 0}

    def sync(self: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("post-commit sync failure")
        return SyncOutput(required=0)

    monkeypatch.setattr(Collection, "sync_collection", sync)
    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        receipt = await service.coordinated_mutation(
            operation="anki_decks_create",
            idempotency_key="transient-key",
            request={"name": "Transient"},
            mutate=lambda adapter: adapter.create_deck("Transient"),
        )
    assert receipt["local_committed"] is True
    assert receipt["remote_synced"] is False
    assert receipt["retryable"] is True
    assert receipt["sync_error"]["kind"] == "SYNC"

    replayed = False

    def must_not_replay(_: object) -> dict[str, object]:
        nonlocal replayed
        replayed = True
        raise AssertionError("mutation replayed")

    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as restarted:
        await restarted.sync_login("user", "password", "https://sync.example.test/")
        retried = await restarted.coordinated_mutation(
            operation="anki_decks_create",
            idempotency_key="transient-key",
            request={"name": "Transient"},
            mutate=must_not_replay,
        )
    assert replayed is False
    assert retried["remote_synced"] is True


@pytest.mark.anyio
async def test_post_sync_full_sync_requirement_names_the_local_commit(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    calls = {"count": 0}

    def sync(self: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        calls["count"] += 1
        if calls["count"] == 1:
            return SyncOutput(required=0)
        return SyncOutput(required=2, server_media_usn=3)

    monkeypatch.setattr(Collection, "sync_collection", sync)
    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        with pytest.raises(FullSyncRequiredError) as failure:
            await service.coordinated_mutation(
                operation="anki_decks_create",
                idempotency_key="full-required-key",
                request={"name": "Needs Full"},
                mutate=lambda adapter: adapter.create_deck("Needs Full"),
            )
        operation = await service.get_operation("full-required-key")
        status = await service.status()

    message = str(failure.value)
    assert "the local mutation was applied" in message
    assert "FULL_SYNC" in message
    assert "full-required-key" in message
    assert "anki_status(check_server=true)" in message
    assert operation["receipt"]["local_committed"] is True
    assert operation["receipt"]["retryable"] is True
    assert operation["receipt"]["sync_required"] == "FULL_SYNC"
    assert status["pending_full_sync"] == "FULL_SYNC"
    assert status["next_sync_required"] == "FULL_SYNC"


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
        "SYNC_USER1": f"{USERNAME}:{PASSWORD}",
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


def test_stale_client_sees_full_sync_requirement_from_official_server(
    official_sync_server: str, tmp_path: Path
) -> None:
    """A client whose schema diverged from the server must be reported as blocked."""
    path_a = str(tmp_path / "client-a" / "collection.anki2")
    Path(path_a).parent.mkdir()
    seed = Collection(path_a)
    try:
        deck_id = seed.decks.id("Study")
        note = seed.new_note(seed.models.current())
        note["Front"] = "a-front"
        note["Back"] = "a-back"
        seed.add_note(note, deck_id)
    finally:
        seed.close()

    async def scenario() -> None:
        async with AnkiCollectionService(
            path_a, max_page_size=100, sync_on_write=True
        ) as client_a:
            await client_a.sync_login(USERNAME, PASSWORD, official_sync_server)
            assert (await client_a.sync(sync_media=False))["required"] == "FULL_UPLOAD"
            await client_a.full_sync(upload=True)
            assert (await client_a.sync(sync_media=False))["required"] == "NO_CHANGES"

            path_b = str(tmp_path / "client-b" / "collection.anki2")
            Path(path_b).parent.mkdir()
            Collection(path_b).close()
            async with AnkiCollectionService(
                path_b, max_page_size=100, sync_on_write=True
            ) as client_b:
                await client_b.sync_login(USERNAME, PASSWORD, official_sync_server)
                assert (await client_b.sync(sync_media=False))["required"] in {
                    "FULL_SYNC",
                    "FULL_DOWNLOAD",
                }
                await client_b.full_sync(upload=False)
                # A local schema modification plus a local change makes B the newer schema.
                await client_b.executor.run(
                    lambda adapter: (
                        adapter.collection.set_schema_modified(),
                        adapter.collection.decks.id("Divergent"),
                    )
                )
                diverged = await client_b.sync(sync_media=False)
                assert diverged["required"] in {"FULL_SYNC", "FULL_UPLOAD"}
                await client_b.full_sync(upload=True)

            # The server's schema now differs from A's. Marking A's local schema
            # modified mirrors an import/restore and exposes the divergence to the
            # read-only requirement probe (sync_status reports local schema changes).
            await client_a.executor.run(
                lambda adapter: adapter.collection.set_schema_modified()
            )
            probed = await client_a.status(check_server=True)
            assert probed["next_sync_required"] == "FULL_SYNC"
            assert probed["ready"] is False
            assert probed["readiness_reason"] == "full_sync_required"

            with pytest.raises(FullSyncRequiredError, match="FULL_SYNC"):
                await client_a.coordinated_mutation(
                    operation="anki_decks_create",
                    idempotency_key="blocked-full-sync-key",
                    request={"name": "Blocked"},
                    mutate=lambda adapter: adapter.create_deck("Blocked"),
                )
            assert (await client_a.status())["pending_mutations"] == 0

    asyncio.run(scenario())
