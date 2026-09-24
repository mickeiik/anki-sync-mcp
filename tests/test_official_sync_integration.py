from __future__ import annotations

import asyncio
import os
import socket
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from anki.collection import Collection
from anki.errors import SyncError

from anki_mcp.collection import AnkiCollectionService, FullSyncRequiredError


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


def test_complete_phase1_lifecycle_against_official_sync_server(
    official_sync_server: str, tmp_path: Path
) -> None:
    """Prove full-sync gating, write sync, recovery, and idempotency end to end."""
    client_a_path = str(tmp_path / "client-a" / "collection.anki2")
    Path(client_a_path).parent.mkdir()
    collection = Collection(client_a_path)
    try:
        model = collection.models.by_name("Basic")
        assert model is not None
        note_type_id = int(model["id"])
    finally:
        collection.close()

    note_id = 0
    original_receipt: dict[str, object] | None = None

    async def write_from_client_a() -> None:
        nonlocal note_id, original_receipt
        async with AnkiCollectionService(
            client_a_path, max_page_size=100, sync_on_write=True
        ) as service:
            await service.sync_login("phase1-user", "phase1-password", official_sync_server)
            initial = await service.sync(sync_media=False)
            assert initial["required"] == "NO_CHANGES"
            with pytest.raises(FullSyncRequiredError, match="FULL_UPLOAD"):
                await service.coordinated_mutation(
                    operation="anki_notes_create",
                    idempotency_key="official-lifecycle-key",
                    request={"front": "official lifecycle"},
                    mutate=lambda adapter: adapter.create_note(
                        1,
                        note_type_id,
                        {"Front": "official lifecycle", "Back": "first value"},
                        ["phase1"],
                    ),
                )
            blocked = await service.status()
            assert blocked["pending_full_sync"] == "FULL_UPLOAD"
            assert blocked["readiness_reason"] == "full_sync_required"
            assert blocked["pending_mutations"] == 1

            full = await service.full_sync(upload=True)
            assert full["completed"] is True
            assert full["direction"] == "upload"
            original_receipt = await service.coordinated_mutation(
                operation="anki_notes_create",
                idempotency_key="official-lifecycle-key",
                request={"front": "official lifecycle"},
                mutate=lambda adapter: (_ for _ in ()).throw(
                    AssertionError("local mutation was replayed")
                ),
            )
            assert original_receipt["remote_synced"] is True
            note_id = int(original_receipt["result"]["note_id"])  # type: ignore[index]

    asyncio.run(write_from_client_a())

    async def verify_restart_and_second_client() -> None:
        async with AnkiCollectionService(
            client_a_path, max_page_size=100, sync_on_write=True
        ) as restarted:
            status = await restarted.status()
            assert status["authenticated"] is True
            assert status["pending_mutations"] == 0
            repeated = await restarted.coordinated_mutation(
                operation="anki_notes_create",
                idempotency_key="official-lifecycle-key",
                request={"front": "official lifecycle"},
                mutate=lambda adapter: (_ for _ in ()).throw(
                    AssertionError("local mutation was replayed after restart")
                ),
            )
            assert repeated == original_receipt

        client_b_path = str(tmp_path / "client-b" / "collection.anki2")
        Path(client_b_path).parent.mkdir()
        Collection(client_b_path).close()
        async with AnkiCollectionService(
            client_b_path, max_page_size=100, sync_on_write=True
        ) as second_client:
            await second_client.sync_login("phase1-user", "phase1-password", official_sync_server)
            required = await second_client.sync(sync_media=False)
            assert required["required"] == "FULL_DOWNLOAD"
            await second_client.full_sync(upload=False)
            downloaded = await second_client.get_note(note_id)
            assert downloaded["id"] == note_id

            updated = await second_client.coordinated_mutation(
                operation="anki_notes_update_fields",
                idempotency_key="official-update-key",
                request={"note_id": note_id, "fields": {"Back": "second value"}},
                mutate=lambda adapter: adapter.update_note_fields(
                    note_id, {"Back": "second value"}
                ),
            )
            assert updated["local_committed"] is True
            assert updated["remote_synced"] is True

        async with AnkiCollectionService(client_a_path, max_page_size=100) as first_client:
            refreshed = await first_client.coordinated_read(
                lambda adapter: adapter.get_note(note_id), sync_before=True
            )
        fields = {item["name"]: item["value"] for item in refreshed["fields"]}
        assert fields["Back"] == "second value"

    asyncio.run(verify_restart_and_second_client())


class RestartableSyncServer:
    """Official sync server that can be stopped, edited offline, and restarted."""

    def __init__(self, base: Path) -> None:
        self.port = _free_loopback_port()
        self.base = base
        self.process: subprocess.Popen[str] | None = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        environment = {
            **os.environ,
            "SYNC_BASE": str(self.base),
            "SYNC_HOST": "127.0.0.1",
            "SYNC_PORT": str(self.port),
            "SYNC_USER1": "phase1-user:phase1-password",
            "RUST_LOG": "anki=warn",
        }
        self.process = subprocess.Popen(  # noqa: S603 - fixed interpreter/module, test-only
            [sys.executable, "-m", "anki.syncserver"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                output = self.process.stdout.read() if self.process.stdout is not None else ""
                pytest.fail(f"official sync server exited during startup: {output}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            self.stop()
            pytest.fail("official sync server did not become ready")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process = None

    def collection_path(self) -> Path:
        hits = list(self.base.glob("*/collection.anki2"))
        if not hits:
            raise AssertionError(f"no server collection under {self.base}")
        return hits[0]

    def insert_stale_note(self, *, note_usn: int) -> None:
        """Clone a server note with an usn at/below the client's last-synced usn.

        The old usn keeps the server from sending the note, so the client never learns
        about it; bumping ``col.mod`` is also required, because otherwise the client's
        meta comparison reports no changes and no normal sync (and no sanity check)
        runs. Anki's end-of-sync sanity check then fails, which is the failure this
        test reproduces.
        """
        connection = sqlite3.connect(str(self.collection_path()), timeout=10)
        try:
            connection.execute("pragma busy_timeout = 10000")
            source = connection.execute(
                "select guid, mid, mod, tags, flds, sfld, csum, flags, data from notes limit 1"
            ).fetchone()
            if source is None:
                raise AssertionError("the server collection has no note to clone")
            new_id = (connection.execute("select max(id) from notes").fetchone()[0] or 0) + 1
            connection.execute(
                "insert into notes (id, guid, mid, mod, usn, tags, flds, sfld, csum, flags, "
                "data) values (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    new_id,
                    os.urandom(8).hex(),
                    source[1],
                    source[2],
                    note_usn,
                    source[3],
                    source[4],
                    source[5],
                    source[6],
                    source[7],
                    source[8],
                ),
            )
            connection.execute("update col set mod = ?", (int(time.time() * 1000) + 5000,))
            connection.commit()
        finally:
            connection.close()


@pytest.fixture
def restartable_sync_server(tmp_path: Path) -> Iterator[RestartableSyncServer]:
    server = RestartableSyncServer(tmp_path / "restartable-sync-server")
    server.start()
    try:
        yield server
    finally:
        server.stop()


def test_sanity_failure_requires_confirmed_full_sync_against_official_server(
    restartable_sync_server: RestartableSyncServer, tmp_path: Path
) -> None:
    """A server-induced sanity failure must not drop the hkey or fake a login error."""
    client_path = str(tmp_path / "client" / "collection.anki2")
    Path(client_path).parent.mkdir()
    Collection(client_path).close()

    async def scenario() -> None:
        async with AnkiCollectionService(
            client_path, max_page_size=100, sync_on_write=True
        ) as service:
            await service.sync_login(
                "phase1-user", "phase1-password", restartable_sync_server.endpoint
            )
            await service.sync(sync_media=False)
            # A fresh client must upload before its first write can sync.
            with pytest.raises(FullSyncRequiredError):
                await service.coordinated_mutation(
                    operation="anki_decks_create",
                    idempotency_key="sanity-bootstrap",
                    request={"name": "Bootstrap"},
                    mutate=lambda adapter: adapter.create_deck("Bootstrap"),
                )
            await service.full_sync(upload=True)
            note_type_id = await service.executor.run(
                lambda adapter: int(adapter.collection.models.by_name("Basic")["id"])
            )
            baseline = await service.coordinated_mutation(
                operation="anki_notes_create",
                idempotency_key="sanity-baseline",
                request={"front": "baseline"},
                mutate=lambda adapter: adapter.create_note(
                    1, note_type_id, {"Front": "baseline", "Back": "b"}, []
                ),
            )
            assert baseline["remote_synced"] is True
            state = await service.executor.run(
                lambda adapter: {
                    "usn": int(adapter.collection.db.scalar("select usn from col")),
                    "auth": adapter._state.load_sync_auth(),
                }
            )
            assert state["auth"] is not None

            # Stop the server, insert a note whose usn is at/below the client's
            # last-synced usn, and restart: the next sync's sanity check fails.
            restartable_sync_server.stop()
            restartable_sync_server.insert_stale_note(note_usn=max(state["usn"] - 1, 0))
            restartable_sync_server.start()

            # Before the fix this failure discarded the hkey, so the *next* write
            # reported AUTHENTICATION_FAILED even though the credential was valid;
            # this call itself raised the raw SyncError in both versions.
            with pytest.raises(SyncError):
                await service.coordinated_mutation(
                    operation="anki_decks_create",
                    idempotency_key="sanity-failing",
                    request={"name": "AfterCorruption"},
                    mutate=lambda adapter: adapter.create_deck("AfterCorruption"),
                )
            persisted = await service.executor.run(
                lambda adapter: adapter._state.load_sync_auth()
            )
            assert persisted is not None
            status = await service.status()
            assert status["authenticated"] is True
            assert status["next_sync_required"] == "FULL_SYNC"
            assert status["ready"] is False
            assert status["readiness_reason"] == "full_sync_required"

            # A live sync confirms the requirement before the mutation runs.
            with pytest.raises(FullSyncRequiredError):
                await service.coordinated_mutation(
                    operation="anki_decks_create",
                    idempotency_key="sanity-blocked",
                    request={"name": "Blocked"},
                    mutate=lambda adapter: adapter.create_deck("Blocked"),
                )
            await service.full_sync(upload=True)
            resolved = await service.coordinated_mutation(
                operation="anki_decks_create",
                idempotency_key="sanity-resolved",
                request={"name": "Resolved"},
                mutate=lambda adapter: adapter.create_deck("Resolved"),
            )
            assert resolved["remote_synced"] is True

    asyncio.run(scenario())
