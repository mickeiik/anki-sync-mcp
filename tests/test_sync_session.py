from __future__ import annotations

import asyncio
import json
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

from anki_mcp.collection import (
    SYNC_REQUIRED_NAMES,
    AnkiCollectionService,
    FullSyncRequiredError,
    SyncLoginRequiredError,
)
from anki_mcp.state import PersistentState

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
    # A successful login authenticates against the server, so the session is valid.
    assert status["sync_session_valid"] is True
    assert status["sync_session_checked_at"]
    assert status["last_sync_error"] is None


@pytest.mark.anyio
async def test_status_recheck_reports_a_full_sync_requirement(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    monkeypatch.setattr(
        Collection, "sync_status", lambda self, auth: SyncStatusResponse(required=2)
    )
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        status = await service.status(recheck=True)
    assert status["next_sync_required"] == "FULL_SYNC"
    assert status["pending_full_sync"] == "FULL_SYNC"
    # The local-only recheck leaves the login-established validity unchanged.
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
        status = await service.status(recheck=True)
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
        status = await service.status(recheck=True)
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
    assert retried["sync_error"] is None
    assert retried["sync_required"] is None


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
    assert "anki_status(recheck=true)" in message
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
    """A locally-diverged client is reported blocked by the local requirement recheck.

    The recheck derives the requirement from LOCAL state only; it cannot see the
    server's divergence, which is why a local schema modification is what triggers it.
    """
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
            # modified mirrors an import/restore. The recheck recomputes the
            # requirement from LOCAL state only (sync_status reports local schema
            # changes) — it makes no network call and cannot see the server's
            # divergence itself.
            await client_a.executor.run(
                lambda adapter: adapter.collection.set_schema_modified()
            )
            probed = await client_a.status(recheck=True)
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


def _write_persisted_status(collection_path: str, status: dict[str, object]) -> None:
    state_dir = Path(collection_path).parent / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "operation-status.json").write_text(json.dumps(status), encoding="utf-8")


@pytest.mark.anyio
async def test_persisted_full_sync_requirement_blocks_readiness(
    collection_path: str,
) -> None:
    # A requirement persisted without a matching pending_full_sync must still block.
    _write_persisted_status(collection_path, {"next_sync_required": "FULL_SYNC"})
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        status = await service.status()
    assert status["ready"] is False
    assert status["readiness_reason"] == "full_sync_required"


@pytest.mark.anyio
async def test_wrong_typed_persisted_session_state_is_dropped(
    collection_path: str,
) -> None:
    _write_persisted_status(
        collection_path,
        {
            "next_sync_required": 123,
            "sync_session_valid": "yes",
            "sync_session_checked_at": 12345,
            "last_sync_error": "boom",
        },
    )
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        status = await service.status()
    assert status["next_sync_required"] is None
    assert status["sync_session_valid"] is None
    assert status["sync_session_checked_at"] is None
    assert status["last_sync_error"] is None


@pytest.mark.anyio
async def test_endpoint_switch_clears_stale_sync_requirement(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=2, server_media_usn=1),
    )
    endpoint_a = "https://a.example.test/"
    endpoint_b = "https://b.example.test/"
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", endpoint_a)
        await service.sync(sync_media=False)
        assert (await service.status())["pending_full_sync"] == "FULL_SYNC"

        await service.sync_login("user", "password", endpoint_b)
        switched = await service.status()
        assert switched["pending_full_sync"] is None
        assert switched["next_sync_required"] is None
        # The re-login authenticated against the new server, so it is valid.
        assert switched["sync_session_valid"] is True

        await service.sync(sync_media=False)
        await service.sync_login("user", "password", endpoint_b)
        same_endpoint = await service.status()
    assert same_endpoint["pending_full_sync"] == "FULL_SYNC"
    assert same_endpoint["next_sync_required"] == "FULL_SYNC"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("required", "expected_tool"),
    [
        (2, "anki_sync_full_upload(confirm=true) or anki_sync_full_download(confirm=true)"),
        (3, "anki_sync_full_download(confirm=true)"),
        (4, "anki_sync_full_upload(confirm=true)"),
    ],
)
async def test_full_sync_required_message_names_the_follow_up_tool(
    collection_path: str,
    monkeypatch: pytest.MonkeyPatch,
    required: int,
    expected_tool: str,
) -> None:
    _stub_login(monkeypatch)
    calls = {"count": 0}

    def sync(self: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        calls["count"] += 1
        if calls["count"] == 1:
            return SyncOutput(required=0)
        return SyncOutput(required=required, server_media_usn=1)

    monkeypatch.setattr(Collection, "sync_collection", sync)
    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        with pytest.raises(FullSyncRequiredError) as failure:
            await service.coordinated_mutation(
                operation="anki_decks_create",
                idempotency_key="follow-up-key",
                request={"name": "Follow Up"},
                mutate=lambda adapter: adapter.create_deck("Follow Up"),
            )
    message = str(failure.value)
    assert SYNC_REQUIRED_NAMES[required] in message
    assert expected_tool in message


@pytest.mark.anyio
async def test_failed_recheck_reports_sync_session_unverified(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)

    def fail(self: Collection, auth: SyncAuth) -> SyncStatusResponse:
        raise NetworkError("server unreachable", None, None, None)

    monkeypatch.setattr(Collection, "sync_status", fail)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        status = await service.status(recheck=True)
    assert status["ready"] is False
    assert status["readiness_reason"] == "sync_session_unverified"


@pytest.mark.anyio
async def test_post_restore_upload_pending_blocks_readiness(collection_path: str) -> None:
    _write_persisted_status(collection_path, {"post_restore_upload": True})
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        status = await service.status()
    assert status["ready"] is False
    assert status["readiness_reason"] == "post_restore_upload_pending"


@pytest.mark.anyio
async def test_replay_resumes_after_post_commit_full_sync_resolves(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    calls = {"count": 0}

    def sync(self: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        calls["count"] += 1
        if calls["count"] == 2:
            return SyncOutput(required=2, server_media_usn=1)
        return SyncOutput(required=0)

    monkeypatch.setattr(Collection, "sync_collection", sync)
    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        with pytest.raises(FullSyncRequiredError):
            await service.coordinated_mutation(
                operation="anki_decks_create",
                idempotency_key="resume-key",
                request={"name": "Resume"},
                mutate=lambda adapter: adapter.create_deck("Resume"),
            )
        blocked = await service.get_operation("resume-key")
        assert blocked["receipt"]["retryable"] is True
        assert blocked["receipt"]["sync_required"] == "FULL_SYNC"

        replayed = False

        def must_not_replay(_: object) -> dict[str, object]:
            nonlocal replayed
            replayed = True
            raise AssertionError("mutation replayed")

        resumed = await service.coordinated_mutation(
            operation="anki_decks_create",
            idempotency_key="resume-key",
            request={"name": "Resume"},
            mutate=must_not_replay,
        )
    assert replayed is False
    assert resumed["remote_synced"] is True
    assert resumed["sync_required"] is None
    assert resumed["sync_error"] is None


def test_reconciliation_clears_receipt_requirement_state(collection_path: str) -> None:
    state = PersistentState(collection_path)
    try:
        requirement = {"sync_required": "FULL_SYNC", "sync_error": {"kind": "SYNC"}}
        state.put_receipt(
            "upload-key",
            "anki_notes_create",
            "upload-hash",
            {
                "state": "committed",
                "local_committed": True,
                "remote_synced": False,
                "media_synced": None,
                "retryable": True,
                "result": {"note_id": 1},
                **requirement,
            },
        )
        state.put_receipt(
            "discard-key",
            "anki_notes_create",
            "discard-hash",
            {
                "state": "outcome_unknown",
                "local_committed": False,
                "remote_synced": False,
                "retryable": True,
                "result": None,
                **requirement,
            },
        )

        state.mark_all_remote_synced()
        uploaded = state.get_receipt("upload-key")
        assert uploaded is not None
        assert uploaded[2]["sync_required"] is None
        assert uploaded[2]["sync_error"] is None

        state.mark_pending_discarded_by_full_download()
        discarded = state.get_receipt("discard-key")
        assert discarded is not None
        assert discarded[2]["sync_required"] is None
        assert discarded[2]["sync_error"] is None
    finally:
        state.close()


@pytest.mark.anyio
async def test_legacy_pending_full_sync_derives_requirement(collection_path: str) -> None:
    _write_persisted_status(collection_path, {"pending_full_sync": {"required": 2}})
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        status = await service.status()
    assert status["next_sync_required"] == "FULL_SYNC"
    assert status["pending_full_sync"] == "FULL_SYNC"


@pytest.mark.anyio
async def test_malformed_pending_full_sync_is_treated_as_absent(collection_path: str) -> None:
    _write_persisted_status(collection_path, {"pending_full_sync": {"required": 99}})
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        status = await service.status()
    assert status["next_sync_required"] is None
    assert status["pending_full_sync"] is None


@pytest.mark.anyio
async def test_endpoint_switch_persists_the_cleared_requirement(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=2, server_media_usn=1),
    )
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://a.example.test/")
        await service.sync(sync_media=False)
        assert (await service.status())["pending_full_sync"] == "FULL_SYNC"
        await service.sync_login("user", "password", "https://b.example.test/")

    async with AnkiCollectionService(collection_path, max_page_size=100) as reopened:
        status = await reopened.status()
    assert status["pending_full_sync"] is None
    assert status["next_sync_required"] is None


@pytest.mark.anyio
async def test_endpoint_identity_survives_auth_invalidation(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=2, server_media_usn=1),
    )
    endpoint_a = "https://a.example.test/"
    endpoint_b = "https://b.example.test/"
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", endpoint_a)
        await service.sync(sync_media=False)
        assert (await service.status())["pending_full_sync"] == "FULL_SYNC"

        # Invalidate auth: the last authenticated endpoint must survive it, so a
        # re-login to a DIFFERENT server clears the stale requirement...
        await service.executor.run(lambda adapter: adapter._invalidate_sync_auth())
        await service.sync_login("user", "password", endpoint_b)
        switched = await service.status()
        assert switched["pending_full_sync"] is None
        assert switched["next_sync_required"] is None

        # ...while a re-login to the SAME server preserves it.
        await service.sync(sync_media=False)
        await service.sync_login("user", "password", endpoint_b)
        await service.executor.run(lambda adapter: adapter._invalidate_sync_auth())
        await service.sync_login("user", "password", endpoint_b)
        same_endpoint = await service.status()
    assert same_endpoint["pending_full_sync"] == "FULL_SYNC"
    assert same_endpoint["next_sync_required"] == "FULL_SYNC"


def _write_persisted_sync_auth(collection_path: str, auth: dict[str, object]) -> None:
    state_dir = Path(collection_path).parent / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "sync-auth").write_text(json.dumps(auth), encoding="utf-8")


def _read_persisted_status(collection_path: str) -> dict[str, object]:
    return json.loads(
        (Path(collection_path).parent / "state" / "operation-status.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.mark.anyio
async def test_legacy_sidecar_seeds_endpoint_identity_from_configured_endpoint(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy sidecar (no persisted username) clears an armed requirement.

    The endpoint identity is still seeded from configured_endpoint, but the ACCOUNT
    is unknown. By design (safety-first) an unknown previous account with an armed
    requirement is treated as a switch, so even a same-endpoint login with the SAME
    username clears it — the requirement re-arms on the next sync/recheck.
    """
    _stub_login(monkeypatch)
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=2, server_media_usn=1),
    )
    endpoint_a = "https://a.example.test/"

    def seed_legacy_sidecar() -> None:
        # A pre-upgrade sidecar: sync-auth has configured_endpoint but no username,
        # and the status file has no last_sync_endpoint/last_sync_username.
        _write_persisted_sync_auth(
            collection_path,
            {"hkey": "legacy-key", "endpoint": "", "configured_endpoint": endpoint_a},
        )
        _write_persisted_status(
            collection_path,
            {"pending_full_sync": {"required": 2}, "next_sync_required": "FULL_SYNC"},
        )

    # Same username and a different username both clear, because the previous
    # account is unknown and cannot be proven to be the same one.
    for username in ("user", "other-user"):
        seed_legacy_sidecar()
        async with AnkiCollectionService(collection_path, max_page_size=100) as service:
            assert (await service.status())["pending_full_sync"] == "FULL_SYNC"
            await service.sync_login(username, "password", endpoint_a)
            cleared = await service.status()
        assert cleared["pending_full_sync"] is None
        assert cleared["next_sync_required"] is None


@pytest.mark.anyio
async def test_unknown_origin_login_clears_a_pending_requirement(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    # Pending requirement with no endpoint identity at all (no sync-auth file).
    _write_persisted_status(collection_path, {"pending_full_sync": {"required": 2}})
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        assert (await service.status())["pending_full_sync"] == "FULL_SYNC"
        # An unknown origin cannot be proven safe, so even an AnkiWeb login clears it.
        await service.sync_login("user", "password", None)
        status = await service.status()
    assert status["pending_full_sync"] is None
    assert status["next_sync_required"] is None


@pytest.mark.anyio
async def test_account_switch_on_same_endpoint_clears_the_requirement(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=2, server_media_usn=1),
    )
    endpoint = "https://a.example.test/"
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user-a", "password", endpoint)
        await service.sync(sync_media=False)
        assert (await service.status())["pending_full_sync"] == "FULL_SYNC"

        # A different account on the SAME endpoint must clear the requirement.
        await service.sync_login("user-b", "password", endpoint)
        switched = await service.status()
        assert switched["pending_full_sync"] is None
        assert switched["next_sync_required"] is None

        # The SAME account on the SAME endpoint preserves it.
        await service.sync(sync_media=False)
        await service.sync_login("user-b", "password", endpoint)
        same_account = await service.status()
    assert same_account["pending_full_sync"] == "FULL_SYNC"
    assert same_account["next_sync_required"] == "FULL_SYNC"


@pytest.mark.anyio
async def test_server_directed_endpoint_migration_is_recorded(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    migrated = "https://sync.ankiweb.net/"
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=0, new_endpoint=migrated),
    )
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", None)
        result = await service.sync(sync_media=False)
        assert result["endpoint_changed"] is True
    # The migrated endpoint is persisted as the authenticated identity, so a later
    # re-login to the migrated URL is not treated as a switch.
    assert _read_persisted_status(collection_path)["last_sync_endpoint"] == migrated


@pytest.mark.anyio
async def test_successful_login_establishes_session_validity(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        status = await service.status()
    assert status["sync_session_valid"] is True
    assert status["sync_session_checked_at"]


@pytest.mark.anyio
async def test_ankiweb_same_account_relogin_preserves_the_requirement(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=2, server_media_usn=1),
    )
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user-a", "password", None)
        await service.sync(sync_media=False)
        assert (await service.status())["pending_full_sync"] == "FULL_SYNC"

        # AnkiWeb (endpoint=None) same-account re-login must preserve the requirement.
        await service.sync_login("user-a", "password", None)
        assert (await service.status())["pending_full_sync"] == "FULL_SYNC"

        # A different AnkiWeb account must clear it.
        await service.sync_login("user-b", "password", None)
        switched = await service.status()
    assert switched["pending_full_sync"] is None
    assert switched["next_sync_required"] is None


@pytest.mark.anyio
@pytest.mark.parametrize("source", ["status", "sync-auth"])
async def test_empty_string_username_is_treated_as_unknown(
    collection_path: str, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    _stub_login(monkeypatch)
    endpoint_a = "https://a.example.test/"
    status: dict[str, object] = {
        "pending_full_sync": {"required": 2},
        "next_sync_required": "FULL_SYNC",
        "last_sync_endpoint": endpoint_a,
    }
    auth: dict[str, object] = {
        "hkey": "legacy-key",
        "endpoint": "",
        "configured_endpoint": endpoint_a,
    }
    if source == "status":
        status["last_sync_username"] = ""
    else:
        auth["username"] = ""
    _write_persisted_status(collection_path, status)
    _write_persisted_sync_auth(collection_path, auth)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        # An empty string is not a known identity, so an armed requirement cannot be
        # preserved across it and a same-endpoint login clears it.
        assert (await service.status())["pending_full_sync"] == "FULL_SYNC"
        await service.sync_login("user", "password", endpoint_a)
        cleared = await service.status()
    assert cleared["pending_full_sync"] is None
    assert cleared["next_sync_required"] is None


@pytest.mark.anyio
async def test_relogin_with_configured_endpoint_after_migration_preserves_the_requirement(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    configured = "https://sync.example.test/"
    # A same-origin migration is the only one validate_sync_migration_endpoint trusts.
    migrated = "https://sync.example.test/sync/"
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(
            required=2, server_media_usn=1, new_endpoint=migrated
        ),
    )
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", configured)
        await service.sync(sync_media=False)
        assert (await service.status())["pending_full_sync"] == "FULL_SYNC"

        # The server migrated the endpoint; a re-login with the still-configured
        # endpoint is the same server, so it is not a switch.
        await service.sync_login("user", "password", configured)
        preserved = await service.status()
    assert preserved["pending_full_sync"] == "FULL_SYNC"
    assert preserved["next_sync_required"] == "FULL_SYNC"


@pytest.mark.anyio
async def test_failed_login_leaves_the_session_unverified(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    def login(_: Collection, username: str, password: str, endpoint: str | None) -> SyncAuth:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("remote login rejected")
        return SyncAuth(hkey="session-key", endpoint=endpoint or "")

    monkeypatch.setattr(Collection, "sync_login", login)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        assert (await service.status())["sync_session_valid"] is True
        with pytest.raises(RuntimeError, match="rejected"):
            await service.sync_login("user", "wrong", "https://sync.example.test/")
        status = await service.status()
    assert status["sync_session_valid"] is None
    assert status["authenticated"] is False


@pytest.mark.anyio
async def test_crash_window_does_not_pair_armed_requirement_with_new_auth(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash (or failed final status write) during a switching login must stay safe.

    With the status persisted BEFORE the auth, a crash leaves either no persisted auth
    or a status whose identity matches the auth. The old order (auth first) left an
    armed requirement paired with an auth for the NEW server, so a full upload would
    run against the wrong remote collection.
    """
    _stub_login(monkeypatch)
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=2, server_media_usn=1),
    )
    # Make the crash window deterministic and network-free: a full sync that slips
    # past the guard would execute this upload against the wrong identity.
    monkeypatch.setattr(Collection, "create_backup", lambda self, **kwargs: True)

    def must_not_upload(self: Collection, **kwargs: object) -> None:
        raise AssertionError("full sync upload executed against the new identity")

    monkeypatch.setattr(Collection, "full_upload_or_download", must_not_upload)
    endpoint_a = "https://a.example.test/"
    endpoint_b = "https://b.example.test/"

    calls = {"count": 0}
    original_save_status = PersistentState.save_status

    def fail_on_final_status_write(self: PersistentState, status: dict[str, object]) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated ENOSPC on the final status write")
        original_save_status(self, status)

    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", endpoint_a)
        await service.sync(sync_media=False)
        assert (await service.status())["pending_full_sync"] == "FULL_SYNC"

        # The switching login's final status write fails after the auth write would
        # have happened in the old ordering.
        monkeypatch.setattr(PersistentState, "save_status", fail_on_final_status_write)
        with pytest.raises(OSError, match="ENOSPC"):
            await service.sync_login("user", "password", endpoint_b)

    async with AnkiCollectionService(collection_path, max_page_size=100) as reopened:
        status = await reopened.status()
        assert status["authenticated"] is False
        # No auth was persisted, so a full upload cannot run against the new identity.
        with pytest.raises(SyncLoginRequiredError):
            await reopened.full_sync(upload=True)
        # A retry to the new identity must not leave the stale requirement armed.
        await reopened.sync_login("user", "password", endpoint_b)
        retried = await reopened.status()
    assert retried["pending_full_sync"] is None
    assert retried["next_sync_required"] is None


@pytest.mark.anyio
async def test_origin_inconsistent_persisted_identity_is_treated_as_unknown(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    _write_persisted_sync_auth(
        collection_path,
        {
            "hkey": "stale-key",
            "endpoint": "",
            "configured_endpoint": "https://b.example.test/",
        },
    )
    _write_persisted_status(
        collection_path,
        {
            "pending_full_sync": {"required": 2},
            "next_sync_required": "FULL_SYNC",
            "last_sync_endpoint": "https://a.example.test/",
        },
    )
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        # The mismatched identity is inconsistent: the agent distrusts BOTH the
        # identity and the requirement, so the stale full sync cannot target the
        # wrong server even without a re-login.
        assert await service.executor.run(lambda adapter: adapter._last_sync_endpoint) is None
        assert (await service.status())["pending_full_sync"] is None
        await service.sync_login("user", "password", "https://a.example.test/")
        cleared = await service.status()
    assert cleared["pending_full_sync"] is None
    assert cleared["next_sync_required"] is None


@pytest.mark.anyio
async def test_same_origin_path_difference_is_not_treated_as_inconsistent(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    _write_persisted_sync_auth(
        collection_path,
        {
            "hkey": "key",
            "endpoint": "",
            "configured_endpoint": "https://a.example.test/sync/",
            "username": "user",
        },
    )
    _write_persisted_status(
        collection_path,
        {
            "pending_full_sync": {"required": 2},
            "next_sync_required": "FULL_SYNC",
            "last_sync_endpoint": "https://a.example.test/",
            "last_sync_username": "user",
        },
    )
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        # Only the path differs, so the identity is preserved and the requirement stays.
        await service.sync_login("user", "password", "https://a.example.test/sync/")
        preserved = await service.status()
    assert preserved["pending_full_sync"] == "FULL_SYNC"
    assert preserved["next_sync_required"] == "FULL_SYNC"


@pytest.mark.anyio
async def test_failed_non_network_full_sync_leaves_session_invalid_after_restart(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=4),
    )
    monkeypatch.setattr(Collection, "create_backup", lambda self, **kwargs: True)

    def fail(self: Collection, **kwargs: object) -> None:
        raise RuntimeError("upstream full sync failure")

    monkeypatch.setattr(Collection, "full_upload_or_download", fail)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        await service.sync(sync_media=False)
        assert (await service.status())["pending_full_sync"] == "FULL_UPLOAD"
        with pytest.raises(RuntimeError, match="upstream"):
            await service.full_sync(upload=True)

    async with AnkiCollectionService(collection_path, max_page_size=100) as reopened:
        status = await reopened.status()
    assert status["sync_session_valid"] is False
    assert status["authenticated"] is False


@pytest.mark.anyio
async def test_failed_non_network_restore_upload_leaves_session_invalid_after_restart(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)

    def fail(self: Collection, **kwargs: object) -> None:
        raise RuntimeError("upstream restore upload failure")

    monkeypatch.setattr(Collection, "full_upload_or_download", fail)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        created = await service.create_backup()
        filename = Path(str(created["path"])).name
        await service.sync_login("user", "password", "https://sync.example.test/")
        with pytest.raises(RuntimeError, match="upstream"):
            await service.restore_backup(filename, "upload_now")

    async with AnkiCollectionService(collection_path, max_page_size=100) as reopened:
        status = await reopened.status()
    assert status["sync_session_valid"] is False
    assert status["authenticated"] is False


@pytest.mark.anyio
async def test_origin_inconsistent_persisted_pair_is_distrusted_and_disarmed(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash can pair an armed requirement with an auth for a different server."""
    _stub_login(monkeypatch)
    calls: list[object] = []

    def record(self: Collection, *args: object, **kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(Collection, "full_upload_or_download", record)
    _write_persisted_sync_auth(
        collection_path,
        {
            "hkey": "key-b",
            "endpoint": "https://b.example.test/",
            "configured_endpoint": "https://b.example.test/",
        },
    )
    _write_persisted_status(
        collection_path,
        {
            "pending_full_sync": {"required": 2},
            "next_sync_required": "FULL_SYNC",
            "last_sync_endpoint": "https://a.example.test/",
        },
    )

    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        status = await service.status()
        assert status["pending_full_sync"] is None
        assert status["next_sync_required"] is None
        # The stale requirement can no longer drive a wrong-remote full upload.
        with pytest.raises(ValueError, match="full sync was not requested"):
            await service.full_sync(upload=True)

    assert calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_endpoint", "auth_endpoint"),
    [
        ("https://a.example.test/", ""),
        (None, "https://b.example.test/"),
    ],
)
async def test_mixed_identity_pair_is_distrusted(
    collection_path: str,
    monkeypatch: pytest.MonkeyPatch,
    status_endpoint: str | None,
    auth_endpoint: str,
) -> None:
    """One side custom and the other AnkiWeb/None is a crash artifact, not an identity."""
    _stub_login(monkeypatch)
    _write_persisted_sync_auth(
        collection_path,
        {"hkey": "key", "endpoint": auth_endpoint, "configured_endpoint": auth_endpoint},
    )
    _write_persisted_status(
        collection_path,
        {
            "pending_full_sync": {"required": 4},
            "next_sync_required": "FULL_UPLOAD",
            "last_sync_endpoint": status_endpoint,
        },
    )

    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        status = await service.status()

    assert status["pending_full_sync"] is None
    assert status["next_sync_required"] is None


@pytest.mark.anyio
async def test_trailing_dot_hostname_is_still_the_same_origin(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_login(monkeypatch)
    _write_persisted_sync_auth(
        collection_path,
        {
            "hkey": "key",
            "endpoint": "https://a.example.test/",
            "configured_endpoint": "https://a.example.test/",
        },
    )
    _write_persisted_status(
        collection_path,
        {
            "pending_full_sync": {"required": 2},
            "next_sync_required": "FULL_SYNC",
            "last_sync_endpoint": "https://a.example.test./",
        },
    )

    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        status = await service.status()

    assert status["pending_full_sync"] == "FULL_SYNC"
