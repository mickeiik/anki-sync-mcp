from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from anki.collection import Collection
from anki.sync import SyncAuth, SyncOutput
from starlette.testclient import TestClient

from anki_mcp.app import create_app
from anki_mcp.collection import (
    AnkiCollectionService,
    CollectionAdapter,
    FullSyncRequiredError,
)
from anki_mcp.config import Settings


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


def _arm_full_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub login and a sync that always reports a server-confirmed FULL_SYNC."""
    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(
            hkey="session-key", endpoint=endpoint or ""
        ),
    )
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=2, server_media_usn=1),
    )


@pytest.mark.anyio
async def test_bypass_ops_commit_under_full_sync_while_normal_op_refuses(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_full_sync(monkeypatch)
    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        await service.sync(sync_media=False)

        # A normal write is refused at the pre-sync gate, before the mutation runs.
        with pytest.raises(FullSyncRequiredError) as failure:
            await service.coordinated_mutation(
                operation="anki_decks_create",
                idempotency_key="normal-op-key",
                request={"name": "Refused"},
                mutate=lambda adapter: (_ for _ in ()).throw(
                    AssertionError("normal mutation ran despite FULL_SYNC")
                ),
            )
        assert "recovery:" in str(failure.value)

        # The bypass operations commit locally and report the pending full sync.
        check_receipt = await service.coordinated_mutation(
            operation="anki_maintenance_check_database",
            idempotency_key="bypass-check-key",
            request={},
            mutate=lambda adapter: adapter.check_database(),
        )
        assert check_receipt["local_committed"] is True
        assert check_receipt["remote_synced"] is False
        assert check_receipt["retryable"] is True
        # deterministic: the stubbed sync always returns required=2 (FULL_SYNC)
        assert check_receipt["sync_required"] == "FULL_SYNC"

        trash_receipt = await service.coordinated_mutation(
            operation="anki_media_empty_trash",
            idempotency_key="bypass-trash-key",
            request={},
            mutate=lambda adapter: adapter.empty_media_trash(),
        )
        assert trash_receipt["local_committed"] is True
        assert trash_receipt["remote_synced"] is False
        assert trash_receipt["retryable"] is True
        # distinct key, same deterministic required=2 server response
        assert trash_receipt["sync_required"] == "FULL_SYNC"


@pytest.mark.anyio
async def test_status_recovery_names_both_tools(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_full_sync(monkeypatch)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        recovery = (await service.status())["recovery"]

    assert "anki_sync_full_upload(confirm=true)" in recovery["upload"]
    assert "replaces the sync copy" in recovery["upload"]
    assert "anki_sync_full_download(confirm=true)" in recovery["download"]
    assert "sync copy replaces" in recovery["download"]


@pytest.mark.anyio
async def test_bypass_preview_passes_with_sync_on_read(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_full_sync(monkeypatch)
    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_read=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        await service.sync(sync_media=False)

        # The bypass operation is a read-only repair ADVISORY; it must pass the gate.
        preview = await service.coordinated_read(
            lambda adapter: adapter.preview_check_database(),
            sync_before=True,
            operation="anki_maintenance_check_database",
        )
        assert "state_fingerprint" in preview

        # A normal read with sync_before still refuses on the armed full sync.
        with pytest.raises(FullSyncRequiredError) as failure:
            await service.coordinated_read(
                lambda adapter: (_ for _ in ()).throw(
                    AssertionError("normal read ran despite FULL_SYNC")
                ),
                sync_before=True,
                operation="anki_decks_list",
            )
        assert "recovery:" in str(failure.value)

        # Omitting the operation (default None) also refuses: the bypass allowance
        # only applies when the caller threads the operation through explicitly.
        with pytest.raises(FullSyncRequiredError):
            await service.coordinated_read(
                lambda adapter: (_ for _ in ()).throw(
                    AssertionError("unlabeled read ran despite FULL_SYNC")
                ),
                sync_before=True,
            )


@pytest.mark.anyio
async def test_replaying_bypass_key_returns_receipt_then_resumes_sync(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_full_sync(monkeypatch)
    calls: list[int] = []

    def committed_mutation(adapter: CollectionAdapter) -> dict[str, Any]:
        calls.append(1)
        return adapter.check_database()

    def fail_if_rerun(adapter: CollectionAdapter) -> dict[str, Any]:
        raise AssertionError("the replayed bypass mutation re-ran")

    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        await service.sync(sync_media=False)

        first = await service.coordinated_mutation(
            operation="anki_maintenance_check_database",
            idempotency_key="replay-bypass-key",
            request={},
            mutate=committed_mutation,
        )
        assert first["local_committed"] is True
        assert first["sync_required"] == "FULL_SYNC"

        # Replaying the same key while FULL_SYNC is still armed returns the stored
        # receipt instead of refusing, and never re-invokes the mutation.
        replayed = await service.coordinated_mutation(
            operation="anki_maintenance_check_database",
            idempotency_key="replay-bypass-key",
            request={},
            mutate=fail_if_rerun,
        )
        assert replayed["result"] == first["result"]
        assert replayed["local_committed"] is True
        assert replayed["remote_synced"] is False
        assert replayed["retryable"] is True
        assert replayed["sync_required"] == "FULL_SYNC"
        assert len(calls) == 1

        # Once the server resolves the requirement, replaying the key resumes the
        # sync and records it, still without re-running the mutation.
        monkeypatch.setattr(
            Collection,
            "sync_collection",
            lambda self, auth, sync_media: SyncOutput(required=0, server_media_usn=0),
        )
        resolved = await service.coordinated_mutation(
            operation="anki_maintenance_check_database",
            idempotency_key="replay-bypass-key",
            request={},
            mutate=fail_if_rerun,
        )
        assert resolved["remote_synced"] is True
        assert resolved["retryable"] is False
        assert resolved["sync_required"] is None
        assert len(calls) == 1


@pytest.mark.anyio
async def test_replaying_bypass_key_after_sync_failure_returns_receipt(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_full_sync(monkeypatch)
    calls: list[int] = []

    def boom(self: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        raise RuntimeError("boom")

    monkeypatch.setattr(Collection, "sync_collection", boom)

    def committed_mutation(adapter: CollectionAdapter) -> dict[str, Any]:
        calls.append(1)
        return adapter.check_database()

    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")

        # A generic sync failure after a committed bypass mutation must return a
        # retryable receipt, not raise.
        first = await service.coordinated_mutation(
            operation="anki_maintenance_check_database",
            idempotency_key="replay-sync-error-key",
            request={},
            mutate=committed_mutation,
        )
        assert first["local_committed"] is True
        assert first["remote_synced"] is False
        assert first["retryable"] is True
        assert first["sync_error"]["kind"] == "SYNC"

        # Replaying the same key swallows the same generic failure the same way.
        replayed = await service.coordinated_mutation(
            operation="anki_maintenance_check_database",
            idempotency_key="replay-sync-error-key",
            request={},
            mutate=lambda adapter: (_ for _ in ()).throw(
                AssertionError("the replayed bypass mutation re-ran")
            ),
        )
        assert replayed["retryable"] is True
        assert replayed["sync_error"]["kind"] == "SYNC"
        assert len(calls) == 1


@pytest.mark.anyio
async def test_replaying_committed_normal_op_under_full_sync_refuses(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_full_sync(monkeypatch)
    sync_calls = 0

    def flaky_sync(self: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        nonlocal sync_calls
        sync_calls += 1
        # The pre-sync call is healthy; the post-mutation sync fails generically.
        if sync_calls == 1:
            return SyncOutput(required=0, server_media_usn=0)
        raise RuntimeError("boom")

    monkeypatch.setattr(Collection, "sync_collection", flaky_sync)

    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        first = await service.coordinated_mutation(
            operation="anki_decks_create",
            idempotency_key="replay-normal-key",
            request={"name": "Created"},
            mutate=lambda adapter: adapter.create_deck("Created"),
        )
        assert first["local_committed"] is True
        assert first["retryable"] is True
        assert first["sync_error"]["kind"] == "SYNC"

        # Once the server confirms FULL_SYNC, replaying the committed normal op must
        # refuse (the bypass return does not apply) and never re-run the mutation.
        monkeypatch.setattr(
            Collection,
            "sync_collection",
            lambda self, auth, sync_media: SyncOutput(required=2, server_media_usn=1),
        )
        with pytest.raises(FullSyncRequiredError):
            await service.coordinated_mutation(
                operation="anki_decks_create",
                idempotency_key="replay-normal-key",
                request={"name": "Created"},
                mutate=lambda adapter: (_ for _ in ()).throw(
                    AssertionError("the replayed normal mutation re-ran")
                ),
            )


def test_app_preview_passes_bypass_operation_through_sync_gate(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_full_sync(monkeypatch)
    with TestClient(create_app(_app_settings(collection_path, monkeypatch))) as client:
        headers = _mcp_headers(client)
        _mcp_call(client, headers, 2, "anki_sync_login", {})
        _mcp_call(client, headers, 3, "anki_sync", {"sync_media": False})

        # The app preview closure must thread the operation through to
        # coordinated_read (app.py:607), so a bypass preview clears the armed gate.
        bypass = _mcp_call(
            client, headers, 4, "anki_maintenance_check_database_preview", {}
        )
        assert bypass.get("isError") is not True
        payload = json.loads(bypass["content"][0]["text"])
        assert "state_fingerprint" in payload["impact"]

        # A non-bypass preview is still refused by the same gate.
        refused = _mcp_call(
            client, headers, 5, "anki_maintenance_empty_cards_preview", {}
        )
        assert refused.get("isError") is True


def _app_settings(collection_path: str, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("MCP_AUTH_TOKEN", "full-sync-token")
    monkeypatch.setenv("ANKI_COLLECTION_PATH", collection_path)
    monkeypatch.setenv("ANKI_SYNC_ON_READ", "true")
    monkeypatch.setenv("ANKI_SYNC_ON_WRITE", "true")
    monkeypatch.setenv("ANKI_SYNC_USERNAME", "user")
    monkeypatch.setenv("ANKI_SYNC_PASSWORD", "password")
    monkeypatch.setenv("MCP_SCOPES", "read,write,admin,destructive")
    monkeypatch.setenv("ANKI_ALLOW_DESTRUCTIVE", "true")
    return Settings(_env_file=None)


def _mcp_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/mcp",
        headers={
            "Authorization": "Bearer full-sync-token",
            "Accept": "application/json, text/event-stream",
        },
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
    return {
        "Authorization": "Bearer full-sync-token",
        "Accept": "application/json, text/event-stream",
        "Mcp-Session-Id": response.headers["mcp-session-id"],
    }


def _mcp_call(
    client: TestClient, headers: dict[str, str], request_id: int, name: str, arguments: dict
) -> dict:
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
    return response.json()["result"]


def test_receipt_message_is_directional() -> None:
    cases = {
        "FULL_DOWNLOAD": "anki_sync_full_download(confirm=true)",
        "FULL_UPLOAD": "anki_sync_full_upload(confirm=true)",
    }
    for requirement, tool in cases.items():
        message = CollectionAdapter._full_sync_required_receipt_message(
            "directional-key", requirement
        )
        assert tool in message

    full_sync_message = CollectionAdapter._full_sync_required_receipt_message(
        "directional-key", "FULL_SYNC"
    )
    assert "anki_sync_full_upload(confirm=true)" in full_sync_message
    assert "anki_sync_full_download(confirm=true)" in full_sync_message
