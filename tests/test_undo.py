from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from anki.collection import Collection
from starlette.testclient import TestClient

from anki_mcp.app import create_app
from anki_mcp.collection import AnkiCollectionService
from anki_mcp.config import Settings


@pytest.fixture
def undo_collection(tmp_path: Path) -> Iterator[tuple[str, int]]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        note_type_id = int(collection.models.current()["id"])
    finally:
        collection.close()
    yield path, note_type_id


async def _create_note(service: AnkiCollectionService, note_type_id: int, front: str) -> int:
    receipt = await service.coordinated_mutation(
        operation="anki_notes_create",
        idempotency_key=f"undo-setup-{front}",
        request={"front": front},
        mutate=lambda adapter: adapter.create_note(
            1, note_type_id, {"Front": front, "Back": "answer"}, []
        ),
    )
    return int(receipt["result"]["note_id"])


@pytest.mark.anyio
async def test_undo_and_redo_gate_on_expected_operation(
    undo_collection: tuple[str, int],
) -> None:
    path, note_type_id = undo_collection
    async with AnkiCollectionService(path, max_page_size=100) as service:
        initial = await service.undo_status()
        assert initial["undo"] == ""
        assert initial["redo"] == ""
        assert initial["last_step"] == 0
        assert initial["durable"] is False
        assert "cleared by successful synchronization" in initial["note"]

        note_id = await _create_note(service, note_type_id, "undo me")
        status = await service.undo_status()
        operation = status["undo"]
        assert operation != ""

        with pytest.raises(ValueError, match="undo stack head"):
            await service.undo("Not This Operation")
        assert (await service.get_note(note_id))["id"] == note_id

        undone = await service.undo(operation)
        assert undone["operation_undone"] == operation
        assert undone["operation"] == operation
        with pytest.raises(LookupError):
            await service.get_note(note_id)

        redo_status = await service.undo_status()
        assert redo_status["redo"] == operation
        redone = await service.redo(operation)
        assert redone["operation_undone"] == operation
        assert (await service.get_note(note_id))["id"] == note_id


@pytest.mark.anyio
async def test_replayed_undo_returns_receipt_without_undoing_again(
    undo_collection: tuple[str, int],
) -> None:
    path, note_type_id = undo_collection
    async with AnkiCollectionService(path, max_page_size=100) as service:
        note_id = await _create_note(service, note_type_id, "replay undo")
        operation = (await service.undo_status())["undo"]

        first = await service.coordinated_mutation(
            operation="anki_undo",
            idempotency_key="undo-once",
            request={"expect_operation": operation},
            mutate=lambda adapter: adapter.undo(operation),
        )
        assert first["result"]["operation_undone"] == operation
        with pytest.raises(LookupError):
            await service.get_note(note_id)

        replayed = await service.coordinated_mutation(
            operation="anki_undo",
            idempotency_key="undo-once",
            request={"expect_operation": operation},
            mutate=lambda adapter: (_ for _ in ()).throw(
                AssertionError("undo mutation was replayed")
            ),
        )
        assert replayed == first
        with pytest.raises(LookupError):
            await service.get_note(note_id)


def _tool_names(
    path: Path, *, allow_undo: bool, scopes: str, sync_on_write: bool = False
) -> list[str]:
    settings = Settings(
        _env_file=None,
        MCP_AUTH_TOKEN="test-token",
        ANKI_COLLECTION_PATH=str(path),
        MCP_SCOPES=scopes,
        ANKI_ALLOW_UNDO="true" if allow_undo else "false",
        ANKI_SYNC_ON_WRITE="true" if sync_on_write else "false",
    )
    headers = {
        "Authorization": "Bearer test-token",
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
        return [tool["name"] for tool in listed.json()["result"]["tools"]]


def test_undo_tools_require_flag_and_destructive_scope(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    Collection(str(path)).close()

    disabled = _tool_names(path, allow_undo=False, scopes="read,write,admin,destructive")
    assert "anki_undo_status" in disabled
    assert "anki_undo" not in disabled
    assert "anki_redo" not in disabled

    enabled = _tool_names(path, allow_undo=True, scopes="read,write,admin,destructive")
    assert "anki_undo" in enabled
    assert "anki_redo" in enabled

    no_scope = _tool_names(path, allow_undo=True, scopes="read,write,admin")
    assert "anki_undo" not in no_scope


def test_undo_tools_are_not_registered_while_sync_on_write_is_enabled(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    Collection(str(path)).close()

    syncing = _tool_names(
        path, allow_undo=True, scopes="read,write,admin,destructive", sync_on_write=True
    )
    assert "anki_undo_status" in syncing
    assert "anki_undo" not in syncing
    assert "anki_redo" not in syncing


def test_undo_status_reports_disabled_reason(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    Collection(str(path)).close()

    def status(
        *, allow_undo: bool, sync_on_write: bool, scopes: str = "read,write,admin,destructive"
    ) -> dict[str, Any]:
        settings = Settings(
            _env_file=None,
            MCP_AUTH_TOKEN="test-token",
            ANKI_COLLECTION_PATH=str(path),
            MCP_SCOPES=scopes,
            ANKI_ALLOW_UNDO="true" if allow_undo else "false",
            ANKI_SYNC_ON_WRITE="true" if sync_on_write else "false",
        )
        headers = {
            "Authorization": "Bearer test-token",
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
            called = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "anki_undo_status", "arguments": {}},
                },
            )
            result = called.json()["result"]
            assert result.get("isError") is not True, result
            return json.loads(result["content"][0]["text"])

    syncing = status(allow_undo=True, sync_on_write=True)
    assert "ANKI_SYNC_ON_WRITE" in syncing["disabled_reason"]
    flagged_off = status(allow_undo=False, sync_on_write=False)
    assert "ANKI_ALLOW_UNDO" in flagged_off["disabled_reason"]
    assert status(allow_undo=True, sync_on_write=False)["disabled_reason"] is None
    no_destructive = status(
        allow_undo=True, sync_on_write=False, scopes="read,write,admin"
    )
    assert "destructive" in no_destructive["disabled_reason"]
    both_blocked = status(allow_undo=False, sync_on_write=True)
    assert "ANKI_SYNC_ON_WRITE" in both_blocked["disabled_reason"]
    assert "ANKI_ALLOW_UNDO" in both_blocked["disabled_reason"]
