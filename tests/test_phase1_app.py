from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from anki.collection import Collection
from anki.sync import SyncAuth, SyncOutput
from pydantic import SecretStr
from starlette.testclient import TestClient

from anki_mcp.app import create_app
from anki_mcp.config import Settings


@pytest.fixture
def phase_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    collection_path = tmp_path / "collection.anki2"
    collection = Collection(str(collection_path))
    try:
        collection.decks.id("Study")
    finally:
        collection.close()
    monkeypatch.setenv("MCP_AUTH_TOKEN", "phase-token")
    monkeypatch.setenv("ANKI_COLLECTION_PATH", str(collection_path))
    monkeypatch.setenv("ANKI_SYNC_USERNAME", "")
    monkeypatch.delenv("ANKI_SYNC_PASSWORD", raising=False)
    monkeypatch.setenv("ANKI_SYNC_HOST", "https://sync.example.test/")
    monkeypatch.setenv("ANKI_SYNC_ON_WRITE", "false")
    return Settings(_env_file=None)


def test_scope_and_safety_flags_control_tool_discovery(phase_settings: Settings) -> None:
    with TestClient(create_app(phase_settings)) as client:
        headers = {
            "Authorization": "Bearer phase-token",
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
        listed = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
    tools = listed.json()["result"]["tools"]
    names = [tool["name"] for tool in tools]
    assert names == [
        "anki_status",
        "anki_operations_list",
        "anki_operations_get",
        "anki_metrics",
        "anki_sync_login",
        "anki_sync",
        "anki_backup_create",
        "anki_backups_list",
        "anki_decks_list",
        "anki_decks_get",
        "anki_deck_options_get",
        "anki_deck_presets_list",
        "anki_deck_presets_get",
        "anki_decks_create",
        "anki_decks_update",
        "anki_decks_update_config",
        "anki_deck_limits_update",
        "anki_scheduler_settings_update",
        "anki_fsrs_optimize_preview",
        "anki_fsrs_optimize_apply",
        "anki_fsrs_simulate",
        "anki_deck_presets_create",
        "anki_deck_presets_update",
        "anki_deck_presets_assign",
        "anki_notes_search",
        "anki_notes_get",
        "anki_notes_create",
        "anki_notes_create_batch",
        "anki_notes_update_fields",
        "anki_notes_add_tags",
        "anki_notes_remove_tags",
        "anki_tags_list",
        "anki_tags_rename",
        "anki_cards_search",
        "anki_cards_get",
        "anki_reviews_list",
        "anki_review_stats",
        "anki_cards_create",
        "anki_cards_update",
        "anki_cards_change_deck",
        "anki_cards_suspend",
        "anki_cards_unsuspend",
        "anki_cards_set_flag",
        "anki_cards_reposition",
        "anki_note_types_list",
        "anki_note_types_get",
        "anki_media_list",
        "anki_media_get",
        "anki_media_check",
        "anki_media_store",
        "anki_media_rename",
        "anki_export_apkg",
        "anki_export_notes_csv",
        "anki_import_files_list",
        "anki_undo_status",
    ]
    assert all(tool["inputSchema"]["additionalProperties"] is False for tool in tools)
    assert "anki_decks_delete" not in names
    assert "anki_cards_delete" not in names
    assert "anki_notes_delete" not in names
    assert "anki_tags_delete" not in names
    assert "anki_tags_merge_preview" not in names
    assert "anki_tags_merge" not in names
    assert "anki_note_types_create" not in names
    assert "anki_note_types_update" not in names
    assert "anki_note_types_delete" not in names
    assert "anki_media_delete" not in names
    assert "anki_media_empty_trash_preview" not in names
    assert "anki_media_empty_trash" not in names
    assert "anki_sync_full_download" not in names
    assert "anki_sync_full_upload" not in names

    read_only = phase_settings.model_copy(update={"scopes_csv": "read"})
    with TestClient(create_app(read_only)) as client:
        headers = {
            "Authorization": "Bearer phase-token",
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
        listed = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == [
        "anki_status",
        "anki_operations_list",
        "anki_operations_get",
        "anki_metrics",
        "anki_backups_list",
        "anki_decks_list",
        "anki_decks_get",
        "anki_deck_options_get",
        "anki_deck_presets_list",
        "anki_deck_presets_get",
        "anki_fsrs_simulate",
        "anki_notes_search",
        "anki_notes_get",
        "anki_tags_list",
        "anki_cards_search",
        "anki_cards_get",
        "anki_reviews_list",
        "anki_review_stats",
        "anki_note_types_list",
        "anki_note_types_get",
        "anki_media_list",
        "anki_media_get",
        "anki_media_check",
        "anki_export_apkg",
        "anki_export_notes_csv",
        "anki_import_files_list",
        "anki_undo_status",
    ]

    all_enabled = phase_settings.model_copy(
        update={
            "scopes_csv": "read,write,admin,destructive",
            "allow_destructive": True,
            "allow_full_sync": True,
            "allow_schema_changes": True,
        }
    )
    with TestClient(create_app(all_enabled)) as client:
        headers = {
            "Authorization": "Bearer phase-token",
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
        listed = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
    all_names = [tool["name"] for tool in listed.json()["result"]["tools"]]
    assert "anki_decks_delete" in all_names
    assert "anki_cards_delete" in all_names
    assert "anki_notes_delete" in all_names
    assert "anki_tags_delete" in all_names
    assert "anki_tags_merge_preview" in all_names
    assert "anki_tags_merge" in all_names
    assert "anki_note_types_create" in all_names
    assert "anki_note_types_update" in all_names
    assert "anki_note_types_delete" in all_names
    assert "anki_media_delete" in all_names
    assert "anki_media_empty_trash_preview" in all_names
    assert "anki_media_empty_trash" in all_names
    assert "anki_sync_full_download" in all_names
    assert "anki_sync_full_upload" in all_names


def test_note_and_card_control_tools_return_durable_receipts(
    phase_settings: Settings,
) -> None:
    collection = Collection(str(phase_settings.collection_path))
    try:
        deck_id = int(collection.decks.id_for_name("Study") or 0)
        target_id = int(collection.decks.id("Target"))
        model = collection.models.by_name("Basic")
        assert model is not None
        model_id = int(model["id"])
    finally:
        collection.close()

    with TestClient(create_app(phase_settings)) as client:
        headers = {
            "Authorization": "Bearer phase-token",
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
            return json.loads(result["content"][0]["text"])

        created = call(
            "anki_notes_create",
            {
                "deck_id": deck_id,
                "note_type_id": model_id,
                "fields": {"Front": "protocol", "Back": "created"},
                "tags": ["phase1"],
                "idempotency_key": "protocol-create-key",
            },
        )
        repeated = call(
            "anki_notes_create",
            {
                "deck_id": deck_id,
                "note_type_id": model_id,
                "fields": {"Front": "protocol", "Back": "created"},
                "tags": ["phase1"],
                "idempotency_key": "protocol-create-key",
            },
        )
        note_id = created["result"]["note_id"]
        card_id = created["result"]["card_ids"][0]
        changed = call(
            "anki_notes_update_fields",
            {
                "note_id": note_id,
                "fields": {"Back": "updated"},
                "idempotency_key": "protocol-update-key",
            },
        )
        call(
            "anki_notes_add_tags",
            {
                "note_ids": [note_id],
                "tags": ["durable"],
                "idempotency_key": "protocol-tag-key",
            },
        )
        moved = call(
            "anki_cards_change_deck",
            {
                "card_ids": [card_id],
                "deck_id": target_id,
                "idempotency_key": "protocol-move-key",
            },
        )
        suspended = call(
            "anki_cards_suspend",
            {"card_ids": [card_id], "idempotency_key": "protocol-suspend-key"},
        )
        unsuspended = call(
            "anki_cards_unsuspend",
            {"card_ids": [card_id], "idempotency_key": "protocol-unsuspend-key"},
        )
        fetched = call("anki_notes_get", {"note_id": note_id})

    assert repeated == created
    assert created["local_committed"] is True
    assert created["result"]["created"] is True
    assert changed["result"]["updated"] is True
    assert moved["result"]["deck_id"] == target_id
    assert suspended["result"]["suspended"] is True
    assert unsuspended["result"]["suspended"] is False
    assert "durable" in fetched["tags"]
    assert {field["name"]: field["value"] for field in fetched["fields"]}["Back"] == "updated"


def test_status_backup_and_readiness_are_actionable(phase_settings: Settings) -> None:
    with TestClient(create_app(phase_settings)) as client:
        readiness = client.get("/health/ready")
        assert readiness.status_code == 503
        assert readiness.json() == {
            "status": "not_ready",
            "reason": "authentication_required",
        }

        headers = {
            "Authorization": "Bearer phase-token",
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

        def call(request_id: int, name: str) -> dict[str, Any]:
            response = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": {}},
                },
            )
            result = response.json()["result"]
            assert result.get("isError") is not True
            return json.loads(result["content"][0]["text"])

        status = call(2, "anki_status")
        backup = call(3, "anki_backup_create")

    assert status["authenticated"] is False
    assert status["readiness_reason"] == "authentication_required"
    assert backup["requested"] is True


def test_download_if_empty_bootstrap_runs_during_application_startup(
    phase_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_sync_directions: list[bool] = []
    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(
            hkey="bootstrap-key", endpoint=endpoint
        ),
    )
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=3, server_media_usn=4),
    )
    monkeypatch.setattr(Collection, "create_backup", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        Collection,
        "full_upload_or_download",
        lambda self, *, auth, server_usn, upload: full_sync_directions.append(upload),
    )
    bootstrap_settings = phase_settings.model_copy(
        update={
            "bootstrap_mode": "download_if_empty",
            "sync_username": "bootstrap-user",
            "sync_password": SecretStr("bootstrap-password"),
        }
    )

    with TestClient(create_app(bootstrap_settings)) as client:
        readiness = client.get("/health/ready")

    assert readiness.status_code == 200
    assert readiness.json() == {"status": "ready"}
    assert full_sync_directions == [False]
