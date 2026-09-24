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
from anki_mcp.guard import ConfirmationRegistry


@pytest.fixture
def phase3_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    collection_path = tmp_path / "collection.anki2"
    collection = Collection(str(collection_path))
    try:
        deck_id = collection.decks.id("Guarded")
        note = collection.new_note(collection.models.current())
        note["Front"] = "guarded note"
        note["Back"] = "answer"
        note.tags = ["guarded-tag"]
        collection.add_note(note, deck_id)
    finally:
        collection.close()
    monkeypatch.setenv("MCP_AUTH_TOKEN", "phase3-token")
    monkeypatch.setenv("ANKI_COLLECTION_PATH", str(collection_path))
    monkeypatch.setenv("ANKI_SYNC_ON_WRITE", "false")
    monkeypatch.setenv("MCP_SCOPES", "read,write,admin,destructive")
    monkeypatch.setenv("ANKI_ALLOW_DESTRUCTIVE", "true")
    monkeypatch.setenv("ANKI_ALLOW_SCHEMA_CHANGES", "true")
    monkeypatch.setenv("ANKI_ALLOW_FULL_SYNC", "true")
    yield Settings(_env_file=None)


def _session(client: TestClient) -> tuple[dict[str, str], Any]:
    headers = {
        "Authorization": "Bearer phase3-token",
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
    request_id = 1

    def call(name: str, arguments: dict[str, object]) -> tuple[bool, dict[str, Any]]:
        nonlocal request_id
        request_id += 1
        result = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        ).json()["result"]
        text = result["content"][0]["text"]
        payload = json.loads(text[text.index("{") :])
        return bool(result.get("isError")), payload

    return headers, call


def test_confirmation_tokens_are_short_lived_single_use_and_request_bound() -> None:
    now = [100.0]
    registry = ConfirmationRegistry(ttl_seconds=5, clock=lambda: now[0])
    token = registry.issue("anki_notes_delete", {"note_ids": [1]})

    with pytest.raises(ValueError, match="does not match"):
        registry.consume(token, "anki_notes_delete", {"note_ids": [2]})
    registry.consume(token, "anki_notes_delete", {"note_ids": [1]})
    with pytest.raises(ValueError, match="invalid or already used"):
        registry.consume(token, "anki_notes_delete", {"note_ids": [1]})

    expired = registry.issue("anki_notes_delete", {"note_ids": [1]})
    now[0] += 6
    with pytest.raises(ValueError, match="expired"):
        registry.consume(expired, "anki_notes_delete", {"note_ids": [1]})


def test_guarded_note_deletion_requires_matching_preview_and_creates_backup(
    phase3_settings: Settings,
) -> None:
    collection = Collection(str(phase3_settings.collection_path))
    try:
        note_id = int(collection.find_notes("guarded note")[0])
    finally:
        collection.close()

    with TestClient(create_app(phase3_settings)) as client:
        _, call = _session(client)
        rejected, error = call(
            "anki_notes_delete",
            {"note_ids": [note_id], "confirmation_token": "not-a-preview-token"},
        )
        assert rejected is True
        assert error["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"

        failed, unchanged = call("anki_notes_get", {"note_id": note_id})
        assert failed is False
        assert unchanged["id"] == note_id

        failed, preview = call("anki_notes_delete_preview", {"note_ids": [note_id]})
        assert failed is False
        assert preview["impact"]["notes"] == 1
        assert preview["impact"]["cards"] == 1
        assert len(preview["impact"]["state_fingerprint"]) == 64
        assert preview["expires_in_seconds"] == phase3_settings.confirmation_ttl_seconds

        failed, receipt = call(
            "anki_notes_delete",
            {
                "note_ids": [note_id],
                "confirmation_token": preview["confirmation_token"],
                "idempotency_key": "guarded-delete",
            },
        )
        assert failed is False
        assert receipt["result"]["deleted"] == 1
        backup_path = Path(receipt["result"]["backup"]["path"])
        assert backup_path.is_file()

    collection = Collection(str(phase3_settings.collection_path))
    try:
        assert collection.find_notes("guarded note") == []
    finally:
        collection.close()


def test_schema_tools_require_full_sync_maintenance_and_preview(
    phase3_settings: Settings,
) -> None:
    no_maintenance = phase3_settings.model_copy(update={"allow_full_sync": False})
    with TestClient(create_app(no_maintenance)) as client:
        headers, _ = _session(client)
        listed = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}},
        ).json()["result"]["tools"]
        assert "anki_note_types_create" not in {tool["name"] for tool in listed}

    with TestClient(create_app(phase3_settings)) as client:
        _, call = _session(client)
        failed, preview = call(
            "anki_note_types_change_preview",
            {
                "operation": "create",
                "name": "Guarded Type",
                "fields": ["Front", "Back"],
                "templates": [
                    {
                        "name": "Card 1",
                        "question_format": "{{Front}}",
                        "answer_format": "{{Back}}",
                    }
                ],
            },
        )
        assert failed is False
        assert preview["impact"]["full_sync_required"] is True
        assert preview["impact"]["backup_required"] is True

        arguments: dict[str, object] = {
            "name": "Guarded Type",
            "fields": ["Front", "Back"],
            "templates": [
                {
                    "name": "Card 1",
                    "question_format": "{{Front}}",
                    "answer_format": "{{Back}}",
                }
            ],
            "confirmation_token": preview["confirmation_token"],
            "idempotency_key": "guarded-schema-create",
        }
        changed_arguments = {**arguments, "name": "Different Type"}
        failed, error = call("anki_note_types_create", changed_arguments)
        assert failed is True
        assert error["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"

        failed, receipt = call("anki_note_types_create", arguments)
        assert failed is False
        assert Path(receipt["result"]["backup"]["path"]).is_file()


def test_schema_apply_accepts_explicit_current_backup(
    phase3_settings: Settings,
) -> None:
    with TestClient(create_app(phase3_settings)) as client:
        _, call = _session(client)
        failed, note_types = call("anki_note_types_list", {})
        assert failed is False
        note_type_id = note_types["items"][0]["id"]
        failed, before = call("anki_note_types_get", {"note_type_id": note_type_id})
        assert failed is False
        mappings = [
            {
                "source_ordinal": template["ordinal"],
                "name": template["name"],
                "question_format": template["question_format"] + " explicit backup",
                "answer_format": template["answer_format"],
            }
            for template in before["templates"]
        ]
        failed, preview = call(
            "anki_note_types_change_preview",
            {
                "operation": "templates_update",
                "note_type_id": note_type_id,
                "template_mappings": mappings,
            },
        )
        assert failed is False
        assert preview["impact"]["backup_required"] is True
        assert preview["impact"]["full_sync_required"] is True

        failed, backup = call("anki_backup_create", {})
        assert failed is False
        assert backup["created"] is True
        assert backup["reason"] is None
        assert Path(backup["path"]).is_file()

        failed, receipt = call(
            "anki_templates_update",
            {
                "note_type_id": note_type_id,
                "mappings": mappings,
                "confirmation_token": preview["confirmation_token"],
                "idempotency_key": "schema-explicit-backup",
            },
        )

        assert failed is False
        assert receipt["result"]["template_count"] == len(mappings)
        assert receipt["result"]["backup"]["created"] is False
        assert receipt["result"]["backup"]["path"] == backup["path"]
        assert receipt["result"]["backup"]["reason"] == "no_collection_changes_since_last_backup"


def test_schema_apply_reports_backup_gate_failure_and_logs_correlation(
    phase3_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    native_create_backup = Collection.create_backup
    monkeypatch.setattr(Collection, "create_backup", lambda *args, **kwargs: False)

    with TestClient(create_app(phase3_settings)) as client:
        _, call = _session(client)
        failed, note_types = call("anki_note_types_list", {})
        assert failed is False
        note_type_id = note_types["items"][0]["id"]
        failed, before = call("anki_note_types_get", {"note_type_id": note_type_id})
        assert failed is False
        mappings = [
            {
                "source_ordinal": template["ordinal"],
                "name": template["name"],
                "question_format": template["question_format"] + " changed",
                "answer_format": template["answer_format"],
            }
            for template in before["templates"]
        ]
        failed, preview = call(
            "anki_note_types_change_preview",
            {
                "operation": "templates_update",
                "note_type_id": note_type_id,
                "template_mappings": mappings,
            },
        )
        assert failed is False

        with caplog.at_level("ERROR", logger="anki_mcp.app"):
            failed, error = call(
                "anki_templates_update",
                {
                    "note_type_id": note_type_id,
                    "mappings": mappings,
                    "confirmation_token": preview["confirmation_token"],
                    "idempotency_key": "schema-backup-gate",
                },
            )

        assert failed is True
        assert error["code"] == "BACKUP_FAILED"
        assert error["message"] == "required collection backup could not be created"
        assert error["correlation_id"] in caplog.text
        assert "BackupFailedError" in caplog.text

        failed, operation_error = call(
            "anki_operations_get", {"idempotency_key": "schema-backup-gate"}
        )
        assert failed is True
        assert operation_error["code"] == "NOT_FOUND"
        failed, after = call("anki_note_types_get", {"note_type_id": note_type_id})
        assert failed is False
        assert after["templates"] == before["templates"]

        monkeypatch.setattr(Collection, "create_backup", native_create_backup)
        failed, receipt = call(
            "anki_templates_update",
            {
                "note_type_id": note_type_id,
                "mappings": mappings,
                "confirmation_token": preview["confirmation_token"],
                "idempotency_key": "schema-backup-gate",
            },
        )
        assert failed is False
        assert receipt["result"]["template_count"] == len(mappings)


def test_native_backup_failure_is_redacted_from_client(
    phase3_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_detail = f"failed to write {phase3_settings.collection_path}"

    def fail_backup(*args: object, **kwargs: object) -> bool:
        raise RuntimeError(sensitive_detail)

    monkeypatch.setattr(Collection, "create_backup", fail_backup)

    with TestClient(create_app(phase3_settings)) as client:
        _, call = _session(client)
        with caplog.at_level("ERROR", logger="anki_mcp.app"):
            failed, error = call("anki_backup_create", {})

    assert failed is True
    assert error["code"] == "BACKUP_FAILED"
    assert error["message"] == "required collection backup could not be created"
    assert sensitive_detail not in error["message"]
    assert error["correlation_id"] in caplog.text
    assert sensitive_detail in caplog.text


def test_confirmation_rejects_stale_deck_impact(phase3_settings: Settings) -> None:
    with TestClient(create_app(phase3_settings)) as client:
        _, call = _session(client)
        failed, deck = call("anki_decks_create", {"name": "Stale Preview"})
        assert failed is False
        deck_id = deck["result"]["id"]

        failed, preview = call("anki_decks_delete_preview", {"deck_id": deck_id})
        assert failed is False
        assert preview["impact"]["cards"] == 0

        failed, _ = call(
            "anki_cards_create",
            {"deck_id": deck_id, "front": "created later", "back": "answer"},
        )
        assert failed is False
        failed, error = call(
            "anki_decks_delete",
            {
                "deck_id": deck_id,
                "confirmation_token": preview["confirmation_token"],
            },
        )
        assert failed is True
        assert error["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"


@pytest.mark.anyio
async def test_tag_delete_preview_honors_search_scan_bound(
    phase3_settings: Settings,
) -> None:
    async with AnkiCollectionService(
        phase3_settings.collection_path,
        max_page_size=100,
        max_search_scan=0,
        sync_on_write=False,
    ) as service:
        with pytest.raises(ValueError, match="MCP_MAX_SEARCH_SCAN"):
            await service.coordinated_read(
                lambda adapter: adapter.preview_tag_delete("guarded-tag")
            )


def test_review_answer_tool_is_omitted_unless_explicitly_enabled(
    phase3_settings: Settings,
) -> None:
    with TestClient(create_app(phase3_settings)) as client:
        headers, _ = _session(client)
        listed = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}},
        ).json()["result"]["tools"]
    assert "anki_cards_answer" not in {tool["name"] for tool in listed}

    enabled = phase3_settings.model_copy(update={"allow_review_answers": True})
    with TestClient(create_app(enabled)) as client:
        headers, _ = _session(client)
        listed = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}},
        ).json()["result"]["tools"]
    assert "anki_cards_answer" in {tool["name"] for tool in listed}


@pytest.mark.anyio
async def test_review_answer_records_real_scheduling_change(
    phase3_settings: Settings,
) -> None:
    collection = Collection(str(phase3_settings.collection_path))
    try:
        card_id = int(collection.find_cards("guarded note")[0])
    finally:
        collection.close()

    async with AnkiCollectionService(
        phase3_settings.collection_path,
        max_page_size=100,
        sync_on_write=False,
    ) as service:
        before = await service.get_card(card_id)
        receipt = await service.coordinated_mutation(
            "anki_cards_answer",
            "answer-review",
            {"card_id": card_id, "rating": 3, "answer_seconds": 2},
            lambda adapter: adapter.answer_card(card_id, 3, 2),
        )
        after = await service.get_card(card_id)

    assert receipt["result"] == {"id": card_id, "rating": 3, "answered": True}
    assert after["scheduling"]["reps"] == before["scheduling"]["reps"] + 1
