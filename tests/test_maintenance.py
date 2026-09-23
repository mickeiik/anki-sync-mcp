from __future__ import annotations

import json
from pathlib import Path

import pytest
from anki.collection import Collection
from starlette.testclient import TestClient

from anki_mcp.app import create_app
from anki_mcp.collection import AnkiCollectionService, CollectionAdapter
from anki_mcp.config import Settings


def _seed(path: Path, empty: int, valid: int = 0) -> list[int]:
    """Create a collection with `empty` empty-card notes and `valid` normal notes."""
    collection = Collection(str(path))
    empty_note_ids: list[int] = []
    try:
        model = collection.models.current()
        deck = int(collection.decks.id("Maintenance"))
        for index in range(empty):
            note = collection.new_note(model)
            note["Front"] = ""
            note["Back"] = f"empty-back-{index}"
            collection.add_note(note, deck)
            empty_note_ids.append(int(note.id))
        for index in range(valid):
            note = collection.new_note(model)
            note["Front"] = f"valid-front-{index}"
            note["Back"] = f"valid-back-{index}"
            collection.add_note(note, deck)
    finally:
        collection.close()
    return empty_note_ids


def _settings(path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("MCP_AUTH_TOKEN", "maintenance-token")
    monkeypatch.setenv("ANKI_COLLECTION_PATH", str(path))
    monkeypatch.setenv("ANKI_SYNC_ON_WRITE", "false")
    monkeypatch.setenv("MCP_SCOPES", "read,write,admin,destructive")
    monkeypatch.setenv("ANKI_ALLOW_DESTRUCTIVE", "true")
    return Settings(_env_file=None)


def _initialize(client: TestClient) -> dict[str, str]:
    headers = {
        "Authorization": "Bearer maintenance-token",
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
    return headers


def _call(
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


def _payload(result: dict) -> dict:
    assert result.get("isError") is not True, result
    return json.loads(result["content"][0]["text"])


@pytest.mark.anyio
async def test_preview_check_database_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "collection.anki2"
    _seed(path, empty=0, valid=1)

    calls: list[str] = []

    def recording_fix_integrity(self: Collection) -> tuple[str, bool]:
        calls.append("fix_integrity")
        return ("recorded integrity report", False)

    monkeypatch.setattr(Collection, "fix_integrity", recording_fix_integrity)

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        preview = await service.executor.run(lambda adapter: adapter.preview_check_database())
        # The preview must never run Anki's repairable integrity routine.
        assert calls == []
        assert preview["card_count"] == 1
        assert preview["note_count"] == 1
        assert isinstance(preview["size_bytes"], int)
        assert preview["size_bytes"] > 0
        assert isinstance(preview["state_fingerprint"], str)
        applied = await service.executor.run(lambda adapter: adapter.check_database())
        assert calls == ["fix_integrity"]

    assert applied["ok"] is False
    assert applied["report"] == "recorded integrity report"
    assert applied["report_truncated"] is False


@pytest.mark.anyio
async def test_preview_and_apply_empty_cards(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    _seed(path, empty=1, valid=1)

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        before = await service.executor.run(
            lambda adapter: (adapter.collection.card_count(), adapter.collection.note_count())
        )
        preview = await service.executor.run(lambda adapter: adapter.preview_empty_cards())
        applied = await service.executor.run(lambda adapter: adapter.empty_cards())
        after = await service.executor.run(
            lambda adapter: (adapter.collection.card_count(), adapter.collection.note_count())
        )

    assert preview["notes"] >= 1
    assert preview["cards"] >= 1
    assert preview["notes_to_delete"] >= 1
    assert applied["emptied"] is True
    assert applied["cards_removed"] == preview["cards"]
    assert applied["notes_removed"] == preview["notes_to_delete"]
    assert after[0] == before[0] - applied["cards_removed"]
    assert after[1] == before[1] - applied["notes_removed"]


@pytest.mark.anyio
async def test_optimize_database_runs_and_reports_logical_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "collection.anki2"
    _seed(path, empty=0, valid=1)

    calls: list[str] = []
    original_optimize = Collection.optimize

    def recording_optimize(self: Collection) -> None:
        calls.append("optimize")
        original_optimize(self)

    monkeypatch.setattr(Collection, "optimize", recording_optimize)

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        preview = await service.executor.run(
            lambda adapter: adapter.preview_optimize_database()
        )
        applied = await service.executor.run(lambda adapter: adapter.optimize_database())

    assert calls == ["optimize"]
    assert preview["card_count"] == 1
    assert preview["note_count"] == 1
    assert isinstance(preview["size_bytes"], int)
    assert applied["optimized"] is True
    assert isinstance(applied["size_before_bytes"], int)
    assert isinstance(applied["size_after_bytes"], int)
    assert applied["size_before_bytes"] > 0


def test_app_empty_cards_guarded_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "collection.anki2"
    _seed(path, empty=1, valid=1)

    with TestClient(create_app(_settings(path, monkeypatch))) as client:
        headers = _initialize(client)
        preview = _payload(
            _call(client, headers, 2, "anki_maintenance_empty_cards_preview", {})
        )
        assert preview["impact"]["notes"] >= 1
        token = preview["confirmation_token"]

        garbage = _call(
            client,
            headers,
            3,
            "anki_maintenance_empty_cards",
            {"confirmation_token": "not-a-token"},
        )
        assert garbage.get("isError") is True
        assert "DESTRUCTIVE_CONFIRMATION_REQUIRED" in garbage["content"][0]["text"]
        still_there = _payload(
            _call(client, headers, 4, "anki_maintenance_empty_cards_preview", {})
        )
        assert still_there["impact"]["notes"] >= 1

        applied = _payload(
            _call(
                client,
                headers,
                5,
                "anki_maintenance_empty_cards",
                {"confirmation_token": token, "idempotency_key": "maintenance-empty-1"},
            )
        )

    assert applied["state"] == "committed"
    assert applied["result"]["emptied"] is True
    assert applied["result"]["cards_removed"] == preview["impact"]["cards"]
    assert Path(applied["result"]["backup"]["path"]).is_file()
    assert token not in json.dumps(applied)


def test_app_empty_cards_rejects_stale_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "collection.anki2"
    note_ids = _seed(path, empty=2)

    with TestClient(create_app(_settings(path, monkeypatch))) as client:
        headers = _initialize(client)
        preview = _payload(
            _call(client, headers, 2, "anki_maintenance_empty_cards_preview", {})
        )
        assert preview["impact"]["notes"] == 2
        token = preview["confirmation_token"]

        # Changing the empty-card impact invalidates the preview token. An open collection
        # blocks a second writer, so the change is made through the app: delete one of the
        # seeded empty-card notes.
        delete_preview = _payload(
            _call(
                client,
                headers,
                3,
                "anki_notes_delete_preview",
                {"note_ids": [note_ids[0]]},
            )
        )
        _call(
            client,
            headers,
            4,
            "anki_notes_delete",
            {
                "note_ids": [note_ids[0]],
                "confirmation_token": delete_preview["confirmation_token"],
                "idempotency_key": "maintenance-stale-delete",
            },
        )

        stale = _call(
            client,
            headers,
            5,
            "anki_maintenance_empty_cards",
            {"confirmation_token": token, "idempotency_key": "maintenance-stale"},
        )
        assert stale.get("isError") is True
        assert "DESTRUCTIVE_CONFIRMATION_REQUIRED" in stale["content"][0]["text"]

        remaining = _payload(
            _call(client, headers, 6, "anki_maintenance_empty_cards_preview", {})
        )

    assert remaining["impact"]["notes"] == 1


@pytest.mark.anyio
async def test_empty_cards_scan_bound_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    _seed(path, empty=1)

    async with AnkiCollectionService(
        str(path), max_page_size=100, max_search_scan=0
    ) as service:
        with pytest.raises(ValueError, match="MCP_MAX_SEARCH_SCAN"):
            await service.executor.run(lambda adapter: adapter.preview_empty_cards())


@pytest.mark.anyio
async def test_empty_cards_fingerprint_tracks_ids_not_counts(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    note_ids = _seed(path, empty=1)

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        first = await service.executor.run(lambda adapter: adapter.preview_empty_cards())

        def swap(adapter: CollectionAdapter) -> None:
            collection = adapter.collection
            collection.remove_notes([note_ids[0]])
            model = collection.models.current()
            deck = int(collection.decks.id("Maintenance"))
            note = collection.new_note(model)
            note["Front"] = ""
            note["Back"] = "different-empty"
            collection.add_note(note, deck)

        await service.executor.run(swap)
        second = await service.executor.run(lambda adapter: adapter.preview_empty_cards())

    assert (first["notes"], first["cards"]) == (second["notes"], second["cards"])
    assert first["state_fingerprint"] != second["state_fingerprint"]


def test_app_check_database_guarded_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "collection.anki2"
    _seed(path, empty=0, valid=1)

    with TestClient(create_app(_settings(path, monkeypatch))) as client:
        headers = _initialize(client)
        preview = _payload(
            _call(client, headers, 2, "anki_maintenance_check_database_preview", {})
        )
        assert preview["impact"]["card_count"] == 1
        token = preview["confirmation_token"]

        garbage = _call(
            client,
            headers,
            3,
            "anki_maintenance_check_database",
            {"confirmation_token": "not-a-token"},
        )
        assert garbage.get("isError") is True
        assert "DESTRUCTIVE_CONFIRMATION_REQUIRED" in garbage["content"][0]["text"]

        applied = _payload(
            _call(
                client,
                headers,
                4,
                "anki_maintenance_check_database",
                {"confirmation_token": token, "idempotency_key": "maintenance-check-1"},
            )
        )

    assert applied["state"] == "committed"
    assert Path(applied["result"]["backup"]["path"]).is_file()
    assert token not in json.dumps(applied)


def test_app_optimize_guarded_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "collection.anki2"
    _seed(path, empty=0, valid=1)

    with TestClient(create_app(_settings(path, monkeypatch))) as client:
        headers = _initialize(client)
        preview = _payload(
            _call(client, headers, 2, "anki_maintenance_optimize_preview", {})
        )
        assert preview["impact"]["card_count"] == 1
        token = preview["confirmation_token"]

        garbage = _call(
            client,
            headers,
            3,
            "anki_maintenance_optimize",
            {"confirmation_token": "not-a-token"},
        )
        assert garbage.get("isError") is True
        assert "DESTRUCTIVE_CONFIRMATION_REQUIRED" in garbage["content"][0]["text"]

        applied = _payload(
            _call(
                client,
                headers,
                4,
                "anki_maintenance_optimize",
                {"confirmation_token": token, "idempotency_key": "maintenance-optimize-1"},
            )
        )

    assert applied["state"] == "committed"
    assert applied["result"]["optimized"] is True
    assert Path(applied["result"]["backup"]["path"]).is_file()
    assert token not in json.dumps(applied)
