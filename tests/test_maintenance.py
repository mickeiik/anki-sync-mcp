from __future__ import annotations

import json
from pathlib import Path

import pytest
from anki._backend_generated import RustBackendGenerated
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


def _seed_tagged_notes(path: Path, tags_by_note: list[list[str]]) -> list[int]:
    """Create a collection with one note per tag list."""
    collection = Collection(str(path))
    note_ids: list[int] = []
    try:
        model = collection.models.current()
        deck = int(collection.decks.id("Maintenance"))
        for index, tags in enumerate(tags_by_note):
            note = collection.new_note(model)
            note["Front"] = f"tagged-front-{index}"
            note["Back"] = f"tagged-back-{index}"
            note.tags = tags
            collection.add_note(note, deck)
            note_ids.append(int(note.id))
    finally:
        collection.close()
    return note_ids


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
    backend_calls: list[str] = []

    def recording_fix_integrity(self: Collection) -> tuple[str, bool]:
        calls.append("fix_integrity")
        return ("recorded integrity report", False)

    original_backend_check = RustBackendGenerated.check_database

    def recording_backend_check(self: RustBackendGenerated) -> list[str]:
        backend_calls.append("check_database")
        return list(original_backend_check(self))

    monkeypatch.setattr(Collection, "fix_integrity", recording_fix_integrity)
    monkeypatch.setattr(RustBackendGenerated, "check_database", recording_backend_check)

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        backend_calls.clear()
        preview = await service.executor.run(lambda adapter: adapter.preview_check_database())
        # The preview must never run Anki's repairable integrity routine, directly or via
        # fix_integrity.
        assert calls == []
        assert backend_calls == []
        assert preview["card_count"] == 1
        assert preview["note_count"] == 1
        assert isinstance(preview["state_fingerprint"], str)
        # The seeded note carries no tags, so nothing is unused.
        assert preview["unused_tags"] == []
        assert preview["unused_tags_total"] == 0
        assert preview["unused_tags_truncated"] is False
        applied = await service.executor.run(lambda adapter: adapter.check_database())
        assert calls == ["fix_integrity"]

    assert applied["ok"] is False
    assert applied["report"] == "recorded integrity report"
    assert applied["report_truncated"] is False


@pytest.mark.anyio
async def test_check_database_discloses_and_removes_unused_tags(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    note_ids = _seed_tagged_notes(path, [["keep", "zombie"], ["keep"]])

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        # Drop the note carrying "zombie" so the tag stays in the registry unused.
        await service.executor.run(
            lambda adapter: adapter.collection.remove_notes([note_ids[0]])
        )
        preview = await service.executor.run(lambda adapter: adapter.preview_check_database())
        assert preview["unused_tags"] == ["zombie"]
        assert preview["unused_tags_total"] == 1
        assert preview["unused_tags_truncated"] is False

        applied = await service.executor.run(lambda adapter: adapter.check_database())
        assert applied["unused_tags_removed"] == ["zombie"]
        assert applied["unused_tags_removed_total"] == 1
        assert applied["unused_tags_removed_truncated"] is False

        registry = await service.executor.run(lambda adapter: adapter.collection.tags.all())
        assert "zombie" not in registry
        assert "keep" in registry

        second_preview = await service.executor.run(
            lambda adapter: adapter.preview_check_database()
        )
        second_applied = await service.executor.run(lambda adapter: adapter.check_database())

    assert second_preview["unused_tags"] == []
    assert second_applied["unused_tags_removed_total"] == 0


@pytest.mark.anyio
async def test_check_database_discloses_ligature_tags_anki_folds(tmp_path: Path) -> None:
    # Anki's tags table uses a `unicase` collation that does not equate every ligature with
    # its expansion: a note tagged "ffi" does not keep an unused "ﬃ" (U+FB03) row alive.
    # The disclosure must use that collation, not Python case folding.
    path = tmp_path / "collection.anki2"
    note_ids = _seed_tagged_notes(path, [["ffi"], ["\ufb03"]])

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        await service.executor.run(
            lambda adapter: adapter.collection.remove_notes([note_ids[1]])
        )
        preview = await service.executor.run(lambda adapter: adapter.preview_check_database())
        applied = await service.executor.run(lambda adapter: adapter.check_database())

    assert preview["unused_tags"] == ["\ufb03"]
    assert applied["unused_tags_removed"] == ["\ufb03"]


@pytest.mark.anyio
async def test_check_database_fingerprint_tracks_unused_tags(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    note_ids = _seed_tagged_notes(path, [["zombie"], ["keep"]])

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        first = await service.executor.run(lambda adapter: adapter.preview_check_database())
        assert first["unused_tags"] == []

        def orphan_tag(adapter: CollectionAdapter) -> None:
            note = adapter.collection.get_note(note_ids[0])
            note.tags = ["keep"]
            adapter.collection.update_note(note)

        # Dropping "zombie" from its note leaves it in the registry unused, without
        # changing the card or note counts.
        await service.executor.run(orphan_tag)
        second = await service.executor.run(lambda adapter: adapter.preview_check_database())

    assert (first["card_count"], first["note_count"]) == (
        second["card_count"],
        second["note_count"],
    )
    assert second["unused_tags"] == ["zombie"]
    assert first["state_fingerprint"] != second["state_fingerprint"]


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


def test_app_check_database_rejects_stale_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "collection.anki2"
    note_ids = _seed_tagged_notes(path, [["zombie"]])

    with TestClient(create_app(_settings(path, monkeypatch))) as client:
        headers = _initialize(client)
        preview = _payload(
            _call(client, headers, 2, "anki_maintenance_check_database_preview", {})
        )
        assert preview["impact"]["unused_tags"] == []
        token = preview["confirmation_token"]

        # Removing the tag from its only note leaves "zombie" in the registry unused. The card
        # and note counts are unchanged, so only the unused-tag set can invalidate the token.
        _call(
            client,
            headers,
            3,
            "anki_notes_remove_tags",
            {
                "note_ids": [note_ids[0]],
                "tags": ["zombie"],
                "idempotency_key": "maintenance-check-stale-tags",
            },
        )

        stale = _call(
            client,
            headers,
            4,
            "anki_maintenance_check_database",
            {"confirmation_token": token, "idempotency_key": "maintenance-check-stale"},
        )
        assert stale.get("isError") is True
        assert "DESTRUCTIVE_CONFIRMATION_REQUIRED" in stale["content"][0]["text"]

        refreshed = _payload(
            _call(client, headers, 5, "anki_maintenance_check_database_preview", {})
        )

    assert refreshed["impact"]["card_count"] == preview["impact"]["card_count"]
    assert refreshed["impact"]["note_count"] == preview["impact"]["note_count"]
    assert refreshed["impact"]["unused_tags"] == ["zombie"]


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
        assert preview["impact"]["unused_tags"] == []
        assert preview["impact"]["unused_tags_total"] == 0
        assert preview["impact"]["unused_tags_truncated"] is False
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
    assert applied["result"]["report"]
    assert applied["result"]["unused_tags_removed"] == []
    assert applied["result"]["unused_tags_removed_total"] == 0
    assert applied["result"]["unused_tags_removed_truncated"] is False
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
    assert applied["result"]["size_before_bytes"] > 0
    assert Path(applied["result"]["backup"]["path"]).is_file()
    assert token not in json.dumps(applied)
