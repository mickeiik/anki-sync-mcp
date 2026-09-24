from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from anki.collection import Collection
from anki.decks import DeckManager
from anki.sync import SyncAuth, SyncOutput

from anki_mcp.collection import AnkiCollectionService, CollectionAdapter


@pytest.fixture
def populated_collection(tmp_path) -> Iterator[str]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        deck_id = collection.decks.id("Languages::Spanish")
        note = collection.new_note(collection.models.current())
        note["Front"] = "hola"
        note["Back"] = "hello"
        collection.add_note(note, deck_id)
        collection.save()
    finally:
        collection.close()
    yield path


@pytest.mark.anyio
async def test_lists_decks_with_hierarchy(populated_collection: str) -> None:
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        result = await service.list_decks(offset=0, limit=20)
    names = {deck["name"] for deck in result["items"]}
    assert {"Default", "Languages", "Languages::Spanish"}.issubset(names)
    assert result["offset"] == 0
    assert result["limit"] == 20
    assert result["total"] >= 3


@pytest.mark.anyio
async def test_gets_deck_by_stable_id(populated_collection: str) -> None:
    collection = Collection(populated_collection)
    try:
        deck_id = collection.decks.id_for_name("Languages::Spanish")
        assert deck_id is not None
    finally:
        collection.close()
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        result = await service.get_deck(deck_id)
    assert result["id"] == deck_id
    assert result["name"] == "Languages::Spanish"
    assert result["config"]["new_cards_per_day"] == 20
    assert result["config"]["reviews_per_day"] == 200


@pytest.mark.anyio
async def test_searches_and_gets_cards(populated_collection: str) -> None:
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        search = await service.search_cards(query="hola", offset=0, limit=10)
        assert search["total"] == 1
        card_id = search["items"][0]["id"]
        card = await service.get_card(card_id)
    assert card["id"] == card_id
    assert card["note_id"] > 0
    assert card["deck_name"] == "Languages::Spanish"
    assert card["question"]
    assert card["answer"]


@pytest.mark.anyio
async def test_page_size_is_bounded(populated_collection: str) -> None:
    async with AnkiCollectionService(populated_collection, max_page_size=2) as service:
        with pytest.raises(ValueError, match="limit"):
            await service.search_cards(query="", offset=0, limit=3)


@pytest.mark.anyio
async def test_search_scan_is_bounded_before_query(populated_collection: str) -> None:
    async with AnkiCollectionService(
        populated_collection, max_page_size=100, max_search_scan=0
    ) as service:
        with pytest.raises(ValueError, match="MCP_MAX_SEARCH_SCAN"):
            await service.search_cards(query="", offset=0, limit=10)


@pytest.mark.anyio
async def test_rendered_card_fields_are_bounded(populated_collection: str) -> None:
    async with AnkiCollectionService(
        populated_collection,
        max_page_size=100,
        max_rendered_field_bytes=64,
        max_card_fields=1,
    ) as service:
        search = await service.search_cards(query="hola", offset=0, limit=10)
        card = await service.get_card(search["items"][0]["id"])
        created = await service.create_card(card["deck_id"], "x" * 1000, "y" * 1000)
        created_card = await service.get_card(created["id"])
    assert len(card["question"].encode("utf-8")) <= 64
    assert len(card["answer"].encode("utf-8")) <= 64
    assert card["question_truncated"] is True
    assert card["answer_truncated"] is True
    assert all(len(field["value"].encode("utf-8")) <= 64 for field in created_card["fields"])
    assert len(created_card["fields"]) == 1
    assert created_card["fields_omitted"] == 1
    assert all(field["value_truncated"] for field in created_card["fields"])


@pytest.mark.anyio
async def test_unknown_card_and_deck_are_not_found(populated_collection: str) -> None:
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        with pytest.raises(LookupError, match="deck"):
            await service.get_deck(999_999_999)
        with pytest.raises(LookupError, match="card"):
            await service.get_card(999_999_999)


@pytest.mark.anyio
async def test_default_deck_cannot_be_deleted(populated_collection: str) -> None:
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        with pytest.raises(ValueError, match="Default"):
            await service.delete_deck(1)
        assert (await service.get_deck(1))["name"] == "Default"


@pytest.mark.anyio
async def test_deck_crud_cycle(populated_collection: str) -> None:
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        created = await service.create_deck("Projects::Anki MCP")
        existing = await service.create_deck("Projects::Anki MCP")
        updated = await service.update_deck(created["id"], "Projects::Anki Server")
        persisted = await service.get_deck(created["id"])
        deleted = await service.delete_deck(created["id"])
        with pytest.raises(LookupError, match="deck"):
            await service.get_deck(created["id"])

    assert created["created"] is True
    assert existing == {"id": created["id"], "created": False}
    assert updated == {"id": created["id"], "updated": True}
    assert persisted["name"] == "Projects::Anki Server"
    assert deleted == {"id": created["id"], "deleted": True}


@pytest.mark.anyio
async def test_delete_deck_undoes_when_removal_applies_then_raises(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = Collection(populated_collection)
    try:
        deck_id = int(collection.decks.id("Delete Rollback"))
        note = collection.new_note(collection.models.current())
        note["Front"] = "preserve"
        note["Back"] = "me"
        collection.add_note(note, deck_id)
        card_id = int(collection.find_cards(f"nid:{int(note.id)}")[0])
    finally:
        collection.close()
    original_remove = DeckManager.remove

    def apply_then_raise(self: DeckManager, deck_ids: list[int]) -> None:
        original_remove(self, deck_ids)
        raise RuntimeError("deck removal failed after apply")

    monkeypatch.setattr(DeckManager, "remove", apply_then_raise)
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        with pytest.raises(RuntimeError, match="failed after apply"):
            await service.delete_deck(deck_id)
        assert (await service.get_deck(deck_id))["name"] == "Delete Rollback"
        assert (await service.get_card(card_id))["deck_id"] == deck_id


@pytest.mark.anyio
async def test_deck_receipts_do_not_guess_anki_canonical_names(populated_collection: str) -> None:
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        normalized = await service.create_deck("Canonical::")
        existing = await service.create_deck("Collision")
        renamed = await service.update_deck(normalized["id"], "Collision")
        normalized_deck = await service.get_deck(normalized["id"])
        existing_deck = await service.get_deck(existing["id"])

    assert normalized == {"id": normalized["id"], "created": True}
    assert renamed == {"id": normalized["id"], "updated": True}
    assert normalized_deck["name"] == "Collision+"
    assert existing_deck["name"] == "Collision"


@pytest.mark.anyio
async def test_card_crud_cycle(populated_collection: str) -> None:
    collection = Collection(populated_collection)
    try:
        deck_id = collection.decks.id_for_name("Languages::Spanish")
        assert deck_id is not None
    finally:
        collection.close()

    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        created = await service.create_card(deck_id, "adiós", "goodbye")
        created_card = await service.get_card(created["id"])
        updated = await service.update_card(created["id"], "hasta luego", "see you", None)
        updated_card = await service.get_card(created["id"])
        deleted = await service.delete_card(created["id"])
        with pytest.raises(LookupError, match="card"):
            await service.get_card(created["id"])

    assert {field["name"]: field["value"] for field in created_card["fields"]} == {
        "Front": "adiós",
        "Back": "goodbye",
    }
    assert {field["name"]: field["value"] for field in updated_card["fields"]} == {
        "Front": "hasta luego",
        "Back": "see you",
    }
    assert updated == {
        "id": created["id"],
        "note_id": created["note_id"],
        "deck_id": deck_id,
        "updated": True,
    }
    assert deleted == {"id": created["id"], "deleted": True}


@pytest.mark.anyio
async def test_delete_card_undoes_when_removal_applies_then_raises(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = Collection(populated_collection)
    try:
        card_id = int(collection.find_cards("deck:Languages::Spanish")[0])
    finally:
        collection.close()
    original_remove = Collection.remove_cards_and_orphaned_notes

    def apply_then_raise(self: Collection, card_ids: list[int]) -> None:
        original_remove(self, card_ids)
        raise RuntimeError("card removal failed after apply")

    monkeypatch.setattr(Collection, "remove_cards_and_orphaned_notes", apply_then_raise)
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        with pytest.raises(RuntimeError, match="failed after apply"):
            await service.delete_card(card_id)
        assert (await service.get_card(card_id))["id"] == card_id


@pytest.mark.anyio
async def test_card_update_can_move_to_another_deck(populated_collection: str) -> None:
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        source = await service.create_deck("Source")
        target = await service.create_deck("Target")
        card = await service.create_card(source["id"], "front", "back")
        moved = await service.update_card(card["id"], None, None, target["id"])
    assert moved["deck_id"] == target["id"]


@pytest.mark.anyio
async def test_card_field_update_rejects_non_basic_note_type(populated_collection: str) -> None:
    collection = Collection(populated_collection)
    try:
        model = collection.models.by_name("Cloze")
        assert model is not None
        deck_id = collection.decks.id_for_name("Languages::Spanish")
        assert deck_id is not None
        note = collection.new_note(model)
        note["Text"] = "{{c1::hola}}"
        note["Back Extra"] = "hello"
        collection.add_note(note, deck_id)
        card_id = int(collection.find_cards(f"nid:{int(note.id)}")[0])
    finally:
        collection.close()

    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        target = await service.create_deck("Must Not Move")
        with pytest.raises(ValueError, match="Basic"):
            await service.update_card(card_id, "changed", None, target["id"])
        with pytest.raises(ValueError, match="Basic"):
            await service.update_card(card_id, None, None, target["id"])
        unchanged = await service.get_card(card_id)
    assert unchanged["deck_id"] == deck_id


@pytest.mark.anyio
async def test_create_card_does_not_render_after_committing(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = Collection(populated_collection)
    try:
        deck_id = collection.decks.id_for_name("Languages::Spanish")
        assert deck_id is not None
    finally:
        collection.close()

    def fail_rich_read(self: CollectionAdapter, card_id: int) -> dict[str, object]:
        raise RuntimeError("render failed")

    monkeypatch.setattr(CollectionAdapter, "get_card", fail_rich_read)
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        receipt = await service.create_card(deck_id, "receipt", "safe")

    assert receipt["created"] is True


@pytest.mark.anyio
async def test_create_card_cleans_up_when_postcheck_raises(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = Collection(populated_collection)
    try:
        deck_id = collection.decks.id_for_name("Languages::Spanish")
        assert deck_id is not None
        note_count = len(collection.find_notes(""))
    finally:
        collection.close()
    original_find_cards = Collection.find_cards

    def fail_postcheck(self: Collection, query: str, *args: object, **kwargs: object) -> list[int]:
        if query.startswith("nid:"):
            raise RuntimeError("postcheck failed")
        return [int(card_id) for card_id in original_find_cards(self, query, *args, **kwargs)]

    monkeypatch.setattr(Collection, "find_cards", fail_postcheck)
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        with pytest.raises(RuntimeError, match="postcheck failed"):
            await service.create_card(deck_id, "cleanup", "required")

    collection = Collection(populated_collection)
    try:
        assert len(collection.find_notes("")) == note_count
    finally:
        collection.close()


@pytest.mark.anyio
async def test_update_card_rolls_back_when_postcheck_raises(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = Collection(populated_collection)
    try:
        card_id = int(collection.find_cards("deck:Languages::Spanish")[0])
        note_id = int(collection.get_card(card_id).nid)
    finally:
        collection.close()
    original_find_cards = Collection.find_cards
    note_search_calls = 0

    def fail_second_note_search(
        self: Collection, query: str, *args: object, **kwargs: object
    ) -> list[int]:
        nonlocal note_search_calls
        if query == f"nid:{note_id}":
            note_search_calls += 1
            if note_search_calls == 2:
                raise RuntimeError("post-update check failed")
        return [int(card_id) for card_id in original_find_cards(self, query, *args, **kwargs)]

    monkeypatch.setattr(Collection, "find_cards", fail_second_note_search)
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        with pytest.raises(RuntimeError, match="post-update check failed"):
            await service.update_card(card_id, "must roll back", "must roll back", None)

    collection = Collection(populated_collection)
    try:
        note = collection.get_note(note_id)
        assert note["Front"] == "hola"
        assert note["Back"] == "hello"
    finally:
        collection.close()


@pytest.mark.anyio
async def test_update_card_restores_deck_and_fields_when_move_applies_then_raises(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = Collection(populated_collection)
    try:
        card_id = int(collection.find_cards("deck:Languages::Spanish")[0])
        card = collection.get_card(card_id)
        note_id = int(card.nid)
        original_deck_id = int(card.did)
        target_deck_id = int(collection.decks.id("Rollback Target"))
    finally:
        collection.close()
    original_set_deck = Collection.set_deck
    failed = False

    def apply_then_raise(self: Collection, card_ids: list[int], deck_id: int) -> None:
        nonlocal failed
        original_set_deck(self, card_ids, deck_id)
        if deck_id == target_deck_id and not failed:
            failed = True
            raise RuntimeError("move failed after apply")

    monkeypatch.setattr(Collection, "set_deck", apply_then_raise)
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        with pytest.raises(RuntimeError, match="move failed after apply"):
            await service.update_card(card_id, "changed", "changed", target_deck_id)

    collection = Collection(populated_collection)
    try:
        restored_card = collection.get_card(card_id)
        restored_note = collection.get_note(note_id)
        assert int(restored_card.did) == original_deck_id
        assert restored_note["Front"] == "hola"
        assert restored_note["Back"] == "hello"
    finally:
        collection.close()


@pytest.mark.anyio
async def test_card_field_update_rejects_basic_copy_with_front_and_back(
    populated_collection: str,
) -> None:
    collection = Collection(populated_collection)
    try:
        basic = collection.models.by_name("Basic")
        assert basic is not None
        copied = collection.models.copy(basic)
        deck_id = collection.decks.id_for_name("Languages::Spanish")
        assert deck_id is not None
        note = collection.new_note(copied)
        note["Front"] = "copy front"
        note["Back"] = "copy back"
        collection.add_note(note, deck_id)
        card_id = int(collection.find_cards(f"nid:{int(note.id)}")[0])
    finally:
        collection.close()

    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        with pytest.raises(ValueError, match="built-in Basic"):
            await service.update_card(card_id, "changed", None, None)


@pytest.mark.anyio
async def test_create_and_update_reject_modified_basic_with_multiple_templates(
    populated_collection: str,
) -> None:
    collection = Collection(populated_collection)
    try:
        basic = collection.models.by_name("Basic")
        assert basic is not None
        template = collection.models.new_template("Card 2")
        template["qfmt"] = "{{Back}}"
        template["afmt"] = "{{Front}}"
        collection.models.add_template(basic, template)
        collection.models.update_dict(basic)
        deck_id = collection.decks.id_for_name("Languages::Spanish")
        assert deck_id is not None
        existing_card_id = int(collection.find_cards("deck:Languages::Spanish")[0])
        note_count = len(collection.find_notes(""))
    finally:
        collection.close()

    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        with pytest.raises(ValueError, match="single-card"):
            await service.create_card(deck_id, "front", "back")
        with pytest.raises(ValueError, match="single-card"):
            await service.update_card(existing_card_id, "changed", None, None)

    collection = Collection(populated_collection)
    try:
        assert len(collection.find_notes("")) == note_count
    finally:
        collection.close()


@pytest.mark.anyio
async def test_card_deck_and_field_names_are_bounded(populated_collection: str) -> None:
    long_name = "x" * 1000
    collection = Collection(populated_collection)
    try:
        model = collection.models.new("Long metadata")
        field = collection.models.new_field(long_name)
        collection.models.add_field(model, field)
        template = collection.models.new_template("Card 1")
        template["qfmt"] = "{{" + long_name + "}}"
        template["afmt"] = "{{FrontSide}}"
        collection.models.add_template(model, template)
        collection.models.add(model)
        deck_id = collection.decks.id(long_name)
        note = collection.new_note(model)
        note[long_name] = "value"
        collection.add_note(note, deck_id)
        card_id = int(collection.find_cards(f"nid:{int(note.id)}")[0])
    finally:
        collection.close()

    async with AnkiCollectionService(
        populated_collection, max_page_size=100, max_rendered_field_bytes=64
    ) as service:
        card = await service.get_card(card_id)

    assert len(card["deck_name"].encode()) <= 64
    assert card["deck_name_truncated"] is True
    assert len(card["fields"][0]["name"].encode()) <= 64
    assert card["fields"][0]["name_truncated"] is True


@pytest.mark.anyio
async def test_sync_login_uses_configured_credentials_and_sync_reuses_host_key(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    login_calls: list[tuple[str, str, str | None]] = []
    sync_auth: list[SyncAuth] = []

    def login(_: Collection, username: str, password: str, endpoint: str | None) -> SyncAuth:
        login_calls.append((username, password, endpoint))
        return SyncAuth(hkey="host-key-secret", endpoint=endpoint or "https://ankiweb.net/")

    def sync(_: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        sync_auth.append(auth)
        return SyncOutput(required=1, server_message="complete", host_number=3)

    monkeypatch.setattr(Collection, "sync_login", login)
    monkeypatch.setattr(Collection, "sync_collection", sync)

    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        logged_in = await service.sync_login(
            "sync-user", "sync-password", "https://sync.example.test/"
        )
        synced = await service.sync(sync_media=True)

    assert login_calls == [("sync-user", "sync-password", "https://sync.example.test/")]
    assert logged_in == {"authenticated": True, "endpoint_kind": "custom"}
    assert "host-key-secret" not in repr(logged_in)
    assert sync_auth[0].hkey == "host-key-secret"
    assert synced == {
        "required": "NORMAL_SYNC",
        "server_message": "complete",
        "server_message_truncated": False,
        "host_number": 3,
        "media_sync_requested": True,
        "media_sync": {"completed": True, "checked": "", "added": "", "removed": ""},
        "endpoint_changed": False,
    }


@pytest.mark.anyio
async def test_sync_applies_valid_endpoint_migration_without_exposing_it(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoints: list[str | None] = []

    def login(_: Collection, username: str, password: str, endpoint: str | None) -> SyncAuth:
        return SyncAuth(hkey="host-key", endpoint=endpoint)

    def sync(_: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        endpoints.append(auth.endpoint)
        if len(endpoints) == 1:
            return SyncOutput(required=0, new_endpoint="https://sync17.ankiweb.net/sync/")
        return SyncOutput(required=0)

    monkeypatch.setattr(Collection, "sync_login", login)
    monkeypatch.setattr(Collection, "sync_collection", sync)
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        await service.sync_login("user", "password", None)
        migrated = await service.sync(sync_media=False)
        await service.sync(sync_media=False)

    assert migrated["endpoint_changed"] is True
    assert "sync17.ankiweb.net" not in repr(migrated)
    assert endpoints == [
        "",
        "https://sync17.ankiweb.net/sync/",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "new_endpoint",
    [
        "http://attacker.example.test/",
        "https://:443",
        "https:// /",
        "https://127.0.0.1/",
        "https://evil.example.test/",
    ],
)
async def test_insecure_endpoint_migration_is_rejected_and_invalidates_auth(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch, new_endpoint: str
) -> None:
    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(hkey="host-key", endpoint=endpoint),
    )
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=0, new_endpoint=new_endpoint),
    )
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://original.example.test/")
        with pytest.raises(ValueError):
            await service.sync(sync_media=False)
        with pytest.raises(RuntimeError, match="login"):
            await service.sync(sync_media=False)

    async with AnkiCollectionService(populated_collection, max_page_size=100) as restarted:
        status = await restarted.status()
    assert status["authenticated"] is False


@pytest.mark.anyio
async def test_sync_requires_login(populated_collection: str) -> None:
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        with pytest.raises(RuntimeError, match="login"):
            await service.sync(sync_media=False)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("required", "expected"),
    [(2, "FULL_SYNC"), (3, "FULL_DOWNLOAD"), (4, "FULL_UPLOAD")],
)
async def test_full_sync_requirements_are_reported_without_choosing_direction(
    populated_collection: str,
    monkeypatch: pytest.MonkeyPatch,
    required: int,
    expected: str,
) -> None:
    calls = 0

    def login(_: Collection, username: str, password: str, endpoint: str | None) -> SyncAuth:
        return SyncAuth(hkey="host-key", endpoint=endpoint)

    def sync(_: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        nonlocal calls
        calls += 1
        return SyncOutput(required=required)

    monkeypatch.setattr(Collection, "sync_login", login)
    monkeypatch.setattr(Collection, "sync_collection", sync)
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test")
        result = await service.sync(sync_media=False)
    assert result["required"] == expected
    assert calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("required", "upload", "direction", "sync_media", "expected_server_usn", "backup_created"),
    [
        (2, False, "download", False, None, True),
        (2, True, "upload", True, 42, True),
        (3, False, "download", False, None, False),
        (4, True, "upload", True, 42, False),
    ],
)
async def test_confirmed_full_sync_uses_required_direction_and_requests_backup(
    populated_collection: str,
    monkeypatch: pytest.MonkeyPatch,
    required: int,
    upload: bool,
    direction: str,
    sync_media: bool,
    expected_server_usn: int | None,
    backup_created: bool,
) -> None:
    full_calls: list[tuple[str, int | None, bool]] = []
    backups: list[str] = []

    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(hkey="host-key", endpoint=endpoint),
    )
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=required, server_media_usn=42),
    )

    def full_sync(
        self: Collection, *, auth: SyncAuth, server_usn: int | None, upload: bool
    ) -> None:
        full_calls.append((auth.hkey, server_usn, upload))

    def backup(
        self: Collection, *, backup_folder: str, force: bool, wait_for_completion: bool
    ) -> bool:
        assert force is True
        assert wait_for_completion is True
        backups.append(backup_folder)
        return backup_created

    monkeypatch.setattr(Collection, "full_upload_or_download", full_sync)
    monkeypatch.setattr(Collection, "create_backup", backup)
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        await service.sync(sync_media=sync_media)
        result = await service.full_sync(upload=upload)

    assert result == {
        "completed": True,
        "direction": direction,
        "backup_created": backup_created,
    }
    assert full_calls == [("host-key", expected_server_usn, upload)]
    assert backups == [str(Path(populated_collection).parent / "backups")]


@pytest.mark.anyio
async def test_full_sync_rejects_missing_requirement_and_wrong_direction(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(hkey="host-key", endpoint=endpoint),
    )
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=3, server_media_usn=42),
    )
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        with pytest.raises(ValueError, match="requested"):
            await service.full_sync(upload=False)
        await service.sync(sync_media=False)
        with pytest.raises(ValueError, match="download"):
            await service.full_sync(upload=True)


@pytest.mark.anyio
async def test_failed_relogin_clears_previous_sync_auth(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    def login(_: Collection, username: str, password: str, endpoint: str | None) -> SyncAuth:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("remote login rejected")
        return SyncAuth(hkey="first-host-key", endpoint=endpoint)

    monkeypatch.setattr(Collection, "sync_login", login)
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test")
        with pytest.raises(RuntimeError, match="rejected"):
            await service.sync_login("user", "wrong", "https://sync.example.test")
        with pytest.raises(RuntimeError, match="login"):
            await service.sync(sync_media=False)


@pytest.mark.anyio
async def test_failed_sync_preserves_auth_and_bounds_remote_message(
    populated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_attempts = 0

    def login(_: Collection, username: str, password: str, endpoint: str | None) -> SyncAuth:
        return SyncAuth(hkey="host-key", endpoint=endpoint)

    def sync(_: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        nonlocal sync_attempts
        sync_attempts += 1
        if sync_attempts == 1:
            return SyncOutput(required=0, server_message="x" * 1000)
        raise RuntimeError("remote sync failed")

    monkeypatch.setattr(Collection, "sync_login", login)
    monkeypatch.setattr(Collection, "sync_collection", sync)
    async with AnkiCollectionService(
        populated_collection, max_page_size=100, max_rendered_field_bytes=64
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test")
        result = await service.sync(sync_media=False)
        assert len(result["server_message"].encode()) <= 64
        assert result["server_message_truncated"] is True
        with pytest.raises(RuntimeError, match="failed"):
            await service.sync(sync_media=False)
        # An unexpected (non-auth, non-network) failure keeps the still-valid hkey;
        # the next sync revalidates instead of reporting a dropped session.
        assert (await service.status())["authenticated"] is True
        with pytest.raises(RuntimeError, match="failed"):
            await service.sync(sync_media=False)


@pytest.mark.anyio
async def test_concurrent_operations_use_one_collection_thread(populated_collection: str) -> None:
    async with AnkiCollectionService(populated_collection, max_page_size=100) as service:
        assert service.executor.max_workers == 1
        worker_ids = await asyncio.gather(
            *(service.executor.run(lambda _: threading.get_ident()) for _ in range(10))
        )
    assert len(set(worker_ids)) == 1
    assert worker_ids[0] != threading.get_ident()
