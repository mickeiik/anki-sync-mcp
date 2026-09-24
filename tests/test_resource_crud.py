from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

import pytest
from anki.collection import Collection
from anki.media import MediaManager

from anki_mcp.collection import AnkiCollectionService


@pytest.fixture
def resource_collection(tmp_path: Path) -> Iterator[str]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        deck_id = collection.decks.id("Resources")
        note = collection.new_note(collection.models.current())
        note["Front"] = "resource note"
        note["Back"] = "answer"
        note.tags = ["old-tag"]
        collection.add_note(note, deck_id)
    finally:
        collection.close()
    yield path


@pytest.mark.anyio
async def test_note_delete_removes_note_and_generated_cards(resource_collection: str) -> None:
    collection = Collection(resource_collection)
    try:
        note_id = int(collection.find_notes("resource note")[0])
        card_ids = [int(card_id) for card_id in collection.find_cards(f"nid:{note_id}")]
    finally:
        collection.close()

    async with AnkiCollectionService(resource_collection, max_page_size=100) as service:
        deleted = await service.delete_notes([note_id])
        with pytest.raises(LookupError, match="note"):
            await service.get_note(note_id)
        for card_id in card_ids:
            with pytest.raises(LookupError, match="card"):
                await service.get_card(card_id)

    assert deleted == {"note_ids": [note_id], "deleted": 1}


@pytest.mark.anyio
async def test_tag_resources_can_be_listed_renamed_and_deleted(resource_collection: str) -> None:
    async with AnkiCollectionService(resource_collection, max_page_size=100) as service:
        listed = await service.list_tags(offset=0, limit=20)
        renamed = await service.rename_tag("old-tag", "new-tag")
        renamed_notes = await service.search_notes("tag:new-tag", offset=0, limit=20)
        deleted = await service.delete_tag("new-tag")
        final = await service.list_tags(offset=0, limit=20)

    assert listed["items"] == [{"name": "old-tag", "name_truncated": False}]
    assert renamed == {"old_name": "old-tag", "new_name": "new-tag", "updated_notes": 1}
    assert renamed_notes["total"] == 1
    assert deleted == {
        "name": "new-tag",
        "updated_notes": 1,
        "deleted": True,
        "deleted_tags": ["new-tag"],
        "deleted_tags_total": 1,
        "deleted_tags_truncated": False,
    }
    assert final["items"] == []


@pytest.mark.anyio
async def test_note_type_crud_includes_fields_templates_and_css(resource_collection: str) -> None:
    templates = [
        {
            "name": "Card 1",
            "question_format": "{{Question}}",
            "answer_format": "{{FrontSide}}<hr>{{Answer}}",
        }
    ]
    async with AnkiCollectionService(resource_collection, max_page_size=100) as service:
        created = await service.create_note_type(
            "Interview",
            ["Question", "Answer"],
            templates,
            ".card { color: black; }",
        )
        listed = await service.list_note_types(offset=0, limit=20)
        fetched = await service.get_note_type(created["id"])
        updated = await service.update_note_type(
            created["id"],
            "Interview Updated",
            ["Prompt", "Response"],
            [
                {
                    "name": "Prompt Card",
                    "question_format": "{{Prompt}}",
                    "answer_format": "{{FrontSide}}<hr>{{Response}}",
                }
            ],
            ".card { color: blue; }",
        )
        changed = await service.get_note_type(created["id"])
        deleted = await service.delete_note_type(created["id"])
        with pytest.raises(LookupError, match="note type"):
            await service.get_note_type(created["id"])

    assert any(item["id"] == created["id"] for item in listed["items"])
    assert [(field["name"], field["ordinal"]) for field in fetched["fields"]] == [
        ("Question", 0),
        ("Answer", 1),
    ]
    assert fetched["templates"][0]["question_format"] == "{{Question}}"
    assert fetched["css"] == ".card { color: black; }"
    assert updated == {"id": created["id"], "updated": True}
    assert changed["name"] == "Interview Updated"
    assert [field["name"] for field in changed["fields"]] == ["Prompt", "Response"]
    assert changed["templates"][0]["name"] == "Prompt Card"
    assert changed["css"] == ".card { color: blue; }"
    assert deleted == {"id": created["id"], "deleted": True, "deleted_notes": 0}


@pytest.mark.anyio
async def test_media_crud_round_trip_is_bounded_and_uses_safe_filenames(
    resource_collection: str,
) -> None:
    async with AnkiCollectionService(resource_collection, max_page_size=100) as service:
        stored = await service.store_media("greeting.txt", base64.b64encode(b"hello").decode())
        replaced = await service.store_media(
            "greeting.txt", base64.b64encode(b"replacement").decode()
        )
        listed = await service.list_media(offset=0, limit=20)
        fetched = await service.get_media("greeting.txt")
        renamed = await service.rename_media("greeting.txt", "renamed.txt")
        moved = await service.get_media("renamed.txt")
        deleted = await service.delete_media("renamed.txt")
        with pytest.raises(LookupError, match="media"):
            await service.get_media("renamed.txt")
        with pytest.raises(ValueError, match="filename"):
            await service.store_media("../escape.txt", base64.b64encode(b"bad").decode())

    assert stored == {"filename": "greeting.txt", "size_bytes": 5, "created": True}
    assert replaced == {"filename": "greeting.txt", "size_bytes": 11, "created": False}
    assert listed["items"] == [{"filename": "greeting.txt", "size_bytes": 11}]
    assert base64.b64decode(fetched["content_base64"]) == b"replacement"
    assert fetched["size_bytes"] == 11
    assert renamed == {"old_filename": "greeting.txt", "filename": "renamed.txt", "updated": True}
    assert base64.b64decode(moved["content_base64"]) == b"replacement"
    assert deleted == {"filename": "renamed.txt", "deleted": True}


@pytest.mark.anyio
async def test_media_store_rolls_back_apply_then_raise_failures(
    resource_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write = MediaManager.write_data

    async with AnkiCollectionService(resource_collection, max_page_size=100) as service:
        await service.store_media("existing.txt", base64.b64encode(b"original").decode())
        await service.store_media(
            "rename-source.txt", base64.b64encode(b"rename original").decode()
        )
        media_dir = Path(await service.executor.run(lambda adapter: adapter.collection.media.dir()))
        outside = media_dir.parent / "outside.txt"
        outside.write_bytes(b"outside")
        (media_dir / "linked.txt").symlink_to(outside)
        with pytest.raises(ValueError, match="symbolic link"):
            await service.store_media("linked.txt", base64.b64encode(b"overwrite").decode())

        def apply_then_raise(self: MediaManager, filename: str, data: bytes) -> str:
            stored_name = original_write(self, filename, data)
            if data.startswith(b"failing"):
                raise OSError("write failed after applying")
            return stored_name

        monkeypatch.setattr(MediaManager, "write_data", apply_then_raise)
        with pytest.raises(OSError, match="after applying"):
            await service.store_media(
                "existing.txt", base64.b64encode(b"failing replacement").decode()
            )
        restored = await service.get_media("existing.txt")
        with pytest.raises(OSError, match="after applying"):
            await service.store_media("new.txt", base64.b64encode(b"failing creation").decode())
        with pytest.raises(LookupError, match="media"):
            await service.get_media("new.txt")

        original_trash = MediaManager.trash_files

        def trash_then_raise(self: MediaManager, filenames: list[str]) -> None:
            original_trash(self, filenames)
            if filenames == ["rename-source.txt"]:
                raise OSError("trash failed after applying")

        monkeypatch.setattr(MediaManager, "trash_files", trash_then_raise)
        with pytest.raises(OSError, match="after applying"):
            await service.rename_media("rename-source.txt", "rename-target.txt")
        rename_source = await service.get_media("rename-source.txt")
        with pytest.raises(LookupError, match="media"):
            await service.get_media("rename-target.txt")

    assert base64.b64decode(restored["content_base64"]) == b"original"
    assert base64.b64decode(rename_source["content_base64"]) == b"rename original"
