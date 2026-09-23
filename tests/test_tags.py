from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from anki.collection import Collection

from anki_mcp.collection import AnkiCollectionService


@pytest.fixture
def tag_collection(tmp_path: Path) -> Iterator[str]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        model = collection.models.current()
        deck_id = int(collection.decks.id("Tags"))
        for tags in (
            ["a", "a::b"],
            ["a::b"],
            ["z"],
            ["source", "target"],
            ["source::child", "target::child"],
            ["source::solo"],
        ):
            note = collection.new_note(model)
            note["Front"] = "front"
            note["Back"] = "back"
            note.tags = list(tags)
            collection.add_note(note, deck_id)
    finally:
        collection.close()
    yield path


@pytest.mark.anyio
async def test_list_tags_without_counts_keeps_shape(tag_collection: str) -> None:
    async with AnkiCollectionService(tag_collection, max_page_size=100) as service:
        page = await service.list_tags(offset=0, limit=100)
    assert page["total"] >= 1
    assert all("note_count" not in item for item in page["items"])
    assert all("name" in item and "name_truncated" in item for item in page["items"])


@pytest.mark.anyio
async def test_list_tags_with_counts_is_tree_inclusive(tag_collection: str) -> None:
    async with AnkiCollectionService(tag_collection, max_page_size=100) as service:
        page = await service.list_tags(offset=0, limit=100, include_counts=True)
    counts = {item["name"]: item["note_count"] for item in page["items"]}
    assert counts["a"] == 2
    assert counts["a::b"] == 2
    assert counts["z"] == 1


@pytest.mark.anyio
async def test_preview_tag_merge_validates_and_fingerprints(tag_collection: str) -> None:
    async with AnkiCollectionService(tag_collection, max_page_size=100) as service:
        with pytest.raises(LookupError):
            await service.preview_tag_merge("missing", "target")
        with pytest.raises(ValueError):
            await service.preview_tag_merge("  ", "target")
        with pytest.raises(ValueError):
            await service.preview_tag_merge("source", "source")
        with pytest.raises(ValueError):
            await service.preview_tag_merge("source", "source::child")
        first = await service.preview_tag_merge("source", "target")
        second = await service.preview_tag_merge("source", "target")
    assert first["notes"] == 3
    assert first["duplicate_notes"] == 2
    assert first["state_fingerprint"] == second["state_fingerprint"]


@pytest.mark.anyio
async def test_merge_tags_dedupes_and_renames_children(tag_collection: str) -> None:
    async with AnkiCollectionService(tag_collection, max_page_size=100) as service:
        result = await service.merge_tags("source", "target")
    assert result["source"] == "source"
    assert result["target"] == "target"
    assert result["deleted"] is True
    assert result["duplicate_notes_fixed"] >= 1

    collection = Collection(tag_collection)
    try:
        tags = set(collection.tags.all())
        assert "source" not in tags
        assert not any(tag.startswith("source::") for tag in tags)
        assert "target" in tags
        assert "target::child" in tags
        assert "target::solo" in tags
        for note_id in collection.find_notes(""):
            note = collection.get_note(note_id)
            assert len(note.tags) == len(set(note.tags)), note.tags
    finally:
        collection.close()
