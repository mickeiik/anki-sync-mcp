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
        with pytest.raises(ValueError):
            await service.preview_tag_merge("source", "SOURCE::child")
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


@pytest.mark.anyio
async def test_tag_delete_and_merge_disclose_child_tags(tag_collection: str) -> None:
    async with AnkiCollectionService(tag_collection, max_page_size=100) as service:
        delete_preview = await service.executor.run(lambda adapter: adapter.preview_tag_delete("a"))
        assert delete_preview["tags"] == ["a", "a::b"]
        assert delete_preview["tags_total"] == 2
        assert delete_preview["tags_truncated"] is False

        deleted = await service.executor.run(lambda adapter: adapter.delete_tag("a"))
        assert deleted["deleted_tags"] == ["a", "a::b"]
        assert deleted["deleted_tags_total"] == 2
        assert deleted["deleted_tags_truncated"] is False

        merge_preview = await service.preview_tag_merge("source", "target")
        assert merge_preview["source_tags"] == ["source", "source::child", "source::solo"]
        assert merge_preview["target_tags"] == ["target", "target::child"]
        assert merge_preview["source_tags_total"] == 3
        assert merge_preview["source_tags_truncated"] is False
        assert merge_preview["target_tags_total"] == 2
        assert merge_preview["target_tags_truncated"] is False

        merged = await service.merge_tags("source", "target")
        assert merged["removed_tags"] == ["source", "source::child", "source::solo"]
        assert merged["merged_into_tags"] == ["target", "target::child"]
        assert merged["removed_tags_total"] == 3
        assert merged["removed_tags_truncated"] is False
        assert merged["merged_into_tags_total"] == 2
        assert merged["merged_into_tags_truncated"] is False


@pytest.mark.anyio
async def test_merge_tags_removes_unused_registry_tag(tmp_path: Path) -> None:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        model = collection.models.current()
        deck_id = int(collection.decks.id("Tags"))
        note = collection.new_note(model)
        note["Front"] = "front"
        note["Back"] = "back"
        note.tags = ["unused", "keep"]
        collection.add_note(note, deck_id)
        stripped = collection.get_note(note.id)
        stripped.tags = ["keep"]
        collection.update_note(stripped)
        assert "unused" in collection.tags.all()  # unused tags linger in the registry
    finally:
        collection.close()

    async with AnkiCollectionService(path, max_page_size=100) as service:
        preview = await service.preview_tag_merge("unused", "target")
        assert preview["notes"] == 0
        result = await service.merge_tags("unused", "target")
    assert result["updated_notes"] == 0
    assert result["deleted"] is True

    collection = Collection(path)
    try:
        tags = set(collection.tags.all())
        assert "unused" not in tags
        assert "keep" in tags
    finally:
        collection.close()


@pytest.mark.anyio
async def test_preview_tag_delete_counts_notes_carrying_child_tags(tmp_path: Path) -> None:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        model = collection.models.current()
        deck_id = int(collection.decks.id("Tags"))
        for tags in (["parent"], ["parent::child"], ["parent::child::grand"]):
            note = collection.new_note(model)
            note["Front"] = "front"
            note["Back"] = "back"
            note.tags = list(tags)
            collection.add_note(note, deck_id)
    finally:
        collection.close()

    async with AnkiCollectionService(path, max_page_size=100) as service:
        preview = await service.executor.run(lambda adapter: adapter.preview_tag_delete("parent"))
        deleted = await service.executor.run(lambda adapter: adapter.delete_tag("parent"))

    assert preview["notes"] == 3
    assert deleted["updated_notes"] == preview["notes"]


@pytest.mark.anyio
async def test_tag_disclosure_truncates_but_fingerprint_binds_full_list(
    tag_collection: str,
) -> None:
    async with AnkiCollectionService(tag_collection, max_page_size=1) as service:
        delete_preview = await service.executor.run(lambda adapter: adapter.preview_tag_delete("a"))
        merge_preview = await service.preview_tag_merge("source", "target")

        def _add_beyond_page_tag(adapter) -> None:
            collection = adapter.collection
            for note_id in collection.find_notes(""):
                note = collection.get_note(note_id)
                if "a" in note.tags:
                    note.tags = [*note.tags, "a::c"]
                    collection.update_note(note)
                    return

        await service.executor.run(_add_beyond_page_tag)
        refreshed = await service.executor.run(lambda adapter: adapter.preview_tag_delete("a"))

    assert delete_preview["tags"] == ["a"]
    assert delete_preview["tags_total"] == 2
    assert delete_preview["tags_truncated"] is True
    assert merge_preview["source_tags"] == ["source"]
    assert merge_preview["source_tags_total"] == 3
    assert merge_preview["source_tags_truncated"] is True
    assert merge_preview["target_tags"] == ["target"]
    assert merge_preview["target_tags_total"] == 2
    assert merge_preview["target_tags_truncated"] is True

    # The extra child is beyond the disclosed page, yet it changes the full-list fingerprint.
    assert refreshed["tags"] == ["a"]
    assert refreshed["tags_total"] == 3
    assert refreshed["tags_truncated"] is True
    assert refreshed["notes"] == delete_preview["notes"]
    assert refreshed["state_fingerprint"] != delete_preview["state_fingerprint"]


@pytest.mark.anyio
async def test_tag_delete_preview_matches_apply_for_case_variant_tags(tmp_path: Path) -> None:
    """A hand-edited DB can hold a case variant; the preview must match the apply.

    Anki matches tags case-insensitively (``tags.remove`` does too), so the preview
    count must use the same comparison.
    """
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        model = collection.models.current()
        deck_id = int(collection.decks.id("Tags"))
        note_ids: list[int] = []
        for _ in range(2):
            note = collection.new_note(model)
            note["Front"] = "front"
            note["Back"] = "back"
            note.tags = ["parent"]
            collection.add_note(note, deck_id)
            note_ids.append(int(note.id))
    finally:
        collection.close()

    async with AnkiCollectionService(path, max_page_size=100) as service:

        def make_case_variant(adapter: object) -> None:
            collection = adapter.collection  # type: ignore[attr-defined]
            # Force a case variant directly in the DB (safely integer/value literal).
            collection.db.execute(
                f"update notes set tags = ' Parent ' where id = {note_ids[1]}"
            )

        await service.executor.run(make_case_variant)
        preview = await service.executor.run(
            lambda adapter: adapter.preview_tag_delete("parent")
        )
        applied = await service.executor.run(lambda adapter: adapter.delete_tag("parent"))

    assert preview["notes"] == applied["updated_notes"]
    assert preview["notes"] >= 2
