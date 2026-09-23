from __future__ import annotations

import base64
import hashlib
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from anki.collection import Collection

from anki_mcp.collection import AnkiCollectionService, ResourceLimitError


@pytest.fixture
def export_collection(tmp_path: Path) -> Iterator[tuple[str, int, int]]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        model = collection.models.current()
        deck_a = int(collection.decks.id("DeckA"))
        deck_b = int(collection.decks.id("DeckB"))
        for front, back in (("front-a1", "back-a1"), ("front-a2", "back-a2")):
            note = collection.new_note(model)
            note["Front"] = front
            note["Back"] = back
            collection.add_note(note, deck_a)
        note = collection.new_note(model)
        note["Front"] = "front-b1"
        note["Back"] = "back-b1"
        collection.add_note(note, deck_b)
    finally:
        collection.close()
    yield path, deck_a, deck_b


@pytest.mark.anyio
async def test_export_apkg_writes_verified_zip(
    export_collection: tuple[str, int, int],
) -> None:
    path, deck_a, _ = export_collection
    async with AnkiCollectionService(path, max_page_size=100) as service:
        result = await service.export_apkg(deck_a, True, True, False)

    exported = Path(result["path"])
    assert exported.is_file()
    assert exported.parent.name == "exports"
    assert exported.name.startswith("export-") and exported.suffix == ".apkg"
    assert result["size_bytes"] > 0
    assert result["size_bytes"] == exported.stat().st_size
    assert result["sha256"] == hashlib.sha256(exported.read_bytes()).hexdigest()
    assert result["exported_notes"] == 2  # only DeckA's cards
    with zipfile.ZipFile(exported) as archive:
        assert set(archive.namelist()) == {
            "collection.anki2",
            "collection.anki21b",
            "media",
            "meta",
        }


@pytest.mark.anyio
async def test_export_notes_csv_scopes_and_inline(
    export_collection: tuple[str, int, int],
) -> None:
    path, deck_a, _ = export_collection
    async with AnkiCollectionService(path, max_page_size=100) as service:
        plain = await service.export_notes_csv(deck_a, True, True, True, False, False, False)
        inlined = await service.export_notes_csv(deck_a, True, True, True, False, False, True)
        whole = await service.export_notes_csv(None, True, True, True, False, False, False)

    plain_path = Path(plain["path"])
    text = plain_path.read_text(encoding="utf-8")
    assert "front-a1" in text and "back-a2" in text
    assert "front-b1" not in text
    assert plain["rows"] == 2
    assert whole["rows"] == 3
    assert plain["sha256"] == hashlib.sha256(plain_path.read_bytes()).hexdigest()
    assert "content_base64" not in plain
    assert "inline_omitted" not in plain

    assert base64.b64decode(inlined["content_base64"]) == Path(inlined["path"]).read_bytes()


@pytest.mark.anyio
async def test_export_notes_csv_omits_inline_when_over_response_budget(
    export_collection: tuple[str, int, int],
) -> None:
    path, deck_a, _ = export_collection
    async with AnkiCollectionService(path, max_page_size=100, max_response_bytes=16) as service:
        result = await service.export_notes_csv(deck_a, True, True, True, False, False, True)

    assert "content_base64" not in result
    assert result["inline_omitted"] is True
    assert result["inline_reason"] == "inline payload would exceed MCP_MAX_RESPONSE_BYTES"


@pytest.mark.anyio
async def test_export_is_bounded_by_search_scan(
    export_collection: tuple[str, int, int],
) -> None:
    path, deck_a, _ = export_collection
    async with AnkiCollectionService(path, max_page_size=100, max_search_scan=1) as service:
        with pytest.raises(ResourceLimitError, match="MCP_MAX_SEARCH_SCAN"):
            await service.export_apkg(deck_a, True, True, False)
        with pytest.raises(ResourceLimitError, match="MCP_MAX_SEARCH_SCAN"):
            await service.export_notes_csv(None, True, True, True, False, False, False)


@pytest.fixture
def subtree_collection(tmp_path: Path) -> Iterator[tuple[str, int, int]]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        model = collection.models.current()
        parent = int(collection.decks.id("Parent"))
        child = int(collection.decks.id("Parent::Child"))
        note = collection.new_note(model)
        note["Front"] = "front-parent"
        note["Back"] = "back-parent"
        collection.add_note(note, parent)
        for index in range(6):
            note = collection.new_note(model)
            note["Front"] = f"front-child-{index}"
            note["Back"] = f"back-child-{index}"
            collection.add_note(note, child)
    finally:
        collection.close()
    yield path, parent, child


@pytest.fixture
def media_collection(tmp_path: Path) -> Iterator[tuple[str, int]]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        model = collection.models.current()
        deck = int(collection.decks.id("DeckA"))
        note = collection.new_note(model)
        note["Front"] = 'front-a1 <img src="pic.png">'
        note["Back"] = "back-a1"
        collection.add_note(note, deck)
        collection.media.write_data("pic.png", b"\x89PNG\r\n\x1a\n" + b"Y" * 200)
    finally:
        collection.close()
    yield path, deck


@pytest.mark.anyio
async def test_export_invalid_deck_writes_nothing(
    export_collection: tuple[str, int, int],
) -> None:
    path, _, _ = export_collection
    exports_dir = Path(path).parent / "exports"
    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(LookupError, match="not found"):
            await service.export_apkg(123456789, True, True, False)
        with pytest.raises(LookupError, match="not found"):
            await service.export_notes_csv(123456789, True, True, True, False, False, False)
    assert not exports_dir.exists()


@pytest.mark.anyio
async def test_export_bound_counts_child_decks(
    subtree_collection: tuple[str, int, int],
) -> None:
    path, parent, _ = subtree_collection
    exports_dir = Path(path).parent / "exports"
    # Direct deck count is 1, subtree count is 7; 3 is strictly between them.
    async with AnkiCollectionService(path, max_page_size=100, max_search_scan=3) as service:
        with pytest.raises(ResourceLimitError, match="MCP_MAX_SEARCH_SCAN"):
            await service.export_apkg(parent, True, True, False)
        with pytest.raises(ResourceLimitError, match="MCP_MAX_SEARCH_SCAN"):
            await service.export_notes_csv(parent, True, True, True, False, False, False)
        with pytest.raises(ResourceLimitError, match="MCP_MAX_SEARCH_SCAN"):
            await service.export_apkg(None, True, True, False)
    assert not exports_dir.exists()


@pytest.mark.anyio
async def test_export_option_flags_take_effect(media_collection: tuple[str, int]) -> None:
    path, deck = media_collection
    base_members = {"collection.anki2", "collection.anki21b", "media", "meta"}
    async with AnkiCollectionService(path, max_page_size=100) as service:
        with_media = await service.export_apkg(deck, True, True, False)
        without_media = await service.export_apkg(deck, False, True, False)
        with_deck = await service.export_notes_csv(deck, True, True, True, False, False, False)
        without_deck = await service.export_notes_csv(deck, True, True, False, False, False, False)

    with zipfile.ZipFile(Path(with_media["path"])) as archive:
        extra = [name for name in archive.namelist() if name not in base_members]
        assert extra
        assert all(archive.getinfo(name).file_size > 0 for name in extra)
    with zipfile.ZipFile(Path(without_media["path"])) as archive:
        assert [name for name in archive.namelist() if name not in base_members] == []

    with_deck_text = Path(with_deck["path"]).read_text(encoding="utf-8")
    without_deck_text = Path(without_deck["path"]).read_text(encoding="utf-8")
    assert "#deck column" in with_deck_text
    assert "#deck column" not in without_deck_text


@pytest.mark.anyio
async def test_export_notes_csv_inlines_within_tiny_budget(
    export_collection: tuple[str, int, int],
) -> None:
    path, deck_a, _ = export_collection
    async with AnkiCollectionService(path, max_page_size=100, max_response_bytes=4097) as service:
        result = await service.export_notes_csv(deck_a, True, True, True, False, False, True)

    assert "inline_omitted" not in result
    assert base64.b64decode(result["content_base64"]) == Path(result["path"]).read_bytes()
