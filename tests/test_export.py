from __future__ import annotations

import base64
import hashlib
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from anki.collection import Collection

from anki_mcp.collection import AnkiCollectionService


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
    assert result["exported_cards"] == 2  # only DeckA's cards
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
    assert result["inline_reason"] == "exceeds MCP_MAX_RESPONSE_BYTES"


@pytest.mark.anyio
async def test_export_is_bounded_by_search_scan(
    export_collection: tuple[str, int, int],
) -> None:
    path, deck_a, _ = export_collection
    async with AnkiCollectionService(path, max_page_size=100, max_search_scan=1) as service:
        with pytest.raises(ValueError, match="MCP_MAX_SEARCH_SCAN"):
            await service.export_apkg(deck_a, True, True, False)
        with pytest.raises(ValueError, match="MCP_MAX_SEARCH_SCAN"):
            await service.export_notes_csv(None, True, True, True, False, False, False)
