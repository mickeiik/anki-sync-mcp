from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from anki.collection import Collection
from starlette.testclient import TestClient

from anki_mcp.app import create_app
from anki_mcp.collection import AnkiCollectionService, ImportFileError, ResourceLimitError
from anki_mcp.config import Settings


def _create_source(path: Path, deck_name: str = "Deck A") -> int:
    """Create a small collection with two Basic notes; return the deck ID."""
    collection = Collection(str(path))
    try:
        model = collection.models.current()
        deck_id = int(collection.decks.id(deck_name))
        for front, back in (("front-a1", "back-a1"), ("front-a2", "back-a2")):
            note = collection.new_note(model)
            note["Front"] = front
            note["Back"] = back
            collection.add_note(note, deck_id)
    finally:
        collection.close()
    return deck_id


def _empty_collection(path: Path) -> None:
    collection = Collection(str(path))
    collection.close()


def _imports_dir(path: str | Path) -> Path:
    folder = Path(path).parent / "imports"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


async def _export_fixtures(path: str, deck_id: int) -> tuple[dict, dict]:
    async with AnkiCollectionService(path, max_page_size=100) as service:
        apkg = await service.export_apkg(deck_id, False, True, False)
        csv = await service.export_notes_csv(deck_id, True, True, False, False, False, False)
    return apkg, csv


async def _field_tuples(service: AnkiCollectionService) -> set[tuple[str, ...]]:
    """Return every note's full field tuple (all fields, in order)."""
    searched = await service.search_notes("", 0, 100)
    return {
        tuple(field["value"] for field in (await service.get_note(item["id"]))["fields"])
        for item in searched["items"]
    }


@pytest.fixture
def source_collection(tmp_path: Path) -> Iterator[tuple[str, int]]:
    source = tmp_path / "source" / "collection.anki2"
    source.parent.mkdir(parents=True)
    deck_id = _create_source(source)
    yield str(source), deck_id


# --------------------------------------------------------------------------------------
# list / path validation / size cap
# --------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_import_files_orders_newest_first(tmp_path: Path) -> None:
    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    (imports / "old.csv").write_text("a,b\n")
    (imports / "new.apkg").write_bytes(b"x")
    (imports / "skip.txt").write_text("ignore me")
    os.utime(imports / "old.csv", (1_000_000, 1_000_000))
    os.utime(imports / "new.apkg", (2_000_000, 2_000_000))

    async with AnkiCollectionService(str(target), max_page_size=100) as service:
        page = await service.list_import_files(0, 10)

    assert [item["filename"] for item in page["items"]] == ["new.apkg", "old.csv"]
    assert page["total"] == 2
    assert all("size_bytes" in item and "mtime" in item for item in page["items"])


@pytest.mark.anyio
async def test_import_path_validation(tmp_path: Path) -> None:
    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    (imports / "good.apkg").write_bytes(b"x")
    (imports / "wrong.txt").write_bytes(b"x")
    outside = tmp_path / "outside.apkg"
    outside.write_bytes(b"x")
    (imports / "link.apkg").symlink_to(outside)

    async with AnkiCollectionService(str(target), max_page_size=100) as service:
        with pytest.raises(LookupError, match="not found"):
            await service.preview_import_apkg("missing.apkg")
        with pytest.raises(ValueError, match="plain filename"):
            await service.preview_import_apkg("../outside.apkg")
        with pytest.raises(ValueError, match="plain filename"):
            await service.preview_import_apkg("sub/good.apkg")
        with pytest.raises(ValueError, match="plain filename"):
            await service.preview_import_apkg("sub\\good.apkg")
        with pytest.raises(ValueError, match=r"\.apkg or \.csv"):
            await service.preview_import_apkg("wrong.txt")
        with pytest.raises(ValueError, match="symbolic link"):
            await service.preview_import_apkg("link.apkg")
        with pytest.raises(ValueError, match=r"\.csv file"):
            await service.preview_import_csv("good.apkg", None)


@pytest.mark.anyio
async def test_import_size_cap(tmp_path: Path) -> None:
    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    (imports / "big.csv").write_text("f,b\n" + ("x" * 500) + "\n")

    async with AnkiCollectionService(
        str(target), max_page_size=100, max_import_bytes=64
    ) as service:
        with pytest.raises(ResourceLimitError, match="ANKI_MAX_IMPORT_BYTES"):
            await service.preview_import_csv("big.csv", None)


@pytest.mark.anyio
async def test_preview_apkg_rejects_corrupt_and_missing_members(tmp_path: Path) -> None:
    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    (imports / "corrupt.apkg").write_bytes(b"not a zip")
    with zipfile.ZipFile(imports / "missing.apkg", "w") as archive:
        archive.writestr("hello.txt", "hi")
    # A structurally valid zip whose member data was altered: namelist still works,
    # but the CRC integrity check must reject it.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("meta", "{}")
        archive.writestr("collection.anki2", b"collection-bytes")
    raw = bytearray(buffer.getvalue())
    raw[raw.find(b"collection-bytes")] ^= 0xFF
    (imports / "tampered.apkg").write_bytes(bytes(raw))

    async with AnkiCollectionService(str(target), max_page_size=100) as service:
        with pytest.raises(ValueError, match="valid zip archive"):
            await service.preview_import_apkg("corrupt.apkg")
        with pytest.raises(ValueError, match="missing required collection members"):
            await service.preview_import_apkg("missing.apkg")
        with pytest.raises(ValueError, match="integrity check"):
            await service.preview_import_apkg("tampered.apkg")


@pytest.mark.anyio
async def test_preview_apkg_rejects_declared_uncompressed_bomb(tmp_path: Path) -> None:
    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    with zipfile.ZipFile(imports / "bomb.apkg", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("meta", "{}")
        archive.writestr("collection.anki2", b"\0" * 2_000_000)

    async with AnkiCollectionService(
        str(target), max_page_size=100, max_import_bytes=65_536
    ) as service:
        # The compressed file is tiny; only the declared uncompressed size trips the cap.
        with pytest.raises(ResourceLimitError, match="ANKI_MAX_IMPORT_BYTES"):
            await service.preview_import_apkg("bomb.apkg")


# --------------------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_apkg_round_trip(source_collection: tuple[str, int], tmp_path: Path) -> None:
    source, deck_id = source_collection
    apkg, _ = await _export_fixtures(source, deck_id)

    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    shutil.copy2(apkg["path"], imports / "fixture.apkg")

    async with AnkiCollectionService(str(target), max_page_size=100) as service:
        preview = await service.preview_import_apkg("fixture.apkg")
        assert preview["validated"] is True
        assert preview["sha256"] == apkg["sha256"]

        result = await service.import_apkg("fixture.apkg", True, 2, 2, True, False)
        searched = await service.search_notes("", 0, 100)
        field_tuples = await _field_tuples(service)

    assert result["notes_added"] == 2
    assert result["notes_found"] == 2
    assert {note["first_field"] for note in searched["items"]} == {"front-a1", "front-a2"}
    assert field_tuples == {("front-a1", "back-a1"), ("front-a2", "back-a2")}


@pytest.mark.anyio
async def test_csv_round_trip(source_collection: tuple[str, int], tmp_path: Path) -> None:
    source, deck_id = source_collection
    _, csv = await _export_fixtures(source, deck_id)

    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    shutil.copy2(csv["path"], imports / "fixture.csv")

    collection = Collection(str(target))
    try:
        notetype_id = int(collection.models.by_name("Basic")["id"])
        deck = int(collection.decks.id("Imported"))
    finally:
        collection.close()

    async with AnkiCollectionService(str(target), max_page_size=100) as service:
        preview = await service.preview_import_csv("fixture.csv", "\t")
        assert preview["delimiter"] == "\t"
        assert preview["is_html"] is True
        assert preview["sample_rows"][0][:2] == ["front-a1", "back-a1"]

        result = await service.import_csv(
            "fixture.csv", notetype_id, [1, 2], deck, "\t", True, [], "preserve"
        )
        searched = await service.search_notes("", 0, 100)
        cards = await service.search_cards("", 0, 100)
        field_tuples = await _field_tuples(service)

    assert result["notes_added"] == 2
    assert result["notes_found"] == 2
    assert {note["first_field"] for note in searched["items"]} == {
        "front-a1",
        "front-a2",
    }
    assert field_tuples == {("front-a1", "back-a1"), ("front-a2", "back-a2")}
    assert cards["total"] == 2


@pytest.mark.anyio
async def test_csv_field_columns_must_map_every_field(
    source_collection: tuple[str, int], tmp_path: Path
) -> None:
    source, deck_id = source_collection
    _, csv = await _export_fixtures(source, deck_id)

    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    shutil.copy2(csv["path"], imports / "fixture.csv")

    collection = Collection(str(target))
    try:
        notetype_id = int(collection.models.by_name("Basic")["id"])
    finally:
        collection.close()

    async with AnkiCollectionService(str(target), max_page_size=100) as service:
        with pytest.raises(ValueError, match="field_columns must map every field"):
            await service.import_csv(
                "fixture.csv", notetype_id, [1], 1, "\t", True, [], "preserve"
            )
        with pytest.raises(ValueError, match="dupe_resolution"):
            await service.import_csv(
                "fixture.csv", notetype_id, [1, 2], 1, "\t", True, [], "bogus"
            )


@pytest.mark.anyio
async def test_csv_field_columns_reject_out_of_range_and_duplicates(
    source_collection: tuple[str, int], tmp_path: Path
) -> None:
    source, deck_id = source_collection
    _, csv = await _export_fixtures(source, deck_id)

    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    shutil.copy2(csv["path"], imports / "fixture.csv")

    collection = Collection(str(target))
    try:
        notetype_id = int(collection.models.by_name("Basic")["id"])
    finally:
        collection.close()

    async with AnkiCollectionService(str(target), max_page_size=100) as service:
        # The exported CSV has three columns; index 5 is out of range.
        with pytest.raises(ValueError, match="outside the"):
            await service.import_csv(
                "fixture.csv", notetype_id, [1, 5], 1, "\t", True, [], "preserve"
            )
        # Two fields cannot map to the same non-zero column.
        with pytest.raises(ValueError, match="same column"):
            await service.import_csv(
                "fixture.csv", notetype_id, [1, 1], 1, "\t", True, [], "preserve"
            )


@pytest.mark.anyio
async def test_csv_reports_first_field_matches(tmp_path: Path) -> None:
    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    (imports / "first.csv").write_text("front-a1\tback-a1\nfront-a2\tback-a2\n")
    (imports / "second.csv").write_text("front-a1\tchanged\n")

    collection = Collection(str(target))
    try:
        notetype_id = int(collection.models.by_name("Basic")["id"])
    finally:
        collection.close()

    async with AnkiCollectionService(str(target), max_page_size=100) as service:
        first = await service.import_csv(
            "first.csv", notetype_id, [1, 2], 1, "\t", None, [], "preserve"
        )
        second = await service.import_csv(
            "second.csv", notetype_id, [1, 2], 1, "\t", None, [], "preserve"
        )

    assert first["notes_added"] == 2
    # The second row matches an existing note by first field only.
    assert second["notes_added"] == 0
    assert second["notes_matched_first_field"] == 1
    assert first["notes_matched_first_field"] == 0


@pytest.mark.anyio
async def test_import_apkg_rejects_file_swapped_after_preview(
    source_collection: tuple[str, int], tmp_path: Path
) -> None:
    source, deck_id = source_collection
    apkg, _ = await _export_fixtures(source, deck_id)

    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    staged = imports / "fixture.apkg"
    shutil.copy2(apkg["path"], staged)

    async with AnkiCollectionService(str(target), max_page_size=100) as service:
        preview = await service.preview_import_apkg("fixture.apkg")
        assert preview["validated"] is True

        staged.write_bytes(b"not a zip archive")
        with pytest.raises(ValueError, match="valid zip archive"):
            await service.import_apkg("fixture.apkg", True, 2, 2, True, False)


# --------------------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------------------


def _tool_names(
    path: Path,
    *,
    allow_import: bool,
    scopes: str = "read,write,admin,destructive",
    allow_schema: bool = False,
    allow_full_sync: bool = False,
) -> list[str]:
    settings = Settings(
        _env_file=None,
        MCP_AUTH_TOKEN="test-token",
        ANKI_COLLECTION_PATH=str(path),
        MCP_SCOPES=scopes,
        ANKI_SYNC_ON_WRITE=False,
        ANKI_ALLOW_IMPORT=allow_import,
        ANKI_ALLOW_SCHEMA_CHANGES=allow_schema,
        ANKI_ALLOW_FULL_SYNC=allow_full_sync,
    )
    headers = {
        "Authorization": "Bearer test-token",
        "Accept": "application/json, text/event-stream",
    }
    with TestClient(create_app(settings)) as client:
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
    return [tool["name"] for tool in listed.json()["result"]["tools"]]


def test_import_tools_require_flags_and_destructive_scope(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    Collection(str(path)).close()
    gated = {
        "anki_import_apkg_preview",
        "anki_import_apkg",
        "anki_import_csv_preview",
        "anki_import_csv",
    }

    disabled = _tool_names(path, allow_import=False)
    assert "anki_import_files_list" in disabled
    assert not (gated & set(disabled))

    csv_only = _tool_names(path, allow_import=True)
    assert {"anki_import_csv_preview", "anki_import_csv"} <= set(csv_only)
    assert not ({"anki_import_apkg_preview", "anki_import_apkg"} & set(csv_only))

    full = _tool_names(path, allow_import=True, allow_schema=True, allow_full_sync=True)
    assert gated <= set(full)

    # apkg tools require all three flags; each missing flag hides them independently.
    no_full_sync = _tool_names(path, allow_import=True, allow_schema=True, allow_full_sync=False)
    assert {"anki_import_csv_preview", "anki_import_csv"} <= set(no_full_sync)
    assert not ({"anki_import_apkg_preview", "anki_import_apkg"} & set(no_full_sync))

    no_schema = _tool_names(path, allow_import=True, allow_schema=False, allow_full_sync=True)
    assert {"anki_import_csv_preview", "anki_import_csv"} <= set(no_schema)
    assert not ({"anki_import_apkg_preview", "anki_import_apkg"} & set(no_schema))

    no_scope = _tool_names(path, allow_import=True, scopes="read,write,admin")
    assert "anki_import_files_list" in no_scope
    assert not (gated & set(no_scope))


# --------------------------------------------------------------------------------------
# app-level token flow, idempotency, backups
# --------------------------------------------------------------------------------------


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


def _import_settings(path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("MCP_AUTH_TOKEN", "import-token")
    monkeypatch.setenv("ANKI_COLLECTION_PATH", str(path))
    monkeypatch.setenv("ANKI_SYNC_ON_WRITE", "false")
    monkeypatch.setenv("MCP_SCOPES", "read,write,admin,destructive")
    monkeypatch.setenv("ANKI_ALLOW_DESTRUCTIVE", "true")
    monkeypatch.setenv("ANKI_ALLOW_SCHEMA_CHANGES", "true")
    monkeypatch.setenv("ANKI_ALLOW_FULL_SYNC", "true")
    monkeypatch.setenv("ANKI_ALLOW_IMPORT", "true")
    return Settings(_env_file=None)


_HEADERS = {
    "Authorization": "Bearer import-token",
    "Accept": "application/json, text/event-stream",
}


def _initialize(client: TestClient) -> dict[str, str]:
    headers = dict(_HEADERS)
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


def test_app_apkg_import_token_flow_replay_and_backup(
    source_collection: tuple[str, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, deck_id = source_collection
    apkg, _ = asyncio.run(_export_fixtures(source, deck_id))

    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    shutil.copy2(apkg["path"], _imports_dir(target) / "fixture.apkg")

    settings = _import_settings(target, monkeypatch)
    with TestClient(create_app(settings)) as client:
        headers = _initialize(client)

        wrong = _call(
            client,
            headers,
            2,
            "anki_import_apkg",
            {
                "filename": "fixture.apkg",
                "confirmation_token": "bogus",
                "idempotency_key": "apkg-1",
            },
        )
        assert "DESTRUCTIVE_CONFIRMATION_REQUIRED" in wrong["content"][0]["text"]

        preview = _payload(
            _call(client, headers, 3, "anki_import_apkg_preview", {"filename": "fixture.apkg"})
        )
        arguments = {
            "filename": "fixture.apkg",
            "confirmation_token": preview["confirmation_token"],
            "idempotency_key": "apkg-1",
        }
        applied = _payload(_call(client, headers, 4, "anki_import_apkg", arguments))
        assert applied["state"] == "committed"
        assert applied["result"]["notes_added"] == 2
        assert Path(applied["result"]["backup"]["path"]).is_file()

        replayed = _payload(_call(client, headers, 5, "anki_import_apkg", arguments))
        assert replayed == applied

        # A retry after the token was consumed must re-preview and still replay the
        # stored receipt instead of conflicting on the changed token.
        fresh_preview = _payload(
            _call(client, headers, 6, "anki_import_apkg_preview", {"filename": "fixture.apkg"})
        )
        replayed_with_fresh_token = _payload(
            _call(
                client,
                headers,
                7,
                "anki_import_apkg",
                {
                    "filename": "fixture.apkg",
                    "confirmation_token": fresh_preview["confirmation_token"],
                    "idempotency_key": "apkg-1",
                },
            )
        )
        assert replayed_with_fresh_token == applied

        listed = _payload(_call(client, headers, 8, "anki_notes_search", {"query": ""}))
        assert listed["total"] == 2

    collection = Collection(str(target))
    try:
        assert collection.note_count() == 2
    finally:
        collection.close()


def test_app_csv_import_token_flow_replay_and_backup(
    source_collection: tuple[str, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, deck_id = source_collection
    _, csv = asyncio.run(_export_fixtures(source, deck_id))

    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    shutil.copy2(csv["path"], _imports_dir(target) / "fixture.csv")

    collection = Collection(str(target))
    try:
        notetype_id = int(collection.models.by_name("Basic")["id"])
    finally:
        collection.close()

    settings = _import_settings(target, monkeypatch)
    with TestClient(create_app(settings)) as client:
        headers = _initialize(client)

        preview = _payload(
            _call(client, headers, 2, "anki_import_csv_preview", {"filename": "fixture.csv"})
        )
        assert preview["impact"]["delimiter"] == "\t"

        arguments = {
            "filename": "fixture.csv",
            "notetype_id": notetype_id,
            "field_columns": [1, 2],
            "deck_id": 1,
            "confirmation_token": preview["confirmation_token"],
            "idempotency_key": "csv-1",
        }
        applied = _payload(_call(client, headers, 3, "anki_import_csv", arguments))
        assert applied["state"] == "committed"
        assert applied["result"]["notes_added"] == 2
        assert Path(applied["result"]["backup"]["path"]).is_file()

        replayed = _payload(_call(client, headers, 4, "anki_import_csv", arguments))
        assert replayed == applied

        # Re-preview and replay the same key with a fresh token: the stored receipt is
        # returned and no second import happens.
        fresh_preview = _payload(
            _call(client, headers, 5, "anki_import_csv_preview", {"filename": "fixture.csv"})
        )
        replayed_with_fresh_token = _payload(
            _call(
                client,
                headers,
                6,
                "anki_import_csv",
                {**arguments, "confirmation_token": fresh_preview["confirmation_token"]},
            )
        )
        assert replayed_with_fresh_token == applied

        listed = _payload(_call(client, headers, 7, "anki_notes_search", {"query": ""}))
        assert listed["total"] == 2

    collection = Collection(str(target))
    try:
        assert collection.note_count() == 2
    finally:
        collection.close()


def test_app_csv_import_rejects_tail_rewrite_past_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    staged = imports / "fixture.csv"
    rows = "\n".join(f"front-{index}\tback-{index}" for index in range(600))
    staged.write_text(rows + "\n")

    collection = Collection(str(target))
    try:
        notetype_id = int(collection.models.by_name("Basic")["id"])
    finally:
        collection.close()

    settings = _import_settings(target, monkeypatch)
    with TestClient(create_app(settings)) as client:
        headers = _initialize(client)
        preview = _payload(
            _call(client, headers, 2, "anki_import_csv_preview", {"filename": "fixture.csv"})
        )

        # Rewrite a byte past the 4096-byte sample while keeping the file length.
        original = bytearray(staged.read_bytes())
        assert len(original) > 4096
        original[-2] = ord("X") if original[-2] != ord("X") else ord("Y")
        staged.write_bytes(bytes(original))
        assert len(staged.read_bytes()) == len(original)

        refused = _call(
            client,
            headers,
            3,
            "anki_import_csv",
            {
                "filename": "fixture.csv",
                "notetype_id": notetype_id,
                "field_columns": [1, 2],
                "deck_id": 1,
                "confirmation_token": preview["confirmation_token"],
                "idempotency_key": "csv-tail-1",
            },
        )
        assert refused.get("isError") is True
        assert "DESTRUCTIVE_CONFIRMATION_REQUIRED" in refused["content"][0]["text"]

    assert Collection(str(target)).note_count() == 0


def test_app_apkg_predictable_file_failure_is_invalid_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target" / "collection.anki2"
    target.parent.mkdir(parents=True)
    _empty_collection(target)
    imports = _imports_dir(target)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("meta", "")
        archive.writestr("collection.anki2", b"collection-bytes")
    (imports / "emptymeta.apkg").write_bytes(buffer.getvalue())

    settings = _import_settings(target, monkeypatch)
    with TestClient(create_app(settings)) as client:
        headers = _initialize(client)
        # The empty meta passes preview validation but fails predictably at import.
        preview = _payload(
            _call(client, headers, 2, "anki_import_apkg_preview", {"filename": "emptymeta.apkg"})
        )
        arguments = {
            "filename": "emptymeta.apkg",
            "confirmation_token": preview["confirmation_token"],
            "idempotency_key": "apkg-bad-1",
        }
        failed = _call(client, headers, 3, "anki_import_apkg", arguments)
        assert failed.get("isError") is True
        assert "INVALID_ARGUMENT" in failed["content"][0]["text"]

        # The receipt was deleted, so the replay is refused again rather than
        # returning a stale "committed" outcome.
        replay = _call(client, headers, 4, "anki_import_apkg", arguments)
        assert replay.get("isError") is True
        assert "DESTRUCTIVE_CONFIRMATION_REQUIRED" in replay["content"][0]["text"]

    assert Collection(str(target)).note_count() == 0


def _craft_unsupported_method_apkg(path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("meta", b"{}")
        archive.writestr("collection.anki2", b"collection-bytes")
    raw = bytearray(buffer.getvalue())
    raw[8:10] = (99).to_bytes(2, "little")  # local header: unsupported method
    central = raw.find(b"PK\x01\x02")
    assert central > 0
    raw[central + 10 : central + 12] = (99).to_bytes(2, "little")
    path.write_bytes(bytes(raw))


def test_apkg_with_unsupported_compression_is_a_clean_file_error(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    Collection(str(path)).close()
    _craft_unsupported_method_apkg(_imports_dir(path) / "crafted.apkg")

    async def scenario() -> None:
        async with AnkiCollectionService(path, max_page_size=100) as service:
            with pytest.raises(ImportFileError):
                await service.coordinated_read(
                    lambda adapter: adapter.preview_import_apkg_file("crafted.apkg")
                )
            with pytest.raises(ImportFileError):
                await service.coordinated_mutation(
                    "anki_import_apkg",
                    "crafted-key",
                    {"filename": "crafted.apkg"},
                    lambda adapter: adapter.import_apkg(
                        "crafted.apkg", False, 2, 2, True, False
                    ),
                )
            # A predictable file failure deletes its receipt, so the replay re-executes.
            with pytest.raises(LookupError):
                await service.get_operation("crafted-key")

    asyncio.run(scenario())
