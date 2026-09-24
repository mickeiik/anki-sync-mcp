from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from anki.collection import Collection
from starlette.testclient import TestClient

from anki_mcp.app import create_app
from anki_mcp.collection import AnkiCollectionService
from anki_mcp.config import Settings


def _trashed_collection(path: Path, files: dict[str, bytes]) -> None:
    """Create a collection whose media files have been moved into media.trash."""
    collection = Collection(str(path))
    try:
        media_dir = Path(collection.media.dir())
        media_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (media_dir / name).write_bytes(content)
        collection.media.trash_files(list(files))
    finally:
        collection.close()


def _trash_folder(path: Path) -> Path:
    return path.parent / "media.trash"


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


def _settings(path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("MCP_AUTH_TOKEN", "trash-token")
    monkeypatch.setenv("ANKI_COLLECTION_PATH", str(path))
    monkeypatch.setenv("ANKI_SYNC_ON_WRITE", "false")
    monkeypatch.setenv("MCP_SCOPES", "read,write,admin,destructive")
    monkeypatch.setenv("ANKI_ALLOW_DESTRUCTIVE", "true")
    return Settings(_env_file=None)


def _initialize(client: TestClient) -> dict[str, str]:
    headers = {
        "Authorization": "Bearer trash-token",
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


@pytest.mark.anyio
async def test_preview_reports_trashed_media_and_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    _trashed_collection(path, {"a.txt": b"aaaa", "b.txt": b"bb"})

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        first = await service.executor.run(lambda adapter: adapter.preview_media_empty_trash())
        second = await service.executor.run(lambda adapter: adapter.preview_media_empty_trash())
        (_trash_folder(path) / "c.txt").write_bytes(b"ccc")
        third = await service.executor.run(lambda adapter: adapter.preview_media_empty_trash())

    assert first["files"] == 2
    assert first["bytes"] == 6
    assert first["state_fingerprint"] == second["state_fingerprint"]
    assert first["state_fingerprint"] != third["state_fingerprint"]


@pytest.mark.anyio
async def test_preview_reports_zero_when_nothing_trashed(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    collection = Collection(str(path))
    collection.close()

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        preview = await service.executor.run(lambda adapter: adapter.preview_media_empty_trash())

    assert preview["files"] == 0
    assert preview["bytes"] == 0
    assert "state_fingerprint" in preview


@pytest.mark.anyio
async def test_empty_removes_trashed_files_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    _trashed_collection(path, {"a.txt": b"aaaa", "b.txt": b"bb"})
    trash = _trash_folder(path)

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        removed = await service.executor.run(lambda adapter: adapter.empty_media_trash())
        again = await service.executor.run(lambda adapter: adapter.empty_media_trash())

    assert removed == {"emptied": True, "files_removed": 2}
    assert again == {"emptied": True, "files_removed": 0}
    assert not trash.exists() or not any(trash.iterdir())


def test_app_empty_trash_rejects_stale_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "collection.anki2"
    _trashed_collection(path, {"a.txt": b"aaaa"})
    trash = _trash_folder(path)

    with TestClient(create_app(_settings(path, monkeypatch))) as client:
        headers = _initialize(client)
        preview = _payload(_call(client, headers, 2, "anki_media_empty_trash_preview", {}))
        assert preview["impact"]["files"] == 1
        token = preview["confirmation_token"]

        # Adding a trashed file changes the impact, so the preview token no longer matches.
        (trash / "b.txt").write_bytes(b"bb")
        refreshed = _payload(_call(client, headers, 3, "anki_media_empty_trash_preview", {}))
        assert refreshed["impact"]["state_fingerprint"] != preview["impact"]["state_fingerprint"]

        stale = _call(
            client,
            headers,
            4,
            "anki_media_empty_trash",
            {"confirmation_token": token, "idempotency_key": "empty-trash-stale"},
        )
        assert stale.get("isError") is True
        assert "DESTRUCTIVE_CONFIRMATION_REQUIRED" in stale["content"][0]["text"]
        assert (trash / "a.txt").exists()
        assert (trash / "b.txt").exists()


def test_app_empty_trash_guarded_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "collection.anki2"
    _trashed_collection(path, {"a.txt": b"aaaa", "b.txt": b"bb"})
    trash = _trash_folder(path)

    with TestClient(create_app(_settings(path, monkeypatch))) as client:
        headers = _initialize(client)
        preview = _payload(_call(client, headers, 2, "anki_media_empty_trash_preview", {}))
        assert preview["impact"]["files"] == 2
        token = preview["confirmation_token"]

        garbage = _call(
            client,
            headers,
            3,
            "anki_media_empty_trash",
            {"confirmation_token": "not-a-token"},
        )
        assert garbage.get("isError") is True
        assert "DESTRUCTIVE_CONFIRMATION_REQUIRED" in garbage["content"][0]["text"]
        assert (trash / "a.txt").exists()
        assert (trash / "b.txt").exists()

        applied = _payload(
            _call(
                client,
                headers,
                4,
                "anki_media_empty_trash",
                {"confirmation_token": token, "idempotency_key": "empty-trash-1"},
            )
        )

    assert applied["state"] == "committed"
    assert applied["result"]["files_removed"] == 2
    assert Path(applied["result"]["backup"]["path"]).is_file()
    assert token not in json.dumps(applied)
    assert not trash.exists() or not any(trash.iterdir())


@pytest.mark.anyio
async def test_preview_and_empty_reject_symlinked_trash_folder(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    collection = Collection(str(path))
    try:
        media_dir = Path(collection.media.dir())
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / "live.txt").write_bytes(b"live")
    finally:
        collection.close()
    os.symlink(media_dir, media_dir.parent / "media.trash")

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        with pytest.raises(ValueError, match="symbolic link"):
            await service.executor.run(
                lambda adapter: adapter.preview_media_empty_trash()
            )
        with pytest.raises(ValueError, match="symbolic link"):
            await service.executor.run(lambda adapter: adapter.empty_media_trash())

    assert (media_dir / "live.txt").read_bytes() == b"live"


@pytest.mark.anyio
async def test_preview_rejects_non_directory_trash_path(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    collection = Collection(str(path))
    collection.close()
    (tmp_path / "media.trash").write_bytes(b"not a dir")

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        with pytest.raises(ValueError, match="not a directory"):
            await service.executor.run(
                lambda adapter: adapter.preview_media_empty_trash()
            )


@pytest.mark.anyio
async def test_preview_rejects_non_regular_trash_entry(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    collection = Collection(str(path))
    collection.close()
    (tmp_path / "media.trash" / "subdir").mkdir(parents=True)

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        with pytest.raises(ValueError, match="non-regular"):
            await service.executor.run(
                lambda adapter: adapter.preview_media_empty_trash()
            )


@pytest.mark.anyio
async def test_preview_raises_when_scan_bound_is_exceeded(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    _trashed_collection(path, {"a.txt": b"aaaa", "b.txt": b"bb"})

    async with AnkiCollectionService(
        str(path), max_page_size=100, max_search_scan=1
    ) as service:
        with pytest.raises(ValueError, match="MCP_MAX_SEARCH_SCAN"):
            await service.executor.run(
                lambda adapter: adapter.preview_media_empty_trash()
            )


@pytest.mark.anyio
async def test_preview_handles_non_utf8_trash_filename(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    collection = Collection(str(path))
    collection.close()
    trash = tmp_path / "media.trash"
    trash.mkdir()
    (trash / os.fsdecode(os.fsencode(b"bad\xff.bin"))).write_bytes(b"x")

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        preview = await service.executor.run(
            lambda adapter: adapter.preview_media_empty_trash()
        )

    assert preview["files"] == 1


@pytest.mark.anyio
async def test_preview_lists_trashed_items_and_paginates(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    _trashed_collection(path, {"a.txt": b"aaaa", "b.txt": b"bb", "c.txt": b"ccc"})

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        full = await service.executor.run(lambda adapter: adapter.preview_media_empty_trash())
        page = await service.executor.run(
            lambda adapter: adapter.preview_media_empty_trash(offset=1, limit=1)
        )

    assert full["items"] == [
        {"filename": "a.txt", "size_bytes": 4},
        {"filename": "b.txt", "size_bytes": 2},
        {"filename": "c.txt", "size_bytes": 3},
    ]
    assert full["total"] == 3
    assert full["offset"] == 0
    assert full["limit"] == 100
    assert full["has_more"] is False

    assert page["items"] == [{"filename": "b.txt", "size_bytes": 2}]
    assert page["total"] == 3
    assert page["offset"] == 1
    assert page["limit"] == 1
    assert page["has_more"] is True
    # Paging must not change the fingerprint, which binds the FULL trash contents.
    assert page["state_fingerprint"] == full["state_fingerprint"]


def test_app_empty_trash_guarded_flow_with_paging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "collection.anki2"
    _trashed_collection(path, {"a.txt": b"aaaa", "b.txt": b"bb"})
    trash = _trash_folder(path)

    with TestClient(create_app(_settings(path, monkeypatch))) as client:
        headers = _initialize(client)
        preview = _payload(
            _call(
                client,
                headers,
                2,
                "anki_media_empty_trash_preview",
                {"offset": 1, "limit": 1},
            )
        )
        assert preview["impact"]["items"] == [{"filename": "b.txt", "size_bytes": 2}]
        token = preview["confirmation_token"]

        applied = _payload(
            _call(
                client,
                headers,
                3,
                "anki_media_empty_trash",
                {
                    "confirmation_token": token,
                    "offset": 1,
                    "limit": 1,
                    "idempotency_key": "empty-trash-paged",
                },
            )
        )

    assert applied["state"] == "committed"
    assert applied["result"]["files_removed"] == 2
    assert not trash.exists() or not any(trash.iterdir())


@pytest.mark.anyio
async def test_symlink_entry_is_emptied_without_following_the_target(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    collection = Collection(str(path))
    try:
        media_dir = Path(collection.media.dir())
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / "live.txt").write_bytes(b"live")
    finally:
        collection.close()
    trash = tmp_path / "media.trash"
    trash.mkdir()
    (trash / "link.txt").symlink_to(media_dir / "live.txt")

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        preview = await service.executor.run(
            lambda adapter: adapter.preview_media_empty_trash()
        )
        removed = await service.executor.run(lambda adapter: adapter.empty_media_trash())

    assert preview["files"] == 1
    assert removed == {"emptied": True, "files_removed": 1}
    assert not (trash / "link.txt").exists() and not (trash / "link.txt").is_symlink()
    assert (media_dir / "live.txt").read_bytes() == b"live"
