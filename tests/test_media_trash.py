from __future__ import annotations

import json
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

    assert first["files"] == 2
    assert first["bytes"] == 6
    assert first["state_fingerprint"] == second["state_fingerprint"]


@pytest.mark.anyio
async def test_preview_reports_zero_when_nothing_trashed(tmp_path: Path) -> None:
    path = tmp_path / "collection.anki2"
    collection = Collection(str(path))
    collection.close()

    async with AnkiCollectionService(str(path), max_page_size=100) as service:
        preview = await service.executor.run(lambda adapter: adapter.preview_media_empty_trash())

    assert preview["files"] == 0
    assert preview["bytes"] == 0


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
