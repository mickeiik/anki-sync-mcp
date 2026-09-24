from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from anki.collection import Collection
from anki.errors import NetworkError
from anki.sync import SyncAuth, SyncOutput

from anki_mcp.collection import (
    AnkiCollectionService,
    DuplicateNoteError,
    FullSyncRequiredError,
    IdempotencyConflictError,
)
from anki_mcp.state import PersistentState


@pytest.fixture
def collection_path(tmp_path: Path) -> Iterator[str]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        deck_id = collection.decks.id("Study")
        note = collection.new_note(collection.models.current())
        note["Front"] = "existing"
        note["Back"] = "answer"
        collection.add_note(note, deck_id)
    finally:
        collection.close()
    yield path


@pytest.mark.anyio
async def test_general_note_workflows_validate_fields_duplicates_and_tags(
    collection_path: str,
) -> None:
    collection = Collection(collection_path)
    try:
        deck_id = collection.decks.id_for_name("Study")
        model = collection.models.by_name("Basic")
        assert deck_id is not None and model is not None
        model_id = int(model["id"])
    finally:
        collection.close()

    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        created = await service.create_note(
            deck_id=deck_id,
            note_type_id=model_id,
            fields={"Front": "new", "Back": "value"},
            tags=["phase-one", "MCP"],
        )
        found = await service.search_notes("tag:phase-one", offset=0, limit=10)
        note = await service.get_note(created["note_id"])
        updated = await service.update_note_fields(created["note_id"], {"Back": "changed"})
        tagged = await service.add_note_tags([created["note_id"]], ["durable"])
        untagged = await service.remove_note_tags([created["note_id"]], ["MCP"])
        final = await service.get_note(created["note_id"])

        with pytest.raises(DuplicateNoteError):
            await service.create_note(
                deck_id=deck_id,
                note_type_id=model_id,
                fields={"Front": "new", "Back": "different"},
                tags=[],
            )
        with pytest.raises(ValueError, match="fields"):
            await service.create_note(
                deck_id=deck_id,
                note_type_id=model_id,
                fields={"Front": "missing back"},
                tags=[],
            )
        with pytest.raises(ValueError, match="unknown field"):
            await service.update_note_fields(created["note_id"], {"Missing": "x"})

    assert found["items"][0]["id"] == created["note_id"]
    assert note["note_type_id"] == model_id
    assert note["card_ids"] == created["card_ids"]
    assert {item["name"]: item["value"] for item in note["fields"]} == {
        "Front": "new",
        "Back": "value",
    }
    assert updated == {"note_id": created["note_id"], "updated": True}
    assert tagged["updated_note_ids"] == [created["note_id"]]
    assert untagged["updated_note_ids"] == [created["note_id"]]
    assert "durable" in final["tags"]
    assert "MCP" not in final["tags"]
    assert {item["name"]: item["value"] for item in final["fields"]}["Back"] == "changed"


@pytest.mark.anyio
async def test_batch_create_is_bounded_atomic_and_rejects_duplicates(collection_path: str) -> None:
    collection = Collection(collection_path)
    try:
        deck_id = int(collection.decks.id_for_name("Study") or 0)
        model = collection.models.by_name("Basic")
        assert model is not None
        model_id = int(model["id"])
        initial_count = collection.note_count()
    finally:
        collection.close()

    requests = [
        {
            "deck_id": deck_id,
            "note_type_id": model_id,
            "fields": {"Front": "batch one", "Back": "1"},
            "tags": ["batch"],
        },
        {
            "deck_id": deck_id,
            "note_type_id": model_id,
            "fields": {"Front": "batch two", "Back": "2"},
            "tags": ["batch"],
        },
    ]
    async with AnkiCollectionService(
        collection_path, max_page_size=100, max_batch_size=2
    ) as service:
        created = await service.create_notes_batch(requests)
        with pytest.raises(ValueError, match="batch"):
            await service.create_notes_batch([*requests, requests[0]])
        with pytest.raises(DuplicateNoteError):
            await service.create_notes_batch(
                [
                    {
                        "deck_id": deck_id,
                        "note_type_id": model_id,
                        "fields": {"Front": "would otherwise commit", "Back": "x"},
                        "tags": [],
                    },
                    {
                        "deck_id": deck_id,
                        "note_type_id": model_id,
                        "fields": {"Front": "existing", "Back": "x"},
                        "tags": [],
                    },
                ]
            )

        with pytest.raises(DuplicateNoteError, match="batch"):
            await service.create_notes_batch(
                [
                    {
                        "deck_id": deck_id,
                        "note_type_id": model_id,
                        "fields": {"Front": "<b>same rendered value</b>", "Back": "x"},
                        "tags": [],
                    },
                    {
                        "deck_id": deck_id,
                        "note_type_id": model_id,
                        "fields": {"Front": "same rendered value", "Back": "y"},
                        "tags": [],
                    },
                ]
            )

    collection = Collection(collection_path)
    try:
        assert collection.note_count() == initial_count + 2
        assert not collection.find_notes('"would otherwise commit"')
    finally:
        collection.close()
    assert len(created["notes"]) == 2
    assert all(item["card_ids"] for item in created["notes"])


@pytest.mark.anyio
async def test_card_controls_support_arbitrary_note_types(collection_path: str) -> None:
    collection = Collection(collection_path)
    try:
        source = int(collection.decks.id("Cloze Source"))
        target = int(collection.decks.id("Cloze Target"))
        model = collection.models.by_name("Cloze")
        assert model is not None
        note = collection.new_note(model)
        note["Text"] = "{{c1::arbitrary}} card"
        note["Back Extra"] = "extra"
        collection.add_note(note, source)
        card_id = int(collection.find_cards(f"nid:{int(note.id)}")[0])
    finally:
        collection.close()

    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        moved = await service.change_card_deck([card_id], target)
        suspended = await service.suspend_cards([card_id])
        suspended_card = await service.get_card(card_id)
        unsuspended = await service.unsuspend_cards([card_id])
        active_card = await service.get_card(card_id)

    assert moved == {"card_ids": [card_id], "deck_id": target, "updated": True}
    assert suspended == {"card_ids": [card_id], "suspended": True}
    assert unsuspended == {"card_ids": [card_id], "suspended": False}
    assert suspended_card["scheduling"]["queue"] == -1
    assert active_card["scheduling"]["queue"] != -1


@pytest.mark.anyio
async def test_sync_auth_and_pending_full_sync_state_survive_restart(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(
            hkey="persistent-key", endpoint=endpoint
        ),
    )
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=2, server_media_usn=7),
    )
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        await service.sync(sync_media=True)
        status = await service.status()
    assert status["authenticated"] is True
    assert status["pending_full_sync"] == "FULL_SYNC"
    assert "persistent-key" not in repr(status)

    async with AnkiCollectionService(collection_path, max_page_size=100) as restarted:
        recovered = await restarted.status()
    assert recovered["authenticated"] is True
    assert recovered["pending_full_sync"] == "FULL_SYNC"


@pytest.mark.anyio
async def test_coordinator_retries_only_sync_after_local_commit(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_calls = 0

    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(
            hkey="persistent-key", endpoint=endpoint
        ),
    )

    def sync(_: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 2:
            raise NetworkError("post-commit network failure", None, None, None)
        return SyncOutput(required=0)

    monkeypatch.setattr(Collection, "sync_collection", sync)
    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        first = await service.coordinated_mutation(
            operation="anki_decks_create",
            idempotency_key="stable-operation-key",
            request={"name": "Durable Receipt"},
            mutate=lambda adapter: adapter.create_deck("Durable Receipt"),
        )
    assert first["local_committed"] is True
    assert first["remote_synced"] is False
    assert first["retryable"] is True

    mutation_replayed = False

    def must_not_replay(_: object) -> dict[str, object]:
        nonlocal mutation_replayed
        mutation_replayed = True
        raise AssertionError("mutation replayed")

    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as restarted:
        retried = await restarted.coordinated_mutation(
            operation="anki_decks_create",
            idempotency_key="stable-operation-key",
            request={"name": "Durable Receipt"},
            mutate=must_not_replay,
        )
        same = await restarted.coordinated_mutation(
            operation="anki_decks_create",
            idempotency_key="stable-operation-key",
            request={"name": "Durable Receipt"},
            mutate=must_not_replay,
        )
        with pytest.raises(IdempotencyConflictError):
            await restarted.coordinated_mutation(
                operation="anki_decks_create",
                idempotency_key="stable-operation-key",
                request={"name": "different"},
                mutate=must_not_replay,
            )

    assert mutation_replayed is False
    assert retried["remote_synced"] is True
    assert retried["result"] == first["result"]
    assert same == retried


@pytest.mark.anyio
async def test_media_coordinator_syncs_media_and_retries_without_replaying_mutation(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_calls: list[bool] = []
    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(
            hkey="persistent-key", endpoint=endpoint
        ),
    )

    def sync(_: Collection, auth: SyncAuth, sync_media: bool) -> SyncOutput:
        sync_calls.append(sync_media)
        if len(sync_calls) == 2:
            raise NetworkError("post-commit media sync failure", None, None, None)
        return SyncOutput(required=0)

    monkeypatch.setattr(Collection, "sync_collection", sync)
    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        first = await service.coordinated_mutation(
            operation="anki_media_store",
            idempotency_key="media-sync-key",
            request={"filename": "sync.txt"},
            mutate=lambda adapter: {"created": True},
            sync_media=True,
        )

    replayed = False

    def must_not_replay(_: object) -> dict[str, object]:
        nonlocal replayed
        replayed = True
        raise AssertionError("media mutation replayed")

    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as restarted:
        retried = await restarted.coordinated_mutation(
            operation="anki_media_store",
            idempotency_key="media-sync-key",
            request={"filename": "sync.txt"},
            mutate=must_not_replay,
            sync_media=True,
        )
        await restarted.coordinated_read(
            lambda adapter: adapter.list_media(0, 10),
            sync_before=True,
            sync_media=True,
        )

    assert sync_calls == [True, True, True, True]
    assert first["remote_synced"] is False
    assert first["media_synced"] is False
    assert first["retryable"] is True
    assert retried["remote_synced"] is True
    assert retried["media_synced"] is True
    assert retried["retryable"] is False
    assert replayed is False


def test_full_sync_reconciliation_preserves_pending_media_transfer(collection_path: str) -> None:
    state = PersistentState(collection_path)
    try:
        media_receipt = {
            "state": "committed",
            "local_committed": True,
            "remote_synced": False,
            "media_synced": False,
            "retryable": False,
            "result": {"filename": "pending.txt"},
        }
        state.put_receipt("media-key", "anki_media_store", "media-hash", media_receipt)
        state.put_receipt(
            "note-key",
            "anki_notes_create",
            "note-hash",
            {**media_receipt, "media_synced": None, "result": {"note_id": 1}},
        )

        state.mark_pending_discarded_by_full_download()

        media = state.get_receipt("media-key")
        note = state.get_receipt("note-key")
        assert media is not None
        assert media[2]["local_committed"] is True
        assert media[2]["remote_synced"] is True
        assert media[2]["media_synced"] is False
        assert media[2]["retryable"] is True
        assert note is not None
        assert note[2]["state"] == "discarded_by_full_download"
        assert note[2]["local_committed"] is False

        state.mark_all_remote_synced()
        uploaded = state.get_receipt("media-key")
        assert uploaded is not None
        assert uploaded[2]["remote_synced"] is True
        assert uploaded[2]["media_synced"] is False
        assert uploaded[2]["retryable"] is True
        assert state.pending_receipt_count() == 1
    finally:
        state.close()


@pytest.mark.anyio
async def test_coordinator_fails_closed_before_mutation_and_persists_full_sync_state(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(
            hkey="persistent-key", endpoint=endpoint
        ),
    )
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=3),
    )
    mutated = False

    def mutate(_: object) -> dict[str, object]:
        nonlocal mutated
        mutated = True
        return {"unexpected": True}

    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        with pytest.raises(FullSyncRequiredError, match="FULL_DOWNLOAD"):
            await service.coordinated_mutation(
                operation="test_write",
                idempotency_key="blocked-key",
                request={"value": 1},
                mutate=mutate,
            )
    assert mutated is False

    async with AnkiCollectionService(collection_path, max_page_size=100) as restarted:
        status = await restarted.status()
    assert status["pending_full_sync"] == "FULL_DOWNLOAD"
    assert status["ready"] is False
    assert status["readiness_reason"] == "full_sync_required"


@pytest.mark.anyio
async def test_explicit_backup_is_created_in_persistent_backup_directory(
    collection_path: str,
) -> None:
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        result = await service.create_backup()
    assert result["requested"] is True
    assert result["created"] is True
    assert result["reason"] is None
    backup_directory = Path(collection_path).parent / "backups"
    assert backup_directory.is_dir()
    assert any(backup_directory.iterdir())


@pytest.mark.anyio
async def test_explicit_backup_reports_when_none_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = str(tmp_path / "fresh.anki2")
    Collection(path).close()
    monkeypatch.setattr(Collection, "create_backup", lambda *args, **kwargs: False)

    async with AnkiCollectionService(path, max_page_size=100) as service:
        result = await service.create_backup()

    assert result["requested"] is True
    assert result["created"] is False
    assert result["path"] is None
    assert result["reason"] == "no_valid_backup_available"


@pytest.mark.anyio
async def test_controlled_bootstrap_downloads_only_into_an_empty_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = str(tmp_path / "fresh.anki2")
    Collection(path).close()
    login_calls: list[tuple[str, str, str | None]] = []
    full_sync_directions: list[bool] = []

    def login(_: Collection, username: str, password: str, endpoint: str | None) -> SyncAuth:
        login_calls.append((username, password, endpoint))
        return SyncAuth(hkey="bootstrap-key", endpoint=endpoint)

    monkeypatch.setattr(Collection, "sync_login", login)
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: SyncOutput(required=3, server_media_usn=9),
    )
    monkeypatch.setattr(Collection, "create_backup", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        Collection,
        "full_upload_or_download",
        lambda self, *, auth, server_usn, upload: full_sync_directions.append(upload),
    )

    async with AnkiCollectionService(path, max_page_size=100) as service:
        result = await service.bootstrap(
            "download_if_empty", "bootstrap-user", "bootstrap-password", "http://localhost:8080/"
        )
        status = await service.status()

    assert login_calls == [("bootstrap-user", "bootstrap-password", "http://localhost:8080/")]
    assert full_sync_directions == [False]
    assert result["bootstrapped"] is True
    assert result["direction"] == "download"
    assert status["ready"] is True


@pytest.mark.anyio
async def test_controlled_bootstrap_refuses_a_nonempty_local_collection(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    login_called = False

    def login(_: Collection, username: str, password: str, endpoint: str | None) -> SyncAuth:
        nonlocal login_called
        login_called = True
        return SyncAuth(hkey="unexpected", endpoint=endpoint)

    monkeypatch.setattr(Collection, "sync_login", login)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        with pytest.raises(ValueError, match="empty local collection"):
            await service.bootstrap(
                "download_if_empty", "bootstrap-user", "bootstrap-password", None
            )
    assert login_called is False


@pytest.mark.anyio
async def test_coordinator_fails_closed_when_receipt_persistence_fails_after_mutation(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_put_receipt = PersistentState.put_receipt

    def fail_committed_receipt(
        state: PersistentState,
        key: str,
        operation: str,
        request_hash: str,
        receipt: dict[str, object],
    ) -> None:
        if receipt.get("local_committed") is True:
            raise OSError("simulated crash window")
        original_put_receipt(state, key, operation, request_hash, receipt)

    monkeypatch.setattr(PersistentState, "put_receipt", fail_committed_receipt)
    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        with pytest.raises(OSError, match="crash window"):
            await service.coordinated_mutation(
                operation="anki_decks_create",
                idempotency_key="crash-window-key",
                request={"name": "Crash Window"},
                mutate=lambda adapter: adapter.create_deck("Crash Window"),
            )

    replayed = False

    def must_not_replay(_: object) -> dict[str, object]:
        nonlocal replayed
        replayed = True
        return {"unexpected": True}

    async with AnkiCollectionService(collection_path, max_page_size=100) as restarted:
        recovered = await restarted.coordinated_mutation(
            operation="anki_decks_create",
            idempotency_key="crash-window-key",
            request={"name": "Crash Window"},
            mutate=must_not_replay,
        )

    assert replayed is False
    assert recovered["state"] == "outcome_unknown"
    assert recovered["local_committed"] is None
    assert recovered["remote_synced"] is False
    assert recovered["retryable"] is False


@pytest.mark.anyio
async def test_full_download_marks_post_commit_receipt_as_discarded(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_results = iter([SyncOutput(required=0), SyncOutput(required=3, server_media_usn=8)])
    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(
            hkey="persistent-key", endpoint=endpoint
        ),
    )
    monkeypatch.setattr(
        Collection, "sync_collection", lambda self, auth, sync_media: next(sync_results)
    )
    monkeypatch.setattr(Collection, "create_backup", lambda *args, **kwargs: True)
    monkeypatch.setattr(Collection, "full_upload_or_download", lambda *args, **kwargs: None)

    async with AnkiCollectionService(
        collection_path, max_page_size=100, sync_on_write=True
    ) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        with pytest.raises(FullSyncRequiredError, match="FULL_DOWNLOAD"):
            await service.coordinated_mutation(
                operation="anki_decks_create",
                idempotency_key="discarded-key",
                request={"name": "Discarded"},
                mutate=lambda adapter: adapter.create_deck("Discarded"),
            )
        await service.full_sync(upload=False)
        replayed = await service.coordinated_mutation(
            operation="anki_decks_create",
            idempotency_key="discarded-key",
            request={"name": "Discarded"},
            mutate=lambda adapter: (_ for _ in ()).throw(AssertionError("mutation replayed")),
        )
        status = await service.status()

    assert replayed["state"] == "discarded_by_full_download"
    assert replayed["local_committed"] is False
    assert replayed["remote_synced"] is False
    assert replayed["result"] is None
    assert status["pending_mutations"] == 0


@pytest.mark.anyio
async def test_retryable_sync_failure_retains_authentication_across_restart(
    collection_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Collection,
        "sync_login",
        lambda self, username, password, endpoint: SyncAuth(
            hkey="persistent-key", endpoint=endpoint
        ),
    )
    monkeypatch.setattr(
        Collection,
        "sync_collection",
        lambda self, auth, sync_media: (_ for _ in ()).throw(
            NetworkError("sync failed", None, None, None)
        ),
    )

    async with AnkiCollectionService(collection_path, max_page_size=100) as service:
        await service.sync_login("user", "password", "https://sync.example.test/")
        with pytest.raises(NetworkError, match="sync failed"):
            await service.sync(sync_media=False)
        assert (await service.status())["authenticated"] is True

    async with AnkiCollectionService(collection_path, max_page_size=100) as restarted:
        status = await restarted.status()
    assert status["authenticated"] is True
