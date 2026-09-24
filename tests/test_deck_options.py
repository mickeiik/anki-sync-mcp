from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from anki.collection import Collection
from anki.decks import DeckId

from anki_mcp.collection import AnkiCollectionService, ResourceLimitError


@pytest.fixture
def deck_options_collection(tmp_path: Path) -> Iterator[tuple[str, int, int]]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        parent_id = int(collection.decks.id("Options"))
        child_id = int(collection.decks.id("Options::Child"))
    finally:
        collection.close()
    yield path, parent_id, child_id


@pytest.fixture
def counted_deck_collection(tmp_path: Path) -> Iterator[tuple[str, int, int]]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        parent_id = int(collection.decks.id("Counted"))
        child_id = int(collection.decks.id("Counted::Child"))
        model = collection.models.current()

        parent_note = collection.new_note(model)
        parent_note["Front"] = "parent card"
        parent_note["Back"] = "answer"
        collection.add_note(parent_note, parent_id)

        child_note = collection.new_note(model)
        child_note["Front"] = "child card"
        child_note["Back"] = "answer"
        collection.add_note(child_note, child_id)
    finally:
        collection.close()
    yield path, parent_id, child_id


@pytest.mark.anyio
async def test_deck_options_counts_reflect_due_tree(
    counted_deck_collection: tuple[str, int, int],
) -> None:
    path, parent_id, child_id = counted_deck_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        parent = await service.get_deck_options(parent_id, include_sections=("counts",))
        child = await service.get_deck_options(child_id, include_sections=("counts",))

    parent_counts = parent["sections"]["counts"]
    child_counts = child["sections"]["counts"]
    assert parent_counts["new"] == 2  # includes the child deck's due cards
    assert parent_counts["total_in_deck"] == 1
    assert parent_counts["total_including_children"] == 2
    assert child_counts["new"] == 1
    assert child_counts["total_in_deck"] == 1


@pytest.mark.anyio
async def test_deck_options_counts_empty_default_deck_absent_from_due_tree(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        collection.decks.id("Other")  # leaves the Default deck empty and childless
        assert collection.decks.find_deck_in_tree(collection.sched.deck_due_tree(), 1) is None
    finally:
        collection.close()

    async with AnkiCollectionService(path, max_page_size=100) as service:
        result = await service.get_deck_options(1, include_sections=("counts",))

    assert result["sections"]["counts"] == {
        "new": 0,
        "review": 0,
        "learning": 0,
        "new_uncapped": 0,
        "review_uncapped": 0,
        "total_in_deck": 0,
        "total_including_children": 0,
    }


@pytest.mark.anyio
async def test_deck_options_are_compact_by_default_and_expand_requested_sections(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, parent_id, child_id = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        compact = await service.get_deck_options(child_id)
        expanded = await service.get_deck_options(
            child_id, include_sections=("counts", "parents", "global_settings")
        )

    assert compact == {
        "deck_id": child_id,
        "name": "Options::Child",
        "preset": {
            "id": 1,
            "name": "Default",
            "new_cards_per_day": 20,
            "reviews_per_day": 200,
            "max_answer_seconds": 60,
            "desired_retention": 0.9,
        },
        "limits": {
            "this_deck": {
                "new_cards_per_day": None,
                "reviews_per_day": None,
                "desired_retention": None,
            },
            "today": {
                "new_cards_per_day": None,
                "reviews_per_day": None,
            },
        },
        "effective_limits": {
            "new_cards_per_day": {
                "value": 20,
                "source": "preset",
                "source_deck_id": child_id,
                "inherited": False,
            },
            "reviews_per_day": {
                "value": 200,
                "source": "preset",
                "source_deck_id": child_id,
                "inherited": False,
            },
            "desired_retention": {
                "value": 0.9,
                "source": "preset",
                "source_deck_id": child_id,
                "inherited": False,
            },
        },
        "apply_all_parent_limits": False,
    }
    assert "sections" not in compact
    assert expanded["sections"]["parents"]["deck_ids"] == [parent_id]
    assert expanded["sections"]["parents"]["preset_ids"] == [1]
    assert expanded["sections"]["parents"]["limits_applied"] is False
    assert expanded["sections"]["counts"] == {
        "new": 0,
        "review": 0,
        "learning": 0,
        "new_uncapped": 0,
        "review_uncapped": 0,
        "total_in_deck": 0,
        "total_including_children": 0,
    }
    assert expanded["sections"]["global_settings"] == {
        "new_cards_ignore_review_limit": False,
        "fsrs": False,
    }


@pytest.mark.anyio
async def test_preset_get_is_compact_and_partial_update_preserves_unmentioned_options(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, _ = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        compact = await service.get_deck_preset(1)
        updated = await service.update_deck_preset(
            1,
            name="Focused",
            options={
                "learn_steps": [2.0, 15.0],
                "new_card_gather_priority": "NEW_CARD_GATHER_PRIORITY_RANDOM_CARDS",
                "bury_new": True,
                "seconds_to_show_question": 4.5,
                "fsrs_params_6": [
                    0.212,
                    1.2931,
                    2.3065,
                    8.2956,
                    6.4133,
                    0.8334,
                    3.0194,
                    0.001,
                    1.8722,
                    0.1666,
                    0.796,
                    1.4835,
                    0.0614,
                    0.2629,
                    1.6483,
                    0.6014,
                    1.8729,
                    0.5425,
                    0.0912,
                    0.0658,
                    0.1542,
                ],
            },
        )
        expanded = await service.get_deck_preset(
            1,
            include_sections=(
                "learning",
                "new_cards",
                "reviews",
                "lapses",
                "burying",
                "display_audio",
                "fsrs",
                "easy_days",
            ),
        )

    assert compact == {
        "id": 1,
        "name": "Default",
        "use_count": 3,
        "new_cards_per_day": 20,
        "reviews_per_day": 200,
        "max_answer_seconds": 60,
        "desired_retention": 0.9,
    }
    assert "sections" not in compact
    assert updated == {"id": 1, "updated": True, "changed_fields": 6, "affected_decks": 3}
    assert expanded["name"] == "Focused"
    assert expanded["reviews_per_day"] == 200
    assert expanded["sections"]["learning"]["learn_steps"] == [2.0, 15.0]
    assert (
        expanded["sections"]["new_cards"]["new_card_gather_priority"]
        == "NEW_CARD_GATHER_PRIORITY_RANDOM_CARDS"
    )
    assert expanded["sections"]["burying"]["bury_new"] is True
    assert expanded["sections"]["display_audio"]["seconds_to_show_question"] == 4.5
    assert expanded["sections"]["fsrs"]["fsrs_params_6"][:2] == [0.212, 1.2931]
    assert set(expanded["sections"]) == {
        "learning",
        "new_cards",
        "reviews",
        "lapses",
        "burying",
        "display_audio",
        "fsrs",
        "easy_days",
    }


@pytest.mark.anyio
async def test_scoped_limits_can_be_set_and_cleared_without_changing_the_preset(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, child_id = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        scheduler_settings = await service.update_deck_scheduler_settings(
            apply_all_parent_limits=True,
            new_cards_ignore_review_limit=True,
            fsrs_enabled=None,
        )
        this_deck = await service.update_deck_limits(
            child_id,
            scope="this_deck",
            values={
                "new_cards_per_day": 30,
                "reviews_per_day": 300,
                "desired_retention": 0.92,
            },
            clear_fields=(),
        )
        today = await service.update_deck_limits(
            child_id,
            scope="today",
            values={"new_cards_per_day": 7, "reviews_per_day": 70},
            clear_fields=(),
        )
        configured = await service.get_deck_options(child_id, include_sections=("global_settings",))
        cleared = await service.update_deck_limits(
            child_id,
            scope="today",
            values={},
            clear_fields=("new_cards_per_day", "reviews_per_day"),
        )
        cleared_this_deck = await service.update_deck_limits(
            child_id,
            scope="this_deck",
            values={},
            clear_fields=(
                "new_cards_per_day",
                "reviews_per_day",
                "desired_retention",
            ),
        )
        final = await service.get_deck_options(child_id)

    assert scheduler_settings == {
        "scope": "collection",
        "updated": True,
        "apply_all_parent_limits": True,
        "new_cards_ignore_review_limit": True,
        "fsrs_enabled": False,
    }
    assert this_deck == {"deck_id": child_id, "scope": "this_deck", "updated": True}
    assert today == {"deck_id": child_id, "scope": "today", "updated": True}
    assert configured["preset"]["new_cards_per_day"] == 20
    assert configured["limits"] == {
        "this_deck": {
            "new_cards_per_day": 30,
            "reviews_per_day": 300,
            "desired_retention": 0.92,
        },
        "today": {"new_cards_per_day": 7, "reviews_per_day": 70},
    }
    assert configured["effective_limits"] == {
        "new_cards_per_day": {
            "value": 7,
            "source": "today",
            "source_deck_id": child_id,
            "inherited": False,
        },
        "reviews_per_day": {
            "value": 70,
            "source": "today",
            "source_deck_id": child_id,
            "inherited": False,
        },
        "desired_retention": {
            "value": 0.92,
            "source": "this_deck",
            "source_deck_id": child_id,
            "inherited": False,
        },
    }
    assert configured["apply_all_parent_limits"] is True
    assert configured["sections"]["global_settings"]["new_cards_ignore_review_limit"] is True
    assert cleared == {"deck_id": child_id, "scope": "today", "updated": True}
    assert cleared_this_deck == {
        "deck_id": child_id,
        "scope": "this_deck",
        "updated": True,
    }
    assert final["limits"]["today"] == {
        "new_cards_per_day": None,
        "reviews_per_day": None,
    }
    assert final["effective_limits"] == {
        "new_cards_per_day": {
            "value": 20,
            "source": "preset",
            "source_deck_id": child_id,
            "inherited": False,
        },
        "reviews_per_day": {
            "value": 200,
            "source": "preset",
            "source_deck_id": child_id,
            "inherited": False,
        },
        "desired_retention": {
            "value": 0.9,
            "source": "preset",
            "source_deck_id": child_id,
            "inherited": False,
        },
    }


@pytest.mark.anyio
async def test_effective_limits_include_enabled_parent_constraints(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, parent_id, child_id = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_limits(
            parent_id,
            scope="this_deck",
            values={"new_cards_per_day": 5, "reviews_per_day": 50},
            clear_fields=(),
        )
        await service.update_deck_scheduler_settings(
            apply_all_parent_limits=True,
            new_cards_ignore_review_limit=None,
            fsrs_enabled=None,
        )
        options = await service.get_deck_options(child_id, include_sections=("parents",))

    assert options["effective_limits"]["new_cards_per_day"] == {
        "value": 5,
        "source": "this_deck",
        "source_deck_id": parent_id,
        "inherited": True,
    }
    assert options["effective_limits"]["reviews_per_day"] == {
        "value": 50,
        "source": "this_deck",
        "source_deck_id": parent_id,
        "inherited": True,
    }
    assert options["sections"]["parents"]["limits_applied"] is True
    assert options["sections"]["parents"]["limit_chain"][0]["deck_id"] == parent_id


@pytest.mark.anyio
async def test_presets_can_be_listed_created_and_assigned(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, child_id = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        created = await service.create_deck_preset("Cloned", clone_from_config_id=1)
        listed = await service.list_deck_presets(offset=0, limit=100)
        assigned = await service.assign_deck_preset(child_id, created["id"])
        options = await service.get_deck_options(child_id)

    assert created["created"] is True
    assert listed["total"] == 2
    assert [item["name"] for item in listed["items"]] == ["Cloned", "Default"]
    assert assigned == {
        "deck_id": child_id,
        "config_id": created["id"],
        "updated": True,
    }
    assert options["preset"]["id"] == created["id"]


@pytest.mark.anyio
async def test_invalid_backend_preset_patch_is_reported_as_an_argument_error(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, _ = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(ValueError, match="invalid deck option update"):
            await service.update_deck_preset(
                1,
                name=None,
                options={"fsrs_params_6": [0.1, 0.2]},
            )
        unchanged = await service.get_deck_preset(1, include_sections=("fsrs",))

    assert unchanged["sections"]["fsrs"]["fsrs_params_6"] == []


@pytest.mark.anyio
async def test_deck_preset_delete_preview_and_apply_reassign_decks_to_default(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, child_id = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        created = await service.create_deck_preset("Doomed", clone_from_config_id=1)
        await service.assign_deck_preset(child_id, created["id"])
        scm_before = await service.executor.run(
            lambda adapter: int(adapter.collection.db.scalar("select scm from col"))
        )
        preview = await service.preview_deck_preset_delete(created["id"])
        applied = await service.delete_deck_preset(created["id"])
        reassigned = await service.get_deck_options(child_id)
        with pytest.raises(LookupError):
            await service.get_deck_preset(created["id"])
        with pytest.raises(LookupError):
            await service.delete_deck_preset(created["id"])
        scm_after = await service.executor.run(
            lambda adapter: int(adapter.collection.db.scalar("select scm from col"))
        )

    assert preview["id"] == created["id"]
    assert preview["name"] == "Doomed"
    assert preview["affected_decks"] == [{"id": child_id, "name": "Options::Child"}]
    assert preview["affected_decks_total"] == 1
    assert preview["affected_decks_truncated"] is False
    assert preview["fallback_config_id"] == 1
    assert preview["fallback_config_name"] == "Default"
    assert preview["backup_required"] is True
    assert preview["full_sync_required"] is True
    assert len(preview["state_fingerprint"]) == 64
    assert applied == {
        "id": created["id"],
        "deleted": True,
        "decks_reassigned": 1,
        "deck_ids": [child_id],
        "fallback_config_id": 1,
        "full_sync_required": True,
    }
    assert reassigned["preset"]["id"] == 1
    assert reassigned["preset"]["name"] == "Default"
    assert scm_after != scm_before  # delete forces a schema change -> full sync


@pytest.mark.anyio
async def test_deck_preset_delete_fingerprint_tracks_affected_decks(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, child_id = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        created = await service.create_deck_preset("Doomed", clone_from_config_id=1)
        unassigned = await service.preview_deck_preset_delete(created["id"])
        await service.assign_deck_preset(child_id, created["id"])
        assigned = await service.preview_deck_preset_delete(created["id"])

    assert unassigned["affected_decks_total"] == 0
    assert assigned["affected_decks_total"] == 1
    assert unassigned["state_fingerprint"] != assigned["state_fingerprint"]


@pytest.mark.anyio
async def test_deck_preset_delete_refuses_default_and_unknown_ids(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, _ = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(ValueError, match="default deck preset cannot be deleted"):
            await service.preview_deck_preset_delete(1)
        with pytest.raises(ValueError, match="default deck preset cannot be deleted"):
            await service.delete_deck_preset(1)
        with pytest.raises(LookupError):
            await service.preview_deck_preset_delete(4242)
        with pytest.raises(LookupError):
            await service.delete_deck_preset(4242)


@pytest.mark.anyio
async def test_unused_deck_preset_deletes_without_reassignments(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, _ = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        created = await service.create_deck_preset("Unused", clone_from_config_id=1)
        preview = await service.preview_deck_preset_delete(created["id"])
        applied = await service.delete_deck_preset(created["id"])

    assert preview["affected_decks"] == []
    assert applied["decks_reassigned"] == 0
    assert applied["deck_ids"] == []


@pytest.mark.anyio
async def test_update_deck_limits_keeps_target_preset_with_a_later_sorted_preset_present(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, child_id = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.create_deck_preset("Zeta", clone_from_config_id=1)
        await service.update_deck_limits(
            child_id,
            scope="this_deck",
            values={"new_cards_per_day": 5},
            clear_fields=(),
        )
        preset_id = (await service.get_deck_options(child_id))["preset"]["id"]
        applied_limit = (await service.get_deck_options(child_id))["limits"]["this_deck"][
            "new_cards_per_day"
        ]

    assert preset_id == 1
    assert applied_limit == 5


@pytest.mark.anyio
async def test_update_deck_preset_does_not_move_other_decks(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, parent_id, child_id = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        zeta = await service.create_deck_preset("Zeta", clone_from_config_id=1)
        await service.assign_deck_preset(child_id, zeta["id"])
        await service.update_deck_preset(
            1, name=None, options={"desired_retention": 0.85}
        )
        deck_one = (await service.get_deck_options(1))["preset"]["id"]
        parent = (await service.get_deck_options(parent_id))["preset"]["id"]
        child = (await service.get_deck_options(child_id))["preset"]["id"]
        updated = await service.get_deck_preset(1)

    assert deck_one == 1
    assert parent == 1
    assert child == zeta["id"]
    assert updated["desired_retention"] == 0.85


@pytest.mark.anyio
async def test_update_deck_scheduler_settings_keeps_deck_presets(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, _ = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.create_deck_preset("Zeta", clone_from_config_id=1)
        await service.update_deck_scheduler_settings(
            apply_all_parent_limits=True,
            new_cards_ignore_review_limit=None,
            fsrs_enabled=None,
        )
        deck_one = (await service.get_deck_options(1))["preset"]["id"]

    assert deck_one == 1


@pytest.mark.anyio
async def test_delete_unused_preset_keeps_other_decks_presets(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, parent_id, _ = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        custom = await service.create_deck_preset("Custom", clone_from_config_id=1)
        await service.assign_deck_preset(1, custom["id"])
        doomed = await service.create_deck_preset("Doomed", clone_from_config_id=1)
        await service.delete_deck_preset(doomed["id"])
        deck_one = (await service.get_deck_options(1))["preset"]["id"]
        parent = (await service.get_deck_options(parent_id))["preset"]["id"]

    assert deck_one == custom["id"]
    assert parent == 1


@pytest.mark.anyio
async def test_delete_shared_preset_reassigns_only_its_decks(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, parent_id, child_id = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        keeper = await service.create_deck_preset("Keeper", clone_from_config_id=1)
        await service.assign_deck_preset(1, keeper["id"])
        shared = await service.create_deck_preset("Shared", clone_from_config_id=1)
        await service.assign_deck_preset(parent_id, shared["id"])
        await service.assign_deck_preset(child_id, shared["id"])
        applied = await service.delete_deck_preset(shared["id"])
        deck_one = (await service.get_deck_options(1))["preset"]["id"]
        parent = (await service.get_deck_options(parent_id))["preset"]["id"]
        child = (await service.get_deck_options(child_id))["preset"]["id"]

    assert applied["decks_reassigned"] == 2
    assert sorted(applied["deck_ids"]) == sorted([parent_id, child_id])
    assert deck_one == keeper["id"]
    assert parent == 1
    assert child == 1


@pytest.mark.anyio
async def test_update_deck_preset_succeeds_when_decks_exceed_the_scan_bound(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, child_id = deck_options_collection

    async with AnkiCollectionService(
        path, max_page_size=100, max_search_scan=2
    ) as service:
        zeta = await service.create_deck_preset("Zeta", clone_from_config_id=1)
        await service.assign_deck_preset(child_id, zeta["id"])
        result = await service.update_deck_preset(
            1, name=None, options={"desired_retention": 0.85}
        )
        deck_one = (await service.get_deck_options(1))["preset"]["id"]
        child = (await service.get_deck_options(child_id))["preset"]["id"]

    assert result["updated"] is True
    assert deck_one == 1
    assert child == zeta["id"]


@pytest.mark.anyio
async def test_deck_preset_delete_scan_bound_raises_resource_limit_error(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, _ = deck_options_collection

    async with AnkiCollectionService(
        path, max_page_size=100, max_search_scan=2
    ) as service:
        created = await service.create_deck_preset("Doomed", clone_from_config_id=1)
        with pytest.raises(ResourceLimitError, match="MCP_MAX_SEARCH_SCAN"):
            await service.preview_deck_preset_delete(created["id"])
        with pytest.raises(ResourceLimitError, match="MCP_MAX_SEARCH_SCAN"):
            await service.delete_deck_preset(created["id"])


@pytest.mark.anyio
async def test_deck_preset_delete_excludes_filtered_decks(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, child_id = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        created = await service.create_deck_preset("Shared", clone_from_config_id=1)
        await service.assign_deck_preset(child_id, created["id"])
        await service.executor.run(
            lambda adapter: adapter.collection.decks.new_filtered("Cram")
        )
        preview = await service.preview_deck_preset_delete(created["id"])
        applied = await service.delete_deck_preset(created["id"])

    assert preview["affected_decks"] == [{"id": child_id, "name": "Options::Child"}]
    assert applied["decks_reassigned"] == 1
    assert applied["deck_ids"] == [child_id]


@pytest.mark.anyio
async def test_update_deck_config_state_rejects_unknown_selected_config(
    deck_options_collection: tuple[str, int, int],
) -> None:
    path, _, _ = deck_options_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(ValueError, match="is not part of the update"):
            await service.executor.run(
                lambda adapter: adapter._update_deck_config_state(
                    1,
                    adapter.collection.decks.get_deck_configs_for_update(
                        cast("DeckId", 1)
                    ),
                    selected_config_id=999999,
                )
            )
