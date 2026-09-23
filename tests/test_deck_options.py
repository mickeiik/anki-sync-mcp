from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from anki.collection import Collection

from anki_mcp.collection import AnkiCollectionService


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
