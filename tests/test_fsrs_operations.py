from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator
from pathlib import Path
from zipfile import ZipFile

import pytest
from anki import buildinfo, scheduler_pb2
from anki._backend_generated import RustBackendGenerated
from anki.collection import Collection
from anki.config_pb2 import ConfigKey
from anki.decks import DeckManager

from anki_mcp.collection import AnkiCollectionService, ResourceLimitError


@pytest.fixture
def fsrs_collection(tmp_path: Path) -> Iterator[tuple[str, int]]:
    path = str(tmp_path / "collection.anki2")
    collection = Collection(path)
    try:
        deck_id = int(collection.decks.id("FSRS"))
        note = collection.new_note(collection.models.current())
        note["Front"] = "FSRS card"
        note["Back"] = "answer"
        collection.add_note(note, deck_id)
        card_id = int(collection.find_cards(f"nid:{int(note.id)}")[0])
        card = collection.get_card(card_id)
        card.type = 2
        card.queue = 2
        card.due = collection.sched.today + 100
        card.ivl = 100
        card.reps = 3
        card.factor = 2500
        collection.update_card(card)
        now_ms = int(time.time() * 1000)
        day_ms = 86_400_000
        collection.db.executemany(
            """
            insert into revlog(id, cid, usn, ease, ivl, lastIvl, factor, time, type)
            values (?, ?, -1, ?, ?, ?, 2500, 1000, 1)
            """,
            [
                (now_ms - 100 * day_ms, card_id, 3, 1, 0),
                (now_ms - 90 * day_ms, card_id, 3, 10, 1),
                (now_ms - 50 * day_ms, card_id, 3, 50, 10),
            ],
        )
        new_note = collection.new_note(collection.models.current())
        new_note["Front"] = "New FSRS card"
        new_note["Back"] = "answer"
        collection.add_note(new_note, deck_id)
    finally:
        collection.close()
    yield path, deck_id


def _add_review_state_card(
    path: str, deck_id: int, front: str, *, with_revlog: bool = False
) -> int:
    collection = Collection(path)
    try:
        note = collection.new_note(collection.models.current())
        note["Front"] = front
        note["Back"] = "answer"
        collection.add_note(note, deck_id)
        card_id = int(collection.find_cards(f"nid:{int(note.id)}")[0])
        card = collection.get_card(card_id)
        card.type = 2
        card.queue = 2
        card.due = collection.sched.today + 10
        card.ivl = 10
        card.reps = 1
        collection.update_card(card)
        if with_revlog:
            db = collection.db
            assert db is not None
            db.execute(
                """
                insert into revlog(id, cid, usn, ease, ivl, lastIvl, factor, time, type)
                values (?, ?, -1, 3, 10, 1, 2500, 1000, 1)
                """,
                int(db.scalar("select coalesce(max(id), 0) + 1 from revlog")),
                card_id,
            )
        return card_id
    finally:
        collection.close()


@pytest.mark.anyio
async def test_scheduler_settings_can_enable_fsrs(
    fsrs_collection: tuple[str, int],
) -> None:
    path, deck_id = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        updated = await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        options = await service.get_deck_options(deck_id, ("global_settings",))

    assert updated == {
        "scope": "collection",
        "updated": True,
        "apply_all_parent_limits": False,
        "new_cards_ignore_review_limit": False,
        "fsrs_enabled": True,
    }
    assert options["sections"]["global_settings"]["fsrs"] is True


@pytest.mark.anyio
async def test_fsrs_simulator_uses_preset_defaults_and_is_compact(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        compact = await service.simulate_fsrs(
            config_id=1,
            mode="review",
            deck_size=1_000,
            days_to_simulate=30,
            desired_retention=None,
            search=None,
            include_daily=False,
        )
        optimal = await service.simulate_fsrs(
            config_id=1,
            mode="optimal_retention",
            deck_size=1_000,
            days_to_simulate=365,
            desired_retention=None,
            search=None,
            include_daily=False,
        )

    assert compact["mode"] == "review"
    assert compact["config_id"] == 1
    assert compact["days_to_simulate"] == 30
    assert compact["summary"]["total_reviews"] >= 0
    assert compact["summary"]["total_new_cards"] >= 0
    assert "daily" not in compact
    assert optimal["mode"] == "optimal_retention"
    assert 0.7 <= optimal["optimal_retention"] <= 0.99


@pytest.mark.anyio
async def test_fsrs_optimizer_applies_native_parameters(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, deck_id = fsrs_collection
    optimized = [0.2 + index / 100 for index in range(21)]

    def compute_fsrs_params(self, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["search"] == 'preset:"Default" -is:suspended'
        assert kwargs["health_check"] is True
        return scheduler_pb2.ComputeFsrsParamsResponse(
            params=optimized,
            fsrs_items=432,
            health_check_passed=True,
        )

    monkeypatch.setattr(RustBackendGenerated, "compute_fsrs_params", compute_fsrs_params)

    async with AnkiCollectionService(path, max_page_size=100) as service:
        result = await service.optimize_fsrs(1, search=None, health_check=True)
        preset = await service.get_deck_preset(1, ("fsrs",))

    assert result == {
        "config_id": 1,
        "optimized": True,
        "search": 'preset:"Default" -is:suspended',
        "training_items": 432,
        "health_check_requested": True,
        "health_check_passed": True,
        "previous_parameters": [],
        "parameters": pytest.approx(optimized),
        "affected_decks": 2,
        "affected_deck_ids": [1, deck_id],
        "affected_decks_detail": [
            {"id": 1, "name": "Default"},
            {"id": deck_id, "name": "FSRS"},
        ],
    }
    assert preset["sections"]["fsrs"]["fsrs_params_6"] == pytest.approx(optimized)


@pytest.mark.anyio
async def test_fsrs_optimizer_supports_a_non_default_shared_preset(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, deck_id = fsrs_collection
    candidate = [0.2 + index / 100 for index in range(21)]

    def compute_fsrs_params(self, **kwargs):  # type: ignore[no-untyped-def]
        assert 'preset:"Custom FSRS"' in kwargs["search"]
        return scheduler_pb2.ComputeFsrsParamsResponse(
            params=candidate,
            fsrs_items=10,
            health_check_passed=True,
        )

    monkeypatch.setattr(RustBackendGenerated, "compute_fsrs_params", compute_fsrs_params)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        created = await service.create_deck_preset("Custom FSRS", 1)
        config_id = created["id"]
        await service.assign_deck_preset(deck_id, config_id)
        preview = await service.preview_fsrs_optimization(config_id, None, True)

    assert config_id != 1
    assert preview["affected_decks"] == 1
    assert preview["affected_deck_ids"] == [deck_id]
    assert preview["affected_decks_detail"] == [{"id": deck_id, "name": "FSRS"}]
    assert preview["candidate_parameters"] == pytest.approx(candidate)


@pytest.mark.anyio
async def test_fsrs_optimizer_preview_rejects_failed_health_check_without_mutation(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    candidate = [0.2 + index / 100 for index in range(21)]

    def compute_fsrs_params(self, **kwargs):  # type: ignore[no-untyped-def]
        return scheduler_pb2.ComputeFsrsParamsResponse(
            params=candidate,
            fsrs_items=432,
            health_check_passed=False,
        )

    monkeypatch.setattr(RustBackendGenerated, "compute_fsrs_params", compute_fsrs_params)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        before = await service.get_deck_preset(1, ("fsrs",))
        with pytest.raises(ValueError, match="health check failed"):
            await service.preview_fsrs_optimization(1, search=None, health_check=True)
        after = await service.get_deck_preset(1, ("fsrs",))

    assert after == before


@pytest.mark.anyio
async def test_fsrs_optimizer_health_check_can_be_explicitly_bypassed(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    candidate = [0.2 + index / 100 for index in range(21)]

    def compute_fsrs_params(self, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["health_check"] is False
        return scheduler_pb2.ComputeFsrsParamsResponse(
            params=candidate,
            fsrs_items=432,
            health_check_passed=False,
        )

    monkeypatch.setattr(RustBackendGenerated, "compute_fsrs_params", compute_fsrs_params)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        preview = await service.preview_fsrs_optimization(1, search=None, health_check=False)

    assert preview["health_check_requested"] is False
    assert preview["health_check_passed"] is False
    assert preview["candidate_parameters"] == pytest.approx(candidate)


@pytest.mark.anyio
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
async def test_fsrs_reschedule_rejects_non_finite_parameters_before_impact(
    fsrs_collection: tuple[str, int], value: float
) -> None:
    path, _ = fsrs_collection
    parameters = [0.2 + index / 100 for index in range(21)]
    parameters[7] = value

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        with pytest.raises(ValueError, match="parameter 7 must be finite"):
            await service.preview_fsrs_reschedule(1, 0.91, parameters)


@pytest.mark.anyio
async def test_fsrs_reschedule_rejects_native_invalid_finite_parameters_before_impact(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection
    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        with pytest.raises(ValueError, match="invalid FSRS parameters"):
            await service.preview_fsrs_reschedule(1, 0.91, [-1.0] * 21)


@pytest.mark.anyio
async def test_fsrs_simulation_covers_workload_and_daily_output_contracts(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection
    async with AnkiCollectionService(path, max_page_size=100) as service:
        workload = await service.simulate_fsrs(1, "workload", 100, 30, None, None, False)
        review = await service.simulate_fsrs(1, "review", 100, 30, None, None, True)

    assert workload["mode"] == "workload"
    assert set(workload["retention"]) == {
        "cost_seconds",
        "memorized_cards",
        "review_count",
    }
    assert review["days_to_simulate"] == 30
    assert len(review["daily"]) == 30
    assert set(review["daily"][0]) == {
        "day",
        "reviews",
        "new_cards",
        "time_seconds",
        "knowledge_acquisition",
    }


@pytest.mark.anyio
async def test_fsrs_optimizer_rejects_insufficient_review_history(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(ValueError, match="not enough review history"):
            await service.optimize_fsrs(1, search=None, health_check=True)


@pytest.mark.anyio
async def test_fsrs_operations_translate_invalid_searches(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(ValueError, match="invalid FSRS search"):
            await service.optimize_fsrs(1, search="(", health_check=True)
        with pytest.raises(ValueError, match="invalid FSRS search"):
            await service.simulate_fsrs(1, "workload", 100, 30, None, "(", False)


@pytest.mark.anyio
async def test_fsrs_operations_bound_the_selected_search_not_the_collection(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    optimized = [0.2 + index / 100 for index in range(21)]
    _add_review_state_card(path, 1, "Unselected default card")

    def compute_fsrs_params(self, **kwargs):  # type: ignore[no-untyped-def]
        return scheduler_pb2.ComputeFsrsParamsResponse(
            params=optimized,
            fsrs_items=1,
            health_check_passed=True,
        )

    monkeypatch.setattr(RustBackendGenerated, "compute_fsrs_params", compute_fsrs_params)

    async with AnkiCollectionService(
        path, max_page_size=100, max_search_scan=2
    ) as service:
        optimized_result = await service.optimize_fsrs(
            1, search="deck:FSRS -is:new", health_check=True
        )
        simulated = await service.simulate_fsrs(
            1, "review", 100, 30, None, "deck:FSRS -is:new", False
        )

    assert optimized_result["training_items"] == 1
    assert simulated["summary"]["total_reviews"] >= 0


@pytest.mark.anyio
async def test_fsrs_selected_search_stops_after_limit_plus_one(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    original = Collection.find_cards
    observed_orders: list[object] = []

    def bounded_find_cards(self, query, order=False, reverse=False):  # type: ignore[no-untyped-def]
        observed_orders.append(order)
        return original(self, query, order=order, reverse=reverse)

    monkeypatch.setattr(Collection, "find_cards", bounded_find_cards)
    async with AnkiCollectionService(
        path, max_page_size=100, max_search_scan=1
    ) as service:
        with pytest.raises(ResourceLimitError, match="observed_at_least=2"):
            await service.preview_fsrs_optimization(1, search="deck:FSRS", health_check=True)

    assert observed_orders == ["c.id asc limit 2"]


@pytest.mark.anyio
async def test_fsrs_reschedule_previews_and_applies_changed_retention(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        card_id = (await service.search_cards("deck:FSRS -is:new", 0, 1))["items"][0]["id"]
        before = await service.get_card(card_id)
        preview = await service.preview_fsrs_reschedule(1, 0.91, None)
        result = await service.reschedule_fsrs(1, 0.91, None)
        after = await service.get_card(card_id)
        preset = await service.get_deck_preset(1)

    assert preview["config_id"] == 1
    assert preview["cards"] == 1
    assert preview["decks"] == 2
    assert len(preview["state_fingerprint"]) == 64
    assert result == {
        "config_id": 1,
        "rescheduled": True,
        "cards": 1,
        "decks": 2,
        "desired_retention": 0.91,
        "parameters_changed": False,
    }
    assert after["scheduling"]["due"] != before["scheduling"]["due"]
    assert preset["desired_retention"] == pytest.approx(0.91)


@pytest.mark.anyio
async def test_fsrs_reschedule_fingerprint_tracks_card_eligibility(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        review_card = (await service.search_cards("deck:FSRS -is:new", 0, 1))["items"][0]
        before = await service.preview_fsrs_reschedule(1, 0.91, None)
        await service.suspend_cards([review_card["id"]])
        after = await service.preview_fsrs_reschedule(1, 0.91, None)

    assert before["cards"] == 1
    assert after["cards"] == 0
    assert after["state_fingerprint"] != before["state_fingerprint"]


@pytest.mark.anyio
async def test_fsrs_reschedule_fingerprint_tracks_per_deck_retention_override(
    fsrs_collection: tuple[str, int],
) -> None:
    path, deck_id = fsrs_collection
    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        before = await service.preview_fsrs_reschedule(1, 0.91, None)

        await service.update_deck_limits(
            deck_id, "this_deck", {"desired_retention": 0.88}, ()
        )
        after = await service.preview_fsrs_reschedule(1, 0.91, None)

    assert after["state_fingerprint"] != before["state_fingerprint"]


@pytest.mark.anyio
async def test_fsrs_reschedule_honors_per_deck_retention_override(
    fsrs_collection: tuple[str, int],
) -> None:
    path, deck_id = fsrs_collection
    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        card_id = (await service.search_cards("deck:FSRS -is:new", 0, 1))["items"][0]["id"]
        before = await service.get_card(card_id)
        await service.update_deck_limits(
            deck_id, "this_deck", {"desired_retention": 0.88}, ()
        )
        preview = await service.preview_fsrs_reschedule(1, 0.91, None)
        result = await service.reschedule_fsrs(1, 0.91, None)
        after = await service.get_card(card_id)
        options = await service.get_deck_options(deck_id)

    assert preview["cards"] == 1
    assert result["rescheduled"] is True
    assert after["scheduling"]["due"] == before["scheduling"]["due"]
    assert options["limits"]["this_deck"]["desired_retention"] == pytest.approx(0.88)


@pytest.mark.anyio
async def test_fsrs_reschedule_fingerprint_tracks_ordered_review_history(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        before = await service.preview_fsrs_reschedule(1, 0.91, None)

        def swap_review_payloads(adapter):  # type: ignore[no-untyped-def]
            db = adapter.collection.db
            assert db is not None
            rows = db.all(
                "select id, ease, ivl, lastIvl, factor, time, type "
                "from revlog order by id limit 2"
            )
            assert len(rows) == 2
            first, second = rows
            db.execute(
                "update revlog set ease=?, ivl=?, lastIvl=?, factor=?, time=?, type=? "
                "where id=?",
                *second[1:],
                first[0],
            )
            db.execute(
                "update revlog set ease=?, ivl=?, lastIvl=?, factor=?, time=?, type=? "
                "where id=?",
                *first[1:],
                second[0],
            )

        await service.executor.run(swap_review_payloads)
        after = await service.preview_fsrs_reschedule(1, 0.91, None)

    assert after["state_fingerprint"] != before["state_fingerprint"]


@pytest.mark.anyio
async def test_fsrs_reschedule_fingerprint_ignores_history_before_configured_date(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        await service.update_deck_preset(
            1, None, {"ignore_revlogs_before_date": "2026-06-01"}
        )
        before = await service.preview_fsrs_reschedule(1, 0.91, None)

        def change_ignored_review(adapter):  # type: ignore[no-untyped-def]
            db = adapter.collection.db
            assert db is not None
            db.execute(
                "update revlog set time = time + 1 "
                "where id = (select min(id) from revlog)"
            )

        await service.executor.run(change_ignored_review)
        after = await service.preview_fsrs_reschedule(1, 0.91, None)

    assert after["state_fingerprint"] == before["state_fingerprint"]


@pytest.mark.anyio
async def test_fsrs_reschedule_fingerprint_tracks_scheduler_day(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    day = [100]

    def sched_timing_today(self):  # type: ignore[no-untyped-def]
        return scheduler_pb2.SchedTimingTodayResponse(
            days_elapsed=day[0], next_day_at=1_000_000 + day[0] * 86_400
        )

    monkeypatch.setattr(RustBackendGenerated, "sched_timing_today", sched_timing_today)

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        before = await service.preview_fsrs_reschedule(1, 0.91, None)
        day[0] += 1
        after = await service.preview_fsrs_reschedule(1, 0.91, None)

    assert after["state_fingerprint"] != before["state_fingerprint"]


@pytest.mark.anyio
async def test_fsrs_reschedule_fingerprint_tracks_reset_markers(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        before = await service.preview_fsrs_reschedule(1, 0.91, None)

        def insert_reset_marker(adapter):  # type: ignore[no-untyped-def]
            db = adapter.collection.db
            assert db is not None
            card_id = int(db.scalar("select cid from revlog limit 1"))
            timestamps = [int(value) for value in db.list("select id from revlog order by id")]
            reset_at = timestamps[-2] + 1
            db.execute(
                """
                insert into revlog(id, cid, usn, ease, ivl, lastIvl, factor, time, type)
                values (?, ?, -1, 0, 0, 0, 0, 0, 4)
                """,
                reset_at,
                card_id,
            )

        await service.executor.run(insert_reset_marker)
        after = await service.preview_fsrs_reschedule(1, 0.91, None)

    assert after["state_fingerprint"] != before["state_fingerprint"]


@pytest.mark.anyio
async def test_fsrs_reschedule_fingerprint_tracks_load_balancer_candidate_state(
    fsrs_collection: tuple[str, int],
) -> None:
    path, deck_id = fsrs_collection
    excluded_card_id = _add_review_state_card(path, deck_id, "No FSRS history")

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        await service.executor.run(
            lambda adapter: adapter.collection.set_config_bool(
                ConfigKey.Bool.LOAD_BALANCER_ENABLED, True
            )
        )
        before = await service.preview_fsrs_reschedule(1, 0.91, None)

        def change_excluded_due(adapter):  # type: ignore[no-untyped-def]
            card = adapter.collection.get_card(excluded_card_id)
            card.due += 1
            adapter.collection.update_card(card)

        await service.executor.run(change_excluded_due)
        after = await service.preview_fsrs_reschedule(1, 0.91, None)

    assert before["cards"] == after["cards"] == 1
    assert after["state_fingerprint"] != before["state_fingerprint"]


@pytest.mark.anyio
async def test_fsrs_reschedule_fingerprint_tracks_load_balancer_reviewed_today(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        await service.executor.run(
            lambda adapter: adapter.collection.set_config_bool(
                ConfigKey.Bool.LOAD_BALANCER_ENABLED, True
            )
        )
        new_card_id = (await service.search_cards("deck:FSRS is:new", 0, 1))["items"][0]["id"]
        before = await service.preview_fsrs_reschedule(1, 0.91, None)

        def add_review_today(adapter):  # type: ignore[no-untyped-def]
            db = adapter.collection.db
            assert db is not None
            db.execute(
                """
                insert into revlog(id, cid, usn, ease, ivl, lastIvl, factor, time, type)
                values (?, ?, -1, 3, 1, 0, 2500, 1000, 1)
                """,
                int(time.time() * 1000),
                new_card_id,
            )

        await service.executor.run(add_review_today)
        after = await service.preview_fsrs_reschedule(1, 0.91, None)

    assert before["cards"] == after["cards"] == 1
    assert after["state_fingerprint"] != before["state_fingerprint"]


@pytest.mark.anyio
async def test_fsrs_reschedule_deck_discovery_avoids_unbounded_legacy_load(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection

    def reject_unbounded_load(self):  # type: ignore[no-untyped-def]
        raise AssertionError("DeckManager.all() must not be used")

    monkeypatch.setattr(DeckManager, "all", reject_unbounded_load)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        preview = await service.preview_fsrs_reschedule(1, 0.91, None)

    assert preview["decks"] == 2


@pytest.mark.anyio
async def test_fsrs_reschedule_candidate_discovery_accepts_exact_limit_across_decks(
    fsrs_collection: tuple[str, int],
) -> None:
    path, deck_id = fsrs_collection
    _add_review_state_card(path, deck_id, "Second eligible card", with_revlog=True)

    async with AnkiCollectionService(
        path, max_page_size=100, max_search_scan=2
    ) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        preview = await service.preview_fsrs_reschedule(1, 0.91, None)

    assert preview["cards"] == 2
    assert preview["decks"] == 2


@pytest.mark.anyio
async def test_fsrs_reschedule_candidate_discovery_rejects_one_over_limit_before_loading(
    fsrs_collection: tuple[str, int],
) -> None:
    path, deck_id = fsrs_collection
    _add_review_state_card(path, deck_id, "Ineligible review-state card 1")
    _add_review_state_card(path, deck_id, "Ineligible review-state card 2")

    async with AnkiCollectionService(
        path, max_page_size=100, max_search_scan=2
    ) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        with pytest.raises(
            ValueError,
            match=r"preset 1 across 2 decks: maximum=2, observed_at_least=3 candidate cards",
        ):
            await service.preview_fsrs_reschedule(1, 0.91, None)


@pytest.mark.anyio
async def test_fsrs_reschedule_rejects_preset_deck_scope_over_limit(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(
        path, max_page_size=100, max_search_scan=1
    ) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        with pytest.raises(
            ValueError,
            match=r"preset 1: maximum=1, observed_at_least=2 decks",
        ):
            await service.preview_fsrs_reschedule(1, 0.91, None)


@pytest.mark.anyio
async def test_backup_before_rejects_native_false_with_old_backup(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    backup_folder = Path(path).parent / "backups"
    backup_folder.mkdir()
    (backup_folder / "old.colpkg").write_bytes(b"old backup")
    mutation_called = False

    def create_backup(self, **kwargs):  # type: ignore[no-untyped-def]
        return False

    def mutation() -> dict[str, bool]:
        nonlocal mutation_called
        mutation_called = True
        return {"mutated": True}

    monkeypatch.setattr(Collection, "create_backup", create_backup)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(RuntimeError, match="current pre-operation"):
            await service.executor.run(lambda adapter: adapter.backup_before(mutation))

    assert mutation_called is False


@pytest.mark.anyio
async def test_backup_before_rejects_crc_valid_unrestorable_backup(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    backup_folder = Path(path).parent / "backups"
    backup_folder.mkdir()
    with ZipFile(backup_folder / "garbage.colpkg", "w") as archive:
        archive.writestr("meta", b"garbage")
        archive.writestr("collection.anki21b", b"garbage")
        archive.writestr("collection.anki2", b"garbage")
        archive.writestr("media", b"garbage")
    mutation_called = False

    def mutation() -> dict[str, bool]:
        nonlocal mutation_called
        mutation_called = True
        return {"mutated": True}

    monkeypatch.setattr(Collection, "create_backup", lambda *args, **kwargs: False)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(RuntimeError, match="current pre-operation"):
            await service.executor.run(lambda adapter: adapter.backup_before(mutation))

    assert mutation_called is False


@pytest.mark.anyio
async def test_backup_before_skips_newer_invalid_backup(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    backup_folder = Path(path).parent / "backups"
    backup_folder.mkdir()
    collection = Collection(path)
    try:
        assert collection.create_backup(
            backup_folder=str(backup_folder), force=True, wait_for_completion=True
        )
    finally:
        collection.close()
    valid = next(backup_folder.glob("*.colpkg"))
    old_time = valid.stat().st_mtime - 10
    os.utime(valid, (old_time, old_time))
    invalid = backup_folder / "newer.colpkg"
    with ZipFile(invalid, "w") as archive:
        archive.writestr("meta", b"garbage")
        archive.writestr("collection.anki21b", b"garbage")
        archive.writestr("collection.anki2", b"garbage")
        archive.writestr("media", b"garbage")

    monkeypatch.setattr(Collection, "create_backup", lambda *args, **kwargs: False)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        result = await service.executor.run(
            lambda adapter: adapter.backup_before(lambda: {"mutated": True})
        )

    assert result["mutated"] is True
    assert result["backup"]["path"] == str(valid)


@pytest.mark.anyio
async def test_backup_before_rejects_native_success_without_new_file(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    backup_folder = Path(path).parent / "backups"
    backup_folder.mkdir()
    (backup_folder / "old.colpkg").write_bytes(b"old backup")
    mutation_called = False

    def create_backup(self, **kwargs):  # type: ignore[no-untyped-def]
        return True

    def mutation() -> dict[str, bool]:
        nonlocal mutation_called
        mutation_called = True
        return {"mutated": True}

    monkeypatch.setattr(Collection, "create_backup", create_backup)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(RuntimeError, match="current pre-operation"):
            await service.executor.run(lambda adapter: adapter.backup_before(mutation))

    assert mutation_called is False


@pytest.mark.anyio
async def test_backup_before_propagates_native_failure_without_mutating(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    mutation_called = False

    def create_backup(self, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("native backup failed")

    def mutation() -> dict[str, bool]:
        nonlocal mutation_called
        mutation_called = True
        return {"mutated": True}

    monkeypatch.setattr(Collection, "create_backup", create_backup)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(RuntimeError, match="native backup failed"):
            await service.executor.run(lambda adapter: adapter.backup_before(mutation))

    assert mutation_called is False


@pytest.mark.anyio
async def test_backup_before_returns_new_nonempty_backup_receipt(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    native_create_backup = Collection.create_backup

    def create_backup(
        self, *, backup_folder: str, force: bool, wait_for_completion: bool
    ) -> bool:  # type: ignore[no-untyped-def]
        assert force is True
        assert wait_for_completion is True
        return native_create_backup(
            self,
            backup_folder=backup_folder,
            force=force,
            wait_for_completion=wait_for_completion,
        )

    monkeypatch.setattr(Collection, "create_backup", create_backup)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        result = await service.executor.run(
            lambda adapter: adapter.backup_before(lambda: {"mutated": True})
        )

    assert result["mutated"] is True
    assert result["backup"]["created"] is True
    assert Path(result["backup"]["path"]).is_file()


@pytest.mark.anyio
async def test_backup_failure_keeps_idempotent_mutation_retryable(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    attempts = 0
    native_create_backup = Collection.create_backup

    def create_backup(
        self, *, backup_folder: str, force: bool, wait_for_completion: bool
    ) -> bool:  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("native backup failed")
        return native_create_backup(
            self,
            backup_folder=backup_folder,
            force=force,
            wait_for_completion=wait_for_completion,
        )

    monkeypatch.setattr(Collection, "create_backup", create_backup)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(RuntimeError, match="native backup failed"):
            await service.coordinated_mutation(
                "test_guarded",
                "backup-retry",
                {"value": 1},
                lambda adapter: adapter.backup_before(lambda: {"mutated": True}),
            )
        receipt = await service.coordinated_mutation(
            "test_guarded",
            "backup-retry",
            {"value": 1},
            lambda adapter: adapter.backup_before(lambda: {"mutated": True}),
        )

    assert attempts == 2
    assert receipt["state"] == "committed"
    assert receipt["result"]["mutated"] is True


@pytest.mark.anyio
async def test_fsrs_reschedule_requires_an_actual_fsrs_change(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(ValueError, match="must change"):
            await service.preview_fsrs_reschedule(1, 0.9, None)


@pytest.mark.anyio
async def test_fsrs_reschedule_preview_requires_fsrs_to_be_enabled(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        with pytest.raises(ValueError, match="must be enabled"):
            await service.preview_fsrs_reschedule(1, 0.91, None)


@pytest.mark.anyio
async def test_fsrs_reschedule_rejects_malformed_ignore_date(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection
    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        await service.update_deck_preset(
            1, None, {"ignore_revlogs_before_date": "not-a-date"}
        )
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            await service.preview_fsrs_reschedule(1, 0.91, None)


@pytest.mark.anyio
async def test_anki_26_5_native_fsrs_compatibility_smoke(
    fsrs_collection: tuple[str, int],
) -> None:
    """Exercise each pinned native backend path separately from mocked contract tests."""
    path, _ = fsrs_collection
    assert buildinfo.version == "26.05"

    async with AnkiCollectionService(path, max_page_size=100) as service:
        try:
            await service.preview_fsrs_optimization(1, None, True)
        except ValueError as exc:
            assert "not enough review history" in str(exc)
        for mode in ("review", "workload", "optimal_retention"):
            result = await service.simulate_fsrs(1, mode, 100, 7, None, None, True)
            assert result["mode"] == mode
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        impact = await service.preview_fsrs_reschedule(1, 0.91, None)
        applied = await service.reschedule_fsrs(1, 0.91, None)

    assert impact["cards"] == 1
    assert applied["rescheduled"] is True


@pytest.mark.anyio
async def test_fsrs_optimize_keeps_deck_presets_with_a_later_sorted_preset(
    fsrs_collection: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = fsrs_collection
    candidate = [0.2 + index / 100 for index in range(21)]

    def compute_fsrs_params(self, **kwargs):  # type: ignore[no-untyped-def]
        return scheduler_pb2.ComputeFsrsParamsResponse(
            params=candidate, fsrs_items=10, health_check_passed=True
        )

    monkeypatch.setattr(RustBackendGenerated, "compute_fsrs_params", compute_fsrs_params)
    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.create_deck_preset("Zeta", None)
        before = (await service.get_deck_options(1))["preset"]["id"]
        await service.optimize_fsrs(1, search=None, health_check=True)
        after = (await service.get_deck_options(1))["preset"]["id"]

    assert before == 1
    assert after == 1


@pytest.mark.anyio
async def test_fsrs_reschedule_keeps_deck_presets_with_a_later_sorted_preset(
    fsrs_collection: tuple[str, int],
) -> None:
    path, _ = fsrs_collection

    async with AnkiCollectionService(path, max_page_size=100) as service:
        await service.update_deck_scheduler_settings(None, None, fsrs_enabled=True)
        await service.create_deck_preset("Zeta", None)
        before = (await service.get_deck_options(1))["preset"]["id"]
        await service.reschedule_fsrs(1, 0.91, None)
        after = (await service.get_deck_options(1))["preset"]["id"]

    assert before == 1
    assert after == 1
