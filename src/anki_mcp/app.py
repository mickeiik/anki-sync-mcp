from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Generic, Literal, NoReturn, TypeVar, cast
from uuid import uuid4

from anki.errors import NetworkError, SyncError, SyncErrorKind, UndoEmpty
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp

from anki_mcp.auth import BearerAuthMiddleware, RequestBodyLimitMiddleware
from anki_mcp.collection import (
    UPDATE_CONDITIONS,
    AnkiCollectionService,
    BackupFailedError,
    CollectionAdapter,
    DuplicateNoteError,
    FullSyncRequiredError,
    IdempotencyConflictError,
    MediaSyncFailedError,
    ResourceLimitError,
    RestoreFailedError,
    SyncLoginRequiredError,
)
from anki_mcp.config import Settings
from anki_mcp.guard import ConfirmationRegistry

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)
Offset = StrictInt
PageLimit = StrictInt
StableId = Annotated[StrictInt, Field(gt=0)]
Confirmation = StrictBool
SyncMedia = StrictBool
DeckName = Annotated[StrictStr, Field(min_length=1, max_length=512)]
SearchQuery = Annotated[StrictStr, Field(max_length=4096)]
CardText = Annotated[StrictStr, Field(max_length=262_144)]
IdempotencyKey = Annotated[StrictStr, Field(min_length=1, max_length=128)]
OperationName = Annotated[StrictStr, Field(min_length=1, max_length=128)]
Tag = Annotated[StrictStr, Field(min_length=1, max_length=512)]
ResourceName = Annotated[StrictStr, Field(min_length=1, max_length=512)]
MediaFilename = Annotated[StrictStr, Field(min_length=1, max_length=255)]
BackupFilename = Annotated[StrictStr, Field(min_length=1, max_length=255)]
ImportFilename = Annotated[StrictStr, Field(min_length=1, max_length=255)]
CsvDelimiter = Annotated[StrictStr, Field(min_length=1, max_length=1)]
RestoreMode = Literal["upload_now", "local_only"]
CsvDupeResolution = Literal["preserve", "update", "duplicate"]
UpdateCondition = Literal["ALWAYS", "IF_NEWER", "NEVER"]
MediaContent = Annotated[StrictStr, Field(min_length=1, max_length=22_369_624)]
InlineImportContent = Annotated[StrictStr, Field(min_length=1, max_length=22_369_624)]
StableIds = Annotated[list[StableId], Field(min_length=1, max_length=500)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
CsvColumn = Annotated[
    StrictInt, Field(ge=1, description="1-based CSV column number; 1 is the first column")
]
CsvFieldColumns = Annotated[list[CsvColumn], Field(max_length=1000)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
CardFlag = Annotated[StrictInt, Field(ge=0, le=7)]
DailyLimit = Annotated[StrictInt, Field(ge=0, le=999_999)]
Retention = Annotated[StrictFloat, Field(ge=0.7, le=0.99)]
Tags = Annotated[list[Tag], Field(max_length=100)]
NoteFields = dict[Annotated[StrictStr, Field(min_length=1, max_length=512)], CardText]
ConfirmationToken = Annotated[StrictStr, Field(min_length=1, max_length=256)]
SchemaChangeOperation = Literal["create", "update", "fields_update", "templates_update", "delete"]
ReviewRating = Annotated[StrictInt, Field(ge=1, le=4)]
AnswerSeconds = Annotated[StrictInt, Field(ge=0, le=86_400)]
ReviewOrder = Literal["newest", "oldest"]
ReviewDays = Literal[0, 30, 90, 365]
CardGetSection = Literal["review_summary", "fsrs"]
CardGetSections = Annotated[tuple[CardGetSection, ...], Field(max_length=2)]
ReviewEventField = Literal[
    "review_kind", "rating_label", "intervals", "answer_time", "ease", "memory_state"
]
ReviewEventFields = Annotated[tuple[ReviewEventField, ...], Field(max_length=6)]
ReviewStatsSection = Literal[
    "buttons",
    "card_counts",
    "hours",
    "today",
    "eases",
    "intervals",
    "future_due",
    "added",
    "reviews",
    "rollover_hour",
    "difficulty",
    "retrievability",
    "stability",
    "true_retention",
    "fsrs",
]
ReviewStatsSections = Annotated[tuple[ReviewStatsSection, ...], Field(max_length=15)]
DeckOptionSection = Literal["counts", "parents", "global_settings"]
DeckOptionSections = Annotated[tuple[DeckOptionSection, ...], Field(max_length=3)]
DeckPresetSection = Literal[
    "learning",
    "new_cards",
    "reviews",
    "lapses",
    "burying",
    "display_audio",
    "fsrs",
    "easy_days",
]
DeckPresetSections = Annotated[tuple[DeckPresetSection, ...], Field(max_length=8)]
DailyDeckLimitField = Literal["new_cards_per_day", "reviews_per_day"]
DailyDeckLimitFields = Annotated[tuple[DailyDeckLimitField, ...], Field(max_length=2)]
ThisDeckLimitField = Literal[
    "new_cards_per_day", "reviews_per_day", "desired_retention"
]
ThisDeckLimitFields = Annotated[tuple[ThisDeckLimitField, ...], Field(max_length=3)]
NonNegativeFloat = Annotated[StrictFloat, Field(ge=0)]
PositiveFloat = Annotated[StrictFloat, Field(gt=0)]
Steps = Annotated[list[NonNegativeFloat], Field(max_length=100)]
FsrsParameters = Annotated[list[StrictFloat], Field(max_length=30)]
FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
FsrsRescheduleParameters = Annotated[
    list[FiniteFloat],
    Field(
        min_length=21,
        max_length=21,
        description="Finite 21-value FSRS-6 parameter vector.",
    ),
]
FsrsSimulationMode = Literal["review", "workload", "optimal_retention"]
FsrsSimulationDays = Annotated[StrictInt, Field(ge=1, le=730)]
FsrsSimulationDeckSize = Annotated[
    StrictInt,
    Field(
        ge=1,
        le=1_000_000,
        description="Hypothetical simulated deck size; it is not inferred from the collection.",
    ),
]
DeckPresetId = Annotated[
    StrictInt,
    Field(gt=0, description="Shared deck preset ID from anki_deck_presets_list or deck options."),
]
EasyDaysPercentages = Annotated[list[NonNegativeFloat], Field(min_length=7, max_length=7)]
NewCardInsertOrder = Literal["NEW_CARD_INSERT_ORDER_DUE", "NEW_CARD_INSERT_ORDER_RANDOM"]
NewCardGatherPriority = Literal[
    "NEW_CARD_GATHER_PRIORITY_DECK",
    "NEW_CARD_GATHER_PRIORITY_DECK_THEN_RANDOM_NOTES",
    "NEW_CARD_GATHER_PRIORITY_LOWEST_POSITION",
    "NEW_CARD_GATHER_PRIORITY_HIGHEST_POSITION",
    "NEW_CARD_GATHER_PRIORITY_RANDOM_NOTES",
    "NEW_CARD_GATHER_PRIORITY_RANDOM_CARDS",
]
NewCardSortOrder = Literal[
    "NEW_CARD_SORT_ORDER_TEMPLATE",
    "NEW_CARD_SORT_ORDER_NO_SORT",
    "NEW_CARD_SORT_ORDER_TEMPLATE_THEN_RANDOM",
    "NEW_CARD_SORT_ORDER_RANDOM_NOTE_THEN_TEMPLATE",
    "NEW_CARD_SORT_ORDER_RANDOM_CARD",
]
ReviewMix = Literal[
    "REVIEW_MIX_MIX_WITH_REVIEWS", "REVIEW_MIX_AFTER_REVIEWS", "REVIEW_MIX_BEFORE_REVIEWS"
]
ReviewCardOrder = Literal[
    "REVIEW_CARD_ORDER_DAY",
    "REVIEW_CARD_ORDER_DAY_THEN_DECK",
    "REVIEW_CARD_ORDER_DECK_THEN_DAY",
    "REVIEW_CARD_ORDER_INTERVALS_ASCENDING",
    "REVIEW_CARD_ORDER_INTERVALS_DESCENDING",
    "REVIEW_CARD_ORDER_EASE_ASCENDING",
    "REVIEW_CARD_ORDER_EASE_DESCENDING",
    "REVIEW_CARD_ORDER_RETRIEVABILITY_ASCENDING",
    "REVIEW_CARD_ORDER_RETRIEVABILITY_DESCENDING",
    "REVIEW_CARD_ORDER_RELATIVE_OVERDUENESS",
    "REVIEW_CARD_ORDER_RANDOM",
    "REVIEW_CARD_ORDER_ADDED",
    "REVIEW_CARD_ORDER_REVERSE_ADDED",
]
LeechAction = Literal["LEECH_ACTION_SUSPEND", "LEECH_ACTION_TAG_ONLY"]
QuestionAction = Literal["QUESTION_ACTION_SHOW_ANSWER", "QUESTION_ACTION_SHOW_REMINDER"]
AnswerAction = Literal[
    "ANSWER_ACTION_BURY_CARD",
    "ANSWER_ACTION_ANSWER_AGAIN",
    "ANSWER_ACTION_ANSWER_GOOD",
    "ANSWER_ACTION_ANSWER_HARD",
    "ANSWER_ACTION_SHOW_REMINDER",
]


class NoteCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    deck_id: StableId
    note_type_id: StableId
    fields: NoteFields
    tags: Tags = Field(default_factory=list)


class NoteTypeTemplateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: ResourceName
    question_format: CardText
    answer_format: CardText


class NoteTypeFieldMappingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: ResourceName
    source_ordinal: NonNegativeInt | None = None


class NoteTypeTemplateMappingInput(NoteTypeTemplateInput):
    source_ordinal: NonNegativeInt | None = None


class DeckTodayLimitPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    new_cards_per_day: DailyLimit | None = None
    reviews_per_day: DailyLimit | None = None


class DeckThisLimitPatch(DeckTodayLimitPatch):
    desired_retention: Retention | None = None


class DeckTodayLimitUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    scope: Literal["today"]
    values: DeckTodayLimitPatch | None = None
    clear_fields: DailyDeckLimitFields = ()


class DeckThisLimitUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    scope: Literal["this_deck"]
    values: DeckThisLimitPatch | None = None
    clear_fields: ThisDeckLimitFields = ()


DeckLimitUpdate = Annotated[
    DeckTodayLimitUpdate | DeckThisLimitUpdate,
    Field(discriminator="scope"),
]


class DeckPresetPatch(BaseModel):
    """Typed patch over every user-facing field in Anki's deck preset config."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    learn_steps: Steps | None = None
    relearn_steps: Steps | None = None
    fsrs_params_4: FsrsParameters | None = None
    fsrs_params_5: FsrsParameters | None = None
    fsrs_params_6: FsrsParameters | None = None
    new_per_day: DailyLimit | None = None
    reviews_per_day: DailyLimit | None = None
    new_per_day_minimum: DailyLimit | None = None
    initial_ease: PositiveFloat | None = None
    easy_multiplier: PositiveFloat | None = None
    hard_multiplier: PositiveFloat | None = None
    lapse_multiplier: NonNegativeFloat | None = None
    interval_multiplier: PositiveFloat | None = None
    maximum_review_interval: PositiveInt | None = None
    minimum_lapse_interval: PositiveInt | None = None
    graduating_interval_good: PositiveInt | None = None
    graduating_interval_easy: PositiveInt | None = None
    new_card_insert_order: NewCardInsertOrder | None = None
    new_card_gather_priority: NewCardGatherPriority | None = None
    new_card_sort_order: NewCardSortOrder | None = None
    new_mix: ReviewMix | None = None
    review_order: ReviewCardOrder | None = None
    interday_learning_mix: ReviewMix | None = None
    leech_action: LeechAction | None = None
    leech_threshold: PositiveInt | None = None
    disable_autoplay: StrictBool | None = None
    cap_answer_time_to_secs: NonNegativeInt | None = None
    show_timer: StrictBool | None = None
    stop_timer_on_answer: StrictBool | None = None
    seconds_to_show_question: NonNegativeFloat | None = None
    seconds_to_show_answer: NonNegativeFloat | None = None
    question_action: QuestionAction | None = None
    answer_action: AnswerAction | None = None
    wait_for_audio: StrictBool | None = None
    skip_question_when_replaying_answer: StrictBool | None = None
    bury_new: StrictBool | None = None
    bury_reviews: StrictBool | None = None
    bury_interday_learning: StrictBool | None = None
    desired_retention: Retention | None = None
    ignore_revlogs_before_date: Annotated[StrictStr, Field(max_length=32)] | None = None
    easy_days_percentages: EasyDaysPercentages | None = None
    historical_retention: Retention | None = None
    param_search: SearchQuery | None = None


FieldMappings = Annotated[list[NoteTypeFieldMappingInput], Field(min_length=1, max_length=1000)]
TemplateMappings = Annotated[
    list[NoteTypeTemplateMappingInput], Field(min_length=1, max_length=1000)
]


class ResponseTooLargeError(RuntimeError):
    """Raised when a tool result exceeds the configured serialized response budget."""


class ConfirmationRequiredError(ValueError):
    """Raised when a guarded operation lacks a valid preview token."""


class FsrsAffectedDeck(BaseModel):
    id: int
    name: str


class FsrsOptimizationImpact(BaseModel):
    config_id: int
    search: str
    training_items: int
    health_check_requested: bool
    health_check_passed: bool
    current_parameters: list[float]
    candidate_parameters: list[float]
    affected_decks: int
    affected_deck_ids: list[int]
    affected_decks_detail: list[FsrsAffectedDeck]
    state_fingerprint: str


class FsrsOptimizationPreviewResult(BaseModel):
    operation: Literal["anki_fsrs_optimize_apply"]
    impact: FsrsOptimizationImpact
    confirmation_token: str
    expires_in_seconds: int


class FsrsOptimizationApplied(BaseModel):
    config_id: int
    optimized: Literal[True]
    search: str
    training_items: int
    health_check_requested: bool
    health_check_passed: bool
    previous_parameters: list[float]
    parameters: list[float]
    affected_decks: int
    affected_deck_ids: list[int]
    affected_decks_detail: list[FsrsAffectedDeck]


class FsrsMutationReceipt(BaseModel, Generic[T]):  # noqa: UP046
    idempotency_key: str
    state: Literal["outcome_unknown", "committed", "discarded_by_full_download"]
    local_committed: bool | None
    remote_synced: bool
    media_synced: bool | None
    retryable: bool
    result: T | None


class FsrsRescheduleImpact(BaseModel):
    config_id: int
    decks: int
    cards: int
    desired_retention: float
    parameters_changed: bool
    state_fingerprint: str


class FsrsReschedulePreviewResult(BaseModel):
    operation: Literal["anki_fsrs_reschedule"]
    impact: FsrsRescheduleImpact
    confirmation_token: str
    expires_in_seconds: int


class BackupReceipt(BaseModel):
    created: bool
    path: str
    reason: str | None = None


class FsrsRescheduleApplied(BaseModel):
    config_id: int
    rescheduled: Literal[True]
    cards: int
    decks: int
    desired_retention: float
    parameters_changed: bool
    backup: BackupReceipt


class FsrsReviewSummary(BaseModel):
    total_reviews: int
    total_new_cards: int
    total_time_seconds: float
    final_knowledge_acquisition: float


class FsrsDailyReview(BaseModel):
    day: int
    reviews: int
    new_cards: int
    time_seconds: float
    knowledge_acquisition: float


class FsrsSimulationBase(BaseModel):
    config_id: int
    search: str
    days_to_simulate: int
    deck_size: int


class FsrsReviewSimulationResult(FsrsSimulationBase):
    mode: Literal["review"]
    summary: FsrsReviewSummary
    daily: list[FsrsDailyReview] | None = None


class FsrsWorkloadRetention(BaseModel):
    cost_seconds: dict[str, float]
    memorized_cards: dict[str, float]
    review_count: dict[str, int]


class FsrsWorkloadSimulationResult(FsrsSimulationBase):
    mode: Literal["workload"]
    retention: FsrsWorkloadRetention
    reviewless_end_memorized: float


class FsrsOptimalRetentionSimulationResult(FsrsSimulationBase):
    mode: Literal["optimal_retention"]
    optimal_retention: float


FsrsSimulationResult = Annotated[
    FsrsReviewSimulationResult
    | FsrsWorkloadSimulationResult
    | FsrsOptimalRetentionSimulationResult,
    Field(discriminator="mode"),
]


def create_app(settings: Settings) -> ASGIApp:
    """Create the complete ASGI application and MCP registry."""

    service = AnkiCollectionService(
        settings.collection_path,
        settings.max_page_size,
        settings.max_search_scan,
        settings.max_rendered_field_bytes,
        settings.max_card_fields,
        settings.max_batch_size,
        settings.max_media_bytes,
        settings.sync_timeout_seconds,
        settings.sync_on_read,
        settings.sync_on_write,
        settings.max_response_bytes,
        settings.max_import_bytes,
    )
    mcp = FastMCP(
        "anki-mcp",
        instructions="Authenticated sync and deck/card CRUD for one Anki collection.",
        streamable_http_path=settings.mcp_path,
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=settings.allowed_hosts,
            allowed_origins=settings.allowed_origins,
        ),
    )
    registered_tool_names: list[str] = []
    confirmations = ConfirmationRegistry(settings.confirmation_ttl_seconds)

    def scoped_tool(
        name: str, scope: str, *, enabled: bool = True
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def register(function: Callable[..., Any]) -> Callable[..., Any]:
            if scope in settings.scopes and enabled:
                mcp.tool(name=name)(function)
                registered_tool_names.append(name)
            return function

        return register

    def raise_tool_error(
        code: str,
        message: str,
        cause: Exception,
        *,
        log_cause: bool = False,
    ) -> NoReturn:
        correlation_id = str(uuid4())
        payload = {
            "code": code,
            "message": message,
            "correlation_id": correlation_id,
        }
        if log_cause:
            LOGGER.error(
                "tool failure code=%s correlation_id=%s exception_type=%s",
                code,
                correlation_id,
                type(cause).__name__,
                exc_info=(type(cause), cause, cause.__traceback__),
            )
        raise ToolError(json.dumps(payload, separators=(",", ":"))) from cause

    async def execute(
        operation: Awaitable[T], receipt: Callable[[T], dict[str, Any]] | None = None
    ) -> T | dict[str, Any]:
        try:
            raw_result = await operation
            result = receipt(raw_result) if receipt is not None else raw_result
            encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(encoded) > settings.max_response_bytes:
                raise ResponseTooLargeError("tool response exceeds MCP_MAX_RESPONSE_BYTES")
            return result
        except DuplicateNoteError as exc:
            raise_tool_error("DUPLICATE_NOTE", str(exc), exc)
        except IdempotencyConflictError as exc:
            raise_tool_error("CONFLICT", str(exc), exc)
        except FullSyncRequiredError as exc:
            raise_tool_error("FULL_SYNC_REQUIRED", str(exc), exc)
        except LookupError as exc:
            raise_tool_error("NOT_FOUND", str(exc), exc)
        except ConfirmationRequiredError as exc:
            raise_tool_error("DESTRUCTIVE_CONFIRMATION_REQUIRED", str(exc), exc)
        except ResourceLimitError as exc:
            raise_tool_error("RESOURCE_LIMIT_EXCEEDED", str(exc), exc)
        except ValueError as exc:
            raise_tool_error("INVALID_ARGUMENT", str(exc), exc)
        except SyncLoginRequiredError as exc:
            raise_tool_error("AUTHENTICATION_FAILED", str(exc), exc)
        except MediaSyncFailedError as exc:
            raise_tool_error("MEDIA_SYNC_FAILED", str(exc), exc)
        except BackupFailedError as exc:
            raise_tool_error(
                "BACKUP_FAILED",
                "required collection backup could not be created",
                exc,
                log_cause=True,
            )
        except RestoreFailedError as exc:
            raise_tool_error("RESTORE_FAILED", str(exc), exc)
        except UndoEmpty as exc:
            raise_tool_error("UNDO_UNAVAILABLE", str(exc), exc)
        except ResponseTooLargeError as exc:
            raise_tool_error("RESPONSE_TOO_LARGE", str(exc), exc)
        except NetworkError as exc:
            raise_tool_error("NETWORK_ERROR", "remote sync network request failed", exc)
        except SyncError as exc:
            code = "AUTHENTICATION_FAILED" if exc.kind == SyncErrorKind.AUTH else "SYNC_ERROR"
            raise_tool_error(code, "remote sync operation failed", exc)
        except Exception as exc:
            raise_tool_error(
                "INTERNAL_ERROR",
                "internal collection operation failed",
                exc,
                log_cause=True,
            )

    async def mutate(
        operation: str,
        idempotency_key: str | None,
        request: dict[str, Any],
        function: Callable[[CollectionAdapter], dict[str, Any]],
        sync_media: bool = False,
        sync_after: bool = True,
    ) -> dict[str, Any]:
        key = idempotency_key or str(uuid4())
        result = await execute(
            service.coordinated_mutation(
                operation, key, request, function, sync_media=sync_media, sync_after=sync_after
            )
        )
        if not isinstance(result, dict):  # pragma: no cover - coordinator always returns a receipt
            raise RuntimeError("mutation coordinator returned an invalid receipt")
        return result

    async def preview(
        operation: str,
        request: dict[str, Any],
        function: Callable[[CollectionAdapter], dict[str, Any]],
    ) -> dict[str, Any]:
        impact = await execute(service.coordinated_read(function))
        if not isinstance(impact, dict):  # pragma: no cover - previews always return mappings
            raise RuntimeError("impact preview returned an invalid result")
        return {
            "operation": operation,
            "impact": impact,
            "confirmation_token": confirmations.issue(
                operation, {"request": request, "impact": impact}
            ),
            "expires_in_seconds": settings.confirmation_ttl_seconds,
        }

    async def guarded_mutate(
        operation: str,
        idempotency_key: str | None,
        request: dict[str, Any],
        confirmation_token: str,
        guard_request: dict[str, Any],
        preview_function: Callable[[CollectionAdapter], dict[str, Any]],
        function: Callable[[CollectionAdapter], dict[str, Any]],
        *,
        sync_media: bool = False,
        sync_after: bool = True,
        pre_backup: bool = True,
    ) -> dict[str, Any]:
        def guarded(adapter: CollectionAdapter) -> dict[str, Any]:
            current_impact = preview_function(adapter)
            confirmation_request = {"request": guard_request, "impact": current_impact}
            try:
                confirmations.validate(confirmation_token, operation, confirmation_request)
            except ValueError as exc:
                raise ConfirmationRequiredError(f"{exc}; run preview again") from exc

            def confirmed_mutation() -> dict[str, Any]:
                try:
                    confirmations.consume(confirmation_token, operation, confirmation_request)
                except ValueError as exc:
                    raise ConfirmationRequiredError(f"{exc}; run preview again") from exc
                return function(adapter)

            if pre_backup:
                return adapter.backup_before(confirmed_mutation)
            return confirmed_mutation()

        return await mutate(
            operation,
            idempotency_key,
            request,
            guarded,
            sync_media=sync_media,
            sync_after=sync_after,
        )

    @scoped_tool(name="anki_status", scope="read")
    async def status(recheck: StrictBool = False) -> dict[str, Any]:
        """Return actionable local collection, authentication, sync, and recovery status.

        recheck recomputes the next-write requirement from LOCAL collection state
        (collection.sync_status) and performs NO network request; it cannot observe
        server-side changes such as another client or a remote replace. To confirm the
        server will accept a write, call anki_sync (which performs the incremental sync
        and reports the ``required`` value) or simply attempt the write — the pre-sync
        refuses before committing.
        """
        return await execute(service.status(recheck))

    @scoped_tool(name="anki_operations_list", scope="read")
    async def operations_list(
        offset: Offset = 0, limit: PageLimit = settings.max_page_size
    ) -> dict[str, Any]:
        """List durable mutation operations and their synchronization state."""
        return await execute(service.list_operations(offset, limit))

    @scoped_tool(name="anki_operations_get", scope="read")
    async def operations_get(idempotency_key: IdempotencyKey) -> dict[str, Any]:
        """Get one durable mutation operation by idempotency key."""
        return await execute(service.get_operation(idempotency_key))

    @scoped_tool(name="anki_metrics", scope="read")
    async def metrics() -> dict[str, Any]:
        """Return content-free durable mutation and synchronization metrics."""
        return await execute(service.metrics())

    @scoped_tool(name="anki_sync_login", scope="admin")
    async def sync_login() -> dict[str, Any]:
        """Authenticate to the configured AnkiWeb or self-hosted sync endpoint."""
        if not settings.sync_username.strip():
            cause = ValueError("ANKI_SYNC_USERNAME is not configured")
            raise_tool_error("INVALID_ARGUMENT", str(cause), cause)
        if settings.sync_password is None:
            cause = ValueError("ANKI_SYNC_PASSWORD is not configured")
            raise_tool_error("INVALID_ARGUMENT", str(cause), cause)
        return await execute(
            service.sync_login(
                settings.sync_username,
                settings.sync_password.get_secret_value(),
                settings.sync_endpoint,
            )
        )

    @scoped_tool(name="anki_sync", scope="write")
    async def sync(sync_media: SyncMedia = True) -> dict[str, Any]:
        """Synchronize the collection with the authenticated remote server."""
        return await execute(service.sync(sync_media))

    @scoped_tool(name="anki_sync_full_download", scope="admin", enabled=settings.allow_full_sync)
    async def sync_full_download(confirm: Confirmation = False) -> dict[str, Any]:
        """Replace local data after the server requires a full download and confirmation is true."""
        if not confirm:
            cause = ValueError("confirm must be true for full download")
            raise_tool_error("INVALID_ARGUMENT", str(cause), cause)
        return await execute(service.full_sync(upload=False))

    @scoped_tool(name="anki_sync_full_upload", scope="admin", enabled=settings.allow_full_sync)
    async def sync_full_upload(
        confirm: Confirmation = False, force: StrictBool = False
    ) -> dict[str, Any]:
        """Replace remote data after a full upload is required or a restore needs overwriting."""
        if not confirm:
            cause = ValueError("confirm must be true for full upload")
            raise_tool_error("INVALID_ARGUMENT", str(cause), cause)
        return await execute(service.full_sync(upload=True, force=force))

    @scoped_tool(name="anki_backup_create", scope="admin")
    async def backup_create() -> dict[str, Any]:
        """Create an explicit local collection backup in persistent storage.

        Anki skips writing a new backup when the collection is unchanged since the last one.
        ``created=false`` with a non-null ``path`` means that existing backup was reused as a
        valid current pre-operation backup. ``reason`` is null when a backup was created,
        "no_collection_changes_since_last_backup" when an existing backup was reused, and
        "no_valid_backup_available" when none could be selected.
        """
        return await execute(service.create_backup())

    @scoped_tool(name="anki_backups_list", scope="read")
    async def backups_list(
        offset: Offset = 0, limit: PageLimit = settings.max_page_size
    ) -> dict[str, Any]:
        """List available local collection backups, newest first."""
        return await execute(service.list_backups(offset, limit))

    @scoped_tool(
        name="anki_backup_restore_preview",
        scope="destructive",
        enabled=settings.allow_restore and settings.allow_full_sync,
    )
    async def backup_restore_preview(filename: BackupFilename) -> dict[str, Any]:
        """Validate one backup and preview the impact of restoring it; issues a token."""
        return await preview(
            "anki_backup_restore",
            {"filename": filename},
            lambda adapter: adapter.preview_backup_restore(filename),
        )

    @scoped_tool(
        name="anki_backup_restore",
        scope="destructive",
        enabled=settings.allow_restore and settings.allow_full_sync,
    )
    async def backup_restore(
        filename: BackupFilename,
        mode: RestoreMode,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey,
    ) -> dict[str, Any]:
        """Replace the live collection with a verified backup after a matching preview token."""
        return await guarded_mutate(
            "anki_backup_restore",
            idempotency_key,
            {"filename": filename, "mode": mode},
            confirmation_token,
            {"filename": filename},
            lambda adapter: adapter.preview_backup_restore(filename),
            lambda adapter: adapter.restore_backup(filename, mode),
            sync_after=False,
            pre_backup=False,
        )

    @scoped_tool(name="anki_decks_list", scope="read")
    async def decks_list(
        offset: Offset = 0,
        limit: PageLimit = settings.max_page_size,
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """List decks with stable IDs and hierarchy, using bounded offset pagination."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.list_decks(offset=offset, limit=limit), sync_before
            )
        )

    @scoped_tool(name="anki_decks_get", scope="read")
    async def decks_get(deck_id: StableId, sync_before: SyncMedia = False) -> dict[str, Any]:
        """Get metadata for one deck by stable Anki deck ID."""
        return await execute(
            service.coordinated_read(lambda adapter: adapter.get_deck(deck_id), sync_before)
        )

    @scoped_tool(name="anki_deck_options_get", scope="read")
    async def deck_options_get(
        deck_id: StableId,
        include_sections: DeckOptionSections = (),
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """Get compact layered deck options, with optional counts, parents, and globals."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.get_deck_options(deck_id, include_sections), sync_before
            )
        )

    @scoped_tool(name="anki_deck_presets_list", scope="read")
    async def deck_presets_list(
        offset: Offset = 0,
        limit: PageLimit = settings.max_page_size,
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """List compact deck preset summaries and shared-use counts."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.list_deck_presets(offset, limit), sync_before
            )
        )

    @scoped_tool(name="anki_deck_presets_get", scope="read")
    async def deck_presets_get(
        config_id: StableId,
        include_sections: DeckPresetSections = (),
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """Get compact preset defaults, optionally projecting detailed option groups."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.get_deck_preset(config_id, include_sections), sync_before
            )
        )

    @scoped_tool(name="anki_decks_create", scope="write")
    async def decks_create(
        name: DeckName, idempotency_key: IdempotencyKey | None = None
    ) -> dict[str, Any]:
        """Create a deck by name, or return the existing deck with that name."""
        return await mutate(
            "anki_decks_create",
            idempotency_key,
            {"name": name},
            lambda adapter: adapter.create_deck(name),
        )

    @scoped_tool(name="anki_decks_update", scope="write")
    async def decks_update(
        deck_id: StableId,
        name: DeckName,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Rename a deck by stable Anki deck ID."""
        return await mutate(
            "anki_decks_update",
            idempotency_key,
            {"deck_id": deck_id, "name": name},
            lambda adapter: adapter.update_deck(deck_id, name),
        )

    @scoped_tool(name="anki_decks_update_config", scope="admin")
    async def decks_update_config(
        deck_id: StableId,
        new_cards_per_day: DailyLimit | None = None,
        reviews_per_day: DailyLimit | None = None,
        max_answer_seconds: NonNegativeInt | None = None,
        desired_retention: Retention | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Update supported bounded options on a deck's shared configuration."""
        request = {
            "deck_id": deck_id,
            "new_cards_per_day": new_cards_per_day,
            "reviews_per_day": reviews_per_day,
            "max_answer_seconds": max_answer_seconds,
            "desired_retention": desired_retention,
        }
        return await mutate(
            "anki_decks_update_config",
            idempotency_key,
            request,
            lambda adapter: adapter.update_deck_config(
                deck_id,
                new_cards_per_day,
                reviews_per_day,
                max_answer_seconds,
                desired_retention,
            ),
        )

    @scoped_tool(name="anki_deck_limits_update", scope="admin")
    async def deck_limits_update(
        deck_id: StableId,
        update: DeckLimitUpdate,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Patch or clear validated this-deck or today-only limits for one deck."""
        value_dict = (
            update.values.model_dump(exclude_none=True) if update.values is not None else {}
        )
        request = {
            "deck_id": deck_id,
            "update": {
                "scope": update.scope,
                "values": value_dict,
                "clear_fields": list(update.clear_fields),
            },
        }
        return await mutate(
            "anki_deck_limits_update",
            idempotency_key,
            request,
            lambda adapter: adapter.update_deck_limits(
                deck_id,
                update.scope,
                value_dict,
                update.clear_fields,
            ),
        )

    @scoped_tool(name="anki_scheduler_settings_update", scope="admin")
    async def scheduler_settings_update(
        apply_all_parent_limits: StrictBool | None = None,
        new_cards_ignore_review_limit: StrictBool | None = None,
        fsrs_enabled: StrictBool | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Patch collection-wide scheduler and FSRS enablement settings."""
        request = {
            "apply_all_parent_limits": apply_all_parent_limits,
            "new_cards_ignore_review_limit": new_cards_ignore_review_limit,
            "fsrs_enabled": fsrs_enabled,
        }
        return await mutate(
            "anki_scheduler_settings_update",
            idempotency_key,
            request,
            lambda adapter: adapter.update_deck_scheduler_settings(
                apply_all_parent_limits,
                new_cards_ignore_review_limit,
                fsrs_enabled,
            ),
        )

    @scoped_tool(name="anki_fsrs_optimize_preview", scope="admin")
    async def fsrs_optimize_preview(
        config_id: DeckPresetId,
        search: SearchQuery | None = None,
        health_check: StrictBool = True,
    ) -> FsrsOptimizationPreviewResult:
        """Preview native FSRS-6 optimization for a preset ID without changing it.

        Search defaults to preset param_search, then its non-suspended cards. A requested
        failed health check is rejected; health_check=false explicitly bypasses that check.
        The preview reports candidate/current parameters, training count, and affected decks.
        """
        request = {"config_id": config_id, "search": search, "health_check": health_check}
        result = await preview(
            "anki_fsrs_optimize_apply",
            request,
            lambda adapter: adapter.preview_fsrs_optimization(config_id, search, health_check),
        )
        return FsrsOptimizationPreviewResult.model_validate(result)

    @scoped_tool(name="anki_fsrs_optimize_apply", scope="admin")
    async def fsrs_optimize_apply(
        config_id: DeckPresetId,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey,
        search: SearchQuery | None = None,
        health_check: StrictBool = True,
    ) -> FsrsMutationReceipt[FsrsOptimizationApplied]:
        """Apply a matching optimization preview to every deck sharing one preset ID.

        Use identical preview arguments. idempotency_key is required, stable across retries of
        this logical request, and replays return the durable receipt.
        """
        guard_request = {"config_id": config_id, "search": search, "health_check": health_check}
        request = {**guard_request, "confirmation_token": confirmation_token}

        def guarded(adapter: CollectionAdapter) -> dict[str, Any]:
            current_impact = adapter.preview_fsrs_optimization(config_id, search, health_check)
            try:
                confirmations.consume(
                    confirmation_token,
                    "anki_fsrs_optimize_apply",
                    {"request": guard_request, "impact": current_impact},
                )
            except ValueError as exc:
                raise ConfirmationRequiredError(f"{exc}; run optimization preview again") from exc
            return adapter.apply_fsrs_optimization(
                config_id, search, health_check, impact=current_impact
            )

        result = await mutate("anki_fsrs_optimize_apply", idempotency_key, request, guarded)
        return FsrsMutationReceipt[FsrsOptimizationApplied].model_validate(result)

    @scoped_tool(name="anki_fsrs_simulate", scope="read")
    async def fsrs_simulate(
        config_id: DeckPresetId,
        mode: FsrsSimulationMode,
        deck_size: FsrsSimulationDeckSize,
        days_to_simulate: FsrsSimulationDays,
        desired_retention: Retention | None = None,
        search: SearchQuery | None = None,
        include_daily: StrictBool = False,
        sync_before: SyncMedia = False,
    ) -> FsrsSimulationResult:
        """Simulate a preset ID with Anki 26.5 native FSRS without changing the collection.

        review estimates reviews/new cards/time seconds and knowledge acquisition; workload
        maps retention to cost seconds, memorized cards, and review count; optimal_retention
        estimates a target. Search defaults to preset param_search, then non-suspended cards.
        deck_size is hypothetical; include_daily returns exactly days_to_simulate rows.
        """
        result = await execute(
            service.coordinated_read(
                lambda adapter: adapter.simulate_fsrs(
                    config_id,
                    mode,
                    deck_size,
                    days_to_simulate,
                    desired_retention,
                    search,
                    include_daily,
                ),
                sync_before,
            )
        )
        return cast(FsrsSimulationResult, result)

    @scoped_tool(
        name="anki_fsrs_reschedule_preview",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def fsrs_reschedule_preview(
        config_id: DeckPresetId,
        desired_retention: Retention | None = None,
        parameters: FsrsRescheduleParameters | None = None,
    ) -> FsrsReschedulePreviewResult:
        """Preview native rescheduling after changing one shared preset ID.

        FSRS must first be enabled with anki_scheduler_settings_update. parameters is a finite
        21-value FSRS-6 vector. This destructive-scope tool requires ANKI_ALLOW_DESTRUCTIVE;
        execution requires identical change arguments and the unexpired token returned here.
        """
        request = {
            "config_id": config_id,
            "desired_retention": desired_retention,
            "parameters": parameters,
        }
        result = await preview(
            "anki_fsrs_reschedule",
            request,
            lambda adapter: adapter.preview_fsrs_reschedule(
                config_id, desired_retention, parameters
            ),
        )
        return FsrsReschedulePreviewResult.model_validate(result)

    @scoped_tool(
        name="anki_fsrs_reschedule",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def fsrs_reschedule(
        config_id: DeckPresetId,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey,
        desired_retention: Retention | None = None,
        parameters: FsrsRescheduleParameters | None = None,
    ) -> FsrsMutationReceipt[FsrsRescheduleApplied]:
        """Apply a matching FSRS reschedule preview after a verified fresh backup.

        The finite 21-value parameters and retention must match preview. idempotency_key is
        required and must remain stable across retries of this logical request. A stale token
        requires a new preview. This affects every deck sharing the preset ID.
        """
        guard_request = {
            "config_id": config_id,
            "desired_retention": desired_retention,
            "parameters": parameters,
        }
        request = {**guard_request, "confirmation_token": confirmation_token}
        result = await guarded_mutate(
            "anki_fsrs_reschedule",
            idempotency_key,
            request,
            confirmation_token,
            guard_request,
            lambda adapter: adapter.preview_fsrs_reschedule(
                config_id, desired_retention, parameters
            ),
            lambda adapter: adapter.reschedule_fsrs(config_id, desired_retention, parameters),
        )
        return FsrsMutationReceipt[FsrsRescheduleApplied].model_validate(result)

    @scoped_tool(name="anki_deck_presets_create", scope="admin")
    async def deck_presets_create(
        name: ResourceName,
        clone_from_config_id: StableId | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Create a deck preset from Anki defaults or clone an existing preset."""
        return await mutate(
            "anki_deck_presets_create",
            idempotency_key,
            {"name": name, "clone_from_config_id": clone_from_config_id},
            lambda adapter: adapter.create_deck_preset(name, clone_from_config_id),
        )

    @scoped_tool(name="anki_deck_presets_update", scope="admin")
    async def deck_presets_update(
        config_id: StableId,
        name: ResourceName | None = None,
        options: DeckPresetPatch | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Patch only supplied fields on a shared preset; omitted options remain unchanged."""
        option_dict = options.model_dump(exclude_none=True) if options is not None else {}
        return await mutate(
            "anki_deck_presets_update",
            idempotency_key,
            {"config_id": config_id, "name": name, "options": option_dict},
            lambda adapter: adapter.update_deck_preset(config_id, name, option_dict),
        )

    @scoped_tool(name="anki_deck_presets_assign", scope="admin")
    async def deck_presets_assign(
        deck_id: StableId,
        config_id: StableId,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Assign an existing shared preset to one normal deck."""
        return await mutate(
            "anki_deck_presets_assign",
            idempotency_key,
            {"deck_id": deck_id, "config_id": config_id},
            lambda adapter: adapter.assign_deck_preset(deck_id, config_id),
        )

    @scoped_tool(
        name="anki_decks_delete_preview",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def decks_delete_preview(deck_id: StableId) -> dict[str, Any]:
        """Preview the decks, cards, and notes affected by a deck deletion."""
        request = {"deck_id": deck_id}
        return await preview(
            "anki_decks_delete", request, lambda adapter: adapter.preview_deck_delete(deck_id)
        )

    @scoped_tool(
        name="anki_decks_delete",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def decks_delete(
        deck_id: StableId,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Delete a deck after a matching impact preview and required backup."""
        request = {"deck_id": deck_id}
        return await guarded_mutate(
            "anki_decks_delete",
            idempotency_key,
            request,
            confirmation_token,
            request,
            lambda adapter: adapter.preview_deck_delete(deck_id),
            lambda adapter: adapter.delete_deck(deck_id),
        )

    @scoped_tool(name="anki_notes_search", scope="read")
    async def notes_search(
        query: SearchQuery = "",
        offset: Offset = 0,
        limit: PageLimit = settings.max_page_size,
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """Search notes with Anki search syntax and bounded offset pagination."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.search_notes(query, offset, limit), sync_before
            )
        )

    @scoped_tool(name="anki_notes_get", scope="read")
    async def notes_get(note_id: StableId, sync_before: SyncMedia = False) -> dict[str, Any]:
        """Get general note fields, tags, note type, cards, and metadata by stable ID."""
        return await execute(
            service.coordinated_read(lambda adapter: adapter.get_note(note_id), sync_before)
        )

    @scoped_tool(name="anki_notes_create", scope="write")
    async def notes_create(
        deck_id: StableId,
        note_type_id: StableId,
        fields: NoteFields,
        tags: Tags | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Create one validated, duplicate-checked note with a durable idempotent receipt."""
        normalized_tags = tags or []
        request = {
            "deck_id": deck_id,
            "note_type_id": note_type_id,
            "fields": fields,
            "tags": normalized_tags,
        }
        return await mutate(
            "anki_notes_create",
            idempotency_key,
            request,
            lambda adapter: adapter.create_note(deck_id, note_type_id, fields, normalized_tags),
        )

    @scoped_tool(name="anki_notes_create_batch", scope="write")
    async def notes_create_batch(
        notes: list[NoteCreateInput], idempotency_key: IdempotencyKey | None = None
    ) -> dict[str, Any]:
        """Create a bounded atomic batch of validated notes with one idempotency key."""
        if not notes or len(notes) > settings.max_batch_size:
            cause = ValueError(f"notes must contain between 1 and {settings.max_batch_size} items")
            raise_tool_error("INVALID_ARGUMENT", str(cause), cause)
        requests = [note.model_dump() for note in notes]
        return await mutate(
            "anki_notes_create_batch",
            idempotency_key,
            {"notes": requests},
            lambda adapter: adapter.create_notes_batch(requests),
        )

    @scoped_tool(name="anki_notes_update_fields", scope="write")
    async def notes_update_fields(
        note_id: StableId,
        fields: NoteFields,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Patch named fields on a general note after note-type validation."""
        return await mutate(
            "anki_notes_update_fields",
            idempotency_key,
            {"note_id": note_id, "fields": fields},
            lambda adapter: adapter.update_note_fields(note_id, fields),
        )

    @scoped_tool(name="anki_notes_add_tags", scope="write")
    async def notes_add_tags(
        note_ids: StableIds,
        tags: Tags,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Add normalized tags to a bounded set of notes."""
        return await mutate(
            "anki_notes_add_tags",
            idempotency_key,
            {"note_ids": note_ids, "tags": tags},
            lambda adapter: adapter.add_note_tags(note_ids, tags),
        )

    @scoped_tool(name="anki_notes_remove_tags", scope="write")
    async def notes_remove_tags(
        note_ids: StableIds,
        tags: Tags,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Remove normalized tags from a bounded set of notes."""
        return await mutate(
            "anki_notes_remove_tags",
            idempotency_key,
            {"note_ids": note_ids, "tags": tags},
            lambda adapter: adapter.remove_note_tags(note_ids, tags),
        )

    @scoped_tool(
        name="anki_notes_delete_preview",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def notes_delete_preview(note_ids: StableIds) -> dict[str, Any]:
        """Preview the notes and generated cards affected by note deletion."""
        request = {"note_ids": note_ids}
        return await preview(
            "anki_notes_delete", request, lambda adapter: adapter.preview_notes_delete(note_ids)
        )

    @scoped_tool(
        name="anki_notes_delete",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def notes_delete(
        note_ids: StableIds,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Delete notes after a matching impact preview and required backup."""
        request = {"note_ids": note_ids}
        return await guarded_mutate(
            "anki_notes_delete",
            idempotency_key,
            request,
            confirmation_token,
            request,
            lambda adapter: adapter.preview_notes_delete(note_ids),
            lambda adapter: adapter.delete_notes(note_ids),
        )

    @scoped_tool(name="anki_tags_list", scope="read")
    async def tags_list(
        offset: Offset = 0,
        limit: PageLimit = settings.max_page_size,
        include_counts: StrictBool = False,
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """List collection tags with bounded offset pagination.

        With include_counts=true each tag also reports note_count, the number of
        notes carrying that tag or any descendant (child) tag; enabling it scans
        the collection and is bounded by MCP_MAX_SEARCH_SCAN.
        """
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.list_tags(offset, limit, include_counts), sync_before
            )
        )

    @scoped_tool(name="anki_tags_rename", scope="write")
    async def tags_rename(
        old_name: Tag,
        new_name: Tag,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Rename a tag across the collection."""
        return await mutate(
            "anki_tags_rename",
            idempotency_key,
            {"old_name": old_name, "new_name": new_name},
            lambda adapter: adapter.rename_tag(old_name, new_name),
        )

    @scoped_tool(
        name="anki_tags_delete_preview",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def tags_delete_preview(name: Tag) -> dict[str, Any]:
        """Preview the notes and child tags (``name::*``) that will lose this tag."""
        request = {"name": name}
        return await preview(
            "anki_tags_delete", request, lambda adapter: adapter.preview_tag_delete(name)
        )

    @scoped_tool(
        name="anki_tags_delete",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def tags_delete(
        name: Tag,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Remove a tag after a matching impact preview and required backup."""
        request = {"name": name}
        return await guarded_mutate(
            "anki_tags_delete",
            idempotency_key,
            request,
            confirmation_token,
            request,
            lambda adapter: adapter.preview_tag_delete(name),
            lambda adapter: adapter.delete_tag(name),
        )

    @scoped_tool(
        name="anki_tags_merge_preview",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def tags_merge_preview(source: Tag, target: Tag) -> dict[str, Any]:
        """Preview a tag merge: notes affected and the source/target tag trees involved."""
        request = {"source": source, "target": target}
        return await preview(
            "anki_tags_merge", request, lambda adapter: adapter.preview_tag_merge(source, target)
        )

    @scoped_tool(
        name="anki_tags_merge",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def tags_merge(
        source: Tag,
        target: Tag,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Merge a tag and its children into another tag after preview and required backup."""
        request = {"source": source, "target": target}
        return await guarded_mutate(
            "anki_tags_merge",
            idempotency_key,
            request,
            confirmation_token,
            request,
            lambda adapter: adapter.preview_tag_merge(source, target),
            lambda adapter: adapter.merge_tags(source, target),
        )

    @scoped_tool(name="anki_cards_search", scope="read")
    async def cards_search(
        query: SearchQuery = "",
        offset: Offset = 0,
        limit: PageLimit = settings.max_page_size,
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """Search cards with Anki search syntax and bounded offset pagination."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.search_cards(query=query, offset=offset, limit=limit),
                sync_before,
            )
        )

    @scoped_tool(name="anki_cards_get", scope="read")
    async def cards_get(
        card_id: StableId,
        include_sections: CardGetSections = (),
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """Get card content and scheduling, with optional review_summary and fsrs sections."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.get_card(card_id, include_sections), sync_before
            )
        )

    @scoped_tool(name="anki_reviews_list", scope="read")
    async def reviews_list(
        card_id: StableId | None = None,
        deck_id: StableId | None = None,
        query: SearchQuery | None = None,
        include_children: StrictBool = False,
        offset: Offset = 0,
        limit: PageLimit = settings.max_page_size,
        order: ReviewOrder = "newest",
        include_fields: ReviewEventFields = (),
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """List compact review events, optionally including requested detail fields."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.list_reviews(
                    card_id,
                    deck_id,
                    query,
                    include_children,
                    offset,
                    limit,
                    order,
                    include_fields,
                ),
                sync_before,
            )
        )

    @scoped_tool(name="anki_review_stats", scope="read")
    async def review_stats(
        card_id: StableId | None = None,
        deck_id: StableId | None = None,
        query: SearchQuery | None = None,
        include_children: StrictBool = False,
        days: ReviewDays = 30,
        include_sections: ReviewStatsSections = (),
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """Return compact review analytics with optional detailed graph sections."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.review_stats(
                    card_id, deck_id, query, include_children, days, include_sections
                ),
                sync_before,
            )
        )

    @scoped_tool(name="anki_cards_create", scope="write")
    async def cards_create(
        deck_id: StableId,
        front: CardText,
        back: CardText,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Create one Basic note/card in a deck."""
        return await mutate(
            "anki_cards_create",
            idempotency_key,
            {"deck_id": deck_id, "front": front, "back": back},
            lambda adapter: adapter.create_card(deck_id, front, back),
        )

    @scoped_tool(name="anki_cards_update", scope="write")
    async def cards_update(
        card_id: StableId,
        front: CardText | None = None,
        back: CardText | None = None,
        deck_id: StableId | None = None,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Update a Basic card's Front/Back fields and/or move it to another deck."""
        return await mutate(
            "anki_cards_update",
            idempotency_key,
            {"card_id": card_id, "front": front, "back": back, "deck_id": deck_id},
            lambda adapter: adapter.update_card(card_id, front, back, deck_id),
        )

    @scoped_tool(name="anki_cards_change_deck", scope="write")
    async def cards_change_deck(
        card_ids: StableIds,
        deck_id: StableId,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Move arbitrary supported cards to another deck by stable IDs."""
        return await mutate(
            "anki_cards_change_deck",
            idempotency_key,
            {"card_ids": card_ids, "deck_id": deck_id},
            lambda adapter: adapter.change_card_deck(card_ids, deck_id),
        )

    @scoped_tool(name="anki_cards_suspend", scope="write")
    async def cards_suspend(
        card_ids: StableIds, idempotency_key: IdempotencyKey | None = None
    ) -> dict[str, Any]:
        """Suspend arbitrary supported cards by stable IDs."""
        return await mutate(
            "anki_cards_suspend",
            idempotency_key,
            {"card_ids": card_ids},
            lambda adapter: adapter.suspend_cards(card_ids),
        )

    @scoped_tool(name="anki_cards_unsuspend", scope="write")
    async def cards_unsuspend(
        card_ids: StableIds, idempotency_key: IdempotencyKey | None = None
    ) -> dict[str, Any]:
        """Unsuspend arbitrary supported cards by stable IDs."""
        return await mutate(
            "anki_cards_unsuspend",
            idempotency_key,
            {"card_ids": card_ids},
            lambda adapter: adapter.unsuspend_cards(card_ids),
        )

    @scoped_tool(name="anki_cards_set_flag", scope="write")
    async def cards_set_flag(
        card_ids: StableIds,
        flag: CardFlag,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Set or clear the flag on a bounded set of cards."""
        return await mutate(
            "anki_cards_set_flag",
            idempotency_key,
            {"card_ids": card_ids, "flag": flag},
            lambda adapter: adapter.set_card_flag(card_ids, flag),
        )

    @scoped_tool(name="anki_cards_reposition", scope="admin")
    async def cards_reposition(
        card_ids: StableIds,
        starting_from: PositiveInt,
        step_size: PositiveInt = 1,
        randomize: StrictBool = False,
        shift_existing: StrictBool = True,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Reposition a bounded set of new cards using explicit scheduling controls."""
        request = {
            "card_ids": card_ids,
            "starting_from": starting_from,
            "step_size": step_size,
            "randomize": randomize,
            "shift_existing": shift_existing,
        }
        return await mutate(
            "anki_cards_reposition",
            idempotency_key,
            request,
            lambda adapter: adapter.reposition_cards(
                card_ids, starting_from, step_size, randomize, shift_existing
            ),
        )

    @scoped_tool(
        name="anki_cards_answer",
        scope="admin",
        enabled=settings.allow_review_answers,
    )
    async def cards_answer(
        card_id: StableId,
        rating: ReviewRating,
        answer_seconds: AnswerSeconds = 0,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Record a real review answer when explicitly enabled by the operator."""
        return await mutate(
            "anki_cards_answer",
            idempotency_key,
            {"card_id": card_id, "rating": rating, "answer_seconds": answer_seconds},
            lambda adapter: adapter.answer_card(card_id, rating, answer_seconds),
        )

    @scoped_tool(
        name="anki_cards_delete_preview",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def cards_delete_preview(card_id: StableId) -> dict[str, Any]:
        """Preview card deletion and whether its orphaned note will also be deleted."""
        request = {"card_id": card_id}
        return await preview(
            "anki_cards_delete", request, lambda adapter: adapter.preview_card_delete(card_id)
        )

    @scoped_tool(
        name="anki_cards_delete",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def cards_delete(
        card_id: StableId,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Delete one card after a matching impact preview and required backup."""
        request = {"card_id": card_id}
        return await guarded_mutate(
            "anki_cards_delete",
            idempotency_key,
            request,
            confirmation_token,
            request,
            lambda adapter: adapter.preview_card_delete(card_id),
            lambda adapter: adapter.delete_card(card_id),
        )

    @scoped_tool(name="anki_note_types_list", scope="read")
    async def note_types_list(
        offset: Offset = 0,
        limit: PageLimit = settings.max_page_size,
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """List note types with stable IDs and bounded pagination."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.list_note_types(offset, limit), sync_before
            )
        )

    @scoped_tool(name="anki_note_types_get", scope="read")
    async def note_types_get(
        note_type_id: StableId, sync_before: SyncMedia = False
    ) -> dict[str, Any]:
        """Get fields, templates, CSS, kind, and usage for one note type."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.get_note_type(note_type_id), sync_before
            )
        )

    @scoped_tool(
        name="anki_note_types_change_preview",
        scope="admin",
        enabled=settings.allow_schema_changes and settings.allow_full_sync,
    )
    async def note_types_change_preview(
        operation: SchemaChangeOperation,
        note_type_id: StableId | None = None,
        name: ResourceName | None = None,
        fields: list[ResourceName] | None = None,
        templates: list[NoteTypeTemplateInput] | None = None,
        css: CardText = "",
        field_mappings: FieldMappings | None = None,
        template_mappings: TemplateMappings | None = None,
    ) -> dict[str, Any]:
        """Preview an exact proposed schema mutation and issue a maintenance token."""
        operation_names = {
            "create": "anki_note_types_create",
            "update": "anki_note_types_update",
            "fields_update": "anki_note_type_fields_update",
            "templates_update": "anki_templates_update",
            "delete": "anki_note_types_delete",
        }
        template_values = (
            [template.model_dump() for template in templates] if templates is not None else None
        )
        field_mapping_values = (
            [mapping.model_dump() for mapping in field_mappings]
            if field_mappings is not None
            else None
        )
        template_mapping_values = (
            [mapping.model_dump() for mapping in template_mappings]
            if template_mappings is not None
            else None
        )
        if operation == "create":
            if (
                note_type_id is not None
                or name is None
                or fields is None
                or template_values is None
                or field_mapping_values is not None
                or template_mapping_values is not None
            ):
                cause = ValueError("create preview requires only name, fields, templates, and css")
                raise_tool_error("INVALID_ARGUMENT", str(cause), cause)
            mutation_request = {
                "name": name,
                "fields": fields,
                "templates": template_values,
                "css": css,
            }
        elif operation == "update":
            if (
                note_type_id is None
                or name is None
                or fields is None
                or template_values is None
                or field_mapping_values is not None
                or template_mapping_values is not None
            ):
                cause = ValueError(
                    "update preview requires note_type_id, name, fields, templates, and css"
                )
                raise_tool_error("INVALID_ARGUMENT", str(cause), cause)
            mutation_request = {
                "note_type_id": note_type_id,
                "name": name,
                "fields": fields,
                "templates": template_values,
                "css": css,
            }
        elif operation == "fields_update":
            if (
                note_type_id is None
                or field_mapping_values is None
                or name is not None
                or fields is not None
                or template_values is not None
                or template_mapping_values is not None
                or css
            ):
                cause = ValueError(
                    "fields_update preview requires only note_type_id and field_mappings"
                )
                raise_tool_error("INVALID_ARGUMENT", str(cause), cause)
            mutation_request = {
                "note_type_id": note_type_id,
                "mappings": field_mapping_values,
            }
        elif operation == "templates_update":
            if (
                note_type_id is None
                or template_mapping_values is None
                or name is not None
                or fields is not None
                or template_values is not None
                or field_mapping_values is not None
                or css
            ):
                cause = ValueError(
                    "templates_update preview requires only note_type_id and template_mappings"
                )
                raise_tool_error("INVALID_ARGUMENT", str(cause), cause)
            mutation_request = {
                "note_type_id": note_type_id,
                "mappings": template_mapping_values,
            }
        else:
            if (
                note_type_id is None
                or name is not None
                or fields is not None
                or template_values is not None
                or field_mapping_values is not None
                or template_mapping_values is not None
                or css
            ):
                cause = ValueError("delete preview requires only note_type_id")
                raise_tool_error("INVALID_ARGUMENT", str(cause), cause)
            mutation_request = {"note_type_id": note_type_id}
        guard_request = {"operation": operation, "request": mutation_request}
        return await preview(
            operation_names[operation],
            guard_request,
            lambda adapter: adapter.preview_note_type_change(operation, note_type_id),
        )

    @scoped_tool(
        name="anki_note_types_create",
        scope="admin",
        enabled=settings.allow_schema_changes and settings.allow_full_sync,
    )
    async def note_types_create(
        name: ResourceName,
        fields: list[ResourceName],
        templates: list[NoteTypeTemplateInput],
        confirmation_token: ConfirmationToken,
        css: CardText = "",
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Create a note type after preview, backup, and full-sync maintenance gating."""
        template_values = [template.model_dump() for template in templates]
        request = {"name": name, "fields": fields, "templates": template_values, "css": css}
        return await guarded_mutate(
            "anki_note_types_create",
            idempotency_key,
            request,
            confirmation_token,
            {"operation": "create", "request": request},
            lambda adapter: adapter.preview_note_type_change("create", None),
            lambda adapter: adapter.create_note_type(name, fields, template_values, css),
        )

    @scoped_tool(
        name="anki_note_types_update",
        scope="admin",
        enabled=settings.allow_schema_changes and settings.allow_full_sync,
    )
    async def note_types_update(
        note_type_id: StableId,
        name: ResourceName,
        fields: list[ResourceName],
        templates: list[NoteTypeTemplateInput],
        confirmation_token: ConfirmationToken,
        css: CardText = "",
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Replace note-type data after guarded full-sync maintenance approval."""
        template_values = [template.model_dump() for template in templates]
        request = {
            "note_type_id": note_type_id,
            "name": name,
            "fields": fields,
            "templates": template_values,
            "css": css,
        }
        return await guarded_mutate(
            "anki_note_types_update",
            idempotency_key,
            request,
            confirmation_token,
            {"operation": "update", "request": request},
            lambda adapter: adapter.preview_note_type_change("update", note_type_id),
            lambda adapter: adapter.update_note_type(
                note_type_id, name, fields, template_values, css
            ),
        )

    @scoped_tool(
        name="anki_note_type_fields_update",
        scope="admin",
        enabled=settings.allow_schema_changes and settings.allow_full_sync,
    )
    async def note_type_fields_update(
        note_type_id: StableId,
        mappings: FieldMappings,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Change fields after guarded full-sync maintenance approval."""
        mapping_values = [mapping.model_dump() for mapping in mappings]
        request = {"note_type_id": note_type_id, "mappings": mapping_values}
        return await guarded_mutate(
            "anki_note_type_fields_update",
            idempotency_key,
            request,
            confirmation_token,
            {"operation": "fields_update", "request": request},
            lambda adapter: adapter.preview_note_type_change("fields_update", note_type_id),
            lambda adapter: adapter.update_note_type_fields(note_type_id, mapping_values),
        )

    @scoped_tool(
        name="anki_templates_update",
        scope="admin",
        enabled=settings.allow_schema_changes and settings.allow_full_sync,
    )
    async def templates_update(
        note_type_id: StableId,
        mappings: TemplateMappings,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Change templates after guarded full-sync maintenance approval."""
        mapping_values = [mapping.model_dump() for mapping in mappings]
        request = {"note_type_id": note_type_id, "mappings": mapping_values}
        return await guarded_mutate(
            "anki_templates_update",
            idempotency_key,
            request,
            confirmation_token,
            {"operation": "templates_update", "request": request},
            lambda adapter: adapter.preview_note_type_change("templates_update", note_type_id),
            lambda adapter: adapter.update_templates(note_type_id, mapping_values),
        )

    @scoped_tool(
        name="anki_note_types_delete",
        scope="destructive",
        enabled=(
            settings.allow_destructive
            and settings.allow_schema_changes
            and settings.allow_full_sync
        ),
    )
    async def note_types_delete(
        note_type_id: StableId,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Delete a note type after preview, backup, and full-sync maintenance gating."""
        request = {"note_type_id": note_type_id}
        return await guarded_mutate(
            "anki_note_types_delete",
            idempotency_key,
            request,
            confirmation_token,
            {"operation": "delete", "request": request},
            lambda adapter: adapter.preview_note_type_change("delete", note_type_id),
            lambda adapter: adapter.delete_note_type(note_type_id),
        )

    @scoped_tool(name="anki_media_list", scope="read")
    async def media_list(
        offset: Offset = 0,
        limit: PageLimit = settings.max_page_size,
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """List collection media filenames and sizes with bounded pagination."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.list_media(offset, limit),
                sync_before,
                sync_media=True,
            )
        )

    @scoped_tool(name="anki_media_get", scope="read")
    async def media_get(filename: MediaFilename, sync_before: SyncMedia = False) -> dict[str, Any]:
        """Read one bounded media file as base64 by safe filename."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.get_media(filename), sync_before, sync_media=True
            )
        )

    @scoped_tool(name="anki_media_check", scope="read")
    async def media_check(
        offset: Offset = 0,
        limit: PageLimit = settings.max_page_size,
        sync_before: SyncMedia = False,
    ) -> dict[str, Any]:
        """Report bounded missing and unused media after optional synchronization."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.check_media(offset, limit),
                sync_before,
                sync_media=True,
            )
        )

    @scoped_tool(name="anki_media_store", scope="write")
    async def media_store(
        filename: MediaFilename,
        content_base64: MediaContent,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Create or replace one bounded media file from base64 content."""
        return await mutate(
            "anki_media_store",
            idempotency_key,
            {"filename": filename, "content_base64": content_base64},
            lambda adapter: adapter.store_media(filename, content_base64),
            sync_media=True,
        )

    @scoped_tool(name="anki_media_rename", scope="write")
    async def media_rename(
        old_filename: MediaFilename,
        new_filename: MediaFilename,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Rename one bounded media file without overwriting an existing target."""
        return await mutate(
            "anki_media_rename",
            idempotency_key,
            {"old_filename": old_filename, "new_filename": new_filename},
            lambda adapter: adapter.rename_media(old_filename, new_filename),
            sync_media=True,
        )

    @scoped_tool(
        name="anki_media_delete_preview",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def media_delete_preview(filename: MediaFilename) -> dict[str, Any]:
        """Preview the bounded media file that will be moved to Anki's trash."""
        request = {"filename": filename}
        return await preview(
            "anki_media_delete", request, lambda adapter: adapter.preview_media_delete(filename)
        )

    @scoped_tool(
        name="anki_media_delete",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def media_delete(
        filename: MediaFilename,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Move media to trash after a matching impact preview and required backup."""
        request = {"filename": filename}
        return await guarded_mutate(
            "anki_media_delete",
            idempotency_key,
            request,
            confirmation_token,
            request,
            lambda adapter: adapter.preview_media_delete(filename),
            lambda adapter: adapter.delete_media(filename),
            sync_media=True,
        )

    @scoped_tool(
        name="anki_media_empty_trash_preview",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def media_empty_trash_preview(
        offset: Offset = 0, limit: PageLimit = settings.max_page_size
    ) -> dict[str, Any]:
        """Preview the trashed media files that will be permanently deleted.

        ``items`` lists the trashed files (filename, size_bytes) for the given
        offset/limit page; ``state_fingerprint`` covers the entire trash, so it is
        unchanged by the paging arguments and always binds the full contents.
        """
        request: dict[str, Any] = {"offset": offset, "limit": limit}
        return await preview(
            "anki_media_empty_trash",
            request,
            lambda adapter: adapter.preview_media_empty_trash(offset, limit),
        )

    @scoped_tool(
        name="anki_media_empty_trash",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def media_empty_trash(
        confirmation_token: ConfirmationToken,
        offset: Offset = 0,
        limit: PageLimit = settings.max_page_size,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Permanently delete trashed media after a matching impact preview and required backup.

        Use the same offset/limit as the preview so the re-computed impact matches.
        The verified backup contains the collection database only, so emptied media cannot be
        restored.
        """
        request: dict[str, Any] = {"offset": offset, "limit": limit}
        return await guarded_mutate(
            "anki_media_empty_trash",
            idempotency_key,
            request,
            confirmation_token,
            request,
            lambda adapter: adapter.preview_media_empty_trash(offset, limit),
            lambda adapter: adapter.empty_media_trash(),
            sync_media=True,
        )

    @scoped_tool(
        name="anki_maintenance_check_database_preview",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def maintenance_check_database_preview() -> dict[str, Any]:
        """Preview a read-only collection snapshot (card and note counts).

        Anki's database check also repairs, so the integrity problems are not reported here;
        the apply performs the check-and-repair. Anki's check rebuilds the tag registry from
        the notes, so the preview also lists the unused tags (those no note references) that
        the apply will remove.
        """
        request: dict[str, Any] = {}
        return await preview(
            "anki_maintenance_check_database",
            request,
            lambda adapter: adapter.preview_check_database(),
        )

    @scoped_tool(
        name="anki_maintenance_check_database",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def maintenance_check_database(
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Check and repair the collection database under the preview token and pre-op backup.

        The result reports the unused tags Anki's repair removed (those no note references).
        """
        request: dict[str, Any] = {}
        return await guarded_mutate(
            "anki_maintenance_check_database",
            idempotency_key,
            request,
            confirmation_token,
            request,
            lambda adapter: adapter.preview_check_database(),
            lambda adapter: adapter.check_database(),
        )

    @scoped_tool(
        name="anki_maintenance_empty_cards_preview",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def maintenance_empty_cards_preview() -> dict[str, Any]:
        """Preview the empty cards that will be deleted from the collection."""
        request: dict[str, Any] = {}
        return await preview(
            "anki_maintenance_empty_cards",
            request,
            lambda adapter: adapter.preview_empty_cards(),
        )

    @scoped_tool(
        name="anki_maintenance_empty_cards",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def maintenance_empty_cards(
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Delete empty cards and orphaned notes after a matching impact preview."""
        request: dict[str, Any] = {}
        return await guarded_mutate(
            "anki_maintenance_empty_cards",
            idempotency_key,
            request,
            confirmation_token,
            request,
            lambda adapter: adapter.preview_empty_cards(),
            lambda adapter: adapter.empty_cards(),
        )

    @scoped_tool(
        name="anki_maintenance_optimize_preview",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def maintenance_optimize_preview() -> dict[str, Any]:
        """Preview collection counts before optimizing the database."""
        request: dict[str, Any] = {}
        return await preview(
            "anki_maintenance_optimize",
            request,
            lambda adapter: adapter.preview_optimize_database(),
        )

    @scoped_tool(
        name="anki_maintenance_optimize",
        scope="destructive",
        enabled=settings.allow_destructive,
    )
    async def maintenance_optimize(
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey | None = None,
    ) -> dict[str, Any]:
        """Vacuum and analyze the collection database after a matching count preview.

        The apply reports the logical database size before and after optimizing.
        """
        request: dict[str, Any] = {}
        return await guarded_mutate(
            "anki_maintenance_optimize",
            idempotency_key,
            request,
            confirmation_token,
            request,
            lambda adapter: adapter.preview_optimize_database(),
            lambda adapter: adapter.optimize_database(),
        )

    @scoped_tool(name="anki_export_apkg", scope="read")
    async def export_apkg(
        deck_id: StableId | None = None,
        include_media: StrictBool = True,
        include_scheduling: StrictBool = True,
        include_deck_configs: StrictBool = False,
        inline: StrictBool = False,
    ) -> dict[str, Any]:
        """Export the collection or one deck to a generated .apkg file under <parent>/exports.

        With inline=true the package bytes are also returned as content_base64 when they fit
        within MCP_MAX_RESPONSE_BYTES; otherwise inline_omitted is reported.
        """
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.export_apkg(
                    deck_id, include_media, include_scheduling, include_deck_configs, inline
                )
            )
        )

    @scoped_tool(name="anki_export_notes_csv", scope="read")
    async def export_notes_csv(
        deck_id: StableId | None = None,
        with_html: StrictBool = True,
        with_tags: StrictBool = True,
        with_deck: StrictBool = True,
        with_notetype: StrictBool = False,
        with_guid: StrictBool = False,
        inline: StrictBool = False,
    ) -> dict[str, Any]:
        """Export notes to a generated .csv file, optionally inlining small content as base64."""
        return await execute(
            service.coordinated_read(
                lambda adapter: adapter.export_notes_csv(
                    deck_id, with_html, with_tags, with_deck, with_notetype, with_guid, inline
                )
            )
        )

    @scoped_tool(name="anki_import_files_list", scope="read")
    async def import_files_list(
        offset: Offset = 0, limit: PageLimit = settings.max_page_size
    ) -> dict[str, Any]:
        """List staged .apkg and .csv import files, newest first."""
        return await execute(service.list_import_files(offset, limit))

    @scoped_tool(
        name="anki_import_apkg_preview",
        scope="destructive",
        enabled=(
            settings.allow_import and settings.allow_schema_changes and settings.allow_full_sync
        ),
    )
    async def import_apkg_preview(
        filename: ImportFilename | None = None,
        content_base64: InlineImportContent | None = None,
        merge_notetypes: StrictBool = False,
        update_notes: UpdateCondition = "NEVER",
        update_notetypes: UpdateCondition = "NEVER",
        with_scheduling: StrictBool = True,
        with_deck_configs: StrictBool = False,
    ) -> dict[str, Any]:
        """Validate an .apkg and preview its import; issues a confirmation token.

        Supply exactly one of ``filename`` (a file already staged in the imports
        folder) or ``content_base64`` (the package bytes inline). Inline content is
        staged under a content-derived name in the imports folder.
        """
        resolved = await execute(
            service.resolve_import_source(filename, content_base64, ".apkg")
        )
        if not isinstance(resolved, str):  # pragma: no cover - resolver returns a filename
            raise RuntimeError("import source resolution returned an invalid result")
        filename = resolved
        request = {
            "filename": filename,
            "merge_notetypes": merge_notetypes,
            "update_notes": update_notes,
            "update_notetypes": update_notetypes,
            "with_scheduling": with_scheduling,
            "with_deck_configs": with_deck_configs,
        }
        return await preview(
            "anki_import_apkg",
            request,
            lambda adapter: adapter.preview_import_apkg_file(filename),
        )

    @scoped_tool(
        name="anki_import_apkg",
        scope="destructive",
        enabled=(
            settings.allow_import and settings.allow_schema_changes and settings.allow_full_sync
        ),
    )
    async def import_apkg(
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey,
        filename: ImportFilename | None = None,
        content_base64: InlineImportContent | None = None,
        merge_notetypes: StrictBool = False,
        update_notes: UpdateCondition = "NEVER",
        update_notetypes: UpdateCondition = "NEVER",
        with_scheduling: StrictBool = True,
        with_deck_configs: StrictBool = False,
    ) -> dict[str, Any]:
        """Import an .apkg after a matching preview token and verified backup.

        Supply exactly one of ``filename`` or ``content_base64`` (see
        ``anki_import_apkg_preview``); inline content is staged under a
        content-derived name in the imports folder.
        """
        resolved = await execute(
            service.resolve_import_source(filename, content_base64, ".apkg")
        )
        if not isinstance(resolved, str):  # pragma: no cover - resolver returns a filename
            raise RuntimeError("import source resolution returned an invalid result")
        filename = resolved
        request = {
            "filename": filename,
            "merge_notetypes": merge_notetypes,
            "update_notes": update_notes,
            "update_notetypes": update_notetypes,
            "with_scheduling": with_scheduling,
            "with_deck_configs": with_deck_configs,
        }
        return await guarded_mutate(
            "anki_import_apkg",
            idempotency_key,
            request,
            confirmation_token,
            request,
            lambda adapter: adapter.preview_import_apkg_file(filename),
            lambda adapter: adapter.import_apkg(
                filename,
                merge_notetypes,
                UPDATE_CONDITIONS[update_notes],
                UPDATE_CONDITIONS[update_notetypes],
                with_scheduling,
                with_deck_configs,
            ),
        )

    @scoped_tool(
        name="anki_import_csv_preview",
        scope="destructive",
        enabled=settings.allow_import,
    )
    async def import_csv_preview(
        filename: ImportFilename | None = None,
        content_base64: InlineImportContent | None = None,
        delimiter: CsvDelimiter | None = None,
    ) -> dict[str, Any]:
        """Validate a .csv and preview its metadata and sample rows; issues a token.

        Supply exactly one of ``filename`` or ``content_base64``; inline content is
        staged under a content-derived name in the imports folder.
        """
        resolved = await execute(
            service.resolve_import_source(filename, content_base64, ".csv")
        )
        if not isinstance(resolved, str):  # pragma: no cover - resolver returns a filename
            raise RuntimeError("import source resolution returned an invalid result")
        filename = resolved
        request = {"filename": filename, "delimiter": delimiter}
        return await preview(
            "anki_import_csv",
            request,
            lambda adapter: adapter.preview_import_csv(filename, delimiter),
        )

    @scoped_tool(
        name="anki_import_csv",
        scope="destructive",
        enabled=settings.allow_import,
    )
    async def import_csv(
        notetype_id: StableId,
        field_columns: CsvFieldColumns,
        deck_id: StableId,
        confirmation_token: ConfirmationToken,
        idempotency_key: IdempotencyKey,
        filename: ImportFilename | None = None,
        content_base64: InlineImportContent | None = None,
        delimiter: CsvDelimiter | None = None,
        is_html: StrictBool | None = None,
        tags: Tags | None = None,
        dupe_resolution: CsvDupeResolution = "preserve",
    ) -> dict[str, Any]:
        """Import a .csv into a note type and deck after a matching preview token.

        Supply exactly one of ``filename`` or ``content_base64``; inline content is
        staged under a content-derived name in the imports folder. The preview token
        binds the file identity (content digest) and delimiter; the note type, field
        mapping, deck, tags, and duplicate resolution are chosen at apply time. The
        same authenticated caller must perform both preview and apply.
        """
        resolved = await execute(
            service.resolve_import_source(filename, content_base64, ".csv")
        )
        if not isinstance(resolved, str):  # pragma: no cover - resolver returns a filename
            raise RuntimeError("import source resolution returned an invalid result")
        filename = resolved
        normalized_tags = tags or []
        request = {
            "filename": filename,
            "notetype_id": notetype_id,
            "field_columns": field_columns,
            "deck_id": deck_id,
            "delimiter": delimiter,
            "is_html": is_html,
            "tags": normalized_tags,
            "dupe_resolution": dupe_resolution,
        }
        return await guarded_mutate(
            "anki_import_csv",
            idempotency_key,
            request,
            confirmation_token,
            {"filename": filename, "delimiter": delimiter},
            lambda adapter: adapter.preview_import_csv(filename, delimiter),
            lambda adapter: adapter.import_csv(
                filename,
                notetype_id,
                field_columns,
                deck_id,
                delimiter,
                is_html,
                normalized_tags,
                dupe_resolution,
            ),
        )

    @scoped_tool(name="anki_undo_status", scope="read")
    async def undo_status() -> dict[str, Any]:
        """Report the in-memory undo and redo stack heads and, when undo/redo are not
        registered, the reason (for example while ANKI_SYNC_ON_WRITE is enabled)."""
        result = await execute(service.coordinated_read(lambda adapter: adapter.undo_status()))
        if not isinstance(result, dict):  # pragma: no cover - adapter always returns a mapping
            raise RuntimeError("undo status returned an invalid result")
        reasons: list[str] = []
        if settings.sync_on_write:
            reasons.append(
                "ANKI_SYNC_ON_WRITE is enabled and a successful synchronization clears "
                "Anki's in-memory undo history"
            )
        if not settings.allow_undo:
            reasons.append("ANKI_ALLOW_UNDO is disabled")
        if "destructive" not in settings.scopes:
            reasons.append("MCP_SCOPES does not include destructive")
        result["disabled_reason"] = (
            f"undo and redo are not registered: {'; '.join(reasons)}" if reasons else None
        )
        return result

    @scoped_tool(
        name="anki_undo",
        scope="destructive",
        enabled=settings.allow_undo and not settings.sync_on_write,
    )
    async def undo(
        expect_operation: OperationName, idempotency_key: IdempotencyKey
    ) -> dict[str, Any]:
        """Undo the current stack head after confirming it matches expect_operation."""
        return await mutate(
            "anki_undo",
            idempotency_key,
            {"expect_operation": expect_operation},
            lambda adapter: adapter.undo(expect_operation),
        )

    @scoped_tool(
        name="anki_redo",
        scope="destructive",
        enabled=settings.allow_undo and not settings.sync_on_write,
    )
    async def redo(
        expect_operation: OperationName, idempotency_key: IdempotencyKey
    ) -> dict[str, Any]:
        """Redo the current redo stack head after confirming it matches expect_operation."""
        return await mutate(
            "anki_redo",
            idempotency_key,
            {"expect_operation": expect_operation},
            lambda adapter: adapter.redo(expect_operation),
        )

    # FastMCP currently generates argument models with Pydantic's extra="ignore".
    # Tighten both runtime validation and the JSON schemas advertised to clients.
    for tool_name in registered_tool_names:
        registered = mcp._tool_manager.get_tool(tool_name)  # pyright: ignore[reportPrivateUsage]
        if registered is None:  # pragma: no cover
            raise RuntimeError(f"failed to register {tool_name}")
        registered.fn_metadata.arg_model.model_config["extra"] = "forbid"
        registered.fn_metadata.arg_model.model_config["hide_input_in_errors"] = True
        registered.fn_metadata.arg_model.model_rebuild(force=True)
        registered.parameters = registered.fn_metadata.arg_model.model_json_schema()
        properties = registered.parameters.get("properties", {})
        offset_schema = properties.get("offset")
        if offset_schema is not None:
            offset_schema["minimum"] = 0
        limit_schema = properties.get("limit")
        if limit_schema is not None:
            limit_schema["minimum"] = 1
            limit_schema["maximum"] = settings.max_page_size
        for id_name in ("deck_id", "card_id", "note_id", "note_type_id"):
            id_schema = properties.get(id_name)
            if id_schema is not None:
                id_schema["minimum"] = 1

    original_call_tool = mcp._tool_manager.call_tool  # pyright: ignore[reportPrivateUsage]

    async def call_tool_with_stable_validation_errors(
        name: str,
        arguments: dict[str, Any],
        context: Any = None,
        convert_result: bool = False,
    ) -> Any:
        try:
            return await original_call_tool(
                name, arguments, context=context, convert_result=convert_result
            )
        except ToolError as exc:
            if isinstance(exc.__cause__, ValidationError):
                raise_tool_error(
                    "INVALID_ARGUMENT", "tool arguments failed validation", exc.__cause__
                )
            raise

    mcp._tool_manager.call_tool = (  # pyright: ignore[reportPrivateUsage]
        call_tool_with_stable_validation_errors
    )

    mcp_app = mcp.streamable_http_app()

    async def live(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def ready(_: Request) -> Response:
        try:
            current_status = await service.status()
        except Exception:
            current_status = {"ready": False, "readiness_reason": "collection_unavailable"}
        if not current_status["ready"]:
            return JSONResponse(
                {
                    "status": "not_ready",
                    "reason": current_status["readiness_reason"],
                },
                status_code=503,
            )
        return JSONResponse({"status": "ready"})

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncGenerator[None]:
        async with service, mcp_app.router.lifespan_context(mcp_app):
            if settings.bootstrap_mode != "disabled":
                password = (
                    settings.sync_password.get_secret_value()
                    if settings.sync_password is not None
                    else None
                )
                await service.bootstrap(
                    settings.bootstrap_mode,
                    settings.sync_username,
                    password,
                    settings.sync_endpoint,
                )
            yield

    routes = [
        Route("/health/live", live, methods=["GET"]),
        Route("/health/ready", ready, methods=["GET"]),
        *mcp_app.routes,
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.settings = settings
    app.state.collection_service = service
    return BearerAuthMiddleware(
        RequestBodyLimitMiddleware(app, settings.max_request_bytes, settings.mcp_path),
        settings.auth_token.get_secret_value(),
        settings.mcp_path,
    )
