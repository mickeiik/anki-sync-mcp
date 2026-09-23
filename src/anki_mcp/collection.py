from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import threading
import time
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

import anki.lang
from anki._backend import RustBackend
from anki.collection import AddNoteRequest, Collection, DeckIdLimit
from anki.config_pb2 import ConfigKey
from anki.consts import CARD_TYPE_REV, QUEUE_TYPE_SUSPENDED
from anki.decks import UpdateDeckConfigs
from anki.errors import (
    BackendError,
    InvalidInput,
    NetworkError,
    NotFoundError,
    SearchError,
    UndoEmpty,
)
from anki.import_export_pb2 import (
    CsvMetadata,
    ExportAnkiPackageOptions,
    ImportAnkiPackageOptions,
    ImportAnkiPackageRequest,
    ImportAnkiPackageUpdateCondition,
    ImportCsvRequest,
)
from anki.media import media_paths_from_col_path
from anki.scheduler_pb2 import SimulateFsrsReviewRequest
from anki.sync import SyncAuth
from anki.utils import field_checksum
from google.protobuf.json_format import MessageToDict

from anki_mcp.config import validate_sync_migration_endpoint
from anki_mcp.state import PersistentState

if TYPE_CHECKING:
    from anki.cards import CardId
    from anki.decks import DeckConfigId, DeckId
    from anki.models import NotetypeId
    from anki.notes import NoteId

T = TypeVar("T")
AUDIT_LOGGER = logging.getLogger("anki_mcp.audit")
SYNC_REQUIRED_NAMES = (
    "NO_CHANGES",
    "NORMAL_SYNC",
    "FULL_SYNC",
    "FULL_DOWNLOAD",
    "FULL_UPLOAD",
)
IMPORT_SUFFIXES = (".apkg", ".csv")
CSV_DELIMITER_CHARS: dict[str, CsvMetadata.Delimiter.ValueType] = {
    "\t": CsvMetadata.Delimiter.TAB,
    "|": CsvMetadata.Delimiter.PIPE,
    ";": CsvMetadata.Delimiter.SEMICOLON,
    ":": CsvMetadata.Delimiter.COLON,
    ",": CsvMetadata.Delimiter.COMMA,
    " ": CsvMetadata.Delimiter.SPACE,
}
CSV_DELIMITER_NAMES: dict[int, str] = {
    value: char for char, value in CSV_DELIMITER_CHARS.items()
}
CSV_DUPE_RESOLUTIONS: dict[str, CsvMetadata.DupeResolution.ValueType] = {
    "preserve": CsvMetadata.DupeResolution.PRESERVE,
    "update": CsvMetadata.DupeResolution.UPDATE,
    "duplicate": CsvMetadata.DupeResolution.DUPLICATE,
}
UPDATE_CONDITIONS: dict[str, ImportAnkiPackageUpdateCondition.ValueType] = {
    "ALWAYS": ImportAnkiPackageUpdateCondition.Value(
        "IMPORT_ANKI_PACKAGE_UPDATE_CONDITION_ALWAYS"
    ),
    "IF_NEWER": ImportAnkiPackageUpdateCondition.Value(
        "IMPORT_ANKI_PACKAGE_UPDATE_CONDITION_IF_NEWER"
    ),
    "NEVER": ImportAnkiPackageUpdateCondition.Value(
        "IMPORT_ANKI_PACKAGE_UPDATE_CONDITION_NEVER"
    ),
}
UNDO_HISTORY_NOTE = (
    "Undo/redo history is held in memory for the running collection process; it is cleared by "
    "successful synchronization and is not durable across restarts or full downloads."
)
DECK_PRESET_SECTIONS: dict[str, tuple[str, ...]] = {
    "learning": (
        "learn_steps",
        "relearn_steps",
        "initial_ease",
        "graduating_interval_good",
        "graduating_interval_easy",
    ),
    "new_cards": (
        "new_per_day_minimum",
        "new_card_insert_order",
        "new_card_gather_priority",
        "new_card_sort_order",
        "new_mix",
    ),
    "reviews": (
        "easy_multiplier",
        "hard_multiplier",
        "lapse_multiplier",
        "interval_multiplier",
        "maximum_review_interval",
        "minimum_lapse_interval",
        "review_order",
        "interday_learning_mix",
    ),
    "lapses": ("leech_action", "leech_threshold"),
    "burying": ("bury_new", "bury_reviews", "bury_interday_learning"),
    "display_audio": (
        "disable_autoplay",
        "show_timer",
        "stop_timer_on_answer",
        "seconds_to_show_question",
        "seconds_to_show_answer",
        "question_action",
        "answer_action",
        "wait_for_audio",
        "skip_question_when_replaying_answer",
    ),
    "fsrs": (
        "fsrs_params_4",
        "fsrs_params_5",
        "fsrs_params_6",
        "historical_retention",
        "ignore_revlogs_before_date",
        "param_search",
    ),
    "easy_days": ("easy_days_percentages",),
}


class SyncLoginRequiredError(RuntimeError):
    """Raised when synchronization is requested before remote login."""


class FullSyncRequiredError(RuntimeError):
    """Raised when a normal operation encounters a one-way sync requirement."""


class DuplicateNoteError(ValueError):
    """Raised when a note request would create a duplicate."""


class IdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused with different content."""


class ResourceLimitError(ValueError):
    """Raised when a bounded operation proves its configured work limit was exceeded."""


class ImportFileError(ValueError):
    """Raised when a staged import file cannot be read or validated as expected."""


IMPORT_FILE_FAILURES = (
    BackendError,
    OSError,
    UnicodeDecodeError,
    NotImplementedError,
    RuntimeError,
    zlib.error,
)


class MediaSyncFailedError(TimeoutError):
    """Raised when requested media synchronization does not complete safely."""


class BackupFailedError(RuntimeError):
    """Raised when a guarded mutation cannot obtain a verified current backup."""


class RestoreFailedError(RuntimeError):
    """Raised when applying a backup fails and the live collection may be unusable."""


class CollectionAdapter:
    """Synchronous adapter; an instance must only be used by its owning thread."""

    def __init__(
        self,
        path: str | Path,
        max_page_size: int,
        max_search_scan: int = 10_000,
        max_rendered_field_bytes: int = 262_144,
        max_card_fields: int = 100,
        max_batch_size: int = 50,
        max_media_bytes: int = 1_048_576,
        sync_timeout_seconds: float = 300,
        sync_on_read: bool = False,
        sync_on_write: bool = False,
        max_response_bytes: int = 1_048_576,
        max_import_bytes: int = 268_435_456,
    ) -> None:
        if anki.lang.current_i18n is None:
            anki.lang.set_lang("en_US")
        self.collection = Collection(str(path))
        self.max_page_size = max_page_size
        self.max_search_scan = max_search_scan
        self.max_rendered_field_bytes = max_rendered_field_bytes
        self.max_card_fields = max_card_fields
        self.max_batch_size = max_batch_size
        self.max_media_bytes = max_media_bytes
        self.sync_timeout_seconds = sync_timeout_seconds
        self.sync_on_read = sync_on_read
        self.sync_on_write = sync_on_write
        self.max_response_bytes = max_response_bytes
        self.max_import_bytes = max_import_bytes
        self._backup_folder = Path(path).parent / "backups"
        self._exports_folder = Path(path).parent / "exports"
        self._imports_folder = Path(path).parent / "imports"
        self._imports_folder.mkdir(parents=True, exist_ok=True)
        self._state = PersistentState(path)
        persisted_auth = self._state.load_sync_auth()
        self._sync_auth = (
            SyncAuth(
                hkey=str(persisted_auth["hkey"]),
                endpoint=str(persisted_auth.get("endpoint") or ""),
            )
            if persisted_auth
            else None
        )
        self._configured_sync_endpoint = (
            str(persisted_auth.get("configured_endpoint") or "") or None if persisted_auth else None
        )
        status = self._state.load_status()
        pending = status.get("pending_full_sync")
        self._pending_full_sync = (
            (int(pending["required"]), pending.get("server_media_usn"))
            if isinstance(pending, dict)
            else None
        )
        self._last_sync_at = status.get("last_sync_at")
        self._last_media_sync_at = status.get("last_media_sync_at")
        self._media_sync_progress = status.get("media_sync_progress")
        self._post_restore_upload = bool(status.get("post_restore_upload", False))

    def close(self) -> None:
        try:
            self.collection.close()
        finally:
            self._state.close()

    def check_ready(self) -> bool:
        if self.collection.db is None:
            return False
        return self.collection.db.scalar("select 1") == 1

    @staticmethod
    def _impact_fingerprint(value: Any) -> str:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _save_operational_status(self) -> None:
        pending = None
        if self._pending_full_sync is not None:
            pending = {
                "required": self._pending_full_sync[0],
                "server_media_usn": self._pending_full_sync[1],
            }
        self._state.save_status(
            {
                "last_sync_at": self._last_sync_at,
                "last_media_sync_at": self._last_media_sync_at,
                "media_sync_progress": self._media_sync_progress,
                "pending_full_sync": pending,
                "post_restore_upload": self._post_restore_upload,
            }
        )

    def _invalidate_sync_auth(self) -> None:
        self._sync_auth = None
        self._configured_sync_endpoint = None
        self._pending_full_sync = None
        self._state.clear_sync_auth()
        self._save_operational_status()

    def status(self) -> dict[str, Any]:
        collection_ready = self.check_ready()
        pending_name = (
            SYNC_REQUIRED_NAMES[self._pending_full_sync[0]]
            if self._pending_full_sync is not None
            else None
        )
        authenticated = self._sync_auth is not None
        ready = collection_ready and authenticated and pending_name is None
        if not collection_ready:
            reason = "collection_unavailable"
        elif pending_name is not None:
            reason = "full_sync_required"
        elif not authenticated:
            reason = "authentication_required"
        else:
            reason = None
        return {
            "anki_version": version("anki"),
            "collection_ready": collection_ready,
            "authenticated": authenticated,
            "ready": ready,
            "readiness_reason": reason,
            "pending_full_sync": pending_name,
            "post_restore_upload_pending": bool(self._post_restore_upload),
            "last_sync_at": self._last_sync_at,
            "last_media_sync_at": self._last_media_sync_at,
            "media_sync_progress": self._media_sync_progress,
            "pending_mutations": self._state.pending_receipt_count(),
        }

    def get_operation(self, idempotency_key: str) -> dict[str, Any]:
        operation = self._state.get_operation(idempotency_key)
        if operation is None:
            raise LookupError(f"operation {idempotency_key} not found")
        return operation

    def list_operations(self, offset: int, limit: int) -> dict[str, Any]:
        self._page([], offset, limit)
        items, total = self._state.list_operations(offset, limit)
        return {
            "items": items,
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + limit < total,
        }

    def metrics(self) -> dict[str, Any]:
        metrics = self._state.metrics()
        metrics["sync"] = {
            "last_successful_collection_sync_at": self._last_sync_at,
            "last_successful_media_sync_at": self._last_media_sync_at,
            "media_progress": self._media_sync_progress,
            "pending_full_sync": (
                SYNC_REQUIRED_NAMES[self._pending_full_sync[0]]
                if self._pending_full_sync is not None
                else None
            ),
        }
        return metrics

    def create_backup(self) -> dict[str, Any]:
        self._backup_folder.mkdir(parents=True, exist_ok=True)
        existing = {
            path: (
                path.stat().st_dev,
                path.stat().st_ino,
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in self._backup_folder.glob("*.colpkg")
        }
        try:
            native_created = self.collection.create_backup(
                backup_folder=str(self._backup_folder), force=True, wait_for_completion=True
            )
        except Exception as exc:
            raise BackupFailedError(f"native backup failed: {exc}") from exc
        candidates = set(self._backup_folder.glob("*.colpkg"))
        fresh = {
            path
            for path in candidates
            if path not in existing
            or existing[path]
            != (
                path.stat().st_dev,
                path.stat().st_ino,
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
        }
        selected = fresh if native_created else candidates
        path = next(
            (
                candidate
                for candidate in sorted(
                    selected, key=lambda item: item.stat().st_mtime_ns, reverse=True
                )
                if self._is_valid_backup(candidate)
            ),
            None,
        )
        return {
            "requested": True,
            "created": bool(native_created and path is not None),
            "path": str(path) if path is not None else None,
        }

    def _is_valid_backup(self, path: Path) -> bool:
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            return False
        try:
            with ZipFile(path) as archive:
                members = set(archive.namelist())
                required = {"meta", "collection.anki21b", "collection.anki2", "media"}
                if not required <= members or archive.testzip() is not None:
                    return False
            with TemporaryDirectory(prefix="anki-mcp-backup-check-") as temporary:
                restore_folder = Path(temporary)
                media_folder = restore_folder / "media"
                media_folder.mkdir()
                restored_collection = restore_folder / "collection.anki2"
                RustBackend().import_collection_package(
                    col_path=str(restored_collection),
                    backup_path=str(path),
                    media_folder=str(media_folder),
                    media_db=str(restore_folder / "media.db2"),
                )
                return restored_collection.is_file() and restored_collection.stat().st_size > 0
        except Exception:
            return False

    def backup_before(self, mutation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Obtain a verified current backup immediately before a guarded mutation."""
        backup = self.create_backup()
        path = backup.get("path")
        backup_path = Path(path) if isinstance(path, str) else None
        if backup_path is None:
            raise BackupFailedError("required current pre-operation backup is unavailable")
        return {**mutation(), "backup": backup}

    def list_backups(self, offset: int, limit: int) -> dict[str, Any]:
        """List validation-agnostic backup metadata, newest first."""
        items: list[dict[str, Any]] = []
        if self._backup_folder.is_dir():
            for path in self._backup_folder.glob("*.colpkg"):
                if path.is_file() and not path.is_symlink():
                    stat = path.stat()
                    items.append(
                        {
                            "filename": path.name,
                            "size_bytes": stat.st_size,
                            "mtime": stat.st_mtime,
                        }
                    )
        items.sort(key=lambda item: item["mtime"], reverse=True)
        return self._page(items, offset, limit)

    def _backup_path(self, filename: str, *, require_exists: bool) -> Path:
        if (
            "\x00" in filename
            or not filename.strip()
            or Path(filename).name != filename
            or any(separator in filename for separator in ("/", "\\"))
        ):
            raise ValueError("backup filename must be a plain filename without path separators")
        if not filename.endswith(".colpkg"):
            raise ValueError("backup filename must end with .colpkg")
        path = self._backup_folder / filename
        if path.is_symlink():
            raise ValueError("backup filename must not reference a symbolic link")
        if path.resolve().parent != self._backup_folder.resolve():
            raise ValueError("backup filename must resolve inside the backup folder")
        if require_exists and not path.is_file():
            raise LookupError(f"backup {filename} not found")
        return path

    def preview_backup_restore(self, filename: str) -> dict[str, Any]:
        """Validate a named backup and report the impact of restoring it."""
        path = self._backup_path(filename, require_exists=True)
        if not self._is_valid_backup(path):
            raise ValueError("backup is not a valid Anki collection package")
        stat = path.stat()
        return {
            "filename": filename,
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "validated": True,
            "current": {
                "cards": int(self.collection.card_count()),
                "notes": int(self.collection.note_count()),
                "decks": int(self.collection.decks.count()),
            },
            "post_restore_modes": ["upload_now", "local_only"],
        }

    def restore_backup(self, filename: str, mode: str) -> dict[str, Any]:
        """Replace the live collection with a verified backup, then report recovery state."""
        if mode not in {"upload_now", "local_only"}:
            raise ValueError("mode must be upload_now or local_only")
        if mode == "upload_now" and self._sync_auth is None:
            raise SyncLoginRequiredError(
                "sync login is required before uploading a restored collection"
            )
        path = self._backup_path(filename, require_exists=True)
        if not self._is_valid_backup(path):
            raise ValueError("backup is not a valid Anki collection package")
        media_folder = self.collection.media.dir()
        media_db = media_paths_from_col_path(self.collection.path)[1]
        # Snapshot the requested package first: the pre-restore backup can reuse or
        # overwrite a same-second native filename, and the restore must always apply
        # exactly the content that was previewed.
        with TemporaryDirectory(prefix="anki-mcp-restore-") as temporary:
            snapshot = Path(temporary) / "target.colpkg"
            shutil.copy2(path, snapshot)
            target_before = path.stat()
            backup = self.create_backup()
            if not isinstance(backup.get("path"), str):
                raise BackupFailedError(
                    "required current pre-restore backup is unavailable"
                )
            pre_restore_backup = str(backup["path"])
            try:
                target_after = path.stat()
            except FileNotFoundError:
                target_after = None
            target_overwritten = target_after is None or (
                target_before.st_dev,
                target_before.st_ino,
                target_before.st_size,
                target_before.st_mtime_ns,
            ) != (
                target_after.st_dev,
                target_after.st_ino,
                target_after.st_size,
                target_after.st_mtime_ns,
            )
            self.collection.close()
            try:
                RustBackend().import_collection_package(
                    col_path=self.collection.path,
                    backup_path=str(snapshot),
                    media_folder=media_folder,
                    media_db=media_db,
                )
                self.collection.reopen()
                self.collection._load_scheduler()  # pyright: ignore[reportPrivateUsage]
            except Exception as exc:
                try:
                    self.collection.reopen()
                except Exception:
                    reopen_failed = True
                else:
                    reopen_failed = False
                message = (
                    "restore failed; the live collection may have been replaced or emptied — "
                    f"restore the collection from the pre-restore backup at {pre_restore_backup}"
                )
                if reopen_failed:
                    message += "; the sidecar must be restarted before further use"
                raise RestoreFailedError(message) from exc
        self._pending_full_sync = None
        self._state.mark_pending_discarded_by_restore()
        self._post_restore_upload = True
        self._save_operational_status()

        overwrite_warning = (
            "the pre-restore backup now occupies the requested filename; original file "
            "content was preserved for the restore"
        )
        if mode == "local_only":
            server_sync_required: str | None = None
            if self._sync_auth is None:
                warning = (
                    "sync login is required to inspect the server; it still holds its pre-restore "
                    "state — once logged in, use anki_sync_full_upload(force=true) to replace the "
                    "server instead"
                )
            else:
                probe_failed = False
                try:
                    server_sync_required = SYNC_REQUIRED_NAMES[
                        self.collection.sync_status(self._sync_auth).required
                    ]
                except NetworkError:
                    probe_failed = True
                if probe_failed:
                    warning = (
                        "the server's state could not be checked; it still holds its pre-restore "
                        "state — use anki_sync_full_upload(force=true) to replace the server "
                        "instead of letting a normal sync merge"
                    )
                elif server_sync_required in {"FULL_SYNC", "FULL_DOWNLOAD", "FULL_UPLOAD"}:
                    warning = (
                        f"the server requires {server_sync_required}; a normal sync will be "
                        "refused — use anki_sync_full_upload(force=true) to replace the server"
                    )
                elif server_sync_required == "NORMAL_SYNC":
                    warning = (
                        "a normal sync would merge the server's newer changes onto this backup — "
                        "use anki_sync_full_upload(force=true) to overwrite the server instead"
                    )
                else:
                    warning = (
                        "the server reports no changes relative to this backup; a normal sync "
                        "should be a no-op — use anki_sync_full_upload(force=true) to replace the "
                        "server anyway"
                    )
            if target_overwritten:
                warning = f"{warning}; {overwrite_warning}"
            return {
                "restored": True,
                "mode": mode,
                "remote_replaced": False,
                "server_sync_required": server_sync_required,
                "target_overwritten": target_overwritten,
                "backup": backup,
                "warning": warning,
            }

        try:
            self.collection.full_upload_or_download(
                auth=self._sync_auth, server_usn=None, upload=True
            )
        except Exception as exc:
            if not isinstance(exc, NetworkError):
                self._invalidate_sync_auth()
            raise
        self._post_restore_upload = False
        self._last_sync_at = datetime.now(UTC).isoformat()
        self._state.mark_all_remote_synced()
        self._save_operational_status()
        media_warning = (
            "the remote collection was replaced but remote media was not transferred; run "
            "anki_sync(sync_media=true) to push media"
        )
        return {
            "restored": True,
            "mode": mode,
            "remote_replaced": True,
            "media_sync_required": True,
            "server_sync_required": None,
            "target_overwritten": target_overwritten,
            "backup": backup,
            "warning": (
                f"{media_warning}; {overwrite_warning}" if target_overwritten else media_warning
            ),
        }

    def bootstrap(
        self,
        mode: str,
        username: str,
        password: str | None,
        endpoint: str | None,
    ) -> dict[str, Any]:
        """Perform an explicitly configured, download-only initialization of an empty client."""
        if mode == "disabled":
            return {"bootstrapped": False, "reason": "disabled"}
        if mode != "download_if_empty":
            raise ValueError("unsupported bootstrap mode")
        if self.collection.note_count() or self.collection.card_count():
            raise ValueError("download_if_empty bootstrap requires an empty local collection")
        if self._sync_auth is None:
            if not username.strip() or password is None:
                raise SyncLoginRequiredError(
                    "bootstrap requires sync credentials or persisted authentication"
                )
            self.sync_login(username, password, endpoint)
        sync_result = self.sync(sync_media=False)
        required = sync_result["required"]
        if required in {"FULL_SYNC", "FULL_DOWNLOAD"}:
            result = self.full_sync(upload=False)
            return {"bootstrapped": True, **result}
        if required == "FULL_UPLOAD":
            raise FullSyncRequiredError(
                "FULL_UPLOAD cannot be resolved by download_if_empty bootstrap"
            )
        return {"bootstrapped": True, "direction": None, "sync": sync_result}

    def coordinated_read(
        self,
        read: Callable[[CollectionAdapter], T],
        sync_before: bool,
        sync_media: bool = False,
    ) -> T:
        if sync_before or self.sync_on_read:
            self._sync_or_raise_full_sync(sync_media=sync_media)
        return read(self)

    def coordinated_mutation(
        self,
        operation: str,
        idempotency_key: str,
        request: dict[str, Any],
        mutate: Callable[[CollectionAdapter], dict[str, Any]],
        sync_media: bool = False,
        sync_after: bool = True,
    ) -> dict[str, Any]:
        started = time.monotonic()

        def audit(receipt: dict[str, Any]) -> None:
            target_count = max(
                (
                    len(value)
                    for key, value in request.items()
                    if key.endswith("_ids") and isinstance(value, list)
                ),
                default=1,
            )
            AUDIT_LOGGER.info(
                json.dumps(
                    {
                        "event": "anki_mutation",
                        "tool": operation,
                        "operation_id": hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[
                            :16
                        ],
                        "target_count": target_count,
                        "local_committed": receipt.get("local_committed"),
                        "remote_synced": receipt.get("remote_synced"),
                        "media_synced": receipt.get("media_synced"),
                        "retryable": receipt.get("retryable"),
                        "duration_ms": round((time.monotonic() - started) * 1000, 3),
                    },
                    separators=(",", ":"),
                )
            )

        if not idempotency_key or len(idempotency_key.encode("utf-8")) > 128:
            raise ValueError("idempotency_key must contain 1 to 128 UTF-8 bytes")
        request_hash = self._state.request_hash(operation, request)
        existing = self._state.get_receipt(idempotency_key)
        if existing is not None:
            recorded_operation, recorded_hash, receipt = existing
            if recorded_operation != operation or recorded_hash != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used with different content"
                )
            media_pending = receipt.get("media_synced") is False
            if (
                receipt.get("local_committed")
                and receipt.get("retryable")
                and (not receipt.get("remote_synced") or media_pending)
            ):
                try:
                    self._sync_or_raise_full_sync(sync_media=sync_media or media_pending)
                except FullSyncRequiredError:
                    receipt["retryable"] = False
                    self._state.put_receipt(idempotency_key, operation, request_hash, receipt)
                    raise
                receipt["remote_synced"] = True
                if media_pending:
                    receipt["media_synced"] = True
                receipt["retryable"] = False
                self._state.put_receipt(idempotency_key, operation, request_hash, receipt)
            audit(receipt)
            return receipt

        if sync_after and self.sync_on_write:
            self._sync_or_raise_full_sync(sync_media=sync_media)
        intent: dict[str, Any] = {
            "idempotency_key": idempotency_key,
            "state": "outcome_unknown",
            "local_committed": None,
            "remote_synced": None if not sync_after else False,
            "media_synced": (False if sync_media else None) if sync_after else None,
            "retryable": False,
            "result": None,
        }
        self._state.put_receipt(idempotency_key, operation, request_hash, intent)
        try:
            result = mutate(self)
        except (
            ValueError,
            LookupError,
            BackupFailedError,
            UndoEmpty,
            SyncLoginRequiredError,
            RestoreFailedError,
        ):
            self._state.delete_receipt(idempotency_key)
            raise
        receipt: dict[str, Any] = {
            "idempotency_key": idempotency_key,
            "state": "committed",
            "local_committed": True,
            "remote_synced": None if not sync_after else not self.sync_on_write,
            "media_synced": (
                (not self.sync_on_write if sync_media else None) if sync_after else None
            ),
            "retryable": False,
            "result": result,
        }
        self._state.put_receipt(idempotency_key, operation, request_hash, receipt)
        if sync_after and self.sync_on_write:
            try:
                self._sync_or_raise_full_sync(sync_media=sync_media)
            except FullSyncRequiredError:
                receipt["remote_synced"] = False
                self._state.put_receipt(idempotency_key, operation, request_hash, receipt)
                raise
            except Exception:
                receipt["remote_synced"] = False
                receipt["retryable"] = True
                self._state.put_receipt(idempotency_key, operation, request_hash, receipt)
                audit(receipt)
                return receipt
            receipt["remote_synced"] = True
            if sync_media:
                receipt["media_synced"] = True
            self._state.put_receipt(idempotency_key, operation, request_hash, receipt)
        audit(receipt)
        return receipt

    def _sync_or_raise_full_sync(self, sync_media: bool) -> dict[str, Any]:
        result = self.sync(sync_media)
        if result["required"] in {"FULL_SYNC", "FULL_DOWNLOAD", "FULL_UPLOAD"}:
            raise FullSyncRequiredError(
                f"{result['required']} requires operator-controlled full synchronization"
            )
        return result

    def _page(self, values: list[Any], offset: int, limit: int) -> dict[str, Any]:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit < 1 or limit > self.max_page_size:
            raise ValueError(f"limit must be between 1 and {self.max_page_size}")
        total = len(values)
        return {
            "items": values[offset : offset + limit],
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + limit < total,
        }

    def list_decks(self, offset: int, limit: int) -> dict[str, Any]:
        if self.collection.decks.count() > self.max_search_scan:
            raise ValueError(
                "collection exceeds MCP_MAX_SEARCH_SCAN; use a larger configured bound"
            )
        decks = []
        for item in sorted(self.collection.decks.all_names_and_ids(), key=lambda d: d.name):
            name, name_truncated = self._truncate_rendered(item.name)
            decks.append(
                {
                    "id": int(item.id),
                    "name": name,
                    "name_truncated": name_truncated,
                    "hierarchy": name.split("::"),
                }
            )
        return self._page(decks, offset, limit)

    def get_deck(self, deck_id: int) -> dict[str, Any]:
        deck = self.collection.decks.get(cast("DeckId", deck_id), default=False)
        if deck is None:
            raise LookupError(f"deck {deck_id} not found")
        name, name_truncated = self._truncate_rendered(str(deck["name"]))
        description, description_truncated = self._truncate_rendered(str(deck.get("desc", "")))
        result: dict[str, Any] = {
            "id": int(deck["id"]),
            "name": name,
            "name_truncated": name_truncated,
            "hierarchy": name.split("::"),
            "modified": int(deck.get("mod", 0)),
            "description": description,
            "description_truncated": description_truncated,
            "dynamic": bool(deck.get("dyn", 0)),
        }
        if not result["dynamic"]:
            result["config_id"] = int(deck.get("conf", 1))
            config = self.collection.decks.config_dict_for_deck_id(cast("DeckId", deck_id))
            config_name, config_name_truncated = self._truncate_rendered(
                str(config.get("name", ""))
            )
            result["config"] = {
                "name": config_name,
                "name_truncated": config_name_truncated,
                "new_cards_per_day": int(config.get("new", {}).get("perDay", 0)),
                "reviews_per_day": int(config.get("rev", {}).get("perDay", 0)),
                "max_answer_seconds": int(config.get("maxTaken", 0)),
                "desired_retention": float(config.get("desiredRetention", 0.9)),
            }
        return result

    def create_deck(self, name: str) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("deck name must not be blank")
        existing = self.collection.decks.id_for_name(name)
        if existing is not None:
            return {"id": int(existing), "created": False}
        deck_id = self.collection.decks.id(name)
        if deck_id is None:  # pragma: no cover - creation returns an ID
            raise RuntimeError("Anki did not return a deck ID")
        return {"id": int(deck_id), "created": True}

    def update_deck(self, deck_id: int, name: str) -> dict[str, Any]:
        self.get_deck(deck_id)
        if not name.strip():
            raise ValueError("deck name must not be blank")
        self.collection.decks.rename(cast("DeckId", deck_id), name)
        return {"id": deck_id, "updated": True}

    def update_deck_config(
        self,
        deck_id: int,
        new_cards_per_day: int | None = None,
        reviews_per_day: int | None = None,
        max_answer_seconds: int | None = None,
        desired_retention: float | None = None,
    ) -> dict[str, Any]:
        deck = self.collection.decks.get(cast("DeckId", deck_id), default=False)
        if deck is None:
            raise LookupError(f"deck {deck_id} not found")
        if bool(deck.get("dyn", 0)):
            raise ValueError("dynamic decks do not have an editable deck configuration")
        updates = (
            new_cards_per_day,
            reviews_per_day,
            max_answer_seconds,
            desired_retention,
        )
        if all(value is None for value in updates):
            raise ValueError("at least one deck configuration update must be provided")
        for name, value in (
            ("new_cards_per_day", new_cards_per_day),
            ("reviews_per_day", reviews_per_day),
            ("max_answer_seconds", max_answer_seconds),
        ):
            if value is not None and not 0 <= value <= 999_999:
                raise ValueError(f"{name} must be between 0 and 999999")
        if desired_retention is not None and not 0.7 <= desired_retention <= 0.99:
            raise ValueError("desired_retention must be between 0.7 and 0.99")
        config = self.collection.decks.config_dict_for_deck_id(cast("DeckId", deck_id))
        if new_cards_per_day is not None:
            config.setdefault("new", {})["perDay"] = new_cards_per_day
        if reviews_per_day is not None:
            config.setdefault("rev", {})["perDay"] = reviews_per_day
        if max_answer_seconds is not None:
            config["maxTaken"] = max_answer_seconds
        if desired_retention is not None:
            config["desiredRetention"] = desired_retention
        self.collection.decks.update_config(config)
        return {"id": deck_id, "config_id": int(config["id"]), "updated": True}

    def _deck_config_state(self, deck_id: int) -> Any:
        deck = self.collection.decks.get(cast("DeckId", deck_id), default=False)
        if deck is None:
            raise LookupError(f"deck {deck_id} not found")
        if bool(deck.get("dyn", 0)):
            raise ValueError("dynamic decks do not have editable deck options")
        return self.collection.decks.get_deck_configs_for_update(cast("DeckId", deck_id))

    def _preset_entry(self, config_id: int) -> tuple[Any, Any]:
        state = self.collection.decks.get_deck_configs_for_update(cast("DeckId", 1))
        for entry in state.all_config:
            if int(entry.config.id) == config_id:
                return state, entry
        raise LookupError(f"deck preset {config_id} not found")

    def _preset_summary(self, entry: Any) -> dict[str, Any]:
        config = entry.config.config
        name, name_truncated = self._truncate_rendered(str(entry.config.name))
        result = {
            "id": int(entry.config.id),
            "name": name,
            "use_count": int(entry.use_count),
            "new_cards_per_day": int(config.new_per_day),
            "reviews_per_day": int(config.reviews_per_day),
            "max_answer_seconds": int(config.cap_answer_time_to_secs),
            "desired_retention": round(float(config.desired_retention), 6),
        }
        if name_truncated:
            result["name_truncated"] = True
        return result

    def list_deck_presets(self, offset: int, limit: int) -> dict[str, Any]:
        state = self.collection.decks.get_deck_configs_for_update(cast("DeckId", 1))
        if len(state.all_config) > self.max_search_scan:
            raise ValueError(
                "deck presets exceed MCP_MAX_SEARCH_SCAN; use a larger configured bound"
            )
        values = sorted(
            (self._preset_summary(entry) for entry in state.all_config),
            key=lambda item: str(item["name"]).casefold(),
        )
        return self._page(values, offset, limit)

    def get_deck_preset(
        self, config_id: int, include_sections: Sequence[str] = ()
    ) -> dict[str, Any]:
        _, entry = self._preset_entry(config_id)
        requested = list(dict.fromkeys(include_sections))
        unknown = set(requested) - set(DECK_PRESET_SECTIONS)
        if unknown:
            raise ValueError(f"unknown deck preset sections: {', '.join(sorted(unknown))}")
        result = self._preset_summary(entry)
        if requested:
            values = MessageToDict(
                entry.config.config,
                preserving_proto_field_name=True,
                always_print_fields_with_no_presence=True,
            )
            result["sections"] = {
                section: {field: values[field] for field in DECK_PRESET_SECTIONS[section]}
                for section in requested
            }
        return result

    def _update_deck_config_state(
        self,
        deck_id: int,
        state: Any,
        *,
        apply_all_parent_limits: bool | None = None,
        new_cards_ignore_review_limit: bool | None = None,
        fsrs_enabled: bool | None = None,
        fsrs_reschedule: bool = False,
    ) -> None:
        try:
            self.collection.decks.update_deck_configs(
                UpdateDeckConfigs(
                    target_deck_id=deck_id,
                    configs=[entry.config for entry in state.all_config],
                    card_state_customizer=state.card_state_customizer,
                    limits=state.current_deck.limits,
                    new_cards_ignore_review_limit=(
                        state.new_cards_ignore_review_limit
                        if new_cards_ignore_review_limit is None
                        else new_cards_ignore_review_limit
                    ),
                    fsrs=state.fsrs if fsrs_enabled is None else fsrs_enabled,
                    apply_all_parent_limits=(
                        state.apply_all_parent_limits
                        if apply_all_parent_limits is None
                        else apply_all_parent_limits
                    ),
                    fsrs_reschedule=fsrs_reschedule,
                    fsrs_health_check=state.fsrs_health_check,
                )
            )
        except InvalidInput as exc:
            raise ValueError("invalid deck option update") from exc

    def update_deck_preset(
        self, config_id: int, name: str | None, options: dict[str, Any]
    ) -> dict[str, Any]:
        state, entry = self._preset_entry(config_id)
        if name is None and not options:
            raise ValueError("at least one deck preset update must be provided")
        if name is not None:
            if not name.strip():
                raise ValueError("deck preset name must not be blank")
            for candidate in state.all_config:
                if (
                    int(candidate.config.id) != config_id
                    and str(candidate.config.name).casefold() == name.casefold()
                ):
                    raise ValueError(f"deck preset {name} already exists")
            entry.config.name = name

        config = entry.config.config
        descriptors = {
            field.name: field for field in config.DESCRIPTOR.fields if field.name != "other"
        }
        unknown = set(options) - set(descriptors)
        if unknown:
            raise ValueError(f"unknown deck preset options: {', '.join(sorted(unknown))}")
        for field_name, value in options.items():
            descriptor = descriptors[field_name]
            if descriptor.is_repeated:
                target = getattr(config, field_name)
                del target[:]
                target.extend(value)
            elif descriptor.enum_type is not None:
                enum_value = descriptor.enum_type.values_by_name.get(str(value))
                if enum_value is None:
                    raise ValueError(f"unsupported {field_name} value")
                setattr(config, field_name, enum_value.number)
            else:
                setattr(config, field_name, value)

        affected_decks = int(entry.use_count)
        self._update_deck_config_state(1, state)
        return {
            "id": config_id,
            "updated": True,
            "changed_fields": len(options) + int(name is not None),
            "affected_decks": affected_decks,
        }

    def create_deck_preset(self, name: str, clone_from_config_id: int | None) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("deck preset name must not be blank")
        presets = self.collection.decks.all_config()
        if any(str(config["name"]).casefold() == name.casefold() for config in presets):
            raise ValueError(f"deck preset {name} already exists")
        clone = None
        if clone_from_config_id is not None:
            clone = self.collection.decks.get_config(cast("DeckConfigId", clone_from_config_id))
            if clone is None:
                raise LookupError(f"deck preset {clone_from_config_id} not found")
        created = self.collection.decks.add_config(name, clone)
        return {"id": int(created["id"]), "created": True}

    def assign_deck_preset(self, deck_id: int, config_id: int) -> dict[str, Any]:
        self._preset_entry(config_id)
        deck = self.collection.decks.get(cast("DeckId", deck_id), default=False)
        if deck is None:
            raise LookupError(f"deck {deck_id} not found")
        if bool(deck.get("dyn", 0)):
            raise ValueError("dynamic decks cannot be assigned a deck preset")
        self.collection.decks.set_config_id_for_deck_dict(deck, cast("DeckConfigId", config_id))
        return {"deck_id": deck_id, "config_id": config_id, "updated": True}

    def update_deck_limits(
        self,
        deck_id: int,
        scope: str,
        values: dict[str, Any],
        clear_fields: Sequence[str],
    ) -> dict[str, Any]:
        if scope not in {"this_deck", "today"}:
            raise ValueError("scope must be this_deck or today")
        allowed = {"new_cards_per_day", "reviews_per_day"}
        if scope == "this_deck":
            allowed.add("desired_retention")
        unknown = (set(values) | set(clear_fields)) - allowed
        if unknown:
            raise ValueError(f"unsupported {scope} limit fields: {', '.join(sorted(unknown))}")
        overlap = set(values) & set(clear_fields)
        if overlap:
            raise ValueError(
                f"limit fields cannot be set and cleared together: {', '.join(sorted(overlap))}"
            )
        if not values and not clear_fields:
            raise ValueError("at least one deck limit update must be provided")

        state = self._deck_config_state(deck_id)
        limits = state.current_deck.limits
        field_names = {
            "new_cards_per_day": "new",
            "reviews_per_day": "review",
            "desired_retention": "desired_retention",
        }
        if scope == "today":
            field_names = {
                "new_cards_per_day": "new_today",
                "reviews_per_day": "review_today",
            }
        for field, value in values.items():
            target = field_names[field]
            setattr(limits, target, value)
            if scope == "today":
                setattr(limits, f"{target}_active", True)
        for field in clear_fields:
            target = field_names[field]
            limits.ClearField(target)
            if scope == "today":
                setattr(limits, f"{target}_active", False)

        self._update_deck_config_state(deck_id, state)
        return {"deck_id": deck_id, "scope": scope, "updated": True}

    def update_deck_scheduler_settings(
        self,
        apply_all_parent_limits: bool | None,
        new_cards_ignore_review_limit: bool | None,
        fsrs_enabled: bool | None,
    ) -> dict[str, Any]:
        if (
            apply_all_parent_limits is None
            and new_cards_ignore_review_limit is None
            and fsrs_enabled is None
        ):
            raise ValueError("at least one collection-wide scheduler setting must be provided")
        state = self._deck_config_state(1)
        self._update_deck_config_state(
            1,
            state,
            apply_all_parent_limits=apply_all_parent_limits,
            new_cards_ignore_review_limit=new_cards_ignore_review_limit,
            fsrs_enabled=fsrs_enabled,
        )
        return {
            "scope": "collection",
            "updated": True,
            "apply_all_parent_limits": (
                bool(state.apply_all_parent_limits)
                if apply_all_parent_limits is None
                else apply_all_parent_limits
            ),
            "new_cards_ignore_review_limit": (
                bool(state.new_cards_ignore_review_limit)
                if new_cards_ignore_review_limit is None
                else new_cards_ignore_review_limit
            ),
            "fsrs_enabled": bool(state.fsrs) if fsrs_enabled is None else fsrs_enabled,
        }

    @staticmethod
    def _fsrs_search(entry: Any, search: str | None) -> str:
        selected = search if search is not None else str(entry.config.config.param_search)
        if selected.strip():
            return selected
        name = str(entry.config.name).replace('"', '\\"')
        return f'preset:"{name}" -is:suspended'

    def _bounded_fsrs_search(self, entry: Any, search: str | None, operation: str) -> str:
        selected = self._fsrs_search(entry, search)
        try:
            selected_cards = self.collection.find_cards(
                selected, order=f"c.id asc limit {self.max_search_scan + 1}"
            )
        except SearchError as exc:
            raise ValueError(f"invalid FSRS search: {exc}") from exc
        if len(selected_cards) > self.max_search_scan:
            raise ResourceLimitError(
                f"FSRS {operation}: maximum={self.max_search_scan}, "
                f"observed_at_least={len(selected_cards)}; narrow search or raise "
                "MCP_MAX_SEARCH_SCAN"
            )
        return selected

    def _bounded_preset_decks(self, config_id: int, operation: str) -> list[dict[str, Any]]:
        db = self.collection.db
        assert db is not None
        rows = db.all(
            "select id, name from decks order by id limit ?", self.max_search_scan + 1
        )
        if len(rows) > self.max_search_scan:
            raise ResourceLimitError(
                f"FSRS {operation} for preset {config_id}: maximum={self.max_search_scan}, "
                f"observed_at_least={len(rows)} collection decks; raise MCP_MAX_SEARCH_SCAN "
                "to inspect the shared preset scope"
            )
        affected: list[dict[str, Any]] = []
        for deck_id, name in rows:
            deck = self.collection.decks.get(cast("DeckId", int(deck_id)), default=False)
            if (
                deck is not None
                and not bool(deck.get("dyn", 0))
                and int(deck.get("conf", 1)) == config_id
            ):
                affected.append({"id": int(deck_id), "name": str(name)})
        return affected

    @staticmethod
    def _ignore_revlogs_before_ms(value: str) -> int:
        if not value.strip():
            return 0
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError as exc:
            raise ValueError("ignore_revlogs_before_date must use YYYY-MM-DD") from exc
        return int(parsed.timestamp() * 1000)

    @staticmethod
    def _active_fsrs_parameters(config: Any, defaults: Any) -> list[float]:
        for field in ("fsrs_params_6", "fsrs_params_5", "fsrs_params_4"):
            values = list(getattr(config, field))
            if values:
                return [float(value) for value in values]
        return [float(value) for value in defaults.fsrs_params_6]

    def preview_fsrs_optimization(
        self, config_id: int, search: str | None, health_check: bool
    ) -> dict[str, Any]:
        _, entry = self._preset_entry(config_id)
        config = entry.config.config
        selected_search = self._bounded_fsrs_search(entry, search, "optimization")
        affected_decks = self._bounded_preset_decks(config_id, "optimization")
        previous = [float(value) for value in config.fsrs_params_6]
        try:
            response = self.collection._backend.compute_fsrs_params(  # pyright: ignore[reportPrivateUsage]
                search=selected_search,
                current_params=previous,
                ignore_revlogs_before_ms=self._ignore_revlogs_before_ms(
                    str(config.ignore_revlogs_before_date)
                ),
                num_of_relearning_steps=len(config.relearn_steps),
                health_check=health_check,
            )
        except (InvalidInput, SearchError) as exc:
            raise ValueError(f"invalid FSRS optimization input: {exc}") from exc
        parameters = [float(value) for value in response.params]
        if not parameters:
            raise ValueError("not enough review history to optimize FSRS parameters")
        self._validate_fsrs_parameters(parameters)
        if health_check and not bool(response.health_check_passed):
            raise ValueError("FSRS optimizer health check failed; preset was not changed")
        return {
            "config_id": config_id,
            "search": selected_search,
            "training_items": int(response.fsrs_items),
            "health_check_requested": health_check,
            "health_check_passed": bool(response.health_check_passed),
            "current_parameters": previous,
            "candidate_parameters": parameters,
            "affected_decks": len(affected_decks),
            "affected_deck_ids": [deck["id"] for deck in affected_decks],
            "affected_decks_detail": affected_decks,
            "state_fingerprint": self._impact_fingerprint(
                {
                    "config_id": config_id,
                    "search": selected_search,
                    "training_items": int(response.fsrs_items),
                    "current_parameters": previous,
                    "candidate_parameters": parameters,
                    "affected_decks": affected_decks,
                }
            ),
        }

    def apply_fsrs_optimization(
        self,
        config_id: int,
        search: str | None,
        health_check: bool,
        impact: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected_impact = impact or self.preview_fsrs_optimization(config_id, search, health_check)
        state, entry = self._preset_entry(config_id)
        config = entry.config.config
        del config.fsrs_params_6[:]
        config.fsrs_params_6.extend(selected_impact["candidate_parameters"])
        self._update_deck_config_state(1, state)
        return {
            "config_id": config_id,
            "optimized": True,
            "search": selected_impact["search"],
            "training_items": selected_impact["training_items"],
            "health_check_requested": selected_impact["health_check_requested"],
            "health_check_passed": selected_impact["health_check_passed"],
            "previous_parameters": selected_impact["current_parameters"],
            "parameters": selected_impact["candidate_parameters"],
            "affected_decks": selected_impact["affected_decks"],
            "affected_deck_ids": selected_impact["affected_deck_ids"],
            "affected_decks_detail": selected_impact["affected_decks_detail"],
        }

    def optimize_fsrs(
        self, config_id: int, search: str | None, health_check: bool
    ) -> dict[str, Any]:
        """Internal compatibility wrapper; public MCP callers use preview/apply."""
        return self.apply_fsrs_optimization(config_id, search, health_check)

    @staticmethod
    def _validate_fsrs_parameters(parameters: Sequence[float]) -> list[float]:
        values = [float(value) for value in parameters]
        if len(values) != 21:
            raise ValueError("FSRS parameters must contain exactly 21 values")
        for index, value in enumerate(values):
            if not math.isfinite(value):
                raise ValueError(f"FSRS parameter {index} must be finite")
            if value < 0:
                raise ValueError(
                    f"invalid FSRS parameters: parameter {index} is below the Anki 26.5 bound"
                )
        return values

    def _validate_native_fsrs_parameters(
        self, config_id: int, parameters: Sequence[float], desired_retention: float
    ) -> None:
        request, _ = self._fsrs_simulation_request(
            config_id,
            deck_size=1,
            days_to_simulate=1,
            desired_retention=desired_retention,
            search="nid:0",
        )
        del request.params[:]
        request.params.extend(parameters)
        try:
            self.collection._backend.simulate_fsrs_review(  # pyright: ignore[reportPrivateUsage]
                request
            )
        except InvalidInput as exc:
            raise ValueError(f"invalid FSRS parameters: {exc}") from exc

    def _fsrs_simulation_request(
        self,
        config_id: int,
        deck_size: int,
        days_to_simulate: int,
        desired_retention: float | None,
        search: str | None,
    ) -> tuple[SimulateFsrsReviewRequest, str]:
        state, entry = self._preset_entry(config_id)
        config = entry.config.config
        defaults = state.defaults.config
        selected_search = self._fsrs_search(entry, search)
        easy_days = list(config.easy_days_percentages) or list(defaults.easy_days_percentages)
        request = SimulateFsrsReviewRequest(
            params=self._active_fsrs_parameters(config, defaults),
            desired_retention=(
                float(config.desired_retention)
                if desired_retention is None
                else desired_retention
            ),
            deck_size=deck_size,
            days_to_simulate=days_to_simulate,
            new_limit=int(config.new_per_day),
            review_limit=int(config.reviews_per_day),
            max_interval=int(config.maximum_review_interval),
            search=selected_search,
            new_cards_ignore_review_limit=bool(state.new_cards_ignore_review_limit),
            easy_days_percentages=easy_days,
            review_order=config.review_order,
            suspend_after_lapse_count=int(config.leech_threshold),
            historical_retention=float(config.historical_retention),
            learning_step_count=len(config.learn_steps),
            relearning_step_count=len(config.relearn_steps),
        )
        return request, selected_search

    def simulate_fsrs(
        self,
        config_id: int,
        mode: str,
        deck_size: int,
        days_to_simulate: int,
        desired_retention: float | None,
        search: str | None,
        include_daily: bool,
    ) -> dict[str, Any]:
        if mode not in {"review", "workload", "optimal_retention"}:
            raise ValueError("mode must be review, workload, or optimal_retention")
        _, entry = self._preset_entry(config_id)
        selected_search = self._bounded_fsrs_search(entry, search, "simulation")
        request, selected_search = self._fsrs_simulation_request(
            config_id, deck_size, days_to_simulate, desired_retention, selected_search
        )
        result: dict[str, Any] = {
            "mode": mode,
            "config_id": config_id,
            "search": selected_search,
            "days_to_simulate": days_to_simulate,
            "deck_size": deck_size,
        }
        backend = self.collection._backend  # pyright: ignore[reportPrivateUsage]
        if mode == "optimal_retention":
            try:
                result["optimal_retention"] = round(
                    float(backend.compute_optimal_retention(request)), 6
                )
            except (InvalidInput, SearchError) as exc:
                raise ValueError(f"invalid FSRS simulation input: {exc}") from exc
            return result
        if mode == "workload":
            try:
                response = backend.simulate_fsrs_workload(request)
            except (InvalidInput, SearchError) as exc:
                raise ValueError(f"invalid FSRS simulation input: {exc}") from exc
            result["retention"] = {
                "cost_seconds": {str(key): float(value) for key, value in response.cost.items()},
                "memorized_cards": {
                    str(key): float(value) for key, value in response.memorized.items()
                },
                "review_count": {
                    str(key): int(value) for key, value in response.review_count.items()
                },
            }
            result["reviewless_end_memorized"] = float(response.reviewless_end_memorized)
            return result

        try:
            response = backend.simulate_fsrs_review(request)
        except (InvalidInput, SearchError) as exc:
            raise ValueError(f"invalid FSRS simulation input: {exc}") from exc
        daily_reviews = [int(value) for value in response.daily_review_count]
        daily_new = [int(value) for value in response.daily_new_count]
        daily_time = [float(value) for value in response.daily_time_cost]
        knowledge = [float(value) for value in response.accumulated_knowledge_acquisition]
        lengths = {len(daily_reviews), len(daily_new), len(daily_time), len(knowledge)}
        if lengths != {days_to_simulate}:
            raise RuntimeError(
                "Anki 26.5 FSRS simulator returned incompatible daily array lengths"
            )
        result["summary"] = {
            "total_reviews": sum(daily_reviews),
            "total_new_cards": sum(daily_new),
            "total_time_seconds": round(sum(daily_time), 6),
            "final_knowledge_acquisition": knowledge[-1] if knowledge else 0.0,
        }
        if include_daily:
            result["daily"] = [
                {
                    "day": index + 1,
                    "reviews": daily_reviews[index],
                    "new_cards": daily_new[index],
                    "time_seconds": daily_time[index],
                    "knowledge_acquisition": knowledge[index],
                }
                for index in range(len(daily_reviews))
            ]
        return result

    def _fsrs_reschedule_impact(
        self,
        config_id: int,
        desired_retention: float | None,
        parameters: Sequence[float] | None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        state, entry = self._preset_entry(config_id)
        config = entry.config.config
        current_retention = round(float(config.desired_retention), 6)
        target_retention = (
            current_retention if desired_retention is None else round(desired_retention, 6)
        )
        current_parameters = [float(value) for value in config.fsrs_params_6]
        target_parameters = (
            current_parameters if parameters is None else self._validate_fsrs_parameters(parameters)
        )
        if parameters is not None:
            self._validate_native_fsrs_parameters(config_id, target_parameters, target_retention)
        if target_retention == current_retention and target_parameters == current_parameters:
            raise ValueError("FSRS rescheduling must change desired retention or parameters")
        if not state.fsrs:
            raise ValueError("FSRS must be enabled before cards can be rescheduled")

        db = self.collection.db
        assert db is not None
        bounded_deck_ids = [
            int(deck_id)
            for deck_id in db.list(
                "select id from decks order by id limit ?", self.max_search_scan + 1
            )
        ]
        if len(bounded_deck_ids) > self.max_search_scan:
            raise ResourceLimitError(
                f"FSRS reschedule for preset {config_id}: maximum={self.max_search_scan}, "
                f"observed_at_least={len(bounded_deck_ids)} decks; split preset scope or raise "
                "MCP_MAX_SEARCH_SCAN"
            )
        decks = []
        for deck_id in bounded_deck_ids:
            deck = self.collection.decks.get(cast("DeckId", deck_id), default=False)
            if (
                deck is not None
                and not bool(deck.get("dyn", 0))
                and int(deck.get("conf", 1)) == config_id
            ):
                limits = self.collection.decks.get_deck_configs_for_update(
                    cast("DeckId", deck_id)
                ).current_deck.limits
                decks.append(
                    {
                        "id": deck_id,
                        "config_id": int(deck.get("conf", 1)),
                        "desired_retention": (
                            float(limits.desired_retention)
                            if limits.HasField("desired_retention")
                            else None
                        ),
                    }
                )
        deck_ids_json = json.dumps([deck["id"] for deck in decks])
        candidate_rows = db.all(
            """
                select id, did, odid, type, queue, due, odue, ivl
                from cards
                where (
                    (odid = 0 and did in (select value from json_each(?)))
                    or odid in (select value from json_each(?))
                )
                  and type = ? and queue != ?
                order by id limit ?
                """,
            deck_ids_json,
            deck_ids_json,
            CARD_TYPE_REV,
            QUEUE_TYPE_SUSPENDED,
            self.max_search_scan + 1,
        )
        if len(candidate_rows) > self.max_search_scan:
            raise ResourceLimitError(
                f"FSRS reschedule for preset {config_id} across {len(decks)} decks: "
                f"maximum={self.max_search_scan}, observed_at_least={len(candidate_rows)} "
                "candidate cards; split preset scope or raise MCP_MAX_SEARCH_SCAN"
            )
        candidate_ids = [int(row[0]) for row in candidate_rows]
        ignore_before = self._ignore_revlogs_before_ms(str(config.ignore_revlogs_before_date))
        candidate_ids_json = json.dumps(candidate_ids)
        valid_card_ids: set[int] = set()
        review_history_hash = hashlib.sha256()
        last_review_id = -1
        while True:
            review_rows = db.all(
                """
                select cid, id, ease, ivl, lastIvl, factor, time, type
                from revlog
                where cid in (select value from json_each(?))
                  and id > ?
                  and (
                    (ease > 0 and not (type = 3 and factor = 0)
                     and (id > ? or type = 0))
                    or (type = 4 and factor = 0)
                  )
                order by id limit 1000
                """,
                candidate_ids_json,
                last_review_id,
                ignore_before,
            )
            if not review_rows:
                break
            for row in review_rows:
                if (
                    int(row[2]) > 0
                    and int(row[1]) > ignore_before
                    and (int(row[3]) >= 1 or int(row[3]) <= -86_400)
                    and not (int(row[7]) == 3 and int(row[5]) == 0)
                ):
                    valid_card_ids.add(int(row[0]))
                review_history_hash.update(
                    json.dumps(
                        list(row), ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                )
                review_history_hash.update(b"\n")
            last_review_id = int(review_rows[-1][1])
        card_states = []
        for card_id in candidate_ids:
            if card_id not in valid_card_ids:
                continue
            card = self.collection.get_card(cast("CardId", card_id))
            if card.type != CARD_TYPE_REV or card.queue == QUEUE_TYPE_SUSPENDED:
                continue
            memory_state = card.memory_state
            card_states.append(
                {
                    "id": card_id,
                    "type": int(card.type),
                    "queue": int(card.queue),
                    "due": int(card.due),
                    "original_due": int(card.odue),
                    "interval": int(card.ivl),
                    "reps": int(card.reps),
                    "lapses": int(card.lapses),
                    "left": int(card.left),
                    "desired_retention": card.desired_retention,
                    "decay": card.decay,
                    "memory_state": (
                        None
                        if memory_state is None
                        else {
                            "stability": float(memory_state.stability),
                            "difficulty": float(memory_state.difficulty),
                        }
                    ),
                }
            )
        timing = self.collection.sched._timing_today()
        reviewed_today = int(
            db.scalar(
                """
                select count(distinct r.cid)
                from revlog r join cards c on r.cid = c.id
                where r.id > ? and r.ease > 0 and (r.type < 3 or r.factor != 0)
                  and (
                    (c.odid = 0 and c.did in (select value from json_each(?)))
                    or c.odid in (select value from json_each(?))
                  )
                """,
                (int(timing.next_day_at) - 86_400) * 1000,
                deck_ids_json,
                deck_ids_json,
            )
        )
        impact = {
            "config_id": config_id,
            "decks": len(decks),
            "cards": len(card_states),
            "desired_retention": target_retention,
            "parameters_changed": target_parameters != current_parameters,
            "state_fingerprint": self._impact_fingerprint(
                {
                    "decks": decks,
                    "cards": card_states,
                    "load_balance_cards": [list(row) for row in candidate_rows],
                    "review_history": review_history_hash.hexdigest(),
                    "scheduler": {
                        "fsrs": bool(state.fsrs),
                        "days_elapsed": int(timing.days_elapsed),
                        "next_day_at": int(timing.next_day_at),
                        "load_balancer": self.collection.get_config_bool(
                            ConfigKey.Bool.LOAD_BALANCER_ENABLED
                        ),
                        "reviewed_today": reviewed_today,
                        "maximum_review_interval": int(config.maximum_review_interval),
                        "historical_retention": float(config.historical_retention),
                        "easy_days_percentages": list(config.easy_days_percentages),
                        "ignore_revlogs_before_date": str(config.ignore_revlogs_before_date),
                    },
                    "current_retention": current_retention,
                    "current_parameters": current_parameters,
                }
            ),
        }
        return state, entry, impact

    def preview_fsrs_reschedule(
        self,
        config_id: int,
        desired_retention: float | None,
        parameters: Sequence[float] | None,
    ) -> dict[str, Any]:
        _, _, impact = self._fsrs_reschedule_impact(
            config_id, desired_retention, parameters
        )
        return impact

    def reschedule_fsrs(
        self,
        config_id: int,
        desired_retention: float | None,
        parameters: Sequence[float] | None,
    ) -> dict[str, Any]:
        state, entry, impact = self._fsrs_reschedule_impact(
            config_id, desired_retention, parameters
        )
        config = entry.config.config
        if desired_retention is not None:
            config.desired_retention = desired_retention
        if parameters is not None:
            del config.fsrs_params_6[:]
            config.fsrs_params_6.extend(parameters)
        self._update_deck_config_state(1, state, fsrs_reschedule=True)
        return {
            "config_id": config_id,
            "rescheduled": True,
            "cards": impact["cards"],
            "decks": impact["decks"],
            "desired_retention": impact["desired_retention"],
            "parameters_changed": impact["parameters_changed"],
        }

    def get_deck_options(
        self, deck_id: int, include_sections: Sequence[str] = ()
    ) -> dict[str, Any]:
        allowed_sections = {"counts", "parents", "global_settings"}
        requested = list(dict.fromkeys(include_sections))
        unknown = set(requested) - allowed_sections
        if unknown:
            raise ValueError(f"unknown deck option sections: {', '.join(sorted(unknown))}")
        state = self._deck_config_state(deck_id)
        deck = self.collection.decks.get(cast("DeckId", deck_id), default=False)
        if deck is None:  # pragma: no cover - validated by _deck_config_state
            raise LookupError(f"deck {deck_id} not found")
        def resolved_layers(source_deck_id: int, source_state: Any) -> tuple[Any, Any, Any]:
            entry = next(
                item
                for item in source_state.all_config
                if int(item.config.id) == int(source_state.current_deck.config_id)
            )
            limits = source_state.current_deck.limits

            def optional(field: str) -> Any:
                if not limits.HasField(field):
                    return None
                value = getattr(limits, field)
                return round(float(value), 6) if field == "desired_retention" else value

            preset = self._preset_summary(entry)
            preset.pop("use_count")
            this_deck = {
                "new_cards_per_day": optional("new"),
                "reviews_per_day": optional("review"),
                "desired_retention": optional("desired_retention"),
            }
            today = {
                "new_cards_per_day": optional("new_today") if limits.new_today_active else None,
                "reviews_per_day": (
                    optional("review_today") if limits.review_today_active else None
                ),
            }

            def effective(field: str) -> dict[str, Any]:
                if field in today and today[field] is not None:
                    value, layer = today[field], "today"
                elif this_deck[field] is not None:
                    value, layer = this_deck[field], "this_deck"
                else:
                    value, layer = preset[field], "preset"
                return {
                    "value": value,
                    "source": layer,
                    "source_deck_id": source_deck_id,
                    "inherited": False,
                }

            effective_limits = {
                field: effective(field)
                for field in (
                    "new_cards_per_day",
                    "reviews_per_day",
                    "desired_retention",
                )
            }
            return preset, {"this_deck": this_deck, "today": today}, effective_limits

        preset, limit_layers, effective_limits = resolved_layers(deck_id, state)
        parents = self.collection.decks.parents(cast("DeckId", deck_id))
        parent_chain = []
        for parent in parents:
            parent_id = int(parent["id"])
            _, _, parent_effective = resolved_layers(
                parent_id, self._deck_config_state(parent_id)
            )
            parent_chain.append(
                {
                    "deck_id": parent_id,
                    "name": str(parent["name"]),
                    "effective_limits": {
                        field: parent_effective[field]
                        for field in ("new_cards_per_day", "reviews_per_day")
                    },
                }
            )
            if state.apply_all_parent_limits:
                for field in ("new_cards_per_day", "reviews_per_day"):
                    if parent_effective[field]["value"] < effective_limits[field]["value"]:
                        effective_limits[field] = {
                            **parent_effective[field],
                            "inherited": True,
                        }

        result: dict[str, Any] = {
            "deck_id": deck_id,
            "name": str(deck["name"]),
            "preset": preset,
            "limits": limit_layers,
            "effective_limits": effective_limits,
            "apply_all_parent_limits": bool(state.apply_all_parent_limits),
        }
        sections: dict[str, Any] = {}
        if "parents" in requested:
            sections["parents"] = {
                "deck_ids": [int(parent["id"]) for parent in parents],
                "preset_ids": [int(value) for value in state.current_deck.parent_config_ids],
                "limits_applied": bool(state.apply_all_parent_limits),
                "limit_chain": parent_chain,
            }
        if "counts" in requested:
            node = self.collection.decks.find_deck_in_tree(
                self.collection.sched.deck_due_tree(), cast("DeckId", deck_id)
            )
            if node is None:  # pragma: no cover - a fetched deck is present in the tree
                raise RuntimeError("deck was not present in Anki's deck tree")
            sections["counts"] = {
                "new": int(node.new_count),
                "review": int(node.review_count),
                "learning": int(node.learn_count),
                "new_uncapped": int(node.new_uncapped),
                "review_uncapped": int(node.review_uncapped),
                "total_in_deck": int(node.total_in_deck),
                "total_including_children": int(node.total_including_children),
            }
        if "global_settings" in requested:
            sections["global_settings"] = {
                "new_cards_ignore_review_limit": bool(state.new_cards_ignore_review_limit),
                "fsrs": bool(state.fsrs),
            }
        if sections:
            result["sections"] = sections
        return result

    def _raise_after_delete_failure(
        self,
        operation_error: Exception,
        resource_name: str,
        verify_present: Callable[[], object],
    ) -> None:
        try:
            verify_present()
        except LookupError:
            try:
                self.collection.undo()
                verify_present()
            except Exception:
                raise RuntimeError(
                    f"{resource_name} deletion failed and rollback was incomplete"
                ) from operation_error
        raise operation_error

    def delete_deck(self, deck_id: int) -> dict[str, Any]:
        self.get_deck(deck_id)
        if deck_id == 1:
            raise ValueError("the Default deck cannot be deleted")
        try:
            self.collection.decks.remove([cast("DeckId", deck_id)])
        except Exception as operation_error:
            self._raise_after_delete_failure(
                operation_error, "deck", lambda: self.get_deck(deck_id)
            )
        return {"id": deck_id, "deleted": True}

    def preview_deck_delete(self, deck_id: int) -> dict[str, Any]:
        deck = self.get_deck(deck_id)
        if deck_id == 1:
            raise ValueError("the Default deck cannot be deleted")
        name = str(deck["name"])
        affected_deck_ids = [
            int(item["id"])
            for item in self.collection.decks.all()
            if str(item["name"]) == name or str(item["name"]).startswith(f"{name}::")
        ]
        card_ids = {
            int(card_id)
            for affected_id in affected_deck_ids
            for card_id in self.collection.find_cards(f"did:{affected_id}")
        }
        note_ids = {
            int(self.collection.get_card(cast("CardId", card_id)).nid) for card_id in card_ids
        }
        return {
            "decks": len(affected_deck_ids),
            "cards": len(card_ids),
            "notes_at_risk": len(note_ids),
            "state_fingerprint": self._impact_fingerprint(
                {
                    "deck_ids": sorted(affected_deck_ids),
                    "card_ids": sorted(card_ids),
                    "note_ids": sorted(note_ids),
                }
            ),
        }

    def search_notes(self, query: str, offset: int, limit: int) -> dict[str, Any]:
        self._page([], offset, limit)
        if int(self.collection.note_count()) > self.max_search_scan:
            raise ValueError(
                "collection exceeds MCP_MAX_SEARCH_SCAN; use a larger configured bound"
            )
        note_ids = [int(note_id) for note_id in self.collection.find_notes(query)]
        result = self._page(note_ids, offset, limit)
        result["items"] = [self._note_summary(note_id) for note_id in result["items"]]
        return result

    def _note_summary(self, note_id: int) -> dict[str, Any]:
        try:
            note = self.collection.get_note(cast("NoteId", note_id))
        except NotFoundError as exc:
            raise LookupError(f"note {note_id} not found") from exc
        first_field, first_field_truncated = self._truncate_rendered(note.fields[0])
        return {
            "id": int(note.id),
            "note_type_id": int(note.mid),
            "first_field": first_field,
            "first_field_truncated": first_field_truncated,
            "tags": list(note.tags),
        }

    def get_note(self, note_id: int) -> dict[str, Any]:
        try:
            note = self.collection.get_note(cast("NoteId", note_id))
        except NotFoundError as exc:
            raise LookupError(f"note {note_id} not found") from exc
        model = self.collection.models.get(note.mid)
        if model is None:  # pragma: no cover - Anki preserves referenced note types
            raise LookupError(f"note type for note {note_id} not found")
        model_name, model_name_truncated = self._truncate_rendered(str(model["name"]))
        fields: list[dict[str, Any]] = []
        note_items = note.items()
        for name, value in note_items[: self.max_card_fields]:
            bounded_name, name_truncated = self._truncate_rendered(name)
            bounded_value, value_truncated = self._truncate_rendered(value)
            fields.append(
                {
                    "name": bounded_name,
                    "name_truncated": name_truncated,
                    "value": bounded_value,
                    "value_truncated": value_truncated,
                }
            )
        return {
            "id": int(note.id),
            "note_type_id": int(note.mid),
            "note_type_name": model_name,
            "note_type_name_truncated": model_name_truncated,
            "fields": fields,
            "fields_omitted": max(0, len(note_items) - self.max_card_fields),
            "tags": list(note.tags),
            "card_ids": [
                int(card_id) for card_id in self.collection.find_cards(f"nid:{int(note.id)}")
            ],
            "modified": int(note.mod),
        }

    def _prepare_note(
        self,
        deck_id: int,
        note_type_id: int,
        fields: dict[str, str],
        tags: list[str],
    ) -> Any:
        self.get_deck(deck_id)
        model = self.collection.models.get(cast("NotetypeId", note_type_id))
        if model is None:
            raise LookupError(f"note type {note_type_id} not found")
        expected_fields = self.collection.models.field_names(model)
        if set(fields) != set(expected_fields):
            raise ValueError(f"fields must exactly match note type fields: {expected_fields}")
        note = self.collection.new_note(model)
        for field_name in expected_fields:
            note[field_name] = fields[field_name]
        note.tags = self._normalize_tags(tags)
        check = int(note.duplicate_or_empty())
        if check == 1:
            raise ValueError("the note's first field must not be empty")
        if check == 2:
            raise DuplicateNoteError("a note with the same first field already exists")
        return note

    def _normalize_tags(self, tags: list[str]) -> list[str]:
        if any(not tag.strip() for tag in tags):
            raise ValueError("tags must not contain blank values")
        return list(dict.fromkeys(tags))

    def create_note(
        self,
        deck_id: int,
        note_type_id: int,
        fields: dict[str, str],
        tags: list[str],
    ) -> dict[str, Any]:
        note = self._prepare_note(deck_id, note_type_id, fields, tags)
        self.collection.add_note(note, cast("DeckId", deck_id))
        card_ids = [int(card_id) for card_id in self.collection.find_cards(f"nid:{int(note.id)}")]
        if not card_ids:
            self.collection.remove_notes([note.id])
            raise ValueError("note did not generate any cards")
        return {"note_id": int(note.id), "card_ids": card_ids, "created": True}

    def create_notes_batch(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        if not requests or len(requests) > self.max_batch_size:
            raise ValueError(f"batch must contain between 1 and {self.max_batch_size} notes")
        prepared: list[tuple[Any, int]] = []
        fingerprints: set[tuple[int, int]] = set()
        for request in requests:
            note = self._prepare_note(
                int(request["deck_id"]),
                int(request["note_type_id"]),
                dict(request["fields"]),
                list(request.get("tags", [])),
            )
            fingerprint = (int(note.mid), field_checksum(note.fields[0]))
            if fingerprint in fingerprints:
                raise DuplicateNoteError("batch contains duplicate first fields")
            fingerprints.add(fingerprint)
            prepared.append((note, int(request["deck_id"])))
        self.collection.add_notes(
            [
                AddNoteRequest(note=note, deck_id=cast("DeckId", deck_id))
                for note, deck_id in prepared
            ]
        )
        results = []
        for note, _ in prepared:
            card_ids = [
                int(card_id) for card_id in self.collection.find_cards(f"nid:{int(note.id)}")
            ]
            if not card_ids:  # pragma: no cover - validated normal note types generate cards
                raise RuntimeError("atomic note batch produced a note without cards")
            results.append({"note_id": int(note.id), "card_ids": card_ids, "created": True})
        return {"notes": results, "created": len(results)}

    def update_note_fields(self, note_id: int, fields: dict[str, str]) -> dict[str, Any]:
        if not fields:
            raise ValueError("at least one field update is required")
        try:
            note = self.collection.get_note(cast("NoteId", note_id))
        except NotFoundError as exc:
            raise LookupError(f"note {note_id} not found") from exc
        unknown = set(fields) - set(note.keys())
        if unknown:
            raise ValueError(f"unknown field: {', '.join(sorted(unknown))}")
        for name, value in fields.items():
            note[name] = value
        check = int(note.duplicate_or_empty())
        if check == 1:
            raise ValueError("the note's first field must not be empty")
        if check == 2:
            raise DuplicateNoteError("a note with the same first field already exists")
        self.collection.update_note(note)
        return {"note_id": note_id, "updated": True}

    def _update_note_tags(
        self, note_ids: list[int], tags: list[str], *, remove: bool
    ) -> dict[str, Any]:
        if not note_ids or len(note_ids) > self.max_batch_size:
            raise ValueError(
                f"note_ids must contain between 1 and {self.max_batch_size} stable IDs"
            )
        normalized = self._normalize_tags(tags)
        if not normalized:
            raise ValueError("tags must not be empty")
        notes = []
        for note_id in note_ids:
            try:
                note = self.collection.get_note(cast("NoteId", note_id))
            except NotFoundError as exc:
                raise LookupError(f"note {note_id} not found") from exc
            if remove:
                remove_names = {tag.casefold() for tag in normalized}
                note.tags = [tag for tag in note.tags if tag.casefold() not in remove_names]
            else:
                note.tags = list(dict.fromkeys([*note.tags, *normalized]))
            notes.append(note)
        self.collection.update_notes(notes)
        return {"updated_note_ids": note_ids}

    def add_note_tags(self, note_ids: list[int], tags: list[str]) -> dict[str, Any]:
        return self._update_note_tags(note_ids, tags, remove=False)

    def remove_note_tags(self, note_ids: list[int], tags: list[str]) -> dict[str, Any]:
        return self._update_note_tags(note_ids, tags, remove=True)

    def delete_notes(self, note_ids: list[int]) -> dict[str, Any]:
        if not note_ids or len(note_ids) > self.max_batch_size:
            raise ValueError(
                f"note_ids must contain between 1 and {self.max_batch_size} stable IDs"
            )
        if len(set(note_ids)) != len(note_ids):
            raise ValueError("note_ids must not contain duplicates")
        for note_id in note_ids:
            self.get_note(note_id)
        self.collection.remove_notes([cast("NoteId", note_id) for note_id in note_ids])
        return {"note_ids": note_ids, "deleted": len(note_ids)}

    def preview_notes_delete(self, note_ids: list[int]) -> dict[str, Any]:
        if not note_ids or len(note_ids) > self.max_batch_size:
            raise ValueError(
                f"note_ids must contain between 1 and {self.max_batch_size} stable IDs"
            )
        if len(set(note_ids)) != len(note_ids):
            raise ValueError("note_ids must not contain duplicates")
        cards = 0
        card_ids_by_note: dict[int, list[int]] = {}
        for note_id in note_ids:
            card_ids = self.get_note(note_id)["card_ids"]
            cards += len(card_ids)
            card_ids_by_note[note_id] = card_ids
        return {
            "notes": len(note_ids),
            "cards": cards,
            "state_fingerprint": self._impact_fingerprint(card_ids_by_note),
        }

    def list_tags(self, offset: int, limit: int) -> dict[str, Any]:
        tags = self.collection.tags.all()
        if len(tags) > self.max_search_scan:
            raise ValueError(
                "collection exceeds MCP_MAX_SEARCH_SCAN; use a larger configured bound"
            )
        items = []
        for tag in sorted(tags, key=str.casefold):
            name, name_truncated = self._truncate_rendered(tag)
            items.append({"name": name, "name_truncated": name_truncated})
        return self._page(items, offset, limit)

    def rename_tag(self, old_name: str, new_name: str) -> dict[str, Any]:
        if not old_name.strip() or not new_name.strip():
            raise ValueError("tag names must not be blank")
        if old_name not in self.collection.tags.all():
            raise LookupError(f"tag {old_name} not found")
        if int(self.collection.note_count()) > self.max_search_scan:
            raise ValueError(
                "collection exceeds MCP_MAX_SEARCH_SCAN; use a larger configured bound"
            )
        result = self.collection.tags.rename(old_name, new_name)
        return {
            "old_name": old_name,
            "new_name": new_name,
            "updated_notes": int(result.count),
        }

    def delete_tag(self, name: str) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("tag name must not be blank")
        if name not in self.collection.tags.all():
            raise LookupError(f"tag {name} not found")
        if int(self.collection.note_count()) > self.max_search_scan:
            raise ValueError(
                "collection exceeds MCP_MAX_SEARCH_SCAN; use a larger configured bound"
            )
        result = self.collection.tags.remove(name)
        return {"name": name, "updated_notes": int(result.count), "deleted": True}

    def preview_tag_delete(self, name: str) -> dict[str, Any]:
        if name not in self.collection.tags.all():
            raise LookupError(f"tag {name} not found")
        if int(self.collection.note_count()) > self.max_search_scan:
            raise ValueError(
                "collection exceeds MCP_MAX_SEARCH_SCAN; use a larger configured bound"
            )
        note_ids = [
            int(note_id)
            for note_id in self.collection.find_notes("")
            if name in self.collection.get_note(note_id).tags
        ]
        return {
            "notes": len(note_ids),
            "tag": name,
            "state_fingerprint": self._impact_fingerprint(sorted(note_ids)),
        }

    def list_note_types(self, offset: int, limit: int) -> dict[str, Any]:
        models = self.collection.models.all_names_and_ids()
        if len(models) > self.max_search_scan:
            raise ValueError(
                "collection exceeds MCP_MAX_SEARCH_SCAN; use a larger configured bound"
            )
        items = []
        for item in sorted(models, key=lambda model: model.name.casefold()):
            name, name_truncated = self._truncate_rendered(item.name)
            model = self.collection.models.get(cast("NotetypeId", item.id))
            items.append(
                {
                    "id": int(item.id),
                    "name": name,
                    "name_truncated": name_truncated,
                    "note_count": self.collection.models.use_count(model) if model else 0,
                }
            )
        return self._page(items, offset, limit)

    def get_note_type(self, note_type_id: int) -> dict[str, Any]:
        model = self.collection.models.get(cast("NotetypeId", note_type_id))
        if model is None:
            raise LookupError(f"note type {note_type_id} not found")
        name, name_truncated = self._truncate_rendered(str(model["name"]))
        css, css_truncated = self._truncate_rendered(str(model.get("css", "")))
        fields = []
        for field in model.get("flds", [])[: self.max_card_fields]:
            field_name, field_name_truncated = self._truncate_rendered(str(field["name"]))
            fields.append(
                {
                    "name": field_name,
                    "name_truncated": field_name_truncated,
                    "ordinal": int(field["ord"]),
                }
            )
        templates = []
        for template in model.get("tmpls", [])[: self.max_card_fields]:
            template_name, template_name_truncated = self._truncate_rendered(str(template["name"]))
            question, question_truncated = self._truncate_rendered(str(template.get("qfmt", "")))
            answer, answer_truncated = self._truncate_rendered(str(template.get("afmt", "")))
            templates.append(
                {
                    "name": template_name,
                    "name_truncated": template_name_truncated,
                    "ordinal": int(template["ord"]),
                    "question_format": question,
                    "question_format_truncated": question_truncated,
                    "answer_format": answer,
                    "answer_format_truncated": answer_truncated,
                }
            )
        return {
            "id": int(model["id"]),
            "name": name,
            "name_truncated": name_truncated,
            "kind": "cloze" if int(model.get("type", 0)) == 1 else "standard",
            "css": css,
            "css_truncated": css_truncated,
            "fields": fields,
            "fields_omitted": max(0, len(model.get("flds", [])) - self.max_card_fields),
            "templates": templates,
            "templates_omitted": max(0, len(model.get("tmpls", [])) - self.max_card_fields),
            "note_count": self.collection.models.use_count(model),
        }

    def _validate_note_type_parts(self, fields: list[str], templates: list[dict[str, str]]) -> None:
        if not fields or len(fields) > self.max_card_fields:
            raise ValueError(f"fields must contain between 1 and {self.max_card_fields} names")
        if any(not name.strip() for name in fields):
            raise ValueError("field names must not be blank")
        if len({name.casefold() for name in fields}) != len(fields):
            raise ValueError("field names must be unique")
        if not templates or len(templates) > self.max_card_fields:
            raise ValueError(f"templates must contain between 1 and {self.max_card_fields} items")
        names = [template["name"] for template in templates]
        if any(not name.strip() for name in names):
            raise ValueError("template names must not be blank")
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("template names must be unique")

    def create_note_type(
        self,
        name: str,
        fields: list[str],
        templates: list[dict[str, str]],
        css: str,
    ) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("note type name must not be blank")
        if self.collection.models.by_name(name) is not None:
            raise ValueError(f"note type {name} already exists")
        self._validate_note_type_parts(fields, templates)
        model = self.collection.models.new(name)
        for field_name in fields:
            self.collection.models.add_field(model, self.collection.models.new_field(field_name))
        for template_input in templates:
            template = self.collection.models.new_template(template_input["name"])
            template["qfmt"] = template_input["question_format"]
            template["afmt"] = template_input["answer_format"]
            self.collection.models.add_template(model, template)
        model["css"] = css
        self.collection.models.add(model)
        return {"id": int(model["id"]), "created": True}

    def update_note_type(
        self,
        note_type_id: int,
        name: str,
        fields: list[str],
        templates: list[dict[str, str]],
        css: str,
    ) -> dict[str, Any]:
        model = self.collection.models.get(cast("NotetypeId", note_type_id))
        if model is None:
            raise LookupError(f"note type {note_type_id} not found")
        if not name.strip():
            raise ValueError("note type name must not be blank")
        existing = self.collection.models.by_name(name)
        if existing is not None and int(existing["id"]) != note_type_id:
            raise ValueError(f"note type {name} already exists")
        self._validate_note_type_parts(fields, templates)
        current_fields = model.get("flds", [])
        current_templates = model.get("tmpls", [])
        if len(fields) != len(current_fields) or len(templates) != len(current_templates):
            raise ValueError("note type update must preserve field and template counts")
        model["name"] = name
        model["css"] = css
        for field, field_name in zip(current_fields, fields, strict=True):
            field["name"] = field_name
        for template, template_input in zip(current_templates, templates, strict=True):
            template["name"] = template_input["name"]
            template["qfmt"] = template_input["question_format"]
            template["afmt"] = template_input["answer_format"]
        self.collection.models.update_dict(model)
        return {"id": note_type_id, "updated": True}

    def update_note_type_fields(
        self, note_type_id: int, mappings: list[dict[str, Any]]
    ) -> dict[str, Any]:
        model = self.collection.models.get(cast("NotetypeId", note_type_id))
        if model is None:
            raise LookupError(f"note type {note_type_id} not found")
        if not mappings or len(mappings) > self.max_card_fields:
            raise ValueError(
                f"fields must contain between 1 and {self.max_card_fields} explicit mappings"
            )
        names = [str(mapping["name"]) for mapping in mappings]
        if any(not name.strip() for name in names):
            raise ValueError("field names must not be blank")
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("field names must be unique")
        current = list(model.get("flds", []))
        source_ordinals = [mapping.get("source_ordinal") for mapping in mappings]
        retained = [ordinal for ordinal in source_ordinals if ordinal is not None]
        if any(
            not isinstance(ordinal, int) or not 0 <= ordinal < len(current) for ordinal in retained
        ):
            raise ValueError("field source_ordinal does not identify an existing field")
        if len(set(retained)) != len(retained):
            raise ValueError("field source_ordinal values must not be reused")
        desired = []
        for name, source_ordinal in zip(names, source_ordinals, strict=True):
            if source_ordinal is None:
                field = self.collection.models.new_field(name)
                self.collection.models.add_field(model, field)
            else:
                field = current[source_ordinal]
                self.collection.models.rename_field(model, field, name)
            desired.append(field)
        for ordinal, field in enumerate(current):
            if ordinal not in retained:
                self.collection.models.remove_field(model, field)
        for index, field in enumerate(desired):
            self.collection.models.reposition_field(model, field, index)
        self.collection.models.update_dict(model)
        return {"id": note_type_id, "updated": True, "field_count": len(desired)}

    def update_templates(self, note_type_id: int, mappings: list[dict[str, Any]]) -> dict[str, Any]:
        model = self.collection.models.get(cast("NotetypeId", note_type_id))
        if model is None:
            raise LookupError(f"note type {note_type_id} not found")
        if int(model.get("type", 0)) == 1:
            raise ValueError("cloze note types do not support template structure changes")
        if not mappings or len(mappings) > self.max_card_fields:
            raise ValueError(
                f"templates must contain between 1 and {self.max_card_fields} explicit mappings"
            )
        names = [str(mapping["name"]) for mapping in mappings]
        if any(not name.strip() for name in names):
            raise ValueError("template names must not be blank")
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("template names must be unique")
        current = list(model.get("tmpls", []))
        source_ordinals = [mapping.get("source_ordinal") for mapping in mappings]
        retained = [ordinal for ordinal in source_ordinals if ordinal is not None]
        if any(
            not isinstance(ordinal, int) or not 0 <= ordinal < len(current) for ordinal in retained
        ):
            raise ValueError("template source_ordinal does not identify an existing template")
        if len(set(retained)) != len(retained):
            raise ValueError("template source_ordinal values must not be reused")
        desired = []
        for mapping, source_ordinal in zip(mappings, source_ordinals, strict=True):
            name = str(mapping["name"])
            if source_ordinal is None:
                template = self.collection.models.new_template(name)
                self.collection.models.add_template(model, template)
            else:
                template = current[source_ordinal]
                template["name"] = name
            template["qfmt"] = str(mapping["question_format"])
            template["afmt"] = str(mapping["answer_format"])
            desired.append(template)
        for ordinal, template in enumerate(current):
            if ordinal not in retained:
                self.collection.models.remove_template(model, template)
        for index, template in enumerate(desired):
            self.collection.models.reposition_template(model, template, index)
        self.collection.models.update_dict(model)
        return {"id": note_type_id, "updated": True, "template_count": len(desired)}

    def delete_note_type(self, note_type_id: int) -> dict[str, Any]:
        model = self.collection.models.get(cast("NotetypeId", note_type_id))
        if model is None:
            raise LookupError(f"note type {note_type_id} not found")
        note_count = int(self.collection.models.use_count(model))
        self.collection.models.remove(cast("NotetypeId", note_type_id))
        return {"id": note_type_id, "deleted": True, "deleted_notes": note_count}

    def preview_note_type_change(self, operation: str, note_type_id: int | None) -> dict[str, Any]:
        allowed = {"create", "update", "fields_update", "templates_update", "delete"}
        if operation not in allowed:
            raise ValueError("unsupported note type change operation")
        note_count = 0
        if operation != "create":
            if note_type_id is None:
                raise ValueError("note_type_id is required for this operation")
            note_count = int(self.get_note_type(note_type_id)["note_count"])
        elif note_type_id is not None:
            raise ValueError("note_type_id must be omitted for create")
        models = sorted(
            (
                {"id": int(model.id), "name": model.name}
                for model in self.collection.models.all_names_and_ids()
            ),
            key=lambda model: model["id"],
        )
        current_model = None
        if note_type_id is not None:
            model = self.collection.models.get(cast("NotetypeId", note_type_id))
            if model is None:  # pragma: no cover - get_note_type above validates this
                raise LookupError(f"note type {note_type_id} not found")
            current_model = {
                "name": str(model["name"]),
                "css": str(model.get("css", "")),
                "fields": [str(field["name"]) for field in model.get("flds", [])],
                "templates": [
                    {
                        "name": str(template["name"]),
                        "question_format": str(template["qfmt"]),
                        "answer_format": str(template["afmt"]),
                    }
                    for template in model.get("tmpls", [])
                ],
                "note_count": note_count,
            }
        return {
            "operation": operation,
            "note_type_id": note_type_id,
            "affected_notes": note_count,
            "backup_required": True,
            "full_sync_required": True,
            "state_fingerprint": self._impact_fingerprint(
                {"models": models, "current_model": current_model}
            ),
        }

    def _media_path(self, filename: str, *, require_exists: bool) -> Path:
        if (
            "\x00" in filename
            or not filename.strip()
            or Path(filename).name != filename
            or any(separator in filename for separator in ("/", "\\"))
        ):
            raise ValueError("media filename must be a plain filename without path separators")
        path = Path(self.collection.media.dir()) / filename
        if path.is_symlink():
            raise ValueError("media filename must not reference a symbolic link")
        if require_exists and not path.is_file():
            raise LookupError(f"media {filename} not found")
        return path

    def list_media(self, offset: int, limit: int) -> dict[str, Any]:
        items = []
        for path in Path(self.collection.media.dir()).iterdir():
            if path.is_file() and not path.is_symlink():
                items.append({"filename": path.name, "size_bytes": path.stat().st_size})
                if len(items) > self.max_search_scan:
                    raise ValueError(
                        "collection exceeds MCP_MAX_SEARCH_SCAN; use a larger configured bound"
                    )
        sorted_items = sorted(items, key=lambda item: item["filename"].casefold())
        return self._page(sorted_items, offset, limit)

    def get_media(self, filename: str) -> dict[str, Any]:
        path = self._media_path(filename, require_exists=True)
        size = path.stat().st_size
        if size > self.max_media_bytes:
            raise ValueError(f"media file exceeds ANKI_MAX_MEDIA_BYTES ({self.max_media_bytes})")
        return {
            "filename": filename,
            "size_bytes": size,
            "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        }

    def store_media(self, filename: str, content_base64: str) -> dict[str, Any]:
        path = self._media_path(filename, require_exists=False)
        try:
            content = base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("content_base64 must be valid base64") from exc
        if len(content) > self.max_media_bytes:
            raise ValueError(f"media content exceeds ANKI_MAX_MEDIA_BYTES ({self.max_media_bytes})")
        existed = self.collection.media.have(filename)
        previous_content = path.read_bytes() if existed else None
        try:
            if existed:
                self.collection.media.trash_files([filename])
            stored_name = self.collection.media.write_data(filename, content)
            if stored_name != filename:
                raise RuntimeError("Anki did not preserve the requested media filename")
        except Exception as operation_error:
            if previous_content is not None:
                try:
                    if self.collection.media.have(filename):
                        self.collection.media.trash_files([filename])
                    restored_name = self.collection.media.write_data(filename, previous_content)
                    if restored_name != filename or path.read_bytes() != previous_content:
                        raise RuntimeError("media replacement rollback verification failed")
                except Exception:
                    raise RuntimeError(
                        "media replacement failed and rollback was incomplete"
                    ) from operation_error
            else:
                try:
                    if self.collection.media.have(filename):
                        self.collection.media.trash_files([filename])
                    if self.collection.media.have(filename):
                        raise RuntimeError("created media remained after rollback")
                except Exception:
                    raise RuntimeError(
                        "media creation failed and rollback was incomplete"
                    ) from operation_error
            raise
        return {"filename": filename, "size_bytes": len(content), "created": not existed}

    def rename_media(self, old_filename: str, new_filename: str) -> dict[str, Any]:
        old_path = self._media_path(old_filename, require_exists=True)
        self._media_path(new_filename, require_exists=False)
        if self.collection.media.have(new_filename):
            raise ValueError(f"media {new_filename} already exists")
        content = old_path.read_bytes()
        if len(content) > self.max_media_bytes:
            raise ValueError(f"media file exceeds ANKI_MAX_MEDIA_BYTES ({self.max_media_bytes})")
        try:
            stored_name = self.collection.media.write_data(new_filename, content)
            if stored_name != new_filename:
                raise RuntimeError("Anki did not preserve the requested media filename")
            self.collection.media.trash_files([old_filename])
        except Exception as operation_error:
            try:
                if self.collection.media.have(new_filename):
                    self.collection.media.trash_files([new_filename])
                if not self.collection.media.have(old_filename):
                    restored_name = self.collection.media.write_data(old_filename, content)
                    if restored_name != old_filename:
                        raise RuntimeError("Anki did not restore the original media filename")
                if old_path.read_bytes() != content:
                    raise RuntimeError("media rename rollback verification failed")
            except Exception:
                raise RuntimeError(
                    "media rename failed and rollback was incomplete"
                ) from operation_error
            raise
        return {"old_filename": old_filename, "filename": stored_name, "updated": True}

    def delete_media(self, filename: str) -> dict[str, Any]:
        self._media_path(filename, require_exists=True)
        self.collection.media.trash_files([filename])
        return {"filename": filename, "deleted": True}

    def preview_media_delete(self, filename: str) -> dict[str, Any]:
        path = self._media_path(filename, require_exists=True)
        stat = path.stat()
        return {
            "filename": filename,
            "bytes": stat.st_size,
            "state_fingerprint": self._impact_fingerprint(
                {"size": stat.st_size, "modified_ns": stat.st_mtime_ns}
            ),
        }

    def check_media(self, offset: int, limit: int) -> dict[str, Any]:
        self._page([], offset, limit)
        response = self.collection.media.check()
        missing = sorted((str(name) for name in response.missing), key=str.casefold)
        unused = sorted((str(name) for name in response.unused), key=str.casefold)
        if len(missing) > self.max_search_scan or len(unused) > self.max_search_scan:
            raise ValueError(
                "media check exceeds MCP_MAX_SEARCH_SCAN; use a larger configured bound"
            )
        return {
            "missing": missing[offset : offset + limit],
            "unused": unused[offset : offset + limit],
            "missing_total": len(missing),
            "unused_total": len(unused),
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < max(len(missing), len(unused)),
            "have_trash": bool(response.have_trash),
        }

    def search_cards(self, query: str, offset: int, limit: int) -> dict[str, Any]:
        # Validate bounds before potentially expensive collection search.
        self._page([], offset, limit)
        if int(self.collection.card_count()) > self.max_search_scan:
            raise ValueError(
                "collection exceeds MCP_MAX_SEARCH_SCAN; use a larger configured bound"
            )
        card_ids = [int(card_id) for card_id in self.collection.find_cards(query)]
        result = self._page(card_ids, offset, limit)
        result["items"] = [self._card_summary(card_id) for card_id in result["items"]]
        return result

    def _card_summary(self, card_id: int) -> dict[str, Any]:
        card = self.collection.get_card(cast("CardId", card_id))
        return {"id": int(card.id), "note_id": int(card.nid), "deck_id": int(card.did)}

    def get_card(
        self, card_id: int, include_sections: Sequence[str] = ()
    ) -> dict[str, Any]:
        requested = set(include_sections)
        unknown = requested - {"review_summary", "fsrs"}
        if unknown:
            raise ValueError(f"unknown card sections: {', '.join(sorted(unknown))}")
        try:
            card = self.collection.get_card(cast("CardId", card_id))
        except NotFoundError as exc:
            raise LookupError(f"card {card_id} not found") from exc
        deck = self.collection.decks.get(card.did, default=False)
        if deck is None:
            raise LookupError(f"deck for card {card_id} not found")
        question, question_truncated = self._truncate_rendered(card.question())
        answer, answer_truncated = self._truncate_rendered(card.answer())
        deck_name, deck_name_truncated = self._truncate_rendered(str(deck["name"]))
        note = card.note()
        fields: list[dict[str, Any]] = []
        note_items = note.items()
        for name, value in note_items[: self.max_card_fields]:
            bounded_name, name_truncated = self._truncate_rendered(name)
            bounded_value, value_truncated = self._truncate_rendered(value)
            fields.append(
                {
                    "name": bounded_name,
                    "name_truncated": name_truncated,
                    "value": bounded_value,
                    "value_truncated": value_truncated,
                }
            )
        result: dict[str, Any] = {
            "id": int(card.id),
            "note_id": int(card.nid),
            "deck_id": int(card.did),
            "deck_name": deck_name,
            "deck_name_truncated": deck_name_truncated,
            "template_ordinal": int(card.ord),
            "flags": int(card.flags),
            "modified": int(card.mod),
            "question": question,
            "question_truncated": question_truncated,
            "answer": answer,
            "answer_truncated": answer_truncated,
            "fields": fields,
            "fields_omitted": max(0, len(note_items) - self.max_card_fields),
            "scheduling": {
                "type": int(card.type),
                "queue": int(card.queue),
                "due": int(card.due),
                "interval": int(card.ivl),
                "ease_factor": int(card.factor),
                "reps": int(card.reps),
                "lapses": int(card.lapses),
                "left": int(card.left),
            },
        }
        if not requested:
            return result

        stats = self.collection.card_stats_data(card.id)
        if "review_summary" in requested:
            result["review_summary"] = {
                "added_at": int(stats.added),
                "first_review_at": int(stats.first_review) or None,
                "latest_review_at": int(stats.latest_review) or None,
                "reviews": int(stats.reviews),
                "lapses": int(stats.lapses),
                "average_answer_seconds": float(stats.average_secs),
                "total_answer_seconds": float(stats.total_secs),
                "preset": str(stats.preset),
                "original_deck": str(stats.original_deck) or None,
            }
        if "fsrs" in requested:
            result["fsrs"] = {
                "memory_state": (
                    {
                        "stability": float(stats.memory_state.stability),
                        "difficulty": float(stats.memory_state.difficulty),
                    }
                    if stats.HasField("memory_state")
                    else None
                ),
                "retrievability": (
                    float(stats.fsrs_retrievability)
                    if stats.HasField("memory_state")
                    else None
                ),
                "desired_retention": (
                    float(stats.desired_retention) if stats.HasField("memory_state") else None
                ),
                "parameters": [float(value) for value in stats.fsrs_params],
            }
        return result

    def _review_scope(
        self,
        card_id: int | None,
        deck_id: int | None,
        query: str | None,
        include_children: bool,
    ) -> tuple[list[int], str, dict[str, Any], str]:
        selectors = (card_id is not None, deck_id is not None, query is not None)
        if sum(selectors) != 1:
            raise ValueError("exactly one of card_id, deck_id, or query must be provided")
        if include_children and deck_id is None:
            raise ValueError("include_children is only valid with deck_id")

        if card_id is not None:
            try:
                self.collection.get_card(cast("CardId", card_id))
            except NotFoundError as exc:
                raise LookupError(f"card {card_id} not found") from exc
            search = f"cid:{card_id}"
            scope = {"kind": "card", "card_id": card_id}
            attribution = "exact_card"
            return [card_id], search, scope, attribution
        elif deck_id is not None:
            deck = self.get_deck(deck_id)
            deck_ids = [deck_id]
            if include_children:
                database = self.collection.db
                if database is None:  # pragma: no cover - reads require an open collection
                    raise RuntimeError("collection database is unavailable")
                deck_count = int(database.scalar("select count() from decks") or 0)
                if deck_count > self.max_search_scan:
                    raise ValueError(
                        "deck hierarchy exceeds MCP_MAX_SEARCH_SCAN; raise the configured bound"
                    )
                deck_name = str(deck["name"])
                deck_ids.extend(
                    int(item.id)
                    for item in self.collection.decks.all_names_and_ids()
                    if item.name.startswith(f"{deck_name}::")
                )
            search = " OR ".join(f"did:{value}" for value in deck_ids)
            scope = {
                "kind": "deck",
                "deck_id": deck_id,
                "include_children": include_children,
            }
            attribution = "current_card_membership"
        else:
            search = cast(str, query)
            scope = {"kind": "query", "query": search}
            attribution = "current_card_membership"

        if int(self.collection.card_count()) > self.max_search_scan:
            raise ValueError(
                "collection exceeds MCP_MAX_SEARCH_SCAN; narrow card scope is required"
            )
        card_ids = [int(value) for value in self.collection.find_cards(search)]
        if len(card_ids) > self.max_search_scan:
            raise ValueError(
                "review scope exceeds MCP_MAX_SEARCH_SCAN; narrow the scope or raise the bound"
            )
        return card_ids, search, scope, attribution

    def _review_log_count(self, card_ids: list[int]) -> int:
        database = self.collection.db
        if database is None:  # pragma: no cover - reads only run against an open collection
            raise RuntimeError("collection database is unavailable")
        total = 0
        for start in range(0, len(card_ids), 500):
            chunk = card_ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            total += int(
                database.scalar(f"select count() from revlog where cid in ({placeholders})", *chunk)
                or 0
            )
        return total

    @staticmethod
    def _review_event(card_id: int, entry: Any, include_fields: set[str]) -> dict[str, Any]:
        rating = int(entry.button_chosen)
        result: dict[str, Any] = {
            "card_id": card_id,
            "reviewed_at": int(entry.time),
            "rating": rating,
        }
        if "review_kind" in include_fields:
            kinds = ("learning", "review", "relearning", "filtered", "manual", "rescheduled")
            kind = int(entry.review_kind)
            result["review_kind"] = kinds[kind] if 0 <= kind < len(kinds) else "unknown"
        if "rating_label" in include_fields:
            result["rating_label"] = (
                ("", "Again", "Hard", "Good", "Easy")[rating]
                if 1 <= rating <= 4
                else None
            )
        if "intervals" in include_fields:
            result["interval_seconds"] = int(entry.interval)
            result["previous_interval_seconds"] = int(entry.last_interval)
        if "answer_time" in include_fields:
            result["answer_seconds"] = float(entry.taken_secs)
        if "ease" in include_fields:
            result["ease"] = int(entry.ease)
        if "memory_state" in include_fields:
            result["memory_state"] = (
                {
                    "stability": float(entry.memory_state.stability),
                    "difficulty": float(entry.memory_state.difficulty),
                }
                if entry.HasField("memory_state")
                else None
            )
        return result

    def list_reviews(
        self,
        card_id: int | None,
        deck_id: int | None,
        query: str | None,
        include_children: bool,
        offset: int,
        limit: int,
        order: str,
        include_fields: Sequence[str] = (),
    ) -> dict[str, Any]:
        self._page([], offset, limit)
        if order not in {"newest", "oldest"}:
            raise ValueError("order must be newest or oldest")
        requested_fields = set(include_fields)
        unknown = requested_fields - {
            "review_kind",
            "rating_label",
            "intervals",
            "answer_time",
            "ease",
            "memory_state",
        }
        if unknown:
            raise ValueError(f"unknown review fields: {', '.join(sorted(unknown))}")
        card_ids, _, scope, attribution = self._review_scope(
            card_id, deck_id, query, include_children
        )
        total = self._review_log_count(card_ids)
        if total > self.max_search_scan:
            raise ValueError(
                "review history exceeds MCP_MAX_SEARCH_SCAN; narrow the scope or raise the bound"
            )
        events = [
            self._review_event(current_card_id, entry, requested_fields)
            for current_card_id in card_ids
            for entry in self.collection.get_review_logs(cast("CardId", current_card_id))
        ]
        events.sort(
            key=lambda event: (event["reviewed_at"], event["card_id"]),
            reverse=order == "newest",
        )
        return {
            "items": events[offset : offset + limit],
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + limit < total,
            "scope": scope,
            "attribution": attribution,
        }

    @staticmethod
    def _review_summary(graphs: dict[str, Any], days: int) -> dict[str, Any]:
        reviews = graphs["reviews"]
        counts_by_day = reviews["count"]
        time_by_day = reviews["time"]
        daily: list[dict[str, Any]] = []
        total_reviews = 0
        total_millis = 0
        for day_text in sorted(counts_by_day, key=int):
            count = sum(int(value) for value in counts_by_day[day_text].values())
            millis = sum(int(value) for value in time_by_day.get(day_text, {}).values())
            total_reviews += count
            total_millis += millis
            daily.append(
                {
                    "day": int(day_text),
                    "reviews": count,
                    "answer_seconds": millis / 1_000,
                }
            )

        period = {0: "all_time", 30: "one_month", 90: "three_months", 365: "one_year"}[
            days
        ]
        button_period = graphs["buttons"][period]
        rating_counts = [
            sum(int(values[index]) for values in button_period.values()) for index in range(4)
        ]
        retained = sum(
            int(value)
            for category in ("young", "mature")
            for value in button_period[category][1:]
        )
        forgotten = sum(
            int(button_period[category][0]) for category in ("young", "mature")
        )
        retention_total = retained + forgotten
        total_seconds = total_millis / 1_000
        return {
            "reviews": total_reviews,
            "answer_seconds": total_seconds,
            "average_answer_seconds": (
                total_seconds / total_reviews if total_reviews else None
            ),
            "retention": retained / retention_total if retention_total else None,
            "ratings": dict(zip(("again", "hard", "good", "easy"), rating_counts, strict=True)),
            "daily": daily,
        }

    def review_stats(
        self,
        card_id: int | None,
        deck_id: int | None,
        query: str | None,
        include_children: bool,
        days: int,
        include_sections: Sequence[str] = (),
    ) -> dict[str, Any]:
        if days not in {0, 30, 90, 365}:
            raise ValueError("days must be one of 0, 30, 90, or 365")
        allowed_sections = {
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
        }
        requested_sections = list(dict.fromkeys(include_sections))
        unknown = set(requested_sections) - allowed_sections
        if unknown:
            raise ValueError(f"unknown review statistic sections: {', '.join(sorted(unknown))}")
        card_ids, search, scope, attribution = self._review_scope(
            card_id, deck_id, query, include_children
        )
        if self._review_log_count(card_ids) > self.max_search_scan:
            raise ValueError(
                "review history exceeds MCP_MAX_SEARCH_SCAN; narrow the scope or raise the bound"
            )
        graphs = self.collection._backend.graphs(  # pyright: ignore[reportPrivateUsage]
            search=search, days=days
        )
        graph_data = MessageToDict(
            graphs,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
        result: dict[str, Any] = {
            "scope": scope,
            "attribution": attribution,
            "days": days,
            "summary": self._review_summary(graph_data, days),
        }
        if requested_sections:
            result["sections"] = {
                section: graph_data[section] for section in requested_sections
            }
        return result

    def _basic_model(self) -> dict[str, Any]:
        model = self.collection.models.by_name("Basic")
        if model is None:
            raise RuntimeError("Basic note type is unavailable")
        fields = model.get("flds", [])
        templates = model.get("tmpls", [])
        supported = (
            int(model.get("type", -1)) == 0
            and [(field.get("name"), field.get("ord")) for field in fields]
            == [("Front", 0), ("Back", 1)]
            and len(templates) == 1
            and int(templates[0].get("ord", -1)) == 0
        )
        if not supported:
            raise ValueError("card operations require the built-in Basic single-card note type")
        return model

    def create_card(self, deck_id: int, front: str, back: str) -> dict[str, Any]:
        self.get_deck(deck_id)
        if not front.strip():
            raise ValueError("front must not be blank")
        model = self._basic_model()
        note = self.collection.new_note(model)
        note["Front"] = front
        note["Back"] = back
        self.collection.add_note(note, cast("DeckId", deck_id))
        try:
            card_ids = self.collection.find_cards(f"nid:{int(note.id)}")
            if len(card_ids) != 1:
                raise ValueError("Basic note must generate exactly one card")
        except Exception:
            self.collection.remove_notes([note.id])
            raise
        return {
            "id": int(card_ids[0]),
            "note_id": int(note.id),
            "deck_id": deck_id,
            "created": True,
        }

    def update_card(
        self, card_id: int, front: str | None, back: str | None, deck_id: int | None
    ) -> dict[str, Any]:
        try:
            card = self.collection.get_card(cast("CardId", card_id))
        except NotFoundError as exc:
            raise LookupError(f"card {card_id} not found") from exc
        if front is None and back is None and deck_id is None:
            raise ValueError("at least one card update must be provided")
        if front is not None and not front.strip():
            raise ValueError("front must not be blank")
        original_deck_id = int(card.did)
        if deck_id is not None:
            self.get_deck(deck_id)
        note = card.note()
        basic = self._basic_model()
        basic_id = int(basic["id"])
        sibling_cards = self.collection.find_cards(f"nid:{int(note.id)}")
        if int(note.mid) != basic_id or len(sibling_cards) != 1:
            raise ValueError("card updates require the built-in Basic single-card note type")
        previous_front = note["Front"]
        previous_back = note["Back"]
        note_changed = front is not None or back is not None
        if note_changed:
            if front is not None:
                note["Front"] = front
            if back is not None:
                note["Back"] = back
            try:
                self.collection.update_note(note)
                if len(self.collection.find_cards(f"nid:{int(note.id)}")) != 1:
                    raise ValueError("Basic note update must retain exactly one card")
            except Exception as operation_error:
                note["Front"] = previous_front
                note["Back"] = previous_back
                try:
                    self.collection.update_note(note)
                except Exception:
                    raise RuntimeError(
                        "card update failed and field rollback was incomplete"
                    ) from operation_error
                raise operation_error
        try:
            if deck_id is not None:
                self.collection.set_deck([cast("CardId", card_id)], deck_id)
        except Exception as operation_error:
            rollback_failed = False
            try:
                self.collection.set_deck([cast("CardId", card_id)], original_deck_id)
            except Exception:
                rollback_failed = True
            if note_changed:
                note["Front"] = previous_front
                note["Back"] = previous_back
                try:
                    self.collection.update_note(note)
                except Exception:
                    rollback_failed = True
            if rollback_failed:
                raise RuntimeError(
                    "card update failed and rollback was incomplete"
                ) from operation_error
            raise operation_error
        return {
            "id": card_id,
            "note_id": int(note.id),
            "deck_id": deck_id if deck_id is not None else int(card.did),
            "updated": True,
        }

    def delete_card(self, card_id: int) -> dict[str, Any]:
        self.get_card(card_id)
        try:
            self.collection.remove_cards_and_orphaned_notes([cast("CardId", card_id)])
        except Exception as operation_error:
            self._raise_after_delete_failure(
                operation_error, "card", lambda: self.get_card(card_id)
            )
        return {"id": card_id, "deleted": True}

    def preview_card_delete(self, card_id: int) -> dict[str, Any]:
        card = self.get_card(card_id)
        sibling_ids = [
            int(sibling_id) for sibling_id in self.collection.find_cards(f"nid:{card['note_id']}")
        ]
        return {
            "cards": 1,
            "note_id": card["note_id"],
            "orphaned_note_deleted": len(sibling_ids) == 1,
            "state_fingerprint": self._impact_fingerprint(sorted(sibling_ids)),
        }

    def answer_card(self, card_id: int, rating: int, answer_seconds: int) -> dict[str, Any]:
        if rating not in {1, 2, 3, 4}:
            raise ValueError("rating must be between 1 and 4")
        if not 0 <= answer_seconds <= 86_400:
            raise ValueError("answer_seconds must be between 0 and 86400")
        try:
            card = self.collection.get_card(cast("CardId", card_id))
        except NotFoundError as exc:
            raise LookupError(f"card {card_id} not found") from exc
        card.timer_started = time.time() - answer_seconds
        self.collection.sched.answerCard(card, cast("Any", rating))
        return {"id": card_id, "rating": rating, "answered": True}

    def _validate_card_ids(self, card_ids: list[int]) -> None:
        if not card_ids or len(card_ids) > self.max_batch_size:
            raise ValueError(
                f"card_ids must contain between 1 and {self.max_batch_size} stable IDs"
            )
        for card_id in card_ids:
            self.get_card(card_id)

    def change_card_deck(self, card_ids: list[int], deck_id: int) -> dict[str, Any]:
        self._validate_card_ids(card_ids)
        self.get_deck(deck_id)
        self.collection.set_deck([cast("CardId", card_id) for card_id in card_ids], deck_id)
        return {"card_ids": card_ids, "deck_id": deck_id, "updated": True}

    def suspend_cards(self, card_ids: list[int]) -> dict[str, Any]:
        self._validate_card_ids(card_ids)
        self.collection.sched.suspend_cards([cast("CardId", card_id) for card_id in card_ids])
        return {"card_ids": card_ids, "suspended": True}

    def unsuspend_cards(self, card_ids: list[int]) -> dict[str, Any]:
        self._validate_card_ids(card_ids)
        self.collection.sched.unsuspend_cards([cast("CardId", card_id) for card_id in card_ids])
        return {"card_ids": card_ids, "suspended": False}

    def set_card_flag(self, card_ids: list[int], flag: int) -> dict[str, Any]:
        self._validate_card_ids(card_ids)
        if not 0 <= flag <= 7:
            raise ValueError("flag must be between 0 and 7")
        changes = self.collection._backend.set_flag(  # pyright: ignore[reportPrivateUsage]
            card_ids=card_ids, flag=flag
        )
        return {"card_ids": card_ids, "flag": flag, "updated": int(changes.count)}

    def reposition_cards(
        self,
        card_ids: list[int],
        starting_from: int,
        step_size: int,
        randomize: bool,
        shift_existing: bool,
    ) -> dict[str, Any]:
        self._validate_card_ids(card_ids)
        if starting_from < 1:
            raise ValueError("starting_from must be at least 1")
        if step_size < 1:
            raise ValueError("step_size must be at least 1")
        for card_id in card_ids:
            card = self.collection.get_card(cast("CardId", card_id))
            if int(card.type) != 0:
                raise ValueError("card repositioning only supports new cards")
        changes = self.collection.sched.reposition_new_cards(
            [cast("CardId", card_id) for card_id in card_ids],
            starting_from,
            step_size,
            randomize,
            shift_existing,
        )
        return {"card_ids": card_ids, "repositioned": int(changes.count)}

    def sync_login(self, username: str, password: str, endpoint: str | None) -> dict[str, Any]:
        self._sync_auth = None
        self._configured_sync_endpoint = None
        self._pending_full_sync = None
        self._state.clear_sync_auth()
        self._save_operational_status()
        self._sync_auth = self.collection.sync_login(username, password, endpoint)
        self._configured_sync_endpoint = endpoint or None
        self._state.save_sync_auth(
            {
                "hkey": self._sync_auth.hkey,
                "endpoint": self._sync_auth.endpoint or "",
                "configured_endpoint": self._configured_sync_endpoint or "",
            }
        )
        return {"authenticated": True, "endpoint_kind": "custom" if endpoint else "ankiweb"}

    def _wait_for_media_sync(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.sync_timeout_seconds
        while True:
            status = self.collection._backend.media_sync_status()  # pyright: ignore[reportPrivateUsage]
            progress = {
                "checked": str(status.progress.checked),
                "added": str(status.progress.added),
                "removed": str(status.progress.removed),
            }
            if progress != self._media_sync_progress:
                self._media_sync_progress = progress
                self._save_operational_status()
            if not status.active:
                self._last_media_sync_at = datetime.now(UTC).isoformat()
                self._save_operational_status()
                return {"completed": True, **progress}
            if time.monotonic() >= deadline:
                try:
                    self.collection.abort_media_sync()
                except Exception as exc:
                    raise MediaSyncFailedError(
                        "media synchronization timed out and could not be aborted"
                    ) from exc
                raise MediaSyncFailedError(
                    "media synchronization did not complete before the sync timeout"
                )
            time.sleep(0.05)

    def sync(self, sync_media: bool) -> dict[str, Any]:
        if self._sync_auth is None:
            raise SyncLoginRequiredError("sync login is required before synchronization")
        try:
            output = self.collection.sync_collection(self._sync_auth, sync_media)
            endpoint_changed = bool(output.new_endpoint)
            if endpoint_changed:
                self._sync_auth.endpoint = validate_sync_migration_endpoint(
                    output.new_endpoint, self._configured_sync_endpoint
                )
            if output.required in {2, 3, 4}:
                self._pending_full_sync = (
                    output.required,
                    output.server_media_usn if sync_media else None,
                )
            else:
                self._pending_full_sync = None
            self._last_sync_at = datetime.now(UTC).isoformat()
            required = SYNC_REQUIRED_NAMES[output.required]
            if self._post_restore_upload and required in {"NO_CHANGES", "NORMAL_SYNC"}:
                self._post_restore_upload = False
            self._state.save_sync_auth(
                {
                    "hkey": self._sync_auth.hkey,
                    "endpoint": self._sync_auth.endpoint or "",
                    "configured_endpoint": self._configured_sync_endpoint or "",
                }
            )
            self._save_operational_status()
            media_sync = self._wait_for_media_sync() if sync_media else None
            server_message, server_message_truncated = self._truncate_rendered(
                output.server_message
            )
            return {
                "required": required,
                "server_message": server_message,
                "server_message_truncated": server_message_truncated,
                "host_number": output.host_number,
                "media_sync_requested": sync_media,
                "media_sync": media_sync,
                "endpoint_changed": endpoint_changed,
            }
        except Exception as exc:
            if not isinstance(exc, (NetworkError, TimeoutError)):
                self._invalidate_sync_auth()
            raise

    def full_sync(self, upload: bool, force: bool = False) -> dict[str, Any]:
        if self._sync_auth is None:
            raise SyncLoginRequiredError("sync login is required before full synchronization")
        if force:
            if not upload:
                raise ValueError("force is only supported for a full upload")
            if not self._post_restore_upload:
                raise ValueError("no post-restore upload is pending")
            server_usn = None
        else:
            if self._pending_full_sync is None:
                raise ValueError("a full sync was not requested by the remote server")
            required, server_usn = self._pending_full_sync
            if required == 3 and upload:
                raise ValueError("the remote server requires a full download")
            if required == 4 and not upload:
                raise ValueError("the remote server requires a full upload")
        try:
            self._backup_folder.mkdir(parents=True, exist_ok=True)
            backup_created = self.collection.create_backup(
                backup_folder=str(self._backup_folder),
                force=True,
                wait_for_completion=True,
            )
            self.collection.full_upload_or_download(
                auth=self._sync_auth,
                server_usn=server_usn,
                upload=upload,
            )
        except Exception as exc:
            if not isinstance(exc, NetworkError):
                self._invalidate_sync_auth()
            raise
        self._pending_full_sync = None
        self._post_restore_upload = False
        self._last_sync_at = datetime.now(UTC).isoformat()
        if upload:
            self._state.mark_all_remote_synced()
        else:
            self._state.mark_pending_discarded_by_full_download()
        self._save_operational_status()
        return {
            "completed": True,
            "direction": "upload" if upload else "download",
            "backup_created": backup_created,
        }

    def _export_card_count(self, deck_id: int | None) -> int:
        if deck_id is None:
            return int(self.collection.card_count())
        self.get_deck(deck_id)
        deck_ids = self.collection.decks.deck_and_child_ids(cast("DeckId", deck_id))
        return sum(len(self.collection.find_cards(f"did:{int(did)}")) for did in deck_ids)

    def _require_within_export_bound(self, deck_id: int | None) -> None:
        if self._export_card_count(deck_id) > self.max_search_scan:
            raise ResourceLimitError(
                "export scope exceeds MCP_MAX_SEARCH_SCAN; use a larger configured bound"
            )

    def _new_export_path(self, suffix: str) -> tuple[str, Path]:
        filename = (
            f"export-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}{suffix}"
        )
        if (
            "\x00" in filename
            or not filename.strip()
            or Path(filename).name != filename
            or any(separator in filename for separator in ("/", "\\"))
        ):
            raise ValueError("export filename must be a plain filename without path separators")
        self._exports_folder.mkdir(parents=True, exist_ok=True)
        path = self._exports_folder / filename
        if path.is_symlink():
            raise ValueError("export filename must not reference a symbolic link")
        return filename, path

    def _export_result(self, filename: str, path: Path, extra: dict[str, Any]) -> dict[str, Any]:
        size = path.stat().st_size
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        return {
            "filename": filename,
            "path": str(path.resolve()),
            "size_bytes": size,
            "sha256": digest,
            **extra,
        }

    def export_apkg(
        self,
        deck_id: int | None,
        include_media: bool,
        include_scheduling: bool,
        include_deck_configs: bool,
    ) -> dict[str, Any]:
        """Export the collection (or one deck) to a generated ``.apkg`` file."""
        self._require_within_export_bound(deck_id)
        filename, path = self._new_export_path(".apkg")
        temp_path = path.with_name(f"{path.name}.{uuid4().hex[:8]}.tmp{path.suffix}")
        try:
            exported_notes = int(
                self.collection.export_anki_package(
                    out_path=str(temp_path),
                    options=ExportAnkiPackageOptions(
                        with_scheduling=include_scheduling,
                        with_deck_configs=include_deck_configs,
                        with_media=include_media,
                        legacy=False,
                    ),
                    limit=DeckIdLimit(cast("DeckId", deck_id)) if deck_id is not None else None,
                )
            )
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        return self._export_result(filename, path, {"exported_notes": exported_notes})

    def export_notes_csv(
        self,
        deck_id: int | None,
        with_html: bool,
        with_tags: bool,
        with_deck: bool,
        with_notetype: bool,
        with_guid: bool,
        inline: bool,
    ) -> dict[str, Any]:
        """Export notes (whole collection or one deck) to a generated ``.csv`` file."""
        self._require_within_export_bound(deck_id)
        filename, path = self._new_export_path(".csv")
        temp_path = path.with_name(f"{path.name}.{uuid4().hex[:8]}.tmp{path.suffix}")
        try:
            rows = int(
                self.collection.export_note_csv(
                    out_path=str(temp_path),
                    limit=DeckIdLimit(cast("DeckId", deck_id)) if deck_id is not None else None,
                    with_html=with_html,
                    with_tags=with_tags,
                    with_deck=with_deck,
                    with_notetype=with_notetype,
                    with_guid=with_guid,
                )
            )
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        result = self._export_result(filename, path, {"rows": rows})
        if inline:
            size = int(result["size_bytes"])
            if 4 * ((size + 2) // 3) + 1024 > self.max_response_bytes:
                result["inline_omitted"] = True
                result["inline_reason"] = "inline payload would exceed MCP_MAX_RESPONSE_BYTES"
            else:
                encoded = base64.b64encode(path.read_bytes())
                if len(encoded) + 1024 <= self.max_response_bytes:
                    result["content_base64"] = encoded.decode("ascii")
                else:
                    result["inline_omitted"] = True
                    result["inline_reason"] = "inline payload would exceed MCP_MAX_RESPONSE_BYTES"
        return result

    def _import_path(self, filename: str, *, require_exists: bool) -> Path:
        if (
            "\x00" in filename
            or not filename.strip()
            or Path(filename).name != filename
            or any(separator in filename for separator in ("/", "\\"))
        ):
            raise ValueError("import filename must be a plain filename without path separators")
        if not filename.endswith(IMPORT_SUFFIXES):
            raise ValueError("import filename must end with .apkg or .csv")
        path = self._imports_folder / filename
        if path.is_symlink():
            raise ValueError("import filename must not reference a symbolic link")
        if path.resolve().parent != self._imports_folder.resolve():
            raise ValueError("import filename must resolve inside the imports folder")
        if require_exists and not path.is_file():
            raise LookupError(f"import {filename} not found")
        if path.is_file() and path.stat().st_size > self.max_import_bytes:
            raise ResourceLimitError(
                f"import {filename} exceeds ANKI_MAX_IMPORT_BYTES ({self.max_import_bytes})"
            )
        return path

    def list_import_files(self, offset: int, limit: int) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        if self._imports_folder.is_dir():
            for path in self._imports_folder.iterdir():
                if path.is_file() and not path.is_symlink() and path.suffix in IMPORT_SUFFIXES:
                    stat = path.stat()
                    items.append(
                        {
                            "filename": path.name,
                            "size_bytes": stat.st_size,
                            "mtime": stat.st_mtime,
                        }
                    )
        items.sort(key=lambda item: item["mtime"], reverse=True)
        return self._page(items, offset, limit)

    def preview_import_apkg_file(self, filename: str) -> dict[str, Any]:
        path = self._import_path(filename, require_exists=True)
        if path.suffix != ".apkg":
            raise ValueError("apkg import requires a .apkg file")
        try:
            with ZipFile(path) as archive:
                infos = archive.infolist()
                # Reject declared zip bombs before testzip() inflates every member.
                declared_size = sum(info.file_size for info in infos)
                if declared_size > self.max_import_bytes:
                    raise ResourceLimitError(
                        f"import {filename} declares uncompressed content larger than "
                        f"ANKI_MAX_IMPORT_BYTES ({self.max_import_bytes})"
                    )
                members = {info.filename for info in infos}
                if archive.testzip() is not None:
                    raise ValueError("import package failed its integrity check")
            if "meta" not in members or not members & {
                "collection.anki21b",
                "collection.anki2",
            }:
                raise ValueError("import package is missing required collection members")
            stat = path.stat()
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        except BadZipFile as exc:
            raise ValueError("import package is not a valid zip archive") from exc
        except IMPORT_FILE_FAILURES as exc:
            raise ImportFileError(f"staged import file could not be read: {exc}") from exc
        return {
            "filename": filename,
            "size_bytes": stat.st_size,
            "sha256": digest,
            "validated": True,
        }

    def import_apkg(
        self,
        filename: str,
        merge_notetypes: bool,
        update_notes: ImportAnkiPackageUpdateCondition.ValueType,
        update_notetypes: ImportAnkiPackageUpdateCondition.ValueType,
        with_scheduling: bool,
        with_deck_configs: bool,
    ) -> dict[str, Any]:
        path = self._import_path(filename, require_exists=True)
        try:
            # Re-validate the archive so a file swapped after preview cannot be imported.
            self.preview_import_apkg_file(filename)
            response = self.collection.import_anki_package(
                ImportAnkiPackageRequest(
                    package_path=str(path),
                    options=ImportAnkiPackageOptions(
                        merge_notetypes=merge_notetypes,
                        update_notes=update_notes,
                        update_notetypes=update_notetypes,
                        with_scheduling=with_scheduling,
                        with_deck_configs=with_deck_configs,
                    ),
                )
            )
        except IMPORT_FILE_FAILURES as exc:
            raise ImportFileError(f"staged import file could not be read: {exc}") from exc
        log = response.log
        return {
            "filename": filename,
            "notes_added": len(log.new),
            "notes_updated": len(log.updated),
            "notes_duplicate": len(log.duplicate),
            "notes_conflicting": len(log.conflicting),
            "notes_matched_first_field": len(log.first_field_match),
            "notes_found": int(log.found_notes),
            "merge_notetypes": merge_notetypes,
            "update_notes": update_notes,
            "update_notetypes": update_notetypes,
            "with_scheduling": with_scheduling,
            "with_deck_configs": with_deck_configs,
        }

    @staticmethod
    def _csv_delimiter(delimiter: str | None) -> CsvMetadata.Delimiter.ValueType | None:
        if delimiter is None:
            return None
        value = CSV_DELIMITER_CHARS.get(delimiter)
        if value is None:
            raise ValueError("delimiter must be one of tab, |, ;, :, comma, or space")
        return value

    def _csv_sample_rows(
        self, path: Path, delimiter_char: str, *, max_rows: int = 5, max_bytes: int = 4096
    ) -> list[list[str]]:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            content = handle.read(max_bytes)
        lines = [
            line for line in content.splitlines() if line.strip() and not line.startswith("#")
        ]
        if len(content) >= max_bytes and content and not content.endswith("\n"):
            lines = lines[:-1]
        sample: list[list[str]] = []
        for row in csv.reader(lines, delimiter=delimiter_char):
            sample.append([cell[:256] for cell in row])
            if len(sample) >= max_rows:
                break
        return sample

    def preview_import_csv(self, filename: str, delimiter: str | None) -> dict[str, Any]:
        path = self._import_path(filename, require_exists=True)
        if path.suffix != ".csv":
            raise ValueError("csv import requires a .csv file")
        delimiter_value = self._csv_delimiter(delimiter)
        try:
            metadata = self.collection.get_csv_metadata(str(path), delimiter_value)
            effective = (
                delimiter
                if delimiter is not None
                else CSV_DELIMITER_NAMES.get(int(metadata.delimiter), "\t")
            )
            stat = path.stat()
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            sample_rows = self._csv_sample_rows(path, effective)
        except IMPORT_FILE_FAILURES as exc:
            raise ImportFileError(f"staged import file could not be read: {exc}") from exc
        return {
            "filename": filename,
            "size_bytes": stat.st_size,
            "sha256": digest,
            "delimiter": effective,
            "is_html": bool(metadata.is_html),
            "column_labels": list(metadata.column_labels),
            "sample_rows": sample_rows,
        }

    @staticmethod
    def _validate_field_columns(
        field_columns: Sequence[int], metadata: CsvMetadata, filename: str
    ) -> None:
        column_count = len(metadata.column_labels)
        for column in field_columns:
            if column < 0 or column > column_count:
                raise ValueError(
                    f"field_columns entry {column} is outside the {column_count} "
                    f"columns of {filename}"
                )
        nonzero = [column for column in field_columns if column != 0]
        if len(nonzero) != len(set(nonzero)):
            raise ValueError("field_columns must not map more than one field to the same column")

    def import_csv(
        self,
        filename: str,
        notetype_id: int,
        field_columns: Sequence[int],
        deck_id: int,
        delimiter: str | None,
        is_html: bool | None,
        tags: Sequence[str],
        dupe_resolution: str,
    ) -> dict[str, Any]:
        try:
            path = self._import_path(filename, require_exists=True)
            if path.suffix != ".csv":
                raise ValueError("csv import requires a .csv file")
            # Belt-and-braces: re-validate the staged file (and recompute its content
            # digest) immediately before the backend import. The preview token binds
            # the digest through the guarded impact; this narrows the swap window.
            self.preview_import_csv(filename, delimiter)
            model = self.collection.models.get(cast("NotetypeId", notetype_id))
            if model is None:
                raise LookupError(f"note type {notetype_id} not found")
            field_count = len(model["flds"])
            if len(field_columns) != field_count:
                raise ValueError(
                    f"field_columns must map every field of note type {notetype_id} "
                    f"({field_count} fields, got {len(field_columns)})"
                )
            self.get_deck(deck_id)
            try:
                dupe_value = CSV_DUPE_RESOLUTIONS[dupe_resolution]
            except KeyError as exc:
                raise ValueError(
                    "dupe_resolution must be preserve, update, or duplicate"
                ) from exc
            delimiter_value = self._csv_delimiter(delimiter)
            metadata = self.collection.get_csv_metadata(str(path), delimiter_value)
            if delimiter_value is not None:
                metadata.delimiter = delimiter_value
            self._validate_field_columns(field_columns, metadata, filename)
            metadata.global_notetype.id = notetype_id
            del metadata.global_notetype.field_columns[:]
            metadata.global_notetype.field_columns.extend(field_columns)
            metadata.deck_id = deck_id
            if is_html is not None:
                metadata.is_html = is_html
            del metadata.global_tags[:]
            metadata.global_tags.extend(tags)
            metadata.dupe_resolution = dupe_value
            response = self.collection.import_csv(
                ImportCsvRequest(path=str(path), metadata=metadata)
            )
        except IMPORT_FILE_FAILURES as exc:
            raise ImportFileError(f"staged import file could not be read: {exc}") from exc
        log = response.log
        return {
            "filename": filename,
            "notetype_id": notetype_id,
            "deck_id": deck_id,
            "notes_added": len(log.new),
            "notes_updated": len(log.updated),
            "notes_duplicate": len(log.duplicate),
            "notes_conflicting": len(log.conflicting),
            "notes_matched_first_field": len(log.first_field_match),
            "notes_found": int(log.found_notes),
        }

    def undo_status(self) -> dict[str, Any]:
        status = self.collection.undo_status()
        return {
            "undo": str(status.undo),
            "redo": str(status.redo),
            "last_step": int(status.last_step),
            "durable": False,
            "note": UNDO_HISTORY_NOTE,
        }

    def undo(self, expect_operation: str) -> dict[str, Any]:
        return self._history_step(expect_operation, redo=False)

    def redo(self, expect_operation: str) -> dict[str, Any]:
        return self._history_step(expect_operation, redo=True)

    def _history_step(self, expect_operation: str, *, redo: bool) -> dict[str, Any]:
        status = self.collection.undo_status()
        label = "redo" if redo else "undo"
        head = status.redo if redo else status.undo
        if str(head) != expect_operation:
            raise ValueError(
                f"{label} stack head is {head!r}, not {expect_operation!r}; "
                "call anki_undo_status first"
            )
        result = self.collection.redo() if redo else self.collection.undo()
        return {
            "operation_undone": expect_operation,
            "operation": str(result.operation),
            "new_status": self.undo_status(),
        }

    def _truncate_rendered(self, value: str) -> tuple[str, bool]:
        encoded = value.encode("utf-8")
        if len(encoded) <= self.max_rendered_field_bytes:
            return value, False
        truncated = encoded[: self.max_rendered_field_bytes].decode("utf-8", errors="ignore")
        return truncated, True


class CollectionExecutor:
    """A single worker thread that exclusively owns one Anki collection."""

    def __init__(
        self,
        path: str | Path,
        max_page_size: int,
        max_search_scan: int = 10_000,
        max_rendered_field_bytes: int = 262_144,
        max_card_fields: int = 100,
        max_batch_size: int = 50,
        max_media_bytes: int = 1_048_576,
        sync_timeout_seconds: float = 300,
        sync_on_read: bool = False,
        sync_on_write: bool = False,
        max_response_bytes: int = 1_048_576,
        max_import_bytes: int = 268_435_456,
    ) -> None:
        self._path = path
        self._max_page_size = max_page_size
        self._max_search_scan = max_search_scan
        self._max_rendered_field_bytes = max_rendered_field_bytes
        self._max_card_fields = max_card_fields
        self._max_batch_size = max_batch_size
        self._max_media_bytes = max_media_bytes
        self._sync_timeout_seconds = sync_timeout_seconds
        self._sync_on_read = sync_on_read
        self._sync_on_write = sync_on_write
        self._max_response_bytes = max_response_bytes
        self._max_import_bytes = max_import_bytes
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anki-collection")
        self._adapter: CollectionAdapter | None = None
        self._worker_id: int | None = None

    @property
    def max_workers(self) -> int:
        """Expose the executor's actual configured concurrency for diagnostics and tests."""
        return int(self._pool._max_workers)  # pyright: ignore[reportPrivateUsage]

    def start(self) -> None:
        def open_collection() -> None:
            self._worker_id = threading.get_ident()
            self._adapter = CollectionAdapter(
                self._path,
                self._max_page_size,
                self._max_search_scan,
                self._max_rendered_field_bytes,
                self._max_card_fields,
                self._max_batch_size,
                self._max_media_bytes,
                self._sync_timeout_seconds,
                self._sync_on_read,
                self._sync_on_write,
                self._max_response_bytes,
                self._max_import_bytes,
            )

        self._pool.submit(open_collection).result()

    def submit(self, function: Callable[[CollectionAdapter], T]) -> Future[T]:
        def invoke() -> T:
            if self._adapter is None:
                raise RuntimeError("collection executor is not started")
            if threading.get_ident() != self._worker_id:
                raise RuntimeError("collection accessed outside owning thread")
            return function(self._adapter)

        return self._pool.submit(invoke)

    async def run(self, function: Callable[[CollectionAdapter], T]) -> T:
        return await asyncio.wrap_future(self.submit(function))

    async def close(self) -> None:
        if self._adapter is not None:
            await self.run(lambda adapter: adapter.close())
            self._adapter = None
        self._pool.shutdown(wait=True, cancel_futures=True)


class AnkiCollectionService:
    """Async facade over the dedicated collection executor."""

    def __init__(
        self,
        path: str | Path,
        max_page_size: int,
        max_search_scan: int = 10_000,
        max_rendered_field_bytes: int = 262_144,
        max_card_fields: int = 100,
        max_batch_size: int = 50,
        max_media_bytes: int = 1_048_576,
        sync_timeout_seconds: float = 300,
        sync_on_read: bool = False,
        sync_on_write: bool = False,
        max_response_bytes: int = 1_048_576,
        max_import_bytes: int = 268_435_456,
    ) -> None:
        self.executor = CollectionExecutor(
            path,
            max_page_size,
            max_search_scan,
            max_rendered_field_bytes,
            max_card_fields,
            max_batch_size,
            max_media_bytes,
            sync_timeout_seconds,
            sync_on_read,
            sync_on_write,
            max_response_bytes,
            max_import_bytes,
        )

    async def __aenter__(self) -> AnkiCollectionService:
        self.executor.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.executor.close()

    async def list_decks(self, offset: int, limit: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.list_decks(offset, limit))

    async def get_deck(self, deck_id: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.get_deck(deck_id))

    async def create_deck(self, name: str) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.create_deck(name))

    async def update_deck(self, deck_id: int, name: str) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.update_deck(deck_id, name))

    async def update_deck_config(
        self,
        deck_id: int,
        new_cards_per_day: int | None = None,
        reviews_per_day: int | None = None,
        max_answer_seconds: int | None = None,
        desired_retention: float | None = None,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.update_deck_config(
                deck_id,
                new_cards_per_day,
                reviews_per_day,
                max_answer_seconds,
                desired_retention,
            )
        )

    async def get_deck_options(
        self, deck_id: int, include_sections: Sequence[str] = ()
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.get_deck_options(deck_id, include_sections)
        )

    async def list_deck_presets(self, offset: int, limit: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.list_deck_presets(offset, limit))

    async def get_deck_preset(
        self, config_id: int, include_sections: Sequence[str] = ()
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.get_deck_preset(config_id, include_sections)
        )

    async def update_deck_preset(
        self, config_id: int, name: str | None, options: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.update_deck_preset(config_id, name, options)
        )

    async def create_deck_preset(
        self, name: str, clone_from_config_id: int | None
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.create_deck_preset(name, clone_from_config_id)
        )

    async def assign_deck_preset(self, deck_id: int, config_id: int) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.assign_deck_preset(deck_id, config_id)
        )

    async def update_deck_limits(
        self,
        deck_id: int,
        scope: str,
        values: dict[str, Any],
        clear_fields: Sequence[str],
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.update_deck_limits(
                deck_id,
                scope,
                values,
                clear_fields,
            )
        )

    async def update_deck_scheduler_settings(
        self,
        apply_all_parent_limits: bool | None,
        new_cards_ignore_review_limit: bool | None,
        fsrs_enabled: bool | None,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.update_deck_scheduler_settings(
                apply_all_parent_limits,
                new_cards_ignore_review_limit,
                fsrs_enabled,
            )
        )

    async def optimize_fsrs(
        self, config_id: int, search: str | None, health_check: bool
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.optimize_fsrs(config_id, search, health_check)
        )

    async def preview_fsrs_optimization(
        self, config_id: int, search: str | None, health_check: bool
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.preview_fsrs_optimization(config_id, search, health_check)
        )

    async def apply_fsrs_optimization(
        self, config_id: int, search: str | None, health_check: bool
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.apply_fsrs_optimization(config_id, search, health_check)
        )

    async def simulate_fsrs(
        self,
        config_id: int,
        mode: str,
        deck_size: int,
        days_to_simulate: int,
        desired_retention: float | None,
        search: str | None,
        include_daily: bool,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.simulate_fsrs(
                config_id,
                mode,
                deck_size,
                days_to_simulate,
                desired_retention,
                search,
                include_daily,
            )
        )

    async def preview_fsrs_reschedule(
        self,
        config_id: int,
        desired_retention: float | None,
        parameters: Sequence[float] | None,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.preview_fsrs_reschedule(
                config_id, desired_retention, parameters
            )
        )

    async def reschedule_fsrs(
        self,
        config_id: int,
        desired_retention: float | None,
        parameters: Sequence[float] | None,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.reschedule_fsrs(config_id, desired_retention, parameters)
        )

    async def delete_deck(self, deck_id: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.delete_deck(deck_id))

    async def search_notes(self, query: str, offset: int, limit: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.search_notes(query, offset, limit))

    async def get_note(self, note_id: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.get_note(note_id))

    async def create_note(
        self,
        deck_id: int,
        note_type_id: int,
        fields: dict[str, str],
        tags: list[str],
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.create_note(deck_id, note_type_id, fields, tags)
        )

    async def create_notes_batch(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.create_notes_batch(requests))

    async def update_note_fields(self, note_id: int, fields: dict[str, str]) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.update_note_fields(note_id, fields))

    async def add_note_tags(self, note_ids: list[int], tags: list[str]) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.add_note_tags(note_ids, tags))

    async def remove_note_tags(self, note_ids: list[int], tags: list[str]) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.remove_note_tags(note_ids, tags))

    async def delete_notes(self, note_ids: list[int]) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.delete_notes(note_ids))

    async def list_tags(self, offset: int, limit: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.list_tags(offset, limit))

    async def rename_tag(self, old_name: str, new_name: str) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.rename_tag(old_name, new_name))

    async def delete_tag(self, name: str) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.delete_tag(name))

    async def list_note_types(self, offset: int, limit: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.list_note_types(offset, limit))

    async def get_note_type(self, note_type_id: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.get_note_type(note_type_id))

    async def create_note_type(
        self,
        name: str,
        fields: list[str],
        templates: list[dict[str, str]],
        css: str,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.create_note_type(name, fields, templates, css)
        )

    async def update_note_type(
        self,
        note_type_id: int,
        name: str,
        fields: list[str],
        templates: list[dict[str, str]],
        css: str,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.update_note_type(note_type_id, name, fields, templates, css)
        )

    async def update_note_type_fields(
        self, note_type_id: int, mappings: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.update_note_type_fields(note_type_id, mappings)
        )

    async def update_templates(
        self, note_type_id: int, mappings: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.update_templates(note_type_id, mappings)
        )

    async def delete_note_type(self, note_type_id: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.delete_note_type(note_type_id))

    async def list_media(self, offset: int, limit: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.list_media(offset, limit))

    async def get_media(self, filename: str) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.get_media(filename))

    async def store_media(self, filename: str, content_base64: str) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.store_media(filename, content_base64)
        )

    async def rename_media(self, old_filename: str, new_filename: str) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.rename_media(old_filename, new_filename)
        )

    async def delete_media(self, filename: str) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.delete_media(filename))

    async def check_media(self, offset: int, limit: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.check_media(offset, limit))

    async def export_apkg(
        self,
        deck_id: int | None,
        include_media: bool,
        include_scheduling: bool,
        include_deck_configs: bool,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.export_apkg(
                deck_id, include_media, include_scheduling, include_deck_configs
            )
        )

    async def export_notes_csv(
        self,
        deck_id: int | None,
        with_html: bool,
        with_tags: bool,
        with_deck: bool,
        with_notetype: bool,
        with_guid: bool,
        inline: bool,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.export_notes_csv(
                deck_id, with_html, with_tags, with_deck, with_notetype, with_guid, inline
            )
        )

    async def list_import_files(self, offset: int, limit: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.list_import_files(offset, limit))

    async def preview_import_apkg(self, filename: str) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.preview_import_apkg_file(filename)
        )

    async def import_apkg(
        self,
        filename: str,
        merge_notetypes: bool,
        update_notes: ImportAnkiPackageUpdateCondition.ValueType,
        update_notetypes: ImportAnkiPackageUpdateCondition.ValueType,
        with_scheduling: bool,
        with_deck_configs: bool,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.import_apkg(
                filename,
                merge_notetypes,
                update_notes,
                update_notetypes,
                with_scheduling,
                with_deck_configs,
            )
        )

    async def preview_import_csv(self, filename: str, delimiter: str | None) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.preview_import_csv(filename, delimiter)
        )

    async def import_csv(
        self,
        filename: str,
        notetype_id: int,
        field_columns: Sequence[int],
        deck_id: int,
        delimiter: str | None,
        is_html: bool | None,
        tags: Sequence[str],
        dupe_resolution: str,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.import_csv(
                filename,
                notetype_id,
                field_columns,
                deck_id,
                delimiter,
                is_html,
                tags,
                dupe_resolution,
            )
        )

    async def undo_status(self) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.undo_status())

    async def undo(self, expect_operation: str) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.undo(expect_operation))

    async def redo(self, expect_operation: str) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.redo(expect_operation))

    async def search_cards(self, query: str, offset: int, limit: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.search_cards(query, offset, limit))

    async def get_card(
        self, card_id: int, include_sections: Sequence[str] = ()
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.get_card(card_id, include_sections)
        )

    async def list_reviews(
        self,
        card_id: int | None,
        deck_id: int | None,
        query: str | None,
        include_children: bool,
        offset: int,
        limit: int,
        order: str,
        include_fields: Sequence[str] = (),
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.list_reviews(
                card_id,
                deck_id,
                query,
                include_children,
                offset,
                limit,
                order,
                include_fields,
            )
        )

    async def review_stats(
        self,
        card_id: int | None,
        deck_id: int | None,
        query: str | None,
        include_children: bool,
        days: int,
        include_sections: Sequence[str] = (),
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.review_stats(
                card_id, deck_id, query, include_children, days, include_sections
            )
        )

    async def create_card(self, deck_id: int, front: str, back: str) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.create_card(deck_id, front, back))

    async def update_card(
        self, card_id: int, front: str | None, back: str | None, deck_id: int | None
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.update_card(card_id, front, back, deck_id)
        )

    async def delete_card(self, card_id: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.delete_card(card_id))

    async def change_card_deck(self, card_ids: list[int], deck_id: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.change_card_deck(card_ids, deck_id))

    async def suspend_cards(self, card_ids: list[int]) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.suspend_cards(card_ids))

    async def unsuspend_cards(self, card_ids: list[int]) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.unsuspend_cards(card_ids))

    async def set_card_flag(self, card_ids: list[int], flag: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.set_card_flag(card_ids, flag))

    async def reposition_cards(
        self,
        card_ids: list[int],
        starting_from: int,
        step_size: int,
        randomize: bool,
        shift_existing: bool,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.reposition_cards(
                card_ids, starting_from, step_size, randomize, shift_existing
            )
        )

    async def coordinated_read(
        self,
        read: Callable[[CollectionAdapter], T],
        sync_before: bool = False,
        sync_media: bool = False,
    ) -> T:
        return await self.executor.run(
            lambda adapter: adapter.coordinated_read(read, sync_before, sync_media)
        )

    async def coordinated_mutation(
        self,
        operation: str,
        idempotency_key: str,
        request: dict[str, Any],
        mutate: Callable[[CollectionAdapter], dict[str, Any]],
        sync_media: bool = False,
        sync_after: bool = True,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.coordinated_mutation(
                operation, idempotency_key, request, mutate, sync_media, sync_after
            )
        )

    async def status(self) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.status())

    async def get_operation(self, idempotency_key: str) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.get_operation(idempotency_key))

    async def list_operations(self, offset: int, limit: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.list_operations(offset, limit))

    async def metrics(self) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.metrics())

    async def create_backup(self) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.create_backup())

    async def list_backups(self, offset: int, limit: int) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.list_backups(offset, limit))

    async def preview_backup_restore(self, filename: str) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.preview_backup_restore(filename)
        )

    async def restore_backup(self, filename: str, mode: str) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.restore_backup(filename, mode)
        )

    async def bootstrap(
        self,
        mode: str,
        username: str,
        password: str | None,
        endpoint: str | None,
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.bootstrap(mode, username, password, endpoint)
        )

    async def sync_login(
        self, username: str, password: str, endpoint: str | None
    ) -> dict[str, Any]:
        return await self.executor.run(
            lambda adapter: adapter.sync_login(username, password, endpoint)
        )

    async def sync(self, sync_media: bool) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.sync(sync_media))

    async def full_sync(self, upload: bool, force: bool = False) -> dict[str, Any]:
        return await self.executor.run(lambda adapter: adapter.full_sync(upload, force))

    async def check_ready(self) -> bool:
        return await self.executor.run(lambda adapter: adapter.check_ready())
