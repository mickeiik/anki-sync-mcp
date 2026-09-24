from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PersistentState:
    """Small durable state store owned by the collection worker thread."""

    def __init__(self, collection_path: str | Path) -> None:
        self.directory = Path(collection_path).parent / "state"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sync_auth_path = self.directory / "sync-auth"
        self.status_path = self.directory / "operation-status.json"
        self.connection = sqlite3.connect(self.directory / "idempotency.sqlite")
        self.connection.execute(
            """
            create table if not exists mutation_receipts (
                idempotency_key text primary key,
                operation text not null,
                request_hash text not null,
                receipt_json text not null
            )
            """
        )
        columns = {
            str(row[1])
            for row in self.connection.execute("pragma table_info(mutation_receipts)").fetchall()
        }
        if "created_at" not in columns:
            self.connection.execute(
                "alter table mutation_receipts add column created_at text not null default ''"
            )
        if "updated_at" not in columns:
            self.connection.execute(
                "alter table mutation_receipts add column updated_at text not null default ''"
            )
        migrated_at = datetime.now(UTC).isoformat()
        self.connection.execute(
            "update mutation_receipts set created_at = ? where created_at = ''", (migrated_at,)
        )
        self.connection.execute(
            "update mutation_receipts set updated_at = ? where updated_at = ''", (migrated_at,)
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def request_hash(operation: str, request: dict[str, Any]) -> str:
        canonical = json.dumps(
            {"operation": operation, "request": request},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def get_receipt(self, key: str) -> tuple[str, str, dict[str, Any]] | None:
        row = self.connection.execute(
            "select operation, request_hash, receipt_json from mutation_receipts "
            "where idempotency_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        operation, request_hash, receipt_json = row
        receipt = json.loads(receipt_json)
        if not isinstance(receipt, dict):  # pragma: no cover - only this class writes records
            raise RuntimeError("invalid mutation receipt")
        return str(operation), str(request_hash), receipt

    def put_receipt(
        self, key: str, operation: str, request_hash: str, receipt: dict[str, Any]
    ) -> None:
        encoded = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        now = datetime.now(UTC).isoformat()
        self.connection.execute(
            """
            insert into mutation_receipts(
                idempotency_key, operation, request_hash, receipt_json, created_at, updated_at
            )
            values (?, ?, ?, ?, ?, ?)
            on conflict(idempotency_key) do update set
                receipt_json = excluded.receipt_json,
                updated_at = excluded.updated_at
            """,
            (key, operation, request_hash, encoded, now, now),
        )
        self.connection.commit()

    def get_operation(self, key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            select operation, receipt_json, created_at, updated_at
            from mutation_receipts where idempotency_key = ?
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        operation, receipt_json, created_at, updated_at = row
        return {
            "idempotency_key": key,
            "operation": str(operation),
            "created_at": str(created_at),
            "updated_at": str(updated_at),
            "receipt": json.loads(receipt_json),
        }

    def list_operations(self, offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        total = int(self.connection.execute("select count(*) from mutation_receipts").fetchone()[0])
        rows = self.connection.execute(
            """
            select idempotency_key, operation, receipt_json, created_at, updated_at
            from mutation_receipts order by updated_at desc, rowid desc limit ? offset ?
            """,
            (limit, offset),
        ).fetchall()
        return (
            [
                {
                    "idempotency_key": str(key),
                    "operation": str(operation),
                    "created_at": str(created_at),
                    "updated_at": str(updated_at),
                    "receipt": json.loads(receipt_json),
                }
                for key, operation, receipt_json, created_at, updated_at in rows
            ],
            total,
        )

    def metrics(self) -> dict[str, Any]:
        rows = self.connection.execute(
            "select operation, receipt_json from mutation_receipts"
        ).fetchall()
        by_operation: dict[str, int] = {}
        committed = 0
        outcome_unknown = 0
        pending = 0
        for operation, receipt_json in rows:
            name = str(operation)
            by_operation[name] = by_operation.get(name, 0) + 1
            receipt = json.loads(receipt_json)
            committed += receipt.get("state") == "committed"
            outcome_unknown += receipt.get("state") == "outcome_unknown"
            pending += (
                receipt.get("remote_synced") is False or receipt.get("media_synced") is False
            ) and receipt.get("state") not in {
                "discarded_by_full_download",
                "discarded_by_restore",
            }
        return {
            "mutations": {
                "total": len(rows),
                "committed": committed,
                "outcome_unknown": outcome_unknown,
                "pending": pending,
                "by_operation": dict(sorted(by_operation.items())),
            }
        }

    def delete_receipt(self, key: str) -> None:
        self.connection.execute("delete from mutation_receipts where idempotency_key = ?", (key,))
        self.connection.commit()

    def pending_receipt_count(self) -> int:
        rows = self.connection.execute("select receipt_json from mutation_receipts").fetchall()
        return sum(
            (receipt.get("remote_synced") is False or receipt.get("media_synced") is False)
            and receipt.get("state")
            not in {"discarded_by_full_download", "discarded_by_restore"}
            for row in rows
            if isinstance((receipt := json.loads(row[0])), dict)
        )

    def mark_all_remote_synced(self) -> None:
        """Reconcile locally committed receipts after an operator-selected full upload."""
        rows = self.connection.execute(
            "select idempotency_key, receipt_json from mutation_receipts"
        ).fetchall()
        for key, receipt_json in rows:
            receipt = json.loads(receipt_json)
            if receipt.get("local_committed") and (
                not receipt.get("remote_synced") or receipt.get("media_synced") is False
            ):
                receipt["remote_synced"] = True
                receipt["retryable"] = receipt.get("media_synced") is False
                receipt["state"] = "committed"
                receipt["sync_required"] = None
                receipt["sync_error"] = None
                encoded = json.dumps(
                    receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                self.connection.execute(
                    """
                    update mutation_receipts set receipt_json = ?, updated_at = ?
                    where idempotency_key = ?
                    """,
                    (encoded, datetime.now(UTC).isoformat(), key),
                )
        self.connection.commit()

    def mark_pending_discarded_by_full_download(self) -> None:
        """Mark local-only mutation results as discarded by a full download."""
        self._mark_pending_discarded("discarded_by_full_download")

    def mark_pending_discarded_by_restore(self) -> None:
        """Mark local-only mutation results as discarded by a collection restore."""
        self._mark_pending_discarded("discarded_by_restore")

    def _mark_pending_discarded(self, discarded_state: str) -> None:
        rows = self.connection.execute(
            "select idempotency_key, receipt_json from mutation_receipts"
        ).fetchall()
        for key, receipt_json in rows:
            receipt = json.loads(receipt_json)
            changed = False
            if receipt.get("media_synced") is False and receipt.get("local_committed"):
                receipt.update(
                    {
                        "state": "committed",
                        "remote_synced": True,
                        "retryable": True,
                        "sync_required": None,
                        "sync_error": None,
                    }
                )
                changed = True
            elif (
                receipt.get("local_committed") or receipt.get("state") == "outcome_unknown"
            ) and not receipt.get("remote_synced"):
                receipt.update(
                    {
                        "state": discarded_state,
                        "local_committed": False,
                        "remote_synced": False,
                        "retryable": False,
                        "result": None,
                        "sync_required": None,
                        "sync_error": None,
                    }
                )
                changed = True
            if changed:
                encoded = json.dumps(
                    receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                self.connection.execute(
                    """
                    update mutation_receipts set receipt_json = ?, updated_at = ?
                    where idempotency_key = ?
                    """,
                    (encoded, datetime.now(UTC).isoformat(), key),
                )
        self.connection.commit()

    def load_sync_auth(self) -> dict[str, Any] | None:
        return self._load_json(self.sync_auth_path)

    def save_sync_auth(self, auth: dict[str, Any]) -> None:
        self._atomic_json(self.sync_auth_path, auth, secret=True)

    def clear_sync_auth(self) -> None:
        self.sync_auth_path.unlink(missing_ok=True)

    def load_status(self) -> dict[str, Any]:
        return self._load_json(self.status_path) or {}

    def save_status(self, status: dict[str, Any]) -> None:
        self._atomic_json(self.status_path, status, secret=False)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unable to load persistent state file {path.name}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid persistent state file {path.name}")
        return value

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any], *, secret: bool) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        descriptor = os.open(temporary, flags, 0o600 if secret else 0o644)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            if secret:
                path.chmod(0o600)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
