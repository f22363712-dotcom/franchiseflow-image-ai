"""Persistence contract for FranchiseFlow's job and batch state.

The web application currently writes one JSON record per job/batch and uses a
separate JSON index for idempotency.  This module gives the application a
stable seam for a later SQLite (or Postgres) migration without changing the
current API in the middle of the migration.  ``JsonStorageRepository`` is a
thin adapter around the existing stores; it deliberately does not claim to be
a distributed queue.

The adapter is intentionally unused by ``app.py`` in this slice.  Keeping the
contract additive lets the existing demo remain unchanged while future work
can migrate one call site at a time and run both backends against the same
contract tests.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

from batch_store import (
    BATCH_ID_RE,
    create_batch as create_batch_record,
    load_batch as load_batch_record,
    save_batch as save_batch_record,
)
from idempotency_store import IdempotencyStore
from idempotency_store import (
    IdempotencyConflict,
    IdempotencyError,
    IdempotencyReservationError,
    normalize_key,
)
from job_store import (
    create_job as create_job_record,
    load_job as load_job_record,
    save_job as save_job_record,
    ALLOWED_TRANSITIONS,
    STATES,
)


JsonRecord = dict[str, Any]


def storage_backend_from_environment(
    environment: Mapping[str, str] | None = None,
) -> str:
    """Validate the selected control-plane backend.

    The application keeps JSON as its default for backwards compatibility.
    SQLite must be selected explicitly so a deployment cannot appear migrated
    merely because a database file happens to exist beside the outputs.
    """

    environment = os.environ if environment is None else environment
    backend = str(environment.get("FRANCHISEFLOW_STORAGE_BACKEND", "json")).strip().lower()
    if backend not in {"json", "sqlite"}:
        raise ValueError(
            "FRANCHISEFLOW_STORAGE_BACKEND must be one of: json, sqlite"
        )
    return backend


class StorageRepository(Protocol):
    """Minimal persistence interface used by the production migration.

    A repository stores logical state only.  Uploaded media and rendered
    artifacts remain object/file storage concerns and are referenced by paths
    or keys in the records.  Implementations must preserve the state-machine
    checks already enforced by ``job_store`` and the idempotency reservation
    states (``pending`` and ``committed``).
    """

    @property
    def capabilities(self) -> Mapping[str, bool]:
        """Report guarantees callers may rely on during rollout."""

    def create_job(self, job_id: str, payload: Mapping[str, Any]) -> JsonRecord:
        ...

    def load_job(self, job_id: str) -> JsonRecord | None:
        ...

    def save_job(self, job_id: str, job: Mapping[str, Any]) -> None:
        ...

    def list_jobs(
        self,
        *,
        limit: int | None = None,
        status: str | None = None,
    ) -> list[JsonRecord]:
        ...

    def create_batch(self, batch_id: str, payload: Mapping[str, Any]) -> JsonRecord:
        ...

    def load_batch(self, batch_id: str) -> JsonRecord | None:
        ...

    def save_batch(self, batch_id: str, batch: Mapping[str, Any]) -> None:
        ...

    def list_batches(self, *, limit: int | None = None) -> list[JsonRecord]:
        ...

    def get_idempotency(self, scope: str, key: str) -> JsonRecord | None:
        ...

    def pending_idempotency(self) -> list[JsonRecord]:
        ...

    def reserve_idempotency(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        *,
        resource_id: str,
        status_code: int = 202,
    ) -> JsonRecord:
        ...

    def commit_idempotency(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        resource_id: str,
        *,
        status_code: int = 202,
    ) -> JsonRecord:
        ...

    def release_idempotency(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        *,
        resource_id: str | None = None,
    ) -> bool:
        ...


@dataclass(slots=True)
class JsonStorageRepository:
    """Adapter exposing the current atomic JSON stores through the contract.

    ``output_dir`` and ``batch_dir`` use the same paths as ``app.py``.  The
    adapter does not acquire a process lock around a multi-step workflow; the
    caller must retain the current ``IDEMPOTENCY_LOCK`` while performing
    check/reserve/create/commit.  This is the key limitation a SQLite backend
    must remove with transactional uniqueness and an atomic claim operation.
    """

    output_dir: Path
    batch_dir: Path | None = None
    idempotency_path: Path | None = None
    _idempotency: IdempotencyStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.batch_dir = Path(self.batch_dir or (self.output_dir / "batches"))
        self.idempotency_path = Path(
            self.idempotency_path or (self.output_dir / "idempotency.json")
        )
        self._idempotency = IdempotencyStore(self.idempotency_path)

    @property
    def capabilities(self) -> Mapping[str, bool]:
        """Explicitly document what the JSON adapter can and cannot promise."""

        return {
            "atomic_local_writes": True,
            "durable_job_records": True,
            "idempotency_reservations": True,
            "cross_process_claim": False,
            "cross_replica_consistency": False,
            "tenant_scoping": False,
        }

    def create_job(self, job_id: str, payload: Mapping[str, Any]) -> JsonRecord:
        return create_job_record(self.output_dir, job_id, dict(payload))

    def load_job(self, job_id: str) -> JsonRecord | None:
        return load_job_record(self.output_dir, job_id)

    def save_job(self, job_id: str, job: Mapping[str, Any]) -> None:
        save_job_record(self.output_dir, job_id, dict(job))

    def list_jobs(
        self,
        *,
        limit: int | None = None,
        status: str | None = None,
    ) -> list[JsonRecord]:
        jobs: list[JsonRecord] = []
        for state_path in sorted(self.output_dir.glob("*/job.json"), key=lambda p: p.name):
            job = load_job_record(self.output_dir, state_path.parent.name)
            if job is None or (status is not None and job.get("status") != status):
                continue
            jobs.append(job)
        jobs.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return jobs if limit is None else jobs[: max(0, int(limit))]

    def create_batch(self, batch_id: str, payload: Mapping[str, Any]) -> JsonRecord:
        return create_batch_record(self.batch_dir, batch_id, dict(payload))

    def load_batch(self, batch_id: str) -> JsonRecord | None:
        return load_batch_record(self.batch_dir, batch_id)

    def save_batch(self, batch_id: str, batch: Mapping[str, Any]) -> None:
        save_batch_record(self.batch_dir, batch_id, dict(batch))

    def list_batches(self, *, limit: int | None = None) -> list[JsonRecord]:
        batches: list[JsonRecord] = []
        for state_path in sorted(self.batch_dir.glob("*.json"), key=lambda p: p.name):
            batch = load_batch_record(self.batch_dir, state_path.stem)
            if batch is not None:
                batches.append(batch)
        batches.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return batches if limit is None else batches[: max(0, int(limit))]

    def get_idempotency(self, scope: str, key: str) -> JsonRecord | None:
        return self._idempotency.get(scope, key)

    def pending_idempotency(self) -> list[JsonRecord]:
        return self._idempotency.pending_entries()

    def reserve_idempotency(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        *,
        resource_id: str,
        status_code: int = 202,
    ) -> JsonRecord:
        return self._idempotency.reserve(
            scope,
            key,
            fingerprint,
            resource_id=resource_id,
            status_code=status_code,
        )

    def commit_idempotency(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        resource_id: str,
        *,
        status_code: int = 202,
    ) -> JsonRecord:
        return self._idempotency.commit(
            scope,
            key,
            fingerprint,
            resource_id,
            status_code=status_code,
        )

    def release_idempotency(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        *,
        resource_id: str | None = None,
    ) -> bool:
        return self._idempotency.release(
            scope,
            key,
            fingerprint,
            resource_id=resource_id,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_payload(raw: str, kind: str) -> JsonRecord:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{kind} state is unreadable") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{kind} state is invalid")
    return dict(value)


@dataclass(slots=True)
class SQLiteStorageRepository:
    """Transactional SQLite implementation of :class:`StorageRepository`.

    The repository stores the control plane only.  Media and rendered files
    remain outside SQLite and are referenced by keys in ``payload_json``.  A
    repository instance is scoped to one tenant; ``local`` keeps the existing
    single-machine behaviour.  The application does not select this backend
    yet, so this class is an additive migration seam rather than a queue.
    """

    db_path: Path
    tenant_id: str = "local"
    busy_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.tenant_id = str(self.tenant_id or "local")
        self.busy_timeout_ms = max(0, int(self.busy_timeout_ms))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            # WAL is persistent per database.  Running this per connection is
            # intentional so a newly opened worker observes the same mode.
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'local',
                    batch_id TEXT,
                    status TEXT NOT NULL,
                    phase TEXT,
                    payload_json TEXT NOT NULL,
                    claim_token TEXT,
                    lease_until TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_queue_idx
                    ON jobs (tenant_id, status, created_at, job_id);
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'local',
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS batches_updated_idx
                    ON batches (tenant_id, updated_at, batch_id);
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    tenant_id TEXT NOT NULL DEFAULT 'local',
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, scope, key)
                );
                CREATE INDEX IF NOT EXISTS idempotency_pending_idx
                    ON idempotency_keys (tenant_id, state, created_at);
                """
            )

    @property
    def capabilities(self) -> Mapping[str, bool]:
        return {
            "atomic_local_writes": True,
            "durable_job_records": True,
            "idempotency_reservations": True,
            "cross_process_claim": False,
            "cross_replica_consistency": False,
            "tenant_scoping": True,
            "compare_and_swap": True,
            "sqlite_transactions": True,
        }

    def _check_tenant(self, record: Mapping[str, Any]) -> str:
        tenant = str(record.get("tenant_id") or self.tenant_id)
        if tenant != self.tenant_id:
            raise ValueError("tenant_id does not match repository scope")
        return tenant

    @staticmethod
    def _validate_job_id(job_id: str) -> str:
        value = str(job_id)
        if not value or value in {".", ".."} or Path(value).name != value or any(ch in value for ch in "\\/"):
            raise ValueError("invalid job id")
        return value

    @staticmethod
    def _validate_batch_id(batch_id: str) -> str:
        value = str(batch_id)
        if not BATCH_ID_RE.fullmatch(value or ""):
            raise ValueError("invalid batch id")
        return value

    @staticmethod
    def _version(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            version = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("version must be an integer") from exc
        if version < 0:
            raise ValueError("version must be non-negative")
        return version

    def _job_record(self, row: sqlite3.Row) -> JsonRecord:
        record = _decode_payload(row["payload_json"], "job")
        record.update(
            {
                "job_id": row["job_id"],
                "tenant_id": row["tenant_id"],
                "status": row["status"],
                "version": int(row["version"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
        if row["batch_id"] is not None:
            record["batch_id"] = row["batch_id"]
        return record

    def _batch_record(self, row: sqlite3.Row) -> JsonRecord:
        record = _decode_payload(row["payload_json"], "batch")
        record.update(
            {
                "batch_id": row["batch_id"],
                "tenant_id": row["tenant_id"],
                "status": row["status"],
                "version": int(row["version"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
        return record

    def _idempotency_record(self, row: sqlite3.Row) -> JsonRecord:
        record = _decode_payload(row["payload_json"], "idempotency")
        record.update(
            {
                "tenant_id": row["tenant_id"],
                "scope": row["scope"],
                "key": row["key"],
                "fingerprint": row["fingerprint"],
                "resource_id": row["resource_id"],
                "state": row["state"],
                "status_code": int(row["status_code"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
        return record

    @staticmethod
    def _job_columns(record: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            record.get("batch_id"),
            record.get("status", "queued"),
            record.get("phase"),
            json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")),
            record.get("claim_token"),
            record.get("lease_until"),
            int(record.get("retry_count", 0) or 0),
            int(record.get("version", 0)),
            record["created_at"],
            record["updated_at"],
        )

    @staticmethod
    def _batch_columns(record: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            record.get("status", "queued"),
            json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")),
            int(record.get("version", 0)),
            record["created_at"],
            record["updated_at"],
        )

    def _prepare_job(
        self,
        job_id: str,
        payload: Mapping[str, Any],
        *,
        current: JsonRecord | None = None,
        version: int | None = None,
        preserve_created_at: bool = False,
    ) -> JsonRecord:
        value = dict(payload)
        value["job_id"] = self._validate_job_id(job_id)
        value["tenant_id"] = self._check_tenant(value)
        status = value.get("status", current.get("status", "queued") if current else "queued")
        if status not in STATES:
            raise ValueError(f"unsupported job state: {status}")
        if current and status not in ALLOWED_TRANSITIONS.get(str(current.get("status")), {str(current.get("status"))}):
            raise ValueError(f"illegal job state transition: {current.get('status')} -> {status}")
        value["status"] = status
        if preserve_created_at and current:
            value["created_at"] = current.get("created_at")
        else:
            value.setdefault("created_at", _utc_now())
        value["updated_at"] = _utc_now()
        value["version"] = self._version(version if version is not None else value.get("version"), 0)
        return value

    def create_job(self, job_id: str, payload: Mapping[str, Any]) -> JsonRecord:
        job_id = self._validate_job_id(job_id)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? AND job_id = ?",
                (self.tenant_id, job_id),
            ).fetchone()
            if row is not None:
                current = self._job_record(row)
                record = self._prepare_job(
                    job_id,
                    payload,
                    current=current,
                    version=int(row["version"]) + 1,
                    preserve_created_at=True,
                )
                connection.execute(
                    """UPDATE jobs SET batch_id=?, status=?, phase=?, payload_json=?,
                       claim_token=?, lease_until=?, retry_count=?, version=?,
                       created_at=?, updated_at=? WHERE tenant_id=? AND job_id=?""",
                    self._job_columns(record) + (self.tenant_id, job_id),
                )
                return record
            record = self._prepare_job(job_id, payload)
            connection.execute(
                """INSERT INTO jobs
                   (job_id, tenant_id, batch_id, status, phase, payload_json,
                    claim_token, lease_until, retry_count, version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, self.tenant_id) + self._job_columns(record),
            )
            return record

    def load_job(self, job_id: str) -> JsonRecord | None:
        job_id = self._validate_job_id(job_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? AND job_id = ?",
                (self.tenant_id, job_id),
            ).fetchone()
        return None if row is None else self._job_record(row)

    def save_job(self, job_id: str, job: Mapping[str, Any]) -> None:
        job_id = self._validate_job_id(job_id)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? AND job_id = ?",
                (self.tenant_id, job_id),
            ).fetchone()
            current = None if row is None else self._job_record(row)
            record = self._prepare_job(
                job_id,
                job,
                current=current,
                version=(int(row["version"]) + 1 if row is not None else 0),
                preserve_created_at=bool(current),
            )
            if row is None:
                connection.execute(
                    """INSERT INTO jobs
                       (job_id, tenant_id, batch_id, status, phase, payload_json,
                        claim_token, lease_until, retry_count, version, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (job_id, self.tenant_id) + self._job_columns(record),
                )
            else:
                connection.execute(
                    """UPDATE jobs SET batch_id=?, status=?, phase=?, payload_json=?,
                       claim_token=?, lease_until=?, retry_count=?, version=?,
                       created_at=?, updated_at=? WHERE tenant_id=? AND job_id=?""",
                    self._job_columns(record) + (self.tenant_id, job_id),
                )

    def list_jobs(
        self,
        *,
        limit: int | None = None,
        status: str | None = None,
    ) -> list[JsonRecord]:
        query = "SELECT * FROM jobs WHERE tenant_id = ?"
        params: list[Any] = [self.tenant_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC, job_id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._job_record(row) for row in rows]

    def compare_and_swap_job(
        self,
        job_id: str,
        expected_version: int,
        job: Mapping[str, Any],
    ) -> JsonRecord | None:
        """Update one job only when its version still equals ``expected_version``.

        ``None`` indicates a stale read or a missing row.  This primitive is
        deliberately separate from worker claiming; no queue is wired to it.
        """
        job_id = self._validate_job_id(job_id)
        expected = self._version(expected_version)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? AND job_id = ?",
                (self.tenant_id, job_id),
            ).fetchone()
            if row is None or int(row["version"]) != expected:
                return None
            current = self._job_record(row)
            record = self._prepare_job(
                job_id,
                job,
                current=current,
                version=expected + 1,
                preserve_created_at=True,
            )
            cursor = connection.execute(
                """UPDATE jobs SET batch_id=?, status=?, phase=?, payload_json=?,
                   claim_token=?, lease_until=?, retry_count=?, version=?,
                   created_at=?, updated_at=?
                   WHERE tenant_id=? AND job_id=? AND version=?""",
                self._job_columns(record) + (self.tenant_id, job_id, expected),
            )
            return record if cursor.rowcount == 1 else None

    def create_batch(self, batch_id: str, payload: Mapping[str, Any]) -> JsonRecord:
        batch_id = self._validate_batch_id(batch_id)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM batches WHERE tenant_id = ? AND batch_id = ?",
                (self.tenant_id, batch_id),
            ).fetchone()
            value = dict(payload)
            value["batch_id"] = batch_id
            value["tenant_id"] = self._check_tenant(value)
            if row is None:
                value.setdefault("created_at", _utc_now())
                value["version"] = self._version(value.get("version"), 0)
            else:
                previous = self._batch_record(row)
                value.setdefault("created_at", previous.get("created_at"))
                value["version"] = int(row["version"]) + 1
            value.setdefault("status", "queued")
            value["updated_at"] = _utc_now()
            if row is None:
                connection.execute(
                    """INSERT INTO batches
                       (batch_id, tenant_id, status, payload_json, version, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (batch_id, self.tenant_id) + self._batch_columns(value),
                )
            else:
                connection.execute(
                    """UPDATE batches SET status=?, payload_json=?, version=?, created_at=?, updated_at=?
                       WHERE tenant_id=? AND batch_id=?""",
                    self._batch_columns(value) + (self.tenant_id, batch_id),
                )
            return value

    def load_batch(self, batch_id: str) -> JsonRecord | None:
        batch_id = self._validate_batch_id(batch_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM batches WHERE tenant_id = ? AND batch_id = ?",
                (self.tenant_id, batch_id),
            ).fetchone()
        return None if row is None else self._batch_record(row)

    def save_batch(self, batch_id: str, batch: Mapping[str, Any]) -> None:
        self.create_batch(batch_id, batch)

    def list_batches(self, *, limit: int | None = None) -> list[JsonRecord]:
        query = "SELECT * FROM batches WHERE tenant_id = ? ORDER BY updated_at DESC, batch_id DESC"
        params: list[Any] = [self.tenant_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._batch_record(row) for row in rows]

    def get_idempotency(self, scope: str, key: str) -> JsonRecord | None:
        normalized = normalize_key(key)
        if normalized is None:
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM idempotency_keys WHERE tenant_id=? AND scope=? AND key=?",
                (self.tenant_id, str(scope), normalized),
            ).fetchone()
        return None if row is None else self._idempotency_record(row)

    def pending_idempotency(self) -> list[JsonRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM idempotency_keys
                   WHERE tenant_id=? AND state='pending'
                   ORDER BY created_at ASC, scope ASC, key ASC""",
                (self.tenant_id,),
            ).fetchall()
        return [self._idempotency_record(row) for row in rows]

    def _get_idempotency_locked(
        self,
        connection: sqlite3.Connection,
        scope: str,
        key: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM idempotency_keys WHERE tenant_id=? AND scope=? AND key=?",
            (self.tenant_id, str(scope), key),
        ).fetchone()

    def reserve_idempotency(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        *,
        resource_id: str,
        status_code: int = 202,
    ) -> JsonRecord:
        normalized = normalize_key(key)
        if normalized is None:
            raise IdempotencyError("Idempotency-Key must not be empty")
        if not fingerprint:
            raise IdempotencyError("idempotency fingerprint is required")
        if not resource_id:
            raise IdempotencyError("idempotency resource ID is required")
        scope = str(scope)
        with self._transaction() as connection:
            existing = self._get_idempotency_locked(connection, scope, normalized)
            if existing is not None:
                if existing["fingerprint"] != fingerprint:
                    raise IdempotencyConflict("Idempotency-Key was already used with a different request")
                return self._idempotency_record(existing)
            now = _utc_now()
            record = {
                "tenant_id": self.tenant_id,
                "scope": scope,
                "key": normalized,
                "fingerprint": str(fingerprint),
                "resource_id": str(resource_id),
                "state": "pending",
                "status_code": int(status_code),
                "created_at": now,
                "updated_at": now,
            }
            try:
                connection.execute(
                    """INSERT INTO idempotency_keys
                       (tenant_id, scope, key, fingerprint, resource_id, state,
                        status_code, payload_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        self.tenant_id,
                        scope,
                        normalized,
                        record["fingerprint"],
                        record["resource_id"],
                        "pending",
                        int(status_code),
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                existing = self._get_idempotency_locked(connection, scope, normalized)
                if existing is None:
                    raise
                if existing["fingerprint"] != fingerprint:
                    raise IdempotencyConflict("Idempotency-Key was already used with a different request")
                return self._idempotency_record(existing)
            return record

    def commit_idempotency(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        resource_id: str,
        *,
        status_code: int = 202,
    ) -> JsonRecord:
        normalized = normalize_key(key)
        if normalized is None:
            raise IdempotencyError("Idempotency-Key must not be empty")
        if not fingerprint or not resource_id:
            raise IdempotencyError("idempotency fingerprint and resource ID are required")
        scope = str(scope)
        with self._transaction() as connection:
            previous = self._get_idempotency_locked(connection, scope, normalized)
            if previous is None:
                raise IdempotencyReservationError("idempotency reservation is missing")
            if previous["fingerprint"] != fingerprint:
                raise IdempotencyConflict("Idempotency-Key was already used with a different request")
            if str(previous["resource_id"]) != str(resource_id):
                raise IdempotencyReservationError("idempotency resource ID does not match reservation")
            if previous["state"] == "committed":
                return self._idempotency_record(previous)
            now = _utc_now()
            record = self._idempotency_record(previous)
            record.update({"state": "committed", "status_code": int(status_code), "updated_at": now})
            connection.execute(
                """UPDATE idempotency_keys SET state=?, status_code=?, payload_json=?, updated_at=?
                   WHERE tenant_id=? AND scope=? AND key=?""",
                (
                    "committed",
                    int(status_code),
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                    now,
                    self.tenant_id,
                    scope,
                    normalized,
                ),
            )
            return record

    def release_idempotency(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        *,
        resource_id: str | None = None,
    ) -> bool:
        normalized = normalize_key(key)
        if normalized is None:
            raise IdempotencyError("Idempotency-Key must not be empty")
        scope = str(scope)
        with self._transaction() as connection:
            previous = self._get_idempotency_locked(connection, scope, normalized)
            if previous is None:
                return False
            if previous["fingerprint"] != fingerprint:
                raise IdempotencyConflict("Idempotency-Key was already used with a different request")
            if resource_id is not None and str(previous["resource_id"]) != str(resource_id):
                raise IdempotencyReservationError("idempotency resource ID does not match reservation")
            if previous["state"] != "pending":
                return False
            connection.execute(
                "DELETE FROM idempotency_keys WHERE tenant_id=? AND scope=? AND key=?",
                (self.tenant_id, scope, normalized),
            )
            return True


def assert_storage_contract(repository: StorageRepository) -> Sequence[str]:
    """Return missing method names for lightweight adapter smoke checks.

    This helper avoids a runtime dependency on a Protocol implementation
    checker.  It is useful to fail fast when a future SQLite adapter is wired
    into a migration branch but has not implemented every operation yet.
    """

    required = (
        "create_job",
        "load_job",
        "save_job",
        "list_jobs",
        "create_batch",
        "load_batch",
        "save_batch",
        "list_batches",
        "get_idempotency",
        "pending_idempotency",
        "reserve_idempotency",
        "commit_idempotency",
        "release_idempotency",
    )
    return tuple(name for name in required if not callable(getattr(repository, name, None)))

