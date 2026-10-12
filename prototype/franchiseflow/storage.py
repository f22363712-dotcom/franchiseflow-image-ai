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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from batch_store import (
    create_batch as create_batch_record,
    load_batch as load_batch_record,
    save_batch as save_batch_record,
)
from idempotency_store import IdempotencyStore
from job_store import (
    create_job as create_job_record,
    load_job as load_job_record,
    save_job as save_job_record,
)


JsonRecord = dict[str, Any]


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

