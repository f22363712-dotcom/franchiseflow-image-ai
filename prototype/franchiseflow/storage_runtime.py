"""Runtime selection seam for FranchiseFlow storage backends.

The application still uses its existing JSON stores.  This factory only makes
it possible for a caller or migration test to construct either repository
through one small, explicit entry point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from storage import JsonStorageRepository, SQLiteStorageRepository

DEFAULT_SQLITE_FILENAME = "franchiseflow.db"
_SUPPORTED_BACKENDS = ("json", "sqlite")


def build_repository(
    *,
    backend: str,
    output_dir: str | Path,
    batch_dir: str | Path | None,
    idempotency_path: str | Path | None,
    sqlite_path: str | Path | None = None,
) -> JsonStorageRepository | SQLiteStorageRepository:
    """Build the requested storage repository for a runtime configuration.

    ``output_dir``, ``batch_dir`` and ``idempotency_path`` identify the current
    JSON stores.  SQLite keeps its control-plane database at the explicit
    ``sqlite_path`` when supplied, or at ``output_dir/franchiseflow.db`` by
    default.  Constructing a repository does not switch ``app.py`` to it.
    """
    normalized_backend = str(backend).strip().lower()
    if normalized_backend not in _SUPPORTED_BACKENDS:
        allowed = ", ".join(_SUPPORTED_BACKENDS)
        raise ValueError(f"backend must be one of: {allowed}")

    if normalized_backend == "json":
        return JsonStorageRepository(
            output_dir=Path(output_dir),
            batch_dir=None if batch_dir is None else Path(batch_dir),
            idempotency_path=(
                None if idempotency_path is None else Path(idempotency_path)
            ),
        )

    db_path = (
        Path(sqlite_path)
        if sqlite_path is not None
        else Path(output_dir) / DEFAULT_SQLITE_FILENAME
    )
    return SQLiteStorageRepository(db_path)
