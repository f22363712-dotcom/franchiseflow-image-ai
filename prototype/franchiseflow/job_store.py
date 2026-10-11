"""Atomic JSON state for local FranchiseFlow jobs."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATES = {"queued", "running", "needs_review", "completed", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_path(root: Path, job_id: str) -> Path:
    if not job_id or Path(job_id).name != job_id or any(ch in job_id for ch in "\\/"):
        raise ValueError("invalid job id")
    return root / job_id / "job.json"


def create_job(root: str | Path, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    job = dict(payload)
    status = job.get("status", "queued")
    if status not in STATES:
        raise ValueError(f"unsupported job state: {status}")
    job["job_id"] = job_id
    job.setdefault("created_at", _now())
    job["updated_at"] = _now()
    save_job(root, job_id, job)
    return job


def load_job(root: str | Path, job_id: str) -> dict[str, Any] | None:
    path = _job_path(Path(root), job_id)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"job state is unreadable: {job_id}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"job state is invalid: {job_id}")
    return value


def save_job(root: str | Path, job_id: str, job: dict[str, Any]) -> None:
    path = _job_path(Path(root), job_id)
    status = job.get("status")
    if status not in STATES:
        raise ValueError(f"unsupported job state: {status}")
    path.parent.mkdir(parents=True, exist_ok=True)
    job["updated_at"] = _now()
    fd, temp_name = tempfile.mkstemp(prefix=".job-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(job, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
