"""Atomic JSON state for local FranchiseFlow jobs."""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATES = {"queued", "running", "needs_review", "completed", "failed", "vision_failed"}

ALLOWED_TRANSITIONS = {
    "queued": {"queued", "running", "needs_review", "completed", "failed"},
    "running": {"running", "queued", "needs_review", "completed", "failed", "vision_failed"},
    "needs_review": {"needs_review", "queued", "completed", "failed"},
    "completed": {"completed"},
    "failed": {"failed"},
    "vision_failed": {"vision_failed"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_path(root: Path, job_id: str) -> Path:
    if not job_id or job_id in {".", ".."} or Path(job_id).name != job_id or any(ch in job_id for ch in "\\/"):
        raise ValueError("invalid job id")
    return root / job_id / "job.json"


def create_job(root: str | Path, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    job = dict(payload)
    status = job.get("status", "queued")
    if status not in STATES:
        raise ValueError(f"unsupported job state: {status}")
    job["status"] = status
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
    if path.exists():
        previous = load_job(root, job_id)
        previous_status = previous.get("status") if previous else None
        if previous_status and status not in ALLOWED_TRANSITIONS.get(previous_status, {previous_status}):
            raise ValueError(f"illegal job state transition: {previous_status} -> {status}")
    path.parent.mkdir(parents=True, exist_ok=True)
    job["updated_at"] = _now()
    fd, temp_name = tempfile.mkstemp(prefix=".job-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(job, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Windows file scanners and browser preview processes can briefly hold
        # the destination open. Retry the atomic swap instead of losing the
        # durable state update on a transient sharing violation.
        for attempt in range(5):
            try:
                os.replace(temp_name, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
