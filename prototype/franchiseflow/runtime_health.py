"""Read-only runtime diagnostics for the single-process local service."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping


APP_VERSION = "0.2.0"
_JOB_STATES = {"queued", "running", "needs_review", "completed", "failed", "vision_failed"}


def server_bind(environment: Mapping[str, str] | None = None) -> tuple[str, int]:
    environment = os.environ if environment is None else environment
    host = str(environment.get("HOST", "127.0.0.1")).strip()
    if not host or any(ord(char) < 32 for char in host):
        raise ValueError("HOST must be a non-empty host name or IP address")
    try:
        port = int(environment.get("PORT", "5000"))
    except (TypeError, ValueError) as exc:
        raise ValueError("PORT must be an integer between 1 and 65535") from exc
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be an integer between 1 and 65535")
    return host, port


def _directory_state(directory: Path) -> dict[str, bool]:
    # Permission inspection leaves the directory unchanged. This is a health
    # signal, not a guarantee against a later disk-full or filesystem failure.
    available = directory.is_dir()
    return {"available": available, "writable": available and os.access(directory, os.W_OK | os.X_OK)}


def _json_state(root: Path, pattern: str, identity: str, *, jobs: bool = False) -> dict[str, Any]:
    records = 0
    unreadable = 0
    queued = 0
    try:
        paths = list(root.glob(pattern))
    except OSError:
        return {"records": 0, "unreadable": 1, "queued": 0}
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            expected_id = path.parent.name if pattern.startswith("*/") else path.stem
            if not isinstance(value, dict) or value.get(identity) != expected_id:
                raise ValueError("invalid identity")
            if jobs and value.get("status") not in _JOB_STATES:
                raise ValueError("invalid status")
        except (OSError, UnicodeError, ValueError):
            unreadable += 1
            continue
        records += 1
        if jobs and value.get("status") == "queued":
            queued += 1
    return {"records": records, "unreadable": unreadable, "queued": queued}


def build_health_snapshot(
    *,
    upload_dir: Path,
    output_dir: Path,
    profile_dir: Path,
    batch_dir: Path,
    package_dir: Path,
    idempotency_store: Any,
    worker_active: int,
    worker_scheduled: int,
    worker_capacity: int,
    recovery_started: bool,
) -> dict[str, Any]:
    tools = {name: bool(shutil.which(name)) for name in ("ffmpeg", "ffprobe")}
    storage = {
        "uploads": _directory_state(upload_dir),
        "outputs": _directory_state(output_dir),
        "brand_profiles": _directory_state(profile_dir),
        "batches": _directory_state(batch_dir),
        "packages": _directory_state(package_dir),
    }
    jobs = _json_state(output_dir, "*/job.json", "job_id", jobs=True)
    batches = _json_state(batch_dir, "*.json", "batch_id")
    profiles = _json_state(profile_dir, "*/profile.json", "profile_id")
    try:
        pending = idempotency_store.pending_entries()
        idempotency = {
            "state": "ready" if idempotency_store.path.exists() else "not_initialized",
            "pending": len(pending),
        }
    except (OSError, RuntimeError, ValueError):
        idempotency = {"state": "unreadable", "pending": None}
    healthy = (
        all(tools.values())
        and all(item["writable"] for item in storage.values())
        and all(item["unreadable"] == 0 for item in (jobs, batches, profiles))
        and idempotency["state"] != "unreadable"
    )
    return {
        "status": "healthy" if healthy else "degraded",
        "version": APP_VERSION,
        "worker": {
            "mode": "single_process",
            "active": worker_active,
            "queued": jobs["queued"],
            "scheduled": worker_scheduled,
            "capacity": worker_capacity,
        },
        "tools": tools,
        "storage": storage,
        "persistence": {
            "backend": "local_json",
            "recovery_started": recovery_started,
            "jobs": {key: value for key, value in jobs.items() if key != "queued"},
            "batches": {key: value for key, value in batches.items() if key != "queued"},
            "brand_profiles": {key: value for key, value in profiles.items() if key != "queued"},
            "idempotency": idempotency,
        },
    }
