"""Small atomic JSON store for multi-video FranchiseFlow submissions."""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BATCH_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(root: str | Path, batch_id: str) -> Path:
    if not BATCH_ID_RE.fullmatch(batch_id or ""):
        raise ValueError("invalid batch id")
    return Path(root) / f"{batch_id}.json"


def create_batch(root: str | Path, batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    batch = dict(payload)
    batch["batch_id"] = batch_id
    batch.setdefault("created_at", _now())
    batch["updated_at"] = _now()
    save_batch(root, batch_id, batch)
    return batch


def load_batch(root: str | Path, batch_id: str) -> dict[str, Any] | None:
    path = _path(root, batch_id)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"batch state is unreadable: {batch_id}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"batch state is invalid: {batch_id}")
    return value


def save_batch(root: str | Path, batch_id: str, batch: dict[str, Any]) -> None:
    path = _path(root, batch_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    batch["updated_at"] = _now()
    fd, temp_name = tempfile.mkstemp(prefix=".batch-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(batch, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
