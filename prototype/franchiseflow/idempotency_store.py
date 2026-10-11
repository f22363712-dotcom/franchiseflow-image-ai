"""Small durable idempotency store for local FranchiseFlow POST requests.

The web app is intentionally a single-process local worker.  This module keeps
the request key/fingerprint/resource mapping on disk so a browser retry after a
process restart cannot create a second job or batch.  The application owns the
process-level lock around a check-and-create operation; this store only
provides atomic JSON persistence and deterministic request fingerprints.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable


MAX_KEY_LENGTH = 255
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_STORE_LOCK = threading.RLock()


class IdempotencyError(ValueError):
    """Base error for malformed idempotency requests or state."""


class IdempotencyConflict(IdempotencyError):
    """The key has already been used with a different request payload."""


def normalize_key(raw_key: str | None) -> str | None:
    """Validate an ``Idempotency-Key`` header.

    A missing header means idempotency is disabled for that request.  Empty,
    control-character, and overlong values are rejected so they cannot create
    ambiguous JSON keys or hide accidental client bugs.
    """
    if raw_key is None:
        return None
    raw_text = str(raw_key)
    if _CONTROL_CHARS.search(raw_text):
        raise IdempotencyError("Idempotency-Key contains control characters")
    key = raw_text.strip()
    if not key:
        raise IdempotencyError("Idempotency-Key must not be empty")
    if len(key) > MAX_KEY_LENGTH:
        raise IdempotencyError(f"Idempotency-Key must be at most {MAX_KEY_LENGTH} characters")
    return key


def _iter_multidict_items(mapping: Any) -> Iterable[tuple[str, str]]:
    """Yield all form values while retaining repeated field order."""
    keys = sorted(str(key) for key in mapping.keys())
    for key in keys:
        values = mapping.getlist(key) if hasattr(mapping, "getlist") else [mapping.get(key)]
        for value in values:
            yield key, "" if value is None else str(value)


def _read_file_bytes(file_item: Any) -> bytes:
    stream = getattr(file_item, "stream", None)
    if stream is None:
        raise IdempotencyError("Idempotency-Key requires replayable uploaded files")
    try:
        position = stream.tell()
        content = stream.read()
        stream.seek(position)
    except (AttributeError, OSError, ValueError) as exc:
        raise IdempotencyError("Idempotency-Key requires replayable uploaded files") from exc
    return bytes(content or b"")


def request_fingerprint(request: Any, scope: str) -> str:
    """Hash the semantic multipart request and endpoint scope.

    File bytes are hashed instead of embedded in the fingerprint payload.  All
    streams are rewound to their original position before returning, allowing
    the normal upload path to consume them afterwards.
    """
    payload: dict[str, Any] = {
        "scope": str(scope),
        "form": list(_iter_multidict_items(request.form)),
        "files": [],
    }
    for field in sorted(str(key) for key in request.files.keys()):
        for file_item in request.files.getlist(field):
            filename = str(getattr(file_item, "filename", "") or "")
            content_type = str(getattr(file_item, "content_type", "") or "")
            body = _read_file_bytes(file_item)
            payload["files"].append({
                "field": field,
                "filename": filename,
                "content_type": content_type,
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            })
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _entry_key(scope: str, key: str) -> str:
    return f"{scope}\u0000{key}"


class IdempotencyStore:
    """Atomic JSON mapping of ``(scope, key)`` to a durable resource ID."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "entries": {}}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("idempotency state is unreadable") from exc
        if not isinstance(value, dict) or not isinstance(value.get("entries", {}), dict):
            raise RuntimeError("idempotency state is invalid")
        return {"version": 1, "entries": dict(value.get("entries", {}))}

    def get(self, scope: str, key: str) -> dict[str, Any] | None:
        normalized = normalize_key(key)
        if normalized is None:
            return None
        with _STORE_LOCK:
            return self._read()["entries"].get(_entry_key(str(scope), normalized))

    def put(self, scope: str, key: str, fingerprint: str, resource_id: str, status_code: int = 202) -> dict[str, Any]:
        normalized = normalize_key(key)
        if normalized is None:
            raise IdempotencyError("Idempotency-Key must not be empty")
        if not fingerprint or not resource_id:
            raise IdempotencyError("idempotency fingerprint and resource ID are required")
        with _STORE_LOCK:
            state = self._read()
            entry_key = _entry_key(str(scope), normalized)
            previous = state["entries"].get(entry_key)
            if previous is not None:
                if previous.get("fingerprint") != fingerprint:
                    raise IdempotencyConflict("Idempotency-Key was already used with a different request")
                return previous
            entry = {
                "scope": str(scope),
                "key": normalized,
                "fingerprint": fingerprint,
                "resource_id": str(resource_id),
                "status_code": int(status_code),
            }
            state["entries"][entry_key] = entry
            self._write(state)
            return entry

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".idempotency-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

