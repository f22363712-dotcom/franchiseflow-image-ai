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
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


MAX_KEY_LENGTH = 255
DEFAULT_PENDING_TTL_SECONDS = 15 * 60
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_STORE_LOCK = threading.RLock()


class IdempotencyError(ValueError):
    """Base error for malformed idempotency requests or state."""


class IdempotencyConflict(IdempotencyError):
    """The key has already been used with a different request payload."""


class IdempotencyReservationError(IdempotencyError):
    """A reservation cannot be changed without creating a duplicate resource."""


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


def _hash_file_stream(file_item: Any) -> tuple[int, str]:
    stream = getattr(file_item, "stream", None)
    if stream is None:
        raise IdempotencyError("Idempotency-Key requires replayable uploaded files")
    try:
        position = stream.tell()
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        stream.seek(position)
    except (AttributeError, OSError, ValueError) as exc:
        raise IdempotencyError("Idempotency-Key requires replayable uploaded files") from exc
    return size, digest.hexdigest()


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
            size, digest = _hash_file_stream(file_item)
            payload["files"].append({
                "field": field,
                "filename": filename,
                "content_type": content_type,
                "size": size,
                "sha256": digest,
            })
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _entry_key(scope: str, key: str) -> str:
    return f"{scope}\u0000{key}"


class IdempotencyStore:
    """Atomic JSON mapping of ``(scope, key)`` to a durable resource ID.

    Entries have two states.  ``pending`` is written before the resource is
    created; ``committed`` is written after its durable state exists.  The
    legacy :meth:`put` method still creates a committed entry so existing
    callers and old index files remain compatible.
    """

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
        entries: dict[str, Any] = {}
        for key, raw_entry in dict(value.get("entries", {})).items():
            if not isinstance(raw_entry, dict):
                raise RuntimeError("idempotency state is invalid")
            entry = dict(raw_entry)
            # Version-one entries were committed by definition.  Treating a
            # missing state as committed keeps upgrades replay-safe.
            entry.setdefault("state", "committed")
            if entry["state"] not in {"pending", "committed"}:
                raise RuntimeError("idempotency state is invalid")
            entries[str(key)] = entry
        return {"version": 1, "entries": entries}

    def get(self, scope: str, key: str) -> dict[str, Any] | None:
        normalized = normalize_key(key)
        if normalized is None:
            return None
        with _STORE_LOCK:
            return self._read()["entries"].get(_entry_key(str(scope), normalized))

    @staticmethod
    def is_pending(entry: dict[str, Any] | None) -> bool:
        return bool(entry and entry.get("state", "committed") == "pending")

    @staticmethod
    def is_stale(
        entry: dict[str, Any] | None,
        *,
        now: float | None = None,
        ttl_seconds: float = DEFAULT_PENDING_TTL_SECONDS,
    ) -> bool:
        if not IdempotencyStore.is_pending(entry):
            return False
        try:
            created_at = float(entry.get("created_at", 0.0))
        except (TypeError, ValueError):
            return True
        current = time.time() if now is None else float(now)
        return current - created_at >= max(0.0, float(ttl_seconds))

    def pending_entries(self) -> list[dict[str, Any]]:
        """Return pending entries for startup reconciliation."""
        with _STORE_LOCK:
            return [dict(entry) for entry in self._read()["entries"].values()
                    if self.is_pending(entry)]

    def reserve(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        resource_id: str | None = None,
        status_code: int = 202,
    ) -> dict[str, Any]:
        """Reserve a stable resource ID before resource creation.

        Repeated reservations for the same semantic request return the first
        reservation.  A different fingerprint is rejected even when the first
        request is still pending.
        """
        normalized = normalize_key(key)
        if normalized is None:
            raise IdempotencyError("Idempotency-Key must not be empty")
        if not fingerprint:
            raise IdempotencyError("idempotency fingerprint is required")
        stable_id = str(resource_id or uuid.uuid4().hex[:12])
        if not stable_id:
            raise IdempotencyError("idempotency resource ID is required")
        with _STORE_LOCK:
            state = self._read()
            entry_key = _entry_key(str(scope), normalized)
            previous = state["entries"].get(entry_key)
            if previous is not None:
                if previous.get("fingerprint") != fingerprint:
                    raise IdempotencyConflict("Idempotency-Key was already used with a different request")
                return previous
            now = time.time()
            entry = {
                "scope": str(scope),
                "key": normalized,
                "fingerprint": fingerprint,
                "resource_id": stable_id,
                "status_code": int(status_code),
                "state": "pending",
                "created_at": now,
                "updated_at": now,
            }
            state["entries"][entry_key] = entry
            self._write(state)
            return entry

    def commit(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        resource_id: str,
        status_code: int = 202,
    ) -> dict[str, Any]:
        """Commit a reservation after its durable resource exists."""
        normalized = normalize_key(key)
        if normalized is None:
            raise IdempotencyError("Idempotency-Key must not be empty")
        if not fingerprint or not resource_id:
            raise IdempotencyError("idempotency fingerprint and resource ID are required")
        with _STORE_LOCK:
            state = self._read()
            entry_key = _entry_key(str(scope), normalized)
            previous = state["entries"].get(entry_key)
            if previous is None:
                raise IdempotencyReservationError("idempotency reservation is missing")
            if previous.get("fingerprint") != fingerprint:
                raise IdempotencyConflict("Idempotency-Key was already used with a different request")
            if str(previous.get("resource_id")) != str(resource_id):
                raise IdempotencyReservationError("idempotency resource ID does not match reservation")
            if previous.get("state", "committed") == "committed":
                return previous
            now = time.time()
            previous = dict(previous)
            previous.update({"state": "committed", "status_code": int(status_code), "updated_at": now})
            state["entries"][entry_key] = previous
            self._write(state)
            return previous

    def release(
        self,
        scope: str,
        key: str,
        fingerprint: str,
        resource_id: str | None = None,
    ) -> bool:
        """Release a pending reservation after validation fails.

        Committed entries are retained.  This prevents a late error handler
        from deleting a mapping for a resource that was already created.
        """
        normalized = normalize_key(key)
        if normalized is None:
            raise IdempotencyError("Idempotency-Key must not be empty")
        with _STORE_LOCK:
            state = self._read()
            entry_key = _entry_key(str(scope), normalized)
            previous = state["entries"].get(entry_key)
            if previous is None:
                return False
            if previous.get("fingerprint") != fingerprint:
                raise IdempotencyConflict("Idempotency-Key was already used with a different request")
            if resource_id is not None and str(previous.get("resource_id")) != str(resource_id):
                raise IdempotencyReservationError("idempotency resource ID does not match reservation")
            if previous.get("state", "committed") != "pending":
                return False
            state["entries"].pop(entry_key, None)
            self._write(state)
            return True

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
                "state": "committed",
                "created_at": time.time(),
                "updated_at": time.time(),
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
            # A browser or scanner can briefly hold the index open on
            # Windows. Retry the atomic swap before reporting storage loss.
            for attempt in range(5):
                try:
                    os.replace(temp_name, self.path)
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

