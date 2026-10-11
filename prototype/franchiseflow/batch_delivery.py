"""Build safe, auditable download packages for FranchiseFlow batches.

The batch index is deliberately separate from the child job artifacts.  This
module packages only child jobs that are actually completed and records every
other child in the root manifest so a partial delivery cannot look complete.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class BatchPackageError(ValueError):
    """Raised when a batch has no safe, completed artifacts to deliver."""


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _safe_child_file(output_root: Path, job_id: str, output_url: str) -> Path:
    """Resolve an output URL while keeping it inside its child job directory."""
    if not _ID_RE.fullmatch(job_id or ""):
        raise BatchPackageError(f"invalid child job id: {job_id}")
    raw = str(output_url or "")
    prefix = f"/outputs/{job_id}/"
    if raw.startswith(prefix):
        relative = raw[len(prefix):]
    else:
        # Durable job state written by older versions may contain just the
        # basename.  Do not accept arbitrary paths here.
        relative = raw
    if not relative or Path(relative).name != relative or relative in {".", ".."}:
        raise BatchPackageError(f"unsafe output path for {job_id}")
    job_dir = (output_root / job_id).resolve()
    candidate = (job_dir / relative).resolve()
    if candidate.parent != job_dir or not candidate.is_file():
        raise BatchPackageError(f"output artifact is missing or outside the job directory: {job_id}/{relative}")
    return candidate


def _safe_manifest(output_root: Path, job_id: str) -> Path:
    if not _ID_RE.fullmatch(job_id or ""):
        raise BatchPackageError(f"invalid child job id: {job_id}")
    job_dir = (output_root / job_id).resolve()
    manifest = (job_dir / "manifest.json").resolve()
    if manifest.parent != job_dir or not manifest.is_file():
        raise BatchPackageError(f"child manifest is missing: {job_id}")
    return manifest


def _job_entry(job: dict[str, Any] | None, job_id: str) -> dict[str, Any]:
    if job is None:
        return {"job_id": job_id, "status": "missing", "included": False, "reason": "job state is missing"}
    status = str(job.get("status") or "unknown")
    entry: dict[str, Any] = {
        "job_id": job_id,
        "video_file": job.get("video_file"),
        "status": status,
        "included": False,
        "branch_count": job.get("branch_count"),
        "branch_labels": job.get("branch_labels", []),
    }
    if status != "completed":
        entry["reason"] = "child job is not completed"
    if job.get("error"):
        entry["error"] = job.get("error")
    return entry


def build_batch_package(
    batch: dict[str, Any],
    jobs: dict[str, dict[str, Any] | None],
    output_root: str | Path,
    package_root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Create or replace a ZIP package and return ``(path, manifest)``.

    Completed children with missing or unsafe artifacts are reported in the
    manifest as omitted.  A package is returned only when at least one child
    is safe to include; callers can surface :class:`BatchPackageError` as a
    ``409 Conflict`` while the batch is still processing.
    """
    batch_id = str(batch.get("batch_id") or "")
    if not _ID_RE.fullmatch(batch_id):
        raise BatchPackageError("invalid batch id")
    output_root = Path(output_root)
    package_root = Path(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    included: list[tuple[str, Path, list[tuple[str, Path, str, int]]]] = []
    for raw_job_id in batch.get("job_ids", []):
        job_id = str(raw_job_id)
        job = jobs.get(job_id)
        entry = _job_entry(job, job_id)
        if job is None or job.get("status") != "completed":
            entries.append(entry)
            continue
        try:
            manifest = _safe_manifest(output_root, job_id)
            raw_outputs = job.get("outputs") or job.get("output_urls") or []
            if not isinstance(raw_outputs, list) or not raw_outputs:
                raise BatchPackageError(f"completed child has no output videos: {job_id}")
            artifacts: list[tuple[str, Path, str, int]] = []
            seen_filenames: set[str] = set()
            for raw_output in raw_outputs:
                path = _safe_child_file(output_root, job_id, str(raw_output))
                if path.name in seen_filenames:
                    raise BatchPackageError(f"duplicate output artifact for {job_id}: {path.name}")
                seen_filenames.add(path.name)
                digest, size = _sha256_and_size(path)
                artifacts.append((path.name, path, digest, size))
            manifest_digest, manifest_size = _sha256_and_size(manifest)
            entry.update({
                "included": True,
                "manifest_path": f"jobs/{job_id}/manifest.json",
                "manifest_sha256": manifest_digest,
                "manifest_bytes": manifest_size,
                "outputs": [
                    {
                        "filename": filename,
                        "path": f"jobs/{job_id}/outputs/{filename}",
                        "sha256": digest,
                        "bytes": size,
                    }
                    for filename, _path, digest, size in artifacts
                ],
            })
            included.append((job_id, manifest, artifacts))
        except (BatchPackageError, OSError) as exc:
            entry["reason"] = str(exc)
        entries.append(entry)

    if not included:
        raise BatchPackageError("batch has no completed output artifacts yet")

    package_manifest: dict[str, Any] = {
        "package_format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "status": batch.get("status"),
        "brand_profile_id": batch.get("brand_profile_id"),
        "requested_count": batch.get("requested_count", len(entries)),
        "counts": batch.get("counts", {}),
        "jobs": entries,
        "included_job_count": len(included),
        "omitted_job_count": len(entries) - len(included),
    }
    final_path = package_root / f"{batch_id}.zip"
    temp_path = package_root / f".{batch_id}.{uuid.uuid4().hex}.tmp"
    try:
        with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            archive.writestr(
                "batch_manifest.json",
                json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n",
            )
            for job_id, manifest, artifacts in included:
                archive.write(manifest, f"jobs/{job_id}/manifest.json")
                for filename, path, _digest, _size in artifacts:
                    archive.write(path, f"jobs/{job_id}/outputs/{filename}")
        try:
            os.replace(temp_path, final_path)
        except PermissionError:
            # A previous Flask ``send_file`` response may still hold the
            # cached archive open on Windows.  Keep that download valid and
            # publish this regeneration under a fresh path.
            final_path = package_root / f"{batch_id}-{uuid.uuid4().hex}.zip"
            os.replace(temp_path, final_path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
    return final_path, package_manifest
