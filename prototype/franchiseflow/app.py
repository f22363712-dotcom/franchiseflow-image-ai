from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# Support imports from spikes/franchiseflow.
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "spikes" / "franchiseflow"))

from candidate_validator import validate_candidate
from job_store import create_job, load_job, save_job
from batch_store import create_batch, load_batch, save_batch
from brand_profiles import create_profile, list_profiles, load_profile, profile_asset_paths
from tracker import _normalize_asset_bindings, normalize_branch_labels, parse_quad, run_tracker
from scene_samples import sample_video_scenes
from scene_reuse import transfer_candidate
from vision_client import VisionClient
from input_media import (
    SUPPORTED_SOURCE_EXTS,
    inspect_video,
    normalize_video_for_processing,
    probe_video_codec,
    source_requires_normalization,
)

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR = BASE_DIR / "brand_profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR = OUTPUT_DIR / "batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTS = SUPPORTED_SOURCE_EXTS
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MAX_BRANCHES = 10
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
# Processing budgets are deliberately explicit.  They keep the local worker
# bounded while allowing real brand shoots with several hard-cut shots.  A
# caller can still tighten either budget in a deployment or test by replacing
# these module constants before submitting work.
MAX_VIDEO_SECONDS = 60.0
MAX_SCENES = 8
MAX_ACTIVE_JOBS = 4
# A batch is deliberately bounded. Each item still uses the normal single-job
# pipeline and durable job record; this limit prevents one request from
# filling the local queue indefinitely.
MAX_BATCH_VIDEOS = 8
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_IMAGE_PIXELS = 20_000_000
MAX_SURFACES_PER_SCENE = 4
SURFACE_OVERLAP_THRESHOLD = 0.25
MAX_PROFILE_COUNT = 100

# Reject oversized multipart bodies before Werkzeug spools them to disk. The
# small allowance covers multipart headers and form fields; the exact payload
# total is checked again after each file is saved below.
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + (4 * 1024 * 1024)

JOBS: dict[str, dict[str, Any]] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="franchiseflow")
FUTURES: dict[str, Any] = {}
FUTURE_LOCK = threading.Lock()
SYNC_LOCK = threading.Lock()
SYNC_ACTIVE = 0
_RECOVERY_STARTED = False


def _set_job(job: dict[str, Any]) -> dict[str, Any]:
    JOBS[job["job_id"]] = job
    save_job(OUTPUT_DIR, job["job_id"], job)
    _write_manifest(job)
    return job


def _public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove local filesystem paths before a summary reaches the API."""
    public = dict(summary or {})
    if public.get("video"):
        public["video"] = Path(str(public["video"])).name
    if isinstance(public.get("output_videos"), list):
        public["output_videos"] = [Path(str(item)).name for item in public["output_videos"]]
    return public


def _write_manifest(job: dict[str, Any]) -> None:
    """Persist a small, secret-free audit record beside each job."""
    job_dir = OUTPUT_DIR / job["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    summary = job.get("summary") or {}
    manifest = {
        "job_id": job.get("job_id"),
        "batch_id": job.get("batch_id"),
        "status": job.get("status"),
        "phase": job.get("phase"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "video_file": job.get("video_file"),
        "processing_video_file": job.get("processing_video_file"),
        "video_metadata": job.get("video_metadata"),
        "branch_files": job.get("branch_files", []),
        "branch_labels": job.get("branch_labels", []),
        "branch_count": job.get("branch_count"),
        "brand_profile_id": job.get("brand_profile_id"),
        "brand_profile_default_fit": job.get("brand_profile_default_fit"),
        "brand_profile_surface_policy": job.get("brand_profile_surface_policy"),
        "brand_profile_scene_reuse_policy": job.get("brand_profile_scene_reuse_policy"),
        "brand_profile_assets": job.get("brand_profile_assets", []),
        "initialization_mode": job.get("initialization_mode"),
        "model": job.get("model"),
        "target_hint": job.get("target_hint"),
        "candidate": job.get("candidate"),
        "candidates": job.get("candidates", []),
        "asset_bindings": job.get("asset_bindings"),
        "manual_scene_quads": job.get("manual_scene_quads"),
        "reviewed_quad": job.get("reviewed_quad"),
        "vision_audit": job.get("vision_audit"),
        "scenes": job.get("scenes", []),
        "sample_preview": job.get("sample_preview"),
        "contact_sheet_preview": job.get("contact_sheet_preview"),
        "scene_plan": job.get("scene_plan", []),
        "outputs": job.get("outputs", []),
        "quality_review": job.get("quality_review"),
        "summary": {
            key: value for key, value in summary.items()
            if key not in {"output_videos"}
        },
        "error": job.get("error"),
    }
    temp = job_dir / ".manifest.json.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, job_dir / "manifest.json")


def _job(job_id: str) -> dict[str, Any] | None:
    if not JOB_ID_RE.fullmatch(job_id or ""):
        return None
    try:
        job = load_job(OUTPUT_DIR, job_id)
    except (ValueError, RuntimeError, OSError):
        return None
    if job is not None:
        JOBS[job_id] = job
    return job


def _list_jobs(limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    """Read durable job summaries so review work survives a browser restart."""
    limit = max(1, min(int(limit), 100))
    jobs: list[dict[str, Any]] = []
    for state_path in OUTPUT_DIR.glob("*/job.json"):
        try:
            job = load_job(OUTPUT_DIR, state_path.parent.name)
        except (ValueError, RuntimeError, OSError):
            continue
        if not job or (status and job.get("status") != status):
            continue
        jobs.append({
            "job_id": job.get("job_id", state_path.parent.name),
            "batch_id": job.get("batch_id"),
            "status": job.get("status"),
            "phase": job.get("phase"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "branch_count": job.get("branch_count"),
            "branch_labels": job.get("branch_labels", []),
            "scene_count": len(job.get("scenes", [])) if isinstance(job.get("scenes"), list) else None,
            "initialization_mode": job.get("initialization_mode"),
            "error": job.get("error"),
            "outputs": job.get("outputs", []),
        })
    jobs.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return jobs[:limit]


def _active_future_count() -> int:
    with FUTURE_LOCK:
        return sum(1 for future in FUTURES.values() if not future.done())


def _active_slot_count() -> int:
    with SYNC_LOCK:
        sync_active = SYNC_ACTIVE
    return _active_future_count() + sync_active


def _future_done(job_id: str, future: Any) -> None:
    with FUTURE_LOCK:
        if FUTURES.get(job_id) is future:
            FUTURES.pop(job_id, None)
    if _RECOVERY_STARTED:
        _schedule_queued_jobs()


def _submit_job(job_id: str) -> None:
    with SYNC_LOCK:
        with FUTURE_LOCK:
            existing = FUTURES.get(job_id)
            if existing is not None and not existing.done():
                return
            active = SYNC_ACTIVE + sum(1 for future in FUTURES.values() if not future.done())
            if active >= MAX_ACTIVE_JOBS:
                raise RuntimeError(f"active job queue is full ({MAX_ACTIVE_JOBS} jobs); try again later")
            future = EXECUTOR.submit(_process_job, job_id)
            FUTURES[job_id] = future
    # A completed Future invokes its callback synchronously. Register outside
    # FUTURE_LOCK so an immediate executor cannot deadlock the request thread.
    future.add_done_callback(lambda completed, key=job_id: _future_done(key, completed))


def _reserve_sync_slot() -> bool:
    global SYNC_ACTIVE
    with SYNC_LOCK:
        if _active_future_count() + SYNC_ACTIVE >= MAX_ACTIVE_JOBS:
            return False
        SYNC_ACTIVE += 1
        return True


def _release_sync_slot() -> None:
    global SYNC_ACTIVE
    with SYNC_LOCK:
        SYNC_ACTIVE = max(0, SYNC_ACTIVE - 1)


def _first_frame(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError("unable to decode first frame from uploaded video")
    return frame


def _frame_at(path: Path, frame_index: int) -> np.ndarray:
    """Decode one source frame for scene-local human quad validation."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened() or frame_index < 0:
            raise ValueError("unable to open uploaded video for scene review")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"unable to decode scene start frame {frame_index}")
        return frame
    finally:
        cap.release()


def _save_contact_sheet(samples: list[tuple[int, np.ndarray]], directory: Path) -> tuple[bytes, int, int, float]:
    first_height, first_width = samples[0][1].shape[:2]
    scale = min(1.0, 640.0 / first_width)
    panel_width = max(1, round(first_width * scale))
    panel_height = max(1, round(first_height * scale))
    panels = []
    for frame_index, frame in samples:
        panel = cv2.resize(frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, f"frame {frame_index}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        panels.append(panel)
    sheet = np.concatenate(panels, axis=1)
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("failed to encode vision contact sheet")
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "first.jpg"), samples[0][1])
    cv2.imwrite(str(directory / "contact_sheet.jpg"), sheet)
    return encoded.tobytes(), panel_width, panel_height, scale


def _branch_files(request_files) -> list[Any]:
    branches = [item for item in request_files.getlist("branches") if item and item.filename]
    if not branches:
        branches = [request_files.get(f"branch_{i}") for i in range(1, MAX_BRANCHES + 1)]
        branches = [item for item in branches if item and item.filename]
    return branches


def _source_filename(item: Any) -> str:
    filename = getattr(item, "filename", None)
    return str(filename) if filename else Path(item).name


def _branch_labels(raw_value: str | None, branch_files: list[Any], default_labels: list[str] | None = None) -> list[str]:
    """Read optional human labels while keeping uploaded filenames as a fallback."""
    labels = None
    if raw_value:
        try:
            labels = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid branch_labels JSON: {exc}") from exc
    if labels is None:
        labels = default_labels or [Path(_source_filename(item)).stem for item in branch_files]
    try:
        return normalize_branch_labels(labels, [Path(_source_filename(item)) for item in branch_files])
    except ValueError as exc:
        raise ValueError(f"Invalid branch_labels: {exc}") from exc


def _parse_manual_scene_quads(raw_value: str | None) -> list[list[list[float]]] | None:
    """Parse optional per-scene operator calibration in detected-scene order."""
    if not raw_value:
        return None
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid manual_scene_quads JSON: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("manual_scene_quads must be a non-empty JSON array")
    try:
        return [parse_quad(item).round(3).tolist() for item in payload]
    except (TypeError, ValueError, cv2.error) as exc:
        raise ValueError(f"Invalid manual_scene_quads: {exc}") from exc


def _create_job_from_request(
    video_file_override: Any | None = None,
    batch_id_override: str | None = None,
) -> dict[str, Any]:
    """Create one durable job from the current request.

    ``video_file_override`` is used only by ``/api/batches`` so each uploaded
    video can pass through the exact same validation and storage path as
    ``/api/jobs`` without mutating Flask's request files.
    """
    video_file = video_file_override if video_file_override is not None else request.files.get("video")
    if video_file is None or not video_file.filename:
        raise ValueError("Missing video file")
    if Path(video_file.filename).suffix.lower() not in ALLOWED_VIDEO_EXTS:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTS))
        raise ValueError(f"Video must use a supported video extension: {allowed}")

    brand_profile_id = request.form.get("brand_profile_id") or None
    brand_profile = None
    if brand_profile_id:
        try:
            brand_profile = load_profile(PROFILE_DIR, brand_profile_id)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(str(exc)) from exc
        if brand_profile is None:
            raise ValueError(f"brand profile not found: {brand_profile_id}")
    branch_files = _branch_files(request.files)
    if brand_profile and branch_files:
        raise ValueError("brand_profile_id cannot be combined with branch uploads")
    if brand_profile:
        branch_files = profile_asset_paths(PROFILE_DIR, brand_profile)
    if not 1 <= len(branch_files) <= MAX_BRANCHES:
        raise ValueError(f"1 to 10 branch image assets required, got {len(branch_files)}")
    for branch in branch_files:
        filename = _source_filename(branch)
        if Path(filename).suffix.lower() not in ALLOWED_IMAGE_EXTS:
            raise ValueError(f"Branch image {filename} has unsupported extension")
    profile_labels = brand_profile.get("branch_labels") if brand_profile else None
    branch_labels = _branch_labels(request.form.get("branch_labels"), branch_files, default_labels=profile_labels)

    job_id = uuid.uuid4().hex[:8]
    upload_dir = UPLOAD_DIR / job_id
    output_dir = OUTPUT_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_name = secure_filename(video_file.filename) or "input.mp4"
        video_file.save(str(upload_dir / video_name))
        branch_names: list[str] = []
        for index, branch in enumerate(branch_files, start=1):
            source_name = _source_filename(branch)
            suffix = Path(source_name).suffix.lower()
            name = f"branch_{index}{suffix}"
            if hasattr(branch, "save"):
                branch.save(str(upload_dir / name))
            else:
                shutil.copyfile(branch, upload_dir / name)
            branch_names.append(name)

        source_video_path = upload_dir / video_name
        # Keep the upload unchanged for audit.  Non-MP4 sources are converted
        # once into a deterministic working asset so HEVC/MOV/WebM inputs do
        # not depend on whatever codec support the local OpenCV build happens
        # to have.  Existing MP4 flows retain their original path.
        processing_video_name = video_name
        source_codec = probe_video_codec(source_video_path)
        if source_requires_normalization(source_video_path):
            processing_video_name = "processing_input.mp4"
            processing_path = upload_dir / processing_video_name
            try:
                normalize_video_for_processing(source_video_path, processing_path)
            except RuntimeError as exc:
                raise ValueError(f"video normalization failed: {exc}") from exc
            if processing_path.stat().st_size > MAX_UPLOAD_BYTES:
                raise ValueError("normalized video exceeds the upload size limit")
        video_path = upload_dir / processing_video_name
        branch_paths = [upload_dir / name for name in branch_names]
        total_bytes = source_video_path.stat().st_size + sum(path.stat().st_size for path in branch_paths)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise ValueError(f"upload size exceeds the {MAX_UPLOAD_BYTES} byte limit")
        video_metadata = inspect_video(video_path)
        video_metadata["source_extension"] = source_video_path.suffix.lower()
        video_metadata["processing_extension"] = video_path.suffix.lower()
        video_metadata["source_codec"] = source_codec.get("codec_name")
        video_metadata["source_pixel_format"] = source_codec.get("pix_fmt")
        duration = float(video_metadata["duration_seconds_estimate"])
        if duration > MAX_VIDEO_SECONDS:
            raise ValueError(f"video duration {duration:.2f}s exceeds the {MAX_VIDEO_SECONDS:.2f}s limit")
        first_frame = _first_frame(video_path)

        for path in branch_paths:
            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.size == 0:
                raise ValueError(f"branch image {path.name} could not be decoded")
            if decoded.shape[0] * decoded.shape[1] > MAX_IMAGE_PIXELS:
                raise ValueError(f"branch image {path.name} exceeds the {MAX_IMAGE_PIXELS} pixel limit")

        raw_manual_scene_quads = request.form.get("manual_scene_quads")
        manual = request.form.get("use_manual_quad", "false").lower() in {"true", "1", "yes"} or bool(raw_manual_scene_quads)
        raw_asset_bindings = request.form.get("asset_bindings")
        asset_bindings = None
        if raw_asset_bindings:
            try:
                asset_bindings = json.loads(raw_asset_bindings)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid asset_bindings JSON: {exc}") from exc
            if not isinstance(asset_bindings, list):
                raise ValueError("asset_bindings must be a JSON array")
        manual_quad = None
        manual_scene_quads = _parse_manual_scene_quads(raw_manual_scene_quads)
        if manual:
            raw_quad = request.form.get("manual_quad")
            if raw_quad:
                try:
                    manual_quad = parse_quad(json.loads(raw_quad)).round(3).tolist()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid manual quad JSON: {exc}") from exc
            else:
                default_quad_path = REPO_ROOT / "spikes" / "franchiseflow" / "quad.json"
                if not default_quad_path.exists():
                    raise ValueError("manual quad is required when no default calibration exists")
                manual_quad = parse_quad(default_quad_path).round(3).tolist()
            frame_height, frame_width = first_frame.shape[:2]
            manual_points = np.asarray(manual_quad, dtype=np.float32)
            if (np.any(manual_points[:, 0] < 0) or np.any(manual_points[:, 0] >= frame_width) or
                    np.any(manual_points[:, 1] < 0) or np.any(manual_points[:, 1] >= frame_height)):
                raise ValueError("manual quad lies outside uploaded video frame")
            if manual_scene_quads:
                for index, scene_quad in enumerate(manual_scene_quads, start=1):
                    scene_points = np.asarray(scene_quad, dtype=np.float32)
                    if (np.any(scene_points[:, 0] < 0) or np.any(scene_points[:, 0] >= frame_width) or
                            np.any(scene_points[:, 1] < 0) or np.any(scene_points[:, 1] >= frame_height)):
                        raise ValueError(f"manual scene quad {index} lies outside uploaded video frame")
    except ValueError:
        shutil.rmtree(upload_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    job = create_job(OUTPUT_DIR, job_id, {
        "status": "queued",
        "phase": "queued",
        "batch_id": batch_id_override,
        "video_file": video_name,
        "processing_video_file": processing_video_name,
        "video_metadata": video_metadata,
        "branch_files": branch_names,
        "branch_labels": branch_labels,
        "branch_count": len(branch_names),
        "initialization_mode": "manual_fallback" if manual else "vision_model",
        "manual_quad": manual_quad,
        "manual_scene_quads": manual_scene_quads,
        "asset_bindings": asset_bindings,
        "target_hint": request.form.get("target_hint") or (brand_profile.get("target_hint") if brand_profile else None),
        "brand_profile_id": brand_profile_id,
        "brand_profile_default_fit": brand_profile.get("default_fit") if brand_profile else None,
        "brand_profile_surface_policy": brand_profile.get("surface_policy") if brand_profile else None,
        "brand_profile_scene_reuse_policy": brand_profile.get("scene_reuse_policy") if brand_profile else None,
        "brand_profile_assets": brand_profile.get("branch_assets", []) if brand_profile else [],
        "output_urls": [],
    })
    JOBS[job_id] = job
    return job


def _paths(job: dict[str, Any]) -> tuple[Path, list[Path], Path]:
    job_id = job["job_id"]
    upload_dir = UPLOAD_DIR / job_id
    processing_name = job.get("processing_video_file") or job["video_file"]
    return upload_dir / processing_name, [upload_dir / name for name in job["branch_files"]], OUTPUT_DIR / job_id


def _read_brand_reference_images(sign_paths: list[Path]) -> list[tuple[bytes, str]]:
    """Load uploaded branch artwork as optional appearance references.

    The vision client caps the number and size of references.  This helper
    keeps file/path concerns in the web layer and preserves the original
    branch files for compositing unchanged.
    """
    mime_by_suffix = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    references: list[tuple[bytes, str]] = []
    for path in sign_paths[:4]:
        mime = mime_by_suffix.get(path.suffix.lower())
        if mime is None:
            continue
        try:
            references.append((path.read_bytes(), mime))
        except OSError:
            continue
    return references


def _normalise_candidate(candidate: dict[str, Any], scale: float, panel_width: int | None = None, panel_count: int | None = None) -> dict[str, Any]:
    points = [[float(point[0]), float(point[1])] for point in candidate["quad"]]
    panel_index = None
    if panel_width and panel_count:
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        mean_x = sum(point[0] for point in points) / len(points)
        panel_index = min(panel_count - 1, max(0, int(mean_x // panel_width)))
        left = panel_index * panel_width
        right = left + panel_width
        if max_x - min_x > panel_width * 1.05 or min_x < left - 1 or max_x > right + 1:
            raise ValueError("candidate crosses contact-sheet panel boundary")
        points = [[point[0] - left, point[1]] for point in points]
    result = {
        **candidate,
        "quad": [[round(point[0] / scale, 3), round(point[1] / scale, 3)] for point in points],
    }
    if panel_index is not None:
        result["source_panel_index"] = panel_index
    return result


def _quad_overlap_ratio(first: list | np.ndarray, second: list | np.ndarray) -> float:
    """Return intersection area divided by the smaller quad area.

    A high ratio means the model likely returned the same physical surface
    twice.  We refuse to choose between such candidates automatically.
    """
    first_points = parse_quad(first).astype(np.float32)
    second_points = parse_quad(second).astype(np.float32)
    first_area = abs(float(cv2.contourArea(first_points)))
    second_area = abs(float(cv2.contourArea(second_points)))
    if min(first_area, second_area) <= 0.0:
        return 1.0
    try:
        intersection, _ = cv2.intersectConvexConvex(first_points, second_points)
    except cv2.error:
        return 1.0
    return max(0.0, min(1.0, float(intersection) / min(first_area, second_area)))


def _stable_candidate_key(candidate: dict[str, Any]) -> tuple[float, float, float, str]:
    points = np.asarray(candidate["quad"], dtype=np.float32)
    return (
        round(float(np.min(points[:, 1])), 3),
        round(float(np.min(points[:, 0])), 3),
        -round(float(candidate.get("confidence", 0.0)), 6),
        json.dumps(candidate["quad"], separators=(",", ":")),
    )


def _review_candidate_error(
    candidates: list[dict[str, Any]],
    model: str,
    audit: dict[str, Any],
    reason: str,
    validations: list[dict[str, Any]] | None = None,
) -> "CandidateReviewRequired":
    validation = {
        "accepted": False,
        "requires_human_review": True,
        "reasons": [reason],
        "candidate_validations": validations or [],
    }
    audit["validation_reasons"] = [reason]
    audit["requires_human_review"] = True
    return CandidateReviewRequired(
        candidates[0] if candidates else None,
        validation,
        model,
        audit,
        candidates=candidates,
        validations=validations or [],
    )


def _locate_and_validate(
    video_path: Path,
    samples: list[tuple[int, np.ndarray]],
    sample_dir: Path,
    target_hint: str | None,
    reference_images: list[tuple[bytes, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    contact_sheet, panel_width, panel_height, scale = _save_contact_sheet(samples, sample_dir)
    first_height, first_width = samples[0][1].shape[:2]
    client = VisionClient()
    panel_count = len(samples)
    sheet_width = panel_width * panel_count
    try:
        result = client.locate_candidates(
            contact_sheet,
            sheet_width,
            panel_height,
            target_hint=target_hint,
            reference_images=reference_images,
        )
    except RuntimeError as exc:
        # Invalid/no-candidate model output is recoverable by drawing a quad in
        # the review UI.  Network/authentication failures remain hard errors.
        message = str(exc)
        if message.startswith("vision_failed: invalid candidate response") or message.startswith("vision model could not find a replaceable planar sign"):
            audit = {
                "input_sha256": hashlib.sha256(contact_sheet).hexdigest(),
                "candidate_count": 0,
                "model": str(getattr(client, "model", "unknown")),
                "reference_asset_count": len(reference_images or []),
                "reference_asset_sha256": [hashlib.sha256(item[0]).hexdigest() for item in (reference_images or [])],
            }
            raise _review_candidate_error([], audit["model"], audit, message) from exc
        raise
    # Keep old mocked clients and old gateways usable during migration.
    if not isinstance(result, dict) or not isinstance(result.get("candidates"), list):
        encoded_first = cv2.imencode(".jpg", samples[0][1])[1].tobytes()
        legacy = client.locate_sign(
            encoded_first,
            first_width,
            first_height,
            target_hint=target_hint,
            reference_images=reference_images,
        )
        result = {"model": legacy.get("model", client.model), "candidates": [legacy]}

    audit = {
        "input_sha256": hashlib.sha256(contact_sheet).hexdigest(),
        "candidate_count": len(result["candidates"]),
        "model": str(result.get("model", client.model)),
        "reference_asset_count": len(reference_images or []),
        "reference_asset_sha256": [hashlib.sha256(item[0]).hexdigest() for item in (reference_images or [])],
    }

    model = str(result.get("model", client.model))
    raw_candidates = result["candidates"]
    if not raw_candidates:
        raise _review_candidate_error([], model, audit, "the model returned no candidates")
    candidates: list[dict[str, Any]] = []
    for raw_candidate in result["candidates"]:
        try:
            candidate = _normalise_candidate(raw_candidate, scale, panel_width=panel_width, panel_count=panel_count)
        except (TypeError, ValueError) as exc:
            audit["candidate_count"] = len(raw_candidates)
            raise _review_candidate_error([], model, audit, f"invalid contact-sheet candidate: {exc}") from exc
        candidates.append(candidate)

    audit["candidate_count"] = len(candidates)
    if len(candidates) > MAX_SURFACES_PER_SCENE:
        raise _review_candidate_error(candidates, model, audit, f"more than {MAX_SURFACES_PER_SCENE} candidate surfaces")

    validations: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            validation = validate_candidate(samples, candidate)
        except (TypeError, ValueError, cv2.error) as exc:
            validation = {
                "accepted": False,
                "requires_human_review": True,
                "reasons": [f"candidate validation failed: {exc}"],
                "sample_metrics": [],
            }
        validations.append(validation)
    if any(not item.get("accepted") for item in validations):
        reasons = [reason for item in validations for reason in item.get("reasons", [])]
        raise _review_candidate_error(candidates, model, audit, "; ".join(reasons) or "one or more candidates failed validation", validations)

    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            overlap = _quad_overlap_ratio(left["quad"], right["quad"])
            if overlap >= SURFACE_OVERLAP_THRESHOLD:
                raise _review_candidate_error(
                    candidates,
                    model,
                    audit,
                    f"ambiguous candidate overlap ({overlap:.3f})",
                    validations,
                )

    ordered = sorted(zip(candidates, validations), key=lambda item: _stable_candidate_key(item[0]))
    candidates = [item[0] for item in ordered]
    validations = [item[1] for item in ordered]
    audit["accepted_candidate_count"] = len(candidates)
    audit["surface_count"] = len(candidates)
    audit["accepted_candidate_confidences"] = [item.get("confidence") for item in candidates]
    audit["validation_reasons"] = [item.get("reasons", []) for item in validations]
    return candidates, validations, model, audit


class CandidateReviewRequired(RuntimeError):
    def __init__(self, candidate: dict[str, Any] | None, validation: dict[str, Any], model: str, audit: dict[str, Any], *, candidates: list[dict[str, Any]] | None = None, validations: list[dict[str, Any]] | None = None):
        super().__init__("candidate requires human review")
        self.candidate = candidate
        self.candidates = candidates or ([candidate] if candidate else [])
        self.validations = validations or []
        self.validation = validation
        self.model = model
        self.audit = audit


def _scene_plan_entry(
    scene: dict[str, Any],
    quad: list | np.ndarray | None = None,
    *,
    mode: str,
    status: str = "validated",
    surface_id: str | None = None,
    surfaces: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the stable manifest consumed by the scene-aware tracker."""
    scene_id = str(scene["scene_id"])
    if surfaces is None:
        if quad is None:
            raise ValueError("scene plan requires quad or surfaces")
        surfaces = [{"quad": quad, "surface_id": surface_id}]
    normalized_surfaces: list[dict[str, Any]] = []
    for index, item in enumerate(surfaces, start=1):
        points = parse_quad(item.get("quad", item.get("corners"))).round(3).tolist()
        normalized_surfaces.append({
            "surface_id": str(item.get("surface_id") or f"{scene_id}_surface_{index:03d}"),
            "quad": points,
            "initialization_mode": str(item.get("initialization_mode") or mode),
            "status": str(item.get("status") or status),
        })
    return {
        "scene_id": scene_id,
        "start_frame": int(scene["start_frame"]),
        "end_frame": int(scene["end_frame"]),
        "cut_before": scene.get("cut_before"),
        "cut_score": float(scene.get("cut_score", 0.0)),
        "transition_type": str(scene.get("transition_type", "none")),
        "surfaces": normalized_surfaces,
    }


def _scene_view(
    scene: dict[str, Any],
    sample_preview: str,
    candidate: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    *,
    status: str = "validated",
    mode: str = "vision_model",
    candidates: list[dict[str, Any]] | None = None,
    validations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return JSON-only scene state for the API and browser review panel."""
    result = {
        "scene_id": scene["scene_id"],
        "start_frame": int(scene["start_frame"]),
        "end_frame": int(scene["end_frame"]),
        "cut_before": scene.get("cut_before"),
        "cut_score": float(scene.get("cut_score", 0.0)),
        "transition_type": str(scene.get("transition_type", "none")),
        "status": status,
        "initialization_mode": mode,
        "preview": sample_preview,
        "candidate": candidate,
        "candidates": candidates or ([candidate] if candidate else []),
        "validation": validation,
        "validations": validations or [],
        "surfaces": [],
    }
    candidate_items = candidates or ([candidate] if candidate else [])
    result["surfaces"] = [{
        "surface_id": f"{scene['scene_id']}_surface_{index:03d}",
        "quad": item.get("quad"),
        "status": status,
        "initialization_mode": mode,
        "candidate": item,
        "validation": (validations[index - 1] if validations and index - 1 < len(validations) else validation),
    } for index, item in enumerate(candidate_items, start=1) if item.get("quad")]
    result["surface_count"] = len(result["surfaces"])
    return result


def _manual_candidate(quad: list | np.ndarray) -> dict[str, Any]:
    """Represent an operator calibration without implying model provenance."""
    return {
        "quad": parse_quad(quad).round(3).tolist(),
        "confidence": 1.0,
        "reason": "operator_calibrated",
    }


def _validate_manual_candidate(scene: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Run the same multi-frame safety checks for an offline calibration."""
    try:
        validation = validate_candidate(scene["samples"], candidate)
    except (TypeError, ValueError, cv2.error) as exc:
        validation = {
            "accepted": False,
            "requires_human_review": True,
            "reasons": [f"manual candidate validation failed: {exc}"],
            "sample_metrics": [],
        }
    return {**validation, "source": "manual_fallback"}


def _reuse_scene_candidates(
    anchor_frame: np.ndarray,
    anchor_candidates: list[dict[str, Any]],
    scene: dict[str, Any],
    source_scene_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
    """Transfer every prior surface only when each one passes fresh checks."""
    transferred: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for candidate in anchor_candidates:
        projected = transfer_candidate(anchor_frame, scene["samples"][0][1], candidate.get("quad"))
        if projected is None:
            return None
        validation = validate_candidate(scene["samples"], projected)
        if not validation.get("accepted"):
            return None
        projected["reason"] = f"validated_reuse_from_{source_scene_id}"
        transferred.append(projected)
        validations.append({**validation, "source": "scene_reuse", "source_scene_id": source_scene_id})
    for left_index, left in enumerate(transferred):
        for right in transferred[left_index + 1:]:
            overlap = _quad_overlap_ratio(left["quad"], right["quad"])
            if overlap >= SURFACE_OVERLAP_THRESHOLD:
                return None
    return transferred, validations, {
        "model": "scene_reuse",
        "source": "scene_reuse",
        "source_scene_id": source_scene_id,
        "candidate_count": len(transferred),
        "accepted_candidate_count": len(transferred),
        "surface_count": len(transferred),
        "validation_reasons": [item.get("reasons", []) for item in validations],
    }


def _locate_scenes(
    video_path: Path,
    scenes: list[dict[str, Any]],
    sample_dir: Path,
    target_hint: str | None,
    job_id: str,
    scene_reuse_policy: str = "vision_each_scene",
    reference_images: list[tuple[bytes, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, dict[str, Any]]:
    """Locate up to four non-overlapping planar surfaces per scene."""
    views: list[dict[str, Any]] = []
    plan: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    model_name: str | None = None
    previous_anchor: np.ndarray | None = None
    previous_candidates: list[dict[str, Any]] | None = None
    previous_scene_id: str | None = None
    for scene in scenes:
        scene_dir = sample_dir / scene["scene_id"]
        samples = scene["samples"]
        reused = None
        reuse_attempted = bool(
            scene_reuse_policy == "reuse_if_verified"
            and previous_anchor is not None
            and previous_candidates
            and previous_scene_id
        )
        if reuse_attempted:
            reused = _reuse_scene_candidates(previous_anchor, previous_candidates, scene, previous_scene_id)
        try:
            if reused is not None:
                candidates, validations, audit = reused
                model = "scene_reuse"
            else:
                candidates, validations, model, audit = _locate_and_validate(
                    video_path,
                    samples,
                    scene_dir,
                    target_hint,
                    reference_images=reference_images,
                )
                if reuse_attempted:
                    audit["reuse_attempted"] = True
                    audit["reuse_fallback"] = True
                    audit["reuse_source_scene_id"] = previous_scene_id
        except CandidateReviewRequired as exc:
            review_audit = dict(exc.audit)
            if reuse_attempted:
                review_audit["reuse_attempted"] = True
                review_audit["reuse_fallback"] = True
                review_audit["reuse_source_scene_id"] = previous_scene_id
            views.append(_scene_view(
                scene,
                f"/outputs/{job_id}/samples/{scene['scene_id']}/first.jpg",
                exc.candidate,
                exc.validation,
                status="needs_review",
                mode="vision_model",
                candidates=exc.candidates,
                validations=exc.validations,
            ))
            audits.append(review_audit)
            model_name = model_name or exc.model
            continue
        mode = "reused_candidate" if reused is not None else "vision_model"
        view = _scene_view(
            scene,
            f"/outputs/{job_id}/samples/{scene['scene_id']}/first.jpg",
            candidates[0],
            validations[0],
            status="validated",
            mode=mode,
            candidates=candidates,
            validations=validations,
        )
        views.append(view)
        plan.append(_scene_plan_entry(
            scene,
            mode=mode,
            surfaces=[
                {"surface_id": f"{scene['scene_id']}_surface_{index:03d}", "quad": item["quad"], "status": "validated", "initialization_mode": mode}
                for index, item in enumerate(candidates, start=1)
            ],
        ))
        audits.append(audit)
        model_name = model_name or model
        previous_anchor = samples[0][1]
        previous_candidates = candidates
        previous_scene_id = str(scene["scene_id"])
    return views, plan, model_name, {
        "scenes": audits,
        "scene_count": len(scenes),
        "candidate_count": sum(int(item.get("candidate_count", 0)) for item in audits),
    }


def _review_surfaces_for_scene(
    scene: dict[str, Any],
    item: dict[str, Any],
    video_path: Path,
    used_surface_ids: set[str],
) -> list[dict[str, Any]]:
    """Validate one scene's complete human surface selection."""
    raw_surfaces = item.get("surfaces")
    if raw_surfaces is None:
        if item.get("quad") is None:
            raise ValueError(f"missing review surfaces for {scene['scene_id']}")
        raw_surfaces = [{"quad": item.get("quad")}]
    if not isinstance(raw_surfaces, list) or not 1 <= len(raw_surfaces) <= MAX_SURFACES_PER_SCENE:
        raise ValueError(f"review surfaces for {scene['scene_id']} must contain 1 to {MAX_SURFACES_PER_SCENE} items")

    frame = _frame_at(video_path, int(scene["start_frame"]))
    height, width = frame.shape[:2]
    normalized: list[dict[str, Any]] = []
    for index, raw_surface in enumerate(raw_surfaces, start=1):
        if not isinstance(raw_surface, dict):
            raise ValueError(f"review surface {index} for {scene['scene_id']} must be an object")
        surface_id = str(raw_surface.get("surface_id") or f"{scene['scene_id']}_surface_{index:03d}")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", surface_id) or surface_id in used_surface_ids:
            raise ValueError(f"review surface ID is invalid or duplicated: {surface_id}")
        points = parse_quad(raw_surface.get("quad", raw_surface.get("corners")))
        if (np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width) or
                np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height)):
            raise ValueError(f"review quad for {scene['scene_id']} lies outside its scene frame")
        normalized.append({
            "surface_id": surface_id,
            "quad": points,
            "status": "reviewed",
            "initialization_mode": "reviewed_candidate",
        })
        used_surface_ids.add(surface_id)
    for left_index, left in enumerate(normalized):
        for right in normalized[left_index + 1:]:
            overlap = _quad_overlap_ratio(left["quad"], right["quad"])
            if overlap >= SURFACE_OVERLAP_THRESHOLD:
                raise ValueError(f"review surfaces for {scene['scene_id']} overlap ({overlap:.3f})")
    return normalized


def _derive_asset_bindings(scene_plan: list[dict[str, Any]], branch_count: int) -> list[dict[str, Any]] | None:
    """Keep the single-surface legacy mapping for callers outside the worker."""
    counts = [len(scene.get("surfaces", [])) for scene in scene_plan]
    if not counts or max(counts, default=1) <= 1:
        return None
    # Multi-surface jobs must be explicit. Returning an all-surfaces mapping
    # here used to hide an ambiguous semantic choice from the operator.
    return None


def _process_job(job_id: str) -> None:
    job = _job(job_id)
    if job is None:
        return
    try:
        job["status"] = "running"
        job["phase"] = "sampling"
        _set_job(job)
        video_path, sign_paths, output_dir = _paths(job)
        reference_images = _read_brand_reference_images(sign_paths)
        detected_scenes = sample_video_scenes(video_path, max_scenes=MAX_SCENES)
        samples = detected_scenes[0]["samples"]
        sample_dir = output_dir / "samples"
        manual_quad = job.get("manual_quad")
        scene_plan: list[dict[str, Any]] | None = None
        scene_views: list[dict[str, Any]] = []
        model_name = job.get("model")
        candidate = job.get("candidate")
        validation = job.get("validation")
        if manual_quad is not None and not (job.get("scene_plan") and job.get("initialization_mode") == "reviewed_candidate"):
            quad = parse_quad(manual_quad)
            mode = job.get("initialization_mode", "manual_fallback")
            manual_scene_quads = job.get("manual_scene_quads")
            if manual_scene_quads is not None and len(manual_scene_quads) != len(detected_scenes):
                raise ValueError(
                    f"manual_scene_quads has {len(manual_scene_quads)} items, but video contains {len(detected_scenes)} scenes"
                )
            manual_candidates: list[dict[str, Any]] = []
            manual_validations: list[dict[str, Any]] = []
            for scene_index, scene in enumerate(detected_scenes):
                scene_quad = manual_scene_quads[scene_index] if manual_scene_quads else manual_quad
                scene_candidate = _manual_candidate(scene_quad)
                scene_validation = _validate_manual_candidate(scene, scene_candidate)
                manual_candidates.append(scene_candidate)
                manual_validations.append(scene_validation)
                scene_dir = sample_dir / scene["scene_id"]
                _save_contact_sheet(scene["samples"], scene_dir)
                scene_views.append(_scene_view(
                    scene,
                    f"/outputs/{job_id}/samples/{scene['scene_id']}/first.jpg",
                    scene_candidate,
                    scene_validation,
                    status="validated" if scene_validation.get("accepted") else "needs_review",
                    mode=mode,
                    candidates=[scene_candidate],
                    validations=[scene_validation],
                ))
                if scene_validation.get("accepted"):
                    scene_plan = scene_plan or []
                    scene_plan.append(_scene_plan_entry(scene, scene_candidate["quad"], mode=mode))
            if any(not item.get("accepted") for item in manual_validations):
                first_failed = next(item for item in manual_validations if not item.get("accepted"))
                job.update({
                    "status": "needs_review",
                    "phase": "human_review",
                    "scenes": scene_views,
                    "scene_plan": [],
                    "candidate": manual_candidates[0] if manual_candidates else None,
                    "candidates": manual_candidates,
                    "validation": first_failed,
                    "validations": manual_validations,
                    "error": "one or more manual scene calibrations require human review",
                    "sample_preview": f"/outputs/{job_id}/samples/{detected_scenes[0]['scene_id']}/first.jpg",
                    "contact_sheet_preview": f"/outputs/{job_id}/samples/{detected_scenes[0]['scene_id']}/contact_sheet.jpg",
                })
                _set_job(job)
                return
            candidate = manual_candidates[0] if manual_candidates else None
            validation = manual_validations[0] if manual_validations else None
            if len(detected_scenes) == 1:
                quad = parse_quad(candidate["quad"])
            else:
                # The tracker consumes the first scene's quad for its legacy
                # argument and uses the scene manifest for every later shot.
                quad = parse_quad(manual_candidates[0]["quad"])
            job.update({
                "candidate": candidate,
                "candidates": manual_candidates,
                "validation": validation,
                "validations": manual_validations,
            })
            if len(detected_scenes) == 1:
                _save_contact_sheet(samples, sample_dir)
        elif job.get("scene_plan") and job.get("initialization_mode") == "reviewed_candidate":
            # A human-reviewed scene manifest is durable and must be reused
            # exactly; re-running the model would change the user's decision.
            mode = "reviewed_candidate"
            quad = parse_quad(job["scene_plan"][0]["surfaces"][0]["quad"])
            scene_plan = job["scene_plan"]
            for scene in detected_scenes:
                scene_dir = sample_dir / scene["scene_id"]
                _save_contact_sheet(scene["samples"], scene_dir)
                matching = next((item for item in scene_plan if item["scene_id"] == scene["scene_id"]), None)
                scene_views.append(_scene_view(scene, f"/outputs/{job_id}/samples/{scene['scene_id']}/first.jpg", None, None, status="validated", mode=mode))
                if matching is None:
                    raise ValueError(f"reviewed scene plan is missing {scene['scene_id']}")
        else:
            mode = "vision_model"
            job["phase"] = "vision_candidates"
            _set_job(job)
            scene_views, scene_plan, model_name, vision_audit = _locate_scenes(
                video_path,
                detected_scenes,
                sample_dir,
                job.get("target_hint"),
                job_id,
                scene_reuse_policy=job.get("brand_profile_scene_reuse_policy", "vision_each_scene"),
                reference_images=reference_images,
            )
            unresolved = [scene for scene in scene_views if scene.get("status") != "validated"]
            if unresolved:
                first_unresolved = unresolved[0]
                job.update({
                    "status": "needs_review",
                    "phase": "human_review",
                    "scenes": scene_views,
                    "vision_audit": vision_audit,
                    "model": model_name,
                    "candidate": first_unresolved.get("candidate"),
                    "candidates": first_unresolved.get("candidates", []),
                    "validation": first_unresolved.get("validation"),
                    "validations": first_unresolved.get("validations", []),
                    "error": "one or more scene candidates require human review",
                    "sample_preview": f"/outputs/{job_id}/samples/{detected_scenes[0]['scene_id']}/first.jpg",
                    "contact_sheet_preview": f"/outputs/{job_id}/samples/{detected_scenes[0]['scene_id']}/contact_sheet.jpg",
                })
                _set_job(job)
                return
            candidate = scene_views[0].get("candidate") if scene_views else None
            validation = scene_views[0].get("validation") if scene_views else None
            quad = parse_quad(candidate["quad"])
            job.update({
                "candidate": candidate,
                "candidates": scene_views[0].get("candidates", []),
                "validation": validation,
                "validations": scene_views[0].get("validations", []),
                "model": model_name,
                "vision_audit": vision_audit,
            })

        job.update({
            "scenes": scene_views,
            "scene_plan": scene_plan or [],
            "sample_preview": f"/outputs/{job_id}/samples/{detected_scenes[0]['scene_id']}/first.jpg",
            "contact_sheet_preview": f"/outputs/{job_id}/samples/{detected_scenes[0]['scene_id']}/contact_sheet.jpg",
        })
        surface_ids = [surface["surface_id"] for scene in (scene_plan or []) for surface in scene.get("surfaces", [])]
        requires_explicit_bindings = any(len(scene.get("surfaces", [])) > 1 for scene in (scene_plan or []))
        profile_fit = job.get("brand_profile_default_fit")
        profile_surface_policy = job.get("brand_profile_surface_policy")
        auto_map_all_surfaces = profile_surface_policy == "all_surfaces"
        if (job.get("asset_bindings") is None and profile_fit in {"cover", "contain"} and surface_ids
                and (not requires_explicit_bindings or auto_map_all_surfaces)):
            job["asset_bindings"] = [
                {"branch_index": index, "surface_ids": surface_ids, "fit": profile_fit}
                for index in range(1, len(sign_paths) + 1)
            ]
        try:
            asset_bindings = _normalize_asset_bindings(
                job.get("asset_bindings"),
                len(sign_paths),
                surface_ids,
                require_explicit=requires_explicit_bindings,
            )
        except ValueError as exc:
            if requires_explicit_bindings and job.get("asset_bindings") is None:
                job.update({
                    "status": "needs_review",
                    "phase": "human_review",
                    "error": str(exc),
                    "asset_bindings_required": True,
                })
                _set_job(job)
                return
            raise
        job["asset_bindings"] = asset_bindings
        _set_job(job)

        job["phase"] = "tracking"
        _set_job(job)
        summary = run_tracker(
            video=video_path,
            quad=quad,
            signs=sign_paths,
            out_dir=output_dir,
            initialization_mode=mode,
            model_name=model_name,
            source_candidate=candidate,
            scene_plan=scene_plan or None,
            asset_bindings=asset_bindings,
            branch_labels=job.get("branch_labels"),
        )
        summary["scene_count"] = len(scene_plan) if scene_plan else len(detected_scenes)
        summary["surface_count"] = sum(len(scene["surfaces"]) for scene in scene_plan) if scene_plan else 1
        output_urls = [f"/outputs/{job_id}/{Path(path).name}" for path in summary.get("output_videos", [])]
        quality_gate = summary.get("quality_gate")
        if isinstance(quality_gate, dict) and not quality_gate.get("passed", False):
            gate_reason = quality_gate.get("reason") or "tracking quality requires review"
            job.update({
                "status": "needs_review",
                "phase": "quality_review",
                "summary": _public_summary(summary),
                "outputs": output_urls,
                "output_urls": output_urls,
                "initialization_mode": mode,
                "error": f"tracking quality gate requires review: {gate_reason}",
                "quality_gate": quality_gate,
            })
            _set_job(job)
            return
        job.update({"status": "completed", "phase": "complete", "summary": _public_summary(summary), "outputs": output_urls, "output_urls": output_urls, "initialization_mode": mode})
    except CandidateReviewRequired as exc:
        job.update({
            "status": "needs_review",
            "phase": "human_review",
            "candidate": exc.candidate,
            "candidates": exc.candidates,
            "validation": exc.validation,
            "validations": exc.validations,
            "model": exc.model,
            "vision_audit": exc.audit,
            "error": "candidate requires human review",
            "sample_preview": f"/outputs/{job_id}/samples/first.jpg",
            "contact_sheet_preview": f"/outputs/{job_id}/samples/contact_sheet.jpg",
        })
    except Exception as exc:
        message = str(exc)
        vision_error = message.startswith("vision_failed") or "vision gateway" in message.lower() or "candidate response" in message.lower()
        input_error = message.startswith("manual_scene_quads") or message.startswith("manual scene quad")
        job.update({"status": "vision_failed" if vision_error else "failed", "phase": "vision_failed" if vision_error else "failed", "error": message, "input_error": input_error})
    _set_job(job)


def _recover_jobs() -> None:
    """Resume queued/running jobs after a local process restart."""
    global _RECOVERY_STARTED
    _RECOVERY_STARTED = True
    _schedule_queued_jobs()


def _schedule_queued_jobs() -> None:
    """Fill available worker slots from durable queued/running job state."""
    for state_path in OUTPUT_DIR.glob("*/job.json"):
        if _active_slot_count() >= MAX_ACTIVE_JOBS:
            break
        try:
            job = load_job(OUTPUT_DIR, state_path.parent.name)
        except (ValueError, RuntimeError, OSError):
            continue
        if not job or job.get("status") not in {"queued", "running"}:
            continue
        if job.get("job_id") in FUTURES:
            continue
        if job.get("status") == "running":
            job["status"] = "queued"
            job["phase"] = "recovered_after_restart"
            _set_job(job)
        try:
            _submit_job(job["job_id"])
        except RuntimeError:
            break


def _batch_public(batch: dict[str, Any]) -> dict[str, Any]:
    """Return a batch manifest plus current public state for each child job."""
    job_ids = [str(item) for item in batch.get("job_ids", [])]
    jobs: list[dict[str, Any]] = []
    for job_id in job_ids:
        job = _job(job_id)
        if job is None:
            jobs.append({"job_id": job_id, "status": "missing"})
            continue
        jobs.append({
            "job_id": job_id,
            "status": job.get("status"),
            "phase": job.get("phase"),
            "branch_count": job.get("branch_count"),
            "branch_labels": job.get("branch_labels", []),
            "error": job.get("error"),
            "outputs": job.get("outputs", []),
            "summary": job.get("summary", {}),
        })
    statuses = [item.get("status") for item in jobs]
    if not statuses:
        status = "failed"
    elif all(item == "completed" for item in statuses):
        status = "completed"
    elif any(item in {"failed", "vision_failed", "missing"} for item in statuses):
        status = "partial_failed" if any(item == "completed" for item in statuses) else "failed"
    elif any(item == "needs_review" for item in statuses):
        status = "needs_review"
    elif any(item == "running" for item in statuses):
        status = "running"
    else:
        status = "queued"
    counts = {
        "total": len(statuses),
        "completed": sum(item == "completed" for item in statuses),
        "running": sum(item == "running" for item in statuses),
        "queued": sum(item == "queued" for item in statuses),
        "needs_review": sum(item == "needs_review" for item in statuses),
        "failed": sum(item in {"failed", "vision_failed", "missing"} for item in statuses),
    }
    return {
        "batch_id": batch.get("batch_id"),
        "status": status,
        "brand_profile_id": batch.get("brand_profile_id"),
        "requested_count": batch.get("requested_count", len(job_ids)),
        "counts": counts,
        "job_ids": job_ids,
        "jobs": jobs,
        "created_at": batch.get("created_at"),
        "updated_at": batch.get("updated_at"),
    }


@app.before_request
def _recover_jobs_for_wsgi() -> None:
    """Recover durable work when the module is loaded by a WSGI server."""
    global _RECOVERY_STARTED
    if app.config.get("TESTING") or _RECOVERY_STARTED:
        return
    _RECOVERY_STARTED = True
    _recover_jobs()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/brand-profiles", methods=["POST"])
def create_brand_profile_endpoint():
    """Persist reusable branch artwork without exposing filesystem paths."""
    if len(list_profiles(PROFILE_DIR)) >= MAX_PROFILE_COUNT:
        return jsonify({"error": f"brand profile limit reached ({MAX_PROFILE_COUNT})"}), 429
    brand_name = (request.form.get("brand_name") or "").strip()
    if not brand_name:
        return jsonify({"error": "brand_name is required"}), 400
    branch_files = _branch_files(request.files)
    if not 1 <= len(branch_files) <= MAX_BRANCHES:
        return jsonify({"error": f"1 to 10 branch image assets required, got {len(branch_files)}"}), 400
    try:
        labels = _branch_labels(request.form.get("branch_labels"), branch_files)
        profile_id = (request.form.get("profile_id") or f"brand_{uuid.uuid4().hex[:10]}").strip()
        default_fit = request.form.get("default_fit", "cover")
        surface_policy = request.form.get("surface_policy", "review")
        scene_reuse_policy = request.form.get("scene_reuse_policy", "vision_each_scene")
        with tempfile.TemporaryDirectory(prefix="franchiseflow-profile-") as temporary:
            sources: list[Path] = []
            for index, branch in enumerate(branch_files, start=1):
                filename = _source_filename(branch)
                suffix = Path(filename).suffix.lower()
                if suffix not in ALLOWED_IMAGE_EXTS:
                    raise ValueError(f"Branch image {filename} has unsupported extension")
                destination = Path(temporary) / f"branch_{index}{suffix}"
                if hasattr(branch, "save"):
                    branch.save(str(destination))
                else:
                    shutil.copyfile(branch, destination)
                decoded = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
                if decoded is None or decoded.size == 0:
                    raise ValueError(f"branch image {filename} could not be decoded")
                if decoded.shape[0] * decoded.shape[1] > MAX_IMAGE_PIXELS:
                    raise ValueError(f"branch image {filename} exceeds the {MAX_IMAGE_PIXELS} pixel limit")
                sources.append(destination)
            profile = create_profile(
                PROFILE_DIR,
                profile_id,
                brand_name=brand_name,
                target_hint=(request.form.get("target_hint") or None),
                branch_labels=labels,
                branch_sources=sources,
                default_fit=default_fit,
                surface_policy=surface_policy,
                scene_reuse_policy=scene_reuse_policy,
            )
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(profile), 201


@app.route("/api/brand-profiles", methods=["GET"])
def list_brand_profiles_endpoint():
    return jsonify({"profiles": list_profiles(PROFILE_DIR)})


@app.route("/api/brand-profiles/<profile_id>", methods=["GET"])
def get_brand_profile_endpoint(profile_id: str):
    try:
        profile = load_profile(PROFILE_DIR, profile_id)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    if profile is None:
        return jsonify({"error": "Brand profile not found"}), 404
    return jsonify(profile)


@app.route("/api/batches", methods=["POST"])
def create_batch_endpoint():
    """Submit several videos against one reusable brand profile.

    Every video remains an ordinary FranchiseFlow job, so review, retry and
    output download keep using the existing job endpoints. The batch record is
    only an index and never hides a failed child job.
    """
    profile_id = (request.form.get("brand_profile_id") or "").strip()
    if not profile_id:
        return jsonify({"error": "brand_profile_id is required for a batch"}), 400
    if _branch_files(request.files):
        return jsonify({"error": "batch requests use the brand profile assets; do not upload branches"}), 400
    videos = [item for item in request.files.getlist("videos") if item and item.filename]
    # Accept ``video`` as a one-item alias for clients that build fields
    # dynamically, while keeping ``videos`` the documented multi-file field.
    if not videos:
        videos = [item for item in request.files.getlist("video") if item and item.filename]
    if not 1 <= len(videos) <= MAX_BATCH_VIDEOS:
        return jsonify({"error": f"1 to {MAX_BATCH_VIDEOS} videos required, got {len(videos)}"}), 400
    try:
        profile = load_profile(PROFILE_DIR, profile_id)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    if profile is None:
        return jsonify({"error": f"brand profile not found: {profile_id}"}), 400
    for video in videos:
        if Path(video.filename).suffix.lower() not in ALLOWED_VIDEO_EXTS:
            allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTS))
            return jsonify({"error": f"Video {video.filename} must use a supported video extension: {allowed}"}), 400

    batch_id = uuid.uuid4().hex[:12]
    batch = create_batch(BATCH_DIR, batch_id, {
        "status": "queued",
        "brand_profile_id": profile_id,
        "requested_count": len(videos),
        "job_ids": [],
    })
    created_jobs: list[dict[str, Any]] = []
    try:
        for video in videos:
            created_jobs.append(_create_job_from_request(video_file_override=video, batch_id_override=batch_id))
    except (ValueError, RuntimeError) as exc:
        for job in created_jobs:
            job_id = job.get("job_id")
            if job_id:
                JOBS.pop(job_id, None)
                shutil.rmtree(UPLOAD_DIR / job_id, ignore_errors=True)
                shutil.rmtree(OUTPUT_DIR / job_id, ignore_errors=True)
        (BATCH_DIR / f"{batch_id}.json").unlink(missing_ok=True)
        return jsonify({"error": str(exc), "batch_id": batch_id}), 400

    batch["job_ids"] = [job["job_id"] for job in created_jobs]
    save_batch(BATCH_DIR, batch_id, batch)
    queued_count = 0
    for job in created_jobs:
        try:
            _submit_job(job["job_id"])
        except RuntimeError as exc:
            # A full worker pool is expected for a batch. The durable queued
            # state lets the normal scheduler pick this child up later.
            if "active job queue is full" in str(exc).lower():
                queued_count += 1
                continue
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = str(exc)
            _set_job(job)
    response = _batch_public(batch)
    response["queued_waiting_for_worker"] = queued_count
    return jsonify(response), 202


@app.route("/api/batches/<batch_id>", methods=["GET"])
def get_batch_endpoint(batch_id: str):
    try:
        batch = load_batch(BATCH_DIR, batch_id)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    if batch is None:
        return jsonify({"error": "Batch not found"}), 404
    return jsonify(_batch_public(batch))


@app.route("/api/batches", methods=["GET"])
def list_batches_endpoint():
    """List durable batch indexes so clients can recover after a restart."""
    raw_limit = request.args.get("limit", "20")
    try:
        limit = max(1, min(int(raw_limit), 100))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    batches: list[dict[str, Any]] = []
    for state_path in BATCH_DIR.glob("*.json"):
        batch_id = state_path.stem
        try:
            batch = load_batch(BATCH_DIR, batch_id)
        except (ValueError, RuntimeError, OSError):
            continue
        if batch is not None:
            batches.append(_batch_public(batch))
    batches.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return jsonify({"batches": batches[:limit]})


@app.route("/api/jobs", methods=["POST"])
def create_job_endpoint():
    if _active_slot_count() >= MAX_ACTIVE_JOBS:
        return jsonify({"error": f"active job queue is full ({MAX_ACTIVE_JOBS} jobs); try again later"}), 429
    job = None
    try:
        job = _create_job_from_request()
        _submit_job(job["job_id"])
        return jsonify(job), 202
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        if job is not None:
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = str(exc)
            _set_job(job)
        return jsonify({"error": str(exc)}), 429


@app.route("/api/jobs", methods=["GET"])
def list_jobs_endpoint():
    raw_limit = request.args.get("limit", "20")
    status = request.args.get("status") or None
    if status is not None and status not in {"queued", "running", "needs_review", "completed", "failed", "vision_failed"}:
        return jsonify({"error": "unsupported job status filter"}), 400
    try:
        limit = int(raw_limit)
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    return jsonify({"jobs": _list_jobs(limit=limit, status=status)})


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/jobs/<job_id>/review", methods=["POST"])
def review_job(job_id: str):
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "needs_review":
        return jsonify({"error": "Job is not waiting for human review"}), 409
    if _active_slot_count() >= MAX_ACTIVE_JOBS:
        return jsonify({"error": f"active job queue is full ({MAX_ACTIVE_JOBS} jobs); try again later"}), 429
    payload = request.get_json(silent=True) or {}
    requested_asset_bindings = payload.get("asset_bindings")
    if requested_asset_bindings is not None and not isinstance(requested_asset_bindings, list):
        return jsonify({"error": "asset_bindings must be a JSON array"}), 400
    video_path = _paths(job)[0]
    scene_payload = payload.get("scenes")
    reviewed_plan: list[dict[str, Any]] | None = None
    quad: np.ndarray | None = None
    try:
        if scene_payload is not None:
            if not isinstance(scene_payload, list) or not isinstance(job.get("scenes"), list) or len(scene_payload) != len(job["scenes"]):
                raise ValueError("review scenes must include every detected scene exactly once")
            if not scene_payload:
                raise ValueError("review scenes must contain at least one scene")
            by_id = {str(item.get("scene_id")): item for item in scene_payload if isinstance(item, dict)}
            if len(by_id) != len(scene_payload):
                raise ValueError("review scene IDs must be unique")
            reviewed_plan = []
            used_surface_ids: set[str] = set()
            for scene in job["scenes"]:
                scene_id = str(scene["scene_id"])
                item = by_id.get(scene_id)
                if item is None:
                    raise ValueError(f"missing review quad for {scene_id}")
                surfaces = _review_surfaces_for_scene(scene, item, video_path, used_surface_ids)
                reviewed_plan.append(_scene_plan_entry(scene, mode="reviewed_candidate", surfaces=surfaces))
            quad = parse_quad(reviewed_plan[0]["surfaces"][0]["quad"])
        else:
            first_frame = _first_frame(video_path)
            quad = parse_quad(payload.get("quad"))
            height, width = first_frame.shape[:2]
            if np.any(quad[:, 0] < 0) or np.any(quad[:, 0] >= width) or np.any(quad[:, 1] < 0) or np.any(quad[:, 1] >= height):
                raise ValueError("review quad lies outside the first frame")
    except (ValueError, TypeError, cv2.error) as exc:
        return jsonify({"error": f"Invalid review quad: {exc}"}), 400
    if reviewed_plan is not None:
        surface_ids = [surface["surface_id"] for scene in reviewed_plan for surface in scene.get("surfaces", [])]
        requires_explicit_bindings = any(len(scene.get("surfaces", [])) > 1 for scene in reviewed_plan)
        try:
            normalized_bindings = _normalize_asset_bindings(
                requested_asset_bindings,
                int(job.get("branch_count") or len(job.get("branch_files", []))),
                surface_ids,
                require_explicit=requires_explicit_bindings,
            )
        except ValueError as exc:
            return jsonify({"error": f"Invalid asset_bindings: {exc}"}), 400
    else:
        if requested_asset_bindings is not None:
            existing_plan = job.get("scene_plan") or []
            surface_ids = [surface["surface_id"] for scene in existing_plan for surface in scene.get("surfaces", [])]
            if not surface_ids:
                surface_ids = ["surface_001"]
            try:
                normalized_bindings = _normalize_asset_bindings(
                    requested_asset_bindings,
                    int(job.get("branch_count") or len(job.get("branch_files", []))),
                    surface_ids,
                    require_explicit=any(len(scene.get("surfaces", [])) > 1 for scene in existing_plan),
                )
            except ValueError as exc:
                return jsonify({"error": f"Invalid asset_bindings: {exc}"}), 400
        else:
            normalized_bindings = None
    previous = dict(job)
    job.update({"status": "queued", "phase": "queued", "initialization_mode": "reviewed_candidate", "manual_quad": quad.round(3).tolist(), "reviewed_quad": quad.round(3).tolist()})
    if normalized_bindings is not None:
        job["asset_bindings"] = normalized_bindings
    if reviewed_plan is not None:
        job["scene_plan"] = reviewed_plan
        job["scenes"] = [{**scene, "status": "validated", "initialization_mode": "reviewed_candidate", "surfaces": plan["surfaces"]}
                         for scene, plan in zip(job.get("scenes", []), reviewed_plan)]
    _set_job(job)
    try:
        _submit_job(job_id)
    except RuntimeError as exc:
        previous["status"] = "needs_review"
        previous["phase"] = "human_review"
        _set_job(previous)
        return jsonify({"error": str(exc)}), 429
    return jsonify(job), 202


@app.route("/api/jobs/<job_id>/outputs", methods=["GET"])
def get_job_outputs(job_id: str):
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "completed":
        return jsonify({"error": "Job is not completed", "status": job.get("status")}), 409
    return jsonify({"job_id": job_id, "status": job["status"], "outputs": job.get("outputs", []), "summary": job.get("summary", {})})


@app.route("/api/jobs/<job_id>/quality-review", methods=["POST"])
def quality_review_job(job_id: str):
    """Accept or reject a rendered batch after the tracking quality gate."""
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "needs_review" or job.get("phase") != "quality_review":
        return jsonify({"error": "Job is not waiting for output quality review"}), 409
    payload = request.get_json(silent=True) or {}
    decision = payload.get("decision")
    if decision not in {"accept", "reject"}:
        return jsonify({"error": "decision must be accept or reject"}), 400
    if decision == "accept":
        job.update({
            "status": "completed",
            "phase": "complete",
            "quality_review": {"decision": "accepted", "failed_fraction": (job.get("quality_gate") or {}).get("failed_fraction")},
            "error": None,
        })
    else:
        job.update({
            "status": "failed",
            "phase": "quality_rejected",
            "quality_review": {"decision": "rejected", "failed_fraction": (job.get("quality_gate") or {}).get("failed_fraction")},
            "error": "rendered outputs were rejected during quality review",
        })
    _set_job(job)
    return jsonify(job)


@app.route("/api/render", methods=["POST"])
def render_job():
    if not _reserve_sync_slot():
        return jsonify({"error": f"active job queue is full ({MAX_ACTIVE_JOBS} jobs); try again later"}), 429
    try:
        job = _create_job_from_request()
        _process_job(job["job_id"])
        job = _job(job["job_id"]) or job
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        _release_sync_slot()
    if job.get("status") != "completed":
        code = 400 if job.get("input_error") else (502 if "HTTP 401" in job.get("error", "") or "HTTP 403" in job.get("error", "") else 500)
        return jsonify({
            "error": job.get("error", "render failed"),
            "status": job.get("status"),
            "job_id": job.get("job_id"),
            "review_url": f"/api/jobs/{job['job_id']}/review" if job.get("status") == "needs_review" else None,
        }), code
    return jsonify({
        "job_id": job["job_id"],
        "status": job["status"],
        "vision_quad": job.get("candidate", {}).get("quad") if job.get("candidate") else job.get("manual_quad"),
        "initialization_mode": job.get("initialization_mode"),
        "summary": job.get("summary", {}),
        "vision_audit": job.get("vision_audit"),
        "sample_preview": job.get("sample_preview"),
        "contact_sheet_preview": job.get("contact_sheet_preview"),
        "manifest_url": f"/outputs/{job['job_id']}/manifest.json",
        "outputs": job.get("outputs", []),
    })


@app.route("/api/demo_results", methods=["GET"])
def get_demo_results():
    demo_dir = REPO_ROOT / "spikes" / "franchiseflow" / "outputs_demo_h264"
    if not demo_dir.exists():
        return jsonify({"error": "Demo outputs not ready"}), 404
    summary_path = demo_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary = _public_summary(summary)
    # This checked-in sample predates the current initialization enum and was
    # produced from an explicit manual quad.  Keep the sample usable while
    # reporting its provenance honestly to the UI and API consumers.
    summary["initialization_mode"] = "manual_fallback"
    summary["legacy_demo"] = True
    summary.setdefault("asset_bindings", [
        {"branch_index": index, "surface_ids": ["surface_001"], "fit": "cover"}
        for index, _ in enumerate(sorted(demo_dir.glob("branch_*.mp4")), start=1)
    ])
    job_id = "prebuilt_demo"
    target_job_dir = OUTPUT_DIR / job_id
    target_job_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for path in demo_dir.glob("branch_*.mp4"):
        shutil.copy2(path, target_job_dir / path.name)
    outputs = [f"/outputs/{job_id}/{path.name}" for path in sorted(target_job_dir.glob("branch_*.mp4"))]
    job_data = {
        "job_id": job_id,
        "status": "completed",
        "vision_quad": summary.get("first_frame_quad"),
        "initialization_mode": "manual_fallback",
        "summary": summary,
        "outputs": outputs,
    }
    JOBS[job_id] = job_data
    return jsonify(job_data)


@app.route("/outputs/<job_id>/<path:filename>")
def serve_job_output(job_id: str, filename: str):
    if not JOB_ID_RE.fullmatch(job_id or ""):
        return jsonify({"error": "Not found"}), 404
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(str(job_dir), filename)


if __name__ == "__main__":
    _recover_jobs()
    app.run(host="127.0.0.1", port=5000, debug=False)
