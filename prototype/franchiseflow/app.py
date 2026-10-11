from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import sys
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
from tracker import parse_quad, run_tracker
from scene_samples import sample_video_scenes
from vision_client import VisionClient

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTS = {".mp4"}
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MAX_BRANCHES = 10
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
MAX_VIDEO_SECONDS = 15.0
MAX_ACTIVE_JOBS = 4
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_IMAGE_PIXELS = 20_000_000

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
        "status": job.get("status"),
        "phase": job.get("phase"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "video_file": job.get("video_file"),
        "branch_files": job.get("branch_files", []),
        "branch_count": job.get("branch_count"),
        "initialization_mode": job.get("initialization_mode"),
        "model": job.get("model"),
        "target_hint": job.get("target_hint"),
        "candidate": job.get("candidate"),
        "reviewed_quad": job.get("reviewed_quad"),
        "vision_audit": job.get("vision_audit"),
        "scenes": job.get("scenes", []),
        "sample_preview": job.get("sample_preview"),
        "contact_sheet_preview": job.get("contact_sheet_preview"),
        "scene_plan": job.get("scene_plan", []),
        "outputs": job.get("outputs", []),
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


def _create_job_from_request() -> dict[str, Any]:
    video_file = request.files.get("video")
    if video_file is None or not video_file.filename:
        raise ValueError("Missing video file")
    if Path(video_file.filename).suffix.lower() not in ALLOWED_VIDEO_EXTS:
        raise ValueError("Video must be an MP4 file")

    branch_files = _branch_files(request.files)
    if not 1 <= len(branch_files) <= MAX_BRANCHES:
        raise ValueError(f"1 to 10 branch image assets required, got {len(branch_files)}")
    for branch in branch_files:
        if Path(branch.filename).suffix.lower() not in ALLOWED_IMAGE_EXTS:
            raise ValueError(f"Branch image {branch.filename} has unsupported extension")

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
            suffix = Path(branch.filename).suffix.lower()
            name = f"branch_{index}{suffix}"
            branch.save(str(upload_dir / name))
            branch_names.append(name)

        video_path = upload_dir / video_name
        branch_paths = [upload_dir / name for name in branch_names]
        total_bytes = video_path.stat().st_size + sum(path.stat().st_size for path in branch_paths)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise ValueError(f"upload size exceeds the {MAX_UPLOAD_BYTES} byte limit")
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if fps <= 0 or frame_count <= 0:
            raise ValueError("unable to read uploaded video duration")
        duration = frame_count / fps
        if duration > MAX_VIDEO_SECONDS:
            raise ValueError(f"video duration {duration:.2f}s exceeds the {MAX_VIDEO_SECONDS:.2f}s limit")
        first_frame = _first_frame(video_path)

        for path in branch_paths:
            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.size == 0:
                raise ValueError(f"branch image {path.name} could not be decoded")
            if decoded.shape[0] * decoded.shape[1] > MAX_IMAGE_PIXELS:
                raise ValueError(f"branch image {path.name} exceeds the {MAX_IMAGE_PIXELS} pixel limit")

        manual = request.form.get("use_manual_quad", "false").lower() in {"true", "1", "yes"}
        manual_quad = None
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
    except ValueError:
        shutil.rmtree(upload_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    job = create_job(OUTPUT_DIR, job_id, {
        "status": "queued",
        "phase": "queued",
        "video_file": video_name,
        "branch_files": branch_names,
        "branch_count": len(branch_names),
        "initialization_mode": "manual_fallback" if manual else "vision_model",
        "manual_quad": manual_quad,
        "target_hint": request.form.get("target_hint") or None,
        "output_urls": [],
    })
    JOBS[job_id] = job
    return job


def _paths(job: dict[str, Any]) -> tuple[Path, list[Path], Path]:
    job_id = job["job_id"]
    upload_dir = UPLOAD_DIR / job_id
    return upload_dir / job["video_file"], [upload_dir / name for name in job["branch_files"]], OUTPUT_DIR / job_id


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


def _locate_and_validate(video_path: Path, samples: list[tuple[int, np.ndarray]], sample_dir: Path, target_hint: str | None) -> tuple[dict[str, Any] | None, dict[str, Any], str, dict[str, Any]]:
    contact_sheet, panel_width, panel_height, scale = _save_contact_sheet(samples, sample_dir)
    first_height, first_width = samples[0][1].shape[:2]
    client = VisionClient()
    panel_count = len(samples)
    sheet_width = panel_width * panel_count
    result = client.locate_candidates(contact_sheet, sheet_width, panel_height, target_hint=target_hint)
    # Keep old mocked clients and old gateways usable during migration.
    if not isinstance(result, dict) or not isinstance(result.get("candidates"), list):
        encoded_first = cv2.imencode(".jpg", samples[0][1])[1].tobytes()
        legacy = client.locate_sign(encoded_first, first_width, first_height, target_hint=target_hint)
        result = {"model": legacy.get("model", client.model), "candidates": [legacy]}

    audit = {
        "input_sha256": hashlib.sha256(contact_sheet).hexdigest(),
        "candidate_count": len(result["candidates"]),
        "model": str(result.get("model", client.model)),
    }

    rejected: list[dict[str, Any]] = []
    for raw_candidate in result["candidates"]:
        try:
            candidate = _normalise_candidate(raw_candidate, scale, panel_width=panel_width, panel_count=panel_count)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"vision_failed: invalid contact-sheet candidate: {exc}") from exc
        validation = validate_candidate(samples, candidate)
        if validation["accepted"]:
            audit["accepted_candidate_confidence"] = candidate.get("confidence")
            audit["validation_reasons"] = validation.get("reasons", [])
            return candidate, validation, str(result.get("model", client.model)), audit
        rejected.append({"candidate": candidate, "validation": validation})

    if rejected:
        best = max(rejected, key=lambda item: float(item["candidate"].get("confidence", 0.0)))
        audit["validation_reasons"] = best["validation"].get("reasons", [])
        raise CandidateReviewRequired(best["candidate"], best["validation"], str(result.get("model", client.model)), audit)
    raise RuntimeError("vision_failed: the model returned no valid candidates")


class CandidateReviewRequired(RuntimeError):
    def __init__(self, candidate: dict[str, Any], validation: dict[str, Any], model: str, audit: dict[str, Any]):
        super().__init__("candidate requires human review")
        self.candidate = candidate
        self.validation = validation
        self.model = model
        self.audit = audit


def _scene_plan_entry(scene: dict[str, Any], quad: list | np.ndarray, *, mode: str, status: str = "validated", surface_id: str | None = None) -> dict[str, Any]:
    """Build the stable manifest consumed by the scene-aware tracker."""
    points = parse_quad(quad).round(3).tolist()
    scene_id = str(scene["scene_id"])
    return {
        "scene_id": scene_id,
        "start_frame": int(scene["start_frame"]),
        "end_frame": int(scene["end_frame"]),
        "cut_before": scene.get("cut_before"),
        "cut_score": float(scene.get("cut_score", 0.0)),
        "surfaces": [{
            "surface_id": surface_id or f"{scene_id}_surface_001",
            "quad": points,
            "initialization_mode": mode,
            "status": status,
        }],
    }


def _scene_view(scene: dict[str, Any], sample_preview: str, candidate: dict[str, Any] | None = None, validation: dict[str, Any] | None = None, *, status: str = "validated", mode: str = "vision_model") -> dict[str, Any]:
    """Return JSON-only scene state for the API and browser review panel."""
    result = {
        "scene_id": scene["scene_id"],
        "start_frame": int(scene["start_frame"]),
        "end_frame": int(scene["end_frame"]),
        "cut_before": scene.get("cut_before"),
        "cut_score": float(scene.get("cut_score", 0.0)),
        "status": status,
        "initialization_mode": mode,
        "preview": sample_preview,
        "candidate": candidate,
        "validation": validation,
        "surfaces": [],
    }
    if candidate and candidate.get("quad"):
        result["surfaces"] = [{
            "surface_id": f"{scene['scene_id']}_surface_001",
            "quad": candidate["quad"],
            "status": status,
            "initialization_mode": mode,
        }]
    return result


def _locate_scenes(video_path: Path, scenes: list[dict[str, Any]], sample_dir: Path, target_hint: str | None, job_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, dict[str, Any]]:
    """Locate one primary planar surface independently in each detected scene."""
    views: list[dict[str, Any]] = []
    plan: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    model_name: str | None = None
    for scene in scenes:
        scene_dir = sample_dir / scene["scene_id"]
        samples = scene["samples"]
        try:
            candidate, validation, model, audit = _locate_and_validate(video_path, samples, scene_dir, target_hint)
        except CandidateReviewRequired as exc:
            views.append(_scene_view(scene, f"/outputs/{job_id}/samples/{scene['scene_id']}/first.jpg", exc.candidate, exc.validation, status="needs_review", mode="vision_model"))
            audits.append(exc.audit)
            model_name = model_name or exc.model
            continue
        views.append(_scene_view(scene, f"/outputs/{job_id}/samples/{scene['scene_id']}/first.jpg", candidate, validation, status="validated", mode="vision_model"))
        plan.append(_scene_plan_entry(scene, candidate["quad"], mode="vision_model"))
        audits.append(audit)
        model_name = model_name or model
    return views, plan, model_name, {
        "scenes": audits,
        "scene_count": len(scenes),
        "candidate_count": sum(int(item.get("candidate_count", 0)) for item in audits),
    }


def _process_job(job_id: str) -> None:
    job = _job(job_id)
    if job is None:
        return
    try:
        job["status"] = "running"
        job["phase"] = "sampling"
        _set_job(job)
        video_path, sign_paths, output_dir = _paths(job)
        detected_scenes = sample_video_scenes(video_path, max_scenes=2)
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
            for scene in detected_scenes:
                scene_dir = sample_dir / scene["scene_id"]
                _save_contact_sheet(scene["samples"], scene_dir)
                scene_views.append(_scene_view(
                    scene,
                    f"/outputs/{job_id}/samples/{scene['scene_id']}/first.jpg",
                    candidate if len(detected_scenes) == 1 else None,
                    validation,
                    status="validated",
                    mode=mode,
                ))
                scene_plan = scene_plan or []
                scene_plan.append(_scene_plan_entry(scene, quad, mode=mode))
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
            scene_views, scene_plan, model_name, vision_audit = _locate_scenes(video_path, detected_scenes, sample_dir, job.get("target_hint"), job_id)
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
                    "validation": first_unresolved.get("validation"),
                    "error": "one or more scene candidates require human review",
                    "sample_preview": f"/outputs/{job_id}/samples/{detected_scenes[0]['scene_id']}/first.jpg",
                    "contact_sheet_preview": f"/outputs/{job_id}/samples/{detected_scenes[0]['scene_id']}/contact_sheet.jpg",
                })
                _set_job(job)
                return
            candidate = scene_views[0].get("candidate") if scene_views else None
            validation = scene_views[0].get("validation") if scene_views else None
            quad = parse_quad(candidate["quad"])
            job.update({"candidate": candidate, "validation": validation, "model": model_name, "vision_audit": vision_audit})

        job.update({
            "scenes": scene_views,
            "scene_plan": scene_plan or [],
            "sample_preview": f"/outputs/{job_id}/samples/{detected_scenes[0]['scene_id']}/first.jpg",
            "contact_sheet_preview": f"/outputs/{job_id}/samples/{detected_scenes[0]['scene_id']}/contact_sheet.jpg",
        })
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
        )
        summary["scene_count"] = len(scene_plan) if scene_plan else len(detected_scenes)
        summary["surface_count"] = sum(len(scene["surfaces"]) for scene in scene_plan) if scene_plan else 1
        output_urls = [f"/outputs/{job_id}/{Path(path).name}" for path in summary.get("output_videos", [])]
        job.update({"status": "completed", "phase": "complete", "summary": _public_summary(summary), "outputs": output_urls, "output_urls": output_urls, "initialization_mode": mode})
    except CandidateReviewRequired as exc:
        job.update({"status": "needs_review", "phase": "human_review", "candidate": exc.candidate, "validation": exc.validation, "model": exc.model, "vision_audit": exc.audit, "error": "candidate requires human review", "sample_preview": f"/outputs/{job_id}/samples/first.jpg", "contact_sheet_preview": f"/outputs/{job_id}/samples/contact_sheet.jpg"})
    except Exception as exc:
        message = str(exc)
        vision_error = message.startswith("vision_failed") or "vision gateway" in message.lower() or "candidate response" in message.lower()
        job.update({"status": "vision_failed" if vision_error else "failed", "phase": "vision_failed" if vision_error else "failed", "error": message})
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
    video_path = UPLOAD_DIR / job_id / job["video_file"]
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
            for scene in job["scenes"]:
                scene_id = str(scene["scene_id"])
                item = by_id.get(scene_id)
                if item is None:
                    raise ValueError(f"missing review quad for {scene_id}")
                points = parse_quad(item.get("quad"))
                frame = _frame_at(video_path, int(scene["start_frame"]))
                height, width = frame.shape[:2]
                if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
                    raise ValueError(f"review quad for {scene_id} lies outside its scene frame")
                reviewed_plan.append(_scene_plan_entry(scene, points, mode="reviewed_candidate", status="reviewed"))
            quad = parse_quad(reviewed_plan[0]["surfaces"][0]["quad"])
        else:
            first_frame = _first_frame(video_path)
            quad = parse_quad(payload.get("quad"))
            height, width = first_frame.shape[:2]
            if np.any(quad[:, 0] < 0) or np.any(quad[:, 0] >= width) or np.any(quad[:, 1] < 0) or np.any(quad[:, 1] >= height):
                raise ValueError("review quad lies outside the first frame")
    except (ValueError, TypeError, cv2.error) as exc:
        return jsonify({"error": f"Invalid review quad: {exc}"}), 400
    previous = dict(job)
    job.update({"status": "queued", "phase": "queued", "initialization_mode": "reviewed_candidate", "manual_quad": quad.round(3).tolist(), "reviewed_quad": quad.round(3).tolist()})
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
        code = 502 if "HTTP 401" in job.get("error", "") or "HTTP 403" in job.get("error", "") else 500
        return jsonify({"error": job.get("error", "render failed"), "status": job.get("status")}), code
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
        "initialization_mode": summary.get("initialization_mode", "manual_fallback"),
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
