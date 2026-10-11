from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# Support imports from spikes/franchiseflow
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "spikes" / "franchiseflow"))

from tracker import run_tracker
from vision_client import VisionClient

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTS = {".mp4"}
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

JOBS: dict[str, dict[str, Any]] = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/render", methods=["POST"])
def render_job():
    if "video" not in request.files:
        return jsonify({"error": "Missing video file"}), 400

    video_file = request.files["video"]
    if not video_file.filename or Path(video_file.filename).suffix.lower() not in ALLOWED_VIDEO_EXTS:
        return jsonify({"error": "Video must be an MP4 file"}), 400

    # Collect branch assets
    branch_files = []
    if "branches" in request.files:
        branch_files = request.files.getlist("branches")
    else:
        for key in ["branch_1", "branch_2", "branch_3"]:
            if key in request.files and request.files[key].filename:
                branch_files.append(request.files[key])

    if len(branch_files) != 3:
        return jsonify({"error": f"Exactly 3 branch image assets required, got {len(branch_files)}"}), 400

    for bf in branch_files:
        if Path(bf.filename).suffix.lower() not in ALLOWED_IMAGE_EXTS:
            return jsonify({"error": f"Branch image {bf.filename} has unsupported extension"}), 400

    job_id = uuid.uuid4().hex[:8]
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_out_dir = OUTPUT_DIR / job_id
    job_out_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    video_path = job_upload_dir / secure_filename(video_file.filename)
    video_file.save(str(video_path))

    saved_branch_paths: list[Path] = []
    for i, bf in enumerate(branch_files):
        ext = Path(bf.filename).suffix.lower()
        b_path = job_upload_dir / f"branch_{i+1}{ext}"
        bf.save(str(b_path))
        saved_branch_paths.append(b_path)

    # Check first frame
    cap = cv2.VideoCapture(str(video_path))
    ok, first_frame = cap.read()
    cap.release()
    if not ok or first_frame is None:
        return jsonify({"error": "Unable to decode first frame from uploaded video"}), 400

    height, width = first_frame.shape[:2]

    use_manual_quad = request.form.get("use_manual_quad", "false").lower() in ("true", "1", "yes")
    initialization_mode = "manual_fallback" if use_manual_quad else "vision_model"

    if use_manual_quad:
        # Check if manual coordinates were supplied in form
        manual_raw = request.form.get("manual_quad")
        if manual_raw:
            try:
                quad = json.loads(manual_raw)
            except Exception as e:
                return jsonify({"error": f"Invalid manual quad JSON: {e}"}), 400
        else:
            default_quad_path = REPO_ROOT / "spikes" / "franchiseflow" / "quad.json"
            if default_quad_path.exists():
                quad = json.loads(default_quad_path.read_text(encoding="utf-8"))
            else:
                quad = [[220.0, 55.0], [900.0, 300.0], [910.0, 410.0], [220.0, 165.0]]
    else:
        # Live Vision API call
        ok, enc = cv2.imencode(".jpg", first_frame)
        if not ok:
            return jsonify({"error": "Failed to encode frame for vision API"}), 500
        try:
            client = VisionClient()
            result = client.locate_sign(enc.tobytes(), width, height)
            quad = result["quad"]
        except Exception as exc:
            return jsonify({"error": f"Vision API error: {exc}", "status": "failed"}), 502

    illumination = request.form.get("illumination", "false").lower() in ("true", "1", "yes")

    try:
        summary = run_tracker(
            video=video_path,
            quad=quad,
            signs=saved_branch_paths,
            out_dir=job_out_dir,
            illumination=illumination,
            initialization_mode=initialization_mode,
        )
    except Exception as exc:
        return jsonify({"error": f"Tracking error: {exc}", "status": "failed"}), 500

    # Build output URLs
    output_urls = [
        f"/outputs/{job_id}/{p.name}"
        for p in [job_out_dir / f"branch_{i+1}_{p.stem}.mp4" for i, p in enumerate(saved_branch_paths)]
    ]

    job_data = {
        "job_id": job_id,
        "status": "completed",
        "vision_quad": quad,
        "initialization_mode": initialization_mode,
        "summary": summary,
        "outputs": output_urls,
    }
    JOBS[job_id] = job_data
    return jsonify(job_data)


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(JOBS[job_id])


@app.route("/api/demo_results", methods=["GET"])
def get_demo_results():
    demo_dir = REPO_ROOT / "spikes" / "franchiseflow" / "outputs_demo_h264"
    if not demo_dir.exists():
        return jsonify({"error": "Demo outputs not ready"}), 404
    summary_path = demo_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    job_id = "prebuilt_demo"
    target_job_dir = OUTPUT_DIR / job_id
    target_job_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for p in demo_dir.glob("branch_*.mp4"):
        shutil.copy2(p, target_job_dir / p.name)
    outputs = [f"/outputs/{job_id}/{p.name}" for p in sorted(target_job_dir.glob("branch_*.mp4"))]
    job_data = {
        "job_id": job_id,
        "status": "completed",
        "vision_quad": summary.get("first_frame_quad", [[220.0, 55.0], [900.0, 300.0], [910.0, 410.0], [220.0, 165.0]]),
        "initialization_mode": "manual_fallback",
        "summary": summary,
        "outputs": outputs,
    }
    JOBS[job_id] = job_data
    return jsonify(job_data)


@app.route("/outputs/<job_id>/<filename>")
def serve_job_output(job_id: str, filename: str):
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(str(job_dir), filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
