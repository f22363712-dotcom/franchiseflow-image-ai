import io
import json
import shutil
import sys
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

# Add prototype/franchiseflow and spikes/franchiseflow to path
TEST_DIR = Path(__file__).resolve().parent
PROTOTYPE_DIR = TEST_DIR.parent
REPO_ROOT = PROTOTYPE_DIR.parents[1]
sys.path.insert(0, str(PROTOTYPE_DIR))
sys.path.insert(0, str(REPO_ROOT / "spikes" / "franchiseflow"))


from app import app, JOBS, UPLOAD_DIR, OUTPUT_DIR, PROFILE_DIR, BATCH_DIR, _locate_scenes, _normalise_candidate, _public_summary
from job_store import create_job, load_job, save_job
from batch_store import create_batch
from scene_samples import sample_video_scenes


def _make_dummy_mp4():
    buf = io.BytesIO()
    # Create a tiny 2-frame 320x240 MP4
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, 10, (320, 240))
    frame = np.full((240, 320, 3), 100, dtype=np.uint8)
    writer.write(frame)
    writer.write(frame)
    writer.release()
    data = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)
    return io.BytesIO(data)


def _make_two_scene_mp4():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    first = np.full((240, 320, 3), (20, 50, 100), dtype=np.uint8)
    second = np.full((240, 320, 3), (180, 220, 30), dtype=np.uint8)
    for frame in (first, first, second, second):
        writer.write(frame)
    writer.release()
    data = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)
    return io.BytesIO(data)


def _make_dummy_image(text="sign"):
    img = np.full((100, 200, 3), 120, dtype=np.uint8)
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buf = cv2.imencode(".png", img)
    return io.BytesIO(buf.tobytes())


class FranchiseFlowAppTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        JOBS.clear()

    def test_index_page_returns_200(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"FranchiseFlow", res.data)
        self.assertIn(b"/api/jobs", res.data)
        self.assertIn(b"needs_review", res.data)
        self.assertIn(b"review-quad", res.data)
        self.assertIn(b"brand-profile-options", res.data)
        self.assertIn(b"btn-save-brand-profile", res.data)


    def test_missing_video_returns_400(self):
        data = {
            "branch_1": (_make_dummy_image("b1"), "b1.png"),
            "branch_2": (_make_dummy_image("b2"), "b2.png"),
            "branch_3": (_make_dummy_image("b3"), "b3.png"),
        }
        res = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_empty_branches_returns_400(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
        }
        res = self.client.post("/api/jobs", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_unsupported_extension_returns_400(self):
        data = {
            "video": (io.BytesIO(b"fake txt"), "video.txt"),
            "branch_1": (_make_dummy_image("b1"), "b1.png"),
            "branch_2": (_make_dummy_image("b2"), "b2.png"),
            "branch_3": (_make_dummy_image("b3"), "b3.png"),
        }
        res = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_vision_api_failure_returns_error_without_fake_result(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branch_1": (_make_dummy_image("b1"), "b1.png"),
            "branch_2": (_make_dummy_image("b2"), "b2.png"),
            "branch_3": (_make_dummy_image("b3"), "b3.png"),
        }
        mock_instance = MagicMock()
        mock_instance.locate_sign.side_effect = RuntimeError("vision gateway returned HTTP 403; check base URL, key, and gateway access")
        with patch("app.VisionClient", return_value=mock_instance):
            res = self.client.post("/api/render", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 502)
            json_data = res.get_json()
            self.assertIn("error", json_data)
            self.assertIn("HTTP 403", json_data["error"])
            self.assertEqual(len(JOBS), 1)
            self.assertEqual(next(iter(JOBS.values()))["status"], "vision_failed")

    def test_successful_render_with_mocked_pipeline(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branch_1": (_make_dummy_image("b1"), "b1.png"),
            "branch_2": (_make_dummy_image("b2"), "b2.png"),
            "branch_3": (_make_dummy_image("b3"), "b3.png"),
        }
        mock_quad = [[20.0, 20.0], [100.0, 20.0], [100.0, 80.0], [20.0, 80.0]]
        mock_summary = {
            "frames": 2,
            "fps": 10.0,
            "failed_frames": [],
            "processing_seconds": 0.5,
            "output_videos": ["out1.mp4", "out2.mp4", "out3.mp4"],
        }
        mock_instance = MagicMock()
        mock_instance.locate_sign.return_value = {"quad": mock_quad, "model": "gemini-3.1-flash-image"}

        with patch("app.VisionClient", return_value=mock_instance):
            with patch("app.run_tracker", return_value=mock_summary):
                with patch("app.validate_candidate", return_value={"accepted": True, "requires_human_review": False, "reasons": [], "sample_metrics": []}):
                    res = self.client.post("/api/render", data=data, content_type="multipart/form-data")
                self.assertEqual(res.status_code, 200)
                json_data = res.get_json()
                self.assertEqual(json_data["status"], "completed")
                self.assertEqual(json_data["summary"]["scene_count"], 1)
                self.assertEqual(json_data["vision_quad"], mock_quad)
                self.assertEqual(len(json_data["outputs"]), 3)
                self.assertIn("vision_audit", json_data)
                self.assertEqual(json_data["vision_audit"]["candidate_count"], 1)
                self.assertTrue((OUTPUT_DIR / json_data["job_id"] / "manifest.json").exists())
                job_id = json_data["job_id"]

                # Check GET /api/jobs/<job_id>
                get_res = self.client.get(f"/api/jobs/{job_id}")
                self.assertEqual(get_res.status_code, 200)
                self.assertEqual(get_res.get_json()["status"], "completed")

    def test_rejects_video_over_duration_cap(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app.MAX_VIDEO_SECONDS", 0.01):
            res = self.client.post("/api/jobs", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn("duration", res.get_json()["error"])

    def test_rejects_upload_over_size_cap(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app.MAX_UPLOAD_BYTES", 1):
            res = self.client.post("/api/jobs", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn("size", res.get_json()["error"])

    def test_frontend_persists_job_and_review_preview(self):
        app_js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("localStorage", app_js)
        self.assertIn("sample_preview", app_js)
        self.assertIn("redrawReview", app_js)
        self.assertIn("quality-review", app_js)
        self.assertIn("brand_profile_id", app_js)
        self.assertIn("/api/brand-profiles", app_js)
        self.assertIn("brand-profile-options", app_js)
        self.assertIn("btn-save-brand-profile", app_js)
        self.assertIn("saveCurrentBrandProfile", app_js)

    def test_frontend_exposes_batch_production_controls(self):
        html = (Path(__file__).resolve().parents[1] / "templates" / "index.html").read_text(encoding="utf-8")
        app_js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(encoding="utf-8")
        css = (Path(__file__).resolve().parents[1] / "static" / "app.css").read_text(encoding="utf-8")
        for marker in (
            "batch-form",
            "batch-profile-select",
            "batch-video-input",
            "batch-results",
            "batch-jobs",
        ):
            self.assertIn(marker, html)
        for marker in (
            "POST",
            "/api/batches",
            "pollBatch",
            "/api/batches/${encodeURIComponent(batchId)}",
            "/api/batches/${encodeURIComponent(batch.batch_id)}/download",
            "/api/jobs/${encodeURIComponent(jobId)}/retry",
            "重试此视频",
            "打开人工复核",
            "franchiseflow_batch_id",
        ):
            self.assertIn(marker, app_js)
        for marker in ("batch-section", "batch-controls", "batch-job-card", "status-needs_review"):
            self.assertIn(marker, css)

    def test_quality_review_can_accept_or_reject_rendered_batch(self):
        accepted_id = f"quality_accept_test_{uuid.uuid4().hex[:6]}"
        create_job(OUTPUT_DIR, accepted_id, {
            "status": "needs_review",
            "phase": "quality_review",
            "quality_gate": {"passed": False, "failed_frames": 2, "failed_fraction": 0.4, "max_failed_fraction": 0.25},
            "outputs": ["/outputs/quality_accept_test/branch_1.mp4"],
            "summary": {"quality_gate": {"passed": False}},
        })
        accepted = self.client.post(f"/api/jobs/{accepted_id}/quality-review", json={"decision": "accept"})
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.get_json()["status"], "completed")
        self.assertEqual(accepted.get_json()["quality_review"]["decision"], "accepted")

        rejected_id = f"quality_reject_test_{uuid.uuid4().hex[:6]}"
        create_job(OUTPUT_DIR, rejected_id, {
            "status": "needs_review",
            "phase": "quality_review",
            "quality_gate": {"passed": False, "failed_frames": 3, "failed_fraction": 0.6, "max_failed_fraction": 0.25},
            "outputs": [],
        })
        rejected = self.client.post(f"/api/jobs/{rejected_id}/quality-review", json={"decision": "reject"})
        self.assertEqual(rejected.status_code, 200)
        self.assertEqual(rejected.get_json()["status"], "failed")

    def test_quality_review_rejects_invalid_decision(self):
        job_id = f"quality_invalid_test_{uuid.uuid4().hex[:6]}"
        create_job(OUTPUT_DIR, job_id, {"status": "needs_review", "phase": "quality_review", "quality_gate": {"passed": False}})
        response = self.client.post(f"/api/jobs/{job_id}/quality-review", json={"decision": "maybe"})
        self.assertEqual(response.status_code, 400)

    def test_normalises_contact_sheet_candidate_back_to_source_panel(self):
        candidate = {"quad": [[660, 50], [740, 50], [740, 120], [660, 120]], "confidence": 0.9}
        mapped = _normalise_candidate(candidate, scale=1.0, panel_width=640, panel_count=3)
        self.assertEqual(mapped["source_panel_index"], 1)
        self.assertEqual(mapped["quad"], [[20.0, 50.0], [100.0, 50.0], [100.0, 120.0], [20.0, 120.0]])

    def test_get_demo_results(self):
        res = self.client.get("/api/demo_results")
        self.assertIn(res.status_code, [200, 404])
        if res.status_code == 200:
            data = res.get_json()
            self.assertEqual(data["status"], "completed")
            self.assertEqual(len(data["outputs"]), 3)
            self.assertEqual(data["initialization_mode"], "manual_fallback")
            self.assertTrue(data["summary"].get("legacy_demo"))

    def test_create_job_accepts_one_branch_and_returns_202(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app._submit_job") as submit:
            res = self.client.post("/api/jobs", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 202)
        payload = res.get_json()
        self.assertEqual(payload["status"], "queued")
        submit.assert_called_once_with(payload["job_id"])
        get_res = self.client.get(f"/api/jobs/{payload['job_id']}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.get_json()["branch_count"], 1)

    def test_job_idempotency_replays_original_and_rejects_payload_change(self):
        key = f"job-replay-{uuid.uuid4().hex}"

        def build_data(label="b1"):
            return {
                "video": (_make_dummy_mp4(), "video.mp4"),
                "branches": [(_make_dummy_image(label), "b1.png")],
            }

        with patch("app._submit_job") as submit:
            first = self.client.post(
                "/api/jobs",
                data=build_data(),
                headers={"Idempotency-Key": key},
                content_type="multipart/form-data",
            )
            self.assertEqual(first.status_code, 202)
            first_payload = first.get_json()
            JOBS.clear()  # prove the retry reads the durable resource record
            replay = self.client.post(
                "/api/jobs",
                data=build_data(),
                headers={"Idempotency-Key": key},
                content_type="multipart/form-data",
            )
            self.assertEqual(replay.status_code, 202)
            self.assertEqual(replay.get_json()["job_id"], first_payload["job_id"])
            changed = self.client.post(
                "/api/jobs",
                data=build_data("changed"),
                headers={"Idempotency-Key": key},
                content_type="multipart/form-data",
            )
            self.assertEqual(changed.status_code, 409)
            self.assertIn("different request", changed.get_json()["error"])
            submit.assert_called_once_with(first_payload["job_id"])

    def test_batch_idempotency_replays_original_without_duplicate_children(self):
        profile_id = f"test_batch_idempotency_{uuid.uuid4().hex[:8]}"
        profile_data = {
            "profile_id": profile_id,
            "brand_name": "Idempotent Batch Brand",
            "branches": [(_make_dummy_image("a"), "a.png")],
        }

        def build_data():
            return {
                "brand_profile_id": profile_id,
                "videos": [(_make_dummy_mp4(), "one.mp4"), (_make_dummy_mp4(), "two.mp4")],
            }

        try:
            self.assertEqual(self.client.post("/api/brand-profiles", data=profile_data, content_type="multipart/form-data").status_code, 201)
            key = f"batch-replay-{uuid.uuid4().hex}"
            submitted: list[str] = []
            with patch("app._submit_job", side_effect=lambda job_id: submitted.append(job_id)):
                first = self.client.post(
                    "/api/batches",
                    data=build_data(),
                    headers={"Idempotency-Key": key},
                    content_type="multipart/form-data",
                )
                self.assertEqual(first.status_code, 202)
                first_payload = first.get_json()
                replay = self.client.post(
                    "/api/batches",
                    data=build_data(),
                    headers={"Idempotency-Key": key},
                    content_type="multipart/form-data",
                )
            self.assertEqual(replay.status_code, 202)
            self.assertEqual(replay.get_json()["batch_id"], first_payload["batch_id"])
            self.assertEqual(replay.get_json()["job_ids"], first_payload["job_ids"])
            self.assertEqual(submitted, first_payload["job_ids"])
        finally:
            shutil.rmtree(PROFILE_DIR / profile_id, ignore_errors=True)

    def test_brand_profile_can_be_created_and_listed(self):
        profile_id = f"test_brand_{uuid.uuid4().hex[:8]}"
        data = {
            "profile_id": profile_id,
            "brand_name": "North Star Coffee",
            "target_hint": "dark charcoal fascia with white wordmark",
            "branch_labels": json.dumps(["滨河路店", "市中心店"], ensure_ascii=False),
            "branches": [(_make_dummy_image("one"), "one.png"), (_make_dummy_image("two"), "two.png")],
        }
        try:
            response = self.client.post("/api/brand-profiles", data=data, content_type="multipart/form-data")
            self.assertEqual(response.status_code, 201)
            profile = response.get_json()
            self.assertEqual(profile["profile_id"], profile_id)
            self.assertEqual(profile["scene_reuse_policy"], "vision_each_scene")
            self.assertEqual(profile["branch_labels"], ["滨河路店", "市中心店"])
            self.assertEqual(len(profile["branch_files"]), 2)
            self.assertEqual(len(profile["branch_assets"]), 2)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in profile["branch_assets"]))
            self.assertTrue(all(item["bytes"] > 0 for item in profile["branch_assets"]))
            listed = self.client.get("/api/brand-profiles").get_json()["profiles"]
            self.assertTrue(any(item["profile_id"] == profile_id for item in listed))
            fetched = self.client.get(f"/api/brand-profiles/{profile_id}")
            self.assertEqual(fetched.status_code, 200)
            self.assertEqual(fetched.get_json()["target_hint"], "dark charcoal fascia with white wordmark")
        finally:
            shutil.rmtree(PROFILE_DIR / profile_id, ignore_errors=True)

    def test_verified_scene_reuse_skips_second_vision_call(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        scenes = [
            {"scene_id": "scene_001", "start_frame": 0, "end_frame": 1, "samples": [(0, frame), (1, frame)]},
            {"scene_id": "scene_002", "start_frame": 2, "end_frame": 3, "samples": [(2, frame), (3, frame)]},
        ]
        candidate = {"quad": [[20, 20], [100, 20], [100, 80], [20, 80]], "confidence": 0.9}
        transferred = {"quad": [[30, 25], [110, 25], [110, 85], [30, 85]], "confidence": 0.88}
        validation = {"accepted": True, "requires_human_review": False, "reasons": [], "sample_metrics": []}
        with tempfile.TemporaryDirectory(prefix="ff_scene_reuse_app_") as sample_dir, \
             patch("app._locate_and_validate", return_value=([candidate], [validation], "gemini-3.1-flash-image", {"model": "gemini-3.1-flash-image", "candidate_count": 1})) as locate, \
             patch("app.transfer_candidate", return_value=transferred), \
             patch("app.validate_candidate", return_value=validation):
            views, plan, model, audit = _locate_scenes(
                Path("unused.mp4"), scenes, Path(sample_dir), None, "reuse-test", scene_reuse_policy="reuse_if_verified"
            )
        self.assertEqual(locate.call_count, 1)
        self.assertEqual([item["initialization_mode"] for item in views], ["vision_model", "reused_candidate"])
        self.assertEqual([item["surfaces"][0]["initialization_mode"] for item in plan], ["vision_model", "reused_candidate"])
        self.assertEqual(audit["scenes"][1]["source"], "scene_reuse")

    def test_job_reuses_brand_profile_without_branch_uploads(self):
        profile_id = f"test_reuse_{uuid.uuid4().hex[:8]}"
        profile_data = {
            "profile_id": profile_id,
            "brand_name": "Reusable Brand",
            "branch_labels": json.dumps(["A 店", "B 店"], ensure_ascii=False),
            "branches": [(_make_dummy_image("a"), "a.png"), (_make_dummy_image("b"), "b.png")],
        }
        try:
            created_profile = self.client.post("/api/brand-profiles", data=profile_data, content_type="multipart/form-data")
            self.assertEqual(created_profile.status_code, 201)
            job_data = {"video": (_make_dummy_mp4(), "video.mp4"), "brand_profile_id": profile_id}
            with patch("app._submit_job") as submit:
                response = self.client.post("/api/jobs", data=job_data, content_type="multipart/form-data")
            self.assertEqual(response.status_code, 202)
            job = response.get_json()
            self.assertEqual(job["brand_profile_id"], profile_id)
            self.assertEqual(job["branch_labels"], ["A_店", "B_店"])
            self.assertEqual(job["branch_count"], 2)
            self.assertEqual(len(job["brand_profile_assets"]), 2)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in job["brand_profile_assets"]))
            self.assertTrue((UPLOAD_DIR / job["job_id"] / "branch_1.png").exists())
            submit.assert_called_once_with(job["job_id"])
        finally:
            shutil.rmtree(PROFILE_DIR / profile_id, ignore_errors=True)

    def test_batch_submits_multiple_videos_against_one_brand_profile(self):
        profile_id = f"test_batch_{uuid.uuid4().hex[:8]}"
        profile_data = {
            "profile_id": profile_id,
            "brand_name": "Batch Brand",
            "branch_labels": json.dumps(["A 店"], ensure_ascii=False),
            "branches": [(_make_dummy_image("a"), "a.png")],
        }
        try:
            created_profile = self.client.post("/api/brand-profiles", data=profile_data, content_type="multipart/form-data")
            self.assertEqual(created_profile.status_code, 201)
            submitted: list[str] = []
            data = {
                "brand_profile_id": profile_id,
                "videos": [(_make_dummy_mp4(), "one.mp4"), (_make_dummy_mp4(), "two.mp4")],
            }
            with patch("app._submit_job", side_effect=lambda job_id: submitted.append(job_id)):
                response = self.client.post("/api/batches", data=data, content_type="multipart/form-data")
            self.assertEqual(response.status_code, 202)
            payload = response.get_json()
            self.assertEqual(payload["status"], "queued")
            self.assertEqual(payload["brand_profile_id"], profile_id)
            self.assertEqual(payload["requested_count"], 2)
            self.assertEqual(len(payload["job_ids"]), 2)
            self.assertEqual(submitted, payload["job_ids"])
            fetched = self.client.get(f"/api/batches/{payload['batch_id']}")
            self.assertEqual(fetched.status_code, 200)
            fetched_payload = fetched.get_json()
            self.assertEqual(fetched_payload["status"], "queued")
            self.assertEqual(fetched_payload["jobs"][0]["video_file"], "one.mp4")
            self.assertEqual([item["brand_profile_id"] for item in [self.client.get(f"/api/jobs/{job_id}").get_json() for job_id in payload["job_ids"]]], [profile_id, profile_id])
            listed = self.client.get("/api/batches?limit=5")
            self.assertEqual(listed.status_code, 200)
            self.assertTrue(any(item["batch_id"] == payload["batch_id"] for item in listed.get_json()["batches"]))
        finally:
            shutil.rmtree(PROFILE_DIR / profile_id, ignore_errors=True)

    def test_batch_requires_profile_and_rejects_branch_uploads(self):
        data = {"videos": [(_make_dummy_mp4(), "one.mp4")]}
        response = self.client.post("/api/batches", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("brand_profile_id", response.get_json()["error"])

    def test_batch_storage_failure_returns_507_and_removes_partial_batch(self):
        profile_id = f"test_batch_storage_{uuid.uuid4().hex[:8]}"
        profile_data = {
            "profile_id": profile_id,
            "brand_name": "Storage Failure Brand",
            "branches": [(_make_dummy_image("a"), "a.png")],
        }
        try:
            created = self.client.post("/api/brand-profiles", data=profile_data, content_type="multipart/form-data")
            self.assertEqual(created.status_code, 201)
            with patch("app._create_job_from_request", side_effect=OSError("disk full")):
                response = self.client.post(
                    "/api/batches",
                    data={"brand_profile_id": profile_id, "videos": [(_make_dummy_mp4(), "one.mp4")]},
                    content_type="multipart/form-data",
                )
            self.assertEqual(response.status_code, 507)
            payload = response.get_json()
            self.assertIn("disk full", payload["error"])
            self.assertFalse((BATCH_DIR / f"{payload['batch_id']}.json").exists())
        finally:
            shutil.rmtree(PROFILE_DIR / profile_id, ignore_errors=True)

    def test_batch_keeps_children_queued_when_worker_pool_is_full(self):
        profile_id = f"test_batch_queue_{uuid.uuid4().hex[:8]}"
        profile_data = {
            "profile_id": profile_id,
            "brand_name": "Batch Queue Brand",
            "branches": [(_make_dummy_image("a"), "a.png")],
        }
        try:
            self.assertEqual(self.client.post("/api/brand-profiles", data=profile_data, content_type="multipart/form-data").status_code, 201)
            data = {
                "brand_profile_id": profile_id,
                "videos": [(_make_dummy_mp4(), "one.mp4"), (_make_dummy_mp4(), "two.mp4")],
            }
            with patch("app._submit_job", side_effect=RuntimeError("active job queue is full (4 jobs); try again later")):
                response = self.client.post("/api/batches", data=data, content_type="multipart/form-data")
            self.assertEqual(response.status_code, 202)
            payload = response.get_json()
            self.assertEqual(payload["queued_waiting_for_worker"], 2)
            self.assertEqual(payload["counts"]["queued"], 2)
        finally:
            shutil.rmtree(PROFILE_DIR / profile_id, ignore_errors=True)

    def test_batch_download_packages_completed_children_and_audit_manifests(self):
        batch_id = f"download_{uuid.uuid4().hex[:8]}"
        completed_id = f"job_complete_{uuid.uuid4().hex[:8]}"
        failed_id = f"job_failed_{uuid.uuid4().hex[:8]}"
        completed_dir = OUTPUT_DIR / completed_id
        failed_dir = OUTPUT_DIR / failed_id
        try:
            completed_dir.mkdir(parents=True, exist_ok=True)
            video_path = completed_dir / "branch_1_store.mp4"
            video_path.write_bytes(b"verified-video-artifact")
            manifest = {
                "job_id": completed_id,
                "status": "completed",
                "vision_audit": {"model": "gemini-3.1-flash-image", "candidate_count": 1},
            }
            (completed_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            create_job(OUTPUT_DIR, completed_id, {
                "status": "completed",
                "batch_id": batch_id,
                "video_file": "shoot_a.mp4",
                "branch_count": 1,
                "branch_labels": ["store"],
                "outputs": [f"/outputs/{completed_id}/{video_path.name}"],
            })
            create_job(OUTPUT_DIR, failed_id, {
                "status": "failed",
                "batch_id": batch_id,
                "video_file": "shoot_b.mp4",
                "error": "vision unavailable",
                "outputs": [],
            })
            create_batch(BATCH_DIR, batch_id, {
                "brand_profile_id": "north-star",
                "requested_count": 2,
                "job_ids": [completed_id, failed_id],
            })

            response = self.client.get(f"/api/batches/{batch_id}/download")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "application/zip")
            with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
                names = set(archive.namelist())
                self.assertIn("batch_manifest.json", names)
                self.assertIn(f"jobs/{completed_id}/manifest.json", names)
                self.assertIn(f"jobs/{completed_id}/outputs/{video_path.name}", names)
                self.assertNotIn(f"jobs/{failed_id}/manifest.json", names)
                package_manifest = json.loads(archive.read("batch_manifest.json"))
                included = next(item for item in package_manifest["jobs"] if item["job_id"] == completed_id)
                omitted = next(item for item in package_manifest["jobs"] if item["job_id"] == failed_id)
                self.assertTrue(included["included"])
                self.assertEqual(included["outputs"][0]["bytes"], video_path.stat().st_size)
                self.assertEqual(len(included["outputs"][0]["sha256"]), 64)
                self.assertFalse(omitted["included"])
                self.assertEqual(package_manifest["included_job_count"], 1)
                self.assertEqual(package_manifest["omitted_job_count"], 1)
            response.close()
        finally:
            shutil.rmtree(completed_dir, ignore_errors=True)
            shutil.rmtree(failed_dir, ignore_errors=True)
            (BATCH_DIR / f"{batch_id}.json").unlink(missing_ok=True)
            try:
                (BATCH_DIR / "packages" / f"{batch_id}.zip").unlink(missing_ok=True)
            except PermissionError:
                # Flask's Windows test client can still hold the send_file
                # handle until the response object is collected.
                pass

    def test_batch_download_returns_conflict_until_a_completed_artifact_exists(self):
        batch_id = f"download_pending_{uuid.uuid4().hex[:8]}"
        job_id = f"job_pending_{uuid.uuid4().hex[:8]}"
        try:
            create_job(OUTPUT_DIR, job_id, {
                "status": "needs_review",
                "batch_id": batch_id,
                "video_file": "shoot.mp4",
                "outputs": [],
            })
            create_batch(BATCH_DIR, batch_id, {"requested_count": 1, "job_ids": [job_id]})
            response = self.client.get(f"/api/batches/{batch_id}/download")
            self.assertEqual(response.status_code, 409)
            self.assertIn("no completed", response.get_json()["error"])
            self.assertEqual(response.get_json()["batch"]["jobs"][0]["status"], "needs_review")
        finally:
            shutil.rmtree(OUTPUT_DIR / job_id, ignore_errors=True)
            (BATCH_DIR / f"{batch_id}.json").unlink(missing_ok=True)

    def test_batch_download_rejects_completed_output_outside_child_directory(self):
        batch_id = f"download_path_{uuid.uuid4().hex[:8]}"
        job_id = f"job_path_{uuid.uuid4().hex[:8]}"
        job_dir = OUTPUT_DIR / job_id
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "manifest.json").write_text("{}", encoding="utf-8")
            create_job(OUTPUT_DIR, job_id, {
                "status": "completed",
                "batch_id": batch_id,
                "video_file": "shoot.mp4",
                "outputs": [f"/outputs/{job_id}/../other.mp4"],
            })
            create_batch(BATCH_DIR, batch_id, {"requested_count": 1, "job_ids": [job_id]})
            response = self.client.get(f"/api/batches/{batch_id}/download")
            self.assertEqual(response.status_code, 409)
            self.assertIn("no completed", response.get_json()["error"])
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            (BATCH_DIR / f"{batch_id}.json").unlink(missing_ok=True)


    def test_brand_profile_default_fit_maps_single_surface(self):
        profile_id = f"test_fit_{uuid.uuid4().hex[:8]}"
        profile_data = {
            "profile_id": profile_id,
            "brand_name": "Fit Brand",
            "default_fit": "contain",
            "branches": [(_make_dummy_image("a"), "a.png"), (_make_dummy_image("b"), "b.png")],
        }
        mock_instance = MagicMock()
        mock_instance.locate_candidates.return_value = {
            "model": "gemini-3.1-flash-image",
            "candidates": [{"quad": [[20, 20], [100, 20], [100, 80], [20, 80]], "confidence": 0.9}],
        }
        accepted = {"accepted": True, "requires_human_review": False, "reasons": [], "sample_metrics": []}
        summary = {"frames": 2, "fps": 10.0, "failed_frames": [], "output_videos": []}
        try:
            created = self.client.post("/api/brand-profiles", data=profile_data, content_type="multipart/form-data")
            self.assertEqual(created.status_code, 201)
            with patch("app.VisionClient", return_value=mock_instance), \
                 patch("app.validate_candidate", return_value=accepted), \
                 patch("app.run_tracker", return_value=summary) as tracker:
                response = self.client.post(
                    "/api/render",
                    data={"video": (_make_dummy_mp4(), "video.mp4"), "brand_profile_id": profile_id},
                    content_type="multipart/form-data",
                )
            self.assertEqual(response.status_code, 200)
            bindings = tracker.call_args.kwargs["asset_bindings"]
            self.assertEqual(len(bindings), 2)
            self.assertTrue(all(item["fit"] == "contain" for item in bindings))
        finally:
            shutil.rmtree(PROFILE_DIR / profile_id, ignore_errors=True)

    def test_brand_profile_all_surface_policy_maps_multi_surface_job(self):
        profile_id = f"test_surfaces_{uuid.uuid4().hex[:8]}"
        profile_data = {
            "profile_id": profile_id,
            "brand_name": "Multi Surface Brand",
            "surface_policy": "all_surfaces",
            "branches": [(_make_dummy_image("a"), "a.png"), (_make_dummy_image("b"), "b.png")],
        }
        mock_instance = MagicMock()
        mock_instance.locate_candidates.return_value = {
            "model": "gemini-3.1-flash-image",
            "candidates": [
                {"quad": [[20, 20], [100, 20], [100, 80], [20, 80]], "confidence": 0.9},
                {"quad": [[140, 20], [220, 20], [220, 80], [140, 80]], "confidence": 0.8},
            ],
        }
        accepted = {"accepted": True, "requires_human_review": False, "reasons": [], "sample_metrics": []}
        summary = {"frames": 4, "fps": 10.0, "failed_frames": [], "output_videos": []}
        try:
            created = self.client.post("/api/brand-profiles", data=profile_data, content_type="multipart/form-data")
            self.assertEqual(created.status_code, 201)
            with patch("app.VisionClient", return_value=mock_instance), \
                 patch("app.validate_candidate", return_value=accepted), \
                 patch("app.run_tracker", return_value=summary) as tracker:
                response = self.client.post(
                    "/api/render",
                    data={"video": (_make_two_scene_mp4(), "video.mp4"), "brand_profile_id": profile_id},
                    content_type="multipart/form-data",
                )
            self.assertEqual(response.status_code, 200)
            bindings = tracker.call_args.kwargs["asset_bindings"]
            self.assertEqual(len(bindings), 2)
            self.assertEqual(len(bindings[0]["surface_ids"]), 4)
            self.assertEqual(bindings[0]["surface_ids"], [
                "scene_001_surface_001", "scene_001_surface_002",
                "scene_002_surface_001", "scene_002_surface_002",
            ])
        finally:
            shutil.rmtree(PROFILE_DIR / profile_id, ignore_errors=True)

    def test_create_job_persists_explicit_branch_labels(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "one.png"), (_make_dummy_image("b2"), "two.png")],
            "branch_labels": json.dumps(["滨河路店", "市中心店"], ensure_ascii=False),
        }
        with patch("app._submit_job"):
            response = self.client.post("/api/jobs", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertEqual(payload["branch_labels"], ["滨河路店", "市中心店"])
        persisted = self.client.get(f"/api/jobs/{payload['job_id']}").get_json()
        self.assertEqual(persisted["branch_labels"], ["滨河路店", "市中心店"])

    def test_list_jobs_reads_persisted_review_state(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app._submit_job"):
            created = self.client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()
        job_id = created["job_id"]
        job = load_job(OUTPUT_DIR, job_id)
        job.update({"status": "needs_review", "phase": "human_review", "error": "candidate requires review"})
        save_job(OUTPUT_DIR, job_id, job)
        response = self.client.get("/api/jobs?status=needs_review&limit=5")
        self.assertEqual(response.status_code, 200)
        listed = response.get_json()["jobs"]
        self.assertTrue(any(item["job_id"] == job_id and item["status"] == "needs_review" for item in listed))

    def test_list_jobs_rejects_invalid_filter(self):
        response = self.client.get("/api/jobs?status=unknown")
        self.assertEqual(response.status_code, 400)

    def test_failed_job_retry_reuses_inputs_and_clears_stale_outputs(self):
        job_id = f"retry_{uuid.uuid4().hex[:8]}"
        upload_dir = UPLOAD_DIR / job_id
        output_dir = OUTPUT_DIR / job_id
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            (upload_dir / "video.mp4").write_bytes(b"source")
            (upload_dir / "branch_1.png").write_bytes(b"branch")
            (output_dir / "stale.mp4").write_bytes(b"stale")
            create_job(OUTPUT_DIR, job_id, {
                "status": "failed", "phase": "failed", "error": "temporary failure",
                "video_file": "video.mp4", "processing_video_file": "video.mp4",
                "branch_files": ["branch_1.png"], "branch_count": 1, "outputs": ["/outputs/retry/stale.mp4"],
            })
            with patch("app._submit_job") as submit:
                response = self.client.post(f"/api/jobs/{job_id}/retry")
            self.assertEqual(response.status_code, 202)
            submit.assert_called_once_with(job_id)
            retried = response.get_json()
            self.assertEqual(retried["status"], "queued")
            self.assertEqual(retried["outputs"], [])
            self.assertIsNone(retried["error"])
            self.assertFalse((output_dir / "stale.mp4").exists())
            self.assertTrue((upload_dir / "video.mp4").exists())
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
            (OUTPUT_DIR / f"{job_id}.json").unlink(missing_ok=True)

    def test_retry_queue_full_restores_failed_state_and_outputs(self):
        job_id = f"retry_full_{uuid.uuid4().hex[:8]}"
        upload_dir = UPLOAD_DIR / job_id
        output_dir = OUTPUT_DIR / job_id
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            (upload_dir / "video.mp4").write_bytes(b"source")
            (upload_dir / "branch_1.png").write_bytes(b"branch")
            (output_dir / "stale.mp4").write_bytes(b"stale")
            create_job(OUTPUT_DIR, job_id, {
                "status": "vision_failed", "phase": "vision_failed", "error": "gateway",
                "video_file": "video.mp4", "processing_video_file": "video.mp4", "branch_files": ["branch_1.png"],
            })
            with patch("app._submit_job", side_effect=RuntimeError("active job queue is full (4 jobs); try again later")):
                response = self.client.post(f"/api/jobs/{job_id}/retry")
            self.assertEqual(response.status_code, 429)
            persisted = self.client.get(f"/api/jobs/{job_id}").get_json()
            self.assertEqual(persisted["status"], "vision_failed")
            self.assertEqual(persisted["error"], "gateway")
            self.assertTrue((output_dir / "stale.mp4").exists())
            self.assertFalse((OUTPUT_DIR / ".retry-backups").exists())
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
            (OUTPUT_DIR / f"{job_id}.json").unlink(missing_ok=True)

    def test_retry_storage_failure_restores_failed_state_and_outputs(self):
        job_id = f"retry_oserror_{uuid.uuid4().hex[:8]}"
        upload_dir = UPLOAD_DIR / job_id
        output_dir = OUTPUT_DIR / job_id
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            (upload_dir / "video.mp4").write_bytes(b"source")
            (upload_dir / "branch_1.png").write_bytes(b"branch")
            (output_dir / "stale.mp4").write_bytes(b"stale")
            create_job(OUTPUT_DIR, job_id, {
                "status": "failed", "phase": "failed", "error": "temporary failure",
                "video_file": "video.mp4", "processing_video_file": "video.mp4", "branch_files": ["branch_1.png"],
            })
            with patch("app._submit_job", side_effect=OSError("state locked")):
                response = self.client.post(f"/api/jobs/{job_id}/retry")
            self.assertEqual(response.status_code, 507)
            persisted = self.client.get(f"/api/jobs/{job_id}").get_json()
            self.assertEqual(persisted["status"], "failed")
            self.assertEqual(persisted["error"], "temporary failure")
            self.assertTrue((output_dir / "stale.mp4").exists())
            self.assertFalse((OUTPUT_DIR / ".retry-backups").exists())
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
            shutil.rmtree(OUTPUT_DIR / ".retry-backups", ignore_errors=True)
            (OUTPUT_DIR / f"{job_id}.json").unlink(missing_ok=True)

    def test_retry_preserves_human_reviewed_scene_plan_and_bindings(self):
        job_id = f"retry_reviewed_{uuid.uuid4().hex[:8]}"
        upload_dir = UPLOAD_DIR / job_id
        output_dir = OUTPUT_DIR / job_id
        scene_plan = [{
            "scene_id": "scene_001",
            "start_frame": 0,
            "end_frame": 1,
            "surfaces": [{
                "surface_id": "scene_001_surface_001",
                "quad": [[20, 20], [100, 20], [100, 80], [20, 80]],
            }],
        }]
        bindings = [{"branch_index": 1, "surface_ids": ["scene_001_surface_001"], "fit": "cover"}]
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            (upload_dir / "video.mp4").write_bytes(b"source")
            (upload_dir / "branch_1.png").write_bytes(b"branch")
            (output_dir / "stale.mp4").write_bytes(b"stale")
            create_job(OUTPUT_DIR, job_id, {
                "status": "failed", "phase": "failed", "error": "tracking failure",
                "initialization_mode": "reviewed_candidate",
                "video_file": "video.mp4", "processing_video_file": "video.mp4",
                "branch_files": ["branch_1.png"], "branch_count": 1,
                "scene_plan": scene_plan, "asset_bindings": bindings,
            })
            with patch("app._submit_job"):
                response = self.client.post(f"/api/jobs/{job_id}/retry")
            self.assertEqual(response.status_code, 202)
            retried = response.get_json()
            self.assertEqual(retried["initialization_mode"], "reviewed_candidate")
            self.assertEqual(retried["scene_plan"], scene_plan)
            self.assertEqual(retried["asset_bindings"], bindings)
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
            shutil.rmtree(OUTPUT_DIR / ".retry-backups", ignore_errors=True)
            (OUTPUT_DIR / f"{job_id}.json").unlink(missing_ok=True)

    def test_retry_rejects_completed_job_and_missing_inputs(self):
        job_id = f"retry_invalid_{uuid.uuid4().hex[:8]}"
        missing_id = f"retry_missing_{uuid.uuid4().hex[:8]}"
        try:
            create_job(OUTPUT_DIR, job_id, {"status": "completed", "phase": "complete"})
            response = self.client.post(f"/api/jobs/{job_id}/retry")
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.get_json()["status"], "completed")
            create_job(OUTPUT_DIR, missing_id, {
                "status": "failed", "phase": "failed", "video_file": "missing.mp4",
                "processing_video_file": "missing.mp4", "branch_files": ["branch.png"],
            })
            missing = self.client.post(f"/api/jobs/{missing_id}/retry")
            self.assertEqual(missing.status_code, 409)
            self.assertIn("missing", missing.get_json()["error"])
        finally:
            shutil.rmtree(UPLOAD_DIR / job_id, ignore_errors=True)
            shutil.rmtree(OUTPUT_DIR / job_id, ignore_errors=True)
            (OUTPUT_DIR / f"{job_id}.json").unlink(missing_ok=True)
            shutil.rmtree(UPLOAD_DIR / missing_id, ignore_errors=True)
            shutil.rmtree(OUTPUT_DIR / missing_id, ignore_errors=True)
            (OUTPUT_DIR / f"{missing_id}.json").unlink(missing_ok=True)

    def test_create_job_normalizes_mov_source_and_keeps_original_name(self):
        data = {
            "video": (_make_dummy_mp4(), "phone_capture.mov"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }

        def copy_as_working(source, destination):
            shutil.copyfile(source, destination)
            return destination

        with patch("app.normalize_video_for_processing", side_effect=copy_as_working), patch("app._submit_job") as submit:
            response = self.client.post("/api/jobs", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertEqual(payload["video_file"], "phone_capture.mov")
        self.assertEqual(payload["processing_video_file"], "processing_input.mp4")
        self.assertEqual(payload["video_metadata"]["source_extension"], ".mov")
        self.assertEqual(payload["video_metadata"]["processing_extension"], ".mp4")
        self.assertTrue((UPLOAD_DIR / payload["job_id"] / "phone_capture.mov").exists())
        self.assertTrue((UPLOAD_DIR / payload["job_id"] / "processing_input.mp4").exists())
        submit.assert_called_once_with(payload["job_id"])

    def test_hard_cut_video_sends_stable_scene_plan_to_tracker(self):
        data = {
            "video": (_make_two_scene_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        mock_instance = MagicMock()
        mock_instance.locate_candidates.return_value = {
            "model": "gemini-3.1-flash-image",
            "candidates": [{"quad": [[20, 20], [100, 20], [100, 80], [20, 80]], "confidence": 0.9}],
        }
        summary = {"frames": 4, "fps": 10.0, "failed_frames": [], "output_videos": []}
        with patch("app.VisionClient", return_value=mock_instance), \
             patch("app.validate_candidate", return_value={"accepted": True, "requires_human_review": False, "reasons": [], "sample_metrics": []}), \
             patch("app.run_tracker", return_value=summary) as tracker, \
             patch("app.sample_video_scenes", wraps=sample_video_scenes) as scene_sampler:
            response = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["summary"]["scene_count"], 2)
        call = tracker.call_args.kwargs
        self.assertEqual([item["scene_id"] for item in call["scene_plan"]], ["scene_001", "scene_002"])
        self.assertEqual([item["start_frame"] for item in call["scene_plan"]], [0, 2])
        self.assertEqual(scene_sampler.call_args.kwargs["max_scenes"], 8)

    def test_manual_scene_quads_validate_and_reach_tracker_per_scene(self):
        data = {
            "video": (_make_two_scene_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
            "use_manual_quad": "true",
            "manual_quad": json.dumps([[20, 20], [100, 20], [100, 80], [20, 80]]),
            "manual_scene_quads": json.dumps([
                [[20, 20], [100, 20], [100, 80], [20, 80]],
                [[140, 30], [220, 30], [220, 90], [140, 90]],
            ]),
        }
        accepted = {"accepted": True, "requires_human_review": False, "reasons": [], "sample_metrics": []}
        summary = {"frames": 4, "fps": 10.0, "failed_frames": [], "output_videos": []}
        with patch("app.validate_candidate", return_value=accepted), patch("app.run_tracker", return_value=summary) as tracker:
            response = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["initialization_mode"], "manual_fallback")
        plan = tracker.call_args.kwargs["scene_plan"]
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["surfaces"][0]["quad"][0], [20.0, 20.0])
        self.assertEqual(plan[1]["surfaces"][0]["quad"][0], [140.0, 30.0])
        self.assertEqual(payload["summary"]["scene_count"], 2)

    def test_manual_scene_quad_count_must_match_detected_scenes(self):
        data = {
            "video": (_make_two_scene_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
            "use_manual_quad": "true",
            "manual_quad": json.dumps([[20, 20], [100, 20], [100, 80], [20, 80]]),
            "manual_scene_quads": json.dumps([[[20, 20], [100, 20], [100, 80], [20, 80]]]),
        }
        response = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("manual_scene_quads", response.get_json()["error"])

    def test_target_hint_is_persisted_and_forwarded_to_vision_client(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
            "target_hint": "Acme Coffee neon fascia",
        }
        mock_instance = MagicMock()
        mock_instance.locate_candidates.return_value = {
            "model": "gemini-3.1-flash-image",
            "candidates": [{"quad": [[20, 20], [100, 20], [100, 80], [20, 80]], "confidence": 0.9}],
        }
        summary = {"frames": 2, "fps": 10.0, "failed_frames": [], "output_videos": []}
        with patch("app.VisionClient", return_value=mock_instance), \
             patch("app.validate_candidate", return_value={"accepted": True, "requires_human_review": False, "reasons": [], "sample_metrics": []}), \
             patch("app.run_tracker", return_value=summary):
            response = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        job_id = response.get_json()["job_id"]
        persisted = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(persisted["target_hint"], "Acme Coffee neon fascia")
        mock_instance.locate_candidates.assert_called_once()
        self.assertEqual(mock_instance.locate_candidates.call_args.kwargs["target_hint"], "Acme Coffee neon fascia")
        references = mock_instance.locate_candidates.call_args.kwargs["reference_images"]
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0][1], "image/png")
        self.assertEqual(references[0][0], (_make_dummy_image("b1").getvalue()))

    def test_each_scene_can_pass_multiple_stable_surfaces_to_tracker(self):
        data = {
            "video": (_make_two_scene_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png"), (_make_dummy_image("b2"), "b2.png")],
            "asset_bindings": json.dumps([
                {"branch_index": 1, "surface_ids": ["scene_001_surface_001", "scene_001_surface_002", "scene_002_surface_001", "scene_002_surface_002"], "fit": "cover"},
                {"branch_index": 2, "surface_ids": ["scene_001_surface_001", "scene_001_surface_002", "scene_002_surface_001", "scene_002_surface_002"], "fit": "cover"},
            ]),
        }
        candidates = [
            {"quad": [[20, 20], [100, 20], [100, 80], [20, 80]], "confidence": 0.9, "reason": "front"},
            {"quad": [[140, 20], [220, 20], [220, 80], [140, 80]], "confidence": 0.8, "reason": "side"},
        ]
        mock_instance = MagicMock()
        mock_instance.locate_candidates.return_value = {
            "model": "gemini-3.1-flash-image",
            "candidates": candidates,
        }
        summary = {"frames": 4, "fps": 10.0, "failed_frames": [], "output_videos": ["a.mp4", "b.mp4"]}
        accepted = {"accepted": True, "requires_human_review": False, "reasons": [], "sample_metrics": []}
        with patch("app.VisionClient", return_value=mock_instance), \
             patch("app.validate_candidate", return_value=accepted), \
             patch("app.run_tracker", return_value=summary) as tracker:
            response = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        call = tracker.call_args.kwargs
        self.assertEqual([len(scene["surfaces"]) for scene in call["scene_plan"]], [2, 2])
        self.assertEqual(
            [surface["surface_id"] for surface in call["scene_plan"][0]["surfaces"]],
            ["scene_001_surface_001", "scene_001_surface_002"],
        )
        self.assertEqual(
            [surface["surface_id"] for surface in call["scene_plan"][1]["surfaces"]],
            ["scene_002_surface_001", "scene_002_surface_002"],
        )
        self.assertEqual(response.get_json()["summary"]["surface_count"], 4)

    def test_multi_surface_auto_job_requires_explicit_bindings(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png"), (_make_dummy_image("b2"), "b2.png")],
        }
        mock_instance = MagicMock()
        mock_instance.locate_candidates.return_value = {
            "model": "gemini-3.1-flash-image",
            "candidates": [
                {"quad": [[20, 20], [100, 20], [100, 80], [20, 80]], "confidence": 0.9},
                {"quad": [[140, 20], [220, 20], [220, 80], [140, 80]], "confidence": 0.8},
            ],
        }
        accepted = {"accepted": True, "requires_human_review": False, "reasons": [], "sample_metrics": []}
        with patch("app.VisionClient", return_value=mock_instance), \
             patch("app.validate_candidate", return_value=accepted), \
             patch("app.run_tracker") as tracker:
            response = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 500)
        payload = response.get_json()
        self.assertEqual(payload["status"], "needs_review")
        self.assertIn("asset_bindings", payload["error"])
        self.assertIsNotNone(payload["job_id"])
        tracker.assert_not_called()

    def test_tracking_quality_gate_does_not_report_completed(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        mock_instance = MagicMock()
        mock_instance.locate_candidates.return_value = {
            "model": "gemini-3.1-flash-image",
            "candidates": [{"quad": [[20, 20], [100, 20], [100, 80], [20, 80]], "confidence": 0.9}],
        }
        rejected_gate = {
            "status": "review", "passed": False, "failed_frames": 1,
            "failed_fraction": 1.0, "max_failed_fraction": 0.25,
            "reason": "failed_frame_fraction_exceeds_threshold",
        }
        summary = {"frames": 2, "fps": 10.0, "failed_frames": [1], "output_videos": [], "quality_gate": rejected_gate}
        with patch("app.VisionClient", return_value=mock_instance), \
             patch("app.validate_candidate", return_value={"accepted": True, "requires_human_review": False, "reasons": [], "sample_metrics": []}), \
             patch("app.run_tracker", return_value=summary):
            response = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 500)
        payload = response.get_json()
        self.assertEqual(payload["status"], "needs_review")
        self.assertIsNotNone(payload["job_id"])
        self.assertIn("quality gate", payload["error"])

    def test_overlapping_candidates_require_review_for_every_scene(self):
        data = {
            "video": (_make_two_scene_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        overlapping = [
            {"quad": [[20, 20], [100, 20], [100, 80], [20, 80]], "confidence": 0.9},
            {"quad": [[25, 25], [105, 25], [105, 85], [25, 85]], "confidence": 0.8},
        ]
        mock_instance = MagicMock()
        mock_instance.locate_candidates.return_value = {
            "model": "gemini-3.1-flash-image",
            "candidates": overlapping,
        }
        accepted = {"accepted": True, "requires_human_review": False, "reasons": [], "sample_metrics": []}
        with patch("app.VisionClient", return_value=mock_instance), patch("app.validate_candidate", return_value=accepted):
            response = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 500)
        job = next(iter(JOBS.values()))
        self.assertEqual(job["status"], "needs_review")
        self.assertEqual(len(job["scenes"]), 2)
        self.assertTrue(all(scene["status"] == "needs_review" for scene in job["scenes"]))
        self.assertTrue(all(len(scene["candidates"]) == 2 for scene in job["scenes"]))

    def test_review_accepts_multiple_surfaces_and_keeps_ids(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app._submit_job"):
            created = self.client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()
        job_id = created["job_id"]
        job = load_job(OUTPUT_DIR, job_id)
        job.update({
            "status": "needs_review",
            "scenes": [{"scene_id": "scene_001", "start_frame": 0, "end_frame": 1, "status": "needs_review"}],
        })
        save_job(OUTPUT_DIR, job_id, job)
        payload = {"scenes": [{
            "scene_id": "scene_001",
            "surfaces": [
                {"surface_id": "front_panel", "quad": [[20, 20], [100, 20], [100, 80], [20, 80]]},
                {"surface_id": "side_panel", "quad": [[140, 20], [220, 20], [220, 80], [140, 80]]},
            ],
        }], "asset_bindings": [
            {"branch_index": 1, "surface_ids": ["front_panel"], "fit": "cover"},
        ]}
        with patch("app._submit_job") as submit:
            response = self.client.post(f"/api/jobs/{job_id}/review", json=payload)
        self.assertEqual(response.status_code, 202)
        saved = load_job(OUTPUT_DIR, job_id)
        self.assertEqual(
            [surface["surface_id"] for surface in saved["scene_plan"][0]["surfaces"]],
            ["front_panel", "side_panel"],
        )
        self.assertEqual(saved["asset_bindings"][0]["surface_ids"], ["front_panel"])
        submit.assert_called_once_with(job_id)

    def test_review_rejects_overlapping_surfaces(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app._submit_job"):
            created = self.client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()
        job_id = created["job_id"]
        job = load_job(OUTPUT_DIR, job_id)
        job.update({
            "status": "needs_review",
            "scenes": [{"scene_id": "scene_001", "start_frame": 0, "end_frame": 1, "status": "needs_review"}],
        })
        save_job(OUTPUT_DIR, job_id, job)
        payload = {"scenes": [{
            "scene_id": "scene_001",
            "surfaces": [
                {"surface_id": "a", "quad": [[20, 20], [100, 20], [100, 80], [20, 80]]},
                {"surface_id": "b", "quad": [[25, 25], [105, 25], [105, 85], [25, 85]]},
            ],
        }]}
        response = self.client.post(f"/api/jobs/{job_id}/review", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("overlap", response.get_json()["error"].lower())

    def test_review_rejects_multi_surface_without_bindings(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app._submit_job"):
            created = self.client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()
        job_id = created["job_id"]
        job = load_job(OUTPUT_DIR, job_id)
        job.update({
            "status": "needs_review",
            "scenes": [{"scene_id": "scene_001", "start_frame": 0, "end_frame": 1, "status": "needs_review"}],
        })
        save_job(OUTPUT_DIR, job_id, job)
        payload = {"scenes": [{
            "scene_id": "scene_001",
            "surfaces": [
                {"surface_id": "front_panel", "quad": [[20, 20], [100, 20], [100, 80], [20, 80]]},
                {"surface_id": "side_panel", "quad": [[140, 20], [220, 20], [220, 80], [140, 80]]},
            ],
        }]}
        response = self.client.post(f"/api/jobs/{job_id}/review", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("asset_bindings", response.get_json()["error"])

    def test_scene_review_state_is_persisted_before_worker_returns(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        mock_instance = MagicMock()
        mock_instance.locate_candidates.return_value = {
            "model": "gemini-3.1-flash-image",
            "candidates": [{"quad": [[20, 20], [100, 20], [100, 80], [20, 80]], "confidence": 0.2}],
        }
        rejected = {"accepted": False, "requires_human_review": True, "reasons": ["inconsistent_samples"], "sample_metrics": []}
        with patch("app.VisionClient", return_value=mock_instance), patch("app.validate_candidate", return_value=rejected):
            response = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 500)
        job = next(iter(JOBS.values()))
        self.assertEqual(job["status"], "needs_review")
        self.assertEqual(load_job(OUTPUT_DIR, job["job_id"])["status"], "needs_review")

    def test_create_job_rejects_more_than_ten_branches(self):
        data = {"video": (_make_dummy_mp4(), "video.mp4")}
        for index in range(11):
            data.setdefault("branches", []).append((_make_dummy_image(f"b{index}"), f"b{index}.png"))
        res = self.client.post("/api/jobs", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn("1 to 10", res.get_json()["error"])

    def test_review_endpoint_only_accepts_needs_review_job(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app._submit_job"):
            created = self.client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()
        job_id = created["job_id"]
        job = load_job(OUTPUT_DIR, job_id)
        job.update({"status": "needs_review", "candidate": {"quad": [[20, 20], [100, 20], [100, 80], [20, 80]]}})
        save_job(OUTPUT_DIR, job_id, job)
        with patch("app._submit_job") as submit:
            res = self.client.post(f"/api/jobs/{job_id}/review", json={"quad": [[20, 20], [100, 20], [100, 80], [20, 80]]})
        self.assertEqual(res.status_code, 202)
        self.assertEqual(res.get_json()["initialization_mode"], "reviewed_candidate")
        submit.assert_called_once_with(job_id)

    def test_review_rejects_unknown_asset_binding_before_queueing(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app._submit_job"):
            created = self.client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()
        job_id = created["job_id"]
        job = load_job(OUTPUT_DIR, job_id)
        job.update({"status": "needs_review", "candidate": {"quad": [[20, 20], [100, 20], [100, 80], [20, 80]]}})
        save_job(OUTPUT_DIR, job_id, job)
        with patch("app._submit_job") as submit:
            response = self.client.post(
                f"/api/jobs/{job_id}/review",
                json={
                    "quad": [[20, 20], [100, 20], [100, 80], [20, 80]],
                    "asset_bindings": [{"branch_index": 1, "surface_ids": ["unknown_surface"], "fit": "cover"}],
                },
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unknown surface", response.get_json()["error"].lower())
        submit.assert_not_called()

    def test_review_queue_full_keeps_job_reviewable(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app._submit_job"):
            created = self.client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()
        job_id = created["job_id"]
        job = load_job(OUTPUT_DIR, job_id)
        job.update({"status": "needs_review", "candidate": {"quad": [[20, 20], [100, 20], [100, 80], [20, 80]]}})
        save_job(OUTPUT_DIR, job_id, job)
        with patch("app._submit_job", side_effect=RuntimeError("active job queue is full")):
            response = self.client.post(f"/api/jobs/{job_id}/review", json={"quad": [[20, 20], [100, 20], [100, 80], [20, 80]]})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(load_job(OUTPUT_DIR, job_id)["status"], "needs_review")

    def test_multi_scene_review_persists_scene_plan(self):
        data = {
            "video": (_make_two_scene_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app._submit_job"):
            created = self.client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()
        job_id = created["job_id"]
        job = load_job(OUTPUT_DIR, job_id)
        job.update({
            "status": "needs_review",
            "scenes": [
                {"scene_id": "scene_001", "start_frame": 0, "end_frame": 1, "status": "needs_review"},
                {"scene_id": "scene_002", "start_frame": 2, "end_frame": 3, "status": "needs_review"},
            ],
        })
        save_job(OUTPUT_DIR, job_id, job)
        payload = {"scenes": [
            {"scene_id": "scene_001", "quad": [[20, 20], [100, 20], [100, 80], [20, 80]]},
            {"scene_id": "scene_002", "quad": [[30, 30], [110, 30], [110, 90], [30, 90]]},
        ]}
        with patch("app._submit_job") as submit:
            response = self.client.post(f"/api/jobs/{job_id}/review", json=payload)
        self.assertEqual(response.status_code, 202)
        saved = load_job(OUTPUT_DIR, job_id)
        self.assertEqual(saved["initialization_mode"], "reviewed_candidate")
        self.assertEqual([item["scene_id"] for item in saved["scene_plan"]], ["scene_001", "scene_002"])
        self.assertEqual(saved["scene_plan"][1]["surfaces"][0]["quad"][0], [30.0, 30.0])
        submit.assert_called_once_with(job_id)

    def test_empty_scene_review_payload_is_rejected(self):
        data = {
            "video": (_make_two_scene_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch("app._submit_job"):
            created = self.client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()
        job_id = created["job_id"]
        job = load_job(OUTPUT_DIR, job_id)
        job.update({"status": "needs_review", "scenes": []})
        save_job(OUTPUT_DIR, job_id, job)
        response = self.client.post(f"/api/jobs/{job_id}/review", json={"scenes": []})
        self.assertEqual(response.status_code, 400)

    def test_outputs_endpoint_returns_branch_urls(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png"), (_make_dummy_image("b2"), "b2.png")],
        }
        with patch("app._submit_job"):
            created = self.client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()
        job_id = created["job_id"]
        job = load_job(OUTPUT_DIR, job_id)
        job.update({"status": "completed", "outputs": [f"/outputs/{job_id}/branch_1.mp4", f"/outputs/{job_id}/branch_2.mp4"]})
        save_job(OUTPUT_DIR, job_id, job)
        res = self.client.get(f"/api/jobs/{job_id}/outputs")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.get_json()["outputs"]), 2)

    def test_output_route_supports_http_range_requests(self):
        job_dir = OUTPUT_DIR / "range_test"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "sample.mp4").write_bytes(b"0123456789")
        try:
            response = self.client.get("/outputs/range_test/sample.mp4", headers={"Range": "bytes=0-3"})
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.data, b"0123")
            self.assertTrue(response.headers["Content-Range"].startswith("bytes 0-3/10"))
            response.close()
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    def test_rejects_path_like_job_ids_without_server_error(self):
        for url in ("/api/jobs/%2e%2e", "/api/jobs/%2e%2e/outputs", "/outputs/%2e%2e/app.py"):
            response = self.client.get(url)
            self.assertIn(response.status_code, (404, 400), url)

    def test_corrupt_persisted_job_is_treated_as_missing(self):
        job_dir = OUTPUT_DIR / "corrupt_job"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text("{broken", encoding="utf-8")
        try:
            response = self.client.get("/api/jobs/corrupt_job")
            self.assertEqual(response.status_code, 404)
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    def test_public_summary_redacts_local_paths(self):
        summary = _public_summary({"video": r"C:\secret\input.mp4", "output_videos": [r"C:\secret\branch.mp4"]})
        self.assertEqual(summary["video"], "input.mp4")
        self.assertEqual(summary["output_videos"], ["branch.mp4"])

    def test_rejects_manual_quad_outside_uploaded_video(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
            "use_manual_quad": "true",
            "manual_quad": json.dumps([[-1, 20], [100, 20], [100, 80], [-1, 80]]),
        }
        response = self.client.post("/api/jobs", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("manual quad", response.get_json()["error"].lower())

    def test_rejects_unreadable_branch_image(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(io.BytesIO(b"not an image"), "b1.png")],
        }
        response = self.client.post("/api/jobs", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("decode", response.get_json()["error"].lower())

    def test_rejects_new_job_when_active_queue_is_full(self):
        class PendingFuture:
            def done(self):
                return False

        import app as app_module
        app_module.FUTURES.clear()

    def test_sync_compatibility_endpoint_cannot_bypass_active_queue_limit(self):
        import app as app_module

        class PendingFuture:
            def done(self):
                return False

        app_module.FUTURES.clear()
        app_module.FUTURES["existing"] = PendingFuture()
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch.object(app_module, "MAX_ACTIVE_JOBS", 1):
            response = self.client.post("/api/render", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 429)
        self.assertIn("queue", response.get_json()["error"].lower())
        app_module.FUTURES.clear()
        app_module.FUTURES["existing"] = PendingFuture()
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branches": [(_make_dummy_image("b1"), "b1.png")],
        }
        with patch.object(app_module, "MAX_ACTIVE_JOBS", 1):
            response = self.client.post("/api/jobs", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 429)
        self.assertIn("queue", response.get_json()["error"].lower())
        app_module.FUTURES.clear()

    def test_submit_job_removes_completed_future(self):
        import app as app_module

        class ImmediateFuture:
            def done(self):
                return True

            def add_done_callback(self, callback):
                callback(self)

        class ImmediateExecutor:
            def submit(self, function, job_id):
                return ImmediateFuture()

        app_module.FUTURES.clear()
        with patch.object(app_module, "EXECUTOR", ImmediateExecutor()):
            app_module._submit_job("completed-future")
        self.assertNotIn("completed-future", app_module.FUTURES)

    def test_wsgi_first_request_triggers_recovery_once(self):
        import app as app_module

        app_module._RECOVERY_STARTED = False
        with patch.object(app_module, "_recover_jobs") as recover:
            app_module.app.config["TESTING"] = False
            self.client.get("/")
            self.client.get("/")
        recover.assert_called_once_with()
        app_module.app.config["TESTING"] = True


if __name__ == "__main__":
    unittest.main()
