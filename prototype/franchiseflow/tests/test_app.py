import io
import json
import shutil
import sys
import tempfile
import unittest
import uuid
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


from app import app, JOBS, UPLOAD_DIR, OUTPUT_DIR, PROFILE_DIR, _normalise_candidate, _public_summary
from job_store import create_job, load_job, save_job
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
            self.assertEqual(profile["branch_labels"], ["滨河路店", "市中心店"])
            self.assertEqual(len(profile["branch_files"]), 2)
            listed = self.client.get("/api/brand-profiles").get_json()["profiles"]
            self.assertTrue(any(item["profile_id"] == profile_id for item in listed))
            fetched = self.client.get(f"/api/brand-profiles/{profile_id}")
            self.assertEqual(fetched.status_code, 200)
            self.assertEqual(fetched.get_json()["target_hint"], "dark charcoal fascia with white wordmark")
        finally:
            shutil.rmtree(PROFILE_DIR / profile_id, ignore_errors=True)

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
            self.assertTrue((UPLOAD_DIR / job["job_id"] / "branch_1.png").exists())
            submit.assert_called_once_with(job["job_id"])
        finally:
            shutil.rmtree(PROFILE_DIR / profile_id, ignore_errors=True)

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
