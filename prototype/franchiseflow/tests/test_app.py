import io
import json
import shutil
import sys
import tempfile
import unittest
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


from app import app, JOBS, UPLOAD_DIR, OUTPUT_DIR
from job_store import load_job, save_job


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
            self.assertEqual(next(iter(JOBS.values()))["status"], "failed")

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
                self.assertEqual(json_data["vision_quad"], mock_quad)
                self.assertEqual(len(json_data["outputs"]), 3)
                job_id = json_data["job_id"]

                # Check GET /api/jobs/<job_id>
                get_res = self.client.get(f"/api/jobs/{job_id}")
                self.assertEqual(get_res.status_code, 200)
                self.assertEqual(get_res.get_json()["status"], "completed")

    def test_get_demo_results(self):
        res = self.client.get("/api/demo_results")
        self.assertIn(res.status_code, [200, 404])
        if res.status_code == 200:
            data = res.get_json()
            self.assertEqual(data["status"], "completed")
            self.assertEqual(len(data["outputs"]), 3)

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


if __name__ == "__main__":
    unittest.main()
