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

    def test_fewer_than_three_branches_returns_400(self):
        data = {
            "video": (_make_dummy_mp4(), "video.mp4"),
            "branch_1": (_make_dummy_image("b1"), "b1.png"),
            "branch_2": (_make_dummy_image("b2"), "b2.png"),
        }
        res = self.client.post("/api/render", data=data, content_type="multipart/form-data")
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
            self.assertEqual(len(JOBS), 0)

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
        mock_instance.locate_sign.return_value = {"quad": mock_quad, "model": "GLM-5.3-flash"}

        with patch("app.VisionClient", return_value=mock_instance):
            with patch("app.run_tracker", return_value=mock_summary):
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



if __name__ == "__main__":
    unittest.main()
