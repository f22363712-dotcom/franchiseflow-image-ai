import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracker import PlanarTracker, composite, project_quad, run_tracker


def _create_textured_image(width=640, height=480):
    """Create an image with rich high-frequency textures for SIFT."""
    rng = np.random.RandomState(42)
    # Base background
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    # Add high-contrast textured patch inside a specific region [100, 100] to [300, 250]
    for y in range(80, 260, 20):
        for x in range(80, 320, 20):
            color = (int(rng.randint(0, 255)), int(rng.randint(0, 255)), int(rng.randint(0, 255)))
            cv2.rectangle(img, (x, y), (x + 18, y + 18), color, -1)
            cv2.putText(img, f"{x},{y}", (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    return img


class TrackerSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ff_test_tracker_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_synthetic_fixture(self, num_frames=6, blank_frame_idx=3):
        video_path = self.temp_dir / "synthetic.mp4"
        width, height, fps = 640, 480, 10
        base_frame = _create_textured_image(width, height)
        quad = np.float32([[100, 100], [300, 100], [300, 250], [100, 250]])

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        ground_truth_quads = []

        for i in range(num_frames):
            if i == 0:
                writer.write(base_frame)
                ground_truth_quads.append(quad.copy())
            elif i == blank_frame_idx:
                # Forced complete feature loss: blank gray frame
                blank = np.full((height, width, 3), 128, dtype=np.uint8)
                writer.write(blank)
                ground_truth_quads.append(None)
            else:
                # Small known rigid translation + slight scale
                dx = i * 4.0
                dy = i * 2.0
                M = np.float32([
                    [1.0, 0.0, dx],
                    [0.0, 1.0, dy],
                    [0.0, 0.0, 1.0]
                ])
                transformed = cv2.warpPerspective(base_frame, M, (width, height))
                writer.write(transformed)
                ground_truth_quads.append(project_quad(M, quad))

        writer.release()

        # Create 3 small dummy branch signs (RGBA)
        signs = []
        for name, col in [("b1", (255, 0, 0, 255)), ("b2", (0, 255, 0, 255)), ("b3", (0, 0, 255, 255))]:
            sign_path = self.temp_dir / f"{name}.png"
            sign_img = np.full((100, 200, 4), col, dtype=np.uint8)
            cv2.putText(sign_img, name, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255, 255), 2)
            cv2.imwrite(str(sign_path), sign_img)
            signs.append(sign_path)

        return video_path, quad, signs, ground_truth_quads

    def test_run_tracker_synthetic_motion_and_failed_frames(self):
        video_path, quad, signs, gt_quads = self._create_synthetic_fixture(num_frames=6, blank_frame_idx=3)
        out_dir = self.temp_dir / "out"

        summary = run_tracker(
            video=video_path,
            quad=quad,
            signs=signs,
            out_dir=out_dir,
            min_inliers=6,
            ransac_px=3.0,
            no_debug=True
        )

        self.assertEqual(summary["frames"], 6)
        self.assertIn(3, summary["failed_frames"])
        self.assertEqual(len(summary["output_videos"]), 3)

        # Verify summary.json and tracking.jsonl
        self.assertTrue((out_dir / "summary.json").exists())
        self.assertTrue((out_dir / "tracking.jsonl").exists())

        # Check tracking.jsonl
        lines = (out_dir / "tracking.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 6)
        rec_blank = json.loads(lines[3])
        self.assertEqual(rec_blank["status"], "failed")
        self.assertEqual(rec_blank["render_policy"], "unmodified_original_frame")

        # Check all 3 output videos have matching frame counts
        for out_vid in summary["output_videos"]:
            cap = cv2.VideoCapture(out_vid)
            count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
            cap.release()
            self.assertEqual(count, 6)

    def test_run_tracker_validates_inputs(self):
        # Missing video
        with self.assertRaises(ValueError):
            run_tracker(self.temp_dir / "non_existent.mp4", [[0, 0], [10, 0], [10, 10], [0, 10]], [], self.temp_dir / "out")

        # Wrong number of signs
        video_path, quad, signs, _ = self._create_synthetic_fixture(num_frames=2, blank_frame_idx=-1)
        with self.assertRaises(ValueError):
            run_tracker(video_path, quad, signs[:2], self.temp_dir / "out")


    @patch("shutil.which", return_value=None)
    def test_ffmpeg_missing_raises_runtime_error(self, mock_which):
        video_path, quad, signs, _ = self._create_synthetic_fixture(num_frames=2, blank_frame_idx=-1)
        with self.assertRaises(RuntimeError) as ctx:
            run_tracker(video_path, quad, signs, self.temp_dir / "out")
        self.assertIn("FFmpeg is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
