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

from tracker import PlanarTracker, composite, main, project_quad, run_tracker


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

        # Zero and more than ten signs are invalid
        video_path, quad, signs, _ = self._create_synthetic_fixture(num_frames=2, blank_frame_idx=-1)
        with self.assertRaises(ValueError):
            run_tracker(video_path, quad, [], self.temp_dir / "out")
        with self.assertRaises(ValueError):
            run_tracker(video_path, quad, signs * 4, self.temp_dir / "out")

    def test_run_tracker_supports_one_and_five_branches_and_records_manual_mode(self):
        video_path, quad, signs, _ = self._create_synthetic_fixture(num_frames=2, blank_frame_idx=-1)
        one = run_tracker(video_path, quad, signs[:1], self.temp_dir / "one", no_debug=True)
        self.assertEqual(one["branch_count"], 1)
        self.assertEqual(one["initialization_mode"], "manual_fallback")
        self.assertEqual(len(one["output_videos"]), 1)

        extra = []
        for index in range(2):
            path = self.temp_dir / f"extra_{index}.png"
            cv2.imwrite(str(path), np.full((100, 200, 4), (50 + index, 90, 140, 255), dtype=np.uint8))
            extra.append(path)
        five = run_tracker(video_path, quad, signs + extra, self.temp_dir / "five", no_debug=True)
        self.assertEqual(five["branch_count"], 5)
        self.assertEqual(len(five["output_videos"]), 5)

    def test_run_tracker_supports_two_independent_surfaces(self):
        video_path = self.temp_dir / "two_surfaces.mp4"
        frame = _create_textured_image()
        for y in range(80, 260, 20):
            for x in range(360, 600, 20):
                color = (int((x + y) % 220), 70 + (x % 150), 210)
                cv2.rectangle(frame, (x, y), (x + 18, y + 18), color, -1)
                cv2.putText(frame, f"{x},{y}", (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
        for _ in range(3):
            writer.write(frame)
        writer.release()
        sign = self.temp_dir / "surface_sign.png"
        sign_img = np.full((100, 200, 4), (255, 0, 0, 255), dtype=np.uint8)
        cv2.putText(sign_img, "LOCAL", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255, 255), 2)
        cv2.imwrite(str(sign), sign_img)
        quad_a = np.float32([[100, 100], [300, 100], [300, 250], [100, 250]])
        quad_b = np.float32([[360, 100], [600, 100], [600, 250], [360, 250]])

        summary = run_tracker(video_path, quad_a, [sign], self.temp_dir / "two_surface_out", no_debug=True, surface_quads=[quad_a, quad_b])

        self.assertEqual(summary["surface_count"], 2)
        self.assertEqual(summary["branch_count"], 1)
        records = [json.loads(line) for line in (self.temp_dir / "two_surface_out" / "tracking.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records[0]["surface_details"]), 2)
        self.assertEqual(len(summary["output_videos"]), 1)

    def test_run_tracker_supports_two_hard_cut_scenes_with_stable_ids(self):
        """A scene reset must use that scene's source frame and keep the timeline."""
        video_path = self.temp_dir / "two_scenes.mp4"
        first = _create_textured_image()
        second = np.full_like(first, 95)
        # Give the second scene a separate, textured planar surface.
        for y in range(110, 270, 20):
            for x in range(350, 590, 20):
                color = (int((x + y) % 220), 70 + (x % 150), 210)
                cv2.rectangle(second, (x, y), (x + 18, y + 18), color, -1)
                cv2.putText(second, f"{x},{y}", (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
        for _ in range(3):
            writer.write(first)
        for _ in range(3):
            writer.write(second)
        writer.release()

        sign = self.temp_dir / "scene_sign.png"
        sign_img = np.full((100, 200, 4), (255, 0, 0, 255), dtype=np.uint8)
        cv2.imwrite(str(sign), sign_img)
        quad_a = np.float32([[100, 100], [300, 100], [300, 250], [100, 250]])
        quad_b = np.float32([[350, 110], [590, 110], [590, 270], [350, 270]])
        scene_plan = [
            {"scene_id": "scene_001", "start_frame": 0, "end_frame": 2,
             "surfaces": [{"surface_id": "surface_001", "quad": quad_a.tolist(),
                           "initialization_mode": "manual_fallback"}]},
            {"scene_id": "scene_002", "start_frame": 3, "end_frame": 5,
             "surfaces": [{"surface_id": "surface_002", "quad": quad_b.tolist(),
                           "initialization_mode": "reviewed_candidate"}]},
        ]

        summary = run_tracker(
            video_path, quad_a, [sign], self.temp_dir / "two_scene_out", no_debug=True,
            scene_plan=scene_plan,
        )

        self.assertEqual(summary["frames"], 6)
        self.assertEqual(summary["scene_count"], 2)
        self.assertEqual([item["scene_id"] for item in summary["scenes"]], ["scene_001", "scene_002"])
        self.assertEqual(len(summary["branch_scene_surface_map"]), 1)
        self.assertEqual(summary["branch_scene_surface_map"][0]["scenes"][1]["surface_ids"], ["surface_002"])
        records = [json.loads(line) for line in (self.temp_dir / "two_scene_out" / "tracking.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([records[i]["scene_id"] for i in (0, 2, 3, 5)], ["scene_001", "scene_001", "scene_002", "scene_002"])
        self.assertEqual(records[0]["surface_details"][0]["surface_id"], "surface_001")
        self.assertEqual(records[3]["surface_details"][0]["surface_id"], "surface_002")

    def test_run_tracker_rejects_non_contiguous_scene_plan(self):
        video_path, quad, signs, _ = self._create_synthetic_fixture(num_frames=4, blank_frame_idx=-1)
        scene_plan = [
            {"scene_id": "scene_001", "start_frame": 0, "end_frame": 1,
             "surfaces": [{"surface_id": "surface_001", "quad": quad.tolist()}]},
            # Frame 2 is intentionally missing from the plan.
            {"scene_id": "scene_002", "start_frame": 3, "end_frame": 3,
             "surfaces": [{"surface_id": "surface_002", "quad": quad.tolist()}]},
        ]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            run_tracker(video_path, quad, signs[:1], self.temp_dir / "invalid_scene_out", scene_plan=scene_plan)

    @patch("tracker.run_tracker", return_value={"branch_count": 1})
    def test_cli_accepts_one_sign_asset(self, run_mock):
        with patch("sys.argv", ["tracker.py", "--video", "video.mp4", "--quad-json", "quad.json", "--signs", "one.png", "--out-dir", "out"]):
            with patch("builtins.print"):
                main()
        self.assertEqual(run_mock.call_args.kwargs["signs"], [Path("one.png")])


    @patch("shutil.which", return_value=None)
    def test_ffmpeg_missing_raises_runtime_error(self, mock_which):
        video_path, quad, signs, _ = self._create_synthetic_fixture(num_frames=2, blank_frame_idx=-1)
        with self.assertRaises(RuntimeError) as ctx:
            run_tracker(video_path, quad, signs, self.temp_dir / "out")
        self.assertIn("FFmpeg is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
