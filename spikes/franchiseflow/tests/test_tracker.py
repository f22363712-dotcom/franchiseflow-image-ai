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

from metrics import evaluate_tracking_quality
from tracker import PlanarTracker, composite, main, normalize_branch_labels, probe_output_contract, project_quad, run_tracker, _normalize_scene_plan


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

    def test_quality_gate_requires_review_when_failed_fraction_is_high(self):
        gate = evaluate_tracking_quality(
            [{"frame": 0, "status": "ok"}, {"frame": 1, "status": "failed"}, {"frame": 2, "status": "failed"}],
            total_frames=3,
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["status"], "review")
        self.assertEqual(gate["reason"], "failed_frame_fraction_exceeds_threshold")

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
        self.assertTrue(all(item["validated"] for item in summary["output_validation"]))
        self.assertTrue(all(item["codec"] == "h264" and item["pix_fmt"] == "yuv420p" for item in summary["output_validation"]))
        self.assertTrue(all(len(item["sha256"]) == 64 for item in summary["output_validation"]))

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

    @patch("tracker.subprocess.run")
    @patch("tracker.shutil.which", return_value="ffprobe")
    def test_output_contract_rejects_wrong_codec(self, _which, run):
        output = self.temp_dir / "wrong.mp4"
        output.write_bytes(b"encoded")
        run.return_value = type("Completed", (), {
            "returncode": 0,
            "stdout": json.dumps({"streams": [{
                "codec_type": "video", "codec_name": "mpeg4", "pix_fmt": "yuv420p",
                "width": 320, "height": 240, "nb_read_frames": "2", "avg_frame_rate": "10/1",
            }]}),
            "stderr": "",
        })()
        with self.assertRaisesRegex(RuntimeError, "output validation failed.*codec=mpeg4"):
            probe_output_contract(
                output,
                expected_frames=2,
                expected_width=320,
                expected_height=240,
                expected_fps=10.0,
                require_audio=False,
            )

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

    def test_branch_labels_are_preserved_in_outputs_and_manifest(self):
        video_path, quad, signs, _ = self._create_synthetic_fixture(num_frames=2, blank_frame_idx=-1)
        summary = run_tracker(
            video_path,
            quad,
            signs[:2],
            self.temp_dir / "named_branches",
            no_debug=True,
            branch_labels=["滨河路店", "滨河路店"],
        )
        self.assertEqual(summary["branch_labels"], ["滨河路店", "滨河路店_2"])
        self.assertEqual([Path(path).name for path in summary["output_videos"]], [
            "branch_1_滨河路店.mp4", "branch_2_滨河路店_2.mp4",
        ])

    def test_branch_labels_reject_wrong_count(self):
        with self.assertRaisesRegex(ValueError, "exactly 2 labels"):
            normalize_branch_labels(["only_one"], ["a.png", "b.png"])

    def test_contain_fit_preserves_aspect_inside_target_quad(self):
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        sign = np.full((20, 80, 4), (0, 0, 255, 255), dtype=np.uint8)
        quad = np.float32([[20, 20], [100, 20], [100, 100], [20, 100]])
        contained = composite(frame, sign, quad, fit="contain")
        covered = composite(frame, sign, quad, fit="cover")
        self.assertLess(int(contained[22, 60, 2]), 20)
        self.assertGreater(int(contained[60, 60, 2]), 180)
        self.assertGreater(int(covered[22, 60, 2]), 180)

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

        summary = run_tracker(
            video_path,
            quad_a,
            [sign],
            self.temp_dir / "two_surface_out",
            no_debug=True,
            surface_quads=[quad_a, quad_b],
            asset_bindings=[{"branch_index": 1, "surface_ids": ["surface_001", "surface_002"], "fit": "cover"}],
        )

        self.assertEqual(summary["surface_count"], 2)
        self.assertEqual(summary["branch_count"], 1)
        records = [json.loads(line) for line in (self.temp_dir / "two_surface_out" / "tracking.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records[0]["surface_details"]), 2)
        self.assertEqual(len(summary["output_videos"]), 1)

    def test_asset_bindings_render_each_branch_only_on_owned_surface(self):
        video_path = self.temp_dir / "bound_surfaces.mp4"
        frame = _create_textured_image()
        for y in range(80, 260, 20):
            for x in range(360, 600, 20):
                color = (int((x + y) % 220), 70 + (x % 150), 210)
                cv2.rectangle(frame, (x, y), (x + 18, y + 18), color, -1)
                cv2.putText(frame, f"{x},{y}", (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
        writer.write(frame)
        writer.write(frame)
        writer.release()
        sign_one = self.temp_dir / "sign_one.png"
        sign_two = self.temp_dir / "sign_two.png"
        cv2.imwrite(str(sign_one), np.full((100, 200, 4), (0, 0, 255, 255), dtype=np.uint8))
        cv2.imwrite(str(sign_two), np.full((100, 200, 4), (0, 255, 0, 255), dtype=np.uint8))
        quad_a = np.float32([[100, 100], [300, 100], [300, 250], [100, 250]])
        quad_b = np.float32([[360, 100], [600, 100], [600, 250], [360, 250]])

        summary = run_tracker(
            video_path,
            quad_a,
            [sign_one, sign_two],
            self.temp_dir / "bound_surface_out",
            no_debug=True,
            surface_quads=[quad_a, quad_b],
            asset_bindings=[
                {"branch_index": 1, "surface_ids": ["surface_001"], "fit": "cover"},
                {"branch_index": 2, "surface_ids": ["surface_002"], "fit": "cover"},
            ],
        )
        self.assertEqual(summary["asset_bindings"][0]["surface_ids"], ["surface_001"])
        self.assertEqual(summary["branch_scene_surface_map"][0]["scenes"][0]["surface_ids"], ["surface_001"])
        self.assertEqual(summary["branch_scene_surface_map"][1]["scenes"][0]["surface_ids"], ["surface_002"])
        output_one = cv2.VideoCapture(str(self.temp_dir / "bound_surface_out" / "branch_1_sign_one.mp4"))
        output_two = cv2.VideoCapture(str(self.temp_dir / "bound_surface_out" / "branch_2_sign_two.mp4"))
        ok_one, frame_one = output_one.read()
        ok_two, frame_two = output_two.read()
        output_one.release()
        output_two.release()
        self.assertTrue(ok_one and ok_two)
        center_a = (175, 200)
        center_b = (175, 480)
        original_a = frame[center_a]
        original_b = frame[center_b]
        self.assertGreater(int(frame_one[center_a][2]), 180)
        self.assertGreater(int(frame_two[center_b][1]), 180)
        self.assertLess(float(np.linalg.norm(frame_one[center_b].astype(float) - original_b)), 45.0)
        self.assertLess(float(np.linalg.norm(frame_two[center_a].astype(float) - original_a)), 45.0)

        shared = run_tracker(
            video_path,
            quad_a,
            [sign_one, sign_two],
            self.temp_dir / "shared_surface_out",
            no_debug=True,
            surface_quads=[quad_a, quad_b],
            asset_bindings=[
                {"branch_index": 1, "surface_ids": ["surface_001"], "fit": "cover"},
                {"branch_index": 2, "surface_ids": ["surface_001"], "fit": "cover"},
            ],
        )
        self.assertEqual(shared["branch_scene_surface_map"][0]["scenes"][0]["surface_ids"], ["surface_001"])
        self.assertEqual(shared["branch_scene_surface_map"][1]["scenes"][0]["surface_ids"], ["surface_001"])

    def test_multi_surface_requires_non_overlapping_explicit_bindings(self):
        video_path, quad, signs, _ = self._create_synthetic_fixture(num_frames=2, blank_frame_idx=-1)
        quad_b = np.float32([[230, 100], [390, 100], [390, 220], [230, 220]])
        with self.assertRaisesRegex(ValueError, "asset_bindings are required"):
            run_tracker(video_path, quad, signs[:1], self.temp_dir / "missing_bindings", no_debug=True, surface_quads=[quad, quad_b])
        with self.assertRaisesRegex(ValueError, "repeats a surface_id"):
            run_tracker(
                video_path,
                quad,
                signs[:2],
                self.temp_dir / "duplicate_bindings",
                no_debug=True,
                surface_quads=[quad, quad_b],
                asset_bindings=[
                    {"branch_index": 1, "surface_ids": ["surface_001", "surface_001"]},
                    {"branch_index": 2, "surface_ids": ["surface_001"]},
                ],
            )
        with self.assertRaisesRegex(ValueError, "unknown surface"):
            run_tracker(
                video_path,
                quad,
                signs[:1],
                self.temp_dir / "unknown_binding",
                no_debug=True,
                surface_quads=[quad, quad_b],
                asset_bindings=[{"branch_index": 1, "surface_ids": ["missing_surface"]}],
            )

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

    def test_scene_plan_accepts_three_scenes_and_rejects_nine(self):
        quad = [[100, 100], [300, 100], [300, 250], [100, 250]]
        scenes = [
            {"scene_id": f"scene_{index + 1:03d}", "start_frame": index * 2, "end_frame": index * 2 + 1,
             "surfaces": [{"surface_id": f"surface_{index + 1:03d}", "quad": quad}]}
            for index in range(3)
        ]
        normalized = _normalize_scene_plan(scenes)
        self.assertEqual(len(normalized), 3)
        self.assertEqual(normalized[-1]["end_frame"], 5)
        too_many = [
            {"scene_id": f"scene_{index:03d}", "start_frame": index - 1, "end_frame": index - 1,
             "surfaces": [{"surface_id": f"surface_{index:03d}", "quad": quad}]}
            for index in range(1, 10)
        ]
        with self.assertRaisesRegex(ValueError, "between 1 and 8 scenes"):
            _normalize_scene_plan(too_many)

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
