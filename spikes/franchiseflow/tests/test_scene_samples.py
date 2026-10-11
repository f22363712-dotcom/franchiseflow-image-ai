import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR.parent))

from scene_samples import sample_video_scenes  # noqa: E402


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV mp4v writer is unavailable")
    for frame in frames:
        writer.write(frame)
    writer.release()


class SceneSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ff_scene_samples_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_continuous_video_returns_one_scene_and_full_resolution_samples(self):
        frames = [np.full((48, 80, 3), 40 + index, dtype=np.uint8) for index in range(5)]
        path = self.temp_dir / "continuous.mp4"
        _write_video(path, frames)

        scenes = sample_video_scenes(path)

        self.assertEqual(len(scenes), 1)
        scene = scenes[0]
        self.assertEqual((scene["start_frame"], scene["end_frame"]), (0, 4))
        self.assertEqual(scene["total_frames"], 5)
        self.assertEqual((scene["width"], scene["height"]), (80, 48))
        self.assertEqual([index for index, _ in scene["samples"]], [0, 2, 4])
        self.assertTrue(all(frame.shape == (48, 80, 3) for _, frame in scene["samples"]))

    def test_hard_cut_returns_inclusive_scene_bounds(self):
        first = np.full((48, 80, 3), (20, 50, 100), dtype=np.uint8)
        second = np.full((48, 80, 3), (180, 220, 30), dtype=np.uint8)
        path = self.temp_dir / "hard_cut.mp4"
        _write_video(path, [first, first, second, second])

        scenes = sample_video_scenes(path, max_scenes=2)

        self.assertEqual([(scene["start_frame"], scene["end_frame"]) for scene in scenes], [(0, 1), (2, 3)])
        self.assertEqual([index for index, _ in scenes[0]["samples"]], [0, 1])
        self.assertEqual([index for index, _ in scenes[1]["samples"]], [2, 3])
        self.assertEqual([scene["cut_before"] for scene in scenes], [None, 2])

    def test_three_hard_cut_scenes_are_preserved_with_eight_scene_budget(self):
        colours = [(20, 50, 100), (180, 220, 30), (30, 180, 220)]
        frames = [np.full((48, 80, 3), colour, dtype=np.uint8) for colour in colours for _ in range(2)]
        path = self.temp_dir / "three_scenes.mp4"
        _write_video(path, frames)

        scenes = sample_video_scenes(path, max_scenes=8)

        self.assertEqual(len(scenes), 3)
        self.assertEqual([(scene["start_frame"], scene["end_frame"]) for scene in scenes], [(0, 1), (2, 3), (4, 5)])

    def test_rejects_video_with_more_scenes_than_allowed(self):
        colours = [
            (20, 50, 100),
            (180, 220, 30),
            (30, 180, 220),
        ]
        frames = [np.full((48, 80, 3), colour, dtype=np.uint8) for colour in colours for _ in range(2)]
        path = self.temp_dir / "too_many_scenes.mp4"
        _write_video(path, frames)

        with self.assertRaisesRegex(ValueError, r"exceeds max_scenes=2"):
            sample_video_scenes(path, max_scenes=2)

    def test_rejects_nine_scenes_with_eight_scene_budget(self):
        colours = [
            (20, 50, 100), (180, 220, 30), (30, 180, 220),
            (220, 40, 80), (40, 220, 100), (100, 40, 220),
            (220, 160, 20), (20, 180, 160), (160, 20, 180),
        ]
        frames = [np.full((48, 80, 3), colour, dtype=np.uint8) for colour in colours for _ in range(2)]
        path = self.temp_dir / "nine_scenes.mp4"
        _write_video(path, frames)

        with self.assertRaisesRegex(ValueError, r"exceeds max_scenes=8"):
            sample_video_scenes(path, max_scenes=8)

    def test_rejects_missing_video(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            sample_video_scenes(self.temp_dir / "missing.mp4")


if __name__ == "__main__":
    unittest.main()
