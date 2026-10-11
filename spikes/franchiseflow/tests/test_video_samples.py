import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_samples import sample_video_frames


class VideoSampleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ff_samples_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_returns_first_middle_and_last_decodable_frames(self):
        path = self.temp_dir / "sample.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
        for index in range(7):
            frame = np.full((64, 96, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        samples = sample_video_frames(path, count=3)
        self.assertEqual([index for index, _ in samples], [0, 3, 6])
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(frame.shape == (64, 96, 3) for _, frame in samples))

    def test_rejects_missing_video(self):
        with self.assertRaises(ValueError):
            sample_video_frames(self.temp_dir / "missing.mp4")


if __name__ == "__main__":
    unittest.main()
