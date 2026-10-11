import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

# The suite is also run as a discovery directory, so import the spike modules
# from their parent explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from input_media import inspect_video, normalize_video_for_processing, source_requires_normalization


class InputMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ff_input_media_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_inspect_video_returns_decoder_visible_metadata(self):
        source = self.temp_dir / "source.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 48))
        self.assertTrue(writer.isOpened())
        for value in (40, 80, 120):
            writer.write(np.full((48, 80, 3), value, dtype=np.uint8))
        writer.release()

        metadata = inspect_video(source)

        self.assertEqual(metadata["width"], 80)
        self.assertEqual(metadata["height"], 48)
        self.assertEqual(metadata["frame_count_estimate"], 3)
        self.assertAlmostEqual(metadata["duration_seconds_estimate"], 0.3, places=2)
        self.assertEqual(metadata["container_extension"], ".mp4")

    def test_normalize_requires_ffmpeg(self):
        with patch("input_media.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "FFmpeg"):
                normalize_video_for_processing(self.temp_dir / "input.mov", self.temp_dir / "working.mp4")

    def test_normalize_surfaces_ffmpeg_failure_without_claiming_success(self):
        source = self.temp_dir / "input.mov"
        source.write_bytes(b"source")
        result = type("Completed", (), {"returncode": 1, "stderr": "unsupported codec", "stdout": ""})()
        with patch("input_media.shutil.which", return_value="ffmpeg"), patch("input_media.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "normalization failed.*unsupported codec"):
                normalize_video_for_processing(source, self.temp_dir / "working.mp4")

    def test_normalize_command_maps_video_optional_audio_and_h264(self):
        source = self.temp_dir / "input.mov"
        source.write_bytes(b"source")
        destination = self.temp_dir / "working.mp4"
        result = type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

        def fake_run(command, **kwargs):
            destination.write_bytes(b"normalized")
            return result

        with patch("input_media.shutil.which", return_value="ffmpeg"), patch("input_media.subprocess.run", side_effect=fake_run) as run:
            output = normalize_video_for_processing(source, destination)

        self.assertEqual(output, destination)
        command = run.call_args.args[0]
        self.assertIn("-map", command)
        self.assertIn("0:v:0", command)
        self.assertIn("0:a?", command)
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)

    def test_non_mp4_source_requires_normalization(self):
        self.assertTrue(source_requires_normalization(self.temp_dir / "phone.mov"))

    def test_mp4_with_hevc_requires_normalization(self):
        with patch("input_media.probe_video_codec", return_value={"codec_name": "hevc", "pix_fmt": "yuv420p"}):
            self.assertTrue(source_requires_normalization(self.temp_dir / "phone.mp4"))

    def test_mp4_with_unknown_codec_requires_normalization(self):
        with patch("input_media.probe_video_codec", return_value={}):
            self.assertTrue(source_requires_normalization(self.temp_dir / "unknown.mp4"))

    def test_mp4_with_supported_codec_keeps_legacy_path(self):
        with patch("input_media.probe_video_codec", return_value={"codec_name": "mpeg4", "pix_fmt": "yuv420p"}):
            self.assertFalse(source_requires_normalization(self.temp_dir / "fixture.mp4"))

    def test_mp4_with_non_browser_pixel_format_requires_normalization(self):
        with patch("input_media.probe_video_codec", return_value={"codec_name": "h264", "pix_fmt": "yuv444p"}):
            self.assertTrue(source_requires_normalization(self.temp_dir / "wide_color.mp4"))


if __name__ == "__main__":
    unittest.main()
