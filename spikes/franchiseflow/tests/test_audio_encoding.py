import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracker import transcode_to_h264


class AudioEncodingTests(unittest.TestCase):
    @patch("tracker.subprocess.run")
    @patch("tracker.shutil.which", return_value="ffmpeg")
    def test_transcode_maps_optional_source_audio(self, _which, run):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        transcode_to_h264(Path("raw.mp4"), Path("out.mp4"), audio_source=Path("source.mp4"))
        command = run.call_args.args[0]
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)
        self.assertIn("-map", command)
        self.assertIn("1:a?", command)
        self.assertIn("aac", command)

    @patch("tracker.subprocess.run")
    @patch("tracker.shutil.which", return_value="ffmpeg")
    def test_transcode_without_source_audio_keeps_video_only(self, _which, run):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        transcode_to_h264(Path("raw.mp4"), Path("out.mp4"))
        command = run.call_args.args[0]
        self.assertIn("-an", command)
        self.assertNotIn("1:a?", command)


if __name__ == "__main__":
    unittest.main()
