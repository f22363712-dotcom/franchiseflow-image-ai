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

    @patch("tracker.shutil.which", side_effect=["ffmpeg", None])
    def test_run_tracker_does_not_silently_drop_audio_when_ffprobe_missing(self, which):
        import tempfile
        import cv2
        import numpy as np
        from tracker import run_tracker
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
            frame = np.full((480, 640, 3), 128, dtype=np.uint8)
            for y in range(80, 260, 20):
                for x in range(80, 320, 20):
                    cv2.rectangle(frame, (x, y), (x + 18, y + 18), (40 + (x + y) % 200, 180, 60), -1)
                    cv2.putText(frame, f"{x},{y}", (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            writer.write(frame)
            writer.release()
            sign = root / "sign.png"
            cv2.imwrite(str(sign), np.zeros((20, 20, 4), dtype=np.uint8))
            with self.assertRaises(RuntimeError) as ctx:
                run_tracker(video, [[100, 100], [300, 100], [300, 250], [100, 250]], [sign], root / "out", no_debug=True)
            self.assertIn("ffprobe", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
