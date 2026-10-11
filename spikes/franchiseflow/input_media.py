"""Input-video inspection and source normalization for FranchiseFlow.

Mobile editors commonly export MOV, M4V, WebM, or an MP4 carrying a codec that
the local OpenCV build cannot decode.  The tracker should receive one small,
deterministic working asset while the original upload remains available for
audit and re-processing.  This module keeps that conversion separate from
planar tracking so the input contract can be tested on its own.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2


SUPPORTED_SOURCE_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
# These codecs are frequent in phone/editor exports and are not guaranteed to
# be available in an OpenCV build.  A normal MPEG-4 Part 2 MP4 remains on the
# existing path for backwards compatibility with the current fixtures.
CODECS_REQUIRING_NORMALIZATION = {"hevc", "h265", "prores", "av1", "av01"}


def probe_video_codec(path: str | Path) -> dict[str, str]:
    """Read the first video stream's codec when ffprobe is available.

    A missing ffprobe is handled conservatively by returning an empty result;
    the existing OpenCV admission check will still reject an undecodable file.
    """
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return {}
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt",
            "-of", "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return {}
    try:
        import json

        streams = json.loads(result.stdout or "{}").get("streams", [])
        stream = streams[0] if streams else {}
    except (TypeError, ValueError, IndexError):
        return {}
    return {
        key: str(stream[key])
        for key in ("codec_name", "pix_fmt")
        if stream.get(key)
    }


def source_requires_normalization(path: str | Path) -> bool:
    """Return whether a source should be transcoded before OpenCV sees it."""
    source = Path(path)
    if source.suffix.lower() != ".mp4":
        return True
    # An unknown codec is unsafe to pass directly to OpenCV.  This includes a
    # missing ffprobe binary, a malformed probe response, and an unreadable
    # stream.  Normalizing may cost one extra encode, but it avoids silently
    # sending HEVC/AV1 MP4 files down a decoder-dependent path.
    probe = probe_video_codec(source)
    codec = probe.get("codec_name", "").lower()
    pixel_format = probe.get("pix_fmt", "").lower()
    return (
        not codec
        or codec in CODECS_REQUIRING_NORMALIZATION
        or (bool(pixel_format) and pixel_format not in {"yuv420p", "yuvj420p"})
    )


def inspect_video(path: str | Path) -> dict[str, Any]:
    """Return decoder-visible metadata and reject an unreadable source.

    ``CAP_PROP_FRAME_COUNT`` is a quick admission estimate only.  The scene
    sampler later performs a sequential decode and records the exact count.
    Keeping the estimate in the job record makes the input contract visible to
    the UI without pretending it is a frame-accurate measurement.
    """
    source = Path(path)
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise ValueError(f"unable to open video source: {source.name}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0.0 or frame_count <= 0.0 or width <= 0 or height <= 0:
        raise ValueError(f"unable to read video metadata: {source.name}")
    return {
        "fps": round(fps, 6),
        "frame_count_estimate": int(frame_count),
        "duration_seconds_estimate": round(frame_count / fps, 6),
        "width": width,
        "height": height,
        "container_extension": source.suffix.lower(),
    }


def normalize_video_for_processing(source: str | Path, destination: str | Path) -> Path:
    """Convert a source to browser/OpenCV-friendly H.264 MP4 with optional audio."""
    input_path = Path(source)
    output_path = Path(destination)
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError(
            "FFmpeg is required to normalize MOV/WebM and unsupported video sources but was not found in PATH."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-sn",
        "-dn",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown FFmpeg error").strip()
        raise RuntimeError(f"FFmpeg input normalization failed: {detail}")
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError("FFmpeg input normalization produced no working video")
    return output_path
