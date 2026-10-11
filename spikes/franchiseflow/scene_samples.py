"""Video decoding helpers for scene-aware FranchiseFlow jobs.

The first pass decodes the source sequentially and keeps only 64x64 frames for
the hard-cut detector. Once scene boundaries are known, a second sequential
pass copies the original-resolution start, middle, and end frames for each
scene. The source frame count is therefore the number of frames that were
actually decoded, rather than a container metadata estimate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scene_detector import SceneSegment, detect_scenes


def _open_capture(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"cannot open video: {path}")
    return cap


def _validate_frame(frame: Any, *, path: Path, frame_index: int) -> np.ndarray:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError(f"cannot decode frame {frame_index} from video: {path}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"video frame {frame_index} is not a BGR image: {path}")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError(f"video frame {frame_index} has invalid dimensions: {path}")
    return frame


def _sample_indices(segment: SceneSegment) -> tuple[int, ...]:
    """Return unique, inclusive start/middle/end source indices."""
    middle = (segment.start_frame + segment.end_frame) // 2
    return tuple(sorted({segment.start_frame, middle, segment.end_frame}))


def sample_video_scenes(video_path: str | Path, *, max_scenes: int = 8) -> list[dict[str, Any]]:
    """Detect hard-cut scenes and return original-resolution keyframes.

    ``samples`` contains ``(global_frame_index, BGR_frame)`` tuples. Scene
    bounds are inclusive. The decoder never trusts ``CAP_PROP_FRAME_COUNT``;
    ``total_frames`` is the exact number of frames successfully read during
    the first pass. ``max_scenes`` is an explicit safety budget for hard-cut
    segmentation. The web product supplies its own configured limit; callers
    using this lower-level helper can choose a smaller budget for lightweight
    jobs.
    """
    path = Path(video_path)
    if not path.exists():
        raise ValueError(f"video does not exist: {path}")
    if not isinstance(max_scenes, int) or isinstance(max_scenes, bool) or max_scenes < 1:
        raise ValueError("max_scenes must be at least 1")

    cap: cv2.VideoCapture | None = None
    try:
        cap = _open_capture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"video has invalid FPS: {path}")

        small_frames: list[tuple[int, np.ndarray]] = []
        width: int | None = None
        height: int | None = None
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bgr = _validate_frame(frame, path=path, frame_index=frame_index)
            current_height, current_width = bgr.shape[:2]
            if width is None:
                width, height = int(current_width), int(current_height)
            elif (current_width, current_height) != (width, height):
                raise ValueError(f"video frame dimensions changed at frame {frame_index}: {path}")
            # Keep only the compact representation needed for cut scoring.
            small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
            small_frames.append((frame_index, small))
            frame_index += 1
    finally:
        if cap is not None:
            cap.release()

    total_frames = len(small_frames)
    if total_frames == 0 or width is None or height is None:
        raise ValueError(f"cannot decode any frame from video: {path}")

    scenes = detect_scenes(small_frames)
    if len(scenes) > max_scenes:
        raise ValueError(f"video has {len(scenes)} scenes, exceeds max_scenes={max_scenes}")

    target_indices = {
        frame_index
        for scene in scenes
        for frame_index in _sample_indices(scene)
    }
    sampled_frames: dict[int, np.ndarray] = {}
    cap = None
    try:
        cap = _open_capture(path)
        second_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(second_fps) or second_fps <= 0.0:
            raise ValueError(f"video has invalid FPS on sample pass: {path}")
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bgr = _validate_frame(frame, path=path, frame_index=frame_index)
            if bgr.shape[:2] != (height, width):
                raise ValueError(f"video frame dimensions changed at frame {frame_index}: {path}")
            if frame_index in target_indices:
                sampled_frames[frame_index] = bgr.copy()
            frame_index += 1
        if frame_index != total_frames:
            raise ValueError(
                f"video decode count changed between passes: {total_frames} then {frame_index}: {path}"
            )
    finally:
        if cap is not None:
            cap.release()

    missing = sorted(target_indices.difference(sampled_frames))
    if missing:
        raise ValueError(f"cannot decode requested scene sample frames {missing}: {path}")

    metadata = {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
    }
    result: list[dict[str, Any]] = []
    for scene in scenes:
        result.append({
            "scene_id": scene.scene_id,
            "start_frame": scene.start_frame,
            "end_frame": scene.end_frame,
            "cut_before": scene.cut_before,
            "cut_score": scene.cut_score,
            "samples": [
                (index, sampled_frames[index])
                for index in _sample_indices(scene)
            ],
            **metadata,
        })
    return result
