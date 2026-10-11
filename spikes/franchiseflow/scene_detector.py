"""Small, deterministic hard-cut detector for FranchiseFlow videos.

The detector deliberately handles only hard edits. Camera motion, exposure
changes, and short feature loss are left to the planar tracker. A cut is
reported only when both the resized-frame difference and the colour-histogram
change pass their thresholds. This keeps the scene layer conservative: a
missed cut can be sent to review, while a false cut would create two unrelated
tracking jobs from one continuous shot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class SceneSegment:
    """A frame-inclusive scene interval and the cut that starts it."""

    scene_id: str
    start_frame: int
    end_frame: int
    cut_before: int | None = None
    cut_score: float = 0.0

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)

    def to_dict(self) -> dict[str, int | float | str | None]:
        """Return a JSON-friendly representation for a job manifest."""
        return {
            "scene_id": self.scene_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "cut_before": self.cut_before,
            "cut_score": round(float(self.cut_score), 6),
        }


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame)
    if value.size == 0:
        raise ValueError("scene detector received an empty frame")
    if value.ndim == 2:
        value = cv2.cvtColor(value.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif value.ndim == 3 and value.shape[2] == 4:
        value = cv2.cvtColor(value.astype(np.uint8), cv2.COLOR_BGRA2BGR)
    elif value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("scene detector frames must be grayscale, BGR, or BGRA images")
    return value.astype(np.uint8, copy=False)


def _normalised_frame(frame: np.ndarray) -> np.ndarray:
    """Resize a frame to a small fixed representation for stable comparisons."""
    bgr = _as_bgr(frame)
    return cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def frame_difference_score(first: np.ndarray, second: np.ndarray) -> float:
    """Return mean absolute colour difference in the inclusive range [0, 1]."""
    a = _normalised_frame(first)
    b = _normalised_frame(second)
    return float(np.clip(np.mean(np.abs(a - b)) / 255.0, 0.0, 1.0))


def histogram_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Compare per-channel colour histograms, returning a value in [-1, 1]."""
    a = _as_bgr(first)
    b = _as_bgr(second)
    scores: list[float] = []
    for channel in range(3):
        hist_a = cv2.calcHist([a], [channel], None, [32], [0, 256])
        hist_b = cv2.calcHist([b], [channel], None, [32], [0, 256])
        cv2.normalize(hist_a, hist_a, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        cv2.normalize(hist_b, hist_b, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        scores.append(float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)))
    return float(np.clip(np.mean(scores), -1.0, 1.0))


def _cut_score(difference: float, correlation: float) -> float:
    # Difference and histogram change both contribute. Clamp correlation because
    # OpenCV can return a tiny amount outside the mathematical range.
    return float(np.clip(0.5 * difference + 0.5 * (1.0 - np.clip(correlation, -1.0, 1.0)), 0.0, 1.0))


def detect_scenes(
    frames: Sequence[tuple[int, np.ndarray]],
    *,
    difference_threshold: float = 0.30,
    histogram_threshold: float = 0.70,
    min_scene_frames: int = 2,
) -> list[SceneSegment]:
    """Split sequential decoded frames at conservative hard cuts.

    ``frames`` must be ordered by source frame index. The function does not
    decode a video or sample frames itself, which keeps it deterministic and
    straightforward to unit-test. A cut is accepted when the adjacent pair
    has a colour difference at least ``difference_threshold`` *and* histogram
    correlation at most ``histogram_threshold``. Segments shorter than
    ``min_scene_frames`` are merged with their predecessor (or successor for
    the first segment).
    """
    if not 0.0 <= difference_threshold <= 1.0:
        raise ValueError("difference_threshold must be between 0 and 1")
    if not -1.0 <= histogram_threshold <= 1.0:
        raise ValueError("histogram_threshold must be between -1 and 1")
    if min_scene_frames < 1:
        raise ValueError("min_scene_frames must be at least 1")
    if not frames:
        return []

    previous_index, previous_frame = frames[0]
    segments: list[SceneSegment] = []
    current_start_index = 0
    current_cut_before: int | None = None
    current_cut_score = 0.0

    for position in range(1, len(frames)):
        index, frame = frames[position]
        if index <= previous_index:
            raise ValueError("scene detector frame indices must be strictly increasing")
        difference = frame_difference_score(previous_frame, frame)
        correlation = histogram_correlation(previous_frame, frame)
        score = _cut_score(difference, correlation)
        is_cut = difference >= difference_threshold and correlation <= histogram_threshold
        if is_cut and position - current_start_index >= min_scene_frames:
            segments.append(SceneSegment(
                scene_id=f"scene_{len(segments) + 1:03d}",
                start_frame=frames[current_start_index][0],
                end_frame=previous_index,
                cut_before=current_cut_before,
                cut_score=current_cut_score,
            ))
            current_start_index = position
            current_cut_before = index
            current_cut_score = score
        previous_index, previous_frame = index, frame

    segments.append(SceneSegment(
        scene_id=f"scene_{len(segments) + 1:03d}",
        start_frame=frames[current_start_index][0],
        end_frame=frames[-1][0],
        cut_before=current_cut_before,
        cut_score=current_cut_score,
    ))

    # A final short segment is usually a transition or accidental flash. Merge
    # it into the previous scene, preserving the previous scene's identity.
    if len(segments) > 1 and segments[-1].frame_count < min_scene_frames:
        previous = segments[-2]
        segments[-2] = SceneSegment(
            scene_id=previous.scene_id,
            start_frame=previous.start_frame,
            end_frame=segments[-1].end_frame,
            cut_before=previous.cut_before,
            cut_score=previous.cut_score,
        )
        segments.pop()

    # Renumber after any merge so the manifest contains contiguous IDs.
    return [
        SceneSegment(
            scene_id=f"scene_{position:03d}",
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            cut_before=segment.cut_before,
            cut_score=segment.cut_score,
        )
        for position, segment in enumerate(segments, start=1)
    ]

