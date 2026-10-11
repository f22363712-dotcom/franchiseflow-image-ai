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
    transition_type: str = "none"

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
            "transition_type": self.transition_type,
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
    soft_window: int = 3,
    soft_difference_threshold: float = 0.25,
    soft_histogram_threshold: float = 0.25,
) -> list[SceneSegment]:
    """Split frames at hard cuts and conservative dissolve boundaries.

    A hard cut uses adjacent-frame scores. A dissolve is considered only when
    frames on both sides of a short window differ strongly, the intervening
    changes remain gradual, and the endpoint histograms are clearly different.
    This gives downstream tracking a fresh anchor while preserving a marker in
    the scene manifest so low-confidence transitions can still be reviewed.
    """
    if not 0.0 <= difference_threshold <= 1.0:
        raise ValueError("difference_threshold must be between 0 and 1")
    if not -1.0 <= histogram_threshold <= 1.0:
        raise ValueError("histogram_threshold must be between -1 and 1")
    if min_scene_frames < 1:
        raise ValueError("min_scene_frames must be at least 1")
    if soft_window < 2:
        raise ValueError("soft_window must be at least 2")
    if not 0.0 <= soft_difference_threshold <= 1.0:
        raise ValueError("soft_difference_threshold must be between 0 and 1")
    if not -1.0 <= soft_histogram_threshold <= 1.0:
        raise ValueError("soft_histogram_threshold must be between -1 and 1")
    if not frames:
        return []

    for position in range(1, len(frames)):
        if frames[position][0] <= frames[position - 1][0]:
            raise ValueError("scene detector frame indices must be strictly increasing")
    if len(frames) == 1:
        return [SceneSegment(scene_id="scene_001", start_frame=frames[0][0], end_frame=frames[0][0])]

    pair_metrics: list[tuple[float, float, float]] = []
    for position in range(1, len(frames)):
        difference = frame_difference_score(frames[position - 1][1], frames[position][1])
        correlation = histogram_correlation(frames[position - 1][1], frames[position][1])
        pair_metrics.append((difference, correlation, _cut_score(difference, correlation)))

    hard_boundaries = [
        (position, pair_metrics[position - 1][2], "none")
        for position in range(1, len(frames))
        if pair_metrics[position - 1][0] >= difference_threshold and
        pair_metrics[position - 1][1] <= histogram_threshold
    ]
    hard_positions = {item[0] for item in hard_boundaries}

    soft_candidates: list[tuple[int, float]] = []
    if len(frames) >= soft_window * 2:
        for position in range(soft_window, len(frames) - soft_window + 1):
            if any(abs(position - hard_position) <= soft_window for hard_position in hard_positions):
                continue
            before = frames[position - soft_window][1]
            after = frames[position + soft_window - 1][1]
            endpoint_difference = frame_difference_score(before, after)
            endpoint_correlation = histogram_correlation(before, after)
            local = pair_metrics[position - soft_window:position + soft_window - 1]
            local_difference = float(np.median([item[0] for item in local])) if local else 1.0
            if (endpoint_difference >= soft_difference_threshold and
                    endpoint_correlation <= soft_histogram_threshold and
                    local_difference < difference_threshold):
                soft_candidates.append((position, _cut_score(endpoint_difference, endpoint_correlation)))

    # Keep one strongest candidate per transition and never place it next to a
    # hard cut. A gradual exposure shift therefore remains one scene.
    accepted_soft: list[tuple[int, float, str]] = []
    for position, score in sorted(soft_candidates, key=lambda item: item[1], reverse=True):
        if any(abs(position - other[0]) < soft_window for other in accepted_soft):
            continue
        accepted_soft.append((position, score, "dissolve"))
    boundaries = sorted(hard_boundaries + accepted_soft, key=lambda item: item[0])

    segments: list[SceneSegment] = []
    current_start_index = 0
    current_cut_before: int | None = None
    current_cut_score = 0.0
    current_transition = "none"
    for position, score, transition_type in boundaries:
        if position - current_start_index < min_scene_frames:
            continue
        segments.append(SceneSegment(
            scene_id=f"scene_{len(segments) + 1:03d}",
            start_frame=frames[current_start_index][0],
            end_frame=frames[position - 1][0],
            cut_before=current_cut_before,
            cut_score=current_cut_score,
            transition_type=current_transition,
        ))
        current_start_index = position
        current_cut_before = frames[position][0]
        current_cut_score = score
        current_transition = transition_type

    segments.append(SceneSegment(
        scene_id=f"scene_{len(segments) + 1:03d}",
        start_frame=frames[current_start_index][0],
        end_frame=frames[-1][0],
        cut_before=current_cut_before,
        cut_score=current_cut_score,
        transition_type=current_transition,
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
            transition_type=previous.transition_type,
        )
        segments.pop()

    return [
        SceneSegment(
            scene_id=f"scene_{position:03d}",
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            cut_before=segment.cut_before,
            cut_score=segment.cut_score,
            transition_type=segment.transition_type,
        )
        for position, segment in enumerate(segments, start=1)
    ]

