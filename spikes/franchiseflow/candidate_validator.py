"""Multi-frame geometric checks for model-proposed planar sign candidates."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _valid_quad(quad: Any, width: int, height: int) -> bool:
    points = np.asarray(quad, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False
    if not cv2.isContourConvex(np.round(points).astype(np.int32)):
        return False
    if abs(cv2.contourArea(points)) < max(100.0, width * height * 0.0005):
        return False
    return bool(np.all(points[:, 0] >= 0) and np.all(points[:, 0] < width) and
                np.all(points[:, 1] >= 0) and np.all(points[:, 1] < height))


def _texture_points(frame: np.ndarray, quad: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.015, edgeThreshold=15)
    return sift.detectAndCompute(gray, mask)


def _project_from_anchor(first_frame: np.ndarray, sample: np.ndarray, quad: np.ndarray, min_inliers: int) -> tuple[np.ndarray | None, int]:
    key0, desc0 = _texture_points(first_frame, quad)
    if desc0 is None or len(key0) < min_inliers:
        return None, len(key0)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.015, edgeThreshold=15)
    key, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return None, 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc0, desc, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < min_inliers:
        return None, len(good)
    src = np.asarray([key0[m.queryIdx].pt for m in good], np.float32)
    dst = np.asarray([key[m.trainIdx].pt for m in good], np.float32)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or mask is None:
        return None, 0
    inliers = int(mask.ravel().sum())
    if inliers < min_inliers:
        return None, inliers
    projected = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), matrix).reshape(4, 2)
    return projected, inliers


def validate_candidate(samples: list[tuple[int, np.ndarray]], candidate: dict[str, Any], min_texture_points: int = 8) -> dict[str, Any]:
    if not samples:
        return {"accepted": False, "requires_human_review": False, "reasons": ["no_samples"], "sample_metrics": []}
    first = samples[0][1]
    height, width = first.shape[:2]
    try:
        quad = np.asarray(candidate.get("quad"), dtype=np.float32)
    except (AttributeError, TypeError, ValueError):
        quad = np.empty((0, 2), np.float32)
    if not _valid_quad(quad, width, height):
        return {"accepted": False, "requires_human_review": False, "reasons": ["invalid_quad"], "sample_metrics": []}

    _, first_desc = _texture_points(first, quad)
    first_count = 0 if first_desc is None else len(first_desc)
    metrics: list[dict[str, Any]] = [{
        "frame_index": samples[0][0], "texture_points": first_count,
        "quad": quad.round(3).tolist(), "bounds_ok": True,
        "inliers": first_count, "status": "ok" if first_count >= min_texture_points else "insufficient_texture",
    }]
    reasons: list[str] = []
    if first_count < min_texture_points:
        reasons.append("insufficient_texture")

    for frame_index, frame in samples[1:]:
        projected, inliers = _project_from_anchor(first, frame, quad, min_texture_points)
        if projected is None:
            metrics.append({"frame_index": frame_index, "texture_points": 0, "inliers": inliers, "bounds_ok": False, "status": "inconsistent_samples"})
            reasons.append("inconsistent_samples")
            continue
        sample_height, sample_width = frame.shape[:2]
        bounds_ok = _valid_quad(projected, sample_width, sample_height)
        metrics.append({"frame_index": frame_index, "texture_points": inliers, "inliers": inliers, "bounds_ok": bounds_ok, "quad": projected.round(3).tolist(), "status": "ok" if bounds_ok else "quad_out_of_bounds"})
        if not bounds_ok:
            reasons.append("quad_out_of_bounds")

    reasons = list(dict.fromkeys(reasons))
    return {
        "accepted": not reasons,
        "requires_human_review": bool(reasons) and "invalid_quad" not in reasons,
        "reasons": reasons,
        "sample_metrics": metrics,
    }
