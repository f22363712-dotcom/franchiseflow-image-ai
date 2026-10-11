"""Conservative transfer of a verified planar candidate between scene anchors."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _valid_quad(quad: np.ndarray, width: int, height: int) -> bool:
    points = np.asarray(quad, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False
    if not cv2.isContourConvex(np.round(points).astype(np.int32)):
        return False
    if abs(cv2.contourArea(points)) < max(100.0, width * height * 0.0005):
        return False
    return bool(
        np.all(points[:, 0] >= 0)
        and np.all(points[:, 0] < width)
        and np.all(points[:, 1] >= 0)
        and np.all(points[:, 1] < height)
    )


def transfer_candidate(
    anchor_frame: np.ndarray,
    target_frame: np.ndarray,
    anchor_quad: list | np.ndarray,
    *,
    min_inliers: int = 12,
    ratio: float = 0.75,
    ransac_px: float = 3.0,
) -> dict[str, Any] | None:
    """Return a projected candidate only when cross-scene geometry is strong.

    The anchor quad is masked before feature extraction, while the target frame
    is searched globally. This lets a sign move between shots without trusting
    the old pixel coordinates. Any weak or degenerate transfer returns ``None``
    so the caller can ask the vision model for a fresh candidate.
    """
    anchor = np.asarray(anchor_quad, dtype=np.float32)
    if anchor.shape != (4, 2) or not np.isfinite(anchor).all():
        return None
    anchor_height, anchor_width = anchor_frame.shape[:2]
    target_height, target_width = target_frame.shape[:2]
    if not _valid_quad(anchor, anchor_width, anchor_height):
        return None

    anchor_gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(anchor_gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(anchor).astype(np.int32), 255)
    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.015, edgeThreshold=15)
    anchor_keypoints, anchor_descriptors = sift.detectAndCompute(anchor_gray, mask)
    target_keypoints, target_descriptors = sift.detectAndCompute(target_gray, None)
    if anchor_descriptors is None or target_descriptors is None:
        return None
    if len(anchor_keypoints) < min_inliers or len(target_keypoints) < min_inliers:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(anchor_descriptors, target_descriptors, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    if len(good) < min_inliers:
        return None
    source = np.asarray([anchor_keypoints[item.queryIdx].pt for item in good], dtype=np.float32)
    destination = np.asarray([target_keypoints[item.trainIdx].pt for item in good], dtype=np.float32)
    matrix, inlier_mask = cv2.findHomography(source, destination, cv2.RANSAC, ransac_px)
    if matrix is None or inlier_mask is None:
        return None
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    if inlier_count < min_inliers:
        return None
    projected = cv2.perspectiveTransform(anchor.reshape(-1, 1, 2), matrix).reshape(4, 2)
    if not _valid_quad(projected, target_width, target_height):
        return None
    projected_source = cv2.perspectiveTransform(source[inliers].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residual = np.linalg.norm(projected_source - destination[inliers], axis=1)
    median_residual = float(np.median(residual)) if len(residual) else float("inf")
    p95_residual = float(np.percentile(residual, 95)) if len(residual) else float("inf")
    if not np.isfinite(median_residual) or median_residual > ransac_px * 1.5:
        return None
    confidence = min(0.99, inlier_count / max(len(good), 1))
    return {
        "quad": projected.round(3).tolist(),
        "confidence": round(float(confidence), 6),
        "reason": "validated_cross_scene_transfer",
        "transfer": {
            "good_matches": len(good),
            "inliers": inlier_count,
            "median_residual_px": round(median_residual, 4),
            "p95_residual_px": round(p95_residual, 4),
        },
    }
