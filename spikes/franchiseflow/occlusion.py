"""Conservative foreground-preservation hints for planar replacements.

This helper aligns a current frame back to the anchor plane and looks for a
large, localized photometric residual. It can preserve the original pixels in
that region when confidence is high. It is deliberately not a general video
matting model: uncertain or fragmented residuals return an empty mask.
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_occlusion_mask(
    anchor: np.ndarray,
    current: np.ndarray,
    homography: np.ndarray,
    quad: np.ndarray,
    *,
    min_component_fraction: float = 0.015,
    min_residual_fraction: float = 0.03,
) -> dict:
    """Return a current-frame mask for a high-confidence foreground hint."""
    if anchor.ndim != 3 or current.ndim != 3 or anchor.shape != current.shape:
        raise ValueError("anchor and current frames must have the same BGR shape")
    if homography is None or np.asarray(homography).shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("homography must be a finite 3x3 matrix")
    if not 0.0 < min_component_fraction <= 1.0:
        raise ValueError("min_component_fraction must be between 0 and 1")
    if not 0.0 < min_residual_fraction <= 1.0:
        raise ValueError("min_residual_fraction must be between 0 and 1")

    height, width = anchor.shape[:2]
    try:
        inverse = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise ValueError("homography is singular") from exc
    aligned = cv2.warpPerspective(current, inverse, (width, height), flags=cv2.INTER_LINEAR)
    plane = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(plane, np.round(np.asarray(quad, dtype=np.float32)).astype(np.int32), 255)
    interior = cv2.erode(plane, np.ones((9, 9), np.uint8)) > 0
    interior_area = int(np.count_nonzero(interior))
    if interior_area == 0:
        return {"status": "none", "mask": np.zeros((height, width), np.uint8), "area_fraction": 0.0}

    anchor_gray = cv2.GaussianBlur(cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY), (0, 0), 1.5).astype(np.float32)
    aligned_gray = cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), (0, 0), 1.5).astype(np.float32)
    delta = aligned_gray - anchor_gray
    values = delta[interior]
    offset = float(np.median(values))
    residual = np.abs(delta - offset)
    residual_values = residual[interior]
    median = float(np.median(residual_values))
    mad = float(np.median(np.abs(residual_values - median))) + 1.0
    threshold = max(28.0, median + 5.0 * mad)
    candidate = ((residual > threshold) & interior).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    filtered = np.zeros_like(candidate)
    minimum_area = max(200, int(interior_area * min_component_fraction))
    largest_area = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= minimum_area:
            filtered[labels == label] = 255
            largest_area = max(largest_area, area)
    area_fraction = float(np.count_nonzero(filtered) / interior_area)
    residual_fraction = float(np.count_nonzero(candidate) / interior_area)
    if largest_area < minimum_area or area_fraction < min_residual_fraction:
        return {
            "status": "none",
            "mask": np.zeros((height, width), np.uint8),
            "area_fraction": round(area_fraction, 4),
            "residual_fraction": round(residual_fraction, 4),
            "threshold": round(threshold, 3),
        }

    # Map the anchor-space mask back into the current frame. Nearest-neighbour
    # interpolation keeps the preservation region binary.
    current_mask = cv2.warpPerspective(filtered, np.asarray(homography, dtype=np.float32), (width, height), flags=cv2.INTER_NEAREST)
    current_mask = cv2.dilate(current_mask, np.ones((3, 3), np.uint8))
    return {
        "status": "occlusion_hint",
        "mask": current_mask,
        "area_fraction": round(area_fraction, 4),
        "residual_fraction": round(residual_fraction, 4),
        "threshold": round(threshold, 3),
    }
