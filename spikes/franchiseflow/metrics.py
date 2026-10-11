"""Tracking evaluation and summary metrics for FranchiseFlow."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def compute_summary(
    records: list[dict[str, Any]],
    fps: float,
    total_frames: int,
    elapsed_seconds: float,
    video_path: str | Path,
    quad: list[list[float]] | np.ndarray,
    output_videos: list[str | Path],
    mode: str = "manual_fallback",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tracked = [r for r in records[1:] if r.get("status") == "ok"]
    failed = [r["frame"] for r in records if r.get("status") == "failed"]
    quad_list = quad.tolist() if isinstance(quad, np.ndarray) else quad

    summary: dict[str, Any] = {
        "video": str(Path(video_path).resolve()),
        "initialization_mode": mode,
        "first_frame_quad": quad_list,
        "scope": "planar sign tracking and multi-branch perspective compositing; safe original frame fallback",
        "metrics_warning": "Reprojection residuals are internal fit indicators, not ground-truth corner accuracy or visual proof.",
        "frames": total_frames,
        "fps": fps,
        "duration_seconds": round(total_frames / max(fps, 1e-3), 2),
        "failed_frames": failed,
        "failed_fraction": round(len(failed) / max(total_frames - 1, 1), 4),
        "anchor_sift_frames": sum(r.get("method") == "anchor_sift" for r in tracked),
        "anchor_lk_frames": sum(r.get("method") == "anchor_lk" for r in tracked),
        "first_frame_sift_features": records[0].get("inliers", 0) if records else 0,
        "median_inliers": float(np.median([r["inliers"] for r in tracked])) if tracked and any("inliers" in r for r in tracked) else None,
        "reprojection_median_px": float(np.median([r["reprojection_median_px"] for r in tracked])) if tracked and any("reprojection_median_px" in r for r in tracked) else None,
        "reprojection_p95_px": float(np.percentile([r["reprojection_p95_px"] for r in tracked], 95)) if tracked and any("reprojection_p95_px" in r for r in tracked) else None,
        "processing_seconds": round(elapsed_seconds, 2),
        "output_videos": [str(Path(p).resolve()) for p in output_videos],
    }
    if extra:
        summary.update(extra)
    return summary
