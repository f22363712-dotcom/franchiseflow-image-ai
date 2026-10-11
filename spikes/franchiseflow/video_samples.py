"""Small, deterministic video sampling helpers used before tracking."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sample_video_frames(video: str | Path, count: int = 3) -> list[tuple[int, np.ndarray]]:
    path = Path(video)
    if not path.exists():
        raise ValueError(f"video does not exist: {path}")
    if count < 1:
        raise ValueError("sample count must be positive")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"cannot open video: {path}")
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if reported <= 0:
        cap.release()
        raise ValueError(f"video has no readable frame count: {path}")

    indices = sorted(set(round(i * (reported - 1) / max(count - 1, 1)) for i in range(count)))
    samples: list[tuple[int, np.ndarray]] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if ok and frame is not None:
            samples.append((index, frame))
    cap.release()
    if not samples:
        raise ValueError(f"cannot decode any sample frame from {path}")
    return samples
