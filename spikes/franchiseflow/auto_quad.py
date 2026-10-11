"""Locate a storefront sign in one frame with the configured vision API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from vision_client import VisionClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    height, width = image.shape[:2]
    result = VisionClient().locate_sign(args.image.read_bytes(), width, height)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"quad": result["quad"], "confidence": result.get("confidence"), "model": result["model"]}, indent=2), encoding="utf-8")
    if args.preview:
        import numpy as np
        preview = image.copy()
        cv2.polylines(preview, [np.asarray(result["quad"], dtype=np.int32)], True, (0, 255, 255), 4)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), preview)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
