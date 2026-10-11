"""Locate a storefront sign in one frame with the configured vision API."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from vision_client import VisionClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate a storefront sign with vision API")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--target-hint", type=str, default=None, help="Optional text description of target sign")
    args = parser.parse_args()

    if not args.image.exists():
        sys.stderr.write(f"Error: image file does not exist: {args.image}\n")
        sys.exit(1)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        sys.stderr.write(f"Error: cannot decode image: {args.image}\n")
        sys.exit(1)

    height, width = image.shape[:2]
    try:
        client = VisionClient()
        result = client.locate_sign(args.image.read_bytes(), width, height, target_hint=args.target_hint)
    except Exception as exc:
        sys.stderr.write(f"Vision API error: {exc}\n")
        if args.out_json.exists():
            args.out_json.unlink()
        if args.preview and args.preview.exists():
            args.preview.unlink()
        sys.exit(1)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"quad": result["quad"], "confidence": result.get("confidence"), "model": result["model"]}, indent=2),
        encoding="utf-8",
    )
    if args.preview:
        preview = image.copy()
        cv2.polylines(preview, [np.asarray(result["quad"], dtype=np.int32)], True, (0, 255, 255), 4)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), preview)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
