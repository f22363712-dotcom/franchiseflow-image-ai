"""Throwaway planar-sign video replacement spike.

The first-frame quadrilateral is MANUALLY supplied. This validates tracking and
compositing only; it does not establish an automated Image AI product.
Dependencies: Python, opencv-python (SIFT), numpy, Pillow.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def transcode_to_h264(input_path: Path, output_path: Path, audio_source: Path | None = None) -> Path:
    """Transcode video to H.264 and optionally copy audio from the source video."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("FFmpeg is required for browser-compatible H.264 video encoding but was not found in PATH.")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_path),
    ]
    if audio_source is not None:
        cmd += ["-i", str(audio_source)]
    cmd += [
        "-map", "0:v:0",
    ]
    if audio_source is not None:
        cmd += ["-map", "1:a?", "-c:a", "aac"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "fast",
        "-crf", "22",
        "-shortest",
        str(output_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg transcode failed: {res.stderr}")
    return output_path



def read_quad(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    quad = np.asarray(data.get("quad", data.get("corners")) if isinstance(data, dict) else data, np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("quad JSON must contain four finite [x,y] points, TL TR BR BL")
    if not cv2.isContourConvex(quad.astype(np.int32)) or abs(cv2.contourArea(quad)) < 100:
        raise ValueError("quad must be a convex nondegenerate polygon")
    return quad


def quad_mask(shape: tuple[int, int], quad: np.ndarray, dilation: int = 0) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    if dilation:
        mask = cv2.dilate(mask, np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8))
    return mask


def project_quad(matrix: np.ndarray, quad: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(quad.reshape(-1, 1, 2), matrix).reshape(4, 2)


def validate_homography(matrix, src, dst, quad, frame_shape, threshold, min_inliers):
    """Return a tested H and fit indicators; never infer physical correctness."""
    if matrix is None or not np.isfinite(matrix).all():
        return None, {"reason": "no_homography"}
    predicted = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(predicted - dst, axis=1)
    keep = errors <= threshold
    count = int(keep.sum())
    if count < min_inliers:
        return None, {"reason": "too_few_inliers", "inliers": count}
    # Refit only the consensus set; reported residuals come from those points.
    matrix, _ = cv2.findHomography(src[keep], dst[keep], 0)
    if matrix is None or not np.isfinite(matrix).all():
        return None, {"reason": "refit_failed"}
    projected = project_quad(matrix, quad)
    area = abs(cv2.contourArea(projected.astype(np.float32)))
    anchor_area = abs(cv2.contourArea(quad.astype(np.float32)))
    height, width = frame_shape
    valid = (cv2.isContourConvex(projected.astype(np.int32)) and
             0.08 * anchor_area < area < 12 * anchor_area and
             np.all(projected[:, 0] > -0.2 * width) and
             np.all(projected[:, 0] < 1.2 * width) and
             np.all(projected[:, 1] > -0.2 * height) and
             np.all(projected[:, 1] < 1.2 * height))
    spread = abs(cv2.contourArea(cv2.convexHull(src[keep].astype(np.float32)))) / anchor_area
    if not valid or spread < 0.04:
        return None, {"reason": "degenerate_geometry", "inliers": count, "spread": float(spread)}
    errors = np.linalg.norm(cv2.perspectiveTransform(src[keep].reshape(-1, 1, 2), matrix).reshape(-1, 2) - dst[keep], axis=1)
    return (matrix, src[keep], dst[keep]), {
        "inliers": count, "candidates": len(src), "inlier_ratio": count / len(src),
        "reprojection_median_px": float(np.median(errors)),
        "reprojection_p95_px": float(np.percentile(errors, 95)),
        "spread": float(spread), "quad": projected.round(3).tolist(),
    }


class PlanarTracker:
    def __init__(self, frame, quad, min_inliers=8, threshold=3.0, ratio=0.75):
        self.quad = quad
        self.shape = frame.shape[:2]
        self.min_inliers, self.threshold, self.ratio = min_inliers, threshold, ratio
        self.gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.015, edgeThreshold=15)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.key0, self.desc0 = self.sift.detectAndCompute(self.gray0, quad_mask(self.shape, quad))
        if self.desc0 is None or len(self.key0) < min_inliers:
            raise ValueError(f"only {len(self.key0)} SIFT features inside sign; choose a textured planar sign")
        self.prev_gray = self.gray0
        self.anchor_pts = np.asarray([k.pt for k in self.key0], np.float32)
        self.prev_pts = self.anchor_pts.copy()
        self.last_quad = quad.copy()
        self.lk = dict(winSize=(25, 25), maxLevel=4,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01))

    def track(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Absolute first-frame matching keeps the anchor fixed throughout video.
        key, desc = self.sift.detectAndCompute(gray, None)
        anchor_reason = "no_current_features"
        if desc is not None:
            pairs = self.matcher.knnMatch(self.desc0, desc, k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
            if len(good) >= self.min_inliers:
                src = np.asarray([self.key0[m.queryIdx].pt for m in good], np.float32)
                dst = np.asarray([key[m.trainIdx].pt for m in good], np.float32)
                h, _ = cv2.findHomography(src, dst, cv2.RANSAC, self.threshold)
                result, detail = validate_homography(h, src, dst, self.quad, self.shape, self.threshold, self.min_inliers)
                if result is not None:
                    h, self.anchor_pts, self.prev_pts = result
                    self.prev_gray = gray
                    self.last_quad = project_quad(h, self.quad)
                    return h, {"status": "ok", "method": "anchor_sift", "matches": len(good), **detail}
                anchor_reason = detail["reason"]
            else:
                anchor_reason = f"only_{len(good)}_matches"

        # LK retains original anchor-point coordinates, then estimates anchor->now.
        # FB screening discards points with ambiguous one-way optical flow.
        if self.prev_pts is not None and len(self.prev_pts) >= self.min_inliers:
            forward, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts.reshape(-1, 1, 2), None, **self.lk)
            if forward is not None:
                backward, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, forward, None, **self.lk)
                if backward is not None:
                    fb = np.linalg.norm(backward.reshape(-1, 2) - self.prev_pts, axis=1)
                    keep = status.ravel().astype(bool) & status_back.ravel().astype(bool) & (fb < 1.0)
                    src, dst = self.anchor_pts[keep], forward.reshape(-1, 2)[keep]
                    if len(src) >= self.min_inliers:
                        h, _ = cv2.findHomography(src, dst, cv2.RANSAC, self.threshold)
                        result, detail = validate_homography(h, src, dst, self.quad, self.shape, self.threshold, self.min_inliers)
                        if result is not None:
                            h, self.anchor_pts, self.prev_pts = result
                            self.prev_gray = gray
                            self.last_quad = project_quad(h, self.quad)
                            return h, {"status": "ok", "method": "anchor_lk", "anchor_failure": anchor_reason,
                                       "fb_median_px": float(np.median(fb[keep])), **detail}
        self.prev_pts = None
        self.prev_gray = gray
        return None, {"status": "failed", "method": "none", "reason": anchor_reason,
                      "render_policy": "unmodified_original_frame"}


def load_sign(path: Path):
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def composite(frame, sign, quad, illumination=False):
    height, width = frame.shape[:2]
    sh, sw = sign.shape[:2]
    src = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]])
    transform = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(sign, transform, (width, height), flags=cv2.INTER_LINEAR)
    colour = warped[:, :, :3].astype(np.float32)
    alpha = warped[:, :, 3:4].astype(np.float32) / 255
    if illumination:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        field = cv2.GaussianBlur(gray, (0, 0), max(8, min(sw, sh) / 8))
        selected = field[alpha[:, :, 0] > 0.9]
        if len(selected):
            factor = np.clip(field / max(float(np.median(selected)), 1), 0.6, 1.3)
            colour *= factor[:, :, None]
    return np.clip(frame.astype(np.float32) * (1 - alpha) + colour * alpha, 0, 255).astype(np.uint8)


def parse_quad(value: Any) -> np.ndarray:
    if isinstance(value, (str, Path)):
        return read_quad(Path(value))
    quad = np.asarray(value.get("quad", value.get("corners")) if isinstance(value, dict) else value, np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("quad must contain four finite [x,y] points, TL TR BR BL")
    if not cv2.isContourConvex(np.round(quad).astype(np.int32)) or abs(cv2.contourArea(quad)) < 100:
        raise ValueError("quad must be a convex nondegenerate polygon")
    return quad


def run_tracker(
    video: str | Path,
    quad: list | np.ndarray | str | Path,
    signs: list[str | Path],
    out_dir: str | Path,
    illumination: bool = False,
    max_frames: int = 0,
    min_inliers: int = 8,
    ransac_px: float = 3.0,
    ratio: float = 0.75,
    no_debug: bool = False,
    initialization_mode: str = "manual_fallback",
    preserve_audio: bool = True,
    model_name: str | None = None,
    source_candidate: dict | None = None,
) -> dict[str, Any]:
    video_path = Path(video)
    if not video_path.exists():
        raise ValueError(f"video does not exist: {video_path}")
    if not 1 <= len(signs) <= 10:
        raise ValueError(f"between 1 and 10 signs are required, got {len(signs)}")
    if initialization_mode not in {"vision_model", "manual_fallback", "reviewed_candidate"}:
        raise ValueError(f"unsupported initialization mode: {initialization_mode}")
    for sign_path in signs:
        if not Path(sign_path).exists():
            raise ValueError(f"sign does not exist: {sign_path}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("FFmpeg is required for browser-compatible H.264 video encoding but was not found in PATH.")

    def has_audio_stream(path: Path) -> bool:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return False
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return probe.returncode == 0 and bool(probe.stdout.strip())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quad_pts = parse_quad(quad)

    cap = cv2.VideoCapture(str(video_path))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"cannot decode {video_path}")

    tracker = PlanarTracker(first, quad_pts, min_inliers, ransac_px, ratio)
    loaded_signs = [load_sign(Path(p)) for p in signs]
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    height, width = first.shape[:2]
    target_paths = [out_dir / f"branch_{i+1}_{Path(p).stem}.mp4" for i, p in enumerate(signs)]
    raw_paths = [out_dir / f"_raw_branch_{i+1}_{Path(p).stem}.mp4" for i, p in enumerate(signs)]
    target_debug = out_dir / "tracking_debug.mp4"
    raw_debug = out_dir / "_raw_tracking_debug.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = [cv2.VideoWriter(str(p), fourcc, fps, (width, height)) for p in raw_paths]
    debug = None if no_debug else cv2.VideoWriter(str(raw_debug), fourcc, fps, (width, height))
    if not all(w.isOpened() for w in writers) or (debug is not None and not debug.isOpened()):
        cap.release()
        for w in writers:
            w.release()
        if debug is not None:
            debug.release()
        raise ValueError("cannot open MP4 output writers")

    records, t0 = [], time.perf_counter()
    frame, index = first, 0
    log_file = (out_dir / "tracking.jsonl").open("w", encoding="utf-8")
    try:
        while frame is not None:
            if index == 0:
                h = np.eye(3)
                detail = {
                    "status": "ok",
                    "method": "manual_first_frame" if initialization_mode == "manual_fallback" else "vision_first_frame",
                    "quad": quad_pts.tolist(),
                    "inliers": len(tracker.key0),
                    "reprojection_median_px": 0.0,
                    "reprojection_p95_px": 0.0,
                }
            else:
                h, detail = tracker.track(frame)
            detail.update(frame=index, seconds=index / fps)
            records.append(detail)
            log_file.write(json.dumps(detail) + "\n")
            current_quad = project_quad(h, quad_pts) if h is not None else None
            for writer, sign in zip(writers, loaded_signs):
                writer.write(composite(frame, sign, current_quad, illumination) if current_quad is not None else frame)
            if debug is not None:
                annotated = frame.copy()
                if current_quad is not None:
                    cv2.polylines(annotated, [np.round(current_quad).astype(np.int32)], True, (0, 255, 255), 2)
                label = f"{index} {detail['status']} {detail['method']} inliers={detail.get('inliers', 0)} err={detail.get('reprojection_median_px', 0):.2f}px"
                cv2.rectangle(annotated, (0, 0), (width, 38), (0, 0, 0), -1)
                cv2.putText(
                    annotated,
                    label,
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if h is not None else (0, 0, 255),
                    2,
                )
                debug.write(annotated)
            index += 1
            if max_frames and index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
    finally:
        cap.release()
        log_file.close()
        for writer in writers:
            writer.release()
        if debug is not None:
            debug.release()

    # Transcode OpenCV mp4v outputs to web-compatible H.264 (yuv420p + faststart)
    audio_present = bool(preserve_audio and has_audio_stream(video_path))
    for raw_p, target_p in zip(raw_paths, target_paths):
        transcode_to_h264(raw_p, target_p, audio_source=video_path if audio_present else None)
        try:
            raw_p.unlink(missing_ok=True)
        except Exception:
            pass

    if debug is not None:
        transcode_to_h264(raw_debug, target_debug)
        try:
            raw_debug.unlink(missing_ok=True)
        except Exception:
            pass

    from metrics import compute_summary

    summary = compute_summary(
        records=records,
        fps=fps,
        total_frames=index,
        elapsed_seconds=time.perf_counter() - t0,
        video_path=video_path,
        quad=quad_pts,
        output_videos=target_paths,
        mode=initialization_mode,
        extra={
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "branch_count": len(signs),
            "model": model_name,
            "audio_policy": "preserve_input_audio" if preserve_audio else "disabled",
            "audio_present": audio_present,
            "source_candidate": source_candidate,
        },
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--quad", type=Path, help="manual TL TR BR BL pixel coordinates")
    parser.add_argument("--quad-json", type=Path, help="JSON file containing quad corners")
    parser.add_argument("--signs", type=Path, nargs="+", required=True, metavar="PNG")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--ransac-px", type=float, default=3.0)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--illumination", action="store_true", help="approximate low-frequency lighting, not occlusion handling")
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--initialization-mode", choices=["vision_model", "manual_fallback", "reviewed_candidate"], default=None)
    parser.add_argument("--model", dest="model_name", default=None)
    args = parser.parse_args()

    quad_arg = args.quad or args.quad_json
    if not quad_arg:
        raise ValueError("either --quad or --quad-json must be provided")

    mode = args.initialization_mode or "manual_fallback"

    summary = run_tracker(
        video=args.video,
        quad=quad_arg,
        signs=args.signs,
        out_dir=args.out_dir,
        illumination=args.illumination,
        max_frames=args.max_frames,
        min_inliers=args.min_inliers,
        ransac_px=args.ransac_px,
        ratio=args.ratio,
        no_debug=args.no_debug,
        initialization_mode=mode,
        model_name=args.model_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
