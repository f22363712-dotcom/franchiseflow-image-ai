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


_INITIALIZATION_MODES = {"vision_model", "manual_fallback", "reviewed_candidate"}


def _normalize_scene_plan(scene_plan: list[dict] | None) -> list[dict] | None:
    """Validate and normalize the stable scene/surface manifest.

    The tracker deliberately accepts a small, explicit manifest instead of
    guessing scene boundaries.  Frame ranges are inclusive and must cover a
    contiguous timeline beginning at frame zero.
    """
    if scene_plan is None:
        return None
    if not isinstance(scene_plan, list) or not 1 <= len(scene_plan) <= 2:
        raise ValueError("scene_plan must contain between 1 and 2 scenes")

    normalized: list[dict] = []
    previous_end = -1
    seen_scene_ids: set[str] = set()
    seen_surface_ids: set[str] = set()
    for scene_index, scene in enumerate(scene_plan):
        if not isinstance(scene, dict):
            raise ValueError("scene_plan entries must be objects")
        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id or len(scene_id) > 64 or scene_id in seen_scene_ids:
            raise ValueError("scene_id must be a unique non-empty string")
        seen_scene_ids.add(scene_id)
        start = scene.get("start_frame")
        end = scene.get("end_frame")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("scene frame bounds must be integers")
        if start != previous_end + 1 or end < start:
            raise ValueError("scene frame ranges must be contiguous starting at frame 0")
        surfaces = scene.get("surfaces")
        if not isinstance(surfaces, list) or not 1 <= len(surfaces) <= 4:
            raise ValueError("each scene must contain between 1 and 4 surfaces")
        normalized_surfaces: list[dict] = []
        for surface in surfaces:
            if not isinstance(surface, dict):
                raise ValueError("scene surfaces must be objects")
            surface_id = surface.get("surface_id")
            if not isinstance(surface_id, str) or not surface_id or len(surface_id) > 64 or surface_id in seen_surface_ids:
                raise ValueError("surface_id must be a unique non-empty string")
            seen_surface_ids.add(surface_id)
            mode = surface.get("initialization_mode", scene.get("initialization_mode", "manual_fallback"))
            if mode not in _INITIALIZATION_MODES:
                raise ValueError(f"unsupported scene surface initialization mode: {mode}")
            normalized_surfaces.append({
                "surface_id": surface_id,
                "quad": parse_quad(surface.get("quad", surface.get("corners"))),
                "initialization_mode": mode,
            })
        normalized.append({
            "scene_id": scene_id,
            "start_frame": start,
            "end_frame": end,
            "surfaces": normalized_surfaces,
        })
        previous_end = end
    return normalized


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
    surface_quads: list[list] | None = None,
    scene_plan: list[dict] | None = None,
) -> dict[str, Any]:
    video_path = Path(video)
    if not video_path.exists():
        raise ValueError(f"video does not exist: {video_path}")
    if not 1 <= len(signs) <= 10:
        raise ValueError(f"between 1 and 10 signs are required, got {len(signs)}")
    if initialization_mode not in {"vision_model", "manual_fallback", "reviewed_candidate"}:
        raise ValueError(f"unsupported initialization mode: {initialization_mode}")
    normalized_scene_plan = _normalize_scene_plan(scene_plan)
    for sign_path in signs:
        if not Path(sign_path).exists():
            raise ValueError(f"sign does not exist: {sign_path}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("FFmpeg is required for browser-compatible H.264 video encoding but was not found in PATH.")

    def has_audio_stream(path: Path) -> bool:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            if preserve_audio:
                raise RuntimeError("FFprobe is required to preserve input audio but was not found in PATH.")
            return False
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if probe.returncode != 0:
            if preserve_audio:
                raise RuntimeError(f"FFprobe failed while checking input audio: {probe.stderr.strip()}")
            return False
        return bool(probe.stdout.strip())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quad_pts = parse_quad(quad)
    if surface_quads is None:
        surface_points = [quad_pts]
    else:
        if not 1 <= len(surface_quads) <= 4:
            raise ValueError(f"between 1 and 4 surface quads are supported, got {len(surface_quads)}")
        surface_points = [parse_quad(item) for item in surface_quads]

    cap = cv2.VideoCapture(str(video_path))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"cannot decode {video_path}")

    if normalized_scene_plan is None:
        trackers = [PlanarTracker(first, surface, min_inliers, ransac_px, ratio) for surface in surface_points]
        surface_count = len(trackers)
    else:
        # Scene-local trackers are initialized at each scene's own source
        # frame.  No scene plan is allowed to borrow the first scene anchor.
        trackers = []
        surface_points = []
        surface_count = 0
    max_surface_count = max((len(scene["surfaces"]) for scene in (normalized_scene_plan or [])), default=surface_count)
    scene_stats = [
        {
            "scene_id": scene["scene_id"],
            "start_frame": scene["start_frame"],
            "end_frame": scene["end_frame"],
            "surface_ids": [surface["surface_id"] for surface in scene["surfaces"]],
            "failed_frames": 0,
            "surface_failures": {surface["surface_id"]: 0 for surface in scene["surfaces"]},
        }
        for scene in (normalized_scene_plan or [])
    ]
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
    scene_index = 0
    active_scene = None
    active_scene_stats = None
    log_file = (out_dir / "tracking.jsonl").open("w", encoding="utf-8")
    try:
        while frame is not None:
            if normalized_scene_plan is not None:
                if scene_index >= len(normalized_scene_plan):
                    raise ValueError("scene_plan ended before the decoded video timeline")
                scene_config = normalized_scene_plan[scene_index]
                if index > scene_config["end_frame"]:
                    scene_index += 1
                    if scene_index >= len(normalized_scene_plan):
                        raise ValueError("scene_plan ended before the decoded video timeline")
                    scene_config = normalized_scene_plan[scene_index]
                if index != scene_config["start_frame"] and index < scene_config["start_frame"]:
                    raise ValueError("scene_plan frame ranges are not contiguous with decoded timeline")
                if index == scene_config["start_frame"]:
                    active_scene = scene_config
                    active_scene_stats = scene_stats[scene_index]
                    surface_points = [surface["quad"] for surface in active_scene["surfaces"]]
                    surface_count = len(surface_points)
                    trackers = []
                    for surface in active_scene["surfaces"]:
                        try:
                            trackers.append(PlanarTracker(frame, surface["quad"], min_inliers, ransac_px, ratio))
                        except (ValueError, cv2.error) as exc:
                            # Keep a positional placeholder so a failed surface
                            # cannot accidentally borrow another surface's H.
                            trackers.append(None)
                    max_surface_count = max(max_surface_count, surface_count)

            surface_details = []
            homographies = []
            for surface_index, tracker in enumerate(trackers):
                surface_quad = surface_points[surface_index]
                surface_config = active_scene["surfaces"][surface_index] if active_scene is not None else None
                surface_mode = surface_config["initialization_mode"] if surface_config else initialization_mode
                scene_id = active_scene["scene_id"] if active_scene is not None else None
                surface_id = surface_config["surface_id"] if surface_config else None
                if tracker is None:
                    h = None
                    surface_detail = {
                        "status": "failed",
                        "method": "none",
                        "reason": "surface_initialization_failed",
                        "render_policy": "unmodified_original_frame",
                        "initialization_mode": surface_mode,
                    }
                elif active_scene is not None and index == active_scene["start_frame"]:
                    h = np.eye(3)
                    surface_detail = {
                        "status": "ok",
                        "method": "manual_first_frame" if surface_mode == "manual_fallback" else "vision_first_frame",
                        "quad": surface_quad.tolist(),
                        "inliers": len(tracker.key0),
                        "reprojection_median_px": 0.0,
                        "reprojection_p95_px": 0.0,
                    }
                elif active_scene is None and index == 0:
                    h = np.eye(3)
                    surface_detail = {
                        "status": "ok",
                        "method": "manual_first_frame" if initialization_mode == "manual_fallback" else "vision_first_frame",
                        "quad": surface_quad.tolist(),
                        "inliers": len(tracker.key0),
                        "reprojection_median_px": 0.0,
                        "reprojection_p95_px": 0.0,
                    }
                else:
                    h, surface_detail = tracker.track(frame)
                surface_detail["surface_index"] = surface_index
                if scene_id is not None:
                    surface_detail.update({"scene_id": scene_id, "surface_id": surface_id, "initialization_mode": surface_mode})
                surface_details.append(surface_detail)
                homographies.append(h)

            if normalized_scene_plan is not None:
                valid_details = [item for item in surface_details if item.get("status") == "ok"]
                detail = {
                    "status": "ok" if len(valid_details) == surface_count else "failed",
                    "method": "scene_surface_tracking",
                    "scene_id": active_scene["scene_id"],
                    "surface_count": surface_count,
                    "surface_details": surface_details,
                    "inliers": int(np.median([item.get("inliers", 0) for item in valid_details])) if valid_details else 0,
                    "reprojection_median_px": float(np.median([item["reprojection_median_px"] for item in valid_details if "reprojection_median_px" in item])) if any("reprojection_median_px" in item for item in valid_details) else 0.0,
                }
                if detail["status"] == "failed":
                    active_scene_stats["failed_frames"] += 1
                for item in surface_details:
                    if item.get("status") != "ok":
                        active_scene_stats["surface_failures"][item["surface_id"]] += 1
            elif surface_count == 1:
                detail = surface_details[0]
            else:
                valid_details = [item for item in surface_details if item.get("status") == "ok"]
                detail = {
                    "status": "ok" if len(valid_details) == surface_count else "failed",
                    "method": "multi_surface",
                    "surface_count": surface_count,
                    "surface_details": surface_details,
                    "inliers": int(np.median([item.get("inliers", 0) for item in valid_details])) if valid_details else 0,
                    "reprojection_median_px": float(np.median([item["reprojection_median_px"] for item in valid_details if "reprojection_median_px" in item])) if any("reprojection_median_px" in item for item in valid_details) else 0.0,
                }
            detail.update(frame=index, seconds=index / fps)
            records.append(detail)
            log_file.write(json.dumps(detail) + "\n")
            current_quads = [project_quad(h, surface) if h is not None else None for h, surface in zip(homographies, surface_points)]
            for writer, sign in zip(writers, loaded_signs):
                rendered = frame
                for current_quad in current_quads:
                    if current_quad is not None:
                        rendered = composite(rendered, sign, current_quad, illumination)
                writer.write(rendered)
            if debug is not None:
                annotated = frame.copy()
                for current_quad in current_quads:
                    if current_quad is not None:
                        cv2.polylines(annotated, [np.round(current_quad).astype(np.int32)], True, (0, 255, 255), 2)
                label = f"{index} {detail['status']} {detail['method']} surfaces={surface_count} inliers={detail.get('inliers', 0)} err={detail.get('reprojection_median_px', 0):.2f}px"
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

    if normalized_scene_plan is not None:
        processed_last = index - 1
        expected_last = normalized_scene_plan[-1]["end_frame"]
        if expected_last != processed_last:
            limit_note = " after max_frames truncation" if max_frames else ""
            raise ValueError(
                f"scene_plan must end at processed frame {processed_last}{limit_note}; got {expected_last}"
            )

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

    scene_summary = [
        {
            "scene_id": item["scene_id"],
            "start_frame": item["start_frame"],
            "end_frame": item["end_frame"],
            "surface_ids": item["surface_ids"],
            "failed_frames": item["failed_frames"],
            "surface_failures": [
                {"surface_id": surface_id, "failed_frames": failed_frames}
                for surface_id, failed_frames in item["surface_failures"].items()
            ],
        }
        for item in scene_stats
    ]
    if normalized_scene_plan is None:
        scene_summary = [{
            "scene_id": "scene_001",
            "start_frame": 0,
            "end_frame": max(0, index - 1),
            "surface_ids": [f"surface_{position:03d}" for position in range(1, surface_count + 1)],
            "failed_frames": sum(1 for record in records if record.get("status") != "ok"),
            "surface_failures": [],
        }]
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
            "surface_count": max_surface_count if normalized_scene_plan is not None else surface_count,
            "surface_quads": (
                [surface.round(3).tolist() for scene in normalized_scene_plan for surface in [entry["quad"] for entry in scene["surfaces"]]]
                if normalized_scene_plan is not None else
                [surface.round(3).tolist() for surface in surface_points]
            ),
            "surface_failures": [
                {"frame": record["frame"], "surfaces": [item.get("surface_index") for item in record.get("surface_details", []) if item.get("status") != "ok"]}
                for record in records if record.get("surface_details") and record.get("status") != "ok"
            ],
            "model": model_name,
            "audio_policy": "preserve_input_audio" if preserve_audio else "disabled",
            "audio_present": audio_present,
            "source_candidate": source_candidate,
            "scene_count": len(scene_summary),
            "scenes": scene_summary,
        },
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--quad", type=Path, help="manual TL TR BR BL pixel coordinates")
    parser.add_argument("--quad-json", type=Path, help="JSON file containing quad corners")
    parser.add_argument("--surface-quad-json", type=Path, nargs="+", help="optional additional surface quad JSON files (up to four total surfaces)")
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
    surface_quads = None
    if args.surface_quad_json:
        surface_quads = [read_quad(path) for path in args.surface_quad_json]
        if quad_arg and not any(np.array_equal(parse_quad(quad_arg), surface) for surface in surface_quads):
            surface_quads.insert(0, parse_quad(quad_arg))

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
        surface_quads=surface_quads,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
