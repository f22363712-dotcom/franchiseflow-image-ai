"""Small OpenAI-compatible vision client for first-frame sign localization.

The gateway and key are deliberately environment-only. This module makes one
vision request per source video, then leaves frame-to-frame tracking to the
local geometric tracker.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE lines without adding a dotenv dependency."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def _json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("vision response JSON must be an object")
    return value


def _is_convex_quad(pts: list[list[float]]) -> bool:
    signs: list[bool] = []
    for i in range(4):
        dx1 = pts[(i + 1) % 4][0] - pts[i][0]
        dy1 = pts[(i + 1) % 4][1] - pts[i][1]
        dx2 = pts[(i + 2) % 4][0] - pts[(i + 1) % 4][0]
        dy2 = pts[(i + 2) % 4][1] - pts[(i + 1) % 4][1]
        cp = dx1 * dy2 - dy1 * dx2
        if cp != 0:
            signs.append(cp > 0)
    return len(signs) == 4 and (all(signs) or not any(signs))


def normalize_quad(value: Any, width: int, height: int) -> list[list[float]]:
    """Accept pixel or 0..1 coordinates and return validated pixel corners."""
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("vision response quad must contain four corners")
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each quad corner must be [x, y]")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("each quad corner coordinate must be a number") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("quad contains NaN or non-finite values")
        points.append([x, y])

    all_unit = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)
    if all_unit:
        points = [[x * width, y * height] for x, y in points]
    else:
        has_fractional_unit = any((0.0 < x < 1.0 or 0.0 < y < 1.0) for x, y in points)
        has_large_pixel = any((x > 1.0 or y > 1.0) for x, y in points)
        if has_fractional_unit and has_large_pixel:
            raise ValueError("quad contains mixed normalized and pixel coordinates")

    if any(x < -0.05 * width or x > 1.05 * width or y < -0.05 * height or y > 1.05 * height for x, y in points):
        raise ValueError("quad lies outside the image")

    if not _is_convex_quad(points):
        raise ValueError("quad must be a convex polygon")

    area = abs(sum(points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1] for i in range(4))) / 2
    if area < 100:
        raise ValueError("quad area is too small")
    return [[round(x, 3), round(y, 3)] for x, y in points]


class VisionClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None, timeout: int = 45):
        load_env_file()
        self.api_key = api_key or os.getenv("BANANA_API_KEY")
        self.base_url = (base_url or os.getenv("BANANA_API_BASE_URL") or "https://e.c.h-e.top/v1").rstrip("/")
        self.model = model or os.getenv("BANANA_VISION_MODEL") or "gemini-3.1-flash-image"
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("BANANA_API_KEY is missing; put it in the environment or .env")

    def locate_candidates(self, image_bytes: bytes, width: int, height: int, target_hint: str | None = None, mime: str = "image/jpeg") -> dict[str, Any]:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        prompt = "Find every clearly visible physical planar storefront sign or flat advertising board in this image. "
        # The locator is intentionally brand-agnostic.  A concrete brand or
        # sign description is supplied by the caller when it is known; the
        # default must not anchor the model to the demo fixture's wording.
        if target_hint:
            hint = str(target_hint).strip()[:500]
            prompt += f"Target description supplied by the user: {hint}. "
        else:
            prompt += "Target: the primary storefront sign or flat advertising board. "
        prompt += (
            "Return only JSON with key candidates. candidates must be an array of objects with quad, confidence, and reason. "
            "quad must be four image pixel corners in clockwise order [top-left, top-right, bottom-right, bottom-left]. "
            "Use pixel coordinates for the full supplied image size (the input may contain three side-by-side video frames; do not reset x to a panel-local origin). If no clear planar sign exists, return an empty candidates array. "
            "Bound each physical signboard polygon strictly; do not include brick walls, windows, doors, or street."
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": f"Image size is {width}x{height}. {prompt}"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            ]}],
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(1_000_000)
        except urllib.error.HTTPError as exc:
            # Never echo gateway response bodies because proxies sometimes reflect headers.
            raise RuntimeError(f"vision gateway returned HTTP {exc.code}; check base URL, key, and gateway access") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"vision gateway timed out after {self.timeout}s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)) or "timed out" in str(exc.reason).lower():
                raise RuntimeError(f"vision gateway timed out after {self.timeout}s") from exc
            raise RuntimeError(f"vision gateway is unreachable: {exc.reason}") from exc
        try:
            envelope = json.loads(raw)
            content = envelope["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            answer = _json_from_text(str(content))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("vision gateway returned an unexpected response shape") from exc

        raw_candidates = answer.get("candidates")
        if raw_candidates is None:
            if answer.get("quad") is None:
                raise RuntimeError(f"vision model could not find a replaceable planar sign: {answer.get('reason', 'no reason')}")
            raw_candidates = [{"quad": answer.get("quad"), "confidence": answer.get("confidence", 0.0), "reason": answer.get("reason", "legacy single candidate")}]
        if not isinstance(raw_candidates, list):
            raise RuntimeError("vision gateway returned an unexpected candidate list")

        candidates: list[dict[str, Any]] = []
        invalid_reasons: list[str] = []
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict) or raw_candidate.get("quad") is None:
                invalid_reasons.append("candidate_missing_quad")
                continue
            try:
                quad = normalize_quad(raw_candidate["quad"], width, height)
            except ValueError as exc:
                invalid_reasons.append(str(exc))
                continue
            try:
                confidence = float(raw_candidate.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            candidates.append({
                "quad": quad,
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": str(raw_candidate.get("reason", "")),
            })

        if invalid_reasons:
            # A mixed valid/invalid response is ambiguous: silently dropping a
            # candidate can make the UI replace the wrong physical surface.
            raise RuntimeError("vision_failed: invalid candidate response: " + ", ".join(invalid_reasons))
        if not candidates:
            reason = answer.get("reason") or ", ".join(invalid_reasons) or "no candidates"
            raise RuntimeError(f"vision model could not find a replaceable planar sign: {reason}")
        return {"model": self.model, "candidates": candidates}

    def locate_sign(self, image_bytes: bytes, width: int, height: int, mime: str = "image/jpeg", target_hint: str | None = None) -> dict[str, Any]:
        result = self.locate_candidates(image_bytes, width, height, target_hint=target_hint, mime=mime)
        candidate = result["candidates"][0]
        return {**candidate, "model": result["model"]}
