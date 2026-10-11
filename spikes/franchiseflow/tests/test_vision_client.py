import io
import json
import socket
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_client import VisionClient, _json_from_text, normalize_quad


class VisionClientParsingTests(unittest.TestCase):
    def test_parses_fenced_json(self):
        self.assertEqual(_json_from_text('```json\n{"quad": [[0,0],[1,0],[1,1],[0,1]]}\n```')["quad"][0], [0, 0])

    def test_normalizes_unit_coordinates(self):
        quad = normalize_quad([[0, 0], [1, 0], [1, 1], [0, 1]], 1000, 500)
        self.assertEqual(quad[2], [1000.0, 500.0])

    def test_rejects_tiny_quad(self):
        with self.assertRaises(ValueError):
            normalize_quad([[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]], 1000, 500)

    def test_rejects_non_four_points(self):
        with self.assertRaises(ValueError):
            normalize_quad([[0, 0], [100, 0], [100, 100]], 1000, 500)
        with self.assertRaises(ValueError):
            normalize_quad([[0, 0], [100, 0], [100, 100], [50, 100], [0, 100]], 1000, 500)

    def test_rejects_non_finite_points(self):
        with self.assertRaises(ValueError):
            normalize_quad([[0, 0], [float("nan"), 0], [100, 100], [0, 100]], 1000, 500)
        with self.assertRaises(ValueError):
            normalize_quad([[0, 0], [float("inf"), 0], [100, 100], [0, 100]], 1000, 500)

    def test_rejects_out_of_bounds_quad(self):
        with self.assertRaises(ValueError):
            normalize_quad([[-200, 0], [100, 0], [100, 100], [0, 100]], 1000, 500)

    def test_rejects_non_convex_quad(self):
        # A bow-tie / self-intersecting polygon
        with self.assertRaises(ValueError):
            normalize_quad([[10, 10], [100, 100], [100, 10], [10, 100]], 1000, 500)

    def test_parses_openai_compatible_response(self):
        payload = {"choices": [{"message": {"content": '```json\n{"quad": [[0.1,0.1],[0.9,0.1],[0.9,0.5],[0.1,0.5]], "confidence": 0.91}\n```'}}]}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, _limit):
                return json.dumps(payload).encode()

        with patch("vision_client.urllib.request.urlopen", return_value=FakeResponse()):
            result = VisionClient(api_key="test-key", base_url="https://example.test/v1", model="GLM-5.3-flash").locate_sign(b"jpeg", 1000, 500)
        self.assertEqual(result["quad"][2], [900.0, 250.0])
        self.assertEqual(result["model"], "GLM-5.3-flash")

    def test_parses_multiple_candidates(self):
        payload = {"choices": [{"message": {"content": json.dumps({
            "candidates": [
                {"quad": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.5], [0.1, 0.5]], "confidence": 0.91, "reason": "sign"},
                {"quad": [[0.2, 0.6], [0.4, 0.6], [0.4, 0.8], [0.2, 0.8]], "confidence": 0.42, "reason": "poster"},
            ]
        })}}]}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, _limit):
                return json.dumps(payload).encode()

        with patch("vision_client.urllib.request.urlopen", return_value=FakeResponse()):
            result = VisionClient(api_key="test-key", base_url="https://example.test/v1").locate_candidates(b"jpeg", 1000, 500)
        self.assertEqual(result["model"], "gemini-3.1-flash-image")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["candidates"][0]["quad"][2], [900.0, 250.0])

    def test_rejects_response_with_any_mixed_coordinate_candidate(self):
        payload = {"choices": [{"message": {"content": json.dumps({
            "candidates": [
                {"quad": [[0.1, 0.1], [900, 0.1], [900, 250], [0.1, 250]], "confidence": 0.9, "reason": "mixed"},
                {"quad": [[100, 100], [900, 100], [900, 250], [100, 250]], "confidence": 0.8, "reason": "valid"},
            ]
        })}}]}

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, _limit): return json.dumps(payload).encode()

        with patch("vision_client.urllib.request.urlopen", return_value=FakeResponse()):
            with self.assertRaises(RuntimeError):
                VisionClient(api_key="test-key", base_url="https://example.test/v1").locate_candidates(b"jpeg", 1000, 500)


    def test_raises_on_http_403_without_leaking_body(self):
        secret_body = b'{"error": {"message": "upstream_secret_token_12345"}}'
        error = urllib.error.HTTPError(
            url="https://example.test/v1/chat/completions",
            code=403,
            msg="Forbidden",
            hdrs={"X-Custom": "upstream_secret_token_12345"},
            fp=io.BytesIO(secret_body)
        )
        with patch("vision_client.urllib.request.urlopen", side_effect=error):
            client = VisionClient(api_key="test-key", base_url="https://example.test/v1")
            with self.assertRaises(RuntimeError) as ctx:
                client.locate_sign(b"jpeg", 1000, 500)
            self.assertIn("HTTP 403", str(ctx.exception))
            self.assertNotIn("upstream_secret_token_12345", str(ctx.exception))


    def test_raises_on_timeout(self):
        with patch("vision_client.urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            client = VisionClient(api_key="test-key", base_url="https://example.test/v1", timeout=10)
            with self.assertRaises(RuntimeError) as ctx:
                client.locate_sign(b"jpeg", 1000, 500)
            self.assertIn("timed out after 10s", str(ctx.exception))

        with patch("vision_client.urllib.request.urlopen", side_effect=urllib.error.URLError(socket.timeout("timed out"))):
            client = VisionClient(api_key="test-key", base_url="https://example.test/v1", timeout=10)
            with self.assertRaises(RuntimeError) as ctx:
                client.locate_sign(b"jpeg", 1000, 500)
            self.assertIn("timed out after 10s", str(ctx.exception))

    def test_raises_on_quad_null_with_reason(self):
        payload = {"choices": [{"message": {"content": json.dumps({"quad": None, "reason": "No flat signage detected"})}}]}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, _limit):
                return json.dumps(payload).encode()

        with patch("vision_client.urllib.request.urlopen", return_value=FakeResponse()):
            client = VisionClient(api_key="test-key", base_url="https://example.test/v1")
            with self.assertRaises(RuntimeError) as ctx:
                client.locate_sign(b"jpeg", 1000, 500)
            self.assertIn("could not find a replaceable planar sign", str(ctx.exception))
            self.assertIn("No flat signage detected", str(ctx.exception))

    def test_raises_on_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("vision_client.load_env_file"):
                with self.assertRaises(RuntimeError) as ctx:
                    VisionClient(api_key="")
                self.assertIn("BANANA_API_KEY is missing", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
