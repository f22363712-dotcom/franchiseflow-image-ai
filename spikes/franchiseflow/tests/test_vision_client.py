import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import json

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
            result = VisionClient(api_key="test-key", base_url="https://example.test/v1").locate_sign(b"jpeg", 1000, 500)
        self.assertEqual(result["quad"][2], [900.0, 250.0])
        self.assertEqual(result["model"], "GLM-5.3-flash")


if __name__ == "__main__":
    unittest.main()
