import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_client import _json_from_text, normalize_quad


class VisionClientParsingTests(unittest.TestCase):
    def test_parses_fenced_json(self):
        self.assertEqual(_json_from_text('```json\n{"quad": [[0,0],[1,0],[1,1],[0,1]]}\n```')["quad"][0], [0, 0])

    def test_normalizes_unit_coordinates(self):
        quad = normalize_quad([[0, 0], [1, 0], [1, 1], [0, 1]], 1000, 500)
        self.assertEqual(quad[2], [1000.0, 500.0])

    def test_rejects_tiny_quad(self):
        with self.assertRaises(ValueError):
            normalize_quad([[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01]], 1000, 500)


if __name__ == "__main__":
    unittest.main()
