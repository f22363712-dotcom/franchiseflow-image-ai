import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class ReleaseContractTests(unittest.TestCase):
    def test_packaging_files_and_reproducible_command_exist(self):
        self.assertTrue((ROOT / "requirements.txt").exists())
        self.assertTrue((ROOT / "LICENSE").exists())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("ffmpeg -y -i spikes/franchiseflow/fixtures/real_storefront.mp4", readme)
        self.assertIn("/api/jobs", readme)

    def test_public_copy_does_not_make_unverified_perfect_output_claims(self):
        texts = [
            (ROOT / "README.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "FranchiseFlow_Demo脚本.md").read_text(encoding="utf-8"),
        ]
        banned = re.compile(r"Zero Jitter|100%|广播级|全自动并行生成|不保留原片音频")
        for text in texts:
            self.assertIsNone(banned.search(text))
        combined = "\n".join(texts)
        self.assertIn("重投影拟合残差", combined)
        self.assertIn("前景遮挡", combined)

    def test_tracked_sources_have_no_secret_patterns(self):
        for path in ROOT.rglob("*.py"):
            if ".git" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.assertIsNone(re.search(r"sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{20,}", text))


if __name__ == "__main__":
    unittest.main()
