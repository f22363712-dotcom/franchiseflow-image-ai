import shutil
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from job_store import create_job, load_job, save_job


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ff_job_store_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_create_and_reload_job_from_atomic_json(self):
        job = create_job(self.root, "abc123", {"branch_files": ["branch_1.png"], "status": "queued"})
        self.assertEqual(job["status"], "queued")
        self.assertEqual(load_job(self.root, "abc123")["branch_files"], ["branch_1.png"])

        job["status"] = "completed"
        save_job(self.root, "abc123", job)
        self.assertEqual(load_job(self.root, "abc123")["status"], "completed")
        self.assertTrue((self.root / "abc123" / "job.json").exists())

    def test_missing_job_returns_none(self):
        self.assertIsNone(load_job(self.root, "missing"))

    def test_rejects_path_like_job_id(self):
        with self.assertRaises(ValueError):
            create_job(self.root, "..", {})
        with self.assertRaises(ValueError):
            load_job(self.root, "../outside")


if __name__ == "__main__":
    unittest.main()
