import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_rejects_illegal_completed_to_queued_transition(self):
        job = create_job(self.root, "abc123", {})
        job["status"] = "completed"
        save_job(self.root, "abc123", job)
        job["status"] = "queued"
        with self.assertRaises(ValueError):
            save_job(self.root, "abc123", job)

    def test_save_retries_transient_windows_replace_error(self):
        job = create_job(self.root, "retry123", {"status": "queued"})
        job["status"] = "running"
        real_replace = __import__("os").replace
        calls = {"count": 0}

        def flaky_replace(source, destination):
            calls["count"] += 1
            if calls["count"] == 1:
                raise PermissionError("temporary sharing violation")
            return real_replace(source, destination)

        with patch("job_store.os.replace", side_effect=flaky_replace):
            save_job(self.root, "retry123", job)
        self.assertEqual(calls["count"], 2)
        self.assertEqual(load_job(self.root, "retry123")["status"], "running")


if __name__ == "__main__":
    unittest.main()
