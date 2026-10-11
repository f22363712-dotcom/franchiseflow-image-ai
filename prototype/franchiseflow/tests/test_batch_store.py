import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_store import create_batch, load_batch, save_batch


class BatchStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ff_batch_store_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_save_retries_transient_windows_replace_error(self):
        batch = create_batch(self.root, "retry123", {"status": "queued", "job_ids": []})
        batch["status"] = "running"
        real_replace = __import__("os").replace
        calls = {"count": 0}

        def flaky_replace(source, destination):
            calls["count"] += 1
            if calls["count"] == 1:
                raise PermissionError("temporary sharing violation")
            return real_replace(source, destination)

        with patch("batch_store.os.replace", side_effect=flaky_replace):
            save_batch(self.root, "retry123", batch)
        self.assertEqual(calls["count"], 2)
        self.assertEqual(load_batch(self.root, "retry123")["status"], "running")


if __name__ == "__main__":
    unittest.main()
