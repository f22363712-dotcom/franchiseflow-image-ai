import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROTOTYPE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTOTYPE_DIR))

import app as app_module
from storage import SQLiteStorageRepository, storage_backend_from_environment


class StorageBackendConfigurationTests(unittest.TestCase):
    def test_invalid_backend_fails_with_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "FRANCHISEFLOW_STORAGE_BACKEND.*json.*sqlite"):
            storage_backend_from_environment({"FRANCHISEFLOW_STORAGE_BACKEND": "postgres"})

    def test_environment_defaults_to_json_and_explicit_sqlite_path(self):
        self.assertEqual(storage_backend_from_environment({}), "json")
        self.assertEqual(
            storage_backend_from_environment({"FRANCHISEFLOW_STORAGE_BACKEND": "SQLITE"}),
            "sqlite",
        )


class AppSQLiteBackendTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ff_app_sqlite_"))
        self.outputs = self.root / "outputs"
        self.batches = self.outputs / "batches"
        self.outputs.mkdir()
        self.batches.mkdir()
        self.repository = SQLiteStorageRepository(self.root / "state.db")
        self.patches = [
            patch.object(app_module, "STORAGE_BACKEND", "sqlite"),
            patch.object(app_module, "STORAGE_REPOSITORY", self.repository),
            patch.object(app_module, "OUTPUT_DIR", self.outputs),
            patch.object(app_module, "BATCH_DIR", self.batches),
            patch.object(app_module, "JOBS", {}),
            patch.object(app_module, "FUTURES", {}),
            patch.object(app_module, "SYNC_ACTIVE", 0),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_wrappers_and_idempotency_helpers_use_sqlite_without_json_records(self):
        job = app_module.create_job(self.outputs, "sqlite-job", {"status": "queued", "phase": "queued"})
        self.assertEqual(app_module.load_job(self.outputs, "sqlite-job")["job_id"], "sqlite-job")
        job["status"] = "running"
        app_module.save_job(self.outputs, "sqlite-job", job)
        self.assertEqual(self.repository.load_job("sqlite-job")["status"], "running")

        batch = app_module.create_batch(self.batches, "sqlite-batch", {"status": "queued", "job_ids": ["sqlite-job"]})
        self.assertEqual(app_module.load_batch(self.batches, "sqlite-batch")["batch_id"], "sqlite-batch")
        batch["status"] = "running"
        app_module.save_batch(self.batches, "sqlite-batch", batch)
        self.assertEqual(self.repository.load_batch("sqlite-batch")["status"], "running")

        app_module._idempotency_reserve("jobs", "sqlite-key", "fp", "sqlite-job")
        self.assertEqual(self.repository.get_idempotency("jobs", "sqlite-key")["state"], "pending")
        app_module._idempotency_record("jobs", "sqlite-key", "fp", "sqlite-job", 202)
        self.assertEqual(self.repository.get_idempotency("jobs", "sqlite-key")["state"], "committed")
        self.assertFalse((self.outputs / "idempotency.json").exists())

    def test_recovery_schedule_reads_sqlite_list_jobs(self):
        self.repository.create_job("queued-sqlite", {"status": "queued", "phase": "queued"})
        with patch.object(app_module, "_submit_job") as submit:
            app_module._schedule_queued_jobs()
        submit.assert_called_once_with("queued-sqlite")


if __name__ == "__main__":
    unittest.main()
