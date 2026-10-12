import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idempotency_store import IdempotencyConflict
from storage import (
    JsonStorageRepository,
    SQLiteStorageRepository,
    assert_storage_contract,
)


class StorageContractMixin:
    def exercise_contract(self, repository):
        self.assertEqual(assert_storage_contract(repository), ())
        job = repository.create_job("job_1", {"status": "queued", "batch_id": "batch_1"})
        self.assertEqual(job["job_id"], "job_1")
        self.assertEqual(repository.load_job("job_1")["status"], "queued")
        job["status"] = "running"
        repository.save_job("job_1", job)
        self.assertEqual(repository.list_jobs(status="running")[0]["job_id"], "job_1")

        batch = repository.create_batch("batch_1", {"status": "queued", "job_ids": ["job_1"]})
        self.assertEqual(repository.load_batch("batch_1")["batch_id"], "batch_1")
        self.assertEqual(repository.list_batches()[0]["batch_id"], "batch_1")

        reserved = repository.reserve_idempotency(
            "jobs", "request-1", "fingerprint-1", resource_id="job_1"
        )
        self.assertEqual(reserved["state"], "pending")
        self.assertEqual(repository.pending_idempotency()[0]["resource_id"], "job_1")
        committed = repository.commit_idempotency(
            "jobs", "request-1", "fingerprint-1", "job_1"
        )
        self.assertEqual(committed["state"], "committed")
        self.assertEqual(repository.pending_idempotency(), [])


class JsonAndSQLiteContractTests(StorageContractMixin, unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ff_sqlite_storage_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_json_backend_contract(self):
        self.exercise_contract(JsonStorageRepository(self.root / "json"))

    def test_sqlite_backend_contract(self):
        self.exercise_contract(SQLiteStorageRepository(self.root / "state.db"))


class SQLiteStorageRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ff_sqlite_storage_"))
        self.db_path = self.root / "state.db"
        self.repository = SQLiteStorageRepository(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_initializes_wal_foreign_keys_and_busy_timeout_per_connection(self):
        with self.repository._connection() as connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertEqual(
                connection.execute("PRAGMA busy_timeout").fetchone()[0],
                self.repository.busy_timeout_ms,
            )
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        self.assertTrue({"jobs", "batches", "idempotency_keys"}.issubset(tables))

    def test_idempotency_is_unique_and_conflict_is_visible_across_instances(self):
        first = SQLiteStorageRepository(self.db_path)
        second = SQLiteStorageRepository(self.db_path)
        reserved = first.reserve_idempotency(
            "jobs", "same-key", "fp-a", resource_id="job-a"
        )
        replay = second.reserve_idempotency(
            "jobs", "same-key", "fp-a", resource_id="job-b"
        )
        self.assertEqual(replay["resource_id"], reserved["resource_id"])
        with self.assertRaises(IdempotencyConflict):
            second.reserve_idempotency(
                "jobs", "same-key", "fp-b", resource_id="job-b"
            )
        self.assertTrue(second.release_idempotency("jobs", "same-key", "fp-a"))
        self.assertIsNone(first.get_idempotency("jobs", "same-key"))

    def test_compare_and_swap_rejects_stale_version(self):
        created = self.repository.create_job("cas-job", {"status": "queued"})
        self.assertEqual(created["version"], 0)
        updated = self.repository.compare_and_swap_job(
            "cas-job", 0, {"status": "running", "phase": "tracking"}
        )
        self.assertEqual(updated["version"], 1)
        self.assertEqual(self.repository.load_job("cas-job")["phase"], "tracking")
        self.assertIsNone(
            self.repository.compare_and_swap_job("cas-job", 0, {"status": "completed"})
        )
        self.assertEqual(self.repository.load_job("cas-job")["status"], "running")

    def test_tenant_scope_is_persisted_and_isolated(self):
        local = SQLiteStorageRepository(self.db_path)
        acme = SQLiteStorageRepository(self.db_path, tenant_id="acme")
        local.create_job("local-job", {"status": "queued"})
        acme.create_job("acme-job", {"status": "queued"})
        self.assertIsNone(local.load_job("acme-job"))
        self.assertIsNone(acme.load_job("local-job"))
        self.assertEqual(acme.load_job("acme-job")["tenant_id"], "acme")

    def test_payload_json_keeps_arbitrary_job_and_batch_fields(self):
        self.repository.create_job(
            "rich-job",
            {"status": "queued", "brand_profile_id": "brand-1", "scene_plan": {"x": 1}},
        )
        self.repository.create_batch(
            "rich-batch",
            {"status": "queued", "branch_ids": ["a", "b"], "audit": {"source": "test"}},
        )
        self.assertEqual(self.repository.load_job("rich-job")["scene_plan"], {"x": 1})
        self.assertEqual(self.repository.load_batch("rich-batch")["audit"], {"source": "test"})


if __name__ == "__main__":
    unittest.main()
