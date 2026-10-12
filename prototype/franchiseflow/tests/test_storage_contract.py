import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage import JsonStorageRepository, assert_storage_contract


class StorageContractTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ff_storage_contract_"))
        self.repository = JsonStorageRepository(self.root / "outputs")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_json_adapter_implements_contract_and_reports_limits(self):
        self.assertEqual(assert_storage_contract(self.repository), ())
        self.assertTrue(self.repository.capabilities["atomic_local_writes"])
        self.assertFalse(self.repository.capabilities["cross_process_claim"])
        self.assertFalse(self.repository.capabilities["tenant_scoping"])

    def test_job_and_batch_operations_preserve_existing_store_semantics(self):
        job = self.repository.create_job("job_1", {"status": "queued", "batch_id": "batch_1"})
        self.assertEqual(self.repository.load_job("job_1")["status"], "queued")
        job["status"] = "running"
        self.repository.save_job("job_1", job)
        self.assertEqual(self.repository.list_jobs(status="running")[0]["job_id"], "job_1")

        batch = self.repository.create_batch("batch_1", {"status": "queued", "job_ids": ["job_1"]})
        self.assertEqual(self.repository.load_batch("batch_1")["batch_id"], "batch_1")
        self.assertEqual(self.repository.list_batches()[0]["batch_id"], "batch_1")

    def test_idempotency_reservation_keeps_pending_then_commits(self):
        reserved = self.repository.reserve_idempotency(
            "jobs", "request-1", "fingerprint-1", resource_id="job_1"
        )
        self.assertEqual(reserved["state"], "pending")
        self.assertEqual(self.repository.pending_idempotency()[0]["resource_id"], "job_1")
        committed = self.repository.commit_idempotency(
            "jobs", "request-1", "fingerprint-1", "job_1"
        )
        self.assertEqual(committed["state"], "committed")
        self.assertEqual(self.repository.pending_idempotency(), [])


if __name__ == "__main__":
    unittest.main()
