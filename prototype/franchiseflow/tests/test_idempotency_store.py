import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from werkzeug.datastructures import MultiDict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idempotency_store import (
    IdempotencyConflict,
    IdempotencyError,
    IdempotencyStore,
    normalize_key,
    request_fingerprint,
)


class IdempotencyStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ff_idempotency_"))
        self.store = IdempotencyStore(self.root / "idempotency.json")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_put_and_reload_survives_new_store_instance(self):
        entry = self.store.put("jobs", "request-1", "fp-1", "job123")
        self.assertEqual(entry["resource_id"], "job123")
        reloaded = IdempotencyStore(self.root / "idempotency.json")
        self.assertEqual(reloaded.get("jobs", "request-1")["resource_id"], "job123")
        self.assertTrue((self.root / "idempotency.json").exists())

    def test_same_key_different_fingerprint_is_conflict(self):
        self.store.put("jobs", "request-1", "fp-1", "job123")
        with self.assertRaises(IdempotencyConflict):
            self.store.put("jobs", "request-1", "fp-2", "job456")

    def test_scope_keeps_job_and_batch_keys_independent(self):
        self.store.put("jobs", "request-1", "fp-1", "job123")
        self.store.put("batches", "request-1", "fp-2", "batch123")
        self.assertEqual(self.store.get("jobs", "request-1")["resource_id"], "job123")
        self.assertEqual(self.store.get("batches", "request-1")["resource_id"], "batch123")

    def test_invalid_key_is_rejected(self):
        self.assertIsNone(normalize_key(None))
        with self.assertRaises(IdempotencyError):
            normalize_key(" ")
        with self.assertRaises(IdempotencyError):
            normalize_key("x\n")
        with self.assertRaises(IdempotencyError):
            normalize_key("x" * 256)

    def test_request_fingerprint_rewinds_uploaded_streams(self):
        file_item = Mock(filename="clip.mp4", content_type="video/mp4", stream=io.BytesIO(b"video"))
        request = Mock(form=MultiDict([("brand_profile_id", "brand_1")]), files=MultiDict([("videos", file_item)]))
        fingerprint = request_fingerprint(request, "batches")
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(file_item.stream.tell(), 0)

    def test_json_state_is_atomic_and_secret_free(self):
        self.store.put("jobs", "request-1", "fp-1", "job123")
        content = json.loads((self.root / "idempotency.json").read_text(encoding="utf-8"))
        self.assertEqual(content["version"], 1)
        self.assertEqual(list(content["entries"])[0].split("\u0000"), ["jobs", "request-1"])


if __name__ == "__main__":
    unittest.main()
