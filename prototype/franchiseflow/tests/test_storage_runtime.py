from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage import JsonStorageRepository, SQLiteStorageRepository
from storage_runtime import build_repository


class StorageRuntimeFactoryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ff_storage_runtime_"))
        self.output_dir = self.root / "outputs"
        self.batch_dir = self.output_dir / "batches"
        self.idempotency_path = self.output_dir / "idempotency.json"
        self.output_dir.mkdir()
        self.batch_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_builds_json_repository_with_configured_paths(self):
        repository = build_repository(
            backend="json",
            output_dir=self.output_dir,
            batch_dir=self.batch_dir,
            idempotency_path=self.idempotency_path,
        )

        self.assertIsInstance(repository, JsonStorageRepository)
        self.assertEqual(repository.output_dir, self.output_dir)
        self.assertEqual(repository.batch_dir, self.batch_dir)
        self.assertEqual(repository.idempotency_path, self.idempotency_path)

    def test_builds_sqlite_repository_with_explicit_path(self):
        sqlite_path = self.root / "state" / "custom.db"
        repository = build_repository(
            backend="sqlite",
            output_dir=self.output_dir,
            batch_dir=self.batch_dir,
            idempotency_path=self.idempotency_path,
            sqlite_path=sqlite_path,
        )

        self.assertIsInstance(repository, SQLiteStorageRepository)
        self.assertEqual(repository.db_path, sqlite_path)
        self.assertTrue(sqlite_path.exists())

    def test_sqlite_uses_default_path_under_output_dir(self):
        repository = build_repository(
            backend="sqlite",
            output_dir=self.output_dir,
            batch_dir=self.batch_dir,
            idempotency_path=self.idempotency_path,
        )

        self.assertEqual(repository.db_path, self.output_dir / "franchiseflow.db")
        self.assertTrue(repository.db_path.exists())

    def test_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "backend.*json.*sqlite"):
            build_repository(
                backend="postgres",
                output_dir=self.output_dir,
                batch_dir=self.batch_dir,
                idempotency_path=self.idempotency_path,
            )


if __name__ == "__main__":
    unittest.main()
