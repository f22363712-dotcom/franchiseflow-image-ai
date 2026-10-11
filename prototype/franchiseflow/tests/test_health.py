import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


PROTOTYPE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTOTYPE_DIR))
import app as app_module
from idempotency_store import IdempotencyStore


class HealthEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.uploads = self.root / "uploads"
        self.outputs = self.root / "outputs"
        self.profiles = self.root / "profiles"
        self.batches = self.outputs / "batches"
        self.packages = self.batches / "packages"
        for directory in (self.uploads, self.outputs, self.profiles, self.batches, self.packages):
            directory.mkdir(parents=True, exist_ok=True)
        self.patches = [
            patch.object(app_module, "UPLOAD_DIR", self.uploads),
            patch.object(app_module, "OUTPUT_DIR", self.outputs),
            patch.object(app_module, "PROFILE_DIR", self.profiles),
            patch.object(app_module, "BATCH_DIR", self.batches),
            patch.object(app_module, "BATCH_PACKAGE_DIR", self.packages),
            patch.object(app_module, "IDEMPOTENCY_STORE", IdempotencyStore(self.outputs / "idempotency.json")),
            patch.object(app_module, "FUTURES", {}),
            patch.object(app_module, "SYNC_ACTIVE", 0),
            patch.object(app_module, "_RECOVERY_STARTED", False),
        ]
        for item in self.patches:
            item.start()
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def _job(self, job_id, status):
        directory = self.outputs / job_id
        directory.mkdir()
        (directory / "job.json").write_text(json.dumps({"job_id": job_id, "status": status}), encoding="utf-8")

    def test_health_is_read_only_and_does_not_trigger_recovery(self):
        self._job("queued1", "queued")
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        app_module.app.config["TESTING"] = False
        with patch("runtime_health.shutil.which", return_value="/secret/tool/path"), patch.object(app_module, "_recover_jobs") as recover:
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "healthy")
        self.assertEqual(response.get_json()["persistence"]["idempotency"]["state"], "not_initialized")
        recover.assert_not_called()
        self.assertFalse(app_module._RECOVERY_STARTED)
        after = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertNotIn(str(self.root), response.get_data(as_text=True))
        self.assertNotIn("/secret/tool/path", response.get_data(as_text=True))

    def test_health_counts_running_worker_and_durable_queue(self):
        self._job("queued1", "queued")
        self._job("queued2", "queued")
        self._job("running1", "running")
        self._job("finished1", "completed")
        future = MagicMock()
        future.done.return_value = False
        future.running.return_value = True
        waiting_future = MagicMock()
        waiting_future.done.return_value = False
        waiting_future.running.return_value = False
        with patch.object(app_module, "FUTURES", {"running1": future, "queued1": waiting_future}), patch.object(app_module, "SYNC_ACTIVE", 1), patch("runtime_health.shutil.which", return_value="tool"):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        worker = response.get_json()["worker"]
        self.assertEqual(worker["active"], 2)
        self.assertEqual(worker["queued"], 2)
        self.assertEqual(worker["scheduled"], 2)
        self.assertEqual(response.get_json()["persistence"]["jobs"]["records"], 4)

    def test_health_reports_missing_tools_and_unwritable_storage(self):
        with patch("runtime_health.shutil.which", return_value=None), patch("runtime_health.os.access", return_value=False):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 503)
        body = response.get_json()
        self.assertEqual(body["status"], "degraded")
        self.assertEqual(body["tools"], {"ffmpeg": False, "ffprobe": False})
        self.assertFalse(body["storage"]["uploads"]["writable"])
        self.assertFalse(body["storage"]["outputs"]["writable"])

    def test_health_reports_unreadable_json_without_exposing_contents(self):
        secret = "LOCAL_PRIVATE_VALUE"
        (self.outputs / "idempotency.json").write_text(secret, encoding="utf-8")
        directory = self.outputs / "broken_job"
        directory.mkdir()
        (directory / "job.json").write_text(secret, encoding="utf-8")
        with patch("runtime_health.shutil.which", return_value="tool"):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 503)
        persistence = response.get_json()["persistence"]
        self.assertEqual(persistence["idempotency"]["state"], "unreadable")
        self.assertEqual(persistence["jobs"]["unreadable"], 1)
        self.assertNotIn(secret, response.get_data(as_text=True))
        self.assertNotIn("broken_job", response.get_data(as_text=True))


class ServerBindTests(unittest.TestCase):
    def test_bind_defaults_and_env_overrides(self):
        from runtime_health import server_bind
        self.assertEqual(server_bind({}), ("127.0.0.1", 5000))
        self.assertEqual(server_bind({"HOST": "0.0.0.0", "PORT": "8080"}), ("0.0.0.0", 8080))

    def test_invalid_bind_configuration_fails_clearly(self):
        from runtime_health import server_bind
        for env in ({"PORT": "wat"}, {"PORT": "0"}, {"PORT": "65536"}, {"HOST": "\n"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                server_bind(env)


if __name__ == "__main__":
    unittest.main()
