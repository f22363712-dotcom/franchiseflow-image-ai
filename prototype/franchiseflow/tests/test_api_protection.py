import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROTOTYPE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTOTYPE_DIR))
import app as app_module


class ApiProtectionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"FRANCHISEFLOW_API_KEY": "test-secret"}, clear=False)
        self.env.start()
        app_module.app.config["TESTING"] = True
        app_module.JOBS.clear()
        self.client = app_module.app.test_client()

    def tearDown(self):
        self.env.stop()

    def test_api_requires_key_and_does_not_echo_token(self):
        response = self.client.get("/api/jobs")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json(), {"error": "authentication required"})
        self.assertNotIn("test-secret", response.get_data(as_text=True))
        self.assertEqual(response.headers.get("WWW-Authenticate"), "Bearer")

    def test_render_endpoint_is_protected_before_parsing_uploads(self):
        response = self.client.post("/api/render")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json(), {"error": "authentication required"})

    def test_bearer_and_x_api_key_authenticate(self):
        bearer = self.client.get("/api/jobs", headers={"Authorization": "Bearer test-secret"})
        self.assertEqual(bearer.status_code, 200)
        header = self.client.get("/api/jobs", headers={"X-API-Key": "test-secret"})
        self.assertEqual(header.status_code, 200)

    def test_health_is_minimal_anonymously_and_detailed_when_authorized(self):
        anonymous = self.client.get("/api/health")
        self.assertIn(anonymous.status_code, (200, 503))
        self.assertEqual(set(anonymous.get_json()), {"status", "version"})
        authorized = self.client.get("/api/health", headers={"X-API-Key": "test-secret"})
        self.assertIn(authorized.status_code, (200, 503))
        self.assertIn("worker", authorized.get_json())
        self.assertIn("storage", authorized.get_json())

    def test_demo_route_is_disabled_without_explicit_demo_mode(self):
        response = self.client.get("/api/demo_results", headers={"X-API-Key": "test-secret"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json(), {"error": "demo mode is disabled"})

    def test_demo_route_still_requires_auth_when_enabled(self):
        with patch.dict(os.environ, {"DEMO_MODE": "1"}, clear=False):
            anonymous = self.client.get("/api/demo_results")
            self.assertEqual(anonymous.status_code, 401)

    def test_outputs_require_auth_and_authorized_range_is_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "outputs"
            output_file = output_root / "range_job" / "branch_1.mp4"
            output_file.parent.mkdir(parents=True)
            output_file.write_bytes(b"0123456789")
            with patch.object(app_module, "OUTPUT_DIR", output_root):
                anonymous = self.client.get("/outputs/range_job/branch_1.mp4", headers={"Range": "bytes=0-3"})
                self.assertEqual(anonymous.status_code, 401)
                authorized = self.client.get(
                    "/outputs/range_job/branch_1.mp4",
                    headers={"X-API-Key": "test-secret", "Range": "bytes=0-3"},
                )
                self.assertEqual(authorized.status_code, 206)
                self.assertEqual(authorized.data, b"0123")
                authorized.close()

    def test_public_job_error_hides_absolute_path(self):
        job_id = "public_error_test"
        app_module.JOBS[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "error": r"ffmpeg failed at C:\private\uploads\secret.mp4",
        }
        with patch.object(app_module, "_job", return_value=app_module.JOBS[job_id]):
            response = self.client.get(f"/api/jobs/{job_id}", headers={"X-API-Key": "test-secret"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("private", response.get_data(as_text=True))
        self.assertEqual(response.get_json()["error"], "processing failed; server details are withheld")


if __name__ == "__main__":
    unittest.main()
