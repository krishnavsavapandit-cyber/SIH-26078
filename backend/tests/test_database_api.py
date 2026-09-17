"""
Unit and integration tests for Database ORM and FastAPI REST API routes.
"""

import unittest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.database import SessionLocal, init_db
from backend.app.db import crud

class TestDatabaseAndAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.db = SessionLocal()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")

    def test_runs_and_events_endpoints(self):
        # Trigger quick test run
        resp = self.client.post("/api/pipeline/run", json={
            "run_id": "test_api_run_01",
            "scenario_type": "monsoon_depression",
            "seed": 999
        })
        self.assertEqual(resp.status_code, 200)
        run_data = resp.json()
        self.assertEqual(run_data["run_id"], "test_api_run_01")
        self.assertIn("verification", run_data)
        self.assertIn("alerts", run_data)

        # Query /api/runs
        runs_resp = self.client.get("/api/runs")
        self.assertEqual(runs_resp.status_code, 200)
        runs = runs_resp.json()
        self.assertTrue(any(r["run_id"] == "test_api_run_01" for r in runs))

        # Query /api/events
        events_resp = self.client.get("/api/events?run_id=test_api_run_01")
        self.assertEqual(events_resp.status_code, 200)
        events = events_resp.json()
        self.assertGreater(len(events), 0)

        # Query /api/verification/test_api_run_01
        ver_resp = self.client.get("/api/verification/test_api_run_01")
        self.assertEqual(ver_resp.status_code, 200)
        ver = ver_resp.json()
        self.assertIn("contingency", ver)
        self.assertIn("trajectory_error", ver)

        # Query /api/alerts/test_api_run_01
        alert_resp = self.client.get("/api/alerts/test_api_run_01")
        self.assertEqual(alert_resp.status_code, 200)
        alerts = alert_resp.json()
        self.assertGreater(len(alerts), 0)

if __name__ == "__main__":
    unittest.main()
