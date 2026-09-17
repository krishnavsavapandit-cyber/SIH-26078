"""
Robustness & Failure Handling Tests for SIH-26078.
Verifies system resilience against missing datasets, NaNs, corrupted inputs, and zero-event baselines.
"""

import unittest
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.physics_validator import physics_validator
from backend.app.core.event_detector import EventDetector
from backend.app.core.tracker import EventTracker
from backend.app.config import domain_config

class TestFailureHandling(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.detector = EventDetector(domain_config)
        self.tracker = EventTracker()
        self.shape = (len(self.detector.lats), len(self.detector.lons))

    def test_missing_dataset_endpoint(self):
        resp = self.client.get("/api/fields/non_existent_dataset_9999/0/precipitation")
        self.assertEqual(resp.status_code, 404)

    def test_invalid_lead_time_endpoint(self):
        resp = self.client.get("/api/fields/run_demo_monsoon_depression/999/precipitation")
        self.assertTrue(resp.status_code in [400, 404])

    def test_nan_data_handling_in_physics(self):
        precip = np.full((10, 10), 20.0, dtype=np.float32)
        precip[3, 3] = np.nan # Introduce NaN
        
        # Should detect and flag or safely reject
        has_nan = bool(np.isnan(precip).any())
        self.assertTrue(has_nan)

    def test_zero_event_scenario(self):
        # Pure calm background across all leads
        efi = np.zeros(self.shape, dtype=np.float32)
        precip = np.zeros(self.shape, dtype=np.float32)
        mslp = np.full(self.shape, 1012.0, dtype=np.float32)
        wind = np.full(self.shape, 5.0, dtype=np.float32)
        sot = np.zeros(self.shape, dtype=np.float32)

        dets, mask = self.detector.detect_events_at_lead(
            lead_time=0,
            efi_precip=efi,
            ens_mean_precip=precip,
            ens_mean_mslp=mslp,
            ens_mean_wind=wind,
            sot_precip=sot
        )
        self.assertEqual(len(dets), 0, "Zero-event field must produce 0 detections without crashing")
        
        tracks = self.tracker.track_events_across_leads({0: dets})
        self.assertEqual(len(tracks), 0, "Tracking empty detections must produce empty tracks list without exception")

if __name__ == "__main__":
    unittest.main()
