"""
Unit tests for Spatial Event Detection and Spatio-Temporal Tracking Engines.
"""

import unittest
import numpy as np
from backend.app.config import domain_config
from backend.app.core.event_detector import EventDetector
from backend.app.core.tracker import EventTracker

class TestDetectionAndTracking(unittest.TestCase):
    def setUp(self):
        self.detector = EventDetector(domain_config, min_area_pixels=3)
        self.tracker = EventTracker()
        self.shape = (len(self.detector.lats), len(self.detector.lons))

    def test_single_spot_detection(self):
        efi = np.zeros(self.shape, dtype=np.float32)
        precip = np.zeros(self.shape, dtype=np.float32)
        mslp = np.full(self.shape, 1008.0, dtype=np.float32)
        wind = np.full(self.shape, 12.0, dtype=np.float32)
        sot = np.zeros(self.shape, dtype=np.float32)

        # Place extreme cluster around lat idx 40, lon idx 50
        efi[38:43, 48:53] = 0.85
        precip[38:43, 48:53] = 95.0
        mslp[38:43, 48:53] = 992.0
        wind[38:43, 48:53] = 28.0
        sot[38:43, 48:53] = 1.5

        dets, mask = self.detector.detect_events_at_lead(
            lead_time=12,
            efi_precip=efi,
            ens_mean_precip=precip,
            ens_mean_mslp=mslp,
            ens_mean_wind=wind,
            sot_precip=sot
        )

        self.assertEqual(len(dets), 1, "Should detect exactly 1 extreme weather entity")
        d = dets[0]
        self.assertAlmostEqual(d["peak_efi"], 0.85, delta=0.01)
        self.assertAlmostEqual(d["peak_precip_mm"], 95.0, delta=0.01)
        self.assertEqual(d["severity_category"], "EXTREME")
        self.assertGreater(d["area_km2"], 1000.0)
        self.assertGreater(np.sum(mask), 0)

    def test_multi_lead_tracking_continuity(self):
        detections_by_lead = {}
        for idx, lead in enumerate([0, 6, 12]):
            efi = np.zeros(self.shape, dtype=np.float32)
            precip = np.zeros(self.shape, dtype=np.float32)
            mslp = np.full(self.shape, 1008.0, dtype=np.float32)
            wind = np.full(self.shape, 12.0, dtype=np.float32)
            sot = np.zeros(self.shape, dtype=np.float32)

            r_lat = 40 + idx * 2
            r_lon = 60 - idx * 3
            efi[r_lat:r_lat+4, r_lon:r_lon+4] = 0.80
            precip[r_lat:r_lat+4, r_lon:r_lon+4] = 80.0

            dets, _ = self.detector.detect_events_at_lead(
                lead_time=lead,
                efi_precip=efi,
                ens_mean_precip=precip,
                ens_mean_mslp=mslp,
                ens_mean_wind=wind,
                sot_precip=sot
            )
            detections_by_lead[lead] = dets

        tracks = self.tracker.track_events_across_leads(detections_by_lead)
        self.assertEqual(len(tracks), 1, "Should track the moving system as 1 continuous trajectory")
        t = tracks[0]
        self.assertEqual(len(t["trajectory_points"]), 3)
        self.assertEqual(t["duration_hours"], 12)
        self.assertGreater(t["total_distance_km"], 50.0)
        self.assertIn(t["heading_compass"], ["NW", "WNW", "W"])

if __name__ == "__main__":
    unittest.main()
