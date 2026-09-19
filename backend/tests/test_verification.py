"""
Unit tests for Quantitative Verification Engine.
"""

import unittest
import numpy as np
from backend.app.core.verification import VerificationEngine

class TestVerificationEngine(unittest.TestCase):
    def setUp(self):
        self.verifier = VerificationEngine()

    def test_contingency_math(self):
        # Create test masks: 10x10
        gt_mask = np.zeros((1, 10, 10), dtype=np.float32)
        pred_mask = np.zeros((1, 10, 10), dtype=np.float32)
        
        # 4 pixels in GT, 4 pixels in Pred, 2 overlapping
        gt_mask[0, 2:4, 2:4] = 1.0 # 4 pixels
        pred_mask[0, 2:4, 3:5] = 1.0 # 4 pixels
        
        # TP = 2, FP = 2, FN = 2, TN = 94
        # Precision = 2/4 = 0.5, Recall = 2/4 = 0.5, F1 = 0.5
        # IoU = 2 / (2+2+2) = 2/6 = 0.333
        
        gt_metadata = {"trajectory": [{"lead_time": 0, "lat": 18.0, "lon": 85.0, "intensity_deficit_hpa": 10.0, "peak_precip_mm6h": 50.0}]}
        
        # Create dummy mock dataset
        class DummyDataset:
            def __init__(self):
                self.data = {
                    "lead_time": np.array([0]),
                    "gt_event_mask": np.array(gt_mask),
                    "precipitation": np.zeros((1, 5, 10, 10)),
                    "gt_precipitation": np.zeros((1, 10, 10)),
                    "mslp": np.zeros((1, 5, 10, 10)),
                    "gt_mslp": np.zeros((1, 10, 10)),
                    "temperature_2m": np.zeros((1, 5, 10, 10)),
                    "gt_temperature_2m": np.zeros((1, 10, 10)),
                    "u_wind_850": np.zeros((1, 5, 10, 10)),
                    "gt_u_wind_850": np.zeros((1, 10, 10)),
                    "v_wind_850": np.zeros((1, 5, 10, 10)),
                    "gt_v_wind_850": np.zeros((1, 10, 10))
                }
            def __contains__(self, key):
                return key in self.data
            def __getitem__(self, key):
                class ArrayWrapper:
                    def __init__(self, arr):
                        self.values = arr
                return ArrayWrapper(self.data[key])
                
        ds = DummyDataset()
        tracks = [{
            "track_id": "TRK_001",
            "trajectory_points": [{"lead_time": 0, "lat": 18.1, "lon": 85.2, "peak_precip_mm": 52.0, "min_mslp_hpa": 1002.0}]
        }]

        res = self.verifier.verify_run(ds, gt_metadata, tracks, pred_mask)
        c = res["contingency"]
        
        self.assertEqual(c["true_positives"], 2)
        self.assertEqual(c["false_positives"], 2)
        self.assertEqual(c["false_negatives"], 2)
        self.assertAlmostEqual(c["precision"], 0.5, delta=1e-3)
        self.assertAlmostEqual(c["recall"], 0.5, delta=1e-3)
        self.assertAlmostEqual(c["f1_score"], 0.5, delta=1e-3)
        self.assertAlmostEqual(c["spatial_iou"], 2.0 / 6.0, delta=1e-3)

        # Haversine distance from (18.0, 85.0) to (18.1, 85.2) ~ 23.8 km
        self.assertGreater(res["trajectory_error"]["mean_displacement_km"], 15.0)
        self.assertLess(res["trajectory_error"]["mean_displacement_km"], 35.0)

if __name__ == "__main__":
    unittest.main()
