"""
Unit tests for Synthetic Meteorological Data Engine.
"""

import unittest
import numpy as np
from pathlib import Path
from backend.app.config import domain_config, DATASET_DIR
from backend.app.core.synthetic_engine import SyntheticWeatherEngine

class TestSyntheticEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SyntheticWeatherEngine(domain_config)

    def test_grid_dimensions(self):
        self.assertGreater(self.engine.num_lats, 50)
        self.assertGreater(self.engine.num_lons, 50)
        self.assertEqual(self.engine.num_leads, len(domain_config.lead_times_hours))
        self.assertEqual(self.engine.num_members, domain_config.num_ensemble_members)

    def test_deterministic_generation(self):
        run_id = "test_det_run"
        ds1, gt1 = self.engine.generate_scenario(run_id=run_id, seed=123)
        ds2, gt2 = self.engine.generate_scenario(run_id=run_id, seed=123)
        
        np.testing.assert_allclose(
            ds1["precipitation"].values,
            ds2["precipitation"].values,
            err_msg="Synthetic generation must be bitwise reproducible with identical seed"
        )
        self.assertEqual(gt1["event_id"], gt2["event_id"])
        self.assertEqual(len(gt1["trajectory"]), len(gt2["trajectory"]))

    def test_physical_bounds(self):
        ds, gt = self.engine.generate_scenario(run_id="test_bounds_run", seed=456)
        
        precip = ds["precipitation"].values
        self.assertTrue(np.all(precip >= 0.0), "Precipitation must be non-negative")
        
        rh = ds["relative_humidity"].values
        self.assertTrue(np.all(rh >= 0.0) and np.all(rh <= 100.0), "RH must be between 0% and 100%")
        
        t2m = ds["temperature_2m"].values
        self.assertTrue(np.all(t2m >= 250.0) and np.all(t2m <= 340.0), "2m Temperature must be in plausible range")

    def test_hidden_ground_truth_integrity(self):
        ds, gt = self.engine.generate_scenario(run_id="test_gt_run", seed=789)
        self.assertIn("trajectory", gt)
        self.assertEqual(len(gt["trajectory"]), len(domain_config.lead_times_hours))
        for pt in gt["trajectory"]:
            self.assertIn("lat", pt)
            self.assertIn("lon", pt)
            self.assertIn("intensity_deficit_hpa", pt)
            self.assertIn("peak_precip_mm6h", pt)

if __name__ == "__main__":
    unittest.main()
