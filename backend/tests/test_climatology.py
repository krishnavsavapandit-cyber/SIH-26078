"""
Unit tests for 30-Year Climatological Baseline Engine.
"""

import unittest
import numpy as np
from backend.app.config import domain_config
from backend.app.core.climatology import ClimatologyEngine

class TestClimatologyEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ClimatologyEngine(domain_config)
        self.clim_ds = self.engine.get_climatology()

    def test_climatology_variables(self):
        required_vars = [
            "precip_mean", "precip_std", "precip_p50", "precip_p90", "precip_p99",
            "mslp_mean", "mslp_std", "mslp_p01", "wind_mean", "temp_mean"
        ]
        for var in required_vars:
            self.assertIn(var, self.clim_ds, f"Climatology dataset must contain {var}")

    def test_percentile_monotonicity(self):
        p50 = self.clim_ds["precip_p50"].values
        p90 = self.clim_ds["precip_p90"].values
        p95 = self.clim_ds["precip_p95"].values
        p99 = self.clim_ds["precip_p99"].values
        
        self.assertTrue(np.all(p90 >= p50 - 1e-4), "P90 must be >= P50")
        self.assertTrue(np.all(p95 >= p90 - 1e-4), "P95 must be >= P90")
        self.assertTrue(np.all(p99 >= p95 - 1e-4), "P99 must be >= P95")

    def test_caching(self):
        engine2 = ClimatologyEngine(domain_config)
        self.assertTrue(engine2.clim_file.exists())
        self.assertTrue(engine2.meta_file.exists())

if __name__ == "__main__":
    unittest.main()
