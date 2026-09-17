"""
Unit tests for EFI (Extreme Forecast Index) and SOT Calculation Engine.
"""

import unittest
import numpy as np
from backend.app.config import domain_config
from backend.app.core.climatology import ClimatologyEngine
from backend.app.core.efi_engine import EFIEngine

class TestEFIEngine(unittest.TestCase):
    def setUp(self):
        self.clim_engine = ClimatologyEngine(domain_config)
        self.clim_ds = self.clim_engine.get_climatology()
        self.efi_engine = EFIEngine(self.clim_ds)
        self.num_lats = len(domain_config.lead_times_hours) # dummy
        self.lats = self.clim_engine.lats
        self.lons = self.clim_engine.lons

    def test_efi_bounds(self):
        # Normal distribution centered at climatology mean
        clim_mean = self.clim_ds["precip_mean"].values
        clim_std = self.clim_ds["precip_std"].values
        
        # Test 1: Ensemble matches climatology
        mock_ens = np.zeros((10, len(self.lats), len(self.lons)), dtype=np.float32)
        for m in range(10):
            mock_ens[m] = np.maximum(0.0, clim_mean + np.random.normal(0, 0.5, size=clim_mean.shape))
            
        res = self.efi_engine.compute_efi_field(mock_ens, variable="precipitation")
        efi = res["efi"]
        
        self.assertTrue(np.all(efi >= -1.0) and np.all(efi <= 1.0), "EFI values must be strictly bounded in [-1, +1]")
        # Mean EFI over baseline should be close to 0
        self.assertAlmostEqual(float(np.mean(efi)), 0.0, delta=0.35)

    def test_extreme_anomaly_detection(self):
        # Synthetic extreme event: 100 mm rain over Odisha
        mock_ens = np.zeros((10, len(self.lats), len(self.lons)), dtype=np.float32)
        mock_ens[:, 50:60, 60:70] = 120.0 # High rainfall spot
        
        res = self.efi_engine.compute_efi_field(mock_ens, variable="precipitation")
        efi_spot = res["efi"][50:60, 60:70]
        sot_spot = res["sot"][50:60, 60:70]
        
        self.assertTrue(np.all(efi_spot > 0.60), "Extreme precipitation spot must produce high EFI (>0.60)")
        self.assertTrue(np.any(sot_spot > 0.0), "Extreme precipitation exceeding P99 must produce positive SOT")

if __name__ == "__main__":
    unittest.main()
