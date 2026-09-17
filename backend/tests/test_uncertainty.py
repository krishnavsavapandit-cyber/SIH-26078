"""
Unit tests for Ensemble Uncertainty Quantification Engine.
"""

import unittest
import numpy as np
from backend.app.config import domain_config
from backend.app.core.synthetic_engine import SyntheticWeatherEngine
from backend.app.core.uncertainty import UncertaintyEngine
from backend.app.core.tracker import EventTracker
from backend.app.core.event_detector import EventDetector
from backend.app.core.climatology import ClimatologyEngine
from backend.app.core.efi_engine import EFIEngine

class TestUncertaintyEngine(unittest.TestCase):
    def setUp(self):
        self.synthetic_engine = SyntheticWeatherEngine(domain_config)
        self.uncertainty_engine = UncertaintyEngine(domain_config)
        self.clim_engine = ClimatologyEngine(domain_config)
        self.efi_engine = EFIEngine(self.clim_engine.get_climatology())
        self.detector = EventDetector(domain_config)
        self.tracker = EventTracker()

    def test_grid_uncertainty(self):
        ds, _ = self.synthetic_engine.generate_scenario(run_id="test_unc_grid", seed=55)
        unc = self.uncertainty_engine.compute_ensemble_grid_uncertainty(ds, variable="precipitation")
        
        self.assertIn("mean", unc)
        self.assertIn("std", unc)
        self.assertIn("p10", unc)
        self.assertIn("p90", unc)
        
        self.assertTrue(np.all(unc["p90"] >= unc["p50"] - 1e-4))
        self.assertTrue(np.all(unc["p50"] >= unc["p10"] - 1e-4))

    def test_cone_radius_expansion(self):
        ds, _ = self.synthetic_engine.generate_scenario(run_id="test_cone", seed=77)
        
        dets_by_lead = {}
        for t_idx, lead in enumerate(ds["lead_time"].values[:5]):
            lead_int = int(lead)
            p_ens = ds["precipitation"].values[t_idx]
            m_ens = ds["mslp"].values[t_idx]
            u_ens = ds["u_wind_850"].values[t_idx]
            v_ens = ds["v_wind_850"].values[t_idx]
            
            efi_p = self.efi_engine.compute_efi_field(p_ens, "precipitation")
            efi_m = self.efi_engine.compute_efi_field(m_ens, "mslp")
            
            dets, _ = self.detector.detect_events_at_lead(
                lead_int, efi_p["efi"], efi_p["ens_mean"], efi_m["ens_mean"],
                np.mean(np.sqrt(u_ens**2 + v_ens**2), axis=0), efi_p["sot"]
            )
            dets_by_lead[lead_int] = dets
            
        tracks = self.tracker.track_events_across_leads(dets_by_lead)
        self.assertGreater(len(tracks), 0)
        
        cone_res = self.uncertainty_engine.compute_track_uncertainty_cone(ds, tracks[0])
        cone_pts = cone_res["cone_points"]
        
        self.assertGreater(len(cone_pts), 1)
        r_start = cone_pts[0]["cone_radius_km"]
        r_end = cone_pts[-1]["cone_radius_km"]
        self.assertGreater(r_end, r_start, "Uncertainty cone radius must expand over lead times")

if __name__ == "__main__":
    unittest.main()
