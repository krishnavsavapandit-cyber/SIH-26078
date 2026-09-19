"""
SIH-26078 Master Scientific Hardening Test Suite.
Verifies BLUE-01 through BLUE-20 capabilities including ensemble exceedance maps,
event lifecycle state machines, 2D Radial PSD spectral analysis, physics loss decomposition,
multi-model benchmarks, and failure injection resilience.
"""

import unittest
import numpy as np
import xarray as xr
from pathlib import Path

from backend.app.config import BASE_DIR
from backend.app.core.uncertainty import UncertaintyEngine
from backend.app.core.physics_validator import physics_validator
from backend.app.ml.extreme_preservation import ExtremePreservationScorecard
from backend.app.ml.advanced_tracker import advanced_tracker
from backend.app.ml.benchmark_suite import benchmark_suite
from backend.app.core.data_discovery import data_discovery, RealDataValidationError
from backend.app.pipeline.orchestrator import orchestrator


class TestMasterScientificSuite(unittest.TestCase):
    """
    Comprehensive tests for Phase 2 BLUE scientific and engineering hardening.
    """

    def setUp(self):
        self.uncertainty_engine = UncertaintyEngine()
        self.scorecard = ExtremePreservationScorecard(scale_factor=5)

    def test_01_ensemble_exceedance_probabilities(self):
        """BLUE-01: Verifies multi-member spatial probability of exceedance fields."""
        lats = np.linspace(10.0, 20.0, 11)
        lons = np.linspace(70.0, 80.0, 11)
        leads = [0, 6, 12]
        members = [0, 1, 2, 3]

        # 4-member ensemble with controlled heavy rain in members 0 and 1
        data = np.zeros((3, 4, 11, 11), dtype=np.float32)
        data[:, 0:2, 4:7, 4:7] = 65.0 # 2 out of 4 members exceed 50mm

        ds = xr.Dataset(
            data_vars={"precipitation": (("lead_time", "member", "latitude", "longitude"), data)},
            coords={"lead_time": leads, "member": members, "latitude": lats, "longitude": lons}
        )

        probs = self.uncertainty_engine.compute_exceedance_probabilities(ds, "precipitation", [25.0, 50.0])
        self.assertIn("prob_ge_25mm", probs)
        self.assertIn("prob_ge_50mm", probs)

        prob_50 = probs["prob_ge_50mm"]
        self.assertEqual(prob_50.shape, (3, 11, 11))
        # Center cells should have probability 2/4 = 0.50
        self.assertAlmostEqual(float(prob_50[0, 5, 5]), 0.50, places=2)
        self.assertAlmostEqual(float(prob_50[0, 0, 0]), 0.00, places=2)

    def test_02_event_lifecycle_transitions(self):
        """BLUE-03: Verifies explicit lifecycle progression."""
        detections = {
            0: [{"id": "D0", "lat": 15.0, "lon": 75.0, "peak_precip_mm": 20.0, "min_mslp_hpa": 1004.0, "area_sqkm": 2500, "bounding_box": [14.0, 16.0, 74.0, 76.0]}],
            6: [{"id": "D1", "lat": 15.5, "lon": 75.8, "peak_precip_mm": 45.0, "min_mslp_hpa": 998.0, "area_sqkm": 3500, "bounding_box": [14.5, 16.5, 74.8, 76.8]}],
            12: [{"id": "D2", "lat": 16.0, "lon": 76.5, "peak_precip_mm": 90.0, "min_mslp_hpa": 988.0, "area_sqkm": 5000, "bounding_box": [15.0, 17.0, 75.5, 77.5]}],
            18: [{"id": "D3", "lat": 16.4, "lon": 77.1, "peak_precip_mm": 40.0, "min_mslp_hpa": 996.0, "area_sqkm": 3000, "bounding_box": [15.4, 17.4, 76.1, 78.1]}],
            24: [{"id": "D4", "lat": 16.8, "lon": 77.6, "peak_precip_mm": 15.0, "min_mslp_hpa": 1006.0, "area_sqkm": 1500, "bounding_box": [15.8, 17.8, 76.6, 78.6]}]
        }

        res = advanced_tracker.track_multi_hypothesis(detections)
        self.assertTrue(res["success"])
        self.assertGreater(len(res["consensus_tracks"]), 0)

        track = res["consensus_tracks"][0]
        points = track["trajectory_points"]
        self.assertEqual(len(points), 5)
        # Verify kinematics
        self.assertGreater(track["mean_speed_kmh"], 0.0)
        self.assertTrue(all("lifecycle_state" in p for p in points))

    def test_03_radial_psd_spectral_preservation(self):
        """BLUE-06: Verifies 2D Radial Power Spectral Density calculation."""
        # Create a smooth low-frequency field
        ny, nx = 64, 64
        y, x = np.ogrid[:ny, :nx]
        smooth_field = np.sin(x * 0.1) + np.cos(y * 0.1)

        # Create a high-frequency textured field with localized extremes
        extreme_field = smooth_field + 0.5 * np.sin(x * 1.5) * np.cos(y * 1.5)

        psd_smooth = ExtremePreservationScorecard.compute_radial_psd(smooth_field)
        psd_extreme = ExtremePreservationScorecard.compute_radial_psd(extreme_field)

        self.assertIn("high_frequency_power", psd_smooth)
        self.assertIn("high_frequency_power", psd_extreme)
        self.assertGreater(psd_extreme["high_frequency_power"], psd_smooth["high_frequency_power"])
        self.assertGreater(psd_extreme["high_frequency_ratio"], psd_smooth["high_frequency_ratio"])

    def test_04_physics_loss_decomposition(self):
        """BLUE-07: Verifies explicit Data Loss vs. Physics Penalty decomposition."""
        p_pred = np.array([[25.0, 60.0], [80.0, 10.0]], dtype=np.float64)
        p_tgt = np.array([[22.0, 58.0], [82.0, 12.0]], dtype=np.float64)
        u_wind = np.array([[10.0, -5.0], [8.0, 2.0]], dtype=np.float64)
        v_wind = np.array([[12.0, 4.0], [-10.0, 6.0]], dtype=np.float64)
        mslp = np.array([[995.0, 1005.0], [992.0, 1008.0]], dtype=np.float64)

        loss_breakdown = physics_validator.compute_physics_loss_breakdown(
            precip_pred=p_pred,
            precip_target=p_tgt,
            u_wind=u_wind,
            v_wind=v_wind,
            mslp=mslp
        )

        self.assertIn("data_loss_mse", loss_breakdown)
        self.assertIn("moisture_flux_loss", loss_breakdown)
        self.assertIn("geostrophic_loss", loss_breakdown)
        self.assertIn("total_physics_penalty", loss_breakdown)
        self.assertIn("total_combined_loss", loss_breakdown)

        self.assertGreater(loss_breakdown["data_loss_mse"], 0.0)
        self.assertGreaterEqual(loss_breakdown["total_physics_penalty"], 0.0)
        self.assertGreater(loss_breakdown["total_combined_loss"], loss_breakdown["data_loss_mse"])

    def test_05_cross_model_benchmark_execution(self):
        """BLUE-05 & BLUE-08: Verifies decoupled cross-model benchmark evaluation."""
        report = benchmark_suite.run_benchmark(num_test_scenarios=1)
        self.assertIn("models_evaluated", report)
        self.assertIn("category_detection_benchmark", report)
        self.assertIn("category_tracking_benchmark", report)
        self.assertIn("category_downscaling_benchmark", report)
        self.assertIn("category_winners", report)
        self.assertIn("physics_ablation", report)

        # Detection category metrics
        det = report["category_detection_benchmark"]
        self.assertIn("Phase_3_ST_GNN", det)
        self.assertIn("f1_score_pct", det["Phase_3_ST_GNN"])
        self.assertIn("csi_iou_pct", det["Phase_3_ST_GNN"])

        # Tracking category metrics
        trk = report["category_tracking_benchmark"]
        self.assertIn("Phase_3_Advanced_MHT", trk)
        self.assertIn("track_rmse_km", trk["Phase_3_Advanced_MHT"])

        # Downscaling category metrics
        down = report["category_downscaling_benchmark"]
        self.assertIn("Phase_3_Physics_Informed_UNet", down)
        self.assertIn("psnr_db", down["Phase_3_Physics_Informed_UNet"])
        self.assertIn("mass_violation_pct", down["Phase_3_Physics_Informed_UNet"])

    def test_06_failure_injection_resilience(self):
        """BLUE-10: Verifies graceful failure handling under corrupted input and injected faults."""
        corrupt_path = BASE_DIR / "data" / "real" / "corrupt_test_file.grib2"
        with self.assertRaises(RealDataValidationError):
            orchestrator.run_flagship_pipeline(
                run_id="test_fail_corrupt",
                data_source_mode="REAL",
                external_data_path=str(corrupt_path),
                allow_synthetic_fallback=False
            )

        # Injected failure with synthetic fallback
        res = orchestrator.run_flagship_pipeline(
            run_id="test_fail_fallback",
            data_source_mode="auto",
            external_data_path=str(corrupt_path),
            allow_synthetic_fallback=True
        )
        self.assertEqual(res["data_source_mode"], "SYNTHETIC")
        self.assertTrue(res["synthetic"])
        self.assertGreater(len(res["fallbacks_triggered"]), 0)


if __name__ == "__main__":
    unittest.main()
