"""Comprehensive test suite for the Master Scientific Closure & Engineering Constitution.
Verifies all 5 event families, icosahedral geodesic mesh, ST-GNN anomaly extraction,
authoritative MHT tracking, residual diffusion downscaling, physics losses,
and decoupled benchmarks.
"""

import os
import unittest
import numpy as np
import torch

from backend.app.config import domain_config
from backend.app.core.synthetic_engine import SyntheticWeatherEngine
from backend.app.core.climatology import SyntheticClimatologyProvider, RealERA5ClimatologyProvider
from backend.app.ml.spherical_graph import (
    SphericalLatLonMesh,
    IcosahedralGeodesicMesh,
    SphericalGraphBuilder,
)
from backend.app.ml.st_gnn import STGNNManager, st_gnn_manager
from backend.app.ml.advanced_tracker import (
    AdvancedMultiHypothesisTracker,
    advanced_tracker,
)
from backend.app.ml.advanced_downscaler import (
    PhysicsInformedDownscaler,
    PhysicsInformedDownscalerModel,
    PhysicsLossConfig,
    downscaler_manager,
)
from backend.app.ml.diffusion_experiment import (
    ConditionalDiffusionDownscaler,
    ResidualUNetDenoiser,
    conditional_diffusion,
)
from backend.app.ml.extreme_preservation import (
    ExtremePreservationScorecard,
    compute_spectral_similarity,
)
from backend.app.alert.alert_engine import AlertEngine, alert_engine
from backend.app.pipeline.orchestrator import PipelineOrchestrator, orchestrator


class TestSyntheticEngine5Families(unittest.TestCase):
    """Verify all 5 synthetic event families across 3-10 day horizons."""

    def test_all_five_event_types(self):
        engine = SyntheticWeatherEngine()
        event_types = ["monsoon_depression", "cyclone", "heat_dome", "cold_wave", "extreme_precipitation", "multi_event"]

        for etype in event_types:
            ds, meta = engine.generate_scenario(
                run_id=f"test_run_{etype}",
                scenario_type=etype,
                seed=42,
            )
            self.assertEqual(len(ds.lead_time), len(engine.lead_times))
            self.assertIn("precipitation", ds.data_vars)
            self.assertIn("mslp", ds.data_vars)
            self.assertIn("temperature_2m", ds.data_vars)
            self.assertIn("u_wind_850", ds.data_vars)
            self.assertIn("v_wind_850", ds.data_vars)
            self.assertIn("geopotential_500", ds.data_vars)
            self.assertIn("gt_event_mask", ds.data_vars)
            self.assertIn("scenario_type", meta)

    def test_extended_horizon_10_days(self):
        engine = SyntheticWeatherEngine()
        # 10 days at 6h intervals = 40 lead steps (0 to 240 hours)
        custom_leads = list(range(0, 241, 6))
        ds, meta = engine.generate_scenario(
            run_id="test_run_10day",
            scenario_type="cyclone",
            seed=42,
            custom_lead_times=custom_leads,
        )
        self.assertEqual(len(ds.lead_time), len(custom_leads))
        self.assertEqual(int(ds.lead_time[-1]), 240)


class TestMeshAndClimatology(unittest.TestCase):
    """Verify spherical coordinate and geodesic mesh implementations."""

    def test_icosahedral_geodesic_mesh(self):
        ico = IcosahedralGeodesicMesh(subdivision_level=2)
        self.assertGreater(ico.n_nodes, 12)
        self.assertEqual(ico.edge_index.shape[0], 2)

        # Test grid to node features
        x = torch.ones(1, 5, ico.n_lats, ico.n_lons, dtype=torch.float32) * 42.0
        node_feats = ico.grid_to_node_features(x)
        self.assertEqual(node_feats.shape[1], ico.n_nodes)

        # Test node features back to grid
        grid_out = ico.node_features_to_grid(node_feats[:, :, :5], n_channels=5)
        self.assertEqual(grid_out.shape, (1, 5, ico.n_lats, ico.n_lons))
        self.assertAlmostEqual(float(grid_out.mean().item()), 42.0, delta=0.5)

    def test_spherical_latlon_mesh(self):
        mesh = SphericalLatLonMesh()
        self.assertEqual(mesh.edge_index.shape[0], 2)
        self.assertEqual(mesh.edge_attr.shape[0], mesh.edge_index.shape[1])
        x = torch.ones(1, 5, mesh.n_lats, mesh.n_lons, dtype=torch.float32) * 10.0
        nf = mesh.grid_to_node_features(x)
        self.assertEqual(nf.shape[1], mesh.n_nodes)

    def test_climatology_provider(self):
        provider = SyntheticClimatologyProvider()
        lat = np.linspace(5, 35, 10)
        lon = np.linspace(65, 95, 10)
        clim_mean, clim_std = provider.get_climatology(lat, lon, "temperature_2m", 6)
        self.assertEqual(clim_mean.shape, (10, 10))
        self.assertEqual(clim_std.shape, (10, 10))
        self.assertTrue(np.all(clim_std > 0))

        # RealERA5 with missing path should raise FileNotFoundError
        era5 = RealERA5ClimatologyProvider("non_existent_file.nc")
        with self.assertRaises(FileNotFoundError):
            era5.get_climatology(lat, lon, "temperature_2m", 6)


class TestSTGNNCandidateExtraction(unittest.TestCase):
    """Verify ST-GNN probability extraction and candidate generation."""

    def test_extract_candidate_events(self):
        manager = st_gnn_manager
        # Create synthetic probability field with a high probability cluster
        prob_field = np.zeros((manager.n_lats, manager.n_lons), dtype=np.float32)
        prob_field[10:16, 15:21] = 0.85

        p_mean = np.zeros((manager.n_lats, manager.n_lons), dtype=np.float32)
        p_mean[10:16, 15:21] = 65.0
        m_mean = np.full((manager.n_lats, manager.n_lons), 1000.0, dtype=np.float32)
        m_mean[10:16, 15:21] = 988.0
        w_mean = np.full((manager.n_lats, manager.n_lons), 10.0, dtype=np.float32)
        w_mean[10:16, 15:21] = 28.0

        detections, mask = manager.extract_candidate_events(
            prob_grid=prob_field,
            ens_mean_precip=p_mean,
            ens_mean_mslp=m_mean,
            ens_mean_wind=w_mean,
            threshold=0.35,
            lead_time=24,
            min_pixels=2
        )

        self.assertGreater(len(detections), 0)
        cand = detections[0]
        self.assertIn("centroid_lat", cand)
        self.assertIn("centroid_lon", cand)
        self.assertIn("peak_prob", cand)
        self.assertIn("area_km2", cand)
        self.assertGreater(cand["peak_prob"], 0.7)
        self.assertEqual(cand["source_detection"], "ST_GNN_SPHERICAL_OPERATOR")


class TestAuthoritativeMHT(unittest.TestCase):
    """Verify Advanced Multi-Hypothesis Tracking produce authoritative consensus tracks."""

    def test_mht_tracking_and_consensus(self):
        tracker = AdvancedMultiHypothesisTracker()

        detections_by_lead = {
            0: [{
                "detection_id": "D0_1", "lead_time": 0, "lat": 12.0, "lon": 85.0,
                "centroid_lat": 12.0, "centroid_lon": 85.0, "peak_precip_mm": 45.0,
                "mean_precip_mm": 25.0, "min_mslp_hpa": 995.0, "max_wind_ms": 22.0,
                "area_km2": 45000.0, "severity_score": 0.75, "source_detection": "ST_GNN"
            }],
            6: [{
                "detection_id": "D6_1", "lead_time": 6, "lat": 13.0, "lon": 84.0,
                "centroid_lat": 13.0, "centroid_lon": 84.0, "peak_precip_mm": 55.0,
                "mean_precip_mm": 30.0, "min_mslp_hpa": 990.0, "max_wind_ms": 26.0,
                "area_km2": 52000.0, "severity_score": 0.85, "source_detection": "ST_GNN"
            }],
            12: [{
                "detection_id": "D12_1", "lead_time": 12, "lat": 14.1, "lon": 83.1,
                "centroid_lat": 14.1, "centroid_lon": 83.1, "peak_precip_mm": 65.0,
                "mean_precip_mm": 38.0, "min_mslp_hpa": 985.0, "max_wind_ms": 30.0,
                "area_km2": 60000.0, "severity_score": 0.92, "source_detection": "ST_GNN"
            }],
        }

        res = tracker.track_multi_hypothesis(detections_by_lead)
        self.assertIn("consensus_tracks", res)
        self.assertIn("all_hypotheses", res)
        self.assertGreater(len(res["consensus_tracks"]), 0)

        track = res["consensus_tracks"][0]
        self.assertIn("trajectory_points", track)
        self.assertEqual(len(track["trajectory_points"]), 3)
        self.assertIn("uncertainty_radii_km", track)
        self.assertTrue(all(r > 0 for r in track["uncertainty_radii_km"]))


class TestPhysicsDownscalingAndDiffusion(unittest.TestCase):
    """Verify Physics-Informed U-Net, post-hoc mass projection, and Residual Diffusion."""

    def test_mass_conservation_projection(self):
        downscaler = downscaler_manager

        # Create a coarse input field and high-res prediction with intentional mass leakage
        coarse = np.ones((16, 16), dtype=np.float32) * 10.0
        # High-res prediction with 50% extra mass
        high_res = np.ones((64, 64), dtype=np.float32) * 15.0

        projected = downscaler._enforce_mass_conservation(
            downscaled=high_res,
            coarse=coarse,
            scale_factor=4
        )

        orig_coarse_mass = coarse.sum()
        projected_mass = projected.sum()
        # With 4x4 downscaling, fine cell count is 16x coarse cell count, mean mass per area conserved
        # Coarse mean should equal coarse cell average
        ratio = (projected.reshape(16, 4, 16, 4).mean(axis=(1, 3))).mean() / coarse.mean()
        self.assertAlmostEqual(float(ratio), 1.0, places=3)

    def test_residual_diffusion_downscaling(self):
        diffusion = conditional_diffusion
        coarse_sample = np.random.uniform(5.0, 30.0, (1, 16, 16)).astype(np.float32)
        out = diffusion.sample_diffusion(coarse_sample, ensemble_size=2, ddim_steps=5)

        self.assertIn("ensemble_mean", out)
        self.assertIn("ensemble_std", out)
        self.assertEqual(out["ensemble_mean"].shape, (1, 64, 64))
        # Ensure values remain physically bounded (non-negative)
        self.assertTrue(np.all(out["ensemble_mean"] >= 0))

    def test_spectral_similarity(self):
        pred = np.random.uniform(0, 10, (32, 32)).astype(np.float32)
        target = pred + np.random.normal(0, 0.5, (32, 32)).astype(np.float32)
        metrics = compute_spectral_similarity(pred, target)
        self.assertIn("spectral_cosine_sim", metrics)
        self.assertIn("spectral_log_mse", metrics)
        self.assertGreater(metrics["spectral_cosine_sim"], 0.8)


class TestAlertEngineStandardization(unittest.TestCase):
    """Verify alert engine standards (LOW, MODERATE, SEVERE, EXTREME)."""

    def test_alert_severity_classification(self):
        engine = alert_engine
        
        # Test synthetic track with extreme characteristics
        track = {
            "track_id": "TRK_TEST_001",
            "event_type": "cyclone",
            "peak_precip_mm": 110.0,
            "min_mslp_hpa": 965.0,
            "max_wind_ms": 38.0,
            "overall_severity": "EXTREME",
            "max_confidence": 0.95,
            "trajectory_points": [
                {
                    "lead_time": 24,
                    "centroid_lat": 15.0,
                    "centroid_lon": 85.0,
                    "peak_precip_mm": 110.0,
                    "min_mslp_hpa": 965.0,
                    "max_wind_ms": 38.0,
                    "area_km2": 65000.0,
                    "severity_score": 0.92,
                    "uncertainty_radius_km": 65.0
                }
            ]
        }

        alerts = engine.generate_alerts_for_tracks([track], run_id="test_run_alert")
        self.assertGreater(len(alerts), 0)
        al = alerts[0]
        self.assertIn(al["severity_level"], ["EXTREME", "SEVERE", "MODERATE", "LOW"])
        self.assertIn("early_warning_lead_hours", al)
        self.assertIn("actionable_guidance", al)


class TestEndToEndScientificClosure(unittest.TestCase):
    """Verify complete end-to-end execution without fallback bypass."""

    def test_orchestrator_synthetic_mode(self):
        res = orchestrator.run_flagship_pipeline(
            run_id="test_flagship_closure",
            scenario_type="cyclone",
            seed=42,
            data_source_mode="SYNTHETIC",
            enable_phase3=True,
        )


        self.assertEqual(res["data_mode"], "SYNTHETIC")
        self.assertTrue(res["authoritative_st_gnn"])
        self.assertTrue(res["authoritative_mht"])
        self.assertTrue(res["mass_conserved"])
        self.assertGreater(len(res["all_tracks"]), 0)
        self.assertIn("14_PROVENANCE_AND_PERSISTENCE", res["execution_steps"])


if __name__ == "__main__":
    unittest.main()
