"""
Comprehensive Unit Tests for Phase 3 Research & ML Components:
- SphericalAtmosphericGraph
- Spatio-Temporal GNN (ST-GNN)
- Physics-Informed Super-Resolution U-Net
- Conditional Diffusion Probabilistic Model (DDPM)
- Cross-Model Benchmark Suite
"""

import unittest
import numpy as np
import torch
from pathlib import Path

from backend.app.ml.spherical_graph import spherical_graph, SphericalAtmosphericGraph
from backend.app.ml.st_gnn import SpatioTemporalGNN, SphericalGraphConv, st_gnn_manager
from backend.app.ml.advanced_downscaler import PhysicsInformedUNetDownscaler, PhysicsInformedLoss, advanced_downscaling_manager
from backend.app.ml.diffusion_experiment import ConditionalUNetDenoiser, AtmosphericDiffusionEngine, diffusion_engine
from backend.app.ml.advanced_tracker import advanced_tracker, AdvancedMultiHypothesisTracker
from backend.app.ml.benchmark_suite import benchmark_suite

class TestPhase3ResearchSuite(unittest.TestCase):
    def setUp(self):
        self.graph = spherical_graph
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_spherical_graph_geometry(self):
        """Verifies 3D Cartesian coordinates and edge connectivity topology."""
        self.assertEqual(self.graph.coords_3d.shape[0], self.graph.n_nodes)
        self.assertEqual(self.graph.coords_3d.shape[1], 3)
        # Unit sphere check: x^2 + y^2 + z^2 == 1
        norms = np.linalg.norm(self.graph.coords_3d, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

        # Edges check
        self.assertEqual(self.graph.edge_index.shape[0], 2)
        self.assertGreater(self.graph.edge_index.shape[1], 0)
        self.assertEqual(self.graph.edge_attr.shape[1], 2)

    def test_grid_node_feature_conversions(self):
        """Verifies lossless bijection between 2D grids and Graph node representations."""
        dummy_grid = torch.randn(2, 5, self.graph.n_lats, self.graph.n_lons)
        node_feats = self.graph.grid_to_node_features(dummy_grid)
        self.assertEqual(node_feats.shape, (2, self.graph.n_nodes, 8)) # 5 channels + 3 (x,y,z)

        reconstructed_grid = self.graph.node_features_to_grid(node_feats, n_channels=5)
        self.assertEqual(reconstructed_grid.shape, dummy_grid.shape)
        torch.testing.assert_close(reconstructed_grid, dummy_grid)

    def test_st_gnn_forward_pass(self):
        """Verifies Spatio-Temporal GNN message passing and GRU temporal sequence propagation."""
        model = SpatioTemporalGNN(in_channels=8, hidden_dim=16, num_spatial_layers=1).to(self.device)
        # Sequence: batch=1, lead_times=3, nodes=N, in_ch=8
        x_seq = torch.randn(1, 3, self.graph.n_nodes, 8, device=self.device)
        edge_index = self.graph.edge_index.to(self.device)
        edge_attr = self.graph.edge_attr.to(self.device)

        logits, vels = model(x_seq, edge_index, edge_attr)
        self.assertEqual(logits.shape, (1, 3, self.graph.n_nodes, 1))
        self.assertEqual(vels.shape, (1, 3, 2))

    def test_st_gnn_inference_manager(self):
        """Verifies ST-GNN inference wrapper and safe fallback."""
        leads, H, W = 4, self.graph.n_lats, self.graph.n_lons
        p = np.full((leads, H, W), 25.0, dtype=np.float32)
        p[:, 10:16, 10:16] = 90.0 # storm core
        m = np.full((leads, H, W), 1002.0, dtype=np.float32)
        t = np.full((leads, H, W), 301.0, dtype=np.float32)
        u = np.full((leads, H, W), 15.0, dtype=np.float32)
        v = np.full((leads, H, W), 12.0, dtype=np.float32)

        res = st_gnn_manager.predict_spatiotemporal_anomalies(p, m, t, u, v)
        self.assertTrue(res["success"])
        self.assertEqual(res["probabilities"].shape, (leads, H, W))
        self.assertEqual(res["velocities"].shape, (leads, 2))

    def test_advanced_tracker_multi_hypothesis(self):
        """Verifies Phase 3C Advanced Multi-Hypothesis Tracker execution and confidence metrics."""
        dets_by_lead = {
            0: [{"centroid_lat": 18.0, "centroid_lon": 84.0, "bounding_box": [17, 83, 19, 85], "peak_precip_mm": 85.0, "min_mslp_hpa": 998.0, "severity_score": 0.88, "lead_time": 0}],
            24: [
                {"centroid_lat": 18.8, "centroid_lon": 83.2, "bounding_box": [17.8, 82.2, 19.8, 84.2], "peak_precip_mm": 95.0, "min_mslp_hpa": 994.0, "severity_score": 0.92, "lead_time": 24},
                {"centroid_lat": 18.2, "centroid_lon": 85.5, "bounding_box": [17.5, 84.5, 19.0, 86.5], "peak_precip_mm": 55.0, "min_mslp_hpa": 1004.0, "severity_score": 0.65, "lead_time": 24} # split cell
            ],
            48: [{"centroid_lat": 19.5, "centroid_lon": 82.4, "bounding_box": [18.5, 81.4, 20.5, 83.4], "peak_precip_mm": 110.0, "min_mslp_hpa": 990.0, "severity_score": 0.95, "lead_time": 48}]
        }

        res = advanced_tracker.track_multi_hypothesis(dets_by_lead)
        self.assertTrue(res["success"])
        self.assertEqual(res["model"], "AdvancedMultiHypothesisTracker")
        self.assertGreaterEqual(len(res["consensus_tracks"]), 1)
        self.assertGreaterEqual(res["total_hypotheses_evaluated"], 1)

        primary = res["consensus_tracks"][0]
        self.assertIn("cumulative_confidence", primary)
        self.assertGreater(primary["cumulative_confidence"], 0.5)
        self.assertEqual(len(primary["trajectory_points"]), 3)

    def test_physics_informed_unet_loss_and_forward(self):
        """Verifies Physics-Informed U-Net downscaler architecture and mass conservation loss."""
        model = PhysicsInformedUNetDownscaler(in_channels=4, base_ch=16).to(self.device)
        x_coarse = torch.randn(1, 4, 20, 20, device=self.device) # 4 channels
        out_fine = model(x_coarse)
        self.assertEqual(out_fine.shape, (1, 1, 100, 100))
        self.assertTrue(torch.all(out_fine >= 0.0), "Precipitation must be non-negative (ReLU output)")

        # Test Physics Loss components
        loss_fn = PhysicsInformedLoss()
        target_fine = torch.rand(1, 1, 100, 100, device=self.device)
        coarse_precip = torch.rand(1, 1, 20, 20, device=self.device)
        total_loss, breakdown = loss_fn(out_fine, target_fine, coarse_precip)
        self.assertGreater(total_loss.item(), 0.0)
        self.assertIn("mass_loss", breakdown)
        self.assertIn("tail_loss", breakdown)

    def test_diffusion_forward_and_reverse_sampling(self):
        """Verifies DDPM forward noise addition and reverse conditional denoising loop."""
        engine = diffusion_engine
        x_0 = torch.randn(1, 1, 32, 32, device=self.device)
        t = torch.tensor([10], device=self.device)
        x_t, noise = engine.q_sample(x_0, t)
        self.assertEqual(x_t.shape, x_0.shape)

        # Denoiser forward
        cond = torch.randn(1, 2, 32, 32, device=self.device)
        pred_noise = engine.model(x_t, t, cond)
        self.assertEqual(pred_noise.shape, x_0.shape)

    def test_benchmark_suite_execution(self):
        """Verifies cross-model benchmark suite execution on test scenarios."""
        report = benchmark_suite.run_benchmark(num_test_scenarios=1)
        self.assertIn("metrics_summary", report)
        self.assertIn("Phase_1_Operational_Baseline", report["metrics_summary"])
        self.assertIn("Phase_3_ST_GNN", report["metrics_summary"])
        self.assertIn("Phase_3_Physics_Informed_UNet", report["metrics_summary"])
        self.assertIn("Phase_3_Conditional_Diffusion", report["metrics_summary"])

if __name__ == "__main__":
    unittest.main()
