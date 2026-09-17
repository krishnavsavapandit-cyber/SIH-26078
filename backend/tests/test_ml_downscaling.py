"""
Unit tests for Learned Downscaling Engine.
"""

import unittest
import numpy as np
import torch
from backend.app.ml.downscaler_baseline import downscaling_engine, SuperResolutionDownscaler

class TestDownscaling(unittest.TestCase):
    def setUp(self):
        self.engine = downscaling_engine
        self.coarse = np.full((20, 20), 15.0, dtype=np.float32)
        # Add a storm peak
        self.coarse[8:12, 8:12] = 85.0

    def test_bicubic_interpolation_shape(self):
        target_shape = (100, 100)
        fine = self.engine.bicubic_downscale(self.coarse, target_shape)
        self.assertEqual(fine.shape, target_shape)
        self.assertTrue(np.all(fine >= 0.0), "Downscaled precipitation must remain non-negative")

    def test_ml_downscaler_forward_pass(self):
        model = SuperResolutionDownscaler()
        # Input tensor shape: (batch_size=2, channels=2, H=20, W=20)
        x = torch.randn(2, 2, 20, 20)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 100, 100))
        self.assertTrue(torch.all(out >= 0.0))

    def test_downscaler_training_loop(self):
        # Quick 2-epoch smoke test
        res = self.engine.train_downscaler(num_samples=10, epochs=2)
        self.assertEqual(res["epochs"], 2)
        self.assertLess(res["final_loss"], 1.0)

    def test_side_by_side_comparison(self):
        fine_gt = np.full((100, 100), 15.0, dtype=np.float32)
        fine_gt[40:60, 40:60] = 95.0
        
        comp = self.engine.compare_downscalers(self.coarse, fine_gt)
        self.assertIn("bicubic_baseline", comp)
        self.assertIn("ml_downscaler", comp)
        self.assertIn("rmse", comp["bicubic_baseline"])
        self.assertIn("psnr_db", comp["bicubic_baseline"])
        self.assertIn("p99_relative_error", comp["ml_downscaler"])

if __name__ == "__main__":
    unittest.main()
