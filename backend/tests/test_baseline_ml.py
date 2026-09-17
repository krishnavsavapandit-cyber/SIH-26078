"""
Unit tests for Machine Learning Baseline Models.
"""

import unittest
import torch
import numpy as np
from backend.app.ml.baseline_models import CNNEventDetector, DiceBCELoss, ml_baseline_trainer

class TestMLBaselines(unittest.TestCase):
    def test_cnn_architecture_forward(self):
        model = CNNEventDetector(in_channels=5, base_filters=16)
        # Input batch: 2 samples, 5 channels, 64x64 grid
        x = torch.randn(2, 5, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 64, 64))

    def test_dice_bce_loss(self):
        loss_fn = DiceBCELoss()
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        loss = loss_fn(logits, targets)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_partitioned_training_smoke(self):
        # Quick 1-epoch training test
        res = ml_baseline_trainer.train_baseline(epochs=1, batch_size=4)
        self.assertEqual(res["model_name"], "CNN_Event_Detector_Baseline")
        self.assertEqual(res["epochs_trained"], 1)
        self.assertIn("test_metrics", res)
        self.assertIn("test_f1_score", res["test_metrics"])

if __name__ == "__main__":
    unittest.main()
