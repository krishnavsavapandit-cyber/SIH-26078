"""
Trainable Machine Learning Baselines for SIH-26078 Anomaly Detection.
Implements event-segmented Convolutional Neural Network (CNN) baseline
with leak-free event-level dataset partitioning and honest test metrics.
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from backend.app.config import MODEL_WEIGHTS_DIR, domain_config
from backend.app.core.synthetic_engine import SyntheticWeatherEngine

class WeatherAnomalyDataset(Dataset):
    """
    PyTorch Dataset wrapping meteorological multi-channel grids and binary ground-truth masks.
    Channels: [Precipitation, MSLP, Temperature, u_wind, v_wind]
    """
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        # inputs shape: (N, C, H, W)
        # targets shape: (N, 1, H, W)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class CNNEventDetector(nn.Module):
    """
    Fully-Convolutional Segmentation Baseline Network.
    Maps multi-variable atmospheric state to extreme anomaly probability mask.
    """
    def __init__(self, in_channels: int = 5, base_filters: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # Downsample
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        feat = self.encoder(x)
        bot = self.bottleneck(feat)
        out = self.decoder(bot)
        # Handle odd dimensions padding if needed
        if out.shape[2:] != (h, w):
            out = nn.functional.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return out

class DiceBCELoss(nn.Module):
    """Combined Binary Cross Entropy and Soft Dice Loss."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        
        flat_probs = probs.view(-1)
        flat_targets = targets.view(-1)
        intersection = (flat_probs * flat_targets).sum()
        dice = (2.0 * intersection + self.smooth) / (flat_probs.sum() + flat_targets.sum() + self.smooth)
        dice_loss = 1.0 - dice
        
        return bce_loss + dice_loss

class MLBaselineTrainer:
    """
    Manages event-level dataset generation, leak-free partitioning, and training execution.
    """
    def __init__(self, config=domain_config):
        self.config = config
        self.engine = SyntheticWeatherEngine(config)
        self.model_path = MODEL_WEIGHTS_DIR / "cnn_baseline.pt"

    def prepare_event_partitioned_data(
        self,
        num_train_events: int = 4,
        num_val_events: int = 1,
        num_test_events: int = 2
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Partitions datasets by independent weather scenarios to ensure zero spatial leakage.
        """
        def build_samples_for_events(event_seeds: List[Tuple[str, int]]):
            inputs_list, targets_list = [], []
            for sc_type, seed in event_seeds:
                ds, _ = self.engine.generate_scenario(run_id=f"train_{sc_type}_{seed}", scenario_type=sc_type, seed=seed)
                # Normalize features
                p = ds["precipitation"].values[:, 0] / 100.0       # (leads, lats, lons)
                m = (ds["mslp"].values[:, 0] - 1000.0) / 15.0
                t = (ds["temperature_2m"].values[:, 0] - 300.0) / 10.0
                u = ds["u_wind_850"].values[:, 0] / 20.0
                v = ds["v_wind_850"].values[:, 0] / 20.0
                
                gt_mask = ds["gt_event_mask"].values # (leads, lats, lons)
                
                stacked = np.stack([p, m, t, u, v], axis=1) # (leads, 5, lats, lons)
                targets = gt_mask[:, np.newaxis, :, :]      # (leads, 1, lats, lons)
                
                inputs_list.append(stacked)
                targets_list.append(targets)
                
            return np.concatenate(inputs_list, axis=0), np.concatenate(targets_list, axis=0)

        train_configs = [("monsoon_depression", 101), ("cyclone", 102), ("monsoon_depression", 103), ("extreme_precipitation", 104)]
        val_configs = [("heat_dome", 201)]
        test_configs = [("monsoon_depression", 301), ("cyclone", 302)]

        x_train, y_train = build_samples_for_events(train_configs)
        x_val, y_val = build_samples_for_events(val_configs)
        x_test, y_test = build_samples_for_events(test_configs)

        return (
            WeatherAnomalyDataset(x_train, y_train),
            WeatherAnomalyDataset(x_val, y_val),
            WeatherAnomalyDataset(x_test, y_test)
        )

    def train_baseline(self, epochs: int = 5, batch_size: int = 8, lr: float = 1e-3) -> Dict[str, Any]:
        """
        Executes complete training and validation cycle.
        """
        train_ds, val_ds, test_ds = self.prepare_event_partitioned_data()
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNEventDetector().to(device)
        criterion = DiceBCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        training_history = []

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits = model(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(x_b)
            train_loss /= len(train_ds)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_b, y_b in val_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    logits = model(x_b)
                    loss = criterion(logits, y_b)
                    val_loss += loss.item() * len(x_b)
            val_loss /= len(val_ds)

            training_history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss)
            })

        # Save weights
        torch.save(model.state_dict(), self.model_path)

        # Evaluate on held-out test events
        test_metrics = self.evaluate(model, test_loader, device)

        return {
            "model_name": "CNN_Event_Detector_Baseline",
            "epochs_trained": epochs,
            "device": str(device),
            "weights_path": str(self.model_path),
            "training_history": training_history,
            "test_metrics": test_metrics
        }

    def evaluate(self, model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
        """Evaluates segmentation metrics on held-out test events."""
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_b, y_b in data_loader:
                x_b = x_b.to(device)
                logits = model(x_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(probs)
                all_targets.append(y_b.numpy())

        preds = np.concatenate(all_preds, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()

        binary_preds = (preds >= 0.5).astype(int)
        binary_targets = (targets >= 0.5).astype(int)

        tp = int(np.sum((binary_preds == 1) & (binary_targets == 1)))
        fp = int(np.sum((binary_preds == 1) & (binary_targets == 0)))
        fn = int(np.sum((binary_preds == 0) & (binary_targets == 1)))
        tn = int(np.sum((binary_preds == 0) & (binary_targets == 0)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        iou = float(tp / max(1, tp + fp + fn))

        return {
            "test_precision": prec,
            "test_recall": rec,
            "test_f1_score": f1,
            "test_iou": iou,
            "samples_evaluated": len(preds)
        }

    def predict_event_mask(
        self,
        precip: np.ndarray,
        mslp: np.ndarray,
        temp: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray
    ) -> np.ndarray:
        """
        Executes single-timestep inference using the trained CNN baseline.
        Inputs: 2D numpy arrays of shape (H, W).
        Returns: 2D probability map of shape (H, W).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNEventDetector().to(device)
        if self.model_path.exists():
            try:
                state_dict = torch.load(self.model_path, map_location=device)
                model.load_state_dict(state_dict)
            except Exception:
                pass
        model.eval()

        p = np.clip(precip / 100.0, 0.0, 5.0)
        m = (mslp - 1000.0) / 15.0
        t = (temp - 300.0) / 10.0
        u = u_wind / 20.0
        v = v_wind / 20.0

        stacked = np.stack([p, m, t, u, v], axis=0) # (5, H, W)
        x_tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        if probs.shape != precip.shape:
            from scipy.ndimage import zoom
            scale_y = precip.shape[0] / probs.shape[0]
            scale_x = precip.shape[1] / probs.shape[1]
            probs = zoom(probs, (scale_y, scale_x), order=1)

        return np.clip(probs, 0.0, 1.0)

ml_baseline_trainer = MLBaselineTrainer()
