"""
Learned Meteorological Downscaling Engine for SIH-26078.
Performs 5x spatial resolution super-resolution from 25km (coarse) to 5km (fine) grid.
Compares standard bicubic interpolation against a trainable physics-aware CNN downscaler.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.optim as optim

from backend.app.config import MODEL_WEIGHTS_DIR, domain_config

class SuperResolutionDownscaler(nn.Module):
    """
    Convolutional Super-Resolution Network for Meteorological Fields.
    Upscales coarse atmospheric fields by 5x (e.g. 128x120 -> 640x600 or patched 32x32 -> 160x160).
    """
    def __init__(self, in_channels: int = 2, num_filters: int = 32):
        super().__init__()
        # in_channels: [coarse_field, fine_elevation_downsampled]
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=5, padding=2)
        self.relu1 = nn.PReLU()
        
        self.res_block1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.PReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters)
        )
        
        self.res_block2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.PReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters)
        )

        # 5x Upsampling via bilinear + refinement convs
        self.upsample = nn.Upsample(scale_factor=5, mode='bilinear', align_corners=True)
        self.refine = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(num_filters, 1, kernel_size=3, padding=1),
            nn.ReLU() # Non-negative output for precipitation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.relu1(self.conv1(x))
        res1 = feat + self.res_block1(feat)
        res2 = res1 + self.res_block2(res1)
        up = self.upsample(res2)
        out = self.refine(up)
        return out

class DownscalingEngine:
    """
    Manages downscaling evaluation, bicubic baseline comparison, and ML model execution.
    """
    def __init__(self):
        self.model_path = MODEL_WEIGHTS_DIR / "downscaler_cnn.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_or_create_model()

    def _load_or_create_model(self) -> SuperResolutionDownscaler:
        model = SuperResolutionDownscaler().to(self.device)
        if self.model_path.exists():
            try:
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            except Exception as e:
                print(f"[WARN] Downscaler weight load notice: {e}")
        return model

    def bicubic_downscale(self, coarse_grid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Standard bicubic spline interpolation baseline."""
        zoom_y = target_shape[0] / coarse_grid.shape[0]
        zoom_x = target_shape[1] / coarse_grid.shape[1]
        fine = scipy.ndimage.zoom(coarse_grid, (zoom_y, zoom_x), order=3)
        return np.maximum(0.0, fine).astype(np.float32)

    def ml_downscale(self, coarse_grid: np.ndarray, orography_coarse: np.ndarray = None) -> np.ndarray:
        """Applies trained CNN super-resolution downscaler."""
        self.model.eval()
        if orography_coarse is None:
            orography_coarse = np.zeros_like(coarse_grid)

        # Normalize
        c_norm = coarse_grid / 100.0
        o_norm = orography_coarse / 3000.0
        
        inp = np.stack([c_norm, o_norm], axis=0)[np.newaxis, ...] # (1, 2, H, W)
        inp_tensor = torch.tensor(inp, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out_tensor = self.model(inp_tensor)
            out_grid = out_tensor.squeeze().cpu().numpy() * 100.0

        return np.maximum(0.0, out_grid).astype(np.float32)

    def compare_downscalers(
        self,
        coarse_field: np.ndarray,
        ground_truth_fine: np.ndarray,
        orography_coarse: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Calculates side-by-side metrics between Bicubic and ML Downscaler against fine truth.
        """
        target_shape = ground_truth_fine.shape
        bicubic_res = self.bicubic_downscale(coarse_field, target_shape)
        ml_res = self.ml_downscale(coarse_field, orography_coarse)

        # Handle size match
        if ml_res.shape != target_shape:
            ml_res = scipy.ndimage.zoom(ml_res, (target_shape[0] / ml_res.shape[0], target_shape[1] / ml_res.shape[1]), order=1)

        def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
            diff = pred - gt
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mae = float(np.mean(np.abs(diff)))
            
            # Pearson correlation
            f_p, f_g = pred.flatten(), gt.flatten()
            r = float(np.corrcoef(f_p, f_g)[0, 1]) if np.std(f_p) > 1e-4 and np.std(f_g) > 1e-4 else 0.0
            
            # PSNR
            max_val = max(1.0, float(np.max(gt)))
            mse = max(1e-6, np.mean(diff ** 2))
            psnr = float(20.0 * np.log10(max_val / np.sqrt(mse)))
            
            # Extreme P99 value preservation
            p99_gt = float(np.percentile(gt, 99))
            p99_pred = float(np.percentile(pred, 99))
            p99_error = abs(p99_pred - p99_gt) / max(1.0, p99_gt)

            return {
                "rmse": rmse,
                "mae": mae,
                "pearson_r": r,
                "psnr_db": psnr,
                "p99_relative_error": float(p99_error),
                "peak_intensity": float(np.max(pred)),
                "total_mass": float(np.sum(pred))
            }

        return {
            "bicubic_baseline": compute_metrics(bicubic_res, ground_truth_fine),
            "ml_downscaler": compute_metrics(ml_res, ground_truth_fine),
            "ground_truth_stats": {
                "peak_intensity": float(np.max(ground_truth_fine)),
                "p99_intensity": float(np.percentile(ground_truth_fine, 99)),
                "total_mass": float(np.sum(ground_truth_fine))
            }
        }

    def train_downscaler(self, num_samples: int = 60, epochs: int = 5) -> Dict[str, Any]:
        """Trains the CNN Downscaler on coarse-fine synthetic patches."""
        # Generate synthetic coarse-fine pairs
        rng = np.random.RandomState(42)
        x_data, y_data = [], []
        
        for _ in range(num_samples):
            # Fine patch (100x100)
            fine_y = np.zeros((100, 100), dtype=np.float32)
            # Add vortex peak
            cy, cx = rng.uniform(30, 70), rng.uniform(30, 70)
            y_grid, x_grid = np.ogrid[:100, :100]
            dist = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
            fine_y += 120.0 * np.exp(- (dist / 20.0)**2)
            
            # Coarse patch (20x20)
            coarse_x = scipy.ndimage.zoom(fine_y, 0.2, order=1)
            orog_x = rng.uniform(0, 1500, size=(20, 20)).astype(np.float32)
            
            inp = np.stack([coarse_x / 100.0, orog_x / 3000.0], axis=0) # (2, 20, 20)
            target = fine_y[np.newaxis, ...] / 100.0                     # (1, 100, 100)
            
            x_data.append(inp)
            y_data.append(target)

        x_t = torch.tensor(np.array(x_data), dtype=torch.float32).to(self.device)
        y_t = torch.tensor(np.array(y_data), dtype=torch.float32).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        history = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(x_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()
            history.append({"epoch": epoch, "loss": float(loss.item())})

        torch.save(self.model.state_dict(), self.model_path)
        return {"epochs": epochs, "history": history, "final_loss": history[-1]["loss"]}

downscaling_engine = DownscalingEngine()
