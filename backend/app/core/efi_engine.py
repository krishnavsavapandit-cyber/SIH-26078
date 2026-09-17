"""
Extreme Forecast Index (EFI) & Shift of Tails (SOT) Engine for SIH-26078.
Quantifies atmospheric anomalies by comparing ensemble forecast distributions
against the 30-year multi-decadal climatological reference CDFs.
"""

import numpy as np
import xarray as xr
from typing import Dict, Any, Tuple

class EFIEngine:
    """
    Computes ECMWF-inspired Extreme Forecast Index (EFI) and Shift of Tails (SOT)
    to identify abnormal and extreme weather shifts in medium-range forecasts.
    """
    def __init__(self, climatology_ds: xr.Dataset):
        self.clim = climatology_ds
        # Numerical quadrature points for EFI integral (avoiding division by zero at bounds)
        self.p_points = np.linspace(0.02, 0.98, 49) # 49 quantile evaluation points
        self.weights = 1.0 / np.sqrt(self.p_points * (1.0 - self.p_points))
        self.norm_factor = 2.0 / (np.pi * np.sum(self.weights) * (self.p_points[1] - self.p_points[0]))

    def compute_efi_field(
        self,
        ensemble_field: np.ndarray,  # shape: (members, lats, lons)
        variable: str = "precipitation"
    ) -> Dict[str, np.ndarray]:
        """
        Computes EFI, SOT, Standardized Anomaly (Z-score), and Ensemble Mean & Spread.
        """
        num_members, num_lats, num_lons = ensemble_field.shape
        
        # 1. Ensemble Mean and Spread (Std Dev)
        ens_mean = np.mean(ensemble_field, axis=0)
        ens_std = np.std(ensemble_field, axis=0)
        
        # 2. Extract Climatological Baseline parameters for variable
        if variable == "precipitation":
            clim_mean = self.clim["precip_mean"].values
            clim_std = self.clim["precip_std"].values
            clim_p90 = self.clim["precip_p90"].values
            clim_p99 = self.clim["precip_p99"].values
        elif variable == "mslp":
            clim_mean = self.clim["mslp_mean"].values
            clim_std = self.clim["mslp_std"].values
            clim_p90 = self.clim["mslp_mean"].values - 1.28 * self.clim["mslp_std"].values
            clim_p99 = self.clim["mslp_p01"].values
        elif "wind" in variable:
            clim_mean = self.clim["wind_mean"].values
            clim_std = self.clim["wind_std"].values
            clim_p90 = self.clim["wind_mean"].values + 1.28 * self.clim["wind_std"].values
            clim_p99 = self.clim["wind_p99"].values
        else:
            clim_mean = self.clim["temp_mean"].values
            clim_std = self.clim["temp_std"].values
            clim_p90 = self.clim["temp_mean"].values + 1.28 * self.clim["temp_std"].values
            clim_p99 = self.clim["temp_mean"].values + 2.33 * self.clim["temp_std"].values

        # 3. Standardized Anomaly Z-Score
        z_score = np.where(clim_std > 1e-4, (ens_mean - clim_mean) / (clim_std + 1e-5), 0.0)

        # 4. Compute EFI via Empirical CDF Integration
        # For each quantile p, evaluate climatological value Qc(p)
        # and calculate what fraction of ensemble members exceed or fall below Qc(p).
        efi_accum = np.zeros((num_lats, num_lons), dtype=np.float64)
        dp = self.p_points[1] - self.p_points[0]

        # Reshape ensemble field for fast broadcasting: (members, lats * lons)
        ens_flat = ensemble_field.reshape(num_members, -1) # (M, N)
        clim_mean_flat = clim_mean.flatten()
        clim_std_flat = clim_std.flatten()

        for p, w in zip(self.p_points, self.weights):
            # Climatological quantile Qc(p) assuming Gaussian/Gamma parameterization
            # z_p quantile:
            z_p = np.sqrt(2.0) * self._erfinv(2.0 * p - 1.0)
            qc_p = np.maximum(0.0 if variable == "precipitation" else -999.0, 
                              clim_mean_flat + z_p * clim_std_flat) # shape (N,)
            
            # Forecast CDF at Qc(p): F_f(Q_c(p)) = fraction of members <= qc_p
            # F_f shape: (N,)
            f_f = np.mean(ens_flat <= qc_p[np.newaxis, :], axis=0)
            
            # Difference (p - F_f)
            diff = (p - f_f)
            efi_accum += (w * diff * dp).reshape(num_lats, num_lons)

        # Scale by normalizer: 2/pi integral
        efi_grid = np.clip(efi_accum * (2.0 / np.pi), -1.0, 1.0).astype(np.float32)

        # 5. Shift of Tails (SOT)
        # SOT = (Forecast P90 - Clim P99) / (Clim P99 - Clim P90)
        ens_p90 = np.percentile(ensemble_field, 90, axis=0)
        sot_denominator = np.maximum(1.0, clim_p99 - clim_p90)
        sot_grid = np.maximum(0.0, (ens_p90 - clim_p99) / sot_denominator).astype(np.float32)

        return {
            "efi": efi_grid,
            "sot": sot_grid,
            "z_score": z_score.astype(np.float32),
            "ens_mean": ens_mean.astype(np.float32),
            "ens_std": ens_std.astype(np.float32),
            "ens_p90": ens_p90.astype(np.float32),
            "clim_mean": clim_mean.astype(np.float32),
            "clim_p99": clim_p99.astype(np.float32)
        }

    def _erfinv(self, x: float) -> float:
        """Approximation of inverse error function for normal quantile."""
        # Winitzki approximation
        a = 0.147
        x2 = x * x
        if abs(x) >= 1.0:
            return np.sign(x) * 4.0
        t1 = 2.0 / (np.pi * a) + np.log(1.0 - x2) / 2.0
        t2 = np.log(1.0 - x2) / a
        val = np.sqrt(np.maximum(0.0, np.sqrt(t1 * t1 - t2) - t1))
        return np.sign(x) * val
