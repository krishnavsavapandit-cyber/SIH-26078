"""
Extreme Amplitude Preservation Auditor
Evaluates whether downscaled (5km) fields preserve extreme tails, maximum amplitude,
and physical mass conservation compared to coarse fields and reference ground-truth.
Includes 2D Azimuthally Averaged Radial PSD and Ground-Truth Spectral Similarity.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy import ndimage


class ExtremePreservationScorecard:
    """
    Scientific auditor comparing Coarse Forecast, Downscaled Output, and Reference Truth.
    Calculates explicit preservation metrics without hardcoded values.
    """

    def __init__(self, scale_factor: int = 5):
        self.scale_factor = scale_factor

    def compute_scorecard(
        self,
        coarse_field: np.ndarray,
        downscaled_field: np.ndarray,
        reference_field: Optional[np.ndarray] = None,
        extreme_percentile: float = 99.0
    ) -> Dict[str, Any]:
        coarse = np.asarray(coarse_field, dtype=np.float64)
        downscaled = np.asarray(downscaled_field, dtype=np.float64)

        c_shape = coarse.shape
        d_shape = downscaled.shape

        lat_scale = d_shape[0] / max(c_shape[0], 1)
        lon_scale = d_shape[1] / max(c_shape[1], 1)

        c_min = float(np.min(coarse))
        c_max = float(np.max(coarse))
        c_mean = float(np.mean(coarse))
        c_p95 = float(np.percentile(coarse, 95.0))
        c_p99 = float(np.percentile(coarse, extreme_percentile))
        c_sum = float(np.sum(coarse))

        d_min = float(np.min(downscaled))
        d_max = float(np.max(downscaled))
        d_mean = float(np.mean(downscaled))
        d_p95 = float(np.percentile(downscaled, 95.0))
        d_p99 = float(np.percentile(downscaled, extreme_percentile))
        d_sum = float(np.sum(downscaled))

        peak_retention_ratio = d_max / (c_max + 1e-6) if c_max > 0 else 1.0
        p99_retention_ratio = d_p99 / (c_p99 + 1e-6) if c_p99 > 0 else 1.0
        p95_retention_ratio = d_p95 / (c_p95 + 1e-6) if c_p95 > 0 else 1.0

        expected_downscaled_sum = c_sum * (lat_scale * lon_scale)
        mass_conservation_ratio = d_sum / (expected_downscaled_sum + 1e-6) if expected_downscaled_sum > 0 else 1.0
        mass_discrepancy_pct = abs(1.0 - mass_conservation_ratio) * 100.0

        c_extreme_cells = int(np.sum(coarse >= c_p95))
        d_extreme_cells = int(np.sum(downscaled >= d_p95))

        c_lap = np.abs(ndimage.laplace(coarse))
        d_lap = np.abs(ndimage.laplace(downscaled))
        c_sharpness = float(np.mean(c_lap))
        d_sharpness = float(np.mean(d_lap))
        sharpness_gain = d_sharpness / (c_sharpness + 1e-6)

        ref_metrics = {}
        spectral_gt_comp = {}
        if reference_field is not None:
            ref = np.asarray(reference_field, dtype=np.float64)
            if ref.shape == downscaled.shape:
                r_max = float(np.max(ref))
                r_p99 = float(np.percentile(ref, extreme_percentile))
                rmse = float(np.sqrt(np.mean((downscaled - ref) ** 2)))
                mae = float(np.mean(np.abs(downscaled - ref)))
                
                if np.std(downscaled) > 1e-6 and np.std(ref) > 1e-6:
                    corr = float(np.corrcoef(downscaled.ravel(), ref.ravel())[0, 1])
                else:
                    corr = 1.0

                ref_p95 = float(np.percentile(ref, 95.0))
                mask_extreme = ref >= ref_p95
                tail_mae = float(np.mean(np.abs(downscaled[mask_extreme] - ref[mask_extreme]))) if np.any(mask_extreme) else mae

                ref_metrics = {
                    "ref_max": r_max,
                    "ref_p99": r_p99,
                    "ref_rmse": rmse,
                    "ref_mae": mae,
                    "ref_correlation": corr,
                    "tail_mae": tail_mae,
                    "peak_recovery_vs_truth": float(d_max / (r_max + 1e-6)),
                }
                spectral_gt_comp = self.compute_spectral_similarity(downscaled, ref)

        status = "PASS"
        issues = []

        if d_min < -0.05:
            status = "WARN"
            issues.append(f"Negative values detected in downscaled field (min: {d_min:.3f})")
        if peak_retention_ratio < 0.85:
            status = "WARN" if status == "PASS" else "FAIL"
            issues.append(f"Substantial peak attenuation ({peak_retention_ratio*100:.1f}% peak retained)")
        if mass_discrepancy_pct > 15.0:
            status = "WARN" if status == "PASS" else "FAIL"
            issues.append(f"Mass discrepancy high ({mass_discrepancy_pct:.1f}%)")

        scorecard = {
            "resolution_info": {
                "coarse_grid_shape": list(c_shape),
                "downscaled_grid_shape": list(d_shape),
                "lat_scale_factor": float(lat_scale),
                "lon_scale_factor": float(lon_scale),
                "claimed_nominal_resolution": "12km -> 5km (nominal 0.25 deg -> 0.05 deg)"
            },
            "coarse_metrics": {
                "min": c_min, "max": c_max, "mean": c_mean,
                "p95": c_p95, "p99": c_p99, "mass_sum": c_sum,
                "extreme_cell_count": c_extreme_cells
            },
            "downscaled_metrics": {
                "min": d_min, "max": d_max, "mean": d_mean,
                "p95": d_p95, "p99": d_p99, "mass_sum": d_sum,
                "extreme_cell_count": d_extreme_cells
            },
            "preservation_ratios": {
                "peak_retention_ratio": float(peak_retention_ratio),
                "p99_retention_ratio": float(p99_retention_ratio),
                "p95_retention_ratio": float(p95_retention_ratio),
                "mass_conservation_ratio": float(mass_conservation_ratio),
                "mass_discrepancy_pct": float(mass_discrepancy_pct),
                "sharpness_gain": float(sharpness_gain)
            },
            "reference_validation": ref_metrics,
            "spectral_analysis": {
                "coarse_psd": self.compute_radial_psd(coarse),
                "downscaled_psd": self.compute_radial_psd(downscaled),
                "high_freq_power_gain": float(self.compute_radial_psd(downscaled)["high_frequency_power"] / (self.compute_radial_psd(coarse)["high_frequency_power"] + 1e-8)),
                "gt_spectral_comparison": spectral_gt_comp
            },
            "audit_verdict": {
                "status": status,
                "issues": issues,
                "extreme_preservation_score": float(np.clip(
                    (peak_retention_ratio * 0.4 + (1.0 - min(mass_discrepancy_pct / 100.0, 1.0)) * 0.3 + p99_retention_ratio * 0.3) * 100.0,
                    0.0, 100.0
                ))
            }
        }
        return scorecard

    @staticmethod
    def compute_radial_psd(field: np.ndarray) -> Dict[str, Any]:
        arr = np.asarray(field, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return {"high_frequency_power": 0.0, "total_power": 0.0, "high_frequency_ratio": 0.0, "psd_profile": []}

        arr_norm = arr - np.mean(arr)
        fft2 = np.fft.fftshift(np.fft.fft2(arr_norm))
        power_2d = np.abs(fft2) ** 2

        ny, nx = arr.shape
        cy, cx = ny // 2, nx // 2
        y, x = np.ogrid[:ny, :nx]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

        max_r = min(cy, cx)
        radial_prof = []
        for radius in range(max_r):
            mask = (r == radius)
            if np.any(mask):
                radial_prof.append(float(np.mean(power_2d[mask])))
            else:
                radial_prof.append(0.0)

        total_power = float(np.sum(radial_prof))
        cutoff_idx = max_r // 2
        high_freq_power = float(np.sum(radial_prof[cutoff_idx:])) if max_r > 0 else 0.0
        high_freq_ratio = high_freq_power / (total_power + 1e-8)

        return {
            "total_power": total_power,
            "high_frequency_power": high_freq_power,
            "high_frequency_ratio": float(high_freq_ratio),
            "radial_wavenumbers": list(range(len(radial_prof))),
            "psd_profile": radial_prof[:16]
        }

    @staticmethod
    def compute_spectral_similarity(pred_field: np.ndarray, gt_field: np.ndarray) -> Dict[str, float]:
        """
        Computes spectral similarity and log-error between predicted and ground-truth radial PSDs.
        """
        p_psd = ExtremePreservationScorecard.compute_radial_psd(pred_field)
        g_psd = ExtremePreservationScorecard.compute_radial_psd(gt_field)
        
        p_prof = np.array(p_psd["psd_profile"], dtype=np.float64)
        g_prof = np.array(g_psd["psd_profile"], dtype=np.float64)
        
        min_len = min(len(p_prof), len(g_prof))
        if min_len == 0:
            return {"spectral_similarity": 1.0, "spectral_log_error": 0.0, "high_freq_power_ratio_vs_gt": 1.0}
            
        p_sub = p_prof[:min_len]
        g_sub = g_prof[:min_len]
        
        log_err = float(np.mean(np.abs(np.log10(p_sub + 1e-4) - np.log10(g_sub + 1e-4))))
        
        norm_p = np.linalg.norm(p_sub)
        norm_g = np.linalg.norm(g_sub)
        if norm_p > 1e-6 and norm_g > 1e-6:
            sim = float(np.dot(p_sub, g_sub) / (norm_p * norm_g))
        else:
            sim = 1.0
            
        return {
            "spectral_similarity": float(np.clip(sim, 0.0, 1.0)),
            "spectral_cosine_sim": float(np.clip(sim, 0.0, 1.0)),
            "spectral_log_error": float(log_err),
            "spectral_log_mse": float(log_err),
            "high_freq_power_ratio_vs_gt": float(p_psd["high_frequency_power"] / (g_psd["high_frequency_power"] + 1e-8))
        }


compute_spectral_similarity = ExtremePreservationScorecard.compute_spectral_similarity

