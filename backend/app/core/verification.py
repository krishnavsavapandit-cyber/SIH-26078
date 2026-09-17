"""
Quantitative Verification Engine for SIH-26078.
Rigorously evaluates forecast predictions and trajectories against hidden ground-truth telemetry
computing spatial contingency tables, displacement errors, continuous field errors, and timing offsets.
"""

import math
import numpy as np
import xarray as xr
from typing import Dict, Any, List, Optional

class VerificationEngine:
    """
    Computes genuine mathematical verification metrics between predictions and ground truth.
    """
    def __init__(self):
        pass

    def verify_run(
        self,
        ds: xr.Dataset,
        gt_metadata: Dict[str, Any],
        predicted_tracks: List[Dict[str, Any]],
        detected_masks: np.ndarray # (leads, lats, lons)
    ) -> Dict[str, Any]:
        """
        Runs comprehensive quantitative verification.
        """
        lead_times = ds["lead_time"].values
        num_leads = len(lead_times)
        
        gt_masks = ds["gt_event_mask"].values # (leads, lats, lons)
        gt_trajectory = {pt["lead_time"]: pt for pt in gt_metadata["trajectory"]}

        # 1. Pixel-Level Contingency Metrics across all leads
        pred_binary = (detected_masks > 0.5).astype(int)
        gt_binary = (gt_masks > 0.5).astype(int)

        tp = int(np.sum((pred_binary == 1) & (gt_binary == 1)))
        fp = int(np.sum((pred_binary == 1) & (gt_binary == 0)))
        fn = int(np.sum((pred_binary == 0) & (gt_binary == 1)))
        tn = int(np.sum((pred_binary == 0) & (gt_binary == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
        far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
        miss_rate = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

        spatial_intersection = tp
        spatial_union = tp + fp + fn
        iou = float(spatial_intersection / max(1, spatial_union))
        dice = float((2.0 * spatial_intersection) / max(1, (2 * tp + fp + fn)))

        # 2. Continuous Meteorological Field Errors (Ensemble Mean vs Ground Truth)
        field_metrics = {}
        for var in ["precipitation", "mslp", "temperature_2m", "u_wind_850", "v_wind_850"]:
            ens_mean = np.mean(ds[var].values, axis=1) # (leads, lats, lons)
            gt_field = ds[f"gt_{var}"].values          # (leads, lats, lons)
            
            diff = ens_mean - gt_field
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mae = float(np.mean(np.abs(diff)))
            
            # Pearson correlation
            flat_pred = ens_mean.flatten()
            flat_gt = gt_field.flatten()
            if np.std(flat_pred) > 1e-4 and np.std(flat_gt) > 1e-4:
                corr = float(np.corrcoef(flat_pred, flat_gt)[0, 1])
            else:
                corr = 0.0
                
            field_metrics[var] = {
                "rmse": rmse,
                "mae": mae,
                "pearson_r": corr
            }

        # 3. Trajectory and Centroid Displacement Verification
        primary_track = predicted_tracks[0] if predicted_tracks else None
        lead_breakdowns = []
        displacement_errors_km = []
        intensity_errors_precip = []
        intensity_errors_mslp = []

        for lead_time in lead_times:
            lead_int = int(lead_time)
            gt_pt = gt_trajectory.get(lead_int)
            
            # Find predicted point at this lead time
            pred_pt = None
            if primary_track:
                for pt in primary_track["trajectory_points"]:
                    if pt["lead_time"] == lead_int:
                        pred_pt = pt
                        break

            # Lead slice contingency
            t_idx = int(np.where(lead_times == lead_time)[0][0])
            p_slice = pred_binary[t_idx]
            g_slice = gt_binary[t_idx]
            
            t_tp = int(np.sum((p_slice == 1) & (g_slice == 1)))
            t_fp = int(np.sum((p_slice == 1) & (g_slice == 0)))
            t_fn = int(np.sum((p_slice == 0) & (g_slice == 1)))
            t_iou = float(t_tp / max(1, t_tp + t_fp + t_fn))

            if gt_pt and pred_pt:
                disp_km = self._haversine_distance(
                    pred_pt["lat"], pred_pt["lon"],
                    gt_pt["lat"], gt_pt["lon"]
                )
                precip_err = abs(pred_pt["peak_precip_mm"] - gt_pt["peak_precip_mm6h"])
                mslp_err = abs(pred_pt["min_mslp_hpa"] - (1012.0 - gt_pt["intensity_deficit_hpa"]))
                
                displacement_errors_km.append(disp_km)
                intensity_errors_precip.append(precip_err)
                intensity_errors_mslp.append(mslp_err)

                lead_breakdowns.append({
                    "lead_time": lead_int,
                    "gt_lat": gt_pt["lat"],
                    "gt_lon": gt_pt["lon"],
                    "pred_lat": pred_pt["lat"],
                    "pred_lon": pred_pt["lon"],
                    "displacement_error_km": disp_km,
                    "gt_peak_precip_mm": gt_pt["peak_precip_mm6h"],
                    "pred_peak_precip_mm": pred_pt["peak_precip_mm"],
                    "precip_error_mm": precip_err,
                    "step_iou": t_iou
                })
            elif gt_pt:
                # Missed detection at this step
                lead_breakdowns.append({
                    "lead_time": lead_int,
                    "gt_lat": gt_pt["lat"],
                    "gt_lon": gt_pt["lon"],
                    "pred_lat": None,
                    "pred_lon": None,
                    "displacement_error_km": None,
                    "gt_peak_precip_mm": gt_pt["peak_precip_mm6h"],
                    "pred_peak_precip_mm": 0.0,
                    "precip_error_mm": gt_pt["peak_precip_mm6h"],
                    "step_iou": 0.0
                })

        mean_disp_error_km = float(np.mean(displacement_errors_km)) if displacement_errors_km else 0.0
        max_disp_error_km = float(np.max(displacement_errors_km)) if displacement_errors_km else 0.0
        mean_precip_peak_err = float(np.mean(intensity_errors_precip)) if intensity_errors_precip else 0.0

        return {
            "contingency": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "csi_threat_score": csi,
                "false_alarm_ratio": far,
                "miss_rate": miss_rate,
                "spatial_iou": iou,
                "dice_coefficient": dice
            },
            "trajectory_error": {
                "mean_displacement_km": mean_disp_error_km,
                "max_displacement_km": max_disp_error_km,
                "mean_peak_precip_error_mm": mean_precip_peak_err,
                "track_points_evaluated": len(displacement_errors_km)
            },
            "field_metrics": field_metrics,
            "lead_time_breakdowns": lead_breakdowns
        }

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in kilometers."""
        r = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return float(r * c)
