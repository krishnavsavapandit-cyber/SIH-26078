"""
Spatial Clustering & Extreme Event Detection Engine for SIH-26078.
Segments, identifies, contours, and characterizes coherent meteorological anomalies
from multi-variable EFI fields and standardized climatological anomaly z-scores.
"""

import numpy as np
from scipy import ndimage
from typing import List, Dict, Any, Tuple

from backend.app.config import domain_config

class EventDetector:
    """
    Identifies discrete spatial anomaly objects using connected component analysis
    and computes geometric, kinematic, and intensity metrics.
    """
    def __init__(self, config=domain_config, min_area_pixels: int = 4):
        self.config = config
        self.min_area_pixels = min_area_pixels
        self.lats = np.arange(config.lat_min, config.lat_max + 1e-5, config.grid_res_deg)
        self.lons = np.arange(config.lon_min, config.lon_max + 1e-5, config.grid_res_deg)
        self.num_lats = len(self.lats)
        self.num_lons = len(self.lons)

    def detect_events_at_lead(
        self,
        lead_time: int,
        efi_precip: np.ndarray,      # (lats, lons)
        ens_mean_precip: np.ndarray, # (lats, lons)
        ens_mean_mslp: np.ndarray,   # (lats, lons)
        ens_mean_wind: np.ndarray,   # (lats, lons)
        sot_precip: np.ndarray,      # (lats, lons)
        z_score_precip: np.ndarray = None,
        efi_threshold: float = 0.50,
        precip_threshold_mm: float = 35.0
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detects distinct extreme weather entities at a given lead time.
        """
        if z_score_precip is None:
            z_score_precip = np.zeros_like(efi_precip)

        # 1. Composite Extreme Anomaly Mask:
        # High EFI shifted above local climatology with high z-score or SOT exceedance
        extreme_mask = (efi_precip >= efi_threshold) & (
            (z_score_precip >= 1.8) | (sot_precip >= 0.2) | (ens_mean_precip >= precip_threshold_mm)
        )
        
        # 2. Connected Component Labeling
        structure = np.ones((3, 3), dtype=int)
        labeled_mask, num_features = ndimage.label(extreme_mask, structure=structure)
        
        detected_events = []
        filtered_mask = np.zeros_like(extreme_mask, dtype=np.float32)

        if num_features == 0:
            return detected_events, filtered_mask

        # 3. Analyze each discrete connected cluster
        for feature_id in range(1, num_features + 1):
            cluster_mask = (labeled_mask == feature_id)
            pixel_count = int(np.sum(cluster_mask))
            
            if pixel_count < self.min_area_pixels:
                continue # Discard small micro-anomalies / noise

            filtered_mask[cluster_mask] = 1.0

            lat_indices, lon_indices = np.where(cluster_mask)
            cluster_lats = self.lats[lat_indices]
            cluster_lons = self.lons[lon_indices]
            
            weights = np.maximum(0.1, ens_mean_precip[lat_indices, lon_indices]) * (1.0 + np.maximum(0.0, efi_precip[lat_indices, lon_indices]))
            total_weight = np.sum(weights)
            
            centroid_lat = float(np.sum(cluster_lats * weights) / total_weight)
            centroid_lon = float(np.sum(cluster_lons * weights) / total_weight)

            min_lat, max_lat = float(np.min(cluster_lats)), float(np.max(cluster_lats))
            min_lon, max_lon = float(np.min(cluster_lons)), float(np.max(cluster_lons))
            
            cluster_precip = ens_mean_precip[cluster_mask]
            cluster_efi = efi_precip[cluster_mask]
            cluster_mslp = ens_mean_mslp[cluster_mask]
            cluster_wind = ens_mean_wind[cluster_mask]
            cluster_sot = sot_precip[cluster_mask]

            peak_precip = float(np.max(cluster_precip))
            mean_precip = float(np.mean(cluster_precip))
            peak_efi = float(np.max(cluster_efi))
            mean_efi = float(np.mean(cluster_efi))
            min_mslp = float(np.min(cluster_mslp))
            max_wind = float(np.max(cluster_wind))
            max_sot = float(np.max(cluster_sot))

            pixel_area_km2 = (self.config.grid_res_deg * 111.0) * (self.config.grid_res_deg * 111.0 * np.cos(np.radians(centroid_lat)))
            area_km2 = float(pixel_count * pixel_area_km2)

            severity_score = float(np.clip(
                0.35 * peak_efi + 
                0.35 * min(1.0, peak_precip / 120.0) + 
                0.15 * min(1.0, max_wind / 28.0) +
                0.15 * min(1.0, max_sot / 2.5),
                0.0, 1.0
            ))

            if severity_score >= 0.65 or peak_precip >= 80.0:
                severity_category = "EXTREME"
            elif severity_score >= 0.40 or peak_precip >= 45.0:
                severity_category = "SEVERE"
            elif severity_score >= 0.25:
                severity_category = "MODERATE"
            else:
                severity_category = "WATCH"

            polygon_coords = self._extract_boundary_polygon(lat_indices, lon_indices)

            detection = {
                "detection_id": f"DET_{lead_time:03d}_{feature_id}",
                "lead_time": lead_time,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "bounding_box": [min_lat, min_lon, max_lat, max_lon],
                "area_km2": area_km2,
                "pixel_count": pixel_count,
                "peak_precip_mm": peak_precip,
                "mean_precip_mm": mean_precip,
                "peak_efi": peak_efi,
                "mean_efi": mean_efi,
                "min_mslp_hpa": min_mslp,
                "max_wind_ms": max_wind,
                "max_sot": max_sot,
                "severity_score": severity_score,
                "severity_category": severity_category,
                "polygon": polygon_coords
            }
            detected_events.append(detection)

        detected_events.sort(key=lambda x: x["severity_score"], reverse=True)
        return detected_events, filtered_mask

    def _extract_boundary_polygon(self, lat_indices: np.ndarray, lon_indices: np.ndarray) -> List[List[float]]:
        lats = self.lats[lat_indices]
        lons = self.lons[lon_indices]
        
        c_lat = np.mean(lats)
        c_lon = np.mean(lons)
        
        angles = np.arctan2(lats - c_lat, lons - c_lon)
        sort_order = np.argsort(angles)
        
        step = max(1, len(sort_order) // 24)
        hull_pts = []
        for idx in sort_order[::step]:
            hull_pts.append([float(lats[idx]), float(lons[idx])])
            
        if hull_pts and hull_pts[0] != hull_pts[-1]:
            hull_pts.append(hull_pts[0])
            
        return hull_pts
