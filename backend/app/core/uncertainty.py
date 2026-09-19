"""
Ensemble Uncertainty Quantification Engine for SIH-26078.
Calculates spatial spread, quantile distributions, ensemble member spaghetti tracks,
and geometric cones of uncertainty across medium-range forecast lead times.
"""

import math
import numpy as np
import xarray as xr
from typing import List, Dict, Any, Tuple

from backend.app.config import domain_config

class UncertaintyEngine:
    """
    Computes rigorous probabilistic uncertainty metrics and spatio-temporal cones.
    """
    def __init__(self, config=domain_config):
        self.config = config
        self.lats = np.arange(config.lat_min, config.lat_max + 1e-5, config.grid_res_deg)
        self.lons = np.arange(config.lon_min, config.lon_max + 1e-5, config.grid_res_deg)

    def compute_ensemble_grid_uncertainty(
        self,
        ds: xr.Dataset,
        variable: str = "precipitation"
    ) -> Dict[str, np.ndarray]:
        """
        Computes 3D (leads, lats, lons) ensemble statistics.
        """
        data = ds[variable].values # (leads, members, lats, lons)
        
        ens_mean = np.mean(data, axis=1)
        ens_std = np.std(data, axis=1)
        ens_p10 = np.percentile(data, 10, axis=1)
        ens_p50 = np.percentile(data, 50, axis=1)
        ens_p90 = np.percentile(data, 90, axis=1)
        
        # Coefficient of variation (relative spread)
        cv = np.where(ens_mean > 1e-3, ens_std / (ens_mean + 1e-5), 0.0)

        return {
            "mean": ens_mean.astype(np.float32),
            "std": ens_std.astype(np.float32),
            "p10": ens_p10.astype(np.float32),
            "p50": ens_p50.astype(np.float32),
            "p90": ens_p90.astype(np.float32),
            "cv": cv.astype(np.float32)
        }

    def compute_exceedance_probabilities(
        self,
        ds: xr.Dataset,
        variable: str = "precipitation",
        thresholds: List[float] = [25.0, 50.0, 100.0]
    ) -> Dict[str, np.ndarray]:
        """
        Computes spatial probability of exceedance fields across lead times:
        P(variable >= threshold) = (1/M) sum_{m=1}^M I(variable_m >= threshold).
        """
        data = ds[variable].values
        if data.ndim == 3:
            data = data[:, np.newaxis, :, :]

        probs = {}
        for thresh in thresholds:
            key = f"prob_ge_{int(thresh)}mm" if variable == "precipitation" else f"prob_ge_{int(thresh)}"
            prob_field = np.mean((data >= thresh).astype(np.float32), axis=1)
            probs[key] = prob_field

        return probs

    def compute_track_uncertainty_cone(
        self,
        ds: xr.Dataset,
        primary_track: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates ensemble member spaghetti trajectories and the composite uncertainty cone.
        """
        lead_times = ds["lead_time"].values
        if "ensemble_member" in ds.coords or "ensemble_member" in ds.dims:
            num_members = len(ds["ensemble_member"])
        elif "member" in ds.coords or "member" in ds.dims:
            num_members = len(ds["member"])
        else:
            num_members = 1
        
        member_tracks: List[List[Dict[str, Any]]] = [[] for _ in range(num_members)]
        cone_points: List[Dict[str, Any]] = []

        # Find member-specific local minimum MSLP / maximum precip near track centroid
        for pt in primary_track["trajectory_points"]:
            lead_idx = int(np.where(lead_times == pt["lead_time"])[0][0])
            c_lat, c_lon = pt["lat"], pt["lon"]
            
            # Sub-window around track centroid (+/- 3.0 degrees)
            lat_mask = (self.lats >= c_lat - 3.5) & (self.lats <= c_lat + 3.5)
            lon_mask = (self.lons >= c_lon - 3.5) & (self.lons <= c_lon + 3.5)
            
            sub_lats = self.lats[lat_mask]
            sub_lons = self.lons[lon_mask]
            
            member_lats = []
            member_lons = []
            member_precips = []
            member_mslps = []

            for m in range(num_members):
                sub_mslp = ds["mslp"].values[lead_idx, m][np.ix_(lat_mask, lon_mask)]
                sub_precip = ds["precipitation"].values[lead_idx, m][np.ix_(lat_mask, lon_mask)]
                
                # Member vortex center: local min MSLP
                min_idx = np.unravel_index(np.argmin(sub_mslp), sub_mslp.shape)
                m_lat = float(sub_lats[min_idx[0]])
                m_lon = float(sub_lons[min_idx[1]])
                m_peak_p = float(np.max(sub_precip))
                m_min_p = float(np.min(sub_mslp))

                member_lats.append(m_lat)
                member_lons.append(m_lon)
                member_precips.append(m_peak_p)
                member_mslps.append(m_min_p)

                member_tracks[m].append({
                    "lead_time": int(pt["lead_time"]),
                    "lat": m_lat,
                    "lon": m_lon,
                    "peak_precip_mm": m_peak_p,
                    "min_mslp_hpa": m_min_p
                })

            # Positional Spread & Uncertainty Cone Radius
            d_lats_km = (np.array(member_lats) - c_lat) * 111.0
            d_lons_km = (np.array(member_lons) - c_lon) * (111.0 * math.cos(math.radians(c_lat)))
            distances_km = np.sqrt(d_lats_km**2 + d_lons_km**2)
            
            # Cone radius: 90th percentile distance + baseline physical error growth
            p90_radius_km = float(np.percentile(distances_km, 90))
            base_growth_km = 30.0 + 1.2 * (pt["lead_time"] ** 1.1)
            cone_radius_km = float(max(p90_radius_km, base_growth_km))

            # Approximate circular cone perimeter coordinates
            cone_poly = self._generate_circle_polygon(c_lat, c_lon, cone_radius_km)

            cone_points.append({
                "lead_time": int(pt["lead_time"]),
                "center_lat": c_lat,
                "center_lon": c_lon,
                "cone_radius_km": cone_radius_km,
                "cone_polygon": cone_poly,
                "precip_p10": float(np.percentile(member_precips, 10)),
                "precip_p50": float(np.percentile(member_precips, 50)),
                "precip_p90": float(np.percentile(member_precips, 90)),
                "mslp_p10": float(np.percentile(member_mslps, 10)),
                "mslp_p50": float(np.percentile(member_mslps, 50)),
                "mslp_p90": float(np.percentile(member_mslps, 90)),
                "member_centroids": [[lat, lon] for lat, lon in zip(member_lats, member_lons)]
            })

        # Formulate full spaghetti tracks
        spaghetti = []
        for m, pts in enumerate(member_tracks):
            spaghetti.append({
                "member_id": m,
                "points": pts
            })

        return {
            "track_id": primary_track["track_id"],
            "cone_points": cone_points,
            "spaghetti_tracks": spaghetti
        }

    def _generate_circle_polygon(self, lat: float, lon: float, radius_km: float, num_pts: int = 24) -> List[List[float]]:
        """Generates latitude-longitude polygon vertices for an uncertainty circle."""
        poly = []
        for i in range(num_pts):
            theta = 2.0 * math.pi * i / num_pts
            d_lat = (radius_km * math.cos(theta)) / 111.0
            d_lon = (radius_km * math.sin(theta)) / (111.0 * math.cos(math.radians(lat)))
            poly.append([float(lat + d_lat), float(lon + d_lon)])
        poly.append(poly[0]) # close loop
        return poly
