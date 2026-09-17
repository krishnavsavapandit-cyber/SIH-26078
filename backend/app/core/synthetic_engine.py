"""
Synthetic Meteorological Data Engine for SIH-26078.
Generates physically consistent, multi-variable ensemble weather forecasts
along with hidden ground-truth telemetry for extreme atmospheric anomalies.
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import xarray as xr

from backend.app.config import domain_config, DATASET_DIR

class SyntheticWeatherEngine:
    """
    Generates deterministic, physically grounded atmospheric datasets
    with controlled forecast perturbations and exact ground-truth tracks.
    """
    def __init__(self, config=domain_config):
        self.config = config
        self.lats = np.arange(config.lat_min, config.lat_max + 1e-5, config.grid_res_deg)
        self.lons = np.arange(config.lon_min, config.lon_max + 1e-5, config.grid_res_deg)
        self.num_lats = len(self.lats)
        self.num_lons = len(self.lons)
        self.lead_times = config.lead_times_hours
        self.num_leads = len(self.lead_times)
        self.num_members = config.num_ensemble_members

        self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)
        self.elevation_map = self._generate_orography()

    def _generate_orography(self) -> np.ndarray:
        elev = np.zeros((self.num_lats, self.num_lons), dtype=np.float32)
        for i, lat in enumerate(self.lats):
            for j, lon in enumerate(self.lons):
                if lat >= 27.0:
                    elev[i, j] += min(4500.0, 1000.0 * (lat - 27.0) ** 1.2)
                if 8.0 <= lat <= 20.0 and 73.0 <= lon <= 76.5:
                    dist_to_crest = abs(lon - 74.5)
                    elev[i, j] += max(0.0, 1500.0 * math.exp(-0.5 * (dist_to_crest / 0.8) ** 2))
                if 12.0 <= lat <= 24.0 and 75.0 <= lon <= 83.0:
                    elev[i, j] += 400.0 + 100.0 * math.sin(lat * 0.2)
        return elev

    def generate_scenario(
        self,
        run_id: str = "run_demo_monsoon_depression",
        scenario_type: str = "monsoon_depression",
        seed: int = 42,
        displacement_bias_km: float = 25.0,
        intensity_bias_pct: float = -5.0
    ) -> Tuple[xr.Dataset, Dict[str, Any]]:
        rng = np.random.RandomState(seed)

        gt_metadata = self._initialize_ground_truth(scenario_type, rng)

        gt_arrays = {var: np.zeros((self.num_leads, self.num_lats, self.num_lons), dtype=np.float32) 
                     for var in self.config.variables}
        gt_masks = np.zeros((self.num_leads, self.num_lats, self.num_lons), dtype=np.float32)

        for t_idx, lead_hours in enumerate(self.lead_times):
            gt_center_lat, gt_center_lon, gt_intensity = self._get_gt_position_and_intensity(
                gt_metadata, lead_hours
            )
            gt_metadata["trajectory"].append({
                "lead_time": lead_hours,
                "lat": float(gt_center_lat),
                "lon": float(gt_center_lon),
                "intensity_deficit_hpa": float(gt_intensity),
                "max_wind_ms": float(gt_intensity * 2.2),
                "peak_precip_mm6h": float(min(220.0, 35.0 + gt_intensity * 6.5))
            })

            bg = self._compute_background_state(lead_hours)
            vortex, mask = self._compute_vortex_perturbation(
                gt_center_lat, gt_center_lon, gt_intensity, lead_hours
            )

            gt_masks[t_idx] = mask
            gt_arrays["mslp"][t_idx] = bg["mslp"] + vortex["mslp"]
            gt_arrays["temperature_2m"][t_idx] = bg["temperature_2m"] + vortex["temperature_2m"] - (self.elevation_map * 0.0065)
            gt_arrays["u_wind_850"][t_idx] = bg["u_wind_850"] + vortex["u_wind_850"]
            gt_arrays["v_wind_850"][t_idx] = bg["v_wind_850"] + vortex["v_wind_850"]
            gt_arrays["relative_humidity"][t_idx] = np.clip(bg["relative_humidity"] + vortex["relative_humidity"], 10.0, 100.0)
            gt_arrays["geopotential_500"][t_idx] = bg["geopotential_500"] + vortex["geopotential_500"]
            
            orog_precip = self._compute_orographic_precip(gt_arrays["u_wind_850"][t_idx], gt_arrays["v_wind_850"][t_idx])
            gt_arrays["precipitation"][t_idx] = np.clip(vortex["precipitation"] + orog_precip, 0.0, 300.0)

        ens_arrays = {var: np.zeros((self.num_leads, self.num_members, self.num_lats, self.num_lons), dtype=np.float32)
                      for var in self.config.variables}

        for m in range(self.num_members):
            m_seed = seed + 100 * (m + 1)
            m_rng = np.random.RandomState(m_seed)
            
            track_angle_bias = m_rng.normal(0, 0.12)
            intensity_member_bias = m_rng.normal(intensity_bias_pct / 100.0, 0.08)

            for t_idx, lead_hours in enumerate(self.lead_times):
                error_scale = (lead_hours / 120.0) ** 1.1
                
                gt_pos = gt_metadata["trajectory"][t_idx]
                dlat = m_rng.normal(0, 0.15 * error_scale + 0.03) + (displacement_bias_km / 111.0) * error_scale * math.cos(track_angle_bias)
                dlon = m_rng.normal(0, 0.20 * error_scale + 0.03) + (displacement_bias_km / (111.0 * math.cos(math.radians(gt_pos["lat"])))) * error_scale * math.sin(track_angle_bias)
                
                member_lat = gt_pos["lat"] + dlat
                member_lon = gt_pos["lon"] + dlon
                member_intensity = max(2.0, gt_pos["intensity_deficit_hpa"] * (1.0 + intensity_member_bias + m_rng.normal(0, 0.05 * error_scale)))

                bg_m = self._compute_background_state(lead_hours)
                vortex_m, _ = self._compute_vortex_perturbation(member_lat, member_lon, member_intensity, lead_hours)

                synoptic_noise = m_rng.normal(0, 0.25 * (1 + error_scale), size=(self.num_lats, self.num_lons))
                
                ens_arrays["mslp"][t_idx, m] = bg_m["mslp"] + vortex_m["mslp"] + synoptic_noise * 0.3
                ens_arrays["temperature_2m"][t_idx, m] = bg_m["temperature_2m"] + vortex_m["temperature_2m"] - (self.elevation_map * 0.0065) + synoptic_noise * 0.2
                ens_arrays["u_wind_850"][t_idx, m] = bg_m["u_wind_850"] + vortex_m["u_wind_850"] + synoptic_noise * 0.4
                ens_arrays["v_wind_850"][t_idx, m] = bg_m["v_wind_850"] + vortex_m["v_wind_850"] + synoptic_noise * 0.4
                ens_arrays["relative_humidity"][t_idx, m] = np.clip(bg_m["relative_humidity"] + vortex_m["relative_humidity"] + synoptic_noise * 1.0, 5.0, 100.0)
                ens_arrays["geopotential_500"][t_idx, m] = bg_m["geopotential_500"] + vortex_m["geopotential_500"] + synoptic_noise * 3.0
                
                orog_p = self._compute_orographic_precip(ens_arrays["u_wind_850"][t_idx, m], ens_arrays["v_wind_850"][t_idx, m])
                precip_noise = np.maximum(0.0, m_rng.exponential(scale=0.8 * error_scale, size=(self.num_lats, self.num_lons)))
                ens_arrays["precipitation"][t_idx, m] = np.clip(vortex_m["precipitation"] * (1.0 + intensity_member_bias) + orog_p + precip_noise, 0.0, 350.0)

        ds = xr.Dataset(
            data_vars={
                var: (["lead_time", "ensemble_member", "latitude", "longitude"], ens_arrays[var])
                for var in self.config.variables
            },
            coords={
                "lead_time": self.lead_times,
                "ensemble_member": np.arange(self.num_members),
                "latitude": self.lats,
                "longitude": self.lons,
            },
            attrs={
                "run_id": run_id,
                "scenario_type": scenario_type,
                "seed": seed,
                "grid_resolution_deg": self.config.grid_res_deg,
                "domain": "India & South Asia",
                "creation_source": "SIH26078 Deterministic Synthetic Meteorological Engine"
            }
        )

        for var in self.config.variables:
            ds[f"gt_{var}"] = (["lead_time", "latitude", "longitude"], gt_arrays[var])
        ds["gt_event_mask"] = (["lead_time", "latitude", "longitude"], gt_masks)

        nc_filepath = DATASET_DIR / f"{run_id}.nc"
        meta_filepath = DATASET_DIR / f"{run_id}_metadata.json"
        
        ds.to_netcdf(nc_filepath)
        with open(meta_filepath, "w") as f:
            json.dump(gt_metadata, f, indent=2)

        return ds, gt_metadata

    def _initialize_ground_truth(self, scenario_type: str, rng: np.random.RandomState) -> Dict[str, Any]:
        if scenario_type == "monsoon_depression":
            start_lat = float(rng.uniform(17.5, 19.5))
            start_lon = float(rng.uniform(88.5, 91.0))
            peak_deficit = float(rng.uniform(14.0, 22.0))
            event_name = "Deep Monsoon Depression (Bay of Bengal)"
        elif scenario_type == "coastal_cyclone":
            start_lat = float(rng.uniform(12.0, 15.0))
            start_lon = float(rng.uniform(86.0, 89.0))
            peak_deficit = float(rng.uniform(22.0, 35.0))
            event_name = "Tropical Cyclonic Vortex"
        else:
            start_lat = float(rng.uniform(20.0, 24.0))
            start_lon = float(rng.uniform(77.0, 82.0))
            peak_deficit = float(rng.uniform(8.0, 14.0))
            event_name = "Severe Convective Cluster"

        return {
            "event_id": f"EVT_{rng.randint(1000, 9999)}",
            "event_name": event_name,
            "scenario_type": scenario_type,
            "start_lat": start_lat,
            "start_lon": start_lon,
            "peak_deficit_hpa": peak_deficit,
            "trajectory": []
        }

    def _get_gt_position_and_intensity(self, gt_metadata: Dict[str, Any], lead_hours: int) -> Tuple[float, float, float]:
        t_frac = lead_hours / 120.0
        start_lat = gt_metadata["start_lat"]
        start_lon = gt_metadata["start_lon"]
        
        lat = start_lat + 4.8 * t_frac + 0.5 * math.sin(t_frac * math.pi)
        lon = start_lon - 13.5 * t_frac - 1.2 * (t_frac ** 2)
        
        peak_deficit = gt_metadata["peak_deficit_hpa"]
        if lead_hours <= 48:
            intensity = 6.0 + (peak_deficit - 6.0) * (lead_hours / 48.0)
        elif lead_hours <= 72:
            intensity = peak_deficit
        else:
            decay_frac = (lead_hours - 72.0) / 48.0
            intensity = max(4.0, peak_deficit * (1.0 - 0.65 * decay_frac))
            
        return lat, lon, intensity

    def _compute_background_state(self, lead_hours: int) -> Dict[str, np.ndarray]:
        mslp = 1012.0 - 12.0 * ((self.lat_grid - 6.0) / 32.0) + 2.0 * np.sin(self.lon_grid * 0.1)
        t2m = 301.0 + 5.0 * np.sin((self.lat_grid - 8.0) / 20.0 * math.pi) - 2.0 * np.cos(self.lon_grid * 0.1)
        u_wind = 8.0 + 7.0 * np.sin((self.lat_grid - 5.0) / 15.0) * np.cos((self.lon_grid - 70.0) / 20.0)
        v_wind = 4.0 + 5.0 * np.cos((self.lat_grid - 10.0) / 12.0)
        rh = 75.0 + 10.0 * np.sin((self.lat_grid - 5.0) / 20.0)
        z500 = 5860.0 - 120.0 * ((self.lat_grid - 6.0) / 32.0)

        return {
            "mslp": mslp.astype(np.float32),
            "temperature_2m": t2m.astype(np.float32),
            "u_wind_850": u_wind.astype(np.float32),
            "v_wind_850": v_wind.astype(np.float32),
            "relative_humidity": rh.astype(np.float32),
            "geopotential_500": z500.astype(np.float32),
        }

    def _compute_vortex_perturbation(
        self, center_lat: float, center_lon: float, intensity_deficit: float, lead_hours: int
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        d_lat_km = (self.lat_grid - center_lat) * 111.0
        d_lon_km = (self.lon_grid - center_lon) * (111.0 * np.cos(np.radians(center_lat)))
        r_km = np.sqrt(d_lat_km**2 + d_lon_km**2) + 1e-4
        azimuth = np.arctan2(d_lat_km, d_lon_km)

        r_max = 85.0
        v_max = intensity_deficit * 2.2
        v_tangential = np.where(
            r_km <= r_max,
            v_max * (r_km / r_max),
            v_max * ((r_max / r_km) ** 0.65)
        )
        
        inflow_angle = np.radians(20.0)
        u_vortex = -v_tangential * np.sin(azimuth + inflow_angle)
        v_vortex = v_tangential * np.cos(azimuth + inflow_angle)

        mslp_vortex = -intensity_deficit * np.exp(- (r_km / 160.0) ** 1.4)

        spiral_phase = azimuth - 1.8 * np.log(np.maximum(1.0, r_km / 35.0))
        band_modulation = 0.5 + 0.5 * np.cos(2.0 * spiral_phase)
        
        core_precip = (intensity_deficit * 5.5) * np.exp(- (r_km / 90.0) ** 2)
        band_precip = (intensity_deficit * 3.0) * np.exp(- (r_km / 220.0) ** 1.2) * band_modulation
        precip_vortex = np.maximum(0.0, core_precip + band_precip)

        event_mask = np.where((r_km <= 200.0) & (precip_vortex >= 20.0), 1.0, 0.0).astype(np.float32)

        t2m_vortex = - 2.5 * np.exp(- (r_km / 120.0) ** 2)
        rh_vortex = 22.0 * np.exp(- (r_km / 180.0) ** 1.5)
        z500_vortex = - (intensity_deficit * 6.0) * np.exp(- (r_km / 220.0) ** 1.3)

        return {
            "mslp": mslp_vortex.astype(np.float32),
            "temperature_2m": t2m_vortex.astype(np.float32),
            "u_wind_850": u_vortex.astype(np.float32),
            "v_wind_850": v_vortex.astype(np.float32),
            "relative_humidity": rh_vortex.astype(np.float32),
            "geopotential_500": z500_vortex.astype(np.float32),
            "precipitation": precip_vortex.astype(np.float32)
        }, event_mask

    def _compute_orographic_precip(self, u_wind: np.ndarray, v_wind: np.ndarray) -> np.ndarray:
        grad_y, grad_x = np.gradient(self.elevation_map, self.config.grid_res_deg * 111000.0)
        w_proxy = np.maximum(0.0, u_wind * grad_x + v_wind * grad_y)
        orog_rain = np.clip(w_proxy * 80.0, 0.0, 30.0)
        return orog_rain.astype(np.float32)
