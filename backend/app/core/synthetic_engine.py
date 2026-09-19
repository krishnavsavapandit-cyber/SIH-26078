"""
Synthetic Meteorological Data Engine for SIH-26078.
Generates physically consistent, multi-variable ensemble weather forecasts
along with hidden ground-truth telemetry for 5 extreme atmospheric event families:
1. Cyclone / Monsoon Depression
2. Heat Dome / Heatwave
3. Cold Wave
4. Extreme Precipitation / Convective Cluster
5. Multi-Event (Coexisting simultaneous systems with distinct trajectories)
Supports flexible lead-time horizons from 3 to 10 days (up to 240 hours).
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import xarray as xr

from backend.app.config import domain_config, DATASET_DIR, DomainConfig


class SyntheticWeatherEngine:
    """
    Generates deterministic, physically grounded atmospheric datasets
    with controlled forecast perturbations and exact hidden ground-truth tracks.
    """
    def __init__(self, config: DomainConfig = domain_config):
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
        intensity_bias_pct: float = -5.0,
        custom_lead_times: Optional[List[int]] = None
    ) -> Tuple[xr.Dataset, Dict[str, Any]]:
        rng = np.random.RandomState(seed)
        lead_times = custom_lead_times if custom_lead_times is not None else self.lead_times
        num_leads = len(lead_times)

        gt_metadata = self._initialize_ground_truth(scenario_type, rng, lead_times)

        gt_arrays = {var: np.zeros((num_leads, self.num_lats, self.num_lons), dtype=np.float32) 
                     for var in self.config.variables}
        gt_masks = np.zeros((num_leads, self.num_lats, self.num_lons), dtype=np.float32)

        max_lead_horizon = max(lead_times[-1], 120)

        for t_idx, lead_hours in enumerate(lead_times):
            bg = self._compute_background_state(lead_hours)
            
            if scenario_type == "multi_event":
                # Compute two simultaneous events
                ev1_lat, ev1_lon, ev1_int = self._get_gt_position_and_intensity_cyclone(gt_metadata["event_1"], lead_hours, max_lead_horizon)
                ev2_lat, ev2_lon, ev2_int = self._get_gt_position_and_intensity_heatdome(gt_metadata["event_2"], lead_hours, max_lead_horizon)

                pert1, mask1 = self._compute_vortex_perturbation(ev1_lat, ev1_lon, ev1_int)
                pert2, mask2 = self._compute_heatdome_perturbation(ev2_lat, ev2_lon, ev2_int)

                pert = {var: pert1[var] + pert2[var] for var in self.config.variables}
                mask = np.clip(mask1 + mask2, 0.0, 1.0)

                gt_metadata["event_1"]["trajectory"].append({
                    "lead_time": lead_hours, "lat": float(ev1_lat), "lon": float(ev1_lon),
                    "intensity_metric": float(ev1_int), "event_type": "cyclone"
                })
                gt_metadata["event_2"]["trajectory"].append({
                    "lead_time": lead_hours, "lat": float(ev2_lat), "lon": float(ev2_lon),
                    "intensity_metric": float(ev2_int), "event_type": "heat_dome"
                })
                gt_metadata["trajectory"].append({
                    "lead_time": lead_hours, "lat": float(ev1_lat), "lon": float(ev1_lon),
                    "primary_intensity": float(ev1_int), "secondary_intensity": float(ev2_int)
                })

            elif scenario_type == "heat_dome":
                c_lat, c_lon, c_int = self._get_gt_position_and_intensity_heatdome(gt_metadata, lead_hours, max_lead_horizon)
                pert, mask = self._compute_heatdome_perturbation(c_lat, c_lon, c_int)
                gt_metadata["trajectory"].append({
                    "lead_time": lead_hours, "lat": float(c_lat), "lon": float(c_lon),
                    "intensity_deficit_hpa": 0.0, "temp_anomaly_k": float(c_int),
                    "max_wind_ms": 12.0, "peak_precip_mm6h": 0.0
                })

            elif scenario_type == "cold_wave":
                c_lat, c_lon, c_int = self._get_gt_position_and_intensity_coldwave(gt_metadata, lead_hours, max_lead_horizon)
                pert, mask = self._compute_coldwave_perturbation(c_lat, c_lon, c_int)
                gt_metadata["trajectory"].append({
                    "lead_time": lead_hours, "lat": float(c_lat), "lon": float(c_lon),
                    "intensity_deficit_hpa": 0.0, "temp_anomaly_k": float(-c_int),
                    "max_wind_ms": 18.0, "peak_precip_mm6h": 0.0
                })

            elif scenario_type in ["extreme_precipitation", "convective_cluster"]:
                c_lat, c_lon, c_int = self._get_gt_position_and_intensity_precip(gt_metadata, lead_hours, max_lead_horizon)
                pert, mask = self._compute_extreme_precip_perturbation(c_lat, c_lon, c_int)
                gt_metadata["trajectory"].append({
                    "lead_time": lead_hours, "lat": float(c_lat), "lon": float(c_lon),
                    "intensity_deficit_hpa": 6.0, "max_wind_ms": 20.0,
                    "peak_precip_mm6h": float(c_int)
                })

            else:  # monsoon_depression, coastal_cyclone, cyclone
                c_lat, c_lon, c_int = self._get_gt_position_and_intensity_cyclone(gt_metadata, lead_hours, max_lead_horizon)
                pert, mask = self._compute_vortex_perturbation(c_lat, c_lon, c_int)
                gt_metadata["trajectory"].append({
                    "lead_time": lead_hours, "lat": float(c_lat), "lon": float(c_lon),
                    "intensity_deficit_hpa": float(c_int),
                    "max_wind_ms": float(c_int * 2.2),
                    "peak_precip_mm6h": float(min(240.0, 35.0 + c_int * 6.5))
                })

            gt_masks[t_idx] = mask
            gt_arrays["mslp"][t_idx] = bg["mslp"] + pert["mslp"]
            gt_arrays["temperature_2m"][t_idx] = bg["temperature_2m"] + pert["temperature_2m"] - (self.elevation_map * 0.0065)
            gt_arrays["u_wind_850"][t_idx] = bg["u_wind_850"] + pert["u_wind_850"]
            gt_arrays["v_wind_850"][t_idx] = bg["v_wind_850"] + pert["v_wind_850"]
            gt_arrays["relative_humidity"][t_idx] = np.clip(bg["relative_humidity"] + pert["relative_humidity"], 5.0, 100.0)
            gt_arrays["geopotential_500"][t_idx] = bg["geopotential_500"] + pert["geopotential_500"]
            
            orog_precip = self._compute_orographic_precip(gt_arrays["u_wind_850"][t_idx], gt_arrays["v_wind_850"][t_idx])
            gt_arrays["precipitation"][t_idx] = np.clip(pert["precipitation"] + orog_precip, 0.0, 450.0)

        # Generate Ensemble Members
        ens_arrays = {var: np.zeros((num_leads, self.num_members, self.num_lats, self.num_lons), dtype=np.float32)
                      for var in self.config.variables}

        for m in range(self.num_members):
            m_seed = seed + 100 * (m + 1)
            m_rng = np.random.RandomState(m_seed)
            track_angle_bias = m_rng.normal(0, 0.12)
            intensity_member_bias = m_rng.normal(intensity_bias_pct / 100.0, 0.08)

            for t_idx, lead_hours in enumerate(lead_times):
                error_scale = (lead_hours / 120.0) ** 1.1 + 0.1
                bg_m = self._compute_background_state(lead_hours)
                
                # Synthetic Member perturbation with error growth
                dlat = m_rng.normal(0, 0.15 * error_scale) + (displacement_bias_km / 111.0) * error_scale * math.cos(track_angle_bias)
                dlon = m_rng.normal(0, 0.20 * error_scale) + (displacement_bias_km / 111.0) * error_scale * math.sin(track_angle_bias)
                
                synoptic_noise = m_rng.normal(0, 0.25 * (1 + error_scale), size=(self.num_lats, self.num_lons)).astype(np.float32)
                
                # Perturb GT state for member
                ens_arrays["mslp"][t_idx, m] = gt_arrays["mslp"][t_idx] + synoptic_noise * 0.3
                ens_arrays["temperature_2m"][t_idx, m] = gt_arrays["temperature_2m"][t_idx] + synoptic_noise * 0.2
                ens_arrays["u_wind_850"][t_idx, m] = gt_arrays["u_wind_850"][t_idx] + synoptic_noise * 0.4
                ens_arrays["v_wind_850"][t_idx, m] = gt_arrays["v_wind_850"][t_idx] + synoptic_noise * 0.4
                ens_arrays["relative_humidity"][t_idx, m] = np.clip(gt_arrays["relative_humidity"][t_idx] + synoptic_noise * 1.0, 5.0, 100.0)
                ens_arrays["geopotential_500"][t_idx, m] = gt_arrays["geopotential_500"][t_idx] + synoptic_noise * 3.0
                
                precip_noise = np.maximum(0.0, m_rng.exponential(scale=0.8 * error_scale, size=(self.num_lats, self.num_lons))).astype(np.float32)
                ens_arrays["precipitation"][t_idx, m] = np.clip(gt_arrays["precipitation"][t_idx] * (1.0 + intensity_member_bias) + precip_noise, 0.0, 450.0)

        ds = xr.Dataset(
            data_vars={
                var: (["lead_time", "ensemble_member", "latitude", "longitude"], ens_arrays[var])
                for var in self.config.variables
            },
            coords={
                "lead_time": lead_times,
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
                "creation_source": "SIH26078 Deterministic Synthetic Meteorological Engine",
                "is_synthetic": 1,
                "provenance": "SYNTHETIC_CONTROLLED_BENCHMARK"
            }
        )

        for var in self.config.variables:
            ds[f"gt_{var}"] = (["lead_time", "latitude", "longitude"], gt_arrays[var])
        ds["gt_event_mask"] = (["lead_time", "latitude", "longitude"], gt_masks)

        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        nc_filepath = DATASET_DIR / f"{run_id}.nc"
        meta_filepath = DATASET_DIR / f"{run_id}_metadata.json"
        
        ds.to_netcdf(nc_filepath)
        with open(meta_filepath, "w") as f:
            json.dump(gt_metadata, f, indent=2)

        return ds, gt_metadata

    def _initialize_ground_truth(self, scenario_type: str, rng: np.random.RandomState, lead_times: List[int]) -> Dict[str, Any]:
        if scenario_type == "multi_event":
            ev1 = {
                "event_id": f"EVT_CYC_{rng.randint(1000, 9999)}",
                "event_name": "Bay of Bengal Cyclone",
                "scenario_type": "cyclone",
                "start_lat": float(rng.uniform(14.0, 16.5)),
                "start_lon": float(rng.uniform(87.0, 89.5)),
                "peak_deficit_hpa": float(rng.uniform(18.0, 26.0)),
                "trajectory": []
            }
            ev2 = {
                "event_id": f"EVT_HEAT_{rng.randint(1000, 9999)}",
                "event_name": "Northwest India Heat Dome",
                "scenario_type": "heat_dome",
                "start_lat": float(rng.uniform(25.0, 28.0)),
                "start_lon": float(rng.uniform(72.0, 75.5)),
                "peak_anomaly_k": float(rng.uniform(6.0, 9.5)),
                "trajectory": []
            }
            return {
                "event_id": f"EVT_MULTI_{rng.randint(1000, 9999)}",
                "event_name": "Dual Simultaneous Weather Extremes (Cyclone + Heat Dome)",
                "scenario_type": "multi_event",
                "event_1": ev1,
                "event_2": ev2,
                "trajectory": []
            }

        elif scenario_type == "heat_dome":
            return {
                "event_id": f"EVT_{rng.randint(1000, 9999)}",
                "event_name": "Severe Synoptic Heat Dome",
                "scenario_type": "heat_dome",
                "start_lat": float(rng.uniform(24.0, 27.0)),
                "start_lon": float(rng.uniform(73.0, 77.0)),
                "peak_anomaly_k": float(rng.uniform(6.5, 9.0)),
                "trajectory": []
            }

        elif scenario_type == "cold_wave":
            return {
                "event_id": f"EVT_{rng.randint(1000, 9999)}",
                "event_name": "Sub-Himalayan Severe Cold Wave",
                "scenario_type": "cold_wave",
                "start_lat": float(rng.uniform(28.0, 31.0)),
                "start_lon": float(rng.uniform(76.0, 80.0)),
                "peak_anomaly_k": float(rng.uniform(7.0, 11.0)),
                "trajectory": []
            }

        elif scenario_type in ["extreme_precipitation", "convective_cluster"]:
            return {
                "event_id": f"EVT_{rng.randint(1000, 9999)}",
                "event_name": "Mesoscale Convective Cloudburst Cluster",
                "scenario_type": "extreme_precipitation",
                "start_lat": float(rng.uniform(18.0, 22.0)),
                "start_lon": float(rng.uniform(75.0, 80.0)),
                "peak_precip_mm": float(rng.uniform(160.0, 240.0)),
                "trajectory": []
            }

        else:
            return {
                "event_id": f"EVT_{rng.randint(1000, 9999)}",
                "event_name": "Deep Monsoon Depression (Bay of Bengal)",
                "scenario_type": scenario_type,
                "start_lat": float(rng.uniform(17.5, 19.5)),
                "start_lon": float(rng.uniform(88.5, 91.0)),
                "peak_deficit_hpa": float(rng.uniform(16.0, 25.0)),
                "trajectory": []
            }

    def _get_gt_position_and_intensity_cyclone(self, meta: Dict[str, Any], lead_hours: int, max_lead: int) -> Tuple[float, float, float]:
        t_frac = lead_hours / max(max_lead, 120)
        lat = meta["start_lat"] + 5.5 * t_frac + 0.6 * math.sin(t_frac * math.pi)
        lon = meta["start_lon"] - 14.5 * t_frac - 1.5 * (t_frac ** 2)
        peak_deficit = meta["peak_deficit_hpa"]
        if t_frac <= 0.40:
            intensity = 6.0 + (peak_deficit - 6.0) * (t_frac / 0.40)
        elif t_frac <= 0.65:
            intensity = peak_deficit
        else:
            decay_frac = (t_frac - 0.65) / 0.35
            intensity = max(4.0, peak_deficit * (1.0 - 0.70 * decay_frac))
        return lat, lon, intensity

    def _get_gt_position_and_intensity_heatdome(self, meta: Dict[str, Any], lead_hours: int, max_lead: int) -> Tuple[float, float, float]:
        t_frac = lead_hours / max(max_lead, 120)
        lat = meta["start_lat"] + 1.8 * t_frac
        lon = meta["start_lon"] + 2.5 * t_frac
        peak_anom = meta.get("peak_anomaly_k", 7.5)
        # Heat dome builds slowly and persists
        if t_frac <= 0.30:
            intensity = 3.0 + (peak_anom - 3.0) * (t_frac / 0.30)
        elif t_frac <= 0.75:
            intensity = peak_anom
        else:
            intensity = max(2.5, peak_anom * (1.0 - 0.5 * (t_frac - 0.75) / 0.25))
        return lat, lon, intensity

    def _get_gt_position_and_intensity_coldwave(self, meta: Dict[str, Any], lead_hours: int, max_lead: int) -> Tuple[float, float, float]:
        t_frac = lead_hours / max(max_lead, 120)
        lat = meta["start_lat"] - 4.5 * t_frac
        lon = meta["start_lon"] + 1.2 * t_frac
        peak_anom = meta.get("peak_anomaly_k", 8.0)
        intensity = peak_anom * (0.5 + 0.5 * math.sin(min(1.0, t_frac * 1.5) * math.pi))
        return lat, lon, intensity

    def _get_gt_position_and_intensity_precip(self, meta: Dict[str, Any], lead_hours: int, max_lead: int) -> Tuple[float, float, float]:
        t_frac = lead_hours / max(max_lead, 120)
        lat = meta["start_lat"] + 2.5 * t_frac
        lon = meta["start_lon"] - 3.5 * t_frac
        peak_p = meta.get("peak_precip_mm", 180.0)
        intensity = peak_p * math.exp(-0.5 * ((t_frac - 0.45) / 0.25) ** 2)
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

    def _compute_vortex_perturbation(self, center_lat: float, center_lon: float, intensity_deficit: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        d_lat_km = (self.lat_grid - center_lat) * 111.0
        d_lon_km = (self.lon_grid - center_lon) * (111.0 * np.cos(np.radians(center_lat)))
        r_km = np.sqrt(d_lat_km**2 + d_lon_km**2) + 1e-4
        azimuth = np.arctan2(d_lat_km, d_lon_km)

        r_max = 85.0
        v_max = intensity_deficit * 2.2
        v_tangential = np.where(r_km <= r_max, v_max * (r_km / r_max), v_max * ((r_max / r_km) ** 0.65))
        
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

    def _compute_heatdome_perturbation(self, center_lat: float, center_lon: float, temp_anomaly: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        d_lat_km = (self.lat_grid - center_lat) * 111.0
        d_lon_km = (self.lon_grid - center_lon) * (111.0 * np.cos(np.radians(center_lat)))
        r_km = np.sqrt(d_lat_km**2 + d_lon_km**2) + 1e-4

        # Broad anticyclonic ridge
        t2m_pert = temp_anomaly * np.exp(- (r_km / 350.0) ** 2)
        mslp_pert = 6.0 * np.exp(- (r_km / 400.0) ** 2)  # High pressure
        rh_pert = - 30.0 * np.exp(- (r_km / 300.0) ** 2) # Dry subsidence
        z500_pert = 80.0 * np.exp(- (r_km / 400.0) ** 2) # 500 hPa ridge
        
        event_mask = np.where((r_km <= 350.0) & (t2m_pert >= 3.5), 1.0, 0.0).astype(np.float32)

        return {
            "mslp": mslp_pert.astype(np.float32),
            "temperature_2m": t2m_pert.astype(np.float32),
            "u_wind_850": np.zeros_like(t2m_pert),
            "v_wind_850": np.zeros_like(t2m_pert),
            "relative_humidity": rh_pert.astype(np.float32),
            "geopotential_500": z500_pert.astype(np.float32),
            "precipitation": np.zeros_like(t2m_pert)
        }, event_mask

    def _compute_coldwave_perturbation(self, center_lat: float, center_lon: float, cold_anomaly: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        d_lat_km = (self.lat_grid - center_lat) * 111.0
        d_lon_km = (self.lon_grid - center_lon) * (111.0 * np.cos(np.radians(center_lat)))
        r_km = np.sqrt(d_lat_km**2 + d_lon_km**2) + 1e-4

        t2m_pert = - cold_anomaly * np.exp(- (r_km / 320.0) ** 2)
        mslp_pert = 8.0 * np.exp(- (r_km / 350.0) ** 2)  # Continental anticyclone
        v_wind_pert = - 12.0 * np.exp(- (r_km / 280.0) ** 2) # Northerly cold surge
        
        event_mask = np.where((r_km <= 320.0) & (t2m_pert <= -4.0), 1.0, 0.0).astype(np.float32)

        return {
            "mslp": mslp_pert.astype(np.float32),
            "temperature_2m": t2m_pert.astype(np.float32),
            "u_wind_850": np.zeros_like(t2m_pert),
            "v_wind_850": v_wind_pert.astype(np.float32),
            "relative_humidity": - 15.0 * np.exp(- (r_km / 250.0) ** 2),
            "geopotential_500": - 40.0 * np.exp(- (r_km / 300.0) ** 2),
            "precipitation": np.zeros_like(t2m_pert)
        }, event_mask

    def _compute_extreme_precip_perturbation(self, center_lat: float, center_lon: float, peak_precip: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        d_lat_km = (self.lat_grid - center_lat) * 111.0
        d_lon_km = (self.lon_grid - center_lon) * (111.0 * np.cos(np.radians(center_lat)))
        r_km = np.sqrt(d_lat_km**2 + d_lon_km**2) + 1e-4

        precip_pert = peak_precip * np.exp(- (r_km / 75.0) ** 2)
        mslp_pert = - 7.0 * np.exp(- (r_km / 100.0) ** 2)
        rh_pert = 30.0 * np.exp(- (r_km / 120.0) ** 2)
        
        event_mask = np.where((r_km <= 100.0) & (precip_pert >= 30.0), 1.0, 0.0).astype(np.float32)

        return {
            "mslp": mslp_pert.astype(np.float32),
            "temperature_2m": - 3.0 * np.exp(- (r_km / 80.0) ** 2),
            "u_wind_850": - 15.0 * np.sin(d_lon_km / 50.0) * np.exp(- (r_km / 100.0) ** 2),
            "v_wind_850": 15.0 * np.cos(d_lat_km / 50.0) * np.exp(- (r_km / 100.0) ** 2),
            "relative_humidity": rh_pert.astype(np.float32),
            "geopotential_500": - 30.0 * np.exp(- (r_km / 120.0) ** 2),
            "precipitation": precip_pert.astype(np.float32)
        }, event_mask

    def _compute_orographic_precip(self, u_wind: np.ndarray, v_wind: np.ndarray) -> np.ndarray:
        grad_y, grad_x = np.gradient(self.elevation_map, self.config.grid_res_deg * 111000.0)
        w_proxy = np.maximum(0.0, u_wind * grad_x + v_wind * grad_y)
        orog_rain = np.clip(w_proxy * 80.0, 0.0, 30.0)
        return orog_rain.astype(np.float32)
