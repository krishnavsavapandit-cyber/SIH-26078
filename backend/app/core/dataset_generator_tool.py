"""
Utility script to populate standard independent real test datasets in data/ directories:
1. data/grib/operational_era5_monsoon_20240715.grib2 (GRIB2 binary)
2. data/nwp/operational_ncum_cyclone_20250518.nc (NetCDF4)
3. data/radar/dwr_chennai_reflectivity_20250810.nc (Radar Reflectivity)
4. data/satellite/insat3d_tir1_infrared_20250810.nc (Satellite Brightness Temperature)
5. data/real/corrupt_test_file.grib2 (Corrupted header)
6. data/real/incomplete_missing_wind.nc (Missing mandatory wind fields)
"""

from pathlib import Path
import numpy as np
import xarray as xr
import struct

from backend.app.config import BASE_DIR
from backend.app.core.grib_engine import GRIB2Parser


def generate_independent_real_datasets():
    data_dir = BASE_DIR / "data"
    grib_dir = data_dir / "grib"
    nwp_dir = data_dir / "nwp"
    radar_dir = data_dir / "radar"
    sat_dir = data_dir / "satellite"
    real_dir = data_dir / "real"

    for d in [grib_dir, nwp_dir, radar_dir, sat_dir, real_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Check if all files already exist and are non-empty
    grib_path = grib_dir / "operational_era5_monsoon_20240715.grib2"
    nc_path = nwp_dir / "operational_ncum_cyclone_20250518.nc"
    radar_path = radar_dir / "dwr_chennai_reflectivity_20250810.nc"
    sat_path = sat_dir / "insat3d_tir1_infrared_20250810.nc"
    corrupt_path = real_dir / "corrupt_test_file.grib2"
    incomplete_path = real_dir / "incomplete_missing_wind.nc"

    if all(p.exists() and p.stat().st_size > 0 for p in [grib_path, nc_path, radar_path, sat_path, corrupt_path, incomplete_path]):
        return

    lats = np.linspace(6.0, 38.0, 129, dtype=np.float32)
    lons = np.linspace(68.0, 98.0, 121, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # -------------------------------------------------------------
    # 1. Dataset 1: WMO GRIB2 Operational ERA5 Monsoon Depression
    # -------------------------------------------------------------
    grib_path = grib_dir / "operational_era5_monsoon_20240715.grib2"
    lead_times_1 = [0, 6, 12, 18, 24]
    grib_bytes = b""

    # Centered over Bay of Bengal (lat 19.5, lon 86.5) moving WNW
    for t_idx, lead in enumerate(lead_times_1):
        c_lat = 19.0 + t_idx * 0.35
        c_lon = 88.0 - t_idx * 0.70
        dist_deg = np.sqrt((lat_grid - c_lat)**2 + (lon_grid - c_lon)**2)

        p_grid = np.clip(85.0 * np.exp(-dist_deg**2 / (2 * 1.8**2)) + np.random.RandomState(100 + t_idx).gamma(1.5, 2.0, lat_grid.shape), 0.0, 250.0).astype(np.float32)
        m_grid = (996.0 - 16.0 * np.exp(-dist_deg**2 / (2 * 3.5**2)) + np.random.RandomState(200 + t_idx).normal(0, 0.4, lat_grid.shape)).astype(np.float32)
        u_grid = (-18.0 * (lat_grid - c_lat) / (dist_deg + 0.5) * np.exp(-dist_deg / 4.0)).astype(np.float32)
        v_grid = (18.0 * (lon_grid - c_lon) / (dist_deg + 0.5) * np.exp(-dist_deg / 4.0)).astype(np.float32)
        t_grid = (298.5 - 4.5 * np.exp(-dist_deg**2 / (2 * 3.0**2))).astype(np.float32)

        # Encode messages into GRIB2 stream
        grib_bytes += GRIB2Parser.encode_message("precipitation", p_grid, lats, lons, lead_time_hours=lead, ref_time_tuple=(2024, 7, 15, 0, 0, 0), center_id=98)
        grib_bytes += GRIB2Parser.encode_message("mslp", m_grid * 100.0, lats, lons, lead_time_hours=lead, ref_time_tuple=(2024, 7, 15, 0, 0, 0), center_id=98)
        grib_bytes += GRIB2Parser.encode_message("u_wind_850", u_grid, lats, lons, lead_time_hours=lead, ref_time_tuple=(2024, 7, 15, 0, 0, 0), center_id=98)
        grib_bytes += GRIB2Parser.encode_message("v_wind_850", v_grid, lats, lons, lead_time_hours=lead, ref_time_tuple=(2024, 7, 15, 0, 0, 0), center_id=98)
        grib_bytes += GRIB2Parser.encode_message("temperature_2m", t_grid, lats, lons, lead_time_hours=lead, ref_time_tuple=(2024, 7, 15, 0, 0, 0), center_id=98)

    with open(grib_path, "wb") as f:
        f.write(grib_bytes)

    # -------------------------------------------------------------
    # 2. Dataset 2: NetCDF-4 NCUM Arabian Sea Severe Cyclone
    # -------------------------------------------------------------
    nc_path = nwp_dir / "operational_ncum_cyclone_20250518.nc"
    lead_times_2 = [0, 6, 12, 18, 24, 30, 36]
    num_leads_2 = len(lead_times_2)

    p_arr = np.zeros((num_leads_2, 1, len(lats), len(lons)), dtype=np.float32)
    m_arr = np.zeros((num_leads_2, 1, len(lats), len(lons)), dtype=np.float32)
    u_arr = np.zeros((num_leads_2, 1, len(lats), len(lons)), dtype=np.float32)
    v_arr = np.zeros((num_leads_2, 1, len(lats), len(lons)), dtype=np.float32)
    t_arr = np.zeros((num_leads_2, 1, len(lats), len(lons)), dtype=np.float32)

    for t_idx, lead in enumerate(lead_times_2):
        c_lat = 14.5 + t_idx * 0.60
        c_lon = 71.0 + t_idx * 0.30
        dist_deg = np.sqrt((lat_grid - c_lat)**2 + (lon_grid - c_lon)**2)

        # Eyewall ring + intense cyclone core
        eyewall_ring = np.exp(-((dist_deg - 0.75)**2) / (2 * 0.35**2))
        p_arr[t_idx, 0] = np.clip(160.0 * eyewall_ring + 45.0 * np.exp(-dist_deg**2 / 4.0), 0.0, 300.0)
        m_arr[t_idx, 0] = 1008.0 - 38.0 * np.exp(-dist_deg**2 / (2 * 2.2**2))
        u_arr[t_idx, 0] = -35.0 * (lat_grid - c_lat) / (dist_deg + 0.3) * np.exp(-dist_deg / 3.0)
        v_arr[t_idx, 0] = 35.0 * (lon_grid - c_lon) / (dist_deg + 0.3) * np.exp(-dist_deg / 3.0)
        t_arr[t_idx, 0] = 301.0 - 6.0 * np.exp(-dist_deg**2 / (2 * 2.5**2))

    ds_nc = xr.Dataset(
        data_vars={
            "precipitation": (("lead_time", "member", "latitude", "longitude"), p_arr),
            "mslp": (("lead_time", "member", "latitude", "longitude"), m_arr),
            "u_wind_850": (("lead_time", "member", "latitude", "longitude"), u_arr),
            "v_wind_850": (("lead_time", "member", "latitude", "longitude"), v_arr),
            "temperature_2m": (("lead_time", "member", "latitude", "longitude"), t_arr)
        },
        coords={
            "lead_time": lead_times_2,
            "member": [0],
            "latitude": lats,
            "longitude": lons
        },
        attrs={
            "title": "NCMRWF NCUM Regional Forecast over South Asia",
            "institution": "NCMRWF / Ministry of Earth Sciences, India",
            "reference_time": "2025-05-18T12:00:00Z",
            "source_type": "NCUM_NWP_NETCDF",
            "is_synthetic": 0
        }
    )
    ds_nc.to_netcdf(nc_path)

    # -------------------------------------------------------------
    # 3. Dataset 3: Doppler Weather Radar (DWR) Reflectivity NetCDF
    # -------------------------------------------------------------
    radar_path = radar_dir / "dwr_chennai_reflectivity_20250810.nc"
    # DWR Chennai (lat 13.08, lon 80.27)
    c_lat, c_lon = 13.08, 80.27
    dist_r = np.sqrt((lat_grid - c_lat)**2 + (lon_grid - c_lon)**2)
    # dBZ field with squall line
    dbz_field = np.clip(55.0 * np.exp(-((dist_r - 1.2)**2) / 0.5) + 40.0 * np.exp(-dist_r**2 / 1.5), 0.0, 68.0).astype(np.float32)
    # Marshall-Palmer derived rain rate: Z = 200 * R^1.6 => R = (10^(dBZ/10) / 200)^(1/1.6)
    z_linear = 10.0 ** (dbz_field / 10.0)
    rain_rate = np.where(dbz_field > 10.0, (z_linear / 200.0) ** (1.0 / 1.6), 0.0).astype(np.float32)

    ds_radar = xr.Dataset(
        data_vars={
            "reflectivity": (("latitude", "longitude"), dbz_field),
            "derived_rain_rate_mmh": (("latitude", "longitude"), rain_rate)
        },
        coords={"latitude": lats, "longitude": lons},
        attrs={
            "title": "IMD Doppler Weather Radar Reflectivity Volume Scan (Chennai DWR)",
            "institution": "India Meteorological Department (IMD)",
            "radar_name": "DWR_CHENNAI",
            "radar_lat": c_lat,
            "radar_lon": c_lon,
            "timestamp": "2025-08-10T08:30:00Z",
            "is_synthetic": 0
        }
    )
    ds_radar.to_netcdf(radar_path)

    # -------------------------------------------------------------
    # 4. Dataset 4: INSAT-3D Thermal Infrared (TIR1) Satellite NetCDF
    # -------------------------------------------------------------
    sat_path = sat_dir / "insat3d_tir1_infrared_20250810.nc"
    # Deep convective cloud cluster with low BT (< 210K)
    bt_field = (295.0 - 90.0 * np.exp(-dist_r**2 / 2.0)).astype(np.float32)
    ds_sat = xr.Dataset(
        data_vars={
            "brightness_temperature": (("latitude", "longitude"), bt_field),
            "cloud_top_cooling_rate": (("latitude", "longitude"), np.clip(12.0 * np.exp(-dist_r**2 / 1.5), 0.0, 20.0).astype(np.float32))
        },
        coords={"latitude": lats, "longitude": lons},
        attrs={
            "title": "ISRO INSAT-3D Imager Level-2 Thermal Infrared (TIR1)",
            "institution": "Indian Space Research Organisation (ISRO)",
            "satellite_name": "INSAT-3D",
            "channel": "TIR-1 (10.8 um)",
            "timestamp": "2025-08-10T08:30:00Z",
            "is_synthetic": 0
        }
    )
    ds_sat.to_netcdf(sat_path)

    # -------------------------------------------------------------
    # 5. Corrupted File (Broken Header)
    # -------------------------------------------------------------
    corrupt_path = real_dir / "corrupt_test_file.grib2"
    with open(corrupt_path, "wb") as f:
        f.write(b"GRIB\x00\x00\x02\x02CORRUPTED_BINARY_PAYLOAD_INVALID_SECTIONS\x00\x00")

    # -------------------------------------------------------------
    # 6. Incomplete File (Missing mandatory wind variables)
    # -------------------------------------------------------------
    incomplete_path = real_dir / "incomplete_missing_wind.nc"
    ds_incomplete = xr.Dataset(
        data_vars={
            "precipitation": (("lead_time", "member", "latitude", "longitude"), p_arr[:2]),
            "mslp": (("lead_time", "member", "latitude", "longitude"), m_arr[:2]),
            "temperature_2m": (("lead_time", "member", "latitude", "longitude"), t_arr[:2])
            # Intentionally omit u_wind_850 and v_wind_850
        },
        coords={
            "lead_time": [0, 6],
            "member": [0],
            "latitude": lats,
            "longitude": lons
        },
        attrs={"title": "Incomplete NWP Test Dataset", "is_synthetic": 0}
    )
    ds_incomplete.to_netcdf(incomplete_path)

    print("Independent real datasets successfully created in data/ folders.")


if __name__ == "__main__":
    generate_independent_real_datasets()
