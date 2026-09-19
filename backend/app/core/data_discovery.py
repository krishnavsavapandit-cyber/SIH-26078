"""
Real Data Discovery and Ingestion Registry for SIH-26078.
Inspects configured data directories (data/, data/real/, data/nwp/, data/grib/,
data/radar/, data/satellite/, backend/storage/real/), sniffs formats, extracts
cryptographic hashes and metadata, validates required meteorological variables,
and routes to the appropriate GRIB2, NetCDF, Radar, or Satellite adapters.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import xarray as xr

from backend.app.config import BASE_DIR, STORAGE_DIR, domain_config, DomainConfig
from backend.app.core.grib_engine import GRIB2Parser, GRIB2Message

# Define standardized discovery paths
DISCOVERY_PATHS = [
    BASE_DIR / "data",
    BASE_DIR / "data" / "real",
    BASE_DIR / "data" / "nwp",
    BASE_DIR / "data" / "grib",
    BASE_DIR / "data" / "radar",
    BASE_DIR / "data" / "satellite",
    STORAGE_DIR / "real",
    STORAGE_DIR / "datasets" / "real"
]

# Required operational variables for full REAL mode
MANDATORY_REAL_VARIABLES = ["precipitation", "mslp", "u_wind_850", "v_wind_850", "temperature_2m"]


class RealDataValidationError(Exception):
    """Raised when a real dataset fails format, integrity, or required variable validation."""
    pass


class RealDataDiscovery:
    """
    Automatic discovery, inspection, and format detection engine for real weather datasets.
    """

    def __init__(self, config: DomainConfig = domain_config):
        self.config = config
        self._ensure_discovery_dirs()

    def _ensure_discovery_dirs(self):
        """Ensures all standard discovery directories exist."""
        for p in DISCOVERY_PATHS:
            p.mkdir(parents=True, exist_ok=True)

    def scan_discovered_datasets(self) -> List[Dict[str, Any]]:
        """
        Scans all configured discovery directories and returns detailed metadata for all valid real files.
        """
        datasets = []
        seen_hashes = set()

        for dir_path in DISCOVERY_PATHS:
            if not dir_path.exists():
                continue

            for file_path in dir_path.iterdir():
                if file_path.is_file() and not file_path.name.endswith(".json"):
                    inspection = self.inspect_file(file_path)
                    if inspection["is_supported"]:
                        file_hash = inspection["sha256"]
                        if file_hash not in seen_hashes:
                            seen_hashes.add(file_hash)
                            datasets.append(inspection)

        return datasets

    discover_all_datasets = scan_discovered_datasets

    def get_preferred_real_dataset(self) -> Optional[Path]:
        """
        Returns the path of the most suitable discovered real dataset, or None if none available.
        """
        scanned = self.scan_discovered_datasets()
        valid_real = [d for d in scanned if d.get("is_valid_for_pipeline", False)]
        if valid_real:
            return Path(valid_real[0]["file_path"])
        return None

    def inspect_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Performs deep binary and header inspection on a candidate data file.
        Truthfully identifies format (GRIB2, NetCDF4, Radar, Satellite, or Unsupported).
        """
        if not file_path.exists():
            return {
                "file_path": str(file_path),
                "filename": file_path.name,
                "is_supported": False,
                "error": "File does not exist"
            }

        file_size = file_path.stat().st_size
        if file_size == 0:
            return {
                "file_path": str(file_path),
                "filename": file_path.name,
                "is_supported": False,
                "error": "File is 0 bytes (empty)"
            }

        # Compute SHA-256 hash
        sha256_hash = self.compute_sha256(file_path)

        # Check format
        # 1. GRIB / GRIB2
        if GRIB2Parser.is_grib_file(file_path) or file_path.suffix.lower() in [".grib", ".grib2", ".grb", ".grb2"]:
            return self._inspect_grib(file_path, sha256_hash, file_size)

        # 2. NetCDF-4 / HDF5
        if file_path.suffix.lower() in [".nc", ".nc4", ".cdf", ".h5", ".hdf"]:
            return self._inspect_netcdf_or_specialized(file_path, sha256_hash, file_size)

        # Unsupported format
        return {
            "file_path": str(file_path),
            "filename": file_path.name,
            "sha256": sha256_hash,
            "size_bytes": file_size,
            "detected_format": "UNSUPPORTED",
            "is_supported": False,
            "is_valid_for_pipeline": False,
            "validation_status": "FAILED",
            "failure_reason": f"Unsupported meteorological file format extension '{file_path.suffix}'"
        }

    def _inspect_grib(self, file_path: Path, sha256_hash: str, file_size: int) -> Dict[str, Any]:
        """Inspects and validates a real GRIB2 dataset."""
        try:
            messages = GRIB2Parser.parse_file(file_path)
            if not messages:
                return {
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "sha256": sha256_hash,
                    "size_bytes": file_size,
                    "detected_format": "GRIB",
                    "is_supported": False,
                    "is_valid_for_pipeline": False,
                    "validation_status": "FAILED",
                    "failure_reason": "Corrupted or non-standard GRIB file (no valid GRIB2 messages found)"
                }

            variables_found = sorted(list(set(m.param_name for m in messages)))
            lead_times = sorted(list(set(m.forecast_lead_hours for m in messages)))
            ref_time = messages[0].ref_time_iso
            center_id = messages[0].center_id

            # Validate required variables
            missing_vars = [v for v in MANDATORY_REAL_VARIABLES if v not in variables_found]
            is_valid = len(missing_vars) == 0

            return {
                "file_path": str(file_path),
                "filename": file_path.name,
                "sha256": sha256_hash,
                "size_bytes": file_size,
                "detected_format": "GRIB2",
                "source_type": "GRIB2",
                "source_label": f"WMO GRIB2 Real Operational NWP (Center ID: {center_id})",
                "is_supported": True,
                "is_synthetic": False,
                "is_valid_for_pipeline": is_valid,
                "validation_status": "VALIDATED" if is_valid else "VALIDATION_FAILED",
                "missing_required_variables": missing_vars,
                "failure_reason": f"Missing required real meteorological fields: {missing_vars}" if missing_vars else None,
                "variables": variables_found,
                "lead_times": lead_times,
                "reference_time": ref_time,
                "grid_dimensions": [messages[0].nj, messages[0].ni],
                "coordinates": {
                    "lat_min": messages[0].lat_first,
                    "lat_max": messages[0].lat_last,
                    "lon_min": messages[0].lon_first,
                    "lon_max": messages[0].lon_last
                }
            }
        except Exception as e:
            return {
                "file_path": str(file_path),
                "filename": file_path.name,
                "sha256": sha256_hash,
                "size_bytes": file_size,
                "detected_format": "GRIB2",
                "is_supported": False,
                "is_valid_for_pipeline": False,
                "validation_status": "FAILED",
                "failure_reason": f"Error parsing GRIB2 file: {str(e)}"
            }

    def _inspect_netcdf_or_specialized(self, file_path: Path, sha256_hash: str, file_size: int) -> Dict[str, Any]:
        """Inspects and validates a NetCDF4, Radar, or Satellite dataset."""
        try:
            with xr.open_dataset(file_path) as ds:
                vars_list = list(ds.data_vars.keys())

                # Check if this is a specialized Radar dataset
                is_radar = any(r_key in vars_list for r_key in ["reflectivity", "dBZ", "DBZH", "TH", "radar_reflectivity"])
                is_satellite = any(s_key in vars_list for s_key in ["brightness_temperature", "TIR1", "channel_13", "ir_108", "satellite_bt"])

                if is_radar:
                    # Radar dataset
                    return {
                        "file_path": str(file_path),
                        "filename": file_path.name,
                        "sha256": sha256_hash,
                        "size_bytes": file_size,
                        "detected_format": "RADAR_NETCDF",
                        "source_type": "RADAR_DOPPLER",
                        "source_label": "Operational Doppler Weather Radar (DWR Reflectivity)",
                        "is_supported": True,
                        "is_synthetic": False,
                        "is_valid_for_pipeline": True,
                        "validation_status": "VALIDATED",
                        "variables": vars_list,
                        "radar_features": ["reflectivity_dBZ", "derived_rain_rate_mmh"],
                        "domain_bounds": {
                            "lat_min": float(ds.latitude.min()) if "latitude" in ds else 6.0,
                            "lat_max": float(ds.latitude.max()) if "latitude" in ds else 38.0,
                            "lon_min": float(ds.longitude.min()) if "longitude" in ds else 68.0,
                            "lon_max": float(ds.longitude.max()) if "longitude" in ds else 98.0
                        }
                    }

                if is_satellite:
                    # Satellite dataset
                    return {
                        "file_path": str(file_path),
                        "filename": file_path.name,
                        "sha256": sha256_hash,
                        "size_bytes": file_size,
                        "detected_format": "SATELLITE_NETCDF",
                        "source_type": "GEO_SATELLITE",
                        "source_label": "Geostationary Meteorological Satellite (TIR1 Infrared Channel)",
                        "is_supported": True,
                        "is_synthetic": False,
                        "is_valid_for_pipeline": True,
                        "validation_status": "VALIDATED",
                        "variables": vars_list,
                        "satellite_features": ["brightness_temperature_K", "cloud_top_cooling_rate"],
                        "domain_bounds": {
                            "lat_min": float(ds.latitude.min()) if "latitude" in ds else 6.0,
                            "lat_max": float(ds.latitude.max()) if "latitude" in ds else 38.0,
                            "lon_min": float(ds.longitude.min()) if "longitude" in ds else 68.0,
                            "lon_max": float(ds.longitude.max()) if "longitude" in ds else 98.0
                        }
                    }

                # Standard NWP NetCDF
                missing_vars = [v for v in MANDATORY_REAL_VARIABLES if v not in vars_list]
                is_valid = len(missing_vars) == 0

                return {
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "sha256": sha256_hash,
                    "size_bytes": file_size,
                    "detected_format": "NETCDF4",
                    "source_type": "NETCDF4",
                    "source_label": "Operational NWP / Reanalysis (ERA5/NCUM NetCDF-4)",
                    "is_supported": True,
                    "is_synthetic": False,
                    "is_valid_for_pipeline": is_valid,
                    "validation_status": "VALIDATED" if is_valid else "VALIDATION_FAILED",
                    "missing_required_variables": missing_vars,
                    "failure_reason": f"Missing required real meteorological fields: {missing_vars}" if missing_vars else None,
                    "variables": vars_list,
                    "lead_times": ds.lead_time.values.tolist() if "lead_time" in ds else [0],
                    "domain_bounds": {
                        "lat_min": float(ds.latitude.min()) if "latitude" in ds else 6.0,
                        "lat_max": float(ds.latitude.max()) if "latitude" in ds else 38.0,
                        "lon_min": float(ds.longitude.min()) if "longitude" in ds else 68.0,
                        "lon_max": float(ds.longitude.max()) if "longitude" in ds else 98.0
                    }
                }
        except Exception as e:
            return {
                "file_path": str(file_path),
                "filename": file_path.name,
                "sha256": sha256_hash,
                "size_bytes": file_size,
                "detected_format": "NETCDF4",
                "is_supported": False,
                "is_valid_for_pipeline": False,
                "validation_status": "FAILED",
                "failure_reason": f"Error opening NetCDF file: {str(e)}"
            }

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Computes SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


data_discovery = RealDataDiscovery()
