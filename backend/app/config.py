"""
Configuration settings for SIH-26078 Extreme Weather Tracking System.
Defines domain boundaries (Indian Subcontinent & Surrounding Oceans),
grid resolutions, ensemble dimensions, and file paths.
"""

from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "backend" / "storage"
DATASET_DIR = STORAGE_DIR / "datasets"
CLIMATOLOGY_DIR = STORAGE_DIR / "climatology"
MODEL_WEIGHTS_DIR = STORAGE_DIR / "models"
DB_PATH = STORAGE_DIR / "meteo_intelligence.db"

# Ensure runtime directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
CLIMATOLOGY_DIR.mkdir(parents=True, exist_ok=True)
MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

class DomainConfig(BaseModel):
    # Geographical domain: India & Bay of Bengal / Arabian Sea
    lat_min: float = 6.0
    lat_max: float = 38.0
    lon_min: float = 68.0
    lon_max: float = 98.0
    
    # Coarse resolution (approx 25km / 0.25 deg) for medium range
    grid_res_deg: float = 0.25
    
    # Fine resolution (approx 5km / 0.05 deg) for learned downscaling
    fine_grid_res_deg: float = 0.05
    
    # Forecast time dimensions: 0h to 120h at 6h intervals (21 steps)
    lead_times_hours: list[int] = [h for h in range(0, 126, 6)]
    
    # Default ensemble size
    num_ensemble_members: int = 10
    
    # Core meteorological variables
    variables: list[str] = [
        "precipitation",        # mm / 6h
        "temperature_2m",       # Kelvin (K)
        "u_wind_850",           # m/s
        "v_wind_850",           # m/s
        "mslp",                 # hPa
        "relative_humidity",    # % (0-100)
        "geopotential_500"      # m^2 / s^2 (or gpm)
    ]

domain_config = DomainConfig()
