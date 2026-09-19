# SIH-26078 Data Card

## 1. Overview of Data Assets

The SIH-26078 Weather Intelligence Platform ingests and processes multi-modal operational atmospheric data across standard numerical weather prediction (NWP), Doppler Weather Radar (DWR), Geostationary Satellite imagery, and controlled thermodynamic synthetic benchmarks.

---

## 2. Dataset Specifications

### 2.1 ERA5 Reanalysis / NWP Forecast (WMO GRIB2 Binary)
- **File Artifact**: `data/grib/operational_era5_monsoon_20240715.grib2`
- **SHA-256 Hash**: `c47aaa00aa7e657002915e575a48e2836c6da9bdbc0aeedc11d1f9aaadf92c0c`
- **Format**: WMO GRIB Edition 2 Binary Stream (Sections 0–8, Simple Packing, Discipline 0 Meteorological Products).
- **Domain**: South Asia ($6.0^\circ\text{N} - 38.0^\circ\text{N}$, $68.0^\circ\text{E} - 98.0^\circ\text{E}$), Grid: $129 \times 121$ nodes ($0.25^\circ \approx 28\text{ km}$).
- **Mandatory Variables**:
  1. `precipitation`: Total surface accumulation (kg/m² or mm)
  2. `mslp`: Mean Sea Level Pressure (Pa)
  3. `u_wind_850`: 850 hPa Zonal wind vector (m/s)
  4. `v_wind_850`: 850 hPa Meridional wind vector (m/s)
  5. `temperature_2m`: 2-meter air temperature (K)
- **Forecast Horizon**: $T+0\text{h}$ to $T+24\text{h}$ (6-hour intervals).

### 2.2 NCUM Regional Forecast (NetCDF-4 / HDF5)
- **File Artifact**: `data/nwp/operational_ncum_cyclone_20250518.nc`
- **SHA-256 Hash**: `d3b6a85ef929b3e774e9152b8f95e5aa09e31d47e77a2dfdd25416528e82f05b`
- **Format**: NetCDF-4 / HDF5 Hierarchical Data Format.
- **Provider**: NCMRWF / Ministry of Earth Sciences, India.
- **Dimensions**: `lead_time: 7` ($T+0\text{h}$ to $T+36\text{h}$), `member: 1`, `latitude: 129`, `longitude: 121`.
- **Meteorological Phenomenon**: Severe Arabian Sea Tropical Cyclone Eyewall Dynamics.

### 2.3 Doppler Weather Radar Reflectivity (IMD DWR NetCDF)
- **File Artifact**: `data/radar/dwr_chennai_reflectivity_20250810.nc`
- **SHA-256 Hash**: `f4599bb39267950b84ae65d5cc9bc5f5dfc0ff737317aa56ad8533aa158c4432`
- **Station Location**: Chennai DWR ($13.08^\circ\text{N}, 80.27^\circ\text{E}$).
- **Observed Variable**: Radar Reflectivity Factor ($Z$ in $\text{dBZ}$, range $0 - 68\text{ dBZ}$).
- **Physical Transformation**: Marshall-Palmer Z-R Equation ($Z = 200 R^{1.6} \implies R = (10^{\text{dBZ}/10} / 200)^{1/1.6}\text{ mm/h}$).

### 2.4 INSAT-3D Geostationary Satellite (Thermal Infrared NetCDF)
- **File Artifact**: `data/satellite/insat3d_tir1_infrared_20250810.nc`
- **SHA-256 Hash**: `54ac1b618caa36e5e2afd4f4f8ed2cb740a6f38a293ba8b69f0e1b7eb009a077`
- **Instrument**: INSAT-3D Imager Level-2 TIR-1 Channel ($10.8\ \mu\text{m}$).
- **Observed Variable**: Cloud-Top Brightness Temperature ($T_B$ in Kelvin, range $195\text{K} - 310\text{K}$).
- **Physical Transformation**: Arkin / GOES Precipitation Index ($R = \max(0, (235 - T_B) \times 0.25)\text{ mm/h}$).

### 2.5 Controlled Physics-Grounded Synthetic Weather Benchmark
- **Engine**: `backend/app/core/synthetic_engine.py` (`SyntheticWeatherEngine`)
- **Physics Equations**: Rankine Vortex Dynamics, Clapeyron Moisture Saturation, Geostrophic Wind-Pressure Balance, Orographic Lifting.
- **Controlled Scenarios**: `monsoon_depression`, `cyclone_vortex`, `heat_dome`, `cold_wave`, `moving_anomaly`.
- **Ground Truth**: Exact mathematical moving-event trajectories with known centroid velocities and precipitation fields.

---

## 3. Data Integrity & Validation Rules
1. **Zero Synthetic Substitution**: REAL mode files missing any mandatory NWP variable fail validation with `RealDataValidationError`.
2. **Cryptographic Provenance**: Every file parsed is fingerprinted with SHA-256 before tensor allocation.
3. **Format Sniffing**: Binary header checks (`GRIB` magic bytes, `\x89HDF` NetCDF signatures) prevent file extension spoofing.
