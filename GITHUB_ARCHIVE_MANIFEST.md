# SIH-26078 METEO-INTELLIGENCE — GITHUB ARCHIVE MANIFEST

This manifest provides complete documentation of the archived SIH-26078 project repository, including included datasets, model checkpoints, runtime configuration, excluded temporary artifacts, and full instructions for cloning, restoring, and running the system from scratch.

---

## 1. Project Overview & Architecture
- **Project**: SIH-26078 Meteo-Intelligence
- **Scope**: AI-Driven Spatio-Temporal Tracking & Hyperlocal 5km Super-Resolution Downscaling of Extreme Weather Anomalies
- **Stack**:
  - **Backend**: Python (FastAPI, PyTorch, Xarray, NetCDF4, NumPy, SciPy, SQLAlchemy, Pydantic)
  - **Frontend**: React 18, TypeScript, Vite, TailwindCSS, Lucide-React
  - **Storage**: SQLite (`meteo_intelligence.db`), PyTorch checkpoints (`.pt`), NetCDF-4 (`.nc`), WMO GRIB2 (`.grib2`)

---

## 2. Included Datasets & Scientific Artifacts

All core operational datasets, reference climatologies, and model checkpoints are bundled in the repository:

### A. Real NWP & Observational Datasets (`data/`)
1. `data/grib/operational_era5_monsoon_20240715.grib2` (785 KB): WMO GRIB2 multi-lead operational monsoon depression NWP forecast.
2. `data/radar/dwr_chennai_reflectivity_20250810.nc` (133 KB): IMD Doppler Weather Radar (DWR Chennai) reflectivity volume scan.
3. `data/satellite/insat3d_tir1_infrared_20250810.nc` (134 KB): ISRO INSAT-3D Thermal Infrared (TIR1) brightness temperature field.
4. `data/real/incomplete_missing_wind.nc` (385 KB): Real validation test fixture for graceful missing-variable compliance.
5. `data/real/corrupt_test_file.grib2` (51 B): Real validation test fixture for corrupted header error trapping.

### B. Reference Climatology & Persistence (`backend/storage/`)
1. `backend/storage/climatology/monsoon_30yr_climatology.nc` (1.02 MB): 30-year daily baseline climatology (mean & standard deviation) for multi-variable EFI & Shift-of-Tails anomaly scoring.
2. `backend/storage/meteo_intelligence.db` (2.08 MB): SQLite database containing all persisted benchmark runs, 4D event trajectories, quantitative verification scores, alerts, and SHA-256 provenance chains.

### C. Pretrained ML Checkpoints (`backend/storage/models/`)
1. `physics_unet_downscaler.pt` & `rollback_physics_unet.pt` (1.05 MB): Conservation-constrained Physics U-Net 5x super-resolution weights.
2. `cnn_baseline.pt` & `rollback_cnn_baseline.pt` (762 KB): Standard convolutional downscaler baseline weights.
3. `diffusion_denoiser.pt` & `rollback_ddpm.pt` (198 KB): Conditional Denoising Diffusion Probabilistic Model (DDPM) stochastic realization weights.
4. `downscaler_cnn.pt` (207 KB): Standard CNN benchmark comparison weights.
5. `st_gnn.pt` & `rollback_st_gnn.pt` (83 KB): Spherical icosahedral mesh Spatio-Temporal Graph Neural Network anomaly segmentation weights.

### D. Scientific Audit & Benchmark Certifications
1. `MASTER_BENCHMARK_REPORT.json`: Comprehensive cross-model evaluation scorecard (ST-GNN vs Persistence/Thresholding, Physics U-Net vs Bicubic/CNN, DDPM ensembles).
2. `MODEL_CARD.md`: Standard scientific model card detailing atmospheric boundary conditions, invariants, and causal validation.
3. `DATA_CARD.md`: Lineage, WMO/ERA5/NCUM schema compliance, and data provenance.
4. `FINAL_SIH26078_COMPLETION_REPORT.md` & `FINAL_YELLOW_TO_GREEN_CERTIFICATION.md`: System-wide verification metrics.

---

## 3. Files Intentionally Excluded & Why

The following transient items are excluded via `.gitignore`:
1. **`node_modules/` & `.npm/`**: Excluded to avoid committing tens of thousands of duplicate dependency files. Recreated via `npm install`.
2. **Python `__pycache__/`, `*.pyc`, `.pytest_cache/`**: Transient bytecode and local test execution caches. Recreated automatically on execution.
3. **`backend/storage/datasets/*.nc`**: Transient ~97MB individual forecast simulation files generated on-the-fly during full-suite local training runs. The pipeline dynamically generates or streams them as needed.
4. **Log files (`*.log`)**: Local runtime terminal logs.

---

## 4. System Requirements & Reproduction Instructions

### Prerequisites
- **Python**: 3.10 to 3.14 (Windows `py` launcher, or `python` on Linux/macOS)
- **Node.js**: v18.0.0 or higher (v24.19.0 recommended)
- **npm**: v9.0.0 or higher

---

### Step 1: Clone the Repository
```bash
git clone <GITHUB_REPOSITORY_URL>
cd SIH-26078
```

---

### Step 2: Install Python Dependencies
```bash
# Windows
py -m pip install -r requirements.txt

# Linux / macOS
python3 -m pip install -r requirements.txt
```

---

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

### Step 4: Start the Backend Service
From the repository root:
```bash
# Windows
py -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Linux / macOS
python3 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- API Health Endpoint: `http://127.0.0.1:8000/api/health`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

---

### Step 5: Start the Frontend Application
In a separate terminal:
```bash
cd frontend
npm run dev
```
- Open `http://localhost:3000` in any modern web browser.
- The UI will automatically proxy `/api` requests to `http://127.0.0.1:8000`.

---

### Step 6: Verify Backend Tests & Frontend Production Build
```bash
# Build frontend
cd frontend
npm run build
cd ..

# Run backend test suite
py -m pytest backend/tests -v
```

---

## 5. Summary of Archived Assets

| Category | Item Count | Total Size | Acceptance on GitHub |
| :--- | :--- | :--- | :--- |
| **Backend & Frontend Source Code** | 68 files | ~1.8 MB | Standard Git (Supported) |
| **Pretrained Model Checkpoints (`.pt`)** | 9 files | ~4.2 MB | Standard Git (Supported) |
| **SQLite State & Runs DB (`.db`)** | 1 file | 2.08 MB | Standard Git (Supported) |
| **NWP Datasets & Climatology (`.nc`, `.grib2`)** | 6 files | ~2.5 MB | Standard Git (Supported) |
| **Scientific Reports & Documentation** | 16 files | ~0.3 MB | Standard Git (Supported) |
| **Total Archival Footprint** | **114 files** | **~9.7 MB** | **100% within standard GitHub 100MB limit** |
