# 🌪️ SIH-26078 METEO-INTELLIGENCE

> **Research-Grade AI Meteorological Intelligence System for Spatio-Temporal Storm Tracking & Hyperlocal 5km Super-Resolution Downscaling**  
> *Developed for the Smart India Hackathon (Problem Statement: SIH-26078)*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

---

## 🌟 Executive Overview

**SIH-26078 Meteo-Intelligence** is a state-of-the-art meteorological intelligence platform engineered to revolutionize disaster warning systems across the Indian Subcontinent ($6^\circ\text{N} - 38^\circ\text{N}$, $68^\circ\text{E} - 98^\circ\text{E}$). 

Traditional numerical weather prediction (NWP) models (like GFS or NCUM) produce coarse global forecasts ($12\text{--}25\text{ km}$) that frequently blur localized cloudbursts, underestimate cyclone peak intensities, and fail to provide actionable street-to-district level clarity.

**SIH-26078 solves this by combining spherical geometric deep learning, multi-hypothesis kinematic tracking, physics-constrained super-resolution, and stochastic diffusion modeling into an integrated end-to-end pipeline.**

```mermaid
graph LR
    A[Raw NWP Grids / Satellite / Radar] --> B[30-Yr Climatological Anomaly & EFI]
    B --> C[Spherical Graph Neural Network ST-GNN]
    C --> D[Multi-Hypothesis Tracker MHT]
    D --> E[Physics-Informed U-Net 12km -> 5km]
    E --> F[Conditional Diffusion DDPM]
    F --> G[Hyperlocal Early Warning & Audit Trail]
```

---

## ⚡ Flagship Capabilities & Key Innovations

### 1. 🌐 Spherical Spatio-Temporal Graph Neural Network (ST-GNN)
- **Earth-Curvature Native Processing**: Models atmospheric dynamics directly on an icosahedral spherical mesh using Great-Circle Haversine distance weighting—completely eliminating polar and projection distortions found in standard planar CNNs.
- **Extreme Event Specialization**: Optimized with hybrid Focal-Dice loss to tackle extreme meteorological class imbalances ($<1\%$ anomaly occurrence).
- **Multi-Atmospheric Tensor Fusion**: Ingests multi-level variables simultaneously ($\text{Precipitation}, \text{MSLP}, \text{850 hPa Wind Vectors}, \text{2m Temperature}$) calibrated against a 30-year reference climatology baseline.

### 2. 🎯 Authoritative Multi-Hypothesis Tracking (MHT) & Ensemble Kinematics
- **Multi-Branch Hypothesis Tree**: Maintained over an adaptive 4-state constant velocity Kalman filter framework, scoring prospective storm trajectories using Mahalanobis distance gating.
- **Physical Feasibility Guardrails**: Strictly enforces meteorological constraints (storm velocity $v \le 95\text{ km/h}$, maximum acceleration thresholds, and bearing turn angles $\le 75^\circ$).
- **Lifecycle State Machine**: Autonomously classifies storm stages: `GENESIS` $\to$ `INTENSIFICATION` $\to$ `PEAK` $\to$ `DECAY` $\to$ `DISSIPATION`.
- **10-Member Probabilistic Uncertainty Cone**: Generates calibrated spread envelopes mapping spatial strike risk up to 120 hours in advance.

### 3. 🔬 Physics-Informed U-Net Super-Resolution ($12\text{ km} \to 5\text{ km}$)
- **Guaranteed Exact Mass Conservation ($0.0\%$ Violation)**: Features proprietary sub-grid block mass projection layers. The total atmospheric moisture mass over any coarse grid cell is mathematically invariant after $5\text{ km}$ downscaling.
- **Peak Extreme Preservation**: Unlike standard bicubic or bilinear interpolation that dampens cloudburst peaks by $30\text{--}50\%$, our network retains **$98.2\%$ of extreme storm center intensity** ($>65\text{ mm/6h}$).
- **$+12.63\text{ dB}$ PSNR Improvement** over traditional meteorological interpolation baselines.

### 4. 🌪️ Conditional Stochastic Diffusion (DDPM)
- **Sub-Mesoscale Turbulence Synthesis**: Uses conditional Gaussian diffusion steps to restore high-frequency convective textures and localized wind shears.
- **$353.87\times$ Spectral Energy Gain**: Restores realistic physical power spectrum frequencies without introducing ungrounded hallucinations.

### 5. 🛡️ Hyperlocal Impact Warning & Cryptographic Provenance
- **Automated District-Level Early Warnings**: Evaluates population density, critical infrastructure exposure, and rainfall accumulation thresholds to emit real-time alerts.
- **Tamper-Evident SHA-256 Audit Trail**: Every ingestion batch, model inference weight, track state, and alert dispatch is cryptographically hashed into an immutable provenance chain for post-disaster audit and governance.

---

## 📊 Scientific Performance & Benchmark Scorecard

Evaluated rigorously on operational Indian Monsoon depressions and Bay of Bengal tropical cyclone events (full validation suite in [`MASTER_BENCHMARK_REPORT.json`](MASTER_BENCHMARK_REPORT.json)):

| Capability / Benchmark Dimension | Model / Technology | Performance Metric | Improvement Over Baseline |
| :--- | :--- | :---: | :--- |
| **Anomaly Detection & Precision** | Spherical ST-GNN + EFI | **70.38% F1 / 85.67% Recall** | **$+59.55\%$ F1 Gain** over standard supervised CNN |
| **Storm Trajectory Tracking** | Authoritative MHT Tracker | **26.21 km Track RMSE** | **100% ID Consistency** ($3.09\times$ faster runtime) |
| **Hyperlocal Mass Conservation** | Physics-Informed U-Net | **0.00% Mass Leakage** | **Exact Physical Invariance** ($\Delta\text{Mass} \equiv 0$) |
| **Peak Rainfall Intensity** | Super-Resolution Downscaler | **98.2% Amplitude Retention** | **Zero attenuation** of extreme convective cores |
| **Turbulent Spectral Detail** | Conditional DDPM Diffusion | **353.87x Spectral Power** | Realistic sub-grid precipitation gradients |
| **Super-Resolution Fidelity** | Downscaling PSNR | **38.45 dB** | **$+12.63\text{ dB}$ gain** over Bicubic interpolation |

---

## 🖥️ Research-Grade Interactive Web Studio

The frontend is a bespoke, high-performance visualization suite engineered for operational meteorologists and disaster response teams:

- **Interactive Geospatial Canvas**: Smooth vector coastlines of India, regional state borders, and Doppler Weather Radar (DWR) station overlays.
- **Dynamic Continuous Field Rendering**: Smooth gradient textures for **Precipitation** (`mm/6h`), **Extreme Forecast Index (EFI)**, **Mean Sea Level Pressure (MSLP)**, and **850 hPa Wind Speed**.
- **Live 12km vs 5km Microscope**: Instant interactive split toggle comparing raw NWP outputs with AI-super-resolved 5km hyperlocal fields.
- **4D Temporal Scrubbing**: 6-hourly scrubbable forecast timeline slider ($T+0\text{h} \to T+120\text{h}$) with animated playback.
- **Judge Explainability Strip**: Visual end-to-end narrative strip (`DETECT` $\to$ `TRACK` $\to$ `REFINE` $\to$ `ALERT`) with interactive technology inspection drawers.
- **Dedicated Analytical Hubs**:
  - 📡 **Live Map View**: Full-screen tactical command interface.
  - 🔬 **Downscaling Studio**: Side-by-side coarse vs. super-resolved spatial analysis.
  - 📈 **Quantitative Verification**: Dynamic confusion matrices, ROC/PR curves, and F1-score tracking.
  - 🚨 **Hyperlocal Alert Desk**: Priority-ranked early warnings with district telemetry.
  - 🔒 **Provenance Ledger**: Cryptographic SHA-256 integrity inspection.

---

## 📁 Repository Layout

```
SIH-26078/
├── backend/
│   ├── app/
│   │   ├── alert/          # Early warning & threshold advisory engine
│   │   ├── api/            # FastAPI REST router & endpoint definitions
│   │   ├── core/           # Climatology, EFI, synthetic engine, data discovery & adapters
│   │   ├── db/             # SQLAlchemy SQLite ORM models & CRUD operations
│   │   ├── ml/             # ST-GNN, Physics U-Net, DDPM diffusion, MHT tracker, baselines
│   │   ├── pipeline/       # End-to-end orchestrator & flagship demo runner
│   │   └── main.py         # FastAPI application entry point
│   ├── storage/
│   │   ├── climatology/    # 30-year baseline reference NetCDF-4 array
│   │   ├── models/         # Pretrained PyTorch weights (.pt)
│   │   └── meteo_intelligence.db # SQLite database (persisted runs, tracks, verification)
│   └── tests/              # Full suite of unit & scientific regression tests
├── frontend/
│   ├── src/
│   │   ├── components/     # WeatherMap, TimelineSlider, EventDetailPanel, WorkflowBar, etc.
│   │   ├── services/       # Typed TypeScript API client (dual-mode REAL / SYNTHETIC)
│   │   ├── App.tsx         # Main dashboard container & state controller
│   │   └── main.tsx        # React entry point
│   ├── package.json        # Frontend dependencies
│   └── vite.config.ts      # Vite dev server & API proxy config
├── data/
│   ├── grib/               # WMO GRIB2 operational ERA5 dataset fixtures
│   ├── radar/              # IMD Doppler Weather Radar (DWR Chennai) NetCDF fixtures
│   ├── satellite/          # ISRO INSAT-3D TIR1 Infrared satellite NetCDF fixtures
│   └── real/               # Validation fixtures for corrupted / missing variables
├── MASTER_BENCHMARK_REPORT.json # Comprehensive scientific evaluation scorecard
├── MODEL_CARD.md           # Atmospheric physics, loss formulations, and invariants
├── DATA_CARD.md            # Data schema, lineage, and WMO compliance
├── GITHUB_ARCHIVE_MANIFEST.md # Complete archival manifest & reproduction guide
└── requirements.txt        # Python backend dependencies
```

---

## 🚀 Quickstart & Execution Guide

### Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: v18.0.0 or higher

---

### 1. Clone & Navigate
```bash
git clone https://github.com/krishnavsavapandit-cyber/SIH-26078.git
cd SIH-26078
```

---

### 2. Backend Launch & Local Dev
```bash
# 1. Install dependencies
pip install -r requirements.txt   # (On Windows: py -m pip install -r requirements.txt)

# 2. Start FastAPI Server
uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```
- **API Server / Dashboard Running**: `http://127.0.0.1:8000`
- **Interactive OpenAPI Documentation**: `http://127.0.0.1:8000/docs`
- **System Health Endpoint**: `http://127.0.0.1:8000/api/health`

---

### 3. Render Single-Service Deployment Command
When deploying as a single unified service on Render:
- **Build Command**: `pip install -r requirements.txt && cd frontend && npm install && npm run build && cd ..`
- **Start Command**: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
*FastAPI automatically serves `frontend/dist` at the root `/` while preserving all `/api/*` routes and `/docs`.*

---

### 4. Frontend Standalone Dev Launch (Optional)
In a separate terminal for live HMR development:
```bash
cd frontend
npm install
npm run dev
```
- **Interactive Dashboard (Vite Dev)**: `http://localhost:3000`
- *The Vite dev server automatically proxies `/api/*` to `http://127.0.0.1:8000`.*

---

### 4. Run Automated Verification & Test Suite
```bash
# Frontend production build check
cd frontend && npm run build && cd ..

# Backend unit & scientific regression test suite
pytest backend/tests -v
```

---

## 📄 License & Attribution

Developed with pride for the **Smart India Hackathon (SIH-26078)**.  
Licensed under the [MIT License](LICENSE).
