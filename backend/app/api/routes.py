"""
FastAPI REST API Routes for SIH-26078 Extreme Weather Intelligence System.
Exposes calculated meteorological fields, EFI maps, detections, trajectories,
uncertainty cones, verification metrics, alerts, downscaling comparisons,
physics audits, explainability diagnostics, execution controls,
real-data discovery, inspection, and dual-mode runtime status.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import xarray as xr
import json

from backend.app.config import DATASET_DIR, domain_config
from backend.app.db.database import get_db
from backend.app.db import crud
from backend.app.pipeline.orchestrator import orchestrator
from backend.app.core.physics_validator import physics_validator
from backend.app.core.data_adapter import weather_adapter
from backend.app.core.data_discovery import data_discovery, RealDataValidationError
from backend.app.ml.downscaler_baseline import downscaling_engine
from backend.app.ml.explainability import anomaly_explainer
from backend.app.ml.st_gnn import st_gnn_manager
from backend.app.ml.advanced_downscaler import advanced_downscaling_manager
from backend.app.ml.diffusion_experiment import diffusion_engine
from backend.app.ml.advanced_tracker import advanced_tracker
from backend.app.ml.benchmark_suite import benchmark_suite, BENCHMARK_RESULTS_PATH
from backend.app.ml.extreme_preservation import ExtremePreservationScorecard

preservation_scorecard = ExtremePreservationScorecard(scale_factor=5)
router = APIRouter(prefix="/api", tags=["Weather Intelligence API"])


class GenerateRequest(BaseModel):
    run_id: str = "run_demo_monsoon_depression"
    scenario_type: str = "monsoon_depression"
    seed: int = 42
    displacement_bias_km: float = 25.0
    intensity_bias_pct: float = -5.0
    data_source_mode: str = "auto"       # "auto", "REAL", "SYNTHETIC"
    external_data_path: Optional[str] = None
    allow_synthetic_fallback: bool = True


class FlagshipRequest(BaseModel):
    run_id: str = "run_flagship_monsoon"
    scenario_type: str = "monsoon_depression"
    seed: int = 42
    displacement_bias_km: float = 25.0
    intensity_bias_pct: float = -5.0
    enable_phase3: bool = True
    inject_failure: bool = False
    data_source_mode: str = "auto"       # "auto", "REAL", "SYNTHETIC"
    external_data_path: Optional[str] = None
    allow_synthetic_fallback: bool = True


class InspectRequest(BaseModel):
    file_path: str


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SIH-26078 Weather Anomaly Intelligence Backend",
        "version": "3.1.0-real-data-flagship",
        "domain": "India & South Asia"
    }


# ==========================================
# Data Mode, Discovery & Ingestion Endpoints
# ==========================================

@router.get("/data/mode-status")
def get_data_mode_status(db: Session = Depends(get_db)):
    """
    Authoritative runtime state for frontend indicator.
    Returns current active execution mode (REAL vs SYNTHETIC), preferred real file,
    and discovered real datasets with cryptographic hashes.
    """
    preferred_real = data_discovery.get_preferred_real_dataset()
    scanned = data_discovery.scan_discovered_datasets()
    latest_run = db.query(crud.ForecastRunRecord).order_by(crud.ForecastRunRecord.created_at.desc()).first()

    mode = "SYNTHETIC"
    active_meta = None
    if latest_run and latest_run.provenance:
        params = latest_run.provenance.parameters_json or {}
        mode = params.get("data_source_mode", "SYNTHETIC")
        active_meta = {
            "run_id": latest_run.run_id,
            "data_source_mode": mode,
            "source_type": params.get("source_type", "CONTROLLED_PHYSICS_BENCHMARK"),
            "source_name": params.get("source_name", "N/A"),
            "synthetic": params.get("synthetic", True),
            "dataset_sha256": latest_run.provenance.dataset_sha256,
            "input_artifact": params.get("input_artifact", "N/A"),
            "variables": params.get("variables_used", [])
        }

    return {
        "status": "ONLINE",
        "active_mode": mode,
        "is_real_data_available": preferred_real is not None,
        "preferred_real_dataset": str(preferred_real) if preferred_real else None,
        "discovered_datasets_count": len(scanned),
        "discovered_datasets": scanned,
        "active_run_metadata": active_meta,
        "supported_adapters": weather_adapter.get_supported_adapters()
    }


@router.get("/data/discover")
def discover_datasets():
    """Scans all configured discovery directories for supported real datasets."""
    return data_discovery.scan_discovered_datasets()


@router.post("/data/inspect")
def inspect_dataset(req: InspectRequest):
    """Performs deep binary header inspection and variable validation on a data file."""
    p = Path(req.file_path)
    return data_discovery.inspect_file(p)


@router.get("/datasets")
def list_datasets():
    datasets = []
    for f in DATASET_DIR.glob("*.nc"):
        datasets.append({
            "run_id": f.stem,
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "created_at": f.stat().st_mtime
        })
    return datasets


# ==========================================
# Pipeline Execution Endpoints
# ==========================================

@router.post("/dataset/generate")
def generate_dataset(req: GenerateRequest, db: Session = Depends(get_db)):
    result = orchestrator.run_full_pipeline(
        run_id=req.run_id,
        scenario_type=req.scenario_type,
        seed=req.seed,
        displacement_bias_km=req.displacement_bias_km,
        intensity_bias_pct=req.intensity_bias_pct,
        data_source_mode=req.data_source_mode,
        external_data_path=req.external_data_path,
        allow_synthetic_fallback=req.allow_synthetic_fallback,
        db=db
    )
    return result


@router.post("/pipeline/run")
def trigger_pipeline(req: GenerateRequest, db: Session = Depends(get_db)):
    return generate_dataset(req, db)


@router.post("/pipeline/flagship-run")
def trigger_flagship_pipeline(req: FlagshipRequest, db: Session = Depends(get_db)):
    """
    Executes the complete 14-stage scientific pipeline:
    Data Ingestion -> Climatology -> EFI -> Dynamic Footprint -> ST-GNN -> MHT -> Uncertainty ->
    12km->5km Downscaling -> Physics Audit -> Diffusion -> Extreme Scorecard -> 5km Warning -> Verification -> Provenance.
    """
    res = orchestrator.run_flagship_pipeline(
        run_id=req.run_id,
        scenario_type=req.scenario_type,
        seed=req.seed,
        displacement_bias_km=req.displacement_bias_km,
        intensity_bias_pct=req.intensity_bias_pct,
        enable_phase3=req.enable_phase3,
        inject_failure=req.inject_failure,
        data_source_mode=req.data_source_mode,
        external_data_path=req.external_data_path,
        allow_synthetic_fallback=req.allow_synthetic_fallback,
        db=db
    )
    return res


# ==========================================
# Runs, Events & Provenance Query Endpoints
# ==========================================

@router.get("/runs")
def list_runs(db: Session = Depends(get_db)):
    runs = crud.get_all_runs(db)
    result = []
    for r in runs:
        prov = r.provenance
        params = prov.parameters_json if prov else {}
        mode = params.get("data_source_mode", "SYNTHETIC")
        is_synth = params.get("synthetic", True)
        result.append({
            "run_id": r.run_id,
            "scenario_type": r.scenario_type,
            "status": r.status,
            "grid_resolution_deg": r.grid_resolution_deg,
            "num_members": r.num_members,
            "num_lead_steps": r.num_lead_steps,
            "created_at": r.created_at,
            "data_source_mode": mode,
            "synthetic": is_synth
        })
    return result


@router.get("/runs/{run_id}")
def get_run_details(run_id: str, db: Session = Depends(get_db)):
    run = crud.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    events = crud.get_events_for_run(db, run_id)
    ver = crud.get_verification_for_run(db, run_id)
    alerts = crud.get_alerts_for_run(db, run_id)
    prov = crud.get_provenance_for_run(db, run_id)
    params = prov.parameters_json if prov else {}

    return {
        "run_id": run.run_id,
        "scenario_type": run.scenario_type,
        "status": run.status,
        "data_source_mode": params.get("data_source_mode", "SYNTHETIC"),
        "synthetic": params.get("synthetic", True),
        "source_type": params.get("source_type", "CONTROLLED_PHYSICS_BENCHMARK"),
        "events_count": len(events),
        "events": [
            {
                "event_id": e.event_id,
                "track_id": e.track_id,
                "peak_severity": e.peak_severity,
                "peak_precip_mm": e.peak_precip_mm,
                "duration_hours": e.duration_hours,
                "heading": e.heading_compass
            }
            for e in events
        ],
        "verification_summary": {
            "f1_score": ver.f1_score if ver else None,
            "spatial_iou": ver.spatial_iou if ver else None,
            "mean_displacement_km": ver.mean_displacement_km if ver else None
        } if ver else None,
        "alerts_count": len(alerts),
        "provenance": {
            "model_version": prov.model_version if prov else "v3.1.0",
            "sha256": prov.dataset_sha256 if prov else None,
            "data_source_mode": params.get("data_source_mode", "SYNTHETIC"),
            "synthetic": params.get("synthetic", True),
            "input_artifact": params.get("input_artifact", "N/A"),
            "variables_used": params.get("variables_used", [])
        } if prov else None
    }


@router.get("/events")
def list_events(run_id: Optional[str] = None, db: Session = Depends(get_db)):
    if run_id:
        events = crud.get_events_for_run(db, run_id)
    else:
        runs = crud.get_all_runs(db)
        if not runs:
            return []
        events = crud.get_events_for_run(db, runs[0].run_id)

    return [
        {
            "event_id": e.event_id,
            "run_id": e.run_id,
            "track_id": e.track_id,
            "event_name": e.event_name,
            "start_lead_time": e.start_lead_time,
            "end_lead_time": e.end_lead_time,
            "duration_hours": e.duration_hours,
            "peak_severity": e.peak_severity,
            "peak_precip_mm": e.peak_precip_mm,
            "min_mslp_hpa": e.min_mslp_hpa,
            "mean_speed_kmh": e.mean_speed_kmh,
            "total_distance_km": e.total_distance_km,
            "heading_compass": e.heading_compass,
            "trajectory_points": e.trajectory_points_json,
            "uncertainty_cone": e.uncertainty_cone_json
        }
        for e in events
    ]


@router.get("/events/{event_id}")
def get_event(event_id: str, db: Session = Depends(get_db)):
    evt = crud.get_event_by_id(db, event_id)
    if not evt:
        raise HTTPException(status_code=404, detail="Event not found")
    return {
        "event_id": evt.event_id,
        "run_id": evt.run_id,
        "track_id": evt.track_id,
        "event_name": evt.event_name,
        "start_lead_time": evt.start_lead_time,
        "end_lead_time": evt.end_lead_time,
        "duration_hours": evt.duration_hours,
        "peak_severity": evt.peak_severity,
        "peak_precip_mm": evt.peak_precip_mm,
        "min_mslp_hpa": evt.min_mslp_hpa,
        "mean_speed_kmh": evt.mean_speed_kmh,
        "total_distance_km": evt.total_distance_km,
        "heading_compass": evt.heading_compass,
        "trajectory_points": evt.trajectory_points_json,
        "uncertainty_cone": evt.uncertainty_cone_json
    }


@router.get("/events/{event_id}/trajectory")
def get_event_trajectory(event_id: str, db: Session = Depends(get_db)):
    evt = crud.get_event_by_id(db, event_id)
    if not evt:
        raise HTTPException(status_code=404, detail="Event not found")
    return {
        "event_id": evt.event_id,
        "track_id": evt.track_id,
        "heading": evt.heading_compass,
        "points": evt.trajectory_points_json
    }


@router.get("/events/{event_id}/uncertainty")
def get_event_uncertainty(event_id: str, db: Session = Depends(get_db)):
    evt = crud.get_event_by_id(db, event_id)
    if not evt:
        raise HTTPException(status_code=404, detail="Event not found")
    return evt.uncertainty_cone_json or {}


@router.get("/fields/{run_id}/{lead_time}/{variable}")
def get_field_data(
    run_id: str, lead_time: int, variable: str,
    member: Optional[int] = None
):
    nc_path = DATASET_DIR / f"{run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        if variable not in ds and f"gt_{variable}" not in ds:
            raise HTTPException(status_code=400, detail=f"Variable {variable} not found in dataset")

        lats = ds["latitude"].values.tolist()
        lons = ds["longitude"].values.tolist()
        leads = ds["lead_time"].values

        if lead_time not in leads:
            raise HTTPException(status_code=400, detail=f"Lead time {lead_time} not in dataset")

        t_idx = int(np.where(leads == lead_time)[0][0])

        if variable.startswith("gt_"):
            raw_field = ds[variable].values[t_idx]
            ens_mean = raw_field
            ens_std = np.zeros_like(raw_field)
        elif member is not None and "member" in ds.dims and member < len(ds.member):
            raw_field = ds[variable].values[t_idx, member]
            ens_mean = raw_field
            ens_std = np.zeros_like(raw_field)
        else:
            raw_field = ds[variable].values[t_idx]
            if raw_field.ndim >= 3:
                ens_mean = np.mean(raw_field, axis=0)
                ens_std = np.std(raw_field, axis=0)
            else:
                ens_mean = raw_field
                ens_std = np.zeros_like(raw_field)

        return {
            "run_id": run_id,
            "lead_time": lead_time,
            "variable": variable,
            "latitudes": lats,
            "longitudes": lons,
            "min_val": float(np.min(ens_mean)),
            "max_val": float(np.max(ens_mean)),
            "mean_grid": ens_mean.tolist(),
            "std_grid": ens_std.tolist()
        }


@router.get("/efi/{run_id}/{lead_time}")
def get_efi_data(run_id: str, lead_time: int, variable: str = "precipitation"):
    nc_path = DATASET_DIR / f"{run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        leads = ds["lead_time"].values
        if lead_time not in leads:
            raise HTTPException(status_code=400, detail=f"Lead time {lead_time} not found")
        t_idx = int(np.where(leads == lead_time)[0][0])
        var_data = ds[variable].values[t_idx]

        res = orchestrator.efi_engine.compute_efi_field(var_data, variable=variable)
        lats = ds["latitude"].values.tolist()
        lons = ds["longitude"].values.tolist()

        return {
            "run_id": run_id,
            "lead_time": lead_time,
            "variable": variable,
            "latitudes": lats,
            "longitudes": lons,
            "efi_grid": res["efi"].tolist(),
            "sot_grid": res["sot"].tolist(),
            "z_score_grid": res["z_score"].tolist(),
            "peak_efi": float(np.max(res["efi"])),
            "max_sot": float(np.max(res["sot"]))
        }


@router.get("/verification/{run_id}")
def get_verification_report(run_id: str, db: Session = Depends(get_db)):
    ver = crud.get_verification_for_run(db, run_id)
    if not ver:
        raise HTTPException(status_code=404, detail="Verification report not found")
    return ver.full_report_json


@router.get("/alerts/{run_id}")
def get_alerts_for_run(run_id: str, db: Session = Depends(get_db)):
    alerts = crud.get_alerts_for_run(db, run_id)
    return [
        {
            "alert_id": a.alert_id,
            "run_id": a.run_id,
            "event_id": a.event_id,
            "alert_level": a.alert_level,
            "lead_time_onset": a.lead_time_onset,
            "peak_lead_time": a.peak_lead_time,
            "affected_regions": a.affected_regions,
            "primary_threat": a.primary_threat,
            "trigger_evidence": a.trigger_evidence,
            "physics_audit_passed": a.physics_audit_passed
        }
        for a in alerts
    ]


@router.get("/provenance/{run_id}")
def get_provenance_record(run_id: str, db: Session = Depends(get_db)):
    prov = crud.get_provenance_for_run(db, run_id)
    if not prov:
        raise HTTPException(status_code=404, detail="Provenance record not found")
    params = prov.parameters_json or {}
    return {
        "run_id": prov.run_id,
        "dataset_sha256": prov.dataset_sha256,
        "random_seed": prov.random_seed,
        "model_version": prov.model_version,
        "execution_duration_sec": prov.execution_duration_sec,
        "pipeline_steps_executed": prov.pipeline_steps_executed,
        "parameters": params,
        "data_source_mode": params.get("data_source_mode", "SYNTHETIC"),
        "synthetic": params.get("synthetic", True),
        "source_type": params.get("source_type", "CONTROLLED_PHYSICS_BENCHMARK"),
        "input_artifact": params.get("input_artifact", "N/A"),
        "created_at": str(prov.created_at)
    }


# ==========================================
# ML & Research Microscope Endpoints
# ==========================================

@router.get("/downscaling/compare/{run_id}/{lead_time}")
def get_downscaling_comparison(run_id: str, lead_time: int):
    nc_path = DATASET_DIR / f"{run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        leads = ds["lead_time"].values
        if lead_time not in leads:
            raise HTTPException(status_code=400, detail=f"Lead time {lead_time} not found")
        t_idx = int(np.where(leads == lead_time)[0][0])
        coarse_precip = np.mean(ds["precipitation"].values[t_idx], axis=0)
        coarse_mslp = np.mean(ds["mslp"].values[t_idx], axis=0)

        # High-intensity storm patch
        max_idx = np.unravel_index(np.argmax(coarse_precip), coarse_precip.shape)
        cy, cx = max_idx[0], max_idx[1]
        y0, y1 = max(0, cy - 12), min(coarse_precip.shape[0], cy + 12)
        x0, x1 = max(0, cx - 12), min(coarse_precip.shape[1], cx + 12)

        p_patch = coarse_precip[y0:y1, x0:x1]
        m_patch = coarse_mslp[y0:y1, x0:x1]

        res = downscaling_engine.compare_downscaling_methods(p_patch, m_patch)
        return {
            "run_id": run_id,
            "lead_time": lead_time,
            "coarse_patch": p_patch.tolist(),
            "bicubic_patch": res["bicubic_field"].tolist(),
            "ml_patch": res["ml_field"].tolist(),
            "metrics": res["metrics"]
        }


@router.get("/physics/audit/{run_id}/{lead_time}")
def get_physics_audit(run_id: str, lead_time: int, variable: str = "precipitation"):
    nc_path = DATASET_DIR / f"{run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        leads = ds["lead_time"].values
        if lead_time not in leads:
            raise HTTPException(status_code=400, detail="Lead time not found")
        t_idx = int(np.where(leads == lead_time)[0][0])
        var_data = np.mean(ds[variable].values[t_idx], axis=0)

        report = physics_validator.validate_field(
            field=var_data,
            variable_name=variable,
            grid_resolution_deg=0.25
        )
        return report


@router.get("/explainability/{event_id}")
def get_explainability(event_id: str, db: Session = Depends(get_db)):
    evt = crud.get_event_by_id(db, event_id)
    if not evt:
        raise HTTPException(status_code=404, detail="Event not found")

    nc_path = DATASET_DIR / f"{evt.run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        pts = evt.trajectory_points_json
        peak_pt = pts[int(np.argmax([p["peak_precip_mm"] for p in pts]))]
        peak_lead = peak_pt["lead_time"]
        t_idx = int(np.where(ds["lead_time"].values == peak_lead)[0][0])

        p = np.mean(ds["precipitation"].values[t_idx], axis=0)
        m = np.mean(ds["mslp"].values[t_idx], axis=0)
        t = np.mean(ds["temperature_2m"].values[t_idx], axis=0) if "temperature_2m" in ds else np.full_like(p, 300.0)
        u = np.mean(ds["u_wind_850"].values[t_idx], axis=0) if "u_wind_850" in ds else np.zeros_like(p)
        v = np.mean(ds["v_wind_850"].values[t_idx], axis=0) if "v_wind_850" in ds else np.zeros_like(p)

        diag = anomaly_explainer.explain_event(
            event_id=event_id,
            precip_field=p,
            mslp_field=m,
            temp_field=t,
            u_wind_field=u,
            v_wind_field=v,
            event_centroid_lat=peak_pt["lat"],
            event_centroid_lon=peak_pt["lon"]
        )
        return diag


@router.get("/research/st-gnn/{run_id}")
def get_st_gnn_predictions(run_id: str):
    """Executes Spherical ST-GNN on the run dataset and returns graph-propagated forecast fields."""
    nc_path = DATASET_DIR / f"{run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        p_seq = np.mean(ds["precipitation"].values, axis=1)
        m_seq = np.mean(ds["mslp"].values, axis=1)
        t_seq = np.mean(ds["temperature_2m"].values, axis=1) if "temperature_2m" in ds else np.full_like(p_seq, 300.0)
        u_seq = np.mean(ds["u_wind_850"].values, axis=1) if "u_wind_850" in ds else np.zeros_like(p_seq)
        v_seq = np.mean(ds["v_wind_850"].values, axis=1) if "v_wind_850" in ds else np.zeros_like(p_seq)

        res = st_gnn_manager.predict_spatiotemporal_anomalies(p_seq, m_seq, t_seq, u_seq, v_seq)
        return {
            "run_id": run_id,
            "model": "SphericalSpatioTemporalGNN",
            "mesh_nodes": st_gnn_manager.graph.n_nodes,
            "mesh_edges": st_gnn_manager.graph.n_edges,
            "anomaly_field_shape": list(res["probabilities"].shape),
            "lead_time_metrics": {
                "max_probability": float(np.max(res["probabilities"])),
                "high_anomaly_cells": int(np.sum(res["probabilities"] > 0.60))
            },
            "centroid_velocities": res["velocities"].tolist(),
            "fallback_triggered": res.get("fallback_triggered", False)
        }


@router.get("/research/physics-downscaling/{run_id}/{lead_time}")
def get_physics_downscaling(run_id: str, lead_time: int):
    """Runs Physics-Informed Super-Resolution U-Net with mass conservation metrics."""
    nc_path = DATASET_DIR / f"{run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        leads = ds["lead_time"].values
        if lead_time not in leads:
            raise HTTPException(status_code=400, detail="Lead time not found")
        t_idx = int(np.where(leads == lead_time)[0][0])

        coarse_p = np.mean(ds["precipitation"].values[t_idx], axis=0)
        coarse_m = np.mean(ds["mslp"].values[t_idx], axis=0)
        u = np.mean(ds["u_wind_850"].values[t_idx], axis=0) if "u_wind_850" in ds else np.zeros_like(coarse_p)
        v = np.mean(ds["v_wind_850"].values[t_idx], axis=0) if "v_wind_850" in ds else np.zeros_like(coarse_p)
        wind = np.sqrt(u**2 + v**2)

        max_idx = np.unravel_index(np.argmax(coarse_p), coarse_p.shape)
        cy, cx = max_idx[0], max_idx[1]
        y0, y1 = max(0, cy - 12), min(coarse_p.shape[0], cy + 12)
        x0, x1 = max(0, cx - 12), min(coarse_p.shape[1], cx + 12)

        p_patch = coarse_p[y0:y1, x0:x1]
        m_patch = coarse_mslp[y0:y1, x0:x1]
        w_patch = wind[y0:y1, x0:x1]
        target_shape = (p_patch.shape[0] * 5, p_patch.shape[1] * 5)

        pi_res = advanced_downscaling_manager.downscale_field(
            coarse_precip=p_patch,
            coarse_mslp=m_patch,
            coarse_wind=w_patch,
            target_shape=target_shape
        )

        return {
            "run_id": run_id,
            "lead_time": lead_time,
            "model": "PhysicsInformedUNetDownscaler",
            "coarse_patch": p_patch.tolist(),
            "physics_fine_patch": pi_res["fine_field"].tolist(),
            "mean_intensity": pi_res["mean_fine"],
            "peak_intensity": pi_res["max_fine"],
            "p99_intensity": pi_res["p99_fine"],
            "fallback_triggered": pi_res.get("fallback_triggered", False)
        }


@router.get("/research/diffusion/{run_id}/{lead_time}")
def get_diffusion_realizations(run_id: str, lead_time: int, num_members: int = 2):
    """Generates stochastic ensemble downscaled realizations via Conditional Diffusion."""
    nc_path = DATASET_DIR / f"{run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        leads = ds["lead_time"].values
        if lead_time not in leads:
            raise HTTPException(status_code=400, detail="Lead time not found")
        t_idx = int(np.where(leads == lead_time)[0][0])
        coarse_p = np.mean(ds["precipitation"].values[t_idx], axis=0)

        max_idx = np.unravel_index(np.argmax(coarse_p), coarse_p.shape)
        cy, cx = max_idx[0], max_idx[1]
        y0, y1 = max(0, cy - 8), min(coarse_p.shape[0], cy + 8)
        x0, x1 = max(0, cx - 8), min(coarse_p.shape[1], cx + 8)
        p_patch = coarse_p[y0:y1, x0:x1]

        diff_res = diffusion_engine.sample_ensemble_realizations(
            coarse_precip=p_patch,
            num_members=num_members,
            fine_shape=(p_patch.shape[0] * 5, p_patch.shape[1] * 5)
        )

        return {
            "run_id": run_id,
            "lead_time": lead_time,
            "model": diff_res["model"],
            "num_members": num_members,
            "ensemble_mean": diff_res["ensemble_mean"].tolist(),
            "ensemble_spread": diff_res["ensemble_spread"].tolist(),
            "fallback_triggered": diff_res.get("fallback_triggered", False)
        }


@router.get("/research/tracking-hypotheses/{run_id}")
def get_tracking_hypotheses(run_id: str, db: Session = Depends(get_db)):
    events = crud.get_events_for_run(db, run_id)
    if not events:
        raise HTTPException(status_code=404, detail="No events found for run")

    dets_by_lead: Dict[int, List[Dict[str, Any]]] = {}
    for evt in events:
        for pt in evt.trajectory_points_json:
            lt = pt["lead_time"]
            if lt not in dets_by_lead:
                dets_by_lead[lt] = []
            det_item = {
                "centroid_lat": pt["lat"],
                "centroid_lon": pt["lon"],
                "bounding_box": pt.get("bounding_box", [pt["lat"]-0.5, pt["lon"]-0.5, pt["lat"]+0.5, pt["lon"]+0.5]),
                "peak_precip_mm": pt["peak_precip_mm"],
                "min_mslp_hpa": pt["min_mslp_hpa"],
                "severity_score": pt.get("severity_score", 0.8),
                "lead_time": lt
            }
            dets_by_lead[lt].append(det_item)

    mht_res = advanced_tracker.track_multi_hypothesis(dets_by_lead)
    return mht_res


@router.get("/data-adapter/info")
def get_data_adapter_info():
    """Provides real data ingestion interface readiness metadata."""
    return weather_adapter.get_supported_adapters()


@router.get("/microscope/extreme-preservation/{run_id}/{lead_time}")
def get_extreme_preservation_microscope(run_id: str, lead_time: int):
    nc_path = DATASET_DIR / f"{run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        leads = ds["lead_time"].values
        if lead_time not in leads:
            raise HTTPException(status_code=400, detail="Lead time not found")
        t_idx = int(np.where(leads == lead_time)[0][0])
        coarse_p = np.mean(ds["precipitation"].values[t_idx], axis=0)
        coarse_m = np.mean(ds["mslp"].values[t_idx], axis=0)

        down_out = advanced_downscaling_manager.downscale_field(
            coarse_precip=coarse_p,
            coarse_mslp=coarse_m,
            target_shape=(coarse_p.shape[0] * 5, coarse_p.shape[1] * 5)
        )
        fine_p = down_out["fine_field"]

        scorecard = preservation_scorecard.compute_scorecard(
            coarse_field=coarse_p,
            downscaled_field=fine_p
        )

        return {
            "run_id": run_id,
            "lead_time": lead_time,
            "scorecard": scorecard,
            "coarse_sample_patch": coarse_p[10:25, 10:25].tolist(),
            "downscaled_sample_patch": fine_p[50:125, 50:125].tolist()
        }


@router.get("/events/{event_id}/footprint-history")
def get_event_footprint_history(event_id: str, db: Session = Depends(get_db)):
    evt = crud.get_event_by_id(db, event_id)
    if not evt:
        raise HTTPException(status_code=404, detail="Event not found")

    pts = evt.trajectory_points_json
    footprint_history = []
    for pt in pts:
        footprint_history.append({
            "lead_time": pt["lead_time"],
            "centroid_lat": pt["lat"],
            "centroid_lon": pt["lon"],
            "bounding_box": pt.get("bounding_box", [pt["lat"]-0.5, pt["lon"]-0.5, pt["lat"]+0.5, pt["lon"]+0.5]),
            "prev_bounding_box": pt.get("prev_bounding_box", pt.get("bounding_box")),
            "bbox_delta": pt.get("bbox_delta", [0.0, 0.0, 0.0, 0.0]),
            "area_km2": pt.get("area_km2", 0.0),
            "area_expansion_rate_km2h": pt.get("area_expansion_rate_km2h", 0.0),
            "footprint_evolution": pt.get("footprint_evolution", "STABLE"),
            "peak_precip_mm": pt.get("peak_precip_mm", 0.0),
            "step_speed_kmh": pt.get("step_speed_kmh", 0.0),
            "step_bearing_deg": pt.get("step_bearing_deg", 0.0),
            "polygon": pt.get("polygon", [])
        })

    return {
        "event_id": evt.event_id,
        "track_id": evt.track_id,
        "run_id": evt.run_id,
        "total_timesteps": len(footprint_history),
        "footprint_evolution_history": footprint_history
    }
