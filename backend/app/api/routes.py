"""
FastAPI REST API Routes for SIH-26078 Extreme Weather Intelligence System.
Exposes calculated meteorological fields, EFI maps, detections, trajectories,
uncertainty cones, verification metrics, alerts, downscaling comparisons,
physics audits, explainability diagnostics, and execution controls.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import xarray as xr

from backend.app.config import DATASET_DIR, domain_config
from backend.app.db.database import get_db
from backend.app.db import crud
from backend.app.pipeline.orchestrator import orchestrator
from backend.app.core.physics_validator import physics_validator
from backend.app.ml.downscaler_baseline import downscaling_engine
from backend.app.ml.explainability import anomaly_explainer
from backend.app.ml.st_gnn import st_gnn_manager
from backend.app.ml.advanced_downscaler import advanced_downscaling_manager
from backend.app.ml.diffusion_experiment import diffusion_engine
from backend.app.ml.advanced_tracker import advanced_tracker
from backend.app.ml.benchmark_suite import benchmark_suite, BENCHMARK_RESULTS_PATH
import json

router = APIRouter(prefix="/api", tags=["Weather Intelligence API"])

class GenerateRequest(BaseModel):
    run_id: str = "run_demo_monsoon_depression"
    scenario_type: str = "monsoon_depression"
    seed: int = 42
    displacement_bias_km: float = 25.0
    intensity_bias_pct: float = -5.0

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SIH-26078 Weather Anomaly Intelligence Backend",
        "version": "2.0.0-phase2-competition-core",
        "domain": "India & South Asia"
    }

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

@router.post("/dataset/generate")
def generate_dataset(req: GenerateRequest, db: Session = Depends(get_db)):
    result = orchestrator.run_full_pipeline(
        run_id=req.run_id,
        scenario_type=req.scenario_type,
        seed=req.seed,
        displacement_bias_km=req.displacement_bias_km,
        intensity_bias_pct=req.intensity_bias_pct,
        db=db
    )
    return result

@router.post("/pipeline/run")
def trigger_pipeline(req: GenerateRequest, db: Session = Depends(get_db)):
    return generate_dataset(req, db)

@router.get("/runs")
def list_runs(db: Session = Depends(get_db)):
    runs = crud.get_all_runs(db)
    return [
        {
            "run_id": r.run_id,
            "scenario_type": r.scenario_type,
            "status": r.status,
            "grid_resolution_deg": r.grid_resolution_deg,
            "num_members": r.num_members,
            "num_lead_steps": r.num_lead_steps,
            "created_at": r.created_at
        }
        for r in runs
    ]

@router.get("/runs/{run_id}")
def get_run_details(run_id: str, db: Session = Depends(get_db)):
    run = crud.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    events = crud.get_events_for_run(db, run_id)
    ver = crud.get_verification_for_run(db, run_id)
    alerts = crud.get_alerts_for_run(db, run_id)
    prov = crud.get_provenance_for_run(db, run_id)

    return {
        "run_id": run.run_id,
        "scenario_type": run.scenario_type,
        "status": run.status,
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
            "model_version": prov.model_version if prov else "v2.0.0",
            "sha256": prov.dataset_sha256 if prov else None
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
        elif member is not None:
            raw_field = ds[variable].values[t_idx, member]
            ens_mean = raw_field
            ens_std = np.zeros_like(raw_field)
        else:
            raw_field = ds[variable].values[t_idx]
            ens_mean = np.mean(raw_field, axis=0)
            ens_std = np.std(raw_field, axis=0)

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
        ens_data = ds[variable].values[t_idx]
        
        efi_res = orchestrator.efi_engine.compute_efi_field(ens_data, variable=variable)
        
        return {
            "run_id": run_id,
            "lead_time": lead_time,
            "variable": variable,
            "latitudes": ds["latitude"].values.tolist(),
            "longitudes": ds["longitude"].values.tolist(),
            "efi_grid": efi_res["efi"].tolist(),
            "sot_grid": efi_res["sot"].tolist(),
            "z_score_grid": efi_res["z_score"].tolist(),
            "peak_efi": float(np.max(efi_res["efi"])),
            "max_sot": float(np.max(efi_res["sot"]))
        }

@router.get("/downscaling/{run_id}/{lead_time}")
def get_downscaling_comparison(run_id: str, lead_time: int):
    nc_path = DATASET_DIR / f"{run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        leads = ds["lead_time"].values
        if lead_time not in leads:
            raise HTTPException(status_code=400, detail="Lead time not found")
        t_idx = int(np.where(leads == lead_time)[0][0])
        coarse_p = np.mean(ds["precipitation"].values[t_idx], axis=0) # (lats, lons)
        gt_p = ds["gt_precipitation"].values[t_idx]
        
        # Subsample high-intensity storm patch (e.g. 24x24 coarse -> 120x120 fine)
        max_idx = np.unravel_index(np.argmax(coarse_p), coarse_p.shape)
        cy, cx = max_idx[0], max_idx[1]
        
        y0, y1 = max(0, cy - 12), min(coarse_p.shape[0], cy + 12)
        x0, x1 = max(0, cx - 12), min(coarse_p.shape[1], cx + 12)
        
        patch_coarse = coarse_p[y0:y1, x0:x1]
        # Fine ground truth target (approx 5x)
        fine_gt_patch = downscaling_engine.bicubic_downscale(gt_p[y0:y1, x0:x1], (patch_coarse.shape[0] * 5, patch_coarse.shape[1] * 5))
        
        comp_metrics = downscaling_engine.compare_downscalers(patch_coarse, fine_gt_patch)
        
        bicubic_grid = downscaling_engine.bicubic_downscale(patch_coarse, fine_gt_patch.shape)
        ml_grid = downscaling_engine.ml_downscale(patch_coarse)

        return {
            "run_id": run_id,
            "lead_time": lead_time,
            "metrics": comp_metrics,
            "coarse_patch": patch_coarse.tolist(),
            "bicubic_patch": bicubic_grid.tolist(),
            "ml_patch": ml_grid.tolist(),
            "fine_gt_patch": fine_gt_patch.tolist()
        }

@router.get("/physics-audit/{run_id}/{lead_time}")
def get_physics_audit(run_id: str, lead_time: int):
    nc_path = DATASET_DIR / f"{run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        leads = ds["lead_time"].values
        if lead_time not in leads:
            raise HTTPException(status_code=400, detail="Lead time not found")
        t_idx = int(np.where(leads == lead_time)[0][0])
        
        p = np.mean(ds["precipitation"].values[t_idx], axis=0)
        t = np.mean(ds["temperature_2m"].values[t_idx], axis=0)
        rh = np.mean(ds["relative_humidity"].values[t_idx], axis=0)
        u = np.mean(ds["u_wind_850"].values[t_idx], axis=0)
        v = np.mean(ds["v_wind_850"].values[t_idx], axis=0)
        mslp = np.mean(ds["mslp"].values[t_idx], axis=0)

        res = physics_validator.validate_atmospheric_state(
            precip=p, temperature_2m=t, relative_humidity=rh,
            u_wind=u, v_wind=v, mslp=mslp
        )
        return res.dict()

@router.get("/explainability/{run_id}/{event_id}")
def get_explainability(run_id: str, event_id: str, db: Session = Depends(get_db)):
    evt = crud.get_event_by_id(db, event_id)
    if not evt:
        raise HTTPException(status_code=404, detail="Event not found")
    
    pts = evt.trajectory_points_json
    peak_pt = max(pts, key=lambda x: x["peak_precip_mm"])
    
    diag = anomaly_explainer.explain_detection(
        peak_efi=peak_pt.get("peak_efi", 0.85),
        z_score=2.8,
        sot_val=peak_pt.get("max_sot", 1.2),
        precip_mm=peak_pt["peak_precip_mm"],
        wind_ms=peak_pt["max_wind_ms"],
        mslp_deficit=1012.0 - peak_pt["min_mslp_hpa"],
        ens_spread=4.5
    )
    return diag

@router.get("/verification/{run_id}")
def get_verification_report(run_id: str, db: Session = Depends(get_db)):
    ver = crud.get_verification_for_run(db, run_id)
    if not ver:
        raise HTTPException(status_code=404, detail="Verification report not found for run")
    return ver.full_report_json

@router.get("/alerts/{run_id}")
def get_alerts(run_id: str, db: Session = Depends(get_db)):
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
def get_provenance(run_id: str, db: Session = Depends(get_db)):
    prov = crud.get_provenance_for_run(db, run_id)
    if not prov:
        raise HTTPException(status_code=404, detail="Provenance not found for run")
    return {
        "run_id": prov.run_id,
        "dataset_sha256": prov.dataset_sha256,
        "random_seed": prov.random_seed,
        "model_version": prov.model_version,
        "execution_duration_sec": prov.execution_duration_sec,
        "pipeline_steps_executed": prov.pipeline_steps_executed,
        "parameters": prov.parameters_json,
        "created_at": prov.created_at
    }

# =========================================================================
# Phase 3 Research & Scientific Benchmarking Endpoints
# =========================================================================

@router.get("/benchmark")
def get_benchmark_results():
    """Returns the latest cross-model scientific benchmark report."""
    if BENCHMARK_RESULTS_PATH.exists():
        try:
            with open(BENCHMARK_RESULTS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return benchmark_suite.run_benchmark(num_test_scenarios=1)

@router.post("/benchmark/run")
def trigger_benchmark(num_scenarios: int = 1):
    """Executes fresh cross-model benchmark evaluation."""
    return benchmark_suite.run_benchmark(num_test_scenarios=num_scenarios)

@router.get("/research/st-gnn/{run_id}")
def get_st_gnn_inference(run_id: str):
    """Executes Spatio-Temporal GNN message passing across forecast sequence."""
    nc_path = DATASET_DIR / f"{run_id}.nc"
    if not nc_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with xr.open_dataset(nc_path) as ds:
        p_seq = np.mean(ds["precipitation"].values, axis=1) # (T, H, W)
        m_seq = np.mean(ds["mslp"].values, axis=1)
        t_seq = np.mean(ds["temperature_2m"].values, axis=1)
        u_seq = np.mean(ds["u_wind_850"].values, axis=1)
        v_seq = np.mean(ds["v_wind_850"].values, axis=1)

        res = st_gnn_manager.predict_spatiotemporal_anomalies(p_seq, m_seq, t_seq, u_seq, v_seq)
        return {
            "run_id": run_id,
            "model": res["model"],
            "success": res["success"],
            "lead_times": ds["lead_time"].values.tolist(),
            "probabilities_summary": {
                "mean_probability": float(np.mean(res["probabilities"])),
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
        u = np.mean(ds["u_wind_850"].values[t_idx], axis=0)
        v = np.mean(ds["v_wind_850"].values[t_idx], axis=0)
        wind = np.sqrt(u**2 + v**2)

        # High-intensity storm patch
        max_idx = np.unravel_index(np.argmax(coarse_p), coarse_p.shape)
        cy, cx = max_idx[0], max_idx[1]
        y0, y1 = max(0, cy - 12), min(coarse_p.shape[0], cy + 12)
        x0, x1 = max(0, cx - 12), min(coarse_p.shape[1], cx + 12)

        p_patch = coarse_p[y0:y1, x0:x1]
        m_patch = coarse_m[y0:y1, x0:x1]
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

        # Storm patch for fast CPU diffusion
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
    """Executes Advanced Multi-Hypothesis Tracker across run detections."""
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

