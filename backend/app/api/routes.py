"""
FastAPI REST API Routes for SIH-26078 Extreme Weather Intelligence System.
Exposes calculated meteorological fields, EFI maps, detections, trajectories,
uncertainty cones, verification metrics, alerts, and execution controls.
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
        "version": "1.0.0",
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
            "model_version": prov.model_version if prov else "v1.0.0",
            "sha256": prov.dataset_sha256 if prov else None
        } if prov else None
    }

@router.get("/events")
def list_events(run_id: Optional[str] = None, db: Session = Depends(get_db)):
    if run_id:
        events = crud.get_events_for_run(db, run_id)
    else:
        # Get latest run's events
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
            raw_field = ds[variable].values[t_idx] # (lats, lons)
            ens_mean = raw_field
            ens_std = np.zeros_like(raw_field)
        elif member is not None:
            raw_field = ds[variable].values[t_idx, member]
            ens_mean = raw_field
            ens_std = np.zeros_like(raw_field)
        else:
            raw_field = ds[variable].values[t_idx] # (members, lats, lons)
            ens_mean = np.mean(raw_field, axis=0)
            ens_std = np.std(raw_field, axis=0)

        # Downsample or serialize grid for web rendering
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
