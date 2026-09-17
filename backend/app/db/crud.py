"""
CRUD helper operations for metadata, runs, events, verification, and alerts.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.app.db.models import (
    ForecastRunRecord, EventRecord, VerificationRecord, AlertRecord, ProvenanceRecord
)

def create_or_update_run(db: Session, run_data: Dict[str, Any]) -> ForecastRunRecord:
    existing = db.query(ForecastRunRecord).filter(ForecastRunRecord.run_id == run_data["run_id"]).first()
    if existing:
        for k, v in run_data.items():
            setattr(existing, k, v)
        db.commit()
        db.refresh(existing)
        return existing
    
    record = ForecastRunRecord(**run_data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_run(db: Session, run_id: str) -> Optional[ForecastRunRecord]:
    return db.query(ForecastRunRecord).filter(ForecastRunRecord.run_id == run_id).first()

def get_all_runs(db: Session) -> List[ForecastRunRecord]:
    return db.query(ForecastRunRecord).order_by(ForecastRunRecord.created_at.desc()).all()

def save_event_records(db: Session, run_id: str, events: List[Dict[str, Any]]):
    # Clear existing events for this run to avoid duplicates on re-run
    db.query(EventRecord).filter(EventRecord.run_id == run_id).delete()
    
    for evt in events:
        record = EventRecord(
            event_id=evt.get("event_id", f"EVT_{evt['track_id']}"),
            run_id=run_id,
            track_id=evt["track_id"],
            event_name=evt.get("event_name", "Extreme Weather Anomaly"),
            start_lead_time=evt["start_lead_time"],
            end_lead_time=evt["end_lead_time"],
            duration_hours=evt["duration_hours"],
            peak_severity=evt["peak_severity"],
            peak_precip_mm=evt["peak_precip_mm"],
            min_mslp_hpa=evt["min_mslp_hpa"],
            mean_speed_kmh=evt.get("mean_speed_kmh", 0.0),
            total_distance_km=evt.get("total_distance_km", 0.0),
            heading_compass=evt.get("heading_compass", "WNW"),
            status=evt.get("status", "ACTIVE"),
            trajectory_points_json=evt["trajectory_points"],
            uncertainty_cone_json=evt.get("uncertainty_cone")
        )
        db.add(record)
    db.commit()

def get_events_for_run(db: Session, run_id: str) -> List[EventRecord]:
    return db.query(EventRecord).filter(EventRecord.run_id == run_id).all()

def get_event_by_id(db: Session, event_id: str) -> Optional[EventRecord]:
    return db.query(EventRecord).filter(
        (EventRecord.event_id == event_id) | (EventRecord.track_id == event_id)
    ).first()

def save_verification_record(db: Session, run_id: str, ver_data: Dict[str, Any]) -> VerificationRecord:
    existing = db.query(VerificationRecord).filter(VerificationRecord.run_id == run_id).first()
    if existing:
        db.delete(existing)
        db.commit()

    record = VerificationRecord(
        run_id=run_id,
        precision=ver_data["contingency"]["precision"],
        recall=ver_data["contingency"]["recall"],
        f1_score=ver_data["contingency"]["f1_score"],
        csi_threat_score=ver_data["contingency"]["csi_threat_score"],
        spatial_iou=ver_data["contingency"]["spatial_iou"],
        mean_displacement_km=ver_data["trajectory_error"]["mean_displacement_km"],
        precip_rmse=ver_data["field_metrics"].get("precipitation", {}).get("rmse", 0.0),
        mslp_rmse=ver_data["field_metrics"].get("mslp", {}).get("rmse", 0.0),
        full_report_json=ver_data
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_verification_for_run(db: Session, run_id: str) -> Optional[VerificationRecord]:
    return db.query(VerificationRecord).filter(VerificationRecord.run_id == run_id).first()

def save_alert_records(db: Session, run_id: str, alerts: List[Dict[str, Any]]):
    db.query(AlertRecord).filter(AlertRecord.run_id == run_id).delete()
    for alt in alerts:
        record = AlertRecord(
            alert_id=alt["alert_id"],
            run_id=run_id,
            event_id=alt["event_id"],
            alert_level=alt["alert_level"],
            lead_time_onset=alt["lead_time_onset"],
            peak_lead_time=alt["peak_lead_time"],
            affected_regions=alt["affected_regions"],
            primary_threat=alt["primary_threat"],
            trigger_evidence=alt["trigger_evidence"],
            physics_audit_passed=alt.get("physics_audit_passed", True)
        )
        db.add(record)
    db.commit()

def get_alerts_for_run(db: Session, run_id: str) -> List[AlertRecord]:
    return db.query(AlertRecord).filter(AlertRecord.run_id == run_id).all()

def save_provenance_record(db: Session, prov_data: Dict[str, Any]) -> ProvenanceRecord:
    existing = db.query(ProvenanceRecord).filter(ProvenanceRecord.run_id == prov_data["run_id"]).first()
    if existing:
        db.delete(existing)
        db.commit()

    record = ProvenanceRecord(**prov_data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_provenance_for_run(db: Session, run_id: str) -> Optional[ProvenanceRecord]:
    return db.query(ProvenanceRecord).filter(ProvenanceRecord.run_id == run_id).first()
