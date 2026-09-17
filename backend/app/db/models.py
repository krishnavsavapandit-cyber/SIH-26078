"""
SQLAlchemy ORM Schemas for SIH-26078 Metadata & Verification Layer.
Stores metadata, tracks, uncertainty bounds, alerts, and verification logs.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.app.db.database import Base

class ForecastRunRecord(Base):
    __tablename__ = "forecast_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), unique=True, index=True, nullable=False)
    scenario_type = Column(String(64), nullable=False)
    dataset_path = Column(String(256), nullable=False)
    grid_resolution_deg = Column(Float, default=0.25)
    num_members = Column(Integer, default=10)
    num_lead_steps = Column(Integer, default=21)
    status = Column(String(32), default="COMPLETED")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    events = relationship("EventRecord", back_populates="run", cascade="all, delete-orphan")
    verification = relationship("VerificationRecord", back_populates="run", uselist=False, cascade="all, delete-orphan")
    alerts = relationship("AlertRecord", back_populates="run", cascade="all, delete-orphan")
    provenance = relationship("ProvenanceRecord", back_populates="run", uselist=False, cascade="all, delete-orphan")

class EventRecord(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(64), index=True, nullable=False)
    run_id = Column(String(64), ForeignKey("forecast_runs.run_id"), nullable=False)
    track_id = Column(String(64), index=True, nullable=False)
    event_name = Column(String(128), default="Extreme Meteorological Anomaly")
    start_lead_time = Column(Integer, nullable=False)
    end_lead_time = Column(Integer, nullable=False)
    duration_hours = Column(Integer, default=0)
    peak_severity = Column(Float, default=0.0)
    peak_precip_mm = Column(Float, default=0.0)
    min_mslp_hpa = Column(Float, default=1012.0)
    mean_speed_kmh = Column(Float, default=0.0)
    total_distance_km = Column(Float, default=0.0)
    heading_compass = Column(String(16), default="WNW")
    status = Column(String(32), default="ACTIVE")
    
    # JSON payload for full trajectory points and uncertainty cone
    trajectory_points_json = Column(JSON, nullable=False)
    uncertainty_cone_json = Column(JSON, nullable=True)

    run = relationship("ForecastRunRecord", back_populates="events")

class VerificationRecord(Base):
    __tablename__ = "verification_records"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("forecast_runs.run_id"), unique=True, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    csi_threat_score = Column(Float, nullable=False)
    spatial_iou = Column(Float, nullable=False)
    mean_displacement_km = Column(Float, nullable=False)
    precip_rmse = Column(Float, default=0.0)
    mslp_rmse = Column(Float, default=0.0)
    full_report_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("ForecastRunRecord", back_populates="verification")

class AlertRecord(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(64), unique=True, index=True, nullable=False)
    run_id = Column(String(64), ForeignKey("forecast_runs.run_id"), nullable=False)
    event_id = Column(String(64), index=True, nullable=False)
    alert_level = Column(String(32), nullable=False) # NORMAL, WATCH, HIGH, SEVERE, EXTREME
    lead_time_onset = Column(Integer, nullable=False)
    peak_lead_time = Column(Integer, nullable=False)
    affected_regions = Column(JSON, nullable=False)
    primary_threat = Column(String(128), nullable=False)
    trigger_evidence = Column(JSON, nullable=False)
    physics_audit_passed = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("ForecastRunRecord", back_populates="alerts")

class ProvenanceRecord(Base):
    __tablename__ = "provenance_records"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("forecast_runs.run_id"), unique=True, nullable=False)
    dataset_sha256 = Column(String(64), nullable=False)
    random_seed = Column(Integer, nullable=False)
    model_version = Column(String(64), default="v1.0.0-phase1")
    execution_duration_sec = Column(Float, default=0.0)
    pipeline_steps_executed = Column(JSON, nullable=False)
    parameters_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("ForecastRunRecord", back_populates="provenance")
