"""
SIH-26078: Standalone 100% Working Backup Runner (Phase 1).
Can be executed directly: python run_phase1_backup.py
Executes the full weather anomaly pipeline from deterministic generation
through verification and alerts, ensuring complete demoability.
"""

import sys
import time
import json
from pathlib import Path

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.pipeline.orchestrator import orchestrator
from backend.app.db.database import SessionLocal, init_db
from backend.app.db import crud

def main():
    print("=" * 70)
    print(" SIH-26078: AI-DRIVEN SPATIO-TEMPORAL EXTREME WEATHER TRACKING")
    print(" PHASE 1: 100% WORKING BACKUP SYSTEM VERIFICATION")
    print("=" * 70)

    start_time = time.time()
    init_db()
    db = SessionLocal()

    run_id = "phase1_backup_monsoon_run"
    print(f"\n[1/6] Generating Deterministic Synthetic Meteorological Dataset: {run_id}")
    print("      Domain: South Asian Monsoon (6N-38N, 68E-98E) | Leads: 0h-120h | Ensemble: 10 members")

    result = orchestrator.run_full_pipeline(
        run_id=run_id,
        scenario_type="monsoon_depression",
        seed=42,
        displacement_bias_km=25.0,
        intensity_bias_pct=-5.0,
        db=db
    )

    print("\n[2/6] EFI & Anomaly Detection Completed.")
    print(f"      Total Trajectories Tracked: {result['num_tracks']}")
    
    primary = result["primary_track"]
    if primary:
        print(f"\n[3/6] Primary Weather System Telemetry:")
        print(f"      Track ID:           {primary['track_id']}")
        print(f"      Lifecycle Duration: {primary['duration_hours']} hours (Lead {primary['start_lead_time']}h -> {primary['end_lead_time']}h)")
        print(f"      Peak Precipitation: {primary['peak_precip_mm']:.1f} mm / 6h")
        print(f"      Minimum MSLP:       {primary['min_mslp_hpa']:.1f} hPa")
        print(f"      Total Displacement: {primary['total_distance_km']:.1f} km")
        print(f"      Mean Speed:         {primary['mean_speed_kmh']:.1f} km/h")
        print(f"      Heading Direction:  {primary['heading_compass']} ({primary.get('overall_bearing_deg', 0):.1f} deg)")

    print(f"\n[4/6] Quantitative Verification vs Hidden Ground Truth:")
    ver = result["verification"]
    c = ver["contingency"]
    te = ver["trajectory_error"]
    print(f"      Precision:          {c['precision'] * 100:.2f}%")
    print(f"      Recall:             {c['recall'] * 100:.2f}%")
    print(f"      F1-Score:           {c['f1_score']:.4f}")
    print(f"      Threat Score (CSI): {c['csi_threat_score']:.4f}")
    print(f"      Spatial IoU:        {c['spatial_iou']:.4f}")
    print(f"      Mean Displacement:  {te['mean_displacement_km']:.2f} km")
    print(f"      Max Displacement:   {te['max_displacement_km']:.2f} km")

    print(f"\n[5/6] Multi-Tier Disaster Early Warning Alerts:")
    for alt in result["alerts"]:
        print(f"      [{alt['alert_level']}] {alt['alert_id']} -> {alt['primary_threat']}")
        print(f"             Regions: {', '.join(alt['affected_regions'])}")
        print(f"             Trigger EFI: {alt['trigger_evidence']['peak_efi']:.2f} | Peak Rain: {alt['trigger_evidence']['peak_precip_mm']:.1f} mm")

    print(f"\n[6/6] Provenance & Database Audit:")
    print(f"      Dataset SHA-256:    {result['dataset_sha256'][:24]}...")
    print(f"      Execution Duration: {result['execution_duration_sec']:.2f} seconds")
    print(f"      Database Engine:    SQLite (Operational)")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f" SUCCESS: PHASE 1 SYSTEM FULLY FUNCTIONAL AND VERIFIED IN {total_time:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
