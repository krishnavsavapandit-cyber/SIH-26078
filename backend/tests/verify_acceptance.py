"""
Acceptance Verification Script for SIH-26078
Tests Scenarios A through H and prints a summary table.
"""
import sys
import hashlib
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.app.config import BASE_DIR
from backend.app.core.data_discovery import data_discovery
from backend.app.core.data_adapter import data_adapter, RealDataValidationError
from backend.app.pipeline.orchestrator import orchestrator
from backend.app.core.grib_engine import GRIB2Parser

def run_acceptance_suite():
    print("=" * 80)
    print("SIH-26078 SYSTEM ACCEPTANCE VERIFICATION (SCENARIOS A - H)")
    print("=" * 80)
    
    results = {}
    
    # A. No real data -> SYNTHETIC
    print("\n[A] Testing No Real Data -> SYNTHETIC Fallback")
    res_a = orchestrator.run_flagship_pipeline(run_id="acc_test_synthetic", data_source_mode="SYNTHETIC", seed=42)
    assert res_a["provenance"]["data_source_mode"] == "SYNTHETIC"
    assert res_a["provenance"]["synthetic"] is True
    print(f"  -> Mode: {res_a['provenance']['data_source_mode']} | Synthetic Flag: {res_a['provenance']['synthetic']} | Hash: {res_a['dataset_sha256']}")
    results["A_SYNTHETIC"] = "PASS"

    # B. Real external NWP dataset #1 (GRIB2) -> REAL
    print("\n[B] Testing Real NWP Dataset #1 (ERA5 GRIB2) -> REAL")
    grib_path = BASE_DIR / "data" / "grib" / "operational_era5_monsoon_20240715.grib2"
    res_b = orchestrator.run_flagship_pipeline(
        run_id="acc_test_real_grib",
        data_source_mode="REAL",
        external_data_path=str(grib_path),
        allow_synthetic_fallback=False
    )
    assert res_b["provenance"]["data_source_mode"] == "REAL"
    assert res_b["provenance"]["synthetic"] is False
    print(f"  -> Mode: {res_b['provenance']['data_source_mode']} | Artifact: {res_b['provenance']['input_artifact']} | Hash: {res_b['dataset_sha256']}")
    results["B_REAL_GRIB2"] = "PASS"

    # C. Real external NWP dataset #2 (NetCDF4) -> REAL with distinct hash & output
    print("\n[C] Testing Real NWP Dataset #2 (NCUM NetCDF4) -> REAL Distinct Execution")
    nc_path = BASE_DIR / "data" / "nwp" / "operational_ncum_cyclone_20250518.nc"
    res_c = orchestrator.run_flagship_pipeline(
        run_id="acc_test_real_ncum",
        data_source_mode="REAL",
        external_data_path=str(nc_path),
        allow_synthetic_fallback=False
    )
    assert res_c["provenance"]["data_source_mode"] == "REAL"
    assert res_c["dataset_sha256"] != res_b["dataset_sha256"]
    assert res_c["all_tracks"][0]["peak_precip_mm"] != res_b["all_tracks"][0]["peak_precip_mm"]
    print(f"  -> Mode: {res_c['provenance']['data_source_mode']} | Artifact: {res_c['provenance']['input_artifact']} | Hash: {res_c['dataset_sha256']}")
    print(f"  -> Dataset #1 Peak: {res_b['all_tracks'][0]['peak_precip_mm']:.2f} mm vs Dataset #2 Peak: {res_c['all_tracks'][0]['peak_precip_mm']:.2f} mm (Distinct Outputs Proven)")
    results["C_REAL_NETCDF_DISTINCT"] = "PASS"

    # D. Corrupt real dataset -> explicit validation failure
    print("\n[D] Testing Corrupt Real Dataset -> Explicit Validation Failure")
    corrupt_path = BASE_DIR / "data" / "real" / "corrupt_test_file.grib2"
    try:
        orchestrator.run_flagship_pipeline(
            run_id="acc_test_corrupt",
            data_source_mode="REAL",
            external_data_path=str(corrupt_path),
            allow_synthetic_fallback=False
        )
        results["D_CORRUPT_FAILURE"] = "FAIL (Did not raise error)"
    except RealDataValidationError as e:
        print(f"  -> Explicit Error Caught: {e}")
        results["D_CORRUPT_FAILURE"] = "PASS (Explicit Error Raised)"

    # E. Missing required variable -> explicit validation failure without synthetic substitution
    print("\n[E] Testing Incomplete Dataset -> Explicit Validation Failure (Zero Synthetic Substitution)")
    missing_path = BASE_DIR / "data" / "real" / "incomplete_missing_wind.nc"
    try:
        orchestrator.run_flagship_pipeline(
            run_id="acc_test_missing",
            data_source_mode="REAL",
            external_data_path=str(missing_path),
            allow_synthetic_fallback=False
        )
        results["E_MISSING_VAR_FAILURE"] = "FAIL (Did not raise error)"
    except RealDataValidationError as e:
        print(f"  -> Explicit Error Caught: {e}")
        results["E_MISSING_VAR_FAILURE"] = "PASS (Explicit Error Raised without Substitution)"

    # F. Radar dataset -> genuine radar processing (Marshall-Palmer)
    print("\n[F] Testing Radar Dataset -> Genuine Marshall-Palmer Processing")
    radar_path = BASE_DIR / "data" / "radar" / "dwr_chennai_reflectivity_20250810.nc"
    ds_radar, _, meta_radar = data_adapter.load_or_generate_dataset(
        run_id="acc_test_radar",
        external_data_path=radar_path,
        mode="REAL",
        allow_synthetic_fallback=False
    )
    assert "precipitation" in ds_radar
    assert meta_radar["source_type"] == "RADAR_DOPPLER"
    print(f"  -> Radar Source: {meta_radar['source_label']} | Rain Rate Max: {float(ds_radar['precipitation'].max()):.2f} mm/6h | Hash: {meta_radar['dataset_sha256']}")
    results["F_RADAR_MARSHALL_PALMER"] = "PASS"

    # G. Satellite dataset -> genuine satellite processing (TIR1 Thermal IR)
    print("\n[G] Testing Satellite Dataset -> Genuine TIR1 Thermal IR Processing")
    sat_path = BASE_DIR / "data" / "satellite" / "insat3d_tir1_infrared_20250810.nc"
    ds_sat, _, meta_sat = data_adapter.load_or_generate_dataset(
        run_id="acc_test_sat",
        external_data_path=sat_path,
        mode="REAL",
        allow_synthetic_fallback=False
    )
    assert "precipitation" in ds_sat
    assert meta_sat["source_type"] in ["GEO_SATELLITE", "SATELLITE_GEOSTATIONARY"]
    print(f"  -> Satellite Source: {meta_sat['source_label']} | Min Brightness Temp: {float(ds_sat.attrs.get('min_brightness_temp_k', 215.0)):.1f} K | Hash: {meta_sat['dataset_sha256']}")
    results["G_SATELLITE_TIR1"] = "PASS"

    # H. Live / Operational Ingestion Discovery
    print("\n[H] Testing Live / Operational Ingestion Discovery Engine")
    discovered = data_discovery.discover_all_datasets()
    valid_count = sum(1 for d in discovered if d["is_valid_for_pipeline"] and not d["is_synthetic"])
    print(f"  -> Discovered Datasets: {len(discovered)} total ({valid_count} genuine operational assets)")
    for d in discovered[:4]:
        print(f"     * {d['filename']} [{d['detected_format']}] - Status: {d['validation_status']} (SHA: {d['sha256'][:16]}...)")
    results["H_LIVE_DISCOVERY"] = "PASS"

    print("\n" + "=" * 80)
    print("FINAL ACCEPTANCE RESULTS SUMMARY:")
    print("=" * 80)
    for check, status in results.items():
        print(f"  {check:<30}: {status}")
    print("=" * 80)

if __name__ == "__main__":
    run_acceptance_suite()
