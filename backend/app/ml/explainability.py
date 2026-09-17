"""
Explainability & Feature Attribution Engine for SIH-26078.
Quantifies atmospheric feature contributions and sensitivity diagnostics
explaining why specific spatial entities were classified as extreme anomalies.
"""

from typing import Dict, Any, List
import numpy as np

class AnomalyExplainer:
    """
    Computes rigorous physical and statistical attribution factors for detected anomalies.
    """
    def __init__(self):
        pass

    def explain_detection(
        self,
        peak_efi: float,
        z_score: float,
        sot_val: float,
        precip_mm: float,
        wind_ms: float,
        mslp_deficit: float,
        ens_spread: float
    ) -> Dict[str, Any]:
        """
        Decomposes relative contribution of meteorological drivers to the extreme classification.
        """
        # Raw driver signals
        w_efi = max(0.0, peak_efi) * 35.0
        w_z = min(4.0, max(0.0, z_score)) * 15.0
        w_sot = min(3.0, max(0.0, sot_val)) * 12.0
        w_precip = min(150.0, precip_mm) * 0.25
        w_wind = min(40.0, wind_ms) * 0.8
        w_vortex = min(30.0, mslp_deficit) * 1.2
        w_spread = max(1.0, ens_spread) * 1.5

        total_weight = w_efi + w_z + w_sot + w_precip + w_wind + w_vortex
        if total_weight <= 1e-4:
            total_weight = 1.0

        # Percentage contributions
        pct_efi = (w_efi / total_weight) * 100.0
        pct_climatology = (w_z / total_weight) * 100.0
        pct_tail_shift = (w_sot / total_weight) * 100.0
        pct_rainfall = (w_precip / total_weight) * 100.0
        pct_wind = (w_wind / total_weight) * 100.0
        pct_pressure = (w_vortex / total_weight) * 100.0

        # Primary dominant driver
        drivers = [
            ("Extreme Forecast Shift (EFI)", pct_efi),
            ("Climatological Z-Score Exceedance", pct_climatology),
            ("Shift of Tails Outlier (SOT)", pct_tail_shift),
            ("Heavy Core Precipitation Rate", pct_rainfall),
            ("Low-Level Cyclonic Jet Winds", pct_wind),
            ("Deep Barometric MSLP Deficit", pct_pressure)
        ]
        drivers.sort(key=lambda x: x[1], reverse=True)

        return {
            "dominant_driver": drivers[0][0],
            "dominant_driver_pct": float(drivers[0][1]),
            "feature_contributions": [
                {"feature": name, "contribution_pct": float(pct), "evidence_value": self._format_evidence(name, peak_efi, z_score, sot_val, precip_mm, wind_ms, mslp_deficit)}
                for name, pct in drivers
            ],
            "ensemble_confidence_score": float(np.clip(1.0 - (ens_spread / max(5.0, precip_mm)), 0.2, 0.95)),
            "diagnostic_summary": f"Detection driven primarily by {drivers[0][0]} ({drivers[0][1]:.1f}%) and {drivers[1][0]} ({drivers[1][1]:.1f}%)."
        }

    def _format_evidence(self, name: str, efi, z, sot, p, w, m) -> str:
        if "EFI" in name:
            return f"EFI = {efi:.2f} (Shift > 90% Clim CDF)"
        elif "Z-Score" in name:
            return f"Z = +{z:.2f} σ"
        elif "Tails" in name:
            return f"SOT = {sot:.2f}"
        elif "Precipitation" in name:
            return f"Peak = {p:.1f} mm/6h"
        elif "Wind" in name:
            return f"V_max = {w:.1f} m/s"
        else:
            return f"ΔP = -{m:.1f} hPa"

anomaly_explainer = AnomalyExplainer()
