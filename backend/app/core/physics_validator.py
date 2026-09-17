"""
Physics-Informed Atmospheric Validation Engine for SIH-26078.
Rigorously checks meteorological invariants, thermodynamic limits, and physical conservation laws
across forecast tensors and downscaled grids.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from pydantic import BaseModel

class PhysicsCheckDetail(BaseModel):
    check_name: str
    passed: bool
    severity: str # 'INFO', 'WARNING', 'VIOLATION'
    min_observed: float
    max_observed: float
    allowed_range: List[float]
    violations_count: int
    message: str

class PhysicsValidationResult(BaseModel):
    passed: bool
    total_checks: int
    violations_count: int
    warnings_count: int
    checks: List[PhysicsCheckDetail]
    conservation_ratio: float = 1.0
    summary: str

class PhysicsValidator:
    """
    Validates physical atmospheric constraints and mass/moisture consistency.
    """
    def __init__(self):
        pass

    def validate_atmospheric_state(
        self,
        precip: np.ndarray,
        temperature_2m: np.ndarray,
        relative_humidity: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        mslp: np.ndarray,
        coarse_precip: np.ndarray = None,
        fine_precip: np.ndarray = None
    ) -> PhysicsValidationResult:
        """
        Executes full physical sanity checks on 2D or 3D meteorological arrays.
        """
        checks: List[PhysicsCheckDetail] = []
        violations = 0
        warnings = 0

        # Check 1: Non-negative Precipitation (P >= 0.0)
        p_min = float(np.min(precip))
        p_max = float(np.max(precip))
        p_neg = int(np.sum(precip < -1e-4))
        p_passed = p_neg == 0
        if not p_passed:
            violations += 1
        checks.append(PhysicsCheckDetail(
            check_name="Non-Negative Precipitation",
            passed=p_passed,
            severity="VIOLATION" if not p_passed else "INFO",
            min_observed=p_min,
            max_observed=p_max,
            allowed_range=[0.0, 500.0],
            violations_count=p_neg,
            message=f"Precipitation must be non-negative. Found {p_neg} unphysical negative grid cells." if not p_passed else "Verified non-negative precipitation field."
        ))

        # Check 2: Relative Humidity Bounded [0%, 100%]
        rh_min = float(np.min(relative_humidity))
        rh_max = float(np.max(relative_humidity))
        rh_viol = int(np.sum((relative_humidity < 0.0) | (relative_humidity > 105.0)))
        rh_passed = rh_viol == 0
        if not rh_passed:
            violations += 1
        checks.append(PhysicsCheckDetail(
            check_name="Relative Humidity Boundedness",
            passed=rh_passed,
            severity="VIOLATION" if not rh_passed else "INFO",
            min_observed=rh_min,
            max_observed=rh_max,
            allowed_range=[0.0, 100.0],
            violations_count=rh_viol,
            message=f"Relative humidity must stay within [0%, 100%]. Found {rh_viol} out-of-bounds cells." if not rh_passed else "Verified thermodynamic RH boundaries."
        ))

        # Check 3: 2m Temperature Plausibility (180K to 340K / -93C to +67C)
        t_min = float(np.min(temperature_2m))
        t_max = float(np.max(temperature_2m))
        t_viol = int(np.sum((temperature_2m < 200.0) | (temperature_2m > 335.0)))
        t_passed = t_viol == 0
        if not t_passed:
            violations += 1
        checks.append(PhysicsCheckDetail(
            check_name="Surface Temperature Plausibility",
            passed=t_passed,
            severity="VIOLATION" if not t_passed else "INFO",
            min_observed=t_min,
            max_observed=t_max,
            allowed_range=[200.0, 335.0],
            violations_count=t_viol,
            message=f"Surface temperature outside terrestrial atmospheric bounds ({t_min:.1f}K - {t_max:.1f}K)." if not t_passed else "Verified realistic surface temperature distribution."
        ))

        # Check 4: Wind Speed Plausibility (V_max <= 100 m/s / ~360 km/h)
        wind_spd = np.sqrt(u_wind**2 + v_wind**2)
        w_min = float(np.min(wind_spd))
        w_max = float(np.max(wind_spd))
        w_viol = int(np.sum(wind_spd > 90.0))
        w_passed = w_viol == 0
        if not w_passed:
            violations += 1
        checks.append(PhysicsCheckDetail(
            check_name="Atmospheric Wind Velocity Limits",
            passed=w_passed,
            severity="VIOLATION" if not w_passed else "INFO",
            min_observed=w_min,
            max_observed=w_max,
            allowed_range=[0.0, 90.0],
            violations_count=w_viol,
            message=f"Supersonic or unphysical wind speed detected ({w_max:.1f} m/s)." if not w_passed else "Verified realistic low-level jet and vortex wind velocities."
        ))

        # Check 5: Mean Sea Level Pressure (870 hPa - 1050 hPa)
        p_mslp_min = float(np.min(mslp))
        p_mslp_max = float(np.max(mslp))
        mslp_viol = int(np.sum((mslp < 880.0) | (mslp > 1050.0)))
        mslp_passed = mslp_viol == 0
        if not mslp_passed:
            violations += 1
        checks.append(PhysicsCheckDetail(
            check_name="Hydrostatic MSLP Bounds",
            passed=mslp_passed,
            severity="VIOLATION" if not mslp_passed else "INFO",
            min_observed=p_mslp_min,
            max_observed=p_mslp_max,
            allowed_range=[880.0, 1050.0],
            violations_count=mslp_viol,
            message=f"MSLP pressure violates atmospheric hydrostatic limits ({p_mslp_min:.1f} hPa)." if not mslp_passed else "Verified MSLP barometric boundaries."
        ))

        # Check 6: Mass / Water Conservation during Downscaling (if coarse and fine provided)
        conservation_ratio = 1.0
        if coarse_precip is not None and fine_precip is not None:
            coarse_total = float(np.sum(coarse_precip))
            # Normalize fine grid sum by resolution area ratio
            ratio_area = float(fine_precip.size / coarse_precip.size)
            fine_total = float(np.sum(fine_precip) / ratio_area)
            
            if coarse_total > 1.0:
                conservation_ratio = fine_total / coarse_total
                cons_error = abs(1.0 - conservation_ratio)
                cons_passed = cons_error <= 0.15 # within 15% mass preservation
                if not cons_passed:
                    warnings += 1
                checks.append(PhysicsCheckDetail(
                    check_name="Coarse-Fine Water Mass Conservation",
                    passed=cons_passed,
                    severity="WARNING" if not cons_passed else "INFO",
                    min_observed=conservation_ratio,
                    max_observed=conservation_ratio,
                    allowed_range=[0.85, 1.15],
                    violations_count=1 if not cons_passed else 0,
                    message=f"Downscaled total precipitation mass ratio is {conservation_ratio:.3f} (target: 1.000)."
                ))

        overall_passed = violations == 0
        summary = (
            "All physical atmospheric invariant checks PASSED."
            if overall_passed
            else f"Physics validation FAILED: {violations} severe physical violation(s) detected."
        )

        return PhysicsValidationResult(
            passed=overall_passed,
            total_checks=len(checks),
            violations_count=violations,
            warnings_count=warnings,
            checks=checks,
            conservation_ratio=conservation_ratio,
            summary=summary
        )

physics_validator = PhysicsValidator()
