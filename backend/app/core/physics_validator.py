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

    def validate_field(
        self,
        field: np.ndarray,
        variable_name: str = "precipitation",
        grid_resolution_deg: float = 0.05
    ) -> Dict[str, Any]:
        """
        Validates an individual meteorological array (e.g. 5km downscaled field).
        """
        f = np.asarray(field, dtype=np.float64)
        min_v = float(np.min(f))
        max_v = float(np.max(f))
        mean_v = float(np.mean(f))
        
        status = "PASS"
        issues = []
        
        if variable_name == "precipitation":
            if min_v < -1e-4:
                status = "FAIL"
                issues.append(f"Negative precipitation detected: min {min_v:.4f}")
            if max_v > 500.0:
                status = "WARN"
                issues.append(f"Extreme precipitation amplitude: max {max_v:.1f} mm/6h")
        elif variable_name == "mslp":
            if min_v < 870.0 or max_v > 1050.0:
                status = "FAIL"
                issues.append(f"MSLP out of hydrostatic bounds: [{min_v:.1f}, {max_v:.1f}] hPa")

        return {
            "status": status,
            "variable": variable_name,
            "min_value": min_v,
            "max_value": max_v,
            "mean_value": mean_v,
            "grid_resolution_deg": grid_resolution_deg,
            "issues": issues,
            "passed": (status in ["PASS", "WARN"])
        }

    def compute_physics_loss_breakdown(
        self,
        precip_pred: np.ndarray,
        precip_target: np.ndarray,
        u_wind: Any = None,
        v_wind: Any = None,
        mslp: Any = None,
        temperature_2m: Any = None
    ) -> Dict[str, float]:
        """
        Computes explicit Data Loss vs. Physics Penalty Decomposition.
        Terms:
        - data_loss: MSE against target
        - moisture_flux_loss: Divergence of horizontal moisture transport
        - geostrophic_loss: Residual of geostrophic wind balance
        - boundary_penalty: Penalty for negative precipitation or out-of-bound thermodynamic state
        - total_physics_loss: Weighted sum of physical violations
        - total_combined_loss: data_loss + lambda * total_physics_loss
        """
        p_pred = np.asarray(precip_pred, dtype=np.float64)
        p_tgt = np.asarray(precip_target, dtype=np.float64)

        # 1. Data Loss (MSE)
        data_mse = float(np.mean((p_pred - p_tgt) ** 2))

        # 2. Non-negativity and thermodynamic bound penalty
        neg_penalty = float(np.mean(np.maximum(0.0, -p_pred) ** 2) * 100.0)
        upper_penalty = float(np.mean(np.maximum(0.0, p_pred - 450.0) ** 2) * 10.0)
        bound_penalty = neg_penalty + upper_penalty

        # 3. Moisture Flux Divergence Loss (if wind provided)
        moisture_loss = 0.0
        if u_wind is not None and v_wind is not None:
            u = np.asarray(u_wind, dtype=np.float64)
            v = np.asarray(v_wind, dtype=np.float64)
            if u.shape == p_pred.shape and v.shape == p_pred.shape:
                q = np.sqrt(np.maximum(0.0, p_pred))
                qu = q * u
                qv = q * v
                dqu_dx = np.gradient(qu, axis=-1)
                dqv_dy = np.gradient(qv, axis=-2)
                div_flux = dqu_dx + dqv_dy
                moisture_loss = float(np.mean((p_pred * 0.1 - np.maximum(0.0, -div_flux)) ** 2) * 0.05)

        # 4. Geostrophic balance loss (if mslp and wind provided)
        geostrophic_loss = 0.0
        if mslp is not None and u_wind is not None and v_wind is not None:
            m = np.asarray(mslp, dtype=np.float64)
            u = np.asarray(u_wind, dtype=np.float64)
            v = np.asarray(v_wind, dtype=np.float64)
            if m.shape == u.shape == v.shape:
                dp_dx = np.gradient(m * 100.0, axis=-1)
                dp_dy = np.gradient(m * 100.0, axis=-2)
                f_rho = 5e-5 * 1.2
                u_g = -dp_dy / (f_rho * 1e5 + 1e-4)
                v_g = dp_dx / (f_rho * 1e5 + 1e-4)
                geo_residual = (u - u_g) ** 2 + (v - v_g) ** 2
                geostrophic_loss = float(np.mean(np.clip(geo_residual, 0.0, 100.0)) * 0.01)

        total_physics = bound_penalty + moisture_loss + geostrophic_loss
        lambda_phys = 0.15
        total_loss = data_mse + lambda_phys * total_physics

        return {
            "data_loss_mse": data_mse,
            "data_loss_mae": float(np.mean(np.abs(p_pred - p_tgt))),
            "boundary_penalty": bound_penalty,
            "moisture_flux_loss": moisture_loss,
            "geostrophic_loss": geostrophic_loss,
            "total_physics_penalty": total_physics,
            "lambda_physics_weight": lambda_phys,
            "total_combined_loss": total_loss,
            "physics_data_ratio": float(total_physics / (data_mse + 1e-6))
        }


physics_validator = PhysicsValidator()


