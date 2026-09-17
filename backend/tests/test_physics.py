"""
Unit tests for Physics-Informed Atmospheric Validation Engine.
"""

import unittest
import numpy as np
from backend.app.core.physics_validator import physics_validator

class TestPhysicsValidator(unittest.TestCase):
    def setUp(self):
        self.shape = (10, 10)
        self.precip = np.full(self.shape, 25.0, dtype=np.float32)
        self.t2m = np.full(self.shape, 300.0, dtype=np.float32)
        self.rh = np.full(self.shape, 75.0, dtype=np.float32)
        self.u = np.full(self.shape, 10.0, dtype=np.float32)
        self.v = np.full(self.shape, 8.0, dtype=np.float32)
        self.mslp = np.full(self.shape, 1005.0, dtype=np.float32)

    def test_clean_atmospheric_state(self):
        res = physics_validator.validate_atmospheric_state(
            precip=self.precip,
            temperature_2m=self.t2m,
            relative_humidity=self.rh,
            u_wind=self.u,
            v_wind=self.v,
            mslp=self.mslp
        )
        self.assertTrue(res.passed, "Clean meteorological fields must pass all physics checks")
        self.assertEqual(res.violations_count, 0)

    def test_negative_precipitation_violation(self):
        bad_p = self.precip.copy()
        bad_p[2, 3] = -12.5 # Negative rain violation
        
        res = physics_validator.validate_atmospheric_state(
            precip=bad_p,
            temperature_2m=self.t2m,
            relative_humidity=self.rh,
            u_wind=self.u,
            v_wind=self.v,
            mslp=self.mslp
        )
        self.assertFalse(res.passed, "Negative precipitation must fail validation")
        self.assertGreater(res.violations_count, 0)
        p_check = next(c for c in res.checks if c.check_name == "Non-Negative Precipitation")
        self.assertFalse(p_check.passed)

    def test_unphysical_humidity_violation(self):
        bad_rh = self.rh.copy()
        bad_rh[4, 5] = 145.0 # Exceeds 100% saturation limit
        
        res = physics_validator.validate_atmospheric_state(
            precip=self.precip,
            temperature_2m=self.t2m,
            relative_humidity=bad_rh,
            u_wind=self.u,
            v_wind=self.v,
            mslp=self.mslp
        )
        self.assertFalse(res.passed)
        rh_check = next(c for c in res.checks if c.check_name == "Relative Humidity Boundedness")
        self.assertFalse(rh_check.passed)

    def test_mass_conservation_check(self):
        coarse_p = np.full((10, 10), 20.0, dtype=np.float32)
        fine_p = np.full((50, 50), 20.0, dtype=np.float32) # Equal mass
        
        res = physics_validator.validate_atmospheric_state(
            precip=self.precip,
            temperature_2m=self.t2m,
            relative_humidity=self.rh,
            u_wind=self.u,
            v_wind=self.v,
            mslp=self.mslp,
            coarse_precip=coarse_p,
            fine_precip=fine_p
        )
        self.assertTrue(res.passed)
        self.assertAlmostEqual(res.conservation_ratio, 1.0, delta=0.05)

if __name__ == "__main__":
    unittest.main()
