import unittest

import numpy as np

from eisfit.io import polar_to_complex, wrap_phase_degrees
from eisfit.models import ShieldModel, cole_impedance, paper_impedance


class ModelTests(unittest.TestCase):
    def test_phase_wrapping(self):
        wrapped = wrap_phase_degrees(np.array([359.0, 180.0, -181.0, -10.0]))
        np.testing.assert_allclose(wrapped, [-1.0, -180.0, 179.0, -10.0])

    def test_polar_conversion(self):
        z = polar_to_complex(np.array([2.0]), np.array([270.0]))
        np.testing.assert_allclose(z.real, [0.0], atol=1e-12)
        np.testing.assert_allclose(z.imag, [-2.0], atol=1e-12)

    def test_shield_matches_nominal_test_board_response(self):
        z = ShieldModel().impedance(np.array([1000.0]))[0]
        self.assertAlmostEqual(z.real, 2000.0)
        self.assertAlmostEqual(z.imag, -10947.9, delta=2.0)

    def test_cole_limits(self):
        p = {"r_inf": 500.0, "delta_r": 7500.0, "tau_s": 1e-3, "alpha": 0.75}
        z = cole_impedance(np.array([1e-9, 1e12]), p)
        self.assertAlmostEqual(z[0].real, 8000.0, delta=1e-3)
        self.assertAlmostEqual(z[1].real, 500.0, delta=1.0)

    def test_paper_limits(self):
        p = {"r1": 200.0, "r2": 1800.0, "c1_f": 5e-9}
        z = paper_impedance(np.array([1e-9, 1e15]), p)
        self.assertAlmostEqual(z[0].real, 1800.0, delta=1e-6)
        self.assertAlmostEqual(z[1].real, 180.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()

