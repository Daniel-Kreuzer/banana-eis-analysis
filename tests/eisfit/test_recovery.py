import unittest

import numpy as np

from eisfit.cuckoo import CuckooSearchConfig
from eisfit.fit import FitConfig, fit_spectrum
from eisfit.models import evaluate_model


class SyntheticRecoveryTests(unittest.TestCase):
    def test_recovers_cole_model_without_electrode(self):
        frequency = np.geomspace(100.0, 180000.0, 60)
        expected = {
            "r_inf": 800.0,
            "delta_r": 8000.0,
            "tau_s": 2e-4,
            "alpha": 0.72,
        }
        impedance = evaluate_model(frequency, "cole", "none", expected)
        result = fit_spectrum(
            frequency,
            impedance,
            "cole",
            "none",
            FitConfig(
                restarts=2,
                seed=222,
                correct_shield=False,
                cuckoo=CuckooSearchConfig(population=35, iterations=700, step_scale=0.025),
            ),
        )
        self.assertLess(result.metrics["nrmse_complex"], 0.01)
        for name, value in expected.items():
            relative_error = abs(result.parameters[name] - value) / value
            self.assertLess(relative_error, 0.10, msg=f"{name} recovery failed")

    def test_recovers_paper_model_without_electrode(self):
        frequency = np.geomspace(100.0, 180000.0, 60)
        expected = {"r1": 250.0, "r2": 5000.0, "c1_f": 8e-8}
        impedance = evaluate_model(frequency, "paper", "none", expected)
        result = fit_spectrum(
            frequency,
            impedance,
            "paper",
            "none",
            FitConfig(
                restarts=2,
                seed=1234,
                correct_shield=False,
                cuckoo=CuckooSearchConfig(population=35, iterations=700, step_scale=0.025),
            ),
        )
        self.assertLess(result.metrics["nrmse_complex"], 0.015)
        for name, value in expected.items():
            relative_error = abs(result.parameters[name] - value) / value
            self.assertLess(relative_error, 0.15, msg=f"{name} recovery failed")


if __name__ == "__main__":
    unittest.main()
