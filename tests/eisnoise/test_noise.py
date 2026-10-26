import tempfile
import unittest
from pathlib import Path

import numpy as np

from eisnoise.simulation import (
    NoiseStudyConfig,
    _is_bound_hit,
    add_relative_complex_gaussian_noise,
    noise_profile,
    run_study,
    _task,
)
from eisnoise.engine import parameter_specs


class NoiseStudyTests(unittest.TestCase):
    def test_noise_profile_has_elevated_low_frequency_band(self):
        config = NoiseStudyConfig()
        frequency = np.array([100.0, 300.0, 450.0, 600.0, 1000.0])
        profile = noise_profile(frequency, config)
        np.testing.assert_allclose(profile[:2], 0.08)
        self.assertGreater(profile[2], 0.01)
        np.testing.assert_allclose(profile[3:], 0.01)

    def test_relative_complex_noise_is_reproducible(self):
        ideal = np.full(20_000, 1000.0 + 1000.0j)
        sigma = np.full(ideal.size, 0.05)
        first = add_relative_complex_gaussian_noise(ideal, sigma, np.random.default_rng(7))
        second = add_relative_complex_gaussian_noise(ideal, sigma, np.random.default_rng(7))
        np.testing.assert_allclose(first, second)
        rms_relative = np.sqrt(np.mean(np.abs(first - ideal) ** 2)) / np.abs(ideal[0])
        self.assertAlmostEqual(rms_relative, 0.05, delta=0.001)

    def test_bound_hit_uses_log_search_coordinate(self):
        frequency = np.geomspace(100, 180_000, 50)
        measured = np.full(50, 1000.0 + 0j)
        c1_spec = {spec.name: spec for spec in parameter_specs(
            "paper", "none", frequency, measured
        )}["c1_f"]
        self.assertEqual(_is_bound_hit(2.922e-9, c1_spec), 0)
        self.assertEqual(_is_bound_hit(c1_spec.lower, c1_spec), 1)

    def test_noisy_condition_is_generated_raw_then_deembedded(self):
        config = NoiseStudyConfig(
            replicates=1, points=8, population=5, iterations=1, restarts=1, workers=1
        )
        row = _task({
            "config": config.__dict__, "model": "cole",
            "condition": "relative_complex_gaussian", "replicate": 0,
        })
        self.assertEqual(row["noise_application_domain"], "raw_embedded")

    def test_smoke_study_writes_expected_rows(self):
        config = NoiseStudyConfig(
            replicates=2, points=8, population=5, iterations=1, restarts=1, workers=1
        )
        with tempfile.TemporaryDirectory() as directory:
            replicates, summary = run_study(config, Path(directory))
            self.assertTrue(replicates.exists())
            self.assertTrue(summary.exists())
            self.assertEqual(len(replicates.read_text(encoding="utf-8").splitlines()), 9)
            self.assertGreater(len(summary.read_text(encoding="utf-8").splitlines()), 10)


if __name__ == "__main__":
    unittest.main()
