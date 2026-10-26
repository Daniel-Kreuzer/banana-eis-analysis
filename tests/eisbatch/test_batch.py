import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eisbatch.configuration import BatchConfig
from eisbatch.database import Database
from eisbatch.discovery import Capture, parse_filename
from eisbatch.engine import Spectrum
from eisbatch.exporting import export_results
from eisbatch.gui import _between, _display_values, _domain_nrmse, _trend_scale
from eisbatch.pipeline import _fit_job, automatic_workers, run_jobs, selected_indices


class BatchTests(unittest.TestCase):
    def test_sensorpal_filename_metadata(self):
        banana, date = parse_filename(Path(
            "banana1_SensorPal_ExportData_Friday, 18 September 2026_14_19_5.xlsx"
        ))
        self.assertEqual(banana, "banana1")
        self.assertEqual(date, "2026-09-18")

    def test_configuration_identity_changes_with_experiment(self):
        base = BatchConfig()
        self.assertEqual(base.identity(), BatchConfig().identity())
        self.assertNotEqual(base.identity(), BatchConfig(iterations=500).identity())
        self.assertNotEqual(base.identity(), BatchConfig(frequency_points=40).identity())

    def test_frequency_selection_is_deterministic_and_includes_ends(self):
        indices = selected_indices(60, 40)
        self.assertEqual(len(indices), 40)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 59)
        np.testing.assert_array_equal(indices, selected_indices(60, 40))

    def test_date_range_and_full_model_display(self):
        self.assertTrue(_between("2026-09-20", "2026-09-18", "2026-09-25"))
        self.assertFalse(_between("2026-09-30", "2026-09-18", "2026-09-25"))
        frequency = np.array([1000.0, 2000.0])
        predicted = np.array([1000 - 500j, 900 - 300j])
        from eisbatch.engine import ShieldModel
        fixed = ShieldModel().impedance(frequency)
        raw = predicted + fixed
        config = BatchConfig().payload()
        row = {
            "frequency_json": json.dumps(frequency.tolist()),
            "real_json": json.dumps(raw.real.tolist()), "imag_json": json.dumps(raw.imag.tolist()),
            "selected_indices_json": json.dumps([0, 1]),
            "predicted_real_json": json.dumps(predicted.real.tolist()),
            "predicted_imag_json": json.dumps(predicted.imag.tolist()),
            "config_json": json.dumps(config),
        }
        _, shown_raw, shown_model = _display_values(row, False)
        np.testing.assert_allclose(shown_raw, raw)
        np.testing.assert_allclose(shown_model, raw)
        _, corrected, fitted = _display_values(row, True)
        np.testing.assert_allclose(corrected, predicted)
        np.testing.assert_allclose(fitted, predicted)

    def test_domain_nrmse_distinguishes_corrected_and_raw_signal(self):
        frequency = np.array([100.0, 200.0, 1000.0])
        fitted = np.full(3, 1000.0 - 500.0j)
        corrected = fitted + np.array([500.0 + 100.0j, -100.0 + 50.0j, 20.0 - 10.0j])
        from eisbatch.engine import ShieldModel
        fixed = ShieldModel().impedance(frequency)
        raw = corrected + fixed
        deembedded_nrmse = np.sqrt(np.mean(np.abs(fitted - corrected) ** 2)) / np.sqrt(
            np.mean(np.abs(corrected) ** 2)
        )
        row = {
            "nrmse": float(deembedded_nrmse),
            "frequency_json": json.dumps(frequency.tolist()),
            "real_json": json.dumps(raw.real.tolist()),
            "imag_json": json.dumps(raw.imag.tolist()),
            "selected_indices_json": json.dumps([0, 1, 2]),
            "predicted_real_json": json.dumps(fitted.real.tolist()),
            "predicted_imag_json": json.dumps(fitted.imag.tolist()),
            "config_json": json.dumps(BatchConfig().payload()),
        }
        shown_deembedded, shown_raw = _domain_nrmse(row)
        self.assertAlmostEqual(shown_deembedded, deembedded_nrmse)
        self.assertLess(shown_raw, shown_deembedded)

    def test_automatic_workers_is_conservative(self):
        self.assertGreaterEqual(automatic_workers(), 1)
        self.assertLessEqual(automatic_workers(), 4)

    def test_parameter_trend_uses_log_scale_for_large_positive_range(self):
        scale, ratio = _trend_scale([6881.9, 629502.3, 1001.8, 723.8398])
        self.assertEqual(scale, "log")
        self.assertGreater(ratio, 100)
        self.assertEqual(_trend_scale([700.0, 900.0, 1100.0])[0], "linear")
        self.assertEqual(_trend_scale([0.0, 1000.0])[0], "linear")

    def test_worker_fits_serialized_capture(self):
        frequency = np.geomspace(1000, 100000, 10)
        impedance = np.full(10, 1000.0 + 0j)
        capture = {
            "id": 7,
            "frequency_json": json.dumps(frequency.tolist()),
            "real_json": json.dumps(impedance.real.tolist()),
            "imag_json": json.dumps(impedance.imag.tolist()),
        }
        config = BatchConfig(population=5, iterations=1, restarts=1,
                             correct_fixed_path=False, frequency_points=6)
        payload = _fit_job(capture, config.payload(), "paper", "none")
        self.assertEqual(payload["capture_id"], 7)
        self.assertEqual(len(payload["selected_indices"]), 6)
        self.assertIn("nrmse_complex", payload["result"]["metrics"])

    def test_same_fit_slot_is_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spectrum = Spectrum(np.array([1, 2, 3, 4], dtype=float), np.ones(4, dtype=complex))
            capture = Capture(root / "banana1_test.xlsx", "abc", "banana1", "2026-09-18",
                              spectrum, "ready", "")
            database = Database(root / "test.sqlite")
            capture_id = database.add_capture(capture)
            config_id = database.add_config(BatchConfig(iterations=1, population=5, restarts=1))
            result = {
                "objective": 1.0,
                "metrics": {"nrmse_complex": .1, "aicc": 2.0, "aic": 1.0, "bic": 3.0},
                "parameters": {"r1": 5.0}, "run_objectives": [1.0],
                "predicted_real_ohm": [1, 1, 1, 1], "predicted_imag_ohm": [0, 0, 0, 0],
                "evaluations": 10, "best_seed": 1,
            }
            database.save_result(capture_id, config_id, "paper", "none", result, [0, 1, 2, 3])
            result["objective"] = 0.5
            database.save_result(capture_id, config_id, "paper", "none", result, [0, 1, 2, 3])
            rows = database.fit_rows(config_id)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["objective"], 0.5)
            database.close()

    def test_parallel_pipeline_persists_all_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frequency = np.geomspace(1000, 100000, 8)
            spectrum = Spectrum(frequency, np.full(8, 1000.0 + 0j))
            capture = Capture(root / "banana1_test.xlsx", "parallel", "banana1", "2026-09-18",
                              spectrum, "ready", "")
            database = Database(root / "parallel.sqlite")
            database.add_capture(capture)
            config = BatchConfig(population=5, iterations=1, restarts=1,
                                 correct_fixed_path=False, frequency_points=6)
            succeeded, failed = run_jobs(database, config)
            self.assertEqual((succeeded, failed), (8, 0))
            rows = database.fit_rows(database.add_config(config))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["status"] == "complete" for row in rows))
            long_path, wide_path = export_results(database, root / "exports")
            self.assertGreater(len(long_path.read_text(encoding="utf-8").splitlines()), 8)
            self.assertEqual(len(wide_path.read_text(encoding="utf-8").splitlines()), 2)
            database.close()


if __name__ == "__main__":
    unittest.main()
