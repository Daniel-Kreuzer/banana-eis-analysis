import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

from eisml.config import TrainingConfig
from eisml.data import load_dataset, validate_common_rows


class ConfigTests(unittest.TestCase):
    def test_fingerprint_is_stable_and_changes_with_configuration(self):
        self.assertEqual(TrainingConfig().fingerprint, TrainingConfig().fingerprint)
        self.assertNotEqual(TrainingConfig().fingerprint, TrainingConfig(random_seed=9).fingerprint)


class DataTests(unittest.TestCase):
    def test_manifest_controls_features_and_order(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            manifest = {"datasets": {"sample": {
                "training_features": ["a", "b"], "target": "ripeness_class",
                "group_column": "banana_id", "audit_only": ["ignored"],
            }}}
            (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with closing(sqlite3.connect(root / "ml_datasets.sqlite")) as connection:
                connection.execute("CREATE TABLE sample (banana_id TEXT, measurement_date TEXT, "
                                   "capture_id INTEGER, ripeness_class TEXT, a REAL, b REAL, ignored REAL)")
                connection.execute("INSERT INTO sample VALUES ('banana1','2026-01-01',1,'unripe',2,3,999)")
                connection.commit()
            dataset = load_dataset(root, "sample")
            self.assertEqual(dataset.features, ("a", "b"))
            np.testing.assert_array_equal(dataset.x, [[2.0, 3.0]])
            validate_common_rows(dataset, dataset)


if __name__ == "__main__":
    unittest.main()
