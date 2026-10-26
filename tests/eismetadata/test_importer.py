import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from openpyxl import Workbook

from eismetadata.importer import import_metadata, preview_metadata


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "fits.sqlite"
        connection = sqlite3.connect(self.database)
        connection.executescript("""
            CREATE TABLE captures (
                id INTEGER PRIMARY KEY, banana_id TEXT, measurement_date TEXT, is_current INTEGER
            );
            CREATE TABLE fit_configs (
                id INTEGER PRIMARY KEY, display_name TEXT
            );
            CREATE TABLE fits (
                id INTEGER PRIMARY KEY, capture_id INTEGER, config_id INTEGER,
                fruit_model TEXT, electrode_model TEXT, status TEXT, nrmse REAL,
                aicc REAL, aic REAL, bic REAL, objective REAL, parameters_json TEXT
            );
            INSERT INTO captures VALUES(1,'banana1','2026-09-30',1);
            INSERT INTO fit_configs VALUES(1,'test-config');
            INSERT INTO fits VALUES(
                1,1,1,'cole','none','complete',0.1,10,9,11,0.01,
                '{"r_inf": 100.0, "delta_r": 500.0, "r0": 600.0}'
            );
        """)
        connection.commit()
        connection.close()
        self.dimensions = self.root / "dimensions.xlsx"
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["Banana ID", "Arc Length(cm)", "Circumference(cm)", "Signal Path(cm)", "mass(g)"])
        sheet.append([1, 22.5, 12.4, 4.5, 129])
        workbook.save(self.dimensions)
        self.ripeness = self.root / "ripeness.xlsx"
        workbook = Workbook()
        sheet = workbook.active
        sheet.append([None, None, "Day"])
        sheet.append(["Banana id", 1, 2])
        sheet.append([1, "U", "R"])
        workbook.save(self.ripeness)

    def tearDown(self):
        self.temp.cleanup()

    def test_unmatched_ripeness_is_retained_and_reported(self):
        preview = preview_metadata(
            self.database, self.dimensions, self.ripeness, date(2026, 9, 30)
        )
        self.assertEqual(len(preview.unmatched_ripeness), 1)
        import_metadata(self.database, self.dimensions, self.ripeness, date(2026, 9, 30))
        connection = sqlite3.connect(self.database)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM banana_properties").fetchone()[0], 1)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM ripeness_observations").fetchone()[0], 2)
        rows = connection.execute(
            "SELECT measurement_date,has_eis_capture FROM ripeness_capture_status ORDER BY measurement_date"
        ).fetchall()
        self.assertEqual(rows, [("2026-09-30", 1), ("2026-10-01", 0)])
        joined = connection.execute(
            "SELECT delta_r,initial_mass_g,ripeness_class FROM ml_dataset_view"
        ).fetchone()
        self.assertEqual(joined, (500.0, 129.0, "unripe"))
        connection.close()

    def test_repeat_import_updates_without_duplicates(self):
        import_metadata(self.database, self.dimensions, self.ripeness, date(2026, 9, 30))
        import_metadata(self.database, self.dimensions, self.ripeness, date(2026, 9, 30))
        connection = sqlite3.connect(self.database)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM banana_properties").fetchone()[0], 1)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM ripeness_observations").fetchone()[0], 2)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM metadata_imports").fetchone()[0], 2)
        connection.close()


if __name__ == "__main__":
    unittest.main()
