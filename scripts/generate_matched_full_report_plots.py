"""Generate the original report-figure set for later-day full-range fits only.

These figures are the controlled visual counterpart to the restricted >1 kHz
figures because both contain the same 259 later-day captures.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import generate_report_plots as report


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/processed/circuit_fits.sqlite"
OUTPUT = ROOT / "output/low_frequency_sensitivity/matched_full_report_figures"
DAY_ONE = "2026-09-30"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report.OUTPUT = OUTPUT
    report.DB = DB
    report.style()
    connection = sqlite3.connect(DB)
    connection.row_factory = sqlite3.Row
    config = report.selected_config(connection)
    rows = [row for row in report.load_rows(connection, config["id"])
            if row["measurement_date"] != DAY_ONE]
    connection.close()
    if len(rows) % 8:
        raise RuntimeError("Matched later-day results are not complete sets of eight models")
    report.add_delta_aicc(rows)
    report.plot_fit_quality(rows)
    report.plot_model_selection(rows)
    report.plot_diagnostics(rows)
    report.plot_residual_frequency(rows)
    report.parameter_trend_figure(rows, "cole")
    report.parameter_trend_figure(rows, "paper")
    report.plot_electrode_parameter_trends(rows)
    report.plot_electrode_effect(rows)
    report.write_summary(rows, config)
    print(f"Generated {len(rows)} matched full-range report records in {OUTPUT}")


if __name__ == "__main__":
    main()
