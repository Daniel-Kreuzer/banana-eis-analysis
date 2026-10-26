"""Generate the original report-figure set for the >1 kHz restricted fits.

This deliberately reuses the plotting functions from generate_report_plots.py
so that the restricted outputs have the same metrics, layout and styling as the
existing whole-spectrum report figures.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

import generate_report_plots as report


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "output/low_frequency_sensitivity/restricted_frequency_fits.sqlite"
OUTPUT = ROOT / "output/low_frequency_sensitivity/restricted_report_figures"


def rms_nrmse(measured: np.ndarray, fitted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(fitted - measured) ** 2)) /
                 max(np.sqrt(np.mean(np.abs(measured) ** 2)), 1e-15))


def load_rows(connection: sqlite3.Connection, config: dict) -> list[dict]:
    rows: list[dict] = []
    query = """
        SELECT * FROM restricted_fits
        WHERE status='complete'
        ORDER BY measurement_date, banana_id, fruit_model, electrode_model
    """
    shield = report.ShieldModel(config["fixed_resistance_ohm"], config["fixed_capacitance_f"])
    for source in connection.execute(query):
        row = dict(source)
        frequency = np.asarray(json.loads(row["frequency_json"]), dtype=float)
        corrected = (np.asarray(json.loads(row["measured_corrected_real_json"]), dtype=float) +
                     1j * np.asarray(json.loads(row["measured_corrected_imag_json"]), dtype=float))
        predicted = (np.asarray(json.loads(row["predicted_real_json"]), dtype=float) +
                     1j * np.asarray(json.loads(row["predicted_imag_json"]), dtype=float))
        fixed = shield.impedance(frequency) if config["correct_fixed_path"] else np.zeros_like(corrected)
        raw = corrected + fixed
        complete = predicted + fixed
        parameters = json.loads(row["parameters_json"])
        hits = json.loads(row["boundary_hits_json"])
        residual = predicted - corrected
        magnitude_nrmse = float(
            np.sqrt(np.mean((np.abs(predicted) - np.abs(corrected)) ** 2)) /
            max(np.sqrt(np.mean(np.abs(corrected) ** 2)), 1e-15)
        )
        phase_error = np.angle(np.exp(1j * (np.angle(predicted) - np.angle(corrected))))
        phase_rmse_deg = float(np.rad2deg(np.sqrt(np.mean(phase_error ** 2))))
        objectives = np.asarray(json.loads(row["run_objectives_json"]), dtype=float)
        restart_spread = float((objectives.max() - objectives.min()) /
                               max(abs(objectives.min()), 1e-15))
        real_scale = max(float(np.sqrt(np.mean(corrected.real ** 2))), 1e-15)
        imag_scale = max(float(np.sqrt(np.mean(corrected.imag ** 2))), 1e-15)
        row.update({
            "frequency": frequency,
            "corrected": corrected,
            "predicted": predicted,
            "raw": raw,
            "complete": complete,
            "parameters": parameters,
            "nrmse": float(row["fitting_domain_nrmse"]),
            "raw_nrmse": rms_nrmse(raw, complete),
            "magnitude_nrmse": magnitude_nrmse,
            "phase_rmse_deg": phase_rmse_deg,
            "relative_residual": np.abs(residual) / np.maximum(np.abs(corrected), 1e-15) * 100,
            "real_bias_pct": float(np.mean(residual.real) / real_scale * 100),
            "imag_bias_pct": float(np.mean(residual.imag) / imag_scale * 100),
            "boundary_hit": any(v["lower"] or v["upper"] for v in hits.values()),
            "boundary_parameters": [name for name, value in hits.items()
                                    if value["lower"] or value["upper"]],
            "restart_spread": restart_spread,
        })
        rows.append(row)
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report.OUTPUT = OUTPUT
    report.DB = DB
    report.style()
    connection = sqlite3.connect(DB)
    connection.row_factory = sqlite3.Row
    metadata = dict(connection.execute("SELECT key,value FROM study_metadata"))
    config = json.loads(metadata["source_config_json"])
    rows = load_rows(connection, config)
    connection.close()
    if len(rows) % 8:
        raise RuntimeError("Restricted results are not a complete set of eight models per capture")
    report.add_delta_aicc(rows)
    report.plot_fit_quality(rows)
    report.plot_model_selection(rows)
    report.plot_diagnostics(rows)
    report.plot_residual_frequency(rows)
    report.parameter_trend_figure(rows, "cole")
    report.parameter_trend_figure(rows, "paper")
    report.plot_electrode_parameter_trends(rows)
    report.plot_electrode_effect(rows)
    report.write_summary(rows, {"display_name": metadata["study_name"]})
    print(f"Generated {len(rows)} restricted-fit report records in {OUTPUT}")


if __name__ == "__main__":
    main()
