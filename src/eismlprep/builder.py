"""Build auditable circuit-parameter and raw-spectrum ML datasets."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from eisfit.models import ShieldModel, parameter_specs


TARGET_CONFIGURATION = "P30-I800-R5-Fall-6447b2b7"
FRUIT_MODELS = ("cole", "paper")
ELECTRODE_MODELS = ("none", "capacitor", "warburg", "randles_cpe")
PARAMETER_NAMES = (
    "r_inf", "delta_r", "tau_s", "alpha", "r0", "r1", "r2", "c1_f",
    "c_e_f", "sigma", "r_e", "q_e", "n_e",
)
OPTIMIZED_PARAMETER_NAMES = tuple(name for name in PARAMETER_NAMES if name != "r0")
PHYSICAL_ALL = ("initial_mass_g", "signal_path_cm", "arc_length_cm", "circumference_cm")
PHYSICAL_PRACTICAL = ("initial_mass_g", "signal_path_cm")
IDENTIFIERS = ("banana_id", "measurement_date", "study_day")
TARGET = "ripeness_class"
FIT_AUDIT = (
    "fit_id", "capture_id", "configuration", "deembedded_nrmse", "aicc", "objective",
    "boundary_hit", "boundary_count", "boundary_parameters",
)


@dataclass(frozen=True)
class BuildSummary:
    output_directory: Path
    sqlite_path: Path
    manifest_path: Path
    circuit_rows: int
    raw_rows: int
    files: tuple[Path, ...]
    boundary_counts: dict[str, int]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def _normalized_position(value: float, lower: float, upper: float, scale: str) -> float:
    if scale == "linear":
        return (value - lower) / (upper - lower)
    return (np.log10(value) - np.log10(lower)) / (np.log10(upper) - np.log10(lower))


def _boundary_fields(row: sqlite3.Row, parameters: dict[str, float], tolerance: float) -> dict:
    frequency = _loads(row["frequency_json"])
    real = _loads(row["real_json"])
    imag = _loads(row["imag_json"])
    indices = np.asarray(json.loads(row["selected_indices_json"]), dtype=int)
    selected_frequency = frequency[indices]
    measured = real[indices] + 1j * imag[indices]
    config = json.loads(row["config_json"])
    if config["correct_fixed_path"]:
        fixed = ShieldModel(
            config["fixed_resistance_ohm"], config["fixed_capacitance_f"]
        ).impedance(selected_frequency)
        measured = measured - fixed
    specs = parameter_specs(
        row["fruit_model"], row["electrode_model"], selected_frequency, measured
    )
    flags = {f"{name}_at_boundary": False for name in OPTIMIZED_PARAMETER_NAMES}
    affected: list[str] = []
    for spec in specs:
        value = float(parameters[spec.name])
        position = _normalized_position(value, spec.lower, spec.upper, spec.scale)
        hit = bool(position <= tolerance or position >= 1.0 - tolerance)
        flags[f"{spec.name}_at_boundary"] = hit
        if hit:
            affected.append(spec.name)
    return {
        "boundary_hit": bool(affected),
        "boundary_count": len(affected),
        "boundary_parameters": ",".join(affected),
        **flags,
    }


def _circuit_rows(connection: sqlite3.Connection, tolerance: float) -> list[dict]:
    rows = connection.execute(
        """SELECT
               f.id AS fit_id,f.capture_id,f.fruit_model,f.electrode_model,
               f.nrmse,f.aicc,f.objective,f.parameters_json,f.selected_indices_json,
               c.banana_id,c.measurement_date,c.frequency_json,c.real_json,c.imag_json,
               fc.display_name AS configuration,fc.config_json,
               bp.initial_mass_g,bp.signal_path_cm,bp.arc_length_cm,bp.circumference_cm,
               ro.study_day,ro.ripeness_class
           FROM fits AS f
           JOIN captures AS c ON c.id=f.capture_id AND c.is_current=1
           JOIN fit_configs AS fc ON fc.id=f.config_id
           JOIN banana_properties AS bp ON bp.banana_id=c.banana_id
           JOIN ripeness_observations AS ro
             ON ro.banana_id=c.banana_id AND ro.measurement_date=c.measurement_date
           WHERE f.status='complete' AND fc.display_name=?
           ORDER BY f.fruit_model,f.electrode_model,c.banana_id,c.measurement_date""",
        (TARGET_CONFIGURATION,),
    ).fetchall()
    result: list[dict] = []
    for row in rows:
        parameters = {key: float(value) for key, value in json.loads(row["parameters_json"]).items()}
        if not all(np.isfinite(value) for value in parameters.values()):
            continue
        record = {
            "fit_id": row["fit_id"],
            "capture_id": row["capture_id"],
            "banana_id": row["banana_id"],
            "measurement_date": row["measurement_date"],
            "study_day": row["study_day"],
            "ripeness_class": row["ripeness_class"],
            "configuration": row["configuration"],
            "fruit_model": row["fruit_model"],
            "electrode_model": row["electrode_model"],
            "initial_mass_g": row["initial_mass_g"],
            "signal_path_cm": row["signal_path_cm"],
            "arc_length_cm": row["arc_length_cm"],
            "circumference_cm": row["circumference_cm"],
            "deembedded_nrmse": row["nrmse"],
            "aicc": row["aicc"],
            "objective": row["objective"],
            **{name: parameters.get(name) for name in PARAMETER_NAMES},
        }
        record.update(_boundary_fields(row, parameters, tolerance))
        result.append(record)
    return result


def _raw_rows(connection: sqlite3.Connection, config: dict) -> tuple[list[dict], np.ndarray]:
    rows = connection.execute(
        """SELECT
               c.id AS capture_id,c.banana_id,c.measurement_date,
               c.frequency_json,c.real_json,c.imag_json,
               bp.initial_mass_g,bp.signal_path_cm,bp.arc_length_cm,bp.circumference_cm,
               ro.study_day,ro.ripeness_class
           FROM captures AS c
           JOIN banana_properties AS bp ON bp.banana_id=c.banana_id
           JOIN ripeness_observations AS ro
             ON ro.banana_id=c.banana_id AND ro.measurement_date=c.measurement_date
           WHERE c.is_current=1
           ORDER BY c.banana_id,c.measurement_date"""
    ).fetchall()
    if not rows:
        return [], np.array([])
    later_grids = [
        _loads(row["frequency_json"])
        for row in rows
        if float(_loads(row["frequency_json"]).min()) < 1000.0
    ]
    if not later_grids:
        raise ValueError("no later-day 100 Hz sweep was found for the raw-data grid")
    grid = later_grids[0][later_grids[0] > 1000.0]
    if len(grid) != 34:
        raise ValueError(f"expected 34 later-day frequencies above 1 kHz; found {len(grid)}")
    for frequency in later_grids[1:]:
        candidate = frequency[frequency > 1000.0]
        if len(candidate) != len(grid) or not np.allclose(candidate, grid, rtol=1e-9, atol=1e-9):
            raise ValueError("later-day captures do not share the same frequencies above 1 kHz")
    log_grid = np.log10(grid)
    result: list[dict] = []
    for row in rows:
        frequency = _loads(row["frequency_json"])
        order = np.argsort(frequency)
        log_frequency = np.log10(frequency[order])
        real = _loads(row["real_json"])[order]
        imag = _loads(row["imag_json"])[order]
        impedance = real + 1j * imag
        if config["correct_fixed_path"]:
            impedance = impedance - ShieldModel(
                config["fixed_resistance_ohm"], config["fixed_capacitance_f"]
            ).impedance(frequency[order])
        real, imag = impedance.real, impedance.imag
        exact_indices = []
        for target in grid:
            matches = np.flatnonzero(np.isclose(frequency[order], target, rtol=1e-9, atol=1e-9))
            if matches.size != 1:
                exact_indices = []
                break
            exact_indices.append(int(matches[0]))
        interpolated = not exact_indices
        if interpolated:
            interpolated_real = np.interp(log_grid, log_frequency, real)
            interpolated_imag = np.interp(log_grid, log_frequency, imag)
        else:
            interpolated_real = real[np.asarray(exact_indices, dtype=int)]
            interpolated_imag = imag[np.asarray(exact_indices, dtype=int)]
        record = {
            "capture_id": row["capture_id"],
            "banana_id": row["banana_id"],
            "measurement_date": row["measurement_date"],
            "study_day": row["study_day"],
            "ripeness_class": row["ripeness_class"],
            "raw_interpolated": interpolated,
            "initial_mass_g": row["initial_mass_g"],
            "signal_path_cm": row["signal_path_cm"],
            "arc_length_cm": row["arc_length_cm"],
            "circumference_cm": row["circumference_cm"],
        }
        for index, (real_value, imag_value) in enumerate(
            zip(interpolated_real, interpolated_imag, strict=True), start=1
        ):
            record[f"z_real_{index:02d}"] = float(real_value)
            record[f"z_imag_{index:02d}"] = float(imag_value)
        result.append(record)
    return result, grid


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sqlite_value(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_table(connection: sqlite3.Connection, name: str, rows: list[dict], columns: list[str]) -> None:
    connection.execute(f'DROP TABLE IF EXISTS "{name}"')
    definitions = []
    for column in columns:
        sample = next((row.get(column) for row in rows if row.get(column) is not None), None)
        sql_type = "REAL" if isinstance(sample, (float, np.floating)) else "INTEGER" if isinstance(
            sample, (int, bool, np.integer, np.bool_)
        ) else "TEXT"
        definitions.append(f'"{column}" {sql_type}')
    connection.execute(f'CREATE TABLE "{name}" ({", ".join(definitions)})')
    placeholders = ",".join("?" for _ in columns)
    names = ",".join(f'"{column}"' for column in columns)
    connection.executemany(
        f'INSERT INTO "{name}" ({names}) VALUES ({placeholders})',
        [[_sqlite_value(row.get(column)) for column in columns] for row in rows],
    )


def _circuit_parameters(fruit: str, electrode: str) -> list[str]:
    result = ["r_inf", "delta_r", "tau_s", "alpha", "r0"] if fruit == "cole" else ["r1", "r2", "c1_f"]
    if electrode == "capacitor":
        result.append("c_e_f")
    elif electrode == "warburg":
        result.append("sigma")
    elif electrode == "randles_cpe":
        result.extend(("r_e", "q_e", "n_e"))
    return result


def build_datasets(database_path: str | Path, output_directory: str | Path,
                   boundary_tolerance: float = 0.001) -> BuildSummary:
    database_path = Path(database_path).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        configuration = connection.execute(
            "SELECT config_json FROM fit_configs WHERE display_name=?", (TARGET_CONFIGURATION,)
        ).fetchone()
        if configuration is None:
            raise ValueError(f"required fitting configuration not found: {TARGET_CONFIGURATION}")
        config_payload = json.loads(configuration["config_json"])
        if (config_payload["population"], config_payload["iterations"], config_payload["restarts"]) != (30, 800, 5):
            raise ValueError("target configuration does not contain population 30, iterations 800, restarts 5")
        circuit_rows = _circuit_rows(connection, boundary_tolerance)
        raw_rows, raw_grid = _raw_rows(connection, config_payload)
    finally:
        connection.close()
    expected = len(raw_rows) * len(FRUIT_MODELS) * len(ELECTRODE_MODELS)
    if len(circuit_rows) != expected:
        raise ValueError(f"expected {expected} complete circuit rows but found {len(circuit_rows)}")

    files: list[Path] = []
    database_output = output / "ml_datasets.sqlite"
    if database_output.exists():
        database_output.unlink()
    output_connection = sqlite3.connect(database_output)
    try:
        master_columns = [
            *IDENTIFIERS, TARGET, "fit_id", "capture_id", "configuration", "fruit_model",
            "electrode_model", *PHYSICAL_ALL, "deembedded_nrmse", "aicc", "objective",
            "boundary_hit", "boundary_count", "boundary_parameters", *PARAMETER_NAMES,
            *(f"{name}_at_boundary" for name in OPTIMIZED_PARAMETER_NAMES),
        ]
        master_path = output / "master_circuit_fits.csv"
        _write_csv(master_path, circuit_rows, master_columns)
        files.append(master_path)
        _write_table(output_connection, "master_circuit_fits", circuit_rows, master_columns)

        manifests: dict[str, dict] = {}
        boundary_counts: dict[str, int] = {}
        for fruit in FRUIT_MODELS:
            for electrode in ELECTRODE_MODELS:
                key = f"{fruit}_{electrode}"
                selected = [
                    row for row in circuit_rows
                    if row["fruit_model"] == fruit and row["electrode_model"] == electrode
                ]
                parameters = _circuit_parameters(fruit, electrode)
                boundary_columns = [f"{name}_at_boundary" for name in parameters if name != "r0"]
                boundary_counts[key] = sum(bool(row["boundary_hit"]) for row in selected)
                for variant, physical in (("all", PHYSICAL_ALL), ("practical", PHYSICAL_PRACTICAL)):
                    table = f"{key}_{variant}"
                    columns = [
                        *IDENTIFIERS, TARGET, "fit_id", "capture_id", *physical,
                        *parameters, "deembedded_nrmse", "aicc", "objective",
                        "boundary_hit", "boundary_count", "boundary_parameters", *boundary_columns,
                    ]
                    path = output / f"{table}.csv"
                    _write_csv(path, selected, columns)
                    files.append(path)
                    _write_table(output_connection, table, selected, columns)
                    manifests[table] = {
                        "rows": len(selected),
                        "training_features": [*physical, *parameters],
                        "target": TARGET,
                        "group_column": "banana_id",
                        "audit_only": [
                            "fit_id", "capture_id", "measurement_date", "study_day",
                            "deembedded_nrmse", "aicc", "objective", "boundary_hit",
                            "boundary_count", "boundary_parameters", *boundary_columns,
                        ],
                    }

        raw_feature_columns = [
            item for index in range(1, len(raw_grid) + 1)
            for item in (f"z_real_{index:02d}", f"z_imag_{index:02d}")
        ]
        for variant, physical in (("all", PHYSICAL_ALL), ("practical", PHYSICAL_PRACTICAL)):
            table = f"raw_spectrum_{variant}"
            columns = [
                *IDENTIFIERS, TARGET, "capture_id", "raw_interpolated", *physical,
                *raw_feature_columns,
            ]
            path = output / f"{table}.csv"
            _write_csv(path, raw_rows, columns)
            files.append(path)
            _write_table(output_connection, table, raw_rows, columns)
            manifests[table] = {
                "rows": len(raw_rows),
                "training_features": [*physical, *raw_feature_columns],
                "target": TARGET,
                "group_column": "banana_id",
                "audit_only": [
                    "capture_id", "measurement_date", "study_day", "raw_interpolated",
                ],
            }
        output_connection.commit()
    finally:
        output_connection.close()

    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps({
        "created_at": _now(),
        "source_database": str(database_path),
        "fitting_configuration": TARGET_CONFIGURATION,
        "fitting_settings": {"population": 30, "iterations": 800, "restarts": 5},
        "boundary_tolerance_normalized": boundary_tolerance,
        "boundary_note": "Search-bound proximity is an audit field, not an MLP feature.",
        "physical_variants": {
            "all": list(PHYSICAL_ALL),
            "practical": list(PHYSICAL_PRACTICAL),
        },
        "raw_spectrum": {
            "representation": "de-embedded real and imaginary impedance",
            "fixed_path_removed": bool(config_payload["correct_fixed_path"]),
            "interpolation_axis": "log10(frequency_hz)",
            "grid_source": "the 34 measured frequencies above 1 kHz in later-day 100 Hz sweeps",
            "interpolation_policy": "later days use exact measured values; Day 1 is linearly interpolated in log-frequency",
            "frequency_hz": raw_grid.tolist(),
            "shared_range_hz": [float(raw_grid[0]), float(raw_grid[-1])],
        },
        "cross_validation": {
            "group_column": "banana_id",
            "note": "Keep every measurement from a banana in the same fold.",
        },
        "datasets": manifests,
        "boundary_hit_rows_by_model": boundary_counts,
    }, indent=2), encoding="utf-8")
    files.append(manifest_path)
    return BuildSummary(
        output, database_output, manifest_path, len(circuit_rows), len(raw_rows),
        tuple(files), boundary_counts,
    )
