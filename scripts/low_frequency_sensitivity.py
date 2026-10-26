"""Restricted-frequency circuit-fit sensitivity study.

Fits later-day spectra using only measured frequencies strictly above 1 kHz.
Results are checkpointed in a separate SQLite database and never modify the
primary batch-fit or machine-learning databases.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from math import gamma, pi, sin
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = ROOT / "data/processed/circuit_fits.sqlite"
OUTPUT = ROOT / "output/low_frequency_sensitivity"
RESULT_DB = OUTPUT / "restricted_frequency_fits.sqlite"
sys.path.insert(0, str(ROOT / "src"))

from eisfit.fit import FitResult, _metrics, fit_spectrum  # noqa: E402
from eisfit.models import decode_parameters, evaluate_model, parameter_specs  # noqa: E402
from eisfit.cuckoo import CuckooSearchResult, _latin_hypercube, _levy_steps  # noqa: E402
from eisbatch.configuration import BatchConfig  # noqa: E402


FRUITS = ("cole", "paper")
ELECTRODES = ("none", "capacitor", "warburg", "randles_cpe")
CONFIG_NAME = "P30-I800-R5-strictly-above-1kHz-sensitivity-v1"
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS study_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS restricted_fits (
    id INTEGER PRIMARY KEY,
    capture_id INTEGER NOT NULL,
    banana_id TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    fruit_model TEXT NOT NULL,
    electrode_model TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    fitting_domain_nrmse REAL,
    aicc REAL,
    aic REAL,
    bic REAL,
    objective REAL,
    parameters_json TEXT,
    bounds_json TEXT,
    boundary_hits_json TEXT,
    run_objectives_json TEXT,
    frequency_json TEXT,
    measured_corrected_real_json TEXT,
    measured_corrected_imag_json TEXT,
    predicted_real_json TEXT,
    predicted_imag_json TEXT,
    selected_indices_json TEXT,
    evaluations INTEGER,
    best_seed INTEGER,
    elapsed_seconds REAL,
    completed_at TEXT,
    UNIQUE(capture_id, fruit_model, electrode_model)
);
CREATE INDEX IF NOT EXISTS restricted_status ON restricted_fits(status);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def original_config(connection: sqlite3.Connection) -> tuple[sqlite3.Row, BatchConfig]:
    matches = []
    for row in connection.execute("SELECT * FROM fit_configs"):
        payload = json.loads(row["config_json"])
        if payload.get("population") == 30 and payload.get("iterations") == 800 and payload.get("restarts") == 5:
            matches.append((row, BatchConfig(**payload)))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one P30/I800/R5 configuration; found {len(matches)}")
    return matches[0]


def initialize_result_db(config_row: sqlite3.Row, config: BatchConfig) -> sqlite3.Connection:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(RESULT_DB)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    metadata = {
        "study_name": CONFIG_NAME,
        "source_database": str(SOURCE_DB.resolve()),
        "source_config_id": str(config_row["id"]),
        "source_config_name": str(config_row["display_name"]),
        "source_config_json": config.canonical_json(),
        "frequency_rule": "measured frequency_hz > 1000.0",
        "hardware_correction": "same as source P30/I800/R5 configuration",
        "objective": config.objective,
        "created_at": now(),
    }
    connection.executemany(
        "INSERT OR REPLACE INTO study_metadata(key,value) VALUES(?,?)", metadata.items()
    )
    connection.commit()
    return connection


def later_day_captures(source: sqlite3.Connection) -> tuple[str, list[dict]]:
    day_one = source.execute(
        "SELECT MIN(measurement_date) AS day_one FROM captures WHERE is_current=1"
    ).fetchone()["day_one"]
    rows = [dict(row) for row in source.execute(
        """SELECT * FROM captures
           WHERE is_current=1 AND measurement_date<>? AND min_frequency_hz<=1000
           ORDER BY measurement_date, banana_id""", (day_one,)
    )]
    return str(day_one), rows


def normalized_boundary_hits(parameters: dict, bounds: list[dict], tolerance: float = 0.001) -> dict:
    result = {}
    for bound in bounds:
        name = bound["name"]
        value = float(parameters[name])
        lower, upper = float(bound["lower"]), float(bound["upper"])
        if bound["scale"] == "linear":
            position = (value - lower) / (upper - lower)
        else:
            position = (math.log10(value) - math.log10(lower)) / (math.log10(upper) - math.log10(lower))
        result[name] = {
            "lower": bool(position <= tolerance),
            "upper": bool(position >= 1.0 - tolerance),
            "normalized_position": float(position),
        }
    return result


def batch_objective(frequency: np.ndarray, measured: np.ndarray, fruit: str, electrode: str, specs):
    omega = 2.0 * np.pi * frequency
    floor = max(float(np.median(np.abs(measured))) * 0.01, 1e-12)
    weights = 1.0 / np.maximum(np.abs(measured), floor)

    def evaluate(unit_positions: np.ndarray) -> np.ndarray:
        positions = np.atleast_2d(unit_positions)
        values = {}
        for column, spec in enumerate(specs):
            if spec.scale == "linear":
                decoded = spec.lower + positions[:, column] * (spec.upper - spec.lower)
            else:
                decoded = 10.0 ** (math.log10(spec.lower) + positions[:, column] *
                                   (math.log10(spec.upper) - math.log10(spec.lower)))
            values[spec.name] = decoded[:, None]
        jw = 1j * omega[None, :]
        if fruit == "cole":
            predicted = values["r_inf"] + values["delta_r"] / (
                1.0 + np.power(jw * values["tau_s"], values["alpha"])
            )
        else:
            predicted = values["r2"] * (1.0 + jw * values["r1"] * values["c1_f"]) / (
                1.0 + jw * (values["r1"] + values["r2"]) * values["c1_f"]
            )
        if electrode == "capacitor":
            predicted = predicted + 1.0 / (jw * values["c_e_f"])
        elif electrode == "warburg":
            predicted = predicted + values["sigma"] / np.sqrt(jw)
        elif electrode == "randles_cpe":
            predicted = predicted + values["r_e"] / (
                1.0 + values["r_e"] * values["q_e"] * np.power(jw, values["n_e"])
            )
        residual = predicted - measured[None, :]
        return np.mean(np.square(np.abs(residual) * weights[None, :]), axis=1)
    return evaluate


def fast_cuckoo_search(objective, dimensions: int, seed: int, config) -> CuckooSearchResult:
    """Vectorised equivalent of eisfit.cuckoo.cuckoo_search."""
    config.validate(); rng = np.random.default_rng(seed)
    nests = _latin_hypercube(rng, config.population, dimensions)
    fitness = np.asarray(objective(nests), dtype=float); evaluations = config.population
    best_index = int(np.argmin(fitness)); best = nests[best_index].copy(); best_fitness = float(fitness[best_index])
    history = [best_fitness]
    for iteration in range(config.iterations):
        levy = _levy_steps(rng, nests.shape, config.levy_beta)
        perturbation = rng.normal(size=nests.shape)
        proposals = np.clip(nests + config.step_scale * levy * (nests - best) * perturbation, 0.0, 1.0)
        proposed_fitness = np.asarray(objective(proposals)); evaluations += config.population
        improved = proposed_fitness < fitness; nests[improved] = proposals[improved]; fitness[improved] = proposed_fitness[improved]
        perm_a = rng.permutation(config.population); perm_b = rng.permutation(config.population)
        discovery_mask = rng.random(nests.shape) < config.abandonment_probability
        abandonment = np.clip(nests + rng.random(nests.shape) * (nests[perm_a] - nests[perm_b]) * discovery_mask, 0.0, 1.0)
        abandonment_fitness = np.asarray(objective(abandonment)); evaluations += config.population
        improved = abandonment_fitness < fitness; nests[improved] = abandonment[improved]; fitness[improved] = abandonment_fitness[improved]
        radius = max(0.002, 0.08 * (1.0 - iteration / config.iterations))
        elite_candidates = np.clip(best + rng.normal(0.0, radius, size=(config.population, dimensions)), 0.0, 1.0)
        elite_fitness = np.asarray(objective(elite_candidates)); evaluations += config.population
        order = np.argsort(fitness)[::-1]; elite_order = np.argsort(elite_fitness)
        for nest_index, candidate_index in zip(order, elite_order, strict=True):
            if elite_fitness[candidate_index] < fitness[nest_index]:
                nests[nest_index] = elite_candidates[candidate_index]; fitness[nest_index] = elite_fitness[candidate_index]
        iteration_best = int(np.argmin(fitness))
        if fitness[iteration_best] < best_fitness:
            best_fitness = float(fitness[iteration_best]); best = nests[iteration_best].copy()
        else:
            worst_index = int(np.argmax(fitness)); nests[worst_index] = best; fitness[worst_index] = best_fitness
        history.append(best_fitness)
    return CuckooSearchResult(best, best_fitness, history, evaluations)


def fast_fit_spectrum(frequency: np.ndarray, impedance: np.ndarray, fruit: str, electrode: str, fit_config) -> FitResult:
    corrected = impedance - fit_config.shield.impedance(frequency) if fit_config.correct_shield else impedance.copy()
    specs = parameter_specs(fruit, electrode, frequency, corrected)
    objective = batch_objective(frequency, corrected, fruit, electrode, specs)
    runs = [fast_cuckoo_search(objective, len(specs), fit_config.seed + restart, fit_config.cuckoo)
            for restart in range(fit_config.restarts)]
    best_index = int(np.argmin([run.objective for run in runs])); best = runs[best_index]
    parameters = decode_parameters(best.position, specs)
    predicted = evaluate_model(frequency, fruit, electrode, parameters)
    return FitResult(
        fruit_model=fruit, electrode_model=electrode, parameters=parameters, objective=best.objective,
        metrics=_metrics(corrected, predicted, len(specs)), corrected_impedance_ohm=corrected,
        predicted_impedance_ohm=predicted, run_objectives=[float(run.objective) for run in runs],
        best_seed=fit_config.seed + best_index, history=best.history,
        evaluations=sum(run.evaluations for run in runs),
        configuration={"fit": {"restarts": fit_config.restarts, "seed": fit_config.seed,
                               "objective": fit_config.objective, "correct_shield": fit_config.correct_shield},
                       "cuckoo": asdict(fit_config.cuckoo), "shield": asdict(fit_config.shield),
                       "parameter_bounds": [asdict(spec) for spec in specs]},
    )


def fit_job(capture: dict, config_payload: dict, fruit: str, electrode: str, vectorized: bool = False) -> dict:
    config = BatchConfig(**config_payload)
    frequency_all = np.asarray(json.loads(capture["frequency_json"]), dtype=float)
    impedance_all = np.asarray(json.loads(capture["real_json"]), dtype=float) + 1j * np.asarray(
        json.loads(capture["imag_json"]), dtype=float
    )
    indices = np.flatnonzero(frequency_all > 1000.0)
    if len(indices) < 4:
        raise ValueError(f"Only {len(indices)} measured points are strictly above 1 kHz")
    started = time.perf_counter()
    fitter = fast_fit_spectrum if vectorized else fit_spectrum
    result = fitter(frequency_all[indices], impedance_all[indices], fruit, electrode, config.fit_config())
    elapsed = time.perf_counter() - started
    payload = result.to_dict(include_spectra=True)
    bounds = payload["configuration"]["parameter_bounds"]
    hits = normalized_boundary_hits(payload["parameters"], bounds)
    return {
        "capture_id": capture["id"],
        "banana_id": capture["banana_id"],
        "measurement_date": capture["measurement_date"],
        "fruit_model": fruit,
        "electrode_model": electrode,
        "result": payload,
        "bounds": bounds,
        "boundary_hits": hits,
        "selected_indices": indices.tolist(),
        "selected_frequency_hz": frequency_all[indices].tolist(),
        "elapsed_seconds": elapsed,
    }


def save_success(connection: sqlite3.Connection, payload: dict) -> None:
    result = payload["result"]
    metrics = result["metrics"]
    corrected = np.asarray(result["corrected_real_ohm"]) + 1j * np.asarray(result["corrected_imag_ohm"])
    connection.execute(
        """INSERT INTO restricted_fits(
            capture_id,banana_id,measurement_date,fruit_model,electrode_model,status,
            fitting_domain_nrmse,aicc,aic,bic,objective,parameters_json,bounds_json,
            boundary_hits_json,run_objectives_json,frequency_json,
            measured_corrected_real_json,measured_corrected_imag_json,
            predicted_real_json,predicted_imag_json,selected_indices_json,evaluations,
            best_seed,elapsed_seconds,completed_at
        ) VALUES(?,?,?,?,?,'complete',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(capture_id,fruit_model,electrode_model) DO UPDATE SET
            status='complete',error_message=NULL,fitting_domain_nrmse=excluded.fitting_domain_nrmse,
            aicc=excluded.aicc,aic=excluded.aic,bic=excluded.bic,objective=excluded.objective,
            parameters_json=excluded.parameters_json,bounds_json=excluded.bounds_json,
            boundary_hits_json=excluded.boundary_hits_json,run_objectives_json=excluded.run_objectives_json,
            frequency_json=excluded.frequency_json,
            measured_corrected_real_json=excluded.measured_corrected_real_json,
            measured_corrected_imag_json=excluded.measured_corrected_imag_json,
            predicted_real_json=excluded.predicted_real_json,predicted_imag_json=excluded.predicted_imag_json,
            selected_indices_json=excluded.selected_indices_json,evaluations=excluded.evaluations,
            best_seed=excluded.best_seed,elapsed_seconds=excluded.elapsed_seconds,
            completed_at=excluded.completed_at""",
        (
            payload["capture_id"], payload["banana_id"], payload["measurement_date"],
            payload["fruit_model"], payload["electrode_model"], metrics["nrmse_complex"],
            metrics["aicc"], metrics["aic"], metrics["bic"], result["objective"],
            json.dumps(result["parameters"]), json.dumps(payload["bounds"]),
            json.dumps(payload["boundary_hits"]), json.dumps(result["run_objectives"]),
            json.dumps(payload["selected_frequency_hz"]),
            json.dumps(corrected.real.tolist()), json.dumps(corrected.imag.tolist()),
            json.dumps(result["predicted_real_ohm"]), json.dumps(result["predicted_imag_ohm"]),
            json.dumps(payload["selected_indices"]), result["evaluations"], result["best_seed"],
            payload["elapsed_seconds"], now(),
        ),
    )
    connection.commit()


def save_failure(connection: sqlite3.Connection, capture: dict, fruit: str, electrode: str, message: str) -> None:
    connection.execute(
        """INSERT INTO restricted_fits(
            capture_id,banana_id,measurement_date,fruit_model,electrode_model,status,error_message,completed_at
        ) VALUES(?,?,?,?,?,'failed',?,?)
        ON CONFLICT(capture_id,fruit_model,electrode_model) DO UPDATE SET
            status='failed',error_message=excluded.error_message,completed_at=excluded.completed_at""",
        (capture["id"], capture["banana_id"], capture["measurement_date"], fruit, electrode, message, now()),
    )
    connection.commit()


def run(workers: int, max_jobs: int | None = None, vectorized: bool = False) -> None:
    source = sqlite3.connect(SOURCE_DB)
    source.row_factory = sqlite3.Row
    config_row, config = original_config(source)
    day_one, captures = later_day_captures(source)
    result_db = initialize_result_db(config_row, config)
    result_db.execute("INSERT OR REPLACE INTO study_metadata(key,value) VALUES('day_one_date',?)", (day_one,))
    result_db.execute("INSERT OR REPLACE INTO study_metadata(key,value) VALUES('later_capture_count',?)", (str(len(captures)),))
    result_db.commit()
    existing = {
        (row["capture_id"], row["fruit_model"], row["electrode_model"])
        for row in result_db.execute("SELECT capture_id,fruit_model,electrode_model FROM restricted_fits WHERE status='complete'")
    }
    jobs = [(capture, fruit, electrode) for capture in captures for fruit in FRUITS for electrode in ELECTRODES
            if (capture["id"], fruit, electrode) not in existing]
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    total = len(jobs)
    print(f"{CONFIG_NAME}: day one={day_one}; later captures={len(captures)}; pending fits={total}; workers={workers}", flush=True)
    if not jobs:
        result_db.close(); source.close(); return
    started = time.perf_counter()
    complete = failed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fit_job, capture, config.payload(), fruit, electrode, vectorized):
                   (capture, fruit, electrode) for capture, fruit, electrode in jobs}
        for future in as_completed(futures):
            capture, fruit, electrode = futures[future]
            try:
                save_success(result_db, future.result())
                complete += 1
            except Exception as exc:
                save_failure(result_db, capture, fruit, electrode, repr(exc))
                failed += 1
            done = complete + failed
            if done == 1 or done % 10 == 0 or done == total:
                elapsed = time.perf_counter() - started
                rate = done / max(elapsed, 1e-9)
                remaining = (total - done) / max(rate, 1e-9)
                print(f"{done}/{total} complete={complete} failed={failed} elapsed={elapsed/60:.1f} min remaining~{remaining/60:.1f} min", flush=True)
    result_db.close(); source.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--vectorized", action="store_true")
    args = parser.parse_args()
    run(args.workers, args.max_jobs, args.vectorized)


if __name__ == "__main__":
    main()
