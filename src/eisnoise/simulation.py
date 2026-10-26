"""Generate raw embedded spectra, de-embed them, and refit fruit models."""

from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .engine import (
    CuckooSearchConfig,
    FitConfig,
    ShieldModel,
    evaluate_model,
    fit_spectrum,
    parameter_specs,
)


REFERENCE_PARAMETERS: dict[str, dict[str, float]] = {
    "cole": {
        "r_inf": 462.6542624817607,
        "delta_r": 1275.0599467767006,
        "tau_s": 7.825247766784613e-06,
        "alpha": 0.8081010880352602,
        "r0": 1737.7142092584613,
    },
    "paper": {
        "r1": 864.1934517959908,
        "r2": 1645.1018275775402,
        "c1_f": 2.922359961474727e-09,
    },
}

CONDITIONS = ("no_noise", "relative_complex_gaussian")


@dataclass(frozen=True)
class NoiseStudyConfig:
    replicates: int = 200
    points: int = 50
    minimum_frequency_hz: float = 100.0
    maximum_frequency_hz: float = 180_000.0
    low_noise_fraction: float = 0.08
    baseline_noise_fraction: float = 0.01
    low_noise_end_hz: float = 300.0
    transition_end_hz: float = 600.0
    noise_application_domain: str = "raw_embedded"
    fixed_series_resistance_ohm: float = 2000.0
    fixed_series_capacitance_f: float = 15e-9 * 470e-9 / (15e-9 + 470e-9)
    population: int = 20
    iterations: int = 300
    restarts: int = 3
    seed: int = 20261008
    workers: int = 0
    reference_parameters: dict[str, dict[str, float]] = field(
        default_factory=lambda: {name: values.copy() for name, values in REFERENCE_PARAMETERS.items()}
    )

    def validate(self) -> None:
        if self.replicates < 1:
            raise ValueError("replicates must be positive")
        if self.points < 3:
            raise ValueError("points must be at least 3")
        if not 0 < self.minimum_frequency_hz < self.maximum_frequency_hz:
            raise ValueError("frequency bounds are invalid")
        if not 0 <= self.baseline_noise_fraction <= self.low_noise_fraction:
            raise ValueError("noise fractions must satisfy 0 <= baseline <= low-frequency")
        if not self.minimum_frequency_hz <= self.low_noise_end_hz < self.transition_end_hz:
            raise ValueError("noise transition frequencies are invalid")
        if self.population < 5 or self.iterations < 1 or self.restarts < 1:
            raise ValueError("optimizer settings are invalid")
        if self.noise_application_domain != "raw_embedded":
            raise ValueError("noise_application_domain must be 'raw_embedded'")
        if set(self.reference_parameters) != {"cole", "paper"}:
            raise ValueError("reference parameters must contain Cole and paper models")

    def frequency_hz(self) -> np.ndarray:
        return np.geomspace(self.minimum_frequency_hz, self.maximum_frequency_hz, self.points)

    def resolved_workers(self) -> int:
        if self.workers > 0:
            return self.workers
        logical = os.cpu_count() or 1
        return max(1, min(4, logical // 2 if logical > 2 else 1))


def noise_profile(frequency_hz: np.ndarray, config: NoiseStudyConfig) -> np.ndarray:
    """Relative RMS complex-noise level, with a cosine taper above 300 Hz."""
    frequency = np.asarray(frequency_hz, dtype=float)
    result = np.full_like(frequency, config.baseline_noise_fraction)
    result[frequency <= config.low_noise_end_hz] = config.low_noise_fraction
    transition = (frequency > config.low_noise_end_hz) & (frequency < config.transition_end_hz)
    position = (
        (frequency[transition] - config.low_noise_end_hz)
        / (config.transition_end_hz - config.low_noise_end_hz)
    )
    blend = 0.5 * (1.0 + np.cos(np.pi * position))
    result[transition] = config.baseline_noise_fraction + (
        config.low_noise_fraction - config.baseline_noise_fraction
    ) * blend
    return result


def add_relative_complex_gaussian_noise(
    ideal: np.ndarray, relative_sigma: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Add circular zero-mean complex noise with requested RMS relative magnitude."""
    scale = np.asarray(relative_sigma, dtype=float) * np.abs(ideal) / math.sqrt(2.0)
    noise = scale * (rng.normal(size=ideal.size) + 1j * rng.normal(size=ideal.size))
    return ideal + noise


def _is_bound_hit(value: float, spec, tolerance: float = 1e-4) -> int:
    """Test proximity to a bound in the optimizer's normalized coordinate."""
    if spec.scale == "linear":
        coordinate = (value - spec.lower) / (spec.upper - spec.lower)
    else:
        coordinate = (
            (math.log10(value) - math.log10(spec.lower))
            / (math.log10(spec.upper) - math.log10(spec.lower))
        )
    return int(coordinate <= tolerance or coordinate >= 1.0 - tolerance)


def _task(payload: dict[str, Any]) -> dict[str, Any]:
    config = NoiseStudyConfig(**payload["config"])
    model = payload["model"]
    condition = payload["condition"]
    replicate = payload["replicate"]
    frequency = config.frequency_hz()
    true_parameters = config.reference_parameters[model]
    ideal_fruit = evaluate_model(frequency, model, "none", true_parameters)
    shield = ShieldModel(
        config.fixed_series_resistance_ohm, config.fixed_series_capacitance_f
    )
    ideal_raw = ideal_fruit + shield.impedance(frequency)
    data_seed = config.seed + replicate + (0 if model == "cole" else 1_000_000)
    if condition == "relative_complex_gaussian":
        # Noise is generated in the raw measurement domain, where the fruit and
        # known fixed series path are both present. The normal fitting path then
        # removes the known series impedance before fitting the fruit model.
        measured = add_relative_complex_gaussian_noise(
            ideal_raw, noise_profile(frequency, config), np.random.default_rng(data_seed)
        )
        correct_shield = True
    else:
        # Retain the original ideal fruit-only baseline. With exact arithmetic,
        # adding and immediately removing the known path would be equivalent.
        measured = ideal_fruit.copy()
        correct_shield = False
    condition_offset = 0 if condition == "no_noise" else 10_000_000
    model_offset = 0 if model == "cole" else 1_000_000
    fit_seed = config.seed + condition_offset + model_offset + replicate * config.restarts
    fit_config = FitConfig(
        restarts=config.restarts,
        seed=fit_seed,
        objective="relative",
        correct_shield=correct_shield,
        shield=shield,
        cuckoo=CuckooSearchConfig(
            population=config.population,
            iterations=config.iterations,
        ),
    )
    result = fit_spectrum(frequency, measured, model, "none", fit_config)
    specs = parameter_specs(model, "none", frequency, measured)
    specs_by_name = {spec.name: spec for spec in specs}
    parameter_names = list(true_parameters)
    row: dict[str, Any] = {
        "model": model,
        "condition": condition,
        "noise_application_domain": "raw_embedded" if condition != "no_noise" else "none",
        "replicate": replicate,
        "data_seed": data_seed if condition != "no_noise" else "",
        "fit_seed": fit_seed,
        "objective": result.objective,
        "nrmse": result.metrics["nrmse_complex"],
        "aicc": result.metrics["aicc"],
        "evaluations": result.evaluations,
    }
    for name in parameter_names:
        true_value = true_parameters[name]
        estimated = result.parameters[name]
        row[f"true_{name}"] = true_value
        row[f"estimated_{name}"] = estimated
        row[f"error_{name}"] = estimated - true_value
        row[f"relative_error_pct_{name}"] = 100.0 * (estimated - true_value) / true_value
        if name in specs_by_name:
            row[f"bound_hit_{name}"] = _is_bound_hit(estimated, specs_by_name[name])
        else:
            row[f"bound_hit_{name}"] = 0
    return row


def summarize(rows: list[dict[str, Any]], config: NoiseStudyConfig) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for model in ("cole", "paper"):
        for condition in CONDITIONS:
            selected = [row for row in rows if row["model"] == model and row["condition"] == condition]
            for parameter, true_value in config.reference_parameters[model].items():
                estimates = np.asarray([row[f"estimated_{parameter}"] for row in selected], dtype=float)
                errors = estimates - true_value
                summary.append({
                    "model": model,
                    "condition": condition,
                    "parameter": parameter,
                    "true_value": true_value,
                    "count": len(estimates),
                    "mean_estimate": float(np.mean(estimates)),
                    "median_estimate": float(np.median(estimates)),
                    "standard_deviation": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
                    "bias": float(np.mean(errors)),
                    "percent_bias": float(100.0 * np.mean(errors) / true_value),
                    "rmse": float(np.sqrt(np.mean(errors ** 2))),
                    "relative_rmse_pct": float(100.0 * np.sqrt(np.mean(errors ** 2)) / true_value),
                    "p2_5": float(np.percentile(estimates, 2.5)),
                    "p97_5": float(np.percentile(estimates, 97.5)),
                    "bound_hit_rate_pct": float(100.0 * np.mean([
                        float(row[f"bound_hit_{parameter}"]) for row in selected
                    ])),
                })
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "model", "condition", "replicate", "parameter", "true_value", "count",
        "data_seed", "fit_seed", "objective", "nrmse", "aicc", "evaluations",
    ]
    ordered = [name for name in preferred if name in fieldnames]
    ordered.extend(name for name in fieldnames if name not in ordered)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_study(
    config: NoiseStudyConfig,
    output_directory: str | Path,
    reuse_no_noise_from: str | Path | None = None,
) -> tuple[Path, Path]:
    config.validate()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    payload_config = asdict(config)
    reused_rows: list[dict[str, Any]] = []
    if reuse_no_noise_from is not None:
        source = Path(reuse_no_noise_from)
        reused_rows = [row for row in _read_csv(source) if row["condition"] == "no_noise"]
        expected = 2 * config.replicates
        if len(reused_rows) != expected:
            raise ValueError(
                f"Expected {expected} reusable no-noise rows in {source}, found {len(reused_rows)}"
            )

    conditions = ("relative_complex_gaussian",) if reused_rows else CONDITIONS
    tasks = [
        {"config": payload_config, "model": model, "condition": condition, "replicate": replicate}
        for model in ("cole", "paper")
        for condition in conditions
        for replicate in range(config.replicates)
    ]
    rows: list[dict[str, Any]] = reused_rows
    with ProcessPoolExecutor(max_workers=config.resolved_workers()) as executor:
        futures = [executor.submit(_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % max(1, len(tasks) // 20) == 0 or completed == len(tasks):
                print(f"Completed {completed}/{len(tasks)} fits", flush=True)
    rows.sort(key=lambda row: (row["model"], row["condition"], row["replicate"]))
    summary_rows = summarize(rows, config)
    replicate_path = output / "noise_recovery_replicates.csv"
    summary_path = output / "noise_recovery_summary.csv"
    _write_csv(replicate_path, rows)
    _write_csv(summary_path, summary_rows)
    (output / "study_configuration.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    return replicate_path, summary_path
