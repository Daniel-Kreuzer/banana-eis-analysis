"""Repeated Cuckoo Search fitting and fit diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .cuckoo import CuckooSearchConfig, cuckoo_search
from .models import (
    ELECTRODE_MODELS,
    FRUIT_MODELS,
    ShieldModel,
    decode_parameters,
    evaluate_model,
    parameter_specs,
)


@dataclass(frozen=True)
class FitConfig:
    restarts: int = 5
    seed: int = 20260923
    objective: str = "relative"
    cuckoo: CuckooSearchConfig = CuckooSearchConfig()
    correct_shield: bool = True
    shield: ShieldModel = ShieldModel()


@dataclass
class FitResult:
    fruit_model: str
    electrode_model: str
    parameters: dict[str, float]
    objective: float
    metrics: dict[str, float]
    corrected_impedance_ohm: np.ndarray
    predicted_impedance_ohm: np.ndarray
    run_objectives: list[float]
    best_seed: int
    history: list[float]
    evaluations: int
    configuration: dict[str, Any]

    def to_dict(self, include_spectra: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "fruit_model": self.fruit_model,
            "electrode_model": self.electrode_model,
            "parameters": self.parameters,
            "objective": self.objective,
            "metrics": self.metrics,
            "run_objectives": self.run_objectives,
            "best_seed": self.best_seed,
            "history": self.history,
            "evaluations": self.evaluations,
            "configuration": self.configuration,
        }
        if include_spectra:
            result["corrected_real_ohm"] = self.corrected_impedance_ohm.real.tolist()
            result["corrected_imag_ohm"] = self.corrected_impedance_ohm.imag.tolist()
            result["predicted_real_ohm"] = self.predicted_impedance_ohm.real.tolist()
            result["predicted_imag_ohm"] = self.predicted_impedance_ohm.imag.tolist()
        return result


def _objective_function(
    frequency_hz: np.ndarray,
    measured_z: np.ndarray,
    fruit_model: str,
    electrode_model: str,
    specs,
    objective_name: str,
):
    floor = max(float(np.median(np.abs(measured_z))) * 0.01, 1e-12)

    def objective(unit_vector: np.ndarray) -> float:
        parameters = decode_parameters(unit_vector, specs)
        predicted = evaluate_model(frequency_hz, fruit_model, electrode_model, parameters)
        residual = predicted - measured_z
        if objective_name == "relative":
            weights = 1.0 / np.maximum(np.abs(measured_z), floor)
            return float(np.mean(np.square(np.abs(residual) * weights)))
        if objective_name == "absolute":
            return float(np.mean(np.square(np.abs(residual))))
        raise ValueError("objective must be 'relative' or 'absolute'")

    return objective


def _metrics(measured: np.ndarray, predicted: np.ndarray, parameter_count: int) -> dict[str, float]:
    residual = predicted - measured
    squared = np.square(residual.real) + np.square(residual.imag)
    rss = max(float(np.sum(squared)), np.finfo(float).tiny)
    observations = 2 * len(measured)
    rmse = float(np.sqrt(np.mean(squared)))
    nrmse = rmse / max(float(np.sqrt(np.mean(np.square(np.abs(measured))))), 1e-12)
    aic = observations * np.log(rss / observations) + 2 * parameter_count
    denominator = observations - parameter_count - 1
    aicc = aic + (2 * parameter_count * (parameter_count + 1) / denominator) if denominator > 0 else np.inf
    bic = observations * np.log(rss / observations) + parameter_count * np.log(observations)
    return {
        "rss_complex_ohm2": rss,
        "rmse_complex_ohm": rmse,
        "nrmse_complex": nrmse,
        "aic": float(aic),
        "aicc": float(aicc),
        "bic": float(bic),
    }


def fit_spectrum(
    frequency_hz: np.ndarray,
    measured_impedance_ohm: np.ndarray,
    fruit_model: str,
    electrode_model: str,
    config: FitConfig | None = None,
) -> FitResult:
    cfg = config or FitConfig()
    if fruit_model not in FRUIT_MODELS or electrode_model not in ELECTRODE_MODELS:
        raise ValueError("unsupported fruit/electrode model combination")
    if cfg.restarts < 1:
        raise ValueError("restarts must be positive")

    frequency = np.asarray(frequency_hz, dtype=float)
    measured = np.asarray(measured_impedance_ohm, dtype=complex)
    if cfg.correct_shield:
        corrected = measured - cfg.shield.impedance(frequency)
    else:
        corrected = measured.copy()

    specs = parameter_specs(fruit_model, electrode_model, frequency, corrected)
    objective = _objective_function(
        frequency, corrected, fruit_model, electrode_model, specs, cfg.objective
    )

    runs = []
    for restart in range(cfg.restarts):
        seed = cfg.seed + restart
        runs.append(cuckoo_search(objective, len(specs), seed, cfg.cuckoo))
    best_index = int(np.argmin([run.objective for run in runs]))
    best = runs[best_index]
    parameters = decode_parameters(best.position, specs)
    predicted = evaluate_model(frequency, fruit_model, electrode_model, parameters)

    return FitResult(
        fruit_model=fruit_model,
        electrode_model=electrode_model,
        parameters=parameters,
        objective=best.objective,
        metrics=_metrics(corrected, predicted, len(specs)),
        corrected_impedance_ohm=corrected,
        predicted_impedance_ohm=predicted,
        run_objectives=[float(run.objective) for run in runs],
        best_seed=cfg.seed + best_index,
        history=best.history,
        evaluations=sum(run.evaluations for run in runs),
        configuration={
            "fit": {
                "restarts": cfg.restarts,
                "seed": cfg.seed,
                "objective": cfg.objective,
                "correct_shield": cfg.correct_shield,
            },
            "cuckoo": asdict(cfg.cuckoo),
            "shield": asdict(cfg.shield),
            "parameter_bounds": [asdict(spec) for spec in specs],
        },
    )
