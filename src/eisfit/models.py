"""Complex impedance models and bounded parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable

import numpy as np


FRUIT_MODELS = ("cole", "paper")
ELECTRODE_MODELS = ("none", "capacitor", "warburg", "randles_cpe")


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    scale: str = "log"

    def decode(self, value_01: np.ndarray | float) -> np.ndarray | float:
        if self.scale == "linear":
            return self.lower + np.asarray(value_01) * (self.upper - self.lower)
        lo = np.log10(self.lower)
        hi = np.log10(self.upper)
        return np.power(10.0, lo + np.asarray(value_01) * (hi - lo))


@dataclass(frozen=True)
class ShieldModel:
    resistance_ohm: float = 2000.0
    capacitance_f: float = 15e-9 * 470e-9 / (15e-9 + 470e-9)

    def impedance(self, frequency_hz: np.ndarray) -> np.ndarray:
        omega = 2.0 * pi * np.asarray(frequency_hz, dtype=float)
        return self.resistance_ohm + 1.0 / (1j * omega * self.capacitance_f)


def cole_impedance(omega: np.ndarray, p: dict[str, float]) -> np.ndarray:
    return p["r_inf"] + p["delta_r"] / (
        1.0 + np.power(1j * omega * p["tau_s"], p["alpha"])
    )


def paper_impedance(omega: np.ndarray, p: dict[str, float]) -> np.ndarray:
    # R2 in parallel with the series branch R1 + C1.
    return p["r2"] * (1.0 + 1j * omega * p["r1"] * p["c1_f"]) / (
        1.0 + 1j * omega * (p["r1"] + p["r2"]) * p["c1_f"]
    )


def no_electrode(omega: np.ndarray, p: dict[str, float]) -> np.ndarray:
    return np.zeros_like(omega, dtype=complex)


def capacitor_electrode(omega: np.ndarray, p: dict[str, float]) -> np.ndarray:
    return 1.0 / (1j * omega * p["c_e_f"])


def warburg_electrode(omega: np.ndarray, p: dict[str, float]) -> np.ndarray:
    # Semi-infinite Warburg: sigma / sqrt(j*w), with phase -45 degrees.
    return p["sigma"] / np.sqrt(1j * omega)


def randles_cpe_electrode(omega: np.ndarray, p: dict[str, float]) -> np.ndarray:
    return p["r_e"] / (
        1.0 + p["r_e"] * p["q_e"] * np.power(1j * omega, p["n_e"])
    )


_FRUIT_FUNCTIONS: dict[str, Callable[[np.ndarray, dict[str, float]], np.ndarray]] = {
    "cole": cole_impedance,
    "paper": paper_impedance,
}

_ELECTRODE_FUNCTIONS: dict[str, Callable[[np.ndarray, dict[str, float]], np.ndarray]] = {
    "none": no_electrode,
    "capacitor": capacitor_electrode,
    "warburg": warburg_electrode,
    "randles_cpe": randles_cpe_electrode,
}


def parameter_specs(
    fruit_model: str,
    electrode_model: str,
    frequency_hz: np.ndarray,
    measured_z: np.ndarray,
) -> list[ParameterSpec]:
    if fruit_model not in FRUIT_MODELS:
        raise ValueError(f"Unknown fruit model: {fruit_model}")
    if electrode_model not in ELECTRODE_MODELS:
        raise ValueError(f"Unknown electrode model: {electrode_model}")

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    magnitude = np.abs(np.asarray(measured_z, dtype=complex))
    positive = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
    reference = float(np.median(positive)) if positive.size else 1000.0
    maximum = float(np.max(positive)) if positive.size else reference
    r_lower = max(reference * 1e-5, 1e-3)
    r_upper = max(maximum * 200.0, 1e4)
    f_min = float(np.min(frequency_hz))
    f_max = float(np.max(frequency_hz))

    if fruit_model == "cole":
        specs = [
            ParameterSpec("r_inf", r_lower, r_upper),
            ParameterSpec("delta_r", r_lower, r_upper),
            ParameterSpec("tau_s", 1.0 / (2 * pi * f_max) * 1e-3,
                          1.0 / (2 * pi * f_min) * 1e3),
            ParameterSpec("alpha", 0.10, 1.0, "linear"),
        ]
    else:
        specs = [
            ParameterSpec("r1", r_lower, r_upper),
            ParameterSpec("r2", r_lower, r_upper),
            ParameterSpec("c1_f", 1e-12, 1e-2),
        ]

    if electrode_model == "capacitor":
        specs.append(ParameterSpec("c_e_f", 1e-12, 1e-2))
    elif electrode_model == "warburg":
        sigma_upper = max(maximum * np.sqrt(2 * pi * f_max) * 200.0, 1e5)
        specs.append(ParameterSpec("sigma", 1e-6, sigma_upper))
    elif electrode_model == "randles_cpe":
        specs.extend(
            [
                ParameterSpec("r_e", r_lower, r_upper),
                ParameterSpec("q_e", 1e-12, 1.0),
                ParameterSpec("n_e", 0.10, 1.0, "linear"),
            ]
        )
    return specs


def decode_parameters(unit_vector: np.ndarray, specs: list[ParameterSpec]) -> dict[str, float]:
    values = {
        spec.name: float(spec.decode(value))
        for spec, value in zip(specs, unit_vector, strict=True)
    }
    if "r_inf" in values and "delta_r" in values:
        values["r0"] = values["r_inf"] + values["delta_r"]
    return values


def evaluate_model(
    frequency_hz: np.ndarray,
    fruit_model: str,
    electrode_model: str,
    parameters: dict[str, float],
) -> np.ndarray:
    omega = 2.0 * pi * np.asarray(frequency_hz, dtype=float)
    return (
        _FRUIT_FUNCTIONS[fruit_model](omega, parameters)
        + _ELECTRODE_FUNCTIONS[electrode_model](omega, parameters)
    )

