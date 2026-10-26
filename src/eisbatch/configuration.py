"""Versioned batch-fit configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import eisfit

from .engine import CuckooSearchConfig, FitConfig, ShieldModel


def _engine_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in ("cuckoo.py", "fit.py", "io.py", "models.py"):
        digest.update((Path(eisfit.__file__).resolve().parent / name).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class BatchConfig:
    population: int = 30
    iterations: int = 800
    restarts: int = 5
    seed: int = 20260923
    abandonment_probability: float = 0.25
    levy_beta: float = 1.5
    step_scale: float = 0.01
    objective: str = "relative"
    correct_fixed_path: bool = True
    fixed_resistance_ohm: float = 2000.0
    fixed_capacitance_f: float = 15e-9 * 470e-9 / (15e-9 + 470e-9)
    frequency_points: int | None = None
    frequency_selection: str = "even_log_index"
    engine_version: str = "eisfit-hybrid-cs-v1"
    engine_fingerprint: str = field(default_factory=_engine_fingerprint)

    def validate(self) -> None:
        CuckooSearchConfig(
            population=self.population,
            iterations=self.iterations,
            abandonment_probability=self.abandonment_probability,
            levy_beta=self.levy_beta,
            step_scale=self.step_scale,
        ).validate()
        if self.restarts < 1:
            raise ValueError("restarts must be positive")
        if self.frequency_points is not None and self.frequency_points < 4:
            raise ValueError("frequency points must be at least 4, or blank for all")

    def payload(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))

    def identity(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def short_name(self) -> str:
        points = "all" if self.frequency_points is None else str(self.frequency_points)
        return f"P{self.population}-I{self.iterations}-R{self.restarts}-F{points}-{self.identity()[:8]}"

    def fit_config(self) -> FitConfig:
        return FitConfig(
            restarts=self.restarts,
            seed=self.seed,
            objective=self.objective,
            correct_shield=self.correct_fixed_path,
            shield=ShieldModel(self.fixed_resistance_ohm, self.fixed_capacitance_f),
            cuckoo=CuckooSearchConfig(
                population=self.population,
                iterations=self.iterations,
                abandonment_probability=self.abandonment_probability,
                levy_beta=self.levy_beta,
                step_scale=self.step_scale,
            ),
        )
