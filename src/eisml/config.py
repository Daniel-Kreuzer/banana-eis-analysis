"""Configuration for reproducible MLP studies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TrainingConfig:
    outer_folds: int = 5
    inner_folds: int = 4
    random_candidates: int = 24
    random_seed: int = 4022
    max_iterations: int = 5000
    max_function_evaluations: int = 50000
    alpha_min: float = 1e-6
    alpha_max: float = 1e-1
    solver: str = "lbfgs"
    selection_metric: str = "f1_macro"
    activations: tuple[str, ...] = ("relu", "tanh")
    circuit_architectures: tuple[tuple[int, ...], ...] = (
        (3,), (5,), (10,), (16,), (5, 3), (10, 3), (16, 8)
    )
    raw_architectures: tuple[tuple[int, ...], ...] = (
        (16,), (32,), (64,), (32, 16), (64, 32)
    )
    preprocessing_variants: tuple[str, ...] = ("standard", "robust", "log_standard")
    fit_final_model: bool = True
    workers: int = 0
    schema_version: int = 2
    notes: dict[str, str] = field(default_factory=lambda: {
        "split": "Grouped by banana_id; inner folds select hyperparameters and outer folds test.",
        "preprocessing": (
            "Compare standard scaling, robust scaling, and log10 plus standard scaling for "
            "strictly positive wide-range circuit parameters. All transforms are fitted in-fold."
        ),
        "workers": "Zero selects half the logical CPUs, capped at four.",
    })

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

