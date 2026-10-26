"""Bounded Cuckoo Search using Levy flights and nest abandonment."""

from __future__ import annotations

from dataclasses import dataclass
from math import gamma, pi, sin
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CuckooSearchConfig:
    population: int = 30
    iterations: int = 800
    abandonment_probability: float = 0.25
    levy_beta: float = 1.5
    step_scale: float = 0.01

    def validate(self) -> None:
        if self.population < 5:
            raise ValueError("population must be at least 5")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if not 0.0 < self.abandonment_probability < 1.0:
            raise ValueError("abandonment_probability must be between 0 and 1")
        if not 1.0 < self.levy_beta <= 2.0:
            raise ValueError("levy_beta must be in (1, 2]")
        if self.step_scale <= 0:
            raise ValueError("step_scale must be positive")


@dataclass
class CuckooSearchResult:
    position: np.ndarray
    objective: float
    history: list[float]
    evaluations: int


def _latin_hypercube(rng: np.random.Generator, count: int, dimensions: int) -> np.ndarray:
    result = np.empty((count, dimensions), dtype=float)
    for dimension in range(dimensions):
        bins = (np.arange(count) + rng.random(count)) / count
        rng.shuffle(bins)
        result[:, dimension] = bins
    return result


def _levy_steps(
    rng: np.random.Generator, shape: tuple[int, int], beta: float
) -> np.ndarray:
    sigma_u = (
        gamma(1.0 + beta)
        * sin(pi * beta / 2.0)
        / (gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u, size=shape)
    v = rng.normal(0.0, 1.0, size=shape)
    return u / np.power(np.abs(v) + np.finfo(float).eps, 1.0 / beta)


def cuckoo_search(
    objective: Callable[[np.ndarray], float],
    dimensions: int,
    seed: int,
    config: CuckooSearchConfig | None = None,
) -> CuckooSearchResult:
    """Minimize an objective over a unit hypercube."""
    cfg = config or CuckooSearchConfig()
    cfg.validate()
    if dimensions < 1:
        raise ValueError("dimensions must be positive")

    rng = np.random.default_rng(seed)
    nests = _latin_hypercube(rng, cfg.population, dimensions)
    fitness = np.asarray([objective(nest) for nest in nests], dtype=float)
    evaluations = cfg.population
    best_index = int(np.argmin(fitness))
    best = nests[best_index].copy()
    best_fitness = float(fitness[best_index])
    history = [best_fitness]

    for iteration in range(cfg.iterations):
        levy = _levy_steps(rng, nests.shape, cfg.levy_beta)
        perturbation = rng.normal(size=nests.shape)
        proposals = nests + cfg.step_scale * levy * (nests - best) * perturbation
        proposals = np.clip(proposals, 0.0, 1.0)
        proposed_fitness = np.asarray([objective(candidate) for candidate in proposals])
        evaluations += cfg.population

        improved = proposed_fitness < fitness
        nests[improved] = proposals[improved]
        fitness[improved] = proposed_fitness[improved]

        # Discover and abandon nests using the standard differential random walk.
        perm_a = rng.permutation(cfg.population)
        perm_b = rng.permutation(cfg.population)
        discovery_mask = rng.random(nests.shape) < cfg.abandonment_probability
        differential = nests[perm_a] - nests[perm_b]
        abandonment = nests + rng.random(nests.shape) * differential * discovery_mask
        abandonment = np.clip(abandonment, 0.0, 1.0)
        abandonment_fitness = np.asarray([objective(candidate) for candidate in abandonment])
        evaluations += cfg.population
        improved = abandonment_fitness < fitness
        nests[improved] = abandonment[improved]
        fitness[improved] = abandonment_fitness[improved]

        # Bounded elite exploration improves convergence in log-scaled parameter
        # spaces while retaining the stochastic CS search.  Its radius decays as
        # the population converges.
        radius = max(0.002, 0.08 * (1.0 - iteration / cfg.iterations))
        elite_candidates = np.clip(
            best + rng.normal(0.0, radius, size=(cfg.population, dimensions)),
            0.0,
            1.0,
        )
        elite_fitness = np.asarray([objective(candidate) for candidate in elite_candidates])
        evaluations += cfg.population
        order = np.argsort(fitness)[::-1]
        elite_order = np.argsort(elite_fitness)
        for nest_index, candidate_index in zip(order, elite_order, strict=True):
            if elite_fitness[candidate_index] < fitness[nest_index]:
                nests[nest_index] = elite_candidates[candidate_index]
                fitness[nest_index] = elite_fitness[candidate_index]

        iteration_best = int(np.argmin(fitness))
        if fitness[iteration_best] < best_fitness:
            best_fitness = float(fitness[iteration_best])
            best = nests[iteration_best].copy()
        else:
            # Explicit elitism in case the best nest was in the abandoned set.
            worst_index = int(np.argmax(fitness))
            nests[worst_index] = best
            fitness[worst_index] = best_fitness
        history.append(best_fitness)

    return CuckooSearchResult(best, best_fitness, history, evaluations)
