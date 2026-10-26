"""Parallel fitting jobs and deterministic frequency selection."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np

from .configuration import BatchConfig
from .database import Database
from .engine import ELECTRODE_MODELS, FRUIT_MODELS, fit_spectrum


def automatic_workers() -> int:
    """Use about half the logical CPUs, reserve capacity, and avoid excess RAM use."""
    cores = os.cpu_count() or 2
    return max(1, min(4, cores // 2, max(1, cores - 2)))


def selected_indices(count: int, requested: int | None) -> np.ndarray:
    if requested is None or requested >= count:
        return np.arange(count, dtype=int)
    indices = np.rint(np.linspace(0, count - 1, requested)).astype(int)
    return np.unique(indices)


def _fit_job(capture: dict, config_payload: dict, fruit: str, electrode: str) -> dict:
    config = BatchConfig(**config_payload)
    frequency = np.asarray(json.loads(capture["frequency_json"]), dtype=float)
    impedance = np.asarray(json.loads(capture["real_json"]), dtype=float) + 1j * np.asarray(
        json.loads(capture["imag_json"]), dtype=float
    )
    indices = selected_indices(len(frequency), config.frequency_points)
    result = fit_spectrum(
        frequency[indices], impedance[indices], fruit, electrode, config.fit_config()
    )
    payload = result.to_dict(include_spectra=True)
    return {
        "capture_id": capture["id"], "fruit": fruit, "electrode": electrode,
        "selected_indices": indices.tolist(), "result": payload,
    }


def run_jobs(database: Database, config: BatchConfig, refit: bool = False,
             progress: Callable[[int, int, str], None] | None = None) -> tuple[int, int]:
    config_id = database.add_config(config)
    combinations = [(fruit, electrode) for fruit in FRUIT_MODELS for electrode in ELECTRODE_MODELS]
    jobs = database.pending_jobs(config_id, combinations, refit)
    total, completed, failed = len(jobs), 0, 0
    if total == 0:
        return 0, 0
    workers = automatic_workers()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for capture, _, fruit, electrode in jobs:
            database.mark_running(capture["id"], config_id, fruit, electrode)
            future = executor.submit(_fit_job, capture, config.payload(), fruit, electrode)
            futures[future] = (capture, fruit, electrode)
        for future in as_completed(futures):
            capture, fruit, electrode = futures[future]
            completed += 1
            try:
                payload = future.result()
                database.save_result(
                    capture["id"], config_id, fruit, electrode,
                    payload["result"], payload["selected_indices"],
                )
                message = f"{capture['banana_id']} {capture['measurement_date']}: {fruit} + {electrode}"
            except Exception as exc:
                failed += 1
                database.save_failure(capture["id"], config_id, fruit, electrode, str(exc))
                message = f"Failed: {capture['filename']} — {fruit} + {electrode}"
            if progress:
                progress(completed, total, message)
    return completed - failed, failed

