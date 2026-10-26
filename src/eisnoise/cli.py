"""Command-line interface for the Monte Carlo noise study."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .reporting import create_plots
from .simulation import NoiseStudyConfig, run_study


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Monte Carlo EIS parameter-recovery study")
    result.add_argument("--output", type=Path, default=Path("noise_study_output"))
    result.add_argument("--replicates", type=int, default=200)
    result.add_argument("--population", type=int, default=20)
    result.add_argument("--iterations", type=int, default=300)
    result.add_argument("--restarts", type=int, default=3)
    result.add_argument("--workers", type=int, default=0, help="0 selects a conservative automatic value")
    result.add_argument("--low-noise-percent", type=float, default=8.0)
    result.add_argument("--baseline-noise-percent", type=float, default=1.0)
    result.add_argument(
        "--reuse-no-noise-from", type=Path,
        help="Reuse no-noise rows from an earlier replicate CSV and run only embedded noisy fits",
    )
    result.add_argument("--smoke", action="store_true", help="Run a quick pipeline validation")
    return result


def main() -> None:
    args = parser().parse_args()
    config = NoiseStudyConfig(
        replicates=args.replicates,
        population=args.population,
        iterations=args.iterations,
        restarts=args.restarts,
        workers=args.workers,
        low_noise_fraction=args.low_noise_percent / 100.0,
        baseline_noise_fraction=args.baseline_noise_percent / 100.0,
    )
    if args.smoke:
        config = replace(config, replicates=3, population=5, iterations=3, restarts=1, workers=1)
    replicates, summary = run_study(config, args.output, args.reuse_no_noise_from)
    plots = create_plots(config, replicates, args.output)
    print(f"Replicate results: {replicates}")
    print(f"Summary: {summary}")
    for plot in plots:
        print(f"Plot: {plot}")


if __name__ == "__main__":
    main()

