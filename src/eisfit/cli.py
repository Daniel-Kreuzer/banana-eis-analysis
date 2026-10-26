"""Command-line entry point."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .cuckoo import CuckooSearchConfig
from .fit import FitConfig, fit_spectrum
from .io import load_spectrum
from .models import ELECTRODE_MODELS, FRUIT_MODELS, ShieldModel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit fruit EIS equivalent-circuit models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", help="fit one SensorPal workbook or CSV")
    fit.add_argument("input")
    fit.add_argument("--output-dir", default="fit_output")
    fit.add_argument("--fruit-model", choices=(*FRUIT_MODELS, "all"), default="all")
    fit.add_argument("--electrode-model", choices=(*ELECTRODE_MODELS, "all"), default="all")
    fit.add_argument("--restarts", type=int, default=5)
    fit.add_argument("--seed", type=int, default=20260923)
    fit.add_argument("--population", type=int, default=30)
    fit.add_argument("--iterations", type=int, default=800)
    fit.add_argument("--abandonment-probability", type=float, default=0.25)
    fit.add_argument("--step-scale", type=float, default=0.01)
    fit.add_argument("--objective", choices=("relative", "absolute"), default="relative")
    fit.add_argument("--no-shield-correction", action="store_true")
    fit.add_argument("--shield-r-ohm", type=float, default=2000.0)
    fit.add_argument("--shield-c-f", type=float, default=15e-9 * 470e-9 / (15e-9 + 470e-9))
    subparsers.add_parser("gui", help="open the interactive fit viewer")
    return parser


def _write_outputs(output_dir: Path, spectrum, results) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": spectrum.source,
        "metadata": spectrum.metadata,
        "frequency_hz": spectrum.frequency_hz.tolist(),
        "results": [result.to_dict(include_spectra=True) for result in results],
    }
    (output_dir / "fit_results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )

    with (output_dir / "fit_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        parameter_names = sorted({key for result in results for key in result.parameters})
        metric_names = sorted({key for result in results for key in result.metrics})
        fields = ["fruit_model", "electrode_model", "objective", "best_seed"] + metric_names + parameter_names
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
                "fruit_model": result.fruit_model,
                "electrode_model": result.electrode_model,
                "objective": result.objective,
                "best_seed": result.best_seed,
                **result.metrics,
                **result.parameters,
            }
            writer.writerow(row)

    with (output_dir / "fitted_spectra.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "fruit_model", "electrode_model", "frequency_hz",
            "measured_real_ohm", "measured_imag_ohm",
            "corrected_real_ohm", "corrected_imag_ohm",
            "predicted_real_ohm", "predicted_imag_ohm",
        ])
        for result in results:
            for index, frequency in enumerate(spectrum.frequency_hz):
                writer.writerow([
                    result.fruit_model, result.electrode_model, frequency,
                    spectrum.impedance_ohm[index].real, spectrum.impedance_ohm[index].imag,
                    result.corrected_impedance_ohm[index].real, result.corrected_impedance_ohm[index].imag,
                    result.predicted_impedance_ohm[index].real, result.predicted_impedance_ohm[index].imag,
                ])


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "gui":
        from .gui import launch

        launch()
        return 0
    spectrum = load_spectrum(args.input)
    fruit_models = FRUIT_MODELS if args.fruit_model == "all" else (args.fruit_model,)
    electrode_models = ELECTRODE_MODELS if args.electrode_model == "all" else (args.electrode_model,)
    config = FitConfig(
        restarts=args.restarts,
        seed=args.seed,
        objective=args.objective,
        correct_shield=not args.no_shield_correction,
        shield=ShieldModel(args.shield_r_ohm, args.shield_c_f),
        cuckoo=CuckooSearchConfig(
            population=args.population,
            iterations=args.iterations,
            abandonment_probability=args.abandonment_probability,
            step_scale=args.step_scale,
        ),
    )
    results = []
    for fruit_model in fruit_models:
        for electrode_model in electrode_models:
            print(f"Fitting {fruit_model} + {electrode_model} ...", flush=True)
            result = fit_spectrum(
                spectrum.frequency_hz,
                spectrum.impedance_ohm,
                fruit_model,
                electrode_model,
                config,
            )
            results.append(result)
            print(
                f"  objective={result.objective:.6g} "
                f"NRMSE={result.metrics['nrmse_complex']:.4g}",
                flush=True,
            )
    _write_outputs(Path(args.output_dir), spectrum, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
