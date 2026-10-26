"""Create publication-ready diagnostic plots from a completed noise study."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .engine import ShieldModel, evaluate_model
from .simulation import NoiseStudyConfig, add_relative_complex_gaussian_noise, noise_profile


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tukey_outlier_count(values: list[float]) -> int:
    """Count points outside the same 1.5-IQR whiskers used by boxplot."""
    data = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(data, [25, 75])
    spread = q3 - q1
    lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
    return int(np.count_nonzero((data < lower) | (data > upper)))


def create_plots(config: NoiseStudyConfig, replicates_path: Path, output_directory: Path) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(replicates_path)
    frequency = config.frequency_hz()
    sigma = noise_profile(frequency, config)
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    rng = np.random.default_rng(config.seed)
    for column, model in enumerate(("cole", "paper")):
        ideal_fruit = evaluate_model(frequency, model, "none", config.reference_parameters[model])
        ideal_raw = ideal_fruit + ShieldModel(
            config.fixed_series_resistance_ohm, config.fixed_series_capacitance_f
        ).impedance(frequency)
        noisy_raw = add_relative_complex_gaussian_noise(ideal_raw, sigma, rng)
        axes[0, column].loglog(frequency, np.abs(ideal_raw), "k-", label="Ideal raw")
        axes[0, column].loglog(frequency, np.abs(noisy_raw), "o", ms=4, alpha=0.75, label="Example noisy raw")
        axes[0, column].set(title=f"{model.title()} magnitude", ylabel="|Z| (Ω)")
        axes[1, column].semilogx(frequency, np.angle(ideal_raw, deg=True), "k-")
        axes[1, column].semilogx(frequency, np.angle(noisy_raw, deg=True), "o", ms=4, alpha=0.75)
        axes[1, column].set(title=f"{model.title()} phase", xlabel="Frequency (Hz)", ylabel="Phase (°)")
        for axis in (axes[0, column], axes[1, column]):
            axis.axvspan(100, 300, color="tab:red", alpha=0.08)
            axis.grid(True, which="both", alpha=0.25)
        axes[0, column].legend()
    path = output_directory / "ideal_and_example_noisy_spectra.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    rng = np.random.default_rng(config.seed)
    fixed_path = ShieldModel(
        config.fixed_series_resistance_ohm, config.fixed_series_capacitance_f
    ).impedance(frequency)
    for column, model in enumerate(("cole", "paper")):
        ideal_fruit = evaluate_model(
            frequency, model, "none", config.reference_parameters[model]
        )
        ideal_raw = ideal_fruit + fixed_path
        noisy_deembedded = add_relative_complex_gaussian_noise(
            ideal_raw, sigma, rng
        ) - fixed_path
        axes[0, column].loglog(
            frequency, np.abs(ideal_fruit), "k-", label="Ideal de-embedded"
        )
        axes[0, column].loglog(
            frequency, np.abs(noisy_deembedded), "o", ms=4, alpha=0.75,
            label="Example noisy de-embedded",
        )
        axes[0, column].set(title=f"{model.title()} magnitude", ylabel="|Z| (Ω)")
        axes[1, column].semilogx(
            frequency, np.angle(ideal_fruit, deg=True), "k-"
        )
        axes[1, column].semilogx(
            frequency, np.angle(noisy_deembedded, deg=True), "o", ms=4, alpha=0.75
        )
        axes[1, column].set(
            title=f"{model.title()} phase", xlabel="Frequency (Hz)", ylabel="Phase (°)"
        )
        for axis in (axes[0, column], axes[1, column]):
            axis.axvspan(100, 300, color="tab:red", alpha=0.08)
            axis.grid(True, which="both", alpha=0.25)
        axes[0, column].legend()
    path = output_directory / "ideal_and_example_noisy_deembedded_spectra.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    for model in ("cole", "paper"):
        parameters = list(config.reference_parameters[model])
        figure, axes = plt.subplots(
            1, len(parameters), figsize=(3.2 * len(parameters), 4.2), constrained_layout=True
        )
        axes = np.atleast_1d(axes)
        for axis, parameter in zip(axes, parameters, strict=True):
            values = []
            labels = []
            for condition, label in (("no_noise", "No noise"),
                                     ("relative_complex_gaussian", "Relative Gaussian")):
                selected = [
                    float(row[f"relative_error_pct_{parameter}"])
                    for row in rows if row["model"] == model and row["condition"] == condition
                ]
                values.append(selected)
                labels.append(label)
            # Extreme optimizer failures can be orders of magnitude from the
            # central distribution. Keep Tukey whiskers readable and report
            # the omitted points explicitly instead of letting them flatten
            # the boxes against the axis.
            axis.boxplot(values, tick_labels=labels, showfliers=False)
            outlier_counts = [_tukey_outlier_count(group) for group in values]
            if any(outlier_counts):
                axis.text(
                    0.5, 0.98,
                    f"Outliers beyond whiskers: {outlier_counts[0]} / {outlier_counts[1]}",
                    transform=axis.transAxes, ha="center", va="top", fontsize=7,
                )
            axis.axhline(0, color="black", lw=1)
            axis.set(title=parameter, ylabel="Recovery error (%)")
            axis.tick_params(axis="x", rotation=20)
            axis.grid(True, axis="y", alpha=0.25)
        figure.suptitle(f"{model.title()} parameter recovery")
        path = output_directory / f"{model}_parameter_recovery.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

        noisy_rows = [
            row for row in rows
            if row["model"] == model and row["condition"] == "relative_complex_gaussian"
        ]
        matrix = np.asarray([
            [float(row[f"relative_error_pct_{parameter}"]) for parameter in parameters]
            for row in noisy_rows
        ])
        correlation = np.corrcoef(matrix, rowvar=False)
        figure, axis = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
        image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(parameters)), parameters, rotation=35, ha="right")
        axis.set_yticks(range(len(parameters)), parameters)
        axis.set_title(f"{model.title()} recovery-error correlations")
        for i in range(len(parameters)):
            for j in range(len(parameters)):
                axis.text(j, i, f"{correlation[i, j]:.2f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, label="Correlation")
        path = output_directory / f"{model}_parameter_correlations.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.semilogx(frequency, sigma * 100, "o-")
    axis.axvspan(100, 300, color="tab:red", alpha=0.08, label="Elevated-noise band")
    axis.set(xlabel="Frequency (Hz)", ylabel="RMS relative complex noise (%)",
             title="Applied frequency-dependent noise profile")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    path = output_directory / "noise_profile.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths

