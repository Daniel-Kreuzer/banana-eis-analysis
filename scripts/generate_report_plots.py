"""Generate report-ready circuit-fitting results figures from the batch database.

The script is deliberately read-only: it does not refit spectra or modify the
SQLite database.  It selects the P30/I800/R5 configuration and reconstructs
additional diagnostics from the stored spectra and model predictions.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/processed/circuit_fits.sqlite"
OUTPUT = ROOT / "output/circuit_fit_report_figures"
sys.path.insert(0, str(ROOT / "src"))

from eisfit.models import ShieldModel, parameter_specs  # noqa: E402


FRUIT_ORDER = ("cole", "paper")
ELECTRODE_ORDER = ("none", "capacitor", "warburg", "randles_cpe")
FRUIT_LABEL = {"cole": "Cole", "paper": "Simplified"}
ELECTRODE_LABEL = {
    "none": "None",
    "capacitor": "Capacitor",
    "warburg": "Warburg",
    "randles_cpe": "Randles/CPE",
}
COLORS = {
    "none": "#4C78A8",
    "capacitor": "#F58518",
    "warburg": "#54A24B",
    "randles_cpe": "#B279A2",
}
FRUIT_PARAMETERS = {
    "cole": ("r_inf", "delta_r", "tau_s", "alpha", "r0"),
    "paper": ("r1", "r2", "c1_f"),
}
PARAMETER_LABEL = {
    "r_inf": r"$R_\infty$ ($\Omega$)",
    "delta_r": r"$\Delta R$ ($\Omega$)",
    "tau_s": r"$\tau$ (s)",
    "alpha": r"$\alpha$",
    "r0": r"$R_0$ ($\Omega$)",
    "r1": r"$R_1$ ($\Omega$)",
    "r2": r"$R_2$ ($\Omega$)",
    "c1_f": r"$C_1$ (F)",
    "c_e_f": r"$C_e$ (F)",
    "sigma": r"$\sigma$ ($\Omega\,s^{-1/2}$)",
    "r_e": r"$R_e$ ($\Omega$)",
    "q_e": r"$Q_e$",
    "n_e": r"$n_e$",
}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def selected_config(connection: sqlite3.Connection) -> sqlite3.Row:
    rows = connection.execute("SELECT * FROM fit_configs").fetchall()
    matches = []
    for row in rows:
        config = json.loads(row["config_json"])
        if config.get("population") == 30 and config.get("iterations") == 800 and config.get("restarts") == 5:
            matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one P30/I800/R5 configuration, found {len(matches)}")
    return matches[0]


def load_rows(connection: sqlite3.Connection, config_id: int) -> list[dict]:
    query = """
        SELECT f.*, c.banana_id, c.measurement_date, c.frequency_json,
               c.real_json, c.imag_json, fc.config_json
        FROM fits f
        JOIN captures c ON c.id=f.capture_id
        JOIN fit_configs fc ON fc.id=f.config_id
        WHERE f.config_id=? AND f.status='complete' AND c.is_current=1
        ORDER BY c.measurement_date, c.banana_id, f.fruit_model, f.electrode_model
    """
    rows = []
    for source in connection.execute(query, (config_id,)):
        row = dict(source)
        frequency = np.asarray(json.loads(row["frequency_json"]), dtype=float)
        raw = np.asarray(json.loads(row["real_json"]), dtype=float) + 1j * np.asarray(
            json.loads(row["imag_json"]), dtype=float
        )
        indices = np.asarray(json.loads(row["selected_indices_json"]), dtype=int)
        frequency = frequency[indices]
        raw = raw[indices]
        predicted = np.asarray(json.loads(row["predicted_real_json"]), dtype=float) + 1j * np.asarray(
            json.loads(row["predicted_imag_json"]), dtype=float
        )
        config = json.loads(row["config_json"])
        shield = ShieldModel(config["fixed_resistance_ohm"], config["fixed_capacitance_f"])
        fixed = shield.impedance(frequency) if config["correct_fixed_path"] else np.zeros_like(raw)
        corrected = raw - fixed
        complete = predicted + fixed
        parameters = json.loads(row["parameters_json"])

        def rms_nrmse(measured: np.ndarray, fitted: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.abs(fitted - measured) ** 2)) /
                         max(np.sqrt(np.mean(np.abs(measured) ** 2)), 1e-15))

        magnitude_nrmse = float(
            np.sqrt(np.mean((np.abs(predicted) - np.abs(corrected)) ** 2)) /
            max(np.sqrt(np.mean(np.abs(corrected) ** 2)), 1e-15)
        )
        phase_error = np.angle(np.exp(1j * (np.angle(predicted) - np.angle(corrected))))
        phase_rmse_deg = float(np.rad2deg(np.sqrt(np.mean(phase_error ** 2))))
        residual = predicted - corrected
        relative_residual = np.abs(residual) / np.maximum(np.abs(corrected), 1e-15) * 100
        real_scale = max(float(np.sqrt(np.mean(corrected.real ** 2))), 1e-15)
        imag_scale = max(float(np.sqrt(np.mean(corrected.imag ** 2))), 1e-15)

        specs = parameter_specs(row["fruit_model"], row["electrode_model"], frequency, corrected)
        boundary_parameters = []
        for spec in specs:
            value = parameters[spec.name]
            if spec.scale == "linear":
                position = (value - spec.lower) / (spec.upper - spec.lower)
            else:
                position = (math.log10(value) - math.log10(spec.lower)) / (
                    math.log10(spec.upper) - math.log10(spec.lower)
                )
            if position <= 0.001 or position >= 0.999:
                boundary_parameters.append(spec.name)

        objectives = np.asarray(json.loads(row["run_objectives_json"]), dtype=float)
        restart_spread = float((objectives.max() - objectives.min()) /
                               max(abs(objectives.min()), 1e-15))
        row.update({
            "frequency": frequency,
            "corrected": corrected,
            "predicted": predicted,
            "raw": raw,
            "complete": complete,
            "parameters": parameters,
            "raw_nrmse": rms_nrmse(raw, complete),
            "magnitude_nrmse": magnitude_nrmse,
            "phase_rmse_deg": phase_rmse_deg,
            "relative_residual": relative_residual,
            "real_bias_pct": float(np.mean(residual.real) / real_scale * 100),
            "imag_bias_pct": float(np.mean(residual.imag) / imag_scale * 100),
            "boundary_hit": bool(boundary_parameters),
            "boundary_parameters": boundary_parameters,
            "restart_spread": restart_spread,
        })
        rows.append(row)
    return rows


def key_label(row: dict) -> str:
    return f"{FRUIT_LABEL[row['fruit_model']]}\n{ELECTRODE_LABEL[row['electrode_model']]}"


def combo_rows(rows: list[dict]) -> list[tuple[str, str, list[dict]]]:
    return [
        (fruit, electrode, [r for r in rows if r["fruit_model"] == fruit and r["electrode_model"] == electrode])
        for fruit in FRUIT_ORDER for electrode in ELECTRODE_ORDER
    ]


def boxplot(axis, data, labels, ylabel, title, log=False) -> None:
    artists = axis.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.65)
    for index, patch in enumerate(artists["boxes"]):
        patch.set_facecolor(COLORS[ELECTRODE_ORDER[index % 4]])
        patch.set_alpha(0.72)
    axis.set(title=title, ylabel=ylabel)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(True, axis="y")
    if log:
        axis.set_yscale("log")


def plot_fit_quality(rows: list[dict]) -> None:
    combos = combo_rows(rows)
    labels = [f"{FRUIT_LABEL[f]}\n{ELECTRODE_LABEL[e]}" for f, e, _ in combos]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.5))
    metrics = (
        ("nrmse", "De-embedded complex NRMSE (%)"),
        ("raw_nrmse", "Raw-domain complex NRMSE (%)"),
        ("magnitude_nrmse", "Magnitude NRMSE (%)"),
        ("phase_rmse_deg", "Phase RMSE (degrees)"),
    )
    for axis, (field, title) in zip(axes.flat, metrics, strict=True):
        multiplier = 1 if field == "phase_rmse_deg" else 100
        data = [np.asarray([r[field] * multiplier for r in group]) for _, _, group in combos]
        boxplot(axis, data, labels, title, title)
    fig.suptitle("Fit accuracy by fruit and electrode model", fontsize=14, fontweight="bold")
    save(fig, "figure_01_fit_accuracy")


def add_delta_aicc(rows: list[dict]) -> None:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["capture_id"]].append(row)
    for group in groups.values():
        minimum = min(float(row["aicc"]) for row in group)
        for row in group:
            row["delta_aicc"] = float(row["aicc"]) - minimum
            row["aicc_winner"] = abs(row["delta_aicc"]) < 1e-9


def plot_model_selection(rows: list[dict]) -> None:
    combos = combo_rows(rows)
    labels = [f"{FRUIT_LABEL[f]}\n{ELECTRODE_LABEL[e]}" for f, e, _ in combos]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    delta = [np.asarray([max(r["delta_aicc"], 1e-3) for r in group]) for _, _, group in combos]
    boxplot(axes[0], delta, labels, r"$\Delta$AICc", "Capture-level model support", log=True)
    wins = [100 * np.mean([r["aicc_winner"] for r in group]) for _, _, group in combos]
    bars = axes[1].bar(np.arange(8), wins, color=[COLORS[e] for _, e, _ in combos], alpha=0.8)
    axes[1].set_xticks(np.arange(8), labels, rotation=35, ha="right")
    axes[1].set(title="Lowest-AICc frequency", ylabel="Captures won (%)", ylim=(0, max(wins) * 1.18 + 1))
    axes[1].grid(True, axis="y")
    for bar, value in zip(bars, wins, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}%",
                     ha="center", va="bottom", fontsize=8)
    fig.suptitle("Model selection across matched captures", fontsize=14, fontweight="bold")
    save(fig, "figure_02_model_selection")


def plot_diagnostics(rows: list[dict]) -> None:
    combos = combo_rows(rows)
    labels = [f"{FRUIT_LABEL[f]}\n{ELECTRODE_LABEL[e]}" for f, e, _ in combos]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    boundary = [100 * np.mean([r["boundary_hit"] for r in group]) for _, _, group in combos]
    axes[0].bar(np.arange(8), boundary, color=[COLORS[e] for _, e, _ in combos], alpha=0.8)
    axes[0].set_xticks(np.arange(8), labels, rotation=35, ha="right")
    axes[0].set(title="Boundary-hit rate", ylabel="Fits with ≥1 boundary hit (%)")
    spread = [np.asarray([max(r["restart_spread"] * 100, 1e-6) for r in group]) for _, _, group in combos]
    boxplot(axes[1], spread, labels, "Relative objective spread (%)", "Restart-objective spread", log=True)
    bias_real = [np.asarray([r["real_bias_pct"] for r in group]) for _, _, group in combos]
    bias_imag = [np.asarray([r["imag_bias_pct"] for r in group]) for _, _, group in combos]
    x = np.arange(8)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].plot(x, [np.median(v) for v in bias_real], "o-", label="Real residual")
    axes[2].plot(x, [np.median(v) for v in bias_imag], "s-", label="Imaginary residual")
    axes[2].set_xticks(x, labels, rotation=35, ha="right")
    axes[2].set(title="Median signed residual bias", ylabel="Bias relative to component RMS (%)")
    axes[2].legend()
    axes[2].grid(True, axis="y")
    fig.suptitle("Fit diagnostics and optimisation reliability", fontsize=14, fontweight="bold")
    save(fig, "figure_03_fit_diagnostics")


def binned_curve(group: list[dict], edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frequencies = np.concatenate([r["frequency"] for r in group])
    residuals = np.concatenate([r["relative_residual"] for r in group])
    bins = np.digitize(frequencies, edges) - 1
    x, median, low, high = [], [], [], []
    for index in range(len(edges) - 1):
        values = residuals[bins == index]
        if values.size:
            x.append(math.sqrt(edges[index] * edges[index + 1]))
            median.append(np.median(values))
            low.append(np.percentile(values, 25))
            high.append(np.percentile(values, 75))
    return tuple(np.asarray(v) for v in (x, median, low, high))


def plot_residual_frequency(rows: list[dict]) -> None:
    minimum = min(float(np.min(r["frequency"])) for r in rows)
    maximum = max(float(np.max(r["frequency"])) for r in rows)
    edges = np.geomspace(minimum * 0.999, maximum * 1.001, 23)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for axis, fruit in zip(axes, FRUIT_ORDER, strict=True):
        for electrode in ELECTRODE_ORDER:
            group = [r for r in rows if r["fruit_model"] == fruit and r["electrode_model"] == electrode]
            x, median, low, high = binned_curve(group, edges)
            axis.semilogx(x, median, "o-", ms=3, lw=1.5, color=COLORS[electrode],
                          label=ELECTRODE_LABEL[electrode])
            axis.fill_between(x, low, high, color=COLORS[electrode], alpha=0.12)
        axis.axvspan(100, 300, color="#E45756", alpha=0.08, label="100–300 Hz" if fruit == "cole" else None)
        axis.set(title=f"{FRUIT_LABEL[fruit]} model", xlabel="Frequency (Hz)")
        axis.grid(True, which="both")
    axes[0].set_ylabel("De-embedded relative residual (%)")
    axes[0].legend(ncol=2)
    axes[1].legend(ncol=2)
    fig.suptitle("Frequency dependence of fitting residuals (median and IQR)", fontsize=14, fontweight="bold")
    save(fig, "figure_04_residuals_by_frequency")


def median_iqr(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.median(array)), float(np.percentile(array, 25)), float(np.percentile(array, 75))


def parameter_trend_figure(rows: list[dict], fruit: str) -> None:
    parameters = FRUIT_PARAMETERS[fruit]
    dates = sorted({r["measurement_date"] for r in rows})
    cols = 3
    rows_n = math.ceil(len(parameters) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(12.0, 3.7 * rows_n), squeeze=False)
    for axis, parameter in zip(axes.flat, parameters, strict=False):
        all_values = []
        for electrode in ELECTRODE_ORDER:
            medians, lows, highs = [], [], []
            for date in dates:
                values = [r["parameters"][parameter] for r in rows
                          if r["fruit_model"] == fruit and r["electrode_model"] == electrode
                          and r["measurement_date"] == date]
                median, low, high = median_iqr(values)
                medians.append(median); lows.append(low); highs.append(high)
                all_values.extend(values)
            x = np.arange(len(dates))
            axis.plot(x, medians, "o-", ms=3, color=COLORS[electrode], label=ELECTRODE_LABEL[electrode])
            axis.fill_between(x, lows, highs, color=COLORS[electrode], alpha=0.10)
        if parameter != "alpha" and min(all_values) > 0 and max(all_values) / min(all_values) >= 100:
            axis.set_yscale("log")
        axis.set(title=PARAMETER_LABEL[parameter])
        axis.set_xticks(np.arange(len(dates)), [d[5:] for d in dates], rotation=40, ha="right")
        axis.grid(True, which="both")
    for axis in axes.flat[len(parameters):]:
        axis.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{FRUIT_LABEL[fruit]} fruit-parameter trends (median and IQR across bananas)",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.13)
    number = "05" if fruit == "cole" else "06"
    name = "cole" if fruit == "cole" else "simplified"
    save(fig, f"figure_{number}_{name}_parameter_trends")


def plot_electrode_parameter_trends(rows: list[dict]) -> None:
    specs = (("capacitor", "c_e_f"), ("warburg", "sigma"),
             ("randles_cpe", "r_e"), ("randles_cpe", "q_e"), ("randles_cpe", "n_e"))
    dates = sorted({r["measurement_date"] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2))
    fruit_colors = {"cole": "#1F77B4", "paper": "#D62728"}
    for axis, (electrode, parameter) in zip(axes.flat, specs, strict=False):
        all_values = []
        for fruit in FRUIT_ORDER:
            medians, lows, highs = [], [], []
            for date in dates:
                values = [r["parameters"][parameter] for r in rows
                          if r["fruit_model"] == fruit and r["electrode_model"] == electrode
                          and r["measurement_date"] == date]
                median, low, high = median_iqr(values)
                medians.append(median); lows.append(low); highs.append(high); all_values.extend(values)
            x = np.arange(len(dates))
            axis.plot(x, medians, "o-", ms=3, color=fruit_colors[fruit], label=FRUIT_LABEL[fruit])
            axis.fill_between(x, lows, highs, color=fruit_colors[fruit], alpha=0.12)
        if parameter != "n_e" and min(all_values) > 0 and max(all_values) / min(all_values) >= 100:
            axis.set_yscale("log")
        axis.set(title=f"{ELECTRODE_LABEL[electrode]}: {PARAMETER_LABEL[parameter]}")
        axis.set_xticks(np.arange(len(dates)), [d[5:] for d in dates], rotation=40, ha="right")
        axis.grid(True, which="both")
    axes.flat[-1].remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Electrode-parameter trends (median and IQR across bananas)",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.13)
    save(fig, "figure_07_electrode_parameter_trends")


def ratio_annotation(log_ratio: float) -> str:
    ratio = 10 ** log_ratio
    if 0.5 <= ratio <= 2.0:
        return f"{(ratio - 1) * 100:+.1f}%"
    return f"×{ratio:.2g}" if ratio >= 1 else f"÷{1 / ratio:.2g}"


def plot_electrode_effect(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    columns = ELECTRODE_ORDER[1:]
    for axis, fruit in zip(axes, FRUIT_ORDER, strict=True):
        parameters = FRUIT_PARAMETERS[fruit]
        lookup = {(r["capture_id"], r["electrode_model"]): r for r in rows if r["fruit_model"] == fruit}
        matrix = np.full((len(parameters), len(columns)), np.nan)
        annotations = np.full(matrix.shape, "", dtype=object)
        for i, parameter in enumerate(parameters):
            for j, electrode in enumerate(columns):
                ratios = []
                for capture_id in {r["capture_id"] for r in rows if r["fruit_model"] == fruit}:
                    base = lookup.get((capture_id, "none")); altered = lookup.get((capture_id, electrode))
                    if base and altered:
                        ratios.append(math.log10(altered["parameters"][parameter] /
                                                 base["parameters"][parameter]))
                matrix[i, j] = np.median(ratios)
                annotations[i, j] = ratio_annotation(matrix[i, j])
        limit = max(0.25, float(np.nanmax(np.abs(matrix))))
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(np.arange(len(columns)), [ELECTRODE_LABEL[c] for c in columns], rotation=25, ha="right")
        axis.set_yticks(np.arange(len(parameters)), [PARAMETER_LABEL[p] for p in parameters])
        axis.set_title(f"{FRUIT_LABEL[fruit]} model")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, annotations[i, j], ha="center", va="center", fontsize=9,
                          color="white" if abs(matrix[i, j]) > 0.55 * limit else "black")
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(r"Median $\log_{10}(\theta_{electrode}/\theta_{none})$")
    fig.suptitle("Paired effect of electrode modelling on estimated fruit parameters",
                 fontsize=14, fontweight="bold")
    save(fig, "figure_08_electrode_effect_on_fruit_parameters")


def write_summary(rows: list[dict], config: sqlite3.Row) -> None:
    fields = ["fruit_model", "electrode_model", "n", "median_deembedded_nrmse_pct",
              "median_raw_nrmse_pct", "median_magnitude_nrmse_pct", "median_phase_rmse_deg",
              "median_delta_aicc", "aicc_win_rate_pct", "boundary_hit_rate_pct",
              "median_restart_objective_spread_pct"]
    with (OUTPUT / "fit_quality_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fruit, electrode, group in combo_rows(rows):
            writer.writerow({
                "fruit_model": FRUIT_LABEL[fruit],
                "electrode_model": ELECTRODE_LABEL[electrode],
                "n": len(group),
                "median_deembedded_nrmse_pct": np.median([r["nrmse"] for r in group]) * 100,
                "median_raw_nrmse_pct": np.median([r["raw_nrmse"] for r in group]) * 100,
                "median_magnitude_nrmse_pct": np.median([r["magnitude_nrmse"] for r in group]) * 100,
                "median_phase_rmse_deg": np.median([r["phase_rmse_deg"] for r in group]),
                "median_delta_aicc": np.median([r["delta_aicc"] for r in group]),
                "aicc_win_rate_pct": np.mean([r["aicc_winner"] for r in group]) * 100,
                "boundary_hit_rate_pct": np.mean([r["boundary_hit"] for r in group]) * 100,
                "median_restart_objective_spread_pct": np.median([r["restart_spread"] for r in group]) * 100,
            })
    note = f"""Circuit-fitting report figures

Database: {DB}
Configuration: {config['display_name']}
Complete current fits analysed: {len(rows)} ({len(rows) // 8} matched captures × 8 models)

Figure 01: de-embedded and raw complex NRMSE, magnitude NRMSE and circular phase RMSE.
Figure 02: capture-level delta AICc and lowest-AICc win frequency.
Figure 03: boundary hits, restart-objective spread and signed residual bias.
Figure 04: relative residuals binned by measured frequency; lines are medians and bands are IQRs.
Figures 05–06: fruit-parameter trends by date, separated by fruit model.
Figure 07: electrode-parameter trends by date.
Figure 08: paired parameter shifts caused by electrode modelling, relative to the no-electrode fit.

Important limitations
- Outliers are omitted from boxplot rendering but remain in all calculations.
- The residual-frequency plot bins actual measurements; it does not interpolate spectra.
- Day-one fits used a 1 kHz lower limit whereas later fits extended to 100 Hz. The parameter-trend
  figures therefore show the fitted study results but do not establish that every temporal change is
  biological. A true common-range parameter sensitivity analysis would require refitting.
- Runtime was not stored, so it cannot be reconstructed. Restart parameter vectors were not stored,
  so only objective consistency—not parameter consistency—can be assessed retrospectively.
"""
    (OUTPUT / "README.txt").write_text(note, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    style()
    connection = sqlite3.connect(DB)
    connection.row_factory = sqlite3.Row
    config = selected_config(connection)
    rows = load_rows(connection, config["id"])
    connection.close()
    if len(rows) % 8:
        raise RuntimeError("The selected results are not a complete set of eight models per capture")
    add_delta_aicc(rows)
    plot_fit_quality(rows)
    plot_model_selection(rows)
    plot_diagnostics(rows)
    plot_residual_frequency(rows)
    parameter_trend_figure(rows, "cole")
    parameter_trend_figure(rows, "paper")
    plot_electrode_parameter_trends(rows)
    plot_electrode_effect(rows)
    write_summary(rows, config)
    print(f"Generated report figures in {OUTPUT}")


if __name__ == "__main__":
    main()
