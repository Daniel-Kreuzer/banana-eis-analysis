"""Analyse paired full-range and >1 kHz restricted circuit fits."""

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
SOURCE_DB = ROOT / "data/processed/circuit_fits.sqlite"
OUTPUT = ROOT / "output/low_frequency_sensitivity"
RESULT_DB = OUTPUT / "restricted_frequency_fits.sqlite"
sys.path.insert(0, str(ROOT / "src"))

from eisfit.models import ShieldModel, parameter_specs  # noqa: E402


FRUITS = ("cole", "paper")
ELECTRODES = ("none", "capacitor", "warburg", "randles_cpe")
FRUIT_LABEL = {"cole": "Cole", "paper": "Simplified"}
ELECTRODE_LABEL = {"none": "None", "capacitor": "Capacitor", "warburg": "Warburg", "randles_cpe": "Randles/CPE"}
COLORS = {"full": "#4C78A8", "restricted": "#F58518"}
PARAM_LABEL = {
    "r_inf": "R_inf", "delta_r": "Delta R", "tau_s": "tau", "alpha": "alpha", "r0": "R0",
    "r1": "R1", "r2": "R2", "c1_f": "C1", "c_e_f": "Ce", "sigma": "sigma",
    "r_e": "Re", "q_e": "Qe", "n_e": "ne",
}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.22, "grid.linewidth": 0.6,
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def original_config_id(connection: sqlite3.Connection) -> int:
    ids = []
    for row in connection.execute("SELECT * FROM fit_configs"):
        payload = json.loads(row["config_json"])
        if payload.get("population") == 30 and payload.get("iterations") == 800 and payload.get("restarts") == 5:
            ids.append(int(row["id"]))
    if len(ids) != 1:
        raise RuntimeError(f"Expected one original P30/I800/R5 configuration, found {len(ids)}")
    return ids[0]


def hit_flags(parameters: dict, bounds: list[dict], tolerance: float = 0.001) -> dict:
    hits = {}
    for bound in bounds:
        value = float(parameters[bound["name"]]); lower = float(bound["lower"]); upper = float(bound["upper"])
        if bound["scale"] == "linear":
            position = (value - lower) / (upper - lower)
        else:
            position = (math.log10(value) - math.log10(lower)) / (math.log10(upper) - math.log10(lower))
        hits[bound["name"]] = {"lower": position <= tolerance, "upper": position >= 1 - tolerance,
                                "normalized_position": position}
    return hits


def rms_nrmse(measured: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(predicted - measured) ** 2)) /
                 max(np.sqrt(np.mean(np.abs(measured) ** 2)), 1e-15))


def load_original(source: sqlite3.Connection, config_id: int) -> dict[tuple[int, str, str], dict]:
    query = """
      SELECT f.*,c.banana_id,c.measurement_date,c.frequency_json,c.real_json,c.imag_json,fc.config_json
      FROM fits f JOIN captures c ON c.id=f.capture_id JOIN fit_configs fc ON fc.id=f.config_id
      WHERE f.config_id=? AND c.is_current=1
    """
    result = {}
    for source_row in source.execute(query, (config_id,)):
        row = dict(source_row)
        frequency_all = np.asarray(json.loads(row["frequency_json"]), dtype=float)
        raw_all = np.asarray(json.loads(row["real_json"]), dtype=float) + 1j * np.asarray(json.loads(row["imag_json"]), dtype=float)
        indices = np.asarray(json.loads(row["selected_indices_json"]), dtype=int)
        frequency = frequency_all[indices]; raw = raw_all[indices]
        predicted = np.asarray(json.loads(row["predicted_real_json"]), dtype=float) + 1j * np.asarray(json.loads(row["predicted_imag_json"]), dtype=float)
        config = json.loads(row["config_json"])
        fixed = ShieldModel(config["fixed_resistance_ohm"], config["fixed_capacitance_f"]).impedance(frequency)
        corrected = raw - fixed if config["correct_fixed_path"] else raw.copy()
        parameters = json.loads(row["parameters_json"])
        bounds = [vars(spec) for spec in parameter_specs(row["fruit_model"], row["electrode_model"], frequency, corrected)]
        row.update({"frequency": frequency, "corrected": corrected, "predicted": predicted,
                    "parameters": parameters, "bounds": bounds, "hits": hit_flags(parameters, bounds)})
        result[(row["capture_id"], row["fruit_model"], row["electrode_model"])] = row
    return result


def load_restricted(connection: sqlite3.Connection) -> dict[tuple[int, str, str], dict]:
    result = {}
    for source_row in connection.execute("SELECT * FROM restricted_fits"):
        row = dict(source_row)
        if row["status"] == "complete":
            row.update({
                "frequency": np.asarray(json.loads(row["frequency_json"]), dtype=float),
                "corrected": np.asarray(json.loads(row["measured_corrected_real_json"]), dtype=float) +
                             1j * np.asarray(json.loads(row["measured_corrected_imag_json"]), dtype=float),
                "predicted": np.asarray(json.loads(row["predicted_real_json"]), dtype=float) +
                             1j * np.asarray(json.loads(row["predicted_imag_json"]), dtype=float),
                "parameters": json.loads(row["parameters_json"]),
                "bounds": json.loads(row["bounds_json"]),
                "hits": json.loads(row["boundary_hits_json"]),
            })
        result[(row["capture_id"], row["fruit_model"], row["electrode_model"])] = row
    return result


def pair_rows(original: dict, restricted: dict) -> list[dict]:
    pairs = []
    for key, reduced in restricted.items():
        if reduced["status"] != "complete" or key not in original or original[key]["status"] != "complete":
            continue
        full = original[key]
        mask = full["frequency"] > 1000.0
        shared_frequency = full["frequency"][mask]
        if len(shared_frequency) != len(reduced["frequency"]) or not np.allclose(shared_frequency, reduced["frequency"], rtol=1e-8, atol=1e-6):
            raise RuntimeError(f"Shared frequency mismatch for {key}")
        shared_measured = full["corrected"][mask]
        if not np.allclose(shared_measured, reduced["corrected"], rtol=1e-8, atol=1e-6):
            raise RuntimeError(f"Shared measurement mismatch for {key}")
        pairs.append({
            "capture_id": key[0], "fruit_model": key[1], "electrode_model": key[2],
            "banana_id": full["banana_id"], "measurement_date": full["measurement_date"],
            "frequency": shared_frequency, "shared_measured": shared_measured,
            "full_shared_predicted": full["predicted"][mask], "restricted_shared_predicted": reduced["predicted"],
            "full_shared_nrmse": rms_nrmse(shared_measured, full["predicted"][mask]),
            "restricted_shared_nrmse": rms_nrmse(shared_measured, reduced["predicted"]),
            "full_domain_nrmse": float(full["nrmse"]),
            "restricted_domain_nrmse": float(reduced["fitting_domain_nrmse"]),
            "full_aicc": float(full["aicc"]), "restricted_aicc": float(reduced["aicc"]),
            "full_parameters": full["parameters"], "restricted_parameters": reduced["parameters"],
            "full_bounds": full["bounds"], "restricted_bounds": reduced["bounds"],
            "full_hits": full["hits"], "restricted_hits": reduced["hits"],
            "full_any_hit": any(v["lower"] or v["upper"] for v in full["hits"].values()),
            "restricted_any_hit": any(v["lower"] or v["upper"] for v in reduced["hits"].values()),
        })
    groups = defaultdict(list)
    for pair in pairs:
        groups[(pair["capture_id"], "full")].append(pair)
        groups[(pair["capture_id"], "restricted")].append(pair)
    for (_, domain), group in groups.items():
        field = f"{domain}_aicc"
        minimum = min(p[field] for p in group)
        for pair in group:
            pair[f"{domain}_delta_aicc"] = pair[field] - minimum
            pair[f"{domain}_aicc_winner"] = abs(pair[f"{domain}_delta_aicc"]) < 1e-9
    return pairs


def combo_group(pairs: list[dict], fruit: str, electrode: str) -> list[dict]:
    return [p for p in pairs if p["fruit_model"] == fruit and p["electrode_model"] == electrode]


def combo_label(fruit: str, electrode: str) -> str:
    return f"{FRUIT_LABEL[fruit]} + {ELECTRODE_LABEL[electrode]}"


def export_metrics(pairs: list[dict]) -> None:
    fields = ["capture_id", "banana_id", "measurement_date", "fruit_model", "electrode_model", "n_points_shared",
              "full_domain_nrmse", "restricted_domain_nrmse", "full_shared_nrmse", "restricted_shared_nrmse",
              "shared_nrmse_change", "full_aicc", "restricted_aicc", "full_delta_aicc", "restricted_delta_aicc",
              "full_aicc_winner", "restricted_aicc_winner", "full_any_boundary_hit", "restricted_any_boundary_hit"]
    with (OUTPUT / "paired_fit_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for p in pairs:
            writer.writerow({
                "capture_id": p["capture_id"], "banana_id": p["banana_id"], "measurement_date": p["measurement_date"],
                "fruit_model": FRUIT_LABEL[p["fruit_model"]], "electrode_model": ELECTRODE_LABEL[p["electrode_model"]],
                "n_points_shared": len(p["frequency"]), "full_domain_nrmse": p["full_domain_nrmse"],
                "restricted_domain_nrmse": p["restricted_domain_nrmse"], "full_shared_nrmse": p["full_shared_nrmse"],
                "restricted_shared_nrmse": p["restricted_shared_nrmse"],
                "shared_nrmse_change": p["restricted_shared_nrmse"] - p["full_shared_nrmse"],
                "full_aicc": p["full_aicc"], "restricted_aicc": p["restricted_aicc"],
                "full_delta_aicc": p["full_delta_aicc"], "restricted_delta_aicc": p["restricted_delta_aicc"],
                "full_aicc_winner": p["full_aicc_winner"], "restricted_aicc_winner": p["restricted_aicc_winner"],
                "full_any_boundary_hit": p["full_any_hit"], "restricted_any_boundary_hit": p["restricted_any_hit"],
            })


def export_parameters(pairs: list[dict]) -> None:
    fields = ["capture_id", "banana_id", "measurement_date", "fruit_model", "electrode_model", "parameter",
              "full_value", "restricted_value", "log10_restricted_over_full", "full_lower_bound", "full_upper_bound",
              "restricted_lower_bound", "restricted_upper_bound", "lower_bound_changed", "upper_bound_changed",
              "full_lower_hit", "full_upper_hit", "restricted_lower_hit", "restricted_upper_hit"]
    with (OUTPUT / "paired_parameters_bounds_and_hits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for p in pairs:
            full_bounds = {b["name"]: b for b in p["full_bounds"]}; restricted_bounds = {b["name"]: b for b in p["restricted_bounds"]}
            for name in restricted_bounds:
                full_value = p["full_parameters"][name]; reduced_value = p["restricted_parameters"][name]
                fb, rb = full_bounds[name], restricted_bounds[name]
                writer.writerow({
                    "capture_id": p["capture_id"], "banana_id": p["banana_id"], "measurement_date": p["measurement_date"],
                    "fruit_model": FRUIT_LABEL[p["fruit_model"]], "electrode_model": ELECTRODE_LABEL[p["electrode_model"]],
                    "parameter": name, "full_value": full_value, "restricted_value": reduced_value,
                    "log10_restricted_over_full": math.log10(reduced_value / full_value),
                    "full_lower_bound": fb["lower"], "full_upper_bound": fb["upper"],
                    "restricted_lower_bound": rb["lower"], "restricted_upper_bound": rb["upper"],
                    "lower_bound_changed": not math.isclose(fb["lower"], rb["lower"], rel_tol=1e-12, abs_tol=0),
                    "upper_bound_changed": not math.isclose(fb["upper"], rb["upper"], rel_tol=1e-12, abs_tol=0),
                    "full_lower_hit": p["full_hits"][name]["lower"], "full_upper_hit": p["full_hits"][name]["upper"],
                    "restricted_lower_hit": p["restricted_hits"][name]["lower"],
                    "restricted_upper_hit": p["restricted_hits"][name]["upper"],
                })


def export_day_one(original: dict, day_one: str) -> None:
    fields = ["capture_id", "banana_id", "measurement_date", "fruit_model", "electrode_model", "parameter",
              "value", "lower_bound", "upper_bound", "lower_hit", "upper_hit"]
    with (OUTPUT / "day_one_parameter_bounds_and_hits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in original.values():
            if row["measurement_date"] != day_one or row["status"] != "complete": continue
            for bound in row["bounds"]:
                name = bound["name"]; hit = row["hits"][name]
                writer.writerow({"capture_id": row["capture_id"], "banana_id": row["banana_id"],
                                 "measurement_date": day_one, "fruit_model": FRUIT_LABEL[row["fruit_model"]],
                                 "electrode_model": ELECTRODE_LABEL[row["electrode_model"]], "parameter": name,
                                 "value": row["parameters"][name], "lower_bound": bound["lower"],
                                 "upper_bound": bound["upper"], "lower_hit": hit["lower"], "upper_hit": hit["upper"]})


def export_failures(restricted: dict) -> None:
    fields = ["capture_id", "banana_id", "measurement_date", "fruit_model", "electrode_model", "status", "error_message"]
    with (OUTPUT / "unsuccessful_fits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in restricted.values():
            if row["status"] != "complete":
                writer.writerow({field: row.get(field, "") for field in fields})


def summary_rows(pairs: list[dict]) -> list[dict]:
    summaries = []
    for fruit in FRUITS:
        for electrode in ELECTRODES:
            group = combo_group(pairs, fruit, electrode)
            full = np.asarray([p["full_shared_nrmse"] * 100 for p in group]); reduced = np.asarray([p["restricted_shared_nrmse"] * 100 for p in group])
            summaries.append({
                "fruit_model": FRUIT_LABEL[fruit], "electrode_model": ELECTRODE_LABEL[electrode], "n": len(group),
                "full_mean_pct": full.mean(), "full_median_pct": np.median(full),
                "full_q1_pct": np.percentile(full, 25), "full_q3_pct": np.percentile(full, 75),
                "restricted_mean_pct": reduced.mean(), "restricted_median_pct": np.median(reduced),
                "restricted_q1_pct": np.percentile(reduced, 25), "restricted_q3_pct": np.percentile(reduced, 75),
                "median_paired_change_pct_points": np.median(reduced - full),
                "improved_proportion_pct": np.mean(reduced < full) * 100,
                "full_any_hit_pct": np.mean([p["full_any_hit"] for p in group]) * 100,
                "restricted_any_hit_pct": np.mean([p["restricted_any_hit"] for p in group]) * 100,
                "full_aicc_win_pct": np.mean([p["full_aicc_winner"] for p in group]) * 100,
                "restricted_aicc_win_pct": np.mean([p["restricted_aicc_winner"] for p in group]) * 100,
            })
    with (OUTPUT / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    return summaries


def plot_shared_nrmse(pairs: list[dict]) -> None:
    labels = [combo_label(f, e).replace(" + ", "\n") for f in FRUITS for e in ELECTRODES]
    full_medians, restricted_medians, full_errors, restricted_errors = [], [], [], []
    for fruit, electrode in [(f, e) for f in FRUITS for e in ELECTRODES]:
        group = combo_group(pairs, fruit, electrode)
        full = np.asarray([p["full_shared_nrmse"] * 100 for p in group])
        restricted = np.asarray([p["restricted_shared_nrmse"] * 100 for p in group])
        for values, medians, errors in ((full, full_medians, full_errors),
                                        (restricted, restricted_medians, restricted_errors)):
            median = float(np.median(values)); q1, q3 = np.percentile(values, [25, 75])
            medians.append(median); errors.append((median - q1, q3 - median))
    x = np.arange(len(labels)); width = 0.36
    fig, axis = plt.subplots(figsize=(12.0, 5.0))
    axis.bar(x-width/2, full_medians, width, yerr=np.asarray(full_errors).T,
             capsize=3, label="Full-range fit", color=COLORS["full"])
    axis.bar(x+width/2, restricted_medians, width, yerr=np.asarray(restricted_errors).T,
             capsize=3, label=">1 kHz fit", color=COLORS["restricted"])
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylabel("Shared-band de-embedded NRMSE (%)")
    axis.set_title("Actual shared-band fitting error (median with interquartile range)")
    axis.grid(True, axis="y"); axis.legend()
    save(fig, "figure_01_paired_shared_band_nrmse")


def plot_model_comparison(pairs: list[dict]) -> None:
    labels = [combo_label(f, e).replace(" + ", "\n") for f in FRUITS for e in ELECTRODES]
    fig, axis = plt.subplots(figsize=(11.0, 4.5))
    x = np.arange(8); width = 0.36
    full = [np.mean([p["full_aicc_winner"] for p in combo_group(pairs, f, e)]) * 100 for f in FRUITS for e in ELECTRODES]
    reduced = [np.mean([p["restricted_aicc_winner"] for p in combo_group(pairs, f, e)]) * 100 for f in FRUITS for e in ELECTRODES]
    axis.bar(x-width/2, full, width, label="Full-range selection", color=COLORS["full"])
    axis.bar(x+width/2, reduced, width, label="Restricted >1 kHz selection", color=COLORS["restricted"])
    axis.set_xticks(x, labels, rotation=35, ha="right"); axis.set_ylabel("Measurements with lowest AICc (%)")
    axis.set_title("AICc winner frequency within each frequency selection"); axis.grid(True, axis="y"); axis.legend()
    save(fig, "figure_03_aicc_winner_rates")


def boundary_rates(pairs: list[dict], fruit: str, direction: str) -> tuple[list[str], np.ndarray]:
    labels, values = [], []
    for electrode in ELECTRODES:
        group = combo_group(pairs, fruit, electrode)
        names = list(group[0]["restricted_hits"])
        for name in names:
            labels.append(f"{ELECTRODE_LABEL[electrode]}: {PARAM_LABEL[name]}")
            values.append([
                np.mean([p["full_hits"][name][direction] for p in group]) * 100,
                np.mean([p["restricted_hits"][name][direction] for p in group]) * 100,
            ])
    return labels, np.asarray(values)


def plot_boundaries(pairs: list[dict]) -> None:
    for fruit in FRUITS:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.4))
        for col, direction in enumerate(("lower", "upper")):
            labels, matrix = boundary_rates(pairs, fruit, direction)
            image = axes[col].imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
            axes[col].set_xticks([0, 1], ["Full", ">1 kHz"])
            axes[col].set_yticks(np.arange(len(labels)), labels)
            axes[col].set_title(f"{direction.capitalize()}-bound hits")
            for i in range(len(labels)):
                for j in range(2):
                    axes[col].text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center",
                                   fontsize=8, color="white" if matrix[i,j] > 55 else "black")
        fig.suptitle(f"{FRUIT_LABEL[fruit]} parameter-bound hit rates", fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=0.19, right=0.90, top=0.90, bottom=0.08, wspace=0.55)
        color_axis = fig.add_axes([0.93, 0.18, 0.018, 0.64])
        fig.colorbar(image, cax=color_axis, label="Fits reaching bound (%)")
        stem = "figure_04a_cole_parameter_boundary_heatmap" if fruit == "cole" else "figure_04b_simplified_parameter_boundary_heatmap"
        for suffix in ("png", "pdf"):
            fig.savefig(OUTPUT / f"{stem}.{suffix}", bbox_inches="tight")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.0, 4.4))
    labels = [combo_label(f, e).replace(" + ", "\n") for f in FRUITS for e in ELECTRODES]
    full = [np.mean([p["full_any_hit"] for p in combo_group(pairs, f, e)]) * 100 for f in FRUITS for e in ELECTRODES]
    reduced = [np.mean([p["restricted_any_hit"] for p in combo_group(pairs, f, e)]) * 100 for f in FRUITS for e in ELECTRODES]
    x = np.arange(8); width = .36
    axis.bar(x-width/2, full, width, label="Full range", color=COLORS["full"])
    axis.bar(x+width/2, reduced, width, label=">1 kHz", color=COLORS["restricted"])
    axis.set_xticks(x, labels, rotation=35, ha="right"); axis.set_ylabel("Fits with any bound hit (%)")
    axis.set_title("Any-parameter boundary-hit rate"); axis.grid(True, axis="y"); axis.legend()
    save(fig, "figure_05_any_boundary_hit_rates")


def plot_parameter_changes(pairs: list[dict]) -> None:
    for fruit, number in zip(FRUITS, (6, 7), strict=True):
        parameter_names = []
        for electrode in ELECTRODES:
            group = combo_group(pairs, fruit, electrode)
            for name in group[0]["restricted_hits"]:
                if name not in parameter_names:
                    parameter_names.append(name)
        cols = 3
        rows = math.ceil(len(parameter_names) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(12.3, 3.2 * rows), squeeze=False)
        for axis, name in zip(axes.flat, parameter_names, strict=False):
            labels, data = [], []
            for electrode in ELECTRODES:
                group = combo_group(pairs, fruit, electrode)
                if name not in group[0]["restricted_parameters"]:
                    continue
                labels.append(ELECTRODE_LABEL[electrode])
                data.append([math.log10(p["restricted_parameters"][name] / p["full_parameters"][name]) for p in group])
            axis.axhline(0, color="#555555", lw=0.9)
            box = axis.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
            for patch in box["boxes"]:
                patch.set_facecolor("#B3CDE3")
            axis.set_title(PARAM_LABEL[name])
            axis.set_ylabel("log10 fold change")
            axis.tick_params(axis="x", rotation=25)
            axis.grid(True, axis="y")
        for axis in axes.flat[len(parameter_names):]:
            axis.set_visible(False)
        fig.suptitle(
            f"{FRUIT_LABEL[fruit]} parameter changes: restricted estimate relative to full-range estimate\n"
            "0 = unchanged, +1 = 10x larger, -1 = 10x smaller",
            fontsize=13, fontweight="bold")
        save(fig, f"figure_{number:02d}_{fruit}_parameter_changes")


def representative_capture(pairs: list[dict]) -> int:
    grouped = defaultdict(list)
    for p in pairs: grouped[p["capture_id"]].append((p["restricted_shared_nrmse"] - p["full_shared_nrmse"]) * 100)
    grouped = {capture: values for capture, values in grouped.items() if len(values) == 8}
    if not grouped:
        raise RuntimeError("No capture has all eight paired fits")
    per_capture = {capture: float(np.median(values)) for capture, values in grouped.items()}
    target = float(np.median(list(per_capture.values())))
    return min(per_capture, key=lambda capture: abs(per_capture[capture] - target))


def plot_representative(pairs: list[dict], original: dict) -> int:
    capture_id = representative_capture(pairs)
    selected = [p for p in pairs if p["capture_id"] == capture_id]
    selected.sort(key=lambda p: (FRUITS.index(p["fruit_model"]), ELECTRODES.index(p["electrode_model"])))
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.0), sharex=True)
    for axis, p in zip(axes.flat, selected, strict=True):
        full = original[(capture_id, p["fruit_model"], p["electrode_model"])]
        axis.loglog(full["frequency"], np.abs(full["corrected"]), "o", ms=2.4, color="black", label="Measurement")
        axis.loglog(full["frequency"], np.abs(full["predicted"]), "-", lw=1.4, color=COLORS["full"], label="Full fit")
        axis.loglog(p["frequency"], np.abs(p["restricted_shared_predicted"]), "-", lw=1.4,
                    color=COLORS["restricted"], label=">1 kHz fit")
        axis.axvspan(100, 1000, color="#E45756", alpha=.08)
        axis.set_title(combo_label(p["fruit_model"], p["electrode_model"])); axis.grid(True, which="both")
    axes[0,0].set_ylabel("|Z| (ohm)"); axes[1,0].set_ylabel("|Z| (ohm)")
    for axis in axes[1,:]: axis.set_xlabel("Frequency (Hz)")
    handles, labels = axes[0,0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=3)
    first = selected[0]
    fig.suptitle(f"Representative paired fits: {first['banana_id']} on {first['measurement_date']}",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=.12)
    save(fig, "figure_08_representative_fitted_responses")

    fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.0), sharex=True, sharey=True)
    for axis, p in zip(axes.flat, selected, strict=True):
        full = original[(capture_id, p["fruit_model"], p["electrode_model"])]
        axis.semilogx(full["frequency"], np.angle(full["corrected"], deg=True), "o", ms=2.4,
                      color="black", label="Measurement")
        axis.semilogx(full["frequency"], np.angle(full["predicted"], deg=True), "-", lw=1.4,
                      color=COLORS["full"], label="Full fit")
        axis.semilogx(p["frequency"], np.angle(p["restricted_shared_predicted"], deg=True), "-", lw=1.4,
                      color=COLORS["restricted"], label=">1 kHz fit")
        axis.axvspan(100, 1000, color="#E45756", alpha=.08)
        axis.set_title(combo_label(p["fruit_model"], p["electrode_model"])); axis.grid(True, which="both")
    axes[0,0].set_ylabel("Phase (degrees)"); axes[1,0].set_ylabel("Phase (degrees)")
    for axis in axes[1,:]: axis.set_xlabel("Frequency (Hz)")
    handles, labels = axes[0,0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(f"Representative paired phase responses: {first['banana_id']} on {first['measurement_date']}",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=.12)
    save(fig, "figure_09_representative_phase_responses")
    return capture_id


def write_analysis_json(pairs: list[dict], summaries: list[dict], restricted: dict,
                        original: dict, day_one: str, representative: int) -> None:
    failures = [r for r in restricted.values() if r["status"] != "complete"]
    parameter_rows = []
    bounds_changed = 0; parameters_compared = 0
    for p in pairs:
        fb = {b["name"]: b for b in p["full_bounds"]}; rb = {b["name"]: b for b in p["restricted_bounds"]}
        for name in rb:
            parameters_compared += 1
            changed = not (math.isclose(fb[name]["lower"], rb[name]["lower"], rel_tol=1e-12) and
                           math.isclose(fb[name]["upper"], rb[name]["upper"], rel_tol=1e-12))
            bounds_changed += int(changed)
            parameter_rows.append({"fruit_model": p["fruit_model"], "electrode_model": p["electrode_model"],
                                   "parameter": name, "log_ratio": math.log10(p["restricted_parameters"][name] /
                                                                              p["full_parameters"][name])})
    day_one_rows = [r for r in original.values() if r["measurement_date"] == day_one and r["status"] == "complete"]
    day_one_any = sum(any(v["lower"] or v["upper"] for v in r["hits"].values()) for r in day_one_rows)
    payload = {
        "day_one": day_one, "paired_measurements_per_combination": len(pairs)//8,
        "total_paired_fits": len(pairs), "unsuccessful_restricted_fits": len(failures),
        "summary": summaries, "bounds_changed_parameter_rows": bounds_changed,
        "bounds_compared_parameter_rows": parameters_compared,
        "day_one_fit_count": len(day_one_rows), "day_one_any_boundary_hit_count": day_one_any,
        "day_one_any_boundary_hit_pct": day_one_any / max(len(day_one_rows), 1) * 100,
        "representative_capture_id": representative,
    }
    (OUTPUT / "analysis_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True); setup_style()
    source = sqlite3.connect(SOURCE_DB); source.row_factory = sqlite3.Row
    result = sqlite3.connect(RESULT_DB); result.row_factory = sqlite3.Row
    config_id = original_config_id(source)
    original = load_original(source, config_id); restricted = load_restricted(result)
    day_one = result.execute("SELECT value FROM study_metadata WHERE key='day_one_date'").fetchone()[0]
    pairs = pair_rows(original, restricted)
    export_metrics(pairs); export_parameters(pairs); export_day_one(original, day_one); export_failures(restricted)
    summaries = summary_rows(pairs)
    plot_shared_nrmse(pairs); plot_model_comparison(pairs); plot_boundaries(pairs); plot_parameter_changes(pairs)
    representative = plot_representative(pairs, original)
    write_analysis_json(pairs, summaries, restricted, original, day_one, representative)
    result.close(); source.close()
    print(f"Analysed {len(pairs)} paired fits in {OUTPUT}")


if __name__ == "__main__":
    main()
