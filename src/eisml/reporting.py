"""Create concise, publication-ready figures from a completed MLP run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


CLASS_ORDER = ("unripe", "ripe", "overripe")
CLASS_LABELS = ("Unripe", "Ripe", "Overripe")
ACTIVATION_COLORS = {"relu": "#2878B5", "tanh": "#D97706"}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(row: dict, key: str) -> float:
    return float(row[key])


def _display_base(base: str) -> str:
    if base == "raw_spectrum":
        return "Raw spectrum"
    fruit, electrode = base.split("_", 1)
    electrode_names = {
        "none": "None", "capacitor": "Capacitor", "warburg": "Warburg",
        "randles_cpe": "Randles/CPE",
    }
    fruit_name = "Simplified" if fruit == "paper" else fruit.title()
    return f"{fruit_name} + {electrode_names[electrode]}"


def _base(dataset: str) -> str:
    for suffix in ("_practical", "_all"):
        if dataset.endswith(suffix):
            return dataset[:-len(suffix)]
    return dataset


def _variant(dataset: str) -> str:
    return "Practical" if dataset.endswith("_practical") else "All"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11,
        "axes.labelsize": 9.5, "figure.titlesize": 12, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.bbox": "tight",
    })


def _heatmap(rows: list[dict], destination: Path) -> None:
    order = [
        "cole_none", "cole_capacitor", "cole_warburg", "cole_randles_cpe",
        "paper_none", "paper_capacitor", "paper_warburg", "paper_randles_cpe",
        "raw_spectrum",
    ]
    all_columns = (("All", "relu"), ("All", "tanh"), ("Practical", "relu"), ("Practical", "tanh"))
    lookup: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (_base(row["dataset"]), _variant(row["dataset"]), row["activation"])
        if key not in lookup or _number(row, "test_f1_macro") > _number(lookup[key], "test_f1_macro"):
            lookup[key] = row
    present_bases = [base for base in order if any(key[0] == base for key in lookup)]
    columns = [column for column in all_columns
               if any((base, *column) in lookup for base in present_bases)]
    values = np.full((len(present_bases), len(columns)), np.nan)
    for row_index, base in enumerate(present_bases):
        for column_index, (variant, activation) in enumerate(columns):
            selected = lookup.get((base, variant, activation))
            if selected is not None:
                values[row_index, column_index] = _number(selected, "test_f1_macro")
    fig, axis = plt.subplots(figsize=(max(5.8, 1.55 * len(columns) + 2.8),
                                      max(2.8, 0.52 * len(present_bases) + 1.6)))
    palette = plt.get_cmap("Blues").copy()
    palette.set_bad("#eeeeee")
    image = axis.imshow(np.ma.masked_invalid(values), cmap=palette, vmin=0.50, vmax=0.82,
                        aspect="auto")
    axis.set_xticks(range(len(columns)),
                    [f"{variant}\n{activation}" for variant, activation in columns])
    axis.set_yticks(range(len(present_bases)), [_display_base(base) for base in present_bases])
    axis.set_xlabel("Physical feature set and activation function")
    axis.set_title("Ripeness-classification performance across model configurations")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                axis.text(column, row, "Not run", ha="center", va="center", color="#666666")
                continue
            selected = lookup[(present_bases[row], *columns[column])]
            preprocessing = selected.get("preprocessing", "standard")
            marker = {"standard": "S", "robust": "R", "log_standard": "L"}.get(preprocessing, "")
            axis.text(column, row, f"{value:.3f}\n{marker}", ha="center", va="center",
                      color="white" if value > 0.72 else "black", fontsize=9)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Outer-CV macro F1")
    fig.text(0.01, 0.01, "Best preprocessing shown per cell: S=standard, R=robust, L=log+standard. Practical: mass and signal path.",
             fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def _top_models(rows: list[dict], destination: Path) -> None:
    top = sorted(rows, key=lambda row: _number(row, "test_f1_macro"), reverse=True)[:10]
    top.reverse()
    labels = [f'{_display_base(_base(row["dataset"]))} · {_variant(row["dataset"])} · '
              f'{row["activation"]} · {row.get("preprocessing", "standard")}'
              for row in top]
    metrics = (
        ("test_f1_macro", "Macro F1", "#2878B5"),
        ("test_balanced_accuracy", "Balanced accuracy", "#54A24B"),
        ("test_accuracy", "Accuracy", "#D97706"),
    )
    y = np.arange(len(top))
    fig, axis = plt.subplots(figsize=(10.5, 6.4))
    offsets = (-0.22, 0, 0.22)
    for offset, (field, label, color) in zip(offsets, metrics, strict=True):
        axis.barh(y + offset, [_number(row, field) for row in top], height=0.20,
                  label=label, color=color)
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 0.84)
    axis.set_xlabel("Outer cross-validation score")
    axis.set_title(f"{len(top)} strongest MLP configurations")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def _best_class_performance(best: dict, destination: Path) -> None:
    precision = [_number(best, f"test_precision_{label}") for label in CLASS_ORDER]
    recall = [_number(best, f"test_recall_{label}") for label in CLASS_ORDER]
    f1 = [_number(best, f"test_f1_{label}") for label in CLASS_ORDER]
    x = np.arange(3)
    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    width = 0.24
    axis.bar(x - width, precision, width, label="Precision", color="#2878B5")
    axis.bar(x, recall, width, label="Recall", color="#54A24B")
    axis.bar(x + width, f1, width, label="F1", color="#D97706")
    axis.set_xticks(x, CLASS_LABELS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Best model performance by ripeness class")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    for offset, values in zip((-width, 0, width), (precision, recall, f1), strict=True):
        for position, value in enumerate(values):
            axis.text(position + offset, value + 0.018, f"{value:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def _best_confusion(run: Path, best: dict, destination: Path) -> np.ndarray:
    experiment = run / "experiments" / best["dataset"] / best["activation"]
    if best.get("preprocessing"):
        experiment = experiment / best["preprocessing"]
    predictions = _read_csv(experiment / "predictions.csv")
    actual = [row["true_class"] for row in predictions]
    predicted = [row["predicted_class"] for row in predictions]
    matrix = confusion_matrix(actual, predicted, labels=CLASS_ORDER)
    percentages = matrix / matrix.sum(axis=1, keepdims=True) * 100
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    axis.set_xticks(range(3), CLASS_LABELS)
    axis.set_yticks(range(3), CLASS_LABELS)
    axis.set_xlabel("Predicted ripeness")
    axis.set_ylabel("True ripeness")
    axis.set_title("Best model: grouped outer-CV predictions")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, f"{matrix[row, column]}\n({percentages[row, column]:.1f}%)",
                      ha="center", va="center",
                      color="white" if percentages[row, column] > 55 else "black")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Row percentage (%)")
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)
    return matrix


def _comparison_table(rows: list[dict], destination: Path) -> None:
    fields = [
        "rank", "dataset", "activation", "preprocessing", "physical_variant", "features",
        "macro_f1", "balanced_accuracy", "accuracy", "fold_macro_f1_sd", "log_loss",
    ]
    ordered = sorted(rows, key=lambda row: _number(row, "test_f1_macro"), reverse=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ordered, start=1):
            writer.writerow({
                "rank": rank, "dataset": row["dataset"], "activation": row["activation"],
                "preprocessing": row.get("preprocessing", "standard"),
                "physical_variant": row["physical_variant"], "features": row["features"],
                "macro_f1": f'{_number(row, "test_f1_macro"):.4f}',
                "balanced_accuracy": f'{_number(row, "test_balanced_accuracy"):.4f}',
                "accuracy": f'{_number(row, "test_accuracy"):.4f}',
                "fold_macro_f1_sd": f'{_number(row, "fold_f1_macro_std"):.4f}',
                "log_loss": f'{_number(row, "test_log_loss"):.4f}',
            })


def generate_report(run_directory: str | Path) -> Path:
    _style()
    run = Path(run_directory).resolve()
    rows = _read_csv(run / "summary.csv")
    if not rows:
        raise ValueError("The run summary is empty")
    output = run / "report_figures"
    output.mkdir(exist_ok=True)
    best = max(rows, key=lambda row: _number(row, "test_f1_macro"))
    _heatmap(rows, output / "figure_1_configuration_heatmap.png")
    _top_models(rows, output / "figure_2_top_models.png")
    _best_class_performance(best, output / "figure_3_best_model_class_metrics.png")
    matrix = _best_confusion(run, best, output / "figure_4_best_model_confusion_matrix.png")
    _comparison_table(rows, output / "table_1_all_model_results.csv")

    activation_means: dict[str, float | None] = {}
    physical_means: dict[str, float | None] = {}
    for activation in ("relu", "tanh"):
        values = [_number(row, "test_f1_macro") for row in rows if row["activation"] == activation]
        activation_means[activation] = float(np.mean(values)) if values else None
    for physical in ("all", "practical"):
        values = [_number(row, "test_f1_macro") for row in rows if row["physical_variant"] == physical]
        physical_means[physical] = float(np.mean(values)) if values else None
    majority = max(int(best[f"test_support_{label}"]) for label in CLASS_ORDER) / sum(
        int(best[f"test_support_{label}"]) for label in CLASS_ORDER
    )
    summary = {
        "best_dataset": best["dataset"], "best_activation": best["activation"],
        "best_preprocessing": best.get("preprocessing", "standard"),
        "best_macro_f1": _number(best, "test_f1_macro"),
        "best_accuracy": _number(best, "test_accuracy"),
        "best_balanced_accuracy": _number(best, "test_balanced_accuracy"),
        "best_fold_macro_f1_sd": _number(best, "fold_f1_macro_std"),
        "majority_class_accuracy": majority,
        "activation_mean_macro_f1": activation_means,
        "physical_variant_mean_macro_f1": physical_means,
        "best_confusion_matrix": matrix.tolist(),
    }
    (output / "report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    def display_mean(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "not evaluated"

    text = f"""# MLP result summary

- Best configuration: `{best['dataset']}` with `{best['activation']}` activation and
  `{best.get('preprocessing', 'standard')}` preprocessing.
- Outer-CV macro F1: **{_number(best, 'test_f1_macro'):.3f}**.
- Outer-CV accuracy: **{_number(best, 'test_accuracy'):.3f}**.
- Outer-CV balanced accuracy: **{_number(best, 'test_balanced_accuracy'):.3f}**.
- Across-fold macro-F1 SD: **{_number(best, 'fold_f1_macro_std'):.3f}**.
- Majority-class accuracy reference: **{majority:.3f}**.
- Mean macro F1 across all practical-feature configurations: **{display_mean(physical_means['practical'])}**;
  across all full-feature configurations: **{display_mean(physical_means['all'])}**.
- Mean macro F1 across tanh configurations: **{display_mean(activation_means['tanh'])}**;
  across ReLU configurations: **{display_mean(activation_means['relu'])}**.

The test predictions are out-of-fold predictions from banana-grouped nested cross-validation.
Accordingly, measurements from a banana used for testing were not present in that fold's training
or hyperparameter-selection data.
"""
    (output / "report_summary.md").write_text(text, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    arguments = parser.parse_args()
    print(generate_report(arguments.run_directory))


if __name__ == "__main__":
    main()
