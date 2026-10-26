"""Nested grouped cross-validation and reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from joblib import Parallel, delayed, parallel_config

from .config import TrainingConfig
from .data import Dataset, dataset_names, load_dataset, validate_common_rows


ProgressCallback = Callable[[str], None]
CLASS_ORDER = ("unripe", "ripe", "overripe")
LOG_PARAMETER_NAMES = frozenset({
    "r_inf", "delta_r", "tau_s", "r0", "r1", "r2", "c1_f",
    "c_e_f", "sigma", "r_e", "q_e",
})


@dataclass(frozen=True)
class RunSummary:
    run_directory: Path
    summary_csv: Path
    experiments_completed: int
    experiments_reused: int


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _labels(y: np.ndarray) -> tuple[str, ...]:
    present = {str(value) for value in y}
    if present == set(CLASS_ORDER):
        return CLASS_ORDER
    return tuple(sorted(present))


def _preprocessor(dataset: Dataset, variant: str) -> object:
    if variant == "standard":
        return StandardScaler()
    if variant == "robust":
        return RobustScaler()
    if variant != "log_standard":
        raise ValueError(f"Unknown preprocessing variant: {variant}")
    if dataset.is_raw:
        raise ValueError("log_standard is not valid for signed raw real/imaginary impedance")
    log_indices = [index for index, name in enumerate(dataset.features) if name in LOG_PARAMETER_NAMES]
    linear_indices = [index for index in range(len(dataset.features)) if index not in log_indices]
    if not log_indices:
        raise ValueError(f"No log-transformable circuit parameters found in {dataset.name}")
    if np.any(dataset.x[:, log_indices] <= 0):
        names = [dataset.features[index] for index in log_indices]
        raise ValueError(f"Non-positive value found in log-transform features: {names}")
    log_pipeline = Pipeline([
        ("log10", FunctionTransformer(np.log10, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    transformers = [("log_parameters", log_pipeline, log_indices)]
    if linear_indices:
        transformers.append(("linear_features", StandardScaler(), linear_indices))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def _make_pipeline(
    dataset: Dataset, preprocessing: str, activation: str,
    architecture: tuple[int, ...], alpha: float, config: TrainingConfig, seed: int,
) -> Pipeline:
    steps: list[tuple[str, object]] = [("preprocess", _preprocessor(dataset, preprocessing))]
    steps.append(("mlp", MLPClassifier(
        hidden_layer_sizes=architecture,
        activation=activation,
        solver=config.solver,
        alpha=alpha,
        max_iter=config.max_iterations,
        max_fun=config.max_function_evaluations,
        random_state=seed,
    )))
    return Pipeline(steps)


def _candidate_grid(
    dataset: Dataset, activation: str, config: TrainingConfig
) -> list[dict[str, object]]:
    architectures = config.raw_architectures if dataset.is_raw else config.circuit_architectures
    rng_seed = int(hashlib.sha256(
        f"{config.random_seed}:{dataset.name}:{activation}".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    candidates: list[dict[str, object]] = []
    # Random search in log(alpha), with each architecture represented before repeats.
    order = list(rng.permutation(len(architectures)))
    for index in range(config.random_candidates):
        if index and index % len(architectures) == 0:
            order = list(rng.permutation(len(architectures)))
        architecture = architectures[order[index % len(architectures)]]
        alpha = float(10.0 ** rng.uniform(np.log10(config.alpha_min), np.log10(config.alpha_max)))
        candidates.append({"hidden_layer_sizes": list(architecture), "alpha": alpha})
    return candidates


def _split_records(
    dataset: Dataset, config: TrainingConfig
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    splitter = StratifiedGroupKFold(
        n_splits=config.outer_folds, shuffle=True, random_state=config.random_seed
    )
    splits = list(splitter.split(dataset.x, dataset.y, dataset.groups))
    records = []
    for fold, (train, test) in enumerate(splits, start=1):
        train_groups = sorted(set(dataset.groups[train]))
        test_groups = sorted(set(dataset.groups[test]))
        if set(train_groups) & set(test_groups):
            raise RuntimeError(f"Banana leakage detected in outer fold {fold}")
        records.append({
            "outer_fold": fold,
            "train_bananas": train_groups,
            "test_bananas": test_groups,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_class_counts": dict(Counter(map(str, dataset.y[train]))),
            "test_class_counts": dict(Counter(map(str, dataset.y[test]))),
        })
    return splits, {"group_column": dataset.group_name, "outer_folds": records}


def _align_probabilities(model: Pipeline, x: np.ndarray, labels: tuple[str, ...]) -> np.ndarray:
    probability = model.predict_proba(x)
    classes = [str(value) for value in model.classes_]
    aligned = np.zeros((len(x), len(labels)), dtype=float)
    for source, label in enumerate(classes):
        aligned[:, labels.index(label)] = probability[:, source]
    return aligned


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray,
             labels: tuple[str, ...]) -> dict[str, float]:
    lexical_labels = tuple(sorted(labels))
    lexical_probability = probability[:, [labels.index(label) for label in lexical_labels]]
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, lexical_probability, labels=list(lexical_labels))),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    for index, label in enumerate(labels):
        result[f"precision_{label}"] = float(precision[index])
        result[f"recall_{label}"] = float(recall[index])
        result[f"f1_{label}"] = float(f1[index])
        result[f"support_{label}"] = int(support[index])
    return result


def _inner_score(dataset: Dataset, train_indices: np.ndarray, preprocessing: str, activation: str,
                 candidate: dict, config: TrainingConfig, outer_fold: int) -> float:
    x, y, groups = (
        dataset.x[train_indices], dataset.y[train_indices], dataset.groups[train_indices]
    )
    splitter = StratifiedGroupKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=config.random_seed + outer_fold,
    )
    scores = []
    for inner_fold, (inner_train, validation) in enumerate(splitter.split(x, y, groups), start=1):
        model = _make_pipeline(
            dataset, preprocessing, activation,
            tuple(candidate["hidden_layer_sizes"]), float(candidate["alpha"]),
            config, config.random_seed + outer_fold * 100 + inner_fold,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x[inner_train], y[inner_train])
        prediction = model.predict(x[validation])
        scores.append(f1_score(y[validation], prediction, average="macro", zero_division=0))
    return float(np.mean(scores))


def _select_candidate(dataset: Dataset, train: np.ndarray, preprocessing: str, activation: str,
                      candidates: list[dict], config: TrainingConfig,
                      outer_fold: int) -> tuple[dict, list[dict]]:
    workers = config.workers or max(1, min(4, (os.cpu_count() or 2) // 2))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        scores = Parallel(n_jobs=workers)(
            delayed(_inner_score)(dataset, train, preprocessing, activation, candidate, config, outer_fold)
            for candidate in candidates
        )
    trials = [
        {"candidate": index, **candidate, "mean_inner_macro_f1": score}
        for index, (candidate, score) in enumerate(zip(candidates, scores, strict=True), start=1)
    ]
    # Stable tie-break: smaller network, then stronger regularization.
    best = max(trials, key=lambda trial: (
        trial["mean_inner_macro_f1"],
        -sum(trial["hidden_layer_sizes"]),
        trial["alpha"],
    ))
    return {"hidden_layer_sizes": best["hidden_layer_sizes"], "alpha": best["alpha"]}, trials


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion(path: Path, y_true: np.ndarray, y_pred: np.ndarray,
                    labels: tuple[str, ...], title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
             xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True", title=title)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _dataset_description(name: str) -> dict[str, str]:
    physical = "practical" if name.endswith("_practical") else "all"
    base = name[: -(len(physical) + 1)]
    if base == "raw_spectrum":
        return {"representation": "raw_spectrum", "fruit_model": "", "electrode_model": "",
                "physical_variant": physical}
    fruit, electrode = base.split("_", 1)
    return {"representation": "circuit_parameters", "fruit_model": fruit,
            "electrode_model": electrode, "physical_variant": physical}


def _plot_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ordered = sorted(rows, key=lambda row: row["test_f1_macro"])
    labels = [
        f'{row["dataset"]} · {row["activation"]} · {row.get("preprocessing", "standard")}'
        for row in ordered
    ]
    scores = [row["test_f1_macro"] for row in ordered]
    errors = [row["fold_f1_macro_std"] for row in ordered]
    height = max(6.0, len(rows) * 0.31)
    fig, axis = plt.subplots(figsize=(10.5, height))
    positions = np.arange(len(rows))
    colors = ["#2a6fbb" if row["activation"] == "relu" else "#d97706" for row in ordered]
    axis.barh(positions, scores, xerr=errors, color=colors, alpha=0.9, capsize=2)
    axis.set(yticks=positions, yticklabels=labels, xlim=(0, 1), xlabel="Combined outer-fold macro F1",
             title="MLP model comparison (error bars: SD across outer folds)")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _run_experiment(dataset: Dataset, preprocessing: str, activation: str,
                    splits: list[tuple[np.ndarray, np.ndarray]], config: TrainingConfig,
                    directory: Path, progress: ProgressCallback) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    labels = _labels(dataset.y)
    candidates = _candidate_grid(dataset, activation, config)
    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selections: list[dict] = []
    oof_prediction = np.empty(len(dataset.y), dtype=object)
    oof_probability = np.zeros((len(dataset.y), len(labels)), dtype=float)
    for outer_fold, (train, test) in enumerate(splits, start=1):
        progress(
            f"{dataset.name} / {activation} / {preprocessing}: "
            f"outer fold {outer_fold}/{len(splits)}"
        )
        selected, trials = _select_candidate(
            dataset, train, preprocessing, activation, candidates, config, outer_fold
        )
        model = _make_pipeline(
            dataset, preprocessing, activation,
            tuple(selected["hidden_layer_sizes"]), float(selected["alpha"]),
            config, config.random_seed + outer_fold,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(dataset.x[train], dataset.y[train])
        prediction = model.predict(dataset.x[test])
        probability = _align_probabilities(model, dataset.x[test], labels)
        oof_prediction[test] = prediction
        oof_probability[test] = probability
        metrics = _metrics(dataset.y[test], prediction, probability, labels)
        fold_rows.append({
            "outer_fold": outer_fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_bananas": len(set(dataset.groups[train])),
            "test_bananas": len(set(dataset.groups[test])),
            "hidden_layer_sizes": str(tuple(selected["hidden_layer_sizes"])),
            "alpha": selected["alpha"],
            "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in caught),
            **metrics,
        })
        selections.append({
            "outer_fold": outer_fold,
            "selected": selected,
            "inner_trials": trials,
        })
        for local, row_index in enumerate(test):
            banana, date, capture = dataset.identifiers[row_index]
            row = {
                "row_index": int(row_index), "outer_fold": outer_fold,
                "banana_id": banana, "measurement_date": date, "capture_id": capture,
                "true_class": str(dataset.y[row_index]), "predicted_class": str(prediction[local]),
            }
            row.update({f"probability_{label}": float(probability[local, position])
                        for position, label in enumerate(labels)})
            prediction_rows.append(row)
    combined = _metrics(dataset.y, oof_prediction, oof_probability, labels)
    aggregate = {
        "dataset": dataset.name,
        "activation": activation,
        "preprocessing": preprocessing,
        **_dataset_description(dataset.name),
        "features": len(dataset.features),
        "rows": len(dataset.y),
        "bananas": len(set(dataset.groups)),
        "outer_convergence_warnings": sum(bool(row["convergence_warning"]) for row in fold_rows),
        **{f"test_{key}": value for key, value in combined.items()},
    }
    for metric in ("f1_macro", "balanced_accuracy", "accuracy", "log_loss"):
        values = np.asarray([row[metric] for row in fold_rows], dtype=float)
        aggregate[f"fold_{metric}_mean"] = float(values.mean())
        aggregate[f"fold_{metric}_std"] = float(values.std(ddof=1))
    _write_csv(directory / "fold_metrics.csv", fold_rows)
    _write_csv(directory / "predictions.csv", sorted(prediction_rows, key=lambda row: row["row_index"]))
    _write_json(directory / "selected_hyperparameters.json", selections)
    _write_json(directory / "metrics.json", aggregate)
    _plot_confusion(directory / "confusion_matrix.png", dataset.y, oof_prediction, labels,
                    f"{dataset.name} — MLP-{activation} — {preprocessing}")
    if config.fit_final_model:
        full_train = np.arange(len(dataset.y))
        selected, trials = _select_candidate(
            dataset, full_train, preprocessing, activation, candidates, config, 99
        )
        final_model = _make_pipeline(
            dataset, preprocessing, activation,
            tuple(selected["hidden_layer_sizes"]), float(selected["alpha"]),
            config, config.random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            final_model.fit(dataset.x, dataset.y)
        joblib.dump({
            "pipeline": final_model,
            "features": dataset.features,
            "classes": [str(value) for value in final_model.classes_],
            "dataset": dataset.name,
            "activation": activation,
            "preprocessing": preprocessing,
            "selected": selected,
            "note": "Fitted on all bananas after nested-CV evaluation; not used for test metrics.",
        }, directory / "final_model.joblib")
        _write_json(directory / "final_model_selection.json", {
            "selected": selected, "inner_trials": trials,
        })
    (directory / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return aggregate


def _existing_run(output_directory: Path, fingerprint: str) -> Path | None:
    candidates = sorted(output_directory.glob(f"run_*_{fingerprint}"))
    return candidates[-1] if candidates else None


def run_training(dataset_directory: str | Path, output_directory: str | Path,
                 config: TrainingConfig | None = None,
                 selected_datasets: Iterable[str] | None = None,
                 resume: bool = True, progress: ProgressCallback | None = None) -> RunSummary:
    config = config or TrainingConfig()
    progress = progress or (lambda message: None)
    dataset_directory = Path(dataset_directory).resolve()
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    names = list(selected_datasets) if selected_datasets is not None else dataset_names(dataset_directory)
    if not names:
        raise ValueError("No datasets selected")
    if min(config.outer_folds, config.inner_folds, config.random_candidates, config.max_iterations) < 1:
        raise ValueError("Fold counts, candidate count and maximum iterations must be positive")
    run_payload = json.dumps(
        {"configuration": config.as_dict(), "datasets": sorted(names)},
        sort_keys=True, separators=(",", ":"),
    )
    run_fingerprint = hashlib.sha256(run_payload.encode("utf-8")).hexdigest()[:10]
    run_directory = _existing_run(output_directory, run_fingerprint) if resume else None
    if run_directory is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_directory = output_directory / f"run_{stamp}_{run_fingerprint}"
        run_directory.mkdir(parents=True)
    manifest_bytes = (dataset_directory / "dataset_manifest.json").read_bytes()
    run_record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_directory": str(dataset_directory),
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "configuration_hash": config.fingerprint,
        "run_hash": run_fingerprint,
        "configuration": config.as_dict(),
        "datasets": names,
        "workers": config.workers or max(1, min(4, (os.cpu_count() or 2) // 2)),
        "note": "Candidate trials use bounded automatic parallelism; experiments run sequentially.",
    }
    config_path = run_directory / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing["configuration_hash"] != config.fingerprint:
            raise ValueError("Run directory configuration does not match this training request")
        if existing["dataset_manifest_sha256"] != run_record["dataset_manifest_sha256"]:
            raise ValueError("Prepared dataset changed; start a new run instead of resuming")
    else:
        _write_json(config_path, run_record)
    reference = load_dataset(dataset_directory, names[0])
    splits, split_plan = _split_records(reference, config)
    _write_json(run_directory / "split_plan.json", split_plan)
    summary_rows: list[dict] = []
    completed = reused = 0
    experiments_root = run_directory / "experiments"
    for name in names:
        dataset = load_dataset(dataset_directory, name)
        validate_common_rows(reference, dataset)
        preprocessing_variants = tuple(
            variant for variant in config.preprocessing_variants
            if not (dataset.is_raw and variant == "log_standard")
        )
        for activation in config.activations:
            for preprocessing in preprocessing_variants:
                experiment = experiments_root / name / activation / preprocessing
                metrics_path = experiment / "metrics.json"
                if resume and (experiment / "COMPLETE").is_file() and metrics_path.is_file():
                    progress(f"Reusing completed {name} / {activation} / {preprocessing}")
                    summary_rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                    reused += 1
                    continue
                result = _run_experiment(
                    dataset, preprocessing, activation, splits, config, experiment, progress
                )
                summary_rows.append(result)
                completed += 1
                _write_json(run_directory / "progress.json", {
                    "completed_this_invocation": completed,
                    "reused_this_invocation": reused,
                    "last_experiment": f"{name}/{activation}/{preprocessing}",
                })
    summary_rows.sort(key=lambda row: row["test_f1_macro"], reverse=True)
    summary_csv = run_directory / "summary.csv"
    _write_csv(summary_csv, summary_rows)
    _write_json(run_directory / "summary.json", summary_rows)
    _plot_summary(run_directory / "model_comparison.png", summary_rows)
    progress(f"Run complete: {run_directory}")
    return RunSummary(run_directory, summary_csv, completed, reused)

