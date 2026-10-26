"""Manifest-driven access to prepared ML datasets."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Dataset:
    name: str
    features: tuple[str, ...]
    target_name: str
    group_name: str
    identifiers: tuple[tuple[str, str, str], ...]
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray

    @property
    def is_raw(self) -> bool:
        return self.name.startswith("raw_spectrum_")


def load_manifest(dataset_directory: str | Path) -> dict:
    path = Path(dataset_directory) / "dataset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_names(dataset_directory: str | Path) -> list[str]:
    return sorted(load_manifest(dataset_directory)["datasets"])


def load_dataset(dataset_directory: str | Path, name: str) -> Dataset:
    directory = Path(dataset_directory)
    manifest = load_manifest(directory)
    if name not in manifest["datasets"]:
        raise KeyError(f"Unknown dataset in manifest: {name}")
    specification = manifest["datasets"][name]
    features = tuple(specification["training_features"])
    target = specification["target"]
    group = specification["group_column"]
    database = directory / "ml_datasets.sqlite"
    columns = ("banana_id", "measurement_date", "capture_id", target, group, *features)
    quoted = ",".join(f'"{column}"' for column in columns)
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(
            f'SELECT {quoted} FROM "{name}" ORDER BY banana_id,measurement_date,capture_id'
        ).fetchall()
    if not rows:
        raise ValueError(f"Dataset {name} contains no rows")
    width = len(columns)
    if any(len(row) != width for row in rows):
        raise ValueError(f"Unexpected row width in {name}")
    identifiers = tuple((str(row[0]), str(row[1]), str(row[2])) for row in rows)
    y = np.asarray([str(row[3]) for row in rows], dtype=object)
    groups = np.asarray([str(row[4]) for row in rows], dtype=object)
    x = np.asarray([row[5:] for row in rows], dtype=float)
    if not np.all(np.isfinite(x)):
        locations = np.argwhere(~np.isfinite(x))
        raise ValueError(f"Dataset {name} has non-finite feature values at {locations[:5].tolist()}")
    return Dataset(name, features, target, group, identifiers, x, y, groups)


def validate_common_rows(reference: Dataset, candidate: Dataset) -> None:
    if reference.identifiers != candidate.identifiers:
        raise ValueError(
            f"{candidate.name} does not contain the same ordered measurements as {reference.name}; "
            "shared folds cannot be guaranteed"
        )
    if not np.array_equal(reference.y, candidate.y) or not np.array_equal(
        reference.groups, candidate.groups
    ):
        raise ValueError(f"Labels or banana groups differ in {candidate.name}")

