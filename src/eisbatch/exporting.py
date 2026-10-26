"""CSV exports for audit and downstream analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .database import Database


def export_results(database: Database, output_folder: str | Path) -> tuple[Path, Path]:
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    rows = database.fit_rows()
    long_path = output / "circuit_fit_results_long.csv"
    parameter_names = sorted({
        name for row in rows if row["parameters_json"] for name in json.loads(row["parameters_json"])
    })
    fields = [
        "capture_id", "banana_id", "measurement_date", "filename", "file_hash", "configuration", "config_id",
        "fruit_model", "electrode_model", "status", "nrmse", "aicc", "aic", "bic",
        "objective", "evaluations", "best_seed", *parameter_names,
    ]
    with long_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            parameters = json.loads(row["parameters_json"]) if row["parameters_json"] else {}
            writer.writerow({key: row[key] for key in fields if key in row.keys()} | parameters |
                            {"configuration": row["display_name"]})

    wide_path = output / "circuit_fit_features_wide.csv"
    complete = [row for row in rows if row["status"] == "complete"]
    feature_names = set()
    records: dict[tuple, dict] = {}
    for row in complete:
        key = (row["capture_id"], row["config_id"])
        record = records.setdefault(key, {
            "capture_id": row["capture_id"],
            "banana_id": row["banana_id"], "measurement_date": row["measurement_date"],
            "filename": row["filename"], "file_hash": row["file_hash"],
            "config_id": row["config_id"], "configuration": row["display_name"],
        })
        prefix = f"{row['fruit_model']}__{row['electrode_model']}__"
        values = {"nrmse": row["nrmse"], "aicc": row["aicc"]}
        values.update(json.loads(row["parameters_json"]))
        for name, value in values.items():
            feature = prefix + name
            feature_names.add(feature)
            record[feature] = value
    wide_fields = ["capture_id", "banana_id", "measurement_date", "filename", "file_hash",
                   "config_id", "configuration", *sorted(feature_names)]
    with wide_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=wide_fields)
        writer.writeheader()
        writer.writerows(records.values())
    return long_path, wide_path
