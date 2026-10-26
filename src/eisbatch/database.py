"""SQLite persistence for captures, versioned configurations and fit results."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .configuration import BatchConfig
from .discovery import Capture


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS captures (
    id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    banana_id TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    point_count INTEGER NOT NULL,
    min_frequency_hz REAL NOT NULL,
    max_frequency_hz REAL NOT NULL,
    frequency_json TEXT NOT NULL,
    real_json TEXT NOT NULL,
    imag_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    qc_status TEXT NOT NULL,
    qc_message TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 1,
    discovered_at TEXT NOT NULL,
    UNIQUE(source_path, file_hash)
);
CREATE INDEX IF NOT EXISTS captures_banana_date ON captures(banana_id, measurement_date);
CREATE TABLE IF NOT EXISTS fit_configs (
    id INTEGER PRIMARY KEY,
    config_hash TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fits (
    id INTEGER PRIMARY KEY,
    capture_id INTEGER NOT NULL REFERENCES captures(id),
    config_id INTEGER NOT NULL REFERENCES fit_configs(id),
    fruit_model TEXT NOT NULL,
    electrode_model TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    objective REAL,
    nrmse REAL,
    aicc REAL,
    aic REAL,
    bic REAL,
    parameters_json TEXT,
    run_objectives_json TEXT,
    predicted_real_json TEXT,
    predicted_imag_json TEXT,
    selected_indices_json TEXT,
    evaluations INTEGER,
    best_seed INTEGER,
    completed_at TEXT,
    UNIQUE(capture_id, config_id, fruit_model, electrode_model)
);
CREATE INDEX IF NOT EXISTS fits_config_status ON fits(config_id, status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def add_capture(self, capture: Capture) -> int:
        spectrum = capture.spectrum
        self.connection.execute(
            "UPDATE captures SET is_current=0 WHERE source_path=? AND file_hash<>?",
            (str(capture.path), capture.file_hash),
        )
        values = (
            str(capture.path), capture.path.name, capture.file_hash, capture.banana_id,
            capture.measurement_date, len(spectrum.frequency_hz), float(spectrum.frequency_hz.min()),
            float(spectrum.frequency_hz.max()), json.dumps(spectrum.frequency_hz.tolist()),
            json.dumps(spectrum.impedance_ohm.real.tolist()), json.dumps(spectrum.impedance_ohm.imag.tolist()),
            json.dumps(spectrum.metadata, default=str), capture.qc_status, capture.qc_message, _now(),
        )
        self.connection.execute(
            """INSERT INTO captures(
                source_path,filename,file_hash,banana_id,measurement_date,point_count,
                min_frequency_hz,max_frequency_hz,frequency_json,real_json,imag_json,
                metadata_json,qc_status,qc_message,discovered_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(source_path,file_hash) DO UPDATE SET
                banana_id=excluded.banana_id, measurement_date=excluded.measurement_date,
                qc_status=excluded.qc_status, qc_message=excluded.qc_message,
                metadata_json=excluded.metadata_json, is_current=1""",
            values,
        )
        row = self.connection.execute(
            "SELECT id FROM captures WHERE source_path=? AND file_hash=?",
            (str(capture.path), capture.file_hash),
        ).fetchone()
        self.connection.commit()
        return int(row["id"])

    def add_config(self, config: BatchConfig) -> int:
        config.validate()
        self.connection.execute(
            """INSERT INTO fit_configs(config_hash,display_name,config_json,created_at)
               VALUES(?,?,?,?) ON CONFLICT(config_hash) DO NOTHING""",
            (config.identity(), config.short_name(), config.canonical_json(), _now()),
        )
        row = self.connection.execute(
            "SELECT id FROM fit_configs WHERE config_hash=?", (config.identity(),)
        ).fetchone()
        self.connection.commit()
        return int(row["id"])

    def captures(self, current_only: bool = True) -> list[sqlite3.Row]:
        where = "WHERE is_current=1" if current_only else ""
        return list(self.connection.execute(
            f"SELECT * FROM captures {where} ORDER BY banana_id, measurement_date, id"
        ))

    def configs(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM fit_configs ORDER BY id DESC"))

    def completion_counts(self, config_id: int) -> dict[int, int]:
        rows = self.connection.execute(
            """SELECT capture_id, COUNT(*) AS completed FROM fits
               WHERE config_id=? AND status='complete' GROUP BY capture_id""",
            (config_id,),
        )
        return {int(row["capture_id"]): int(row["completed"]) for row in rows}

    def pending_jobs(self, config_id: int, combinations: Iterable[tuple[str, str]], refit: bool) -> list[tuple]:
        jobs = []
        for capture in self.captures():
            for fruit, electrode in combinations:
                existing = self.connection.execute(
                    """SELECT status FROM fits WHERE capture_id=? AND config_id=?
                       AND fruit_model=? AND electrode_model=?""",
                    (capture["id"], config_id, fruit, electrode),
                ).fetchone()
                if refit or existing is None or existing["status"] != "complete":
                    jobs.append((dict(capture), config_id, fruit, electrode))
        return jobs

    def mark_running(self, capture_id: int, config_id: int, fruit: str, electrode: str) -> None:
        self.connection.execute(
            """INSERT INTO fits(capture_id,config_id,fruit_model,electrode_model,status)
               VALUES(?,?,?,?, 'running')
               ON CONFLICT(capture_id,config_id,fruit_model,electrode_model)
               DO UPDATE SET status='running', error_message=NULL""",
            (capture_id, config_id, fruit, electrode),
        )
        self.connection.commit()

    def save_result(self, capture_id: int, config_id: int, fruit: str, electrode: str,
                    result: dict, selected_indices: list[int]) -> None:
        metrics = result["metrics"]
        self.connection.execute(
            """INSERT INTO fits(
                capture_id,config_id,fruit_model,electrode_model,status,objective,nrmse,aicc,aic,bic,
                parameters_json,run_objectives_json,predicted_real_json,predicted_imag_json,
                selected_indices_json,evaluations,best_seed,completed_at
            ) VALUES(?,?,?,?, 'complete',?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(capture_id,config_id,fruit_model,electrode_model) DO UPDATE SET
                status='complete',error_message=NULL,objective=excluded.objective,nrmse=excluded.nrmse,
                aicc=excluded.aicc,aic=excluded.aic,bic=excluded.bic,
                parameters_json=excluded.parameters_json,run_objectives_json=excluded.run_objectives_json,
                predicted_real_json=excluded.predicted_real_json,predicted_imag_json=excluded.predicted_imag_json,
                selected_indices_json=excluded.selected_indices_json,evaluations=excluded.evaluations,
                best_seed=excluded.best_seed,completed_at=excluded.completed_at""",
            (
                capture_id, config_id, fruit, electrode, result["objective"], metrics["nrmse_complex"],
                metrics["aicc"], metrics["aic"], metrics["bic"], json.dumps(result["parameters"]),
                json.dumps(result["run_objectives"]), json.dumps(result["predicted_real_ohm"]),
                json.dumps(result["predicted_imag_ohm"]), json.dumps(selected_indices),
                result["evaluations"], result["best_seed"], _now(),
            ),
        )
        self.connection.commit()

    def save_failure(self, capture_id: int, config_id: int, fruit: str, electrode: str, message: str) -> None:
        self.connection.execute(
            """INSERT INTO fits(capture_id,config_id,fruit_model,electrode_model,status,error_message,completed_at)
               VALUES(?,?,?,?, 'failed',?,?)
               ON CONFLICT(capture_id,config_id,fruit_model,electrode_model) DO UPDATE SET
               status='failed',error_message=excluded.error_message,completed_at=excluded.completed_at""",
            (capture_id, config_id, fruit, electrode, message, _now()),
        )
        self.connection.commit()

    def fit_rows(self, config_id: int | None = None) -> list[sqlite3.Row]:
        where = "WHERE f.config_id=?" if config_id is not None else ""
        params = (config_id,) if config_id is not None else ()
        return list(self.connection.execute(
            f"""SELECT f.*,c.banana_id,c.measurement_date,c.filename,c.file_hash,c.frequency_json,
                       c.real_json,c.imag_json,fc.display_name,fc.config_json
                FROM fits f JOIN captures c ON c.id=f.capture_id
                JOIN fit_configs fc ON fc.id=f.config_id {where}
                ORDER BY c.banana_id,c.measurement_date,f.fruit_model,f.electrode_model""",
            params,
        ))
