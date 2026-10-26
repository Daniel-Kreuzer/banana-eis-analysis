"""Discover and validate SensorPal captures."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .engine import Spectrum, load_spectrum


BANANA_PATTERN = re.compile(r"^(banana[0-9a-z-]*)_", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*"
    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    re.IGNORECASE,
)


@dataclass
class Capture:
    path: Path
    file_hash: str
    banana_id: str
    measurement_date: str
    spectrum: Spectrum
    qc_status: str
    qc_message: str


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_filename(path: Path) -> tuple[str, str]:
    banana_match = BANANA_PATTERN.search(path.name)
    if not banana_match:
        raise ValueError("filename must begin with banana ID, for example banana1_")
    date_match = DATE_PATTERN.search(path.name)
    if not date_match:
        raise ValueError("SensorPal date was not found in filename")
    date = datetime.strptime(date_match.group(1), "%d %B %Y").date().isoformat()
    return banana_match.group(1).lower(), date


def _quality(spectrum: Spectrum) -> tuple[str, str]:
    frequency = spectrum.frequency_hz
    messages = []
    if len(frequency) < 20:
        messages.append(f"only {len(frequency)} frequency points")
    if len(np.unique(frequency)) != len(frequency):
        messages.append("duplicate frequencies")
    if np.any(np.diff(frequency) <= 0):
        messages.append("frequencies are not strictly increasing")
    if np.max(np.abs(np.diff(np.angle(spectrum.impedance_ohm)))) > np.pi:
        messages.append("large adjacent phase discontinuity")
    return ("warning", "; ".join(messages)) if messages else ("ready", "")


def load_capture(path: Path) -> Capture:
    banana_id, measurement_date = parse_filename(path)
    spectrum = load_spectrum(path)
    status, message = _quality(spectrum)
    return Capture(path.resolve(), _hash_file(path), banana_id, measurement_date, spectrum, status, message)


def discover(folder: str | Path) -> tuple[list[Capture], list[tuple[Path, str]]]:
    root = Path(folder)
    captures, failures = [], []
    candidates = (*root.rglob("*.xlsx"), *root.rglob("*.csv"))
    paths = sorted(path for path in candidates if BANANA_PATTERN.search(path.name))
    for path in paths:
        try:
            captures.append(load_capture(path))
        except Exception as exc:
            failures.append((path, str(exc)))
    return captures, failures
