"""SensorPal workbook and generic CSV loading."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Spectrum:
    frequency_hz: np.ndarray
    impedance_ohm: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def validate(self) -> None: # Check whether data is in the format expected
        if self.frequency_hz.ndim != 1 or self.impedance_ohm.ndim != 1:
            raise ValueError("frequency and impedance must be one-dimensional")
        if len(self.frequency_hz) != len(self.impedance_ohm):
            raise ValueError("frequency and impedance lengths differ")
        if len(self.frequency_hz) < 4:
            raise ValueError("at least four frequency points are required")
        if not np.all(np.isfinite(self.frequency_hz)) or np.any(self.frequency_hz <= 0):
            raise ValueError("frequencies must be finite and positive")
        if not np.all(np.isfinite(self.impedance_ohm.real)) or not np.all(
            np.isfinite(self.impedance_ohm.imag)
        ):
            raise ValueError("impedance values must be finite")


def wrap_phase_degrees(phase_deg: np.ndarray) -> np.ndarray: # Removes the effect of phase wrapping
    return (np.asarray(phase_deg, dtype=float) + 180.0) % 360.0 - 180.0 # Neat lil formular


def polar_to_complex(magnitude_ohm: np.ndarray, phase_deg: np.ndarray) -> np.ndarray: # 
    phase_rad = np.deg2rad(wrap_phase_degrees(phase_deg))
    return np.asarray(magnitude_ohm, dtype=float) * np.exp(1j * phase_rad)


def _normalise_header(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _load_sensorpal_xlsx(path: Path) -> Spectrum:
    try: # Need openpyxl to read the excel workbooks
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("openpyxl is required to read SensorPal .xlsx files") from exc

    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    checkpoint = sheet.cell(1, 1).value
    frequencies: list[float] = []
    magnitudes: list[float] = []
    phases: list[float] = []
    for row in range(3, sheet.max_row + 1): # extract the data from the sheets
        frequency = sheet.cell(row, 1).value
        magnitude = sheet.cell(row, 2).value
        phase = sheet.cell(row, 4).value
        if all(isinstance(value, (int, float)) for value in (frequency, magnitude, phase)):
            frequencies.append(float(frequency))
            magnitudes.append(float(magnitude))
            phases.append(float(phase))

    metadata: dict[str, Any] = {"checkpoint": checkpoint}
    for row in range(2, min(sheet.max_row, 30) + 1):
        key = sheet.cell(row, 8).value
        value = sheet.cell(row, 9).value
        unit = sheet.cell(row, 10).value
        if key is not None:
            metadata[str(key)] = {"value": value, "unit": unit}
    workbook.close()

    spectrum = Spectrum( # create a spectrum object
        np.asarray(frequencies),
        polar_to_complex(np.asarray(magnitudes), np.asarray(phases)),
        metadata,
        str(path),
    )
    spectrum.validate()
    return spectrum


def _load_csv(path: Path) -> Spectrum:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        lookup = {_normalise_header(name): name for name in reader.fieldnames}

        def find(*aliases: str) -> str | None:
            for alias in aliases:
                if alias in lookup:
                    return lookup[alias]
            return None

        f_col = find("frequencyhz", "frequency", "freqhz") # Find the different columns associated with our datafields
        mag_col = find("magnitudeohm", "magnitude", "zmagohm")
        phase_col = find("phasedeg", "phase", "phaseangledeg")
        real_col = find("realohm", "real", "rezohm")
        imag_col = find("imagohm", "imaginaryohm", "imag", "imzohm")
        if f_col is None:
            raise ValueError("CSV needs a frequency_hz column")
        if not ((mag_col and phase_col) or (real_col and imag_col)):
            raise ValueError("CSV needs magnitude/phase or real/imaginary columns")

        frequencies = []
        first = []
        second = []
        for row in reader:
            frequencies.append(float(row[f_col]))
            if mag_col and phase_col:
                first.append(float(row[mag_col]))
                second.append(float(row[phase_col]))
            else:
                first.append(float(row[real_col]))
                second.append(float(row[imag_col]))

    if mag_col and phase_col:
        impedance = polar_to_complex(np.asarray(first), np.asarray(second))
    else:
        impedance = np.asarray(first) + 1j * np.asarray(second)
    spectrum = Spectrum(np.asarray(frequencies), impedance, source=str(path))
    spectrum.validate()
    return spectrum


def load_spectrum(path: str | Path) -> Spectrum:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".xlsx":
        return _load_sensorpal_xlsx(source)
    if suffix == ".csv":
        return _load_csv(source)
    raise ValueError(f"Unsupported input format: {source.suffix}")
