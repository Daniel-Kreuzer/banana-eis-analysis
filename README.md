# Banana EIS analysis

This repository contains the software and dataset developed for a longitudinal
banana-ripeness study using electrical impedance spectroscopy (EIS). It provides
tools for SensorPal data import, equivalent-circuit fitting, metadata integration,
machine-learning dataset preparation, MLP classification and Monte Carlo noise
analysis.

The analysis software is independent of the embedded firmware. An unsuccessful
attempt was made during development to replace the licensed IAR workflow with a
GCC-based build, after which acquisition moved to the validated SensorPal
workflow. Firmware source from that experiment is deliberately not distributed
because it was unused by the final software and included third-party licensing
constraints.

## Repository layout

```text
banana-eis-analysis/
├── data/
│   ├── raw/                 # SensorPal workbooks, grouped by measurement day
│   ├── metadata/            # Banana dimensions and ripeness observations
│   └── processed/           # Circuit-fit database and ML-ready tables
├── docs/                    # Dataset and workflow documentation
├── scripts/                 # Reproducible report and sensitivity analyses
├── src/                     # Installable Python packages
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Installation

Python 3.11 or later is recommended. From the repository root, create and
activate a virtual environment, then install the project:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

All paths used by the repository are relative to the repository root or selected
through a file browser. The project can therefore be moved or cloned to a new
location without editing source files.

## Applications

```bash
eis-fit gui          # Fit and inspect one spectrum
eis-batch            # Batch fitting and longitudinal visualisation
eis-metadata         # Import dimensions and ripeness into a fitting database
eis-ml-prepare       # Build model-specific ML tables
eis-ml               # Train and evaluate grouped MLP classifiers
eis-noise --help     # Monte Carlo parameter-recovery study
```

Equivalent module commands such as `python -m eisbatch` may also be used.

The included processed database is `data/processed/circuit_fits.sqlite`. The raw
capture root is `data/raw`, and the prepared ML tables are in `data/processed`.
Generated figures and trained-model runs are intentionally not versioned; they
can be recreated from the software and supplied data.

## Reproducibility

The principal fitting configuration used population 30, 800 iterations and five
independent restarts. Eight circuit combinations were evaluated: two fruit models
(Cole and the simplified model) combined with no interface element, a capacitor,
a Warburg element or a Randles/CPE interface.

ML evaluation groups observations by banana identifier so that measurements from
one banana cannot be divided between training and test folds. See
[`docs/SOFTWARE.md`](docs/SOFTWARE.md) and [`docs/DATASET.md`](docs/DATASET.md).

## Testing

```bash
python -m pytest
```

## Licensing and attribution

No open-source licence has yet been selected for the original Python software or
dataset. Add an appropriate repository-level licence before public release if
reuse rights are to be granted.

No Analog Devices or IAR firmware source is included in this repository. The
Python analysis operates on exported SensorPal measurements and does not require
the original embedded project.
