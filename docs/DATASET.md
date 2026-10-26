# Dataset

## Contents

The repository contains longitudinal EIS measurements for 30 bananas over ten
study days. There are 289 measured spectra. Some later measurements are absent
because individual fruit became unsuitable for measurement, while the ripeness
assessment table retains 300 banana-day observations.

- `data/raw/Day1` to `data/raw/Day10`: original SensorPal `.xlsx` exports.
- `data/metadata/BananaDimensions.xlsx`: day-one physical measurements.
- `data/metadata/Ripeness.xlsx`: U/R/O ripeness observations, interpreted as
  unripe, ripe and overripe.
- `data/processed/circuit_fits.sqlite`: captures, metadata, fit configurations,
  fitted parameters, predicted spectra and fit-quality statistics.
- `data/processed/ml_datasets.sqlite` and CSV files: model-specific tables used
  by the MLP workflow.

Day-one spectra cover 1--180 kHz. Subsequent measurements cover
100 Hz--180 kHz. Measurements use logarithmic spacing; the later-day acquisitions
contain 50 frequency points. Raw-spectrum ML tables use the common band above
1 kHz and interpolate day-one measurements only onto the later-day frequency grid
within that measured band.

## Processed variants

Each circuit combination has an `all` and `practical` CSV. The `all` variant uses
mass, signal-path length, arc length and circumference. The `practical` variant
uses mass and signal-path length only. The raw-spectrum variants contain
de-embedded real and imaginary impedance values rather than circuit parameters.

The processed tables also retain de-embedded NRMSE, AICc and parameter-boundary
flags for auditing fit quality. Repeated days from the same banana are correlated
and must be kept in the same validation group.

## Path portability

Paths stored in the copied fitting database are repository-relative. Resolve them
against the repository root when opening a source workbook programmatically.
