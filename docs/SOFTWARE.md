# Software workflow

## Packages

- `eisfit`: SensorPal parsing, circuit definitions, hybrid Cuckoo Search and
  single-spectrum fitting.
- `eisbatch`: scalable discovery, versioned fitting, SQLite storage, exports and
  longitudinal plots.
- `eismetadata`: imports physical dimensions and U/R/O ripeness observations.
- `eismlprep`: joins measurements, circuit parameters, fit diagnostics and
  physical metadata into model-specific learning tables.
- `eisml`: grouped nested cross-validation, MLP hyperparameter selection,
  evaluation and report figures.
- `eisnoise`: relative complex Gaussian-noise simulation and parameter-recovery
  analysis.

## Typical sequence

1. Run `eis-batch` and select a directory containing day subdirectories of
   SensorPal workbooks.
2. Import physical and ripeness metadata with `eis-metadata`.
3. Generate learning tables with `eis-ml-prepare`.
4. Train and evaluate the MLP studies with `eis-ml`.
5. Use the scripts in `scripts/` for report-specific spectrum, model-selection
   and frequency-range sensitivity figures.

The included database and ML tables allow users to begin at steps 3 or 4 without
rerunning the computationally expensive circuit fits.

## Acquisition-software context

An experimental GCC build system was investigated because a permanent IAR
licence was unavailable. That implementation was not successfully validated and
was not used to produce the final dataset. Measurements were exported using the
SensorPal workflow instead. The experimental firmware is therefore documented as
project history but is not distributed with this analysis repository.

## Generated files

Analysis scripts write to `output/` by default. That directory is ignored by Git
because figures, restricted-frequency databases and trained models can be
regenerated. Input locations are repository-relative.
