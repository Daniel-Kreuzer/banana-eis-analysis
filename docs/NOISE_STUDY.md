# Fruit-model noise sensitivity study

This separate program performs a Monte Carlo parameter-recovery study for the
Cole and paper fruit models. It does not include electrode-interface models.
For the noisy condition, it constructs the raw embedded impedance by adding the
known fixed series hardware path, applies noise to that complete measurement,
then removes the path before fitting the fruit model.

The default ground truth is taken from banana 16 measured on 5 October 2026:

- Cole: `r_inf=462.654 Ω`, `delta_r=1275.060 Ω`, `tau=7.825 µs`,
  `alpha=0.8081` (`r0=1737.714 Ω`).
- Paper: `r1=864.193 Ω`, `r2=1645.102 Ω`, `c1=2.922 nF`.

Each model independently generates an ideal 50-point logarithmic spectrum from
100 Hz to 180 kHz. Two conditions are fitted:

1. no added noise;
2. circular relative complex Gaussian noise applied to the complete raw
   measurement, with 8% RMS noise from 100–300 Hz,
   a smooth cosine transition from 300–600 Hz, and 1% RMS noise thereafter.

The no-noise repetitions measure optimizer/recovery variability. The noisy
repetitions combine noise sensitivity and optimizer variability. Every fit is
an independent cold start.

## Run

After installing the repository from its root, run:

```powershell
eis-noise --output noise_study_output
```

An existing compatible no-noise baseline can be reused so only the corrected
noisy condition is recomputed:

```powershell
eis-noise --output embedded_noise_output `
  --reuse-no-noise-from previous_output/noise_recovery_replicates.csv
```

The full defaults run 200 repetitions per model and condition with population
20, 300 iterations and 3 restarts. Automatic parallelism uses approximately
half the logical processors, capped at four workers.

Validate the complete pipeline quickly with:

```powershell
eis-noise --smoke --output smoke_output
```

Important options include `--replicates`, `--population`, `--iterations`,
`--restarts`, `--low-noise-percent` and `--baseline-noise-percent`.

## Outputs

- `study_configuration.json`: exact reproducible settings and ground truth.
- `noise_recovery_replicates.csv`: every fit, seed, score and parameter error.
- `noise_recovery_summary.csv`: bias, standard deviation, RMSE, 95% interval
  and bound-hit rate for each parameter.
- `ideal_and_example_noisy_spectra.png`: ideal and example noisy spectra.
- `noise_profile.png`: applied frequency-dependent noise level.
- Model-specific parameter-recovery boxplots and correlation heatmaps.

