# Holdout scorecard (reporting-only reaggregation, 2026-11-10)

Predeclared metric definitions and verification gates in
`scripts/holdout_scorecard.py`; every gate reproduces the frozen record
before release. All numbers two decimals unless noted.

### holdout (8 origins)

*8 origin blocks — bootstrap CIs here are coarse. LightGBM rows are previously observed (gate reference folds), not an untouched holdout result. Theta rows are the recovered, verification-locked single-run predictions.*

| model | pooled WAPE | mean-cat WAPE | relMAE | MASE | bias (pp) | beats naive | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seasonal_naive | 2.65% | 5.15% | 1.000 | 1.296 | +0.63 | 0/11 | 5.26% | 5.46% | 5.15% | 4.76% |
| theta_method | 3.43% | 4.98% | 1.027 | 1.257 | +2.52 | 7/11 | 4.90% | 5.00% | 5.09% | 4.94% |
| global_lightgbm | 4.48% | 5.64% | 1.262 | 1.435 | +3.80 | 5/11 | 5.00% | 5.43% | 5.71% | 6.41% |

### gate_full (32 origins)

*Full gate reference window (2017-10..2025-03, all previously observed folds). Includes the holdout origins in its average.*

| model | pooled WAPE | mean-cat WAPE | relMAE | MASE | bias (pp) | beats naive | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seasonal_naive | 5.43% | 8.16% | 1.000 | 2.141 | -2.36 | 0/11 | 8.34% | 8.09% | 7.54% | 8.66% |
| global_lightgbm | 5.48% | 8.49% | 1.028 | 2.164 | +2.39 | 4/11 | 6.59% | 7.67% | 8.57% | 11.11% |

### exploration (24 origins)

| model | pooled WAPE | mean-cat WAPE | relMAE | MASE | bias (pp) | beats naive | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seasonal_naive | 6.54% | 9.30% | 1.000 | 2.423 | -3.55 | 0/11 | 9.52% | 9.09% | 8.45% | 10.14% |
| theta_method | 4.49% | 7.51% | 0.807 | 1.959 | +0.75 | 9/11 | 5.80% | 6.23% | 7.81% | 10.20% |

### Origin-block bootstrap CIs (paired deltas vs naive unless labeled)

| window | pair | mean-category delta (pp) | CI | pooled delta (pp) | CI | categories improved |
|---|---|---:|---|---:|---|---:|
| holdout | theta_vs_naive | +0.17 | -0.17..+0.54 | -0.77 | -1.24..-0.36 | 7 (4.0..8.0) |
| holdout | lightgbm_vs_naive | -0.48 | -0.89..-0.08 | -1.83 | -2.49..-1.27 | 5 (3.0..6.0) |
| holdout | theta_vs_lightgbm | +0.65 | +0.46..+0.88 | +1.05 | +0.79..+1.30 | 9 (7.0..10.0) |
| gate_full | lightgbm_vs_naive | -0.33 | -0.75..+0.14 | -0.04 | -0.92..+0.90 | 4 (3.0..6.0) |
| exploration | theta_vs_naive | +1.78 | +0.61..+3.25 | +2.05 | +0.73..+3.49 | 9 (6.0..11.0) |

*The holdout window has 8 origin blocks; its CIs are coarse by construction and should be read as scale-of-evidence, not as significance tests.*

### First-release vs final-vintage actuals (same stored predictions)

| window | model | mean-cat WAPE (first release) | mean-cat WAPE (final) | pooled WAPE (first release) | pooled WAPE (final) | bias first release (pp) | bias final (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| holdout | seasonal_naive | 3.48% | 5.15% | 2.66% | 2.65% | -1.73 | +0.63 |
| holdout | theta_method | 2.29% | 4.98% | 1.38% | 3.43% | +0.12 | +2.52 |
| holdout | global_lightgbm | 2.77% | 5.64% | 1.92% | 4.48% | +1.36 | +3.80 |
| gate_full | seasonal_naive | 7.23% | 8.16% | 6.07% | 5.43% | -4.32 | -2.36 |
| gate_full | global_lightgbm | 5.84% | 8.49% | 3.69% | 5.48% | +0.34 | +2.39 |

### First-release bootstrap (holdout)

- theta_vs_naive: mean-category delta +1.19pp (CI +0.83..+1.57)
- lightgbm_vs_naive: mean-category delta +0.71pp (CI +0.30..+1.18)

