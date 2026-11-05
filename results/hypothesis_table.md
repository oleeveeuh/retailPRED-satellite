# Hypothesis results — exploration origins (report window: 2021-01..2022-10)

Report-window mean-category WAPE delta vs seasonal naive (positive = candidate
better), with the choose window (2017-10..2020-12) shown for selection honesty.

| candidate | folds | naive WAPE | candidate WAPE | delta (pp) | improved |
|---|---:|---:|---:|---:|---:|
| h5_theta | 396 | 10.51% | 7.02% | +3.48 | 10/11 |
| h3b theta_lgbm_naive_simplex | 396 | 10.51% | 7.55% | +2.96 | 9/11 |
| h3b theta_naive_per_category | 396 | 10.51% | 7.72% | +2.79 | 9/11 |
| h3b theta_naive_global | 396 | 10.51% | 7.77% | +2.74 | 10/11 |
| h3 level_blend/global | 396 | 10.51% | 9.32% | +1.18 | 7/11 |
| h3 log_blend/global | 396 | 10.51% | 9.38% | +1.13 | 7/11 |
| h3 log_blend/per_category | 396 | 10.51% | 9.44% | +1.07 | 5/11 |
| h3 level_blend/per_category | 396 | 10.51% | 9.46% | +1.04 | 5/11 |
| h1 h1_drop_unrevised (chosen) | 396 | 10.51% | 10.92% | -0.41 | 6/11 |
| h2_revision_features | 396 | 10.51% | 11.19% | -0.68 | 5/11 |
| h4_yoy_diff | 396 | 10.51% | 11.38% | -0.88 | 8/11 |
| h4_stl_yoy | 396 | 10.51% | 11.57% | -1.06 | 5/11 |
| h4_log_level | 396 | 10.51% | 29.58% | -19.07 | 2/11 |

Provenance: "b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86"
