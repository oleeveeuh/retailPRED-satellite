# Category win matrix (exploration origins only)

Per-category WAPE (%), pooled over horizons; bold = category winner.

### full_exploration

| category | naive | lightgbm | theta | lgbm+naive (global w) | theta+naive (global w) | theta+lgbm+naive (simplex) | winner |
|---|---:|---:|---:|---:|---:|---:|---:|
| building materials | 6.89 | 5.49 | 5.28 ** | 5.95 | 5.79 | 6.64 |     theta |
| clothing | 15.28 | 19.73 | 13.50 | 15.56 | 12.71 | 12.60 ** |           theta+lgbm+naive (simplex) |
| electronics | 14.31 | 16.92 | 12.65 ** | 14.76 | 12.92 | 12.65 |     theta |
| food & bev | 4.58 | 2.42 | 2.50 | 2.84 | 3.01 | 2.04 ** |           theta+lgbm+naive (simplex) |
| furniture | 8.70 | 12.06 | 8.29 | 9.34 | 7.72 ** | 7.76 |         theta+naive (global w) |
| gasoline | 15.77 | 12.01 | 9.48 ** | 12.82 | 11.48 | 9.48 |     theta |
| general merch | 7.61 | 3.87 ** | 5.05 | 5.93 | 6.26 | 4.07 |   lightgbm |
| health & personal | 2.90 | 5.16 | 3.51 | 3.07 | 2.77 ** | 2.90 |         theta+naive (global w) |
| motor vehicles | 7.64 | 7.46 | 5.27 ** | 6.84 | 5.80 | 5.80 |     theta |
| retail total | 5.11 | 4.75 | 3.33 ** | 3.86 | 3.51 | 3.75 |     theta |
| sporting goods | 13.46 | 15.17 | 13.78 | 13.65 | 13.31 ** | 13.78 |         theta+naive (global w) |
| **categories won (of 11)** | **0** | **1** | **5** | **0** | **3** | **2** | |
| **beats naive (of 11)** | — | 6 | 9 | 6 | 11 | 9 | |
| mean category WAPE (for reference) | 9.30% | 9.55% | 7.51% | 8.60% | 7.75% | 7.41% | |

### choose (2017-10..2020-12)

| category | naive | lightgbm | theta | lgbm+naive (global w) | theta+naive (global w) | theta+lgbm+naive (simplex) | winner |
|---|---:|---:|---:|---:|---:|---:|---:|
| building materials | 5.36 | 7.02 | 6.06 | 5.81 | 5.50 | 5.34 ** |           theta+lgbm+naive (simplex) |
| clothing | 15.32 | 16.42 | 15.05 | 15.22 | 14.22 | 14.18 ** |           theta+lgbm+naive (simplex) |
| electronics | 16.50 | 16.78 | 15.56 ** | 16.61 | 16.03 | 15.56 |     theta |
| food & bev | 4.84 | 2.71 | 3.11 | 3.33 | 3.33 | 2.42 ** |           theta+lgbm+naive (simplex) |
| furniture | 7.99 | 10.04 | 8.34 | 8.33 | 7.78 | 7.76 ** |           theta+lgbm+naive (simplex) |
| gasoline | 12.76 | 9.19 | 8.83 ** | 10.79 | 10.08 | 8.83 |     theta |
| general merch | 4.41 | 2.60 | 2.85 | 3.38 | 3.52 | 2.59 ** |           theta+lgbm+naive (simplex) |
| health & personal | 2.57 ** | 5.42 | 4.32 | 3.56 | 3.35 | 2.57 | naive |
| motor vehicles | 6.30 | 8.08 | 6.39 | 6.24 | 5.75 ** | 5.75 |         theta+naive (global w) |
| retail total | 3.42 | 4.74 | 4.00 | 3.11 | 3.06 | 3.03 ** |           theta+lgbm+naive (simplex) |
| sporting goods | 12.07 | 11.72 | 11.49 ** | 11.81 | 11.70 | 11.49 |     theta |
| **categories won (of 11)** | **1** | **0** | **3** | **0** | **1** | **6** | |
| **beats naive (of 11)** | — | 4 | 6 | 7 | 9 | 10 | |
| mean category WAPE (for reference) | 8.32% | 8.61% | 7.82% | 8.02% | 7.66% | 7.23% | |

### report (2021-01..2022-10)

| category | naive | lightgbm | theta | lgbm+naive (global w) | theta+naive (global w) | theta+lgbm+naive (simplex) | winner |
|---|---:|---:|---:|---:|---:|---:|---:|
| building materials | 8.80 | 3.57 ** | 4.30 | 6.12 | 6.16 | 8.27 |   lightgbm |
| clothing | 15.23 | 24.39 | 11.32 | 16.03 | 10.59 | 10.37 ** |           theta+lgbm+naive (simplex) |
| electronics | 10.80 | 17.14 | 7.97 | 11.79 | 7.93 ** | 7.97 |         theta+naive (global w) |
| food & bev | 4.22 | 2.01 | 1.64 | 2.16 | 2.56 | 1.52 ** |           theta+lgbm+naive (simplex) |
| furniture | 9.68 | 14.88 | 8.21 | 10.76 | 7.65 ** | 7.76 |         theta+naive (global w) |
| gasoline | 19.49 | 15.50 | 10.28 ** | 15.32 | 13.22 | 10.28 |     theta |
| general merch | 11.97 | 5.60 ** | 8.05 | 9.42 | 10.01 | 6.09 |   lightgbm |
| health & personal | 3.38 | 4.78 | 2.33 | 2.37 | 1.93 ** | 3.38 |         theta+naive (global w) |
| motor vehicles | 9.37 | 6.67 | 3.83 ** | 7.61 | 5.86 | 5.86 |     theta |
| retail total | 7.30 | 4.76 | 2.45 ** | 4.83 | 4.08 | 4.69 |     theta |
| sporting goods | 15.34 ** | 19.81 | 16.89 | 16.13 | 15.49 | 16.89 | naive |
| **categories won (of 11)** | **1** | **2** | **3** | **0** | **3** | **2** | |
| **beats naive (of 11)** | — | 6 | 10 | 7 | 10 | 9 | |
| mean category WAPE (for reference) | 10.51% | 10.83% | 7.02% | 9.32% | 7.77% | 7.55% | |
