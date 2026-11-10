# H7 scorecard — final configuration

Candidate: **direct LGBM base+growth** | 24 exploration origins (2017-10-01..2022-10-01) | verification gates passed (h=1 equivalence, fold-key identity, naive/theta frozen reproduction).

All deltas paired, positive = candidate better. Actuals: final-vintage (governing); first-release re-scoring is the descriptive companion.

## choose — final-vintage actuals (15 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 5.00% | 8.32% | 2.160 | -0.68pp | 7.01 | 6.49 | 8.16 | 11.65 |
| theta | 4.95% | 7.82% | 2.033 | +1.66pp | 5.47 | 6.02 | 8.10 | 11.73 |
| candidate | 5.58% | 8.39% | 2.154 | +2.89pp | 5.50 | 6.58 | 8.53 | 13.01 |

- relMAE (candidate vs naive): 1.097 | categories beating naive: 4/11 | beating theta: 3/11
- vs naive mean-cat: -0.07pp (CI -0.88..+0.96) | pooled -0.58pp (CI -1.28..+0.19)
- vs theta mean-cat: -0.58pp (CI -0.84..-0.24) | pooled -0.63pp (CI -1.01..-0.18)

REGATE v2 mirror: C1 FAIL (CI -0.88.., floor -0.07pp, pooled guard -0.58pp) | C2 FAIL (worst health_and_personal_care +2.99pp, retail_total +1.40pp) | C3 FAIL (CI ..-0.24, worst vs theta building_material_and_garden +1.46pp) → **FAIL**

## choose — first-release actuals (descriptive) (15 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 6.04% | 7.54% | 2.096 | -2.88pp | 6.30 | 5.58 | 7.46 | 10.84 |
| theta | 3.94% | 5.87% | 1.618 | -0.59pp | 2.89 | 3.85 | 6.20 | 10.53 |
| candidate | 3.88% | 5.98% | 1.612 | +0.60pp | 2.75 | 4.15 | 5.96 | 11.05 |

- relMAE (candidate vs naive): 0.760 | categories beating naive: 11/11 | beating theta: 4/11
- vs naive mean-cat: +1.56pp (CI +0.93..+2.20) | pooled +2.16pp (CI +1.65..+2.64)
- vs theta mean-cat: -0.11pp (CI -0.39..+0.21) | pooled +0.06pp (CI -0.26..+0.41)

## report — final-vintage actuals (9 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 8.55% | 10.51% | 2.862 | -7.32pp | 12.88 | 12.55 | 8.70 | 7.99 |
| theta | 3.87% | 7.02% | 1.836 | -0.45pp | 6.23 | 6.48 | 7.30 | 8.04 |
| candidate | 5.63% | 9.85% | 2.507 | +1.38pp | 9.34 | 11.11 | 9.53 | 9.37 |

- relMAE (candidate vs naive): 0.917 | categories beating naive: 6/11 | beating theta: 2/11
- vs naive mean-cat: +0.66pp (CI -0.60..+2.17) | pooled +2.92pp (CI +0.85..+5.07)
- vs theta mean-cat: -2.82pp (CI -4.69..-1.17) | pooled -1.76pp (CI -2.78..-0.68)

REGATE v2 mirror: C1 FAIL (CI -0.60.., floor +0.66pp, pooled guard +2.92pp) | C2 FAIL (worst sporting_goods_and_hobby +8.22pp, retail_total -2.72pp) | C3 FAIL (CI ..-1.17, worst vs theta clothing_and_accessories +6.76pp) → **FAIL**

## report — first-release actuals (descriptive) (9 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 9.27% | 10.09% | 2.830 | -8.62pp | 13.56 | 12.92 | 7.80 | 6.15 |
| theta | 3.24% | 4.12% | 1.118 | -1.85pp | 2.02 | 3.70 | 4.70 | 5.98 |
| candidate | 4.71% | 7.02% | 1.842 | -0.04pp | 5.61 | 8.62 | 6.99 | 6.79 |

- relMAE (candidate vs naive): 0.662 | categories beating naive: 10/11 | beating theta: 2/11
- vs naive mean-cat: +3.08pp (CI +1.57..+4.86) | pooled +4.56pp (CI +2.76..+6.50)
- vs theta mean-cat: -2.90pp (CI -4.72..-1.25) | pooled -1.47pp (CI -2.52..-0.38)

## full_exploration — final-vintage actuals (24 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 6.54% | 9.30% | 2.423 | -3.55pp | 9.52 | 9.09 | 8.45 | 10.14 |
| theta | 4.49% | 7.51% | 1.959 | +0.75pp | 5.80 | 6.23 | 7.81 | 10.20 |
| candidate | 5.60% | 9.02% | 2.287 | +2.23pp | 7.11 | 8.48 | 8.99 | 11.47 |

- relMAE (candidate vs naive): 0.982 | categories beating naive: 6/11 | beating theta: 2/11
- vs naive mean-cat: +0.28pp (CI -0.46..+1.14) | pooled +0.94pp (CI -0.31..+2.23)
- vs theta mean-cat: -1.50pp (CI -2.45..-0.73) | pooled -1.12pp (CI -1.67..-0.56)

REGATE v2 mirror: C1 FAIL (CI -0.46.., floor +0.28pp, pooled guard +0.94pp) | C2 FAIL (worst sporting_goods_and_hobby +2.69pp, retail_total -0.40pp) | C3 FAIL (CI ..-0.73, worst vs theta clothing_and_accessories +3.33pp) → **FAIL**

## full_exploration — first-release actuals (descriptive) (24 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 7.43% | 8.64% | 2.371 | -5.35pp | 9.38 | 8.70 | 7.64 | 8.84 |
| theta | 3.64% | 5.13% | 1.431 | -1.13pp | 2.52 | 3.78 | 5.57 | 8.59 |
| candidate | 4.24% | 6.41% | 1.698 | +0.32pp | 3.95 | 6.02 | 6.42 | 9.22 |

- relMAE (candidate vs naive): 0.709 | categories beating naive: 10/11 | beating theta: 2/11
- vs naive mean-cat: +2.23pp (CI +1.42..+3.13) | pooled +3.19pp (CI +2.24..+4.19)
- vs theta mean-cat: -1.28pp (CI -2.32..-0.44) | pooled -0.60pp (CI -1.19..-0.04)

Coarse-CI note: 9 report / 24 full-exploration origin blocks — intervals are wide by construction and reported as such (scorecard precedent).

This scorecard is an exploration-side selection result. It promotes nothing; the only promotion path is the future-origin re-gate (REGATE_DESIGN.md).
