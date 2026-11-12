# H9 scorecard — final configuration

Candidate: **r3_per_row_macro8_market64** | 24 exploration origins (2017-10-01..2022-10-01) | verification gates passed (h=1 equivalence, fold-key identity, naive/theta frozen reproduction).

All deltas paired, positive = candidate better. Actuals: final-vintage (governing); first-release re-scoring is the descriptive companion.

## choose — final-vintage actuals (15 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 5.00% | 8.32% | 2.160 | -0.68pp | 7.01 | 6.49 | 8.16 | 11.65 |
| theta | 4.95% | 7.82% | 2.033 | +1.66pp | 5.47 | 6.02 | 8.10 | 11.73 |
| candidate | 5.24% | 8.13% | 2.094 | +2.56pp | 5.55 | 6.34 | 8.46 | 12.23 |

- relMAE (candidate vs naive): 1.048 | categories beating naive: 5/11 | beating theta: 3/11
- vs naive mean-cat: +0.19pp (CI -0.56..+1.10) | pooled -0.23pp (CI -0.94..+0.58)
- vs theta mean-cat: -0.32pp (CI -0.52..-0.11) | pooled -0.29pp (CI -0.61..+0.04)

REGATE v2 mirror: C1 FAIL (CI -0.56.., floor +0.19pp, pooled guard -0.23pp) | C2 FAIL (worst health_and_personal_care +2.37pp, retail_total +0.93pp) | C3 FAIL (CI ..-0.11, worst vs theta building_material_and_garden +1.23pp) → **FAIL**

## choose — first-release actuals (descriptive) (15 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 6.04% | 7.54% | 2.096 | -2.88pp | 6.30 | 5.58 | 7.46 | 10.84 |
| theta | 3.94% | 5.87% | 1.618 | -0.59pp | 2.89 | 3.85 | 6.20 | 10.53 |
| candidate | 3.74% | 5.80% | 1.573 | +0.28pp | 2.82 | 3.82 | 6.18 | 10.37 |

- relMAE (candidate vs naive): 0.732 | categories beating naive: 11/11 | beating theta: 7/11
- vs naive mean-cat: +1.75pp (CI +1.17..+2.34) | pooled +2.30pp (CI +1.76..+2.84)
- vs theta mean-cat: +0.08pp (CI -0.20..+0.41) | pooled +0.20pp (CI +0.00..+0.41)

## report — final-vintage actuals (9 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 8.55% | 10.51% | 2.862 | -7.32pp | 12.88 | 12.55 | 8.70 | 7.99 |
| theta | 3.87% | 7.02% | 1.836 | -0.45pp | 6.23 | 6.48 | 7.30 | 8.04 |
| candidate | 5.46% | 9.82% | 2.521 | +0.70pp | 9.23 | 11.22 | 9.75 | 9.04 |

- relMAE (candidate vs naive): 0.917 | categories beating naive: 6/11 | beating theta: 2/11
- vs naive mean-cat: +0.69pp (CI -0.45..+2.04) | pooled +3.09pp (CI +1.23..+5.06)
- vs theta mean-cat: -2.80pp (CI -4.85..-0.95) | pooled -1.58pp (CI -2.85..-0.61)

REGATE v2 mirror: C1 FAIL (CI -0.45.., floor +0.69pp, pooled guard +3.09pp) | C2 FAIL (worst sporting_goods_and_hobby +7.95pp, retail_total -3.25pp) | C3 FAIL (CI ..-0.95, worst vs theta clothing_and_accessories +6.63pp) → **FAIL**

## report — first-release actuals (descriptive) (9 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 9.27% | 10.09% | 2.830 | -8.62pp | 13.56 | 12.92 | 7.80 | 6.15 |
| theta | 3.24% | 4.12% | 1.118 | -1.85pp | 2.02 | 3.70 | 4.70 | 5.98 |
| candidate | 4.79% | 7.21% | 1.922 | -0.71pp | 5.28 | 9.11 | 7.47 | 6.92 |

- relMAE (candidate vs naive): 0.692 | categories beating naive: 10/11 | beating theta: 2/11
- vs naive mean-cat: +2.88pp (CI +1.44..+4.57) | pooled +4.48pp (CI +2.76..+6.26)
- vs theta mean-cat: -3.09pp (CI -5.22..-1.28) | pooled -1.55pp (CI -2.79..-0.57)

## full_exploration — final-vintage actuals (24 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 6.54% | 9.30% | 2.423 | -3.55pp | 9.52 | 9.09 | 8.45 | 10.14 |
| theta | 4.49% | 7.51% | 1.959 | +0.75pp | 5.80 | 6.23 | 7.81 | 10.20 |
| candidate | 5.33% | 8.85% | 2.254 | +1.75pp | 7.09 | 8.40 | 9.03 | 10.89 |

- relMAE (candidate vs naive): 0.957 | categories beating naive: 6/11 | beating theta: 1/11
- vs naive mean-cat: +0.44pp (CI -0.22..+1.18) | pooled +1.20pp (CI +0.03..+2.40)
- vs theta mean-cat: -1.34pp (CI -2.39..-0.50) | pooled -0.85pp (CI -1.49..-0.35)

REGATE v2 mirror: C1 FAIL (CI -0.22.., floor +0.44pp, pooled guard +1.20pp) | C2 FAIL (worst sporting_goods_and_hobby +2.78pp, retail_total -0.89pp) | C3 FAIL (CI ..-0.50, worst vs theta clothing_and_accessories +2.96pp) → **FAIL**

## full_exploration — first-release actuals (descriptive) (24 origins)

| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 7.43% | 8.64% | 2.371 | -5.35pp | 9.38 | 8.70 | 7.64 | 8.84 |
| theta | 3.64% | 5.13% | 1.431 | -1.13pp | 2.52 | 3.78 | 5.57 | 8.59 |
| candidate | 4.19% | 6.38% | 1.704 | -0.15pp | 3.85 | 6.03 | 6.73 | 8.86 |

- relMAE (candidate vs naive): 0.710 | categories beating naive: 10/11 | beating theta: 2/11
- vs naive mean-cat: +2.26pp (CI +1.52..+3.07) | pooled +3.24pp (CI +2.35..+4.21)
- vs theta mean-cat: -1.25pp (CI -2.41..-0.30) | pooled -0.55pp (CI -1.23..-0.02)

Coarse-CI note: 9 report / 24 full-exploration origin blocks — intervals are wide by construction and reported as such (scorecard precedent).

This scorecard is an exploration-side selection result. It promotes nothing; the only promotion path is the future-origin re-gate (REGATE_DESIGN.md).
