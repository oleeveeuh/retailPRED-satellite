# Recovered v1 training summary — `train_73_features.py` (deterministic re-derivation)

> v1 protocol, frozen 2025/2026 data, flaws inherited (monthly actuals expanded across a daily grid; holdout = last 52 rows ≈ 7 calendar weeks; inclusive rolling windows frozen into the CSVs)
>
> `recovered via deterministic re-derivation; original outputs were never committed` — this is recovery for the record and audit, **not** an accuracy claim, and nothing here is comparable to any v2 number (native monthly folds, point-in-time selection).

- Tag `v1-undergrad` @ `605d962bb0691dae5b06722cc2232a9eff051ea2` (2026-10-02); trainer unmodified, SHA-256 `5fcba763ab7b5d1c…`
- Environment: Python 3.11.17, joblib 1.4.2, lightgbm 4.5.0, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.7.1 (sklearn pinned to the original run's recorded 1.7.1 per Amendment A; LightGBM original version unrecorded in pickles)
- Runs: 2026-11-12T14:16:16.496995 (34.5s) and 2026-11-12T14:16:58.666756 (32.3s), identical invocations from the extracted tag root

## Table 1 — recovered metrics, per category × model

| Category | Model | Train MAE | Holdout MAE (last 52 rows) | Derived holdout MAPE % * |
|---|---|---:|---:|---:|
| automobile_dealers | RandomForest | 75.78 | 1,686.52 | 10.08 |
| automobile_dealers | LGBM | 40.71 | 1,818.22 | 10.91 |
| building_material_and_garden_equipment | RandomForest | 7.23 | 120.86 | 9.10 |
| building_material_and_garden_equipment | LGBM | 4.08 | 131.74 | 9.61 |
| clothing_and_clothing_accessories_stores | RandomForest | 4.29 | 83.37 | 9.13 |
| clothing_and_clothing_accessories_stores | LGBM | 2.44 | 60.44 | 6.45 |
| electronics_and_appliance_stores | RandomForest | 58.23 | 597.89 | 0.68 |
| electronics_and_appliance_stores | LGBM | 94.40 | 738.10 | 0.83 |
| food_and_beverage_stores | RandomForest | 47.28 | 1,060.58 | 10.61 |
| food_and_beverage_stores | LGBM | 25.99 | 1,062.79 | 10.52 |
| furniture_and_home_furnishings_stores | RandomForest | 1.04 | 4.11 | 0.32 |
| furniture_and_home_furnishings_stores | LGBM | 1.78 | 5.23 | 0.38 |
| gasoline_stations | RandomForest | 39.80 | 595.48 | 9.61 |
| gasoline_stations | LGBM | 23.64 | 345.33 | 5.43 |
| general_merchandise_stores | RandomForest | 13.33 | 375.34 | 11.87 |
| general_merchandise_stores | LGBM | 7.25 | 367.91 | 11.53 |
| health_and_personal_care_stores | RandomForest | 5.52 | 17.21 | 0.29 |
| health_and_personal_care_stores | LGBM | 9.83 | 23.95 | 0.38 |
| sporting_goods_hobby_and_musical_instrument_stores | RandomForest | 1.13 | 29.51 | 1.54 |
| sporting_goods_hobby_and_musical_instrument_stores | LGBM | 1.90 | 32.98 | 1.70 |
| total_sales | RandomForest | 39.05 | 1,010.09 | 11.20 |
| total_sales | LGBM | 23.15 | 1,000.81 | 11.02 |
| **mean of 11 categories** | **RandomForest** | **26.61** | **507.36** | **6.77** |
| **mean of 11 categories** | **LGBM** | **21.38** | **507.95** | **6.25** |

\* *Derived — MAE ≠ MAPE. This MAPE is computed from recovered holdout predictions on the 52-row daily-grid holdout; it is NOT the audit's 9.76 %/14.11 % quantity (a different metric on a different, weekly-grid split of 429 pseudo-replicated rows). Never merge them.*

## Table 2 — pickle-equivalence verdicts (gate 3)

| Pickle (backend/ml/models/) | Shape | max \|Δ\| committed vs recovered predictions | Embedded metrics == recovered? | Verdict |
|---|---|---:|---|---|
| `automobile_dealers_RandomForest_model.pkl` | model_data | 4.02e+03 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `automobile_dealers_LGBM_model.pkl` | model_data | 4.02e+03 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `building_material_and_garden_equipment_RandomForest_model.pkl` | model_data | 160 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `building_material_and_garden_equipment_LGBM_model.pkl` | model_data | 106 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `clothing_and_clothing_accessories_stores_RandomForest_model.pkl` | model_data | 2.63e+03 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `clothing_and_clothing_accessories_stores_LGBM_model.pkl` | model_data | 2.8e+03 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `electronics_and_appliance_stores_RandomForest_model.pkl` | model_data | 4.37e-11 | no | run-level within float noise (embedded metric differs ≤ 1e-13 — last-bit data-parse drift) |
| `electronics_and_appliance_stores_LGBM_model.pkl` | model_data | 0 | yes | run-level |
| `food_and_beverage_stores_RandomForest_model.pkl` | model_data | 2.32e+03 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `food_and_beverage_stores_LGBM_model.pkl` | model_data | 2.43e+03 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `furniture_and_home_furnishings_stores_RandomForest_model.pkl` | model_data | 6.82e-13 | no | run-level (predictions equal to float noise; embedded metric exact) |
| `furniture_and_home_furnishings_stores_LGBM_model.pkl` | model_data | 0 | yes | run-level |
| `gasoline_stations_RandomForest_model.pkl` | model_data | 2.45e+03 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `gasoline_stations_LGBM_model.pkl` | model_data | 2.8e+03 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `general_merchandise_stores_RandomForest_model.pkl` | model_data | 953 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `general_merchandise_stores_LGBM_model.pkl` | model_data | 963 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `health_and_personal_care_stores_RandomForest_model.pkl` | model_data | 2.73e-12 | no | run-level within float noise (embedded metric differs ≤ 1e-13 — last-bit data-parse drift) |
| `health_and_personal_care_stores_LGBM_model.pkl` | model_data | 0 | yes | run-level |
| `sporting_goods_hobby_and_musical_instrument_stores_RandomForest_model.pkl` | model_data | 6.82e-13 | no | run-level within float noise (embedded metric differs ≤ 1e-13 — last-bit data-parse drift) |
| `sporting_goods_hobby_and_musical_instrument_stores_LGBM_model.pkl` | model_data | 0 | yes | run-level |
| `total_sales_RandomForest_model.pkl` | model_data | 1.04e+04 | no | NOT reproducible on frozen data (trained on earlier data — see findings) |
| `total_sales_LGBM_model.pkl` | bare_Booster | 1.6e+03 | no | not-comparable (bare Booster — schema overwritten post-training; no embedded metrics) |

Reading: the four pure-daily categories (Family B — electronics, furniture, health, sporting) reproduce at run level: the committed pickles, the committed CSVs, and the original run's own recorded metrics agree (LGBM exactly; RF to ≤ 5e-14, last-bit parsing noise). The seven mixed-grain categories (Family A) do not: their committed models disagree with the frozen CSVs by hundreds-to-thousands of prediction units, because the committed models were trained on earlier data (pickle timestamps 2026-01-12; tag commit 2026-10-02). For those categories this release is a protocol-level re-derivation on the frozen data — **not** the original numbers, which survive only as the pickle-embedded metrics (also released, labeled). One anomaly: `total_sales_LGBM_model.pkl` is a bare `lightgbm.Booster`, not the trainer's model_data dict — overwritten by a later save path; it has no embedded metrics and is reported, never substituted.

## Table 3 — the honest comparison (three labeled columns, never mixed)

| Column 1: recovered holdout MAE, v1 protocol (this recovery) | Column 2: the audit's snapshot claims — a DIFFERENT metric on a DIFFERENT split | Column 3: context only — the recorded fabrication |
|---|---|---|
| Mean holdout MAE across 11 categories: **507.95** (LGBM), **507.36** (RandomForest); per-category values in Table 1. v1 protocol, frozen 2025/2026 data, flaws inherited. Units: target units on a 52-row daily-grid holdout. **Not an accuracy claim. Not comparable to column 2 or to any v2 number.** | Quoted from `docs/v2/V1_BASELINE.md` (frozen 2026-11 audit baseline): LightGBM **9.76 %**, Random Forest **10.06 %**, Seasonal Naive **14.11 %** — average MAPE, 429 logged rows, “Monthly actuals repeated across weekly rows; trained with inclusive target features” (source: `training_outputs/validation_metrics.json`, generated 2026-02-14 — a different, weekly-grid validation path, never committed to the tag). Pseudo-replicated targets; **not** mergeable with column 1. | The deployed demo's headline — “90.2 % accuracy, 0.26 % MAPE” — is recorded by `docs/v2/PUBLICATION_AUDIT.md` as **fabricated or internally inconsistent**. Appears here solely as the recorded fabrication. **Not data.** |

Caption: **every number on this page inherits v1's measurement flaws** — monthly actuals expanded across a daily grid, a “52-week” holdout that is actually the last 52 rows (≈ 7 calendar weeks), inclusive rolling windows frozen into the CSVs, and for seven categories committed models that predate the frozen data. Their value is historical recovery and audit, not accuracy.

## Provenance

Full provenance — tag SHA, all 11 data SHAs, trainer SHA, 42-pickle SHA inventory, library versions, run transcripts with digests, gate-by-gate outcomes, and the original run's embedded metrics (labeled) — is in [v1_training_73features_summary.recovered.json](v1_training_73features_summary.recovered.json). Predeclaration: [V1_PERFORMANCE_RECOVERY.md](../V1_PERFORMANCE_RECOVERY.md) (committed before any execution, git order verifiable); Amendment A records the sklearn-1.7.1 correction, also pre-execution. Findings: FINDINGS.md §14.
