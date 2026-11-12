# V1 performance recovery — deterministic re-derivation of the `v1-undergrad` training run

Status: **predeclaration, written and committed before any recovery code
ran, any venv was created, and any trainer was executed.** Owner direction
2026-11-12: the `v1-undergrad` tag preserves a complete, deterministic
training pipeline (committed data, committed code, fixed seeds), but its
performance numbers were never committed — the API served
`robust_training_summary.json`, and neither that file nor
`training_73features_summary.json` is in the tag. This experiment recovers
the missing numbers by re-deriving the run and proving the recovery
against the committed pickles. Any change to §§1–8 after the first
execution is a goalpost move and is void, same rule as
[REGATE_DESIGN.md](REGATE_DESIGN.md) §3/C4.

This is forensic archaeology following the theta fold-recovery playbook
(`scripts/holdout_theta_recovery.py`, FINDINGS §6): predeclaration first,
verification gates that stop on mismatch, every number labeled with what
it is and inherits. **Recovery, not reinterpretation** — v1's known flaws
are inherited and stated, never corrected.

## 1. The frozen question

- **Question.** What did v1's committed training pipeline
  (`backend/ml/train_73_features.py`) measure on v1's own frozen data, and
  does the committed pickle set prove deterministic recovery of that run?
- **Primary artifact recovered**: the never-committed
  `training_outputs/training_73features_summary.json` — per category ×
  model (RandomForest, LGBM): train MAE and test MAE on the temporal
  holdout (last 52 rows), under `random_state=42`, `n_estimators=200`.
- **Equivalence target**: the 42 committed pickles, which are the
  original run's own outputs and embed, per model, the original metrics
  dict (`model_data['metrics'] = {'train_mae', 'test_mae'}`) and the
  feature list — so recovery can be checked against the original run's
  *recorded numbers*, not merely against itself.
- **No comparability claims.** The recovered numbers describe v1's
  protocol on mixed-grain grids built from monthly actuals. They are not
  comparable to any v2 number (native monthly folds, point-in-time
  selection) and must never be presented as v2-adjacent accuracy. Their
  value is historical recovery and audit, not accuracy.

## 2. Frozen inputs (recorded before execution)

Tag: `v1-undergrad` in `~/retailPRED`, commit **`605d962bb0691dae5b06722cc2232a9eff051ea2`**
(2026-10-02 16:52:30 -0700, "ci: fix invalid secrets context in deploy job
condition"). The tag is read-only: never checked out; extracted 2026-11-12
via `git archive` into `/tmp/v1_recovery_tag/`. Committed pickles were
copied **before any execution** to `/tmp/v1_recovery_reference/` because
the trainer overwrites `backend/ml/models/` when it runs.

### 2.1 Trainer and context code

| File | SHA-256 |
|---|---|
| `backend/ml/train_73_features.py` (the recovery target) | `5fcba763ab7b5d1c229f78d8a6e62e91742f8c58be073faa33c2b04ca3448a08` |
| `backend/ml/train_statistical_models.py` (context only, out of scope) | `a657a900750d837b385e53a701b2e61f0a28c5b3834e0bd577e71600839d33fe` |
| `requirements.txt` (root; minimums) | `d820b01c35d51682b5f4efcce87e4b2545874a97db2c382d3dfe42b4250148e0` |
| `backend/requirements.txt` (backend; **exact pins**) | `de8d622ea1e90c93aae59ec57775d81fc073a68d05428650175fd9c116eb51f1` |

### 2.2 The 11 frozen data files

All: 76 columns = 73 features + `y` + `index` + `year`; no NaN in `y`.

| CSV (`project_root/data_multi_resolution/`) | SHA-256 |
|---|---|
| `retail_automobile_dealers_multi_resolution.csv` | `b1eba73e41f0d65cdc579693099abd5036d0fb6b3dec9897b06ced690d6d66d3` |
| `retail_building_material_and_garden_equipment_multi_resolution.csv` | `4d8b7e2f68581ad5e5dbaab0ef15e7713c6609697599bb63ca7b7525539ed1c4` |
| `retail_clothing_and_clothing_accessories_stores_multi_resolution.csv` | `12f7fb688f6aaf766416f50dd885b0a488851a7d3096b78d7c1e196f7c88f002` |
| `retail_electronics_and_appliance_stores_multi_resolution.csv` | `c14a21b820eff6fd0641608695bb54893373b09ab8b8f89bed4452359897c3eb` |
| `retail_food_and_beverage_stores_multi_resolution.csv` | `ee5ead8e818b730a518527e4cc5d12948f4a6ec34bf86b2aba4302e6b655a75d` |
| `retail_furniture_and_home_furnishings_stores_multi_resolution.csv` | `df1240333d433002384df5a0f2f400cb7981fa579df13ddbd148ad505de1993e` |
| `retail_gasoline_stations_multi_resolution.csv` | `8510aed9a45b21575dbda5fe4f40d6e348f44a5734588a2c3bdefaf9031bcb80` |
| `retail_general_merchandise_stores_multi_resolution.csv` | `04c15a455d2e4b62dfbfd1030d3707abf8735ec383c58884f701e87253580f3b` |
| `retail_health_and_personal_care_stores_multi_resolution.csv` | `e0f1f2e6c384f866860c1661a783ba8a19bebf0d4994fa2bcbbd639797f03b2b` |
| `retail_sporting_goods_hobby_and_musical_instrument_stores_multi_resolution.csv` | `51af637118d05c2d9607e1bea6a221081780a9c361cea873a529127a2bf265a8` |
| `retail_total_sales_multi_resolution.csv` | `b38fd8f9aee9645d42a14b048e911016d41fc57b8a2bf0fbea757fb4fc1c0b88` |

### 2.3 The 42 committed pickles (equivalence targets)

`backend/ml/models/` — 22 pickles written by this trainer (filename
pattern `{category_key}_{model_type}_model.pkl` matches the code exactly;
these are the primary equivalence targets):

| Pickle | SHA-256 |
|---|---|
| `automobile_dealers_LGBM_model.pkl` | `6783a8622e8146923d03972722343142c877da8e82c1270a3358861a77439638` |
| `automobile_dealers_RandomForest_model.pkl` | `f1b1a7105c8c2fe338dc87f6cd01536d0ce593c2cc06470570ad8688173c0fe3` |
| `building_material_and_garden_equipment_LGBM_model.pkl` | `a4941889fdd25ec9a3a89a6019d1cfea5c7d86f72a29808b07e3e8b26cbf0012` |
| `building_material_and_garden_equipment_RandomForest_model.pkl` | `9e880d8ed2da32e34af594ff8f7eb68de160b9c9cbd4fbd8c65b3f9b7c4663e6` |
| `clothing_and_clothing_accessories_stores_LGBM_model.pkl` | `dbf2069a32a7f11211263d7b455e02c5dcb9d7127d2be20a2403c3622813eae8` |
| `clothing_and_clothing_accessories_stores_RandomForest_model.pkl` | `4a7ab83026eb0172a8d1b25ec69b65c353adb27b02e3b281c612902855cf6f2d` |
| `electronics_and_appliance_stores_LGBM_model.pkl` | `292097d973299e04d862fecb3d004849f720a1028111a3498e95696dade4f15d` |
| `electronics_and_appliance_stores_RandomForest_model.pkl` | `b10caa855fbe283684540a21779b9af30f07b5a9cfe364922d896df487d6c4dc` |
| `food_and_beverage_stores_LGBM_model.pkl` | `4d82e52bcc104e5f5d42e695410b0850cc60d5f084afc77bb84327ec1614fa4e` |
| `food_and_beverage_stores_RandomForest_model.pkl` | `05ae43b859ae499aae38f58db743597ae2a8fe13012f3f506ce1b995bae4997e` |
| `furniture_and_home_furnishings_stores_LGBM_model.pkl` | `64fbc0c208d6e2fe926dae4398ecde625c4ebb8624db2f8791ad5c046af446c4` |
| `furniture_and_home_furnishings_stores_RandomForest_model.pkl` | `7534ab69b878b28589b24f1e461ef97a7799cd59d883ae65261432bdc2dc6d45` |
| `gasoline_stations_LGBM_model.pkl` | `9473d819bb7d94e890b4a3c73297df4c418027ce7f75ba49cbade5470eb0937e` |
| `gasoline_stations_RandomForest_model.pkl` | `5b659fafc443360a32e9a173c17561f2c0bcbd544b85b530865d8d7f6250c5ad` |
| `general_merchandise_stores_LGBM_model.pkl` | `eced4a9d6b3b68910c0348a4cb9ece63e6862a1bd45f31efbaa95f3ea3bbccda` |
| `general_merchandise_stores_RandomForest_model.pkl` | `5d5ff3d8d9584d23690f88e915da01353c4be5e7c0de14b98d79d76e70451a3d` |
| `health_and_personal_care_stores_LGBM_model.pkl` | `efd02f2436af71935a3c6691419e9ac41f890da372b12bf798ba8d8bd8c32367` |
| `health_and_personal_care_stores_RandomForest_model.pkl` | `e2c51f15b66e768ee76d8086d7af3d51d46589725e0ce5b4b32e567070d8ec5f` |
| `sporting_goods_hobby_and_musical_instrument_stores_LGBM_model.pkl` | `133ba94d447d1291db46a578e4d1efa67f8bdb152d08d12f2019fe44161aa2aa` |
| `sporting_goods_hobby_and_musical_instrument_stores_RandomForest_model.pkl` | `e439162907e9d7bdcd2fa810d4ec5c2171670095656d6c0b7d5c6b78bc0a708e` |
| `total_sales_LGBM_model.pkl` | `a060a398d5e07c25d6fa1dcaf62ff8a28c4d3800f513bdeaac63e8b3c6f6033b` |
| `total_sales_RandomForest_model.pkl` | `2f5459ff0dc887bc02ed68928935d46c16027905d9d85cb85df2bd1c8e9bbe91` |

`backend/training_outputs/models/` — 20 pickles in display-name
subdirectories. **Provenance caveat, stated up front**: this trainer
writes flat files into `backend/ml/models/` only; the display-name
layout does not match this trainer's output pattern, so these are
duplicated outputs of *possibly different origin or vintage* (different
bytes from their `backend/ml/models/` counterparts — e.g.
`automobile_dealers_LGBM` = `6783a862…` vs `Automobile_Dealers/LGBM` =
`5e94a94f…` at minimum differing in embedded timestamps). They are
loaded and compared as **secondary evidence** and reported per category,
with their actual origin classified from the embedded `model_data` keys;
they are never merged silently into the primary equivalence claim.
Note the inventory is 22 + 20 = 42, **not** 11 × 2 duplicated —
`total_sales` exists only under `backend/ml/models/`.

| Pickle (`backend/training_outputs/models/<Category>/`) | SHA-256 |
|---|---|
| `Automobile_Dealers/LGBM_model.pkl` | `5e94a94ff817e256320da3b4d94e0bf54bdfd720a62e82066a0c87c4b1f2f597` |
| `Automobile_Dealers/RandomForest_model.pkl` | `3e4dca3860e338d279d6e2fb4c7fef01e5f5e017b07cee390755178ad7bdde1b` |
| `Building_Materials_Garden/LGBM_model.pkl` | `dadbde1325c4c3cd1d418ce4fb760fc34884059f20026714a3055de218edaf05` |
| `Building_Materials_Garden/RandomForest_model.pkl` | `4aef95f0246ca5385b53d60e57b84503948708f4dbd502206ae6a0c724aae8ec` |
| `Clothing_Accessories/LGBM_model.pkl` | `a273a763308da7452c18c50ad4fb08fa71302a857ccf87520fecb8a6b4bd88d2` |
| `Clothing_Accessories/RandomForest_model.pkl` | `aeb70c4d5e3e8f79206ca691a20545c25e0f0f42b82dbe6e3568481a89ec3b35` |
| `Electronics_and_Appliances/LGBM_model.pkl` | `65d57b51b644fdf44b1cb1db6586e8eb43847078f89ec09b6264dc487192e17f` |
| `Electronics_and_Appliances/RandomForest_model.pkl` | `83214b1ea7d4bc17d1e7e75e3d561ea593476a8c59c6c187a8250b0170e97072` |
| `Food_Beverage_Stores/LGBM_model.pkl` | `79d318db3a0e4263fe566a11677d8d8a5d20188d5e60a79c3d5053a8b4c62521` |
| `Food_Beverage_Stores/RandomForest_model.pkl` | `debcf61477d84a04dae07abe9f2cc2e1214ff46230b9140745163aadcd5f002e` |
| `Furniture_Home_Furnishings/LGBM_model.pkl` | `6f084420c5e9be775b1389bdc550129f7954695838904f5f465bcee0e7efdf40` |
| `Furniture_Home_Furnishings/RandomForest_model.pkl` | `92f2aef67abbbe6def9b2de660ffe734657a05bb9cd2230f9114969419029eb3` |
| `Gasoline_Stations/LGBM_model.pkl` | `20ceadd61c159503555f3f89cf7a77c76c209714509b4cc4255c9cceb8f79949` |
| `Gasoline_Stations/RandomForest_model.pkl` | `8bcf6bc46c366fd149ace26112f50d528ae0402b9ca929fb00feb6687bfeb2b8` |
| `General_Merchandise/LGBM_model.pkl` | `6a0b006a16a91d574d10c6ad621ca9b3a3d58c8e4934a3a8544868e49be0539b` |
| `General_Merchandise/RandomForest_model.pkl` | `f36762ededb3351895a777e95e1a43c5a471f06a8ce34a306116cb03aa9fc120` |
| `Health_Personal_Care/LGBM_model.pkl` | `5982e944229446e23d0e8a7944f2cb364969301eb9bb052dfab80bd4fd23cf3c` |
| `Health_Personal_Care/RandomForest_model.pkl` | `9a543eb32e1715f0242fa95fa2e01185b2c74cab8f36dcbdf2a967c79fbda6e3` |
| `Sporting_Goods_Hobby/LGBM_model.pkl` | `2b8de74f7a11a7553328f487d9192d5e28a45504a9701c36da71ec0437ab7738` |
| `Sporting_Goods_Hobby/RandomForest_model.pkl` | `e738126cd35137e8555dc1b82b89f88d7ca532090dc902c7fdb0575cd71d9cbf` |

## 3. What the frozen data actually is (facts, recorded before execution)

The "multi-resolution" CSVs are stacked mixed-grain grids, not weekly
files. Two families, verified by direct inspection 2026-11-12:

- **Family A — 7 categories** (automobile_dealers,
  building_material_and_garden_equipment,
  clothing_and_clothing_accessories_stores, food_and_beverage_stores,
  gasoline_stations, general_merchandise_stores, total_sales): 456 rows =
  **120 monthly rows** (2015-01-31 … 2024-12-31, one value per month) +
  **335 daily rows** (2025-01-01 … 2025-12-01, `y` varying by day) +
  **1 trailing month-end row** (2025-12-31). Last 52 rows (the holdout):
  2025-10-12 … 2025-12-31, 50 daily steps + the trailing month-end row.
- **Family B — 4 categories** (electronics_and_appliance_stores,
  furniture_and_home_furnishings_stores, health_and_personal_care_stores,
  sporting_goods_hobby_and_musical_instrument_stores): 5,814 **pure
  daily** rows, 2010-01-01 … 2025-12-01. Last 52 rows: 2025-10-11 …
  2025-12-01.
- **Consequences inherited, not corrected**: the code comment "last 52
  weeks (1 year)" is wrong on these files — the holdout is the last 52
  *rows* ≈ **7 calendar weeks** of a daily grid; monthly actuals are
  expanded across a high-frequency grid (the audit's pseudo-replication,
  in daily rather than weekly form); `yoy_change` is **entirely NaN** in
  every Family B row (5,814/5,814) and NaN at grid warm-up edges in
  Family A; the trainer nonetheless reports `n_features: 73` — the dead
  column is still counted. All of this is stated in every released
  table; none of it is repaired.

## 4. Procedure (exact commands, frozen)

1. **Environment** (before any training): create
   `/tmp/v1_recovery_venv` with `python3.11 -m venv` (3.11.17, matching
   the tag's "Tested with Python 3.11" note) and install the tag's own
   **exact backend pins** — `backend/requirements.txt`'s ML stack
   (`scikit-learn==1.5.2`, `lightgbm==4.5.0`, `pandas==2.2.3`,
   `numpy==2.1.3`, `joblib==1.4.2`) — chosen over the root minimums
   because exact pins maximize the chance of run-level recovery; the
   root minimums (`pandas>=2.0, numpy>=1.24, scikit-learn>=1.3,
   lightgbm>=4.0`) are the recorded fallback if exact pins cannot
   install. Resolved versions of *everything installed* are recorded in
   every provenance payload. Never touch `~/retailPRED/.venv-v2` or
   `~/retailPRED-satellite/.venv`.
2. **Pickle loads first**: before any training, attempt to load all 42
   reference pickles in this environment. Any that fail to unpickle are
   reported per category with the exact error — never substituted.
3. **Run 1**: from the extracted root, `python backend/ml/train_73_features.py`
   (invoked as `/tmp/v1_recovery_venv/bin/python`). Captures the trainer's
   own stdout/stderr transcript. Writes
   `training_outputs/training_73features_summary.json` and overwrites
   `backend/ml/models/*.pkl` (the pre-run committed copies are safe in
   `/tmp/v1_recovery_reference/`).
4. **Run 2**: the identical invocation a second time, after snapshotting
   Run 1's summary and model outputs. Byte-compare the two summaries
   modulo timestamps; byte-compare the two pickles sets.
5. **Gates** (§5) applied to Run 1 vs Run 2 and vs the reference pickles.
6. **Release** (§6–§7): recovered summary with provenance wrapper,
   Markdown rendering, pickle-equivalence table, honest comparison
   table, FINDINGS §14.

Silent patches are forbidden: the trainer is run **unmodified** (SHA
frozen in §2.1); any compatibility shim would be a different experiment.
The 2026-11-11 rule applies — assert, don't trust.

## 5. Verification gates (stop-on-mismatch)

1. **Feature-count gate**: every category's loaded feature count == 73
   (from the trainer's own per-category log lines, cross-checked against
   the CSV column counts in §2.2). Mismatch → stop.
2. **Determinism gate**: two runs produce identical *metric payloads* —
   everything except `training_date` and `duration_seconds`, which are
   recorded separately. Additionally byte-compare the pickles from the
   two runs (modulo their embedded `training_time` fields, compared
   separately). Mismatch → stop with diagnostics.
3. **Pickle-equivalence gate**: for every pickle that loads, compare on
   identical holdout rows:
   a. committed-model holdout predictions vs re-trained-model holdout
      predictions — record max |Δ| per category × model;
   b. the metrics **embedded in the committed pickle** (the original
      run's own recorded numbers) vs the recovered run's metrics.
   Classification: **run-level** if (b) matches exactly and (a) is zero
   (or float-noise only, reported exactly); **metrics-level** if
   versions/threads drift produces nonzero (a) — report the deltas
   honestly and classify accordingly. A pickle that fails to unpickle is
   reported per category with the error, never substituted.
4. **Provenance gate**: every released artifact records the tag SHA, all
   input SHAs (§2), the extracted-command transcript, Python and library
   versions, and the label: `recovered via deterministic re-derivation;
   original outputs were never committed`.

Stop-on-mismatch: any gate failure halts release until diagnosed. An
honest failure — "not recoverable at run level, recoverable at metrics
level, reasons recorded" — is a valid outcome and is released as such.

## 6. Derived quantities (labeled, never silent)

- **Derived holdout MAPE** per category × model: computed from recovered
  holdout predictions and `y_test`, explicitly labeled *derived — MAE ≠
  MAPE, and this MAPE is not the audit's 9.76 %/14.11 % quantity* (that
  was a different metric on a different split over 429 pseudo-replicated
  rows). Never merged with any audit number.
- **Aggregate** recovered holdout MAE: mean across the 11 categories,
  labeled as the same simple mean v1's summary would have implied.

## 7. Release plan

- `results/v1_training_73features_summary.recovered.json` — provenance
  wrapper (tag SHA, input SHAs, versions, gate results, transcript
  digest) around the recovered metric payload.
- `results/v1_training_73features_summary.recovered.md` — per category ×
  model table: train MAE, holdout MAE, derived holdout MAPE; pickle
  equivalence table (loaded? / max |Δ| / embedded-metrics match /
  verdict).
- The honest comparison table: three labeled columns — recovered holdout
  MAE; the audit's snapshot claims (9.76 % / 14.11 % MAPE, 429
  pseudo-replicated rows, quoted from docs/v2/AUDIT.md as a different
  metric on a different split); the fabricated headline (90.2 % / 0.26 %)
  appearing solely as the recorded fabrication, never as data. Caption
  states plainly: these numbers inherit v1's measurement flaws; their
  value is historical recovery and audit, not accuracy. Labeling header
  on every released table: `v1 protocol, frozen 2025/2026 data, flaws
  inherited (monthly actuals expanded across a daily grid; holdout =
  last 52 rows ≈ 7 calendar weeks; inclusive rolling windows frozen into
  the CSVs)`.
- FINDINGS §14: what was committed, what wasn't, the gates' results, the
  pickle-equivalence outcome, and the corrected record that
  `train_73_features.py` substantiates the historical "73 features"
  claim (73-feature panel, 52-row temporal holdout, TimeSeriesSplit
  imported, fixed seeds) while no v1 *metric* survives except this
  recovery.
- Satellite commits, in order: predeclaration → recovery script(s) →
  artifacts + findings. Local-only unless the owner says push. Satellite
  test suite run at the end — green or unchanged.

## 8. Out of scope (reported as open questions, never silently skipped)

- `backend/ml/train_statistical_models.py` — not recovered here; the
  recovery covers the LGBM/RandomForest trainer only.
- `robust_training_summary.json` (the "robust" trainer's summary the API
  actually served) — not in the tag, trainer itself not identified in
  the tag; recorded as the continuing gap in the committed record.
- `backend/training_outputs/models/*` origin (§2.3 caveat) is classified
  from evidence, not assumed.
