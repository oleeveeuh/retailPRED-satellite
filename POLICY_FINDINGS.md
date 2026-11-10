# POLICY FINDINGS — the regime switch beats naive and the blend, but not pure theta

Outcome of the predeclared build in [POLICY.md](POLICY.md): a small, causal,
trained policy (LightGBM classifier, isotonic calibration, threshold
τ = 0.50) that chooses seasonal naive or the theta method per category and
per forecast origin. Status: **the policy beats naive decisively and the
fixed blend marginally on the REPORT window, and hits the 10/11 primary
count — but guardrail 2 (beat everything) fails because pure theta is
better still. By the predeclared rule this earns POLICY_FINDINGS.md, not
DRAFT_GATE_POLICY.md. Nothing here promotes a method; the serving decision
stays fail-closed.**

Sample-size constraint, restated: **n = 99 REPORT pairs** (9 origins × 11
categories) after a frozen policy fitted on **77 CHOOSE rows** (7 origins ×
11 categories with complete published-error history, out of 165 CHOOSE
pairs). Every uncertainty below is an origin-block bootstrap 95% CI
(2,000 resamples of origins — the independent unit; folds within an origin
are dependent). The spent 2023–2025 holdout was never read: 352 gate rows
were dropped at load and 0 theta rows exist past the boundary.

## 1. The frozen evaluation (one pass)

| REPORT window (origins 2021-01..2022-10) | mean-category WAPE | Δ vs policy |
|---|---:|---:|
| seasonal naive | 10.51% | +3.08pp (CI +0.70..+5.13) |
| **switched policy** | **7.43%** | — |
| fixed blend 0.5θ+0.5n | 7.77% | +0.34pp (CI −0.76..+1.28) |
| pure theta | 7.02% | **−0.41pp (CI −0.59..−0.21)** |

- **Primary metric:** the policy's WAPE beats naive in **10 of 11
  categories** (bootstrap CI 6..11) — the platform gate's criterion-2
  shape. The one loss is sporting goods & hobby (16.89% vs naive 15.34%).
- **Guardrail 1 (bounded downside, CHOOSE OOF):** passes with room to
  spare — the policy *beat* naive by 1.59pp out-of-fold on CHOOSE
  (11.71% vs 13.30%), and beat pure theta (11.96%) and the blend (12.08%)
  there too.
- **Guardrail 2 (beat everything, REPORT):** **fails.** The policy beats
  naive ✓ and the blend ✓, but pure theta's advantage is uniform — the
  entire delta CI sits below zero, so at n = 99 the policy never wins that
  comparison.

Regime-call quality on REPORT: 81/99 theta-calls, precision 0.716
(CI 0.58..0.82), recall 0.806 (CI 0.69..0.89). On CHOOSE OOF: precision
0.768, recall 0.935 (77 pairs).

## 2. Why the switch cannot beat the regime winner here

The labels flip exactly as the investigation predicted — theta wins 49.1%
of CHOOSE pairs (crash-and-calm mix) vs **72.7%** of REPORT pairs (the
volatile recovery, where trend-tracking dominates). But that flip is
between windows, not within them: REPORT is a *single-regime* evaluation.
In a window where one model wins nearly three quarters of the pairs, the
best fixed choice is a ceiling, and a switch can only lose by misranking.
Concretely: the policy called naive on 18 of 99 pairs; theta won 14 of
those 18 (its calls were right on 4). Pure theta takes all 72 of its
window's winners; the policy gave back exactly the 14 misranks — the
−0.41pp. The policy's naive-side skill, which made it the best method on
the regime-mixed CHOOSE window, has no surface to pay off on REPORT.

## 3. Calibration transferred poorly (measured, and part of the verdict)

- Brier: 0.155 on CHOOSE OOF → **0.276 on REPORT** — worse than calling
  every pair theta at its 72.7% base rate (0.204).
- The isotonic map (fitted on 77 OOF rows) collapses most mass onto one
  step: 7 of 10 REPORT deciles share P ≈ 0.789, and realized rates scatter
  0.60..1.00 across them with no ordering. The two low-probability deciles
  (means 0.133 and 0.302) realized **0.80** theta-win — the pairs the
  policy withheld from theta were mostly theta-winners.
- The dominant defect is structural: the theta-win base rate moves 59.7%
  (CHOOSE OOF) → 72.7% (REPORT) between windows, and a calibrator fitted
  inside CHOOSE cannot know that. Fixing it after seeing REPORT would be
  REPORT-informed tuning, which the protocol forbids; it is recorded as a
  finding, not patched.

Full tables: `results/policy_report.json` (`calibration` block); figures
`figures/policy_tau_sweep.png` (the frozen τ sits in the sweep's flat
bottom region, 0.30–0.52 — the choice is robust within CHOOSE) and
`figures/policy_reliability.png`.

## 4. What this means for retailPRED

1. **The platform's theta result is strengthened, not weakened.** A
   trained switch with honest causal features — including the high-value
   lagged-performance feature the forensics pointed to — cannot out-pick
   plain theta on the volatile-phase window, and theta holds the lowest
   holdout mean-category WAPE of the three evaluated models (+0.17pp vs
   naive at 7/11). The defensible claim is "lowest measured mean-category
   WAPE among the three evaluated models on every evaluated window," not
   "best real-time forecaster": the scorecard's origin-block CI cannot
   separate theta from naive on the holdout (+0.17pp, CI −0.17..+0.54),
   pooled WAPE there resolves in naive's favor, and the only holdout
   ranking the window resolves is theta over the gate LightGBM (FINDINGS
   §3 scorecard).
2. **The switch hypothesis is half-confirmed, on the half that matters
   less.** It dominates every fixed choice on regime-mixed CHOOSE origins
   (the "capture both extremes" thesis works in-sample-of-regime), but a
   gate window will usually be dominated by one regime, and then the
   switch is strictly a handicap. A switch could only earn its keep on an
   evaluation grid that contains both regimes in force — the pre-2023
   grid cannot provide that again, and the post-2025 calm grid (the
   re-gate path in FINDINGS §5) is precisely the window where the
   naive-side would need to carry it.
3. **Probability calibration across regime boundaries is the specific
   defect to beat.** Any future policy attempt should predeclare a
   base-rate-robust calibration target (e.g., regime-stratified
   calibration or ranking-first decision rules) and treat the REPORT
   Brier degradation measured here as the number to improve on.
4. Per the platform contract: **no DRAFT gate document is issued.** The
   policy layer's outputs are this file, `POLICY.md`, the results JSONs,
   and the tested `src/regime/` implementation.

## 5. Provenance

- Data: `data/alfred_mrts_vintages.csv`, SHA-256
  `b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86`.
- Labels from frozen folds only: `reference/vintage_backtest_folds.csv`
  (naive; holdout rows dropped at load) and `results/folds_h5_theta.csv`;
  blend reference weight `global_theta_weight = 0.5` from
  `results/h3b_theta_blend.json`.
- Protocol: [POLICY.md](POLICY.md), committed before any measurement
  (`a4be6c8`). Components, one commit each: features/labels (`44ba8bf`),
  training/calibration (`5606d97`), frozen evaluation (`051d444`), this
  document. Every results JSON embeds data hash, code revision, protocol,
  package versions, and the pair counts.
- Tests: 16 synthetic-panel cases in `tests/test_regime.py` covering every
  leakage assertion (publication arithmetic, 14-month floor, future-vintage
  invariance, tie resolution, split safety, loader boundary guard).
- Environment: python 3.11.17, lightgbm 4.5.0, scikit-learn 1.5.2, pandas
  2.2.3, numpy 2.1.3 (the pinned gate environment).
- One feature defect was caught and fixed pre-measurement: `drift_12` had
  been implemented as the mean log level; the committed definition is the
  POLICY.md §4.2 mean of the last 12 YoY log changes.
