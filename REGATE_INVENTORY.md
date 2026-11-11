# REGATE_INVENTORY — the post-2025 calm-origin grid, counted

Status: **measured 2026-11-10** by the predeclared
[scripts/regate_inventory.py](scripts/regate_inventory.py) (commits `bc362b0`,
`657487c`; results in
`results/regate_origin_inventory.json`). Companion to
[REGATE_DESIGN.md](REGATE_DESIGN.md) and FINDINGS §5.1. No model was fitted
and no fold was scored; the spent holdout was touched only as an enumeration
cross-check.

## Verification

The enumeration reproduced the frozen reference grid exactly: all 24
exploration origins and all 8 holdout origins of
`reference/vintage_backtest_folds.csv`, on identical protocol constants
(horizons 1/3/6/12, min-train 60, 45-day availability, step 1,
latest-revision actuals). The verdict below therefore reflects the archive,
not machinery drift.

## Verdict: the reconstructable re-gate grid is EMPTY

- **0 of 24 required calm origins exist.** Every candidate origin from
  2025-04 through 2025-08 — the only candidates the panel's calendar end
  (2026-08) could mature — fails the frozen 45-day availability coverage:
  each as-of panel ends one month short of its origin month.
- **Why:** ALFRED archives a vintage on the 15th of each month; MRTS advance
  releases land mid-month, usually days *after* the snapshot cut. A vintage
  dated M/15 typically knows through M−2, while the frozen rule demands the
  origin month itself. Release-day live pulls — which the 2026 pipeline
  would record — know values the archive's snapshot dates systematically
  miss.
- **Consequence: the released 2025-04 → 2026-08 months (seventeen of them)
  are permanently unrecoverable as point-in-time gate origins.** Their final
  values exist and may be reported descriptively, but they can never serve
  the re-gate without a goalpost move.
- Historical context: the 45-day rule has always been sparse — 2–5 surviving
  origins per year since 2017 (2017:3, 2018:5, 2019:5, 2020:2, 2021:5,
  2022:4, 2023:3, 2024:3, 2025:2). Post-2025-03 the rate is zero on this
  archive.

## Timeline to ≥24 calm origins (live-pipeline path only)

FINDINGS §5.1's stated mechanism — the unpaused schedule recording one
vintage per month — is the only accumulation path. From the current edge
(2026-08), assuming monthly publication resumes and every origin passes
coverage:

| Counting rule | 24th origin | Calendar end needed | ETA |
|---|---|---|---|
| Accumulated (origin exists) | 2027-03 | 2027-03 | **≈ 2027-06** |
| Fully matured (h=12 targets released) | 2027-03 | 2028-03 | **≈ 2028-06** |

FINDINGS' "≈ end of 2027" sits between the two rules. REGATE_DESIGN.md does
not yet pin which one governs window opening.

## Decisions this inventory surfaces (owner, before the first live origin)

1. **Pin the counting rule** — accumulated vs fully-matured origins. A ~1
   year swing; under C4, pinning it after origins accumulate is void.
2. **Pin the availability semantics for live-recorded origins** — the 45-day
   rule was declared for retrospective ALFRED reconstruction. Live origins
   carry actual pull timestamps; the re-gate should state whether as-of is
   the recorded ingest time (release-day knowledge) or the declared 45-day
   bound. Same C4 logic: decide before the first vintage accumulates.
3. **Unpause is the critical path.** Every month the schedule stays paused
   is a month the re-gate window does not accumulate. The first
   live-recorded origin will be the first Census release after unpausing
   (September/October 2026, currently pending publication — verified absent
   from the Census API on 2026-11-10).
