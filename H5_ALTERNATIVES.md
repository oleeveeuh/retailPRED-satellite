# H5 — literature-backed alternatives

Implemented here: **the theta method** (run via `scripts/run_candidate.py
--candidate h5_theta`, results in `results/cand_h5_theta.json`). Sketched
rather than run: **MIDAS-style mixed-frequency features** and a **small
dynamic factor model** — both with explicit reasons tied to this panel and
the compute boundary.

## Implemented: theta method

Assimakopoulos & Nikolopoulos (2000), "Theta model: a decomposition approach
to forecasting"; non-seasonal variant, because the MRTS panel input is already
seasonally adjusted. Implementation (`vintageforensics.candidates.ThetaMethod`):

- Theta-lines via the second-difference decomposition
  `Z_t(θ) = θ·y_t + (1−θ)·(a + b·t)`, with `a + b·t` the OLS line over the
  full training window — the standard decomposition (equivalent to the
  Hyndman & Billah 2003 formulation for the θ ∈ {0, 2} case).
- θ = 0 line forecast: the OLS line evaluated at future months (the trend
  extrapolation).
- θ = 2 line forecast: simple exponential smoothing state after the last
  training point; α chosen to minimize one-step in-sample SSE over a fixed
  grid 0.01..0.99 — training data only, no origin leakage.
- Forecast: `ŷ(h) = ½·(SES₂ + a + b·(n−1+h))`.
- Fits per series per origin through the same vintage harness as every other
  model; no parameters touched after seeing any evaluation origin.

## Sketched: MIDAS-style mixed-frequency features

Classic MIDAS (Ghysels, Santa-Clara & Valkanov 2004) regresses a low-frequency
target on higher-frequency indicators with a parsimonious distributed-lag
weight function, e.g.
`y_t = β₀ + Σ_k θ_k(λ)·x^(k)_t + ε_t` with Almon/Beta-lag weights over the
~13 weekly observations inside a month.

Why it is only sketched here:

1. **The archive has no higher-frequency regressors.** The vintage panel is
   monthly-only; a faithful MIDAS needs new data sources (weekly initial
   claims, daily gasoline spot prices, weekly CPI), fetched per vintage date
   — a new ingestion pipeline outside this repo's boundary.
2. **The failure shape doesn't ask for more information at short horizons.**
   LightGBM already wins h=1 by +2.9pp; the deficit is h=12 recursion
   amplification and calm-regime variance — not a nowcasting-information gap
   that higher-frequency data would fill.
3. Parameter budget: the current global model already has 11×~300 training
   rows per origin; MIDAS-style Almon aggregation of the existing monthly
   lags would be a regularization variant, closer in spirit to H4 than to
   genuine mixed-frequency nowcasting.

## Sketched: small dynamic factor model (DFM)

Nowcasting framework of Giannone, Reichlin & Small (2008): state-space model
`x_t = Λ f_t + e_t`, factors following a VAR(1), estimated by EM/Kalman
filter on each vintage, forecasts from the filtered factor path plus
idiosyncratic AR terms.

Why it is only sketched here:

1. **The panel has essentially one common factor and already carries it.**
   `retail_total` is registered alongside its 10 component categories; the
   categories co-move almost entirely with it (Phase 1: pooled aggregate WAPE
   5.87% vs mean-category 9.55% — the common component is the easy part).
   A DFM's point forecast at h≥3 converges to the common trend, which naive
   already tracks at 5.7–6.3%.
2. **Estimated compute is at the boundary.** A faithful per-origin DFM
   (11 series, EM re-estimated at every origin, ragged-edge mixture of
   preliminary/revised months) is a laptop-hour-scale build; the brief says
   sketch beyond that. The cheaper discriminative test — does the common
   trend carry h=12? — is already answered by the naive benchmark itself.
3. **What a DFM would plausibly fix is not where the deficit lives.** The
   failure concentrates in volatile idiosyncratic categories (clothing,
   sporting goods) at long horizons — the part a 1-factor DFM smooths toward
   the common component, i.e., toward naive's answer, which is the H3
   direction and is already measured there.
