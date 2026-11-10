#!/usr/bin/env python3
"""Research-overview figures: window-scoped model comparison, the
identical-protocol family benchmark, per-horizon small multiples, the H7
feature ladder, and the policy's per-category REPORT outcome. Reads only
frozen results JSONs and fold CSVs (stop-on-mismatch verification against
the published aggregates before drawing).

Two render modes:

* default (no --theme): the canonical figures/ renders, unchanged.
* --theme fieldnotes: the retailPRED site's field-notes palette (validated
  categorical trio + neutral baseline, CVD-checked on the cream surface),
  optionally --out DIR and --scale 2 for retina. In this mode the two policy
  diagnostic figures (tau sweep, reliability) are ALSO regenerated from the
  frozen policy_train.json / policy_report.json — the canonical copies of
  those stay owned by scripts/eval_policy.py.

No results file is ever written: this script reads frozen JSONs only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES, ROOT  # noqa: E402

BLUE = "#2a78d6"      # slot 1: regime-switch policy
ORANGE = "#eb6834"    # slot 2: theta method
AQUA = "#1baf7a"      # slot 3: global LightGBM
PURPLE = "#7d5ba6"    # slot 4: Chronos-2 zero-shot
BROWN = "#8c6d46"     # slot 5: NHITS trained
INK = "#0b0b0b"
INK_2 = "#52514e"     # naive baseline
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# The retailPRED field-notes palette — owner-chosen deep tones on cream:
# sumi-green #1E342E (naive), muted red #a43b36 (theta), sage #89927a
# (policy), sakura rose #d4708c (LightGBM). Grouped-bar order is
# naive | policy | theta | LightGBM so the two deep tones are separated by
# the sage and the rose; adjacency on cream validates with CVD 31.9 /
# normal 33.7 (worst adjacent pair rose<->wine). The two deep tones sit
# outside the strict lightness band by design — identity is carried by the
# direct value labels, legends, and the lightness ordering. Diverging
# pair: sage = policy-better, muted red = worse.
FIELDNOTES = {
    "surface": "#fbf9f4",       # cream card surface the figures sit on
    "ink": "#25251f",
    "ink2": "#52514e",
    "muted": "#77776c",
    "grid": "#d8d1c4",
    "edge": "#c9c2b2",
    "font_sans": ["Source Sans 3", "IBM Plex Sans", "DejaVu Sans"],
    "model_colors": {
        "seasonal_naive": "#1E342E",       # sumi-green (owner pick, neutral baseline)
        "theta_method": "#482728",         # muted wine (owner pick)
        "regime_switch_policy": "#89927a", # sage (site accent)
        "global_lightgbm": "#d4708c",      # sakura rose (owner pick)
        # Two added slots for the H6/H8 family benchmarks. CVD-validated on
        # cream against the four above (scripts/validate_palette.js, worst
        # adjacent pair 15.3 protan / 16.0 normal); the two deep owner tones
        # stay outside the strict lightness band by the suite's documented
        # convention — direct value labels and the legend carry identity.
        "nhits_trained": "#1f7a68",        # pine
        "chronos_2_zero_shot": "#2f5aa8",  # indigo
    },
    "positive": "#89927a",     # sage — the site accent; caption and dict
                               # comment both promise sage positives
                               # (#2b579b here was a koiai-indigo leftover)
    "negative": "#a43b36",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

SHORT = {
    "building_material_and_garden": "building materials",
    "clothing_and_accessories": "clothing",
    "electronics_and_appliances": "electronics",
    "food_and_beverage": "food & beverage",
    "furniture_and_home_furnishings": "furniture",
    "gasoline_stations": "gasoline stations",
    "general_merchandise": "general merchandise",
    "health_and_personal_care": "health & personal care",
    "motor_vehicle_and_parts": "motor vehicles & parts",
    "retail_total": "retail total",
    "sporting_goods_and_hobby": "sporting goods",
}

MODEL_STYLE = {
    "seasonal_naive": ("naive", INK_2),
    "regime_switch_policy": ("policy", BLUE),
    "theta_method": ("theta", ORANGE),
    "global_lightgbm": ("LightGBM", AQUA),
}
# The identical-protocol family benchmark (H6 Chronos, H8 NHITS beside the
# frozen naive / theta / gate-LightGBM records). Display order puts the
# neutral baseline first and the frontier last; adjacency is CVD-validated
# in the themed palette (see FIELDNOTES comment).
FAMILY_ORDER = [
    ("seasonal_naive", "naive"),
    ("nhits_trained", "NHITS (trained)"),
    ("chronos_2_zero_shot", "Chronos-2 (zero-shot)"),
    ("global_lightgbm", "LightGBM (gate)"),
    ("theta_method", "theta"),
]
FAMILY_COLORS = {
    "seasonal_naive": INK_2,
    "nhits_trained": BROWN,
    "chronos_2_zero_shot": PURPLE,
    "global_lightgbm": AQUA,
    "theta_method": ORANGE,
}
WINDOW_LABELS = {
    "exploration_2017_2022": "exploration\n2017-10 .. 2022-10",
    "report_2021_2022": "report\n2021-01 .. 2022-10",
    "holdout_2023_2025": "holdout\n2023-01 .. 2025-03\n(frozen single run)",
}

# Themed renders add a small surface gap between adjacent grouped bars
# (mark spec); the canonical style keeps its original geometry.
BAR_GAP = 0.0

# Diverging pair for the per-category delta figure (positive / negative).
DIVERGING = (BLUE, "#e34948")

# Themed sage bars are too low-chroma for white inside-labels; themed mode
# draws value labels outside the bar ends in ink instead.
LABEL_OUTSIDE = False


def apply_theme(style: dict) -> None:
    """Apply a named style block to matplotlib rcParams + module style maps."""
    global BAR_GAP, DIVERGING, LABEL_OUTSIDE
    plt.rcParams.update({
        "figure.facecolor": style["surface"],
        "axes.facecolor": style["surface"],
        "axes.edgecolor": style["edge"],
        "axes.labelcolor": style["ink2"],
        "xtick.color": style["muted"],
        "ytick.color": style["muted"],
        "text.color": style["ink"],
        "grid.color": style["grid"],
        "font.sans-serif": style["font_sans"],
    })
    colors = style["model_colors"]
    MODEL_STYLE["seasonal_naive"] = ("naive", colors["seasonal_naive"])
    MODEL_STYLE["global_lightgbm"] = ("LightGBM", colors["global_lightgbm"])
    MODEL_STYLE["theta_method"] = ("theta", colors["theta_method"])
    MODEL_STYLE["regime_switch_policy"] = ("policy", colors["regime_switch_policy"])
    FAMILY_COLORS.update({
        "seasonal_naive": colors["seasonal_naive"],
        "nhits_trained": colors["nhits_trained"],
        "chronos_2_zero_shot": colors["chronos_2_zero_shot"],
        "global_lightgbm": colors["global_lightgbm"],
        "theta_method": colors["theta_method"],
    })
    BAR_GAP = 0.02
    DIVERGING = (style["positive"], style["negative"])
    LABEL_OUTSIDE = True


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def research_windows(summary) -> None:
    rows = summary["rows"]
    windows = list(WINDOW_LABELS)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    n_models = 4
    width = 0.19 - BAR_GAP
    x = np.arange(len(windows))
    for slot, (model, (label, color)) in enumerate(MODEL_STYLE.items()):
        offsets = x + (slot - (n_models - 1) / 2) * width
        values, positions = [], []
        for xi, window in zip(offsets, windows):
            row = next((r for r in rows if r["evaluation_window"] == window
                        and r["model_name"] == model), None)
            if row is None:
                continue
            values.append(100 * row["metric_value"])
            positions.append(xi)
        bars = ax.bar(positions, values, width=width, color=color, label=label, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, [WINDOW_LABELS[w] for w in windows], fontsize=8.5)
    ax.set_ylabel("mean-category WAPE (%)")
    ax.set_ylim(0, 12.4)
    ax.set_title("MRTS mean-category WAPE by evaluation window — lower is better",
                 fontsize=10, color=INK_2, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncols=2)
    ax.text(0.99, 0.965, "policy: report window only (single frozen pass)\n"
            "holdout: frozen single run; LightGBM slice previously\n"
            "observed in the gate run (gate reference folds)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "research_windows.png", bbox_inches="tight")
    plt.close(fig)


def _mean_cat_by_horizon(path, model_names, max_origin="2022-10-01"):
    """Mean-category WAPE (%) per horizon from a frozen fold CSV (stdlib csv
    only). WAPE per (series, horizon) = sum|pred-actual| / sum|actual|, then
    the equal mean across the 11 categories — the protocol metric."""
    num = defaultdict(float)
    den = defaultdict(float)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model_name"] not in model_names:
                continue
            if max_origin and row["forecast_origin"] > max_origin:
                continue
            actual = float(row["actual"])
            key = (row["series_id"], int(row["horizon"]))
            num[key] += abs(float(row["prediction"]) - actual)
            den[key] += abs(actual)
    per_horizon = defaultdict(list)
    for (series, horizon), numerator in num.items():
        per_horizon[horizon].append(100 * numerator / den[(series, horizon)])
    return {h: sum(v) / len(v) for h, v in sorted(per_horizon.items())}


def _verify_published(computed, published, label):
    """Stop-on-mismatch gate: every computed series must reproduce the
    published two-decimal aggregates before any figure is drawn."""
    for horizon, expected in zip((1, 3, 6, 12), published):
        got = computed.get(horizon)
        if got is None or abs(got - expected) > 0.005:
            sys.exit(f"verification gate failed: {label} h={horizon} "
                     f"computed {got:.4f} vs published {expected}")


def model_families(summary) -> None:
    """The identical-protocol family table (FINDINGS §12): theta, Chronos-2,
    NHITS, gate LightGBM, seasonal naive on the same origin grid. The holdout
    column is the single post-hoc descriptive pass — annotated as such."""
    rows = summary["rows"]
    chronos_x = load("cand_h6_chronos.json")
    nhits_x = load("cand_h8_nhits.json")
    chronos_h = load("holdout_chronos.json")
    nhits_h = load("holdout_nhits.json")

    def row_value(window, model):
        r = next(r for r in rows if r["evaluation_window"] == window
                 and r["model_name"] == model)
        return 100 * r["metric_value"]

    exploration = {
        "seasonal_naive": row_value("exploration_2017_2022", "seasonal_naive"),
        "global_lightgbm": row_value("exploration_2017_2022", "global_lightgbm"),
        "theta_method": row_value("exploration_2017_2022", "theta_method"),
        "chronos_2_zero_shot": 100 * chronos_x["windows"]["full"]["candidate"]["mean_category_wape"],
        "nhits_trained": 100 * nhits_x["windows"]["full"]["candidate"]["mean_category_wape"],
    }
    holdout = {
        "seasonal_naive": row_value("holdout_2023_2025", "seasonal_naive"),
        "global_lightgbm": row_value("holdout_2023_2025", "global_lightgbm"),
        "theta_method": row_value("holdout_2023_2025", "theta_method"),
        "chronos_2_zero_shot": 100 * chronos_h["candidate"]["mean_category_wape"],
        "nhits_trained": 100 * nhits_h["candidate"]["mean_category_wape"],
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    windows = ["exploration_2017_2022", "holdout_2023_2025"]
    data = {"exploration_2017_2022": exploration, "holdout_2023_2025": holdout}
    n_models = len(FAMILY_ORDER)
    width = 0.15 - BAR_GAP
    x = np.arange(len(windows))
    for slot, (model, label) in enumerate(FAMILY_ORDER):
        offsets = x + (slot - (n_models - 1) / 2) * width
        values = [data[w][model] for w in windows]
        bars = ax.bar(offsets, values, width=width, color=FAMILY_COLORS[model],
                      label=label, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, ["exploration\n2017-10 .. 2022-10\n(24 origins)",
                      "holdout\n2023-01 .. 2025-03\n(single post-hoc pass)"],
                  fontsize=8.5)
    ax.set_ylabel("mean-category WAPE (%)")
    ax.set_ylim(0, 11.4)
    ax.set_title("One vintage protocol, five model families — lower is better",
                 fontsize=10, color=INK_2, loc="left", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005, 1.0, 0.06),
              ncols=5, fontsize=8, frameon=False,
              handlelength=1.1, handletextpad=0.45, columnspacing=0.9)
    ax.text(0.99, 0.985, "holdout: post-hoc, descriptive only —\n"
            "no selection or promotion input; LightGBM slice\n"
            "previously observed in the gate run",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_families.png", bbox_inches="tight")
    plt.close(fig)


def horizon_families() -> None:
    """Exploration-window mean-category WAPE by horizon, one panel per model
    with the seasonal naive as the dotted neutral reference — small multiples
    so six series never have to fight over one legend."""
    gate_folds = ROOT / "reference" / "vintage_backtest_folds.csv"
    naive_h = _mean_cat_by_horizon(gate_folds, {"seasonal_naive"})
    gate_h = _mean_cat_by_horizon(gate_folds, {"global_lightgbm"})
    theta_h = _mean_cat_by_horizon(ROOT / "results" / "folds_h5_theta.csv",
                                   {"theta_method"})
    chronos_h = _mean_cat_by_horizon(ROOT / "results" / "folds_h6_chronos.csv",
                                     {"chronos_2_zero_shot"})
    nhits_h = _mean_cat_by_horizon(ROOT / "results" / "folds_h8_nhits.csv",
                                   {"nhits_trained"})
    direct_h = {int(k): 100 * v["mean_category"] for k, v in
                load("h7_scorecard.json")["windows"]["full_exploration"]["final_vintage"]["stats"]["candidate"]["by_horizon"].items()}

    # Verification gates: reproduce the published two-decimal tables
    # (CHRONOS_MRTS.md / NHITS_MRTS.md / h7_scorecard.md) before drawing.
    _verify_published(naive_h, (9.52, 9.09, 8.45, 10.14), "seasonal naive")
    _verify_published(gate_h, (7.17, 8.49, 9.63, 12.89), "gate LightGBM")
    _verify_published(theta_h, (5.80, 6.23, 7.81, 10.20), "theta")
    _verify_published(chronos_h, (5.88, 6.64, 8.10, 10.37), "Chronos-2")
    _verify_published(nhits_h, (6.08, 7.30, 9.60, 12.64), "NHITS")
    _verify_published(direct_h, (7.11, 8.48, 8.99, 11.47), "direct LightGBM + growth")

    panels = [
        ("naive (baseline)", naive_h, None, "-"),
        ("theta", theta_h, FAMILY_COLORS["theta_method"], "-"),
        ("Chronos-2 (zero-shot)", chronos_h, FAMILY_COLORS["chronos_2_zero_shot"], "-"),
        ("NHITS (trained)", nhits_h, FAMILY_COLORS["nhits_trained"], "-"),
        ("LightGBM (gate, recursive)", gate_h, FAMILY_COLORS["global_lightgbm"], (0, (5, 3))),
        ("LightGBM (direct + growth)", direct_h, FAMILY_COLORS["global_lightgbm"], "-"),
    ]
    horizons = [1, 3, 6, 12]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4), sharex=True, sharey=True)
    for idx, (ax, (title, values, color, dash)) in enumerate(zip(axes.flat, panels)):
        if idx != 0:  # every panel overlays the dotted naive reference
            ax.plot(horizons, [naive_h[h] for h in horizons], color=MUTED,
                    lw=1.4, ls=(0, (2, 2)), zorder=2)
        ax.plot(horizons, [values[h] for h in horizons], color=color or INK_2,
                lw=2.2, ls=dash, marker="o", ms=5,
                markerfacecolor=color or INK_2, zorder=3)
        ax.set_title(title, fontsize=9.5, color=INK_2, loc="left")
        end = values[12]
        ax.annotate(f"{end:.1f}", (12, end), xytext=(6, 2),
                    textcoords="offset points", fontsize=8.5, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean-category WAPE (%)")
    for ax in axes[1, :]:
        ax.set_xlabel("horizon (months ahead)")
        ax.set_xticks(horizons)
    axes[0, 0].set_ylim(4.5, 14.2)
    fig.suptitle("Exploration origins, final-vintage actuals — dotted line = seasonal naive",
                 x=0.02, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(FIGURES / "horizon_families.png", bbox_inches="tight")
    plt.close(fig)


def feature_ladder() -> None:
    """The H7 choose-window ladder: incremental mean-category WAPE delta per
    rung, origin stability, and the frozen verdict. Reads the frozen rung
    JSONs only."""
    rungs = [
        ("direct_base", "base (anchor)"),
        ("direct_growth", "+ growth"),
        ("direct_trend", "+ trend"),
        ("direct_cross", "+ cross-category"),
        ("direct_revision", "+ revision"),
        ("direct_external", "+ external macro"),
        ("removal_growth", "− growth (removal check)"),
    ]
    labels, deltas, stability, verdicts = [], [], [], []
    for rung, label in rungs:
        data = load(f"h7_rung_{rung}.json")
        advancement = data.get("advancement") or {}
        labels.append(label)
        deltas.append(advancement.get("mean_category_delta_pp", 0.0))
        improved = advancement.get("improved_origins")
        stability.append("—" if improved is None
                         else f"{improved}/{advancement.get('n_choose_origins', 15)}")
        note = data.get("verdict_note", "")
        if rung == "removal_growth":
            note = "CONFIRMS GROWTH"
        elif rung == "direct_base":
            note = "ANCHOR"
        verdicts.append(note)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(labels))[::-1]
    neutral = "#c9c2b2"
    colors = [neutral if v == "ANCHOR" else
              (DIVERGING[0] if d > 0 else DIVERGING[1])
              for d, v in zip(deltas, verdicts)]
    ax.barh(y, deltas, height=0.58, color=colors, zorder=3)
    ax.axvline(0, color="#c3c2b7", lw=1, zorder=2)
    ax.grid(axis="y", visible=False)
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("choose-window mean-category WAPE delta vs previous best (pp); "
                  "positive = better")
    for yi, delta, stable, verdict in zip(y, deltas, stability, verdicts):
        ax.text(delta + (0.008 if delta >= 0 else -0.008), yi,
                f"{delta:+.2f}" if verdict != "ANCHOR" else "0.00",
                va="center", ha="left" if delta >= 0 else "right",
                fontsize=8.5, color=INK_2)
        ax.text(0.28, yi, f"{stable} origins improved   {verdict}",
                va="center", ha="left", fontsize=8.5,
                color=INK_2 if verdict in ("ADVANCE", "CONFIRMS GROWTH") else MUTED,
                fontweight="bold" if verdict == "ADVANCE" else "normal")
    ax.set_xlim(-0.22, 0.62)
    ax.margins(y=0.08)
    ax.set_title("Feature-group ladder (direct per-horizon LightGBM) — choose "
                 "window only; the report window was scored once, after the "
                 "ladder concluded",
                 fontsize=10, color=INK_2, loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_ladder.png", bbox_inches="tight")
    plt.close(fig)


def policy_categories(report) -> None:
    primary = report["primary_categories_beating_naive"]
    deltas = {
        series: 100 * (primary["naive_category_wape"][series]
                       - primary["policy_category_wape"][series])
        for series in primary["policy_category_wape"]
    }
    data = sorted(deltas.items(), key=lambda kv: kv[1])
    names = [SHORT[s] for s, _ in data]
    values = [v for _, v in data]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.barh(names, values, color=[DIVERGING[0] if v > 0 else DIVERGING[1] for v in values],
                   height=0.62, zorder=3)
    ax.axvline(0, color="#c3c2b7", lw=1, zorder=2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("WAPE delta (pp), naive − policy on the report window; "
                  "positive = policy better")
    for bar, value in zip(bars, values):
        if LABEL_OUTSIDE:
            ax.text(value + (0.12 if value >= 0 else -0.12),
                    bar.get_y() + bar.get_height() / 2, f"{value:+.2f}",
                    va="center", ha="left" if value >= 0 else "right",
                    fontsize=8.5, color=INK_2)
        else:
            inside = abs(value) > 1.6
            ax.text(value - 0.12 if inside and value > 0 else value + 0.12,
                    bar.get_y() + bar.get_height() / 2, f"{value:+.2f}",
                    va="center", ha="right" if inside and value > 0 else "left",
                    fontsize=8.5, color="#ffffff" if inside else INK_2)
    ax.margins(x=0.14)
    improved = primary["count"]
    ax.text(0.99, 1.02, f"{improved} of 11 categories improved — "
            "single frozen evaluation", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "policy_categories_report.png", bbox_inches="tight")
    plt.close(fig)


def themed_policy_tau_sweep(train, report, style) -> None:
    """Tau sweep from the frozen CHOOSE OOF sweep (policy_train.json); the
    REPORT diamond is the frozen single-pass point from policy_report.json."""
    sweep = train["tau_sweep_choose_oof"]
    refs = train["tau_sweep_reference_mean_cat_wape"]
    tau = float(train["tau"])
    chosen = next(row for row in sweep if abs(row["tau"] - tau) < 1e-9)
    report_policy = 100 * report["secondary_mean_category_wape"]["policy"]
    policy_color = style["model_colors"]["regime_switch_policy"]
    taus = [row["tau"] for row in sweep]
    wapes = [100 * row["policy_mean_cat_wape"] for row in sweep]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(taus, wapes, color=policy_color, lw=2,
            label="switched policy (CHOOSE OOF)", zorder=4)
    for key, color, name in (
        ("naive", style["model_colors"]["seasonal_naive"], "naive"),
        ("theta", style["model_colors"]["theta_method"], "pure theta"),
        ("blend", "#b86a62", "fixed blend 0.5"),  # diluted crimson: half theta
    ):
        ax.axhline(100 * refs[key], color=color, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.axvline(tau, color=style["muted"], lw=1, ls=(0, (1, 2)), zorder=1)
    ax.scatter([tau], [100 * chosen["policy_mean_cat_wape"]], s=64, color=policy_color,
               edgecolor=style["surface"], linewidth=1.5, zorder=5)
    ax.scatter([tau], [report_policy], s=72, marker="D", color=style["ink"],
               edgecolor=style["surface"], linewidth=1.5, zorder=6)

    floor = min(min(wapes), report_policy)
    ax.set_ylim(floor * 0.90, max(wapes) * 1.02)
    label_x = 1.01
    ref_colors = {
        "naive": style["model_colors"]["seasonal_naive"],
        "theta": style["model_colors"]["theta_method"],
        "blend": "#b86a62",
    }
    for key, name in (("naive", "naive"), ("theta", "pure theta"), ("blend", "fixed blend 0.5")):
        value = refs[key]
        ax.annotate(f"{name} {100 * value:.2f}", xy=(label_x, 100 * value),
                    xycoords=("axes fraction", "data"), fontsize=8.5,
                    color=ref_colors[key],
                    va="center", annotation_clip=False)
    ax.annotate(f"frozen tau = {tau:.2f}", xy=(tau, 100 * chosen["policy_mean_cat_wape"]),
                xytext=(-12, 16), textcoords="offset points", fontsize=8.5,
                color=style["muted"], ha="right")
    ax.annotate("REPORT, evaluated once", xy=(tau, report_policy),
                xytext=(-10, -4), textcoords="offset points", fontsize=8.5,
                color=style["ink"], va="top", ha="right")
    ax.set_xlabel("tau: use theta when P(theta wins) >= tau")
    ax.set_ylabel("mean-category WAPE (%)")
    ax.set_title("Policy tau sweep — CHOOSE origins (out-of-fold); diamond = frozen REPORT point",
                 fontsize=10, color=style["ink2"], loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "policy_tau_sweep.png", bbox_inches="tight")
    plt.close(fig)


def themed_policy_reliability(report, style) -> None:
    """Reliability of P(theta wins) from the frozen calibration deciles in
    policy_report.json (both windows; never recomputed)."""
    calibration = report["calibration"]
    oof = calibration["reliability_choose_oof_deciles_isotonic_in_sample"]
    rep = calibration["reliability_report_deciles"]
    brier_oof = calibration["brier_choose_oof"]
    brier_rep = calibration["brier_report_frozen"]
    policy_color = style["model_colors"]["regime_switch_policy"]
    n_oof = sum(row["n"] for row in oof)
    n_rep = sum(row["n"] for row in rep)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lim = (0, 1)
    ax.plot(lim, lim, color=style["muted"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("perfect calibration", xy=(0.36, 0.41), fontsize=8.5,
                color=style["muted"], rotation=38)

    ax.plot([row["mean_p"] for row in oof], [row["realized_rate"] for row in oof],
            color=policy_color, lw=1.6, zorder=3)
    ax.scatter([row["mean_p"] for row in oof], [row["realized_rate"] for row in oof],
               s=52, color=policy_color, edgecolor=style["surface"], linewidth=1.2,
               zorder=4, label=f"CHOOSE OOF (n={n_oof}, Brier {brier_oof:.3f})")
    ax.scatter([row["mean_p"] for row in rep], [row["realized_rate"] for row in rep],
               s=64, marker="D", color=style["ink"], edgecolor=style["surface"],
               linewidth=1.2, zorder=5,
               label=f"REPORT frozen (n={n_rep}, Brier {brier_rep:.3f})")
    for i, row in enumerate(rep):
        ax.annotate(f"n={row['n']}", (row["mean_p"], row["realized_rate"]),
                    xytext=(8, 6 if i % 2 == 0 else -10),
                    textcoords="offset points", fontsize=7.5, color=style["ink2"])

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("mean P(theta wins) in decile bin")
    ax.set_ylabel("realized theta-win rate")
    ax.set_title("Reliability of P(theta wins) — deciles of scored pairs",
                 fontsize=10, color=style["ink2"], loc="left")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "policy_reliability.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=["fieldnotes"], default=None,
                        help="render with a named site palette (default: canonical style)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: the satellite figures/)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="render scale (2 = retina)")
    args = parser.parse_args()

    global FIGURES
    if args.out is not None:
        FIGURES = args.out
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.scale != 1.0:
        plt.rcParams["figure.dpi"] = 150 * args.scale

    if args.theme == "fieldnotes":
        apply_theme(FIELDNOTES)
        research_windows(load("mrts_window_summary.json"))
        policy_categories(load("policy_report.json"))
        themed_policy_tau_sweep(load("policy_train.json"), load("policy_report.json"), FIELDNOTES)
        themed_policy_reliability(load("policy_report.json"), FIELDNOTES)
        model_families(load("mrts_window_summary.json"))
        horizon_families()
        feature_ladder()
    else:
        research_windows(load("mrts_window_summary.json"))
        policy_categories(load("policy_report.json"))
        model_families(load("mrts_window_summary.json"))
        horizon_families()
        feature_ladder()
    print("figures written:", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
