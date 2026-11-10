#!/usr/bin/env python3
"""Research-overview figures: window-scoped model comparison and the policy's
per-category REPORT outcome. Reads only frozen results JSONs.

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
import json
import sys
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
    else:
        research_windows(load("mrts_window_summary.json"))
        policy_categories(load("policy_report.json"))
    print("figures written:", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
