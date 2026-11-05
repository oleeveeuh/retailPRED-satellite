#!/usr/bin/env python3
"""Research-overview figures: window-scoped model comparison and the policy's
per-category REPORT outcome. Reads only frozen results JSONs."""

from __future__ import annotations

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
    "global_lightgbm": ("LightGBM", AQUA),
    "theta_method": ("theta", ORANGE),
    "regime_switch_policy": ("policy", BLUE),
}
WINDOW_LABELS = {
    "exploration_2017_2022": "exploration\n2017-10 .. 2022-10",
    "report_2021_2022": "report\n2021-01 .. 2022-10",
    "holdout_2023_2025": "holdout\n2023-01 .. 2025-03\n(frozen single run)",
}


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def research_windows(summary) -> None:
    rows = summary["rows"]
    windows = list(WINDOW_LABELS)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    n_models = 4
    width = 0.19
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
    ax.text(0.99, 0.965, "policy evaluated on the report window only\n"
            "(single frozen pass); holdout never re-derived",
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
    bars = ax.barh(names, values, color=[BLUE if v > 0 else "#e34948" for v in values],
                   height=0.62, zorder=3)
    ax.axvline(0, color="#c3c2b7", lw=1, zorder=2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("WAPE delta (pp), naive − policy on the report window; "
                  "positive = policy better")
    for bar, value in zip(bars, values):
        inside = abs(value) > 1.6
        ax.text(value - 0.12 if inside and value > 0 else value + 0.12,
                bar.get_y() + bar.get_height() / 2, f"{value:+.2f}",
                va="center", ha="right" if inside and value > 0 else "left",
                fontsize=8.5, color="#ffffff" if inside else INK_2)
    ax.margins(x=0.14)
    improved = primary["count"]
    ax.text(0.99, 0.965, f"{improved} of 11 categories improved — "
            "single frozen evaluation", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "policy_categories_report.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    research_windows(load("mrts_window_summary.json"))
    policy_categories(load("policy_report.json"))
    print("figures written:", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
