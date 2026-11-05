#!/usr/bin/env python3
"""Phase 1 figures for ERROR_ANALYSIS.md (exploration origins only).

Static PNGs styled with the validated diverging pair (blue #2a78d6 = LightGBM
improves over naive, red #e34948 = LightGBM regresses), recessive chrome, and
direct labels. No holdout data is read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES, RESULTS

BLUE = "#2a78d6"      # diverging cool pole: LightGBM improves
RED = "#e34948"       # diverging warm pole: LightGBM regresses
INK = "#0b0b0b"
INK_2 = "#52514e"
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


def bar_labels(bars, ax, fmt="{:+.2f}", positive_good=True):
    for bar in bars:
        width = bar.get_width()
        value = width
        inside = abs(width) > 0.9
        label_color = "#ffffff" if inside else INK
        x = value - 0.05 if inside and width > 0 else value + 0.05
        y = bar.get_y() + bar.get_height() / 2
        ha = "right" if inside and width > 0 else "left"
        if inside and width < 0:
            ha, x = "left", value + 0.05
        ax.text(x, y, fmt.format(value), va="center", ha=ha, fontsize=8.5,
                color=label_color if inside else INK_2)


def category_delta(ax, table):
    data = sorted(table, key=lambda r: r["delta_pp"])
    names = [SHORT[r["series_id"]] for r in data]
    values = [r["delta_pp"] for r in data]
    colors = [BLUE if v > 0 else RED for v in values]
    bars = ax.barh(names, values, color=colors, height=0.62, zorder=3)
    ax.axvline(0, color="#c3c2b7", lw=1, zorder=2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("WAPE delta, naive − LightGBM (pp); positive = LightGBM better")
    bar_labels(bars, ax)
    ax.text(0.99, 0.03,
            "blue = LightGBM improves · red = LightGBM regresses",
            transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)


def heatmap(ax, table):
    categories = sorted({r["series_id"] for r in table}, key=lambda s: list(SHORT).index(s))
    horizons = sorted({int(r["horizon"]) for r in table})
    grid = np.full((len(categories), len(horizons)), np.nan)
    for r in table:
        grid[categories.index(r["series_id"]), horizons.index(int(r["horizon"]))] = r["delta_pp"]
    limit = float(np.nanmax(np.abs(grid)))
    im = ax.imshow(grid, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", [RED, "#f0efec", BLUE]), vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(horizons)), [f"h={h}" for h in horizons])
    ax.set_yticks(range(len(categories)), [SHORT[c] for c in categories])
    ax.grid(False)
    for i in range(len(categories)):
        for j in range(len(horizons)):
            value = grid[i, j]
            on_dark = abs(value) > 0.55 * limit
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if on_dark else INK_2)
    ax.set_title("WAPE delta (pp) by category × horizon", fontsize=10, color=INK_2, loc="left")
    return im


def two_panel(ax_left, ax_right, tables):
    regimes = tables["by_regime"]
    names = [r["regime"].replace("_", " ") for r in regimes]
    values = [r["delta_pp"] for r in regimes]
    bars = ax_left.bar(names, values, color=[BLUE if v > 0 else RED for v in values],
                       width=0.58, zorder=3)
    ax_left.axhline(0, color="#c3c2b7", lw=1)
    ax_left.grid(axis="x", visible=False)
    ax_left.set_title("by target regime", fontsize=10, color=INK_2, loc="left")
    for bar, value in zip(bars, values):
        ax_left.text(bar.get_x() + bar.get_width() / 2, value,
                     f"{value:+.2f}", ha="center",
                     va="bottom" if value > 0 else "top", fontsize=8.5, color=INK_2)
    ax_left.margins(y=0.18)

    years = tables["by_origin_year"]
    names = [str(int(r["origin_year"])) for r in years]
    values = [r["delta_pp"] for r in years]
    bars = ax_right.bar(names, values, color=[BLUE if v > 0 else RED for v in values],
                        width=0.58, zorder=3)
    ax_right.axhline(0, color="#c3c2b7", lw=1)
    ax_right.grid(axis="x", visible=False)
    ax_right.set_title("by origin year", fontsize=10, color=INK_2, loc="left")
    for bar, value in zip(bars, values):
        ax_right.text(bar.get_x() + bar.get_width() / 2, value,
                      f"{value:+.2f}", ha="center",
                      va="bottom" if value > 0 else "top", fontsize=8.5, color=INK_2)
    ax_right.margins(y=0.18)


def revision_scatter(ax, overlap):
    xs = [r["median_abs_revision_pct"] for r in overlap]
    ys = [r["lgbm_minus_naive_pp"] for r in overlap]
    ax.scatter(xs, ys, s=56, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    label_offsets = {
        "gasoline_stations": (7, -14, "left"),
        "general_merchandise": (5, 6, "left"),
        "retail_total": (-5, 8, "right"),
        "motor_vehicle_and_parts": (5, -2, "left"),
    }
    for r in overlap:
        name = SHORT[r["series_id"]]
        dx, dy, ha = label_offsets.get(r["series_id"], (5, 5, "left"))
        ax.annotate(name, (r["median_abs_revision_pct"], r["lgbm_minus_naive_pp"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    ha=ha, color=INK_2)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("median |preliminary → revised| revision (% of level)")
    ax.set_ylabel("WAPE delta (pp)\npositive = LightGBM better")
    spearman = json.load(open(RESULTS / "decomposition.json"))["revision_overlap_spearman"]
    ax.text(0.98, 0.04, f"Spearman ρ = {spearman:+.2f}", transform=ax.transAxes,
            ha="right", fontsize=9, color=INK_2)


def main() -> None:
    data = json.load(open(RESULTS / "decomposition.json"))
    tables = data["tables"]
    FIGURES.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    category_delta(ax, tables["by_series"])
    fig.tight_layout()
    fig.savefig(FIGURES / "category_delta.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    heatmap(ax, tables["series_x_horizon"])
    fig.tight_layout()
    fig.savefig(FIGURES / "heatmap_category_horizon.png", bbox_inches="tight")
    plt.close(fig)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    two_panel(ax_left, ax_right, tables)
    fig.tight_layout()
    fig.savefig(FIGURES / "regime_year.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    revision_scatter(ax, data["overlap_table"])
    fig.tight_layout()
    fig.savefig(FIGURES / "revision_vs_delta.png", bbox_inches="tight")
    plt.close(fig)

    print("figures written:", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
