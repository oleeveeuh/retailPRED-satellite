#!/usr/bin/env python3
"""Method schematic: what vintage-safe backtesting did, and where the
classifier sits. Illustrative diagram — origin grid and horizon are the
real protocol values; no metrics are drawn. Fieldnotes theme."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))

SURFACE = "#fbf9f4"
PAPER = "#f5f1e8"
INK = "#25251f"
INK2 = "#52514e"
MUTED = "#77776c"
GRID = "#d8d1c4"
SAGE = "#89927a"
SAGE_LIGHT = "#dfe1d5"
WINE = "#482728"
ROSE = "#d4708c"
SUMI = "#1E342E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c9c2b2",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
})

# month index helper: months since 0 AD (year*12 + month)
def m(year: int, month: int = 1) -> float:
    return year * 12 + month

EXPL_ORIGINS = [
    ("2017-10", 2017, 10), ("2017-11", 2017, 11), ("2017-12", 2017, 12),
    ("2018-01", 2018, 1), ("2018-04", 2018, 4), ("2018-07", 2018, 7),
    ("2018-08", 2018, 8), ("2018-10", 2018, 10), ("2019-03", 2019, 3),
    ("2019-04", 2019, 4), ("2019-07", 2019, 7), ("2019-10", 2019, 10),
    ("2019-12", 2019, 12), ("2020-03", 2020, 3), ("2020-06", 2020, 6),
    ("2021-01", 2021, 1), ("2021-03", 2021, 3), ("2021-08", 2021, 8),
    ("2021-10", 2021, 10), ("2021-11", 2021, 11), ("2022-01", 2022, 1),
    ("2022-05", 2022, 5), ("2022-08", 2022, 8), ("2022-10", 2022, 10),
]
HOLDOUT_ORIGINS = [
    ("2023-01", 2023, 1), ("2023-07", 2023, 7), ("2023-10", 2023, 10),
    ("2024-01", 2024, 1), ("2024-03", 2024, 3), ("2024-12", 2024, 12),
    ("2025-01", 2025, 1), ("2025-03", 2025, 3),
]
X0 = m(1992, 1)  # training history start (the panel's left edge)
XLIM = (m(1991, 6), m(2025, 9))


def backtest_band(ax) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    n_expl = len(EXPL_ORIGINS)
    n_hold = len(HOLDOUT_ORIGINS)
    rows = n_expl + n_hold
    ax.set_ylim(rows + 0.6, -3.7)  # fold 0 on top; headroom for the legend
    ax.set_xlim(*XLIM)

    for i, (label, year, month) in enumerate(EXPL_ORIGINS):
        origin = m(year, month)
        ax.barh(i, origin - X0, left=X0, height=0.6, color=SAGE_LIGHT,
                edgecolor=SAGE, linewidth=0.9, zorder=3)
        ax.barh(i, 12, left=origin, height=0.34, color=ROSE, alpha=0.55, zorder=3)
        ax.plot([origin, origin], [i - 0.3, i + 0.3], color=WINE, lw=1.6, zorder=4)
    hold_top = n_expl
    for j, (label, year, month) in enumerate(HOLDOUT_ORIGINS):
        i = hold_top + j
        origin = m(year, month)
        ax.barh(i, origin - X0, left=X0, height=0.6, facecolor="none",
                edgecolor=MUTED, linewidth=0.9, linestyle=(0, (3, 2)), zorder=3)
        ax.barh(i, 12, left=origin, height=0.34, facecolor="none",
                edgecolor=ROSE, linewidth=0.9, linestyle=(0, (3, 2)), zorder=3)
        ax.plot([origin, origin], [i - 0.3, i + 0.3], color=MUTED, lw=1.4, zorder=4)

    ax.axhline(n_expl - 0.5, color=GRID, lw=1.2, zorder=1)
    ax.set_yticks([0, n_expl - 1, rows - 1])
    ax.set_yticklabels(["origin 2017-10", "origin 2022-10", "origin 2025-03"],
                       fontsize=8, color=INK2)
    ax.set_xticks([m(y) for y in (1992, 1997, 2002, 2007, 2012, 2017, 2022)])
    ax.set_xticklabels(["1992", "1997", "2002", "2007", "2012", "2017", "2022"],
                       fontsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.grid(axis="y", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    handles = [
        Patch(facecolor=SAGE_LIGHT, edgecolor=SAGE, linewidth=0.9,
              label="training: every vintage published ≤ origin (+45d availability)"),
        Patch(facecolor=ROSE, alpha=0.55, label="forecast fan: h = 1 / 3 / 6 / 12"),
        Line2D([0], [0], color=WINE, lw=1.6, label="origin"),
        Line2D([0], [0], color=MUTED, lw=1, linestyle=(0, (3, 2)),
               label="holdout: scored exactly once, never re-opened"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02),
              fontsize=8, frameon=False, ncols=2, handlelength=1.6,
              columnspacing=1.2)
    ax.set_title("vintage-safe backtesting — one refit per origin, only the visible past",
                 fontsize=10.5, color=INK2, loc="left", pad=56)


def classifier_band(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def chip(x, y, w, h, text, fc, ec, tc, fs=9, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                    facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=4, fontweight=weight)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

    chip(0.02, 0.58, 0.17, 0.3, "seasonal naive", SUMI, SUMI, SURFACE)
    chip(0.02, 0.12, 0.17, 0.3, "theta method", WINE, WINE, SURFACE)
    chip(0.38, 0.35, 0.26, 0.3,
         "policy classifier\nP(theta wins) ≥ τ = 0.50", SAGE, SAGE, SURFACE, weight="bold")
    chip(0.76, 0.35, 0.22, 0.3, "one pick per\ncategory × origin", "#ffffff", ROSE, INK)
    arrow(0.19, 0.73, 0.38, 0.56)
    arrow(0.19, 0.27, 0.38, 0.44)
    arrow(0.64, 0.5, 0.76, 0.5)
    ax.text(0.285, 0.14, "features: trend · volatility · revision churn ·\nlagged performance — all point-in-time",
            fontsize=7.5, color=MUTED, ha="left")
    ax.text(0.5, 0.9, "the learned switch — menu fixed to {naive, theta}",
            fontsize=9, color=INK2, ha="center")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "figures" / "method_schematic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.4, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=(3.0, 1.25), hspace=0.34,
                          left=0.11, right=0.975, top=0.88, bottom=0.09)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    backtest_band(ax1)
    classifier_band(ax2)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
