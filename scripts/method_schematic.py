#!/usr/bin/env python3
"""Method schematic: what vintage-safe backtesting did, and where the
classifier sits. Card-first layout — wide aspect and large type so it
stays legible at gallery-thumbnail size and in the dialog. Illustrative
diagram: origin grid and protocol values are real; no metrics are drawn.
Fieldnotes theme."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

SURFACE = "#e4e4d7"  # the gallery card-art light green
INK = "#25251f"
INK2 = "#52514e"
MUTED = "#6e6d66"
SAGE = "#89927a"
SAGE_LIGHT = "#dfe1d5"
WINE = "#482728"
ROSE = "#d4708c"
SUMI = "#1E342E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
})


def m(year: int, month: int = 1) -> float:
    return year * 12 + month


EXPL_ORIGINS = [
    (2017, 10), (2017, 11), (2017, 12), (2018, 1), (2018, 4), (2018, 7),
    (2018, 8), (2018, 10), (2019, 3), (2019, 4), (2019, 7), (2019, 10),
    (2019, 12), (2020, 3), (2020, 6), (2021, 1), (2021, 3), (2021, 8),
    (2021, 10), (2021, 11), (2022, 1), (2022, 5), (2022, 8), (2022, 10),
]
HOLDOUT_ORIGINS = [
    (2023, 1), (2023, 7), (2023, 10), (2024, 1),
    (2024, 3), (2024, 12), (2025, 1), (2025, 3),
]
X0 = m(1992, 1)
XLIM = (m(1991, 3), m(2025, 11))


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "figures" / "method_schematic.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.0, 4.4))
    gs = fig.add_gridspec(2, 1, height_ratios=(2.55, 1.3), hspace=0.55,
                          left=0.085, right=0.985, top=0.80, bottom=0.10)

    # ---- band 1: the vintage-safe staircase -------------------------------
    ax = fig.add_subplot(gs[0])
    n_expl = len(EXPL_ORIGINS)
    n_hold = len(HOLDOUT_ORIGINS)
    rows = n_expl + n_hold
    ax.set_ylim(rows + 0.9, -2.6)
    ax.set_xlim(*XLIM)
    ax.axis("off")

    for i, (year, month) in enumerate(EXPL_ORIGINS):
        origin = m(year, month)
        ax.barh(i, origin - X0, left=X0, height=0.66, color=SAGE_LIGHT,
                edgecolor=SAGE, linewidth=1.1, zorder=3)
        ax.barh(i, 12, left=origin, height=0.4, color=ROSE, alpha=0.6, zorder=3)
        ax.plot([origin, origin], [i - 0.33, i + 0.33], color=WINE, lw=2.0, zorder=4)
    for j, (year, month) in enumerate(HOLDOUT_ORIGINS):
        i = n_expl + j
        origin = m(year, month)
        ax.barh(i, origin - X0, left=X0, height=0.66, facecolor="none",
                edgecolor=MUTED, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
        ax.barh(i, 12, left=origin, height=0.4, facecolor="none",
                edgecolor=ROSE, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
        ax.plot([origin, origin], [i - 0.33, i + 0.33], color=MUTED, lw=1.6, zorder=4)
    ax.axhline(n_expl - 0.5, color=MUTED, lw=1.0, linestyle=(0, (1, 2)), zorder=1)

    ax.set_yticks([0, n_expl - 1, rows - 1])
    ax.set_yticklabels(["origin 2017-10", "origin 2022-10", "origin 2025-03"],
                       fontsize=13, color=INK2)
    ax.set_xticks([m(y) for y in (1992, 1997, 2002, 2007, 2012, 2017, 2022)])
    ax.set_xticklabels(["1992", "1997", "2002", "2007", "2012", "2017", "2022"],
                       fontsize=13)
    ax.tick_params(length=0)

    handles = [
        Patch(facecolor=SAGE_LIGHT, edgecolor=SAGE, linewidth=1.1,
              label="training: vintages published ≤ origin (+45d)"),
        Patch(facecolor=ROSE, alpha=0.6, label="forecast fan: h = 1/3/6/12"),
        Line2D([0], [0], color=WINE, lw=2.0, label="origin"),
        Line2D([0], [0], color=MUTED, lw=1.2, linestyle=(0, (3, 2)),
               label="holdout: scored once"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
              fontsize=12, frameon=False, ncols=2, handlelength=1.5,
              columnspacing=1.0, handletextpad=0.5)
    ax.set_title("vintage-safe backtesting — one refit per origin, only the visible past",
                 fontsize=16, color=INK2, loc="left", pad=64)

    # ---- band 2: the learned switch ---------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    def chip(x, y, w, h, text, fc, ec, tc="#fbf9f4", fs=12, weight="bold"):
        ax2.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                     facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3))
        ax2.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fs, color=tc, zorder=4, fontweight=weight)

    def arrow(x0, y0, x1, y1):
        ax2.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", color=INK2, lw=1.8))

    ax2.text(0.5, 0.94, "the learned switch — menu fixed to {naive, theta}",
             fontsize=12.5, color=INK2, ha="center")
    chip(0.02, 0.30, 0.11, 0.34, "naive", SUMI, SUMI, fs=12.5)
    chip(0.16, 0.30, 0.11, 0.34, "theta", WINE, WINE, fs=12.5)
    chip(0.37, 0.20, 0.31, 0.52, "policy classifier\nP(theta wins) ≥ τ = 0.50",
         SAGE, SAGE, "#fbf9f4", fs=11.5)
    chip(0.73, 0.26, 0.25, 0.42, "one pick per\ncategory × origin",
         "#fbf9f4", ROSE, INK, fs=11.5)
    arrow(0.13, 0.40, 0.37, 0.40)
    arrow(0.27, 0.55, 0.37, 0.55)
    arrow(0.68, 0.47, 0.73, 0.47)
    ax2.text(0.5, -0.02, "features: trend · volatility · revision churn · lagged performance — all point-in-time",
             fontsize=10.5, color=MUTED, ha="center")

    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
