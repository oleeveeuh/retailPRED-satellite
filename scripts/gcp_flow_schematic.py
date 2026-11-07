#!/usr/bin/env python3
"""v1 GCP scheduled-flow schematic: what the undergraduate Airflow pipeline
did each week, and where its epistemic failure sites sat on the flow.
Card-first layout in the fieldnotes theme; companion to method_schematic.py
and databricks_cycle_schematic.py. Illustrative: the task graph, cadence,
and failure sites are real; no metrics are drawn. Canvas keeps the portfolio
card's 2:1 art-frame aspect so object-fit: cover crops nothing legible."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# The site's loaded fonts (index.html): Source Sans 3 is the body voice,
# IBM Plex Mono the label/caption voice. Both are installed locally, so
# matplotlib resolves them instead of silently falling back to DejaVu Sans.
plt.rcParams["font.sans-serif"] = ["Source Sans 3", "Helvetica Neue", "Arial",
                                   "DejaVu Sans"]
plt.rcParams["font.monospace"] = ["IBM Plex Mono", "Courier New",
                                  "DejaVu Sans Mono"]

SURFACE = "#e4e4d7"
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
    "text.color": INK,
})


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "figures" / "gcp_flow_schematic.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.42, 3.30))
    ax = fig.add_axes((0.02, 0.17, 0.96, 0.67))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.055, 0.955, " ".join("THE V1 SCHEDULED FLOW"),
             fontsize=14, color=INK2, ha="left", va="top")
    fig.text(0.055, 0.885,
             "airflow on gcp · cron 0 2 * * 1 · mondays 02:00 utc",
             fontsize=8, color=MUTED, ha="left", va="top",
             fontfamily="IBM Plex Mono")

    nodes = [
        ("airflow\nmon 02:00", SUMI, SUMI, "#fbf9f4"),
        ("fetch census\nactuals", SAGE_LIGHT, SAGE, INK),
        ("update rolling\npredictions", SAGE_LIGHT, SAGE, INK),
        ("metrics +\nanomalies", SAGE_LIGHT, SAGE, INK),
        ("export +\npush github", SAGE_LIGHT, SAGE, INK),
        ("vercel\ndashboard", "#fbf9f4", ROSE, INK),
    ]
    W, H, Y = 0.145, 0.26, 0.60
    xs = [0.005 + i * 0.166 for i in range(len(nodes))]
    for (label, fc, ec, tc), x in zip(nodes, xs):
        ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle="round,pad=0.006",
                                    facecolor=fc, edgecolor=ec, linewidth=1.3,
                                    zorder=3))
        ax.text(x + W / 2, Y + H / 2, label, ha="center", va="center",
                fontsize=8.6, color=tc, zorder=4, linespacing=1.2)
    for x0 in xs[:-1]:
        ax.annotate("", xy=(x0 + 0.166 - 0.014, Y + H / 2),
                    xytext=(x0 + W + 0.004, Y + H / 2),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5))

    # the three epistemic failure sites, staggered so their notes never
    # collide, each tied to its node with a thin wine tick
    failures = [
        (xs[2] + W / 2, 0.365,
         "pickles: inclusive features\nleak the target month"),
        (xs[3] + W / 2, 0.505,
         "429 rows = monthly × weekly\npseudo-replication"),
        (xs[5] + W / 2 - 0.02, 0.365,
         "headline accuracy:\nno model produced it"),
    ]
    for cx, top, text in failures:
        ax.plot([cx, cx], [Y - 0.015, min(top + 0.075, Y - 0.03)],
                color=WINE, lw=0.9, linestyle=(0, (2, 2)), zorder=2)
        ax.text(cx, top, text, ha="center", va="top", fontsize=7.2,
                color=WINE, fontfamily="IBM Plex Mono", linespacing=1.35)

    # the always-on serving strip, separate from the weekly batch
    ax.add_patch(FancyBboxPatch((0.005, 0.075), 0.46, 0.13,
                                boxstyle="round,pad=0.006",
                                facecolor=SAGE_LIGHT, edgecolor=SAGE,
                                linewidth=1.2, zorder=3))
    ax.text(0.235, 0.14, "always-on serving — fastapi · sqlite · per-category pickles",
            ha="center", va="center", fontsize=7.8, color=INK2, zorder=4)
    ax.text(0.52, 0.14,
            "orchestration worked: jobs ran, data flowed, the site rendered",
            ha="left", va="center", fontsize=8.2, color=INK2, zorder=4,
            style="italic")

    fig.text(0.5, 0.075,
             "the stack: python · airflow on gcp · lightgbm · xgboost · scikit-learn · shap · fastapi · sqlite · vercel",
             fontsize=8, color=MUTED, ha="center", va="top",
             fontfamily="IBM Plex Mono")
    fig.text(0.5, 0.032,
             "failures were epistemic, not operational — nothing in the flow could fail",
             fontsize=8, color=WINE, ha="center", va="top",
             fontfamily="IBM Plex Mono")

    # pad the tight render to the card's exact aspect (320.7 x 160) so the
    # gallery's object-fit: cover crop is a no-op; wings match the surface.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.2)
    buf.seek(0)
    img = plt.imread(buf)
    h, w = img.shape[:2]
    target = 320.7 / 160
    wing = matplotlib.colors.to_rgba(SURFACE)
    if w < h * target:  # too tall: widen symmetrically
        pad = int(round(h * target)) - w
        wings = np.full((h, 1, 4), wing)
        img = np.concatenate([wings.repeat(pad // 2, 1), img,
                              wings.repeat(pad - pad // 2, 1)], axis=1)
    elif h < w / target:  # too wide: pad top and bottom
        pad = int(round(w / target)) - h
        wings = np.full((1, w, 4), wing)
        img = np.concatenate([wings.repeat(pad // 2, 0), img,
                              wings.repeat(pad - pad // 2, 0)], axis=0)
    plt.imsave(out, img)
    print(f"written: {out} ({img.shape[1]}x{img.shape[0]}, "
          f"aspect {img.shape[1] / img.shape[0]:.3f})")


if __name__ == "__main__":
    main()
