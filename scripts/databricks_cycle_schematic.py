#!/usr/bin/env python3
"""Databricks scheduled-cycle schematic: what the four-task live loop does
each cycle. Card-first layout — wide aspect and large type so it stays
legible at gallery-thumbnail size and in the dialog. Illustrative diagram:
task graph, gates, and cadence are real; no metrics are drawn. Fieldnotes
theme. Companion to method_schematic.py; save keeps the portfolio card's
2:1 art-frame aspect so object-fit: cover crops nothing legible."""

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

SURFACE = "#e4e4d7"  # the gallery card-art light green
INK = "#25251f"
INK2 = "#52514e"
MUTED = "#6e6d66"
SAGE = "#89927a"
SAGE_LIGHT = "#dfe1d5"
WINE = "#482728"
ROSE = "#d4708c"

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
})


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "figures" /
        "databricks_cycle_schematic.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.42, 3.30))
    ax = fig.add_axes((0.02, 0.16, 0.96, 0.72))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # heading: spaced-caps title (manual tracking — no letter-spacing in
    # matplotlib) + the declared cadence in the mono voice
    fig.text(0.055, 0.955, " ".join("THE SCHEDULED CYCLE"),
             fontsize=14, color=INK2, ha="left", va="top")
    fig.text(0.055, 0.885,
             "cron 0 30 14 * * ? · daily 14:30 utc · declared paused",
             fontsize=8, color=MUTED, ha="left", va="top",
             fontfamily="IBM Plex Mono")

    def chip(x, y, w, h, text, fc, ec, tc, fs=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                    facecolor=fc, edgecolor=ec, linewidth=1.4,
                                    zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=4, linespacing=1.25)

    chips = [
        (0.015, "pull census\nrelease", SAGE_LIGHT, SAGE, INK),
        (0.255, "score promoted\nmodel", SAGE, SAGE, "#fbf9f4"),
        (0.495, "evaluate\nrealized errors", SAGE_LIGHT, SAGE, INK),
        (0.735, "publish\nsnapshot", "#fbf9f4", ROSE, INK),
    ]
    W, H, Y = 0.20, 0.30, 0.55
    for x, label, fc, ec, tc in chips:
        chip(x, Y, W, H, label, fc, ec, tc)
    artifacts = [
        "→ silver (immutable merge)",
        "→ gold.forecasts",
        "→ gold.realized_errors",
        "→ live-data branch",
    ]
    for (x, *_), note in zip(chips, artifacts):
        ax.text(x + W / 2, Y - 0.055, note, ha="center", va="top",
                fontsize=7.2, color=MUTED, fontfamily="IBM Plex Mono")

    for x0 in (0.215, 0.455, 0.695):
        ax.annotate("", xy=(x0 + 0.04, Y + H / 2), xytext=(x0, Y + H / 2),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.8))

    # the fail-closed branch off the scoring task — the verified state
    ax.annotate("", xy=(0.355, 0.335), xytext=(0.355, Y),
                arrowprops=dict(arrowstyle="->", color=WINE, lw=1.6))
    ax.add_patch(FancyBboxPatch((0.155, 0.085), 0.40, 0.25,
                                boxstyle="round,pad=0.008",
                                facecolor="none", edgecolor=WINE,
                                linewidth=1.3, linestyle=(0, (4, 3)), zorder=3))
    ax.text(0.355, 0.21, "gate: frozen decision → refuse",
            ha="center", va="center", fontsize=9.5, color=WINE, zorder=4)
    ax.text(0.355, 0.13, "serving stays empty — nothing unearned is served",
            ha="center", va="center", fontsize=7.6, color=WINE, zorder=4)

    # the loop that matures predictions into evidence for the next gate
    ax.annotate("", xy=(0.555, 0.21), xytext=(0.62, Y - 0.02),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.3,
                                linestyle=(0, (4, 3)),
                                connectionstyle="arc3,rad=0.25"))
    ax.text(0.645, 0.24, "matured actuals\nfeed the next gate",
            ha="left", va="center", fontsize=7.6, color=INK2,
            linespacing=1.3)
    ax.text(0.835, 0.24, "identical snapshots\nare skipped",
            ha="left", va="center", fontsize=7.6, color=MUTED,
            fontfamily="IBM Plex Mono", linespacing=1.3)

    fig.text(0.5, 0.075,
             "one refit per cycle at scoring time — the model is never persisted",
             fontsize=8.5, color=MUTED, ha="center", va="top",
             fontfamily="IBM Plex Mono")
    fig.text(0.5, 0.032,
             "offline cadence: airflow weekly validation · mondays 02:00 utc",
             fontsize=8.5, color=MUTED, ha="center", va="top",
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
