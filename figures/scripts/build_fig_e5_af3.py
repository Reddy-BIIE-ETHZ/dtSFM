#!/usr/bin/env python3
"""build_fig_e5_af3.py — dtSFM v3 paper Figure e5.

Single-panel complementarity scatter showing dtSFM v3 + AlphaFold-3 jointly
covering the (within-class retrieval × across-scaffold structural verification)
plane on the 323-pair §F.4.2 cohort.

  x = v3 cosine percentile in the 522,776-drug library  (0 = worst, 100 = best)
  y = AF3 iPTM                                          (binding-interface quality)
  color = §F.4 leakage class A/B/C (consistent with Fig e4)
  grey  = negative controls (no expected class)

Headline message: top-right quadrant = both methods agree (anchor wins);
top-left quadrant = v3 missed but AF3 caught (complementarity wins —
Dapansutrile, MSA-2, diABZI 3); negatives cluster at the AF3 floor (~0.7).

Source data: audit/dtsfm/repurposing/F42_results.tsv  (323-pair cohort, LOCKED 2026-05-09)
Output:      dtSFM-Figures/fig_e5_af3_complementarity.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402
from calm.figures.biie_style import (  # noqa: E402
    apply_style, BIIE, save_figure,
)


LIBRARY_SIZE = 522_776

CLASS_COLOR = {
    "A": BIIE.BLUE_PURPLE,
    "B": BIIE.TEAL,
    "C": BIIE.MAGENTA,
}
NEG_COLOR = BIIE.GREY_DARK

# Annotations split by panel — Panel A (full range) gets the cross-scaffold
# "v3 misses · AF3 catches" cases; Panel B (zoom 99-100%) gets the top-percentile
# anchors that cluster against the right edge in Panel A.

# (target, drug_name) → label offset (dx_pct, dy_iptm)
PANEL_A_ANNOTATIONS = {
    ("NLRP3",  "Dapansutrile"):      (0,    +0.05),
    ("STING1", "MSA-2"):             (0,    +0.06),
    ("STING1", "diABZI compound 3"): (0,    -0.06),
    # ADU-S100 (STING1 Phase 2, percentile 99.825) — outside panel-b zoom,
    # so labeled here in the full-range panel.
    ("STING1", "ADU-S100"):          (-15,  +0.05),
}
PANEL_B_ANNOTATIONS = {
    ("CD73",   "AB680"):     (-0.012, -0.05),
    ("NLRP3",  "MCC950"):    (-0.012, +0.06),
}

# Compounds that are "called out" anywhere in the figure get filled circles
# in both panels; all other dots are open circles of the correct class color.
CALLOUT_COMPOUNDS = set(PANEL_A_ANNOTATIONS.keys()) | set(PANEL_B_ANNOTATIONS.keys())


def main():
    apply_style()

    df = pd.read_csv(ROOT / "audit/dtsfm/repurposing/F42_results.tsv", sep="\t")

    # Compute v3 cosine percentile from rank or synthetic_rank
    def to_pct(row):
        r = row.get("rank")
        if pd.notna(r):
            return 100.0 * (1.0 - r / LIBRARY_SIZE)
        s = row.get("synthetic_rank")
        if pd.notna(s):
            return 100.0 * (1.0 - s / LIBRARY_SIZE)
        return np.nan

    df["v3_pct"] = df.apply(to_pct, axis=1)
    df = df.dropna(subset=["v3_pct", "iptm"]).copy()

    # Tag negative controls
    df["is_negative"] = df["stratum"] == "negative_control"
    df["color_class"] = df["expected_class"].fillna("")
    df["is_callout"] = df.apply(
        lambda r: (r["target"], r["drug_name"]) in CALLOUT_COMPOUNDS, axis=1
    )

    # ---- Figure: 2 panels (full range + zoomed top) side by side ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1]})

    DOT_SIZE = 30   # uniform dot size for all compounds

    def _plot_data(ax, annotations, show_quadrant_label=False):
        """Render the scatter on ax. All dots same size; called-out compounds
        get filled circles, all other dots are open circles in their class color."""
        # Reference lines (always)
        ax.axhline(0.7, ls="--", lw=0.6, color=BIIE.GREY_DARK, zorder=1)
        ax.axvline(99,  ls=":",  lw=0.6, color=BIIE.GREY_DARK, zorder=1)

        non_neg = df[~df["is_negative"]]
        # OPEN (non-callout) — class-colored edge, white fill
        open_pts = non_neg[~non_neg["is_callout"]]
        for cls, color in CLASS_COLOR.items():
            m = open_pts[open_pts["color_class"] == cls]
            if len(m):
                ax.scatter(m["v3_pct"], m["iptm"], s=DOT_SIZE,
                           facecolors="white", edgecolors=color,
                           linewidths=0.9, alpha=0.95, zorder=2)
        # screening dots without an expected class → grey-edge open
        m = open_pts[open_pts["color_class"] == ""]
        if len(m):
            ax.scatter(m["v3_pct"], m["iptm"], s=DOT_SIZE,
                       facecolors="white", edgecolors=BIIE.GREY_MID,
                       linewidths=0.9, alpha=0.85, zorder=2)

        # FILLED (callouts) — class-colored fill, black edge
        filled_pts = non_neg[non_neg["is_callout"]]
        for cls, color in CLASS_COLOR.items():
            m = filled_pts[filled_pts["color_class"] == cls]
            if len(m):
                ax.scatter(m["v3_pct"], m["iptm"], s=DOT_SIZE,
                           color=color, edgecolors=BIIE.BLACK,
                           linewidths=0.7, alpha=1.0, zorder=4)

        # Negative controls — same size, distinct grey-edge open with x marker
        neg = df[df["is_negative"]]
        if len(neg):
            ax.scatter(neg["v3_pct"], neg["iptm"], s=DOT_SIZE,
                       facecolors="white", edgecolors=NEG_COLOR,
                       linewidths=0.9, marker="o", zorder=3)

        # Annotate compounds belonging to this panel
        for (tgt, drug), (dx, dy) in annotations.items():
            row = df[(df["target"] == tgt) & (df["drug_name"] == drug)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            x = float(r["v3_pct"])
            y = float(r["iptm"])
            label = drug.replace("compound 3", "3").strip()
            ax.annotate(label,
                        xy=(x, y),
                        xytext=(x + dx, y + dy),
                        ha="center", va="center",
                        fontsize=7.0, weight="bold", color=BIIE.BLACK,
                        arrowprops=dict(arrowstyle="-", color=BIIE.GREY_DARK,
                                        lw=0.5, shrinkA=2, shrinkB=4))

        if show_quadrant_label:
            ax.text(15, 0.93,
                    "v3 misses · AF3 catches",
                    fontsize=7.5, weight="bold", style="italic",
                    color=BIIE.MAGENTA, ha="left", va="center", alpha=0.9)

    # ====== Panel A — full range (0–100% percentile, all strata) ======
    _plot_data(ax1, annotations=PANEL_A_ANNOTATIONS,
               show_quadrant_label=True)
    ax1.set_xlim(0, 102)
    ax1.set_ylim(0.35, 1.0)
    ax1.set_xlabel("v3 cosine percentile in 522,776-drug library",
                   fontweight="bold", labelpad=6)
    ax1.set_ylabel("AF3 iPTM (binding-interface quality)",
                   fontweight="bold", labelpad=6)
    ax1.set_title("a   full range", loc="left",
                  fontweight="bold", fontsize=9.0)
    ax1.text(1.5, 0.705, "iPTM = 0.7  (binder threshold)", fontsize=6.0,
             color=BIIE.GREY_DARK, style="italic", va="bottom", ha="left")
    ax1.text(98.5, 0.37, "top 1 %", fontsize=6.0,
             color=BIIE.GREY_DARK, style="italic",
             va="bottom", ha="right", rotation=90)
    # Indicator: x-axis zoom region for panel b (top 0.1%)
    ax1.axvspan(99.9, 100, ymin=0.0, ymax=1.0,
                facecolor=BIIE.MAGENTA, alpha=0.10, zorder=0)
    ax1.text(99.85, 0.95, "→ b", fontsize=6.5,
             color=BIIE.MAGENTA, style="italic", weight="bold",
             va="top", ha="right", alpha=0.85)

    # ====== Panel B — zoom on x: 99.9–100% (top 0.1% only) ======
    _plot_data(ax2, annotations=PANEL_B_ANNOTATIONS,
               show_quadrant_label=False)
    ax2.set_xlim(99.9, 100.005)
    ax2.set_ylim(0.35, 1.0)
    ax2.set_xticks([99.9, 99.95, 100.0])     # 3 hash marks only
    ax2.set_xlabel("v3 cosine percentile  (zoom: top 0.1 %)",
                   fontweight="bold", labelpad=6)
    ax2.set_ylabel("AF3 iPTM",
                   fontweight="bold", labelpad=6)
    ax2.set_title("b   zoom: top 0.1 % library percentile",
                  loc="left", fontweight="bold", fontsize=9.0)
    ax2.text(99.901, 0.705, "iPTM = 0.7", fontsize=6.0,
             color=BIIE.GREY_DARK, style="italic", va="bottom", ha="left")

    # ---- Title ----
    fig.text(0.5, 0.965,
             "Figure e5 | v3 + AF3 complementarity on 323-pair cohort",
             ha="center", va="bottom",
             fontsize=10.5, weight="bold", color=BIIE.BLACK)

    # ---- Class legend (top row, between title and panels) ----
    legend_y = 0.925
    fig.text(0.05, legend_y, "A",
             color=CLASS_COLOR["A"], fontsize=8.0, weight="bold",
             ha="left", va="bottom")
    fig.text(0.075, legend_y, "pair-in-train",
             color=BIIE.GREY_DARK, fontsize=7.0,
             ha="left", va="bottom")
    fig.text(0.21, legend_y, "B",
             color=CLASS_COLOR["B"], fontsize=8.0, weight="bold",
             ha="left", va="bottom")
    fig.text(0.235, legend_y, "target-trained, drug-OOD",
             color=BIIE.GREY_DARK, fontsize=7.0,
             ha="left", va="bottom")
    fig.text(0.46, legend_y, "C",
             color=CLASS_COLOR["C"], fontsize=8.0, weight="bold",
             ha="left", va="bottom")
    fig.text(0.485, legend_y, "drug + target both OOD",
             color=BIIE.GREY_DARK, fontsize=7.0,
             ha="left", va="bottom")
    fig.text(0.71, legend_y, "○",
             color=NEG_COLOR, fontsize=8.0, weight="bold",
             ha="left", va="bottom")
    fig.text(0.73, legend_y, "negative control",
             color=BIIE.GREY_DARK, fontsize=7.0,
             ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    paths = save_figure(fig, "fig_e5_af3_complementarity",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
