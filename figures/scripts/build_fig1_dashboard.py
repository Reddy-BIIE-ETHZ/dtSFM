#!/usr/bin/env python3
"""build_fig1_dashboard.py — dtSFM v3 paper Figure 1.

Single-panel performance dashboard rendered as a table-figure: held-out
metrics for all four output heads (global retrieval, interface, contact,
affinity), in-distribution vs OOD, on the locked v3 production checkpoint
(epoch_010.pt, B-3 run b3_20260506_191149).

Source data: `eval_dtsfm_v3_quick.py` console output for split=in_dist
and split=val on the same checkpoint and held-out-validation TSV.

Eval setup is surfaced in the figure header so the reader sees the unique-
pool denominators at a glance.

Output: dtSFM-Figures/fig1_dashboard.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt        # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from calm.figures.biie_style import (  # noqa: E402
    apply_style, BIIE, save_figure,
)


# ============================================================================ #
# Numbers — directly from the two eval outputs (epoch_010, unique-pool)
# ============================================================================ #
# ID = split=in_dist : 3,955 unique drugs × 932 unique proteins (3,968 pairs)
# OOD = split=val   : 3,948 unique drugs × 311 unique proteins (3,968 pairs)

# Each row is (metric_label, id_value_str, ood_value_str)
# Section header rows are tuples like ("__SECTION__", section_label)

ROWS = [
    ("__SECTION__", "Global retrieval"),
    ("D → T    R@1",           "47 %",   "32 %"),
    ("D → T    R@10",          "88 %",   "65 %"),
    ("D → T    R@100",         "97 %",   "90 %"),
    ("D → T    median rank",    "1",      "3"),
    ("T → D    R@1",           "41 %",   "28 %"),
    ("T → D    R@10",          "78 %",   "55 %"),
    ("T → D    R@100",         "92 %",   "78 %"),
    ("T → D    median rank",    "1",      "6"),

    ("__SECTION__", "Interface head  (per-atom)"),
    ("AUROC",                  "0.96",   "0.90"),
    ("F1",                     "0.94",   "0.97"),
    ("Precision",              "0.99",   "0.99"),
    ("Recall",                 "0.90",   "0.95"),

    ("__SECTION__", "Contact head  (atom × residue)"),
    ("AUROC",                  "0.99",   "0.98"),
    ("IoU @ 0.5",              "0.12",   "0.13"),

    ("__SECTION__", "Affinity head  (pAff regression)"),
    ("Pearson r",              "0.64",   "0.48"),
    ("Spearman ρ",             "0.61",   "0.47"),
    ("RMSE  (log10 K_d)",      "1.04",   "1.22"),
]

# Layout (axes coordinates 0..100)
COL_METRIC = 4
COL_ID     = 65
COL_OOD    = 88
ROW_TOP    = 76      # leaves room for compact title + 2-line column header
ROW_DY_REG = 3.0     # tighter row spacing for sub-panel use
ROW_DY_SEC = 3.5
SECTION_GAP_AFTER = 1.2


def main():
    apply_style()

    # Compact size — ready to drop in as a sub-panel
    fig = plt.figure(figsize=(4.8, 4.6))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- Single-line title (compact) ----
    fig.text(0.5, 0.965,
             "dtSFM v3 — held-out performance, epoch_010.pt",
             ha="center", va="bottom",
             fontsize=10.5, weight="bold", color=BIIE.BLACK)
    fig.text(0.5, 0.935,
             "unique-pool retrieval; 3,968 held-out pairs sampled",
             ha="center", va="bottom", style="italic",
             fontsize=7.5, color=BIIE.GREY_DARK)

    # ---- Column headers ----
    y_hdr = ROW_TOP + 5.5
    ax.text(COL_METRIC, y_hdr, "Metric",
            ha="left", va="center", fontsize=9.0, weight="bold",
            color=BIIE.BLACK)
    ax.text(COL_ID, y_hdr, "in-distribution",
            ha="center", va="center", fontsize=9.0, weight="bold",
            color=BIIE.ID_FILL)
    ax.text(COL_OOD, y_hdr, "OOD",
            ha="center", va="center", fontsize=9.0, weight="bold",
            color=BIIE.BLACK)

    # Pool-size sub-label under each data column header
    ax.text(COL_ID, y_hdr - 2.7,
            "3,955 × 932",
            ha="center", va="center", fontsize=6.5, style="italic",
            color=BIIE.GREY_DARK)
    ax.text(COL_OOD, y_hdr - 2.7,
            "3,948 × 311",
            ha="center", va="center", fontsize=6.5, style="italic",
            color=BIIE.GREY_DARK)
    # Tiny clarifier (drugs × proteins) on the metric column line
    ax.text(COL_METRIC, y_hdr - 2.7,
            "(drugs × proteins)",
            ha="left", va="center", fontsize=6.5, style="italic",
            color=BIIE.GREY_DARK)

    # Header underline
    ax.plot([COL_METRIC - 1, 97], [y_hdr - 4.5, y_hdr - 4.5],
            color=BIIE.BLACK, linewidth=0.8, clip_on=False)

    # ---- Body rows ----
    y = ROW_TOP
    band_toggle = False
    for row in ROWS:
        if row[0] == "__SECTION__":
            # Section header
            y -= SECTION_GAP_AFTER
            ax.text(COL_METRIC - 1.0, y, row[1],
                    ha="left", va="center",
                    fontsize=8.5, weight="bold", color=BIIE.BLACK)
            # Thin line under section header
            ax.plot([COL_METRIC - 1, 97], [y - 1.4, y - 1.4],
                    color=BIIE.GREY_MID, linewidth=0.5, clip_on=False)
            y -= ROW_DY_SEC
            band_toggle = False
            continue

        metric, id_val, ood_val = row

        # Alternating light-grey row band
        if band_toggle:
            ax.add_patch(Rectangle(
                (COL_METRIC - 2, y - 1.3), 102, ROW_DY_REG,
                facecolor=BIIE.GREY_LIGHT, alpha=0.35,
                edgecolor="none", clip_on=False, zorder=-1,
            ))

        ax.text(COL_METRIC, y, metric,
                ha="left", va="center", fontsize=7.5, color=BIIE.BLACK)
        ax.text(COL_ID, y, id_val,
                ha="center", va="center", fontsize=8.0, weight="bold",
                color=BIIE.ID_FILL,
                family="DejaVu Sans Mono")
        ax.text(COL_OOD, y, ood_val,
                ha="center", va="center", fontsize=8.0,
                color=BIIE.BLACK,
                family="DejaVu Sans Mono")

        y -= ROW_DY_REG
        band_toggle = not band_toggle

    paths = save_figure(fig, "fig1_dashboard",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
