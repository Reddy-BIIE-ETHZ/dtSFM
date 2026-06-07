#!/usr/bin/env python3
"""build_fig2_retrieval.py — dtSFM v3 paper Figure 2.

Two-panel bar chart of bidirectional pool-512 retrieval performance:
  (a) Drug → Target retrieval (safety screening direction)
  (b) Target → Drug retrieval (repurposing / library prioritization direction)

Filled bars = in-distribution; open bars (white fill, black edge) = OOD
(held-out clusters). Numerical labels above bars. No legend (BIIE style).
Direction printed below the x-axis as the xlabel; panel letters "a" / "b"
appear top-left as the title.

Source data:
  • R@10 / R@100 — manuscript §3.1 / §3.2 tables
  • R@1 — `audit/dtsfm/dtsfm-DECODER_HANDOFF.md` §3.1 (D→T OOD, T→D OOD/ID)
  • D→T R@1 in-distribution = preliminary placeholder (65 %, marked with *).
    Exact value pending pool-512 eval CSV lookup; flagged in caption.

Output: dtSFM-Figures/fig2_retrieval_bidirectional.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]   # data/dtsfm/scripts/figures/<file> → CALM-0.1.0
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
from calm.figures.biie_style import (  # noqa: E402
    apply_style, BIIE, save_figure,
    SIZE_2PANEL,
)


# ---- Data ----
METRICS = ["R@1", "R@10", "R@100"]

# Panel (a) Drug → Target retrieval (pool 310 unique proteins)
DRUG_TO_TARGET_ID  = [65, 95, 99]      # R@1 = 65 is PLACEHOLDER (asterisk)
DRUG_TO_TARGET_OOD = [32, 65, 89]
DRUG_TO_TARGET_PLACEHOLDER_MASK_ID  = [True, False, False]
DRUG_TO_TARGET_PLACEHOLDER_MASK_OOD = [False, False, False]
DRUG_TO_TARGET_MEDIAN_RANK_ID  = 1
DRUG_TO_TARGET_MEDIAN_RANK_OOD = 3

# Panel (b) Target → Drug retrieval (pool 3,948 unique drugs)
TARGET_TO_DRUG_ID  = [54, 89, 96]
TARGET_TO_DRUG_OOD = [22, 55, 78]
TARGET_TO_DRUG_PLACEHOLDER_MASK_ID  = [False, False, False]
TARGET_TO_DRUG_PLACEHOLDER_MASK_OOD = [False, False, False]
TARGET_TO_DRUG_MEDIAN_RANK_ID  = 1
TARGET_TO_DRUG_MEDIAN_RANK_OOD = "5–13"


def label_bars_with_asterisk(ax, bars, vals, placeholder_mask, padding=0.8,
                             fontsize=5.0, weight="bold"):
    """Place value labels above each bar; append '*' if placeholder is True."""
    for bar, val, is_ph in zip(bars, vals, placeholder_mask):
        if val is None:
            continue
        suffix = "*" if is_ph else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + padding,
            f"{val:.0f} %{suffix}",
            ha="center", va="bottom",
            fontsize=fontsize, weight=weight, color=BIIE.BLACK,
        )


def draw_panel(ax, panel_letter, direction_label,
               vals_id, vals_ood,
               ph_mask_id, ph_mask_ood,
               ylabel="Recall (%)"):
    x = np.arange(len(METRICS))
    width = 0.32
    bars_id = ax.bar(x - width / 2 - 0.005, vals_id, width,
                     color=BIIE.ID_FILL, edgecolor=BIIE.BLACK, linewidth=0.8)
    bars_ood = ax.bar(x + width / 2 + 0.005, vals_ood, width,
                      color=BIIE.OOD_FILL,
                      edgecolor=BIIE.BLACK, linewidth=0.8)
    label_bars_with_asterisk(ax, bars_id,  vals_id,  ph_mask_id,  padding=0.8)
    label_bars_with_asterisk(ax, bars_ood, vals_ood, ph_mask_ood, padding=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_xlim(-0.6, len(METRICS) - 0.4)
    ax.set_ylim(0, 110)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(direction_label, fontweight="bold", labelpad=10)
    ax.set_title(panel_letter, loc="left", fontweight="bold")


def main():
    apply_style()

    # Slightly wider than SIZE_2PANEL to accommodate right-side legend
    # without squeezing the panels (locked SIZE_2PANEL stays for other
    # 2-panel figures; this one needs the extra inch for the legend column).
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    draw_panel(axes[0],
               panel_letter="a",
               direction_label="Drug → Target",
               vals_id=DRUG_TO_TARGET_ID,
               vals_ood=DRUG_TO_TARGET_OOD,
               ph_mask_id=DRUG_TO_TARGET_PLACEHOLDER_MASK_ID,
               ph_mask_ood=DRUG_TO_TARGET_PLACEHOLDER_MASK_OOD)

    draw_panel(axes[1],
               panel_letter="b",
               direction_label="Target → Drug",
               vals_id=TARGET_TO_DRUG_ID,
               vals_ood=TARGET_TO_DRUG_OOD,
               ph_mask_id=TARGET_TO_DRUG_PLACEHOLDER_MASK_ID,
               ph_mask_ood=TARGET_TO_DRUG_PLACEHOLDER_MASK_OOD)

    # Right-margin frameless legend (vertical, ID stacked on OOD).
    # BIIE README permits ax.legend(frameon=False) as a fallback; here we
    # use fig.legend at right margin to consolidate the convention across
    # both panels and use the previously-empty right-side white space.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=BIIE.ID_FILL, edgecolor=BIIE.BLACK, linewidth=0.8,
              label="in-distribution"),
        Patch(facecolor=BIIE.OOD_FILL, edgecolor=BIIE.BLACK, linewidth=0.8,
              label="out-of-distribution"),
    ]
    fig.legend(handles=legend_handles, loc="center right",
               bbox_to_anchor=(0.998, 0.62), ncol=1, frameon=False,
               fontsize=8.0, handlelength=1.4, handletextpad=0.6,
               labelspacing=0.8)

    # Reserve a thin right margin for the legend; panels fill the left ~88%.
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    paths = save_figure(fig, "fig2_retrieval_bidirectional",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
