#!/usr/bin/env python3
"""build_fig5_panelA_pipeline.py — dtSFM Fig 5 panel a: 5-stage design pipeline.

Schematic of the generative-design cascade:
  decoder generation -> encoder rerank -> AF3 verify -> proteome safety -> AF3 selectivity

Clean box-and-arrow flow in the locked style. No panel letter/title (added in
PowerPoint). Output: dtSFM-Figures/fig5_panelA_pipeline_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

# (stage no, title, line1, line2)
STAGES = [
    ("1", "Decoder", "target-conditioned", "generation"),
    ("2", "Encoder rerank", "dtSFM cosine", "prioritize binders"),
    ("3", "AF3 verify", "on-target cofold", "1,146 / 1,199 bind"),
    ("4", "Proteome safety", "off-target screen", "vs human proteome"),
    ("5", "AF3 selectivity", "paralog cofold", "binding profile"),
]


def main():
    apply_style()
    n = len(STAGES)
    fig, ax = plt.subplots(figsize=(9.2, 1.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    bw, gap = 15.5, 3.6
    x0 = 1.0
    bh, by = 22, 4
    for i, (num, title, l1, l2) in enumerate(STAGES):
        x = x0 + i * (bw + gap)
        # box
        ax.add_patch(FancyBboxPatch((x, by), bw, bh,
                                    boxstyle="round,pad=0.3,rounding_size=2.2",
                                    linewidth=1.1, edgecolor=BIIE.BLUE,
                                    facecolor="#EAF1FB", zorder=2))
        # stage-number chip
        ax.add_patch(plt.Circle((x + 2.7, by + bh - 3.0), 1.9, color=BIIE.BLUE, zorder=3))
        ax.text(x + 2.7, by + bh - 3.0, num, ha="center", va="center",
                fontsize=7.5, weight="bold", color="white", zorder=4)
        ax.text(x + bw / 2 + 1.4, by + bh - 3.0, title, ha="center", va="center",
                fontsize=8.2, weight="bold", color=BIIE.BLACK, zorder=4)
        ax.text(x + bw / 2, by + bh / 2 - 1.5, l1, ha="center", va="center",
                fontsize=6.6, color=BIIE.GREY_DARK, zorder=4)
        ax.text(x + bw / 2, by + 3.4, l2, ha="center", va="center",
                fontsize=6.6, style="italic", color=BIIE.BLUE, zorder=4)
        # arrow to next
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + bw + 0.2, by + bh / 2),
                                         (x + bw + gap - 0.2, by + bh / 2),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.3, color=BIIE.GREY_DARK, zorder=1))

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    paths = save_figure(fig, "fig5_panelA_pipeline_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
