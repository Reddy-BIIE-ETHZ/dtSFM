#!/usr/bin/env python3
"""build_fig5_panelB_quality.py — dtSFM Fig 5 panel b: generation quality.

What the base decoder produces, chemically (AF3 confirmation is panel c). Pooled
dot plot, one dot per generated candidate:

  x = QED (drug-likeness)
  y = max Tanimoto to the nearest APPROVED drug (chemical similarity to marketed
      compounds; an approved drug scores 1.0 against itself)

Read-out: every candidate sits well below 0.4 (median 0.22) — novel scaffolds,
not rediscoveries of marketed drugs. QED is modest (median 0.36): the base
decoder is not ADMET-optimised — that is Paper 2 (Generative Design Loop).

Source: audit/dtsfm/decoder_af3/F5_2_selectivity_reranked.tsv
Output: dtSFM-Figures/fig5_panelB_quality_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import csv                              # noqa: E402
import statistics as st                 # noqa: E402

import matplotlib.pyplot as plt         # noqa: E402
import matplotlib.patheffects as pe     # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

SRC = ROOT / "audit/dtsfm/decoder_af3/F5_2_selectivity_reranked.tsv"
NOVEL_THRESH = 0.4


def main():
    apply_style()
    qed, tan = [], []
    for r in csv.DictReader(open(SRC), delimiter="\t"):
        try:
            q, t = float(r["qed"]), float(r["max_tanimoto_to_approved"])
        except (TypeError, ValueError):
            continue
        qed.append(q)
        tan.append(t)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    YMAX = 0.45

    # novelty region (Tanimoto < 0.4 to any approved drug)
    ax.axhspan(0.0, NOVEL_THRESH, color=BIIE.BLUE, alpha=0.05, zorder=0)
    ax.axhline(NOVEL_THRESH, ls="--", lw=0.8, color=BIIE.GREY_DARK, zorder=1)

    ax.scatter(qed, tan, s=11, color=BIIE.BLUE, alpha=0.40, edgecolors="none",
               zorder=3)

    # approved drugs sit at 1.0 — off the cropped scale (arrow at top)
    ax.annotate("approved drugs = 1.0 (off-scale)", xy=(0.5, YMAX),
                xytext=(0.5, YMAX - 0.015), ha="center", va="top",
                fontsize=5.8, color=BIIE.GREY_DARK, style="italic",
                arrowprops=dict(arrowstyle="-|>", color=BIIE.GREY_DARK, lw=0.6),
                annotation_clip=False)
    ax.text(0.97, 0.06, "novel scaffolds (< 0.4 to any approved drug)",
            transform=ax.transAxes, fontsize=6.0, color=BIIE.BLUE,
            style="italic", ha="right", va="bottom", path_effects=halo)
    ax.text(0.03, 0.93,
            f"median QED {st.median(qed):.2f}\nmedian Tanimoto {st.median(tan):.2f}  (n = {len(qed):,})",
            transform=ax.transAxes, fontsize=6.4, color=BIIE.GREY_DARK,
            ha="left", va="top", path_effects=halo)

    ax.set_xlabel("QED (drug-likeness)", labelpad=4)
    ax.set_ylabel("max Tanimoto to nearest approved drug", labelpad=4)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, YMAX)
    ax.text(0.0, 1.04, "b", transform=ax.transAxes, fontweight="bold",
            fontsize=9, ha="left", va="bottom")

    fig.tight_layout()
    paths = save_figure(fig, "fig5_panelB_quality_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    print(f"  n={len(qed)} medQED={st.median(qed):.2f} medTan={st.median(tan):.2f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
