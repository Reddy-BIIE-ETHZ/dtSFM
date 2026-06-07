#!/usr/bin/env python3
"""build_fig3_panelB_retrieval.py — dtSFM v3 Fig 3, Panel B (PREVIEW).

Strict leakage-controlled headline: cumulative retrieval of the n=27 Class-B
(drug, off-target gene) pairs across 5 TKIs at top-K thresholds of the
4,910-gene proteome screen, vs the random-screening baseline (K / 4910).

Class B = drug SMILES held out of training, target gene seen in training, this
specific pair never co-occurred -> a true retrospective off-target prediction.

Source: audit/dtsfm/safety_screen_summary.json (headline_class_b)
Output: dtSFM-Figures/fig3_panelB_retrieval_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                      # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

A = ROOT / "audit/dtsfm"
GENE_POOL = 4910


def main():
    apply_style()
    h = json.loads((A / "safety_screen_summary.json").read_text())["headline_class_b"]
    n_pairs = h["n_pairs_total"]
    n_drugs = h["n_drugs_contributing"]

    Ks = [10, 50, 100, 500]
    pct = [100 * h[f"top{k}_pct"] for k in Ks]
    rand = [100 * k / GENE_POOL for k in Ks]
    fold = [p / r for p, r in zip(pct, rand)]

    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    x = np.arange(len(Ks))
    bars = ax.bar(x, pct, width=0.62, color=BIIE.BLUE,
                  edgecolor=BIIE.BLACK, linewidth=0.8, zorder=3,
                  label="dtSFM v3")

    # random-screening baseline
    ax.plot(x, rand, marker="o", markersize=4, color=BIIE.ALERT,
            linewidth=1.0, linestyle=(0, (3, 2)), zorder=4)

    # % labels + enrichment fold above each bar
    for xi, p, f in zip(x, pct, fold):
        ax.text(xi, p + 1.5, f"{p:.0f}%", ha="center", va="bottom",
                fontsize=7.0, weight="bold", color=BIIE.BLACK)
        ax.text(xi, p + 7.0, f"{f:.0f}×", ha="center", va="bottom",
                fontsize=6.0, color=BIIE.ALERT)

    ax.set_xticks(x)
    ax.set_xticklabels([f"top-{k}" for k in Ks])
    ax.set_xlim(-0.6, len(Ks) - 0.4)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Class B off-targets retrieved (%)")
    ax.set_xlabel(f"dtSFM screen depth\n(n = {n_pairs} pairs, {n_drugs} TKIs, "
                  f"{GENE_POOL:,} genes)")
    ax.text(0.02, 1.04, "b", transform=ax.transAxes, fontweight="bold",
            fontsize=9, ha="left", va="bottom")
    # enrichment legend cue + random-baseline key
    ax.text(0.04, 0.96, "× = fold-enrichment\nover random screening",
            transform=ax.transAxes, fontsize=5.8, color=BIIE.ALERT,
            va="top", ha="left")
    # key glyph: a small circle on the dashed baseline (dotted line each side)
    ax.plot([0.05, 0.11], [0.80, 0.80], transform=ax.transAxes, color=BIIE.ALERT,
            linewidth=1.0, linestyle=(0, (3, 2)), clip_on=False)
    ax.plot([0.08], [0.80], transform=ax.transAxes, color=BIIE.ALERT,
            linewidth=0, marker="o", markersize=3, clip_on=False)
    ax.text(0.118, 0.80, "random screen", transform=ax.transAxes, fontsize=5.8,
            color=BIIE.ALERT, va="center", ha="left")

    fig.tight_layout()
    paths = save_figure(fig, "fig3_panelB_retrieval_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    print(f"  pct={[round(p,1) for p in pct]} fold={[round(f,1) for f in fold]}")
    plt.close(fig)


if __name__ == "__main__":
    main()
