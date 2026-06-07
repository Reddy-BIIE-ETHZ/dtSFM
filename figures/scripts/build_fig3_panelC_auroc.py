#!/usr/bin/env python3
"""build_fig3_panelC_auroc.py — dtSFM v3 Fig 3, Panel C (PREVIEW).

Per-drug quantitative concordance with Klaeger 2017: zero-shot AUROC for
recovering each TKI's Klaeger kinome hits (Kd <= 30 uM) by dtSFM cosine rank,
across the 276-kinase Klaeger comparison set.

Drugs are grouped into drug-OOD (SMILES never in training -> generalization to
novel chemistry) and in-training (ID), colored by group. The drug-OOD set holds
its own — ibrutinib leads the whole panel at 0.94.

Source: audit/dtsfm/safety_per_drug_concordance.tsv
Output: dtSFM-Figures/fig3_panelC_auroc_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                      # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from matplotlib.patches import Patch    # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

A = ROOT / "audit/dtsfm"
DRUG_OOD = {"ibrutinib", "ponatinib", "crizotinib", "acalabrutinib"}
COL_ID = BIIE.BLUE
COL_OOD = BIIE.GOLD
BASE = 0.5


def main():
    apply_style()
    data = []
    with open(A / "safety_per_drug_concordance.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            data.append((r["drug"], float(r["auroc_hit_30uM"])))
    mean_auroc = sum(a for _, a in data) / len(data)

    ood = sorted([(d, a) for d, a in data if d in DRUG_OOD], key=lambda t: -t[1])
    idd = sorted([(d, a) for d, a in data if d not in DRUG_OOD], key=lambda t: -t[1])

    # display rows top->bottom: OOD group, gap, ID group
    rows = [(d, a, "OOD") for d, a in ood] + [None] + [(d, a, "ID") for d, a in idd]

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    yticks, ylabels = [], []
    for i, row in enumerate(rows):
        if row is None:
            continue
        d, a, g = row
        color = COL_OOD if g == "OOD" else COL_ID
        ax.barh(i, a - BASE, left=BASE, height=0.68, color=color,
                edgecolor=BIIE.BLACK, linewidth=0.6, zorder=3)
        ax.text(a + 0.008, i, f"{a:.2f}", va="center", ha="left",
                fontsize=6.4, color=BIIE.BLACK)
        yticks.append(i)
        ylabels.append(d)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.0)
    ax.invert_yaxis()
    ax.set_xlim(BASE, 1.0)
    ax.set_xlabel("Klaeger kinome AUROC  (Kd ≤ 30 µM)")
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.text(0.0, 1.04, "c", transform=ax.transAxes, fontweight="bold",
            fontsize=9, ha="left", va="bottom")

    ax.legend(handles=[Patch(facecolor=COL_ID, edgecolor=BIIE.BLACK, label="drug in training (ID)"),
                       Patch(facecolor=COL_OOD, edgecolor=BIIE.BLACK, label="drug-OOD (novel chemistry)")],
              loc="lower right", fontsize=5.8, frameon=False,
              handlelength=1.1, handletextpad=0.4, borderaxespad=0.3)

    fig.tight_layout()
    paths = save_figure(fig, "fig3_panelC_auroc_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    print(f"  mean AUROC@30uM = {mean_auroc:.3f}; OOD={len(ood)} ID={len(idd)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
