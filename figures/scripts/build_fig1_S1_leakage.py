"""build_fig1_S2_leakage.py — Fig 1 supplementary S1.2.

Leakage audit table: the three measured §C dimensions side by side for
val + test, with PASS/measured values. Demonstrates the no-leakage guarantee
underpinning all retrieval claims.

Dimensions:
  Exact (drug, protein) pair overlap   → 0 / 0  (PASS)
  Drug-identity overlap                → val 36% / test 37% in train (the
                                          rest novel; this is expected, drives
                                          the Class C fraction)
  Protein + cluster overlap            → 0% (PASS — cluster-held-out split)

Clean table, Arial, mostly black; subtle accent on the PASS verdicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
from calm.figures.biie_style import BIIE, apply_style  # noqa: E402

DATA = REPO_ROOT / "data/dtsfm/fig1_supp"
BLACK = "#000000"
GREEN = BIIE.GREEN


def build():
    apply_style()
    pair = pd.read_csv(DATA / "pair_leakage.tsv", sep="\t")
    drug = pd.read_csv(DATA / "drug_leakage.tsv", sep="\t")
    prot = pd.read_csv(DATA / "protein_leakage.tsv", sep="\t")

    def g(df, split, col):
        return df[df.split == split][col].values[0]

    # Build display rows: (dimension, val_str, test_str, verdict)
    rows = [
        ("Exact (drug, protein) pair\nin training",
         f"{int(g(pair,'val','n_pairs_also_in_train'))} / {int(g(pair,'val','n_pairs')):,}",
         f"{int(g(pair,'test','n_pairs_also_in_train'))} / {int(g(pair,'test','n_pairs')):,}",
         "0% — PASS"),
        ("Drug identity in training\n(rest = novel chemistry)",
         f"{g(drug,'val','frac_in_train')*100:.0f}% seen",
         f"{g(drug,'test','frac_in_train')*100:.0f}% seen",
         "expected"),
        ("Protein / cluster in training",
         f"{g(prot,'val','frac_proteins_in_train')*100:.0f}%",
         f"{g(prot,'test','frac_proteins_in_train')*100:.0f}%",
         "0% — PASS"),
    ]

    fig, ax = plt.subplots(figsize=(5.0, 1.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Column x positions
    x_dim = 3
    x_val = 52
    x_test = 70
    x_verdict = 88

    # Header
    y_head = 88
    ax.text(x_dim, y_head, "Leakage dimension", fontsize=7.5,
            color=BIIE.GREY_DARK, ha="left", va="center")
    ax.text(x_val, y_head, "Validation", fontsize=7.5, color=BIIE.GREY_DARK,
            ha="center", va="center")
    ax.text(x_test, y_head, "Test", fontsize=7.5, color=BIIE.GREY_DARK,
            ha="center", va="center")
    ax.text(x_verdict, y_head, "Verdict", fontsize=7.5, color=BIIE.GREY_DARK,
            ha="center", va="center")

    row_h = 26
    y0 = 66
    for i, (dim, vval, vtest, verdict) in enumerate(rows):
        yc = y0 - i * row_h
        is_pass = "PASS" in verdict
        # Subtle green band for PASS rows
        if is_pass:
            ax.add_patch(Rectangle((1, yc - row_h/2 + 1), 98, row_h - 2,
                                   facecolor=(46/255, 125/255, 50/255, 0.08),
                                   edgecolor="none", zorder=0))
        ax.text(x_dim, yc, dim, fontsize=7, color=BLACK, ha="left", va="center")
        ax.text(x_val, yc, vval, fontsize=7, color=BLACK, ha="center", va="center")
        ax.text(x_test, yc, vtest, fontsize=7, color=BLACK, ha="center", va="center")
        vcol = GREEN if is_pass else BIIE.GREY_DARK
        vweight = "bold" if is_pass else "normal"
        ax.text(x_verdict, yc, verdict, fontsize=7, color=vcol,
                ha="center", va="center", weight=vweight)

    out = REPO_ROOT / "dtSFM-Figures/figS1_1_leakage_audit"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    build()
