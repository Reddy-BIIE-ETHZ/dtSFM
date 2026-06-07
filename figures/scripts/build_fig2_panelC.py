"""build_fig2_panelC.py — dtSFM v3 affinity head (PRELIMINARY supplement).

DEMOTED from main Fig 2 (2026-05-22). The affinity head is trained on
PDBbind + SAIR labels, where SAIR affinities are Boltz-2-PREDICTED, not
measured. So this scatter is largely the head reproducing its own
(mostly Boltz-2-derived) training labels — a label-consistency check,
NOT experimental-affinity validation. Real-Kd validation is the §3b
dtSFM-vs-Boltz-2 experimental run. Renders to a supplementary figure.

B&W minimal: small black points, grey y=x line.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
from calm.figures.biie_style import BIIE, apply_style  # noqa: E402

PERPAIR = REPO_ROOT / "data/dtsfm/fig2/perpair"
BLACK = "#000000"


def load(split):
    a = np.loadtxt(PERPAIR / f"affinity_per_pair_{split}.tsv", skiprows=1)
    return a[:, 0], a[:, 1]  # pred, target


def build():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.7))

    for ax, split, label, panel in [(axes[0], "in_dist", "In-distribution", ""),
                                     (axes[1], "val", "Out-of-distribution", "")]:
        pred, tgt = load(split)
        r = np.corrcoef(pred, tgt)[0, 1]
        ax.scatter(tgt, pred, s=2, c=BLACK, alpha=0.25, edgecolors="none",
                   rasterized=True)
        lim = [min(tgt.min(), pred.min()) - 0.5, max(tgt.max(), pred.max()) + 0.5]
        ax.plot(lim, lim, "--", color=BIIE.GREY_MID, lw=0.8, zorder=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Training-target pAffinity\n(PDBbind + Boltz-2/SAIR labels)",
                      fontsize=6.8, color=BLACK)
        if split == "in_dist":
            ax.set_ylabel("Predicted pAffinity", fontsize=7.5, color=BLACK)
        ax.tick_params(axis="both", labelsize=7, colors=BLACK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BLACK); ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_color(BLACK); ax.spines["bottom"].set_linewidth(0.8)
        ax.set_title(label, fontsize=7.5, color=BLACK, loc="left", pad=4)
        ax.text(0.05, 0.95, f"r = {r:.2f}\nn = {len(pred):,}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                color=BLACK, linespacing=1.3)
        if panel:
            ax.text(-0.28, 1.10, panel, transform=ax.transAxes, fontsize=10,
                    weight="bold", color=BLACK, ha="left", va="top")

    fig.suptitle("Preliminary affinity head — reproduces (mostly Boltz-2-derived) "
                 "training labels, not experimental affinity",
                 fontsize=6.6, color=BLACK, y=1.02)
    fig.subplots_adjust(wspace=0.30)
    out = REPO_ROOT / "dtSFM-Figures/figS2_affinity_preliminary"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    build()
