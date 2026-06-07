"""build_fig2_panelD.py — dtSFM v3 interface + contact head ROC.

Fig 2 Panel D. ROC curves for the interface (per-atom) and contact
(atom × residue) heads, in-distribution vs OOD (cluster-held-out val),
at epoch 10. AUROC in legend.

B&W minimal: ID solid black, OOD dashed grey; chance diagonal dotted.
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


def load_roc(head, split):
    p = PERPAIR / f"{head}_roc_{split}.tsv"
    auroc = None
    for line in p.read_text().splitlines():
        if line.startswith("# auroc="):
            auroc = float(line.split("=")[1])
            break
    data = np.loadtxt(p, skiprows=2)  # skip "fpr tpr" + "# auroc=" lines
    return data[:, 0], data[:, 1], auroc


def build():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.7))

    for ax, head, panel in [(axes[0], "interface", "c"), (axes[1], "contact", "")]:
        for split, style, color, lab in [("in_dist", "-", BLACK, "In-distribution"),
                                         ("val", "--", BIIE.BLUE, "Out-of-distribution")]:
            fpr, tpr, au = load_roc(head, split)
            ax.plot(fpr, tpr, style, color=color, lw=1.2,
                    label=f"{lab}  (AUROC {au:.2f})")
        ax.plot([0, 1], [0, 1], ":", color=BIIE.GREY_LIGHT, lw=0.8, zorder=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("False positive rate", fontsize=7.5, color=BLACK)
        if head == "interface":
            ax.set_ylabel("True positive rate", fontsize=7.5, color=BLACK)
        ax.tick_params(axis="both", labelsize=7, colors=BLACK)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BLACK); ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_color(BLACK); ax.spines["bottom"].set_linewidth(0.8)
        # Title pushed well above the plot (croppable gap for PowerPoint assembly).
        title = "Interface head (per-atom)" if head == "interface" else "Contact head (atom × residue)"
        ax.set_title(title, fontsize=7.5, color=BLACK, loc="left", pad=22)
        ax.legend(loc="lower right", fontsize=6, frameon=False, labelcolor=BLACK,
                  handlelength=1.6, handletextpad=0.5, borderaxespad=0.3)
        if panel:
            ax.text(-0.30, 1.34, panel, transform=ax.transAxes, fontsize=10,
                    weight="bold", color=BLACK, ha="left", va="top")

    fig.subplots_adjust(wspace=0.30)
    out = REPO_ROOT / "dtSFM-Figures/fig2_panelD_roc"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    build()
