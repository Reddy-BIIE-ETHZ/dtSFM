"""build_fig2_panelAB.py — dtSFM v3 encoder retrieval performance.

Fig 2 Panels A (drug->target) + B (target->drug). Recall-at-K bars,
in-distribution vs OOD (cluster-held-out val), at the locked epoch-10
checkpoint, with bootstrap 95% CIs over the per-query ranks.

B&W: solid black = in-distribution, white/thin-edge = OOD.
Error bars = 95% bootstrap CI (1000 resamples of the query set).
Median rank lives in the Panel E dashboard, not here.
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
KS = [1, 5, 10, 50, 100]
N_BOOT = 1000
RNG = np.random.default_rng(42)


def rk_with_ci(ranks):
    """Point R@K (%) + 95% bootstrap CI (%) per K, from a 1D rank array."""
    ranks = np.asarray(ranks)
    n = len(ranks)
    point, lo, hi = [], [], []
    # Pre-draw bootstrap indices once for all K
    boot_idx = RNG.integers(0, n, size=(N_BOOT, n))
    for k in KS:
        hit = (ranks <= k).astype(float)
        point.append(100 * hit.mean())
        boot = 100 * hit[boot_idx].mean(axis=1)
        lo.append(np.percentile(boot, 2.5))
        hi.append(np.percentile(boot, 97.5))
    return np.array(point), np.array(lo), np.array(hi)


def load_ranks(direction, split):
    return np.loadtxt(PERPAIR / f"ranks_{direction}_{split}.tsv", skiprows=1)


def build():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))

    handles = None
    for ax, direction, panel in [(axes[0], "d2t", "a"), (axes[1], "t2d", "b")]:
        id_pt, id_lo, id_hi = rk_with_ci(load_ranks(direction, "in_dist"))
        ood_pt, ood_lo, ood_hi = rk_with_ci(load_ranks(direction, "val"))
        x = np.arange(len(KS))
        w = 0.38
        id_err = np.vstack([id_pt - id_lo, id_hi - id_pt])
        ood_err = np.vstack([ood_pt - ood_lo, ood_hi - ood_pt])
        b1 = ax.bar(x - w/2, id_pt, w, color=BLACK, edgecolor="none",
                    label="In-distribution",
                    yerr=id_err, error_kw=dict(elinewidth=0.7, capsize=1.5,
                                               capthick=0.7, ecolor=BIIE.GREY_MID))
        b2 = ax.bar(x + w/2, ood_pt, w, facecolor="white", edgecolor=BLACK,
                    linewidth=0.8, label="Out-of-distribution",
                    yerr=ood_err, error_kw=dict(elinewidth=0.7, capsize=1.5,
                                                capthick=0.7, ecolor=BIIE.GREY_MID))
        if handles is None:
            handles = [b1, b2]
        ax.set_xticks(x)
        ax.set_xticklabels([f"R@{k}" for k in KS], fontsize=7, color=BLACK)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Recall (%)", fontsize=7.5, color=BLACK)
        ax.tick_params(axis="both", labelsize=7, colors=BLACK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BLACK); ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_color(BLACK); ax.spines["bottom"].set_linewidth(0.8)
        title = ("Drug → Target  (safety screening)" if direction == "d2t"
                 else "Target → Drug  (repurposing)")
        ax.set_title(title, fontsize=7.5, color=BLACK, loc="left", pad=4)
        ax.text(-0.16, 1.10, panel, transform=ax.transAxes, fontsize=10,
                weight="bold", color=BLACK, ha="left", va="top")

    fig.legend(handles=handles, labels=["In-distribution", "Out-of-distribution"],
               loc="lower center", ncol=2, fontsize=7, frameon=False,
               labelcolor=BLACK, handlelength=1.3, handletextpad=0.5,
               columnspacing=2.0, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.32, bottom=0.22)

    out = REPO_ROOT / "dtSFM-Figures/fig2_panelAB_retrieval"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    build()
