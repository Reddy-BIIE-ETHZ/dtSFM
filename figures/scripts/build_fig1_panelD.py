"""build_fig1_panelD.py — dtSFM v3 training convergence.

Fig 1 Panel D. Per-epoch held-out validation R@10 for both retrieval
directions, with epoch_010 (locked production checkpoint) marked.

Conventional matplotlib line plot. Arial, minimal color per locked style.

Input: data/dtsfm/fig1_panelD/quick_eval_val.csv (B-3 run, OOD/cluster-held-out val)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
from calm.figures.biie_style import BIIE, apply_style  # noqa: E402

DATA_DIR = REPO_ROOT / "data/dtsfm/fig1_panelD"
BLACK = "#000000"
LOCKED_EPOCH = 10


def build():
    apply_style()
    df = pd.read_csv(DATA_DIR / "quick_eval_val.csv")
    # Dedup epochs (keep last occurrence per epoch), sort
    df = df.drop_duplicates("epoch", keep="last").sort_values("epoch")
    ep = df["epoch"].values
    d2t = df["d2t_R@10"].values * 100
    t2d = df["t2d_R@10"].values * 100

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    ax.plot(ep, d2t, "-o", color=BIIE.BLUE, markersize=3, linewidth=1.3,
            label="Drug → Target (safety)")
    ax.plot(ep, t2d, "-s", color=BIIE.GREEN, markersize=3, linewidth=1.3,
            label="Target → Drug (repurposing)")

    # Locked checkpoint marker
    ax.axvline(LOCKED_EPOCH, color=BIIE.GREY_MID, linestyle="--", linewidth=0.8,
               zorder=0)
    ax.text(LOCKED_EPOCH - 0.3, ax.get_ylim()[0] + 3, "locked\nepoch 10",
            fontsize=6, color=BIIE.GREY_DARK, ha="right", va="bottom")

    ax.set_xlabel("Training epoch", fontsize=7.5, color=BLACK)
    ax.set_ylabel("Held-out R@10 (%)", fontsize=7.5, color=BLACK)
    ax.tick_params(axis="both", labelsize=7, colors=BLACK)
    ax.set_xlim(0.5, ep.max() + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, int(ep.max()) + 1, 3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=6.5,
              frameon=False, labelcolor=BLACK, handlelength=1.6,
              handletextpad=0.5, borderaxespad=0.3)

    # Panel label
    ax.text(-0.18, 1.08, "d", transform=ax.transAxes,
            fontsize=10, weight="bold", color=BLACK, ha="left", va="top")

    out = REPO_ROOT / "dtSFM-Figures/fig1_panelD_convergence"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}.pdf + .png  (epochs {ep.min()}..{ep.max()})")
    plt.close(fig)


if __name__ == "__main__":
    build()
