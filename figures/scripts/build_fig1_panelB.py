"""build_fig1_panelB.py — dtSFM v3 training data summary.

Fig 1 Panel B. Conventional matplotlib: ax.barh + proper axis.
Two side-by-side sub-axes via gridspec:
  LEFT  — bar chart of training pair counts per source
  RIGHT — text-only coverage summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from calm.figures.biie_style import BIIE, apply_style  # noqa: E402

# Locked numbers
PDBBIND_PAIRS = 19_037
SAIR_PAIRS    = 695_710
TOTAL_PAIRS   = PDBBIND_PAIRS + SAIR_PAIRS
UNIQUE_DRUGS  = 522_776
UNIQUE_PROTS  = 22_964
assert TOTAL_PAIRS == 714_747

BLACK = "#000000"


def build():
    apply_style()
    fig = plt.figure(figsize=(7.0, 1.9))
    gs = GridSpec(1, 2, width_ratios=[1.6, 1.0], wspace=0.45, figure=fig)

    # ----------------- LEFT: bar chart -----------------
    ax = fig.add_subplot(gs[0, 0])
    sources = ["PDBbind v2020\n(Tier 1)", "SAIR Boltz-1x\n(Tier 1.5)"]
    counts  = [PDBBIND_PAIRS, SAIR_PAIRS]
    colors  = [BIIE.BLUE, "#7FAEF0"]

    bars = ax.barh(sources, counts, color=colors, edgecolor="none", height=0.6)

    # Numeric labels at end of each bar
    for bar, n in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{n:,}", ha="left", va="center", fontsize=7, color=BLACK)

    ax.set_xlim(0, max(counts) * 1.18)
    ax.set_xlabel("Training pairs", fontsize=7.5, color=BLACK)
    ax.tick_params(axis="x", labelsize=7, colors=BLACK)
    ax.tick_params(axis="y", labelsize=7, colors=BLACK, length=0)
    # Explicit ticks at 0, 250K, 500K
    ax.set_xticks([0, 250_000, 500_000])
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{int(x/1000)}K" if x else "0"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.invert_yaxis()  # PDBbind on top

    # Panel label
    ax.text(-0.18, 1.18, "b", transform=ax.transAxes,
            fontsize=10, weight="bold", color=BLACK, ha="left", va="top")

    # ----------------- RIGHT: text-only coverage summary -----------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    ax2.text(0, 1.05, "Coverage", fontsize=7.5, color=BIIE.GREY_DARK,
             ha="left", va="top")

    rows = [
        (0.78, f"{UNIQUE_DRUGS:,}", "unique drugs"),
        (0.48, f"{UNIQUE_PROTS:,}", "unique proteins"),
        (0.18, f"{TOTAL_PAIRS:,}",  "(drug, protein) pairs"),
    ]
    for y, big, sub in rows:
        ax2.text(0, y, big, fontsize=10, weight="bold", color=BLACK,
                 ha="left", va="center")
        ax2.text(0, y - 0.14, sub, fontsize=7, color=BIIE.GREY_DARK,
                 ha="left", va="center")

    out = Path("dtSFM-Figures/fig1_panelB_trainingdata")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    build()
