"""build_fig1_S1_tanimoto.py — Fig 1 supplementary S1.1.

ECFP4 max-Tanimoto of held-out drugs (val + test, 5K subsample each) to the
nearest training-set drug, as a bucketed bar histogram. Discloses split
softness: ~37% of held-out drugs are near-duplicates (Tan >= 0.9), ~1-2%
strictly novel (< 0.3).

Conventional matplotlib grouped bars, Arial, minimal color per locked style.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
from calm.figures.biie_style import BIIE, apply_style  # noqa: E402

DATA = REPO_ROOT / "data/dtsfm/fig1_supp"
BLACK = "#000000"


def build():
    apply_style()
    df = pd.read_csv(DATA / "drug_tanimoto_buckets.tsv", sep="\t")
    buckets = ["[0.0, 0.3)", "[0.3, 0.5)", "[0.5, 0.7)", "[0.7, 0.9)", "[0.9, 1.0]"]
    labels = ["0.0–0.3", "0.3–0.5", "0.5–0.7", "0.7–0.9", "0.9–1.0"]
    val = df[df.split == "val"].set_index("bucket").reindex(buckets)["frac"].values * 100
    test = df[df.split == "test"].set_index("bucket").reindex(buckets)["frac"].values * 100

    x = np.arange(len(buckets))
    w = 0.38

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.bar(x - w/2, val, w, color=BIIE.BLUE, label="Validation", edgecolor="none")
    ax.bar(x + w/2, test, w, color=BIIE.GREEN, label="Test", edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, color=BLACK)
    ax.set_xlabel("ECFP4 max-Tanimoto to nearest training drug", fontsize=7.5, color=BLACK)
    ax.set_ylabel("Held-out drugs (%)", fontsize=7.5, color=BLACK)
    ax.tick_params(axis="both", labelsize=7, colors=BLACK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)

    # Interpretive bucket annotations just under the tick labels
    ann = ["strictly\nnovel", "novel\nscaffold", "", "", "near-\nduplicate"]
    for xi, a in zip(x, ann):
        if a:
            ax.annotate(a, xy=(xi, 0), xytext=(0, -28),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=4.8, color=BIIE.GREY_DARK, style="italic",
                        annotation_clip=False)

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=6.5,
              frameon=False, labelcolor=BLACK, handlelength=1.4,
              handletextpad=0.5)

    ax.set_ylim(0, 42)
    out = REPO_ROOT / "dtSFM-Figures/figS1_2_tanimoto_novelty"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    build()
