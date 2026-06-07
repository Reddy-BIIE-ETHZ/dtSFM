"""build_fig1_panelE.py — Class A/B/C leakage breakdown table.

Fig 1 Panel E. Two-column table (Training vs Validation) showing how the
714,747 curated pairs split across the leakage taxonomy:

  A = exact (drug, protein) pair in training       (memorization)
  B = drug seen in training, pair new              (drug-level generalization)
  C = drug novel to training                       (chemistry-novel)

Training split = 100% Class A by construction; validation = 0% A, split B/C.
The contrast (diagonal of zeros) is the no-leakage integrity story.

Numbers computed from split_index + metadata_v3 join (2026-05-20):
  Training: 592,888 pairs (all Class A)
  Validation: 65,951 pairs → A=0, B=24,753 (37.5%), C=41,198 (62.5%)

Style: clean table, Arial, mostly black; subtle blue/green/purple row accents.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
from calm.figures.biie_style import BIIE, apply_style  # noqa: E402

BLACK = "#000000"

# ---- Computed pair counts (split_index × metadata_v3 join) ----
TRAIN_TOTAL = 592_888
VAL_TOTAL   = 65_951
ROWS = [
    # (class, definition, train_count, val_count, accent)
    ("A", "exact pair in training",  592_888, 0,      BIIE.BLUE),
    ("B", "drug seen, pair new",     0,       24_753, BIIE.GREEN),
    ("C", "drug novel to training",  0,       41_198, BIIE.PURPLE),
]


def tint(hex_color, alpha=0.12):
    """Return an RGBA tuple = hex_color at low alpha over white."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r/255, g/255, b/255, alpha)


def build():
    apply_style()
    fig, ax = plt.subplots(figsize=(3.7, 2.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Panel label
    ax.text(0, 99, "e", fontsize=10, weight="bold", color=BLACK,
            ha="left", va="top")

    # Column x-centers
    x_class = 6      # class letter (left edge)
    x_defn  = 14     # definition text (left-aligned)
    x_train = 70     # training count (center)
    x_val   = 90     # validation count (center)

    # Header row
    y_head = 84
    ax.text(x_defn, y_head, "Class", fontsize=7.5, color=BIIE.GREY_DARK,
            ha="left", va="center")
    ax.text(x_train, y_head, "Training", fontsize=7.5, color=BIIE.GREY_DARK,
            ha="center", va="center")
    ax.text(x_val, y_head, "Validation", fontsize=7.5, color=BIIE.GREY_DARK,
            ha="center", va="center")

    # Rows
    row_h = 22
    y0 = 70
    for i, (cls, defn, ntr, nval, accent) in enumerate(ROWS):
        yc = y0 - i * row_h
        # Subtle tinted background band for the row
        ax.add_patch(Rectangle((2, yc - row_h/2 + 1), 96, row_h - 2,
                               facecolor=tint(accent, 0.10), edgecolor="none",
                               zorder=0))
        # Accent bar on the left
        ax.add_patch(Rectangle((2, yc - row_h/2 + 1), 1.6, row_h - 2,
                               facecolor=accent, edgecolor="none", zorder=1))
        # Class letter (bold, accent color)
        ax.text(x_class, yc + 2.5, f"Class {cls}", fontsize=8.5, weight="bold",
                color=accent, ha="left", va="center")
        # Definition (small grey, below class)
        ax.text(x_class, yc - 4.5, defn, fontsize=6.5, color=BIIE.GREY_DARK,
                ha="left", va="center")
        # Training count
        tr_txt = f"{ntr:,}" if ntr else "0"
        ax.text(x_train, yc, tr_txt, fontsize=8.5, color=BLACK,
                ha="center", va="center",
                weight="bold" if ntr else "normal")
        # Validation count + percent
        val_txt = f"{nval:,}" if nval else "0"
        ax.text(x_val, yc + 2.5, val_txt, fontsize=8.5, color=BLACK,
                ha="center", va="center",
                weight="bold" if nval else "normal")
        if nval:
            ax.text(x_val, yc - 4.5, f"{100*nval/VAL_TOTAL:.0f}%",
                    fontsize=6.5, color=BIIE.GREY_DARK, ha="center", va="center")

    # Total row
    y_tot = y0 - len(ROWS) * row_h + 2
    ax.text(x_class, y_tot, "Total pairs", fontsize=7, color=BLACK,
            ha="left", va="center")
    ax.text(x_train, y_tot, f"{TRAIN_TOTAL:,}", fontsize=7, color=BLACK,
            ha="center", va="center")
    ax.text(x_val, y_tot, f"{VAL_TOTAL:,}", fontsize=7, color=BLACK,
            ha="center", va="center")

    out = REPO_ROOT / "dtSFM-Figures/fig1_panelE_taxonomy"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    build()
