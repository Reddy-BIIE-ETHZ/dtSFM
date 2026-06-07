#!/usr/bin/env python3
"""build_fig_e6_overlays.py — dtSFM v3 paper Figure 4c.

2x2 structural overlay figure showing AF3-predicted poses of v3-retrieved
compounds bound to NLRP3 (canonical + alt site), CD73, STING1. Ligands
colored by §F.4 leakage class (A blue-purple / B teal / C magenta).

Layout (v2): all text OUTSIDE the panel box. Each cell is an image-only panel
with a thin grey rectangle border; per-compound class + name + iPTM/PAE sit
in a small row BENEATH the panel, one line per compound, class-colored. All
labels in default Arial (no italics, no monospace).

Composes pre-rendered PyMOL PNGs (from render_fig_e6_overlays.py) into a
single matplotlib figure with BIIE typography.

Run order:
    1. rsync 9 *_model.cif files from ALPS → data/dtsfm/fig_e6_cifs/
    2. pymol -cq data/dtsfm/scripts/figures/render_fig_e6_overlays.py
    3. python data/dtsfm/scripts/figures/build_fig_e6_overlays.py

Output: dtSFM-Figures/fig4_panelC_overlays_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import csv                             # noqa: E402

import matplotlib.pyplot as plt        # noqa: E402
import matplotlib.image as mpimg       # noqa: E402
import numpy as np                     # noqa: E402
from calm.figures.biie_style import (  # noqa: E402
    apply_style, BIIE, save_figure,
)

FRAME = "#CFCFCF"          # light-grey cell frame


def autocrop_whitespace(img, threshold=0.99, margin_px=10):
    """Crop a (H, W, 3 or 4) image to the bounding box of non-white content."""
    if img.ndim != 3 or img.shape[2] < 3:
        return img
    rgb = img[..., :3].astype(np.float32)
    if rgb.max() > 1.0 + 1e-6:
        rgb = rgb / 255.0
    non_white = (rgb < threshold).any(axis=2)
    if not non_white.any():
        return img
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    r0, r1 = max(0, rows[0] - margin_px), min(img.shape[0], rows[-1] + margin_px)
    c0, c1 = max(0, cols[0] - margin_px), min(img.shape[1], cols[-1] + margin_px)
    return img[r0:r1, c0:c1]


RENDER_DIR = ROOT / "data/dtsfm/fig_e6_renders"

CLASS_COLOR = {
    "A":       BIIE.BLUE_PURPLE,
    "B":       BIIE.TEAL,
    "C":       BIIE.MAGENTA,
    "C_dark":  "#8A2BE2",
    "C_light": "#FF69B4",
}

PANELS = [
    ("NLRP3_canonical", "NLRP3 — canonical pocket", "NLRP3",
     [("Glyburide", "A"), ("MCC950", "B")]),
    ("NLRP3_alt", "NLRP3 — alternative site", "NLRP3",
     [("JC124", "C")]),
    ("CD73", "CD73", "CD73",
     [("LY3475070", "A"), ("AB680", "B"), ("PSB-12379", "C")]),
    ("STING1", "STING1", "STING1",
     [("ADU-S100", "C"), ("SR-717", "C_dark"), ("MSA-2", "C_light")]),
]


def load_metrics():
    """(target, drug) -> (iptm, pae, smiles) from the F42 cohort."""
    m = {}
    with open(ROOT / "audit/dtsfm/repurposing/F42_results.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            m[(r["target"], r["drug_name"])] = (
                r["iptm"], r["interface_pae_min"], r["drug_smiles"])
    return m


def main():
    apply_style()

    METRICS = load_metrics()
    fig = plt.figure(figsize=(7.2, 8.6))
    # 2 panel rows, each row = image + text-row pair.
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        height_ratios=[10, 3.0, 10, 3.0],
        hspace=0.10, wspace=0.10,
        top=0.94, bottom=0.02, left=0.03, right=0.97,
    )
    POS = [(0, 0), (0, 1), (2, 0), (2, 1)]

    for (img_row, col), (rkey, title, f42tgt, compounds) in zip(POS, PANELS):
        ax_img = fig.add_subplot(gs[img_row, col])
        ax_lab = fig.add_subplot(gs[img_row + 1, col])

        png_path = RENDER_DIR / f"render_{rkey}.png"
        if png_path.exists():
            img = autocrop_whitespace(mpimg.imread(png_path), threshold=0.985,
                                      margin_px=12)
            ax_img.imshow(img)
        else:
            ax_img.text(0.5, 0.5, f"render missing:\n{png_path.name}",
                        ha="center", va="center", transform=ax_img.transAxes,
                        fontsize=8, color=BIIE.GREY_DARK)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        # thin grey rectangle border
        for sp in ax_img.spines.values():
            sp.set_visible(True)
            sp.set_color(FRAME)
            sp.set_linewidth(0.7)
        ax_img.set_title(title, fontweight="bold", fontsize=9.5,
                         loc="center", pad=4)

        # Text row OUTSIDE the image, Arial regular: one line per compound,
        # "class letter + name + iPTM + PAE", class-colored.
        ax_lab.axis("off")
        ax_lab.set_xlim(0, 1); ax_lab.set_ylim(0, 1)
        n = len(compounds)
        dy = 0.95 / max(n, 1)
        for k, (drug, cls) in enumerate(compounds):
            color = CLASS_COLOR[cls]
            iptm, pae, _ = METRICS.get((f42tgt, drug), ("", "", ""))
            row = (f"{cls.split('_')[0]}   {drug}   iPTM {iptm}   PAE {float(pae):.1f} Å"
                   if iptm else f"{cls.split('_')[0]}   {drug}")
            y = 0.92 - k * dy
            ax_lab.text(0.02, y, row, ha="left", va="top",
                        fontsize=7.4, weight="bold", color=color)

    # ---- Panel label (Fig 4c) + subtitle ----
    fig.text(0.012, 0.992, "c", ha="left", va="top",
             fontsize=11.0, weight="bold", color=BIIE.BLACK)
    fig.text(0.5, 0.982,
             "AF3-predicted poses of retrieved compounds bind the canonical pockets",
             ha="center", va="top",
             fontsize=9.5, weight="bold", color=BIIE.BLACK)

    # ---- Top class legend ----
    legend_y = 0.962
    segs = [
        (0.05, "A", "(pair in training)", CLASS_COLOR["A"]),
        (0.30, "B", "(drug seen, novel pairing)", CLASS_COLOR["B"]),
        (0.66, "C", "(drug OOD, novel chemistry)", CLASS_COLOR["C"]),
    ]
    for x, letter, defn, col in segs:
        fig.text(x, legend_y, letter, color=col, fontsize=8.5, weight="bold",
                 ha="left", va="center")
        fig.text(x + 0.022, legend_y, defn, color=BIIE.GREY_DARK, fontsize=7.0,
                 ha="left", va="center")
    fig.text(0.96, legend_y, "(STING1: 3 C shades)", color=BIIE.GREY_DARK,
             fontsize=6.4, ha="right", va="center")

    paths = save_figure(fig, "fig4_panelC_overlays_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
