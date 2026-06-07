#!/usr/bin/env python3
"""build_fig3_panelD_ibrutinib.py — dtSFM v3 Fig 3, Panel D (PREVIEW).

Ibrutinib (drug-OOD) AF3 cofold gallery: one molecule, three dtSFM-predicted
off-target kinases, structurally verified by AlphaFold-3. Composes the three
PyMOL renders (render_fig3_panelD_ibrutinib.py) into a labelled row.

Layout (v2): all text OUTSIDE the panel box (legible at print size). Each cell
has a thin grey rectangle border around the structure render; the target name,
"NOVEL" / "wet-lab validated" tag, dtSFM rank, and iPTM/PAE all sit in a small
text row UNDER the image, in default Arial (no italics, no monospace). Single
ibrutinib SMILES sits at the bottom (one molecule across the three targets).

  BLK   — wet-lab validated (Klaeger 2017, prior study); top dtSFM hit
  ERBB3 — NOVEL: absent from Klaeger's ibrutinib panel, surfaced purely in
          silico (dtSFM rank + AF3), clinical-grade pose  [headline]
  JAK3  — wet-lab validated (Klaeger 2017); weaker pose (higher interface PAE)

Numbers are sourced live: iPTM + interface PAE from the AF3 summary JSONs,
dtSFM gene rank from safety_screen_results_v2.tsv.

Output: dtSFM-Figures/fig3_panelD_ibrutinib_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                      # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
import matplotlib.image as mpimg        # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

A = ROOT / "audit/dtsfm"
AF3 = A / "safety_panel_af3/ibrutinib"
RENDERS = ROOT / "data/dtsfm/fig3_panelD_renders"
FRAME = "#CFCFCF"          # light-grey cell frame

# (dirname, TARGET, role label, is_novel)
COFOLDS = [
    ("ibrutinib_blk",   "BLK",   "wet-lab validated", False),
    ("ibrutinib_erbb3", "ERBB3", "NOVEL — not in Klaeger panel", True),
    ("ibrutinib_jak3",  "JAK3",  "wet-lab validated", False),
]


def af3_metrics(dirname):
    j = json.loads((AF3 / dirname / f"{dirname}_summary_confidences.json").read_text())
    iptm = j["iptm"]
    pae = j["chain_pair_pae_min"][0][1]      # protein->ligand interface PAE (Å)
    return iptm, pae


def gene_ranks():
    """Roll up ibrutinib's full proteome screen to gene level (same source as
    Panel A); covers ERBB3, which is novel and absent from the named-pair table."""
    p2g = {}
    with open(A / "protein_id_to_gene_symbol.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["gene_symbol"]:
                p2g[r["protein_idx"]] = r["gene_symbol"]
    best = {}
    with open(A / "proteome_screens_panel/ibrutinib_d2t.csv") as f:
        for r in csv.DictReader(f):
            g = p2g.get(r["protein_idx"])
            if not g:
                continue
            c = float(r["global_cosine"])
            if g not in best or c > best[g]:
                best[g] = c
    order = sorted(best, key=lambda g: -best[g])
    return {g: i + 1 for i, g in enumerate(order)}


def crop_white(img):
    """Trim near-white margins."""
    a = img[:, :, :3] if img.ndim == 3 else img
    mask = (a < 0.985).any(axis=2)
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    pad = 6
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, img.shape[1])
    return img[r0:r1, c0:c1]


def main():
    apply_style()
    ranks = gene_ranks()

    import textwrap
    # ibrutinib SMILES — one molecule, so shown ONCE at the bottom (not per cell).
    smi = ""
    with open(A / "safety_panel_klaeger2017.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["drug_name"] == "ibrutinib":
                smi = r["canonical_smiles"].strip()
                break
    smi_w = "\n".join(textwrap.wrap(smi, 92))

    fig = plt.figure(figsize=(7.4, 4.2))
    # rows: image | text-row | spacer | smiles
    gs = fig.add_gridspec(nrows=2, ncols=3,
                          height_ratios=[10, 3.0],
                          hspace=0.06, wspace=0.12,
                          top=0.90, bottom=0.16, left=0.025, right=0.975)
    for ci, (dirname, tgt, role, novel) in enumerate(COFOLDS):
        ax_img = fig.add_subplot(gs[0, ci])
        ax_lab = fig.add_subplot(gs[1, ci])

        ax_img.imshow(crop_white(mpimg.imread(RENDERS / f"render_{tgt}.png")))
        ax_img.set_xticks([]); ax_img.set_yticks([])
        # thin grey rectangle border
        for sp in ax_img.spines.values():
            sp.set_visible(True)
            sp.set_color(FRAME)
            sp.set_linewidth(0.7)

        iptm, pae = af3_metrics(dirname)
        rk = ranks.get(tgt, "?")
        pae_col = BIIE.GREEN if pae <= 2.0 else (BIIE.BLACK if pae <= 5.0 else BIIE.ALERT)

        # Text row OUTSIDE the box — Arial regular, no italics, no monospace.
        ax_lab.axis("off")
        ax_lab.set_xlim(0, 1); ax_lab.set_ylim(0, 1)
        # row 1: TARGET (blue, bold) at left + tag (right)
        ax_lab.text(0.0, 0.95, tgt, ha="left", va="top",
                    fontsize=10.0, weight="bold", color=BIIE.BLUE)
        ax_lab.text(1.0, 0.95, role, ha="right", va="top",
                    fontsize=7.4,
                    weight="bold" if novel else "normal",
                    color=BIIE.ALERT if novel else BIIE.GREY_DARK)
        # row 2: dtSFM rank
        ax_lab.text(0.0, 0.58, f"dtSFM rank #{rk}", ha="left", va="top",
                    fontsize=7.4, color=BIIE.GREY_DARK)
        # row 3: iPTM and PAE
        ax_lab.text(0.0, 0.22, f"iPTM {iptm:.2f}   PAE {pae:.1f} Å",
                    ha="left", va="top", fontsize=7.6, color=pae_col)

    fig.text(0.012, 0.985, "d", fontweight="bold", fontsize=9, ha="left", va="top")
    fig.text(0.5, 0.965, "Ibrutinib (drug-OOD) — AF3-verified off-target cofolds",
             ha="center", va="top", fontsize=8.5, weight="bold", color=BIIE.BLACK)
    # ibrutinib SMILES once, at the bottom
    fig.text(0.5, 0.06, smi_w, ha="center", va="bottom", fontsize=5.6,
             color=BIIE.GREY_MID, linespacing=1.2)
    fig.text(0.5, 0.015, "ibrutinib", ha="center", va="bottom", fontsize=6.6,
             color=BIIE.GREY_DARK)
    paths = save_figure(fig, "fig3_panelD_ibrutinib_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    print(f"  ranks={ranks.get('BLK')},{ranks.get('ERBB3')},{ranks.get('JAK3')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
