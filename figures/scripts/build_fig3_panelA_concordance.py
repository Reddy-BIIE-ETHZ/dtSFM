#!/usr/bin/env python3
"""build_fig3_panelA_concordance.py — dtSFM v3 paper Figure 3, Panel A (PREVIEW v4).

Side-by-side vertical concordance: Klaeger 2017 kinobeads (measured) vs dtSFM v3
(proteome screen), over the curated named off-target panel.

Actionable framing: the dtSFM value is the off-target's GENE RANK in that drug's
full proteome screen (1 = top of dtSFM's ranked safety list). A wet-lab team
would screen the top-K dtSFM hits; the panel shows that the real off-targets land
near the top, so a top-50/top-100 screen is comprehensive.

  - Genes run vertically; the 9 Klaeger-measured TKIs are columns in both panels.
  - SHARED block (top): off-targets in Klaeger's kinobeads kinome. Both panels
    dense; hot cells coincide -> dtSFM recovers the measured off-target pattern.
  - EXTENSION block (bottom): named off-targets ABSENT from Klaeger's kinobeads
    set (ERBB2/4, KDR, FLT1/4, BLK, ...). Klaeger panel stops; the dtSFM panel
    KEEPS GOING (still dense) -> proteome scale beats the wet-lab kinase panel.

Color: blue -> white -> red diverging.
  left  = measured affinity (potent Kd = red);
  right = dtSFM gene rank (top of safety list = red).

Source data (audit/dtsfm/):
  proteome_screens_panel/<drug>_d2t.csv  full per-drug proteome screen (22,965 proteins)
  protein_id_to_gene_symbol.tsv          protein_idx -> gene rollup
  klaeger2017/klaeger_kdapp_long.tsv     full Klaeger Kd matrix
  safety_screen_results_v2.tsv           kinome membership (shared vs extension split)
  safety_panel_klaeger2017.tsv           named off-target panel per drug

Output: dtSFM-Figures/fig3_panelA_concordance_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                      # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

A = ROOT / "audit/dtsfm"

DRUG_ORDER = [
    "erlotinib", "gefitinib",                 # EGFR
    "ibrutinib",                              # BTK
    "imatinib", "dasatinib", "ponatinib",     # ABL / multi
    "sunitinib", "sorafenib",                 # VEGFR / RAF
    "crizotinib",                             # ALK
]
DRUG_OOD = {"ibrutinib", "ponatinib", "crizotinib"}

FAMILY_ORDER = [
    ("EGFR/ERBB", ["EGFR", "ERBB2", "ERBB4"]),
    ("TEC",       ["BTK", "BMX", "ITK", "TEC", "TXK"]),
    ("SRC",       ["SRC", "FYN", "LYN", "LCK", "HCK", "FGR", "BLK", "YES1", "CSK"]),
    ("ABL",       ["ABL1", "ABL2"]),
    ("EPHA",      ["EPHA2", "EPHA3", "EPHA4", "EPHA5", "EPHA6", "EPHA7", "EPHA8"]),
    ("EPHB",      ["EPHB1", "EPHB2", "EPHB3", "EPHB4", "EPHB6"]),
    ("DDR",       ["DDR1", "DDR2"]),
    ("KIT/PDGFR/FLT", ["KIT", "PDGFRA", "PDGFRB", "CSF1R", "FLT3", "FLT1", "KDR", "FLT4"]),
    ("FGFR",      ["FGFR1", "FGFR2", "FGFR3", "FGFR4"]),
    ("RAF",       ["BRAF", "RAF1"]),
    ("ALK/MET",   ["ALK", "ROS1", "MET", "MST1R", "RET", "TEK"]),
    ("AMPK",      ["PRKAA2"]),
    ("other",     ["JAK3", "RIPK2"]),
]

# Color calibration (diverging blue-white-red; 0.5 = white)
PKD_LO, PKD_MID, PKD_HI = 3.0, 6.0, 9.0          # Klaeger: white at 1 uM (pKd 6) =
                                                  # its data-supported off-target
                                                  # threshold (48/51 named <1 uM)
RANK_RED, RANK_WHITE = 1.0, 30.0                   # dtSFM: white at gene-rank 30
                                                  # (strict, actionable screen size)
RANK_BLUE = 4914.0


def kl_norm(kd_nM):
    pkd = 9.0 - math.log10(max(kd_nM, 1e-3))
    return min(1.0, max(0.0, (pkd - PKD_LO) / (PKD_HI - PKD_LO)))


def dt_norm(rank):
    # log-rank -> [0,1]; white at RANK_WHITE, red at rank 1, blue deep in list
    lo, hi = math.log10(RANK_WHITE), math.log10(RANK_RED)        # hi=0
    half = (math.log10(max(rank, 1.0)) - lo) / (hi - lo)         # 0 at white, 1 at red
    if half >= 0:                                                 # rank <= white
        return 0.5 + 0.5 * min(1.0, half)
    # rank > white: scale toward blue using white..RANK_BLUE
    span = math.log10(RANK_BLUE) - math.log10(RANK_WHITE)
    frac = (math.log10(max(rank, 1.0)) - math.log10(RANK_WHITE)) / span
    return max(0.0, 0.5 - 0.5 * min(1.0, frac))


def gene_ranks_for_drug(drug):
    """Roll up the full proteome screen to gene level; return {gene: rank}."""
    best = {}
    with open(A / "proteome_screens_panel" / f"{drug}_d2t.csv") as f:
        for r in csv.DictReader(f):
            g = PIDX2GENE.get(r["protein_idx"])
            if not g:
                continue
            c = float(r["global_cosine"])
            if g not in best or c > best[g]:
                best[g] = c
    order = sorted(best, key=lambda g: -best[g])
    return {g: i + 1 for i, g in enumerate(order)}


PIDX2GENE = {}


def load():
    with open(A / "protein_id_to_gene_symbol.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["gene_symbol"]:
                PIDX2GENE[r["protein_idx"]] = r["gene_symbol"]

    panel = {}
    with open(A / "safety_panel_klaeger2017.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            panel[r["drug_name"]] = [g for g in r["off_targets_klaeger2017"].split(",") if g]

    kl = {}
    with open(A / "klaeger2017/klaeger_kdapp_long.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (r["drug_name"].lower(), r["kinase_gene"])
            kd = r["kdapp_nM"].strip()
            kl[key] = ("binder", float(kd)) if kd else ("neg", None)

    in_kinome = set()
    with open(A / "safety_screen_results_v2.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["kinome_rank"].strip():
                in_kinome.add(r["off_target_gene"])

    granks = {d: gene_ranks_for_drug(d) for d in DRUG_ORDER}
    return panel, kl, in_kinome, granks


def main():
    apply_style()
    panel, kl, in_kinome, granks = load()

    named_union = {g for gs in panel.values() for g in gs}
    shared_genes, ext_genes = [], []
    for fam, genes in FAMILY_ORDER:
        for g in genes:
            if g not in named_union:
                continue
            (shared_genes if g in in_kinome else ext_genes).append(g)
    genes = shared_genes + ext_genes
    n_shared, n_ext, n_tot = len(shared_genes), len(ext_genes), len(genes)
    gidx = {g: i for i, g in enumerate(genes)}
    nd = len(DRUG_ORDER)

    # Klaeger (shared rows): norm or NaN (grey, untested)
    M_kl = np.full((n_shared, nd), np.nan)
    for ci, d in enumerate(DRUG_ORDER):
        for ri, g in enumerate(shared_genes):
            v = kl.get((d.lower(), g))
            if v is None:
                continue
            M_kl[ri, ci] = kl_norm(v[1]) if v[0] == "binder" else 0.06

    # dtSFM (all rows): dense gene-rank, every cell
    M_dt = np.full((n_tot, nd), np.nan)
    for ci, d in enumerate(DRUG_ORDER):
        for g in genes:
            r = granks[d].get(g)
            if r is not None:
                M_dt[gidx[g], ci] = dt_norm(r)

    bwr = LinearSegmentedColormap.from_list("bwr_biie",
                                            [BIIE.BLUE, "#FFFFFF", BIIE.ALERT])
    cmap_kl = LinearSegmentedColormap.from_list("bwr_biie",
                                                [BIIE.BLUE, "#FFFFFF", BIIE.ALERT])
    cmap_kl.set_bad("#E6E6E6")

    # ---- Layout ----
    fig_h = 0.125 * n_tot + 1.6
    fig = plt.figure(figsize=(5.8, fig_h))
    top = 0.85
    bot_dt = 0.03
    rh = (top - bot_dt) / n_tot
    panel_w = 0.30
    klc_left = 0.045
    kl_left = 0.135
    dt_left = 0.55
    kl_bot = top - n_shared * rh

    ax_kl = fig.add_axes([kl_left, kl_bot, panel_w, n_shared * rh])
    ax_dt = fig.add_axes([dt_left, bot_dt, panel_w, n_tot * rh])

    ax_kl.imshow(M_kl, cmap=cmap_kl, vmin=0, vmax=1, aspect="auto",
                 interpolation="nearest", extent=[-0.5, nd - 0.5, n_shared - 0.5, -0.5])
    im = ax_dt.imshow(M_dt, cmap=bwr, vmin=0, vmax=1, aspect="auto",
                      interpolation="nearest", extent=[-0.5, nd - 0.5, n_tot - 0.5, -0.5])

    xlab = [f"{d}*" if d in DRUG_OOD else d for d in DRUG_ORDER]
    for ax, nr in ((ax_kl, n_shared), (ax_dt, n_tot)):
        ax.set_xticks(np.arange(-0.5, nd, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
        ax.grid(which="minor", color="#CFCFCF", linewidth=0.4)
        ax.set_xlim(-0.5, nd - 0.5)
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(which="both", length=0)
        ax.xaxis.set_ticks_position("top")
        ax.set_xticks(range(nd))
        ax.set_xticklabels(xlab, rotation=90, fontsize=6.0, va="bottom")

    # gene labels centered in the gutter so they read as shared by both panels
    label_x = (kl_left + panel_w + dt_left) / 2
    for i, g in enumerate(genes):
        fig.text(label_x, top - (i + 0.5) * rh, g, ha="center", va="center",
                 fontsize=5.4, color=BIIE.BLACK)

    # titles
    fig.text(kl_left + panel_w / 2, top + 0.075, "Klaeger 2017\nkinobeads",
             ha="center", va="bottom", fontsize=7.0, color=BIIE.BLACK, weight="bold")
    fig.text(dt_left + panel_w / 2, top + 0.075, "dtSFM v3\nproteome screen",
             ha="center", va="bottom", fontsize=7.0, color=BIIE.BLACK, weight="bold")

    # Klaeger colorbar (left): measured Kd
    cax_kl = fig.add_axes([klc_left, kl_bot + n_shared * rh * 0.15,
                           0.018, n_shared * rh * 0.7])
    cb_kl = fig.colorbar(plt.cm.ScalarMappable(cmap=bwr,
                         norm=plt.Normalize(0, 1)), cax=cax_kl)
    cb_kl.set_ticks([kl_norm(10000), kl_norm(1000), kl_norm(100), kl_norm(10)])
    cb_kl.set_ticklabels(["10 µM", "1 µM", "100 nM", "10 nM"], fontsize=4.8)
    cb_kl.set_label("Klaeger Kd", fontsize=5.4)
    cb_kl.ax.yaxis.set_ticks_position("left")
    cb_kl.ax.yaxis.set_label_position("left")
    cb_kl.outline.set_linewidth(0.5)

    # dtSFM colorbar (right): gene rank, actionable top-K ticks
    cax_dt = fig.add_axes([dt_left + panel_w + 0.085, bot_dt + n_tot * rh * 0.30,
                           0.018, n_tot * rh * 0.4])
    cb_dt = fig.colorbar(im, cax=cax_dt)
    cb_dt.set_ticks([dt_norm(1), dt_norm(10), dt_norm(30), dt_norm(100), dt_norm(1000)])
    cb_dt.set_ticklabels(["top 1", "top 10", "top 30", "top 100", "top 1000"],
                         fontsize=4.6)
    cb_dt.set_label("dtSFM gene rank", fontsize=5.4)
    cb_dt.outline.set_linewidth(0.5)

    fig.text(0.01, 0.985, "a", fontweight="bold", fontsize=9, ha="left", va="top")

    paths = save_figure(fig, "fig3_panelA_concordance_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    print(f"  shared={n_shared} ext={n_ext} drugs={nd}")
    plt.close(fig)


if __name__ == "__main__":
    main()
