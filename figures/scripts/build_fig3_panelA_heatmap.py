#!/usr/bin/env python3
"""build_fig3_panelA_heatmap.py — dtSFM v3 paper Figure 3, Panel A (PREVIEW).

dtSFM x Klaeger 2017 off-target concordance matrix.

  - Rows  = 10 clinical TKIs (grouped by intended-target family).
  - Cols  = named off-target kinases (Klaeger panel), grouped by kinase family,
            restricted to kinases present in Klaeger's measured kinome.
  - Fill  = dtSFM within-kinome retrieval percentile (GREYSCALE; dark = dtSFM
            ranks this kinase near the top of the drug's kinome).
  - Marker = Klaeger-2017 Kinobeads-confirmed off-target (Kd measured), colored
            by leakage class (A blue / B green / C purple), sized by potency
            (-log10 Kd). Concordance = colored markers on dark cells; Class-B
            (green) markers on dark cells are the strict retrospective-prediction
            headline.

Source data (audit/dtsfm/):
  safety_panel_klaeger2017.tsv     named off-target panel per drug
  safety_panel_pair_leakage.tsv    per-pair leakage class (A/B/C)
  safety_screen_heatmap_data.tsv   within-kinome percentile + Klaeger Kd

Output: dtSFM-Figures/fig3_panelA_heatmap_PREVIEW.{pdf,png}
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
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

A = ROOT / "audit/dtsfm"

# Row order: grouped by intended-target family. * = drug-OOD (SMILES not in v3).
DRUG_ORDER = [
    "erlotinib", "gefitinib",                 # EGFR
    "acalabrutinib", "ibrutinib",             # BTK
    "imatinib", "dasatinib", "ponatinib",     # ABL / multi
    "sunitinib", "sorafenib",                 # VEGFR / RAF
    "crizotinib",                             # ALK
]
DRUG_OOD = {"acalabrutinib", "ibrutinib", "ponatinib", "crizotinib"}

# Column families (biological grouping). Filtered to genes present in kinome.
FAMILIES = [
    ("EGFR",   ["EGFR", "ERBB2", "ERBB4"]),
    ("TEC",    ["BTK", "BMX", "ITK", "TEC", "TXK"]),
    ("SRC",    ["SRC", "FYN", "LYN", "LCK", "HCK", "FGR", "BLK", "YES1", "FRK", "CSK"]),
    ("ABL",    ["ABL1", "ABL2"]),
    ("EPHA",   ["EPHA2", "EPHA3", "EPHA4", "EPHA5", "EPHA6", "EPHA7", "EPHA8"]),
    ("EPHB",   ["EPHB1", "EPHB2", "EPHB3", "EPHB4", "EPHB6"]),
    ("DDR",    ["DDR1", "DDR2"]),
    ("KIT/PDGFR/FLT", ["KIT", "PDGFRA", "PDGFRB", "CSF1R", "FLT3", "FLT1", "KDR", "FLT4"]),
    ("FGFR",   ["FGFR1", "FGFR2", "FGFR3", "FGFR4"]),
    ("RAF",    ["BRAF", "RAF1"]),
    ("ALK/MET", ["ALK", "ROS1", "MET", "MST1R", "RET", "TEK"]),
    ("other",  ["JAK3", "RIPK2", "PRKAA1", "PRKAA2"]),
]

CLASS_COLOR = {"A": BIIE.BLUE, "B": BIIE.GREEN, "C": BIIE.PURPLE}


def load():
    panel = {}
    with open(A / "safety_panel_klaeger2017.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            panel[r["drug_name"]] = [g for g in r["off_targets_klaeger2017"].split(",") if g]

    cls = {}
    with open(A / "safety_panel_pair_leakage.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cls[(r["drug_name"], r["off_target_gene"])] = r["leakage_class"]

    pct, kd = {}, {}
    with open(A / "safety_screen_heatmap_data.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (r["drug_name"], r["kinase_gene"])
            if r["percentile_of_kinome"].strip():
                pct[key] = float(r["percentile_of_kinome"])
            if r["klaeger_kdapp_nM"].strip():
                kd[key] = float(r["klaeger_kdapp_nM"])
    return panel, cls, pct, kd


def main():
    apply_style()
    panel, cls, pct, kd = load()

    named_union = {g for gs in panel.values() for g in gs}
    in_kinome = {g for (d, g) in pct}

    # Build ordered column list + family spans
    cols, col_family, fam_spans = [], [], []
    for fam, genes in FAMILIES:
        present = [g for g in genes if g in named_union and g in in_kinome]
        if not present:
            continue
        start = len(cols)
        for g in present:
            cols.append(g)
            col_family.append(fam)
        fam_spans.append((fam, start, len(cols)))
    ncol = len(cols)
    nrow = len(DRUG_ORDER)
    col_idx = {g: i for i, g in enumerate(cols)}

    # Fill matrix: percentile (0-100), NaN where not measured for that drug
    M = np.full((nrow, ncol), np.nan)
    for ri, d in enumerate(DRUG_ORDER):
        for g in cols:
            if (d, g) in pct:
                M[ri, col_idx[g]] = pct[(d, g)]

    # Greyscale colormap: white (low) -> near-black (high retrieval)
    grey = LinearSegmentedColormap.from_list("dt_grey", ["#FFFFFF", "#1A1A1A"])
    grey.set_bad("#F0F0F0")

    fig_w = 0.215 * ncol + 2.4
    fig_h = 0.30 * nrow + 1.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # PowerNorm: keep low/mid percentiles light so only top ranks darken,
    # making the Klaeger-confirmed markers-on-dark concordance read clearly.
    im = ax.imshow(M, cmap=grey, norm=PowerNorm(gamma=2.4, vmin=0, vmax=100),
                   aspect="auto", interpolation="nearest")

    # Cell gridlines
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    # Markers: Klaeger-confirmed off-targets, colored by leakage class, sized by potency
    for ri, d in enumerate(DRUG_ORDER):
        for g in panel.get(d, []):
            if g not in col_idx:
                continue
            ci = col_idx[g]
            k = kd.get((d, g))
            leak = cls.get((d, g), "C")
            color = CLASS_COLOR.get(leak, BIIE.PURPLE)
            # thin class-colored border on every named-panel cell
            ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, fill=False,
                                       edgecolor=color, linewidth=0.7, zorder=3))
            if k is not None:
                pkd = 9.0 - math.log10(max(k, 1e-3))      # nM -> pKd
                size = 8 + max(0.0, pkd - 5.0) * 13.0
                ax.scatter(ci, ri, s=size, facecolor=color, edgecolor="white",
                           linewidth=0.5, zorder=4)

    # Family separators + labels
    for fam, start, end in fam_spans:
        if start > 0:
            ax.axvline(start - 0.5, color=BIIE.GREY_DARK, linewidth=0.8, zorder=5)
        ax.text((start + end - 1) / 2, -1.15, fam, ha="center", va="bottom",
                fontsize=5.6, color=BIIE.GREY_DARK, rotation=0)

    # Axis labels
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(cols, rotation=90, fontsize=5.3)
    ax.set_yticks(range(nrow))
    ylabels = [f"{d}*" if d in DRUG_OOD else d for d in DRUG_ORDER]
    ax.set_yticklabels(ylabels, fontsize=7.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)

    # Colorbar for percentile
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label("dtSFM kinome percentile", fontsize=6.0)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    cbar.outline.set_linewidth(0.6)

    # Legend (markers): leakage class — placed below in whitespace
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                   markerfacecolor=CLASS_COLOR["B"], markeredgecolor="white",
                   label="Class B (drug-OOD pair held out — retrospective)"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                   markerfacecolor=CLASS_COLOR["A"], markeredgecolor="white",
                   label="Class A (pair in training — memorized)"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                   markerfacecolor=CLASS_COLOR["C"], markeredgecolor="white",
                   label="Class C (drug fully OOD)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.32),
              fontsize=5.6, ncol=1, handletextpad=0.4, borderaxespad=0.0,
              labelspacing=0.3)

    ax.text(-0.06, 1.18, "a", transform=ax.transAxes, fontweight="bold",
            fontsize=9, ha="left", va="top")

    fig.tight_layout()
    paths = save_figure(fig, "fig3_panelA_heatmap_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    # diagnostics
    print(f"  rows={nrow} cols={ncol}")
    print(f"  named off-targets dropped (not in kinome): "
          f"{sorted(named_union - in_kinome)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
