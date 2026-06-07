"""build_fig1_panelC.py — dtSFM v3 chemical + protein space coverage.

Fig 1 Panel C. Conventional matplotlib:
  LEFT  — ax.hexbin density of 522,776 drug embeddings (MoLFormer-XL UMAP)
  RIGHT — ax.scatter of 22,990 protein embeddings (ESM-2-650M UMAP) colored
          by gene-family prefix groups.

Inputs (rsynced from Euler):
  data/dtsfm/fig1_panelC/drug_umap.tsv      (drug_idx, umap_x, umap_y)
  data/dtsfm/fig1_panelC/protein_umap.tsv   (protein_idx, umap_x, umap_y, gene_symbol)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]  # CALM-0.1.0
sys.path.insert(0, str(REPO_ROOT / "src"))
from calm.figures.biie_style import BIIE, apply_style  # noqa: E402

DATA_DIR = REPO_ROOT / "data/dtsfm/fig1_panelC"
BLACK = "#000000"


# Gene-family prefix groups (lookup by gene symbol prefix)
FAMILIES = [
    ("Kinases",         ["JAK", "MAP", "CDK", "EGFR", "BTK", "ALK", "FLT",
                          "ABL", "SRC", "AKT", "MTOR", "PIK3", "AURK", "RAF",
                          "ERBB", "PLK", "TYK", "TBK", "MEK", "ROS1", "RET",
                          "MET", "KIT", "PDGF", "KDR", "FGFR", "TRK", "NTRK",
                          "CSF1R", "FLT3", "FLT1", "FLT4", "JNK", "HCK", "FYN",
                          "LCK", "ITK", "TEC"], "#1565C0"),  # BIIE.BLUE
    ("GPCRs",           ["ADR", "HTR", "CHR", "DRD", "OPRM", "OPRD", "OPRK",
                          "HRH", "CCKR", "CCR", "CXCR", "S1PR", "P2RY",
                          "MTNR", "GHSR", "GIPR", "GLP1R", "AGTR", "EDNR",
                          "FFAR", "TBXA", "PTGER", "PTGDR", "GPER", "MC",
                          "BDKRB"], "#2E7D32"),  # BIIE.GREEN
    ("Ion channels",    ["KCN", "CACN", "SCN", "CLCN", "TRPV", "TRPA", "TRPM",
                          "TRPC", "ASIC", "GABR", "GLR", "CHRN", "CHRM",
                          "P2RX", "HCN"], "#6A1B9A"),  # BIIE.PURPLE
    ("Proteases",       ["CASP", "MMP", "CTSL", "CTSB", "CTSD", "CTSK",
                          "ELANE", "TPSAB", "PRSS", "F2", "F7", "F9", "F10",
                          "F11", "KLKB", "PLAU", "TMPRSS", "ADAM", "BACE",
                          "RENIN", "AGT"], "#F59E0B"),  # BIIE.GOLD
    ("Nuclear receptors", ["NR", "ESR", "AR", "PPAR", "RXR", "VDR", "RAR",
                          "THR", "PR", "GR"], "#C62828"),  # BIIE.ALERT
]
OTHER_COLOR = "#CCCCCC"


def assign_family(gene_symbol: str) -> tuple[str, str]:
    if not isinstance(gene_symbol, str) or not gene_symbol:
        return ("Other", OTHER_COLOR)
    gs = gene_symbol.upper()
    for fam_name, prefixes, color in FAMILIES:
        for p in prefixes:
            if gs.startswith(p):
                return (fam_name, color)
    return ("Other", OTHER_COLOR)


def build():
    apply_style()

    drug = pd.read_csv(DATA_DIR / "drug_umap.tsv", sep="\t")
    prot = pd.read_csv(DATA_DIR / "protein_umap.tsv", sep="\t")
    print(f"drug: {len(drug):,}  protein: {len(prot):,}")

    # Assign families to proteins
    fam = [assign_family(g) for g in prot["gene_symbol"]]
    prot["family"] = [f[0] for f in fam]
    prot["color"] = [f[1] for f in fam]
    counts = prot["family"].value_counts()
    print("Family counts:")
    print(counts)

    fig = plt.figure(figsize=(7.2, 3.3))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30, figure=fig)

    # ---------------- LEFT: drug UMAP hexbin density + anchor overlay -----------
    ax1 = fig.add_subplot(gs[0, 0])
    hb = ax1.hexbin(drug["umap_x"], drug["umap_y"], gridsize=80,
                    cmap="Blues", mincnt=1, linewidths=0)
    ax1.set_xlabel("UMAP 1", fontsize=7.5, color=BLACK)
    ax1.set_ylabel("UMAP 2", fontsize=7.5, color=BLACK)
    ax1.tick_params(axis="both", labelsize=7, colors=BLACK)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color(BLACK)
    ax1.spines["bottom"].set_color(BLACK)
    ax1.set_title(f"Drug space  (n = {len(drug):,})",
                  fontsize=7.5, color=BLACK, loc="left", pad=4)
    # Colorbar
    cb = fig.colorbar(hb, ax=ax1, fraction=0.04, pad=0.02)
    cb.set_label("Drugs per hex", fontsize=6.5, color=BLACK)
    cb.ax.tick_params(labelsize=6.5, colors=BLACK)
    cb.outline.set_visible(False)

    # Anchor overlay — paper-relevant drugs from Figs 3, 4, 5
    # Colored dots by source + auto-positioned text labels via adjustText
    from adjustText import adjust_text
    anchor_tsv = DATA_DIR / "anchor_umap.tsv"
    if anchor_tsv.exists():
        anc = pd.read_csv(anchor_tsv, sep="\t").reset_index(drop=True)
        src_colors = {"Klaeger TKI": BIIE.BLUE, "Repurposing": BIIE.GREEN,
                      "Generative": BIIE.ALERT}
        for _, r in anc.iterrows():
            color = src_colors.get(r["source"], BLACK)
            ax1.scatter(r["umap_x"], r["umap_y"], s=22,
                        c=color, edgecolors="white", linewidths=0.7, zorder=10)
        texts = []
        for _, r in anc.iterrows():
            t = ax1.text(r["umap_x"], r["umap_y"], r["drug_name"],
                         fontsize=6, color=BLACK, zorder=11)
            texts.append(t)
        adjust_text(texts, ax=ax1,
                    arrowprops=dict(arrowstyle="-", color=BIIE.GREY_DARK,
                                    lw=0.4, alpha=0.6),
                    expand=(1.2, 1.4), force_text=(0.4, 0.6),
                    only_move={"text": "xy"})

        # Source-color legend (top-left of drug panel)
        for i, (src, col) in enumerate(src_colors.items()):
            xi = 0.02
            yi = 0.97 - i * 0.06
            ax1.scatter([xi], [yi], s=14, c=col, edgecolors="white",
                        linewidths=0.5, transform=ax1.transAxes,
                        clip_on=False, zorder=12)
            ax1.text(xi + 0.045, yi, src, fontsize=5.8, color=BLACK,
                     va="center", ha="left", transform=ax1.transAxes,
                     zorder=12)

    # Panel label
    ax1.text(-0.10, 1.10, "c", transform=ax1.transAxes,
             fontsize=10, weight="bold", color=BLACK, ha="left", va="top")

    # ---------------- RIGHT: protein UMAP scatter colored by family ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    # Plot "Other" first (background)
    other = prot[prot["family"] == "Other"]
    ax2.scatter(other["umap_x"], other["umap_y"], s=3,
                c=OTHER_COLOR, alpha=0.45, edgecolors="none", rasterized=True)
    # Plot families on top
    family_order = [f[0] for f in FAMILIES]
    for fam_name in family_order:
        sub = prot[prot["family"] == fam_name]
        if len(sub) == 0:
            continue
        color = sub.iloc[0]["color"]
        ax2.scatter(sub["umap_x"], sub["umap_y"], s=5,
                    c=color, alpha=0.85, edgecolors="none",
                    label=f"{fam_name} ({len(sub):,})", rasterized=True)
    ax2.set_xlabel("UMAP 1", fontsize=7.5, color=BLACK)
    ax2.set_ylabel("UMAP 2", fontsize=7.5, color=BLACK)
    ax2.tick_params(axis="both", labelsize=7, colors=BLACK)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(BLACK)
    ax2.spines["bottom"].set_color(BLACK)
    ax2.set_title(f"Protein space  (n = {len(prot):,})",
                  fontsize=7.5, color=BLACK, loc="left", pad=4)
    # Family legend (top-left of protein panel, inside axes)
    leg = ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),
                     fontsize=5.8, frameon=False, labelcolor=BLACK,
                     handletextpad=0.3, borderaxespad=0.2,
                     labelspacing=0.3, markerscale=1.2)

    out = REPO_ROOT / "dtSFM-Figures/fig1_panelC_coverage"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    build()
