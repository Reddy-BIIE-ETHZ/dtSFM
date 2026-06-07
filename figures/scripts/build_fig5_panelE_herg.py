#!/usr/bin/env python3
"""build_fig5_panelF_herg.py — dtSFM Fig 5 panel f: AF3's hERG blind spot.

The honest "limit as diagnosis" panel: AF3 vs wet-lab for the FLT3 series.

Binary dual-gate call (binds = AF3 iPTM >= 0.7 AND PAE <= 5), kinase-screen style.
Columns are ordered so the three off-targets with KNOWN wet-lab binding come
first (KIT, PDGFRB via Klaeger 2017 kinobeads; KCNH2/hERG via clinical QT
liability), then the four AF3-only off-targets (PDGFRA, KDR, FLT1, FLT4).

Each approved anchor gets two rows: its AF3 prediction and a wet-lab row
(measured cells filled red = binds; unmeasured cells left blank). The dtSFM
FLT3 designs are novel, so they have AF3 rows only.

Read-out: AF3 reproduces the wet-lab kinase binding (KIT, PDGFRB agree) but
the KCNH2/hERG column exposes the blind spot — wet-lab says BINDS (red) while
AF3 says no-bind (blue) for the same approved drugs. hERG-sparing of the dtSFM
designs is therefore wet-lab-arbitrated, not AF3-certified.

Source: F5_3_cofold_results.tsv + F5_3_cofold_results_anchors.tsv (AF3);
        Klaeger 2017 + clinical QT liability (wet-lab anchors).
Output: dtSFM-Figures/fig5_panelE_herg_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import csv                              # noqa: E402

import numpy as np                      # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
import matplotlib.lines as ml           # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle      # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

LEADS = ROOT / "audit/dtsfm/decoder_af3/F5_3_cofold_results.tsv"
ANCH = ROOT / "audit/dtsfm/decoder_af3/F5_3_cofold_results_anchors.tsv"
PAE_THRESH, IPTM_THRESH = 5.0, 0.7

# wet-lab-measured off-targets first (KIT/PDGFRB Klaeger, KCNH2 clinical), then AF3-only
COLS = ["KIT", "PDGFRB", "KCNH2", "PDGFRA", "KDR", "FLT1", "FLT4"]
HERG_IDX = COLS.index("KCNH2")
N_MEASURED = 3
WETLAB_BINDS = {"KIT", "PDGFRB", "KCNH2"}   # all known to bind (red)

# (sample, display, kind)  kind in {lab, spacer, af3_anchor, af3_lead}
# wet-lab block first, a blank spacer row, then the AF3 block.
ROWS = [
    ("quizartinib", "quizartinib", "lab"),
    ("midostaurin", "midostaurin", "lab"),
    (None, "", "spacer"),
    ("quizartinib", "quizartinib", "af3_anchor"),
    ("midostaurin", "midostaurin", "af3_anchor"),
    ("dec_FLT3_0469", "d0469", "af3_lead"),
    ("dec_FLT3_0480", "d0480", "af3_lead"),
    ("dec_FLT3_0484", "d0484", "af3_lead"),
    ("dec_FLT3_0489", "d0489", "af3_lead"),
    ("dec_FLT3_0502", "d0502", "af3_lead"),
]
SPACER_IDX = 2
BRIGHT_GREEN = "#22C55E"


def main():
    apply_style()
    leads = {(r["parent_sample"], r["off_target"]): r
             for r in csv.DictReader(open(LEADS), delimiter="\t")}
    anch = {(r["parent_sample"], r["off_target"]): r
            for r in csv.DictReader(open(ANCH), delimiter="\t")}

    M = np.full((len(ROWS), len(COLS)), np.nan)
    for i, (s, _, kind) in enumerate(ROWS):
        if kind == "spacer":
            continue
        for j, c in enumerate(COLS):
            if kind == "lab":
                if c in WETLAB_BINDS:
                    M[i, j] = 1.0            # wet-lab binds; unmeasured -> blank
                continue
            src = anch if kind == "af3_anchor" else leads
            r = src.get((s, c))
            if r:
                pae, iptm = float(r["interface_pae_min"]), float(r["iptm"])
                M[i, j] = 1.0 if (iptm >= IPTM_THRESH and pae <= PAE_THRESH) else 0.0

    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    cmap = ListedColormap([BIIE.BLUE, BIIE.ALERT])     # 0 = no-bind, 1 = binds
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Clean light-gray outline on every cell (incl. blanks); skip the spacer gap.
    for i, (_, _, kind) in enumerate(ROWS):
        if kind == "spacer":
            continue
        for j in range(len(COLS)):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=BIIE.GREY_LIGHT, linewidth=0.6,
                                   zorder=3))
    for s in ax.spines.values():           # no dark black axis lines
        s.set_visible(False)

    ax.set_xticks(range(len(COLS)))
    xlabs = [("hERG" if c == "KCNH2" else c) for c in COLS]
    ax.set_xticklabels(xlabs, fontsize=6.6, rotation=45, ha="right")
    ax.get_xticklabels()[HERG_IDX].set_color(BIIE.ALERT)
    ax.get_xticklabels()[HERG_IDX].set_fontweight("bold")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([d for _, d, _ in ROWS], fontsize=6.6)  # all black, regular
    ax.tick_params(length=0)

    # block labels on the far left (in the margin)
    ax.text(-0.34, 0.5, "wet-lab", rotation=90, ha="center", va="center",
            fontsize=6.6, weight="bold", color=BIIE.GREY_DARK,
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(-0.34, (SPACER_IDX + 1 + len(ROWS) - 1) / 2.0, "AF3", rotation=90,
            ha="center", va="center", fontsize=6.6, weight="bold",
            color=BIIE.GREY_DARK, transform=ax.get_yaxis_transform(), clip_on=False)

    # measured | AF3-only split conveyed by labels (no dark divider line)
    ax.text((N_MEASURED - 1) / 2.0, -0.75, "wet-lab-measured", ha="center",
            va="bottom", fontsize=6.0, color=BIIE.GREY_DARK, style="italic")
    ax.text(N_MEASURED + (len(COLS) - N_MEASURED - 1) / 2.0, -0.75,
            "AF3-only off-targets", ha="center", va="bottom", fontsize=6.0,
            color=BIIE.GREY_DARK, style="italic")
    h = [
        ml.Line2D([], [], marker="s", linestyle="none", markersize=8,
                  markerfacecolor=BIIE.ALERT, markeredgecolor="none",
                  label="binds (AF3 iPTM ≥ 0.7 · PAE ≤ 5, or wet-lab)"),
        ml.Line2D([], [], marker="s", linestyle="none", markersize=8,
                  markerfacecolor=BIIE.BLUE, markeredgecolor="none",
                  label="no-bind (AF3)"),
        ml.Line2D([], [], marker="s", linestyle="none", markersize=8,
                  markerfacecolor="white", markeredgecolor=BIIE.GREY_LIGHT,
                  label="not measured (wet-lab)"),
    ]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=5.6, frameon=False, handletextpad=0.4, labelspacing=0.5)

    ax.text(-0.10, 1.14, "e", transform=ax.transAxes, fontweight="bold",
            fontsize=9, ha="left", va="bottom")
    fig.text(0.5, 0.015,
             "hERG column: wet-lab says BINDS (red) but AF3 says no-bind (blue) — "
             "the blind spot. AF3 reproduces the measured kinase binding (KIT, PDGFRB).",
             ha="center", va="bottom", fontsize=5.6, color=BIIE.GREY_DARK,
             style="italic")

    fig.subplots_adjust(left=0.28, right=0.74, top=0.82, bottom=0.24)
    paths = save_figure(fig, "fig5_panelE_herg_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
