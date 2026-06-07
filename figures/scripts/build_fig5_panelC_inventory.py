#!/usr/bin/env python3
"""build_fig5_panelC_inventory.py — dtSFM Fig 5 panel c: AF3 binder-gate (FACS-style).

Main Paper-1 message, worked backwards: the decoder generates a LARGE pool of
candidates, many of them as structurally confident (AF3) as the approved drug
itself. The confident pool for downstream optimization. No selectivity / wet-lab
claims.

FACS-style pooled dot plot (all 16 targets together):

  x = AF3 interface PAE (A), REVERSED so 10 sits at the left and 0 at the
      right — tighter interfaces appear on the right, best variants top-right.
  y = AF3 iPTM   (interface confidence)
  binder gate  = top-right (iPTM >= 0.7 AND PAE <= 5)                       [grey dashed]
  anchor-grade = stringent gate (iPTM >= 0.9 AND PAE <= 1.67)               [red box]
                 — all 15 clinical anchors fall inside it, so a variant here is
                 as AF3-confident as the approved drug.

  blue dot = generated candidate (dtSFM design)
  gold dot = approved anchor (wet-lab-known; validates the gate)
  grey open = negative-control decoy drug (should fall outside)

Source: F5_results.tsv + F5_anchor_drift.tsv + F5_negative_floor.tsv
Output: dtSFM-Figures/fig5_panelC_inventory_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import csv                              # noqa: E402
from collections import defaultdict    # noqa: E402

import numpy as np                      # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
import matplotlib.lines as ml           # noqa: E402
from matplotlib.patches import Rectangle    # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

RES = ROOT / "audit/dtsfm/decoder_af3/F5_results.tsv"
DRIFT = ROOT / "audit/dtsfm/decoder_af3/F5_anchor_drift.tsv"
NEG = ROOT / "audit/dtsfm/decoder_af3/F5_negative_floor.tsv"
DECODER = {"top_cosine", "scaffold_div", "mid_cosine"}
IPTM_BIND, PAE_BIND = 0.7, 5.0           # broad binder gate
IPTM_STR, PAE_STR = 0.9, 1.667           # stringent anchor-grade gate (1/PAE >= 0.6)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    apply_style()
    variants, per_tgt = [], defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(RES), delimiter="\t"):
        if r["stratum"] not in DECODER:
            continue
        i, p = f(r["iptm"]), f(r["interface_pae_min"])
        if i is None or p is None:
            continue
        strict = i >= IPTM_STR and p <= PAE_STR
        variants.append((i, p, strict))                     # (iptm, pae, strict)
        per_tgt[r["target"]][1] += 1
        per_tgt[r["target"]][0] += int(strict)
    anchors = [(f(r["iptm"]), f(r["interface_pae_min"]))
               for r in csv.DictReader(open(DRIFT), delimiter="\t")
               if f(r["iptm"]) and f(r["interface_pae_min"])]
    decoys = [(f(r["iptm"]), f(r["interface_pae_min"]))
              for r in csv.DictReader(open(NEG), delimiter="\t")
              if f(r["iptm"]) and f(r["interface_pae_min"])]

    n_str = sum(s for *_, s in variants)
    n_tot = len(variants)

    fig = plt.figure(figsize=(6.4, 4.3))
    ax = fig.add_axes([0.10, 0.20, 0.50, 0.70])

    # Axes: x = PAE (Å) reversed (10 left → 0 right), y = iPTM. Best top-right.
    XMAX = 10.0
    ax.axhline(IPTM_BIND, ls="--", lw=0.7, color=BIIE.GREY_MID, zorder=1)
    ax.axvline(PAE_BIND, ls="--", lw=0.7, color=BIIE.GREY_MID, zorder=1)

    var = np.array([(i, p) for i, p, _ in variants])   # (iptm, pae)
    dec = np.array(decoys)                              # (iptm, pae)
    anc = np.array(anchors)                             # (iptm, pae)
    # x = pae, y = iptm
    ax.scatter(dec[:, 1], dec[:, 0], s=16, facecolors="none",
               edgecolors=BIIE.GREY_MID, linewidths=0.6, zorder=2)
    ax.scatter(var[:, 1], var[:, 0], s=9, color=BIIE.BLUE, alpha=0.38,
               edgecolors="none", zorder=3)
    ax.scatter(anc[:, 1], anc[:, 0], s=10, color=BIIE.GOLD,
               edgecolors=BIIE.BLACK, linewidths=0.3, zorder=4)

    # Stringent anchor-grade gate (dotted red): PAE <= PAE_STR AND iPTM >= IPTM_STR.
    # In (PAE, iPTM) with reversed-x axis: x range [0, PAE_STR], y range [IPTM_STR, 1].
    ax.add_patch(Rectangle((0.0, IPTM_STR), PAE_STR, 1.0 - IPTM_STR,
                           fill=False, edgecolor=BIIE.ALERT, linewidth=1.0,
                           linestyle=":", zorder=5))

    ax.set_xlabel("AF3 interface PAE  (Å)", labelpad=4)
    ax.set_ylabel("AF3 iPTM (interface confidence)", labelpad=4)
    ax.set_xlim(XMAX, 0.0)        # reversed: 10 on left, 0 on right
    ax.set_xticks([10, 8, 6, 4, 2, 0])
    ax.set_ylim(0.4, 1.0)
    ax.text(-0.02, 1.04, "d", transform=ax.transAxes, fontweight="bold",
            fontsize=9, ha="left", va="bottom")

    h = [
        ml.Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                  markerfacecolor=BIIE.BLUE, markeredgecolor="none", alpha=0.6,
                  label=f"generated candidate (n = {n_tot:,})"),
        ml.Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                  markerfacecolor=BIIE.GOLD, markeredgecolor=BIIE.BLACK,
                  label="approved anchor (gate validation)"),
        ml.Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                  markerfacecolor="none", markeredgecolor=BIIE.GREY_MID,
                  label=f"negative-control decoy (n = {len(decoys)})"),
    ]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=1, fontsize=5.8, frameon=False, handletextpad=0.4,
              labelspacing=0.3)

    # per-target anchor-grade %: clean horizontal stacked bars (right)
    ax2 = fig.add_axes([0.74, 0.20, 0.18, 0.70])
    order = sorted(per_tgt, key=lambda t: per_tgt[t][0] / per_tgt[t][1])  # asc -> best on top
    yb = np.arange(len(order))
    for k, t in enumerate(order):
        s, tot = per_tgt[t]
        pct = 100.0 * s / tot
        ax2.barh(k, pct, color=BIIE.BLUE, height=0.72, zorder=2)
        ax2.barh(k, 100 - pct, left=pct, color=BIIE.GREY_LIGHT, height=0.72, zorder=1)
        ax2.text(102, k, f"{s}/{tot}", ha="left", va="center", fontsize=5.2,
                 color=BIIE.GREY_DARK)
    ax2.set_yticks(yb)
    ax2.set_yticklabels(order, fontsize=5.8)
    ax2.set_ylim(-0.6, len(order) - 0.4)
    ax2.set_xlim(0, 100)
    ax2.set_xticks([0, 50, 100])
    ax2.tick_params(labelsize=5.6, length=2)
    ax2.set_xlabel("anchor-grade (%)", fontsize=6.4, labelpad=3)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.tick_params(axis="y", length=0)

    paths = save_figure(fig, "fig5_panelC_inventory_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    print(f"  anchor-grade {n_str}/{n_tot}")
    plt.close(fig)


if __name__ == "__main__":
    main()
