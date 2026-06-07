#!/usr/bin/env python3
"""build_fig4_panelB_zoom.py — dtSFM v3 Fig 4 panel b: per-target class composition.

Replaces the recovery/novel bar chart with three per-target zoomed scatters
(top library percentile × AF3 iPTM), colored by leakage class. The class
*composition* at the top of each screen is the diagnostic of training coverage:
  CD73  -> top hits ~all class A (memorized; training-saturated)
  STING1-> top hits ~all class B (novel predictions; but at the AF3 floor)
  NLRP3 -> mixed A/B
Negative controls (grey) mark the AF3 drug-like iPTM floor in each panel.

Source: audit/dtsfm/repurposing/F42_results.tsv
Output: dtSFM-Figures/fig4_panelB_zoom_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
import matplotlib.lines as ml           # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

LIBRARY = 522_776
TARGETS = ["NLRP3", "CD73", "STING1"]
CLASS_COLOR = {"A": BIIE.BLUE_PURPLE, "B": BIIE.TEAL, "C": BIIE.MAGENTA}


def main():
    apply_style()
    df = pd.read_csv(ROOT / "audit/dtsfm/repurposing/F42_results.tsv", sep="\t")
    df = df.dropna(subset=["cosine", "iptm"]).copy()
    df["is_neg"] = df["stratum"] == "negative_control"
    df["cls"] = df["expected_class"].fillna("")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for ax, tgt in zip(axes, TARGETS):
        sub = df[df["target"] == tgt].copy()
        cmin, cmax = sub["cosine"].min(), sub["cosine"].max()
        sub["xn"] = 100.0 * (sub["cosine"] - cmin) / (cmax - cmin)   # 0-100% per target
        ax.axhline(0.7, ls="--", lw=0.6, color=BIIE.GREY_DARK, zorder=1)
        nn = sub[~sub["is_neg"]]
        for cls, col in CLASS_COLOR.items():
            m = nn[nn["cls"] == cls]
            if len(m):
                ax.scatter(m["xn"], m["iptm"], s=22, facecolors=col,
                           edgecolors="white", linewidths=0.4, alpha=0.9, zorder=3)
        m = nn[nn["cls"] == ""]
        if len(m):
            ax.scatter(m["xn"], m["iptm"], s=18, facecolors="white",
                       edgecolors=BIIE.GREY_MID, linewidths=0.7, zorder=2)
        neg = sub[sub["is_neg"]]
        ax.scatter(neg["xn"], neg["iptm"], s=22, facecolors="white",
                   edgecolors=BIIE.GREY_DARK, linewidths=0.8, marker="o", zorder=2)
        ax.set_ylim(0.35, 1.0)
        ax.set_xlim(-5, 105)
        ax.set_title(tgt, fontsize=8.5, fontweight="bold", loc="center")
        ax.set_xlabel("dtSFM cosine (norm. %)", fontsize=6.5)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("AF3 iPTM", fontsize=7.5)
    axes[0].text(-0.30, 1.06, "b", transform=axes[0].transAxes, fontweight="bold",
                 fontsize=9, ha="left", va="top")

    h = [ml.Line2D([], [], marker="o", linestyle="none", markersize=5,
                   markerfacecolor=CLASS_COLOR[c], markeredgecolor="white",
                   label=lab)
         for c, lab in [("A", "A (pair in training)"), ("B", "B (drug seen, novel pairing)"),
                        ("C", "C (drug OOD, novel chemistry)")]]
    h.append(ml.Line2D([], [], marker="o", linestyle="none", markersize=5,
                       markerfacecolor="white", markeredgecolor=BIIE.GREY_DARK,
                       label="negative control"))
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=5.8, frameon=False,
               handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.26, wspace=0.10)
    paths = save_figure(fig, "fig4_panelB_zoom_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    # diagnostic: class composition of top-50-cosine per target
    for tgt in TARGETS:
        t50 = df[(df["target"] == tgt) & (df["stratum"] == "top50_cosine")]
        print(f"  {tgt} top50_cosine class mix:", t50["cls"].value_counts().to_dict())
    plt.close(fig)


if __name__ == "__main__":
    main()
