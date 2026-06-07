#!/usr/bin/env python3
"""build_fig4_panelA_screen.py — dtSFM v3 Fig 4 panel a: screen × AF3 at scale.

Single-panel complementarity scatter for the 323-pair §F.4.2 cohort sampled from
the 522,776-drug repurposing screen:

  x = v3 cosine percentile in the 522,776-drug library  (100 = top of screen)
  y = AF3 iPTM                                          (interface quality)
  color = leakage class A/B/C; grey ○ = negative control

Message: the screen surfaces a dense field of top-ranked library compounds that
also clear the AF3 binder threshold (iPTM 0.7) — candidate leads at scale. The
upper-left "v3 misses · AF3 catches" compounds (Dapansutrile, MSA-2, diABZI 3)
show the two models are complementary. Negative controls cluster at the AF3
drug-like floor (~0.7), the limit that makes the orthogonal call necessary.

Source: audit/dtsfm/repurposing/F42_results.tsv
Output: dtSFM-Figures/fig4_panelA_screen_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

LIBRARY = 522_776
CLASS_COLOR = {"A": BIIE.BLUE_PURPLE, "B": BIIE.TEAL, "C": BIIE.MAGENTA}
# Genuine "dtSFM misses · AF3 catches" = low cosine percentile, AF3 binds.
# (ADU-S100 is NOT a miss — percentile 99.83 — so it is not called out here.)
ANNOT = {
    ("NLRP3", "Dapansutrile"): (12, +0.05),
    ("STING1", "MSA-2"): (12, +0.05),
    ("STING1", "diABZI compound 3"): (14, -0.05),
}


def main():
    apply_style()
    df = pd.read_csv(ROOT / "audit/dtsfm/repurposing/F42_results.tsv", sep="\t")

    def pct(r):
        v = r.get("rank")
        v = v if pd.notna(v) else r.get("synthetic_rank")
        return 100.0 * (1.0 - v / LIBRARY) if pd.notna(v) else np.nan
    df["v3_pct"] = df.apply(pct, axis=1)
    df = df.dropna(subset=["v3_pct", "iptm"]).copy()
    df["is_neg"] = df["stratum"] == "negative_control"
    df["cls"] = df["expected_class"].fillna("")
    callouts = set(ANNOT.keys())
    df["callout"] = df.apply(lambda r: (r["target"], r["drug_name"]) in callouts, axis=1)

    nn = df[~df["is_neg"]]
    neg = df[df["is_neg"]]

    def scatter(ax, with_callouts):
        ax.axhline(0.7, ls="--", lw=0.6, color=BIIE.GREY_DARK, zorder=1)
        op = nn[~nn["callout"]]
        for cls, col in CLASS_COLOR.items():
            m = op[op["cls"] == cls]
            if len(m):
                ax.scatter(m["v3_pct"], m["iptm"], s=24, facecolors="white",
                           edgecolors=col, linewidths=0.8, alpha=0.95, zorder=2)
        m = op[op["cls"] == ""]
        if len(m):
            ax.scatter(m["v3_pct"], m["iptm"], s=24, facecolors="white",
                       edgecolors=BIIE.GREY_MID, linewidths=0.8, alpha=0.8, zorder=2)
        fp = nn[nn["callout"]]
        for cls, col in CLASS_COLOR.items():
            m = fp[fp["cls"] == cls]
            if len(m):
                ax.scatter(m["v3_pct"], m["iptm"], s=26, color=col,
                           edgecolors=BIIE.BLACK, linewidths=0.6, zorder=4)
        ax.scatter(neg["v3_pct"], neg["iptm"], s=24, facecolors="white",
                   edgecolors=BIIE.GREY_DARK, linewidths=0.8, marker="o", zorder=3)
        if with_callouts:
            for (tgt, drug), (dx, dy) in ANNOT.items():
                row = df[(df["target"] == tgt) & (df["drug_name"] == drug)]
                if not len(row):
                    continue
                r = row.iloc[0]
                ax.annotate(drug.replace("compound 3", "3").strip(),
                            xy=(r["v3_pct"], r["iptm"]),
                            xytext=(r["v3_pct"] + dx, r["iptm"] + dy),
                            ha="center", va="center", fontsize=6.4, weight="bold",
                            color=BIIE.BLACK,
                            arrowprops=dict(arrowstyle="-", color=BIIE.GREY_DARK,
                                            lw=0.5, shrinkA=2, shrinkB=4))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)

    # Left — full library percentile (the complementarity view)
    scatter(axL, with_callouts=True)
    axL.axvline(99, ls=":", lw=0.6, color=BIIE.GREY_DARK, zorder=1)
    axL.text(1.5, 0.71, "iPTM 0.7 binder threshold", fontsize=5.8,
             color=BIIE.GREY_DARK, style="italic", va="bottom")
    axL.set_xlim(0, 102)
    axL.set_ylim(0.35, 1.0)
    axL.set_xlabel("dtSFM cosine percentile in 522,776-drug library", labelpad=5)
    axL.set_ylabel("AF3 iPTM (interface quality)", labelpad=5)
    axL.text(0.0, 1.05, "a", transform=axL.transAxes, fontweight="bold",
             fontsize=9, ha="left", va="bottom")

    # Right — zoom to the top 0.01% of the library (rank <= ~52)
    n_top = int((df["v3_pct"] >= 99.99).sum())
    scatter(axR, with_callouts=False)
    axR.set_xlim(99.988, 100.002)
    axR.set_xticks([99.99, 99.995, 100.0])
    axR.set_xticklabels(["99.99", "99.995", "100"])
    axR.set_xlabel("percentile — top 0.01% zoom (rank ≤ 52)", labelpad=5)
    axR.text(0.5, 1.05, f"{n_top} pairs in the top 0.01% of the screen",
             transform=axR.transAxes, ha="center", va="bottom", fontsize=6.6,
             color=BIIE.GREY_DARK, style="italic")

    # shared legend below
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], marker="o", linestyle="none", markersize=5,
                   markerfacecolor="white", markeredgecolor=CLASS_COLOR[c],
                   label=lab) for c, lab in
         [("A", "A (pair in training)"), ("B", "B (drug seen, novel pairing)"),
          ("C", "C (drug OOD, novel chemistry)")]]
    h.append(ml.Line2D([], [], marker="o", linestyle="none", markersize=5,
                       markerfacecolor="white", markeredgecolor=BIIE.GREY_DARK,
                       label="negative control"))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=4, fontsize=5.8, frameon=False, handletextpad=0.3,
               columnspacing=1.4, labelspacing=0.3)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.20, wspace=0.08)
    paths = save_figure(fig, "fig4_panelA_screen_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
