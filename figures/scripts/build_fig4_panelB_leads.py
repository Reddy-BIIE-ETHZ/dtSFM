#!/usr/bin/env python3
"""build_fig4_panelB_leads.py — dtSFM v3 Fig 4 panel b: recovery vs novel leads.

For each repurposing target, the AF3 structural-confirmation rate (combined
criterion: iPTM >= 0.7 AND interface PAE <= 5 Å) of top-ranked 522K-library
screen compounds, split by leakage class:
  - MEMORIZED (class A): (drug, target) pair already in training -> recovery /
    pipeline validation, NOT new leads.
  - NOVEL (class B/C): genuine new drug-target predictions -> candidate new leads.
A per-target NEGATIVE-CONTROL floor line marks the AF3 drug-like iPTM floor.

Diagnostic story (first-of-kind dtSFM; limits = roadmap):
  CD73 is training-saturated -> top hits are memorized, ~99% confirm (validation,
  0 novel tested). STING1's novel hits sit at/under the AF3 floor (neg-ctrl 60%)
  -> floor-limited. NLRP3 is the cleanest novel set (40% vs 0% neg). Where two
  INDEPENDENT models agree, trust it.

Source: audit/dtsfm/repurposing/F42_results.tsv
Output: dtSFM-Figures/fig4_panelB_leads_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                      # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from matplotlib.patches import Patch    # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

A = ROOT / "audit/dtsfm/repurposing"
TARGETS = ["NLRP3", "CD73", "STING1"]
SCREEN = {"top50_cosine", "scaffold_div"}


def fnum(x):
    x = (x or "").strip()
    try:
        return float(x)
    except ValueError:
        return None


def main():
    apply_style()
    rows = list(csv.DictReader(open(A / "F42_results.tsv"), delimiter="\t"))

    # per target: memorized(A) / novel(BC) over SCREEN strata; neg-control
    stat = {t: {"A": [0, 0], "BC": [0, 0], "neg": [0, 0]} for t in TARGETS}
    for r in rows:
        t = r["target"]
        if t not in stat:
            continue
        ip, pae = fnum(r["iptm"]), fnum(r["interface_pae_min"])
        if ip is None or pae is None:
            continue
        passed = ip >= 0.7 and pae <= 5.0
        strat = r["stratum"]
        cls = r["expected_class"].strip().upper()
        if strat == "negative_control":
            bucket = "neg"
        elif strat in SCREEN:
            bucket = "A" if cls == "A" else "BC"
        else:
            continue
        stat[t][bucket][1] += 1
        stat[t][bucket][0] += int(passed)

    def rate(pt):
        return 100 * pt[0] / pt[1] if pt[1] else None

    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    x = np.arange(len(TARGETS))
    w = 0.36
    for i, t in enumerate(TARGETS):
        a, bc, neg = stat[t]["A"], stat[t]["BC"], stat[t]["neg"]
        # memorized (recovery) bar
        ra = rate(a)
        if ra is not None:
            ax.bar(x[i] - w / 2, ra, w, color=BIIE.BLUE, edgecolor=BIIE.BLACK,
                   linewidth=0.7, zorder=3)
            ax.text(x[i] - w / 2, ra + 1.5, f"{a[0]}/{a[1]}", ha="center",
                    va="bottom", fontsize=5.6, color=BIIE.BLACK)
        # novel (new-leads) bar
        rbc = rate(bc)
        if rbc is not None:
            ax.bar(x[i] + w / 2, rbc, w, color=BIIE.GREEN, edgecolor=BIIE.BLACK,
                   linewidth=0.7, zorder=3)
            ax.text(x[i] + w / 2, rbc + 1.5, f"{bc[0]}/{bc[1]}", ha="center",
                    va="bottom", fontsize=5.6, color=BIIE.BLACK)
        else:
            ax.text(x[i] + w / 2, 3, "no novel\nhits tested\n(saturated)",
                    ha="center", va="bottom", fontsize=5.0, style="italic",
                    color=BIIE.GREY_DARK)
        # negative-control floor line across the group
        rneg = rate(neg)
        if rneg is not None:
            ax.plot([x[i] - w - 0.04, x[i] + w + 0.04], [rneg, rneg],
                    color=BIIE.ALERT, linewidth=1.1, linestyle=(0, (3, 2)), zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS, fontsize=7.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("AF3-confirmed  (iPTM ≥ 0.7 & PAE ≤ 5 Å)  %")
    ax.set_xlabel("repurposing target", fontweight="bold", labelpad=8)
    ax.text(0.0, 1.04, "b", transform=ax.transAxes, fontweight="bold",
            fontsize=9, ha="left", va="bottom")

    handles = [
        Patch(facecolor=BIIE.BLUE, edgecolor=BIIE.BLACK, label="memorized (class A) — recovery"),
        Patch(facecolor=BIIE.GREEN, edgecolor=BIIE.BLACK, label="novel (class B/C) — new leads"),
        plt.Line2D([], [], color=BIIE.ALERT, linewidth=1.1, linestyle=(0, (3, 2)),
                   label="negative-control floor"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=1, fontsize=5.8, frameon=False, handlelength=1.4,
              handletextpad=0.5, labelspacing=0.3)

    fig.tight_layout()
    paths = save_figure(fig, "fig4_panelB_leads_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    for t in TARGETS:
        print(f"  {t}: A={stat[t]['A']} BC={stat[t]['BC']} neg={stat[t]['neg']}")
    plt.close(fig)


if __name__ == "__main__":
    main()
