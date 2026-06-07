#!/usr/bin/env python3
"""build_fig3_safety.py — dtSFM v3 paper Figure 3.

Two-panel proteome-scale safety screening result on 10 clinical TKIs:
  (a) Cohort-level Class B retrieval rate at K={10, 50, 100, 500} of 4,910 genes
      — strict drug-OOD pairs with target seen in training (n=27 across 5 drugs).
  (b) Per-drug Class B top-50 hit rate for the 5 contributing drugs, with
      cohort-mean line at 70 %.

Source data:
  audit/dtsfm/safety_screen_summary.json      (Class B headline + per-drug counts)
  audit/dtsfm/safety_panel_pair_leakage.tsv   (pair-level leakage classification)

Output: dtSFM-Figures/fig3_safety_screening.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
from calm.figures.biie_style import (  # noqa: E402
    apply_style, BIIE, save_figure,
)


# ---- Load data ----
SUMMARY_JSON = ROOT / "audit/dtsfm/safety_screen_summary.json"


def main():
    apply_style()

    summary = json.loads(SUMMARY_JSON.read_text())
    headline = summary["headline_class_b"]
    per_drug = summary["per_drug"]

    # ---- Panel (a) data: cohort Class B retrieval at K = 10, 50, 100, 500 ----
    K_LABELS = ["top-10", "top-50", "top-100", "top-500"]
    K_PCTS = [
        100 * headline["top10_pct"],
        100 * headline["top50_pct"],
        100 * headline["top100_pct"],
        100 * headline["top500_pct"],
    ]
    N_TOTAL = headline["n_pairs_total"]      # 27

    # ---- Panel (b) data: per-drug Class B top-50 hit rate ----
    contributing_drugs = headline["drugs"]   # 5-drug Class B cohort
    drug_b_stats = []
    for d in contributing_drugs:
        b = per_drug[d]["by_class"]["B"]
        n_total = b["n_total"]
        top50 = b["top50"]
        pct = 100 * top50 / n_total if n_total else float("nan")
        drug_b_stats.append((d, pct, n_total))
    # Sort by n_total desc (most-data first), then by pct desc
    drug_b_stats.sort(key=lambda r: (-r[2], -r[1]))
    drug_names  = [r[0] for r in drug_b_stats]
    drug_pcts   = [r[1] for r in drug_b_stats]
    drug_ns     = [r[2] for r in drug_b_stats]
    cohort_mean_pct = 100 * headline["top50_pct"]   # 70.4

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Panel (a) — cohort headline
    ax = axes[0]
    x = np.arange(len(K_LABELS))
    bars = ax.bar(x, K_PCTS, width=0.55,
                  color=BIIE.ID_FILL, edgecolor=BIIE.BLACK, linewidth=0.8)
    for bar, val in zip(bars, K_PCTS):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{val:.0f} %", ha="center", va="bottom",
                fontsize=5.0, weight="bold", color=BIIE.BLACK)
    ax.set_xticks(x)
    ax.set_xticklabels(K_LABELS)
    ax.set_xlim(-0.6, len(K_LABELS) - 0.4)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Class B pairs retrieved (%)")
    ax.set_xlabel(f"Retrieval threshold (n = {N_TOTAL} pairs / 4 910 genes)",
                  fontweight="bold", labelpad=10)
    ax.set_title("a", loc="left", fontweight="bold")

    # Panel (b) — per-drug Class B top-50 hit rate
    ax = axes[1]
    x = np.arange(len(drug_names))
    bars = ax.bar(x, drug_pcts, width=0.6,
                  color=BIIE.ID_FILL, edgecolor=BIIE.BLACK, linewidth=0.8)
    for bar, val in zip(bars, drug_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{val:.0f} %", ha="center", va="bottom",
                fontsize=5.0, weight="bold", color=BIIE.BLACK)
    # Cohort mean dashed line
    ax.axhline(cohort_mean_pct, ls="--", lw=0.8, color=BIIE.GREY_DARK)
    ax.text(len(drug_names) - 0.5, cohort_mean_pct + 1.5,
            f"cohort mean {cohort_mean_pct:.0f} %",
            ha="right", va="bottom", fontsize=5.0, style="italic",
            color=BIIE.GREY_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(drug_names, drug_ns)],
                       fontsize=7.5)
    ax.set_xlim(-0.6, len(drug_names) - 0.4)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Class B top-50 hit rate (%)")
    ax.set_xlabel("Per-drug breakdown", fontweight="bold", labelpad=10)
    ax.set_title("b", loc="left", fontweight="bold")

    fig.tight_layout()
    paths = save_figure(fig, "fig3_safety_screening",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
