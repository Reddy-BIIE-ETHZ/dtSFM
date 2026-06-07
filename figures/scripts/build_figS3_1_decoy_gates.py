#!/usr/bin/env python3
"""build_figS3_1_decoy_gates.py — Supp Fig S3.1: decoy vs design gate-pass bars.

Mirrors the P2_figS2-1 style: side-by-side bars showing % of designs vs decoys
passing the binder gate (iPTM>=0.7 AND PAE<=5) and the anchor-grade gate
(iPTM>=0.9 AND PAE<=1.667). Demonstrates that the anchor-grade gate effectively
excludes negative-control decoy drugs.

P1 numbers (locked):
  designs (n=1200):  binder 1146 (95.5%)   anchor-grade 850 (70.8%)
  decoys  (n=80):    binder   28 (35.0%)   anchor-grade   2  (2.5%)

Source: audit/dtsfm/decoder_af3/F5_results.tsv (decoder strata) +
        audit/dtsfm/decoder_af3/F5_negative_floor.tsv (decoys).
Output: dtSFM-Figures/figS3_1_decoy_gates_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

DECODER = {"top_cosine", "scaffold_div", "mid_cosine"}
IPTM_BIND, PAE_BIND = 0.7, 5.0
IPTM_AG, PAE_AG = 0.9, 1.667


def passes(r, iptm_min, pae_max):
    try:
        return float(r["iptm"]) >= iptm_min and float(r["interface_pae_min"]) <= pae_max
    except (TypeError, ValueError, KeyError):
        return False


def count(path, filt=None):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    if filt:
        rows = [r for r in rows if filt(r)]
    n_bind = sum(1 for r in rows if passes(r, IPTM_BIND, PAE_BIND))
    n_ag = sum(1 for r in rows if passes(r, IPTM_AG, PAE_AG))
    return len(rows), n_bind, n_ag


def main():
    apply_style()
    n_d, d_bind, d_ag = count(ROOT / "audit/dtsfm/decoder_af3/F5_results.tsv",
                              filt=lambda r: r["stratum"] in DECODER)
    n_n, n_bind, n_ag = count(ROOT / "audit/dtsfm/decoder_af3/F5_negative_floor.tsv")
    print(f"designs n={n_d}: binder {d_bind} ({100*d_bind/n_d:.1f}%) "
          f"anchor-grade {d_ag} ({100*d_ag/n_d:.1f}%)")
    print(f"decoys  n={n_n}: binder {n_bind} ({100*n_bind/n_n:.1f}%) "
          f"anchor-grade {n_ag} ({100*n_ag/n_n:.1f}%)")

    pct = [
        [100 * d_bind / n_d, 100 * d_ag / n_d],   # designs
        [100 * n_bind / n_n, 100 * n_ag / n_n],   # decoys
    ]
    raw = [(d_bind, d_ag, n_d), (n_bind, n_ag, n_n)]
    labels = [f"designs\n(n = {n_d:,})", f"decoys\n(n = {n_n})"]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    x = np.arange(2)
    w = 0.36
    bar_b = ax.bar(x - w/2, [pct[0][0], pct[1][0]], width=w,
                   color=BIIE.GREY_LIGHT, edgecolor=BIIE.GREY_DARK, linewidth=0.6,
                   label="binder gate")
    bar_a = ax.bar(x + w/2, [pct[0][1], pct[1][1]], width=w,
                   color=BIIE.ALERT, edgecolor=BIIE.BLACK, linewidth=0.6,
                   label="anchor-grade gate")

    # Value labels on top of each bar
    for bars, vals, rcol in [(bar_b, [pct[0][0], pct[1][0]], 0),
                              (bar_a, [pct[0][1], pct[1][1]], 1)]:
        for i, (b, v) in enumerate(zip(bars, vals)):
            n_pass = raw[i][rcol]
            ax.text(b.get_x() + b.get_width() / 2, v + 1.8,
                    f"{v:.1f}%\n({n_pass}/{raw[i][2]})",
                    ha="center", va="bottom", fontsize=6.4,
                    color=BIIE.GREY_DARK, linespacing=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% passing gate", fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.4, frameon=False,
              handletextpad=0.5, handlelength=1.0)

    fig.tight_layout()
    paths = save_figure(fig, "figS3_1_decoy_gates_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
