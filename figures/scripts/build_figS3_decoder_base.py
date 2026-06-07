#!/usr/bin/env python3
"""build_figS3_decoder_base.py — Supp Fig S3: decoder base sampling performance.

Post-hoc sampling characterization of the cross-attentive SFM decoder applied
to 16 immunology/oncology targets. 4 panels:

  (a) QED distribution across the 1,200-candidate cohort. Drug-likeness.
  (b) Lipinski-Ro5 compliance rate per target (16 bars).
  (c) dtSFM encoder cosine of each reranked generated drug against its intended
      target embedding (n = 849). Decoder generates molecules the encoder
      recognises as compatible with the conditioning target.
  (d) Chemistry novelty: max ECFP4 Tanimoto similarity of each reranked
      generated drug to its nearest compound in the 522,776-compound v3
      training library, binned (n = 849).

Training-loss curves are not local; this figure characterises sampling.

Source: audit/dtsfm/decoder_af3/F5_results.tsv (panels a, b — 1,200 cohort)
        audit/dtsfm/decoder_af3/F5_2_selectivity_reranked.tsv (panels c, d — 849)
Output: dtSFM-Figures/figS3_decoder_base_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

DECODER = {"top_cosine", "scaffold_div", "mid_cosine"}
NOVELTY_BINS = [(0.0, 0.3, "0.0–0.3\n(novel)"),
                (0.3, 0.5, "0.3–0.5\n(novel scaffold)"),
                (0.5, 0.7, "0.5–0.7\n(moderate)"),
                (0.7, 0.9, "0.7–0.9\n(close analog)"),
                (0.9, 1.01, "0.9–1.0\n(near-duplicate)")]


def load_decoder():
    rows = list(csv.DictReader(open(ROOT / "audit/dtsfm/decoder_af3/F5_results.tsv"),
                               delimiter="\t"))
    return [r for r in rows if r["stratum"] in DECODER]


def load_reranked():
    return list(csv.DictReader(open(ROOT / "audit/dtsfm/decoder_af3/F5_2_selectivity_reranked.tsv"),
                               delimiter="\t"))


def main():
    apply_style()
    dec = load_decoder()
    rer = load_reranked()
    print(f"decoder cohort n={len(dec)}; reranked n={len(rer)}")

    fig = plt.figure(figsize=(8.4, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30,
                          left=0.08, right=0.97, top=0.94, bottom=0.10)

    # --- (a) QED distribution across the 1,200 cohort ---
    ax = fig.add_subplot(gs[0, 0])
    qed = [float(r["qed"]) for r in dec if r["qed"]]
    med_qed = np.median(qed)
    ax.hist(qed, bins=30, color=BIIE.BLUE, alpha=0.85,
            edgecolor=BIIE.GREY_DARK, linewidth=0.3)
    ax.axvline(med_qed, ls="--", lw=1.0, color=BIIE.ALERT)
    ax.set_xlabel("QED (drug-likeness)", fontsize=7.6)
    ax.set_ylabel("count", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(-0.18, 1.06, "a", transform=ax.transAxes, fontweight="bold",
            fontsize=10, ha="left", va="top")
    ax.set_title(f"QED distribution  (n = {len(qed):,} generated)",
                 fontsize=7.4, color=BIIE.GREY_DARK, pad=4)

    # --- (b) Lipinski Ro5 compliance per target ---
    ax = fig.add_subplot(gs[0, 1])
    per_t = defaultdict(lambda: [0, 0])  # (compliant, total)
    for r in dec:
        per_t[r["target"]][1] += 1
        if r["ro5_compliant"] == "True":
            per_t[r["target"]][0] += 1
    order = sorted(per_t, key=lambda t: per_t[t][0] / per_t[t][1], reverse=True)
    y = np.arange(len(order))
    pct = [100 * per_t[t][0] / per_t[t][1] for t in order]
    ax.barh(y, pct, color=BIIE.BLUE, edgecolor=BIIE.GREY_DARK, linewidth=0.3)
    for i, t in enumerate(order):
        c, tot = per_t[t]
        ax.text(pct[i] + 1.5, i, f"{c}/{tot}", va="center", fontsize=5.6,
                color=BIIE.GREY_DARK)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=6.4)
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Ro5 compliant (%)", fontsize=7.6)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.text(-0.22, 1.06, "b", transform=ax.transAxes, fontweight="bold",
            fontsize=10, ha="left", va="top")
    ax.set_title("Lipinski Ro5 compliance per target",
                 fontsize=7.4, color=BIIE.GREY_DARK, pad=4)

    # --- (c) Encoder cosine to intended target (reranked, n=849) ---
    ax = fig.add_subplot(gs[1, 0])
    cos = [float(r["intended_target_cosine"]) for r in rer if r["intended_target_cosine"]]
    med_cos = np.median(cos)
    ax.hist(cos, bins=30, color=BIIE.BLUE, alpha=0.85,
            edgecolor=BIIE.GREY_DARK, linewidth=0.3)
    ax.axvline(med_cos, ls="--", lw=1.0, color=BIIE.ALERT)
    ax.set_xlabel("dtSFM encoder cosine to intended target", fontsize=7.6)
    ax.set_ylabel("count", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(-0.18, 1.06, "c", transform=ax.transAxes, fontweight="bold",
            fontsize=10, ha="left", va="top")
    ax.set_title(f"Encoder cosine to conditioning target  (n = {len(cos):,} reranked)",
                 fontsize=7.4, color=BIIE.GREY_DARK, pad=4)

    # --- (d) Chemistry novelty vs training library ---
    ax = fig.add_subplot(gs[1, 1])
    tan = [float(r["max_tanimoto_to_training"]) for r in rer
           if r["max_tanimoto_to_training"]]
    counts = []
    for lo, hi, _ in NOVELTY_BINS:
        counts.append(sum(1 for t in tan if lo <= t < hi))
    pct = [100 * c / len(tan) for c in counts]
    x = np.arange(len(NOVELTY_BINS))
    cols = [BIIE.GREEN, BIIE.BLUE, BIIE.GREY_MID, BIIE.GOLD, BIIE.ALERT]
    bars = ax.bar(x, pct, color=cols, edgecolor=BIIE.GREY_DARK, linewidth=0.3)
    for b, c, p in zip(bars, counts, pct):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.8, f"{p:.0f}%\n({c})",
                ha="center", va="bottom", fontsize=5.8, color=BIIE.GREY_DARK,
                linespacing=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in NOVELTY_BINS], fontsize=5.6,
                       linespacing=1.0)
    ax.set_ylabel("% of reranked candidates", fontsize=7.6)
    ax.set_ylim(0, max(pct) * 1.25)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(-0.18, 1.06, "d", transform=ax.transAxes, fontweight="bold",
            fontsize=10, ha="left", va="top")
    ax.set_title(f"Chemistry novelty: max Tanimoto to 522,776-compound training library  (n = {len(tan):,})",
                 fontsize=7.0, color=BIIE.GREY_DARK, pad=4)

    paths = save_figure(fig, "figS3_decoder_base_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
