#!/usr/bin/env python3
"""build_fig5_panelE_agreement.py — dtSFM Fig 5 panel e: dtSFM ∩ AF3 agreement.

The "agreement = confidence" panel for the GENERATIVE cohort, using the two
genuinely independent models:

  x = dtSFM cosine, per-target normalized to 0-100 % (retrieval; same relative
      convention as Fig 4b — raw cosine is uninterpretable and target-baselined)
  y = AF3 iPTM     (structure; an independent verifier — NOTE we deliberately
                    do NOT use Boltz-2 here: Boltz-2 descends from the Boltz-1
                    SAIR structures that produced dtSFM's training labels, so it
                    is not independent. AF3 is the clean orthogonal referee.)

Message: dtSFM (retrieval) and AF3 (structure) carry independent information
(near-zero Pearson r over the generated candidates — far below any tautology
concern). AF3's confirmation is therefore genuine corroboration, not a restated
cosine. The generated candidates sit in the doubly-supported corner: high dtSFM
cosine AND AF3 iPTM ≥ 0.7. The four d-series variants shown structurally in
panel d are highlighted for cross-reference (no lead / selectivity claims).

This panel absorbs the former standalone AF3-orthogonality supplement.

Source: audit/dtsfm/decoder_af3/F5_results.tsv
Output: dtSFM-Figures/fig5_panelD_agreement_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import csv                              # noqa: E402
import math                             # noqa: E402

import matplotlib.pyplot as plt         # noqa: E402
import matplotlib.lines as ml           # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

F5 = ROOT / "audit/dtsfm/decoder_af3/F5_results.tsv"
DECODER = {"top_cosine", "scaffold_div", "mid_cosine"}

# The d-series variants shown structurally in panel d (cross-reference only —
# no lead / STRONG / selectivity claims; Paper 1 reports what the decoder made).
DSERIES = {
    "dec_ALK_0044", "dec_FLT3_0502", "dec_MAP2K1_0830", "dec_MAP2K2_0929",
}
def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def main():
    apply_style()
    import matplotlib.patheffects as pe
    raw = []  # (sample, target, cos, iptm, pae, af3_pass)
    with open(F5) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["stratum"] not in DECODER:
                continue
            c, i, p = f(r["cosine_v3"]), f(r["iptm"]), f(r["interface_pae_min"])
            if c is None or i is None or p is None:
                continue
            raw.append((r["sample_name"], r["target"], c, i, p,
                        r.get("af3_pass_combined", "").lower() in ("true", "1")))

    # Per-target min-max normalize cosine to 0-100 % (Fig 4b convention).
    cmin, cmax = {}, {}
    for _, t, c, *_ in raw:
        cmin[t] = min(c, cmin.get(t, c))
        cmax[t] = max(c, cmax.get(t, c))

    def norm(t, c):
        span = cmax[t] - cmin[t]
        return 100.0 * (c - cmin[t]) / span if span > 0 else 50.0

    # (sample, cosN, iptm, 1/pae, pass)
    cand = [(s, norm(t, c), i, 1.0 / max(p, 0.5), pa) for s, t, c, i, p, pa in raw]
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5), sharex=True)

    def panel(ax, yidx, ythr, ylabel, ylim):
        r = pearson([c[1] for c in cand], [c[yidx] for c in cand])
        ax.axhline(ythr, ls="--", lw=0.9, color=BIIE.ALERT, zorder=1)
        others = [c for c in cand if c[0] not in DSERIES]
        pas = [c for c in others if c[4]]
        fail = [c for c in others if not c[4]]
        ax.scatter([c[1] for c in fail], [c[yidx] for c in fail], s=9,
                   facecolors="none", edgecolors=BIIE.GREY_MID, linewidths=0.5,
                   alpha=0.7, zorder=2)
        ax.scatter([c[1] for c in pas], [c[yidx] for c in pas], s=9,
                   color=BIIE.BLUE, alpha=0.42, edgecolors="none", zorder=3)
        st = [c for c in cand if c[0] in DSERIES]
        ax.scatter([c[1] for c in st], [c[yidx] for c in st], s=14,
                   color=BIIE.GOLD, edgecolors=BIIE.BLACK, linewidths=0.5, zorder=5)
        ax.text(0.5, -0.26, f"r = {r:+.2f}    n = {len(cand):,}",
                transform=ax.transAxes, fontsize=7.0, color=BIIE.GREY_DARK,
                ha="center", va="top")
        ax.set_xlabel("dtSFM cosine (norm. %, per target)", labelpad=4)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.set_xlim(-3, 103)
        ax.set_ylim(*ylim)
        return r

    r_i = panel(axes[0], 2, 0.7, "AF3 iPTM (structure)", (0.55, 1.02))
    r_p = panel(axes[1], 3, 1.0 / 5.0, "1 / AF3 interface PAE  (Å$^{-1}$)", (0.0, 1.35))
    axes[0].text(0.0, 1.05, "c", transform=axes[0].transAxes, fontweight="bold",
                 fontsize=9, ha="left", va="bottom")

    h = [
        ml.Line2D([], [], marker="o", linestyle="none", markersize=5,
                  markerfacecolor=BIIE.GOLD, markeredgecolor=BIIE.BLACK,
                  label="generated variants shown in panel d"),
        ml.Line2D([], [], marker="o", linestyle="none", markersize=5,
                  markerfacecolor=BIIE.BLUE, markeredgecolor="none", alpha=0.6,
                  label="candidate, AF3-pass (both gates)"),
        ml.Line2D([], [], marker="o", linestyle="none", markersize=5,
                  markerfacecolor="none", markeredgecolor=BIIE.GREY_MID,
                  label="candidate, AF3-fail"),
        ml.Line2D([], [], linestyle="--", lw=0.9, color=BIIE.ALERT,
                  label="AF3 binder gate (iPTM ≥ 0.7 · PAE ≤ 5)"),
    ]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, fontsize=6.0, frameon=False, handletextpad=0.4,
               columnspacing=1.4, labelspacing=0.3)

    fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.26, wspace=0.28)
    paths = save_figure(fig, "fig5_panelD_agreement_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    print(f"  r(cos,iPTM)={r_i:+.3f}  r(cos,1/PAE)={r_p:+.3f}  n={len(cand)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
