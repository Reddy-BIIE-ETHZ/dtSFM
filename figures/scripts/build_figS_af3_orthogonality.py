#!/usr/bin/env python3
"""build_figS_af3_orthogonality.py — dtSFM v3 supplement: dtSFM ⊥ AlphaFold-3.

Shows that dtSFM encoder cosine and AF3 structural confidence (iPTM, interface
PAE) are independent — far below any tautology threshold (r ≪ 0.85) — so AF3 is a
genuinely orthogonal verifier, not a re-readout of dtSFM. Two cohorts:

  Row 1  Decoder-designed molecules (F5_results, 16 targets, n≈1,295): WITHIN a
         target, cosine carries ~no information about AF3 iPTM (per-target r ≈ 0).
  Row 2  §5.3 repurposing (NLRP3/CD73/STING1 incl. neg-controls, n≈286): moderate
         correlation across the wide binder→non-binder range — still well below
         tautology; this is the complementarity regime.

Source (audit/dtsfm/):
  decoder_af3/F5_results.tsv                cosine_v3, iptm, interface_pae_min
  tautology_dtsfm_vs_af3_section5_3.tsv     dtsfm_cosine, af3_iptm, af3_ifc_pae_A

Output: dtSFM-Figures/figS_af3_orthogonality_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                      # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

A = ROOT / "audit/dtsfm"


def fcol(rows, *keys):
    out = []
    for r in rows:
        vals = []
        ok = True
        for k in keys:
            v = (r.get(k) or "").strip()
            if not v:
                ok = False
                break
            vals.append(float(v))
        if ok:
            out.append(vals)
    return np.array(out) if out else np.empty((0, len(keys)))


def load_decoder():
    rows = list(csv.DictReader(open(A / "decoder_af3/F5_results.tsv"), delimiter="\t"))
    ci = fcol(rows, "cosine_v3", "iptm")
    cp = fcol(rows, "cosine_v3", "interface_pae_min")
    # per-target Pearson on cosine vs iptm
    import collections
    bt = collections.defaultdict(list)
    for r in rows:
        c, i = (r.get("cosine_v3") or "").strip(), (r.get("iptm") or "").strip()
        if c and i:
            bt[r["target"]].append((float(c), float(i)))
    rs = [np.corrcoef([a for a, _ in v], [b for _, b in v])[0, 1]
          for v in bt.values() if len(v) >= 20]
    return ci, cp, (min(rs), max(rs))


def load_repurp():
    rows = list(csv.DictReader(open(A / "tautology_dtsfm_vs_af3_section5_3.tsv"), delimiter="\t"))
    ci = fcol(rows, "dtsfm_cosine", "af3_iptm")
    cp = fcol(rows, "dtsfm_cosine", "af3_ifc_pae_A")
    return ci, cp


def panel(ax, xy, ylabel, hline, r_extra=None, paneltag="", tpos="top"):
    x, y = xy[:, 0], xy[:, 1]
    r = np.corrcoef(x, y)[0, 1]
    ax.scatter(x, y, s=4, c=BIIE.BLACK, alpha=0.18, edgecolors="none", rasterized=True)
    if hline is not None:
        ax.axhline(hline, color=BIIE.ALERT, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xlabel("dtSFM cosine", fontsize=7)
    txt = f"r = {r:.2f}   (n = {len(x):,})"
    if r_extra:
        txt += f"\nper-target r: {r_extra[0]:+.2f} … {r_extra[1]:+.2f}"
    ty, va = (0.96, "top") if tpos == "top" else (0.04, "bottom")
    ax.text(0.04, ty, txt, transform=ax.transAxes, ha="left", va=va,
            fontsize=6.2, color=BIIE.BLACK, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax.tick_params(labelsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if paneltag:
        ax.text(-0.22, 1.06, paneltag, transform=ax.transAxes, fontsize=9,
                weight="bold", ha="left", va="top")


def main():
    apply_style()
    dec_ci, dec_cp, dec_pt = load_decoder()

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    panel(axes[0], dec_ci, "AF3 iPTM", 0.7, r_extra=dec_pt, paneltag="a", tpos="bottom")
    panel(axes[1], dec_cp, "AF3 interface PAE (Å)", 5.0, paneltag="b", tpos="top")

    fig.suptitle("dtSFM cosine is independent of AlphaFold-3 confidence on decoder-designed molecules\n"
                 "(per-target r ~ 0; |r| << 0.85 tautology threshold — AF3 is an independent verifier)",
                 fontsize=7.4, y=0.995)
    fig.text(0.5, 0.01, f"n = {len(dec_ci):,} decoder cofolds across 16 targets.  "
             "(§5.3 repurposing cohort: see Fig 4a.)", ha="center", va="bottom",
             fontsize=6.0, color=BIIE.GREY_DARK)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.18, wspace=0.30)
    paths = save_figure(fig, "figS_af3_orthogonality_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    print(f"  decoder per-target r range: {dec_pt[0]:+.2f}..{dec_pt[1]:+.2f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
