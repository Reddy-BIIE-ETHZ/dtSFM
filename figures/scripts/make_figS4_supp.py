#!/usr/bin/env python3
"""make_figS4_supp.py — Supp S4 structural "shock-and-awe" supplement.

Two deliverables, both reading /tmp/figS4_meta.json + dtSFM-Figures/_figS4_overlays/:

  1) THE WALL  (figS4_wall_PREVIEW): one 16-row x (anchor + 10 design) mega-grid.
     Every top-10 design for every target, each individually overlaid on the
     approved-drug anchor (orange). Per-tile: design ID + iPTM/PAE (tiny). Left
     row label: target, drug, n designs, ensemble-avg iPTM/PAE. The visual
     density is the point — a proteome of structurally-credible candidates.

  2) DETAIL PAGES (figS4_<target>_PREVIEW, one per target): anchor reference cell
     + each top-10 design in its own cell, large enough to read the wrapped
     SMILES + iPTM/PAE. The auditable companion to the wall.

Run after PyMOL:
    pymol -cq data/dtsfm/scripts/figures/render_figS4_overlays.py
    PYTHONPATH=src python3 data/dtsfm/scripts/figures/make_figS4_supp.py
Output: dtSFM-Figures/figS4_wall_PREVIEW.{pdf,png}
        dtSFM-Figures/figS4_<target>_PREVIEW.{pdf,png}   (x16)
"""
from __future__ import annotations

import sys
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                     # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402
import matplotlib.image as mpimg       # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

RENDER = ROOT / "dtSFM-Figures/_figS4_overlays"
META = json.loads(Path("/tmp/figS4_meta.json").read_text())
ORANGE = "#EA580C"
FRAME = "#CFCFCF"          # light-gray cell frame (kinase-heatmap aesthetic)


def composite_white(img):
    img = img.astype(np.float32)
    if img.max() > 1.0 + 1e-6:
        img /= 255.0
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3:4]
        img = img[..., :3] * a + (1.0 - a)
    elif img.ndim == 3:
        img = img[..., :3]
    return np.clip(img, 0.0, 1.0)


def autocrop(img, thr=0.985, pad=6):
    nz = (img < thr).any(axis=2)
    if not nz.any():
        return img
    r = np.where(nz.any(axis=1))[0]
    c = np.where(nz.any(axis=0))[0]
    return img[max(0, r[0]-pad):min(img.shape[0], r[-1]+pad),
              max(0, c[0]-pad):min(img.shape[1], c[-1]+pad)]


def tile(path, pad=6):
    """White-composited, autocropped, square-padded image tile (or None)."""
    if not Path(path).exists():
        return None
    img = autocrop(composite_white(mpimg.imread(path)), pad=pad)
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.ones((s, s, 3), dtype=np.float32)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return canvas


def frame_cell(ax, color=FRAME, lw=0.6):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(color); sp.set_linewidth(lw)


def avg(designs):
    if not designs:
        return None, None
    return (sum(d["iptm"] for d in designs) / len(designs),
            sum(d["pae"] for d in designs) / len(designs))


# ---------------------------------------------------------------- THE WALL ----
def build_wall():
    apply_style()
    tgts = list(META.keys())
    nrow = len(tgts)
    NDES = 10
    ncol = 1 + 1 + NDES                      # label | anchor | 10 designs
    wr = [2.4, 1.15] + [1.0] * NDES
    fig = plt.figure(figsize=(11.6, 15.8))
    gs = fig.add_gridspec(nrow, ncol, width_ratios=wr,
                          hspace=0.06, wspace=0.05,
                          left=0.005, right=0.995, top=0.985, bottom=0.01)

    for r, tgt in enumerate(tgts):
        m = META[tgt]
        ai, ap = avg(m["designs"])
        # left label
        axL = fig.add_subplot(gs[r, 0]); axL.axis("off")
        axL.set_xlim(0, 1); axL.set_ylim(0, 1)
        axL.text(0.0, 0.74, tgt, ha="left", va="center", fontsize=10.5,
                 weight="bold", color=BIIE.BLACK)
        axL.text(0.0, 0.50, f"+ {m['drug']}", ha="left", va="center",
                 fontsize=7.0, color=ORANGE)
        axL.text(0.0, 0.28, f"{len(m['designs'])} designs  ·  avg iPTM {ai:.2f} · PAE {ap:.1f} Å",
                 ha="left", va="center", fontsize=5.8, color=BIIE.GREY_DARK)
        # anchor reference tile
        axA = fig.add_subplot(gs[r, 1])
        t = tile(RENDER / f"anchor_{tgt}.png")
        if t is not None:
            axA.imshow(t)
        frame_cell(axA, color=ORANGE, lw=1.0)
        if r == 0:
            axA.set_title("drug", fontsize=6.2, color=ORANGE, pad=2)
        # design tiles
        for k in range(NDES):
            ax = fig.add_subplot(gs[r, 2 + k])
            if k < len(m["designs"]):
                d = m["designs"][k]
                t = tile(RENDER / f"{d['id']}.png")
                if t is not None:
                    ax.imshow(t)
                frame_cell(ax)
                did = "d" + d["id"].split("_")[-1]
                ax.text(0.03, 0.03, f"{did}\n{d['iptm']:.2f}·{d['pae']:.1f}",
                        transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=3.7, color=BIIE.GREY_DARK, linespacing=0.95)
            else:
                ax.axis("off")

    paths = save_figure(fig, "figS4_wall_PREVIEW", output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  wall {fmt}: {p}")
    plt.close(fig)


# --------------------------------------------------------- DETAIL PAGES ----
def build_detail(tgt):
    apply_style()
    m = META[tgt]
    cells = [("ANCHOR", None)] + [("DES", d) for d in m["designs"]]
    NCOL = 3
    nslots = len(cells)
    nrow = (nslots + NCOL - 1) // NCOL
    fig = plt.figure(figsize=(8.4, 2.85 * nrow + 0.6))
    gs = fig.add_gridspec(nrow * 2, NCOL, height_ratios=[10, 4.2] * nrow,
                          hspace=0.10, wspace=0.10,
                          left=0.02, right=0.98, top=0.95, bottom=0.02)
    ai, ap = avg(m["designs"])
    fig.text(0.02, 0.985, f"{tgt}", ha="left", va="top", fontsize=13,
             weight="bold", color=BIIE.BLACK)
    fig.text(0.135, 0.985, f"+ {m['drug']} (approved-drug anchor, orange)",
             ha="left", va="top", fontsize=8.5, color=ORANGE)
    fig.text(0.98, 0.985,
             f"top {len(m['designs'])} designs · avg iPTM {ai:.2f} · PAE {ap:.1f} Å",
             ha="right", va="top", fontsize=7.5, color=BIIE.GREY_DARK)

    for i, (kind, d) in enumerate(cells):
        r, c = divmod(i, NCOL)
        ax = fig.add_subplot(gs[2 * r, c])
        axl = fig.add_subplot(gs[2 * r + 1, c])
        axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
        if kind == "ANCHOR":
            t = tile(RENDER / f"anchor_{tgt}.png", pad=8)
            if t is not None:
                ax.imshow(t)
            frame_cell(ax, color=ORANGE, lw=1.2)
            axl.text(0.5, 0.92, m["drug"], ha="center", va="top",
                     fontsize=8.5, weight="bold", color=ORANGE)
            axl.text(0.5, 0.55, "approved drug (reference pose)", ha="center",
                     va="top", fontsize=6.0, color=BIIE.GREY_DARK, style="italic")
        else:
            t = tile(RENDER / f"{d['id']}.png", pad=8)
            if t is not None:
                ax.imshow(t)
            frame_cell(ax)
            did = "d" + d["id"].split("_")[-1]
            axl.text(0.5, 0.96, did, ha="center", va="top", fontsize=8.5,
                     weight="bold", color=BIIE.BLUE)
            axl.text(0.5, 0.66, f"iPTM {d['iptm']:.2f}  ·  PAE {d['pae']:.1f} Å",
                     ha="center", va="top", fontsize=6.6, color=BIIE.GREY_DARK)
            smi = "\n".join(textwrap.wrap(d.get("smiles", ""), 34)[:3])
            axl.text(0.5, 0.40, smi, ha="center", va="top", fontsize=4.4,
                     color=BIIE.GREY_MID, family="monospace", linespacing=1.0)

    # blank remaining slots
    for i in range(len(cells), nrow * NCOL):
        r, c = divmod(i, NCOL)
        fig.add_subplot(gs[2 * r, c]).axis("off")
        fig.add_subplot(gs[2 * r + 1, c]).axis("off")

    paths = save_figure(fig, f"figS4_{tgt}_PREVIEW", output_dir=ROOT / "dtSFM-Figures")
    print(f"  {tgt}: {paths['png'].name}")
    plt.close(fig)


def main():
    build_wall()
    for tgt in META:
        build_detail(tgt)
    print("done")


if __name__ == "__main__":
    main()
