"""build_fig1_panelA.py — dtSFM v3 architecture schematic.

Fig 1 Panel A (full-width top, single-figure spans page width).
Style: Vazquez-Lombardi iconography (Immunity 2022). Locked BIIE palette.

Layout (horizontal):
  drug branch (top-left)  -->                       --> retrieval head
                              cross-attn encoder    --> interface head
  protein branch (bot-left) -->  (center, large)   --> contact head
                                                   --> affinity head

Run:
    cd /Users/reddys/Downloads/CALM-0.1.0
    PYTHONPATH=src python3 data/dtsfm/scripts/figures/build_fig1_panelA.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from calm.figures.biie_style import BIIE, apply_style, save_figure  # noqa: E402

# Optional: RDKit for the dasatinib 2D depiction. Fail gracefully if not available.
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ---------------------------------------------------------------------------- #
# Layout constants — single canvas of 100 × 40 units, mapped to figsize inches
# ---------------------------------------------------------------------------- #
CANVAS_W = 100
CANVAS_H = 44

# Block coordinates (lower-left x, y, width, height)
BLK_DRUG_INPUT   = (2,  31, 14, 8)
BLK_MOLFORMER    = (20, 31, 16, 8)
BLK_PROT_INPUT   = (2,  5,  14, 8)
BLK_ESM2         = (20, 5,  16, 8)
BLK_ENCODER      = (42, 13, 26, 18)
BLK_HEAD_RETR    = (72, 33, 26, 5.5)
BLK_HEAD_INTF    = (72, 26, 26, 5.5)
BLK_HEAD_CONT    = (72, 19, 26, 5.5)
BLK_HEAD_AFF     = (72, 12, 26, 5.5)

# Color assignments — minimal palette per locked style (2026-05-19)
# Filled blocks keep their color; ALL text labels are black.
COL_DRUG    = BIIE.BLUE       # #1565C0
COL_PROT    = BIIE.GREEN      # #2E7D32
COL_ENCODER = BIIE.PURPLE     # #6A1B9A
COL_HEAD_1  = BIIE.BLUE       # retrieval
COL_HEAD_2  = BIIE.GREEN      # interface
COL_HEAD_3  = BIIE.GOLD       # contact (#F59E0B)
COL_HEAD_4  = BIIE.ALERT      # affinity (#C62828)
COL_FROZEN  = BIIE.GREY_MID   # frozen indicator
COL_ARROW   = BIIE.GREY_DARK  # flow arrows
BLACK       = "#000000"

DASATINIB = "CC1=C(C(=NC=N1)NC2=NC=C(C(=N2)C(=O)NC3=CC(=CC=C3C)Cl)O)NCCN1CCN(CC1)CCO"
PROTEIN_SAMPLE = "M K K F Q T L M T..."  # display-only stylization


def add_block(ax, xywh, color, label_lines, fontsize=8, label_color="white",
              linewidth=0, alpha=1.0, edgecolor=None, label_offset_x=0,
              text_align="center"):
    """Rounded-rectangle block. label_lines = list of (text, weight) pairs."""
    x, y, w, h = xywh
    if edgecolor is None:
        edgecolor = color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.6",
        facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha,
    )
    ax.add_patch(box)
    if isinstance(label_lines, str):
        label_lines = [(label_lines, "bold")]
    elif label_lines and isinstance(label_lines[0], str):
        # Treat as plain string list — first line bold, rest normal
        label_lines = [(s, "bold" if i == 0 else "normal")
                       for i, s in enumerate(label_lines)]
    # Compose into a single text artist using \n for built-in line spacing
    text = "\n".join(s for s, _ in label_lines)
    if text_align == "left":
        tx = x + 1.0 + label_offset_x
        ha = "left"
    else:
        tx = x + w/2 + label_offset_x
        ha = "center"
    ty = y + h/2
    # First-line bold + rest plain isn't natively supported in one call.
    # If multiple lines, render in two artists: bold top line + plain rest.
    if len(label_lines) == 1:
        ax.text(tx, ty, label_lines[0][0], ha=ha, va="center",
                fontsize=fontsize, color=label_color,
                weight=label_lines[0][1])
    else:
        # Lead bold line
        bold_text = label_lines[0][0]
        rest = "\n".join(s for s, _ in label_lines[1:])
        # Vertical layout: bold above, rest below
        n_rest = len(label_lines) - 1
        line_h = fontsize * 0.045  # in data units; tuned empirically
        bold_y = ty + line_h * (n_rest) / 2 + line_h * 0.3
        rest_y = bold_y - line_h - 0.1
        ax.text(tx, bold_y, bold_text, ha=ha, va="center",
                fontsize=fontsize, color=label_color, weight="bold")
        ax.text(tx, rest_y, rest, ha=ha, va="top",
                fontsize=fontsize * 0.9, color=label_color, weight="normal",
                linespacing=1.2)


def add_annotation(ax, x, y, text, fontsize=7, color=None, ha="center", va="top",
                   italic=False):
    if color is None:
        color = BIIE.GREY_DARK
    style = "italic" if italic else "normal"
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color, style=style)


def add_frozen_lock(ax, x, y, size=1.2):
    """Small lock icon next to frozen blocks."""
    ax.text(x, y, "❄", ha="center", va="center", fontsize=10, color=COL_FROZEN,
            zorder=10)


def add_flow_arrow(ax, start_xy, end_xy, color=None, lw=1.2, alpha=0.9,
                   connectionstyle="arc3,rad=0.0"):
    if color is None:
        color = COL_ARROW
    arrow = FancyArrowPatch(
        start_xy, end_xy,
        arrowstyle="->,head_length=4,head_width=3",
        connectionstyle=connectionstyle,
        color=color, linewidth=lw, alpha=alpha,
        zorder=2,
    )
    ax.add_patch(arrow)


def add_dasatinib_inset(ax, center_xy, zoom=0.25):
    """Add a small 2D depiction of dasatinib using RDKit. Fallback: hexagon."""
    cx, cy = center_xy
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(DASATINIB)
        # PNG bytes
        img = Draw.MolToImage(mol, size=(260, 130), kekulize=True,
                              wedgeBonds=True, fitImage=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        arr = plt.imread(buf, format="png")
        im = OffsetImage(arr, zoom=zoom)
        ab = AnnotationBbox(im, (cx, cy), frameon=False, pad=0)
        ax.add_artist(ab)
    else:
        # Fallback: hexagon
        hexpts = np.array([
            [cx + 1.5*np.cos(a), cy + 1.5*np.sin(a)]
            for a in np.linspace(0, 2*np.pi, 7)
        ])
        ax.plot(hexpts[:, 0], hexpts[:, 1], color=COL_DRUG, lw=1.5)


def add_protein_ribbon(ax, xywh, color):
    """Stylized AA sequence ribbon icon: dotted sinusoid + label."""
    x, y, w, h = xywh
    cx = x + w/2
    cy = y + h/2
    xs = np.linspace(x + 1, x + w - 1, 60)
    ys = cy + 0.5 * np.sin(np.linspace(0, 3*np.pi, 60))
    ax.plot(xs, ys, color="white", lw=1.5, alpha=0.9, zorder=5)


def add_head_icon(ax, xywh, kind, color):
    """Icon-free per locked minimal style. Placeholder kept for API compat."""
    return


def build():
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Branch labels (small grey above each branch group; black-grey text)
    add_annotation(ax, 9, 42, "Drug branch", fontsize=7,
                   color=BIIE.GREY_DARK, va="top")
    add_annotation(ax, 9, 2.0, "Protein branch", fontsize=7,
                   color=BIIE.GREY_DARK, va="bottom")
    add_annotation(ax, 85, 42, "Output heads", fontsize=7,
                   color=BIIE.GREY_DARK, va="top")

    # ---- DRUG INPUT (text-only) ----
    drug_x, drug_y, drug_w, drug_h = BLK_DRUG_INPUT
    add_block(ax, BLK_DRUG_INPUT, "white", [""], edgecolor=COL_DRUG,
              linewidth=1.2, label_color=COL_DRUG)
    ax.text(drug_x + drug_w/2, drug_y + drug_h/2, "Drug SMILES",
            ha="center", va="center", fontsize=7.5, color=BLACK)

    # ---- MoLFormer block ----
    # Single-artist multi-line: bold title above, italic frozen below
    add_block(ax, BLK_MOLFORMER, COL_DRUG, [""])
    mf_cx = BLK_MOLFORMER[0] + BLK_MOLFORMER[2]/2
    mf_cy = BLK_MOLFORMER[1] + BLK_MOLFORMER[3]/2
    ax.text(mf_cx, mf_cy + 1.0, "MoLFormer-XL",
            ha="center", va="center", fontsize=7.5, color="white")
    ax.text(mf_cx, mf_cy - 1.4, "frozen",
            ha="center", va="center", fontsize=6.5, color="white", style="italic")
    add_annotation(ax, BLK_MOLFORMER[0] + BLK_MOLFORMER[2]/2,
                       BLK_MOLFORMER[1] - 0.4,
                       "768-d global", fontsize=6.5, color=BIIE.GREY_DARK, va="top")

    # ---- PROTEIN INPUT (text-only) ----
    prot_x, prot_y, prot_w, prot_h = BLK_PROT_INPUT
    add_block(ax, BLK_PROT_INPUT, "white", [""], edgecolor=COL_PROT,
              linewidth=1.2)
    ax.text(prot_x + prot_w/2, prot_y + prot_h/2, "Protein sequence",
            ha="center", va="center", fontsize=7.5, color=BLACK)

    # ---- ESM-2 block ----
    add_block(ax, BLK_ESM2, COL_PROT, [""])
    esm_cx = BLK_ESM2[0] + BLK_ESM2[2]/2
    esm_cy = BLK_ESM2[1] + BLK_ESM2[3]/2
    ax.text(esm_cx, esm_cy + 1.0, "ESM-2-650M",
            ha="center", va="center", fontsize=7.5, color="white")
    ax.text(esm_cx, esm_cy - 1.4, "frozen",
            ha="center", va="center", fontsize=6.5, color="white", style="italic")
    add_annotation(ax, BLK_ESM2[0] + BLK_ESM2[2]/2,
                       BLK_ESM2[1] - 0.4,
                       "1280-d per-residue", fontsize=6.5, color=BIIE.GREY_DARK, va="top")

    # ---- ENCODER (center) ----
    # Render the title as one bold artist, the spec lines as one plain artist below.
    add_block(ax, BLK_ENCODER, COL_ENCODER, [""], label_color="white")
    enc_cx = BLK_ENCODER[0] + BLK_ENCODER[2]/2
    enc_cy = BLK_ENCODER[1] + BLK_ENCODER[3]/2
    ax.text(enc_cx, enc_cy + 2.6, "Cross-attention\nencoder",
            ha="center", va="center", fontsize=8.5, weight="bold",
            color="white", linespacing=1.0)
    ax.text(enc_cx, enc_cy - 2.4, "2L × 8H × 2048ff\nd = 512  ·  14.4M params",
            ha="center", va="center", fontsize=6.8, color="white",
            linespacing=1.3)
    add_annotation(ax, BLK_ENCODER[0] + BLK_ENCODER[2]/2,
                       BLK_ENCODER[1] - 0.4,
                       "drug-target joint representation",
                       fontsize=6.5, color=BIIE.GREY_DARK, va="top", italic=True)

    # ---- 4 OUTPUT HEADS (short headline; sub-label below the block) ----
    head_text_offset = 3.5  # offset from block left edge to avoid icon

    # Helper: emit headline + sub-label-below (no icons per locked minimal style)
    def emit_head(blk, color, kind, headline, sub):
        add_block(ax, blk, color, [headline], fontsize=7.5, label_color="white",
                  text_align="center", label_offset_x=0)
        add_annotation(ax, blk[0] + blk[2]/2, blk[1] - 0.3, sub,
                       fontsize=6.3, color=BIIE.GREY_DARK, va="top", italic=True)

    emit_head(BLK_HEAD_RETR, COL_HEAD_1, "retrieval",
              "Global retrieval", "cosine, D↔T")
    emit_head(BLK_HEAD_INTF, COL_HEAD_2, "interface",
              "Interface", "binding-pose probability")
    emit_head(BLK_HEAD_CONT, COL_HEAD_3, "contact",
              "Contact", "atom × residue")
    emit_head(BLK_HEAD_AFF, COL_HEAD_4, "affinity",
              "Affinity", "pAffinity regression")

    # ---- Flow arrows ----
    # Drug input -> MoLFormer
    add_flow_arrow(ax,
        (BLK_DRUG_INPUT[0] + BLK_DRUG_INPUT[2], BLK_DRUG_INPUT[1] + BLK_DRUG_INPUT[3]/2),
        (BLK_MOLFORMER[0], BLK_MOLFORMER[1] + BLK_MOLFORMER[3]/2))
    # MoLFormer -> Encoder (curve down into the encoder top)
    add_flow_arrow(ax,
        (BLK_MOLFORMER[0] + BLK_MOLFORMER[2], BLK_MOLFORMER[1] + BLK_MOLFORMER[3]/2),
        (BLK_ENCODER[0], BLK_ENCODER[1] + BLK_ENCODER[3] * 0.8),
        connectionstyle="arc3,rad=-0.18", lw=1.4)
    # Protein input -> ESM-2
    add_flow_arrow(ax,
        (BLK_PROT_INPUT[0] + BLK_PROT_INPUT[2], BLK_PROT_INPUT[1] + BLK_PROT_INPUT[3]/2),
        (BLK_ESM2[0], BLK_ESM2[1] + BLK_ESM2[3]/2))
    # ESM-2 -> Encoder (curve up into the encoder bottom)
    add_flow_arrow(ax,
        (BLK_ESM2[0] + BLK_ESM2[2], BLK_ESM2[1] + BLK_ESM2[3]/2),
        (BLK_ENCODER[0], BLK_ENCODER[1] + BLK_ENCODER[3] * 0.2),
        connectionstyle="arc3,rad=0.18", lw=1.4)

    # Encoder -> 4 heads (4 small arrows from encoder right edge)
    enc_right = BLK_ENCODER[0] + BLK_ENCODER[2]
    enc_cy = BLK_ENCODER[1] + BLK_ENCODER[3]/2
    for head_blk in (BLK_HEAD_RETR, BLK_HEAD_INTF, BLK_HEAD_CONT, BLK_HEAD_AFF):
        target_y = head_blk[1] + head_blk[3]/2
        add_flow_arrow(ax,
            (enc_right, enc_cy),
            (head_blk[0], target_y),
            connectionstyle="arc3,rad=0.0", lw=1.0, alpha=0.7)

    # Panel title (small, top-left corner)
    ax.text(0.5, 43.5, "a", fontsize=12, weight="bold",
            color=BIIE.BLACK, ha="left", va="top")

    # Save
    out = Path("dtSFM-Figures/fig1_panelA_architecture")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    build()
