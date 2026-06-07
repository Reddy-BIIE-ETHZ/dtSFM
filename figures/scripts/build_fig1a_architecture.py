#!/usr/bin/env python3
"""build_fig1a_architecture.py — Fig 1a dtSFM architecture schematic.

Extends the Vibe Coding generic SFM schematic (Agent/Target encoders → FFN →
mean-pool → contrastive head) to the full dtSFM architecture, fixing the
representation-flow issues of the encoder-only template:

  - Drug side has TWO representations: a MoLFormer-XL global embedding
    (768→512) AND per-atom features from an atom-level MLP (element + coords).
  - Protein side has TWO representations: ESM-2 per-residue (1,280→512) that
    flows into cross-attention, AND a pooled global vector for retrieval.
  - The pooled/global vectors feed the global-retrieval (contrastive) head;
    the per-atom / per-residue representations survive into cross-attention so
    the interface (per-atom) and contact (atom × residue) heads have their
    inputs.
  - The cross-attentive autoregressive DECODER is shown as the fifth output —
    the generative component that is new in dtSFM (gold, to highlight novelty).

Self-contained (no BIIE style import); sans-serif (Arial) per figure
convention. Output: dtSFM-Figures/fig1a_architecture_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "dtSFM-Figures"

plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ---- VC-style pastel palette ------------------------------------------------
DRUG_FILL, DRUG_EDGE = "#FBE5D6", "#C55A11"      # peach (drug / agent)
PROT_FILL, PROT_EDGE = "#DEEBF7", "#2E75B6"      # light blue (protein / target)
TRAIN_FILL, TRAIN_EDGE = "#E2EFDA", "#548235"    # light green (trainable)
HEAD_FILL, HEAD_EDGE = "#E8E1F0", "#7030A0"      # lavender (output heads)
HEADLITE_FILL = "#F2EEF7"                          # de-emphasised head (affinity)
DEC_FILL, DEC_EDGE = "#FFF2CC", "#BF9000"        # gold (decoder — novel)
TXT = "#222222"
GREY = "#666666"

fig, ax = plt.subplots(figsize=(12.2, 7.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, title, subtitle=None, fill="#FFFFFF", edge="#333333",
        title_size=10, sub_size=7.2, lw=1.4, title_color=TXT, sub_color=GREY,
        title_weight="bold"):
    """Rounded box with bold title + optional grey subtitle. (x,y) = bottom-left."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.4,rounding_size=1.6",
                       linewidth=lw, edgecolor=edge, facecolor=fill,
                       mutation_aspect=0.6, zorder=2)
    ax.add_patch(p)
    cx = x + w / 2
    if subtitle:
        ax.text(cx, y + h * 0.66, title, ha="center", va="center",
                fontsize=title_size, weight=title_weight, color=title_color, zorder=3)
        ax.text(cx, y + h * 0.30, subtitle, ha="center", va="center",
                fontsize=sub_size, color=sub_color, zorder=3, linespacing=1.15)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, weight=title_weight, color=title_color, zorder=3)
    return (x, y, w, h)


def arrow(p0, p1, color="#555555", lw=1.6, label=None, label_color=GREY,
          label_size=6.6, label_dy=1.6, rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                        linewidth=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + label_dy, label, ha="center", va="bottom",
                fontsize=label_size, color=label_color, style="italic", zorder=3)


def right(b):  return (b[0] + b[2], b[1] + b[3] / 2)
def left(b):   return (b[0], b[1] + b[3] / 2)
def top(b):    return (b[0] + b[2] / 2, b[1] + b[3])
def bottom(b): return (b[0] + b[2] / 2, b[1])


# ============================================================ INPUTS (col 1)
drug_in = box(2, 70, 15, 10, "SMILES string", "(Drug)",
              fill=DRUG_FILL, edge=DRUG_EDGE, title_size=9.5, sub_size=7.5)
prot_in = box(2, 20, 15, 10, "Target Sequence", "(Protein)",
              fill=PROT_FILL, edge=PROT_EDGE, title_size=9.5, sub_size=7.5)

# ============================================================ FROZEN ENCODERS (col 2)
molformer = box(22, 70, 24, 12, "MoLFormer-XL  (frozen)",
                "compound SMILES → 768-d\nglobal embedding",
                fill=DRUG_FILL, edge=DRUG_EDGE, title_size=9.5, sub_size=7.0)
atom_mlp = box(22, 53, 24, 11, "Atom-level MLP",
               "element identity + 3D coords\n→ per-atom features",
               fill=DRUG_FILL, edge=DRUG_EDGE, title_size=9.0, sub_size=7.0)

esm = box(22, 28, 24, 12, "ESM-2-650M  (frozen)",
          "target sequence → 1,280-d\nper-residue representation",
          fill=PROT_FILL, edge=PROT_EDGE, title_size=9.5, sub_size=7.0)
pool = box(22, 13, 24, 10, "Mean Pool",
           "per-residue → global vector",
           fill=PROT_FILL, edge=PROT_EDGE, title_size=9.0, sub_size=7.0)

# input → encoder arrows
arrow(right(drug_in), left(molformer), color=DRUG_EDGE)
arrow((drug_in[0] + drug_in[2], drug_in[1] + 2), (atom_mlp[0], atom_mlp[1] + atom_mlp[3] / 2),
      color=DRUG_EDGE, rad=-0.15)
arrow(right(prot_in), left(esm), color=PROT_EDGE)
arrow((esm[0] + esm[2] / 2, esm[1]), (pool[0] + pool[2] / 2, pool[1] + pool[3]),
      color=PROT_EDGE, lw=1.3)

# ============================================================ CROSS-ATTENTION (col 3)
ca = box(50, 40, 15, 44,
         "dtSFM\nCross-Attention\nEncoder",
         "trainable\n2 layers · 8 heads\nd = 512 · FFN 2,048\n14.4M params\n\nJoint drug–target\nrepresentation",
         fill=TRAIN_FILL, edge=TRAIN_EDGE, title_size=10.5, sub_size=7.2, lw=1.8)

# four representation arrows into the cross-attention encoder, labelled
arrow(right(molformer), (50, 76), color=DRUG_EDGE, label="global 768→512", label_dy=1.1)
arrow(right(atom_mlp),  (50, 60), color=DRUG_EDGE, label="per-atom", label_dy=1.1, rad=0.08)
arrow((esm[0] + esm[2], esm[1] + esm[3] / 2), (50, 50), color=PROT_EDGE,
      label="per-residue 1,280→512", label_dy=1.1, rad=0.10)
arrow((pool[0] + pool[2], pool[1] + pool[3] / 2), (50, 44), color=PROT_EDGE,
      label="global (pooled)", label_dy=-3.0, rad=0.18)

# ============================================================ STAGE 1: SCORING HEADS (upper right)
ax.text(86, 95, "Scoring heads", ha="center", va="top", fontsize=8.5,
        style="italic", color=GREY)
ca_r = (65, 62)

# Global-retrieval head (tall — carries the contrastive mechanism)
retr = box(72, 78, 27, 14, "Global-retrieval Head",
           "scores a pair by cosine similarity   ·   shared latent space d = 512\n"
           "→ Drug-to-Target (off-target safety)   ← Target-to-Drug (repurposing)\n"
           "Symmetric InfoNCE   ·   learned temperature  t = k$_B$T",
           fill=HEAD_FILL, edge=HEAD_EDGE, title_size=9.5, sub_size=6.6)
iface = box(72, 67, 27, 9, "Interface Head",
            "per-atom binding-interface membership",
            fill=HEAD_FILL, edge=HEAD_EDGE, title_size=9.0, sub_size=7.0)
contact = box(72, 56, 27, 9, "Contact Head",
              "atom × residue contact map",
              fill=HEAD_FILL, edge=HEAD_EDGE, title_size=9.0, sub_size=7.0)
affin = box(72, 45.5, 27, 8.5, "Affinity Head",
            "binding-strength regression  (preliminary)",
            fill=HEADLITE_FILL, edge="#A98FC2", title_size=9.0, sub_size=6.8,
            title_color="#6B6B6B", sub_color="#999999", lw=1.1)

for h in (retr, iface, contact, affin):
    arrow(ca_r, left(h), color=TRAIN_EDGE if h is not affin else "#A98FC2",
          lw=1.5 if h is not affin else 1.1, rad=-0.04)

# ============================================================ STAGE 2: GENERATIVE DECODER (lower lane, downstream)
ax.text(72, 33.5, "Generative decoder", ha="left", va="top", fontsize=8.5,
        style="italic", color=DEC_EDGE)

decoder = box(48, 11, 26, 19, "Cross-Attentive Decoder",
              "autoregressive SMILES generation,\none token at a time, conditioned on\nthe target via cross-attention over\nthe encoder's per-residue features",
              fill=DEC_FILL, edge=DEC_EDGE, title_size=9.8, sub_size=7.0)
gen_out = box(80, 13.5, 19, 14, "Novel drug\ncandidate", "(generated SMILES)",
              fill=DRUG_FILL, edge=DRUG_EDGE, title_size=9.5, sub_size=7.2)

# encoder joint representation → decoder (the "after the encoder" flow)
arrow((57.5, 40), (57.5, 30), color=DEC_EDGE, lw=1.8,
      label="target representation", label_color=DEC_EDGE, label_dy=0.2)
arrow(right(decoder), left(gen_out), color=DEC_EDGE, lw=1.8)

# "NEW" flag on decoder
ax.text(decoder[0] + decoder[2] - 1.5, decoder[1] + decoder[3] - 1.5, "NEW",
        ha="right", va="top", fontsize=7.5, weight="bold", color=DEC_EDGE, zorder=4)

# ---- frozen / trainable legend ----------------------------------------------
leg = [
    Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=DRUG_FILL,
           markeredgecolor=DRUG_EDGE, markersize=10, label="drug branch (frozen encoder)"),
    Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=PROT_FILL,
           markeredgecolor=PROT_EDGE, markersize=10, label="protein branch (frozen encoder)"),
    Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=TRAIN_FILL,
           markeredgecolor=TRAIN_EDGE, markersize=10, label="trainable cross-attention"),
    Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=HEAD_FILL,
           markeredgecolor=HEAD_EDGE, markersize=10, label="output heads"),
    Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=DEC_FILL,
           markeredgecolor=DEC_EDGE, markersize=10, label="generative decoder (new in dtSFM)"),
]
ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.0, -0.02), ncol=3,
          fontsize=7.2, frameon=False, handletextpad=0.4, columnspacing=1.4)

ax.text(0.5, 99, "a", fontsize=15, weight="bold", ha="left", va="top")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.06)
for ext in ("pdf", "png"):
    p = OUT / f"fig1a_architecture_PREVIEW.{ext}"
    fig.savefig(p, dpi=200)
    print(f"  {ext}: {p}")
plt.close(fig)
