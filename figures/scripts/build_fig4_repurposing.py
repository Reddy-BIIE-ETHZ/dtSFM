#!/usr/bin/env python3
"""build_fig4_repurposing.py — dtSFM v3 paper Figure 4.

Single-panel named-compound table showing v3 cosine retrieval of published
anchor compounds across three immunology targets (NLRP3, CD73, STING1).
Rendered in a Claude-Code / Markdown-table style: bold column headers,
hairline row dividers, alternating row bands, monospace numerics,
color-coded leakage class column, inline percentile sparkline.

Headline message: known binders are retrieved at the very top of the
522,776-drug v3 library — validating the cosine score as a real
binding-strength signal and licensing the inference that other top-ranked
compounds are credible new-binder candidates.

Class convention (consistent with §F.4):
    A = exact (drug, target) pair seen in v3 train (memorization upper bound)
    B = target seen, drug NOT seen (drug-OOD, true repurposing claim)
    C = drug NOT seen AND target NOT seen (full OOD; ranked via synthetic
        MoLFormer-XL → encoder pathway)

Failure rows (italic, grey) document the within-target across-scaffold-class
limitation discussed in §F.4.

Source data: audit/dtsfm/repurposing/F4_anchor_ranks_full.tsv
Output:      dtSFM-Figures/fig4_repurposing.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt        # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from calm.figures.biie_style import (  # noqa: E402
    apply_style, BIIE, save_figure,
)


# ============================================================================ #
# Curated anchor compound table (top 5–6 per target + 1–2 failure cases)
# Each row: (drug_name, stage, class, rank_int, pct_float, is_failure)
# Rank is int (cosine rank for Class A/B; synthetic rank for Class C).
# ============================================================================ #

CLASS_COLOR = {
    "A": BIIE.BLUE_PURPLE,
    "B": BIIE.TEAL,
    "C": BIIE.MAGENTA,
}

# §C.6 species-ortholog audit: these anchors' only training binding evidence is
# the MOUSE ortholog of the target; the human target sequence is OOD. Flagged
# with * (kept in class B; see footnote).
MOUSE_ORTHOLOG = {"MCC950", "AB680", "AOPCP"}

# (target_label, [rows ...]) where rows are tuples
# (drug, stage, class, rank, percentile, is_failure)
SECTIONS = [
    ("NLRP3", [
        ("MCC950",        "research",         "B",      3,   99.999, False),
        ("Inzomelid",     "Phase 2",          "C",     50,   99.990, False),
        ("JC124",         "research",         "C",    475,   99.909, False),
        ("NP3-562",       "Phase 1",          "C",  2_203,   99.579, False),
        ("Glyburide",     "approved, other",  "A",  2_936,   99.438, False),
        ("Dapansutrile",  "Phase 2",          "C", 214_145, 59.04,   True),
    ]),
    ("CD73", [
        ("AB680",         "Phase 2",          "B",    503,   99.904, False),
        ("PSB-12379",     "research",         "C",    833,   99.841, False),
        ("LY3475070",     "Phase 1",          "A",  1_118,   99.786, False),
        ("AOPCP",         "research",         "B",  1_276,   99.756, False),
        ("AMP",           "natural ligand",   "B",  1_612,   99.692, False),
        ("Dipyridamole",  "approved, other",  "B",198_556,   62.02,  True),
    ]),
    ("STING1", [
        ("ADU-S100",      "Phase 2 (disc.)",  "C",    913,   99.825, False),
        ("SR-717",        "research",         "C",  3_956,   99.243, False),
        ("2'3'-cGAMP",    "natural ligand",   "C",  5_301,   98.986, False),
        ("MK-1454",       "Phase 2",          "C",  5_698,   98.910, False),
        ("C-176",         "research, antag.", "C", 14_996,   97.131, False),
        ("diABZI 3",      "research",         "C",181_035,   65.37,  True),
        ("MSA-2",         "research",         "C",314_284,   39.88,  True),
    ]),
]

LIBRARY_SIZE = 522_776   # for the rank denominator in the column header


# Column x-positions (axes coordinates 0..100) — no sparkline column;
# the 99.999% etc. values carry the visual story on their own.
COL_TARGET = 3
COL_DRUG   = 13
COL_STAGE  = 38
COL_CLASS  = 60
COL_RANK   = 80
COL_PCT    = 97


def fmt_rank(r: int) -> str:
    """Comma-thousands."""
    return f"{r:,}"


def fmt_pct(p: float) -> str:
    """Percent with appropriate decimal places (more decimals near 100)."""
    if p >= 99.99:
        return f"{p:.3f}"
    if p >= 90:
        return f"{p:.2f}"
    return f"{p:.1f}"


def main():
    apply_style()

    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- Title ----
    fig.text(0.5, 0.965,
             "Figure 4 | Drug repurposing — v3 cosine ranks of published anchor compounds",
             ha="center", va="bottom",
             fontsize=11.0, weight="bold", color=BIIE.BLACK)
    fig.text(0.5, 0.940,
             "Library: 522,776 unique drugs   |   targets: NLRP3, CD73, STING1   |   "
             "v3 epoch_010.pt cosine retrieval",
             ha="center", va="bottom", style="italic",
             fontsize=7.5, color=BIIE.GREY_DARK)

    # ---- Class A/B/C legend at top (between subtitle and table header) ----
    legend_y = 0.905
    fig.text(0.06, legend_y, "A",
             color=CLASS_COLOR["A"], fontsize=8.0, weight="bold",
             ha="left", va="bottom")
    fig.text(0.085, legend_y, "pair-in-train",
             color=BIIE.GREY_DARK, fontsize=7.0,
             ha="left", va="bottom")
    fig.text(0.30, legend_y, "B",
             color=CLASS_COLOR["B"], fontsize=8.0, weight="bold",
             ha="left", va="bottom")
    fig.text(0.325, legend_y, "target-trained, drug-OOD",
             color=BIIE.GREY_DARK, fontsize=7.0,
             ha="left", va="bottom")
    fig.text(0.66, legend_y, "C",
             color=CLASS_COLOR["C"], fontsize=8.0, weight="bold",
             ha="left", va="bottom")
    fig.text(0.685, legend_y, "drug + target both OOD",
             color=BIIE.GREY_DARK, fontsize=7.0,
             ha="left", va="bottom")

    # ---- Column headers ----
    y_hdr = 87
    headers = [
        (COL_TARGET, "Target",          "left"),
        (COL_DRUG,   "Compound",        "left"),
        (COL_STAGE,  "Stage",           "left"),
        (COL_CLASS,  "Class",           "center"),
        (COL_RANK,   "Rank",            "right"),
        (COL_PCT,    "Percentile (%)",  "right"),
    ]
    for x, txt, ha in headers:
        ax.text(x, y_hdr, txt, ha=ha, va="center",
                fontsize=8.5, weight="bold", color=BIIE.BLACK)
    # Prominent denominator under "Rank" — bold black, not tiny grey
    ax.text(COL_RANK, y_hdr - 2.4,
            f"of {LIBRARY_SIZE:,}",
            ha="right", va="center",
            fontsize=7.5, weight="bold", color=BIIE.BLACK)
    # Header underline (solid)
    ax.plot([1, 99], [y_hdr - 4.0, y_hdr - 4.0],
            color=BIIE.BLACK, linewidth=1.0, clip_on=False)

    # ---- Body: 3 target sections ----
    ROW_DY = 3.6
    SECTION_GAP = 1.5
    y = y_hdr - 7.5    # bigger gap to accommodate the 2-line Rank header

    band_toggle = False

    for sec_idx, (target_name, rows) in enumerate(SECTIONS):
        for row_idx, (drug, stage, cls, rank, pct, is_failure) in enumerate(rows):
            # Alternating row bands (very subtle)
            if band_toggle:
                ax.add_patch(Rectangle(
                    (1, y - 1.6), 98, ROW_DY,
                    facecolor=BIIE.GREY_LIGHT, alpha=0.40,
                    edgecolor="none", clip_on=False, zorder=-1,
                ))

            # Target name only on the first row of the section, bold
            if row_idx == 0:
                ax.text(COL_TARGET, y, target_name,
                        ha="left", va="center",
                        fontsize=9.0, weight="bold", color=BIIE.BLACK)

            # Failure rows: italic but full black text (no dimming);
            # italics alone signal "documented limitation".
            txt_style = "italic" if is_failure else "normal"

            # Compound name (sometimes long, e.g., 2'3'-cGAMP); append * for
            # mouse-ortholog-only training cases (see footnote).
            ax.text(COL_DRUG, y, drug + ("*" if drug in MOUSE_ORTHOLOG else ""),
                    ha="left", va="center",
                    fontsize=8.0, color=BIIE.BLACK, style=txt_style,
                    weight="normal" if is_failure else "bold")

            # Stage
            ax.text(COL_STAGE, y, stage,
                    ha="left", va="center",
                    fontsize=7.5, color=BIIE.BLACK, style=txt_style)

            # Class — colored bold letter, centered (full color, even for failures)
            ax.text(COL_CLASS, y, cls,
                    ha="center", va="center",
                    fontsize=9.0, weight="bold",
                    color=CLASS_COLOR[cls])

            # Rank — right-aligned, monospace
            ax.text(COL_RANK, y, fmt_rank(rank),
                    ha="right", va="center",
                    fontsize=8.0, color=BIIE.BLACK, style=txt_style)

            # Percentile — right-aligned, monospace
            ax.text(COL_PCT, y, fmt_pct(pct),
                    ha="right", va="center",
                    fontsize=8.0, color=BIIE.BLACK, style=txt_style)

            y -= ROW_DY
            band_toggle = not band_toggle

        # Section separator (thin grey rule between targets)
        if sec_idx < len(SECTIONS) - 1:
            y -= SECTION_GAP
            ax.plot([1, 99], [y + 0.4, y + 0.4],
                    color=BIIE.GREY_MID, linewidth=0.4, clip_on=False)
            y -= SECTION_GAP
            band_toggle = False

    # ---- Footnote: mouse-ortholog disclosure (§C.6 audit) ----
    ax.text(3, 1.8,
            "* target seen in training only via its mouse ortholog; human "
            "target sequence is OOD (retrieval relies on cross-species transfer).",
            ha="left", va="center", fontsize=6.3, style="italic",
            color=BIIE.GREY_DARK, clip_on=False)

    paths = save_figure(fig, "fig4_repurposing",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
