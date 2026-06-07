"""build_fig2_panelE.py — dtSFM v3 encoder performance dashboard.

Fig 2 Panel E. Compact table of all four output heads' metrics at the locked
epoch-10 checkpoint across three splits: in-distribution, OOD (cluster-held-out
val), and test.

Clean table, Arial, mostly black; subtle header rule. Values pulled live from
the committed quick_eval CSVs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
from calm.figures.biie_style import BIIE, apply_style  # noqa: E402

DATA = REPO_ROOT / "data/dtsfm/fig2"
BLACK = "#000000"


def e10(f):
    rows = [r for r in csv.DictReader(open(DATA / f)) if r["epoch"] == "10"]
    return rows[-1]


def build():
    apply_style()
    idd, val, test = e10("quick_eval_in_dist.csv"), e10("quick_eval_val.csv"), e10("quick_eval_test.csv")

    def pct(r, k): return f"{float(r[k])*100:.1f}"
    def rnk(r, k): return f"{float(r[k]):.0f}"
    def num(r, k): return f"{float(r[k]):.2f}"

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    x_metric = 2
    cols = [(58, "In-dist", idd), (75, "OOD val", val), (92, "Test", test)]
    row_h = 7.5
    LIGHT_BLUE = "#E6EEF5"
    zebra = {"i": 0}

    def data_row(yc, label, fmt, key):
        if zebra["i"] % 2 == 0:
            ax.add_patch(plt.Rectangle((0, yc - row_h / 2 + 0.6), 100, row_h - 1.2,
                                       facecolor=LIGHT_BLUE, edgecolor="none", zorder=0))
        zebra["i"] += 1
        ax.text(x_metric, yc, label, fontsize=7, color=BLACK, ha="left", va="center")
        for x, _, r in cols:
            ax.text(x, yc, fmt(r, key), fontsize=7, color=BLACK, ha="center", va="center")

    def section(y_top, title, size_key, size_unit, rows):
        ax.text(x_metric, y_top, title, fontsize=7.5, color=BLACK, ha="left", va="center")
        for x, name, r in cols:
            ax.text(x, y_top, name, fontsize=7.0, color=BIIE.GREY_DARK, ha="center", va="center")
            if size_key:
                ax.text(x, y_top - 4.2, f"{int(float(r[size_key])):,} {size_unit}",
                        fontsize=5.2, color=BIIE.GREY_MID, ha="center", va="center")
        yc = y_top - (9.5 if size_key else 6.5)
        for label, fmt, key in rows:
            data_row(yc, label, fmt, key)
            yc -= row_h
        return yc

    y = section(96, "Drug → Target  (retrieve target)", "pool_n_unique_proteins", "proteins",
                [("R@10 (%)", pct, "d2t_R@10"), ("median rank", rnk, "d2t_median_rank")])
    zebra["i"] = 0
    y = section(y - 5, "Target → Drug  (retrieve drug)", "pool_n_unique_drugs", "drugs",
                [("R@10 (%)", pct, "t2d_R@10"), ("median rank", rnk, "t2d_median_rank")])
    zebra["i"] = 0
    section(y - 5, "Output heads", None, None,
            [("Affinity  Pearson r", num, "affinity_pearson_r"),
             ("Interface  AUROC", num, "interface_auroc"),
             ("Contact  AUROC", num, "contact_auroc")])

    out = REPO_ROOT / "dtSFM-Figures/fig2_panelE_dashboard"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    build()
