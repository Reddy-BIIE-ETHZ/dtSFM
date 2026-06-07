#!/usr/bin/env python3
"""make_fig6_gallery.py — Fig 6: dtSFM structural-validation gallery (16 targets).

Message: dtSFM generates MANY designs whose AF3 structural confidence is
indistinguishable from experimentally-validated drugs — for each target an
ensemble of designs (blues) converges on the same pocket as the approved drug
(orange). 4x4 grid; per cell: target + "+ <drug>" + "N designs · avg iPTM/PAE".
No panel letter/title (added in PowerPoint). Per-design SMILES live in the
supplementary one-by-one figure.

Run after PyMOL:
    pymol -cq data/dtsfm/scripts/figures/render_fig6_gallery.py
    PYTHONPATH=src python3 data/dtsfm/scripts/figures/make_fig6_gallery.py
Output: dtSFM-Figures/fig6_structural_gallery_PREVIEW.{pdf,png}
"""
from __future__ import annotations

import sys
import csv
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                     # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402
import matplotlib.image as mpimg       # noqa: E402
from rdkit import Chem                  # noqa: E402
from rdkit.Chem import AllChem, DataStructs  # noqa: E402
from rdkit import RDLogger              # noqa: E402
RDLogger.DisableLog("rdApp.*")
from calm.figures.biie_style import apply_style, BIIE, save_figure  # noqa: E402

RENDER = ROOT / "dtSFM-Figures/_fig6_overlays"
CIF = ROOT / "data/dtsfm/fig5_5_cifs"
N_ENSEMBLE = 5

# (target, approved drug) — grid order
REPS = [
    ("ALK", "lorlatinib"), ("FLT3", "midostaurin"),
    ("MAP2K1", "trametinib"), ("MAP2K2", "trametinib"),
    ("BTK", "acalabrutinib"), ("EGFR", "afatinib"),
    ("JAK1", "tofacitinib"), ("JAK2", "tofacitinib"),
    ("JAK3", "tofacitinib"), ("TYK2", "deucravacitinib"),
    ("CDK4", "palbociclib"), ("CDK6", "palbociclib"),
    ("PARP1", "talazoparib"), ("PARP2", "olaparib"),
    ("PIK3CA", "alpelisib"), ("F11", "asundexian"),
]
NCOL = 4
top10 = json.loads((Path("/tmp/top10.json")).read_text()) if os.path.exists("/tmp/top10.json") else {}


def autocrop(img, thr=0.985, pad=8):
    if img.ndim != 3 or img.shape[2] < 3:
        return img
    rgb = img[..., :3].astype(np.float32)
    if rgb.max() > 1.0 + 1e-6:
        rgb /= 255.0
    nz = (rgb < thr).any(axis=2)
    if not nz.any():
        return img
    r = np.where(nz.any(axis=1))[0]
    c = np.where(nz.any(axis=0))[0]
    return img[max(0, r[0]-pad):min(img.shape[0], r[-1]+pad),
              max(0, c[0]-pad):min(img.shape[1], c[-1]+pad)]


def metrics():
    m = {}
    for r in csv.DictReader(open(ROOT / "audit/dtsfm/decoder_af3/F5_results.tsv"),
                            delimiter="\t"):
        try:
            m[r["sample_name"]] = (float(r["iptm"]), float(r["interface_pae_min"]))
        except (TypeError, ValueError):
            pass
    return m


def design_smiles():
    """sample_name -> drug_smiles for every decoder candidate (n = 1200)."""
    s = {}
    for r in csv.DictReader(open(ROOT / "audit/dtsfm/decoder_af3/F5_results.tsv"),
                            delimiter="\t"):
        if r.get("drug_smiles"):
            s[r["sample_name"]] = r["drug_smiles"]
    return s


def anchor_smiles_map():
    """(target_gene, drug_name) -> canonical SMILES of approved-drug anchor.
    Source: decoder_target_binders.tsv (every (target, drug) row carries SMILES)."""
    a = {}
    for r in csv.DictReader(
            open(ROOT / "audit/dtsfm/decoder_target_binders.tsv"),
            delimiter="\t"):
        smi = r.get("smiles_canonical") or r.get("smiles_input")
        if smi:
            a[(r["target_gene"], r["drug_name"])] = smi
    return a


def _morgan(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def tanimoto(s1, s2):
    f1, f2 = _morgan(s1), _morgan(s2)
    if f1 is None or f2 is None:
        return None
    return DataStructs.TanimotoSimilarity(f1, f2)


# d1056 (PARP2) is a verbatim training-set duplicate; swap for clean novel d1062.
SWAP = {"dec_PARP2_1056": "dec_PARP2_1062"}


def shown_decoders(tgt):
    avail = {os.path.basename(os.path.dirname(p))
             for p in glob.glob(str(CIF / f"dec_{tgt}_*/dec_{tgt}_*_model.cif"))}
    ordered = [s for s in top10.get(tgt, []) if s in avail]
    ordered += sorted(avail - set(ordered))
    return [SWAP.get(s, s) for s in ordered[:N_ENSEMBLE]]


def main():
    apply_style()
    M = metrics()
    DSMI = design_smiles()
    ASMI = anchor_smiles_map()
    nrow = (len(REPS) + NCOL - 1) // NCOL
    fig = plt.figure(figsize=(8.6, 10.8))
    gs = fig.add_gridspec(nrows=nrow * 2, ncols=NCOL, height_ratios=[10, 3.2] * nrow,
                          hspace=0.16, wspace=0.08,
                          left=0.02, right=0.98, top=0.99, bottom=0.01)

    for i, (tgt, drug) in enumerate(REPS):
        r, c = divmod(i, NCOL)
        ax_img = fig.add_subplot(gs[2 * r, c])
        ax_lab = fig.add_subplot(gs[2 * r + 1, c])
        png = RENDER / f"{tgt}.png"
        if png.exists():
            ax_img.imshow(autocrop(mpimg.imread(png)))
        ax_img.axis("off")

        decs = shown_decoders(tgt)
        vals = [M[s] for s in decs if s in M]
        anchor_smi = ASMI.get((tgt, drug))
        tans = []
        for s in decs:
            t_val = tanimoto(DSMI.get(s), anchor_smi)
            if t_val is not None:
                tans.append(t_val)
        dlist = ", ".join("d" + s.split("_")[-1] for s in decs)
        if vals:
            ai = sum(v[0] for v in vals) / len(vals)
            ap = sum(v[1] for v in vals) / len(vals)
            mtxt = f"mean iPTM = {ai:.2f}   mean PAE = {ap:.1f} Å"
        else:
            mtxt = ""
        if tans and len(tans) == len(decs):
            ttxt = f"mean Tanimoto to {drug} = {sum(tans)/len(tans):.2f}"
        elif tans:
            ttxt = f"mean Tanimoto to {drug} = {sum(tans)/len(tans):.2f}   (n = {len(tans)} of {len(decs)})"
        else:
            ttxt = f"Tanimoto to {drug}: unavailable"

        ax_lab.axis("off"); ax_lab.set_xlim(0, 1); ax_lab.set_ylim(0, 1)
        ax_lab.text(0.0, 0.98, tgt, ha="left", va="top", fontsize=9.0,
                    weight="bold", color=BIIE.BLACK)
        ax_lab.text(1.0, 0.98, f"+ {drug}", ha="right", va="top", fontsize=6.6,
                    color="#EA580C")
        ax_lab.text(0.0, 0.68, dlist, ha="left", va="top", fontsize=5.6,
                    color=BIIE.BLUE)
        ax_lab.text(0.0, 0.40, mtxt, ha="left", va="top", fontsize=6.2,
                    color=BIIE.GREY_DARK)
        ax_lab.text(0.0, 0.12, ttxt, ha="left", va="top", fontsize=5.8,
                    color=BIIE.GREY_DARK)

    paths = save_figure(fig, "fig6_structural_gallery_PREVIEW",
                        output_dir=ROOT / "dtSFM-Figures")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
