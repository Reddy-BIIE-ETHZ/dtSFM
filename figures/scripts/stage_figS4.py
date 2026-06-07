#!/usr/bin/env python3
"""stage_figS4.py — prepare top-10 design CIFs + metadata for the Supp S4 gallery.

 Supp S4 is the "shock-and-awe" structural supplement: for every one of the 16
targets, the top-10 most AF3-confident decoder designs are shown INDIVIDUALLY
(one-by-one), each overlaid on the genuine approved-drug anchor cofold. This
script:

  1) reads the deposit manifest (the rescued on-target cofolds),
  2) ranks designs per target by AF3 confidence (interface PAE asc, iPTM desc),
  3) takes the top-10 (fewer where we generated fewer),
  4) ensures each chosen design CIF is staged as fig5_5_cifs/<id>/<id>_model.cif
     (copying from the flat deposit/cifs/<id>_model.cif when missing), and
  5) writes /tmp/figS4_meta.json consumed by render_figS4_overlays.py and
     make_figS4_supp.py.

The approved-drug anchor per target is a REAL staged cofold (not a decoder).

Run on Mac:  python3 data/dtsfm/scripts/figures/stage_figS4.py
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "audit/dtsfm/decoder_af3/deposit/manifest.tsv"
DEPOSIT_CIFS = ROOT / "audit/dtsfm/decoder_af3/deposit/cifs"
STAGE = ROOT / "data/dtsfm/fig5_5_cifs"
META_OUT = Path("/tmp/figS4_meta.json")
N_TOP = 10

# target -> (staged anchor cofold id, approved-drug display name). Real cofolds.
ANCHOR = {
    "ALK": ("anchor2_ALK_crizotinib", "crizotinib"),
    "FLT3": ("anchor2_FLT3_quizartinib", "quizartinib"),
    "MAP2K1": ("anchor2_MAP2K1_selumetinib", "selumetinib"),
    "MAP2K2": ("anchor2_MAP2K2_selumetinib", "selumetinib"),
    "BTK": ("anchor_BTK_acalabrutinib", "acalabrutinib"),
    "EGFR": ("anchor_EGFR_afatinib", "afatinib"),
    "JAK1": ("anchor_JAK1_tofacitinib", "tofacitinib"),
    "JAK2": ("anchor_JAK2_tofacitinib", "tofacitinib"),
    "JAK3": ("anchor_JAK3_tofacitinib", "tofacitinib"),
    "TYK2": ("anchor_TYK2_deucravacitinib", "deucravacitinib"),
    "CDK4": ("anchor_CDK4_palbociclib", "palbociclib"),
    "CDK6": ("anchor_CDK6_palbociclib", "palbociclib"),
    "PARP1": ("anchor_PARP1_talazoparib", "talazoparib"),
    "PARP2": ("anchor_PARP2_olaparib", "olaparib"),
    "PIK3CA": ("anchor_PIK3CA_alpelisib", "alpelisib"),
    "F11": ("anchor_F11_asundexian", "asundexian"),
}
TARGETS = list(ANCHOR.keys())


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def ensure_staged(sample: str) -> bool:
    """Make sure fig5_5_cifs/<sample>/<sample>_model.cif exists. Returns True if present."""
    dst_dir = STAGE / sample
    dst = dst_dir / f"{sample}_model.cif"
    if dst.exists():
        return True
    src = DEPOSIT_CIFS / f"{sample}_model.cif"
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main():
    rows = defaultdict(list)
    for r in csv.DictReader(open(MANIFEST), delimiter="\t"):
        tgt = r["target"]
        if tgt not in ANCHOR:
            continue
        i, p = f(r["iptm"]), f(r["interface_pae_min"])
        if i is None or p is None:
            continue
        rows[tgt].append({
            "id": r["sample"], "iptm": i, "pae": p,
            "smiles": r.get("drug_smiles", ""),
        })

    meta = {}
    for tgt in TARGETS:
        cand = sorted(rows[tgt], key=lambda d: (d["pae"], -d["iptm"]))
        chosen, staged = [], 0
        for d in cand:
            if len(chosen) >= N_TOP:
                break
            if ensure_staged(d["id"]):
                chosen.append(d)
                staged += 1
        anc_id, drug = ANCHOR[tgt]
        anc_ok = (STAGE / anc_id / f"{anc_id}_model.cif").exists()
        meta[tgt] = {"anchor_id": anc_id, "drug": drug,
                     "anchor_ok": anc_ok, "designs": chosen}
        flag = "" if anc_ok else "  [ANCHOR MISSING]"
        print(f"{tgt:8s} top-{len(chosen):<2d} (of {len(cand)} avail)  anchor={drug}{flag}")

    META_OUT.write_text(json.dumps(meta, indent=2))
    tot = sum(len(m["designs"]) for m in meta.values())
    print(f"\n{tot} design cells across {len(meta)} targets  ->  {META_OUT}")


if __name__ == "__main__":
    main()
