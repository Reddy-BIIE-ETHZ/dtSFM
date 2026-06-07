"""render_fig6_gallery.py — PyMOL renders for Fig 6 (structural-validation gallery).

Per target: an ENSEMBLE of the most AF3-confident generated designs (shades of
BLUE) overlaid (CEALIGN) with the approved drug (ORANGE) in the same pocket —
showing many dtSFM designs all converge where the experimentally-validated drug
binds.

Run on Mac:  pymol -cq data/dtsfm/scripts/figures/render_fig6_gallery.py
Output: dtSFM-Figures/_fig6_overlays/<target>.png
"""
import os
import glob
import json
from pymol import cmd

CIF = "data/dtsfm/fig5_5_cifs"
OUT = "dtSFM-Figures/_fig6_overlays"
os.makedirs(OUT, exist_ok=True)
N_ENSEMBLE = 5

BLUES = ["0x1565C0", "0x1E88E5", "0x42A5F5", "0x0D47A1", "0x64B5F6"]
COL_PROT = "0xCCCCCC"
COL_ANC = "0xEA580C"

# anchor sample per target (4 from prior work + 12 new)
ANCHOR = {
    "ALK": "dec_ALK_0067", "FLT3": "dec_FLT3_0541",
    "MAP2K1": "dec_MAP2K1_0872", "MAP2K2": "dec_MAP2K2_0950",
    "BTK": "anchor_BTK_acalabrutinib", "EGFR": "anchor_EGFR_afatinib",
    "JAK1": "anchor_JAK1_tofacitinib", "JAK2": "anchor_JAK2_tofacitinib",
    "JAK3": "anchor_JAK3_tofacitinib", "TYK2": "anchor_TYK2_deucravacitinib",
    "CDK4": "anchor_CDK4_palbociclib", "CDK6": "anchor_CDK6_palbociclib",
    "PARP1": "anchor_PARP1_talazoparib", "PARP2": "anchor_PARP2_olaparib",
    "PIK3CA": "anchor_PIK3CA_alpelisib", "F11": "anchor_F11_asundexian",
}
TARGETS = list(ANCHOR.keys())
_only = os.environ.get("ONLY_TARGET")
if _only:
    TARGETS = [_only]
top10 = json.load(open("/tmp/top10.json")) if os.path.exists("/tmp/top10.json") else {}

# d1056 (PARP2) is a verbatim training-set compound (exact match to training
# drug #92882); replace it with the next clean, novel, high-confidence PARP2
# design d1062 (iPTM 0.98 / PAE 0.80) so the gallery shows only de novo designs.
SWAP = {"dec_PARP2_1056": "dec_PARP2_1062"}


def decoders_for(tgt):
    """Up to N available decoder samples for a target, top-PAE order first."""
    avail = {os.path.basename(os.path.dirname(p))
             for p in glob.glob(f"{CIF}/dec_{tgt}_*/dec_{tgt}_*_model.cif")}
    ordered = [s for s in top10.get(tgt, []) if s in avail]
    ordered += sorted(avail - set(ordered))
    return [SWAP.get(s, s) for s in ordered[:N_ENSEMBLE]]

cmd.set("ray_opaque_background", 0)
cmd.set("orthoscopic", 1)
cmd.set("ray_shadows", 0)
cmd.set("stick_radius", 0.18)

for tgt in TARGETS:
    decs = decoders_for(tgt)
    if not decs:
        print(f"[skip] {tgt}: no decoders")
        continue
    cmd.reinitialize()
    cmd.bg_color("white")
    # 1) load everything (ref + extra decoders + anchor) and align to ref
    objs = []
    for k, s in enumerate(decs):
        obj = "ref" if k == 0 else f"d{k}"
        cmd.load(f"{CIF}/{s}/{s}_model.cif", obj)
        if k > 0:
            try:
                cmd.cealign("ref and polymer", f"{obj} and polymer")
            except Exception as e:
                print(f"[cealign fail] {tgt} {s}: {e}")
        objs.append((obj, BLUES[k % len(BLUES)]))
    anc = ANCHOR.get(tgt)
    apath = f"{CIF}/{anc}/{anc}_model.cif" if anc else None
    has_anc = bool(apath and os.path.exists(apath))
    if has_anc:
        cmd.load(apath, "anc")
        try:
            cmd.cealign("ref and polymer", "anc and polymer")
        except Exception as e:
            print(f"[cealign fail anc] {tgt}: {e}")

    # 2) hide ALL, then show ONLY the ref cartoon (grey) + every ligand's sticks
    cmd.hide("everything")
    cmd.show("cartoon", "ref and polymer")
    cmd.color(COL_PROT, "ref and polymer")
    cmd.set("cartoon_transparency", 0.5, "ref")
    sels = []
    for obj, col in objs:
        cmd.show("sticks", f"{obj} and organic")
        cmd.color(col, f"{obj} and organic")
        cmd.util.cnc(f"{obj} and organic")
        cmd.color(col, f"{obj} and organic and elem C")
        sels.append(f"({obj} and organic)")
    if has_anc:
        cmd.show("sticks", "anc and organic")
        cmd.color(COL_ANC, "anc and organic")
        cmd.util.cnc("anc and organic")
        cmd.color(COL_ANC, "anc and organic and elem C")
        sels.append("(anc and organic)")
    sel = " or ".join(sels)
    cmd.orient(sel)
    cmd.zoom(sel, 4)
    cmd.ray(950, 950)
    cmd.png(f"{OUT}/{tgt}.png", dpi=300)
    print(f"[OK] {tgt}: {len(decs)} designs + {'anchor' if apath and os.path.exists(apath) else 'NO-ANCHOR'}")

print("done")
