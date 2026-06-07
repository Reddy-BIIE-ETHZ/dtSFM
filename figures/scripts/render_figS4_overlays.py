"""render_figS4_overlays.py — PyMOL renders for Supp S4 (one-by-one gallery).

Unlike Fig 6 (a 5-design ENSEMBLE per target), Supp S4 shows every top-10 design
INDIVIDUALLY: one render per design, that single design (BLUE) CEALIGN-overlaid
on the genuine approved-drug anchor (ORANGE) in the same pocket, grey cartoon
shown once. Also renders the anchor alone (orange) per target as the detail-page
reference cell.

Reads /tmp/figS4_meta.json (written by stage_figS4.py).

Run on Mac:  pymol -cq data/dtsfm/scripts/figures/render_figS4_overlays.py
Output: dtSFM-Figures/_figS4_overlays/<design_id>.png  +  anchor_<target>.png
"""
import os
import json
from pymol import cmd

CIF = "data/dtsfm/fig5_5_cifs"
OUT = "dtSFM-Figures/_figS4_overlays"
META = "/tmp/figS4_meta.json"
os.makedirs(OUT, exist_ok=True)

COL_DESIGN = "0x1565C0"   # BIIE blue
COL_PROT = "0xCCCCCC"
COL_ANC = "0xEA580C"      # orange
RAY = 880

meta = json.load(open(META))

cmd.set("ray_opaque_background", 0)
cmd.set("orthoscopic", 1)
cmd.set("ray_shadows", 0)
cmd.set("stick_radius", 0.20)


def cif_path(sample):
    return f"{CIF}/{sample}/{sample}_model.cif"


def render_pair(design_id, anchor_id, out_png):
    """One design (blue) overlaid on the anchor (orange); grey cartoon = design protein."""
    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.load(cif_path(design_id), "ref")
    has_anc = os.path.exists(cif_path(anchor_id))
    if has_anc:
        cmd.load(cif_path(anchor_id), "anc")
        try:
            cmd.cealign("ref and polymer", "anc and polymer")
        except Exception as e:
            print(f"[cealign fail] {design_id} <- {anchor_id}: {e}")
    cmd.hide("everything")
    cmd.show("cartoon", "ref and polymer")
    cmd.color(COL_PROT, "ref and polymer")
    cmd.set("cartoon_transparency", 0.5, "ref")
    sels = []
    cmd.show("sticks", "ref and organic")
    cmd.color(COL_DESIGN, "ref and organic")
    cmd.util.cnc("ref and organic")
    cmd.color(COL_DESIGN, "ref and organic and elem C")
    sels.append("(ref and organic)")
    if has_anc:
        cmd.show("sticks", "anc and organic")
        cmd.color(COL_ANC, "anc and organic")
        cmd.util.cnc("anc and organic")
        cmd.color(COL_ANC, "anc and organic and elem C")
        sels.append("(anc and organic)")
    sel = " or ".join(sels)
    cmd.orient(sel)
    cmd.zoom(sel, 4)
    cmd.ray(RAY, RAY)
    cmd.png(out_png, dpi=200)


def render_anchor_alone(anchor_id, out_png):
    cmd.reinitialize()
    cmd.bg_color("white")
    if not os.path.exists(cif_path(anchor_id)):
        print(f"[skip anchor] {anchor_id} missing")
        return
    cmd.load(cif_path(anchor_id), "anc")
    cmd.hide("everything")
    cmd.show("cartoon", "anc and polymer")
    cmd.color(COL_PROT, "anc and polymer")
    cmd.set("cartoon_transparency", 0.5, "anc")
    cmd.show("sticks", "anc and organic")
    cmd.color(COL_ANC, "anc and organic")
    cmd.util.cnc("anc and organic")
    cmd.color(COL_ANC, "anc and organic and elem C")
    cmd.orient("anc and organic")
    cmd.zoom("anc and organic", 4)
    cmd.ray(RAY, RAY)
    cmd.png(out_png, dpi=200)


n_done = 0
for tgt, m in meta.items():
    anc = m["anchor_id"]
    render_anchor_alone(anc, f"{OUT}/anchor_{tgt}.png")
    for d in m["designs"]:
        render_pair(d["id"], anc, f"{OUT}/{d['id']}.png")
        n_done += 1
    print(f"[OK] {tgt}: {len(m['designs'])} designs + anchor ({m['drug']})")

print(f"done — {n_done} design overlays + {len(meta)} anchor refs")
