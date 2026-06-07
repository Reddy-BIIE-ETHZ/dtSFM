"""render_fig_e6_overlays.py — PyMOL render for Fig e6 structural overlays.

Per-target PyMOL render of three v3-retrieved compounds bound to the target
protein. Ligands colored by §F.4 leakage class via element-carbon-only coloring
(heteroatoms keep element colors — N blue, O red, etc. — for chemistry
readability).

Pattern adapted from the decoder chat's `scripts/figures/render_fig5_5_overlays.py`
(F5.5) and the techniques documented in `src/calm/figures/README.md` §
"Structural overlay rendering" (locked 2026-05-11).

Run on Mac with CIFs local at data/dtsfm/fig_e6_cifs/:
    pymol -cq data/dtsfm/scripts/figures/render_fig_e6_overlays.py

Outputs: data/dtsfm/fig_e6_renders/render_<TARGET>.png  (3 files at 1200×900 / 300 DPI)
"""

import os
from pymol import cmd


CIF_BASE = "/Users/reddys/Downloads/CALM-0.1.0/data/dtsfm/fig_e6_cifs"
OUT_BASE = "/Users/reddys/Downloads/CALM-0.1.0/data/dtsfm/fig_e6_renders"
os.makedirs(OUT_BASE, exist_ok=True)

# Protein cartoon = light grey (matches decoder F5.5)
COL_PROTEIN = "0xC8C8C8"

# Class-color hex (BIIE palette — these are the canonical class colors used
# in Fig e4 / e5 captions and on the bottom labels of Fig e6).
# Class B = TEAL (locked palette: #3BC8B8 deep enough for stick render).
# Class A and C deepened for stick visibility against light-grey cartoon.
COL_A = "0x2848D8"   # deep BLUE_PURPLE
COL_B = "0x009E8E"   # deep TEAL
COL_C = "0xC03DB0"   # deep MAGENTA
# STING1 has 3 Class-C compounds; use 3 magenta-spectrum shades.
COL_C_DARK  = "0x7B1FA2"   # purple
COL_C_LIGHT = "0xF06292"   # hot pink

CLASS_COLOR = {
    "A":       COL_A,
    "B":       COL_B,
    "C":       COL_C,
    "C_dark":  COL_C_DARK,
    "C_light": COL_C_LIGHT,
}

# Per-target sample list + view parameters
TARGETS = {
    # NLRP3 split into its TWO binding sites (tighter zoom on each):
    "NLRP3_canonical": {
        "samples": [
            ("f42_101_NLRP3_Glyburide", "Glyburide", "A"),
            ("f42_002_NLRP3_MCC950",    "MCC950",    "B"),
        ],
        "buffer": 2.0,
        "extra_turn": None,
        "cartoon_transparency": 0.15,
    },
    "NLRP3_alt": {
        "samples": [
            ("f42_103_NLRP3_JC124",     "JC124",     "C"),
        ],
        "buffer": 2.0,
        "extra_turn": None,
        "cartoon_transparency": 0.15,
    },
    "CD73": {
        "samples": [
            ("f42_201_CD73_LY3475070", "LY3475070", "A"),
            ("f42_200_CD73_AB680",     "AB680",     "B"),
            ("f42_204_CD73_PSB-12379", "PSB-12379", "C"),
        ],
        "buffer": 3.0,
        # 90° y-rotation showed LY3475070 (extended molecule) edge-on;
        # pull back to 65° so all three ligands have visible 3D depth.
        "extra_turn": ("y", 65),
        "cartoon_transparency": 0.0,
    },
    "STING1": {
        "samples": [
            ("f42_310_STING1_ADU-S100", "ADU-S100", "C"),
            ("f42_311_STING1_SR-717",   "SR-717",   "C_dark"),
            ("f42_317_STING1_MSA-2",    "MSA-2",    "C_light"),
        ],
        "buffer": 3.0,
        "extra_turn": None,
        "cartoon_transparency": 0.0,
    },
}


def safe_obj(sample):
    """PyMOL object names can't contain '-'."""
    return sample.replace("-", "_")


def render_target(target, params):
    samples = params["samples"]
    buffer = params.get("buffer", 3.0)
    extra_turn = params.get("extra_turn", None)
    cartoon_transparency = params.get("cartoon_transparency", 0.0)

    cmd.delete("all")

    # Load all CIFs; rename objects for safety
    loaded = []   # (obj, sample, drug, class_key)
    for sample, drug, cls in samples:
        cif = f"{CIF_BASE}/{sample}_model.cif"
        if not os.path.exists(cif):
            print(f"  ⚠ MISSING: {cif}")
            continue
        obj = safe_obj(sample)
        cmd.load(cif, obj)
        loaded.append((obj, sample, drug, cls))

    if not loaded:
        print(f"  no CIFs for {target}; skipping")
        return

    # Pick reference = first loaded; CEALIGN all others to it (polymer chain)
    ref_obj = loaded[0][0]
    for obj, _, _, _ in loaded[1:]:
        try:
            cmd.cealign(f"{ref_obj} and polymer", f"{obj} and polymer")
        except Exception as e:
            print(f"  ⚠ cealign failed for {obj}: {e}")

    cmd.hide("everything")

    # Show one protein cartoon (reference); all proteins are aligned to it
    cmd.show("cartoon", f"{ref_obj} and polymer")
    cmd.color(COL_PROTEIN, f"{ref_obj} and polymer")
    if cartoon_transparency > 0:
        cmd.set("cartoon_transparency", cartoon_transparency)

    # Show all ligands as sticks; color CARBONS ONLY by class
    # (heteroatoms keep element colors for chemistry readability)
    for obj, _, _, cls in loaded:
        lig_sel = f"{obj} and not polymer"
        cmd.show("sticks", lig_sel)
        cmd.color(CLASS_COLOR[cls], f"{lig_sel} and elem C")
        cmd.set("stick_radius", 0.22, lig_sel)

    # Camera recipe (F5.5 pattern): orient on ligand union, slight x-tilt, zoom 3 Å
    all_lig = " or ".join(f"({obj} and not polymer)" for obj, _, _, _ in loaded)
    cmd.orient(all_lig)
    cmd.turn("x", -20)
    if extra_turn is not None:
        axis, deg = extra_turn
        cmd.turn(axis, deg)
    cmd.zoom(all_lig, buffer)

    # Render quality (F5.5 settings)
    cmd.bg_color("white")
    cmd.set("ray_shadow", 0)
    cmd.set("ray_opaque_background", 1)
    cmd.set("ambient", 0.35)
    cmd.set("specular", 0.2)
    cmd.set("antialias", 2)

    out_png = f"{OUT_BASE}/render_{target}.png"
    print(f"  → rendering {out_png}")
    cmd.ray(1200, 900)
    cmd.png(out_png, dpi=300)


def main():
    for target, params in TARGETS.items():
        print(f"\n=== Rendering {target} ({len(params['samples'])} compounds) ===")
        render_target(target, params)
    print("\nDone.")


main()
