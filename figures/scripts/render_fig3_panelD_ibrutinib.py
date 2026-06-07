"""render_fig3_panelD_ibrutinib.py — PyMOL render for Fig 3 Panel D.

Ibrutinib (drug-OOD) AF3 cofold gallery: the same molecule docked into three
dtSFM-predicted off-target kinases, structurally verified by AlphaFold-3.

  BLK   — dtSFM rank #1  / proteome; Klaeger-validated; iPTM 0.97, PAE 0.82 Å
  ERBB3 — dtSFM rank #9          ; NOVEL (absent from Klaeger panel);
                                   iPTM 0.97, PAE 1.12 Å  [headline]
  JAK3  — dtSFM rank #13         ; Klaeger-validated; iPTM 0.85, PAE 4.10 Å

Conventions adapted from render_fig_e6_overlays.py (locked 2026-05-11): light-grey
cartoon, ligand sticks with carbons colored (heteroatoms element-colored), white
ray-traced background.

Run on Mac (CIFs local):
    pymol -cq data/dtsfm/scripts/figures/render_fig3_panelD_ibrutinib.py

Outputs: data/dtsfm/fig3_panelD_renders/render_<TARGET>.png (1100x1100 / 300 DPI)
"""

import os
from pymol import cmd

CIF_BASE = "/Users/reddys/Downloads/CALM-0.1.0/audit/dtsfm/safety_panel_af3/ibrutinib"
OUT_BASE = "/Users/reddys/Downloads/CALM-0.1.0/data/dtsfm/fig3_panelD_renders"
os.makedirs(OUT_BASE, exist_ok=True)

COL_PROTEIN = "0xC8C8C8"          # light grey cartoon
COL_LIG_C = "0x1565C0"            # ibrutinib carbons = BIIE blue (one molecule, one colour)

# (dirname, TARGET, extra y-turn for a clean pocket view)
COFOLDS = [
    ("ibrutinib_blk",   "BLK",   0),
    ("ibrutinib_erbb3", "ERBB3", 0),
    ("ibrutinib_jak3",  "JAK3",  0),
]


def render(dirname, target, yturn):
    cif = f"{CIF_BASE}/{dirname}/{dirname}_model.cif"
    if not os.path.exists(cif):
        print(f"  MISSING: {cif}")
        return
    cmd.delete("all")
    cmd.load(cif, target)
    cmd.hide("everything")

    # protein cartoon, slightly translucent so the pocketed ligand reads
    cmd.show("cartoon", f"{target} and polymer")
    cmd.color(COL_PROTEIN, f"{target} and polymer")
    cmd.set("cartoon_transparency", 0.25)

    # ibrutinib sticks; carbons blue, heteroatoms element-coloured
    lig = f"{target} and not polymer"
    cmd.show("sticks", lig)
    cmd.set("stick_radius", 0.25, lig)
    cmd.color(COL_LIG_C, f"{lig} and elem C")

    # camera: orient on ligand, slight tilt, zoom to show ligand + pocket walls
    cmd.orient(lig)
    cmd.turn("x", -20)
    if yturn:
        cmd.turn("y", yturn)
    cmd.zoom(lig, 7.0)

    cmd.bg_color("white")
    cmd.set("ray_shadow", 0)
    cmd.set("ray_opaque_background", 1)
    cmd.set("ambient", 0.35)
    cmd.set("specular", 0.2)
    cmd.set("antialias", 2)

    out = f"{OUT_BASE}/render_{target}.png"
    print(f"  -> {out}")
    cmd.ray(1100, 1100)
    cmd.png(out, dpi=300)


def main():
    for dirname, target, yturn in COFOLDS:
        print(f"=== {target} ({dirname}) ===")
        render(dirname, target, yturn)
    print("Done.")


main()
