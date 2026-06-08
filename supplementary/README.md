# Supplementary

| File | Description |
|------|-------------|
| `supp_table_S1_design_smiles.tsv` | Per-design SMILES for the 16-target generative gallery, with drug-likeness (Ro5), AlphaFold-3 iPTM / interface-PAE, Tanimoto-to-anchor, and structural-alert (ChemFlag) annotations. |
| `figS6_structural_gallery.pdf` | **Supplementary Figure S6** — full structural-validation gallery (17 pp.): a wall of all 16 targets with up to 10 decoder designs each (blue) overlaid on the approved drug (orange) in the AlphaFold-3 cofold pocket, with per-design iPTM/PAE, followed by one detail page per target. Hosted here because it is too large to embed in the main PDF; the preprint links to it. |

## Notes

- **ChemFlag** flags common structural alerts (e.g. alkyl halide
  `[CX4][Cl,Br,I]`, bare phosphine `[#15;!$([#15]~[#8])]`, free thiol
  `[#16X2H1]`). Flagged rows are shaded in the manuscript table; a flag is a
  cosmetic-liability heads-up for medicinal chemists, not a binding judgement.
- The PARP2 example shown in Fig 7 is design **d1062** (a novel design); an
  earlier candidate (d1056) was a verbatim training-set compound and was
  replaced.
- Generated molecules disclosed here are dedicated to the public (LICENSE §3).
