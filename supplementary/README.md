# Supplementary

| File | Description |
|------|-------------|
| `supp_table_S1_design_smiles.tsv` | Per-design SMILES for the 16-target generative gallery, with drug-likeness (Ro5), AlphaFold-3 iPTM / interface-PAE, Tanimoto-to-anchor, and structural-alert (ChemFlag) annotations. |

## Notes

- **ChemFlag** flags common structural alerts (e.g. alkyl halide
  `[CX4][Cl,Br,I]`, bare phosphine `[#15;!$([#15]~[#8])]`, free thiol
  `[#16X2H1]`). Flagged rows are shaded in the manuscript table; a flag is a
  cosmetic-liability heads-up for medicinal chemists, not a binding judgement.
- The PARP2 example shown in Fig 7 is design **d1062** (a novel design); an
  earlier candidate (d1056) was a verbatim training-set compound and was
  replaced.
- Generated molecules disclosed here are dedicated to the public (LICENSE §3).
