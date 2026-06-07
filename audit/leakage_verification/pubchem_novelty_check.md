# PubChem novelty check — generated designs (independent, post-audit)

**Date:** 2026-06-07 · **Method:** RDKit InChIKey (exact-structure) lookup against
PubChem (PUG REST `inchikey/{key}/cids`). A "known" call = an exact-structure
match exists in PubChem; "novel" = no exact match. Complements the ECFP4-Tanimoto
novelty already reported in the manuscript.

## Anchors (sanity — should all be known/correct)

| Set | Result |
|-----|--------|
| Decoder approved-drug anchors | **41 / 41 match PubChem by name** |
| Repurposing anchors (NLRP3/CD73/STING1) | **31 / 31 correct** (C‑176, H‑151, diABZI first mis-flagged by PubChem *name*-ambiguity; confirmed correct via recorded CIDs 2059265 / 149175126 / 131986624) |

No anchor-SMILES error in Paper 1.

## Generated designs

**Reported designs (Supplementary Table S1 / Fig 6 gallery — 80 designs):**
**80 / 80 novel** — none has an exact PubChem match.

**Full generation cohort (1,166 unique SMILES across 1,295 rows):**

| | count | % |
|---|---|---|
| Novel (no PubChem exact match) | 1,136 | 97.4% |
| Known (exact PubChem match) | 30 | 2.6% |

Breakdown of the 30 known:
- **11 = anchor recovery** — the decoder regenerated the target's own approved drug
  (Lorlatinib/ALK, Acalabrutinib/BTK, Palbociclib/CDK4, Afatinib/EGFR, Asundexian/F11,
  Midostaurin/FLT3, Tofacitinib/JAK1, Trametinib/MAP2K1, Talazoparib/PARP1,
  Olaparib/PARP2, Alpelisib/PIK3CA). Expected, and consistent with the anchor-recovery
  result; not a novelty concern.
- **19 = other known compounds** — 5 are degenerate ALK-tail generations
  (aspirin, loratadine, warfarin, amoxicillin, thyroxine) that **fail the filters**
  (not anchor-grade, not in the reranked cohort); the rest are rediscovered known
  PARP/PI3K inhibitor chemotypes for these heavily-studied target classes.

**Placement of the 30 known in reported sets:**

| Reported set | known compounds present |
|---|---|
| STRONG / MODERATE (17 wet-lab candidates) | **0** |
| Supp Table S1 / Fig 6 gallery (80 shown) | **0** |
| Reranked-849 cohort | 3 (dec_MAP2K2_0924, dec_PIK3CA_1134, dec_PIK3CA_1180 — none an anchor) |
| Anchor-grade structural gate | 25 (incl. the 11 anchor recoveries) |

> **Relation to the manuscript's "regenerated from training data" count.** The
> paper reports the number of designs that exactly reproduce a *training-set*
> compound (the memorization metric for a generative model). This PubChem check is
> a **broader** net — exact-structure matches anywhere in PubChem — so it also
> counts anchor recovery and known compounds that were never in training. The two
> numbers measure different things and are not in conflict.

**Conclusion:** every design *reported as a novel candidate* in the manuscript
(the 17 wet-lab STRONG/MODERATE and the 80-design gallery) is confirmed absent from
PubChem. The full-cohort known-compound rate is 2.6%, ~⅓ of which is desirable
anchor recovery; the remainder are tail generations (filtered out) or rediscovered
known chemotypes for well-studied targets.

> Reproduce: `audit/leakage_verification/` checks + the PubChem-novelty script
> (RDKit + PUG REST). The earlier verbatim training duplicate (d1056) was already
> replaced by the novel d1062 in Table S1 / Fig 7.
