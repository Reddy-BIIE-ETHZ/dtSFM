# §C Leakage Verification — Findings Summary

**Audit posture for dtSFM v3 (approved 2026-05-16):**
- No "OOD" terminology used in Paper 1.
- Class A/B/C taxonomy (per WIP §5.4.2) is the unified leakage framework across §3/§5.1/§5.2/§5.4.
- Cluster-based split is dataset hygiene; sequence-novel protein generalization is NOT claimed.
- Tanimoto distribution is a descriptive characterization of test-set chemistry composition, not a class assignment.

**All four active §C dimensions measured.** Status:

| Dimension | TSV | Verdict |
|---|---|---|
| §C.1 Drug-id overlap | `drug_leakage.tsv` | val 36% / test 37% in train; 63-64% novel by exact SMILES |
| §C.2 Protein-id disjointness | `protein_leakage.tsv` | **PASS** (0% overlap; sanity check, not OOD claim) |
| §C.3 Pair-OOD strict | `pair_leakage.tsv` | **PASS** (0 exact (drug, protein) pairs leak) |
| §C.5 Drug Tanimoto-novelty | `drug_tanimoto_*.tsv` | **The load-bearing chemistry-OOD signal.** See below. |

---

## §C.5 Tanimoto bucket distribution (5K-query subsample per split, seed=42)

**Source:** `drug_tanimoto_buckets.tsv`

| Bucket | val | test | Interpretation |
|---|---:|---:|---|
| [0.0, 0.3) — truly novel | 0.8% | 2.3% | No close analog in train |
| [0.3, 0.5) — novel scaffold variant | 29.2% | 26.5% | Distinct scaffold, related features |
| [0.5, 0.7) — moderate similarity | 20.4% | 21.6% | Recognizable family, distinct |
| [0.7, 0.9) — close analog | 12.1% | 11.6% | Similar analog known |
| [0.9, 1.0] — near-duplicate / exact | 37.4% | 38.1% | Same compound family / identical compound |

**Summary stats (`drug_tanimoto_summary.tsv`):**

| Split | n | mean max-Tan | median | p25 | p75 | n≥0.7 | n≥0.9 | exact-match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 4,999 | 0.71 | 0.69 | 0.46 | 1.00 | 2,481 (50%) | 1,874 (37%) | 1,844 (37%) |
| test | 4,995 | 0.71 | 0.70 | 0.47 | 1.00 | 2,487 (50%) | 1,903 (38%) | 1,874 (37%) |

## Headline interpretation

- **~37% of test drugs have an EXACT SMILES match in train.** This is consistent with
  the §C.1 drug-id overlap and reflects how the same drug is paired with multiple
  proteins across the training corpus (e.g., a kinase inhibitor like dasatinib
  paired with dozens of kinase targets).
- **~1-2% of test drugs are strictly novel by chemistry** (max Tanimoto < 0.3 to any
  training drug). This is the truly chemistry-OOD subset.
- **~30% of test drugs are in the "novel scaffold variant" bucket** (0.3–0.5), which
  is a defensible "novel chemistry" cohort with larger N.

## Implication for §3 retrieval claims

The dtSFM v3 §3 R@K metrics (e.g., d2t R@1 held-out val = 31.7%) are computed
on the full val/test pool. Under the Class A/B/C taxonomy applied to §3:

- **Class A** (exact pair in train): 0% by construction (pair_leakage PASS)
- **Class B** (drug+protein both in train, pair not): 0% by construction (val/test proteins are not in train per protein_leakage PASS)
- **Class C** (drug not in training): ~64% of val/test pairs (drug_leakage)
- **Drug-seen non-Class-C** (drug in train paired with other proteins not in val/test): ~36%

For paper-grade reporting, §3 R@K should be stratified as **Class C vs
drug-seen non-Class-C** — corresponds to the two operating regimes the
encoder is being tested against (chemistry-novelty regime vs within-drug-family
generalization across new protein contexts).

The ECFP4 Tanimoto distribution is a separate **descriptive characterization**
of the test set's chemistry composition (not a class assignment): median 0.70
max-Tanimoto to any training drug; ~37% near-duplicate; ~30% novel-scaffold-variant;
~1-2% strictly novel chemistry. Reported as a histogram in §3 supplementary.

## Recommended supplementary table for the paper

Per-class R@K (Class C vs drug-seen non-Class-C) re-computed using existing
encoder + per-drug class label from `drug_leakage.tsv`. Production of this
table is a small post-hoc analysis on the existing encoder; no retrain needed.

---

## §5.2 Species-ortholog disclosure (NEW finding 2026-05-16)

The Class A/B/C taxonomy as defined in §5.4.2 operates at the
**exact-protein-idx** level. It does **not** capture **species-ortholog
leakage**, where a drug is paired with a sequence-homologous ortholog of the
evaluation target in training.

### The MCC950 case (load-bearing example)

In our §5.2 repurposing screens, MCC950 ranks #3/522,776 for human NLRP3
(Q96P20) at cosine 0.371. The annotation labels this Class B (drug seen,
target seen, exact pair not paired). This is technically correct:

- MCC950 IS in v3 metadata (`pair_idx 23843`, `drug_idx 20138`)
- MCC950 is paired with `uniprot:Q8R4B8` (**MOUSE Nlrp3**) in training
- Human NLRP3 (`uniprot:Q96P20`) has 265 OTHER drugs paired in v3
- The exact (MCC950, human-NLRP3) pair is NOT in training → Class B

However, mouse Nlrp3 and human NLRP3 share high sequence homology (typically
~70–90%). The encoder's MCC950 → human-NLRP3 retrieval is therefore best
understood as **transitive species-ortholog generalization**, not pure
de-novo within-species chemistry discovery.

### Recommended §5.2 narrative wording

> "Reference compounds for each §5.2 target span the Class A/B/C taxonomy.
> MCC950, ranked #3 of 522,776 for human NLRP3, is Class B at the
> exact-protein-idx level (drug present in v3 training corpus paired with
> the mouse Nlrp3 ortholog `uniprot:Q8R4B8`, but not with human NLRP3
> `uniprot:Q96P20`). The encoder's retrieval reflects transitive generalization
> across species-ortholog pairs in training plus 264 (drug × human-NLRP3)
> Class A pairs from non-reference compounds (e.g., glyburide, Class A,
> rank not reported). We refer to this combined regime as 'within-family
> + ortholog generalization', distinct from pure de-novo within-target
> generalization."

### §C.6 — Scope-limited ortholog audit (paper-discussed compounds only)

By design, ortholog leakage is documented only for compounds
discussed in the paper text/figures, not for the full 522K-drug corpus or
for every screen hit. The scope is:

- §5.1 safety-screen panel (10 Klaeger 2017 clinical kinase inhibitors)
- §5.2 reference compounds per target (rows in `canonical_reference_binders.tsv`,
  ~7 per target × 3 targets = ~21)
- §5.2 named hits in the WIP §5.2 text (top-3 per target, ~9 compounds)
- §5.4 named decoder candidates (17 in the §5.4.7 portfolio table)
- §5.4 named clinical anchor compounds (in §5.4-A negative-selectivity panel)

**Total scope: ~60–80 named compounds.**

For each, the §C.6 check produces a TSV with columns:

| compound | intended_target | v3_class | training_paired_proteins | best_ortholog_identity | has_ortholog_leakage |
|---|---|---|---|---|---|

#### §C.6 — first pass executed 2026-05-16

Scope of this pass: the **21 in-training compounds** from
`canonical_reference_binders.tsv` (subset of the ~60–80 paper-discussed
set). All 21 anchor compounds × their training-pair `protein_id`s were
extracted from `metadata_v3.csv` and classified against canonical human
uniprots (NLRP3=Q96P20, CD73=P21589, STING1=Q86WV6) and known mouse
orthologs (mouse Nlrp3=Q8R4B8, mouse Nt5e=Q61503). Full table:
`species_ortholog_disclosures.tsv`.

**Surfaced (beyond the documented MCC950 case):**

| target | anchor | drug_idx | verdict | training pairings |
|---|---|---|---|---|
| NLRP3 | **MCC950** | 20138 | MOUSE_ORTHOLOG_ONLY | mouse Nlrp3 only (Q8R4B8) — _confirmed prior disclosure_ |
| NLRP3 | **Glyburide** | 11409 | both human + mouse | human Q96P20 + mouse Q8R4B8 + 6 off-target (8 pairs) |
| CD73 | **AB680** | 511365 | MOUSE_ORTHOLOG_ONLY | mouse Nt5e only (Q61503) — **NEW** |
| CD73 | **AOPCP** | 265390 | MOUSE_ORTHOLOG_ONLY | mouse Nt5e (Q61503) + 1 off-target (PLCG1) — **NEW** |
| CD73 | LY3475070 | 273007 | human only | human P21589 only — no ortholog issue |

Other 16 in-training anchors are **off-target controls** (Loratadine,
Amoxicillin, Levothyroxine, Warfarin, Dipyridamole, AMP, Adenosine,
Tranilast). Their training pairings touch unrelated proteins; the §5.2
encoder retrieval of these compounds against NLRP3/CD73/STING1 is not
explained by direct or ortholog training pair, but the panel is included
to test the encoder's "no false-positive" behavior on off-pathway drugs.

**STING1 finding:** No clinical / canonical STING1 anchor is in v3
training; only the 4 off-target negatives (Loratadine, Amoxicillin,
Levothyroxine, Warfarin) appear. STING1 §5.2 retrieval is therefore
**not** explainable by anchor or ortholog memorization — interesting
because the encoder still produces non-trivial STING1 rankings (see
`screen_STING1_aggregate_top1000_annotated.tsv`). Worth highlighting in
the §5.2 narrative: STING1 is the cleanest of the three targets w.r.t.
anchor-mediated leakage but has its own distinct caveat (top-1 is
Class B per §5.3 representatives table; drug-side generalization, not
target-side memorization).

**Two new species-ortholog disclosures (AB680, AOPCP)** materially
change the §5.2 CD73 narrative. The §5.2 aggregate said "CD73 top-100
= 100% Class A" — that's still true at the protein-idx level (Class A
means exact (drug, protein) pair seen in training). But the named
clinical/canonical anchors AB680 and AOPCP are **mouse-ortholog Class B
for human CD73 retrieval**, not Class A. This is an analogue of the
MCC950 disclosure and should be added to the §5.2 CD73 narrative
wording:

> "AB680 and AOPCP, both ranked in the top-tier of the human-CD73 §5.2
> retrieval, are paired in v3 training only with mouse Nt5e
> (`uniprot:Q61503`), not human CD73 (`uniprot:P21589`). Their high
> retrieval scores against the human target reflect transitive
> species-ortholog generalization, analogous to the documented MCC950
> case for human NLRP3."

**Open follow-up (post-`paper1-pre-audit`):** extend the audit from
the 21 anchor compounds to the ~40–60 additional paper-discussed
compounds (named §5.2 hits, §5.4 decoder portfolio, §5.4-A
negative-selectivity panel). The same script
(`data/dtsfm/scripts/audit_c6_species_ortholog.py` — to write) would
operate identically; only the input compound list changes.

## Subsample note

The 5K-per-split subsample was used because the full 63K val + 53K test queries
took an estimated 6 hours of single-threaded compute (per-query 109 MB
bitwise-AND memory traffic dominates). 5K is statistically sufficient for
bucket-fraction precision (±1% on the fractions reported above). Seed 42 ensures
reproducibility. The full eval can be run if reviewers request it — the
numpy-vectorized implementation in `leakage_tanimoto_fast.py` would benefit
from a multi-core implementation (e.g., joblib parallel over queries) to drop
runtime to ~30 min for the full set.
