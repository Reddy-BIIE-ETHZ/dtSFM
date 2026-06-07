# §5.3 representative AF3 cofolds — anchor, top decoder, negative control

Selection of 9 cofolds (3 targets × 3 roles) chosen as structurally-
representative anchors for the §5.3 narrative. Full per-cofold data is in
`f42_outputs/<cofold_dir>/` (CIF samples seed-42_sample-{0..4} +
`*_summary_confidences.json` + `*_ranking_scores.csv`). Manifest table is
`REPRESENTATIVES_section_5_3.tsv`.

Role definitions:
- **anchor (gold standard)** — clinical or research canonical inhibitor /
  agonist per `canonical_reference_binders.tsv`.
- **top decoder candidate** — highest-ranked repurposing hit from the
  `screen_<target>_top1000_annotated.tsv` (f42 indices ascend with rank).
- **negative control** — Aspirin: off-pathway drug with no expected
  binding to the inflammation / cGAS-STING / purinergic axes. Identical
  drug across all three targets so reviewers can compare interface
  signatures of the same molecule against three unrelated pockets.

## Results

| target | role | cofold | global iPTM | ifc iPTM (chain pair) | ifc PAE_min (Å) | clash |
|---|---|---|--:|--:|--:|:-:|
| NLRP3 | anchor (gold standard) | `f42_002_NLRP3_MCC950` | 0.73 | 0.73 | **8.38** | 0 |
| NLRP3 | top decoder candidate | `f42_000_NLRP3_didx39497` | 0.77 | 0.77 | **5.96** | 0 |
| NLRP3 | negative control | `f42_114_NLRP3_Aspirin` | 0.76 | 0.76 | **6.90** | 0 |
| CD73 | anchor (gold standard) | `f42_200_CD73_AB680` | 0.95 | 0.95 | **0.98** | 0 |
| CD73 | top decoder candidate | `f42_115_CD73_didx272410` | 0.81 | 0.81 | **3.53** | 0 |
| CD73 | negative control | `f42_209_CD73_Aspirin` | 0.80 | 0.80 | **4.68** | 0 |
| STING1 | anchor (gold standard) | `f42_317_STING1_MSA-2` | 0.79 | 0.79 | **3.16** | 0 |
| STING1 | top decoder candidate | `f42_210_STING1_didx7088` | 0.65 | 0.65 | **7.50** | 0 |
| STING1 | negative control | `f42_322_STING1_Aspirin` | 0.80 | 0.80 | **3.82** | 0 |

## Interpretation

**Discrimination quality by target:**

- **CD73** — clean ordering by ifc PAE_min: anchor (0.98 Å) ≪ top decoder
  (3.53 Å) < neg control (4.68 Å). AF3 strongly favors the known clinical
  anchor; the encoder's top decoder candidate scores intermediate
  structural plausibility; the off-pathway negative control is worst.
  This is the canonical "v3 + AF3 complementarity" signature: encoder
  prioritizes binders, AF3 confirms.
- **NLRP3** — **inverted ordering** by ifc PAE_min: MCC950 anchor
  (8.38 Å) > Aspirin neg control (6.90 Å). AF3 ranks the negative
  control's NLRP3 pose better than the clinical-canonical anchor's pose.
  Two non-exclusive explanations: (i) AF3 has known issues with
  diaryl-sulfonylurea drug-pose prediction on NACHT ATPase domains;
  (ii) MCC950 in v3 metadata is paired only with mouse Nlrp3 (per
  `FINDINGS.md` §"§5.2 Species-ortholog disclosure"), and the cofold is
  performed against human NLRP3 — the ortholog-pocket subtleties may
  drive AF3 into a non-canonical pose. Both contribute to the "encoder
  cosine and AF3 iPTM uncorrelated on novel cases" learning (see
  `learnings_absfm_decoder_phase1_v01_closed.md`).
- **STING1** — **inverted ordering** by ifc PAE_min: top decoder 7.50 Å
  > Aspirin 3.82 Å. AF3 ranks Aspirin's STING1 pose better than the
  encoder's top decoder. Consistent with the global pattern that AF3
  small-mol iPTM/PAE has a structurally-plausible ceiling on any small
  drug-like molecule against a well-folded pocket (see
  `learnings_af3_ceiling_conserved_folds.md`): AF3 is not a primary
  discriminator on drug-like molecules; relative-to-anchor selectivity
  is the load-bearing signal.

**For §5.3.3 (v3 + AF3 complementarity):** the table is intentionally
mixed-evidence — it would be misleading to commit only CD73 (where the
expected ordering holds). Paper §5.3 should report all three targets to
be transparent about where AF3 confirms the encoder, where it diverges,
and what known failure modes apply.

**Top-decoder leakage class caveat.** Per `screen_<target>_top1000_annotated.tsv`,
the rank-1 candidates chosen as "top decoder" representatives are:

| target | drug_idx | rank-1 cosine | leakage_class |
|---|--:|--:|:-:|
| NLRP3 | 39497 | 0.380 | **A** (drug + target both in training; memorization) |
| CD73 | 272410 | 0.411 | **A** (drug + target both in training; memorization) |
| STING1 | 7088 | 0.351 | **B** (drug seen with other targets; not paired with STING1) |

No Class C (truly de-novo: drug-OOD + target-OOD) candidate ranks #1 on
any of these three targets in the §5.2 repurposing screens. This is
consistent with §5.2's aggregate finding (NLRP3 top-100 = 94% Class B,
CD73 top-100 = 100% Class A — see `F4_top100_leakage_composition.tsv`)
and informs §5.3's framing: §5.3 verifies that AF3 structurally accepts
the encoder's rank-1 picks (mostly), but the picks themselves are
memorization-anchored, not de-novo discoveries. §5.4 / §5.5 (decoder
campaigns) are where the de-novo claims live.

## Provenance

- Local cache: `audit/repurposing/f42_outputs/` (323 cofolds,
  8.8 MB; pre-pulled from ALPS `af3_f42/outputs/` by prior chat).
- Per-cofold sample CIFs: 5 seeds (`seed-42_sample-{0..4}/model.cif`).
- Summary confidences schema: `iptm`, `ptm`, `ranking_score`,
  `chain_pair_iptm` (2×2), `chain_pair_pae_min` (2×2 Å),
  `chain_ptm`, `fraction_disordered`, `has_clash`. Chain 0 = protein,
  chain 1 = ligand (AF3 convention).
- Anchor selection: from `canonical_reference_binders.tsv` rows tagged
  `gold_standard` per target.
- Top decoder selection: smallest f42 ordering index for each target's
  `didx<NN>` series — which the f42 cohort builder writes in descending
  encoder-rank order (so f42_000 = rank-1 on NLRP3 cohort, etc.).

No ALPS pull was required for this manifest — all 323 cofolds were
already mirrored locally.
