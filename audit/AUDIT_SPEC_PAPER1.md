# AUDIT_SPEC_PAPER1.md — orthogonal audit spec for dtSFM v3 Paper 1

**Tag under audit:** `paper1-pre-audit`
**Audit operator:** Codex (OpenAI), clean session
**Output destination:** `audit/orthogonal_verification/codex_paper1_<sessionid>.md`

This document instructs the orthogonal auditor (Codex) how to audit
dtSFM v3 Paper 1 claims at the `paper1-pre-audit` git tag. Per
the orthogonal-audit protocol, Agent B
(Codex) has **no access to Agent A's chat history** — it reads only the
committed repository.

---

## 1. Inputs the auditor needs

1. The git tree at tag `paper1-pre-audit` (clone + `git checkout paper1-pre-audit`).
2. `docs/reproducing_paper_claims_paper1.md` — the canonical claim → file mapping (47 link targets, all of which resolve at this tag per a pre-audit sweep).
3. the release-audit blueprint (internal planning doc, not part of this public release) — the master scope spec.
4. `audit/leakage_verification/FINDINGS.md` — Class A/B/C taxonomy + species-ortholog disclosures.
5. the tooling-debt log (internal, not shipped) — reproducibility debt log explaining any TODO-downgraded rows.

The encoder model + decoder checkpoint hashes are in `audit/archival_checkpoint/checkpoint_manifest.tsv`; weights live on the private HuggingFace repo `SFM-BIIE-ETHZ/dtSFM-v3`.

**PREREQUISITE — Git LFS materialization (read first).** Most claim-source `.tsv` files in `audit/` are tracked by Git LFS (`*.tsv` rule in `.gitattributes`). A clone that has not run `git lfs pull` will see ~130-byte pointer stubs instead of data, which makes every TSV-sourced claim unverifiable. **Before auditing, confirm LFS content is materialized:** run `git lfs pull` (if the environment has network + git-lfs), or verify that e.g. `head audit/leakage_verification/pair_leakage.tsv` shows a real header row (`split  n_pairs  …`) and **not** a line beginning `version https://git-lfs.github.com/spec/v1`. If files are still pointer stubs and cannot be pulled, STOP and report "LFS not materialized" rather than marking TSV-sourced claims FAIL — that is an environment gap, not a claim failure. The plain-text summary `audit/dataset_stats.tsv` is intentionally kept **non-LFS** and is always readable.

---

## 2. Three-leg audit protocol

Per blueprint §G.2, run all three legs:

### Leg 1 — SA + CL claim verification

For each row in `docs/reproducing_paper_claims_paper1.md`'s claim tables:
- Locate the cited source file at the tag.
- Run the listed recipe (or an equivalent that achieves the same arithmetic).
- Compare the recomputed value to the WIP-text claim cited in the row.
- Emit verdict: **PASS** / **PARTIAL** / **FAIL** / **TODO** (for rows the doc already marks TODO with explanation — these are not failures).

Special handling:
- Rows whose source path is "on Euler; rsync" are TODOs (not FAILs) since the underlying §5.4 aggregators are preliminary and being regenerated post-audit.
- Rows whose source is "TODO — preliminary aggregate being regenerated" are TODOs by Sai's pre-audit direction, not audit failures.

### Leg 2 — Direct leakage measurement

For each of the 4 active §C dimensions in `audit/leakage_verification/`:
- `pair_leakage.tsv` — exact (drug_smiles, protein_id) pair leakage between train and val/test (claim: 0 PASS)
- `drug_leakage.tsv` — drug_smiles overlap fraction (claim: val=36%, test=37%)
- `protein_leakage.tsv` — protein_id overlap fraction (claim: 0%)
- `drug_tanimoto_buckets.tsv` + `drug_tanimoto_summary.tsv` — ECFP4 chemistry-novelty distribution (claim: 5K-subsample per split, seed=42)

Re-measure each. Report measured value vs claimed value.

Excluded by design: §C.4 cluster leakage (MMseqs2 protein-clustering). dtSFM v3 does **not** claim protein-sequence OOD; the cluster split is dataset hygiene only. Preserved for due diligence in `leakage_verification/_excluded_from_audit/` but **not** audited as a paper claim.

### Leg 3 — Filtered metric reporting (where leakage > 0)

For each dimension where Leg 2 finds non-zero leakage λ, report:

```
R_filtered = (R_raw − λ) / (1 − λ)
```

For dtSFM v3 the practical filtered claim is:
- §3 R@K is **already** sampled from val/test pairs that pass exact-pair leakage (Class A excluded by split construction). The filtered correction λ applies to the drug-side: 64-65% of val/test queries are Class C, 35-36% are drug-seen non-Class-C. The stratified R@K in `audit/eval_summaries/stratified_class_c_val.csv` already provides the disaggregated numbers.

---

## 3. Deliverable schema

Write `audit/orthogonal_verification/codex_paper1_<sessionid>.md`. Required sections:

### 3.1 Per-claim verdict table

One row per claim ID from the paper-claims doc. Columns:

| claim_id | source_path_resolves | verdict | measured_value | claimed_value | notes |
|---|---|---|---|---|---|

Verdict ∈ {PASS, PARTIAL, FAIL, TODO}.

### 3.2 Leg-2 leakage measurements

One row per of the 4 active §C dimensions. Columns:

| dimension | measured_value | claimed_value | verdict | notes |
|---|---|---|---|---|

### 3.3 Leg-3 filtered metrics

For each non-zero leakage dimension, the filtered R@K table. If all leakage dimensions are 0 (the expected outcome per `pair_leakage.tsv` claim of 0 PASS), state explicitly.

### 3.4 Audit gaps surfaced

Any WIP claim **not** mapped in the paper-claims doc. These are unverified-by-construction gaps. List with reference to WIP §-number.

### 3.5 Headline summary

3-5 sentences. Counts of PASS / PARTIAL / FAIL / TODO. Any single most-important finding.

---

## 4. Post-audit handoff

Once 3.1–3.5 is written, Agent A (Sai + claude) will:

- Triage PARTIALs / FAILs / surfaced gaps; write `audit/orthogonal_verification/audit_closure_paper1.md` documenting each disposition.
- Apply fixes where warranted.
- Tag `paper1-audit-closed` at the commit that includes both 3.1–3.5 and the closure notes.

---

## 5. Known-acknowledged TODOs that are NOT audit failures

Per pre-audit direction by Sai (2026-05-16):

- §5.4 F5_3_* decoder-portfolio aggregates: `F5_3_strong_candidates.tsv`, `F5_3_paper_headline_table.tsv`, `F5_3_candidate_summary.tsv`, `F5_2_selectivity_reranked.tsv`, `F5_anchor_drift.tsv`, `F5_summary.tsv` — preliminary, being regenerated from raw cofolds post-audit. Marked **TODO** in the paper-claims doc.
- 3 Euler-side decoder_af3 files (`decoder_cohort.tsv`, `F5_results.tsv`, `F5_2_named_offtarget_exact.tsv`) — exist on Euler but not committed in this tree at this tag. Not audit failures; will land alongside the F5_3_* regeneration.
- Extended §C.6 ortholog audit on the additional 40–60 paper-discussed compounds (beyond the 21 anchors in `species_ortholog_disclosures.tsv`). First pass committed at `f1cbdcf`; extension deferred.
- Per-bucket Tanimoto-stratified R@K (distinct from Class C / drug-seen stratification, which is done). Separate post-Tanimoto analysis.
- **§5.3-CIFS raw AF3 CIF binaries** for the 9 representative cofolds — large structure files held AF3/ALPS-side. The numeric iPTM/PAE values are committed (`F42_results.tsv`, `F5_3_anchor_table.tsv`) and the 9-cofold manifest is committed (`REPRESENTATIVES_section_5_3.md`); only the raw `.cif` binaries are deferred.
- **§5.5.2-FIGURE** `atpE_top4_overlays.pdf` — a render artifact regenerable from committed CIFs; not committed. The underlying atpE candidate numbers are committed.

Codex should mark these rows TODO, not FAIL.

---

**Last updated:** 2026-05-16 at tag `paper1-pre-audit`.
