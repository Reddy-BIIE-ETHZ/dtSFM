# Reproducing dtSFM v3 Paper 1 Claims

This document maps every numerical claim in [the Paper 1 WIP](the dtSFM Paper 1 preprint)
to a committed file path + reproduction recipe. Every row below is one
verifiable claim. This is the input artifact for the Codex orthogonal
audit (orthogonal-audit protocol).

**Audit posture:** any claim a reader can't trace to a file via this doc
should be flagged. If a claim says "X happens" but no row below shows
"X = file.tsv:column:row", the WIP needs editing.

## Unified Class A/B/C taxonomy (paper-wide)

dtSFM v3 uses a single leakage taxonomy across §3, §5.1, §5.2, and §5.4
(per WIP §5.4.2). The word "OOD" is **not used** in Paper 1; we describe
data composition explicitly with this taxonomy:

| Class | Definition |
|---|---|
| **A** | Exact `(drug, protein)` pair appears in training (memorization regime) |
| **B** | Drug seen in training AND protein seen in training, but this specific pair not paired in training (within-family generalization regime) |
| **C** | Drug NOT in training (chemistry-novelty regime; protein status independent) |

The held-out PDB-entry-cluster split for §3 val/test produces splits with
0% Class A (by construction; verified in `pair_leakage.tsv`) and 0% Class B
(by construction; val/test proteins are not in train per `protein_leakage.tsv`);
val/test composition is dominated by **Class C drug-novel** (64%) + the
remainder which is "drug seen in train (paired with other proteins not in
val/test)" — labeled `drug-seen non-Class-C` for §3 stratification.

The ECFP4 Tanimoto distribution is a separate **descriptive characterization**
of chemistry novelty in the test set (median 0.70 max-Tanimoto to train),
reported as a histogram in §3 supplementary, NOT used as a class assignment.

---

## How to use

Each claim has:

- **ID**: stable identifier (e.g. `P1-§3.1-R1`) usable in Codex audit logs
- **WIP claim** (verbatim or close): the textual claim in the manuscript
- **Source artifact**: file path + (column / row / aggregate) anchor
- **Reproduction recipe**: minimal Python/CLI snippet to recompute

If a recipe says "see [`script.py`]", the script is in [`data/dtsfm/scripts/`](../data/dtsfm/scripts/)
and is runnable against the committed inputs.

---

## §2 — Training data

| ID | Claim | Source | Recipe |
|---|---|---|---|
| P1-§2-N_PAIRS | 714,747 drug-target pairs | `metadata_v3.csv` row count | `wc -l metadata_v3.csv` minus header |
| P1-§2-N_PDBbind | 19,037 PDBbind pairs | `metadata_v3.csv` rows with `protein_id LIKE 'pdb:%'` | grep + count |
| P1-§2-N_SAIR | 695,710 SAIR pairs | `metadata_v3.csv` rows with `protein_id LIKE 'uniprot:%'` | grep + count |
| P1-§2-N_DRUGS | 522,776 unique drugs | `metadata_v3.csv` unique `drug_smiles` | pandas `nunique` |
| P1-§2-N_PROTEINS | 22,964 unique proteins | `metadata_v3.csv` unique `protein_id` | pandas `nunique` |
| P1-§2-SPLIT_VAL | 2,296 held-out val PDB-entry clusters (data-hygiene mechanism; sequence-OOD NOT claimed) | `audit/split_index/val_pair_idx.parquet` joined with metadata `cluster_id` | see [`audit/split_index/README.md`](../audit/split_index/README.md) |
| P1-§2-SPLIT_TEST | 2,296 held-out test PDB-entry clusters | `audit/split_index/test_pair_idx.parquet` | same |
| P1-§2-HOLDOUT_24 | 24 curated evaluation pairs (21 confirmed absent from training corpus; 3 documented in-training per §F1 of split_index/README.md) | `audit/heldout_validation_pairs.tsv` | direct read |

---

## §3 — Encoder validation (B-3, epoch 10)

All §3 retrieval claims trace to:
[`audit/eval_summaries/quick_eval_val.csv`](../audit/eval_summaries/quick_eval_val.csv) (held-out cluster val)
and [`quick_eval_in_dist.csv`](../audit/eval_summaries/quick_eval_in_dist.csv) (within-train-cluster pairs).

| ID | Claim (epoch 10 values) | Source | Recipe |
|---|---|---|---|
| P1-§3.1-D2T-R1 | drug→target R@1 (held-out val) = 31.7% | `quick_eval_val.csv` row epoch=10 col `d2t_R@1` | `grep ',10,val,' file.csv | awk -F, '{print $5}'` |
| P1-§3.1-D2T-R5 | drug→target R@5 = 56.4% | same, `d2t_R@5` | same |
| P1-§3.1-D2T-R10 | drug→target R@10 = 65.2% | same, `d2t_R@10` | same |
| P1-§3.2-T2D-R1 | target→drug R@1 (held-out val) = 27.7% | same, `t2d_R@1` | same |
| P1-§3.2-T2D-R5 | target→drug R@5 = 46.6% | same, `t2d_R@5` | same |
| P1-§3.2-T2D-R10 | target→drug R@10 = 55.0% | same, `t2d_R@10` | same |
| P1-§3-STRAT | R@K stratified by Class C vs drug-seen non-Class-C | TODO — small post-hoc analysis on existing encoder | re-run pool retrieval with per-query class label |
| P1-§3-TAN | ECFP4 Tanimoto distribution of test-drug chemistry-novelty (descriptive, not a class assignment) | [`audit/leakage_verification/drug_tanimoto_summary.tsv`](../audit/leakage_verification/drug_tanimoto_summary.tsv) | numpy-vectorized; see `data/dtsfm/scripts/leakage_tanimoto_fast.py` |
| P1-§3.3-AFFINITY | Affinity Pearson per Class A/B/C stratum (§S.6 taxonomy) | [`audit/affinity_calibration/F6_phase3_metrics.tsv`](../audit/affinity_calibration/F6_phase3_metrics.tsv) | direct parse; see §S.6 row labels |
| P1-§3.4-SPEED | ms per pair inference | TODO — speed bench script pending submission | see [audit/eval_summaries/](../audit/eval_summaries/) |

---

## §5.1 — Safety screening

| ID | Claim | Source | Recipe |
|---|---|---|---|
| P1-§5.1-PANEL | Klaeger 2017 panel of 10 clinical kinase inhibitors | [`audit/klaeger2017/klaeger_drugs.tsv`](../audit/klaeger2017/klaeger_drugs.tsv) | wc -l |
| P1-§5.1-RECALL | Top-K recall per drug for known off-targets | [`audit/safety_screen_results.tsv`](../audit/safety_screen_results.tsv) | direct parse |
| P1-§5.1-SUMMARY | Aggregate safety-screen recall | [`audit/safety_screen_summary.json`](../audit/safety_screen_summary.json) | direct read |
| P1-§5.1-LEAKAGE | Pair-leakage classification of panel | [`audit/safety_panel_pair_leakage.tsv`](../audit/safety_panel_pair_leakage.tsv) | direct parse |

---

## §5.2 — Drug repurposing

| ID | Claim | Source | Recipe |
|---|---|---|---|
| P1-§5.2-NLRP3-TOP1000 | NLRP3 top-1000 ranking of 522,776 drugs | [`audit/repurposing/screen_NLRP3_top1000_annotated.tsv`](../audit/repurposing/screen_NLRP3_top1000_annotated.tsv) | direct |
| P1-§5.2-CD73-TOP1000 | CD73 top-1000 ranking | [`screen_CD73_top1000_annotated.tsv`](../audit/repurposing/screen_CD73_top1000_annotated.tsv) | direct |
| P1-§5.2-STING1-AGG | STING1 aggregate top-1000 ranking | [`screen_STING1_aggregate_top1000_annotated.tsv`](../audit/repurposing/screen_STING1_aggregate_top1000_annotated.tsv) | direct |
| P1-§5.2-STING1-SEQ | STING1 sequence-only sub-claim | [`screen_STING1_seqonly_top1000_annotated.tsv`](../audit/repurposing/screen_STING1_seqonly_top1000_annotated.tsv) | direct |
| P1-§5.2-MCC950-RANK | MCC950 ranks #3 / 522K for NLRP3 (Class B) | NLRP3 top-1000, row where `drug_name=='MCC950'` | grep row, read rank + leakage_class |
| P1-§5.2-LEAKAGE-COMP | Top-100 leakage class composition per target (NLRP3 78A/22B; CD73 100A; STING1-agg 6A/94B; STING1-seq 91A/9B) | [`audit/repurposing/F4_top100_leakage_composition.tsv`](../audit/repurposing/F4_top100_leakage_composition.tsv) | aggregator: see `data/dtsfm/scripts/classify_panel_pair_leakage.py` |
| P1-§5.2-ANCHOR-RANKS | Per-anchor rank table | [`audit/repurposing/F4_anchor_ranks.tsv`](../audit/repurposing/F4_anchor_ranks.tsv) | direct |
| P1-§5.2-SUMMARY | Aggregate §5.2 summary | [`audit/repurposing/F4_summary.json`](../audit/repurposing/F4_summary.json) | direct |

---

## §5.3 — AF3 structural verification of repurposing

| ID | Claim | Source | Recipe |
|---|---|---|---|
| P1-§5.3.1-AGGREGATES | Per-target × per-stratum AF3 iPTM/PAE means | [`audit/decoder_af3/F5_summary.tsv`](../audit/decoder_af3/F5_summary.tsv) (filter `target ∈ {NLRP3, CD73, STING1}`) | pandas filter |
| P1-§5.3.2-ANCHORS | Anchor compound table (3 targets × 1 anchor each) | F5_summary subset | filter `tag == 'gold_standard'` |
| P1-§5.3.3-COMPLEMENTARITY | v3 + AF3 complementarity finding (qualitative) | WIP §5.3.3 text | manuscript-level claim — confirm wording |
| P1-§5.3-CIFS | AF3 CIF files for representative cofolds | `audit/decoder_af3/af3_outputs/<id>/` (on Euler/ALPS; manifest in eval_summaries) | direct file inspection |
| P1-§5.3-N_COFOLDS | 1,520 AF3 cofold summary_confidences.json files | `audit/decoder_af3/af3_outputs/*/<id>_summary_confidences.json` | `find ... | wc -l` |

---

## §5.4 — De novo generative design

| ID | Claim | Source | Recipe |
|---|---|---|---|
| P1-§5.4.1-VOCAB | Decoder vocab 2,363 (MoLFormer 2,362 + [EOS]) | [`tokenizer/dtsfm_v3_decoder/`](../tokenizer/dtsfm_v3_decoder/) | `python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('tokenizer/dtsfm_v3_decoder', trust_remote_code=True); print(len(tok))"` |
| P1-§5.4.1-PARAMS | 27M decoder params (6 layers, d=512, 8 heads, FFN 2048) | [`audit/archival_checkpoint/checkpoint_manifest.tsv`](../audit/archival_checkpoint/checkpoint_manifest.tsv) row `decoder_v02_step50k` | direct |
| P1-§5.4.2-COHORT | Cohort 1,295 pairs across 16 targets | [`audit/decoder_af3/decoder_cohort.tsv`](../audit/decoder_af3/decoder_cohort.tsv) | `wc -l` (1,296 = 1,295 + header) |
| P1-§5.4.3-CASCADE | Filter cascade outcome | [`audit/decoder_af3/F5_results.tsv`](../audit/decoder_af3/F5_results.tsv) | per-filter count via pandas |
| P1-§5.4.3-STRONG | 12 STRONG + 5 MODERATE (17 total) | [`audit/decoder_af3/F5_3_strong_candidates.tsv`](../audit/decoder_af3/F5_3_strong_candidates.tsv) (18 rows = 17 + header) | `wc -l` |
| P1-§5.4.4-PAE | PAE-margin discriminating metric | [`F5_2_selectivity_reranked.tsv`](../audit/decoder_af3/F5_2_selectivity_reranked.tsv) + [`F5_anchor_drift.tsv`](../audit/decoder_af3/F5_anchor_drift.tsv) | direct |
| P1-§5.4.5-PER_TARGET | FLT3 5 / MAP2K2 3 / MAP2K1 2 / ALK 2 (STRONG) | [`F5_3_candidate_summary.tsv`](../audit/decoder_af3/F5_3_candidate_summary.tsv) | `groupby('target')` |
| P1-§5.4.6-DEC_FLT3_0502 | Headline candidate dec_FLT3_0502 (PAE-margin 7.36, iPTM 0.96, QED 0.64, Tan 0.15) | [`F5_3_strong_candidates.tsv`](../audit/decoder_af3/F5_3_strong_candidates.tsv) row | filter row |
| P1-§5.4.7-PORTFOLIO | 17-candidate portfolio table | [`F5_3_paper_headline_table.tsv`](../audit/decoder_af3/F5_3_paper_headline_table.tsv) | direct |
| P1-§5.4.A-NEG_PANEL | Per-target negative-selectivity profile | [`audit/decoder_af3/F5_2_named_offtarget_exact.tsv`](../audit/decoder_af3/F5_2_named_offtarget_exact.tsv) | direct |
| P1-§5.4-PHASE0 | Decoder phase 0 competence (10 targets STRONG/MODERATE) | [`audit/decoder_phase0_competence_map.csv`](../audit/decoder_phase0_competence_map.csv) | direct |

---

## §5.5.2 — atpE pilot (Paper 1 scope per §P.2)

| ID | Claim | Source | Recipe |
|---|---|---|---|
| P1-§5.5.2-CANDIDATES | 1,027 patent-novel atpE candidates | [`audit/biie_lmic/atpE_deliverable/atpE_final_deliverable.tsv`](../audit/biie_lmic/atpE_deliverable/atpE_final_deliverable.tsv) | `wc -l` minus header |
| P1-§5.5.2-CARDIAC | 94.5% predicted reduced cardiac liability | same file, filter `hERG_iptm < anchor_hERG_iptm` | pandas filter |
| P1-§5.5.2-TANIMOTO | 100% Tanimoto-novel (max 0.213, mean 0.098) | same file, column `max_tanimoto_vs_anchor_class` | pandas describe |
| P1-§5.5.2-CIFS | 5 representative AF3 cofold CIFs | [`audit/biie_lmic/atpE_deliverable/cifs/`](../audit/biie_lmic/atpE_deliverable/cifs/) | ls |
| P1-§5.5.2-FIGURE | 4-panel structural overlay figure | [`atpE_deliverable/figures/atpE_top4_overlays.pdf`](../audit/biie_lmic/atpE_deliverable/figures/) | direct |

---

## §C — Leakage verification (cross-cutting audit)

All §C numbers trace to [`audit/leakage_verification/`](../audit/leakage_verification/).
Reproduction recipes are documented in the respective scripts at `data/dtsfm/scripts/leakage_*.py`.

dtSFM v3 OOD posture is **pair-OOD + drug-OOD only** — no protein-sequence OOD claim is made.
The cluster-based train/val/test split is dataset hygiene, not a sequence-OOD assertion;
MMseqs2 protein-cluster leakage is therefore excluded from the active §C spec (preserved
in [`leakage_verification/_excluded_from_audit/`](../audit/leakage_verification/_excluded_from_audit/)
for due-diligence audit-trail purposes).

| ID | Claim | Source | Recipe |
|---|---|---|---|
| P1-§C.1-DRUG | drug-id overlap val 36% / test 37% (counts shared canonical SMILES between train and val/test); 63-64% of test drugs are novel by exact-SMILES match | [`drug_leakage.tsv`](../audit/leakage_verification/drug_leakage.tsv) | `python data/dtsfm/scripts/leakage_drug.py` |
| P1-§C.2-PROTEIN | protein-id overlap = 0% — confirms split disjointness at protein-id level (not a sequence-OOD claim) | [`protein_leakage.tsv`](../audit/leakage_verification/protein_leakage.tsv) | `python data/dtsfm/scripts/leakage_protein.py` |
| P1-§C.3-PAIR | exact (drug, protein) pair leakage = 0 (PASS) — strict pair-OOD validation | [`pair_leakage.tsv`](../audit/leakage_verification/pair_leakage.tsv) | `python data/dtsfm/scripts/leakage_pair.py` |
| P1-§C.5-TANIMOTO | Drug-novelty stratification by ECFP4 Tanimoto-bucket. **5K-query subsample per split (seed=42)**; mean max-Tan val/test = 0.71/0.71; ~37% exact-SMILES match in train; ~38% ≥0.9 near-duplicates; ~30% novel scaffold variants (<0.5); ~1-2% strictly novel (<0.3) | [`drug_tanimoto_buckets.tsv`](../audit/leakage_verification/drug_tanimoto_buckets.tsv), [`drug_tanimoto_summary.tsv`](../audit/leakage_verification/drug_tanimoto_summary.tsv), [`drug_tanimoto_per_drug.tsv`](../audit/leakage_verification/drug_tanimoto_per_drug.tsv) | `sbatch scripts/euler_leakage_tanimoto.slurm` (numpy-vectorized fast version with `--max_queries_per_split 5000`) |

---

## §S.6 — Affinity head calibration supplementary

Full S.6 audit trail in [`audit/affinity_calibration/`](../audit/affinity_calibration/) with the F6_phase1–phase35 TSVs.

| ID | Claim | Source |
|---|---|---|
| P1-§S.6.3-HEADLINE | Affinity headline Pearson per stratum | [`F6_phase3_summary.json`](../audit/affinity_calibration/F6_phase3_summary.json) |
| P1-§S.6.4-STRATUM | Stratum collapse | [`F6_phase3_metrics.tsv`](../audit/affinity_calibration/F6_phase3_metrics.tsv) |
| P1-§S.6.5-RECALIBRATION | Affine recalibration honesty | [`F6_phase35_metrics_before_after.tsv`](../audit/affinity_calibration/F6_phase35_metrics_before_after.tsv) |
| P1-§S.6.7-FUTURE | Future-work documentation | WIP §S.6.7 |
| P1-§S.6-SCATTER | Calibration scatter PNG | [`F6_phase3_calibration_scatter.png`](../audit/affinity_calibration/F6_phase3_calibration_scatter.png) |

---

## §F1 — Heldout-filter audit finding (pre-emptive disclosure)

| ID | Claim | Source |
|---|---|---|
| P1-§F1-FINDING | 3 of 24 "prospective eval" pairs were in training due to filter bug; documented and disclosed; §2 manuscript text reframed | [`audit/split_index/README.md`](../audit/split_index/README.md) §F1–F4 |

---

## Codex audit checklist (for orthogonal review)

When Codex audits the committed repo at the `paper1-pre-audit` git tag:

1. Read this doc end-to-end. Every claim ID should map to a file path that exists.
2. For each claim ID, run the listed reproduction recipe and report PASS / PARTIAL / FAIL.
3. Cross-check the WIP text for any numerical claim NOT in this doc — those are unverified claims (flag as audit gaps).
4. Report verdict counts in `audit/orthogonal_verification/codex_paper1_<sessionid>.md`.
5. Tag closure as `paper1-audit-closed` per the orthogonal-audit protocol (§G.3–G.5).

---

**Pending items expected to land before final audit:**
- P1-§3.4-SPEED — pending speed bench Euler GPU submission
- P1-§5.3-CIFS — representative CIFs need to be pulled from Euler/ALPS into audit/decoder_af3/
- Stratified §3 retrieval metrics by Tanimoto bucket (post-Tanimoto analysis): re-compute R@K per bucket using existing encoder, document in §3 supplementary.

**Excluded from audit (with rationale):**
- P1-§C.4-CLUSTER (MMseqs2 protein-cluster leakage) — not load-bearing because dtSFM v3 does not claim protein-sequence OOD. Measured for due-diligence; results preserved in [`leakage_verification/_excluded_from_audit/`](../audit/leakage_verification/_excluded_from_audit/).

**New finding (2026-05-16) — Species-ortholog leakage:**
- P1-§5.2-MCC950-ORTH — MCC950 ranks #3 for human NLRP3 in §5.2 retrieval. Class B labeling is technically correct (per-protein-idx exact-pair logic) but the encoder learned MCC950-class chemistry via the training pair `(MCC950, mouse Nlrp3 = uniprot:Q8R4B8)`. The §5.2 narrative should disclose this as transitive species-ortholog generalization. See [`audit/leakage_verification/FINDINGS.md`](../audit/leakage_verification/FINDINGS.md) §5.2 disclosure section for the recommended wording. Scope-limited §C.6 ortholog audit (paper-discussed compounds only, ~60–80 drugs) deferred to post-`paper1-pre-audit` work.

**Last updated:** 2026-05-16 by autonomous audit-execution session.
