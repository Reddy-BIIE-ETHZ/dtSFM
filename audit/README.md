# Audit — orthogonal AI verification (Paper 1, CLOSED)

dtSFM's numerical claims were independently re-derived by **Codex (OpenAI)** from
the artifacts committed to this repository, with no access to the development
sessions that produced them. This is the "orthogonal verifier" principle applied
to the *paper* rather than to a single prediction: a second AI, with a different
architecture and training, recomputes each claim from raw data and reports
agreements and discrepancies. It complements the **AlphaFold-3** structural
verification of individual binding predictions
([`docs/alphafold3_cofold_protocol.md`](../docs/alphafold3_cofold_protocol.md)).

## Result

**Paper 1 audit is CLOSED — 48 PASS / 3 PARTIAL / 0 FAIL.** It took five Codex
passes, but **no numerical claim was ever found wrong** — every transient FAIL was
an environment/tooling trap (Git-LFS materialization), not science. Leakage was
verified 100% across all four dimensions:

| Leakage dimension | measured (val / test) | claimed | verdict |
|---|---|---|---|
| Exact (drug, protein) pair | 0 / 0 | 0 / 0 | PASS |
| Drug-identity overlap | 35.8% / 36.7% | 36% / 37% | PASS |
| Protein / cluster overlap | 0% / 0% | 0% | PASS |
| Tanimoto (5K, seed 42) | mean 0.71 / 0.71 | ~0.71 | PASS |

## Contents

| Path | Description |
|------|-------------|
| `orthogonal_verification/codex_paper1_session001.md` | Codex's per-claim PASS/PARTIAL/FAIL verdicts (Leg 1 SA + CL). |
| `orthogonal_verification/codex_paper1_session_pass5.md` | Final green pass (FAIL = 0). |
| `orthogonal_verification/audit_closure_paper1.md` | Developer's disposition of every verdict; the closure record. |
| `AUDIT_SPEC_PAPER1.md` | The Codex audit instructions (3-leg protocol, claim list). |
| `../docs/reproducing_paper_claims.md` | Canonical claim → file-path map (one row per numerical claim). |
| `leakage_verification/` | Leg 2: the four leakage measurements + FINDINGS, Class A/B/C taxonomy, species-ortholog disclosures. The cluster-leakage due-diligence is preserved under `_excluded_from_audit/` (not used to support any claim). |
| `decoder_af3/` | §5.4 generative-design claim sources (F5 cascade results, strong/moderate candidates, selectivity, anchor-drift/tautology/negative-floor guards). |
| `repurposing/` | §5.2/§5.3 claim sources (NLRP3 / CD73 / STING1 top-1000 rankings, anchor ranks, anchor AF3 table, F42 results). |
| `affinity_calibration/` | §3.3 / §S.6 affinity-head calibration metrics + summaries. |
| `klaeger2017/` | §5.1 Klaeger 2017 panel (drug list). |
| `safety_screen_*`, `safety_panel_*` | §5.1 off-target safety-screen results. |
| `eval_summaries/` | §3 retrieval + speed-benchmark summaries. |
| `split_index/` | Train/val/test pair-index parquet + reproduction recipe (bit-exact from `metadata_v3` + seed 42). |
| `archival_checkpoint/` | Checkpoint SHA256 manifest (encoder + decoder + tokenizer) with HF URLs. |

## Files deferred to Zenodo (too large for git)

A few multi-MB intermediate tables are not committed here and will be deposited
to Zenodo with the candidate pools and AF3 structures:

- `affinity_calibration/F6_phase1_*_classified.tsv`, `F6_phase2_*_predicted.tsv`,
  `F6_phase1_pdbbind_within_dist.tsv` (per-compound affinity cohorts; 12–78 MB each)
- `affinity_calibration/F6_v3_train_drugs.tsv` (the 522,776-compound training library; derivable from public PDBbind v2020 + SAIR)
- `klaeger2017/klaeger_kdapp_long.tsv` (Klaeger 2017 published supplementary K_d^app table)

The compact summary/metric files those feed (`F6_phase3_metrics.tsv`,
`F6_phase3_summary.json`, `safety_panel_klaeger2017.tsv`, etc.) **are** committed
here, so the headline claims remain traceable from this repository.
