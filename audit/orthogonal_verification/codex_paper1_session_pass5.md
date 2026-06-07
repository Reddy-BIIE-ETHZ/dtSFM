# Codex Orthogonal Verification — dtSFM v3 Paper 1 (Pass 5, GREEN)

*paper1-audit @ 617ee38 (de-LFS'd claim sources). Reproduced from the Codex
session output for in-repo provenance. Disposition in audit_closure_paper1.md.*

## 3.5 Headline summary

PASS=48, PARTIAL=3, FAIL=0, TODO=10. Most important finding: strict exact-pair
leakage is zero in val/test while drug overlap is ~36–37%, so chemistry-novel
(Class C) stratification is critical for fair interpretation.

## 3.2 Leg-2 leakage measurements (all PASS)

| dimension | measured (val / test) | claimed | verdict |
|---|---|---|---|
| pair_leakage | 0.0000 / 0.0000 | 0 / 0 | PASS |
| drug_leakage | 0.3579 / 0.3668 | 0.36 / 0.37 | PASS |
| protein_leakage | 0.0000 / 0.0000 | 0 / 0 | PASS |
| drug_tanimoto (5k, seed42) | mean 0.7105 / 0.7131 | ~0.71 | PASS |

## 3.3 Leg-3 filtered metrics

Pair/protein leakage = 0 → no correction. Drug overlap λ≈0.36–0.37 is non-zero;
the spec-designated filtered view is the committed Class-C-vs-drug-seen
stratification (`stratified_class_c_val.csv`), not a scalar-corrected R@K.

## 3.1 Verdict counts by section

- §2 data (8 claims): all PASS (dataset_stats.tsv + split_index).
- §3 encoder validation (10): all PASS.
- §5.1 safety (4), §5.2 repurposing (8): PASS except §5.2-MCC950-RANK PARTIAL
  (parser read a Source-cell parenthetical as a path; data present + correct).
- §5.3 (5): PASS / §5.3.2-ANCHORS PARTIAL (same parser artifact) / §5.3.1 +
  §5.3-CIFS TODO (Euler/ALPS-side, acknowledged §5).
- §5.4 decoder (11): §5.4.1/§5.4-PHASE0 PASS; aggregates TODO per spec §5;
  §5.4.2-COHORT PARTIAL (Euler-only cohort, effectively TODO).
- §5.5 LMIC (5): PASS except figure TODO.
- §C leakage (4), §S.6 affinity (5), §F1 (1): all PASS.

The 3 PARTIALs = 2 Source-cell parenthetical parser artifacts (files present,
numbers correct; parentheticals removed post-pass-5) + 1 Euler-side cohort
(effectively a §5 TODO). Zero science gaps.
