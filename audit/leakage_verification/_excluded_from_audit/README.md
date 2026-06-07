# Cluster (MMseqs2) leakage check — EXCLUDED from dtSFM v3 audit

This directory contains the MMseqs2-based cluster leakage measurement that
was run on 2026-05-16 against the v3 splits. **The result is preserved here
for due-diligence audit-trail purposes but is NOT load-bearing for any Paper
1 claim.**

## Why excluded

dtSFM v3 does **not** claim protein-sequence OOD. The cluster-based
train/val/test split in `audit/split_index/` is a dataset-hygiene
mechanism to avoid trivial pair memorization, not a claim that val/test
proteins are sequence-novel relative to train.

The dtSFM v3 OOD posture is exclusively:

- **Pair-OOD** — held-out (drug, protein) pairs, validated at 0% leakage
  via `pair_leakage.tsv` (PASS).
- **Drug-OOD** — held-out drug SMILES + chemical-novelty stratification by
  ECFP4 Tanimoto, validated via `drug_leakage.tsv` and
  `drug_tanimoto_*.tsv`.

Protein-sequence clustering is not part of the OOD claim and not required
to support §3 retrieval metrics. The cluster-leakage TSVs preserved here
document that we measured sequence-similarity between val/test and train
protein sets, but this is not a load-bearing audit verification.

## Reason for original investigation

The MMseqs2 cluster check was initially included in the §C leakage spec
as a remnant pattern from the Vibe Coding SFM paper (which DID claim
protein-OOD across multiple identity thresholds for tSFM/eSFM/mhcSFM/
crisprSFM/mir-SFM/dtSFM-VC). dtSFM v3 inherits the cluster-split
mechanism from VC's data infrastructure but does not inherit the OOD claim.

## Result (for the record)

At 100% sequence identity, ~61% of val proteins have an exact-sequence
match in train (driven by PDB chains being clustered per-entry rather
than per-sequence; e.g., all T4 lysozyme PDB entries are separate
clusters but share identical sequence). This finding confirms the
non-sequence-OOD nature of the cluster split and supports the §2
wording recommendation to remove "OOD" labeling from protein-cluster
descriptions.

See:
- `cluster_leakage_summary.tsv` — verdict per threshold
- `cluster_leakage_per_query_test.tsv` — per-test-protein max identity to train
- `cluster_leakage_per_query_val.tsv` — per-val-protein max identity to train

## WIP §2 wording update recommendation (approved 2026-05-16)

The current §2 text:

> "Cluster-based train/validation/test splits hold out entire protein
> clusters (2,296 OOD val + 2,296 OOD test) plus a curated set of 24
> drug–target pairs..."

Should be reworded to (no "OOD" terminology paper-wide; Class A/B/C
taxonomy unified across §3/§5.1/§5.2/§5.4 per WIP §5.4.2):

> "Train/validation/test splits hold out entire PDB-entry clusters
> (2,296 val + 2,296 test), providing pair-level held-out evaluation
> at scale. Held-out clusters are dataset-hygiene infrastructure; we
> do not claim sequence-novel protein generalization. Data composition
> across §3/§5.1/§5.2/§5.4 is reported under the Class A/B/C taxonomy
> defined in §5.4.2 (Class A = exact pair in training; Class B =
> drug+protein both in training, pair not; Class C = drug not in
> training). Test-set chemistry composition is separately characterized
> by ECFP4 Tanimoto distribution to nearest training drug (Methods §X).
> The 24 curated drug-target pairs support evaluation on specific
> clinical kinase inhibitors (21 confirmed absent from training corpus;
> 3 documented in-training per audit §F1)."
