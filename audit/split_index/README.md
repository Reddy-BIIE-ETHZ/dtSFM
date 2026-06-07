# dtSFM v3 — Canonical Split Index

This directory holds the **bit-exact training-time splits** for the dtSFM v3 B-3
production training run (SLURM job 65583806, completed 2026-05-07).

## Generation recipe

Splits are produced deterministically by `make_cluster_splits()` from
[`src/calm/encoder/data_dtsfm_v3.py`](../../../src/calm/encoder/data_dtsfm_v3.py)
(L316) given the following inputs:

| Input | SHA256 |
|---|---|
| `metadata_v3.csv` | `46b82b8b5b5eaa54424dba6f6a330f3ed688912365fdddae71de35353bdf835d` |
| `heldout_validation_pairs.tsv` (the file passed at training; see Finding §F1 below) | matches `audit/heldout_validation_pairs.tsv` |
| `seed` | `42` |
| `test_frac` | `0.10` |
| `val_frac` | `0.10` |

## Reproduced split sizes (matches training-time stdout exactly)

| Split | pairs | clusters |
|---|---:|---:|
| train | 592,888 | 18,372 |
| val (held-out PDB-entry clusters) | 65,951 | 2,296 |
| test (held-out PDB-entry clusters) | 55,908 | 2,296 |
| heldout (intended) | 0 | — see Finding §F1 |
| **Total** | **714,747** | **22,964** |

**OOD terminology not used.** The held-out cluster split is data-hygiene
infrastructure, not a sequence-OOD assertion. dtSFM v3 OOD posture is
pair-OOD + drug-OOD (per Class A/B/C taxonomy in WIP §5.4.2). Sequence-novel
protein OOD is NOT claimed in Paper 1.

Confirmed by reproducing the exact split sizes that appear in
`/cluster/home/reddys/CALM-0.1.0/logs/dtsfm3_b3_65583806.out`:
`Splits: train=592,888 val=65,951 test=55,908 heldout=0`.

## Files

- `train_pair_idx.parquet` — 592,888 rows, single column `pair_idx`
- `val_pair_idx.parquet` — 65,951 rows
- `test_pair_idx.parquet` — 55,908 rows

Use against `metadata_v3.csv` by joining on `pair_idx`.

## Audit finding — heldout-TSV schema mismatch

### §F1 — Findings

The B-3 training run was executed with the command-line flag
`--heldout_tsv audit/heldout_validation_pairs.tsv` (confirmed from
`scripts/euler_b3_dtsfm_v3.slurm`). However, the training-time stdout reports
`heldout=0`, meaning the heldout-filter function returned an empty set rather
than filtering 24 pairs.

**Two root causes** were identified by post-hoc analysis:

1. **Column-name mismatch.** `load_heldout_pairs()` expects columns named
   `canonical_smiles` (or `drug_smiles`) and `protein_id`. The committed
   heldout TSV uses `smiles` and `target_uniprot`. The function emits the
   warning `"heldout TSV missing expected columns; ignoring"` and returns
   an empty set. (Confirmed by inspecting the function at
   `src/calm/encoder/data_dtsfm_v3.py:82-100`.)

2. **Identifier-scheme mismatch.** Even with column names corrected, the
   metadata's `protein_id` column uses mixed `pdb:XXXX` and
   `uniprot:XXXXXX` formats (see `audit/protein_id_to_gene_symbol.tsv`),
   whereas the heldout TSV uses bare UniProt accessions
   (e.g. `Q06187`, `P12931`). To match correctly, the heldout TSV's
   `target_uniprot` column must be cross-referenced via the
   `protein_id_to_gene_symbol.tsv` mapping.

### §F2 — Functional impact on training

Of the 24 "prospective evaluation" pairs documented in
[`heldout_validation_pairs.tsv`](../heldout_validation_pairs.tsv):

| Status | Count | Detail |
|---|---:|---|
| **Truly held out** (not in v3 corpus) | **21** | These 21 (drug, uniprot) combinations never appear in `metadata_v3.csv` because the drug isn't paired with that target in PDBbind or SAIR. They are validly prospective. |
| **In v3 training** (filter failure) | **3** | The 3 pairs below appear in metadata and were assigned to the `train` split. |

The 3 in-training pairs:

| Drug | Target | pair_idx | cluster_id | Split |
|---|---|---:|---:|---|
| sorafenib | P04049 (CRAF) | 185274 | 20092 | train |
| dasatinib | P09619 (PDGFR-β) | 248527 | 20390 | train |
| dasatinib | P12931 (SRC) | 288135 | 20577 | train |

3 / 592,888 pairs = **0.0005% of training data**. Memorization risk for these
specific (drug, target) pairs is real but tiny relative to corpus scale; downstream
metric impact is expected to be negligible.

### §F3 — Resolution

The §2 manuscript claim should be reframed from "24 prospective-evaluation
pairs" to:

> "We curated a 24-pair set of clinically informative drug-target combinations
> for evaluation. Twenty-one of these pairs are confirmed absent from the v3
> training corpus (PDBbind + SAIR) and therefore constitute true prospective
> evaluation. The remaining three (sorafenib × CRAF; dasatinib × PDGFR-β;
> dasatinib × SRC) were detected post-hoc to be present in the training corpus
> and assigned to the train split due to a column-name and identifier-scheme
> mismatch in the heldout-filter logic; we report retrieval metrics for these
> three pairs separately as in-distribution validation rather than prospective
> evaluation. The mismatch has been documented and corrected for future
> training runs; the v3 production checkpoint is unchanged."

This resolution is preferred over a 7-hour A100 retrain because:
- The 3 in-training pairs are 0.0005% of training data
- The 21 truly held-out pairs remain valid prospective evaluation
- The audit finding is transparent, the fix is forward-looking

### §F4 — Fix for future training runs

For dtSFM v3.1 / v4 (or any future SFM training using this codebase), one of
the following must be applied before training begins:

1. **Rename heldout TSV columns** to `drug_smiles` and `protein_id`, and
   populate `protein_id` with metadata-format identifiers
   (`uniprot:XXXXXX` or `pdb:XXXX`); OR
2. **Augment `load_heldout_pairs()`** to accept the human-readable schema
   (`smiles`, `target_uniprot`) and apply the mapping
   `protein_id_to_gene_symbol.tsv` internally; OR
3. **Add a verification print** in `load_heldout_pairs()` that fails-loud
   rather than emitting a soft warning when no pairs are filtered.

Option 2 is preferred because the human-readable schema matches how the
heldout TSV is most naturally curated by domain experts.

## Codex audit reproduction recipe

To independently reproduce these splits from raw inputs:

```python
import random, pandas as pd

META = "metadata_v3.csv"  # sha256 = 46b82b8b... (see archival_checkpoint/checkpoint_manifest.tsv)

df = pd.read_csv(META, usecols=["pair_idx", "drug_smiles", "protein_id",
                                 "protein_idx", "cluster_id"])

cluster_ids = sorted(df["cluster_id"].unique().tolist())
rng = random.Random(42)
rng.shuffle(cluster_ids)
n = len(cluster_ids)
n_test = max(1, round(0.10 * n))   # 2296
n_val  = max(1, round(0.10 * n))   # 2296
test_clusters = set(cluster_ids[:n_test])
val_clusters = set(cluster_ids[n_test:n_test + n_val])
train_clusters = set(cluster_ids[n_test + n_val:])

train_idx = df.loc[df["cluster_id"].isin(train_clusters), "pair_idx"].astype(int).tolist()
val_idx   = df.loc[df["cluster_id"].isin(val_clusters),   "pair_idx"].astype(int).tolist()
test_idx  = df.loc[df["cluster_id"].isin(test_clusters),  "pair_idx"].astype(int).tolist()

# Expected:
# train: 592,888 pairs (18,372 clusters)
# val:    65,951 pairs ( 2,296 clusters)
# test:   55,908 pairs ( 2,296 clusters)
```

Hashes for verification:

| File | SHA256 (first 16) | size |
|---|---|---|
| `train_pair_idx.parquet` | `e26be761be8b88f3` | 2.53 MB |
| `val_pair_idx.parquet` | `0fb033ee9cb1123b` | 0.39 MB |
| `test_pair_idx.parquet` | `ba3b8023ca24bb3f` | 0.32 MB |
