# Evaluation

Retrieval and head-level evaluation of the dtSFM encoder.

| Script | What it measures |
|--------|------------------|
| [`calm.encoder.eval_dtsfm_pool512`](../src/calm/encoder/eval_dtsfm_pool512.py) | pool-512 retrieval R@1 / R@5 / R@10 (1 positive + 511 random negatives, averaged over trials), bidirectional drug↔target |
| [`calm.encoder.eval_dtsfm_v3_quick`](../src/calm/encoder/eval_dtsfm_v3_quick.py) | fast smoke evaluation during/after training |
| [`calm.encoder.metrics`](../src/calm/encoder/metrics.py) | retrieval metric primitives (recall@K, MRR, rank) |
| [`calm.encoder.affinity_calibration`](../src/calm/encoder/affinity_calibration.py) | affinity-head calibration vs measured potencies |

## Leakage controls

The test set holds out **whole MMseqs2 protein-sequence clusters** at 80%
identity, so no test protein shares a cluster with any training protein. The
data loader ([`calm.encoder.data_dtsfm_v3`](../src/calm/encoder/data_dtsfm_v3.py))
additionally excludes a held-out validation pair list from the training subset.
Leakage across the protein/cluster/pair axes was verified to be zero.

## Protein-cluster ID/OOD axis

Retrieval is reported across clustering stringencies (e.g. mmseqs_040 / 060 / 080
and identity_100) to separate in-distribution from out-of-distribution
generalisation. For dtSFM, OOD retrieval is competitive with in-distribution — a
signature predicted by the molecular-recognition framework.
