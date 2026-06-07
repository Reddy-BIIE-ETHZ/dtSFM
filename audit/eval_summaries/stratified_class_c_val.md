# §3 supplementary — stratified R@K by drug-side training-set membership

Computed 2026-05-16 from b3 epoch_010 checkpoint on the val split.

## Cohort

| | pairs | unique drugs | shared pool of unique proteins |
|---|--:|--:|--:|
| Class C (drug-OOD: drug_smiles NOT in training set) | 5,070 | 5,064 | 460 |
| drug-seen (drug_smiles IN training, paired with non-val proteins) | 2,930 | 2,880 | 460 |
| **total** | **8,000** | 7,944 (some overlap = 0 by construction) | 460 |

Cohort drawn by random sample of 8,000 val pairs (seed=42) from the full
val split (41,198 Class C pairs + 24,753 drug-seen pairs available).
Both strata retrieve against the **same** unique-protein pool, so R@K
denominators are comparable.

## Result (D→T retrieval over shared pool)

| stratum | R@1 | R@5 | R@10 | R@50 | R@100 | mean rank | median rank |
|---|--:|--:|--:|--:|--:|--:|--:|
| **Class C** (drug-OOD) | 20.7% | 40.2% | 49.2% | 71.2% | 80.8% | 57.2 | **10** |
| **drug-seen** | 42.2% | 73.0% | 80.2% | 89.9% | 93.5% | 20.3 | **1** |
| _random baseline_ | 0.22% | 1.09% | 2.17% | 10.9% | 21.7% | 230 | 230 |

## Reading

- **Class C generalization is real, ~half-strength of drug-seen.**
  Class C R@1 = 20.7% is **95× random**; R@10 = 49.2% is **22× random**.
  The encoder is not pattern-matching on memorized drug fingerprints —
  there is genuine drug-novelty generalization. But the gap to
  drug-seen (R@1 42.2% / median rank 1) is also real: drug-seen drugs
  benefit substantially from having been observed in training paired
  with non-val proteins.

- **Median rank 1 vs 10.** The most striking number is the median-rank
  gap. For half of drug-seen queries the encoder retrieves the correct
  target rank-1 out of 460 proteins. For half of Class C queries the
  median correct-target rank is 10. Both are far better than random
  (median 230), but the order-of-magnitude gap is the paper-relevant
  signal that memorization-leaning effects are present.

- **Interaction with §5.2 / §5.3 narrative.** §5.2 repurposing claims
  hold mostly on Class B (drug-seen-with-other-targets) hits, which are
  the drug-seen stratum here. §5.3's representative-table caveat
  (none of the rank-1 picks are Class C) is consistent: the encoder
  is stronger on drug-seen retrieval and the §5.2 repurposing screens
  surface those first.

## Reproduction

```
sbatch scripts/euler_eval_stratified_class_c.slurm     # Euler GPU, ~4 min
```

Script: `data/dtsfm/scripts/eval_stratified_class_c.py`.
Output: `audit/eval_summaries/stratified_class_c_val.csv`.
