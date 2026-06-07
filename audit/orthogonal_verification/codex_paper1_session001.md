# Codex Orthogonal Verification — dtSFM v3 Paper 1

*Pass 3 (paper1-audit @ 2691ae8, Git-LFS materialized). Reproduced from the
Codex session output for in-repo provenance. Disposition in
`audit_closure_paper1.md`.*

## 3.5 Headline summary

PASS=41, PARTIAL=5, FAIL=5, TODO=10. Most important finding: exact (drug,
protein) pair leakage is 0 in val/test, while drug-side overlap remains
~36–37%, so evaluation is strict pair-OOD but only partially drug-novel.

## 3.2 Leg-2 leakage measurements

| dimension | measured_value | claimed_value | verdict |
|---|---|---|---|
| pair leakage | val=0.0, test=0.0 | 0 / 0 | PASS |
| drug overlap | val=0.3579, test=0.3668 | 0.36 / 0.37 | PASS |
| protein overlap | val=0.0, test=0.0 | 0 / 0 | PASS |
| tanimoto summary (5k, seed42) | val mean=0.7105, test mean=0.7131 | ~0.71 | PASS |

## 3.3 Leg-3 filtered metrics

- Pair leakage λ=0 for val/test → no pair-level correction required.
- Drug overlap λ≈0.36–0.37 → spec-designated filtered view is the committed
  stratified Class C vs drug-seen metrics, not a single scalar-corrected R@K.

## Pass-3 verdict counts by section

- §2 data: 3 PASS / 2 FAIL (parser parenthetical) / 1 spurious-TODO — all
  resolved in closure.
- §3 encoder validation: all PASS (retrieval R@K, stratified, Tanimoto,
  affinity, speed).
- §5.1 safety, §5.2 repurposing, §C leakage, §S.6 affinity: PASS (one §5.2
  PARTIAL resolved in closure).
- §5.3 / §5.4 / §5.5: PASS where committed; AF3-CIF binaries + atpE figure →
  TODO; §5.4 decoder aggregates → TODO per spec §5.

Full per-claim table is in the Codex session log; dispositions tracked in
`audit_closure_paper1.md`.
