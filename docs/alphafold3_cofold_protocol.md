# AlphaFold-3 cofolding protocol (orthogonal structural verification)

Every dtSFM binding prediction in the paper — off-target hits, repurposing
shortlist, and generated molecules — is independently checked by **AlphaFold-3**.
AF3 shares no architecture, training data, or learned representation with dtSFM,
so structural agreement is genuine corroboration rather than circular
confirmation. Empirically, dtSFM cosine and AF3 confidence are essentially
uncorrelated (Pearson ≈ 0) on these candidate sets.

## Acceptance gates

| Gate | Criterion | Use |
|------|-----------|-----|
| **Binder gate** | iPTM ≥ 0.70 **AND** interface-PAE ≤ 5 Å | candidate is a plausible binder |
| **Anchor-grade gate** | iPTM ≥ 0.90 **AND** interface-PAE ≤ 1.67 Å | candidate matches an approved-drug-quality complex |

## Pipeline

1. **Inputs.** One AF3 job per (drug SMILES, protein sequence) pair. The drug is
   supplied as a ligand (SMILES → CCD/where needed), the protein as a single
   chain.
2. **MSA.** Protein MSAs are pre-generated with MMseqs2 and injected as
   `unpairedMsa`, with `pairedMsa=""` and `templates=[]`. (Do **not** pass an
   empty `unpairedMsa` string — AF3 then skips the MSA entirely; omit the key to
   trigger the data pipeline, or inject a real MSA as here.)
3. **Cofold.** Run AF3 inference (5 seeds × 5 models by default; the released
   numbers use the top-ranked model per job).
4. **Scoring.** Parse `iptm` and the minimum interface PAE from the AF3 summary;
   apply the gates above.
5. **Reporting.** Only candidates passing the relevant gate are reported. For
   selectivity claims, the approved-drug anchor is always cofolded alongside and
   compared *relative to* the anchor (AF3 iPTM saturates at ~0.7–0.9 on
   conserved-fold paralogs, so absolute values are not selectivity evidence).

## Compute

Cofolding (>2,000 jobs) was run on the **Alps** supercomputer (CSCS, GH200).
Raw cofold structures (CIF) are being deposited to Zenodo; a manifest will be
linked here on release.
