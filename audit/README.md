# Audit — can you trust these numbers?

Short answer: **a second, independent AI checked every number in the paper, and
none of them was wrong.**

Here's the idea. dtSFM was built by one AI agent (Claude Code) working with a
domain expert. So before publishing, a *different* AI — **Codex (OpenAI)**, with
a different architecture, different training, and **no access to the development
conversations** — was given only the committed files in this repository and asked
to independently re-derive every quantitative claim in the paper from the raw
data. Where it could reproduce a number, that's a **PASS**. Where it couldn't,
that's flagged. This is the same "orthogonal verifier" principle the paper uses
for the science itself (AlphaFold-3 independently checks every binding
prediction) — here applied to the *manuscript*.

## The result

**Paper 1 audit is CLOSED: 48 PASS · 3 PARTIAL · 0 FAIL.**

It took five rounds, but **no claim was ever found numerically wrong** — every
transient failure was a tooling glitch (a file-download hiccup), not a science
problem. The three PARTIALs are bookkeeping (a file path the auditor's parser
misread, and one large table left on the compute cluster); all are explained in
the closure record. Leakage — the thing that most often inflates ML results — was
measured directly and matched the paper exactly:

| What was checked | Result (validation / test) | Verdict |
|---|---|---|
| Same exact (drug, protein) pair seen in training? | 0% / 0% | ✅ none |
| Same drug seen (with a different protein)? | 35.8% / 36.7% | ✅ matches paper's 36% / 37% |
| Same protein/cluster seen in training? | 0% / 0% | ✅ none |
| How chemically similar are test drugs to training? | mean Tanimoto 0.71 | ✅ as reported |

## Verify it yourself

Every number in the paper is mapped to the exact file that produces it in
[`../docs/reproducing_paper_claims.md`](../docs/reproducing_paper_claims.md) — one
row per claim. Pick any claim, open the listed file, and check. The auditor's own
instructions are in [`AUDIT_SPEC_PAPER1.md`](AUDIT_SPEC_PAPER1.md); its verdicts
and our disposition of them are in [`orthogonal_verification/`](orthogonal_verification/).

## What's in this folder

**The audit itself**
- [`orthogonal_verification/`](orthogonal_verification/) — Codex's per-claim verdicts (`codex_paper1_session001.md`, final green pass `…_pass5.md`) and the developer's closure record (`audit_closure_paper1.md`).
- [`AUDIT_SPEC_PAPER1.md`](AUDIT_SPEC_PAPER1.md) — the instructions the auditor followed (the 3-part protocol + claim list).
- [`leakage_verification/`](leakage_verification/) — the four leakage measurements, the Class A/B/C novelty taxonomy (`FINDINGS.md`), and species-ortholog disclosures. The stricter cluster-leakage due-diligence is preserved under `_excluded_from_audit/` (measured for transparency; not used to support any claim).

**The evidence the claims rest on** (compact tables; big raw data is on Zenodo)
- [`decoder_af3/`](decoder_af3/) — generative-design results (§5.4): the full cascade, the strong/moderate candidates, selectivity, and the anchor-drift / tautology / negative-control guards.
- [`repurposing/`](repurposing/) — NLRP3 / CD73 / STING1 rankings and anchor tables (§5.2–§5.3).
- [`affinity_calibration/`](affinity_calibration/) — affinity-head calibration (§3.3 / §S.6).
- `klaeger2017/`, `safety_screen_*`, `safety_panel_*` — off-target safety screen (§5.1).
- `eval_summaries/` — retrieval + speed benchmarks (§3).
- [`split_index/`](split_index/) — the exact train/val/test split (reproducible bit-for-bit from `metadata_v3` + seed 42).
- [`archival_checkpoint/`](archival_checkpoint/) — SHA256 fingerprints of the released weights, so you can confirm the model on Hugging Face is the one in the paper.

> **A note on transparency.** This audit reports limitations as diagnoses, not
> caveats to hide. Where dtSFM disagrees with AlphaFold-3, where a candidate is
> chemically close to a known drug, where the affinity head is still preliminary —
> it's all here. The goal is AI-verifiable science.

> **Large files on Zenodo.** A few multi-MB per-compound tables (full affinity
> cohorts, the 522,776-compound library, the Klaeger K_d table) are deposited to
> Zenodo rather than committed here; the compact summaries they feed *are* here, so
> every headline number stays traceable. See the top-level `zenodo/` folder.
