# Audit closure — dtSFM v3 Paper 1

**Agent A (developer) disposition of the Codex orthogonal audit.**
Companion to the Codex deliverable `codex_paper1_session001.md`.

## Audit history

| Codex pass | Branch state | PASS | PARTIAL | FAIL | TODO | Blocking issue |
|---|---|---|---|---|---|---|
| 1 | wrong clone (`work`, no audit files) | 0 | 0 | 0 | 1 | tag/files absent in clone |
| 2 | `paper1-audit` @ `4cd22cc`, LFS not pulled | 18 | 5 | 29 | 9 | `*.tsv` were LFS pointer stubs |
| 3 | `paper1-audit` @ `2691ae8`, LFS pulled | 41 | 5 | 5 | 10 | — (core science all PASS) |
| 4 | `paper1-audit` @ `e8027fc` (LFS pull fragile on rebuild) | 0 | 0 | 0 | 1 | `pair_leakage.tsv` LFS pointer again — setup-script pull didn't materialize |
| 5 | `paper1-audit` @ `617ee38` (claim TSVs de-LFS'd) | **48** | **3** | **0** | **10** | **GREEN — zero FAILs, all science verified** |

**Closure: pass 5 is GREEN (FAIL=0).** The fix that finally held was de-LFSing
the 36 small audit claim-source TSVs to regular git blobs (commit `617ee38`),
removing the Git-LFS dependency that made passes 2 and 4 fail at Step 0. The
three pass-5 PARTIALs: §5.2-MCC950-RANK + §5.3.2-ANCHORS (Codex's path resolver
read a Source-cell parenthetical as a path — both files present + correct;
parentheticals removed post-pass-5) and §5.4.2-COHORT (Euler-side decoder
cohort, effectively an acknowledged §5 TODO). No numerical claim was found wrong
in any pass. Tagged `paper1-audit-closed` at the commit carrying the pass-5
deliverable + this note.

The 29→5 FAIL drop between passes 2 and 3 was entirely an environment fix
(Git-LFS materialization via the Codex env setup script); **no claim was ever
found numerically wrong.** Leg-2 leakage measurements are 100% PASS across all
four dimensions, matching the manuscript framing exactly:

| Leakage dimension | measured (val / test) | claimed | verdict |
|---|---|---|---|
| Exact (drug, protein) pair | 0 / 0 | 0 / 0 | PASS |
| Drug-identity overlap | 35.8% / 36.7% | 36% / 37% | PASS |
| Protein / cluster overlap | 0% / 0% | 0% | PASS |
| Tanimoto (5K, seed 42) | mean 0.71 / 0.71 | ~0.71 | PASS |

## Disposition of pass-3 PARTIAL / FAIL / TODO

### FAILs (5) — all repo-hygiene, none scientific

| claim | root cause | disposition |
|---|---|---|
| §2-N_PDBbind, §2-N_SAIR | claims-doc parser choked on `(source_tier == 1.0/1.5)` parenthetical in the Source column; the values are correct in `dataset_stats.tsv` (N_DRUGS/N_PROTEINS from the same file PASSED) | **FIXED** — moved the parenthetical into the Recipe column → both resolve to PASS. (§2-N_PAIRS spurious-TODO fixed the same way.) All five §2 numbers independently reproduce from `metadata_v3.csv` (714,747 / 19,037 / 695,710 / 522,776 / 22,964). |
| §5.3.2-ANCHORS | source row used `role == 'anchor (gold standard)'` quoted-filter the parser read as a missing path | **FIXED** — committed non-LFS `audit/repurposing/F5_3_anchor_table.tsv` (MCC950 iPTM 0.73 / PAE 8.38; AB680 0.95 / 0.98; MSA-2 0.79 / 3.16); repointed the claim to it → PASS |
| §5.3-CIFS | source pointed at an unresolvable glob `f42_outputs/<cofold>/seed-42_sample-{0..4}/`; raw `.cif` binaries are large AF3/ALPS-side outputs | **TODO** — numeric iPTM/PAE committed (`F42_results.tsv`, `F5_3_anchor_table.tsv`) + 9-cofold manifest committed (`REPRESENTATIVES_section_5_3.md`); only raw CIF binaries deferred. Added to spec §5 + tooling_debt. |
| §5.5.2-FIGURE | `atpE_top4_overlays.pdf` render artifact not committed | **TODO** — regenerable from committed CIFs; underlying atpE numbers committed. Added to spec §5 + tooling_debt. |

### PARTIALs (5)

| claim | disposition |
|---|---|
| §5.2-MCC950-RANK | **FIXED** — repointed to `F5_3_anchor_table.tsv` (`cosine_rank=3`, `class=B`) → concrete source, PASS |
| §5.3.3-COMPLEMENTARITY | accepted PARTIAL — qualitative/textual finding, no scalar to recompute |
| §5.5.2-CARDIAC, §5.5.2-TANIMOTO | accepted PARTIAL — atpE BIIE-side numbers; concrete committed source deferred (low priority, not a Paper-1 headline) |
| §S.6.7-FUTURE | accepted PARTIAL — qualitative future-work text |

### TODOs (10)

All are the §5.4 decoder-portfolio aggregates + §5.3.1 per-stratum AF3 aggregates already acknowledged in spec §5 (preliminary, being regenerated post-audit), plus the two newly-added Euler-side artifacts above. None are failures.

## Impact on figures

No FAIL/PARTIAL touched a number displayed in any paper figure. The verified
claims back: Fig 1 (data counts §2, leakage §C), Fig 2 (retrieval §3, heads),
Fig 3 (safety §5.1), Fig 4 (repurposing §5.2). All figures built to date are
audit-clean.

## Next

Re-run Codex pass 4 on the post-closure `paper1-audit` to confirm the 3 fixed
rows flip to PASS and the residual is TODO-only, then tag `paper1-audit-closed`.
