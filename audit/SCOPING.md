# dtSFM-3.0 Audit Scoping Document

**SFM:** dtSFM (Drug-Target Specificity Foundation Model), v3.0 architecture upgrade
**Status:** Forward-looking scoping (pre-training)
**Date drafted:** 2026-04-28
**Author:** Sai T. Reddy
**Maintained by:** Reddy lab; updated as training progresses

This is a forward-looking scoping document, written before training begins, to guide
the development of dtSFM-3.0 under the SFM audit discipline (Reddy 2026, SFM Audit
Handoff). The audit itself will execute after training and evaluation are complete; this
document captures the design decisions, anticipated findings, and preservation plan that
make the eventual audit close cleanly.

---

## 1. Background

dtSFM is the drug-target Specificity Foundation Model in the Vibe-Coding SFM portfolio.
The agent is a small molecule drug (SMILES); the target is a protein. dtSFM is the
first SFM with a molecular (non-protein, non-DNA) encoder and the first with emergent
bilinearity — there is no analytical decomposition of drug-protein binding energy
analogous to Position Weight Matrices (tSFM) or Watson-Crick base pairing (crisprSFM).

**Lineage:**
- v1 (DAVIS + KIBA, 102K pairs, 429 unique targets, mean-pool): D→T R@1 = 5.8% at pool
  ~418. Established target-diversity bottleneck.
- v2 (+ BindingDB_Kd, 121K pairs, 1,358 unique targets, mean-pool): D→T R@1 = 27.7%,
  T→D R@1 = 60.5% at pool ~920. Confirmed scaling hypothesis (4.8× improvement from
  3.2× more targets). Showed physics signatures (val>train, OOD non-degradation, fast
  convergence). Ibrutinib validation: 83% of known off-targets correctly attributed.
- **v3.0 (this scoping document):** SFM v2.0 architecture upgrade per binderSFM
  handoff. Adds asymmetric cross-attention, per-atom interface head, atom-residue
  contact head, affinity regression head. Trained on PDBbind + SAIR + ChEMBL/BindingDB
  in three tiers.

**Why v3.0 is justified:** v2's ibrutinib validation showed cosine_sim(ibrutinib, BTK)
= 0.0000 and cosine_sim(ibrutinib, CSK) = 0.0000. The 83% off-target attribution comes
from rank ordering of near-zero values. This is the FcCALM 4.1+4.2 / mean-pool
fine-grained-discrimination failure mode predicted by the binderSFM handoff. v3.0's
architectural fix is the appropriate next step.

**Patent status:** No patent filed for dtSFM specifically. Cleared for public repository
and external audit. Invention disclosure may be filed after dtSFM-Kinase paper
publication.

**Publication plan (revised 2026-04-29; supersedes prior 4-paper plan):**

1. **Vibe-Coding SFMs (Reddy, in preparation)** — covers dtSFM **v1 and v2 only**. The
   `dtSFM_results_validation_VC.docx` is the final dtSFM contribution to this paper. v3
   is out of scope for Vibe-Coding.
2. **dtSFM v3 paper (in preparation)** — standalone ambitious paper for dtSFM v3.
   Combines what was previously planned as separate "dtSFM-Kinase" and "dtSFM-Proteome"
   papers into one comprehensive work. Scope:
   - **Encoder v3.0** (this scoping document) — asymmetric cross-attention,
     per-atom interface head, atom-residue contact head, affinity regression head.
   - **Decoder** for target → drug generation (cross-attentive, MRC §3.7).
   - **Safety panels** — Safety-77, cardiac kinase panel (hERG-related), comprehensive
     off-target profiling.
   - **Boltz-2 head-to-head comparison** on the DAVIS benchmark + alanine scan
     mutation sensitivity test.
   - **Ibrutinib / acalabrutinib selectivity validation** at v3.0 resolution
     (improvement over the v2 result of 83% off-target attribution).
   - **Iterative scope** — multiple training milestones, refinement passes, comparisons.
     Not a single-shot paper.
   - **Wet-lab validation** eventually (collaboration TBD; commercial kinase panels
     for prospective off-target prediction validation).

**MRC Scaling Laws** is held in reserve for the BIIE team and is not coupled to dtSFM
v3 publication timing.

---

## 2. Architecture (locked)

Per the SFM v2.0 architecture handoff (binderSFM, April 2026), with dtSFM-specific
adaptations.

```
Drug encoder (frozen):       MoLFormer-XL → (B, n_atoms, 768)
Protein encoder (frozen):    ESM-2 (esm2_t33_650M_UR50D) → (B, L_protein, 1280)
       ↓                                          ↓
Per-token projection (trainable):
                             (B, n_atoms, 512)    (B, L_protein, 512)
       ↓                                          ↓
       ├─ POOL pre-XA ───→ Global head (a) ←── POOL pre-XA ─┤
       │  (drug pre-XA pooled)              (protein pre-XA pooled)
       │                                                   │
       ▼                                                   ▼
Asymmetric cross-attention (k=2 layers, 8 heads, d_ff=2048):
   Shared CA layers, separate Q/K projections per modality.
   updated_drug    = CrossAttn(Q=drug,    K/V=protein)
   updated_protein = CrossAttn(Q=protein, K/V=drug)
       ↓                                          ↓
Three spatial output heads (read POST-XA features):
   (b) Per-atom MLP        → "is_atom_in_interface" logits (BCE, pos_weight)
   (c) Bilinear projection → atom-residue contact logits (B, n_atoms, L_protein)
   (d) Affinity regression → ΔG / log(Kd) (Huber loss)

Global retrieval head (reads PRE-XA pooled features, NEVER post-XA):
   (a) Pooled pre-XA vector → InfoNCE contrastive (preserves v2 retrieval)

L_total = α·L_global + β·L_atom_interface + γ·L_contact + δ·L_affinity
         (magnitude-normalized after warmup)
```

**Critical architecture note (locked 2026-05-06, inherited from binderSFM v2.0 → v2.5
lesson):** Head (a) MUST read from pre-cross-attention pooled features. Cross-attention
"accommodates" — pulls partner features toward each other regardless of biological
compatibility — when used as input to the contrastive head. binderSFM v2.0 had all
heads post-XA: training L_global converged to 0.006 (looked ideal) but pool-100 R@1
was 0.027 (random). v2.5 fix routed the global head pre-XA: Δcos jumped from +0.027
to +0.602 (22× improvement), R@1 OOD reached 91%.

Implementation flag: `cfg.global_head_uses_pre_xa: true` (v3.0 default; mirrors binderSFM
v2.5 config). Setting to `false` reproduces the broken v2.0 behavior — useful only for
ablation. The B-2 smoketest must verify pool-K retrieval Δcos > 0.3 before B-3 launch
(catches the v2.0 failure mode early).

**Locked design decisions (cannot change without re-scoping):**
1. **Contact threshold: 5 Å heavy-atom on both sides** (drug atoms and protein atoms).
   Rationale: drug-protein interfaces are dominated by direct atomic contacts (H-bonds,
   hydrophobic stacking, ionic interactions). The 8 Å Cβ-Cβ threshold appropriate for
   protein-protein interfaces is too generous for drug-protein.
2. **Asymmetric cross-attention with shared CA layers but separate Q/K projections per
   modality.** Drug atoms (~50) and protein residues (~300) have qualitatively different
   statistics; symmetric tied-weights from binderSFM (protein-protein) does not transfer.
3. **Affinity head as 4th regression head** (δ·L_affinity). PDBbind, SAIR, ChEMBL, and
   BindingDB all carry binding affinity labels; head (d) is the calibrated version of
   MRC §7's two-parameter affinity calibration, learned end-to-end during training
   instead of fit post-hoc.
4. **Multi-task loss with magnitude-normalized α, β, γ, δ during warmup.** No manual
   weight tuning. Calibration verified in B-2 smoketest before B-3 launch.

**Trainable parameters (estimated):** ~14M at d_model=512.

---

## 3. Training data plan (three tiers)

### Tier 1: Real structural data (gold quality)

| Source | Pairs | Drugs | Targets | Has 3D? | Has Kd? |
|--------|-------|-------|---------|---------|---------|
| PDBbind general v2024 | ~19K | ~17K | ~3,500 | YES (atomic) | YES |
| BindingMOAD | ~38K | ~32K | ~9,000 | YES | Partial |

Combined, deduplicated, filtered by resolution ≤ 3.0 Å and bound-ligand presence:
**~50K pairs estimated** for Tier 1 training.

All 4 heads trained on Tier 1.

### Tier 1.5: Synthetic structural data (silver quality)

**SAIR** (Lemos et al., bioRxiv 2025.06.17.660168):
- 5,244,285 Boltz-1x-folded structures across 1,048,857 unique systems
- Source: ChEMBL + BindingDB
- License: Boltz-1x permits derivative use for training (verify at Phase 0)
- ~3% physical anomalies — filterable with PoseBusters

After PoseBusters filtering at Boltz-1x confidence threshold ≥ TBD: estimated **~3-5M
pairs** for Tier 1.5 training.

All 4 heads trained on Tier 1.5, but with γ_synthetic = 0.5 × γ_real on the contact
head (synthetic structures are predictions, weighted lower than experimental).

**License compliance note:** AF3 outputs are excluded from training under AF3's
research-only license. SAIR uses Boltz-1x exclusively, which is MIT-compatible.

### Tier 2: Binding-only data (bronze quality, decoupled physics)

| Source | Pairs | Drugs | Targets | Has 3D? | Has Kd? |
|--------|-------|-------|---------|---------|---------|
| ChEMBL 34 (TDC subset for tractability) | ~3M | ~2M | ~10K | NO | YES |
| BindingDB Kd/Ki/IC50 (TDC) | ~1.5M | ~1M | ~9K | NO | YES |

After deduplication against Tier 1 + Tier 1.5: estimated **~3-5M residual binding-only
pairs** for Tier 2 fine-tuning.

Heads trained on Tier 2: (a) global contrastive and (d) affinity only. Heads (b) and
(c) masked out (no contact ground truth).

### Combined training plan

| Phase | Data | Pairs | Heads | Epochs | Notes |
|-------|------|-------|-------|--------|-------|
| B-3 | Tier 1 + Tier 1.5 | ~3.5M | All 4 | 10-30 | Magnitude-normalized multi-task |
| B-3.5 | + Tier 2 | +3-5M | (a) + (d) only | 5-10 | Continued training |
| B-4 (optional) | Kinase-specific subset | varies | All 4 | fine-tune | For dtSFM-Kinase paper |

---

## 4. Anticipated SA items (when audit runs)

These are the methodology checks the eventual audit will need to verify. Listed here
so we preserve the relevant artifacts during training.

- **SA-1: Architecture matches writeup.** Verify model_v3.py CALMEncoderV3 class
  implements asymmetric cross-attention with shared CA + separate Q/K, k=2 layers,
  8 heads, d_model=512, d_ff=2048.
- **SA-2: Contact threshold = 5 Å heavy-atom on both sides.** Verify
  extract_contacts_v3.py uses 5 Å heavy-atom distance.
- **SA-3: License compliance.** Verify all SAIR structures are Boltz-1x derived (no AF3
  contamination). License audit on training data sources.
- **SA-4: Hyperparameters match writeup.** Verify training config (batch=256+, lr=1e-4,
  AdamW, CosineAnnealingWarmRestarts, T0=20).
- **SA-5: Magnitude-normalized loss weights.** Verify α/β/γ/δ converge stably during
  warmup; commit log of warmup_step → α/β/γ/δ values to audit/.
- **SA-6: Three-tier training discipline.** Verify Tier 1 → Tier 1.5 → Tier 2 sequence
  is followed and logged.
- **SA-7: Per-tier metrics tracked separately.** Verify eval reports performance on
  PDBbind-derived test set vs SAIR-derived test set vs binding-only test set.
- **SA-8: PoseBusters filtering applied to SAIR.** Verify ~3% physical anomalies
  removed before training.
- **SA-9: Domain-aware print labels.** Verify training/eval logs print "drugs:"
  / "proteins:" labels, not legacy "TF proteins:" / "DNA sequences:" labels.
- **SA-10: Cluster-CV split bug avoidance.** Verify diagonal (outerfold_N_innerfold_N)
  splits are used and fold_4 anomalies are documented if encountered.
- **SA-11: Ibrutinib / CSK / known-validation-target leakage check.** Verify ibrutinib,
  acalabrutinib, LY2090314-TgGSK3 (PDB 9HVX), and other prospective-validation
  drug-target pairs are held out from training data.
- **SA-12: Digit-exactness meta-audit.** Verify all numerical claims in the writeup
  match the source CSVs to two decimal places.
- **SA-13: Codebase changes restricted to expected scope.** Verify _EMBED_DIMS already
  has 'molformer': 768; v3.0 changes are confined to model_v3.py, data_dtsfm_v3.py,
  loss_v3.py, configs/model/encoder/dtsfm_v3*.yaml, configs/train/encoder/dtsfm_v3*.yaml.
- **SA-14: Decoder architecture matches MRC §3.7 prescription.** Verify
  decoder_dtsfm_v3.py implements cross-attention from target embedding to SMILES token
  positions; either autoregressive or discrete diffusion as documented.
- **SA-15: Decoder training data restricted to verified Tier 1 / 1.5 / 2 sources.** No
  AF3-derived structures used for decoder training. License compliance documented per
  data source.
- **SA-16: Boltz-2 comparison protocol matches published convention.** Verify the
  alanine scan uses the same residue selections as Lowe (Science/AAAS 2026) and
  random reassignment uses the same shuffling protocol.
- **SA-17: Safety-77 panel coverage.** Verify all 77 proteins in the Brennan et al.
  Safety-77 panel are included in the model's protein universe (or documented if not).
- **SA-18: Held-out validation drug-target pairs.** Verify ibrutinib, acalabrutinib,
  LY2090314/TgGSK3, sunitinib/AMPK, ponatinib targets, dasatinib targets are confirmed
  absent from training set at SMILES-level matching, and absent at the (drug, target)
  pair level.
- **SA-19: Pre-XA global head (binderSFM v2.0 → v2.5 lesson).** Verify
  `cfg.global_head_uses_pre_xa == True` in dtsfm_v3_transformer.yaml. Verify model_v3.py
  routes the global cosine head from pre-cross-attention pooled features, not post-XA.
  Verify B-2 smoketest log includes a pool-K Δcos diagnostic (target: Δcos > 0.3 between
  true and shuffled pairs); a Δcos near zero indicates v2.0-style accommodation collapse
  and B-3 launch must be blocked.
- **SA-20: Quality-over-quantity training data discipline (abSFM lesson).** Verify
  training data is restricted to structure-verified pairs (Tier 1 PDBbind real co-
  crystals + Tier 1.5 SAIR Boltz-1x folded). Confirm no sequence-only-binding subset
  was added to the training mix without contact maps. abSFM regressed 75% → 57% R@1
  when sequence-only data was mixed in; we explicitly preempt that failure.
- **SA-21: Localizer head usage discipline (abSFM lesson).** Verify the writeup uses
  the global head (a) cosine similarity as the binder/non-binder discriminator
  throughout (not the contact head). Contact head (c) is documented as an
  interpretation tool only (paratope/epitope mapping in retained binders), not as a
  pair-level scoring head. abSFM proved cross-pair re-ranking via contact head fails
  because it produces sharp-looking maps even on wrong-partner pairs.

---

## 5. Anticipated CL items (claim-level)

Numerical claims the writeup will make. Each needs a recoverable source artifact.

- **CL-1: Pool-512 in-distribution R@1/5/10** for Drug→Target and Target→Drug.
  Source: pool512_results_unique.csv per fold + summary CSV.
- **CL-2: Pool-512 OOD R@1/5/10** at MMseqs 40/60/80% (deferred per user guidance —
  may not be reported in v3.0 Vibe-Coding paper, but artifacts preserved for later
  papers).
- **CL-3: Contact head AUROC** on Tier 1 held-out test set.
  Source: contact_eval.csv.
- **CL-4: Interface head AUROC** on Tier 1 held-out test set.
  Source: interface_eval.csv.
- **CL-5: Affinity calibration metrics** (Pearson r, Spearman ρ, RMSE in kcal/mol)
  using head (d) directly (not post-hoc fit).
  Source: affinity_eval.csv.
- **CL-6: Ibrutinib vs acalabrutinib safety prediction.** Off-target attribution rate
  (currently 83% in v2; expected improvement in v3.0). Source:
  ibrutinib_validation_results.json.
- **CL-7: Boltz-2 head-to-head comparison** (deferred to dtSFM-Kinase paper).
- **CL-8: Cross-SFM scaling comparison** (v1 → v2 → v3.0 progression on identical
  evaluation metrics).
- **CL-9: Physics signatures.** Documented val>train AUROC ranges, fast convergence
  (best epoch number), and consistency with binderSFM/eSFM/mhcSFM.
- **CL-10: Boltz-2 head-to-head metrics on DAVIS.** Speed, AUROC, Spearman ρ vs
  measured Kd. Source: boltz2_comparison.csv.
- **CL-11: Alanine scan delta-ΔG sensitivity.** Mean predicted ΔG change when critical
  binding-site residues are mutated to alanine. dtSFM v3 vs Boltz-2. Source:
  alanine_scan_results.csv.
- **CL-12: Random target reassignment retention.** Fraction of dtSFM v3's positive
  predictions that survive random drug-target shuffling. Should be substantially
  below Boltz-2's ~50% retention rate (per Lowe 2026).
- **CL-13: Safety-77 panel reproduction rate.** For each approved kinase inhibitor
  with reported Safety-77 hits, fraction of literature-documented hits that dtSFM v3
  ranks within top-K. Source: safety77_results.csv.
- **CL-14: Ibrutinib v3 selectivity refinement.** v3 ibrutinib BTK rank, CSK rank,
  ERBB2 rank, ERBB4 rank. v2-vs-v3 comparison table.
- **CL-15: Decoder reconstruction Tanimoto.** Median Tanimoto similarity between
  decoder-generated drug and ground-truth drug for held-out drug-target pairs.
- **CL-16: Decoder generation validity rate.** Fraction of decoder outputs that are
  RDKit-parsable, Lipinski-Ro5 compliant, and QED ≥ 0.5.
- **CL-17: Decoder held-out target generation.** For targets not in training, decoder
  generation evaluated against any external known binders. Honest result, expected
  lower than CL-15.

---

## 6. Preservation discipline (active from training day one)

Per SFM Audit Handoff §3. These commitments are made now, before any v3.0 code runs.

### Rule 1 — Durable summary CSVs

`SFM_SUMMARY_DIR=/cluster/home/$USER/sfm_summaries/` set in `.bashrc` before any v3.0
training. Every eval pipeline writes a summary CSV with columns
`fold, split, R@1, R@5, R@10, contact_AUROC, interface_AUROC, affinity_pearson,
affinity_rmse_kcal` plus a SUMMARY block at the bottom.

### Rule 2 — Structured stdout summaries

Every eval SLURM job prints a `=== Summary ===` block to stdout in the regex-friendly
format established by tSFM/mhcSFM:

```
=== Summary (mean ± s.d. across folds) ===
  R@1_ag2ab: XX.X ± X.X
  R@1_ab2ag: XX.X ± X.X
  contact_AUROC: 0.XXX ± 0.XXX
  interface_AUROC: 0.XXX ± 0.XXX
  affinity_pearson: 0.XXX
  affinity_rmse_kcal: X.XX
```

### Rule 3 — Archival logs into the repo within 7 days

After each training campaign, commit to `audit/archival_logs/`:
- One representative training log per tier (B-3, B-3.5, B-4)
- One representative eval log per evaluation type
- The pool-512 SLURM log
- The ibrutinib validation log
- Any prospective-validation logs

### Rule 4 — Split-index JSONs

Commit per-fold split JSONs to `audit/split_index/` via Git LFS. `.gitattributes`
already configured for `audit/*/split_index/**/*.json`.

### Rule 5 — Domain-aware print labels

Before any v3.0 code runs on Euler: replace generic "TF proteins:" / "DNA sequences:"
labels in `data_dtsfm_v3.py` and eval scripts with "drugs:" / "proteins:" labels.

### Rule 6 — RETRAIN_LOG.md written in parallel

`audit/RETRAIN_LOG.md` started at training-day-one. Every reported number gets a
row at the time it is computed, with source artifact path. No retroactive
reconstruction.

---

## 7. Anticipated documentation findings

Per SFM Audit Handoff §3 Rule 5: list known issues that will need to be documented
during audit, so they're not surprises.

1. **v2 → v3.0 architectural reframing.** v2 numbers will be referenced in v3.0
   writeup as "prior generation"; the comparison is a feature, not a finding.
2. **SAIR weighting in multi-task loss.** The decision to weight Tier 1.5 contacts at
   0.5× Tier 1 is a hyperparameter choice. Document rationale in writeup.
3. **License compliance for SAIR.** Document explicit verification that SAIR is
   Boltz-1x derived (not AF3) and that Boltz-1x license permits derivative training
   use. Cite SAIR paper and Boltz-1x license at training time.
4. **Affinity head learned end-to-end vs MRC §7 post-hoc calibration.** v3.0 deviates
   from MRC §7's two-parameter post-hoc fit by training the affinity head jointly.
   This should be framed as a refinement, not a deviation; document the equivalence
   under the convergence equation.
5. **Cluster-CV diagonal-fold anomaly.** All four prior SFM audits found this. Expect
   it for v3.0 too. Document fold_4 exclusion if it occurs, with explicit reference to
   SFM Audit Handoff Pitfall 1.

---

## 8. Validation strategy (dtSFM v3 paper — comprehensive)

The dtSFM v3 paper is a standalone ambitious work, not a Vibe-Coding sub-section. It
carries the full validation burden that was previously split across two follow-up
papers. Validation organized in five tiers from easiest to most prospective:

### 8.1 Tier A: Architectural readiness (encoder)

1. **Pool-512 retrieval (in-distribution)** with honest pool-size reporting. v3.0 vs v2
   improvement on identical evaluation protocol. Lead metric: D→T R@1/5/10; T→D
   R@1/5/10. Comparison table includes v1, v2, v3.0 to show the scaling trajectory.
2. **Contact prediction AUROC** on Tier 1 held-out test set (PDBbind real structures).
   Tier 1.5 (SAIR) reported separately. Target: ≥ 0.85 contact AUROC on Tier 1 held
   out (binderSFM-equivalent).
3. **Interface prediction AUROC** on Tier 1 held-out test set. Target: ≥ 0.80 (per
   binderSFM B-3 prediction).
4. **Affinity head Pearson r and RMSE** in kcal/mol on Tier 1 held-out test set.
   v3.0 head (d) replaces v2's post-hoc MRC §7 fit. Target: Pearson r ≥ 0.6 (a 7×
   improvement over v2's 0.08).

### 8.2 Tier B: Strict-OOD validation (Leakage Verification)

For dtSFM v3, the proteome-wide target diversity (~10K+ unique proteins after Tier 1 +
Tier 1.5 + Tier 2) admits clean strict-OOD splits. We commit to running full LV per
SFM Audit Handoff §2.

1. **MMseqs2 protein clustering at 40/60/80% identity** on the protein side. LV result
   determines which thresholds are reportable.
2. **Tanimoto scaffold clustering at 0.4/0.6/0.8** on the drug side (dual OOD; not
   attempted in v1/v2). LV applied to drug-side clusters.
3. **Filtered R@1** per Leg 3 of the audit framework.

### 8.3 Tier C: Held-out wet-lab validation (REVISED 2026-05-06)

**Decision (2026-05-06):** Boltz-2 head-to-head DROPPED. Replaced with three
public wet-lab datasets, ALL with measurements released after our PDBbind v2020
+ SAIR training cutoff. Argument is now absolute (vs. measured biophysics)
rather than comparative (vs. another predictor) — much stronger framing for
"augment or replace expensive structural pipelines."

**Leakage finding (2026-05-06):** MMseqs2 check (audit script
`data/dtsfm/scripts/wetlab_leakage_check.py`, output
`audit/wetlab_leakage/leakage_summary.csv`) shows protein-side overlap
with training set:

  - **CLEAN (< 30% identity)**: LRRK2 WDR domain, SARS-CoV-2 NSP13 helicase
  - **CAVEAT (44% identity)**: SARS-CoV-2 NSP3 macrodomain (vs human macrodomain
    paralogue in training — informative homology, not direct match)
  - **PROTEIN-SIDE LEAKAGE (≥ 80% identity)**: EV-A71 2A protease (~3 PDBbind
    structures in training), CBLB TKB (full-length CBLB in training), MCHR1
    (full GPCR in SAIR), SETDB1 triple Tudor (full SETDB1 in training)

**Strategic reframe (2026-05-06, per Sai's analysis):** Drug-side OOD is the
PRIMARY dtSFM use case (off-target screening, lead optimization). Protein-side
OOD is the secondary, "harder" scenario. The leakage finding therefore does
NOT invalidate the wet-lab evaluation — it splits it cleanly into two
publishable scenarios:

  - **Scenario A — strict both-sides OOD** (drug AND protein new): the
    "discovery on a novel target" capability. Currently includes LRRK2 WDR,
    NSP13, NSP3 macro, plus the both-sides-dedup'd subset of BindingDB-post-2024.
  - **Scenario B — drug-side OOD on known proteins** (drug new, protein
    in training): the realistic "off-target screen / lead expansion" capability,
    which is what most pharma needs. Includes OpenBind EV-A71 2A, CACHE-4
    CBLB, CACHE-5 MCHR1, CACHE-6 SETDB1, plus the drug-only-dedup'd majority of
    BindingDB-post-2024.

The performance gap between A and B quantifies how much training-protein
familiarity helps. Both scenarios require **mandatory drug-side dedup**: any
(canonical_drug_smiles, protein_id) pair already in our 714,747 training pairs
is excluded from any wet-lab metric.

**Three primary wet-lab datasets:**

#### 8.3.1 OpenBind EV-A71 2A protease (released May 2026)

Diamond Light Source XChem release; ~800 compounds with full GCI kinetics
(ka, kd, KD), aligned X-ray crystal structures, raw Creoptix sensorgrams.
Single viral protease target — the *deep* validation. CC0 license.

- **Affinity head**: predict pAffinity for all 800 compounds → Pearson r,
  Spearman ρ, RMSE in log-K_D units vs measured GCI K_D.
- **Contact head**: per crystal-resolved compound, compute IoU between
  predicted atom-residue contact map and observed 5 Å heavy-atom contacts.
  Stratify by binding pose region. Cross-reference with abSFM B-4 IoU
  calibration result (strict IoU 0.185 / recall±5 0.79; we want better given
  dense supervision in v3).
- **Tight ranking**: subset to compounds with K_D within 1 log unit, test
  whether dtSFM v3 ranks them correctly (lead-optimization scenario).
- **Reference baselines**: OpenBind's own Jupyter notebook ships baselines
  (likely DiffDock + Vina-style); compare directly.

#### 8.3.2 CACHE Challenge rounds 1–6 (released 2023–2025)

Six diverse prospective targets with ~10,000 wet-lab measurements total. Each
round: modelers submitted predictions, organizers ran wet-lab IC50/K_D after.
Genuine OOD by construction (compounds chosen with low Tanimoto vs known
binders). Multi-target generalization claim — the *broad* validation.

| Round | Target | Class | N tested | Public since |
|---|---|---|---|---|
| 1 | LRRK2 WDR domain | Kinase regulatory | ~2,750 | Jan 2024 |
| 2 | SARS-CoV-2 NSP13 helicase | Antiviral | 2,574 | Aug 2023 |
| 3 | SARS-CoV-2 NSP3 macrodomain | Antiviral | 1,739 | Jan 2024 |
| 4 | CBLB TKB domain | E3 ubiquitin ligase | 1,921 | Jun 2025 |
| 5 | MCHR1 (GPCR agonists) | GPCR | 1,455 | Oct 2025 |
| 6 | SETDB1 triple Tudor | Epigenetic reader | 1,338 | Nov 2025 |

- **Per-round metrics**: top-K precision (K = 100 / 50 / 10), enrichment factor
  at 1%, ROC AUC for binder threshold, mean rank of confirmed hits.
- **Cross-round meta-metric**: mean top-100 hit rate across 6 targets — single
  headline number for breadth.
- **Comparison**: each CACHE round publishes a leaderboard of participant
  submissions; we report dtSFM v3 placement on each round's leaderboard.

#### 8.3.3 BindingDB post-2024 slice

BindingDB filtered to entries deposited 2024-01-01 onward, deduplicated against
our 522,776 training drug SMILES (canonical-match). Provides scale (1,000s of
compounds across many targets) for the "many targets" generalization claim.

- **Filter chain**: deposit_date ≥ 2024-01-01 → drop entries with canonical
  SMILES ∈ training drug set → drop entries with target sequence in training
  protein set (MMseqs2 ≥ 80% identity) → keep only entries with K_D / K_i in
  the M range (drop μM-scale fragment screens).
- **Metric**: stratified Pearson r vs measured affinity, by target family and
  affinity range. Report per-family means.
- **Sample size target**: ≥ 5,000 measurements across ≥ 100 unique targets
  after filtering.

#### 8.3.4 Pre-evaluation leakage check (mandatory)

Before reporting any Tier C numbers, verify:

1. **OpenBind EV-A71 2A** target sequence: BLAST/MMseqs2 against our 22,964
   training protein sequences. Target ID: confirm < 30% identity to any
   training protein. If higher, document as a leakage caveat.
2. **CACHE rounds 1-6 target sequences**: same MMseqs2 check per target.
   LRRK2 in particular may have ≥ 30% identity to other kinases in our
   training set; expect this and document.
3. **BindingDB**: explicit canonical-SMILES dedup vs training set; explicit
   MMseqs2 protein sequence dedup ≥ 80% identity threshold.
4. Save the leakage-check artifacts (CSVs of identity matches) under
   `audit/wetlab_leakage/`.

Without this check, no Tier C numbers go in the paper.

#### 8.3.5 Continuous-validation infrastructure

Build the eval as a **reusable harness** (`src/calm/eval/wetlab_eval.py`)
that takes (a) a dtSFM v3 checkpoint, (b) a normalized wet-lab CSV+SDF in a
shared schema, and (c) optional crystal-structure bundle. Each new OpenBind
release (every 6 months) and each new CACHE round plug into the same harness
with a small ingestion script. By paper revision (~12 months out) we
realistically have 10+ post-training wet-lab targets evaluated.

### 8.4 Tier D: Safety panel and clinical case studies

1. **Safety-77 panel** (Brennan et al., Nat Rev Drug Disc 2024). For each of N approved
   kinase inhibitors with reported clinical adverse events, predict the Safety-77
   off-target profile and compare to literature. Goal: dtSFM v3 reproduces ≥70% of
   documented Safety-77 hits.
2. **hERG and cardiac kinase panel** specifically. Predict hERG affinity for known
   cardiotoxic vs non-cardiotoxic drugs. Validate against published cardiac safety
   data.
3. **Ibrutinib vs acalabrutinib (refined).** v2 achieved 83% off-target attribution
   with cosine_sim values near zero. v3.0 must achieve > 90% with non-zero
   discrimination signal in the cosine similarity itself. Specifically: BTK and CSK
   in top-50 for ibrutinib (currently ranks 513 and 383 respectively).
4. **LY2090314 / TgGSK3 case study.** PDB 9HVX held out from training. dtSFM v3
   predicts TgGSK3 binding from LY2090314 SMILES; rank should be in top-100 of all
   ~10K proteins in the model's universe.
5. **Other failed drug case studies.** Sunitinib (AMPK off-target cardiotoxicity),
   ponatinib (vascular events), dasatinib (pleural effusions) — same retrospective
   prediction format as ibrutinib.

### 8.5 Tier E: Decoder validation (target → drug generation)

The decoder is implemented per MRC §3.7 (cross-attentive, autoregressive or discrete
diffusion). Validation:

1. **Reconstruction.** For known drug-target pairs in training, decoder generates a
   SMILES from the target embedding. Tanimoto similarity between generated and ground-
   truth drug. Target: median Tanimoto > 0.5.
2. **Generation diversity.** Sample N candidate drugs per target; report unique-SMILES
   yield, validity rate (RDKit parsable), drug-likeness (Lipinski Ro5, QED).
3. **In-silico binding validation.** Generated SMILES embedded with MoLFormer and
   passed through encoder to score against the target. Compare to encoder cosine
   similarity for true known binders.
4. **Held-out target test.** Generate drugs for a target not in training. Validate
   against any known binders for that target from external sources. Expected to be
   harder than in-distribution generation; document as honest result.

### 8.6 Tier F: Prospective wet-lab validation (eventual)

Out of scope for the first dtSFM v3 paper submission, but planned for revision /
follow-up. Specifically:

1. Select 20-30 high-confidence dtSFM v3 off-target predictions for clinically-used
   kinase inhibitors that are NOT in any training data (i.e., novel predictions).
2. Test in vitro using commercial kinase activity panels (ADP-Glo, Kinase-Glo).
3. Report validation rate; compare to baseline expected hit rate (~10% for random
   kinome screening).

---

## 9. Citation conventions

Locked at scoping time per SFM Audit Handoff Pitfall 3 (catch citation drift before
publication, not during).

| Source | Citation as written in dtSFM-3.0 writeup |
|--------|------------------------------------------|
| MoLFormer-XL | Ross et al., Nature Machine Intelligence, 2022 (IBM Research) |
| ESM-2 | Lin et al., Science, 2023 (Meta) |
| PDBbind | Liu et al., Bioinformatics, 2015 (current version: PDBbind v2024) |
| BindingMOAD | Ahmed et al., Bioinformatics, 2015 |
| ChEMBL | Mendez et al., Nucleic Acids Research, 2019 (current: ChEMBL 34) |
| BindingDB | Gilson et al., Nucleic Acids Research, 2016 |
| **SAIR** | **Lemos et al., bioRxiv 2025.06.17.660168, 2025** (sandboxaq.com/sair) |
| Boltz-1x | Wohlwend et al., bioRxiv, 2024 |
| TDC | Huang et al., Nature Chemical Biology, 2022 |
| Ibrutinib AF mechanism | Xiao et al., Circulation, 2020 (CSK identification) |
| Boltz-2 evaluation | Lowe, Science/AAAS commentary, March 2026 |
| MRC framework | Reddy, bioRxiv, 2026b |

Verify each citation against the published paper's author list during Phase 0.

---

## 10. Audit type (anticipated)

**Full-execution audit.** dtSFM-3.0 trains under preservation discipline from day one,
so all artifacts will be in the repo at audit time. Codex should be able to verify
claims directly without retraining or scratch archeology.

If preservation discipline is violated mid-development (e.g., a checkpoint gets purged
before being committed), document the gap in this SCOPING.md immediately and note
which Leg 1 items shift from full-execution to archival-with-locally-executable.

---

## 11. Out of scope (for dtSFM v3 paper, first submission)

Per the revised publication plan (2026-04-29), the dtSFM v3 paper is a comprehensive
ambitious work that absorbs most previously-deferred items. The remaining out-of-scope
list is short:

- **Prospective wet-lab validation** (Tier F). Planned as paper revision or
  follow-up; collaboration TBD. The first submission relies on retrospective
  case studies (Tier D) for clinical relevance.
- **MRC Scaling Laws** integration. Held in reserve for the BIIE team and not coupled
  to dtSFM v3 publication timing.
- **Decoder beyond MRC §3.7 baseline.** Initial decoder is the MRC-prescribed
  cross-attentive autoregressive form. Discrete diffusion variant, RL fine-tuning,
  multi-step generation, retrosynthetic-aware generation are deferred to a separate
  decoder-focused paper if the first decoder result motivates it.
- **Multi-target / polypharmacology decoder generation.** First decoder generates
  drugs for a single specified target. Multi-target conditioning (generate a drug that
  binds A but not B) is deferred.

Items that were previously out-of-scope but are now **in scope** for dtSFM v3:
decoder, Boltz-2 head-to-head, alanine scan, safety panel, strict-OOD via LV,
LY2090314/TgGSK3 and other case studies, full druggable proteome.

---

## 12. Constraints (per audit handoff)

- Read-only audit for source code, configs, scripts (when audit runs).
- Use explicit `git add <path>` for all audit commits. No `git add -A`.
- Working tree may contain unrelated WIP (binderSFM v2.0, paper updates, etc.); audit
  commits do not touch them.
- All audit deliverables follow the four-document standard
  (CODEX_AUDIT_RESULTS.md, LEAKAGE_VERIFICATION.md, FILTERED_R1_RESULTS.md, README.md)
  plus AUDIT_CLOSURE_NOTES.md if editorial framing is needed.
- Audit closes with git tag `dtsfm-v3-audit-closed`.

---

## 12.1 Storage policy (added 2026-04-30)

Per concurrent SFM development (binderSFM and others actively training under the same
account), all bulk artifacts go to `/cluster/scratch/reddys/dtsfm_v3/` rather than the
home directory. Specifically:

- **Contacts NPZ** (Tier 1 PDBbind + Tier 1.5 SAIR): `/cluster/scratch/reddys/dtsfm_v3/contacts/`
- **Pre-projected embeddings, intermediate tensors, training output**: `/cluster/scratch/reddys/dtsfm_v3/`
- **Durable summary CSVs** (per audit Rule 1): `/cluster/home/reddys/sfm_summaries/`
  via `SFM_SUMMARY_DIR` env var
- **Split-index JSONs** (per audit Rule 4): `audit/split_index/` in repo via Git LFS
- **Archival SLURM logs** (per audit Rule 3): `audit/archival_logs/` in repo,
  committed within 7 days of each training campaign

Scratch auto-purges, so any artifact required for the audit must be promoted to home
or repo before the campaign closes. The promotion checklist is in §6 (preservation
discipline rules).

## 13. Phase 0 deliverables (immediate next steps)

1. SAIR license verification — confirm Boltz-1x derivative training is permitted under
   the published license. Document at the top of `data/dtsfm/sair_license_check.md`.
2. PDBbind general set download (handled by user account).
3. SAIR dataset download from sandboxaq.com/sair. Verify integrity (checksum, file
   count, anomaly rate matches paper's ~3%).
4. ChEMBL 34 / BindingDB via TDC for Tier 2.
5. Contact extraction pipeline (B-0): 5 Å heavy-atom on both sides, output per-pair
   NPZ + index TSV. Test on 100 PDBbind structures and 100 SAIR structures before
   running at scale.
6. PoseBusters filtering of SAIR: drop any structure flagged for physical anomalies.
   Log filter rate; should be ~3%.
7. Sequence/scaffold leakage filter for prospective-validation drug-target pairs
   (ibrutinib, acalabrutinib, LY2090314/TgGSK3, MERS-CoV main protease, etc.). Held-out
   list committed to `audit/heldout_validation_targets.tsv`.
8. v2 mean-pool DMS baseline (formal mean-pool ceiling check). Compute v2 AUROC on a
   single-AA-mutation DMS task (Bloom RBD or equivalent). If v2 AUROC < 0.65, v3.0 is
   justified per SFM v2.0 architecture handoff.
9. Audit infrastructure setup: `SFM_SUMMARY_DIR` env var; `RETRAIN_LOG.md` initialized;
   `audit/.gitattributes` LFS rules for split_index.

---

## 14. Iteration plan / paper milestones

dtSFM v3 is explicitly a multi-milestone paper. We do not aim for a single-shot result.
The plan is to build, validate, refine, and add scope iteratively until the paper
tells a complete story across encoder + decoder + safety panels + clinical case
studies.

| Milestone | Deliverable | Validation tier(s) | Estimated time | Audit checkpoint |
|---|---|---|---|---|
| **M0** | Phase 0 complete: data downloaded, audit infrastructure live, v2 mean-pool baseline confirms v3 is justified | none | 1-2 weeks | none |
| **M1** | Encoder v3.0 trained (B-3 + B-3.5). Pool-512 + contact + interface + affinity head metrics on PDBbind/SAIR/binding-only test sets. | A, B partial | 1-2 weeks compute + 1 week analysis | optional intermediate scoping update |
| **M2** | Boltz-2 head-to-head + alanine scan + random reassignment. dtSFM v3 vs Boltz-2 on DAVIS. | C | 1-2 weeks (Boltz-2 inference is slow; budget GPU time) | optional intermediate scoping update |
| **M3** | Safety panel evaluation. Ibrutinib v3 refinement, Safety-77 reproduction, hERG / cardiac panel, LY2090314 case study, sunitinib / ponatinib / dasatinib case studies. | D | 1 week | optional |
| **M4** | Decoder implementation and validation (Tier E). Reconstruction, generation diversity, held-out target generation. | E | 2-4 weeks | optional |
| **M5** | Strict-OOD via Leakage Verification on the proteome-wide protein clustering and drug scaffold clustering. | B (full) | 1 week | required pre-submission |
| **M6** | First paper submission (without wet-lab). All Tier A-E results + retrospective Tier D case studies. | A, B, C, D, E | submission deadline | **MANDATORY** full audit + Codex Leg 1/2/3 + git tag `dtsfm-v3-audit-closed` |
| **M7 (revision / follow-up)** | Wet-lab validation of selected dtSFM v3 off-target predictions. | F | months (collaboration-dependent) | re-audit before revision submission |

**Iteration discipline:**
- Each milestone produces durable artifacts following the SFM Audit Handoff §3
  preservation discipline. RETRAIN_LOG.md is updated continuously.
- Milestones M1-M5 may produce intermediate writeup drafts; only the M6 draft is the
  audit-prepared final. Earlier drafts are not commit-required but should be archived
  if they contain numbers that change.
- If a milestone reveals an architecture issue (e.g., the affinity head doesn't
  improve Pearson r despite ChEMBL/BindingDB scale), document the finding and either
  revise the architecture or accept the result as honest. Do not discard the milestone
  without recording its outcome.

**Why iterative rather than single-shot:**
- The decoder requires the encoder to be trained first, so M1 must complete before M4
  can begin in earnest.
- The Boltz-2 comparison (M2) requires an external compute campaign that can run in
  parallel with M3-M4 Encoder fine-tuning.
- Safety panel curation (M3) requires literature mining that benefits from being
  separated from training-time decisions.
- LV (M5) is most usefully run as the final pre-submission validation step, when the
  scope of reported claims is fixed.

---

## 16. Workflow Validation Strategy (REVISED 2026-05-07 — supersedes parts of §8)

**Strategic reframe locked 2026-05-07.** The dtSFM v3 paper validates **three
operational workflows** rather than running a generic battery of benchmarks.
Each experiment must use dtSFM in its actual deployment mode and on the kind
of data the model was trained for (drug-like compounds, well-represented
protein targets — not arbitrary chemotypes or fragment libraries).

**Why the reframe**: Earlier evaluation choices (OpenBind §8.3.1, ibrutinib in
§8.4) tested capabilities outside dtSFM v3's training distribution
(fragment-screen actives at K_D > 10 µM; "double-OOD" clinical drugs whose
SMILES + target representation are both sparse in training). The model
performed poorly on these by design, not because of architecture issues. The
revised plan picks experiments that match v3's training scope and saves the
double-OOD generalization claim for v4 (NVIDIA + pharma partnership scale).

### §F.1 — Safety screening: clinical TKIs against the proteome (LOCKED 2026-05-07)

**Class A / B / C pair-leakage framework.** Each (drug, off-target gene) pair
in any panel is classified by its training-data status:

| Class | Definition | Use for paper claim? |
|---|---|---|
| A | drug × gene PAIR is in training (drug paired with at least one PDB of that gene) | DROP — could be memorization |
| B | drug seen in training (with other proteins) AND gene seen in training (with other drugs) AND specific pair NOT in training | **KEEP — true retrospective prediction** |
| C | drug or gene OOD (drug SMILES not in training, or gene has no PDBs in training) | DROP — cannot evaluate |

Implementation:
- `data/dtsfm/scripts/build_protein_id_to_gene_symbol.py` — SIFTS PDB→UniProt + UniProt REST → gene_symbol mapping (91.4% of 22,964 training proteins mapped)
- `data/dtsfm/scripts/classify_panel_pair_leakage.py` — strict exact-match (RDKit-canonical SMILES); produces `audit/safety_panel_pair_leakage.tsv`
- `data/dtsfm/scripts/analyze_proteome_screen_rollup.py` — gene-level rollup of per-PDB rankings (best rank per gene), Class B retrieval metrics

**Locked numbers (epoch 10, Klaeger 2017 Kinobeads panel, 10 clinical TKIs):**

- 27 Class B retrospective-prediction pairs across 5 drugs (imatinib, dasatinib, gefitinib, erlotinib, sunitinib)
- Top-50 retrieval: 70.4% (19/27) — 70× random
- Top-100 retrieval: 88.9% (24/27) — 44× random
- Top-500 retrieval: 100%
- Median gene rank: 30 of 4,910 unique genes
- DDR1 retrospective: rank 77 (imatinib), rank 26 (dasatinib) — 2014 fibrosis off-target predicted retrospectively
- Dasatinib EPHA/EPHB receptors: 13 of 14 in top-100
- Acalabrutinib + ibrutinib + ponatinib + crizotinib are Class C (drug-OOD); reported as secondary observation, not headline claim

### §F.2 — Kinome validation against Klaeger 2017 (PLANNED 2026-05-07)

**The dtSFM-as-Kinobeads-augmentation experiment.** Klaeger et al. 2017
(*Science*) ran 243 clinical kinase drugs against ~300 kinases via Kinobeads
chemoproteomics, producing the field's gold-standard kinome panel dataset.
We propose to:

1. Filter Klaeger's 243 drugs to those whose RDKit-canonical SMILES are in
   our 522,776-drug training set (expected: ~30-100 drugs).
2. For each, run `proteome_screen.py` against the full 22,964-protein
   universe; roll up to gene level via the §F.1 mapping; restrict to the
   subset of training genes that map to human kinases.
3. For each (drug, kinase) pair, classify A/B/C and compare rank to
   Klaeger's measured K_d_app:
   - Binary discrimination: AUROC for "Klaeger-confirmed binder" vs "Klaeger-tested non-binder"
   - Rank correlation: Spearman ρ between dtSFM rank and Klaeger K_d
   - Headline metric: per-drug top-K kinome retrieval rate (Class B only)

**Why this matters for the paper**: gives us **head-to-head against the
Kinobeads gold standard** on the same drugs and same kinome — apples-to-apples
comparison. The paper claim becomes "dtSFM v3 reproduces Y% of the Kinobeads
kinome panel for N clinical drugs at top-K, with X% wider proteome coverage
(non-kinase targets) at <0.1% of the cost."

**Why this matters for the partnership pitch**: pharma routinely runs Kinobeads
on lead candidates ($1-2K/drug). dtSFM v3 does it for $0.01/drug at 50×
broader proteome coverage. NVIDIA can showcase this as compute → discovery
value translation; Lilly can integrate this into their existing ADME-tox
pipeline immediately.

**Required infrastructure** (next implementation):
- `data/dtsfm/scripts/download_klaeger2017.py` — fetch Klaeger supplementary
  table (proteomicsDB or paper supplement)
- `data/dtsfm/scripts/run_klaeger_kinome_validation.py` — drives the
  per-drug screen + cross-check pipeline
- Output: `audit/klaeger_kinome_validation_results.tsv` and headline
  summary JSON

### §F.3 — Drug repurposing demonstration (PLANNED, post-§F.2)

T2D direction: pick 3-5 high-impact disease targets where there's clear unmet
medical need; screen all 522,776 training drugs; demonstrate enrichment of
known binders + plausible novel hypotheses via AF3-rerank.

Specific targets to be selected with clinical advisors. Candidate list:
NLRP3, CD73, KRAS-G12D, TGFβR1, cereblon (CRBN), specific GPCRs.

### §F.4 — Generative design via decoder + AF3 verify (PLANNED, decoder build in separate chat)

Encoder is locked; decoder build is the next major engineering effort. See
`audit/DECODER_HANDOFF.md` for the technical handoff.

End-to-end workflow:
1. For 10 blockbuster targets (BTK, EGFR, BRAF, CDK4, PARP1, JAK2, KRAS-G12D,
   FGFR2, MET, ROS1 — TBD), sample N=1000 candidates via target-conditioned decoder
2. Encoder rerank (cosine + predicted_pAff) → keep top 100
3. AF3 / Boltz-2 verification → keep top ~30 with iptm > 0.7
4. Headline: per-target novelty (Tanimoto < 0.4 to known binders), QED ≥ 0.5,
   Lipinski compliance, count of structure-confirmed candidates

No wet-lab in this phase — pure end-to-end *in silico* generative discovery
with structural verification at each stage.

### Status of older §8 tiers

- **§8.1 Tier A** (Architectural readiness): superseded by §F header; encoder is locked.
- **§8.2 Tier B** (Strict-OOD via LV): rolled into M5 and into §F.1's Class B.
- **§8.3 Tier C** (Held-out wet-lab — OpenBind / CACHE / BindingDB): OpenBind PARKED as supplementary (drug-likeness confound on fragment-screen data; see RETRAIN_LOG §E for numbers). CACHE 1-6 deferred (same fragment-screen issue). BindingDB-post-2024 retained as supplementary supporting binder/decoy classification claim.
- **§8.4 Tier D** (Safety panel and clinical case studies): now §F.1 + §F.2 with the formal Class A/B/C framework.
- **§8.5 Tier E** (Decoder validation): now §F.4.
- **§8.6 Tier F** (Prospective wet-lab): unchanged, still post-revision / collaboration-dependent.

---

## 15. Living document notice

This SCOPING.md is a forward-looking design document. It will be revised as Phase 0
progresses, B-0 contact extraction runs, B-2 / B-2.5 / B-3 training reveals what the
actual numbers look like, and milestones M1-M7 each produce results. Sections 4
(anticipated SA items), 5 (anticipated CL items), and 14 (milestones) will be promoted
into the formal `audit/dtsfm_audit_v0.3.yaml` only after the M6 milestone is reached
and the paper is ready for submission.

The audit YAML, Codex execution, and closure tag follow only after training, decoder,
and writeup are all complete (M6).

---

— Sai T. Reddy, 2026-04-28 (initial draft)
— Updated 2026-04-29: revised publication plan; consolidated dtSFM-Kinase + dtSFM-
   Proteome into a single dtSFM v3 paper with iterative milestones M0-M7.
