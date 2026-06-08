# Zenodo deposit — contents manifest

Title: **dtSFM v3 — candidate pools and AlphaFold-3 verification**
License: CC-BY-4.0 · Access: open

This is the large-data companion to the GitHub code repo and the Hugging Face
weights. It holds the artifacts that are too large for git but that back the
paper's claims. Each item lists its **source path** (on the lab's compute) so the
upload script can gather them. Bundle each group as a `.tar.zst` (or `.zip`)
before upload to keep the deposit tidy.

| # | Bundle (upload name) | Source path(s) | Backs |
|---|----------------------|----------------|-------|
| 1 | `repurposing_rankings.tar.zst` | `audit/dtsfm/repurposing/screen_{NLRP3,CD73,STING1_*}_top1000_annotated.tsv` + full (non-top-1000) ranking exports | §5.2 |
| 2 | `affinity_calibration_cohorts.tar.zst` | `audit/dtsfm/affinity_calibration/F6_phase1_*_classified.tsv`, `F6_phase2_*_predicted.tsv`, `F6_phase1_pdbbind_within_dist.tsv` | §3.3 / §S.6 |
| 3 | `training_library.tar.zst` | `audit/dtsfm/affinity_calibration/F6_v3_train_drugs.tsv` (522,776 compounds; derivable from public PDBbind v2020 + SAIR) | §2 |
| 4 | `decoder_cohort_af3.tar.zst` | `audit/dtsfm/decoder_af3/af3_outputs/<candidate>/_summary_confidences.json` (1,520 files) + representative CIFs | §5.4 |
| 5 | `repurposing_af3_cifs.tar.zst` | §5.3 representative cofold CIFs (anchor + top decoder + negative control per target) — on ALPS scratch | §5.3 |
| 6 | `klaeger2017_panel.tar.zst` | `audit/dtsfm/klaeger2017/klaeger_kdapp_long.tsv` (Klaeger 2017 published K_d^app supplement) | §5.1 |
| 7 | `atpE_pilot_deliverable.tar.zst` | `audit/dtsfm/biie_lmic/atpE_deliverable/` (1,027-candidate pool + 5 CIFs + figure) | §5.5.2 |

> **Provenance.** The compact summary/metric files that these feed
> (`F6_phase3_metrics.tsv`, `safety_panel_klaeger2017.tsv`, `F5_*` decoder tables,
> `F4_*`/`F42_*` repurposing tables, the leakage TSVs) are committed in the
> GitHub repo under `audit/`, so every headline number is already traceable
> without this deposit. This deposit provides the underlying per-compound and
> per-structure raw data.

> **Not here:** model weights (Hugging Face `SFM-BIIE-ETHZ/dtSFM-v3`); source
> code (GitHub `SFM-BIIE-ETHZ/dtSFM`).
