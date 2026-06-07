# Zenodo deposit — status

| Field | Value |
|-------|-------|
| Deposition ID | `20581780` |
| Reserved DOI | `10.5281/zenodo.20581780` |
| State | **draft (unpublished)** — metadata set, **6 data bundles uploaded** |
| Draft edit URL | https://zenodo.org/deposit/20581780 |
| Title | dtSFM v3 — candidate pools and AlphaFold-3 verification (Paper 1 supplement) |
| License | CC-BY-4.0 |

The reserved DOI is stable and becomes resolvable the moment the deposit is
**published** (manual click on the Zenodo web UI, or via the API). Publish only
when the paper is ready — publishing is permanent.

## Uploaded bundles (in the draft)

| Bundle | Size | Backs |
|--------|------|-------|
| `decoder_design_af3_cofolds.tar.gz` | 143 MB | §5.4 — 888 on-target AF3 cofold CIFs + 868 confidence summaries + manifest (Figs 5–6) |
| `affinity_calibration_cohorts.tar.gz` | 57 MB | §3.3 / §S.6 — per-compound affinity cohorts + 522K training-drug library |
| `approved_drug_anchor_cofolds.tar.gz` | 38 MB | approved-drug anchor cofolds (gallery reference) |
| `atpE_pilot_deliverable.tar.gz` | 18 MB | §5.5.2 atpE pilot (full deliverable bundle) |
| `repurposing_af3_cofolds.tar.gz` | 1.4 MB | §5.3 — NLRP3/CD73/STING1 representative cofolds |
| `klaeger2017_kdapp_long.tsv.gz` | 0.6 MB | §5.1 Klaeger 2017 K_d^app table |

## Remaining to do
1. **(Optional)** The ~624 off-target *selectivity* cofold structures (~8.8 GB raw)
   are **not** in this bundle — their confidence summaries are already inside
   `decoder_design_af3_cofolds.tar.gz`. Add the raw CIFs as a separate large
   bundle if reviewers want them.
2. Add the **bioRxiv DOI** to the metadata `related_identifiers` once the preprint
   is posted.
3. Review the draft on the web UI, then **Publish** (mints the DOI permanently).

> No access token is stored in this repository. Provide it via `ZENODO_TOKEN`
> when running the upload script.
