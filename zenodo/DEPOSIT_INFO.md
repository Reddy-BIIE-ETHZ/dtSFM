# Zenodo deposit — status

| Field | Value |
|-------|-------|
| Deposition ID | `20581780` |
| Reserved DOI | `10.5281/zenodo.20581780` |
| State | **draft (unpublished)** — metadata set, files pending |
| Draft edit URL | https://zenodo.org/deposit/20581780 |
| Title | dtSFM v3 — candidate pools and AlphaFold-3 verification (Paper 1 supplement) |
| License | CC-BY-4.0 |

The reserved DOI is stable and becomes resolvable the moment the deposit is
**published** (manual click on the Zenodo web UI, or via the API). Publish only
when the paper is ready — publishing is permanent.

## Remaining to do
1. Bundle the files listed in [`CONTENTS_MANIFEST.md`](CONTENTS_MANIFEST.md)
   (mostly on Euler/ALPS) and upload them to this draft with
   [`zenodo_upload.py`](zenodo_upload.py) using `--deposition-id 20581780`.
2. Add the bioRxiv DOI to the metadata `related_identifiers` once the preprint
   is posted.
3. Review the draft on the web UI, then **Publish**.

> No access token is stored in this repository. Provide it via `ZENODO_TOKEN`
> when running the upload script.
