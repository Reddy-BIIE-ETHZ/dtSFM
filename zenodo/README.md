# Zenodo deposit (Paper 1 supplement)

Tooling + metadata to deposit the large dtSFM Paper-1 data to Zenodo and mint a
citable DOI. The deposit holds what's too big for git (candidate pools, AF3
structures, affinity cohorts, the atpE pilot). See
[`CONTENTS_MANIFEST.md`](CONTENTS_MANIFEST.md) for the file list + source paths.

## How to run

```bash
# 1. Create a Zenodo personal access token (scopes: deposit:write, deposit:actions)
#    https://zenodo.org/account/settings/applications/tokens/new/
export ZENODO_TOKEN=...

# 2. Gather + bundle the source files (per CONTENTS_MANIFEST.md), e.g. on Euler:
#    tar --zstd -cf repurposing_rankings.tar.zst audit/dtsfm/repurposing/screen_*_top1000_annotated.tsv
#    ... (one bundle per manifest row)

# 3. Dry-run on the sandbox first (recommended):
python zenodo_upload.py --sandbox --files bundle1.tar.zst bundle2.tar.zst ...

# 4. Real deposit (creates a DRAFT — does NOT publish):
python zenodo_upload.py --files bundle1.tar.zst bundle2.tar.zst ...
```

The script creates a draft, sets metadata from
[`zenodo_metadata.json`](zenodo_metadata.json), and uploads files. **It does not
publish** — review on the Zenodo web UI and click *Publish* when the paper is
ready (publishing mints the DOI permanently).

## After publishing

1. Copy the minted DOI into:
   - the top-level `README.md` badges (replace `DOI-pending`),
   - `CITATION.cff`,
   - `zenodo_metadata.json` → `related_identifiers` (link the bioRxiv DOI),
   - the paper's "Data and code availability" section.
2. Requires `pip install requests`.

## Open decision (for Sai)

Use your **personal** Zenodo account, or an **ETH/BIIE institutional Zenodo
community**? If the lab has a Zenodo community, add its identifier to
`zenodo_metadata.json` under `"communities": [{"identifier": "..."}]` before the
real run.
