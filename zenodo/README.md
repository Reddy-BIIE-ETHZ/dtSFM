# Zenodo deposit — dtSFM v3 supporting data

The large dtSFM v3 supporting data — generated candidate pools, AlphaFold-3 cofold
structures, affinity-calibration cohorts, and the atpE pilot — are archived on
Zenodo (too large for git). See [`CONTENTS_MANIFEST.md`](CONTENTS_MANIFEST.md)
for the file list and [`DEPOSIT_INFO.md`](DEPOSIT_INFO.md) for the contents of
the record.

**DOI:** [10.5281/zenodo.20581780](https://doi.org/10.5281/zenodo.20581780)

This folder also contains the tooling used to create the deposit, so the archive
is fully reproducible.

## Reproducing / extending the deposit

```bash
# 1. Create a Zenodo personal access token (scopes: deposit:write, deposit:actions)
#    https://zenodo.org/account/settings/applications/tokens/new/
export ZENODO_TOKEN=...

# 2. Bundle the source files listed in CONTENTS_MANIFEST.md, e.g.:
#    tar --zstd -cf repurposing_rankings.tar.zst <files>

# 3. (optional) dry-run against the Zenodo sandbox:
python zenodo_upload.py --sandbox --files bundle1.tar.zst ...

# 4. Upload to a new draft (does not publish):
python zenodo_upload.py --files bundle1.tar.zst ...
```

[`zenodo_upload.py`](zenodo_upload.py) creates a draft, applies the metadata in
[`zenodo_metadata.json`](zenodo_metadata.json), and uploads the files; it never
publishes. Requires `pip install requests`.
