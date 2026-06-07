# Archival Checkpoint Manifest

This directory contains the **immutable record** of the dtSFM v3 production
checkpoints, with cryptographic hashes that fix the exact bytes against which
all claims in Paper 1 and Paper 2 must verify.

The artifacts on Hugging Face (`SFM-BIIE-ETHZ/dtSFM-v3`) are the **canonical
distribution** of these checkpoints. This manifest is the audit-trail anchor
that links the bytes on Hugging Face to the original training runs.

---

## Manifest

[`checkpoint_manifest.tsv`](checkpoint_manifest.tsv) — 4 rows:

| Artifact | Status |
|---|---|
| `encoder_b3_epoch010.pt` | Production encoder; SHA256 anchored |
| `encoder_b3_best_by_retrieval.pt` | **Identical to epoch 10** (same SHA256) — confirms best-by-retrieval criterion selects epoch 10 |
| `decoder_v02_step50000.pt` | Production decoder; SHA256 anchored |
| `metadata_v3.csv` | Training corpus metadata; SHA256 anchored |

## How auditors should use this

To verify that the checkpoint on Hugging Face is the one used in the paper:

```bash
# WHERE: anywhere with internet + sha256sum

# Download from HF
hf download SFM-BIIE-ETHZ/dtSFM-v3 --local-dir ./check

# Verify hashes
shasum -a 256 check/encoder_b3_epoch010.pt
# Expected: d9015638405bec90169a82d53411b584708b78650f5424425d1339d2f0700c20

shasum -a 256 check/decoder_v02_step50000.pt
# Expected: b5dc4039014f3476b3c64afe9e518acb80d430f18c9570d8905075d412213dbd

shasum -a 256 check/metadata_v3.csv
# Expected: 46b82b8b5b5eaa54424dba6f6a330f3ed688912365fdddae71de35353bdf835d
```

If hashes match: bit-exact reproduction of the paper's model.

If hashes don't match: model has been updated; consult the manifest version
history for which paper version corresponds to which checkpoint.

## Versioning policy

This manifest is **append-only after first publication**. If retraining or
recalibration produces a new production checkpoint, a new row is added with
a new `artifact_id` and the corresponding HF repo URL is updated to point at
the new file. The old hash remains in the manifest for permanent traceability.

## Distribution

The production checkpoints are distributed from the public Hugging Face repo
`SFM-BIIE-ETHZ/dtSFM-v3`. The SHA256 hashes above are the immutable anchor —
the same bytes are served from Hugging Face.
