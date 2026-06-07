# Application 2 — Library repurposing (target → drug)

Given a protein target, rank a large compound library for binding compatibility.
dtSFM embeds the target once and scores it against pre-computed embeddings of
every library molecule by cosine similarity; the top of the ranked list is the
repurposing/screening shortlist.

**Headline result.** Across three targets with no approved small-molecule drug
in the relevant pocket — **NLRP3, CD73, STING1** — dtSFM's top candidates yield
**46 novel molecules that clear the AlphaFold-3 binder gate** (iPTM ≥ 0.7 AND
interface-PAE ≤ 5 Å), drawn from the 522,776-compound training library.

## Method

```python
from calm.encoder.model_v3 import CALMEncoderV3
import torch, torch.nn.functional as F

enc = CALMEncoderV3.from_pretrained("encoder_b3_epoch010.pt").eval()
target = enc.encode_protein(target_sequence)              # (1, 512)
lib = torch.stack([enc.encode_drug(s) for s in library_smiles])  # (L, 512)
shortlist = F.cosine_similarity(target, lib).argsort(descending=True)[:1000]
```

The ranking is plain cosine retrieval over the joint embedding space — the same
`CALMEncoderV3` used for safety screening, run in the opposite direction.
Shortlisted candidates are then cofolded with AlphaFold-3 as an orthogonal
referee (see [`docs/alphafold3_cofold_protocol.md`](../../docs/alphafold3_cofold_protocol.md)).

> **Note.** dtSFM cosine and AF3 confidence are essentially uncorrelated
> (Pearson ≈ 0) on these candidates — AF3 is a genuine independent filter, not a
> circular re-confirmation of the dtSFM ranking. Only molecules that pass *both*
> are reported.
