# Application 1 — Off-target safety screening (drug → target)

Given an approved drug, rank the human proteome for likely off-targets. dtSFM
embeds the drug once and scores it against every protein embedding by cosine
similarity; the ranked list is the off-target hypothesis set.

**Headline result.** Against the Klaeger 2017 kinase-inhibitor chemoproteomic
panel, dtSFM places experimentally documented off-targets at a **median rank of
30 out of 4,910 genes (top 0.6%)**.

## Code

| Step | Module |
|------|--------|
| Off-target retrieval + Klaeger benchmark | [`calm.encoder.validation_offtarget`](../../src/calm/encoder/validation_offtarget.py) |
| Optional Klaeger fine-tune (held-out kinases) | [`calm.encoder.train_klaeger_finetune`](../../src/calm/encoder/train_klaeger_finetune.py) |
| Ibrutinib case study (Fig 3d) | [`calm.encoder.validation_ibrutinib`](../../src/calm/encoder/validation_ibrutinib.py) |
| Decoder-side safety screen (generated molecules) | [`calm.decoder.screen_decoder_safety`](../../src/calm/decoder/screen_decoder_safety.py) |

## Method (sketch)

```python
from calm.encoder.model_v3 import CALMEncoderV3
import torch, torch.nn.functional as F

enc = CALMEncoderV3.from_pretrained("encoder_b3_epoch010.pt").eval()
drug = enc.encode_drug(drug_smiles)                       # (1, 512)
prot = torch.stack([enc.encode_protein(s) for s in proteome])  # (G, 512)
ranks = F.cosine_similarity(drug, prot).argsort(descending=True)
```

The benchmark numbers and figure data are reproduced from the validation
modules above; AlphaFold-3 then confirms the top-ranked off-targets structurally
(binder gate iPTM ≥ 0.7 AND interface-PAE ≤ 5 Å).
