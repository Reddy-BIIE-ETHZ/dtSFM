---
license: other
license_name: sfm-research-preview-license-v1.0
license_link: LICENSE.md
tags:
  - drug-discovery
  - drug-target
  - cheminformatics
  - protein
  - generative-design
  - sfm
  - biology
  - bioinformatics
pipeline_tag: feature-extraction
---

# dtSFM-v3 — Drug–Target Specificity Foundation Model (production encoder + generative decoder)

**Paper:** *A drug–target specificity foundation model for off-target prediction, repurposing, and generative design* · doi: 10.64898/2026.06.08.730844
**Code:** [github.com/Reddy-BIIE-ETHZ/dtSFM](https://github.com/Reddy-BIIE-ETHZ/dtSFM)
**All SFM models:** [huggingface.co/SFM-BIIE-ETHZ](https://huggingface.co/SFM-BIIE-ETHZ)

> This is the **production dtSFM** (v3): a full-scale cross-attention encoder paired with a
> cross-attentive autoregressive decoder. The smaller encoder-only prototype from the
> *Vibe Coding SFMs* paper lives separately at
> [`SFM-BIIE-ETHZ/dtSFM_VC-SFM`](https://huggingface.co/SFM-BIIE-ETHZ/dtSFM_VC-SFM).

---

## What it does

dtSFM maps a **(drug SMILES, protein sequence)** pair to a binding-compatibility score, and
**generates** novel target-conditioned drug-like molecules — all from sequence, without
constructing a 3-D structure. It is built on the SFM principle that transformer softmax
attention is mathematically isomorphic to the Boltzmann distribution of molecular binding.

Three applications run on this single model:

- **Off-target safety screening** (drug → proteome): documented off-targets at median rank 30 / 4,910 genes (top 0.6%) vs the Klaeger 2017 chemoproteomic panel.
- **Library repurposing** (target → drug): 46 novel candidates clear the AlphaFold-3 binder gate across NLRP3 / CD73 / STING1.
- **Generative design** (target → novel drug): 850 / 1,200 (71%) of generated molecules match the AlphaFold-3 structural confidence of the approved drug.

| Component | Model |
|-----------|-------|
| Drug encoder | MoLFormer-XL (frozen, SMILES → 768-d) |
| Protein encoder | ESM-2-650M (frozen, → 1,280-d per residue) |
| Cross-attention encoder | trainable, 2 layers · 8 heads · d=512 · 14.4 M params · 4 heads |
| Decoder | cross-attentive autoregressive SMILES generator (~28 M params) |
| Training data | PDBbind v2020 + SAIR (714,747 pairs · 522,776 drugs · 22,964 proteins) |
| Split | whole MMseqs2 protein clusters held out at 80% identity; zero pair/protein/cluster leakage |

---

## Files in this repo

| File | Description |
|------|-------------|
| `encoder_b3_epoch010.pt` | locked production encoder (B-3, 4 heads) |
| `decoder_v02_step50000.pt` | cross-attentive generative decoder (checkpoint @ 50k steps) |

---

## Quick start

```python
from huggingface_hub import hf_hub_download
import torch, torch.nn.functional as F
from calm.encoder.model_v3 import CALMEncoderV3
from calm.decoder.model_dtsfm_v3 import CALMDecoderV3

# --- retrieval / scoring (drug ↔ target) ---
enc = CALMEncoderV3.from_pretrained(
    hf_hub_download("SFM-BIIE-ETHZ/dtSFM-v3", "encoder_b3_epoch010.pt")).eval()
drug   = enc.encode_drug("CC(=O)Oc1ccccc1C(=O)O")               # aspirin
target = enc.encode_protein("MTEYKLVVVGAGGVGKSALTIQLIQ...")
score  = F.cosine_similarity(drug, target, dim=-1)

# --- generation (target → novel molecules) ---
dec = CALMDecoderV3.from_pretrained(
    hf_hub_download("SFM-BIIE-ETHZ/dtSFM-v3", "decoder_v02_step50000.pt")).eval()
smiles = dec.generate(target_sequence="MTEYKLVVVGAGG...", n=100, temperature=0.8)
```

Install the codebase from [github.com/Reddy-BIIE-ETHZ/dtSFM](https://github.com/Reddy-BIIE-ETHZ/dtSFM)
(`conda env create -f environment.yml`).

---

## Orthogonal verification

Every structural claim is checked by **AlphaFold-3** as an orthogonal referee — it shares no
architecture, training data, or representation with dtSFM (dtSFM-cosine ↔ AF3-confidence
correlation ≈ 0), so structural agreement is genuine corroboration, not circular confirmation.
Binder gate: iPTM ≥ 0.7 AND interface-PAE ≤ 5 Å. Anchor-grade gate: iPTM ≥ 0.9 AND PAE ≤ 1.67 Å.

---

## Citation

```bibtex
@article{reddy2026dtsfm,
  title   = {A drug–target specificity foundation model for off-target prediction, repurposing, and generative design},
  author  = {Reddy, Sai T.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.06.08.730844}
}
```

## License

Released under the **SFM Research Preview License v1.0-preview** (see `LICENSE.md`).
Free for research use — academic, non-profit, government, and industry research. The specific
molecules disclosed in the accompanying preprints are dedicated to the public. Commercial-use
and patent-licensing terms are deferred and being arranged with ETH Zürich / BIIE; the SFM
architectures and training methods are the subject of pending patent applications.
For commercial enquiries: sai.reddy@ethz.ch
