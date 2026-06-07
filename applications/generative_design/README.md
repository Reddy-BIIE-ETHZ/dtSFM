# Application 3 — Generative design (target → novel drug)

Given a protein target, generate novel drug-like molecules conditioned on its
sequence. The cross-attentive autoregressive decoder samples SMILES token-by-token
while attending to the target's per-residue ESM-2 features; a five-stage cascade
then validates, deduplicates, drug-likeness-filters, encoder-reranks, and
structurally verifies the output.

**Headline result.** Across 16 targets with an approved reference drug, of the
1,200 generated candidates, **850 (71%) reach anchor-grade AlphaFold-3 structural
confidence** (iPTM ≥ 0.9 AND interface-PAE ≤ 1.67 Å) — indistinguishable from the
approved drug — and 1,146 (95.5%) clear the binder gate.

## Code

| Step | Module |
|------|--------|
| Train decoder | [`calm.decoder.train_decoder_dtsfm_v3`](../../src/calm/decoder/train_decoder_dtsfm_v3.py) |
| Sample SMILES for a target | [`calm.decoder.generate_decoder_dtsfm_v3`](../../src/calm/decoder/generate_decoder_dtsfm_v3.py) |
| 5-stage screening cascade (validity → dedup → drug-likeness → encoder-rerank → top-K) | [`calm.decoder.screen_decoder_dtsfm_v3`](../../src/calm/decoder/screen_decoder_dtsfm_v3.py) |
| Pre-generation encoder sanity check per target | [`calm.decoder.decoder_smoketest_dtsfm_v3`](../../src/calm/decoder/decoder_smoketest_dtsfm_v3.py) |
| SMILES tokenizer | [`calm.decoder.tokenizer`](../../src/calm/decoder/tokenizer.py) · [`build_tokenizer`](../../src/calm/decoder/build_tokenizer.py) |

## Quick start

```python
from calm.decoder.model_dtsfm_v3 import CALMDecoderV3
dec = CALMDecoderV3.from_pretrained("decoder_v02_step50000.pt").eval()
smiles = dec.generate(target_sequence="MTEYKLVVVGAGG...", n=100, temperature=0.8)
```

Generated molecules are cofolded with AlphaFold-3 as an orthogonal referee.
The per-design SMILES, Tanimoto-to-anchor, and AF3 metrics for the 16-target
gallery are in [`../../supplementary/`](../../supplementary/) (Supp Table S1).
