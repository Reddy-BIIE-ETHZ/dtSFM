# dtSFM v3 Decoder Tokenizer

This directory contains the **exact tokenizer** used by the dtSFM v3 generative
decoder (per WIP §5.4.1).

## What this is

A standard 🤗 Hugging Face `AutoTokenizer` save format. To load:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("path/to/tokenizer/dtsfm_v3_decoder",
                                    trust_remote_code=True)
print(len(tok))           # 2363 (includes the custom [EOS] token)
print(tok.eos_token_id)   # 2362
print(tok.pad_token_id)   # 2
```

## How it was built

Programmatic extension of `ibm/MoLFormer-XL-both-10pct` per
[`src/calm/decoder/data_decoder_dtsfm_v3.py`](../../src/calm/decoder/data_decoder_dtsfm_v3.py)
`load_molformer_tokenizer(add_eos=True)`:

1. Load MoLFormer-XL pretrained tokenizer from `ibm/MoLFormer-XL-both-10pct`
   (2,362 BPE tokens; no native EOS)
2. Ensure pad_token is set (use existing `[PAD]` from vocab if available)
3. Add custom `[EOS]` token via `tok.add_special_tokens({"eos_token": "[EOS]"})`
4. New `[EOS]` token gets id = 2362; final `len(tokenizer) = 2363`

## Why a custom EOS was needed

MoLFormer-XL's pretrained tokenizer has no native end-of-sequence symbol
because it was pretrained as a masked-language-model on SMILES, not as an
autoregressive generative model.

In dtSFM v3 decoder v0.1 (without `[EOS]`), the decoder ran to
`max_new_tokens` on every sample, producing structurally-imbalanced output
(unclosed rings/parens). RDKit validity was only 0–1.1%.

dtSFM v3 decoder **v0.2** (with `[EOS]`) trains the decoder to emit `[EOS]`
at natural SMILES endpoints. Generation halts on first `[EOS]`. RDKit
validity rose to 10–17% per target.

## Files (5)

| File | Size | SHA256 (first 16) | Purpose |
|---|---:|---|---|
| `tokenizer.json` | 54,190 | `cce202529bcbcf55` | Full tokenizer state (preferred load target for tokenizers ≥0.13) |
| `vocab.json` | 32,191 | `954c763ca85d251c` | BPE vocabulary mapping (token string → id) |
| `tokenizer_config.json` | 1,546 | `13200c36784f88ab` | AutoTokenizer config (class name, special-token settings) |
| `special_tokens_map.json` | 833 | `9382fd25c28b6842` | Special tokens registry (pad, eos, etc.) |
| `added_tokens.json` | 20 | `b4b7d4defa900a73` | Tokens added beyond the base BPE — `[EOS] → 2362` |

## Reproducibility

Independent reproduction:

```bash
pip install transformers
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
if tok.pad_token_id is None:
    tok.add_special_tokens({'pad_token': '[PAD]'})
if tok.eos_token_id is None:
    tok.add_special_tokens({'eos_token': '[EOS]'})
tok.save_pretrained('./dtsfm_v3_decoder')
"
# Compare SHA256 to the table above
shasum -a 256 dtsfm_v3_decoder/*
```

Output hashes should match within tokenizer-library version compatibility. If
hashes differ, check `transformers` and `tokenizers` library versions — newer
versions may produce semantically-identical but byte-different `tokenizer.json`
files.
