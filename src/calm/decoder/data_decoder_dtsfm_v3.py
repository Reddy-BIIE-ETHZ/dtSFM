"""data_decoder_dtsfm_v3.py — Dataset + collate for the dtSFM v3 decoder.

Yields (protein_emb, smiles_token_ids) pairs from metadata_v3.csv.

  - SMILES tokenization: MoLFormer-XL tokenizer (consistent with encoder
    workflow), with <bos>...<eos> special tokens. Tokenized lazily per batch
    item (cheap; <1ms per SMILES).
  - Protein embeddings: loaded from cached fp16 .pt files at __getitem__
    time. Truncated to max_protein_len if longer.
  - Held-out validation pairs are excluded by (drug_smiles, protein_id) match
    against audit/dtsfm/heldout_validation_pairs.tsv.
  - For smoketest, supports `subsample=4000` to sample a small training subset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class DecoderDtsfmV3Dataset(Dataset):
    """Dataset of (protein_emb, smiles_input_ids) pairs from v3 training metadata."""

    def __init__(
        self,
        metadata_csv: Path,
        protein_dir: Path,
        tokenizer,                                           # huggingface PreTrainedTokenizer
        max_smiles_tokens: int = 256,
        max_protein_len: int = 1024,
        held_out_pairs_tsv: Path | None = None,
        subsample: int | None = None,
        seed: int = 42,
        append_eos: bool = False,
    ):
        df = pd.read_csv(
            metadata_csv,
            usecols=["drug_smiles", "drug_idx", "protein_id", "protein_idx"],
        )
        n_initial = len(df)

        if held_out_pairs_tsv is not None and Path(held_out_pairs_tsv).exists():
            ho = pd.read_csv(held_out_pairs_tsv, sep="\t")
            # Exclude any (drug_smiles, protein_id) pair listed as held-out
            if {"drug_smiles", "protein_id"}.issubset(ho.columns):
                ho_pairs = set(zip(ho["drug_smiles"].astype(str),
                                   ho["protein_id"].astype(str)))
                mask = [(s, p) not in ho_pairs
                        for s, p in zip(df["drug_smiles"].astype(str),
                                        df["protein_id"].astype(str))]
                df = df[mask].reset_index(drop=True)
                print(f"  [dataset] excluded {n_initial - len(df)} held-out pairs "
                      f"({len(df)} remaining)")
            else:
                print(f"  [dataset] WARN: held-out TSV missing drug_smiles/protein_id "
                      f"columns; not excluding")

        if subsample is not None and subsample < len(df):
            df = df.sample(n=subsample, random_state=seed).reset_index(drop=True)
            print(f"  [dataset] subsampled to {len(df)} pairs (seed={seed})")

        self.df = df
        self.protein_dir = Path(protein_dir)
        self.tokenizer = tokenizer
        self.max_smiles_tokens = int(max_smiles_tokens)
        self.max_protein_len = int(max_protein_len)
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.append_eos = bool(append_eos)
        self.eos_token_id = (
            int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
        )
        if self.append_eos and self.eos_token_id is None:
            raise RuntimeError(
                "append_eos=True but tokenizer has no eos_token_id. "
                "Pass add_eos=True to load_molformer_tokenizer().")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        smiles = str(row["drug_smiles"])
        protein_idx = int(row["protein_idx"])

        # Tokenize SMILES. Reserve 1 token for EOS append if append_eos.
        max_len = self.max_smiles_tokens - (1 if self.append_eos else 0)
        enc = self.tokenizer(
            smiles,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0).long()                  # (T,)
        attn_mask = enc["attention_mask"].squeeze(0).bool()             # (T,) True = valid

        # Append EOS if requested (v0.2 — gives the model a stop signal)
        if self.append_eos and self.eos_token_id is not None:
            eos_id = torch.tensor([self.eos_token_id], dtype=torch.long)
            input_ids = torch.cat([input_ids, eos_id])
            attn_mask = torch.cat([attn_mask, torch.ones(1, dtype=torch.bool)])

        # Load protein per-residue embedding from cache (fp16)
        prot_path = self.protein_dir / f"{protein_idx:06d}.pt"
        prot = torch.load(prot_path, map_location="cpu", weights_only=True)
        if not isinstance(prot, torch.Tensor):
            raise RuntimeError(f"Expected tensor at {prot_path}, got {type(prot)}")
        if prot.shape[0] > self.max_protein_len:
            prot = prot[:self.max_protein_len]
        protein_mask = torch.ones(prot.shape[0], dtype=torch.bool)

        return {
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "protein_emb":    prot,
            "protein_mask":   protein_mask,
            "smiles":         smiles,           # kept for debug logging only
        }


def collate_decoder_batch(batch: list[dict], pad_token_id: int) -> dict:
    """Pad input_ids + protein_emb to max-in-batch."""
    B = len(batch)
    max_T = max(b["input_ids"].shape[0] for b in batch)
    max_L = max(b["protein_emb"].shape[0] for b in batch)
    prot_dim = batch[0]["protein_emb"].shape[1]
    prot_dtype = batch[0]["protein_emb"].dtype

    input_ids = torch.full((B, max_T), pad_token_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_T), dtype=torch.bool)
    protein_emb = torch.zeros((B, max_L, prot_dim), dtype=prot_dtype)
    protein_mask = torch.zeros((B, max_L), dtype=torch.bool)

    for i, b in enumerate(batch):
        T = b["input_ids"].shape[0]
        L = b["protein_emb"].shape[0]
        input_ids[i, :T]   = b["input_ids"]
        attn_mask[i, :T]   = b["attention_mask"]
        protein_emb[i, :L] = b["protein_emb"]
        protein_mask[i, :L] = b["protein_mask"]

    return {
        "input_ids":      input_ids,
        "attention_mask": attn_mask,
        "protein_emb":    protein_emb,
        "protein_mask":   protein_mask,
    }


def load_molformer_tokenizer(
    model_name: str = "ibm/MoLFormer-XL-both-10pct",
    add_eos: bool = False,
):
    """Load MoLFormer-XL tokenizer. Same model_name as encoder embedding pipeline.

    If add_eos=True (v0.2 path), inject a custom [EOS] token. MoLFormer's
    pretrained tokenizer has no native EOS; without one, the decoder has no
    natural stop signal during sampling, which was the dominant cause of
    v0.1 generation invalidity (every output ran to max_new_tokens with
    unclosed rings/parens). v0.2 trains the decoder to emit [EOS] at natural
    SMILES endpoints; generation halts on first [EOS].

    Use len(tokenizer) (not tokenizer.vocab_size) for the model's vocab_size
    when add_eos=True — the latter doesn't include the added special token.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        if "[PAD]" in tok.get_vocab():
            tok.pad_token = "[PAD]"
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    if add_eos and tok.eos_token_id is None:
        # Add custom [EOS] token. This grows len(tokenizer) by 1; vocab_size
        # attribute does NOT update (it reflects the original BPE vocab only).
        tok.add_special_tokens({"eos_token": "[EOS]"})
    return tok
