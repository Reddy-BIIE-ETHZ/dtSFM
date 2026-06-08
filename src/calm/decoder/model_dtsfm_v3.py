"""model_dtsfm_v3.py — CALMDecoderV3, the dtSFM v3 generative decoder.

Architecture (v0.1, locked 2026-05-09):

    Cross-attentive autoregressive transformer.

    Inputs:
        input_ids        (B, T)        SMILES token ids (MoLFormer-XL tokenizer)
        attention_mask   (B, T) bool   True = valid token
        protein_emb      (B, L, 1280)  cached ESM-2 per-residue
        protein_mask     (B, L) bool   True = valid residue

    Token + position embeddings (sinusoidal positional)
        → 6 standard nn.TransformerDecoderLayer blocks
            self-attn (causal mask)
            cross-attn (Q = SMILES tokens, KV = protein per-residue projection)
            FFN d=512 → 2048 → 512
        → final LayerNorm
        → tied output projection (token-embedding weight)

    Output:
        logits           (B, T, vocab_size)

Conditioning is on the target's ESM-2 per-residue features, projected to
d_model=512 by a frozen-friendly linear+GELU+LN head. This is independent of
the v3 encoder's protein_proj — the decoder learns its own conditioning
projection so its gradient path stays clean and the encoder remains untouched.

v0.1 design choices:
    - MoLFormer-XL tokenizer (vocab ≈ 2,363 BPE tokens) — chosen so generated
      SMILES re-encode cleanly through MoLFormer for any downstream rerank.
    - Tied input/output token embeddings (saves ~1.2M params, standard for AR).
    - Sinusoidal position encoding (no extra learned params, well-tested).
    - 6 layers, d=512, 8 heads, d_ff=2048 → ~25M trainable params.
    - bf16 mixed precision in training; model itself stays fp32.

v0.2 will add:
    - Auxiliary cosine-distillation loss (decoder hidden-state pooled →
      project to MoLFormer-768 space → cosine vs cached drug embedding of
      ground-truth drug). Deferred to v0.2 — v0.1 = pure CE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding. No learned params."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        T = x.shape[1]
        return x + self.pe[:T].unsqueeze(0)


class CALMDecoderV3(nn.Module):
    """dtSFM v3 generative decoder — AR transformer with protein cross-attention.

    Config (DictConfig) expected fields:
        d_model            : int (default 512)
        n_heads            : int (default 8)
        n_layers           : int (default 6)
        d_ff               : int (default 2048)
        dropout            : float (default 0.1)
        vocab_size         : int (from MoLFormer tokenizer)
        pad_token_id       : int (from MoLFormer tokenizer)
        protein_emb_dim    : int (default 1280, ESM-2 650M)
        max_smiles_tokens  : int (default 256)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)
        vocab_size = int(cfg.vocab_size)
        pad_id = int(cfg.pad_token_id)

        # Token embedding (shared with output projection — tied weights)
        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        # Re-zero the pad row (Embedding does this internally on padding_idx)
        with torch.no_grad():
            self.tok_emb.weight[pad_id].fill_(0.0)

        self.pos_emb = SinusoidalPositionalEncoding(
            d_model=d, max_len=int(cfg.max_smiles_tokens) + 8
        )

        # Protein → d_model conditioning projection (decoder-local; encoder
        # protein_proj is NOT reused — keeps gradients clean)
        self.protein_proj = nn.Sequential(
            nn.Linear(int(cfg.protein_emb_dim), d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        for m in self.protein_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Standard PyTorch decoder stack — pre-norm, GELU FFN
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=int(cfg.n_heads),
            dim_feedforward=int(cfg.d_ff),
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(cfg.n_layers))
        self.final_norm = nn.LayerNorm(d)

        # Tied output projection: bias is learned, weight reuses tok_emb.weight
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """(T, T) bool mask where True = position is masked out (future)."""
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,         # (B, T) int64
        attention_mask: torch.Tensor,    # (B, T) bool, True = valid
        protein_emb: torch.Tensor,       # (B, L, 1280)
        protein_mask: torch.Tensor,      # (B, L) bool, True = valid
    ) -> dict[str, torch.Tensor]:
        B, T = input_ids.shape

        # Token + position embeddings
        x = self.tok_emb(input_ids)                                     # (B, T, d)
        x = self.pos_emb(x)                                             # (B, T, d)

        # Protein cross-attn keys/values
        memory = self.protein_proj(protein_emb)                         # (B, L, d)

        # Build masks for nn.TransformerDecoder convention:
        #   tgt_mask:                  (T, T) bool — True = mask (causal)
        #   tgt_key_padding_mask:      (B, T) bool — True = mask (padded position)
        #   memory_key_padding_mask:   (B, L) bool — True = mask (padded residue)
        causal = self._causal_mask(T, x.device)
        tgt_pad = ~attention_mask
        mem_pad = ~protein_mask

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=mem_pad,
        )
        out = self.final_norm(out)                                      # (B, T, d)

        # Tied output projection: logits = out @ tok_emb.weight^T + bias
        logits = out @ self.tok_emb.weight.t() + self.output_bias       # (B, T, V)

        return {"logits": logits, "hidden_states": out}

    def count_parameters(self) -> dict[str, int]:
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "tok_emb": n(self.tok_emb),
            "protein_proj": n(self.protein_proj),
            "decoder_blocks": n(self.decoder),
            "final_norm": n(self.final_norm),
            "output_bias": self.output_bias.numel(),
            "total": n(self),
        }


def build_decoder_v3_model(cfg: DictConfig) -> CALMDecoderV3:
    """Factory mirroring build_encoder_v3_model() API."""
    model_cfg = cfg.model.decoder if hasattr(cfg, "model") else cfg
    return CALMDecoderV3(model_cfg)


def default_decoder_v3_cfg(vocab_size: int, pad_token_id: int) -> dict:
    """v0.1 locked defaults. vocab_size + pad_token_id come from MoLFormer tokenizer."""
    return {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "protein_emb_dim": 1280,
        "max_smiles_tokens": 256,
    }
