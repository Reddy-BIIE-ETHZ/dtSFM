"""
CALMEncoderV3 — dtSFM v3 drug-target encoder.

Asymmetric cross-attention model with four output heads:

    1. Global contrastive head      (pre-XA pooled features → InfoNCE)
    2. Per-atom interface head      (post-XA drug atom feats → MLP → is-interface)
    3. Atom-residue contact head    (post-XA bilinear drug × protein → contact map)
    4. Affinity regression head     (post-XA pooled drug + protein → pAffinity)

Architecture provenance & lessons baked in:
    - Pre-XA global head (cfg.global_head_uses_pre_xa=True default).
      Cross-attention "accommodation" collapses InfoNCE rankings if both
      modalities can attend before the global ranking is computed. Independent
      encoders + late dot-product = CLIP.
    - Spatial heads (interface / contact / affinity) DO use post-XA features —
      partner-conditioning is biologically meaningful for "where does this drug
      contact this protein" but harmful for "how well do they match overall".
    - Asymmetric: drug and protein are different modalities with different
      shapes (drug ~30 atoms, protein ~500 residues), different input encoders
      (MoLFormer mean-pool + atom-MLP for drug; ESM-2 per-residue for protein),
      and one direction's "where do I bind" (drug→protein) is not symmetric to
      the other (protein→drug). Separate layer stacks per direction; NO weight
      tying.

Inputs (one batch):
    drug_global   : (B, 768)              — pre-cached MoLFormer-XL mean-pool
    drug_elem_ids : (B, N_atoms)          — int64; element index per atom (0..n_elements-1)
    drug_xyz      : (B, N_atoms, 3)       — float; atom coordinates (will be centered)
    drug_mask     : (B, N_atoms)          — bool; True = valid atom
    protein_emb   : (B, L_res, 1280)      — pre-cached ESM-2 per-residue
    protein_mask  : (B, L_res)            — bool; True = valid residue

Outputs (dict):
    global_features_drug    : (B, d_model) L2-normalized
    global_features_protein : (B, d_model) L2-normalized
    logit_scale             : scalar (learned InfoNCE temperature)
    interface_logits_drug   : (B, N_atoms)        — pre-sigmoid
    contact_logits          : (B, N_atoms, L_res) — pre-sigmoid
    affinity_pred           : (B,)                — pAffinity prediction
    drug_atom_post          : (B, N_atoms, d_model) — debug
    protein_per_res_post    : (B, L_res, d_model)   — debug
    drug_global_proj_pre    : (B, d_model)          — debug (pre-XA, pre-norm)
    protein_global_pre      : (B, d_model)          — debug (pre-XA, pre-norm)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# Standard small-molecule element vocabulary. Indices match canonical periodic-
# table priorities for SMILES; element 0 is reserved for "unknown / non-standard".
ELEMENT_VOCAB: list[str] = ["unk", "C", "H", "N", "O", "S", "F", "Cl", "Br", "P", "I"]
N_ELEMENTS = len(ELEMENT_VOCAB)
ELEMENT_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(ELEMENT_VOCAB)}


def element_to_idx(element: str) -> int:
    """Map a one- or two-letter element string to its vocab index."""
    e = element.strip().capitalize()
    return ELEMENT_TO_IDX.get(e, 0)  # unk = 0


# --------------------------------------------------------------------------- #
# 1. Atom-level encoder for drugs
# --------------------------------------------------------------------------- #
class AtomMLP(nn.Module):
    """Tiny per-atom encoder: (element_id, xyz) → d_model.

    A 3-layer MLP with a learned element embedding. Coordinates are centered
    per-molecule by the caller (translation-invariant input). Rotation
    invariance is NOT enforced — left to the model to learn from data, or
    swap to a GVP/EGNN at v3.1 if the contact head shows rotational artifacts.
    """

    def __init__(
        self,
        d_model: int,
        n_elements: int = N_ELEMENTS,
        elem_dim: int = 32,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.elem_embed = nn.Embedding(n_elements, elem_dim)
        self.mlp = nn.Sequential(
            nn.Linear(elem_dim + 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.elem_embed.weight, mean=0.0, std=0.02)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        elem_ids: torch.Tensor,   # (B, N) int64
        xyz: torch.Tensor,        # (B, N, 3) float
        mask: torch.Tensor,       # (B, N) bool
    ) -> torch.Tensor:            # (B, N, d_model)
        # Center per-molecule (translation invariance)
        # xyz_c = xyz - mean_over_valid_atoms(xyz)
        m = mask.unsqueeze(-1).to(xyz.dtype)                   # (B, N, 1)
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)      # (B, 1, 1)
        centroid = (xyz * m).sum(dim=1, keepdim=True) / denom   # (B, 1, 3)
        xyz_c = (xyz - centroid) * m                            # zero-out padded

        e = self.elem_embed(elem_ids)                          # (B, N, elem_dim)
        h = torch.cat([e, xyz_c], dim=-1)                      # (B, N, elem_dim + 3)
        out = self.mlp(h)                                      # (B, N, d_model)
        # Mask out padded atoms (forces downstream code to honor the mask)
        out = out * m
        return out


# --------------------------------------------------------------------------- #
# 2. Cross-attention layer (asymmetric stack uses two of these per direction)
# --------------------------------------------------------------------------- #
class CrossAttentionLayer(nn.Module):
    """Single cross-attention + FFN block, pre-norm style.

    Here we instantiate TWO independent stacks (drug→protein and
    protein→drug) since dtSFM is asymmetric.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        attn = self.cross_attn
        if hasattr(attn, "in_proj_weight") and attn.in_proj_weight is not None:
            nn.init.xavier_uniform_(attn.in_proj_weight)
            if attn.in_proj_bias is not None:
                nn.init.zeros_(attn.in_proj_bias)
        nn.init.xavier_uniform_(attn.out_proj.weight)
        if attn.out_proj.bias is not None:
            nn.init.zeros_(attn.out_proj.bias)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for ln in (self.norm_q, self.norm_kv, self.norm_ffn):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)

    def forward(
        self,
        query: torch.Tensor,           # (B, L_q, d_model)
        key_value: torch.Tensor,       # (B, L_kv, d_model)
        kv_pad_mask: torch.Tensor,     # (B, L_kv) — True = PADDED (PyTorch convention)
    ) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=kv_pad_mask, need_weights=False,
        )
        x = query + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


class AsymmetricCrossAttentionBlock(nn.Module):
    """Two independent cross-attention stacks (drug→protein, protein→drug).

    Per layer:
        new_drug    = drug_to_protein_layers[i](query=drug,    kv=protein, mask=protein_mask)
        new_protein = protein_to_drug_layers[i](query=protein, kv=drug,    mask=drug_mask)

    No weight tying (drug ≠ protein).
    """

    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
    ):
        super().__init__()
        self.drug_to_protein = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.protein_to_drug = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(
        self,
        drug_feat: torch.Tensor,       # (B, N_atoms, d_model)
        protein_feat: torch.Tensor,    # (B, L_res, d_model)
        drug_mask: torch.Tensor,       # (B, N_atoms) bool, True = VALID
        protein_mask: torch.Tensor,    # (B, L_res)  bool, True = VALID
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # PyTorch MultiheadAttention key_padding_mask: True = MASK OUT
        kv_pad_drug = ~drug_mask
        kv_pad_protein = ~protein_mask
        for l_d2p, l_p2d in zip(self.drug_to_protein, self.protein_to_drug):
            new_drug = l_d2p(drug_feat, protein_feat, kv_pad_mask=kv_pad_protein)
            new_protein = l_p2d(protein_feat, drug_feat, kv_pad_mask=kv_pad_drug)
            drug_feat, protein_feat = new_drug, new_protein
        return drug_feat, protein_feat


# --------------------------------------------------------------------------- #
# 3. Output heads
# --------------------------------------------------------------------------- #
class AttnPool(nn.Module):
    """Mask-aware attention pooling: (B, L, d) → (B, d)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(feat).squeeze(-1)              # (B, L)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        return (feat * weights).sum(dim=1)                # (B, d)


class InterfaceHead(nn.Module):
    """Per-atom is-interface MLP. (B, N_atoms, d) → (B, N_atoms)."""

    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


class AtomResContactHead(nn.Module):
    """Atom-residue contact predictor (bilinear).

    Drug atoms project through `proj_drug`, protein residues through
    `proj_protein` (separate weights — drug and protein chemistry are
    distinct), to a small contact subspace. Logit for atom i × residue j:

        logit[i, j] = (proj_drug[i] · proj_protein[j]) / sqrt(d_contact) * scale + bias
    """

    def __init__(self, d_model: int, d_contact: int = 64, init_temp: float = 1.0):
        super().__init__()
        self.proj_drug = nn.Linear(d_model, d_contact, bias=False)
        self.proj_protein = nn.Linear(d_model, d_contact, bias=False)
        self.bias = nn.Parameter(torch.zeros([]))
        self.scale = nn.Parameter(torch.tensor(init_temp))
        nn.init.xavier_uniform_(self.proj_drug.weight)
        nn.init.xavier_uniform_(self.proj_protein.weight)

    def forward(
        self,
        drug_feat: torch.Tensor,       # (B, N_atoms, d_model)
        protein_feat: torch.Tensor,    # (B, L_res, d_model)
    ) -> torch.Tensor:                  # (B, N_atoms, L_res)
        proj_d = self.proj_drug(drug_feat)
        proj_p = self.proj_protein(protein_feat)
        d_contact = proj_d.shape[-1]
        logits = torch.einsum("bid,bjd->bij", proj_d, proj_p) / (d_contact ** 0.5)
        return logits * self.scale + self.bias


class AffinityHead(nn.Module):
    """Pooled-drug ⊕ pooled-protein → pAffinity scalar."""

    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, drug_pooled: torch.Tensor, protein_pooled: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([drug_pooled, protein_pooled], dim=-1)   # (B, 2*d_model)
        return self.net(x).squeeze(-1)                         # (B,)


# --------------------------------------------------------------------------- #
# 4. Main v3 model
# --------------------------------------------------------------------------- #
class CALMEncoderV3(nn.Module):
    """dtSFM v3 — drug-target encoder.

    See module docstring for full architecture diagram and lesson provenance.

    Config (DictConfig) expected fields:
        d_model                    : int (default 512)
        cross_attention.n_layers   : int (default 2)
        cross_attention.n_heads    : int (default 8)
        cross_attention.d_ff       : int (default 2048)
        cross_attention.dropout    : float (default 0.1)
        dropout                    : float (default 0.1) — for heads + atom MLP
        tau                        : float (default 0.07) — initial InfoNCE temp
        max_scale                  : float (default 100) — clamp on logit_scale
        global_head_uses_pre_xa    : bool (default True) — LOCKED v2.5 lesson
        drug_global_dim            : int (default 768)  — MoLFormer dim
        protein_emb_dim            : int (default 1280) — ESM-2 dim
        atom.elem_dim              : int (default 32)
        atom.hidden                : int (default 256)
        heads.interface_hidden     : int (default 256)
        heads.contact_d            : int (default 64)
        heads.affinity_hidden      : int (default 256)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg.d_model)
        dropout = float(getattr(cfg, "dropout", 0.1))

        # ---------------- Drug global path (MoLFormer mean-pool → d_model) ---------------- #
        drug_global_dim = int(getattr(cfg, "drug_global_dim", 768))
        self.drug_global_proj = nn.Sequential(
            nn.Linear(drug_global_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        for m in self.drug_global_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # ---------------- Drug atom path (atom-MLP) ---------------- #
        atom_cfg = getattr(cfg, "atom", None) or {}
        elem_dim = int(getattr(atom_cfg, "elem_dim", 32)) if atom_cfg else 32
        atom_hidden = int(getattr(atom_cfg, "hidden", 256)) if atom_cfg else 256
        self.atom_encoder = AtomMLP(
            d_model=d_model,
            n_elements=N_ELEMENTS,
            elem_dim=elem_dim,
            hidden=atom_hidden,
            dropout=dropout,
        )

        # ---------------- Protein path (ESM-2 → d_model linear projection) ---------------- #
        protein_emb_dim = int(getattr(cfg, "protein_emb_dim", 1280))
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_emb_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        for m in self.protein_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # ---------------- Cross-attention block (asymmetric, k layers) ---------------- #
        ca_cfg = cfg.cross_attention
        self.cross_attn = AsymmetricCrossAttentionBlock(
            n_layers=int(ca_cfg.n_layers),
            d_model=d_model,
            n_heads=int(ca_cfg.n_heads),
            d_ff=int(ca_cfg.d_ff),
            dropout=float(getattr(ca_cfg, "dropout", dropout)),
        )

        # ---------------- Heads ---------------- #
        heads_cfg = getattr(cfg, "heads", None) or {}
        self.protein_pool_pre = AttnPool(d_model)   # PRE-XA, used by global head
        self.drug_pool_post = AttnPool(d_model)     # POST-XA, used by affinity head
        self.protein_pool_post = AttnPool(d_model)  # POST-XA, used by affinity head

        self.interface_head = InterfaceHead(
            d_model=d_model,
            hidden=int(getattr(heads_cfg, "interface_hidden", 256)),
            dropout=dropout,
        )
        self.contact_head = AtomResContactHead(
            d_model=d_model,
            d_contact=int(getattr(heads_cfg, "contact_d", 64)),
        )
        self.affinity_head = AffinityHead(
            d_model=d_model,
            hidden=int(getattr(heads_cfg, "affinity_hidden", 256)),
            dropout=dropout,
        )

        # InfoNCE temperature (matches v2 logit_scale convention)
        tau = float(getattr(cfg, "tau", 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / tau))
        self.max_scale = float(getattr(cfg, "max_scale", 100.0))

    # -------------------------------------------------------------------- #
    # Forward
    # -------------------------------------------------------------------- #
    def forward(
        self,
        drug_global: torch.Tensor,    # (B, 768)
        drug_elem_ids: torch.Tensor,  # (B, N_atoms) int64
        drug_xyz: torch.Tensor,       # (B, N_atoms, 3)
        drug_mask: torch.Tensor,      # (B, N_atoms) bool
        protein_emb: torch.Tensor,    # (B, L_res, 1280)
        protein_mask: torch.Tensor,   # (B, L_res) bool
    ) -> dict[str, torch.Tensor]:
        # 1. Encode each modality independently → d_model features
        drug_atom_pre = self.atom_encoder(drug_elem_ids, drug_xyz, drug_mask)  # (B, N, d)
        protein_per_res_pre = self.protein_proj(protein_emb)                   # (B, L, d)
        # Mask-zero protein padded positions (for cleanliness)
        protein_per_res_pre = protein_per_res_pre * protein_mask.unsqueeze(-1).to(protein_per_res_pre.dtype)

        # 2. Asymmetric cross-attention → post-XA features (used by spatial heads)
        drug_atom_post, protein_per_res_post = self.cross_attn(
            drug_atom_pre, protein_per_res_pre, drug_mask, protein_mask,
        )

        # 3. Global head — PRE-XA features (locked v2.5 lesson, B-11)
        global_uses_pre_xa = bool(getattr(self.cfg, "global_head_uses_pre_xa", True))
        if global_uses_pre_xa:
            drug_global_proj = self.drug_global_proj(drug_global)              # (B, d_model)
            protein_global_pre = self.protein_pool_pre(protein_per_res_pre, protein_mask)
        else:
            # Legacy / ablation path (NOT recommended — fails Δcos diagnostic)
            drug_global_proj = self.drug_pool_post(drug_atom_post, drug_mask)
            protein_global_pre = self.protein_pool_post(protein_per_res_post, protein_mask)

        global_drug_norm = F.normalize(drug_global_proj, dim=-1)
        global_protein_norm = F.normalize(protein_global_pre, dim=-1)

        # 4. Spatial heads — POST-XA features
        interface_logits_drug = self.interface_head(drug_atom_post)            # (B, N)
        contact_logits = self.contact_head(drug_atom_post, protein_per_res_post)  # (B, N, L)

        # 5. Affinity head — POST-XA pooled (drug + protein)
        drug_pooled_post = self.drug_pool_post(drug_atom_post, drug_mask)
        protein_pooled_post = self.protein_pool_post(protein_per_res_post, protein_mask)
        affinity_pred = self.affinity_head(drug_pooled_post, protein_pooled_post)  # (B,)

        return {
            # Global (used by InfoNCE)
            "global_features_drug": global_drug_norm,
            "global_features_protein": global_protein_norm,
            "logit_scale": self.logit_scale.exp().clamp(max=self.max_scale),
            # Spatial heads
            "interface_logits_drug": interface_logits_drug,
            "contact_logits": contact_logits,
            "affinity_pred": affinity_pred,
            # Debug / for ablation diagnostics (Δcos check at B-2 smoketest)
            "drug_atom_post": drug_atom_post,
            "protein_per_res_post": protein_per_res_post,
            "drug_global_proj_pre": drug_global_proj,
            "protein_global_pre": protein_global_pre,
            "_global_head_uses_pre_xa": torch.tensor(global_uses_pre_xa),
        }

    # -------------------------------------------------------------------- #
    # Convenience: parameter count
    # -------------------------------------------------------------------- #
    def count_parameters(self) -> dict[str, int]:
        """Return per-component trainable parameter counts."""
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "atom_encoder": n(self.atom_encoder),
            "protein_proj": n(self.protein_proj),
            "drug_global_proj": n(self.drug_global_proj),
            "cross_attn": n(self.cross_attn),
            "interface_head": n(self.interface_head),
            "contact_head": n(self.contact_head),
            "affinity_head": n(self.affinity_head),
            "pools_and_temp": (
                n(self.protein_pool_pre) + n(self.drug_pool_post) + n(self.protein_pool_post)
                + 1  # logit_scale
            ),
            "total": n(self),
        }


def build_encoder_v3_model(cfg: DictConfig) -> nn.Module:
    """Factory for the dtSFM v3 model. Mirrors build_encoder_v2_model API."""
    model_cfg = cfg.model.encoder if hasattr(cfg, "model") else cfg
    return CALMEncoderV3(model_cfg)
