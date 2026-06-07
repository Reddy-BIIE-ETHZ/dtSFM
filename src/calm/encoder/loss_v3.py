"""
Multi-task loss for dtSFM v3.

Combines four losses with magnitude normalization (EMA on each loss's running
absolute value) so that all four end up at scale ~1.0 before the user-set
priority weights (α_k) are applied:

    L_total = α_global   · L_global   / EMA(|L_global|)
            + α_interface · L_interface / EMA(|L_interface|)
            + α_contact   · L_contact   / EMA(|L_contact|)
            + α_affinity  · L_affinity  / EMA(|L_affinity|)

Default α_k:
    α_global = 1.0,  α_interface = 1.0,  α_contact = 1.0,  α_affinity = 0.5

The EMA is updated with the *current* (un-normalized) loss values BEFORE the
division, so the first few steps use the EMA's initialization (1.0) and the
normalization stabilizes within ~100 steps. Setting `ema_decay = 0` disables
normalization (sanity check / ablation).

Loss flavors:
    InfoNCE (global): symmetric (drug→protein and protein→drug), using the
        model's learned logit_scale. Standard CLIP form.
    BCE (interface):  pos-weighted binary cross-entropy on per-atom is-interface
        logits. Mask-aware (only valid atoms contribute).
    BCE (contact):    pos-weighted binary cross-entropy on (atoms × residues)
        logits. Mask = outer product of drug_mask and protein_mask. Per-batch
        pos_weight = N_neg / max(N_pos, 1).
    MSE (affinity):   mean-squared error on raw pAffinity. Optional mask for
        pairs with non-`=` qualifier (PDBbind affinity_log entries with `<`,
        `<=`, `~` qualifiers should be excluded — drop them in the dataloader,
        or pass `affinity_mask=False` for those rows).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Component losses
# --------------------------------------------------------------------------- #
def info_nce_loss(
    z_drug: torch.Tensor,       # (B, d) L2-normalized
    z_protein: torch.Tensor,    # (B, d) L2-normalized
    logit_scale: torch.Tensor,  # scalar
) -> torch.Tensor:
    """Symmetric InfoNCE / CLIP loss.

    Each row's positive is the diagonal element. Returns mean of the two
    cross-entropy directions (drug→protein, protein→drug).
    """
    B = z_drug.shape[0]
    logits = logit_scale * (z_drug @ z_protein.t())   # (B, B)
    targets = torch.arange(B, device=logits.device)
    L_d2p = F.cross_entropy(logits, targets)
    L_p2d = F.cross_entropy(logits.t(), targets)
    return 0.5 * (L_d2p + L_p2d)


def per_atom_interface_loss(
    logits: torch.Tensor,   # (B, N_atoms)
    targets: torch.Tensor,  # (B, N_atoms) {0, 1} or bool
    mask: torch.Tensor,     # (B, N_atoms) bool, True = valid atom
) -> torch.Tensor:
    """Pos-weighted BCE on per-atom is-interface logits.

    pos_weight = (#negatives in batch) / max(#positives in batch, 1).
    """
    valid = mask.bool()
    if valid.sum() == 0:
        return logits.new_zeros(())

    t = targets.float()
    n_pos = (t * valid.float()).sum().clamp(min=1.0)
    n_neg = ((1.0 - t) * valid.float()).sum().clamp(min=1.0)
    pos_weight = (n_neg / n_pos).detach()

    loss_per = F.binary_cross_entropy_with_logits(
        logits, t, reduction="none", pos_weight=pos_weight,
    )
    loss = (loss_per * valid.float()).sum() / valid.float().sum()
    return loss


def atom_residue_contact_loss(
    logits: torch.Tensor,         # (B, N, L)
    targets: torch.Tensor,        # (B, N, L) {0, 1} or bool
    drug_mask: torch.Tensor,      # (B, N) bool
    protein_mask: torch.Tensor,   # (B, L) bool
) -> torch.Tensor:
    """Pos-weighted BCE on the atom-residue contact map.

    Joint mask = drug_mask ⊗ protein_mask. pos_weight set per-batch.
    """
    joint = drug_mask.unsqueeze(2) & protein_mask.unsqueeze(1)   # (B, N, L)
    if joint.sum() == 0:
        return logits.new_zeros(())

    t = targets.float()
    n_pos = (t * joint.float()).sum().clamp(min=1.0)
    n_neg = ((1.0 - t) * joint.float()).sum().clamp(min=1.0)
    pos_weight = (n_neg / n_pos).detach()

    loss_per = F.binary_cross_entropy_with_logits(
        logits, t, reduction="none", pos_weight=pos_weight,
    )
    loss = (loss_per * joint.float()).sum() / joint.float().sum()
    return loss


def affinity_mse_loss(
    pred: torch.Tensor,             # (B,)
    target: torch.Tensor,           # (B,) pAffinity (=−log10 K_M)
    valid: torch.Tensor | None,     # (B,) bool — True = use this row
) -> torch.Tensor:
    """MSE on pAffinity. Rows with valid=False are dropped from the average."""
    if valid is None:
        return F.mse_loss(pred, target)
    valid_f = valid.float()
    n = valid_f.sum().clamp(min=1.0)
    sq = (pred - target).pow(2)
    return (sq * valid_f).sum() / n


# --------------------------------------------------------------------------- #
# 2. Multi-task loss with EMA magnitude normalization
# --------------------------------------------------------------------------- #
@dataclass
class MultiTaskLossWeights:
    """Initial priority weights (α_k)."""
    global_: float = 1.0
    interface: float = 1.0
    contact: float = 1.0
    affinity: float = 0.5


class MultiTaskLossV3(nn.Module):
    """Magnitude-normalized 4-head loss.

    Stores a non-trainable buffer `ema_<key>` per loss head; updated with
    `ema_decay * old + (1 - ema_decay) * abs(loss_value)` at each forward.
    Normalized loss is `loss_value / max(ema_<key>, ema_floor)`.

    Set `ema_decay=0.0` to disable normalization (sanity check).
    """

    def __init__(
        self,
        weights: MultiTaskLossWeights | None = None,
        ema_decay: float = 0.99,
        ema_floor: float = 1e-6,
    ):
        super().__init__()
        self.weights = weights or MultiTaskLossWeights()
        self.ema_decay = float(ema_decay)
        self.ema_floor = float(ema_floor)

        # Non-trainable running averages — initialized at 1.0 so first step
        # produces normalized losses on the same order as raw.
        for k in ("global", "interface", "contact", "affinity"):
            self.register_buffer(f"ema_{k}", torch.tensor(1.0))

    @torch.no_grad()
    def _update_ema(self, key: str, value: torch.Tensor) -> None:
        if self.ema_decay <= 0.0:
            return
        buf = getattr(self, f"ema_{key}")
        new_val = self.ema_decay * buf + (1.0 - self.ema_decay) * value.detach().abs()
        buf.copy_(new_val.clamp(min=self.ema_floor))

    def _normalize(self, key: str, loss_val: torch.Tensor) -> torch.Tensor:
        if self.ema_decay <= 0.0:
            return loss_val
        buf = getattr(self, f"ema_{key}")
        return loss_val / buf.clamp(min=self.ema_floor)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the four losses, update EMAs, return (total, breakdown).

        Expected keys in `outputs`:
            global_features_drug, global_features_protein, logit_scale,
            interface_logits_drug, contact_logits, affinity_pred

        Expected keys in `targets`:
            interface_target_drug      : (B, N_atoms) {0, 1}
            contact_target             : (B, N_atoms, L_res) {0, 1}
            affinity_target            : (B,) float
            affinity_valid             : (B,) bool — optional; True = use this row
            drug_mask                  : (B, N_atoms) bool
            protein_mask               : (B, L_res)  bool
        """
        # --- Global (InfoNCE) ---
        L_global = info_nce_loss(
            outputs["global_features_drug"],
            outputs["global_features_protein"],
            outputs["logit_scale"],
        )
        # --- Interface (per-atom, drug side) ---
        L_interface = per_atom_interface_loss(
            outputs["interface_logits_drug"],
            targets["interface_target_drug"],
            targets["drug_mask"],
        )
        # --- Contact (atom × residue) ---
        L_contact = atom_residue_contact_loss(
            outputs["contact_logits"],
            targets["contact_target"],
            targets["drug_mask"],
            targets["protein_mask"],
        )
        # --- Affinity (MSE) ---
        L_affinity = affinity_mse_loss(
            outputs["affinity_pred"],
            targets["affinity_target"],
            targets.get("affinity_valid", None),
        )

        # Update EMAs with the raw (un-normalized) losses
        self._update_ema("global", L_global)
        self._update_ema("interface", L_interface)
        self._update_ema("contact", L_contact)
        self._update_ema("affinity", L_affinity)

        # Normalize and apply priority weights
        L_g = self.weights.global_   * self._normalize("global",    L_global)
        L_i = self.weights.interface * self._normalize("interface", L_interface)
        L_c = self.weights.contact   * self._normalize("contact",   L_contact)
        L_a = self.weights.affinity  * self._normalize("affinity",  L_affinity)

        L_total = L_g + L_i + L_c + L_a

        breakdown = {
            "loss_total":     L_total.detach(),
            # Raw (for monitoring head behavior)
            "loss_global_raw":    L_global.detach(),
            "loss_interface_raw": L_interface.detach(),
            "loss_contact_raw":   L_contact.detach(),
            "loss_affinity_raw":  L_affinity.detach(),
            # Normalized + weighted (sums to L_total)
            "loss_global":    L_g.detach(),
            "loss_interface": L_i.detach(),
            "loss_contact":   L_c.detach(),
            "loss_affinity":  L_a.detach(),
            # EMAs (for warmup-completeness check at start of training)
            "ema_global":    self.ema_global.detach(),
            "ema_interface": self.ema_interface.detach(),
            "ema_contact":   self.ema_contact.detach(),
            "ema_affinity":  self.ema_affinity.detach(),
        }
        return L_total, breakdown


# --------------------------------------------------------------------------- #
# 3. Helpers for deriving per-atom and per-residue interface targets from a
#    contact map. The training pipeline calls these inside the dataloader.
# --------------------------------------------------------------------------- #
def derive_atom_interface_from_contacts(
    contact_map: torch.Tensor,    # (N_atoms, L_res) bool
) -> torch.Tensor:                # (N_atoms,) {0, 1}
    """An atom is 'interface' if it has any contact with any protein residue."""
    return contact_map.any(dim=-1).long()


def derive_residue_interface_from_contacts(
    contact_map: torch.Tensor,    # (N_atoms, L_res) bool
) -> torch.Tensor:                # (L_res,) {0, 1}
    """A residue is 'interface' if it has any contact with any drug atom."""
    return contact_map.any(dim=0).long()
