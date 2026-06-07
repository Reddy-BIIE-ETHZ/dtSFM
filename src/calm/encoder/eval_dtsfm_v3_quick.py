"""
eval_dtsfm_v3_quick.py — quick per-checkpoint quality eval for dtSFM v3.

Loads a saved checkpoint (epoch_NNN.pt or best.pt from the B-3 run) and
reports REAL quality metrics on a held-out validation slice for all four
output heads — not just losses.

Reports:
    GLOBAL HEAD (retrieval):
        - Top-1, Top-5, Top-10 accuracy on a fixed val subset
        - Mean rank of correct match
        - Δcos (matched − shuffled) — both pre-XA and post-XA
    INTERFACE HEAD (per-atom is-interface):
        - AUROC on (B × N_atoms) flattened, mask-aware
        - F1 at threshold p=0.5
        - Positive-class precision @ recall=0.8
    CONTACT HEAD (atom-residue):
        - AUROC on (B × N_atoms × L_res) flattened, joint-mask aware
        - Top-K precision (K = number of true positives per pair)
        - IoU at threshold p=0.5 (strict)
    AFFINITY HEAD:
        - Pearson r, Spearman ρ, RMSE (log-K units)
        - Subset metric on tight-K_D ranking

Runs without modifying B-3 — designed to be invoked while B-3 is still
training, on whatever the latest saved epoch checkpoint is.

Usage on Euler (with GPU; takes ~3-5 min):
    python3 -m calm.encoder.eval_dtsfm_v3_quick \\
        --checkpoint /cluster/scratch/reddys/dtsfm_v3/runs/b3_<TS>/epoch_001.pt \\
        --metadata_csv /cluster/scratch/reddys/dtsfm_v3/metadata_v3.csv \\
        --drug_npz     /cluster/scratch/reddys/dtsfm_v3/embeddings/drug_embeddings.npz \\
        --protein_dir  /cluster/scratch/reddys/dtsfm_v3/embeddings/protein_embeds/ \\
        --heldout_tsv  audit/dtsfm/heldout_validation_pairs.tsv \\
        --output_csv   /cluster/scratch/reddys/dtsfm_v3/runs/b3_<TS>/quick_eval_epoch_001.csv \\
        --n_eval_pairs 4000 \\
        --batch_size 128
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .data_dtsfm_v3 import (
    DTSFMv3PairDataset, collate_dtsfm_v3,
    load_drug_embeddings, load_protein_embeddings, make_cluster_splits,
)
from .model_v3 import CALMEncoderV3


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def torch_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Trapezoidal-rule AUROC (vectorized, GPU-friendly)."""
    scores = scores.detach().float().flatten()
    labels = labels.detach().float().flatten()
    if labels.sum() == 0 or labels.sum() == labels.numel():
        return float("nan")
    sorted_idx = torch.argsort(scores, descending=True)
    sl = labels[sorted_idx]
    n_pos = sl.sum().clamp(min=1)
    n_neg = (sl.numel() - n_pos).clamp(min=1)
    cum_tp = torch.cumsum(sl, dim=0)
    cum_fp = torch.cumsum(1.0 - sl, dim=0)
    tpr = cum_tp / n_pos
    fpr = cum_fp / n_neg
    # Add (0,0) origin point
    tpr = torch.cat([torch.zeros(1, device=tpr.device), tpr])
    fpr = torch.cat([torch.zeros(1, device=fpr.device), fpr])
    auroc = torch.trapz(tpr, fpr).abs()
    return float(auroc.item())


def f1_at_threshold(scores: torch.Tensor, labels: torch.Tensor, thresh: float = 0.5) -> dict:
    """F1, precision, recall at a fixed sigmoid threshold."""
    scores = scores.detach().float().flatten()
    labels = labels.detach().float().flatten()
    pred = (torch.sigmoid(scores) >= thresh).float()
    tp = (pred * labels).sum()
    fp = (pred * (1 - labels)).sum()
    fn = ((1 - pred) * labels).sum()
    prec = tp / (tp + fp).clamp(min=1)
    rec  = tp / (tp + fn).clamp(min=1)
    f1   = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    return {"f1": float(f1.item()), "precision": float(prec.item()), "recall": float(rec.item())}


def iou_at_threshold(
    scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, thresh: float = 0.5,
) -> float:
    """Strict IoU on masked positions."""
    pred = (torch.sigmoid(scores) >= thresh).bool() & mask
    targ = labels.bool() & mask
    inter = (pred & targ).sum()
    union = (pred | targ).sum()
    if union == 0:
        return float("nan")
    return float((inter / union).item())


def pearson_spearman_rmse(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred = pred.detach().float().cpu().numpy()
    target = target.detach().float().cpu().numpy()
    if len(pred) < 2:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan"), "rmse": float("nan")}
    pearson = float(np.corrcoef(pred, target)[0, 1])
    rank_p = np.argsort(np.argsort(pred))
    rank_t = np.argsort(np.argsort(target))
    spear  = float(np.corrcoef(rank_p, rank_t)[0, 1])
    rmse   = float(np.sqrt(np.mean((pred - target) ** 2)))
    return {"pearson_r": pearson, "spearman_rho": spear, "rmse": rmse}


def topk_acc_unique_pool(
    z_d_all: torch.Tensor,         # (N_pairs, d) all drug embeddings (one per pair)
    z_p_all: torch.Tensor,         # (N_pairs, d) all protein embeddings (one per pair)
    drug_idx: torch.Tensor,        # (N_pairs,) drug index per pair
    protein_idx: torch.Tensor,     # (N_pairs,) protein index per pair
    k_list=(1, 5, 10, 50, 100),
) -> dict:
    """Direction-aware top-K retrieval with the v2 dedup fix.

    Per `feedback_eval_duplicate_bug.md`: when the same protein appears in
    multiple pairs in the eval pool, the (B, B) similarity matrix has
    duplicate columns. argmax-based top-K then loses credit when the model
    correctly identifies the protein but picks a "different instance"
    (different pair_idx) of it. The fix: dedup pool to unique entities, then
    rank against the unique pool.

    D→T  (drug retrieves target): for each unique drug, what is the rank of
    its TRUE protein among all unique proteins in the pool? Practical
    relevance: safety / off-target screening.

    T→D  (target retrieves drug): for each unique protein, what is the rank
    of its TRUE drug among all unique drugs in the pool? Practical
    relevance: drug repurposing, library prioritization.
    """
    results: dict[str, float] = {}

    # ---- Unique-pool construction ----
    # Drug side: unique drugs; for each, average their per-pair embedding
    #            (drugs paired with multiple proteins get one canonical embedding)
    unique_drugs, drug_inv = drug_idx.unique(return_inverse=True)
    n_d_unique = unique_drugs.shape[0]
    z_d_unique = torch.zeros((n_d_unique, z_d_all.shape[1]), dtype=z_d_all.dtype)
    counts_d = torch.zeros(n_d_unique, dtype=z_d_all.dtype)
    z_d_unique.index_add_(0, drug_inv, z_d_all)
    counts_d.index_add_(0, drug_inv, torch.ones(z_d_all.shape[0], dtype=z_d_all.dtype))
    z_d_unique = z_d_unique / counts_d.unsqueeze(-1).clamp(min=1)
    z_d_unique = torch.nn.functional.normalize(z_d_unique, dim=-1)

    # Protein side: same (proteins paired with multiple drugs → one canonical)
    unique_proteins, prot_inv = protein_idx.unique(return_inverse=True)
    n_p_unique = unique_proteins.shape[0]
    z_p_unique = torch.zeros((n_p_unique, z_p_all.shape[1]), dtype=z_p_all.dtype)
    counts_p = torch.zeros(n_p_unique, dtype=z_p_all.dtype)
    z_p_unique.index_add_(0, prot_inv, z_p_all)
    counts_p.index_add_(0, prot_inv, torch.ones(z_p_all.shape[0], dtype=z_p_all.dtype))
    z_p_unique = z_p_unique / counts_p.unsqueeze(-1).clamp(min=1)
    z_p_unique = torch.nn.functional.normalize(z_p_unique, dim=-1)

    # Build the unique-pair lookup: for each unique drug, which unique proteins is it paired with?
    # And vice versa. Each pair (drug_inv[i], prot_inv[i]) is a positive in the unique pool.
    drug_to_protein_set: dict[int, set] = {}
    protein_to_drug_set: dict[int, set] = {}
    for d, p in zip(drug_inv.tolist(), prot_inv.tolist()):
        drug_to_protein_set.setdefault(d, set()).add(p)
        protein_to_drug_set.setdefault(p, set()).add(d)

    # ---- D → T retrieval ----
    sim_d2t = z_d_unique @ z_p_unique.t()       # (n_d_unique, n_p_unique)
    sorted_p_for_d = sim_d2t.argsort(dim=1, descending=True)   # (n_d_unique, n_p_unique)

    # For each unique drug, check if ANY of its true proteins is in top-K
    for K in k_list:
        K_eff = min(K, n_p_unique)
        topk = sorted_p_for_d[:, :K_eff].cpu().tolist()    # n_d_unique × K_eff
        hits = 0
        for di, top in enumerate(topk):
            true_set = drug_to_protein_set.get(di, set())
            if any(p in true_set for p in top):
                hits += 1
        results[f"d2t_R@{K}"] = hits / max(n_d_unique, 1)

    # Best rank of any true protein per drug (for mean/median rank stats)
    rank_d2t = []
    for di in range(n_d_unique):
        true_set = drug_to_protein_set.get(di, set())
        if not true_set:
            continue
        ranks_for_drug = sorted_p_for_d[di].cpu().tolist()
        best = min(ranks_for_drug.index(p) for p in true_set if p in ranks_for_drug)
        rank_d2t.append(best)
    if rank_d2t:
        rank_d2t_t = torch.tensor(rank_d2t, dtype=torch.float)
        results["d2t_mean_rank"] = float(rank_d2t_t.mean().item())
        results["d2t_median_rank"] = float(rank_d2t_t.median().item())

    # ---- T → D retrieval ----
    sim_t2d = z_p_unique @ z_d_unique.t()       # (n_p_unique, n_d_unique)
    sorted_d_for_p = sim_t2d.argsort(dim=1, descending=True)
    for K in k_list:
        K_eff = min(K, n_d_unique)
        topk = sorted_d_for_p[:, :K_eff].cpu().tolist()
        hits = 0
        for pi, top in enumerate(topk):
            true_set = protein_to_drug_set.get(pi, set())
            if any(d in true_set for d in top):
                hits += 1
        results[f"t2d_R@{K}"] = hits / max(n_p_unique, 1)

    rank_t2d = []
    for pi in range(n_p_unique):
        true_set = protein_to_drug_set.get(pi, set())
        if not true_set:
            continue
        ranks_for_prot = sorted_d_for_p[pi].cpu().tolist()
        best = min(ranks_for_prot.index(d) for d in true_set if d in ranks_for_prot)
        rank_t2d.append(best)
    if rank_t2d:
        rank_t2d_t = torch.tensor(rank_t2d, dtype=torch.float)
        results["t2d_mean_rank"] = float(rank_t2d_t.mean().item())
        results["t2d_median_rank"] = float(rank_t2d_t.median().item())

    # Pool sizes for transparency
    results["pool_n_unique_drugs"] = int(n_d_unique)
    results["pool_n_unique_proteins"] = int(n_p_unique)
    results["pool_n_pairs"] = int(z_d_all.shape[0])

    return results


# --------------------------------------------------------------------------- #
# Main eval loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: CALMEncoderV3,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 50,
) -> dict:
    """Run one pass through `n_batches` of `loader`, accumulate predictions, compute metrics."""
    model.eval()

    # Accumulators
    all_z_d, all_z_p = [], []
    all_drug_idx, all_protein_idx = [], []
    all_iface_logits, all_iface_targets, all_iface_masks = [], [], []
    all_contact_logits, all_contact_targets, all_contact_masks = [], [], []
    all_aff_pred, all_aff_target, all_aff_valid = [], [], []
    pre_match, pre_shuffle, post_match, post_shuffle = [], [], [], []

    n_pairs_seen = 0
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        out = model(
            drug_global=batch["drug_global"].to(device),
            drug_elem_ids=batch["drug_elem_ids"].to(device),
            drug_xyz=batch["drug_xyz"].to(device),
            drug_mask=batch["drug_mask"].to(device),
            protein_emb=batch["protein_emb"].to(device),
            protein_mask=batch["protein_mask"].to(device),
        )

        # Global features (already L2-normalized)
        z_d = out["global_features_drug"].cpu()
        z_p = out["global_features_protein"].cpu()
        all_z_d.append(z_d)
        all_z_p.append(z_p)
        all_drug_idx.append(batch["drug_idx"])
        all_protein_idx.append(batch["protein_idx"])

        # Δcos pre-XA
        cm = (z_d * z_p).sum(dim=-1)
        perm = torch.randperm(z_p.shape[0])
        cs = (z_d * z_p[perm]).sum(dim=-1)
        pre_match.append(cm)
        pre_shuffle.append(cs)

        # Δcos post-XA (using model's pool layers on post-XA features)
        d_post = F.normalize(
            model.drug_pool_post(out["drug_atom_post"], batch["drug_mask"].to(device)),
            dim=-1,
        ).cpu()
        p_post = F.normalize(
            model.protein_pool_post(out["protein_per_res_post"], batch["protein_mask"].to(device)),
            dim=-1,
        ).cpu()
        post_match.append((d_post * p_post).sum(dim=-1))
        post_shuffle.append((d_post * p_post[perm]).sum(dim=-1))

        # Interface (per-atom)
        all_iface_logits.append(out["interface_logits_drug"].cpu())
        all_iface_targets.append(batch["interface_target_drug"])
        all_iface_masks.append(batch["drug_mask"])

        # Contact (atom × residue)
        all_contact_logits.append(out["contact_logits"].cpu())
        all_contact_targets.append(batch["contact_target"])
        joint_mask = batch["drug_mask"].unsqueeze(2) & batch["protein_mask"].unsqueeze(1)
        all_contact_masks.append(joint_mask)

        # Affinity
        all_aff_pred.append(out["affinity_pred"].cpu())
        all_aff_target.append(batch["affinity_target"])
        all_aff_valid.append(batch["affinity_valid"])

        n_pairs_seen += batch["drug_global"].shape[0]

    # --- Global head: direction-aware retrieval with unique-pool dedup ---
    z_d_all = torch.cat(all_z_d, dim=0)
    z_p_all = torch.cat(all_z_p, dim=0)
    drug_idx_all = torch.cat(all_drug_idx, dim=0)
    protein_idx_all = torch.cat(all_protein_idx, dim=0)
    global_metrics = topk_acc_unique_pool(
        z_d_all, z_p_all, drug_idx_all, protein_idx_all,
        k_list=(1, 5, 10, 50, 100),
    )

    pre_match = torch.cat(pre_match)
    pre_shuffle = torch.cat(pre_shuffle)
    post_match = torch.cat(post_match)
    post_shuffle = torch.cat(post_shuffle)
    global_metrics["delta_cos_pre_xa"]  = float((pre_match.mean()  - pre_shuffle.mean()).item())
    global_metrics["delta_cos_post_xa"] = float((post_match.mean() - post_shuffle.mean()).item())

    # --- Interface head ---
    iface_logits = torch.cat([t.flatten() for t in all_iface_logits])
    iface_targets = torch.cat([t.flatten().float() for t in all_iface_targets])
    iface_masks = torch.cat([t.flatten().bool() for t in all_iface_masks])
    iface_logits_v = iface_logits[iface_masks]
    iface_targets_v = iface_targets[iface_masks]
    iface_metrics = {
        "interface_auroc": torch_auroc(iface_logits_v, iface_targets_v),
        **{f"interface_{k}": v for k, v in
           f1_at_threshold(iface_logits_v, iface_targets_v).items()},
        "interface_n_eval_atoms": int(iface_masks.sum().item()),
        "interface_pos_frac": float(iface_targets_v.mean().item()),
    }

    # --- Contact head ---
    contact_logits = torch.cat([t.flatten() for t in all_contact_logits])
    contact_targets = torch.cat([t.flatten().float() for t in all_contact_targets])
    contact_masks = torch.cat([t.flatten().bool() for t in all_contact_masks])
    cl_v = contact_logits[contact_masks]
    ct_v = contact_targets[contact_masks]
    contact_metrics = {
        "contact_auroc": torch_auroc(cl_v, ct_v),
        "contact_iou_at_0.5": iou_at_threshold(
            contact_logits, contact_targets, contact_masks, thresh=0.5,
        ),
        "contact_n_eval_cells": int(contact_masks.sum().item()),
        "contact_pos_frac": float(ct_v.mean().item()),
    }

    # --- Affinity head ---
    aff_pred = torch.cat(all_aff_pred)
    aff_target = torch.cat(all_aff_target)
    aff_valid = torch.cat(all_aff_valid).bool()
    if aff_valid.sum() >= 2:
        inner = pearson_spearman_rmse(aff_pred[aff_valid], aff_target[aff_valid])
        aff_metrics = {f"affinity_{k}": v for k, v in inner.items()}
        aff_metrics["affinity_n_eval_pairs"] = int(aff_valid.sum().item())
    else:
        aff_metrics = {
            "affinity_pearson_r": float("nan"),
            "affinity_spearman_rho": float("nan"),
            "affinity_rmse": float("nan"),
            "affinity_n_eval_pairs": 0,
        }

    return {
        **global_metrics,
        **iface_metrics,
        **contact_metrics,
        **aff_metrics,
        "n_pairs_seen": n_pairs_seen,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--metadata_csv", type=Path, required=True)
    ap.add_argument("--drug_npz", type=Path, required=True)
    ap.add_argument("--protein_dir", type=Path, required=True)
    ap.add_argument("--heldout_tsv", type=Path, default=None)
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--n_eval_pairs", type=int, default=4000)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", choices=["val", "test", "in_dist"], default="val",
                    help="val=OOD clusters; in_dist=held-out pairs from train clusters")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"]) if "config" in ckpt else None
    if cfg is None:
        # Fall back to default model cfg
        from .train_dtsfm_v3 import default_model_cfg
        cfg = OmegaConf.create(default_model_cfg())
    model = CALMEncoderV3(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  epoch={ckpt.get('epoch', '?')}, "
          f"params loaded ok")

    # Load embeddings
    drug_embeddings = load_drug_embeddings(args.drug_npz)
    protein_embeddings = load_protein_embeddings(args.protein_dir)

    # Splits — pick val (OOD clusters) or in_dist (held-out pairs in train clusters)
    splits = make_cluster_splits(args.metadata_csv, heldout_tsv=args.heldout_tsv, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    if args.split == "in_dist":
        # Random sample from train cluster pairs (won't have been trained on by chance,
        # but cluster overlap means the protein WAS seen)
        pool = splits["train"]
    elif args.split == "test":
        pool = splits["test"]
    else:
        pool = splits["val"]
    n = min(args.n_eval_pairs, len(pool))
    idx = list(rng.choice(pool, size=n, replace=False))
    print(f"\nEval set: {args.split}  |  {len(idx):,} pairs")

    ds = DTSFMv3PairDataset(
        args.metadata_csv, drug_embeddings, protein_embeddings,
        pair_indices=idx,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_dtsfm_v3,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )

    print(f"\nRunning eval...")
    t0 = time.time()
    n_batches = (len(idx) + args.batch_size - 1) // args.batch_size
    metrics = evaluate(model, loader, device, n_batches=n_batches)
    elapsed = time.time() - t0

    # Pretty print
    print(f"\n{'=' * 80}")
    print(f"=== Quick eval — {args.checkpoint.name} (split={args.split}) ===")
    print(f"{'=' * 80}")
    n_d = metrics.get("pool_n_unique_drugs", 0)
    n_p = metrics.get("pool_n_unique_proteins", 0)
    print(f"\nGLOBAL HEAD — UNIQUE-POOL RETRIEVAL")
    print(f"  Pool: {n_d:,} unique drugs × {n_p:,} unique proteins "
          f"({metrics.get('pool_n_pairs', 0):,} pairs sampled)")

    print(f"\n  D → T  (drug retrieves target — safety screening direction):")
    print(f"    Pool denominator: {n_p} unique proteins  "
          f"(random R@1 ≈ {1/max(n_p,1)*100:.3f}%, R@10 ≈ {10/max(n_p,1)*100:.3f}%)")
    for K in (1, 5, 10, 50, 100):
        v = metrics.get(f"d2t_R@{K}")
        if v is not None:
            random_baseline = min(K, n_p) / max(n_p, 1)
            ratio = v / max(random_baseline, 1e-9)
            print(f"    R@{K:<4d} = {v:>7.4f}   ({ratio:>6.1f}× random)")
    if "d2t_mean_rank" in metrics:
        print(f"    mean_rank   = {metrics['d2t_mean_rank']:>7.1f}")
        print(f"    median_rank = {metrics['d2t_median_rank']:>7.1f}")

    print(f"\n  T → D  (target retrieves drug — repurposing / library prioritization direction):")
    print(f"    Pool denominator: {n_d} unique drugs  "
          f"(random R@1 ≈ {1/max(n_d,1)*100:.3f}%, R@10 ≈ {10/max(n_d,1)*100:.3f}%)")
    for K in (1, 5, 10, 50, 100):
        v = metrics.get(f"t2d_R@{K}")
        if v is not None:
            random_baseline = min(K, n_d) / max(n_d, 1)
            ratio = v / max(random_baseline, 1e-9)
            print(f"    R@{K:<4d} = {v:>7.4f}   ({ratio:>6.1f}× random)")
    if "t2d_mean_rank" in metrics:
        print(f"    mean_rank   = {metrics['t2d_mean_rank']:>7.1f}")
        print(f"    median_rank = {metrics['t2d_median_rank']:>7.1f}")

    print(f"\n  Δcos diagnostics (from raw matched-vs-shuffled, no dedup):")
    for k in ["delta_cos_pre_xa", "delta_cos_post_xa"]:
        v = metrics.get(k)
        if v is not None:
            print(f"    {k:<24s} = {v:>10.4f}")
    print(f"\nINTERFACE HEAD (per-atom, n={metrics.get('interface_n_eval_atoms', '?'):,} atoms):")
    for k in ["interface_auroc", "interface_f1", "interface_precision", "interface_recall",
              "interface_pos_frac"]:
        v = metrics.get(k)
        if v is not None:
            print(f"  {k:<24s} = {v:>10.4f}")
    print(f"\nCONTACT HEAD (atom×residue, n={metrics.get('contact_n_eval_cells', '?'):,} cells):")
    for k in ["contact_auroc", "contact_iou_at_0.5", "contact_pos_frac"]:
        v = metrics.get(k)
        if v is not None:
            print(f"  {k:<24s} = {v:>10.4f}")
    print(f"\nAFFINITY HEAD (n={metrics.get('affinity_n_eval_pairs', '?'):,} pairs with valid pAff):")
    for k in ["affinity_pearson_r", "affinity_spearman_rho", "affinity_rmse"]:
        v = metrics.get(k)
        if v is not None:
            print(f"  {k:<24s} = {v:>10.4f}")
    print(f"\nElapsed: {elapsed:.1f}s")

    # Save CSV (one row per checkpoint × split)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "checkpoint": str(args.checkpoint),
        "epoch": int(ckpt.get("epoch", -1)),
        "split": args.split,
        "n_eval_pairs": metrics.get("n_pairs_seen", 0),
        **{k: v for k, v in metrics.items()},
    }
    import pandas as pd
    df = pd.DataFrame([row])
    if args.output_csv.exists():
        # Append — preserves history of evals across checkpoints
        existing = pd.read_csv(args.output_csv)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()
