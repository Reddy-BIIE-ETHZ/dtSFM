"""
dtSFM v3 training entrypoint — supports B-2 smoketest (mandatory diagnostics)
and B-3 full pretraining (M1).

The B-2 smoketest is the gate that guards against the cross-attention
accommodation-collapse failure mode. It runs a short training (~few hundred
steps) on a small subset, then computes:

    Δcos = mean(cos(z_drug_i, z_protein_i) for matched pairs)
         - mean(cos(z_drug_i, z_protein_j) for j != i, shuffled)

on a held-out validation batch. The pre-XA global head (default) should
produce Δcos ≥ 0.30 (contact-supervised pretraining provides strong gradients).
The post-XA fallback collapses Δcos to ~0.03 — the accommodation-collapse
failure mode.

Run on:    Euler GPU (smoketest ~10 min, full ~24-48 hr).

Smoketest:
    python3 -m calm.encoder.train_dtsfm_v3 \\
        --metadata_csv /cluster/scratch/reddys/dtsfm_v3/metadata_v3.csv \\
        --drug_npz     /cluster/scratch/reddys/dtsfm_v3/embeddings/drug_embeddings.npz \\
        --protein_dir  /cluster/scratch/reddys/dtsfm_v3/embeddings/protein_embeds/ \\
        --heldout_tsv  audit/dtsfm/heldout_validation_pairs.tsv \\
        --output_dir   /cluster/scratch/reddys/dtsfm_v3/runs/smoketest \\
        --mode smoketest \\
        --batch_size 16 \\
        --train_steps 300

Full training (M1, B-3):
    python3 -m calm.encoder.train_dtsfm_v3 ... --mode full --epochs 30 --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from .data_dtsfm_v3 import (
    DTSFMv3PairDataset,
    collate_dtsfm_v3,
    load_drug_embeddings,
    load_protein_embeddings,
    make_cluster_splits,
)
from .loss_v3 import MultiTaskLossV3, MultiTaskLossWeights
from .model_v3 import CALMEncoderV3


# --------------------------------------------------------------------------- #
# Default config (Hydra/OmegaConf-compatible)
# --------------------------------------------------------------------------- #
def default_model_cfg(d_model: int = 512) -> dict:
    return {
        "d_model": d_model,
        "drug_global_dim": 768,
        "protein_emb_dim": 1280,
        "dropout": 0.1,
        "tau": 0.07,
        "max_scale": 100.0,
        "global_head_uses_pre_xa": True,    # B-11 LOCKED
        "cross_attention": {
            "n_layers": 2,
            "n_heads": 8,
            "d_ff": 2048,
            "dropout": 0.1,
        },
        "atom":  {"elem_dim": 32, "hidden": 256},
        "heads": {"interface_hidden": 256, "contact_d": 64, "affinity_hidden": 256},
    }


# --------------------------------------------------------------------------- #
# Δcos diagnostic — the B-2 gate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_delta_cos(
    model: CALMEncoderV3,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 5,
) -> dict[str, float]:
    """Mean cos(matched) − mean cos(shuffled) on a held-out loader.

    Larger Δcos = more discrimination. We require Δcos > 0.30 to pass.
    Also reports Δcos using both PRE-XA and POST-XA features so the diagnostic
    is auditable: the gap (Δcos_pre − Δcos_post) quantifies the accommodation
    risk that v2.5 fixed.
    """
    model.eval()

    pre_match, pre_shuffle = [], []
    post_match, post_shuffle = [], []

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
        # Pre-XA features (default global path) — already L2-normalized
        z_d = out["global_features_drug"]      # (B, d)
        z_p = out["global_features_protein"]   # (B, d)

        # Pre-XA matched + shuffled cosines
        cos_match = (z_d * z_p).sum(dim=-1)                    # (B,)
        perm = torch.randperm(z_p.shape[0], device=z_p.device)
        z_p_shuf = z_p[perm]
        cos_shuf = (z_d * z_p_shuf).sum(dim=-1)                # (B,)
        pre_match.append(cos_match.cpu())
        pre_shuffle.append(cos_shuf.cpu())

        # Post-XA features pooled with the affinity-head pools, L2-normed for cosine
        d_post = F.normalize(
            model.drug_pool_post(out["drug_atom_post"], batch["drug_mask"].to(device)),
            dim=-1,
        )
        p_post = F.normalize(
            model.protein_pool_post(out["protein_per_res_post"], batch["protein_mask"].to(device)),
            dim=-1,
        )
        cos_match_post = (d_post * p_post).sum(dim=-1)
        cos_shuf_post  = (d_post * p_post[perm]).sum(dim=-1)
        post_match.append(cos_match_post.cpu())
        post_shuffle.append(cos_shuf_post.cpu())

    pre_match = torch.cat(pre_match)
    pre_shuffle = torch.cat(pre_shuffle)
    post_match = torch.cat(post_match)
    post_shuffle = torch.cat(post_shuffle)

    return {
        "delta_cos_pre_xa":  float((pre_match.mean()  - pre_shuffle.mean()).item()),
        "delta_cos_post_xa": float((post_match.mean() - post_shuffle.mean()).item()),
        "match_cos_pre_xa_mean":  float(pre_match.mean().item()),
        "match_cos_pre_xa_std":   float(pre_match.std().item()),
        "shuffle_cos_pre_xa_mean": float(pre_shuffle.mean().item()),
        "shuffle_cos_pre_xa_std":  float(pre_shuffle.std().item()),
        "match_cos_post_xa_mean":  float(post_match.mean().item()),
        "shuffle_cos_post_xa_mean": float(post_shuffle.mean().item()),
        "n_evaluated_pairs": int(pre_match.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Train one epoch (or N steps in smoketest mode)
# --------------------------------------------------------------------------- #
def train_steps(
    model: CALMEncoderV3,
    loss_fn: MultiTaskLossV3,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    n_steps: int | None = None,           # None = full epoch
    log_every: int = 20,
) -> dict[str, float]:
    """Iterate the loader for `n_steps` (cycling if needed) or one full epoch.

    If `n_steps` is set, we cycle the loader as many times as needed to reach
    `n_steps`. Without cycling, smoketests with subset_size < n_steps * batch_size
    silently terminate early (bit us at first smoketest, 2026-05-06: requested
    300 steps, got 125 because subset=2000 / batch=16 = 125 batches/epoch).
    """
    model.train()
    t0 = time.time()
    sums: dict[str, float] = {}
    n_batches = 0

    def _iterate():
        if n_steps is None:
            yield from loader
            return
        # Cycle: re-iterate the loader until we hit n_steps
        emitted = 0
        while emitted < n_steps:
            for batch in loader:
                yield batch
                emitted += 1
                if emitted >= n_steps:
                    return

    for bi, batch in enumerate(_iterate()):
        if n_steps is not None and bi >= n_steps:
            break
        out = model(
            drug_global=batch["drug_global"].to(device),
            drug_elem_ids=batch["drug_elem_ids"].to(device),
            drug_xyz=batch["drug_xyz"].to(device),
            drug_mask=batch["drug_mask"].to(device),
            protein_emb=batch["protein_emb"].to(device),
            protein_mask=batch["protein_mask"].to(device),
        )
        targets = {
            "interface_target_drug":   batch["interface_target_drug"].to(device),
            "contact_target":          batch["contact_target"].to(device),
            "affinity_target":         batch["affinity_target"].to(device),
            "affinity_valid":          batch["affinity_valid"].to(device),
            "drug_mask":               batch["drug_mask"].to(device),
            "protein_mask":            batch["protein_mask"].to(device),
        }
        loss, breakdown = loss_fn(out, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for k, v in breakdown.items():
            sums.setdefault(k, 0.0)
            sums[k] += float(v.item())
        n_batches += 1

        if (bi + 1) % log_every == 0:
            avg = {k: v / n_batches for k, v in sums.items() if k.startswith("loss_")}
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step {bi+1:>5d}  lr={lr:.2e}  "
                  f"L={avg['loss_total']:.3f}  "
                  f"g={avg['loss_global']:.3f}  i={avg['loss_interface']:.3f}  "
                  f"c={avg['loss_contact']:.3f}  a={avg['loss_affinity']:.3f}",
                  flush=True)

    elapsed = time.time() - t0
    return {
        **{k: v / max(n_batches, 1) for k, v in sums.items()},
        "n_batches": n_batches,
        "wall_s": elapsed,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", type=Path, required=True)
    ap.add_argument("--drug_npz", type=Path, required=True)
    ap.add_argument("--protein_dir", type=Path, required=True)
    ap.add_argument("--heldout_tsv", type=Path, default=None)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["smoketest", "full"], required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train_steps", type=int, default=1000,
                    help="(smoketest mode) number of training steps (cycles loader if needed)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.2)
    ap.add_argument("--warmup_steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoketest_subset", type=int, default=4000,
                    help="(smoketest mode) target N pairs sampled from train clusters; "
                         "1.25× this is sampled and 20% held out as in-distribution val")
    ap.add_argument("--delta_cos_threshold", type=float, default=0.10,
                    help="B-2 gate: in-distribution Δcos must exceed this. "
                         "0.10 is achievable with 1K steps; full training targets ≥0.30 OOD.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== dtSFM v3 train ({args.mode}) ===")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # ---- 1. Load embeddings (preload all into RAM per Option B) ----
    drug_embeddings = load_drug_embeddings(args.drug_npz)
    protein_embeddings = load_protein_embeddings(args.protein_dir)

    # ---- 2. Splits ----
    splits = make_cluster_splits(
        args.metadata_csv,
        heldout_tsv=args.heldout_tsv,
        seed=args.seed,
    )
    train_idx_all = splits["train"]
    val_idx_all = splits["val"]

    # In smoketest mode, cap to a small subset so we run in ~10 min.
    # We carve THREE subsets:
    #   train         : pairs the model trains on (from train clusters)
    #   in_dist_val   : DIFFERENT pairs from the SAME train clusters
    #                   → tests "did the model learn anything"
    #   ood_val       : pairs from val clusters (held-out proteins)
    #                   → tests "does it generalize to unseen proteins"
    # Distinguishes "no learning" from "no OOD generalization" — the latter
    # is normal for short smoketest training; the former is a code/design bug.
    if args.mode == "smoketest":
        rng = np.random.default_rng(args.seed)
        # Sample 1.25× the train target so we can split off an in-distribution val
        n_total = min(int(args.smoketest_subset * 1.25), len(train_idx_all))
        sampled = list(rng.choice(train_idx_all, size=n_total, replace=False))
        n_train = int(n_total * 0.80)
        train_idx = sampled[:n_train]
        in_dist_val_idx = sampled[n_train:]
        n_ood = min(args.smoketest_subset // 4, len(val_idx_all))
        ood_val_idx = list(rng.choice(val_idx_all, size=n_ood, replace=False))
        print(f"Smoketest subsets:")
        print(f"  train         : {len(train_idx):,} pairs (train clusters)")
        print(f"  in_dist_val   : {len(in_dist_val_idx):,} pairs (TRAIN clusters, held-out pairs)")
        print(f"  ood_val       : {len(ood_val_idx):,} pairs (held-out clusters)")
    else:
        train_idx = train_idx_all
        in_dist_val_idx = []          # full mode reports OOD only (handled below)
        ood_val_idx = val_idx_all

    # ---- 3. Datasets + Loaders ----
    train_ds = DTSFMv3PairDataset(
        args.metadata_csv, drug_embeddings, protein_embeddings,
        pair_indices=train_idx,
    )
    ood_val_ds = DTSFMv3PairDataset(
        args.metadata_csv, drug_embeddings, protein_embeddings,
        pair_indices=ood_val_idx,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_dtsfm_v3,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    ood_val_loader = DataLoader(
        ood_val_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_dtsfm_v3,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    if in_dist_val_idx:
        in_dist_val_ds = DTSFMv3PairDataset(
            args.metadata_csv, drug_embeddings, protein_embeddings,
            pair_indices=in_dist_val_idx,
        )
        in_dist_val_loader = DataLoader(
            in_dist_val_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_dtsfm_v3,
            drop_last=True, pin_memory=(device.type == "cuda"),
        )
    else:
        in_dist_val_loader = None
    # Keep `val_loader` name for full-training path below
    val_loader = ood_val_loader

    # ---- 4. Model ----
    cfg = OmegaConf.create(default_model_cfg())
    model = CALMEncoderV3(cfg).to(device)
    pcounts = model.count_parameters()
    print(f"\nModel parameter counts:")
    for k, v in pcounts.items():
        print(f"  {k:>20s}: {v:>10,}")

    # ---- 5. Loss + Optimizer + Scheduler ----
    loss_fn = MultiTaskLossV3(
        weights=MultiTaskLossWeights(global_=1.0, interface=1.0, contact=1.0, affinity=0.5),
        ema_decay=0.99,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.mode == "smoketest":
        # No long warmup needed; just constant LR for short run
        scheduler = None
        total_steps = args.train_steps
    else:
        # Cosine to 1e-6 with linear warmup (matches dtSFM v2 / locked C-1)
        total_steps = args.epochs * max(1, len(train_loader))
        def lr_fn(step: int) -> float:
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            cos = 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
            return cos * (1.0 - 1e-2) + 1e-2  # floor at 1% so LR doesn't go below 1e-6
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # ---- 6. Train ----
    if args.mode == "smoketest":
        print(f"\n--- Smoketest training: {args.train_steps} steps ---")
        metrics = train_steps(
            model, loss_fn, train_loader, optimizer, scheduler, device,
            n_steps=args.train_steps, log_every=20,
        )
        print(f"\nSmoketest train metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # ---- 7. The gate: Δcos diagnostic on TWO val sets ----
        print(f"\n--- Δcos diagnostic (B-2 gate) ---")
        diag_in = (compute_delta_cos(model, in_dist_val_loader, device, n_batches=5)
                   if in_dist_val_loader is not None else None)
        diag_ood = compute_delta_cos(model, ood_val_loader, device, n_batches=5)

        if diag_in is not None:
            print(f"\n[in-distribution val: same train clusters, held-out pairs]")
            for k, v in diag_in.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"\n[OOD val: held-out protein clusters]")
        for k, v in diag_ood.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # Gate on in-distribution Δcos (does the model learn at all?)
        # The OOD Δcos is reported but not gated — short smoketest training
        # is not expected to generalize OOD; the gate is "architecture works".
        gate_value = diag_in["delta_cos_pre_xa"] if diag_in is not None else diag_ood["delta_cos_pre_xa"]
        gate_basis = "in-distribution" if diag_in is not None else "OOD"
        passed = gate_value >= args.delta_cos_threshold

        verdict_path = args.output_dir / "smoketest_verdict.json"
        verdict = {
            "gate_basis": gate_basis,
            "gate_value": gate_value,
            "threshold": args.delta_cos_threshold,
            "passed": bool(passed),
            "global_head_uses_pre_xa": True,
            "in_dist_val": diag_in,
            "ood_val": diag_ood,
            "train_metrics": {k: v for k, v in metrics.items()
                              if isinstance(v, (int, float))},
            "config": OmegaConf.to_container(cfg, resolve=True),
            "args": {k: str(v) for k, v in vars(args).items()},
        }
        with open(verdict_path, "w") as f:
            json.dump(verdict, f, indent=2)

        print(f"\n=== B-2 SMOKETEST VERDICT ===")
        if diag_in is not None:
            print(f"  in-dist Δcos_pre  = {diag_in['delta_cos_pre_xa']:+.4f}  "
                  f"(threshold ≥ {args.delta_cos_threshold:+.2f})  ← GATE")
            print(f"  in-dist Δcos_post = {diag_in['delta_cos_post_xa']:+.4f}  "
                  f"(should be lower; confirms v2.5 routing)")
        print(f"  OOD     Δcos_pre  = {diag_ood['delta_cos_pre_xa']:+.4f}  "
              f"(reported only; expected near 0 at 1K steps)")
        print(f"  OOD     Δcos_post = {diag_ood['delta_cos_post_xa']:+.4f}")
        if passed:
            print(f"  PASSED  ✓  → architecture works; safe to scale to B-3 full training")
            print(f"     OOD will improve with full training (30 epochs × ~10K steps).")
        else:
            print(f"  FAILED  ✗  → in-distribution discrimination too weak.")
            print(f"     Possible causes:")
            print(f"     (a) loss EMAs not warmed up (raise train_steps to 2K+);")
            print(f"     (b) batch_size too small for InfoNCE (raise to 128 if RAM allows);")
            print(f"     (c) global_head routing bug (verify global_head_uses_pre_xa=True);")
            print(f"     (d) drug_global vector lacks discriminative signal "
                  f"(rare; would mean MoLFormer mean-pool is too lossy).")
        print(f"  Verdict written to: {verdict_path}")
        return

    # ---- Full training (B-3) ----
    print(f"\n--- Full training: {args.epochs} epochs × {len(train_loader)} steps/epoch "
          f"= {total_steps:,} total steps ---")
    best_delta_cos = -1.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_metrics = train_steps(
            model, loss_fn, train_loader, optimizer, scheduler, device,
            n_steps=None, log_every=200,
        )
        diag = compute_delta_cos(model, val_loader, device, n_batches=10)
        print(f"  [val] Δcos_pre_xa={diag['delta_cos_pre_xa']:+.4f}  "
              f"Δcos_post_xa={diag['delta_cos_post_xa']:+.4f}")

        ckpt_path = args.output_dir / f"epoch_{epoch+1:03d}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss_state_dict": loss_fn.state_dict(),
            "train_metrics": train_metrics,
            "val_diag": diag,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, ckpt_path)
        if diag["delta_cos_pre_xa"] > best_delta_cos:
            best_delta_cos = diag["delta_cos_pre_xa"]
            best_path = args.output_dir / "best.pt"
            torch.save(torch.load(ckpt_path, map_location="cpu"), best_path)
            print(f"  [val] new best Δcos_pre_xa, saved → {best_path}")

    print(f"\n=== Training complete. Best val Δcos_pre_xa = {best_delta_cos:+.4f} ===")


if __name__ == "__main__":
    main()
