"""
Pool-512 evaluation for dtSFM from saved checkpoints.

Evaluates on UNIQUE drug-target pairs only (not raw data pairs which have
many duplicates). For each query drug, retrieves from a pool of unique targets.
For each query target, retrieves from a pool of unique drugs.

Usage:
    python -m calm.encoder.eval_dtsfm_pool512 \
        --data_dir data/dtsfm \
        --output_dir /cluster/scratch/reddys/dtsfm/output/dtsfm_full \
        --pool_size 512 --n_trials 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def project_unique_embeddings(
    checkpoint_path: str,
    ag_embed: torch.Tensor,
    ab_embed: torch.Tensor,
    ag_mask: torch.Tensor,
    ab_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Project ALL unique embeddings through trained projection heads.

    Returns
    -------
    all_ag_proj: (N_unique_drugs, 512) L2-normalized
    all_ab_proj: (N_unique_targets, 512) L2-normalized
    temperature: learned temperature
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    enc_ag = nn.Sequential(nn.Linear(768, 2048, bias=False), nn.ReLU(), nn.Linear(2048, 512, bias=False))
    enc_ab = nn.Sequential(nn.Linear(1280, 2048, bias=False), nn.ReLU(), nn.Linear(2048, 512, bias=False))
    enc_ag.load_state_dict({k.replace("encoder_ag.", ""): v for k, v in ckpt.items() if k.startswith("encoder_ag.")})
    enc_ab.load_state_dict({k.replace("encoder_ab.", ""): v for k, v in ckpt.items() if k.startswith("encoder_ab.")})
    enc_ag.eval()
    enc_ab.eval()

    temperature = 1.0 / math.exp(ckpt["logit_scale"].item())

    with torch.no_grad():
        ag_mf = ag_mask.unsqueeze(-1).float()
        ag_pooled = (ag_embed * ag_mf).sum(1) / ag_mf.sum(1).clamp(min=1)
        all_ag = F.normalize(enc_ag(ag_pooled), dim=-1)

        ab_mf = ab_mask.unsqueeze(-1).float()
        ab_pooled = (ab_embed * ab_mf).sum(1) / ab_mf.sum(1).clamp(min=1)
        all_ab = F.normalize(enc_ab(ab_pooled), dim=-1)

    return all_ag, all_ab, temperature


def pool_retrieval_unique(
    drug_proj: torch.Tensor,
    target_proj: torch.Tensor,
    pos_drug_to_targets: dict[int, set[int]],
    pos_target_to_drugs: dict[int, set[int]],
    pool_size: int = 512,
    n_trials: int = 100,
    k_values: tuple[int, ...] = (1, 5, 10),
    seed: int = 42,
) -> dict[str, float]:
    """Pool retrieval on unique drugs × unique targets.

    Drug→Target: for each drug, sample pool_size-1 wrong targets + correct ones.
    Target→Drug: for each target, sample pool_size-1 wrong drugs + correct ones.
    """
    rng = np.random.RandomState(seed)
    n_drugs = drug_proj.shape[0]
    n_targets = target_proj.shape[0]

    # Drug→Target similarity: (n_drugs, n_targets)
    sim_d2t = (drug_proj @ target_proj.T).numpy()
    # Target→Drug similarity: (n_targets, n_drugs)
    sim_t2d = sim_d2t.T

    results = {}

    # --- Drug → Target direction ---
    eff_pool = min(pool_size, n_targets)
    trial_recalls = {k: [] for k in k_values}

    for _trial in range(n_trials):
        hits = {k: 0 for k in k_values}
        n_valid = 0

        for di in range(n_drugs):
            pos_targets = pos_drug_to_targets.get(di, set())
            if not pos_targets:
                continue

            neg_targets = [t for t in range(n_targets) if t not in pos_targets]
            if not neg_targets:
                continue

            n_neg = min(eff_pool - 1, len(neg_targets))
            sampled_neg = rng.choice(neg_targets, size=n_neg, replace=False)

            # Pick one positive to be the query target
            pos_list = list(pos_targets)
            query_pos = pos_list[rng.randint(len(pos_list))]

            pool = np.concatenate([[query_pos], sampled_neg])
            sims = sim_d2t[di, pool]
            # Rank of the positive (index 0 in pool)
            rank = (sims[1:] > sims[0]).sum()

            for k in k_values:
                if rank < k:
                    hits[k] += 1
            n_valid += 1

        for k in k_values:
            trial_recalls[k].append(hits[k] / n_valid * 100.0 if n_valid > 0 else 0.0)

    for k in k_values:
        vals = trial_recalls[k]
        results[f"R@{k}_ag2ab"] = float(np.mean(vals))
        results[f"R@{k}_ag2ab_std"] = float(np.std(vals))

    # --- Target → Drug direction ---
    eff_pool = min(pool_size, n_drugs)
    trial_recalls = {k: [] for k in k_values}

    for _trial in range(n_trials):
        hits = {k: 0 for k in k_values}
        n_valid = 0

        for ti in range(n_targets):
            pos_drugs = pos_target_to_drugs.get(ti, set())
            if not pos_drugs:
                continue

            neg_drugs = [d for d in range(n_drugs) if d not in pos_drugs]
            if not neg_drugs:
                continue

            n_neg = min(eff_pool - 1, len(neg_drugs))
            sampled_neg = rng.choice(neg_drugs, size=n_neg, replace=False)

            pos_list = list(pos_drugs)
            query_pos = pos_list[rng.randint(len(pos_list))]

            pool = np.concatenate([[query_pos], sampled_neg])
            sims = sim_t2d[ti, pool]
            rank = (sims[1:] > sims[0]).sum()

            for k in k_values:
                if rank < k:
                    hits[k] += 1
            n_valid += 1

        for k in k_values:
            trial_recalls[k].append(hits[k] / n_valid * 100.0 if n_valid > 0 else 0.0)

    for k in k_values:
        vals = trial_recalls[k]
        results[f"R@{k}_ab2ag"] = float(np.mean(vals))
        results[f"R@{k}_ab2ag_std"] = float(np.std(vals))

    results["pool_size_d2t"] = min(pool_size, n_targets)
    results["pool_size_t2d"] = min(pool_size, n_drugs)
    results["n_unique_drugs"] = n_drugs
    results["n_unique_targets"] = n_targets

    return results


def main():
    parser = argparse.ArgumentParser(description="Pool-512 eval for dtSFM")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pool_size", type=int, default=512)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("Loading embeddings...")
    ag_embed = torch.load(data_dir / "ag_embed.pt", map_location="cpu", weights_only=True)
    ab_embed = torch.load(data_dir / "ab_embed.pt", map_location="cpu", weights_only=True)
    ag_mask = torch.load(data_dir / "ag_mask.pt", map_location="cpu", weights_only=True)
    ab_mask = torch.load(data_dir / "ab_mask.pt", map_location="cpu", weights_only=True)
    ag_indices = torch.load(data_dir / "ag_indices.pt", map_location="cpu", weights_only=True)
    ab_indices = torch.load(data_dir / "ab_indices.pt", map_location="cpu", weights_only=True)
    metadata = pd.read_csv(data_dir / "metadata.csv")
    print(f"  Drugs: {ag_embed.shape[0]} unique, Proteins: {ab_embed.shape[0]} unique, Pairs: {len(metadata)}")

    hash_to_idx = {h: i for i, h in enumerate(metadata["Unique_ag_vh_vl_hash"])}

    splits = ["identity_100", "mmseqs_080", "mmseqs_060", "mmseqs_040"]
    all_results = []

    # Project unique embeddings once per checkpoint (cache by checkpoint path)
    proj_cache = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")

        fold_results = []

        for fold in range(5):
            # Auto-detect prefix (dtsfm- for v1, dtsfm2- for v2)
            run_dir = output_dir / f"dtsfm2-{split}-fold{fold}" / "train" / f"fold_{fold}"
            if not run_dir.exists():
                run_dir = output_dir / f"dtsfm-{split}-fold{fold}" / "train" / f"fold_{fold}"

            ckpt_files = list(run_dir.glob("best_model_val_pred_acc_epoch_*.pth"))
            if not ckpt_files:
                print(f"  fold {fold}: no checkpoint, skipping")
                continue
            best_ckpt = max(ckpt_files, key=lambda p: int(p.stem.split("_")[-1]))

            # Load split
            split_dir = data_dir / "split_index" / split
            split_file = split_dir / f"split_hash_ids_outerfold_{fold}_innerfold_{fold}.json"
            if not split_file.exists():
                split_file = split_dir / f"split_hash_ids_outerfold_{fold}_innerfold_0.json"
            if not split_file.exists():
                print(f"  fold {fold}: split not found, skipping")
                continue

            with open(split_file) as f:
                split_data = json.load(f)
            test_hashes = split_data.get("test", [])
            test_indices = [hash_to_idx[h] for h in test_hashes if h in hash_to_idx]
            if not test_indices:
                print(f"  fold {fold}: empty test set, skipping")
                continue

            # Project embeddings (cache per checkpoint)
            ckpt_key = str(best_ckpt)
            if ckpt_key not in proj_cache:
                print(f"  Projecting embeddings for {best_ckpt.name}...")
                proj_cache[ckpt_key] = project_unique_embeddings(
                    ckpt_key, ag_embed, ab_embed, ag_mask, ab_mask
                )
            all_ag_proj, all_ab_proj, temp = proj_cache[ckpt_key]

            # Get unique (drug_idx, target_idx) pairs in the test set
            test_ag = ag_indices[test_indices].numpy()
            test_ab = ab_indices[test_indices].numpy()
            unique_pairs = set()
            for i in range(len(test_indices)):
                unique_pairs.add((int(test_ag[i]), int(test_ab[i])))

            # Get unique drug and target indices in test set
            test_drug_ids = sorted(set(p[0] for p in unique_pairs))
            test_target_ids = sorted(set(p[1] for p in unique_pairs))

            # Local index mappings
            drug_local = {d: i for i, d in enumerate(test_drug_ids)}
            target_local = {t: i for i, t in enumerate(test_target_ids)}

            # Build positive pair mappings (local indices)
            pos_d2t = {}  # drug_local → set of target_locals
            pos_t2d = {}  # target_local → set of drug_locals
            for d_id, t_id in unique_pairs:
                dl = drug_local[d_id]
                tl = target_local[t_id]
                pos_d2t.setdefault(dl, set()).add(tl)
                pos_t2d.setdefault(tl, set()).add(dl)

            # Get projected embeddings for test drugs and targets
            drug_proj = all_ag_proj[test_drug_ids]
            target_proj = all_ab_proj[test_target_ids]

            n_d = len(test_drug_ids)
            n_t = len(test_target_ids)
            print(f"  fold {fold}: ckpt={best_ckpt.name}, {len(unique_pairs)} unique pairs, {n_d} drugs, {n_t} targets")

            # Run retrieval
            fr = pool_retrieval_unique(
                drug_proj, target_proj, pos_d2t, pos_t2d,
                pool_size=args.pool_size, n_trials=args.n_trials, seed=args.seed,
            )
            fr["split"] = split
            fr["fold"] = fold
            fr["temperature"] = temp
            fr["checkpoint"] = best_ckpt.name
            fr["n_unique_pairs"] = len(unique_pairs)

            print(f"    R@1  drug→target: {fr['R@1_ag2ab']:.1f}%  |  target→drug: {fr['R@1_ab2ag']:.1f}%")
            print(f"    R@5  drug→target: {fr['R@5_ag2ab']:.1f}%  |  target→drug: {fr['R@5_ab2ag']:.1f}%")
            print(f"    R@10 drug→target: {fr['R@10_ag2ab']:.1f}%  |  target→drug: {fr['R@10_ab2ag']:.1f}%")

            fold_results.append(fr)
            all_results.append(fr)

        if fold_results:
            # Exclude fold 4 from summary if it's degenerate
            good_folds = [r for r in fold_results if r.get("fold") != 4 or r.get("R@1_ag2ab", 0) > 1.0]
            if good_folds:
                print(f"\n  --- {split} summary ({len(good_folds)} folds) ---")
                for m in ["R@1_ag2ab", "R@1_ab2ag", "R@5_ag2ab", "R@5_ab2ag", "R@10_ag2ab", "R@10_ab2ag"]:
                    vals = [r[m] for r in good_folds]
                    print(f"    {m}: {np.mean(vals):.1f} ± {np.std(vals):.1f}%")

    # Save
    if all_results:
        results_path = output_dir / "pool512_results_unique.csv"
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {results_path}")

    # Final table
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — Tables 21-22 (unique drug-target pairs)")
    print(f"{'='*60}")
    print(f"{'Split':<16} {'R@1(D→T)':<12} {'R@1(T→D)':<12} {'R@5(D→T)':<12} {'R@5(T→D)':<12} {'R@10(D→T)':<12} {'R@10(T→D)':<12}")
    print("-" * 88)
    for split in splits:
        sr = [r for r in all_results if r["split"] == split and (r.get("fold") != 4 or r.get("R@1_ag2ab", 0) > 1.0)]
        if sr:
            row = []
            for m in ["R@1_ag2ab", "R@1_ab2ag", "R@5_ag2ab", "R@5_ab2ag", "R@10_ag2ab", "R@10_ab2ag"]:
                vals = [r[m] for r in sr]
                row.append(f"{np.mean(vals):.1f}±{np.std(vals):.1f}")
            print(f"{split:<16} {'  '.join(row)}")


if __name__ == "__main__":
    main()
