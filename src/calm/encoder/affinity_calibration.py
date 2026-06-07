"""
Affinity calibration for dtSFM: map cosine similarity to binding free energy.

From MRC §7: the contrastive similarity score is the Boltzmann energy up to
a two-parameter linear calibration:

    ΔG = α · cos_sim + β

where α encodes temperature and unit conversion, β is the thermodynamic
reference state.

This script:
1. Loads a trained dtSFM checkpoint (projection heads + logit_scale)
2. Loads drug and protein embeddings
3. Projects them through the trained projection heads
4. Computes cosine similarity for all DAVIS pairs with known Kd
5. Converts Kd to ΔG: ΔG = RT·ln(Kd) where R=1.987 cal/(mol·K), T=298.15 K
6. Fits α and β by linear regression on train set
7. Evaluates Pearson/Spearman correlation on held-out test set

Usage:
    python -m calm.encoder.affinity_calibration \
        --checkpoint /path/to/best_model.pth \
        --data_dir /path/to/dtsfm/data \
        --davis_csv /path/to/davis_raw.csv \
        --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Build the projection heads (same architecture as training)
# ---------------------------------------------------------------------------


def build_projection_heads(
    dim_ag: int = 768,
    dim_ab: int = 1280,
    d_ff: int = 2048,
    d_model: int = 512,
) -> tuple[nn.Sequential, nn.Sequential]:
    """Build FFN projection heads matching dtSFM training config."""
    encoder_ag = nn.Sequential(
        nn.Linear(dim_ag, d_ff, bias=False),
        nn.ReLU(),
        nn.Linear(d_ff, d_model, bias=False),
    )
    encoder_ab = nn.Sequential(
        nn.Linear(dim_ab, d_ff, bias=False),
        nn.ReLU(),
        nn.Linear(d_ff, d_model, bias=False),
    )
    return encoder_ag, encoder_ab


# ---------------------------------------------------------------------------
# 2. Load checkpoint and compute cosine similarities
# ---------------------------------------------------------------------------


def compute_cosine_similarities(
    checkpoint_path: str,
    ag_embed_path: str,
    ab_embed_path: str,
    ag_mask_path: str,
    ab_mask_path: str,
    ag_indices_path: str,
    ab_indices_path: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load trained model and compute cosine similarities for all pairs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ag_projected: (N_unique_drugs, d_model) L2-normalized
        ab_projected: (N_unique_targets, d_model) L2-normalized
    """
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build projection heads
    encoder_ag, encoder_ab = build_projection_heads()
    encoder_ag.load_state_dict({
        k.replace("encoder_ag.", ""): v for k, v in ckpt.items() if k.startswith("encoder_ag.")
    })
    encoder_ab.load_state_dict({
        k.replace("encoder_ab.", ""): v for k, v in ckpt.items() if k.startswith("encoder_ab.")
    })
    encoder_ag = encoder_ag.to(device).eval()
    encoder_ab = encoder_ab.to(device).eval()

    logit_scale = ckpt["logit_scale"].item()
    temperature = 1.0 / math.exp(logit_scale)
    print(f"  Learned logit_scale: {logit_scale:.4f}")
    print(f"  Learned temperature τ: {temperature:.6f}")

    # Load embeddings
    ag_embed = torch.load(ag_embed_path, map_location=device, weights_only=True)
    ab_embed = torch.load(ab_embed_path, map_location=device, weights_only=True)
    ag_mask = torch.load(ag_mask_path, map_location=device, weights_only=True)
    ab_mask = torch.load(ab_mask_path, map_location=device, weights_only=True)

    print(f"  Drug embeddings: {ag_embed.shape}")
    print(f"  Protein embeddings: {ab_embed.shape}")

    # Mean pooling over sequence length (masked)
    with torch.no_grad():
        # Drug embeddings: mean pool then project
        ag_mask_f = ag_mask.unsqueeze(-1).float()  # (N, L, 1)
        ag_pooled = (ag_embed * ag_mask_f).sum(dim=1) / ag_mask_f.sum(dim=1).clamp(min=1)
        ag_proj = encoder_ag(ag_pooled)
        ag_proj = F.normalize(ag_proj, dim=-1)

        # Protein embeddings: mean pool then project
        ab_mask_f = ab_mask.unsqueeze(-1).float()
        ab_pooled = (ab_embed * ab_mask_f).sum(dim=1) / ab_mask_f.sum(dim=1).clamp(min=1)
        ab_proj = encoder_ab(ab_pooled)
        ab_proj = F.normalize(ab_proj, dim=-1)

    print(f"  Projected drugs: {ag_proj.shape}")
    print(f"  Projected proteins: {ab_proj.shape}")

    return ag_proj.cpu(), ab_proj.cpu(), temperature


# ---------------------------------------------------------------------------
# 3. Affinity calibration
# ---------------------------------------------------------------------------


def kd_to_dg(kd_nm: float, T: float = 298.15) -> float:
    """Convert Kd (in nM) to ΔG (in kcal/mol).

    ΔG = RT·ln(Kd) where Kd is in molar units.
    R = 1.987 cal/(mol·K) = 0.001987 kcal/(mol·K)
    """
    R = 0.001987  # kcal/(mol·K)
    kd_molar = kd_nm * 1e-9  # convert nM to M
    return R * T * math.log(kd_molar)


def run_calibration(
    checkpoint_path: str,
    data_dir: str,
    davis_csv: str,
    output_dir: str,
    device: str = "cpu",
    test_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    """Run the full affinity calibration pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(data_dir)

    print("=" * 60)
    print("dtSFM Affinity Calibration (MRC §7)")
    print("=" * 60)

    # --- Step 1: Compute cosine similarities ---
    print("\nStep 1: Computing cosine similarities...")
    ag_proj, ab_proj, temperature = compute_cosine_similarities(
        checkpoint_path=checkpoint_path,
        ag_embed_path=str(data / "ag_embed.pt"),
        ab_embed_path=str(data / "ab_embed.pt"),
        ag_mask_path=str(data / "ag_mask.pt"),
        ab_mask_path=str(data / "ab_mask.pt"),
        ag_indices_path=str(data / "ag_indices.pt"),
        ab_indices_path=str(data / "ab_indices.pt"),
        device=device,
    )

    # --- Step 2: Load DAVIS data with Kd values ---
    print("\nStep 2: Loading DAVIS binding affinity data...")
    davis = pd.read_csv(davis_csv)

    # Load metadata to map SMILES/sequences to embedding indices
    metadata = pd.read_csv(data / "metadata.csv")
    unique_drugs = metadata["drug_smiles"].drop_duplicates().tolist()
    unique_targets = metadata["target_seq"].drop_duplicates().tolist()
    drug_to_idx = {s: i for i, s in enumerate(unique_drugs)}
    target_to_idx = {s: i for i, s in enumerate(unique_targets)}

    # Match DAVIS pairs to our embedding indices
    records = []
    for _, row in davis.iterrows():
        smiles = row["Drug"]
        seq = row["Target"]
        kd = row["Y"]  # Kd in nM

        if smiles in drug_to_idx and seq in target_to_idx:
            drug_idx = drug_to_idx[smiles]
            target_idx = target_to_idx[seq]

            # Compute cosine similarity for this pair
            cos_sim = (ag_proj[drug_idx] @ ab_proj[target_idx]).item()

            # Convert Kd to ΔG (skip non-binders with Kd = 10000)
            if kd < 10000:
                dg = kd_to_dg(kd)
            else:
                dg = kd_to_dg(10000)  # use ceiling value

            records.append({
                "drug_id": row["Drug_ID"],
                "target_id": row["Target_ID"],
                "kd_nm": kd,
                "dg_kcal_mol": dg,
                "cos_sim": cos_sim,
                "is_binder": kd < 10000,
            })

    df = pd.DataFrame(records)
    print(f"  Matched {len(df)} DAVIS pairs to embeddings")
    print(f"  Binders: {df.is_binder.sum()}, Non-binders: {(~df.is_binder).sum()}")

    # --- Step 3: Split train/test ---
    print("\nStep 3: Splitting train/test for calibration...")
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    n_test = int(len(df) * test_fraction)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]
    print(f"  Train: {len(df_train)}, Test: {len(df_test)}")

    # --- Step 4: Fit linear calibration ΔG = α·cos_sim + β ---
    print("\nStep 4: Fitting affinity calibration (ΔG = α·cos_sim + β)...")

    # Fit on binders only (non-binders have a ceiling Kd, not a true measurement)
    train_binders = df_train[df_train.is_binder]
    test_binders = df_test[df_test.is_binder]
    print(f"  Training on {len(train_binders)} binder pairs")
    print(f"  Testing on {len(test_binders)} binder pairs")

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        train_binders["cos_sim"].values,
        train_binders["dg_kcal_mol"].values,
    )
    alpha = slope
    beta = intercept

    print(f"\n  Calibration parameters:")
    print(f"    α (slope):     {alpha:.4f} kcal/mol per unit cos_sim")
    print(f"    β (intercept): {beta:.4f} kcal/mol")
    print(f"    α sign check:  {'CORRECT (negative)' if alpha < 0 else 'UNEXPECTED (positive)'}")
    print(f"    Train R²:      {r_value**2:.4f}")

    # --- Step 5: Evaluate on test set ---
    print("\nStep 5: Evaluating on test set...")

    # Predict ΔG for test binders
    test_binders = test_binders.copy()
    test_binders["dg_predicted"] = alpha * test_binders["cos_sim"] + beta

    # Correlations
    pearson_r, pearson_p = stats.pearsonr(
        test_binders["dg_kcal_mol"], test_binders["dg_predicted"]
    )
    spearman_r, spearman_p = stats.spearmanr(
        test_binders["dg_kcal_mol"], test_binders["dg_predicted"]
    )

    # RMSE
    rmse = np.sqrt(np.mean(
        (test_binders["dg_kcal_mol"] - test_binders["dg_predicted"]) ** 2
    ))

    print(f"  Test set results ({len(test_binders)} binder pairs):")
    print(f"    Pearson r:   {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"    Spearman ρ:  {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"    RMSE:        {rmse:.4f} kcal/mol")

    # --- Step 6: Binder vs non-binder discrimination ---
    print("\nStep 6: Binder vs non-binder discrimination (full test set)...")
    df_test = df_test.copy()
    df_test["dg_predicted"] = alpha * df_test["cos_sim"] + beta

    # AUROC: can cosine similarity discriminate binders from non-binders?
    from sklearn.metrics import roc_auc_score
    auroc_cos = roc_auc_score(df_test["is_binder"].astype(int), df_test["cos_sim"])
    auroc_dg = roc_auc_score(
        df_test["is_binder"].astype(int), -df_test["dg_predicted"]
    )
    print(f"  AUROC (cos_sim, raw):    {auroc_cos:.4f}")
    print(f"  AUROC (-ΔG, calibrated): {auroc_dg:.4f}")

    # --- Step 7: Save results ---
    results = {
        "calibration": {
            "alpha": float(alpha),
            "beta": float(beta),
            "alpha_sign": "negative (correct)" if alpha < 0 else "positive (unexpected)",
            "learned_temperature": float(temperature),
            "train_R_squared": float(r_value ** 2),
        },
        "test_binders": {
            "n_pairs": int(len(test_binders)),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_r),
            "spearman_p": float(spearman_p),
            "rmse_kcal_mol": float(rmse),
        },
        "discrimination": {
            "n_test_total": int(len(df_test)),
            "n_binders": int(df_test.is_binder.sum()),
            "n_nonbinders": int((~df_test.is_binder).sum()),
            "auroc_cosine_sim": float(auroc_cos),
            "auroc_calibrated_dg": float(auroc_dg),
        },
        "cosine_sim_stats": {
            "binders_mean": float(df[df.is_binder]["cos_sim"].mean()),
            "binders_std": float(df[df.is_binder]["cos_sim"].std()),
            "nonbinders_mean": float(df[~df.is_binder]["cos_sim"].mean()),
            "nonbinders_std": float(df[~df.is_binder]["cos_sim"].std()),
        },
    }

    results_path = out / "affinity_calibration_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Save per-pair predictions
    df.to_csv(out / "davis_cosine_similarities.csv", index=False)
    df_test.to_csv(out / "davis_test_predictions.csv", index=False)
    print(f"  Per-pair data saved to {out}/davis_*.csv")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Calibration: ΔG = {alpha:.4f} · cos_sim + ({beta:.4f})")
    print(f"  Pearson r = {pearson_r:.4f}, Spearman ρ = {spearman_r:.4f}")
    print(f"  RMSE = {rmse:.2f} kcal/mol")
    print(f"  AUROC (binder discrimination) = {auroc_cos:.4f}")
    print(f"  Learned τ = {temperature:.6f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Affinity calibration for dtSFM (MRC §7)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to best model checkpoint (.pth)",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory with embedding tensors (ag_embed.pt, etc.)",
    )
    parser.add_argument(
        "--davis_csv", type=str, required=True,
        help="Path to davis_raw.csv with Kd measurements",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save calibration results",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (cpu or cuda)",
    )
    args = parser.parse_args()

    run_calibration(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        davis_csv=args.davis_csv,
        output_dir=args.output_dir,
        device=args.device,
    )
