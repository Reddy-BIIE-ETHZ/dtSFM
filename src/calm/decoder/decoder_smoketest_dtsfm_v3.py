"""decoder_smoketest_dtsfm_v3.py — Phase 0 sanity check for the dtSFM v3 decoder.

This is the FIRST script run in the decoder workstream. It does NOT train any
decoder. Its only jobs are:

  Phase A:  Confirm the locked v3 encoder (epoch_010.pt) loads on this Euler
            node and produces the expected output shapes for one known
            (drug, protein) pair from training.

  Phase B:  Build a "competence map" for the 10 decoder targets that were
            selected for design — for each target gene, compute the encoder's
            global protein vector, then the global drug vector for each known
            approved binder of that target, then their cosine similarity.
            High mean cosine = encoder recognises that target's chemotype well
            and the decoder can plausibly generate against it.
            Low mean cosine = encoder blind spot — the decoder should not be
            deployed on that target without further validation.

The 10 approved decoder targets (kinases + serine protease only, all expected
to be low-risk for the encoder's training distribution):

    BTK | EGFR | CDK4/6 | JAK family (JAK1/2/3, TYK2) | ALK |
    PARP1/2 | mTOR | PI3Kalpha | FXIa (F11) | FLT3

This script runs in a few minutes on a single A100 / RTX 6000.
Output:
    summary CSV   →  audit/dtsfm/decoder_phase0_competence_map.csv
    per-binder    →  audit/dtsfm/decoder_phase0_per_binder.csv
    forward-pass  →  printed to stdout in =Summary= block
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Import shared helpers from the kinase fine-tune script (already battle-tested)
from calm.encoder.model_v3 import CALMEncoderV3, element_to_idx, N_ELEMENTS
from calm.encoder.train_dtsfm_v3 import default_model_cfg
from calm.encoder.train_klaeger_finetune import (
    atoms_from_smiles_etkdg,
    build_smiles_to_idx,
    load_drug_embeddings,
    load_protein_embeddings,
    build_gene_to_protein_idx,
)


# --------------------------------------------------------------------------- #
# Note: SMILES canonicalization + drug_idx resolution + leakage class are now
# pre-computed offline by data/dtsfm/scripts/build_decoder_target_binders.py
# (which runs on the Euler login node — needs internet for PubChem). This
# script reads the resolved TSV and uses drug_idx + target_pidx_canonical
# directly. No SMILES handling here.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Phase A — encoder forward pass on one Class-A pair
# --------------------------------------------------------------------------- #
def phase_a_forward_pass(
    model: CALMEncoderV3,
    metadata_df: pd.DataFrame,
    drug_embeddings: dict,
    protein_embeddings: dict,
    device: str,
) -> dict:
    """Pick the first metadata row whose drug + protein are both cached, run a
    forward pass, and return shapes + cosine for inspection."""
    print("\n" + "=" * 60)
    print("PHASE A — encoder forward pass on one known training pair")
    print("=" * 60)

    # Find a row whose drug_idx + protein_idx are cached AND whose SMILES
    # successfully embeds with RDKit ETKDG. Some training SMILES have unusual
    # stereochem / charged atoms that ETKDG can't handle — try up to N rows
    # before giving up on Phase A (Phase B does not need ETKDG, so a Phase A
    # failure should not block the rest of the smoketest).
    row = None
    atoms = None
    n_tried = 0
    for _, r in metadata_df.iterrows():
        if int(r["drug_idx"]) not in drug_embeddings:
            continue
        if int(r["protein_idx"]) not in protein_embeddings:
            continue
        n_tried += 1
        smi_try = str(r["drug_smiles"])
        try:
            a = atoms_from_smiles_etkdg(smi_try, max_atoms=256)
        except Exception as e:
            print(f"  [skip row {n_tried}] ETKDG raised on {smi_try[:60]}...: "
                  f"{type(e).__name__}: {e}")
            a = None
        if a is None:
            if n_tried <= 5:
                print(f"  [skip row {n_tried}] ETKDG returned None for "
                      f"{smi_try[:60]}{'...' if len(smi_try) > 60 else ''}")
            if n_tried >= 30:
                raise RuntimeError("Phase A: 30 consecutive metadata rows failed ETKDG — something is wrong")
            continue
        row = r
        atoms = a
        break
    if row is None or atoms is None:
        raise RuntimeError("No metadata row had both drug+protein cached AND ETKDG-embeddable")

    smi = str(row["drug_smiles"])
    drug_idx = int(row["drug_idx"])
    prot_idx = int(row["protein_idx"])
    prot_id = str(row["protein_id"])
    print(f"  Test pair:  drug_idx={drug_idx}, protein_idx={prot_idx}  ({prot_id})  "
          f"(after {n_tried} ETKDG attempts)")
    print(f"  SMILES:     {smi[:80]}{'...' if len(smi) > 80 else ''}")
    # atoms_from_smiles_etkdg returns (elem_ids: int64 (N,), xyz: float32 (N, 3))
    # already converted to integer element indices — no string handling needed.
    elem_ids_1d, xyz_2d = atoms
    n_atoms = elem_ids_1d.shape[0]
    elem_ids = elem_ids_1d.unsqueeze(0).to(device)                 # (1, N) int64
    xyz_t = xyz_2d.unsqueeze(0).to(device)                          # (1, N, 3) float32
    drug_mask = torch.ones(1, n_atoms, dtype=torch.bool, device=device)

    # Drug global = cached MoLFormer (load_drug_embeddings returns torch.Tensor
    # already, not numpy — so use torch.as_tensor for robustness either way)
    drug_global = torch.as_tensor(drug_embeddings[drug_idx]).float().unsqueeze(0).to(device)  # (1, 768)

    # Protein per-residue ESM-2
    prot = protein_embeddings[prot_idx].float().to(device)          # (L, 1280)
    L_res = prot.shape[0]
    prot_emb = prot.unsqueeze(0)                                   # (1, L, 1280)
    prot_mask = torch.ones(1, L_res, dtype=torch.bool, device=device)

    print(f"  Input shapes:")
    print(f"    drug_global   {tuple(drug_global.shape)}")
    print(f"    drug_atoms    elem={tuple(elem_ids.shape)}, xyz={tuple(xyz_t.shape)}, N_atoms={n_atoms}")
    print(f"    protein_emb   {tuple(prot_emb.shape)}, L_res={L_res}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        out = model(drug_global, elem_ids, xyz_t, drug_mask, prot_emb, prot_mask)

    print("  Output shapes:")
    for k in ("global_features_drug", "global_features_protein",
              "interface_logits_drug", "contact_logits", "affinity_pred"):
        print(f"    {k:<28} {tuple(out[k].shape)}")
    cos = (out["global_features_drug"] * out["global_features_protein"]).sum().item()
    aff = out["affinity_pred"].item()
    print(f"  Cosine(drug, target) = {cos:+.4f}    Predicted pAffinity = {aff:+.3f}")
    if cos < 0.3:
        print(f"  WARNING: Cosine is unexpectedly low for a Class-A training pair (expected > 0.5).")
    else:
        print(f"  PASS: Cosine is in the expected range for a Class-A training pair.")

    return {
        "drug_idx": drug_idx, "protein_idx": prot_idx, "protein_id": prot_id,
        "n_atoms": n_atoms, "L_res": L_res,
        "cosine": cos, "affinity_pred": aff,
    }


# --------------------------------------------------------------------------- #
# Phase B — competence map across the 10 approved decoder targets
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_target_global(
    model: CALMEncoderV3, prot_emb: torch.Tensor, device: str,
) -> torch.Tensor:
    """Compute the encoder's L2-normalized 512-d protein global vector for one
    target. This uses the PRE-XA path — no drug needed, no cross-attention.
    """
    prot_emb = prot_emb.float().unsqueeze(0).to(device)            # (1, L, 1280)
    L = prot_emb.shape[1]
    mask = torch.ones(1, L, dtype=torch.bool, device=device)
    h = model.protein_proj(prot_emb)                                # (1, L, d)
    h = h * mask.unsqueeze(-1).to(h.dtype)
    pooled = model.protein_pool_pre(h, mask)                        # (1, d)
    return F.normalize(pooled, dim=-1).squeeze(0)                   # (d,)


@torch.no_grad()
def encode_drug_global(
    model: CALMEncoderV3, drug_global_768, device: str,
) -> torch.Tensor:
    """Compute the encoder's L2-normalized 512-d drug global vector for one
    cached MoLFormer fingerprint. Pre-XA path — no atoms, no cross-attention.
    Accepts either np.ndarray or torch.Tensor input.
    """
    x = torch.as_tensor(drug_global_768).float().unsqueeze(0).to(device)
    h = model.drug_global_proj(x)                                   # (1, d)
    return F.normalize(h, dim=-1).squeeze(0)                        # (d,)


@torch.no_grad()
def compute_random_baseline(
    model: CALMEncoderV3,
    drug_embeddings: dict[int, torch.Tensor],
    target_vec: dict[str, torch.Tensor],
    n_random: int,
    device: str,
    seed: int = 42,
) -> dict[str, dict]:
    """For each target, compute the cosine distribution of N random training
    drugs vs that target's global vector. This is the chance baseline used to
    convert raw cosines into z-scores (margin in std-units of the random
    distribution).

    Returns {gene: {'mean': float, 'std': float, 'p95': float, 'n': int}}.
    """
    rng = np.random.default_rng(seed)
    all_drug_idxs = np.array(list(drug_embeddings.keys()))
    sample = rng.choice(all_drug_idxs, size=min(n_random, len(all_drug_idxs)),
                        replace=False)
    # Stack random drug embeddings → encode once → (N, d)
    stacked = torch.stack([
        torch.as_tensor(drug_embeddings[int(i)]).float() for i in sample
    ]).to(device)                                                       # (N, 768)
    h = model.drug_global_proj(stacked)                                  # (N, d)
    rand_d = F.normalize(h, dim=-1)                                      # (N, d)

    out: dict[str, dict] = {}
    for gene, t_vec in target_vec.items():
        cos = (rand_d @ t_vec).cpu().numpy()                             # (N,)
        out[gene] = {
            "mean": float(cos.mean()),
            "std": float(cos.std()),
            "p95": float(np.percentile(cos, 95)),
            "n": int(len(cos)),
        }
    return out


def phase_b_competence_map(
    model: CALMEncoderV3,
    binder_tsv: Path,
    drug_embeddings: dict[int, torch.Tensor],
    protein_embeddings: dict[int, torch.Tensor],
    device: str,
    out_summary_csv: Path,
    out_per_binder_csv: Path,
    n_random_baseline: int = 1000,
) -> None:
    print("\n" + "=" * 60)
    print("PHASE B — encoder competence map for 10 decoder targets")
    print("=" * 60)

    binders = pd.read_csv(binder_tsv, sep="\t")
    print(f"  Loaded {len(binders)} (target, drug) rows from {binder_tsv.name}")
    expected_cols = {"target_group", "target_gene", "drug_name", "drug_idx",
                     "target_pidx_canonical", "leakage_class", "resolution_method"}
    missing = expected_cols - set(binders.columns)
    if missing:
        raise RuntimeError(
            f"Binder TSV is missing pre-resolved columns: {missing}. "
            f"Run data/dtsfm/scripts/build_decoder_target_binders.py first.")

    # ---- Encode each unique target gene's protein once ----
    target_vec: dict[str, torch.Tensor] = {}
    target_pidx_used: dict[str, int] = {}
    for _, r in binders.drop_duplicates("target_gene").iterrows():
        gene = str(r["target_gene"])
        pidx = r["target_pidx_canonical"]
        if pd.isna(pidx) or int(pidx) not in protein_embeddings:
            continue
        target_vec[gene] = encode_target_global(model, protein_embeddings[int(pidx)], device)
        target_pidx_used[gene] = int(pidx)
    n_unique_genes = binders["target_gene"].nunique()
    print(f"  Targets resolved & encoded: {len(target_vec)} of {n_unique_genes} unique genes")
    for gene in sorted(binders["target_gene"].unique()):
        if gene not in target_vec:
            print(f"    MISSING target embedding: {gene}")

    # ---- Random-pair baseline per target (1 GPU pass for all sampled drugs) ----
    print(f"  Computing random-pair baseline (N={n_random_baseline} drugs per target)...")
    baseline = compute_random_baseline(
        model, drug_embeddings, target_vec, n_random_baseline, device,
    )
    for gene, b in sorted(baseline.items()):
        print(f"    baseline {gene:<10}  mean={b['mean']:+.3f}  std={b['std']:.3f}  "
              f"p95={b['p95']:+.3f}  (N={b['n']})")

    # ---- Per-binder cosine + z-score against the baseline ----
    per_binder_rows = []
    for _, row in binders.iterrows():
        gene = str(row["target_gene"])
        drug_idx = int(row["drug_idx"]) if not pd.isna(row["drug_idx"]) else -1
        cos = None
        z = None
        status = ""
        if gene not in target_vec:
            status = "target_missing"
        elif drug_idx < 0:
            status = "drug_not_in_training"  # Class C
        elif drug_idx not in drug_embeddings:
            status = "drug_idx_no_embedding"
        else:
            try:
                t_vec = target_vec[gene]
                d_vec = encode_drug_global(model, drug_embeddings[drug_idx], device)
                cos = float((t_vec * d_vec).sum().item())
                b = baseline[gene]
                z = (cos - b["mean"]) / max(b["std"], 1e-6)
                status = "ok"
            except Exception as e:
                cos = None
                status = f"error_{type(e).__name__}"
                print(f"    [error] {row['drug_name']} ({gene}): {type(e).__name__}: {e}")
        per_binder_rows.append({
            "target_group":      str(row["target_group"]),
            "target_gene":       gene,
            "drug_name":         str(row["drug_name"]),
            "drug_idx":          drug_idx,
            "leakage_class":     str(row.get("leakage_class", "")),
            "resolution_method": str(row.get("resolution_method", "")),
            "cosine":            cos,
            "z_vs_baseline":     z,
            "status":            status,
        })

    per_binder_df = pd.DataFrame(per_binder_rows)
    out_per_binder_csv.parent.mkdir(parents=True, exist_ok=True)
    per_binder_df.to_csv(out_per_binder_csv, sep="\t", index=False)
    print(f"  Wrote per-binder CSV → {out_per_binder_csv}")

    # ---- Aggregate per target_group ----
    ok = per_binder_df[per_binder_df["status"] == "ok"]
    grouped = ok.groupby("target_group").agg(
        n_binders_evaluated=("cosine", "size"),
        mean_cosine=("cosine", "mean"),
        min_cosine=("cosine", "min"),
        max_cosine=("cosine", "max"),
        mean_z=("z_vs_baseline", "mean"),
    ).reset_index()
    tried = per_binder_df.groupby("target_group").size().rename("n_rows_tried")
    grouped = grouped.merge(tried, on="target_group", how="right").fillna(
        {"n_binders_evaluated": 0}
    )
    # Per-group baseline (use the canonical gene's baseline — first gene in group)
    group_to_first_gene = (
        binders.drop_duplicates("target_group").set_index("target_group")["target_gene"]
        .to_dict()
    )
    grouped["baseline_mean"] = grouped["target_group"].map(
        lambda g: baseline.get(group_to_first_gene.get(g, ""), {}).get("mean")
    )
    grouped["baseline_std"] = grouped["target_group"].map(
        lambda g: baseline.get(group_to_first_gene.get(g, ""), {}).get("std")
    )
    grouped = grouped.sort_values("target_group").reset_index(drop=True)
    grouped.to_csv(out_summary_csv, sep="\t", index=False)
    print(f"  Wrote summary CSV   → {out_summary_csv}")

    # ---- Verdict table ----
    # Verdict thresholds based on margin (z-score against random baseline):
    #   z >= 3.0   STRONG     known binders are way out in the tail of the
    #                          random distribution — encoder discriminates well
    #   z >= 1.5   MODERATE   visible signal, decoder usable but flagged
    #   z >= 0.5   WEAK       above chance but small margin — encoder blind-spot risk
    #   z < 0.5    NULL       indistinguishable from random — decoder will fail here
    print("\n  === Encoder competence map (margin-based verdict) ===")
    print(f"  {'Target':<12} {'rows':>5} {'eval':>5}  "
          f"{'mean cos':>9} {'base μ':>8} {'base σ':>7}  {'mean z':>7}  Verdict")
    print("  " + "-" * 80)
    for _, g in grouped.iterrows():
        n_eval = int(g["n_binders_evaluated"])
        n_tried = int(g["n_rows_tried"])
        mz = g["mean_z"]
        if n_eval == 0:
            verdict = "NO DRUG MATCHES — check resolver / SMILES"
        elif pd.isna(mz):
            verdict = "n/a"
        elif mz >= 3.0:
            verdict = "STRONG — decoder OK to deploy"
        elif mz >= 1.5:
            verdict = "MODERATE — usable, monitor"
        elif mz >= 0.5:
            verdict = "WEAK — encoder blind-spot risk"
        else:
            verdict = "NULL — DO NOT deploy decoder here"
        mc = g["mean_cosine"]
        bm = g["baseline_mean"]
        bs = g["baseline_std"]
        def f(x, w, prec):
            return f"{x:+{w}.{prec}f}" if (x is not None and not pd.isna(x)) else " " * (w-2) + "n/a"
        print(f"  {g['target_group']:<12} {n_tried:>5} {n_eval:>5}  "
              f"{f(mc,9,3)} {f(bm,8,3)} {f(bs,7,3)}  {f(mz,7,2)}  {verdict}")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to epoch_010.pt (the locked v3 encoder)")
    ap.add_argument("--metadata_v3_csv", type=Path, required=True)
    ap.add_argument("--drug_npz", type=Path, required=True)
    ap.add_argument("--protein_dir", type=Path, required=True)
    ap.add_argument("--binder_tsv", type=Path, required=True,
                    help="Pre-resolved decoder_target_binders.tsv (built by "
                         "data/dtsfm/scripts/build_decoder_target_binders.py)")
    ap.add_argument("--out_summary_csv", type=Path, required=True)
    ap.add_argument("--out_per_binder_csv", type=Path, required=True)
    ap.add_argument("--n_random_baseline", type=int, default=1000,
                    help="Random drugs sampled per target for the chance baseline")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"=== dtSFM v3 decoder Phase 0 smoketest ===")
    print(f"Device:  {device}")
    if device == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")

    print("\n[1/4] Loading checkpoint + building model...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt.get("config") or default_model_cfg())
    model = CALMEncoderV3(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    pc = model.count_parameters()
    print(f"    Base epoch in ckpt: {ckpt.get('epoch','?')}")
    print(f"    Total trainable parameters: {pc['total']:,}  (expected ~14.4 M)")

    print("\n[2/4] Loading embeddings...")
    drug_embeddings = load_drug_embeddings(args.drug_npz)
    protein_embeddings = load_protein_embeddings(args.protein_dir)
    print(f"    drug embeddings cached:       {len(drug_embeddings):,}")
    print(f"    protein embeddings cached:    {len(protein_embeddings):,}")

    print("\n[3/4] Phase A — forward pass on one Class-A pair")
    metadata_df = pd.read_csv(
        args.metadata_v3_csv,
        usecols=["drug_smiles", "drug_idx", "protein_id", "protein_idx"],
        nrows=2000,  # search budget for finding one ETKDG-embeddable row
    )
    try:
        phase_a_forward_pass(model, metadata_df, drug_embeddings, protein_embeddings, device)
    except Exception as e:
        # Phase A is a sanity check; failure does NOT block Phase B (the load-
        # bearing decoder competence map, which uses only the global heads).
        import traceback
        print(f"\n  Phase A FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("  Continuing to Phase B — Phase B uses only the global drug + protein")
        print("  heads (frozen MoLFormer / ESM-2 → linear projection), no atom encoder")
        print("  or cross-attention. A Phase A failure does NOT impact decoder design.\n")

    print("\n[4/4] Phase B — competence map for 10 decoder targets")
    phase_b_competence_map(
        model=model,
        binder_tsv=args.binder_tsv,
        drug_embeddings=drug_embeddings,
        protein_embeddings=protein_embeddings,
        device=device,
        out_summary_csv=args.out_summary_csv,
        out_per_binder_csv=args.out_per_binder_csv,
        n_random_baseline=args.n_random_baseline,
    )

    print("=== Smoketest complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
