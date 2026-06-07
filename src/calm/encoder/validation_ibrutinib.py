"""
Validation: Ibrutinib vs Acalabrutinib off-target safety prediction.

Ibrutinib (BTK inhibitor) causes atrial fibrillation in 10-16% of patients
due to off-target inhibition of CSK (C-terminal Src kinase). Acalabrutinib
is a more selective BTK inhibitor with much less CSK inhibition and lower
AF rates (~3%).

This script tests whether dtSFM can predict:
1. Both drugs bind BTK (their intended target)
2. Ibrutinib binds CSK more strongly than acalabrutinib
3. Ibrutinib's full off-target profile is broader than acalabrutinib's

Neither drug's exact SMILES appears in the training data, so this is a
genuinely prospective prediction.

Usage:
    python -m calm.encoder.validation_ibrutinib \
        --checkpoint /path/to/best_model.pth \
        --data_dir data/dtsfm_v2 \
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


# Drug SMILES from PubChem
IBRUTINIB_SMILES = "C=CC(=O)N1CCC[C@@H](C1)n1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21"
ACALABRUTINIB_SMILES = "CC#Cc1nc(Nc2ccc(OC(=O)N3CCC[C@@H]3C)cc2)c2n1ccn(C)c2=O"

# Key targets (UniProt IDs for reference)
# BTK: Q06187 (intended target for both drugs)
# CSK: P41240 (off-target causing ibrutinib AF)


def embed_smiles(smiles_list: list[str], device: str = "cpu") -> torch.Tensor:
    """Embed SMILES strings with MoLFormer-XL.

    Returns token-level embeddings (N, L, 768).
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct",
        deterministic_eval=True,
        trust_remote_code=True,
    )
    model = model.to(device).eval()

    with torch.no_grad():
        inputs = tokenizer(
            smiles_list, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (N, L, 768)
        mask = inputs["attention_mask"]  # (N, L)

    return hidden.cpu(), mask.cpu()


def run_validation(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    device: str = "cpu",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(data_dir)

    print("=" * 60)
    print("dtSFM Validation: Ibrutinib vs Acalabrutinib Safety")
    print("=" * 60)

    # --- Step 1: Load checkpoint and build projection heads ---
    print("\nStep 1: Loading trained model...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    enc_ag = nn.Sequential(
        nn.Linear(768, 2048, bias=False), nn.ReLU(),
        nn.Linear(2048, 512, bias=False),
    )
    enc_ab = nn.Sequential(
        nn.Linear(1280, 2048, bias=False), nn.ReLU(),
        nn.Linear(2048, 512, bias=False),
    )
    enc_ag.load_state_dict({
        k.replace("encoder_ag.", ""): v
        for k, v in ckpt.items() if k.startswith("encoder_ag.")
    })
    enc_ab.load_state_dict({
        k.replace("encoder_ab.", ""): v
        for k, v in ckpt.items() if k.startswith("encoder_ab.")
    })
    enc_ag.eval()
    enc_ab.eval()

    temperature = 1.0 / math.exp(ckpt["logit_scale"].item())
    print(f"  Learned temperature: {temperature:.4f}")

    # --- Step 2: Embed the two drugs ---
    print("\nStep 2: Embedding ibrutinib and acalabrutinib with MoLFormer...")
    drug_hidden, drug_mask = embed_smiles(
        [IBRUTINIB_SMILES, ACALABRUTINIB_SMILES], device=device
    )
    print(f"  Ibrutinib tokens: {drug_mask[0].sum().item()}")
    print(f"  Acalabrutinib tokens: {drug_mask[1].sum().item()}")

    # Mean pool and project
    with torch.no_grad():
        mask_f = drug_mask.unsqueeze(-1).float()
        drug_pooled = (drug_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        drug_proj = F.normalize(enc_ag(drug_pooled), dim=-1)  # (2, 512)

    ibrutinib_emb = drug_proj[0]   # (512,)
    acalabrutinib_emb = drug_proj[1]  # (512,)

    # --- Step 3: Load and project all unique protein targets ---
    print("\nStep 3: Loading protein embeddings...")
    ab_embed = torch.load(
        str(data / "ab_embed.pt"), map_location="cpu", weights_only=True
    )
    ab_mask = torch.load(
        str(data / "ab_mask.pt"), map_location="cpu", weights_only=True
    )

    with torch.no_grad():
        ab_mf = ab_mask.unsqueeze(-1).float()
        ab_pooled = (ab_embed * ab_mf).sum(1) / ab_mf.sum(1).clamp(min=1)
        all_ab_proj = F.normalize(enc_ab(ab_pooled), dim=-1)  # (1358, 512)

    # Free memory
    del ab_embed, ab_mask
    print(f"  Projected {all_ab_proj.shape[0]} unique proteins")

    # --- Step 4: Map target names to indices ---
    metadata = pd.read_csv(data / "metadata.csv")
    unique_targets = metadata["target_seq"].drop_duplicates().tolist()
    target_id_map = {}
    for _, row in metadata.drop_duplicates("target_seq").iterrows():
        seq = row["target_seq"]
        idx = unique_targets.index(seq)
        target_id_map[row["target_id"]] = idx

    # Find BTK and CSK indices
    btk_idx = target_id_map.get("BTK")
    csk_idx = target_id_map.get("CSK")
    print(f"\n  BTK index: {btk_idx}")
    print(f"  CSK index: {csk_idx}")

    if btk_idx is None or csk_idx is None:
        print("ERROR: BTK or CSK not found in target_id_map")
        print(f"Available targets with BTK/CSK: {[k for k in target_id_map if 'BTK' in k or 'CSK' in k]}")
        return

    # --- Step 5: Compute cosine similarities ---
    print("\nStep 4: Computing drug-target similarities...")

    # Ibrutinib similarities to all targets
    ibr_sims = (ibrutinib_emb @ all_ab_proj.T).numpy()  # (1358,)
    aca_sims = (acalabrutinib_emb @ all_ab_proj.T).numpy()  # (1358,)

    # Key comparisons
    ibr_btk = float(ibr_sims[btk_idx])
    ibr_csk = float(ibr_sims[csk_idx])
    aca_btk = float(aca_sims[btk_idx])
    aca_csk = float(aca_sims[csk_idx])

    print(f"\n  {'':30s} {'BTK (intended)':>15s} {'CSK (off-target)':>18s}")
    print(f"  {'-'*65}")
    print(f"  {'Ibrutinib (less selective)':30s} {ibr_btk:>15.4f} {ibr_csk:>18.4f}")
    print(f"  {'Acalabrutinib (more selective)':30s} {aca_btk:>15.4f} {aca_csk:>18.4f}")
    print(f"  {'Difference (ibr - aca)':30s} {ibr_btk - aca_btk:>15.4f} {ibr_csk - aca_csk:>18.4f}")

    # --- Step 6: Rank all targets for each drug ---
    print("\nStep 5: Ranking all targets for each drug...")

    ibr_ranked = np.argsort(-ibr_sims)
    aca_ranked = np.argsort(-aca_sims)

    ibr_btk_rank = int(np.where(ibr_ranked == btk_idx)[0][0]) + 1
    ibr_csk_rank = int(np.where(ibr_ranked == csk_idx)[0][0]) + 1
    aca_btk_rank = int(np.where(aca_ranked == btk_idx)[0][0]) + 1
    aca_csk_rank = int(np.where(aca_ranked == csk_idx)[0][0]) + 1

    n_targets = len(ibr_sims)
    print(f"\n  {'':30s} {'BTK rank':>12s} {'CSK rank':>12s}   (of {n_targets} targets)")
    print(f"  {'-'*60}")
    print(f"  {'Ibrutinib':30s} {ibr_btk_rank:>12d} {ibr_csk_rank:>12d}")
    print(f"  {'Acalabrutinib':30s} {aca_btk_rank:>12d} {aca_csk_rank:>12d}")
    print(f"  {'Random chance':30s} {n_targets//2:>12d} {n_targets//2:>12d}")

    # --- Step 7: Full selectivity profile comparison ---
    print("\nStep 6: Top-20 predicted targets for each drug...")

    # Reverse lookup: index to target_id
    idx_to_tid = {v: k for k, v in target_id_map.items()}

    print(f"\n  Ibrutinib top-20 predicted targets:")
    for i, tidx in enumerate(ibr_ranked[:20]):
        tid = idx_to_tid.get(int(tidx), f"idx_{tidx}")
        sim = ibr_sims[tidx]
        marker = ""
        if int(tidx) == btk_idx:
            marker = " <-- BTK (intended)"
        elif int(tidx) == csk_idx:
            marker = " <-- CSK (off-target, AF risk)"
        print(f"    {i+1:3d}. {tid:20s}  cos_sim = {sim:.4f}{marker}")

    print(f"\n  Acalabrutinib top-20 predicted targets:")
    for i, tidx in enumerate(aca_ranked[:20]):
        tid = idx_to_tid.get(int(tidx), f"idx_{tidx}")
        sim = aca_sims[tidx]
        marker = ""
        if int(tidx) == btk_idx:
            marker = " <-- BTK (intended)"
        elif int(tidx) == csk_idx:
            marker = " <-- CSK (off-target, AF risk)"
        print(f"    {i+1:3d}. {tid:20s}  cos_sim = {sim:.4f}{marker}")

    # --- Step 8: Broader off-target profile ---
    print("\nStep 7: Off-target breadth comparison...")

    # Known ibrutinib off-targets from literature
    known_offtargets = ["CSK", "EGFR", "ERBB2", "ERBB4", "TEC", "BMX",
                        "TXK", "ITK", "HCK", "FGR", "LYN", "FYN", "YES1", "SRC"]

    print(f"\n  Known ibrutinib off-targets — where does each drug rank them?")
    print(f"  {'Target':15s} {'Ibrutinib rank':>16s} {'Acalabrutinib rank':>20s} {'Ibr closer?':>14s}")
    print(f"  {'-'*70}")

    n_ibr_closer = 0
    n_compared = 0
    for ot in known_offtargets:
        if ot in target_id_map:
            ot_idx = target_id_map[ot]
            ibr_rank = int(np.where(ibr_ranked == ot_idx)[0][0]) + 1
            aca_rank = int(np.where(aca_ranked == ot_idx)[0][0]) + 1
            closer = "YES" if ibr_rank < aca_rank else "no"
            if ibr_rank < aca_rank:
                n_ibr_closer += 1
            n_compared += 1
            print(f"  {ot:15s} {ibr_rank:>16d} {aca_rank:>20d} {closer:>14s}")
        else:
            print(f"  {ot:15s} {'not in data':>16s}")

    if n_compared > 0:
        print(f"\n  Ibrutinib ranks off-target closer: {n_ibr_closer}/{n_compared} ({100*n_ibr_closer/n_compared:.0f}%)")
        print(f"  (If random, expected: 50%)")

    # --- Save results ---
    results = {
        "drug_comparison": {
            "ibrutinib_btk_sim": ibr_btk,
            "ibrutinib_csk_sim": ibr_csk,
            "acalabrutinib_btk_sim": aca_btk,
            "acalabrutinib_csk_sim": aca_csk,
            "ibrutinib_btk_rank": ibr_btk_rank,
            "ibrutinib_csk_rank": ibr_csk_rank,
            "acalabrutinib_btk_rank": aca_btk_rank,
            "acalabrutinib_csk_rank": aca_csk_rank,
            "n_targets": n_targets,
        },
        "selectivity_prediction": {
            "ibr_csk_sim_minus_aca_csk_sim": ibr_csk - aca_csk,
            "prediction_correct": ibr_csk > aca_csk,
            "ibr_ranks_offtargets_closer_pct": (
                100 * n_ibr_closer / n_compared if n_compared > 0 else None
            ),
        },
        "temperature": float(temperature),
        "note": "Neither ibrutinib nor acalabrutinib SMILES appear in training data. BTK and CSK protein sequences are in training.",
    }

    with open(out / "ibrutinib_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {out}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="dtSFM ibrutinib vs acalabrutinib safety validation"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_validation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
