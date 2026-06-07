"""
train_klaeger_finetune.py — fine-tune dtSFM v3 on Klaeger 2017 Kinobeads
to produce dtSFM-kinase, with explicit binary hit/n.i. supervision and
real measured K_d_app regression on hits.

Initialized from epoch_010.pt (locked v3 production checkpoint). Same
architecture, same encoders frozen. Lower learning rate (1e-5). Multi-task
loss adapted for the fine-tune setting:

    L_global   : InfoNCE on Klaeger train pairs (positive pairs only)
    L_binary   : BCE on Klaeger hit (1) vs n.i. (0) labels — NEW for fine-tune
    L_affinity : MSE on Klaeger pK_d_app (hits only) — uses real measurements
    L_contact  : SKIP (no contact ground truth for Klaeger pairs without crystals)
    L_interface: SKIP (same reason)

Drug atom features come from RDKit ETKDG conformers (same as inference time).
Protein features come from cached ESM-2 .pt files (same training cache).

Run on:    Euler GPU (~1-2 hr wall on A100/Quadro RTX 6000)
Output:    --output_dir/  with per-epoch checkpoints + best_by_test_auroc.pt

Usage:
    python3 -m calm.encoder.train_klaeger_finetune \\
        --base_checkpoint /cluster/scratch/reddys/dtsfm_v3/runs/b3_20260506_191149/epoch_010.pt \\
        --train_pairs_tsv audit/dtsfm/klaeger2017/finetune/klaeger_finetune_train_pairs.tsv \\
        --test_pairs_tsv  audit/dtsfm/klaeger2017/finetune/klaeger_finetune_test_pairs.tsv \\
        --drug_npz   /cluster/scratch/reddys/dtsfm_v3/embeddings/drug_embeddings.npz \\
        --protein_dir /cluster/scratch/reddys/dtsfm_v3/embeddings/protein_embeds/ \\
        --gene_mapping audit/dtsfm/protein_id_to_gene_symbol.tsv \\
        --metadata_v3_csv /cluster/scratch/reddys/dtsfm_v3/metadata_v3.csv \\
        --output_dir /cluster/scratch/reddys/dtsfm_v3/runs/dtsfm_kinase_finetune_$(date +%Y%m%d_%H%M%S) \\
        --epochs 10 \\
        --batch_size 64 \\
        --lr 1e-5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from .model_v3 import CALMEncoderV3, element_to_idx, N_ELEMENTS
from .train_dtsfm_v3 import default_model_cfg


# --------------------------------------------------------------------------- #
# 1. Klaeger fine-tune dataset
# --------------------------------------------------------------------------- #
def atoms_from_smiles_etkdg(smiles: str, max_atoms: int = 256):
    """Generate atom features (element_ids, xyz) from SMILES via RDKit ETKDGv3.

    Robustness: tries sanitize → no-kekulize → manual aromaticity, in that order.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    def _try_parse(s: str):
        for sanitize in (True, False):
            try:
                m = Chem.MolFromSmiles(s, sanitize=sanitize)
                if m is None:
                    continue
                if not sanitize:
                    Chem.SanitizeMol(m, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
                return m
            except Exception:
                continue
        return None

    try:
        mol = _try_parse(smiles)
        if mol is None:
            return None
        try:
            mol = Chem.AddHs(mol)
        except Exception:
            pass
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            try:
                mol = Chem.RemoveHs(mol)
            except Exception:
                pass
            if AllChem.EmbedMolecule(mol, params) != 0:
                return None
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=50)
        except Exception:
            pass
        try:
            mol = Chem.RemoveHs(mol)
        except Exception:
            pass
        if mol.GetNumConformers() == 0:
            return None
        conf = mol.GetConformer()
        elems, coords = [], []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            pos = conf.GetAtomPosition(atom.GetIdx())
            elems.append(element_to_idx(atom.GetSymbol()))
            coords.append([pos.x, pos.y, pos.z])
            if len(elems) >= max_atoms:
                break
        if not elems:
            return None
        return (
            torch.tensor(elems, dtype=torch.int64),
            torch.tensor(coords, dtype=torch.float32),
        )
    except Exception:
        return None


def load_drug_embeddings(npz_path: Path) -> dict:
    """Load drug NPZ into {smiles → 768-dim fp16 tensor}.

    For Klaeger fine-tune we need lookup by SMILES, but our cache is keyed by
    drug_idx. We need a smiles → drug_idx map from metadata."""
    print(f"  Loading drug embeddings: {npz_path}")
    data = np.load(npz_path)
    drug_idx = data["drug_idx"].astype(np.int64)
    embeddings = torch.from_numpy(np.asarray(data["embeddings"], dtype=np.float16))
    return {int(idx): embeddings[i] for i, idx in enumerate(drug_idx)}


def build_smiles_to_idx(metadata_v3_csv: Path) -> dict[str, int]:
    """smiles → drug_idx from metadata_v3."""
    df = pd.read_csv(metadata_v3_csv, usecols=["drug_smiles", "drug_idx"])
    return dict(zip(df["drug_smiles"].astype(str), df["drug_idx"].astype(int)))


def load_protein_embeddings(embeds_dir: Path) -> dict[int, torch.Tensor]:
    """Load all protein .pt files into {protein_idx → fp16 (L,1280) tensor}."""
    print(f"  Preloading protein embeddings: {embeds_dir}")
    cache: dict[int, torch.Tensor] = {}
    for f in sorted(embeds_dir.glob("*.pt")):
        idx = int(f.stem)
        cache[idx] = torch.load(f, map_location="cpu", weights_only=True)
    print(f"    {len(cache):,} proteins cached")
    return cache


def build_gene_to_protein_idx(gene_mapping_tsv: Path,
                              metadata_v3_csv: Path) -> dict[str, int]:
    """For each gene_symbol, pick the canonical protein_idx (one with most pairs in training)."""
    df_map = pd.read_csv(gene_mapping_tsv, sep="\t")
    df_meta = pd.read_csv(metadata_v3_csv, usecols=["protein_id", "protein_idx"])
    pair_counts = df_meta.groupby("protein_idx").size().to_dict()
    df_map["n_pairs"] = df_map["protein_idx"].map(pair_counts).fillna(0).astype(int)
    df_map = df_map[df_map["gene_symbol"].notna() & (df_map["gene_symbol"] != "")]
    df_map = df_map.sort_values("n_pairs", ascending=False)
    return dict(df_map.drop_duplicates("gene_symbol").set_index("gene_symbol")["protein_idx"])


class KlaegerFineTuneDataset(Dataset):
    """One sample = one (drug, gene, K_d_app, label) row from the Klaeger fine-tune table.

    Returns:
        drug_global   : (768,) fp16
        drug_elem_ids : (N,) int64
        drug_xyz      : (N, 3) float32
        protein_emb   : (L, 1280) fp16
        is_hit_30uM   : 0/1 int — primary fine-tune binary label
        is_hit_1uM    : 0/1 int
        is_hit_100nM  : 0/1 int
        kdapp_pK      : float (NaN if non-hit; affinity head only sees hits)
        is_quantified : bool — has measured K_d? (used to mask affinity loss)
    """

    def __init__(
        self, pairs_tsv: Path,
        drug_smiles_to_idx: dict[str, int],
        drug_embeddings: dict[int, torch.Tensor],
        protein_embeddings: dict[int, torch.Tensor],
        gene_to_pidx: dict[str, int],
        max_atoms: int = 256,
        max_protein_len: int = 1024,
    ):
        self.df = pd.read_csv(pairs_tsv, sep="\t")
        # Pre-resolve drug_idx and protein_idx
        self.df["drug_idx"] = self.df["drug_smiles"].map(drug_smiles_to_idx)
        self.df["protein_idx"] = self.df["kinase_gene"].map(gene_to_pidx)
        before = len(self.df)
        self.df = self.df.dropna(subset=["drug_idx", "protein_idx"])
        self.df["drug_idx"] = self.df["drug_idx"].astype(int)
        self.df["protein_idx"] = self.df["protein_idx"].astype(int)
        self.df = self.df.reset_index(drop=True)
        print(f"    Pairs after drug/gene resolution: {len(self.df):,} of {before:,}")

        self.drug_embeddings = drug_embeddings
        self.protein_embeddings = protein_embeddings
        self.max_atoms = max_atoms
        self.max_protein_len = max_protein_len

        # Pre-cache atom features per unique SMILES (slow up-front, fast at training time)
        unique_smis = self.df["drug_smiles"].drop_duplicates().tolist()
        print(f"    Generating ETKDG atoms for {len(unique_smis):,} unique drugs...")
        t0 = time.time()
        self.atom_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        n_fail = 0
        for i, s in enumerate(unique_smis):
            atoms = atoms_from_smiles_etkdg(s, max_atoms=max_atoms)
            if atoms is None:
                n_fail += 1
            else:
                self.atom_cache[s] = atoms
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(unique_smis)}  rate={  (i+1) / (time.time()-t0):.1f}/s  "
                      f"fail={n_fail}", flush=True)
        print(f"    Atom cache built: {len(self.atom_cache):,} drugs, "
              f"{n_fail} failed ({time.time()-t0:.1f}s)")

        # Drop pairs whose drug failed atom generation
        self.df = self.df[self.df["drug_smiles"].isin(self.atom_cache)].reset_index(drop=True)
        print(f"    Final pair count: {len(self.df):,}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict | None:
        r = self.df.iloc[i]
        smi = r["drug_smiles"]
        drug_global = self.drug_embeddings.get(int(r["drug_idx"]))
        if drug_global is None:
            return None
        atoms = self.atom_cache.get(smi)
        if atoms is None:
            return None
        elem_ids, xyz = atoms

        protein_full = self.protein_embeddings.get(int(r["protein_idx"]))
        if protein_full is None:
            return None
        L = min(protein_full.shape[0], self.max_protein_len)
        protein_emb = protein_full[:L]

        N = elem_ids.shape[0]

        return {
            "drug_global":  drug_global,
            "drug_elem_ids": elem_ids,
            "drug_xyz":      xyz,
            "n_atoms":       N,
            "protein_emb":   protein_emb,
            "n_residues":    L,
            "is_hit_30uM":  int(r.get("is_hit_30uM", 0)),
            "is_hit_1uM":   int(r.get("is_hit_1uM", 0)),
            "is_hit_100nM": int(r.get("is_hit_100nM", 0)),
            "kdapp_pK":     float(r["kdapp_pK"]) if pd.notna(r["kdapp_pK"]) else float("nan"),
            "is_quantified": str(r["kdapp_status"]) in {"numeric", "low_confidence"},
        }


def collate(samples):
    samples = [s for s in samples if s is not None]
    if len(samples) == 0:
        raise RuntimeError("collate: empty batch")
    B = len(samples)
    N_max = max(s["n_atoms"] for s in samples)
    L_max = max(s["n_residues"] for s in samples)
    drug_global = torch.stack([s["drug_global"] for s in samples], dim=0)
    drug_elem_ids = torch.zeros((B, N_max), dtype=torch.int64)
    drug_xyz      = torch.zeros((B, N_max, 3), dtype=torch.float32)
    drug_mask     = torch.zeros((B, N_max), dtype=torch.bool)
    protein_emb   = torch.zeros((B, L_max, samples[0]["protein_emb"].shape[1]), dtype=torch.float32)
    protein_mask  = torch.zeros((B, L_max), dtype=torch.bool)
    is_hit_30uM   = torch.zeros(B, dtype=torch.float32)
    is_hit_1uM    = torch.zeros(B, dtype=torch.float32)
    is_hit_100nM  = torch.zeros(B, dtype=torch.float32)
    kdapp_pK      = torch.zeros(B, dtype=torch.float32)
    is_quantified = torch.zeros(B, dtype=torch.bool)
    for i, s in enumerate(samples):
        N, L = s["n_atoms"], s["n_residues"]
        drug_elem_ids[i, :N] = s["drug_elem_ids"]
        drug_xyz[i, :N]      = s["drug_xyz"]
        drug_mask[i, :N]     = True
        protein_emb[i, :L]   = s["protein_emb"].float()
        protein_mask[i, :L]  = True
        is_hit_30uM[i]   = s["is_hit_30uM"]
        is_hit_1uM[i]    = s["is_hit_1uM"]
        is_hit_100nM[i]  = s["is_hit_100nM"]
        kdapp_pK[i]      = s["kdapp_pK"]
        is_quantified[i] = s["is_quantified"]

    return {
        "drug_global":   drug_global.float(),
        "drug_elem_ids": drug_elem_ids,
        "drug_xyz":      drug_xyz,
        "drug_mask":     drug_mask,
        "protein_emb":   protein_emb,
        "protein_mask":  protein_mask,
        "is_hit_30uM":  is_hit_30uM,
        "is_hit_1uM":   is_hit_1uM,
        "is_hit_100nM": is_hit_100nM,
        "kdapp_pK":     kdapp_pK,
        "is_quantified": is_quantified,
    }


# --------------------------------------------------------------------------- #
# 2. Add a binary classification head to dtSFM v3 for the fine-tune
# --------------------------------------------------------------------------- #
class BinaryClassifierHead(torch.nn.Module):
    """Tiny 2-layer MLP on concat(global_drug, global_protein) → binder probability.

    Initialized fresh; pre-trained weights (cross_attn / contact / etc.) preserved
    via dtSFM v3 base. Output is logit (no sigmoid)."""

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * d_model, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, g_drug, g_protein):
        x = torch.cat([g_drug, g_protein], dim=-1)
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# 3. Metrics helpers
# --------------------------------------------------------------------------- #
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores); sl = labels[order]
    tpr = np.cumsum(sl == 1) / n_pos
    fpr = np.cumsum(sl == 0) / n_neg
    tpr = np.concatenate([[0.0], tpr]); fpr = np.concatenate([[0.0], fpr])
    return float(np.trapz(tpr, fpr))


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", type=Path, required=True,
                    help="dtSFM v3 checkpoint to fine-tune from (e.g., epoch_010.pt)")
    ap.add_argument("--train_pairs_tsv", type=Path, required=True)
    ap.add_argument("--test_pairs_tsv",  type=Path, required=True)
    ap.add_argument("--drug_npz", type=Path, required=True)
    ap.add_argument("--protein_dir", type=Path, required=True)
    ap.add_argument("--gene_mapping", type=Path, required=True)
    ap.add_argument("--metadata_v3_csv", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--alpha_global",   type=float, default=1.0)
    ap.add_argument("--alpha_binary",   type=float, default=2.0)
    ap.add_argument("--alpha_affinity", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== dtSFM-kinase fine-tune ===")
    print(f"  device:           {device}")
    print(f"  base_checkpoint:  {args.base_checkpoint}")
    print(f"  train pairs:      {args.train_pairs_tsv}")
    print(f"  test pairs:       {args.test_pairs_tsv}")
    print(f"  output dir:       {args.output_dir}")
    print(f"  epochs:           {args.epochs}, lr={args.lr}, batch={args.batch_size}")

    # ---- Load base model ----
    print("\n[1/5] Loading base checkpoint + building model...")
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt.get("config") or default_model_cfg())
    model = CALMEncoderV3(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    binary_head = BinaryClassifierHead(d_model=int(cfg.d_model)).to(device)
    print(f"    Base epoch in ckpt: {ckpt.get('epoch','?')}")

    # ---- Load embeddings + caches ----
    print("\n[2/5] Loading embeddings + caches...")
    smiles_to_idx = build_smiles_to_idx(args.metadata_v3_csv)
    drug_embeddings = load_drug_embeddings(args.drug_npz)
    drug_smi_to_emb = {smi: drug_embeddings[idx] for smi, idx in smiles_to_idx.items()
                        if idx in drug_embeddings}
    protein_embeddings = load_protein_embeddings(args.protein_dir)
    gene_to_pidx = build_gene_to_protein_idx(args.gene_mapping, args.metadata_v3_csv)
    print(f"    smiles → drug_idx mapping:    {len(smiles_to_idx):,}")
    print(f"    drug embeddings cached:       {len(drug_embeddings):,}")
    print(f"    gene → protein_idx mapping:   {len(gene_to_pidx):,}")

    # ---- Build datasets ----
    print("\n[3/5] Building datasets...")
    print("  TRAIN:")
    train_ds = KlaegerFineTuneDataset(
        args.train_pairs_tsv, smiles_to_idx, drug_embeddings,
        protein_embeddings, gene_to_pidx,
    )
    print("  TEST:")
    test_ds = KlaegerFineTuneDataset(
        args.test_pairs_tsv, smiles_to_idx, drug_embeddings,
        protein_embeddings, gene_to_pidx,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Optimizer + scheduler ----
    print("\n[4/5] Optimizer + scheduler...")
    params = list(model.parameters()) + list(binary_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    def lr_fn(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # ---- Training loop ----
    print("\n[5/5] Fine-tuning...")
    pos_weight = torch.tensor([
        (train_ds.df["fineturne_label"] == 0).sum()
        / max(1, (train_ds.df["fineturne_label"] == 1).sum())
    ], device=device)
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  Positive class weight (n_neg/n_pos): {pos_weight.item():.2f}")

    @torch.no_grad()
    def evaluate_test() -> dict:
        model.eval(); binary_head.eval()
        all_b_logits, all_paff_pred, all_kdapp = [], [], []
        all_label_30, all_label_1, all_label_100, all_quant = [], [], [], []
        for batch in test_loader:
            out = model(
                drug_global=batch["drug_global"].to(device),
                drug_elem_ids=batch["drug_elem_ids"].to(device),
                drug_xyz=batch["drug_xyz"].to(device),
                drug_mask=batch["drug_mask"].to(device),
                protein_emb=batch["protein_emb"].to(device),
                protein_mask=batch["protein_mask"].to(device),
            )
            b_logit = binary_head(out["global_features_drug"], out["global_features_protein"])
            all_b_logits.append(b_logit.cpu())
            all_paff_pred.append(out["affinity_pred"].cpu())
            all_kdapp.append(batch["kdapp_pK"])
            all_label_30.append(batch["is_hit_30uM"])
            all_label_1.append(batch["is_hit_1uM"])
            all_label_100.append(batch["is_hit_100nM"])
            all_quant.append(batch["is_quantified"])
        b_logits  = torch.cat(all_b_logits).numpy()
        paff_pred = torch.cat(all_paff_pred).numpy()
        kdapp     = torch.cat(all_kdapp).numpy()
        label_30  = torch.cat(all_label_30).numpy()
        label_1   = torch.cat(all_label_1).numpy()
        label_100 = torch.cat(all_label_100).numpy()
        quant     = torch.cat(all_quant).numpy().astype(bool)
        # AUROCs at three thresholds
        results = {
            "auroc_30uM_binary":  auroc(b_logits, label_30),
            "auroc_1uM_binary":   auroc(b_logits, label_1),
            "auroc_100nM_binary": auroc(b_logits, label_100),
            "auroc_30uM_paff":    auroc(paff_pred, label_30),
            "auroc_1uM_paff":     auroc(paff_pred, label_1),
            "auroc_100nM_paff":   auroc(paff_pred, label_100),
            "n_test_pairs":       int(len(b_logits)),
            "n_quant":            int(quant.sum()),
        }
        # Pearson r on quantified pairs
        if quant.sum() > 2:
            r = np.corrcoef(paff_pred[quant], kdapp[quant])[0, 1]
            results["pearson_paff_vs_pK"] = float(r)
        return results

    history = []
    best_auroc = -1.0
    for epoch in range(args.epochs):
        model.train(); binary_head.train()
        t0 = time.time()
        sums = defaultdict(float)
        n_batches = 0
        for batch in train_loader:
            out = model(
                drug_global=batch["drug_global"].to(device),
                drug_elem_ids=batch["drug_elem_ids"].to(device),
                drug_xyz=batch["drug_xyz"].to(device),
                drug_mask=batch["drug_mask"].to(device),
                protein_emb=batch["protein_emb"].to(device),
                protein_mask=batch["protein_mask"].to(device),
            )
            # Global InfoNCE on this batch (same as encoder training)
            z_d = out["global_features_drug"]
            z_p = out["global_features_protein"]
            B = z_d.shape[0]
            sim = out["logit_scale"] * (z_d @ z_p.t())
            tgt = torch.arange(B, device=device)
            L_global = 0.5 * (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.t(), tgt))

            # Binary classification on hit_30uM
            b_logit = binary_head(z_d, z_p)
            label_30 = batch["is_hit_30uM"].to(device)
            L_binary = bce_loss(b_logit, label_30)

            # Affinity regression on quantified pairs
            quant = batch["is_quantified"].to(device)
            paff_pred = out["affinity_pred"]
            if quant.sum() > 0:
                kdapp_pK = batch["kdapp_pK"].to(device)
                L_aff = F.mse_loss(paff_pred[quant], kdapp_pK[quant])
            else:
                L_aff = torch.tensor(0.0, device=device)

            loss = (args.alpha_global   * L_global
                  + args.alpha_binary   * L_binary
                  + args.alpha_affinity * L_aff)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            sums["L_total"]   += float(loss.item())
            sums["L_global"]  += float(L_global.item())
            sums["L_binary"]  += float(L_binary.item())
            sums["L_aff"]     += float(L_aff.item())
            n_batches += 1
        avg = {k: v/n_batches for k, v in sums.items()}
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch+1}/{args.epochs}  ({elapsed:.0f}s, {n_batches} batches)")
        print(f"    L_total={avg['L_total']:.3f}  L_global={avg['L_global']:.3f}  "
              f"L_binary={avg['L_binary']:.3f}  L_aff={avg['L_aff']:.3f}")

        # Evaluate on test
        eval_metrics = evaluate_test()
        print(f"    [test] AUROC binary  30uM/1uM/100nM = "
              f"{eval_metrics['auroc_30uM_binary']:.3f} / "
              f"{eval_metrics['auroc_1uM_binary']:.3f} / "
              f"{eval_metrics['auroc_100nM_binary']:.3f}")
        print(f"    [test] AUROC pAff    30uM/1uM/100nM = "
              f"{eval_metrics['auroc_30uM_paff']:.3f} / "
              f"{eval_metrics['auroc_1uM_paff']:.3f} / "
              f"{eval_metrics['auroc_100nM_paff']:.3f}")
        if "pearson_paff_vs_pK" in eval_metrics:
            print(f"    [test] Pearson r (pAff vs pK_d_app, n={eval_metrics['n_quant']}): "
                  f"{eval_metrics['pearson_paff_vs_pK']:.3f}")

        history.append({"epoch": epoch+1, **avg, **eval_metrics})

        # Save checkpoint
        ckpt_path = args.output_dir / f"finetune_epoch_{epoch+1:03d}.pt"
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "binary_head_state_dict": binary_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "args": {k: str(v) for k, v in vars(args).items()},
            "eval_metrics": eval_metrics,
        }, ckpt_path)

        # Track best by 100nM AUROC (drug-like potency, the headline metric)
        if eval_metrics["auroc_100nM_binary"] > best_auroc:
            best_auroc = eval_metrics["auroc_100nM_binary"]
            best_path = args.output_dir / "best_by_test_auroc.pt"
            import shutil
            shutil.copy(ckpt_path, best_path)
            print(f"    [best] new best 100nM binary AUROC = {best_auroc:.3f} → {best_path}")

    # Final summary
    history_df = pd.DataFrame(history)
    history_path = args.output_dir / "history.csv"
    history_df.to_csv(history_path, index=False)
    summary_path = args.output_dir / "summary.json"
    summary = {
        "epochs": args.epochs,
        "best_100nM_binary_auroc": best_auroc,
        "final_epoch_metrics": history[-1],
        "training_size": len(train_ds),
        "test_size": len(test_ds),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n=== dtSFM-kinase fine-tune complete ===")
    print(f"  Best 100nM binary AUROC on test: {best_auroc:.3f}")
    print(f"  History: {history_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
