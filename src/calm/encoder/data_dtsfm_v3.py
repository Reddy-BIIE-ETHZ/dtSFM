"""
Dataset and DataLoader utilities for dtSFM v3.

Reads `metadata_v3.csv` (built by build_v3_metadata_index.py) plus per-pair
contact NPZs (built by extract_contacts_v3.py / extract_pdbbind_contacts.py)
and the cached embeddings (built by embed_drugs_molformer_v3.py /
embed_proteins_esm2_v3.py).

Memory model (Option B locked at B-1 spec):
    - All 22,964 protein per-residue ESM-2 embeddings preloaded into a single
      RAM-resident dict {protein_idx: tensor(L_i, 1280) fp16}. Total ~25 GB.
    - All 522,776 drug mean-pool MoLFormer vectors loaded as a single fp16
      array (522,776, 768) ~ 800 MB.
    - Contact NPZs read on-the-fly per (pair_idx) since each is small (~10 KB)
      and we only need a slice per training step.

Held-out filtering:
    Pairs whose (drug_smiles_canonical, protein_id) match an entry in
    `audit/dtsfm/heldout_validation_pairs.tsv` are NEVER returned by the
    training subset (and ARE returned by an explicit "heldout" subset for
    zero-shot validation). The TSV is the source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .model_v3 import element_to_idx, N_ELEMENTS


# --------------------------------------------------------------------------- #
# 1. Cached embedding loaders
# --------------------------------------------------------------------------- #
def load_drug_embeddings(npz_path: Path) -> dict[int, torch.Tensor]:
    """Load the single drug NPZ into a dict {drug_idx: (768,) fp16 tensor}.

    Returns a dict for O(1) lookup. Total RAM ~ 800 MB.
    """
    print(f"Loading drug embeddings from {npz_path}")
    data = np.load(npz_path)
    drug_idx = data["drug_idx"].astype(np.int64)
    embeddings = torch.from_numpy(np.asarray(data["embeddings"], dtype=np.float16))
    print(f"  loaded {embeddings.shape[0]:,} drug embeddings, "
          f"shape={tuple(embeddings.shape)}, dtype={embeddings.dtype}")
    return {int(idx): embeddings[i] for i, idx in enumerate(drug_idx)}


def load_protein_embeddings(
    embeds_dir: Path,
    expected_n: int | None = None,
) -> dict[int, torch.Tensor]:
    """Load every protein .pt into a dict {protein_idx: (L, 1280) fp16 tensor}.

    Each file is named `{protein_idx:06d}.pt`. Total RAM ~ 25 GB.
    """
    print(f"Preloading protein embeddings from {embeds_dir}")
    files = sorted(embeds_dir.glob("*.pt"))
    if expected_n is not None and len(files) != expected_n:
        print(f"  WARNING: expected {expected_n} files, found {len(files)}")
    cache: dict[int, torch.Tensor] = {}
    for i, f in enumerate(files):
        idx = int(f.stem)
        cache[idx] = torch.load(f, map_location="cpu", weights_only=True)
        if (i + 1) % 5000 == 0:
            print(f"  [{i+1:>6d}/{len(files):>6d}] loaded", flush=True)
    print(f"  loaded {len(cache):,} protein embeddings; "
          f"first={tuple(next(iter(cache.values())).shape)}")
    return cache


# --------------------------------------------------------------------------- #
# 2. Held-out filter
# --------------------------------------------------------------------------- #
def load_heldout_pairs(tsv_path: Path | None) -> set[tuple[str, str]]:
    """Read held-out validation pairs TSV → set of (canonical_smiles, protein_id).

    Returns empty set if `tsv_path` is None or missing.
    """
    if tsv_path is None or not tsv_path.exists():
        return set()
    df = pd.read_csv(tsv_path, sep="\t")
    out: set[tuple[str, str]] = set()
    # Accept either ('canonical_smiles', 'protein_id') columns or
    # ('drug_smiles', 'protein_id') columns; normalise.
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "drug_smiles"
    pid_col = "protein_id"
    if smiles_col not in df.columns or pid_col not in df.columns:
        print(f"  [warn] heldout TSV missing expected columns; ignoring")
        return out
    for _, r in df.iterrows():
        out.add((str(r[smiles_col]), str(r[pid_col])))
    return out


# --------------------------------------------------------------------------- #
# 3. Per-pair NPZ reader
# --------------------------------------------------------------------------- #
@dataclass
class PairNPZ:
    """Lightweight in-memory representation of a contact NPZ slice."""
    elem_ids: torch.Tensor      # (N_atoms,) int64
    xyz: torch.Tensor           # (N_atoms, 3) float32
    contact_map: torch.Tensor   # (N_atoms, L_res) bool
    n_atoms: int
    n_residues: int


def read_pair_npz(npz_path: str | Path) -> PairNPZ | None:
    """Read one contact NPZ → PairNPZ. Returns None on read error."""
    try:
        d = np.load(npz_path, allow_pickle=True)
        elements = [str(e) for e in d["ligand_atom_elements"]]
        elem_ids = torch.tensor(
            [element_to_idx(e) for e in elements], dtype=torch.int64,
        )
        xyz = torch.from_numpy(np.asarray(d["ligand_xyz"], dtype=np.float32))
        contact_map = torch.from_numpy(np.asarray(d["contact_map"], dtype=np.bool_))
        return PairNPZ(
            elem_ids=elem_ids,
            xyz=xyz,
            contact_map=contact_map,
            n_atoms=int(elem_ids.shape[0]),
            n_residues=int(contact_map.shape[1]),
        )
    except Exception as e:
        print(f"  [warn] failed to read {npz_path}: {e}")
        return None


# --------------------------------------------------------------------------- #
# 4. Dataset
# --------------------------------------------------------------------------- #
class DTSFMv3PairDataset(Dataset):
    """One sample = one (drug, protein) pair with contacts.

    `__getitem__` returns a dict with all fields needed by the model + loss:

        drug_global   : (768,) fp16
        drug_elem_ids : (N_atoms,) int64
        drug_xyz      : (N_atoms, 3) float32
        drug_mask     : always all True at this stage; collate sets pad-mask
        n_atoms       : int
        protein_emb   : (L_res, 1280) fp16
        n_residues    : int
        contact_target          : (N_atoms, L_res) bool
        interface_target_drug   : (N_atoms,) {0, 1} long
        interface_target_protein: (L_res,)   {0, 1} long
        affinity_target         : float
        affinity_valid          : bool
        source_tier             : float (1.0 or 1.5)
        pair_idx                : int (for debugging)
    """

    def __init__(
        self,
        metadata_csv: Path,
        drug_embeddings: dict[int, torch.Tensor],
        protein_embeddings: dict[int, torch.Tensor],
        pair_indices: Iterable[int] | None = None,
        max_atoms: int = 256,
        max_residues: int = 1024,
        affinity_qualifier_filter: tuple[str, ...] = ("=",),
    ):
        super().__init__()
        df = pd.read_csv(metadata_csv)
        if pair_indices is not None:
            df = df.iloc[list(pair_indices)].reset_index(drop=True)
        self.df = df
        self.drug_embeddings = drug_embeddings
        self.protein_embeddings = protein_embeddings
        self.max_atoms = int(max_atoms)
        self.max_residues = int(max_residues)
        self.affinity_qualifier_filter = set(affinity_qualifier_filter)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict | None:
        row = self.df.iloc[i]
        npz = read_pair_npz(row["npz_path"])
        if npz is None:
            return None
        # Truncate atoms / residues if needed (rare; max_atoms covers 99.9% of NPZs)
        N = min(npz.n_atoms, self.max_atoms)
        # Protein embedding may be truncated to <= max_residues already in B-0c
        protein_emb_full = self.protein_embeddings[int(row["protein_idx"])]
        L_emb = protein_emb_full.shape[0]
        # The contact NPZ's L_res may differ from the cached embedding's L if
        # the embedding was truncated to 1024 but the structure has more
        # residues. Take the min so they align.
        L = min(npz.n_residues, L_emb, self.max_residues)
        if N == 0 or L == 0:
            return None

        elem_ids = npz.elem_ids[:N]
        xyz = npz.xyz[:N]
        contact_map = npz.contact_map[:N, :L]
        protein_emb = protein_emb_full[:L]

        # Derive per-atom and per-residue interface targets from contacts
        atom_interface = contact_map.any(dim=-1).long()
        res_interface = contact_map.any(dim=0).long()

        # Affinity: mask out non-`=` qualifiers (default config)
        qualifier = str(row.get("qualifier", "="))
        affinity_valid = qualifier in self.affinity_qualifier_filter
        affinity_target = float(row["affinity_log"])

        return {
            "drug_global":   self.drug_embeddings[int(row["drug_idx"])],
            "drug_elem_ids": elem_ids,
            "drug_xyz":      xyz,
            "n_atoms":       N,
            "protein_emb":   protein_emb,
            "n_residues":    L,
            "contact_target":          contact_map,
            "interface_target_drug":   atom_interface,
            "interface_target_protein": res_interface,
            "affinity_target": affinity_target,
            "affinity_valid":  affinity_valid,
            "source_tier":     float(row["source_tier"]),
            "pair_idx":        int(row["pair_idx"]),
            # Identifier indices for direction-aware unique-pool retrieval (eval only)
            "drug_idx":        int(row["drug_idx"]),
            "protein_idx":     int(row["protein_idx"]),
        }


# --------------------------------------------------------------------------- #
# 5. Collate function (variable-length atoms + residues → padded batch)
# --------------------------------------------------------------------------- #
def collate_dtsfm_v3(samples: list[dict | None]) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length pairs to (B, N_max, ...) / (B, L_max, ...).

    Drops None samples (read failures).
    """
    samples = [s for s in samples if s is not None]
    if len(samples) == 0:
        raise RuntimeError("collate received an empty batch (all reads failed)")

    B = len(samples)
    N_max = max(s["n_atoms"] for s in samples)
    L_max = max(s["n_residues"] for s in samples)

    # Atom-side padded tensors
    drug_global   = torch.stack([s["drug_global"] for s in samples], dim=0)  # (B, 768)
    drug_elem_ids = torch.zeros((B, N_max), dtype=torch.int64)
    drug_xyz      = torch.zeros((B, N_max, 3), dtype=torch.float32)
    drug_mask     = torch.zeros((B, N_max), dtype=torch.bool)
    interface_target_drug = torch.zeros((B, N_max), dtype=torch.long)
    # Residue-side padded tensors
    # Use float32 for protein_emb collation; will be cast to model dtype before forward
    protein_emb   = torch.zeros((B, L_max, samples[0]["protein_emb"].shape[1]), dtype=torch.float32)
    protein_mask  = torch.zeros((B, L_max), dtype=torch.bool)
    interface_target_protein = torch.zeros((B, L_max), dtype=torch.long)
    # Joint contact map
    contact_target = torch.zeros((B, N_max, L_max), dtype=torch.bool)
    # Scalars
    affinity_target = torch.zeros(B, dtype=torch.float32)
    affinity_valid  = torch.zeros(B, dtype=torch.bool)
    source_tier     = torch.zeros(B, dtype=torch.float32)
    pair_idx        = torch.zeros(B, dtype=torch.long)
    drug_idx        = torch.zeros(B, dtype=torch.long)
    protein_idx     = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(samples):
        N, L = s["n_atoms"], s["n_residues"]
        drug_elem_ids[i, :N] = s["drug_elem_ids"]
        drug_xyz[i, :N]      = s["drug_xyz"]
        drug_mask[i, :N]     = True
        interface_target_drug[i, :N] = s["interface_target_drug"]

        protein_emb[i, :L]   = s["protein_emb"].float()  # cast fp16 → fp32
        protein_mask[i, :L]  = True
        interface_target_protein[i, :L] = s["interface_target_protein"]

        contact_target[i, :N, :L] = s["contact_target"]

        affinity_target[i] = s["affinity_target"]
        affinity_valid[i]  = s["affinity_valid"]
        source_tier[i]     = s["source_tier"]
        pair_idx[i]        = s["pair_idx"]
        drug_idx[i]        = s["drug_idx"]
        protein_idx[i]     = s["protein_idx"]

    return {
        "drug_global":   drug_global.float(),  # cast fp16 → fp32 for model
        "drug_elem_ids": drug_elem_ids,
        "drug_xyz":      drug_xyz,
        "drug_mask":     drug_mask,
        "protein_emb":   protein_emb,
        "protein_mask":  protein_mask,
        "interface_target_drug":    interface_target_drug,
        "interface_target_protein": interface_target_protein,
        "contact_target":           contact_target,
        "affinity_target": affinity_target,
        "affinity_valid":  affinity_valid,
        "source_tier":     source_tier,
        "pair_idx":        pair_idx,
        "drug_idx":        drug_idx,
        "protein_idx":     protein_idx,
    }


# --------------------------------------------------------------------------- #
# 6. Train/val/test split helper (cluster-based on protein_idx)
# --------------------------------------------------------------------------- #
def make_cluster_splits(
    metadata_csv: Path,
    heldout_tsv: Path | None = None,
    test_frac: float = 0.10,
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split pair_idx into train/val/test by cluster (protein_idx).

    Held-out validation pairs (if any) are routed to a separate "heldout"
    subset and removed from train/val/test entirely.

    Returns a dict with keys: train, val, test, heldout (lists of pair_idx).
    """
    import random

    df = pd.read_csv(metadata_csv,
                     usecols=["pair_idx", "drug_smiles", "protein_id",
                              "protein_idx", "cluster_id"])
    # Apply held-out filter
    heldout_pairs = load_heldout_pairs(heldout_tsv)
    if heldout_pairs:
        is_heldout = df.apply(
            lambda r: (str(r["drug_smiles"]), str(r["protein_id"])) in heldout_pairs,
            axis=1,
        )
    else:
        is_heldout = pd.Series([False] * len(df))
    heldout_idx = df.loc[is_heldout, "pair_idx"].astype(int).tolist()
    df = df.loc[~is_heldout].reset_index(drop=True)

    cluster_ids = sorted(df["cluster_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(cluster_ids)
    n = len(cluster_ids)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    test_clusters = set(cluster_ids[:n_test])
    val_clusters = set(cluster_ids[n_test:n_test + n_val])
    train_clusters = set(cluster_ids[n_test + n_val:])

    train_idx = df.loc[df["cluster_id"].isin(train_clusters), "pair_idx"].astype(int).tolist()
    val_idx   = df.loc[df["cluster_id"].isin(val_clusters),   "pair_idx"].astype(int).tolist()
    test_idx  = df.loc[df["cluster_id"].isin(test_clusters),  "pair_idx"].astype(int).tolist()

    print(f"Splits: train={len(train_idx):,}  val={len(val_idx):,}  "
          f"test={len(test_idx):,}  heldout={len(heldout_idx):,}")
    print(f"        protein clusters: train={len(train_clusters)}  "
          f"val={len(val_clusters)}  test={len(test_clusters)}")
    return {
        "train":   train_idx,
        "val":     val_idx,
        "test":    test_idx,
        "heldout": heldout_idx,
    }
