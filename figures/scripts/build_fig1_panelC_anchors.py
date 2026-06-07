"""Build the anchor overlay TSV for Fig 1 Panel C.

For each paper anchor drug:
  1. Canonicalize its SMILES with RDKit
  2. Match against drug_idx_to_smiles.tsv (also canonicalized)
  3. If matched → use that drug_idx's UMAP coord from drug_umap.tsv
  4. If not matched → Tanimoto-nearest in training, use its coord (mark as approx)

Output: data/dtsfm/fig1_panelC/anchor_umap.tsv
  drug_name, source, smiles, drug_idx, umap_x, umap_y, match_type
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

REPO_ROOT = Path("/Users/reddys/Downloads/CALM-0.1.0")
DATA_DIR = REPO_ROOT / "data/dtsfm/fig1_panelC"

# ---------------------------------------------------------------------------- #
# Anchor list — paper-relevant drugs spanning Figs 3, 4, 5
# ---------------------------------------------------------------------------- #
ANCHORS = [
    # Klaeger TKIs (§3 safety panel) — drug-OOD by design; will Tanimoto-NN match
    ("dasatinib",    "Klaeger TKI",  "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1"),
    ("imatinib",     "Klaeger TKI",  "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
    ("sunitinib",    "Klaeger TKI",  "CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\\C(=O)Nc3ccc(F)cc32)c1C"),
    ("ibrutinib",    "Klaeger TKI",  "C=CC(=O)N1CCC[C@H]1c1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21"),
    ("erlotinib",    "Klaeger TKI",  "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"),
    ("gefitinib",    "Klaeger TKI",  "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"),
    # F4 repurposing anchors (§4)
    ("MCC950",       "Repurposing",  "CC(C)(C)[C@H](OC(=O)NS(=O)(=O)c1cc(C(C)(C)C)c2c(c1)CCO2)C(=O)O"),  # NLRP3 gold-standard, may need adjustment
    ("AB680",        "Repurposing",  "O=P(O)(O)[C@@H](c1ccccc1)Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O"),  # CD73 Phase 2; approximation
    ("ADU-S100",     "Repurposing",  "O=P1(S)O[C@@H]2[C@H](n3cnc4c(=O)[nH]c(N)nc43)O[C@H](COP(=S)(O)O[C@@H]3[C@H](O1)CO[C@H]3n1cnc3c(=O)[nH]c(N)nc31)[C@H]2F"),  # STING1
    # Generative leads (§5)
    ("trametinib",   "Generative",   "CC1=C2C(=C(N(C1=O)C)NC3=C(C=C(C=C3)I)F)C(=O)N(C(=O)N2C4=CC=CC(=C4)NC(=O)C)C5CC5"),
    ("lorlatinib",   "Generative",   "C[C@@H]1C2=C(C=CC(=C2)F)C(=O)N(CC3=NN(C(=C3C4=CC(=C(N=C4)N)O1)C#N)C)C"),
    ("midostaurin",  "Generative",   "C[C@@]12[C@@H]([C@@H](C[C@@H](O1)N3C4=CC=CC=C4C5=C6C(=C7C8=CC=CC=C8N2C7=C53)CNC6=O)N(C)C(=O)C9=CC=CC=C9)OC"),
    ("quizartinib",  "Generative",   "CC(C)(C)c1cc(NC(=O)Nc2ccc(-c3cn4c(n3)sc3cc(OCCN5CCOCC5)ccc34)cc2)no1"),
]


def canon(smi):
    """Canonical SMILES; None on parse failure."""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def fp(smi):
    """ECFP4 (radius 2, 1024-bit) Morgan fingerprint for Tanimoto."""
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) if m else None


def main():
    print("Loading lookups...")
    drug_idx_smi = pd.read_csv(DATA_DIR / "drug_idx_to_smiles.tsv", sep="\t")
    print(f"  drug_idx_to_smiles: {len(drug_idx_smi):,}")
    drug_umap = pd.read_csv(DATA_DIR / "drug_umap.tsv", sep="\t")
    print(f"  drug_umap: {len(drug_umap):,}")

    # Canonicalize the training-pool SMILES (slow but one-time)
    print("Canonicalizing 522K training SMILES...")
    drug_idx_smi["canon"] = [canon(s) for s in drug_idx_smi["drug_smiles"]]
    drop = drug_idx_smi["canon"].isna().sum()
    print(f"  {drop} unparseable")
    drug_idx_smi = drug_idx_smi.dropna(subset=["canon"])
    canon_to_idx = dict(zip(drug_idx_smi["canon"], drug_idx_smi["drug_idx"]))
    print(f"  canon→idx lookup: {len(canon_to_idx):,}")

    coord = dict(zip(drug_umap["drug_idx"], zip(drug_umap["umap_x"], drug_umap["umap_y"])))

    rows = []
    needs_nn = []
    for name, source, smi in ANCHORS:
        c = canon(smi)
        if c is None:
            print(f"  {name}: CANNOT parse SMILES")
            continue
        if c in canon_to_idx:
            di = int(canon_to_idx[c])
            if di in coord:
                x, y = coord[di]
                rows.append({"drug_name": name, "source": source, "smiles": c,
                             "drug_idx": di, "umap_x": x, "umap_y": y,
                             "match_type": "exact"})
                print(f"  {name}: exact match drug_idx={di}")
                continue
        needs_nn.append((name, source, c))
        print(f"  {name}: not in training, will Tanimoto-NN")

    # Tanimoto NN for those not directly matched
    if needs_nn:
        print(f"Computing Tanimoto NN for {len(needs_nn)} anchors...")
        # Precompute training fingerprints (subset that has UMAP coord)
        train_idx_arr = drug_idx_smi["drug_idx"].values
        train_canon = drug_idx_smi["canon"].values
        print(f"  Fingerprinting {len(train_canon):,} training drugs (slow ~30s)...")
        train_fps = []
        for s in train_canon:
            m = Chem.MolFromSmiles(s)
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) if m else None)

        for name, source, c in needs_nn:
            anc_fp = fp(c)
            if anc_fp is None:
                continue
            # Bulk Tanimoto
            sims = DataStructs.BulkTanimotoSimilarity(anc_fp, train_fps)
            best_i = int(np.argmax(sims))
            best_sim = sims[best_i]
            best_idx = int(train_idx_arr[best_i])
            if best_idx in coord:
                x, y = coord[best_idx]
                rows.append({"drug_name": name, "source": source, "smiles": c,
                             "drug_idx": best_idx, "umap_x": x, "umap_y": y,
                             "match_type": f"NN_Tan={best_sim:.2f}"})
                print(f"  {name}: NN drug_idx={best_idx} Tanimoto={best_sim:.2f}")

    out = DATA_DIR / "anchor_umap.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"Wrote {out} ({len(rows)} anchors)")


if __name__ == "__main__":
    main()
