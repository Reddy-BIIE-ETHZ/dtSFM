#!/usr/bin/env python3
"""screen_decoder_safety.py — Stage 5 proteome-wide off-target safety screen.

For each §F.5 candidate that passes:
    - Class C (drug not in v3 training)         [from F5_leakage_analysis.tsv]
    - AF3 iPTM ≥ 0.7 AND PAE ≤ 5 Å              [from F5_results.tsv]
    - max Tanimoto to approved drugs < 0.4      [filter; "novel vs approved"]
    - (strict) max Tanimoto to training < 0.4   [filter; "patent-novel scaffold"]

Re-encode the candidate's canonical SMILES through MoLFormer-XL → encoder
drug_global_proj → 512-d L2-normalized drug global, then compute cosine vs
all 22,964 training protein global vectors → rank → roll up to gene-symbol
level (best rank per gene, mirroring §F.1 framework).

Outputs (per filter regime):
    F5_safety_screen_strict.tsv    — "novel composition of matter" set
    F5_safety_screen_relaxed.tsv   — "novel chemotype vs approved" set

Each row contains:
    sample_name, target_intended, canonical_smiles,
    intended_target_rank, intended_target_cosine,
    top10_off_target_genes, top10_off_target_cosines, top10_off_target_ranks,
    n_other_decoder_targets_in_top50  (cross-target promiscuity flag),
    max_off_target_cosine, leakage_class,
    max_tanimoto_to_training, max_tanimoto_to_approved

Antibody analogue: this is the small-molecule equivalent of running each novel
anti-target Fab against a panel of 22,964 surface proteins to identify
cross-reactivities. Candidates that bind the intended target in the top-50
AND have no top-100 off-target hits in the safety panel proceed to wet-lab.

Run on Euler GPU. ~5-15 min for the full §F.5 candidate set.
Python 3.6+ compatible.
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from calm.encoder.model_v3 import CALMEncoderV3
from calm.encoder.train_dtsfm_v3 import default_model_cfg as default_encoder_cfg


# --------------------------------------------------------------------------- #
# MoLFormer + encoder reencoding (mirrors ReencodingPipeline in screen_decoder)
# --------------------------------------------------------------------------- #
class Reencoder:
    def __init__(self, molformer_name, encoder_checkpoint, device):
        from transformers import AutoModel, AutoTokenizer
        print("  Loading MoLFormer model ({})...".format(molformer_name))
        self.tokenizer = AutoTokenizer.from_pretrained(
            molformer_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            if "[PAD]" in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = "[PAD]"
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.molformer = AutoModel.from_pretrained(
            molformer_name, trust_remote_code=True, deterministic_eval=True,
        ).to(device).eval()
        for p in self.molformer.parameters():
            p.requires_grad_(False)

        print("  Loading v3 encoder from {}...".format(encoder_checkpoint))
        ckpt = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
        ecfg = OmegaConf.create(ckpt.get("config") or default_encoder_cfg())
        self.encoder = CALMEncoderV3(ecfg).to(device).eval()
        self.encoder.load_state_dict(ckpt["model_state_dict"])
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def smiles_to_drug_global_512(self, smiles_list, batch_size=32):
        out = []
        for start in range(0, len(smiles_list), batch_size):
            chunk = smiles_list[start:start + batch_size]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            ).to(self.device)
            mf = self.molformer(input_ids=enc["input_ids"],
                                attention_mask=enc["attention_mask"])
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (mf.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            h = self.encoder.drug_global_proj(pooled.float())
            out.append(F.normalize(h, dim=-1).cpu())
        return torch.cat(out, dim=0)                                    # (N, 512)

    @torch.no_grad()
    def proteome_global_matrix(self, protein_dir, max_len=1024):
        """Compute (N_proteins, 512) L2-normalized global vectors for the full
        training proteome. Runs once per session — slow first time, fast after.
        """
        protein_dir = Path(protein_dir)
        files = sorted(protein_dir.glob("*.pt"))
        n = len(files)
        print("  Computing proteome global vectors for {:,} proteins...".format(n))
        out = torch.zeros(n, 512, dtype=torch.float32)
        protein_idxs = []
        t0 = time.time()
        for i, f in enumerate(files):
            pidx = int(f.stem)
            protein_idxs.append(pidx)
            prot = torch.load(f, map_location="cpu", weights_only=True).float()
            if prot.shape[0] > max_len:
                prot = prot[:max_len]
            x = prot.unsqueeze(0).to(self.device)                       # (1, L, 1280)
            mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=self.device)
            h = self.encoder.protein_proj(x) * mask.unsqueeze(-1).float()
            pooled = self.encoder.protein_pool_pre(h, mask)
            out[pidx] = F.normalize(pooled, dim=-1).squeeze(0).cpu()
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (n - i - 1) / rate
                print("    {}/{}  rate={:.0f}/s  ETA={:.0f}s".format(
                    i + 1, n, rate, eta))
        print("    proteome globals computed in {:.0f}s".format(time.time() - t0))
        return out, protein_idxs


def load_gene_mapping(gene_mapping_tsv):
    """{protein_idx: gene_symbol}"""
    out = {}
    with open(gene_mapping_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                pidx = int(row["protein_idx"])
            except (KeyError, ValueError):
                continue
            gene = (row.get("gene_symbol") or "").strip() or None
            out[pidx] = gene
    return out


def load_target_to_pidxs(binder_tsv):
    """{target_gene: set of training protein_idxs}"""
    out = defaultdict(set)
    with open(binder_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                pidx = int(row["target_pidx_canonical"])
            except (KeyError, ValueError):
                continue
            gene = (row.get("target_gene") or "").strip()
            if gene:
                out[gene].add(pidx)
    return dict(out)


def proteome_rank(drug_512, proteome_512, gene_map, decoder_target_genes,
                  intended_gene, top_k=50):
    """For one drug global, compute cosines vs all proteins, roll up to gene
    level (best rank per gene), return:
        intended_rank, intended_cosine,
        top_k_genes, top_k_cosines, top_k_ranks,
        n_other_decoder_targets_in_top50.
    """
    cos = (proteome_512 @ drug_512).cpu().numpy()                       # (N_proteins,)
    # Sort proteins by cosine descending
    order = np.argsort(-cos)
    # Roll up to gene level: best rank per gene
    seen_genes = set()
    gene_ranks = []  # list of (gene, rank, cosine)
    intended_rank = None
    intended_cos = None
    for rank, pidx in enumerate(order, start=1):
        gene = gene_map.get(int(pidx))
        if gene is None or gene in seen_genes:
            continue
        seen_genes.add(gene)
        gene_ranks.append((gene, rank, float(cos[pidx])))
        if gene == intended_gene and intended_rank is None:
            intended_rank = len(gene_ranks)  # rank in unique-gene list
            intended_cos = float(cos[pidx])

    top_k_genes = gene_ranks[:top_k]
    top_50 = gene_ranks[:50]
    n_decoder_in_top50 = sum(
        1 for g, _, _ in top_50
        if g in decoder_target_genes and g != intended_gene
    )
    return {
        "intended_rank":   intended_rank,
        "intended_cos":    intended_cos,
        "top_k_genes":     [g for g, _, _ in top_k_genes],
        "top_k_cosines":   [c for _, _, c in top_k_genes],
        "top_k_ranks":     [r for _, r, _ in top_k_genes],
        "n_other_decoder_targets_in_top50": n_decoder_in_top50,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_tsv", type=Path,
                    default=Path("audit/dtsfm/decoder_af3/F5_results.tsv"))
    ap.add_argument("--leakage_tsv", type=Path,
                    default=Path("audit/dtsfm/decoder_af3/F5_leakage_analysis.tsv"))
    ap.add_argument("--binder_tsv", type=Path,
                    default=Path("audit/dtsfm/decoder_target_binders.tsv"))
    ap.add_argument("--gene_mapping", type=Path,
                    default=Path("audit/dtsfm/protein_id_to_gene_symbol.tsv"))
    ap.add_argument("--encoder_checkpoint", type=Path, required=True)
    ap.add_argument("--protein_dir", type=Path, required=True)
    ap.add_argument("--molformer_name", default="ibm/MoLFormer-XL-both-10pct")
    ap.add_argument("--output_dir", type=Path,
                    default=Path("audit/dtsfm/decoder_af3"))
    ap.add_argument("--iptm_threshold", type=float, default=0.7)
    ap.add_argument("--pae_threshold", type=float, default=5.0)
    ap.add_argument("--tanimoto_approved_threshold", type=float, default=0.4)
    ap.add_argument("--tanimoto_training_threshold", type=float, default=0.4)
    ap.add_argument("--top_k_off_target", type=int, default=50)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== screen_decoder_safety (Stage 5: proteome safety screen) ===")
    print("Device: {}".format(device))

    # --- Load joined results + leakage ---
    print("\n[1/5] Joining F5_results.tsv + F5_leakage_analysis.tsv ...")
    with open(args.results_tsv) as f:
        results = list(csv.DictReader(f, delimiter="\t"))
    with open(args.leakage_tsv) as f:
        leakage = {r["sample_name"]: r for r in csv.DictReader(f, delimiter="\t")}
    for r in results:
        l = leakage.get(r["sample_name"], {})
        for k in ("leakage_class", "max_tanimoto_to_training",
                  "max_tanimoto_to_approved", "nearest_training_smiles",
                  "nearest_approved_smiles", "canonical_smiles"):
            if k not in r or not r.get(k):
                r[k] = l.get(k, "")
    print("  Joined {} cohort rows".format(len(results)))

    # --- Apply filters ---
    def pass_af3(r):
        try:
            return (float(r.get("iptm") or 0) >= args.iptm_threshold and
                    float(r.get("interface_pae_min") or 99) <= args.pae_threshold)
        except (ValueError, TypeError):
            return False

    def f_tan(r, key):
        try:
            return float(r.get(key) or 1.0)
        except ValueError:
            return 1.0

    decoder_strata = {"top_cosine", "scaffold_div", "mid_cosine"}
    relaxed = []
    strict = []
    for r in results:
        if r.get("stratum") not in decoder_strata:
            continue
        if r.get("leakage_class") != "C":
            continue
        if not pass_af3(r):
            continue
        tan_app = f_tan(r, "max_tanimoto_to_approved")
        if tan_app >= args.tanimoto_approved_threshold:
            continue
        relaxed.append(r)
        tan_train = f_tan(r, "max_tanimoto_to_training")
        if tan_train < args.tanimoto_training_threshold:
            strict.append(r)

    print("\n[2/5] Filter cascade (decoder rows only):")
    print("  Total decoder rows in cohort:                 {}".format(
        sum(1 for r in results if r.get("stratum") in decoder_strata)))
    print("  Class C:                                      {}".format(
        sum(1 for r in results
            if r.get("stratum") in decoder_strata and r.get("leakage_class") == "C")))
    print("  Class C + AF3 pass:                           {}".format(
        sum(1 for r in results
            if r.get("stratum") in decoder_strata and r.get("leakage_class") == "C"
            and pass_af3(r))))
    print("  Class C + AF3 pass + Tan_approved < {:.1f}:        {} (RELAXED set)".format(
        args.tanimoto_approved_threshold, len(relaxed)))
    print("  Class C + AF3 pass + Tan_approved < {:.1f} +".format(args.tanimoto_approved_threshold))
    print("    Tan_training < {:.1f}:                          {} (STRICT set)".format(
        args.tanimoto_training_threshold, len(strict)))

    if not relaxed and not strict:
        print("\n  No candidates pass any filter. Stopping.")
        return 0

    # --- Build proteome global matrix ---
    print("\n[3/5] Building proteome global matrix (one-time, ~1-2 min on GPU)...")
    reenc = Reencoder(args.molformer_name, args.encoder_checkpoint, device)
    proteome, protein_idxs = reenc.proteome_global_matrix(args.protein_dir)
    proteome = proteome.to(device)
    gene_map = load_gene_mapping(args.gene_mapping)
    target_to_pidxs = load_target_to_pidxs(args.binder_tsv)
    decoder_target_genes = set(target_to_pidxs.keys())

    # --- Re-encode and rank ---
    print("\n[4/5] Re-encoding survivors + computing proteome ranks...")
    union = list({r["sample_name"]: r for r in (relaxed + strict)}.values())
    smiles_list = [r.get("canonical_smiles") or r.get("drug_smiles") for r in union]
    drug_globals = reenc.smiles_to_drug_global_512(smiles_list).to(device)
    print("  Encoded {} unique survivor SMILES".format(len(smiles_list)))

    rank_out = {}
    for i, r in enumerate(union):
        intended_gene = r["target"]
        info = proteome_rank(
            drug_globals[i], proteome, gene_map,
            decoder_target_genes, intended_gene,
            top_k=args.top_k_off_target,
        )
        rank_out[r["sample_name"]] = info

    # --- Write outputs ---
    print("\n[5/5] Writing per-filter safety-screen outputs...")
    def write_filtered(rows, out_path, label):
        out_rows = []
        for r in rows:
            info = rank_out[r["sample_name"]]
            out_rows.append({
                "sample_name":               r["sample_name"],
                "target_intended":           r["target"],
                "stratum":                   r["stratum"],
                "canonical_smiles":          r.get("canonical_smiles") or r.get("drug_smiles"),
                "leakage_class":             r.get("leakage_class"),
                "max_tanimoto_to_training":  r.get("max_tanimoto_to_training"),
                "max_tanimoto_to_approved":  r.get("max_tanimoto_to_approved"),
                "iptm":                      r.get("iptm"),
                "interface_pae_min":         r.get("interface_pae_min"),
                "intended_target_rank":      info["intended_rank"],
                "intended_target_cosine":    "{:.4f}".format(info["intended_cos"]) if info["intended_cos"] is not None else "",
                "top10_off_target_genes":    ";".join(info["top_k_genes"][:10]),
                "top10_off_target_cosines":  ";".join("{:.4f}".format(c) for c in info["top_k_cosines"][:10]),
                "top10_off_target_ranks":    ";".join(str(rk) for rk in info["top_k_ranks"][:10]),
                "max_off_target_cosine":     "{:.4f}".format(max(info["top_k_cosines"]) if info["top_k_cosines"] else 0.0),
                "n_other_decoder_targets_in_top50": info["n_other_decoder_targets_in_top50"],
                "qed":                       r.get("qed"),
                "mw":                        r.get("mw"),
                "logp":                      r.get("logp"),
                "ro5_compliant":             r.get("ro5_compliant"),
            })
        with open(out_path, "w", newline="") as f:
            if out_rows:
                writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), delimiter="\t")
                writer.writeheader()
                for x in out_rows:
                    writer.writerow({k: ("" if v is None else v) for k, v in x.items()})
        print("  Wrote {} ({} rows) → {}".format(label, len(out_rows), out_path))

        # Per-target breakdown
        per_target = defaultdict(list)
        for x in out_rows:
            per_target[x["target_intended"]].append(x)
        print("    Per-target counts:")
        for t in sorted(per_target):
            n = len(per_target[t])
            ranks = [int(x["intended_target_rank"]) for x in per_target[t]
                     if x["intended_target_rank"] not in (None, "")]
            median_rank = sorted(ranks)[len(ranks) // 2] if ranks else None
            n_top50 = sum(1 for r_ in ranks if r_ <= 50)
            print("      {:<10} n={:>3}  median_intended_rank={:>5}  "
                  "n_intended_in_top50={:>3}".format(
                      t, n, median_rank, n_top50))
        return out_rows

    relaxed_rows = write_filtered(
        relaxed, args.output_dir / "F5_safety_screen_relaxed.tsv", "RELAXED"
    )
    strict_rows = write_filtered(
        strict, args.output_dir / "F5_safety_screen_strict.tsv", "STRICT"
    )

    # --- Console summary ---
    print("\n=== Summary ===")
    print("  RELAXED set (Class C + AF3 pass + Tan_approved < 0.4):  {} candidates".format(len(relaxed_rows)))
    print("  STRICT set (RELAXED + Tan_training < 0.4):              {} candidates".format(len(strict_rows)))
    if strict_rows:
        intended_top50_strict = sum(
            1 for x in strict_rows
            if x["intended_target_rank"] not in (None, "") and int(x["intended_target_rank"]) <= 50
        )
        clean_offtarget_strict = sum(
            1 for x in strict_rows if int(x["n_other_decoder_targets_in_top50"]) == 0
        )
        print("  STRICT — intended target in top-50 of 4,910 genes:      {} / {}".format(
            intended_top50_strict, len(strict_rows)))
        print("  STRICT — no other decoder-target gene in top-50:        {} / {}".format(
            clean_offtarget_strict, len(strict_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
