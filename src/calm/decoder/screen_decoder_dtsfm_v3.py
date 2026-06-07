"""screen_decoder_dtsfm_v3.py — generative-design ranking pipeline.

For each decoder target, generate N candidates, drop invalid + duplicates,
compute drug-likeness, re-encode through frozen MoLFormer + v3 encoder for
ranking, and output the top-K.

Workflow per target (e.g. BTK):

    1. Load target's protein embedding (cached ESM-2 per-residue fp16)
    2. Sample N candidates from the decoder via temperature sampling
    3. RDKit-parse each → keep valid + non-trivial → canonicalize → dedupe
    4. Compute drug-likeness for each unique candidate:
         - MW (Lipinski: must be < 500)
         - LogP (Lipinski: must be < 5)
         - HBD (Lipinski: ≤ 5 H-bond donors)
         - HBA (Lipinski: ≤ 10 H-bond acceptors)
         - QED (Bickerton 2012, blended drug-likeness, 0-1)
         - n_lipinski_violations (0 = full compliance)
    5. Re-tokenize each canonical SMILES via the SAME MoLFormer tokenizer the
       v3 encoder was trained on, run MoLFormer to get a 768-d mean-pooled
       drug-global embedding, project through the encoder's drug_global_proj
       to get the 512-d L2-normalized global feature.
    6. Cosine similarity vs the target's 512-d global protein feature.
    7. Rerank by cosine, keep top-K (default 100).
    8. Write per-target TSV with columns:
         target_gene, target_pidx, sample_idx_at_generation,
         canonical_smiles, mw, logp, hbd, hba, qed,
         n_lipinski_violations, ro5_compliant,
         encoder_cosine, rank_by_cosine

Reads the same binder TSV the decoder smoketest used (resolved from PubChem
on 2026-05-08) to map gene_symbol → protein_idx.

Wall: ~30 min on a single GPU for 10 targets × N=1000 samples.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from calm.decoder.model_dtsfm_v3 import CALMDecoderV3
from calm.decoder.data_decoder_dtsfm_v3 import load_molformer_tokenizer
from calm.encoder.model_v3 import CALMEncoderV3
from calm.encoder.train_dtsfm_v3 import default_model_cfg as default_encoder_cfg
from calm.decoder.generate_decoder_dtsfm_v3 import sample_smiles


# --------------------------------------------------------------------------- #
# RDKit / drug-likeness
# --------------------------------------------------------------------------- #
def compute_druglikeness(smiles: str) -> dict | None:
    """Return drug-likeness metrics for a SMILES string, or None on parse fail.

    Lipinski Ro5 components:
      MW < 500, LogP < 5, HBD ≤ 5, HBA ≤ 10
    QED (Bickerton et al. 2012): RDKit's blended drug-likeness score in [0, 1].
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() < 3:
        return None
    try:
        canon = Chem.MolToSmiles(mol)
        mw = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        qed = float(QED.qed(mol))
        violations = sum([mw >= 500, logp >= 5, hbd > 5, hba > 10])
        return {
            "canonical_smiles":       canon,
            "mw":                     round(mw, 2),
            "logp":                   round(logp, 3),
            "hbd":                    hbd,
            "hba":                    hba,
            "qed":                    round(qed, 4),
            "n_lipinski_violations":  violations,
            "ro5_compliant":          (violations == 0),
        }
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Re-encode a list of SMILES through MoLFormer → v3 encoder → 512-d global
# --------------------------------------------------------------------------- #
class ReencodingPipeline:
    """Holds frozen MoLFormer + v3 encoder; converts SMILES → 512-d global features.

    SMILES → MoLFormer tokenizer → MoLFormer (frozen) → mean-pool → 768-d
           → encoder.drug_global_proj → L2-normalize → 512-d
    """

    def __init__(
        self,
        molformer_name: str,
        encoder_checkpoint: Path,
        device: str,
    ):
        self.device = device
        # MoLFormer (frozen)
        from transformers import AutoModel, AutoTokenizer
        print(f"  Loading MoLFormer model ({molformer_name})...")
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
        ).to(device)
        self.molformer.eval()
        for p in self.molformer.parameters():
            p.requires_grad_(False)

        # v3 encoder (we only need drug_global_proj and protein side, but
        # loading the full model is simpler than slicing the state_dict)
        print(f"  Loading v3 encoder from {encoder_checkpoint.name}...")
        ckpt = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
        ecfg = OmegaConf.create(ckpt.get("config") or default_encoder_cfg())
        self.encoder = CALMEncoderV3(ecfg).to(device)
        self.encoder.load_state_dict(ckpt["model_state_dict"])
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def smiles_to_drug_global_768(self, smiles_list: list[str], batch_size: int = 64) -> torch.Tensor:
        """SMILES → MoLFormer tokenize → MoLFormer model → mean-pool → (N, 768)."""
        out_chunks: list[torch.Tensor] = []
        for start in range(0, len(smiles_list), batch_size):
            chunk = smiles_list[start:start + batch_size]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.molformer(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            hidden = outputs.last_hidden_state                          # (B, T, 768)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            out_chunks.append(pooled.float().cpu())
        return torch.cat(out_chunks, dim=0)                             # (N, 768)

    @torch.no_grad()
    def drug_global_768_to_512(self, drug_global_768: torch.Tensor) -> torch.Tensor:
        """768-d MoLFormer global → 512-d encoder global, L2-normalized."""
        x = drug_global_768.to(self.device)
        h = self.encoder.drug_global_proj(x)
        return F.normalize(h, dim=-1)                                   # (N, 512)

    @torch.no_grad()
    def target_global_512(self, protein_emb: torch.Tensor) -> torch.Tensor:
        """Cached ESM-2 (L, 1280) → encoder protein_proj + AttnPool → (512,) L2-normalized."""
        x = protein_emb.float().unsqueeze(0).to(self.device)
        L = x.shape[1]
        mask = torch.ones(1, L, dtype=torch.bool, device=self.device)
        h = self.encoder.protein_proj(x) * mask.unsqueeze(-1).float()
        pooled = self.encoder.protein_pool_pre(h, mask)
        return F.normalize(pooled, dim=-1).squeeze(0)                   # (512,)


# --------------------------------------------------------------------------- #
# Per-target screening
# --------------------------------------------------------------------------- #
def screen_one_target(
    *,
    decoder: CALMDecoderV3,
    decoder_tokenizer,
    pipeline: ReencodingPipeline,
    target_gene: str,
    target_pidx: int,
    target_protein_emb: torch.Tensor,
    n_samples: int,
    sample_batch: int,
    temperature: float,
    max_new_tokens: int,
    top_k: int,
    device: str,
) -> pd.DataFrame:
    """End-to-end screen for a single target. Returns a DataFrame with top_k rows."""
    print(f"\n  ==== {target_gene} (protein_idx {target_pidx}, "
          f"L_res={target_protein_emb.shape[0]}) ====")
    t0 = time.time()

    # ---- 1. Sample N candidates from decoder ----
    print(f"    [1/5] Generating {n_samples} candidates "
          f"(T={temperature}, max_new={max_new_tokens}, batch={sample_batch})...")
    raw_smiles: list[str] = []
    for batch_start in range(0, n_samples, sample_batch):
        b = min(sample_batch, n_samples - batch_start)
        chunk = sample_smiles(
            model=decoder, tokenizer=decoder_tokenizer,
            protein_emb=target_protein_emb.to(device),
            n_samples=b, max_new_tokens=max_new_tokens,
            temperature=temperature, device=device,
            pad_token_id=int(decoder_tokenizer.pad_token_id),
        )
        raw_smiles.extend(chunk)
    print(f"          generated {len(raw_smiles)} raw strings in {time.time()-t0:.1f}s")

    # ---- 2. RDKit validate + canonicalize + dedupe ----
    print(f"    [2/5] RDKit validate + canonicalize + dedupe...")
    seen: dict[str, dict] = {}
    n_valid_raw = 0
    for i, smi in enumerate(raw_smiles):
        d = compute_druglikeness(smi)
        if d is None:
            continue
        n_valid_raw += 1
        canon = d["canonical_smiles"]
        if canon in seen:
            continue
        d["sample_idx_at_generation"] = i
        seen[canon] = d
    n_unique = len(seen)
    print(f"          valid raw: {n_valid_raw} / {len(raw_smiles)} "
          f"({100.0 * n_valid_raw / max(len(raw_smiles), 1):.1f}%); "
          f"unique canonical: {n_unique}")
    if n_unique == 0:
        # Return empty frame with the right schema so downstream still works
        return pd.DataFrame(columns=[
            "target_gene", "target_pidx", "sample_idx_at_generation",
            "canonical_smiles", "mw", "logp", "hbd", "hba", "qed",
            "n_lipinski_violations", "ro5_compliant",
            "encoder_cosine", "rank_by_cosine",
        ])

    # ---- 3. Drug-likeness already computed in step 2 ----
    print(f"    [3/5] Drug-likeness already computed inline.")

    # ---- 4. Re-encode each candidate through MoLFormer + encoder ----
    print(f"    [4/5] Re-encoding {n_unique} unique candidates "
          f"(MoLFormer → encoder.drug_global_proj)...")
    t1 = time.time()
    canon_list = list(seen.keys())
    drug_768 = pipeline.smiles_to_drug_global_768(canon_list, batch_size=64)
    drug_512 = pipeline.drug_global_768_to_512(drug_768)                # (n_unique, 512)
    target_512 = pipeline.target_global_512(target_protein_emb)         # (512,)
    cosines = (drug_512 @ target_512).cpu().numpy()                     # (n_unique,)
    print(f"          re-encoded in {time.time()-t1:.1f}s")

    # ---- 5. Rank + assemble + take top-K ----
    print(f"    [5/5] Ranking by encoder cosine, keeping top-{top_k}...")
    rows = []
    for canon, cos in zip(canon_list, cosines):
        d = seen[canon].copy()
        d["target_gene"] = target_gene
        d["target_pidx"] = target_pidx
        d["encoder_cosine"] = float(cos)
        rows.append(d)
    df = pd.DataFrame(rows).sort_values("encoder_cosine", ascending=False).reset_index(drop=True)
    df["rank_by_cosine"] = df.index + 1
    df = df.head(top_k)

    # Reorder columns
    cols = ["target_gene", "target_pidx", "sample_idx_at_generation",
            "canonical_smiles", "mw", "logp", "hbd", "hba", "qed",
            "n_lipinski_violations", "ro5_compliant",
            "encoder_cosine", "rank_by_cosine"]
    df = df[cols]

    print(f"    Top-{len(df)} retained. "
          f"Top cosine: {df['encoder_cosine'].iloc[0]:+.4f}; "
          f"median cosine of top-K: {df['encoder_cosine'].median():+.4f}; "
          f"Ro5 compliant: {int(df['ro5_compliant'].sum())} / {len(df)}; "
          f"mean QED: {df['qed'].mean():.3f}")
    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder_checkpoint", type=Path, required=True)
    ap.add_argument("--encoder_checkpoint", type=Path, required=True)
    ap.add_argument("--molformer_name", default="ibm/MoLFormer-XL-both-10pct")
    ap.add_argument("--binder_tsv", type=Path, required=True)
    ap.add_argument("--protein_dir", type=Path, required=True)
    ap.add_argument("--target_genes", nargs="+", default=None,
                    help="Subset of gene symbols to screen; default = all unique in binder_tsv")
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--sample_batch", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_eos", action="store_true",
                    help="v0.2: load tokenizer with [EOS]; sampler stops on EOS. "
                         "Must match how the decoder was trained.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== dtSFM v3 decoder Phase 3 — generation + rerank deliverable ===")
    print(f"Device:  {device}")
    if device == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Output:  {args.output_dir}")

    # ---- Load decoder ----
    print(f"\n[1/4] Loading decoder checkpoint ({args.decoder_checkpoint.name})...")
    dckpt = torch.load(args.decoder_checkpoint, map_location=device, weights_only=False)
    dcfg = OmegaConf.create(dckpt["config"])
    decoder = CALMDecoderV3(dcfg).to(device)
    decoder.load_state_dict(dckpt["model_state_dict"])
    decoder.eval()
    decoder_tokenizer = load_molformer_tokenizer(args.molformer_name, add_eos=args.use_eos)
    # Sanity-check: tokenizer vocab matches checkpoint
    if len(decoder_tokenizer) != int(dcfg.vocab_size):
        print(f"    WARN: len(tokenizer)={len(decoder_tokenizer)} but checkpoint "
              f"vocab_size={int(dcfg.vocab_size)}. Did you forget --use_eos?")
    print(f"    Decoder step at save: {dckpt.get('step', '?')}")
    print(f"    Decoder param count:  {sum(p.numel() for p in decoder.parameters()):,}")

    # ---- Load reencoding pipeline (MoLFormer + v3 encoder) ----
    print(f"\n[2/4] Loading reencoding pipeline (MoLFormer + v3 encoder)...")
    pipeline = ReencodingPipeline(
        molformer_name=args.molformer_name,
        encoder_checkpoint=args.encoder_checkpoint,
        device=device,
    )

    # ---- Resolve targets ----
    print(f"\n[3/4] Resolving target genes...")
    binders = pd.read_csv(args.binder_tsv, sep="\t")
    if args.target_genes is None:
        # Default: all unique genes with target_pidx_canonical present
        target_rows = (
            binders.dropna(subset=["target_pidx_canonical"])
                   .drop_duplicates("target_gene")
                   [["target_gene", "target_pidx_canonical"]]
        )
    else:
        target_rows = (
            binders[binders["target_gene"].isin(args.target_genes)]
                   .dropna(subset=["target_pidx_canonical"])
                   .drop_duplicates("target_gene")
                   [["target_gene", "target_pidx_canonical"]]
        )
    target_rows = target_rows.reset_index(drop=True)
    print(f"    Targets to screen: {len(target_rows)}")
    for _, r in target_rows.iterrows():
        print(f"      {r['target_gene']:<10}  protein_idx {int(r['target_pidx_canonical'])}")

    # ---- Loop over targets ----
    print(f"\n[4/4] Screening {len(target_rows)} targets × N={args.n_samples} candidates each...")
    all_top: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for _, r in target_rows.iterrows():
        gene = str(r["target_gene"])
        pidx = int(r["target_pidx_canonical"])
        prot = torch.load(args.protein_dir / f"{pidx:06d}.pt",
                          map_location="cpu", weights_only=True).float()
        if prot.shape[0] > 1024:
            prot = prot[:1024]

        df_top = screen_one_target(
            decoder=decoder, decoder_tokenizer=decoder_tokenizer,
            pipeline=pipeline,
            target_gene=gene, target_pidx=pidx, target_protein_emb=prot,
            n_samples=args.n_samples, sample_batch=args.sample_batch,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            top_k=args.top_k, device=device,
        )
        per_target_path = args.output_dir / f"phase3_{gene}_top{args.top_k}.tsv"
        df_top.to_csv(per_target_path, sep="\t", index=False)
        print(f"    Wrote {per_target_path}")
        all_top.append(df_top)
        if len(df_top) > 0:
            summary_rows.append({
                "target_gene":     gene,
                "target_pidx":     pidx,
                "n_top_retained":  len(df_top),
                "top1_cosine":     float(df_top["encoder_cosine"].iloc[0]),
                "median_cosine":   float(df_top["encoder_cosine"].median()),
                "mean_qed":        float(df_top["qed"].mean()),
                "ro5_compliant":   int(df_top["ro5_compliant"].sum()),
                "mean_mw":         float(df_top["mw"].mean()),
                "mean_logp":       float(df_top["logp"].mean()),
            })
        else:
            summary_rows.append({
                "target_gene": gene, "target_pidx": pidx,
                "n_top_retained": 0, "top1_cosine": None,
                "median_cosine": None, "mean_qed": None,
                "ro5_compliant": 0, "mean_mw": None, "mean_logp": None,
            })

    # ---- Master summary ----
    summary_df = pd.DataFrame(summary_rows).sort_values("target_gene")
    summary_path = args.output_dir / "phase3_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\n  Wrote master summary → {summary_path}")

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nTotal targets screened: {len(summary_df)}")
    print(f"Targets with ≥1 top retained: {int((summary_df['n_top_retained'] > 0).sum())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
