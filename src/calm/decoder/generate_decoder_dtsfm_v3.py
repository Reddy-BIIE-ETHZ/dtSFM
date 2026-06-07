"""generate_decoder_dtsfm_v3.py — Eyeball test for the dtSFM v3 decoder.

Loads a (smoketest or full) decoder checkpoint, samples N SMILES per target
via temperature sampling, parses each with RDKit, and prints validity rate +
canonical-SMILES output.

Termination: MoLFormer tokenizer has no <eos> token, so we sample for
fixed --max_new_tokens steps and rely on RDKit parse to determine which
prefixes form valid molecules. (v0.2 will add a custom <eos> token.)

Usage (Euler GPU job):
    python3 -u -m calm.decoder.generate_decoder_dtsfm_v3 \\
        --checkpoint    /cluster/scratch/.../decoder_v3_smoketest.pt \\
        --binder_tsv    audit/dtsfm/decoder_target_binders.tsv \\
        --protein_dir   /cluster/scratch/.../embeddings/protein_embeds/ \\
        --target_genes  BTK EGFR CDK4 \\
        --n_samples     20 \\
        --temperature   0.8 \\
        --max_new_tokens 100 \\
        --output_tsv    audit/dtsfm/decoder_phase1_eyeball.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from calm.decoder.model_dtsfm_v3 import CALMDecoderV3
from calm.decoder.data_decoder_dtsfm_v3 import load_molformer_tokenizer


def rdkit_validate(smiles: str) -> tuple[bool, str | None]:
    """Returns (is_valid, canonical_smiles_or_None)."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return False, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, None
    try:
        canon = Chem.MolToSmiles(mol)
        if mol.GetNumHeavyAtoms() < 3:
            return False, canon  # Trivial / single-atom ions
        return True, canon
    except Exception:
        return False, None


@torch.no_grad()
def sample_smiles(
    model: CALMDecoderV3,
    tokenizer,
    protein_emb: torch.Tensor,         # (L, 1280) on device
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    pad_token_id: int,
) -> list[str]:
    """Generate n_samples SMILES strings via temperature sampling.

    Strategy: start each sample with a random non-pad token from the vocab
    (proxy for <bos> since MoLFormer has none), then autoregressively sample
    until max_new_tokens. Decode each sequence by stripping pad tokens.
    """
    model.eval()
    L = protein_emb.shape[0]
    # Replicate protein for batched sampling
    protein_batch = protein_emb.unsqueeze(0).expand(n_samples, L, -1).contiguous()
    protein_mask = torch.ones(n_samples, L, dtype=torch.bool, device=device)

    # Seed with a small set of plausible SMILES start tokens (alphabetic / structural)
    # — we pick at random to diversify the samples.
    vocab = tokenizer.get_vocab()
    seed_candidates = [
        vocab[t] for t in ("C", "c", "N", "n", "O", "S", "F", "Cl", "Br", "P")
        if t in vocab
    ]
    if not seed_candidates:
        # Fallback: any common-element token
        seed_candidates = [i for i in range(min(50, len(vocab))) if i != pad_token_id]
    seed_ids = torch.tensor(
        [seed_candidates[i % len(seed_candidates)] for i in range(n_samples)],
        dtype=torch.long, device=device,
    ).unsqueeze(1)                                                       # (N, 1)

    # Detect EOS handling — if the tokenizer has an EOS token (v0.2), we
    # track per-sample completion and stop emitting once a sample has hit EOS.
    eos_token_id = (
        int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
    )
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)

    cur = seed_ids
    for _ in range(max_new_tokens - 1):
        attn = torch.ones_like(cur, dtype=torch.bool)
        out = model(
            input_ids=cur,
            attention_mask=attn,
            protein_emb=protein_batch,
            protein_mask=protein_mask,
        )
        logits = out["logits"][:, -1, :].float() / max(temperature, 1e-6)
        logits[:, pad_token_id] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)                # (N, 1)

        # If a sample has finished (already emitted EOS), keep emitting pad
        # so it's harmless when we strip later
        if eos_token_id is not None and finished.any():
            next_tok[finished] = pad_token_id
        cur = torch.cat([cur, next_tok], dim=1)

        # Mark samples that just emitted EOS as finished
        if eos_token_id is not None:
            finished = finished | (next_tok.squeeze(1) == eos_token_id)
            if finished.all():
                break

    # Decode each row. If EOS exists, truncate at first EOS occurrence.
    out_strs: list[str] = []
    for row in cur:
        ids = row.tolist()
        if eos_token_id is not None and eos_token_id in ids:
            ids = ids[: ids.index(eos_token_id)]
        s = tokenizer.decode(ids, skip_special_tokens=True).strip()
        out_strs.append(s)
    return out_strs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--binder_tsv", type=Path, required=True,
                    help="Pre-resolved decoder_target_binders.tsv (for gene→protein_idx)")
    ap.add_argument("--protein_dir", type=Path, required=True)
    ap.add_argument("--target_genes", nargs="+", required=True,
                    help="Gene symbols to generate against, e.g. BTK EGFR CDK4")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--molformer_name", default="ibm/MoLFormer-XL-both-10pct")
    ap.add_argument("--output_tsv", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print(f"=== dtSFM v3 decoder — Phase 1 eyeball generation test ===")
    print(f"Device: {device}")

    # ---- Load checkpoint + model ----
    print(f"\n[1/4] Loading checkpoint {args.checkpoint.name}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = CALMDecoderV3(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"    Step at save: {ckpt.get('step', '?')}")
    print(f"    Final loss:   {ckpt.get('final_loss', '?'):.4f}" if isinstance(ckpt.get('final_loss'), float) else "")
    print(f"    Param count:  {sum(p.numel() for p in model.parameters()):,}")

    # ---- Tokenizer ----
    print(f"\n[2/4] Loading tokenizer ({args.molformer_name})...")
    tokenizer = load_molformer_tokenizer(args.molformer_name)
    pad_id = int(tokenizer.pad_token_id)
    print(f"    vocab_size: {tokenizer.vocab_size}    pad: {pad_id}")

    # ---- Resolve target_genes → protein_idx via binder TSV ----
    print(f"\n[3/4] Resolving target genes → protein_idx...")
    binders = pd.read_csv(args.binder_tsv, sep="\t")
    gene_to_pidx: dict[str, int] = {}
    for gene in args.target_genes:
        rows = binders[binders["target_gene"] == gene]
        if len(rows) == 0:
            print(f"    WARN: {gene} not in binder TSV — skipping")
            continue
        pidx = int(rows.iloc[0]["target_pidx_canonical"])
        gene_to_pidx[gene] = pidx
        print(f"    {gene:<10} → protein_idx {pidx}")

    # ---- Generate per target ----
    print(f"\n[4/4] Generating {args.n_samples} samples per target (T={args.temperature}, max_new={args.max_new_tokens})...")
    all_rows = []
    for gene, pidx in gene_to_pidx.items():
        prot = torch.load(args.protein_dir / f"{pidx:06d}.pt",
                          map_location=device, weights_only=True).float()
        if prot.shape[0] > 1024:
            prot = prot[:1024]
        print(f"\n  --- {gene} (protein_idx {pidx}, L_res={prot.shape[0]}) ---")

        smiles_list = sample_smiles(
            model=model, tokenizer=tokenizer, protein_emb=prot,
            n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, device=device, pad_token_id=pad_id,
        )

        n_valid = 0
        for i, smi in enumerate(smiles_list):
            valid, canon = rdkit_validate(smi)
            if valid:
                n_valid += 1
            tag = "VALID  " if valid else "invalid"
            print(f"    {tag} [{i+1:>2}] {smi[:90]}{'...' if len(smi) > 90 else ''}")
            if valid and canon:
                print(f"             canon: {canon}")
            all_rows.append({
                "target_gene": gene,
                "target_pidx": pidx,
                "sample_idx":  i,
                "raw_smiles":  smi,
                "valid":       valid,
                "canonical":   canon,
            })
        rate = 100.0 * n_valid / max(len(smiles_list), 1)
        print(f"    {gene}: {n_valid}/{len(smiles_list)} valid ({rate:.0f}%)")

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.output_tsv, sep="\t", index=False)
    print(f"\n  Wrote per-sample TSV → {args.output_tsv}")

    # ---- Summary block ----
    print("\n=== Summary ===")
    df = pd.DataFrame(all_rows)
    overall = df["valid"].sum()
    print(f"  total_samples:    {len(df)}")
    print(f"  total_valid:      {overall}")
    print(f"  overall_validity: {100.0 * overall / max(len(df), 1):.1f}%")
    by_gene = df.groupby("target_gene")["valid"].agg(["sum", "size"])
    by_gene["pct"] = 100.0 * by_gene["sum"] / by_gene["size"]
    for gene, r in by_gene.iterrows():
        print(f"  {gene:<10} valid_rate {int(r['sum'])}/{int(r['size'])} = {r['pct']:.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
