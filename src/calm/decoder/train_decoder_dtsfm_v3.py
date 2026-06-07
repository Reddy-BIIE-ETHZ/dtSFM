"""train_decoder_dtsfm_v3.py — Train the dtSFM v3 decoder (v0.1, CE-only).

v0.1 design (locked 2026-05-09 with Sai):
    - Pure cross-entropy on SMILES tokens (auxiliary cosine deferred to v0.2).
    - Teacher forcing: predict tokens[:, 1:] from tokens[:, :-1].
    - bf16 mixed precision; AdamW lr=2e-4 cosine to 1e-6, 5K-step warmup
      capped at half of max_steps for short smoketest runs.
    - Saves loss CSV per step + checkpoint at end.

Smoketest target: 4K random training pairs, 1K steps, ~10 min on a single GPU.
Verifies the pipeline learns at all (loss decreases, perplexity drops from
~vocab_size to ~tens). NOT a quality test — just plumbing + gradient flow.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from calm.decoder.model_dtsfm_v3 import CALMDecoderV3, default_decoder_v3_cfg
from calm.decoder.data_decoder_dtsfm_v3 import (
    DecoderDtsfmV3Dataset,
    collate_decoder_batch,
    load_molformer_tokenizer,
)


def warmup_cosine_lr(step: int, warmup_steps: int, max_steps: int,
                     base_lr: float, min_lr: float) -> float:
    """Linear warmup → cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main() -> int:
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--metadata_v3_csv", type=Path, required=True)
    ap.add_argument("--protein_dir", type=Path, required=True)
    ap.add_argument("--held_out_pairs_tsv", type=Path, default=None,
                    help="Optional TSV of held-out (drug_smiles, protein_id) pairs to exclude")
    ap.add_argument("--subsample", type=int, default=None,
                    help="Sample N pairs for smoketest; None = full set")
    ap.add_argument("--seed", type=int, default=42)
    # Model
    ap.add_argument("--molformer_name", default="ibm/MoLFormer-XL-both-10pct")
    ap.add_argument("--max_smiles_tokens", type=int, default=256)
    ap.add_argument("--max_protein_len", type=int, default=1024)
    ap.add_argument("--use_eos", action="store_true",
                    help="v0.2: add custom [EOS] token + train model to emit it")
    # Training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_steps", type=int, default=100,
                    help="Capped at max_steps//2 for short runs")
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--bf16", action="store_true",
                    help="Use bf16 autocast (default: True on cuda)")
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.set_defaults(bf16=True)
    # Logging / output
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=0,
                    help="Save intermediate checkpoint every N steps (0 = end only)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print(f"=== dtSFM v3 decoder training (v0.1, CE-only) ===")
    print(f"Device:  {device}")
    if device == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Output:  {args.output_dir}")

    # ---- Tokenizer ----
    print("\n[1/4] Loading MoLFormer-XL tokenizer...")
    tokenizer = load_molformer_tokenizer(args.molformer_name, add_eos=args.use_eos)
    # Use len(tokenizer) — includes any added special tokens (vocab_size attr does not).
    vocab_size = len(tokenizer)
    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
    print(f"    use_eos:       {args.use_eos}")
    print(f"    vocab_size:    {vocab_size:,}  (len(tokenizer))")
    print(f"    pad_token_id:  {pad_id}")
    print(f"    bos / eos:     {tokenizer.bos_token_id} / {eos_id}")

    # ---- Model ----
    print("\n[2/4] Building decoder model...")
    cfg = OmegaConf.create(default_decoder_v3_cfg(vocab_size, pad_id))
    cfg.max_smiles_tokens = args.max_smiles_tokens
    model = CALMDecoderV3(cfg).to(device)
    pc = model.count_parameters()
    print(f"    Trainable params:  {pc['total']:,}")
    for k, v in pc.items():
        if k != "total":
            print(f"      {k:<20} {v:>12,}")

    # ---- Data ----
    print("\n[3/4] Building dataset + dataloader...")
    dataset = DecoderDtsfmV3Dataset(
        metadata_csv=args.metadata_v3_csv,
        protein_dir=args.protein_dir,
        tokenizer=tokenizer,
        max_smiles_tokens=args.max_smiles_tokens,
        max_protein_len=args.max_protein_len,
        held_out_pairs_tsv=args.held_out_pairs_tsv,
        subsample=args.subsample,
        seed=args.seed,
        append_eos=args.use_eos,
    )
    print(f"    Dataset size:  {len(dataset):,}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_decoder_batch(b, pad_id),
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    steps_per_epoch = len(loader)
    print(f"    Batches/epoch: {steps_per_epoch}  (batch_size={args.batch_size})")

    # ---- Optimizer ----
    print("\n[4/4] Setting up optimizer + training loop...")
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    warmup = min(args.warmup_steps, max(1, args.max_steps // 2))
    print(f"    AdamW lr={args.lr} → {args.min_lr} (cosine, warmup={warmup})")
    print(f"    bf16 autocast: {args.bf16 and device == 'cuda'}")
    print(f"    max_steps:     {args.max_steps}")

    # ---- Train loop ----
    log_path = args.output_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "perplexity", "lr", "tok_per_s",
                         "elapsed_s"])

    model.train()
    step = 0
    t0 = time.time()
    smoothed_loss = None
    print("\n=== training begins ===")
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            # Move to device
            input_ids   = batch["input_ids"].to(device, non_blocking=True)
            attn_mask   = batch["attention_mask"].to(device, non_blocking=True)
            protein_emb = batch["protein_emb"].to(device, non_blocking=True).float()
            protein_mask = batch["protein_mask"].to(device, non_blocking=True)

            # LR schedule
            lr = warmup_cosine_lr(step, warmup, args.max_steps,
                                  args.lr, args.min_lr)
            for g in optim.param_groups:
                g["lr"] = lr

            # Forward + loss
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if (args.bf16 and device == "cuda")
                else torch.amp.autocast(device_type=device, enabled=False)
            )
            with ctx:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    protein_emb=protein_emb,
                    protein_mask=protein_mask,
                )
                # Teacher-forced next-token: predict tokens[1:] from tokens[:-1]
                logits  = out["logits"][:, :-1, :]
                targets = input_ids[:, 1:]
                # Mask: only score positions where target is a real (non-pad) token
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    targets.reshape(-1),
                    ignore_index=pad_id,
                    reduction="mean",
                )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            loss_val = loss.item()
            ppl = math.exp(min(loss_val, 30.0))   # cap for safety
            smoothed_loss = (
                loss_val if smoothed_loss is None
                else 0.98 * smoothed_loss + 0.02 * loss_val
            )

            elapsed = time.time() - t0
            n_real_tokens = int(attn_mask.sum().item())
            tok_per_s = n_real_tokens * (step + 1) / max(elapsed, 1e-6)

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, f"{loss_val:.4f}", f"{ppl:.2f}",
                                 f"{lr:.3e}", f"{tok_per_s:.0f}",
                                 f"{elapsed:.1f}"])

            if step % args.log_every == 0 or step == args.max_steps - 1:
                print(f"  step {step:5d}/{args.max_steps}  "
                      f"loss {loss_val:.3f}  smoothed {smoothed_loss:.3f}  "
                      f"PPL {ppl:8.1f}  lr {lr:.2e}  "
                      f"{tok_per_s/1000:.1f} ktok/s  "
                      f"elapsed {elapsed:.0f}s",
                      flush=True)

            # Periodic checkpoint
            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                periodic_path = args.output_dir / f"decoder_v3_step_{step:07d}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config":           OmegaConf.to_container(cfg),
                    "tokenizer_name":   args.molformer_name,
                    "step":             step,
                    "smoothed_loss":    smoothed_loss,
                }, periodic_path)
                print(f"    [checkpoint] saved → {periodic_path.name}", flush=True)

            step += 1

    # ---- Save checkpoint ----
    ckpt_path = args.output_dir / "decoder_v3_smoketest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config":           OmegaConf.to_container(cfg),
        "tokenizer_name":   args.molformer_name,
        "step":             step,
        "final_loss":       smoothed_loss,
        "training_args":    vars(args),
    }, ckpt_path)
    print(f"\n  Saved checkpoint → {ckpt_path}")
    print(f"  Loss CSV → {log_path}")

    # ---- Summary block (regex-friendly per audit_discipline.md Rule 2) ----
    print("\n=== Summary ===")
    print(f"  total_steps:       {step}")
    print(f"  final_loss:        {smoothed_loss:.4f}")
    final_ppl = math.exp(min(smoothed_loss, 30.0))
    print(f"  final_perplexity:  {final_ppl:.2f}")
    print(f"  vocab_size:        {vocab_size}")
    print(f"  random_PPL:        {vocab_size:.0f}")
    print(f"  total_walltime_s:  {time.time() - t0:.0f}")
    print(f"  param_count:       {pc['total']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
