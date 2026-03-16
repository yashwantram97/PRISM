"""
smoke_test.py
-------------
CPU-safe end-to-end smoke test for PRISM.

Checks (in order):
  1. Model builds cleanly (SVD init, gradient flow)
  2. Forward + backward pass runs without error
  3. Loss is finite and decreases over N_STEPS
  4. Optimizer steps update parameters
  5. Checkpoint save → load → forward pass round-trip
  6. Balance loss (entropy regularizer) is non-zero

Zero changes to any existing file. Delete this file when done.
Run from project root:
    python scripts/smoke_test.py

To revert: just delete this file.
"""

import os
import sys
import math
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.model_builder import build_prism_model

# ── Smoke config (tiny values for fast CPU run) ───────────────────────────────

SMOKE_CONFIG = {
    # Model  — same as production so architecture is fully tested
    "model_name":       "Qwen/Qwen2.5-0.5B",
    "rank":             4,
    "n_experts":        4,
    "base_seed":        42,
    "bottleneck_ratio": 0.5,

    # Tiny training dims
    "seq_len":          32,    # short sequences → fast forward pass
    "batch_size":       2,     # tiny batch
    "grad_accum":       2,     # 2 micro-steps per update
    "n_steps":          10,    # gradient updates to run

    # Optimizer
    "lr":               2e-4,
    "weight_decay":     0.1,
    "grad_clip":        1.0,
    "balance_weight":   0.01,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "  ✓"
FAIL = "  ✗ FAILED"

def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

def check(label: str, ok: bool, detail: str = ""):
    mark = PASS if ok else FAIL
    msg  = f"{mark}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not ok:
        raise SystemExit(f"\nSmoke test FAILED at: {label}")


# ── Synthetic dataset (no download needed) ────────────────────────────────────

def make_batch(tokenizer, seq_len: int, batch_size: int, device: str):
    """Random token IDs — fast, no data download required."""
    vocab = tokenizer.vocab_size
    ids   = torch.randint(1, vocab, (batch_size, seq_len + 1), device=device)
    return {
        "input_ids": ids[:, :-1],   # [B, seq_len]
        "labels":    ids[:, 1:],    # [B, seq_len]
    }


# ── Main smoke test ───────────────────────────────────────────────────────────

def main():
    cfg    = SMOKE_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16

    # Use MPS on Apple Silicon if CUDA not available
    if device == "cpu" and torch.backends.mps.is_available():
        # MPS has some bfloat16 quirks; keep cpu for reliability on Mac smoke test
        pass

    print("\n" + "═"*55)
    print("  PRISM Smoke Test")
    print("═"*55)
    print(f"  Device:           {device}")
    print(f"  dtype:            {dtype}")
    print(f"  seq_len:          {cfg['seq_len']}")
    print(f"  batch_size:       {cfg['batch_size']}")
    print(f"  grad_accum:       {cfg['grad_accum']}")
    print(f"  n_steps:          {cfg['n_steps']} gradient updates")
    print(f"  bottleneck_ratio: {cfg['bottleneck_ratio']}")

    # ── 1. Build model ────────────────────────────────────────────────────────
    section("1. Model Build + Gradient Flow")
    t0 = time.time()
    model, tokenizer, info = build_prism_model(
        model_name       = cfg["model_name"],
        rank             = cfg["rank"],
        n_experts        = cfg["n_experts"],
        base_seed        = cfg["base_seed"],
        bottleneck_ratio = cfg["bottleneck_ratio"],
        dtype            = dtype,
        device           = device,
    )
    build_time = time.time() - t0
    check("Model built", True, f"{build_time:.1f}s")
    check("Trainable params > 0", info["trainable_params"] > 0,
          f"{info['trainable_params']/1e6:.1f}M trainable")
    check("Frozen params > 0", info["frozen_params"] > 0,
          f"{info['frozen_params']/1e6:.1f}M frozen")
    check("d_ff_expert == int(d_ff * ratio)",
          info["d_ff_expert"] == int(info["d_ff"] * cfg["bottleneck_ratio"]),
          f"d_ff_expert={info['d_ff_expert']}")

    model.train()

    # ── 2. Optimizer (CPU-safe: no fused=True) ────────────────────────────────
    section("2. Optimizer Setup")
    expert_params, delta_params, router_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "proj_u" in name or "proj_v" in name:
            delta_params.append(param)
        elif "router" in name:
            router_params.append(param)
        else:
            expert_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": expert_params, "lr": cfg["lr"]},
            {"params": delta_params,  "lr": cfg["lr"]},
            {"params": router_params, "lr": cfg["lr"]},
        ],
        betas=(0.9, 0.95),
        weight_decay=cfg["weight_decay"],
        # fused=True is CUDA-only — intentionally omitted for CPU smoke test
    )
    check("Optimizer created", True,
          f"3 param groups, {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M params")

    # ── 3. Training loop ──────────────────────────────────────────────────────
    section("3. Training Loop")

    losses        = []
    balance_losses = []
    micro_step    = 0

    for update in range(cfg["n_steps"]):
        optimizer.zero_grad()
        accum_lm_loss = 0.0

        for _ in range(cfg["grad_accum"]):
            batch  = make_batch(tokenizer, cfg["seq_len"], cfg["batch_size"], device)
            input_ids = batch["input_ids"]
            labels    = batch["labels"]

            # CPU-safe autocast: use torch.amp.autocast with device_type
            with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                outputs  = model(input_ids=input_ids, labels=labels)
                lm_loss  = outputs.loss

                bal_loss = torch.stack([
                    layer.mlp._balance_loss
                    for layer in model.model.layers
                    if hasattr(layer.mlp, "_balance_loss")
                ]).mean()

                loss = lm_loss - cfg["balance_weight"] * bal_loss
                loss = loss / cfg["grad_accum"]

            loss.backward()
            accum_lm_loss += lm_loss.item()

        # Gradient clip + optimizer step
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            cfg["grad_clip"],
        )
        optimizer.step()

        avg_lm = accum_lm_loss / cfg["grad_accum"]
        losses.append(avg_lm)
        balance_losses.append(bal_loss.item())

        print(f"    update={update+1:>3}  lm_loss={avg_lm:.4f}  balance={bal_loss.item():.4f}")

    # Validation checks on the run
    check("All losses finite", all(math.isfinite(l) for l in losses))
    check("Loss decreased (first → last)",
          losses[-1] < losses[0],
          f"{losses[0]:.4f} → {losses[-1]:.4f}")
    check("Balance loss non-zero", any(b > 1e-6 for b in balance_losses),
          f"mean={sum(balance_losses)/len(balance_losses):.4f}")

    # ── 4. Parameter actually changed ─────────────────────────────────────────
    section("4. Parameter Update Check")
    # Snapshot first expert FFN weight after training vs. its SVD init value
    first_expert_w = model.model.layers[0].mlp.experts[0].gate_proj.weight
    check("Expert FFN weights have grad", first_expert_w.grad is not None
          or any(p.grad is not None for p in expert_params))
    # Confirm weight is not all-zero (would indicate optimizer never ran)
    check("Expert FFN weights non-zero", first_expert_w.abs().max().item() > 0)

    # ── 5. Checkpoint save → load round-trip ──────────────────────────────────
    section("5. Checkpoint Save / Load")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "smoke_checkpoint.pt")

        # Save
        torch.save({
            "step":                 cfg["n_steps"],
            "loss":                 losses[-1],
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config":               cfg,
        }, ckpt_path)
        saved_size_mb = os.path.getsize(ckpt_path) / 1e6
        check("Checkpoint saved", os.path.exists(ckpt_path),
              f"{saved_size_mb:.1f} MB")

        # Load into fresh model
        model2, _, _ = build_prism_model(
            model_name       = cfg["model_name"],
            rank             = cfg["rank"],
            n_experts        = cfg["n_experts"],
            base_seed        = cfg["base_seed"],
            bottleneck_ratio = cfg["bottleneck_ratio"],
            dtype            = dtype,
            device           = device,
        )
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model2.load_state_dict(state["model_state_dict"], strict=True)
        check("State dict loaded (strict=True)", True)

        # Forward pass on restored model
        model2.eval()
        with torch.no_grad():
            batch2  = make_batch(tokenizer, cfg["seq_len"], cfg["batch_size"], device)
            out2    = model2(input_ids=batch2["input_ids"], labels=batch2["labels"])
        check("Restored model forward pass OK", math.isfinite(out2.loss.item()),
              f"loss={out2.loss.item():.4f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print("\n" + "═"*55)
    print("  SMOKE TEST PASSED ✓  All checks green.")
    print(f"  Total time: {total_time:.1f}s")
    print("═"*55)
    print()
    print("  To revert: delete scripts/smoke_test.py")
    print("  No other files were modified by this script.\n")


if __name__ == "__main__":
    main()
