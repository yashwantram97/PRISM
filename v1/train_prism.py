"""
PRISM Training Script
----------------------
Fine-tunes the PRISM model on FineWeb-Edu (5B token slice).

Key design decisions:
  - AdamW with separate LR groups (router + delta can use higher LR)
  - Entropy regularizer as balance loss (maximise expert diversity)
  - Gradient clipping (1.0) — important with always-active experts
  - Periodic SVD gate check at step 1000 (go/no-go decision)
  - Checkpoint every 5000 steps + async final save
"""

import os
import sys
import logging
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, IterableDataset
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from models.model_builder import build_prism_model
from diagnostics.svd_check import svd_ratio_check


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    # Model
    "model_name":        "Qwen/Qwen2.5-0.5B",
    "rank":              4,
    "n_experts":         4,
    "base_seed":         42,
    "bottleneck_ratio":  0.5,    # d_ff_expert = d_ff * ratio (0.25=315M, 0.5=629M, 1.0=1.26B trainable)

    # Data
    "dataset":           "HuggingFaceFW/fineweb-edu",
    "dataset_split":     "train",
    "total_tokens":      5_000_000_000,   # 5B tokens
    "seq_len":           512,

    # Training
    "per_device_batch":  4,
    "grad_accum":        64,             # effective batch = 4*512*64 = 131,072 tokens
    "lr":                2e-4,
    "lr_delta":          2e-4,           # same for A,B — can tune separately
    "lr_router":         2e-4,
    "weight_decay":      0.1,
    "warmup_steps":      100,
    "grad_clip":         1.0,
    "balance_weight":    0.01,           # entropy regularizer weight
    "dtype":             "bfloat16",

    # Checkpointing
    "output_dir":        "./checkpoints/prism_r4",
    "save_every":        5000,
    "gate_check_step":   1000,           # SVD ratio gate check — go/no-go
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class PackedDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            # tokenize on the fly
            tokens = self.tokenizer(
                example["text"],
                truncation=False,
                padding=False,
                add_special_tokens=True,
            )["input_ids"]

            buffer.extend(tokens)
            buffer.append(self.tokenizer.eos_token_id)

            # yield fixed-size chunks
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                yield chunk


def build_dataloader(tokenizer, config: dict) -> DataLoader:
    """Stream FineWeb-Edu, tokenize, pack into seq_len chunks."""

    dataset = load_dataset(
        config["dataset"],
        split=config["dataset_split"],
        streaming=True,
    )

    packed_dataset = PackedDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        seq_len=config["seq_len"]
    )

    def collate(batch):
        ids = torch.tensor(batch, dtype=torch.long)
        return {
            "input_ids": ids[:, :-1],   # input
            "labels":    ids[:, 1:],    # shifted target
        }

    return DataLoader(
        packed_dataset,
        batch_size=config["per_device_batch"],
        collate_fn=collate,
        num_workers=0,                            # must be 0 for IterableDataset — workers each get their own copy of the same iterator
        pin_memory=torch.cuda.is_available(),     # no-op (and warning) on CPU
    )


# ── Optimizer ─────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Separate parameter groups so we can tune LR per component if needed.
    All currently use the same LR — easy to change.
    """
    expert_params = []
    delta_params  = []
    router_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "proj_u" in name or "proj_v" in name:
            delta_params.append(param)
        elif "router" in name:
            router_params.append(param)
        else:
            expert_params.append(param)

    param_groups = [
        {"params": expert_params, "lr": config["lr"],        "name": "expert_ffn"},
        {"params": delta_params,  "lr": config["lr_delta"],  "name": "delta_ab"},
        {"params": router_params, "lr": config["lr_router"], "name": "router"},
    ]

    print(f"  Optimizer groups:")
    print(f"    expert_ffn: {sum(p.numel() for p in expert_params)/1e6:.1f}M params")
    print(f"    delta_ab:   {sum(p.numel() for p in delta_params)/1e6:.1f}M params")
    print(f"    router:     {sum(p.numel() for p in router_params)/1e6:.3f}M params")

    return torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
        fused=torch.cuda.is_available(),  # CUDA-only optimisation; crashes on CPU if True
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train(config: dict = CONFIG):

    os.makedirs(config["output_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16

    # ── Init logging ──────────────────────────────────────────────────────────
    class TqdmToLogger(object):
        """File-like object to redirect tqdm output to a logger."""
        def __init__(self, logger, level=logging.INFO):
            self.logger = logger
            self.level = level
            self.linebuf = ''
        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())
        def flush(self):
            pass

    log_file = os.path.join(config["output_dir"], "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # Override built-in print to use our logger so we capture everything
    global print
    print = logger.info

    print(f"Logging to {log_file}")

    # ── Build model ───────────────────────────────────────────────────────────
    model, tokenizer, info = build_prism_model(
        model_name       = config["model_name"],
        rank             = config["rank"],
        n_experts        = config["n_experts"],
        base_seed        = config["base_seed"],
        bottleneck_ratio = config["bottleneck_ratio"],
        dtype            = dtype,
        device           = device,
    )

    # ── Dataloader ────────────────────────────────────────────────────────────
    dataloader = build_dataloader(tokenizer, config)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer  = build_optimizer(model, config)

    total_tokens   = config["total_tokens"]
    tokens_per_step = config["per_device_batch"] * config["seq_len"]
    total_steps    = total_tokens // tokens_per_step
    grad_updates   = total_steps // config["grad_accum"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=grad_updates,
    )

    print(f"\nTraining plan:")
    print(f"  Total tokens:     {total_tokens/1e9:.1f}B")
    print(f"  Tokens/step:      {tokens_per_step:,}")
    print(f"  Forward steps:    {total_steps:,}")
    print(f"  Grad updates:     {grad_updates:,}")
    print(f"  Effective batch:  {config['per_device_batch']*config['seq_len']*config['grad_accum']:,} tokens\n")

    # ── Enable Gradient Checkpointing ─────────────────────────────────────────
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()

    step         = 0
    update_step  = 0
    accum_loss   = 0.0
    accum_tokens = 0

    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    pbar = tqdm(total=grad_updates, desc="Training", file=tqdm_out)
    for batch in dataloader:
        if step >= total_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        # Forward
        with torch.amp.autocast('cuda', dtype=dtype):
            outputs = model(input_ids=input_ids, labels=labels)
            lm_loss = outputs.loss

            # Collect balance loss from all MoE layers
            balance_loss = torch.stack([
                layer.mlp._balance_loss
                for layer in model.model.layers
                if hasattr(layer.mlp, "_balance_loss")
            ]).mean()

            # Total loss: LM loss - balance_weight * entropy
            # (negative because we maximise entropy)
            loss = lm_loss - config["balance_weight"] * balance_loss
            loss = loss / config["grad_accum"]

        loss.backward()

        accum_loss   += lm_loss.item()
        accum_tokens += input_ids.numel()
        step         += 1

        # Gradient update
        if step % config["grad_accum"] == 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config["grad_clip"],
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            update_step += 1

            # ── Logging ───────────────────────────────────────────────────────
            if "log_every" in config and update_step % config["log_every"] == 0:
                avg_loss = accum_loss / config["grad_accum"] / config["log_every"]
                accum_loss = 0.0

            # ── Gate check at step 1000 ───────────────────────────────────────
            if update_step == config["gate_check_step"]:
                print(f"\n{'='*50}")
                print(f"GATE CHECK at step {update_step}")
                print(f"{'='*50}")
                model.eval()
                ratio = svd_ratio_check(model, tokenizer, device)
                model.train()

                # Compute a fresh avg_loss so the checkpoint doesn't record a
                # potentially stale/zero value from a recent log-reset.
                steps_since_last_log = (update_step % config["log_every"]) or config["log_every"]
                gate_avg_loss = accum_loss / max(steps_since_last_log * config["grad_accum"], 1)

                if ratio < 2.0:
                    print(f"\n⚠️  SVD ratio {ratio:.2f} < 2.0 — GATE FAILED")
                    print("Delta is not differentiating tokens.")
                    print("Check proj_u/proj_v gradients before continuing.")
                    print("Stopping training — debug before proceeding.\n")
                    save_checkpoint(model, optimizer, update_step, gate_avg_loss, config)
                    return
                else:
                    print(f"✓ SVD ratio {ratio:.2f} — gate passed, continuing\n")

            # ── Checkpointing ─────────────────────────────────────────────────
            if update_step % config["save_every"] == 0:
                save_checkpoint(model, optimizer, update_step,
                                accum_loss, config)

            lr_now = scheduler.get_last_lr()[0] if update_step > 0 else config["lr"]
            pbar.set_postfix({
                "loss": f"{loss.item() * config['grad_accum']:.4f}",
                "balance": f"{balance_loss.item():.4f}",
                "lr": f"{lr_now:.2e}"
            })
            pbar.update(1)

    pbar.close()
    # ── Final checkpoint ──────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, update_step, accum_loss, config,
                    name="final")
    print("Training complete.")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step, loss, config, name=None):
    tag  = name or f"step_{step}"
    path = os.path.join(config["output_dir"], f"checkpoint_{tag}.pt")
    torch.save({
        "step":                step,
        "loss":                loss,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "config":              config,
    }, path)
    print(f"  Checkpoint saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()