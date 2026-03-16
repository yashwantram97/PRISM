"""
PRISM Model Builder
--------------------
Converts Qwen2.5-3B into a PRISM model:
  - Loads pretrained weights
  - Freezes attention, embeddings, layernorms
  - Replaces each FFN layer with PRISMMoE
  - Keeps expert weights trainable (they specialize during fine-tuning)
  - Verifies parameter counts and gradient flow
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prism_moe import PRISMMoE


# Modules whose names contain these strings will be frozen
FREEZE_KEYWORDS = [
    "self_attn",    # all attention weights (Q, K, V, O) — reasoning backbone, keep frozen
    "embed_tokens", # token embeddings — large (136M), domain-agnostic, keep frozen
    "lm_head",      # output projection — large (136M), keep frozen unless massive domain shift
    # "norm" intentionally excluded — layernorms (~86K params) are trainable so they
    # can adapt feature scale distributions to the fine-tuning domain (QLoRA best practice)
]


def build_prism_model(
    model_name:       str   = "Qwen/Qwen2.5-3B",
    rank:             int   = 4,
    n_experts:        int   = 4,
    base_seed:        int   = 42,
    dtype:            torch.dtype = torch.bfloat16,
    device:           str   = "cuda",
    bottleneck_ratio: float = 0.5,   # d_ff_expert = d_ff * bottleneck_ratio
) -> tuple:
    """
    Load Qwen2.5-3B and convert it to a PRISM model.

    Returns:
        model:     the converted PRISM model
        tokenizer: the corresponding tokenizer
        info:      dict with parameter counts and layer info
    """

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    d_model     = model.config.hidden_size                         # e.g. 896
    d_ff        = model.config.intermediate_size                   # e.g. 4864
    n_layers    = model.config.num_hidden_layers                   # e.g. 24
    d_ff_expert = max(1, int(d_ff * bottleneck_ratio))             # bottleneck width per expert

    print(f"  d_model={d_model}, d_ff={d_ff}, n_layers={n_layers}")
    print(f"  d_ff_expert={d_ff_expert}  (d_ff × {bottleneck_ratio:.2f} = {d_ff}×{bottleneck_ratio:.2f}, compression {d_ff/d_ff_expert:.1f}x per expert)")

    # ── Step 1: Freeze attention, embeddings, layernorms ─────────────────────
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in FREEZE_KEYWORDS):
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"  Frozen {frozen_count/1e6:.1f}M params (attention + embeddings)")

    # ── Step 2: Replace FFN layers with PRISMMoE ──────────────────────────────
    print(f"  Replacing {n_layers} FFN layers with PRISMMoE "
          f"(n_experts={n_experts}, rank={rank})...")

    for layer_idx, layer in enumerate(model.model.layers):
        original_ffn = layer.mlp

        layer.mlp = PRISMMoE(
            d_model      = d_model,
            d_ff         = d_ff,
            rank         = rank,
            n_experts    = n_experts,
            original_ffn = original_ffn,
            layer_idx    = layer_idx,
            base_seed    = base_seed,
            d_ff_expert  = d_ff_expert,
        ).to(dtype=dtype, device=device)

    # ── Step 3: Parameter audit ───────────────────────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params

    # Break down trainable params
    expert_base_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and any(k in name for k in ["gate_proj", "up_proj", "down_proj"])
    )
    delta_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and any(k in name for k in ["proj_u", "proj_v"])
    )
    router_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and "router" in name
    )

    print(f"\n{'='*62}")
    print(f"PRISM Model Parameter Summary")
    print(f"{'='*62}")
    print(f"  bottleneck_ratio:     {bottleneck_ratio:.2f}  →  d_ff_expert={d_ff_expert}  ({d_ff/d_ff_expert:.1f}x compression)")
    print(f"  Total params:         {total_params/1e6:>8.1f}M")
    print(f"  Frozen params:        {frozen_params/1e6:>8.1f}M  (attention + embeddings)")
    print(f"  Trainable params:     {trainable_params/1e6:>8.1f}M")
    print(f"    Expert base (FFN):  {expert_base_params/1e6:>8.1f}M  (bottleneck gate/up/down proj)")
    print(f"    Rank-r delta (A,B): {delta_params/1e6:>8.1f}M  (proj_u + proj_v)")
    print(f"    Router weights:     {router_params/1e6:>8.3f}M")
    print(f"{'='*62}")
    print(f"  Rank-r overhead:      {delta_params/trainable_params*100:.2f}% of trainable")
    print(f"{'='*62}\n")

    # ── Step 4: Gradient flow sanity check ───────────────────────────────────
    _verify_gradient_flow(model)

    info = {
        "d_model":            d_model,
        "d_ff":               d_ff,
        "d_ff_expert":        d_ff_expert,
        "bottleneck_ratio":   bottleneck_ratio,
        "n_layers":           n_layers,
        "n_experts":          n_experts,
        "rank":               rank,
        "total_params":       total_params,
        "trainable_params":   trainable_params,
        "frozen_params":      frozen_params,
        "expert_base_params": expert_base_params,
        "delta_params":       delta_params,
        "router_params":      router_params,
    }

    return model, tokenizer, info


def _verify_gradient_flow(model: nn.Module):
    """
    Quick sanity check: verify gradients flow to the right parameters.
    Runs one dummy forward+backward pass and checks grad existence.
    """
    print("  Verifying gradient flow...")

    # Find tokenizer vocab size from embedding
    vocab_size = model.model.embed_tokens.weight.shape[0]
    dummy_input = torch.zeros(1, 4, dtype=torch.long,
                              device=next(model.parameters()).device)
    dummy_input[0, :] = torch.randint(0, vocab_size, (4,))

    # Forward pass
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model(input_ids=dummy_input, labels=dummy_input)
    out.loss.backward()

    # Check gradients
    has_grad_expert  = False
    has_grad_delta   = False
    has_grad_router  = False
    no_grad_attn     = True

    for name, param in model.named_parameters():
        if param.grad is not None:
            if "gate_proj" in name or "up_proj" in name or "down_proj" in name:
                has_grad_expert = True
            if "proj_u" in name or "proj_v" in name:
                has_grad_delta = True
            if "router" in name:
                has_grad_router = True
        if param.grad is not None and "self_attn" in name:
            no_grad_attn = False

    status = {
        "Expert FFN grads":  "✓" if has_grad_expert else "✗ MISSING",
        "Delta (A,B) grads": "✓" if has_grad_delta  else "✗ MISSING",
        "Router grads":      "✓" if has_grad_router  else "✗ MISSING",
        "Attention frozen":  "✓" if no_grad_attn     else "✗ LEAKING",
    }

    for check, result in status.items():
        print(f"    {check}: {result}")

    # Zero gradients after check
    model.zero_grad()

    all_ok = has_grad_expert and has_grad_delta and has_grad_router and no_grad_attn
    if all_ok:
        print("  Gradient flow: ALL CHECKS PASSED\n")
    else:
        raise RuntimeError("Gradient flow check failed — review frozen/trainable setup")


def load_prism_checkpoint(
    checkpoint_path:  str,
    rank:             int   = 4,
    n_experts:        int   = 4,
    model_name:       str   = "Qwen/Qwen2.5-0.5B",
    bottleneck_ratio: float = 0.5,
    device:           str   = "cuda",
) -> tuple:
    """
    Rebuild model architecture and load saved weights.
    Use after training to resume or evaluate.

    model_name and bottleneck_ratio are auto-read from the checkpoint's
    saved config dict (if present), falling back to the supplied arguments.
    This ensures the architecture always matches what was trained.
    """
    state = torch.load(checkpoint_path, map_location=device)

    # Prefer values saved inside the checkpoint so architecture always matches
    saved_cfg      = state.get("config", {})
    model_name       = saved_cfg.get("model_name",       model_name)
    bottleneck_ratio = saved_cfg.get("bottleneck_ratio", bottleneck_ratio)

    model, tokenizer, info = build_prism_model(
        model_name=model_name,
        rank=rank,
        n_experts=n_experts,
        bottleneck_ratio=bottleneck_ratio,
        device=device,
    )
    model.load_state_dict(state["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}  "
          f"(model={model_name}, bottleneck_ratio={bottleneck_ratio})")
    return model, tokenizer, info