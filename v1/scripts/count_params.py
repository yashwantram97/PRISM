"""
count_params.py
---------------
Builds the PRISM model and prints a detailed parameter audit:
  - Total params
  - Frozen params  (attention, embeddings, norms, lm_head)
  - Trainable params  (expert FFN + delta + router)
    - Expert FFN  (gate_proj, up_proj, down_proj across all experts/layers)
    - Delta A+B   (proj_u + proj_v  —  rank-r modulation)
    - Router      (linear gate per layer)

Run from the project root:
    python scripts/count_params.py
    python scripts/count_params.py --model Qwen/Qwen2.5-3B --rank 8 --n_experts 4
"""

import argparse
import sys
import os

# ── Allow importing from project root ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.model_builder import build_prism_model


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt(n: int) -> str:
    """Format param count as e.g. '1.44B' or '2.20M'."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    return f"{n:,}"


def count_params(model) -> dict:
    total       = 0
    frozen      = 0
    expert_ffn  = 0   # gate_proj, up_proj, down_proj inside PRISMExperts
    delta       = 0   # proj_u (A) + proj_v (B)
    router      = 0   # router linear

    for name, param in model.named_parameters():
        n = param.numel()
        total += n

        if not param.requires_grad:
            frozen += n
            continue

        # Classify trainable params
        if "proj_u" in name or "proj_v" in name:
            delta += n
        elif "router" in name:
            router += n
        else:
            expert_ffn += n

    trainable = total - frozen
    return {
        "total":      total,
        "frozen":     frozen,
        "trainable":  trainable,
        "expert_ffn": expert_ffn,
        "delta":      delta,
        "router":     router,
    }


def per_layer_summary(model) -> None:
    """Print per-layer expert/delta/router breakdown."""
    print("\n── Per-layer breakdown ──────────────────────────────────────────────")
    print(f"  {'Layer':>5}  {'Expert FFN':>12}  {'Delta A+B':>10}  {'Router':>8}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*10}  {'─'*8}")

    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):
            print(f"  {i:>5}  (plain FFN — not replaced)")
            continue

        ffn_params  = sum(p.numel() for name, p in mlp.named_parameters()
                          if "proj_u" not in name and "proj_v" not in name and "router" not in name)
        delta_params = sum(p.numel() for name, p in mlp.named_parameters()
                           if "proj_u" in name or "proj_v" in name)
        router_params = sum(p.numel() for name, p in mlp.named_parameters()
                            if "router" in name)

        print(f"  {i:>5}  {fmt(ffn_params):>12}  {fmt(delta_params):>10}  {fmt(router_params):>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Count PRISM model parameters")
    parser.add_argument("--model",            default="Qwen/Qwen2.5-0.5B", help="HF model name")
    parser.add_argument("--rank",              type=int,   default=4,    help="Rank r for delta")
    parser.add_argument("--n_experts",         type=int,   default=4,    help="Number of experts")
    parser.add_argument("--bottleneck_ratio",  type=float, default=0.5,  help="d_ff_expert = d_ff * ratio (0.25–1.0)")
    parser.add_argument("--base_seed",         type=int,   default=42,   help="Base random seed")
    parser.add_argument("--verbose",           action="store_true",      help="Show per-layer breakdown")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16

    print(f"\nBuilding PRISM model...")
    print(f"  Base:             {args.model}")
    print(f"  Rank:             {args.rank}")
    print(f"  Experts:          {args.n_experts}")
    print(f"  Bottleneck ratio: {args.bottleneck_ratio}")
    print(f"  Device:           {device}\n")

    model, tokenizer, info = build_prism_model(
        model_name       = args.model,
        rank             = args.rank,
        n_experts        = args.n_experts,
        base_seed        = args.base_seed,
        bottleneck_ratio = args.bottleneck_ratio,
        dtype            = dtype,
        device           = device,
    )

    counts = count_params(model)

    trainable = counts["trainable"]
    total     = counts["total"]
    frozen    = counts["frozen"]
    delta     = counts["delta"]
    expert_ffn = counts["expert_ffn"]
    router    = counts["router"]

    print("\n" + "═" * 55)
    print("  PRISM Parameter Audit")
    print("═" * 55)
    print(f"  Total params      : {fmt(total):>10}  (100.00%)")
    print(f"  ├─ Frozen         : {fmt(frozen):>10}  ({frozen/total*100:.2f}%)")
    print(f"  └─ Trainable      : {fmt(trainable):>10}  ({trainable/total*100:.2f}%)")
    print(f"     ├─ Expert FFN  : {fmt(expert_ffn):>10}  ({expert_ffn/total*100:.2f}%)")
    print(f"     ├─ Delta (A+B) : {fmt(delta):>10}  ({delta/total*100:.2f}%)")
    print(f"     └─ Router      : {fmt(router):>10}  ({router/total*100:.2f}%)")
    print("═" * 55)

    # Delta overhead relative to expert FFN
    if expert_ffn > 0:
        overhead = delta / expert_ffn * 100
        print(f"\n  Delta overhead vs expert FFN : {overhead:.3f}%")

    # Params per expert (FFN only)
    n_layers  = len(model.model.layers)
    per_expert = expert_ffn / (args.n_experts * n_layers) if n_layers else 0
    print(f"  Layers             : {n_layers}")
    print(f"  Params / expert    : {fmt(int(per_expert))}")
    print(f"  Active params/tok  : {fmt(total - frozen)}  (all experts always run)")

    if args.verbose:
        per_layer_summary(model)

    print()


if __name__ == "__main__":
    main()
