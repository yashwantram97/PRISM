"""
SVD Ratio Gate Check (Experiment 1)
-------------------------------------
After 1000 training steps, verify the rank-r delta is producing
token-specific behaviour. Compares W_eff for "the" vs "void".

If rank-r is working:
  W_eff_void - W_eff_the  should be approximately rank-1
  (the difference is concentrated in one dominant direction)
  → SVD ratio s[0]/s[1] >> 1

If rank-r is NOT working:
  The difference is noise, spread across all singular values
  → SVD ratio s[0]/s[1] ≈ 1

Thresholds:
  ratio > 5.0  → strong token-specific behaviour
  ratio 2-5    → weak but present
  ratio < 2.0  → mechanism broken, stop and debug
"""

import torch


def svd_ratio_check(
    model,
    tokenizer,
    device: str,
    word_a: str = "the",
    word_b: str = "void",
    layer_idx: int = 0,
    expert_idx: int = 0,
) -> float:
    """
    Compute SVD ratio of (W_eff_b - W_eff_a) for two tokens.
    W_eff = up_proj.weight.T + outer(proj_u(x), proj_v_cols)
            (rank-1 approximation of the effective weight after delta)

    Args:
        model:      PRISM model (eval mode)
        tokenizer:  corresponding tokenizer
        device:     "cuda" or "cpu"
        word_a:     function/common word (e.g. "the")
        word_b:     semantic/content word (e.g. "void")
        layer_idx:  which layer to probe (default: 0)
        expert_idx: which expert to probe (default: 0)

    Returns:
        ratio: s[0] / s[1] of the difference matrix
    """
    model.eval()

    tok_a = tokenizer(word_a, return_tensors="pt").to(device)
    tok_b = tokenizer(word_b, return_tensors="pt").to(device)

    W_effs = {word_a: None, word_b: None}
    hooks  = []

    def make_hook(word):
        def hook(module, inp, out):
            x = inp[0]                          # [T, d_model]
            # Use last token position
            x_last = x[-1].unsqueeze(0)         # [1, d_model]

            with torch.no_grad():
                u = module.proj_u(x_last)       # [1, rank]
                v = module.proj_v(u)             # [1, d_ff]

                # rank-1 approximation of delta as outer product
                # delta = B @ (A @ x) ≈ outer(u.squeeze(), v.squeeze())
                u_vec = u.squeeze().float()      # [rank]
                v_vec = v.squeeze().float()      # [d_ff]

                # Effective weight: base up_proj + rank-1 perturbation
                # Shape: [d_model, d_ff]
                W_base = module.up_proj.weight.T.float()  # [d_model, d_ff]

                # For rank-1 approx: outer product of first proj_u col and proj_v row
                # This is the dominant direction of the delta
                a_col = module.proj_u.weight[0].float()   # [d_model] first row of A
                b_row = module.proj_v.weight[:, 0].float() # [d_ff]   first col of B
                delta_rank1 = torch.outer(a_col, b_row) * u_vec[0]

                W_eff = W_base + delta_rank1
                W_effs[word] = W_eff.detach().cpu()

        return hook

    expert = model.model.layers[layer_idx].mlp.experts[expert_idx]

    for word in [word_a, word_b]:
        hooks.append(expert.register_forward_hook(make_hook(word)))

    with torch.no_grad():
        model(**tok_a)
        model(**tok_b)

    for h in hooks:
        h.remove()

    if W_effs[word_a] is None or W_effs[word_b] is None:
        print("  WARNING: hooks did not fire — check model architecture")
        return 0.0

    # SVD of the difference
    diff = W_effs[word_b] - W_effs[word_a]      # [d_model, d_ff]
    _, s, _ = torch.linalg.svd(diff.float())

    ratio = (s[0] / (s[1] + 1e-8)).item()

    print(f"\n  SVD Ratio Check: '{word_a}' vs '{word_b}' (layer {layer_idx}, expert {expert_idx})")
    print(f"  Top-5 singular values: {s[:5].tolist()}")
    print(f"  s[0]/s[1] ratio: {ratio:.3f}")

    if ratio > 5.0:
        print(f"  ✓ STRONG: delta is highly token-specific (ratio > 5.0)")
    elif ratio > 2.0:
        print(f"  ✓ PASS: delta is token-specific (ratio 2.0–5.0)")
    else:
        print(f"  ✗ FAIL: delta is NOT token-specific (ratio < 2.0)")
        print(f"         Check proj_u/proj_v gradients and learning rate")

    return ratio


def svd_check_all_experts(model, tokenizer, device, layer_idx=0):
    """
    Run SVD check for all experts in a layer.
    Useful to verify symmetry breaking worked —
    different experts should show different ratios.
    """
    print(f"\n  SVD check across all experts (layer {layer_idx}):")
    ratios = []
    n_experts = len(model.model.layers[layer_idx].mlp.experts)

    for i in range(n_experts):
        ratio = svd_ratio_check(
            model, tokenizer, device,
            layer_idx=layer_idx,
            expert_idx=i,
        )
        ratios.append(ratio)

    print(f"\n  Summary — ratios per expert: {[f'{r:.2f}' for r in ratios]}")
    print(f"  Min: {min(ratios):.2f}  Max: {max(ratios):.2f}  "
          f"Spread: {max(ratios)-min(ratios):.2f}")

    if max(ratios) - min(ratios) < 0.5:
        print("  ⚠️  WARNING: all experts have similar ratios — "
              "symmetry breaking may not be working")
    else:
        print("  ✓ Experts show different ratios — symmetry breaking confirmed")

    return ratios