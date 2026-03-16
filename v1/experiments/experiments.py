# PRISM — Post-1000-Step Experiments
# =====================================
# Run these after 1000 training steps to verify the mechanism works.
# If all pass → continue training to convergence.
# If any fail → debug before spending more compute.
#
# Usage:
#   python experiments.py --checkpoint ./checkpoints/prism_r4/checkpoint_step_1000.pt
#
# Each experiment is self-contained and prints PASS/FAIL clearly.
# Experiments 1–3 are GO/NO-GO gates. Run in order.
# Experiments 4–6 run after full training convergence, not at 1000 steps.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import json
import os
import sys

# Insert the project root (one level above experiments/) so that
# 'from models.model_builder import ...' resolves correctly regardless
# of where the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_builder import build_prism_model, load_prism_checkpoint


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def result(label: str, passed: bool, detail: str = ""):
    icon = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {icon}  {label}")
    if detail:
        print(f"         {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: SVD Ratio Gate Check
# ───────────────────────────────────
# PURPOSE: Verify rank-r delta is producing token-specific behaviour.
#          This is the most important check — if it fails, nothing else matters.
#
# WHAT IT MEASURES:
#   Compute W_eff for two tokens ("the" and "void").
#   W_eff = effective weight seen by down_proj = up_proj.weight + delta contribution
#   Compute SVD of (W_eff_void - W_eff_the).
#   If delta is token-specific → difference is approximately rank-1 → s[0] >> s[1]
#   If delta is random noise   → difference is full-rank → s[0] ≈ s[1]
#
# THRESHOLDS:
#   ratio > 5.0  → strong, delta is highly token-specific
#   ratio 2–5    → weak but present, training may improve it
#   ratio < 2.0  → FAIL — delta not differentiating tokens, stop and debug
#
# WHAT TO DO IF IT FAILS:
#   1. Check proj_v gradients — if they're all zero, B is stuck at init
#   2. Check proj_u gradients — if tiny, A is not learning
#   3. Try increasing lr_delta from 2e-4 to 5e-4
#   4. Verify symmetry breaking worked (run experiment 1b below)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_1_svd_ratio(model, tokenizer, device,
                            word_a="the", word_b="void",
                            layer_idx=0, expert_idx=0):
    """
    SVD ratio of (W_eff_word_b - W_eff_word_a).
    Hooks into expert forward to capture the rank-r delta output.
    """
    section("EXPERIMENT 1: SVD Ratio Gate Check")
    print(f"  Comparing: '{word_a}' (function) vs '{word_b}' (semantic)")
    print(f"  Layer {layer_idx}, Expert {expert_idx}")

    model.eval()

    tok_a = tokenizer(word_a, return_tensors="pt").to(device)
    tok_b = tokenizer(word_b, return_tensors="pt").to(device)

    captured = {}
    hooks    = []

    def make_hook(word):
        def hook(module, inp, out):
            x = inp[0]
            with torch.no_grad():
                x_last = x[-1:].float()                  # last token [1, d_model]
                u      = module.proj_u(x_last)            # [1, rank]
                v      = module.proj_v(u)                 # [1, d_ff_expert]
                # Effective weight delta contribution (sum of rank-1 outer products)
                u_scale = u.squeeze()
                delta_rank_r = sum(
                    torch.outer(module.proj_u.weight[k].float(), 
                                module.proj_v.weight[:, k].float()) * u_scale[k].item()
                    for k in range(module.proj_u.weight.shape[0])
                )
                W_base = module.up_proj.weight.T.float()  # [d_model, d_ff_expert]
                captured[word] = (W_base + delta_rank_r).detach().cpu()
        return hook

    expert = model.model.layers[layer_idx].mlp.experts[expert_idx]
    hooks.append(expert.register_forward_hook(make_hook(word_a)))
    hooks.append(expert.register_forward_hook(make_hook(word_b)))

    with torch.no_grad():
        model(**tok_a)
        model(**tok_b)

    for h in hooks:
        h.remove()

    if word_a not in captured or word_b not in captured:
        result("SVD ratio hook", False, "Hooks did not fire — check model architecture")
        return 0.0

    diff = captured[word_b] - captured[word_a]
    _, s, _ = torch.linalg.svd(diff.float())

    ratio = (s[0] / (s[1] + 1e-8)).item()

    print(f"\n  Top-5 singular values: {[f'{x:.4f}' for x in s[:5].tolist()]}")
    print(f"  s[0] / s[1] ratio:     {ratio:.3f}")

    passed = ratio > 2.0
    result(f"SVD ratio {ratio:.2f} > 2.0 threshold", passed,
           "Strong" if ratio > 5.0 else ("Weak but present" if ratio > 2.0 else
           "FAIL → check proj_u/proj_v grads, try higher lr_delta"))
    return ratio


def experiment_1b_symmetry_check(model, tokenizer, device, layer_idx=0):
    """
    1b: Verify symmetry breaking worked.
    Run SVD check for all 4 experts — they should show different ratios.
    If all identical → symmetry breaking failed → experts will collapse.
    """
    section("EXPERIMENT 1b: Symmetry Breaking Verification")
    print(f"  Checking all experts differ in layer {layer_idx}")

    ratios = []
    n_experts = len(model.model.layers[layer_idx].mlp.experts)

    for i in range(n_experts):
        r = experiment_1_svd_ratio(model, tokenizer, device,
                                   layer_idx=layer_idx, expert_idx=i)
        ratios.append(r)

    spread = max(ratios) - min(ratios)
    passed = spread > 0.5

    print(f"\n  Ratios per expert: {[f'{r:.2f}' for r in ratios]}")
    print(f"  Spread (max-min):  {spread:.2f}")
    result("Expert ratio spread > 0.5", passed,
           "Symmetry breaking confirmed" if passed else
           "Experts too similar → check base_seed and SVD init")
    return ratios


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Delta Magnitude Analysis
# ────────────────────────────────────────
# PURPOSE: Verify delta is doing real work — not collapsed to near-zero
#          and not so large it's overwhelming the base activation.
#
# WHAT IT MEASURES:
#   For each expert in layer 0, compute across 100 batches:
#     h_norm     = ||silu(gate(x)) * up(x)||_2  (base hidden state magnitude)
#     delta_norm = ||proj_v(proj_u(x))||_2       (delta magnitude)
#     ratio      = delta_norm / h_norm
#
# THRESHOLDS:
#   ratio 0.05–0.30 → healthy: delta is doing real work
#   ratio < 0.01    → FAIL: delta collapsed (B stuck at zero)
#   ratio > 0.50    → FAIL: delta overwhelming base (reduce lr_delta)
#
# NOTE: At step 0, proj_v=zeros so ratio=0. By step 1000 it should be nonzero.
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_2_delta_magnitude(model, dataloader, device, n_batches=100):
    """
    Measure delta/h ratio per expert across n_batches.
    Pass a DataLoader that yields {"input_ids": tensor}.
    """
    section("EXPERIMENT 2: Delta Magnitude Analysis")

    model.eval()
    n_experts = len(model.model.layers[0].mlp.experts)
    results   = {i: {"h_norms": [], "delta_norms": [], "ratios": []}
                 for i in range(n_experts)}
    hooks     = []

    def make_hook(expert_idx):
        def hook(module, inp, out):
            x = inp[0].detach().float()                      # [T, d_model]
            with torch.no_grad():
                h     = F.silu(module.gate_proj(x)) * module.up_proj(x)
                delta = module.proj_v(module.proj_u(x))
                h_norm     = h.norm(dim=-1).mean().item()
                delta_norm = delta.norm(dim=-1).mean().item()
                results[expert_idx]["h_norms"].append(h_norm)
                results[expert_idx]["delta_norms"].append(delta_norm)
                results[expert_idx]["ratios"].append(
                    delta_norm / (h_norm + 1e-8))
        return hook

    for i, expert in enumerate(model.model.layers[0].mlp.experts):
        hooks.append(expert.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
            ids = batch["input_ids"].to(device)
            model(input_ids=ids)

    for h in hooks:
        h.remove()

    print(f"  {'Expert':<8} {'h_norm':<12} {'delta_norm':<14} {'ratio':<10} {'Status'}")
    print(f"  {'-'*56}")

    all_pass = True
    for i in range(n_experts):
        h_avg = sum(results[i]["h_norms"])    / len(results[i]["h_norms"])
        d_avg = sum(results[i]["delta_norms"]) / len(results[i]["delta_norms"])
        r_avg = sum(results[i]["ratios"])      / len(results[i]["ratios"])

        passed = 0.01 <= r_avg <= 0.50
        all_pass = all_pass and passed
        status = "✓" if passed else ("collapsed" if r_avg < 0.01 else "too large")
        print(f"  {i:<8} {h_avg:<12.4f} {d_avg:<14.4f} {r_avg:<10.4f} {status}")

    result("All expert delta/h ratios in [0.01, 0.50]", all_pass,
           "If any collapsed: check proj_v gradients and lr_delta" if not all_pass else "")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Routing Entropy
# ──────────────────────────────
# PURPOSE: Verify experts are being used diversely — not collapsed to one expert.
#
# WHAT IT MEASURES:
#   For each token, compute Shannon entropy of gate weights:
#     H = -sum(gate_i * log(gate_i))
#   Average over all tokens and all MoE layers.
#
# REFERENCE POINTS (4 experts):
#   H = log(4) = 1.386 nats → perfectly uniform (each expert gets 25%)
#   H = log(2) = 0.693 nats → 2 experts dominant
#   H < 0.4 nats            → one expert dominant → COLLAPSE
#
# THRESHOLDS:
#   H > 0.8 nats → routing is diverse ✓
#   H < 0.4 nats → FAIL: expert collapse → increase balance_weight (0.01 → 0.05)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_3_routing_entropy(model, dataloader, device, n_batches=200):
    """
    Measure mean routing entropy across all MoE layers over n_batches.
    """
    section("EXPERIMENT 3: Routing Entropy")

    model.eval()
    all_entropies = []
    layer_entropies = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            x_flat = inp[0].view(-1, inp[0].shape[-1]).detach().float()
            with torch.no_grad():
                gates = F.softmax(module.router(x_flat), dim=-1)
                H     = -(gates * (gates + 1e-8).log()).sum(-1)
                if layer_idx not in layer_entropies:
                    layer_entropies[layer_idx] = []
                layer_entropies[layer_idx].extend(H.tolist())
                all_entropies.extend(H.tolist())
        return hook

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "router"):
            hooks.append(layer.mlp.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
            ids = batch["input_ids"].to(device)
            model(input_ids=ids)

    for h in hooks:
        h.remove()

    mean_H = sum(all_entropies) / len(all_entropies)
    min_H  = min(all_entropies)
    max_H  = max(all_entropies)

    print(f"  Reference: log(4)=1.386 (uniform), log(2)=0.693, collapse<0.4")
    print(f"  Mean entropy: {mean_H:.4f} nats")
    print(f"  Min entropy:  {min_H:.4f} nats")
    print(f"  Max entropy:  {max_H:.4f} nats")

    # Per-layer summary (first 4 and last 4)
    n_layers = len(layer_entropies)
    print(f"\n  Per-layer mean entropy (sample):")
    layer_ids = sorted(layer_entropies.keys())
    show = layer_ids[:4] + (["..."] if n_layers > 8 else []) + layer_ids[-4:]
    for lid in show:
        if lid == "...":
            print(f"    ...")
            continue
        vals = layer_entropies[lid]
        print(f"    Layer {lid:>2}: {sum(vals)/len(vals):.4f}")

    passed = mean_H > 0.8
    result(f"Mean entropy {mean_H:.3f} > 0.8 nats", passed,
           "Experts are routing diversely" if passed else
           "Expert collapse → increase balance_weight 0.01→0.05 and retrain")
    return mean_H, layer_entropies


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS 4–6: Run after full training convergence (not at 1000 steps)
# ────────────────────────────────────────────────────────────────────────────
# These measure the trained model's behavior and require a converged checkpoint.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Experiment 4: Expert Differentiation ──────────────────────────────────────
# PURPOSE: Verify experts produce different outputs for the same token,
#          and that semantic tokens produce MORE differentiation than function tokens.
#          This directly tests the "knowledge manipulation" hypothesis.
#
# WHAT IT MEASURES:
#   For each token, run all 4 experts and compute mean pairwise L2 distance.
#   Compare semantic tokens (quantum, neuron...) vs function tokens (the, a, is...).
#
# THRESHOLDS:
#   sem/fun ratio > 1.5x → experts differentiate more for content tokens ✓
#   sem/fun ratio ≈ 1.0  → FAIL: experts not differentiating by token type

def experiment_4_expert_differentiation(model, tokenizer, device):
    """
    Pairwise L2 distance between expert outputs per token type.
    Run after full training convergence.
    """
    section("EXPERIMENT 4: Expert Differentiation (run after convergence)")

    semantic_tokens = ["quantum", "neuron", "gradient", "theorem",
                       "protein", "algorithm", "tensor", "entropy"]
    function_tokens = ["the", "a", "is", "of", "and", "to", "in", "it"]

    model.eval()
    results = {}

    for token in semantic_tokens + function_tokens:
        ids = tokenizer(token, return_tensors="pt").to(device)
        
        # Find target token position
        token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        target_id = token_ids[0]
        positions = (ids["input_ids"][0] == target_id).nonzero()
        if len(positions) == 0:
            continue
        pos = positions[0].item()

        expert_outs = []
        hooks = []

        def make_hook(idx_):
            def hook(module, inp, out):
                expert_outs.append(out.squeeze(0)[pos].detach().cpu().float())
            return hook

        for i, expert in enumerate(model.model.layers[0].mlp.experts):
            hooks.append(expert.register_forward_hook(make_hook(i)))

        with torch.no_grad():
            model(**ids)

        for h in hooks:
            h.remove()

        dists = []
        for i in range(len(expert_outs)):
            for j in range(i + 1, len(expert_outs)):
                dists.append((expert_outs[i] - expert_outs[j]).norm().item())
        results[token] = sum(dists) / len(dists) if dists else 0.0

    sem_avg = sum(results[t] for t in semantic_tokens) / len(semantic_tokens)
    fun_avg = sum(results[t] for t in function_tokens) / len(function_tokens)
    ratio   = sem_avg / (fun_avg + 1e-8)

    print(f"\n  {'Token':<14} {'Type':<10} {'Mean Pairwise Dist'}")
    print(f"  {'-'*40}")
    for t in semantic_tokens:
        print(f"  {t:<14} {'semantic':<10} {results[t]:.4f}")
    print()
    for t in function_tokens:
        print(f"  {t:<14} {'function':<10} {results[t]:.4f}")
    print(f"\n  Semantic avg: {sem_avg:.4f}")
    print(f"  Function avg: {fun_avg:.4f}")
    print(f"  Ratio sem/fun: {ratio:.2f}x")

    passed = ratio > 1.5
    result(f"Semantic/function distance ratio {ratio:.2f}x > 1.5x", passed,
           "Rank-r is steering experts differently by token type" if passed else
           "Experts not differentiating → check delta magnitude and routing entropy")
    return results, ratio


# ── Experiment 5: CKA Expert Diversity ────────────────────────────────────────
# PURPOSE: Measure representational similarity between expert outputs.
#          Low off-diagonal CKA = experts are functionally distinct.
#          High off-diagonal CKA = experts collapsed to same representation.
#
# WHAT IT MEASURES:
#   Centered Kernel Alignment (CKA) between all pairs of expert output matrices.
#   CKA = 1.0 → identical representations
#   CKA = 0.0 → orthogonal representations
#
# THRESHOLDS:
#   off-diagonal CKA < 0.3  → experts are diverse ✓
#   off-diagonal CKA > 0.7  → FAIL: experts collapsed

def _cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XTX = X.T @ X
    YTY = Y.T @ Y
    XTY = X.T @ Y
    num = (XTY * XTY).sum()
    den = ((XTX * XTX).sum().sqrt() * (YTY * YTY).sum().sqrt())
    return (num / (den + 1e-8)).item()


def experiment_5_cka_diversity(model, dataloader, device,
                                n_samples=2000, layer_idx=0):
    """
    CKA matrix between all expert pairs in one layer.
    Run after full training convergence.
    """
    section("EXPERIMENT 5: CKA Expert Diversity (run after convergence)")

    model.eval()
    n_experts = len(model.model.layers[layer_idx].mlp.experts)
    outputs   = {i: [] for i in range(n_experts)}
    hooks     = []

    def make_hook(idx_):
        def hook(module, inp, out):
            if sum(len(v) for v in outputs.values()) < n_samples * n_experts:
                outputs[idx_].append(out.detach().cpu().float()
                                        .view(-1, out.shape[-1]))
        return hook

    for i, expert in enumerate(model.model.layers[layer_idx].mlp.experts):
        hooks.append(expert.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for batch in dataloader:
            if all(len(outputs[i]) * outputs[i][0].shape[0] >= n_samples
                   for i in range(n_experts) if outputs[i]):
                break
            ids = batch["input_ids"].to(device)
            model(input_ids=ids)

    for h in hooks:
        h.remove()

    mats = {}
    for i in range(n_experts):
        if outputs[i]:
            mats[i] = torch.cat(outputs[i])[:n_samples]

    print(f"  CKA matrix (layer {layer_idx}, {n_samples} samples):")
    print(f"  {'':6}", end="")
    for j in range(n_experts):
        print(f"  Exp{j}", end="")
    print()

    off_diag_vals = []
    cka_matrix = {}
    for i in range(n_experts):
        print(f"  Exp{i}  ", end="")
        for j in range(n_experts):
            if i in mats and j in mats:
                c = _cka(mats[i], mats[j])
            else:
                c = 0.0
            cka_matrix[(i, j)] = c
            print(f"  {c:.3f}", end="")
            if i != j:
                off_diag_vals.append(c)
        print()

    mean_off_diag = sum(off_diag_vals) / len(off_diag_vals) if off_diag_vals else 1.0
    print(f"\n  Mean off-diagonal CKA: {mean_off_diag:.4f}")

    passed = mean_off_diag < 0.3
    result(f"Mean off-diagonal CKA {mean_off_diag:.3f} < 0.3", passed,
           "Experts have diverse representations" if passed else
           "Experts collapsed → increase balance_weight or check symmetry breaking")
    return cka_matrix, mean_off_diag


# ── Experiment 6: Gate Weight Consistency ─────────────────────────────────────
# PURPOSE: Verify the router has learned stable, meaningful per-token preferences.
#          Semantic tokens should consistently get high weight on one expert.
#          Function tokens should spread weight more evenly.
#
# WHAT IT MEASURES:
#   For each token, embed it in n_contexts different sentences.
#   Record gate weight vector each time.
#   Compute: dominance = mean(max gate weight)
#            consistency = 1 - coefficient of variation of gate weights
#
# THRESHOLDS:
#   semantic token dominance > 0.5  → one expert consistently dominant ✓
#   function token dominance ≈ 0.25 → weights spread evenly ✓

def experiment_6_gate_consistency(model, tokenizer, device, n_contexts=200):
    """
    Gate weight dominance and consistency for semantic vs function tokens.
    Run after full training convergence.
    """
    section("EXPERIMENT 6: Gate Weight Consistency (run after convergence)")

    semantic_tokens = ["quantum", "neuron", "gradient", "theorem"]
    function_tokens = ["the", "a", "is", "of"]

    context_templates = [
        "The {} was studied carefully.",
        "Scientists analyzed the {} in detail.",
        "In this paper we discuss {}.",
        "Understanding {} is fundamental.",
        "The concept of {} is important.",
        "Researchers found that {} plays a key role.",
        "The {} has been widely observed.",
        "This paper focuses on {}.",
    ]

    model.eval()
    n_experts = len(model.model.layers[0].mlp.experts)
    results   = {}

    for token in semantic_tokens + function_tokens:
        gate_vectors = []

        for ctx_idx in range(n_contexts):
            template = context_templates[ctx_idx % len(context_templates)]
            sentence = template.format(token)
            ids = tokenizer(sentence, return_tensors="pt").to(device)

            # Find token position
            token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
            if not token_ids:
                continue
            target_id = token_ids[0]
            positions = (ids["input_ids"][0] == target_id).nonzero()
            if len(positions) == 0:
                continue
            pos = positions[0].item()

            with torch.no_grad():
                hidden = model.model(
                    input_ids=ids["input_ids"],
                    output_hidden_states=True,
                ).hidden_states[-1][0, pos]   # [d_model]

                gates = F.softmax(
                    model.model.layers[0].mlp.router(hidden.unsqueeze(0)),
                    dim=-1,
                ).squeeze(0).cpu()
                gate_vectors.append(gates)

        if not gate_vectors:
            continue

        gate_matrix = torch.stack(gate_vectors)     # [n_contexts, n_experts]
        mean_gates  = gate_matrix.mean(0)            # [n_experts]
        std_gates   = gate_matrix.std(0)             # [n_experts]
        dominance   = mean_gates.max().item()
        cv          = (std_gates / (mean_gates + 1e-8)).mean().item()
        consistency = max(0.0, 1.0 - cv)

        results[token] = {
            "mean_gates":   mean_gates.tolist(),
            "dominance":    dominance,
            "consistency":  consistency,
        }

    print(f"\n  {'Token':<12} {'Type':<10} {'Dominance':<12} {'Consistency':<12} {'Mean Gate Weights'}")
    print(f"  {'-'*72}")

    for token in semantic_tokens:
        if token not in results:
            continue
        r = results[token]
        g = [f"{v:.2f}" for v in r["mean_gates"]]
        print(f"  {token:<12} {'semantic':<10} {r['dominance']:<12.3f} "
              f"{r['consistency']:<12.3f} {g}")
    print()
    for token in function_tokens:
        if token not in results:
            continue
        r = results[token]
        g = [f"{v:.2f}" for v in r["mean_gates"]]
        print(f"  {token:<12} {'function':<10} {r['dominance']:<12.3f} "
              f"{r['consistency']:<12.3f} {g}")

    sem_dom = sum(results[t]["dominance"] for t in semantic_tokens if t in results)
    sem_dom /= max(1, sum(1 for t in semantic_tokens if t in results))
    fun_dom = sum(results[t]["dominance"] for t in function_tokens if t in results)
    fun_dom /= max(1, sum(1 for t in function_tokens if t in results))

    print(f"\n  Semantic avg dominance: {sem_dom:.3f}  (target > 0.5)")
    print(f"  Function avg dominance: {fun_dom:.3f}  (target ≈ 0.25)")

    passed = sem_dom > 0.5 and fun_dom < 0.45
    result("Gate consistency (semantic>0.5, function<0.45)", passed,
           "Router has learned stable token-expert preferences" if passed else
           "Router not specializing — check balance loss and training duration")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def build_simple_dataloader(tokenizer, device, batch_size=8, seq_len=512,
                             n_batches=200):
    """
    Build a minimal dataloader from FineWeb-Edu for diagnostic use.
    Only loads enough data to run the experiments.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    buffer = []
    chunks = []

    for sample in dataset:
        tokens = tokenizer(sample["text"], truncation=False,
                           add_special_tokens=True)["input_ids"]
        buffer.extend(tokens)
        buffer.append(tokenizer.eos_token_id)
        while len(buffer) >= seq_len + 1:
            chunks.append(buffer[:seq_len + 1])
            buffer = buffer[seq_len + 1:]
        if len(chunks) >= n_batches * batch_size:
            break

    def collate(batch):
        ids = torch.tensor([b for b in batch], dtype=torch.long)
        return {"input_ids": ids[:, :-1]}

    from torch.utils.data import TensorDataset
    all_ids = torch.tensor(chunks[:n_batches * batch_size], dtype=torch.long)
    dataset_t = torch.utils.data.TensorDataset(all_ids)

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return {"input_ids": self.data[i, :-1]}

    return DataLoader(
        SimpleDataset(all_ids),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_gate_check_experiments(model, tokenizer, device):
    """
    Run experiments 1, 1b, 2, 3 — the go/no-go gate check suite.
    Run these at step 1000 before deciding to continue training.
    """
    section("GATE CHECK SUITE (Step 1000)")
    print("  Running experiments 1, 1b, 2, 3")
    print("  All three must pass before continuing training.")

    model.eval()

    # Exp 1: SVD ratio
    ratio = experiment_1_svd_ratio(model, tokenizer, device)

    # Exp 1b: Symmetry check
    ratios_per_expert = experiment_1b_symmetry_check(model, tokenizer, device)

    # Exp 2 + 3 need a dataloader — build minimal one
    print("\n  Building diagnostic dataloader (this takes ~1-2 min)...")
    dataloader = build_simple_dataloader(tokenizer, device,
                                          batch_size=8, seq_len=512,
                                          n_batches=100)

    # Exp 2: Delta magnitude
    delta_results = experiment_2_delta_magnitude(model, dataloader, device,
                                                  n_batches=100)

    # Exp 3: Routing entropy
    mean_H, layer_H = experiment_3_routing_entropy(model, dataloader, device,
                                                    n_batches=200)

    # ── Final verdict ─────────────────────────────────────────────────────────
    section("GATE CHECK VERDICT")
    svd_pass     = ratio > 2.0
    sym_pass     = (max(ratios_per_expert) - min(ratios_per_expert)) > 0.5
    entropy_pass = mean_H > 0.8
    delta_pass   = all(
        0.01 <= sum(delta_results[i]["ratios"]) / len(delta_results[i]["ratios"]) <= 0.50
        for i in delta_results
    )

    result("Exp 1  — SVD ratio > 2.0",             svd_pass)
    result("Exp 1b — Expert ratio spread > 0.5",   sym_pass)
    result("Exp 2  — Delta/h ratio 0.01–0.50",     delta_pass)
    result("Exp 3  — Routing entropy > 0.8 nats",  entropy_pass)

    all_pass = svd_pass and sym_pass and delta_pass and entropy_pass

    print()
    if all_pass:
        print("  ✓ ALL GATES PASSED — continue training to convergence")
    else:
        print("  ✗ GATES FAILED — debug before continuing")
        if not svd_pass:
            print("    → SVD failed: check proj_u/proj_v grads, increase lr_delta")
        if not sym_pass:
            print("    → Symmetry failed: check base_seed, verify SVD init ran")
        if not delta_pass:
            print("    → Delta collapsed: check proj_v grad is nonzero")
        if not entropy_pass:
            print("    → Collapse: increase balance_weight from 0.01 to 0.05")

    return all_pass


def run_convergence_experiments(model, tokenizer, device):
    """
    Run experiments 4, 5, 6 — the full mechanism validation suite.
    Run these after full training convergence.
    """
    section("CONVERGENCE EXPERIMENT SUITE")
    print("  Running experiments 4, 5, 6")

    model.eval()
    dataloader = build_simple_dataloader(tokenizer, device,
                                          batch_size=8, seq_len=512,
                                          n_batches=300)

    diff_results, ratio = experiment_4_expert_differentiation(
        model, tokenizer, device)
    cka_matrix, mean_cka = experiment_5_cka_diversity(
        model, dataloader, device)
    gate_results = experiment_6_gate_consistency(
        model, tokenizer, device)

    section("CONVERGENCE VERDICT")
    diff_pass = ratio > 1.5
    cka_pass  = mean_cka < 0.3
    gate_pass = (
        sum(gate_results[t]["dominance"]
            for t in ["quantum", "neuron", "gradient", "theorem"]
            if t in gate_results) / 4 > 0.5
    )

    result("Exp 4 — Expert differentiation ratio > 1.5x", diff_pass)
    result("Exp 5 — CKA off-diagonal < 0.3",              cka_pass)
    result("Exp 6 — Semantic gate dominance > 0.5",       gate_pass)

    all_pass = diff_pass and cka_pass and gate_pass
    print()
    if all_pass:
        print("  ✓ MECHANISM VALIDATED — rank-r is acting as knowledge manipulator")
        print("  Next: run benchmark eval (lm_eval) and compare to Model D")
    else:
        print("  ✗ MECHANISM NOT FULLY VALIDATED — see individual failures above")

    return all_pass


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM Experiments")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--mode", type=str,
                        choices=["gate_check", "convergence", "all"],
                        default="gate_check",
                        help="gate_check=step 1000 suite, convergence=post-training suite")
    parser.add_argument("--rank",             type=int,   default=4)
    parser.add_argument("--n_experts",         type=int,   default=4)
    parser.add_argument("--model_name",        type=str,   default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--bottleneck_ratio",  type=float, default=0.5,
                        help="Must match the ratio used during training (default: 0.5)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, info = load_prism_checkpoint(
        checkpoint_path=args.checkpoint,
        rank=args.rank,
        n_experts=args.n_experts,
        model_name=args.model_name,
        bottleneck_ratio=args.bottleneck_ratio,
        device=device,
    )
    model.eval()

    if args.mode in ("gate_check", "all"):
        gate_ok = run_gate_check_experiments(model, tokenizer, device)

    if args.mode in ("convergence", "all"):
        conv_ok = run_convergence_experiments(model, tokenizer, device)
