# PRISM — Codebase Knowledge File

> **For use with coding assistants.**
> This file explains what PRISM is, what we are trying to prove, how the
> architecture works, what every file does, and what the experiment expects
> as output. Read this before touching any code.

---

## 1. What We Are Trying to Prove

**The core hypothesis:**

> 4 always-active experts with token-conditioned rank-r modulation can match
> the benchmark performance of 16-expert top-4 sparse MoE — at identical
> active compute per token and 75% fewer total expert parameters.

This is called **PRISM** (Per-token Rank-r Input-conditioned Specialist Modulation).

**The key insight (knowledge manipulation):**

In a standard sparse MoE, 16 experts exist but only 4 run per token. The 12
inactive experts store knowledge that never gets used for that token. PRISM
replaces this with 4 experts that always run, but each expert's behavior is
dynamically steered by a small rank-r perturbation conditioned on the input
token. The rank-r delta manipulates which neurons fire inside each expert,
effectively making each expert behave differently per token — recovering the
functional diversity of 16 separate experts from only 4.

**Why this matters:**

If it works, PRISM is a step toward building a 10B active-parameter model
that performs like a 100B model. The 75% expert parameter saving is reinvested
into depth, width, and total MoE scaling in future stages.

---

## 2. Architecture: How PRISM Works

### 2.1 Standard SwiGLU FFN (what Qwen uses)

```
x → gate_proj → silu(·) ┐
x → up_proj   → ────────┘ × → h → down_proj → output
                elementwise
                multiply
```

```
h   = silu(gate_proj(x)) * up_proj(x)   # [d_ff]  — gated hidden state
out = down_proj(h)                       # [d_model]
```

### 2.2 PRISM Expert (PRISMExpert)

Each PRISM expert is a SwiGLU FFN plus a rank-r modulation:

```
h     = silu(gate_proj(x)) * up_proj(x)   # [d_ff]  — base activation (frozen-init, trainable)
u     = proj_u(x)                          # [rank]  — compress token to rank-r fingerprint
delta = proj_v(u)                          # [d_ff]  — expand to activation space
out   = down_proj(h + delta)               # [d_model] — delta steers which neurons fire
```

**What A (proj_u) and B (proj_v) do:**

- `A [d_model → rank]`: Compresses the token's representation into `rank` numbers.
  Different tokens produce different fingerprints. A learns to detect what is
  relevant about this token for this expert's specialization.

- `B [rank → d_ff]`: Expands the fingerprint into a d_ff-dimensional activation
  boost/suppress pattern. B learns which neurons to amplify or dampen given
  the token fingerprint from A.

- Together: `delta = B @ (A @ x)` = a rank-r matrix applied to x.
  Rank-r means delta can only shift activations in `rank` independent directions.
  Rank-1: one specialization mode. Rank-4: four. Rank-16: sixteen.

**Initialization:**
- `proj_u (A)`: `nn.init.normal_(std=0.01)` — different seed per expert per layer
- `proj_v (B)`: `nn.init.zeros_()` — so delta=0 at step 0, stable warmup from pretrained

### 2.3 PRISM MoE Layer (PRISMMoE)

Replaces one FFN layer in the transformer. Has `n_experts=4` PRISMExperts and
one router. All experts always run. Output is a softmax-weighted sum:

```
gates       = softmax(router(x))                           # [T, 4]
expert_outs = stack([expert_i(x) for i in 0..3])          # [T, 4, d_model]
output      = sum(gates_i * expert_outs_i, dim=experts)   # [T, d_model]
```

**Balance loss (entropy regularizer):**

```
balance_loss = -mean(sum(gates * log(gates + 1e-8), dim=-1))
```

This maximises entropy of the gate distribution, encouraging all 4 experts
to be used roughly equally. Added to total loss as:
`L_total = L_lm - balance_weight * balance_loss`

Note: negative because we maximise entropy (= minimise negative entropy).

### 2.4 Symmetry Breaking

All 4 experts are initialized from the same pretrained FFN weights. Without
intervention, they would receive identical gradients at step 0 and converge
to identical solutions — defeating the purpose of having multiple experts.

Fix: each expert's `proj_u` is initialized with a different random seed:
```python
seed = base_seed + layer_idx * n_experts + expert_idx
torch.manual_seed(seed)
nn.init.normal_(expert.proj_u.weight, std=0.01)
```

This makes delta_0 ≠ delta_1 ≠ delta_2 ≠ delta_3 from step 1, so experts
receive different gradient signals and diverge toward different specializations.

### 2.5 What Is Frozen vs Trainable

```
FROZEN (preserve pretrained Qwen knowledge):
  - self_attn (Q, K, V, O projections)
  - embed_tokens
  - all layernorms (input_layernorm, post_attention_layernorm, model.norm)
  - lm_head (tied to embed_tokens in Qwen2.5)

TRAINABLE (learn during fine-tuning):
  - gate_proj, up_proj, down_proj  — experts specialize during training
  - proj_u (A), proj_v (B)         — rank-r delta learns token steering
  - router                          — learns per-token expert weighting
```

**Why keep expert FFN weights trainable (not frozen):**

If we froze gate/up/down_proj, all 4 experts would have identical base
activations and delta alone would need to do all the differentiation work.
Keeping them trainable lets each expert's weights diverge during fine-tuning
as they receive different gradient signals through the softmax gating. This
gives two levels of specialization: base weight specialization + delta
token-steering.

---

## 3. Experiment Design

### 3.1 The Two Models Being Compared

**Model C-r (PRISM):**
- Base: Qwen2.5-0.5B (for smoke test / proof of concept)
- 4 experts, always active, softmax gating
- Rank-r delta per expert (sweep r = 1, 4, 8, 16)
- Total params: ~1.44B
- Active params/token: ~1.44B (all experts always run)

**Model D (compute-matched sparse MoE baseline):**
- Base: Qwen2.5-0.5B
- 16 experts, top-4 sparse routing
- No rank-r delta
- Total params: ~5.20B
- Active params/token: ~1.44B (only top-4 of 16 run)

**Why they are compute-matched:**
Both run exactly 4 expert FFN forward passes per token. The frozen
attention+embeddings (180M) are identical. Active FFLOPs per token are equal.
PRISM uses 75% fewer total expert parameters (1.26B vs 5.02B expert params).

### 3.2 What Success Looks Like

Primary (benchmark performance):
- PRISM C-r4 within 5% of Model D on MMLU, HellaSwag, GSM8K, ARC-Challenge
- Perplexity gap < 5%

Mechanistic (rank-r is actually working):
- SVD ratio > 3.0 after 1000 steps (Experiment 1 gate check)
- Routing entropy > 0.8 nats (experts are used diversely)
- CKA off-diagonal < 0.3 (expert outputs are distinct)
- Delta/h ratio 0.05–0.30 (delta doing real work, not collapsed)
- Semantic/function pairwise distance ratio > 1.5x (experts differentiate by token type)

### 3.3 Training Setup

```
Dataset:          HuggingFaceFW/fineweb-edu (5B token slice)
Seq length:       512
Batch size:       16 (per device)
Grad accumulation: 16
Effective batch:  131,072 tokens per gradient update
Learning rate:    2e-4 with cosine decay
Warmup steps:     100
Optimizer:        AdamW (beta1=0.9, beta2=0.95, wd=0.1)
Balance weight:   0.01
Gradient clip:    1.0
Precision:        bfloat16
```

**Critical gate at step 1000:**
Run SVD ratio check. If ratio < 2.0, stop training — the delta mechanism is
broken and continuing wastes compute. Debug proj_u/proj_v gradients first.

---

## 4. File-by-File Reference

### `models/prism_moe.py`

**Contains:** `PRISMExpert`, `PRISMMoE`

**PRISMExpert.__init__(d_model, d_ff, rank, original_ffn)**
- Copies gate_proj, up_proj, down_proj from `original_ffn` (pretrained Qwen FFN)
- Creates proj_u [d_model → rank] and proj_v [rank → d_ff]
- proj_v is zero-initialized (proj_u init is done externally for symmetry breaking)

**PRISMExpert.forward(x: [T, d_model]) → [T, d_model]**
- Computes SwiGLU base: `h = silu(gate_proj(x)) * up_proj(x)`
- Computes rank-r delta: `delta = proj_v(proj_u(x))`
- Returns: `down_proj(h + delta)`

**PRISMMoE.__init__(d_model, d_ff, rank, n_experts, original_ffn, layer_idx, base_seed)**
- Builds `n_experts` PRISMExperts
- Applies symmetry breaking: different `proj_u` seed per expert per layer
- Creates router: `nn.Linear(d_model, n_experts, bias=False)`

**PRISMMoE.forward(x: [B, S, D]) → [B, S, D]**
- Flattens to [T, D], computes gates, runs all experts, weighted sum
- Stores `self._balance_loss` (entropy) for retrieval by trainer
- Reshapes output back to [B, S, D]

---

### `models/model_builder.py`

**Contains:** `build_prism_model()`, `load_prism_checkpoint()`, `_verify_gradient_flow()`

**build_prism_model(model_name, rank, n_experts, base_seed, dtype, device)**

Step-by-step:
1. Loads pretrained model with `AutoModelForCausalLM.from_pretrained`
2. Freezes all params matching `FREEZE_KEYWORDS = ['self_attn', 'embed_tokens', 'norm', 'lm_head']`
3. Iterates `model.model.layers`, replaces each `layer.mlp` with `PRISMMoE`
4. Prints parameter audit (frozen / trainable / delta breakdown)
5. Runs `_verify_gradient_flow()` — one dummy forward+backward to confirm:
   - Gradients flow to: gate_proj, proj_u, proj_v, router
   - No gradients leak to: self_attn weights

Returns: `(model, tokenizer, info_dict)`

**FREEZE_KEYWORDS:** Matching is done with `any(kw in name for kw in FREEZE_KEYWORDS)`.
Be careful — "norm" matches both "input_layernorm" and "post_attention_layernorm"
as intended, but would also match any custom layer with "norm" in its name.

**load_prism_checkpoint(checkpoint_path, rank, n_experts, device)**
- Rebuilds the architecture from scratch (always from Qwen pretrained)
- Loads saved state_dict with `strict=False` (tolerates missing keys from frozen params)

---

### `train_prism.py`

**Contains:** `train()`, `build_dataloader()`, `build_optimizer()`, `save_checkpoint()`

**CONFIG dict (top of file)** — all hyperparameters in one place. Change
`model_name` to switch between 0.5B and 3B. Key fields:

```python
"model_name":      "Qwen/Qwen2.5-0.5B"    # or "Qwen/Qwen2.5-3B"
"rank":            4                        # sweep: 1, 4, 8, 16
"gate_check_step": 1000                     # SVD check step — go/no-go
"balance_weight":  0.01                     # entropy reg weight
```

**build_dataloader()** — Streams FineWeb-Edu, tokenizes, packs tokens into
seq_len+1 chunks (input = [:-1], labels = [1:]). Uses streaming to avoid
loading 5B tokens into memory.

**build_optimizer()** — Creates 3 param groups: `expert_ffn`, `delta_ab`,
`router`. All currently use the same LR (2e-4) but groups exist so you can
tune independently if needed (e.g., if delta collapses, raise `lr_delta`).

**train() loop:**
- AMP with bfloat16 autocast + GradScaler
- Collects `_balance_loss` from all MoE layers each step, takes mean
- Total loss: `L_lm - balance_weight * balance_loss` (maximise entropy)
- Gradient clip: 1.0 (important — always-active experts can have large grads)
- Gate check at step 1000: calls `svd_ratio_check()`, stops if ratio < 2.0
- Checkpoints every 5000 steps + final checkpoint

**Checkpoint format:**
```python
{
    "step": int,
    "loss": float,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "config": CONFIG,
}
```

---

### `diagnostics/svd_check.py`

**Contains:** `svd_ratio_check()`, `svd_check_all_experts()`

**svd_ratio_check(model, tokenizer, device, word_a, word_b, layer_idx, expert_idx)**

What it measures: the SVD ratio of `(W_eff_word_b - W_eff_word_a)`.

W_eff is the effective weight matrix after delta — if delta is token-specific,
then W_eff differs between tokens and the difference matrix should be
approximately rank-1 (concentrated in one dominant singular direction).

If delta is working: `s[0] >> s[1]` → ratio >> 1
If delta is noise:   `s[0] ≈ s[1]` → ratio ≈ 1

Thresholds:
- ratio > 5.0 → strong, mechanism working well
- ratio 2–5  → weak but present, training may improve it
- ratio < 2.0 → broken, stop and debug

Default words: `word_a="the"` (function word), `word_b="void"` (semantic word).
These are chosen because they should produce maximally different token embeddings.

**svd_check_all_experts(model, tokenizer, device, layer_idx)**
- Runs `svd_ratio_check` for all 4 experts in a layer
- Reports spread: if all experts have similar ratios → symmetry breaking may
  not have worked properly

---

## 5. Model D (TODO — not yet implemented)

Model D is the compute-matched sparse MoE baseline. It needs to be built
next. Key design:

```
Base:        Qwen2.5-0.5B
Experts:     16
Active/token: top-4 (standard sparse routing)
No rank-r delta
Routing:     learned router with top-k + load balancing loss
             (Switch Transformer auxiliary loss: n * sum(f_i * p_i))
```

Model D should be in `models/sparse_moe.py` with:
- `SparseExpert` class (plain SwiGLU, no delta)
- `SparseTopKMoE` class (top-k routing, auxiliary balance loss)
- Separate builder function in `model_builder.py` or new `model_d_builder.py`

The Switch Transformer balance loss (different from PRISM's entropy loss):
```python
# f_i = fraction of tokens routed to expert i
# p_i = mean router probability for expert i
balance_loss = n_experts * sum(f_i * p_i for i in range(n_experts))
L_total = L_lm + balance_weight * balance_loss
```

---

## 6. Diagnostics Plan (post-training)

After training both models, run these in order:

**Experiment 1: SVD ratio check** (already in `diagnostics/svd_check.py`)
- Gate check at step 1000 — go/no-go
- Run again at convergence for final report

**Experiment 2: Routing entropy**
- Measure mean entropy of gate distribution across 500 batches
- Target: > 0.8 nats (H_max for 4 experts = log(4) = 1.386 nats)
- Fail: < 0.4 nats → expert collapse

**Experiment 3: CKA expert diversity**
- Measure pairwise CKA between expert outputs for 2000 tokens
- Target: off-diagonal CKA < 0.3
- Fail: off-diagonal CKA > 0.7 → experts collapsed to identical representations

**Experiment 4: Gate consistency**
- For semantic tokens (quantum, neuron, gradient...) vs function tokens (the, a, is...)
- Measure dominance (max gate weight) and consistency across contexts
- Target: semantic dominance > 0.5, function dominance ~0.25

**Experiment 5: Delta magnitude**
- Measure delta_norm / h_norm ratio per expert
- Target: 0.05–0.30 (delta doing real work, not collapsed, not overwhelming)

**Experiment 6: Expert differentiation**
- Pairwise L2 distance between expert outputs per token
- Compare semantic vs function tokens
- Target: semantic/function ratio > 1.5x

**Benchmarks (lm-evaluation-harness):**
```bash
lm_eval --model hf \
    --model_args pretrained=./checkpoints/prism_r4 \
    --tasks mmlu,hellaswag,gsm8k,arc_challenge \
    --num_fewshot 5
```

---

## 7. Directory Structure

```
prism/
├── knowledge.md              ← this file
├── requirements.txt
├── train_prism.py            ← main training script (run this)
├── models/
│   ├── __init__.py
│   ├── prism_moe.py          ← PRISMExpert + PRISMMoE (core architecture)
│   ├── model_builder.py      ← build_prism_model() (loads Qwen, replaces FFNs)
│   └── sparse_moe.py         ← TODO: Model D (SparseExpert + SparseTopKMoE)
└── diagnostics/
    ├── __init__.py
    ├── svd_check.py           ← Experiment 1 gate check
    ├── entropy_check.py       ← TODO: Experiment 2
    ├── cka_check.py           ← TODO: Experiment 3
    ├── gate_consistency.py    ← TODO: Experiment 4
    ├── delta_magnitude.py     ← TODO: Experiment 5
    └── expert_differentiation.py  ← TODO: Experiment 6
```

---

## 8. Quick Reference — Key Numbers

### Qwen2.5-0.5B (current target)
```
d_model:    896
d_ff:       4864
n_layers:   24
n_heads:    14  (n_kv_heads=2, GQA)
vocab_size: 151936
```

### PRISM C-r4 on 0.5B
```
Total params:     1.44B
Frozen:           180M   (attention + embeddings)
Trainable:        1.26B  (experts + delta + router)
Delta overhead:   2.2M   (0.17% of total — negligible)
Memory (train):   ~21GB  (fits L4 24GB and Mac M4 Pro 24GB at batch=16)
```

### Model D on 0.5B
```
Total params:     5.20B
Expert params:    5.02B  (16 × 13.07M × 24)
Active/token:     1.44B  (same as PRISM — compute-matched)
```

### The Claim
```
PRISM:   1.44B total,  1.44B active/token,  4 expert FFN passes/token
Model D: 5.20B total,  1.44B active/token,  4 expert FFN passes/token

Same active compute. PRISM uses 75% fewer expert parameters.
If benchmarks match within 5% → hypothesis confirmed.
```

---

## 9. Common Failure Modes and Fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| SVD ratio < 2.0 at step 1000 | proj_v not getting gradients, or LR too low for delta | Check grad flow, increase `lr_delta` |
| All experts identical (CKA > 0.9) | Symmetry breaking not working | Verify different seeds used per expert, check `base_seed` logic |
| Entropy < 0.4 (expert collapse) | Router sends everything to one expert | Increase `balance_weight` from 0.01 to 0.05 |
| NaN loss | Gradient explosion | Reduce LR or verify `grad_clip=1.0` is active |
| Delta/h ratio < 0.01 | delta collapsed to zero | B (proj_v) stuck at zero — check gradients on proj_v |
| Delta/h ratio > 0.5 | Delta overwhelming base | Reduce `lr_delta`, add delta magnitude regularizer |
| OOM on L4 | Activation memory overflow | Reduce `per_device_batch` from 16 to 8 |

---

## 10. What Is NOT In This Codebase Yet

- `models/sparse_moe.py` — Model D (needs to be built next)
- Diagnostic scripts 2–6 (entropy, CKA, gate consistency, delta magnitude, expert diff)
- Evaluation script (wrapping lm-evaluation-harness)
- Rank sweep runner (loop over r=1,4,8,16 automatically)
- Comparison report generator (fills the rank sweep table from the paper)
