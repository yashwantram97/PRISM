# PRISM — Diagnostic Experiments Reference

> **For coding assistants.**
> This file documents every diagnostic experiment in `experiments.py` —
> what it measures, why it exists, how to interpret results, and what to
> do when it fails. Read this before modifying any experiment code.

---

## Quick Reference

| Experiment | When to Run | What It Tests | Pass Threshold | Fail Action |
|---|---|---|---|---|
| 1 — SVD Ratio | Step 1000 | Delta is token-specific | ratio > 2.0 | Raise lr_delta, check proj_v grads |
| 1b — Symmetry | Step 1000 | Experts are different from each other | spread > 0.5 | Check base_seed, SVD init |
| 2 — Delta Magnitude | Step 1000 | Delta is doing real work | ratio 0.01–0.50 | If <0.01: check proj_v grad |
| 3 — Routing Entropy | Step 1000 | Experts used diversely | H > 0.8 nats | Increase balance_weight 0.01→0.05 |
| 4 — Expert Diff | Convergence | Semantic > function differentiation | sem/fun > 1.5x | Check delta magnitude and routing |
| 5 — CKA Diversity | Convergence | Expert outputs are distinct | off-diag < 0.3 | Increase balance_weight or check symmetry |
| 6 — Gate Consistency | Convergence | Router learned stable preferences | semantic dom > 0.5 | Check training duration and balance loss |

**Experiments 1–3 are gates. All must pass at step 1000 before continuing training.**
**Experiments 4–6 run after full convergence to validate the mechanism.**

---

## Architecture Context

Before reading experiments, understand what you're measuring:

```
Per token x [d_model]:

  gates = softmax(router(x))                     [n_experts]
  
  For each expert i:
    h_i     = silu(gate_proj_i(x)) * up_proj_i(x)  [d_ff_expert]  base activation
    delta_i = proj_v_i(proj_u_i(x))                [d_ff_expert]  rank-r perturbation
    out_i   = down_proj_i(h_i + delta_i)           [d_model]
  
  output = sum(gates_i * out_i)                   [d_model]
```

**Key variables:**
- `d_model` — hidden dimension (896 for Qwen2.5-0.5B)
- `d_ff` — full FFN intermediate dim (4864 for Qwen2.5-0.5B)
- `d_ff_expert` — bottleneck per expert = `int(d_ff * bottleneck_ratio)` (e.g. 2432 at ratio=0.5)
- `rank` — rank of A and B matrices (default 4)
- `n_experts` — number of always-active experts (default 4)
- `proj_u` (A) — [d_model → rank] compresses token to rank-r fingerprint
- `proj_v` (B) — [rank → d_ff_expert] expands fingerprint to activation boost
- `_balance_loss` — entropy of gate distribution, stored on PRISMMoE after each forward

**What experts start from:**
- All 4 experts are initialized from SVD slices of the same pretrained Qwen FFN
- Expert i gets singular value chunk i (expert 0: dominant directions, expert 3: least dominant)
- proj_v (B) is zero-initialized — so delta=0 at step 0
- proj_u (A) is randomly initialized with different seed per expert per layer

---

## Experiment 1: SVD Ratio Gate Check

### File location
`experiments.py` → `experiment_1_svd_ratio()`

### Purpose
Verify the rank-r delta is producing **token-specific behaviour** after 1000 training steps.
This is the single most important check. If it fails, the entire PRISM mechanism is broken.

### How it works

The experiment hooks into `PRISMExpert.forward()` and captures the effective weight:

```python
# Inside the hook (simplified):
u      = module.proj_u(x_last)            # [1, rank] — token fingerprint
a_row  = module.proj_u.weight[0]          # [d_model] — first row of A
b_col  = module.proj_v.weight[:, 0]       # [d_ff_expert] — first col of B
delta_rank1 = outer(a_row, b_col) * u[0]  # rank-1 approximation of delta
W_eff  = up_proj.weight.T + delta_rank1   # [d_model, d_ff_expert]
```

It computes `W_eff` for two tokens: `"the"` (function word) and `"void"` (semantic word).
Then SVD of the difference:

```python
diff        = W_eff_void - W_eff_the     # [d_model, d_ff_expert]
_, s, _     = torch.linalg.svd(diff)
ratio       = s[0] / s[1]
```

**Why this works:**
If delta is token-specific, `W_eff` differs between tokens and the difference matrix
is approximately rank-1 (concentrated in one dominant singular direction) → `s[0] >> s[1]`.
If delta is just noise, the difference is full-rank → `s[0] ≈ s[1]`.

### Parameters
```python
experiment_1_svd_ratio(
    model,
    tokenizer,
    device,
    word_a     = "the",   # function word — should have low-specificity delta
    word_b     = "void",  # semantic word — should have high-specificity delta
    layer_idx  = 0,       # which layer to probe (layer 0 is fastest)
    expert_idx = 0,       # which expert to probe
)
```

### Thresholds
```
ratio > 5.0  →  Strong — delta is highly token-specific
ratio 2.0–5  →  Weak but present — training may improve it
ratio < 2.0  →  FAIL — delta not differentiating tokens
```

### Expected output
```
Top-5 singular values: ['0.8432', '0.1231', '0.0987', '0.0654', '0.0432']
s[0] / s[1] ratio: 6.847
✓ PASS  SVD ratio 6.85 > 2.0 threshold
         Strong
```

### Failure diagnosis

**ratio < 2.0 at step 1000:**

1. Check proj_v gradients — if all zero, B is stuck at zero-init and not learning:
   ```python
   for name, param in model.named_parameters():
       if "proj_v" in name and param.grad is not None:
           print(name, param.grad.abs().max().item())
   # Should be nonzero. If all zero → gradient not flowing to proj_v
   ```

2. Check if proj_u is learning at all:
   ```python
   # Compare proj_u.weight at step 0 vs step 1000
   # If unchanged → LR too low or gradient blocked
   ```

3. Fix: increase `lr_delta` from `2e-4` to `5e-4` in CONFIG

4. Check `balance_weight` — if too high (>0.1), entropy loss dominates and
   the delta learning signal gets washed out

---

## Experiment 1b: Symmetry Breaking Verification

### File location
`experiments.py` → `experiment_1b_symmetry_check()`

### Purpose
Verify that the 4 experts are actually different from each other at step 1000.
All experts start from different SVD slices + different proj_u seeds. If symmetry
breaking worked, each expert should show a different SVD ratio.

### How it works
Runs `experiment_1_svd_ratio()` for each of the 4 experts in layer 0.
Computes the spread (max - min) of ratios across experts.

### Thresholds
```
spread > 0.5  →  Experts are showing different token-specificity levels ✓
spread < 0.5  →  Experts too similar → symmetry breaking may have failed
```

### Expected output
```
Ratios per expert: ['6.85', '4.12', '3.67', '7.23']
Spread (max-min):  3.56
✓ PASS  Expert ratio spread > 0.5
         Symmetry breaking confirmed
```

### Failure diagnosis

**All experts have same ratio (spread < 0.5):**

1. Verify `base_seed` is being used in `PRISMMoE.__init__`:
   ```python
   seed = base_seed + layer_idx * n_experts + expert_idx
   torch.manual_seed(seed)
   nn.init.normal_(expert.proj_u.weight, std=0.01)
   ```
   If `base_seed` is the same for all and `layer_idx * n_experts + expert_idx` = 0
   for all experts → bug in seed computation.

2. Verify SVD slicing is actually producing different weights per expert:
   ```python
   # Check that expert 0 and expert 1 have different gate_proj weights
   w0 = model.model.layers[0].mlp.experts[0].gate_proj.weight
   w1 = model.model.layers[0].mlp.experts[1].gate_proj.weight
   print(torch.allclose(w0, w1))  # Should be False
   ```

---

## Experiment 2: Delta Magnitude Analysis

### File location
`experiments.py` → `experiment_2_delta_magnitude()`

### Purpose
Verify that the rank-r delta is doing **real work** — neither collapsed to near-zero
nor so large it's overriding the base activation.

At step 0: proj_v = zeros → delta = 0 (by design).
By step 1000: delta should have grown to a meaningful fraction of h.

### How it works
Hooks into all 4 experts in layer 0. For each token processed:

```python
h     = silu(gate_proj(x)) * up_proj(x)   # base hidden state [T, d_ff_expert]
delta = proj_v(proj_u(x))                  # rank-r delta      [T, d_ff_expert]

h_norm     = ||h||_2 averaged over tokens
delta_norm = ||delta||_2 averaged over tokens
ratio      = delta_norm / h_norm
```

Runs over `n_batches=100` batches and averages.

### Parameters
```python
experiment_2_delta_magnitude(
    model,
    dataloader,  # DataLoader yielding {"input_ids": tensor}
    device,
    n_batches = 100,
)
```

### Thresholds
```
ratio 0.05–0.30  →  Healthy — delta contributing meaningfully
ratio < 0.01     →  FAIL — delta collapsed (B stuck at zero-init)
ratio > 0.50     →  FAIL — delta overwhelming base activation
```

### Expected output
```
Expert   h_norm       delta_norm     ratio      Status
------------------------------------------------------------
0        2.3441       0.2156         0.0920     ✓
1        2.1893       0.1987         0.0907     ✓
2        2.4102       0.2341         0.0971     ✓
3        2.2876       0.1876         0.0820     ✓
✓ PASS  All expert delta/h ratios in [0.01, 0.50]
```

### Failure diagnosis

**ratio < 0.01 (collapsed):**
- proj_v gradient is not flowing. Verify in gradient check:
  ```python
  for name, p in model.named_parameters():
      if "proj_v" in name:
          print(f"{name}: grad={p.grad.abs().max() if p.grad is not None else 'None'}")
  ```
- If grad is None → proj_v might be getting disconnected from computation graph
- If grad is zero → loss signal not reaching proj_v (check balance_weight direction)
- Note: `L_total = L_lm - balance_weight * balance_loss` — the negative sign means
  we maximise entropy. If the sign is flipped accidentally, entropy is minimised
  → expert collapse → delta gets no useful gradient

**ratio > 0.50 (too large):**
- Reduce `lr_delta` from `2e-4` to `5e-5`
- Or add delta magnitude regularizer to loss:
  ```python
  delta_reg = sum(
      module.proj_v(module.proj_u(x)).norm(dim=-1).mean()
      for module in get_all_experts(model)
  )
  loss += 0.001 * delta_reg
  ```

---

## Experiment 3: Routing Entropy

### File location
`experiments.py` → `experiment_3_routing_entropy()`

### Purpose
Verify experts are being used **diversely** — not all tokens going to one expert.
Expert collapse is the most common failure mode in MoE training.

### How it works
Hooks into the `PRISMMoE.forward()` of every layer.
After each forward pass, reads the gate weights:

```python
gates = F.softmax(module.router(x_flat), dim=-1)   # [T, n_experts]
H     = -(gates * (gates + 1e-8).log()).sum(-1)     # [T] — per-token entropy
```

Collects entropies across `n_batches=200` batches and all MoE layers.

### Reference points (4 experts)
```
log(4) = 1.386 nats  →  perfectly uniform (each expert gets 25%)
log(2) = 0.693 nats  →  2 experts dominant
0.400 nats           →  one expert very dominant
0.100 nats           →  near-complete collapse
```

### Thresholds
```
H > 0.8 nats  →  routing is diverse ✓
H < 0.4 nats  →  FAIL — expert collapse
```

### Expected output
```
Reference: log(4)=1.386 (uniform), log(2)=0.693, collapse<0.4
Mean entropy: 1.021 nats
Min entropy:  0.312 nats
Max entropy:  1.384 nats

Per-layer mean entropy (sample):
  Layer  0: 0.987
  Layer  1: 1.043
  Layer  2: 1.012
  ...
  Layer 21: 0.998
  Layer 22: 1.031
  Layer 23: 1.008
✓ PASS  Mean entropy 1.021 > 0.8 nats
         Experts are routing diversely
```

### Failure diagnosis

**H < 0.4 (expert collapse):**

1. Increase `balance_weight` from `0.01` to `0.05` in CONFIG:
   ```python
   CONFIG["balance_weight"] = 0.05
   ```

2. Verify the sign of balance loss in `train_prism.py`:
   ```python
   # CORRECT (maximises entropy):
   loss = lm_loss - balance_weight * balance_loss
   
   # WRONG (minimises entropy → collapse):
   loss = lm_loss + balance_weight * balance_loss
   ```
   The `_balance_loss` stored in `PRISMMoE` is the entropy value (positive).
   Subtracting it from total loss maximises it. This is the correct sign.

3. Check that `_balance_loss` is not detached before being used:
   ```python
   # WRONG:
   self._balance_loss = entropy.detach()  # no gradient
   # CORRECT:
   self._balance_loss = entropy           # gradient flows
   ```

4. If collapse only in certain layers (check per-layer output), try layer-specific
   balance weights (advanced — only needed if standard fix doesn't work).

---

## Experiment 4: Expert Differentiation

### File location
`experiments.py` → `experiment_4_expert_differentiation()`

### Purpose
Directly test the **knowledge manipulation hypothesis**:
> "Rank-r delta makes 4 experts behave as different experts for different tokens"

Specifically: semantic tokens (high information content) should cause experts
to produce more diverse outputs than function tokens (low information content).

### How it works
For each token, runs all 4 experts and computes mean pairwise L2 distance:

```python
for i in range(4):
    for j in range(i+1, 4):
        dist = ||expert_i(x) - expert_j(x)||_2
mean_dist = average(all 6 pairs)
```

Compares semantic tokens (`["quantum", "neuron", "gradient", ...]`) vs
function tokens (`["the", "a", "is", "of", ...]`).

### Thresholds
```
sem/fun ratio > 1.5x  →  rank-r is steering experts differently by token type ✓
sem/fun ratio ≈ 1.0x  →  FAIL — experts not differentiating by token type
```

### Expected output
```
Token          Type       Mean Pairwise Dist
----------------------------------------
quantum        semantic   0.4821
neuron         semantic   0.4632
gradient       semantic   0.5012
...

the            function   0.2341
a              function   0.2198
is             function   0.2456
...

Semantic avg: 0.4876
Function avg: 0.2312
Ratio sem/fun: 2.11x
✓ PASS  Semantic/function distance ratio 2.11x > 1.5x
         Rank-r is steering experts differently by token type
```

### Failure diagnosis

**ratio ≈ 1.0 (no differentiation):**

1. Check delta magnitude first (Exp 2) — if delta is collapsed, experts can't differentiate
2. Check routing entropy (Exp 3) — if experts collapsed, all outputs are similar
3. If both pass but differentiation fails: the delta may be learning the same direction
   for all tokens. Check that proj_u weights are actually different per expert
   (symmetry breaking verification — Exp 1b)

---

## Experiment 5: CKA Expert Diversity

### File location
`experiments.py` → `experiment_5_cka_diversity()`

### Purpose
Measure **representational similarity** between expert outputs using Centered Kernel
Alignment (CKA). This is a more rigorous measure than pairwise distance — it's
invariant to rotation and isotropic scaling, so it measures structural similarity
of representations, not just magnitude differences.

### What CKA means
```
CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F × ||Y^T Y||_F)

CKA = 1.0  →  identical representations (experts collapsed)
CKA = 0.0  →  orthogonal representations (maximum diversity)
```

### How it works
Collects outputs from all 4 experts in layer 0 for 2000 tokens.
Computes full 4×4 CKA matrix. Reports mean of off-diagonal values.

### Thresholds
```
off-diagonal CKA < 0.3  →  experts have diverse representations ✓
off-diagonal CKA > 0.7  →  FAIL — experts functionally collapsed
```

### Expected output
```
CKA matrix (layer 0, 2000 samples):
        Exp0  Exp1  Exp2  Exp3
  Exp0  1.000 0.187 0.203 0.164
  Exp1  0.187 1.000 0.198 0.172
  Exp2  0.203 0.198 1.000 0.189
  Exp3  0.164 0.172 0.189 1.000

Mean off-diagonal CKA: 0.189
✓ PASS  Mean off-diagonal CKA 0.189 < 0.3
         Experts have diverse representations
```

### Failure diagnosis

**off-diagonal CKA > 0.7:**
Same root causes as Experiment 3 failure (expert collapse). Additionally:
- Check that SVD slices actually produced different initializations
- Try adding an explicit diversity loss:
  ```python
  # Add to training loss:
  expert_outputs = [expert(x) for expert in moe.experts]
  diversity_loss = sum(
      F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=-1).abs().mean()
      for i in range(n) for j in range(i+1, n)
  )
  loss += 0.001 * diversity_loss
  ```

---

## Experiment 6: Gate Weight Consistency

### File location
`experiments.py` → `experiment_6_gate_consistency()`

### Purpose
Verify the router has learned **stable, meaningful per-token expert preferences**.
A well-trained router should consistently send semantic tokens to the same expert(s)
regardless of context, while function tokens spread more evenly.

### How it works
For each token, embeds it in `n_contexts=200` different sentence templates:
```
"The {} was studied carefully."
"Scientists analyzed the {} in detail."
... (8 templates, cycling)
```

For each context, extracts the hidden state at the token position and feeds
it through the router to get gate weights. Computes:

```python
mean_gates  = average gate weights over 200 contexts  [n_experts]
dominance   = max(mean_gates)   # how much the top expert dominates
consistency = 1 - CV            # how stable the distribution is
             # CV = std/mean coefficient of variation
```

### Parameters
```python
experiment_6_gate_consistency(
    model,
    tokenizer,
    device,
    n_contexts = 200,   # number of random contexts per token
)
```

### Thresholds
```
semantic dominance > 0.5   →  one expert gets >50% weight consistently ✓
function dominance ≈ 0.25  →  weight spread evenly ✓ (uniform = 0.25 for 4 experts)
```

### Expected output
```
Token        Type       Dominance    Consistency  Mean Gate Weights
------------------------------------------------------------------------
quantum      semantic   0.612        0.743        ['0.61', '0.18', '0.12', '0.09']
neuron       semantic   0.587        0.721        ['0.59', '0.21', '0.13', '0.07']
...

the          function   0.298        0.412        ['0.30', '0.27', '0.24', '0.19']
a            function   0.287        0.389        ['0.29', '0.26', '0.25', '0.20']
...

Semantic avg dominance: 0.601  (target > 0.5)
Function avg dominance: 0.293  (target ≈ 0.25)
✓ PASS  Gate consistency (semantic>0.5, function<0.45)
         Router has learned stable token-expert preferences
```

### Failure diagnosis

**semantic dominance < 0.5 (router not specializing):**

1. May need more training — gate consistency typically emerges later than delta learning
2. Check balance_weight is not too high — if balance loss is dominating, router is
   forced toward uniform distribution regardless of token type
3. Try reducing balance_weight from 0.01 to 0.001 after convergence:
   ```python
   # In fine-tuning phase after initial convergence:
   CONFIG["balance_weight"] = 0.001
   ```

---

## Implementation Notes for Coding Assistants

### How experiments hook into the model

All experiments use PyTorch forward hooks. The pattern is:

```python
# Register
hook_handle = module.register_forward_hook(hook_fn)

# Run forward pass (hook fires automatically)
with torch.no_grad():
    model(**inputs)

# Remove (always remove after use to avoid memory leaks)
hook_handle.remove()
```

**Critical:** Always call `h.remove()` after each experiment. Leaving hooks
registered across experiments causes double-counting and memory leaks.

### Model structure navigation

```python
# Access MoE layer i
moe = model.model.layers[i].mlp         # PRISMMoE instance

# Access expert j in layer i
expert = model.model.layers[i].mlp.experts[j]  # PRISMExpert instance

# Router weights
router = model.model.layers[i].mlp.router      # nn.Linear

# Expert FFN weights
expert.gate_proj.weight    # [d_ff_expert, d_model]
expert.up_proj.weight      # [d_ff_expert, d_model]
expert.down_proj.weight    # [d_model, d_ff_expert]

# Rank-r matrices
expert.proj_u.weight       # [rank, d_model]     (A matrix)
expert.proj_v.weight       # [d_ff_expert, rank]  (B matrix)

# Balance loss (set during last forward pass)
moe._balance_loss          # scalar tensor
```

### DataLoader for experiments

All experiments that take a DataLoader expect batches of the form:
```python
{"input_ids": torch.LongTensor([batch_size, seq_len])}
```

No `labels` key needed — experiments only run forward passes, no loss computation.

Use `build_simple_dataloader()` in `experiments.py` to build a minimal
diagnostic dataloader from FineWeb-Edu without loading 5B tokens.

### Running on CPU (Mac)

All experiments work on CPU. They will be slow (especially Exp 5 with SVD
on large matrices) but functional. Reduce `n_batches` and `n_samples` for
faster iteration:

```python
# Faster on Mac:
experiment_2_delta_magnitude(model, dataloader, device, n_batches=20)
experiment_3_routing_entropy(model, dataloader, device, n_batches=50)
experiment_5_cka_diversity(model, dataloader, device, n_samples=500)
experiment_6_gate_consistency(model, tokenizer, device, n_contexts=50)
```

### Adding new experiments

New experiments should follow the same structure:
1. Accept `(model, ...)` as first argument
2. Call `model.eval()` at start
3. Use hooks — never run in training mode
4. Always call `hook.remove()` after
5. Return the primary metric as a scalar
6. Print `PASS/FAIL` clearly using the `result()` helper
7. Document threshold values in the function docstring

---

## Decision Trees

### At Step 1000

```
Run gate_check suite
        │
        ├── Exp 1: SVD ratio < 2.0?
        │     YES → stop training
        │            check proj_v grads
        │            increase lr_delta to 5e-4
        │            retrain from scratch
        │     NO  → continue
        │
        ├── Exp 1b: spread < 0.5?
        │     YES → stop training
        │            verify base_seed logic
        │            verify SVD slices different per expert
        │            retrain from scratch
        │     NO  → continue
        │
        ├── Exp 2: any ratio < 0.01?
        │     YES → stop training
        │            check proj_v gradient
        │            verify balance loss sign is negative (maximise entropy)
        │     NO  → continue
        │
        └── Exp 3: mean H < 0.4?
              YES → increase balance_weight 0.01 → 0.05
                     continue training (don't restart)
                     re-check entropy at step 2000
              NO  → ALL GATES PASSED → continue to full training
```

### After Convergence

```
Run convergence suite
        │
        ├── Exp 4: sem/fun ratio < 1.5x?
        │     YES → check Exp 2 (delta collapsed?)
        │            check Exp 3 (experts collapsed?)
        │            if both OK → need more training or higher rank
        │
        ├── Exp 5: off-diagonal CKA > 0.3?
        │     YES → increase balance_weight or add explicit diversity loss
        │
        └── Exp 6: semantic dominance < 0.5?
              YES → reduce balance_weight to 0.001 for fine-tuning phase
                     or train longer
        
All pass → run lm_eval benchmarks
           compare PRISM C-r4 vs Model D
           fill rank sweep result table
```

---

## Files Reference

```
prism/
├── experiments.py              ← all experiment code (this file's subject)
│
├── models/
│   ├── prism_moe.py            ← PRISMExpert, PRISMMoE
│   │    PRISMExpert.forward()  ← where hooks attach for Exp 1, 2, 4, 5
│   │    PRISMMoE.forward()     ← where hooks attach for Exp 3, 6
│   │    PRISMMoE._balance_loss ← entropy value read by Exp 3
│   └── model_builder.py        ← build_prism_model, load_prism_checkpoint
│
└── knowledge.md                ← full codebase overview
```
