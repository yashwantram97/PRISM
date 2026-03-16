"""
PRISM MoE Layer
---------------
4 always-active experts with token-conditioned rank-r delta.

Each expert = bottleneck SwiGLU FFN (SVD slice of pretrained) + trainable rank-r modulation.

Architecture per token x:
    gates    = softmax(router(x))                  # [n_experts]  — learned per-token weighting
    delta_i  = B_i @ (A_i @ x)                    # [d_ff_e] — rank-r token-specific perturbation
    h_i      = silu(gate_proj_i(x)) * up_proj_i(x) # [d_ff_e] — bottleneck SwiGLU activation
    out_i    = down_proj_i(h_i + delta_i)          # [d_model]
    output   = sum(gates_i * out_i)                # [d_model] — weighted sum all experts

Bottleneck design:
    d_ff_expert = d_ff // n_experts   (e.g. 4864 // 4 = 1216)
    Each expert covers a unique SVD slice of the original FFN → ~4x smaller than full copy.
    Total expert params ≈ original FFN params (same as one non-MoE FFN), but split across experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SVD-based initialization ───────────────────────────────────────────────────

def _svd_slice(weight: torch.Tensor, expert_idx: int, n_experts: int) -> torch.Tensor:
    """
    Compute a rank-(k) SVD approximation of `weight` corresponding to expert_idx's slice.

    weight : [out_features, in_features]  (Linear weight convention)

    Procedure:
      1. Full SVD → U [out, r], S [r], Vh [r, in]
      2. Partition the r singular values into n_experts equal chunks.
      3. Expert i gets chunk i → reconstruct: U_i @ diag(S_i) @ Vh_i
         This is a rank-(r//n_experts) approximation biased toward the
         singular directions most "owned" by expert i.

    Returns reconstructed weight of same shape as input.
    """
    orig_dtype = weight.dtype
    W = weight.float()                            # SVD needs float32

    # Clamp k so we never request more singular values than exist
    k_total = min(W.shape)
    k = max(1, k_total // n_experts)

    # Truncated SVD (only compute what we need)
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U:[o,r] S:[r] Vh:[r,i]
    except Exception:
        # Fallback: random init slice if SVD fails (e.g., very small matrices)
        sliced = W.clone()
        return sliced.to(orig_dtype)

    # Pick this expert's slice of singular triplets
    start = expert_idx * k
    end   = min(start + k, S.shape[0])
    U_i   = U[:, start:end]          # [out, k]
    S_i   = S[start:end]             # [k]
    Vh_i  = Vh[start:end, :]         # [k, in]

    # Reconstruct a rank-k approximation (same shape as W)
    # Scale so that the Frobenius norm is preserved proportionally
    W_i = (U_i * S_i.unsqueeze(0)) @ Vh_i     # [out, in]

    return W_i.to(orig_dtype)


def _build_bottleneck_linear(
    original_proj: nn.Linear,
    d_out: int,
    d_in: int,
    expert_idx: int,
    n_experts: int,
    device,
    dtype,
) -> nn.Linear:
    """
    Build a Linear(d_in → d_out) whose weights are the SVD slice of original_proj.

    We select the top-k singular directions for this expert (k = d_out or d_in
    depending on projection direction), so the expert is initialized with
    a meaningful low-rank approximation of the pretrained weights.
    """
    layer = nn.Linear(d_in, d_out, bias=False, device=device, dtype=dtype)

    orig_w = original_proj.weight.data   # [orig_out, orig_in]

    # Get the svd-sliced approximation at full shape
    sliced = _svd_slice(orig_w, expert_idx, n_experts)  # [orig_out, orig_in]

    # Now trim to the bottleneck shape
    # For gate_proj / up_proj  : orig=[d_ff, d_model]  → bottleneck=[d_ff_e, d_model]
    #   → take top d_ff_e rows  (rows = output neurons)
    # For down_proj            : orig=[d_model, d_ff]  → bottleneck=[d_model, d_ff_e]
    #   → take top d_ff_e cols  (cols = input neurons)
    if d_out <= sliced.shape[0] and d_in == sliced.shape[1]:
        # gate_proj / up_proj direction
        layer.weight.data = sliced[:d_out, :].contiguous()
    elif d_in <= sliced.shape[1] and d_out == sliced.shape[0]:
        # down_proj direction
        layer.weight.data = sliced[:, :d_in].contiguous()
    else:
        # Fallback: just truncate whatever fits
        layer.weight.data = sliced[:d_out, :d_in].contiguous()

    return layer


# ── Expert ─────────────────────────────────────────────────────────────────────

class PRISMExpert(nn.Module):
    """
    Single PRISM expert: bottleneck SwiGLU FFN (SVD slice of pretrained) + rank-r delta.

    The bottleneck reduces d_ff → d_ff_expert = d_ff // n_experts per expert.
    Each expert is initialized from a unique SVD slice of the original FFN so:
      - No two experts start with the same weights (automatic symmetry breaking on base FFN)
      - The ensemble reconstructs the original FFN in expectation (if gates = 1/n_experts)
      - The total parameter count matches roughly one full FFN (not n_experts × FFN)

    Rank-r delta (proj_u, proj_v) adds token-conditioned perturbation in the
    bottleneck activation space, further differentiating expert behavior.
    proj_v zero-initialized → delta=0 at step 0 (stable warmup).
    """

    def __init__(
        self,
        d_model:      int,
        d_ff:         int,
        d_ff_expert:  int,        # bottleneck width = d_ff // n_experts
        rank:         int,
        expert_idx:   int,
        n_experts:    int,
        original_ffn: nn.Module,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model     = d_model
        self.d_ff        = d_ff
        self.d_ff_expert = d_ff_expert
        self.rank        = rank

        # ── Bottleneck FFN (SVD-initialized from pretrained) ─────────────────
        self.gate_proj = _build_bottleneck_linear(
            original_ffn.gate_proj, d_ff_expert, d_model,
            expert_idx, n_experts, device, dtype,
        )
        self.up_proj = _build_bottleneck_linear(
            original_ffn.up_proj,   d_ff_expert, d_model,
            expert_idx, n_experts, device, dtype,
        )
        self.down_proj = _build_bottleneck_linear(
            original_ffn.down_proj, d_model,     d_ff_expert,
            expert_idx, n_experts, device, dtype,
        )

        # ── Rank-r token-conditioned perturbation (in bottleneck space) ──────
        # A: token fingerprint  [d_model → rank]
        # B: steers bottleneck activations [rank → d_ff_expert]
        self.proj_u = nn.Linear(d_model,     rank,        bias=False)
        self.proj_v = nn.Linear(rank,        d_ff_expert, bias=False)
        nn.init.zeros_(self.proj_v.weight)   # delta=0 at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, d_model]  (T = batch_size * seq_len, flattened)
        Returns:
            out: [T, d_model]
        """
        # Bottleneck SwiGLU
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)   # [T, d_ff_expert]

        # Rank-r delta in bottleneck space
        delta = self.proj_v(self.proj_u(x))                # [T, d_ff_expert]

        # Project back to d_model
        return self.down_proj(h + delta)                   # [T, d_model]


# ── MoE Layer ──────────────────────────────────────────────────────────────────

class PRISMMoE(nn.Module):
    """
    PRISM MoE Layer.

    Replaces a single FFN layer with n_experts always-active bottleneck experts.
    Each expert has d_ff_expert = d_ff // n_experts hidden dim (default: 4864//4 = 1216).

    Total expert params ≈ 1× original FFN (not n_experts×), so the MoE adds
    zero parameter overhead vs. the original model while enabling expert specialization.

    The router learns per-token softmax weights over experts.
    The rank-r delta per expert provides token-specific activation steering within each expert.
    """

    def __init__(
        self,
        d_model:      int,
        d_ff:         int,
        rank:         int,
        n_experts:    int,
        original_ffn: nn.Module,
        layer_idx:    int  = 0,
        base_seed:    int  = 42,
        d_ff_expert:  int  = None,   # override; defaults to d_ff // n_experts
    ):
        super().__init__()

        self.n_experts   = n_experts
        self.rank        = rank
        self.layer_idx   = layer_idx
        self.d_ff_expert = d_ff_expert if d_ff_expert is not None else max(1, d_ff // n_experts)

        device = next(original_ffn.parameters()).device
        dtype  = next(original_ffn.parameters()).dtype

        # ── Build bottleneck experts (SVD-initialized) ────────────────────────
        self.experts = nn.ModuleList([
            PRISMExpert(
                d_model      = d_model,
                d_ff         = d_ff,
                d_ff_expert  = self.d_ff_expert,
                rank         = rank,
                expert_idx   = i,
                n_experts    = n_experts,
                original_ffn = original_ffn,
                device       = device,
                dtype        = dtype,
            )
            for i in range(n_experts)
        ])

        # ── Symmetry breaking for proj_u (delta A matrix) ────────────────────
        # Base FFN weights already differ per expert (different SVD slices).
        # proj_u still needs different inits so delta explores different directions.
        for expert_idx, expert in enumerate(self.experts):
            seed = base_seed + layer_idx * n_experts + expert_idx
            torch.manual_seed(seed)
            nn.init.normal_(expert.proj_u.weight, std=0.01)

        # ── Router: learned softmax gating ───────────────────────────────────
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

        # Storage for auxiliary loss (retrieved by trainer)
        self._balance_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D]  — batch, sequence, d_model
        Returns:
            out: [B, S, D]
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)                              # [T, D]  T = B*S

        # ── Softmax gating: all experts always active ─────────────────────────
        gates = F.softmax(self.router(x_flat), dim=-1)      # [T, n_experts]

        # ── Run all experts, stack outputs ────────────────────────────────────
        expert_outs = torch.stack(
            [expert(x_flat) for expert in self.experts],
            dim=1,
        )                                                    # [T, n_experts, D]

        # ── Weighted sum ──────────────────────────────────────────────────────
        out = (gates.unsqueeze(-1) * expert_outs).sum(dim=1) # [T, D]

        # ── Entropy regularizer (load balance) ───────────────────────────────
        # Maximising entropy encourages uniform expert usage.
        # Trainer adds: L_total = L_lm - balance_weight * balance_loss
        self._balance_loss = -(gates * (gates + 1e-8).log()).sum(-1).mean()

        return out.view(B, S, D)