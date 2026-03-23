"""
Effectiveness Analysis of SA-Topo Loss
=======================================

This module provides quantitative tools to assess whether the SA-Topo Loss
successfully enforces the desired topological ordering of action trajectories
in preference space, and to analyse the trade-offs introduced by the method.

Metrics
-------
1. **Topological Consistency Score (TCS)**
   Given a batch, we measure the Spearman rank correlation between
   preference distances D_W and action distances D_A, weighted by the
   state-similarity gate W_S.  A TCS close to +1 means the policy has
   learned a well-ordered action manifold.

2. **Topology Violation Rate (TVR)**
   Fraction of high-gate pairs (W_S > threshold) where D_A and D_W are
   significantly misaligned (|D̃_A - D̃_W| > δ).

3. **Gate Sparsity**
   Mean value of the state-similarity gate W_S across the batch.  High
   sparsity (low mean W_S) indicates that the training batches contain
   samples with diverse states, which limits the effective supervision
   signal of the topology loss — see the Analysis section below.

4. **BC vs. Topo Loss Trade-off**
   We sweep λ_topo and record both the final BC loss and the TCS to help
   practitioners choose a good value of λ_topo.

Analysis of Effectiveness and Limitations
------------------------------------------
**Why SA-Topo Loss is effective**:

A. *Principled inductive bias*: Multi-objective policies must produce actions
   that vary smoothly as preferences change.  Without an explicit constraint,
   a policy may learn a valid solution for each individual preference but with
   an arbitrary (and potentially discontinuous) action manifold.  SA-Topo Loss
   directly penalises topological violations, turning an implicit desideratum
   into an explicit regulariser.

B. *State gating prevents false alignment*: In multi-task robotics, similar
   actions under different states have no bearing on preference-consistency.
   The state-similarity gate W_S = exp(-‖s_i-s_j‖²/τ) ensures that only
   *comparable* samples (those in approximately the same physical state)
   contribute to the alignment loss.  Without this gating, the loss would be
   misleading and harmful.

C. *Compatibility with any base policy*: Because SA-Topo Loss only requires
   pairwise distances between (state, preference, action) tuples, it can be
   added to any policy architecture — MLP, Transformer, Diffusion Policy —
   without architectural changes.

D. *Scalable to large batches*: The N×N distance matrices can be computed
   entirely in closed-form using batched matrix operations (O(N² · D) time),
   which is affordable for typical batch sizes (N ≤ 512).

**Known limitations and open questions**:

1. *Batch-level supervision*: The topology constraint is only enforced within
   each mini-batch.  If the batch does not contain samples with similar states
   and diverse preferences, the loss gradient is near zero and provides no
   useful signal.  *Mitigation*: use a **preference-aware sampler** that
   actively constructs batches with diverse preferences but similar states.

2. *Temperature sensitivity*: The gate temperature τ controls the locality of
   the topology constraint.  Too large τ → all pairs are penalised, including
   incomparable (different-state) pairs.  Too small τ → almost no pairs
   contribute.  Practitioners should tune τ based on the typical inter-sample
   state distance in their dataset (a useful heuristic: set τ ≈ median
   squared state distance).

3. *Normalisation instability*: Min-max normalisation of D_W and D_A is
   sensitive to outliers and can collapse when all distances in a batch are
   very similar.  The ε term in the denominator prevents division by zero but
   does not fully mitigate this.  *Mitigation*: use percentile clipping before
   normalisation.

4. *Action representation*: D_A is computed on flattened action vectors.  If
   the action space has heterogeneous dimensions (e.g., joint angles + gripper
   force), the loss may be dominated by high-variance dimensions.
   *Mitigation*: normalise each action dimension independently before
   computing distances.

5. *Theoretical gap*: PSTD theory guarantees topological consistency for a
   fixed, known Pareto front.  During training the front is estimated and
   changes with each gradient step.  The emergent consistency is empirical
   rather than formally guaranteed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sa_topo_loss import SATopoLoss, _minmax_normalise, _pairwise_sq_dist


# ---------------------------------------------------------------------------
# Topological Consistency Score (Spearman, gate-weighted)
# ---------------------------------------------------------------------------

def topological_consistency_score(
    states: torch.Tensor,
    preferences: torch.Tensor,
    actions: torch.Tensor,
    tau: float = 1.0,
    gate_threshold: float = 0.0,
) -> float:
    """Compute the gate-weighted Spearman rank correlation between D_W and D_A.

    Parameters
    ----------
    states      : Tensor (N, S)
    preferences : Tensor (N, P)
    actions     : Tensor (N, A)
    tau         : temperature for the state-similarity gate
    gate_threshold : only pairs with W_S > threshold are included

    Returns
    -------
    rho : float in [-1, 1].  +1 = perfect topological consistency.
    """
    with torch.no_grad():
        d_w = _pairwise_sq_dist(preferences).flatten()
        d_a = _pairwise_sq_dist(actions).flatten()
        w_s = torch.exp(-_pairwise_sq_dist(states) / tau).flatten()

    mask = (w_s > gate_threshold).cpu().numpy()
    d_w_np = d_w.cpu().numpy()[mask]
    d_a_np = d_a.cpu().numpy()[mask]

    if len(d_w_np) < 2:
        return float("nan")

    # Spearman via rank correlation
    def _rank(x: np.ndarray) -> np.ndarray:
        order = x.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))
        return ranks

    r_w = _rank(d_w_np)
    r_a = _rank(d_a_np)
    n = len(r_w)
    cov = np.mean((r_w - r_w.mean()) * (r_a - r_a.mean()))
    std_w = r_w.std() + 1e-8
    std_a = r_a.std() + 1e-8
    return float(cov / (std_w * std_a))


# ---------------------------------------------------------------------------
# Topology Violation Rate
# ---------------------------------------------------------------------------

def topology_violation_rate(
    states: torch.Tensor,
    preferences: torch.Tensor,
    actions: torch.Tensor,
    tau: float = 1.0,
    gate_threshold: float = 0.5,
    alignment_delta: float = 0.2,
) -> float:
    """Fraction of high-gate pairs that are significantly misaligned.

    A pair (i,j) is a *violation* if W_S(i,j) > gate_threshold and
    |D̃_A(i,j) - D̃_W(i,j)| > alignment_delta.

    Parameters
    ----------
    gate_threshold   : minimum W_S to consider a pair active
    alignment_delta  : maximum allowed normalised misalignment

    Returns
    -------
    rate : float in [0, 1]
    """
    with torch.no_grad():
        d_w = _minmax_normalise(_pairwise_sq_dist(preferences))
        d_a = _minmax_normalise(_pairwise_sq_dist(actions))
        w_s = torch.exp(-_pairwise_sq_dist(states) / tau)

        active = (w_s > gate_threshold)
        if active.sum() == 0:
            return float("nan")

        misaligned = (d_a - d_w).abs() > alignment_delta
        return (active & misaligned).float().sum().item() / active.float().sum().item()


# ---------------------------------------------------------------------------
# λ_topo sweep utility
# ---------------------------------------------------------------------------

def sweep_lambda_topo(
    states: torch.Tensor,
    preferences: torch.Tensor,
    actions_pred: torch.Tensor,
    actions_gt: torch.Tensor,
    lambdas: list[float] | None = None,
    tau: float = 1.0,
) -> list[dict[str, Any]]:
    """Evaluate SA-Topo and BC losses for a range of λ_topo values.

    This is useful to understand the trade-off between BC fidelity and
    topological alignment at inference time (i.e., given a fixed trained
    policy).  To study the trade-off *during* training, retrain the policy
    with each λ_topo value.

    Parameters
    ----------
    lambdas : list of λ_topo values to evaluate (default: [0, 0.01, …, 1.0])

    Returns
    -------
    results : list of dicts with keys
        lambda_topo, l_bc, l_topo, total, tcs, tvr
    """
    if lambdas is None:
        lambdas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    results = []
    for lam in lambdas:
        loss_fn = SATopoLoss(tau=tau, lambda_topo=lam)
        with torch.no_grad():
            total, l_bc, l_topo = loss_fn(states, preferences, actions_pred, actions_gt)
        tcs = topological_consistency_score(states, preferences, actions_pred, tau=tau)
        tvr = topology_violation_rate(states, preferences, actions_pred, tau=tau)
        results.append({
            "lambda_topo": lam,
            "l_bc":        l_bc.item(),
            "l_topo":      l_topo.item(),
            "total":       total.item(),
            "tcs":         tcs,
            "tvr":         tvr,
        })
    return results


# ---------------------------------------------------------------------------
# Pretty-print a sweep result
# ---------------------------------------------------------------------------

def print_sweep_results(results: list[dict[str, Any]]) -> None:
    """Print λ_topo sweep results as a formatted table."""
    header = f"{'λ_topo':>8} | {'L_BC':>8} | {'L_topo':>8} | {'Total':>8} | {'TCS':>6} | {'TVR':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        tcs = f"{r['tcs']:.3f}" if not np.isnan(r["tcs"]) else "  nan"
        tvr = f"{r['tvr']:.3f}" if not np.isnan(r["tvr"]) else "  nan"
        print(
            f"{r['lambda_topo']:>8.3f} | "
            f"{r['l_bc']:>8.4f} | "
            f"{r['l_topo']:>8.4f} | "
            f"{r['total']:>8.4f} | "
            f"{tcs:>6} | "
            f"{tvr:>6}"
        )


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    N = 64

    print("=" * 60)
    print("SA-Topo Loss Effectiveness Analysis")
    print("=" * 60)

    # Simulate a policy that has learned a *good* topological ordering
    # (actions are a noisy linear function of preferences)
    states = torch.randn(N, 10)
    prefs  = torch.rand(N, 2)
    prefs  = prefs / prefs.sum(dim=-1, keepdim=True)

    # Good policy: actions proportional to preference
    W = torch.randn(2, 20)
    actions_good = prefs @ W + torch.randn(N, 20) * 0.05
    actions_gt   = prefs @ W

    # Bad policy: actions ignore preferences (random)
    actions_bad  = torch.randn(N, 20)

    print("\n[Good policy — actions aligned with preferences]")
    tcs_good = topological_consistency_score(states, prefs, actions_good, tau=2.0)
    tvr_good = topology_violation_rate(states, prefs, actions_good, tau=2.0)
    print(f"  TCS (Spearman corr) = {tcs_good:.3f}  (expected ≈ +1)")
    print(f"  TVR (violation rate)= {tvr_good:.3f}  (expected ≈  0)")

    print("\n[Bad policy — actions ignore preferences]")
    tcs_bad = topological_consistency_score(states, prefs, actions_bad, tau=2.0)
    tvr_bad = topology_violation_rate(states, prefs, actions_bad, tau=2.0)
    print(f"  TCS (Spearman corr) = {tcs_bad:.3f}  (expected ≈  0)")
    print(f"  TVR (violation rate)= {tvr_bad:.3f}  (expected > 0)")

    print("\n[λ_topo sweep — using good policy]")
    sweep_res = sweep_lambda_topo(states, prefs, actions_good, actions_gt, tau=2.0)
    print_sweep_results(sweep_res)

    print("\n[SA-Topo Loss diagnostics — good vs. bad policy]")
    loss_fn = SATopoLoss(tau=2.0, lambda_topo=0.1)
    diag_good = loss_fn.diagnostics(states, prefs, actions_good)
    diag_bad  = loss_fn.diagnostics(states, prefs, actions_bad)
    print(f"  Good policy: {diag_good}")
    print(f"  Bad  policy: {diag_bad}")
