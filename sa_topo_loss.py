"""
State-Aware Behavioral Topology Loss (SA-Topo Loss)
====================================================

Implements the SA-Topo Loss for training multi-objective robot behavior policies
(e.g., Diffusion Policy) based on the PSTD (Pareto-Structured Topology Descent)
framework.

Core Idea
---------
Instead of requiring neural-network *weights* to form a manifold, we require the
robot's *physical behavior trajectories* to form a manifold.  Given the same
physical state S, trajectories generated under preferences w_1 and w_2 must
preserve the topological ordering of the preference space:

    D_action(traj_1, traj_2)  ∝  D_pref(w_1, w_2)

The state-similarity matrix acts as a **gating signal**: only pairs of samples
that share a similar physical state are required to satisfy the topological
alignment constraint.  Pairs with very different states are excluded from the
loss computation, because different states may legitimately produce very
different actions regardless of preference.

Mathematical Formulation
------------------------
Given a training batch of N samples, each described by
  - s_i  : physical state (e.g. robot joint angles + velocities)
  - w_i  : preference / reward weight vector
  - a_i  : action trajectory

1. State-similarity gate (temperature τ):
       W_S(i,j) = exp( -‖s_i - s_j‖²₂ / τ )

2. Preference distance:
       D_W(i,j) = ‖w_i - w_j‖²₂

3. Action-trajectory distance:
       D_A(i,j) = ‖a_i - a_j‖²₂

4. Min-max normalisation (over the batch) to make scales comparable:
       D̃(i,j) = (D(i,j) - min D) / (max D - min D + ε)

5. SA-Topo alignment loss:
       L_SA-Topo = (1/N²) Σ_{i,j} W_S(i,j) · |D̃_A(i,j) - D̃_W(i,j)|²

6. Total loss:
       L_total = L_BC + λ_topo · L_SA-Topo

where L_BC is the standard Behavioral-Cloning MSE loss.

References
----------
- PSTD: Pareto-Structured Topology Descent for multi-objective policy learning.
- Diffusion Policy: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via
  Action Diffusion", RSS 2023.

Spelling note: American English is used throughout ("Behavioral-Cloning").
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: pairwise squared Euclidean distance matrix
# ---------------------------------------------------------------------------

def _pairwise_sq_dist(x: torch.Tensor) -> torch.Tensor:
    """Return the N×N matrix of squared L2 distances between rows of *x*.

    Parameters
    ----------
    x : Tensor of shape (N, D)

    Returns
    -------
    dist : Tensor of shape (N, N)
        dist[i, j] = ‖x_i - x_j‖²₂
    """
    # ‖x_i - x_j‖² = ‖x_i‖² + ‖x_j‖² - 2 x_i·x_j
    sq_norms = (x * x).sum(dim=-1)               # (N,)
    dist = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * x @ x.T
    # Clamp numerical noise to zero
    return dist.clamp(min=0.0)


# ---------------------------------------------------------------------------
# Utility: min-max normalisation of a distance matrix
# ---------------------------------------------------------------------------

def _minmax_normalise(d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise a distance matrix to [0, 1] using min-max scaling.

    Parameters
    ----------
    d   : Tensor of shape (N, N)
    eps : small constant to avoid division by zero

    Returns
    -------
    d_norm : Tensor of shape (N, N), values in [0, 1]
    """
    d_min = d.min()
    d_max = d.max()
    return (d - d_min) / (d_max - d_min + eps)


# ---------------------------------------------------------------------------
# Main loss class
# ---------------------------------------------------------------------------

class SATopoLoss(nn.Module):
    """State-Aware Behavioral Topology Loss (SA-Topo Loss).

    Parameters
    ----------
    tau : float
        Temperature for the state-similarity gate.  Smaller values make the
        gate sharper (only very similar states contribute).  Default: 1.0.
    lambda_topo : float
        Weight of the topology loss relative to the BC loss.  Default: 0.1.
    eps : float
        Small constant for numerical stability in normalisation.  Default: 1e-8.

    Examples
    --------
    >>> loss_fn = SATopoLoss(tau=0.5, lambda_topo=0.1)
    >>> states      = torch.randn(16, 10)   # batch of 16, state dim = 10
    >>> preferences = torch.randn(16,  2)   # 2-dim preference vector
    >>> actions_pred = torch.randn(16, 20)  # predicted action trajectory
    >>> actions_gt   = torch.randn(16, 20)  # ground-truth action trajectory
    >>> total, bc, topo = loss_fn(states, preferences, actions_pred, actions_gt)
    """

    def __init__(
        self,
        tau: float = 1.0,
        lambda_topo: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if lambda_topo < 0:
            raise ValueError(f"lambda_topo must be non-negative, got {lambda_topo}")
        self.tau = tau
        self.lambda_topo = lambda_topo
        self.eps = eps

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        states: torch.Tensor,
        preferences: torch.Tensor,
        actions_pred: torch.Tensor,
        actions_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total, BC, and SA-Topo losses.

        Parameters
        ----------
        states : Tensor of shape (N, S)
            Physical states for each sample in the batch.
        preferences : Tensor of shape (N, P)
            Preference / reward-weight vectors.
        actions_pred : Tensor of shape (N, A)
            Actions predicted by the policy.
        actions_gt : Tensor of shape (N, A)
            Ground-truth demonstration actions (used for BC loss).

        Returns
        -------
        total : scalar Tensor
            L_total = L_BC + λ_topo · L_SA-Topo
        l_bc : scalar Tensor
            Behavioral-Cloning MSE loss.
        l_topo : scalar Tensor
            SA-Topo alignment loss.
        """
        # ---- Behavioral-Cloning MSE loss --------------------------------
        l_bc = F.mse_loss(actions_pred, actions_gt)

        # ---- SA-Topo loss ------------------------------------------------
        l_topo = self.compute_sa_topo_loss(states, preferences, actions_pred)

        total = l_bc + self.lambda_topo * l_topo
        return total, l_bc, l_topo

    # ------------------------------------------------------------------
    # SA-Topo loss (can also be used stand-alone)
    # ------------------------------------------------------------------

    def compute_sa_topo_loss(
        self,
        states: torch.Tensor,
        preferences: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute only the SA-Topo alignment term.

        Parameters
        ----------
        states      : Tensor (N, S)
        preferences : Tensor (N, P)
        actions     : Tensor (N, A)

        Returns
        -------
        l_topo : scalar Tensor
        """
        # 1. State-similarity gate
        d_s = _pairwise_sq_dist(states)                           # (N, N)
        w_s = torch.exp(-d_s / self.tau)                          # (N, N)

        # 2. Preference & action pairwise distances
        d_w = _pairwise_sq_dist(preferences)                      # (N, N)
        d_a = _pairwise_sq_dist(actions)                          # (N, N)

        # 3. Normalise distances to [0, 1]
        d_w_norm = _minmax_normalise(d_w, self.eps)               # (N, N)
        d_a_norm = _minmax_normalise(d_a, self.eps)               # (N, N)

        # 4. Weighted alignment loss
        alignment_sq = (d_a_norm - d_w_norm).pow(2)               # (N, N)
        l_topo = (w_s * alignment_sq).mean()                      # scalar

        return l_topo

    # ------------------------------------------------------------------
    # Diagnostics (useful during training)
    # ------------------------------------------------------------------

    def diagnostics(
        self,
        states: torch.Tensor,
        preferences: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, float]:
        """Return a dictionary of diagnostic scalar values.

        Useful for logging during training without requiring gradient tape.
        """
        with torch.no_grad():
            d_s = _pairwise_sq_dist(states)
            w_s = torch.exp(-d_s / self.tau)

            d_w = _pairwise_sq_dist(preferences)
            d_a = _pairwise_sq_dist(actions)

            d_w_norm = _minmax_normalise(d_w, self.eps)
            d_a_norm = _minmax_normalise(d_a, self.eps)

            alignment_sq = (d_a_norm - d_w_norm).pow(2)
            l_topo = (w_s * alignment_sq).mean()

            # Fraction of pairs that have high state similarity (gate > 0.5)
            gate_active = (w_s > 0.5).float().mean()

            # Pearson correlation between D_A and D_W over active pairs
            mask = (w_s > 0.5).float()
            n_pairs = mask.sum().item()
            if n_pairs > 1:
                da_flat = (d_a_norm * mask).flatten()
                dw_flat = (d_w_norm * mask).flatten()
                da_mean = da_flat.sum() / n_pairs
                dw_mean = dw_flat.sum() / n_pairs
                cov = ((da_flat - da_mean) * (dw_flat - dw_mean)).sum() / n_pairs
                std_a = ((da_flat - da_mean).pow(2).sum() / n_pairs).sqrt()
                std_w = ((dw_flat - dw_mean).pow(2).sum() / n_pairs).sqrt()
                corr = (cov / (std_a * std_w + self.eps)).item()
            else:
                corr = float("nan")

        return {
            "l_topo": l_topo.item(),
            "gate_active_fraction": gate_active.item(),
            "da_dw_correlation": corr,
            "mean_state_similarity": w_s.mean().item(),
        }
