"""
Training Integration Example: Diffusion Policy with SA-Topo Loss
=================================================================

This module demonstrates how to integrate the State-Aware Behavioral Topology
Loss (SA-Topo Loss) into a standard Diffusion Policy training loop for
multi-objective robot behaviour.

Architecture overview
---------------------
The policy network (SimpleDiffusionPolicy) takes as input:
  - The current physical state  s  (e.g. joint positions + velocities)
  - The preference vector       w  (e.g. [speed_weight, energy_weight])
  - A noise level               t  (diffusion timestep)
and predicts the denoised action trajectory a_pred.

During training we add the SA-Topo Loss on top of the standard
Behavioral-Cloning (BC) MSE loss so that the policy learns a
preference-consistent action manifold.

Usage
-----
    python train_example.py

You should see training loss values printed every 50 steps and a summary of
the SA-Topo diagnostics.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sa_topo_loss import SATopoLoss


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def make_synthetic_dataset(
    n_samples: int = 2048,
    state_dim: int = 10,
    pref_dim: int = 2,
    action_dim: int = 20,
) -> TensorDataset:
    """Generate a synthetic multi-objective robot dataset.

    Ground-truth relationship: the action is a linear function of both the
    state and the preference so the optimal policy *should* respect the
    topological ordering of preferences.

    a = W_s · s + W_w · w + noise
    """
    torch.manual_seed(SEED)
    states = torch.randn(n_samples, state_dim)
    prefs = torch.rand(n_samples, pref_dim)          # preferences in [0, 1]
    prefs = prefs / prefs.sum(dim=-1, keepdim=True)  # normalise to simplex

    # Fixed random linear maps (ground truth)
    W_s = torch.randn(state_dim, action_dim) * 0.3
    W_w = torch.randn(pref_dim,  action_dim) * 1.0   # preference has large influence
    noise = torch.randn(n_samples, action_dim) * 0.05

    actions = states @ W_s + prefs @ W_w + noise
    return TensorDataset(states, prefs, actions)


# ---------------------------------------------------------------------------
# Simple Diffusion Policy backbone
# ---------------------------------------------------------------------------

class SimpleDiffusionPolicy(nn.Module):
    """A lightweight MLP that mimics the score network of a Diffusion Policy.

    Inputs  : [state | preference | noisy_action | diffusion_timestep_emb]
    Output  : denoised action prediction
    """

    def __init__(
        self,
        state_dim: int = 10,
        pref_dim: int = 2,
        action_dim: int = 20,
        hidden_dim: int = 256,
        n_layers: int = 4,
        max_timesteps: int = 100,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.max_timesteps = max_timesteps

        # Sinusoidal timestep embedding
        self.t_embed_dim = 32
        input_dim = state_dim + pref_dim + action_dim + self.t_embed_dim

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional embedding for diffusion timestep *t*."""
        half = self.t_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        preference: torch.Tensor,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self._timestep_embedding(t)
        x = torch.cat([state, preference, noisy_action, t_emb], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Diffusion helpers (simplified DDPM)
# ---------------------------------------------------------------------------

class SimpleDDPM:
    """Minimal DDPM scheduler (linear beta schedule)."""

    def __init__(self, n_timesteps: int = 100, device: torch.device = torch.device("cpu")) -> None:
        self.T = n_timesteps
        betas = torch.linspace(1e-4, 0.02, n_timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)."""
        s_a = self.sqrt_alphas_cumprod[t].view(-1, 1)
        s_b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return s_a * x0 + s_b * noise


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-4,
    lambda_topo: float = 0.1,
    tau: float = 1.0,
    device_str: str = "cpu",
) -> SimpleDiffusionPolicy:
    """Train a SimpleDiffusionPolicy with SA-Topo Loss and return the model."""
    device = torch.device(device_str)

    # --- Dataset -----------------------------------------------------------
    dataset = make_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- Model & optimizer -------------------------------------------------
    policy = SimpleDiffusionPolicy().to(device)
    ddpm = SimpleDDPM(device=device)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # --- Loss function -----------------------------------------------------
    loss_fn = SATopoLoss(tau=tau, lambda_topo=lambda_topo)

    print(f"Training for {n_epochs} epochs  |  batch_size={batch_size}  "
          f"|  λ_topo={lambda_topo}  |  τ={tau}")
    print("-" * 70)

    global_step = 0
    for epoch in range(1, n_epochs + 1):
        epoch_bc = 0.0
        epoch_topo = 0.0
        epoch_total = 0.0
        n_batches = 0

        for states, prefs, actions_gt in loader:
            states, prefs, actions_gt = (
                states.to(device),
                prefs.to(device),
                actions_gt.to(device),
            )
            N = states.size(0)

            # Sample random diffusion timesteps
            t = torch.randint(0, ddpm.T, (N,), device=device)
            noise = torch.randn_like(actions_gt)
            noisy_actions = ddpm.add_noise(actions_gt, noise, t)

            # Policy forward pass
            actions_pred = policy(states, prefs, noisy_actions, t)

            # SA-Topo Loss
            total, l_bc, l_topo = loss_fn(
                states=states,
                preferences=prefs,
                actions_pred=actions_pred,
                actions_gt=actions_gt,
            )

            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_bc    += l_bc.item()
            epoch_topo  += l_topo.item()
            epoch_total += total.item()
            n_batches   += 1
            global_step += 1

            if global_step % 50 == 0:
                diag = loss_fn.diagnostics(states, prefs, actions_pred)
                print(
                    f"  step {global_step:5d} | "
                    f"total={total.item():.4f}  "
                    f"bc={l_bc.item():.4f}  "
                    f"topo={l_topo.item():.4f}  "
                    f"gate_active={diag['gate_active_fraction']:.2%}  "
                    f"corr(D_A,D_W)={diag['da_dw_correlation']:.3f}"
                )

        scheduler.step()
        print(
            f"Epoch {epoch:2d}/{n_epochs} | "
            f"avg_total={epoch_total/n_batches:.4f}  "
            f"avg_bc={epoch_bc/n_batches:.4f}  "
            f"avg_topo={epoch_topo/n_batches:.4f}"
        )

    print("-" * 70)
    print("Training complete.")
    return policy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trained_policy = train(
        n_epochs=20,
        batch_size=128,
        lambda_topo=0.1,
        tau=1.0,
        device_str="cpu",
    )
