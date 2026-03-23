# State-Aware Behavioral Topology Loss (SA-Topo Loss)

A PyTorch implementation of the **State-Aware Behavioral Topology Loss
(SA-Topo Loss)** for training multi-objective robot behavior policies.

SA-Topo Loss is grounded in the PSTD (Pareto-Structured Topology Descent)
framework and can be plugged into any policy architecture — MLP, Transformer,
or Diffusion Policy — without architectural changes.

---

## Background

In multi-objective robot learning, a policy must produce actions that vary
*smoothly* and *consistently* as user preferences change. Without an explicit
structural constraint, a policy may learn a valid solution for each individual
preference vector while still having an **arbitrary and potentially
discontinuous action manifold** across the preference space.

SA-Topo Loss directly penalises topological violations by requiring:

> **Given the same physical state, action trajectories must preserve the
> topological ordering of the preference space.**

---

## Mathematical Formulation

Given a training batch of *N* samples, each described by:
- **sᵢ** — physical state (e.g. robot joint angles + velocities)
- **wᵢ** — preference / reward-weight vector
- **aᵢ** — action trajectory

The loss is computed as follows:

### 1. State-similarity gate (temperature τ)

$$W_S(i,j) = \exp\!\left(-\frac{\lVert s_i - s_j \rVert_2^2}{\tau}\right)$$

### 2. Pairwise distances

$$D_W(i,j) = \lVert w_i - w_j \rVert_2^2 \qquad D_A(i,j) = \lVert a_i - a_j \rVert_2^2$$

### 3. Min-max normalisation

$$\tilde{D}(i,j) = \frac{D(i,j) - \min D}{\max D - \min D + \varepsilon}$$

### 4. SA-Topo alignment loss

$$\mathcal{L}_{\text{SA-Topo}} = \frac{1}{N^2} \sum_{i,j} W_S(i,j) \cdot \left|\tilde{D}_A(i,j) - \tilde{D}_W(i,j)\right|^2$$

### 5. Total training objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BC}} + \lambda_{\text{topo}} \cdot \mathcal{L}_{\text{SA-Topo}}$$

where $\mathcal{L}_{\text{BC}}$ is the standard Behavioral-Cloning MSE loss.

---

## File Structure

| File | Description |
|------|-------------|
| `sa_topo_loss.py` | Core SA-Topo Loss implementation (`SATopoLoss` class) |
| `train_example.py` | Diffusion Policy training loop with SA-Topo Loss |
| `analysis.py` | Effectiveness analysis: TCS, TVR, λ_topo sweep, and limitations |
| `tests/test_sa_topo_loss.py` | 28 unit tests |

---

## Quick Start

```python
import torch
from sa_topo_loss import SATopoLoss

loss_fn = SATopoLoss(tau=1.0, lambda_topo=0.1)

states       = torch.randn(32, 10)   # batch of 32, state dim = 10
preferences  = torch.rand(32, 2)     # 2-dim preference vector (normalised)
preferences  = preferences / preferences.sum(dim=-1, keepdim=True)
actions_pred = torch.randn(32, 20)   # predicted action trajectory
actions_gt   = torch.randn(32, 20)   # ground-truth action trajectory

total, l_bc, l_topo = loss_fn(states, preferences, actions_pred, actions_gt)
total.backward()
```

### Training loop integration

```python
for states, prefs, actions_gt in dataloader:
    actions_pred = policy(states, prefs, ...)
    total, l_bc, l_topo = loss_fn(states, prefs, actions_pred, actions_gt)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    # Optional diagnostics
    diag = loss_fn.diagnostics(states, prefs, actions_pred)
    print(diag["da_dw_correlation"])  # should increase during training
```

Run the full training demo:

```bash
python train_example.py
```

---

## Hyperparameter Guide

| Parameter | Default | Effect |
|-----------|---------|--------|
| `tau` | `1.0` | Gate temperature. Set to ≈ median squared state distance in your dataset. Lower → sharper gate (fewer active pairs). |
| `lambda_topo` | `0.1` | Topology loss weight. Start small (0.01–0.1) and increase if TCS plateaus. |

---

## Analysis Metrics

Run `python analysis.py` to see:

- **TCS (Topological Consistency Score)** — Spearman rank correlation between
  $D_W$ and $D_A$ over gate-active pairs. Should increase toward +1 during training.
- **TVR (Topology Violation Rate)** — Fraction of active pairs with large misalignment.
  Should decrease toward 0.
- **λ_topo sweep** — Trade-off table between BC loss and TCS.

Sample output on a well-aligned policy:

```
[Good policy — actions aligned with preferences]
  TCS (Spearman corr) = 0.996  (expected ≈ +1)
  TVR (violation rate)= 0.000  (expected ≈  0)

[Bad policy — actions ignore preferences]
  TCS (Spearman corr) = 0.017  (expected ≈  0)
```

---

## Effectiveness Analysis

**Why SA-Topo Loss works:**

1. **Principled inductive bias** — Directly penalises topological violations
   instead of relying on implicit manifold learning.
2. **State gating prevents false alignment** — Only *comparable* samples
   (similar physical states) contribute to the alignment loss.
3. **Architecture-agnostic** — Works with any policy that predicts action
   vectors.
4. **Scalable** — O(N² · D) pairwise computation using batched matrix
   operations.

**Known limitations:**

1. **Batch-level supervision** — The topology constraint is only enforced
   within each mini-batch. Use a *preference-aware sampler* to construct
   batches with diverse preferences but similar states for stronger signal.
2. **Temperature sensitivity** — Tune `tau` based on the typical
   inter-sample state distance in your dataset.
3. **Normalisation instability** — Min-max normalisation is sensitive to
   outliers; consider percentile clipping for more robust training.
4. **Action representation** — Normalise heterogeneous action dimensions
   before computing distances.
5. **Theoretical gap** — PSTD theory guarantees hold for a known Pareto
   front; during training the front is estimated, so consistency is
   empirical rather than formally guaranteed.

---

## Testing

```bash
python -m pytest tests/test_sa_topo_loss.py -v
```

All 28 tests should pass.