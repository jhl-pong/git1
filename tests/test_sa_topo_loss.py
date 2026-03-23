"""
Unit tests for the SA-Topo Loss implementation.

Run with:
    python -m pytest tests/test_sa_topo_loss.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from sa_topo_loss import SATopoLoss, _minmax_normalise, _pairwise_sq_dist
from analysis import topological_consistency_score, topology_violation_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(N: int = 8, S: int = 4, P: int = 2, A: int = 6, seed: int = 0):
    """Return (states, preferences, actions) tensors for testing."""
    g = torch.Generator()
    g.manual_seed(seed)
    states = torch.randn(N, S, generator=g)
    prefs  = torch.rand(N, P, generator=g)
    prefs  = prefs / prefs.sum(dim=-1, keepdim=True)
    actions = torch.randn(N, A, generator=g)
    return states, prefs, actions


# ---------------------------------------------------------------------------
# _pairwise_sq_dist
# ---------------------------------------------------------------------------

class TestPairwiseSqDist:
    def test_shape(self):
        x = torch.randn(5, 3)
        d = _pairwise_sq_dist(x)
        assert d.shape == (5, 5)

    def test_diagonal_zero(self):
        x = torch.randn(6, 4)
        d = _pairwise_sq_dist(x)
        assert torch.allclose(d.diagonal(), torch.zeros(6), atol=1e-5)

    def test_symmetry(self):
        x = torch.randn(7, 3)
        d = _pairwise_sq_dist(x)
        assert torch.allclose(d, d.T, atol=1e-5)

    def test_non_negative(self):
        x = torch.randn(8, 5)
        d = _pairwise_sq_dist(x)
        assert (d >= 0).all()

    def test_known_value(self):
        # Two vectors: [0, 0] and [3, 4] → squared distance = 25
        x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        d = _pairwise_sq_dist(x)
        assert math.isclose(d[0, 1].item(), 25.0, rel_tol=1e-5)
        assert math.isclose(d[1, 0].item(), 25.0, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# _minmax_normalise
# ---------------------------------------------------------------------------

class TestMinMaxNormalise:
    def test_output_range(self):
        d = torch.rand(6, 6) * 10
        d_norm = _minmax_normalise(d)
        assert d_norm.min().item() >= 0.0 - 1e-6
        assert d_norm.max().item() <= 1.0 + 1e-6

    def test_constant_matrix(self):
        # All same value → normalised to 0 (numerator is 0, denom has eps)
        d = torch.ones(4, 4) * 5.0
        d_norm = _minmax_normalise(d)
        assert torch.allclose(d_norm, torch.zeros(4, 4), atol=1e-5)

    def test_shape_preserved(self):
        d = torch.rand(5, 5)
        assert _minmax_normalise(d).shape == (5, 5)


# ---------------------------------------------------------------------------
# SATopoLoss — construction
# ---------------------------------------------------------------------------

class TestSATopoLossConstruction:
    def test_default_params(self):
        loss = SATopoLoss()
        assert loss.tau == 1.0
        assert loss.lambda_topo == 0.1

    def test_custom_params(self):
        loss = SATopoLoss(tau=0.5, lambda_topo=0.2)
        assert loss.tau == 0.5
        assert loss.lambda_topo == 0.2

    def test_invalid_tau(self):
        with pytest.raises(ValueError, match="tau"):
            SATopoLoss(tau=0.0)
        with pytest.raises(ValueError, match="tau"):
            SATopoLoss(tau=-1.0)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError, match="lambda_topo"):
            SATopoLoss(lambda_topo=-0.1)


# ---------------------------------------------------------------------------
# SATopoLoss — forward pass
# ---------------------------------------------------------------------------

class TestSATopoLossForward:
    def test_output_types(self):
        loss_fn = SATopoLoss()
        s, w, a = _make_batch()
        total, l_bc, l_topo = loss_fn(s, w, a, a.clone())
        assert total.shape == ()       # scalar
        assert l_bc.shape == ()
        assert l_topo.shape == ()

    def test_total_equals_bc_plus_topo(self):
        loss_fn = SATopoLoss(lambda_topo=0.3)
        s, w, a_pred = _make_batch(seed=1)
        _, _, a_gt = _make_batch(seed=2)
        total, l_bc, l_topo = loss_fn(s, w, a_pred, a_gt)
        expected = l_bc + 0.3 * l_topo
        assert torch.allclose(total, expected, atol=1e-6)

    def test_bc_loss_is_zero_when_perfect(self):
        """BC loss must be 0 when predicted == ground-truth actions."""
        loss_fn = SATopoLoss()
        s, w, a = _make_batch()
        _, l_bc, _ = loss_fn(s, w, a, a)
        assert l_bc.item() < 1e-6

    def test_non_negative_losses(self):
        loss_fn = SATopoLoss()
        s, w, a_pred = _make_batch(seed=3)
        _, _, a_gt = _make_batch(seed=4)
        total, l_bc, l_topo = loss_fn(s, w, a_pred, a_gt)
        assert total.item() >= 0.0
        assert l_bc.item() >= 0.0
        assert l_topo.item() >= 0.0

    def test_gradients_flow(self):
        """Ensure that gradients propagate through the total loss."""
        loss_fn = SATopoLoss(lambda_topo=0.1)
        s, w, a_gt = _make_batch()
        a_pred = a_gt.clone().detach().requires_grad_(True)
        total, _, _ = loss_fn(s, w, a_pred, a_gt)
        total.backward()
        assert a_pred.grad is not None
        assert not torch.isnan(a_pred.grad).any()

    def test_lambda_topo_zero_no_topo_contribution(self):
        """With λ_topo=0, total should equal BC loss exactly."""
        loss_fn = SATopoLoss(lambda_topo=0.0)
        s, w, a_pred = _make_batch(seed=5)
        _, _, a_gt = _make_batch(seed=6)
        total, l_bc, _ = loss_fn(s, w, a_pred, a_gt)
        assert torch.allclose(total, l_bc)

    def test_identical_states_full_gate(self):
        """When all states are identical, W_S should be 1 everywhere."""
        loss_fn = SATopoLoss(tau=1.0)
        N, S, P, A = 8, 4, 2, 6
        # All states are the same
        s = torch.zeros(N, S)
        w = torch.rand(N, P)
        w = w / w.sum(dim=-1, keepdim=True)
        a = torch.randn(N, A)
        # Loss should be computed (gate = 1 everywhere)
        l_topo = loss_fn.compute_sa_topo_loss(s, w, a)
        assert torch.isfinite(l_topo)

    def test_high_tau_vs_low_tau(self):
        """High τ → gate is diffuse (larger mean gate); low τ → gate is sharp."""
        s, w, a = _make_batch(N=16, S=8, seed=7)
        from sa_topo_loss import _pairwise_sq_dist
        d_s = _pairwise_sq_dist(s)
        gate_low_tau  = torch.exp(-d_s / 0.01).mean().item()
        gate_high_tau = torch.exp(-d_s / 100.0).mean().item()
        assert gate_high_tau > gate_low_tau

    def test_batch_size_one_is_finite(self):
        """Edge case: batch size 1 should not crash."""
        loss_fn = SATopoLoss()
        s = torch.randn(1, 4)
        w = torch.ones(1, 2) * 0.5
        a = torch.randn(1, 6)
        l_topo = loss_fn.compute_sa_topo_loss(s, w, a)
        assert torch.isfinite(l_topo)


# ---------------------------------------------------------------------------
# SATopoLoss — diagnostics
# ---------------------------------------------------------------------------

class TestSATopoLossDiagnostics:
    def test_keys(self):
        loss_fn = SATopoLoss()
        s, w, a = _make_batch()
        diag = loss_fn.diagnostics(s, w, a)
        assert "l_topo" in diag
        assert "gate_active_fraction" in diag
        assert "da_dw_correlation" in diag
        assert "mean_state_similarity" in diag

    def test_gate_fraction_range(self):
        loss_fn = SATopoLoss()
        s, w, a = _make_batch()
        diag = loss_fn.diagnostics(s, w, a)
        assert 0.0 <= diag["gate_active_fraction"] <= 1.0

    def test_mean_similarity_range(self):
        loss_fn = SATopoLoss()
        s, w, a = _make_batch()
        diag = loss_fn.diagnostics(s, w, a)
        # exp(-…) ∈ (0, 1]
        assert 0.0 < diag["mean_state_similarity"] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Analysis metrics
# ---------------------------------------------------------------------------

class TestAnalysisMetrics:
    def test_tcs_good_policy(self):
        """Policy with actions ~ preference should have high TCS."""
        torch.manual_seed(99)
        N = 64
        states = torch.randn(N, 6)
        prefs  = torch.rand(N, 2)
        prefs  = prefs / prefs.sum(dim=-1, keepdim=True)
        W = torch.randn(2, 10)
        # Actions linearly depend on prefs → topologically consistent
        actions = prefs @ W + torch.randn(N, 10) * 0.01
        tcs = topological_consistency_score(states, prefs, actions, tau=5.0)
        # Correlation should be positive (policy is consistent with preferences)
        assert tcs > 0.0, f"Expected positive TCS for a good policy, got {tcs}"

    def test_tcs_bad_policy(self):
        """Fully random actions should have TCS near zero."""
        torch.manual_seed(11)
        N = 128
        states = torch.randn(N, 6)
        prefs  = torch.rand(N, 2)
        prefs  = prefs / prefs.sum(dim=-1, keepdim=True)
        actions = torch.randn(N, 10)  # completely random
        tcs = topological_consistency_score(states, prefs, actions, tau=5.0)
        # Should be much lower than the good-policy TCS
        assert abs(tcs) < 0.5, f"Expected near-zero TCS for a random policy, got {tcs}"

    def test_tvr_range(self):
        s, w, a = _make_batch(N=16)
        rate = topology_violation_rate(s, w, a, tau=1.0)
        assert 0.0 <= rate <= 1.0

    def test_tvr_perfect_alignment(self):
        """If all action distances exactly equal preference distances, TVR = 0."""
        torch.manual_seed(0)
        N = 16
        states = torch.zeros(N, 4)  # identical states → gate = 1 everywhere
        prefs  = torch.rand(N, 2)
        prefs  = prefs / prefs.sum(dim=-1, keepdim=True)
        # Construct actions so D_A == D_W
        # Use the preference vectors themselves as actions (D_A == D_W by definition)
        actions = prefs.clone()
        rate = topology_violation_rate(
            states, prefs, actions, tau=1.0, gate_threshold=0.5, alignment_delta=0.05
        )
        assert rate == 0.0, f"Expected zero TVR for perfectly aligned policy, got {rate}"
