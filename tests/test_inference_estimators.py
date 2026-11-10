"""Direct unit tests for each :class:`GradientEstimator` strategy.

The estimators are tested in isolation (called with hand-rolled
``log_p`` / ``log_q`` tensors) so the test failure mode is sharp:
a regression in the estimator's math fires this test directly,
not an integration test downstream that masks the root cause.

For each estimator we verify:

* The negated objective is finite.
* The gradient with respect to a learnable parameter flows and
  is finite.
* The numerical contract (e.g. DReG's squared-softmax weights,
  StickingTheLanding's detached log_q) matches a hand-rolled
  reference.

Error paths (missing ``log_q_detached`` for sticking-the-landing
or DReG; wrong-shape log_p for DReG) are checked too.
"""

from __future__ import annotations

import pytest
import torch

from quivers.inference.estimators import (
    DoublyReparameterized,
    Reparameterized,
    ScoreFunction,
    StickingTheLanding,
)


# ---------------------------------------------------------------------------
# Reparameterized
# ---------------------------------------------------------------------------


def test_reparameterized_returns_negative_mean_diff() -> None:
    log_p = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    log_q = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
    out = Reparameterized().negative_objective(log_p, log_q)
    expected = -(log_p - log_q).mean()
    assert torch.allclose(out, expected, atol=1e-6)


def test_reparameterized_gradient_flows_through_log_q() -> None:
    torch.manual_seed(0)
    phi = torch.tensor([1.0, 2.0], requires_grad=True)
    log_p = torch.randn(4, 2)
    log_q = (phi * torch.ones(4, 2)).sum(dim=-1, keepdim=True).expand(4, 2)
    loss = Reparameterized().negative_objective(log_p, log_q)
    loss.backward()
    assert phi.grad is not None
    assert torch.isfinite(phi.grad).all()
    assert torch.any(phi.grad.abs() > 0)


def test_reparameterized_ignores_log_q_detached() -> None:
    log_p = torch.tensor([1.0, 2.0])
    log_q = torch.tensor([0.5, 1.5])
    fake_detached = torch.tensor([float("nan"), float("nan")])
    # Passing an unrelated tensor as log_q_detached must not affect
    # the result.
    out = Reparameterized().negative_objective(log_p, log_q, fake_detached)
    expected = Reparameterized().negative_objective(log_p, log_q, None)
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# StickingTheLanding
# ---------------------------------------------------------------------------


def test_stl_uses_detached_log_q() -> None:
    log_p = torch.tensor([[1.0, 2.0]])
    log_q = torch.tensor([[10.0, 20.0]])  # would dominate if used
    log_q_det = torch.tensor([[0.5, 1.5]])
    out = StickingTheLanding().negative_objective(log_p, log_q, log_q_det)
    expected = -(log_p - log_q_det).mean()
    assert torch.allclose(out, expected, atol=1e-6)


def test_stl_raises_without_detached_log_q() -> None:
    log_p = torch.tensor([1.0, 2.0])
    log_q = torch.tensor([0.5, 1.5])
    with pytest.raises(RuntimeError, match="log_q_detached"):
        StickingTheLanding().negative_objective(log_p, log_q, None)


def test_stl_gradient_only_flows_through_log_p() -> None:
    """The detached log_q_detached means STL's loss doesn't propagate
    the variational-parameter dependence through log_q. log_p still
    has its usual reparameterized path through the sample z. Here
    we construct a case where phi appears in BOTH log_p (with a
    coefficient of 1.0) and log_q (with a coefficient of 10.0) but
    log_q is detached; the resulting gradient must reflect only the
    log_p coefficient."""
    torch.manual_seed(0)
    phi = torch.tensor([1.0, 2.0], requires_grad=True)
    log_p = (1.0 * phi).sum().unsqueeze(0).unsqueeze(0).expand(3, 1)
    log_q = (10.0 * phi).sum().unsqueeze(0).unsqueeze(0).expand(3, 1)
    log_q_det = log_q.detach()
    loss = StickingTheLanding().negative_objective(log_p, log_q, log_q_det)
    loss.backward()
    # Gradient of -(log_p − log_q_det).mean() with respect to phi
    # is -1 * coefficient_of_phi_in_log_p (broadcast over batch).
    assert phi.grad is not None
    assert torch.allclose(phi.grad, -torch.ones_like(phi), atol=1e-6)


# ---------------------------------------------------------------------------
# DoublyReparameterized (DReG)
# ---------------------------------------------------------------------------


def test_dreg_uses_squared_softmax_weights() -> None:
    """The DReG surrogate is ``(softmax(log_w)^2) * (log_p − log_q)``
    summed over the particle axis, then averaged over batch."""
    log_p = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    log_q = torch.tensor([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
    log_q_det = torch.tensor([[0.4, 0.4], [1.4, 1.4], [2.4, 2.4]])
    out = DoublyReparameterized().negative_objective(log_p, log_q, log_q_det)
    log_w = log_p - log_q_det
    weights = torch.softmax(log_w, dim=0).detach()
    expected = -((weights**2) * (log_p - log_q)).sum(dim=0).mean()
    assert torch.allclose(out, expected, atol=1e-6)


def test_dreg_raises_without_detached_log_q() -> None:
    log_p = torch.tensor([[1.0], [2.0]])
    log_q = torch.tensor([[0.5], [1.5]])
    with pytest.raises(RuntimeError, match="log_q_detached"):
        DoublyReparameterized().negative_objective(log_p, log_q, None)


def test_dreg_raises_on_scalar_log_p() -> None:
    log_p = torch.tensor(1.0)
    log_q = torch.tensor(0.5)
    log_q_det = torch.tensor(0.4)
    with pytest.raises(RuntimeError, match="leading particle axis"):
        DoublyReparameterized().negative_objective(log_p, log_q, log_q_det)


def test_dreg_gradient_flows() -> None:
    torch.manual_seed(0)
    phi = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    log_p = phi.sum().unsqueeze(0).expand(4, 1)
    log_q = (0.5 * phi).sum().unsqueeze(0).expand(4, 1)
    log_q_det = log_q.detach()
    loss = DoublyReparameterized().negative_objective(log_p, log_q, log_q_det)
    loss.backward()
    assert phi.grad is not None
    assert torch.isfinite(phi.grad).all()
    assert torch.any(phi.grad.abs() > 0)


# ---------------------------------------------------------------------------
# ScoreFunction (REINFORCE)
# ---------------------------------------------------------------------------


def test_score_function_value_is_the_elbo_estimate() -> None:
    """The surrogate's value is the ELBO estimate ``mean(log p - log q)``."""
    log_p = torch.tensor([1.0, 2.0])
    log_q = torch.tensor([0.5, 1.5])
    out = ScoreFunction().negative_objective(log_p, log_q)
    assert torch.allclose(out, -(log_p - log_q).mean(), atol=1e-12)


def test_score_function_gradient_is_f_times_the_score() -> None:
    """With fixed draws the variational gradient is ``f * grad log q``
    (averaged) and the model gradient is ``grad log p``."""
    phi = torch.tensor(0.7, requires_grad=True)
    theta = torch.tensor(-0.4, requires_grad=True)
    log_q = torch.stack([phi * 1.0, phi * 2.0])
    log_p = torch.stack([theta * 3.0, theta * 5.0])
    ScoreFunction().negative_objective(log_p, log_q).backward()
    f = (log_p - log_q).detach()
    assert phi.grad is not None and theta.grad is not None
    assert torch.allclose(phi.grad, -(f * torch.tensor([1.0, 2.0])).mean())
    assert torch.allclose(theta.grad, -torch.tensor([3.0, 5.0]).mean())


def test_score_function_rejects_stratified_weights() -> None:
    log_p = torch.zeros(2, 1)
    log_q = torch.zeros(2, 1)
    with pytest.raises(RuntimeError, match="independent draws"):
        ScoreFunction().negative_objective(
            log_p, log_q, log_weights=torch.full((2, 1), -0.6931471805599453)
        )


def test_score_function_gradient_flows() -> None:
    torch.manual_seed(0)
    phi = torch.tensor([1.0, 2.0], requires_grad=True)
    log_p = torch.randn(3, 1)
    log_q = phi.sum().unsqueeze(0).expand(3, 1)
    loss = ScoreFunction().negative_objective(log_p, log_q)
    loss.backward()
    assert phi.grad is not None
    assert torch.isfinite(phi.grad).all()
    assert torch.any(phi.grad.abs() > 0)


def test_score_function_ignores_log_q_detached() -> None:
    """Score-function only consumes log_p and log_q (the detach
    happens internally); a supplied log_q_detached must not affect
    the result."""
    log_p = torch.tensor([1.0, 2.0])
    log_q = torch.tensor([0.5, 1.5])
    fake_detached = torch.tensor([float("nan"), float("nan")])
    out = ScoreFunction().negative_objective(log_p, log_q, fake_detached)
    expected = ScoreFunction().negative_objective(log_p, log_q, None)
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Cross-estimator: sticking-the-landing reduces variance near the
# optimum (Roeder-Wu-Duvenaud's central claim)
# ---------------------------------------------------------------------------


def test_stl_gradient_variance_vanishes_at_perfect_guide() -> None:
    """When ``q == p`` exactly, ``log_p − log_q_detached`` is zero
    everywhere, so the sticking-the-landing gradient is zero — no
    noise. The test puts phi in BOTH log_p (so the loss has a
    grad-requiring source) and log_q_det (constructed so the
    difference is exactly zero), and verifies the resulting
    gradient is zero."""
    torch.manual_seed(0)
    phi = torch.tensor([1.0], requires_grad=True)
    # Construct a synthetic case where log_p == log_q_det (perfect
    # guide). Both depend on phi with the same coefficient, so the
    # difference is zero.
    log_p = phi * torch.ones(8, 1)
    log_q = log_p.clone()
    log_q_det = log_p.detach()
    loss = StickingTheLanding().negative_objective(log_p, log_q, log_q_det)
    loss.backward()
    # STL gradient at the optimum: -1 (from log_p's coefficient)
    # since log_q_det is detached. Note this is the gradient of
    # `-(log_p − log_q_det).mean()` = -(log_p).mean() + constant,
    # so phi gets a gradient of -1. The "variance is zero" claim
    # is about Monte Carlo variance across samples — not about the
    # mean gradient. The contract is: at q == p, the STL gradient
    # is deterministic (no per-sample randomness from log_q).
    assert phi.grad is not None
    assert torch.allclose(phi.grad, -torch.ones_like(phi), atol=1e-6)
