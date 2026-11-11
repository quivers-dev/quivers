"""Direct unit tests for each `GradientEstimator` strategy.

The estimators are called on hand-built `ParticleLogDensities` whose
tensors are linear in two scalars: ``phi`` (a variational parameter)
and ``theta`` (a model parameter). The surrogate's value and its
gradients then have exact expected values, so a regression in an
estimator's algebra fails here rather than in a statistical test.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch

from quivers.inference.estimators import (
    DoublyReparameterized,
    GradientEstimator,
    ParticleLogDensities,
    Reparameterized,
    ScoreFunction,
    StickingTheLanding,
    leave_one_out_log_scale,
    power_mean_bound,
)

_A = torch.tensor(
    [[0.3], [-1.2], [0.7]], dtype=torch.float64
)  # theta coefficient in log p
_B = torch.tensor(
    [[1.1], [0.4], [-0.5]], dtype=torch.float64
)  # phi coefficient in log p (pathwise)
_C = torch.tensor(
    [[0.2], [-0.9], [1.5]], dtype=torch.float64
)  # phi coefficient in log q (direct + pathwise)
_D = torch.tensor(
    [[0.6], [0.1], [-0.3]], dtype=torch.float64
)  # phi coefficient in log q (pathwise only)
_UNIFORM = torch.full((3, 1), -math.log(3.0), dtype=torch.float64)
_PHI = 0.4
_THETA = -0.8


@pytest.fixture(autouse=True)
def _float64() -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _particles(phi: torch.Tensor, theta: torch.Tensor) -> ParticleLogDensities:
    log_p = theta * _A + phi * _B
    log_q = phi * _C
    # The detached-parameter density equals log q in value but depends
    # on phi only through the pathwise coefficient _D.
    log_q_detached = log_q.detach() + (phi - phi.detach()) * _D
    log_p_fixed = theta * _A + phi.detach() * _B
    return ParticleLogDensities(
        log_p=log_p,
        log_q=log_q,
        log_weights=_UNIFORM,
        log_q_detached=log_q_detached,
        log_p_fixed=log_p_fixed,
        stratified=False,
        num_particles=3,
    )


def _evaluate(estimator: GradientEstimator, power: float) -> tuple[float, float, float]:
    phi = torch.tensor(_PHI, requires_grad=True)
    theta = torch.tensor(_THETA, requires_grad=True)
    surrogate = estimator.surrogate(power, _particles(phi, theta)).sum()
    surrogate.backward()
    assert phi.grad is not None and theta.grad is not None
    return float(surrogate.detach()), float(phi.grad), float(theta.grad)


def _log_ratio() -> torch.Tensor:
    return _THETA * _A + _PHI * _B - _PHI * _C


def _normalized(power: float) -> torch.Tensor:
    return torch.softmax(_UNIFORM + power * _log_ratio(), dim=0)


POWERS = [0.0, 1.0, 0.5, 2.0]
ESTIMATORS = [
    Reparameterized(),
    StickingTheLanding(),
    DoublyReparameterized(),
    ScoreFunction(),
]


def test_power_mean_bound_special_cases() -> None:
    log_ratio = torch.tensor([[0.5], [-1.0], [2.0]])
    assert torch.allclose(power_mean_bound(log_ratio, _UNIFORM, 0.0), log_ratio.mean(0))
    iwae = torch.logsumexp(log_ratio, 0) - math.log(3.0)
    assert torch.allclose(power_mean_bound(log_ratio, _UNIFORM, 1.0), iwae)
    renyi = (torch.logsumexp(0.5 * log_ratio, 0) - math.log(3.0)) / 0.5
    assert torch.allclose(power_mean_bound(log_ratio, _UNIFORM, 0.5), renyi)


@pytest.mark.parametrize("power", POWERS)
@pytest.mark.parametrize("estimator", ESTIMATORS, ids=["rep", "stl", "dreg", "score"])
def test_surrogate_value_is_the_bound(
    estimator: GradientEstimator, power: float
) -> None:
    value, _, _ = _evaluate(estimator, power)
    expected = float(power_mean_bound(_log_ratio(), _UNIFORM, power).sum())
    assert math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize("power", POWERS)
def test_reparameterized_gradient_is_the_total_derivative(power: float) -> None:
    _, d_phi, d_theta = _evaluate(Reparameterized(), power)
    w = _normalized(power)
    assert math.isclose(d_phi, float((w * (_B - _C)).sum()), rel_tol=1e-10)
    assert math.isclose(d_theta, float((w * _A).sum()), rel_tol=1e-10)


def test_sticking_the_landing_drops_the_direct_score_term() -> None:
    _, d_phi, d_theta = _evaluate(StickingTheLanding(), 0.0)
    assert math.isclose(d_phi, float((_B - _D).mean()), rel_tol=1e-10)
    assert math.isclose(d_theta, float(_A.mean()), rel_tol=1e-10)


@pytest.mark.parametrize("power", [1.0, 0.5, 2.0])
def test_doubly_reparameterized_coefficients(power: float) -> None:
    """Variational gradient: sum(((1 - a) w + a w^2) * pathwise); model
    gradient: sum(w * d log p / d theta), the bound's own weights."""
    _, d_phi, d_theta = _evaluate(DoublyReparameterized(), power)
    w = _normalized(power)
    coefficients = (1.0 - power) * w + power * w**2
    assert math.isclose(d_phi, float((coefficients * (_B - _D)).sum()), rel_tol=1e-10)
    assert math.isclose(d_theta, float((w * _A).sum()), rel_tol=1e-10)


def test_doubly_reparameterized_at_power_zero_is_sticking_the_landing() -> None:
    assert _evaluate(DoublyReparameterized(), 0.0) == pytest.approx(
        _evaluate(StickingTheLanding(), 0.0), rel=1e-12
    )


def _fixed_draw_particles(
    phi: torch.Tensor, theta: torch.Tensor
) -> ParticleLogDensities:
    """Particles as an objective builds them for fixed draws: log p
    depends on theta only, and log q on phi only through its score."""
    return ParticleLogDensities(
        log_p=theta * _A,
        log_q=phi * _C,
        log_weights=_UNIFORM,
        log_q_detached=None,
        log_p_fixed=None,
        stratified=False,
        num_particles=3,
    )


def test_score_function_elbo_gradient() -> None:
    """``a = 0``: guide gradient mean(l * score), model gradient mean(dlog p)."""
    phi = torch.tensor(_PHI, requires_grad=True)
    theta = torch.tensor(_THETA, requires_grad=True)
    ScoreFunction().surrogate(0.0, _fixed_draw_particles(phi, theta)).sum().backward()
    log_ratio = _THETA * _A - _PHI * _C
    assert phi.grad is not None and theta.grad is not None
    assert math.isclose(float(phi.grad), float((log_ratio * _C).mean()), rel_tol=1e-10)
    assert math.isclose(float(theta.grad), float(_A.mean()), rel_tol=1e-10)


@pytest.mark.parametrize("power", [1.0, 0.5, 2.0])
def test_score_function_multi_sample_gradient(power: float) -> None:
    """``a != 0``: grad B at the fixed draws plus B times the summed scores."""
    phi = torch.tensor(_PHI, requires_grad=True)
    theta = torch.tensor(_THETA, requires_grad=True)
    ScoreFunction().surrogate(power, _fixed_draw_particles(phi, theta)).sum().backward()
    log_ratio = _THETA * _A - _PHI * _C
    w = torch.softmax(_UNIFORM + power * log_ratio, dim=0)
    bound = float(power_mean_bound(log_ratio, _UNIFORM, power).sum())
    expected_phi = float((w * -_C).sum()) + bound * float(_C.sum())
    assert phi.grad is not None and theta.grad is not None
    assert math.isclose(float(phi.grad), expected_phi, rel_tol=1e-10)
    assert math.isclose(float(theta.grad), float((w * _A).sum()), rel_tol=1e-10)


def test_sticking_the_landing_check() -> None:
    estimator = StickingTheLanding()
    estimator.check(0.0, 8, stratified=False)
    estimator.check(0.0, 8, stratified=True)
    estimator.check(1.0, 1, stratified=False)
    with pytest.raises(ValueError, match="DoublyReparameterized"):
        estimator.check(1.0, 2, stratified=False)
    with pytest.raises(ValueError, match=r"estimator=Reparameterized\(\)"):
        estimator.check(0.5, 1, stratified=True)


def test_doubly_reparameterized_check() -> None:
    estimator = DoublyReparameterized()
    estimator.check(1.0, 8, stratified=False)
    estimator.check(0.0, 8, stratified=True)
    with pytest.raises(ValueError, match="stratified"):
        estimator.check(1.0, 8, stratified=True)


def test_score_function_check_rejects_stratified_draws() -> None:
    ScoreFunction().check(1.0, 8, stratified=False)
    with pytest.raises(ValueError, match="independent draws"):
        ScoreFunction().check(0.0, 8, stratified=True)


def test_estimators_needing_detached_densities_raise_without_them() -> None:
    particles = ParticleLogDensities(
        log_p=torch.zeros(2, 1),
        log_q=torch.zeros(2, 1),
        log_weights=torch.full((2, 1), -math.log(2.0)),
        log_q_detached=None,
        log_p_fixed=None,
        stratified=False,
        num_particles=2,
    )
    with pytest.raises(RuntimeError, match="log_q_detached"):
        StickingTheLanding().surrogate(0.0, particles)
    with pytest.raises(RuntimeError, match="log_q_detached"):
        DoublyReparameterized().surrogate(1.0, particles)


def test_stratified_weights_carry_gradient_in_reparameterized_elbo() -> None:
    logits = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    log_weights = torch.log_softmax(logits, dim=0).unsqueeze(-1)
    log_ratio = torch.tensor([[1.0], [2.0], [-1.0]])
    particles = ParticleLogDensities(
        log_p=log_ratio,
        log_q=torch.zeros(3, 1),
        log_weights=log_weights,
        log_q_detached=None,
        log_p_fixed=None,
        stratified=True,
        num_particles=1,
    )
    Reparameterized().surrogate(0.0, particles).sum().backward()
    pi = torch.softmax(logits.detach(), dim=0)
    values = log_ratio.reshape(-1)
    expected = pi * (values - (pi * values).sum())
    assert logits.grad is not None
    assert torch.allclose(logits.grad, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# The exponentiated CUBO (ChiVI)
# ---------------------------------------------------------------------------


def test_leave_one_out_log_scale_excludes_the_particle_itself() -> None:
    # Three particles of two strata each, one batch row.
    scaled = torch.tensor([[1.0], [4.0], [2.0], [0.5], [3.0], [-1.0]])
    shift = leave_one_out_log_scale(scaled, 3)
    # Particle maxima are 4, 2, 3: each particle gets the max of the others.
    assert torch.equal(shift, torch.tensor([[3.0], [3.0], [4.0], [4.0], [4.0], [4.0]]))


def test_leave_one_out_log_scale_needs_two_particles() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        leave_one_out_log_scale(torch.zeros(3, 1), 1)


def _moment_case(
    phi: torch.Tensor, theta: torch.Tensor, fixed: bool
) -> ParticleLogDensities:
    log_p = theta * _A if fixed else theta * _A + phi * _B
    return ParticleLogDensities(
        log_p=log_p,
        log_q=phi * _C,
        log_weights=_UNIFORM,
        log_q_detached=None,
        log_p_fixed=None,
        stratified=False,
        num_particles=3,
    )


def _scaled_terms(order: float) -> torch.Tensor:
    log_ratio = _THETA * _A + _PHI * _B - _PHI * _C
    shift = leave_one_out_log_scale(order * log_ratio, 3)
    return torch.exp(_UNIFORM + order * log_ratio - shift)


@pytest.mark.parametrize("order", [2.0, 3.0])
def test_reparameterized_moment_surrogate_is_dieng_eq_3(order: float) -> None:
    """Gradient: sum_k T_k * n * grad log w_k, with T_k the scaled terms."""
    phi = torch.tensor(_PHI, requires_grad=True)
    theta = torch.tensor(_THETA, requires_grad=True)
    Reparameterized().moment_surrogate(
        order, _moment_case(phi, theta, fixed=False)
    ).sum().backward()
    terms = _scaled_terms(order)
    assert phi.grad is not None and theta.grad is not None
    expected_phi = float((terms * order * (_B - _C)).sum())
    expected_theta = float((terms * order * _A).sum())
    assert math.isclose(float(phi.grad), expected_phi, rel_tol=1e-10)
    assert math.isclose(float(theta.grad), expected_theta, rel_tol=1e-10)


@pytest.mark.parametrize("order", [2.0, 3.0])
def test_score_function_moment_surrogate_is_dieng_algorithm_1(order: float) -> None:
    """Guide gradient (1 - n) T grad log q, model gradient n T grad log p."""
    phi = torch.tensor(_PHI, requires_grad=True)
    theta = torch.tensor(_THETA, requires_grad=True)
    ScoreFunction().moment_surrogate(
        order, _moment_case(phi, theta, fixed=True)
    ).sum().backward()
    log_ratio = _THETA * _A - _PHI * _C
    shift = leave_one_out_log_scale(order * log_ratio, 3)
    terms = torch.exp(_UNIFORM + order * log_ratio - shift)
    assert phi.grad is not None and theta.grad is not None
    assert math.isclose(
        float(phi.grad), float((terms * (1.0 - order) * _C).sum()), rel_tol=1e-10
    )
    assert math.isclose(
        float(theta.grad), float((terms * order * _A).sum()), rel_tol=1e-10
    )


@pytest.mark.parametrize(
    "estimator", [StickingTheLanding(), DoublyReparameterized()], ids=["stl", "dreg"]
)
def test_other_estimators_reject_the_exponentiated_cubo(
    estimator: GradientEstimator,
) -> None:
    with pytest.raises(ValueError, match="ScoreFunction"):
        estimator.check_moment(2.0, 4, stratified=False)
