"""The log-space families score and sample exactly where their constrained forms fail.

`LogGamma` is the law of ``log(g)`` for ``g ~ Gamma(a, r)`` and `LogitBeta`
the law of ``logit(p)`` for ``p ~ Beta(a, b)``. Their densities are held to
SciPy's float64 closed forms over a grid that reaches shapes of ``1e-3``
and arguments far into the tails; the samplers are held to the closed-form
mean, variance and distribution function within Monte Carlo error, with
shapes small enough that a constrained-scale sampler clamps half its draws.
Each check is also run on the naive construction (transform a Beta or
Gamma density, take the log or logit of a Beta or Gamma draw) to show that
it rejects what the families exist to replace.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.distributions as D
from torch.distributions import constraints
from scipy import integrate, special, stats

from quivers.continuous import (
    ConditionalLogGamma,
    ConditionalLogitBeta,
    Euclidean,
    LogGamma,
    LogitBeta,
    log_standard_gamma,
    make_fixed_log_gamma,
    make_fixed_logit_beta,
)
from quivers.core.objects import FinSet
from quivers.dsl import loads, parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.inference import LatentRegistry
from quivers.inference.trace import trace
from quivers.qiec import validate_module
from quivers.qiec.distributions import ReferenceBackend
from quivers.qiec.program_runtime import run_program
from quivers.transpile import transpile
from tests._target_runtimes import pyro_log_density

#: Monte Carlo draws per sampler check.
_DRAWS = 200_000

#: The number of standard errors a Monte Carlo estimate may sit from its
#: closed form. Fixed before looking at any estimate: at five standard
#: errors a seeded check over the few dozen comparisons below fails by
#: chance with probability below one in ten thousand.
_Z = 5.0


def _f64(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64)


# ---------------------------------------------------------------------------
# closed forms, from SciPy in float64


def _log_gamma_oracle(y: float, a: float, r: float) -> float:
    """SciPy's log-gamma density, shifted by the log rate."""
    return float(stats.loggamma.logpdf(y + math.log(r), a))


def _logit_beta_oracle(x: float, a: float, b: float) -> float:
    """The logit-Beta density from SciPy's ``log_expit`` and ``betaln``."""
    return float(
        a * special.log_expit(x) + b * special.log_expit(-x) - special.betaln(a, b)
    )


def _log_gamma_cdf(y: float, a: float, r: float) -> float:
    """``P(log g <= y)`` for ``g ~ Gamma(a, r)``.

    Deep in the left tail the regularized incomplete gamma function is
    ``x^a / Gamma(a + 1)`` to a relative error below ``x``, which is
    evaluated from ``log x`` so it stays exact where ``x`` underflows.
    """
    log_x = y + math.log(r)
    if log_x < -30.0:
        return math.exp(a * log_x - special.gammaln(a + 1.0))
    return float(special.gammainc(a, math.exp(log_x)))


def _logit_beta_cdf(x: float, a: float, b: float) -> float:
    """``P(logit p <= x)`` for ``p ~ Beta(a, b)``.

    In either tail the regularized incomplete Beta function is
    ``p^a / (a B(a, b))`` to a relative error of order ``p``, evaluated
    from ``log p = log_expit(x)``.
    """
    if x < -30.0:
        return math.exp(a * special.log_expit(x) - math.log(a) - special.betaln(a, b))
    if x > 30.0:
        return -math.expm1(
            b * special.log_expit(-x) - math.log(b) - special.betaln(a, b)
        )
    return float(special.betainc(a, b, special.expit(x)))


def _log_gamma_cumulants(a: float, r: float) -> tuple[float, float, float]:
    """Mean, variance and fourth cumulant of ``log g``."""
    return (
        float(special.digamma(a)) - math.log(r),
        float(special.polygamma(1, a)),
        float(special.polygamma(3, a)),
    )


def _logit_beta_cumulants(a: float, b: float) -> tuple[float, float, float]:
    """Mean, variance and fourth cumulant of ``logit p``: the cumulants of
    a difference of two independent log-Gamma variables."""
    return (
        float(special.digamma(a) - special.digamma(b)),
        float(special.polygamma(1, a) + special.polygamma(1, b)),
        float(special.polygamma(3, a) + special.polygamma(3, b)),
    )


# ---------------------------------------------------------------------------
# densities

_LOG_GAMMA_SHAPES = (1e-3, 1e-2, 0.5, 1.0, 7.5, 300.0)
_LOG_GAMMA_RATES = (0.2, 1.0, 40.0)
_LOG_GAMMA_POINTS = (-800.0, -100.0, -5.0, 0.0, 2.0, 6.0)

_LOGIT_BETA_SHAPES = (1e-3, 0.05, 1.0, 20.0)
_LOGIT_BETA_POINTS = (-800.0, -40.0, -3.0, 0.0, 3.0, 40.0, 800.0)


def test_log_gamma_density_matches_scipy_over_the_grid() -> None:
    for a in _LOG_GAMMA_SHAPES:
        for r in _LOG_GAMMA_RATES:
            family = LogGamma(_f64(a), _f64(r))
            for y in _LOG_GAMMA_POINTS:
                actual = float(family.log_prob(_f64(y)))
                expected = _log_gamma_oracle(y, a, r)
                assert math.isfinite(actual), (a, r, y)
                assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), (
                    a,
                    r,
                    y,
                )


def test_logit_beta_density_matches_scipy_over_the_grid() -> None:
    for a in _LOGIT_BETA_SHAPES:
        for b in _LOGIT_BETA_SHAPES:
            family = LogitBeta(_f64(a), _f64(b))
            for x in _LOGIT_BETA_POINTS:
                actual = float(family.log_prob(_f64(x)))
                expected = _logit_beta_oracle(x, a, b)
                assert math.isfinite(actual), (a, b, x)
                assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), (
                    a,
                    b,
                    x,
                )


def test_the_issue_point_scores_its_exact_density() -> None:
    """``LogitBeta(1e-3, 1)`` at ``-800`` is ``-7.71``, not the ``-99`` of
    the sigmoid round trip."""
    exact = float(LogitBeta(_f64(1e-3), _f64(1.0)).log_prob(_f64(-800.0)))
    assert exact == pytest.approx(-0.8 - math.log(1000.0), rel=1e-12)


def _naive_log_gamma(y: float, a: float, r: float) -> float:
    """The Gamma density through ``biject_to``, as a constrained site is scored."""
    transform = D.biject_to(constraints.positive)
    point = _f64(y)
    value = transform(point)
    return float(
        D.Gamma(_f64(a), _f64(r)).log_prob(value)
        + transform.log_abs_det_jacobian(point, value)
    )


def _naive_logit_beta(x: float, a: float, b: float) -> float:
    """The Beta density through ``biject_to``, as a constrained site is scored."""
    transform = D.biject_to(constraints.unit_interval)
    point = _f64(x)
    value = transform(point)
    return float(
        D.Beta(_f64(a), _f64(b)).log_prob(value)
        + transform.log_abs_det_jacobian(point, value)
    )


def _misses(
    oracle: Callable[[float, float, float], float],
    candidate: Callable[[float, float, float], float],
    cells: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """The grid cells where a candidate density is not the oracle's."""
    missed = []
    for point, first, second in cells:
        value = candidate(point, first, second)
        expected = oracle(point, first, second)
        if not (math.isfinite(value) and abs(value - expected) <= 1e-9 * abs(expected)):
            missed.append((point, first, second))
    return missed


def test_the_density_check_rejects_the_transformed_constrained_density() -> None:
    gamma_cells = [
        (y, a, r)
        for a in _LOG_GAMMA_SHAPES
        for r in _LOG_GAMMA_RATES
        for y in _LOG_GAMMA_POINTS
    ]
    beta_cells = [
        (x, a, b)
        for a in _LOGIT_BETA_SHAPES
        for b in _LOGIT_BETA_SHAPES
        for x in _LOGIT_BETA_POINTS
    ]
    assert _misses(_log_gamma_oracle, _naive_log_gamma, gamma_cells)
    assert _misses(_logit_beta_oracle, _naive_logit_beta, beta_cells)
    # The issue's point: the round trip reads a wall of about -99.
    assert _naive_logit_beta(-800.0, 1e-3, 1.0) < -90.0
    assert not _misses(
        _log_gamma_oracle,
        lambda y, a, r: float(LogGamma(_f64(a), _f64(r)).log_prob(_f64(y))),
        gamma_cells,
    )
    assert not _misses(
        _logit_beta_oracle,
        lambda x, a, b: float(LogitBeta(_f64(a), _f64(b)).log_prob(_f64(x))),
        beta_cells,
    )


@pytest.mark.parametrize(
    ("family", "pieces"),
    [
        (LogGamma(_f64(1e-3), _f64(1.0)), (-math.inf, -50.0, 5.0, math.inf)),
        (LogGamma(_f64(0.5), _f64(2.0)), (-math.inf, -10.0, 5.0, math.inf)),
        (LogGamma(_f64(7.0), _f64(0.3)), (-math.inf, 0.0, 5.0, math.inf)),
        (LogitBeta(_f64(1e-3), _f64(1.0)), (-math.inf, -50.0, 50.0, math.inf)),
        (LogitBeta(_f64(0.05), _f64(0.02)), (-math.inf, -50.0, 50.0, math.inf)),
        (LogitBeta(_f64(2.0), _f64(5.0)), (-math.inf, -10.0, 10.0, math.inf)),
    ],
    ids=["lg-1e-3", "lg-0.5", "lg-7", "lb-1e-3", "lb-small", "lb-2-5"],
)
def test_the_density_integrates_to_one(family: D.Distribution, pieces: tuple) -> None:
    def density(point: float) -> float:
        return math.exp(float(family.log_prob(_f64(point))))

    total = sum(
        integrate.quad(density, low, high, limit=500, epsabs=1e-12, epsrel=1e-10)[0]
        for low, high in zip(pieces, pieces[1:])
    )
    assert total == pytest.approx(1.0, abs=1e-7)


def test_log_prob_gradients_match_finite_differences() -> None:
    a = _f64(0.3).requires_grad_()
    r = _f64(1.7).requires_grad_()
    y = torch.tensor([-40.0, -1.0, 0.4], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a_, r_, y_: LogGamma(a_, r_).log_prob(y_), (a, r, y)
    )
    first = _f64(0.02).requires_grad_()
    second = _f64(3.0).requires_grad_()
    x = torch.tensor([-300.0, -2.0, 5.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a_, b_, x_: LogitBeta(a_, b_).log_prob(x_), (first, second, x)
    )


def test_entropy_matches_the_monte_carlo_expectation() -> None:
    torch.manual_seed(5)
    for family in (
        LogGamma(_f64(0.4), _f64(3.0)),
        LogitBeta(_f64(0.7), _f64(2.5)),
    ):
        draws = family.sample((_DRAWS,))
        values = -family.log_prob(draws)
        error = float(values.std()) / math.sqrt(_DRAWS)
        assert abs(float(values.mean()) - float(family.entropy())) < _Z * error


# ---------------------------------------------------------------------------
# sampling


def _sample_discrepancy(
    draws: torch.Tensor,
    cumulants: tuple[float, float, float],
    cdf: Callable[[float], float],
    points: tuple[float, ...],
) -> float:
    """The largest standardized gap between the draws and the closed forms.

    The gaps are of the sample mean from the mean, the sample variance
    from the variance (standard error from the fourth cumulant), and the
    empirical distribution function from the closed form at each point.

    Parameters
    ----------
    draws : torch.Tensor
        The draws.
    cumulants : tuple[float, float, float]
        Mean, variance, fourth cumulant.
    cdf : Callable[[float], float]
        The closed-form distribution function.
    points : tuple[float, ...]
        Where the distribution function is compared.

    Returns
    -------
    float
        The largest gap in standard errors.
    """
    values = draws.to(torch.float64)
    n = values.numel()
    mean, variance, fourth = cumulants
    gaps = [
        abs(float(values.mean()) - mean) / math.sqrt(variance / n),
        abs(float(values.var()) - variance)
        / math.sqrt((fourth + 2.0 * variance**2) / n),
    ]
    for point in points:
        expected = cdf(point)
        observed = float((values <= point).double().mean())
        error = math.sqrt(max(expected * (1.0 - expected), 1.0 / n) / n)
        gaps.append(abs(observed - expected) / error)
    return max(gaps)


def _points(mean: float, variance: float, extra: tuple[float, ...]) -> tuple:
    sd = math.sqrt(variance)
    return (mean - 2 * sd, mean - sd, mean, mean + sd, *extra)


_LOG_GAMMA_CASES = [(1e-3, 1.0), (0.05, 2.0), (0.7, 0.5), (12.0, 3.0)]
_LOGIT_BETA_CASES = [(1e-3, 1.0), (0.02, 0.05), (0.8, 2.5), (15.0, 4.0)]


@pytest.mark.parametrize(("a", "r"), _LOG_GAMMA_CASES)
def test_log_gamma_samples_match_the_closed_forms(a: float, r: float) -> None:
    torch.manual_seed(11)
    draws = LogGamma(_f64(a), _f64(r)).sample((_DRAWS,))
    assert torch.isfinite(draws).all()
    cumulants = _log_gamma_cumulants(a, r)
    extra = (-750.0,) if a < 0.01 else ()
    points = _points(cumulants[0], cumulants[1], extra)
    gap = _sample_discrepancy(
        draws, cumulants, lambda y: _log_gamma_cdf(y, a, r), points
    )
    assert gap < _Z


@pytest.mark.parametrize(("a", "b"), _LOGIT_BETA_CASES)
def test_logit_beta_samples_match_the_closed_forms(a: float, b: float) -> None:
    torch.manual_seed(13)
    draws = LogitBeta(_f64(a), _f64(b)).sample((_DRAWS,))
    assert torch.isfinite(draws).all()
    cumulants = _logit_beta_cumulants(a, b)
    extra = (-750.0,) if min(a, b) < 0.01 else ()
    points = _points(cumulants[0], cumulants[1], extra)
    gap = _sample_discrepancy(
        draws, cumulants, lambda x: _logit_beta_cdf(x, a, b), points
    )
    assert gap < _Z


def test_the_sample_check_rejects_transformed_constrained_draws() -> None:
    """The log of a torch Gamma draw and the logit of a torch Beta draw
    at shape ``1e-3`` pile up at the clamp, and the check sees it."""
    torch.manual_seed(17)
    a, r = 1e-3, 1.0
    gamma = D.Gamma(_f64(a), _f64(r)).sample((_DRAWS,))
    clamped = float((gamma == torch.finfo(torch.float64).tiny).double().mean())
    assert clamped > 0.4
    cumulants = _log_gamma_cumulants(a, r)
    points = _points(cumulants[0], cumulants[1], (-750.0,))
    assert (
        _sample_discrepancy(
            gamma.log(), cumulants, lambda y: _log_gamma_cdf(y, a, r), points
        )
        > 10 * _Z
    )
    b = 1.0
    beta = D.Beta(_f64(a), _f64(b)).sample((_DRAWS,))
    logits = torch.logit(beta)
    beta_cumulants = _logit_beta_cumulants(a, b)
    beta_points = _points(beta_cumulants[0], beta_cumulants[1], (-750.0,))
    assert (
        _sample_discrepancy(
            logits, beta_cumulants, lambda x: _logit_beta_cdf(x, a, b), beta_points
        )
        > 10 * _Z
    )


def test_draws_are_finite_down_to_the_documented_shape_floor() -> None:
    torch.manual_seed(19)
    for dtype in (torch.float32, torch.float64):
        floor = 64.0 * torch.finfo(dtype).tiny
        for shape in (floor, 1e-30, 1e-8):
            concentration = torch.full((50_000,), shape, dtype=dtype)
            draws = log_standard_gamma(concentration)
            assert torch.isfinite(draws).all(), (dtype, shape)
            pair = LogitBeta(concentration, torch.ones_like(concentration)).sample()
            assert torch.isfinite(pair).all(), (dtype, shape)


def test_tiny_shapes_put_their_mass_where_the_closed_form_does() -> None:
    """At shape ``1e-8`` the median of ``log g`` is about ``-6.9e7``,
    which no constrained-scale draw can represent."""
    torch.manual_seed(23)
    draws = LogGamma(_f64(1e-8), _f64(1.0)).sample((_DRAWS,))
    median = math.log(0.5) / 1e-8 + special.gammaln(1.0 + 1e-8) / 1e-8
    assert _log_gamma_cdf(median, 1e-8, 1.0) == pytest.approx(0.5, rel=1e-9)
    observed = float((draws <= median).double().mean())
    assert abs(observed - 0.5) < _Z * math.sqrt(0.25 / _DRAWS)


# ---------------------------------------------------------------------------
# reparameterized gradients


def _pathwise_gradient(
    sampler: Callable[[torch.Tensor], torch.Tensor],
    value: float,
    statistic: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[float, float]:
    """The pathwise gradient of ``E[statistic(draw)]`` in one parameter.

    Each draw gets its own copy of the parameter, so the gradient of the
    summed statistic holds one per-draw gradient per copy.

    Returns
    -------
    tuple[float, float]
        The mean per-draw gradient and its standard error.
    """
    parameter = torch.full((_DRAWS,), value, dtype=torch.float64, requires_grad=True)
    statistic(sampler(parameter)).sum().backward()
    assert parameter.grad is not None
    grads = parameter.grad
    return float(grads.mean()), float(grads.std()) / math.sqrt(_DRAWS)


@pytest.mark.parametrize("a", [0.05, 0.5, 3.0])
def test_log_gamma_pathwise_gradients_match_the_closed_forms(a: float) -> None:
    torch.manual_seed(29)
    r = 2.0
    cases = [
        # d E[y] / d a = trigamma(a)
        (
            lambda c: LogGamma(c, _f64(r)).rsample(),
            a,
            lambda y: y,
            float(special.polygamma(1, a)),
        ),
        # d E[exp(y)] / d a = d (a / r) / d a = 1 / r
        (lambda c: LogGamma(c, _f64(r)).rsample(), a, torch.exp, 1.0 / r),
        # d E[exp(y)] / d r = -a / r^2
        (lambda q: LogGamma(_f64(a), q).rsample(), r, torch.exp, -a / r**2),
    ]
    for sampler, value, statistic, expected in cases:
        mean, error = _pathwise_gradient(sampler, value, statistic)
        assert abs(mean - expected) < _Z * error + 1e-12, (a, expected, mean)


@pytest.mark.parametrize(("a", "b"), [(0.05, 1.5), (0.8, 2.5)])
def test_logit_beta_pathwise_gradients_match_the_closed_forms(
    a: float, b: float
) -> None:
    torch.manual_seed(31)
    cases = [
        (
            lambda c: LogitBeta(c, _f64(b)).rsample(),
            a,
            lambda x: x,
            float(special.polygamma(1, a)),
        ),
        (
            lambda c: LogitBeta(_f64(a), c).rsample(),
            b,
            lambda x: x,
            -float(special.polygamma(1, b)),
        ),
        # E[sigmoid(x)] = a / (a + b)
        (
            lambda c: LogitBeta(c, _f64(b)).rsample(),
            a,
            torch.sigmoid,
            b / (a + b) ** 2,
        ),
        (
            lambda c: LogitBeta(_f64(a), c).rsample(),
            b,
            torch.sigmoid,
            -a / (a + b) ** 2,
        ),
    ]
    for sampler, value, statistic, expected in cases:
        mean, error = _pathwise_gradient(sampler, value, statistic)
        assert abs(mean - expected) < _Z * error, (a, b, expected, mean)


def test_the_gradient_check_rejects_a_sampler_without_the_implicit_path() -> None:
    """Detaching ``G'`` keeps only the ``log(U) / a`` path, whose gradient
    is ``1 / a^2`` rather than ``trigamma(a)``."""
    torch.manual_seed(37)

    def detached(concentration: torch.Tensor) -> torch.Tensor:
        boosted = torch._standard_gamma(concentration.detach() + 1.0)
        uniform = torch.rand(concentration.shape, dtype=concentration.dtype)
        return torch.log(boosted) + torch.log1p(-uniform) / concentration

    a = 0.5
    mean, error = _pathwise_gradient(detached, a, lambda y: y)
    assert abs(mean - float(special.polygamma(1, a))) > 10 * _Z * error


# ---------------------------------------------------------------------------
# the runtime surfaces


def test_fixed_factories_score_unclamped_shapes() -> None:
    codomain = Euclidean(name="x", dim=1)
    gamma = make_fixed_log_gamma(1e-9, 2.0, codomain)
    beta = make_fixed_logit_beta(1e-9, 1.0, codomain)
    point = torch.tensor([[-5e8]], dtype=torch.float32)
    unit = torch.zeros(1, 1)
    assert float(gamma.log_prob(unit, point)) == pytest.approx(
        float(LogGamma(1e-9, 2.0).log_prob(point)), rel=1e-6
    )
    assert float(beta.log_prob(unit, point)) == pytest.approx(
        float(LogitBeta(1e-9, 1.0).log_prob(point)), rel=1e-6
    )
    with pytest.raises(ValueError, match="positive concentration"):
        make_fixed_log_gamma(0.0, 1.0, codomain)
    with pytest.raises(ValueError, match="positive concentration0"):
        make_fixed_logit_beta(1.0, -1.0, codomain)


def test_conditional_families_are_reparameterized_on_the_real_line() -> None:
    torch.manual_seed(41)
    domain = FinSet(name="G", cardinality=3)
    codomain = Euclidean(name="y", dim=2)
    for cls in (ConditionalLogGamma, ConditionalLogitBeta):
        kernel = cls(domain, codomain)
        x = torch.tensor([0, 1, 2])
        draws = kernel.rsample(x)
        assert draws.shape == (3, 2)
        draws.sum().backward()
        assert all(p.grad is not None for p in kernel.parameters())
        scores = kernel.log_prob(x, torch.full((3, 2), -400.0))
        assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------------
# the compiled runtime and the reference machine

_SOURCE = """\
object Item : FinSet 4

program shrink : Item -> Item
    sample rate <- LogGamma(concentration=0.002, rate=1.5)
    sample weight <- LogitBeta(concentration1=0.01, concentration0=2.0)
    let spread = exp(rate) + 1.0
    observe y : Item <- Normal(loc=weight, scale=spread)
    return weight

export shrink
"""

_Y = (0.3, -1.2, 0.8, 0.1)

#: Latent points, including ones deep in the tails the constrained forms lose.
_POINTS = (
    {"rate": -0.4, "weight": 0.2},
    {"rate": -650.0, "weight": -480.0},
    {"rate": 1.1, "weight": -2.0},
)


def _hand_joint(point: dict[str, float]) -> float:
    rate, weight = point["rate"], point["weight"]
    return (
        _log_gamma_oracle(rate, 0.002, 1.5)
        + _logit_beta_oracle(weight, 0.01, 2.0)
        + float(
            np.sum(
                stats.norm.logpdf(np.array(_Y), loc=weight, scale=math.exp(rate) + 1.0)
            )
        )
    )


def test_the_compiled_runtime_and_the_reference_machine_agree() -> None:
    program = loads(_SOURCE).morphism
    module = lower_qvr_to_qiec(parse(_SOURCE), module_name="shrink")
    validate_module(module)
    for point in _POINTS:
        expected = _hand_joint(point)
        joint = program.log_joint(
            torch.zeros(1, 1),
            {
                "y": torch.tensor(_Y, dtype=torch.float64),
                **{name: _f64(value) for name, value in point.items()},
            },
        )
        reference = run_program(module, "shrink", data={"y": _Y}, sites=dict(point))
        assert math.isfinite(expected)
        # The compiled runtime holds literal parameters in float32, so it
        # agrees to float32's relative precision; the reference is float64.
        assert float(joint[0]) == pytest.approx(expected, rel=1e-6)
        assert reference.log_joint == pytest.approx(expected, rel=1e-9)


def test_inference_sees_the_sites_on_the_real_line_with_the_identity_bijector() -> None:
    """A latent of either family is unconstrained, so the potential a
    sampler or a guide works on is the exact density itself."""
    program = loads(_SOURCE).morphism
    y = torch.tensor(_Y, dtype=torch.float64)
    registry = LatentRegistry.from_trace(
        program, trace(program, torch.zeros(1, 1), {"y": y})
    )
    for name in ("rate", "weight"):
        site = registry[name]
        assert site.support is constraints.real
        assert isinstance(site.bijector, D.transforms.identity_transform.__class__)
    joint = program.log_joint(
        torch.zeros(1, 1),
        {"y": y, "rate": _f64(-650.0), "weight": _f64(-480.0)},
    )
    assert torch.isfinite(joint).all()


def test_the_reference_sampler_matches_the_closed_form_moments() -> None:
    """Both samplers' means sit within Monte Carlo error of the closed form."""
    backend = ReferenceBackend()
    rng = random.Random(43)
    draws = 20_000
    for family, arguments, cumulants in (
        (
            "LogGamma",
            {"concentration": 0.01, "rate": 3.0},
            _log_gamma_cumulants(0.01, 3.0),
        ),
        (
            "LogitBeta",
            {"concentration1": 0.02, "concentration0": 0.7},
            _logit_beta_cumulants(0.02, 0.7),
        ),
    ):
        values = [float(backend.sample(family, arguments, rng)) for _ in range(draws)]  # type: ignore[arg-type]
        assert all(math.isfinite(value) for value in values)
        mean, variance, _ = cumulants
        assert abs(sum(values) / draws - mean) < _Z * math.sqrt(variance / draws)


def test_the_pyro_rendering_scores_the_reference_joint() -> None:
    emitted = transpile(parse(_SOURCE), target="pyro")
    for point in _POINTS:
        scored = pyro_log_density(emitted, {"y": list(_Y)}, point)
        assert scored == pytest.approx(_hand_joint(point), rel=1e-6)
