"""Truncated Normal densities and draws checked against SciPy."""

import math

import pytest
import torch
from scipy.stats import truncnorm

from quivers.dsl import loads
from quivers.continuous.measure import Normalize, Restrict

from quivers.continuous.families import ConditionalTruncatedNormal
from quivers.continuous.param_source import IdentitySource
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet

from quivers.continuous.inline import _truncated_normal_builder


@pytest.mark.parametrize(
    "low,high",
    [
        (-2.0, 3.0),
        (8.0, 9.0),
        (-9.0, -8.0),
        (20.0, 21.0),
        (100.0, 101.0),
        (-0.000001, 0.000001),
    ],
)
def test_truncated_density_matches_independent_oracle(low: float, high: float) -> None:
    dist = _truncated_normal_builder(
        [torch.tensor(x, dtype=torch.float64) for x in (0.0, 1.0, low, high)]
    )
    value = (low + high) / 2
    actual = dist.log_prob(torch.tensor(value, dtype=torch.float64))
    expected = truncnorm.logpdf(value, low, high)
    assert actual.item() == pytest.approx(expected, rel=1e-10, abs=1e-9)
    if low >= 8:
        floored = -0.5 * value**2 - 0.5 * math.log(2 * math.pi) - math.log(1e-7)
        assert abs(actual.item() - floored) > 10


@pytest.mark.parametrize("low,high", [(8.0, 9.0), (-9.0, -8.0), (100.0, 101.0)])
def test_tail_draws_are_in_support_and_have_expected_quantiles(
    low: float, high: float
) -> None:
    dist = _truncated_normal_builder(
        [torch.tensor(x, dtype=torch.float64) for x in (0.0, 1.0, low, high)]
    )
    with torch.random.fork_rng():
        torch.manual_seed(310)
        draws = dist.rsample(torch.Size([10000]))
    assert (draws >= low).all() and (draws <= high).all()
    probabilities = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    expected = torch.as_tensor(truncnorm.ppf(probabilities.numpy(), low, high))
    # Fixed allowance for a binomial quantile error below 0.02.
    lower = torch.as_tensor(truncnorm.ppf((probabilities - 0.02).numpy(), low, high))
    upper = torch.as_tensor(truncnorm.ppf((probabilities + 0.02).numpy(), low, high))
    actual = draws.quantile(probabilities)
    assert (actual > lower).all() and (actual < upper).all()
    assert (expected > lower).all() and (expected < upper).all()


def test_tail_density_gradient_matches_oracle_difference() -> None:
    mu = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    low, high, y = 8.0, 9.0, 8.2
    dist = _truncated_normal_builder([mu, scale, torch.tensor(low), torch.tensor(high)])
    gradients = torch.autograd.grad(
        dist.log_prob(torch.tensor(y, dtype=torch.float64)), (mu, scale)
    )
    h = 1e-5

    def oracle(loc: float, sigma: float) -> float:
        return float(
            truncnorm.logpdf(
                y, (low - loc) / sigma, (high - loc) / sigma, loc=loc, scale=sigma
            )
        )

    expected = [
        (oracle(h, 1) - oracle(-h, 1)) / (2 * h),
        (oracle(0, 1 + h) - oracle(0, 1 - h)) / (2 * h),
    ]
    for actual, reference in zip(gradients, expected, strict=True):
        assert actual.item() == pytest.approx(reference, rel=1e-6, abs=1e-7)


def test_truncation_bounds_broadcast_per_batch() -> None:
    low = torch.tensor([0.0, 8.0], dtype=torch.float64)
    high = low + 1
    dist = _truncated_normal_builder(
        [
            torch.zeros(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            low,
            high,
        ]
    )
    y = low + 0.3
    expected = torch.as_tensor(truncnorm.logpdf(y.numpy(), low.numpy(), high.numpy()))
    torch.testing.assert_close(dist.log_prob(y), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "low,high",
    [
        (-2.0, 3.0),
        (8.0, 9.0),
        (-101.0, -100.0),
        (100.0, 101.0),
        (-math.inf, 2.0),
        (8.0, math.inf),
    ],
)
def test_quantiles_and_cdf_match_scipy(low: float, high: float) -> None:
    dist = _truncated_normal_builder(
        [torch.tensor(x, dtype=torch.float64) for x in (0.0, 1.0, low, high)]
    )
    q = torch.tensor([0.0, 1e-10, 0.1, 0.5, 0.9, 1 - 1e-10, 1.0], dtype=torch.float64)
    actual = dist.icdf(q)
    expected = torch.as_tensor(truncnorm.ppf(q.numpy(), low, high))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-9)
    torch.testing.assert_close(dist.cdf(actual), q, rtol=1e-8, atol=1e-9)


def test_quantile_pathwise_gradients() -> None:
    loc = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    low = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)
    high = torch.tensor(9.0, dtype=torch.float64, requires_grad=True)

    def quantile(
        mu: torch.Tensor, sigma: torch.Tensor, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return _truncated_normal_builder([mu, sigma, left, right]).icdf(
            torch.tensor(0.3, dtype=torch.float64)
        )

    assert torch.autograd.gradcheck(quantile, (loc, scale, low, high))


def test_support_and_invalid_intervals() -> None:
    args = [torch.tensor(x, dtype=torch.float64) for x in (0.0, 1.0, 8.0, 9.0)]
    dist = _truncated_normal_builder(args)
    with pytest.raises(ValueError):
        dist.log_prob(torch.tensor(7.0))
    with pytest.raises(ValueError, match="low < high"):
        _truncated_normal_builder(args[:2] + [args[3], args[2]])
    with pytest.raises(ValueError):
        dist.icdf(torch.tensor(-0.1))
    expanded = dist.expand(torch.Size([3]))
    assert expanded.rsample(torch.Size([2])).shape == (2, 3)
    torch.testing.assert_close(
        expanded.log_prob(torch.full((3,), 8.5)),
        dist.log_prob(torch.tensor(8.5)).expand(3),
    )


def test_conditional_family_preserves_tiny_scales() -> None:
    domain = FinSet(name="A", cardinality=1)
    codomain = Euclidean(name="Y", dim=1, low=-1.0, high=1.0)
    family = ConditionalTruncatedNormal(
        domain, codomain, param_source=IdentitySource(param_dim=2)
    )
    raw = torch.tensor(
        [[0.0, math.log(1e-10)]], dtype=torch.float64, requires_grad=True
    )
    y = torch.tensor([[0.0]], dtype=torch.float64)
    actual = family.log_prob(raw, y)
    expected = -math.log(1e-10) - 0.5 * math.log(2 * math.pi)
    assert actual.item() == pytest.approx(expected, abs=1e-10)
    gradient = torch.autograd.grad(actual.sum(), raw)[0]
    torch.testing.assert_close(
        gradient, torch.tensor([[0.0, -1.0]], dtype=torch.float64)
    )


def test_tiny_scale_quantile_has_finite_pathwise_gradients() -> None:
    loc = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(1e-10, dtype=torch.float64, requires_grad=True)
    dist = _truncated_normal_builder(
        [loc, scale, torch.tensor(-1.0), torch.tensor(1.0)]
    )
    q = torch.tensor(0.3, dtype=torch.float64)
    sample = dist.icdf(q)
    gradient = torch.autograd.grad(sample, (loc, scale))
    torch.testing.assert_close(gradient[0], torch.ones_like(loc))
    torch.testing.assert_close(gradient[1], torch.special.ndtri(q))


@pytest.mark.parametrize("literal", [False, True])
def test_compiled_truncated_density_preserves_tail_mass(literal: bool) -> None:
    args = (
        "mu=0.0, sigma=1.0, low=8.0, high=9.0"
        if literal
        else "mu=mu, sigma=sigma, low=8.0, high=9.0"
    )
    source = f"""
object Resp : FinSet 1
object Out : Real 1
program p : Resp -> Out
    sample a <- Normal(loc=0.0, scale=1.0)
    observe y : Resp <- TruncatedNormal({args})
    return a
export p
"""
    actual = loads(source).morphism.log_joint(
        torch.zeros(1, 1),
        {
            "a": torch.tensor(0.0),
            "y": torch.tensor([8.5]),
            "mu": torch.tensor(0.0),
            "sigma": torch.tensor(1.0),
        },
    )
    expected = truncnorm.logpdf(8.5, 8, 9) - 0.5 * math.log(2 * math.pi)
    assert actual.item() == pytest.approx(expected, abs=1e-5)


def test_normal_restriction_scores_and_samples_the_tail() -> None:
    base = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
    )
    dist = Normalize(Restrict(base, low=8.0, high=9.0))
    assert dist.log_prob(torch.tensor(8.5)).item() == pytest.approx(
        truncnorm.logpdf(8.5, 8, 9), abs=1e-10
    )
    with torch.random.fork_rng():
        torch.manual_seed(811)
        draws = dist.sample(torch.Size([10000]))
    assert ((draws > 8) & (draws < 9)).all()
    median = draws.median().item()
    assert truncnorm.ppf(0.48, 8, 9) < median < truncnorm.ppf(0.52, 8, 9)
