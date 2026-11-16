"""Endpoint membership and raw mass compose across continuous laws and atoms."""

import math

import pytest
import torch
from torch.distributions import Normal, Poisson

from quivers.continuous._censored import Censored
from quivers.continuous._interval import log_interval_mass
from quivers.continuous.measure import Mixture, Normalize, PointMass, Restrict


@pytest.mark.parametrize("lower_closed", [False, True])
@pytest.mark.parametrize("upper_closed", [False, True])
def test_poisson_endpoint_flags_match_explicit_sum(lower_closed, upper_closed):
    base = Poisson(torch.tensor(2.3, dtype=torch.float64))
    points = torch.arange(
        1 if lower_closed else 2, 4 if upper_closed else 3, dtype=torch.float64
    )
    expected = base.log_prob(points).logsumexp(0)
    actual = log_interval_mass(
        base, 1.0, 3.0, lower_closed=lower_closed, upper_closed=upper_closed
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "lower_closed,upper_closed,expected",
    [(True, True, 1.0), (False, True, 0.0), (True, False, 0.0), (False, False, 0.0)],
)
def test_singleton_atom_requires_both_closed_endpoints(
    lower_closed, upper_closed, expected
):
    mass = log_interval_mass(
        PointMass(2), 2, 2, lower_closed=lower_closed, upper_closed=upper_closed
    )
    assert mass.exp().item() == expected


def test_nested_restriction_intersects_bounds_and_endpoint_flags():
    base = Poisson(torch.tensor(2.3, dtype=torch.float64))
    first = Restrict(base, 1, 4, lower_closed=False)
    second = Restrict(first, 2, 3, upper_closed=False)
    torch.testing.assert_close(
        second.log_normalizer(), base.log_prob(torch.tensor(2.0))
    )
    torch.testing.assert_close(
        Normalize(second).log_prob(torch.tensor(2.0)),
        torch.tensor(0.0, dtype=torch.float64),
    )
    assert second.log_prob(torch.tensor(3.0)).isneginf()


def test_mixture_raw_event_mass_normalizes_exactly_once():
    base = Normal(torch.tensor(0.0, dtype=torch.float64), 1.0)
    half = Restrict(base, 0, None)
    mixture = Mixture(
        torch.tensor([2.0, 3.0], dtype=torch.float64), [half, PointMass(0.0)]
    )
    expected = math.log(4.0)
    assert mixture.log_normalizer().item() == pytest.approx(expected)
    mass = Restrict(mixture, 0, 1).log_normalizer()
    normal_mass = (base.cdf(torch.tensor(1.0)) - 0.5).item()
    assert mass.item() == pytest.approx(math.log(2 * normal_mass + 3))
    normalized = Normalize(mixture)
    assert log_interval_mass(normalized, 0, 1).item() == pytest.approx(
        mass.item() - expected
    )
    # The atom is excluded by an open lower endpoint.
    assert log_interval_mass(mixture, 0, 1, lower_closed=False).item() == pytest.approx(
        math.log(2 * normal_mass)
    )


def test_censored_upper_atom_includes_preexisting_atom_at_boundary():
    clamped = Censored(PointMass(1.0), 0.0, 1.0)
    assert clamped.log_prob(torch.tensor(1.0)).item() == 0.0
    assert clamped.log_cdf_left(torch.tensor(1.0)).isneginf()
    assert clamped.log_survival_function_left(torch.tensor(1.0)).item() == 0.0


def test_batched_endpoint_flags():
    base = Poisson(torch.tensor(1.4, dtype=torch.float64))
    closed = torch.tensor([False, True])
    mass = log_interval_mass(base, 0, 1, lower_closed=closed)
    expected = torch.stack(
        [
            base.log_prob(torch.tensor(1.0)),
            torch.logsumexp(base.log_prob(torch.tensor([0.0, 1.0])), 0),
        ]
    )
    torch.testing.assert_close(mass, expected)


def test_poisson_infinite_limits_have_unit_mass_and_zero_rate_gradient():
    rate = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    mass = log_interval_mass(Poisson(rate), -torch.inf, torch.inf)
    assert mass.item() == 0
    assert torch.autograd.grad(mass, rate)[0].item() == 0


def test_mixture_sampling_uses_weighted_component_masses():
    # Raw component masses differ by four; their input weights are equal.
    first = Mixture(torch.tensor([1.0], dtype=torch.float64), [PointMass(0.0)])
    second = Mixture(torch.tensor([4.0], dtype=torch.float64), [PointMass(1.0)])
    distribution = Normalize(
        Mixture(torch.tensor([0.5, 0.5], dtype=torch.float64), [first, second])
    )
    with torch.random.fork_rng():
        torch.manual_seed(1234)
        mean = distribution.sample(torch.Size([20000])).mean().item()
    assert mean == pytest.approx(0.8, abs=0.015)


def test_mixture_skips_zero_mass_components_and_handles_vector_events():
    empty = Restrict(PointMass(0.0), 1, 2)
    distribution = Mixture(torch.tensor([1.0, 1.0]), [empty, PointMass(3.0)])
    assert (distribution.sample(torch.Size([8])) == 3).all()
    vector = Mixture(
        torch.tensor([1.0, 0.0]),
        [
            torch.distributions.Independent(Normal(torch.tensor([1.0, 2.0]), 0.01), 1),
            torch.distributions.Independent(Normal(torch.zeros(2), 1.0), 1),
        ],
    )
    assert vector.sample(torch.Size([5])).shape == (5, 2)


def test_mixture_only_requires_positive_mass_on_selected_batch_rows():
    first = Restrict(PointMass(torch.tensor([0.0, 1.0])), 0, 0)
    second = Restrict(PointMass(torch.tensor([0.0, 1.0])), 1, 1)
    mixture = Mixture(torch.ones(2, 2), [first, second])
    expected = torch.tensor([[0.0, 1.0]]).expand(4, 2)
    torch.testing.assert_close(mixture.sample(torch.Size([4])), expected)


def test_empty_restriction_refuses_sampling_immediately():
    with pytest.raises(ValueError, match="finite positive mass"):
        Restrict(PointMass(0.0), 1, 2).sample()


def test_mixture_restriction_keeps_zero_mass_components_without_normalizing_them():
    mixture = Mixture(torch.tensor([0.25, 0.75]), [PointMass(0.0), PointMass(1.0)])
    conditional = mixture.restrict_to(1, 2)
    assert conditional.log_normalizer().item() == pytest.approx(0.0, abs=1e-6)
    assert conditional.log_prob(torch.tensor(1.0)).item() == pytest.approx(
        0.0, abs=1e-6
    )
    assert (conditional.sample(torch.Size([5])) == 1).all()


def test_mixture_mass_and_density_broadcast_components_with_different_batch_shapes():
    weights = torch.tensor([0.3, 0.7], dtype=torch.float64)
    mixture = Mixture(
        weights,
        [Normal(torch.tensor([0.0, 1.0], dtype=torch.float64), 1.0), PointMass(0.0)],
    )
    expected = torch.stack(
        [
            torch.logsumexp(
                torch.stack(
                    [
                        weights[0].log() + Normal(loc, 1.0).log_prob(torch.tensor(0.0)),
                        weights[1].log(),
                    ]
                ),
                0,
            )
            for loc in torch.tensor([0.0, 1.0], dtype=torch.float64)
        ]
    )
    torch.testing.assert_close(mixture.log_prob(torch.tensor(0.0)), expected)
    torch.testing.assert_close(
        mixture.log_normalizer(), torch.zeros(2, dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "name,bounds",
    [
        ("ExpBijector", (0.5, 2.0)),
        ("ExpBijector", (-2.0, 2.0)),
        ("SigmoidBijector", (0.2, 0.8)),
        ("SigmoidBijector", (-1.0, 2.0)),
        ("SoftplusBijector", (0.1, 2.0)),
    ],
)
def test_pushforward_event_mass_uses_scalar_preimage(name, bounds):
    from quivers.continuous import bijectors
    from quivers.continuous.measure import Pushforward
    from scipy import stats

    bijector = getattr(bijectors, name)()
    base = Normal(torch.tensor(0.3, dtype=torch.float64), 1.2)
    transformed = Pushforward(base, bijector)
    low, high = bounds
    image_low, image_high = bijector._image_bounds
    inverse_low = (
        -math.inf
        if low <= image_low
        else float(bijector.inverse(torch.tensor(low, dtype=torch.float64)))
    )
    inverse_high = (
        math.inf
        if high >= image_high
        else float(bijector.inverse(torch.tensor(high, dtype=torch.float64)))
    )
    expected = stats.norm.cdf((inverse_high - 0.3) / 1.2) - stats.norm.cdf(
        (inverse_low - 0.3) / 1.2
    )
    assert log_interval_mass(transformed, low, high).item() == pytest.approx(
        math.log(expected), abs=1e-12
    )


def test_composed_and_inverse_bijector_event_mass():
    from quivers.continuous.bijectors import (
        ComposeBijector,
        ExpBijector,
        SigmoidBijector,
    )
    from quivers.continuous.measure import Pushforward

    base = Normal(torch.tensor(0.0, dtype=torch.float64), 1.0)
    composed = ComposeBijector(ExpBijector(), SigmoidBijector())
    transformed = Pushforward(base, composed)
    assert log_interval_mass(transformed, 1, math.e).item() == pytest.approx(0.0)
    inverse = Pushforward(
        torch.distributions.LogNormal(torch.tensor(0.0, dtype=torch.float64), 1.0),
        ExpBijector().inv(),
    )
    torch.testing.assert_close(
        log_interval_mass(inverse, -1, 1), log_interval_mass(base, -1, 1)
    )


def test_independent_event_mass_reduces_only_declared_event_axes():
    from quivers.continuous.measure import Independent

    loc = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    base = Normal(loc, 1.0)
    independent = Independent(base, 1)
    expected = log_interval_mass(base, 0, 2).sum(-1)
    torch.testing.assert_close(log_interval_mass(independent, 0, 2), expected)


def test_restricted_support_matches_open_endpoints_and_discrete_base():
    restricted = Restrict(
        Poisson(torch.tensor(2.0)), 0, 3, lower_closed=False, upper_closed=False
    )
    values = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    assert restricted.support.check(values).tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    assert restricted.support.is_discrete
    continuous = Restrict(
        torch.distributions.Exponential(torch.tensor(1.0)), -1, 2, upper_closed=False
    )
    assert not continuous.support.check(torch.tensor(-0.5))
    transformed = torch.distributions.biject_to(continuous.support)(
        torch.tensor(-100.0)
    )
    assert transformed >= 0


def test_normalized_mixture_and_censoring_share_atom_aware_half_lines():
    normal = Normal(torch.tensor(0.0, dtype=torch.float64), 1.0)
    raw = Mixture(
        torch.tensor([2.0, 3.0], dtype=torch.float64), [normal, PointMass(1.0)]
    )
    probability = Normalize(raw)
    value = torch.tensor(1.0, dtype=torch.float64)
    torch.testing.assert_close(
        probability.log_cdf(value).exp(), (2 * normal.cdf(value) + 3) / 5
    )
    torch.testing.assert_close(
        probability.log_cdf_left(value).exp(), 2 * normal.cdf(value) / 5
    )
    clamped = Censored(raw, 0.0, 1.0)
    torch.testing.assert_close(
        clamped.log_prob(value).exp(), (2 * (1 - normal.cdf(value)) + 3) / 5
    )
    assert log_interval_mass(clamped, 0, 1).item() == pytest.approx(0.0, abs=1e-12)


def test_float32_atom_does_not_round_a_wider_precision_event_bound():
    atom = PointMass(torch.tensor([1.0], dtype=torch.float32))
    assert log_interval_mass(atom, 1.0 + 1e-10, 2.0).isneginf().all()
    assert log_interval_mass(atom, 0.0, 1.0 - 1e-10).isneginf().all()


def test_inverse_composition_uses_existing_component_interval_maps():
    from quivers.continuous.bijectors import (
        ComposeBijector,
        ExpBijector,
        SigmoidBijector,
    )

    composed = ComposeBijector(ExpBijector(), SigmoidBijector())
    low = torch.tensor(-1.0, dtype=torch.float64)
    high = torch.tensor(1.0, dtype=torch.float64)
    mapped = composed.inv()._interval_preimage(low, high)
    torch.testing.assert_close(mapped[0], composed.forward(low))
    torch.testing.assert_close(mapped[1], composed.forward(high))


def test_pushforward_interval_refuses_nan_bounds():
    from quivers.continuous.bijectors import ExpBijector
    from quivers.continuous.measure import Pushforward

    with pytest.raises(ValueError, match="NaN"):
        log_interval_mass(
            Pushforward(Normal(0.0, 1.0), ExpBijector()), float("nan"), 1.0
        )


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([-0.1, 1.1]),
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([float("inf"), 1.0]),
    ],
)
def test_raw_mixture_validates_weights_before_scoring(weights):
    with pytest.raises(ValueError, match="finite nonnegative"):
        Mixture(weights, [PointMass(0.0), PointMass(1.0)])


def test_raw_mixture_requires_common_event_shape():
    with pytest.raises(ValueError, match="same event shape"):
        Mixture(
            torch.ones(2),
            [
                Normal(0.0, 1.0),
                torch.distributions.Independent(Normal(torch.zeros(2), 1.0), 1),
            ],
        )


def test_bijector_extended_images_keep_finite_support_limits():
    from quivers.continuous.bijectors import (
        ComposeBijector,
        ExpBijector,
        SigmoidBijector,
        LogBijector,
    )

    low = torch.tensor(-torch.inf, dtype=torch.float64)
    high = -low
    image = ComposeBijector(ExpBijector(), SigmoidBijector())._interval_image(low, high)
    assert image[0].item() == 1
    assert image[1].item() == pytest.approx(math.e)
    preimage = LogBijector()._interval_preimage(low, high)
    assert preimage[0].item() == 0
    assert preimage[1].isposinf()


@pytest.mark.parametrize(
    "low,high,expected",
    [
        (2.0, 3.0, 0.0),
        (-2.0, -1.0, 0.0),
        (0.0, math.inf, 1.0),
        (-math.inf, math.inf, 1.0),
        (0.2, 0.5, 0.3),
    ],
)
def test_uniform_interval_masks_outside_support_before_tail_evaluation(
    low, high, expected
):
    base = torch.distributions.Uniform(torch.tensor(0.0, dtype=torch.float64), 1.0)
    assert log_interval_mass(base, low, high).exp().item() == pytest.approx(expected)


def test_uniform_interval_keeps_tail_precision_and_parameter_derivatives():
    parameters = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)

    def function(p):
        return log_interval_mass(torch.distributions.Uniform(p[0], p[1]), 0.9, 0.95)

    assert torch.autograd.gradcheck(function, (parameters,))
    assert torch.autograd.gradgradcheck(function, (parameters,))
    mass = log_interval_mass(torch.distributions.Uniform(0.0, 1.0), 1 - 1e-12, None)
    assert mass.item() == pytest.approx(math.log(1 - (1 - 1e-12)))
