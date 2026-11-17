"""Batch selection preserves raw measures, intrinsic coordinates, and derivatives."""

import torch
from torch.distributions import Independent as TorchIndependent, Normal

from quivers.continuous._batch_selection import select_batch
from quivers.continuous.bijectors import (
    AffineBijector,
    ComposeBijector,
    SigmoidBijector,
    StickBreakingBijector,
)
from quivers.continuous.measure import (
    Independent,
    Mixture,
    Normalize,
    PointMass,
    Pushforward,
    Restrict,
)


def test_nested_measure_selection_retains_raw_normalizers_and_gradients():
    weights = torch.tensor(
        [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]], dtype=torch.float64, requires_grad=True
    )
    means = torch.tensor([-0.3, 0.2, 0.6], dtype=torch.float64, requires_grad=True)
    raw = Mixture(
        weights, [Normal(means, 1.0), PointMass(torch.zeros(3, dtype=torch.float64))]
    )
    original = Normalize(
        Restrict(
            raw, low=torch.tensor([-1.0, -0.5, -0.1], dtype=torch.float64), high=1.5
        )
    )
    active = torch.tensor([True, False, True])
    selected = select_batch(original, active)
    value = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64)
    actual = selected.log_prob(value[active])
    expected = original.log_prob(value)[active]
    torch.testing.assert_close(actual, expected)
    for observed, oracle in zip(
        torch.autograd.grad(actual.sum(), (weights, means)),
        torch.autograd.grad(expected.sum(), (weights, means)),
        strict=True,
    ):
        torch.testing.assert_close(observed, oracle)


def test_composed_bijector_parameters_follow_the_same_batch_selection():
    shift = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    transform = ComposeBijector(
        AffineBijector(
            scale=torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64), shift=shift
        ),
        SigmoidBijector(),
    )
    original = Pushforward(Normal(torch.zeros(3, dtype=torch.float64), 1.0), transform)
    active = torch.tensor([False, True, True])
    selected = select_batch(original, active)
    value = transform.forward(
        torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    ).detach()
    actual, expected = (
        selected.log_prob(value[active]),
        original.log_prob(value)[active],
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), shift)[0],
        torch.autograd.grad(expected.sum(), shift)[0],
    )


def test_nested_independent_selection_retains_event_axes_for_raw_restrictions():
    loc = torch.arange(6, dtype=torch.float64).reshape(3, 2) / 10
    coordinates = Restrict(Normal(loc, 1.0), low=0.0, high=2.0)
    active = torch.tensor([True, False, True])
    for original in (TorchIndependent(coordinates, 1), Independent(coordinates, 1)):
        selected = select_batch(original, active)
        assert selected.batch_shape == (2,)
        assert selected.event_shape == (2,)
        value = torch.ones((3, 2), dtype=torch.float64)
        torch.testing.assert_close(
            selected.log_prob(value[active]), original.log_prob(value)[active]
        )


def test_affine_event_parameters_are_retained_within_each_selected_vector():
    loc = torch.zeros((3, 2), dtype=torch.float64)
    original = Pushforward(
        TorchIndependent(Normal(loc, 1.0), 1),
        AffineBijector(scale=torch.tensor([2.0, 3.0]), shift=torch.tensor([1.0, -1.0])),
    )
    active = torch.tensor([True, False, True])
    selected = select_batch(original, active)
    value = torch.tensor([[1.2, -0.2], [1.5, 0.4], [2.1, 0.7]], dtype=torch.float64)
    torch.testing.assert_close(
        selected.sample(torch.Size((4,))).shape, torch.Size((4, 2, 2))
    )
    # The selected transform retains the intrinsic two-coordinate parameter axis.
    assert selected.bijector.scale.shape == (2, 2)
    torch.testing.assert_close(
        selected.log_prob(value[active]), original.log_prob(value)[active]
    )


def test_stick_breaking_pushforward_retains_event_shape_and_joint_jacobian():
    base = TorchIndependent(
        Normal(torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float64), 1.0), 1
    )
    transform = StickBreakingBijector()
    actual = Pushforward(base, transform)
    assert actual.batch_shape == (2,)
    assert actual.event_shape == (3,)
    oracle = torch.distributions.TransformedDistribution(
        base, [torch.distributions.StickBreakingTransform()]
    )
    value = torch.tensor(
        [[0.2, 0.3, 0.5], [0.4, 0.5, 0.1]], dtype=torch.float64, requires_grad=True
    )
    torch.testing.assert_close(actual.log_prob(value), oracle.log_prob(value))
    torch.testing.assert_close(
        torch.autograd.grad(actual.log_prob(value).sum(), value)[0],
        torch.autograd.grad(oracle.log_prob(value).sum(), value)[0],
    )
    selected = select_batch(actual, torch.tensor([False, True]))
    assert selected.sample(torch.Size((5,))).shape == (5, 1, 3)


def test_composed_scalar_and_coupled_jacobians_reduce_each_event_once():
    base = TorchIndependent(Normal(torch.zeros((2, 2), dtype=torch.float64), 1.0), 1)
    transform = ComposeBijector(
        AffineBijector(scale=2.0, shift=0.5), StickBreakingBijector()
    )
    actual = Pushforward(base, transform)
    oracle = torch.distributions.TransformedDistribution(
        base,
        [
            torch.distributions.StickBreakingTransform(),
            torch.distributions.AffineTransform(0.5, 2.0),
        ],
    )
    value = torch.tensor([[0.9, 1.1, 1.5], [1.3, 1.5, 0.7]], dtype=torch.float64)
    torch.testing.assert_close(actual.log_prob(value), oracle.log_prob(value))
    assert actual.event_shape == (3,)
