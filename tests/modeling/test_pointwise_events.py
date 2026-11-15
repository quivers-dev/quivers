"""Pointwise likelihoods reduce events and arrays while retaining rows."""

import math

import pytest
import torch

from quivers.continuous.inline import FixedDistribution
from quivers.continuous.plate import VectorisedObserve
from quivers.continuous.spaces import Euclidean


@pytest.mark.parametrize("width", [1, 3])
def test_pointwise_event_density_and_gradient_match_independent_normal_formula(width):
    location = torch.linspace(-0.4, 0.6, width, dtype=torch.float64).requires_grad_()
    scale = torch.linspace(0.5, 1.5, width, dtype=torch.float64).requires_grad_()
    family = FixedDistribution(
        Euclidean(name="response", dim=width),
        lambda batch, device: torch.distributions.Independent(
            torch.distributions.Normal(location, scale), 1
        ),
    )
    response = torch.arange(4 * width, dtype=torch.float64).reshape(4, width) / 5
    observation = VectorisedObserve(family, response)
    actual = observation.log_prob_pointwise(torch.zeros(4, 1))
    expected = (
        -0.5 * ((response - location) / scale).square()
        - scale.log()
        - 0.5 * math.log(2 * math.pi)
    ).sum(-1)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (4,)
    torch.testing.assert_close(observation.log_prob(torch.zeros(4, 1)), expected.sum())
    for derivative, oracle in zip(
        torch.autograd.grad(actual.sum(), (location, scale), retain_graph=True),
        torch.autograd.grad(expected.sum(), (location, scale)),
        strict=True,
    ):
        torch.testing.assert_close(derivative, oracle)


@pytest.mark.parametrize("array_shape", [(3,), (2, 3)])
def test_pointwise_arrays_reduce_each_rows_independent_elements(array_shape):
    location = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    family = FixedDistribution(
        Euclidean(name="response", dim=1),
        lambda batch, device: torch.distributions.Normal(location, 0.7),
    )
    response = (
        torch.arange(4 * math.prod(array_shape), dtype=torch.float64).reshape(
            4, *array_shape
        )
        / 10
    )
    observation = VectorisedObserve(family, response, array_shape=array_shape)
    actual = observation.log_prob_pointwise(torch.zeros(4, 1))
    expected = (
        (
            -0.5 * ((response - location) / 0.7).square()
            - math.log(0.7)
            - 0.5 * math.log(2 * math.pi)
        )
        .reshape(4, -1)
        .sum(-1)
    )
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (4,)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), location, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), location)[0],
    )
