"""Finite graph tests of EP tilted weights, projection, and message diagnostics."""

from math import log

import pytest
import torch
from torch import nn

from quivers.core.algebras import MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    FiniteStringBeliefPlan,
    RationalRelation,
    RelationTopology,
    StringFactor,
    StringFactorGraph,
    StringVariable,
)
from tests.test_relation_composition import _INPUT

_WORDS = ((), (0,), (1,), (0, 1))


def _unary_plan(*weights):
    # Each word has one path whose boundary is the supplied weight.
    # The fourth path uses a unit epsilon-free intermediate arc.
    topology = RelationTopology(
        (_INPUT,),
        0,
        5,
        (
            Arc(0, 1, (0,), 4),
            Arc(0, 2, (1,), 4),
            Arc(0, 3, (0,), 4),
            Arc(3, 4, (1,), 4),
        ),
        (Boundary(0, 4),),
        (Boundary(0, 0), Boundary(1, 1), Boundary(2, 2), Boundary(4, 3)),
        ("empty", "a", "b", "ab", "unit"),
    )
    factors = tuple(
        StringFactor(("x",), RationalRelation(topology, value, algebra=MARKOV))
        for value in weights
    )
    graph = StringFactorGraph((StringVariable("x", _INPUT),), factors)
    return FiniteStringBeliefPlan(
        graph, {"x": _WORDS}, max_entries=4 * len(factors), max_states=64, max_work=96
    )


def test_projected_graph_messages_fit_moments_and_preserve_tilt_mass():
    weights = nn.Parameter(
        torch.tensor([0.5, 0.125, 0.125, 0.25, 1.0], dtype=torch.float64)
    )
    plan = _unary_plan(weights)
    result = plan.expectation_propagation(
        order=1, max_sweeps=5, tolerance=1e-10, projection_tolerance=1e-12
    )
    assert result.status == "converged"
    assert result.sweeps == 1
    assert result.message_residual == pytest.approx(0.0, abs=1e-13)
    assert result.log_marginals["x"].exp().tolist() == pytest.approx(
        [25 / 64, 15 / 64, 15 / 64, 9 / 64], abs=3e-12
    )
    (site,) = result.site_projections
    assert (site.factor, site.variable, site.diagnostic) == (0, "x", None)
    assert site.log_tilt_mass.item() == pytest.approx(log(1 / 4))
    assert site.fit is not None and site.fit.moment_residual <= 1e-12
    assert site.fit.divergence.item() > 0

    joint = result.log_marginals["x"][-1].exp()
    (actual,) = torch.autograd.grad(joint, weights)
    target = weights[:4] * torch.stack(
        (weights[4], weights[4] ** 2, weights[4] ** 2, weights[4] ** 3)
    )
    probabilities = target / target.sum()
    independent = (probabilities[1] + probabilities[3]) * (
        probabilities[2] + probabilities[3]
    )
    (expected,) = torch.autograd.grad(independent, weights)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_saturated_family_and_geometric_damping_have_explicit_schedules():
    weights = torch.tensor([0.5, 0.125, 0.125, 0.25, 1.0], dtype=torch.float64)
    plan = _unary_plan(weights)
    saturated = plan.expectation_propagation(
        order=2, max_sweeps=5, projection_tolerance=1e-12
    )
    assert saturated.status == "converged"
    assert saturated.log_marginals["x"].exp().tolist() == pytest.approx(
        weights[:4].tolist()
    )
    damped = plan.expectation_propagation(
        order=2, max_sweeps=1, tolerance=None, damping=0.5, projection_tolerance=1e-12
    )
    expected = weights[:4].sqrt()
    assert damped.sweeps == 1 and damped.status == "limit"
    assert damped.log_marginals["x"].exp().tolist() == pytest.approx(
        (expected / expected.sum()).tolist()
    )


def test_site_projection_failure_does_not_claim_outer_convergence():
    weights = torch.tensor([0.5, 0.125, 0.125, 0.25, 1.0], dtype=torch.float64)
    plan = _unary_plan(weights)
    capped = plan.expectation_propagation(
        order=1, max_sweeps=5, projection_max_iterations=0
    )
    assert capped.status == "projection_failed"
    assert capped.sweeps == 0 and capped.message_residual is None
    assert capped.log_marginals["x"].exp().tolist() == [0.25] * 4
    (site,) = capped.site_projections
    assert site.fit is not None and site.fit.status == "limit"
    assert site.diagnostic == "local projection limit"

    weights[0] = 0.0
    impossible = plan.expectation_propagation(order=1, max_sweeps=5)
    assert impossible.status == "projection_failed"
    (unsupported,) = impossible.site_projections
    assert unsupported.fit is None
    assert (
        unsupported.diagnostic == "tilted weights must be positive on the full support"
    )


def test_nonuniform_cavity_is_divided_out_of_the_projected_site():
    interacting = torch.tensor([0.5, 0.125, 0.125, 0.25, 1.0], dtype=torch.float64)
    separable = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0], dtype=torch.float64)
    plan = _unary_plan(interacting, separable)
    result = plan.expectation_propagation(
        order=1, max_sweeps=2, tolerance=None, projection_tolerance=1e-12
    )
    tilted = interacting[:4] * separable[:4] / separable[:4].sum()
    p = tilted / tilted.sum()
    p_a, p_b = (p[1] + p[3]).item(), (p[2] + p[3]).item()
    expected = [(1 - p_a) * (1 - p_b), p_a * (1 - p_b), (1 - p_a) * p_b, p_a * p_b]
    assert result.sweeps == 2 and result.status == "limit"
    assert result.log_marginals["x"].exp().tolist() == pytest.approx(
        expected, abs=3e-12
    )
    assert result.site_projections[0].log_tilt_mass.item() == pytest.approx(
        tilted.sum().log().item()
    )
