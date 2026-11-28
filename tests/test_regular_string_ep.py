"""Exact finite word and site-scale checks for sparse regular EP."""

from itertools import product
from math import log

import pytest
import torch

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import (
    Arc,
    Boundary,
    FiniteStringBeliefPlan,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    StringFactor,
    StringFactorGraph,
    StringVariable,
    regular_string_ep,
)

_TAPE = StringSet(alphabet=FinSet(name="BooleanSymbols", cardinality=2))
_BASE_POLICY = dict(
    max_states=160, max_work=600, max_sweeps=10, projection_max_iterations=400
)


def _relation(arity, scores):
    words = tuple(product(range(2), repeat=arity))
    topology = RelationTopology(
        (_TAPE,) * arity,
        0,
        2,
        tuple(Arc(0, 1, symbols, i) for i, symbols in enumerate(words)),
        (Boundary(0, len(words)),),
        (Boundary(1, len(words)),),
        tuple(str(i) for i in range(len(words) + 1)),
    )
    return RationalRelation(
        topology, torch.tensor([*scores, 0.0], dtype=torch.float64), algebra=LOG_PROB
    )


def _graph(observed=None):
    pair = _relation(2, (0.75, -0.125, -0.25, 0.125))
    left = _relation(1, (-0.05, 0.2))
    right = _relation(1, (0.15, -0.25))
    graph = StringFactorGraph(
        (StringVariable("x", _TAPE), StringVariable("y", _TAPE, observed)),
        (
            StringFactor(("x", "y"), pair),
            StringFactor(("x",), left),
            StringFactor(("y",), right),
        ),
    )
    base = _relation(1, (0.0, 0.0))
    bases = {"x": base, "y": base}
    features = {"x": ((0,), (1,)), "y": ((0,), (1,))}
    return graph, bases, features, (pair, left, right)


def _exact(inputs, observed):
    pair, left, right = inputs
    n = pair._weights().new_full((2, 2), -torch.inf)
    for x, y in product(range(2), repeat=2):
        if observed is None or y == observed[0]:
            n[x, y] = (
                pair.weight(((x,), (y,))) + left.weight(((x,),)) + right.weight(((y,),))
            )
    return n.flatten().softmax(0).reshape(n.shape)


@pytest.mark.parametrize("observed", [None, (1,)])
def test_unpenalized_regular_ep_exact_on_representable_tree_fixture(observed):
    graph, bases, features, raw = _graph(observed)
    result = regular_string_ep(graph, bases, features, penalty=0.0, **_BASE_POLICY)
    assert result.status == "converged" and result.sweeps > 0
    assert result.message_residual is not None and result.message_residual <= 1e-8
    expected = _exact(raw, observed)
    for name, marginal in zip(
        ("x", "y"), (expected.sum(1), expected.sum(0)), strict=True
    ):
        implementation = torch.stack(
            [result.marginals[name].log_prob((value,)) for value in range(2)]
        )
        torch.testing.assert_close(implementation.exp(), marginal, atol=5e-8, rtol=5e-8)

    # A unary site's scale is determined by normalized cavity measure.
    # When y is observed, the pair site's x cavity only integrates y=1.
    unary_x = next(
        site for site in result.sites if site.factor == 1 and site.variable == "x"
    )
    for value in range(2):
        shifts = dict(zip(unary_x.features, unary_x.coefficients, strict=True))
        score = unary_x.log_scale + shifts.get((value,), 0.0)
        torch.testing.assert_close(
            score, raw[1].weight(((value,),)), atol=3e-8, rtol=3e-8
        )


def test_penalty_prunes_sites_and_agrees_with_explicit_finite_ep():
    graph, bases, features, raw = _graph()
    model = regular_string_ep(graph, bases, features, penalty=0.15, **_BASE_POLICY)
    assert model.status == "converged"
    support = {"x": ((0,), (1,)), "y": ((0,), (1,))}
    reference = FiniteStringBeliefPlan(
        graph,
        support,
        max_entries=20,
        max_states=160,
        max_work=600,
    ).expectation_propagation(
        order=1,
        max_sweeps=10,
        projection_penalty=0.15,
        projection_max_iterations=400,
    )
    for name in features:
        actual = torch.stack(
            [model.marginals[name].log_prob((value,)) for value in range(2)]
        )
        torch.testing.assert_close(
            actual, reference.log_marginals[name], atol=3e-7, rtol=3e-7
        )
    assert sum(len(site.features) for site in model.sites) < 8
    strong = regular_string_ep(graph, bases, features, penalty=3.0, **_BASE_POLICY)
    assert strong.status == "converged"
    assert not any(site.features for site in strong.sites)
    for name in features:
        assert strong.marginals[name].log_prob((0,)).item() == pytest.approx(-log(2))


def test_zero_sweeps_local_failure_and_budget_failure_are_distinct():
    graph, bases, features, _ = _graph()
    zero = regular_string_ep(
        graph, bases, features, penalty=0.0, **(_BASE_POLICY | dict(max_sweeps=0))
    )
    assert zero.status == "limit" and zero.sweeps == 0
    assert zero.message_residual is not None and zero.message_residual > 0
    projection_failure = regular_string_ep(
        graph,
        bases,
        features,
        penalty=0.0,
        **(_BASE_POLICY | dict(projection_max_iterations=0)),
    )
    assert projection_failure.status == "projection_failed"
    assert projection_failure.message_residual is None
    assert projection_failure.updates[-1].fit.status == "limit"
    assert projection_failure.marginals["x"].log_prob((0,)).item() == pytest.approx(
        -log(2)
    )
    with pytest.raises(RelationLimitError, match="max_states"):
        regular_string_ep(
            graph, bases, features, penalty=0.0, **(_BASE_POLICY | dict(max_states=1))
        )
