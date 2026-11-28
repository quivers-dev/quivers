"""Independent finite synchronous sum-product checks over live string factors."""

from itertools import product

import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV
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
)
from tests.test_relation_composition import _INPUT
from tests.test_string_factor_graph import _graph


@pytest.mark.parametrize("algebra", (MARKOV, LOG_PROB))
def test_finite_messages_reach_exact_tree_marginals_and_share_live_source(algebra):
    variables, factors, source = _graph(algebra)
    graph = StringFactorGraph(variables, factors)
    support = {variable.name: ((0,), (1,)) for variable in variables}
    plan = FiniteStringBeliefPlan(
        graph, support, max_entries=32, max_states=128, max_work=128
    )
    result = plan.run(max_sweeps=16, tolerance=1e-14)
    assert result.status == "converged"
    assert 0 < result.sweeps < 16
    assert source.calls == 1
    assert result.message_residual <= 1e-14
    for name, words in support.items():
        exact = graph.tree_marginal(name, max_states=512, max_work=1024)
        expected = torch.stack([exact.log_prob(word) for word in words])
        assert torch.allclose(
            result.log_marginals[name], expected, atol=1e-13, rtol=1e-13
        )
    with pytest.raises(RelationLimitError, match="max_entries"):
        FiniteStringBeliefPlan(
            graph, support, max_entries=1, max_states=128, max_work=128
        )
    with pytest.raises(RelationLimitError, match="max_states"):
        FiniteStringBeliefPlan(
            graph, support, max_entries=32, max_states=1, max_work=128
        )


def _cycle_data():
    weights = nn.Parameter(
        torch.tensor([3 / 4, 1 / 4, 1 / 8, 7 / 8, 1.0], dtype=torch.float64)
    )
    topology = RelationTopology(
        (_INPUT, _INPUT),
        1,
        2,
        tuple(
            Arc(0, 1, words, i) for i, words in enumerate(product(range(2), repeat=2))
        ),
        (Boundary(0, 4),),
        (Boundary(1, 4),),
        ("aa", "ab", "ba", "bb", "unit"),
    )
    variables = tuple(StringVariable(name, _INPUT) for name in "xyz")
    names = (("x", "y"), ("y", "z"), ("z", "x"))
    relation = RationalRelation(topology, weights, algebra=MARKOV)
    graph = StringFactorGraph(
        variables, tuple(StringFactor(pair, relation) for pair in names)
    )
    plan = FiniteStringBeliefPlan(
        graph,
        {v.name: ((0,), (1,)) for v in variables},
        max_entries=24,
        max_states=128,
        max_work=128,
    )
    return weights, names, plan


def _explicit_sweeps(table, names, count, damping):
    messages = {
        (factor, name): table.new_full((2,), 0.5)
        for factor, scope in enumerate(names)
        for name in scope
    }

    def product_from_other_factors(chosen, variable):
        result = torch.ones_like(table[0])
        for occurrence, value in messages.items():
            factor, name = occurrence
            if factor != chosen and name == variable:
                result = result * value
        return result / result.sum()

    for _ in range(count):
        updated = {}
        for factor, scope in enumerate(names):
            for dim, name in enumerate(scope):
                incoming = product_from_other_factors(factor, scope[1 - dim])
                terms = table @ incoming if dim == 0 else table.mT @ incoming
                candidate = terms / terms.sum()
                key = factor, name
                updated[key] = (1 - damping) * candidate + damping * messages[key]
        messages = updated
    return product_from_other_factors(-1, "x")


def test_loopy_damping_and_gradients_match_the_declared_finite_schedule():
    weights, names, plan = _cycle_data()
    damping = 0.2
    sweeps = 5
    result = plan.run(max_sweeps=sweeps, tolerance=None, damping=damping)
    assert result.status == "limit" and result.sweeps == sweeps
    current = result.log_marginals["x"].exp()
    table = weights[:4].reshape(2, 2) * weights[4] ** 2
    expected = _explicit_sweeps(table, names, sweeps, damping)
    assert torch.allclose(current, expected, atol=1e-13, rtol=1e-13)
    (actual_gradient,) = torch.autograd.grad(current[0], weights)
    (expected_gradient,) = torch.autograd.grad(expected[0], weights)
    assert torch.allclose(actual_gradient, expected_gradient, atol=1e-13, rtol=1e-13)


def test_finite_iteration_reports_undamped_residual_at_the_returned_messages():
    _, _, plan = _cycle_data()
    initial = plan.run(max_sweeps=0)
    assert initial.status == "limit" and initial.sweeps == 0
    assert initial.message_residual > 0
    tiny_step = plan.run(max_sweeps=1, tolerance=1e-6, damping=0.99999)
    assert tiny_step.status == "limit" and tiny_step.sweeps == 1
    assert tiny_step.message_residual == pytest.approx(
        initial.message_residual, rel=1e-4
    )
