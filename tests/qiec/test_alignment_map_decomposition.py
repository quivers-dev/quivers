"""Typed paired decomposition and exact primitive factor witness checks."""

from fractions import Fraction
from itertools import product

import didactic.api as dx
import pytest
import torch
from torch import nn

from quivers.core.relations import RelationLimitError
from quivers.core.relations import alignment_groups
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    JointAlignmentMapConfig,
    JointAlignmentMapDecomposition,
    JointAlignmentMapGapConfig,
    ParameterId,
    StringFactorGraphSpec,
    dumps,
    loads,
)
from tests.test_relation_alignment_map import _all_paths

_MODULE = "paired_alignment_map_test"
_SOURCE = """object Letters : FinSet 2
object Outer : FinSet 2

relation left [algebra=logprob, states=6]
    input Outer
    output Letters
    initial 1 [weight=large]
    arc 1 -> 2 [labels=(0, eps), weight=unit]
    arc 2 -> 5 [labels=(eps, 0), weight=l0]
    arc 2 -> 5 [labels=(eps, 0), weight=l0]
    arc 2 -> 5 [labels=(eps, 1), weight=l1]
    final 5 [weight=cancel]

relation right [algebra=logprob, states=5]
    input Letters
    output Outer
    initial 2 [weight=small]
    arc 2 -> 3 [labels=(0, eps), weight=unit]
    arc 2 -> 3 [labels=(1, eps), weight=r1]
    arc 3 -> 4 [labels=(eps, 1), weight=unit]
    final 4 [weight=unit]

relation prior [algebra=logprob, states=2]
    output Letters
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0), weight=prior_0]
    arc 0 -> 1 [labels=(1), weight=unit]
    final 1 [weight=unit]

string_graph lexical
    variable x : Letters
    variable first : Outer [observed=(0)]
    variable second : Outer [observed=(1)]
    factor left(first, x)
    factor right(x, second)
    factor prior(x)
"""


def _owner(name):
    return ParameterId.derive(_MODULE, "relation_weights", name)


def _fixture():
    compiler = Compiler(parse(module_to_source(parse(_SOURCE))), module_name=_MODULE)
    compiler.compile()
    graph = compiler.string_graph_specs["lexical"]
    weights = {
        _owner(name): nn.Parameter(torch.tensor(score, dtype=torch.float64))
        for name, score in (
            ("unit", 0.0),
            ("large", 1e16),
            ("cancel", -1e16),
            ("small", 1.0),
            ("l0", 1.0),
            ("l1", 0.0),
            ("r1", 0.5),
            ("prior_0", 0.25),
        )
    }
    solver = JointAlignmentMapConfig(
        max_iterations=1,
        max_ngram_order=1,
        max_features=0,
        max_states=128,
        max_work=256,
    )
    policy = JointAlignmentMapDecomposition(
        solver=solver, groups=((1, 0), (2,)), max_group_states=32, max_group_work=128
    )
    return graph, weights, policy


def _primitive_oracle(graph, weights):
    original = graph.bind(weights)
    all_paths = tuple(
        tuple(_all_paths(factor.relation)) for factor in original._factors
    )
    assignments = {}
    for local in product(*all_paths):
        values = {
            name: {var.observed} if var.observed is not None else set()
            for name, var in original._variables.items()
        }
        for factor, (words, *_) in zip(original._factors, local, strict=True):
            for name, word in zip(factor.variables, words, strict=True):
                values[name].add(word)
        if any(len(word) != 1 for word in values.values()):
            continue
        assignment = tuple((name, next(iter(word))) for name, word in values.items())
        score = sum((path[-1] for path in local), Fraction(0))
        assignments[assignment] = max(assignments.get(assignment, score), score)
    return assignments, all_paths


def _verify_original(result, original):
    assert result.factor_traces is not None
    for trace, choices in zip(result.factor_traces, original, strict=True):
        assert (
            trace.words,
            trace.start,
            trace.arc_ids,
            trace.stop,
            trace.score,
        ) in choices
    assert sum(trace.score for trace in result.factor_traces) == result.search.score


def test_named_qvr_graph_pairs_exactly_and_retains_original_boundary_and_arc_ids():
    model, weights, policy = _fixture()
    raw = dumps(policy)
    assert isinstance(policy := loads(raw), JointAlignmentMapDecomposition)
    assert dumps(policy) == raw
    assert isinstance(model := loads(dumps(model)), StringFactorGraphSpec)
    assert policy.objective == "joint_alignment_map"
    exact, original = _primitive_oracle(model, weights)
    assert max(exact.values()) == Fraction(9, 4)
    singleton = policy.solver.solve(model, weights)
    assert singleton.status == "limit" and singleton.upper_bound == Fraction(11, 4)
    unpaired = policy.with_(groups=((0,), (1,), (2,))).solve(model, weights)
    assert unpaired.search == singleton and unpaired.product_shapes == (
        None,
        None,
        None,
    )
    _verify_original(unpaired, original)

    paired = policy.solve(model, weights)
    result = paired.search
    assert result.status == "optimal" and result.certificate is not None
    assert result.assignment is not None and dict(result.assignment)["x"] == (0,)
    assert result.score == result.upper_bound == exact[result.assignment]
    assert result.score == Fraction(9, 4)
    assert len(result.certificate.alignments) == 2
    assert paired.product_shapes[0] is not None and paired.product_shapes[1] is None
    assert paired.product_shapes[0][0] > 2 and paired.product_shapes[0][1] > 2
    _verify_original(paired, original)
    left, right, prior = paired.factor_traces
    assert (left.start, left.stop, left.words) == (1, 5, ((0,), (0,)))
    assert (right.start, right.stop, right.words) == (2, 4, ((0,), (1,)))
    assert (prior.start, prior.stop, prior.words) == (0, 1, ((0,),))
    assert left.arc_ids in ((0, 1), (0, 2))
    assert right.arc_ids == (0, 2)
    # Combining the first boundary pair as a float would round away
    # the point of log mass on the right boundary before cancellation.
    assert (weights[_owner("large")] + weights[_owner("small")]).item() == 1e16
    assert right.score == 1 and left.score == 1
    peak = paired.product_shapes[0][0]
    assert policy.with_(max_group_states=peak).solve(model, weights).search == result
    with pytest.raises(
        RelationLimitError, match="alignment pair .*product: .*max_states"
    ):
        policy.with_(max_group_states=peak - 1).solve(model, weights)


def test_nested_gap_policy_rechecks_original_traces_after_live_update():
    model, weights, policy = _fixture()
    with torch.no_grad():
        weights[_owner("l1")].fill_(2)
    gap = JointAlignmentMapGapConfig(solver=policy.solver, absolute_gap_ratio=(1, 4))
    encoded = policy.with_(solver=gap)
    runtime = loads(dumps(encoded))
    assert isinstance(runtime, JointAlignmentMapDecomposition)
    actual = runtime.solve(model, weights)
    exact, original = _primitive_oracle(model, weights)
    assert actual.search.status == "gap" and actual.search.certificate is None
    assert actual.search.gap_target_met and actual.search.absolute_gap == Fraction(1, 4)
    assert actual.search.score == max(exact.values()) == Fraction(7, 2)
    assert actual.search.upper_bound == Fraction(15, 4)
    assert dict(actual.search.assignment)["x"] == (1,)
    _verify_original(actual, original)


def test_invalid_partitions_multishared_joins_and_exact_limits_refuse():
    model, weights, policy = _fixture()
    with pytest.raises(dx.ValidationError, match="singleton or paired"):
        policy.with_(groups=((0, 1, 2),))
    with pytest.raises(dx.ValidationError, match="distinct nonnegative"):
        policy.with_(groups=((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="partition all original factors"):
        policy.with_(groups=((0, 1), (3,))).solve(model, weights)
    repeated = model.with_(factors=(model.factors[0], *model.factors))
    with pytest.raises(ValueError, match="exactly one tape variable"):
        policy.with_(groups=((0, 1), (2,), (3,))).solve(repeated, weights)
    with pytest.raises(
        RelationLimitError, match="alignment pair .*product: .*max_states"
    ):
        policy.with_(max_group_states=0).solve(model, weights)
    with pytest.raises(
        RelationLimitError, match="alignment pair .*product: .*max_work"
    ):
        policy.with_(max_group_work=0).solve(model, weights)
    with pytest.raises(
        RelationLimitError, match="alignment decoder exceeds max_states"
    ):
        policy.with_(solver=policy.solver.with_(max_states=0)).solve(model, weights)


@pytest.mark.parametrize("corruption", ["boundary", "arc"])
def test_grouped_paths_refuse_wrong_original_provenance(monkeypatch, corruption):
    model, weights, policy = _fixture()
    exact = alignment_groups._materialize_product

    def wrong_product(*args, **kwargs):
        topology, pairs = exact(*args, **kwargs)
        if corruption == "boundary":
            initial = topology.initial[0].state
            _, left = kwargs["boundary_origins"][initial]
            kwargs["boundary_origins"][initial] = (0, left)
        else:
            markers = kwargs["arc_origins"]
            for index, value in enumerate(markers):
                if value == (1, 0):
                    markers[index] = (1, 1)
        return topology, pairs

    monkeypatch.setattr(alignment_groups, "_materialize_product", wrong_product)
    expected = (
        "outside original support"
        if corruption == "boundary"
        else "valid original trace"
    )
    with pytest.raises(ArithmeticError, match=expected):
        policy.solve(model, weights)
