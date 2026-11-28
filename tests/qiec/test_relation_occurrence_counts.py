"""Independent acyclic and geometric oracles for checked occurrence vectors."""

from fractions import Fraction
from itertools import product
from math import log

import didactic.api as dx
import pytest
import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import ProductiveEpsilonCycleError
from quivers.qiec import (
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedRelationArcCountsRequest,
    FixedRelationRuntimeProvider,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target
from tests.qiec.test_relation_runtime import _module, _request


def _run(request, parameters, module=None):
    module = (
        _module(f"Tensor[Real]([{len(request.arc_ids)}])") if module is None else module
    )
    runtime = RuntimeConfiguration(
        selections=(),
        providers=(
            CoreRuntimeProvider(),
            FixedRelationRuntimeProvider(request, parameters, handler_name="channel"),
        ),
    )
    result = run_named(module, "evaluate", (None,), runtime=runtime)
    assert result.result_type == request.result_type
    assert isinstance(result.value, torch.Tensor)
    return result.value


def _oracle(a, b, ids):
    totals = {
        trace: (a if trace[0] < 2 else b) * (a if trace[1] < 2 else b)
        for trace in product(range(3), repeat=2)
    }
    normalizer = sum(totals.values())
    gradients = [[], []]
    means = []
    for event in ids:
        mass = sum(total * trace.count(event) for trace, total in totals.items())
        means.append(mass / normalizer)
        for owner, parameter in enumerate((a, b)):
            path_derivatives = {
                trace: total
                * sum((arc == 2) == bool(owner) for arc in trace)
                / parameter
                for trace, total in totals.items()
            }
            d_sum = sum(path_derivatives.values())
            d_mass = sum(
                value * trace.count(event) for trace, value in path_derivatives.items()
            )
            gradients[owner].append(
                (d_mass * normalizer - mass * d_sum) / normalizer**2
            )
    return means, gradients


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_ordered_occurrence_counts_keep_shared_owners_distinct_across_offsets(algebra):
    base, parameters, (a_id, b_id, _) = _request("pair_weight", algebra)
    relation = base.relation.with_(
        n_states=1,
        arcs=tuple((0, 0, (0, 0), slot) for slot in (0, 1, 2)) + ((0, 0, (0, 1), 2),),
        final=((0, 3),),
    )
    ids = (2, 0, 1, 0, 3)
    request = FixedRelationArcCountsRequest(
        relation=relation,
        words=((0, 0), (0, 0)),
        arc_ids=ids,
        max_states=16,
        max_work=16,
    )
    wire = loads(dumps(request))
    assert isinstance(wire, FixedRelationArcCountsRequest) and wire == request
    obtained = _run(wire, parameters)
    exact, gradients = _oracle(Fraction(1, 8), Fraction(1, 4), ids)
    assert obtained.tolist() == pytest.approx([float(value) for value in exact])
    observed = torch.autograd.grad(
        obtained[1:4].sum(), (parameters[a_id], parameters[b_id])
    )
    expected = [sum(owner[1:4]) for owner in gradients]
    if algebra is LOG_PROB:
        expected[0] *= Fraction(1, 8)
        expected[1] *= Fraction(1, 4)
    assert [derivative.item() for derivative in observed] == pytest.approx(
        [float(v) for v in expected]
    )

    with pytest.raises(dx.ValidationError, match="valid original arc IDs"):
        request.with_(arc_ids=(False,))
    with pytest.raises(dx.ValidationError, match="valid original arc IDs"):
        request.with_(arc_ids=(len(relation.arcs),))
    assert _run(request.with_(arc_ids=()), parameters).shape == (0,)
    wrong = _module("Tensor[Real]([4])")
    with pytest.raises(ExecutionFailure, match="incompatible fixed relation handler"):
        _run(wire, parameters, wrong)

    module = _module("Tensor[Real]([5])")
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        assert any(
            x.computation == "evaluate" and "foreign handler `channel`" in x.reason
            for x in project_for_target(module, discharged, target).obstacles
        )


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_cyclic_occurrence_counts_and_gradients_match_geometric_sum(algebra):
    base, parameters, (a_id, _, _) = _request("pair_weight", algebra)
    relation = base.relation.with_(
        n_states=1,
        arcs=((0, 0, (None, None), 0), (0, 0, (None, None), 1)),
        final=((0, 3),),
    )
    request = FixedRelationArcCountsRequest(
        relation=relation,
        words=((), ()),
        arc_ids=(1, 0, 1),
        max_states=16,
        max_work=16,
        max_closure_size=2,
    )
    expected = Fraction(1, 8) / (1 - 2 * Fraction(1, 8))
    result = _run(request, parameters)
    assert result.tolist() == pytest.approx([float(expected)] * 3)
    (derivative,) = torch.autograd.grad(result[0], parameters[a_id])
    expected_derivative = (1 - 2 * Fraction(1, 8)) ** -2
    if algebra is LOG_PROB:
        expected_derivative *= Fraction(1, 8)
    assert derivative.item() == pytest.approx(float(expected_derivative))
    with pytest.raises(ProductiveEpsilonCycleError, match="productive"):
        _run(request.with_(max_closure_size=None), parameters)
    with torch.no_grad():
        parameters[a_id].fill_(1.0 if algebra is MARKOV else log(1.0))
    with pytest.raises(ExecutionFailure, match="diverg"):
        _run(request, parameters)
