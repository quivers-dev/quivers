"""Wire and reference-machine checks for finite alignment-marginal ranking."""

from fractions import Fraction
from math import log

import didactic.api as dx
import pytest
import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.relations import ProductiveEpsilonCycleError, RelationLimitError
from quivers.qiec import (
    LOG_WEIGHT,
    CoreRuntimeProvider,
    FixedRelationMarginalMapRequest,
    FixedRelationRuntimeProvider,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target
from tests.qiec.test_relation_runtime import _module, _request


def _query():
    base, parameters, ids = _request("pair_weight")
    request = FixedRelationMarginalMapRequest(
        relation=base.relation,
        inputs=((0,),),
        candidates=(((0,),), ((1,),)),
        max_states=20,
        max_work=30,
    )
    return request, parameters, ids


def test_serialized_marginal_map_score_uses_total_alignment_mass_and_original_owners():
    request, parameters, (a_id, b_id, _) = _query()
    # Candidate 0 has two distinct pathways with one shared owner.
    # Candidate 1 has the single greatest alignment but a lower path sum.
    a, b = parameters[a_id], parameters[b_id]
    with torch.no_grad():
        a.fill_(0.375)
        b.fill_(0.5)
    decoded = loads(dumps(request))
    assert isinstance(decoded, FixedRelationMarginalMapRequest)
    assert decoded == request and decoded.result_type == LOG_WEIGHT
    chosen = decoded.choose(parameters)
    assert chosen is not None
    assert chosen.output == ((0,),)
    assert chosen.exact_weight == 2 * Fraction(3, 8)
    oracle = request.relation.bind(parameters)
    first = oracle.query(((0,), (0,))).best_path()
    second = oracle.query(((0,), (1,))).best_path()
    assert first is not None and second is not None and second.weight > first.weight

    module = _module("LogWeight")
    provider = FixedRelationRuntimeProvider(decoded, parameters, handler_name="channel")
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )

    def evaluate() -> torch.Tensor:
        result = run_named(module, "evaluate", (None,), runtime=runtime)
        assert (
            isinstance(result.value, torch.Tensor) and result.result_type == LOG_WEIGHT
        )
        return result.value

    score = evaluate()
    assert score.item() == pytest.approx(log(0.75))
    gradient = torch.autograd.grad(score, (a, b))
    assert [value.item() for value in gradient] == pytest.approx([8 / 3, 0])

    # Tie order remains the declared candidate order, including a
    # request that starts with the formerly second output.
    with torch.no_grad():
        b.fill_(0.75)
    reversed_request = decoded.with_(candidates=tuple(reversed(decoded.candidates)))
    tie = reversed_request.choose(parameters)
    assert tie is not None and tie.output == ((1,),)
    tie_score = evaluate()
    tie_gradient = torch.autograd.grad(tie_score, (a, b))
    assert [value.item() for value in tie_gradient] == pytest.approx([8 / 3, 0])

    with torch.no_grad():
        b.fill_(1.0)
    shifted = evaluate()
    assert shifted.item() == 0
    gradient = torch.autograd.grad(shifted, (a, b))
    assert [value.item() for value in gradient] == pytest.approx([0, 1])

    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        assert any(
            obstacle.computation == "evaluate"
            and "foreign handler `channel`" in obstacle.reason
            for obstacle in project_for_target(module, discharged, target).obstacles
        )


def test_typed_marginal_map_checks_all_candidates_and_preserves_budget_errors():
    request, parameters, _ = _query()
    marginal_empty = request.with_(inputs=((1,),))
    assert marginal_empty.choose(parameters) is None
    module = _module("LogWeight")
    provider = FixedRelationRuntimeProvider(
        marginal_empty, parameters, handler_name="channel"
    )
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    value = run_named(module, "evaluate", (None,), runtime=runtime).value
    assert isinstance(value, torch.Tensor)
    assert value.item() == -torch.inf and not value.requires_grad

    with pytest.raises(dx.ValidationError, match="invalid symbol"):
        request.with_(candidates=(*request.candidates, ((2,),)))
    with pytest.raises(dx.ValidationError, match="requires Markov"):
        request.with_(relation=request.relation.with_(algebra=LOG_PROB.subject))
    with pytest.raises(dx.ValidationError, match="one word per input tape"):
        request.with_(inputs=())
    with pytest.raises(RelationLimitError, match="max_states"):
        request.with_(max_states=3).choose(parameters)
    with pytest.raises(RelationLimitError, match="max_work"):
        request.with_(max_work=4).choose(parameters)
    cyclic = request.with_(
        relation=request.relation.with_(
            arcs=(*request.relation.arcs, (1, 1, (None, None), 0))
        )
    )
    with pytest.raises(ProductiveEpsilonCycleError, match="acyclic productive"):
        cyclic.choose(parameters)
