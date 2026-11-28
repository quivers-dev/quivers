"""Check static minimization through a named authored QVR score."""

from fractions import Fraction
from hashlib import sha256

import didactic.api as dx
import pytest
import torch
from torch import nn

from quivers.core.relations import RelationLimitError
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedMinimizedRelationRequest,
    FixedRelationRequest,
    FixedRelationRuntimeProvider,
    ParameterId,
    QiecModule,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target

_SOURCE = """object Letters : FinSet 3

relation acceptor [algebra=markov, states=3]
    output Letters
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0,), weight=eighth]
    arc 0 -> 2 [labels=(1,), weight=quarter]
    arc 1 -> 2 [labels=(2,), weight=four]
    arc 2 -> 1 [labels=(2,), weight=unit]
    final 1 [weight=half]
    final 2 [weight=quarter]

relation_score observe : Real [operation=pair_weight, relation=acceptor]
relation_score unbound : Real [operation=pair_weight]
"""
_MODULE = "static_minimization_test"
_LEGACY_OBSERVATION_DIGEST = (
    "8358d37c096f8daae119276966881200b744ce5815813845b9bcde948066c17a"
)


def _fixture():
    compiler = Compiler(parse(module_to_source(parse(_SOURCE))), module_name=_MODULE)
    compiler.compile()
    relation = compiler.relation_specs["acceptor"]
    values = {
        "unit": 1.0,
        "eighth": 0.125,
        "quarter": 0.25,
        "four": 4.0,
        "half": 0.5,
    }
    parameters = {
        ParameterId.derive(_MODULE, "relation_weights", name): torch.tensor(
            value, dtype=torch.float64
        )
        for name, value in values.items()
    }
    observation = FixedRelationRequest(
        relation=relation,
        words=((1, *(2,) * 17),),
        operation="pair_weight",
        max_states=20,
        max_work=20,
    )
    return compiler.qiec_module, observation, parameters


def _run(module, name, provider):
    value = run_named(
        module,
        name,
        runtime=RuntimeConfiguration(
            selections=(), providers=(CoreRuntimeProvider(), provider)
        ),
    ).value
    assert isinstance(value, torch.Tensor)
    return value


def test_static_request_roundtrip_and_authored_observation_snapshot():
    module, observation, parameters = _fixture()
    request = FixedMinimizedRelationRequest(
        observation=observation, max_minimization_states=3, max_minimization_work=32
    )
    wire = dumps(request)
    assert isinstance(request := loads(wire), FixedMinimizedRelationRequest)
    assert dumps(request) == wire
    assert isinstance(module := loads(dumps(module)), QiecModule)
    static = FixedRelationRuntimeProvider.for_score("observe", request, parameters)
    live = FixedRelationRuntimeProvider.for_score("observe", observation, parameters)
    # The original path starts at state 2 with the second prefix. Its
    # 17-symbol suffix makes nine 2->1 moves and eight 1->2 moves.
    before = Fraction(1, 4) * 4**8 * Fraction(1, 2)
    result = _run(module, "observe", static)
    assert result.item() == before
    assert not result.requires_grad
    assert _run(module, "observe", live).item() == before
    query, work = request.plan(parameters)
    assert query.weight().item() == before
    assert work.active_states == 3

    parameters[ParameterId.derive(_MODULE, "relation_weights", "four")].fill_(1.0)
    assert _run(module, "observe", static).item() == before
    updated = Fraction(1, 4) * Fraction(1, 2)
    assert _run(module, "observe", live).item() == updated
    rebuilt = FixedRelationRuntimeProvider.for_score("observe", request, parameters)
    assert _run(module, "observe", rebuilt).item() == updated

    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        assert any(
            item.computation == "observe" and "__relation_score_observe" in item.reason
            for item in project_for_target(module, discharged, target).obstacles
        )


def test_static_request_rejects_same_typed_substitution_budgets_and_trainability():
    module, observation, parameters = _fixture()
    request = FixedMinimizedRelationRequest(
        observation=observation, max_minimization_states=3, max_minimization_work=32
    )
    with pytest.raises(ExecutionFailure, match="authored topology"):
        changed = observation.with_(relation=observation.relation.with_(n_states=4))
        substitute = FixedRelationRuntimeProvider.for_score(
            "observe", request.with_(observation=changed), parameters
        )
        _run(module, "observe", substitute)
    with pytest.raises(ExecutionFailure, match="authored topology"):
        unbound = FixedRelationRuntimeProvider.for_score("unbound", request, parameters)
        _run(module, "unbound", unbound)
    with pytest.raises(RelationLimitError, match="max_states"):
        request.with_(max_minimization_states=2).plan(parameters)
    with pytest.raises(RelationLimitError, match="max_work"):
        request.with_(max_minimization_work=0).plan(parameters)
    with pytest.raises(RelationLimitError, match="max_states"):
        request.with_(observation=observation.with_(max_states=3)).plan(parameters)
    with pytest.raises(dx.ValidationError, match="pair_weight"):
        request.with_(observation=observation.with_(operation="path_entropy"))
    prefix = ParameterId.derive(_MODULE, "relation_weights", "eighth")
    original = parameters[prefix].clone()
    parameters[prefix].fill_(0.75)
    with pytest.raises(ValueError, match="exactly representable"):
        FixedRelationRuntimeProvider.for_score("observe", request, parameters)
    parameters[prefix].copy_(original)
    owner = ParameterId.derive(_MODULE, "relation_weights", "four")
    parameters[owner] = nn.Parameter(parameters[owner])
    with torch.no_grad(), pytest.raises(TypeError, match="nontrainable"):
        request.plan(parameters)


def test_authored_static_request_leaves_legacy_observation_wire_stable():
    _, observation, _ = _fixture()
    digest = sha256(dumps(observation).encode("utf-8")).hexdigest()
    assert digest == _LEGACY_OBSERVATION_DIGEST
