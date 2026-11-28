"""Checked wire and native program execution of nonnegative feature moments."""

from fractions import Fraction
from math import log

import didactic.api as dx
import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import ProductiveEpsilonCycleError, RelationLimitError
from quivers.dsl import lower_qvr_to_qiec, parse
from quivers.qiec import (
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedRelationFeatureMomentRequest,
    FixedRelationFeatureObservation,
    FixedRelationRuntimeProvider,
    ParameterId,
    QiecModule,
    RationalRelationSpec,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    program_entry,
    run_named,
    run_program,
)
from quivers.transpile import project_for_target
from tests.test_relation_feature_moments import (
    _ARC_FEATURES,
    _START_FEATURES,
    _STOP_FEATURES,
    _WORD,
    _enumeration,
    _fixture,
)


def _wire_logs(rows):
    return tuple(tuple(log(item) if item else None for item in row) for row in rows)


def _request(algebra=MARKOV):
    topology = _fixture()
    owners = tuple(
        ParameterId.derive("typed feature fixture", name)
        for name in topology.weight_names
    )
    spec = RationalRelationSpec.of(topology, algebra, owners)
    observation = FixedRelationFeatureObservation(
        relation=spec,
        words=(_WORD,),
        feature_extent=3,
        arc_logs=_wire_logs(_ARC_FEATURES),
        initial_logs=_wire_logs(_START_FEATURES),
        final_logs=_wire_logs(_STOP_FEATURES),
    )
    request = FixedRelationFeatureMomentRequest(
        observation=observation, max_states=10, max_work=20, max_feature_cells=100
    )
    return request, owners


def _module(request, extent=4):
    handler = request.handler_name
    result = f"Tensor[LogWeight]([{extent}])"
    syntax = parse(f"""object Rows : FinSet 1
handler {handler} for Compute[Unit, {result}] : {result} -> {result} [coverage=total, forwards=none, implementation=foreign]
    apply(argument : Unit) resumes 1

define evaluate() : {result} !{{}} =
    with instance relation : Compute[Unit, {result}] in
        handle relation with {handler} in
            let value <- perform relation.apply(unit)
            return value

program prototype(data:Tensor[Real]([Rows])):Rows->Rows
    return data
""")
    module = lower_qvr_to_qiec(syntax, module_name="fixed_feature_native")
    prototype = program_entry(module, "prototype")
    computation = next(item for item in module.computations if item.name == "evaluate")
    # A native QIEC entry may select a checked computation independently
    # of the surface syntax that authored its foreign handler and body.
    entry = prototype.with_(
        name="features",
        computation=computation.id,
        parameters=(),
        return_names=("moments",),
    )
    return module.with_(entries=(entry,))


def _parameters(algebra, owners):
    weights = (Fraction(1), Fraction(1, 4), Fraction(1, 8), Fraction(1), Fraction(0))
    native = torch.tensor([float(value) for value in weights], dtype=torch.float64)
    if algebra is LOG_PROB:
        native = native.log()
        native[0] -= 1000
    return weights, dict(
        zip(owners, (nn.Parameter(value) for value in native), strict=True)
    )


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_serialized_native_program_calls_the_owned_feature_provider(algebra):
    request, owners = _request(algebra)
    native = _module(request)
    wire = dumps(request)
    request = loads(wire)
    module = loads(dumps(native))
    assert isinstance(request, FixedRelationFeatureMomentRequest)
    assert isinstance(module, QiecModule)
    assert dumps(request) == wire
    assert request.observation.arc_logs[0][1] is None
    owner = request.handler_name
    assert owner == native.handlers[0].name
    assert owner == request.with_(max_work=25).handler_name
    weights, params = _parameters(algebra, owners)
    provider = FixedRelationRuntimeProvider(request, params, handler_name=owner)
    validate = provider.validator_for(request.result_type)
    assert validate is not None
    assert validate(torch.tensor([0.0, -1.0, -torch.inf, 0.0]))
    assert not validate(torch.tensor([0.0, -1.0, torch.inf, 0.0]))
    assert not validate(torch.tensor([0.0, -1.0, torch.nan, 0.0]))
    shift = -1000 if algebra is LOG_PROB else 0
    for weights in (weights, (*weights[:1], Fraction(1, 2), *weights[2:])):
        shared = params[owners[1]]
        with torch.no_grad():
            shared.fill_(
                log(float(weights[1])) if algebra is LOG_PROB else float(weights[1])
            )
        program = run_program(
            module, "features", data={}, sites={}, providers=(provider,)
        )
        assert program.log_joint.item() == 0
        actual = program.value
        assert isinstance(actual, torch.Tensor)
        assert tuple(actual.shape) == (4,)
        values = _enumeration(_fixture(), weights)
        total = sum(p for p, _, _, _ in values)
        moments = [sum(p * scores[k] for p, scores, _, _ in values) for k in range(3)]
        assert actual[0].item() == pytest.approx(
            shift + log(float(total)), abs=2e-12, rel=0
        )
        assert actual[1:3].tolist() == pytest.approx(
            [shift + log(float(value)) for value in moments[:2]], abs=2e-12, rel=0
        )
        assert actual[3].item() == -torch.inf
        (observed,) = torch.autograd.grad(actual[1], shared)
        derivative = sum(
            p * scores[0] * slots.count(1) for p, scores, slots, _ in values
        )
        derivative /= moments[0] if algebra is LOG_PROB else moments[0] * weights[1]
        assert observed.item() == pytest.approx(float(derivative), abs=2e-12, rel=0)

    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    named = run_named(module, "evaluate", runtime=runtime)
    assert named.result_type == request.result_type
    assert named.value[-1].item() == -torch.inf

    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        matched = [
            issue.reason
            for issue in project_for_target(module, discharged, target).obstacles
            if issue.computation == "evaluate"
        ]
        assert matched == [
            f"installs the foreign handler `{owner}`, whose clause `apply` a runtime provider implements"
        ]


def test_decoded_observation_ownership_and_result_shape_are_checked():
    request, owners = _request()
    module = _module(request)
    _, weights = _parameters(MARKOV, owners)
    original = request.observation
    changed_rows = ((0.0, 0.0, None), *original.arc_logs[1:])
    alternatives = (
        original.with_(words=((1, 1),)),
        original.with_(arc_logs=changed_rows),
        original.with_(
            relation=original.relation.with_(
                weight_names=("new_entrance", *original.relation.weight_names[1:])
            )
        ),
    )
    for observation in alternatives:
        alternative = request.with_(observation=observation)
        assert alternative.result_type == request.result_type
        assert alternative.handler_name != request.handler_name
        with pytest.raises(ValueError, match="exact observation handler"):
            FixedRelationRuntimeProvider(
                alternative, {}, handler_name=request.handler_name
            )
        provider = FixedRelationRuntimeProvider(
            alternative, weights, handler_name=alternative.handler_name
        )
        with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
            run_program(module, "features", data={}, sites={}, providers=(provider,))

    provider = FixedRelationRuntimeProvider(
        request, weights, handler_name=request.handler_name
    )
    with pytest.raises(ExecutionFailure, match="incompatible fixed relation handler"):
        run_program(
            _module(request, extent=5),
            "features",
            data={},
            sites={},
            providers=(provider,),
        )


def test_exact_feature_cell_counts_and_native_sparse_refusals():
    request, owners = _request()
    _, weights = _parameters(MARKOV, owners)
    # Seven arcs and three boundary marks, plus seven productive states
    # and eight productive arcs. The dormant zero path is structural.
    # Each row occupies one scalar mass and three feature columns.
    with pytest.raises(RelationLimitError, match="original feature cells"):
        request.with_(max_feature_cells=39).plan({})
    with pytest.raises(RelationLimitError, match="packed feature cells"):
        request.with_(max_feature_cells=99).plan(weights)
    with pytest.raises(RelationLimitError, match="max_states"):
        request.with_(max_states=6).plan(weights)
    with pytest.raises(RelationLimitError, match="max_work"):
        request.with_(max_work=7).plan(weights)
    assert request.plan(weights)().shape == (4,)

    observation = request.observation
    with pytest.raises(dx.ValidationError, match="positive feature extent"):
        observation.with_(feature_extent=0)
    with pytest.raises(dx.ValidationError, match="invalid arc log feature"):
        observation.with_(
            arc_logs=((float("inf"), None, None), *observation.arc_logs[1:])
        )
    with pytest.raises(dx.ValidationError, match="invalid final log feature"):
        observation.with_(final_logs=((0.0,),))

    original = observation.relation
    cyclic = observation.with_(
        relation=original.with_(
            n_states=1, initial=((0, 3),), final=((0, 3),), arcs=((0, 0, (None,), 1),)
        ),
        words=((),),
        arc_logs=((0.0, None, None),),
        initial_logs=None,
        final_logs=None,
    )
    with pytest.raises(ProductiveEpsilonCycleError, match="productive epsilon"):
        request.with_(observation=cyclic).plan(weights)


def test_empty_support_round_trips_native_log_zero():
    request, owners = _request()
    request = request.with_(observation=request.observation.with_(words=((1, 1),)))
    _, weights = _parameters(MARKOV, owners)
    provider = FixedRelationRuntimeProvider(
        request, weights, handler_name=request.handler_name
    )
    module = loads(dumps(_module(request)))
    actual = run_program(
        module, "features", data={}, sites={}, providers=(provider,)
    ).value
    assert actual.tolist() == [-torch.inf] * 4
