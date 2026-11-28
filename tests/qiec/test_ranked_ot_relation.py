"""Check native capped ranked preferences against original-path enumeration."""

from fractions import Fraction
from math import log

import didactic.api as dx
import pytest
import torch
from torch import nn

from quivers.core.relations import RelationLimitError
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedRankedOTRelationRequest,
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

_MODULE = "capped_ranked_ot_test"
_SOURCE = """object Inputs : FinSet 2
object Outputs : FinSet 2

relation gen [algebra={algebra}, states=4]
    input Inputs
    output Outputs
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0, 0), weight=a]
    arc 0 -> 1 [labels=(0, 1), weight=b]
    arc 0 -> 1 [labels=(0, 1), weight=b]
    arc 1 -> 2 [labels=(0, 0), weight=suffix]
    arc 1 -> 2 [labels=(1, 0), weight=unit]
    arc 0 -> 2 [labels=(1, eps), weight=direct]
    arc 2 -> 3 [labels=(eps, 1), weight=insertion]
    final 2 [weight=unit]
    final 3 [weight=unit]
"""
_ARCS = (
    (0, 1, 0, 0, "a", (0, 2)),
    (0, 1, 0, 1, "b", (1, 0)),
    (0, 1, 0, 1, "b", (1, 0)),
    (1, 2, 0, 0, "suffix", (1, 0)),
    (1, 2, 1, 0, "unit", (0, 0)),
    (0, 2, 1, None, "direct", (0, 0)),
    (2, 3, None, 1, "insertion", (0, 0)),
)
_PROB = {
    "unit": Fraction(1),
    "a": Fraction(7, 8),
    "b": Fraction(1, 8),
    "suffix": Fraction(1, 2),
    "direct": Fraction(1, 4),
    "insertion": Fraction(1, 4),
}


def _owner(name):
    return ParameterId.derive(_MODULE, "relation_weights", name)


def _value(probability, algebra):
    if algebra == "logprob":
        return log(probability) if probability else float("-inf")
    return float(probability)


def _compile(source):
    compiler = Compiler(parse(module_to_source(parse(source))), module_name=_MODULE)
    compiler.compile()
    return compiler


def _fixture(algebra):
    source = _SOURCE.format(algebra=algebra)
    model = _compile(source).relation_specs["gen"]
    request = FixedRankedOTRelationRequest(
        relation=model,
        words=((0, 0), (1, 0)),
        caps=(1, 2),
        violations=tuple(arc[-1] for arc in _ARCS),
        max_rank_states=100,
        max_rank_work=500,
        max_states=100,
        max_work=100,
    )
    parameters = {
        _owner(name): nn.Parameter(
            torch.tensor(_value(value, algebra), dtype=torch.float64)
        )
        for name, value in _PROB.items()
    }
    return source, request, parameters


def _all_paths(probabilities):
    """List every original finite path, then calculate complete ranks."""
    reached = [(0, (), (), (0, 0), Fraction(1))]
    accepted = []
    for state, x, y, rank, weight in reached:
        if state in (2, 3):
            accepted.append((x, y, rank, weight))
        for source, target, a, b, owner, (r, s) in _ARCS:
            if source == state:
                new_x = x if a is None else (*x, a)
                new_y = y if b is None else (*y, b)
                reached.append(
                    (
                        target,
                        new_x,
                        new_y,
                        (rank[0] + r, rank[1] + s),
                        weight * probabilities[owner],
                    )
                )
    return accepted


def _winning_pairs(paths, caps):
    ranked = [
        (
            x,
            y,
            tuple(min(cap, value) for cap, value in zip(caps, rank, strict=True)),
            weight,
        )
        for x, y, rank, weight in paths
    ]
    inputs = {x for x, *_ in ranked}
    best = {x: min(rank for i, _, rank, _ in ranked if i == x) for x in inputs}
    pairs = {}
    for x, y, rank, weight in ranked:
        if rank == best[x]:
            pairs[x, y] = pairs.get((x, y), Fraction(0)) + weight
    return pairs


def _native(source, request):
    name = request.handler_name
    result = request.result_type.constructor.name
    compiler = _compile(
        source
        + f"""
instance ranked : Compute[Unit, {result}]
handler {name} for Compute[Unit, {result}] : {result} -> {result} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define observe : {result} !{{}} =
    handle ranked with {name} in
        let value <- perform ranked.apply(unit)
        return value
"""
    )
    assert compiler.relation_specs["gen"] == request.relation
    module = loads(dumps(compiler.qiec_module))
    assert isinstance(module, QiecModule)
    return module


def _run(module, provider):
    result = run_named(
        module,
        "observe",
        runtime=RuntimeConfiguration(
            selections=(), providers=(CoreRuntimeProvider(), provider)
        ),
    )
    assert isinstance(result.value, torch.Tensor)
    return result.value


@pytest.mark.parametrize("algebra", ["markov", "logprob"])
def test_native_ranked_preference_filters_paths_before_weight_scoring(algebra):
    source, request, parameters = _fixture(algebra)
    wire = dumps(request)
    assert isinstance(request := loads(wire), FixedRankedOTRelationRequest)
    assert dumps(request) == wire
    assert request.relation.tapes == (("Inputs", 2), ("Outputs", 2))
    assert request.relation.arcs[1] == request.relation.arcs[2]
    paths = _all_paths(_PROB)
    ranked = _winning_pairs(paths, request.caps)
    candidates = {(x, y) for x, y, _, _ in paths}
    for pair in candidates | {((0,), (0,)), ((), ())}:
        actual = request.with_(words=pair).plan(parameters).weight()
        assert actual.item() == pytest.approx(_value(ranked.get(pair, 0), algebra))

    # The two prefixes are (0,2) and (1,0). A common suffix with (1,0)
    # makes the second prefix win under cap one on the first coordinate.
    uncapped = _winning_pairs(paths, (10, 10))
    assert request.words not in uncapped
    assert ranked[request.words] == Fraction(1, 8)
    loser = ((0, 0), (0, 0))
    assert loser not in ranked
    assert sum(w for x, y, _, w in paths if (x, y) == loser) == Fraction(7, 16)
    assert ranked[((0, 1), (0, 0))] == Fraction(7, 8)
    assert ranked[((1,), ())] == Fraction(1, 4)
    input_total = sum(mass for (x, _), mass in ranked.items() if x == request.words[0])
    assert ranked[request.words] / input_total == Fraction(4, 5)

    module = _native(source, request)
    provider = FixedRelationRuntimeProvider(
        request, parameters, handler_name=request.handler_name
    )
    winner = _run(module, provider)
    assert winner.item() == pytest.approx(_value(Fraction(1, 8), algebra))
    gradient = torch.autograd.grad(winner, parameters[_owner("b")])[0]
    assert gradient.item() == pytest.approx(1.0)
    with torch.no_grad():
        parameters[_owner("b")].fill_(_value(Fraction(1, 4), algebra))
    assert _run(module, provider).item() == pytest.approx(
        _value(Fraction(1, 4), algebra)
    )
    with torch.no_grad():
        parameters[_owner("b")].fill_(_value(Fraction(0), algebra))
    assert _run(module, provider).item() == _value(Fraction(0), algebra)
    assert request.with_(words=loser).plan(parameters).weight().item() == _value(
        Fraction(0), algebra
    )

    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        assert any(
            item.computation == "observe" and request.handler_name in item.reason
            for item in project_for_target(module, discharged, target).obstacles
        )


def test_ranked_owner_binding_refuses_same_typed_substitution_and_invalid_data():
    source, request, parameters = _fixture("markov")
    substitute = request.with_(caps=(2, 2))
    assert substitute.result_type == request.result_type
    assert substitute.plan(parameters).weight().item() == 0.0
    assert substitute.handler_name != request.handler_name
    with pytest.raises(ValueError, match="exact observation handler"):
        FixedRelationRuntimeProvider(
            substitute, parameters, handler_name=request.handler_name
        )
    with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
        wrong = FixedRelationRuntimeProvider(
            substitute, parameters, handler_name=substitute.handler_name
        )
        _run(_native(source, request), wrong)
    ordinary = FixedRelationRequest(
        relation=request.relation,
        operation="pair_weight",
        words=request.words,
        max_work=request.max_work,
        max_states=request.max_states,
    )
    with pytest.raises(ValueError, match="requires its ranked request"):
        FixedRelationRuntimeProvider(
            ordinary, parameters, handler_name=request.handler_name
        )
    with pytest.raises(dx.ValidationError, match="nonnegative integer caps"):
        request.with_(caps=(-1, 2))
    with pytest.raises(dx.ValidationError, match="one fixed-width"):
        request.with_(violations=request.violations[1:])
    with pytest.raises(dx.ValidationError, match="one input tape and one output tape"):
        request.with_(relation=request.relation.with_(input_tapes=0))


def test_compilation_and_fixed_observation_have_separate_exact_work_limits():
    _, request, parameters = _fixture("markov")
    unit = request.relation.initial[0][1]
    one = request.with_(
        relation=request.relation.with_(
            n_states=2,
            arcs=((0, 1, (0, 0), unit),),
            initial=((0, unit),),
            final=((1, unit),),
        ),
        words=((0,), (0,)),
        violations=((3, 4),),
        max_rank_states=2,
        max_rank_work=7,
        max_states=2,
        max_work=1,
    )
    # Two product states, one product arc, two epsilon-closure state
    # probes, and one state plus one arc in the symbol transition.
    assert one.plan(parameters).weight().item() == 1.0
    assert one.with_(max_rank_work=6).handler_name == one.handler_name
    with pytest.raises(RelationLimitError, match="ranked compilation exceeds max_work"):
        one.with_(max_rank_work=6).plan(parameters)
    with pytest.raises(
        RelationLimitError, match="ranked compilation exceeds max_states"
    ):
        one.with_(max_rank_states=1).plan(parameters)
    with pytest.raises(
        RelationLimitError, match="fixed-word path plan exceeds max_work"
    ):
        one.with_(max_work=0).plan(parameters)
