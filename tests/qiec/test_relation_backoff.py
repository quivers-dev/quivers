"""Compare the native compact request against an independently expanded DFA."""

from collections import Counter
from fractions import Fraction
from itertools import product
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
    FixedBackoffRelationRequest,
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

_MODULE = "compact_backoff_test"
_SOURCE = """object Symbols : FinSet 4

relation ordinary [algebra={algebra}, states=5]
    output Symbols
    initial 0 [weight=unit]
    arc 0 -> 4 [labels=(0,), weight=zero]
    arc 1 -> 4 [labels=(0,), weight=half]
    arc 1 -> 4 [labels=(1,), weight=shared]
    arc 2 -> 4 [labels=(2,), weight=tail]
    arc 4 -> 0 [labels=(1,), weight=unit]
    final 1 [weight=half]
    final 4 [weight=unit]
"""
_VALUES = {
    "unit": Fraction(1),
    "zero": Fraction(0),
    "half": Fraction(1, 2),
    "shared": Fraction(1, 2),
    "tail": Fraction(3, 4),
    "failure_only": Fraction(1, 4),
}


def _owner(name):
    scope = "failure_weights" if name == "failure_only" else "relation_weights"
    return ParameterId.derive(_MODULE, scope, name)


def _value(probability, algebra):
    if algebra == "logprob":
        return log(probability) if probability else float("-inf")
    return float(probability)


def _compile(source):
    compiler = Compiler(parse(module_to_source(parse(source))), module_name=_MODULE)
    compiler.compile()
    return compiler


def _fixture(algebra, word):
    source = _SOURCE.format(algebra=algebra)
    relation = _compile(source).relation_specs["ordinary"]
    request = FixedBackoffRelationRequest(
        relation=relation,
        failures=((0, 1, _owner("shared")), (1, 2, _owner("failure_only"))),
        word=word,
        max_work=32,
        max_states=32,
    )
    parameters = {
        _owner(name): nn.Parameter(
            torch.tensor(_value(value, algebra), dtype=torch.float64)
        )
        for name, value in _VALUES.items()
    }
    return source, request, parameters


def _expanded(*, ordinary_zero=True):
    """Materialize one transition cell for every state and symbol."""
    ordinary = {
        (0, 0): (4, ("zero",)),
        (1, 0): (4, ("half",)),
        (1, 1): (4, ("shared",)),
        (2, 2): (4, ("tail",)),
        (4, 1): (0, ("unit",)),
    }
    if not ordinary_zero:
        del ordinary[0, 0]
    failure = {0: (1, "shared"), 1: (2, "failure_only")}
    full = {}
    for state in (2, 1, 0, 4, 3):
        for symbol in range(4):
            transition = ordinary.get((state, symbol))
            if state in failure and transition is None:
                parent, owner = failure[state]
                inherited = full[parent, symbol]
                if inherited is not None:
                    target, weights = inherited
                    transition = target, (owner, *weights)
            full[state, symbol] = transition
    return full


def _oracle(word, values, table):
    state = 0
    names = ["unit"]
    for symbol in word:
        transition = table[state, symbol]
        if transition is None:
            return Fraction(0), Counter()
        state, owners = transition
        names.extend(owners)
    final = {1: "half", 4: "unit"}.get(state)
    if final is None:
        return Fraction(0), Counter()
    names.append(final)
    result = Fraction(1)
    for owner in names:
        result *= values[owner]
    return result, Counter(names)


def _native(source, request):
    name = request.handler_name
    result = request.result_type.constructor.name
    compiler = _compile(
        source
        + f"""
instance sparse : Compute[Unit, {result}]
handler {name} for Compute[Unit, {result}] : {result} -> {result} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define observe : {result} !{{}} =
    handle sparse with {name} in
        let value <- perform sparse.apply(unit)
        return value
"""
    )
    assert compiler.relation_specs["ordinary"] == request.relation
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
def test_compact_paths_match_every_small_word_and_keep_explicit_zero(algebra):
    source, request, parameters = _fixture(algebra, (0,))
    table = _expanded()
    for length in range(4):
        for word in product(range(4), repeat=length):
            probability, _ = _oracle(word, _VALUES, table)
            score = request.with_(word=word).plan(parameters).weight()
            assert score.item() == pytest.approx(_value(probability, algebra))
    wrong, _ = _oracle((0,), _VALUES, _expanded(ordinary_zero=False))
    assert wrong == Fraction(1, 4)
    provider = FixedRelationRuntimeProvider(
        request, parameters, handler_name=request.handler_name
    )
    module = _native(source, request)
    assert _run(module, provider).item() == _value(Fraction(0), algebra)
    with torch.no_grad():
        parameters[_owner("zero")].fill_(_value(Fraction(1, 8), algebra))
    assert _run(module, provider).item() == _value(Fraction(1, 8), algebra)


@pytest.mark.parametrize("algebra", ["markov", "logprob"])
def test_checked_native_backoff_shares_original_weights_and_differentiates(algebra):
    source, request, parameters = _fixture(algebra, (1, 1, 2))
    wire = dumps(request)
    assert isinstance(request := loads(wire), FixedBackoffRelationRequest)
    assert dumps(request) == wire
    assert _owner("failure_only") not in request.relation.weight_parameters
    assert request.failures[0][2] == _owner("shared")
    assert request.failures[0][2] in request.relation.weight_parameters
    module = _native(source, request)
    provider = FixedRelationRuntimeProvider(
        request, parameters, handler_name=request.handler_name
    )
    values = dict(_VALUES)
    table = _expanded()
    for shared in (Fraction(1, 2), Fraction(3, 8)):
        values["shared"] = shared
        with torch.no_grad():
            parameters[_owner("shared")].fill_(_value(shared, algebra))
        score = _run(module, provider)
        probability, visits = _oracle(request.word, values, table)
        assert visits["shared"] == 3
        assert score.item() == pytest.approx(_value(probability, algebra))
        names = ("shared", "failure_only", "tail")
        gradients = torch.autograd.grad(
            score, tuple(parameters[_owner(n)] for n in names)
        )
        assert [value.item() for value in gradients] == pytest.approx(
            [
                visits[n]
                if algebra == "logprob"
                else probability * visits[n] / values[n]
                for n in names
            ]
        )

    checked = request.with_(max_states=7, max_work=6)
    assert checked.handler_name == request.handler_name
    assert checked.plan(parameters).work.active_states == 7
    with pytest.raises(RelationLimitError, match="max_work"):
        checked.with_(max_work=5).plan(parameters)
    with pytest.raises(RelationLimitError, match="max_states"):
        checked.with_(max_states=6).plan(parameters)

    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        assert any(
            item.computation == "observe" and request.handler_name in item.reason
            for item in project_for_target(module, discharged, target).obstacles
        )


def test_foreign_same_type_substitution_and_malformed_failure_are_refused():
    source, request, parameters = _fixture("markov", (1, 1, 2))
    substituted = request.with_(
        failures=((0, 2, _owner("shared")), request.failures[1])
    )
    assert substituted.result_type == request.result_type
    assert substituted.handler_name != request.handler_name
    with pytest.raises(ValueError, match="exact observation handler"):
        FixedRelationRuntimeProvider(
            substituted, parameters, handler_name=request.handler_name
        )
    with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
        _run(
            _native(source, request),
            FixedRelationRuntimeProvider(
                substituted, parameters, handler_name=substituted.handler_name
            ),
        )
    ordinary = FixedRelationRequest(
        relation=request.relation,
        operation="pair_weight",
        words=(request.word,),
        max_work=request.max_work,
        max_states=request.max_states,
    )
    with pytest.raises(ValueError, match="requires its backoff request"):
        FixedRelationRuntimeProvider(
            ordinary, parameters, handler_name=request.handler_name
        )
    with pytest.raises(dx.ValidationError, match="failure arcs must be acyclic"):
        request.with_(failures=(*request.failures, (2, 0, _owner("shared"))))
    with pytest.raises(dx.ValidationError, match="at most one failure arc"):
        request.with_(failures=(*request.failures, (0, 2, _owner("shared"))))
    with pytest.raises(dx.ValidationError, match="epsilon-free deterministic"):
        request.with_(
            relation=request.relation.with_(
                arcs=(*request.relation.arcs, (0, 4, (None,), 0))
            )
        )
    with pytest.raises(ValueError, match="missing relation parameter"):
        request.plan(
            {
                key: value
                for key, value in parameters.items()
                if key != _owner("failure_only")
            }
        )
