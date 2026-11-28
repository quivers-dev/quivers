"""Authored transform recipes and checked score owner identity."""

from fractions import Fraction

import pytest
import torch
from torch import nn

from quivers.dsl import Compiler, module_to_source, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_tooling import qiec_bindings
from quivers.qiec import (
    REAL,
    BinaryRelationTransformSpec,
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedTransformedRelationRequest,
    FixedRelationRuntimeProvider,
    HandlerId,
    ParameterId,
    QiecModule,
    RuntimeConfiguration,
    UnaryRelationTransformSpec,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target

_MODULE = "authored_sparse_transforms"
_SOURCE = """object Letters : FinSet 2
object Middle : FinSet 2
object Output : FinSet 2

#! Project the original input tape.
relation projected = project(base, (0))
relation fixed = condition(base, 0, (0))
relation empty_fixed = condition(base, 0, ())
relation inverted = invert(base)
relation trimmed = trim(base)
relation repeated = closure(base)
relation added = union(base, base)
relation serial = concatenate(base, base)
relation overlap = intersect(single, single)
relation joined = join(base, link, 1, 0, keep_shared=true)
relation erased = join(base, link, 1, 0, keep_shared=false)

relation_score score_projected : Real [operation=pair_weight, relation=projected]
relation_score score_fixed : Real [operation=pair_weight, relation=fixed]
relation_score score_empty : Real [operation=pair_weight, relation=empty_fixed]
relation_score score_inverted : Real [operation=pair_weight, relation=inverted]
relation_score score_trimmed : Real [operation=pair_weight, relation=trimmed]
relation_score score_repeated : Real [operation=pair_weight, relation=repeated]
relation_score score_added : Real [operation=pair_weight, relation=added]
relation_score score_serial : Real [operation=pair_weight, relation=serial]
relation_score score_overlap : Real [operation=pair_weight, relation=overlap]
relation_score score_joined : Real [operation=pair_weight, relation=joined]
relation_score score_erased : Real [operation=pair_weight, relation=erased]
relation_score original : Real [operation=pair_weight, relation=base]
relation_score unbound : Real [operation=pair_weight]

define use_joined() : Real !{} =
    score_joined()

relation base [algebra=markov, states=2]
    input Letters
    output Middle
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0, 0), weight=shared]
    arc 0 -> 1 [labels=(0, 0), weight=shared]
    arc 0 -> 1 [labels=(0, 1), weight=other]
    final 1 [weight=one]

relation link [algebra=markov, states=2]
    input Middle
    output Output
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0, 0), weight=shared]
    final 1 [weight=one]

relation single [algebra=markov, states=2]
    input Letters
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0), weight=shared]
    final 1 [weight=one]
"""


def _compile(source: str = _SOURCE) -> tuple[Compiler, QiecModule]:
    compiler = Compiler(parse(source), module_name=_MODULE, file_path="transforms.qvr")
    compiler.compile()
    module = compiler.qiec_module
    assert module is not None
    return compiler, module


def _parameters() -> dict[ParameterId, torch.Tensor]:
    return {
        ParameterId.derive(_MODULE, "relation_weights", name): nn.Parameter(
            torch.tensor(weight, dtype=torch.float64)
        )
        for name, weight in (("one", 1.0), ("shared", 0.125), ("other", 0.25))
    }


def _score(
    module: QiecModule,
    name: str,
    request: FixedTransformedRelationRequest,
    parameters: dict[ParameterId, torch.Tensor],
    *,
    entry: str | None = None,
) -> torch.Tensor:
    provider = FixedRelationRuntimeProvider.for_score(name, request, parameters)
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    result = run_named(module, name if entry is None else entry, runtime=runtime)
    assert (
        isinstance(result.value, torch.Tensor)
        and result.result_type == request.result_type
    )
    return result.value


def test_all_authored_transform_recipes_round_trip_and_score_through_qvr():
    compiler, module = _compile()
    specs = compiler.transformed_relation_specs
    assert tuple(compiler.relation_specs) == ("base", "link", "single")
    assert not compiler.composed_relation_specs
    assert isinstance(specs["projected"], UnaryRelationTransformSpec)
    assert isinstance(specs["joined"], BinaryRelationTransformSpec)
    assert specs["projected"].source == compiler.relation_specs["base"]
    assert specs["joined"].right == compiler.relation_specs["link"]
    assert specs["joined"].keep_shared is True
    assert specs["erased"].keep_shared is False
    assert specs["empty_fixed"].word == ()
    emitted = module_to_source(parse(_SOURCE))
    assert "#! Project the original input tape." in emitted
    emitted_compiler, emitted_module = _compile(emitted)
    assert emitted_compiler.transformed_relation_specs == specs
    assert tuple(item.id for item in emitted_module.handlers) == tuple(
        item.id for item in module.handlers
    )
    assert any(
        binding.name == "joined"
        and binding.kind == "relation"
        and binding.semantic_kind == "type"
        for binding in qiec_bindings(parse(_SOURCE))
    )
    wire = loads(dumps(module))
    assert isinstance(wire, QiecModule)
    weights = _parameters()
    fixtures = (
        ("projected", "projected", ((0,),), Fraction(1, 2)),
        ("fixed", "fixed", ((0,),), Fraction(1, 4)),
        ("empty_fixed", "empty", ((0,),), Fraction(0)),
        ("inverted", "inverted", ((1,), (0,)), Fraction(1, 4)),
        ("trimmed", "trimmed", ((0,), (0,)), Fraction(1, 4)),
        ("repeated", "repeated", ((0, 0), (0, 0)), Fraction(1, 16)),
        ("added", "added", ((0,), (0,)), Fraction(1, 2)),
        ("serial", "serial", ((0, 0), (0, 0)), Fraction(1, 16)),
        ("overlap", "overlap", ((0,),), Fraction(1, 64)),
        ("joined", "joined", ((0,), (0,), (0,)), Fraction(1, 32)),
        ("erased", "erased", ((0,), (0,)), Fraction(1, 32)),
    )
    for name, score, words, expected in fixtures:
        request = FixedTransformedRelationRequest(
            relation=specs[name], words=words, max_states=80, max_work=180
        )
        restored = loads(dumps(request))
        assert isinstance(restored, FixedTransformedRelationRequest)
        actual = _score(
            wire,
            f"score_{score}",
            restored,
            weights,
            entry="use_joined" if name == "joined" else None,
        )
        assert Fraction(actual.item()) == expected
        if name == "joined":
            (derivative,) = torch.autograd.grad(
                actual,
                weights[ParameterId.derive(_MODULE, "relation_weights", "shared")],
            )
            assert derivative.item() == 0.5


def test_bound_transform_and_primitive_score_owners_remain_distinct():
    compiler, module = _compile()
    spec = compiler.transformed_relation_specs["trimmed"]
    assert isinstance(spec, UnaryRelationTransformSpec)
    request = FixedTransformedRelationRequest(
        relation=spec, words=((0,), (0,)), max_states=80, max_work=180
    )
    weights = _parameters()
    changed = spec.with_(source=spec.source.with_(arcs=spec.source.arcs[1:]))
    derived = request.with_(relation=changed)
    assert derived.result_type == request.result_type == REAL
    with pytest.raises(ExecutionFailure, match="authored topology"):
        _score(module, "score_trimmed", derived, weights)
    with pytest.raises(ExecutionFailure, match="authored topology"):
        _score(module, "original", request, weights)

    unbound = next(
        item for item in module.handlers if item.name == "__relation_score_unbound"
    )
    assert unbound.id == HandlerId.derive(
        _MODULE, "relation_scores", "unbound", "pair_weight", "handler"
    )
    assert _score(module, "unbound", derived, weights).item() == 0.125
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        obstacle = next(
            item
            for item in project_for_target(module, discharged, target).obstacles
            if item.computation == "score_joined"
        )
        assert obstacle.reason == (
            "installs the foreign handler `__relation_score_score_joined`, whose "
            "clause `apply` a runtime provider implements"
        )


@pytest.mark.parametrize(
    "old,new,diagnostic",
    (
        ("project(base, (0))", "project(missing, (0))", "unknown authored relation"),
        (
            "project(base, (0))",
            "project(projected, (0))",
            "transformation requires primitive",
        ),
        (
            "project(base, (0))",
            "project(base, (1, 0))",
            "increasing distinct tape positions",
        ),
        ("condition(base, 0, (0))", "condition(base, 0, (3))", "invalid symbol"),
        ("join(base, link, 1, 0", "join(base, link, 0, 0", "same nominal shared tape"),
        ("union(base, base)", "union(base, link)", "identical ordered tape types"),
        (
            "score_inverted : Real",
            "score_inverted : LogWeight",
            "pair weight requires Real",
        ),
        (
            "operation=pair_weight, relation=projected",
            "operation=path_entropy, relation=projected",
            "authored transformations currently support pair_weight",
        ),
    ),
)
def test_authored_transform_refusals_are_located_at_the_source(old, new, diagnostic):
    with pytest.raises(QiecDiagnosticError, match=diagnostic):
        _compile(_SOURCE.replace(old, new, 1))
