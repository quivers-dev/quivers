"""Structural binders and their interpretations can be authored in QVR."""

from __future__ import annotations

import math

import pytest

from quivers.dsl import loads, parse
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_tooling import call_sites
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec import (
    ExecutionResult,
    QiecModule,
    RuntimeConfiguration,
    RuntimeSelection,
    SCORE_ADD,
    dumps as dumps_qiec,
    loads as loads_qiec,
    run_named,
)
from quivers.transpile import UnsupportedConstruct, transpile


_EMPTY = "Binders__ScopedEmpty"
_FRAME = "Binders__ScopedFrame_Expr"
_PROOF = "Binders__ScopedProof_Expr"
_MEMBER = "Binders__ScopedMember_Expr"
_ONE = f"{_FRAME}[{_EMPTY}, Real]"
_TWO = f"{_FRAME}[{_ONE}, Real]"
_PREFIX = "Binders__Scoped"

SOURCE = f"""\
signature Binders
    sorts
        Expr : object
    constructors
        Pair : Expr, Expr -> Expr
    binders
        Outer : binds (outer : Expr) in (body : Expr) -> Expr
        Inner : binds (inner : Expr) in (body : Expr) -> Expr

interpret meaning [structure=cartesian] : Binders -> Real
    rule Pair = pair
    rule Outer = outer
    rule Inner = inner

define pair[G : Type](first : Real, second : Real) : Real !{{}} =
    return 10.0 * first + second

define outer[G : Type](body : {_PROOF}[Real, {_FRAME}[G, Real]] * G) : Real !{{}} =
    let left <- meaning__apply_Outer_0[G](body, 2.0)
    let right <- meaning__apply_Outer_0[G](body, 5.0)
    return left + 2.0 * right

define inner[G : Type](body : {_PROOF}[Real, {_FRAME}[G, Real]] * G) : Real !{{}} =
    let left <- meaning__apply_Inner_0[G](body, 3.0)
    let right <- meaning__apply_Inner_0[G](body, 7.0)
    return left + 2.0 * right

define evaluate() : Real !{{}} =
    let first = construct {_PREFIX}Here_Expr[Real, {_EMPTY}]() as {_MEMBER}[Real, {_ONE}]
    let older = construct {_PREFIX}There_Expr_Expr[Real, {_ONE}](first) as {_MEMBER}[Real, {_TWO}]
    let recent = construct {_PREFIX}Here_Expr[Real, {_ONE}]() as {_MEMBER}[Real, {_TWO}]
    let x = construct {_PREFIX}Hypothesis_Expr[Real, {_TWO}](older) as {_PROOF}[Real, {_TWO}]
    let y = construct {_PREFIX}Hypothesis_Expr[Real, {_TWO}](recent) as {_PROOF}[Real, {_TWO}]
    let body = construct {_PREFIX}Rule_Pair[Real, {_TWO}](x, y) as {_PROOF}[Real, {_TWO}]
    let inner = construct {_PREFIX}Rule_Inner[Real, {_ONE}](body) as {_PROOF}[Real, {_ONE}]
    let proof = construct {_PREFIX}Rule_Outer[Real, {_EMPTY}](inner) as {_PROOF}[Real, {_EMPTY}]
    meaning(proof)
"""


def test_authored_scope_runs_through_the_checked_entry_and_serialization() -> None:
    compiled = loads(SOURCE)
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    expected = sum(
        outer_multiplicity * inner_multiplicity * (10.0 * outer + inner)
        for outer, outer_multiplicity in ((2.0, 1.0), (5.0, 2.0))
        for inner, inner_multiplicity in ((3.0, 1.0), (7.0, 2.0))
    )
    assert float(compiled.run("evaluate").value) == expected
    restored = loads_qiec(dumps_qiec(module))
    assert isinstance(restored, QiecModule)
    assert float(run_named(restored, "evaluate").value) == expected
    emitted = module_to_source(parse(SOURCE))
    assert "[structure=cartesian]" in emitted
    assert float(loads(emitted).run("evaluate").value) == expected
    calls = call_sites(parse(SOURCE))
    assert [
        (site.owner, site.callee, site.static_arguments, site.argument_count)
        for site in calls
        if "__apply_" in site.callee
    ] == [
        ("outer", "meaning__apply_Outer_0", ("G",), 2),
        ("outer", "meaning__apply_Outer_0", ("G",), 2),
        ("inner", "meaning__apply_Inner_0", ("G",), 2),
        ("inner", "meaning__apply_Inner_0", ("G",), 2),
    ]

    shadowed = loads(SOURCE.replace("(older) as", "(recent) as"))
    assert float(shadowed.run("evaluate").value) != expected


@pytest.mark.parametrize(
    "form",
    ["", " [structure=ordered]", " [structure=cartesian, structure=cartesian]"],
)
def test_source_scope_requires_explicit_cartesian_permission(form: str) -> None:
    program = SOURCE.replace(" [structure=cartesian]", form)
    with pytest.raises(
        QiecDiagnosticError, match="requires explicit.*structure=cartesian"
    ) as caught:
        loads(program)
    assert caught.value.code == "qiec-interpretation"
    assert caught.value.line == 10


def test_source_scope_rejects_incorrect_operation_argument() -> None:
    program = SOURCE.replace(
        f"body : {_PROOF}[Real, {_FRAME}[G, Real]] * G",
        f"body : {_PROOF}[Real, {_FRAME}[G, Real]] * {_EMPTY}",
        1,
    )
    with pytest.raises(QiecDiagnosticError, match="typed arguments") as caught:
        loads(program)
    assert caught.value.code == "qiec-interpretation"
    assert caught.value.line == 12


def test_source_scope_rejects_a_witness_at_another_context() -> None:
    wrong = SOURCE.replace(
        f"Here_Expr[Real, {_ONE}]() as {_MEMBER}[Real, {_TWO}]",
        f"Here_Expr[Real, {_EMPTY}]() as {_MEMBER}[Real, {_TWO}]",
        1,
    )
    with pytest.raises(QiecDiagnosticError, match="constructor"):
        loads(wrong)


def test_scoped_source_uses_existing_backend_capability_diagnostics() -> None:
    with pytest.raises(
        UnsupportedConstruct,
        match=r"`meaning__fold_Expr` requires the `case` QIEC capability",
    ):
        transpile(parse(SOURCE), target="stan")


_SCORE_SOURCE = (
    """
instance score : Score
handler collect for Score : Real -> (Real * LogWeight) [coverage=total, implementation=foreign]
    add resumes 1

"""
    + SOURCE.replace("[structure=cartesian]", "[structure=cartesian, effects=[score]]")
    .replace(" !{}", " !{score}")
    .replace(
        "    return 10.0 * first + second",
        "    perform score.add(log_prob(Normal(0.0, 1.0), first))\n"
        "    return 10.0 * first + second",
    )
    + """
define collected() : Real * LogWeight !{} =
    handle score with collect in
        evaluate()
"""
)
_RUNTIME = RuntimeConfiguration(
    selections=(
        RuntimeSelection(
            provider="core",
            options={"handlers": {"collect": {"kind": "score"}}},
        ),
    ),
)


def _score_addresses(result: ExecutionResult) -> tuple[tuple, ...]:
    return tuple(
        event.detail["address"]
        for event in result.trace
        if event.event == "operation.requested"
        and event.detail["operation"] == str(SCORE_ADD)
    )


def test_scoped_source_emits_score_once_per_executed_application() -> None:
    compiled = loads(_SCORE_SOURCE)
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    restored = loads_qiec(dumps_qiec(module))
    assert isinstance(restored, QiecModule)
    result = run_named(restored, "collected", runtime=_RUNTIME)
    value, mass = result.value
    expected_value = sum(
        outer_multiplicity * inner_multiplicity * (10.0 * bound + local)
        for bound, outer_multiplicity in ((2.0, 1.0), (5.0, 2.0))
        for local, inner_multiplicity in ((3.0, 1.0), (7.0, 2.0))
    )
    expected_mass = sum(
        -0.5 * (bound**2 + math.log(2.0 * math.pi)) for bound in (2.0, 2.0, 5.0, 5.0)
    )
    assert float(value) == expected_value
    assert float(mass) == pytest.approx(expected_mass)
    addresses = _score_addresses(result)
    assert len(addresses) == len(set(addresses)) == 4
    assert len({address[0] for address in addresses}) == 1
    assert addresses == _score_addresses(
        run_named(module, "collected", runtime=_RUNTIME)
    )

    original = """\
    let left <- meaning__apply_Outer_0[G](body, 2.0)
    let right <- meaning__apply_Outer_0[G](body, 5.0)
    return left + 2.0 * right
"""
    start = _SCORE_SOURCE.index("define outer[")
    end = _SCORE_SOURCE.index("define inner[", start)
    ignored = _SCORE_SOURCE[start:end].replace(original, "    return 43.0\n")
    ignored = ignored.replace(" !{score}", " !{}")
    ignored_source = _SCORE_SOURCE[:start] + ignored + _SCORE_SOURCE[end:]
    ignored_module = loads(ignored_source).qiec
    assert isinstance(ignored_module, QiecModule)
    ignored_result = run_named(ignored_module, "collected", runtime=_RUNTIME)
    assert tuple(map(float, ignored_result.value)) == (43.0, 0.0)
    assert _score_addresses(ignored_result) == ()


@pytest.mark.parametrize(
    ("option", "message"),
    (
        ("structure=cartesian", "outside the interpretation's closed effects row"),
        (
            "structure=cartesian, effects=[absent]",
            "unknown scoped effect instance 'absent'",
        ),
        (
            "structure=cartesian, effects=[score, score]",
            "duplicate scoped effect instance 'score'",
        ),
        ("structure=cartesian, effects=score", "scoped effects require one effects="),
    ),
)
def test_scoped_source_refuses_missing_or_invalid_effect_requirements(
    option: str, message: str
) -> None:
    replaced = _SCORE_SOURCE.replace("structure=cartesian, effects=[score]", option)
    with pytest.raises(QiecDiagnosticError, match=message) as caught:
        loads(replaced)
    assert caught.value.code == "qiec-interpretation"


def test_scoped_row_distinguishes_instances_of_one_interface() -> None:
    source = _SCORE_SOURCE.replace(
        "instance score : Score", "instance score : Score\ninstance other : Score"
    ).replace("effects=[score]", "effects=[other]")
    with pytest.raises(
        QiecDiagnosticError, match="outside the interpretation's closed effects row"
    ):
        loads(source)


def test_nonempty_authored_row_requires_one_object_sort() -> None:
    source = """\
instance score : Score
signature Mixed
    sorts
        First : object
        Second : object
    constructors
        FirstLeaf : -> First
        SecondLeaf : -> Second
interpret both [structure=cartesian, effects=[score]] : Mixed -> Real
    rule FirstLeaf = value
    rule SecondLeaf = value
define value[G : Type]() : Real !{} =
    return 0.0
"""
    with pytest.raises(
        QiecDiagnosticError, match="effect row requires a single object sort"
    ):
        loads(source)
