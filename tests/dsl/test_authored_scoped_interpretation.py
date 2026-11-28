"""Structural binders and their interpretations can be authored in QVR."""

from __future__ import annotations

import pytest

from quivers.dsl import loads, parse
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_tooling import call_sites
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec import QiecModule, dumps as dumps_qiec, loads as loads_qiec, run_named
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
