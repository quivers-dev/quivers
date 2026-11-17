"""Unresolved finite objects do not suppress independent tooling declarations."""

from __future__ import annotations

import pytest

from quivers.cli.repl_session import ReplSession
from quivers.dsl import Compiler, CompileError, parse
from quivers.dsl.qiec_tooling import analyze_module
from quivers.lsp.document import DocumentState


OPEN = """object Rows : FinSet _
object Alias : Rows
morphism pending : Alias -> Alias [role=parameter]
define dependent = pending >> pending
"""
CLOSED = """object A : FinSet 2
morphism f : A -> A [role=parameter]
define closed = f >> f
"""
PROGRAMS = """program regression(x : Tensor[Real]([Rows])) : Rows -> A
    sample b <- Normal(loc=0.0, scale=1.0)
    observe y : Rows <- Normal(loc=b, scale=1.0)
    return b
program concrete : A -> A
    observe count : A <- Poisson(rate=2.0)
    return count
"""


@pytest.mark.parametrize("reverse", [False, True])
def test_compiler_stages_dependency_closure_and_compiles_unrelated_values(reverse):
    source = (CLOSED + OPEN if reverse else OPEN + CLOSED) + PROGRAMS
    syntax = parse(source)
    compiler = Compiler(syntax)
    checked = compiler.qiec_module
    env = compiler.compile_env(defer_unresolved=True)
    assert env["f"] is compiler.morphisms["f"]
    assert env["f"].domain.cardinality == env["f"].codomain.cardinality == 2
    assert env["closed"] is compiler.morphisms["closed"]
    assert "concrete" in compiler.morphisms
    assert not {"pending", "dependent", "regression"} & compiler.morphisms.keys()
    assert compiler.qiec_module is checked
    deferred = compiler.deferred_statements
    assert len(deferred) == 5  # Rows, Alias, pending, dependent, regression.
    assert compiler.programs["regression"] == env["regression"]
    assert checked is not None
    entries = {entry.name: entry for entry in checked.entries}
    assert [binder.name for binder in entries["regression"].telescope] == ["Rows"]
    assert entries["concrete"].telescope == ()


def test_default_compilation_still_requires_real_extent_evidence():
    with pytest.raises(CompileError, match="Rows"):
        Compiler(parse(OPEN + CLOSED)).compile_env()


def test_closed_error_is_reported_instead_of_hidden_by_unresolved_objects():
    analysis = analyze_module(
        parse(OPEN + CLOSED + "define bad = missing_morphism\n"),
        module_name="staging",
    )
    assert analysis.compile_error is not None
    assert "missing_morphism" in str(analysis.compile_error)


@pytest.mark.parametrize("reverse", [False, True])
def test_repl_exposes_closed_types_and_keeps_symbolic_program_contracts(reverse):
    session = ReplSession()
    response = session.dispatch(
        (CLOSED + OPEN if reverse else OPEN + CLOSED) + PROGRAMS
    )
    assert response.ok, response.diagnostics
    inspected = session.type_of("f")
    assert inspected.ok and "A -> A" in inspected.body
    inspected = session.type_of("regression::x")
    assert inspected.ok and "Tensor[Real]([Rows])" in inspected.body
    run = session.run_computation("concrete --data count=[1,2]")
    assert run.ok, run.diagnostics


def test_lsp_preserves_closed_runtime_environment_and_full_checked_entries():
    document = DocumentState(uri="file:///tmp/mixed-staging.qvr")
    document.update(source=OPEN + CLOSED + PROGRAMS, version=1)
    assert not document.diagnostics
    assert document.compiler is not None
    assert document.env["f"] is document.compiler.morphisms["f"]
    assert document.find_decl("pending") is not None
    assert document.qiec_module is not None
    assert {entry.name for entry in document.qiec_module.entries} == {
        "regression",
        "concrete",
    }


def test_kernel_inspection_uses_same_staged_analysis():
    from quivers.kernel.quivers_kernel import QuiversKernel

    kernel = QuiversKernel()
    kernel.send_response = lambda *args, **kwargs: None
    kernel.iopub_socket = None
    try:
        result = kernel.do_execute(OPEN + CLOSED + PROGRAMS, silent=True)
        assert result["status"] == "ok", result
        inspected = kernel.do_inspect("f", 1)
        assert inspected["found"]
        assert "A -> A" in inspected["data"]["text/plain"]
        result = kernel.do_execute(":run concrete --data count=[1,2]", silent=True)
        assert result["status"] == "ok", result
    finally:
        kernel.shell = None


def test_runtime_value_names_do_not_capture_unrelated_object_extents():
    source = (
        CLOSED
        + """object Rows : FinSet _
program generic[G : FinSet](x : Tensor[Real]([G])) : G -> A
    return x
program scalar(Rows : Real) : A -> A
    let value = Rows + 1.0
    return value
"""
    )
    compiler = Compiler(parse(source))
    env = compiler.compile_env(defer_unresolved=True)
    assert [statement.names for statement in compiler.deferred_statements] == [
        ("Rows",)
    ]
    assert "scalar" in compiler.morphisms
    assert env["generic"] == compiler.programs["generic"]


def test_object_indexed_primitives_are_deferred_through_value_compositions():
    source = (
        OPEN
        + CLOSED
        + "define id_open = identity(Rows)\ndefine composed = id_open >> id_open\n"
    )
    compiler = Compiler(parse(source))
    env = compiler.compile_env(defer_unresolved=True)
    assert "closed" in compiler.morphisms
    assert "id_open" not in compiler.morphisms
    assert "composed" not in compiler.morphisms
    assert env["id_open"].name == "id_open"
