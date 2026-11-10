"""An atom's name in a rule is the atom, on both evaluators.

A rule identifier spelled like a declared atom denotes that atom. In a
category slot that is a ground pattern both the agenda chart and the
checked QIEC search honor the same way; in a chart-position slot it is
a compile error located at the rule, naming the rule and the
identifier, on both compilation paths.
"""

from __future__ import annotations

import math

import pytest

from quivers.dsl import Compiler, loads, parse
from quivers.dsl.ast_nodes import DeductionDecl
from quivers.dsl.compiler import CompileError
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.qiec.program_runtime import run_deduction

SHADOWED_POSITION = """\
object Term : FinSet 4
deduction G : Term -> Term [semiring=LogProb, start=S, depth=2]
    atoms S, N, span, n
    rule up : span(M, N, N) |- span(M, N, S)
    lexicon
        "n" : N = n
"""

STRUCTURED_POSITION = """\
object Term : FinSet 4
deduction G : Term -> Term [semiring=LogProb, start=S, depth=2]
    atoms S, N, f, span, n
    rule up : span(f(I), J, N) |- span(I, J, S)
    lexicon
        "n" : N = n
"""

CATEGORY_ATOM = """\
object Term : FinSet 4
deduction G : Term -> Term [semiring=LogProb, start=S, depth=2]
    atoms S, N, M, span, n, m
    rule up : span(I, J, N) |- span(I, J, S)
    lexicon
        "n" : N = n
        "m" : M = m
"""


def _agenda_compile(source: str) -> None:
    """Compile a source's deduction on the agenda path alone.

    Parameters
    ----------
    source : str
        The QVR text, declaring one deduction.
    """
    module = parse(source)
    (decl,) = (item for item in module.statements if isinstance(item, DeductionDecl))
    Compiler(module)._compile_deduction(decl)


def test_an_atom_in_a_position_slot_is_a_located_error_on_the_qiec_path() -> None:
    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parse(SHADOWED_POSITION), file_path="shadow.qvr")
    error = captured.value
    assert error.code == "qiec-program"
    assert error.line == 4
    assert "rule 'up'" in error.message
    assert "'N'" in error.message
    assert "declared atom" in error.message
    assert "position" in error.message


def test_an_atom_in_a_position_slot_is_a_located_error_on_the_agenda_path() -> None:
    with pytest.raises(CompileError) as captured:
        _agenda_compile(SHADOWED_POSITION)
    error = captured.value
    assert error.line == 4
    assert "rule 'up'" in str(error)
    assert "'N'" in str(error)
    assert "declared atom" in str(error)


def test_loading_reports_the_rule_and_identifier() -> None:
    with pytest.raises((CompileError, QiecDiagnosticError)) as captured:
        loads(SHADOWED_POSITION)
    assert "rule 'up' uses 'N' at slot 1 of 'span'" in str(captured.value)
    assert "unknown indexed family type" not in str(captured.value)


@pytest.mark.parametrize("compile_path", ["agenda", "qiec"])
def test_a_structured_term_in_a_position_slot_is_rejected(compile_path: str) -> None:
    with pytest.raises((CompileError, QiecDiagnosticError)) as captured:
        if compile_path == "agenda":
            _agenda_compile(STRUCTURED_POSITION)
        else:
            lower_qvr_to_qiec(parse(STRUCTURED_POSITION), file_path="structured.qvr")
    assert "rule 'up' puts a 'f' term at slot 0 of 'span'" in str(captured.value)


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [(["n"], 0.0), (["m"], -math.inf), (["n", "n"], -math.inf)],
)
def test_an_atom_in_a_category_slot_is_the_atom_on_both_evaluators(
    sentence: list[str], expected: float
) -> None:
    program = loads(CATEGORY_ATOM)
    agenda = float(program.deductions["G"](sentence).goal_weight().detach())
    checked = run_deduction(program.qiec, "G", tokens=sentence).weight
    assert agenda == expected
    assert checked == expected
