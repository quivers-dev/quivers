"""Stan emission that keeps QVR's shapes, realness, and index origin.

Stan is statically typed and counts from one, so a program it accepts can
still read the wrong entry, truncate a real quotient, or score an observation
against a whole vector. Each test pins one contract of the emission on a
construct fixture (the same sources the probe tier scores numerically), and
every emitted program must pass `stanc`:

* a marginalized latent is enumerated over its declared index object;
* a `let` read as a family's vector argument is declared a `vector`, one
  gathered row per observation when it is a table;
* real literals keep their decimal point and a conversion to real stays
  visible, so `/` is real division;
* the lets of a marginalize body are inlined whatever chain they form, at
  the observe's row and at the latent's QVR value;
* every subscript is translated to Stan's origin at the subscript itself.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLiteral,
    LetExprVar,
)
from quivers.dsl.ast_nodes.let_expressions import LetFactorBinder
from quivers.dsl.ast_nodes.objects import TypeName
from quivers.dsl.parser import parse
from quivers.transpile import transpile
from quivers.transpile.renderers._one_based import counter_value, one_based_subscript
from quivers.transpile.renderers._rows import take_row

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _stan(fixture: str) -> str:
    """The Stan program a construct fixture transpiles to.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    str
        The program text.
    """
    source = (_FIXTURES / f"{fixture}.qvr").read_text()
    return transpile(parse(source), target="stan").decode()


def _int(value: int) -> LetExprLiteral:
    """An integer literal.

    Parameters
    ----------
    value : int
        The value.

    Returns
    -------
    LetExprLiteral
        The literal.
    """
    return LetExprLiteral(value=float(value), integral=True)


_FIXTURE_NAMES: tuple[str, ...] = (
    "let_expressions/let_factor_index_arithmetic",
    "let_expressions/let_host_matrix_row",
    "let_expressions/let_plated_list_operand",
    "let_expressions/let_real_division",
    "let_expressions/let_row_gather",
    "let_expressions/let_row_gather_sampled",
    "steps/marginalize_body_let_chain",
    "steps/marginalize_body_plated_let",
    "steps/marginalize_latent_value",
    "steps/marginalize_let_probs",
    "steps/marginalize_let_vector_probs",
)


@pytest.mark.requires_tool("stanc")
@pytest.mark.parametrize("fixture", _FIXTURE_NAMES)
def test_stanc_accepts_the_program(fixture: str) -> None:
    binary = shutil.which("stanc")
    assert binary is not None
    done = subprocess.run(
        [binary, "--info", "-"],
        input=_stan(fixture).encode(),
        capture_output=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr.decode()
    assert b"int division" not in done.stderr


def test_a_let_probability_argument_is_enumerated_over_the_index_object() -> None:
    program = _stan("steps/marginalize_let_probs")
    assert "vector [4] lps_z" in program
    assert "for (k in 1 : 4)" in program


def test_a_let_read_as_a_vector_argument_is_declared_a_vector() -> None:
    program = _stan("steps/marginalize_let_vector_probs")
    assert "vector [4] prior = to_vector(" in program
    assert "array [4] real prior" not in program


def test_a_gathered_table_row_is_one_vector_per_observation() -> None:
    program = _stan("let_expressions/let_row_gather")
    assert "array [4] vector [2] row_table" in program
    assert "row_table[m_Row] = to_vector(table[group_idx[m_Row]]);" in program
    assert "categorical_lpmf(y[m_Row] | row_table[m_Row])" in program
    assert "array [4] int <lower = 1 , upper = 2> y;" in program


def test_a_returned_outcome_keeps_its_class_count_in_generated_quantities() -> None:
    source = (_FIXTURES / "let_expressions/let_row_gather.qvr").read_text()
    source = source.replace("    return a\n", "    return y\n")
    program = transpile(parse(source), target="stan").decode()
    assert "array [4] int <lower = 1 , upper = 2> y;" in program
    assert "array [4] int <lower = 1 , upper = 2> y_value = y;" in program


def test_real_literals_keep_real_division() -> None:
    assert "real quarter = 1.0/4.0;" in _stan("let_expressions/let_real_division")


def test_a_chain_of_body_lets_is_inlined_whole() -> None:
    program = _stan("steps/marginalize_body_let_chain")
    assert "poisson_lpmf(y[n_Step] | (2.0*r)+1.0)" in program
    assert "doubled" not in program


def test_a_plated_body_let_is_read_at_the_observes_row() -> None:
    program = _stan("steps/marginalize_body_plated_let")
    assert "poisson_lpmf(y[n_Step] | r*(1.0*((n_Step-1)+1)))" in program


def test_a_latent_read_as_a_number_keeps_its_zero_based_value() -> None:
    program = _stan("steps/marginalize_latent_value")
    assert "poisson_lpmf(y[n_Step] | r*(1.0*((k-1)+1)))" in program


def test_index_arithmetic_on_a_binder_is_shifted_to_one_based() -> None:
    program = _stan("let_expressions/let_factor_index_arithmetic")
    assert "y[3],y[2],y[1]" in program


def test_a_two_axis_data_input_is_read_one_row_at_a_time() -> None:
    program = _stan("let_expressions/let_host_matrix_row")
    assert "dot_product(to_vector(X[m_Resp]),to_vector(w_row[m_Resp]))" in program


def test_the_subscript_translation_folds_into_literals_and_counters() -> None:
    assert one_based_subscript(_int(0)) == _int(1)
    assert one_based_subscript(counter_value("k")) == LetExprVar(name="k")
    assert one_based_subscript(
        LetExprBinOp(op="-", left=_int(2), right=_int(1))
    ) == _int(2)
    assert one_based_subscript(
        LetExprBinOp(op="-", left=_int(2), right=LetExprVar(name="t"))
    ) == LetExprBinOp(op="-", left=_int(3), right=LetExprVar(name="t"))
    assert one_based_subscript(LetExprVar(name="j")) == LetExprBinOp(
        op="+", left=LetExprVar(name="j"), right=_int(1)
    )


def test_a_row_is_pushed_to_the_operands_that_carry_it() -> None:
    ranks = {"x": 1, "w": 2, "a": 0}
    row = LetExprVar(name="n")
    # `a + x`: only `x` carries the leading axis.
    assert take_row(
        LetExprBinOp(op="+", left=LetExprVar(name="a"), right=LetExprVar(name="x")),
        row,
        ranks,
        "stan",
    ) == LetExprBinOp(
        op="+",
        left=LetExprVar(name="a"),
        right=LetExprIndex(array=LetExprVar(name="x"), indices=(row,)),
    )
    # `sum(w)` reduces the last axis, so the row is the argument's.
    assert take_row(
        LetExprCall(func="sum", args=(LetExprVar(name="w"),)), row, ranks, "stan"
    ) == LetExprCall(
        func="sum",
        args=(LetExprIndex(array=LetExprVar(name="w"), indices=(row,)),),
    )
    # A factor's row is its body at the binder's value.
    factor = LetExprFactor(
        binders=(LetFactorBinder(var="t", index=TypeName(name="Step")),),
        body=LetExprBinOp(
            op="*", left=LetExprVar(name="a"), right=LetExprVar(name="t")
        ),
    )
    assert take_row(factor, row, ranks, "stan") == LetExprBinOp(
        op="*", left=LetExprVar(name="a"), right=row
    )


def test_a_list_literal_operand_is_read_at_the_loops_row() -> None:
    program = _stan("let_expressions/let_plated_list_operand")
    assert "total[m_Step] = (ys[m_Step]*mu)+{" in program
    assert "}[m_Step];" in program
