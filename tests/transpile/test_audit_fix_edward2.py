"""Regression tests for audit-confirmed edward2 (TFP) transpile bugs.

Each test transpiles a model exercising one family to the ``edward2``
backend and asserts the emitted distribution call is now correct: the
right TFP class name with the right argument names, order, and values.
Assertions are on emitted source text (fast and deterministic); TFP is
not imported.

The bugs fixed here are the edward2 manifestations of the audit's
family-level parameterization / distribution-name defects
(``notes/transpile-audit-findings.md``):

* ``pareto-argorder-swap``: torch ``alpha`` is not a valid TFP
  ``Pareto`` keyword (TFP names it ``concentration``).
* ``lkj-missing-dimension-arg``: TFP ``LKJ`` requires the
  correlation-matrix dimension as its first positional argument.
* ``halfstudentt-support``: the native TFP ``HalfStudentT`` retains
  nonnegative support and the folded density normalization.
* ``matrixnormal-nonexistent``: TFP's ``MatrixNormalLinearOperator``
  takes Cholesky-factor ``LinearOperator``s under ``scale_row`` /
  ``scale_column``, not covariance matrices under
  ``row_covariance`` / ``col_covariance``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from quivers.dsl.parser import parse
from quivers.transpile import transpile

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _emit(qvr_source: str) -> str:
    return transpile(parse(qvr_source), target="edward2").decode("utf-8")


def _emit_file(rel_path: str) -> str:
    return _emit((_REPO_ROOT / rel_path).read_text())


def test_pareto_shape_keyword_is_concentration() -> None:
    """Pareto's shape parameter rides as ``concentration``, TFP's
    keyword, not torch's ``alpha`` (which TFP rejects).

    The fixture declares ``Pareto(1.0, 2.0)``, which binds positionally
    the way ``torch.distributions.Pareto`` does: ``scale=1``,
    shape ``=2``. TFP names that shape ``concentration``.
    """
    out = _emit_file("tests/transpile/fixtures/families/pareto.qvr")
    ast.parse(out)
    assert "edward2.Pareto(" in out
    assert "scale=1" in out
    assert "concentration=2" in out
    # torch's `alpha` keyword must not survive: TFP's Pareto has no
    # `alpha` argument.
    assert "alpha=" not in out


def test_lkj_cholesky_prepends_dimension_and_flags_cholesky() -> None:
    """LKJCholesky emits ``LKJ(<dim>, <concentration>,
    input_output_cholesky=True, ...)``; the dimension comes from the
    correlation-matrix event axis (size 4)."""
    out = _emit_file("docs/examples/source/lkj_cholesky_correlation.qvr")
    ast.parse(out)
    assert "edward2.LKJ(4,eta,input_output_cholesky=True" in out.replace(" ", "")
    # The bare single-arg form (concentration bound to dimension) is
    # gone.
    assert "edward2.LKJ(eta" not in out.replace(" ", "")


def test_halfstudentt_retains_support_and_folded_normalization() -> None:
    """Use TFP's normalized half-distribution with an explicit zero origin."""
    out = _emit_file("docs/examples/source/half_student_t_hierarchical.qvr")
    ast.parse(out)
    compact = out.replace(" ", "")
    assert "edward2.HalfStudentT(loc=0,df=3,scale=1" in compact
    assert "edward2.StudentT(" not in out


def test_matrix_normal_uses_cholesky_scale_linear_operators() -> None:
    """MatrixNormal emits Cholesky-factor ``LinearOperator``s under
    ``scale_row`` / ``scale_column``, not covariance matrices under the
    torch keywords."""
    out = _emit_file("tests/transpile/fixtures/families/matrixnormal.qvr")
    ast.parse(out)
    compact = out.replace(" ", "")
    assert "edward2.MatrixNormalLinearOperator(" in compact
    assert (
        "scale_row=tf.linalg.LinearOperatorLowerTriangular("
        "tf.linalg.cholesky(m_row_covariance))" in compact
    )
    assert (
        "scale_column=tf.linalg.LinearOperatorLowerTriangular("
        "tf.linalg.cholesky(m_col_covariance))" in compact
    )
    assert "loc=m_loc" in compact
    # The invalid TFP call keywords must be gone (the `=None` function
    # parameters keep the `m_row_covariance` / `m_col_covariance`
    # names, so match the keyword-argument binding form specifically).
    assert "row_covariance=m_row_covariance" not in compact
    assert "col_covariance=m_col_covariance" not in compact
