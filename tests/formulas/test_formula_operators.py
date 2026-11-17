"""R formula operators and transforms against R's ``model.matrix``.

Oracle: ``tests/formulas/oracles/brms_model_matrix.json`` from
``brms_model_matrix.R`` (R's ``model.matrix`` and ``predict`` of the stateful
transforms; brms's ``make_standata`` builds its design with the same call).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quivers.formulas import fit, formula_from_data
from quivers.formulas._formula_options import expand_dot_formula, normalize_r_syntax
from quivers.formulas._prior_tables import validate_prior
from quivers.formulas._r_transforms import BSplineBasis
from quivers.formulas.formula import _formula_and_design

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_model_matrix.json").read_text()
)
FRAME = pd.DataFrame(ORACLE["data"])

# R spelling -> the same formula in the parser's argument syntax.
SPELLING = {
    "y ~ poly(x, 3, raw = TRUE)": "y ~ poly(x, 3, raw=True)",
    "y ~ ns(x, df = 4, intercept = TRUE)": "y ~ ns(x, df=4, intercept=True)",
    "y ~ ns(x, knots = c(-0.5, 0.6))": "y ~ ns(x, knots=c(-0.5, 0.6))",
    "y ~ bs(x, df = 5, degree = 2)": "y ~ bs(x, df=5, degree=2)",
    "y ~ bs(x, knots = c(-0.5, 0.6), intercept = TRUE)": (
        "y ~ bs(x, knots=c(-0.5, 0.6), intercept=True)"
    ),
}


def _design(formula):
    parsed = formula_from_data(formula, FRAME)
    return np.column_stack(
        [
            np.ones(len(FRAME)) if column.is_intercept else column.data
            for column in parsed.fixed_columns
        ]
    )


@pytest.mark.parametrize("case", ORACLE["matrices"], ids=lambda c: c["formula"])
def test_design_matches_r_model_matrix(case):
    formula = SPELLING.get(case["formula"], case["formula"])
    expected = np.asarray(case["values"], dtype=np.float64).reshape(
        len(FRAME), len(case["columns"])
    )
    np.testing.assert_allclose(_design(formula), expected, rtol=0, atol=1e-12)


def _new(formula, values):
    _, design = _formula_and_design(formula, FRAME)
    rows = pd.DataFrame({"x": values, "y": 0.0, "z": 1.0, "f": "a", "h": "u"})
    return np.asarray(design.common.evaluate_new_data(rows).design_matrix)[:, 1:]


def test_stateful_transforms_predict_new_rows_with_training_state():
    newx = np.asarray(ORACLE["newx"])
    np.testing.assert_allclose(
        _new("y ~ ns(x, df=3)", newx), np.asarray(ORACLE["ns_new"]), atol=1e-12
    )
    np.testing.assert_allclose(
        _new("y ~ ns(x, knots=c(-0.5, 0.6))", newx),
        np.asarray(ORACLE["ns_knots_new"]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _new("y ~ poly(x, 2)", newx), np.asarray(ORACLE["poly_new"]), atol=1e-12
    )
    inside = np.asarray(ORACLE["bs_inside"])
    np.testing.assert_allclose(
        _new("y ~ bs(x, df=4)", newx[inside]), np.asarray(ORACLE["bs_new"]), atol=1e-12
    )
    # predict.bs continues the boundary polynomials outside the training range.
    np.testing.assert_allclose(
        _new("y ~ bs(x, df=4)", newx), np.asarray(ORACLE["bs_all_new"]), atol=1e-12
    )
    scaled = (newx - ORACLE["scale_center"]) / ORACLE["scale_scale"]
    np.testing.assert_allclose(_new("y ~ scale(x)", newx)[:, 0], scaled, atol=1e-12)
    # brms make_standata builds X with the same model.matrix call.
    brms = np.asarray(ORACLE["brms_X"]).reshape(len(FRAME), -1)
    np.testing.assert_allclose(_design("y ~ ns(x, df=3) + scale(x)"), brms, atol=1e-12)


def test_bs_boundary_knots_and_tied_quantile_knots_match_r():
    bounded = BSplineBasis()
    np.testing.assert_allclose(
        bounded(FRAME["x"].to_numpy(), df=5, Boundary_knots=(-1.2, 1.5)),
        np.asarray(ORACLE["bs_bounded"]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        bounded(np.asarray(ORACLE["newx"])),
        np.asarray(ORACLE["bs_bounded_new"]),
        atol=1e-12,
    )
    tied = BSplineBasis()
    basis = tied(np.asarray(ORACLE["bs_tied_x"], dtype=np.float64), df=5)
    assert tied.knots is not None
    np.testing.assert_allclose(tied.knots[4:-4], ORACLE["bs_tied_knots"], atol=1e-15)
    np.testing.assert_allclose(basis, np.asarray(ORACLE["bs_tied"]), atol=1e-12)
    # A basis that keeps its first column sums to one inside the boundary knots.
    full = BSplineBasis()(FRAME["x"].to_numpy(), df=6, intercept=True)
    np.testing.assert_allclose(full.sum(axis=1), 1.0, atol=1e-14)


def test_bs_rejects_invalid_specifications():
    x = FRAME["x"].to_numpy()
    with pytest.raises(ValueError, match="too small"):
        BSplineBasis()(x, df=2)
    with pytest.raises(ValueError, match="integer >= 1"):
        BSplineBasis()(x, df=4, degree=0)
    with pytest.raises(ValueError, match="finite"):
        BSplineBasis()(x, knots=[0.0, np.inf])
    with pytest.raises(ValueError, match="left boundary"):
        BSplineBasis()(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), df=4)


def test_operator_rewrites_and_labels():
    assert normalize_r_syntax("y ~ I(x^2) + z") == "y ~ I(x**2) + z"
    assert normalize_r_syntax("y ~ (a + b + c)^2") == (
        "y ~ (a + b + c + a:b + a:c + b:c)"
    )
    assert normalize_r_syntax("y ~ x %in% f") == "y ~ x:f"
    assert normalize_r_syntax("y ~ x") == "y ~ x"
    labels = [
        c.name for c in formula_from_data("y ~ scale(x) + I(x^2)", FRAME).fixed_columns
    ]
    assert labels == ["Intercept", "scale(x)", "I(x ** 2)"]
    with pytest.raises(ValueError, match="too small"):
        formula_from_data("y ~ ns(x, df=0)", FRAME)


def test_dot_update_expands_against_the_fitted_formula():
    # The expected strings are R's update.formula output.
    assert expand_dot_formula("y ~ x + z", ". ~ . + w") == "y ~ x + z + w"
    assert expand_dot_formula("y ~ x + z", ". ~ . - z") == "y ~ x"
    assert expand_dot_formula("y ~ x", "log(.) ~ .") == "log(y) ~ x"
    assert expand_dot_formula("y ~ x", "y2 ~ z") == "y2 ~ z"
    fitted = fit("y ~ x + z", data=FRAME, method="svi", num_steps=3, seed=1)
    reduced = fitted.update(formula=". ~ . - z", num_steps=3)
    assert [c.name for c in reduced.formula.fixed_columns] == ["Intercept", "x"]
    grown = fitted.update(formula=". ~ . + f", num_steps=3)
    assert [c.name for c in grown.formula.fixed_columns] == [
        "Intercept",
        "x",
        "z",
        "f_1",
        "f_2",
    ]


def test_cmc_sparse_and_unused_follow_brms():
    no_cell_means = formula_from_data("y ~ 0 + f", FRAME)
    assert [c.name for c in no_cell_means.fixed_columns] == ["f_1", "f_2", "f_3"]
    fitted = fit("y ~ 0 + f", data=FRAME, cmc=False, method="svi", num_steps=3)
    design = np.column_stack([c.data for c in fitted.formula.fixed_columns])
    expected = np.asarray(ORACLE["cmc_false_X"]).reshape(len(FRAME), -1)
    assert len(ORACLE["cmc_false_columns"]) == design.shape[1]
    np.testing.assert_allclose(design, expected, atol=1e-12)
    assert fitted.posterior_epred(new_data=FRAME.iloc[:2], num_samples=2).shape == (
        1,
        2,
        2,
    )
    table = validate_prior([], "y ~ x", data=FRAME, sparse=True)
    brms = {(r["class"], r["coef"]) for r in ORACLE["sparse_prior"]}
    assert ("b", "Intercept") in brms
    assert table.row("intercept").class_ == "b"
    with pytest.raises(ValueError, match="sparse=True"):
        fit("y ~ x", data=FRAME, sparse=True, decomp="QR")
    with_missing = FRAME.assign(u=[1.0] * 11 + [np.nan])
    with pytest.warns(UserWarning, match="missing values in \\['u'\\]"):
        kept = fit("y ~ x", data=with_missing, unused=["u"], method="svi", num_steps=2)
    assert len(kept.formula.response_values) == ORACLE["unused_rows"]
    with pytest.raises(ValueError, match="unused names absent columns"):
        fit("y ~ x", data=FRAME, unused=["absent"])
