"""R's ``factor()`` in a formula, and located errors for unknown names.

The ``factor()`` design is checked against a treatment-coded indicator
matrix built here with NumPy, and against the fit of the same values stored
as a string column. Coefficient names follow R's ``model.matrix`` labels
after brms's renaming (``factor(k)2`` becomes ``factork2``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import bf, default_prior, fit, formula_to_qvr, get_prior
from quivers.formulas.formula import formula_from_data


def _frame(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = np.arange(n)
    frame = pd.DataFrame(
        {
            "x": np.sin(rows * 0.7),
            "z": np.cos(rows * 1.3),
            "se": 0.2 + 0.05 * (rows % 4),
            "k": (rows * 3 + 1) % 4,
            "g": np.array(["a", "b", "c"])[rows % 3],
        }
    )
    frame["y"] = 1.0 + 0.4 * frame.k + 0.3 * rng.standard_normal(n)
    frame["ks"] = frame.k.astype(str)
    return frame


def _fixed_design(formula: str, frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    columns = formula_from_data(formula, frame).fixed_columns
    return (
        [column.name for column in columns],
        np.column_stack([np.asarray(column.data, dtype=float) for column in columns]),
    )


@pytest.mark.parametrize("call", ["factor(k)", "as.factor(k)"])
def test_factor_codes_a_numeric_column_by_treatment_contrasts(call: str) -> None:
    frame = _frame()
    _, design = _fixed_design(f"y ~ {call}", frame)
    levels = np.sort(frame.k.unique())
    expected = np.column_stack(
        [np.ones(len(frame))]
        + [(frame.k.to_numpy() == level).astype(float) for level in levels[1:]]
    )
    np.testing.assert_array_equal(design, expected)
    # A numeric column without factor() is one slope, not three indicators.
    assert _fixed_design("y ~ k", frame)[1].shape[1] == 2


def test_factor_levels_fix_the_reference_level_and_the_column_order() -> None:
    frame = _frame()
    _, design = _fixed_design("y ~ factor(k, levels = c(3, 1, 0, 2))", frame)
    expected = np.column_stack(
        [np.ones(len(frame))]
        + [(frame.k.to_numpy() == level).astype(float) for level in (1, 0, 2)]
    )
    np.testing.assert_array_equal(design, expected)


def test_factor_matches_the_fit_of_the_same_values_as_strings() -> None:
    frame = _frame()
    options = {"method": "svi", "num_steps": 3, "seed": 2}
    by_call = fit("y ~ factor(k) + x", data=frame, **options)
    by_column = fit("y ~ ks + x", data=frame, **options)
    names = [name for name in by_call.variables(num_samples=2) if name[:2] == "b_"]
    assert names == ["b_Intercept", "b_factork1", "b_factork2", "b_factork3", "b_x"]
    new = frame.iloc[[5, 0, 3]].reset_index(drop=True)
    for data in (None, new):
        torch.testing.assert_close(
            by_call.posterior_epred(new_data=data, num_samples=4),
            by_column.posterior_epred(new_data=data, num_samples=4),
        )
    assert list(get_prior("y ~ factor(k)", data=frame))[:4] == [
        "intercept",
        "beta_factor_k_1",
        "beta_factor_k_2",
        "beta_factor_k_3",
    ]


def test_factor_interacts_and_refuses_what_it_cannot_code() -> None:
    frame = _frame()
    fitted = fit("y ~ factor(k) * x", data=frame, method="svi", num_steps=2, seed=1)
    names = [name for name in fitted.variables(num_samples=2) if name[:2] == "b_"]
    assert names[-3:] == ["b_factork1:x", "b_factork2:x", "b_factork3:x"]
    with pytest.raises(ValueError, match=r"values \[9\] outside its levels"):
        fitted.posterior_epred(new_data=frame.iloc[:2].assign(k=9), num_samples=2)
    with pytest.raises(ValueError, match="ordered pandas Categorical"):
        formula_from_data("y ~ factor(k, ordered = TRUE)", frame)
    with pytest.raises(ValueError, match="repeat a level"):
        formula_from_data("y ~ factor(k, levels = c(0, 1, 1, 2, 3))", frame)
    with pytest.raises(ValueError, match=r"values \[3\] outside its levels"):
        formula_from_data("y ~ factor(k, levels = c(0, 1, 2))", frame)
    with pytest.raises(ValueError, match="missing value"):
        formula_from_data(
            "y ~ factor(v)", pd.DataFrame({"y": [0.0, 1.0], "v": [np.nan, np.nan]})
        )


UNKNOWN = [
    ("y ~ nope", "names the variable 'nope', which is not a column"),
    ("nope ~ x", "names the variable 'nope', which is not a column"),
    ("y ~ x + (1 | nope)", "names the variable 'nope', which is not a column"),
    ("y ~ log(nope)", "names the variable 'nope', which is not a column"),
    ("y ~ s(nope)", "names the variable 'nope', which is not a column"),
    ("y ~ frobnicate(x)", "calls the unknown function 'frobnicate'"),
]


@pytest.mark.parametrize(("formula", "message"), UNKNOWN)
def test_an_unknown_name_is_reported_with_the_formula_and_the_columns(
    formula: str, message: str
) -> None:
    frame = _frame()
    for call in (
        lambda: fit(formula, data=frame, method="svi", num_steps=1),
        lambda: get_prior(formula, data=frame),
        lambda: formula_to_qvr(formula, data=frame),
        lambda: default_prior(formula, data=frame),
    ):
        with pytest.raises(ValueError, match=message) as caught:
            call()
        text = str(caught.value)
        assert repr(formula) in text
        assert "data columns are ['g', 'k', 'ks', 'se', 'x', 'y', 'z']" in text
        assert isinstance(caught.value.__cause__, KeyError)


def test_a_distributional_formula_reports_its_unknown_variable() -> None:
    with pytest.raises(ValueError, match="names the variable 'nope'"):
        fit(bf("y ~ x", "sigma ~ nope"), data=_frame(), method="svi", num_steps=1)


def test_bare_mvbind_is_refused_by_every_entry_point() -> None:
    frame = _frame()
    formula = "mvbind(y, z) ~ x"
    for name, call in (
        ("fit", lambda: fit(formula, data=frame, method="svi", num_steps=1)),
        ("get_prior", lambda: get_prior(formula, data=frame)),
        ("formula_to_qvr", lambda: formula_to_qvr(formula, data=frame)),
        ("default_prior", lambda: default_prior(formula, data=frame)),
    ):
        with pytest.raises(ValueError, match="compile through the joint program") as e:
            call()
        assert str(e.value).startswith(f"{name}: ")
        assert f"pass bf({formula!r})" in str(e.value)


def test_the_bf_spelling_of_a_measurement_error_term_reads_its_priors() -> None:
    priors = get_prior(bf("y ~ me(x, se) + z"), data=_frame())
    assert "me_y__x__mean" in priors and "response_y__beta_x" in priors
