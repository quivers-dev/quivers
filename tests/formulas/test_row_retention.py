"""One row decision for missing values, as brms's ``na_omit``.

brms evaluates every model variable in one model frame and removes each row
with a missing value in any of them before building anything
(``R/data-helpers.R``: ``validate_data`` with ``na_action = na_omit``). The
tests put a missing value in each kind of variable and check that every
consumer sees the same rows, that the original positions are retained, and
that a fit equals the fit on the manually filtered frame draw for draw.
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import fit, formula_from_data
from quivers.formulas._fit import BayesianFit
from quivers.formulas._row_retention import (
    complete_rows,
    formula_variables,
    joint_complete_rows,
)


def _frame(rows: int = 24) -> pd.DataFrame:
    generator = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "x": generator.normal(size=rows),
            "z": generator.normal(size=rows),
            "u": generator.normal(size=rows),
            "f": np.array(["p", "q", "r"])[np.arange(rows) % 3],
            "g": np.array(["a", "b", "c", "d"])[np.arange(rows) % 4],
            "w": generator.uniform(0.5, 1.5, rows),
            "t": generator.uniform(1.0, 2.0, rows),
            "n": np.full(rows, 6),
            "status": np.zeros(rows, dtype=int),
            "keep": np.ones(rows, dtype=bool),
        }
    )
    frame["y"] = 0.5 * frame["x"] + generator.normal(scale=0.5, size=rows)
    frame["k"] = generator.integers(0, 7, rows)
    return frame


def test_variable_scanner_reads_names_as_r_all_vars():
    required, optional = formula_variables(
        'y | weights(w) + trials(n) ~ x + log(z) + poly(u, 2) + (1 + x | g) + s(v, bs = "cr")'
    )
    assert required == ("y", "w", "n", "x", "z", "u", "g", "v")
    assert optional == ()
    required, optional = formula_variables("y | mi() ~ mi(x) + z")
    assert required == ("z",) and optional == ("x", "y")
    assert formula_variables("~ offset(log(t)) + `odd name`")[0] == ("t", "odd name")


def test_formula_from_data_codes_groups_on_the_retained_rows():
    frame = _frame()
    frame.loc[[2, 7], "x"] = np.nan
    with pytest.warns(UserWarning, match=r"missing values in \['x'\].*2 of 24"):
        formula = formula_from_data("y ~ x + (1 | g)", frame)
    kept = tuple(i for i in range(24) if i not in (2, 7))
    assert formula.response_values.shape == (22,)
    assert len(formula.group_indices["g"]) == 22
    assert formula.modifiers.subset is not None
    assert formula.modifiers.subset.size == 24
    assert formula.modifiers.subset.positions == kept
    expected = formula_from_data("y ~ x + (1 | g)", frame.iloc[list(kept)])
    assert formula.group_indices == expected.group_indices
    np.testing.assert_array_equal(formula.response_values, expected.response_values)


_CASES = [
    ("response", "y", "y ~ x + (1 | g)", {}),
    ("predictor", "x", "y ~ x + (1 | g)", {}),
    ("factor", "f", "y ~ x + f", {}),
    ("group", "g", "y ~ x + (1 | g)", {}),
    ("group slope", "z", "y ~ x + (1 + z | g)", {}),
    ("offset", "t", "y ~ x + offset(log(t))", {}),
    ("weights", "w", "y | weights(w) ~ x", {}),
    ("subset column", "keep", "y | subset(keep) ~ x", {}),
    ("distributional", "u", "y ~ x", {"distributional": {"sigma": "~ u"}}),
    (
        "nonlinear",
        "z",
        "y ~ a + b * z",
        {"nonlinear": {"a": "~ 1", "b": "~ 1"}},
    ),
]


@pytest.mark.parametrize(
    ("kind", "column", "formula", "options"), _CASES, ids=[case[0] for case in _CASES]
)
def test_missing_value_in_each_kind_of_variable_matches_the_filtered_fit(
    kind, column, formula, options
):
    frame = _frame()
    frame[column] = frame[column].astype(
        object if column in {"g", "f", "keep"} else float
    )
    frame.loc[[1, 9], column] = None
    kept = [i for i in range(24) if i not in (1, 9)]
    with pytest.warns(UserWarning, match="were excluded from the model"):
        fitted = fit(formula, data=frame, method="svi", num_steps=3, seed=4, **options)
    clean = frame.iloc[kept].reset_index(drop=True)
    if column == "keep":
        clean["keep"] = clean["keep"].astype(bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reference = fit(
            formula, data=clean, method="svi", num_steps=3, seed=4, **options
        )
    assert fitted.included_rows == tuple(kept)
    torch.testing.assert_close(
        fitted.posterior_linpred(num_samples=3, seed=2),
        reference.posterior_linpred(num_samples=3, seed=2),
        rtol=0,
        atol=0,
    )
    for name, value in reference.draws(num_samples=2, seed=1).items():
        torch.testing.assert_close(
            fitted.draws(num_samples=2, seed=1)[name], value, rtol=0, atol=0
        )


def test_trials_and_censoring_columns_are_model_variables():
    frame = _frame()
    frame["n"] = frame["n"].astype(float)
    frame.loc[4, "n"] = np.nan
    with pytest.warns(UserWarning, match=r"\['n'\]"):
        counts = fit(
            "k | trials(n) ~ x",
            data=frame,
            family="binomial",
            method="svi",
            num_steps=2,
        )
    assert 4 not in counts.included_rows and len(counts.included_rows) == 23
    frame["status"] = frame["status"].astype(float)
    frame.loc[5, "status"] = np.nan
    frame["yp"] = np.exp(frame["y"])
    with pytest.warns(UserWarning, match=r"\['status'\]"):
        censored = fit(
            "yp | cens(status) ~ x",
            data=frame,
            family="lognormal",
            method="svi",
            num_steps=2,
        )
    assert 5 not in censored.included_rows


def test_pointwise_likelihood_and_refit_partitions_use_original_rows():
    frame = _frame()
    frame.loc[[0, 13], "x"] = np.nan
    with pytest.warns(UserWarning):
        fitted = fit("y ~ x + (1 | g)", data=frame, method="svi", num_steps=2, seed=1)
    pointwise = fitted.log_likelihood(num_samples=2, seed=1)
    values = (
        pointwise
        if isinstance(pointwise, torch.Tensor)
        else next(iter(pointwise.values()))
    )
    assert values.shape[-1] == len(fitted.included_rows) == 22
    with pytest.warns(UserWarning):
        refit = fitted.update(data=frame)
    assert refit.included_rows == fitted.included_rows


def test_new_data_with_missing_required_values_is_refused():
    frame = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = fit("y ~ x + f + (1 | g)", data=frame, method="svi", num_steps=2)
    fresh = frame.iloc[:3].copy()
    for column, message in (
        ("x", "must produce finite numeric values"),
        ("f", "cannot contain missing values"),
        ("g", "missing group identities"),
    ):
        broken = fresh.astype({column: object})
        broken.loc[broken.index[0], column] = None
        with pytest.raises(ValueError, match=message):
            fitted.posterior_epred(new_data=broken, num_samples=1)
    # As brms, a missing group identity is a new level when they are allowed.
    broken = fresh.astype({"g": object})
    broken.loc[broken.index[0], "g"] = None
    draws = fitted.posterior_epred(
        new_data=broken, num_samples=1, seed=1, allow_new_levels=True
    )
    assert torch.isfinite(draws).all()


def test_all_rows_missing_is_refused():
    frame = _frame()
    frame["x"] = np.nan
    with pytest.raises(ValueError, match="All observations in the data were removed"):
        formula_from_data("y ~ x", frame)


def test_complete_rows_ignores_unused_columns_and_mi_variables():
    frame = _frame()
    frame.loc[3, "u"] = np.nan
    assert complete_rows(frame, ("y ~ x",)) is None
    assert complete_rows(frame, ("y ~ mi(u) + x",)) is None
    with pytest.warns(UserWarning):
        assert complete_rows(frame, ("y ~ u",)) == tuple(i for i in range(24) if i != 3)


def test_joint_decision_removes_rows_for_every_response_except_subset_exclusions():
    frame = _frame()
    frame["y2"] = frame["z"] + 1.0
    frame.loc[3, "x"] = np.nan
    frame.loc[5, "u"] = np.nan
    frame["only"] = np.arange(24) != 5
    with pytest.warns(UserWarning):
        rows = joint_complete_rows(
            frame, ((("y ~ x",), ()), (("y2 | subset(only) ~ u",), ()))
        )
    # Row 3 is missing x, used by the first response: removed for both.
    # Row 5 is missing u, used only by a response whose subset excludes it.
    assert rows == tuple(i for i in range(24) if i != 3)


def _tiny(method: Literal["svi", "hmc", "nuts"], frame: pd.DataFrame) -> BayesianFit:
    if method == "svi":
        return fit("y ~ x + (1 | g)", data=frame, method="svi", num_steps=2, seed=5)
    return fit(
        "y ~ x + (1 | g)",
        data=frame,
        method=method,
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        seed=5,
    )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_every_inference_method_fits_and_predicts_with_removed_rows(
    method: Literal["svi", "hmc", "nuts"],
):
    frame = _frame()
    frame.loc[[2, 6], "x"] = np.nan
    with pytest.warns(UserWarning):
        fitted = _tiny(method, frame)
    complete = frame.dropna(subset=["x"])
    draws = fitted.posterior_epred(new_data=complete.iloc[:4], num_samples=2, seed=1)
    assert draws.shape[-1] == 4 and torch.isfinite(draws).all()


def test_interval_upper_endpoint_counts_only_on_interval_rows():
    frame = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "status": [0, 2, 1, 0],
            "upper": [np.nan, 5.0, np.nan, np.nan],
        }
    )
    # The upper endpoint does not apply to exact or right-censored rows, so
    # a missing value there keeps the row, as before this rule existed.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        formula = formula_from_data("y | cens(status, upper) ~ x", frame)
    assert formula.response_values.shape == (4,)
    assert formula.modifiers.subset is None
    # On an interval-censored row the endpoint is required: a located error.
    with pytest.raises(ValueError, match="interval censoring upper endpoint"):
        formula_from_data(
            "y | cens(status, y2 = upper) ~ x", frame.assign(upper=[np.nan] * 4)
        )
    required, optional = formula_variables("y | cens(status, upper) ~ x")
    assert "upper" not in required and "upper" in optional
