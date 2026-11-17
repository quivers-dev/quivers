"""brms ``knots``: user knots of ``s()`` terms, pinned against mgcv and brms.

The oracle is the ``knot_entries`` section of ``oracles/mgcv_smooths.json``,
written by ``oracles/mgcv_smooths.R`` (brms 2.20.4, mgcv 1.8-42), which passes
the knots to ``smoothCon`` and checks the training matrices against brms's
``make_standata(knots = )``. Comparison and tolerance are those of
``test_mgcv_smooths.py``.

For ``bs = 'cs'`` the shrunken penalty assigns two different eigenvalues to
the two null-space eigenvectors of the cubic regression penalty, whose order
in mgcv follows the rounding noise of two numerically zero eigenvalues
(2.10e-13 and 2.03e-13 for these knots). That entry is compared by column
span, as ``test_mgcv_smooths.py`` compares its rounding-dependent entries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import test_mgcv_smooths as smooths

from typing import Literal, cast

from formulae.matrices import CommonEffectsMatrix, DesignMatrices

from quivers.formulas import BayesianFit, bf, fit
from quivers.formulas._joint_options import FormulaFitOptions
from quivers.formulas._multivariate import MultivariateFormula
from quivers.formulas._data_options import formula_data_options
from quivers.formulas._mgcv_smooth import cubic_regression
from quivers.formulas.formula import _formula_and_design
from quivers.inference import MCMCResult
from quivers.modeling import load_fit, save_fit

ENTRIES = smooths.ORACLE["knot_entries"]


def _common(design: DesignMatrices) -> CommonEffectsMatrix:
    """Return the population design, which every formula here has."""
    assert design.common is not None
    return design.common


ROUNDING_DEPENDENT = frozenset({"s(x, bs = 'cs', k = 5)"})
CR_KNOTS = {"x": [0.0, 0.2, 0.5, 0.8, 1.0]}


@pytest.mark.parametrize(
    "entry", ENTRIES, ids=lambda e: f"{e['spec']} {e['knots']['x'][:2]}"
)
def test_user_knots_equal_brms_designs(entry):
    with formula_data_options(entry["knots"]):
        if entry["spec"] not in ROUNDING_DEPENDENT:
            smooths.test_training_and_fresh_designs_equal_brms(entry)
            return
        train, fresh = smooths._frames()
        _, design, name, _ = smooths._smooth(entry["spec"], train)
    # Every column of the smooth, unpenalized and penalized, spans brms's.
    columns = _common(design).slices[name]
    for frame, prefix in ((train, ""), (fresh, "fresh_")):
        values = np.asarray(
            _common(design).evaluate_new_data(frame).design_matrix
            if prefix
            else _common(design).design_matrix
        )[:, columns]
        expected = np.column_stack(
            [
                smooths._matrix(block[prefix + "fixed"], len(frame))
                for block in entry["blocks"]
                if block[prefix + "fixed"]
            ]
            + [
                smooths._matrix(group, len(frame))
                for block in entry["blocks"]
                for group in block[prefix + "random"]
            ]
        )
        smooths._assert_same_span(values, expected)


def test_knots_change_the_basis_and_are_retained_for_prediction():
    train, fresh = smooths._frames()
    _, default = _formula_and_design("y ~ s(x, bs = 'cr', k = 5)", train)
    with formula_data_options(CR_KNOTS):
        _, knotted = _formula_and_design("y ~ s(x, bs = 'cr', k = 5)", train)
    assert not np.allclose(
        np.asarray(_common(default).design_matrix),
        np.asarray(_common(knotted).design_matrix),
    )
    # New rows are evaluated with the retained knots outside any scope.
    inside = np.asarray(_common(knotted).evaluate_new_data(fresh).design_matrix)
    with formula_data_options({"x": [0.0, 0.1, 0.4, 0.7, 1.0]}):
        again = np.asarray(_common(knotted).evaluate_new_data(fresh).design_matrix)
    np.testing.assert_array_equal(inside, again)


def test_invalid_knots_are_refused():
    x = np.linspace(0.0, 1.0, 30)[:, None]
    with pytest.raises(ValueError, match="takes k = 5 knots, 4 given"):
        cubic_regression(x, 5, shrink=None, knots=np.array([0.0, 0.3, 0.6, 1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        cubic_regression(x, 5, shrink=None, knots=np.array([0.0, 0.5, 0.3, 0.8, 1.0]))
    train, _ = smooths._frames()
    cases = {
        "y ~ s(x)": "needs at least 10 knots, got 2",
        "y ~ s(x, z)": "must be given for every covariate",
        "y ~ t2(x, z)": "takes k = 5 knots, 2 given",
        "y ~ s(x, bs = 'ps', k = 8)": "does not include the data",
        "y ~ s(x, bs = 'bs', k = 6)": "two end points, four knots, or 10 knots",
    }
    for formula, message in cases.items():
        knots = {"x": [0.2, 0.8, 0.9]} if "'bs'" in formula else {"x": [0.2, 0.8]}
        with formula_data_options(knots), pytest.raises(ValueError, match=message):
            _formula_and_design(formula, train)
    with pytest.raises(ValueError, match="finite and nonempty"):
        with formula_data_options({"x": []}):
            pass


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_fit_with_knots_predicts_refits_and_round_trips(
    method: Literal["svi", "hmc", "nuts"], tmp_path: Path
):
    train, fresh = smooths._frames()
    frame = pd.DataFrame({"x": train["x"], "y": train["y"]})
    budget: FormulaFitOptions = (
        {"num_steps": 20}
        if method == "svi"
        else {"num_warmup": 20, "num_samples": 5, "num_chains": 1}
    )
    options: FormulaFitOptions = {**budget, "method": method, "seed": 2}
    fitted = fit("y ~ s(x, bs = 'cr', k = 5)", data=frame, knots=CR_KNOTS, **options)
    unknotted = fit("y ~ s(x, bs = 'cr', k = 5)", data=frame, **options)
    new = fresh[["x"]]
    assert fitted.posterior_epred(new_data=new, num_samples=3).shape[-1] == len(new)
    if method != "nuts":
        return
    assert isinstance(fitted.posterior, MCMCResult)
    design = fitted.prediction_plan.designs["mu"]
    knotted = np.asarray(design.common.design_matrix)
    assert not np.allclose(
        knotted,
        np.asarray(_common(unknotted.prediction_plan.designs["mu"]).design_matrix),
    )
    refit = fitted.update(data=frame.iloc[:30])
    refit_design = np.asarray(
        _common(refit.prediction_plan.designs["mu"]).design_matrix
    )
    with formula_data_options(CR_KNOTS):
        _, expected = _formula_and_design("y ~ s(x, bs = 'cr', k = 5)", frame.iloc[:30])
    np.testing.assert_allclose(
        refit_design, np.asarray(_common(expected).design_matrix), atol=1e-12
    )
    path = tmp_path / "knots.qvrfit"
    save_fit(fitted, path)
    restored = load_fit(path)
    assert isinstance(restored, BayesianFit)
    torch.testing.assert_close(
        restored.posterior_epred(new_data=new), fitted.posterior_epred(new_data=new)
    )
    np.testing.assert_array_equal(
        np.asarray(_common(restored.prediction_plan.designs["mu"]).design_matrix),
        knotted,
    )


def test_joint_formulas_take_knots():
    train, _ = smooths._frames()
    frame = pd.DataFrame({"x": train.x, "y": train.y, "y2": train.z})
    fitted = fit(
        cast(MultivariateFormula, bf("y ~ s(x, bs = 'cr', k = 5)") + bf("y2 ~ x")),
        data=frame,
        knots=CR_KNOTS,
        method="svi",
        num_steps=5,
    )
    plain = fit(
        cast(MultivariateFormula, bf("y ~ s(x, bs = 'cr', k = 5)") + bf("y2 ~ x")),
        data=frame,
        method="svi",
        num_steps=5,
    )
    knotted = _common(fitted.prepared["y"].designs["mu"]).design_matrix
    default = _common(plain.prepared["y"].designs["mu"]).design_matrix
    assert not np.allclose(np.asarray(knotted), np.asarray(default))
