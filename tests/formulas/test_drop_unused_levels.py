"""brms ``drop_unused_levels``: declared factor levels that no row uses.

The oracle is ``oracles/brms_drop_unused_levels.json``, written by
``oracles/brms_drop_unused_levels.R`` from brms 2.20.4 ``make_standata`` under
``drop_unused_levels = TRUE`` and ``FALSE``: the population design ``X`` by
column name, and the size and row indexes of the grouping factor. Design
entries are exact products of data values, so they are compared exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import BayesianFit, fit
from quivers.formulas._joint_options import FormulaFitOptions
from quivers.formulas._data_options import formula_data_options
from quivers.formulas._postprocess_names import _r_label, brms_rename
from quivers.formulas.formula import Formula, _formula_and_design
from quivers.inference import MCMCResult
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_drop_unused_levels.json").read_text()
)
FORMULA = "y ~ x * f + h + (1 | g)"


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame(ORACLE["data"])
    for name, levels in ORACLE["levels"].items():
        frame[name] = pd.Categorical(frame[name], categories=levels)
    return frame


def _named_columns(drop: bool) -> tuple[dict[str, np.ndarray], Formula]:
    with formula_data_options(drop_unused_levels=drop):
        formula, design = _formula_and_design(FORMULA, _frame())
    common = design.common
    assert common is not None
    matrix = np.asarray(common.design_matrix)
    columns = {}
    for name, term in common.terms.items():
        block = matrix[:, common.slices[name]]
        for index, label in enumerate(term.labels):
            columns[brms_rename(_r_label(label))] = block[:, index]
    return columns, formula


@pytest.mark.parametrize("drop", [True, False])
def test_design_and_groups_equal_brms_standata(drop):
    expected = ORACLE["drop" if drop else "keep"]
    columns, formula = _named_columns(drop)
    assert set(columns) == set(expected["X_names"])
    rows = np.asarray(expected["X"], dtype=np.float64)
    for index, name in enumerate(expected["X_names"]):
        np.testing.assert_array_equal(columns[name], rows[:, index])
    levels = formula.group_levels["g"]
    assert len(levels) == expected["N_1"]
    codes = [levels[code] for code in formula.group_indices["g"]]
    brms_levels = ORACLE["levels"]["g"]
    assert codes == [brms_levels[code - 1] for code in expected["J_1"]]


def test_unused_group_slope_levels_are_refused_and_flags_checked():
    frame = _frame()
    with formula_data_options(drop_unused_levels=False):
        with pytest.raises(ValueError, match="group-level term"):
            _formula_and_design("y ~ x + (1 + f | g)", frame)
    with pytest.raises(ValueError, match="True or False"):
        with formula_data_options(drop_unused_levels=0):  # type: ignore[arg-type]
            pass


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_kept_levels_fit_predict_refit_and_round_trip(
    method: Literal["svi", "hmc", "nuts"], tmp_path: Path
):
    frame = _frame()
    budget: FormulaFitOptions = (
        {"num_steps": 20}
        if method == "svi"
        else {"num_warmup": 20, "num_samples": 5, "num_chains": 1}
    )
    options: FormulaFitOptions = {**budget, "method": method, "seed": 4}
    if method == "nuts":
        # The checks below concern variable names, shapes and the artifact
        # round trip, not posterior accuracy, so each transition is capped at
        # seven leapfrog steps.
        options["control"] = {"max_treedepth": 3}
    fitted = fit(FORMULA, data=frame, drop_unused_levels=False, **options)
    names = set(fitted.variables(num_samples=2))
    assert {"b_fc", "b_hc", "b_x:fc", "r_g[w,Intercept]"} <= names
    new = pd.DataFrame(
        {
            "x": [0.2, -0.4],
            "f": pd.Categorical(["c", "a"], categories=ORACLE["levels"]["f"]),
            "h": pd.Categorical(["c", "e"], categories=ORACLE["levels"]["h"]),
            "g": pd.Categorical(["w", "u"], categories=ORACLE["levels"]["g"]),
        }
    )
    assert fitted.posterior_epred(new_data=new, num_samples=3).shape[-1] == 2
    dropped = fit(FORMULA, data=frame, **options)
    assert "b_fc" not in set(dropped.variables(num_samples=2))
    if method != "nuts":
        return
    assert isinstance(fitted.posterior, MCMCResult)
    refit = fitted.update(data=frame.iloc[:10])
    assert "b_fc" in set(refit.variables(num_samples=2))
    path = tmp_path / "levels.qvrfit"
    save_fit(fitted, path)
    restored = load_fit(path)
    assert isinstance(restored, BayesianFit)
    torch.testing.assert_close(
        restored.posterior_epred(new_data=new), fitted.posterior_epred(new_data=new)
    )
