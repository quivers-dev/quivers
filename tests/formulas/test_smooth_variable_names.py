"""brms names of smooth-term variables in ``variables()`` and ``summary()``.

Oracle: ``variables()`` of brms 2.20.4 fits
(``oracles/brms_smooth_names.R``) with ``s()`` and ``t2()`` smooths, factor,
ordered-factor and numeric ``by`` variables, the ``fs``, ``gp`` and ``re``
bases, unpenalized smooths, two smooths of one covariate, smooths of a
distributional parameter and of a two-response model, and R's
``make.unique`` on the names brms repairs with it. Names are compared as
sets, exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from quivers.formulas import bf, fit, set_rescor
from quivers.formulas._priors import PriorTarget
from quivers.formulas._smooth import smooth_states
from quivers.formulas._variable_catalogue import (
    class_variables,
    make_unique,
    smooth_names,
)

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_smooth_names.json").read_text()
)
MODELS = {
    "basic": 'y ~ s(x) + s(z, k = 5, bs = "cr") + t2(x, z, k = 4)',
    "by": "y ~ s(x, by = g) + s(z, by = w, k = 5) + t2(x, z, by = g, k = 4) "
    "+ s(w, by = o, k = 4)",
    "bases": 'y ~ s(x, g, bs = "fs", k = 4, m = 1) + s(z, bs = "gp", k = 5) '
    '+ s(x, z, k = 6) + s(g, bs = "re")',
    "duplicate": 'y ~ s(x, k = 4) + s(x, bs = "cr", k = 5)',
    "unpenalized": "y ~ s(x, k = 4, fx = TRUE) + s(z, k = 4) "
    "+ s(w, by = g, k = 4, fx = TRUE)",
    "distributional": bf(
        "y ~ s(x, k = 4)", "sigma ~ s(z, k = 4) + s(x, by = g, k = 4)"
    ),
    "joint": bf("y ~ s(x, k = 4) + s(z, by = g, k = 4)")
    + bf("w ~ s(x, k = 4) + s(z, by = g, k = 4)")
    + set_rescor(False),
}


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame(ORACLE["data"])
    frame["o"] = pd.Categorical(
        frame["o"], categories=["lo", "mid", "hi"], ordered=True
    )
    return frame


@pytest.fixture(scope="module", params=sorted(MODELS))
def fitted(request):
    torch.manual_seed(0)
    return request.param, fit(
        MODELS[request.param], data=_frame(), method="svi", num_steps=2, seed=0
    )


def test_variables_have_brms_names(fitted) -> None:
    key, model = fitted
    names = list(model.variables(num_samples=2))
    assert len(names) == len(set(names))
    # Names and their order are brms's variables().
    assert names == ORACLE["models"][key]


def test_summary_lists_every_smooth_variable(fitted) -> None:
    key, model = fitted
    names = ORACLE["models"][key]
    if key == "joint":
        # A joint summary has one section set per response; its rows are
        # checked through variables() above.
        return
    summary = model.summary(num_samples=4)
    # brms prints bs_<label>_<k> among the population effects without its
    # class prefix, and sds_<label>_<k> as sds(<label>_<k>).
    assert summary.fixed is not None
    assert {name[3:] for name in names if name.startswith("bs_")} <= set(
        summary.fixed.rows
    )
    assert summary.splines is not None
    assert list(summary.splines.rows) == [
        f"sds({name[4:]})"
        for name in model.variables(num_samples=4)
        if name.startswith("sds_")
    ]
    assert set(summary.splines.rows) == {
        f"sds({name[4:]})" for name in names if name.startswith("sds_")
    }
    assert torch.isfinite(summary.splines.values).all()


def test_make_unique_is_r_make_unique() -> None:
    pinned = ORACLE["make_unique"]
    assert make_unique(pinned["input"]) == pinned["output"]
    assert make_unique(["a", "b"]) == ["a", "b"]


def test_wrong_naming_rules_differ_from_brms() -> None:
    # Naming a by-level block without its level, or counting penalties over
    # the term, gives the names the oracle does not contain.
    names = set(ORACLE["models"]["by"])
    assert "sds_sxg_1" not in names and "sds_sxga_1" in names
    assert "sds_t2xzg_4" not in names and "sds_t2xzgb_1" in names
    assert "bs_sxga_1" not in names and "bs_sx:ga_1" in names


def test_inconsistent_smooth_metadata_is_refused() -> None:
    model = fit("y ~ s(x, k = 4)", data=_frame(), method="svi", num_steps=1, seed=0)
    states = smooth_states(model.prediction_plan.designs["mu"])
    assert list(states) == ["s(x, k=4)"]
    names = smooth_names(model.formula, states, infix="")
    assert list(names.population.values()) == ["bs_sx_1"]
    assert list(names.scales.values()) == ["sds_sx_1"]
    stripped = model.formula.with_(design_terms=())
    with pytest.raises(ValueError, match="penalized blocks"):
        smooth_names(stripped, states, infix="")
    target = PriorTarget(site="smooth_unknown_scale", class_="sds", coef="s(x)")
    with pytest.raises(ValueError, match="belongs to no retained smooth block"):
        class_variables(target, torch.zeros(1, 2), infix="", scales=names.scales)
