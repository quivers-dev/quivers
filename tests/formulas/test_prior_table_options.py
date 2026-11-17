"""Prior tables under brms data options, and equated mixture parameters.

brms values come from ``oracles/brms_prior_table_options.R``: the coefficient
rows of ``get_prior`` with ``drop_unused_levels`` and ``knots``, and the Stan
block in which brms declares a mixture parameter equated to another.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import (
    PriorSpec,
    bf,
    default_prior,
    fit,
    mixture,
    set_rescor,
    validate_prior,
)
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_prior_table_options.json").read_text()
)
KNOTS = {"x": [0.0, 0.2, 0.5, 0.8, 1.0]}


def _frame() -> pd.DataFrame:
    data = ORACLE["data"]
    return pd.DataFrame(
        {
            "x": data["x"],
            "y": data["y"],
            "f": pd.Categorical(data["f"], categories=["a", "b", "c"]),
            "g": pd.Categorical(data["g"], categories=["u", "v", "w"]),
        }
    )


def _coefficients(table) -> set[str]:
    return {row.coef for row in table.rows if row.class_ == "b" and row.coef}


def _brms_coefficients(name: str) -> set[str]:
    rows = ORACLE[name]
    return {
        coef
        for cls, coef in zip(rows["class"], rows["coef"], strict=True)
        if cls == "b" and coef
    }


def test_prior_tables_keep_or_drop_unused_levels_as_brms():
    frame = _frame()
    formula = "y ~ x + f + (1 | g)"
    dropped = _coefficients(default_prior(formula, data=frame))
    kept = _coefficients(default_prior(formula, data=frame, drop_unused_levels=False))
    # brms: b_fb and b_x, and b_fc for the level no row uses when it is kept.
    assert _brms_coefficients("dropped") == {"fb", "x"}
    assert _brms_coefficients("kept") == {"fb", "fc", "x"}
    assert len(dropped) == 2 and len(kept) == 3 and "x" in dropped & kept
    assert ORACLE["kept_levels"] == 3 and ORACLE["dropped_levels"] == 2
    unused = sorted(kept - dropped)[-1]
    rule = PriorSpec(distribution="Normal(0.0, 0.5)", class_="b", coef=unused)
    checked = validate_prior(rule, formula, data=frame, drop_unused_levels=False)
    (row,) = [row for row in checked.rows if row.coef == unused]
    assert row.source == "user" and row.prior == "Normal(loc=0.0, scale=0.5)"
    # The level has no coefficient when brms drops it, so the rule is unused.
    with pytest.raises(ValueError):
        validate_prior(rule, formula, data=frame)
    with pytest.raises(ValueError, match="drop_unused_levels"):
        default_prior(formula, data=frame, drop_unused_levels="no")


def test_prior_tables_take_knots_as_brms():
    frame = _frame()
    formula = "y ~ s(x, bs = 'cr', k = 5) + f"
    plain = default_prior(formula, data=frame)
    placed = default_prior(formula, data=frame, knots=KNOTS)
    # brms's table has the same rows with and without knots.
    assert ORACLE["smooth"] == ORACLE["smooth_knots"]
    assert [row.site for row in placed.rows] == [row.site for row in plain.rows]
    assert [row.class_ for row in placed.rows] == [row.class_ for row in plain.rows]
    rule = PriorSpec(distribution="HalfNormal(0.3)", class_="sds")
    checked = validate_prior(rule, formula, data=frame, knots=KNOTS)
    assert any(row.source == "user" for row in checked.rows)
    with pytest.raises(ValueError, match="increas"):
        default_prior(formula, data=frame, knots={"x": [0.0, 0.5, 0.2, 0.8, 1.0]})
    with pytest.raises(ValueError, match="knots"):
        default_prior("y ~ s(x, k = 5)", data=frame, knots=KNOTS)


def test_joint_prior_tables_take_both_options():
    frame = _frame().assign(z=lambda rows: rows.y + rows.x)
    joint = (
        bf("y ~ s(x, bs = 'cr', k = 5) + f")
        + bf("z ~ x + f + (1 | g)")
        + set_rescor(False)
    )
    plain = default_prior(joint, data=frame)
    kept = default_prior(joint, data=frame, knots=KNOTS, drop_unused_levels=False)

    def count(table, response: str) -> int:
        return sum(
            row.class_ == "b" and bool(row.coef) and row.response == response
            for row in table.rows
        )

    # Each response gains the coefficient of the unused level.
    assert count(kept, "y") == count(plain, "y") + 1
    assert count(kept, "z") == count(plain, "z") + 1
    unused = sorted(
        {row.coef for row in kept.rows if row.class_ == "b" and row.coef}
        - {row.coef for row in plain.rows if row.class_ == "b" and row.coef}
    )[-1]
    rule = PriorSpec(
        distribution="Normal(0.0, 0.5)", class_="b", coef=unused, response="z"
    )
    checked = validate_prior(
        rule, joint, data=frame, knots=KNOTS, drop_unused_levels=False
    )
    assert [row.source for row in checked.rows if row.coef == unused].count("user") == 1


def _mixture_frame() -> pd.DataFrame:
    rows = np.arange(30)
    x = (rows * 0.618034) % 1
    y = np.where(rows % 2 == 0, -2.0, 2.0) + 0.3 * x + 0.4 * np.sin(5.0 * rows)
    return pd.DataFrame({"x": x, "y": y})


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_equated_mixture_parameters_are_reported_under_their_own_name(method, tmp_path):
    # brms declares the equated parameter in its transformed parameters, so
    # it is a reported variable.
    assert ORACLE["equated_block"] == "transformed parameters"
    assert ORACLE["equated_assignment"] == "sigma2 = sigma1;"
    settings = (
        {"method": "svi", "num_steps": 5}
        if method == "svi"
        else {"method": method, "num_warmup": 3, "num_samples": 3, "num_chains": 1}
    )
    family = mixture("gaussian", "gaussian", equal={"sigma2": "sigma1"})
    fitted = fit("y ~ x", data=_mixture_frame(), family=family, seed=1, **settings)
    draws = fitted.draws(num_samples=3, seed=2)
    torch.testing.assert_close(draws["sigma2"], draws["sigma1"])
    assert {"sigma1", "sigma2"} <= set(fitted.variables(num_samples=2))
    if method != "svi":
        return
    refitted = fitted.update(num_steps=2)
    assert "sigma2" in refitted.draws(num_samples=1)
    path = tmp_path / "equated.pt"
    save_fit(fitted, path)
    restored = load_fit(path).draws(num_samples=3, seed=2)
    torch.testing.assert_close(restored["sigma2"], draws["sigma2"])


def test_equated_parameters_follow_their_source_kind_and_response():
    frame = _mixture_frame()
    family = mixture("gaussian", "gaussian", equal={"sigma2": "sigma1"})
    predicted = fit(
        "y ~ x",
        data=frame,
        family=family,
        distributional={"sigma1": "~ x"},
        method="svi",
        num_steps=2,
        seed=1,
    )
    # A parameter equated to a predicted one is a row quantity, not a draw.
    assert "sigma2" not in predicted.draws(num_samples=1)
    joint = fit(
        bf("y ~ x", family=family) + bf("x ~ 1") + set_rescor(False),
        data=frame,
        method="svi",
        num_steps=2,
        seed=1,
    )
    draws = joint.draws(num_samples=2)
    torch.testing.assert_close(draws["response_y__sigma2"], draws["response_y__sigma1"])
    free = fit(
        "y ~ x",
        data=frame,
        family=mixture("gaussian", "gaussian"),
        method="svi",
        num_steps=2,
        seed=1,
    ).draws(num_samples=2)
    # Without the equality the two scales are separate sampled sites.
    assert not torch.equal(free["sigma2"], free["sigma1"])
