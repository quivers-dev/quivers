"""Terms name their draws as brms does, in single and multivariate models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import BayesianFit, MultivariateFit, bf, fit, set_rescor
from quivers.formulas._design_variables import design_variables
from quivers.formulas._smooth_effects import conditional_smooths

ROWS = 24
FRAME = pd.DataFrame(
    {
        "y": np.random.default_rng(1).normal(size=ROWS).round(3),
        "o": [0, 1, 2, 3] * 6,
        "c": ["A", "B", "C"] * 8,
        "x": np.random.default_rng(3).normal(size=ROWS).round(3),
        "x2": np.random.default_rng(4).normal(size=ROWS).round(3),
        "z": [0, 1, 2, 3, 2, 1] * 4,
        "g": ["a", "b", "c"] * 8,
    }
)
OPTIONS = {"data": FRAME, "method": "svi", "num_steps": 1, "seed": 0}


def _single(formula: str, family: str) -> BayesianFit:
    fitted = fit(formula, family=family, **OPTIONS)
    assert isinstance(fitted, BayesianFit)
    return fitted


def _joint(formula: str, family: str) -> MultivariateFit:
    fitted = fit(
        bf(formula, family=family) + bf("y ~ 1") + set_rescor(False), **OPTIONS
    )
    assert isinstance(fitted, MultivariateFit)
    return fitted


def test_category_smooths_are_named_by_category():
    """Each coordinate has the names of a smooth of its own parameter."""
    fitted = _single("c ~ s(x, k = 5)", "categorical")
    variables = fitted.variables(num_samples=2, seed=1)
    draws = fitted.draws(num_samples=2, seed=1)
    block = "smooth_s_x_k_5"
    raw = draws[f"{block}_z"].reshape(1, 2, -1, 2)
    for index, parameter in enumerate(("muB", "muC")):
        scale = draws[f"{block}_scale"][..., index]
        torch.testing.assert_close(variables[f"sds_{parameter}_sx_1"], scale)
        torch.testing.assert_close(
            variables[f"bs_{parameter}_sx_1"], draws["beta_s_x_k_5_1"][..., index]
        )
        for basis in range(raw.shape[2]):
            torch.testing.assert_close(
                variables[f"s_{parameter}_sx_1[{basis + 1}]"],
                scale * raw[:, :, basis, index],
            )


def test_smooth_effects_have_one_column_per_category():
    """The effect is the basis at the rows times each category's coefficients."""
    fitted = _single("c ~ x2 + s(x, k = 5)", "categorical")
    effect = conditional_smooths(fitted, "s(x, k=5)", FRAME, num_samples=2, seed=1)
    draws = {
        key: value.double()
        for key, value in fitted.draws(num_samples=2, seed=1).items()
    }
    data = {key: value.double() for key, value in fitted.observations.items()}
    block = "smooth_s_x_k_5"
    coefficients = draws[f"{block}_scale"].unsqueeze(2) * draws[f"{block}_z"].unflatten(
        -1, (-1, 2)
    )
    expected = torch.einsum("nj,cdjk->cdnk", data[block], coefficients) + (
        data["s_x_k_5_1"][:, None] * draws["beta_s_x_k_5_1"].unsqueeze(2)
    )
    torch.testing.assert_close(effect, expected, rtol=1e-5, atol=1e-6)
    names = fitted.conditional_smooths(resolution=4).names()
    assert names == ("muB: s(x,k=5)", "muC: s(x,k=5)")


@pytest.mark.parametrize(
    "formula, family",
    [
        ("c ~ x + (1 | g)", "categorical"),
        ("c ~ s(x, k = 4)", "categorical"),
        ("o ~ x + cs(x2)", "acat"),
        ("o ~ x + mo(z) + (1 + mo(z) | g)", "acat"),
        ("o ~ gp(x, k = 4, c = 1.25)", "acat"),
    ],
)
def test_joint_names_are_the_single_names_with_the_response(formula, family):
    """A response's variables differ from its own model's by the response alone."""
    response = formula.split(" ", 1)[0]
    single = set(_single(formula, family).variables(num_samples=1))
    joint = set(_joint(formula, family).variables(num_samples=1))

    def untagged(name: str) -> str:
        head, bracket, rest = name.partition("[")
        tagged = head.replace(f"_{response}_", "_").replace(f"__{response}", "")
        return tagged.removesuffix(f"_{response}") + bracket + rest

    other = {name for name in joint if f"_{response}" not in name}
    restored = {untagged(name) for name in joint - other}
    assert {
        name for name in single if not name.startswith(("sd_", "cor_", "disc"))
    } <= (restored | other)


def test_level_tables_have_one_column_per_category(monkeypatch: pytest.MonkeyPatch):
    """Group tables list each category's coefficient under its parameter."""
    fitted = _joint("c ~ x + (1 | g)", "categorical")

    def rebuild(*_args):
        raise AssertionError(
            "reading joint level coefficients must not compile a response fit"
        )

    monkeypatch.setattr(MultivariateFit, "_response_fit", rebuild)
    table = fitted.ranef(num_samples=3)["g"]
    assert table.components == ("muB_c_Intercept", "muC_c_Intercept")
    labels, coefficients = fitted.ranef(summary=False, num_samples=3, seed=2)["g"]
    expected = fitted.random_effects(resp="c", num_samples=3, seed=2)["g"]["Intercept"]
    assert labels == table.components
    assert torch.equal(coefficients, expected)
    single = _single("c ~ x + (1 | g)", "categorical").ranef(num_samples=3)["g"]
    assert single.components == ("muB_Intercept", "muC_Intercept")


def test_monotonic_coefficients_are_part_of_the_population():
    """A level's monotonic coefficient adds the population coefficient to it."""
    fitted = _single("y ~ x + mo(z) + (1 + mo(z) | g)", "gaussian")
    fixed = fitted.fixed_effects(num_samples=3, seed=2)
    draws = fitted.draws(num_samples=3, seed=2)
    torch.testing.assert_close(fixed["mo(z)"], draws["bsp_mo_z"])
    varying = fitted.random_effects(num_samples=3, seed=2)["g"]["mo(z)"]
    combined = fitted.coefficients(num_samples=3, seed=2)["g"]["mo(z)"]
    torch.testing.assert_close(combined, varying + draws["bsp_mo_z"].unsqueeze(2))
    assert "x" in fixed


def test_effect_kinds_are_those_of_the_terms():
    """Each design term has brms's type and its own variables."""
    fitted = _single(
        "y ~ x2 + log(x + 9) + mo(z) + s(x, by = x2, k = 4) + gp(x2, k = 3, c = 1.25)",
        "gaussian",
    )
    plan = fitted.prediction_plan
    assert plan is not None
    kinds = {
        source.label: source.kind
        for term in plan.predictors[0].terms
        for source in term.effects()
        if source.kind != "fe"
    }
    assert sorted(kinds.values()) == ["gp", "sm", "sp"]
    design = plan.designs["mu"].common
    assert design is not None
    variables = {label: design_variables(term) for label, term in design.terms.items()}
    smooth = next(label for label, kind in kinds.items() if kind == "sm")
    assert variables[smooth].names == ("x", "x2")
    assert variables[smooth].by == ("x2",)
    monotonic = next(label for label, kind in kinds.items() if kind == "sp")
    assert variables[monotonic].integer == ("z",)
    logged = next(label for label in variables if label.startswith("log"))
    assert variables[logged].names == ("x",)
