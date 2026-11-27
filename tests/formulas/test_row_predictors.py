"""A predictor is read by row exactly when one of its terms varies by row."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from quivers.formulas import (
    BayesianFit,
    NonlinearExpression,
    ResponseModel,
    bf,
    fit,
)
from quivers.formulas._fit import _prepare_formula_entry

type Draw = Mapping[str, npt.NDArray[np.float64]]

ROWS = 24
X = np.random.default_rng(4).normal(size=ROWS).round(3)
FRAME = pd.DataFrame(
    {
        "y": np.random.default_rng(1).normal(size=ROWS).round(3),
        "o": [0, 1, 2, 3] * 6,
        "p1": np.random.default_rng(5).uniform(0.2, 0.4, size=ROWS).round(3),
        "p2": np.random.default_rng(6).uniform(0.2, 0.4, size=ROWS).round(3),
        "x": X,
        "s": np.full(ROWS, 0.2),
        "z": [i % 3 for i in range(ROWS)],
        "loc": [f"l{i % 4}" for i in range(ROWS)],
    }
)
FRAME["p3"] = 1.0 - FRAME["p1"] - FRAME["p2"]
LABELS = ["l0", "l1", "l2", "l3"]
CYCLE = pd.DataFrame(
    np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], float),
    index=LABELS,
    columns=LABELS,
)
CODES = np.array(FRAME["z"])


def _monotonic(prefix: str = "") -> Callable[[Draw], npt.NDArray[np.float64]]:
    def value(draw: Draw) -> npt.NDArray[np.float64]:
        steps = np.concatenate([[0.0], np.cumsum(draw[f"{prefix}simo_moz1"])])
        return draw[f"{prefix}bsp_mo_z"] * 2.0 * steps[CODES]

    return value


def _single(
    formula: str,
    *parameters: str,
    family: str | None = None,
    nl: bool = False,
) -> ResponseModel:
    model = bf(formula, *parameters, family=family, nl=nl)
    assert isinstance(model, ResponseModel)
    return model


CASES: dict[
    str,
    tuple[
        str | ResponseModel,
        str,
        str,
        Callable[[Draw], npt.NDArray[np.float64]],
    ],
] = {
    "ordinal_monotonic": (
        "o ~ 1 + mo(z)",
        "acat",
        "mu",
        lambda draw: draw["intercept"] + _monotonic()(draw),
    ),
    "ordinal_field": (
        "o ~ 1 + car(W, gr = loc)",
        "acat",
        "mu",
        lambda draw: draw["intercept"] + draw["rcar"][np.arange(ROWS) % 4],
    ),
    "ordinal_nonlinear": (
        _single("o ~ a * x", "a ~ 1", nl=True, family="acat"),
        "acat",
        "mu",
        lambda draw: draw["nlpar_a_intercept"] * X,
    ),
    "hurdle_monotonic": (
        _single("o ~ x", "hu ~ 1 + mo(z)", family="hurdle_cumulative"),
        "hurdle_cumulative",
        "hu",
        lambda draw: draw["dpar_hu_intercept"] + _monotonic("dpar_hu_")(draw),
    ),
    "discrimination_monotonic": (
        _single("o ~ x", "disc ~ 0 + mo(z)", family="acat"),
        "acat",
        "disc",
        _monotonic("dpar_disc_"),
    ),
    "nonlinear_monotonic": (
        _single("y ~ a * x", "a ~ 1 + mo(z)", nl=True),
        "gaussian",
        "mu",
        lambda draw: (draw["nlpar_a_intercept"] + _monotonic("nlpar_a_")(draw)) * X,
    ),
    "ordinal_intercept": (
        "o ~ 1",
        "acat",
        "mu",
        lambda draw: np.broadcast_to(draw["intercept"], (ROWS,)),
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_predictors_without_ordinary_covariates_fit(name):
    """The replayed predictor equals its formula evaluated on the draws."""
    formula, family, parameter, expected = CASES[name]
    fitted = fit(
        formula,
        data=FRAME,
        data2={"W": CYCLE},
        family=family,
        method="svi",
        num_steps=2,
        seed=0,
    )
    assert isinstance(fitted, BayesianFit)
    draws = fitted.draws(num_samples=3, seed=1)
    value = fitted.posterior_linpred(parameter=parameter, num_samples=3, seed=1)
    for index in range(3):
        draw = {key: item[0, index].double().numpy() for key, item in draws.items()}
        np.testing.assert_allclose(
            np.broadcast_to(value[0, index].double().numpy(), (ROWS,)),
            expected(draw),
            rtol=2e-5,
            atol=2e-5,
        )
    (likelihood,) = fitted.log_likelihood(num_samples=3, seed=1).values()
    assert likelihood.shape == (1, 3, ROWS)
    assert np.isfinite(likelihood.numpy()).all()


@pytest.mark.parametrize(
    "formula, options, scale",
    [
        (
            "y | se(s, sigma = TRUE) ~ x",
            {"distributional": {"sigma": "~ 1"}},
            lambda draw: np.exp(draw["dpar_sigma_intercept"]) * np.ones(ROWS),
        ),
        (
            "y | se(s, sigma = TRUE) ~ x",
            {"distributional": {"sigma": "~ 0 + mo(z)"}},
            lambda draw: np.exp(_monotonic("dpar_sigma_")(draw)),
        ),
        (
            "y | se(s, sigma = TRUE) ~ x",
            {
                "distributional": {"sigma": NonlinearExpression(expression="b * x")},
                "nonlinear": {"b": "b ~ 1"},
            },
            lambda draw: np.exp(draw["nlpar_b_intercept"] * X),
        ),
    ],
)
def test_known_errors_combine_with_every_residual_scale(formula, options, scale):
    """The density uses the root of the summed squares of both scales."""
    fitted = fit(formula, data=FRAME, method="svi", num_steps=2, seed=0, **options)
    assert isinstance(fitted, BayesianFit)
    draws = fitted.draws(num_samples=2, seed=1)
    likelihood = fitted.log_likelihood(num_samples=2, seed=1)["y"]
    response = FRAME["y"].to_numpy()
    for index in range(2):
        draw = {key: item[0, index].double().numpy() for key, item in draws.items()}
        # A nonlinear model reads its main formula as the expression ``x``.
        mean = draw.get("intercept", 0.0) + draw.get("beta_x", 1.0) * X
        total = np.sqrt(0.2**2 + scale(draw) ** 2)
        expected = (
            -0.5 * ((response - mean) / total) ** 2
            - np.log(total)
            - 0.5 * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(
            likelihood[0, index].double().numpy(), expected, rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("terms", ["1", "x", "0 + mo(z)", "1 + mo(z)"])
def test_dirichlet_precision_takes_any_predictor(terms):
    """The concentration reads the precision of each row."""
    fitted = fit(
        _single("cbind(p1, p2, p3) ~ x", f"phi ~ {terms}", family="dirichlet"),
        data=FRAME,
        method="svi",
        num_steps=2,
        seed=0,
    )
    assert isinstance(fitted, BayesianFit)
    (likelihood,) = fitted.log_likelihood(num_samples=2).values()
    assert likelihood.shape == (1, 2, ROWS)
    assert np.isfinite(likelihood.numpy()).all()


@pytest.mark.parametrize(
    "formula, options, expected",
    [
        ("y ~ 1", {}, False),
        ("y ~ x", {}, True),
        ("y ~ 0 + offset(x)", {}, True),
        ("y ~ 1 + (1 | loc)", {}, True),
        ("y ~ s(x, k = 4)", {}, True),
        ("y ~ 1 + gp(x)", {}, True),
        ("y ~ 1 + mo(z)", {}, True),
        ("o ~ 1 + cs(x)", {"family": "acat"}, False),
        ("o ~ 1 + (cs(x) | loc)", {"family": "acat"}, True),
        ("o ~ 1 + (0 + cs(x) | loc)", {"family": "acat"}, False),
        ("o ~ 1 + car(W, gr = loc)", {"family": "acat"}, True),
        ("y ~ 1", {"predictor_name": "net"}, True),
    ],
)
def test_location_varies_by_row_with_its_scalar_terms(formula, options, expected):
    """Category-specific terms belong to the transitions, not the scalar part."""
    prepared = _prepare_formula_entry(
        formula, data=FRAME, data2={"W": CYCLE}, **options
    )
    prepared.lens.forward(prepared.formula)
    assert prepared.lens.predictors[0].rowwise is expected
