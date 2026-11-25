"""A term over an output axis is the scalar term at every coordinate.

The oracle of each test is the scalar program of the same term: a coordinate of
the vector predictor equals the scalar predictor at that coordinate's draws, on
the fitted rows and on new rows. The nonlinear references are NumPy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import BayesianFit, NonlinearExpression, fit, mixture
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry

ROWS = 24
RNG = np.random.default_rng(9)
FRAME = pd.DataFrame(
    {
        "y": RNG.normal(size=ROWS),
        "c": ["A", "B", "C"] * 8,
        "o": [0, 1, 2, 3] * 6,
        "x": RNG.normal(size=ROWS),
        "w": RNG.normal(size=ROWS),
        "z": [0, 1, 2, 3, 2, 1] * 4,
        "g": ["a", "b", "c", "d"] * 6,
        "h": ["b", "a", "d", "c"] * 6,
        "f": ["m", "n", "m", "n"] * 6,
        "t": list(range(12)) * 2,
        "u": ["p"] * 12 + ["q"] * 12,
        "loc": [f"l{index % 4}" for index in range(ROWS)],
    }
)
NEW = FRAME.iloc[[3, 8, 1, 14]].assign(x=[0.3, -0.7, 1.1, 0.2])
NEIGHBORS = pd.DataFrame(
    np.diag(np.ones(3), 1) + np.diag(np.ones(3), -1),
    index=[f"l{index}" for index in range(4)],
    columns=[f"l{index}" for index in range(4)],
)
KNOWN = pd.DataFrame(
    0.3 * np.ones((4, 4)) + np.diag([1.0, 0.8, 1.2, 0.9]),
    index=list("abcd"),
    columns=list("abcd"),
)
OPTIONS = {
    "data": FRAME,
    "method": "svi",
    "num_steps": 0,
    "data2": {"W": NEIGHBORS, "M": KNOWN},
}
TERMS = [
    "x + w",
    "s(x, k = 5)",
    "s(x, by = f, k = 4)",
    "gp(x, k = 4, c = 1.25)",
    "gp(x, w, k = 3, c = 1.25, iso = FALSE)",
    "gp(x)",
    "mo(z)",
    "x + mo(z):w",
    "(1 + x | g)",
    "(1 + x || g)",
    "(1 + x | gr(g, by = f))",
    "(1 | gr(g, dist = 'student'))",
    "(1 + x | gr(g, cov = M))",
    "(1 | mm(g, h))",
    "x + mo(z) + (1 + mo(z) | g)",
    "x + offset(w)",
    "car(W, gr = loc)",
    "car(W, gr = loc, type = 'icar')",
    "car(W, gr = loc, type = 'bym2')",
    "x + ar(t, u, cov = TRUE)",
]
CENTERED = ["(1 + x | g)", "(1 + x | gr(g, by = f))", "(1 + x | gr(g, cov = M))"]


def _registry(fitted: BayesianFit) -> LatentRegistry:
    return LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    )


def _pin(fitted: BayesianFit, values: dict[str, torch.Tensor]) -> BayesianFit:
    """Give a fit one draw with the stated values."""
    draws = {
        name: values[name].reshape(1, 1, *_registry(fitted)[name].constrained_shape)
        for name in _registry(fitted).names
    }
    return fitted.with_(
        posterior=MCMCResult(
            samples=draws,
            log_densities=torch.zeros(1, 1),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=1,
        )
    )


def _draws(fitted: BayesianFit) -> dict[str, torch.Tensor]:
    """Draw constrained values from the registered transforms."""
    registry = _registry(fitted)
    generator = torch.Generator().manual_seed(11)
    flat = 0.4 * torch.randn(
        registry.total_unconstrained_dim, generator=generator, dtype=torch.float64
    )
    return {
        name: registry[name].constrain(value)[0].double()
        for name, value in registry.unflatten_unconstrained(flat).items()
    }


def _at(
    values: dict[str, torch.Tensor],
    scalar: BayesianFit,
    coordinate: int,
    size: int,
    vector_prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Read the draws of one coordinate under the scalar program's names."""
    result = {}
    for name in _registry(scalar).names:
        shape = tuple(_registry(scalar)[name].constrained_shape)
        for source in (
            name,
            f"{name}_{coordinate}",
            f"{name}_component{coordinate + 1}",
        ):
            value = values.get(vector_prefix + source)
            if value is None:
                continue
            if source != name or value.shape == shape:
                result[name] = value
            elif tuple(value.shape) == (size, *shape):
                result[name] = value[coordinate]
            else:
                # A flat site holds the coordinates of each element consecutively.
                result[name] = value.reshape(-1, size)[:, coordinate].reshape(shape)
            break
        else:
            result[name] = torch.ones(shape, dtype=torch.float64)
    return result


@pytest.mark.parametrize(
    "term, options",
    [(term, {}) for term in TERMS]
    + [(term, {"reparameterize": "centered"}) for term in CENTERED]
    + [("x + w", {"center": True, "decomp": "QR"})],
)
def test_each_category_has_the_scalar_term(term, options):
    vector = fit(f"c ~ {term}", family="categorical", **OPTIONS, **options)
    scalar = fit(f"y ~ {term}", **OPTIONS, **options)
    assert isinstance(vector, BayesianFit) and isinstance(scalar, BayesianFit)
    values = _draws(vector)
    # A shared mixing scale is one site of both programs.
    pinned = _pin(vector, values)
    fresh = "ar(" not in term and term != "gp(x)"
    for coordinate, parameter in enumerate(("muB", "muC")):
        reference = _pin(scalar, _at(values, scalar, coordinate, 2))
        for rows in ({}, {"new_data": NEW}) if fresh else ({},):
            torch.testing.assert_close(
                pinned.posterior_linpred(parameter=parameter, **rows).double(),
                reference.posterior_linpred(**rows).double(),
                rtol=2e-5,
                atol=2e-5,
            )


@pytest.mark.parametrize(
    "term", ["x", "s(x, k = 5)", "gp(x, k = 4, c = 1.25)", "gp(x)", "mo(z)"]
)
@pytest.mark.parametrize("family", ["acat", "cumulative"])
def test_each_transition_has_the_scalar_term(term, family):
    """The effect of one transition is the scalar term at its draws."""
    vector = fit(f"o ~ 0 + w + cs({term})", family=family, **OPTIONS)
    scalar = fit(f"y ~ 0 + w + {term}", **OPTIONS)
    assert isinstance(vector, BayesianFit) and isinstance(scalar, BayesianFit)
    values = _draws(vector)
    profiles = _pin(vector, values).posterior_linpred()[0, 0]

    def renamed(name: str) -> str:
        for plain, specific in (
            ("beta_x", "beta_cs_x"),
            ("beta_s_x_k_5", "beta_cs_s_x_k_5"),
            ("smooth_s_x_k_5", "smooth_cs_s_x_k_5"),
            ("sdgp_gpx", "sdgp_csgpx"),
            ("lscale_gpx", "lscale_csgpx"),
            ("gp_gpx", "gp_csgpx"),
            ("simo_moz1", "simo_csmoz1"),
            ("bsp_mo_z", "bsp_cs_mo_z"),
        ):
            if name.startswith(plain):
                return specific + name.removeprefix(plain)
        return name

    for transition in range(3):
        local = {
            name: values[renamed(name)]
            for name in _registry(scalar).names
            if renamed(name) in values
        }
        assert any(renamed(name) != name for name in local)
        reference = _pin(scalar, _at(local, scalar, transition, 3))
        torch.testing.assert_close(
            profiles[:, transition].double(),
            reference.posterior_linpred()[0, 0].double(),
            rtol=2e-5,
            atol=2e-5,
        )


def test_fields_of_several_predictors_are_those_of_their_own_formulas():
    """Each predictor adds its own spatial field at the rows' locations."""
    fitted = fit(
        "y ~ x + car(W, gr = loc)",
        distributional={"sigma": "~ w + car(W, gr = loc, type = 'icar')"},
        **OPTIONS,
    )
    assert isinstance(fitted, BayesianFit)
    values = _draws(fitted)
    pinned = _pin(fitted, values)
    codes = FRAME["loc"].str[1:].astype(int).to_numpy()
    location = (
        values["intercept"]
        + values["beta_x"] * torch.tensor(FRAME["x"].to_numpy())
        + values["rcar"][codes]
    )
    scale = (
        values["dpar_sigma_intercept"]
        + values["dpar_sigma_beta_w"] * torch.tensor(FRAME["w"].to_numpy())
        + values["dpar_sigma_sdcar"] * values["dpar_sigma_zcar"][codes]
    )
    for parameter, expected in (("mu", location), ("sigma", scale)):
        torch.testing.assert_close(
            pinned.posterior_linpred(parameter=parameter)[0, 0].double(),
            expected,
            rtol=1e-5,
            atol=1e-5,
        )
    names = pinned.variables()
    assert {"sdcar", "car", "sdcar_sigma", "rcar[1]", "zcar_sigma[1]"} <= set(names)


def test_mixture_components_have_their_own_fields():
    fitted = fit(
        "y ~ x + car(W, gr = loc)",
        family=mixture("gaussian", "gaussian", order="none"),
        **OPTIONS,
    )
    assert {"sdcar_mu1", "sdcar_mu2", "rcar_mu1[1]", "rcar_mu2[4]"} <= set(
        fitted.variables()
    )
    with pytest.raises(ValueError, match="mixture models"):
        fit(
            "y ~ x + ar(t, u)",
            family=mixture("gaussian", "gaussian", order="none"),
            **OPTIONS,
        )


def test_categories_read_their_own_or_the_main_nonlinear_expression():
    fitted = fit(
        "c ~ a * x",
        family="categorical",
        nonlinear={"a": "~ 1 + w"},
        distributional={"muB": NonlinearExpression(expression="a * w + x")},
        **OPTIONS,
    )
    assert isinstance(fitted, BayesianFit)
    values = {
        "nlpar_a_intercept": torch.tensor(0.3, dtype=torch.float64),
        "nlpar_a_beta_w": torch.tensor(-0.7, dtype=torch.float64),
    }
    pinned = _pin(fitted, values)
    for rows in (FRAME, NEW):
        a = 0.3 - 0.7 * rows["w"].to_numpy()
        expected = {
            "muB": a * rows["w"].to_numpy() + rows["x"].to_numpy(),
            "muC": a * rows["x"].to_numpy(),
        }
        for parameter, value in expected.items():
            np.testing.assert_allclose(
                pinned.posterior_linpred(parameter=parameter, new_data=rows)[0, 0],
                value,
                rtol=1e-5,
                atol=1e-6,
            )
        logits = np.stack([np.zeros(len(rows)), expected["muB"], expected["muC"]], 1)
        np.testing.assert_allclose(
            pinned.posterior_epred(new_data=rows)[0, 0],
            np.exp(logits) / np.exp(logits).sum(1, keepdims=True),
            rtol=1e-5,
        )
