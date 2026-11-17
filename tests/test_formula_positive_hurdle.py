"""Positive hurdle formulas preserve brms component and response mean semantics."""

import math

import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler, loads
from quivers.dsl.emit import module_to_source
from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.trace import trace
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from quivers.dsl.parser import parse


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [0.0, 0.2, 0.7, 0.0],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.2, -0.3, 0.0, 0.5],
        }
    )


@pytest.mark.parametrize(
    "family, auxiliary", [("hurdle_gamma", "shape"), ("hurdle_lognormal", "sigma")]
)
def test_distributional_hurdles_match_independent_density_and_gradients(
    data, family, auxiliary
):
    location = formula_from_data("y ~ x", data)
    lens = FormulaToQVRModule(
        families[family],
        distributional={
            auxiliary: formula_from_data("y ~ z", data),
            "hu": formula_from_data("y ~ x", data),
        },
    )
    module, complement = lens.forward(location)
    assert lens.backward(module, complement) == location
    loads(module_to_source(module))
    values = {
        name: torch.tensor(value, requires_grad=True)
        for name, value in {
            "intercept": -0.2,
            "beta_x": 0.1,
            f"dpar_{auxiliary}_intercept": -0.4,
            f"dpar_{auxiliary}_beta_z": 0.2,
            "dpar_hu_intercept": -0.7,
            "dpar_hu_beta_x": 0.2,
        }.items()
    }
    observations = (
        lens.fixed_column_observations(location)
        | values
        | {"y": torch.tensor(data.y.to_numpy(), dtype=torch.float32)}
    )
    recorded = trace(
        Compiler(module).compile().morphism,
        torch.zeros(1, 1, dtype=torch.long),
        observations,
    )
    eta = values["intercept"] + values["beta_x"] * observations["x"]
    aux = (
        values[f"dpar_{auxiliary}_intercept"]
        + values[f"dpar_{auxiliary}_beta_z"] * observations[f"dpar_{auxiliary}_z"]
    ).exp()
    hu = (
        values["dpar_hu_intercept"] + values["dpar_hu_beta_x"] * observations["x"]
    ).sigmoid()
    positive = observations["y"] > 0
    point = observations["y"][positive]
    if family == "hurdle_gamma":
        shape = aux[positive]
        rate = shape / eta[positive].exp()
        component = (
            shape * rate.log()
            + (shape - 1) * point.log()
            - rate * point
            - torch.lgamma(shape)
        )
    else:
        sigma = aux[positive]
        component = (
            -0.5 * ((point.log() - eta[positive]) / sigma).square()
            - sigma.log()
            - point.log()
            - 0.5 * math.log(2 * math.pi)
        )
    expected = (
        hu[~positive].log().sum() + torch.log1p(-hu[positive]).sum() + component.sum()
    )
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected)
    for observed, oracle in zip(
        torch.autograd.grad(actual, tuple(values.values()), retain_graph=True),
        torch.autograd.grad(expected, tuple(values.values())),
        strict=True,
    ):
        assert torch.isfinite(observed)
        torch.testing.assert_close(observed, oracle)


@pytest.mark.parametrize(
    "family, auxiliary, value",
    [("hurdle_gamma", "shape", 0.4), ("hurdle_lognormal", "sigma", 0.7)],
)
@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_hurdle_fit_and_posterior_expectation_agree(
    data, family, auxiliary, value, method
):
    result = fit(
        "y ~ x",
        data=data,
        family=family,
        method=method,
        fixed_parameters={auxiliary: value, "hu": 0.25},
        num_steps=2 if method == "svi" else None,
        num_samples=2 if method != "svi" else None,
        num_warmup=2 if method != "svi" else None,
        num_chains=1 if method != "svi" else None,
        seed=19,
    )
    result = result.with_(
        posterior=MCMCResult(
            samples={
                "intercept": torch.full((1, 2), -0.2),
                "beta_x": torch.full((1, 2), 0.1),
            },
            log_densities=torch.zeros(1, 2),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1),
            num_warmup=0,
            num_samples=2,
        )
    )
    eta = torch.tensor(-0.2 + 0.1 * data.x.to_numpy(), dtype=torch.float32)
    component_mean = (
        eta.exp() if family == "hurdle_gamma" else (eta + value**2 / 2).exp()
    )
    torch.testing.assert_close(result.posterior_linpred(), eta.expand(1, 2, 4))
    torch.testing.assert_close(
        result.posterior_epred(), (0.75 * component_mean).expand(1, 2, 4)
    )
    assert set(get_prior("y ~ x", data=data, family=family)) == {
        "intercept",
        "beta_x",
        "hu",
        auxiliary,
    }


@pytest.mark.parametrize(
    "family, auxiliary", [("hurdle_gamma", "shape"), ("hurdle_lognormal", "sigma")]
)
@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_fixed_hurdle_endpoints_fit_with_finite_valid_event_gradients(
    data, family, auxiliary, probability
):
    result = fit(
        "y ~ x",
        data=data.assign(y=[0.1, 0.2, 0.7, 1.1] if probability == 0 else [0.0] * 4),
        family=family,
        method="svi",
        fixed_parameters={auxiliary: 0.7, "hu": probability},
        num_steps=2,
        seed=29,
    )
    assert torch.isfinite(result.posterior_epred(num_samples=2)).all()
    if probability == 1:
        assert result.posterior_epred(num_samples=2).eq(0).all()


@pytest.mark.parametrize("family", ["hurdle_gamma", "hurdle_lognormal"])
@pytest.mark.parametrize("target", sorted(available_targets()))
def test_external_hurdle_backend_classification_is_explicit(data, family, target):
    module = FormulaToQVRModule(families[family]).forward(
        formula_from_data("y ~ 1", data)
    )[0]
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(module_to_source(module)), target=target)
    native = families[family].observe_family
    assert any(native in kind for kind in caught.value.kinds)
