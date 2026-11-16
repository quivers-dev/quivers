"""Poisson hurdles use the same hu role across fitting and prediction."""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit
from scipy.stats import poisson

from quivers.dsl import Compiler, loads, module_to_source
from quivers.formulas import (
    FormulaToQVRModule,
    PriorSpec,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from tests.test_formula_posterior import _recorded


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [0, 1, 3, 0],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.2, -0.3, 0.1, 0.5],
            "w": [0.0, 0.5, 1.0, 2.0],
        }
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_hurdle_probability_distributional_joint_and_all_gradients(data, weighted):
    location = formula_from_data("y~x", data, weights="w" if weighted else None)
    hurdle = formula_from_data("y~z", data)
    names = ("intercept", "beta_x", "dpar_hu_intercept", "dpar_hu_beta_z")
    lens = FormulaToQVRModule(
        families["hurdle_poisson"],
        distributional={"hu": hurdle},
        user_priors={name: "Normal(0,1)" for name in names},
    )
    module, _ = lens.forward(location)
    compiled = Compiler(module).compile().morphism
    source = module_to_source(module)
    assert "HurdlePoisson(zero_prob=hu, rate=mu)" in source
    theta = torch.tensor([0.2, -0.3, -0.4, 0.1], dtype=torch.double, requires_grad=True)
    columns = lens.fixed_column_observations(location)
    observations = (
        columns
        | dict(zip(names, theta, strict=True))
        | {"y": torch.tensor(data.y.to_numpy())}
    )
    actual = compiled.log_joint(torch.zeros(1, 1), observations).sum()
    x, z, y = (
        torch.tensor(data[name].to_numpy(), dtype=torch.double)
        for name in ("x", "z", "y")
    )
    rate = (theta[0] + theta[1] * x).exp()
    hu = (theta[2] + theta[3] * z).sigmoid()
    positive = (
        torch.log1p(-hu)
        + torch.distributions.Poisson(rate).log_prob(y)
        - (-torch.expm1(-rate)).log()
    )
    score = torch.where(y == 0, hu.log(), positive)
    if weighted:
        score = score * torch.tensor(data.w.to_numpy())
    expected = score.sum() + torch.distributions.Normal(0.0, 1.0).log_prob(theta).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, theta, retain_graph=True)[0],
        torch.autograd.grad(expected, theta)[0],
    )
    # Separate scalar reference and SciPy probability calculations agree as well.
    raw = np.where(
        data.y.to_numpy() == 0,
        np.log(expit(-0.4 + 0.1 * data.z)),
        np.log1p(-expit(-0.4 + 0.1 * data.z))
        + poisson.logpmf(data.y, np.exp(0.2 - 0.3 * data.x))
        - np.log(-np.expm1(-np.exp(0.2 - 0.3 * data.x))),
    )
    if weighted:
        raw = raw * data.w.to_numpy()
    reference = loads(source).run(
        "model",
        data={
            **{name: tuple(values.tolist()) for name, values in columns.items()},
            "y": tuple(data.y),
        },
        sites=dict(zip(names, theta.detach().tolist(), strict=True)),
    )
    assert reference.log_joint == pytest.approx(expected.item(), abs=2e-7)
    assert score.detach().sum().item() == pytest.approx(raw.sum(), abs=2e-12)


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_hurdle_probability_fixed_sampling_and_inspection(data, method):
    result = fit(
        "y~x",
        data=data,
        family="hurdle_poisson",
        fixed_parameters={"hu": 0.25},
        method=method,
        num_steps=2 if method == "svi" else None,
        num_samples=2 if method != "svi" else None,
        num_warmup=2 if method != "svi" else None,
        num_chains=1 if method != "svi" else None,
        seed=13,
    )
    assert set(get_prior("y~x", data=data, family="hurdle_poisson")) == {
        "intercept",
        "beta_x",
        "hu",
    }
    overridden = get_prior(
        "y~x",
        data=data,
        family="hurdle_poisson",
        priors=PriorSpec(distribution="Beta(3,4)", class_="hu"),
    )
    assert overridden["hu"] == "Beta(concentration1=3, concentration0=4)"
    fresh = pd.DataFrame({"x": [-0.5, 0.5]})
    assert result.posterior_epred(new_data=fresh, num_samples=2).shape == (1, 2, 2)
    predictive = result.posterior_predict(new_data=fresh, num_samples=2, seed=8)
    assert (predictive >= 0).all() and torch.isfinite(predictive).all()
    for kwargs in [{"fixed_parameters": {"zi": 0.2}}, {"distributional": {"zi": "~z"}}]:
        with pytest.raises(ValueError):
            fit(
                "y~x",
                data=data,
                family="hurdle_poisson",
                method="svi",
                num_steps=0,
                **kwargs,
            )


def test_hurdle_probability_new_rows_keep_both_predictor_scales(data):
    result = fit(
        "y~x",
        data=data,
        family="hurdle_poisson",
        distributional={"hu": "~z"},
        method="svi",
        num_steps=0,
    )
    result = _recorded(
        result,
        {
            name: torch.tensor(value, dtype=torch.double)
            for name, value in {
                "intercept": 0.2,
                "beta_x": -0.3,
                "dpar_hu_intercept": -0.4,
                "dpar_hu_beta_z": 0.1,
            }.items()
        },
    )
    fresh = pd.DataFrame({"x": [-0.5, 1.2], "z": [0.3, -0.4]})
    rate = np.exp(0.2 - 0.3 * fresh.x.to_numpy())
    expected = (1 - expit(-0.4 + 0.1 * fresh.z.to_numpy())) * rate / (-np.expm1(-rate))
    actual = result.posterior_epred(new_data=fresh)
    torch.testing.assert_close(actual, torch.tensor(expected).expand(2, 3, 2))
