"""Joint formula execution delegates all inference and response replay once."""

from dataclasses import replace

import pandas as pd
import pytest
import torch

from quivers.formulas import fit
from quivers.formulas._joint_fit import _fit_joint
from quivers.formulas._multivariate import MultivariateFormula, ResponseModel
from quivers.inference.prior import PriorDraws


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
@pytest.mark.parametrize("rescor", [False, True])
def test_joint_model_uses_existing_inference_and_retained_replay(method, rescor):
    specification = MultivariateFormula(
        responses={
            "first": ResponseModel(formula="y~x"),
            "second": ResponseModel(formula="z~1"),
        },
        rescor=rescor,
    )
    data = pd.DataFrame(
        {"y": [0.3, -0.4, 1.2], "z": [-0.2, 0.5, 0.1], "x": [0.2, -0.1, 0.7]}
    )
    budgets = (
        {"num_steps": 2}
        if method == "svi"
        else {"num_warmup": 2, "num_samples": 3, "num_chains": 1}
    )
    state = torch.random.get_rng_state().clone()
    fitted = _fit_joint(specification, data, method=method, seed=7, **budgets)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert fitted.response_names == ("first", "second")
    means = fitted.posterior_epred(num_samples=3, seed=5)
    predictions = fitted.posterior_predict(num_samples=3, seed=5)
    for name in fitted.response_names:
        assert means[name].shape == predictions[name].shape == (1, 3, 3)
        assert torch.isfinite(means[name]).all()
        torch.testing.assert_close(
            fitted.response(name).posterior_epred(num_samples=3, seed=5), means[name]
        )
    assert torch.equal(torch.random.get_rng_state(), state)
    likelihood = fitted.log_likelihood(num_samples=3, seed=5)
    assert set(likelihood) == (
        {"formula_joint_response"}
        if rescor
        else {"response_first__y", "response_second__z"}
    )
    assert all(
        value.shape == (1, 3, 3) and torch.isfinite(value).all()
        for value in likelihood.values()
    )


def test_heterogeneous_named_response_carriers_and_joint_prior_streams():
    specification = MultivariateFormula(
        responses={
            "time": ResponseModel(formula="y~1"),
            "correct": ResponseModel(formula="choice~1", family="bernoulli"),
        }
    )
    data = pd.DataFrame({"y": [0.3, -0.4, 1.2], "choice": [1, 0, 1]})
    fitted = _fit_joint(
        specification,
        data,
        method="svi",
        num_steps=2,
        sample_prior="yes",
        num_prior_samples=4,
        seed=3,
    )
    assert fitted.prior is not None
    predictions = fitted.posterior_predict(num_samples=4, seed=6)
    alone = fit(
        "choice~1", data=data, family="bernoulli", method="svi", num_steps=2, seed=3
    )
    assert (
        predictions["correct"].dtype
        == alone.posterior_predict(num_samples=4, seed=6).dtype
    )
    assert set(predictions["correct"].unique().tolist()) <= {0.0, 1.0}
    assert predictions["time"].dtype.is_floating_point
    assert predictions["correct"].shape == predictions["time"].shape == (1, 4, 3)
    expected = (
        fitted.draws(num_samples=4, seed=6)["response_correct__intercept"]
        .sigmoid()
        .unsqueeze(-1)
        .expand(1, 4, 3)
    )
    torch.testing.assert_close(
        fitted.posterior_epred(num_samples=4, seed=6)["correct"], expected
    )


def test_correlated_response_sampling_keeps_the_joint_noise_law():
    specification = MultivariateFormula(
        responses={
            "first": ResponseModel(formula="y~1"),
            "second": ResponseModel(formula="z~1"),
        },
        rescor=True,
    )
    data = pd.DataFrame({"y": [0.0] * 4, "z": [0.0] * 4})
    fitted = _fit_joint(specification, data, method="svi", num_steps=1)
    values = {
        "response_first__intercept": torch.tensor(0.2),
        "response_second__intercept": torch.tensor(-0.3),
        "response_first__sigma": torch.tensor(0.7),
        "response_second__sigma": torch.tensor(1.2),
        "rescor": torch.tensor([[1.0, 0.0], [0.6, 0.8]]),
    }
    posterior = PriorDraws(
        samples={
            name: value.expand(1, 500, *value.shape).clone()
            for name, value in values.items()
        },
        num_samples=500,
    )
    fitted = replace(fitted, posterior=posterior)
    predictions = fitted.posterior_predict(seed=11)
    sample = torch.stack((predictions["first"], predictions["second"]), dim=-1).reshape(
        -1, 2
    )
    torch.testing.assert_close(
        sample.mean(0), torch.tensor([0.2, -0.3]), atol=0.07, rtol=0.0
    )
    expected = torch.tensor([[0.49, 0.6 * 0.7 * 1.2], [0.6 * 0.7 * 1.2, 1.44]])
    torch.testing.assert_close(torch.cov(sample.T), expected, atol=0.1, rtol=0.08)
