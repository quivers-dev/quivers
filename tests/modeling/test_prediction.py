"""Posterior reuse and observation-wise densities have independent oracles."""

import math

import pytest
import torch

from quivers.inference import MCMCResult
from quivers.modeling import Model, ModelFit, VariationalMethod

SOURCE = """object Row : FinSet _
object Out : Real 1
program regression(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return b
export regression
"""


def _fit(source: str = SOURCE) -> ModelFit:
    bound = Model(source).bind(
        {"x": torch.tensor([1.0, 2.0, 3.0]), "y": torch.tensor([0.5, 1.0, 1.5])}
    )
    posterior = MCMCResult(
        samples={"b": torch.tensor([[0.0, 0.25, 0.5], [0.75, 1.0, 1.25]])},
        log_densities=torch.zeros(2, 3),
        acceptance_rates=torch.ones(2),
        divergence_counts=torch.zeros(2, dtype=torch.long),
        num_warmup=0,
        num_samples=3,
    )
    return ModelFit(model=bound, posterior=posterior)


def test_log_likelihood_preserves_chains_draws_and_observation_rows():
    fit = _fit()
    actual = fit.log_likelihood()["y"]
    mu = fit.samples["b"][..., None] * fit.model.data["x"]
    expected = -0.5 * ((fit.model.data["y"] - mu).square() + math.log(2 * math.pi))
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3, 3)
    assert fit.log_likelihood(num_samples=2)["y"].shape == (2, 2, 3)
    joint = fit.model.trace({"b": fit.samples["b"][0, 1]}).log_joint
    prior = -0.5 * (fit.samples["b"][0, 1].square() + math.log(2 * math.pi))
    torch.testing.assert_close(actual[0, 1].sum(), (joint - prior).reshape(()))


def test_new_data_prediction_reuses_posterior_and_preserves_original_binding():
    fit = _fit(
        SOURCE.replace("Normal(loc=mu, scale=1.0)", "Normal(loc=mu, scale=0.000001)")
    )
    state = torch.get_rng_state().clone()
    prediction = fit.predict({"x": torch.tensor([4.0, 5.0])}, num_samples=4, seed=31)
    assert prediction["y"].shape == (4, 2)
    expected = prediction["b"].reshape(4, 1) * torch.tensor([4.0, 5.0])
    torch.testing.assert_close(prediction["y"], expected, atol=1e-5, rtol=0)
    assert fit.model.schema.describe()["x"]["type"] == "Tensor[Real]([3])"
    assert fit.model.data["y"].shape == (3,)
    assert torch.equal(state, torch.get_rng_state())
    repeated = fit.predict({"x": torch.tensor([4.0, 5.0])}, num_samples=4, seed=31)
    assert all(torch.equal(value, repeated[name]) for name, value in prediction.items())


def test_new_data_log_likelihood_uses_new_responses():
    fit = _fit()
    data = {"x": torch.tensor([4.0, 5.0]), "y": torch.tensor([-1.0, 2.0])}
    actual = fit.log_likelihood(data)["y"]
    expected = -0.5 * (
        (data["y"] - fit.samples["b"][..., None] * data["x"]).square()
        + math.log(2 * math.pi)
    )
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="requires observations"):
        fit.log_likelihood({"x": data["x"]})


def test_rebind_retains_template_arguments_and_refuses_changed_latent_layout():
    source = SOURCE.replace(
        "program regression(", "program regression(alpha : Real, "
    ).replace("Normal(loc=0.0, scale=1.0)", "Normal(loc=alpha, scale=1.0)")
    bound = Model(source).bind({"x": torch.ones(3), "y": torch.zeros(3), "alpha": 2.5})
    rebound = bound.rebind({"x": torch.ones(2), "alpha": 2.5})
    assert rebound.data["alpha"] == 2.5
    assert rebound.schema.observations == ("y",)
    local = Model(SOURCE.replace("sample b <-", "sample b : Row <-")).bind(
        {"x": torch.ones(3), "y": torch.zeros(3)}
    )
    with pytest.raises(ValueError, match="latent site layout"):
        local.rebind({"x": torch.ones(2)})


def test_rebind_reads_static_latent_layout_without_drawing(monkeypatch):
    fit = _fit()

    def unexpected_trace(*args, **kwargs):
        raise AssertionError("ordinary rebinding should not execute a prior trace")

    monkeypatch.setattr(type(fit.model), "trace", unexpected_trace)
    rebound = fit.model.rebind({"x": torch.ones(2)})
    assert rebound.data["x"].shape == (2,)


def test_variational_pointwise_likelihood_has_one_pseudo_chain():
    fit = Model(SOURCE).fit(
        {"x": torch.ones(3), "y": torch.zeros(3)}, method=VariationalMethod(num_steps=2)
    )
    assert fit.log_likelihood(num_samples=4, seed=7)["y"].shape == (1, 4, 3)
