"""Joint response laws retain event axes across both checked runtimes."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from quivers.formulas._posterior import response_mean, response_rng
from quivers.formulas._response import count_totals, response_tensor
from quivers.formulas._response_modifiers import response_input_name
from quivers.inference.trace import trace
from quivers.inference.mcmc.driver import MCMCResult
from quivers.qiec.program_runtime import run_program


VECTOR_FAMILIES = (
    "multinomial",
    "dirichlet_multinomial",
    "dirichlet",
    "dirichlet2",
    "logistic_normal",
)


def _fixed_posterior(fitted, values, draws=2):
    return fitted.with_(
        posterior=MCMCResult(
            samples={
                key: value.expand(1, draws, *value.shape).clone()
                for key, value in values.items()
            },
            log_densities=torch.zeros(1, draws),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1),
            num_warmup=0,
            num_samples=draws,
        )
    )


def _frame(counts=False):
    events = (
        [[0, 0, 0], [1, 2, 0], [2, 1, 4], [0, 3, 1]]
        if counts
        else [[0.2, 0.3, 0.5], [0.3, 0.2, 0.5], [0.15, 0.25, 0.6], [0.4, 0.3, 0.3]]
    )
    data = pd.DataFrame(events, columns=["a", "b", "c"])
    data["x"], data["z"], data["w"] = (
        [-1.0, 0.0, 1.0, 2.0],
        [0.2, -0.4, 0.3, 0.7],
        [0.0, 0.5, 1.0, 2.0],
    )
    return data


def _score(name, y, eta, phi, sigma, reference):
    if name == "logistic_normal":
        free = [i for i in range(3) if i != reference]
        ratios = (y[:, free] / y[:, reference, None]).log()
        return torch.distributions.Normal(eta, sigma).log_prob(ratios).sum(
            -1
        ) - y.log().sum(-1)
    if name == "dirichlet2":
        alpha = eta.exp()
    else:
        full = torch.zeros((len(y), 3), dtype=torch.float64)
        if reference is None:
            full = eta
        else:
            full[:, [i for i in range(3) if i != reference]] = eta
        probability = full.softmax(-1)
        alpha = probability * phi[:, None]
    if name in {"dirichlet", "dirichlet2"}:
        return torch.distributions.Dirichlet(alpha).log_prob(y)
    n = y.sum(-1)
    combination = torch.lgamma(n + 1) - torch.lgamma(y + 1).sum(-1)
    if name == "multinomial":
        return combination + (y * probability.log()).sum(-1)
    return (
        combination
        + torch.lgamma(phi)
        - torch.lgamma(n + phi)
        + (torch.lgamma(y + alpha) - torch.lgamma(alpha)).sum(-1)
    )


@pytest.mark.parametrize("name", VECTOR_FAMILIES)
@pytest.mark.parametrize("reference", [0, 2, None])
@pytest.mark.parametrize("weighted", [False, True])
def test_vector_density_and_all_predictor_gradients(name, reference, weighted):
    if (name == "dirichlet2" and reference is not None) or (
        name == "logistic_normal" and reference is None
    ):
        pytest.skip("this law fixes its coordinate convention")
    frame = _frame("multinomial" in name)
    family = families[name].with_reference(reference)
    formula = formula_from_data(
        "cbind(a,b,c)~x", frame, weights="w" if weighted else None
    )
    auxiliary = "sigma" if name == "logistic_normal" else "phi"
    distributional = (
        {}
        if name in {"multinomial", "dirichlet2"}
        else {auxiliary: formula_from_data("cbind(a,b,c)~z", frame)}
    )
    lens = FormulaToQVRModule(family, distributional=distributional)
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    width = 3 if reference is None else 2
    values = {
        "intercept": torch.linspace(
            -0.2, 0.2, width, dtype=torch.float64, requires_grad=True
        ),
        "beta_x": torch.linspace(
            0.1, -0.1, width, dtype=torch.float64, requires_grad=True
        ),
    }
    if distributional:
        shape = (2,) if name == "logistic_normal" else ()
        values[f"dpar_{auxiliary}_intercept"] = torch.full(
            shape, -0.3, dtype=torch.float64, requires_grad=True
        )
        values[f"dpar_{auxiliary}_beta_z"] = torch.full(
            shape, 0.15, dtype=torch.float64, requires_grad=True
        )
    observations = lens.fixed_column_observations(formula) | values
    response = response_input_name(formula)
    observations[response] = response_tensor(formula, family)
    if name == "logistic_normal":
        observations["rescor"] = torch.eye(2, dtype=torch.float64)
    recorded = trace(
        Compiler(module).compile().morphism, torch.zeros(1, 1), observations
    )
    eta = values["intercept"] + observations["x"].double()[:, None] * values["beta_x"]
    phi = torch.ones(len(frame), dtype=torch.float64)
    sigma = None
    if distributional:
        coefficient = values[f"dpar_{auxiliary}_intercept"]
        slope = values[f"dpar_{auxiliary}_beta_z"]
        z = observations[f"dpar_{auxiliary}_z"].double()
        linked = (coefficient + (z[:, None] if coefficient.ndim else z) * slope).exp()
        if auxiliary == "phi":
            phi = linked
        else:
            sigma = linked
    score = _score(name, observations[response].double(), eta, phi, sigma, reference)
    if weighted:
        score = score * observations["formula_weights"]
    expected = score.sum()
    actual = recorded.sites[response].log_prob.sum()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    actual_gradient = torch.autograd.grad(
        actual, tuple(values.values()), retain_graph=True
    )
    expected_gradient = torch.autograd.grad(expected, tuple(values.values()))
    for actual_value, expected_value in zip(
        actual_gradient, expected_gradient, strict=True
    ):
        torch.testing.assert_close(actual_value, expected_value, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("name", VECTOR_FAMILIES)
def test_generated_vector_program_matches_checked_reference_joint(name):
    frame = _frame("multinomial" in name)
    formula = formula_from_data("cbind(a,b,c)~x", frame)
    family = families[name]
    lens = FormulaToQVRModule(family)
    module, _ = lens.forward(formula)
    source = module_to_source(module)
    checked = lower_qvr_to_qiec(parse(source))
    width = 3 if name == "dirichlet2" else 2
    sites = {"intercept": [0.2] * width, "beta_x": [-0.1] * width}
    if name in {"dirichlet", "dirichlet_multinomial"}:
        sites["phi"] = 2.3
    elif name == "logistic_normal":
        sites.update(sigma=[0.7, 0.8], rescor=[[1.0, 0.0], [0.0, 1.0]])

    def host(value):
        return tuple(host(item) for item in value) if isinstance(value, list) else value

    data = {
        key: host(value.tolist())
        for key, value in lens.fixed_column_observations(formula).items()
    }
    response = response_input_name(formula)
    data[response] = host(response_tensor(formula, family).tolist())
    reference = run_program(
        checked,
        "model",
        data=data,
        sites={key: host(value) for key, value in sites.items()},
    )
    bindings = {
        key: torch.as_tensor(value, dtype=torch.float64)
        for key, value in (data | sites).items()
    }
    if "multinomial" in name:
        bindings["formula_trials"] = bindings["formula_trials"].long()
        bindings[response] = bindings[response].long()
    recorded = trace(Compiler(module).compile().morphism, torch.zeros(1, 1), bindings)
    assert recorded.log_joint.sum().item() == pytest.approx(
        reference.log_joint.item(), abs=2e-5
    )


def test_alr_monte_carlo_mean_matches_independent_quadrature_and_restores_rng():
    from quivers.continuous._additive_log_ratio import AdditiveLogRatioNormal

    law = AdditiveLogRatioNormal(
        torch.tensor([1.1], dtype=torch.float64),
        torch.tensor([[1.4]], dtype=torch.float64),
        reference=0,
    )
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    nonreference = np.sum(
        weights * (1 / (1 + np.exp(-(1.1 + math.sqrt(2) * 1.4 * nodes))))
    ) / math.sqrt(math.pi)
    expected = torch.tensor([1 - nonreference, nonreference], dtype=torch.float64)
    assert abs(nonreference - torch.sigmoid(torch.tensor(1.1)).item()) > 0.04
    state = torch.random.get_rng_state()
    with response_rng(931, [law.loc]):
        actual = response_mean(law, mc_samples=65536)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0)
    with response_rng(931, [law.loc]):
        torch.testing.assert_close(
            response_mean(law, mc_samples=65536), actual, atol=0, rtol=0
        )
    with pytest.raises(TypeError, match="mc_samples"):
        response_mean(law)
    with pytest.raises(ValueError, match="undefined or nonfinite"):
        response_mean(torch.distributions.Cauchy(0.0, 1.0), mc_samples=10)


@pytest.mark.parametrize("name", VECTOR_FAMILIES)
def test_vector_fit_predictions_preserve_component_order_and_new_row_totals(name):
    frame = _frame("multinomial" in name)
    family = families[name]
    if name != "dirichlet2":
        family = family.with_reference("b")
    result = fit(
        "cbind(a,b,c)~x", data=frame, family=family, method="svi", num_steps=2, seed=42
    )
    assert result.category_labels == ("a", "b", "c")
    assert result.coefficient_labels == (
        ("a", "b", "c") if name == "dirichlet2" else ("a", "c")
    )
    linear = result.posterior_linpred(num_samples=2)
    assert linear.shape == (1, 2, 4, 2 if name == "logistic_normal" else 3)
    draws = result.posterior_predict(num_samples=3, seed=19)
    assert draws.shape == (1, 3, 4, 3)
    assert next(iter(result.log_likelihood(num_samples=2).values())).shape == (1, 2, 4)
    new = pd.DataFrame({"x": [-0.2, 0.7], "n": [0, 9]})
    kwargs = {"trials": "n"} if "multinomial" in name else {}
    expected = result.posterior_epred(
        new_data=new, num_samples=2, mc_samples=512, seed=19, **kwargs
    )
    predicted = result.posterior_predict(new_data=new, num_samples=3, seed=19, **kwargs)
    assert expected.shape == (1, 2, 2, 3)
    assert predicted.shape == (1, 3, 2, 3)
    if "multinomial" in name:
        torch.testing.assert_close(
            predicted.sum(-1), torch.tensor([0, 9]).expand(1, 3, 2)
        )
        torch.testing.assert_close(
            expected.sum(-1),
            torch.tensor([0.0, 9.0], dtype=expected.dtype).expand(1, 2, 2),
        )
    else:
        torch.testing.assert_close(
            predicted.sum(-1), torch.ones_like(predicted[..., 0])
        )
        torch.testing.assert_close(expected.sum(-1), torch.ones_like(expected[..., 0]))
    if name == "logistic_normal":
        assert "rescor" in get_prior("cbind(a,b,c)~x", data=frame, family=family)


def test_exact_count_totals_and_component_links():
    frame = pd.DataFrame({"a": [2**53, 0], "b": [0, 1], "x": [0.0, 1.0]})
    formula = formula_from_data("cbind(a,b)~x", frame)
    assert count_totals(formula).tolist() == [2**53, 1]
    assert families["dirichlet"].with_links(mu="logit").location_link.name == "softmax"
    with pytest.raises(ValueError, match="softmax"):
        families["multinomial"].with_links(mu="identity")
    with pytest.raises(ValueError, match="identity"):
        families["logistic_normal"].with_links(mu="log")


@pytest.mark.parametrize(
    "name", ["dirichlet_multinomial", "dirichlet", "logistic_normal"]
)
def test_mixed_width_shared_group_block_matches_one_joint_gaussian_transform(name):
    frame = _frame("multinomial" in name).assign(g=["u", "v", "u", "v"])
    formula = formula_from_data("cbind(a,b,c)~x+(1|joint|g)", frame, weights="w")
    auxiliary = "sigma" if name == "logistic_normal" else "phi"
    other = formula_from_data("cbind(a,b,c)~z+(1|joint|g)", frame)
    family = families[name].with_reference("b")
    lens = FormulaToQVRModule(family, distributional={auxiliary: other})
    module, _ = lens.forward(formula)
    dimension = 4 if name == "logistic_normal" else 3
    raw = torch.linspace(
        -0.3,
        0.4,
        dimension * (dimension - 1) // 2,
        dtype=torch.float64,
        requires_grad=True,
    )
    factor = torch.distributions.transforms.CorrCholeskyTransform()(raw)
    scales = torch.linspace(
        0.25, 0.6, dimension, dtype=torch.float64, requires_grad=True
    )
    innovations = torch.tensor(
        [[0.2, -0.4, 0.6, 0.3], [-0.3, 0.5, 0.2, -0.2]], dtype=torch.float64
    )[:, :dimension].requires_grad_()
    coefficients = (innovations @ factor.T) * scales
    values = {
        "intercept": torch.tensor([-0.2, 0.1], dtype=torch.float64),
        "beta_x": torch.tensor([0.1, -0.1], dtype=torch.float64),
        "cor_shared_joint": factor,
        "sigma_g_Intercept": scales[:2],
        "z_g_Intercept": innovations[:, :2].reshape(-1),
        f"dpar_{auxiliary}_sigma_g_Intercept": scales[2:]
        if dimension == 4
        else scales[2],
        f"dpar_{auxiliary}_z_g_Intercept": innovations[:, 2:].reshape(-1),
        f"dpar_{auxiliary}_intercept": torch.full(
            (2,) if dimension == 4 else (), -0.3, dtype=torch.float64
        ),
        f"dpar_{auxiliary}_beta_z": torch.full(
            (2,) if dimension == 4 else (), 0.15, dtype=torch.float64
        ),
        "g_idx": torch.tensor([0, 1, 0, 1]),
    }
    if name == "logistic_normal":
        values["rescor"] = torch.eye(2, dtype=torch.float64)
    response = response_input_name(formula)
    observations = (
        lens.fixed_column_observations(formula)
        | values
        | {response: response_tensor(formula, family)}
    )
    recorded = trace(
        Compiler(module).compile().morphism, torch.zeros(1, 1), observations
    )
    groups = values["g_idx"]
    eta = (
        values["intercept"]
        + observations["x"].double()[:, None] * values["beta_x"]
        + coefficients[groups, :2]
    )
    z = observations[f"dpar_{auxiliary}_z"].double()
    aux_eta = (
        values[f"dpar_{auxiliary}_intercept"]
        + (z[:, None] if dimension == 4 else z) * values[f"dpar_{auxiliary}_beta_z"]
    )
    aux_eta = aux_eta + (
        coefficients[groups, 2:] if dimension == 4 else coefficients[groups, 2]
    )
    phi = (
        torch.ones(len(frame), dtype=torch.float64) if dimension == 4 else aux_eta.exp()
    )
    sigma = aux_eta.exp() if dimension == 4 else None
    score = _score(name, observations[response].double(), eta, phi, sigma, 1)
    expected = (score * observations["formula_weights"]).sum()
    actual = recorded.sites[response].log_prob.sum()
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    for first, second in zip(
        torch.autograd.grad(actual, (raw, scales, innovations), retain_graph=True),
        torch.autograd.grad(expected, (raw, scales, innovations)),
        strict=True,
    ):
        torch.testing.assert_close(first, second, atol=5e-6, rtol=5e-6)


def test_monte_carlo_expectation_has_a_bounded_chunk_budget():
    from quivers.continuous._additive_log_ratio import AdditiveLogRatioNormal

    class RecordingLaw(AdditiveLogRatioNormal):
        sizes = []

        def sample(self, sample_shape=torch.Size()):
            self.sizes.append(sample_shape[0])
            return super().sample(sample_shape)

    law = RecordingLaw(torch.zeros(2), torch.eye(2))
    response_mean(law, mc_samples=131)
    assert law.sizes == [64, 64, 3]
    for invalid in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="positive integer"):
            response_mean(law, mc_samples=invalid)


@pytest.mark.parametrize("name", VECTOR_FAMILIES)
def test_vector_neural_predictors_share_reference_coordinates_and_preserve_gradients(
    name,
):
    frame = _frame("multinomial" in name)
    family = families[name]
    if name != "dirichlet2":
        family = family.with_reference("b")
    predictor = torch.nn.Linear(1, 3)
    inputs = torch.tensor(frame[["x"]].values, dtype=torch.float32)
    fitted = fit(
        "cbind(a,b,c)~1",
        data=frame,
        family=family,
        method="svi",
        num_steps=2,
        predictor=predictor,
        predictor_data=inputs,
        seed=51,
    )
    assert predictor.weight.grad is not None
    assert torch.isfinite(predictor.weight.grad).all()
    assert predictor.weight.grad.abs().sum() > 0
    fresh = pd.DataFrame({"irrelevant": [1, 2]})
    kwargs = {"trials": 5} if "multinomial" in name else {}
    result = fitted.posterior_epred(
        new_data=fresh,
        predictor_data=torch.tensor([[-0.2], [0.7]]),
        num_samples=2,
        mc_samples=128,
        **kwargs,
    )
    assert result.shape == (1, 2, 2, 3)
    assert torch.isfinite(result).all()


def test_shared_vector_and_scalar_group_coordinates_survive_new_group_replay():
    frame = _frame().assign(g=["u", "v", "u", "v"])
    fitted = fit(
        "cbind(a,b,c)~1+(1|joint|g)",
        data=frame,
        family=families["dirichlet"].with_reference("b"),
        distributional={"phi": "~1+(1|joint|g)"},
        method="svi",
        num_steps=1,
        seed=29,
    )
    values = {
        "intercept": torch.zeros(2),
        "dpar_phi_intercept": torch.tensor(0.0),
        "sigma_g_Intercept": torch.ones(2),
        "dpar_phi_sigma_g_Intercept": torch.tensor(1.0),
        "cor_shared_joint": torch.eye(3),
        "z_g_Intercept": torch.tensor([0.2, -0.1, 0.6, -0.3]),
        "dpar_phi_z_g_Intercept": torch.tensor([0.4, 1.2]),
    }
    fitted = _fixed_posterior(fitted, values, draws=5)
    new = pd.DataFrame({"g": ["v", "u", "fresh", "fresh"]})
    expected_free = torch.tensor([[0.6, -0.3], [0.2, -0.1]])
    linear = fitted.posterior_linpred(new_data=new, allow_new_levels=True, seed=83)
    torch.testing.assert_close(linear[0, 0, :2][:, [0, 2]], expected_free)
    torch.testing.assert_close(linear[..., 2, :], linear[..., 3, :])
    phi = fitted.posterior_linpred(
        parameter="phi", new_data=new, allow_new_levels=True, seed=83
    )
    torch.testing.assert_close(phi[..., :2], torch.tensor([1.2, 0.4]).expand(1, 5, 2))
    # The donor is shared across all coordinates and auxiliary predictors.
    torch.testing.assert_close(phi[..., 2], 2 * linear[..., 2, 0])
    assert fitted.random_effects()["g"]["Intercept"].shape == (1, 5, 2, 2)
    assert fitted.random_effects(parameter="phi")["g"]["Intercept"].shape == (1, 5, 2)
    mean = fitted.posterior_epred(new_data=new, allow_new_levels=True, seed=83)
    torch.testing.assert_close(mean.sum(-1), torch.ones_like(mean[..., 0]))
    gaussian = fitted.posterior_epred(
        new_data=new, allow_new_levels=True, sample_new_levels="gaussian", seed=83
    )
    assert torch.isfinite(gaussian).all()
    torch.testing.assert_close(gaussian[..., 2, :], gaussian[..., 3, :])


@pytest.mark.parametrize(
    "events",
    [
        [[-1, 2, 3]],
        [[0.5, 1, 2]],
        [[2**53, 1, 0]],
    ],
)
def test_vector_count_contract_rejects_invalid_events_before_fitting(events):
    frame = pd.DataFrame(events, columns=["a", "b", "c"])
    formula = formula_from_data("cbind(a,b,c)~1", frame)
    with pytest.raises(ValueError, match="count"):
        FormulaToQVRModule(families["multinomial"]).forward(formula)


def test_nonfinite_retained_event_is_rejected_by_shared_count_contract():
    formula = formula_from_data("cbind(a,b,c)~1", _frame(True))
    values = formula.response_values.astype(float).copy()
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        count_totals(formula.with_(response_values=values))


def test_vector_count_fresh_rows_require_explicit_totals_and_validate_training_totals():
    frame = _frame(True)
    with pytest.raises(ValueError, match="totals"):
        fit(
            "cbind(a,b,c)~x",
            data=frame,
            family="multinomial",
            binomial_trials=4,
            num_steps=1,
            method="svi",
        )
    fitted = fit(
        "cbind(a,b,c)~x", data=frame, family="multinomial", num_steps=1, method="svi"
    )
    new = pd.DataFrame({"x": [0.0, 1.0]})
    with pytest.raises(ValueError, match="trials"):
        fitted.posterior_epred(new_data=new, num_samples=1)
    with pytest.raises(ValueError, match="trials"):
        fitted.posterior_predict(new_data=new, num_samples=1)
    assert fitted.posterior_linpred(new_data=new, num_samples=1).shape == (1, 1, 2, 3)


@pytest.mark.parametrize("name", VECTOR_FAMILIES)
@pytest.mark.parametrize("method", ["hmc", "nuts"])
def test_vector_laws_support_hamiltonian_inference(name, method):
    frame = _frame("multinomial" in name)
    fixed = (
        {"phi": 2.0}
        if name in {"dirichlet", "dirichlet_multinomial"}
        else {"sigma": 0.7}
        if name == "logistic_normal"
        else {}
    )
    fitted = fit(
        "cbind(a,b,c)~1",
        data=frame,
        family=name,
        fixed_parameters=fixed,
        method=method,
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        seed=55,
    )
    assert all(torch.isfinite(value).all() for value in fitted.draws().values())
    assert fitted.posterior_predict(seed=7).shape == (1, 2, 4, 3)


@pytest.mark.parametrize("name", VECTOR_FAMILIES)
def test_pointwise_vector_likelihood_reduces_event_once_and_ignores_weights(name):
    frame = _frame("multinomial" in name)
    frame["w"] = [0.0, 0.0, 0.5, 2.0]
    family = families[name]
    fitted = fit(
        "cbind(a,b,c)~x",
        data=frame,
        family=family,
        weights="w",
        method="svi",
        num_steps=1,
        seed=86,
    )
    width = 3 if name == "dirichlet2" else 2
    values = {
        "intercept": torch.linspace(-0.2, 0.2, width, dtype=torch.float64),
        "beta_x": torch.linspace(0.1, -0.1, width, dtype=torch.float64),
    }
    if name in {"dirichlet", "dirichlet_multinomial"}:
        values["phi"] = torch.tensor(2.3, dtype=torch.float64)
    elif name == "logistic_normal":
        values.update(
            sigma=torch.tensor([0.7, 0.8], dtype=torch.float64),
            rescor=torch.eye(2, dtype=torch.float64),
        )
    fitted = _fixed_posterior(fitted, values)
    eta = (
        values["intercept"]
        + fitted.observations["x"].double()[:, None] * values["beta_x"]
    )
    phi = torch.full((len(frame),), 2.3, dtype=torch.float64)
    reference = None if name == "dirichlet2" else 0
    oracle = _score(
        name,
        response_tensor(fitted.formula, family).double(),
        eta,
        phi,
        values.get("sigma"),
        reference,
    )
    actual = next(iter(fitted.log_likelihood().values()))
    assert actual.shape == (1, 2, 4)
    torch.testing.assert_close(actual, oracle.expand(1, 2, 4), rtol=2e-6, atol=2e-6)
    assert not torch.equal(actual[..., 1], torch.zeros_like(actual[..., 1]))


def test_alr_posterior_expectation_uses_original_component_order_and_explicit_mc():
    frame = pd.DataFrame({"a": [0.2, 0.3, 0.4], "b": [0.8, 0.7, 0.6]})
    fitted = fit(
        "cbind(a,b)~1", data=frame, family="logistic_normal", method="svi", num_steps=1
    )
    fitted = _fixed_posterior(
        fitted,
        {
            "intercept": torch.tensor([1.1], dtype=torch.float64),
            "sigma": torch.tensor([1.4], dtype=torch.float64),
        },
        draws=1,
    )
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    positive = np.sum(
        weights / (1 + np.exp(-(1.1 + math.sqrt(2) * 1.4 * nodes)))
    ) / math.sqrt(math.pi)
    expected = torch.tensor([1 - positive, positive], dtype=torch.float64)
    with pytest.raises(TypeError, match="mc_samples"):
        fitted.posterior_epred()
    state = torch.random.get_rng_state()
    result = fitted.posterior_epred(mc_samples=65536, seed=41)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(result, expected.expand(1, 1, 3, 2), atol=0.004, rtol=0)
    torch.testing.assert_close(
        result, fitted.posterior_epred(mc_samples=65536, seed=41), atol=0, rtol=0
    )
