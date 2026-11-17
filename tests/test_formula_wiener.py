"""Joint decision/RT formulas preserve likelihoods and response coordinates."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler
from quivers.formulas import FormulaToQVRModule, families, fit, formula_from_data
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.trace import trace
from quivers.qiec._wiener import density


@pytest.fixture
def frame():
    return pd.DataFrame(
        dict(
            rt=[0.3, 0.4, 0.5, 0.6],
            choice=[0, 1, 0, 1],
            x=[-0.2, 0.0, 0.3, 0.8],
            w=[0.0, 0.5, 1.0, 2.0],
        )
    )


def _recorded(result, count=3):
    return result.with_(
        posterior=MCMCResult(
            samples={
                "intercept": torch.full((2, count), 0.3),
                "beta_x": torch.full((2, count), -0.2),
            },
            log_densities=torch.zeros(2, count),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2, dtype=torch.long),
            num_warmup=0,
            num_samples=count,
        )
    )


def _fit(frame, **kwargs):
    return fit(
        "rt | dec(choice) ~ x",
        data=frame,
        family="wiener",
        fixed_parameters={"bs": 1.2, "ndt": 0.1, "bias": 0.4},
        method="svi",
        num_steps=1,
        **kwargs,
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_visible_response_adapter_joint_density_and_every_coefficient_gradient(
    frame, weighted
):
    formula = formula_from_data(
        "rt | dec(choice) ~ x", frame, weights="w" if weighted else None
    )
    lens = FormulaToQVRModule(
        families["wiener"], fixed_parameters={"bs": 1.2, "ndt": 0.1, "bias": 0.4}
    )
    program = Compiler(lens.forward(formula)[0]).compile().morphism
    theta = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    inputs = lens.fixed_column_observations(formula) | {
        "formula_response": torch.tensor(frame.rt.to_numpy(), dtype=torch.float64),
        "intercept": theta[0],
        "beta_x": theta[1],
    }
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), inputs)
    signed = inputs["formula_response"] * (2 * inputs["formula_decision"].long() - 1)
    torch.testing.assert_close(recorded.sites["rt"].value.reshape(-1), signed)
    assert recorded.sites["rt"].is_observed
    weights = frame.w.to_numpy() if weighted else np.ones(4)

    def oracle(t):
        return sum(
            float(w)
            * density(
                dict(boundary=1.2, ndt=0.1, bias=0.4, drift=float(t[0] + t[1] * x)),
                float(y),
            )
            for w, x, y in zip(weights, frame.x, signed, strict=True)
        )

    actual = recorded.sites["rt"].log_prob.sum()
    assert actual.item() == pytest.approx(oracle(theta.detach()), abs=3e-7)
    gradient = torch.autograd.grad(actual, theta)[0]
    for index in range(2):
        direction = torch.zeros(2, dtype=torch.float64)
        direction[index] = 1e-5
        expected = (
            oracle(theta.detach() + direction) - oracle(theta.detach() - direction)
        ) / 2e-5
        assert gradient[index].item() == pytest.approx(expected, abs=3e-7)


@pytest.mark.parametrize("fresh", [False, True])
def test_expectations_use_unsigned_rt_and_lower_upper_joint_choice_probabilities(
    frame, fresh
):
    result = _recorded(_fit(frame))
    new = pd.DataFrame({"x": [0.7, -0.4]}) if fresh else None
    x = frame.x.to_numpy() if new is None else new.x.to_numpy()
    drift = 0.3 - 0.2 * x
    upper = -np.expm1(-2 * drift * 1.2 * 0.4) / -np.expm1(-2 * drift * 1.2)
    mean = 0.1 + (1.2 * upper - 1.2 * 0.4) / drift
    torch.testing.assert_close(
        result.posterior_epred(new_data=new),
        torch.tensor(mean).expand(2, 3, len(x)),
        rtol=2e-7,
        atol=2e-8,
    )
    expected = torch.tensor(np.stack([1 - upper, upper], axis=-1)).expand(
        2, 3, len(x), 2
    )
    torch.testing.assert_close(
        result.posterior_choice_probs(new_data=new), expected, rtol=2e-7, atol=2e-8
    )
    torch.testing.assert_close(
        result.posterior_epred(new_data=new, component="decision"),
        expected,
        rtol=2e-7,
        atol=2e-8,
    )


@pytest.mark.parametrize("fresh", [False, True])
def test_lossless_structured_prediction_reconstructs_signed_joint_draws_without_clamping(
    frame, fresh
):
    result = _recorded(_fit(frame, weights="w"), count=8)
    new = pd.DataFrame({"x": [0.1] * 12}) if fresh else None
    state = torch.get_rng_state().clone()
    signed = result.posterior_predict(new_data=new, seed=73)
    decoded = result.posterior_response(new_data=new, seed=73)
    assert torch.equal(state, torch.get_rng_state())
    assert decoded.decision_labels == ("lower", "upper")
    assert decoded.decision.dtype == torch.long
    assert decoded.value.min() > 0.1
    assert set(decoded.decision.unique().tolist()) == {0, 1}
    torch.testing.assert_close(signed, decoded.value * (2 * decoded.decision - 1))
    assert signed.shape == (2, 8, 12 if fresh else 4)
    assert signed.unique().numel() == signed.numel()
    changed = result.with_(
        observations=dict(result.observations) | {"formula_weights": torch.zeros(4)}
    )
    torch.testing.assert_close(signed, changed.posterior_predict(new_data=new, seed=73))


def test_raw_likelihood_and_analysis_keep_original_measurement_coordinates(frame):
    result = _recorded(_fit(frame, weights="w"))
    raw = torch.tensor(
        [
            density(
                dict(boundary=1.2, ndt=0.1, bias=0.4, drift=0.3 - 0.2 * x),
                rt * (2 * decision - 1),
            )
            for x, rt, decision in zip(frame.x, frame.rt, frame.choice, strict=True)
        ]
    )
    torch.testing.assert_close(
        result.log_likelihood()["rt"],
        raw.expand(2, 3, 4),
        rtol=2e-6,
        atol=2e-7,
        check_dtype=False,
    )
    torch.testing.assert_close(
        result.analysis().observed_data["rt"],
        torch.tensor(frame.rt.to_numpy(), dtype=torch.float32),
    )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_all_inference_engines_fit_joint_response_data(frame, method):
    options = (
        dict(num_steps=2)
        if method == "svi"
        else dict(num_samples=2, num_warmup=2, num_chains=1)
    )
    result = fit(
        "rt ~ x",
        data=frame,
        family="wiener",
        decision="choice",
        method=method,
        fixed_parameters={"bs": 1.2, "ndt": 0.1, "bias": 0.4},
        seed=11,
        **options,
    )
    assert all(
        torch.isfinite(value).all() for value in result.draws(num_samples=2).values()
    )
    assert torch.isfinite(result.posterior_epred(num_samples=2)).all()


@pytest.mark.parametrize("bad", [0.0, -0.1, math.inf])
def test_invalid_reaction_times_rejected_before_fitting(frame, bad):
    frame.loc[0, "rt"] = bad
    with pytest.raises(ValueError, match="finite|positive"):
        _fit(frame)


def test_missing_reaction_time_removes_its_row(frame):
    frame.loc[0, "rt"] = math.nan
    with pytest.warns(UserWarning, match="rt"):
        result = _fit(frame)
    assert result.included_rows == (1, 2, 3)


def test_ndt_must_precede_every_observed_rt(frame):
    with pytest.raises(ValueError, match="ndt"):
        fit(
            "rt | dec(choice) ~ x",
            data=frame,
            family="wiener",
            method="svi",
            num_steps=1,
            fixed_parameters={"ndt": 0.3},
        )


def test_distributional_parameters_have_joint_density_and_all_eight_coefficient_derivatives(
    frame,
):
    frame["z"] = [-0.3, 0.2, 0.7, -0.1]
    formula = formula_from_data("rt | dec(choice) ~ x", frame)
    auxiliaries = {
        name: formula_from_data("rt ~ z", frame) for name in ("bs", "ndt", "bias")
    }
    lens = FormulaToQVRModule(families["wiener"], distributional=auxiliaries)
    program = Compiler(lens.forward(formula)[0]).compile().morphism
    names = [
        "intercept",
        "beta_x",
        "dpar_bs_intercept",
        "dpar_bs_beta_z",
        "dpar_ndt_intercept",
        "dpar_ndt_beta_z",
        "dpar_bias_intercept",
        "dpar_bias_beta_z",
    ]
    theta = torch.tensor(
        [0.3, -0.2, 0.1, 0.2, -2.5, 0.1, -0.4, 0.3],
        dtype=torch.float64,
        requires_grad=True,
    )
    observations = lens.fixed_column_observations(formula) | {
        "formula_response": torch.tensor(frame.rt.to_numpy(), dtype=torch.float64)
    }
    observations |= dict(zip(names, theta, strict=True))
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), observations)

    def oracle(t):
        return sum(
            density(
                dict(
                    boundary=math.exp(t[2] + t[3] * z),
                    ndt=math.exp(t[4] + t[5] * z),
                    bias=1 / (1 + math.exp(-t[6] - t[7] * z)),
                    drift=float(t[0] + t[1] * x),
                ),
                rt * (2 * choice - 1),
            )
            for x, z, rt, choice in zip(
                frame.x, frame.z, frame.rt, frame.choice, strict=True
            )
        )

    score = recorded.sites["rt"].log_prob.sum()
    assert score.item() == pytest.approx(oracle(theta.detach()), abs=4e-7)
    gradient = torch.autograd.grad(score, theta)[0]
    for index in range(len(theta)):
        step = torch.zeros_like(theta)
        step[index] = 1e-5
        expected = (
            oracle(theta.detach() + step) - oracle(theta.detach() - step)
        ) / 2e-5
        assert gradient[index].item() == pytest.approx(expected, abs=5e-7)


@pytest.mark.parametrize("regressed", [False, True])
def test_learned_delay_starts_inside_reaction_time_support(frame, regressed):
    result = fit(
        "rt | dec(choice) ~ x",
        data=frame,
        family="wiener",
        method="svi",
        num_steps=2,
        seed=17,
        distributional={"ndt": "~ x"} if regressed else None,
    )
    assert all(math.isfinite(loss) for loss in result.losses)
    assert torch.isfinite(result.posterior_epred(num_samples=2)).all()


def test_plain_response_decoding_does_not_invent_decisions():
    frame = pd.DataFrame({"y": [0.2, 0.3, 0.5]})
    result = fit("y ~ 1", data=frame, method="svi", num_steps=1)
    decoded = result.posterior_response(num_samples=2, seed=7)
    assert decoded.decision is None and decoded.decision_labels == ()
    torch.testing.assert_close(
        decoded.value, result.posterior_predict(num_samples=2, seed=7)
    )
    with pytest.raises(ValueError, match="decision response modifier"):
        result.posterior_choice_probs(num_samples=2)
