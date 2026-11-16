"""Fitted source and formula models reuse the existing evidence driver."""

import math
import types

import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler
from quivers.continuous import MonadicProgram
from quivers.continuous.program_steps import Score, reading
from quivers.effects import subsample
from quivers.formulas import PriorSpec, RegularizedHorseshoe, fit
from quivers.inference import HMCKernel, MCMC, SMC, SMCResult
from quivers.inference import smc as smc_module
from quivers.inference.mcmc import MetropolisKernel, RandomWalkProposal
from quivers.modeling import MCMCMethod, Model, VariationalMethod


SOURCE = """object Row:FinSet _
object Out:Real 1
program regression(y:Tensor[Real]([Row])):Row->Out
    sample mu <- Normal(loc=0.0,scale=1.0)
    observe y:Row <- Normal(loc=mu,scale=1.0)
    return mu
export regression
"""


def _sampler(particles=256):
    return SMC(
        MetropolisKernel(RandomWalkProposal(0.5)),
        num_particles=particles,
        num_mcmc_steps=2,
        num_pilot_steps=0,
        seed=7,
    )


def _fitted(interface, method, *, weighted=False):
    values = [-0.2, 0.7, 1.1]
    weights = [0.0, 0.5, 2.0]
    if interface == "source":
        source = SOURCE
        data = {"y": torch.tensor(values, dtype=torch.float64)}
        if weighted:
            source = source.replace(
                "(y:Tensor[Real]([Row]))",
                "(y:Tensor[Real]([Row]),w:Tensor[Real]([Row]))",
            ).replace("scale=1.0)\n    return", "scale=1.0) [weights=w]\n    return")
            data["w"] = torch.tensor(weights, dtype=torch.float64)
        bound = Model(source).bind(data)
        driver = (
            VariationalMethod(num_steps=1, seed=17)
            if method == "svi"
            else MCMCMethod(
                sampler=MCMC(
                    HMCKernel(step_size=0.1, num_steps=2),
                    num_warmup=1,
                    num_samples=2,
                    num_chains=1,
                    seed=17,
                )
            )
        )
        return bound.infer(driver)
    return fit(
        "y~1",
        data=pd.DataFrame({"y": values, "w": weights}),
        weights="w" if weighted else None,
        fixed_parameters={"sigma": 1.0},
        fixed_prior="Normal(0.0,1.0)",
        method=method,
        num_steps=1 if method == "svi" else None,
        num_warmup=1 if method != "svi" else None,
        num_samples=2 if method != "svi" else None,
        num_chains=1 if method != "svi" else None,
        seed=17,
    )


@pytest.mark.parametrize("interface", ["source", "formula"])
@pytest.mark.parametrize("method", ["svi", "hmc"])
def test_evidence_matches_independent_conjugate_normal_law(interface, method):
    fitted = _fitted(interface, method)
    posterior = fitted.posterior
    rng = torch.random.get_rng_state().clone()
    result = fitted.evidence(_sampler())
    assert isinstance(result, SMCResult)
    assert fitted.posterior is posterior
    assert torch.equal(rng, torch.random.get_rng_state())
    y = torch.tensor([-0.2, 0.7, 1.1], dtype=torch.float64)
    # Integrate the scalar Gaussian prior analytically by completing its square.
    precision = 1 + y.numel()
    expected = -0.5 * (
        y.numel() * math.log(2 * math.pi)
        + math.log(precision)
        + float(y.square().sum() - y.sum().square() / precision)
    )
    assert result.log_evidence == pytest.approx(expected, abs=0.16)
    latent = "mu" if interface == "source" else "intercept"
    particles = result.samples[latent].double().flatten()
    assert particles.mean().item() == pytest.approx(
        float(y.sum() / precision), abs=0.16
    )
    assert particles.var().item() == pytest.approx(1 / precision, abs=0.10)
    assert result.num_particles == 256


@pytest.mark.parametrize("interface", ["source", "formula"])
def test_weights_estimate_power_target_normalizer_not_normalized_data_evidence(
    interface,
):
    fitted = _fitted(interface, "svi", weighted=True)
    result = fitted.evidence(_sampler())
    y = torch.tensor([-0.2, 0.7, 1.1], dtype=torch.float64)
    w = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64)
    precision = 1 + w.sum()
    expected = -0.5 * (
        w.sum() * math.log(2 * math.pi)
        + precision.log()
        + (w * y.square()).sum()
        - (w * y).sum().square() / precision
    )
    assert result.log_evidence == pytest.approx(float(expected), abs=0.16)
    normalized_sampling_law = torch.distributions.MultivariateNormal(
        torch.zeros(2, dtype=torch.float64),
        torch.diag(1 / w[1:]) + torch.ones(2, 2, dtype=torch.float64),
    ).log_prob(y[1:])
    difference = -0.5 * ((w.sum() - 2) * math.log(2 * math.pi) + w[1:].log().sum())
    torch.testing.assert_close(expected - normalized_sampling_law, difference)
    assert abs(result.log_evidence - float(normalized_sampling_law)) > 0.25


@pytest.mark.parametrize("interface", ["source", "formula"])
def test_adapter_reuses_retained_program_tensors_and_exact_driver_result(
    interface, monkeypatch
):
    fitted = _fitted(interface, "svi")
    sampler = _sampler(4)
    captured = []
    sentinel = object()

    def run(model, x, observations):
        captured.append((model, x, observations))
        return sentinel

    def forbidden(*args, **kwargs):
        raise AssertionError("evidence must not compile, render, or read the posterior")

    monkeypatch.setattr(sampler, "run", run)
    monkeypatch.setattr(Compiler, "compile", forbidden)
    monkeypatch.setattr(type(fitted), "draws", forbidden)
    if interface == "formula":
        monkeypatch.setattr(type(fitted), "_formula_lens", forbidden)
    assert fitted.evidence(sampler) is sentinel
    model, x, observations = captured.pop()
    if interface == "source":
        assert model is fitted.model.program
        assert x is fitted.model.input
        original = fitted.model.data
    else:
        assert model is fitted.program
        assert x.shape == (1, 1)
        original = fitted.observations
    assert observations is not original
    assert observations.keys() == original.keys()
    assert all(observations[name] is original[name] for name in original)


@pytest.mark.parametrize("interface", ["source", "formula"])
def test_opaque_joint_override_is_rejected_before_execution(interface, monkeypatch):
    fitted = _fitted(interface, "svi")
    program = fitted.model.program if interface == "source" else fitted.program

    def forbidden(*args, **kwargs):
        raise AssertionError("opaque evidence target must be rejected before execution")

    program.log_joint = types.MethodType(forbidden, program)
    monkeypatch.setattr(smc_module, "trace", forbidden)
    monkeypatch.setattr(smc_module, "_check_observed_support", forbidden)
    with pytest.raises(
        ValueError, match="does not expose prior/likelihood factor roles"
    ):
        fitted.evidence(_sampler(4))


@pytest.mark.parametrize("interface", ["source", "formula"])
def test_subsampled_observation_policy_is_not_bypassed(interface):
    fitted = _fitted(interface, "svi")
    with subsample("Row", torch.tensor([0])):
        with pytest.raises(ValueError, match="subsampl"):
            fitted.evidence(_sampler(4))


def test_proper_conditional_shrinkage_prior_and_fixed_sigma_share_smc_path():
    fitted = fit(
        "y~0+x",
        data=pd.DataFrame({"y": [0.2, -0.1], "x": [0.0, 1.0]}),
        fixed_parameters={"sigma": 0.7},
        priors=PriorSpec(
            distribution=RegularizedHorseshoe(
                scale_global=0.2, df_global=3.0, df=3.0, df_slab=4.0
            )
        ),
        method="svi",
        num_steps=1,
        seed=19,
    )
    result = fitted.evidence(_sampler(8))
    assert math.isfinite(result.log_evidence)
    assert "sigma" not in result.samples
    assert "beta_x" in result.samples
    assert len(result.samples) > 1  # Hyperpriors remain sampled conditional laws.


@pytest.mark.parametrize("interface", ["source", "formula"])
def test_unknown_prior_mass_is_rejected_before_running_factors(interface):
    fitted = _fitted(interface, "svi")
    base = fitted.model.program if interface == "source" else fitted.program

    def forbidden(env):
        raise AssertionError("evidence cannot silently normalize a global potential")

    program = MonadicProgram(
        base.domain,
        base.codomain,
        (
            *base.steps,
            Score(name="extra", role="potential", score=reading(forbidden, ())),
        ),
        base._return_vars,
    )
    fitted = (
        fitted.with_(model=fitted.model.with_(program=program))
        if interface == "source"
        else fitted.with_(program=program)
    )
    with pytest.raises(ValueError, match="cannot normalize explicit score potentials"):
        fitted.evidence(_sampler(4))


@pytest.mark.parametrize("interface", ["source", "formula"])
def test_seeded_adapter_is_exactly_the_direct_driver_computation(interface):
    fitted = _fitted(interface, "svi")
    if interface == "source":
        program = fitted.model.program
        x, observations = fitted.model.input, fitted.model.tensor_data()
    else:
        program = fitted.program
        x, observations = torch.zeros(1, 1), dict(fitted.observations)
    direct = _sampler(8).run(program, x, observations)
    adapted = fitted.evidence(_sampler(8))
    assert adapted.log_evidence == direct.log_evidence
    assert adapted.num_evaluations == direct.num_evaluations
    torch.testing.assert_close(
        adapted.temperatures, direct.temperatures, rtol=0, atol=0
    )
    for name in direct.samples:
        torch.testing.assert_close(
            adapted.samples[name], direct.samples[name], rtol=0, atol=0
        )


def test_bound_data_dependent_prior_keeps_its_original_scale():
    source = SOURCE.replace(
        "(y:Tensor[Real]([Row]))", "(y:Tensor[Real]([Row]),prior_scale:Real)"
    ).replace("Normal(loc=0.0,scale=1.0)", "Normal(loc=0.0,scale=prior_scale)")
    y = torch.tensor([-0.2, 0.7, 1.1], dtype=torch.float64)
    bound = Model(source).bind({"y": y, "prior_scale": torch.tensor(2.0)})
    fitted = bound.infer(VariationalMethod(num_steps=1, seed=17))
    result = fitted.evidence(_sampler())
    # The joint response covariance is I + 4*11', not I + 11'.
    expected = -0.5 * (
        y.numel() * math.log(2 * math.pi)
        + math.log(1 + 4 * y.numel())
        + float(y.square().sum() - 4 * y.sum().square() / (1 + 4 * y.numel()))
    )
    assert result.log_evidence == pytest.approx(expected, abs=0.18)
