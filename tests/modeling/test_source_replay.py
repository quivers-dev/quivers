"""Source likelihoods and analyses share complete joint-draw replay."""

import math

import pytest
import torch

from quivers.diagnostics import PosteriorAnalysis
from quivers.inference import AutoNormalGuide, MCMCResult, PriorDraws
from quivers.modeling import Model, ModelFit


SOURCE = """object Row : FinSet _
program model(y:Tensor[Real]([Row])) : Row -> Row
    observe y : Row <- Normal(loc=0.25,scale=0.7)
    return y
export model
"""


def _recorded(samples, chains=2, count=4):
    return MCMCResult(
        samples=samples,
        log_densities=torch.zeros(chains, count),
        acceptance_rates=torch.ones(chains),
        divergence_counts=torch.zeros(chains, dtype=torch.long),
        num_warmup=0,
        num_samples=count,
    )


def _deterministic(posterior_kind):
    bound = Model(SOURCE).bind(
        {"y": torch.tensor([-0.3, 0.4, 1.2], dtype=torch.float64)}
    )
    posterior = (
        _recorded({})
        if posterior_kind == "mcmc"
        else PriorDraws(samples={}, num_samples=4)
        if posterior_kind == "prior"
        else AutoNormalGuide(
            bound.require_tensor_program(),
            observed_names={"y"},
            x=bound.input,
            observations=bound.tensor_data(),
        )
    )
    return ModelFit(model=bound, posterior=posterior)


@pytest.mark.parametrize("posterior_kind", ["mcmc", "prior", "guide"])
def test_parameter_free_normal_likelihood_and_waic_need_no_phantom_latent(
    posterior_kind,
):
    fitted = _deterministic(posterior_kind)
    before = torch.random.get_rng_state().clone()
    analysis = fitted.analysis(num_samples=3, seed=7)
    actual = fitted.log_likelihood(num_samples=3, seed=7)["y"]
    response = fitted.model.data["y"]
    score = (
        -0.5 * ((response - 0.25) / 0.7).square()
        - math.log(0.7)
        - 0.5 * math.log(2 * math.pi)
    )
    chains = 2 if posterior_kind == "mcmc" else 1
    torch.testing.assert_close(actual, score.expand(chains, 3, 3))
    torch.testing.assert_close(analysis.log_likelihood["y"], actual)
    assert not analysis.samples and not analysis.summary()
    assert analysis.kind == (
        "variational" if posterior_kind == "guide" else posterior_kind
    )
    if posterior_kind != "mcmc":
        assert not analysis.sample_stats
    result = analysis.waic()
    torch.testing.assert_close(result.elpd_i, score)
    assert result.p_waic == pytest.approx(0, abs=1e-15)
    assert result.waic == pytest.approx(-2 * score.sum().item())
    with pytest.raises(ValueError, match="nonempty"):
        analysis.covariance()
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("posterior_kind", ["mcmc", "prior"])
def test_parameter_free_analysis_optional_loo_uses_retained_score_axes(posterior_kind):
    pytest.importorskip("arviz")
    fitted = _deterministic(posterior_kind)
    posterior = (
        _recorded({}, count=64)
        if posterior_kind == "mcmc"
        else PriorDraws(samples={}, num_samples=64)
    )
    analysis = fitted.with_(posterior=posterior).analysis()
    tree = analysis.to_datatree(coords={"row": ["a", "b", "c"]}, dims={"y": ["row"]})
    assert "posterior" not in tree.children
    assert tuple(tree["log_likelihood"].dataset["y"].coords["row"].values) == (
        "a",
        "b",
        "c",
    )
    estimate = analysis.loo()
    assert estimate.elpd == pytest.approx(
        analysis.log_likelihood["y"][0, 0].sum().item()
    )
    assert bool(estimate.pareto_k.isnull().all())


@pytest.mark.parametrize("empty", [True, False])
def test_incomplete_retained_joint_is_refused_before_ordinary_replay(
    monkeypatch, empty
):
    source = SOURCE.replace(
        "    observe y",
        "    sample a <- Normal(loc=0.0,scale=1.0)\n    sample b <- Normal(loc=a,scale=1.0)\n    observe y",
    )
    bound = Model(source).bind({"y": torch.zeros(2)})
    fitted = ModelFit(
        model=bound, posterior=_recorded({} if empty else {"a": torch.zeros(2, 4)})
    )

    def unexpected_trace(*args, **kwargs):
        raise AssertionError("an incomplete static posterior must fail before replay")

    monkeypatch.setattr("quivers.modeling._replay.trace", unexpected_trace)
    for operation in (fitted.analysis, fitted.log_likelihood):
        with pytest.raises(ValueError, match="sites.*match"):
            operation()


def test_parameter_free_new_data_binding_retains_checked_row_shapes():
    fitted = _deterministic("prior")
    response = torch.tensor([-0.1, 0.8], dtype=torch.float64)
    actual = fitted.log_likelihood({"y": response})["y"]
    expected = (
        -0.5 * ((response - 0.25) / 0.7).square()
        - math.log(0.7)
        - 0.5 * math.log(2 * math.pi)
    )
    torch.testing.assert_close(actual, expected.expand(1, 4, 2))
    assert fitted.model.data["y"].shape == (3,)


def test_empty_parameter_analysis_still_requires_real_matching_draw_axes():
    empty = PosteriorAnalysis(samples={}, log_likelihood={})
    with pytest.raises(ValueError, match="retained draw axes"):
        empty.summary()
    mismatched = PosteriorAnalysis(
        samples={},
        log_likelihood={"y": torch.zeros(1, 3, 2), "z": torch.zeros(1, 4, 2)},
        kind="variational",
    )
    with pytest.raises(ValueError, match="matching"):
        mismatched.waic(var_name="y")
    unsupported = _deterministic("mcmc").with_(
        posterior=_recorded({}).with_(num_samples=5)
    )
    with pytest.raises(ValueError, match="statistics"):
        unsupported.log_likelihood()


def test_constant_likelihood_with_varying_retained_parameters_uses_exact_weights():
    pytest.importorskip("arviz")
    score = torch.tensor([-1.5, -0.7, -2.2], dtype=torch.float64)
    analysis = PosteriorAnalysis(
        samples={"irrelevant": torch.arange(8.0).reshape(2, 4)},
        log_likelihood={"y": score.expand(2, 4, 3)},
    )
    result = analysis.loo()
    torch.testing.assert_close(torch.as_tensor(result.elpd_i.values), score)
    assert bool(result.pareto_k.isnull().all())
    assert result.p == pytest.approx(0, abs=1e-14)


def test_varying_likelihood_keeps_arviz_psis_errors_visible(monkeypatch):
    az = pytest.importorskip("arviz")
    analysis = PosteriorAnalysis(
        samples={"theta": torch.arange(8.0).reshape(2, 4)},
        log_likelihood={"y": torch.arange(24.0).reshape(2, 4, 3)},
    )

    def reject_psis(data, **kwargs):
        assert "log_weights" not in kwargs and "pareto_k" not in kwargs
        raise ValueError("independent PSIS sentinel")

    monkeypatch.setattr(az, "loo", reject_psis)
    with pytest.raises(ValueError, match="PSIS sentinel"):
        analysis.loo()


def test_incomplete_scanned_joint_is_refused_and_restores_seeded_stream():
    from tests.dsl.test_scan_program_cells import STATE_SPACE

    bound = Model(STATE_SPACE).bind(
        {"o": torch.zeros(3, 2)}, input=torch.zeros(1, 3, 2)
    )
    samples = {f"s@{position}": torch.zeros(1, 2, 1, 3) for position in (0, 2)}
    fitted = ModelFit(model=bound, posterior=_recorded(samples, chains=1, count=2))
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="missing latent.*s@1"):
        fitted.log_likelihood(seed=13)
    assert torch.equal(before, torch.random.get_rng_state())
