"""Tests for the adaptive tempered SMC sampler.

Cover:

1. Every resampling scheme is unbiased: over many replications each
   particle's mean number of copies is ``N`` times its weight, within
   five standard errors; a resampler with a fixed offset fails.
2. The temperature ladder: each bisected step keeps the conditional ESS
   at the target, and the bisection agrees with a dense grid search.
3. The log-evidence estimate matches a conjugate model's closed form
   within Monte Carlo error over independent runs, with Metropolis and
   HMC rejuvenation and with a Gibbs block on a discrete site; an
   estimator that averages log weights instead of weights fails.
4. The final particles match the conjugate posterior.
5. `TemperedPotential` splits the log joint into its prior and
   likelihood, the sampler takes a one-row input as `MCMC` does, and a
   seeded run is reproducible without touching the global RNG.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch.distributions import MultivariateNormal, Normal

from quivers.continuous import MonadicProgram
from quivers.dsl import loads
from quivers.inference import SMC, SMCResult, TemperedPotential, resample
from quivers.inference.mcmc import (
    CycleKernel,
    GibbsKernel,
    HMCKernel,
    MCMCKernel,
    MetropolisKernel,
    RandomWalkProposal,
)
from quivers.inference.registry import LatentRegistry
from quivers.inference.smc import ResamplingScheme, _conditional_ess, _increments
from quivers.inference import smc as smc_module
from quivers.inference.trace import Trace, trace
from tests.test_inference_mcmc import _scale_likelihood, _score_program
from tests.test_inference_mcmc_compose import _switch_posterior, _switch_program


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

_WEIGHTS = torch.tensor([0.05, 0.0, 0.31, 0.12, 0.02, 0.5], dtype=torch.float64)
_REPLICATIONS = 20000


def _mean_copies(
    draw: Callable[[torch.Tensor, torch.Generator], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each particle's mean number of copies over replications, and its
    standard error."""
    generator = torch.Generator().manual_seed(0)
    n = _WEIGHTS.numel()
    counts = torch.empty(_REPLICATIONS, n, dtype=torch.float64)
    for r in range(_REPLICATIONS):
        indices = draw(_WEIGHTS, generator)
        assert indices.shape == (n,)
        counts[r] = torch.bincount(indices, minlength=n).to(torch.float64)
    return counts.mean(0), counts.std(0) / math.sqrt(_REPLICATIONS)


def _unbiased(mean: torch.Tensor, error: torch.Tensor) -> bool:
    expected = _WEIGHTS.numel() * _WEIGHTS
    exact = error == 0
    within = (mean - expected).abs() <= 5.0 * error
    return bool(torch.where(exact, mean == expected, within).all())


@pytest.mark.parametrize(
    "scheme", ["multinomial", "systematic", "stratified", "residual"]
)
def test_resampling_is_unbiased(scheme: ResamplingScheme) -> None:
    mean, error = _mean_copies(lambda w, g: resample(w, scheme, g))
    assert _unbiased(mean, error), (mean, _WEIGHTS * _WEIGHTS.numel(), error)
    # A particle of weight zero is never chosen.
    assert float(mean[1]) == 0.0


def test_a_fixed_offset_resampler_fails_the_check() -> None:
    def fixed(weights: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        del generator
        n = weights.numel()
        points = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        cumulative = torch.cumsum(weights / weights.sum(), 0)
        return torch.searchsorted(cumulative, points, right=True).clamp_max(n - 1)

    mean, error = _mean_copies(fixed)
    assert not _unbiased(mean, error)


def test_resample_refuses_bad_weights() -> None:
    with pytest.raises(ValueError, match="positive sum"):
        resample(torch.zeros(3), "systematic")
    with pytest.raises(ValueError, match="positive sum"):
        resample(torch.tensor([0.5, -0.1]), "multinomial")
    with pytest.raises(ValueError, match="unknown scheme"):
        resample(torch.ones(3), "pairwise")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The temperature ladder
# ---------------------------------------------------------------------------


def _normal_model(n: int = 10) -> MonadicProgram:
    model = loads(
        f"object Obs : FinSet {n}\n"
        "program p : Obs -> Obs\n"
        "    sample mu <- Normal(loc=0.0, scale=1.0)\n"
        "    observe y : Obs <- Normal(loc=mu, scale=1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism
    assert isinstance(model, MonadicProgram)
    return model


_Y = torch.linspace(0.5, 2.5, 10)
_X = torch.zeros(1, 1)


def _log_evidence() -> float:
    """``y ~ N(0, I + 1 1^T)`` for ``mu ~ N(0, 1)``, ``y_i ~ N(mu, 1)``."""
    n = _Y.numel()
    covariance = torch.eye(n, dtype=torch.float64) + 1.0
    return float(
        MultivariateNormal(torch.zeros(n, dtype=torch.float64), covariance).log_prob(
            _Y.to(torch.float64)
        )
    )


def test_bisection_matches_a_grid_search() -> None:
    generator = torch.Generator().manual_seed(4)
    log_likelihoods = -torch.rand(300, generator=generator, dtype=torch.float64) * 40.0
    log_weights = torch.full((300,), -math.log(300), dtype=torch.float64)
    sampler = SMC(MetropolisKernel(RandomWalkProposal(1.0)), target_ess=0.6)
    beta = 0.1
    chosen = sampler._next_temperature(beta, log_weights, log_likelihoods)
    grid = torch.linspace(0.0, 1.0 - beta, 20001, dtype=torch.float64)[1:]
    keeps = [
        float(d)
        for d in grid
        if _conditional_ess(log_weights, _increments(log_likelihoods, float(d))) >= 0.6
    ]
    spacing = float(grid[1] - grid[0])
    assert abs((chosen - beta) - max(keeps)) <= spacing
    assert _conditional_ess(
        log_weights, _increments(log_likelihoods, chosen - beta)
    ) == (pytest.approx(0.6, abs=1e-9))


def test_each_step_keeps_the_target_ess() -> None:
    result = SMC(
        MetropolisKernel(RandomWalkProposal(1.0)),
        num_particles=200,
        num_mcmc_steps=2,
        num_pilot_steps=0,
        target_ess=0.7,
        seed=1,
    ).run(_normal_model(), _X, {"y": _Y})
    assert float(result.temperatures[0]) == 0.0
    assert float(result.temperatures[-1]) == 1.0
    assert bool((result.temperatures[1:] > result.temperatures[:-1]).all())
    # Equal weights before each step, so the post-reweighting ESS is the
    # conditional ESS the bisection controlled.
    for value in result.ess[:-1].tolist():
        assert value == pytest.approx(0.7, abs=1e-6)
    assert float(result.ess[-1]) >= 0.7 - 1e-9
    assert result.num_temperatures == result.temperatures.numel() - 1
    assert result.acceptance.shape == result.ess.shape


def test_a_seeded_run_records_its_latents_whatever_the_global_rng_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The run that fixes the latents is the first `trace` the sampler
    # makes; a seeded sampler draws it from its own stream, so the value
    # it records is the same whatever the global RNG holds.
    recorded: list[float] = []

    def recording_trace(
        model: MonadicProgram, x: torch.Tensor, observations: dict[str, torch.Tensor]
    ) -> Trace:
        result = trace(model, x, observations=observations)
        recorded.append(float(result.sites["b"].value.reshape(())))
        return result

    monkeypatch.setattr(smc_module, "trace", recording_trace)
    model = _score_program(_scale_likelihood, ("b", "y"))
    data = {"y": torch.linspace(-1.0, 1.0, 6)}
    sampler = SMC(
        MetropolisKernel(RandomWalkProposal(0.1)),
        num_particles=4,
        num_mcmc_steps=1,
        num_pilot_steps=0,
        seed=3,
    )
    firsts: list[float] = []
    for ambient in range(6):
        torch.manual_seed(ambient)
        before = torch.random.get_rng_state()
        recorded.clear()
        try:
            sampler.run(model, torch.zeros(1, 1), data)
        except ValueError:
            # A prior particle below -1 makes this likelihood raise; the
            # recording has been made by then.
            pass
        assert torch.equal(torch.random.get_rng_state(), before)
        firsts.append(recorded[0])
    assert len(set(firsts)) == 1, firsts


# ---------------------------------------------------------------------------
# The evidence and the posterior
# ---------------------------------------------------------------------------


class _GeometricMeanSMC(SMC):
    """A biased variant: averages the log incremental weights."""

    @staticmethod
    def _log_evidence_increment(
        log_weights: torch.Tensor, increments: torch.Tensor
    ) -> torch.Tensor:
        return (log_weights.exp() * increments).sum()


_RUNS = 8


def _evidence_runs(
    make_kernel: Callable[[], MCMCKernel],
    model: MonadicProgram,
    observations: dict[str, torch.Tensor],
    sampler: type[SMC] = SMC,
    particles: int = 200,
) -> list[SMCResult]:
    return [
        sampler(
            make_kernel(),
            num_particles=particles,
            num_mcmc_steps=3,
            num_pilot_steps=20,
            seed=100 + r,
        ).run(model, _X, observations)
        for r in range(_RUNS)
    ]


def _within_error(estimates: list[float], truth: float) -> tuple[float, float]:
    """The error of the mean estimate and five of its standard errors."""
    values = torch.tensor(estimates, dtype=torch.float64)
    error = abs(float(values.mean()) - truth)
    return error, 5.0 * float(values.std()) / math.sqrt(len(estimates))


@pytest.mark.parametrize("rejuvenation", ["metropolis", "hmc"])
def test_log_evidence_matches_the_conjugate_closed_form(rejuvenation: str) -> None:
    def make() -> MCMCKernel:
        if rejuvenation == "metropolis":
            return MetropolisKernel(RandomWalkProposal(1.0))
        return HMCKernel(num_steps=5, mass_matrix="diagonal")

    results = _evidence_runs(make, _normal_model(), {"y": _Y})
    error, tolerance = _within_error([r.log_evidence for r in results], _log_evidence())
    assert error < tolerance, (error, tolerance)

    # The final particles are the conjugate posterior N(sum y / 11, 1 / 11).
    n = _Y.numel()
    post_mean, post_var = float(_Y.sum()) / (n + 1), 1.0 / (n + 1)
    draws = torch.cat([r.samples["mu"] for r in results]).to(torch.float64)
    effective = sum(r.num_particles * float(r.ess.min()) for r in results)
    assert abs(float(draws.mean()) - post_mean) < 5.0 * math.sqrt(post_var / effective)
    assert abs(float(draws.var()) / post_var - 1.0) < 5.0 * math.sqrt(2.0 / effective)


def test_averaging_log_weights_biases_the_evidence() -> None:
    results = _evidence_runs(
        lambda: MetropolisKernel(RandomWalkProposal(1.0)),
        _normal_model(),
        {"y": _Y},
        sampler=_GeometricMeanSMC,
    )
    error, tolerance = _within_error([r.log_evidence for r in results], _log_evidence())
    assert error > tolerance, (error, tolerance)


def test_gibbs_rejuvenation_estimates_a_mixed_model_evidence() -> None:
    """``z ~ Bernoulli(0.3)``, ``mu | z ~ N(3 z, 1)``, ``y = 2 ~ N(mu, 1)``:
    ``p(y) = sum_z p(z) N(y; 3 z, 2)``."""
    y = torch.tensor(2.0, dtype=torch.float64)
    truth = float(
        torch.logsumexp(
            torch.stack(
                [
                    math.log(0.7) + Normal(0.0, math.sqrt(2.0)).log_prob(y),
                    math.log(0.3) + Normal(3.0, math.sqrt(2.0)).log_prob(y),
                ]
            ),
            0,
        )
    )
    results = _evidence_runs(
        lambda: CycleKernel(
            [GibbsKernel(["z"]), MetropolisKernel(RandomWalkProposal(1.0))]
        ),
        _switch_program(),
        {},
        particles=150,
    )
    error, tolerance = _within_error([r.log_evidence for r in results], truth)
    assert error < tolerance, (error, tolerance)
    probs, _ = _switch_posterior()
    z = torch.cat([r.samples["z"] for r in results]).to(torch.float64)
    effective = sum(r.num_particles * float(r.ess.min()) for r in results)
    p1 = float(probs[1])
    assert abs(float(z.mean()) - p1) < 5.0 * math.sqrt(p1 * (1 - p1) / effective)


# ---------------------------------------------------------------------------
# The tempered potential and the run's contract
# ---------------------------------------------------------------------------


def test_tempered_potential_splits_prior_and_likelihood() -> None:
    model = _normal_model()
    registry = LatentRegistry.from_trace(
        model, trace(model, _X, observations={"y": _Y})
    )
    potential = TemperedPotential(model, registry, _X, {"y": _Y}, temperature=0.3)
    mu = torch.tensor([0.7])
    log_prior, log_likelihood = potential.log_prior_and_likelihood(mu)
    expected_prior = Normal(0.0, 1.0).log_prob(mu).sum()
    expected_likelihood = Normal(mu, 1.0).log_prob(_Y).sum()
    assert torch.allclose(log_prior, expected_prior)
    assert torch.allclose(log_likelihood, expected_likelihood)
    value, grad = potential.value_and_grad(mu)
    assert torch.allclose(value, expected_prior + 0.3 * expected_likelihood)
    expected_grad = -mu + 0.3 * (_Y - mu).sum()
    assert torch.allclose(grad, expected_grad, atol=1e-5)
    full = potential.at_temperature(1.0)
    assert full.temperature == 1.0 and potential.temperature == 0.3
    assert torch.allclose(full.log_density(mu), expected_prior + expected_likelihood)
    with pytest.raises(ValueError, match=r"temperature must be in \[0, 1\]"):
        potential.at_temperature(1.5)


def test_hmc_tune_sets_the_population_metric() -> None:
    model = _normal_model()
    registry = LatentRegistry.from_trace(
        model, trace(model, _X, observations={"y": _Y})
    )
    kernel = HMCKernel(mass_matrix="diagonal")
    kernel.init(registry, model, _X, {"y": _Y}, torch.zeros(1))
    population = torch.linspace(-2.0, 2.0, 41).unsqueeze(1)
    kernel.tune(population)
    n = 41.0
    expected = n / (n + 5.0) * population.var() + 1e-3 * 5.0 / (n + 5.0)
    assert torch.allclose(kernel.inverse_mass_matrix, expected.reshape(1))


def test_smc_takes_one_row_and_is_reproducible() -> None:
    sampler = SMC(
        MetropolisKernel(RandomWalkProposal(1.0)),
        num_particles=20,
        num_mcmc_steps=1,
        num_pilot_steps=0,
        seed=3,
    )
    with pytest.raises(ValueError, match="2 rows"):
        sampler.run(_normal_model(), torch.zeros(2, 1), {"y": _Y})
    torch.manual_seed(11)
    before = torch.random.get_rng_state()
    first = sampler.run(_normal_model(), _X, {"y": _Y})
    assert torch.equal(torch.random.get_rng_state(), before)
    second = sampler.run(_normal_model(), _X, {"y": _Y})
    assert torch.equal(first.samples["mu"], second.samples["mu"])
    assert first.log_evidence == second.log_evidence
    assert first.samples["mu"].shape == (20,)


def test_smc_refuses_bad_settings() -> None:
    kernel = MetropolisKernel(RandomWalkProposal(1.0))
    with pytest.raises(ValueError, match="target_ess"):
        SMC(kernel, target_ess=1.0)
    with pytest.raises(ValueError, match="num_particles"):
        SMC(kernel, num_particles=1)
    with pytest.raises(ValueError, match="resampling scheme"):
        SMC(kernel, resampling="pairwise")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mutually exclusive"):
        SMC(kernel, seed=0, generator=torch.Generator())
