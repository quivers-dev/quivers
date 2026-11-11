"""Tests for simulation-based calibration in `quivers.diagnostics`.

The calibrated reference is a linear-Gaussian program whose posterior
is available in closed form: a scalar intercept and a three-coordinate
plate of slopes with Normal priors, observed through Normal noise at
fixed covariates. A fit that samples that posterior exactly must give
uniform ranks; fits whose draws are too narrow, too wide, or shifted
must not.

Error rates are set before looking at results. A uniformity test is
judged at ``ALPHA = 0.001`` per statistic; the calibrated checks run
5 seeds of 4 statistics, so by the Bonferroni bound the chance that
any of those 20 tests rejects a correct sampler is at most 0.02. The
coverage checks of the calibrated runs use ``family_error = 0.002``
per seed, at most 0.01 over the 5 seeds. All seeds are fixed, so each
outcome is deterministic; the bounds say how unlikely a failure would
be for a correct implementation. Detection checks require the
smallest p-value of each miscalibrated fit to fall below ``ALPHA``;
they use 400 simulations, at which each miscalibration reached the
smallest attainable p-value on all of 10 independent seeds outside the
test (200 simulations left the 1.4-fold widening undetected on one of
the test's seeds).
"""

from __future__ import annotations

import math

import arviz_plots as azp
import numpy as np
import pytest
import torch
from scipy import stats

from quivers.diagnostics import (
    SBCCoverage,
    SBCResult,
    sbc,
    sbc_coverage,
    sbc_ranks,
    sbc_uniformity,
    to_datatree_from_sbc,
)
from quivers.diagnostics.calibration import _binomial_tails, _ecdf_statistic
from quivers.dsl import loads
from quivers.inference import MCMC, HMCKernel, MCMCResult
from quivers.inference.guides import AutoMultivariateNormalGuide

ALPHA = 0.001
NUM_REFERENCE = 4999

SOURCE = """
object Item : FinSet 3

program regression : Item -> Item
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample beta : Item <- Normal(loc=0.0, scale=2.0)
    let m = mu + beta * x_design
    observe y : Item <- Normal(loc=m, scale=0.5)
    return y

export regression
"""

MODEL = loads(SOURCE).morphism
X = torch.zeros(1, 1)
COVARIATES = torch.tensor([1.0, -0.5, 2.0])
DATA = {"x_design": COVARIATES}
_DESIGN = torch.cat([torch.ones(3, 1), torch.diag(COVARIATES)], dim=1).double()
_PRIOR_PRECISION = torch.diag(torch.tensor([1.0, 0.25, 0.25, 0.25])).double()
_NOISE_VARIANCE = 0.25


def _exact_posterior(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and covariance of (mu, beta) given y, in closed form."""
    precision = _PRIOR_PRECISION + _DESIGN.T @ _DESIGN / _NOISE_VARIANCE
    covariance = torch.linalg.inv(precision)
    mean = covariance @ (_DESIGN.T @ y.double().reshape(3) / _NOISE_VARIANCE)
    return mean, covariance


def _gaussian_fit(
    scale: float = 1.0, shift: float = 0.0, num_draws: int = 99
):  # -> Callable[[dict[str, torch.Tensor], int], dict[str, torch.Tensor]]
    """A fit drawing independently from a scaled, shifted exact posterior."""

    def fit(
        observations: dict[str, torch.Tensor], seed: int
    ) -> dict[str, torch.Tensor]:
        mean, covariance = _exact_posterior(observations["y"])
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(num_draws, 4, generator=generator, dtype=torch.float64)
        factor = torch.linalg.cholesky(covariance)
        draws = mean + shift * covariance.diagonal().sqrt() + scale * noise @ factor.T
        return {
            "mu": draws[:, 0].reshape(1, num_draws),
            "beta": draws[:, 1:].reshape(1, num_draws, 3),
        }

    return fit


def _run(fit, seed: int, num_simulations: int = 200, **options) -> SBCResult:
    return sbc(
        MODEL,
        X,
        fit,
        observed_names={"y"},
        num_simulations=num_simulations,
        seed=seed,
        data=DATA,
        **options,
    )


# The rank primitive.


def test_ranks_match_a_hand_computed_case_without_ties() -> None:
    truth = torch.tensor([0.3, 2.0, -1.0])
    draws = torch.tensor([[0.1, 1.0, 0.0], [0.2, 3.0, 1.0], [0.5, 2.5, 2.0]])
    ranks = sbc_ranks(truth, draws, generator=torch.Generator().manual_seed(0))
    assert ranks.dtype == torch.int64
    assert ranks.tolist() == [2, 1, 0]


def test_tied_ranks_are_spread_uniformly_over_the_tie() -> None:
    # Truth 0.5 among (0.1, 0.5, 0.5, 0.9): one draw below, two tied, so
    # the rank is 1, 2, or 3 with probability 1/3 each. With 3000
    # independent copies each count has sd sqrt(3000 * 2/9) = 25.8; the
    # bound below is 4.5 sd.
    copies = 3000
    truth = torch.full((copies,), 0.5)
    draws = torch.tensor([0.1, 0.5, 0.5, 0.9]).unsqueeze(1).expand(4, copies)
    ranks = sbc_ranks(truth, draws, generator=torch.Generator().manual_seed(1))
    counts = torch.bincount(ranks, minlength=5).tolist()
    assert counts[0] == counts[4] == 0
    for count in counts[1:4]:
        assert abs(count - copies / 3) < 4.5 * math.sqrt(copies * 2 / 9)


def test_randomized_ties_make_discrete_ranks_uniform_and_ignoring_ties_does_not() -> (
    None
):
    # A calibrated discrete quantity: truth and draws from one Poisson.
    generator = torch.Generator().manual_seed(2)
    num_simulations, num_draws = 300, 19
    poisson = torch.distributions.Poisson(torch.tensor(2.0))
    with torch.random.fork_rng():
        torch.manual_seed(3)
        truth = poisson.sample((num_simulations,))
        draws = poisson.sample((num_draws, num_simulations))
    randomized = sbc_ranks(truth, draws, generator=generator)
    below_only = (draws < truth).sum(dim=0)
    assert (
        float(sbc_uniformity(randomized, num_draws, num_reference=NUM_REFERENCE))
        > ALPHA
    )
    assert (
        float(sbc_uniformity(below_only, num_draws, num_reference=NUM_REFERENCE))
        < ALPHA
    )


def test_ranks_reject_mismatched_or_non_finite_input() -> None:
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="shape"):
        sbc_ranks(torch.zeros(3), torch.zeros(5, 2), generator=generator)
    with pytest.raises(ValueError, match="finite"):
        sbc_ranks(torch.tensor(float("nan")), torch.zeros(5), generator=generator)


# The uniformity test.


def test_uniformity_statistic_matches_a_hand_computed_case() -> None:
    # Two ranks, L = 1: the one threshold k = 0 counts ranks at or below
    # 0, Binomial(2, 1/2). Ranks (0, 0) give count 2, two-sided tail
    # 2 * P(count >= 2) = 0.5. Under the null the statistic is 0.5 for
    # counts 0 and 2 (probability 1/2) and 1 for count 1, so the exact
    # p-value is 1/2; with 20000 reference draws its Monte Carlo sd is
    # 0.0035, and the bound below is 4.5 sd.
    lower, upper = _binomial_tails(2, 1)
    statistic = _ecdf_statistic(torch.tensor([[0], [0]]), 1, lower, upper)
    assert float(statistic[0]) == pytest.approx(0.5, rel=1e-12)
    p_value = float(sbc_uniformity(torch.tensor([0, 0]), 1, num_reference=20000))
    assert abs(p_value - 0.5) < 4.5 * math.sqrt(0.25 / 20000)


def test_binomial_tail_tables_match_scipy() -> None:
    lower, upper = _binomial_tails(40, 9)
    for k in range(9):
        probability = (k + 1) / 10
        np.testing.assert_allclose(
            lower[k].numpy(),
            stats.binom.cdf(np.arange(41), 40, probability),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            upper[k].numpy(),
            stats.binom.sf(np.arange(41) - 1, 40, probability),
            rtol=1e-10,
        )


@pytest.mark.parametrize(("num_simulations", "num_draws"), [(50, 9), (20, 1)])
def test_uniformity_test_has_its_nominal_size(
    num_simulations: int, num_draws: int
) -> None:
    # 300 replicates of exactly uniform ranks: the fraction of p-values
    # at or below 0.1 is at most 0.1 for a valid test (discreteness can
    # only make it smaller); 30 expected, sd 5.2, bound 4.5 sd above.
    generator = torch.Generator().manual_seed(4)
    rejections = 0
    for replicate in range(300):
        ranks = torch.randint(0, num_draws + 1, (num_simulations,), generator=generator)
        p_value = float(
            sbc_uniformity(ranks, num_draws, num_reference=499, seed=replicate)
        )
        rejections += p_value <= 0.1
    assert rejections <= 30 + 4.5 * math.sqrt(300 * 0.1 * 0.9)
    assert rejections >= 30 - 4.5 * math.sqrt(300 * 0.1 * 0.9)


def test_uniformity_rejects_malformed_ranks() -> None:
    with pytest.raises(ValueError, match="integers"):
        sbc_uniformity(torch.zeros(10), 5)
    with pytest.raises(ValueError, match="lie in"):
        sbc_uniformity(torch.tensor([0, 6]), 5)
    with pytest.raises(ValueError, match="num_reference"):
        sbc_uniformity(torch.tensor([0, 1]), 5, num_reference=0)


# Coverage.


@pytest.mark.parametrize(
    ("num_draws", "mass", "band", "null_mass"),
    [
        (9, 0.8, (1, 8), 0.8),
        (10, 0.9, (0, 10), 1.0),
        (99, 0.9, (5, 94), 0.9),
        (19, 0.5, (5, 14), 0.5),
    ],
)
def test_coverage_band_and_exact_null_mass(
    num_draws: int, mass: float, band: tuple[int, int], null_mass: float
) -> None:
    ranks = torch.arange(num_draws + 1)
    result = sbc_coverage(ranks, num_draws, mass=mass)
    assert isinstance(result, SBCCoverage)
    assert (result.lower_rank, result.upper_rank) == band
    assert result.null_mass == pytest.approx(null_mass, rel=1e-12)
    # Every rank once: the coverage is the null mass exactly.
    assert float(result.coverage[0]) == pytest.approx(null_mass, rel=1e-12)


def test_coverage_intervals_are_bonferroni_clopper_pearson() -> None:
    generator = torch.Generator().manual_seed(5)
    ranks = torch.randint(0, 100, (150, 3), generator=generator)
    ranks[:, 2] = 50
    result = sbc_coverage(ranks, 99, mass=0.9, family_error=0.06)
    confidence = 1.0 - 0.06 / 3
    for column in range(3):
        inside = int(((ranks[:, column] >= 5) & (ranks[:, column] <= 94)).sum())
        interval = stats.binomtest(inside, 150).proportion_ci(
            confidence_level=confidence, method="exact"
        )
        assert float(result.lower[column]) == pytest.approx(
            interval.low, rel=1e-10, abs=1e-15
        )
        assert float(result.upper[column]) == pytest.approx(interval.high, rel=1e-10)
        assert float(result.coverage[column]) == pytest.approx(inside / 150, rel=1e-12)
    assert not bool(result.consistent[2])


# The SBC driver on the closed-form posterior.


@pytest.mark.parametrize("seed", range(5))
def test_exact_posterior_sampling_is_calibrated(seed: int) -> None:
    result = _run(_gaussian_fit(), seed, thin=1)
    assert result.statistic_names == ("mu", "beta[0]", "beta[1]", "beta[2]")
    assert result.ranks.shape == (200, 4)
    assert result.num_draws == 99
    p_values = sbc_uniformity(
        result.ranks, result.num_draws, num_reference=NUM_REFERENCE
    )
    assert bool((p_values > ALPHA).all()), p_values
    coverage = sbc_coverage(result.ranks, result.num_draws, family_error=0.002)
    assert bool(coverage.consistent.all()), coverage.coverage


@pytest.mark.parametrize(
    ("label", "scale", "shift"),
    [("too narrow", 0.7, 0.0), ("too wide", 1.4, 0.0), ("biased", 1.0, 0.4)],
)
@pytest.mark.parametrize("seed", [10, 11])
def test_miscalibrated_posteriors_are_detected(
    label: str, scale: float, shift: float, seed: int
) -> None:
    result = _run(
        _gaussian_fit(scale=scale, shift=shift), seed, num_simulations=400, thin=1
    )
    p_values = sbc_uniformity(
        result.ranks, result.num_draws, num_reference=NUM_REFERENCE
    )
    assert float(p_values.min()) < ALPHA, (label, p_values)
    if label != "biased":
        coverage = sbc_coverage(result.ranks, result.num_draws, family_error=0.002)
        assert not bool(coverage.consistent.any()), (label, coverage.coverage)


def _ar1_fit(phi: float, num_chains: int = 2, num_draws: int = 500):
    """Autocorrelated chains whose stationary law is the exact posterior."""

    def fit(
        observations: dict[str, torch.Tensor], seed: int
    ) -> dict[str, torch.Tensor]:
        mean, covariance = _exact_posterior(observations["y"])
        factor = torch.linalg.cholesky(covariance)
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(
            num_chains, num_draws, 4, generator=generator, dtype=torch.float64
        )
        state = noise[:, 0]
        path = [state]
        innovation = math.sqrt(1.0 - phi**2)
        for t in range(1, num_draws):
            state = phi * state + innovation * noise[:, t]
            path.append(state)
        draws = mean + torch.stack(path, dim=1) @ factor.T
        return {"mu": draws[..., 0], "beta": draws[..., 1:]}

    return fit


def test_autocorrelated_exact_chains_need_ess_thinning() -> None:
    fit = _ar1_fit(phi=0.98)
    unthinned = _run(fit, 20, thin=1)
    thinned = _run(fit, 20, thin="ess")
    assert unthinned.num_draws == 1000
    assert int(thinned.thinning.min()) > 10
    assert thinned.num_draws < 100
    raw_p = sbc_uniformity(
        unthinned.ranks, unthinned.num_draws, num_reference=NUM_REFERENCE
    )
    thin_p = sbc_uniformity(
        thinned.ranks, thinned.num_draws, num_reference=NUM_REFERENCE
    )
    assert float(raw_p.min()) < ALPHA, raw_p
    assert bool((thin_p > ALPHA).all()), thin_p


def test_a_guide_at_the_exact_posterior_is_calibrated_and_a_narrow_one_is_not() -> None:
    def guide_fit(scale: float):
        def fit(
            observations: dict[str, torch.Tensor], seed: int
        ) -> AutoMultivariateNormalGuide:
            mean, covariance = _exact_posterior(observations["y"])
            factor = torch.linalg.cholesky(covariance) * scale
            guide = AutoMultivariateNormalGuide(MODEL, {"y"})
            assert guide.registry.names == ("mu", "beta")
            diagonal = factor.diagonal() - 1e-6
            with torch.no_grad():
                guide.loc.copy_(mean.float())
                guide.scale_diag_raw.copy_(torch.log(torch.expm1(diagonal)).float())
                guide.scale_offdiag.copy_(torch.tril(factor, diagonal=-1).float())
            return guide

        return fit

    calibrated = _run(
        guide_fit(1.0), 30, num_simulations=150, thin=1, num_guide_draws=99
    )
    narrow = _run(guide_fit(0.6), 30, num_simulations=150, thin=1, num_guide_draws=99)
    assert calibrated.num_draws == 99
    calibrated_p = sbc_uniformity(
        calibrated.ranks, calibrated.num_draws, num_reference=NUM_REFERENCE
    )
    narrow_p = sbc_uniformity(
        narrow.ranks, narrow.num_draws, num_reference=NUM_REFERENCE
    )
    assert bool((calibrated_p > ALPHA).all()), calibrated_p
    assert float(narrow_p.min()) < ALPHA, narrow_p


def test_mcmc_fit_runs_under_one_run_inputs_and_reproduces_from_its_seed() -> None:
    def fit(observations: dict[str, torch.Tensor], seed: int) -> MCMCResult:
        kernel = HMCKernel(step_size=0.2, num_steps=8, mass_matrix="identity")
        driver = MCMC(kernel, num_warmup=40, num_samples=60, num_chains=2, seed=seed)
        return driver.run(MODEL, X, observations)

    first = _run(fit, 40, num_simulations=4)
    second = _run(fit, 40, num_simulations=4)
    assert torch.equal(first.ranks, second.ranks)
    assert torch.equal(first.truths, second.truths)
    assert torch.equal(first.thinning, second.thinning)
    assert first.num_draws == int(first.retained.min())
    assert bool((first.thinning >= 1).all())
    assert bool((first.ranks >= 0).all()) and bool(
        (first.ranks <= first.num_draws).all()
    )


def test_runs_reproduce_from_the_seed_and_leave_the_global_generator_alone() -> None:
    state = torch.get_rng_state()
    first = _run(_gaussian_fit(), 50, num_simulations=20, thin=1)
    assert torch.equal(torch.get_rng_state(), state)
    second = _run(_gaussian_fit(), 50, num_simulations=20, thin=1)
    other = _run(_gaussian_fit(), 51, num_simulations=20, thin=1)
    assert torch.equal(first.ranks, second.ranks)
    assert torch.equal(first.truths, second.truths)
    assert not torch.equal(first.truths, other.truths)
    assert first.seed == 50


def test_truths_are_the_prior_draws_of_the_simulated_data() -> None:
    seen: list[torch.Tensor] = []

    def fit(
        observations: dict[str, torch.Tensor], seed: int
    ) -> dict[str, torch.Tensor]:
        seen.append(observations["y"].reshape(3))
        assert torch.equal(observations["x_design"], COVARIATES)
        return _gaussian_fit()(observations, seed)

    result = _run(fit, 60, num_simulations=200, thin=1)
    truths = result.truths
    # The prior: mu ~ N(0, 1) and beta ~ N(0, 4), checked by mean and sd
    # at 4.5 standard errors.
    assert abs(float(truths[:, 0].mean())) < 4.5 / math.sqrt(200)
    assert abs(float(truths[:, 1:].mean())) < 4.5 * 2.0 / math.sqrt(600)
    # The data given mu and beta: residuals are N(0, 0.25).
    observed = torch.stack(seen).double()
    predicted = truths[:, :1] + truths[:, 1:] * COVARIATES.double()
    residual = observed - predicted
    assert abs(float(residual.std()) - 0.5) < 4.5 * 0.5 / math.sqrt(2 * 600)


def test_named_statistics_can_read_the_data() -> None:
    def residual_sum(
        parameters: dict[str, torch.Tensor], data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        predicted = (
            parameters["mu"].unsqueeze(1) + parameters["beta"] * data["x_design"]
        )
        return (data["y"].reshape(1, 3) - predicted).pow(2).sum(dim=1)

    def slopes(
        parameters: dict[str, torch.Tensor], data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return parameters["beta"][:, :2]

    result = _run(
        _gaussian_fit(),
        70,
        thin=1,
        statistics={"rss": residual_sum, "slopes": slopes},
    )
    assert result.statistic_names == ("rss", "slopes[0]", "slopes[1]")
    p_values = sbc_uniformity(
        result.ranks, result.num_draws, num_reference=NUM_REFERENCE
    )
    assert bool((p_values > ALPHA).all()), p_values


def test_sbc_validates_its_arguments() -> None:
    fit = _gaussian_fit()
    with pytest.raises(ValueError, match="rows"):
        sbc(
            MODEL,
            torch.zeros(2, 1),
            fit,
            observed_names={"y"},
            num_simulations=2,
            seed=0,
            data=DATA,
        )
    with pytest.raises(ValueError, match="not sample sites"):
        sbc(MODEL, X, fit, observed_names={"z"}, num_simulations=2, seed=0, data=DATA)
    with pytest.raises(ValueError, match="thin"):
        _run(fit, 0, num_simulations=2, thin=0)
    with pytest.raises(TypeError, match="MCMCResult, a Guide"):
        _run(lambda observations, seed: [1.0], 0, num_simulations=2)

    def broken(
        observations: dict[str, torch.Tensor], seed: int
    ) -> dict[str, torch.Tensor]:
        draws = fit(observations, seed)
        draws["mu"][0, 3] = float("nan")
        return draws

    with pytest.raises(ValueError, match="non-finite"):
        _run(broken, 0, num_simulations=2, thin=1)


def test_datatree_holds_midpoint_pit_values_for_arviz() -> None:
    result = _run(_gaussian_fit(), 80, num_simulations=30, thin=1)
    tree = to_datatree_from_sbc(result)
    group = tree["prior_sbc"]
    assert set(group.data_vars) == set(result.statistic_names)
    values = group["beta[1]"].values
    assert values.shape == (1, 30)
    expected = (result.ranks[:, 2].double().numpy() + 0.5) / (result.num_draws + 1)
    np.testing.assert_allclose(values[0], expected, rtol=0.0, atol=0.0)
    assert ((values > 0.0) & (values < 1.0)).all()
    azp.plot_ecdf_pit(tree, backend="none")
