"""Tests for the convergence estimators of `quivers.diagnostics`.

The oracle is ArviZ (``arviz.rhat``, ``arviz.ess``, ``arviz.mcse``),
evaluated on fixed seeded chains. The quivers estimators run the same
float64 algorithm, so on finite, non-constant input the two differ
only by floating-point summation order and FFT padding length, both
of order 1e-15 relative. The tolerance is fixed at ``rel=1e-10``: wide
enough for that rounding, and many orders of magnitude below the
differences any change of definition produces (the variant tests at
the end show those are at least 1e-3 relative). Order-statistic
results (quantile MCSE) select the same draws and are compared at the
same tolerance.

Degenerate inputs follow the module's documented contract, checked
against ArviZ where the two agree and pinned as a deliberate
departure where they do not (constant coordinates).
"""

from __future__ import annotations

import math

import arviz as az
import numpy as np
import pytest
import torch
from scipy import stats

from quivers.diagnostics import ConvergenceSummary, ess, mcse, rhat, summary
from quivers.diagnostics.convergence import _beta_ppf
from quivers.inference import MCMCResult

REL = 1e-10


def _ar1(seed: int, phi: float, shape: tuple[int, int]) -> np.ndarray:
    generator = np.random.default_rng(seed)
    noise = generator.normal(size=shape)
    chains = np.zeros(shape)
    chains[:, 0] = noise[:, 0] / math.sqrt(1.0 - phi**2)
    for t in range(1, shape[1]):
        chains[:, t] = phi * chains[:, t - 1] + noise[:, t]
    return chains


def _cases() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(20210601)
    locations = generator.normal(size=(4, 500))
    locations[2:] += 2.0
    scales = generator.normal(size=(4, 500))
    scales[2:] *= 4.0
    return {
        "well_mixed": generator.normal(size=(4, 500)),
        "well_mixed_odd_draws": generator.normal(size=(4, 201)),
        "autocorrelated": _ar1(11, 0.95, (4, 1000)),
        "separated_locations": locations,
        "separated_scales": scales,
        "heavy_tailed": generator.standard_cauchy(size=(4, 300)),
        "tied_counts": generator.poisson(2.0, size=(4, 300)).astype(float),
        "short": generator.normal(size=(4, 4)),
        "short_odd": generator.normal(size=(3, 5)),
    }


CASES = _cases()


def _tensor(draws: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(draws, dtype=torch.float64)


@pytest.mark.parametrize("name", sorted(CASES))
def test_rhat_matches_arviz(name: str) -> None:
    draws = CASES[name]
    expected = float(az.rhat(draws, method="rank"))
    assert float(rhat(_tensor(draws))) == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("method", ["bulk", "mean", "sd"])
def test_ess_matches_arviz(name: str, method: str) -> None:
    draws = CASES[name]
    expected = float(az.ess(draws, method=method))
    assert float(ess(_tensor(draws), method=method)) == pytest.approx(  # type: ignore[arg-type]
        expected, rel=REL
    )


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("prob", [0.05, 0.2])
def test_tail_ess_matches_arviz(name: str, prob: float) -> None:
    draws = CASES[name]
    expected = float(az.ess(draws, method="tail", prob=(prob, 1.0 - prob)))
    actual = ess(_tensor(draws), method="tail", prob=prob)
    assert float(actual) == pytest.approx(expected, rel=REL)


def test_tail_ess_defaults_to_the_five_and_ninety_five_percent_quantiles() -> None:
    draws = CASES["autocorrelated"]
    expected = float(az.ess(draws, method="tail", prob=(0.05, 0.95)))
    assert float(ess(_tensor(draws), method="tail")) == pytest.approx(expected, rel=REL)
    assert float(ess(_tensor(draws), method="tail", prob=0.95)) == pytest.approx(
        expected, rel=REL
    )


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("prob", [0.05, 0.3, 0.5, 0.95])
def test_quantile_ess_matches_arviz(name: str, prob: float) -> None:
    draws = CASES[name]
    expected = float(az.ess(draws, method="quantile", prob=prob))
    actual = ess(_tensor(draws), method="quantile", prob=prob)
    assert float(actual) == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("method", ["mean", "sd"])
def test_mcse_matches_arviz(name: str, method: str) -> None:
    draws = CASES[name]
    expected = float(az.mcse(draws, method=method))
    assert float(mcse(_tensor(draws), method=method)) == pytest.approx(  # type: ignore[arg-type]
        expected, rel=REL
    )


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("prob", [0.05, 0.5, 0.9])
def test_quantile_mcse_matches_arviz(name: str, prob: float) -> None:
    draws = CASES[name]
    expected = float(az.mcse(draws, method="quantile", prob=prob))
    actual = mcse(_tensor(draws), method="quantile", prob=prob)
    assert float(actual) == pytest.approx(expected, rel=REL)


def test_single_chain_ess_and_mcse_match_arviz_and_rhat_is_nan() -> None:
    draws = _ar1(5, 0.7, (1, 400))
    tensor = _tensor(draws)
    assert math.isnan(float(rhat(tensor)))
    assert math.isnan(float(az.rhat(draws, method="rank")))
    assert float(ess(tensor)) == pytest.approx(float(az.ess(draws)), rel=REL)
    assert float(mcse(tensor)) == pytest.approx(float(az.mcse(draws)), rel=REL)


def test_event_coordinates_are_diagnosed_independently() -> None:
    generator = np.random.default_rng(3)
    draws = generator.normal(size=(4, 200, 2, 3))
    draws[..., 1, 2] = _ar1(4, 0.9, (4, 200))
    tensor = _tensor(draws)
    r_hat = rhat(tensor)
    bulk = ess(tensor)
    tail = ess(tensor, method="tail")
    assert r_hat.shape == bulk.shape == tail.shape == (2, 3)
    assert r_hat.dtype == torch.float64
    for i in range(2):
        for j in range(3):
            coordinate = draws[..., i, j]
            assert float(r_hat[i, j]) == pytest.approx(
                float(az.rhat(coordinate, method="rank")), rel=REL
            )
            assert float(bulk[i, j]) == pytest.approx(
                float(az.ess(coordinate, method="bulk")), rel=REL
            )
            assert float(tail[i, j]) == pytest.approx(
                float(az.ess(coordinate, method="tail", prob=(0.05, 0.95))), rel=REL
            )


def test_scalar_event_returns_a_zero_dimensional_tensor() -> None:
    result = rhat(_tensor(CASES["well_mixed"]))
    assert result.shape == ()


# Degenerate inputs.


def test_too_few_draws_return_nan() -> None:
    draws = _tensor(np.random.default_rng(1).normal(size=(4, 3)))
    assert math.isnan(float(az.rhat(draws.numpy(), method="rank")))
    assert math.isnan(float(az.ess(draws.numpy(), method="bulk")))
    assert math.isnan(float(rhat(draws)))
    for method in ("bulk", "tail", "mean", "sd"):
        assert math.isnan(float(ess(draws, method=method)))  # type: ignore[arg-type]
    for method in ("mean", "sd"):
        assert math.isnan(float(mcse(draws, method=method)))  # type: ignore[arg-type]
    assert math.isnan(float(mcse(draws, method="quantile", prob=0.5)))


def test_nan_draw_returns_nan_for_its_coordinate_only() -> None:
    generator = np.random.default_rng(2)
    draws = generator.normal(size=(4, 100, 2))
    draws[1, 17, 0] = np.nan
    tensor = _tensor(draws)
    assert math.isnan(float(az.rhat(draws[..., 0], method="rank")))
    assert math.isnan(float(az.ess(draws[..., 0], method="bulk")))
    assert math.isnan(float(az.mcse(draws[..., 0], method="mean")))
    for result in (
        rhat(tensor),
        ess(tensor),
        ess(tensor, method="tail"),
        mcse(tensor),
        mcse(tensor, method="sd"),
        mcse(tensor, method="quantile", prob=0.5),
    ):
        assert math.isnan(float(result[0]))
        assert math.isfinite(float(result[1]))
    assert float(rhat(tensor)[1]) == pytest.approx(
        float(az.rhat(draws[..., 1], method="rank")), rel=REL
    )


@pytest.mark.parametrize("bad", [math.inf, -math.inf])
def test_infinite_draw_returns_nan(bad: float) -> None:
    draws = np.random.default_rng(6).normal(size=(4, 100))
    draws[2, 40] = bad
    tensor = _tensor(draws)
    assert math.isnan(float(rhat(tensor)))
    assert math.isnan(float(ess(tensor)))
    assert math.isnan(float(mcse(tensor)))


def test_constant_coordinate_returns_nan_where_arviz_reports_full_ess() -> None:
    draws = np.full((4, 100), 0.25)
    tensor = _tensor(draws)
    # The deliberate departure: ArviZ reports the total draw count as
    # the ESS of a constant chain, which reads as perfect mixing for a
    # sampler stuck at one value.
    assert float(az.ess(draws, method="bulk")) == draws.size
    with np.errstate(invalid="ignore"):
        assert math.isnan(float(az.rhat(draws, method="rank")))
    assert math.isnan(float(rhat(tensor)))
    for method in ("bulk", "tail", "mean", "sd"):
        assert math.isnan(float(ess(tensor, method=method)))  # type: ignore[arg-type]
    for method in ("mean", "sd"):
        assert math.isnan(float(mcse(tensor, method=method)))  # type: ignore[arg-type]
    assert math.isnan(float(mcse(tensor, method="quantile", prob=0.5)))


def test_chains_stuck_at_different_values_have_infinite_rhat() -> None:
    draws = np.repeat(np.array([[0.0], [1.0], [2.0], [3.0]]), 50, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_rhat = float(az.rhat(draws, method="rank"))
        expected_ess = float(az.ess(draws, method="bulk"))
    assert expected_rhat == math.inf
    assert float(rhat(_tensor(draws))) == math.inf
    assert float(ess(_tensor(draws))) == pytest.approx(expected_ess, rel=REL)


def test_symmetric_two_point_draws_fall_back_on_the_bulk_statistic() -> None:
    # |x - median| is constant, so the folded statistic is undefined;
    # ArviZ and quivers both report the bulk statistic.
    draws = np.random.default_rng(3).choice([-1.0, 1.0], size=(4, 100))
    expected = float(az.rhat(draws, method="rank"))
    assert expected == pytest.approx(float(az.rhat(draws, method="z_scale")), rel=REL)
    assert float(rhat(_tensor(draws))) == pytest.approx(expected, rel=REL)


# Argument validation.


def test_draws_need_chain_and_draw_axes() -> None:
    with pytest.raises(ValueError, match="num_chains, num_draws"):
        rhat(torch.zeros(10))
    with pytest.raises(ValueError, match="at least one chain"):
        ess(torch.zeros(0, 10))


def test_complex_draws_are_rejected() -> None:
    with pytest.raises(TypeError, match="real-valued"):
        rhat(torch.zeros(2, 10, dtype=torch.complex64))


def test_method_and_prob_are_validated() -> None:
    draws = _tensor(CASES["well_mixed"])
    with pytest.raises(ValueError, match="unknown ESS method"):
        ess(draws, method="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown MCSE method"):
        mcse(draws, method="bulk")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires prob"):
        ess(draws, method="quantile")
    with pytest.raises(ValueError, match="requires prob"):
        mcse(draws, method="quantile")
    with pytest.raises(ValueError, match="takes no prob"):
        ess(draws, method="bulk", prob=0.5)
    with pytest.raises(ValueError, match="takes no prob"):
        mcse(draws, method="sd", prob=0.5)
    with pytest.raises(ValueError, match="strictly between"):
        ess(draws, method="tail", prob=0.0)
    with pytest.raises(ValueError, match="strictly between"):
        summary({"x": draws}, probs=(0.5, 1.0))


# Variants that drop part of the definition disagree with the oracle.


def _classic_rhat(chains: np.ndarray, *, split: bool, rank: bool, fold: bool) -> float:
    """Gelman-Rubin R-hat with optional splitting, ranking, and folding."""
    if split:
        half = chains.shape[1] // 2
        chains = np.vstack([chains[:, :half], chains[:, -half:]])

    def transform(values: np.ndarray) -> np.ndarray:
        if not rank:
            return values
        ranks = stats.rankdata(values, method="average").reshape(values.shape)
        return stats.norm.ppf((ranks - 3 / 8) / (values.size + 1 / 4))

    def potential_scale_reduction(values: np.ndarray) -> float:
        n = values.shape[1]
        between = n * values.mean(axis=1).var(ddof=1)
        within = values.var(axis=1, ddof=1).mean()
        return float(np.sqrt((between / within + n - 1) / n))

    bulk = potential_scale_reduction(transform(chains))
    if not fold:
        return bulk
    folded = np.abs(chains - np.median(chains))
    return max(bulk, potential_scale_reduction(transform(folded)))


def test_non_split_rhat_misses_a_shared_trend_that_split_rhat_detects() -> None:
    # Every chain drifts the same way, so the full chains agree with
    # each other while their halves do not.
    generator = np.random.default_rng(12)
    trend = np.linspace(-2.0, 2.0, 400)
    draws = trend + generator.normal(scale=0.5, size=(4, 400))
    oracle = float(az.rhat(draws, method="rank"))
    non_split = _classic_rhat(draws, split=False, rank=True, fold=True)
    assert float(rhat(_tensor(draws))) == pytest.approx(oracle, rel=REL)
    assert _classic_rhat(draws, split=True, rank=True, fold=True) == pytest.approx(
        oracle, rel=REL
    )
    assert oracle > 1.5
    assert non_split < 1.01


def test_non_rank_normalized_rhat_disagrees_on_heavy_tails() -> None:
    # One Cauchy chain's extreme draws inflate its raw variance, which
    # masks the location shift of another chain from the classic
    # statistic but not from the rank-normalized one.
    generator = np.random.default_rng(13)
    draws = generator.standard_cauchy(size=(4, 500))
    draws[3] += 1.5
    oracle = float(az.rhat(draws, method="rank"))
    raw = _classic_rhat(draws, split=True, rank=False, fold=False)
    assert float(rhat(_tensor(draws))) == pytest.approx(oracle, rel=REL)
    assert _classic_rhat(draws, split=True, rank=True, fold=True) == pytest.approx(
        oracle, rel=REL
    )
    assert abs(raw - oracle) / oracle > 1e-3
    assert oracle > 1.01 > raw


def test_unfolded_rhat_misses_a_scale_difference() -> None:
    generator = np.random.default_rng(14)
    draws = generator.normal(size=(4, 500))
    draws[2:] *= 4.0
    oracle = float(az.rhat(draws, method="rank"))
    unfolded = _classic_rhat(draws, split=True, rank=True, fold=False)
    assert float(rhat(_tensor(draws))) == pytest.approx(oracle, rel=REL)
    assert oracle > 1.1
    assert unfolded < 1.01


def test_non_rank_normalized_ess_disagrees_with_bulk_ess_on_heavy_tails() -> None:
    draws = np.random.default_rng(15).standard_cauchy(size=(4, 500))
    oracle = float(az.ess(draws, method="bulk"))
    unnormalized = float(az.ess(draws, method="mean"))
    assert float(ess(_tensor(draws))) == pytest.approx(oracle, rel=REL)
    assert float(ess(_tensor(draws), method="mean")) == pytest.approx(
        unnormalized, rel=REL
    )
    assert abs(unnormalized - oracle) / oracle > 1e-3


# Internals and composites.


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [(1.0, 1.0), (1.5, 3.0), (50.0, 950.0), (2000.5, 2000.5), (3.0, 40000.0)],
)
@pytest.mark.parametrize("level", [0.1586553, 0.8413447, 0.001, 0.999])
def test_beta_quantile_matches_scipy(alpha: float, beta: float, level: float) -> None:
    expected = float(stats.beta.ppf(level, alpha, beta))
    assert _beta_ppf(level, alpha, beta) == pytest.approx(expected, rel=1e-12)


def test_summary_collects_every_estimator() -> None:
    generator = np.random.default_rng(16)
    draws = generator.normal(size=(4, 300, 2))
    draws[..., 1] = _ar1(17, 0.8, (4, 300))
    tensor = _tensor(draws)
    result = summary({"theta": tensor}, probs=(0.1, 0.5))
    site = result["theta"]
    assert isinstance(site, ConvergenceSummary)
    assert site.probs == (0.1, 0.5)
    torch.testing.assert_close(site.r_hat, rhat(tensor), rtol=0.0, atol=0.0)
    torch.testing.assert_close(site.ess_bulk, ess(tensor), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        site.ess_tail, ess(tensor, method="tail"), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(site.mcse_mean, mcse(tensor), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        site.mcse_sd, mcse(tensor, method="sd"), rtol=0.0, atol=0.0
    )
    pooled = draws.reshape(-1, 2)
    np.testing.assert_allclose(site.mean.numpy(), pooled.mean(axis=0), rtol=REL)
    np.testing.assert_allclose(site.sd.numpy(), pooled.std(axis=0, ddof=1), rtol=REL)
    assert site.quantiles.shape == site.mcse_quantiles.shape == (2, 2)
    for row, prob in enumerate((0.1, 0.5)):
        np.testing.assert_allclose(
            site.quantiles[row].numpy(), np.quantile(pooled, prob, axis=0), rtol=REL
        )
        for coordinate in range(2):
            assert float(site.mcse_quantiles[row, coordinate]) == pytest.approx(
                float(az.mcse(draws[..., coordinate], method="quantile", prob=prob)),
                rel=REL,
            )


def test_mcmc_result_reports_through_the_public_estimators() -> None:
    generator = np.random.default_rng(18)
    samples = {
        "mu": _tensor(_ar1(19, 0.6, (3, 200))),
        "beta": _tensor(generator.normal(size=(3, 200, 4))),
    }
    result = MCMCResult(
        samples=samples,
        log_densities=torch.zeros(3, 200),
        acceptance_rates=torch.ones(3),
        divergence_counts=torch.zeros(3, dtype=torch.int64),
        num_warmup=0,
        num_samples=200,
    )
    for name, draws in samples.items():
        torch.testing.assert_close(result.r_hat[name], rhat(draws), rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            result.ess[name], ess(draws, method="bulk"), rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            result.ess_tail[name], ess(draws, method="tail"), rtol=0.0, atol=0.0
        )
    assert result.r_hat["beta"].shape == (4,)
