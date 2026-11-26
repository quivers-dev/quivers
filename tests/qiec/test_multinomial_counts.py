"""Scalar and vector count kernels share normalized high-precision factors."""

import math
import random

import numpy as np
import pytest
from scipy import stats

from quivers.qiec._count_mixtures import _beta_binomial_log_mass
from quivers.qiec._multinomial import (
    multinomial_density,
    dirichlet_multinomial_density,
    multinomial_sample,
    dirichlet_multinomial_sample,
)
from quivers.qiec._sampling import _binomial_sample, _dirichlet_weights
from quivers.qiec.distributions import _dirichlet_sample
from quivers.qiec.reference_families import _betabinomial_sample, _betabinomial_density
from quivers.qiec.reference_support import DistributionError


@pytest.mark.parametrize(
    "n,point", [(0, [0, 0, 0]), (7, [1, 4, 2]), (100, [20, 30, 50])]
)
def test_vector_reference_scores_independent_scipy(n, point):
    p = [0.2, 0.3, 0.5]
    a = [0.7, 1.3, 2.0]
    assert multinomial_density({"total_count": n, "probs": p}, point) == pytest.approx(
        stats.multinomial.logpmf(point, n, p), abs=2e-12
    )
    expected = stats.dirichlet_multinomial.logpmf(point, a, n) if n else 0.0
    assert dirichlet_multinomial_density(
        {"total_count": n, "concentration": a}, point
    ) == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize(
    "n,k,expected",
    [(10, 5, -1.40204271808803003744), (100, 50, -2.53087640397710743272)],
)
def test_scalar_and_vector_beta_factor_share_90_digit_oracle(n, k, expected):
    a = {"total_count": n, "concentration1": 1e16, "concentration0": 1e16}
    assert _beta_binomial_log_mass(n, k, 1e16, 1e16) == pytest.approx(
        expected, abs=3e-12
    )
    assert _betabinomial_density(a, k) == pytest.approx(expected, abs=3e-12)
    assert dirichlet_multinomial_density(
        {"total_count": n, "concentration": [1e16, 1e16]}, [k, n - k]
    ) == pytest.approx(expected, abs=3e-12)


def test_logits_zero_categories_and_large_finite_penalties():
    assert multinomial_density(
        {"total_count": 3, "logits": [0, -1000, -math.inf]}, [2, 1, 0]
    ) == pytest.approx(math.log(3) - 1000)
    assert (
        multinomial_density({"total_count": 3, "probs": [1, 0, 0]}, [2, 1, 0])
        == -math.inf
    )
    assert multinomial_density({"total_count": 3, "probs": [1, 0, 0]}, [3, 0, 0]) == 0


@pytest.mark.parametrize(
    "density", [multinomial_density, dirichlet_multinomial_density]
)
def test_event_support_and_exact_capacity(density):
    a = {"total_count": 3, "probs": [0.2, 0.8], "concentration": [2, 3]}
    for point in [[1, 1], [-1, 4], [0.5, 2.5], [math.nan, 0]]:
        assert density(a, point) == -math.inf
    with pytest.raises(DistributionError, match="categories"):
        density(a, [3])
    with pytest.raises(DistributionError, match="capacity"):
        density({**a, "total_count": 2**53 + 1}, [1, 2])
    with pytest.raises(DistributionError, match="capacity"):
        density(a, [2**53 + 1, 0])


@pytest.mark.parametrize(
    "sampler,parameters",
    [
        (multinomial_sample, {"probs": [0.2, 0.3, 0.5]}),
        (dirichlet_multinomial_sample, {"concentration": [2, 3, 5]}),
    ],
)
def test_reference_sampling_moments_and_exact_totals(sampler, parameters):
    rng = random.Random(319)
    values = np.array(
        [sampler({"total_count": 9, **parameters}, rng) for _ in range(15000)]
    )
    assert np.all(values.sum(-1) == 9)
    np.testing.assert_allclose(values.mean(0), [1.8, 2.7, 4.5], atol=0.06)
    expected = 9 * np.array([0.2, 0.3, 0.5]) * np.array([0.8, 0.7, 0.5])
    if sampler is dirichlet_multinomial_sample:
        expected *= 19 / 11
    np.testing.assert_allclose(values.var(0), expected, rtol=0.04, atol=0.03)


@pytest.mark.parametrize("alpha", [1e-6, 1e-100, 1e-320])
def test_shared_dirichlet_sampler_preserves_tiny_shape_vertex_law(alpha):
    a = [alpha, 2 * alpha, 3 * alpha]
    rng = random.Random(52)
    values = np.array(
        [_dirichlet_sample({"concentration": a}, rng) for _ in range(12000)]
    )
    assert np.mean(values.max(-1) > 0.9999) > 0.999
    np.testing.assert_allclose(values.mean(0), [1 / 6, 2 / 6, 3 / 6], atol=0.015)
    rng = random.Random(52)
    counts = np.array(
        [
            _betabinomial_sample(
                {
                    "total_count": 10,
                    "concentration1": alpha,
                    "concentration0": 2 * alpha,
                },
                rng,
            )
            for _ in range(12000)
        ]
    )
    assert np.mean((counts == 0) | (counts == 10)) > 0.999
    assert counts.mean() == pytest.approx(10 / 3, abs=0.13)


def test_degenerate_sampling_preserves_generator_state_and_skips_gamma():
    class NoDraws(random.Random):
        def random(self):
            raise AssertionError("a degenerate draw must not consume randomness")

    rng = NoDraws()
    assert _binomial_sample(0, 0.5, rng) == 0
    assert _binomial_sample(10, 0, rng) == 0
    assert _binomial_sample(10, 1, rng) == 10
    assert dirichlet_multinomial_sample(
        {"total_count": 0, "concentration": [1e-320, 1e-320]}, rng
    ) == (0, 0)
    assert (
        _betabinomial_sample(
            {"total_count": 0, "concentration1": 1e-320, "concentration0": 1e-320}, rng
        )
        == 0
    )


def test_invalid_host_sampling_results_are_rejected():
    class BadGamma(random.Random):
        def gammavariate(self, alpha, beta):
            return 0.0

    with pytest.raises(DistributionError, match="Gamma sampling"):
        _dirichlet_weights([1.0, 2.0], BadGamma())

    class BadCount(random.Random):
        def binomialvariate(self, n, p):
            return n + 1

    with pytest.raises(DistributionError, match="exceeded remaining trials"):
        _binomial_sample(4, 0.3, BadCount())


@pytest.mark.parametrize(
    "family,parameters",
    [
        ("Binomial", {"probs": 0.3}),
        ("BetaBinomial", {"concentration1": 2.0, "concentration0": 3.0}),
    ],
)
def test_existing_scalar_reference_trials_and_events_are_not_truncated(
    family, parameters
):
    from quivers.qiec.distributions import ReferenceBackend

    backend = ReferenceBackend()
    assert backend.family_log_prob(family, {"total_count": 3, **parameters}, 1.5) == -math.inf
    for bad in [1.5, -1, True, 2**53 + 1]:
        with pytest.raises(DistributionError, match="trials|capacity"):
            backend.family_sample(family, {"total_count": bad, **parameters}, random.Random(1))
        with pytest.raises(DistributionError, match="trials|capacity"):
            backend.family_log_prob(family, {"total_count": bad, **parameters}, 1)


@pytest.mark.parametrize(
    "a,b",
    [(0.0, 1.0), (-1.0, 1.0), (float("inf"), 1.0), (float("nan"), 1.0), (1e308, 1e308)],
)
def test_zero_trial_beta_binomial_validates_concentrations_before_skipping_rng(a, b):
    from quivers.qiec.distributions import ReferenceBackend
    from quivers.continuous._zero_inflated_count import BetaBinomial

    class NoDraws(random.Random):
        def random(self):
            raise AssertionError("invalid parameters must fail before sampling")

    with pytest.raises(ValueError):
        BetaBinomial(0, a, b)
    with pytest.raises(DistributionError, match="concentrations"):
        ReferenceBackend().family_sample(
            "BetaBinomial",
            {"total_count": 0, "concentration1": a, "concentration0": b},
            NoDraws(),
        )
