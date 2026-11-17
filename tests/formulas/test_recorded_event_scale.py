"""Censored and truncated scoring away from the unit data scale.

Two independent oracles are used. The first is scale equivariance: for a
response ``Y`` with density ``f`` and a constant ``c > 0``, the law of
``c Y`` has density ``f(y / c) / c``, so rescaling the data together with
the matching parameters must lower every exact-row log density by exactly
``log c`` and leave every censored-row log probability and every truncation
normalizer unchanged. The constant is a power of two, so the rescaled
float32 responses are exact. The second oracle scores mismatched scales,
where the model's parameters sit at the unit scale and the data do not, as
at a sampler's initial position: SciPy's ``logsf`` and ``logcdf`` give the
far-tail log probabilities directly.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import special, stats

from quivers.formulas import fit
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry

STATUS = np.array([0, 1, -1, 2, 0, -1, 1, 2])
# Responses are carried in float32 and pinned draws in float64. One float32
# rounding of a standardized response of order ten moves a log density by at
# most about 1e-5 relative, which fixes the tolerance.
RTOL = 2e-5
ATOL = 2e-5

LOG_LOCATION = {"intercept": "shift"}
LOCATION_SCALE = {"intercept": "scale", "beta_x": "scale", "sigma": "scale"}
RESCALING: dict[str, dict[str, str]] = {
    "weibull": LOG_LOCATION,
    "gamma": LOG_LOCATION,
    "exponential": LOG_LOCATION,
    "frechet": LOG_LOCATION,
    "lognormal": LOG_LOCATION,
    "hurdle_gamma": LOG_LOCATION,
    "hurdle_lognormal": LOG_LOCATION,
    "inverse_gaussian": {**LOG_LOCATION, "shape": "scale"},
    "shifted_lognormal": {**LOG_LOCATION, "ndt": "scale"},
    "gaussian": LOCATION_SCALE,
    "student_t": LOCATION_SCALE,
    "skew_normal": LOCATION_SCALE,
    "asym_laplace": LOCATION_SCALE,
    "zero_inflated_asym_laplace": LOCATION_SCALE,
    "gen_extreme_value": LOCATION_SCALE,
    "exgaussian": {**LOCATION_SCALE, "beta": "scale"},
}
"""How each latent site moves when the response is multiplied by ``c``:
``shift`` adds ``log c`` (a log-link location) and ``scale`` multiplies by
``c``. Sites not listed are shape parameters and stay fixed."""

POSITIVE = {
    "weibull",
    "gamma",
    "exponential",
    "frechet",
    "lognormal",
    "hurdle_gamma",
    "hurdle_lognormal",
    "inverse_gaussian",
    "shifted_lognormal",
}


def _frame(name: str, factor: float) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    n = STATUS.size
    if name in POSITIVE:
        y = np.clip(rng.gamma(2.0, 1.0, size=n) + 0.25, 0.25, 12.0)
        upper = y + 1.0
    else:
        y = np.clip(rng.normal(size=n), -2.5, 2.5)
        upper = y + 0.5
    y, upper = (np.asarray(v, np.float32).astype(np.float64) for v in (y, upper))
    return pd.DataFrame(
        {
            "y": y * factor,
            "up": upper * factor,
            "x": rng.normal(size=n),
            "st": STATUS,
        }
    )


def _formula(name: str, truncated: bool, factor: float) -> str:
    terms = ["cens(st, up)"]
    if truncated:
        low, high = (0.125, 16.0) if name in POSITIVE else (-4.0, 4.0)
        terms.append(f"trunc(lb={low * factor!r}, ub={high * factor!r})")
    return f"y | {' + '.join(terms)} ~ x"


def _with_draws(fitted, samples: dict[str, torch.Tensor]):
    count = next(iter(samples.values())).shape[1]
    posterior = MCMCResult(
        samples=samples,
        log_densities=torch.zeros(1, count),
        acceptance_rates=torch.ones(1),
        divergence_counts=torch.zeros(1, dtype=torch.long),
        num_warmup=0,
        num_samples=count,
    )
    return fitted.with_(posterior=posterior)


def _latent_draws(fitted, count: int = 3) -> dict[str, torch.Tensor]:
    latent = LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    ).names
    return {
        site: value[:, :count].to(torch.float64)
        for site, value in fitted.draws().items()
        if site in latent
    }


def _log_likelihood(fitted) -> torch.Tensor:
    return next(iter(fitted.log_likelihood().values())).to(torch.float64)


@pytest.mark.parametrize("exponent", [-14, 14])
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("name", sorted(RESCALING))
def test_rescaling_shifts_the_log_density_by_the_jacobian(
    name: str, truncated: bool, exponent: int
) -> None:
    factor = 2.0**exponent
    unit = fit(
        _formula(name, truncated, 1.0),
        data=_frame(name, 1.0),
        family=name,
        method="svi",
        num_steps=1,
        seed=0,
    )
    draws = _latent_draws(unit)
    assert set(RESCALING[name]) <= set(draws)
    baseline = _log_likelihood(_with_draws(unit, draws))
    assert bool(torch.isfinite(baseline).all())

    moved = dict(draws)
    for site, rule in RESCALING[name].items():
        moved[site] = (
            draws[site] + math.log(factor) if rule == "shift" else draws[site] * factor
        )
    # No optimization step: the guide's unit-scale start need not lie in the
    # support the rescaled data leave a bounded-support law, and only the
    # compiled model is needed here.
    scaled = fit(
        _formula(name, truncated, factor),
        data=_frame(name, factor),
        family=name,
        method="svi",
        num_steps=0,
        seed=0,
    )
    observed = _log_likelihood(_with_draws(scaled, moved))
    exact = torch.as_tensor(STATUS == 0)
    expected = baseline - math.log(factor) * exact.to(torch.float64)
    torch.testing.assert_close(observed, expected, rtol=RTOL, atol=ATOL)


def _pinned_intercept_only(name: str, frame: pd.DataFrame, sites: dict[str, float]):
    fitted = fit(
        "y | cens(st) ~ 1", data=frame, family=name, method="svi", num_steps=1, seed=0
    )
    draws = _latent_draws(fitted, 1)
    assert set(draws) == set(sites)
    pinned = {
        site: torch.full_like(draws[site], value) for site, value in sites.items()
    }
    return _log_likelihood(_with_draws(fitted, pinned)).reshape(-1)


MISMATCHED = [2.0**-20, 2.0**-6, 20.0, 200.0, 2.0**14]


@pytest.mark.parametrize("factor", MISMATCHED)
def test_weibull_tails_at_a_mismatched_scale_match_scipy(factor: float) -> None:
    shape = 1.5
    base = np.asarray([0.5, 1.0, 2.0, 3.0], np.float32).astype(np.float64) * np.float32(
        factor
    ).astype(np.float64)
    frame = pd.DataFrame({"y": np.tile(base, 3), "st": np.repeat([0, 1, -1], 4)})
    observed = _pinned_intercept_only(
        "weibull", frame, {"intercept": 0.0, "shape": shape}
    )
    # brms predicts the Weibull mean: scale = mean / Gamma(1 + 1 / shape).
    law = stats.weibull_min(shape, scale=1.0 / special.gamma(1.0 + 1.0 / shape))
    expected = np.concatenate((law.logpdf(base), law.logsf(base), law.logcdf(base)))
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(observed.numpy(), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("factor", MISMATCHED)
def test_lognormal_tails_at_a_mismatched_scale_match_scipy(factor: float) -> None:
    sigma = 0.25
    base = np.asarray([0.5, 1.0, 2.0, 3.0], np.float32).astype(np.float64) * np.float32(
        factor
    ).astype(np.float64)
    frame = pd.DataFrame({"y": np.tile(base, 3), "st": np.repeat([0, 1, -1], 4)})
    observed = _pinned_intercept_only(
        "lognormal", frame, {"intercept": 0.0, "sigma": sigma}
    )
    standardized = np.log(base) / sigma
    expected = np.concatenate(
        (
            stats.norm.logpdf(standardized) - np.log(sigma * base),
            stats.norm.logsf(standardized),
            stats.norm.logcdf(standardized),
        )
    )
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(observed.numpy(), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("family", ["weibull", "lognormal"])
@pytest.mark.parametrize("factor", [20.0, 200.0])
@pytest.mark.parametrize("method", ["nuts", "hmc", "svi"])
def test_right_censored_fits_run_at_a_large_data_scale(
    family: str, factor: float, method: str
) -> None:
    rng = np.random.default_rng(9)
    n = 36
    frame = pd.DataFrame(
        {
            "time": rng.lognormal(0.0, 1.0, n) * factor,
            "c": (np.arange(n) % 5 == 0).astype(int),
        }
    )
    options = (
        {"method": "svi", "num_steps": 3}
        if method == "svi"
        else {"method": method, "num_warmup": 6, "num_samples": 6, "num_chains": 1}
    )
    fitted = fit("time | cens(c) ~ 1", data=frame, family=family, seed=3, **options)
    values = _log_likelihood(fitted)
    assert values.shape[-1] == n and bool(torch.isfinite(values).all())


def test_lognormal_with_a_raw_scale_random_slope_starts_at_the_default_init() -> None:
    rng = np.random.default_rng(9)
    n = 36
    rows = np.arange(n)
    frame = pd.DataFrame(
        {
            "patient": np.repeat(np.arange(n // 2), 2).astype(str),
            "age": rng.normal(45.0, 10.0, n),
            "sex": np.tile(["male", "female"], n // 2),
            "disease": np.array(["other", "GN", "AN", "PKD"])[rows % 4],
            "censored": (rows % 5 == 0).astype(int),
        }
    )
    frame["time"] = rng.lognormal(3.0, 1.0, n)
    for seed in range(3):
        fitted = fit(
            "time | cens(censored) ~ age * sex + disease + (1 + age | patient)",
            data=frame,
            family="lognormal",
            num_warmup=4,
            num_samples=4,
            num_chains=1,
            control={"max_treedepth": 3},
            seed=seed,
        )
        assert bool(torch.isfinite(_log_likelihood(fitted)).all())
