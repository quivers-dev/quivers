"""Bridge sampling, Bayes factors and posterior model probabilities of fits.

Oracles: the R bridgesampling, coda and Matrix functions brms calls
(``oracles/bridge_numerics.R``), closed-form marginal likelihoods of
conjugate Gaussian models, one-dimensional quadrature over a latent scale
with the remaining coefficients integrated analytically, and the existing
SMC evidence. Posterior draws of the oracle models are exact draws, so the
only error is the estimator's Monte Carlo error. Tolerances are fixed before
looking at results: 1e-9 relative for deterministic numerics, four
coefficients of variation (the estimator's own relative-error estimate) for
the normal method, and 0.01 on the log scale for warp3 with 2000 exact draws
of a posterior of dimension at most five.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import integrate, stats

from quivers.diagnostics.bridge import (
    BridgeRelativeError,
    BridgeRepetitionSpread,
    BridgeResult,
    _iterate,
    _solve,
    ar_spectrum0,
    bridge_sampler as bridge_estimate,
    effective_size,
    near_pd,
    posterior_model_probabilities,
)
from quivers.formulas import PriorSpec, bf, fit, set_rescor
from quivers.formulas._evidence import (
    BayesFactor,
    bayes_factor,
    log_posterior_function,
    post_prob,
)
from quivers.inference import SMC
from quivers.inference.mcmc import MetropolisKernel, RandomWalkProposal
from quivers.inference.mcmc.driver import MCMCResult

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "bridge_numerics.json").read_text()
)

Y = np.array([0.42, -0.31, 1.18, 0.77, -0.05, 0.93, 1.41, 0.12, 0.58, -0.62])
X = np.array([0.5, -1.1, 0.3, 1.7, -0.2, 0.9, 1.4, -0.6, 1.2, -0.8])
G = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"])
NORMAL01 = PriorSpec(distribution="Normal(0.0, 1.0)", class_="Intercept")


# --------------------------------------------------------------------------
# Deterministic numerics against R
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["iid", "ar1", "short"])
def test_spectrum0_matches_coda(name):
    case = ORACLE["spectra"][name]
    assert ar_spectrum0(np.asarray(case["x"])) == pytest.approx(case["spec"], rel=1e-9)


def test_spectrum0_of_constant_trend_is_zero():
    assert ar_spectrum0(np.linspace(0.0, 1.0, 50)) == 0.0


def test_effective_size_matches_coda_mcmc_list():
    chains = [np.asarray(ORACLE["ess"]["chain1"]), np.asarray(ORACLE["ess"]["chain2"])]
    np.testing.assert_allclose(effective_size(chains), ORACLE["ess"]["ess"], rtol=1e-9)


@pytest.mark.parametrize("name", ["not_pd", "pd"])
def test_near_pd_matches_matrix_package(name):
    case = ORACLE["near_pd"][name]
    np.testing.assert_allclose(
        near_pd(np.asarray(case["x"])), np.asarray(case["mat"]), rtol=1e-9, atol=1e-12
    )


def test_iterative_scheme_matches_bridgesampling():
    scheme = ORACLE["scheme"]
    q11, q12 = np.asarray(scheme["q11"]), np.asarray(scheme["q12"])
    q21, q22 = np.asarray(scheme["q21"]), np.asarray(scheme["q22"])
    logml, niter = _solve(q11 - q12, q21 - q22, neff=scheme["neff"], maxiter=1000)
    assert logml == pytest.approx(scheme["normal_logml"], rel=1e-9)
    assert niter == scheme["normal_niter"]
    offset = -math.log(2.0) + float(np.log(np.diag(np.asarray(scheme["L"]))).sum())
    logml, niter = _solve(
        offset + q11 - q12, offset + q21 - q22, neff=scheme["neff"], maxiter=1000
    )
    assert logml == pytest.approx(scheme["warp_logml"], rel=1e-9)
    assert niter == scheme["warp_niter"]


def test_stalled_scheme_reports_iterates_and_restart_matches_r():
    scheme = ORACLE["scheme"]
    l1 = np.asarray(scheme["q11"]) - np.asarray(scheme["q12"])
    l2 = np.asarray(scheme["q21"]) - np.asarray(scheme["q22"])
    logml, niter, r_values = _iterate(
        l1, l2, neff=scheme["neff"], r0=0.5, tol=1e-10, maxiter=3, criterion="r"
    )
    assert logml is None and niter == 3
    np.testing.assert_allclose(r_values, scheme["stalled_r"], rtol=1e-9)
    logml, niter = _solve(l1, l2, neff=scheme["neff"], maxiter=3)
    assert logml == pytest.approx(scheme["restart_logml"], rel=1e-9)
    assert niter == 3 + scheme["restart_niter"]


def test_error_measures_match_bridgesampling():
    scheme = ORACLE["scheme"]
    result = BridgeResult(
        logml_repetitions=(scheme["normal_logml"],),
        niter=(scheme["normal_niter"],),
        method="normal",
        neff=scheme["neff"],
        q11=torch.tensor(scheme["q11"], dtype=torch.float64),
        q12=torch.tensor(scheme["q12"], dtype=torch.float64),
        q21=torch.tensor([scheme["q21"]], dtype=torch.float64),
        q22=torch.tensor([scheme["q22"]], dtype=torch.float64),
    )
    errors = result.error_measures()
    assert isinstance(errors, BridgeRelativeError)
    assert errors.re2 == pytest.approx(ORACLE["errors"]["re2"], rel=1e-9)
    assert errors.cv == pytest.approx(ORACLE["errors"]["cv"], rel=1e-9)
    with pytest.raises(ValueError, match="warp3"):
        result.with_(method="warp3").error_measures()


def test_post_prob_matches_bridgesampling():
    case = ORACLE["post_prob"]
    logml = dict(zip("abc", case["logml"], strict=True))
    np.testing.assert_allclose(
        list(posterior_model_probabilities(logml).values()), case["uniform"], rtol=1e-12
    )
    np.testing.assert_allclose(
        list(posterior_model_probabilities(logml, prior_prob=case["prior"]).values()),
        case["with_prior"],
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="sum to one"):
        posterior_model_probabilities(logml, prior_prob=[0.5, 0.3, 0.3])
    with pytest.raises(ValueError, match="match"):
        posterior_model_probabilities(logml, prior_prob=[0.5, 0.5])
    with pytest.raises(ValueError, match="two models"):
        posterior_model_probabilities({"a": -1.0})


def test_estimator_recovers_known_normalizer_and_rejects_wrong_normalizer():
    generator = torch.Generator().manual_seed(5)
    mean, scale = np.array([0.4, -1.0, 2.0]), np.array([0.3, 1.2, 0.7])
    draws = mean + scale * torch.randn(4, 600, 3, generator=generator).double().numpy()
    log_z = 2.75

    def log_q(points: np.ndarray) -> np.ndarray:
        return log_z + stats.norm.logpdf(points, mean, scale).sum(axis=1)

    for method in ("normal", "warp3"):
        result = bridge_estimate(log_q, draws, method=method, repetitions=2, seed=3)
        assert abs(result.logml - log_z) < 0.01
    single = bridge_estimate(log_q, draws, seed=3)
    error = single.error_measures()
    assert isinstance(error, BridgeRelativeError)
    assert abs(single.logml - log_z) < 4 * error.cv
    spread = bridge_estimate(log_q, draws, repetitions=3, seed=3).error_measures()
    assert isinstance(spread, BridgeRepetitionSpread) and spread.iqr >= 0

    # A density off by a constant shifts the estimate by exactly that constant.
    shifted = bridge_estimate(lambda p: log_q(p) - 1.0, draws, seed=3)
    assert shifted.logml == pytest.approx(single.logml - 1.0, abs=1e-9)


def test_estimator_rejects_invalid_draws_and_settings():
    def log_q(points: np.ndarray) -> np.ndarray:
        return -0.5 * (points**2).sum(axis=1)

    draws = np.random.default_rng(0).normal(size=(2, 50, 2))
    with pytest.raises(ValueError, match="method"):
        bridge_estimate(log_q, draws, method="laplace")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="four draws"):
        bridge_estimate(log_q, draws[:, :3])
    constant = draws.copy()
    constant[..., 1] = 0.5
    with pytest.raises(ValueError, match="vary"):
        bridge_estimate(log_q, constant)
    with pytest.raises(ValueError, match="every proposal draw"):
        bridge_estimate(lambda p: np.full(p.shape[0], -np.inf), draws)
    with pytest.raises(ValueError, match="repetitions"):
        bridge_estimate(log_q, draws, repetitions=0)


# --------------------------------------------------------------------------
# Fits with exact posterior draws
# --------------------------------------------------------------------------


def _exact(fitted, samples: dict[str, torch.Tensor]):
    chains, count = next(iter(samples.values())).shape[:2]
    posterior = MCMCResult(
        samples={k: v.to(torch.get_default_dtype()) for k, v in samples.items()},
        log_densities=torch.zeros(chains, count),
        acceptance_rates=torch.ones(chains),
        divergence_counts=torch.zeros(chains),
        num_warmup=0,
        num_samples=count,
    )
    return fitted.with_(posterior=posterior)


def _conjugate_regression(design: str, columns: list[np.ndarray]):
    """Fixed-sigma regression with N(0, 1) coefficients: exact logml and draws."""
    frame = pd.DataFrame({"y": Y, "x": X})
    fitted = fit(
        design,
        data=frame,
        fixed_parameters={"sigma": 1.0},
        priors=[NORMAL01, PriorSpec(distribution="Normal(0.0, 1.0)", class_="b")]
        if len(columns) > 1
        else [NORMAL01],
        method="svi",
        num_steps=1,
        seed=1,
    )
    matrix = np.column_stack(columns)
    covariance = matrix @ matrix.T + np.eye(Y.size)
    exact = float(stats.multivariate_normal(np.zeros(Y.size), covariance).logpdf(Y))
    precision = np.eye(matrix.shape[1]) + matrix.T @ matrix
    mean = np.linalg.solve(precision, matrix.T @ Y)
    chol = np.linalg.cholesky(np.linalg.inv(precision))
    generator = torch.Generator().manual_seed(17)
    z = torch.randn(4, 1000, matrix.shape[1], generator=generator, dtype=torch.float64)
    draws = torch.from_numpy(mean) + z @ torch.from_numpy(chol).T
    names = ["intercept", "beta_x"][: matrix.shape[1]]
    return _exact(fitted, {n: draws[..., i] for i, n in enumerate(names)}), exact


@pytest.fixture(scope="module")
def regression():
    return _conjugate_regression("y ~ x", [np.ones(Y.size), X])


@pytest.fixture(scope="module")
def intercept_only():
    return _conjugate_regression("y ~ 1", [np.ones(Y.size)])


def test_bridge_matches_closed_form_conjugate_regression(regression):
    fitted, exact = regression
    result = fitted.bridge_sampler()
    error = result.error_measures()
    assert isinstance(error, BridgeRelativeError)
    assert error.cv < 0.01
    assert abs(result.logml - exact) < 4 * error.cv
    warp = fitted.bridge_sampler(method="warp3", repetitions=2)
    assert abs(warp.logml - exact) < 0.01
    assert warp.method == "warp3" and warp.repetitions == 2


def _latent_scale_oracle():
    """y ~ 1 with Normal(0, 1) intercept and HalfNormal(1) sigma.

    The intercept is integrated analytically given sigma, and sigma by
    adaptive quadrature; exact draws use the inverse CDF of sigma's marginal
    posterior on a fine grid and the conjugate conditional of the intercept.
    """

    def log_marginal_given(sigma: float) -> float:
        cov = sigma**2 * np.eye(Y.size) + np.ones((Y.size, Y.size))
        return float(stats.multivariate_normal(np.zeros(Y.size), cov).logpdf(Y))

    def joint(sigma: float) -> float:
        return math.exp(log_marginal_given(sigma) + stats.halfnorm.logpdf(sigma) + 12.0)

    value, _ = integrate.quad(joint, 0.0, np.inf, epsabs=1e-13, epsrel=1e-11, limit=200)
    log_ml = math.log(value) - 12.0
    grid = np.linspace(1e-4, 4.0, 200001)
    log_density = np.array([log_marginal_given(s) for s in grid[::50]])
    dense = np.interp(grid, grid[::50], log_density) + stats.halfnorm.logpdf(grid)
    cdf = np.cumsum(np.exp(dense - dense.max()))
    cdf /= cdf[-1]
    rng = np.random.default_rng(23)
    sigma = np.interp(rng.uniform(size=4000), cdf, grid)
    precision = 1.0 + Y.size / sigma**2
    mean = (Y.sum() / sigma**2) / precision
    intercept = mean + rng.normal(size=4000) / np.sqrt(precision)
    return log_ml, {
        "intercept": torch.from_numpy(intercept.reshape(4, 1000)),
        "sigma": torch.from_numpy(sigma.reshape(4, 1000)),
    }


@pytest.fixture(scope="module")
def latent_scale():
    log_ml, samples = _latent_scale_oracle()
    fitted = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": Y}),
        priors=[NORMAL01, PriorSpec(distribution="HalfNormal(1.0)", class_="sigma")],
        method="svi",
        num_steps=1,
        seed=1,
    )
    return _exact(fitted, samples), log_ml


def test_bridge_includes_jacobian_of_constrained_scale(latent_scale):
    fitted, log_ml = latent_scale
    result = fitted.bridge_sampler()
    error = result.error_measures()
    assert isinstance(error, BridgeRelativeError)
    assert abs(result.logml - log_ml) < 4 * error.cv
    assert abs(fitted.bridge_sampler(method="warp3").logml - log_ml) < 0.01

    # The deliberately wrong variant without the log Jacobian of sigma's
    # transform estimates a different quantity and is rejected.
    upars, potential = log_posterior_function(fitted)

    def without_jacobian(points: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            values = [
                float(potential.log_density(torch.from_numpy(row)))
                for row in np.ascontiguousarray(points)
            ]
        return np.asarray(values) - points[:, 1]

    wrong = bridge_estimate(without_jacobian, upars)
    assert abs(wrong.logml - log_ml) > 10 * error.cv + 0.05


def test_bridge_agrees_with_smc_evidence_within_monte_carlo_error(latent_scale):
    fitted, log_ml = latent_scale
    bridge = fitted.bridge_sampler()
    cv = bridge.error_measures()
    assert isinstance(cv, BridgeRelativeError)
    runs = [
        fitted.evidence(
            SMC(
                MetropolisKernel(RandomWalkProposal(0.4)),
                num_particles=400,
                num_mcmc_steps=3,
                num_pilot_steps=0,
                seed=seed,
            )
        ).log_evidence
        for seed in (1, 2, 3, 4)
    ]
    smc_se = float(np.std(runs, ddof=1) / math.sqrt(len(runs)))
    smc = float(np.mean(runs))
    assert abs(smc - log_ml) < 4 * smc_se + 0.05
    assert abs(bridge.logml - smc) < 4 * math.hypot(cv.cv, smc_se) + 0.05


def test_bridge_matches_quadrature_of_hierarchical_model():
    """y ~ 1 + (1 | g), sigma fixed: integrate the group SD by quadrature."""
    frame = pd.DataFrame({"y": Y, "g": G})
    fitted = fit(
        "y ~ 1 + (1 | g)",
        data=frame,
        fixed_parameters={"sigma": 1.0},
        priors=[NORMAL01, PriorSpec(distribution="HalfNormal(1.0)", class_="sd")],
        method="svi",
        num_steps=1,
        seed=1,
    )
    levels = sorted(set(G))
    membership = np.array([[g == level for level in levels] for g in G], dtype=float)

    def log_marginal_given(sd: float) -> float:
        cov = np.ones((Y.size, Y.size)) + sd**2 * membership @ membership.T
        cov += np.eye(Y.size)
        return float(stats.multivariate_normal(np.zeros(Y.size), cov).logpdf(Y))

    value, _ = integrate.quad(
        lambda s: math.exp(log_marginal_given(s) + stats.halfnorm.logpdf(s) + 14.0),
        0.0,
        np.inf,
        epsabs=1e-13,
        epsrel=1e-11,
        limit=200,
    )
    log_ml = math.log(value) - 14.0
    grid = np.linspace(1e-4, 5.0, 100001)
    coarse = np.array([log_marginal_given(s) for s in grid[::25]])
    dense = np.interp(grid, grid[::25], coarse) + stats.halfnorm.logpdf(grid)
    cdf = np.cumsum(np.exp(dense - dense.max()))
    cdf /= cdf[-1]
    rng = np.random.default_rng(29)
    sds = np.interp(rng.uniform(size=4000), cdf, grid)
    intercepts, zs = [], []
    design = np.column_stack([np.ones(Y.size), membership])
    for sd in sds:
        prior_precision = np.diag([1.0, *(1.0 / sd**2,) * len(levels)])
        precision = prior_precision + design.T @ design
        mean = np.linalg.solve(precision, design.T @ Y)
        draw = mean + np.linalg.solve(
            np.linalg.cholesky(precision).T, rng.normal(size=design.shape[1])
        )
        intercepts.append(draw[0])
        zs.append(draw[1:] / sd)
    samples = {
        "intercept": torch.tensor(intercepts).reshape(4, 1000),
        "sigma_g_Intercept": torch.from_numpy(sds.reshape(4, 1000)),
        "z_g_Intercept": torch.tensor(np.asarray(zs)).reshape(4, 1000, len(levels)),
    }
    exact = _exact(fitted, samples)
    result = exact.bridge_sampler()
    error = result.error_measures()
    assert isinstance(error, BridgeRelativeError)
    assert abs(result.logml - log_ml) < 4 * error.cv
    assert abs(exact.bridge_sampler(method="warp3").logml - log_ml) < 0.01


def test_bayes_factor_and_post_prob_match_closed_forms(regression, intercept_only):
    full, exact_full = regression
    reduced, exact_reduced = intercept_only
    factor = bayes_factor(full, reduced, model_names=("full", "reduced"))
    assert isinstance(factor, BayesFactor)
    assert (factor.numerator, factor.denominator) == ("full", "reduced")
    assert math.log(factor.bf) == pytest.approx(exact_full - exact_reduced, abs=0.02)
    on_log = full.bayes_factor(reduced, log=True)
    assert on_log.log and on_log.bf == pytest.approx(math.log(factor.bf), abs=1e-12)
    stored = {"full": full.bridge_sampler(), "reduced": reduced.bridge_sampler()}
    probabilities = post_prob(stored, prior_prob=[0.25, 0.75])
    weights = np.array([0.25 * math.exp(exact_full), 0.75 * math.exp(exact_reduced)])
    expected = weights / weights.sum()
    np.testing.assert_allclose(list(probabilities.values()), expected, atol=0.01)
    assert post_prob({"full": full, "reduced": stored["reduced"]}) == pytest.approx(
        post_prob(stored)
    )


def test_models_of_different_responses_are_not_compared(regression):
    full, _ = regression
    other = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": Y + 1.0}),
        fixed_parameters={"sigma": 1.0},
        priors=[NORMAL01],
        method="svi",
        num_steps=1,
        seed=1,
    )
    with pytest.raises(ValueError, match="different response data"):
        bayes_factor(full, other)
    with pytest.raises(ValueError, match="different response data"):
        post_prob({"a": full, "b": other})


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_refuses_variational_draws_improper_priors_and_potentials(regression):
    full, _ = regression
    variational = fit(
        "y ~ x",
        data=pd.DataFrame({"y": Y, "x": X}),
        method="svi",
        num_steps=1,
        seed=1,
    )
    with pytest.raises(ValueError, match="posterior MCMC draws"):
        variational.bridge_sampler()
    improper = fit(
        "y ~ x",
        data=pd.DataFrame({"y": Y, "x": X}),
        default_priors="brms",
        method="svi",
        num_steps=1,
        seed=1,
    )
    with pytest.raises(ValueError, match="bridge_sampler requires a proper prior"):
        improper.with_(posterior=full.posterior).bridge_sampler()


# --------------------------------------------------------------------------
# Real samplers and joint fits
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["nuts", "hmc"])
def test_bridge_runs_on_sampler_draws(method):
    frame = pd.DataFrame({"y": Y})
    options = dict(num_warmup=150, num_samples=150, num_chains=2)
    fitted = fit(
        "y ~ 1",
        data=frame,
        fixed_parameters={"sigma": 1.0},
        priors=[NORMAL01],
        method=method,
        seed=4,
        **options,
    )
    precision = 1.0 + Y.size
    exact = -0.5 * (
        Y.size * math.log(2 * math.pi)
        + math.log(precision)
        + float((Y**2).sum() - Y.sum() ** 2 / precision)
    )
    result = fitted.bridge_sampler()
    error = result.error_measures()
    assert isinstance(error, BridgeRelativeError)
    assert abs(result.logml - exact) < 4 * error.cv + 0.02


def test_joint_fit_evidence_is_sum_of_independent_responses():
    """Two Gaussian responses without residual correlation factorize."""
    frame = pd.DataFrame({"y": Y, "w": Y[::-1] * 0.5 + 0.2})
    priors = [NORMAL01, PriorSpec(distribution="HalfNormal(1.0)", class_="sigma")]
    joint = fit(
        bf("y ~ 1") + bf("w ~ 1") + set_rescor(False),
        data=frame,
        priors=priors,
        method="svi",
        num_steps=1,
        seed=1,
    )
    log_ml_y, samples_y = _latent_scale_oracle()
    # The second response has its own oracle on its own data.
    singles = []
    for name, values in (("y", Y), ("w", frame["w"].to_numpy())):
        single = fit(
            f"{name} ~ 1",
            data=frame,
            priors=priors,
            method="svi",
            num_steps=1,
            seed=1,
        )
        singles.append(single)
    draws = joint.draws()
    assert {k.split("__")[-1] for k in draws} == {"intercept", "sigma"}
    rng = np.random.default_rng(31)
    samples = {}
    single_estimates = []
    for name, single in zip(("y", "w"), singles, strict=True):
        values = frame[name].to_numpy()
        # Exact draws from each response's posterior by the same construction.
        sigma_grid = np.linspace(1e-4, 4.0, 40001)
        log_given = np.array(
            [
                stats.multivariate_normal(
                    np.zeros(values.size),
                    s**2 * np.eye(values.size) + np.ones((values.size,) * 2),
                ).logpdf(values)
                for s in sigma_grid[::20]
            ]
        )
        dense = np.interp(sigma_grid, sigma_grid[::20], log_given)
        dense = dense + stats.halfnorm.logpdf(sigma_grid)
        cdf = np.cumsum(np.exp(dense - dense.max()))
        cdf /= cdf[-1]
        sigma = np.interp(rng.uniform(size=4000), cdf, sigma_grid)
        precision = 1.0 + values.size / sigma**2
        intercept = (values.sum() / sigma**2) / precision + rng.normal(
            size=4000
        ) / np.sqrt(precision)
        own = {
            "intercept": torch.from_numpy(intercept.reshape(4, 1000)),
            "sigma": torch.from_numpy(sigma.reshape(4, 1000)),
        }
        single_estimates.append(_exact(single, own).bridge_sampler())
        for key in draws:
            if key.startswith(f"response_{name}__"):
                samples[key] = own[key.split("__")[-1]]
    assert log_ml_y == pytest.approx(single_estimates[0].logml, abs=0.02)
    joint_estimate = _exact(joint, samples).bridge_sampler()
    error = joint_estimate.error_measures()
    assert isinstance(error, BridgeRelativeError)
    total = sum(e.logml for e in single_estimates)
    assert abs(joint_estimate.logml - total) < 0.02
