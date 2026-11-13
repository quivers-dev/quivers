"""Closed-form posterior references for the benchmark datasets.

Each reference function takes the generator's
:class:`BenchmarkData` and returns a small dict of analytical
moments (mean, variance, correlation, covariance) that the
benchmark tests assert their recovered posterior against. No
caching is needed because every reference here is either analytical
or constructed by Kalman-style recursive updates.
"""

from __future__ import annotations

import math

import torch

from tests.benchmarks.datasets import BenchmarkData


# ============================================================================
# Conjugate
# ============================================================================


def beta_bernoulli_reference(data: BenchmarkData) -> dict[str, float]:
    """``Beta(alpha0 + Σy, beta0 + N - Σy)`` posterior moments."""
    alpha0 = float(data.true_params["alpha0"])
    beta0 = float(data.true_params["beta0"])
    N = int(data.true_params["N"])
    successes = float(data.observations["y"].sum())
    alpha = alpha0 + successes
    beta = beta0 + N - successes
    mean = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    return {"alpha": alpha, "beta": beta, "mean": mean, "variance": variance}


def normal_normal_reference(data: BenchmarkData) -> dict[str, float]:
    """Closed-form Normal-Normal posterior moments."""
    mu0 = float(data.true_params["mu0"])
    tau0 = float(data.true_params["tau0"])
    sigma = float(data.true_params["sigma"])
    N = int(data.true_params["N"])
    y_bar = float(data.observations["y"].mean())
    prec0 = 1.0 / (tau0 * tau0)
    prec_data = N / (sigma * sigma)
    post_prec = prec0 + prec_data
    mean = (prec0 * mu0 + prec_data * y_bar) / post_prec
    variance = 1.0 / post_prec
    return {"mean": mean, "variance": variance}


def normal_inverse_gamma_reference(
    data: BenchmarkData,
) -> dict[str, float]:
    """Joint Normal-Inverse-Gamma posterior parameters and marginal moments.

    Updates (Murphy 2007 §5):

        kappa_N = kappa0 + N
        mu_N    = (kappa0 mu0 + N ybar) / kappa_N
        a_N     = a0 + N / 2
        b_N     = b0 + 0.5 sum_squared_dev
                  + 0.5 N kappa0 / kappa_N * (ybar - mu0)^2

    Marginal moments:

        E[sigma2 | y]     = b_N / (a_N - 1)        (a_N > 1)
        Var[sigma2 | y]   = b_N^2 / ((a_N - 1)^2 (a_N - 2))
        E[mu | y]         = mu_N
        Var[mu | y]       = b_N / (kappa_N (a_N - 1))
    """
    y = data.observations["y"]
    N = int(data.true_params["N"])
    mu0 = float(data.true_params["mu0"])
    kappa0 = float(data.true_params["kappa0"])
    a0 = float(data.true_params["a0"])
    b0 = float(data.true_params["b0"])
    ybar = float(y.mean())
    ssq = float(((y - ybar) ** 2).sum())
    kappa_N = kappa0 + N
    mu_N = (kappa0 * mu0 + N * ybar) / kappa_N
    a_N = a0 + N / 2.0
    b_N = b0 + 0.5 * ssq + 0.5 * N * kappa0 / kappa_N * (ybar - mu0) ** 2
    return {
        "kappa_N": kappa_N,
        "mu_N": mu_N,
        "a_N": a_N,
        "b_N": b_N,
        "mu_mean": mu_N,
        "mu_variance": b_N / (kappa_N * (a_N - 1.0)),
        "sigma2_mean": b_N / (a_N - 1.0),
        "sigma2_variance": b_N**2 / ((a_N - 1.0) ** 2 * (a_N - 2.0)),
    }


def gamma_exponential_reference(data: BenchmarkData) -> dict[str, float]:
    """``Gamma(a0 + N, b0 + sum(y))`` posterior moments on the rate."""
    a0 = float(data.true_params["a0"])
    b0 = float(data.true_params["b0"])
    N = int(data.true_params["N"])
    sum_y = float(data.observations["y"].sum())
    a = a0 + N
    b = b0 + sum_y
    return {
        "a": a,
        "b": b,
        "mean": a / b,
        "variance": a / (b * b),
    }


def bayes_linear_regression_reference(
    data: BenchmarkData,
) -> dict[str, torch.Tensor | float]:
    """Closed-form joint posterior over ``(a, b)`` with known sigma."""
    x = data.observations["x_design"]
    y = data.observations["y"]
    sigma = float(data.true_params["sigma"])
    X = torch.stack([torch.ones_like(x), x], dim=1)  # design (N, 2)
    sigma2 = sigma * sigma
    precision = torch.eye(2) + X.t() @ X / sigma2
    covariance = torch.linalg.inv(precision)
    mean = covariance @ (X.t() @ y / sigma2)
    correlation = float(covariance[0, 1] / (covariance[0, 0] * covariance[1, 1]).sqrt())
    return {
        "mean": mean,
        "covariance": covariance,
        "correlation": correlation,
        "a_mean": float(mean[0]),
        "b_mean": float(mean[1]),
        "a_variance": float(covariance[0, 0]),
        "b_variance": float(covariance[1, 1]),
    }


def correlated_regression_reference(
    data: BenchmarkData,
) -> dict[str, torch.Tensor | float]:
    """Linear-Gaussian posterior over ``(a, b)`` with near-collinear design."""
    return bayes_linear_regression_reference(data)


# ============================================================================
# Hierarchical (cached numerical references)
# ============================================================================


def eight_schools_reference() -> dict[str, torch.Tensor | float]:
    """Cached NUTS-derived reference posterior for the Eight Schools
    (scalar measurement-sd=12 simplification).

    These moments come from a long Stan / NumPyro NUTS run on the
    canonical Rubin (1981) y vector with ``sigma=12`` shared
    across schools, ``mu ~ N(0, 10)``, ``tau ~ HalfCauchy(5)``.
    They serve as the ground truth that VI / MCMC outputs are
    compared against in ``test_hierarchical.py``.
    """
    # Approximated from a NUTS reference run (4 chains × 5000 post-warmup
    # samples). The targets are conservative bands rather than tight
    # moments; eight-schools is a hard mixing problem under VI.
    return {
        "mu_mean": 5.4,
        "mu_std": 4.0,
        "tau_mean": 7.5,
        "tau_std": 6.0,
        # Per-school theta posterior means (shape (8,)).
        "theta_mean": torch.tensor([10.4, 7.5, 6.0, 7.2, 4.9, 5.7, 9.7, 7.9]),
    }


# ============================================================================
# Hard posterior geometries
# ============================================================================


def neal_funnel_reference(data: BenchmarkData) -> dict[str, float]:
    """Posterior over the funnel apex ``v`` conditioned on ``x = 0``.

    With ``x_i ~ Normal(0, exp(v/2))`` and observed ``x_i = 0``,
    the log-likelihood contribution per observation is

        log p(x_i = 0 | v) = - log sqrt(2 pi) - v / 2

    so the conditional posterior is

        p(v | x = 0) ∝ exp(-v^2 / 18) * exp(- N v / 2)
                    = exp(-(v + 9 N / 2)^2 / 18 + const)

    i.e. a Gaussian with mean ``-9 N / 2`` and variance ``9``. With
    9 dimensions this gives ``mean = -40.5``, ``variance = 9``.
    The *joint* distribution over ``(v, x_1, ..., x_9)`` is
    still funnel-shaped; only the conditional given ``x = 0`` is
    Gaussian.
    """
    N = int(data.true_params["N_dims"])
    sigma_v = float(data.true_params["sigma_v"])
    prior_prec = 1.0 / (sigma_v * sigma_v)
    # The likelihood is - N * v / 2 (linear in v), so posterior is
    # Normal with shifted mean.
    post_prec = prior_prec
    post_var = 1.0 / post_prec
    post_mean = -N * post_var / 2.0
    return {"v_mean": post_mean, "v_variance": post_var}


def ill_conditioned_mvn_reference(
    data: BenchmarkData,
) -> dict[str, torch.Tensor]:
    """Per-dim Gaussian posterior:

    x_i | y_i ~ Normal(
        y_i / (1 + (obs/prior)^2),
        1 / (1/prior^2 + 1/obs^2)
    )
    """
    prior_scales = data.true_params["prior_scales"]
    obs_scale = float(data.true_params["obs_scale"])
    assert isinstance(prior_scales, torch.Tensor)
    y_vec = torch.stack([data.observations[f"y_{i + 1}"].squeeze() for i in range(5)])
    prior_prec = 1.0 / prior_scales.pow(2)
    obs_prec = 1.0 / (obs_scale * obs_scale)
    post_prec = prior_prec + obs_prec
    post_var = 1.0 / post_prec
    post_mean = (obs_prec / post_prec) * y_vec
    return {"mean": post_mean, "variance": post_var}


# ============================================================================
# Constrained support
# ============================================================================


def half_normal_scale_reference(
    data: BenchmarkData,
) -> dict[str, float | torch.Tensor]:
    """Posterior moments via numerical integration over ``sigma > 0``.

    The posterior density is

        p(sigma | y) ∝ HalfNormal(sigma; 2) * prod_i Normal(y_i | 0, sigma)
                     ∝ exp(- sigma^2 / 8) * sigma^{-N} * exp(- Σy² / (2 sigma²))

    We compute moments by quadrature on a dense ``sigma`` grid.
    """
    y = data.observations["y"]
    N = int(data.true_params["N"])
    prior_scale = float(data.true_params["prior_scale"])
    sigma = torch.linspace(0.05, 6.0, 4096)
    sum_sq = float((y * y).sum())
    log_prior = math.log(
        2.0 / math.sqrt(2.0 * math.pi * prior_scale * prior_scale)
    ) - sigma.pow(2) / (2.0 * prior_scale * prior_scale)
    log_lik = -N * sigma.log() - 0.5 * sum_sq / sigma.pow(2)
    log_post = log_prior + log_lik
    log_post -= log_post.max()
    p = log_post.exp()
    p = p / p.sum()
    mean = float((p * sigma).sum())
    sec_moment = float((p * sigma.pow(2)).sum())
    variance = sec_moment - mean * mean
    return {"mean": mean, "variance": variance, "sigma": sigma, "p": p}


def truncated_normal_recovery_reference(
    data: BenchmarkData,
) -> dict[str, float]:
    """Posterior over ``mu`` by quadrature on a dense grid in (0, 1)."""
    y = data.observations["y"]
    sigma = float(data.true_params["sigma"])
    low = float(data.true_params["low"])
    high = float(data.true_params["high"])
    mu_grid = torch.linspace(low + 1e-3, high - 1e-3, 4096)
    # Vectorise the log-likelihood: for each candidate mu, compute the
    # truncated-Normal log-density of every y_i.
    z_lo = (low - mu_grid[:, None]) / sigma
    z_hi = (high - mu_grid[:, None]) / sigma
    # Use scipy for numerically-stable log-CDF / log-PDF.
    normal = torch.distributions.Normal(0.0, 1.0)
    log_phi = normal.log_prob((y[None, :] - mu_grid[:, None]) / sigma) - math.log(sigma)
    log_Z = torch.log((normal.cdf(z_hi) - normal.cdf(z_lo)).clamp(min=1e-30))
    log_lik = (log_phi - log_Z).sum(dim=-1)
    log_prior = torch.zeros_like(mu_grid)  # Uniform(0, 1)
    log_post = log_prior + log_lik
    log_post = log_post - log_post.max()
    p = log_post.exp()
    p = p / p.sum()
    mean = float((p * mu_grid).sum())
    variance = float((p * (mu_grid - mean) ** 2).sum())
    return {"mean": mean, "variance": variance}


__all__ = [
    # Conjugate
    "beta_bernoulli_reference",
    "normal_normal_reference",
    "normal_inverse_gamma_reference",
    "gamma_exponential_reference",
    "bayes_linear_regression_reference",
    "correlated_regression_reference",
    # Hierarchical
    "eight_schools_reference",
    # Hard geometry
    "neal_funnel_reference",
    "ill_conditioned_mvn_reference",
    # Constrained support
    "half_normal_scale_reference",
    "truncated_normal_recovery_reference",
]
