"""Data generators for the synthetic benchmark suite.

Each generator function loads its model from
``tests/benchmarks/models/<name>.qvr`` and returns a
:class:`BenchmarkData` tuple ``(model, observations, true_params)``.
Data draws are deterministic in the supplied ``seed`` so the
benchmark numbers reproduce across runs and platforms.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.dsl import loads


_MODELS_DIR = Path(__file__).parent / "models"


class BenchmarkData(NamedTuple):
    """Triple returned by every benchmark data generator."""

    model: ContinuousMorphism
    observations: dict[str, torch.Tensor]
    true_params: dict[str, float | torch.Tensor | int]


def _load_model(name: str):
    path = _MODELS_DIR / f"{name}.qvr"
    return loads(path.read_text()).morphism


# ============================================================================
# Conjugate models
# ============================================================================


def beta_bernoulli(
    true_theta: float = 0.7, seed: int = 0, N: int = 50
) -> BenchmarkData:
    """``y_i ~ Bernoulli(true_theta)``; model: Beta(2, 2) prior."""
    model = _load_model("beta_bernoulli")
    g = torch.Generator().manual_seed(seed)
    y = torch.bernoulli(torch.full((N,), true_theta), generator=g)
    return BenchmarkData(
        model=model,
        observations={"y": y},
        true_params={"theta": true_theta, "alpha0": 2.0, "beta0": 2.0, "N": N},
    )


def normal_normal(
    true_mu: float = 1.5, sigma: float = 1.0, seed: int = 0, N: int = 30
) -> BenchmarkData:
    """``y_i ~ Normal(true_mu, sigma)``; model: Normal(0, 1) prior."""
    model = _load_model("normal_normal")
    g = torch.Generator().manual_seed(seed)
    y = true_mu + sigma * torch.randn(N, generator=g)
    return BenchmarkData(
        model=model,
        observations={"y": y},
        true_params={
            "mu": true_mu,
            "sigma": sigma,
            "mu0": 0.0,
            "tau0": 1.0,
            "N": N,
        },
    )


def normal_inverse_gamma(
    true_mu: float = 0.3,
    true_sigma2: float = 1.5,
    seed: int = 0,
    N: int = 60,
) -> BenchmarkData:
    """Normal-Inverse-Gamma joint conjugate.

    Generator draws y_i ~ Normal(true_mu, sqrt(true_sigma2)).
    Prior: sigma2 ~ IG(3, 2), mu | sigma2 ~ Normal(0, sigma).
    """
    model = _load_model("normal_inverse_gamma")
    g = torch.Generator().manual_seed(seed)
    y = true_mu + (true_sigma2**0.5) * torch.randn(N, generator=g)
    return BenchmarkData(
        model=model,
        observations={"y": y},
        true_params={
            "mu": true_mu,
            "sigma2": true_sigma2,
            "mu0": 0.0,
            "kappa0": 1.0,
            "a0": 3.0,
            "b0": 2.0,
            "N": N,
        },
    )


def gamma_exponential(
    true_rate: float = 2.0, seed: int = 0, N: int = 80
) -> BenchmarkData:
    """``y_i ~ Exponential(true_rate)``; model: Gamma(2, 1) prior on rate."""
    model = _load_model("gamma_exponential")
    g = torch.Generator().manual_seed(seed)
    # Inverse-CDF for Exponential: F(y) = 1 - exp(-rate * y).
    # ``torch.distributions.Exponential.sample`` does not accept a
    # ``Generator``, so we go through ``torch.rand`` for determinism.
    y = -torch.log(torch.rand(N, generator=g).clamp(min=1e-30)) / true_rate
    return BenchmarkData(
        model=model,
        observations={"y": y},
        true_params={"rate": true_rate, "a0": 2.0, "b0": 1.0, "N": N},
    )


def bayes_linear_regression(
    true_a: float = 0.7,
    true_b: float = -0.5,
    sigma: float = 0.3,
    seed: int = 0,
    N: int = 60,
) -> BenchmarkData:
    """Well-conditioned linear regression with iid Normal(0, 1) design."""
    model = _load_model("bayes_linear_regression")
    g_x = torch.Generator().manual_seed(seed + 1)
    g_y = torch.Generator().manual_seed(seed)
    x_design = torch.randn(N, generator=g_x)
    y = true_a + true_b * x_design + sigma * torch.randn(N, generator=g_y)
    return BenchmarkData(
        model=model,
        observations={"y": y, "x_design": x_design},
        true_params={
            "a": true_a,
            "b": true_b,
            "sigma": sigma,
            "N": N,
            "x_design": x_design,
        },
    )


# ============================================================================
# Hierarchical
# ============================================================================


# Canonical Eight Schools data (Rubin 1981, Gelman et al. 2013).
_EIGHT_SCHOOLS_Y = torch.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])


def eight_schools_centered(seed: int = 0) -> BenchmarkData:
    """Eight Schools data (canonical Rubin 1981 fixtures), centered model."""
    model = _load_model("eight_schools_centered")
    return BenchmarkData(
        model=model,
        observations={"y": _EIGHT_SCHOOLS_Y.clone()},
        true_params={"y_obs": _EIGHT_SCHOOLS_Y.clone(), "sigma_obs": 12.0, "N": 8},
    )


def eight_schools_noncentered(seed: int = 0) -> BenchmarkData:
    """Eight Schools data, non-centered reparameterization."""
    model = _load_model("eight_schools_noncentered")
    return BenchmarkData(
        model=model,
        observations={"y": _EIGHT_SCHOOLS_Y.clone()},
        true_params={"y_obs": _EIGHT_SCHOOLS_Y.clone(), "sigma_obs": 12.0, "N": 8},
    )


# ============================================================================
# Hard posterior geometries
# ============================================================================


def correlated_regression(
    true_a: float = 0.7,
    true_b: float = -0.5,
    correlation_strength: float = 0.95,
    sigma: float = 0.5,
    seed: int = 0,
    N: int = 50,
) -> BenchmarkData:
    """Near-collinear design producing a strongly-correlated posterior."""
    model = _load_model("correlated_regression")
    g_x = torch.Generator().manual_seed(seed + 1)
    g_y = torch.Generator().manual_seed(seed)
    z = torch.randn(N, generator=g_x)
    x_design = correlation_strength + (1.0 - correlation_strength) * z
    y = true_a + true_b * x_design + sigma * torch.randn(N, generator=g_y)
    return BenchmarkData(
        model=model,
        observations={"y": y, "x_design": x_design},
        true_params={
            "a": true_a,
            "b": true_b,
            "sigma": sigma,
            "correlation_strength": correlation_strength,
            "N": N,
            "x_design": x_design,
        },
    )


def neal_funnel(seed: int = 0) -> BenchmarkData:
    """Condition Neal's funnel on nine observations equal to zero.

    Parameters
    ----------
    seed : int
        Accepted by the shared data-factory interface. The fixed zero
        observations require no random draws.

    Returns
    -------
    BenchmarkData
        The model and observations defining the Gaussian conditional
        over v, with analytical mean -40.5 and variance 9.
    """
    model = _load_model("neal_funnel")
    # Condition on x_i = 0 so the inference target is p(v | x = 0).
    return BenchmarkData(
        model=model,
        observations={"x": torch.zeros(9)},
        true_params={"sigma_v": 3.0, "N_dims": 9},
    )


def ill_conditioned_mvn(seed: int = 0) -> BenchmarkData:
    """Ill-conditioned product Gaussian: per-dim observations at fixed scales."""
    model = _load_model("ill_conditioned_mvn")
    g = torch.Generator().manual_seed(seed)
    prior_scales = torch.tensor([100.0, 10.0, 1.0, 0.1, 0.01])
    true_x = prior_scales * torch.randn(5, generator=g)
    obs_scale = 0.1
    y_obs = true_x + obs_scale * torch.randn(5, generator=g)
    return BenchmarkData(
        model=model,
        observations={f"y_{i + 1}": y_obs[i : i + 1] for i in range(5)},
        true_params={
            "prior_scales": prior_scales,
            "obs_scale": obs_scale,
            "true_x": true_x,
        },
    )


# ============================================================================
# Constrained-support stress
# ============================================================================


def half_normal_scale(
    true_sigma: float = 1.5, seed: int = 0, N: int = 80
) -> BenchmarkData:
    """``y_i ~ Normal(0, true_sigma)``; model: HalfNormal(2) prior on sigma."""
    model = _load_model("half_normal_scale")
    g = torch.Generator().manual_seed(seed)
    y = true_sigma * torch.randn(N, generator=g)
    return BenchmarkData(
        model=model,
        observations={"y": y},
        true_params={"sigma": true_sigma, "prior_scale": 2.0, "N": N},
    )


def truncated_normal_recovery(
    true_mu: float = 0.6, seed: int = 0, N: int = 60
) -> BenchmarkData:
    """``y_i ~ TruncatedNormal(true_mu, 0.2, 0, 1)``; uniform prior on mu."""
    model = _load_model("truncated_normal_recovery")
    g = torch.Generator().manual_seed(seed)
    # Inverse-CDF sampler for truncated normal.
    sigma = 0.2
    low, high = 0.0, 1.0
    normal = torch.distributions.Normal(0.0, 1.0)
    alpha = normal.cdf(torch.tensor((low - true_mu) / sigma))
    beta = normal.cdf(torch.tensor((high - true_mu) / sigma))
    u = torch.rand(N, generator=g)
    u_scaled = alpha + u * (beta - alpha)
    y = normal.icdf(u_scaled.clamp(1e-6, 1 - 1e-6)) * sigma + true_mu
    return BenchmarkData(
        model=model,
        observations={"y": y},
        true_params={"mu": true_mu, "sigma": sigma, "low": low, "high": high, "N": N},
    )


__all__ = [
    "BenchmarkData",
    # Conjugate
    "beta_bernoulli",
    "normal_normal",
    "normal_inverse_gamma",
    "gamma_exponential",
    "bayes_linear_regression",
    # Hierarchical
    "eight_schools_centered",
    "eight_schools_noncentered",
    # Hard geometry
    "correlated_regression",
    "neal_funnel",
    "ill_conditioned_mvn",
    # Constrained support
    "half_normal_scale",
    "truncated_normal_recovery",
]
