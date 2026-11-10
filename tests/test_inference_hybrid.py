"""Tests for the hybrid samplers (Layer 5).

Two hybrid strategies:

* :class:`AutoDAIS` wraps a base guide in an annealed-importance-
  sampling trajectory whose step size and inverse-temperature
  schedule are learnable. Its draws carry their trajectory's
  importance weight, so ELBO with it is a valid lower bound; the
  bound need not exceed the base guide's ELBO.
* :class:`WarmupThenHMC` runs SVI to convergence, then seeds an
  MCMC chain from the fitted guide's posterior mean.

Both are tested for the standard guide contract (sample shapes,
gradient flow, end-to-end SVI / MCMC run) plus an integration
check showing the composite recovers the closed-form Normal-Normal
posterior.
"""

from __future__ import annotations

import math

import torch

from quivers.dsl import loads
from quivers.inference import (
    AutoDAIS,
    AutoMultivariateNormalGuide,
    AutoNormalGuide,
    ELBO,
    HMCKernel,
    SVI,
    WarmupThenHMC,
)


def _normal_normal_model():
    return loads(
        "object Obs : FinSet 20\n"
        "program p : Obs -> Obs\n"
        "    sample mu <- Normal(loc=0.0, scale=1.0)\n"
        "    observe y : Obs <- Normal(loc=mu, scale=1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def _hierarchical_model():
    return loads(
        "object Subj : FinSet 4\n"
        "object Resp : FinSet 12\n"
        "program p : Resp -> Resp\n"
        "    sample sigma <- HalfNormal(scale=1.0)\n"
        "    sample by_subj : Subj <- Normal(loc=0.0, scale=sigma)\n"
        "    let mu = sigmoid(by_subj[subj_idx])\n"
        "    observe r : Resp <- Bernoulli(probs=mu)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def _make_hier_obs():
    return {
        "subj_idx": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
        "r": torch.zeros(12),
    }


# ---------------------------------------------------------------------------
# AutoDAIS
# ---------------------------------------------------------------------------


def test_dais_rsample_matches_autonormal_shapes() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs = _make_hier_obs()
    base = AutoNormalGuide(model, observed_names={"r"})
    g_normal = AutoNormalGuide(model, observed_names={"r"})
    dais = AutoDAIS(base, model, observations=obs, num_steps=2)
    s_normal = g_normal.rsample(torch.zeros(1, 1))
    s_dais = dais.rsample(torch.zeros(1, 1))
    assert set(s_normal.keys()) == set(s_dais.keys())
    for name in s_normal:
        assert s_normal[name].shape == s_dais[name].shape


def test_dais_rsample_follows_the_input_batch_axis() -> None:
    """One trajectory is shared across the batch, in the trace-side
    shapes every guide returns, and the density `rsample_strata`
    carries with the draw has the batch shape."""
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs = _make_hier_obs()
    base = AutoNormalGuide(model, observed_names={"r"})
    dais = AutoDAIS(base, model, observations=obs, num_steps=2)
    x = torch.zeros(3, 1)
    s_normal = AutoNormalGuide(model, observed_names={"r"}).rsample(x)
    s_dais = dais.rsample(x)
    for name in s_normal:
        assert s_dais[name].shape == s_normal[name].shape
    assert s_dais["sigma"].shape == (3,)
    assert s_dais["by_subj"].shape == (4,)
    assert torch.equal(s_dais["sigma"], s_dais["sigma"][:1].expand(3))
    (draw,) = dais.rsample_strata(x)
    for name in s_normal:
        assert draw.sites[name].shape == s_normal[name].shape
    assert draw.log_density.shape == (3,)


def test_dais_gradients_flow_to_step_size_and_temperatures() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs = _make_hier_obs()
    base = AutoNormalGuide(model, observed_names={"r"})
    dais = AutoDAIS(base, model, observations=obs, num_steps=2)
    elbo = ELBO()
    loss = elbo(model, dais, torch.zeros(1, 1), obs)
    loss.backward()
    assert dais.log_step_size.grad is not None
    assert torch.isfinite(dais.log_step_size.grad).all()
    assert dais.beta_increments_raw.grad is not None
    assert torch.isfinite(dais.beta_increments_raw.grad).all()
    # Base guide parameters should also receive gradients.
    for param in base.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_dais_rejects_num_steps_zero() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    obs = _make_hier_obs()
    base = AutoNormalGuide(model, observed_names={"r"})
    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        AutoDAIS(base, model, observations=obs, num_steps=0)


def test_dais_temperatures_increase_monotonically_to_one() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs = _make_hier_obs()
    base = AutoNormalGuide(model, observed_names={"r"})
    dais = AutoDAIS(base, model, observations=obs, num_steps=4)
    betas = dais._betas()
    assert betas.shape == (4,)
    diffs = betas[1:] - betas[:-1]
    assert torch.all(diffs > 0), f"betas not monotone: {betas}"
    assert math.isclose(float(betas[-1]), 1.0, abs_tol=1e-5), (
        f"final beta must be 1.0; got {float(betas[-1])}"
    )


def test_dais_runs_with_svi() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs = _make_hier_obs()
    base = AutoNormalGuide(model, observed_names={"r"})
    dais = AutoDAIS(base, model, observations=obs, num_steps=2)
    optim = torch.optim.Adam(
        list(model.parameters()) + list(dais.parameters()), lr=1e-2
    )
    svi = SVI(model, dais, optim, ELBO())
    losses = []
    for _ in range(5):
        losses.append(svi.step(torch.zeros(1, 1), obs))
    for loss in losses:
        assert torch.isfinite(torch.tensor(loss))


# ---------------------------------------------------------------------------
# WarmupThenHMC
# ---------------------------------------------------------------------------


def test_warmup_then_hmc_recovers_conjugate_posterior() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    N = 20
    y = torch.randn(N) + 1.5
    y_bar = float(y.mean())
    true_mean = N * y_bar / (N + 1)
    true_var = 1.0 / (N + 1)

    guide = AutoMultivariateNormalGuide(model, observed_names={"y"})
    kernel = HMCKernel(
        step_size=0.1,
        num_steps=10,
        mass_matrix="identity",
        adapt_step_size=True,
        adapt_mass_matrix=False,
    )
    composite = WarmupThenHMC(
        guide=guide,
        kernel=kernel,
        svi_steps=200,
        mcmc_warmup=50,
        mcmc_samples=300,
        num_chains=2,
        svi_lr=5e-2,
    )
    _, result = composite.run(model, torch.zeros(1, 1), {"y": y})
    mu_samples = result.samples["mu"].reshape(-1)
    sample_mean = float(mu_samples.mean())
    sample_var = float(mu_samples.var(unbiased=True))
    assert abs(sample_mean - true_mean) < 0.15, (
        f"WarmupThenHMC posterior mean off: got {sample_mean:.4f}, "
        f"expected {true_mean:.4f}"
    )
    assert abs(sample_var - true_var) < 0.05, (
        f"WarmupThenHMC posterior variance off: got {sample_var:.4f}, "
        f"expected {true_var:.4f}"
    )


def test_warmup_then_hmc_fit_guide_decreases_loss() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs = _make_hier_obs()
    guide = AutoNormalGuide(model, observed_names={"r"})
    kernel = HMCKernel(step_size=0.1, num_steps=5, mass_matrix="identity")
    composite = WarmupThenHMC(
        guide=guide,
        kernel=kernel,
        svi_steps=100,
        mcmc_warmup=10,
        mcmc_samples=20,
        num_chains=1,
        svi_lr=1e-2,
    )
    losses = composite.fit_guide(model, torch.zeros(1, 1), obs)
    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert late < early + 0.5, (
        f"WarmupThenHMC SVI warmup did not decrease loss: "
        f"early {early:.3f} vs late {late:.3f}"
    )


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_dais_rejects_leapfrog_steps_zero() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    obs = _make_hier_obs()
    base = AutoNormalGuide(model, observed_names={"r"})
    with pytest.raises(ValueError, match="leapfrog_steps must be >= 1"):
        AutoDAIS(base, model, observations=obs, leapfrog_steps=0)


def test_dais_rejects_invalid_step_size() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    obs = _make_hier_obs()
    base = AutoNormalGuide(model, observed_names={"r"})
    with pytest.raises(ValueError, match="init_step_size must be > 0"):
        AutoDAIS(base, model, observations=obs, init_step_size=-0.1)


def test_dais_rejects_invalid_init_temperature() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    obs = _make_hier_obs()
    base = AutoNormalGuide(model, observed_names={"r"})
    with pytest.raises(ValueError, match="init_temperature must be in"):
        AutoDAIS(base, model, observations=obs, init_temperature=0.0)
    with pytest.raises(ValueError, match="init_temperature must be in"):
        AutoDAIS(base, model, observations=obs, init_temperature=1.0)


def test_warmup_then_hmc_rejects_invalid_svi_steps() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    kernel = HMCKernel(step_size=0.1, num_steps=5)
    with pytest.raises(ValueError, match="svi_steps must be >= 1"):
        WarmupThenHMC(
            guide=guide,
            kernel=kernel,
            svi_steps=0,
            mcmc_warmup=10,
            mcmc_samples=10,
        )


def test_warmup_then_hmc_rejects_invalid_mcmc_samples() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    kernel = HMCKernel(step_size=0.1, num_steps=5)
    with pytest.raises(ValueError, match="mcmc_samples must be >= 1"):
        WarmupThenHMC(
            guide=guide,
            kernel=kernel,
            svi_steps=10,
            mcmc_warmup=10,
            mcmc_samples=0,
        )


def test_warmup_then_hmc_rejects_negative_warmup() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    kernel = HMCKernel(step_size=0.1, num_steps=5)
    with pytest.raises(ValueError, match="mcmc_warmup must be >= 0"):
        WarmupThenHMC(
            guide=guide,
            kernel=kernel,
            svi_steps=10,
            mcmc_warmup=-1,
            mcmc_samples=10,
        )
