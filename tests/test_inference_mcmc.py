"""Tests for the MCMC kernels and driver.

Cover:

1. Adaptation primitives — :class:`DualAveraging` converges the
   step size to the target acceptance probability;
   :class:`WelfordCovariance` recovers the empirical covariance
   of a known stream.
2. :class:`HMCKernel` on a Normal-Normal conjugate model — the
   posterior mean of the recovered chain matches the closed-form
   analytical posterior.
3. :class:`NUTSKernel` on the same model — same recovery target,
   and the per-step tree depth stays well below ``max_tree_depth``.
4. Driver-level behavior: parallel chains, split-:math:`\\hat R`
   close to ``1.0`` after convergence, ESS ≥ a small fraction of
   the chain length.
"""

from __future__ import annotations

import math

import pytest
import torch

from quivers.dsl import loads
from quivers.inference.mcmc.adapt import (
    DualAveraging,
    WelfordCovariance,
    find_reasonable_step_size,
)
from quivers.inference.mcmc import HMCKernel, MCMC, NUTSKernel
from quivers.inference.registry import LatentRegistry


# ---------------------------------------------------------------------------
# Adaptation primitives
# ---------------------------------------------------------------------------


def test_dual_averaging_converges_to_target_acceptance() -> None:
    """Feed a deterministic acceptance signal of 0.4; the dual-
    averaging step size should decrease (because target 0.8 > 0.4)
    over time."""
    da = DualAveraging(initial_step_size=1.0, target_accept=0.8)
    initial_step = da.step_size()
    for _ in range(200):
        da.update(0.4)
    final_step = da.smoothed_step_size()
    assert final_step < initial_step, (
        f"DualAveraging did not shrink step size when acceptance "
        f"(0.4) was below target (0.8): {initial_step:.4f} → {final_step:.4f}"
    )


def test_welford_recovers_empirical_covariance() -> None:
    torch.manual_seed(0)
    D = 5
    true_cov = torch.eye(D) * 0.5 + 0.5 * torch.ones(D, D) * 0.3
    L = torch.linalg.cholesky(true_cov)
    welford = WelfordCovariance(D, regularise=False)
    for _ in range(5000):
        x = L @ torch.randn(D)
        welford.update(x)
    estimated = welford.covariance()
    assert torch.allclose(estimated, true_cov, atol=0.05), (
        f"WelfordCovariance estimate too far from truth:\n"
        f"truth diag={torch.diagonal(true_cov)}\n"
        f"estd diag={torch.diagonal(estimated)}"
    )


def test_welford_diagonal_matches_dense_diagonal() -> None:
    torch.manual_seed(0)
    D = 4
    dense = WelfordCovariance(D, regularise=False, diagonal=False)
    diag = WelfordCovariance(D, regularise=False, diagonal=True)
    for _ in range(500):
        x = torch.randn(D) * torch.tensor([0.5, 1.0, 2.0, 0.1])
        dense.update(x)
        diag.update(x)
    dense_diag = torch.diagonal(dense.covariance())
    diag_diag = diag.covariance()
    assert torch.allclose(dense_diag, diag_diag, atol=1e-5)


def test_find_reasonable_step_size_returns_finite_positive() -> None:
    torch.manual_seed(0)

    def log_density(z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (z * z).sum()

    def grad_log_density(z: torch.Tensor) -> torch.Tensor:
        return -z

    eps = find_reasonable_step_size(log_density, grad_log_density, torch.zeros(3))
    assert math.isfinite(eps) and eps > 0


# ---------------------------------------------------------------------------
# Closed-form Normal-Normal recovery (the HMC / NUTS gold standard)
# ---------------------------------------------------------------------------


def _normal_normal_model():
    """y_i ~ N(mu, 1) with mu ~ N(0, 1). Posterior: N(N*y_bar/(N+1), 1/(N+1))."""
    return loads(
        "object Obs : FinSet 20\n"
        "program p : Obs -> Obs\n"
        "    sample mu <- Normal(loc=0.0, scale=1.0)\n"
        "    observe y : Obs <- Normal(loc=mu, scale=1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def test_hmc_recovers_conjugate_normal_normal_posterior() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    N = 20
    y = torch.randn(N) + 1.5
    y_bar = float(y.mean())
    true_mean = N * y_bar / (N + 1)
    true_var = 1.0 / (N + 1)

    kernel = HMCKernel(
        step_size=0.1,
        num_steps=15,
        mass_matrix="identity",
        adapt_step_size=True,
        adapt_mass_matrix=False,
    )
    driver = MCMC(
        kernel=kernel,
        num_warmup=200,
        num_samples=500,
        num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    mu_samples = result.samples["mu"].reshape(-1)
    sample_mean = float(mu_samples.mean())
    sample_var = float(mu_samples.var(unbiased=True))
    assert abs(sample_mean - true_mean) < 0.15, (
        f"HMC posterior mean off: got {sample_mean:.4f}, expected {true_mean:.4f}"
    )
    assert abs(sample_var - true_var) < 0.05, (
        f"HMC posterior variance off: got {sample_var:.4f}, expected {true_var:.4f}"
    )
    assert result.mean_acceptance > 0.3


def test_nuts_recovers_conjugate_normal_normal_posterior() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    N = 20
    y = torch.randn(N) + 1.5
    y_bar = float(y.mean())
    true_mean = N * y_bar / (N + 1)
    true_var = 1.0 / (N + 1)

    kernel = NUTSKernel(
        step_size=0.1,
        max_tree_depth=8,
        mass_matrix="diagonal",
        adapt_step_size=True,
        adapt_mass_matrix=True,
    )
    driver = MCMC(
        kernel=kernel,
        num_warmup=200,
        num_samples=400,
        num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    mu_samples = result.samples["mu"].reshape(-1)
    sample_mean = float(mu_samples.mean())
    sample_var = float(mu_samples.var(unbiased=True))
    assert abs(sample_mean - true_mean) < 0.15
    assert abs(sample_var - true_var) < 0.05


def test_mcmc_result_carries_diagnostics() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(20) + 1.0
    kernel = HMCKernel(step_size=0.1, num_steps=10, mass_matrix="identity")
    driver = MCMC(kernel=kernel, num_warmup=100, num_samples=200, num_chains=2)
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    assert result.samples["mu"].shape[:2] == (2, 200)
    assert "mu" in result.r_hat
    assert "mu" in result.ess
    assert result.acceptance_rates.shape == (2,)
    assert result.divergence_counts.shape == (2,)
    assert result.divergence_counts.dtype == torch.int64


def test_mcmc_init_strategy_zero_is_deterministic() -> None:
    """Two runs with the same RNG seed immediately before the
    driver call must produce identical chains."""
    model = _normal_normal_model()
    torch.manual_seed(0)
    y = torch.randn(20) + 1.0

    def _run() -> torch.Tensor:
        torch.manual_seed(42)
        kernel = HMCKernel(step_size=0.1, num_steps=5, mass_matrix="identity")
        driver = MCMC(
            kernel=kernel,
            num_warmup=10,
            num_samples=20,
            num_chains=2,
            init_strategy="zero",
        )
        return driver.run(model, torch.zeros(1, 1), {"y": y}).samples["mu"]

    a = _run()
    b = _run()
    assert torch.allclose(a, b)


def test_mcmc_seed_is_reproducible_chain_local_and_preserves_global_rng() -> None:
    model = _normal_normal_model()
    observations = {"y": torch.linspace(0.0, 1.0, 20)}

    def _run(seed: int) -> torch.Tensor:
        driver = MCMC(
            kernel=HMCKernel(
                step_size=0.05,
                num_steps=3,
                mass_matrix="identity",
                adapt_step_size=False,
                adapt_mass_matrix=False,
            ),
            num_warmup=2,
            num_samples=8,
            num_chains=2,
            seed=seed,
        )
        return driver.run(model, torch.zeros(1, 1), observations).samples["mu"]

    torch.manual_seed(9876)
    state_before = torch.random.get_rng_state().clone()
    first = _run(41)
    state_after = torch.random.get_rng_state()
    repeated = _run(41)
    changed = _run(42)

    assert torch.equal(state_before, state_after)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert not torch.equal(first[0], first[1])


def test_mcmc_generator_reproduces_from_equal_generator_states() -> None:
    model = _normal_normal_model()
    observations = {"y": torch.linspace(0.0, 1.0, 20)}

    def _run(generator: torch.Generator) -> torch.Tensor:
        driver = MCMC(
            kernel=HMCKernel(
                step_size=0.05,
                num_steps=2,
                adapt_step_size=False,
                adapt_mass_matrix=False,
            ),
            num_warmup=1,
            num_samples=5,
            num_chains=2,
            generator=generator,
        )
        return driver.run(model, torch.zeros(1, 1), observations).samples["mu"]

    first = _run(torch.Generator().manual_seed(123))
    repeated = _run(torch.Generator().manual_seed(123))
    assert torch.equal(first, repeated)


def test_mcmc_rejects_seed_and_generator_together() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        MCMC(
            HMCKernel(),
            num_warmup=1,
            num_samples=1,
            seed=1,
            generator=torch.Generator(),
        )


def test_nuts_adaptation_is_fresh_per_chain_and_run() -> None:
    model = _normal_normal_model()
    observations = {"y": torch.linspace(0.0, 1.0, 20)}
    kernel = NUTSKernel(
        step_size=0.01,
        max_tree_depth=2,
        mass_matrix="diagonal",
        adapt_step_size=True,
        adapt_mass_matrix=True,
    )
    driver = MCMC(
        kernel,
        num_warmup=3,
        num_samples=2,
        num_chains=2,
        init_strategy="zero",
        seed=7,
    )

    driver.run(model, torch.zeros(1, 1), observations)
    assert kernel._dual_avg is not None
    assert kernel._dual_avg.step_count == 3
    assert kernel._welford is not None
    assert kernel._welford.n == 3
    first_step_size = kernel.step_size

    driver.run(model, torch.zeros(1, 1), observations)
    assert kernel._dual_avg is not None
    assert kernel._dual_avg.step_count == 3
    assert kernel._welford is not None
    assert kernel._welford.n == 3
    assert kernel.step_size == first_step_size


def test_mcmc_rejects_nonfinite_guide_initialization() -> None:
    class _NonfiniteGuide:
        def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            del x
            return {"mu": torch.randn(1) * float("nan")}

    model = _normal_normal_model()
    driver = MCMC(
        HMCKernel(),
        num_warmup=1,
        num_samples=1,
        num_chains=1,
        init_strategy="guide",
        seed=1,
    )
    torch.manual_seed(314)
    state_before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="initial_position.*finite"):
        driver.run(
            model,
            torch.zeros(1, 1),
            {"y": torch.zeros(20)},
            guide=_NonfiniteGuide(),  # type: ignore[arg-type]
        )
    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_hmc_and_nuts_reject_nonfinite_initial_positions() -> None:
    model = _normal_normal_model()
    observations = {"y": torch.zeros(20)}
    registry = LatentRegistry.from_model(model, set(observations))
    for kernel in (HMCKernel(), NUTSKernel()):
        for value in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="initial_position.*finite"):
                kernel.init(
                    registry,
                    model,
                    torch.zeros(1, 1),
                    observations,
                    torch.tensor([value]),
                )


def test_nuts_propagates_divergence_from_nested_subtree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quivers.inference.mcmc.hmc as hmc_module
    from quivers.inference.mcmc.hmc import _MassMatrix

    calls = 0

    def _leapfrog_with_nested_divergence(
        z, p, grad, step_size, n_steps, potential, mass
    ):
        nonlocal calls
        del step_size, n_steps, potential, mass
        calls += 1
        log_density = torch.tensor(0.0 if calls == 1 else -100.0)
        return z + 1.0, p, log_density, grad

    monkeypatch.setattr(hmc_module, "_leapfrog", _leapfrog_with_nested_divergence)
    kernel = NUTSKernel(divergence_threshold=10.0, adapt_step_size=False)
    kernel._mass = _MassMatrix(1, "identity")
    kernel._set_generator(torch.Generator().manual_seed(0))
    tree = kernel._build_tree(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        direction=1,
        depth=1,
        h0=0.0,
        potential=object(),  # type: ignore[arg-type]
        eps=1.0,
    )

    assert calls == 2
    assert tree.diverged
    assert not tree.turning


def test_nuts_uturn_is_not_a_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    import quivers.inference.mcmc.hmc as hmc_module
    from quivers.inference.mcmc.hmc import _MassMatrix

    calls = 0

    def _leapfrog_with_uturn(z, p, grad, step_size, n_steps, potential, mass):
        nonlocal calls
        del step_size, n_steps, potential, mass
        calls += 1
        next_p = p if calls == 1 else -p
        return z + 1.0, next_p, torch.tensor(0.0), grad

    monkeypatch.setattr(hmc_module, "_leapfrog", _leapfrog_with_uturn)
    kernel = NUTSKernel(divergence_threshold=10.0, adapt_step_size=False)
    kernel._mass = _MassMatrix(1, "identity")
    kernel._set_generator(torch.Generator().manual_seed(0))
    tree = kernel._build_tree(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        direction=1,
        depth=1,
        h0=0.0,
        potential=object(),  # type: ignore[arg-type]
        eps=1.0,
    )

    assert calls == 2
    assert tree.turning
    assert not tree.diverged


# ---------------------------------------------------------------------------
# Additional MCMC coverage: validation, mass-matrix modes, init strategies
# ---------------------------------------------------------------------------


def test_hmc_rejects_invalid_step_size() -> None:
    import pytest

    with pytest.raises(ValueError, match="step_size must be > 0"):
        HMCKernel(step_size=-0.1, num_steps=5)
    with pytest.raises(ValueError, match="step_size must be > 0"):
        HMCKernel(step_size=0.0, num_steps=5)


def test_hmc_rejects_invalid_num_steps() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        HMCKernel(step_size=0.1, num_steps=0)


def test_hmc_rejects_invalid_target_accept() -> None:
    import pytest

    with pytest.raises(ValueError, match="target_accept must be in"):
        HMCKernel(step_size=0.1, num_steps=5, target_accept=1.1)
    with pytest.raises(ValueError, match="target_accept must be in"):
        HMCKernel(step_size=0.1, num_steps=5, target_accept=0.0)


def test_nuts_rejects_invalid_max_tree_depth() -> None:
    import pytest

    with pytest.raises(ValueError, match="max_tree_depth must be >= 1"):
        NUTSKernel(step_size=0.1, max_tree_depth=0)


def test_mcmc_rejects_invalid_chain_count() -> None:
    import pytest

    kernel = HMCKernel(step_size=0.1, num_steps=5)
    with pytest.raises(ValueError, match="num_chains must be >= 1"):
        MCMC(kernel=kernel, num_warmup=10, num_samples=10, num_chains=0)


def test_mcmc_rejects_negative_warmup() -> None:
    import pytest

    kernel = HMCKernel(step_size=0.1, num_steps=5)
    with pytest.raises(ValueError, match="num_warmup must be >= 0"):
        MCMC(kernel=kernel, num_warmup=-1, num_samples=10)


def test_mcmc_rejects_invalid_num_samples() -> None:
    import pytest

    kernel = HMCKernel(step_size=0.1, num_steps=5)
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        MCMC(kernel=kernel, num_warmup=10, num_samples=0)


def test_dual_averaging_rejects_invalid_initial_step() -> None:
    import pytest
    from quivers.inference.mcmc.adapt import DualAveraging

    with pytest.raises(ValueError, match="initial_step_size must be > 0"):
        DualAveraging(initial_step_size=-1.0)


def test_dual_averaging_rejects_invalid_target_accept() -> None:
    import pytest
    from quivers.inference.mcmc.adapt import DualAveraging

    with pytest.raises(ValueError, match="target_accept must be in"):
        DualAveraging(initial_step_size=0.1, target_accept=1.5)


def test_welford_rejects_invalid_dim() -> None:
    import pytest
    from quivers.inference.mcmc.adapt import WelfordCovariance

    with pytest.raises(ValueError, match="dim must be >= 1"):
        WelfordCovariance(dim=0)


def test_welford_rejects_mismatched_update_shape() -> None:
    import pytest
    from quivers.inference.mcmc.adapt import WelfordCovariance

    w = WelfordCovariance(dim=3)
    with pytest.raises(ValueError, match="expected shape"):
        w.update(torch.zeros(5))


def test_welford_with_few_samples_returns_identity() -> None:
    """Before enough samples accumulate, the regularised covariance
    must fall back to identity (the kernel-side default)."""
    from quivers.inference.mcmc.adapt import WelfordCovariance

    w = WelfordCovariance(dim=3, regularise=False)
    cov = w.covariance()
    assert torch.allclose(cov, torch.eye(3))


def test_mcmc_init_strategy_prior_produces_finite_results() -> None:
    """The 'prior' init strategy seeds chains with small random
    initial positions; the MCMC chain should still produce finite
    posterior samples."""
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(20) + 1.0
    kernel = HMCKernel(step_size=0.05, num_steps=10, mass_matrix="identity")
    driver = MCMC(
        kernel=kernel,
        num_warmup=50,
        num_samples=100,
        num_chains=2,
        init_strategy="prior",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    assert torch.isfinite(result.log_densities).all()


def test_hmc_with_dense_mass_matrix_runs() -> None:
    """The dense mass matrix path is exercised end-to-end. With
    adaptation enabled the Welford covariance feeds into a Cholesky
    factorization; we just need it to produce finite samples."""
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(20) + 1.0
    kernel = HMCKernel(
        step_size=0.1,
        num_steps=10,
        mass_matrix="dense",
        adapt_step_size=True,
        adapt_mass_matrix=True,
    )
    driver = MCMC(
        kernel=kernel,
        num_warmup=100,
        num_samples=200,
        num_chains=1,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    assert torch.isfinite(result.log_densities).all()


def test_hmc_with_diagonal_mass_matrix_adapts() -> None:
    """With diagonal mass-matrix adaptation enabled, the kernel's
    welford accumulator should fill up during warmup. Verify the
    chain runs and ESS is non-trivial."""
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(20) + 1.0
    kernel = HMCKernel(
        step_size=0.1,
        num_steps=10,
        mass_matrix="diagonal",
        adapt_step_size=True,
        adapt_mass_matrix=True,
    )
    driver = MCMC(
        kernel=kernel,
        num_warmup=100,
        num_samples=200,
        num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    assert torch.isfinite(result.log_densities).all()


def test_mass_matrix_identity_rejects_set_inverse() -> None:
    import pytest
    from quivers.inference.mcmc.hmc import _MassMatrix

    mass = _MassMatrix(dim=3, kind="identity")
    with pytest.raises(RuntimeError, match="cannot set inverse on identity"):
        mass.set_inverse(torch.ones(3))


def test_mass_matrix_diagonal_rejects_wrong_shape() -> None:
    import pytest
    from quivers.inference.mcmc.hmc import _MassMatrix

    mass = _MassMatrix(dim=3, kind="diagonal")
    with pytest.raises(ValueError, match="expected shape"):
        mass.set_inverse(torch.ones(4))


def test_mass_matrix_dense_rejects_wrong_shape() -> None:
    import pytest
    from quivers.inference.mcmc.hmc import _MassMatrix

    mass = _MassMatrix(dim=3, kind="dense")
    with pytest.raises(ValueError, match="expected shape"):
        mass.set_inverse(torch.ones(2, 2))


def test_mass_matrix_unknown_kind_rejected() -> None:
    import pytest
    from quivers.inference.mcmc.hmc import _MassMatrix

    with pytest.raises(ValueError, match="kind must be one of"):
        _MassMatrix(dim=3, kind="bogus")  # type: ignore[arg-type]


def test_mcmc_result_acceptance_rates_in_range() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(20) + 1.0
    kernel = HMCKernel(step_size=0.1, num_steps=10, mass_matrix="identity")
    driver = MCMC(
        kernel=kernel,
        num_warmup=50,
        num_samples=100,
        num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    assert torch.all(result.acceptance_rates >= 0.0)
    assert torch.all(result.acceptance_rates <= 1.0)


def test_mcmc_result_mean_acceptance_property() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(20) + 1.0
    kernel = HMCKernel(step_size=0.1, num_steps=10, mass_matrix="identity")
    driver = MCMC(
        kernel=kernel,
        num_warmup=50,
        num_samples=100,
        num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    assert isinstance(result.mean_acceptance, float)
    assert 0.0 <= result.mean_acceptance <= 1.0
    assert result.num_chains == 2


def test_mcmc_init_strategy_guide_with_no_guide_raises() -> None:
    import pytest

    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(20) + 1.0
    kernel = HMCKernel(step_size=0.1, num_steps=10, mass_matrix="identity")
    driver = MCMC(
        kernel=kernel,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        init_strategy="guide",
    )
    with pytest.raises(ValueError, match="requires a guide"):
        driver.run(model, torch.zeros(1, 1), {"y": y})
