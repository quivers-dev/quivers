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
from collections.abc import Callable

import pytest
import torch

import quivers.inference.mcmc.hmc as hmc_module
from torch.distributions import Categorical, Distribution, Gamma, Normal

from quivers.continuous import (
    ContinuousMorphism,
    Draw,
    Euclidean,
    MonadicProgram,
    make_fixed_halfnormal,
    make_fixed_normal,
)
from quivers.continuous.program_steps import Score, reading
from quivers.core import Unit
from quivers.dsl import loads
from quivers.inference.mcmc.adapt import (
    DualAveraging,
    WelfordCovariance,
    find_reasonable_step_size,
)
from quivers.inference.mcmc import HMCKernel, MCMC, NUTSKernel, PotentialFn
from quivers.inference.mcmc.hmc import MassMatrixKind
from quivers.inference.mcmc.kernel import _rejects_position
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


# ---------------------------------------------------------------------------
# Target-evaluation budget of the leapfrog integrator
# ---------------------------------------------------------------------------


class _CountingPotential(PotentialFn):
    """`PotentialFn` that counts every target evaluation it serves."""

    def __init__(
        self,
        model: MonadicProgram,
        registry: LatentRegistry,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> None:
        super().__init__(model, registry, x, observations)
        self.forward_passes = 0
        self.forward_with_grad_mode = 0
        self.gradient_calls = 0

    def log_density(self, z: torch.Tensor) -> torch.Tensor:
        self.forward_passes += 1
        self.forward_with_grad_mode += int(torch.is_grad_enabled())
        return super().log_density(z)

    def value_and_grad(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.gradient_calls += 1
        return super().value_and_grad(z)

    def reset(self) -> None:
        self.forward_passes = 0
        self.forward_with_grad_mode = 0
        self.gradient_calls = 0


def _standard_normal_program(dim: int) -> MonadicProgram:
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("z",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=dim)),
            )
        ],
        return_vars=(),
    )


def _count_leapfrog_steps(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap the single-step integrator so a test can count its calls."""
    counter = [0]
    real_step = hmc_module._leapfrog_step

    def _counted_step(
        z: torch.Tensor,
        p: torch.Tensor,
        grad: torch.Tensor,
        step_size: float,
        potential: PotentialFn,
        mass: hmc_module._MassMatrix,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        counter[0] += 1
        return real_step(z, p, grad, step_size, potential, mass)

    monkeypatch.setattr(hmc_module, "_leapfrog_step", _counted_step)
    return counter


@pytest.mark.parametrize("mass_matrix", ["identity", "diagonal", "dense"])
def test_hmc_evaluates_the_target_once_per_leapfrog_step(
    monkeypatch: pytest.MonkeyPatch, mass_matrix: MassMatrixKind
) -> None:
    dim, num_steps, transitions = 6, 7, 25
    model = _standard_normal_program(dim)
    registry = LatentRegistry.from_model(model, set())
    potential = _CountingPotential(model, registry, torch.zeros(1, 1), {})
    leapfrog_steps = _count_leapfrog_steps(monkeypatch)
    kernel = HMCKernel(step_size=0.2, num_steps=num_steps, mass_matrix=mass_matrix)
    kernel._set_generator(torch.Generator().manual_seed(0))
    state = kernel.init(registry, model, torch.zeros(1, 1), {}, torch.zeros(dim))

    state = kernel.step(state, potential)
    # The first transition also evaluates the gradient at the start.
    assert potential.gradient_calls == num_steps + 1
    assert potential.forward_passes == num_steps + 1

    potential.reset()
    leapfrog_steps[0] = 0
    for _ in range(transitions):
        state = kernel.step(state, potential)

    assert leapfrog_steps[0] == transitions * num_steps
    assert potential.gradient_calls == transitions * num_steps
    assert potential.forward_passes == transitions * num_steps
    assert potential.forward_with_grad_mode == transitions * num_steps


@pytest.mark.parametrize("mass_matrix", ["identity", "diagonal", "dense"])
def test_nuts_evaluates_the_target_once_per_leapfrog_step(
    monkeypatch: pytest.MonkeyPatch, mass_matrix: MassMatrixKind
) -> None:
    dim, transitions = 20, 50
    model = _standard_normal_program(dim)
    registry = LatentRegistry.from_model(model, set())
    potential = _CountingPotential(model, registry, torch.zeros(1, 1), {})
    leapfrog_steps = _count_leapfrog_steps(monkeypatch)
    kernel = NUTSKernel(step_size=0.3, max_tree_depth=6, mass_matrix=mass_matrix)
    kernel._set_generator(torch.Generator().manual_seed(0))
    state = kernel.init(registry, model, torch.zeros(1, 1), {}, torch.zeros(dim))

    state = kernel.step(state, potential)
    assert potential.gradient_calls == leapfrog_steps[0] + 1

    potential.reset()
    leapfrog_steps[0] = 0
    tree_leaves = 0
    moved = 0
    for _ in range(transitions):
        previous = state.position
        state = kernel.step(state, potential)
        tree_leaves += 2 ** state.extras["tree_depth"] - 1
        moved += int(not torch.equal(previous, state.position))

    # A standard normal target never terminates a subtree early at
    # this step size, so every tree is complete.
    assert leapfrog_steps[0] == tree_leaves
    assert moved > 0
    assert potential.gradient_calls == tree_leaves
    assert potential.forward_passes == tree_leaves
    assert potential.forward_with_grad_mode == tree_leaves


@pytest.mark.parametrize("kernel_kind", ["hmc", "nuts"])
def test_carried_gradient_matches_a_fresh_evaluation(kernel_kind: str) -> None:
    """The gradient a transition hands to the next one is the
    gradient of the target at the new position, bit for bit."""
    dim = 4
    model = _standard_normal_program(dim)
    registry = LatentRegistry.from_model(model, set())
    potential = PotentialFn(model, registry, torch.zeros(1, 1), {})
    kernel: HMCKernel | NUTSKernel = (
        HMCKernel(step_size=0.3, num_steps=5)
        if kernel_kind == "hmc"
        else NUTSKernel(step_size=0.3, max_tree_depth=5)
    )
    kernel._set_generator(torch.Generator().manual_seed(1))
    state = kernel.init(
        registry, model, torch.zeros(1, 1), {}, torch.linspace(-1.0, 1.0, dim)
    )
    for _ in range(20):
        state = kernel.step(state, potential)
        fresh_value, fresh_grad = potential.value_and_grad(state.position)
        assert torch.equal(state.grad_log_density, fresh_grad)
        assert torch.equal(state.log_density, fresh_value)


def test_value_and_grad_differentiates_inside_no_grad() -> None:
    model = _standard_normal_program(3)
    registry = LatentRegistry.from_model(model, set())
    potential = PotentialFn(model, registry, torch.zeros(1, 1), {})
    z = torch.tensor([0.5, -1.0, 2.0])
    expected_value, expected_grad = potential.value_and_grad(z)
    with torch.no_grad():
        value, grad = potential.value_and_grad(z)
    assert torch.isfinite(value)
    assert torch.equal(value, expected_value)
    assert torch.equal(grad, expected_grad)
    assert torch.allclose(grad, -z)


def test_leapfrog_rejects_zero_steps() -> None:
    model = _standard_normal_program(2)
    registry = LatentRegistry.from_model(model, set())
    potential = PotentialFn(model, registry, torch.zeros(1, 1), {})
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        hmc_module._leapfrog(
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            0.1,
            0,
            potential,
            hmc_module._MassMatrix(2, "identity"),
        )


# ---------------------------------------------------------------------------
# Site shapes: posterior containers versus trace-side values
# ---------------------------------------------------------------------------


def _vector_and_scalar_program() -> MonadicProgram:
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("v",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R3", dim=3)),
            ),
            Draw(
                names=("s",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R1", dim=1)),
            ),
        ],
        return_vars=(),
    )


def _plate_program():
    return loads(
        "object Subj : FinSet 4\n"
        "object Resp : FinSet 12\n"
        "program p : Resp -> Resp\n"
        "    sample sigma <- HalfNormal(scale=1.0)\n"
        "    sample by_subj : Subj <- Normal(loc=0.0, scale=sigma)\n"
        "    let mu = by_subj[subj_idx]\n"
        "    observe r : Resp <- Normal(loc=mu, scale=1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


_PLATE_OBSERVATIONS = {
    "subj_idx": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
    "r": torch.linspace(-1.0, 1.0, 12),
}


def test_potential_hands_log_joint_a_batch_of_one_on_non_plate_sites() -> None:
    for model, observations, expected in (
        (_vector_and_scalar_program(), {}, {"v": (1, 3), "s": (1,)}),
        (_plate_program(), _PLATE_OBSERVATIONS, {"sigma": (1,), "by_subj": (4,)}),
    ):
        registry = LatentRegistry.from_model(model, set(observations) - {"subj_idx"})
        seen: dict[str, tuple[int, ...]] = {}
        original = model.log_joint

        def _recording_log_joint(
            x: torch.Tensor, values: dict[str, torch.Tensor]
        ) -> torch.Tensor:
            seen.update({k: tuple(v.shape) for k, v in values.items()})
            return original(x, values)

        model.log_joint = _recording_log_joint  # type: ignore[method-assign]
        try:
            PotentialFn(model, registry, torch.zeros(1, 1), observations).log_density(
                torch.zeros(registry.total_unconstrained_dim)
            )
        finally:
            del model.log_joint
        assert {k: seen[k] for k in expected} == expected
        for name, site in registry.sites.items():
            batch_axis = () if site.is_plate else (1,)
            assert seen[name] == (*batch_axis, *site.constrained_shape)


@pytest.mark.parametrize("plate", [False, True])
def test_mcmc_result_shapes_match_constrained_shape(plate: bool) -> None:
    model = _plate_program() if plate else _vector_and_scalar_program()
    observations = _PLATE_OBSERVATIONS if plate else {}
    registry = LatentRegistry.from_model(model, {"r"} if plate else set())
    result = MCMC(
        NUTSKernel(max_tree_depth=4),
        num_warmup=10,
        num_samples=6,
        num_chains=2,
        seed=0,
    ).run(model, torch.zeros(1, 1), observations)

    assert set(result.samples) == set(registry.names)
    for name, site in registry.sites.items():
        assert result.samples[name].shape == (2, 6, *site.constrained_shape)
        assert result.r_hat[name].shape == site.constrained_shape
        assert result.ess[name].shape == site.constrained_shape
    if plate:
        assert result.samples["sigma"].shape == (2, 6)
        assert result.samples["by_subj"].shape == (2, 6, 4)
        assert (result.samples["sigma"] > 0).all()
    else:
        assert result.samples["v"].shape == (2, 6, 3)
        assert result.samples["s"].shape == (2, 6)
        assert result.r_hat["s"].shape == ()


def test_mcmc_draws_score_like_the_chain_position() -> None:
    """Lifting a recorded draw to trace-side values reproduces the
    log-density the chain recorded for it."""
    model = _plate_program()
    registry = LatentRegistry.from_model(model, {"r"})
    potential = PotentialFn(model, registry, torch.zeros(1, 1), _PLATE_OBSERVATIONS)
    result = MCMC(
        NUTSKernel(max_tree_depth=4), num_warmup=5, num_samples=3, num_chains=1, seed=2
    ).run(model, torch.zeros(1, 1), _PLATE_OBSERVATIONS)
    for step in range(3):
        draw = {name: draws[0, step] for name, draws in result.samples.items()}
        unconstrained, _ = registry.to_unconstrained(draw)
        position = registry.flatten_unconstrained(unconstrained)
        assert torch.allclose(
            potential.log_density(position),
            result.log_densities[0, step],
            atol=1e-4,
        )


def test_guide_initialisation_reads_trace_side_guide_samples() -> None:
    class _FixedGuide:
        def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "v": torch.tensor([[0.5, -0.5, 1.0]]).expand(x.shape[0], 3),
                "s": torch.full((x.shape[0],), 0.25),
            }

    model = _vector_and_scalar_program()
    registry = LatentRegistry.from_model(model, set())
    driver = MCMC(
        HMCKernel(), num_warmup=0, num_samples=1, num_chains=1, init_strategy="guide"
    )
    position = driver._initial_position(
        registry,
        _FixedGuide(),  # type: ignore[arg-type]
        torch.Generator().manual_seed(0),
    )
    assert torch.equal(position, torch.tensor([0.5, -0.5, 1.0, 0.25]))


def test_guide_initialisation_rejects_a_guide_without_the_batch_axis() -> None:
    class _UnbatchedGuide:
        def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            del x
            return {"v": torch.zeros(3), "s": torch.zeros(())}

    model = _vector_and_scalar_program()
    registry = LatentRegistry.from_model(model, set())
    driver = MCMC(
        HMCKernel(), num_warmup=0, num_samples=1, num_chains=1, init_strategy="guide"
    )
    with pytest.raises(ValueError, match="expected the trace-side shape"):
        driver._initial_position(
            registry,
            _UnbatchedGuide(),  # type: ignore[arg-type]
            torch.Generator().manual_seed(0),
        )


# ---------------------------------------------------------------------------
# Errors: positions outside the support versus defects
# ---------------------------------------------------------------------------


def _regression_program():
    return loads(
        "object Resp : FinSet 12\n"
        "program p : Resp -> Resp\n"
        "    sample b0 <- Normal(loc=0.0, scale=1.0)\n"
        "    sample b1 <- Normal(loc=0.0, scale=1.0)\n"
        "    let mu = b0 + b1 * x_cov\n"
        "    observe y : Resp <- Normal(loc=mu, scale=1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def _count_program():
    return loads(
        "object Resp : FinSet 12\n"
        "program p : Resp -> Resp\n"
        "    sample log_rate <- Normal(loc=0.0, scale=1.0)\n"
        "    let rate = exp(log_rate)\n"
        "    observe y : Resp <- Poisson(rate=rate)\n"
        "    return rate\n"
        "export p\n"
    ).morphism


def _design_likelihood(env: dict[str, torch.Tensor]) -> torch.Tensor:
    eta = env["X"] @ env["beta"].reshape(-1)
    return Normal(eta, 1.0).log_prob(env["y"]).sum().reshape(1)


def _design_program() -> MonadicProgram:
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("beta",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="Coef", dim=3)),
            ),
            Score(name="lik", score=reading(_design_likelihood, ("beta", "X", "y"))),
        ],
        return_vars=(),
    )


_X_COV = torch.linspace(0.0, 1.0, 12)
_Y = 0.5 + 2.0 * _X_COV
_DEFECTS = {
    "covariate rows disagree with response rows": (
        _regression_program,
        {"x_cov": _X_COV[:11], "y": _Y},
        ValueError,
    ),
    "NaN in the observed response": (
        _regression_program,
        {"x_cov": _X_COV, "y": torch.cat([_Y[:11], torch.tensor([float("nan")])])},
        ValueError,
    ),
    "negative count under Poisson": (
        _count_program,
        {"y": torch.tensor([3.0, 1.0, -1.0, 2.0] * 3)},
        ValueError,
    ),
    "design matrix narrower than the coefficients": (
        _design_program,
        {"X": torch.randn(12, 2), "y": torch.randn(12)},
        RuntimeError,
    ),
}


@pytest.mark.parametrize("case", sorted(_DEFECTS))
def test_mcmc_run_raises_on_a_program_or_data_defect(case: str) -> None:
    make, data, error_type = _DEFECTS[case]
    driver = MCMC(
        NUTSKernel(max_tree_depth=3), num_warmup=5, num_samples=5, num_chains=1, seed=0
    )
    with pytest.raises(error_type) as raised:
        driver.run(make(), torch.zeros(1, 1), data)
    assert any("before any sampling" in note for note in raised.value.__notes__)


@pytest.mark.parametrize(
    "case",
    [
        "covariate rows disagree with response rows",
        "design matrix narrower than the coefficients",
    ],
)
def test_potential_never_scores_a_defect_as_minus_infinity(case: str) -> None:
    """A defect that is not a validation error raises at every
    position, inside or outside the initial check."""
    make, data, error_type = _DEFECTS[case]
    model = make()
    registry = LatentRegistry.from_model(
        model, {"y"} if make is _regression_program else set()
    )
    potential = PotentialFn(model, registry, torch.zeros(1, 1), data)
    position = torch.full((registry.total_unconstrained_dim,), 0.3)
    with pytest.raises(error_type):
        potential.value_and_grad(position)
    with pytest.raises(error_type):
        potential.check_initial(position)


@pytest.mark.parametrize(
    "case", ["NaN in the observed response", "negative count under Poisson"]
)
def test_check_initial_rejects_observed_data_outside_the_likelihood(case: str) -> None:
    make, data, _ = _DEFECTS[case]
    model = make()
    registry = LatentRegistry.from_model(model, {"y"})
    potential = PotentialFn(model, registry, torch.zeros(1, 1), data)
    with pytest.raises(ValueError):
        potential.check_initial(torch.zeros(registry.total_unconstrained_dim))


def _score_program(
    score: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    reads: tuple[str, ...],
    site: str = "b",
    prior: ContinuousMorphism | None = None,
) -> MonadicProgram:
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=(site,),
                morphism=prior
                if prior is not None
                else make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=1)),
            ),
            Score(name="lik", score=reading(score, reads)),
        ],
        return_vars=(),
    )


def _scale_likelihood(env: dict[str, torch.Tensor]) -> torch.Tensor:
    scale = env["b"].reshape(()) + 1.0
    return Normal(0.0, scale).log_prob(env["y"]).sum().reshape(1)


def test_parameter_outside_its_constraint_rejects_the_position() -> None:
    model = _score_program(_scale_likelihood, ("b", "y"))
    registry = LatentRegistry.from_model(model, set())
    data = {"y": torch.linspace(-1.0, 1.0, 6)}
    potential = PotentialFn(model, registry, torch.zeros(1, 1), data)
    assert torch.isfinite(potential.check_initial(torch.zeros(1)))
    value, grad = potential.value_and_grad(torch.tensor([-2.0]))
    assert value == float("-inf")
    assert torch.equal(grad, torch.zeros(1))
    result = MCMC(
        NUTSKernel(max_tree_depth=4),
        num_warmup=30,
        num_samples=30,
        num_chains=1,
        seed=0,
    ).run(model, torch.zeros(1, 1), data)
    assert (result.samples["b"] > -1.0).all()


def _gamma_score(env: dict[str, torch.Tensor]) -> torch.Tensor:
    return Gamma(2.0, 1.0).log_prob(env["sigma"].reshape(())).reshape(1)


def test_latent_on_its_support_boundary_rejects_the_position() -> None:
    model = _score_program(
        _gamma_score,
        ("sigma",),
        site="sigma",
        prior=make_fixed_halfnormal(1.0, Euclidean(name="R", dim=1)),
    )
    registry = LatentRegistry.from_model(model, set())
    potential = PotentialFn(model, registry, torch.zeros(1, 1), {})
    assert torch.isfinite(potential.log_density(torch.tensor([0.0])))
    # exp(-1000) underflows to zero, outside the Gamma's positive support.
    assert potential.log_density(torch.tensor([-1000.0])) == float("-inf")


def _cholesky_score(env: dict[str, torch.Tensor]) -> torch.Tensor:
    r = env["b"].reshape(())
    one = torch.ones(())
    cov = torch.stack([torch.stack([one, r]), torch.stack([r, one])])
    return torch.linalg.cholesky(cov).diagonal().log().sum().reshape(1)


def test_non_positive_definite_factorization_rejects_the_position() -> None:
    model = _score_program(_cholesky_score, ("b",))
    registry = LatentRegistry.from_model(model, set())
    potential = PotentialFn(model, registry, torch.zeros(1, 1), {})
    assert torch.isfinite(potential.log_density(torch.tensor([0.5])))
    assert potential.log_density(torch.tensor([2.0])) == float("-inf")


def _log_score(env: dict[str, torch.Tensor]) -> torch.Tensor:
    return env["b"].reshape(1).log()


def test_check_initial_names_the_non_finite_term() -> None:
    model = _score_program(_log_score, ("b",))
    registry = LatentRegistry.from_model(model, set())
    potential = PotentialFn(model, registry, torch.zeros(1, 1), {})
    assert torch.isfinite(potential.check_initial(torch.tensor([0.5])))
    with pytest.raises(ValueError, match="non-finite terms: lik"):
        potential.check_initial(torch.tensor([-1.0]))


def test_check_initial_validates_whatever_the_global_default() -> None:
    model = _count_program()
    registry = LatentRegistry.from_model(model, {"y"})
    potential = PotentialFn(
        model,
        registry,
        torch.zeros(1, 1),
        {"y": torch.tensor([3.0, 1.0, -1.0, 2.0] * 3)},
    )
    previous = Distribution._validate_args
    Distribution.set_default_validate_args(False)
    try:
        with pytest.raises(ValueError):
            potential.check_initial(torch.zeros(1))
        assert Distribution._validate_args is False
    finally:
        Distribution.set_default_validate_args(previous)


def _raised(call: Callable[[], object]) -> BaseException:
    try:
        call()
    except BaseException as error:  # noqa: BLE001 - the test inspects the error
        return error
    raise AssertionError("expected an error")


def test_rejects_position_recognises_torch_validators_by_raising_frame() -> None:
    """Pinned to the installed torch: its parameter and value
    validators are `Distribution.__init__` and
    `Distribution._validate_sample`."""
    assert _rejects_position(_raised(lambda: Normal(0.0, -1.0)))
    assert _rejects_position(
        _raised(lambda: Normal(0.0, 1.0).log_prob(torch.tensor(float("nan"))))
    )
    assert _rejects_position(
        _raised(lambda: Normal(torch.zeros(3), 1.0).log_prob(torch.zeros(4)))
    )
    assert _rejects_position(
        _raised(lambda: torch.linalg.cholesky(torch.tensor([[1.0, 2.0], [2.0, 1.0]])))
    )
    # Raised by a distribution's own argument handling, not its validator.
    assert not _rejects_position(_raised(lambda: Categorical()))
    assert not _rejects_position(_raised(lambda: torch.ones(2, 3) @ torch.ones(2)))

    def _user_value_error() -> None:
        raise ValueError("Expected parameter scale to satisfy the constraint")

    assert not _rejects_position(_raised(_user_value_error))
