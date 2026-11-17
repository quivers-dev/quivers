"""Pathfinder variational inference: oracles for each of its algorithms.

Tolerances are fixed from floating-point error before the results were seen:
``1e-10`` for quantities that are exact in exact arithmetic on a
well-conditioned five-dimensional problem in double precision, and ``1e-12``
relative error for the ELBO, which repeats the same sums.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import didactic.api as dx
import numpy as np
import pytest
from scipy import stats
import torch

import quivers.inference.pathfinder as pathfinder_module
from quivers.inference.pathfinder import (
    LBFGSSettings,
    OptimizationPath,
    Pathfinder,
    bfgs_approximation,
    compact_factors,
    lbfgs_path,
    normal_approximation,
    psis_log_weights,
    recover_diagonal,
    single_path,
    stream,
)

ORACLES = Path(__file__).parent / "oracles"
DIM = 5
EXACT = LBFGSSettings(history_size=DIM, wolfe_c1=1e-10, wolfe_c2=1e-8)


class Gaussian:
    """Unnormalized ``N(mean, precision^-1)`` log density in float64."""

    dtype = torch.float64

    def __init__(self, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        root = torch.randn(DIM, DIM, generator=generator, dtype=torch.float64)
        self.precision = root @ root.T + DIM * torch.eye(DIM, dtype=torch.float64)
        self.mean = torch.randn(DIM, generator=generator, dtype=torch.float64)

    def log_density(self, z: torch.Tensor) -> torch.Tensor:
        delta = z - self.mean
        return -0.5 * delta @ self.precision @ delta

    def value_and_grad(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.log_density(z), -(self.precision @ (z - self.mean))

    def log_normalizer(self) -> float:
        return 0.5 * DIM * math.log(2 * math.pi) - 0.5 * float(
            torch.logdet(self.precision)
        )


class Banana:
    """A non-Gaussian density: ``x0 ~ N(0, 1)``, ``x1 | x0 ~ N(x0^2 / 2, 0.5)``."""

    dtype = torch.float64

    def log_density(self, z: torch.Tensor) -> torch.Tensor:
        return -0.5 * z[0] ** 2 - (z[1] - 0.5 * z[0] ** 2) ** 2

    def value_and_grad(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leaf = z.detach().clone().requires_grad_(True)
        value = self.log_density(leaf)
        (gradient,) = torch.autograd.grad(value, leaf)
        return value.detach(), gradient

    def numpy_log_density(self, draws: np.ndarray) -> np.ndarray:
        return -0.5 * draws[:, 0] ** 2 - (draws[:, 1] - 0.5 * draws[:, 0] ** 2) ** 2


def _bfgs_dense(
    alpha: np.ndarray, steps: np.ndarray, changes: np.ndarray
) -> np.ndarray:
    """Apply the BFGS inverse update to ``diag(alpha)`` pair by pair."""
    inverse = np.diag(alpha)
    eye = np.eye(alpha.shape[0])
    for s, z in zip(steps, changes, strict=True):
        rho = 1.0 / (s @ z)
        inverse = (eye - rho * np.outer(s, z)) @ inverse @ (
            eye - rho * np.outer(z, s)
        ) + rho * np.outer(s, s)
    return inverse


def test_gaussian_target_best_approximation_is_exact():
    target = Gaussian()
    initial = torch.full((DIM,), 1.5, dtype=torch.float64)
    run = single_path(
        target, initial, seed=3, settings=EXACT, num_elbo_draws=50, num_draws=20
    )
    approximation = bfgs_approximation(
        run.path, recover_diagonal(run.path), run.best_iteration, DIM
    )
    assert approximation is not None
    covariance = torch.linalg.inv(target.precision)
    assert torch.allclose(approximation.mean, target.mean, rtol=0, atol=1e-10)
    assert torch.allclose(approximation.covariance(), covariance, rtol=0, atol=1e-10)
    # With q = p, log p - log q is log Z at every draw.
    assert float(run.elbo.max()) == pytest.approx(target.log_normalizer(), abs=1e-10)
    assert torch.allclose(
        run.log_density - run.log_approx,
        torch.full((20,), target.log_normalizer(), dtype=torch.float64),
        rtol=0,
        atol=1e-10,
    )


def test_history_shorter_than_the_dimension_is_not_exact():
    target = Gaussian()
    short = LBFGSSettings(history_size=2, wolfe_c1=1e-10, wolfe_c2=1e-8)
    path = lbfgs_path(target, torch.full((DIM,), 1.5, dtype=torch.float64), short)
    recovery = recover_diagonal(path)
    errors = []
    for iteration in range(1, path.num_iterations + 1):
        approximation = bfgs_approximation(path, recovery, iteration, 2)
        assert approximation is not None
        difference = approximation.covariance() - torch.linalg.inv(target.precision)
        errors.append(float(difference.abs().max()))
    assert min(errors) > 1e-3


def test_compact_factors_equal_sequential_bfgs_updates_and_factor_the_covariance():
    generator = torch.Generator().manual_seed(11)
    alpha = torch.rand(DIM, generator=generator, dtype=torch.float64) + 0.5
    steps = torch.randn(3, DIM, generator=generator, dtype=torch.float64)
    root = torch.randn(DIM, DIM, generator=generator, dtype=torch.float64)
    hessian = root @ root.T + torch.eye(DIM, dtype=torch.float64)
    changes = steps @ hessian
    beta, gamma = compact_factors(alpha, steps.T, changes.T)
    dense = torch.diag(alpha) + beta @ gamma @ beta.T
    expected = _bfgs_dense(alpha.numpy(), steps.numpy(), changes.numpy())
    assert np.allclose(dense.numpy(), expected, rtol=0, atol=1e-10)
    position = torch.randn(DIM, generator=generator, dtype=torch.float64)
    gradient = torch.randn(DIM, generator=generator, dtype=torch.float64)
    approximation = normal_approximation(position, gradient, alpha, steps.T, changes.T)
    assert approximation is not None
    assert torch.allclose(
        approximation.mean, position + dense @ gradient, rtol=0, atol=1e-10
    )
    factor = (
        approximation.sample(torch.eye(DIM, dtype=torch.float64)) - approximation.mean
    ).T
    assert torch.allclose(factor @ factor.T, dense, rtol=0, atol=1e-10)
    assert approximation.log_det == pytest.approx(float(torch.logdet(dense)), abs=1e-10)
    draws = approximation.sample(
        torch.randn(7, DIM, generator=generator, dtype=torch.float64)
    )
    reference = stats.multivariate_normal(approximation.mean.numpy(), expected).logpdf(
        draws.numpy()
    )
    assert np.allclose(approximation.log_prob(draws).numpy(), reference, atol=1e-10)


def test_a_wrong_inverse_hessian_update_fails_the_gaussian_oracle(monkeypatch):
    def without_curvature_diagonal(
        alpha: torch.Tensor, s: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta, gamma = compact_factors(alpha, s, z)
        m = s.shape[1]
        inverse = torch.linalg.solve_triangular(
            torch.triu(s.T @ z), torch.eye(m, dtype=torch.float64), upper=True
        )
        eta = torch.diagonal(s.T @ z)
        gamma = gamma.clone()
        gamma[m:, m:] -= inverse.T @ torch.diag(eta) @ inverse
        return beta, gamma

    monkeypatch.setattr(
        pathfinder_module, "compact_factors", without_curvature_diagonal
    )
    target = Gaussian()
    run = single_path(
        target,
        torch.full((DIM,), 1.5, dtype=torch.float64),
        seed=3,
        settings=EXACT,
        num_elbo_draws=50,
        num_draws=20,
    )
    covariance = torch.linalg.inv(target.precision)
    recovery = recover_diagonal(run.path)
    for iteration in range(1, run.path.num_iterations + 1):
        approximation = bfgs_approximation(run.path, recovery, iteration, DIM)
        if approximation is not None:
            error = (approximation.covariance() - covariance).abs().max()
            assert float(error) > 1e-3
    assert not float(run.elbo.max()) == pytest.approx(target.log_normalizer(), abs=1e-6)


def test_diagonal_recovery_follows_the_gilbert_lemarechal_update():
    generator = torch.Generator().manual_seed(5)
    positions = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    hessian = torch.diag(torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64))
    gradients = -(positions @ hessian)
    # The fourth step changes the gradient against the position: rejected.
    gradients[4] = gradients[3] + (positions[4] - positions[3])
    path = OptimizationPath(
        positions=positions,
        log_densities=torch.zeros(5, dtype=torch.float64),
        gradients=gradients,
        termination="maximum iterations",
        num_evaluations=5,
    )
    recovery = recover_diagonal(path)
    alpha = np.ones(3)
    expected_alphas, expected_accepted = [], []
    for step in range(4):
        s = (positions[step + 1] - positions[step]).numpy()
        z = (gradients[step] - gradients[step + 1]).numpy()
        if s @ z > 1e-12 * (z @ z):
            a, b, c = z @ (alpha * z), z @ s, s @ (s / alpha)
            alpha = 1 / (a / (b * alpha) + z**2 / b - a * s**2 / (b * c * alpha**2))
            expected_accepted.append(True)
        else:
            expected_accepted.append(False)
        expected_alphas.append(alpha)
    assert recovery.accepted.tolist() == expected_accepted == [True] * 3 + [False]
    assert np.allclose(recovery.alphas.numpy(), np.array(expected_alphas), atol=1e-12)


def test_elbo_estimates_agree_with_an_independent_computation():
    target = Banana()
    run = single_path(
        target,
        torch.tensor([1.8, -1.2], dtype=torch.float64),
        seed=21,
        settings=LBFGSSettings(history_size=3),
        num_elbo_draws=25,
        num_draws=10,
    )
    recovery = recover_diagonal(run.path)
    checked = 0
    for iteration in range(1, run.path.num_iterations + 1):
        approximation = bfgs_approximation(run.path, recovery, iteration, 3)
        if approximation is None:
            assert math.isinf(float(run.elbo[iteration - 1]))
            continue
        noise = torch.randn(25, 2, generator=stream(21, iteration), dtype=torch.float64)
        draws = approximation.sample(noise).numpy()
        covariance = approximation.covariance().numpy()
        log_q = stats.multivariate_normal(
            approximation.mean.numpy(), covariance
        ).logpdf(draws)
        independent = float(np.mean(target.numpy_log_density(draws) - log_q))
        assert float(run.elbo[iteration - 1]) == pytest.approx(independent, rel=1e-12)
        checked += 1
    assert checked >= 2
    assert run.best_iteration == int(torch.argmax(run.elbo)) + 1


@pytest.mark.parametrize("case", ["heavy", "light", "small"])
def test_pareto_smoothed_weights_match_loo(case: str):
    oracle = json.loads((ORACLES / "loo_psis_pathfinder.json").read_text())[case]
    log_weights, shape = psis_log_weights(
        torch.tensor(oracle["log_ratios"], dtype=torch.float64)
    )
    assert np.allclose(log_weights.numpy(), oracle["log_weights"], rtol=0, atol=1e-10)
    expected = oracle["pareto_k"]
    assert math.isinf(shape) if expected == "Inf" else shape == pytest.approx(expected)


def test_pareto_smoothing_keeps_impossible_draws_at_zero_weight():
    ratios = torch.tensor([0.1, -math.inf, 0.3, -0.2, 0.0], dtype=torch.float64)
    log_weights, _ = psis_log_weights(ratios)
    assert math.isinf(float(log_weights[1])) and float(log_weights[1]) < 0
    assert float(torch.logsumexp(log_weights, 0)) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="finite importance ratio"):
        psis_log_weights(torch.full((4,), -math.inf, dtype=torch.float64))


def test_multipath_resampling_is_seeded_and_draws_the_target():
    target = Gaussian()
    driver = Pathfinder(
        num_paths=3,
        num_draws=400,
        num_psis_draws=1200,
        lbfgs=EXACT,
        seed=4,
        save_single_paths=True,
    )
    first, second = driver.run_target(target, DIM), driver.run_target(target, DIM)
    assert torch.equal(first.samples["theta"], second.samples["theta"])
    other = driver.with_(seed=5).run_target(target, DIM)
    assert not torch.equal(first.samples["theta"], other.samples["theta"])
    assert first.resampled and first.num_samples == 1200 and first.num_paths == 3
    assert first.single_paths["theta"].shape == (3, 400, DIM)
    assert set(first.path_index.tolist()) == {0, 1, 2}
    # Each path is exact, so resampled draws are draws of the target: the
    # mean is within five standard errors with an effective size of at
    # least 400 (the per-path draw count), fixed before the run.
    draws = first.samples["theta"][0]
    errors = (draws.mean(0) - target.mean).abs()
    bound = 5 * torch.linalg.inv(target.precision).diagonal().sqrt() / math.sqrt(400)
    assert bool((errors < bound).all())
    single = Pathfinder(num_paths=1, num_draws=300, lbfgs=EXACT).run_target(target, DIM)
    assert not single.resampled and single.num_samples == 300
    assert math.isnan(single.pareto_k)
    pooled = driver.with_(psis_resample=False).run_target(target, DIM)
    assert not pooled.resampled and pooled.num_samples == 1200


def test_invalid_settings_and_failures_are_refused():
    with pytest.raises(dx.ValidationError, match="Wolfe"):
        LBFGSSettings(wolfe_c1=0.5, wolfe_c2=0.4)
    with pytest.raises(dx.ValidationError, match="history_size"):
        LBFGSSettings(history_size=0)
    with pytest.raises(dx.ValidationError, match="num_paths"):
        Pathfinder(num_paths=0)
    with pytest.raises(dx.ValidationError, match="one mapping per path"):
        Pathfinder(num_paths=2, initial_values=({"a": 1.0},))

    class Unbounded(Gaussian):
        def value_and_grad(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.tensor(math.inf, dtype=torch.float64), z

    with pytest.raises(ValueError, match="initial point"):
        Pathfinder(num_paths=1).run_target(Unbounded(), DIM)

    class Nowhere(Gaussian):
        """Gradients of a Gaussian, but no draw has a finite density."""

        def log_density(self, z: torch.Tensor) -> torch.Tensor:
            return torch.tensor(-math.inf, dtype=torch.float64)

        def value_and_grad(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return Gaussian.value_and_grad(Gaussian(), z)

    with pytest.raises(ValueError, match="every path failed"):
        Pathfinder(num_paths=2, num_draws=5).run_target(Nowhere(), DIM)
