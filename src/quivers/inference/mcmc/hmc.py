"""Hamiltonian Monte Carlo and No-U-Turn Sampler.

HMC augments the latent vector :math:`z \\in \\mathbb{R}^D` with an
auxiliary momentum :math:`p \\sim \\mathcal{N}(0, M)` and runs
symplectic-integrator (leapfrog) trajectories on the joint
Hamiltonian

.. math::

    H(z, p) = -\\log \\tilde{p}(z) + \\tfrac{1}{2} p^\\top M^{-1} p,

where :math:`\\tilde{p}` is the Jacobian-corrected unconstrained-
space target. A Metropolis correction at the trajectory endpoint
makes the chain reversible. The two kernels in this module share
the leapfrog primitive and step-size/mass adaptation but differ
in trajectory selection:

* `HMCKernel` runs a fixed number of leapfrog steps per
  proposal (Neal 2011, `doi:10.1201/b10905-7
  <https://doi.org/10.1201/b10905-7>`_).
* `NUTSKernel` builds a tree of leapfrog steps and uses
  the No-U-Turn termination criterion to set the trajectory
  length adaptively (Hoffman-Gelman 2014,
  `doi:10.48550/arXiv.1111.4246
  <https://doi.org/10.48550/arXiv.1111.4246>`_; Betancourt 2017's
  efficient generalized NUTS,
  `doi:10.48550/arXiv.1701.02434
  <https://doi.org/10.48550/arXiv.1701.02434>`_).

Both kernels operate on the flat unconstrained vector supplied by
the `quivers.inference.registry.LatentRegistry`. The
gradient is computed via `torch.autograd.grad` against the
[`quivers.inference.mcmc.kernel.PotentialFn`][quivers.inference.mcmc.kernel.PotentialFn].
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.mcmc.adapt import (
    DualAveraging,
    WelfordCovariance,
)
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
)
from quivers.inference.registry import LatentRegistry


MassMatrixKind = Literal["identity", "diagonal", "dense"]
"""The mass-matrix structure an HMC or NUTS kernel adapts during warmup:
``"identity"`` keeps the unit metric, ``"diagonal"`` adapts per-coordinate
variances, and ``"dense"`` adapts a full covariance."""


class _MassMatrix:
    """Mass-matrix container handling momentum sampling and
    inverse-mass-times-momentum products for the three supported
    forms.

    Stored as the *inverse* mass matrix (i.e. the covariance of the
    momentum-resampling distribution). Identity is a no-op,
    diagonal is a vector of inverse variances, dense is the
    Cholesky factor of the dense inverse mass.
    """

    def __init__(self, dim: int, kind: MassMatrixKind) -> None:
        self.dim = dim
        self.kind = kind
        if kind == "identity":
            self.inv_diag: torch.Tensor | None = None
            self.inv_chol: torch.Tensor | None = None
        elif kind == "diagonal":
            self.inv_diag = torch.ones(dim)
            self.inv_chol = None
        elif kind == "dense":
            self.inv_diag = None
            self.inv_chol = torch.eye(dim)
        else:
            raise ValueError(
                f"_MassMatrix: kind must be one of identity, "
                f"diagonal, dense; got {kind!r}"
            )

    def sample_momentum(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw :math:`p \\sim \\mathcal{N}(0, M)` (note: not
        :math:`M^{-1}`; momentum precision is the *mass* matrix)."""
        if self.kind == "identity":
            return torch.randn(self.dim, generator=generator)
        if self.kind == "diagonal":
            assert self.inv_diag is not None
            # M = diag(1/inv_diag); p ~ N(0, M) → p_i ~ N(0, 1/inv_diag_i).
            return torch.randn(self.dim, generator=generator) / self.inv_diag.sqrt()
        # Dense: M = (L L^T)^{-1} with L = chol(M^{-1}). Sample by
        # solving L^T p = ε for ε ~ N(0, I).
        assert self.inv_chol is not None
        eps = torch.randn(self.dim, generator=generator)
        return torch.linalg.solve_triangular(
            self.inv_chol.t(), eps.unsqueeze(-1), upper=True
        ).squeeze(-1)

    def kinetic(self, p: torch.Tensor) -> torch.Tensor:
        """Return :math:`\\tfrac{1}{2} p^\\top M^{-1} p`."""
        if self.kind == "identity":
            return 0.5 * (p * p).sum()
        if self.kind == "diagonal":
            assert self.inv_diag is not None
            return 0.5 * (self.inv_diag * p * p).sum()
        assert self.inv_chol is not None
        # M^{-1} = L L^T; p^T M^{-1} p = ||L^T p||²
        u = self.inv_chol.t() @ p
        return 0.5 * (u * u).sum()

    def inv_times(self, p: torch.Tensor) -> torch.Tensor:
        """Return :math:`M^{-1} p`."""
        if self.kind == "identity":
            return p
        if self.kind == "diagonal":
            assert self.inv_diag is not None
            return self.inv_diag * p
        assert self.inv_chol is not None
        return self.inv_chol @ (self.inv_chol.t() @ p)

    def set_inverse(self, inv_mass: torch.Tensor) -> None:
        """Install an adapted inverse mass matrix.

        ``inv_mass`` is either a length-``dim`` vector (diagonal
        kind) or a ``dim×dim`` PSD matrix (dense kind). Cached as a
        Cholesky factor for the dense case.
        """
        if self.kind == "identity":
            raise RuntimeError(
                "_MassMatrix: cannot set inverse on identity mass; "
                "construct with kind != 'identity'"
            )
        if self.kind == "diagonal":
            if inv_mass.shape != (self.dim,):
                raise ValueError(
                    f"_MassMatrix.set_inverse(diagonal): expected "
                    f"shape ({self.dim},); got {tuple(inv_mass.shape)}"
                )
            self.inv_diag = inv_mass.clamp(min=1e-12)
        else:
            if inv_mass.shape != (self.dim, self.dim):
                raise ValueError(
                    f"_MassMatrix.set_inverse(dense): expected shape "
                    f"({self.dim}, {self.dim}); got {tuple(inv_mass.shape)}"
                )
            sym = 0.5 * (inv_mass + inv_mass.t())
            sym = sym + 1e-8 * torch.eye(self.dim)
            self.inv_chol = torch.linalg.cholesky(sym)


def _leapfrog(
    z: torch.Tensor,
    p: torch.Tensor,
    grad: torch.Tensor,
    step_size: float,
    n_steps: int,
    potential: PotentialFn,
    mass: _MassMatrix,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ``n_steps`` symplectic leapfrog updates.

    Returns the final ``(z, p, log_density, grad_log_density)`` so
    the caller can compute the Metropolis acceptance from a single
    additional log-density evaluation.
    """
    z_curr = z
    p_curr = p
    g_curr = grad
    for _ in range(n_steps):
        p_half = p_curr + 0.5 * step_size * g_curr
        z_curr = z_curr + step_size * mass.inv_times(p_half)
        _, g_curr = potential.value_and_grad(z_curr)
        p_curr = p_half + 0.5 * step_size * g_curr
    ld_curr = potential.log_density(z_curr.detach()).detach()
    return z_curr, p_curr, ld_curr, g_curr


class HMCKernel(MCMCKernel):
    """Hamiltonian Monte Carlo kernel with fixed trajectory length.

    Parameters
    ----------
    step_size : float
        Leapfrog step size. Adapted during warmup when
        `adapt_step_size` is true.
    num_steps : int
        Leapfrog steps per proposal.
    mass_matrix : {"identity", "diagonal", "dense"}
        Mass-matrix shape. Diagonal / dense are adapted during
        warmup from the empirical covariance of warmup samples.
    target_accept : float
        Target Metropolis acceptance for dual averaging. Default
        ``0.65`` (Beskos et al.'s HMC-optimal acceptance for
        product-form targets).
    divergence_threshold : float
        Energy-error threshold for marking a proposal as divergent.
        Divergent steps still respect Metropolis correctness but
        are reported separately so the user can spot pathological
        regions.
    """

    is_adapting: bool = False

    def __init__(
        self,
        step_size: float = 0.1,
        num_steps: int = 10,
        mass_matrix: MassMatrixKind = "identity",
        target_accept: float = 0.65,
        divergence_threshold: float = 1000.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
    ) -> None:
        if step_size <= 0:
            raise ValueError(f"HMCKernel: step_size must be > 0, got {step_size}")
        if num_steps < 1:
            raise ValueError(f"HMCKernel: num_steps must be >= 1, got {num_steps}")
        if not 0.0 < target_accept < 1.0:
            raise ValueError(
                f"HMCKernel: target_accept must be in (0, 1), got {target_accept}"
            )
        self._step_size = step_size
        self._num_steps = num_steps
        self._mass_kind = mass_matrix
        self._target_accept = target_accept
        self._divergence_threshold = divergence_threshold
        self._adapt_step_size = adapt_step_size
        self._adapt_mass_matrix = adapt_mass_matrix and mass_matrix != "identity"
        self._mass: _MassMatrix | None = None
        self._dual_avg: DualAveraging | None = None
        self._welford: WelfordCovariance | None = None

    @property
    def step_size(self) -> float:
        if self._dual_avg is not None and not self.is_adapting:
            return self._dual_avg.smoothed_step_size()
        if self._dual_avg is not None:
            return self._dual_avg.step_size()
        return self._step_size

    def init(
        self,
        registry: LatentRegistry,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        initial_position: torch.Tensor,
    ) -> KernelState:
        del model, x, observations
        D = registry.total_unconstrained_dim
        if initial_position.shape != (D,):
            raise ValueError(
                f"HMCKernel.init: initial_position must have shape "
                f"({D},); got {tuple(initial_position.shape)}"
            )
        if not torch.isfinite(initial_position).all():
            raise ValueError(
                "HMCKernel.init: initial_position must contain only finite values"
            )
        self.is_adapting = False
        self._mass = _MassMatrix(D, self._mass_kind)
        self._dual_avg = None
        self._welford = None
        if self._adapt_mass_matrix:
            self._welford = WelfordCovariance(
                D, regularise=True, diagonal=(self._mass_kind == "diagonal")
            )
        return KernelState(
            position=initial_position.clone(),
            log_density=torch.tensor(0.0),
            grad_log_density=torch.zeros(D),
        )

    def start_adaptation(self) -> None:
        super().start_adaptation()
        if self._adapt_step_size:
            self._dual_avg = DualAveraging(
                self._step_size, target_accept=self._target_accept
            )

    def stop_adaptation(self) -> None:
        super().stop_adaptation()
        # Freeze the smoothed step size and adapted mass matrix.
        if self._dual_avg is not None:
            # The smoothed step size is what subsequent steps use.
            pass
        if self._welford is not None and self._mass is not None:
            if self._welford.n >= 2:
                cov = self._welford.covariance()
                self._mass.set_inverse(cov)

    def step(
        self,
        state: KernelState,
        potential: PotentialFn,
    ) -> KernelState:
        assert self._mass is not None, "HMCKernel.init was not called"
        z0 = state.position
        # Re-evaluate the cached gradient if this is the first step.
        if state.step_count == 0:
            ld0, g0 = potential.value_and_grad(z0)
            state.log_density = ld0
            state.grad_log_density = g0
        ld0 = state.log_density
        g0 = state.grad_log_density
        p0 = self._mass.sample_momentum(generator=self._generator)
        h0 = -ld0 + self._mass.kinetic(p0)
        eps = self.step_size
        z1, p1, ld1, g1 = _leapfrog(
            z0, p0, g0, eps, self._num_steps, potential, self._mass
        )
        h1 = -ld1 + self._mass.kinetic(p1)
        delta_h = h1 - h0
        log_accept = -delta_h
        if not torch.isfinite(log_accept):
            log_accept = torch.tensor(-float("inf"))
        accept_prob = float(torch.exp(torch.clamp_max(log_accept, 0.0)))
        accept_prob = max(0.0, min(1.0, accept_prob))
        diverged = bool(
            not torch.isfinite(delta_h)
            or torch.abs(delta_h).item() > self._divergence_threshold
        )
        if (
            accept_prob >= 1.0
            or torch.rand((), generator=self._generator).item() < accept_prob
        ):
            new_position = z1.detach()
            new_log_density = ld1.detach()
            new_grad = g1.detach()
            accept_count_delta = 1
        else:
            new_position = z0
            new_log_density = ld0
            new_grad = g0
            accept_count_delta = 0
        if self.is_adapting and self._dual_avg is not None:
            self._dual_avg.update(accept_prob)
        if self.is_adapting and self._welford is not None:
            self._welford.update(new_position.detach())
        return KernelState(
            position=new_position,
            log_density=new_log_density,
            grad_log_density=new_grad,
            step_count=state.step_count + 1,
            accept_count=state.accept_count + accept_count_delta,
            diverged=diverged,
            extras={**state.extras, "accept_prob": accept_prob},
        )


# ---------------------------------------------------------------------------
# NUTS
# ---------------------------------------------------------------------------


class _NUTSBuildTreeResult:
    """Internal accumulator returned by the recursive tree builder."""

    __slots__ = (
        "z_minus",
        "p_minus",
        "grad_minus",
        "z_plus",
        "p_plus",
        "grad_plus",
        "z_proposal",
        "log_density_proposal",
        "log_weight",
        "n_proposals",
        "turning",
        "diverged",
        "sum_accept_prob",
        "n_accept_steps",
    )

    def __init__(
        self,
        z_minus: torch.Tensor,
        p_minus: torch.Tensor,
        grad_minus: torch.Tensor,
        z_plus: torch.Tensor,
        p_plus: torch.Tensor,
        grad_plus: torch.Tensor,
        z_proposal: torch.Tensor,
        log_density_proposal: torch.Tensor,
        log_weight: float,
        n_proposals: int,
        turning: bool,
        diverged: bool,
        sum_accept_prob: float,
        n_accept_steps: int,
    ) -> None:
        self.z_minus = z_minus
        self.p_minus = p_minus
        self.grad_minus = grad_minus
        self.z_plus = z_plus
        self.p_plus = p_plus
        self.grad_plus = grad_plus
        self.z_proposal = z_proposal
        self.log_density_proposal = log_density_proposal
        self.log_weight = log_weight
        self.n_proposals = n_proposals
        self.turning = turning
        self.diverged = diverged
        self.sum_accept_prob = sum_accept_prob
        self.n_accept_steps = n_accept_steps


def _uturn(
    z_minus: torch.Tensor,
    z_plus: torch.Tensor,
    p_minus: torch.Tensor,
    p_plus: torch.Tensor,
    mass: _MassMatrix,
) -> bool:
    """U-turn termination criterion (Hoffman-Gelman 2014 equation 8)
    using the mass-corrected momentum so the test is invariant to
    the mass matrix."""
    delta = z_plus - z_minus
    return bool(
        (delta * mass.inv_times(p_minus)).sum().item() < 0
        or (delta * mass.inv_times(p_plus)).sum().item() < 0
    )


class NUTSKernel(MCMCKernel):
    """No-U-Turn Sampler with multinomial sampling and the standard
    U-turn termination (Hoffman-Gelman 2014 algorithms 3 + 6,
    Betancourt 2017's generalized slice variant for multinomial
    sampling).

    Parameters
    ----------
    step_size : float
        Initial leapfrog step size; adapted via dual averaging.
    max_tree_depth : int
        Maximum tree doubling depth. Default ``10`` (Stan default).
    target_accept : float
        Target tree-averaged Metropolis acceptance for dual
        averaging. Default ``0.8``.
    divergence_threshold : float
        Energy-error threshold above which a leapfrog substep is
        marked divergent and terminates the tree on its branch.
    """

    is_adapting: bool = False

    def __init__(
        self,
        step_size: float = 0.1,
        max_tree_depth: int = 10,
        mass_matrix: MassMatrixKind = "diagonal",
        target_accept: float = 0.8,
        divergence_threshold: float = 1000.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
    ) -> None:
        if step_size <= 0:
            raise ValueError(f"NUTSKernel: step_size must be > 0, got {step_size}")
        if max_tree_depth < 1:
            raise ValueError(
                f"NUTSKernel: max_tree_depth must be >= 1, got {max_tree_depth}"
            )
        if not 0.0 < target_accept < 1.0:
            raise ValueError(
                f"NUTSKernel: target_accept must be in (0, 1), got {target_accept}"
            )
        self._step_size = step_size
        self._max_depth = max_tree_depth
        self._mass_kind = mass_matrix
        self._target_accept = target_accept
        self._divergence_threshold = divergence_threshold
        self._adapt_step_size = adapt_step_size
        self._adapt_mass_matrix = adapt_mass_matrix and mass_matrix != "identity"
        self._mass: _MassMatrix | None = None
        self._dual_avg: DualAveraging | None = None
        self._welford: WelfordCovariance | None = None

    @property
    def step_size(self) -> float:
        if self._dual_avg is not None and not self.is_adapting:
            return self._dual_avg.smoothed_step_size()
        if self._dual_avg is not None:
            return self._dual_avg.step_size()
        return self._step_size

    def init(
        self,
        registry: LatentRegistry,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        initial_position: torch.Tensor,
    ) -> KernelState:
        del model, x, observations
        D = registry.total_unconstrained_dim
        if initial_position.shape != (D,):
            raise ValueError(
                f"NUTSKernel.init: initial_position must have shape "
                f"({D},); got {tuple(initial_position.shape)}"
            )
        if not torch.isfinite(initial_position).all():
            raise ValueError(
                "NUTSKernel.init: initial_position must contain only finite values"
            )
        self.is_adapting = False
        self._mass = _MassMatrix(D, self._mass_kind)
        self._dual_avg = None
        self._welford = None
        if self._adapt_mass_matrix:
            self._welford = WelfordCovariance(
                D, regularise=True, diagonal=(self._mass_kind == "diagonal")
            )
        return KernelState(
            position=initial_position.clone(),
            log_density=torch.tensor(0.0),
            grad_log_density=torch.zeros(D),
        )

    def start_adaptation(self) -> None:
        super().start_adaptation()
        if self._adapt_step_size:
            self._dual_avg = DualAveraging(
                self._step_size, target_accept=self._target_accept
            )

    def stop_adaptation(self) -> None:
        super().stop_adaptation()
        if self._welford is not None and self._mass is not None:
            if self._welford.n >= 2:
                cov = self._welford.covariance()
                self._mass.set_inverse(cov)

    def _build_tree(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        grad: torch.Tensor,
        direction: int,
        depth: int,
        h0: float,
        potential: PotentialFn,
        eps: float,
    ) -> _NUTSBuildTreeResult:
        if depth == 0:
            # Base case: single leapfrog step.
            z1, p1, ld1, g1 = _leapfrog(
                z,
                p,
                grad,
                eps * direction,
                1,
                potential,
                self._mass,  # type: ignore[arg-type]
            )
            assert self._mass is not None
            h1 = float(-ld1 + self._mass.kinetic(p1))
            log_weight = -h1
            energy_err = h1 - h0
            accept_prob = math.exp(min(0.0, -energy_err))
            if not math.isfinite(accept_prob):
                accept_prob = 0.0
            diverged = (
                not math.isfinite(energy_err)
                or abs(energy_err) > self._divergence_threshold
            )
            n_proposals = 1
            return _NUTSBuildTreeResult(
                z_minus=z1,
                p_minus=p1,
                grad_minus=g1,
                z_plus=z1,
                p_plus=p1,
                grad_plus=g1,
                z_proposal=z1,
                log_density_proposal=ld1,
                log_weight=log_weight,
                n_proposals=n_proposals,
                turning=False,
                diverged=diverged,
                sum_accept_prob=accept_prob,
                n_accept_steps=1,
            )
        left = self._build_tree(z, p, grad, direction, depth - 1, h0, potential, eps)
        if left.turning or left.diverged:
            return left
        if direction == -1:
            right = self._build_tree(
                left.z_minus,
                left.p_minus,
                left.grad_minus,
                direction,
                depth - 1,
                h0,
                potential,
                eps,
            )
            z_minus = right.z_minus
            p_minus = right.p_minus
            grad_minus = right.grad_minus
            z_plus = left.z_plus
            p_plus = left.p_plus
            grad_plus = left.grad_plus
        else:
            right = self._build_tree(
                left.z_plus,
                left.p_plus,
                left.grad_plus,
                direction,
                depth - 1,
                h0,
                potential,
                eps,
            )
            z_minus = left.z_minus
            p_minus = left.p_minus
            grad_minus = left.grad_minus
            z_plus = right.z_plus
            p_plus = right.p_plus
            grad_plus = right.grad_plus
        # Multinomial choice between the two subtrees, weighted by
        # their log-density mass. ``log_total = log(e^a + e^b)``
        # via the stable max + log1p(exp(min - max)) form to avoid
        # log(0) when both terms underflow.
        log_w_left = left.log_weight
        log_w_right = right.log_weight
        if math.isfinite(log_w_left) and math.isfinite(log_w_right):
            a = max(log_w_left, log_w_right)
            b = min(log_w_left, log_w_right)
            log_total = a + math.log1p(math.exp(b - a))
        else:
            log_total = max(log_w_left, log_w_right)
        if math.isfinite(log_total):
            prob_right = math.exp(log_w_right - log_total)
        else:
            prob_right = 0.0
        if torch.rand((), generator=self._generator).item() < prob_right:
            z_prop = right.z_proposal
            ld_prop = right.log_density_proposal
        else:
            z_prop = left.z_proposal
            ld_prop = left.log_density_proposal
        assert self._mass is not None
        turning = (
            left.turning
            or right.turning
            or _uturn(z_minus, z_plus, p_minus, p_plus, self._mass)
        )
        diverged = left.diverged or right.diverged
        return _NUTSBuildTreeResult(
            z_minus=z_minus,
            p_minus=p_minus,
            grad_minus=grad_minus,
            z_plus=z_plus,
            p_plus=p_plus,
            grad_plus=grad_plus,
            z_proposal=z_prop,
            log_density_proposal=ld_prop,
            log_weight=log_total,
            n_proposals=left.n_proposals + right.n_proposals,
            turning=turning,
            diverged=diverged,
            sum_accept_prob=left.sum_accept_prob + right.sum_accept_prob,
            n_accept_steps=left.n_accept_steps + right.n_accept_steps,
        )

    def step(
        self,
        state: KernelState,
        potential: PotentialFn,
    ) -> KernelState:
        assert self._mass is not None, "NUTSKernel.init was not called"
        z = state.position
        if state.step_count == 0:
            ld, g = potential.value_and_grad(z)
            state.log_density = ld
            state.grad_log_density = g
        ld = state.log_density
        g = state.grad_log_density
        p = self._mass.sample_momentum(generator=self._generator)
        h0 = float(-ld + self._mass.kinetic(p))
        eps = self.step_size
        z_minus = z
        p_minus = p
        grad_minus = g
        z_plus = z
        p_plus = p
        grad_plus = g
        z_proposal = z
        log_density_proposal = ld
        log_weight = -h0
        depth = 0
        terminated = False
        sum_accept_prob = 0.0
        n_accept_steps = 0
        diverged = False
        while not terminated and depth < self._max_depth:
            direction = (
                1 if torch.rand((), generator=self._generator).item() > 0.5 else -1
            )
            if direction == -1:
                subtree = self._build_tree(
                    z_minus,
                    p_minus,
                    grad_minus,
                    direction,
                    depth,
                    h0,
                    potential,
                    eps,
                )
                z_minus = subtree.z_minus
                p_minus = subtree.p_minus
                grad_minus = subtree.grad_minus
            else:
                subtree = self._build_tree(
                    z_plus,
                    p_plus,
                    grad_plus,
                    direction,
                    depth,
                    h0,
                    potential,
                    eps,
                )
                z_plus = subtree.z_plus
                p_plus = subtree.p_plus
                grad_plus = subtree.grad_plus
            sum_accept_prob += subtree.sum_accept_prob
            n_accept_steps += subtree.n_accept_steps
            if not subtree.turning and not subtree.diverged:
                # Multinomial choice between current proposal and
                # the new subtree. ``log_total = log(e^a + e^b)``
                # via the numerically-stable logaddexp form:
                # ``max(a, b) + log1p(exp(min - max))``. This avoids
                # log(0) when both terms underflow to zero, which
                # happens routinely on ill-conditioned posteriors.
                if math.isfinite(log_weight) and math.isfinite(subtree.log_weight):
                    a = max(log_weight, subtree.log_weight)
                    b = min(log_weight, subtree.log_weight)
                    log_total = a + math.log1p(math.exp(b - a))
                else:
                    log_total = max(log_weight, subtree.log_weight)
                if math.isfinite(log_total):
                    prob_new = math.exp(subtree.log_weight - log_total)
                else:
                    prob_new = 0.0
                if torch.rand((), generator=self._generator).item() < prob_new:
                    z_proposal = subtree.z_proposal
                    log_density_proposal = subtree.log_density_proposal
                log_weight = log_total
            diverged = diverged or subtree.diverged
            terminated = (
                subtree.turning
                or subtree.diverged
                or _uturn(z_minus, z_plus, p_minus, p_plus, self._mass)
            )
            depth += 1
        accept_prob = (
            sum_accept_prob / float(n_accept_steps) if n_accept_steps > 0 else 0.0
        )
        # Evaluate gradient at the chosen proposal for the next step.
        accepted = not torch.equal(z_proposal, state.position)
        if accepted:
            _, new_grad = potential.value_and_grad(z_proposal)
        else:
            new_grad = g
        if self.is_adapting and self._dual_avg is not None:
            self._dual_avg.update(accept_prob)
        if self.is_adapting and self._welford is not None:
            self._welford.update(z_proposal.detach())
        return KernelState(
            position=z_proposal.detach(),
            log_density=log_density_proposal.detach(),
            grad_log_density=new_grad.detach(),
            step_count=state.step_count + 1,
            accept_count=state.accept_count + (1 if accepted else 0),
            diverged=diverged,
            extras={**state.extras, "accept_prob": accept_prob, "tree_depth": depth},
        )


__all__ = [
    "HMCKernel",
    "NUTSKernel",
    "MassMatrixKind",
]
