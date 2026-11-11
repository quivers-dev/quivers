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
  proposal (Neal 2011, `doi:10.1201/b10905-6
  <https://doi.org/10.1201/b10905-6>`_).
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
    AdaptationMode,
    AdaptationSchedule,
    AdaptationWindow,
    DualAveraging,
    WelfordCovariance,
)
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
    StateValue,
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


def _leapfrog_step(
    z: torch.Tensor,
    p: torch.Tensor,
    grad: torch.Tensor,
    step_size: float,
    potential: PotentialFn,
    mass: _MassMatrix,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one leapfrog step with a single target evaluation.

    Parameters
    ----------
    z : torch.Tensor
        Flat unconstrained position, shape ``(D,)``.
    p : torch.Tensor
        Momentum, shape ``(D,)``.
    grad : torch.Tensor
        Gradient of the log-density at ``z``, shape ``(D,)``.
    step_size : float
        Signed leapfrog step size.
    potential : PotentialFn
        Target whose log-density and gradient drive the dynamics.
    mass : _MassMatrix
        Mass matrix defining the kinetic energy.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        The new ``(z, p, log_density, grad_log_density)``, where the
        log-density and its gradient come from one
        `PotentialFn.value_and_grad` call at the new position.
    """
    p_half = p + 0.5 * step_size * grad
    z_new = z + step_size * mass.inv_times(p_half)
    ld_new, g_new = potential.value_and_grad(z_new)
    p_new = p_half + 0.5 * step_size * g_new
    return z_new, p_new, ld_new, g_new


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

    Each step evaluates the target exactly once, through
    `PotentialFn.value_and_grad` at the new position. The gradient
    drives the closing momentum half-step, and the value from the
    last step is the endpoint log-density, so the caller can form
    the Metropolis energy without a further evaluation.

    Parameters
    ----------
    z : torch.Tensor
        Starting flat unconstrained position, shape ``(D,)``.
    p : torch.Tensor
        Starting momentum, shape ``(D,)``.
    grad : torch.Tensor
        Gradient of the log-density at ``z``, shape ``(D,)``.
    step_size : float
        Signed leapfrog step size.
    n_steps : int
        Number of leapfrog steps; at least one.
    potential : PotentialFn
        Target whose log-density and gradient drive the dynamics.
    mass : _MassMatrix
        Mass matrix defining the kinetic energy.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        The endpoint ``(z, p, log_density, grad_log_density)``, all
        detached from autograd.

    Raises
    ------
    ValueError
        If ``n_steps`` is less than one, since the endpoint
        log-density is the value of the last step's evaluation.
    """
    if n_steps < 1:
        raise ValueError(f"_leapfrog: n_steps must be >= 1, got {n_steps}")
    z_curr, p_curr, ld_curr, g_curr = _leapfrog_step(
        z, p, grad, step_size, potential, mass
    )
    for _ in range(n_steps - 1):
        z_curr, p_curr, ld_curr, g_curr = _leapfrog_step(
            z_curr, p_curr, g_curr, step_size, potential, mass
        )
    return z_curr, p_curr, ld_curr, g_curr


def _mapping(value: StateValue) -> dict[str, StateValue]:
    """Read a nested dictionary out of a state dictionary.

    Parameters
    ----------
    value : StateValue
        The entry.

    Returns
    -------
    dict[str, StateValue]
        The entry, as a dictionary.

    Raises
    ------
    ValueError
        If the entry is not a dictionary.
    """
    if not isinstance(value, dict):
        raise ValueError(
            f"expected a nested state dictionary, got {type(value).__name__}"
        )
    return value


_STEP_SIZE_INIT_LOG_ACCEPT = math.log(0.8)
"""The log energy-change threshold the step-size initialization
heuristic brackets, as in Stan's ``init_stepsize``."""


class _HamiltonianKernel(MCMCKernel):
    """Shared adaptation machinery of `HMCKernel` and `NUTSKernel`.

    Holds the mass matrix, the dual-averaging step-size adapter, and
    the Welford covariance estimate, and runs warmup in one of two
    modes. In ``"single"`` mode dual averaging starts from the
    configured step size, the covariance estimate accumulates every
    warmup position, and the metric is installed once, in
    `stop_adaptation`. In ``"windowed"`` mode the warmup follows an
    `AdaptationSchedule`: the step size is first initialized by
    `_initial_step_size` and dual averaging starts from it; every slow
    window accumulates a fresh covariance estimate, installs it as the
    metric at the window's end, and restarts dual averaging from a
    re-initialized step size; the terminal fast window then tunes the
    step size under the final metric.

    Parameters
    ----------
    name : str
        The kernel name errors report.
    step_size : float
        Initial leapfrog step size.
    mass_matrix : MassMatrixKind
        Mass-matrix structure.
    target_accept : float
        Target acceptance statistic for dual averaging.
    divergence_threshold : float
        Energy-error threshold for a divergence.
    adapt_step_size : bool
        Whether warmup adapts the step size.
    adapt_mass_matrix : bool
        Whether warmup adapts the metric; ignored for ``"identity"``.
    adaptation : AdaptationMode
        The warmup mode.
    adaptation_schedule : AdaptationSchedule | None
        The windowed schedule; the Stan defaults when ``None``.

    Raises
    ------
    ValueError
        If ``step_size`` is not positive, ``target_accept`` is outside
        ``(0, 1)``, or ``adaptation`` is not a known mode.
    """

    is_adapting: bool = False

    def __init__(
        self,
        name: str,
        step_size: float,
        mass_matrix: MassMatrixKind,
        target_accept: float,
        divergence_threshold: float,
        adapt_step_size: bool,
        adapt_mass_matrix: bool,
        adaptation: AdaptationMode,
        adaptation_schedule: AdaptationSchedule | None,
    ) -> None:
        if step_size <= 0:
            raise ValueError(f"{name}: step_size must be > 0, got {step_size}")
        if not 0.0 < target_accept < 1.0:
            raise ValueError(
                f"{name}: target_accept must be in (0, 1), got {target_accept}"
            )
        if adaptation not in ("windowed", "single"):
            raise ValueError(
                f"{name}: adaptation must be 'windowed' or 'single', got {adaptation!r}"
            )
        self._name = name
        self._step_size = step_size
        self._mass_kind: MassMatrixKind = mass_matrix
        self._target_accept = target_accept
        self._divergence_threshold = divergence_threshold
        self._adapt_step_size = adapt_step_size
        self._adapt_mass_matrix = adapt_mass_matrix and mass_matrix != "identity"
        self._adaptation: AdaptationMode = adaptation
        self._schedule = (
            adaptation_schedule
            if adaptation_schedule is not None
            else AdaptationSchedule()
        )
        self._mass: _MassMatrix | None = None
        self._dual_avg: DualAveraging | None = None
        self._welford: WelfordCovariance | None = None
        self._windows: tuple[AdaptationWindow, ...] = ()
        self._warmup_iteration = 0
        self._pending_step_size_init = False

    @property
    def adaptation(self) -> AdaptationMode:
        """The warmup mode, ``"windowed"`` or ``"single"``."""
        return self._adaptation

    @property
    def adaptation_schedule(self) -> AdaptationSchedule:
        """The schedule ``"windowed"`` warmup follows."""
        return self._schedule

    @property
    def step_size(self) -> float:
        """The step size the next transition uses.

        During warmup this is dual averaging's current iterate; after
        warmup it is the smoothed step size dual averaging froze; with
        step-size adaptation off it is the configured step size.
        """
        if self._dual_avg is not None and not self.is_adapting:
            return self._dual_avg.smoothed_step_size()
        if self._dual_avg is not None:
            return self._dual_avg.step_size()
        return self._step_size

    @property
    def inverse_mass_matrix(self) -> torch.Tensor:
        """The metric the kernel samples with.

        Returns
        -------
        torch.Tensor
            The inverse mass matrix: ``(D,)`` ones for an identity
            metric, the ``(D,)`` diagonal for a diagonal one, and the
            ``(D, D)`` matrix for a dense one.

        Raises
        ------
        RuntimeError
            If the kernel has not been initialized.
        """
        if self._mass is None:
            raise RuntimeError(f"{self._name}: init has not been called")
        if self._mass.kind == "identity":
            return torch.ones(self._mass.dim)
        if self._mass.kind == "diagonal":
            assert self._mass.inv_diag is not None
            return self._mass.inv_diag.clone()
        assert self._mass.inv_chol is not None
        return self._mass.inv_chol @ self._mass.inv_chol.t()

    def adaptation_windows(self, num_warmup: int) -> tuple[AdaptationWindow, ...]:
        """The windows a warmup of ``num_warmup`` iterations runs.

        In ``"windowed"`` mode with metric adaptation this is
        ``adaptation_schedule.windows(num_warmup)``. A kernel that does
        not adapt its metric, in either mode, runs one fast window over
        the whole warmup; a ``"single"``-mode kernel that adapts its
        metric runs one window that both tunes the step size and
        accumulates the metric estimate.

        Parameters
        ----------
        num_warmup : int
            Number of warmup iterations.

        Returns
        -------
        tuple[AdaptationWindow, ...]
            The windows, tiling ``[0, num_warmup)``.

        Raises
        ------
        ValueError
            If ``num_warmup`` is negative.
        """
        if num_warmup < 0:
            raise ValueError(
                f"{self._name}.adaptation_windows: num_warmup must be >= 0, "
                f"got {num_warmup}"
            )
        if self._adaptation == "windowed" and self._adapt_mass_matrix:
            return self._schedule.windows(num_warmup)
        if num_warmup == 0:
            return ()
        return (
            AdaptationWindow(
                start=0,
                stop=num_warmup,
                adapts_metric=self._adapt_mass_matrix,
            ),
        )

    def tune(self, population: torch.Tensor) -> None:
        """Set the metric from a population's spread.

        A diagonal or dense metric becomes the population's covariance,
        regularised as Stan regularises a warmup estimate; the metric is
        then held fixed through any adaptation that follows, which tunes
        only the step size. An identity metric is left as it is.

        Parameters
        ----------
        population : torch.Tensor
            Flat positions, shape ``(N, D)``, with ``N >= 2``.

        Raises
        ------
        ValueError
            If the population is not ``(N, D)`` with at least two rows.
        RuntimeError
            If the kernel has not been initialized.
        """
        if self._mass is None:
            raise RuntimeError(f"{self._name}.tune: init has not been called")
        if (
            population.dim() != 2
            or population.shape[1] != self._mass.dim
            or (population.shape[0] < 2)
        ):
            raise ValueError(
                f"{self._name}.tune: the population must have shape (N, "
                f"{self._mass.dim}) with N >= 2; got {tuple(population.shape)}"
            )
        if self._mass.kind == "identity":
            return
        estimate = WelfordCovariance(
            self._mass.dim, regularise=True, diagonal=self._mass.kind == "diagonal"
        )
        for row in population:
            estimate.update(row.detach())
        self._mass.set_inverse(estimate.covariance())
        self._welford = None

    def configuration(self) -> dict[str, StateValue]:
        """The kernel's hyperparameters.

        Returns
        -------
        dict[str, StateValue]
            The class, step size, metric kind, target acceptance,
            divergence threshold, adaptation switches and mode, and the
            schedule.
        """
        return {
            **super().configuration(),
            "step_size": self._step_size,
            "mass_matrix": self._mass_kind,
            "target_accept": self._target_accept,
            "divergence_threshold": self._divergence_threshold,
            "adapt_step_size": self._adapt_step_size,
            "adapt_mass_matrix": self._adapt_mass_matrix,
            "adaptation": self._adaptation,
            "schedule": [
                self._schedule.init_buffer,
                self._schedule.term_buffer,
                self._schedule.base_window,
            ],
        }

    def state_dict(self) -> dict[str, StateValue]:
        """The chain's adaptation state.

        Returns
        -------
        dict[str, StateValue]
            Whether the kernel is adapting, the metric, the dual-averaging
            and covariance-estimate states, the warmup windows, the
            position in them, and a pending step-size initialization.
        """
        assert self._mass is not None, f"{self._name}.init was not called"
        return {
            **super().state_dict(),
            "inv_diag": self._mass.inv_diag,
            "inv_chol": self._mass.inv_chol,
            "dual_avg": None
            if self._dual_avg is None
            else dict(self._dual_avg.state_dict()),
            "welford": None
            if self._welford is None
            else dict(self._welford.state_dict()),
            "windows": [[w.start, w.stop, w.adapts_metric] for w in self._windows],
            "warmup_iteration": self._warmup_iteration,
            "pending_step_size_init": self._pending_step_size_init,
        }

    def load_state_dict(self, state: dict[str, StateValue]) -> None:
        """Restore the adaptation state `state_dict` returned.

        Parameters
        ----------
        state : dict[str, StateValue]
            A dictionary `state_dict` produced, for a kernel of the same
            configuration, after `init`.

        Raises
        ------
        ValueError
            If an entry is missing or has the wrong type or shape.
        """
        super().load_state_dict(state)
        assert self._mass is not None, f"{self._name}.init was not called"
        inv_diag, inv_chol = state.get("inv_diag"), state.get("inv_chol")
        if self._mass.kind == "diagonal":
            if not isinstance(inv_diag, torch.Tensor) or inv_diag.shape != (
                self._mass.dim,
            ):
                raise ValueError(f"{self._name}.load_state_dict: bad inv_diag")
            self._mass.inv_diag = inv_diag.clone()
        if self._mass.kind == "dense":
            if not isinstance(inv_chol, torch.Tensor) or inv_chol.shape != (
                self._mass.dim,
                self._mass.dim,
            ):
                raise ValueError(f"{self._name}.load_state_dict: bad inv_chol")
            self._mass.inv_chol = inv_chol.clone()
        dual, welford = state.get("dual_avg"), state.get("welford")
        self._dual_avg = (
            None if dual is None else DualAveraging.from_state_dict(_mapping(dual))
        )
        self._welford = (
            None
            if welford is None
            else WelfordCovariance.from_state_dict(_mapping(welford))
        )
        windows = state.get("windows")
        iteration = state.get("warmup_iteration")
        pending = state.get("pending_step_size_init")
        if (
            not isinstance(windows, list)
            or not isinstance(iteration, int)
            or not isinstance(pending, bool)
        ):
            raise ValueError(f"{self._name}.load_state_dict: bad warmup schedule")
        parsed: list[AdaptationWindow] = []
        for entry in windows:
            if not isinstance(entry, list) or len(entry) != 3:
                raise ValueError(f"{self._name}.load_state_dict: bad window {entry}")
            start, stop, metric = entry
            if not (
                isinstance(start, int)
                and isinstance(stop, int)
                and isinstance(metric, bool)
            ):
                raise ValueError(f"{self._name}.load_state_dict: bad window {entry}")
            parsed.append(
                AdaptationWindow(start=start, stop=stop, adapts_metric=metric)
            )
        self._windows = tuple(parsed)
        self._warmup_iteration = iteration
        self._pending_step_size_init = pending

    def _initial_state(
        self, registry: LatentRegistry, initial_position: torch.Tensor
    ) -> KernelState:
        """Reset the adaptation state and build a chain's first state.

        Parameters
        ----------
        registry : LatentRegistry
            The registry whose flat vector the chain walks.
        initial_position : torch.Tensor
            The starting position, shape ``(D,)``.

        Returns
        -------
        KernelState
            The starting state; its log-density and gradient are
            evaluated by the first transition.

        Raises
        ------
        ValueError
            If ``initial_position`` does not have shape ``(D,)`` or
            holds a non-finite value.
        """
        D = registry.total_unconstrained_dim
        if initial_position.shape != (D,):
            raise ValueError(
                f"{self._name}.init: initial_position must have shape "
                f"({D},); got {tuple(initial_position.shape)}"
            )
        if not torch.isfinite(initial_position).all():
            raise ValueError(
                f"{self._name}.init: initial_position must contain only finite values"
            )
        self.is_adapting = False
        self._mass = _MassMatrix(D, self._mass_kind)
        self._dual_avg = None
        self._welford = None
        self._windows = ()
        self._warmup_iteration = 0
        self._pending_step_size_init = False
        if self._adapt_mass_matrix:
            self._welford = WelfordCovariance(
                D, regularise=True, diagonal=(self._mass_kind == "diagonal")
            )
        return KernelState(
            position=initial_position.clone(),
            log_density=torch.tensor(0.0),
            grad_log_density=torch.zeros(D),
        )

    def start_adaptation(self, num_warmup: int) -> None:
        """Enter warmup.

        In ``"single"`` mode dual averaging starts from the configured
        step size. In ``"windowed"`` mode the schedule for
        ``num_warmup`` iterations is laid out, and the first transition
        initializes the step size before dual averaging starts.

        Parameters
        ----------
        num_warmup : int
            Number of warmup transitions that follow.
        """
        super().start_adaptation(num_warmup)
        if self._adaptation == "single":
            if self._adapt_step_size:
                self._dual_avg = DualAveraging(
                    self._step_size, target_accept=self._target_accept
                )
            return
        self._windows = self.adaptation_windows(num_warmup)
        self._warmup_iteration = 0
        self._dual_avg = None
        self._pending_step_size_init = self._adapt_step_size
        if self._welford is not None:
            self._welford.reset()

    def stop_adaptation(self) -> None:
        """Freeze the adapted step size and metric.

        In ``"single"`` mode the covariance estimate of the whole warmup
        becomes the metric here; in ``"windowed"`` mode the last slow
        window already installed it.
        """
        super().stop_adaptation()
        if self._adaptation != "single":
            return
        if self._welford is not None and self._mass is not None:
            if self._welford.n >= 2:
                cov = self._welford.covariance()
                self._mass.set_inverse(cov)

    def _prepare_transition(self, state: KernelState, potential: PotentialFn) -> None:
        """Evaluate the first state and run a pending step-size
        initialization.

        Parameters
        ----------
        state : KernelState
            The state the transition starts from; its log-density and
            gradient are filled in on the chain's first transition.
        potential : PotentialFn
            The target.
        """
        self._evaluate_first(state, potential)
        if self.is_adapting and self._pending_step_size_init:
            self._restart_step_size(state, potential, self._step_size)

    def _restart_step_size(
        self, state: KernelState, potential: PotentialFn, step_size: float
    ) -> None:
        """Re-initialize the step size and restart dual averaging from it.

        Parameters
        ----------
        state : KernelState
            The state whose position the heuristic integrates from.
        potential : PotentialFn
            The target.
        step_size : float
            The step size the heuristic starts from.
        """
        initial = self._initial_step_size(state, potential, step_size)
        self._dual_avg = DualAveraging(initial, target_accept=self._target_accept)
        self._pending_step_size_init = False

    def _energy_change(
        self, state: KernelState, potential: PotentialFn, step_size: float
    ) -> float:
        """The Hamiltonian's decrease over one leapfrog step from a fresh
        momentum.

        Parameters
        ----------
        state : KernelState
            The state to integrate from.
        potential : PotentialFn
            The target.
        step_size : float
            The leapfrog step size.

        Returns
        -------
        float
            ``H(start) - H(end)``; ``-inf`` when the endpoint's energy
            is not finite.
        """
        assert self._mass is not None
        p = self._mass.sample_momentum(generator=self._generator)
        h0 = float(-state.log_density + self._mass.kinetic(p))
        _, p1, ld1, _ = _leapfrog(
            state.position,
            p,
            state.grad_log_density,
            step_size,
            1,
            potential,
            self._mass,
        )
        h1 = float(-ld1 + self._mass.kinetic(p1))
        if math.isnan(h1):
            h1 = math.inf
        return h0 - h1

    def _initial_step_size(
        self, state: KernelState, potential: PotentialFn, step_size: float
    ) -> float:
        """Bracket a step size by doubling or halving (Stan's
        ``init_stepsize``).

        One leapfrog step from ``state`` under a fresh momentum drawn
        from the current metric gives an energy change
        :math:`\\Delta H`. The heuristic doubles the step size while
        :math:`\\Delta H > \\log 0.8` (the step is accurate enough to
        lengthen) or halves it while :math:`\\Delta H < \\log 0.8`,
        drawing a fresh momentum for each trial, and returns the first
        step size at which the comparison flips. This is Hoffman and
        Gelman's Algorithm 4 with the metric in the kinetic energy and
        Stan's threshold; the momentum comes from the kernel's
        chain-local generator.

        Parameters
        ----------
        state : KernelState
            The state to integrate from, with its log-density and
            gradient evaluated.
        potential : PotentialFn
            The target.
        step_size : float
            The step size to start from.

        Returns
        -------
        float
            The bracketed step size.

        Raises
        ------
        RuntimeError
            If the step size grows past ``1e7`` (the target is likely
            improper) or underflows to zero (no step size gives a
            finite energy).
        """
        eps = step_size
        direction = (
            1
            if self._energy_change(state, potential, eps) > (_STEP_SIZE_INIT_LOG_ACCEPT)
            else -1
        )
        while True:
            delta = self._energy_change(state, potential, eps)
            if direction == 1 and not delta > _STEP_SIZE_INIT_LOG_ACCEPT:
                break
            if direction == -1 and not delta < _STEP_SIZE_INIT_LOG_ACCEPT:
                break
            eps = 2.0 * eps if direction == 1 else 0.5 * eps
            if eps > 1e7:
                raise RuntimeError(
                    f"{self._name}: the step-size initialization grew past 1e7, "
                    "so the posterior is likely improper; check the model"
                )
            if eps == 0.0:
                raise RuntimeError(
                    f"{self._name}: the step-size initialization underflowed to "
                    "zero, so no step size gives a finite energy from this "
                    "position; check the model"
                )
        return eps

    def _adapt(
        self,
        state: KernelState,
        potential: PotentialFn,
        accept_prob: float,
    ) -> None:
        """Fold one warmup transition into the adaptation state.

        Parameters
        ----------
        state : KernelState
            The state the transition produced.
        potential : PotentialFn
            The target, for a step-size re-initialization after a
            metric update.
        accept_prob : float
            The transition's acceptance statistic.
        """
        if not self.is_adapting:
            return
        if self._adaptation == "single":
            if self._dual_avg is not None:
                self._dual_avg.update(accept_prob)
            if self._welford is not None:
                self._welford.update(state.position.detach())
            return
        iteration = self._warmup_iteration
        self._warmup_iteration += 1
        if self._dual_avg is not None:
            self._dual_avg.update(accept_prob)
        window = next((w for w in self._windows if w.start <= iteration < w.stop), None)
        if window is None or not window.adapts_metric or self._welford is None:
            return
        self._welford.update(state.position.detach())
        if iteration + 1 != window.stop:
            return
        assert self._mass is not None
        self._mass.set_inverse(self._welford.covariance())
        self._welford.reset()
        if self._adapt_step_size:
            assert self._dual_avg is not None
            self._restart_step_size(state, potential, self._dual_avg.step_size())


class HMCKernel(_HamiltonianKernel):
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
        ``0.65``, near the asymptotic optimum derived for specified
        high-dimensional product targets [Beskos et al. (2013)](<https://doi.org/10.3150/12-BEJ414>). This
        is a tuning default, not a universal optimum.
    divergence_threshold : float
        Energy-error threshold for marking a proposal as divergent.
        Divergent steps still respect Metropolis correctness but
        are reported separately so the user can spot pathological
        regions.
    adapt_step_size : bool
        Whether warmup adapts the step size by dual averaging.
    adapt_mass_matrix : bool
        Whether warmup adapts a diagonal or dense metric.
    adaptation : {"windowed", "single"}
        The warmup mode (see `AdaptationMode`). Default ``"single"``.
        ``"windowed"`` keeps the initial transient out of the metric
        and tunes the step size under the metric it samples with; it
        suits warmups long enough for the schedule's buffers (150
        iterations under the defaults), since a shorter warmup leaves
        a terminal step-size window of a few iterations.
    adaptation_schedule : AdaptationSchedule, optional
        The windowed schedule; the Stan defaults (75 / 25 / 50) when
        omitted.

    Raises
    ------
    ValueError
        If ``step_size`` is not positive, ``num_steps`` is less than
        one, ``target_accept`` is outside ``(0, 1)``, or
        ``adaptation`` is not a known mode.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        num_steps: int = 10,
        mass_matrix: MassMatrixKind = "identity",
        target_accept: float = 0.65,
        divergence_threshold: float = 1000.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        adaptation: AdaptationMode = "single",
        adaptation_schedule: AdaptationSchedule | None = None,
    ) -> None:
        if num_steps < 1:
            raise ValueError(f"HMCKernel: num_steps must be >= 1, got {num_steps}")
        super().__init__(
            "HMCKernel",
            step_size,
            mass_matrix,
            target_accept,
            divergence_threshold,
            adapt_step_size,
            adapt_mass_matrix,
            adaptation,
            adaptation_schedule,
        )
        self._num_steps = num_steps

    def configuration(self) -> dict[str, StateValue]:
        """The kernel's hyperparameters, the trajectory length included.

        Returns
        -------
        dict[str, StateValue]
            The configuration.
        """
        return {**super().configuration(), "num_steps": self._num_steps}

    def init(
        self,
        registry: LatentRegistry,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        initial_position: torch.Tensor,
    ) -> KernelState:
        """Reset the adaptation state and build the chain's first state.

        Parameters
        ----------
        registry : LatentRegistry
            The registry whose flat vector the chain walks.
        model : MonadicProgram
            The model (unused; the potential carries it).
        x : torch.Tensor
            The input (unused).
        observations : dict[str, torch.Tensor]
            The data (unused).
        initial_position : torch.Tensor
            The starting position, shape ``(D,)``.

        Returns
        -------
        KernelState
            The starting state.

        Raises
        ------
        ValueError
            If ``initial_position`` has the wrong shape or a
            non-finite value.
        """
        del model, x, observations
        return self._initial_state(registry, initial_position)

    def step(
        self,
        state: KernelState,
        potential: PotentialFn,
    ) -> KernelState:
        """Run one HMC transition.

        Parameters
        ----------
        state : KernelState
            The current state.
        potential : PotentialFn
            The target.

        Returns
        -------
        KernelState
            The next state; ``extras["accept_prob"]`` holds the
            Metropolis acceptance probability.
        """
        assert self._mass is not None, "HMCKernel.init was not called"
        self._prepare_transition(state, potential)
        z0 = state.position
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
        new_state = KernelState(
            position=new_position,
            log_density=new_log_density,
            grad_log_density=new_grad,
            step_count=state.step_count + 1,
            accept_count=state.accept_count + accept_count_delta,
            diverged=diverged,
            extras={**state.extras, "accept_prob": accept_prob},
            stats={
                "accepted": float(accept_count_delta),
                "accept_prob": accept_prob,
                "diverged": float(diverged),
                "step_size": eps,
                "num_steps": float(self._num_steps),
                "energy": float(h1 if accept_count_delta else h0),
            },
            gibbs_values=state.gibbs_values,
        )
        self._adapt(new_state, potential, accept_prob)
        return new_state


# ---------------------------------------------------------------------------
# NUTS
# ---------------------------------------------------------------------------


class _NUTSBuildTreeResult:
    """Internal accumulator returned by the recursive tree builder.

    The selected proposal carries its position, log-density, and
    gradient, all computed by the leapfrog step that reached it, so
    the kernel's next transition starts from that gradient without
    evaluating the target again.
    """

    __slots__ = (
        "z_minus",
        "p_minus",
        "grad_minus",
        "z_plus",
        "p_plus",
        "grad_plus",
        "z_proposal",
        "p_proposal",
        "log_density_proposal",
        "grad_proposal",
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
        p_proposal: torch.Tensor,
        log_density_proposal: torch.Tensor,
        grad_proposal: torch.Tensor,
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
        self.p_proposal = p_proposal
        self.log_density_proposal = log_density_proposal
        self.grad_proposal = grad_proposal
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


class NUTSKernel(_HamiltonianKernel):
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
    mass_matrix : {"identity", "diagonal", "dense"}
        Mass-matrix shape. Default ``"diagonal"``.
    target_accept : float
        Target tree-averaged Metropolis acceptance for dual
        averaging. Default ``0.8``.
    divergence_threshold : float
        Energy-error threshold above which a leapfrog substep is
        marked divergent and terminates the tree on its branch.
    adapt_step_size : bool
        Whether warmup adapts the step size by dual averaging.
    adapt_mass_matrix : bool
        Whether warmup adapts a diagonal or dense metric.
    adaptation : {"windowed", "single"}
        The warmup mode (see `AdaptationMode`). Default ``"single"``.
        ``"windowed"`` keeps the initial transient out of the metric
        and tunes the step size under the metric it samples with; it
        suits warmups long enough for the schedule's buffers (150
        iterations under the defaults), since a shorter warmup leaves
        a terminal step-size window of a few iterations.
    adaptation_schedule : AdaptationSchedule, optional
        The windowed schedule; the Stan defaults (75 / 25 / 50) when
        omitted.

    Raises
    ------
    ValueError
        If ``step_size`` is not positive, ``max_tree_depth`` is less
        than one, ``target_accept`` is outside ``(0, 1)``, or
        ``adaptation`` is not a known mode.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        max_tree_depth: int = 10,
        mass_matrix: MassMatrixKind = "diagonal",
        target_accept: float = 0.8,
        divergence_threshold: float = 1000.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        adaptation: AdaptationMode = "single",
        adaptation_schedule: AdaptationSchedule | None = None,
    ) -> None:
        if max_tree_depth < 1:
            raise ValueError(
                f"NUTSKernel: max_tree_depth must be >= 1, got {max_tree_depth}"
            )
        super().__init__(
            "NUTSKernel",
            step_size,
            mass_matrix,
            target_accept,
            divergence_threshold,
            adapt_step_size,
            adapt_mass_matrix,
            adaptation,
            adaptation_schedule,
        )
        self._max_depth = max_tree_depth

    def configuration(self) -> dict[str, StateValue]:
        """The kernel's hyperparameters, the maximum tree depth included.

        Returns
        -------
        dict[str, StateValue]
            The configuration.
        """
        return {**super().configuration(), "max_tree_depth": self._max_depth}

    def init(
        self,
        registry: LatentRegistry,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        initial_position: torch.Tensor,
    ) -> KernelState:
        """Reset the adaptation state and build the chain's first state.

        Parameters
        ----------
        registry : LatentRegistry
            The registry whose flat vector the chain walks.
        model : MonadicProgram
            The model (unused; the potential carries it).
        x : torch.Tensor
            The input (unused).
        observations : dict[str, torch.Tensor]
            The data (unused).
        initial_position : torch.Tensor
            The starting position, shape ``(D,)``.

        Returns
        -------
        KernelState
            The starting state.

        Raises
        ------
        ValueError
            If ``initial_position`` has the wrong shape or a
            non-finite value.
        """
        del model, x, observations
        return self._initial_state(registry, initial_position)

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
                p_proposal=p1,
                log_density_proposal=ld1,
                grad_proposal=g1,
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
            p_prop = right.p_proposal
            ld_prop = right.log_density_proposal
            g_prop = right.grad_proposal
        else:
            z_prop = left.z_proposal
            p_prop = left.p_proposal
            ld_prop = left.log_density_proposal
            g_prop = left.grad_proposal
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
            p_proposal=p_prop,
            log_density_proposal=ld_prop,
            grad_proposal=g_prop,
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
        """Run one NUTS transition.

        Parameters
        ----------
        state : KernelState
            The current state.
        potential : PotentialFn
            The target.

        Returns
        -------
        KernelState
            The next state; ``extras`` holds the tree-averaged
            acceptance statistic under ``"accept_prob"`` and the tree
            depth under ``"tree_depth"``.
        """
        assert self._mass is not None, "NUTSKernel.init was not called"
        self._prepare_transition(state, potential)
        z = state.position
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
        p_proposal = p
        log_density_proposal = ld
        grad_proposal = g
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
                    p_proposal = subtree.p_proposal
                    log_density_proposal = subtree.log_density_proposal
                    grad_proposal = subtree.grad_proposal
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
        accepted = not torch.equal(z_proposal, state.position)
        new_state = KernelState(
            position=z_proposal.detach(),
            log_density=log_density_proposal.detach(),
            grad_log_density=grad_proposal.detach(),
            step_count=state.step_count + 1,
            accept_count=state.accept_count + (1 if accepted else 0),
            diverged=diverged,
            extras={**state.extras, "accept_prob": accept_prob, "tree_depth": depth},
            stats={
                "accepted": float(accepted),
                "accept_prob": accept_prob,
                "diverged": float(diverged),
                "step_size": eps,
                "tree_depth": float(depth),
                "num_steps": float(n_accept_steps),
                "energy": float(-log_density_proposal + self._mass.kinetic(p_proposal)),
            },
            gibbs_values=state.gibbs_values,
        )
        self._adapt(new_state, potential, accept_prob)
        return new_state


__all__ = [
    "HMCKernel",
    "NUTSKernel",
    "MassMatrixKind",
]
