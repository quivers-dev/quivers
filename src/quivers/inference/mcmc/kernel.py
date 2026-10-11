"""MCMC kernel abstract base + shared state container.

Every MCMC kernel under [`quivers.inference.mcmc`][quivers.inference.mcmc] follows the
same contract: an `init` method that produces a starting
`KernelState` from a `LatentRegistry` and an
optional initial flat unconstrained position, and a `step`
method that consumes the current state plus the model + data and
returns the next state.

All kernels operate on the *flat unconstrained* latent vector
provided by the registry. The model's log-joint is evaluated by
unflattening, pushing through per-site bijectors (with the
Jacobian correction), and calling `MonadicProgram.log_joint`
on the constrained sites. The Jacobian-corrected scalar is the
*unconstrained-space* log-density that the kernel walks; this is
the standard NumPyro / Pyro pattern, and it is what makes
constrained-support sites (HalfNormal, Dirichlet, Beta, …)
samplable by gradient-based MCMC without any special-casing in
the kernel itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.registry import LatentRegistry


@dataclass
class KernelState:
    """Mutable container for one chain's MCMC state.

    Attributes
    ----------
    position : torch.Tensor
        Current flat unconstrained latent vector. Shape ``(D,)``
        for one chain, ``(num_chains, D)`` for parallel chains.
    log_density : torch.Tensor
        Unconstrained-space log-density (Jacobian-corrected
        log-joint) at ``position``. Shape ``()`` or
        ``(num_chains,)``.
    grad_log_density : torch.Tensor
        Gradient of ``log_density`` with respect to ``position``,
        same shape as ``position``. Cached so leapfrog re-uses the
        last-evaluated gradient at the proposal endpoint.
    step_count : int
        Number of `step` calls so far (counts both warmup
        and post-warmup steps).
    accept_count : int
        Number of proposals accepted across the chain. Useful for
        reporting acceptance rate.
    diverged : bool
        Whether the most recent step's energy error exceeded the
        kernel's divergence threshold. Reset by each kernel
        as appropriate.
    extras : dict
        Per-kernel additional state (e.g. NUTS tree depth, HMC
        step-size adaptation cumulants).
    """

    position: torch.Tensor
    log_density: torch.Tensor
    grad_log_density: torch.Tensor
    step_count: int = 0
    accept_count: int = 0
    diverged: bool = False
    extras: dict = field(default_factory=dict)


class PotentialFn:
    """Callable that maps a flat unconstrained position to the
    unconstrained-space negative log-density and its gradient.

    HMC and NUTS need both the potential
    :math:`U(z) = -\\log \\tilde{p}(z)` (where
    :math:`\\tilde{p}(z) = p(T(z), y) \\cdot |\\det J_T(z)|` is the
    Jacobian-corrected unconstrained-space joint) and its gradient
    :math:`\\nabla U(z)`. The two are computed in a single
    autograd pass and cached on the kernel state.

    Parameters
    ----------
    model : MonadicProgram
        Generative model.
    registry : LatentRegistry
        Latent-site registry for ``model``.
    x : torch.Tensor
        Model input. Shape ``(batch, ...)``.
    observations : dict[str, torch.Tensor]
        Observed-site values and host data.
    """

    def __init__(
        self,
        model: MonadicProgram,
        registry: LatentRegistry,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> None:
        self._model = model
        self._registry = registry
        self._x = x
        self._observations = observations

    def log_density(self, z: torch.Tensor) -> torch.Tensor:
        """Unconstrained-space log-density (Jacobian-corrected).

        Trajectories that wander to the edge of a constrained
        support can produce values that fall outside
        `torch.distributions`' validation envelope (e.g. exact
        zeros against a strictly-positive support after a long
        leapfrog stride). Rather than letting the resulting
        ``ValueError`` propagate and kill the chain, this method
        returns ``-inf`` for those positions; the kernel reads
        non-finite log-densities as divergent transitions and
        rejects them in the Metropolis step.
        """
        try:
            per_site_unc = self._registry.unflatten_unconstrained(z)
            constrained: dict[str, torch.Tensor] = {}
            log_det_total = torch.zeros((), device=z.device, dtype=z.dtype)
            for site in self._registry.sites.values():
                z_site = per_site_unc[site.name]
                if not site.is_plate:
                    z_site = z_site.unsqueeze(0)
                v = site.bijector(z_site)
                log_det_total = log_det_total + (
                    site.bijector.log_abs_det_jacobian(z_site, v).sum()
                )
                if site.constrained_dim == 1 and v.dim() >= 1 and v.shape[-1] == 1:
                    v = v.squeeze(-1)
                constrained[site.name] = v
            log_p = self._model.log_joint(
                self._x, {**constrained, **self._observations}
            )
            result = log_p.sum() + log_det_total
        except ValueError:
            return torch.tensor(float("-inf"), device=z.device, dtype=z.dtype)
        if not torch.isfinite(result):
            return torch.tensor(float("-inf"), device=z.device, dtype=z.dtype)
        return result

    def value_and_grad(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_density, grad_log_density)`` for ``z``.

        ``z`` is expected to be a detached tensor; we make a fresh
        leaf with ``requires_grad=True`` so gradient propagation
        doesn't leak into the kernel's accumulated state.

        For divergent positions (where the log-density is
        ``-inf``), returns a zero gradient: the kernel rejects
        the trajectory in the Metropolis step anyway, and a zero
        gradient keeps the leapfrog integrator from producing NaN
        downstream.
        """
        z_leaf = z.detach().clone().requires_grad_(True)
        try:
            ld = self.log_density(z_leaf)
            if not torch.isfinite(ld):
                return ld.detach(), torch.zeros_like(z)
            grad = torch.autograd.grad(
                ld, z_leaf, create_graph=False, allow_unused=False
            )[0]
            if grad is None or not torch.isfinite(grad).all():
                return ld.detach(), torch.zeros_like(z)
            return ld.detach(), grad.detach()
        except ValueError, RuntimeError:
            return (
                torch.tensor(float("-inf"), device=z.device, dtype=z.dtype),
                torch.zeros_like(z),
            )


class MCMCKernel(ABC):
    """Abstract Markov kernel on the flat unconstrained latent
    vector.

    Concrete subclasses implement `init` and `step`.
    Adaptation phases (warmup) typically mutate kernel-internal
    state (step size, mass matrix) and freeze it for the sampling
    phase; the kernel's `is_adapting` flag tracks that.
    """

    is_adapting: bool = False
    _generator: torch.Generator | None = None

    def _set_generator(self, generator: torch.Generator) -> None:
        """Install the chain-local random-number generator.

        `MCMC` calls this before initializing each chain. Keeping the
        generator on the kernel lets concrete kernels route momentum,
        proposal-selection, and acceptance draws through one stream
        without changing the public `step` signature.
        """
        self._generator = generator

    @abstractmethod
    def init(
        self,
        registry: LatentRegistry,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        initial_position: torch.Tensor,
    ) -> KernelState:
        """Build the starting `KernelState` from the supplied
        initial flat unconstrained vector. The initial gradient is
        evaluated here so `step` can re-use it."""

    @abstractmethod
    def step(
        self,
        state: KernelState,
        potential: PotentialFn,
    ) -> KernelState:
        """Advance the chain one Metropolis step. The potential
        function is constructed once per `MCMC.run` and
        re-used across every step / chain."""

    def start_adaptation(self) -> None:
        """Enter the adaptation (warmup) phase."""
        self.is_adapting = True

    def stop_adaptation(self) -> None:
        """Freeze the kernel's adapted parameters."""
        self.is_adapting = False


__all__ = ["KernelState", "PotentialFn", "MCMCKernel"]
