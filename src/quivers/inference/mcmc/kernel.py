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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch.distributions import Distribution

from quivers.continuous.program_steps import Draw, Observe
from quivers.continuous.programs import MonadicProgram
from quivers.inference.registry import LatentRegistry, _require_one_run
from quivers.inference.trace import trace


_DISTRIBUTION_VALIDATION_CODE = frozenset(
    {Distribution.__init__.__code__, Distribution._validate_sample.__code__}
)
"""Code objects of the two `torch.distributions` validators: the
parameter-constraint check in `Distribution.__init__` and the value
check in `Distribution._validate_sample`."""


def _rejects_position(error: BaseException) -> bool:
    """Whether an error raised while scoring a position means the
    position lies outside the model's support.

    Two kinds of error qualify. A ``ValueError`` raised by one of
    `torch.distributions`' own validators (a parameter that fails its
    constraint, or a value outside its support) is identified by the
    code object of the innermost traceback frame, not by its message.
    A `torch.linalg.LinAlgError` is a factorization of a matrix that
    the position makes numerically non-positive-definite. Every other
    error is a defect of the program or its data.

    Parameters
    ----------
    error : BaseException
        The error raised by ``log_joint``.

    Returns
    -------
    bool
        ``True`` when the position should be rejected with a
        log-density of ``-inf``; ``False`` when the error must
        propagate.
    """
    if isinstance(error, torch.linalg.LinAlgError):
        return True
    if not isinstance(error, ValueError):
        return False
    frame = error.__traceback__
    if frame is None:
        return False
    while frame.tb_next is not None:
        frame = frame.tb_next
    return frame.tb_frame.f_code in _DISTRIBUTION_VALIDATION_CODE


@contextmanager
def _distribution_validation() -> Iterator[None]:
    """Enable `torch.distributions` argument validation for a block.

    The initial-position check relies on the validators to report
    observed data outside a likelihood's support, so it runs with
    validation on whatever the process-wide default is, and restores
    that default afterwards.

    Yields
    ------
    None
        Control, with validation enabled.
    """
    previous = bool(Distribution._validate_args)
    Distribution.set_default_validate_args(True)
    try:
        yield
    finally:
        Distribution.set_default_validate_args(previous)


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

    The position ``z`` is the registry's flat unconstrained vector.
    Inside `log_density` each latent site is mapped to one
    constrained draw of shape `LatentSite.constrained_shape` and
    then handed to `MonadicProgram.log_joint` as a trace-side value:
    non-plate sites at ``(1, *constrained_shape)``, with the leading
    axis being the program's batch axis, and plate sites at
    ``constrained_shape``.

    `check_initial` scores a starting position strictly and must run
    once before sampling (`MCMC.run` does): it raises on observed data
    that is not finite or lies outside a likelihood's support, which
    `log_density` would otherwise score as an unreachable position.

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
        _require_one_run(x, "PotentialFn")
        self._model = model
        self._registry = registry
        self._x = x
        self._observations = observations

    def log_density(self, z: torch.Tensor) -> torch.Tensor:
        """Unconstrained-space log-density (Jacobian-corrected).

        ``z`` is unflattened into per-site unconstrained tensors,
        pushed through each site's bijector to one draw of shape
        `LatentSite.constrained_shape`, and lifted to trace-side
        values by `LatentRegistry.trace_values` with a batch of one.
        `MonadicProgram.log_joint` thus receives every non-plate site
        at ``(1, *constrained_shape)`` (the program's ``(batch, ...)``
        input convention, which broadcasts against the input ``x``)
        and every plate site at ``constrained_shape``. A custom
        density reached through ``log_joint`` should index that
        leading axis accordingly; the shapes reported by
        `MCMCResult` carry no such axis.

        A position outside the model's support scores ``-inf``, and
        the kernel rejects it. Two errors from ``log_joint`` mean
        that: a ``ValueError`` from `torch.distributions`' own
        parameter or value validation (a scale that underflows to
        zero, a latent value pushed onto its support's boundary), and
        a `torch.linalg.LinAlgError` from factorizing a matrix the
        position makes non-positive-definite. They are recognised by
        where they are raised, not by their message. Every other
        error (a shape or dtype mismatch, an error in a custom score
        function, a missing name) propagates, and so do the
        validation errors that hold at every position, which
        `check_initial` raises before sampling starts.

        Parameters
        ----------
        z : torch.Tensor
            Flat unconstrained position, shape ``(D,)``.

        Returns
        -------
        torch.Tensor
            Scalar Jacobian-corrected log-density, or ``-inf`` where
            the position is outside the model's support.

        Raises
        ------
        ValueError
            If ``z`` does not have the registry's flat length.
        Exception
            Any error ``log_joint`` raises that does not mark the
            position as outside the support.
        """
        trace_side, log_det_total = self._trace_side(z)
        try:
            log_p = self._model.log_joint(self._x, {**trace_side, **self._observations})
        except (ValueError, torch.linalg.LinAlgError) as error:
            if not _rejects_position(error):
                raise
            return torch.tensor(float("-inf"), device=z.device, dtype=z.dtype)
        result = log_p.sum() + log_det_total
        if not torch.isfinite(result):
            return torch.tensor(float("-inf"), device=z.device, dtype=z.dtype)
        return result

    def _trace_side(
        self, z: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Map a flat position to trace-side latent values and the
        total log-Jacobian of the bijectors.

        Parameters
        ----------
        z : torch.Tensor
            Flat unconstrained position, shape ``(D,)``.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor]
            The latent values `LatentRegistry.trace_values` produces
            with a batch of one, and the scalar ``log |det dT/dz|``.
        """
        per_site_unc = self._registry.unflatten_unconstrained(z)
        constrained, log_abs_dets = self._registry.to_constrained(per_site_unc)
        log_det_total = torch.zeros((), device=z.device, dtype=z.dtype)
        for site in self._registry.sites.values():
            log_det_total = log_det_total + log_abs_dets[site.name].sum()
        return self._registry.trace_values(constrained, batch_size=1), log_det_total

    def check_initial(self, z: torch.Tensor) -> torch.Tensor:
        """Score a chain's initial position, raising on any failure.

        Run once before sampling. Observed values must be finite; the
        program is then scored with `torch.distributions` validation
        enabled and no error converted, so observed data outside a
        likelihood's support, mismatched row counts, and errors in
        custom score functions raise here with their own type. A
        position that scores a non-finite log-density raises, naming
        the sites whose log-density is not finite.

        Parameters
        ----------
        z : torch.Tensor
            Flat unconstrained initial position, shape ``(D,)``.

        Returns
        -------
        torch.Tensor
            The finite scalar Jacobian-corrected log-density at ``z``.

        Raises
        ------
        ValueError
            If an observed value is not finite, or the log-density at
            ``z`` is not finite.
        Exception
            Any error ``log_joint`` raises at ``z``.
        """
        site_names = {
            name
            for step in self._model.steps
            if isinstance(step, Draw | Observe)
            for name in step.names
        }
        for name, value in self._observations.items():
            if (
                name in site_names
                and value.dtype.is_floating_point
                and not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"PotentialFn: observed site {name!r} holds non-finite values"
                )
        with torch.no_grad(), _distribution_validation():
            trace_side, log_det_total = self._trace_side(z)
            values = {**trace_side, **self._observations}
            result = self._model.log_joint(self._x, values).sum() + log_det_total
            if torch.isfinite(result):
                return result
            recorded = trace(self._model, self._x, observations=values)
        offending = sorted(
            name
            for name, site in recorded.sites.items()
            if not torch.isfinite(site.log_prob).all()
        )
        if not torch.isfinite(log_det_total):
            offending.append("the bijector log-Jacobian")
        raise ValueError(
            f"PotentialFn: the log-density at the initial position is "
            f"{float(result)}; non-finite terms: {', '.join(offending) or 'none recorded'}"
        )

    def value_and_grad(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_density, grad_log_density)`` for ``z``.

        One call runs one forward pass of `log_density` and one
        backward pass, and the HMC and NUTS integrators make exactly
        one call per leapfrog step. The gradient is taken against a
        fresh leaf copy of ``z`` with autograd enabled, so the call
        differentiates even inside a ``torch.no_grad()`` block and
        gradient propagation does not leak into the kernel's state.

        For divergent positions (where the log-density is
        ``-inf``), returns a zero gradient: the kernel rejects
        the trajectory in the Metropolis step anyway, and a zero
        gradient keeps the leapfrog integrator from producing NaN
        downstream.

        Parameters
        ----------
        z : torch.Tensor
            Flat unconstrained position, shape ``(D,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The detached scalar log-density and its detached
            gradient with respect to ``z`` (shape ``(D,)``).

        Raises
        ------
        Exception
            Any error `log_density` propagates: an error from the
            program or its data rather than a position outside the
            support.
        """
        z_leaf = z.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            ld = self.log_density(z_leaf)
            if not torch.isfinite(ld):
                return ld.detach(), torch.zeros_like(z)
            grad = torch.autograd.grad(
                ld, z_leaf, create_graph=False, allow_unused=False
            )[0]
        if grad is None or not torch.isfinite(grad).all():
            return ld.detach(), torch.zeros_like(z)
        return ld.detach(), grad.detach()


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
