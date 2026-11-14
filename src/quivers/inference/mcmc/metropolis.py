"""Metropolis-Hastings kernels and their proposals.

A `MetropolisKernel` moves the flat unconstrained vector by a
`Proposal`: the proposal draws a candidate :math:`z'` from
:math:`q(\\cdot \\mid z)` and reports the log Hastings correction
:math:`\\log q(z \\mid z') - \\log q(z' \\mid z)`, and the kernel accepts
with probability

.. math::

    \\alpha = \\min\\Bigl(1, \\frac{\\tilde p(z')\\, q(z \\mid z')}
                              {\\tilde p(z)\\, q(z' \\mid z)}\\Bigr),

where :math:`\\tilde p` is the Jacobian-corrected unconstrained-space
target of the `PotentialFn` (Hastings 1970,
`doi:10.1093/biomet/57.1.97 <https://doi.org/10.1093/biomet/57.1.97>`_).
The chain then leaves the target invariant for any proposal whose
correction is right, which makes these kernels the natural partners of
the gradient kernels in a `CycleKernel` or `MixtureKernel`: a cheap
symmetric or involutive move (a transposition of exchangeable blocks, a
translation along a weakly identified ridge) or an independence move
from a fitted approximation.

Two proposals are provided, and a user writes others by subclassing
`Proposal`:

* `RandomWalkProposal`: a Gaussian random walk, symmetric, so its
  correction is zero.
* `IndependenceProposal`: draws from a fixed distribution over the flat
  vector, optionally a defensive mixture with a heavier one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
from torch.distributions import Distribution

from quivers.continuous.programs import MonadicProgram
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
    StateValue,
    _with_generator,
)
from quivers.inference.registry import LatentRegistry


class Proposal(ABC):
    """A Metropolis-Hastings proposal on the flat unconstrained vector.

    Subclasses implement `propose`, and override `bind` when the
    proposal needs the registry's layout (a site's offset in the flat
    vector, for instance, to permute exchangeable blocks).
    """

    def bind(self, registry: LatentRegistry) -> None:
        """Receive the registry of the chain the proposal will move.

        `MetropolisKernel.init` calls this once per chain, before the
        first `propose`. The base implementation records nothing.

        Parameters
        ----------
        registry : LatentRegistry
            The registry whose flat vector the chain walks.
        """
        del registry

    def tune(self, population: torch.Tensor) -> None:
        """Tune the proposal to a population of positions.

        `MetropolisKernel.tune` forwards the population here; the base
        implementation leaves the proposal as it is.

        Parameters
        ----------
        population : torch.Tensor
            Flat positions, shape ``(N, D)``.
        """
        del population

    def configuration(self) -> dict[str, StateValue]:
        """The proposal's configuration, for telling kernels apart.

        The base implementation records the class; a proposal with
        parameters adds them, so that `MCMC.resume` refuses a checkpoint
        written with different ones.

        Returns
        -------
        dict[str, StateValue]
            The configuration.
        """
        return {"proposal": f"{type(self).__module__}.{type(self).__qualname__}"}

    def state_dict(self) -> dict[str, StateValue]:
        """The proposal's mutable state for one chain.

        The base implementation, for a proposal without state, is empty;
        a proposal that adapts or otherwise changes during a run returns
        what `load_state_dict` needs to continue it exactly.

        Returns
        -------
        dict[str, StateValue]
            The state.
        """
        return {}

    def load_state_dict(self, state: dict[str, StateValue]) -> None:
        """Restore what `state_dict` returned, after `bind`.

        Parameters
        ----------
        state : dict[str, StateValue]
            A dictionary `state_dict` produced.
        """
        del state

    @abstractmethod
    def propose(
        self, position: torch.Tensor, generator: torch.Generator | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a candidate and its log Hastings correction.

        Parameters
        ----------
        position : torch.Tensor
            The current flat unconstrained position, shape ``(D,)``.
        generator : torch.Generator | None
            The chain-local generator every random draw must use.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The candidate, shape ``(D,)``, and the scalar
            ``log q(position | candidate) - log q(candidate | position)``,
            zero for a symmetric proposal.
        """


class RandomWalkProposal(Proposal):
    """Gaussian random-walk proposal :math:`z' = z + \\sigma \\odot \\epsilon`.

    The proposal is symmetric, so its Hastings correction is zero.

    Parameters
    ----------
    scale : float | torch.Tensor
        The step scale :math:`\\sigma`, a positive float or a ``(D,)``
        tensor of positive per-coordinate scales.

    Raises
    ------
    ValueError
        If a scale is not positive and finite.
    """

    def __init__(self, scale: float | torch.Tensor) -> None:
        scale_t = torch.as_tensor(scale, dtype=torch.get_default_dtype())
        if not bool(torch.isfinite(scale_t).all()) or not bool((scale_t > 0).all()):
            raise ValueError(
                f"RandomWalkProposal: scale must be positive and finite, got {scale}"
            )
        self._scale = scale_t

    def tune(self, population: torch.Tensor) -> None:
        """Scale the walk to a population's spread.

        The scale becomes :math:`2.38 / \\sqrt{D}` times each
        coordinate's population standard deviation, the asymptotically
        optimal random-walk scaling for a Gaussian target (Roberts,
        Gelman, and Gilks 1997, `doi:10.1214/aoap/1034625254
        <https://doi.org/10.1214/aoap/1034625254>`_).

        Parameters
        ----------
        population : torch.Tensor
            Flat positions, shape ``(N, D)``, with ``N >= 2``.

        Raises
        ------
        ValueError
            If the population has fewer than two rows or a coordinate
            with no spread.
        """
        if population.dim() != 2 or population.shape[0] < 2:
            raise ValueError(
                "RandomWalkProposal.tune: the population must have shape (N, D) "
                f"with N >= 2; got {tuple(population.shape)}"
            )
        spread = population.std(dim=0)
        if not bool((spread > 0).all()):
            raise ValueError(
                "RandomWalkProposal.tune: a coordinate of the population has no "
                "spread, so no scale can be read from it"
            )
        self._scale = 2.38 / math.sqrt(population.shape[1]) * spread

    def configuration(self) -> dict[str, StateValue]:
        """The class and the step scale.

        Returns
        -------
        dict[str, StateValue]
            The configuration.
        """
        return {**super().configuration(), "scale": self._scale.tolist()}

    def state_dict(self) -> dict[str, StateValue]:
        """The step scale, which `tune` may have changed.

        Returns
        -------
        dict[str, StateValue]
            The scale.
        """
        return {"scale": self._scale.clone()}

    def load_state_dict(self, state: dict[str, StateValue]) -> None:
        """Restore the step scale.

        Parameters
        ----------
        state : dict[str, StateValue]
            A dictionary `state_dict` produced.

        Raises
        ------
        ValueError
            If the scale is missing.
        """
        scale = state.get("scale")
        if not isinstance(scale, torch.Tensor):
            raise ValueError("RandomWalkProposal.load_state_dict: bad scale")
        self._scale = scale.clone()

    def bind(self, registry: LatentRegistry) -> None:
        """Check a per-coordinate scale against the flat vector's length.

        Parameters
        ----------
        registry : LatentRegistry
            The registry whose flat vector the chain walks.

        Raises
        ------
        ValueError
            If ``scale`` is a vector of the wrong length.
        """
        dim = registry.total_unconstrained_dim
        if self._scale.dim() > 0 and tuple(self._scale.shape) != (dim,):
            raise ValueError(
                f"RandomWalkProposal: scale has shape {tuple(self._scale.shape)}; "
                f"the flat vector has length {dim}"
            )

    def propose(
        self, position: torch.Tensor, generator: torch.Generator | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``position + scale * N(0, I)``.

        Parameters
        ----------
        position : torch.Tensor
            The current position, shape ``(D,)``.
        generator : torch.Generator | None
            The chain-local generator.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The candidate and a zero correction.
        """
        noise = torch.randn(position.shape, generator=generator, dtype=position.dtype)
        return position + self._scale * noise, torch.zeros((), dtype=position.dtype)


class IndependenceProposal(Proposal):
    """Independence proposal from a fixed distribution over the flat vector.

    Candidates are drawn from :math:`q`, whatever the current position,
    and the correction is :math:`\\log q(z) - \\log q(z')`. With a
    ``defensive`` distribution :math:`d` the proposal is the mixture
    :math:`q = (1 - w)\\, g + w\\, d` of ``distribution`` :math:`g` and
    :math:`d` with weight :math:`w`, which keeps the importance ratio
    :math:`\\tilde p / q` bounded where :math:`g` has lighter tails than
    the target, the usual way to correct a fitted variational
    approximation exactly (Hesterberg 1995,
    `doi:10.1080/00401706.1995.10484303
    <https://doi.org/10.1080/00401706.1995.10484303>`_).

    Parameters
    ----------
    distribution : torch.distributions.Distribution
        The proposal :math:`g`, with event shape ``(D,)`` and no batch
        shape, over the registry's flat unconstrained vector.
    defensive : torch.distributions.Distribution, optional
        The defensive component :math:`d`, shaped as ``distribution``.
    defensive_weight : float
        The weight :math:`w` of ``defensive``, in ``(0, 1)``. Default
        ``0.1``; ignored without ``defensive``.

    Raises
    ------
    ValueError
        If a distribution has a batch shape or an event shape other than
        a vector, or ``defensive_weight`` is outside ``(0, 1)``.
    """

    def __init__(
        self,
        distribution: Distribution,
        defensive: Distribution | None = None,
        defensive_weight: float = 0.1,
    ) -> None:
        for name, dist in (("distribution", distribution), ("defensive", defensive)):
            if dist is None:
                continue
            if tuple(dist.batch_shape) != () or len(dist.event_shape) != 1:
                raise ValueError(
                    f"IndependenceProposal: {name} must have event shape (D,) and "
                    f"no batch shape; got batch {tuple(dist.batch_shape)}, event "
                    f"{tuple(dist.event_shape)}"
                )
        if defensive is not None and not 0.0 < defensive_weight < 1.0:
            raise ValueError(
                "IndependenceProposal: defensive_weight must be in (0, 1), "
                f"got {defensive_weight}"
            )
        if defensive is not None and defensive.event_shape != distribution.event_shape:
            raise ValueError(
                "IndependenceProposal: defensive and distribution have event shapes "
                f"{tuple(defensive.event_shape)} and {tuple(distribution.event_shape)}"
            )
        self._distribution = distribution
        self._defensive = defensive
        self._weight = defensive_weight

    def configuration(self) -> dict[str, StateValue]:
        """The distributions' classes, the weight, and the proposal's
        log-density at two fixed points, which tells distributions with
        different parameters apart.

        Returns
        -------
        dict[str, StateValue]
            The configuration.
        """
        dim = int(self._distribution.event_shape[0])
        probes = (torch.zeros(dim), torch.linspace(-1.0, 1.0, dim))
        return {
            **super().configuration(),
            "distribution": type(self._distribution).__qualname__,
            "defensive": None
            if self._defensive is None
            else type(self._defensive).__qualname__,
            "defensive_weight": self._weight if self._defensive is not None else None,
            "log_prob_probes": [float(self.log_prob(point)) for point in probes],
        }

    def bind(self, registry: LatentRegistry) -> None:
        """Check the proposal's event shape against the flat vector.

        Parameters
        ----------
        registry : LatentRegistry
            The registry whose flat vector the chain walks.

        Raises
        ------
        ValueError
            If the event shape is not ``(D,)``.
        """
        dim = registry.total_unconstrained_dim
        if tuple(self._distribution.event_shape) != (dim,):
            raise ValueError(
                "IndependenceProposal: the distribution has event shape "
                f"{tuple(self._distribution.event_shape)}; the flat vector has "
                f"length {dim}"
            )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """The proposal's log-density.

        Parameters
        ----------
        value : torch.Tensor
            A flat position, shape ``(D,)``.

        Returns
        -------
        torch.Tensor
            The scalar :math:`\\log q(\\text{value})`.
        """
        log_g = self._distribution.log_prob(value)
        if self._defensive is None:
            return log_g
        log_d = self._defensive.log_prob(value)
        return torch.logaddexp(
            math.log1p(-self._weight) + log_g, math.log(self._weight) + log_d
        )

    def propose(
        self, position: torch.Tensor, generator: torch.Generator | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw from the proposal, independently of ``position``.

        Parameters
        ----------
        position : torch.Tensor
            The current position, shape ``(D,)``.
        generator : torch.Generator | None
            The chain-local generator; the component choice and the
            draw both come from it.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The candidate and ``log q(position) - log q(candidate)``.
        """
        source = self._distribution
        if self._defensive is not None:
            pick = torch.rand((), generator=generator).item()
            if pick < self._weight:
                source = self._defensive
        candidate = _with_generator(generator, lambda: source.sample())
        candidate = candidate.to(position.dtype)
        return candidate, self.log_prob(position) - self.log_prob(candidate)


class MetropolisKernel(MCMCKernel):
    """Metropolis-Hastings kernel on the flat unconstrained vector.

    One transition draws a candidate from the `Proposal`, scores it by
    `PotentialFn.log_density` alone, and accepts it with probability
    :math:`\\min(1, \\exp(\\log \\tilde p(z') - \\log \\tilde p(z) +
    \\log c))`, with :math:`\\log c` the proposal's Hastings correction.
    The gradient is evaluated only when the candidate is accepted, so
    the state a gradient kernel receives next carries the gradient at
    its position. A candidate outside the model's support scores
    :math:`-\\infty` and is rejected.

    The kernel has no adaptation of its own; `start_adaptation` and
    `stop_adaptation` only toggle `is_adapting`.

    Parameters
    ----------
    proposal : Proposal
        The proposal.
    """

    def __init__(self, proposal: Proposal) -> None:
        self._proposal = proposal

    @property
    def proposal(self) -> Proposal:
        """The proposal the kernel draws candidates from."""
        return self._proposal

    def tune(self, population: torch.Tensor) -> None:
        """Forward a population to the proposal (`Proposal.tune`).

        Parameters
        ----------
        population : torch.Tensor
            Flat positions, shape ``(N, D)``.
        """
        self._proposal.tune(population)

    def configuration(self) -> dict[str, StateValue]:
        """The class and the proposal's configuration.

        Returns
        -------
        dict[str, StateValue]
            The configuration.
        """
        return {**super().configuration(), "proposal": self._proposal.configuration()}

    def state_dict(self) -> dict[str, StateValue]:
        """Whether the kernel is adapting, and the proposal's state.

        Returns
        -------
        dict[str, StateValue]
            The state.
        """
        return {**super().state_dict(), "proposal": self._proposal.state_dict()}

    def load_state_dict(self, state: dict[str, StateValue]) -> None:
        """Restore what `state_dict` returned.

        Parameters
        ----------
        state : dict[str, StateValue]
            A dictionary `state_dict` produced.

        Raises
        ------
        ValueError
            If an entry is missing or has the wrong type.
        """
        super().load_state_dict(state)
        proposal = state.get("proposal")
        if not isinstance(proposal, dict):
            raise ValueError("MetropolisKernel.load_state_dict: bad proposal state")
        self._proposal.load_state_dict(proposal)

    def init(
        self,
        registry: LatentRegistry,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        initial_position: torch.Tensor,
    ) -> KernelState:
        """Bind the proposal and build the chain's first state.

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
            The starting state; its log-density and gradient are
            evaluated by the first transition.

        Raises
        ------
        ValueError
            If ``initial_position`` has the wrong shape or a non-finite
            value, or the proposal refuses the registry.
        """
        del model, x, observations
        dim = registry.total_unconstrained_dim
        if initial_position.shape != (dim,):
            raise ValueError(
                f"MetropolisKernel.init: initial_position must have shape ({dim},); "
                f"got {tuple(initial_position.shape)}"
            )
        if not bool(torch.isfinite(initial_position).all()):
            raise ValueError(
                "MetropolisKernel.init: initial_position must contain only finite "
                "values"
            )
        self.is_adapting = False
        self._proposal.bind(registry)
        return KernelState(
            position=initial_position.clone(),
            log_density=torch.tensor(0.0),
            grad_log_density=torch.zeros(dim),
        )

    def step(self, state: KernelState, potential: PotentialFn) -> KernelState:
        """Run one Metropolis-Hastings transition.

        Parameters
        ----------
        state : KernelState
            The current state.
        potential : PotentialFn
            The target, conditioned on the state's Gibbs values.

        Returns
        -------
        KernelState
            The next state, with ``stats`` holding ``"accepted"`` and
            ``"accept_prob"``.

        Raises
        ------
        ValueError
            If the proposal returns a candidate of the wrong shape or a
            correction that is not a scalar.
        """
        state = self._evaluate_first(state, potential)
        candidate, log_correction = self._proposal.propose(
            state.position, self._generator
        )
        if candidate.shape != state.position.shape:
            raise ValueError(
                f"MetropolisKernel: the proposal returned a candidate of shape "
                f"{tuple(candidate.shape)}; the position has shape "
                f"{tuple(state.position.shape)}"
            )
        if log_correction.numel() != 1:
            raise ValueError(
                "MetropolisKernel: the proposal's log Hastings correction must be a "
                f"scalar; got shape {tuple(log_correction.shape)}"
            )
        candidate = candidate.detach()
        with torch.no_grad():
            log_density = potential.log_density(candidate)
        log_alpha = float(log_density - state.log_density + log_correction.reshape(()))
        if math.isnan(log_alpha):
            log_alpha = -math.inf
        accept_prob = math.exp(min(0.0, log_alpha))
        uniform = torch.rand((), generator=self._generator).item()
        accepted = uniform < accept_prob
        if accepted:
            new_log_density, new_grad = potential.value_and_grad(candidate)
            position = candidate
        else:
            new_log_density, new_grad = state.log_density, state.grad_log_density
            position = state.position
        return KernelState(
            position=position,
            log_density=new_log_density,
            grad_log_density=new_grad,
            step_count=state.step_count + 1,
            accept_count=state.accept_count + int(accepted),
            diverged=False,
            extras={**state.extras, "accept_prob": accept_prob},
            stats={"accepted": float(accepted), "accept_prob": accept_prob},
            gibbs_values=state.gibbs_values,
        )


__all__ = [
    "Proposal",
    "RandomWalkProposal",
    "IndependenceProposal",
    "MetropolisKernel",
]
