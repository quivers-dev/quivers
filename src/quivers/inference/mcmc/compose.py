"""Composite Markov kernels: deterministic cycles and random mixtures.

If each of the kernels :math:`K_1, \\dots, K_m` leaves the posterior
:math:`\\pi` invariant, so does their composition
:math:`K_1 K_2 \\cdots K_m` (a cycle) and every mixture
:math:`\\sum_i w_i K_i` with weights fixed in advance (Tierney 1994,
`doi:10.1214/aos/1176325750 <https://doi.org/10.1214/aos/1176325750>`_).
The composite kernels here build such chains from any `MCMCKernel`:
gradient kernels, `MetropolisKernel` moves, `GibbsKernel` blocks, and
other composites.

Every component sees the same chain state and the same chain-local
generator. The state carries the log-density and gradient at the
current position, under the potential conditioned on the current
`KernelState.gibbs_values`, so a gradient kernel that follows a
Metropolis move or a Gibbs sweep reuses the gradient its predecessor
left. Each component receives the potential conditioned on the Gibbs
values it starts from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
    StateValue,
)
from quivers.inference.registry import LatentRegistry


def _named_components(
    kernels: Sequence[MCMCKernel] | Mapping[str, MCMCKernel], owner: str
) -> tuple[tuple[str, MCMCKernel], ...]:
    """Name a composite's components and check them.

    Parameters
    ----------
    kernels : Sequence[MCMCKernel] | Mapping[str, MCMCKernel]
        The components, named by the mapping's keys or by their
        positions (``"0"``, ``"1"``, ...).
    owner : str
        The composite's class name, for errors.

    Returns
    -------
    tuple[tuple[str, MCMCKernel], ...]
        ``(name, kernel)`` pairs in order.

    Raises
    ------
    ValueError
        If there are no components, a name is empty or contains a dot,
        or one kernel object appears twice.
    """
    pairs = (
        tuple(kernels.items())
        if isinstance(kernels, Mapping)
        else tuple((str(i), k) for i, k in enumerate(kernels))
    )
    if not pairs:
        raise ValueError(f"{owner}: kernels must hold at least one kernel")
    for name, kernel in pairs:
        if not name or "." in name:
            raise ValueError(
                f"{owner}: component name {name!r} must be non-empty and contain "
                "no dot, which separates a component's name from its statistics"
            )
        if not isinstance(kernel, MCMCKernel):
            raise TypeError(
                f"{owner}: component {name!r} is a {type(kernel).__name__}, not an "
                "MCMCKernel"
            )
    if len({id(k) for _, k in pairs}) != len(pairs):
        raise ValueError(
            f"{owner}: one kernel object appears twice; each component needs its "
            "own adaptation state, so construct a separate kernel for each"
        )
    return pairs


def _moved(before: KernelState, after: KernelState) -> bool:
    """Whether a transition changed the position or a Gibbs value.

    Parameters
    ----------
    before : KernelState
        The state the transition started from.
    after : KernelState
        The state it produced.

    Returns
    -------
    bool
        ``True`` when the chain moved.
    """
    if not torch.equal(before.position, after.position):
        return True
    return any(
        not torch.equal(value, before.gibbs_values[name])
        for name, value in after.gibbs_values.items()
        if name in before.gibbs_values
    )


class _CompositeKernel(MCMCKernel):
    """Shared plumbing of the composite kernels.

    Parameters
    ----------
    kernels : Sequence[MCMCKernel] | Mapping[str, MCMCKernel]
        The components.
    """

    def __init__(
        self, kernels: Sequence[MCMCKernel] | Mapping[str, MCMCKernel]
    ) -> None:
        self._components = _named_components(kernels, type(self).__name__)

    @property
    def components(self) -> dict[str, MCMCKernel]:
        """The components, keyed by name, in order."""
        return dict(self._components)

    @property
    def gibbs_sites(self) -> frozenset[str]:
        """The union of the components' Gibbs sites."""
        out: frozenset[str] = frozenset()
        for _, kernel in self._components:
            out = out | kernel.gibbs_sites
        return out

    def tune(self, population: torch.Tensor) -> None:
        """Forward a population to every component.

        Parameters
        ----------
        population : torch.Tensor
            Flat positions, shape ``(N, D)``.
        """
        for _, kernel in self._components:
            kernel.tune(population)

    def configuration(self) -> dict[str, StateValue]:
        """The class and every component's name and configuration.

        Returns
        -------
        dict[str, StateValue]
            The configuration.
        """
        return {
            **super().configuration(),
            "components": [
                [name, kernel.configuration()] for name, kernel in self._components
            ],
        }

    def state_dict(self) -> dict[str, StateValue]:
        """Whether the composite is adapting, and every component's state.

        Returns
        -------
        dict[str, StateValue]
            The state.
        """
        return {
            **super().state_dict(),
            "components": {
                name: kernel.state_dict() for name, kernel in self._components
            },
        }

    def load_state_dict(self, state: dict[str, StateValue]) -> None:
        """Restore what `state_dict` returned.

        Parameters
        ----------
        state : dict[str, StateValue]
            A dictionary `state_dict` produced.

        Raises
        ------
        ValueError
            If a component's state is missing.
        """
        super().load_state_dict(state)
        components = state.get("components")
        if not isinstance(components, dict):
            raise ValueError(
                f"{type(self).__name__}.load_state_dict: bad component states"
            )
        for name, kernel in self._components:
            component = components.get(name)
            if not isinstance(component, dict):
                raise ValueError(
                    f"{type(self).__name__}.load_state_dict: no state for {name!r}"
                )
            kernel.load_state_dict(component)

    def _set_generator(self, generator: torch.Generator) -> None:
        """Install the chain-local generator on the composite and every
        component.

        Parameters
        ----------
        generator : torch.Generator
            The chain's generator.
        """
        super()._set_generator(generator)
        for _, kernel in self._components:
            kernel._set_generator(generator)

    def init(
        self,
        registry: LatentRegistry,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        initial_position: torch.Tensor,
    ) -> KernelState:
        """Initialize every component for a new chain.

        Parameters
        ----------
        registry : LatentRegistry
            The registry whose flat vector the chain walks.
        model : MonadicProgram
            The model.
        x : torch.Tensor
            The one-row input.
        observations : dict[str, torch.Tensor]
            The data.
        initial_position : torch.Tensor
            The starting position, shape ``(D,)``.

        Returns
        -------
        KernelState
            The first component's starting state.
        """
        self.is_adapting = False
        states = [
            kernel.init(registry, model, x, observations, initial_position)
            for _, kernel in self._components
        ]
        return states[0]

    def stop_adaptation(self) -> None:
        """Freeze every component's adaptation."""
        super().stop_adaptation()
        for _, kernel in self._components:
            kernel.stop_adaptation()

    @staticmethod
    def _run_component(
        name: str,
        kernel: MCMCKernel,
        state: KernelState,
        potential: PotentialFn,
        stats: dict[str, float],
    ) -> KernelState:
        """Advance the chain by one component and collect its statistics.

        Parameters
        ----------
        name : str
            The component's name.
        kernel : MCMCKernel
            The component.
        state : KernelState
            The state it starts from.
        potential : PotentialFn
            The composite's potential.
        stats : dict[str, float]
            The composite's statistics, extended in place with the
            component's under ``"<name>.<key>"``.

        Returns
        -------
        KernelState
            The component's output state.
        """
        conditioned = (
            potential.condition(state.gibbs_values) if state.gibbs_values else potential
        )
        out = kernel.step(state, conditioned)
        stats.update({f"{name}.{key}": value for key, value in out.stats.items()})
        return out

    @staticmethod
    def _finish(
        start: KernelState,
        end: KernelState,
        stats: dict[str, float],
        diverged: bool,
    ) -> KernelState:
        """Package a composite transition as one step of the chain.

        Parameters
        ----------
        start : KernelState
            The state the composite transition started from.
        end : KernelState
            The last component's output.
        stats : dict[str, float]
            The components' prefixed statistics.
        diverged : bool
            Whether any component diverged.

        Returns
        -------
        KernelState
            The composite's output: ``step_count`` advances by one and
            ``accept_count`` by one when the chain moved.
        """
        moved = _moved(start, end)
        stats = {**stats, "accepted": float(moved), "accept_prob": float(moved)}
        return KernelState(
            position=end.position,
            log_density=end.log_density,
            grad_log_density=end.grad_log_density,
            step_count=start.step_count + 1,
            accept_count=start.accept_count + int(moved),
            diverged=diverged,
            extras=end.extras,
            stats=stats,
            gibbs_values=end.gibbs_values,
        )


class CycleKernel(_CompositeKernel):
    """Apply every component once per transition, in order.

    The cycle leaves the posterior invariant when each component does.
    During warmup every component adapts over the same ``num_warmup``
    transitions.

    The composite's own ``stats`` hold each component's statistics
    under ``"<name>.<key>"``, and ``"accepted"`` records whether the
    transition moved the chain (its position or a Gibbs value).

    Parameters
    ----------
    kernels : Sequence[MCMCKernel] | Mapping[str, MCMCKernel]
        The components, named by the mapping's keys or by position.

    Raises
    ------
    ValueError
        If there are no components, a name is empty or holds a dot, or
        one kernel object appears twice.
    TypeError
        If a component is not an `MCMCKernel`.
    """

    def start_adaptation(self, num_warmup: int) -> None:
        """Start every component's warmup of ``num_warmup`` transitions.

        Parameters
        ----------
        num_warmup : int
            Number of warmup transitions.
        """
        super().start_adaptation(num_warmup)
        for _, kernel in self._components:
            kernel.start_adaptation(num_warmup)

    def step(self, state: KernelState, potential: PotentialFn) -> KernelState:
        """Run every component once, in order.

        Parameters
        ----------
        state : KernelState
            The current state.
        potential : PotentialFn
            The target conditioned on ``state.gibbs_values``.

        Returns
        -------
        KernelState
            The state after the last component.
        """
        stats: dict[str, float] = {}
        diverged = False
        current = state
        for name, kernel in self._components:
            current = self._run_component(name, kernel, current, potential, stats)
            diverged = diverged or current.diverged
        return self._finish(state, current, stats, diverged)


class MixtureKernel(_CompositeKernel):
    """Apply one component per transition, chosen at random.

    Component :math:`i` is chosen with probability ``probs[i]``,
    independently of the chain's state, from the chain-local generator,
    so the mixture leaves the posterior invariant when each component
    does. During warmup the choices for all ``num_warmup`` transitions
    are drawn when warmup starts, and each component starts its own
    adaptation for exactly the number of transitions that will select
    it, so a windowed schedule runs to completion.

    The composite's ``stats`` hold the selected component's statistics
    under ``"<name>.<key>"`` (the other components' keys are absent),
    and ``"accepted"`` records whether the chain moved.

    Parameters
    ----------
    kernels : Sequence[MCMCKernel] | Mapping[str, MCMCKernel]
        The components, named by the mapping's keys or by position.
    probs : Sequence[float], optional
        Selection probabilities, positive and summing to one; uniform
        when omitted.

    Raises
    ------
    ValueError
        If the components are invalid (see `CycleKernel`), or ``probs``
        has the wrong length, a non-positive entry, or does not sum to
        one.
    TypeError
        If a component is not an `MCMCKernel`.
    """

    def __init__(
        self,
        kernels: Sequence[MCMCKernel] | Mapping[str, MCMCKernel],
        probs: Sequence[float] | None = None,
    ) -> None:
        super().__init__(kernels)
        count = len(self._components)
        weights = [1.0 / count] * count if probs is None else [float(p) for p in probs]
        if len(weights) != count:
            raise ValueError(
                f"MixtureKernel: probs has {len(weights)} entries for {count} kernels"
            )
        if any(not p > 0.0 for p in weights) or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(
                f"MixtureKernel: probs must be positive and sum to one, got {weights}"
            )
        self._probs = torch.tensor(weights, dtype=torch.float64)
        self._warmup_choices: list[int] = []
        self._warmup_index = 0

    @property
    def probs(self) -> torch.Tensor:
        """The selection probabilities, shape ``(num_components,)``."""
        return self._probs.clone()

    def configuration(self) -> dict[str, StateValue]:
        """The components' configurations and the selection probabilities.

        Returns
        -------
        dict[str, StateValue]
            The configuration.
        """
        return {**super().configuration(), "probs": self._probs.tolist()}

    def state_dict(self) -> dict[str, StateValue]:
        """The components' states and the warmup's drawn choices.

        Returns
        -------
        dict[str, StateValue]
            The state.
        """
        return {
            **super().state_dict(),
            "warmup_choices": list(self._warmup_choices),
            "warmup_index": self._warmup_index,
        }

    def load_state_dict(self, state: dict[str, StateValue]) -> None:
        """Restore what `state_dict` returned.

        Parameters
        ----------
        state : dict[str, StateValue]
            A dictionary `state_dict` produced.

        Raises
        ------
        ValueError
            If the warmup choices are missing or malformed.
        """
        super().load_state_dict(state)
        choices, index = state.get("warmup_choices"), state.get("warmup_index")
        if (
            not isinstance(choices, list)
            or not all(isinstance(c, int) for c in choices)
            or not isinstance(index, int)
        ):
            raise ValueError("MixtureKernel.load_state_dict: bad warmup choices")
        self._warmup_choices = [c for c in choices if isinstance(c, int)]
        self._warmup_index = index

    def init(
        self,
        registry: LatentRegistry,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        initial_position: torch.Tensor,
    ) -> KernelState:
        """Initialize every component and clear the warmup choices.

        Parameters
        ----------
        registry : LatentRegistry
            The registry whose flat vector the chain walks.
        model : MonadicProgram
            The model.
        x : torch.Tensor
            The one-row input.
        observations : dict[str, torch.Tensor]
            The data.
        initial_position : torch.Tensor
            The starting position, shape ``(D,)``.

        Returns
        -------
        KernelState
            The first component's starting state.
        """
        self._warmup_choices = []
        self._warmup_index = 0
        return super().init(registry, model, x, observations, initial_position)

    def start_adaptation(self, num_warmup: int) -> None:
        """Draw the warmup's component choices and start each component's
        adaptation for the transitions that select it.

        Parameters
        ----------
        num_warmup : int
            Number of warmup transitions.
        """
        super().start_adaptation(num_warmup)
        self._warmup_choices = (
            torch.multinomial(
                self._probs, num_warmup, replacement=True, generator=self._generator
            ).tolist()
            if num_warmup > 0
            else []
        )
        self._warmup_index = 0
        for index, (_, kernel) in enumerate(self._components):
            kernel.start_adaptation(self._warmup_choices.count(index))

    def _choose(self) -> int:
        """The component for the next transition.

        Returns
        -------
        int
            Its index.

        Raises
        ------
        RuntimeError
            If warmup runs more transitions than ``start_adaptation`` was
            told.
        """
        if self.is_adapting:
            if self._warmup_index >= len(self._warmup_choices):
                raise RuntimeError(
                    "MixtureKernel: warmup ran more transitions than the "
                    f"{len(self._warmup_choices)} start_adaptation was given"
                )
            choice = self._warmup_choices[self._warmup_index]
            self._warmup_index += 1
            return choice
        return int(torch.multinomial(self._probs, 1, generator=self._generator).item())

    def step(self, state: KernelState, potential: PotentialFn) -> KernelState:
        """Run one randomly chosen component.

        Parameters
        ----------
        state : KernelState
            The current state.
        potential : PotentialFn
            The target conditioned on ``state.gibbs_values``.

        Returns
        -------
        KernelState
            The chosen component's output.
        """
        name, kernel = self._components[self._choose()]
        stats: dict[str, float] = {}
        current = self._run_component(name, kernel, state, potential, stats)
        return self._finish(state, current, stats, current.diverged)


__all__ = ["CycleKernel", "MixtureKernel"]
