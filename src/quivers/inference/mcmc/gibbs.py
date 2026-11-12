"""Gibbs updates of latent sites held outside the flat vector.

A `GibbsKernel` owns a block of named latent sites. `MCMC.run` leaves
them out of the registry, so the gradient and Metropolis kernels move
only the remaining flat unconstrained vector, and holds their values in
`KernelState.gibbs_values`; every kernel then evaluates the potential
conditioned on those values (`PotentialFn.condition`). A Gibbs
transition draws new values for its sites from their full conditional
given the flat vector and every other site, so it leaves the joint
posterior invariant, and composed with a kernel on the flat vector in a
`CycleKernel` or `MixtureKernel` it gives a Markov chain on the whole
latent space (Geman and Geman 1984,
`doi:10.1109/TPAMI.1984.4767596
<https://doi.org/10.1109/TPAMI.1984.4767596>`_).

The full conditional is drawn one of two ways:

* by enumeration, the default, for sites with a finite elementwise
  support (`torch.distributions.constraints.boolean`, or an
  `integer_interval`, as a Bernoulli or Categorical draw has): each
  element in turn, in row-major order, is redrawn from its conditional
  given the others, scoring every value of its support with the model's
  log joint;
* by a user-supplied `GibbsUpdate`, which must draw from (or leave
  invariant) the block's full conditional; this covers conjugate
  updates of continuous sites and discrete structures too large to
  enumerate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import torch
from torch.distributions import constraints as _constraints

from quivers.continuous.plate import PlateDraw
from quivers.continuous.programs import MonadicProgram
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
    StateValue,
)
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import trace


GibbsUpdate = Callable[
    [torch.Generator | None, dict[str, torch.Tensor], dict[str, torch.Tensor]],
    dict[str, torch.Tensor],
]
"""A user-supplied Gibbs update ``update(generator, gibbs_values,
latent_values) -> new_gibbs_values``.

``gibbs_values`` holds the block's current values and ``latent_values``
the constrained values of the registry's sites (the flat vector), both
as trace-side values: a non-plate site carries a leading run axis of
one, a plate site has its constrained shape. The update returns new
values for exactly the block's sites, in the same layout, drawn from
their full conditional (or by any move that leaves it invariant), with
every random draw taken from ``generator``."""


def _enumerable_support(support: _constraints.Constraint) -> torch.Tensor | None:
    """The finite elementwise support of a constraint.

    Parameters
    ----------
    support : Constraint
        A site's support.

    Returns
    -------
    torch.Tensor | None
        The support's values in increasing order, or ``None`` when the
        support is not a finite set of scalars.
    """
    if support is _constraints.boolean:
        return torch.tensor([0.0, 1.0])
    if isinstance(support, _constraints.integer_interval):
        lower = support.lower_bound
        upper = support.upper_bound
        if isinstance(lower, int | float) and isinstance(upper, int | float):
            return torch.arange(int(lower), int(upper) + 1)
    return None


class GibbsKernel(MCMCKernel):
    """Gibbs block over named latent sites.

    Parameters
    ----------
    sites : Sequence[str]
        The latent sites the block updates, in update order. They are
        held outside the flat vector for the whole run.
    update : GibbsUpdate, optional
        A full-conditional sampler for the whole block. When omitted,
        each site is updated by enumerating its finite support element
        by element, which requires a boolean or bounded integer support.

    Raises
    ------
    ValueError
        If ``sites`` is empty or names a site twice.
    """

    def __init__(self, sites: Sequence[str], update: GibbsUpdate | None = None) -> None:
        names = tuple(sites)
        if not names:
            raise ValueError("GibbsKernel: sites must name at least one site")
        if len(set(names)) != len(names):
            raise ValueError(f"GibbsKernel: sites {names} name a site twice")
        self._sites = names
        self._update = update
        self._supports: dict[str, torch.Tensor] = {}
        self._registry: LatentRegistry | None = None

    @property
    def gibbs_sites(self) -> frozenset[str]:
        """The block's sites."""
        return frozenset(self._sites)

    def configuration(self) -> dict[str, StateValue]:
        """The class, the sites, and the update function's qualified name.

        Returns
        -------
        dict[str, StateValue]
            The configuration.
        """
        update = (
            None
            if self._update is None
            else f"{getattr(self._update, '__module__', '')}."
            f"{getattr(self._update, '__qualname__', type(self._update).__qualname__)}"
        )
        return {**super().configuration(), "sites": list(self._sites), "update": update}

    def init(
        self,
        registry: LatentRegistry,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        initial_position: torch.Tensor,
    ) -> KernelState:
        """Check the block against the model and build the first state.

        Without an ``update``, every site's support is read from a run
        of the model (drawn from a forked global generator, which is left
        untouched) and must be finite.

        Parameters
        ----------
        registry : LatentRegistry
            The registry of the flat vector, without the block's sites.
        model : MonadicProgram
            The model.
        x : torch.Tensor
            The one-row input.
        observations : dict[str, torch.Tensor]
            The data.
        initial_position : torch.Tensor
            The flat vector's starting position, shape ``(D,)``.

        Returns
        -------
        KernelState
            The starting state; `MCMC.run` fills in its Gibbs values.

        Raises
        ------
        ValueError
            If a site is in the registry, is not a latent sample site of
            the run, or (without an ``update``) has a support that is not
            a finite set of scalars.
        """
        inside = sorted(name for name in self._sites if name in registry)
        if inside:
            raise ValueError(
                f"GibbsKernel: {inside} are in the registry; a Gibbs site must be "
                "left out of the flat vector (MCMC.run does this for every site a "
                "GibbsKernel names)"
            )
        self._registry = registry
        self.is_adapting = False
        if self._update is None:
            # The run that names the sites draws from a copy of the
            # chain's stream, so a seeded chain reads the same run
            # whatever the global RNG holds, without advancing its stream.
            with torch.random.fork_rng(devices=[]):
                if self._generator is not None:
                    torch.random.set_rng_state(self._generator.get_state())
                recorded = trace(model, x, observations=observations)
            for name in self._sites:
                record = recorded.sites.get(name)
                if record is None or record.is_observed or record.morphism is None:
                    raise ValueError(
                        f"GibbsKernel: {name!r} is not a latent sample site of the run"
                    )
                morphism = record.morphism
                support = (
                    morphism.family.support
                    if isinstance(morphism, PlateDraw)
                    else morphism.support
                )
                values = _enumerable_support(support)
                if values is None:
                    raise ValueError(
                        f"GibbsKernel: site {name!r} has support {support}, which is "
                        "not a finite set of scalars, so its full conditional cannot "
                        "be enumerated; pass update= with a sampler for it"
                    )
                self._supports[name] = values
        return KernelState(
            position=initial_position.clone(),
            log_density=torch.tensor(0.0),
            grad_log_density=torch.zeros(registry.total_unconstrained_dim),
        )

    def _enumerate(
        self,
        values: dict[str, torch.Tensor],
        position: torch.Tensor,
        potential: PotentialFn,
    ) -> dict[str, torch.Tensor]:
        """Redraw every element of every site from its full conditional.

        Parameters
        ----------
        values : dict[str, torch.Tensor]
            The block's current trace-side values.
        position : torch.Tensor
            The flat vector.
        potential : PotentialFn
            The target.

        Returns
        -------
        dict[str, torch.Tensor]
            The new values.

        Raises
        ------
        RuntimeError
            If every value of an element's support scores ``-inf``.
        """
        current = dict(values)
        for name in self._sites:
            support = self._supports[name]
            site_value = current[name].clone()
            flat = site_value.reshape(-1)
            for index in range(flat.numel()):
                scores = torch.empty(support.numel(), dtype=torch.float64)
                for k, candidate in enumerate(support.tolist()):
                    flat[index] = candidate
                    current[name] = site_value.clone()
                    with torch.no_grad():
                        scores[k] = float(
                            potential.condition(current).log_density(position)
                        )
                if not bool(torch.isfinite(scores).any()):
                    raise RuntimeError(
                        f"GibbsKernel: every value of element {index} of site "
                        f"{name!r} scores -inf given the rest of the chain's state"
                    )
                probs = torch.softmax(scores, dim=0)
                choice = int(
                    torch.multinomial(probs, 1, generator=self._generator).item()
                )
                flat[index] = support[choice]
                current[name] = site_value.clone()
        return current

    def step(self, state: KernelState, potential: PotentialFn) -> KernelState:
        """Run one Gibbs sweep over the block.

        Parameters
        ----------
        state : KernelState
            The current state, with the block's values in
            ``gibbs_values``.
        potential : PotentialFn
            The target conditioned on ``state.gibbs_values``.

        Returns
        -------
        KernelState
            The state with the block's new values, and the log-density
            and gradient of the flat vector under the potential
            conditioned on them. ``stats["accepted"]`` is ``1.0`` when
            the sweep changed a value; ``stats["accept_prob"]`` is
            ``1.0``.

        Raises
        ------
        ValueError
            If the state lacks a site of the block, or an ``update``
            returns other sites, shapes, or dtypes than it was given.
        """
        missing = sorted(name for name in self._sites if name not in state.gibbs_values)
        if missing:
            raise ValueError(
                f"GibbsKernel: the chain state holds no values for {missing}"
            )
        current = {name: state.gibbs_values[name] for name in self._sites}
        if self._update is None:
            new = self._enumerate(state.gibbs_values, state.position, potential)
            proposed = {name: new[name] for name in self._sites}
        else:
            assert self._registry is not None, "GibbsKernel.init was not called"
            constrained, _ = self._registry.to_constrained(
                self._registry.unflatten_unconstrained(state.position)
            )
            latent = self._registry.trace_values(constrained, batch_size=1)
            proposed = self._update(
                self._generator,
                {name: value.clone() for name, value in current.items()},
                latent,
            )
            if set(proposed) != set(self._sites):
                raise ValueError(
                    f"GibbsKernel: the update returned sites {sorted(proposed)}; "
                    f"the block is {sorted(self._sites)}"
                )
            for name in self._sites:
                old, value = current[name], proposed[name]
                if value.shape != old.shape or value.dtype != old.dtype:
                    raise ValueError(
                        f"GibbsKernel: the update returned {name!r} with shape "
                        f"{tuple(value.shape)} and dtype {value.dtype}; the site "
                        f"holds shape {tuple(old.shape)} and dtype {old.dtype}"
                    )
        changed = any(
            not torch.equal(proposed[name], current[name]) for name in self._sites
        )
        gibbs_values = {**state.gibbs_values, **proposed}
        log_density, grad = potential.condition(gibbs_values).value_and_grad(
            state.position
        )
        if not math.isfinite(float(log_density)):
            raise RuntimeError(
                "GibbsKernel: the chain's log-density is not finite after the "
                f"update of {sorted(self._sites)}"
            )
        return KernelState(
            position=state.position,
            log_density=log_density,
            grad_log_density=grad,
            step_count=state.step_count + 1,
            accept_count=state.accept_count + int(changed),
            diverged=False,
            extras=state.extras,
            stats={"accepted": float(changed), "accept_prob": 1.0},
            gibbs_values=gibbs_values,
        )


__all__ = ["GibbsKernel", "GibbsUpdate"]
