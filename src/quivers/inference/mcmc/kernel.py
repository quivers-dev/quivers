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

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch.distributions import Distribution
from torch.distributions import constraints as _constraints

from quivers.continuous.program_steps import Draw, Observe
from quivers.continuous.programs import MonadicProgram
from quivers.effects.joint_plan import JointPlans
from quivers.effects.subsample import refuse_subsampling
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


def _check_observed_support(
    model: MonadicProgram,
    observations: Mapping[str, torch.Tensor],
    consumer: str,
) -> None:
    """Refuse observed data outside its site's support.

    The data a potential conditions on do not change between
    evaluations, so they are checked once, against the support of the
    distribution each observed site is drawn from: a value outside it
    makes every position score ``-inf``, which is an error in the data,
    not a region of the posterior.

    Parameters
    ----------
    model : MonadicProgram
        The model.
    observations : Mapping[str, torch.Tensor]
        Observed-site values and host data; host data are not checked.
    consumer : str
        The caller named in the error, such as ``"PotentialFn"``.

    Raises
    ------
    ValueError
        If an observed value is not finite, or lies outside the support
        of its site's distribution, naming the site and the values.
    """
    for spec in model.steps:
        if not isinstance(spec, Draw | Observe) or len(spec.names) != 1:
            continue
        (name,) = spec.names
        if name not in observations:
            continue
        value = torch.as_tensor(observations[name])
        if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
            bad = value[~torch.isfinite(value)]
            raise ValueError(
                f"{consumer}: observed site {name!r} holds non-finite values "
                f"{bad.flatten()[:5].tolist()}"
            )
        morphism = model.step_module(spec)
        support = getattr(morphism, "support", None)
        if not isinstance(support, _constraints.Constraint) or (
            _constraints.is_dependent(support)
        ):
            continue
        inside = support.check(value)
        if not bool(inside.all()):
            outside = (
                value[~inside] if inside.shape == value.shape else value.reshape(-1)
            )
            raise ValueError(
                f"{consumer}: observed site {name!r} holds values outside the "
                f"support {support} of its distribution: "
                f"{outside[:5].tolist()}"
            )


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


type StateValue = (
    torch.Tensor
    | float
    | int
    | bool
    | str
    | None
    | list[StateValue]
    | dict[str, StateValue]
)
"""A value in a kernel's or a chain state's state dictionary: a tensor, a
Python scalar or string, ``None``, or a list or string-keyed dictionary
of such values. These are exactly the values a checkpoint stores and
`torch.load` reads back with ``weights_only=True``, so loading a
checkpoint executes no code."""


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
    stats : dict[str, float]
        Statistics of the most recent transition, replaced by every
        `MCMCKernel.step`: ``"accepted"`` (``1.0`` or ``0.0``) and
        ``"accept_prob"`` for every kernel, with kernel-specific entries
        such as ``"tree_depth"``. A composite kernel prefixes each
        component's entries with the component's name and a dot
        (``"nuts.accepted"``).
    gibbs_values : dict[str, torch.Tensor]
        Current values of the sites a `GibbsKernel` updates, which sit
        outside the flat vector. Each is held as a trace-side value, the
        layout `MonadicProgram.log_joint` consumes: a non-plate site
        carries a leading run axis of one, a plate site has its
        constrained shape. ``log_density`` and ``grad_log_density`` are
        those of the potential conditioned on these values.
    """

    position: torch.Tensor
    log_density: torch.Tensor
    grad_log_density: torch.Tensor
    step_count: int = 0
    accept_count: int = 0
    diverged: bool = False
    extras: dict = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)
    gibbs_values: dict[str, torch.Tensor] = field(default_factory=dict)

    def state_dict(self) -> dict[str, StateValue]:
        """The state as a dictionary a checkpoint can store.

        Returns
        -------
        dict[str, StateValue]
            Every field, with tensors detached; `from_state_dict`
            rebuilds the state exactly.

        Raises
        ------
        TypeError
            If ``extras`` holds a value other than a tensor, a Python
            scalar, or a string.
        """
        extras: dict[str, StateValue] = {}
        for key, value in self.extras.items():
            if not isinstance(value, torch.Tensor | float | int | bool | str):
                raise TypeError(
                    f"KernelState.state_dict: extras[{key!r}] is a "
                    f"{type(value).__name__}, which a checkpoint cannot store"
                )
            extras[str(key)] = (
                value.detach() if isinstance(value, torch.Tensor) else value
            )
        return {
            "position": self.position.detach(),
            "log_density": self.log_density.detach(),
            "grad_log_density": self.grad_log_density.detach(),
            "step_count": self.step_count,
            "accept_count": self.accept_count,
            "diverged": self.diverged,
            "extras": extras,
            "stats": {key: float(value) for key, value in self.stats.items()},
            "gibbs_values": {k: v.detach() for k, v in self.gibbs_values.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, StateValue]) -> "KernelState":
        """Rebuild a state from `state_dict`'s output.

        Parameters
        ----------
        state : dict[str, StateValue]
            A dictionary `state_dict` produced.

        Returns
        -------
        KernelState
            The state.

        Raises
        ------
        ValueError
            If a field is missing or has the wrong type.
        """
        try:
            position = state["position"]
            log_density = state["log_density"]
            grad = state["grad_log_density"]
            step_count = state["step_count"]
            accept_count = state["accept_count"]
            diverged = state["diverged"]
            extras = state["extras"]
            stats = state["stats"]
            gibbs_values = state["gibbs_values"]
        except KeyError as error:
            raise ValueError(f"KernelState.from_state_dict: missing {error}") from None
        if not (
            isinstance(position, torch.Tensor)
            and isinstance(log_density, torch.Tensor)
            and isinstance(grad, torch.Tensor)
            and isinstance(step_count, int)
            and isinstance(accept_count, int)
            and isinstance(diverged, bool)
            and isinstance(extras, dict)
            and isinstance(stats, dict)
            and isinstance(gibbs_values, dict)
        ):
            raise ValueError("KernelState.from_state_dict: a field has the wrong type")
        values: dict[str, torch.Tensor] = {}
        for name, value in gibbs_values.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"KernelState.from_state_dict: Gibbs value {name!r} is not a tensor"
                )
            values[name] = value
        floats: dict[str, float] = {}
        for key, value in stats.items():
            if not isinstance(value, float):
                raise ValueError(
                    f"KernelState.from_state_dict: statistic {key!r} is not a float"
                )
            floats[key] = value
        return cls(
            position=position,
            log_density=log_density,
            grad_log_density=grad,
            step_count=step_count,
            accept_count=accept_count,
            diverged=diverged,
            extras=dict(extras),
            stats=floats,
            gibbs_values=values,
        )


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

    Observed data are checked once, at construction, against the
    support of each observed site's distribution, so data that are
    not finite or lie outside a likelihood's support raise here rather
    than scoring every position ``-inf``; only latent positions
    outside the support score ``-inf``. `check_initial` scores a
    starting position strictly and must run once before sampling
    (`MCMC.run` does), catching the remaining program and data errors.

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

    Raises
    ------
    ValueError
        If ``x`` has more than one row, or an observed value is not
        finite or lies outside its site's support.
    """

    def __init__(
        self,
        model: MonadicProgram,
        registry: LatentRegistry,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> None:
        _require_one_run(x, "PotentialFn")
        _check_observed_support(model, observations, "PotentialFn")
        self._model = model
        self._registry = registry
        self._x = x
        self._base_observations = observations
        self._conditioned: dict[str, torch.Tensor] = {}
        self._observations = observations
        # [forward passes, gradient evaluations], shared with every
        # conditioned copy so that a chain's evaluations are counted in
        # one place.
        self._counts = [0, 0]
        # The program's log joint prepared per layout of the values,
        # shared with every conditioned copy: conditioning changes the
        # values, not their layout.
        self._plans = JointPlans(model, x)

    @property
    def num_evaluations(self) -> int:
        """Forward passes of the log-density so far, gradient ones included.

        Counted across this potential and every potential `condition`
        derived from it; `check_initial` is not counted.
        """
        return self._counts[0]

    @property
    def num_gradient_evaluations(self) -> int:
        """Calls of `value_and_grad` so far, across this potential and
        every potential `condition` derived from it."""
        return self._counts[1]

    @property
    def registry(self) -> LatentRegistry:
        """The registry whose flat vector the potential is a function of."""
        return self._registry

    @property
    def conditioned_values(self) -> dict[str, torch.Tensor]:
        """The site values `condition` fixed, keyed by site name."""
        return dict(self._conditioned)

    def condition(self, values: Mapping[str, torch.Tensor]) -> "PotentialFn":
        """The potential with further sites held at given values.

        The returned potential scores the flat vector exactly as this
        one does, with ``values`` added to the observations
        `MonadicProgram.log_joint` receives. It replaces, rather than
        adds to, any values this potential was itself conditioned on,
        so conditioning on a chain's current `KernelState.gibbs_values`
        is always relative to the data the potential was built with.
        Conditioning is what lets a `GibbsKernel` hold sites outside the
        flat vector: the other kernels move the flat vector under the
        potential conditioned on the Gibbs sites' current values.

        Parameters
        ----------
        values : Mapping[str, torch.Tensor]
            Trace-side values of sites outside the registry (a non-plate
            site with a leading run axis of one, a plate site at its
            constrained shape).

        Returns
        -------
        PotentialFn
            A potential of the same class sharing the model, registry,
            input, and data.

        Raises
        ------
        ValueError
            If a name in ``values`` is a site of the registry or one of
            the observations.
        """
        clash = sorted(
            name
            for name in values
            if name in self._registry or name in self._base_observations
        )
        if clash:
            raise ValueError(
                f"PotentialFn.condition: {clash} are latent sites of the registry "
                "or observations, so they cannot also be conditioned on"
            )
        conditioned = copy.copy(self)
        conditioned._conditioned = dict(values)
        conditioned._observations = {**self._base_observations, **values}
        return conditioned

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
            If ``z`` does not have the registry's flat length, or a plate
            is subsampled, since the potential is the exact log density.
        Exception
            Any error ``log_joint`` raises that does not mark the
            position as outside the support.
        """
        refuse_subsampling("PotentialFn")
        self._counts[0] += 1
        trace_side, log_det_total = self._trace_side(z)
        try:
            log_p = self._plans.log_joint({**trace_side, **self._observations})
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

        Run once before sampling. The program is scored with
        `torch.distributions` validation enabled and no error
        converted, so mismatched row counts, data a score function
        reads that its distribution rejects, and errors in custom
        score functions raise here with their own type. A
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
            If the log-density at ``z`` is not finite.
        Exception
            Any error ``log_joint`` raises at ``z``.
        """
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
        self._counts[1] += 1
        if z.numel() == 0:
            # Every latent is held outside the flat vector (a model a
            # GibbsKernel samples alone): there is nothing to
            # differentiate with respect to.
            with torch.no_grad():
                return self.log_density(z).detach(), torch.zeros_like(z)
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

    @property
    def gibbs_sites(self) -> frozenset[str]:
        """The latent sites this kernel updates outside the flat vector.

        `MCMC.run` leaves these sites out of the registry, holds their
        values in `KernelState.gibbs_values`, and conditions the
        potential on them. A kernel that moves only the flat vector has
        none; a `GibbsKernel` has its blocks, and a composite kernel the
        union of its components'.
        """
        return frozenset()

    def tune(self, population: torch.Tensor) -> None:
        """Tune the kernel to a population of positions.

        A population sampler (`SMC`) calls this after `init`, at every
        temperature, with the current particles; the kernel may set the
        parameters it would otherwise adapt (a metric, a proposal scale)
        from the population's spread. The base implementation leaves the
        kernel as it is.

        Parameters
        ----------
        population : torch.Tensor
            The particles' flat positions, shape ``(N, D)``.
        """
        del population

    def configuration(self) -> dict[str, StateValue]:
        """The kernel's configuration, for telling kernels apart.

        A checkpoint records it and `MCMC.resume` refuses a checkpoint
        written by a kernel whose configuration differs. The base
        implementation records the class; concrete kernels add their
        hyperparameters, and composites their components'.

        Returns
        -------
        dict[str, StateValue]
            The configuration.
        """
        return {"kernel": f"{type(self).__module__}.{type(self).__qualname__}"}

    def state_dict(self) -> dict[str, StateValue]:
        """The kernel's mutable state for one chain.

        Everything a chain's continuation depends on beyond its
        `KernelState` and its generator: whether the kernel is adapting,
        and its adaptation state (step size, metric, schedule
        position). `MCMC` writes it to a checkpoint and restores it with
        `load_state_dict` after `init`, so a resumed chain continues bit
        for bit.

        Returns
        -------
        dict[str, StateValue]
            The state.
        """
        return {"is_adapting": self.is_adapting}

    def load_state_dict(self, state: dict[str, StateValue]) -> None:
        """Restore what `state_dict` returned.

        Called after `init` for the same chain, which has rebuilt the
        kernel's structures for the registry.

        Parameters
        ----------
        state : dict[str, StateValue]
            A dictionary `state_dict` produced.

        Raises
        ------
        ValueError
            If the dictionary lacks an entry or holds one of the wrong
            type.
        """
        flag = state.get("is_adapting")
        if not isinstance(flag, bool):
            raise ValueError(
                f"{type(self).__name__}.load_state_dict: is_adapting must be a bool"
            )
        self.is_adapting = flag

    @staticmethod
    def _evaluate_first(state: KernelState, potential: PotentialFn) -> None:
        """Fill in the log-density and gradient of a chain's first state.

        A kernel's `init` returns a state whose log-density and
        gradient are placeholders, since the potential is not known to
        `init`; the first transition of the chain (``step_count == 0``)
        evaluates them before using them.

        Parameters
        ----------
        state : KernelState
            The state a transition starts from.
        potential : PotentialFn
            The target.
        """
        if state.step_count == 0:
            ld, g = potential.value_and_grad(state.position)
            state.log_density = ld
            state.grad_log_density = g

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

    def start_adaptation(self, num_warmup: int) -> None:
        """Enter the adaptation (warmup) phase.

        Parameters
        ----------
        num_warmup : int
            Number of warmup transitions that follow before
            `stop_adaptation`, so that a kernel can lay out a warmup
            schedule.
        """
        del num_warmup
        self.is_adapting = True

    def stop_adaptation(self) -> None:
        """Freeze the kernel's adapted parameters."""
        self.is_adapting = False


def _with_generator[T](generator: torch.Generator | None, draw: Callable[[], T]) -> T:
    """Run a draw that uses Torch's global RNG on a given generator.

    `torch.distributions` samplers and the program's own draws read the
    global generator. The draw runs in a forked global state seeded from
    ``generator``, which then advances by what the draw consumed, so the
    caller's stream is used and the global state is left as it was.

    Parameters
    ----------
    generator : torch.Generator | None
        The CPU generator to draw from; ``None`` draws from a forked
        copy of the global state, leaving it untouched.
    draw : Callable[[], T]
        The draw.

    Returns
    -------
    T
        What ``draw`` returns.
    """
    with torch.random.fork_rng(devices=[]):
        if generator is not None:
            torch.random.set_rng_state(generator.get_state())
        try:
            return draw()
        finally:
            if generator is not None:
                generator.set_state(torch.random.get_rng_state())


__all__ = ["KernelState", "PotentialFn", "MCMCKernel", "StateValue"]
