"""MCMC orchestrator: chains, warmup, and posterior diagnostics.

The `MCMC` driver wraps a `MCMCKernel` and applies
it to a model: it builds the registry, draws an initial position
(from the prior, an existing guide, or a user-supplied dict),
runs ``num_warmup`` adaptation steps with
``kernel.start_adaptation(num_warmup)``, freezes the kernel via
``kernel.stop_adaptation()``, and then collects ``num_samples``
post-warmup samples per chain.

Chains are run sequentially within a process. Each chain receives
a local generator and fresh kernel adaptation state. An explicit
``seed`` reproduces a run; an explicit ``generator`` advances once
per run. Neither mode mutates Torch's global RNG state. For parallel
chains across a real multi-core / multi-GPU workload, wrap an
`MCMC` instance in a `torch.multiprocessing` pool.

The result is an `MCMCResult` carrying per-site posterior
draws (already pushed through the constraint bijectors so they
sit in the model's natural support). Its rank-normalized split
:math:`\\hat R` (Vehtari et al. 2021,
`doi:10.1214/20-BA1221 <https://doi.org/10.1214/20-BA1221>`_), bulk
and tail effective sample sizes are computed from those draws by
`quivers.diagnostics.rhat` and `quivers.diagnostics.ess`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
import math
from pathlib import Path
import time
from typing import Literal, cast

import torch
from torch.distributions import constraints as _constraints

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.plate import PlateDraw
from quivers.continuous.spaces import ContinuousSpace
from quivers.continuous.programs import MonadicProgram
from quivers.diagnostics.convergence import ess, rhat
from quivers.effects.trace_types import Trace
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.checkpoint import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_FORMAT_VERSION,
    MCMCCheckpoint,
    _CheckpointHeader,
    _digest,
    observation_digest,
    read_checkpoint,
    tensor_digest,
    write_checkpoint,
)
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
    StateValue,
    _check_observed_support,
    _with_generator,
)
from quivers.inference.registry import LatentRegistry, _require_one_run
from quivers.inference.trace import trace


InitStrategy = Literal["prior", "zero", "guide"]
"""How `MCMC` chooses each chain's initial point in the unconstrained
latent space: ``"prior"`` draws a small seeded standard-normal
perturbation of the origin, ``"zero"`` starts at the origin, and
``"guide"`` maps one draw of a fitted guide into the unconstrained
space."""


StopReason = Literal["draws", "seconds", "evaluations"]
"""Why a chain stopped: ``"draws"`` when it collected every draw,
``"seconds"`` when `MCMC`'s ``max_seconds`` ran out, and
``"evaluations"`` when its ``max_evaluations`` did."""

MCMCPhase = Literal["warmup", "sampling"]
"""The phase of a chain's transition: adaptation or sampling."""

MCMCCallback = Callable[[int, int, KernelState, MCMCPhase], None]
"""A progress hook ``callback(chain, iteration, state, phase)``, called
after every transition with the chain's index, the transition's index
within its phase (counted from zero), the state it produced, and the
phase. It runs in the sampling loop; it should not draw from the
chain's generator or modify the state."""


@dataclass
class MCMCResult:
    """Posterior samples, per-chain sampler statistics, and diagnostics.

    The convergence diagnostics are derived from ``samples`` on first
    access by the public estimators of `quivers.diagnostics`, so they
    agree exactly with `quivers.diagnostics.rhat` and
    `quivers.diagnostics.ess` applied to the same draws.

    Attributes
    ----------
    samples : dict[str, torch.Tensor]
        Per-site posterior draws on the constrained support, the sites a
        `GibbsKernel` updates included.
        Shape ``(num_chains, num_samples, *site_shape)``, where
        ``site_shape`` is the site's `LatentSite.constrained_shape`:
        ``()`` for a scalar site, ``(d,)`` for a vector site, and
        ``(|A|, ...)`` for a plate site over ``A``. The draws carry
        no batch axis; `LatentRegistry.trace_values` lifts one draw
        to the ``(batch, ...)`` values a program's ``log_joint`` and
        `trace` consume.
    log_densities : torch.Tensor
        Unconstrained-space log-density (Jacobian-corrected) at
        every posterior draw. Shape ``(num_chains, num_samples)``.
    acceptance_rates : torch.Tensor
        Per-chain post-warmup acceptance rate. Shape
        ``(num_chains,)``. For a composite kernel a transition counts as
        accepted when it moved the chain (its position or a Gibbs
        value).
    divergence_counts : torch.Tensor
        Per-chain post-warmup divergence count. Shape
        ``(num_chains,)``.
    num_warmup : int
        Warmup iterations per chain.
    num_samples : int
        Retained draws per chain.
    component_acceptance : dict[str, torch.Tensor]
        For a composite kernel, each component's post-warmup acceptance
        rate per chain, shape ``(num_chains,)``, keyed by component name
        (``"outer.inner"`` for a component of a nested composite): the
        mean of the component's ``"accepted"`` statistic over the
        transitions that ran it, ``nan`` for a chain in which it never
        ran. Empty for a kernel that is not composite.
    sample_stats : dict[str, torch.Tensor]
        Per-draw sampler statistics, each of shape ``(num_chains,
        num_samples)`` and dtype ``float64``, keyed as the kernel's
        `KernelState.stats`: for `NUTSKernel` ``"accept_prob"``,
        ``"accepted"``, ``"diverged"`` (``0.0`` or ``1.0``),
        ``"step_size"``, ``"tree_depth"``, ``"num_steps"`` (leapfrog
        steps), and ``"energy"`` (the Hamiltonian at the draw);
        `HMCKernel` the same without ``"tree_depth"``. A composite's
        entries carry each component's name as a prefix
        (``"nuts.energy"``), and an entry a transition did not report
        (an unselected mixture component's) is ``nan``.
    stopped_by : tuple[StopReason, ...]
        Why each chain stopped: ``"draws"`` when it finished,
        ``"seconds"`` or ``"evaluations"`` when a budget stopped it.
    num_draws : torch.Tensor
        Post-warmup draws each chain made, shape ``(num_chains,)``. When
        a budget stops a chain early, every per-draw quantity keeps the
        first ``min(num_draws)`` draws of each chain, and ``num_samples``
        is that number.
    num_evaluations : torch.Tensor
        Potential evaluations per chain, gradient ones included, over
        every call of `MCMC.run` and `MCMC.resume`. Shape
        ``(num_chains,)``.
    num_gradient_evaluations : torch.Tensor
        Gradient evaluations per chain. Shape ``(num_chains,)``.
    """

    samples: dict[str, torch.Tensor]
    log_densities: torch.Tensor
    acceptance_rates: torch.Tensor
    divergence_counts: torch.Tensor
    num_warmup: int
    num_samples: int
    component_acceptance: dict[str, torch.Tensor] = field(default_factory=dict)
    sample_stats: dict[str, torch.Tensor] = field(default_factory=dict)
    stopped_by: tuple[StopReason, ...] = ()
    num_draws: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int64)
    )
    num_evaluations: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int64)
    )
    num_gradient_evaluations: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int64)
    )

    @property
    def truncated(self) -> bool:
        """Whether a budget stopped any chain before its last draw."""
        return any(reason != "draws" for reason in self.stopped_by)

    @property
    def num_chains(self) -> int:
        """Number of chains."""
        return int(self.log_densities.shape[0])

    @property
    def mean_acceptance(self) -> float:
        """Post-warmup acceptance rate averaged over chains."""
        return float(self.acceptance_rates.mean().item())

    @property
    def total_divergences(self) -> int:
        """Post-warmup divergences summed over chains."""
        return int(self.divergence_counts.sum().item())

    @cached_property
    def r_hat(self) -> dict[str, torch.Tensor]:
        """Per-site rank-normalized split :math:`\\hat R`.

        Each coordinate reports the maximum of the rank-normalized
        and folded statistics (`quivers.diagnostics.rhat`). Each
        site's float64 tensor has shape ``site_shape``, so a scalar
        site's tensor is zero-dimensional.
        """
        return {name: rhat(draws) for name, draws in self.samples.items()}

    @cached_property
    def ess(self) -> dict[str, torch.Tensor]:
        """Per-site bulk effective sample size pooled across chains.

        Computed by ``quivers.diagnostics.ess(draws, method="bulk")``;
        same shape convention as `r_hat`.
        """
        return {name: ess(draws, method="bulk") for name, draws in self.samples.items()}

    @cached_property
    def ess_tail(self) -> dict[str, torch.Tensor]:
        """Per-site tail effective sample size pooled across chains.

        The smaller of the 5% and 95% quantile effective sample sizes,
        computed by ``quivers.diagnostics.ess(draws, method="tail")``;
        same shape convention as `r_hat`.
        """
        return {name: ess(draws, method="tail") for name, draws in self.samples.items()}


class _ChainRecord:
    """The progress of one chain: everything a checkpoint stores for it.

    Attributes
    ----------
    phase : str
        ``"warmup"``, ``"sampling"``, or ``"done"``.
    iteration : int
        Transitions completed in the current phase.
    state : KernelState
        The chain's state.
    kernel_state : dict[str, StateValue]
        The kernel's `MCMCKernel.state_dict` at the last checkpoint.
    generator_state : torch.Tensor
        The chain generator's state at the last checkpoint.
    draws : dict[str, list[torch.Tensor]]
        Each site's post-warmup draws.
    log_densities : list[float]
        The log-density at each draw.
    stats : list[dict[str, float]]
        Each draw's transition statistics.
    divergences : int
        Post-warmup divergent transitions.
    evaluations : int
        Potential evaluations over the chain's life.
    gradient_evaluations : int
        Gradient evaluations over the chain's life.
    stopped_by : str | None
        Why the chain last stopped.
    """

    __slots__ = (
        "phase",
        "iteration",
        "state",
        "kernel_state",
        "generator_state",
        "draws",
        "log_densities",
        "stats",
        "divergences",
        "evaluations",
        "gradient_evaluations",
        "stopped_by",
    )

    def __init__(
        self, phase: str, state: KernelState, site_names: tuple[str, ...]
    ) -> None:
        self.phase = phase
        self.iteration = 0
        self.state = state
        self.kernel_state: dict[str, StateValue] = {}
        self.generator_state = torch.empty(0, dtype=torch.uint8)
        self.draws: dict[str, list[torch.Tensor]] = {n: [] for n in site_names}
        self.log_densities: list[float] = []
        self.stats: list[dict[str, float]] = []
        self.divergences = 0
        self.evaluations = 0
        self.gradient_evaluations = 0
        self.stopped_by: str | None = None

    def state_dict(self) -> dict[str, StateValue]:
        """The record as a checkpoint stores it.

        Returns
        -------
        dict[str, StateValue]
            Every attribute, draws stacked per site.
        """
        stats: list[StateValue] = [dict(entry) for entry in self.stats]
        return {
            "phase": self.phase,
            "iteration": self.iteration,
            "state": self.state.state_dict(),
            "kernel_state": self.kernel_state,
            "generator_state": self.generator_state,
            "draws": {
                name: torch.stack(values) if values else None
                for name, values in self.draws.items()
            },
            "log_densities": torch.tensor(self.log_densities, dtype=torch.float64),
            "stats": stats,
            "divergences": self.divergences,
            "evaluations": self.evaluations,
            "gradient_evaluations": self.gradient_evaluations,
            "stopped_by": self.stopped_by,
        }

    @classmethod
    def from_state_dict(
        cls, state: StateValue, site_names: tuple[str, ...]
    ) -> "_ChainRecord":
        """Rebuild a record from a checkpoint.

        Parameters
        ----------
        state : StateValue
            What `state_dict` stored.
        site_names : tuple[str, ...]
            The run's sites.

        Returns
        -------
        _ChainRecord
            The record.

        Raises
        ------
        ValueError
            If the stored record is malformed or holds other sites.
        """
        if not isinstance(state, dict):
            raise ValueError("MCMC.resume: a chain record is not a dictionary")
        kernel_state = state.get("kernel_state")
        generator_state = state.get("generator_state")
        draws = state.get("draws")
        log_densities = state.get("log_densities")
        stats = state.get("stats")
        chain_state = state.get("state")
        phase, iteration = state.get("phase"), state.get("iteration")
        divergences = state.get("divergences")
        evaluations = state.get("evaluations")
        gradient_evaluations = state.get("gradient_evaluations")
        stopped_by = state.get("stopped_by")
        if not (
            isinstance(kernel_state, dict)
            and isinstance(generator_state, torch.Tensor)
            and isinstance(draws, dict)
            and isinstance(log_densities, torch.Tensor)
            and isinstance(stats, list)
            and isinstance(chain_state, dict)
            and phase in ("warmup", "sampling", "done")
            and isinstance(iteration, int)
            and isinstance(divergences, int)
            and isinstance(evaluations, int)
            and isinstance(gradient_evaluations, int)
            and (stopped_by is None or isinstance(stopped_by, str))
        ):
            raise ValueError("MCMC.resume: a chain record is malformed")
        if set(draws) != set(site_names):
            raise ValueError(
                f"MCMC.resume: the checkpoint's draws are of sites {sorted(draws)}; "
                f"the run's sites are {sorted(site_names)}"
            )
        assert isinstance(phase, str)
        record = cls(phase, KernelState.from_state_dict(chain_state), site_names)
        record.iteration = iteration
        record.kernel_state = kernel_state
        record.generator_state = generator_state
        for name, stacked in draws.items():
            if stacked is None:
                continue
            if not isinstance(stacked, torch.Tensor):
                raise ValueError(f"MCMC.resume: the draws of {name!r} are malformed")
            record.draws[name] = list(stacked.unbind(0))
        record.log_densities = log_densities.tolist()
        for entry in stats:
            if not isinstance(entry, dict):
                raise ValueError("MCMC.resume: a draw's statistics are malformed")
            record.stats.append(
                {k: float(v) for k, v in entry.items() if isinstance(v, float)}
            )
        record.divergences = divergences
        record.evaluations = evaluations
        record.gradient_evaluations = gradient_evaluations
        record.stopped_by = stopped_by
        return record


class _RunSetup:
    """What one call of `MCMC.run` or `MCMC.resume` derives from the
    model and data.

    Attributes
    ----------
    model : MonadicProgram
        The model.
    x : torch.Tensor
        The one-row input.
    observations : dict[str, torch.Tensor]
        The data.
    registry : LatentRegistry
        The flat vector's registry, without the Gibbs sites.
    potential : PotentialFn
        The potential.
    gibbs_sites : frozenset[str]
        The kernel's Gibbs sites.
    gibbs_morphisms : dict[str, ContinuousMorphism]
        Each Gibbs site's distribution.
    site_names : tuple[str, ...]
        Every sampled site, the registry's first.
    """

    __slots__ = (
        "model",
        "x",
        "observations",
        "registry",
        "potential",
        "gibbs_sites",
        "gibbs_morphisms",
        "site_names",
    )

    def __init__(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        registry: LatentRegistry,
        potential: PotentialFn,
        gibbs_sites: frozenset[str],
        gibbs_morphisms: dict[str, ContinuousMorphism],
    ) -> None:
        self.model = model
        self.x = x
        self.observations = observations
        self.registry = registry
        self.potential = potential
        self.gibbs_sites = gibbs_sites
        self.gibbs_morphisms = gibbs_morphisms
        self.site_names = (*registry.names, *sorted(gibbs_sites))

    def fingerprint(self) -> str:
        """A digest of the model and data the run samples.

        It covers the registry's layout, the Gibbs sites, the input,
        every datum by content (through `observation_digest`, which
        renders tensors, numbers, arrays, and nested sequences of tokens
        alike), and the potential's value at two fixed positions under
        Gibbs values drawn from a fixed generator, so a change to a
        prior, a likelihood, or the data changes it.

        Returns
        -------
        str
            The hexadecimal digest.
        """
        dim = self.registry.total_unconstrained_dim
        gibbs = MCMC._initial_gibbs_values(
            self.model,
            self.x,
            self.observations,
            self.gibbs_sites,
            torch.Generator().manual_seed(0),
        )
        probe = self.potential.condition(gibbs) if gibbs else self.potential
        with torch.no_grad():
            values: list[StateValue] = [
                float(probe.log_density(point))
                for point in (torch.zeros(dim), torch.linspace(-0.5, 0.5, dim))
            ]
        layout: list[StateValue] = [
            [
                site.name,
                list(site.constrained_shape),
                list(site.unconstrained_shape),
                str(site.support),
            ]
            for site in self.registry.sites.values()
        ]
        gibbs_names: list[StateValue] = list(sorted(self.gibbs_sites))
        data: dict[str, StateValue] = {
            name: observation_digest(name, value)
            for name, value in sorted(self.observations.items())
        }
        description: dict[str, StateValue] = {
            "layout": layout,
            "gibbs": gibbs_names,
            "x": tensor_digest(self.x),
            "data": data,
            "probes": values,
        }
        return _digest(description)


class MCMC:
    """MCMC chain runner.

    Parameters
    ----------
    kernel : MCMCKernel
        Markov kernel (e.g. `HMCKernel`, `NUTSKernel`, or a composite).
    num_warmup : int
        Number of adaptation steps. The kernel's adaptation
        machinery (dual averaging, Welford covariance) runs over
        this prefix.
    num_samples : int
        Post-warmup samples per chain.
    num_chains : int
        Independent chains. Default ``4`` (Stan / NumPyro default).
    init_strategy : {"prior", "zero", "guide"}
        How to pick each chain's initial position.
    seed : int, optional
        Base seed used to derive an independent local generator for
        each chain. Repeated calls to `run` with the same seed are
        reproducible and do not consume Torch's global RNG.
    generator : torch.Generator, optional
        Caller-owned CPU generator used to draw one base seed per
        `run`. The generator advances, while Torch's global RNG does
        not. Mutually exclusive with ``seed``.
    checkpoint : MCMCCheckpoint, optional
        Where and how often to checkpoint the chains, so that `resume`
        can continue an interrupted run exactly.
    max_seconds : float, optional
        Wall-clock budget per chain and per call of `run` or `resume`.
    max_evaluations : int, optional
        Budget of potential evaluations (gradient ones included) per
        chain and per call.
    callback : MCMCCallback, optional
        Progress hook called after every transition.

    Raises
    ------
    ValueError
        If a count or budget is out of range, ``seed`` and
        ``generator`` are both given, or ``generator`` is not on the
        CPU.
    TypeError
        If ``seed`` is not an int or ``generator`` not a generator.
    """

    def __init__(
        self,
        kernel: MCMCKernel,
        num_warmup: int,
        num_samples: int,
        num_chains: int = 4,
        init_strategy: InitStrategy = "prior",
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        checkpoint: MCMCCheckpoint | None = None,
        max_seconds: float | None = None,
        max_evaluations: int | None = None,
        callback: MCMCCallback | None = None,
    ) -> None:
        if num_warmup < 0:
            raise ValueError(f"MCMC: num_warmup must be >= 0, got {num_warmup}")
        if num_samples < 1:
            raise ValueError(f"MCMC: num_samples must be >= 1, got {num_samples}")
        if num_chains < 1:
            raise ValueError(f"MCMC: num_chains must be >= 1, got {num_chains}")
        if seed is not None and generator is not None:
            raise ValueError("MCMC: seed and generator are mutually exclusive")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError(f"MCMC: seed must be an int, got {type(seed).__name__}")
        if generator is not None:
            if not isinstance(generator, torch.Generator):
                raise TypeError(
                    "MCMC: generator must be a torch.Generator, "
                    f"got {type(generator).__name__}"
                )
            if generator.device.type != "cpu":
                raise ValueError(
                    "MCMC: generator must be a CPU generator because MCMC "
                    "currently runs chain state on CPU"
                )
        if max_seconds is not None and not max_seconds > 0:
            raise ValueError(f"MCMC: max_seconds must be > 0, got {max_seconds}")
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError(
                f"MCMC: max_evaluations must be >= 1, got {max_evaluations}"
            )
        self.kernel = kernel
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.init_strategy = init_strategy
        self.seed = seed
        self.generator = generator
        self.checkpoint = checkpoint
        self.max_seconds = max_seconds
        self.max_evaluations = max_evaluations
        self.callback = callback

    def _run_seed(self) -> int:
        """Return the base seed for one `run` without using global RNG state."""
        if self.generator is not None:
            return int(
                torch.randint(
                    0,
                    2**63 - 1,
                    (),
                    dtype=torch.int64,
                    generator=self.generator,
                ).item()
            )
        if self.seed is not None:
            return self.seed
        # Preserve compatibility with callers that use
        # ``torch.manual_seed`` while avoiding any draw from, or write
        # to, the ambient generator.
        return torch.initial_seed()

    @staticmethod
    def _chain_generator(base_seed: int, chain_idx: int) -> torch.Generator:
        """Derive a chain stream with SplitMix64 seed mixing."""
        mask = (1 << 64) - 1
        value = (base_seed + (chain_idx + 1) * 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        return torch.Generator().manual_seed(value)

    @staticmethod
    def _guide_rsample(
        guide: Guide,
        x: torch.Tensor,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Sample a guide through ``generator`` while restoring global RNG."""
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(generator.get_state())
            try:
                return guide.rsample(x)
            finally:
                generator.set_state(torch.random.get_rng_state())

    def _initial_position(
        self,
        registry: LatentRegistry,
        guide: Guide | None,
        generator: torch.Generator,
    ) -> torch.Tensor:
        D = registry.total_unconstrained_dim
        if self.init_strategy == "zero":
            return torch.zeros(D)
        if self.init_strategy == "guide":
            if guide is None:
                raise ValueError(
                    "MCMC.run: init_strategy='guide' requires a guide argument"
                )
            # Sample the guide once with a batch of one and read that
            # row back as one draw per site.
            trace_side = self._guide_rsample(guide, torch.zeros(1, 1), generator)
            draw = registry.trace_row(trace_side)
            unc, _ = registry.to_unconstrained(draw)
            return registry.flatten_unconstrained(unc)
        # "prior": draw a fresh standard-normal position, scaled
        # mildly so chains spread out without diverging.
        return 0.1 * torch.randn(D, generator=generator)

    @staticmethod
    def _to_constrained(
        registry: LatentRegistry,
        z_flat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Map a flat unconstrained position to one constrained draw
        per site.

        Parameters
        ----------
        registry : LatentRegistry
            Registry of the sampled model.
        z_flat : torch.Tensor
            Flat unconstrained position, shape ``(D,)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Each site's draw at `LatentSite.constrained_shape`.
        """
        constrained, _ = registry.to_constrained(
            registry.unflatten_unconstrained(z_flat)
        )
        return constrained

    def run(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        guide: Guide | None = None,
    ) -> MCMCResult:
        """Run the configured kernel for ``num_chains`` chains of
        ``num_warmup + num_samples`` steps each.

        Each chain's initial position is scored by
        `PotentialFn.check_initial` before sampling, so an error in
        the program or its data (observed values outside a
        likelihood's support or not finite, mismatched row counts, an
        error in a custom score function) raises here, with a note
        naming the chain, rather than producing a chain that rejects
        every proposal. During sampling only positions outside the
        model's support score ``-inf``; other errors propagate.

        With a ``checkpoint`` the chains are checkpointed as they run,
        and `resume` continues them. With a budget, a chain stops at the
        first transition boundary past it, and the result records why
        in `MCMCResult.stopped_by`.

        Parameters
        ----------
        model : MonadicProgram
            The program to sample.
        x : torch.Tensor
            Program input, shape ``(batch, ...)``.
        observations : dict[str, torch.Tensor]
            Observed-site values and host data.
        guide : Guide, optional
            Fitted guide, required by ``init_strategy="guide"``.

        Returns
        -------
        MCMCResult
            Posterior draws and diagnostics.

        Raises
        ------
        ValueError
            If ``x`` has more than one row (MCMC targets the posterior
            of one run), an observed value is not finite or lies
            outside its site's support, a chain's initial position is
            not finite, or the log-density at an initial position is
            not finite.
        FileExistsError
            If ``checkpoint`` names a directory that already holds a
            checkpoint; `resume` continues it.
        Exception
            Any error the program raises at an initial position, or
            raises during sampling for a reason other than a position
            outside its support.
        """
        if self.checkpoint is not None and self.checkpoint.path.exists():
            raise FileExistsError(
                f"MCMC.run: {self.checkpoint.path} already holds a checkpoint; "
                "continue it with MCMC.resume, or remove it to start afresh"
            )
        run_seed = self._run_seed()
        setup = self._setup(model, x, observations, "MCMC.run", run_seed)
        header = self._header(setup, run_seed)
        records: list[_ChainRecord | None] = [None] * self.num_chains
        return self._execute(setup, header, records, guide)

    def resume(
        self,
        directory: str | Path,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        guide: Guide | None = None,
    ) -> MCMCResult:
        """Continue a run from its checkpoint.

        Finished chains are kept as they are; a chain a budget or an
        interruption stopped continues from its last checkpoint; chains
        not yet started start as `run` would start them. The draws are
        bit for bit those of an uninterrupted run with the same seed.
        Budgets apply afresh to this call. Further checkpoints follow
        this instance's ``checkpoint``, if any.

        Parameters
        ----------
        directory : str | Path
            The directory holding the checkpoint.
        model : MonadicProgram
            The program the checkpoint was written for.
        x : torch.Tensor
            The input it was written for.
        observations : dict[str, torch.Tensor]
            The data it was written for.
        guide : Guide, optional
            Fitted guide, for chains not yet started under
            ``init_strategy="guide"``.

        Returns
        -------
        MCMCResult
            Posterior draws and diagnostics of every chain.

        Raises
        ------
        FileNotFoundError
            If ``directory`` holds no checkpoint.
        ValueError
            If the checkpoint is of another format version, or was
            written for a different model or data, kernel
            configuration, run settings, or seed.
        """
        stored_header, stored_chains = read_checkpoint(
            Path(directory) / CHECKPOINT_FILENAME
        )
        setup = self._setup(
            model, x, observations, "MCMC.resume", stored_header.run_seed
        )
        header = self._header(setup, stored_header.run_seed)
        for field_name, what in (
            ("model_fingerprint", "a different model or data"),
            ("kernel_fingerprint", "a different kernel configuration"),
            (
                "run_fingerprint",
                "different run settings (warmup, samples, chains, or initialization)",
            ),
        ):
            if getattr(header, field_name) != getattr(stored_header, field_name):
                raise ValueError(
                    f"MCMC.resume: the checkpoint in {directory} was written for "
                    f"{what}; resume it with the model, data, kernel, and settings "
                    "it was written with"
                )
        if self.seed is not None and self.seed != stored_header.run_seed:
            raise ValueError(
                f"MCMC.resume: the checkpoint was written with seed "
                f"{stored_header.run_seed}, not {self.seed}"
            )
        if len(stored_chains) != self.num_chains:
            raise ValueError(
                f"MCMC.resume: the checkpoint holds {len(stored_chains)} chains, "
                f"not {self.num_chains}"
            )
        records: list[_ChainRecord | None] = [
            None
            if stored is None
            else _ChainRecord.from_state_dict(stored, setup.site_names)
            for stored in stored_chains
        ]
        return self._execute(setup, header, records, guide)

    def _setup(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        consumer: str,
        run_seed: int,
    ) -> _RunSetup:
        """Check the inputs and build the registry and potential.

        Parameters
        ----------
        model : MonadicProgram
            The model.
        x : torch.Tensor
            The input.
        observations : dict[str, torch.Tensor]
            The data.
        consumer : str
            The method named in errors.
        run_seed : int
            The run's base seed, from which the recording run's stream
            is derived.

        Returns
        -------
        _RunSetup
            The derived structures.

        Raises
        ------
        ValueError
            If the input has several rows, the data are outside their
            support, or the kernel's Gibbs sites do not fit the model.
        """
        _require_one_run(x, consumer)
        _check_observed_support(model, observations, consumer)
        # The latents are the sites one run at (x, observations) reaches,
        # every position of a scan included. The run draws its latents
        # from a stream derived from the run's seed, so a seeded run
        # records the same latents whatever the global RNG holds, and
        # Torch's global RNG is left as it was.
        recording = self._chain_generator(run_seed, -1)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(recording.get_state())
                recorded = trace(model, x, observations=observations)
        except Exception as error:
            error.add_note(
                f"{consumer}: raised while recording the run its latents are read "
                "from, before any sampling"
            )
            raise
        gibbs_sites = self.kernel.gibbs_sites
        gibbs_morphisms = _check_gibbs_sites(recorded, gibbs_sites)
        registry = LatentRegistry.from_trace(model, recorded, exclude=gibbs_sites)
        potential = PotentialFn(model, registry, x, observations)
        return _RunSetup(
            model, x, observations, registry, potential, gibbs_sites, gibbs_morphisms
        )

    def _header(self, setup: _RunSetup, run_seed: int) -> _CheckpointHeader:
        """The checkpoint header of a run.

        Parameters
        ----------
        setup : _RunSetup
            The run's structures.
        run_seed : int
            The run's base seed.

        Returns
        -------
        _CheckpointHeader
            The header.
        """
        settings: dict[str, StateValue] = {
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "init_strategy": self.init_strategy,
        }
        return _CheckpointHeader(
            format_version=CHECKPOINT_FORMAT_VERSION,
            library="quivers",
            model_fingerprint=setup.fingerprint(),
            kernel_fingerprint=_digest(self.kernel.configuration()),
            run_fingerprint=_digest(settings),
            run_seed=run_seed,
        )

    def _start_chain(
        self,
        setup: _RunSetup,
        chain: int,
        generator: torch.Generator,
        guide: Guide | None,
    ) -> _ChainRecord:
        """Initialize a chain and enter its first phase.

        Parameters
        ----------
        setup : _RunSetup
            The run's structures.
        chain : int
            The chain's index.
        generator : torch.Generator
            The chain's generator.
        guide : Guide | None
            The guide for ``init_strategy="guide"``.

        Returns
        -------
        _ChainRecord
            The chain's record, in warmup (or sampling, with no warmup).

        Raises
        ------
        ValueError
            If the initial position is not finite or scores a
            non-finite log-density.
        """
        init_pos = self._initial_position(setup.registry, guide, generator)
        if not torch.isfinite(init_pos).all():
            raise ValueError(
                f"MCMC.run: chain {chain} initial_position must contain "
                "only finite values"
            )
        gibbs_values = self._initial_gibbs_values(
            setup.model, setup.x, setup.observations, setup.gibbs_sites, generator
        )
        chain_potential = (
            setup.potential.condition(gibbs_values) if gibbs_values else setup.potential
        )
        try:
            chain_potential.check_initial(init_pos)
        except Exception as error:
            error.add_note(
                f"MCMC.run: raised while scoring chain {chain}'s initial "
                "position, before any sampling"
            )
            raise
        self.kernel._set_generator(generator)
        state = self.kernel.init(
            setup.registry, setup.model, setup.x, setup.observations, init_pos
        )
        state.gibbs_values = gibbs_values
        if self.num_warmup > 0:
            self.kernel.start_adaptation(self.num_warmup)
            return _ChainRecord("warmup", state, setup.site_names)
        return _ChainRecord("sampling", _begin_sampling(state), setup.site_names)

    def _execute(
        self,
        setup: _RunSetup,
        header: _CheckpointHeader,
        records: list[_ChainRecord | None],
        guide: Guide | None,
    ) -> MCMCResult:
        """Run every unfinished chain and assemble the result.

        Parameters
        ----------
        setup : _RunSetup
            The run's structures.
        header : _CheckpointHeader
            The checkpoint header.
        records : list[_ChainRecord | None]
            Each chain's progress, ``None`` for a chain not yet started.
        guide : Guide | None
            The guide for ``init_strategy="guide"``.

        Returns
        -------
        MCMCResult
            Every chain's draws and diagnostics.
        """
        for chain in range(self.num_chains):
            record = records[chain]
            if record is not None and record.phase == "done":
                continue
            generator = self._chain_generator(header.run_seed, chain)
            if record is None:
                record = self._start_chain(setup, chain, generator, guide)
                records[chain] = record
            else:
                self.kernel._set_generator(generator)
                self.kernel.init(
                    setup.registry,
                    setup.model,
                    setup.x,
                    setup.observations,
                    record.state.position,
                )
                self.kernel.load_state_dict(record.kernel_state)
                generator.set_state(record.generator_state)
            self._advance(setup, header, records, chain, record, generator)
        finished = [record for record in records if record is not None]
        return self._result(setup, finished)

    def _advance(
        self,
        setup: _RunSetup,
        header: _CheckpointHeader,
        records: list[_ChainRecord | None],
        chain: int,
        record: _ChainRecord,
        generator: torch.Generator,
    ) -> None:
        """Run one chain until it finishes or a budget stops it.

        Parameters
        ----------
        setup : _RunSetup
            The run's structures.
        header : _CheckpointHeader
            The checkpoint header.
        records : list[_ChainRecord | None]
            Every chain's record, for checkpoints.
        chain : int
            The chain's index.
        record : _ChainRecord
            Its record, advanced in place.
        generator : torch.Generator
            Its generator.
        """
        potential = setup.potential
        started = time.monotonic()
        last_write = started
        since_write = 0
        evaluations_at_start = potential.num_evaluations
        record.stopped_by = None
        while True:
            if record.phase == "warmup" and record.iteration == self.num_warmup:
                self.kernel.stop_adaptation()
                record.state = _begin_sampling(record.state)
                record.phase = "sampling"
                record.iteration = 0
                continue
            if record.phase == "sampling" and record.iteration == self.num_samples:
                record.phase = "done"
                record.stopped_by = "draws"
                self._write(header, records, record, generator)
                return
            spent = potential.num_evaluations - evaluations_at_start
            reason: StopReason | None = None
            if self.max_seconds is not None and (
                time.monotonic() - started >= self.max_seconds
            ):
                reason = "seconds"
            elif self.max_evaluations is not None and spent >= self.max_evaluations:
                reason = "evaluations"
            if reason is not None:
                record.stopped_by = reason
                self._write(header, records, record, generator)
                return
            evaluations = potential.num_evaluations
            gradients = potential.num_gradient_evaluations
            record.state = self._transition(record.state, potential)
            record.evaluations += potential.num_evaluations - evaluations
            record.gradient_evaluations += (
                potential.num_gradient_evaluations - gradients
            )
            phase: MCMCPhase = "warmup" if record.phase == "warmup" else "sampling"
            if phase == "sampling":
                self._record_draw(setup, record)
            if self.callback is not None:
                self.callback(chain, record.iteration, record.state, phase)
            record.iteration += 1
            since_write += 1
            if self.checkpoint is not None and (
                (
                    self.checkpoint.every is not None
                    and since_write >= self.checkpoint.every
                )
                or (
                    self.checkpoint.every_seconds is not None
                    and time.monotonic() - last_write >= self.checkpoint.every_seconds
                )
            ):
                self._write(header, records, record, generator)
                since_write = 0
                last_write = time.monotonic()

    def _record_draw(self, setup: _RunSetup, record: _ChainRecord) -> None:
        """Append the chain's current state to its draws.

        Parameters
        ----------
        setup : _RunSetup
            The run's structures.
        record : _ChainRecord
            The chain's record.
        """
        state = record.state
        if state.diverged:
            record.divergences += 1
        for name, value in self._to_constrained(setup.registry, state.position).items():
            record.draws[name].append(value)
        for name in setup.gibbs_sites:
            record.draws[name].append(
                _gibbs_draw(state.gibbs_values[name], setup.gibbs_morphisms[name])
            )
        record.log_densities.append(float(state.log_density))
        record.stats.append(dict(state.stats))

    def _write(
        self,
        header: _CheckpointHeader,
        records: list[_ChainRecord | None],
        record: _ChainRecord,
        generator: torch.Generator,
    ) -> None:
        """Checkpoint every chain, capturing the running chain's kernel
        and generator.

        Parameters
        ----------
        header : _CheckpointHeader
            The header.
        records : list[_ChainRecord | None]
            Every chain's record.
        record : _ChainRecord
            The running chain's record.
        generator : torch.Generator
            The running chain's generator.
        """
        record.kernel_state = self.kernel.state_dict()
        record.generator_state = generator.get_state()
        if self.checkpoint is None:
            return
        write_checkpoint(
            self.checkpoint.path,
            header,
            [None if r is None else r.state_dict() for r in records],
        )

    def _result(self, setup: _RunSetup, records: list[_ChainRecord]) -> MCMCResult:
        """Assemble the result from every chain's record.

        Every per-draw quantity keeps the first ``min(num_draws)`` draws
        of each chain, so chains a budget cut short line up; the
        per-chain totals cover every draw each chain made.

        Parameters
        ----------
        setup : _RunSetup
            The run's structures.
        records : list[_ChainRecord]
            Every chain's record.

        Returns
        -------
        MCMCResult
            The result.
        """
        num_draws = [len(record.log_densities) for record in records]
        kept = min(num_draws)
        samples: dict[str, torch.Tensor] = {}
        for name in setup.site_names:
            per_chain = []
            for record in records:
                draws = record.draws[name][:kept]
                if draws:
                    per_chain.append(torch.stack(draws))
                else:
                    per_chain.append(self._empty_draws(setup, record, name))
            samples[name] = torch.stack(per_chain)
        log_densities = torch.tensor(
            [record.log_densities[:kept] for record in records], dtype=torch.float32
        ).reshape(len(records), kept)
        keys = sorted(
            {key for record in records for entry in record.stats for key in entry}
        )
        sample_stats = {
            key: torch.tensor(
                [
                    [entry.get(key, float("nan")) for entry in record.stats[:kept]]
                    for record in records
                ],
                dtype=torch.float64,
            ).reshape(len(records), kept)
            for key in keys
        }
        acceptance = torch.tensor(
            [
                record.state.accept_count / count if count else float("nan")
                for record, count in zip(records, num_draws, strict=True)
            ]
        )
        return MCMCResult(
            samples=samples,
            log_densities=log_densities,
            acceptance_rates=acceptance,
            divergence_counts=torch.tensor(
                [record.divergences for record in records], dtype=torch.int64
            ),
            num_warmup=self.num_warmup,
            num_samples=kept,
            component_acceptance=_component_acceptance(
                [record.stats for record in records]
            ),
            sample_stats=sample_stats,
            stopped_by=tuple(
                cast(StopReason, record.stopped_by or "draws") for record in records
            ),
            num_draws=torch.tensor(num_draws, dtype=torch.int64),
            num_evaluations=torch.tensor(
                [record.evaluations for record in records], dtype=torch.int64
            ),
            num_gradient_evaluations=torch.tensor(
                [record.gradient_evaluations for record in records], dtype=torch.int64
            ),
        )

    def _empty_draws(
        self, setup: _RunSetup, record: _ChainRecord, name: str
    ) -> torch.Tensor:
        """No draws of a site, at the site's shape.

        Parameters
        ----------
        setup : _RunSetup
            The run's structures.
        record : _ChainRecord
            A chain's record, whose state gives a Gibbs site's shape.
        name : str
            The site.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(0, *site_shape)``.
        """
        if name in setup.gibbs_sites:
            draw = _gibbs_draw(
                record.state.gibbs_values[name], setup.gibbs_morphisms[name]
            )
            return torch.empty((0, *draw.shape), dtype=draw.dtype)
        return torch.empty((0, *setup.registry[name].constrained_shape))

    def _transition(self, state: KernelState, potential: PotentialFn) -> KernelState:
        """Advance a chain by one kernel transition.

        Parameters
        ----------
        state : KernelState
            The current state.
        potential : PotentialFn
            The run's potential, conditioned here on the state's Gibbs
            values.

        Returns
        -------
        KernelState
            The next state.
        """
        if state.gibbs_values:
            potential = potential.condition(state.gibbs_values)
        return self.kernel.step(state, potential)

    @staticmethod
    def _initial_gibbs_values(
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        gibbs_sites: frozenset[str],
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Draw a chain's starting values of the Gibbs sites.

        The values come from one run of the model at the data, drawn
        from the chain's generator: each Gibbs site from its prior given
        the sites drawn before it.

        Parameters
        ----------
        model : MonadicProgram
            The model.
        x : torch.Tensor
            The one-row input.
        observations : dict[str, torch.Tensor]
            The data.
        gibbs_sites : frozenset[str]
            The Gibbs sites.
        generator : torch.Generator
            The chain's generator.

        Returns
        -------
        dict[str, torch.Tensor]
            Trace-side values of the Gibbs sites; empty when there are
            none, in which case the generator is not used.
        """
        if not gibbs_sites:
            return {}
        recorded = _with_generator(
            generator, lambda: trace(model, x, observations=observations)
        )
        return {name: recorded.sites[name].value.detach() for name in gibbs_sites}


def _gibbs_draw(value: torch.Tensor, morphism: ContinuousMorphism) -> torch.Tensor:
    """One draw of a Gibbs site at its constrained shape.

    The trace-side value of a non-plate site loses its run axis, and a
    length-one event axis is squeezed off, as the registry does for the
    sites in the flat vector, so a Gibbs site's draws have the shape
    `LatentSite.constrained_shape` would give it.

    Parameters
    ----------
    value : torch.Tensor
        The site's trace-side value.
    morphism : ContinuousMorphism
        The site's distribution.

    Returns
    -------
    torch.Tensor
        The draw.
    """
    plate = isinstance(morphism, PlateDraw)
    family = morphism.family if isinstance(morphism, PlateDraw) else morphism
    codomain = family.codomain
    scalar = (
        math.prod(codomain.event_shape) <= 1
        if isinstance(codomain, ContinuousSpace)
        else True
    )
    draw = value if plate else value[0]
    leading = len(morphism.index_shape) if isinstance(morphism, PlateDraw) else 0
    if scalar and draw.dim() > leading and draw.shape[-1] == 1:
        draw = draw.squeeze(-1)
    return draw


def _check_gibbs_sites(
    recorded: Trace, gibbs_sites: frozenset[str]
) -> dict[str, ContinuousMorphism]:
    """Check the kernel's Gibbs sites against a run of the model.

    Parameters
    ----------
    recorded : Trace
        A run of the model at the data.
    gibbs_sites : frozenset[str]
        The sites the kernel updates outside the flat vector.

    Returns
    -------
    dict[str, ContinuousMorphism]
        Each Gibbs site's distribution.

    Raises
    ------
    ValueError
        If a Gibbs site is not a latent sample site of the run, or a
        latent site with a discrete support is not a Gibbs site.
    """
    latent = {
        label: record
        for label, record in recorded.sites.items()
        if record.morphism is not None
        and not record.is_deterministic
        and not record.is_observed
    }
    unknown = sorted(name for name in gibbs_sites if name not in latent)
    if unknown:
        raise ValueError(
            f"MCMC.run: the kernel's Gibbs sites {unknown} are not latent sample "
            f"sites of the run; its latent sites are {sorted(latent)}"
        )
    discrete = []
    for label, record in latent.items():
        morphism = record.morphism
        support = (
            morphism.family.support
            if isinstance(morphism, PlateDraw)
            else getattr(morphism, "support", None)
        )
        if (
            isinstance(support, _constraints.Constraint)
            and support.is_discrete
            and label not in gibbs_sites
        ):
            discrete.append(label)
    if discrete:
        raise ValueError(
            f"MCMC.run: the latent sites {sorted(discrete)} are discrete, and the "
            "kernel moves only a continuous vector; sample them with a GibbsKernel "
            "in a CycleKernel or MixtureKernel, or marginalize them in the model"
        )
    out: dict[str, ContinuousMorphism] = {}
    for name in gibbs_sites:
        morphism = latent[name].morphism
        assert morphism is not None
        out[name] = morphism
    return out


def _component_acceptance(
    per_chain: list[list[dict[str, float]]],
) -> dict[str, torch.Tensor]:
    """Per-component acceptance rates from every draw's statistics.

    Parameters
    ----------
    per_chain : list[list[dict[str, float]]]
        For each chain, each post-warmup transition's statistics.

    Returns
    -------
    dict[str, torch.Tensor]
        Each component's acceptance rate, shape ``(num_chains,)``, keyed
        by its (possibly dotted, for nested composites) name: the mean
        of its ``"<name>.accepted"`` statistic over the transitions that
        reported it, ``nan`` for a chain in which none did.
    """
    keys = sorted(
        {
            key
            for chain in per_chain
            for entry in chain
            for key in entry
            if key.endswith(".accepted")
        }
    )
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        rates = []
        for chain in per_chain:
            values = [entry[key] for entry in chain if key in entry]
            rates.append(sum(values) / len(values) if values else float("nan"))
        out[key.removesuffix(".accepted")] = torch.tensor(rates)
    return out


def _begin_sampling(state: KernelState) -> KernelState:
    """The state a chain's sampling phase starts from.

    The acceptance count restarts so that the reported rate excludes
    warmup, and the divergence flag clears.

    Parameters
    ----------
    state : KernelState
        The last warmup state.

    Returns
    -------
    KernelState
        The same position, cache, and Gibbs values.
    """
    return KernelState(
        position=state.position,
        log_density=state.log_density,
        grad_log_density=state.grad_log_density,
        step_count=state.step_count,
        accept_count=0,
        diverged=False,
        extras=state.extras,
        stats=state.stats,
        gibbs_values=state.gibbs_values,
    )


__all__ = [
    "MCMC",
    "MCMCResult",
    "InitStrategy",
    "StopReason",
    "MCMCPhase",
    "MCMCCallback",
]
