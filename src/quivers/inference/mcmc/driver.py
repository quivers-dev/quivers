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

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.diagnostics.convergence import ess, rhat
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
    _check_observed_support,
)
from quivers.inference.registry import LatentRegistry, _require_one_run
from quivers.inference.trace import trace


InitStrategy = Literal["prior", "zero", "guide"]
"""How `MCMC` chooses each chain's initial point in the unconstrained
latent space: ``"prior"`` draws a small seeded standard-normal
perturbation of the origin, ``"zero"`` starts at the origin, and
``"guide"`` maps one draw of a fitted guide into the unconstrained
space."""


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
        Per-site posterior draws on the constrained support.
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
        ``(num_chains,)``.
    divergence_counts : torch.Tensor
        Per-chain post-warmup divergence count. Shape
        ``(num_chains,)``.
    num_warmup : int
        Warmup iterations per chain.
    num_samples : int
        Retained draws per chain.
    """

    samples: dict[str, torch.Tensor]
    log_densities: torch.Tensor
    acceptance_rates: torch.Tensor
    divergence_counts: torch.Tensor
    num_warmup: int
    num_samples: int

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


class MCMC:
    """MCMC chain runner.

    Parameters
    ----------
    kernel : MCMCKernel
        Markov kernel (e.g. `HMCKernel`, `NUTSKernel`).
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
        self.kernel = kernel
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.init_strategy = init_strategy
        self.seed = seed
        self.generator = generator

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
        Exception
            Any error the program raises at an initial position, or
            raises during sampling for a reason other than a position
            outside its support.
        """
        _require_one_run(x, "MCMC.run")
        _check_observed_support(model, observations, "MCMC.run")
        # The latents are the sites one run at (x, observations) reaches,
        # every position of a scan included. The run draws its latents
        # from a forked stream, so Torch's global RNG is left as it was.
        try:
            with torch.random.fork_rng(devices=[]):
                recorded = trace(model, x, observations=observations)
        except Exception as error:
            error.add_note(
                "MCMC.run: raised while recording the run its latents are read "
                "from, before any sampling"
            )
            raise
        registry = LatentRegistry.from_trace(model, recorded)
        potential = PotentialFn(model, registry, x, observations)

        site_names = registry.names
        per_chain_samples: dict[str, list[torch.Tensor]] = {n: [] for n in site_names}
        per_chain_log_density = torch.empty(self.num_chains, self.num_samples)
        per_chain_accept = torch.empty(self.num_chains)
        per_chain_divergences = torch.empty(self.num_chains, dtype=torch.int64)
        run_seed = self._run_seed()

        for chain in range(self.num_chains):
            chain_generator = self._chain_generator(run_seed, chain)
            init_pos = self._initial_position(registry, guide, chain_generator)
            if not torch.isfinite(init_pos).all():
                raise ValueError(
                    f"MCMC.run: chain {chain} initial_position must contain "
                    "only finite values"
                )
            try:
                potential.check_initial(init_pos)
            except Exception as error:
                error.add_note(
                    f"MCMC.run: raised while scoring chain {chain}'s initial "
                    "position, before any sampling"
                )
                raise
            self.kernel._set_generator(chain_generator)
            state = self.kernel.init(registry, model, x, observations, init_pos)
            # Warmup.
            if self.num_warmup > 0:
                self.kernel.start_adaptation(self.num_warmup)
                for _ in range(self.num_warmup):
                    state = self.kernel.step(state, potential)
                self.kernel.stop_adaptation()
            # Reset accept-count so the reported rate excludes warmup.
            sampling_state = KernelState(
                position=state.position,
                log_density=state.log_density,
                grad_log_density=state.grad_log_density,
                step_count=state.step_count,
                accept_count=0,
                diverged=False,
                extras=state.extras,
            )
            chain_samples: dict[str, list[torch.Tensor]] = {n: [] for n in site_names}
            sampling_divergences = 0
            for s in range(self.num_samples):
                sampling_state = self.kernel.step(sampling_state, potential)
                if sampling_state.diverged:
                    sampling_divergences += 1
                draws = self._to_constrained(registry, sampling_state.position)
                for n, v in draws.items():
                    chain_samples[n].append(v)
                per_chain_log_density[chain, s] = sampling_state.log_density
            per_chain_accept[chain] = sampling_state.accept_count / float(
                self.num_samples
            )
            per_chain_divergences[chain] = sampling_divergences
            for n, draws_list in chain_samples.items():
                stacked = torch.stack(draws_list, dim=0)
                per_chain_samples[n].append(stacked)

        # Stack per-chain → (num_chains, num_samples, *site_shape).
        samples: dict[str, torch.Tensor] = {}
        for n, chain_draws in per_chain_samples.items():
            samples[n] = torch.stack(chain_draws, dim=0)

        return MCMCResult(
            samples=samples,
            log_densities=per_chain_log_density,
            acceptance_rates=per_chain_accept,
            divergence_counts=per_chain_divergences,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
        )


__all__ = ["MCMC", "MCMCResult", "InitStrategy"]
