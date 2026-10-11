"""MCMC orchestrator: chains, warmup, and posterior diagnostics.

The `MCMC` driver wraps a `MCMCKernel` and applies
it to a model: it builds the registry, draws an initial position
(from the prior, an existing guide, or a user-supplied dict),
runs ``num_warmup`` adaptation steps with
``kernel.start_adaptation()``, freezes the kernel via
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
sit in the model's natural support) plus rank-normalized folded
split-:math:`\\hat R` (Vehtari et al. 2021,
`doi:10.1214/20-BA1221 <https://doi.org/10.1214/20-BA1221>`_) and
bulk effective sample size diagnostics computed across chains.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
)
from quivers.inference.registry import LatentRegistry


InitStrategy = Literal["prior", "zero", "guide"]
"""How `MCMC` chooses each chain's initial point in the unconstrained
latent space: ``"prior"`` draws a small seeded standard-normal
perturbation of the origin, ``"zero"`` starts at the origin, and
``"guide"`` maps one draw of a fitted guide into the unconstrained
space."""


@dataclass
class MCMCResult:
    """Posterior samples and per-chain diagnostics.

    Attributes
    ----------
    samples : dict[str, torch.Tensor]
        Per-site posterior draws on the constrained support.
        Shape ``(num_chains, num_samples, *site_shape)``.
    log_densities : torch.Tensor
        Unconstrained-space log-density (Jacobian-corrected) at
        every posterior draw. Shape ``(num_chains, num_samples)``.
    acceptance_rates : torch.Tensor
        Per-chain post-warmup acceptance rate. Shape
        ``(num_chains,)``.
    divergence_counts : torch.Tensor
        Per-chain post-warmup divergence count. Shape
        ``(num_chains,)``.
    r_hat : dict[str, torch.Tensor]
        Per-site rank-normalized folded split-:math:`\\hat R`.
        Each coordinate reports the conservative maximum of the
        rank-normalized and folded statistics. Each site's tensor
        has the site's shape (one scalar per coordinate).
    ess : dict[str, torch.Tensor]
        Per-site rank-normalized bulk effective sample size pooled
        across chains. Same shape convention as `r_hat`.
    num_warmup : int
    num_samples : int
    """

    samples: dict[str, torch.Tensor]
    log_densities: torch.Tensor
    acceptance_rates: torch.Tensor
    divergence_counts: torch.Tensor
    r_hat: dict[str, torch.Tensor]
    ess: dict[str, torch.Tensor]
    num_warmup: int
    num_samples: int

    @property
    def num_chains(self) -> int:
        return int(self.log_densities.shape[0])

    @property
    def mean_acceptance(self) -> float:
        return float(self.acceptance_rates.mean().item())

    @property
    def total_divergences(self) -> int:
        return int(self.divergence_counts.sum().item())


def _nan_scalar(chains: torch.Tensor) -> torch.Tensor:
    """Return a floating NaN on the input tensor's device."""
    return torch.tensor(float("nan"), dtype=torch.float64, device=chains.device)


def _split_chains(chains: torch.Tensor) -> torch.Tensor:
    """Split each chain in half, dropping the middle draw when odd."""
    half = chains.shape[1] // 2
    return torch.cat([chains[:, :half], chains[:, -half:]], dim=0)


def _rank_normalize(chains: torch.Tensor) -> torch.Tensor:
    """Apply average ranks and Blom's normal-score transform."""
    flat = chains.reshape(-1).to(dtype=torch.float64)
    order = torch.argsort(flat, stable=True)
    sorted_values = flat[order]
    ranks = torch.empty_like(flat)

    starts = (
        torch.cat(
            [
                torch.ones(1, dtype=torch.bool, device=flat.device),
                sorted_values[1:] != sorted_values[:-1],
            ]
        )
        .nonzero(as_tuple=False)
        .flatten()
    )
    ends = torch.cat([starts[1:], torch.tensor([flat.numel()], device=flat.device)])
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        # Average of the one-indexed ranks from start + 1 through end.
        ranks[order[start:end]] = (start + 1 + end) / 2

    probabilities = (ranks - 3.0 / 8.0) / (flat.numel() + 1.0 / 4.0)
    normal_scores = math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)
    return normal_scores.reshape(chains.shape)


def _rhat(chains: torch.Tensor) -> torch.Tensor:
    """Potential scale reduction factor for already transformed chains."""
    n_samples = chains.shape[1]
    chain_means = chains.mean(dim=1)
    between = n_samples * chain_means.var(unbiased=True)
    within = chains.var(dim=1, unbiased=True).mean()
    return torch.sqrt((between / within + n_samples - 1) / n_samples)


def _autocov(chains: torch.Tensor) -> torch.Tensor:
    """Biased per-chain autocovariance estimates for every lag."""
    n_samples = chains.shape[1]
    centered = chains - chains.mean(dim=1, keepdim=True)
    pad = 1
    while pad < 2 * n_samples:
        pad *= 2
    spectrum = torch.fft.rfft(centered, n=pad, dim=1)
    covariance = torch.fft.irfft(spectrum * torch.conj(spectrum), n=pad, dim=1)[
        :, :n_samples
    ]
    return covariance / n_samples


def _ess(chains: torch.Tensor) -> torch.Tensor:
    """Cross-chain ESS with Geyer's initial monotone sequence."""
    if (chains.max() - chains.min()).item() < torch.finfo(torch.float64).resolution:
        return torch.tensor(
            float(chains.numel()), dtype=torch.float64, device=chains.device
        )

    n_chains, n_samples = chains.shape
    autocovariance = _autocov(chains)
    chain_means = chains.mean(dim=1)
    mean_variance = autocovariance[:, 0].mean() * n_samples / (n_samples - 1.0)
    pooled_variance = mean_variance * (n_samples - 1.0) / n_samples
    if n_chains > 1:
        pooled_variance += chain_means.var(unbiased=True)

    rho = torch.zeros(n_samples, dtype=torch.float64, device=chains.device)
    rho_even = torch.tensor(1.0, dtype=torch.float64, device=chains.device)
    rho[0] = rho_even
    rho_odd = 1.0 - (mean_variance - autocovariance[:, 1].mean()) / pooled_variance
    rho[1] = rho_odd

    lag = 1
    while lag < n_samples - 3 and (rho_even + rho_odd).item() > 0.0:
        rho_even = (
            1.0 - (mean_variance - autocovariance[:, lag + 1].mean()) / pooled_variance
        )
        rho_odd = (
            1.0 - (mean_variance - autocovariance[:, lag + 2].mean()) / pooled_variance
        )
        if (rho_even + rho_odd).item() >= 0.0:
            rho[lag + 1] = rho_even
            rho[lag + 2] = rho_odd
        lag += 2

    max_lag = lag - 2
    if rho_even.item() > 0.0:
        rho[max_lag + 1] = rho_even

    lag = 1
    while lag <= max_lag - 2:
        previous_pair = rho[lag - 1] + rho[lag]
        next_pair = rho[lag + 1] + rho[lag + 2]
        if next_pair.item() > previous_pair.item():
            rho[lag + 1] = previous_pair / 2.0
            rho[lag + 2] = previous_pair / 2.0
        lag += 2

    total = n_chains * n_samples
    tau = -1.0 + 2.0 * rho[: max_lag + 1].sum() + rho[max_lag + 1 : max_lag + 2].sum()
    tau = torch.maximum(
        tau,
        torch.tensor(
            1.0 / math.log10(total), dtype=torch.float64, device=chains.device
        ),
    )
    if torch.isnan(rho).any().item():
        return _nan_scalar(chains)
    return total / tau


def _split_rhat_1d(chains: torch.Tensor) -> torch.Tensor:
    """Rank-normalized folded split R-hat for one coordinate.

    The reported value is the conservative maximum of bulk
    rank-normalized split R-hat and the folded statistic, which is
    sensitive to scale and tail disagreement.
    """
    if chains.shape[0] < 2 or chains.shape[1] < 4 or torch.isnan(chains).any().item():
        return _nan_scalar(chains)

    split = _split_chains(chains)
    bulk = _rhat(_rank_normalize(split))
    folded = torch.abs(split - torch.quantile(split, 0.5))
    tail = _rhat(_rank_normalize(folded))
    return torch.maximum(bulk, tail)


def _ess_1d(chains: torch.Tensor) -> torch.Tensor:
    """Rank-normalized bulk ESS for one coordinate across chains."""
    if chains.shape[1] < 4 or torch.isnan(chains).any().item():
        return _nan_scalar(chains)
    return _ess(_rank_normalize(_split_chains(chains)))


def _diagnostics(
    samples: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute per-site :math:`\\hat R` and ESS over the
    ``(num_chains, num_samples, *site_shape)`` posterior tensors."""
    r_hat: dict[str, torch.Tensor] = {}
    ess: dict[str, torch.Tensor] = {}
    for name, draws in samples.items():
        n_chains = draws.shape[0]
        n_samples = draws.shape[1]
        flat = draws.reshape(n_chains, n_samples, -1)
        n_coords = flat.shape[-1]
        rh = torch.empty(n_coords)
        es = torch.empty(n_coords)
        for c in range(n_coords):
            rh[c] = _split_rhat_1d(flat[:, :, c])
            es[c] = _ess_1d(flat[:, :, c])
        site_shape = tuple(draws.shape[2:]) or (1,)
        r_hat[name] = rh.reshape(site_shape)
        ess[name] = es.reshape(site_shape)
    return r_hat, ess


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
            # Sample the guide once, pull out the unconstrained
            # values, flatten.
            x = torch.zeros(1, 1)
            constrained = self._guide_rsample(guide, x, generator)
            unc: dict[str, torch.Tensor] = {}
            for site in registry.sites.values():
                v = constrained[site.name]
                if site.constrained_dim == 1 and v.dim() == (1 if site.is_plate else 1):
                    v_e = v.unsqueeze(-1)
                else:
                    v_e = v
                if not site.is_plate and v_e.dim() == len(site.unconstrained_shape) + 1:
                    v_e = v_e[0]
                z = site.bijector.inv(v_e)
                unc[site.name] = z
            return registry.flatten_unconstrained(unc)
        # "prior": draw a fresh standard-normal position, scaled
        # mildly so chains spread out without diverging.
        return 0.1 * torch.randn(D, generator=generator)

    def _to_constrained(
        self,
        registry: LatentRegistry,
        z_flat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        per_site_unc = registry.unflatten_unconstrained(z_flat)
        out: dict[str, torch.Tensor] = {}
        for site in registry.sites.values():
            z_site = per_site_unc[site.name]
            if not site.is_plate:
                z_site = z_site.unsqueeze(0)
            v = site.bijector(z_site)
            if site.constrained_dim == 1 and v.dim() >= 1 and v.shape[-1] == 1:
                v = v.squeeze(-1)
            out[site.name] = v
        return out

    def run(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        guide: Guide | None = None,
    ) -> MCMCResult:
        """Run the configured kernel for ``num_chains`` chains of
        ``num_warmup + num_samples`` steps each."""
        observed_names = set(observations.keys())
        registry = LatentRegistry.from_model(model, observed_names)
        potential = PotentialFn(model, registry, x, observations)

        site_shapes: dict[str, tuple[int, ...]] = {
            site.name: site.constrained_shape or (1,)
            for site in registry.sites.values()
        }
        per_chain_samples: dict[str, list[torch.Tensor]] = {n: [] for n in site_shapes}
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
            self.kernel._set_generator(chain_generator)
            state = self.kernel.init(registry, model, x, observations, init_pos)
            # Warmup.
            if self.num_warmup > 0:
                self.kernel.start_adaptation()
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
            chain_samples: dict[str, list[torch.Tensor]] = {n: [] for n in site_shapes}
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

        r_hat, ess = _diagnostics(samples)
        return MCMCResult(
            samples=samples,
            log_densities=per_chain_log_density,
            acceptance_rates=per_chain_accept,
            divergence_counts=per_chain_divergences,
            r_hat=r_hat,
            ess=ess,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
        )


__all__ = ["MCMC", "MCMCResult", "InitStrategy"]
