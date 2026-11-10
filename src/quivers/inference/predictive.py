"""Posterior predictive sampling.

Given a trained posterior representation (either a variational
guide or an MCMC chain), `Predictive` repeatedly samples
latents from the posterior and traces the model forward to produce
posterior predictive draws of every site.
"""

from __future__ import annotations

import torch

from quivers.continuous.program_steps import Draw, Observe
from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import trace


class Predictive:
    """Posterior predictive sampler.

    Accepts either a trained `Guide` (variational
    posterior) or an `MCMCResult` (Monte Carlo posterior).
    Variational case: draws ``num_samples`` fresh guide samples.
    MCMC case: iterates over the recorded posterior draws (one
    forward trace per draw, up to ``num_samples`` if specified).

    Both cases trace the model with trace-side latent values. A
    guide's `Guide.rsample` already returns them: non-plate sites at
    ``(batch, *constrained_shape)`` and plate sites at
    ``constrained_shape``. An MCMC draw has shape
    `LatentSite.constrained_shape`, so each draw is lifted by
    `LatentRegistry.trace_values`, which expands non-plate sites
    across the input's batch axis; every input row then sees the
    same posterior draw.

    Parameters
    ----------
    model : MonadicProgram
        Generative model.
    posterior : Guide or MCMCResult
        Trained posterior representation.
    num_samples : int, optional
        Number of predictive draws. Defaults to ``100`` for guides
        and ``num_chains * num_samples`` for MCMC results (use all
        recorded posterior draws). If supplied as an explicit
        integer for an MCMC result it is capped at the available
        draw count.

    Raises
    ------
    TypeError
        If ``posterior`` is neither a `Guide` nor an `MCMCResult`.
    ValueError
        If the resolved ``num_samples`` is less than one, or an
        `MCMCResult` holds samples for a site that is not a latent
        sample site of ``model``.
    """

    def __init__(
        self,
        model: MonadicProgram,
        posterior: Guide | MCMCResult,
        num_samples: int | None = None,
    ) -> None:
        if not isinstance(posterior, (Guide, MCMCResult)):
            raise TypeError(
                f"Predictive: posterior must be Guide or MCMCResult; "
                f"got {type(posterior).__name__}"
            )
        self.model = model
        self.posterior = posterior
        self._mcmc_registry: LatentRegistry | None = None
        if isinstance(posterior, MCMCResult):
            sampled = set(posterior.samples)
            not_sampled = {
                name
                for step in model.steps
                if isinstance(step, Draw | Observe)
                for name in step.names
                if name not in sampled
            }
            self._mcmc_registry = LatentRegistry.from_model(model, not_sampled)
            missing = sampled - set(self._mcmc_registry.names)
            if missing:
                raise ValueError(
                    f"Predictive: MCMCResult samples sites {sorted(missing)} "
                    f"that are not latent sample sites of the model"
                )
            available = posterior.num_chains * posterior.num_samples
            self.num_samples = (
                min(num_samples, available) if num_samples is not None else available
            )
        else:
            self.num_samples = num_samples if num_samples is not None else 100
        if self.num_samples < 1:
            raise ValueError(
                f"Predictive: num_samples must be >= 1, got {self.num_samples}"
            )

    def _iter_mcmc_latents(self, batch_size: int) -> list[dict[str, torch.Tensor]]:
        """Return ``num_samples`` per-draw trace-side latent dicts
        from the MCMC result, flattening the chain and sample axes.

        Parameters
        ----------
        batch_size : int
            Size of the program input's leading batch axis; each
            non-plate draw is expanded across it.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            One dict per selected draw, shaped by
            `LatentRegistry.trace_values`.
        """
        assert isinstance(self.posterior, MCMCResult)
        assert self._mcmc_registry is not None
        samples = self.posterior.samples
        chain_samples = self.posterior.num_chains * self.posterior.num_samples
        # Random selection without replacement (when num_samples
        # equals the total, this is a permutation).
        perm = torch.randperm(chain_samples)[: self.num_samples]
        out: list[dict[str, torch.Tensor]] = []
        for idx in perm.tolist():
            chain = idx // self.posterior.num_samples
            step = idx % self.posterior.num_samples
            draw = {name: draws[chain, step] for name, draws in samples.items()}
            out.append(self._mcmc_registry.trace_values(draw, batch_size))
        return out

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Draw posterior predictive samples.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``.
        observations : dict[str, torch.Tensor] or None
            Additional observed data to condition on.

        Returns
        -------
        dict[str, torch.Tensor]
            One key per site, holding the stacked trace-side values:
            ``(num_samples, batch, *constrained_shape)`` for a
            non-plate latent site, ``(num_samples, *constrained_shape)``
            for a plate latent site, and ``num_samples`` stacked
            trace values for every other stochastic site. Guide and
            MCMC posteriors produce the same shapes.
        """
        observations = observations if observations is not None else {}
        if isinstance(self.posterior, Guide):
            latents_iter: list[dict[str, torch.Tensor]] = [
                self.posterior.rsample(x) for _ in range(self.num_samples)
            ]
        else:
            latents_iter = self._iter_mcmc_latents(int(x.shape[0]))

        collected: dict[str, list[torch.Tensor]] = {}
        for latents in latents_iter:
            all_obs = {**latents, **observations}
            tr = trace(self.model, x, observations=all_obs)
            for name, site in tr.sites.items():
                if site.is_deterministic:
                    continue
                collected.setdefault(name, []).append(site.value)

        return {name: torch.stack(vals, dim=0) for name, vals in collected.items()}


__all__ = ["Predictive"]
