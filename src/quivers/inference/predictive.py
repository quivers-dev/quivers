"""Posterior predictive sampling.

Given a trained posterior representation (either a variational
guide or an MCMC chain), `Predictive` repeatedly samples
latents from the posterior and traces the model forward to produce
posterior predictive draws of every site.
"""

from __future__ import annotations

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.trace import trace


class Predictive:
    """Posterior predictive sampler.

    Accepts either a trained `Guide` (variational
    posterior) or an `MCMCResult` (Monte Carlo posterior).
    Variational case: draws ``num_samples`` fresh guide samples.
    MCMC case: iterates over the recorded posterior draws (one
    forward trace per draw, up to ``num_samples`` if specified).

    An input of ``B`` rows is ``B`` independent runs of the model.
    A guide's `Guide.rsample` gives each run its own draw of a site
    that is not plated, ``(batch, *constrained_shape)``, and one draw
    of a plated site that every run shares. An MCMC draw, at
    `LatentSite.constrained_shape`, is one value every run shares.

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
        If the resolved ``num_samples`` is less than one.
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
        if isinstance(posterior, MCMCResult):
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

    def _iter_mcmc_latents(self) -> list[dict[str, torch.Tensor]]:
        """Return ``num_samples`` recorded posterior draws, one dict each,
        flattening the chain and sample axes.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            One dict per selected draw, each site at its
            `LatentSite.constrained_shape`: one value every run of the
            predictive call shares.
        """
        assert isinstance(self.posterior, MCMCResult)
        samples = self.posterior.samples
        chain_samples = self.posterior.num_chains * self.posterior.num_samples
        # Random selection without replacement (when num_samples
        # equals the total, this is a permutation).
        perm = torch.randperm(chain_samples)[: self.num_samples]
        out: list[dict[str, torch.Tensor]] = []
        for idx in perm.tolist():
            chain = idx // self.posterior.num_samples
            step = idx % self.posterior.num_samples
            out.append({name: draws[chain, step] for name, draws in samples.items()})
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
            One key per site, holding the stacked trace values:
            ``(num_samples, batch, *shape)`` for a site that is not
            plated, and for a plated site ``(num_samples, *shape)`` for
            one run and ``(num_samples, batch, *shape)`` for ``batch >
            1`` runs. Guide and MCMC posteriors produce the same shapes.

        Raises
        ------
        ValueError
            If an `MCMCResult` holds draws of a name that is not a
            sample site of the model.
        """
        observations = observations if observations is not None else {}
        if isinstance(self.posterior, Guide):
            latents_iter: list[dict[str, torch.Tensor]] = [
                self.posterior.rsample(x) for _ in range(self.num_samples)
            ]
        else:
            latents_iter = self._iter_mcmc_latents()

        collected: dict[str, list[torch.Tensor]] = {}
        for latents in latents_iter:
            all_obs = {**latents, **observations}
            tr = trace(self.model, x, observations=all_obs)
            stray = sorted(set(latents) - set(tr.sites))
            if stray:
                raise ValueError(
                    f"Predictive: the posterior holds draws of {stray}, which "
                    f"are not sample sites of the model"
                )
            for name, site in tr.sites.items():
                if site.is_deterministic:
                    continue
                collected.setdefault(name, []).append(site.value)

        return {name: torch.stack(vals, dim=0) for name, vals in collected.items()}


__all__ = ["Predictive"]
