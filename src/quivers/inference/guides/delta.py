"""Point-estimate (MAP / Dirac) variational guide.

`AutoDeltaGuide` is the degenerate case of a variational
guide: a Dirac delta at a single learnable point in unconstrained
space, pushed through the support bijector so the constrained
estimate lies inside the prior's support. Used for MAP estimation
and as the warmup mean for `AutoLaplaceApproximation`.

Log-density returns zero: the delta-mass contribution and its
Jacobian cancel in the ELBO under the standard score-function
trick.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide
from quivers.inference.registry import LatentSite


class AutoDeltaGuide(Guide):
    """Dirac-delta MAP guide with per-site constrained bijector.

    Parameters
    ----------
    model : MonadicProgram
        Generative model.
    observed_names : set[str]
        Variable names treated as observations.
    x : torch.Tensor | None
        Keyword-only. A program input at which to build the latent
        registry from a run, required for a program that scans or calls
        a program that draws, whose latent sites depend on its input.
    observations : dict[str, torch.Tensor] | None
        Keyword-only. The observed values and host data of that run.
    init_value : float
        Initial unconstrained-space coordinate for every latent.
        Default ``0.0``; for the standard bijectors this maps to
        a sensible interior point of each support (the median of
        a HalfNormal, the centre of the unit interval, the uniform
        Dirichlet, etc.). Small Gaussian noise is added so two
        coordinate values don't collide.

    Notes
    -----
    The guide's density is constant, so it scores a latent in a
    [`subsample`][quivers.effects.subsample]d plate as it scores every
    other latent.
    """

    scores_subsampled_rows = True

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        init_value: float = 0.0,
        *,
        x: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self._registry = self.build_registry(
            model, observed_names, x=x, observations=observations
        )

        for site in self._registry.sites.values():
            self.register_parameter(
                f"unconstrained_{site.name}",
                nn.Parameter(
                    torch.full(site.unconstrained_shape, init_value)
                    + torch.randn(site.unconstrained_shape) * 0.01
                ),
            )

    def _site_unconstrained(self, name: str) -> torch.Tensor:
        return getattr(self, f"unconstrained_{name}")

    def _push_through_bijector(self, site: LatentSite, batch: int) -> torch.Tensor:
        z = self._site_unconstrained(site.name)
        if not site.is_plate:
            z = z.unsqueeze(0).expand(batch, *site.unconstrained_shape)
        v, _ = site.constrain(z)
        return v

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the learned point estimates in the prior's
        support."""
        batch = x.shape[0]
        return {
            site.name: self._push_through_bijector(site, batch)
            for site in self._registry.sites.values()
        }

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Delta log-density of each run: zero (the delta term and its
        Jacobian cancel in the ELBO under the standard score-function
        trick).

        The values are still checked against the run-wise contract of
        `Guide.log_prob`.

        Parameters
        ----------
        x : torch.Tensor
            Program input; its leading axis has one row per run.
        sites : dict[str, torch.Tensor]
            Constrained values of every latent site, each one per run or
            one the runs share.

        Returns
        -------
        torch.Tensor
            Zeros of shape ``(B,)``.

        Raises
        ------
        KeyError
            If a latent site is missing from ``sites``.
        ValueError
            If a value is neither one per run nor one the runs share.
        """
        runs = int(x.shape[0])
        for site in self._registry.sites.values():
            if site.name not in sites:
                raise KeyError(f"AutoDeltaGuide.log_prob: missing site {site.name!r}")
            site.align_runs(sites[site.name], runs)
        return torch.zeros(runs, device=x.device)

    @property
    def latent_names(self) -> list[str]:
        return list(self._registry.names)


__all__ = ["AutoDeltaGuide"]
