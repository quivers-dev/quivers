"""Mean-field Normal variational guide.

`AutoNormalGuide` assigns each latent site an independent Normal distribution in
unconstrained space and maps each draw through the site's support bijector.
Plate-latent parameters have shape ``(|A|, d_i)`` and do not acquire the
observed plate's leading batch axis.
"""

from __future__ import annotations

from collections.abc import Mapping
import torch
import torch.distributions as D
import torch.nn as nn

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide
from quivers.inference.registry import LatentRegistry, LatentSite


class AutoNormalGuide(Guide):
    """Mean-field Normal guide with per-site constrained-support
    bijector.

    Parameters
    ----------
    model : MonadicProgram
        Generative model to build a guide for.
    observed_names : set[str]
        Variable names treated as observations (skipped in the
        guide; their values flow through the conditioning data
        dict at trace time).
    x : torch.Tensor | None
        Keyword-only. A program input at which to build the latent
        registry from a run, required for a program that scans or calls
        a program that draws, whose latent sites depend on its input.
    observations : dict[str, torch.Tensor] | None
        Keyword-only. The observed values and host data of that run.
    init_scale : float
        Initial scale (in unconstrained space) of every latent.
        Default ``0.1``; small enough to keep the guide near its
        prior at the start of optimisation.
    initial_values : Mapping[str, torch.Tensor or float] or None
        Optional natural-scale initial centers keyed by latent site. Scalars
        broadcast to a site's shape; arrays must match it. Values must be
        strictly inside the site's support. Other centers remain zero in
        unconstrained space.

    Notes
    -----
    The guide factors over a plate's rows, so under a
    [`subsample`][quivers.effects.subsample] block `log_prob` scores a
    latent in a subsampled plate at the subset's rows, times the same
    factor the model's sites take; every row is still drawn, and the
    rows outside the subset receive no gradient from the step.
    """

    scores_subsampled_rows = True

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        init_scale: float = 0.1,
        *,
        x: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
        initial_values: Mapping[str, torch.Tensor | float] | None = None,
    ) -> None:
        super().__init__()
        registry = self.build_registry(
            model, observed_names, x=x, observations=observations
        )
        self._initialize_registry(registry, init_scale, initial_values)

    @classmethod
    def from_registry(
        cls,
        registry: LatentRegistry,
        *,
        init_scale: float = 0.1,
        initial_values: Mapping[str, torch.Tensor | float] | None = None,
    ) -> AutoNormalGuide:
        """Initialize parameters against an already checked latent registry.

        Parameters
        ----------
        registry : LatentRegistry
            Existing model, site supports, and declared event geometry.
        init_scale : float
            Initial unconstrained Normal scale, as in the ordinary constructor.
        initial_values : Mapping[str, torch.Tensor or float] or None
            Optional natural-scale parameter centers.

        Returns
        -------
        AutoNormalGuide
            A fresh guide without tracing or sampling the model.
        """
        guide = cls.__new__(cls)
        nn.Module.__init__(guide)
        guide._initialize_registry(registry, init_scale, initial_values)
        return guide

    def _initialize_registry(
        self,
        registry: LatentRegistry,
        init_scale: float,
        initial_values: Mapping[str, torch.Tensor | float] | None,
    ) -> None:
        """Share parameter initialization between checked-registry constructors.

        Parameters
        ----------
        registry : LatentRegistry
            Registry used by every guide operation.
        init_scale : float
            Initial unconstrained Normal scale.
        initial_values : Mapping[str, torch.Tensor or float] or None
            Optional constrained centers, checked by the shared initializer.
        """
        self._registry = registry
        init_log_scale = float(torch.tensor(init_scale).log().item())

        for site in self._registry.sites.values():
            self.register_parameter(
                f"loc_{site.name}",
                nn.Parameter(torch.zeros(site.unconstrained_shape)),
            )
            self.register_parameter(
                f"log_scale_{site.name}",
                nn.Parameter(torch.full(site.unconstrained_shape, init_log_scale)),
            )
        if initial_values:
            from quivers.inference._initialization import unconstrained_initial_values

            with torch.no_grad():
                for name, value in unconstrained_initial_values(
                    self._registry, initial_values
                ).items():
                    self._loc(name).copy_(value)

    # ------------------------------------------------------------------
    # Variational parameter access
    # ------------------------------------------------------------------

    def _loc(self, name: str) -> torch.Tensor:
        """Return one site's unconstrained Normal center.

        Parameters
        ----------
        name : str
            Checked latent site name.

        Returns
        -------
        torch.Tensor
            Site parameters retaining the guide's autograd connection.
        """
        return getattr(self, f"loc_{name}")

    def _log_scale(self, name: str) -> torch.Tensor:
        """Return one site's unconstrained log scale.

        Parameters
        ----------
        name : str
            Checked latent site name.

        Returns
        -------
        torch.Tensor
            Site parameters retaining the guide's autograd connection.
        """
        return getattr(self, f"log_scale_{name}")

    def _scale(self, name: str) -> torch.Tensor:
        """Return one site's numerically bounded positive Normal scale.

        Parameters
        ----------
        name : str
            Checked latent site name.

        Returns
        -------
        torch.Tensor
            Site parameters retaining the guide's autograd connection.
        """
        return self._log_scale(name).exp().clamp(min=1e-6)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_site(
        self, site: LatentSite, batch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one site's unconstrained and constrained values.

        Parameters
        ----------
        site : LatentSite
            Checked support and declared site geometry.
        batch : int
            Number of execution runs.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Unconstrained Normal sample and its constrained image.
        """
        loc = self._loc(site.name)
        scale = self._scale(site.name)
        if site.is_plate:
            z = D.Normal(loc, scale).rsample()
        else:
            # Broadcast against the particle / batch axis.
            loc_b = loc.unsqueeze(0).expand(batch, *site.unconstrained_shape)
            scale_b = scale.unsqueeze(0).expand(batch, *site.unconstrained_shape)
            z = D.Normal(loc_b, scale_b).rsample()
        v, _ = site.constrain(z)
        return z, v

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Draw a reparameterized mean-field Normal-then-bijector sample.

        Parameters
        ----------
        x : torch.Tensor
            Execution input whose leading axis gives the number of runs.

        Returns
        -------
        dict[str, torch.Tensor]
            Values indexed by checked latent names.
        """
        batch = x.shape[0]
        result: dict[str, torch.Tensor] = {}
        for site in self._registry.sites.values():
            _, v = self._sample_site(site, batch)
            result[site.name] = v
        return result

    # ------------------------------------------------------------------
    # Log-density
    # ------------------------------------------------------------------

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pushforward log-density of each run's constrained values.

        Uses the change-of-variables identity, run by run:

            log q(v_b) = log Normal(z_b; loc, scale) + log|det J_{T^{-1}}(v_b)|

        where ``z_b = bijector.inv(v_b)`` is read from
        `LatentRegistry.unconstrain_runs`. A latent in a plate that a
        [`subsample`][quivers.effects.subsample] block is active for is
        scored at the subset's rows, times the plate's size over the
        subset's.

        Parameters
        ----------
        x : torch.Tensor
            Execution input carrying the leading run axis.
        sites : dict[str, torch.Tensor]
            Constrained latent values to score, each one per run or one
            the runs share (`Guide.log_prob`).

        Returns
        -------
        torch.Tensor
            One variational log density per execution run.

        Raises
        ------
        KeyError
            If a latent site is missing from ``sites``.
        ValueError
            If a value is neither one per run nor one the runs share.
        """
        runs = int(x.shape[0])
        coordinates = self._registry.unconstrain_runs(sites, runs)
        total = torch.zeros(runs, device=x.device)
        for site in self._registry.sites.values():
            z = coordinates.unconstrained[site.name]
            log_q_z = D.Normal(self._loc(site.name), self._scale(site.name)).log_prob(z)
            terms = log_q_z.sum(-1) + coordinates.log_abs_det[site.name]
            if site.is_plate:
                terms = self._plate_run_totals(site, terms)
            total = total + terms
        return total

    @property
    def latent_names(self) -> list[str]:
        """Return checked latent names in their registry order.

        Returns
        -------
        list[str]
            Site labels, including individual scan occurrences.
        """
        return list(self._registry.names)


__all__ = ["AutoNormalGuide"]
