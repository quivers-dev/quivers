"""NeuTra reparameterisation through a trained normalizing-flow guide.

`NeuTraReparam` draws the latent sites a trained flow guide covers as
the guide's pushforward of a standard Normal, after
[Hoffman, Sountsov, Dillon, Langmore, Tran and Vasudevan (2019)](https://doi.org/10.48550/arXiv.1903.03704).
For one run it draws a base point :math:`u \\sim \\mathcal{N}(0, I_D)`
and maps it through the guide's flow :math:`T` and each site's support
bijector :math:`B`, so the run's covered sites take the values
:math:`v = B(T(u))`, whose law is the guide's law :math:`q`. By the
change of variables,

.. math::

    \\log q(v) = \\log \\mathcal{N}(u; 0, I_D)
        - \\log|\\det J_{B \\circ T}(u)|,

which `NeuTraDraw.log_abs_det` records for the draw. Each site is
scored under the model's own density at its value, so the joint density
a run of the reparameterized model reports is the model's.
"""

from __future__ import annotations

import math

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.spaces import ContinuousSpace
from quivers.effects.reparam.base import Reparam, SiteRequest
from quivers.inference.guides.flow import AutoNormalizingFlow


class NeuTraDraw(ValueModel):
    """One run's draw of a flow guide's sites, with its change of variables.

    Parameters
    ----------
    base
        The base point :math:`u`, shape ``(D,)``.
    sites
        Each covered site's constrained value at
        `LatentSite.constrained_shape`.
    log_abs_det
        :math:`\\log|\\det J_{B \\circ T}(u)|` of the map from the base
        point to the sites' values, a scalar: the flow's term plus every
        site bijector's term.
    """

    base: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    sites: dict[str, torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_abs_det: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def log_density(self) -> torch.Tensor:
        """The guide's log density of the draw, by the change of variables.

        Returns
        -------
        torch.Tensor
            :math:`\\log \\mathcal{N}(u; 0, I_D) - \\log|\\det J_{B
            \\circ T}(u)|`, a scalar.
        """
        base = self.base
        log_normal = -0.5 * (base**2).sum() - 0.5 * base.numel() * math.log(2 * math.pi)
        return log_normal - self.log_abs_det


class NeuTraReparam(Reparam):
    """Reparameterise sites through a trained normalizing-flow guide.

    One strategy object serves every site of the guide it is installed
    on, as in ``reparam({"a": neutra, "b": neutra})``: within a run the
    sites share one draw (`NeuTraDraw`), taken at the first covered site
    the run reaches and discarded when the next run begins
    (`Reparam.begin_run`). A site the guide does not cover, or a site
    whose value an enclosing handler fixed, is drawn or taken as given
    and scored under its own density.

    Parameters
    ----------
    guide : AutoNormalizingFlow
        A trained flow guide (`AutoNormalizingFlow`, `AutoIAFGuide`, or
        `AutoNeuralSplineGuide`), whose law is the pushforward of a
        standard Normal.

    Raises
    ------
    TypeError
        If ``guide`` is not a flow guide.
    """

    def __init__(self, guide: AutoNormalizingFlow) -> None:
        if not isinstance(guide, AutoNormalizingFlow):
            raise TypeError(
                f"NeuTraReparam: the guide is a {type(guide).__name__}; NeuTra "
                "needs a flow guide (AutoNormalizingFlow, AutoIAFGuide, or "
                "AutoNeuralSplineGuide), whose law is a pushforward of a "
                "standard Normal"
            )
        self.guide = guide
        self._draw: NeuTraDraw | None = None

    @property
    def current_draw(self) -> NeuTraDraw | None:
        """The draw the current run's covered sites take.

        Returns
        -------
        NeuTraDraw | None
            The draw, or ``None`` before the run reaches a covered site.
        """
        return self._draw

    def begin_run(self) -> None:
        """Discard the previous run's draw."""
        self._draw = None

    def push(self, base: torch.Tensor) -> NeuTraDraw:
        """Map a base point to the guide's sites.

        Parameters
        ----------
        base : torch.Tensor
            The base point :math:`u`, shape ``(D,)``, ``D`` the guide's
            total unconstrained dimension.

        Returns
        -------
        NeuTraDraw
            The sites' values :math:`B(T(u))` and the log Jacobian of the
            map.

        Raises
        ------
        ValueError
            If ``base`` is not of shape ``(D,)``.
        """
        registry = self.guide.registry
        width = registry.total_unconstrained_dim
        if tuple(base.shape) != (width,):
            raise ValueError(
                f"NeuTraReparam.push: the base point has shape {tuple(base.shape)}; "
                f"the guide's flow takes ({width},)"
            )
        z, flow_log_det = self.guide.push_forward(base)
        constrained, log_abs_dets = registry.to_constrained(
            registry.unflatten_unconstrained(z)
        )
        log_det = flow_log_det
        for terms in log_abs_dets.values():
            log_det = log_det + terms.sum()
        return NeuTraDraw(base=base, sites=constrained, log_abs_det=log_det)

    def draw(self) -> NeuTraDraw:
        """Draw a base point from the standard Normal and push it.

        Returns
        -------
        NeuTraDraw
            The draw.
        """
        loc = self.guide.get_buffer("base_loc")
        return self.push(torch.randn_like(loc))

    def apply(self, site: SiteRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Rewrite one site.

        Parameters
        ----------
        site : SiteRequest
            The site.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The site's value, from the run's draw for a covered site
            whose value is not given, shaped as the site's own draws;
            and its log density under the site's sampleable.

        Raises
        ------
        ValueError
            If a covered site's input has more than one row, since one
            run draws it once.
        """
        registry = self.guide.registry
        if site.given is not None:
            value = site.given
        elif site.name not in registry:
            value = site.sampleable.rsample()
        else:
            if self._draw is None:
                self._draw = self.draw()
            value = self._draw.sites[site.name]
            latent = registry[site.name]
            if not latent.is_plate:
                rows = (
                    site.sampleable.input.shape[0] if site.sampleable.input.dim() else 1
                )
                if rows != 1:
                    raise ValueError(
                        f"NeuTraReparam: site {site.name!r} has an input of {rows} "
                        "rows in one run; a run draws the guide's sites once"
                    )
                codomain = site.sampleable.morphism.codomain
                event = (
                    tuple(codomain.event_shape)
                    if isinstance(codomain, ContinuousSpace)
                    else ()
                )
                value = value.reshape(1, *event)
        return value, site.sampleable.log_prob(value)


__all__ = ["NeuTraDraw", "NeuTraReparam"]
