"""Full-rank and low-rank multivariate-Normal variational guides.

Both guides treat the model's latents as a single flat
unconstrained vector :math:`z \\in \\mathbb{R}^{D}` (where ``D``
is the registry's total unconstrained dimensionality) and
parameterize a joint Gaussian over it:

* `AutoMultivariateNormalGuide` uses a full lower-
  triangular Cholesky factor ``L`` so the covariance
  :math:`\\Sigma = L L^\\top` is dense. Parameter count
  :math:`D + D(D+1)/2`; memory scales :math:`O(D^2)`.
* `AutoLowRankMultivariateNormalGuide` uses
  :math:`\\Sigma = W W^\\top + \\mathrm{diag}(\\sigma^2)` with
  ``W`` a rank-``r`` matrix. Parameter count :math:`D + Dr`;
  memory :math:`O(Dr)`. Log-density goes via the Woodbury
  identity and matrix-determinant lemma in
  `torch.distributions.LowRankMultivariateNormal`, so
  evaluation is :math:`O(Dr^2 + r^3)` instead of :math:`O(D^3)`.

After sampling in unconstrained space the per-site bijector
pushes each site to its constrained support, and the Jacobian
correction is added to the variational log-density. Plate sites
are handled by the registry's flatten / unflatten machinery;
the guide itself sees only the flat vector.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide


class _MVNCommon(Guide):
    """Shared behavior for the full-rank and low-rank MVN guides:
    flat-vector sample / log-prob with per-site bijector + Jacobian
    accounting. Subclasses supply the base distribution
    (``_base_dist``)."""

    loc: nn.Parameter

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        *,
        x: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self._registry = self.build_registry(
            model, observed_names, x=x, observations=observations
        )
        D_total = self._registry.total_unconstrained_dim
        if D_total == 0:
            raise ValueError(
                f"{type(self).__name__}: registry has zero total "
                f"unconstrained dimension; model has no continuous "
                f"latents to guide"
            )
        self._D = D_total
        self.loc = nn.Parameter(torch.zeros(D_total))

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _base_dist(self) -> D.Distribution:
        """Return the flat-vector torch.distribution that this guide
        parameterizes. Defined by each subclass."""

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """One reparameterized draw, unflattened and bijected.

        The MVN samples a single ``(D,)``-shaped unconstrained
        vector. Per-site:

        * Plate sites get their natural ``(|A|, d_i)`` shape via
          `LatentRegistry.unflatten_unconstrained` and pass
          through the bijector directly.
        * Scalar sites are expanded to ``(batch, *site.unconstrained_shape)``
          before bijection so the resulting constrained tensor
          matches `AutoNormalGuide`'s shape convention.
        """
        batch = x.shape[0]
        z_flat = self._base_dist().rsample()
        per_site = self._registry.unflatten_unconstrained(z_flat)

        result: dict[str, torch.Tensor] = {}
        for site in self._registry.sites.values():
            z_site = per_site[site.name]
            if not site.is_plate:
                z_site = z_site.unsqueeze(0).expand(batch, *site.unconstrained_shape)
            v = site.bijector(z_site)
            if site.constrained_dim == 1 and v.dim() >= 1 and v.shape[-1] == 1:
                v = v.squeeze(-1)
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
        """Pushforward log-density at the supplied constrained values.

        :math:`\\log q(v) = \\log p_z(z) + \\sum_i \\log|\\det J_{T_i^{-1}}(v_i)|`
        where the first term is the joint MVN log-density and the
        sum is over the per-site bijector inverses.

        Scalar-site values arrive with a leading batch axis; we
        collapse it by taking the first slice (the joint MVN
        emitted one shared draw across the batch in `rsample`,
        so every batch index carries the same value).
        """
        batch = x.shape[0]
        unconstrained_per_site: dict[str, torch.Tensor] = {}
        log_det_total = torch.zeros((), device=x.device)
        for site in self._registry.sites.values():
            if site.name not in sites:
                raise KeyError(
                    f"{type(self).__name__}.log_prob: missing site {site.name!r}"
                )
            v = sites[site.name]
            if site.constrained_dim == 1:
                if site.is_plate:
                    if v.dim() == 1:
                        v_e = v.unsqueeze(-1)
                    else:
                        v_e = v
                else:
                    if v.dim() == 1:
                        v_e = v.unsqueeze(-1)
                    else:
                        v_e = v
            else:
                v_e = v
            if not site.is_plate and v_e.dim() == len(site.unconstrained_shape) + 1:
                v_e = v_e[0]
            z_site = site.bijector.inv(v_e)
            unconstrained_per_site[site.name] = z_site
            log_det_total = log_det_total + (
                site.bijector.inv.log_abs_det_jacobian(v_e, z_site).sum()
            )

        z_flat = self._registry.flatten_unconstrained(unconstrained_per_site)
        log_q_z = self._base_dist().log_prob(z_flat)
        total = (log_q_z + log_det_total).expand(batch)
        return total

    @property
    def latent_names(self) -> list[str]:
        return list(self._registry.names)


class AutoMultivariateNormalGuide(_MVNCommon):
    """Full-rank multivariate-Normal variational guide.

    Parameterises a joint Gaussian over the registry's flat
    unconstrained vector with a learnable lower-triangular Cholesky
    factor. Captures every pairwise posterior correlation across
    every latent site, the right choice when posterior couplings
    are strong (hierarchical regression with crossed random effects,
    parameter pairs with multiplicative interaction).

    Parameters
    ----------
    model : MonadicProgram
        Generative model to build a guide for.
    observed_names : set[str]
        Variable names treated as observations.
    x : torch.Tensor | None
        Keyword-only. A program input at which to build the latent
        registry from a run, required for a program that scans or calls
        a program that draws, whose latent sites depend on its input.
    observations : dict[str, torch.Tensor] | None
        Keyword-only. The observed values and host data of that run.
    init_scale : float
        Initial diagonal of the Cholesky factor. Default ``0.1``;
        the off-diagonal entries start at ``0``.
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        init_scale: float = 0.1,
        *,
        x: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(model, observed_names, x=x, observations=observations)
        init_diag = torch.full((self._D,), float(init_scale))
        init_diag_raw = torch.log(torch.expm1(init_diag.clamp(min=1e-6)))
        self.scale_diag_raw = nn.Parameter(init_diag_raw)
        self.scale_offdiag = nn.Parameter(torch.zeros(self._D, self._D))

    def _scale_tril(self) -> torch.Tensor:
        """Build the Cholesky factor: strictly-lower-triangular
        learned off-diagonal + softplus-positive learned diagonal."""
        off = torch.tril(self.scale_offdiag, diagonal=-1)
        diag = F.softplus(self.scale_diag_raw) + 1e-6
        return off + torch.diag(diag)

    def _base_dist(self) -> D.Distribution:
        return D.MultivariateNormal(self.loc, scale_tril=self._scale_tril())


class AutoLowRankMultivariateNormalGuide(_MVNCommon):
    """Low-rank-plus-diagonal multivariate-Normal guide.

    Covariance :math:`\\Sigma = W W^\\top + \\mathrm{diag}(\\sigma^2)`
    with ``W`` of shape :math:`(D, r)` and :math:`\\sigma \\in
    \\mathbb{R}^{D}_{>0}`. Memory :math:`O(Dr)`; sampling and
    log-density via Woodbury / matrix-determinant lemma in
    `torch.distributions.LowRankMultivariateNormal`.

    Captures the dominant ``r`` posterior correlation directions
    while remaining tractable for ``D`` in the hundreds-to-
    thousands range, where full-rank is infeasible.

    Parameters
    ----------
    model : MonadicProgram
        Generative model to build a guide for.
    observed_names : set[str]
        Variable names treated as observations.
    x : torch.Tensor | None
        Keyword-only. A program input at which to build the latent
        registry from a run, required for a program that scans or calls
        a program that draws, whose latent sites depend on its input.
    observations : dict[str, torch.Tensor] | None
        Keyword-only. The observed values and host data of that run.
    rank : int
        Number of correlated directions. Default ``5``.
    init_scale : float
        Initial diagonal scale. ``W`` is initialized at zero.
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        rank: int = 5,
        init_scale: float = 0.1,
        *,
        x: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(model, observed_names, x=x, observations=observations)
        if rank < 1:
            raise ValueError(
                f"AutoLowRankMultivariateNormalGuide: rank must be >= 1, got {rank}"
            )
        if rank > self._D:
            raise ValueError(
                f"AutoLowRankMultivariateNormalGuide: rank ({rank}) "
                f"cannot exceed total unconstrained dimension "
                f"({self._D})"
            )
        self._rank = rank
        init_diag = torch.full((self._D,), float(init_scale))
        init_diag_raw = torch.log(torch.expm1(init_diag.clamp(min=1e-6)))
        self.cov_diag_raw = nn.Parameter(init_diag_raw)
        self.cov_factor = nn.Parameter(torch.zeros(self._D, rank))

    def _base_dist(self) -> D.Distribution:
        cov_diag = F.softplus(self.cov_diag_raw) + 1e-6
        return D.LowRankMultivariateNormal(
            self.loc,
            cov_factor=self.cov_factor,
            cov_diag=cov_diag,
        )


__all__ = [
    "AutoMultivariateNormalGuide",
    "AutoLowRankMultivariateNormalGuide",
]
