"""Laplace approximation guide.

A Laplace approximation uses a quadratic expansion of the log-joint around the
maximum a posteriori (MAP) point in unconstrained space:

.. math::

    \\log p(z, y) \\approx \\log p(z^\\star, y)
        + \\tfrac{1}{2} (z - z^\\star)^\\top H (z - z^\\star),

where :math:`z^\\star` is the MAP and ``H`` is the Hessian of the
log-joint evaluated at :math:`z^\\star`. The Laplace posterior is
then :math:`\\mathcal{N}(z^\\star, -H^{-1})`, provided the negative
Hessian is positive definite. The local curvature supplies the Gaussian
covariance: sharper curvature corresponds to less uncertainty in that
direction [MacKay (2003)](<https://assets.cambridge.org/97805216/42989/frontmatter/9780521642989_frontmatter.pdf>); [Tierney and Kadane (1986)](<https://www.tandfonline.com/doi/abs/10.1080/01621459.1986.10478240>). This local
approximation can miss other modes and skewed posterior tails.

Usage pattern
=============

`AutoLaplaceApproximation` is a two-phase guide:

1. **MAP phase.** Until `fit_hessian` is called the guide
   behaves like `AutoDeltaGuide`: variational parameters
   are a single unconstrained-space point estimate, and SVI with
   this guide does MAP optimization.
2. **Hessian phase.** Once `fit_hessian` is invoked the
   Hessian of ``-log p(z, y)`` at the current MAP is computed
   via `torch.autograd.functional.hessian`, inverted via
   Cholesky decomposition with a small jitter, and the guide
   thereafter samples from
   :math:`\\mathcal{N}(z^\\star, H^{-1})` and reports the
   matching Gaussian log-density.

The MAP-phase log-density is zero (matching the delta-guide
convention; the delta mass and its Jacobian cancel in the ELBO
under the standard score-function trick). Once the Hessian is
fit the log-density is the pushforward Gaussian, with the per-
site bijector Jacobian correction folded in.
"""

from __future__ import annotations

import torch
import torch.distributions as D
import torch.nn as nn

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide


class AutoLaplaceApproximation(Guide):
    """Laplace-approximation guide.

    Parameters
    ----------
    model : MonadicProgram
        Generative model.
    observed_names : set[str]
        Variable names treated as observations.
    init_value : float
        Initial unconstrained-space MAP estimate. Default ``0.0``.
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        init_value: float = 0.0,
    ) -> None:
        super().__init__()
        self._registry = self.build_registry(model, observed_names)
        D_total = self._registry.total_unconstrained_dim
        if D_total == 0:
            raise ValueError(
                f"{type(self).__name__}: registry has zero total "
                f"unconstrained dimension; model has no continuous "
                f"latents to guide"
            )
        self._D = D_total
        # MAP estimate in flat unconstrained space.
        init = torch.full((D_total,), float(init_value))
        init = init + 0.01 * torch.randn(D_total)
        self.map_z = nn.Parameter(init)
        # Hessian-phase parameters; initialized to identity scale_tril
        # but only used after fit_hessian() is called.
        self.register_buffer("_hessian_fitted", torch.zeros((), dtype=torch.bool))
        self.register_buffer(
            "_scale_tril",
            torch.eye(D_total) * 1e-3,
            persistent=True,
        )

    # ------------------------------------------------------------------
    # Hessian fitting
    # ------------------------------------------------------------------

    def fit_hessian(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        *,
        jitter: float = 1e-4,
    ) -> None:
        """Compute and cache the Hessian-derived Cholesky factor.

        Solves the eigenproblem of the negative-log-joint Hessian at
        the current MAP, projects negative eigenvalues to ``jitter``
        (so the resulting Gaussian is always positive-definite), and
        stores the matching lower-triangular Cholesky factor of the
        inverse Hessian as the posterior scale_tril.

        Call this after MAP optimization has converged. Subsequent
        `rsample` / `log_prob` calls sample from
        :math:`\\mathcal{N}(z^\\star, H^{-1})`.
        """

        def neg_log_joint(z_flat: torch.Tensor) -> torch.Tensor:
            per_site_unconstrained = self._registry.unflatten_unconstrained(z_flat)
            constrained: dict[str, torch.Tensor] = {}
            log_det_sum = torch.zeros((), device=z_flat.device)
            for site in self._registry.sites.values():
                z_site = per_site_unconstrained[site.name]
                if not site.is_plate:
                    z_site = z_site.unsqueeze(0)
                v = site.bijector(z_site)
                log_det_sum = log_det_sum + (
                    site.bijector.log_abs_det_jacobian(z_site, v).sum()
                )
                if site.constrained_dim == 1 and v.dim() >= 1 and v.shape[-1] == 1:
                    v = v.squeeze(-1)
                constrained[site.name] = v
            log_p = model.log_joint(x, {**constrained, **observations})
            return -(log_p.sum() + log_det_sum)

        H = torch.autograd.functional.hessian(neg_log_joint, self.map_z.detach())
        H = 0.5 * (H + H.t())
        eigvals, eigvecs = torch.linalg.eigh(H)
        eigvals_clamped = eigvals.clamp(min=jitter)
        # Σ = (V Λ V^T)^{-1} = V Λ^{-1} V^T; scale_tril is its Cholesky.
        inv_eigvals = 1.0 / eigvals_clamped
        sigma = (eigvecs * inv_eigvals.unsqueeze(0)) @ eigvecs.t()
        sigma = 0.5 * (sigma + sigma.t())
        sigma = sigma + jitter * torch.eye(self._D, device=sigma.device)
        L = torch.linalg.cholesky(sigma)
        self._scale_tril.copy_(L)
        self._hessian_fitted.fill_(True)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_unconstrained(self) -> torch.Tensor:
        if not bool(self._hessian_fitted):
            # MAP phase: return the point estimate.
            return self.map_z
        return D.MultivariateNormal(self.map_z, scale_tril=self._scale_tril).rsample()

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample from the Laplace posterior, unflatten, and biject."""
        batch = x.shape[0]
        z_flat = self._sample_unconstrained()
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
        """Log-density at the supplied constrained sites.

        Returns zero before `fit_hessian` (MAP-phase delta
        convention); after `fit_hessian` returns the Gaussian
        log-density plus the per-site bijector Jacobian correction.
        """
        batch = x.shape[0]
        if not bool(self._hessian_fitted):
            return torch.zeros(batch, device=x.device)

        unconstrained_per_site: dict[str, torch.Tensor] = {}
        bijector_log_det = torch.zeros((), device=x.device)
        for site in self._registry.sites.values():
            if site.name not in sites:
                raise KeyError(
                    f"{type(self).__name__}.log_prob: missing site {site.name!r}"
                )
            v = sites[site.name]
            if site.constrained_dim == 1 and v.dim() == (1 if site.is_plate else 1):
                v_e = v.unsqueeze(-1)
            else:
                v_e = v
            if not site.is_plate and v_e.dim() == len(site.unconstrained_shape) + 1:
                v_e = v_e[0]
            z_site = site.bijector.inv(v_e)
            unconstrained_per_site[site.name] = z_site
            bijector_log_det = bijector_log_det + (
                site.bijector.inv.log_abs_det_jacobian(v_e, z_site).sum()
            )
        z_flat = self._registry.flatten_unconstrained(unconstrained_per_site)
        gauss = D.MultivariateNormal(self.map_z, scale_tril=self._scale_tril)
        log_q_z = gauss.log_prob(z_flat)
        return (log_q_z + bijector_log_det).expand(batch)

    @property
    def latent_names(self) -> list[str]:
        return list(self._registry.names)

    @property
    def hessian_fitted(self) -> bool:
        """Whether `fit_hessian` has been called."""
        return bool(self._hessian_fitted)


__all__ = ["AutoLaplaceApproximation"]
