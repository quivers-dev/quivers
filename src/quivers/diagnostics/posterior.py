"""Reusable posterior analysis over explicit chain, draw, and observation axes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import TYPE_CHECKING, Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.diagnostics.convergence import (
    ConvergenceSummary,
    summary as convergence_summary,
)

if TYPE_CHECKING:
    from xarray import DataTree as _DataTree
    from quivers.inference.guides.base import Guide
    from quivers.inference.mcmc.driver import MCMCResult
else:
    # Keep core posterior summaries importable without the diagnostics extra;
    # static tooling still sees the concrete optional return type.
    _DataTree = object


def _sample_stats(posterior: MCMCResult | Guide, count: int) -> dict[str, torch.Tensor]:
    """Retain recorded sampler statistics on the selected draw window."""
    from quivers.inference.mcmc.driver import MCMCResult

    if not isinstance(posterior, MCMCResult):
        return {}
    return {
        "lp": posterior.log_densities[:, :count],
        **{name: values[:, :count] for name, values in posterior.sample_stats.items()},
    }


class PosteriorSummary(ValueModel):
    """Pooled uncertainty summaries, with MCMC diagnostics when applicable.

    Attributes
    ----------
    mean, sd : torch.Tensor
        Mean and sample standard deviation on the site's event coordinates.
    probs : tuple[float, ...]
        Requested quantile probabilities.
    quantiles : torch.Tensor
        Quantiles with shape ``(len(probs), *event_shape)``.
    convergence : ConvergenceSummary or None
        Recorded-chain diagnostics, or None for variational draws.
    """

    mean: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    sd: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    probs: tuple[float, ...]
    quantiles: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    convergence: ConvergenceSummary | None = None


class WAICResult(ValueModel):
    """Pointwise WAIC estimates using sample log-likelihood variances.

    Attributes
    ----------
    elpd_waic, p_waic, waic : float
        Expected log predictive density, effective parameter count, and
        deviance-scale criterion ``-2 * elpd_waic``.
    se, se_p_waic : float
        Standard errors of elpd_waic and p_waic across observation units;
        NaN for one unit.
    elpd_i, variance_i : torch.Tensor
        Pointwise ELPD and unbiased sample variance over pooled posterior draws.
    warning : bool
        Whether any pointwise log-likelihood variance exceeds 0.4.
    num_samples, num_observations : int
        Pooled posterior draw count and number of observation units.
    """

    elpd_waic: float
    p_waic: float
    waic: float
    se: float
    se_p_waic: float
    elpd_i: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    variance_i: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    warning: bool
    num_samples: int
    num_observations: int

    @property
    def se_waic(self) -> float:
        """Standard error of the deviance-scale WAIC estimate."""
        return 2 * self.se

    @property
    def waic_i(self) -> torch.Tensor:
        """Deviance-scale contributions on the original observation axes."""
        return -2 * self.elpd_i


def waic(log_likelihood: torch.Tensor) -> WAICResult:
    """Estimate WAIC from raw conditional log likelihoods.

    Parameters
    ----------
    log_likelihood : torch.Tensor
        Finite real scores with shape ``(chain, draw, *observation_shape)``.
        Event dimensions must already be reduced. Scores must refer to the
        same observation units across draws and exclude likelihood weights.

    Returns
    -------
    WAICResult
        Log-mean-exp predictive scores minus sample log-score variances.
        The variance convention matches R's loo::waic (divisor S - 1).

    Raises
    ------
    ValueError
        If scores are nonfinite, empty, or supply fewer than two pooled draws.
    TypeError
        If scores are not a real floating-point tensor.
    """
    if (
        not isinstance(log_likelihood, torch.Tensor)
        or not log_likelihood.is_floating_point()
    ):
        raise TypeError("WAIC requires a real floating-point log-likelihood tensor")
    if log_likelihood.ndim < 3 or not all(log_likelihood.shape):
        raise ValueError("WAIC requires nonempty chain, draw, and observation axes")
    values = log_likelihood.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("WAIC requires finite log likelihoods at every posterior draw")
    draws = values.shape[0] * values.shape[1]
    if draws < 2:
        raise ValueError("WAIC requires at least two pooled posterior draws")
    pooled = values.flatten(0, 1)
    variance = pooled.var(dim=0, correction=1)
    elpd = torch.logsumexp(pooled, dim=0) - math.log(draws) - variance
    count = elpd.numel()
    return WAICResult(
        elpd_waic=elpd.sum().item(),
        p_waic=variance.sum().item(),
        waic=-2 * elpd.sum().item(),
        se=(count * elpd.flatten().var(correction=1)).sqrt().item()
        if count > 1
        else math.nan,
        se_p_waic=(count * variance.flatten().var(correction=1)).sqrt().item()
        if count > 1
        else math.nan,
        elpd_i=elpd,
        variance_i=variance,
        warning=bool((variance > 0.4).any()),
        num_samples=draws,
        num_observations=count,
    )


class PosteriorAnalysis(ValueModel):
    """A reusable tensor posterior and its raw observation likelihoods.

    Attributes
    ----------
    samples : Mapping[str, torch.Tensor]
        Constrained draws with chain and draw axes.
    log_likelihood : Mapping[str, torch.Tensor]
        Raw pointwise scores retaining observation axes.
    observed_data : Mapping[str, torch.Tensor]
        Responses corresponding to the score units.
    sample_stats : Mapping[str, torch.Tensor]
        Recorded per-draw sampler statistics, empty for variational draws.
    kind : "mcmc" or "variational"
        Provenance of the draws. Variational pseudo-chains never supply
        MCMC convergence diagnostics.

    Methods consume these retained tensors without replaying the model.
    Tensor mappings are held by identity; callers own their storage.
    """

    samples: Mapping[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    log_likelihood: Mapping[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    observed_data: Mapping[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    sample_stats: Mapping[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    kind: Literal["mcmc", "variational"] = "mcmc"

    def _validate_axes(self) -> None:
        """Check that posterior and likelihood groups share chain/draw axes."""
        shape = None
        for values in self.samples.values():
            if (
                values.ndim < 2
                or not values.is_floating_point()
                or not all(values.shape)
            ):
                raise ValueError(
                    "posterior samples require nonempty chain and draw axes"
                )
            if shape is None:
                shape = values.shape[:2]
            elif values.shape[:2] != shape:
                raise ValueError("posterior sites have incompatible chain/draw axes")
        if shape is None:
            raise ValueError("posterior analysis requires latent draws")
        if self.kind == "variational" and shape[0] != 1:
            raise ValueError("variational draws must have one pseudo-chain")
        for values in self.log_likelihood.values():
            if (
                values.ndim < 3
                or not all(values.shape[:2])
                or values.shape[:2] != shape
            ):
                raise ValueError(
                    "log likelihoods must retain matching chain/draw and observation axes"
                )
        for values in self.sample_stats.values():
            if values.ndim != 2 or values.shape != shape:
                raise ValueError("sampler statistics must match chain/draw axes")
        if self.kind == "variational" and self.sample_stats:
            raise ValueError("variational draws have no recorded sampler statistics")

    def summary(
        self, *, probs: Sequence[float] = (0.05, 0.5, 0.95)
    ) -> dict[str, PosteriorSummary]:
        """Summarize uncertainty, attaching diagnostics only to recorded chains.

        Parameters
        ----------
        probs : Sequence[float]
            Quantile probabilities strictly between zero and one.

        Returns
        -------
        dict[str, PosteriorSummary]
            Float64 summaries on each site's event axes.
        """
        self._validate_axes()
        levels = tuple(probs)
        if any(
            isinstance(p, bool) or not math.isfinite(p) or not 0 < p < 1 for p in levels
        ):
            raise ValueError(
                "summary probabilities must lie strictly between zero and one"
            )
        diagnostics = (
            convergence_summary(self.samples, probs=levels)
            if self.kind == "mcmc"
            else {}
        )
        result = {}
        for name, values in self.samples.items():
            if (
                values.ndim < 2
                or not values.is_floating_point()
                or not all(values.shape)
            ):
                raise ValueError(
                    "posterior samples require nonempty chain and draw axes"
                )
            if name in diagnostics:
                d = diagnostics[name]
                mean, sd, quantiles = d.mean, d.sd, d.quantiles
            else:
                pooled = values.detach().double().flatten(0, 1)
                mean, sd = pooled.mean(0), pooled.std(0, correction=1)
                q = torch.tensor(levels, dtype=torch.float64, device=pooled.device)
                quantiles = (
                    torch.quantile(pooled, q, dim=0)
                    if levels
                    else pooled.new_empty((0, *pooled.shape[1:]))
                )
            result[name] = PosteriorSummary(
                mean=mean,
                sd=sd,
                probs=levels,
                quantiles=quantiles,
                convergence=diagnostics.get(name),
            )
        return result

    def _likelihood_name(self, var_name: str | None) -> str:
        """Select one observation variable without silently adding score units."""
        if var_name is None:
            if len(self.log_likelihood) != 1:
                raise ValueError(
                    "select var_name when there is not exactly one likelihood variable"
                )
            return next(iter(self.log_likelihood))
        if var_name not in self.log_likelihood:
            raise ValueError(f"unknown log-likelihood variable {var_name!r}")
        return var_name

    def waic(self, *, var_name: str | None = None) -> WAICResult:
        """Compute WAIC for one response's retained raw observation scores.

        Parameters
        ----------
        var_name : str or None
            Observation variable; required for multiple response variables.

        Returns
        -------
        WAICResult
            Pointwise and aggregate WAIC estimates.
        """
        self._validate_axes()
        return waic(self.log_likelihood[self._likelihood_name(var_name)])

    def to_datatree(
        self,
        *,
        coords: Mapping[str, list] | None = None,
        dims: Mapping[str, list[str]] | None = None,
    ) -> _DataTree:
        """Convert retained tensors for the optional ArviZ diagnostics package.

        Parameters
        ----------
        coords : Mapping[str, list] or None
            Coordinate labels for named event or observation dimensions.
        dims : Mapping[str, list[str]] or None
            Dimension names per variable, excluding chain and draw axes.

        Returns
        -------
        xarray.DataTree
            Posterior, likelihood, observed-data, and sampler-stat groups.
            Chain and draw axes are retained for both posterior kinds.
        """
        self._validate_axes()
        from quivers.diagnostics.arviz_io import _groups_to_datatree

        tree = _groups_to_datatree(
            {
                "posterior": self.samples,
                "log_likelihood": self.log_likelihood,
                "observed_data": self.observed_data,
                "sample_stats": self.sample_stats,
            },
            coords=coords,
            dims=dims,
        )
        tree.attrs["quivers_posterior_kind"] = self.kind
        return tree

    def loo(self, *, var_name: str | None = None, pointwise: bool = True) -> object:
        """Run optional ArviZ PSIS-LOO without recomputing model likelihoods.

        Parameters
        ----------
        var_name : str or None
            Observation variable; required for multiple responses.
        pointwise : bool
            Include observation-level ELPD and Pareto shape diagnostics.

        Returns
        -------
        arviz.ELPDData
            ArviZ's PSIS-LOO result. Variational draws use relative efficiency
            one; this does not correct variational approximation error.
        """
        import arviz as az

        name = self._likelihood_name(var_name)
        return az.loo(
            self.to_datatree(),
            var_name=name,
            pointwise=pointwise,
            reff=1.0 if self.kind == "variational" else None,
        )


__all__ = ["PosteriorAnalysis", "PosteriorSummary", "WAICResult", "waic"]
