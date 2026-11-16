"""Reusable posterior analysis over explicit chain, draw, and observation axes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import json
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics._numeric import empirical_summary, probabilities, real_samples
from quivers.diagnostics.contrast import ContrastExpression
from quivers.diagnostics.hypothesis import HypothesisSummary
from quivers.diagnostics.density import PointNullSpec
from quivers.diagnostics.convergence import (
    ConvergenceSummary,
    summary as convergence_summary,
)

if TYPE_CHECKING:
    from xarray import DataTree as _DataTree
    from quivers.inference.guides.base import Guide
    from quivers.inference.mcmc.driver import MCMCResult
    from quivers.inference.prior import PriorDraws
else:
    # Keep core posterior summaries importable without the diagnostics extra;
    # static tooling still sees the concrete optional return type.
    _DataTree = object


def _sample_stats(
    posterior: MCMCResult | PriorDraws | Guide, count: int
) -> dict[str, torch.Tensor]:
    """Retain recorded sampler statistics on the selected draw window.

    Parameters
    ----------
    posterior : MCMCResult, PriorDraws, or Guide
        Recorded sampler output or a draw source without sampler statistics.
    count : int
        Number of leading draws to retain from every recorded chain.

    Returns
    -------
    dict[str, torch.Tensor]
        Recorded log densities and sampler statistics, or an empty mapping
        for independent prior and variational draws.
    """
    from quivers.inference.mcmc.driver import MCMCResult

    if not isinstance(posterior, MCMCResult):
        return {}
    return {
        "lp": posterior.log_densities[:, :count],
        **{name: values[:, :count] for name, values in posterior.sample_stats.items()},
    }


class PosteriorSummary(ValueModel):
    """Pooled uncertainty summaries, with MCMC diagnostics when applicable.

    Parameters
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
    convergence: ConvergenceSummary | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


class PosteriorCovariance(ValueModel):
    """Joint empirical covariance and correlation of retained coordinates.

    Parameters
    ----------
    covariance, correlation : torch.Tensor
        Float64 square matrices, or one matrix per chain when ``chainwise``.
        A constant coordinate has zero covariance and NaN correlations.
    coordinates : tuple[tuple[str, tuple[int, ...]], ...]
        Matrix columns as an exact variable name and row-major event index.
        The same coordinates label matrix rows; no string parsing is needed.
    dimension_names, coordinate_labels : tuple
        Event dimension names and their original labels for each matrix column.
    variables, event_shapes : tuple
        Selected variable order and corresponding original event shapes.
    sample_shape : tuple[int, int]
        Original chain and draw counts, before the empirical reduction.
    chain_labels, draw_labels : tuple
        Original sample-axis labels, or their positional defaults.
    ddof : int
        Degrees of freedom subtracted from the pooled or per-chain draw count.
    chainwise : bool
        Whether each chain was summarized separately.
    kind : str
        Original draw provenance; this record adds no convergence diagnostics.
    """

    covariance: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    correlation: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    coordinates: tuple[tuple[str, tuple[int, ...]], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    dimension_names: tuple[tuple[str, ...], ...]
    coordinate_labels: tuple[tuple[object, ...], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    variables: tuple[str, ...]
    event_shapes: tuple[tuple[int, ...], ...]
    sample_shape: tuple[int, int] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    chain_labels: tuple[object, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    draw_labels: tuple[object, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    ddof: int
    chainwise: bool
    kind: Literal["mcmc", "variational", "prior"]


class WAICResult(ValueModel):
    """Pointwise WAIC estimates using sample log-likelihood variances.

    Parameters
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
        """Standard error of the deviance-scale WAIC estimate.

        Returns
        -------
        float
            Twice the ELPD-scale standard error.
        """
        return 2 * self.se

    @property
    def waic_i(self) -> torch.Tensor:
        """Deviance-scale contributions on the original observation axes.

        Returns
        -------
        torch.Tensor
            Negative twice each retained pointwise ELPD estimate.
        """
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

    Parameters
    ----------
    samples : Mapping[str, torch.Tensor]
        Constrained draws with chain and draw axes.
    log_likelihood : Mapping[str, torch.Tensor]
        Raw pointwise scores retaining observation axes.
    observed_data : Mapping[str, torch.Tensor]
        Responses corresponding to the score units.
    sample_stats : Mapping[str, torch.Tensor]
        Recorded per-draw sampler statistics, empty for variational draws.
    kind : "mcmc" or "variational" or "prior"
        Provenance of the draws. Variational and independent prior layout
        axes never supply MCMC convergence diagnostics.
    group_dims : Mapping
        Optional group-specific event dimensions, preserving observed vector
        events alongside their reduced pointwise likelihood axes.
    coords, dims : Mapping
        Retained coordinate labels and event dimension names, used by default
        for DataTree conversion and covariance. Explicit method arguments override them.

    Notes
    -----
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
    kind: Literal["mcmc", "variational", "prior"] = "mcmc"
    coords: Mapping[str, Sequence[object]] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    dims: Mapping[str, Sequence[str]] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    group_dims: Mapping[str, Mapping[str, Sequence[str]]] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    def _validate_axes(self) -> None:
        """Check that posterior and likelihood groups share chain/draw axes.

        Raises
        ------
        ValueError
            If sample axes are empty or disagree, observation scores use
            different sample axes, or non-MCMC draws claim sampler statistics
            or multiple independent layout axes.
        """
        shape = None
        for values in self.samples.values():
            if (
                values.ndim < 2
                or (not values.is_floating_point() and self.kind != "prior")
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
            # A parameter-free program still has recorded predictive units.
            # Their actual draw axes supply the layout without a dummy latent.
            first = next(iter(self.log_likelihood.values()), None)
            if first is not None and first.ndim >= 3 and all(first.shape[:2]):
                shape = first.shape[:2]
            elif self.sample_stats:
                first = next(iter(self.sample_stats.values()))
                if first.ndim == 2 and all(first.shape):
                    shape = first.shape
            if shape is None:
                raise ValueError("posterior analysis requires retained draw axes")
        if self.kind != "mcmc" and shape[0] != 1:
            raise ValueError(
                "variational draws must have one pseudo-chain; independent prior draws require one layout axis"
            )
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
        if self.kind != "mcmc" and self.sample_stats:
            raise ValueError(
                "variational and independent prior draws have no sampler statistics"
            )

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
            Float64 summaries on each site's event axes. Any coordinate with
            a nonfinite draw has NaN location, dispersion, and quantiles;
            no draws are deleted. Original integer capacity is checked before
            conversion, including independent prior draws.

        Raises
        ------
        ValueError
            If sample axes or quantile probabilities are invalid, or an original
            integer exceeds exact Float64 capacity.
        """
        self._validate_axes()
        levels = probabilities(probs)
        diagnostics = (
            convergence_summary(self.samples, probs=levels)
            if self.kind == "mcmc"
            else {}
        )
        result = {}
        for name, values in self.samples.items():
            if (
                values.ndim < 2
                or (not values.is_floating_point() and self.kind != "prior")
                or not all(values.shape)
            ):
                raise ValueError(
                    "posterior samples require nonempty chain and draw axes"
                )
            if name in diagnostics:
                d = diagnostics[name]
                mean, sd, quantiles = d.mean, d.sd, d.quantiles
            else:
                pooled = real_samples(
                    values, label=f"summary {name!r}", finite=False
                ).flatten(0, 1)
                mean, sd, quantiles = empirical_summary(pooled, levels)
            result[name] = PosteriorSummary(
                mean=mean,
                sd=sd,
                probs=levels,
                quantiles=quantiles,
                convergence=diagnostics.get(name),
            )
        return result

    def covariance(
        self,
        var_names: Sequence[str] | None = None,
        *,
        ddof: int = 1,
        chainwise: bool = False,
        coords: Mapping[str, Sequence[object]] | None = None,
        dims: Mapping[str, Sequence[str]] | None = None,
    ) -> PosteriorCovariance:
        """Summarize joint empirical dependence without replaying a posterior.

        Parameters
        ----------
        var_names : Sequence[str] or None
            Distinct selected sites, in matrix order; None selects insertion order.
            Each site's event coordinates flatten in row-major order.
        ddof : int
            Nonnegative degrees of freedom, strictly below the reduced count.
        chainwise : bool
            Retain a separate matrix per chain instead of pooling chain and draw.
        coords : Mapping[str, Sequence[object]] or None
            Labels for named event axes and optional chain/draw axes.
        dims : Mapping[str, Sequence[str]] or None
            Event-axis names per site, following the DataTree convention.

        Returns
        -------
        PosteriorCovariance
            Float64 covariance and Pearson correlation, structural coordinate
            labels, sample-axis labels, and provenance. Constants have NaN
            correlations. No convergence statistics are manufactured.

        Raises
        ------
        ValueError, TypeError
            If axes, labels, selection, ddof, devices, or numeric carriers are
            invalid; missing/nonfinite draws are refused, without deletion.
            Integer values outside the exact Float64 range are refused before
            casting, as are unrepresentable covariance results.
        """
        from quivers.diagnostics._coordinates import axis_labels, event_dimensions
        from quivers.diagnostics._numeric import real_samples

        self._validate_axes()
        coords = self.coords if coords is None else coords
        dims = self.dims if dims is None else dims
        if isinstance(var_names, str | bytes):
            raise TypeError("var_names must be a sequence of site names")
        names = tuple(self.samples) if var_names is None else tuple(var_names)
        if any(not isinstance(name, str) for name in names):
            raise TypeError("posterior variable names must be strings")
        if not names or len(set(names)) != len(names):
            raise ValueError("covariance requires a nonempty distinct site selection")
        unknown = set(names) - self.samples.keys()
        if unknown:
            raise ValueError(f"unknown posterior variables: {sorted(unknown)}")
        if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
            raise ValueError("ddof must be a nonnegative integer")
        if not isinstance(chainwise, bool):
            raise TypeError("chainwise must be Boolean")
        first = self.samples[names[0]]
        sample_shape = (first.shape[0], first.shape[1])
        chains, draws = sample_shape
        count = draws if chainwise else chains * draws
        if ddof >= count:
            raise ValueError("ddof must be smaller than the reduced sample count")
        shapes = {name: tuple(self.samples[name].shape[2:]) for name in names}
        dimensions = event_dimensions(shapes, coords=coords, dims=dims)
        chain_labels = axis_labels("chain", chains, coords)
        draw_labels = axis_labels("draw", draws, coords)
        columns, structural, dimension_names, labels = [], [], [], []
        device = self.samples[names[0]].device
        for name in names:
            values = self.samples[name]
            if values.device != device:
                raise ValueError("covariance requires samples on one device")
            columns.append(
                real_samples(values, label="covariance").reshape(chains, draws, -1)
            )
            dim_names = dimensions[name]
            dim_labels = [
                axis_labels(dim, size, coords)
                for dim, size in zip(dim_names, shapes[name], strict=True)
            ]
            for index in product(*(range(size) for size in shapes[name])):
                structural.append((name, index))
                dimension_names.append(dim_names)
                labels.append(
                    tuple(
                        label[position]
                        for label, position in zip(dim_labels, index, strict=True)
                    )
                )
        values = torch.cat(columns, dim=-1)
        if not chainwise:
            values = values.flatten(0, 1)
        # Anchor subtraction retains small differences around a large common
        # offset. Scaling also handles opposite finite endpoints whose
        # subtraction would overflow, without summing their raw magnitudes.
        delta = values - values[..., :1, :]
        working = torch.where(
            torch.isfinite(delta).all(dim=-2, keepdim=True), delta, values
        )
        scale = working.abs().amax(dim=-2, keepdim=True)
        normalized = working / torch.where(scale == 0, 1.0, scale)
        centered = normalized - normalized.mean(dim=-2, keepdim=True)
        gram = centered.transpose(-1, -2) @ centered / (count - ddof)
        mantissa, exponent = torch.frexp(scale.squeeze(-2))
        base_mantissa, base_exponent = torch.frexp(gram)
        covariance = torch.ldexp(
            base_mantissa * mantissa.unsqueeze(-1) * mantissa.unsqueeze(-2),
            base_exponent + exponent.unsqueeze(-1) + exponent.unsqueeze(-2),
        )
        if not bool(torch.isfinite(covariance).all()):
            raise ValueError("covariance exceeds Float64 capacity")
        sd = gram.diagonal(dim1=-2, dim2=-1).sqrt()
        correlation = (gram / sd.unsqueeze(-1) / sd.unsqueeze(-2)).clamp(-1.0, 1.0)
        return PosteriorCovariance(
            covariance=covariance,
            correlation=correlation,
            coordinates=tuple(structural),
            dimension_names=tuple(dimension_names),
            coordinate_labels=tuple(labels),
            variables=names,
            event_shapes=tuple(shapes.values()),
            sample_shape=sample_shape,
            chain_labels=chain_labels,
            draw_labels=draw_labels,
            ddof=ddof,
            chainwise=chainwise,
            kind=self.kind,
        )

    def hypothesis(
        self,
        expression: str | ContrastExpression,
        *,
        prior: PosteriorAnalysis | None = None,
        null: float | int | torch.Tensor = 0.0,
        probs: Sequence[float] = (0.025, 0.975),
        robust: bool = False,
        point: PointNullSpec | None = None,
        name: str = "contrast",
        dims: Mapping[str, Sequence[str]] | None = None,
        coords: Mapping[str, Sequence[object]] | None = None,
    ) -> HypothesisSummary:
        """Evaluate one checked QVR contrast over retained joint draws.

        Parameters
        ----------
        expression : str or ContrastExpression
            QVR value expression and optional aliases to original site names.
        prior : PosteriorAnalysis or None
            Joint draws from the corresponding proper encompassing prior.
            Draw counts may differ; event shapes and devices must agree.
        null : float, int, or torch.Tensor
            Finite comparison value on event axes only.
        probs : Sequence[float]
            Requested quantile probabilities in output order.
        robust : bool
            Use median and scaled MAD instead of mean and sample SD.
        point : PointNullSpec or None
            Optional marginal density comparison on the same joint prior and
            posterior contrast; requires a checked Real result and prior draws.
        name : str
            Original label of the derived contrast.
        dims, coords : Mapping or None
            Input event-axis identities and labels, with optional sample labels.
            Distinct axes remain distinct even when their extents coincide.

        Returns
        -------
        HypothesisSummary
            Existing uncertainty, strict-region posterior odds, and optional
            prior-corrected region evidence. Equality frequencies are descriptive;
            requested point_evidence retains a separate marginal density ratio.
        """
        from quivers.diagnostics.contrast import _contrast_hypothesis

        self._validate_axes()
        if prior is not None:
            prior._validate_axes()
        return _contrast_hypothesis(
            self.samples,
            expression,
            prior=prior.samples if prior is not None else None,
            null=null,
            probs=probs,
            robust=robust,
            point=point,
            name=name,
            kind=self.kind,
            dims=dims,
            coords=coords,
        )

    def _likelihood_name(self, var_name: str | None) -> str:
        """Select one observation variable without silently adding score units.

        Parameters
        ----------
        var_name : str or None
            Original observation name, optional for exactly one score variable.

        Returns
        -------
        str
            The selected key in the retained log-likelihood mapping.

        Raises
        ------
        ValueError
            If no unique variable can be inferred or the requested key is absent.
        """
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
        from quivers.diagnostics.arviz_io import (
            _ARVIZ_SAMPLE_STATS,
            _groups_to_datatree,
        )

        groups = {
            "posterior": self.samples,
            "log_likelihood": self.log_likelihood,
            "observed_data": self.observed_data,
            "sample_stats": self.sample_stats,
        }
        tree = _groups_to_datatree(
            groups,
            coords={
                name: list(values)
                for name, values in (self.coords if coords is None else coords).items()
            },
            dims={
                name: list(values)
                for name, values in (self.dims if dims is None else dims).items()
            },
            group_dims={
                group: {name: list(axes) for name, axes in names.items()}
                for group, names in self.group_dims.items()
            }
            if dims is None
            else None,
        )
        tree.attrs["quivers_posterior_kind"] = self.kind
        tree.attrs["quivers_posterior_format"] = "qvr-posterior/v1"
        tree.attrs["quivers_sample_stat_names"] = json.dumps(
            {_ARVIZ_SAMPLE_STATS.get(name, name): name for name in self.sample_stats},
            sort_keys=True,
        )
        tree.attrs["quivers_tensor_dtypes"] = json.dumps(
            {
                group: {
                    (
                        _ARVIZ_SAMPLE_STATS.get(name, name)
                        if group == "sample_stats"
                        else name
                    ): str(value.dtype)
                    for name, value in values.items()
                }
                for group, values in groups.items()
            },
            sort_keys=True,
        )
        return tree

    @classmethod
    def from_datatree(
        cls,
        tree: _DataTree,
        *,
        kind: Literal["mcmc", "variational", "prior"] | None = None,
    ) -> PosteriorAnalysis:
        """Restore retained analysis tensors and labels from an ArviZ DataTree.

        Parameters
        ----------
        tree : xarray.DataTree
            Canonical posterior, likelihood, observed-data, and sample-stat groups.
        kind : {"mcmc", "variational", "prior"} or None
            Explicit provenance for external trees. QVR trees retain it in metadata;
            an override must agree. Chain count never determines provenance.

        Returns
        -------
        PosteriorAnalysis
            Independently owned CPU tensors, dimensions, labels, and provenance.
            This restores analysis, rather than an executable predictor or optimizer.

        Raises
        ------
        ValueError, TypeError
            If provenance, format, labels, groups, or sampled axes are inconsistent.
        """
        from quivers.diagnostics.arviz_io import _posterior_from_datatree

        return _posterior_from_datatree(tree, kind=kind)

    def to_netcdf(self, path: str | Path) -> None:
        """Write retained analysis through the existing DataTree NetCDF format.

        Parameters
        ----------
        path : str or Path
            Output NetCDF file. Requires the diagnostics extra.
        """
        from quivers._atomic import atomic_output

        tree = self.to_datatree()
        with atomic_output(Path(path)) as temporary:
            encoding = {
                f"/{group}": {
                    name: {"dtype": "float32"}
                    for name, variable in node.dataset.data_vars.items()
                    if variable.dtype == "float16"
                }
                for group, node in tree.children.items()
            }
            tree.to_netcdf(temporary, engine="netcdf4", encoding=encoding)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> PosteriorAnalysis:
        """Read a QVR retained analysis without reconstructing a fitted evaluator.

        Parameters
        ----------
        path : str or Path
            NetCDF file written from a QVR posterior DataTree.

        Returns
        -------
        PosteriorAnalysis
            Owned CPU tensors and coordinate metadata, after axis validation.

        Raises
        ------
        ValueError, TypeError
            If the stored tree does not satisfy the retained analysis contract.
        OSError
            If the input file is missing or cannot be opened as NetCDF.
        """
        import xarray as xr

        with xr.open_datatree(path, engine="netcdf4") as tree:
            return cls.from_datatree(tree)

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
            If every likelihood column is constant across draws, exact uniform
            weights replace an undefined tail fit, and Pareto shapes are NaN.
        """
        import arviz as az

        name = self._likelihood_name(var_name)
        tree = self.to_datatree()
        scores = self.log_likelihood[name]
        if (
            scores.numel()
            and bool(torch.isfinite(scores).all())
            and torch.equal(scores, scores[0, 0].expand_as(scores))
        ):
            # Constant likelihood columns have exactly uniform importance
            # weights. No tail distribution exists to fit or diagnose.
            import xarray as xr

            likelihood = tree["log_likelihood"].dataset[name]
            count = scores.shape[0] * scores.shape[1]
            return az.loo(
                tree,
                var_name=name,
                pointwise=pointwise,
                reff=1.0,
                log_weights=xr.full_like(likelihood, -math.log(count)),
                pareto_k=xr.full_like(
                    likelihood.isel(chain=0, draw=0, drop=True), math.nan
                ),
            )
        return az.loo(
            tree,
            var_name=name,
            pointwise=pointwise,
            reff=1.0 if self.kind != "mcmc" or not self.samples else None,
        )


__all__ = [
    "PosteriorAnalysis",
    "PosteriorSummary",
    "PosteriorCovariance",
    "WAICResult",
    "waic",
]
