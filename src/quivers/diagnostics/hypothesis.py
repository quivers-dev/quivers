"""Empirical contrast summaries and region evidence from aligned retained draws."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics._coordinates import axis_labels, event_dimensions
from quivers.diagnostics._numeric import empirical_summary, probabilities, real_samples
from quivers.diagnostics.density import DensityRatio, PointNullSpec, density_ratio


class RegionEvidence(ValueModel):
    """Empirical evidence for a strict region against its complement.

    Parameters
    ----------
    probability, log_odds : torch.Tensor
        Posterior region frequency and its natural-log odds. Zero and unit
        frequencies give negative and positive infinity, without pseudocounts.
    prior_probability, prior_log_odds : torch.Tensor or None
        Corresponding frequencies and log odds from jointly aligned prior draws.
    log_bayes_factor : torch.Tensor or None
        Posterior minus prior log odds. NaN when the empirical prior region has
        probability zero or one: both restricted prior models cannot be formed.
        This empirical region comparison is not point-null density evidence.
    """

    probability: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_odds: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    prior_probability: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    prior_log_odds: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    log_bayes_factor: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


class HypothesisSummary(ValueModel):
    """Uncertainty and strict-region evidence for an empirical contrast.

    Parameters
    ----------
    estimate, est_error : torch.Tensor
        Mean and sample standard deviation, or median and normal-consistent
        median absolute deviation when ``robust``. These describe the supplied
        contrast in its original units, without subtracting the null.
    probs, quantiles : tuple[float, ...], torch.Tensor
        Requested type-7 quantile probabilities and values with a leading
        probability axis. Remaining axes are the original event coordinates.
    null : torch.Tensor
        Finite comparison value broadcast only across event coordinates.
    below, above : RegionEvidence
        Evidence for the strict regions ``contrast < null`` and ``contrast > null``.
        Each complement includes exact ties.
    equal_probability, prior_equal_probability : torch.Tensor, torch.Tensor or None
        Descriptive equality frequencies, without a point-null Bayes factor.
    name, dimension_names, coordinate_labels : str, tuple, tuple
        Original contrast name, named event axes, and their original labels.
    sample_shape, prior_sample_shape : tuple[int, ...], tuple[int, ...] or None
        Original chain and draw counts; prior counts may differ.
    chain_labels, draw_labels : tuple
        Original posterior sample-axis labels or their positional defaults.
    robust : bool
        Whether location and dispersion use median and scaled MAD.
    kind : str
        Caller-supplied provenance; no convergence diagnostics are inferred.
    point_evidence : DensityRatio or None
        Optional marginal posterior/prior density comparison at the null,
        separate from strict-region evidence and descriptive equality frequency.
    """

    estimate: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    est_error: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    probs: tuple[float, ...]
    quantiles: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    null: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    below: RegionEvidence = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    above: RegionEvidence = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    equal_probability: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    prior_equal_probability: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    name: str
    dimension_names: tuple[str, ...]
    coordinate_labels: tuple[tuple[object, ...], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    sample_shape: tuple[int, ...]
    prior_sample_shape: tuple[int, ...] | None = None
    chain_labels: tuple[object, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    draw_labels: tuple[object, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    robust: bool
    kind: Literal["mcmc", "variational", "prior", "unspecified"]
    point_evidence: DensityRatio | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


def _contrasts(values: torch.Tensor, label: str) -> torch.Tensor:
    """Validate and pool two explicit sample axes without changing event axes.

    Parameters
    ----------
    values : torch.Tensor
        Retained real draws with leading chain and draw axes.
    label : str
        Name used in validation errors.

    Returns
    -------
    torch.Tensor
        Detached Float64 draws with the sample axes pooled.

    Raises
    ------
    ValueError
        If a sample axis is absent or any axis is empty.
    """
    values = real_samples(values, label=label)
    if values.ndim < 2 or not all(values.shape):
        raise ValueError(f"{label} requires nonempty chain and draw axes")
    return values.flatten(0, 1)


def _null_value(null: float | int | torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Check a finite null and broadcast it only over event axes.

    Parameters
    ----------
    null : float, int, or torch.Tensor
        Scalar value or event-shaped tensor.
    values : torch.Tensor
        Pooled Float64 contrast draws.

    Returns
    -------
    torch.Tensor
        Null broadcast to the event shape on the contrast device.

    Raises
    ------
    ValueError, TypeError
        If the null is nonfinite, exceeds exact integer capacity, differs in
        device, or cannot broadcast solely over event axes.
    """
    if isinstance(null, torch.Tensor):
        if null.device != values.device:
            raise ValueError("null and contrast must use the same device")
        result = real_samples(null, label="null")
    elif isinstance(null, int):
        # use the same original-carrier guard as retained integer draws.
        from quivers._count_capacity import _MAX_EXACT_COUNT

        if abs(null) > _MAX_EXACT_COUNT:
            raise ValueError("null integer inputs must lie within +/-2**53")
        result = values.new_tensor(null)
    elif isinstance(null, float):
        result = real_samples(values.new_tensor(null), label="null")
    else:
        raise TypeError("null requires a real scalar or tensor")
    if result.ndim > values.ndim - 1:
        raise ValueError("null must broadcast only over contrast event axes")
    try:
        return torch.broadcast_to(result, values.shape[1:])
    except RuntimeError as error:
        raise ValueError("null must broadcast only over contrast event axes") from error


def _region(mask: torch.Tensor, prior_mask: torch.Tensor | None) -> RegionEvidence:
    """Reduce region indicators and correct posterior log odds by prior odds.

    Parameters
    ----------
    mask, prior_mask : torch.Tensor, torch.Tensor or None
        Boolean region indicators with a leading pooled sample axis.

    Returns
    -------
    RegionEvidence
        Empirical strict-region probabilities and log-space evidence.
    """
    probability = mask.to(torch.float64).mean(0)
    log_odds = torch.log(probability) - torch.log1p(-probability)
    if prior_mask is None:
        return RegionEvidence(probability=probability, log_odds=log_odds)
    prior_probability = prior_mask.to(torch.float64).mean(0)
    prior_log_odds = torch.log(prior_probability) - torch.log1p(-prior_probability)
    defined = (prior_probability > 0) & (prior_probability < 1)
    log_bayes_factor = torch.where(
        defined, log_odds - prior_log_odds, torch.full_like(log_odds, torch.nan)
    )
    return RegionEvidence(
        probability=probability,
        log_odds=log_odds,
        prior_probability=prior_probability,
        prior_log_odds=prior_log_odds,
        log_bayes_factor=log_bayes_factor,
    )


def hypothesis_summary(
    contrast: torch.Tensor,
    *,
    null: float | int | torch.Tensor = 0.0,
    probs: Sequence[float] = (0.025, 0.975),
    prior: torch.Tensor | None = None,
    robust: bool = False,
    point: PointNullSpec | None = None,
    name: str = "contrast",
    kind: Literal["mcmc", "variational", "prior", "unspecified"] = "unspecified",
    dims: Sequence[str] | None = None,
    coords: Mapping[str, Sequence[object]] | None = None,
) -> HypothesisSummary:
    """Summarize a retained contrast and evidence for each strict null region.

    Parameters
    ----------
    contrast : torch.Tensor
        Finite real draws shaped ``(chain, draw, *event_shape)``. Construct any
        derived contrast jointly within each retained draw before calling.
    null : float, int, or torch.Tensor
        Finite null value, scalar or broadcastable over event axes only.
    probs : Sequence[float]
        Quantile probabilities strictly between zero and one, in requested order.
    prior : torch.Tensor or None
        Joint prior contrast draws with the same event shape and device. Its
        chain and draw counts may differ. No permutation or inference is run.
    robust : bool
        Return median and MAD divided by the standard-normal 0.75 quantile
        instead of mean and sample SD. Quantiles do not depend on this flag.
    point : PointNullSpec or None
        Explicit marginal point-density request. Requires finite continuous
        floating posterior and prior draws; the two strict regions are unchanged.
    name : str
        Nonempty contrast name, retained without parsing.
    kind : {"mcmc", "variational", "prior", "unspecified"}
        Explicit provenance of the supplied contrast draws.
    dims : Sequence[str] or None
        Named event axes, using the shared retained-analysis coordinate policy.
    coords : Mapping[str, Sequence[object]] or None
        Labels for event, chain, or draw axes; labels retain their original values.

    Returns
    -------
    HypothesisSummary
        Detached Float64 summaries on the original device. Equality frequencies
        are descriptive. Optional point_evidence retains the shared density
        estimator's bandwidths and assumption declarations, without inferring
        posterior model probabilities. Single-draw SD is NaN; constant SD is zero.

    Raises
    ------
    ValueError, TypeError
        If axes, alignment, domains, numeric capacity, or metadata disagree.
        Missing values are refused without deletion. Dispersion overflow is refused;
        values below Float64 resolution may underflow to zero.

    Notes
    -----
    A region Bayes factor compares a common proper encompassing prior restricted
    and renormalized to that region and its complement, under the same likelihood.
    Supplied prior draws must represent that prior jointly. A zero empirical
    prior region frequency makes this estimate undefined (NaN), even though the
    true prior might assign positive mass. No pseudocounts are added. A posterior
    zero frequency may instead produce infinite empirical log evidence; finite
    draws do not establish an exact population probability of zero.
    """
    if point is not None and not isinstance(point, PointNullSpec):
        raise TypeError("point must be PointNullSpec or None")
    if point is not None and prior is None:
        raise ValueError("point-density evidence requires joint prior draws")
    values = _contrasts(contrast, "contrast")
    levels = probabilities(probs)
    if not isinstance(robust, bool):
        raise TypeError("robust must be Boolean")
    if not isinstance(name, str) or not name:
        raise ValueError("contrast name must be a nonempty string")
    if kind not in {"mcmc", "variational", "prior", "unspecified"}:
        raise ValueError("contrast kind must identify its draw provenance")
    cutoff = _null_value(null, values)
    prior_values = None
    if prior is not None:
        prior_values = _contrasts(prior, "prior contrast")
        if prior_values.device != values.device:
            raise ValueError("prior and posterior contrasts must use the same device")
        if prior_values.shape[1:] != values.shape[1:]:
            raise ValueError(
                "prior and posterior contrasts require the same event shape"
            )
    dimension_names = event_dimensions(
        {name: values.shape[1:]},
        dims={name: dims} if dims is not None else None,
        coords=coords,
    )[name]
    estimate, dispersion, quantiles = empirical_summary(values, levels, robust=robust)
    if bool(torch.isinf(dispersion).any()):
        raise ValueError("contrast dispersion exceeds Float64 capacity")
    return HypothesisSummary(
        estimate=estimate,
        est_error=dispersion,
        probs=levels,
        quantiles=quantiles,
        null=cutoff,
        below=_region(
            values < cutoff, prior_values < cutoff if prior_values is not None else None
        ),
        above=_region(
            values > cutoff, prior_values > cutoff if prior_values is not None else None
        ),
        equal_probability=(values == cutoff).to(torch.float64).mean(0),
        prior_equal_probability=(prior_values == cutoff).to(torch.float64).mean(0)
        if prior_values is not None
        else None,
        name=name,
        dimension_names=dimension_names,
        coordinate_labels=tuple(
            axis_labels(dimension, extent, coords)
            for dimension, extent in zip(dimension_names, values.shape[1:], strict=True)
        ),
        sample_shape=tuple(contrast.shape[:2]),
        prior_sample_shape=tuple(prior.shape[:2]) if prior is not None else None,
        chain_labels=axis_labels("chain", contrast.shape[0], coords),
        draw_labels=axis_labels("draw", contrast.shape[1], coords),
        robust=robust,
        kind=kind,
        point_evidence=(
            density_ratio(
                contrast,
                prior,
                null=null,
                bandwidth=point.bandwidth,
                prior_bandwidth=point.prior_bandwidth,
                adjust=point.adjust,
                assumptions=point.assumptions,
                name=name,
                dims=dimension_names,
                coords=coords,
                max_working_elements=point.max_working_elements,
            )
            if point is not None and prior is not None
            else None
        ),
    )


__all__ = ["hypothesis_summary", "HypothesisSummary", "RegionEvidence"]
