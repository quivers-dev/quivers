"""Direct Gaussian density ordinates and declared point-null density evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import didactic.api as dx
import torch

from quivers._count_capacity import _MAX_EXACT_COUNT
from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics._coordinates import axis_labels, event_dimensions
from quivers.diagnostics._numeric import empirical_summary, real_samples, scaled_samples


class PointNullAssumptions(ValueModel):
    """Caller declarations needed to interpret a density ratio as point evidence.

    Parameters
    ----------
    proper_continuous_prior : bool
        The encompassing prior is proper and induces a continuous contrast
        with a finite, positive density at the null. Its corresponding posterior
        also has a density there; neither distribution has an atom at the null.
    compatible_null_prior : bool
        The null model uses the encompassing likelihood restricted to the null,
        and its nuisance prior equals the encompassing prior conditional on that
        null, with the same underlying dominating measures.

    Notes
    -----
    These declarations are not mechanically verified from the draws. They do
    not assert KDE accuracy. Arbitrary nonlinear contrasts may fail the density
    or compatibility assumptions even when each parameter has a proper prior.
    """

    proper_continuous_prior: bool
    compatible_null_prior: bool


class PointNullSpec(ValueModel):
    """Request marginal density evidence from the shared joint contrast streams.

    Parameters
    ----------
    bandwidth : str, float, or torch.Tensor
        Posterior Gaussian bandwidth: nrd0, scott, or a positive scalar/event
        tensor. Bandwidth tensors retain their original identity.
    prior_bandwidth : str, float, torch.Tensor, or None
        Prior bandwidth policy; None estimates it separately under bandwidth.
    adjust : float
        Positive multiplier on both selected bandwidths.
    assumptions : PointNullAssumptions or None
        Optional explicit declarations allowing a point-null BF01 interpretation.
        Without both declarations the result remains a diagnostic density ratio.
    max_working_elements : int
        Positive bound on each shared density estimator's kernel workspace.

    Notes
    -----
    Every retained event coordinate is a separate scalar contrast; this request
    does not estimate a joint multivariate point-null density. The estimator's
    resource, finite-continuous-sample, and bandwidth checks apply when used.
    """

    bandwidth: str | float | torch.Tensor = dx.field(
        default="nrd0", opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    prior_bandwidth: str | float | torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    adjust: float = 1.0
    assumptions: PointNullAssumptions | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    max_working_elements: int = 262144


class DensityOrdinate(ValueModel):
    """A marginal Gaussian KDE ordinate for each scalar event coordinate.

    Parameters
    ----------
    log_density : torch.Tensor
        Natural-log density in the original contrast units.
    at : torch.Tensor
        Evaluation point broadcast only over event coordinates.
    log_bandwidth : torch.Tensor
        Absolute Gaussian kernel bandwidth in log space, including adjustment.
        This representation remains authoritative at extreme numerical scales.
    bandwidth_rule : str
        ``nrd0``, ``scott``, or ``explicit``, describing bandwidth selection.
    adjust : float
        Positive multiplier applied to the selected bandwidth.
    sample_shape : tuple[int, ...]
        Retained chain and draw counts. Draws are equally weighted and pooled.
    kernel_effective_count : torch.Tensor
        Squared sum of local kernel weights divided by their sum of squares.
        This measures local weight concentration, not MCMC effective sample size.
    name, dimension_names, coordinate_labels : str, tuple, tuple
        Original contrast name, named event dimensions, and original labels.
    max_working_elements : int
        Upper bound on each temporary kernel tile, excluding input storage,
        output coordinates, and the shared bandwidth-reduction workspace.
    """

    log_density: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    at: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_bandwidth: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    bandwidth_rule: str
    adjust: float
    sample_shape: tuple[int, ...]
    kernel_effective_count: torch.Tensor = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    name: str
    dimension_names: tuple[str, ...]
    coordinate_labels: tuple[tuple[object, ...], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    max_working_elements: int

    @property
    def bandwidth(self) -> torch.Tensor:
        """Return the absolute bandwidth when representable in Float64.

        Returns
        -------
        torch.Tensor
            Exponentiated log bandwidth. Underflow or overflow may give zero or
            infinity; ``log_bandwidth`` retains the finite scale in those cases.
        """
        return self.log_bandwidth.exp()

    @property
    def num_samples(self) -> int:
        """Return the number of equally weighted pooled draws.

        Returns
        -------
        int
            Product of original chain and draw counts.
        """
        return math.prod(self.sample_shape)


class DensityRatio(ValueModel):
    """Posterior-to-prior density evidence, separate from probability odds.

    Parameters
    ----------
    posterior, prior : DensityOrdinate
        Marginal KDE ordinates at the same original-coordinate null.
    log_ratio : torch.Tensor
        Posterior minus prior log density. This always describes the empirical
        density ratio, independently of any Bayesian interpretation.
    assumptions : PointNullAssumptions or None
        Caller declarations, retained without claiming mechanical verification.
    log_bayes_factor_01 : torch.Tensor or None
        Point-null versus encompassing-alternative log Bayes factor only when
        both declarations are true. KDE approximation error remains; this is
        neither a region-odds ratio nor a posterior model probability.
    """

    posterior: DensityOrdinate = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    prior: DensityOrdinate = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    log_ratio: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    assumptions: PointNullAssumptions | None = None
    log_bayes_factor_01: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


def _draws(samples: torch.Tensor) -> torch.Tensor:
    """Validate floating draws and pool only their two leading sample axes.

    Parameters
    ----------
    samples : torch.Tensor
        Finite floating draws with explicit chain and draw axes.

    Returns
    -------
    torch.Tensor
        Detached Float64 draws with a leading pooled sample axis.

    Raises
    ------
    TypeError, ValueError
        If data are discrete, nonfinite, empty, or have fewer than two draws.
    """
    if not isinstance(samples, torch.Tensor) or not samples.is_floating_point():
        raise TypeError("density ordinates require floating continuous contrast draws")
    values = real_samples(samples, label="density ordinate")
    if values.ndim < 2 or not all(values.shape) or math.prod(values.shape[:2]) < 2:
        raise ValueError("density ordinates require at least two chain/draw samples")
    return values.flatten(0, 1)


def _event_value(
    value: float | int | torch.Tensor, values: torch.Tensor, *, label: str
) -> torch.Tensor:
    """Check a real value and broadcast solely across event coordinates.

    Parameters
    ----------
    value : float, int, or torch.Tensor
        Finite scalar or event-shaped value.
    values : torch.Tensor
        Pooled draws supplying the event shape and device.
    label : str
        Name used in validation errors.

    Returns
    -------
    torch.Tensor
        Float64 value broadcast to the event shape.

    Raises
    ------
    TypeError, ValueError
        If carrier, finiteness, original integer capacity, shape, or device fail.
    """
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            raise TypeError(f"{label} requires a real value, not Boolean")
        if value.device != values.device:
            raise ValueError(f"{label} and draws must use the same device")
        result = real_samples(value, label=label)
    elif isinstance(value, bool):
        raise TypeError(f"{label} requires a real scalar or tensor, not Boolean")
    elif isinstance(value, int):
        if abs(value) > _MAX_EXACT_COUNT:
            raise ValueError(f"{label} integer inputs must lie within +/-2**53")
        result = values.new_tensor(value)
    elif isinstance(value, float):
        result = real_samples(values.new_tensor(value), label=label)
    else:
        raise TypeError(f"{label} requires a real scalar or tensor")
    if result.ndim > values.ndim - 1:
        raise ValueError(f"{label} must broadcast only over event axes")
    try:
        return torch.broadcast_to(result, values.shape[1:])
    except RuntimeError as error:
        raise ValueError(f"{label} must broadcast only over event axes") from error


def _bandwidth(
    values: torch.Tensor, bandwidth: str | float | torch.Tensor, adjust: float
) -> tuple[torch.Tensor, str]:
    """Select a Gaussian bandwidth in scaled coordinates, retaining its logarithm.

    Parameters
    ----------
    values : torch.Tensor
        Validated finite pooled draws.
    bandwidth : str, float, or torch.Tensor
        Automatic rule or positive absolute event bandwidth.
    adjust : float
        Validated positive finite multiplier.

    Returns
    -------
    tuple[torch.Tensor, str]
        Log bandwidth and selected rule name.

    Raises
    ------
    ValueError, TypeError
        If the rule is unknown, a bandwidth is invalid, or a coordinate is constant.
    """
    rule = bandwidth if isinstance(bandwidth, str) else "explicit"
    if not isinstance(bandwidth, str):
        if bool((values.amin(0) == values.amax(0)).any()):
            raise ValueError(
                "density ordinates require nonconstant draws on every coordinate"
            )
        selected = _event_value(bandwidth, values, label="bandwidth")
        if bool((selected <= 0).any()):
            raise ValueError("bandwidth must be strictly positive")
        return selected.log() + math.log(adjust), rule
    if bandwidth not in {"nrd0", "scott"}:
        raise ValueError("bandwidth must be nrd0, scott, or a positive event value")
    normalized, _, scale = scaled_samples(values)
    levels = (0.25, 0.75) if rule == "nrd0" else ()
    _, sd, quantiles = empirical_summary(normalized, levels)
    if bool(((scale == 0) | (sd == 0)).any()):
        raise ValueError(
            "density ordinates require nonconstant draws on every coordinate"
        )
    factor = sd
    if rule == "nrd0":
        iqr_scale = (quantiles[1] - quantiles[0]) / 1.34
        factor = torch.minimum(sd, iqr_scale)
        factor = torch.where(factor > 0, factor, sd) * 0.9
    log_bandwidth = factor.log() + scale.log() - math.log(values.shape[0]) / 5
    return log_bandwidth + math.log(adjust), rule


def _log_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute log absolute differences without overflowing opposite-sign sums.

    Parameters
    ----------
    left, right : torch.Tensor
        Finite, broadcastable Float64 values.

    Returns
    -------
    torch.Tensor
        Log absolute differences, including negative infinity at equality.
    """
    difference = left - right
    return torch.where(
        torch.isfinite(difference),
        difference.abs().log(),
        torch.logaddexp(left.abs().log(), right.abs().log()),
    )


def _linear_distance(
    point: torch.Tensor,
    large: torch.Tensor,
    small: torch.Tensor,
    multiplier: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain a signed affine distance even when intermediate subtraction overflows.

    Parameters
    ----------
    point, large, small : torch.Tensor
        Finite values in original coordinates.
    multiplier : torch.Tensor
        Finite nonnegative factor no greater than two.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Log magnitude and sign of multiplier times (point - large) + large - small.
    """
    direct = multiplier * (point - large) + (large - small)
    scale = torch.maximum(torch.maximum(point.abs(), large.abs()), small.abs())
    divisor = torch.where(scale > 0, scale, 1)
    normalized = multiplier * (point / divisor - large / divisor)
    normalized = normalized + large / divisor - small / divisor
    log_value = torch.where(
        torch.isfinite(direct),
        direct.abs().log(),
        normalized.abs().log() + divisor.log(),
    )
    sign = torch.where(torch.isfinite(direct), direct.sign(), normalized.sign())
    # A zero multiplier cancels the point exactly, even across extreme scales.
    log_value = torch.where(multiplier == 0, _log_distance(large, small), log_value)
    sign = torch.where(multiplier == 0, (large - small).sign(), sign)
    return log_value, sign


def _quadratic_difference(
    point: torch.Tensor,
    small_reference: torch.Tensor,
    large_reference: torch.Tensor,
    small_log_width: torch.Tensor,
    large_log_width: torch.Tensor,
) -> torch.Tensor:
    """Factor the difference between two Gaussian squared standardized distances.

    Parameters
    ----------
    point : torch.Tensor
        Shared finite evaluation point.
    small_reference, large_reference : torch.Tensor
        Finite Gaussian centers, ordered by their bandwidths.
    small_log_width, large_log_width : torch.Tensor
        Log bandwidths with small no greater than large.

    Returns
    -------
    torch.Tensor
        Squared distance for the smaller-bandwidth Gaussian minus that for
        the larger-bandwidth Gaussian. Equal-width tail cancellation retains
        the center difference before multiplying by the large distance.
    """
    log_ratio = small_log_width - large_log_width
    first, first_sign = _linear_distance(
        point, large_reference, small_reference, -torch.expm1(log_ratio)
    )
    second, second_sign = _linear_distance(
        point, large_reference, small_reference, 1 + log_ratio.exp()
    )
    return first_sign * second_sign * (first + second - 2 * small_log_width).exp()


def _kernel_ordinate(
    values: torch.Tensor, at: torch.Tensor, log_bandwidth: torch.Tensor, budget: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce direct Gaussian kernels around nearest retained values in bounded tiles.

    Parameters
    ----------
    values : torch.Tensor
        Finite pooled Float64 samples.
    at, log_bandwidth : torch.Tensor
        Event-shaped evaluation points and log bandwidths.
    budget : int
        Maximum number of elements in each temporary kernel tile.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Log density, local kernel-effective count, retained reference center,
        and log mean kernel weight relative to that center, on event axes.

    Raises
    ------
    ValueError
        If no finite log density is representable at an evaluation coordinate.
    """
    flat = values.reshape(values.shape[0], -1)
    points, widths = at.reshape(-1), log_bandwidth.reshape(-1)
    output, effective = torch.empty_like(points), torch.empty_like(points)
    references, relative_sums = torch.empty_like(points), torch.empty_like(points)
    for start in range(0, points.numel(), budget):
        stop = min(start + budget, points.numel())
        width = stop - start
        rows = max(1, budget // width)
        point, log_width = points[start:stop], widths[start:stop]
        reference = flat[0, start:stop].clone()
        distance = (point - reference).abs()
        for row in range(0, flat.shape[0], rows):
            samples = flat[row : row + rows, start:stop]
            distances = (point - samples).abs()
            closest = distances == distances.amin(0)
            below = torch.where(closest & (samples <= point), samples, -torch.inf).amax(
                0
            )
            above = torch.where(closest & (samples >= point), samples, torch.inf).amin(
                0
            )
            candidate = torch.where(torch.isfinite(below), below, above)
            candidate_distance = (point - candidate).abs()
            # Resolve rounded distant ties by the nearest original sample value.
            same_below = (
                (candidate <= point) & (reference <= point) & (candidate > reference)
            )
            same_above = (
                (candidate >= point) & (reference >= point) & (candidate < reference)
            )
            better = (candidate_distance < distance) | (
                (candidate_distance == distance) & (same_below | same_above)
            )
            reference = torch.where(better, candidate, reference)
            distance = torch.where(better, candidate_distance, distance)
        maximum = points.new_full((width,), -torch.inf)
        first, second = points.new_zeros(width), points.new_zeros(width)
        two = torch.full_like(point, 2)
        for row in range(0, flat.shape[0], rows):
            samples = flat[row : row + rows, start:stop]
            log_delta = _log_distance(reference, samples)
            log_sum, sum_sign = _linear_distance(point, samples, reference, two)
            relative = (
                -0.5
                * (reference - samples).sign()
                * sum_sign
                * (log_delta + log_sum - 2 * log_width).exp()
            )
            next_maximum = torch.maximum(maximum, relative.amax(0))
            safe_maximum = torch.where(torch.isfinite(next_maximum), next_maximum, 0)
            old_scale = (maximum - safe_maximum).exp()
            weights = (relative - safe_maximum).exp()
            first = first * old_scale + weights.sum(0)
            second = second * old_scale.square() + weights.square().sum(0)
            maximum = next_maximum
        relative_sum = maximum + first.log() - math.log(flat.shape[0])
        base = -0.5 * (2 * (_log_distance(point, reference) - log_width)).exp()
        density = base + relative_sum - log_width - 0.5 * math.log(2 * math.pi)
        if not bool(torch.isfinite(density).all()):
            raise ValueError(
                "KDE log density exceeds Float64 capacity; rescale the contrast or bandwidth"
            )
        output[start:stop] = density
        effective[start:stop] = first.square() / second
        references[start:stop], relative_sums[start:stop] = reference, relative_sum
    return (
        output.reshape(at.shape),
        effective.reshape(at.shape),
        references.reshape(at.shape),
        relative_sums.reshape(at.shape),
    )


def _settings(adjust: float, budget: int) -> float:
    """Validate bandwidth adjustment and kernel workspace settings.

    Parameters
    ----------
    adjust : float
        Positive finite bandwidth multiplier.
    budget : int
        Positive integer kernel-tile bound.

    Returns
    -------
    float
        Checked multiplier in a Python floating carrier.

    Raises
    ------
    ValueError, TypeError
        If a setting is Boolean, nonpositive, nonfinite, or has the wrong carrier.
    """
    if isinstance(adjust, bool) or not isinstance(adjust, int | float):
        raise TypeError("adjust must be a positive finite real number")
    try:
        valid = math.isfinite(adjust) and adjust > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("adjust must be a positive finite real number")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("max_working_elements must be a positive integer")
    return float(adjust)


def _ordinate(
    values: torch.Tensor,
    sample_shape: tuple[int, ...],
    *,
    at: float | int | torch.Tensor,
    bandwidth: str | float | torch.Tensor,
    adjust: float,
    name: str,
    dims: Sequence[str] | None,
    coords: Mapping[str, Sequence[object]] | None,
    budget: int,
) -> tuple[DensityOrdinate, tuple[torch.Tensor, torch.Tensor]]:
    """Build an ordinate from validated pooled draws and shared metadata.

    Parameters
    ----------
    values : torch.Tensor
        Finite pooled draws.
    sample_shape : tuple[int, ...]
        Original chain and draw counts.
    at : float, int, or torch.Tensor
        Event-shaped evaluation point.
    bandwidth : str, float, or torch.Tensor
        Automatic rule or absolute Gaussian bandwidth.
    adjust : float
        Checked positive bandwidth multiplier.
    name : str
        Nonempty original contrast name.
    dims, coords : Sequence or Mapping or None
        Shared event dimension names and original coordinate labels.
    budget : int
        Checked positive kernel workspace bound.

    Returns
    -------
    tuple[DensityOrdinate, tuple[torch.Tensor, torch.Tensor]]
        Marginal ordinate and private reference/relative-sum terms for stable
        posterior-to-prior tail comparisons.

    Raises
    ------
    ValueError
        If the contrast name or coordinate metadata is invalid.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("contrast name must be a nonempty string")
    point = _event_value(at, values, label="evaluation point")
    names = event_dimensions(
        {name: values.shape[1:]},
        dims={name: dims} if dims is not None else None,
        coords=coords,
    )[name]
    log_width, rule = _bandwidth(values, bandwidth, adjust)
    log_density, effective, reference, relative_sum = _kernel_ordinate(
        values, point, log_width, budget
    )
    return DensityOrdinate(
        log_density=log_density,
        at=point,
        log_bandwidth=log_width,
        bandwidth_rule=rule,
        adjust=adjust,
        sample_shape=sample_shape,
        kernel_effective_count=effective,
        name=name,
        dimension_names=names,
        coordinate_labels=tuple(
            axis_labels(dimension, extent, coords)
            for dimension, extent in zip(names, values.shape[1:], strict=True)
        ),
        max_working_elements=budget,
    ), (reference, relative_sum)


def log_density_ordinate(
    samples: torch.Tensor,
    *,
    at: float | int | torch.Tensor = 0.0,
    bandwidth: str | float | torch.Tensor = "nrd0",
    adjust: float = 1.0,
    name: str = "contrast",
    dims: Sequence[str] | None = None,
    coords: Mapping[str, Sequence[object]] | None = None,
    max_working_elements: int = 262144,
) -> DensityOrdinate:
    """Estimate one marginal Gaussian log density per retained event coordinate.

    Parameters
    ----------
    samples : torch.Tensor
        Finite floating draws shaped ``(chain, draw, *event)``. At least two
        pooled draws and nonzero spread on every coordinate are required.
    at : float, int, or torch.Tensor
        Finite evaluation point, broadcasting only over event axes.
    bandwidth : str, float, or torch.Tensor
        ``nrd0`` selects ``0.9 * min(SD, IQR/1.34) * N**(-1/5)``, falling back
        to SD when IQR is zero. ``scott`` selects ``SD * N**(-1/5)``.
        A positive scalar or event tensor selects an absolute bandwidth.
    adjust : float
        Positive finite multiplier applied to the selected bandwidth.
    name : str
        Original contrast name, retained without parsing.
    dims : Sequence[str] or None
        Event dimension names using the retained-analysis coordinate policy.
    coords : Mapping[str, Sequence[object]] or None
        Original labels of event dimensions.
    max_working_elements : int
        Positive integer bound on temporary kernel tiles, including both sample
        and event-coordinate tiling. Automatic bandwidth reduction uses the
        shared input-sized summary workspace; nrd0 also sorts for quartiles.

    Returns
    -------
    DensityOrdinate
        Detached Float64 marginal log density, bandwidth, sample counts, and
        local kernel-weight concentration on the original device. This is a
        smoothing estimate, not a jointly multivariate density or evidence claim.

    Raises
    ------
    TypeError, ValueError
        If samples are integer/Boolean, nonfinite, empty, constant, or too few;
        metadata or settings disagree; or log-score capacity is exceeded.

    Notes
    -----
    The direct Gaussian sum has no FFT interpolation or negative-density clipping.
    Finite-sample KDE error depends on bandwidth, tails, dependence, and coordinate
    choice. Floating storage does not establish a continuous underlying law.
    No inference, density gradient, or convergence diagnostic is computed.
    """
    values = _draws(samples)
    multiplier = _settings(adjust, max_working_elements)
    return _ordinate(
        values,
        tuple(samples.shape[:2]),
        at=at,
        bandwidth=bandwidth,
        adjust=multiplier,
        name=name,
        dims=dims,
        coords=coords,
        budget=max_working_elements,
    )[0]


def density_ratio(
    posterior: torch.Tensor,
    prior: torch.Tensor,
    *,
    null: float | int | torch.Tensor = 0.0,
    bandwidth: str | float | torch.Tensor = "nrd0",
    prior_bandwidth: str | float | torch.Tensor | None = None,
    adjust: float = 1.0,
    assumptions: PointNullAssumptions | None = None,
    name: str = "contrast",
    dims: Sequence[str] | None = None,
    coords: Mapping[str, Sequence[object]] | None = None,
    max_working_elements: int = 262144,
) -> DensityRatio:
    """Compare posterior and joint-prior ordinates at the same scalar contrast null.

    Parameters
    ----------
    posterior, prior : torch.Tensor
        Finite floating contrast draws with aligned event shapes and devices.
        Chain and draw counts may differ. Form derived prior contrasts jointly,
        preserving hierarchical dependence; draws are never permuted here.
    null : float, int, or torch.Tensor
        Finite scalar or event-shaped evaluation point.
    bandwidth : str, float, or torch.Tensor
        Posterior bandwidth rule or positive absolute value.
    prior_bandwidth : str, float, torch.Tensor, or None
        Prior bandwidth; None applies the same policy separately to each input.
        Automatic rules use each distribution's own count and scale.
    adjust : float
        Positive multiplier applied to both selected bandwidths.
    assumptions : PointNullAssumptions or None
        Explicit declarations enabling a point-null BF01 interpretation when
        both are true. These declarations cannot be verified from sample arrays.
    name : str
        Original derived-contrast label.
    dims, coords : Sequence or Mapping or None
        Shared event dimensions and original labels.
    max_working_elements : int
        Positive kernel-tile element bound, as for log_density_ordinate.

    Returns
    -------
    DensityRatio
        Posterior/prior log density ratio and separate declaration-dependent
        BF01 field. It never supplies a posterior null-model probability.

    Raises
    ------
    TypeError, ValueError
        If alignment, domains, samples, settings, or declarations are invalid.
        Known integer or Boolean laws and constant samples are refused.

    Notes
    -----
    Under the declared proper continuous prior and compatible null nuisance
    prior, posterior density divided by prior density gives the Savage-Dickey
    point-null BF01. KDE only approximates those densities. An atom at the null,
    a singular contrast, or incompatible nuisance prior invalidates this argument;
    an empirical equality frequency or region odds cannot replace an ordinate.
    """
    post_values, prior_values = _draws(posterior), _draws(prior)
    if post_values.shape[1:] != prior_values.shape[1:]:
        raise ValueError("posterior and prior require the same event shape")
    if post_values.device != prior_values.device:
        raise ValueError("posterior and prior must use the same device")
    if assumptions is not None and not isinstance(assumptions, PointNullAssumptions):
        raise TypeError("assumptions must be PointNullAssumptions or None")
    multiplier = _settings(adjust, max_working_elements)
    post, post_terms = _ordinate(
        post_values,
        tuple(posterior.shape[:2]),
        at=null,
        bandwidth=bandwidth,
        adjust=multiplier,
        name=name,
        dims=dims,
        coords=coords,
        budget=max_working_elements,
    )
    baseline, prior_terms = _ordinate(
        prior_values,
        tuple(prior.shape[:2]),
        at=null,
        bandwidth=bandwidth if prior_bandwidth is None else prior_bandwidth,
        adjust=multiplier,
        name=name,
        dims=dims,
        coords=coords,
        budget=max_working_elements,
    )
    smaller = post.log_bandwidth <= baseline.log_bandwidth
    small_reference = torch.where(smaller, post_terms[0], prior_terms[0])
    large_reference = torch.where(smaller, prior_terms[0], post_terms[0])
    small_width = torch.minimum(post.log_bandwidth, baseline.log_bandwidth)
    large_width = torch.maximum(post.log_bandwidth, baseline.log_bandwidth)
    quadratic = _quadratic_difference(
        post.at, small_reference, large_reference, small_width, large_width
    )
    base_difference = torch.where(smaller, -0.5 * quadratic, 0.5 * quadratic)
    ratio = base_difference + (post_terms[1] - prior_terms[1])
    ratio = ratio + (baseline.log_bandwidth - post.log_bandwidth)
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("KDE log density ratio exceeds Float64 capacity")
    enabled = (
        assumptions is not None
        and assumptions.proper_continuous_prior
        and assumptions.compatible_null_prior
    )
    return DensityRatio(
        posterior=post,
        prior=baseline,
        log_ratio=ratio,
        assumptions=assumptions,
        log_bayes_factor_01=ratio if enabled else None,
    )


__all__ = [
    "PointNullAssumptions",
    "PointNullSpec",
    "DensityOrdinate",
    "DensityRatio",
    "log_density_ordinate",
    "density_ratio",
]
