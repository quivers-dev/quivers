"""Atom carriers and safe densities for scalar distribution compositions."""

from __future__ import annotations

from functools import lru_cache
from typing import cast

import torch
from torch import Tensor
from torch.distributions import (
    Distribution,
    Independent,
    MixtureSameFamily,
    TransformedDistribution,
    constraints,
)


def _family_metadata(distribution):
    """Find the registered native and semantic records for an exact class.

    Parameters
    ----------
    distribution : Distribution
        Native distribution instance.

    Returns
    -------
    tuple or None
        Native and semantic records, or None for an unregistered class.
    """
    from quivers.continuous.family_spec import FAMILY_REGISTRY

    return _class_metadata(type(distribution), len(FAMILY_REGISTRY))


@lru_cache(maxsize=256)
def _class_metadata(distribution_type, registry_size):
    """Cache catalog metadata without inspecting distribution parameter shapes.

    Parameters
    ----------
    distribution_type : type
        Exact native distribution class.
    registry_size : int
        Catalog generation, invalidating misses after registration.

    Returns
    -------
    tuple or None
        Native and semantic records when registered.
    """
    from quivers.continuous.family_spec import FAMILY_REGISTRY
    from quivers.qiec.families import FAMILIES

    # Catalog adapters may use these constructor classes for a specific law.
    # The generic classes must instead derive semantics from their components.
    if distribution_type in {TransformedDistribution, Independent, MixtureSameFamily}:
        return None

    # MixtureSameFamily uses torch's categorical constructor internally;
    # its parameter/event contract is exactly the canonical categorical one.
    if distribution_type is torch.distributions.Categorical:
        return None, FAMILIES["Categorical"]
    for spec in FAMILY_REGISTRY.values():
        if spec.dist_class is distribution_type and spec.name in FAMILIES:
            return spec, FAMILIES[spec.name]
    name = getattr(distribution_type, "_semantic_family", None)
    return (
        (None, FAMILIES[name]) if isinstance(name, str) and name in FAMILIES else None
    )


def _point(distribution, value, *, reject_nan=True):
    """Broadcast a query against native parameters without rounding integers.

    Parameters
    ----------
    distribution : Distribution
        Distribution retaining its batch and event dimensions.
    value : Tensor or float
        Query points.
    reject_nan : bool
        Reject invalid atom-query endpoints. Density evaluation instead masks
        NaNs as values outside the declared support.

    Returns
    -------
    Tensor
        Broadcast scalar query tensor.

    Raises
    ------
    ValueError
        If a query contains NaN.
    """
    from quivers.continuous._interval import _reference

    ref = _reference(distribution)
    point = torch.as_tensor(
        value,
        dtype=(
            torch.long
            if isinstance(value, int) and distribution.support.is_discrete
            else ref.dtype
            if isinstance(value, int | float)
            else None
        ),
        device=ref.device,
    )
    if not point.is_floating_point():
        point = point.to(torch.long if distribution.support.is_discrete else ref.dtype)
    if reject_nan and bool(torch.isnan(point).any()):
        raise ValueError("atom queries cannot contain NaN")
    return point.expand(
        torch.broadcast_shapes(
            point.shape, distribution.batch_shape + distribution.event_shape
        )
    )


def _atomless(distribution):
    """Read an atomless proof from family metadata or a custom class capability.

    Parameters
    ----------
    distribution : Distribution
        Noncomposite native distribution.

    Returns
    -------
    bool
        Whether the family is declared atomless.
    """
    metadata = _family_metadata(distribution)
    return bool(getattr(distribution, "_atomless", False)) or (
        metadata is not None and metadata[1].effective_atomicity == "atomless"
    )


def atomicity(distribution: Distribution) -> str:
    """Prove whether a native law is atomless, discrete, or mixed.

    Parameters
    ----------
    distribution : Distribution
        Registered native law or composition with an explicit capability.

    Returns
    -------
    str
        Homogeneous atomless/discrete classification, or mixed.

    Raises
    ------
    NotImplementedError
        If a custom law or nonbijective transform has no declared proof.
    """
    declared = getattr(distribution, "_atomicity", None)
    if declared in {"atomless", "discrete", "mixed"}:
        return declared
    if isinstance(distribution, Independent):
        return atomicity(distribution.base_dist)
    if isinstance(distribution, MixtureSameFamily):
        return atomicity(distribution.component_distribution)
    if isinstance(distribution, TransformedDistribution):
        if _atomless(distribution):
            return "atomless"
        if all(transform.bijective for transform in distribution.transforms):
            return atomicity(distribution.base_dist)
        raise NotImplementedError("nonbijective transforms require an atomicity proof")
    if cast(constraints.Constraint, distribution.support).is_discrete:
        return "discrete"
    metadata = _family_metadata(distribution)
    if metadata is not None:
        result = metadata[1].effective_atomicity
        if result != "composite":
            return result
    if getattr(distribution, "_atomless", False):
        return "atomless"
    raise NotImplementedError(f"{type(distribution).__name__} has no atomicity proof")


def atom_reference(distribution: Distribution, value: Tensor) -> Tensor:
    """Identify the potential atomic carrier independently of atom weights.

    Parameters
    ----------
    distribution : Distribution
        Native distribution or raw measure.
    value : Tensor
        Query with sample, batch, and event dimensions.

    Returns
    -------
    Tensor
        Boolean carrier membership, reduced over intrinsic event axes.

    Raises
    ------
    NotImplementedError
        If an unknown distribution declares no atom capability.
    """
    method = getattr(distribution, "atom_reference", None)
    if callable(method):
        return cast(Tensor, method(value))
    if isinstance(distribution, TransformedDistribution) and not _atomless(
        distribution
    ):
        point = value
        for transform in reversed(distribution.transforms):
            if not transform.bijective:
                raise NotImplementedError("atom carriers require bijective transforms")
            point = cast(Tensor, transform.inv(point))
        return atom_reference(distribution.base_dist, point)
    if isinstance(distribution, Independent):
        carrier = atom_reference(distribution.base_dist, value)
        dims = distribution.reinterpreted_batch_ndims
        return carrier.all(tuple(range(-dims, 0))) if dims else carrier
    if isinstance(distribution, MixtureSameFamily):
        return atom_reference(
            distribution.component_distribution, distribution._pad(value)
        ).any(-1)
    if cast(constraints.Constraint, distribution.support).is_discrete:
        if distribution.event_shape:
            return cast(constraints.Constraint, distribution.support).check(value)
        metadata = _family_metadata(distribution)
        from quivers.qiec.types import BOOL

        if metadata is not None and metadata[1].element == BOOL:
            return (value == 0) | (value == 1)
        return torch.isfinite(value) & (value == value.floor())
    metadata = _family_metadata(distribution)
    if metadata is not None and metadata[1].atom_locations:
        point = _point(distribution, value)
        carrier = torch.zeros_like(point, dtype=torch.bool)
        for location in metadata[1].atom_locations:
            carrier = carrier | (point == location)
        return carrier
    if _atomless(distribution):
        shape = (
            value.shape[: -len(distribution.event_shape)]
            if distribution.event_shape
            else value.shape
        )
        return torch.zeros(
            torch.broadcast_shapes(shape, distribution.batch_shape),
            dtype=torch.bool,
            device=value.device,
        )
    raise NotImplementedError(
        f"{type(distribution).__name__} has no declared atom carrier"
    )


def log_atom_mass(distribution: Distribution, value: Tensor) -> Tensor:
    """Evaluate raw mass at a point without a continuous-density Jacobian.

    Parameters
    ----------
    distribution : Distribution
        Native distribution or raw measure.
    value : Tensor
        Query with sample, batch, and event dimensions.

    Returns
    -------
    Tensor
        Log atom masses, with negative infinity for atomless locations.

    Raises
    ------
    NotImplementedError
        If no atomicity capability is available.
    """
    method = getattr(distribution, "log_atom_mass", None)
    if callable(method):
        return cast(Tensor, method(value))
    if isinstance(distribution, TransformedDistribution) and not _atomless(
        distribution
    ):
        point = value
        for transform in reversed(distribution.transforms):
            if not transform.bijective:
                raise NotImplementedError("atom masses require bijective transforms")
            point = cast(Tensor, transform.inv(point))
        return log_atom_mass(distribution.base_dist, point)
    if isinstance(distribution, Independent):
        score = log_atom_mass(distribution.base_dist, value)
        dims = distribution.reinterpreted_batch_ndims
        return score.sum(tuple(range(-dims, 0))) if dims else score
    if isinstance(distribution, MixtureSameFamily):
        score = log_atom_mass(
            distribution.component_distribution, distribution._pad(value)
        )
        return torch.logsumexp(score + distribution.mixture_distribution.logits, -1)
    if cast(constraints.Constraint, distribution.support).is_discrete:
        return safe_log_prob(distribution, value)
    metadata = _family_metadata(distribution)
    if metadata is not None and metadata[1].atom_locations:
        carrier = atom_reference(distribution, value)
        return safe_log_prob(distribution, value, active=carrier)
    if _atomless(distribution):
        carrier = atom_reference(distribution, value)
        from quivers.continuous._interval import _reference

        ref = _reference(distribution)
        return torch.full(
            carrier.shape, -torch.inf, dtype=ref.dtype, device=carrier.device
        )
    raise NotImplementedError(
        f"{type(distribution).__name__} has no declared atom mass"
    )


def safe_log_prob(
    distribution: Distribution, value: Tensor, *, active: Tensor | None = None
) -> Tensor:
    """Mask unsupported observations before invoking native validation.

    Parameters
    ----------
    distribution : Distribution
        Native distribution with a declared support and feasible scoring point.
    value : Tensor
        Query points, including observations outside one mixture component.
    active : Tensor or None
        Boolean scalar-score mask. Inactive observations are substituted before
        native scoring, avoiding invalid density gradients at atom boundaries.

    Returns
    -------
    Tensor
        Native scores within support and negative infinity outside support.

    Raises
    ------
    NotImplementedError
        If the support has no deterministic feasible-point construction.
    """
    from quivers.continuous._censored import _interior_point

    point = _point(distribution, value, reject_nan=False)
    valid = cast(constraints.Constraint, distribution.support).check(point)
    if active is not None:
        valid = valid & active
    if isinstance(distribution, Independent):
        dimensions = distribution.reinterpreted_batch_ndims
        if dimensions == 0:
            return safe_log_prob(distribution.base_dist, point, active=valid)
        coordinate_active = valid.reshape(valid.shape + (1,) * dimensions)
        scores = safe_log_prob(distribution.base_dist, point, active=coordinate_active)
        return scores.sum(tuple(range(scores.ndim - dimensions, scores.ndim)))
    safe = _interior_point(distribution, point)
    point_valid = valid.reshape(valid.shape + (1,) * len(distribution.event_shape))
    if distribution.event_shape:
        # Some coupled support checks first reject an invalid structural
        # property for the whole batch. Recheck after feasible substitution
        # so another row cannot evade its remaining support conditions.
        candidate = torch.where(point_valid, point, safe)
        valid = valid & cast(constraints.Constraint, distribution.support).check(
            candidate
        )
        point_valid = valid.reshape(valid.shape + (1,) * len(distribution.event_shape))
    return torch.where(
        valid, distribution.log_prob(torch.where(point_valid, point, safe)), -torch.inf
    )
