"""Validate named initial values once and map them to latent coordinates."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.distributions import constraints

from quivers.inference.registry import LatentRegistry


def unconstrained_initial_values(
    registry: LatentRegistry,
    values: Mapping[str, torch.Tensor | float],
) -> dict[str, torch.Tensor]:
    """Validate support and shape before inverting each latent bijector.

    Scalar initial values broadcast over a site's declared shape. Array
    values must match that shape exactly; row-shaped observation data are
    never interpreted as extra latent batches. The result owns its data.
    """
    unknown = set(values) - set(registry.names)
    if unknown:
        raise ValueError(f"initial_values name unknown latent sites: {sorted(unknown)}")
    result: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        tensor = torch.as_tensor(value)
        if tensor.dtype == torch.bool or tensor.is_complex():
            raise ValueError(f"initial value for {name!r} must be real and non-Boolean")
        tensor = tensor.detach().to(device="cpu", dtype=torch.get_default_dtype())
        site = registry[name]
        if tensor.ndim != 0 and tuple(tensor.shape) != site.constrained_shape:
            raise ValueError(
                f"initial value for {name!r} has shape {tuple(tensor.shape)}; "
                f"expected scalar or {site.constrained_shape}"
            )
        tensor = tensor.expand(site.constrained_shape)
        if not bool(torch.isfinite(tensor).all()) or not bool(
            site.support.check(tensor).all()
        ):
            raise ValueError(
                f"initial value for {name!r} lies outside its finite support"
            )
        constraint = site.support
        while hasattr(constraint, "base_constraint"):
            constraint = getattr(constraint, "base_constraint")
        lower = getattr(constraint, "lower_bound", -torch.inf)
        upper = getattr(constraint, "upper_bound", torch.inf)
        if not bool(((tensor > lower) & (tensor < upper)).all()) or (
            constraint is constraints.simplex and not bool((tensor > 0).all())
        ):
            raise ValueError(
                f"initial value for {name!r} must lie in the interior of its support"
            )
        inverted = site.bijector.inv(tensor)
        assert isinstance(inverted, torch.Tensor)
        unconstrained = inverted.reshape(site.unconstrained_shape)
        if not bool(torch.isfinite(unconstrained).all()):
            raise ValueError(
                f"initial value for {name!r} must lie in the interior of its support"
            )
        result[name] = unconstrained.clone()
    return result
