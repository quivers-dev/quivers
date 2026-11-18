"""Validate named initial values once and map them to latent coordinates."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.distributions import constraints

from quivers.inference.registry import LatentRegistry


def natural_initial_value(
    name: str,
    value: torch.Tensor | float,
    shape: tuple[int, ...],
    support: constraints.Constraint | None,
) -> torch.Tensor:
    """Validate one site's natural-scale initial value against its shape
    and support.

    A scalar broadcasts over the site's shape; an array must match that
    shape exactly, so row-shaped observation data are never interpreted
    as extra latent batches.

    Parameters
    ----------
    name : str
        The site, named in errors.
    value : torch.Tensor | float
        The value on the site's natural scale.
    shape : tuple[int, ...]
        The shape of one draw of the site.
    support : Constraint | None
        The site's support; ``None`` for a site whose support depends on
        other sites, which is then checked only for finiteness.

    Returns
    -------
    torch.Tensor
        The value at ``shape``, on the CPU in the default floating
        dtype, detached from any graph.

    Raises
    ------
    ValueError
        If the value is Boolean or complex, has another shape, is not
        finite, or lies outside the support.
    """
    tensor = torch.as_tensor(value)
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise ValueError(f"initial value for {name!r} must be real and non-Boolean")
    tensor = tensor.detach().to(device="cpu", dtype=torch.get_default_dtype())
    if tensor.ndim != 0 and tuple(tensor.shape) != shape:
        raise ValueError(
            f"initial value for {name!r} has shape {tuple(tensor.shape)}; "
            f"expected scalar or {shape}"
        )
    tensor = tensor.expand(shape)
    if not bool(torch.isfinite(tensor).all()) or (
        support is not None and not bool(support.check(tensor).all())
    ):
        raise ValueError(f"initial value for {name!r} lies outside its finite support")
    return tensor


def unconstrained_initial_values(
    registry: LatentRegistry,
    values: Mapping[str, torch.Tensor | float],
) -> dict[str, torch.Tensor]:
    """Validate support and shape before inverting each latent bijector.

    Each value is checked by `natural_initial_value` and must also lie
    strictly inside its support, where the site's bijector has a finite
    inverse.

    Parameters
    ----------
    registry : LatentRegistry
        The registry whose sites the values name.
    values : Mapping[str, torch.Tensor | float]
        Natural-scale values keyed by site.

    Returns
    -------
    dict[str, torch.Tensor]
        Each site's unconstrained value; the result owns its data.

    Raises
    ------
    ValueError
        If a name is not a site of the registry, or a value has another
        shape, is not finite, or is not in the interior of its support.
    """
    unknown = set(values) - set(registry.names)
    if unknown:
        raise ValueError(f"initial_values name unknown latent sites: {sorted(unknown)}")
    result: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        site = registry[name]
        tensor = natural_initial_value(
            name, value, site.constrained_shape, site.support
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
        unconstrained, _ = site.unconstrain(tensor)
        if not bool(torch.isfinite(unconstrained).all()):
            raise ValueError(
                f"initial value for {name!r} must lie in the interior of its support"
            )
        result[name] = unconstrained.clone()
    return result
