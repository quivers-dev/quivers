"""Mass queries on existing native distributions and unnormalized measures."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution


def log_mass(distribution: Distribution, *, batch_axes: int = 0) -> Tensor:
    """Read a measure normalizer, summing only unretained replication axes.

    Parameters
    ----------
    distribution : torch.distributions.Distribution
        Raw native measure; ordinary probability distributions have mass one.
    batch_axes : int
        Leading independent batch axes retained in the answer.

    Returns
    -------
    torch.Tensor
        Log mass on the requested batch axes, with every event axis reduced.

    Raises
    ------
    TypeError
        If the argument is not a native distribution.
    ValueError
        If retained axes exceed the distribution's batch rank.
    """
    if not isinstance(distribution, Distribution):
        raise TypeError("log_mass requires a native distribution value")
    if (
        isinstance(batch_axes, bool)
        or not isinstance(batch_axes, int)
        or not 0 <= batch_axes <= len(distribution.batch_shape)
    ):
        raise ValueError("log_mass retained axes must be within the batch rank")
    normalizer = getattr(distribution, "log_normalizer", None)
    if callable(normalizer):
        mass = torch.as_tensor(normalizer())
    else:
        # Stored parameters preserve zero derivatives without forcing cached
        # transformations such as log(0) for categorical probability zeros.
        parameters = [
            value
            for name in distribution.arg_constraints
            if isinstance(value := distribution.__dict__.get(name), Tensor)
        ]
        if parameters:
            anchor = sum(
                torch.where(torch.isfinite(point), point, 0) * 0
                for value in parameters
                for point in (
                    value[(0,) * value.ndim] if value.numel() else value.sum(),
                )
            )
            mass = torch.as_tensor(anchor).expand(distribution.batch_shape)
        else:
            mass = torch.zeros(distribution.batch_shape)
    if mass.ndim < len(distribution.batch_shape):
        mass = mass.expand(distribution.batch_shape)
    if mass.ndim > batch_axes:
        mass = mass.sum(tuple(range(batch_axes, mass.ndim)))
    return mass
