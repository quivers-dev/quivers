"""Derivatives by three independent routes.

`autograd` differentiates an oracle's own sum. `richardson` takes
central differences at two steps in float64 and extrapolates, with an
estimate of its error. `BOUNDARY_CASES` lists points where differences
do not apply (exact zeros, ties, empty sums) with derivatives worked out
by hand.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import didactic.api as dx
import torch


def autograd(
    value: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
    point: Mapping[str, float],
) -> dict[str, float]:
    """Differentiate a scalar function of named float64 scalars.

    Parameters
    ----------
    value : Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
        The function.
    point : Mapping[str, float]
        Where to differentiate.

    Returns
    -------
    dict[str, float]
        The derivative in each name; a name the function ignores has
        derivative zero.
    """
    leaves = {
        name: torch.tensor(number, dtype=torch.float64, requires_grad=True)
        for name, number in point.items()
    }
    answer = value(leaves)
    found = torch.autograd.grad(
        answer, list(leaves.values()), allow_unused=True, materialize_grads=True
    )
    return {name: float(grad) for name, grad in zip(leaves, found, strict=True)}


class Difference(dx.Model):
    """A derivative from differences and an estimate of its error.

    Parameters
    ----------
    value : float
        The derivative.
    error : float
        An estimate of its absolute error: the change made by the
        extrapolation plus the rounding of the two differences.
    """

    value: float
    error: float


def richardson(
    value: Callable[[Mapping[str, float]], float],
    point: Mapping[str, float],
    name: str,
    *,
    step: float,
) -> Difference:
    """Differentiate by extrapolated central differences.

    Parameters
    ----------
    value : Callable[[Mapping[str, float]], float]
        The function, in float64.
    point : Mapping[str, float]
        Where to differentiate, in the interior of the domain.
    name : str
        The coordinate.
    step : float
        The larger step; the other step is half of it.

    Returns
    -------
    Difference
        The derivative and its error estimate.
    """

    def central(width: float) -> tuple[float, float]:
        """One central difference.

        Parameters
        ----------
        width : float
            The step.

        Returns
        -------
        tuple of (float, float)
            The difference quotient and the size of the values used.
        """
        above = value({**point, name: point[name] + width})
        below = value({**point, name: point[name] - width})
        return (above - below) / (2 * width), max(abs(above), abs(below))

    coarse, size = central(step)
    fine, _ = central(step / 2)
    extrapolated = (4 * fine - coarse) / 3
    rounding = 4 * torch.finfo(torch.float64).eps * size / step
    return Difference(
        value=extrapolated, error=abs(extrapolated - fine) + rounding
    )


class BoundaryCase(dx.Model):
    """A point where the derivative is known by hand.

    Parameters
    ----------
    name : str
        The case.
    reason : str
        How the derivative is obtained.
    value : float
        The value of the function.
    gradient : tuple of float
        The derivative in each argument.
    """

    name: str
    reason: str
    value: float
    gradient: tuple[float, ...]


BOUNDARY_CASES: Sequence[BoundaryCase] = (
    BoundaryCase(
        name="path_count_at_a_zero_edge",
        reason=(
            "the weight of the walks from 0 to 1 in [[a, b], [0, 0]] is "
            "b / (1 - a), so at a = b = 0 the derivatives are 0 and 1"
        ),
        value=0.0,
        gradient=(0.0, 1.0),
    ),
    BoundaryCase(
        name="sum_with_a_zero_summand",
        reason="3 (1 + 2 t) has derivative 6 at t = 0",
        value=3.0,
        gradient=(6.0,),
    ),
    BoundaryCase(
        name="impossible_sum",
        reason=(
            "the log of a sum of absent alternatives is minus infinity in a "
            "neighbourhood of the absent support, so no mass moves and the "
            "declared derivative is zero"
        ),
        value=float("-inf"),
        gradient=(0.0, 0.0),
    ),
    BoundaryCase(
        name="tie",
        reason=(
            "the derivative of max at a tie is any point of the segment from "
            "(1, 0) to (0, 1); one operation has to return one element for "
            "every way of writing the maximum"
        ),
        value=1.0,
        gradient=(1.0, 0.0),
    ),
)
"""The points where differences do not apply."""
