"""The one comparison rule between a computed value and a reference.

A tolerance is computed from the number of rounded operations, the
precision, and the condition number of the query, before the run. With
unit roundoff ``u`` and ``n`` operations the factor is Higham's
``gamma(n) = n u / (1 - n u)`` (Accuracy and Stability of Numerical
Algorithms, second edition, lemma 3.1).

In the linear domain a sum of products of nonnegative numbers has
relative error at most ``gamma(n)`` for ``n`` roundings on the longest
path from an input to the result. In the logarithmic domain the same
bound holds for the absolute error of the log, with every operation
scaled by the largest magnitude of a logarithm that occurs, because the
rounding of a logarithm is relative to its size. An elementary function
counts as four roundings: the vector libraries in use state an error of
at most two units in the last place, which is four unit roundoffs.

Infinities compare exactly. A value that is not a number equals nothing.
Shapes, types, and devices compare exactly. A missing gradient is a
difference even where the reference gradient is zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction

import didactic.api as dx
import torch

ELEMENTARY = 4
"""Roundings counted for one elementary function."""

ROUNDOFF: dict[torch.dtype, float] = {
    torch.float32: 2.0**-24,
    torch.float64: 2.0**-53,
}
"""The unit roundoff of each precision."""


class Mismatch(AssertionError):
    """A computed value differs from its reference.

    Parameters
    ----------
    reason : str
        ``"value"``, ``"not-a-number"``, ``"infinite"``, ``"shape"``,
        ``"dtype"``, ``"device"``, ``"missing"``, ``"support"``,
        ``"refusal"``, or ``"work"``.
    deviation : float
        The size of the difference; infinite for a structural one.
    detail : str
        What differs.
    """

    def __init__(self, reason: str, deviation: float, detail: str) -> None:
        super().__init__(f"{reason}: {detail} (deviation {deviation:.6g})")
        self.reason = reason
        self.deviation = deviation


def gamma(operations: int, roundoff: float) -> float:
    """Higham's accumulated rounding factor.

    Parameters
    ----------
    operations : int
        The number of roundings.
    roundoff : float
        The unit roundoff.

    Returns
    -------
    float
        ``n u / (1 - n u)``.

    Raises
    ------
    ValueError
        If the factor is not defined because ``n u`` is at least one.
    """
    product = operations * roundoff
    if product >= 1:
        raise ValueError("too many operations for the precision")
    return product / (1 - product)


class Accuracy(dx.Model):
    """The accuracy a run has to reach.

    Parameters
    ----------
    domain : str
        ``"linear"`` for a relative error, ``"log"`` for the absolute
        error of a logarithm.
    operations : int
        Roundings on the longest path from an input to the result.
    magnitude : float
        The largest magnitude of a logarithm that occurs, at least one;
        one in the linear domain.
    condition : float
        The condition number of the query, at least one.
    dtype : str
        ``"float64"`` or ``"float32"``.
    """

    domain: str
    operations: int
    magnitude: float = 1.0
    condition: float = 1.0
    dtype: str = "float64"

    @property
    def torch_dtype(self) -> torch.dtype:
        """The precision as a tensor type.

        Returns
        -------
        torch.dtype
            The type.
        """
        return {"float64": torch.float64, "float32": torch.float32}[self.dtype]

    def tolerance(self, reference: float) -> float:
        """The allowed absolute difference from a reference.

        Parameters
        ----------
        reference : float
            The reference value.

        Returns
        -------
        float
            The tolerance.
        """
        factor = gamma(self.operations, ROUNDOFF[self.torch_dtype]) * self.condition
        if self.domain == "log":
            return factor * max(1.0, self.magnitude)
        return factor * abs(reference)


def compare_number(
    value: float, reference: float | Fraction, accuracy: Accuracy, what: str
) -> None:
    """Compare one number with its reference.

    Parameters
    ----------
    value : float
        The computed number.
    reference : float or Fraction
        The reference.
    accuracy : Accuracy
        The accuracy to reach.
    what : str
        A description for the report.

    Raises
    ------
    Mismatch
        If the two differ by more than the tolerance.
    """
    expected = float(reference)
    if math.isnan(value) or math.isnan(expected):
        raise Mismatch("not-a-number", math.inf, f"{what}: {value} for {expected}")
    if math.isinf(value) or math.isinf(expected):
        if value != expected:
            raise Mismatch("infinite", math.inf, f"{what}: {value} for {expected}")
        return
    deviation = abs(value - expected)
    if deviation > accuracy.tolerance(expected):
        raise Mismatch("value", deviation, f"{what}: {value!r} for {expected!r}")


def compare_tensor(
    value: torch.Tensor,
    reference: torch.Tensor,
    accuracy: Accuracy,
    what: str,
) -> None:
    """Compare a tensor with its reference, entry by entry.

    Parameters
    ----------
    value : torch.Tensor
        The computed tensor.
    reference : torch.Tensor
        The reference, of the shape, type, and device required.
    accuracy : Accuracy
        The accuracy to reach.
    what : str
        A description for the report.

    Raises
    ------
    Mismatch
        If the two differ in shape, type, device, or by more than the
        tolerance in some entry.
    """
    if tuple(value.shape) != tuple(reference.shape):
        raise Mismatch(
            "shape",
            math.inf,
            f"{what}: {tuple(value.shape)} for {tuple(reference.shape)}",
        )
    if value.dtype != reference.dtype:
        raise Mismatch(
            "dtype", math.inf, f"{what}: {value.dtype} for {reference.dtype}"
        )
    if value.device != reference.device:
        raise Mismatch(
            "device", math.inf, f"{what}: {value.device} for {reference.device}"
        )
    flat = value.detach().reshape(-1).tolist()
    expected = reference.detach().reshape(-1).tolist()
    worst: Mismatch | None = None
    for position, (number, wanted) in enumerate(zip(flat, expected, strict=True)):
        try:
            compare_number(number, wanted, accuracy, f"{what}[{position}]")
        except Mismatch as difference:
            if worst is None or difference.deviation > worst.deviation:
                worst = difference
    if worst is not None:
        raise worst


def compare_gradients(
    found: Mapping[str, torch.Tensor | None],
    reference: Mapping[str, float | Fraction],
    accuracy: Accuracy,
    what: str,
) -> None:
    """Compare a gradient per occurrence and per shared parameter.

    Parameters
    ----------
    found : Mapping[str, torch.Tensor or None]
        The computed derivative of each name.
    reference : Mapping[str, float or Fraction]
        The derivative of each name.
    accuracy : Accuracy
        The accuracy to reach.
    what : str
        A description for the report.

    Raises
    ------
    Mismatch
        If a derivative is missing or differs by more than the tolerance;
        the largest difference is reported.
    """
    if set(found) != set(reference):
        raise Mismatch(
            "support",
            math.inf,
            f"{what}: names {sorted(found)} for {sorted(reference)}",
        )
    worst: Mismatch | None = None
    for name, wanted in reference.items():
        gradient = found[name]
        try:
            if gradient is None:
                raise Mismatch(
                    "missing", math.inf, f"{what}: no gradient for {name!r}"
                )
            if gradient.numel() != 1:
                raise Mismatch(
                    "shape", math.inf, f"{what}: {tuple(gradient.shape)} for {name!r}"
                )
            compare_number(float(gradient), wanted, accuracy, f"{what} d/d{name}")
        except Mismatch as difference:
            if worst is None or difference.deviation > worst.deviation:
                worst = difference
    if worst is not None:
        raise worst
