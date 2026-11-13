"""Parameter bijectors and domain maps for inline distribution arguments.

Raw network outputs use bijectors. Inline arguments are already parameters:
positive values are preserved, and a declared closed interval uses its
projection. A projection is a callable and has no inverse or Jacobian API.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from quivers.continuous.bijectors import (
    AffineBijector,
    Bijector,
    ComposeBijector,
    ExpBijector,
    IdentityBijector,
    SigmoidBijector,
    SoftplusBijector,
)
from quivers.core._util import EPS


def _identity(value: Tensor) -> Tensor:
    """Preserve an inline parameter.

    Parameters
    ----------
    value : Tensor
        The parameter.

    Returns
    -------
    Tensor
        The same tensor.
    """
    return value


def positive_parameter(value: Tensor) -> Tensor:
    """Preserve a positive parameter, refusing a value outside its domain.

    Parameters
    ----------
    value : Tensor
        The positive parameter.

    Returns
    -------
    Tensor
        The same tensor, retaining its derivative.

    Raises
    ------
    ValueError
        If any entry is nonpositive or NaN.
    """
    if not bool((value > 0).all()):
        raise ValueError("an inline positive parameter must be greater than zero")
    return value


def closed_interval_clamp(low: float, high: float) -> Callable[[Tensor], Tensor]:
    """Return the projection onto a closed parameter interval.

    Parameters
    ----------
    low : float
        The lower endpoint, or negative infinity.
    high : float
        The upper endpoint, or positive infinity.

    Returns
    -------
    Callable[[Tensor], Tensor]
        A tensor-valued projection preserving values in the interval.

    Raises
    ------
    ValueError
        If the lower endpoint exceeds the upper endpoint.
    """
    if low > high:
        raise ValueError(
            f"closed_interval_clamp: the lower bound {low} exceeds the upper "
            f"bound {high}"
        )

    def project(value: Tensor) -> Tensor:
        """Project a parameter onto the interval.

        Parameters
        ----------
        value : Tensor
            The parameter.

        Returns
        -------
        Tensor
            The projected values.
        """
        return value.clamp(min=low, max=high)

    return project


def _softplus_shifted(shift: float) -> Bijector:
    """Compose softplus with an additive shift.

    Parameters
    ----------
    shift : float
        The lower endpoint of the image.

    Returns
    -------
    Bijector
        Softplus followed by the shift.
    """
    return ComposeBijector(AffineBijector(scale=1.0, shift=shift), SoftplusBijector())


TRANSFORM_TO_BIJECTOR: dict[str, Bijector] = {
    "id": IdentityBijector(),
    "sigmoid": SigmoidBijector(),
    "softplus": _softplus_shifted(EPS),
    "softplus_shifted": _softplus_shifted(0.1),
    "exp": ExpBijector(),
}
"""Bijectors from raw network outputs to family parameters."""


INLINE_CLAMPS: dict[str, Callable[[Tensor], Tensor]] = {
    "id": _identity,
    "softplus": positive_parameter,
    "softplus_shifted": positive_parameter,
    "exp": positive_parameter,
    "sigmoid": closed_interval_clamp(0.0, 1.0),
}
"""Domain maps for parameters provided directly by an inline call.

A positive parameter has no nearest positive replacement for zero, so a
nonpositive value raises. Every positive value is preserved. Probability
parameters use the closed unit interval; the distribution validates any
endpoint excluded by its particular family. ParamSpec.inline_bounds overrides
these maps for family-specific closed domains.
"""


def resolve_transform(transform: str | Bijector) -> Bijector:
    """Resolve a raw parameter transform.

    Parameters
    ----------
    transform : str | Bijector
        A registered string key or a bijector.

    Returns
    -------
    Bijector
        The resolved transform.

    Raises
    ------
    KeyError
        If a string key is unregistered.
    TypeError
        If the argument is neither a string nor a bijector.
    """
    if isinstance(transform, Bijector):
        return transform
    if isinstance(transform, str):
        if transform not in TRANSFORM_TO_BIJECTOR:
            raise KeyError(
                f"unknown transform {transform!r}; "
                f"valid string keys: {sorted(TRANSFORM_TO_BIJECTOR)}"
            )
        return TRANSFORM_TO_BIJECTOR[transform]
    raise TypeError(
        "resolve_transform: expected a string key or a Bijector instance; "
        f"got {type(transform).__name__}"
    )


def resolve_inline_clamp(transform: str | Bijector) -> Callable[[Tensor], Tensor]:
    """Resolve the domain map for an inline parameter.

    Parameters
    ----------
    transform : str | Bijector
        The raw transform identifying the parameter domain. A custom
        bijector has no registered domain map and uses the identity.

    Returns
    -------
    Callable[[Tensor], Tensor]
        The inline domain map.

    Raises
    ------
    KeyError
        If a string key is unregistered.
    TypeError
        If the argument is neither a string nor a bijector.
    """
    if isinstance(transform, Bijector):
        return _identity
    if isinstance(transform, str):
        if transform not in INLINE_CLAMPS:
            raise KeyError(
                f"unknown transform {transform!r}; "
                f"valid string keys: {sorted(INLINE_CLAMPS)}"
            )
        return INLINE_CLAMPS[transform]
    raise TypeError(
        "resolve_inline_clamp: expected a string key or a Bijector instance; "
        f"got {type(transform).__name__}"
    )


__all__: list[str] = []
