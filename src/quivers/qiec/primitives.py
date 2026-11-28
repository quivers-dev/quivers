"""The closed registry of pure primitives QIEC values may apply.

A primitive is nominal: it has a stable identity derived from its name, a
fixed monomorphic signature, and a capability tag a backend either
supports or refuses. Overloading is a surface concern: a source ``+`` on
two integers lowers to ``add_int`` and on two reals to ``add_real``, so
the kernel never has to choose an implementation by inspecting a runtime
value.

The implementations here are what the reference machine runs. They take
the run's `NumberDomain` and their arguments as the machine holds them,
numbers as tensors, and keep every gradient: a primitive applied to a
tensor of positions is applied entrywise by torch's own broadcasting. A
partial primitive refuses an argument outside its domain rather than
answering NaN: where its result is NaN and no argument it reads is (the
logarithm of a negative number, say), and where it divides by zero. The
probability primitives the compiled runtime shares are exported by name.
The host scalar helpers below them compute the same functions on Python
numbers for the reference distribution backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from types import MappingProxyType
from typing import Literal

import didactic.api as dx
import torch

from quivers._positions import demand, depends_on_position
from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec.identifiers import PrimitiveId
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.numbers import NumberDomain, host_index
from quivers.qiec.types import BOOL, INT, REAL, STRING, TypeExpr

type PrimitiveCapability = Literal[
    "arithmetic",
    "comparison",
    "strict-scalar-comparison",
    "boolean",
    "string",
    "conversion",
    "math",
    "special",
    "activation",
    "weight",
    "probability",
    "rule-log-weight",
]
"""The feature a target must support to render a primitive."""


class PrimitiveSignature(QiecModel):
    """One primitive's name, types, and the capability a backend needs for it.

    Parameters
    ----------
    name
        The stable nominal name, such as ``"add_int"``.
    parameters
        The argument types, in order.
    result
        The result type.
    capability
        The feature a target must support to render the primitive.
    total
        Whether all well-typed arguments satisfy its domain. This proof
        permits eager branch selection; unproved primitives default to false.
    """

    name: str
    parameters: tuple[TypeExpr, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    capability: PrimitiveCapability
    total: bool = False

    @property
    def id(self) -> PrimitiveId:
        """The primitive's stable identity.

        Returns
        -------
        PrimitiveId
            An identity derived from the ``builtin`` namespace and the
            name, so the same primitive has the same identity in every
            module.
        """
        return PrimitiveId.derive("builtin", self.name)


def _observation_status(value: int) -> int:
    """Validate one recorded-outcome event code.

    Parameters
    ----------
    value : int
        Code minus one left, zero exact, one right, or two interval.

    Returns
    -------
    int
        The validated code unchanged.

    Raises
    ------
    ValueError
        If the value is not a nonboolean integer in the code set.
    """
    if type(value) is not int or value not in {-1, 0, 1, 2}:
        raise ValueError("observation_status requires integer codes -1, 0, 1, or 2")
    return value


def _interval_upper(lower: float, upper: float) -> float:
    """Validate an active interval's finite, strictly larger upper endpoint.

    Parameters
    ----------
    lower, upper : float
        Real endpoints; the lower endpoint may be infinite.

    Returns
    -------
    float
        The upper endpoint unchanged.

    Raises
    ------
    ValueError
        If an endpoint is boolean or nonreal, lower is NaN, or upper is
        nonfinite or does not exceed lower.
    """
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in (lower, upper)
        )
        or math.isnan(lower)
        or not math.isfinite(upper)
        or upper <= lower
    ):
        raise ValueError(
            "interval_upper requires a finite upper endpoint greater than its lower endpoint"
        )
    return upper


def _positive_parameter(value: float) -> float:
    """Validate a finite, strictly positive scalar parameter.

    Parameters
    ----------
    value : float
        Candidate natural-scale parameter.

    Returns
    -------
    float
        The supplied parameter unchanged.

    Raises
    ------
    ValueError
        If the value is boolean, nonfinite, zero, or negative.
    """
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(
            "positive_parameter requires a finite strictly positive real value"
        )
    return value


def _log_probability(value: float) -> float:
    """Log a probability, retaining the zero-mass endpoint.

    Parameters
    ----------
    value : float
        Finite probability in the closed unit interval.

    Returns
    -------
    float
        Its logarithm, negative infinity at zero.

    Raises
    ------
    ValueError
        If the value is boolean, nonfinite, or outside the unit interval.
    """
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("log_probability requires a finite real value in [0, 1]")
    return math.log(value) if value else -math.inf


def _log1mexp(value: float) -> float:
    """Log a complementary exponential without cancellation.

    Parameters
    ----------
    value : float
        Nonpositive logarithm, including negative infinity.

    Returns
    -------
    float
        Log(1-exp(value)), negative infinity at zero and zero at -infinity.

    Raises
    ------
    ValueError
        If the value is boolean, NaN, or positive.
    """
    if isinstance(value, bool) or math.isnan(value) or value > 0:
        raise ValueError("log1mexp requires a nonpositive real value")
    if value == 0:
        return -math.inf
    if value > -math.log(2):
        return math.log(-math.expm1(value))
    return math.log1p(-math.exp(value))


def _log_ndtr(value: float) -> float:
    """Standard normal log CDF with a standard-library negative-tail kernel.

    Parameters
    ----------
    value : float
        Real argument, including either infinity.

    Returns
    -------
    float
        Standard normal log CDF. A Mills-ratio continued fraction avoids
        erfc underflow in the far negative tail.

    Raises
    ------
    ValueError
        If the value is boolean or NaN.
    """
    if isinstance(value, bool) or math.isnan(value):
        raise ValueError("log_ndtr requires a real value other than NaN")
    if value == math.inf:
        return 0.0
    if value == -math.inf:
        return -math.inf
    if value < -10:
        x = -value
        denominator = x
        for numerator in range(64, 0, -1):
            denominator = x + numerator / denominator
        return -0.5 * x * x - 0.5 * math.log(2 * math.pi) - math.log(denominator)
    if value > 0:
        return math.log1p(-0.5 * math.erfc(value / math.sqrt(2)))
    return math.log(0.5 * math.erfc(-value / math.sqrt(2)))


type Implementation = Callable[..., object]
"""The tensor implementation of one primitive.

It takes the run's `NumberDomain` and the primitive's arguments, numbers as
tensors and strings as strings, and returns the result in the same form.
"""


def _refuse_new_nan(
    name: str, result: torch.Tensor, *arguments: torch.Tensor
) -> torch.Tensor:
    """Refuse a result outside the real domain of a partial primitive.

    A partial primitive applied outside its domain gives NaN where none of
    its arguments is NaN: the logarithm of a negative number, the arcsine of
    a number above one, a fractional power of a negative number. That is a
    refusal, not a value, as the host arithmetic of every target treats it.

    Parameters
    ----------
    name : str
        The primitive.
    result : torch.Tensor
        Its result.
    *arguments : torch.Tensor
        Its numeric arguments.

    Returns
    -------
    torch.Tensor
        The result, unchanged.

    Raises
    ------
    ValueError
        If an entry is NaN and no argument entry it reads is.
    """
    inherited = torch.zeros_like(result, dtype=torch.bool)
    for argument in arguments:
        inherited = inherited | torch.isnan(argument)
    demand(
        ~torch.isnan(result) | inherited,
        lambda: ValueError(f"{name} is applied outside its domain"),
    )
    return result


def _refuse_zero_divisor(name: str, divisor: torch.Tensor) -> None:
    """Refuse a quotient by zero.

    Parameters
    ----------
    name : str
        The primitive.
    divisor : torch.Tensor
        The divisor.

    Raises
    ------
    ZeroDivisionError
        If an entry of the divisor is zero.
    """
    demand(divisor != 0, lambda: ZeroDivisionError(f"{name} divides by zero"))


def _partial(name: str, function: Callable[..., torch.Tensor]) -> Implementation:
    """A real primitive that refuses arguments outside its domain.

    Parameters
    ----------
    name : str
        The primitive.
    function : Callable[..., torch.Tensor]
        The torch function of its real arguments.

    Returns
    -------
    Implementation
        The implementation.
    """

    def apply(_domain: NumberDomain, *arguments: torch.Tensor) -> torch.Tensor:
        """Apply the function and refuse a result outside the domain.

        Parameters
        ----------
        _domain : NumberDomain
            The run's numbers, which the result's dtype already is.
        *arguments : torch.Tensor
            The arguments.

        Returns
        -------
        torch.Tensor
            The result.
        """
        return _refuse_new_nan(name, function(*arguments), *arguments)

    return apply


def _total(function: Callable[..., torch.Tensor]) -> Implementation:
    """A primitive defined on every argument of its type.

    Parameters
    ----------
    function : Callable[..., torch.Tensor]
        The torch function of its arguments.

    Returns
    -------
    Implementation
        The implementation, ignoring the domain.
    """

    def apply(_domain: NumberDomain, *arguments: torch.Tensor) -> torch.Tensor:
        """Apply the function.

        Parameters
        ----------
        _domain : NumberDomain
            The run's numbers, unused.
        *arguments : torch.Tensor
            The arguments.

        Returns
        -------
        torch.Tensor
            The result.
        """
        return function(*arguments)

    return apply


def _strict_less(
    _domain: NumberDomain, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Compare two finite real primals off the equality boundary.

    Parameters
    ----------
    _domain : NumberDomain
        The numeric representation already used for both arguments.
    left, right : torch.Tensor
        The Real arguments to compare.

    Returns
    -------
    torch.Tensor
        Whether the left primal is strictly below the right one.

    Raises
    ------
    ValueError
        If the two primals are equal or either is nonfinite.
    """
    demand(
        torch.isfinite(left) & torch.isfinite(right) & (left != right),
        lambda: ValueError(
            "qiec-forward-comparison-boundary: strict comparison requires finite distinct primals"
        ),
    )
    return left < right


def _truncating_division(
    _domain: NumberDomain, dividend: torch.Tensor, divisor: torch.Tensor
) -> torch.Tensor:
    """Integer division rounding toward zero, as every host target does.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    dividend : torch.Tensor
        The number divided.
    divisor : torch.Tensor
        The number divided by.

    Returns
    -------
    torch.Tensor
        The quotient truncated toward zero.

    Raises
    ------
    ZeroDivisionError
        If the divisor is zero.
    """
    _refuse_zero_divisor("div_int", divisor)
    return torch.div(dividend, divisor, rounding_mode="trunc")


def _truncating_remainder(
    _domain: NumberDomain, dividend: torch.Tensor, divisor: torch.Tensor
) -> torch.Tensor:
    """The remainder of truncating division, carrying the dividend's sign.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    dividend : torch.Tensor
        The number divided.
    divisor : torch.Tensor
        The number divided by.

    Returns
    -------
    torch.Tensor
        ``dividend - divisor * trunc(dividend / divisor)``.

    Raises
    ------
    ZeroDivisionError
        If the divisor is zero.
    """
    _refuse_zero_divisor("mod_int", divisor)
    return torch.fmod(dividend, divisor)


def _quotient(
    _domain: NumberDomain, dividend: torch.Tensor, divisor: torch.Tensor
) -> torch.Tensor:
    """The real quotient, refused for a zero divisor.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    dividend : torch.Tensor
        The number divided.
    divisor : torch.Tensor
        The number divided by.

    Returns
    -------
    torch.Tensor
        ``dividend / divisor``.

    Raises
    ------
    ZeroDivisionError
        If the divisor is zero.
    """
    _refuse_zero_divisor("div_real", divisor)
    return dividend / divisor


def _reciprocal(_domain: NumberDomain, value: torch.Tensor) -> torch.Tensor:
    """One over a number, refused at zero.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    value : torch.Tensor
        The number.

    Returns
    -------
    torch.Tensor
        ``1 / value``.

    Raises
    ------
    ZeroDivisionError
        If the number is zero.
    """
    _refuse_zero_divisor("reciprocal", value)
    return torch.reciprocal(value)


def _reciprocal_root(_domain: NumberDomain, value: torch.Tensor) -> torch.Tensor:
    """One over a square root, refused at zero and below.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    value : torch.Tensor
        The number.

    Returns
    -------
    torch.Tensor
        ``1 / sqrt(value)``.

    Raises
    ------
    ZeroDivisionError
        If the number is zero.
    ValueError
        If it is negative.
    """
    _refuse_zero_divisor("rsqrt", value)
    return _refuse_new_nan("rsqrt", torch.rsqrt(value), value)


def _power(
    _domain: NumberDomain, base: torch.Tensor, exponent: torch.Tensor
) -> torch.Tensor:
    """A real power, refused where it has no real value.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    base : torch.Tensor
        The base.
    exponent : torch.Tensor
        The exponent.

    Returns
    -------
    torch.Tensor
        ``base ** exponent``.

    Raises
    ------
    ZeroDivisionError
        If the base is zero and the exponent negative.
    ValueError
        If the base is negative and the exponent not an integer.
    """
    demand(
        (base != 0) | (exponent >= 0),
        lambda: ZeroDivisionError("pow_real raises zero to a negative power"),
    )
    return _refuse_new_nan("pow_real", torch.pow(base, exponent), base, exponent)


def _first_minimum(
    _domain: NumberDomain, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """The smaller of two numbers, the first at a tie.

    The tie rule is the one of every selective reduction of the library:
    the first operand is selected, so it receives the whole gradient.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    left, right : torch.Tensor
        The operands.

    Returns
    -------
    torch.Tensor
        The selected operand.
    """
    return torch.where(left <= right, left, right)


def _first_maximum(
    _domain: NumberDomain, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """The larger of two numbers, the first at a tie.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    left, right : torch.Tensor
        The operands.

    Returns
    -------
    torch.Tensor
        The selected operand.
    """
    return torch.where(left >= right, left, right)


def _select(
    _domain: NumberDomain,
    condition: torch.Tensor,
    chosen: torch.Tensor,
    other: torch.Tensor,
) -> torch.Tensor:
    """The first value where the condition holds, the second elsewhere.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    condition : torch.Tensor
        The truth value.
    chosen, other : torch.Tensor
        The values.

    Returns
    -------
    torch.Tensor
        The selected value, with the gradient of the selected operand only.
    """
    return torch.where(condition, chosen, other)


def _to_real(domain: NumberDomain, value: torch.Tensor) -> torch.Tensor:
    """An integer or truth value as a real number.

    Parameters
    ----------
    domain : NumberDomain
        The run's numbers.
    value : torch.Tensor
        The value.

    Returns
    -------
    torch.Tensor
        The value in the real dtype.
    """
    return value.to(dtype=domain.real_dtype)


def _to_integer(_domain: NumberDomain, value: torch.Tensor) -> torch.Tensor:
    """A truth value as an integer.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    value : torch.Tensor
        The truth value.

    Returns
    -------
    torch.Tensor
        ``1`` or ``0`` as ``int64``.
    """
    return value.to(dtype=torch.int64)


def _truncate_to_integer(_domain: NumberDomain, value: torch.Tensor) -> torch.Tensor:
    """A real number truncated toward zero, as an integer.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    value : torch.Tensor
        The real number.

    Returns
    -------
    torch.Tensor
        The integer part as ``int64``.

    Raises
    ------
    ValueError
        If the number is not finite.
    """
    demand(
        torch.isfinite(value),
        lambda: ValueError("real_to_int requires a finite real value"),
    )
    return torch.trunc(value).to(dtype=torch.int64)


def _to_string(_domain: NumberDomain, value: torch.Tensor) -> str:
    """An integer written in decimal.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    value : torch.Tensor
        The integer.

    Returns
    -------
    str
        Its decimal spelling; the integer is read as a host index, since a
        string is a host value.
    """
    return str(host_index(value, "int_to_string"))


def _string_relation(
    relation: Callable[[str, str], bool],
) -> Implementation:
    """A relation between strings as a primitive answering a truth value.

    Parameters
    ----------
    relation : Callable[[str, str], bool]
        The relation.

    Returns
    -------
    Implementation
        The implementation.
    """

    def apply(domain: NumberDomain, left: str, right: str) -> torch.Tensor:
        """Relate two strings.

        Parameters
        ----------
        domain : NumberDomain
            The run's numbers, which hold the truth value.
        left, right : str
            The strings.

        Returns
        -------
        torch.Tensor
            Whether the relation holds.
        """
        return domain.number("truth", relation(left, right))

    return apply


def _concatenate(_domain: NumberDomain, left: str, right: str) -> str:
    """Two strings, one after the other.

    Parameters
    ----------
    _domain : NumberDomain
        The run's numbers, unused.
    left, right : str
        The strings.

    Returns
    -------
    str
        Their concatenation.
    """
    return left + right


def observation_weight(value: torch.Tensor) -> torch.Tensor:
    """Validate finite, nonnegative observation weights.

    Parameters
    ----------
    value : torch.Tensor
        A floating tensor of coefficients multiplying log densities.

    Returns
    -------
    torch.Tensor
        The supplied value, unchanged and retaining its gradient.

    Raises
    ------
    ValueError
        If a weight is nonfinite or negative, or the tensor is not real.
    """
    error = ValueError("observation weights must be finite and nonnegative real values")
    if not value.is_floating_point():
        raise error
    demand(torch.isfinite(value) & (value >= 0), lambda: error)
    return value


def observation_status(value: torch.Tensor) -> torch.Tensor:
    """Validate recorded-outcome event codes.

    Parameters
    ----------
    value : torch.Tensor
        Integer codes from minus one through two.

    Returns
    -------
    torch.Tensor
        The codes, unchanged.

    Raises
    ------
    ValueError
        If the tensor is not of integers or a code is outside the set.
    """
    error = ValueError("observation_status requires integer codes -1, 0, 1, or 2")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise error
    demand((value >= -1) & (value <= 2), lambda: error)
    return value


def interval_upper(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Validate an active interval's finite, strictly larger upper endpoint.

    Parameters
    ----------
    lower, upper : torch.Tensor
        Real endpoints; the lower endpoint may be infinite.

    Returns
    -------
    torch.Tensor
        The upper endpoint broadcast to both shapes, with its gradient.

    Raises
    ------
    ValueError
        If an endpoint is not real, the lower is NaN, or the upper is not
        finite or does not exceed the lower.
    """
    if any(
        value.dtype == torch.bool or not value.is_floating_point()
        for value in (lower, upper)
    ):
        raise ValueError("interval_upper requires real endpoints")
    demand(
        ~lower.isnan() & upper.isfinite() & (upper > lower),
        lambda: ValueError(
            "interval_upper requires a finite upper endpoint greater than its lower endpoint"
        ),
    )
    shape = torch.broadcast_shapes(lower.shape, upper.shape)
    return upper if upper.shape == shape else upper.expand(shape)


def positive_parameter(value: torch.Tensor) -> torch.Tensor:
    """Validate a positive real parameter without changing it.

    Parameters
    ----------
    value : torch.Tensor
        Natural-scale parameters.

    Returns
    -------
    torch.Tensor
        The value, unchanged and retaining its gradient.

    Raises
    ------
    ValueError
        If a value is not real, not finite, zero, or negative.
    """
    valid = (
        torch.isfinite(value) & (value > 0)
        if value.is_floating_point()
        else torch.zeros_like(value, dtype=torch.bool)
    )

    def refusal() -> ValueError:
        """The error, naming the invalid entries a run of one position has.

        Returns
        -------
        ValueError
            The error.
        """
        message = "positive_parameter requires finite strictly positive real values"
        if depends_on_position(valid):
            return ValueError(message)
        found = (~valid).nonzero(as_tuple=False)[:16].detach().cpu().tolist()
        return ValueError(f"{message}; invalid tensor positions {found}")

    demand(valid, refusal)
    return value


def log_probability(value: torch.Tensor) -> torch.Tensor:
    """Log probabilities, negative infinity at zero.

    Parameters
    ----------
    value : torch.Tensor
        Finite real probabilities in the closed unit interval.

    Returns
    -------
    torch.Tensor
        Their logarithms. The zero endpoint is masked before the logarithm,
        so its inactive derivative does not leak.

    Raises
    ------
    ValueError
        If a value is not real, not finite, or outside [0, 1].
    """
    error = ValueError("log_probability requires finite real values in [0, 1]")
    if not value.is_floating_point():
        raise error
    demand(torch.isfinite(value) & (value >= 0) & (value <= 1), lambda: error)
    positive = value > 0
    safe = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, safe.log(), -torch.inf)


def _require_rule_log_weight(
    _domain: NumberDomain, value: torch.Tensor
) -> torch.Tensor:
    """Require a finite application weight or exact additive zero.

    Parameters
    ----------
    _domain : NumberDomain
        The number domain, unchanged.
    value : torch.Tensor
        A scalar LogWeight from a checked rule computation.

    Returns
    -------
    torch.Tensor
        The same live weight.

    Raises
    ------
    ValueError
        If the weight is NaN or positive infinity.
    """
    demand(
        ~value.isnan() & ~value.isposinf(),
        lambda: ValueError("computed rule weight must be finite or negative infinity"),
    )
    return value


def log1mexp(value: torch.Tensor) -> torch.Tensor:
    """``log(1 - exp(value))`` without cancellation.

    Parameters
    ----------
    value : torch.Tensor
        Nonpositive real values, including negative infinity.

    Returns
    -------
    torch.Tensor
        Negative infinity at zero and zero at negative infinity; the two
        branches are evaluated at protected arguments only.

    Raises
    ------
    ValueError
        If a value is not real, NaN, or positive.
    """
    error = ValueError("log1mexp requires nonpositive real values")
    if not value.is_floating_point():
        raise error
    demand(~torch.isnan(value) & (value <= 0), lambda: error)
    interior = torch.isfinite(value) & (value < 0)
    safe = torch.where(interior, value, -torch.ones_like(value))
    near = safe > -math.log(2)
    near_arg = torch.where(near, safe, -torch.ones_like(safe))
    far_arg = torch.where(near, -torch.ones_like(safe), safe)
    result = torch.where(
        near, torch.log(-torch.expm1(near_arg)), torch.log1p(-far_arg.exp())
    )
    return torch.where(
        value == 0, -torch.inf, torch.where(value.isneginf(), 0.0, result)
    )


def log_ndtr(value: torch.Tensor) -> torch.Tensor:
    """The standard normal log distribution function.

    Parameters
    ----------
    value : torch.Tensor
        Real values, including either infinity.

    Returns
    -------
    torch.Tensor
        Negative infinity at negative infinity, zero at infinity, with
        the infinite endpoints masked before the function is applied.

    Raises
    ------
    ValueError
        If a value is not real or is NaN.
    """
    error = ValueError("log_ndtr requires real values other than NaN")
    if not value.is_floating_point():
        raise error
    demand(~torch.isnan(value), lambda: error)
    safe = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    result = torch.special.log_ndtr(safe)
    return torch.where(
        value.isneginf(), -torch.inf, torch.where(value.isposinf(), 0.0, result)
    )


def scale_weight(coefficient: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
    """Multiply log weights, a zero coefficient absorbing any density.

    Parameters
    ----------
    coefficient : torch.Tensor
        The coefficient, a scalar or one per density position.
    density : torch.Tensor
        The log density at each position.

    Returns
    -------
    torch.Tensor
        The product, with zero-coefficient positions masked before the
        multiplication, so an infinite density there gives zero and no NaN
        gradient.
    """
    safe = torch.where(coefficient == 0, torch.zeros_like(density), density)
    return coefficient * safe


def _sign(value: torch.Tensor) -> torch.Tensor:
    """The sign of a number, with zero derivative.

    Parameters
    ----------
    value : torch.Tensor
        The argument.

    Returns
    -------
    torch.Tensor
        Minus one, zero, or one.
    """
    return torch.sign(value)


def _signature(
    name: str,
    parameters: tuple[TypeExpr, ...],
    result: TypeExpr,
    capability: PrimitiveCapability,
    implementation: Implementation,
    *,
    total: bool = False,
) -> tuple[PrimitiveSignature, Implementation]:
    """Pair a signature with its tensor implementation.

    Parameters
    ----------
    name : str
        The primitive's name.
    parameters : tuple[TypeExpr, ...]
        The argument types.
    result : TypeExpr
        The result type.
    capability : PrimitiveCapability
        The feature tag.
    implementation : Implementation
        The implementation over machine values.
    total : bool
        Whether the primitive has no additional argument-domain restriction.

    Returns
    -------
    tuple[PrimitiveSignature, Implementation]
        The signature and implementation, ready for the tables below.
    """
    return PrimitiveSignature(
        name=name,
        parameters=parameters,
        result=result,
        capability=capability,
        total=total or capability in {"comparison", "boolean", "string"},
    ), implementation


_F = torch.nn.functional

_ENTRIES: tuple[tuple[PrimitiveSignature, Implementation], ...] = (
    _signature("add_int", (INT, INT), INT, "arithmetic", _total(torch.add), total=True),
    _signature("sub_int", (INT, INT), INT, "arithmetic", _total(torch.sub), total=True),
    _signature("mul_int", (INT, INT), INT, "arithmetic", _total(torch.mul), total=True),
    _signature("div_int", (INT, INT), INT, "arithmetic", _truncating_division),
    _signature("mod_int", (INT, INT), INT, "arithmetic", _truncating_remainder),
    _signature("neg_int", (INT,), INT, "arithmetic", _total(torch.neg), total=True),
    _signature("abs_int", (INT,), INT, "arithmetic", _total(torch.abs), total=True),
    _signature("min_int", (INT, INT), INT, "arithmetic", _first_minimum, total=True),
    _signature("max_int", (INT, INT), INT, "arithmetic", _first_maximum, total=True),
    _signature(
        "add_real", (REAL, REAL), REAL, "arithmetic", _total(torch.add), total=True
    ),
    _signature(
        "sub_real", (REAL, REAL), REAL, "arithmetic", _total(torch.sub), total=True
    ),
    _signature(
        "mul_real", (REAL, REAL), REAL, "arithmetic", _total(torch.mul), total=True
    ),
    _signature("div_real", (REAL, REAL), REAL, "arithmetic", _quotient),
    _signature("neg_real", (REAL,), REAL, "arithmetic", _total(torch.neg), total=True),
    _signature("abs_real", (REAL,), REAL, "arithmetic", _total(torch.abs), total=True),
    _signature(
        "min_real", (REAL, REAL), REAL, "arithmetic", _first_minimum, total=True
    ),
    _signature(
        "max_real", (REAL, REAL), REAL, "arithmetic", _first_maximum, total=True
    ),
    _signature("pow_real", (REAL, REAL), REAL, "math", _power),
    _signature("exp", (REAL,), REAL, "math", _partial("exp", torch.exp)),
    _signature("log", (REAL,), REAL, "math", _partial("log", torch.log)),
    _signature(
        "log_probability", (REAL,), REAL, "probability", _total(log_probability)
    ),
    _signature(
        "positive_parameter", (REAL,), REAL, "probability", _total(positive_parameter)
    ),
    _signature(
        "observation_status", (INT,), INT, "probability", _total(observation_status)
    ),
    _signature(
        "interval_upper", (REAL, REAL), REAL, "probability", _total(interval_upper)
    ),
    _signature("log1mexp", (REAL,), REAL, "probability", _total(log1mexp)),
    _signature("log_ndtr", (REAL,), REAL, "probability", _total(log_ndtr)),
    _signature("sqrt", (REAL,), REAL, "math", _partial("sqrt", torch.sqrt)),
    _signature("eq_int", (INT, INT), BOOL, "comparison", _total(torch.eq)),
    _signature("ne_int", (INT, INT), BOOL, "comparison", _total(torch.ne)),
    _signature("lt_int", (INT, INT), BOOL, "comparison", _total(torch.lt)),
    _signature("le_int", (INT, INT), BOOL, "comparison", _total(torch.le)),
    _signature("gt_int", (INT, INT), BOOL, "comparison", _total(torch.gt)),
    _signature("ge_int", (INT, INT), BOOL, "comparison", _total(torch.ge)),
    _signature("eq_real", (REAL, REAL), BOOL, "comparison", _total(torch.eq)),
    _signature("ne_real", (REAL, REAL), BOOL, "comparison", _total(torch.ne)),
    _signature("lt_real", (REAL, REAL), BOOL, "comparison", _total(torch.lt)),
    _signature(
        "strict_lt_real", (REAL, REAL), BOOL, "strict-scalar-comparison", _strict_less
    ),
    _signature("le_real", (REAL, REAL), BOOL, "comparison", _total(torch.le)),
    _signature("gt_real", (REAL, REAL), BOOL, "comparison", _total(torch.gt)),
    _signature("ge_real", (REAL, REAL), BOOL, "comparison", _total(torch.ge)),
    _signature("eq_bool", (BOOL, BOOL), BOOL, "comparison", _total(torch.eq)),
    _signature("ne_bool", (BOOL, BOOL), BOOL, "comparison", _total(torch.ne)),
    _signature(
        "eq_string",
        (STRING, STRING),
        BOOL,
        "comparison",
        _string_relation(lambda a, b: a == b),
    ),
    _signature(
        "ne_string",
        (STRING, STRING),
        BOOL,
        "comparison",
        _string_relation(lambda a, b: a != b),
    ),
    _signature("and", (BOOL, BOOL), BOOL, "boolean", _total(torch.logical_and)),
    _signature("or", (BOOL, BOOL), BOOL, "boolean", _total(torch.logical_or)),
    _signature("not", (BOOL,), BOOL, "boolean", _total(torch.logical_not)),
    _signature("concat", (STRING, STRING), STRING, "string", _concatenate),
    _signature("int_to_real", (INT,), REAL, "conversion", _to_real),
    _signature("int_to_string", (INT,), STRING, "conversion", _to_string),
    _signature("real_to_int", (REAL,), INT, "conversion", _truncate_to_integer),
    _signature("bool_to_real", (BOOL,), REAL, "conversion", _to_real),
    _signature("bool_to_int", (BOOL,), INT, "conversion", _to_integer),
    _signature("select_real", (BOOL, REAL, REAL), REAL, "boolean", _select),
    _signature("select_int", (BOOL, INT, INT), INT, "boolean", _select),
    _signature("expm1", (REAL,), REAL, "math", _partial("expm1", torch.expm1)),
    _signature("log1p", (REAL,), REAL, "math", _partial("log1p", torch.log1p)),
    _signature("log2", (REAL,), REAL, "math", _partial("log2", torch.log2)),
    _signature("log10", (REAL,), REAL, "math", _partial("log10", torch.log10)),
    _signature("rsqrt", (REAL,), REAL, "math", _reciprocal_root),
    _signature("square", (REAL,), REAL, "math", _total(torch.square)),
    _signature("sign", (REAL,), REAL, "math", _total(_sign)),
    _signature("reciprocal", (REAL,), REAL, "math", _reciprocal),
    _signature("sin", (REAL,), REAL, "math", _partial("sin", torch.sin)),
    _signature("cos", (REAL,), REAL, "math", _partial("cos", torch.cos)),
    _signature("tan", (REAL,), REAL, "math", _partial("tan", torch.tan)),
    _signature("asin", (REAL,), REAL, "math", _partial("asin", torch.asin)),
    _signature("acos", (REAL,), REAL, "math", _partial("acos", torch.acos)),
    _signature("atan", (REAL,), REAL, "math", _partial("atan", torch.atan)),
    _signature("sinh", (REAL,), REAL, "math", _partial("sinh", torch.sinh)),
    _signature("cosh", (REAL,), REAL, "math", _partial("cosh", torch.cosh)),
    _signature("tanh", (REAL,), REAL, "math", _partial("tanh", torch.tanh)),
    _signature("asinh", (REAL,), REAL, "math", _partial("asinh", torch.asinh)),
    _signature("acosh", (REAL,), REAL, "math", _partial("acosh", torch.acosh)),
    _signature("atanh", (REAL,), REAL, "math", _partial("atanh", torch.atanh)),
    _signature("floor", (REAL,), REAL, "math", _total(torch.floor)),
    _signature("ceil", (REAL,), REAL, "math", _total(torch.ceil)),
    _signature("round", (REAL,), REAL, "math", _total(torch.round)),
    _signature("trunc", (REAL,), REAL, "math", _total(torch.trunc)),
    _signature("erf", (REAL,), REAL, "special", _partial("erf", torch.erf)),
    _signature("erfc", (REAL,), REAL, "special", _partial("erfc", torch.erfc)),
    _signature("erfinv", (REAL,), REAL, "special", _partial("erfinv", torch.erfinv)),
    _signature("lgamma", (REAL,), REAL, "special", _partial("lgamma", torch.lgamma)),
    _signature("digamma", (REAL,), REAL, "special", _partial("digamma", torch.digamma)),
    _signature("sigmoid", (REAL,), REAL, "activation", _total(torch.sigmoid)),
    _signature("relu", (REAL,), REAL, "activation", _total(_F.relu)),
    _signature("relu6", (REAL,), REAL, "activation", _total(_F.relu6)),
    _signature("elu", (REAL,), REAL, "activation", _total(_F.elu)),
    _signature("selu", (REAL,), REAL, "activation", _total(_F.selu)),
    _signature("gelu", (REAL,), REAL, "activation", _total(_F.gelu)),
    _signature("silu", (REAL,), REAL, "activation", _total(_F.silu)),
    _signature("mish", (REAL,), REAL, "activation", _total(_F.mish)),
    _signature("softplus", (REAL,), REAL, "activation", _total(_F.softplus)),
    _signature("logsigmoid", (REAL,), REAL, "activation", _total(_F.logsigmoid)),
    _signature("softsign", (REAL,), REAL, "activation", _total(_F.softsign)),
    _signature(
        "observation_weight", (REAL,), REAL, "weight", _total(observation_weight)
    ),
    _signature("as_weight", (REAL,), LOG_WEIGHT, "weight", _total(torch.clone)),
    _signature(
        "require_rule_log_weight",
        (LOG_WEIGHT,),
        LOG_WEIGHT,
        "rule-log-weight",
        _require_rule_log_weight,
    ),
    _signature("weight_value", (LOG_WEIGHT,), REAL, "weight", _total(torch.clone)),
    _signature(
        "add_weight",
        (LOG_WEIGHT, LOG_WEIGHT),
        LOG_WEIGHT,
        "weight",
        _total(torch.add),
    ),
    _signature(
        "scale_weight",
        (REAL, LOG_WEIGHT),
        LOG_WEIGHT,
        "weight",
        _total(scale_weight),
    ),
)


PRIMITIVES: Mapping[str, PrimitiveSignature] = MappingProxyType(
    {signature.name: signature for signature, _ in _ENTRIES}
)
"""Every primitive by name.

The mapping is closed: a name absent here is not a primitive, and no backend
may resolve one dynamically.
"""

IMPLEMENTATIONS: Mapping[str, Implementation] = MappingProxyType(
    {signature.name: implementation for signature, implementation in _ENTRIES}
)
"""The reference machine's tensor implementation of every primitive, by name."""


def primitive(name: str) -> PrimitiveSignature:
    """Look up a primitive by name.

    Parameters
    ----------
    name : str
        The nominal name.

    Returns
    -------
    PrimitiveSignature
        The primitive's signature.

    Raises
    ------
    KeyError
        If no primitive has the name. Callers turn this into a
        source-located diagnostic; the registry itself has no location to
        blame.
    """
    try:
        return PRIMITIVES[name]
    except KeyError as error:
        raise KeyError(f"unknown QIEC primitive {name!r}") from error


__all__ = [
    "IMPLEMENTATIONS",
    "Implementation",
    "PRIMITIVES",
    "PrimitiveCapability",
    "PrimitiveSignature",
    "interval_upper",
    "log1mexp",
    "log_ndtr",
    "log_probability",
    "observation_status",
    "observation_weight",
    "positive_parameter",
    "primitive",
    "scale_weight",
]
