"""The closed registry of pure primitives QIEC values may apply.

A primitive is nominal: it has a stable identity derived from its name, a
fixed monomorphic signature, and a capability tag a backend either
supports or refuses. Overloading is a surface concern: a source ``+`` on
two integers lowers to ``add_int`` and on two reals to ``add_real``, so
the kernel never has to choose an implementation by inspecting a runtime
value. The reference implementations here are what the evaluator runs;
each host runtime carries the same table in its own language.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from types import MappingProxyType
from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec.identifiers import PrimitiveId
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.types import BOOL, INT, REAL, STRING, TypeExpr

type PrimitiveCapability = Literal[
    "arithmetic",
    "comparison",
    "boolean",
    "string",
    "conversion",
    "math",
    "special",
    "activation",
    "weight",
    "probability",
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
    """

    name: str
    parameters: tuple[TypeExpr, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    capability: PrimitiveCapability

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


def _observation_weight(value: float) -> float:
    """Validate a finite, nonnegative observation weight.

    Parameters
    ----------
    value : float
        The coefficient multiplying one observation's log density.

    Returns
    -------
    float
        The supplied coefficient, unchanged.

    Raises
    ------
    ValueError
        If the coefficient is negative, nonfinite, or boolean.
    """
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("observation weights must be finite and nonnegative")
    return value


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


def _truncating_division(dividend: int, divisor: int) -> int:
    """Integer division rounding toward zero, as every host target does.

    Parameters
    ----------
    dividend : int
        The number divided.
    divisor : int
        The number divided by.

    Returns
    -------
    int
        The quotient truncated toward zero. Python's ``//`` floors, which
        differs for a negative operand, so the host semantics are spelled
        out rather than inherited.

    Raises
    ------
    ZeroDivisionError
        If the divisor is zero.
    """
    quotient = abs(dividend) // abs(divisor)
    return quotient if (dividend < 0) == (divisor < 0) else -quotient


def _truncating_remainder(dividend: int, divisor: int) -> int:
    """The remainder of truncating division, carrying the dividend's sign.

    Parameters
    ----------
    dividend : int
        The number divided.
    divisor : int
        The number divided by.

    Returns
    -------
    int
        ``dividend - divisor * trunc(dividend / divisor)``.

    Raises
    ------
    ZeroDivisionError
        If the divisor is zero.
    """
    return dividend - divisor * _truncating_division(dividend, divisor)


def _erfinv(value: float) -> float:
    """The inverse error function.

    Parameters
    ----------
    value : float
        A number in the open interval from minus one to one.

    Returns
    -------
    float
        The ``x`` with ``erf(x) == value``, refined by Newton steps from
        a rational initial estimate; infinite at the endpoints.

    Raises
    ------
    ValueError
        If the value lies outside the closed interval.
    """
    if value < -1.0 or value > 1.0:
        raise ValueError("erfinv is defined on [-1, 1]")
    if value == 1.0:
        return math.inf
    if value == -1.0:
        return -math.inf
    if value == 0.0:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    magnitude = abs(value)
    if magnitude < 0.7:
        a = (0.886226899, -1.645349621, 0.914624893, -0.140543331)
        b = (-2.118377725, 1.442710462, -0.329097515, 0.012229801)
        square = magnitude * magnitude
        estimate = (
            magnitude
            * (((a[3] * square + a[2]) * square + a[1]) * square + a[0])
            / (
                (((b[3] * square + b[2]) * square + b[1]) * square + b[0]) * square
                + 1.0
            )
        )
    else:
        c = (-1.970840454, -1.62490649, 3.429567803, 1.641345311)
        d = (3.543889200, 1.637067800)
        tail = math.sqrt(-math.log((1.0 - magnitude) / 2.0))
        estimate = (((c[3] * tail + c[2]) * tail + c[1]) * tail + c[0]) / (
            (d[1] * tail + d[0]) * tail + 1.0
        )
    for _ in range(3):
        error = math.erf(estimate) - magnitude
        estimate -= error / (2.0 / math.sqrt(math.pi) * math.exp(-estimate * estimate))
    return sign * estimate


def _digamma(value: float) -> float:
    """The logarithmic derivative of the gamma function.

    Parameters
    ----------
    value : float
        A positive number, or a negative non-integer reached through the
        reflection formula.

    Returns
    -------
    float
        ``psi(value)``, by recurrence up to a large argument and the
        asymptotic series there.

    Raises
    ------
    ValueError
        If the value is a nonpositive integer, where the function has a
        pole.
    """
    if value <= 0.0 and value == math.floor(value):
        raise ValueError("digamma has a pole at nonpositive integers")
    if value < 0.0:
        return _digamma(1.0 - value) - math.pi / math.tan(math.pi * value)
    result = 0.0
    while value < 6.0:
        result -= 1.0 / value
        value += 1.0
    inverse = 1.0 / value
    square = inverse * inverse
    return (
        result
        + math.log(value)
        - 0.5 * inverse
        - square
        * (
            1.0 / 12.0
            - square
            * (
                1.0 / 120.0
                - square * (1.0 / 252.0 - square * (1.0 / 240.0 - square / 132.0))
            )
        )
    )


def _sigmoid(value: float) -> float:
    """The logistic function, stably at both tails.

    Parameters
    ----------
    value : float
        The argument.

    Returns
    -------
    float
        ``1 / (1 + exp(-value))``.
    """
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _softplus(value: float) -> float:
    """``log(1 + exp(value))``, stably at both tails.

    Parameters
    ----------
    value : float
        The argument.

    Returns
    -------
    float
        The softplus.
    """
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _gelu(value: float) -> float:
    """The Gaussian error linear unit.

    Parameters
    ----------
    value : float
        The argument.

    Returns
    -------
    float
        ``value * Phi(value)`` with ``Phi`` the standard normal
        distribution function.
    """
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def _selu(value: float) -> float:
    """The scaled exponential linear unit.

    Parameters
    ----------
    value : float
        The argument.

    Returns
    -------
    float
        The unit at its self-normalizing constants.
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * (value if value > 0.0 else alpha * (math.exp(value) - 1.0))


def _sign(value: float) -> float:
    """The sign of a number.

    Parameters
    ----------
    value : float
        The argument.

    Returns
    -------
    float
        Minus one, zero, or one.
    """
    return (value > 0.0) - (value < 0.0)


def _signature(
    name: str,
    parameters: tuple[TypeExpr, ...],
    result: TypeExpr,
    capability: PrimitiveCapability,
    implementation: Callable[..., object],
) -> tuple[PrimitiveSignature, Callable[..., object]]:
    """Pair a signature with its reference implementation.

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
    implementation : Callable[..., object]
        The reference implementation over host values.

    Returns
    -------
    tuple[PrimitiveSignature, Callable[..., object]]
        The signature and implementation, ready for the tables below.
    """
    return PrimitiveSignature(
        name=name, parameters=parameters, result=result, capability=capability
    ), implementation


_ENTRIES: tuple[tuple[PrimitiveSignature, Callable[..., object]], ...] = (
    _signature("add_int", (INT, INT), INT, "arithmetic", lambda a, b: a + b),
    _signature("sub_int", (INT, INT), INT, "arithmetic", lambda a, b: a - b),
    _signature("mul_int", (INT, INT), INT, "arithmetic", lambda a, b: a * b),
    _signature("div_int", (INT, INT), INT, "arithmetic", _truncating_division),
    _signature("mod_int", (INT, INT), INT, "arithmetic", _truncating_remainder),
    _signature("neg_int", (INT,), INT, "arithmetic", lambda a: -a),
    _signature("abs_int", (INT,), INT, "arithmetic", abs),
    _signature("min_int", (INT, INT), INT, "arithmetic", min),
    _signature("max_int", (INT, INT), INT, "arithmetic", max),
    _signature("add_real", (REAL, REAL), REAL, "arithmetic", lambda a, b: a + b),
    _signature("sub_real", (REAL, REAL), REAL, "arithmetic", lambda a, b: a - b),
    _signature("mul_real", (REAL, REAL), REAL, "arithmetic", lambda a, b: a * b),
    _signature("div_real", (REAL, REAL), REAL, "arithmetic", lambda a, b: a / b),
    _signature("neg_real", (REAL,), REAL, "arithmetic", lambda a: -a),
    _signature("abs_real", (REAL,), REAL, "arithmetic", abs),
    _signature("min_real", (REAL, REAL), REAL, "arithmetic", min),
    _signature("max_real", (REAL, REAL), REAL, "arithmetic", max),
    _signature("pow_real", (REAL, REAL), REAL, "math", math.pow),
    _signature("exp", (REAL,), REAL, "math", math.exp),
    _signature("log", (REAL,), REAL, "math", math.log),
    _signature("log_probability", (REAL,), REAL, "probability", _log_probability),
    _signature("positive_parameter", (REAL,), REAL, "probability", _positive_parameter),
    _signature("observation_status", (INT,), INT, "probability", _observation_status),
    _signature("interval_upper", (REAL, REAL), REAL, "probability", _interval_upper),
    _signature("log1mexp", (REAL,), REAL, "probability", _log1mexp),
    _signature("log_ndtr", (REAL,), REAL, "probability", _log_ndtr),
    _signature("sqrt", (REAL,), REAL, "math", math.sqrt),
    _signature("eq_int", (INT, INT), BOOL, "comparison", lambda a, b: a == b),
    _signature("ne_int", (INT, INT), BOOL, "comparison", lambda a, b: a != b),
    _signature("lt_int", (INT, INT), BOOL, "comparison", lambda a, b: a < b),
    _signature("le_int", (INT, INT), BOOL, "comparison", lambda a, b: a <= b),
    _signature("gt_int", (INT, INT), BOOL, "comparison", lambda a, b: a > b),
    _signature("ge_int", (INT, INT), BOOL, "comparison", lambda a, b: a >= b),
    _signature("eq_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a == b),
    _signature("ne_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a != b),
    _signature("lt_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a < b),
    _signature("le_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a <= b),
    _signature("gt_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a > b),
    _signature("ge_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a >= b),
    _signature("eq_bool", (BOOL, BOOL), BOOL, "comparison", lambda a, b: a == b),
    _signature("ne_bool", (BOOL, BOOL), BOOL, "comparison", lambda a, b: a != b),
    _signature("eq_string", (STRING, STRING), BOOL, "comparison", lambda a, b: a == b),
    _signature("ne_string", (STRING, STRING), BOOL, "comparison", lambda a, b: a != b),
    _signature("and", (BOOL, BOOL), BOOL, "boolean", lambda a, b: a and b),
    _signature("or", (BOOL, BOOL), BOOL, "boolean", lambda a, b: a or b),
    _signature("not", (BOOL,), BOOL, "boolean", lambda a: not a),
    _signature("concat", (STRING, STRING), STRING, "string", lambda a, b: a + b),
    _signature("int_to_real", (INT,), REAL, "conversion", float),
    _signature("int_to_string", (INT,), STRING, "conversion", str),
    _signature("real_to_int", (REAL,), INT, "conversion", math.trunc),
    _signature("bool_to_real", (BOOL,), REAL, "conversion", float),
    _signature("bool_to_int", (BOOL,), INT, "conversion", int),
    _signature(
        "select_real",
        (BOOL, REAL, REAL),
        REAL,
        "boolean",
        lambda c, a, b: a if c else b,
    ),
    _signature(
        "select_int", (BOOL, INT, INT), INT, "boolean", lambda c, a, b: a if c else b
    ),
    _signature("expm1", (REAL,), REAL, "math", math.expm1),
    _signature("log1p", (REAL,), REAL, "math", math.log1p),
    _signature("log2", (REAL,), REAL, "math", math.log2),
    _signature("log10", (REAL,), REAL, "math", math.log10),
    _signature("rsqrt", (REAL,), REAL, "math", lambda a: 1.0 / math.sqrt(a)),
    _signature("square", (REAL,), REAL, "math", lambda a: a * a),
    _signature("sign", (REAL,), REAL, "math", _sign),
    _signature("reciprocal", (REAL,), REAL, "math", lambda a: 1.0 / a),
    _signature("sin", (REAL,), REAL, "math", math.sin),
    _signature("cos", (REAL,), REAL, "math", math.cos),
    _signature("tan", (REAL,), REAL, "math", math.tan),
    _signature("asin", (REAL,), REAL, "math", math.asin),
    _signature("acos", (REAL,), REAL, "math", math.acos),
    _signature("atan", (REAL,), REAL, "math", math.atan),
    _signature("sinh", (REAL,), REAL, "math", math.sinh),
    _signature("cosh", (REAL,), REAL, "math", math.cosh),
    _signature("tanh", (REAL,), REAL, "math", math.tanh),
    _signature("asinh", (REAL,), REAL, "math", math.asinh),
    _signature("acosh", (REAL,), REAL, "math", math.acosh),
    _signature("atanh", (REAL,), REAL, "math", math.atanh),
    _signature("floor", (REAL,), REAL, "math", lambda a: float(math.floor(a))),
    _signature("ceil", (REAL,), REAL, "math", lambda a: float(math.ceil(a))),
    _signature("round", (REAL,), REAL, "math", lambda a: float(round(a))),
    _signature("trunc", (REAL,), REAL, "math", lambda a: float(math.trunc(a))),
    _signature("erf", (REAL,), REAL, "special", math.erf),
    _signature("erfc", (REAL,), REAL, "special", math.erfc),
    _signature("erfinv", (REAL,), REAL, "special", _erfinv),
    _signature("lgamma", (REAL,), REAL, "special", math.lgamma),
    _signature("digamma", (REAL,), REAL, "special", _digamma),
    _signature("sigmoid", (REAL,), REAL, "activation", _sigmoid),
    _signature("relu", (REAL,), REAL, "activation", lambda a: max(a, 0.0)),
    _signature("relu6", (REAL,), REAL, "activation", lambda a: min(max(a, 0.0), 6.0)),
    _signature(
        "elu", (REAL,), REAL, "activation", lambda a: a if a > 0.0 else math.expm1(a)
    ),
    _signature("selu", (REAL,), REAL, "activation", _selu),
    _signature("gelu", (REAL,), REAL, "activation", _gelu),
    _signature("silu", (REAL,), REAL, "activation", lambda a: a * _sigmoid(a)),
    _signature(
        "mish", (REAL,), REAL, "activation", lambda a: a * math.tanh(_softplus(a))
    ),
    _signature("softplus", (REAL,), REAL, "activation", _softplus),
    _signature("logsigmoid", (REAL,), REAL, "activation", lambda a: -_softplus(-a)),
    _signature("softsign", (REAL,), REAL, "activation", lambda a: a / (1.0 + abs(a))),
    _signature("observation_weight", (REAL,), REAL, "weight", _observation_weight),
    _signature("as_weight", (REAL,), LOG_WEIGHT, "weight", float),
    _signature("weight_value", (LOG_WEIGHT,), REAL, "weight", float),
    _signature(
        "add_weight", (LOG_WEIGHT, LOG_WEIGHT), LOG_WEIGHT, "weight", lambda a, b: a + b
    ),
    _signature(
        "scale_weight",
        (REAL, LOG_WEIGHT),
        LOG_WEIGHT,
        "weight",
        lambda a, b: 0.0 if a == 0 else a * b,
    ),
)

PRIMITIVES: Mapping[str, PrimitiveSignature] = MappingProxyType(
    {signature.name: signature for signature, _ in _ENTRIES}
)
"""Every primitive by name.

The mapping is closed: a name absent here is not a primitive, and no backend
may resolve one dynamically.
"""

IMPLEMENTATIONS: Mapping[str, Callable[..., object]] = MappingProxyType(
    {signature.name: implementation for signature, implementation in _ENTRIES}
)
"""The reference evaluator's implementation of every primitive, by name."""


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
    "PRIMITIVES",
    "PrimitiveCapability",
    "PrimitiveSignature",
    "primitive",
]
