"""The pure-expression builtins the QIEC lowering resolves by type.

Operators and builtin applications in ``program`` let steps and in QIEC
values resolve to primitives of the closed registry by their operand
types; reductions and last-axis operations resolve to their own terms.
The tables here are the checked, portable registry shared by QIEC lowering
and program elaboration. The eager PyTorch compiler has a deliberately larger
native extension table; names in that extension do not become checked QIEC
primitives implicitly.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from typing import Literal

import didactic.api as dx
import torch

from quivers._positions import anywhere, demand, depends_on_position
from quivers._values import HeldValue, OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.dsl.ast_nodes.let_expressions import LetBinaryOperator
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.terms import ReductionOperator, RowwiseOperator
from quivers.qiec.types import BOOL, INT, REAL, STRING, TypeExpr

_F = torch.nn.functional

#: The variance offset ``layer_norm`` adds when no ``eps`` is given, as
#: ``torch.nn.functional.layer_norm`` does.
LAYER_NORM_EPS = 1e-5

#: The offset ``rms_norm`` adds to the mean square when no ``eps`` is
#: given.
RMS_NORM_EPS = 1e-6


def _tensor_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply checked Real tensors using their common floating precision.

    Parameters
    ----------
    left, right : torch.Tensor
        Matrix operands with the existing Torch batch and event geometry.

    Returns
    -------
    torch.Tensor
        Matrix product. Floating operands use Torch's promoted dtype;
        device, layout and integral-input errors retain ordinary Torch semantics.
    """
    if (
        left.is_floating_point()
        and right.is_floating_point()
        and left.dtype != right.dtype
    ):
        dtype = torch.promote_types(left.dtype, right.dtype)
        left, right = left.to(dtype=dtype), right.to(dtype=dtype)
    return torch.matmul(left, right)


def _tensor_cholesky(value: torch.Tensor) -> torch.Tensor:
    """Factor symmetric positive-definite matrices along the last two axes.

    The lower factor ``L`` with ``L L^T = A`` costs O(n^3) for an
    ``n``-by-``n`` matrix and is differentiable to any order. Only the lower
    triangle of ``A`` is read. A matrix that is not numerically positive
    definite has no factor: its entries are NaN, so a density built from it
    is undefined and a sampler rejects the proposal, as Stan rejects a
    failed ``cholesky_decompose``.

    Parameters
    ----------
    value : torch.Tensor
        Real matrices with any leading batch axes.

    Returns
    -------
    torch.Tensor
        Lower-triangular factors of the same shape.
    """
    factor, info = torch.linalg.cholesky_ex(value)
    failed = info != 0
    if anywhere(failed):
        factor = torch.where(
            failed[..., None, None], torch.full_like(factor, torch.nan), factor
        )
    return factor


def _eager_observation_weight(value: torch.Tensor | float) -> torch.Tensor | float:
    """Check the same finite, nonnegative carrier as the reference primitive.

    Parameters
    ----------
    value : torch.Tensor | float
        A real scalar or a floating tensor of observation weights.

    Returns
    -------
    torch.Tensor | float
        The supplied value, unchanged and retaining its gradients.

    Raises
    ------
    ValueError
        If a weight is nonfinite or negative, or the tensor dtype is not real.
    """
    error = ValueError("observation weights must be finite and nonnegative real values")
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise error
        demand(torch.isfinite(value) & (value >= 0), lambda: error)
        return value
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise error
    return value


def _eager_observation_status(value: torch.Tensor | int) -> torch.Tensor | int:
    """Validate event codes while preserving their scalar or tensor carrier.

    Parameters
    ----------
    value : torch.Tensor or int
        Integer codes from minus one through two, excluding Boolean carriers.

    Returns
    -------
    torch.Tensor or int
        Unchanged validated input.

    Raises
    ------
    ValueError
        If the carrier or any event code is invalid.
    """
    if not isinstance(value, torch.Tensor):
        from quivers.qiec.primitives import _observation_status

        return _observation_status(value)
    error = ValueError("observation_status requires integer codes -1, 0, 1, or 2")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise error
    demand((value >= -1) & (value <= 2), lambda: error)
    return value


def _eager_interval_upper(
    lower: torch.Tensor | float, upper: torch.Tensor | float
) -> torch.Tensor | float:
    """Validate active real interval endpoints without changing their gradients.

    Parameters
    ----------
    lower, upper : torch.Tensor or float
        Lower endpoint, possibly infinite, and a finite strictly larger upper.

    Returns
    -------
    torch.Tensor or float
        Validated upper endpoint with its original gradient connection.

    Raises
    ------
    ValueError
        If either carrier is not real or endpoints violate the active interval.
    """
    if not isinstance(lower, torch.Tensor) and not isinstance(upper, torch.Tensor):
        from quivers.qiec.primitives import _interval_upper

        return _interval_upper(lower, upper)
    reference = lower if isinstance(lower, torch.Tensor) else upper
    assert isinstance(reference, torch.Tensor)
    endpoints = tuple(
        torch.as_tensor(
            value,
            dtype=torch.float64 if type(value) is float else None,
            device=reference.device,
        )
        for value in (lower, upper)
    )
    if any(
        value.dtype == torch.bool or not value.is_floating_point()
        for value in endpoints
    ):
        raise ValueError("interval_upper requires real endpoints")
    low, high = endpoints
    demand(
        ~low.isnan() & high.isfinite() & (high > low),
        lambda: ValueError(
            "interval_upper requires a finite upper endpoint greater than its lower endpoint"
        ),
    )
    shape = torch.broadcast_shapes(low.shape, high.shape)
    return high if high.shape == shape else high.expand(shape)


def _eager_positive_parameter(value: torch.Tensor | float) -> torch.Tensor | float:
    """Validate a positive real parameter without changing its value or gradient.

    Parameters
    ----------
    value : torch.Tensor or float
        Natural-scale scalar or tensor parameter.

    Returns
    -------
    torch.Tensor or float
        The supplied value unchanged, retaining derivative identity.

    Raises
    ------
    ValueError
        If a value is boolean, nonfinite, zero, or negative.
    """
    if not isinstance(value, torch.Tensor):
        from quivers.qiec.primitives import _positive_parameter

        return _positive_parameter(value)
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


def _eager_log_probability(value: torch.Tensor | float) -> torch.Tensor | float:
    """Validate and log a probability, masking the singular zero endpoint.

    Parameters
    ----------
    value : torch.Tensor or float
        Finite real probability in the closed unit interval.

    Returns
    -------
    torch.Tensor or float
        Logarithms, negative infinity at zero. No finite derivative is claimed
        at zero; masking prevents inactive endpoint expressions from leaking.

    Raises
    ------
    ValueError
        If a value is boolean, nonfinite, or outside [0, 1].
    """
    if not isinstance(value, torch.Tensor):
        from quivers.qiec.primitives import _log_probability

        return _log_probability(value)
    error = ValueError("log_probability requires finite real values in [0, 1]")
    if not value.is_floating_point():
        raise error
    demand(torch.isfinite(value) & (value >= 0) & (value <= 1), lambda: error)
    positive = value > 0
    safe = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, safe.log(), -torch.inf)


def _eager_log1mexp(value: torch.Tensor | float) -> torch.Tensor | float:
    """Log complementary exponentials with protected endpoint derivatives.

    Parameters
    ----------
    value : torch.Tensor or float
        Nonpositive real values, including negative infinity.

    Returns
    -------
    torch.Tensor or float
        Log(1-exp(value)); negative infinity at zero. No finite derivative is
        claimed at zero or at an infinite argument.

    Raises
    ------
    ValueError
        If a value is boolean, NaN, or positive.
    """
    if not isinstance(value, torch.Tensor):
        from quivers.qiec.primitives import _log1mexp

        return _log1mexp(value)
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


def _eager_log_ndtr(value: torch.Tensor | float) -> torch.Tensor | float:
    """Normal log CDF with protected infinite endpoint derivatives.

    Parameters
    ----------
    value : torch.Tensor or float
        Real values, including either infinity.

    Returns
    -------
    torch.Tensor or float
        Normal log CDF, negative infinity at -infinity and zero at +infinity.

    Raises
    ------
    ValueError
        If a value is boolean or NaN.
    """
    if not isinstance(value, torch.Tensor):
        from quivers.qiec.primitives import _log_ndtr

        return _log_ndtr(value)
    error = ValueError("log_ndtr requires real values other than NaN")
    if not value.is_floating_point():
        raise error
    demand(~torch.isnan(value), lambda: error)
    safe = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    result = torch.special.log_ndtr(safe)
    return torch.where(
        value.isneginf(), -torch.inf, torch.where(value.isposinf(), 0.0, result)
    )


def _eager_scale_weight(
    coefficient: torch.Tensor | float, density: torch.Tensor | float
) -> torch.Tensor | float:
    """Multiply log weights with zero absorbing even an infinite density.

    Parameters
    ----------
    coefficient : torch.Tensor | float
        The scalar coefficient, or one coefficient per density position.
    density : torch.Tensor | float
        The log density at each position.

    Returns
    -------
    torch.Tensor | float
        The product, with zero-coefficient positions masked before multiplication.
    """
    if not isinstance(coefficient, torch.Tensor) and not isinstance(
        density, torch.Tensor
    ):
        return 0.0 if coefficient == 0 else coefficient * density
    if isinstance(density, torch.Tensor):
        weight = torch.as_tensor(coefficient, device=density.device)
        value = density
    else:
        assert isinstance(coefficient, torch.Tensor)
        weight = coefficient
        value = torch.as_tensor(density, device=weight.device)
    safe = torch.where(weight == 0, torch.zeros_like(value), value)
    return weight * safe


#: Eager implementations for the deterministic-expression builtin surface.
#: Higher-order collection forms are compiled specially because they consume
#: source lambdas rather than evaluated values.
EAGER_BUILTINS: Mapping[str, Callable] = {
    "relu": lambda a: _F.relu(a),
    "relu6": lambda a: _F.relu6(a),
    "leaky_relu": lambda a, slope=0.01: _F.leaky_relu(a, negative_slope=slope),
    "prelu": lambda a, w: _F.prelu(a, w),
    "rrelu": lambda a, lower=1 / 8, upper=1 / 3: _F.rrelu(a, lower=lower, upper=upper),
    "elu": lambda a, alpha=1.0: _F.elu(a, alpha=alpha),
    "selu": lambda a: _F.selu(a),
    "celu": lambda a, alpha=1.0: _F.celu(a, alpha=alpha),
    "gelu": lambda a: _F.gelu(a),
    "silu": lambda a: _F.silu(a),
    "swish": lambda a: _F.silu(a),
    "mish": lambda a: _F.mish(a),
    "hardtanh": lambda a, lo=-1.0, hi=1.0: _F.hardtanh(a, min_val=lo, max_val=hi),
    "hardshrink": lambda a, lam=0.5: _F.hardshrink(a, lambd=lam),
    "hardsigmoid": lambda a: _F.hardsigmoid(a),
    "hardswish": lambda a: _F.hardswish(a),
    "softplus": lambda a, beta=1: _F.softplus(a, beta=beta),
    "softshrink": lambda a, lam=0.5: _F.softshrink(a, lambd=lam),
    "softsign": lambda a: _F.softsign(a),
    "softmax": lambda a: _F.softmax(a, dim=-1),
    "log_softmax": lambda a: _F.log_softmax(a, dim=-1),
    "softmin": lambda a: _F.softmin(a, dim=-1),
    "tanh": lambda a: torch.tanh(a),
    "tanhshrink": lambda a: _F.tanhshrink(a),
    "sigmoid": lambda a: torch.sigmoid(a),
    "logsigmoid": lambda a: _F.logsigmoid(a),
    "threshold": lambda a, t, v: _F.threshold(a, t, v),
    "glu": lambda a: _F.glu(a, dim=-1),
    "normalize": lambda a, p=None: (
        a / a.sum(dim=-1, keepdim=True) if p is None else _F.normalize(a, p=p, dim=-1)
    ),
    "exp": lambda a: torch.exp(a),
    "expm1": lambda a: torch.expm1(a),
    "log": lambda a: torch.log(a),
    "log_probability": _eager_log_probability,
    "positive_parameter": _eager_positive_parameter,
    "observation_status": _eager_observation_status,
    "interval_upper": _eager_interval_upper,
    "log1mexp": _eager_log1mexp,
    "log_ndtr": _eager_log_ndtr,
    "log1p": lambda a: torch.log1p(a),
    "log2": lambda a: torch.log2(a),
    "log10": lambda a: torch.log10(a),
    "sqrt": lambda a: torch.sqrt(a),
    "pow": lambda a, b: torch.pow(a, b),
    "rsqrt": lambda a: torch.rsqrt(a),
    "square": lambda a: torch.square(a),
    "abs": lambda a: torch.abs(a),
    "neg": lambda a: -a,
    "sign": lambda a: torch.sign(a),
    "reciprocal": lambda a: torch.reciprocal(a),
    "clamp": lambda a, lo, hi: torch.clamp(a, min=lo, max=hi),
    "sin": lambda a: torch.sin(a),
    "cos": lambda a: torch.cos(a),
    "tan": lambda a: torch.tan(a),
    "asin": lambda a: torch.asin(a),
    "acos": lambda a: torch.acos(a),
    "atan": lambda a: torch.atan(a),
    "sinh": lambda a: torch.sinh(a),
    "cosh": lambda a: torch.cosh(a),
    "asinh": lambda a: torch.asinh(a),
    "acosh": lambda a: torch.acosh(a),
    "atanh": lambda a: torch.atanh(a),
    "floor": lambda a: torch.floor(a),
    "ceil": lambda a: torch.ceil(a),
    "round": lambda a: torch.round(a),
    "trunc": lambda a: torch.trunc(a),
    "erf": lambda a: torch.erf(a),
    "erfc": lambda a: torch.erfc(a),
    "erfinv": lambda a: torch.erfinv(a),
    "lgamma": lambda a: torch.lgamma(a),
    "digamma": lambda a: torch.digamma(a),
    "sum": lambda a: torch.sum(a, dim=-1),
    "mean": lambda a: torch.mean(a, dim=-1),
    "var": lambda a: torch.var(a, dim=-1),
    "std": lambda a: torch.std(a, dim=-1),
    "min": lambda a, b=None: (
        torch.min(a, dim=-1).values if b is None else torch.minimum(a, b)
    ),
    "max": lambda a, b=None: (
        torch.max(a, dim=-1).values if b is None else torch.maximum(a, b)
    ),
    "argmin": lambda a: torch.argmin(a, dim=-1),
    "argmax": lambda a: torch.argmax(a, dim=-1),
    "prod": lambda a: torch.prod(a, dim=-1),
    "amax": lambda a: torch.amax(a, dim=-1),
    "amin": lambda a: torch.amin(a, dim=-1),
    "logsumexp": lambda a: torch.logsumexp(a, dim=-1),
    "norm": lambda a, p=2.0: torch.linalg.vector_norm(a, ord=p, dim=-1),
    "cumsum": lambda a: torch.cumsum(a, dim=-1),
    "cumprod": lambda a: torch.cumprod(a, dim=-1),
    "cummax": lambda a: torch.cummax(a, dim=-1).values,
    "cummin": lambda a: torch.cummin(a, dim=-1).values,
    "flip": lambda a: torch.flip(a, dims=(-1,)),
    "sort": lambda a: torch.sort(a, dim=-1).values,
    "dropout": lambda a, p=0.5: _F.dropout(a, p=p, training=True),
    "alpha_dropout": lambda a, p=0.5: _F.alpha_dropout(a, p=p, training=True),
    "layer_norm": lambda a, eps=LAYER_NORM_EPS: _F.layer_norm(
        a, normalized_shape=(a.shape[-1],), eps=float(eps)
    ),
    "rms_norm": lambda a, eps=RMS_NORM_EPS: (
        a * torch.rsqrt(a.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    ),
    "matmul": _tensor_matmul,
    "cholesky": _tensor_cholesky,
    "real": lambda a: (
        a.to(dtype=torch.get_default_dtype())
        if isinstance(a, torch.Tensor)
        else float(a)
    ),
    "int": lambda a: a.to(dtype=torch.int64) if isinstance(a, torch.Tensor) else int(a),
    "select": lambda cond, a, b: torch.where(
        torch.as_tensor(cond, dtype=torch.bool),
        torch.as_tensor(a),
        torch.as_tensor(b),
    ),
    "weight": lambda a: a,
    "weight_value": lambda a: a,
    "observation_weight": _eager_observation_weight,
    "scale_weight": _eager_scale_weight,
    "concat": lambda *parts: torch.cat(parts, dim=-1),
    "slice": lambda a, start, length: a[..., int(start) : int(start) + int(length)],
}

#: Operator and operand type to the primitive that implements it.
_BINARY_PRIMITIVES: Mapping[LetBinaryOperator, Mapping[TypeExpr, str]] = {
    "+": {INT: "add_int", REAL: "add_real", STRING: "concat"},
    "-": {INT: "sub_int", REAL: "sub_real"},
    "*": {INT: "mul_int", REAL: "mul_real"},
    "/": {INT: "div_int", REAL: "div_real"},
    "%": {INT: "mod_int"},
    "==": {INT: "eq_int", REAL: "eq_real", BOOL: "eq_bool", STRING: "eq_string"},
    "!=": {INT: "ne_int", REAL: "ne_real", BOOL: "ne_bool", STRING: "ne_string"},
    "<": {INT: "lt_int", REAL: "lt_real"},
    "<=": {INT: "le_int", REAL: "le_real"},
    ">": {INT: "gt_int", REAL: "gt_real"},
    ">=": {INT: "ge_int", REAL: "ge_real"},
    "&&": {BOOL: "and"},
    "||": {BOOL: "or"},
}

#: Builtin name and argument types to the primitive that implements it.
_BUILTIN_PRIMITIVES: Mapping[str, Mapping[tuple[TypeExpr, ...], str]] = {
    "real": {(INT,): "int_to_real", (BOOL,): "bool_to_real"},
    "int": {(REAL,): "real_to_int", (BOOL,): "bool_to_int"},
    "select": {
        (BOOL, REAL, REAL): "select_real",
        (BOOL, INT, INT): "select_int",
    },
    "exp": {(REAL,): "exp"},
    "log": {(REAL,): "log"},
    "log_probability": {(REAL,): "log_probability"},
    "positive_parameter": {(REAL,): "positive_parameter"},
    "observation_status": {(INT,): "observation_status"},
    "interval_upper": {(REAL, REAL): "interval_upper"},
    "log1mexp": {(REAL,): "log1mexp"},
    "log_ndtr": {(REAL,): "log_ndtr"},
    "sqrt": {(REAL,): "sqrt"},
    "pow": {(REAL, REAL): "pow_real"},
    "abs": {(INT,): "abs_int", (REAL,): "abs_real"},
    "min": {(INT, INT): "min_int", (REAL, REAL): "min_real"},
    "max": {(INT, INT): "max_int", (REAL, REAL): "max_real"},
    "weight": {(REAL,): "as_weight"},
    "weight_value": {(LOG_WEIGHT,): "weight_value"},
    "observation_weight": {(REAL,): "observation_weight"},
    "scale_weight": {(REAL, LOG_WEIGHT): "scale_weight"},
    **{
        name: {(REAL,): name}
        for name in (
            "expm1",
            "log1p",
            "log2",
            "log10",
            "rsqrt",
            "square",
            "sign",
            "reciprocal",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "floor",
            "ceil",
            "round",
            "trunc",
            "erf",
            "erfc",
            "erfinv",
            "lgamma",
            "digamma",
            "sigmoid",
            "relu",
            "relu6",
            "elu",
            "selu",
            "gelu",
            "silu",
            "mish",
            "softplus",
            "logsigmoid",
            "softsign",
        )
    },
}

#: Builtin names that collapse the last axis of a tensor: a vector reduces
#: to one number, and a tensor of higher rank to the tensor of its leading
#: axes. The checked lowering spells this as one whole-vector `Reduction`
#: per row, the eager runtime reduces ``dim=-1``, and every renderer
#: reduces the same axis.
_REDUCTIONS: Mapping[str, ReductionOperator] = {
    "sum": "sum",
    "mean": "mean",
    "max": "max",
    "min": "min",
    "logsumexp": "logsumexp",
    "prod": "prod",
}

#: Builtin names that act along a tensor's last axis, keeping its shape.
#: ``normalize`` divides each row by its sum, onto the simplex.
_ROWWISE: Mapping[str, RowwiseOperator] = {
    "softmax": "softmax",
    "log_softmax": "log_softmax",
    "cumsum": "cumsum",
    "sort": "sort",
    "normalize": "normalize",
}

#: Builtins that rebuild the last axis of their tensor arguments, which
#: the checked lowering writes out in terms of selections, tensor
#: literals, comprehensions, and other builtins: ``concat`` joins tensors
#: along it and ``slice`` keeps a window of it (a tensor of entries
#: chosen at literal positions under one comprehension per leading
#: axis); ``matmul`` contracts it against the first axis of a vector or
#: matrix; ``layer_norm`` and ``rms_norm`` normalize each row of it.
_SHAPE_BUILTINS: Mapping[str, frozenset[int]] = {
    "concat": frozenset(range(2, 65)),
    "slice": frozenset({3}),
    "matmul": frozenset({2}),
    "layer_norm": frozenset({1, 2}),
    "rms_norm": frozenset({1, 2}),
}

#: Builtins that act on the last two axes of a tensor of matrices, keeping
#: its shape. The checked lowering spells each as one opaque matrix term,
#: as a factorization has no finite expression in selections and sums.
_MATRIX_FUNCTIONS: frozenset[str] = frozenset({"cholesky"})

#: Higher-order operations whose types are determined by their collection and
#: lambda arguments rather than by one first-order primitive signature.
_COLLECTION_BUILTINS: frozenset[str] = frozenset(
    {"map", "fold", "length", "logsumexp_over", "filter"}
)


type QiecBuiltinForm = Literal[
    "primitive",
    "reduction",
    "rowwise",
    "shape",
    "matrix",
    "collection",
    "unsupported",
    "host-only",
    "measure-query",
]


class BuiltinSpec(QiecModel):
    """One source builtin's eager and checked capabilities.

    ``qiec_form`` records how checked lowering consumes the call. ``filter``
    deliberately has the ``unsupported`` form: it is a known surface operation
    with a stable dynamic-shape diagnostic, not an accidentally unknown name.

    Eager callables and overload registries retain their runtime identities.
    Arity sets, operator tags, and supported targets compare by value.
    """

    name: str
    eager: Callable[..., HeldValue] | None = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    eager_arities: frozenset[int] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    qiec_arities: frozenset[int] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    purity: Literal["pure", "random"]
    shape_rule: str
    qiec_form: QiecBuiltinForm
    qiec_overloads: Mapping[tuple[TypeExpr, ...], str] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    qiec_operator: ReductionOperator | RowwiseOperator | None = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    targets: frozenset[str] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


_COLLECTION_ARITIES: Mapping[str, frozenset[int]] = {
    "map": frozenset({2}),
    "fold": frozenset({3}),
    "length": frozenset({1}),
    "logsumexp_over": frozenset({2}),
    "filter": frozenset({2}),
}
#: Structural queries use checked sampleable arguments rather than direct
#: tensor-call implementations. Their arity and result layout stay together.
_MEASURE_QUERIES: Mapping[str, tuple[int, str]] = {
    "log_mass": (1, "sampleable-total-mass"),
    "log_prob": (2, "sampleable-density"),
}
_RANDOM_EAGER_BUILTINS = frozenset({"alpha_dropout", "dropout", "rrelu"})


def _eager_arities(implementation: Callable | None) -> frozenset[int]:
    if implementation is None:
        return frozenset()
    signature = inspect.signature(implementation)
    positional = tuple(
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    )
    required = sum(parameter.default is parameter.empty for parameter in positional)
    return frozenset(range(required, len(positional) + 1))


def _qiec_form(name: str) -> QiecBuiltinForm:
    if name in _MEASURE_QUERIES:
        return "measure-query"
    if name in _BUILTIN_PRIMITIVES:
        return "primitive"
    if name in _REDUCTIONS:
        return "reduction"
    if name in _ROWWISE:
        return "rowwise"
    if name in _SHAPE_BUILTINS:
        return "shape"
    if name in _MATRIX_FUNCTIONS:
        return "matrix"
    if name == "filter":
        return "unsupported"
    if name in _COLLECTION_BUILTINS:
        return "collection"
    return "host-only"


def _qiec_arities(name: str, form: QiecBuiltinForm) -> frozenset[int]:
    if form == "measure-query":
        return frozenset({_MEASURE_QUERIES[name][0]})
    if form == "primitive":
        return frozenset(len(signature) for signature in _BUILTIN_PRIMITIVES[name])
    if form in ("reduction", "rowwise"):
        arities = {1}
        if name in _BUILTIN_PRIMITIVES:
            arities.update(len(signature) for signature in _BUILTIN_PRIMITIVES[name])
        return frozenset(arities)
    if form == "shape":
        return _SHAPE_BUILTINS[name]
    if form == "matrix":
        return frozenset({1})
    if name in _COLLECTION_ARITIES:
        return _COLLECTION_ARITIES[name]
    return frozenset()


def _shape_rule(name: str, form: QiecBuiltinForm) -> str:
    if form == "measure-query":
        return _MEASURE_QUERIES[name][1]
    if form == "primitive":
        return "scalar-lift"
    if form == "reduction":
        return "last-axis-reduction"
    if form == "rowwise":
        return "last-axis-transform"
    if form == "shape":
        return "last-axis-rebuild"
    if form == "matrix":
        return "last-two-axes-transform"
    if name == "length":
        return "leading-extent"
    if name == "filter":
        return "dynamic-leading-extent"
    if form == "collection":
        return "finite-leading-axis"
    return "eager-native"


_BUILTIN_NAMES = frozenset((*EAGER_BUILTINS, *_COLLECTION_BUILTINS, *_MEASURE_QUERIES))

#: The one capability registry consulted by eager execution, checked lowering,
#: editor completion, and transpiler validation.
BUILTIN_REGISTRY: Mapping[str, BuiltinSpec] = {
    name: BuiltinSpec(
        name=name,
        eager=EAGER_BUILTINS.get(name),
        eager_arities=(
            _COLLECTION_ARITIES[name]
            if name in _COLLECTION_ARITIES
            else _SHAPE_BUILTINS[name]
            if name in _SHAPE_BUILTINS
            else _eager_arities(EAGER_BUILTINS.get(name))
        ),
        qiec_arities=_qiec_arities(name, form := _qiec_form(name)),
        purity="random" if name in _RANDOM_EAGER_BUILTINS else "pure",
        shape_rule=_shape_rule(name, form),
        qiec_form=form,
        qiec_overloads=_BUILTIN_PRIMITIVES.get(name, {}),
        qiec_operator=_REDUCTIONS.get(name) or _ROWWISE.get(name),
        targets=(
            frozenset({"eager", "qiec"})
            if form not in ("host-only", "unsupported")
            else frozenset({"eager"})
        ),
    )
    for name in sorted(_BUILTIN_NAMES)
}

#: Every name the eager let-expression evaluator implements directly.
EAGER_BUILTIN_NAMES: frozenset[str] = frozenset(EAGER_BUILTINS)

#: Builtins intentionally available only to native eager execution.
HOST_ONLY_BUILTINS: frozenset[str] = frozenset(
    name for name, spec in BUILTIN_REGISTRY.items() if spec.qiec_form == "host-only"
)

#: Every checked name a let expression may apply as a builtin. This includes
#: ``filter`` so checked tools can issue its specific dynamic-shape diagnostic.
PURE_BUILTINS: frozenset[str] = frozenset(
    name for name, spec in BUILTIN_REGISTRY.items() if spec.qiec_form != "host-only"
)

#: The operator each binary primitive reads back as.
PRIMITIVE_OPERATORS: Mapping[str, LetBinaryOperator] = {
    primitive: operator
    for operator, by_type in _BINARY_PRIMITIVES.items()
    for primitive in by_type.values()
}

#: The builtins that rebuild the last axis of their arguments.
SHAPE_BUILTINS: frozenset[str] = frozenset(_SHAPE_BUILTINS)

#: The builtins that transform the matrices in a tensor's last two axes.
MATRIX_FUNCTIONS: frozenset[str] = _MATRIX_FUNCTIONS

#: The builtin each call primitive reads back as.
PRIMITIVE_BUILTINS: Mapping[str, str] = {
    primitive: builtin
    for builtin, by_types in _BUILTIN_PRIMITIVES.items()
    for primitive in by_types.values()
}


__all__ = []
