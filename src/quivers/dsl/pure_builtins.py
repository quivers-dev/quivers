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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from quivers.dsl.ast_nodes.let_expressions import LetBinaryOperator
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.terms import ReductionOperator, RowwiseOperator
from quivers.qiec.types import BOOL, INT, REAL, STRING, TypeExpr

_F = torch.nn.functional

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
    "layer_norm": lambda a: _F.layer_norm(a, normalized_shape=(a.shape[-1],)),
    "rms_norm": lambda a: a * torch.rsqrt(a.pow(2).mean(dim=-1, keepdim=True) + 1e-6),
    "real": lambda a: (
        a.to(dtype=torch.get_default_dtype())
        if isinstance(a, torch.Tensor)
        else float(a)
    ),
    "int": lambda a: a.to(dtype=torch.int64) if isinstance(a, torch.Tensor) else int(a),
    "weight": lambda a: a,
    "weight_value": lambda a: a,
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
    "real": {(INT,): "int_to_real"},
    "int": {(REAL,): "real_to_int"},
    "exp": {(REAL,): "exp"},
    "log": {(REAL,): "log"},
    "sqrt": {(REAL,): "sqrt"},
    "pow": {(REAL, REAL): "pow_real"},
    "abs": {(INT,): "abs_int", (REAL,): "abs_real"},
    "min": {(INT, INT): "min_int", (REAL, REAL): "min_real"},
    "max": {(INT, INT): "max_int", (REAL, REAL): "max_real"},
    "weight": {(REAL,): "as_weight"},
    "weight_value": {(LOG_WEIGHT,): "weight_value"},
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

#: Higher-order operations whose types are determined by their collection and
#: lambda arguments rather than by one first-order primitive signature.
_COLLECTION_BUILTINS: frozenset[str] = frozenset(
    {"map", "fold", "length", "logsumexp_over", "filter"}
)


type QiecBuiltinForm = Literal[
    "primitive", "reduction", "rowwise", "collection", "unsupported", "host-only"
]


@dataclass(frozen=True, slots=True)
class BuiltinSpec:
    """One source builtin's eager and checked capabilities.

    ``qiec_form`` records how checked lowering consumes the call. ``filter``
    deliberately has the ``unsupported`` form: it is a known surface operation
    with a stable dynamic-shape diagnostic, not an accidentally unknown name.
    """

    name: str
    eager: Callable | None
    eager_arities: frozenset[int]
    qiec_arities: frozenset[int]
    purity: Literal["pure", "random"]
    shape_rule: str
    qiec_form: QiecBuiltinForm
    qiec_overloads: Mapping[tuple[TypeExpr, ...], str]
    qiec_operator: ReductionOperator | RowwiseOperator | None
    targets: frozenset[str]


_COLLECTION_ARITIES: Mapping[str, frozenset[int]] = {
    "map": frozenset({2}),
    "fold": frozenset({3}),
    "length": frozenset({1}),
    "logsumexp_over": frozenset({2}),
    "filter": frozenset({2}),
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
    if name in _BUILTIN_PRIMITIVES:
        return "primitive"
    if name in _REDUCTIONS:
        return "reduction"
    if name in _ROWWISE:
        return "rowwise"
    if name == "filter":
        return "unsupported"
    if name in _COLLECTION_BUILTINS:
        return "collection"
    return "host-only"


def _qiec_arities(name: str, form: QiecBuiltinForm) -> frozenset[int]:
    if form == "primitive":
        return frozenset(len(signature) for signature in _BUILTIN_PRIMITIVES[name])
    if form in ("reduction", "rowwise"):
        arities = {1}
        if name in _BUILTIN_PRIMITIVES:
            arities.update(len(signature) for signature in _BUILTIN_PRIMITIVES[name])
        return frozenset(arities)
    if name in _COLLECTION_ARITIES:
        return _COLLECTION_ARITIES[name]
    return frozenset()


def _shape_rule(name: str, form: QiecBuiltinForm) -> str:
    if form == "primitive":
        return "scalar-lift"
    if form == "reduction":
        return "last-axis-reduction"
    if form == "rowwise":
        return "last-axis-transform"
    if name == "length":
        return "leading-extent"
    if name == "filter":
        return "dynamic-leading-extent"
    if form == "collection":
        return "finite-leading-axis"
    return "eager-native"


_BUILTIN_NAMES = frozenset((*EAGER_BUILTINS, *_COLLECTION_BUILTINS))

#: The one capability registry consulted by eager execution, checked lowering,
#: editor completion, and transpiler validation.
BUILTIN_REGISTRY: Mapping[str, BuiltinSpec] = {
    name: BuiltinSpec(
        name=name,
        eager=EAGER_BUILTINS.get(name),
        eager_arities=(
            _COLLECTION_ARITIES[name]
            if name in _COLLECTION_ARITIES
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

#: The builtin each call primitive reads back as.
PRIMITIVE_BUILTINS: Mapping[str, str] = {
    primitive: builtin
    for builtin, by_types in _BUILTIN_PRIMITIVES.items()
    for primitive in by_types.values()
}


__all__ = []
