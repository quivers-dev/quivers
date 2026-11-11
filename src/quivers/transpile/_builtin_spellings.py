"""Each target's spelling of every surface math builtin.

A let expression applies a math builtin (``exp``, ``erf``, ``gelu``, ...)
by its QVR name, and no target spells every one of them under that
name: Stan writes the logistic sigmoid ``inv_logit``, JAGS has no
``floor``, Chez Scheme no ``erf``, and WebPPL's own ``tanh`` acts on
tensors rather than numbers. This module holds one table, keyed by
target, that settles each builtin for each target in one of four ways:

* [`NativeCall`][quivers.transpile._builtin_spellings.NativeCall]: the
  target calls a function of its own, named by a dotted path.
* [`ClosedForm`][quivers.transpile._builtin_spellings.ClosedForm]: the
  builtin is written in terms of arithmetic and other builtins, each of
  which the same table spells in turn.
* [`Dedicated`][quivers.transpile._builtin_spellings.Dedicated]: the
  target's renderer spells the builtin with code of its own (the
  pointwise ``max`` / ``min`` / ``pow``, whose spelling depends on the
  operands).
* [`Refusal`][quivers.transpile._builtin_spellings.Refusal]: the
  target cannot express the builtin, and rendering it raises
  [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
  naming the builtin and the target.

Every target's table has an entry for every surface builtin, so adding
a builtin to the registry forces a decision for every target. The
planner expands closed forms and raises refusals before any renderer
reads the expression ([`spell_builtin`][quivers.transpile._builtin_spellings.spell_builtin]),
and a renderer asks [`native_path`][quivers.transpile._builtin_spellings.native_path]
for the name it emits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprCall,
    LetExprLiteral,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.dsl.ast_nodes.let_expressions import LetBinaryOperator
from quivers.dsl.pure_builtins import BUILTIN_REGISTRY
from quivers.transpile._api import UnsupportedConstruct

#: The pointwise math builtins a let expression may apply: every scalar
#: primitive of the builtin registry except the conversions between
#: integers, reals, and weights, which the planner reads structurally.
MATH_BUILTINS: Mapping[str, int] = {
    name: min(spec.qiec_arities)
    for name, spec in sorted(BUILTIN_REGISTRY.items())
    if spec.qiec_form == "primitive"
    and name not in {"int", "real", "weight", "weight_value"}
}


#: The planner's target-neutral lowering, which no renderer reads: it
#: keeps every builtin under its QVR name.
NEUTRAL_TARGET = "ir"


class BuiltinSpelling(dx.TaggedUnion, discriminator="kind"):
    """How one target spells one math builtin."""


class NativeCall(BuiltinSpelling):
    """A call to a function the target provides.

    Parameters
    ----------
    path
        The function's dotted path, outermost name first.
    import_path
        A module the call needs imported, or ``()`` when the path is in
        scope already.
    import_alias
        The alias ``import_path`` is imported under.
    kind
        The discriminator; always ``"native"``.
    """

    path: tuple[str, ...]
    import_path: tuple[str, ...] = ()
    import_alias: str = ""
    kind: Literal["native"] = "native"


class ClosedForm(BuiltinSpelling):
    """The builtin written in terms of arithmetic and other builtins.

    Parameters
    ----------
    body
        The definition, reading its arguments as the variables
        ``__arg0`` and ``__arg1``. A call to a math builtin inside it is
        spelled by the same target's table in turn; a call to any other
        name is a function of the target's own.
    kind
        The discriminator; always ``"closed_form"``.
    """

    body: LetExprNode
    kind: Literal["closed_form"] = "closed_form"


class Dedicated(BuiltinSpelling):
    """The target's renderer spells the builtin with code of its own.

    Parameters
    ----------
    reason
        What the renderer does and why a fixed name does not suffice.
    kind
        The discriminator; always ``"dedicated"``.
    """

    reason: str
    kind: Literal["dedicated"] = "dedicated"


class Refusal(BuiltinSpelling):
    """The target cannot express the builtin.

    Parameters
    ----------
    reason
        Why not, quoted in the diagnostic.
    kind
        The discriminator; always ``"refusal"``.
    """

    reason: str
    kind: Literal["refusal"] = "refusal"


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------

_X = LetExprVar(name="__arg0")
_Y = LetExprVar(name="__arg1")


def _lit(value: float) -> LetExprLiteral:
    return LetExprLiteral(value=value)


def _call(func: str, *args: LetExprNode) -> LetExprCall:
    return LetExprCall(func=func, args=args)


def _op(op: LetBinaryOperator, left: LetExprNode, right: LetExprNode) -> LetExprBinOp:
    return LetExprBinOp(op=op, left=left, right=right)


def _neg(operand: LetExprNode) -> LetExprUnaryOp:
    return LetExprUnaryOp(operand=operand)


def _native(*path: str) -> NativeCall:
    return NativeCall(path=path)


def _form(body: LetExprNode) -> ClosedForm:
    return ClosedForm(body=body)


_SQRT_HALF = 0.7071067811865476
_LN2 = 0.6931471805599453
_LN10 = 2.302585092994046
_SELU_SCALE = 1.0507009873554804934193349852946
_SELU_ALPHA = 1.6732632423543772848170429916717


def _half_even(floor_name: str, step_name: str) -> LetExprNode:
    """Round half to even from a floor and a step function.

    ``r = floor(x + 0.5)`` rounds a tie up; the tie is the case
    ``x + 0.5 - r == 0``, which ``step(-(x + 0.5 - r))`` detects for a
    step that is one at zero, and an odd ``r`` there moves down by one:
    ``r - tie * (r - 2 * floor(r / 2))``.
    """
    shifted = _op("+", _X, _lit(0.5))
    up = _call(floor_name, shifted)
    tie = _call(step_name, _neg(_op("-", shifted, up)))
    parity = _op(
        "-", up, _op("*", _lit(2.0), _call(floor_name, _op("/", up, _lit(2.0))))
    )
    return _op("-", up, _op("*", tie, parity))


#: Definitions in terms of other builtins, shared by every target that
#: lacks a function of its own for the builtin.
_COMMON_FORMS: dict[str, LetExprNode] = {
    "reciprocal": _op("/", _lit(1.0), _X),
    "square": _op("*", _X, _X),
    "rsqrt": _op("/", _lit(1.0), _call("sqrt", _X)),
    "expm1": _op("-", _call("exp", _X), _lit(1.0)),
    "log1p": _call("log", _op("+", _lit(1.0), _X)),
    "log2": _op("/", _call("log", _X), _lit(_LN2)),
    "log10": _op("/", _call("log", _X), _lit(_LN10)),
    "sinh": _op("/", _op("-", _call("exp", _X), _call("exp", _neg(_X))), _lit(2.0)),
    "cosh": _op("/", _op("+", _call("exp", _X), _call("exp", _neg(_X))), _lit(2.0)),
    # `1 - 2 / (exp(2x) + 1)` stays finite where `sinh / cosh` overflows.
    "tanh": _op(
        "-",
        _lit(1.0),
        _op("/", _lit(2.0), _op("+", _call("exp", _op("*", _lit(2.0), _X)), _lit(1.0))),
    ),
    "asinh": _call(
        "log", _op("+", _X, _call("sqrt", _op("+", _op("*", _X, _X), _lit(1.0))))
    ),
    "acosh": _call(
        "log", _op("+", _X, _call("sqrt", _op("-", _op("*", _X, _X), _lit(1.0))))
    ),
    "atanh": _op(
        "*",
        _lit(0.5),
        _call("log", _op("/", _op("+", _lit(1.0), _X), _op("-", _lit(1.0), _X))),
    ),
    "ceil": _neg(_call("floor", _neg(_X))),
    "erfc": _op("-", _lit(1.0), _call("erf", _X)),
    "sigmoid": _op("/", _lit(1.0), _op("+", _lit(1.0), _call("exp", _neg(_X)))),
    # `max(x, 0) + log1p(exp(-|x|))` is `log(1 + exp(x))` without overflow.
    "softplus": _op(
        "+",
        _call("max", _X, _lit(0.0)),
        _call("log1p", _call("exp", _neg(_call("abs", _X)))),
    ),
    "logsigmoid": _op(
        "-",
        _call("min", _X, _lit(0.0)),
        _call("log1p", _call("exp", _neg(_call("abs", _X)))),
    ),
    "relu": _call("max", _X, _lit(0.0)),
    "relu6": _call("min", _call("max", _X, _lit(0.0)), _lit(6.0)),
    "elu": _op(
        "+", _call("max", _X, _lit(0.0)), _call("min", _call("expm1", _X), _lit(0.0))
    ),
    "selu": _op(
        "*",
        _lit(_SELU_SCALE),
        _op(
            "+",
            _call("max", _X, _lit(0.0)),
            _op("*", _lit(_SELU_ALPHA), _call("min", _call("expm1", _X), _lit(0.0))),
        ),
    ),
    "silu": _op("*", _X, _call("sigmoid", _X)),
    "mish": _op("*", _X, _call("tanh", _call("softplus", _X))),
    "softsign": _op("/", _X, _op("+", _lit(1.0), _call("abs", _X))),
    # The exact GELU, `x * Phi(x)`, which is what the reference computes.
    "gelu": _op(
        "*",
        _op("*", _lit(0.5), _X),
        _op("+", _lit(1.0), _call("erf", _op("*", _X, _lit(_SQRT_HALF)))),
    ),
}


def _common(*names: str) -> dict[str, BuiltinSpelling]:
    return {name: _form(_COMMON_FORMS[name]) for name in names}


def _same_name(*names: str) -> dict[str, BuiltinSpelling]:
    return {name: _native(name) for name in names}


def _refuse(reason: str, *names: str) -> dict[str, BuiltinSpelling]:
    return {name: Refusal(reason=reason) for name in names}


_ACTIVATIONS = ("relu", "relu6", "elu", "selu", "silu", "mish", "softsign", "gelu")


# ---------------------------------------------------------------------------
# Python-family targets
# ---------------------------------------------------------------------------

_JAX_SPECIAL = (("jax", "scipy", "special"), "jsp")
_JAX_NN = (("jax", "nn"), "jnn")


def _imported(module: tuple[tuple[str, ...], str], name: str) -> NativeCall:
    path, alias = module
    return NativeCall(path=(alias, name), import_path=path, import_alias=alias)


_TORCH_FUNCTIONAL = (
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

_PYRO: dict[str, BuiltinSpelling] = {
    **{
        name: _native("torch", name)
        for name in MATH_BUILTINS
        if name not in _TORCH_FUNCTIONAL and name not in {"max", "min", "pow"}
    },
    **{name: _native("torch", "nn", "functional", name) for name in _TORCH_FUNCTIONAL},
    "max": _native("torch", "maximum"),
    "min": _native("torch", "minimum"),
    "pow": _native("torch", "pow"),
}

_NUMPYRO: dict[str, BuiltinSpelling] = {
    **{
        name: _native("jnp", name)
        for name in (
            "exp",
            "expm1",
            "log",
            "log1p",
            "log2",
            "log10",
            "sqrt",
            "square",
            "abs",
            "sign",
            "reciprocal",
            "sin",
            "cos",
            "tan",
            "sinh",
            "cosh",
            "floor",
            "ceil",
            "round",
            "trunc",
            "tanh",
        )
    },
    "asin": _native("jnp", "arcsin"),
    "acos": _native("jnp", "arccos"),
    "atan": _native("jnp", "arctan"),
    "asinh": _native("jnp", "arcsinh"),
    "acosh": _native("jnp", "arccosh"),
    "atanh": _native("jnp", "arctanh"),
    "erf": _imported(_JAX_SPECIAL, "erf"),
    "erfc": _imported(_JAX_SPECIAL, "erfc"),
    "erfinv": _imported(_JAX_SPECIAL, "erfinv"),
    "lgamma": _imported(_JAX_SPECIAL, "gammaln"),
    "digamma": _imported(_JAX_SPECIAL, "digamma"),
    "sigmoid": _imported(_JAX_NN, "sigmoid"),
    "relu": _imported(_JAX_NN, "relu"),
    "elu": _imported(_JAX_NN, "elu"),
    "selu": _imported(_JAX_NN, "selu"),
    # `jax.nn.gelu` defaults to the tanh approximation; the exact form
    # is written out instead.
    **_common("gelu", "relu6", "mish", "rsqrt"),
    "silu": _imported(_JAX_NN, "silu"),
    "softplus": _imported(_JAX_NN, "softplus"),
    "logsigmoid": _imported(_JAX_NN, "log_sigmoid"),
    "softsign": _imported(_JAX_NN, "soft_sign"),
    "max": _native("jnp", "maximum"),
    "min": _native("jnp", "minimum"),
    "pow": _native("jnp", "power"),
}


def _pmath(name: str) -> NativeCall:
    return _native("pymc", "math", name)


def _pt(name: str) -> NativeCall:
    return _native("pymc", "pytensorf", "pt", name)


_PYMC: dict[str, BuiltinSpelling] = {
    **{
        name: _pmath(name)
        for name in (
            "exp",
            "expm1",
            "log",
            "log1p",
            "log2",
            "log10",
            "sqrt",
            "abs",
            "sin",
            "cos",
            "tan",
            "sinh",
            "cosh",
            "tanh",
            "floor",
            "ceil",
            "erf",
            "erfc",
            "erfinv",
            "sigmoid",
            "softplus",
        )
    },
    "asin": _pt("arcsin"),
    "acos": _pt("arccos"),
    "atan": _pt("arctan"),
    "asinh": _pt("arcsinh"),
    "acosh": _pt("arccosh"),
    "atanh": _pt("arctanh"),
    "lgamma": _pt("gammaln"),
    "digamma": _pt("psi"),
    "reciprocal": _pt("reciprocal"),
    "square": _pt("sqr"),
    "sign": _pt("sgn"),
    "trunc": _pt("trunc"),
    "round": _pt("round_half_to_even"),
    **_common("rsqrt", "logsigmoid", *_ACTIVATIONS),
    "max": _pmath("maximum"),
    "min": _pmath("minimum"),
    "pow": _pt("pow"),
}


def _tfmath(name: str) -> NativeCall:
    return _native("tf", "math", name)


def _tfnn(name: str) -> NativeCall:
    return _native("tf", "nn", name)


_EDWARD2: dict[str, BuiltinSpelling] = {
    **{
        name: _tfmath(name)
        for name in (
            "exp",
            "expm1",
            "log",
            "log1p",
            "sqrt",
            "rsqrt",
            "square",
            "abs",
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
            "asinh",
            "acosh",
            "atanh",
            "floor",
            "ceil",
            "round",
            "tanh",
            "sigmoid",
            "erf",
            "erfc",
            "erfinv",
            "lgamma",
            "digamma",
        )
    },
    "logsigmoid": _tfmath("log_sigmoid"),
    **{
        name: _tfnn(name)
        for name in ("relu", "elu", "selu", "gelu", "silu", "softplus", "softsign")
    },
    **_common("log2", "log10", "relu6", "mish"),
    # TensorFlow's `trunc` is `tf.truncatediv`'s cousin for integers; a
    # real truncates as the sign times the floor of the magnitude.
    "trunc": _form(_op("*", _call("sign", _X), _call("floor", _call("abs", _X)))),
    "max": _tfmath("maximum"),
    "min": _tfmath("minimum"),
    "pow": _tfmath("pow"),
}


# ---------------------------------------------------------------------------
# Stan
# ---------------------------------------------------------------------------

_STAN: dict[str, BuiltinSpelling] = {
    **_same_name(
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "ceil",
        "cos",
        "cosh",
        "digamma",
        "erf",
        "erfc",
        "exp",
        "expm1",
        "floor",
        "lgamma",
        "log",
        "log10",
        "log1p",
        "log2",
        "sin",
        "sinh",
        "sqrt",
        "square",
        "tan",
        "tanh",
        "trunc",
        "pow",
    ),
    "sigmoid": _native("inv_logit"),
    "logsigmoid": _native("log_inv_logit"),
    "softplus": _native("log1p_exp"),
    "reciprocal": _native("inv"),
    "rsqrt": _native("inv_sqrt"),
    # `erfinv(y) = inv_Phi((y + 1) / 2) / sqrt(2)`.
    "erfinv": _form(
        _op(
            "*",
            _call("inv_Phi", _op("/", _op("+", _X, _lit(1.0)), _lit(2.0))),
            _lit(_SQRT_HALF),
        )
    ),
    # Stan's `round` rounds a tie away from zero; the reference rounds
    # it to even. Stan's `step` is one at zero.
    "round": _form(_half_even("floor", "step")),
    "sign": _form(_op("-", _call("step", _X), _call("step", _neg(_X)))),
    **_common(*_ACTIVATIONS),
    "max": Dedicated(reason="`max` of two integers, `fmax` of reals"),
    "min": Dedicated(reason="`min` of two integers, `fmin` of reals"),
}


# ---------------------------------------------------------------------------
# Julia (Turing, Gen)
# ---------------------------------------------------------------------------


def _julia(root: str) -> dict[str, BuiltinSpelling]:
    special = (root, "Distributions", "SpecialFunctions")
    return {
        **_same_name(
            "abs",
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "ceil",
            "cos",
            "cosh",
            "exp",
            "expm1",
            "floor",
            "log",
            "log10",
            "log1p",
            "log2",
            "sign",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "trunc",
        ),
        # Julia's `round` rounds a tie to even, as the reference does.
        "round": _native("round"),
        "erf": _native(*special, "erf"),
        "erfc": _native(*special, "erfc"),
        "erfinv": _native(*special, "erfinv"),
        "digamma": _native(*special, "digamma"),
        # `logabsgamma` returns the log magnitude and the sign.
        "lgamma": _form(_call("first", _call(".".join((*special, "logabsgamma")), _X))),
        "square": _form(_COMMON_FORMS["square"]),
        "reciprocal": _native("inv"),
        "rsqrt": _form(_COMMON_FORMS["rsqrt"]),
        **_common("sigmoid", "softplus", "logsigmoid", *_ACTIVATIONS),
        "pow": Dedicated(reason="the broadcast power `.^`"),
        "max": Dedicated(reason="the broadcast `max.(a, b)`"),
        "min": Dedicated(reason="the broadcast `min.(a, b)`"),
    }


# ---------------------------------------------------------------------------
# WebPPL
# ---------------------------------------------------------------------------

#: Builtins the WebPPL runtime defines as elementwise globals, under the
#: QVR name prefixed by `_qvr_`, except the ones it has always defined
#: under the bare name.
WEBPPL_RUNTIME_BUILTINS: frozenset[str] = frozenset(
    {
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "ceil",
        "cos",
        "cosh",
        "digamma",
        "erf",
        "erfc",
        "erfinv",
        "expm1",
        "floor",
        "lgamma",
        "log10",
        "log1p",
        "log2",
        "round",
        "sign",
        "sin",
        "sinh",
        "tan",
        "tanh",
        "trunc",
    }
)

_WEBPPL: dict[str, BuiltinSpelling] = {
    **{name: _native(f"_qvr_{name}") for name in WEBPPL_RUNTIME_BUILTINS},
    **_same_name("exp", "log", "sqrt", "abs", "sigmoid", "max", "min", "pow"),
    **_common("square", "reciprocal", "rsqrt", "softplus", "logsigmoid", *_ACTIVATIONS),
}


# ---------------------------------------------------------------------------
# Church (Chez Scheme)
# ---------------------------------------------------------------------------

#: Builtins the Church runtime defines as elementwise procedures, under
#: the QVR name prefixed by `qvr-`.
CHURCH_RUNTIME_BUILTINS: frozenset[str] = frozenset(
    {
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "ceil",
        "cos",
        "cosh",
        "digamma",
        "erf",
        "erfc",
        "erfinv",
        "exp",
        "expm1",
        "floor",
        "lgamma",
        "log",
        "log10",
        "log1p",
        "log2",
        "round",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "trunc",
        "max",
        "min",
        "pow",
    }
)

_CHURCH: dict[str, BuiltinSpelling] = {
    **{name: _native(f"qvr-{name}") for name in CHURCH_RUNTIME_BUILTINS},
    **_common(
        "square",
        "reciprocal",
        "rsqrt",
        "sigmoid",
        "softplus",
        "logsigmoid",
        *_ACTIVATIONS,
    ),
}


# ---------------------------------------------------------------------------
# BUGS / JAGS
# ---------------------------------------------------------------------------

_BUGS_FLOOR = _op(
    "-",
    _call("trunc", _X),
    _op(
        "*",
        _call("step", _neg(_X)),
        _op("-", _lit(1.0), _call("equals", _X, _call("trunc", _X))),
    ),
)

_BUGS: dict[str, BuiltinSpelling] = {
    **_same_name(
        "abs",
        "cos",
        "cosh",
        "exp",
        "log",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "trunc",
        "pow",
        "max",
        "min",
    ),
    "acos": _native("arccos"),
    "acosh": _native("arccosh"),
    "asin": _native("arcsin"),
    "asinh": _native("arcsinh"),
    "atan": _native("arctan"),
    "atanh": _native("arctanh"),
    "sigmoid": _native("ilogit"),
    "lgamma": _native("loggam"),
    # `step` is one at zero, so `trunc(x) - 1` exactly when `x` is a
    # negative non-integer.
    "floor": _form(_BUGS_FLOOR),
    # BUGS / JAGS `round` rounds a tie up.
    "round": _form(_half_even("floor", "step")),
    "sign": _form(_op("-", _call("step", _X), _call("step", _neg(_X)))),
    # `erf(x) = 2 Phi(x sqrt 2) - 1`, through the normal CDF `phi`.
    "erf": _form(
        _op(
            "-",
            _op("*", _lit(2.0), _call("phi", _op("/", _X, _lit(_SQRT_HALF)))),
            _lit(1.0),
        )
    ),
    "erfc": _form(
        _op("*", _lit(2.0), _call("phi", _neg(_op("/", _X, _lit(_SQRT_HALF)))))
    ),
    "erfinv": _form(
        _op(
            "*",
            _call("probit", _op("/", _op("+", _X, _lit(1.0)), _lit(2.0))),
            _lit(_SQRT_HALF),
        )
    ),
    **_refuse("BUGS and JAGS have no digamma function", "digamma"),
    **_common(
        "ceil",
        "expm1",
        "log1p",
        "log2",
        "log10",
        "square",
        "reciprocal",
        "rsqrt",
        "softplus",
        "logsigmoid",
        *_ACTIVATIONS,
    ),
}


#: Every target's spelling of every math builtin.
BUILTIN_SPELLINGS: Mapping[str, Mapping[str, BuiltinSpelling]] = {
    "pyro": _PYRO,
    "numpyro": _NUMPYRO,
    "pymc": _PYMC,
    "edward2": _EDWARD2,
    "stan": _STAN,
    "turing": _julia("Turing"),
    "gen": _julia("Gen"),
    "webppl": _WEBPPL,
    "church": _CHURCH,
    "bugs": _BUGS,
    "jags": _BUGS,
}


def builtin_spelling(target: str, func: str, arity: int) -> BuiltinSpelling | None:
    """The target's spelling of a math builtin call.

    Parameters
    ----------
    target : str
        The transpile target.
    func : str
        The called name.
    arity : int
        The number of arguments.

    Returns
    -------
    BuiltinSpelling | None
        The spelling, or ``None`` when ``func`` at this arity is not a
        pointwise math builtin (a reduction such as one-argument
        ``max``, a let-bound function, or a target's own function), or
        when ``target`` is the neutral IR, which keeps every builtin
        under its QVR name.

    Raises
    ------
    UnsupportedConstruct
        If the target has no table, or its table has no entry for the
        builtin.
    """
    if MATH_BUILTINS.get(func) != arity or target == NEUTRAL_TARGET:
        return None
    table = BUILTIN_SPELLINGS.get(target)
    if table is None or func not in table:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [f"let-expr:builtin:{func}: no {target} spelling is recorded for {func!r}"],
        )
    return table[func]


def _refused(target: str, func: str, reason: str) -> UnsupportedConstruct:
    return UnsupportedConstruct(
        f"qvr-{target}",
        [f"let-expr:builtin:{func}: {target} cannot express {func!r}: {reason}"],
    )


def _instantiate(
    target: str, body: LetExprNode, args: tuple[LetExprNode, ...]
) -> LetExprNode:
    """A closed form's body with its arguments in place.

    A math builtin the body calls is spelled for the target in turn; the
    arguments are already spelled and are not walked again.
    """
    if isinstance(body, LetExprVar):
        if body.name == _X.name:
            return args[0]
        if body.name == _Y.name:
            return args[1]
        return body
    if isinstance(body, LetExprBinOp):
        return LetExprBinOp(
            op=body.op,
            left=_instantiate(target, body.left, args),
            right=_instantiate(target, body.right, args),
        )
    if isinstance(body, LetExprUnaryOp):
        return LetExprUnaryOp(
            op=body.op, operand=_instantiate(target, body.operand, args)
        )
    if isinstance(body, LetExprCall):
        return spell_builtin(
            target,
            body.func,
            tuple(_instantiate(target, argument, args) for argument in body.args),
        )
    return body


def spell_builtin(target: str, func: str, args: tuple[LetExprNode, ...]) -> LetExprNode:
    """A builtin call as the target's renderer reads it.

    A closed form is expanded, and a refusal raised; a native or
    dedicated spelling, and any call that is not a math builtin, keeps
    the call under its QVR name for the renderer to spell.

    Parameters
    ----------
    target : str
        The transpile target.
    func : str
        The called name.
    args : tuple[LetExprNode, ...]
        The arguments, already spelled for the target.

    Returns
    -------
    LetExprNode
        The call or its expansion.

    Raises
    ------
    UnsupportedConstruct
        If the target refuses the builtin or records no spelling for it.
    """
    spelling = builtin_spelling(target, func, len(args))
    if isinstance(spelling, ClosedForm):
        return _instantiate(target, spelling.body, args)
    if isinstance(spelling, Refusal):
        raise _refused(target, func, spelling.reason)
    return LetExprCall(func=func, args=args)


def native_path(target: str, func: str, arity: int) -> tuple[str, ...] | None:
    """The dotted path a renderer emits for a builtin call.

    Parameters
    ----------
    target : str
        The transpile target.
    func : str
        The called name.
    arity : int
        The number of arguments.

    Returns
    -------
    tuple[str, ...] | None
        The native function's path, or ``None`` when ``func`` is not a
        math builtin at this arity or the renderer spells it with
        dedicated code.

    Raises
    ------
    UnsupportedConstruct
        If the builtin reaches the renderer with a closed form or a
        refusal, which the planner resolves before rendering; emitting
        its name would hand the target a function it lacks.
    """
    spelling = builtin_spelling(target, func, arity)
    if spelling is None or isinstance(spelling, Dedicated):
        return None
    if isinstance(spelling, NativeCall):
        return spelling.path
    if isinstance(spelling, Refusal):
        raise _refused(target, func, spelling.reason)
    raise UnsupportedConstruct(
        f"qvr-{target}",
        [
            f"let-expr:builtin:{func}: {target} spells {func!r} as a closed "
            f"form, which reached the renderer unexpanded"
        ],
    )


def native_import(
    target: str, func: str, arity: int
) -> tuple[tuple[str, ...], str] | None:
    """The import a native builtin call needs.

    Parameters
    ----------
    target : str
        The transpile target.
    func : str
        The called name.
    arity : int
        The number of arguments.

    Returns
    -------
    tuple[tuple[str, ...], str] | None
        The module path and its alias, or ``None`` when the call needs
        no import.
    """
    spelling = builtin_spelling(target, func, arity)
    if isinstance(spelling, NativeCall) and spelling.import_path:
        return (spelling.import_path, spelling.import_alias)
    return None


__all__ = [
    "BUILTIN_SPELLINGS",
    "CHURCH_RUNTIME_BUILTINS",
    "MATH_BUILTINS",
    "NEUTRAL_TARGET",
    "WEBPPL_RUNTIME_BUILTINS",
    "BuiltinSpelling",
    "ClosedForm",
    "Dedicated",
    "NativeCall",
    "Refusal",
    "builtin_spelling",
    "native_import",
    "native_path",
    "spell_builtin",
]
