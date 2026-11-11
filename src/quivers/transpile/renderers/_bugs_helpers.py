"""Render [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode] subtrees
to BUGS / JAGS tree-sitter schema vertices.

BUGS / JAGS share an identical expression grammar for the
deterministic-assignment idiom ``<name> <- <expr>``: ``binary_expression``,
``unary_expression``, ``function_call`` (with ``name`` and
``argument_list``), ``identifier``, ``number``, ``indexed_variable``
(with ``index_list`` carrying integer / identifier / range children),
and ``parenthesized_expression``. Neither language has a native
string literal, lambda, method-call, or list-literal at the model-body
level; those LetExpr kinds either get a structural unrolling
(``list -> c(...)`` combine, ``factor -> c(<body[0]>, <body[1]>, ...)``
unroll) or raise `UnsupportedConstruct`.

The helper consumes a lightweight context exposing four bound methods:

* ``ctx.fresh(prefix: str) -> str``
* ``ctx.v(vid: str, kind: str) -> str``
* ``ctx.e(src: str, tgt: str, kind: str) -> None``
* ``ctx.lit(vid: str, text: str) -> None``
* ``ctx.constraint(vid: str, sort: str, value: str) -> None``

The renderer (BUGS or JAGS) is responsible for the surrounding
``deterministic_relation`` / ``stochastic_relation`` and for binding
the helper context. When unrolling factor binders, the helper also
reads ``ctx.cards`` (a ``dict[str, int]``) and ``ctx.target`` (one of
``"bugs"`` / ``"jags"``) to resolve axis sizes and to label the error
tag with the correct backend.
"""

from __future__ import annotations

import math
from typing import Literal, Protocol, runtime_checkable

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLambda,
    LetExprList,
    LetExprLiteral,
    LetExprMethodCall,
    LetExprNode,
    LetExprString,
    LetExprUnaryOp,
    LetExprVar,
    LetFactorBinder,
    LetFactorCase,
)
from quivers.dsl.ast_nodes.objects import (
    DiscreteConstructor,
    ObjectExpr,
    TypeName,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.builtin_spellings import native_path
from quivers.transpile.ir import (
    affine_column_offsets,
    Dim,
    DimDynamic,
    DimStatic,
    IRArg,
    IRArgBroadcast,
    IRArgList,
    IRArgMatrix,
    IRArgRef,
    IRDataInput,
    IRDeterministic,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRArgNumber,
    IRSample,
    LetAffineSource,
    LetExprAffineMap,
    Plate,
)
from quivers.transpile.renderers._one_based import fold_integers
from quivers.transpile.renderers._ranks import LAST_AXIS_REDUCTIONS
from quivers.transpile.renderers._base import IRArgTransform


#: QVR families supported on the non-negative reals whose BUGS / JAGS
#: target distribution is supported on all of R. ``HalfNormal(scale)``
#: lowers to ``dnorm(0, 1/scale^2)``, ``HalfCauchy(scale)`` to
#: ``dt(0, 1/scale^2, 1)`` and ``HalfStudentT(df, scale)`` to
#: ``dt(0, 1/scale^2, df)``; each needs an explicit lower truncation
#: at zero so the emitted support matches the family's, and so the
#: normalising constant picks up the factor of two the folded density
#: carries. The value is the lower bound of the family's support.
HALF_SUPPORT_LOWER_BOUND: dict[str, float] = {
    "HalfNormal": 0.0,
    "HalfCauchy": 0.0,
    "HalfStudentT": 0.0,
}

#: The truncation-suffix keyword each backend spells. Both the ``jags``
#: and the ``bugs`` backend execute through the JAGS engine (the
#: ``bugs`` probe image installs the ``jags`` binary and pyjags, and
#: the gallery / numeric-equivalence harness maps ``"bugs"`` to the
#: JAGS probe script), so both emit JAGS's renormalized truncation
#: ``T(lower, upper)``. JAGS accepts the alternative ``I(lower, upper)``
#: only when every parameter of the truncated distribution is a
#: compile-time constant, and on a latent-parent node (the canonical
#: hierarchical scale prior ``sigma ~ dnorm(0, 1/(tau*tau)) T(0,)`` with
#: latent ``tau``) it rejects ``I(,)`` at compile time; on real
#: OpenBUGS ``I(,)`` on a latent-parent node is interval censoring, a
#: different likelihood, not the folded prior measure the family
#: denotes. ``T(,)`` is the only spelling that both compiles on the
#: JAGS engine and preserves the renormalized one-sided fold, so it is
#: the correct emission for both backends. The grammar exposes both
#: spellings as alternatives of the same ``truncation`` rule, selected
#: through the ``chose-alt-fingerprint`` constraint.
TRUNCATION_FINGERPRINT: dict[str, str] = {
    "jags": "T( , )",
    "bugs": "T( , )",
}


#: The BUGS / JAGS log-gamma and log-factorial builtins the
#: beta-binomial log-pmf is written in terms of. Both live in the
#: ``bugs`` module, which every JAGS engine loads by default and which
#: the OpenBUGS / WinBUGS function library also ships, so the closed
#: form below needs no optional module.
_LOG_GAMMA: str = "loggam"
_LOG_FACTORIAL: str = "logfact"

#: ``sqrt(2 * pi)``, the Gaussian density's normalising constant. A
#: closed-form Gaussian factor writes it as a literal; both engines
#: evaluate a constant expression once per compile either way, and the
#: literal keeps the emitted relation readable.
SQRT_TWO_PI: float = math.sqrt(2.0 * math.pi)


def _let_add(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    return LetExprBinOp(op="+", left=left, right=right)


def _let_sub(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    return LetExprBinOp(op="-", left=left, right=right)


def _let_mul(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    return LetExprBinOp(op="*", left=left, right=right)


def _let_log(inner: LetExprNode) -> LetExprNode:
    return LetExprCall(func="log", args=(inner,))


def _let_signed_sum(
    head: LetExprNode, *tail: tuple[Literal["+", "-"], LetExprNode]
) -> LetExprNode:
    """Left-fold a signed term list into one ``+`` / ``-`` chain.

    Each entry of `tail` pairs the operator that joins it to the
    running total with the term itself, so the result associates
    exactly as the source reads it.
    """
    total = head
    for op, term in tail:
        total = LetExprBinOp(op=op, left=total, right=term)
    return total


def _scalar_arg_expr(backend: str, family: str, slot: str, arg: IRArg) -> LetExprNode:
    """Read one distribution argument back as a let-expression.

    The closed-form densities this module writes out consume their
    arguments as ordinary arithmetic operands rather than as
    distribution-call children, so each one has to come back as a
    `LetExprNode`. A bare reference and a numeric literal both do; an
    index expression, a list, a matrix, or a broadcast wrapper carries
    structure the scalar closed form cannot place, so it raises.
    """
    if isinstance(arg, IRArgNumber):
        return LetExprLiteral(value=arg.value)
    if isinstance(arg, IRArgRef) and not arg.indices:
        return LetExprVar(name=arg.name)
    raise UnsupportedConstruct(
        f"qvr-{backend}",
        [
            f"family:{family}:non-scalar-arg:{slot}: the closed-form "
            f"density reads each argument as a scalar operand, and "
            f"this slot carries a {type(arg).__name__}"
        ],
    )


def beta_binomial_log_pmf(
    backend: str,
    *,
    variate: str,
    args: tuple[IRArg, ...],
    arg_names: tuple[str, ...],
) -> LetExprNode:
    """Build the closed-form beta-binomial log pmf.

    BUGS lacks this distribution, and JAGS provides it only in the optional
    ``mix`` module. The returned expression uses base-library ``loggam`` and
    ``logfact`` functions::

        logfact(n) - logfact(y) - logfact(n - y)
        + loggam(a + y) + loggam(b + n - y) - loggam(a + b + n)
        - loggam(a) - loggam(b) + loggam(a + b)
    """
    by_name = dict(zip(arg_names, args, strict=False))
    missing = [
        slot
        for slot in ("total_count", "concentration1", "concentration0")
        if slot not in by_name
    ]
    if missing:
        raise UnsupportedConstruct(
            f"qvr-{backend}",
            [
                f"family:BetaBinomial:missing-arg:{','.join(missing)}: "
                f"the closed-form density needs the trial count and "
                f"both concentrations; the site supplies "
                f"{list(arg_names)}"
            ],
        )
    n = _scalar_arg_expr(backend, "BetaBinomial", "total_count", by_name["total_count"])
    a = _scalar_arg_expr(
        backend,
        "BetaBinomial",
        "concentration1",
        by_name["concentration1"],
    )
    b = _scalar_arg_expr(
        backend,
        "BetaBinomial",
        "concentration0",
        by_name["concentration0"],
    )
    y = LetExprVar(name=variate)

    def loggam(inner: LetExprNode) -> LetExprNode:
        return LetExprCall(func=_LOG_GAMMA, args=(inner,))

    def logfact(inner: LetExprNode) -> LetExprNode:
        return LetExprCall(func=_LOG_FACTORIAL, args=(inner,))

    return _let_signed_sum(
        logfact(n),
        ("-", logfact(y)),
        ("-", logfact(_let_sub(n, y))),
        ("+", loggam(_let_add(a, y))),
        ("+", loggam(_let_sub(_let_add(b, n), y))),
        ("-", loggam(_let_add(_let_add(a, b), n))),
        ("-", loggam(a)),
        ("-", loggam(b)),
        ("+", loggam(_let_add(a, b))),
    )


def kumaraswamy_log_pdf(
    backend: str,
    *,
    variate: str,
    args: tuple[IRArg, ...],
    arg_names: tuple[str, ...],
) -> LetExprNode:
    """Build the closed-form Kumaraswamy log density.

    For shapes ``a`` and ``b`` on ``(0, 1)``, the returned expression is::

        log(a) + log(b) + (a - 1) * log(y) + (b - 1) * log(1 - y^a)

    It uses only BUGS and JAGS base-library functions. The power ``y^a`` is
    emitted as ``pow(y, a)``, the spelling shared by both engines.
    """
    by_name = dict(zip(arg_names, args, strict=False))
    missing = [
        slot for slot in ("concentration1", "concentration0") if slot not in by_name
    ]
    if missing:
        raise UnsupportedConstruct(
            f"qvr-{backend}",
            [
                f"family:Kumaraswamy:missing-arg:{','.join(missing)}: "
                f"the closed-form density needs both shape "
                f"parameters; the site supplies {list(arg_names)}"
            ],
        )
    a = _scalar_arg_expr(
        backend, "Kumaraswamy", "concentration1", by_name["concentration1"]
    )
    b = _scalar_arg_expr(
        backend, "Kumaraswamy", "concentration0", by_name["concentration0"]
    )
    y = LetExprVar(name=variate)
    one = LetExprLiteral(value=1.0)
    return _let_signed_sum(
        _let_log(a),
        ("+", _let_log(b)),
        ("+", _let_mul(_let_sub(a, one), _let_log(y))),
        (
            "+",
            _let_mul(
                _let_sub(b, one),
                _let_log(
                    _let_sub(
                        one,
                        LetExprCall(func="pow", args=(y, a)),
                    )
                ),
            ),
        ),
    )


#: Half-width of the interval around ``lambda = 1/2`` on which the
#: continuous Bernoulli's normaliser is read at a displaced argument.
#:
#: ``C(lambda) = (log(1 - lambda) - log(lambda)) / (1 - 2 lambda)``
#: is a ratio of two quantities that vanish together at one half. The
#: limit is ``2`` and the function is analytic through it, but the
#: emitted arithmetic is not: at ``lambda = 1/2`` exactly both
#: logarithms of absolute values are ``-inf`` and their difference is
#: ``NaN``, which no engine recovers from. Reading the normaliser at
#: an argument displaced off the singular point instead keeps every
#: intermediate finite. Writing ``d = 1 - 2 lambda``, the normaliser
#: satisfies ``log C = log 2 + d^2 / 3 + O(d^4)``, so a displacement
#: bounded by ``2 * _CONT_BERNOULLI_STABLE_HALF_WIDTH`` moves the
#: value by at most about ``3e-12`` -- far below any tolerance the
#: equivalence suite runs at, and confined to a window the emitted
#: expression leaves untouched everywhere else, where it is exact.
_CONT_BERNOULLI_STABLE_HALF_WIDTH: float = 1.0e-6


def continuous_bernoulli_log_pdf(
    backend: str,
    *,
    variate: str,
    args: tuple[IRArg, ...],
    arg_names: tuple[str, ...],
) -> LetExprNode:
    """Build the closed-form continuous-Bernoulli log density.

    For tilt ``lambda`` on ``(0, 1)``, the density has normalizer::

        C(lambda) = (log(1 - lambda) - log(lambda)) / (1 - 2 * lambda).

    The returned expression splits the log absolute ratio into two logarithms.
    Within ``1e-6`` of one half, it displaces the denominator from zero and
    evaluates the normalizer at the corresponding displaced tilt. The tilted
    terms continue to use the original value. The expression requires only
    ``log``, ``abs``, and ``step`` from the BUGS and JAGS base libraries.
    """
    by_name = dict(zip(arg_names, args, strict=False))
    if "probs" not in by_name:
        raise UnsupportedConstruct(
            f"qvr-{backend}",
            [
                f"family:ContinuousBernoulli:missing-arg:probs: the "
                f"closed-form density needs the tilt parameter; the "
                f"site supplies {list(arg_names)}"
            ],
        )
    lam = _scalar_arg_expr(backend, "ContinuousBernoulli", "probs", by_name["probs"])
    x = LetExprVar(name=variate)
    half = LetExprLiteral(value=0.5)
    one = LetExprLiteral(value=1.0)
    two = LetExprLiteral(value=2.0)
    width = LetExprLiteral(value=_CONT_BERNOULLI_STABLE_HALF_WIDTH)

    def absolute(inner: LetExprNode) -> LetExprNode:
        return LetExprCall(func="abs", args=(inner,))

    # d = 1 - 2 * lambda, and the indicator of the window around zero
    # on which the ratio is read at a displaced argument.
    gap = _let_sub(one, _let_mul(two, lam))
    near = LetExprCall(func="step", args=(_let_sub(width, absolute(gap)),))
    # Displacing by twice the half-width keeps `d_safe` clear of zero
    # for every `d` the indicator selects, including `d = -width`,
    # which the indicator includes and a one-width shift would send to
    # zero.
    gap_safe = _let_add(gap, _let_mul(near, _let_mul(two, width)))
    # The tilt that `d_safe` names, `m = (1 - d_safe) / 2`.
    tilt = _let_mul(_let_sub(one, gap_safe), half)
    return _let_signed_sum(
        _let_mul(x, _let_log(lam)),
        ("+", _let_mul(_let_sub(one, x), _let_log(_let_sub(one, lam)))),
        (
            "+",
            _let_log(absolute(_let_sub(_let_log(_let_sub(one, tilt)), _let_log(tilt)))),
        ),
        ("-", _let_log(absolute(gap_safe))),
    )


def reorder_half_studentt_dt(
    args: tuple[IRArg, ...], arg_names: tuple[str, ...]
) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
    """Reshape ``HalfStudentT(df, scale)`` into the BUGS / JAGS
    ``dt(mu, tau, k)`` argument order.

    ``dt`` is parameterised by location, precision and degrees of
    freedom, in that order, and carries no half-support variant; the
    QVR call site writes ``(df, scale)`` and names a density folded
    at zero. This returns ``(0, 1/(scale*scale), df)``, which the
    caller pairs with the lower truncation
    `half_support_truncation`
    supplies for the family, so the emitted relation is
    ``dt(0, 1/(scale*scale), df) T (0 ,)``: the renormalized
    one-sided fold, not the two-sided Student-t.

    The precision is pre-wrapped here rather than left to the
    alias-rename pipeline because the reordered names no longer line
    up with the family's own slots, exactly as
    ``StudentT``'s reorder does.
    """
    by_name = dict(zip(arg_names, args, strict=True))
    missing = [slot for slot in ("df", "scale") if slot not in by_name]
    if missing:
        raise UnsupportedConstruct(
            "qvr-bugs",
            [
                f"family:HalfStudentT:missing-arg:{','.join(missing)}: "
                f"the `dt(mu, tau, k)` reparameterisation needs both "
                f"the degrees of freedom and the scale; the site "
                f"supplies {list(arg_names)}"
            ],
        )
    return (
        (
            IRArgNumber(value=0.0),
            IRArgTransform(inner=by_name["scale"], transform="inv_square"),
            by_name["df"],
        ),
        ("loc", "tau", "df"),
    )


def reorder_binomial_dbin(
    args: tuple[IRArg, ...], arg_names: tuple[str, ...]
) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
    """Reshape ``Binomial(total_count, probs)`` into the BUGS / JAGS
    ``dbin(p, n)`` argument order.

    ``dbin`` takes the per-trial success probability first and the
    trial count second, the reverse of the torch spelling the QVR
    call site is positional against. Emitting the two in QVR order
    binds the trial count to ``p`` and the probability to ``n``, which
    scores a different pmf at every point rather than up to a
    constant.

    The logit spelling has no ``dbin`` slot, so a site that supplies
    ``logits`` instead of ``probs`` raises rather than emitting a call
    the engine would read as a probability.
    """
    by_name = dict(zip(arg_names, args, strict=True))
    if "probs" not in by_name or "total_count" not in by_name:
        raise UnsupportedConstruct(
            "qvr-bugs",
            [
                f"family:Binomial:missing-arg: `dbin(p, n)` takes a "
                f"probability and a trial count, and neither engine "
                f"has a logit-parameterised binomial; the site "
                f"supplies {list(arg_names)}"
            ],
        )
    return (
        (by_name["probs"], by_name["total_count"]),
        ("p", "n"),
    )


def reorder_pareto_dpar(
    args: tuple[IRArg, ...], arg_names: tuple[str, ...]
) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
    """Reshape ``Pareto(scale, alpha)`` into the BUGS / JAGS
    ``dpar(alpha, c)`` argument order.

    Both parameterisations denote the same density ``alpha c^alpha
    x^{-(alpha + 1)}`` on ``x > c``; they disagree only on which slot
    carries the shape. ``dpar`` names the shape first and the scale
    (the support's lower endpoint) second, so emitting the QVR order
    positionally swaps the two.
    """
    by_name = dict(zip(arg_names, args, strict=True))
    missing = [slot for slot in ("alpha", "scale") if slot not in by_name]
    if missing:
        raise UnsupportedConstruct(
            "qvr-bugs",
            [
                f"family:Pareto:missing-arg:{','.join(missing)}: "
                f"`dpar(alpha, c)` takes the shape and the scale; the "
                f"site supplies {list(arg_names)}"
            ],
        )
    return (
        (by_name["alpha"], by_name["scale"]),
        ("alpha", "c"),
    )


def half_support_truncation(family: str, *, observed: bool) -> tuple[IRArg, ...] | None:
    """Return the one-sided truncation bounds a latent draw from
    ``family`` needs, or `None` when no truncation applies.

    Only latent draws are truncated. On an observed node the JAGS
    engine reads the truncation suffix as censoring rather than
    truncation, which is a different likelihood; an observed
    half-support variate already lies in the support, so the omitted
    suffix costs only the constant ``log 2`` per observation.
    """
    if observed:
        return None
    lower = HALF_SUPPORT_LOWER_BOUND.get(family)
    if lower is None:
        return None
    return (IRArgNumber(value=lower),)


@runtime_checkable
class _BugsLetCtx(Protocol):
    """Structural protocol for the helper's ctx parameter.

    `range_1_to` and `range_between` build the range vertex each
    backend's grammar wants; the two grammars disagree on the
    alternative-selection constraints a `range` carries, so the
    renderer owns the construction and the helper only asks for one.
    `range_1_to` is reached only when the caller supplies a
    declared-plate table, so a caller rendering a standalone
    expression never needs it.
    """

    target: str
    cards: dict[str, int]

    def fresh(self, prefix: str) -> str: ...
    def v(self, vid: str, kind: str) -> str: ...
    def e(self, src: str, tgt: str, kind: str) -> None: ...
    def lit(self, vid: str, text: str) -> None: ...
    def constraint(self, vid: str, sort: str, value: str) -> None: ...
    def range_1_to(self, upper: str) -> str: ...
    def range_between(self, lower: str, upper: str) -> str: ...


class _LetEnv:
    """The helper's per-render environment.

    Bundles the backend ctx with the declared-plate table the index
    and reduction emitters read. The table is a caller-supplied input
    rather than a property of the ctx, because a caller may render a
    standalone expression that sits in no program: every name then has
    no declared shape, and the emitters read every operand as a
    scalar, which is what such an expression means.
    """

    def __init__(
        self,
        ctx: _BugsLetCtx,
        decl_plates: dict[str, Plate],
        row_index: str | None = None,
    ) -> None:
        self._ctx = ctx
        self.decl_plates = decl_plates
        self.target = _target(ctx)
        self.cards = ctx.cards
        # Loop variable of the binding's codomain axis, when the
        # caller emits the binding inside a `for` loop. An affine
        # parameter map reads it as the row it contracts.
        self.row_index = row_index

    def fresh(self, prefix: str) -> str:
        return self._ctx.fresh(prefix)

    def v(self, vid: str, kind: str) -> str:
        return self._ctx.v(vid, kind)

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._ctx.e(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._ctx.lit(vid, text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._ctx.constraint(vid, sort, value)

    def range_1_to(self, upper: str) -> str:
        return self._ctx.range_1_to(upper)

    def range_between(self, lower: str, upper: str) -> str:
        return self._ctx.range_between(lower, upper)


def render_let_expr_bugs(
    ctx: _BugsLetCtx,
    expr: LetExprNode,
    *,
    decl_plates: dict[str, Plate] | None = None,
    row_index: str | None = None,
) -> str:
    """Build a BUGS or JAGS expression schema and return its root vertex.

    Recurses through nested `LetExprNode` values. Raises
    `UnsupportedConstruct` for strings, lambdas, method calls, or factors whose
    binder axes have unknown cardinality.

    `decl_plates` supplies declared shapes so indexed matrix rows retain their
    trailing full-axis slice. `row_index` names the loop variable used when a
    `LetExprAffineMap` contracts one parameter-map row; other expressions ignore
    it.
    """
    return _render(_LetEnv(ctx, decl_plates or {}, row_index), expr)


def _render(ctx: _LetEnv, expr: LetExprNode) -> str:
    """Dispatch one [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode]
    to its per-kind emitter."""
    if isinstance(expr, LetExprLiteral):
        return _emit_number(ctx, expr.value)
    if isinstance(expr, LetExprVar):
        return _emit_identifier(ctx, expr.name)
    if isinstance(expr, LetExprBinOp):
        return _emit_binop(ctx, expr)
    if isinstance(expr, LetExprUnaryOp):
        return _emit_unary(ctx, expr)
    if isinstance(expr, LetExprCall) and expr.func in ("quot", "rem"):
        return _render(ctx, _truncating(expr))
    if isinstance(expr, LetExprCall):
        return _emit_reduction_or_call(ctx, expr)
    if isinstance(expr, LetExprIndex):
        return _emit_index(ctx, expr)
    if isinstance(expr, LetExprList):
        return _emit_list(ctx, expr.items)
    if isinstance(expr, LetExprAffineMap):
        return _emit_affine_map(ctx, expr)
    if isinstance(expr, LetExprFactor):
        return _emit_factor(ctx, expr)
    if isinstance(expr, LetExprString):
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprString:{_target(ctx)}: BUGS / JAGS "
                f"have no native string literal in the model body"
            ],
        )
    if isinstance(expr, LetExprLambda):
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprLambda:{_target(ctx)}: BUGS / JAGS "
                f"have no anonymous function syntax"
            ],
        )
    if isinstance(expr, LetExprMethodCall):
        # BUGS and JAGS have no `receiver.method(args)` dispatch
        # syntax, and unlike Stan they also have no path to a
        # user-defined-function definition the renderer could graft:
        #
        # 1. BUGS / OpenBUGS forbid user-defined functions in the
        #    model body; only built-in distributions and the
        #    standard math library are callable, and there is no
        #    `functions { ... }` block.
        # 2. JAGS allows user functions only via compiled C++
        #    modules linked at JAGS startup (the JAGS Module API);
        #    they cannot be declared inline in the model file.
        # 3. The model body cannot express the inside-algorithm
        #    chart parser anyway: BUGS / JAGS deterministic
        #    relations are non-recursive scalar/array updates over
        #    static index ranges, with no support for the variable-
        #    length span enumeration the CKY-style fixed-point
        #    requires.
        #
        # `deduction_decl` is also stripped by the IR pipeline (see
        # `CATEGORICAL_METADATA_IGNORABLE`),
        # so even if the model body could host a chart parser the
        # renderer has no access to the rules to compile against.
        # Rewriting `m.f(a)` as the static call `f(m, a)` without
        # supplying `f` produces an undefined-symbol model that
        # JAGS / OpenBUGS reject at parse time, so the helper
        # raises instead of emitting a bogus call.
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprMethodCall:{_target(ctx)}: BUGS / "
                f"JAGS have no method-dispatch syntax; the chart-"
                f"parser deduction graft that would supply the "
                f"called function is also impossible because BUGS "
                f"forbids user-defined model-body functions and "
                f"JAGS exposes them only through compiled C++ "
                f"modules linked at startup, not inline"
            ],
        )
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [f"let-expr:{type(expr).__name__}:{_target(ctx)}: unhandled"],
    )


# ---------------------------------------------------------------------------
# Per-kind emitters.
# ---------------------------------------------------------------------------


def _emit_number(ctx: _BugsLetCtx, value: float) -> str:
    """Emit a `number` vertex carrying the textual rendering of `value`."""
    v = ctx.v(ctx.fresh("num"), "number")
    text = str(int(value)) if float(value).is_integer() else repr(value)
    ctx.lit(v, text)
    return v


def _emit_identifier(ctx: _BugsLetCtx, name: str) -> str:
    """Emit a bare `identifier` vertex carrying `name`."""
    v = ctx.v(ctx.fresh("id"), "identifier")
    ctx.lit(v, name)
    return v


def _render_bugs_operand(ctx: _LetEnv, expr: LetExprNode) -> str:
    """Render `expr` as an operand of a binary / unary operator.

    A nested [`LetExprBinOp`][quivers.dsl.ast_nodes.LetExprBinOp] or
    [`LetExprUnaryOp`][quivers.dsl.ast_nodes.LetExprUnaryOp] operand is
    wrapped in a `parenthesized_expression`. The BUGS / JAGS pretty
    printer emits children in source order without re-grouping, so
    ``(a + b) * c`` would otherwise print as ``a + b * c`` and
    reassociate under the language's precedence.
    """
    vid = _render(ctx, expr)
    if isinstance(expr, (LetExprBinOp, LetExprUnaryOp)):
        return _emit_paren(ctx, vid, _arg_edge_kind(expr))
    return vid


def _emit_paren(ctx: _BugsLetCtx, inner: str, inner_kind: str) -> str:
    """Wrap `inner` (of grammar kind `inner_kind`) in a
    `parenthesized_expression` vertex."""
    p = ctx.v(ctx.fresh("paren"), "parenthesized_expression")
    ctx.constraint(p, "chose-alt-fingerprint", "( )")
    ctx.constraint(p, "chose-alt-child-kinds", inner_kind)
    ctx.e(p, inner, "child_of")
    return p


def _emit_binop(ctx: _LetEnv, expr: LetExprBinOp) -> str:
    """Emit a `binary_expression` with left and right field edges.

    The grammar's operator choice is selected by its field and alternative
    fingerprint. BUGS and JAGS do not lift infix operators over array axes.
    `_emit_reduction_or_call` handles supported contractions through `inprod`;
    other axis-carrying operand pairs raise. Elementwise array operations must
    instead be lowered as indexed assignments in a loop.
    """
    left_rank = axis_rank(ctx.decl_plates, expr.left)
    right_rank = axis_rank(ctx.decl_plates, expr.right)
    if left_rank > 0 and right_rank > 0:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:elementwise-axis-operator:{_target(ctx)}: "
                f"{expr.op!r} between a rank-{left_rank} and a "
                f"rank-{right_rank} operand has no BUGS / JAGS "
                f"*expression* form: neither language lifts an infix "
                f"operator over an axis, and the elementwise result "
                f"exists only as a named array built one index at a "
                f"time inside a loop of its own "
                f"(`for (i in 1:N) {{ z[i] <- a[i] + b[i] }}`), which "
                f"a let-binding lowered to one scalar expression has "
                f"nowhere to put. The one axis-carrying operand pair "
                f"that does lower as an expression is the contracted "
                f"product `sum(a * b)`, to `inprod(a, b)`"
            ],
        )
    b = ctx.v(ctx.fresh("be"), "binary_expression")
    ctx.constraint(b, "field:operator", expr.op)
    ctx.constraint(b, "chose-alt-fingerprint", expr.op)
    ctx.e(b, _render_bugs_operand(ctx, expr.left), "left")
    ctx.e(b, _render_bugs_operand(ctx, expr.right), "right")
    return b


def _emit_unary(ctx: _LetEnv, expr: LetExprUnaryOp) -> str:
    """Emit a unary-minus `unary_expression` whose single child rides
    the `operand` field."""
    u = ctx.v(ctx.fresh("ue"), "unary_expression")
    ctx.constraint(u, "field:operator", "-")
    ctx.constraint(u, "chose-alt-fingerprint", "-")
    ctx.e(u, _render_bugs_operand(ctx, expr.operand), "operand")
    return u


def _emit_call(
    ctx: _BugsLetCtx,
    func: str,
    arg_ids: tuple[str, ...],
    arg_kinds: tuple[str, ...],
) -> str:
    """Emit ``<func>(<arg_0>, <arg_1>, ...)`` as a `function_call`.

    The `name` field is an `identifier` vertex; the `arguments` field
    is an `argument_list` whose children carry their grammar kind as
    the edge label (so the panproto walker can pick the right child
    alternative). A pointwise math builtin is called by the BUGS / JAGS
    function
    [`native_path`][quivers.transpile.builtin_spellings.native_path]
    names (``sigmoid`` is ``ilogit``, ...).
    """
    c = ctx.v(ctx.fresh("call"), "function_call")
    path = native_path(_target(ctx), func, len(arg_ids))
    name_id = _emit_identifier(ctx, func if path is None else ".".join(path))
    ctx.e(c, name_id, "name")
    if not arg_ids:
        return c
    al = ctx.v(ctx.fresh("al"), "argument_list")
    ctx.e(c, al, "arguments")
    for aid, akind in zip(arg_ids, arg_kinds, strict=True):
        ctx.e(al, aid, akind)
    return c


#: QVR reduction primitives whose BUGS / JAGS counterpart contracts
#: every axis of its argument. Applied to a rank-1 operand -- the
#: shape every axis reduction in the gallery reduces -- the target
#: builtin computes exactly the QVR reduction; applied to a
#: higher-rank operand it would collapse axes the source keeps, so
#: `_emit_reduction_or_call`
#: raises instead.
_AXIS_REDUCING_CALLS: frozenset[str] = frozenset({"sum", "mean", "prod"})


def _emit_reduction_or_call(ctx: _LetEnv, expr: LetExprCall) -> str:
    """Emit a call, lowering an axis reduction to its target spelling.

    `sum(a * b)` over two rank-1 operands is the inner product, which
    BUGS / JAGS spell `inprod(a, b)`; the elementwise product it is
    written in terms of has no target form on its own. A reduction of
    a single rank-1 operand passes straight through, because the
    target builtin already contracts the one axis the operand
    carries. A reduction of an operand with several axes collapses
    only its last one, which the target builtin cannot spell, so it
    raises, whichever reduction it is.
    """
    if expr.func in LAST_AXIS_REDUCTIONS and len(expr.args) == 1:
        arg = expr.args[0]
        rank = axis_rank(ctx.decl_plates, arg)
        if rank > 1:
            raise UnsupportedConstruct(
                f"qvr-{_target(ctx)}-helper",
                [
                    f"let-expr:axis-reduction:{_target(ctx)}: "
                    f"{expr.func}(...) over a rank-{rank} operand "
                    f"reduces only the innermost axis in QVR, and "
                    f"BUGS / JAGS `{expr.func}` contracts every axis; "
                    f"there is no target spelling for the partial "
                    f"reduction"
                ],
            )
        if (
            rank == 1
            and isinstance(arg, LetExprBinOp)
            and expr.func in _AXIS_REDUCING_CALLS
        ):
            return _emit_contracted_binop(ctx, expr.func, arg)
    return _emit_call(
        ctx,
        expr.func,
        tuple(_render(ctx, a) for a in expr.args),
        tuple(_arg_edge_kind(a) for a in expr.args),
    )


def _emit_contracted_binop(ctx: _LetEnv, func: str, arg: LetExprBinOp) -> str:
    """Lower `sum(<a> * <b>)` over two rank-1 operands to
    `inprod(<a>, <b>)`, and raise on every other shape."""
    left_rank = axis_rank(ctx.decl_plates, arg.left)
    right_rank = axis_rank(ctx.decl_plates, arg.right)
    if func == "sum" and arg.op == "*" and left_rank == 1 and right_rank == 1:
        return _emit_call(
            ctx,
            "inprod",
            (
                _render(ctx, arg.left),
                _render(ctx, arg.right),
            ),
            (_arg_edge_kind(arg.left), _arg_edge_kind(arg.right)),
        )
    if left_rank > 0 and right_rank > 0:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:axis-reduction:{_target(ctx)}: "
                f"{func}(<a> {arg.op} <b>) over two axis-carrying "
                f"operands has no BUGS / JAGS form; only the "
                f"contracted product `sum(a * b)` lowers, to "
                f"`inprod(a, b)`"
            ],
        )
    return _emit_call(
        ctx,
        func,
        (_render(ctx, arg),),
        (_arg_edge_kind(arg),),
    )


def _emit_binary_ids(
    ctx: _LetEnv,
    op: str,
    left: tuple[str, str],
    right: tuple[str, str],
) -> tuple[str, str]:
    """Emit a `binary_expression` over two already-rendered operands,
    parenthesising each that the printer would otherwise re-group."""
    b = ctx.v(ctx.fresh("be"), "binary_expression")
    ctx.constraint(b, "field:operator", op)
    ctx.constraint(b, "chose-alt-fingerprint", op)
    for vid, kind, edge in ((*left, "left"), (*right, "right")):
        if kind in ("binary_expression", "unary_expression"):
            vid = _emit_paren(ctx, vid, kind)
        ctx.e(b, vid, edge)
    return b, "binary_expression"


def _affine_row_slot(ctx: _LetEnv, offset: int) -> tuple[str, str]:
    """Emit the weight row this loop iteration contracts.

    The loop variable already runs ``1:width`` in the target's
    one-based origin, so the head's zero-based `row_offset` shifts it
    without any further rebasing: head `k` of a `width`-wide codomain
    reads row ``<loop> + k * width``. Offset zero needs no arithmetic
    at all.
    """
    if ctx.row_index is None:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprAffineMap:{_target(ctx)}: the map "
                f"contracts one weight row per codomain coordinate "
                f"and so has to be emitted inside the loop over that "
                f"axis, but no loop variable reached the renderer"
            ],
        )
    loop = (_emit_identifier(ctx, ctx.row_index), "identifier")
    if offset == 0:
        return loop
    return _emit_binary_ids(
        ctx, "+", loop, (_emit_number(ctx, float(offset)), "number")
    )


def _affine_named_array(ctx: _LetEnv, expr: LetExprNode, role: str) -> str:
    """The array name an affine-map operand subscripts.

    BUGS and JAGS subscript a named array and nothing else, so an
    operand that is not a bare name has no target spelling and raises
    rather than emitting a subscript of an expression.
    """
    if isinstance(expr, LetExprVar):
        return expr.name
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [
            f"let-expr:LetExprAffineMap:{_target(ctx)}: BUGS / JAGS "
            f"subscript a named array, so the map's {role} has to be "
            f"a bare name, got {type(expr).__name__}"
        ],
    )


def _affine_slice(
    ctx: _LetEnv,
    name: str,
    row: tuple[str, str] | None,
    lower: int,
    width: int,
) -> str:
    """Emit ``<name>[lo:hi]``, or ``<name>[<row>, lo:hi]`` when a row
    slot is supplied.

    `lower` arrives in QVR's zero-based origin and the slice is
    inclusive at both ends in BUGS and JAGS, so it spans
    ``lower + 1`` to ``lower + width``.
    """
    iv = ctx.v(ctx.fresh("iv"), "indexed_variable")
    ctx.e(iv, _emit_identifier(ctx, name), "name")
    il = ctx.v(ctx.fresh("il"), "index_list")
    ctx.e(iv, il, "indices")
    if row is not None:
        ctx.e(il, row[0], row[1])
    ctx.e(
        il,
        ctx.range_between(str(lower + 1), str(lower + width)),
        "range",
    )
    return iv


def _emit_affine_map(ctx: _LetEnv, expr: LetExprAffineMap) -> str:
    """Emit one row of ``W x + b`` as a sum of `inprod` contractions.

    Neither BUGS nor JAGS has a matrix product, so the map is written
    one codomain coordinate at a time and the surrounding `for` loop
    supplies the coordinate. Each factor of the conditioning row
    contributes one `inprod` against its own column block of the
    weight, so a head costs one term per factor rather than one per
    (row, column) pair.
    """
    weight = _affine_named_array(ctx, expr.weight, "weight")
    total: tuple[str, str] | None = None
    for source, column in affine_column_offsets(expr):
        if source.width == 1:
            # A one-wide factor contracts to a single product, so the
            # operand needs no slice and therefore no name to slice.
            # This is what lets an already-subscripted factor through:
            # a plated site reaches the renderer as `<name>[<row>]`,
            # which BUGS and JAGS cannot slice but can multiply.
            term = _emit_binary_ids(
                ctx,
                "*",
                (
                    _affine_slice(
                        ctx,
                        weight,
                        _affine_row_slot(ctx, expr.row_offset),
                        column,
                        1,
                    ),
                    "indexed_variable",
                ),
                (
                    _render_bugs_operand(ctx, source.value),
                    _arg_edge_kind(source.value),
                ),
            )
        else:
            term = (
                _emit_call(
                    ctx,
                    "inprod",
                    (
                        _affine_slice(
                            ctx,
                            weight,
                            _affine_row_slot(ctx, expr.row_offset),
                            column,
                            source.width,
                        ),
                        _affine_slice(
                            ctx,
                            _affine_named_array(
                                ctx, source.value, "conditioning factor"
                            ),
                            None,
                            0,
                            source.width,
                        ),
                    ),
                    ("indexed_variable", "indexed_variable"),
                ),
                "function_call",
            )
        total = term if total is None else _emit_binary_ids(ctx, "+", total, term)
    if total is None:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprAffineMap:{_target(ctx)}: the map's "
                f"conditioning row carries no factors"
            ],
        )
    bias_iv = ctx.v(ctx.fresh("iv"), "indexed_variable")
    ctx.e(
        bias_iv,
        _emit_identifier(ctx, _affine_named_array(ctx, expr.bias, "bias")),
        "name",
    )
    bias_il = ctx.v(ctx.fresh("il"), "index_list")
    ctx.e(bias_iv, bias_il, "indices")
    row_vid, row_kind = _affine_row_slot(ctx, expr.row_offset)
    ctx.e(bias_il, row_vid, row_kind)
    total = _emit_binary_ids(ctx, "+", total, (bias_iv, "indexed_variable"))
    if expr.transform == "exp":
        return _emit_call(ctx, "exp", (total[0],), (total[1],))
    return total[0]


def _emit_index(ctx: _LetEnv, expr: LetExprIndex) -> str:
    """Emit `arr[i0, i1, ...]` as an `indexed_variable` with an
    `index_list` child.

    BUGS / JAGS index expressions are not a separate node kind: the
    grammar reuses `indexed_variable` for both LHS index targets and
    nested expression-position subscripts. The `name` field must be
    an `identifier`; multi-level array nesting (`a[i][j]`) is rare in
    practice for these languages and so the helper expects the outer
    expression's `array` slot to resolve to an `identifier` vertex.
    When `expr.array` produces an `indexed_variable` instead, the
    helper raises rather than silently rewriting the access path.
    """
    if not isinstance(expr.array, LetExprVar):
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprIndex:{_target(ctx)}: BUGS / JAGS "
                f"`indexed_variable` requires a bare `identifier` "
                f"name, got {type(expr.array).__name__}"
            ],
        )
    iv = ctx.v(ctx.fresh("iv"), "indexed_variable")
    name_id = _emit_identifier(ctx, expr.array.name)
    ctx.e(iv, name_id, "name")
    il = ctx.v(ctx.fresh("il"), "index_list")
    ctx.e(iv, il, "indices")
    for idx in expr.indices:
        subscript = _index_slot(ctx, idx)
        ctx.e(il, _render(ctx, subscript), _arg_edge_kind(subscript))
    for dim in residual_event_dims(ctx.decl_plates, expr.array.name, len(expr.indices)):
        ctx.e(il, ctx.range_1_to(dim_upper_text(dim)), "range")
    return iv


def _truncating(expr: LetExprCall) -> LetExprNode:
    """Integer division or remainder in BUGS / JAGS arithmetic.

    BUGS and JAGS hold every value as a real and have no integer
    division, so the quotient truncating toward zero is
    ``trunc(a / b)`` and the remainder ``a - b * trunc(a / b)``.

    Parameters
    ----------
    expr : LetExprCall
        A ``quot(a, b)`` or ``rem(a, b)`` call.

    Returns
    -------
    LetExprNode
        The expression in the language's own functions.
    """
    dividend, divisor = expr.args
    quotient = LetExprCall(
        func="trunc", args=(LetExprBinOp(op="/", left=dividend, right=divisor),)
    )
    if expr.func == "quot":
        return quotient
    return LetExprBinOp(
        op="-",
        left=dividend,
        right=LetExprBinOp(op="*", left=divisor, right=quotient),
    )


def _reads_program_value(ctx: _LetEnv, idx: LetExprNode) -> bool:
    """Whether a subscript reads a value the program binds.

    Parameters
    ----------
    ctx : _LetEnv
        The environment, whose declared-plate table names every input,
        site, and binding of the program.
    idx : LetExprNode
        The subscript.

    Returns
    -------
    bool
        True for a bound name or an entry of one, as opposed to a
        renderer's loop variable.
    """
    while isinstance(idx, LetExprIndex):
        idx = idx.array
    return isinstance(idx, LetExprVar) and idx.name in ctx.decl_plates


def _index_slot(ctx: _LetEnv, idx: LetExprNode) -> LetExprNode:
    """One index-list child, rebased to the target's origin.

    QVR subscripts count from zero; BUGS and JAGS count from one. A
    subscript the source spells as an integer literal (a factor
    binder already substituted to its integer coordinate, or a
    literal the model wrote itself) is thus emitted one higher, and so
    is a subscript that reads a program value (an index covariate, or
    an entry of one, or a drawn class): it holds a QVR code. A
    renderer's own loop variable already runs `1:N` and is left alone.

    Integer-literal arithmetic (`2 - t` once the binder `t` is
    substituted) is evaluated first, so it is a literal subscript too,
    as is a pointwise `max` / `min` of such literals. An arithmetic or
    call subscript that still reads a name mixes the two conventions
    with no way to tell which operand carries the origin, so it raises
    rather than emitting an unadjusted expression.
    """
    idx = fold_integers(idx)
    if (
        isinstance(idx, LetExprCall)
        and idx.func in ("max", "min")
        and len(idx.args) == 2
    ):
        # A pointwise `max` / `min` of integer literals (a clamped
        # factor binder, `max(t - 1, 0)` once `t` is substituted) is a
        # literal subscript too.
        values = [
            int(bound.value)
            for bound in (fold_integers(arg) for arg in idx.args)
            if isinstance(bound, LetExprLiteral) and float(bound.value).is_integer()
        ]
        if len(values) == 2:
            chosen = max(values) if idx.func == "max" else min(values)
            return LetExprLiteral(value=float(chosen + 1), integral=True)
    if isinstance(idx, LetExprLiteral):
        return LetExprLiteral(value=float(int(idx.value) + 1), integral=True)
    if _reads_program_value(ctx, idx):
        return LetExprBinOp(
            op="+", left=idx, right=LetExprLiteral(value=1.0, integral=True)
        )
    if isinstance(idx, (LetExprBinOp, LetExprUnaryOp, LetExprCall)):
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprIndex:{_target(ctx)}: arithmetic "
                f"subscript {type(idx).__name__} cannot be rebased "
                f"from the zero-based QVR origin to the one-based "
                f"{_target(ctx).upper()} origin"
            ],
        )
    return idx


def split_event_dims(
    event_dims: tuple[Dim, ...], family_event_rank: int
) -> tuple[tuple[Dim, ...], tuple[Dim, ...]]:
    """Split a site's event dims into the family's own event shape and
    the residual axes the renderer must replicate over.

    A family produces the trailing `family_event_rank` dims natively:
    a Dirichlet over a Topic axis is one draw on the simplex, so the
    Topic axis is the family's own. Every leading dim is residual: it
    replicates the family independently, and a scalar family with an
    `over=` axis is all residual. BUGS and JAGS have no vector form
    for a scalar family, so a residual axis becomes a loop rather
    than a slice.

    Returns `(native, residual)`.
    """
    rank = max(0, min(family_event_rank, len(event_dims)))
    if rank == 0:
        return (), event_dims
    return event_dims[len(event_dims) - rank :], event_dims[: len(event_dims) - rank]


def dim_upper_text(dim: Dim) -> str:
    """Return the upper-bound text of one plate dim."""
    if isinstance(dim, DimStatic):
        return str(int(dim.size))
    if isinstance(dim, DimDynamic):
        return str(dim.size_name)
    raise UnsupportedConstruct(
        "qvr-bugs-helper",
        [f"dim:{type(dim).__name__}: unknown shape"],
    )


def residual_event_dims(
    decl_plates: dict[str, Plate], name: str, supplied: int
) -> tuple[Dim, ...]:
    """Return the event dims a subscript of `name` leaves unconsumed.

    A gather that supplies exactly one subscript per declared batch
    dim addresses a whole event block, which BUGS / JAGS spell as an
    explicit trailing `1:E` slice per event axis. A subscript list
    that already reaches into the event axes consumes them itself and
    needs no slice.
    """
    plate = decl_plates.get(name)
    if plate is None or not plate.event_dims:
        return ()
    if supplied == len(plate.batch_dims):
        return plate.event_dims
    return ()


def axis_rank(decl_plates: dict[str, Plate], expr: LetExprNode) -> int:
    """Return the number of axes the value of `expr` still carries.

    A bare name carries every axis of its declared plate; a subscript
    consumes one axis per index it supplies. Operators propagate the
    widest operand. Anything the helper cannot resolve reads as a
    scalar, which is the conservative answer: the reduction rewrites
    below only fire on a positively-ranked operand.
    """
    if isinstance(expr, LetExprVar):
        plate = decl_plates.get(expr.name)
        if plate is None:
            return 0
        return len(plate.batch_dims) + len(plate.event_dims)
    if isinstance(expr, LetExprIndex):
        if not isinstance(expr.array, LetExprVar):
            return 0
        plate = decl_plates.get(expr.array.name)
        if plate is None:
            return 0
        declared = len(plate.batch_dims) + len(plate.event_dims)
        return max(declared - len(expr.indices), 0)
    if isinstance(expr, LetExprBinOp):
        return max(
            axis_rank(decl_plates, expr.left),
            axis_rank(decl_plates, expr.right),
        )
    if isinstance(expr, LetExprUnaryOp):
        return axis_rank(decl_plates, expr.operand)
    if isinstance(expr, (LetExprList, LetExprFactor)):
        return 1
    if isinstance(expr, LetExprAffineMap):
        # The map is emitted one row per loop iteration, so what
        # reaches the surrounding relation is a scalar.
        return 0
    return 0


def _emit_list(ctx: _LetEnv, items: tuple[LetExprNode, ...]) -> str:
    """Render a list literal as the BUGS / JAGS `c(...)` combine call.

    Neither language has an inline list-literal surface form; the
    canonical concatenation idiom is the built-in `c(...)` function
    (S-style combine), which both languages parse as a regular
    `function_call`.
    """
    return _emit_call(
        ctx,
        "c",
        tuple(_render(ctx, item) for item in items),
        tuple(_arg_edge_kind(item) for item in items),
    )


def _emit_factor(ctx: _LetEnv, expr: LetExprFactor) -> str:
    """Unroll a rank-1 `factor` expression into the `c(...)` combine
    call BUGS / JAGS use for a vector literal.

    Only the single-binder form has a vector spelling. A multi-binder
    factor denotes a rank-`n` tensor, and neither language has a
    reshape that would turn a flat `c(...)` back into one, so the
    tensor form is emitted as one relation per cell by
    `factor_cells`
    at the statement level rather than as an expression here.
    """
    cells = factor_cells(ctx, expr)
    if len(expr.binders) != 1:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: a "
                f"{len(expr.binders)}-binder factor denotes a rank-"
                f"{len(expr.binders)} tensor, which has no BUGS / "
                f"JAGS expression form; only a statement-level "
                f"binding unrolls it"
            ],
        )
    elements = tuple(_render(ctx, body) for _, body in cells)
    element_kinds = tuple(_arg_edge_kind(body) for _, body in cells)
    return _emit_call(ctx, "c", elements, element_kinds)


def factor_cells(
    ctx: _BugsLetCtx, expr: LetExprFactor
) -> tuple[tuple[tuple[int, ...], LetExprNode], ...]:
    """Enumerate a `factor` expression's cells.

    Returns one `(indices, body)` pair per point of the binders'
    product index space, row-major, with every binder substituted by
    its integer coordinate in the body. `indices` counts from zero,
    matching the QVR origin; the caller rebases it when it writes the
    left-hand subscripts.

    The single-axis `cases` form labels each case body by integer;
    the helper enumerates `0, ..., |I|-1` and looks up the matching
    case, raising when the label set is incomplete.
    """
    if not expr.binders:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: empty binder "
                f"list is structurally ill-formed"
            ],
        )
    sizes = tuple(_factor_axis_size(ctx, b) for b in expr.binders)
    if expr.cases:
        if len(expr.binders) != 1:
            raise UnsupportedConstruct(
                f"qvr-{_target(ctx)}-helper",
                [
                    f"let-expr:LetExprFactor:{_target(ctx)}: cases "
                    f"form requires exactly one binder, got "
                    f"{len(expr.binders)}"
                ],
            )
        return _factor_case_cells(ctx, expr.binders[0], sizes[0], expr.cases)
    if expr.body is None:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [f"let-expr:LetExprFactor:{_target(ctx)}: missing body and no cases"],
        )
    body = expr.body
    return tuple(
        (
            indices,
            _substitute(
                body,
                dict(zip((b.var for b in expr.binders), indices, strict=True)),
            ),
        )
        for indices in _enumerate_indices(sizes)
    )


def _factor_case_cells(
    ctx: _BugsLetCtx,
    binder: LetFactorBinder,
    size: int,
    cases: tuple[LetFactorCase, ...],
) -> tuple[tuple[tuple[int, ...], LetExprNode], ...]:
    """Enumerate the label-keyed case list as `((label,), body)` cells."""
    by_label = {c.label: c.value for c in cases}
    missing = sorted(set(range(size)) - by_label.keys())
    if missing:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: cases form "
                f"missing labels {missing} for binder "
                f"{binder.var!r} of size {size}"
            ],
        )
    return tuple(((label,), by_label[label]) for label in range(size))


def factor_axis_sizes(ctx: _BugsLetCtx, expr: LetExprFactor) -> tuple[int, ...]:
    """Return the static extent of each of `expr`'s binder axes."""
    return tuple(_factor_axis_size(ctx, b) for b in expr.binders)


# ---------------------------------------------------------------------------
# Factor-binder support: axis-size lookup and body substitution.
# ---------------------------------------------------------------------------


def _factor_axis_size(ctx: _BugsLetCtx, binder: LetFactorBinder) -> int:
    """Resolve a factor binder's axis to a static integer size.

    Looks up the binder's index expression in ``ctx.cards``. Raises
    when the axis is a constructor with no static size or when the
    name is unknown.
    """
    name = _object_expr_axis_name(ctx, binder.index)
    size = ctx.cards.get(name)
    if size is None:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: axis "
                f"{name!r} (binder {binder.var!r}) has no static "
                f"cardinality in `ctx.cards`"
            ],
        )
    return int(size)


def _object_expr_axis_name(ctx: _BugsLetCtx, obj: ObjectExpr) -> str:
    """Resolve an `ObjectExpr` to the axis name a `cards` lookup wants.

    Handles `TypeName` directly and `DiscreteConstructor("FinSet", N)`
    as a literal anonymous axis (the size is the integer arg).
    """
    if isinstance(obj, TypeName):
        return obj.name
    if isinstance(obj, DiscreteConstructor) and obj.constructor == "FinSet":
        if len(obj.args) == 1 and obj.args[0].isdigit():
            return obj.args[0]
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: "
                f"non-literal FinSet binder {obj.args!r}"
            ],
        )
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [
            f"let-expr:LetExprFactor:{_target(ctx)}: binder "
            f"index {type(obj).__name__} not supported"
        ],
    )


def _enumerate_indices(
    sizes: tuple[int, ...],
) -> list[tuple[int, ...]]:
    """Row-major enumeration of `product(range(s) for s in sizes)`."""
    if not sizes:
        return [()]
    head, *rest = sizes
    rest_tup = tuple(rest)
    tails = _enumerate_indices(rest_tup)
    return [(i, *tail) for i in range(head) for tail in tails]


def _substitute(expr: LetExprNode, env: dict[str, int]) -> LetExprNode:
    """Substitute integer literals for every `LetExprVar` whose name
    appears in `env`. Recurses through every `LetExprNode` shape."""
    if isinstance(expr, LetExprVar):
        v = env.get(expr.name)
        if v is None:
            return expr
        return LetExprLiteral(value=float(v))
    if isinstance(expr, LetExprLiteral):
        return expr
    if isinstance(expr, LetExprString):
        return expr
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_substitute(expr.left, env),
            right=_substitute(expr.right, env),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(operand=_substitute(expr.operand, env))
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(_substitute(a, env) for a in expr.args),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_substitute(expr.array, env),
            indices=tuple(_substitute(i, env) for i in expr.indices),
        )
    if isinstance(expr, LetExprList):
        return LetExprList(
            items=tuple(_substitute(i, env) for i in expr.items),
        )
    if isinstance(expr, LetExprLambda):
        # Shadowing: drop the bound name from the substitution.
        inner_env = {k: v for k, v in env.items() if k != expr.param}
        return LetExprLambda(
            param=expr.param,
            body=_substitute(expr.body, inner_env),
        )
    if isinstance(expr, LetExprMethodCall):
        return LetExprMethodCall(
            receiver=_substitute(expr.receiver, env),
            method=expr.method,
            args=tuple(_substitute(a, env) for a in expr.args),
        )
    if isinstance(expr, LetExprFactor):
        # Shadowing: drop any rebound names from the substitution.
        bound = {b.var for b in expr.binders}
        inner_env = {k: v for k, v in env.items() if k not in bound}
        return LetExprFactor(
            binders=expr.binders,
            body=(_substitute(expr.body, inner_env) if expr.body is not None else None),
            cases=tuple(
                LetFactorCase(
                    label=c.label,
                    value=_substitute(c.value, inner_env),
                    line=c.line,
                    col=c.col,
                )
                for c in expr.cases
            ),
        )
    return expr


# ---------------------------------------------------------------------------
# Edge-kind lookup: the grammar kind that a child vertex contributes.
# ---------------------------------------------------------------------------


def _arg_edge_kind(expr: LetExprNode) -> str:
    """Return the grammar kind a child vertex registers under.

    BUGS / JAGS `argument_list` / `index_list` discriminate child
    alternatives by vertex kind (the panproto walker reads the child's
    actual kind from the parent's `chose-alt-child-kinds` slot when
    set; otherwise it uses the edge label). The helper labels every
    edge from a list-like parent with the child's grammar kind so the
    parent stays compatible with either resolution path.
    """
    if isinstance(expr, LetExprLiteral):
        return "number"
    if isinstance(expr, LetExprVar):
        return "identifier"
    if isinstance(expr, LetExprBinOp):
        return "binary_expression"
    if isinstance(expr, LetExprUnaryOp):
        return "unary_expression"
    if isinstance(expr, LetExprCall):
        return "function_call"
    if isinstance(expr, LetExprIndex):
        return "indexed_variable"
    if isinstance(expr, (LetExprList, LetExprFactor)):
        # List / factor unroll to `c(...)`, a `function_call`.
        return "function_call"
    # Lambda / method-call / string have no parent edge kind because
    # the helper raises before reaching the parent.
    return "identifier"


def _target(ctx: _BugsLetCtx) -> str:
    """Read the ctx's `target` tag for error messages."""
    return getattr(ctx, "target", "bugs")


# ---------------------------------------------------------------------------
# Shared IR pre-pass: give every empty-plate IRDeterministic whose value
# is axis-carrying the plate it needs to emit as a loop. BUGS and JAGS
# each lack a scalar-to-vector broadcast operator and lift no infix
# operator over an axis, so `let mu = a + b * x_design` and
# `let beta = tau * lambda_local * z_raw` are both rank errors at the
# top level; the only spelling either language has for them is
# ``for (i in 1:N) { mu[i] <- ... }``. Two rules supply the missing
# plate: the plate of the deterministic's first downstream consumer,
# and the plate the expression's own operands already carry.
# ---------------------------------------------------------------------------


def push_scalar_dets_into_loops(ir: IRProgram) -> IRProgram:
    """Lift every empty-plate `IRDeterministic` that denotes an
    axis-carrying value into the plate that value ranges over.

    Two rules supply the plate, in order:

    1. The *consumer* rule covers a deterministic whose expression
       references a plate-less free data input: the plate is that of
       the first `IRObserve` / `IRSample` whose args contain an
       `IRArgRef` to the deterministic's bound name, and the
       referenced free data inputs are retagged with it so subsequent
       emission rebroadcasts them consistently.
    2. The *operand* rule covers a deterministic whose expression
       combines bindings that already carry a plate of their own (the
       horseshoe product ``tau * lambda_local * z_raw`` over a
       coefficient axis): the plate is the one every bare
       positively-ranked operand shares.

    A deterministic whose operands disagree on their axes, or whose
    value the reduction rewrites have already collapsed to a scalar
    (``sum(z * w)`` lowering to ``inprod``), is left alone so the
    emitter reports the shape it cannot spell rather than inventing a
    loop for it.
    """
    det_plate, input_plate_overrides = _consumer_plate_lifts(ir)
    det_plate.update(_operand_plate_lifts(ir, det_plate))
    if not det_plate:
        return ir
    new_inputs = tuple(
        IRDataInput(
            name=inp.name,
            constraint=inp.constraint,
            plate=input_plate_overrides.get(inp.name, inp.plate),
        )
        for inp in ir.inputs
    )
    new_body: list[IRNode] = []
    for node in ir.body:
        if isinstance(node, IRDeterministic) and node.name in det_plate:
            new_body.append(
                IRDeterministic(
                    name=node.name,
                    expr=node.expr,
                    constraint=node.constraint,
                    plate=det_plate[node.name],
                )
            )
        else:
            new_body.append(node)
    return IRProgram(
        name=ir.name,
        inputs=new_inputs,
        body=tuple(new_body),
        module=ir.module,
        cards=ir.cards,
    )


def _consumer_plate_lifts(
    ir: IRProgram,
) -> tuple[dict[str, Plate], dict[str, Plate]]:
    """The consumer rule of
    `push_scalar_dets_into_loops`.

    Returns the plate each lifted deterministic acquires alongside the
    plate override each free data input it reads acquires with it.
    """
    free_input_names: set[str] = set()
    for inp in ir.inputs:
        if not inp.plate.batch_dims and not inp.plate.event_dims:
            free_input_names.add(inp.name)
    det_to_free_refs: dict[str, frozenset[str]] = {}
    for node in ir.body:
        if not isinstance(node, IRDeterministic):
            continue
        if node.plate.batch_dims or node.plate.event_dims:
            continue
        free_refs = collect_letexpr_vars(node.expr) & free_input_names
        if free_refs:
            det_to_free_refs[node.name] = frozenset(free_refs)
    if not det_to_free_refs:
        return {}, {}
    det_consumer_plate: dict[str, Plate] = {}
    for node in ir.body:
        if isinstance(node, (IRObserve, IRSample)) and (
            node.plate.batch_dims or node.plate.event_dims
        ):
            referenced = collect_irargref_names(node.args)
            for det_name in det_to_free_refs:
                if det_name in referenced and det_name not in det_consumer_plate:
                    det_consumer_plate[det_name] = node.plate
    input_plate_overrides: dict[str, Plate] = {}
    for det_name, free_refs in det_to_free_refs.items():
        consumer_plate = det_consumer_plate.get(det_name)
        if consumer_plate is None:
            continue
        for free_ref in free_refs:
            input_plate_overrides[free_ref] = consumer_plate
    return det_consumer_plate, input_plate_overrides


def _operand_plate_lifts(
    ir: IRProgram, already_lifted: dict[str, Plate]
) -> dict[str, Plate]:
    """The operand rule of
    `push_scalar_dets_into_loops`.

    A deterministic qualifies when its value still carries an axis
    (`axis_rank`
    is positive) and every bare operand that carries one agrees on a
    single batch-only plate. The map is built in body order and folded
    back into the declared-plate table as it goes, so a chain of
    deterministics each built from the one before lifts as a whole.
    """
    decl = build_decl_plates(ir)
    decl.update(already_lifted)
    out: dict[str, Plate] = {}
    for node in ir.body:
        if not isinstance(node, IRDeterministic):
            continue
        if node.plate.batch_dims or node.plate.event_dims:
            continue
        if node.name in already_lifted:
            continue
        if axis_rank(decl, node.expr) <= 0:
            continue
        plate = _shared_operand_plate(node.expr, decl)
        if plate is None:
            continue
        out[node.name] = plate
        decl[node.name] = plate
    return out


def _shared_operand_plate(
    expr: LetExprNode, decl_plates: dict[str, Plate]
) -> Plate | None:
    """The single batch-only plate every bare axis-carrying operand of
    `expr` shares, or `None` when they disagree or none carries one.

    Only bare `LetExprVar` operands count. A subscripted reference has
    already consumed the axes its indices supply, so the loop variable
    the lift would introduce has no position to occupy in it, and a
    plate carrying event dims needs a slice rather than a loop index.
    Both leave the deterministic where it is.
    """
    plates: list[Plate] = []
    _collect_bare_var_plates(expr, decl_plates, plates)
    if not plates:
        return None
    first = plates[0]
    if first.event_dims:
        return None
    for plate in plates[1:]:
        if plate.batch_dims != first.batch_dims or plate.event_dims:
            return None
    return first


def _collect_bare_var_plates(
    expr: LetExprNode, decl_plates: dict[str, Plate], out: list[Plate]
) -> None:
    """Append the declared plate of every bare axis-carrying variable
    reachable from `expr` without passing through a subscript."""
    if isinstance(expr, LetExprVar):
        plate = decl_plates.get(expr.name)
        if plate is not None and plate.batch_dims:
            out.append(plate)
        return
    if isinstance(expr, LetExprBinOp):
        _collect_bare_var_plates(expr.left, decl_plates, out)
        _collect_bare_var_plates(expr.right, decl_plates, out)
        return
    if isinstance(expr, LetExprUnaryOp):
        _collect_bare_var_plates(expr.operand, decl_plates, out)
        return
    if isinstance(expr, LetExprCall):
        for arg in expr.args:
            _collect_bare_var_plates(arg, decl_plates, out)
        return
    if isinstance(expr, LetExprIndex):
        for index in expr.indices:
            _collect_bare_var_plates(index, decl_plates, out)
        return


def collect_letexpr_vars(expr: LetExprNode) -> frozenset[str]:
    """Collect every bare-variable name in a let-expression tree."""
    if isinstance(expr, LetExprVar):
        return frozenset({expr.name})
    if isinstance(expr, LetExprLiteral):
        return frozenset()
    if isinstance(expr, LetExprBinOp):
        return collect_letexpr_vars(expr.left) | collect_letexpr_vars(expr.right)
    if isinstance(expr, LetExprUnaryOp):
        return collect_letexpr_vars(expr.operand)
    if isinstance(expr, LetExprCall):
        out: frozenset[str] = frozenset()
        for a in expr.args:
            out = out | collect_letexpr_vars(a)
        return out
    if isinstance(expr, LetExprIndex):
        out2: frozenset[str] = collect_letexpr_vars(expr.array)
        for ix in expr.indices:
            out2 = out2 | collect_letexpr_vars(ix)
        return out2
    if isinstance(expr, LetExprAffineMap):
        out3: frozenset[str] = collect_letexpr_vars(expr.weight) | collect_letexpr_vars(
            expr.bias
        )
        for source in expr.sources:
            out3 = out3 | collect_letexpr_vars(source.value)
        return out3
    return frozenset()


def collect_irargref_names(args: tuple[IRArg, ...]) -> frozenset[str]:
    """Collect every `IRArgRef.name` reachable via the arg tuple."""
    out: set[str] = set()
    for a in args:
        _collect_irargref_names_into(a, out)
    return frozenset(out)


def _collect_irargref_names_into(arg: IRArg, out: set[str]) -> None:
    if isinstance(arg, IRArgRef):
        out.add(arg.name)
        for ix in arg.indices:
            _collect_irargref_names_into(ix, out)
        return
    if isinstance(arg, IRArgBroadcast):
        _collect_irargref_names_into(arg.value, out)
        return
    if isinstance(arg, IRArgList):
        for el in arg.elements:
            _collect_irargref_names_into(el, out)
        return
    if isinstance(arg, IRArgMatrix):
        for row in arg.rows:
            for el in row.elements:
                _collect_irargref_names_into(el, out)
        return
    # `IRArgTransform` (renderer-local wrapper) is structurally a
    # nested IRArg; treat it as opaque here -- nothing in the
    # pre-pass examines transform-wrapped args before the renderer
    # injects them downstream.


def build_decl_plates(ir: IRProgram) -> dict[str, Plate]:
    """Build the declared-plate map for every named binding.

    Combines `ir.inputs` and every node in `ir.body` so the let-expr
    re-indexer can look up the plate of any reference encountered.
    """
    out: dict[str, Plate] = {}
    for inp in ir.inputs:
        out[inp.name] = inp.plate
    stack: list[IRNode] = list(ir.body)
    while stack:
        node = stack.pop()
        if isinstance(node, IRSample):
            out[node.name] = node.plate
        elif isinstance(node, IRObserve):
            out[node.name] = node.plate
        elif isinstance(node, IRDeterministic):
            out[node.name] = node.plate
        elif isinstance(node, IRMarginalize):
            out[node.latent] = node.plate
            stack.extend(node.scope)
    return out


def hoist_list_operands(
    node: IRDeterministic, taken: frozenset[str]
) -> tuple[tuple[IRDeterministic, ...], IRDeterministic]:
    """Bind each list-literal operand of a plated binding to a name.

    A plated binding is emitted as a loop whose body reads each plated
    operand at the loop's row. BUGS and JAGS have no subscript on a
    literal (`c(2.5, 0.5)[i]`), so a list literal combined elementwise
    with a plated value, as in `ys + [2.5, 0.5, 3.5, 1.0]`, is bound to
    a fresh name over the plate's trailing axes (QVR aligns an operand
    of lower rank with the trailing axes), emitted cell by cell like any
    list `let`, and read at the row through that name.

    Parameters
    ----------
    node : IRDeterministic
        The plated binding.
    taken : frozenset[str]
        The names the program already binds.

    Returns
    -------
    tuple[tuple[IRDeterministic, ...], IRDeterministic]
        The bindings that name the literals, in the order they must be
        emitted, and the binding with each literal replaced by its name.
    """
    hoisted: list[IRDeterministic] = []
    used = set(taken)

    def bind(expr: LetExprNode) -> LetExprNode:
        """Replace the list literals among an expression's operands.

        Parameters
        ----------
        expr : LetExprNode
            The expression.

        Returns
        -------
        LetExprNode
            The expression reading each literal through its name.
        """
        if isinstance(expr, LetExprList):
            rank = len(list_sizes(expr))
            if 0 < rank <= len(node.plate.batch_dims):
                serial = 0
                name = f"{node.name}_lit{serial}"
                while name in used:
                    serial += 1
                    name = f"{node.name}_lit{serial}"
                used.add(name)
                hoisted.append(
                    IRDeterministic(
                        name=name,
                        expr=expr,
                        constraint=node.constraint,
                        plate=Plate(
                            event_dims=(), batch_dims=node.plate.batch_dims[-rank:]
                        ),
                    )
                )
                return LetExprVar(name=name)
            return expr
        if isinstance(expr, LetExprBinOp):
            return LetExprBinOp(
                op=expr.op, left=bind(expr.left), right=bind(expr.right)
            )
        if isinstance(expr, LetExprUnaryOp):
            return LetExprUnaryOp(op=expr.op, operand=bind(expr.operand))
        if isinstance(expr, LetExprCall):
            return LetExprCall(func=expr.func, args=tuple(bind(a) for a in expr.args))
        return expr

    if not node.plate.batch_dims or isinstance(node.expr, (LetExprList, LetExprFactor)):
        return (), node
    rewritten = bind(node.expr)
    return tuple(hoisted), IRDeterministic(
        name=node.name,
        expr=rewritten,
        constraint=node.constraint,
        plate=node.plate,
    )


def index_letexpr_refs(
    expr: LetExprNode,
    decl_plates: dict[str, Plate],
    enclosing_plate: Plate,
    loop_names: tuple[str, ...],
) -> LetExprNode:
    """Rewrite each `LetExprVar` whose declared plate shares axes with
    ``enclosing_plate`` into a `LetExprIndex` indexed by the matching
    loop variables.

    Axes are matched by name: for each axis in the var's declared
    batch_dims the helper looks up the parallel loop variable in
    ``enclosing_plate`` and emits it as the index expression. Vars
    whose declared plate has no axes in common with the surrounding
    loop stay as bare names so they broadcast as constants per
    iteration; vars whose declared plate has an axis the surrounding
    loop does not iterate are also left bare so the emitter can flag
    the shape mismatch downstream rather than silently picking the
    wrong loop variable.
    """
    if not loop_names:
        return expr
    axis_to_loop: dict[str, str] = {}
    for dim, lname in zip(enclosing_plate.batch_dims, loop_names, strict=True):
        axis_to_loop[dim.name] = lname
    return _index_letexpr_refs_inner(expr, decl_plates, axis_to_loop)


def _index_letexpr_refs_inner(
    expr: LetExprNode,
    decl_plates: dict[str, Plate],
    axis_to_loop: dict[str, str],
) -> LetExprNode:
    if isinstance(expr, LetExprVar):
        plate = decl_plates.get(expr.name)
        if plate is None or not plate.batch_dims:
            return expr
        indices: list[LetExprNode] = []
        for position, dim in enumerate(plate.batch_dims):
            lname = axis_to_loop.get(dim.name)
            if lname is None or any(
                earlier.name == dim.name for earlier in plate.batch_dims[:position]
            ):
                # An axis the loop does not iterate, or a second axis over
                # the same object (a factor's `A`-by-`A` table), has no
                # loop variable of its own: the name stays whole.
                return expr
            indices.append(LetExprVar(name=lname))
        return LetExprIndex(
            array=LetExprVar(name=expr.name),
            indices=tuple(indices),
        )
    if isinstance(expr, LetExprLiteral):
        return expr
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_index_letexpr_refs_inner(expr.left, decl_plates, axis_to_loop),
            right=_index_letexpr_refs_inner(expr.right, decl_plates, axis_to_loop),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            operand=_index_letexpr_refs_inner(expr.operand, decl_plates, axis_to_loop),
        )
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(
                _index_letexpr_refs_inner(a, decl_plates, axis_to_loop)
                for a in expr.args
            ),
        )
    if isinstance(expr, LetExprIndex):
        subscripts = tuple(
            _index_letexpr_refs_inner(ix, decl_plates, axis_to_loop)
            for ix in expr.indices
        )
        # The subscripts select the array's leading axes; an axis left
        # after them that the surrounding loop iterates is read at the
        # loop's row too, as a gathered table's class axis is in
        # `row_table[n, k] <- table[group_idx[n], k]`.
        trailing: tuple[LetExprNode, ...] = ()
        if isinstance(expr.array, LetExprVar):
            plate = decl_plates.get(expr.array.name)
            if plate is not None:
                trailing = tuple(
                    LetExprVar(name=axis_to_loop[str(dim.name)])
                    for dim in plate.batch_dims[len(subscripts) :]
                    if str(dim.name) in axis_to_loop
                )
        return LetExprIndex(array=expr.array, indices=(*subscripts, *trailing))
    if isinstance(expr, LetExprAffineMap):
        return LetExprAffineMap(
            weight=_index_letexpr_refs_inner(expr.weight, decl_plates, axis_to_loop),
            bias=_index_letexpr_refs_inner(expr.bias, decl_plates, axis_to_loop),
            sources=tuple(
                LetAffineSource(
                    value=_index_letexpr_refs_inner(
                        source.value, decl_plates, axis_to_loop
                    ),
                    width=source.width,
                )
                for source in expr.sources
            ),
            row_offset=expr.row_offset,
            rows=expr.rows,
            transform=expr.transform,
        )
    return expr


__all__ = [
    "hoist_list_operands",
    "axis_rank",
    "build_decl_plates",
    "collect_irargref_names",
    "collect_letexpr_vars",
    "dim_upper_text",
    "factor_axis_sizes",
    "factor_cells",
    "index_letexpr_refs",
    "push_scalar_dets_into_loops",
    "render_let_expr_bugs",
    "residual_event_dims",
    "split_event_dims",
]


class CategoricalMixture(dx.Model):
    """The collapsed form of a marginalized categorical latent.

    A [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] whose
    latent is a `Categorical` over the rows of a stochastic matrix,
    and whose scope is one scalar `Categorical` observation of a row
    picked by that latent, denotes the finite mixture

        p(y = v) = sum_k weights[k] * rows[k, v],

    which is itself a categorical measure on the observation's
    alphabet. This carries the pieces that mixture is written from:
    the mixing-weight reference, the row-matrix name, the atom axis
    the sum runs over, the alphabet axis the mixture lives on, and
    the scope's observation, against whose plate the renderer builds
    the mixture and whose `probs` argument it rewrites.

    The recogniser
    `categorical_mixture`
    is the only constructor; every field it fills is read back out of
    the declaration-plate table, so a renderer can emit the mixture
    without re-deriving any shape.
    """

    weights: IRArgRef
    rows_name: str
    atom_dim: Dim
    alphabet_dim: Dim
    observe: IRObserve


def _axis_key(dim: Dim) -> tuple[str, str]:
    """Return the (axis name, extent) pair two dims are compared on."""
    return (str(dim.name), dim_upper_text(dim))


def categorical_mixture(
    node: IRMarginalize,
    decl_plates: dict[str, Plate],
) -> CategoricalMixture | None:
    """Recognize a categorical mixture that BUGS can collapse to `dcat`.

    The recognized form marginalizes a categorical latent, indexes a matrix of
    categorical rows by that latent, and observes a scalar categorical value.
    The mixing weights carry the latent's batch and atom axes; the row matrix
    carries the atom and output-alphabet axes. Their weighted sum is a
    categorical probability vector. Returns `None` for other marginalization
    shapes.
    """
    if node.family != "Categorical":
        return None
    if node.arg_names != ("probs",) or len(node.args) != 1:
        return None
    weights = node.args[0]
    if not isinstance(weights, IRArgRef) or weights.indices:
        return None
    weight_plate = decl_plates.get(weights.name)
    if weight_plate is None or len(weight_plate.event_dims) != 1:
        return None
    weight_axes = tuple(_axis_key(d) for d in weight_plate.batch_dims)
    latent_axes = frozenset(_axis_key(d) for d in node.plate.batch_dims)
    if len(frozenset(weight_axes)) != len(weight_axes) or not (
        frozenset(weight_axes) <= latent_axes
    ):
        return None
    if len(node.scope) != 1 or not isinstance(node.scope[0], IRObserve):
        return None
    observe = node.scope[0]
    if observe.family != "Categorical":
        return None
    if observe.arg_names != ("probs",) or len(observe.args) != 1:
        return None
    if observe.plate.event_dims:
        return None
    rows = observe.args[0]
    if (
        not isinstance(rows, IRArgRef)
        or len(rows.indices) != 1
        or not isinstance(rows.indices[0], IRArgRef)
        or rows.indices[0].name != node.latent
        or rows.indices[0].indices
    ):
        return None
    row_plate = decl_plates.get(rows.name)
    if (
        row_plate is None
        or len(row_plate.batch_dims) != 1
        or len(row_plate.event_dims) != 1
    ):
        return None
    atom_dim = row_plate.batch_dims[0]
    if _axis_key(atom_dim) != _axis_key(weight_plate.event_dims[0]):
        return None
    return CategoricalMixture(
        weights=weights,
        rows_name=rows.name,
        atom_dim=atom_dim,
        alphabet_dim=row_plate.event_dims[0],
        observe=observe,
    )


class MarginalScopeDensity(dx.Model):
    """The density of one marginalized atom's scope, in closed form.

    A `marginalize` block contributes ``log sum_a w_a f_a`` to the
    joint, where `f_a` is the *density* (not the log density) the
    scope scores at atom `a`. BUGS and JAGS have no `log_prob` form
    for a named distribution, so a renderer that writes the sum has
    to write each `f_a` out; this carries one such term together with
    the property that decides whether the reduction needs the zeros
    trick's lift.

    `mass` is True when the family is a probability *mass* function,
    whose value never exceeds one. The mixture of such densities is
    then at most one as well, so ``-log sum_a w_a f_a`` is
    non-negative and the Poisson rate the zeros trick writes it into
    is in support with no lift, which keeps the emitted program equal
    to the reference measure on the nose. A family with a *density*
    can exceed one and obliges the lift.
    """

    expr: LetExprNode
    mass: bool


def inline_letexpr(expr: LetExprNode, bindings: dict[str, LetExprNode]) -> LetExprNode:
    """Substitute every `LetExprVar` named in `bindings` by its
    expression.

    The marginal reduction writes one arithmetic expression per atom
    rather than one relation per atom, because a BUGS / JAGS name may
    be defined exactly once and the atoms would otherwise each want
    their own copy of the scope's deterministic bindings under the
    same names. Inlining removes the need for a name at all.
    """
    if isinstance(expr, LetExprVar):
        return bindings.get(expr.name, expr)
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=inline_letexpr(expr.left, bindings),
            right=inline_letexpr(expr.right, bindings),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(operand=inline_letexpr(expr.operand, bindings))
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(inline_letexpr(a, bindings) for a in expr.args),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=expr.array,
            indices=tuple(inline_letexpr(i, bindings) for i in expr.indices),
        )
    return expr


def irarg_letexpr(
    backend: str, arg: IRArg, bindings: dict[str, LetExprNode]
) -> LetExprNode:
    """Read one distribution argument back as a let-expression.

    The marginal reduction consumes its arguments as arithmetic
    operands rather than as distribution-call children, so each one
    has to come back as a `LetExprNode`. A reference to a name the
    scope binds resolves to that binding's inlined expression; every
    other bare reference stays a name the surrounding relation
    re-indexes, and a subscripted reference keeps its subscripts.

    A broadcast wrapper, a list, or a matrix literal carries
    structure no scalar arithmetic operand can place, so each raises
    rather than emitting an expression of the wrong shape.
    """
    if isinstance(arg, IRArgNumber):
        return LetExprLiteral(value=arg.value)
    if isinstance(arg, IRArgRef):
        if not arg.indices:
            return bindings.get(arg.name, LetExprVar(name=arg.name))
        if arg.name in bindings:
            raise UnsupportedConstruct(
                f"qvr-{backend}",
                [
                    f"marginalize:subscripted-binding:{arg.name}: the "
                    f"scope binds this name to an expression, which a "
                    f"subscript has no position to index into"
                ],
            )
        return LetExprIndex(
            array=LetExprVar(name=arg.name),
            indices=tuple(
                irarg_letexpr(backend, index, bindings) for index in arg.indices
            ),
        )
    raise UnsupportedConstruct(
        f"qvr-{backend}",
        [
            f"marginalize:scope-arg:{type(arg).__name__}: the "
            f"closed-form scope density reads each argument as a "
            f"scalar arithmetic operand"
        ],
    )


def subscript_letexpr(
    backend: str, base: LetExprNode, index: LetExprNode
) -> LetExprNode:
    """Append one subscript to `base`.

    BUGS and JAGS spell a two-axis lookup with one bracket pair,
    `a[i, j]`, and reject the chained `a[i][j]` a naive wrapper would
    build, so a subscript of an already-subscripted reference extends
    the existing index list rather than nesting a second one.
    """
    if isinstance(base, LetExprVar):
        return LetExprIndex(array=base, indices=(index,))
    if isinstance(base, LetExprIndex):
        return LetExprIndex(array=base.array, indices=(*base.indices, index))
    raise UnsupportedConstruct(
        f"qvr-{backend}",
        [
            f"marginalize:subscript:{type(base).__name__}: only a "
            f"named array can be subscripted in a BUGS / JAGS model "
            f"body"
        ],
    )


def marginal_scope_density(
    backend: str,
    *,
    family: str,
    variate: str,
    args: tuple[LetExprNode, ...],
    arg_names: tuple[str, ...],
) -> MarginalScopeDensity:
    """Build a closed-form BUGS or JAGS density for one marginal atom.

    Supported scope families are `Categorical`, `Poisson`, and `Normal`.
    Categorical density indexes the probability vector with the one-based host
    outcome. Poisson density uses ``pow(mu, y)`` so zero-rate atoms retain
    ``pow(0, 0) = 1`` without evaluating ``log(0)``. Normal density uses the
    Gaussian kernel and normalizer. Other families raise
    `UnsupportedConstruct`.
    """
    by_name = dict(zip(arg_names, args, strict=False))

    def required(slot: str) -> LetExprNode:
        value = by_name.get(slot)
        if value is None:
            raise UnsupportedConstruct(
                f"qvr-{backend}",
                [
                    f"marginalize:scope-arg:{family}:{slot}: the "
                    f"closed-form density needs the {slot!r} "
                    f"argument; the site supplies {list(arg_names)}"
                ],
            )
        return value

    y = LetExprVar(name=variate)
    if family == "Categorical":
        return MarginalScopeDensity(
            expr=subscript_letexpr(backend, required("probs"), y),
            mass=True,
        )
    if family == "Poisson":
        rate = required("rate")
        return MarginalScopeDensity(
            expr=_let_mul(
                LetExprCall(
                    func="exp",
                    args=(
                        _let_sub(
                            LetExprUnaryOp(operand=rate),
                            LetExprCall(func=_LOG_FACTORIAL, args=(y,)),
                        ),
                    ),
                ),
                LetExprCall(func="pow", args=(rate, y)),
            ),
            mass=True,
        )
    if family == "Normal":
        loc = required("loc")
        scale = required("scale")
        standardised = LetExprBinOp(op="/", left=_let_sub(y, loc), right=scale)
        return MarginalScopeDensity(
            expr=LetExprBinOp(
                op="/",
                left=LetExprCall(
                    func="exp",
                    args=(
                        _let_mul(
                            LetExprLiteral(value=-0.5),
                            _let_mul(standardised, standardised),
                        ),
                    ),
                ),
                right=_let_mul(scale, LetExprLiteral(value=SQRT_TWO_PI)),
            ),
            mass=False,
        )
    raise UnsupportedConstruct(
        f"qvr-{backend}",
        [
            f"marginalize:scope-family:{family}: the reduction over "
            f"the latent's atoms needs the scope's density as an "
            f"arithmetic expression, and no closed form is written "
            f"for this family"
        ],
    )


def list_cells(
    expr: LetExprList, prefix: tuple[int, ...]
) -> list[tuple[tuple[int, ...], LetExprNode]]:
    """The scalar cells of a nested list literal with their coordinates.

    Parameters
    ----------
    expr : LetExprList
        The literal.
    prefix : tuple[int, ...]
        The coordinates of the enclosing literals.

    Returns
    -------
    list[tuple[tuple[int, ...], LetExprNode]]
        One `(indices, item)` pair per scalar cell, row-major, indices
        counting from zero.
    """
    cells: list[tuple[tuple[int, ...], LetExprNode]] = []
    for position, item in enumerate(expr.items):
        if isinstance(item, LetExprList):
            cells.extend(list_cells(item, (*prefix, position)))
        else:
            cells.append(((*prefix, position), item))
    return cells


def list_sizes(expr: LetExprList) -> tuple[int, ...]:
    """The extent of each nesting level of a rectangular list literal.

    Parameters
    ----------
    expr : LetExprList
        The literal.

    Returns
    -------
    tuple[int, ...]
        The literal's extent, then its first item's, and so on.

    Raises
    ------
    UnsupportedConstruct
        If the rows of one level differ in extent.
    """
    sizes: list[int] = [len(expr.items)]
    inner = tuple(item for item in expr.items if isinstance(item, LetExprList))
    if not inner:
        return tuple(sizes)
    shapes = {list_sizes(item) for item in inner}
    if len(inner) != len(expr.items) or len(shapes) != 1:
        raise UnsupportedConstruct(
            "qvr-bugs", ["let-expr:list:ragged: a plated list literal is rectangular"]
        )
    return (*sizes, *shapes.pop())
