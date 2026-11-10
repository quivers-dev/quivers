"""The support tables of the plan.

A program's plan is derived from its checked kernel module by
[`quivers.transpile.plan`][quivers.transpile.plan]; this module holds what
the derivation reads beside the module: the shapes, cardinalities, and
declared bounds of the source's objects, the family metadata's sentinel
instances and the argument constraints and supports read off them, the
wire forms an argument takes to satisfy a constraint, and the alphabet
of a class-index draw. Renderers consume the plan and do not materialize
PyTorch tensors.

The tables are implementation of the plan derivation and contribute no
public names; [`Lower`][quivers.transpile.plan.Lower] is the entry point.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable
from typing import Literal

import didactic.api as dx
import torch
import torch.distributions.constraints as c
from torch.distributions.distribution import Distribution

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.core._util import EPS
from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    Expr,
    ExprIdent,
    LetStep,
    MarginalizeStep,
    Module,
    MorphismDecl,
    ObjectDecl,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    ReturnStep,
    SampleStep,
    ScoreStep,
    TypeFromExpr,
)
from quivers.dsl.ast_nodes.declarations import (
    ExportDecl,
    ScalarParam,
    TypeEnumSet,
)
from quivers.dsl.ast_nodes.let_expressions import (
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
)
from quivers.dsl.ast_nodes.objects import (
    ContinuousConstructor,
    DiscreteConstructor,
    ObjectExpr,
    ObjectProduct,
    TypeName,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.dsl.draw_args import (
    encode_index,
    is_matrix,
    list_items,
    matrix_rows,
)
from quivers.transpile.family_meta import (
    FAMILY_META,
    FamilyMeta,
)
from quivers.transpile.ir import (
    CSInterval,
    CSPositive,
    CSPositiveDefinite,
    CSReal,
    CSRealMatrix,
    CSRealVector,
    CSUnitInterval,
    Constraint,
    ConstraintSpec,
    DimStatic,
    IRArg,
    IRArgBroadcast,
    IRArgFamilyRef,
    IRArgKernel,
    IRArgList,
    IRArgMatrix,
    IRArgNumber,
    IRArgRef,
    LetExprAffineMap,
    Plate,
    event_shape_of,
)


def pick_program(module: Module) -> ProgramDecl:
    """Pick the `ProgramDecl` the module's `export` designates.

    When the module declares multiple programs, prefer one
    referenced by an `export` declaration; otherwise pick the
    last declared program.

    An `export` naming a `define` binding rather than a program
    is not a lowering target: a `define` is a morphism-level
    composition (`scan(cell) >> decoder`), and the transpile
    boundary is the probabilistic program. A `define` a step
    references is unfolded by
    [`expand_composite_lets`][quivers.dsl.composite_lets.expand_composite_lets]
    at the call site instead, so the recurrence a `scan` denotes
    reaches the IR through the program that samples it.
    """
    programs: list[ProgramDecl] = []
    exported_names: set[str] = set()
    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            programs.append(stmt)
        elif isinstance(stmt, ExportDecl) and isinstance(stmt.expr, ExprIdent):
            exported_names.add(stmt.expr.name)
    if not programs:
        raise UnsupportedConstruct(
            "qvr-lower",
            ["program:absent"],
        )
    return next(
        (p for p in programs if p.name in exported_names),
        programs[-1],
    )


def exported_return_names(module: Module) -> tuple[str, ...]:
    """The variable names the module's exported program returns.

    A QVR program declared `prog : A -> B` denotes a Markov kernel
    from `A` to `B`, and its `return` clause names the components of
    the value in `B` that the kernel carries. Those names are what a
    faithful emit has to expose through the target's own return
    surface, so they are the contract the export-equivalence tier
    checks a backend against. An empty tuple means the program
    declares no `return` clause and denotes only its joint.
    """
    return tuple(pick_program(module).return_vars)


def _family_meta_or_raise(family: str) -> FamilyMeta:
    """Return ``FAMILY_META[family]`` or raise
    [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
    with a precise kind. Replaces bare-dict-access ``KeyError`` so
    callers up the stack see the documented exception type instead
    of an opaque dict failure."""
    meta = FAMILY_META.get(family)
    if meta is None:
        raise UnsupportedConstruct(
            "qvr-lower",
            [f"family:{family}: not in FAMILY_META registry"],
        )
    return meta


# A parsed bracket-indexed argument string: `name[i0][i1]...`.
_BRACKET_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[[^\]]+\])+)$")
_BRACKET_INDICES_RE = re.compile(r"\[([^\]]+)\]")


# ---------------------------------------------------------------------------
# Declared-morphism parameter maps.
#
# A kernel morphism declared `morphism f : X -> Y ~ Family` with no
# family arguments of its own is a *conditional* family: the runtime
# builds `Conditional<Family>(X, Y)`, which owns a
# [`ParamSource`][quivers.continuous.param_source.ParamSource] from `X`
# to a row of `k * dim(Y)` numbers and reads the family's `k`
# per-coordinate arguments off that row. The declaration names neither
# the map nor its numbers, so a program emitted from the declaration
# alone scores the family at its defaults: a different measure, and on
# a different space whenever `dim(X) != dim(Y)`.
#
# Lowering closes that by putting the map *into* the IR: the map's
# weight and bias become two `IRDataInput` entries (the numbers are
# fixed model data, exactly as the runtime's compile-time draw makes
# them), and one `IRDeterministic` per family argument carries the row
# block the head reads and the head's transform. The site then scores
# against those arguments rather than against the family's defaults.
#
# The encoding is linear algebra, not arithmetic:
# [`LetExprAffineMap`][quivers.transpile.ir.LetExprAffineMap] names the
# weight, the bias, the row block, and the ordered sources whose
# concatenation is the conditioning row, and each renderer spells the
# contraction in its own language. One expression node per head,
# whatever the object's width, so a 16-wide state costs what a 2-wide
# state costs.
# ---------------------------------------------------------------------------


class _ParamHead(dx.Model):
    """One family argument read off a row of the parameter map.

    Head `k` of a family whose codomain is `d` wide reads columns
    ``k * d .. k * d + d`` of the map's output row. `transform` names
    what the runtime applies to the raw column before the family sees
    it:

    * ``identity``: the column is the argument
      (`ConditionalNormal`'s `loc`).
    * ``exp_floor``: the column is a log-parameter, exponentiated and
      floored at [`EPS`][quivers.core._util.EPS]
      (`ConditionalNormal`'s `scale`, whose runtime spelling is
      ``log_sigma.exp().clamp(min=EPS)``).
    """

    arg_name: str
    transform: Literal["identity", "exp_floor"]


#: Families whose conditional runtime class reads its arguments off a
#: parameter-map row, in the family's canonical argument order. A
#: family absent from this table keeps whatever arguments its
#: declaration and option block supply, which is what the emission has
#: always done; see the module note above for why the table is not
#: simply every conditional family.
_CONDITIONAL_HEADS: dict[str, tuple[_ParamHead, ...]] = {
    "Normal": (
        _ParamHead(arg_name="loc", transform="identity"),
        _ParamHead(arg_name="scale", transform="exp_floor"),
    ),
}


def _param_map_weight_name(morphism: str) -> str:
    """Wire name of a morphism's parameter-map weight."""
    return f"{morphism}_param_weight"


def _head_binding_name(site: str, head: _ParamHead) -> str:
    """Wire name of the deterministic binding carrying one head."""
    return f"{site}_{head.arg_name}"


def _head_raw_binding_name(site: str, head: _ParamHead) -> str:
    """Wire name of the pre-floor binding of an ``exp_floor`` head."""
    return f"{site}_{head.arg_name}_raw"


def _collect_step_names(steps: tuple[ProgramStep, ...], names: set[str]) -> None:
    """Add every name the steps bind, marginalize scopes included."""
    for step in steps:
        if isinstance(step, (SampleStep, ObserveStep, ReturnStep)):
            names.update(step.vars)
        elif isinstance(step, MarginalizeStep):
            names.add(step.var)
            _collect_step_names(step.scope, names)
        elif isinstance(step, (LetStep, ScoreStep)):
            names.add(step.name)


def _floor_expr(name: str, coordinate: int) -> LetExprNode:
    """``max(name[coordinate], EPS)`` as ``(a + e + |a - e|) / 2``.

    The runtime floors an exponentiated scale head at
    [`EPS`][quivers.core._util.EPS]. No target-portable two-argument
    `max` exists in the let-expression surface (`max` is the
    axis-reducing aggregator on the array-shaped backends), so the
    floor is spelled with the arithmetic identity, whose only call is
    `abs`.
    """
    value = LetExprIndex(
        array=LetExprVar(name=name),
        indices=(LetExprLiteral(value=float(coordinate)),),
    )
    floor = LetExprLiteral(value=EPS)
    return LetExprBinOp(
        op="/",
        left=LetExprBinOp(
            op="+",
            left=LetExprBinOp(op="+", left=value, right=floor),
            right=LetExprCall(
                func="abs",
                args=(LetExprBinOp(op="-", left=value, right=floor),),
            ),
        ),
        right=LetExprLiteral(value=2.0),
    )


_ARITHMETIC_OPERATORS = frozenset(("+", "-", "*", "/"))


def _arg_names_for(
    meta: FamilyMeta,
    args: tuple[IRArg, ...],
    ctx: _LowerCtx,
) -> tuple[str, ...]:
    """Return the parallel arg-name tuple for `args`.

    Names come from the distribution's positional constructor
    signature, because that is the contract a QVR call site writes
    against: ``~ Pareto(a, b)`` binds its arguments the way
    ``torch.distributions.Pareto`` binds them. The constructor can
    differ from ``arg_constraints``, which is keyed by the
    *constrained* parameters and so both reorders (``Pareto`` is
    keyed ``alpha, scale`` but constructed ``scale, alpha``) and
    omits leading parameters that carry no constraint
    (``RelaxedBernoulli``'s ``temperature``). Reading the wrong one
    transposes or drops a slot and silently changes the density.

    `_STRUCTURAL_ARG_NAMES` overrides the families whose leading
    constructor parameter is supplied structurally by the renderer
    rather than written at the call site. When the constructor
    cannot be introspected the constrained parameters are used; if
    those are a property (Wishart, Uniform), a sentinel supplies
    the instance-level dict. The returned tuple is positional: the
    i'th entry names the i'th user-supplied arg.
    """
    names = _STRUCTURAL_ARG_NAMES.get(meta.qvr_name, ())
    if not names:
        names = _ctor_param_names(meta.distribution_class)
    if not names:
        cls_attr = meta.distribution_class.arg_constraints
        if isinstance(cls_attr, dict):
            names = tuple(cls_attr.keys())
        else:
            sentinel = _make_sentinel(meta, args, ctx)
            names = tuple(sentinel.arg_constraints.keys())
    if len(args) > len(names):
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"family:{meta.qvr_name}:arity-mismatch: user "
                f"supplied {len(args)} args; "
                f"arg_constraints has {len(names)} positions "
                f"({list(names)})"
            ],
        )
    return names[: len(args)]


def _wrap_for_constraint(
    arg: IRArg,
    expected: Constraint | None,
    event_axes: tuple[str, ...],
    axes_index: ObjectExpr | None,
    ctx: _LowerCtx,
) -> IRArg:
    """When the user supplied a scalar but the constraint is
    `IndependentConstraint(base, n>=1)`, wrap as
    `IRArgBroadcast` whose `target_shape` is derived from the
    step's event axes (the `over=` clause when present, falling
    back to the step's `index`).

    Scalar literals (`IRArgNumber`) always qualify. An unindexed
    `IRArgRef` qualifies when it names a scalar binding (a
    `ScalarParam` of the active program); a renderer must then
    broadcast the scalar to the vector arg position rather than
    passing the scalar through as if it were already a tensor of
    the expected shape.
    """
    if expected is None:
        return arg
    if not isinstance(expected, c._IndependentConstraint):
        return arg
    if not isinstance(arg, (IRArgNumber, IRArgRef)):
        return arg
    if isinstance(arg, IRArgRef):
        if arg.indices:
            return arg
        if arg.name not in _scalar_binding_names(ctx):
            return arg
    target = _broadcast_target(expected, event_axes, axes_index, ctx)
    if target is None:
        return arg
    return IRArgBroadcast(value=arg, target_shape=target)


def _broadcast_target(
    expected: c._IndependentConstraint,
    event_axes: tuple[str, ...],
    axes_index: ObjectExpr | None,
    ctx: _LowerCtx,
) -> tuple[int, ...] | None:
    """Derive the broadcast `target_shape` from the step's event
    axes when the expected constraint is `IndependentConstraint`.

    Prefers the axis names supplied by `event_axes` (the `over=`
    clause of the surrounding step). Falls back to a single-axis
    shape derived from `axes_index` when no event axes are
    declared (the bare scalar-family form).
    """
    if event_axes:
        base: list[int] = []
        for axis_name in event_axes:
            size = ctx.cards.get(axis_name)
            if size is None:
                return None
            base.append(size)
        base_shape = tuple(base)
    elif axes_index is not None:
        size = axis_shape(axes_index, ctx.cards)
        if size is None:
            return None
        base_shape = (size,) * expected.event_dim
    else:
        return None
    return event_shape_of(expected, base_shape)


def _codomain_alphabet(morphism_name: str, ctx: _LowerCtx) -> DimStatic | None:
    """The alphabet the declared morphism's codomain names, as a
    [`DimStatic`][quivers.transpile.ir.DimStatic] carrying the
    class count and the codomain's own name, or `None` when the
    step names no declared morphism or the codomain is not a
    finite object.

    ``morphism lm_head : Hidden -> Token ~ Categorical`` says the
    per-row value of every draw through `lm_head` is a `Token`,
    so the draw's alphabet is `|Token|`. The plate the draw is
    replicated over (`observe next_token : Resp <- lm_head(h)`)
    says how many rows there are and nothing about how wide each
    row's alphabet is, so it is not consulted here.
    """
    decl = ctx.morphisms.get(morphism_name)
    if decl is None:
        return None
    shape = _object_expr_shape(decl.codomain, ctx.shapes, (morphism_name,))
    if shape is None or not shape.finite or shape.extent is None:
        return None
    return DimStatic(
        size=shape.extent,
        name=_axis_expr_name(decl.codomain),
    )


class RealBounds(dx.Model):
    """The ``{low=..., high=...}`` bounds a continuous object declares.

    ``object Rate : Real 3 {low=0.0, high=1.0}`` names the box
    ``[0, 1]^3`` rather than ``R^3``, so a variable whose event axis
    or per-row value space is `Rate` lives in that box. Renderers
    turn the pair into the target's own bounded declaration
    (Stan's ``<lower=0, upper=1>``, JAGS' ``T(0, 1)``, ...); a
    variant with neither bound set is the unbounded default.
    """

    low: float | None = None
    high: float | None = None

    @property
    def is_bounded(self) -> bool:
        """True when the declaration constrains at least one side."""
        return self.low is not None or self.high is not None


class ObjectShape(dx.Model):
    """The transpile-visible shape of one ``object`` declaration.

    Three independent readings, one per position an object name can
    occupy (see the module docstring):

    * `extent` is the size the object contributes as an *axis*: the
      cardinality of a finite object (`FinSet N`, an enum set, a
      product of finite factors) and the total coordinate count of a
      continuous one (`Real 28 28` is 784 coordinates wide).
    * `real_width` is set only for `Real`, the one constructor whose
      value is a plain real vector, and is what a program-domain
      factor or a morphism codomain reads to size its wire.
    * `bounds` carries the constructor's ``{low=..., high=...}``.

    `finite` separates the two readings of `extent`: a finite object
    has `|A|` elements, so its extent is an alphabet a class index
    can range over, while a continuous one has `extent` coordinates
    and names no alphabet at all.
    """

    extent: int | None = None
    real_width: int | None = None
    finite: bool = False
    bounds: RealBounds = dx.field(default_factory=RealBounds)


class _LowerCtx(ValueModel):
    """Internal carrier for the resolver / cardinality tables and the
    sentinel cache. Threaded through the lowering recursion.

    The opaque tables are mutable working state shared by reference
    along the recursion, so they compare by identity: two contexts are
    equal exactly when they share those table objects and agree on the
    typed fields."""

    morphisms: dict[str, MorphismDecl] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    lets: dict[str, Expr] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    cards: dict[str, int]
    family_set: frozenset[str]
    sentinel_cache: dict[tuple[str, tuple[str, ...]], Distribution] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    program: ProgramDecl = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    real_widths: dict[str, int] = dx.field(default_factory=dict)
    bounds: dict[str, RealBounds] = dx.field(default_factory=dict)
    shapes: dict[str, ObjectShape] = dx.field(default_factory=dict)
    bound_plates: dict[str, Plate] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    bound_kinds: dict[str, str] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    alphabet_event_dims: dict[str, DimStatic] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    param_map_inputs: dict[str, Plate] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    reserved_names: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Lower-internal helpers: object cardinalities, axis shape, free names.
# ---------------------------------------------------------------------------


def _observe_var(step: ObserveStep) -> str:
    """The single observed variable name of an observe step.

    The transpile models one observed variable per step; a
    multi-variable observe has no single-name IR form.
    """
    if len(step.vars) != 1:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"observe:multiple-vars:{step.vars}: the transpile "
                "emits one observed variable per step"
            ],
        )
    return step.vars[0]


def _size_arg_value(
    arg: str,
    table: dict[str, ObjectShape],
    names: tuple[str, ...],
    constructor: str,
) -> int:
    """Resolve one constructor size argument to an integer.

    An integer literal is itself; a bare name is the extent of a
    previously-declared object, matching the runtime's
    ``_eval_size_arg``. Anything else raises rather than silently
    dropping the declaration, because an object whose size the
    transpile cannot read would otherwise reach a renderer as a
    free ``N_<name>`` the target never declares.
    """
    if arg.isdigit():
        return int(arg)
    prior = table.get(arg)
    if prior is not None and prior.extent is not None:
        return prior.extent
    raise UnsupportedConstruct(
        "qvr-lower",
        [
            f"object:{'/'.join(names)}:{constructor}-size:{arg}: a "
            f"size argument must be an integer literal or the name "
            f"of an object declared earlier in the module with a "
            f"known extent"
        ],
    )


def _continuous_bounds(expr: ContinuousConstructor) -> RealBounds:
    """Read ``{low=..., high=...}`` off a continuous constructor."""
    low = expr.kwargs.get("low")
    high = expr.kwargs.get("high")
    if isinstance(low, str) or isinstance(high, str):
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"object:{expr.constructor}-bound:{low!r}/{high!r}: "
                f"`low=` and `high=` must be numeric literals"
            ],
        )
    return RealBounds(
        low=None if low is None else float(low),
        high=None if high is None else float(high),
    )


def _object_expr_shape(
    expr: ObjectExpr,
    table: dict[str, ObjectShape],
    names: tuple[str, ...],
) -> ObjectShape | None:
    """Return the [`ObjectShape`][quivers.transpile.lower.ObjectShape]
    a type expression denotes, or `None` when the expression names no
    statically-sized object (a `FreeResiduated` universe, a slash, an
    effect-apply)."""
    if isinstance(expr, TypeName):
        if expr.name.isdigit():
            return ObjectShape(extent=int(expr.name), finite=True)
        return table.get(expr.name)
    if isinstance(expr, DiscreteConstructor):
        if not expr.args:
            return None
        return ObjectShape(
            extent=_size_arg_value(expr.args[0], table, names, expr.constructor),
            finite=True,
        )
    if isinstance(expr, ContinuousConstructor):
        if not expr.args:
            return None
        sizes = [_size_arg_value(a, table, names, expr.constructor) for a in expr.args]
        bounds = _continuous_bounds(expr)
        if expr.constructor == "Real":
            width = math.prod(sizes)
            return ObjectShape(
                extent=width,
                real_width=width,
                bounds=bounds,
            )
        # Every other continuous constructor takes its leading
        # argument as the space's dimension; the flattened width of
        # a matrix-valued one (`Covariance n` is n * n reals) is not
        # a plain real vector, so no `real_width` is recorded.
        return ObjectShape(extent=sizes[0], bounds=bounds)
    if isinstance(expr, ObjectProduct):
        factors = [_object_expr_shape(f, table, names) for f in object_factors(expr)]
        if any(f is None or f.extent is None for f in factors):
            return None
        total = 1
        finite = True
        for f in factors:
            assert f is not None and f.extent is not None
            total *= f.extent
            finite = finite and f.finite
        return ObjectShape(extent=total, finite=finite)
    return None


def object_shapes(module: Module) -> dict[str, ObjectShape]:
    """Return name -> [`ObjectShape`][quivers.transpile.lower.ObjectShape]
    for every `object` declaration in `module`, in source order.

    Declarations are read in order so a later one can size itself
    from an earlier name (``object M : FinSet N5``,
    ``object Grid : Real Rows Cols``) exactly as the runtime's
    resolver does. A declaration whose value has no static size (a
    `FreeResiduated` universe, a free monoid, a residuated slash)
    contributes no entry, and every downstream caller treats an
    absent name the way it always has.
    """
    out: dict[str, ObjectShape] = {}
    for stmt in module.statements:
        if not isinstance(stmt, ObjectDecl):
            continue
        init = stmt.init
        shape: ObjectShape | None = None
        if isinstance(init, TypeEnumSet):
            shape = ObjectShape(
                extent=len(init.elements),
                finite=True,
            )
        elif isinstance(init, TypeFromExpr):
            shape = _object_expr_shape(init.expr, out, stmt.names)
        if shape is None:
            continue
        for name in stmt.names:
            out[name] = shape
    return out


def continuous_object_widths(module: Module) -> dict[str, int]:
    """Return name -> width for every ``Real ...`` object declaration.

    ``object State : Real 4`` names a value in R^4: a program with
    that object in its domain is applied to a 4-wide vector, and a
    morphism with it as codomain writes one. ``object Img : Real 28
    28`` names a value in R^784, the runtime's own reading of a
    multi-argument `Real`, so the width is the product of the
    arguments rather than the first of them. ``object Step : FinSet
    64`` names an index axis instead, so it is absent here even
    though
    [`object_cardinalities`][quivers.transpile.lower.object_cardinalities]
    records its 64. The other continuous constructors (``Simplex``,
    ``Sphere``, ``Covariance``, ...) carry a support a plain real
    vector does not describe, so they are absent as well and a caller
    that needs them has to read the constructor itself.
    """
    return {
        name: shape.real_width
        for name, shape in object_shapes(module).items()
        if shape.real_width is not None
    }


def object_bounds(module: Module) -> dict[str, RealBounds]:
    """Return name -> [`RealBounds`][quivers.transpile.lower.RealBounds]
    for every continuous object that declares ``{low=...}`` or
    ``{high=...}``. Objects with neither bound are absent."""
    return {
        name: shape.bounds
        for name, shape in object_shapes(module).items()
        if shape.bounds.is_bounded
    }


def object_factors(expr: ObjectExpr) -> tuple[ObjectExpr, ...]:
    """Flatten a product object expression into its factors, in
    source order. A non-product expression is its own single
    factor."""
    if isinstance(expr, ObjectProduct):
        out: list[ObjectExpr] = []
        for comp in expr.components:
            out.extend(object_factors(comp))
        return tuple(out)
    return (expr,)


def _spec_real_range(
    spec: ConstraintSpec,
) -> tuple[float, float] | None:
    """The closed real range a `ConstraintSpec` denotes, or `None`
    when the spec is not a real-valued support at all (an integer
    outcome, a simplex, a Cholesky factor, ...)."""
    if isinstance(spec, (CSReal, CSRealVector, CSRealMatrix)):
        return (float("-inf"), float("inf"))
    if isinstance(spec, CSPositive):
        return (0.0, float("inf"))
    if isinstance(spec, CSUnitInterval):
        return (0.0, 1.0)
    if isinstance(spec, CSInterval):
        return (spec.lower, spec.upper)
    return None


def _range_spec(
    low: float,
    high: float,
    name: str,
    source: str,
) -> ConstraintSpec:
    """The `ConstraintSpec` for a closed real range.

    A one-sided bound other than ``> 0`` has no IR form, so it
    raises rather than reaching a renderer as an unbounded real: the
    declared support would then be wider in the target than in the
    source program.
    """
    lo_open = low == float("-inf")
    hi_open = high == float("inf")
    if lo_open and hi_open:
        return CSReal()
    if hi_open and low == 0.0:
        return CSPositive(strict=True)
    if lo_open or hi_open:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"object-bounds:one-sided:{source}:{name}: the IR "
                f"carries a real support that is unbounded, "
                f"positive, or a closed interval; declare both "
                f"`low=` and `high=` so the bound reaches the target"
            ],
        )
    if low == 0.0 and high == 1.0:
        return CSUnitInterval()
    return CSInterval(lower=low, upper=high)


def _bounds_of_axes(
    axis_names: tuple[str, ...],
    ctx: _LowerCtx,
    name: str,
) -> RealBounds | None:
    """The declared bounds the axes of one binding agree on.

    An unbounded axis contributes nothing. Two axes declaring
    different boxes describe no single support, so they raise rather
    than one silently winning.
    """
    found: list[tuple[str, RealBounds]] = []
    for axis in axis_names:
        bounds = ctx.bounds.get(axis)
        if bounds is None:
            continue
        if any(prior == bounds for _, prior in found):
            continue
        found.append((axis, bounds))
    if not found:
        return None
    if len(found) > 1:
        clashing = ", ".join(axis for axis, _ in found)
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"object-bounds:conflicting-axes:{name}:{clashing}: "
                f"the axes of one binding declare different "
                f"`{{low=..., high=...}}` boxes, so the binding has "
                f"no single support"
            ],
        )
    return found[0][1]


def _apply_declared_bounds(
    spec: ConstraintSpec,
    plate: Plate,
    ctx: _LowerCtx,
    name: str,
) -> ConstraintSpec:
    """Narrow a family's support by the ``{low=..., high=...}`` box
    the binding's own axis objects declare.

    ``sample z : Batch <- Normal(0, 1) [over=Rate]`` with
    ``object Rate : Real 3 {low=0.0, high=1.0}`` draws a value of
    `Rate`, so the value lives in ``[0, 1]^3``: the family's real
    support intersected with the declared box. A box on a support
    that is not real-valued at all contradicts the family rather
    than narrowing it, and an empty intersection describes no value,
    so both raise.
    """
    axis_names = tuple(dim.name for dim in (*plate.event_dims, *plate.batch_dims))
    bounds = _bounds_of_axes(axis_names, ctx, name)
    if bounds is None:
        return spec
    current = _spec_real_range(spec)
    if current is None:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"object-bounds:non-real-support:{name}:{spec.kind}: "
                f"a `{{low=..., high=...}}` box narrows a real "
                f"support, and this binding's support is not real"
            ],
        )
    low = max(
        current[0],
        bounds.low if bounds.low is not None else float("-inf"),
    )
    high = min(
        current[1],
        bounds.high if bounds.high is not None else float("inf"),
    )
    if low >= high:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"object-bounds:empty-support:{name}:[{low}, {high}]: "
                f"the declared box and the family's own support do "
                f"not overlap"
            ],
        )
    return _range_spec(low, high, name, "axis")


def _assert_alphabet_width(
    stated: int,
    dim: DimStatic,
    name: str,
    source: str,
) -> None:
    """Reject an alphabet argument whose own width contradicts the
    width the declared codomain names.

    The two would score the draw against different alphabets, and no
    target can hold both, so the disagreement is reported at its
    source rather than resolved in favour of either side.
    """
    if stated == dim.size:
        return
    raise UnsupportedConstruct(
        "qvr-lower",
        [
            f"class-index:{name}:alphabet-width:{source}: the "
            f"argument states {stated} classes and the declared "
            f"codomain {dim.name!r} names {dim.size}; the draw has "
            f"one alphabet, so make the two agree"
        ],
    )


def _retarget_alphabet_arg(
    arg: IRArg,
    dim: DimStatic,
    name: str,
    ctx: _LowerCtx,
) -> IRArg:
    """Widen a class-index family's alphabet argument to `dim`.

    Anything that carries its own event shape (an indexed reference,
    a literal vector, a matrix row, a reference to a binding that
    already has event dims) states its width itself and passes
    through. A broadcast already built from the step's axes is
    re-targeted, its earlier shape having been read off the plate
    rather than the codomain.

    A bare reference splits by what binds it. A `sample` states its
    own shape in the source (`<- Family(...) [over=...]`), so a
    scalar one is broadcast to the alphabet width at the call site
    and a plated one raises: a per-row scalar draw is not a per-row
    probability vector, and widening it would silently restate the
    step. A `let` or a free data input states its width through its
    type in the checked module, which the plan reads it from.
    """
    if isinstance(arg, IRArgBroadcast):
        if arg.target_shape == (dim.size,):
            return arg
        return IRArgBroadcast(value=arg.value, target_shape=(dim.size,))
    if isinstance(arg, IRArgList):
        _assert_alphabet_width(len(arg.elements), dim, name, "list")
        return arg
    if isinstance(arg, IRArgMatrix):
        row = arg.rows[0] if arg.rows else None
        if row is not None:
            _assert_alphabet_width(
                len(row.elements),
                dim,
                name,
                "matrix-row",
            )
        return arg
    if not isinstance(arg, IRArgRef) or arg.indices:
        return arg
    bound = ctx.bound_plates.get(arg.name)
    if bound is not None and bound.event_dims:
        trailing = bound.event_dims[-1]
        if isinstance(trailing, DimStatic):
            _assert_alphabet_width(
                trailing.size,
                dim,
                name,
                f"binding:{arg.name}",
            )
        return arg
    if ctx.bound_kinds.get(arg.name) in ("sample", "marginalize"):
        if bound is not None and bound.batch_dims:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"class-index:{name}:alphabet-arg:{arg.name}: the "
                    f"alphabet argument is a plated scalar draw, so "
                    f"it carries one number per row where the family "
                    f"needs one probability per class; draw it "
                    f"`[over=...]` the codomain instead"
                ],
            )
        return IRArgBroadcast(value=arg, target_shape=(dim.size,))
    prior = ctx.alphabet_event_dims.get(arg.name)
    if prior is not None and prior.size != dim.size:
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"class-index:{name}:alphabet-width:binding:"
                f"{arg.name}: the same binding feeds alphabets of "
                f"{prior.size} and {dim.size} classes; one binding "
                f"carries one width, so split it in two"
            ],
        )
    ctx.alphabet_event_dims[arg.name] = dim
    return arg


def _axis_expr_name(expr: ObjectExpr) -> str:
    """A wire-safe identifier for an axis object expression.

    Names the plate a renderer emits a loop variable for, so it has
    to survive into every target's identifier syntax: a named object
    keeps its name, a product joins its factors with an underscore,
    and an anonymous `FinSet N` is `anon`.
    """
    if isinstance(expr, TypeName):
        return expr.name
    if isinstance(expr, ObjectProduct):
        return "_".join(_axis_expr_name(f) for f in object_factors(expr))
    return "anon"


def object_cardinalities(module: Module) -> dict[str, int]:
    """Return name -> axis extent for every statically-sized object
    declaration.

    The extent is what the object contributes in an *axis* position:
    the cardinality of a finite object (`FinSet N`, an enum set, a
    product of finite factors) and the coordinate count of a
    continuous one (`Real 28 28` is 784 wide). An object whose value
    has no static size contributes no entry.
    """
    return {
        name: shape.extent
        for name, shape in object_shapes(module).items()
        if shape.extent is not None
    }


def axis_shape(expr: ObjectExpr, cards: dict[str, int]) -> int | None:
    """Return the cardinality of an axis object expression.

    A product axis (`A * B`) is the flattened cardinality of its
    factors, matching the runtime's `ProductSet`; a factor whose
    size `cards` does not record makes the whole product unsized.
    """
    if isinstance(expr, TypeName):
        if expr.name.isdigit():
            return int(expr.name)
        return cards.get(expr.name)
    if isinstance(expr, DiscreteConstructor) and expr.args:
        try:
            return int(expr.args[0])
        except ValueError:
            return cards.get(expr.args[0])
    if isinstance(expr, ObjectProduct):
        total = 1
        for factor in object_factors(expr):
            size = axis_shape(factor, cards)
            if size is None:
                return None
            total *= size
        return total
    return None


def build_shape_table(
    program: ProgramDecl, cards: dict[str, int]
) -> dict[str, tuple[int, ...]]:
    """Return name -> shape for every sample / observe / let binding
    in `program`."""
    out: dict[str, tuple[int, ...]] = {}
    for step in program.draws:
        if isinstance(step, SampleStep):
            shape = _step_shape(step.index, cards)
            for v in step.vars:
                out[v] = shape
        elif isinstance(step, ObserveStep):
            out[_observe_var(step)] = _step_shape(step.index, cards)
        elif isinstance(step, LetStep):
            out[step.name] = ()
        elif isinstance(step, MarginalizeStep):
            out[step.var] = _step_shape(step.index, cards)
            inner = build_shape_table(
                ProgramDecl(
                    name=program.name,
                    domain=program.domain,
                    codomain=program.codomain,
                    draws=step.scope,
                ),
                cards,
            )
            out.update(inner)
    return out


def _step_shape(index: ObjectExpr | None, cards: dict[str, int]) -> tuple[int, ...]:
    if index is None:
        return ()
    n = axis_shape(index, cards)
    return (n,) if n is not None else ()


def exogenous_data_inputs(program: ProgramDecl, bound: set[str]) -> list[str]:
    """Return the ordered list of exogenous identifier names
    referenced in `program` but not bound by any step.

    Sources: free names in let / score expressions; bracket-indexed
    arg references (`mu[cls]` -> `cls`, `mu`); `via=` fibrations.
    Order is deterministic by traversal order.
    """
    out: list[str] = []
    seen: set[str] = set()

    def consider(name: str) -> None:
        if name in bound or name in seen:
            return
        seen.add(name)
        out.append(name)

    for step in program.draws:
        for n in _names_in_step(step):
            consider(n)
    return out


def _names_in_step(step: ProgramStep) -> list[str]:
    out: list[str] = []
    if isinstance(step, SampleStep):
        for a in step.args or ():
            out.extend(_names_in_raw_arg(a))
    elif isinstance(step, ObserveStep):
        for a in step.args or ():
            out.extend(_names_in_raw_arg(a))
        if step.via is not None:
            out.append(step.via)
    elif isinstance(step, MarginalizeStep):
        for a in step.args or ():
            out.extend(_names_in_raw_arg(a))
        for inner in step.scope:
            out.extend(_names_in_step(inner))
    elif isinstance(step, LetStep):
        out.extend(free_vars_in_let(step.value))
    elif isinstance(step, ScoreStep):
        out.extend(free_vars_in_let(step.value))
    return out


def _names_in_raw_arg(arg: DrawArg | str | float) -> list[str]:
    if isinstance(arg, DrawArgNamed):
        return _names_in_raw_arg(arg.value)
    if isinstance(arg, DrawArgScalar):
        return []
    if isinstance(arg, DrawArgName):
        return _names_in_atom_text(arg.text)
    if isinstance(arg, DrawArgIndex):
        return _names_in_atom_text(encode_index(arg))
    if isinstance(arg, DrawArgDist):
        dist_names: list[str] = []
        for a in arg.args:
            dist_names.extend(_names_in_raw_arg(a))
        return dist_names
    if isinstance(arg, DrawArgList):
        list_names: list[str] = []
        for item in list_items(arg):
            list_names.extend(_names_in_raw_arg(item))
        return list_names
    if not isinstance(arg, str):
        return []
    return _names_in_atom_text(arg)


def _names_in_atom_text(text: str) -> list[str]:
    """Walk a wire-form atom text and return the names it references."""
    m = _BRACKET_RE.match(text)
    if m is None:
        if _is_number_text(text):
            return []
        return [text]
    out: list[str] = [m.group(1)]
    for idx in _BRACKET_INDICES_RE.findall(m.group(2)):
        out.extend(_names_in_atom_text(idx))
    return out


def free_vars_in_let(expr: LetExprNode) -> list[str]:
    """Return the ordered list of free variable names in a let
    expression tree."""
    out: list[str] = []
    seen: set[str] = set()

    def visit(n: LetExprNode, bound: frozenset[str]) -> None:
        if isinstance(n, LetExprVar):
            if n.name in bound or n.name in seen:
                return
            seen.add(n.name)
            out.append(n.name)
            return
        if isinstance(n, LetExprLiteral):
            return
        if isinstance(n, LetExprString):
            return
        if isinstance(n, LetExprBinOp):
            visit(n.left, bound)
            visit(n.right, bound)
            return
        if isinstance(n, LetExprUnaryOp):
            visit(n.operand, bound)
            return
        if isinstance(n, LetExprCall):
            for a in n.args:
                visit(a, bound)
            return
        if isinstance(n, LetExprIndex):
            visit(n.array, bound)
            for i in n.indices:
                visit(i, bound)
            return
        if isinstance(n, LetExprList):
            for it in n.items:
                visit(it, bound)
            return
        if isinstance(n, LetExprLambda):
            visit(n.body, bound | frozenset({n.param}))
            return
        if isinstance(n, LetExprFactor):
            inner_bound = bound | frozenset(b.var for b in n.binders)
            if n.body is not None:
                visit(n.body, inner_bound)
            for case in n.cases:
                visit(case.value, inner_bound)
            return
        if isinstance(n, LetExprMethodCall):
            visit(n.receiver, bound)
            for a in n.args:
                visit(a, bound)
            return
        if isinstance(n, LetExprAffineMap):
            visit(n.weight, bound)
            for source in n.sources:
                visit(source.value, bound)
            visit(n.bias, bound)
            return

    visit(expr, frozenset())
    return out


def free_names_in_arg(arg: IRArg) -> list[str]:
    """Return the ordered list of free names in an IR arg tree."""
    out: list[str] = []
    seen: set[str] = set()

    def visit(a: IRArg) -> None:
        if isinstance(a, IRArgRef):
            if a.name not in seen:
                seen.add(a.name)
                out.append(a.name)
            for i in a.indices:
                visit(i)
            return
        if isinstance(a, IRArgBroadcast):
            visit(a.value)
            return
        if isinstance(a, IRArgList):
            for e in a.elements:
                visit(e)
            return
        if isinstance(a, IRArgMatrix):
            for row in a.rows:
                visit(row)
            return
        if isinstance(a, IRArgFamilyRef):
            if a.name not in seen:
                seen.add(a.name)
                out.append(a.name)
            return
        if isinstance(a, IRArgKernel):
            if a.x_name not in seen:
                seen.add(a.x_name)
                out.append(a.x_name)
            return
        if isinstance(a, IRArgNumber):
            return

    visit(arg)
    return out


_DATA_CONSTRAINT_FACTORY: dict[str, Callable[[], ConstraintSpec]] = {
    "real_matrix": CSRealMatrix,
    "real_vector": CSRealVector,
    "positive_definite": CSPositiveDefinite,
}

_SAMPLE_CONSTRAINT_FACTORY: dict[str, Callable[[], ConstraintSpec]] = {
    "real_matrix": CSRealMatrix,
    "real_vector": CSRealVector,
}


def _anon_codomain_size(codomain: ObjectExpr) -> tuple[int, str] | None:
    """Return ``(size, base_name)`` for an anonymous-cardinality
    codomain (`Real N`), or `None` when the codomain has no usable
    anonymous size.
    """
    if (
        isinstance(codomain, DiscreteConstructor)
        and codomain.args
        and isinstance(codomain.args[0], int)
    ):
        return int(codomain.args[0]), "event"
    if isinstance(codomain, TypeName):
        return None
    if isinstance(codomain, ObjectProduct):
        for comp in codomain.components:
            inner = _anon_codomain_size(comp)
            if inner is not None:
                return inner
    return None


def arg_ref_shape(
    arg: IRArg, shape_table: dict[str, tuple[int, ...]]
) -> tuple[int, ...]:
    """Return the broadcast-evaluated shape of an IR arg referenced
    through `shape_table`."""
    if isinstance(arg, IRArgRef):
        base = shape_table.get(arg.name, ())
        # Indexing peels off one dim per index expression.
        peeled = max(0, len(base) - len(arg.indices))
        return base[:peeled]
    if isinstance(arg, IRArgList):
        return (len(arg.elements),)
    if isinstance(arg, IRArgMatrix):
        return (len(arg.rows), len(arg.rows[0].elements) if arg.rows else 0)
    if isinstance(arg, IRArgBroadcast):
        return arg.target_shape
    return ()


def lower_factors(
    program: ProgramDecl, cards: dict[str, int]
) -> dict[str, tuple[int, ...]]:
    """Return name -> factor shape for every plated binding in
    `program`. Mirror of [`build_shape_table`][quivers.transpile.lower.build_shape_table]
    intended for renderers that need the factor shape (event_dims
    plus batch_dims) of every name."""
    return build_shape_table(program, cards)


def inline_list_lets(
    program: ProgramDecl,
) -> dict[str, tuple[str, ...]]:
    """Return name -> tuple of element names for every let-step
    whose RHS is a [`LetExprList`][quivers.dsl.ast_nodes.let_expressions.LetExprList]
    of bare-variable references."""
    out: dict[str, tuple[str, ...]] = {}
    for step in program.draws:
        if not isinstance(step, LetStep):
            continue
        if isinstance(step.value, LetExprList):
            names: list[str] = []
            ok = True
            for it in step.value.items:
                if not isinstance(it, LetExprVar):
                    ok = False
                    break
                names.append(it.name)
            if ok:
                out[step.name] = tuple(names)
    return out


# ---------------------------------------------------------------------------
# Sentinel parameter construction and `arg_constraints` resolution.
# ---------------------------------------------------------------------------


#: Families whose leading constructor parameter is supplied by the
#: renderer from the sample's event axis rather than written at the QVR
#: call site, so it must not consume a user-supplied argument slot.
#: ``LKJCholesky(dim, concentration)`` takes its matrix dimension from
#: the codomain axis; the call site writes only the concentration.
_STRUCTURAL_ARG_NAMES: dict[str, tuple[str, ...]] = {
    "LKJCholesky": ("concentration",),
}


def _ctor_param_names(cls: type) -> tuple[str, ...]:
    """Return the positional parameter names of ``cls.__init__``.

    Drops ``self``, ``validate_args``, and any variadic parameter, so
    the result is the positional contract a call site binds against.
    Returns an empty tuple when the signature cannot be read (a C
    extension type or a shim without an introspectable ``__init__``),
    which the caller treats as "fall back to the constrained
    parameters".
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except TypeError, ValueError:
        return ()
    return tuple(
        name
        for name, param in params.items()
        if name not in ("self", "validate_args")
        and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    )


def _make_sentinel(
    meta: FamilyMeta,
    args: tuple[IRArg, ...],
    ctx: _LowerCtx,
    arg_names: tuple[str, ...] = (),
) -> Distribution:
    """Build a sentinel `Distribution` instance for `meta` with
    placeholder tensors derived from `args`.

    The instance carries the right `event_shape`, `batch_shape`,
    `support`, and (when `arg_constraints` is a property) the right
    per-arg constraints. Cached by `(family_name, IR-arg-tuple-key)`
    so repeated call sites pay one instantiation cost.

    Reference args are dimensioned using the class-level
    `arg_constraints` (when available) so distributions like
    `Categorical(probs=probs)` get a vector placeholder rather than a
    scalar.
    """
    key = (
        meta.qvr_name,
        tuple(
            f"{name}={_arg_key(arg)}" for name, arg in zip(arg_names, args, strict=True)
        )
        if arg_names
        else tuple(_arg_key(a) for a in args),
    )
    if key in ctx.sentinel_cache:
        return ctx.sentinel_cache[key]
    expected_shapes = _expected_arg_shapes(meta, len(args), arg_names)
    sentinel_args = tuple(
        _arg_to_tensor(a, ctx, expected_shapes[i]) for i, a in enumerate(args)
    )
    try:
        if arg_names:
            kwargs = dict(zip(arg_names, sentinel_args, strict=True))
            instance = meta.distribution_class(**kwargs)
        else:
            instance = meta.distribution_class(*sentinel_args)
    except TypeError as exc:
        # The family requires args the user did not supply (e.g.
        # `~ Wishart` with no call-site arguments). Fall through to
        # the signature-inspection path: build a sentinel from every
        # required constructor parameter, sized by the parameter's
        # `arg_constraints` (or the class's
        # [`Distribution.support`][torch.distributions.constraints]
        # when `arg_constraints` is a property at the class level).
        instance = _construct_sentinel_from_signature(meta, sentinel_args)
        if instance is None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"family:{meta.qvr_name}:sentinel-failed: cannot "
                    f"instantiate {meta.distribution_class.__name__} "
                    f"with placeholder args derived from "
                    f"{args!r}: {exc!r}"
                ],
            ) from exc
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"family:{meta.qvr_name}:sentinel-failed: cannot "
                f"instantiate {meta.distribution_class.__name__} "
                f"with placeholder args derived from "
                f"{args!r}: {exc!r}"
            ],
        ) from exc
    ctx.sentinel_cache[key] = instance
    return instance


def _construct_sentinel_from_signature(
    meta: FamilyMeta,
    user_args: tuple[torch.Tensor | torch.distributions.Distribution, ...],
) -> torch.distributions.Distribution | None:
    """Construct a placeholder
    [`Distribution`][torch.distributions.Distribution] instance for
    `meta.distribution_class` by inspecting its constructor signature
    and supplying a sensible default for every required parameter not
    covered by `user_args`.

    Used as a fallback when
    `_lookup_or_build_sentinel`'s
    primary path (calling the constructor with `user_args` directly)
    raises `TypeError` because the user didn't supply enough
    arguments. The sentinel is only used to derive shape /
    event-dimension information downstream; its numerical values
    are arbitrary.

    Returns `None` if the signature inspection cannot construct a
    valid instance (e.g. the class has an `__init__` that requires
    backend-specific kwargs the inspector doesn't know about); the
    caller falls back to `UnsupportedConstruct`.

    Heuristics per parameter:

    * `df` / `degree_of_freedom` / `nu`: a scalar float at least
      `event_dim + 1` so any positive-definite scale-matrix
      constraint is satisfied.
    * `covariance_matrix` / `scale_matrix` / `precision_matrix` /
      `scale_tril`: a `(dim, dim)` identity matrix where `dim` is
      derived from a matrix-valued `user_args[0]` shape when
      present, or 2 by default.
    * other named parameters: skipped (the constructor's own
      defaults are used).
    """
    sig = inspect.signature(meta.distribution_class.__init__)
    bound_positional = list(user_args)
    kwargs: dict[str, torch.Tensor] = {}
    dim = _infer_sentinel_dim(user_args)
    sigparams = list(sig.parameters.values())[1:]  # drop `self`
    # Group A: every parameter the user did not supply positionally AND
    # that has no default. Pick a sensible sentinel based on the name.
    for i, param in enumerate(sigparams):
        if i < len(bound_positional):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        sentinel = _sentinel_value_for_param(param.name, dim)
        if sentinel is None:
            return None
        kwargs[param.name] = sentinel
    # Group B: parameter families where the constructor enforces
    # "exactly one of {a, b, c} must be non-None" at runtime (Wishart,
    # MultivariateNormal, InverseWishart on covariance/precision/scale).
    # Supply the canonical Cholesky-factor variant when none of the
    # trio is already bound.
    trio_group = (
        "covariance_matrix",
        "precision_matrix",
        "scale_tril",
    )
    param_names = {p.name for p in sigparams}
    bound_names = {p.name for p in sigparams[: len(bound_positional)]} | set(kwargs)
    present_in_sig = [n for n in trio_group if n in param_names]
    if present_in_sig and not any(n in bound_names for n in present_in_sig):
        kwargs[present_in_sig[-1]] = torch.eye(dim)
    try:
        return meta.distribution_class(*bound_positional, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def _sentinel_value_for_param(
    name: str,
    dim: int,
) -> torch.Tensor | None:
    """Pick a sensible sentinel default for a constructor parameter
    by name. Returns `None` for parameters whose name is not
    recognised; the caller falls back to UnsupportedConstruct.
    """
    if name in ("df", "degree_of_freedom", "nu", "concentration"):
        return torch.tensor(float(dim + 1))
    if name in (
        "covariance_matrix",
        "scale_matrix",
        "precision_matrix",
        "scale_tril",
    ):
        return torch.eye(dim)
    if name in ("loc", "mean"):
        return torch.zeros(dim)
    if name in ("scale", "sigma"):
        return torch.tensor(1.0)
    if name in ("rate", "lambda"):
        return torch.tensor(1.0)
    if name in ("low", "lower"):
        return torch.tensor(0.0)
    if name in ("high", "upper"):
        return torch.tensor(1.0)
    if name in ("probs", "p"):
        return torch.full((dim,), 1.0 / max(dim, 1))
    if name in ("total_count",):
        return torch.tensor(1)
    if name == "validate_args":
        return torch.tensor(False)  # not really used; signature ask
    return None


def _infer_sentinel_dim(
    user_args: tuple[torch.Tensor | torch.distributions.Distribution, ...],
) -> int:
    """Best-effort guess at the event dimension for a sentinel
    matrix-valued parameter. Inspects the first matrix-shaped tensor
    in `user_args`; defaults to 2 if nothing useful surfaces."""
    for arg in user_args:
        if isinstance(arg, torch.Tensor) and arg.dim() >= 2:
            return int(arg.shape[-1])
    return 2


def _expected_arg_shapes(
    meta: FamilyMeta,
    n_args: int,
    arg_names: tuple[str, ...] = (),
) -> tuple[tuple[int, ...], ...]:
    """Per-arg expected shape derived from class-level
    `arg_constraints`. Used to size placeholder tensors for the
    sentinel."""
    cls_attr = meta.distribution_class.arg_constraints
    if not isinstance(cls_attr, dict):
        return tuple(() for _ in range(n_args))
    if arg_names:
        return tuple(
            _constraint_default_shape(cls_attr[name]) if name in cls_attr else ()
            for name in arg_names
        )
    out: list[tuple[int, ...]] = []
    for _, constraint in list(cls_attr.items())[:n_args]:
        out.append(_constraint_default_shape(constraint))
    while len(out) < n_args:
        out.append(())
    return tuple(out)


def _constraint_default_shape(constraint: Constraint) -> tuple[int, ...]:
    """Pick a small default shape for a parameter whose constraint
    requires nonzero event_dim."""
    if isinstance(constraint, c._IndependentConstraint):
        return (2,) * constraint.event_dim
    if isinstance(constraint, c._Simplex):
        return (2,)
    if isinstance(constraint, c._PositiveDefinite):
        return (2, 2)
    if isinstance(constraint, c._PositiveSemidefinite):
        return (2, 2)
    if isinstance(constraint, c._CorrCholesky):
        return (2, 2)
    if isinstance(constraint, c._LowerCholesky):
        return (2, 2)
    if isinstance(constraint, c._OneHot):
        return (2,)
    return ()


def _arg_to_tensor(
    arg: IRArg, ctx: _LowerCtx, expected_shape: tuple[int, ...]
) -> torch.Tensor | torch.distributions.Distribution:
    """Materialise a placeholder argument for one IR arg.

    Numeric / ref / list / matrix args produce a `torch.Tensor` of
    `expected_shape`. `IRArgFamilyRef` produces a fully-instantiated
    `torch.distributions.Distribution` so wrapper families that
    accept a base distribution (e.g. `Truncated(base, ...)`) get a
    valid first positional.
    """
    if isinstance(arg, IRArgNumber):
        if expected_shape:
            return torch.full(expected_shape, arg.value, dtype=torch.float32)
        return torch.tensor(arg.value, dtype=torch.float32)
    if isinstance(arg, IRArgRef):
        if expected_shape:
            # For simplex-shaped expectations pad with a uniform mass
            # so the constraint check succeeds.
            return _shape_default_tensor(expected_shape)
        return torch.zeros((), dtype=torch.float32)
    if isinstance(arg, IRArgBroadcast):
        return torch.zeros(arg.target_shape, dtype=torch.float32)
    if isinstance(arg, IRArgList):
        return torch.tensor(
            [e.value if isinstance(e, IRArgNumber) else 0.0 for e in arg.elements],
            dtype=torch.float32,
        )
    if isinstance(arg, IRArgMatrix):
        return torch.tensor(
            [
                [e.value if isinstance(e, IRArgNumber) else 0.0 for e in row.elements]
                for row in arg.rows
            ],
            dtype=torch.float32,
        )
    if isinstance(arg, IRArgFamilyRef):
        decl = ctx.morphisms.get(arg.name)
        if decl is None or decl.init_family is None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"arg:family-ref:{arg.name}: morphism not "
                    f"declared with `~ Family(...)` init"
                ],
            )
        inner_meta = _family_meta_or_raise(decl.init_family.family)
        inner_args = tuple(_raw_to_ir_for_sentinel(a) for a in decl.init_family.args)
        return _make_sentinel(inner_meta, inner_args, ctx)
    raise UnsupportedConstruct("qvr-lower", [f"arg:unknown:{type(arg).__name__}"])


def _shape_default_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    """A placeholder tensor of `shape` valid as a simplex / PD / etc.

    Uniform 1/n entries satisfy Simplex; an identity matrix satisfies
    PositiveDefinite / LowerCholesky.
    """
    if len(shape) == 1:
        n = shape[0]
        return torch.full(shape, 1.0 / n, dtype=torch.float32)
    if len(shape) == 2 and shape[0] == shape[1]:
        return torch.eye(shape[0], dtype=torch.float32)
    return torch.zeros(shape, dtype=torch.float32)


def _raw_to_ir_for_sentinel(raw: DrawArg | str | float) -> IRArg:
    """Cheap arg-to-IR conversion used only for inner sentinel
    construction (no morphism table required)."""
    if isinstance(raw, DrawArgNamed):
        return _raw_to_ir_for_sentinel(raw.value)
    if isinstance(raw, DrawArgScalar):
        return IRArgNumber(value=raw.value)
    if isinstance(raw, DrawArgName):
        return _atom_text_for_sentinel(raw.text)
    if isinstance(raw, DrawArgIndex):
        return _atom_text_for_sentinel(encode_index(raw))
    if isinstance(raw, DrawArgDist):
        raise UnsupportedConstruct(
            "qvr-lower",
            [
                f"nested-distribution-arg:{raw.family}: a "
                "distribution-valued draw argument is not "
                "representable in this backend's IR"
            ],
        )
    if isinstance(raw, DrawArgList):
        if is_matrix(raw):
            return IRArgMatrix(
                rows=tuple(
                    IRArgList(elements=tuple(_raw_to_ir_for_sentinel(e) for e in row))
                    for row in matrix_rows(raw)
                )
            )
        return IRArgList(
            elements=tuple(_raw_to_ir_for_sentinel(e) for e in list_items(raw))
        )
    if isinstance(raw, (int, float)):
        return IRArgNumber(value=float(raw))
    return _atom_text_for_sentinel(raw)


def _atom_text_for_sentinel(text: str) -> IRArg:
    if _is_number_text(text):
        return IRArgNumber(value=float(text))
    return IRArgRef(name=text, indices=())


def _arg_key(a: IRArg) -> str:
    """Stable string key for caching purposes."""
    if isinstance(a, IRArgNumber):
        return f"n:{a.value}"
    if isinstance(a, IRArgRef):
        return f"r:{a.name}:{len(a.indices)}"
    if isinstance(a, IRArgBroadcast):
        return f"b:{a.target_shape}:{_arg_key(a.value)}"
    if isinstance(a, IRArgList):
        return "l:" + ",".join(_arg_key(e) for e in a.elements)
    if isinstance(a, IRArgMatrix):
        return "m:" + ";".join(
            ",".join(_arg_key(e) for e in row.elements) for row in a.rows
        )
    if isinstance(a, IRArgFamilyRef):
        return f"f:{a.name}"
    return type(a).__name__


def _resolve_arg_constraints(
    meta: FamilyMeta,
    args: tuple[IRArg, ...],
    ctx: _LowerCtx,
) -> dict[str, Constraint]:
    """Return the `arg_constraints` dict for `meta`, evaluating the
    sentinel instance when `arg_constraints` is a property rather
    than a class-level dict."""
    cls_attr = meta.distribution_class.arg_constraints
    if isinstance(cls_attr, dict):
        return cls_attr
    instance = _make_sentinel(meta, args, ctx)
    return dict(instance.arg_constraints)


def _resolve_support(
    meta: FamilyMeta,
    args: tuple[IRArg, ...],
    ctx: _LowerCtx,
    arg_names: tuple[str, ...] = (),
) -> Constraint:
    """Return the support of a call site, evaluating the sentinel
    when the family's support is a `dependent_property`."""
    cls_support = meta.distribution_class.support
    if isinstance(cls_support, c.Constraint) and not isinstance(
        cls_support, c._DependentProperty
    ):
        return cls_support
    instance = _make_sentinel(meta, args, ctx, arg_names)
    return instance.support


def _is_number_text(text: str) -> bool:
    """True when `text` parses as a Python float / int literal."""
    try:
        float(text)
    except ValueError:
        return False
    return True


def _scalar_binding_names(ctx: _LowerCtx) -> frozenset[str]:
    """Return the set of identifier names bound to a scalar value in
    the active program.

    Scalar bindings are program parameters whose ``type_params`` entry
    is a [`ScalarParam`][quivers.dsl.ast_nodes.declarations.ScalarParam]
    (`Real` / `Nat`). Used by `_wrap_for_constraint` to decide when an
    unindexed reference must be broadcast to satisfy an
    `IndependentConstraint` arg position.
    """
    program = ctx.program
    if program.type_params is None:
        return frozenset()
    return frozenset(p.name for p in program.type_params if isinstance(p, ScalarParam))


def _collect_let_expr_var_names(expr: LetExprNode, out: set[str]) -> None:
    """Walk `expr` collecting every leaf
    [`LetExprVar.name`][quivers.dsl.ast_nodes.LetExprVar.name]. Used
    by the plate-propagation pass to find which exogenous /
    previously-bound names a `let`-expression reads."""
    if isinstance(expr, LetExprVar):
        out.add(expr.name)
        return
    if isinstance(expr, LetExprBinOp):
        _collect_let_expr_var_names(expr.left, out)
        _collect_let_expr_var_names(expr.right, out)
        return
    if isinstance(expr, LetExprUnaryOp):
        _collect_let_expr_var_names(expr.operand, out)
        return
    if isinstance(expr, LetExprCall):
        for a in expr.args:
            _collect_let_expr_var_names(a, out)
        return
    if isinstance(expr, LetExprIndex):
        _collect_let_expr_var_names(expr.array, out)
        for i in expr.indices:
            _collect_let_expr_var_names(i, out)
        return
    if isinstance(expr, LetExprList):
        for e in expr.items:
            _collect_let_expr_var_names(e, out)
        return
    if isinstance(expr, LetExprLambda):
        _collect_let_expr_var_names(expr.body, out)
        return
    if isinstance(expr, LetExprFactor):
        for case in expr.cases:
            _collect_let_expr_var_names(case.value, out)
        if expr.body is not None:
            _collect_let_expr_var_names(expr.body, out)
        return
    if isinstance(expr, LetExprMethodCall):
        _collect_let_expr_var_names(expr.receiver, out)
        for a in expr.args:
            _collect_let_expr_var_names(a, out)
        return
    if isinstance(expr, LetExprAffineMap):
        _collect_let_expr_var_names(expr.weight, out)
        for source in expr.sources:
            _collect_let_expr_var_names(source.value, out)
        _collect_let_expr_var_names(expr.bias, out)
        return


__all__: list[str] = []
