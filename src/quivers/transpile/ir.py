"""Structural intermediate representation for transpile.

The IR is purely structural: no target-language strings, no schema
vertices, no panproto types. Every node is a `dx.Model` or
`dx.TaggedUnion`. `Lower` emits this representation; every
`Renderer[T]` consumes it.

The IR re-exports the existing let-expression tree
([`LetExprNode`][quivers.dsl.ast_nodes.let_expressions.LetExprNode])
as `IRExpr`: the deterministic let / score expressions ride through
unchanged so per-renderer translation can dispatch over the same
tagged union the surface compiler uses.

Support classification lives in the support predicates below: pure
functions over [`torch.distributions.constraints.Constraint`][torch.distributions.constraints.Constraint]
that renderers consult to pick declaration shapes. The IR stores
constraints as a [`ConstraintSpec`][quivers.transpile.ir.ConstraintSpec]
tagged union so the structural representation survives didactic's
encode / decode round-trip; each variant carries a `to_constraint`
that materialises the corresponding `torch.distributions.constraints.Constraint`
for predicate dispatch. Renderers never introspect constraint
classes directly.
"""

from __future__ import annotations

from typing import Literal

import didactic.api as dx
import torch.distributions.constraints as _constraints
from torch.distributions.constraints import Constraint

from quivers.dsl.ast_nodes.let_expressions import (
    LetExprNode,
)
from quivers.transpile.qiec_ir import IRQiecModule, IRQiecStatic, IRQiecValue


# ---------------------------------------------------------------------------
# Re-export the let-expression tree under the IR namespace.
# ---------------------------------------------------------------------------

#: The let-expression tree, lifted into the IR namespace without
#: wrapping. `IRDeterministic.expr` and `IRScore.expr` carry the
#: existing `LetExprNode` tree unchanged.
IRExpr = LetExprNode


class IRQiecValueExpr(LetExprNode):
    """A checked kernel value in a program body, as a call argument.

    A constructor value built in a program step, such as the
    ``Measurement`` a helper case-analyzes, has no tensor expression
    form; a renderer spells it through the target's QIEC runtime, under
    the value encoding the callee reads.
    """

    value: IRQiecValue
    kind: Literal["qiec_value"] = "qiec_value"


# ---------------------------------------------------------------------------
# The one let-expression variant the surface grammar cannot write.
#
# A declared kernel morphism's parameter map is an affine map from the
# concatenated domain coordinates to a block of the family's argument
# row. Spelling it with the arithmetic variants above costs one node
# per (codomain coordinate, domain coordinate) pair, and every node is
# a validated `dx.Model`, so a 16-wide state took hours to lower and
# emitted a program no reader could follow. `LetExprAffineMap` names
# the contraction instead: one node per head, whatever the widths.
#
# It lives here rather than in `quivers.dsl.ast_nodes` because the QVR
# surface has no syntax for it. Nothing parses to a
# `LetExprAffineMap`; `Lower` is its only constructor, and renderers
# its only readers. Being a `LetExprNode` subclass, it rides in
# `IRDeterministic.expr` alongside the variants the parser does
# produce, so plate derivation, name binding, and declaration
# emission treat a mapped head exactly as they treat any other
# deterministic binding.
# ---------------------------------------------------------------------------


class LetAffineSource(dx.Model):
    """One factor of the conditioning row an affine map reads.

    A morphism whose domain is a product reads the concatenation of
    its factors in declaration order, so
    [`LetExprAffineMap.sources`][quivers.transpile.ir.LetExprAffineMap]
    is ordered and `width` is the column count this factor occupies.
    `value` is the expression the factor's coordinates are read from,
    a `LetExprVar` naming a program input or a previously bound site.
    """

    value: LetExprNode
    width: int


class LetExprAffineMap(LetExprNode):
    """Represent one head of a parameter map's ``W x + b`` operation.

    For ``i`` in ``0 .. rows - 1``, the expression computes::

        y[i] = sum_j weight[row_offset + i, j] * x[j] + bias[row_offset + i]

    where ``x`` concatenates `sources` in order. The `transform` is either
    ``identity`` or coordinatewise ``exp``. Indices are zero-based; renderers
    rebase them for one-based targets. The node stores the contraction rather
    than unrolled arithmetic.
    """

    weight: LetExprNode
    bias: LetExprNode
    sources: tuple[LetAffineSource, ...]
    row_offset: int
    rows: int
    transform: Literal["identity", "exp"]
    kind: Literal["let_expr_affine_map"] = "let_expr_affine_map"


def affine_domain_width(expr: LetExprAffineMap) -> int:
    """Total column count of an affine map's conditioning row.

    Parameters
    ----------
    expr
        The affine map.

    Returns
    -------
    int
        The summed width of the map's sources.
    """
    return sum(source.width for source in expr.sources)


def affine_column_offsets(
    expr: LetExprAffineMap,
) -> tuple[tuple[LetAffineSource, int], ...]:
    """Each source paired with its zero-based first column.

    A renderer whose language cannot concatenate the factors into one
    vector slices the weight column-block-wise instead, and this
    gives it the block boundaries without recomputing the running
    sum.

    Parameters
    ----------
    expr
        The affine map.

    Returns
    -------
    tuple[tuple[LetAffineSource, int], ...]
        Each source with the zero-based column its block starts at, in
        source order.
    """
    out: list[tuple[LetAffineSource, int]] = []
    column = 0
    for source in expr.sources:
        out.append((source, column))
        column += source.width
    return tuple(out)


# ---------------------------------------------------------------------------
# ConstraintSpec: a structural mirror of `torch.distributions.constraints`.
# ---------------------------------------------------------------------------


class ConstraintSpec(dx.TaggedUnion, discriminator="kind"):
    """Structural mirror of one [`torch.distributions.constraints.Constraint`][torch.distributions.constraints.Constraint].

    Stored on every IR node that carries support / constraint
    information so the IR survives didactic's encode / decode round
    trip (raw `Constraint` instances do not). Predicates dispatch on
    `to_constraint()` to keep renderers reading torch's existing
    taxonomy.
    """

    def to_constraint(self) -> Constraint:
        """Materialise the corresponding torch `Constraint`."""
        raise NotImplementedError


class CSReal(ConstraintSpec):
    """`Real()`."""

    kind: Literal["real"] = "real"

    def to_constraint(self) -> Constraint:
        return _constraints.real


class CSPositive(ConstraintSpec):
    """`GreaterThanEq(0.0)` / `GreaterThan(0.0)` / `Positive()`."""

    strict: bool = False
    kind: Literal["positive"] = "positive"

    def to_constraint(self) -> Constraint:
        return _constraints.positive if self.strict else _constraints.nonnegative


class CSUnitInterval(ConstraintSpec):
    """`Interval(0, 1)` / `UnitInterval()`."""

    kind: Literal["unit_interval"] = "unit_interval"

    def to_constraint(self) -> Constraint:
        return _constraints.unit_interval


class CSInterval(ConstraintSpec):
    """`Interval(lower, upper)`."""

    lower: float
    upper: float
    kind: Literal["interval"] = "interval"

    def to_constraint(self) -> Constraint:
        return _constraints.interval(self.lower, self.upper)


class CSSimplex(ConstraintSpec):
    """`Simplex()`."""

    kind: Literal["simplex"] = "simplex"

    def to_constraint(self) -> Constraint:
        return _constraints.simplex


class CSRealVector(ConstraintSpec):
    """`IndependentConstraint(Real(), 1)`."""

    kind: Literal["real_vector"] = "real_vector"

    def to_constraint(self) -> Constraint:
        return _constraints.real_vector


class CSRealMatrix(ConstraintSpec):
    """`IndependentConstraint(Real(), 2)`."""

    kind: Literal["real_matrix"] = "real_matrix"

    def to_constraint(self) -> Constraint:
        return _constraints.independent(_constraints.real, 2)


class CSPositiveDefinite(ConstraintSpec):
    """`PositiveDefinite()`."""

    kind: Literal["positive_definite"] = "positive_definite"

    def to_constraint(self) -> Constraint:
        return _constraints.positive_definite


class CSCorrCholesky(ConstraintSpec):
    """`CorrCholesky()`."""

    kind: Literal["corr_cholesky"] = "corr_cholesky"

    def to_constraint(self) -> Constraint:
        return _constraints.corr_cholesky


class CSLowerCholesky(ConstraintSpec):
    """`LowerCholesky()`."""

    kind: Literal["lower_cholesky"] = "lower_cholesky"

    def to_constraint(self) -> Constraint:
        return _constraints.lower_cholesky


class CSOneHot(ConstraintSpec):
    """`OneHot()` (one-hot vector support of `OneHotCategorical`)."""

    kind: Literal["one_hot"] = "one_hot"

    def to_constraint(self) -> Constraint:
        return _constraints.one_hot


class CSBoolean(ConstraintSpec):
    """`Boolean()`."""

    kind: Literal["boolean"] = "boolean"

    def to_constraint(self) -> Constraint:
        return _constraints.boolean


class CSIntegerInterval(ConstraintSpec):
    """`IntegerInterval(lower, upper)`."""

    lower: int
    upper: int
    kind: Literal["integer_interval"] = "integer_interval"

    def to_constraint(self) -> Constraint:
        return _constraints.integer_interval(self.lower, self.upper)


class CSNonnegativeInteger(ConstraintSpec):
    """`NonnegativeInteger()`."""

    kind: Literal["nonnegative_integer"] = "nonnegative_integer"

    def to_constraint(self) -> Constraint:
        return _constraints.nonnegative_integer


class CSPositiveInteger(ConstraintSpec):
    """`PositiveInteger()`."""

    kind: Literal["positive_integer"] = "positive_integer"

    def to_constraint(self) -> Constraint:
        return _constraints.positive_integer


def from_constraint(c: Constraint) -> ConstraintSpec:
    """Convert a torch `Constraint` into a `ConstraintSpec` variant.

    Used by `Lower` to encode the family's reported support /
    arg_constraint into the IR. Unknown constraint subclasses raise
    so silent type-erasure cannot happen.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    ConstraintSpec
        The structural variant mirroring ``c``.

    Raises
    ------
    ValueError
        If ``c`` is a constraint kind with no structural variant.
    """
    if isinstance(c, _constraints._Real):
        return CSReal()
    if isinstance(c, _constraints._Interval):
        lo = float(c.lower_bound)
        hi = float(c.upper_bound)
        if lo == 0.0 and hi == 1.0:
            return CSUnitInterval()
        if lo == float("-inf") and hi == float("inf"):
            return CSReal()
        return CSInterval(lower=lo, upper=hi)
    if isinstance(c, _constraints._GreaterThan):
        return CSPositive(strict=True) if float(c.lower_bound) == 0.0 else CSReal()
    if isinstance(c, _constraints._GreaterThanEq):
        return CSPositive(strict=False) if float(c.lower_bound) == 0.0 else CSReal()
    if isinstance(c, _constraints._Simplex):
        return CSSimplex()
    if isinstance(c, _constraints._IndependentConstraint):
        if c.event_dim == 1 and isinstance(c.base_constraint, _constraints._Real):
            return CSRealVector()
        if c.event_dim == 2 and isinstance(c.base_constraint, _constraints._Real):
            return CSRealMatrix()
        # Independent of any base: collapse to the base's flat form
        # plus an annotation in `kind`. The structural information
        # we need (support taxonomy) is captured by the base.
        return from_constraint(c.base_constraint)
    if isinstance(c, _constraints._PositiveDefinite):
        return CSPositiveDefinite()
    if isinstance(c, _constraints._PositiveSemidefinite):
        return CSPositiveDefinite()
    if isinstance(c, _constraints._CorrCholesky):
        return CSCorrCholesky()
    if isinstance(c, _constraints._LowerCholesky):
        return CSLowerCholesky()
    if isinstance(c, _constraints._OneHot):
        return CSOneHot()
    if isinstance(c, _constraints._Boolean):
        return CSBoolean()
    if isinstance(c, _constraints._IntegerInterval):
        return CSIntegerInterval(lower=int(c.lower_bound), upper=int(c.upper_bound))
    if isinstance(c, _constraints._IntegerGreaterThan):
        if int(c.lower_bound) == 0:
            return CSNonnegativeInteger()
        if int(c.lower_bound) == 1:
            return CSPositiveInteger()
        return CSNonnegativeInteger()
    msg = (
        f"from_constraint: unrecognised torch constraint kind "
        f"{type(c).__name__} ({c!r})"
    )
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Plate decomposition
# ---------------------------------------------------------------------------


class Dim(dx.TaggedUnion, discriminator="kind"):
    """One dimension of a plate."""


class DimStatic(Dim):
    """Statically known cardinality, from a `FinSet N` lookup."""

    size: int
    name: str
    kind: Literal["static"] = "static"


class DimDynamic(Dim):
    """Length arrives at run time as a data input named `size_name`."""

    size_name: str
    name: str
    kind: Literal["dynamic"] = "dynamic"


class Plate(dx.Model):
    """Multi-dim plate decomposition of a sample / observe step.

    `event_dims` capture the family's joint structure (the event
    axes); `batch_dims` capture replication (the iid axes). The
    user's `iid_over` declaration order is preserved in
    `batch_dims`.
    """

    event_dims: tuple[Dim, ...]
    batch_dims: tuple[Dim, ...]


# ---------------------------------------------------------------------------
# Support predicates over `torch.distributions.constraints.Constraint`.
# ---------------------------------------------------------------------------


def _is_independent(c: Constraint, n: int) -> bool:
    """Return True iff `c` is an `IndependentConstraint` of rank `n`."""
    return isinstance(c, _constraints._IndependentConstraint) and c.event_dim == n


def is_real_scalar(c: Constraint) -> bool:
    """`Real()` or any interval that spans the full real line.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is the whole real line.
    """
    if isinstance(c, _constraints._Real):
        return True
    if isinstance(c, _constraints._Interval):
        return c.lower_bound == float("-inf") and c.upper_bound == float("inf")
    return False


def is_real_positive(c: Constraint) -> bool:
    """`GreaterThan(0)`, `Positive()`, `GreaterThanEq(0)`.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is the positive or nonnegative half-line.
    """
    if isinstance(c, _constraints._GreaterThan):
        return float(c.lower_bound) == 0.0
    if isinstance(c, _constraints._GreaterThanEq):
        return float(c.lower_bound) == 0.0
    return False


def is_real_unit_interval(c: Constraint) -> bool:
    """`Interval(0, 1)` / `UnitInterval()`.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is the unit interval.
    """
    if isinstance(c, _constraints._Interval):
        return float(c.lower_bound) == 0.0 and float(c.upper_bound) == 1.0
    return False


def is_real_bounded_interval(c: Constraint) -> bool:
    """`Interval(lo, hi)` with finite, non-(0,1) bounds.

    Distinguishes `Uniform(-1, 1)`-style supports from `UnitInterval`
    so the type emitter can produce `real <lower=lo, upper=hi>` rather
    than falling through to the unsupported-support fallback.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is a finite interval other than the unit interval.
    """
    if not isinstance(c, _constraints._Interval):
        return False
    lo = float(c.lower_bound)
    hi = float(c.upper_bound)
    if lo == float("-inf") or hi == float("inf"):
        return False
    if lo == 0.0 and hi == 1.0:
        return False
    return lo < hi


def real_interval_bounds(c: Constraint) -> tuple[float, float]:
    """The (lo, hi) bounds of a bounded-interval support.

    Parameters
    ----------
    c
        An interval constraint.

    Returns
    -------
    tuple[float, float]
        The lower and upper bound.
    """
    assert isinstance(c, _constraints._Interval)
    return float(c.lower_bound), float(c.upper_bound)


def is_real_vector(c: Constraint) -> bool:
    """`IndependentConstraint(Real(), 1)` (a vector of real scalars).

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is a vector of unconstrained reals.
    """
    if not _is_independent(c, 1):
        return False
    base = c.base_constraint
    return isinstance(base, _constraints._Real)


def is_real_simplex(c: Constraint) -> bool:
    """`Simplex()`.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is the probability simplex.
    """
    return isinstance(c, _constraints._Simplex)


def is_real_cov_matrix(c: Constraint) -> bool:
    """`PositiveDefinite()` / `PositiveSemiDefinite()`.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is a positive-definite or positive-semidefinite
        matrix.
    """
    if isinstance(c, _constraints._PositiveDefinite):
        return True
    if isinstance(c, _constraints._PositiveSemidefinite):
        return True
    return False


def is_real_corr_chol(c: Constraint) -> bool:
    """`CorrCholesky()` / `LowerCholesky()`.

    The predicate covers any lower-triangular Cholesky factor;
    correlation Cholesky is the constrained subcase.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is a lower-triangular Cholesky factor.
    """
    if isinstance(c, _constraints._CorrCholesky):
        return True
    if isinstance(c, _constraints._LowerCholesky):
        return True
    return False


def is_real_matrix(c: Constraint) -> bool:
    """`IndependentConstraint(Real(), 2)`.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is a matrix of unconstrained reals.
    """
    if not _is_independent(c, 2):
        return False
    base = c.base_constraint
    return isinstance(base, _constraints._Real)


def is_real_one_hot(c: Constraint) -> bool:
    """`OneHot()` (the one-hot vector support of `OneHotCategorical`).

    Degenerate elements of the simplex.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is the one-hot vector support.
    """
    return isinstance(c, _constraints._OneHot)


def is_int_bit(c: Constraint) -> bool:
    """`Boolean()` or `IntegerInterval(0, 1)`.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` admits exactly 0 and 1.
    """
    if isinstance(c, _constraints._Boolean):
        return True
    if isinstance(c, _constraints._IntegerInterval):
        return int(c.lower_bound) == 0 and int(c.upper_bound) == 1
    return False


def is_int_category(c: Constraint) -> bool:
    """`IntegerInterval(0, K-1)` with `K > 2`, or `IntegerInterval(1, K)`.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is a finite range of category indices.
    """
    if not isinstance(c, _constraints._IntegerInterval):
        return False
    lo = int(c.lower_bound)
    hi = int(c.upper_bound)
    if lo == 0 and hi >= 2:
        return True
    if lo == 1 and hi >= lo:
        return True
    return False


def is_int_count(c: Constraint) -> bool:
    """`NonnegativeInteger()`, `PositiveInteger()`, `IntegerGreaterThan(0)`.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    bool
        Whether ``c`` is an integer half-line.
    """
    if isinstance(c, _constraints._IntegerGreaterThan):
        return True
    return False


def event_dim_of(c: Constraint) -> int:
    """Return the constraint's `event_dim` attribute.

    Parameters
    ----------
    c
        A torch support or argument constraint.

    Returns
    -------
    int
        The constraint's ``event_dim``, 0 when it declares none.
    """
    return int(getattr(c, "event_dim", 0))


def event_shape_of(c: Constraint, base_event: tuple[int, ...]) -> tuple[int, ...]:
    """Lift an event shape from a base shape under the constraint.

    For a scalar constraint returns `base_event` unchanged. For an
    `IndependentConstraint(_, n)` returns `base_event` (the
    independence rank is taken into account by the caller, who
    supplies the per-dim sizes).

    Parameters
    ----------
    c
        A torch support or argument constraint.
    base_event
        The event shape before lifting.

    Returns
    -------
    tuple[int, ...]
        The lifted event shape.
    """
    return base_event


# ---------------------------------------------------------------------------
# IRArg: normalized argument to a distribution call.
# ---------------------------------------------------------------------------


class IRArg(dx.TaggedUnion, discriminator="kind"):
    """One argument position in a distribution call."""


class IRArgNumber(IRArg):
    """A numeric literal."""

    value: float
    kind: Literal["number"] = "number"


class IRArgRef(IRArg):
    """A reference to a bound name with zero or more index expressions.

    The bracket-indexed form `name[i0][i1]...` parses each index
    expression as another `IRArg`.
    """

    name: str
    indices: tuple[IRArg, ...] = ()
    kind: Literal["ref"] = "ref"


class IRArgBroadcast(IRArg):
    """A scalar broadcast to satisfy a vector / matrix arg constraint.

    `target_shape` is the broadcast target; the renderer emits its
    native broadcast op (`rep_vector(x, K)` in Stan,
    `jnp.full((K,), x)` in NumPyro, etc.). Lower constructs this
    when the user supplied a scalar but `arg_constraints[arg_name]`
    is `IndependentConstraint(base, n)` with `n >= 1`.
    """

    value: IRArg
    target_shape: tuple[int, ...]
    kind: Literal["broadcast"] = "broadcast"


class IRArgList(IRArg):
    """Vector literal from the `[a, b, c]` grammar production."""

    elements: tuple[IRArg, ...]
    kind: Literal["list"] = "list"


class IRArgMatrix(IRArg):
    """Matrix literal from the `[[a, b], [c, d]]` grammar production."""

    rows: tuple[IRArgList, ...]
    kind: Literal["matrix"] = "matrix"


class IRArgFamilyRef(IRArg):
    """A reference to a morphism whose `~ Family(...)` init clause
    names a distribution.

    Wrappers (`Truncated`, `Mixture`, `Independent`, `Transformed`,
    `LKJCorrelationFactor`) use this for their wrapped-distribution
    argument; the renderer reads the referenced morphism's
    declaration when emitting the wrapper call.
    """

    name: str
    kind: Literal["family_ref"] = "family_ref"


class IRArgKernel(IRArg):
    """A Gaussian-process kernel-covariance argument.

    Carries the kernel family name (``"rbf"`` is the only kernel
    `Lower` emits today), the positive ``length_scale`` hyperparameter,
    the data-input name that holds the input-locations vector
    (``"x"`` by convention), and the static cardinality of the grid
    axis. Renderers emit the backend-specific covariance matrix:
    Stan uses ``gp_exp_quad_cov(x, 1.0, length_scale)`` plus a
    diagonal jitter; NumPyro / Pyro / PyMC emit a
    ``jnp.exp(-0.5 * d2 / length_scale**2)`` expression; Turing /
    Gen build the matrix in Julia; WebPPL / Church emit nested
    loops.

    A small diagonal ``jitter`` is added for numerical positive-
    definiteness before passing the matrix to the
    MultivariateNormal sampler.
    """

    kernel: str
    length_scale: float
    x_name: str
    grid_size: int
    jitter: float = 1e-8
    kind: Literal["kernel"] = "kernel"


# ---------------------------------------------------------------------------
# Structured-args lowering metadata.
#
# A family whose `~ Family` clause carries no explicit arguments
# (today: MultivariateNormal, MatrixNormal, GP) declares a
# `StructuredSampleLowering` on its `FamilyMeta`. `Lower` walks the
# declared `args` tuple in order, synthesising the appropriate IRArg
# variant per spec and deriving every data-input plate from the
# sample's event axes. The result: one uniform code path replaces
# the per-family `_lower_sample_<family>` methods, and the
# per-sample data-input shapes flow from declarative metadata
# rather than family-name branches in `Lower._structured_input_specs`.
# ---------------------------------------------------------------------------


class StructuredArgSpec(dx.TaggedUnion, discriminator="kind"):
    """One arg position in a family's structured no-args lowering."""


class StructuredDataArg(StructuredArgSpec):
    """A data-input arg whose name is synthesised per sample site as
    ``<sample_name>_<arg_name>``.

    `axis_indices` indexes into the family's event-axis tuple to
    build the data-input plate: ``(0, 1)`` for an `event_axis_0 x
    event_axis_1` matrix, ``(0, 0)`` for an `event_axis_0 x
    event_axis_0` square matrix, ``(0,)`` for an `event_axis_0` vector.
    `constraint_kind` selects the IR constraint:

    - ``"real_matrix"`` :class:`CSRealMatrix`
    - ``"real_vector"`` :class:`CSRealVector`
    - ``"positive_definite"`` :class:`CSPositiveDefinite`
    """

    arg_name: str
    axis_indices: tuple[int, ...]
    constraint_kind: Literal[
        "real_matrix",
        "real_vector",
        "positive_definite",
    ]
    kind: Literal["data"] = "data"


class StructuredZeroVectorArg(StructuredArgSpec):
    """A zero-valued :class:`IRArgNumber` stand-in for a vector mean
    that the family treats as the all-zero vector of the sample's
    event size (today: GP's mean argument)."""

    arg_name: str
    kind: Literal["zero_vector"] = "zero_vector"


class StructuredKernelArg(StructuredArgSpec):
    """A GP kernel-covariance arg.

    The renderer reads the kernel family name and the positive
    `length_scale` from the morphism's `[kernel=..., length_scale=...]`
    option block and emits the backend-specific covariance matrix
    over the data-input vector named ``x_input_name``.
    """

    arg_name: str
    x_input_name: str
    kind: Literal["kernel"] = "kernel"


class EventAxisSource(dx.TaggedUnion, discriminator="kind"):
    """Where to read the sample's event axes from."""


class OverOrCodomainAxes(EventAxisSource):
    """Read event axes from the morphism's `[over=...]` option,
    falling back to the codomain product factors. The shape of MN
    and MVN."""

    axis_count: int
    kind: Literal["over_or_codomain"] = "over_or_codomain"


class DomainGridAxis(EventAxisSource):
    """Read the single event axis from the morphism's domain
    (a `FinSet N` object). The shape of GP."""

    kind: Literal["domain_grid"] = "domain_grid"


class StructuredSampleLowering(dx.Model):
    """How `Lower` constructs IR for a no-args `~ Family` sample.

    `args` is the ordered tuple of per-position specs (each one of
    :class:`StructuredDataArg`, :class:`StructuredZeroVectorArg`, or
    :class:`StructuredKernelArg`). `event_axis_source` declares how
    to recover the sample's event axes. `sample_constraint_kind`
    selects the IR constraint on the sample itself
    (``"real_matrix"`` or ``"real_vector"``). `always_apply` means
    this lowering fires unconditionally for the family even when the
    user supplied positional args (today: GP, whose kernel and grid
    axes have no user-facing arg surface).
    """

    args: tuple[StructuredArgSpec, ...]
    event_axis_source: EventAxisSource
    sample_constraint_kind: Literal["real_matrix", "real_vector"]
    always_apply: bool = False


# ---------------------------------------------------------------------------
# IRNode: top-level statements of a program body.
# ---------------------------------------------------------------------------


class IRNode(dx.TaggedUnion, discriminator="kind"):
    """One node in the lowered program body."""


class IRDataInput(IRNode):
    """An exogenous identifier the program reads but never binds.

    Renderers declare it in their data block / function signature /
    free-variable list. `constraint` is the support constraint
    derived from how the input is used.
    """

    name: str
    constraint: ConstraintSpec
    plate: Plate
    kind: Literal["data_input"] = "data_input"


class IRSample(IRNode):
    """A latent draw."""

    name: str
    family: str
    args: tuple[IRArg, ...]
    arg_names: tuple[str, ...]
    constraint: ConstraintSpec
    plate: Plate
    kind: Literal["sample"] = "sample"


class IRObserve(IRNode):
    """An observed value."""

    name: str
    family: str
    args: tuple[IRArg, ...]
    arg_names: tuple[str, ...]
    constraint: ConstraintSpec
    plate: Plate
    via: str | None
    kind: Literal["observe"] = "observe"


class IRDeterministic(IRNode):
    """A let-bound deterministic computation."""

    name: str
    expr: IRExpr
    constraint: ConstraintSpec
    plate: Plate
    kind: Literal["deterministic"] = "deterministic"


class IRScore(IRNode):
    """`target += <expr>` style scalar log-density increment."""

    name: str
    expr: IRExpr
    kind: Literal["score"] = "score"


class IRMarginalize(IRNode):
    """A scope that sums a finite latent out.

    Every renderer enumerates the atoms of the latent's support
    (`marginalize_support`), weighs each atom's scope log-density by
    the atom's prior log-mass, and reduces over the atoms, one
    reduction per group of the block's grouping plate; no target
    samples the latent. A family with no finite support is refused
    under `marginalize:non-finite-support`, except where a target
    declares a continuous latent as a parameter of the joint.
    """

    latent: str
    family: str
    args: tuple[IRArg, ...]
    arg_names: tuple[str, ...]
    constraint: ConstraintSpec
    plate: Plate
    reduction: Literal["logsumexp"]
    scope: tuple[IRNode, ...]
    kind: Literal["marginalize"] = "marginalize"


class IRCall(IRNode):
    """A call of one of the module's computations, bound to a name.

    The callee is rendered from the module as a host function; the
    call runs it under the target's own draw and score primitives for
    the program's `random` and `score` instances, whose stable
    identities the node carries so the renderer can attach them.
    """

    name: str
    callee: str
    static_arguments: tuple[IRQiecStatic, ...]
    arguments: tuple[IRExpr, ...]
    random_instance: str
    score_instance: str
    plate: Plate
    kind: Literal["call"] = "call"


class IRReturn(IRNode):
    """The program's terminal return clause."""

    names: tuple[str, ...]
    kind: Literal["return"] = "return"


# ---------------------------------------------------------------------------
# IRProgram: the lowered top-level program.
# ---------------------------------------------------------------------------


class IRProgram(dx.Model):
    """One lowered root: the checked module and a program's plan.

    `module` is the checked kernel module every target renders from,
    as structural IR; `body` is the plan of the entry point named
    `name`, derived from its computation, and `inputs` are the entry
    point's parameters. A module with no program has an empty plan
    and renders its computations alone.

    `cards` carries the static cardinalities of every QVR object
    used in the program, keyed by object name. Renderers consult
    it when an expression-level construct binds over a finite-set
    axis by name (`LetExprFactor` binders, for instance) and the
    static size is required to unroll the construct.
    """

    name: str
    inputs: tuple[IRDataInput, ...]
    body: tuple[IRNode, ...]
    module: IRQiecModule
    cards: dict[str, int] = dx.Field(default_factory=dict)


__all__ = [
    "CSBoolean",
    "CSCorrCholesky",
    "CSIntegerInterval",
    "CSInterval",
    "CSLowerCholesky",
    "CSNonnegativeInteger",
    "CSOneHot",
    "CSPositive",
    "CSPositiveDefinite",
    "CSPositiveInteger",
    "CSReal",
    "CSRealMatrix",
    "CSRealVector",
    "CSSimplex",
    "CSUnitInterval",
    "ConstraintSpec",
    "Dim",
    "DimDynamic",
    "DimStatic",
    "DomainGridAxis",
    "EventAxisSource",
    "IRArg",
    "IRArgBroadcast",
    "IRArgFamilyRef",
    "IRArgKernel",
    "IRArgList",
    "IRArgMatrix",
    "IRArgNumber",
    "IRArgRef",
    "IRCall",
    "IRDataInput",
    "IRDeterministic",
    "IRMarginalize",
    "IRNode",
    "IRObserve",
    "IRProgram",
    "IRQiecValueExpr",
    "IRReturn",
    "IRSample",
    "IRScore",
    "LetAffineSource",
    "LetExprAffineMap",
    "OverOrCodomainAxes",
    "Plate",
    "StructuredArgSpec",
    "StructuredDataArg",
    "StructuredKernelArg",
    "StructuredSampleLowering",
    "StructuredZeroVectorArg",
    "affine_column_offsets",
    "affine_domain_width",
    "event_dim_of",
    "event_shape_of",
    "from_constraint",
    "is_int_bit",
    "is_int_category",
    "is_int_count",
    "is_real_bounded_interval",
    "is_real_corr_chol",
    "is_real_cov_matrix",
    "is_real_matrix",
    "is_real_one_hot",
    "is_real_positive",
    "is_real_scalar",
    "is_real_simplex",
    "is_real_unit_interval",
    "is_real_vector",
    "real_interval_bounds",
]
