"""Parsed-formula IR: a typed `didactic.api.Model` wrapping
the raw `formulae.matrices.DesignMatrices` so the rest of
the formula frontend operates on typed values.

The Formula IR is the canonical *source* representation of the
formula→QVR lens.  Future versions can register it as a panproto
protocol so the lens machinery applies; for now the compiler walks
this IR directly.

Convention
----------
Each fixed-effect *term* may produce one or more design-matrix
*columns* (a single column for ``x``, two for ``poly(x, 2)``, ``K``
for an unordered factor with ``K + 1`` levels, etc.).  R / brms
assign one coefficient per *column*; this IR follows the same
convention by exploding each term into a tuple of
`FixedColumn` records.  Multi-column terms thus produce
multiple named scalar latents downstream, with deterministic naming
``{term}_1``, ``{term}_2``, ... that mirrors R's
``poly(x, 2)1`` / ``poly(x, 2)2`` display.

Polynomial default: `formulae.design_matrices`'s ``poly``
transform is orthogonal by default (matches R's
``stats::poly``).  Raw monomials are available via
``I(x^2)`` / ``I(x**2)``.  Transforms ``log``, ``exp``, ``sqrt``,
``abs``, ``sin``, ``cos``, ``tan``, ``log10``, ``log2``,
``log1p``, ``expm1`` are wired through the formulae evaluation
namespace so users coming from R get the expected base R behaviour.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Literal
import io
import re
import tokenize

import didactic.api as dx
import formulae as fo
from formulae.matrices import DesignMatrices
import narwhals as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_VALUE, ValueModel, opaque_by_value
from quivers.formulas._design import SymbolicColumn

# brms ``0 + Intercept``: a reserved constant column in class b.
from quivers.formulas._formula_options import (
    BRMS_INTERCEPT,
    display_label,
    normalize_intercept_term,
    normalize_r_syntax,
    without_zero_intercept,
)
from quivers.formulas._r_transforms import r_transform_namespace

# brms drop_unused_levels.
from quivers.formulas._data_options import (
    dropping_unused_levels,
    formula_data2,
    keep_unused_levels,
)
from quivers.formulas._gp import (
    GaussianProcess,
    GaussianProcessBlock,
    gp_namespace,
    gp_rows,
)
from quivers.formulas._group_layout import GroupBlockLayout
from quivers.formulas._group_law import GroupBlockLaw, LevelSite
from quivers.formulas._latent_effects import LatentTerm
from quivers.formulas._priors import PriorTarget
from quivers.formulas._term import FormulaField, FormulaPart
from quivers.formulas._ordered_contrasts import apply_ordered_contrasts
from quivers.formulas._ordinal_coding import (
    CategoryCoding,
    category_positions,
    integer_category_values,
    is_zero_coded,
)
from quivers.formulas._row_retention import (
    complete_rows,
    compose_positions,
    role_columns,
)
from quivers.formulas._mgcv_smooth import BSplineBasis
from quivers.formulas._grouping import (
    GroupingPlan,
    add_grouping_columns,
    fit_group_options,
    fit_membership,
    is_member_call,
    member_covariate,
    rewrite_grouping,
)
from quivers.formulas._monotonic import (
    Monotonic,
    MonotonicTerm,
    monotonic_design,
    monotonic_positions,
    monotonic_slopes,
)
from quivers.formulas._smooth import (
    SmoothState,
    SmoothTransform,
    r_list,
    smooth_namespace,
)


#: R-style transforms wired into formulae's evaluation namespace.
#: Matches base R's standard ``log`` / ``exp`` / ``sqrt`` / etc.
#: behaviour in regression formulas.
_R_TRANSFORMS = {
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "log10": np.log10,
    "log2": np.log2,
    "log1p": np.log1p,
    "expm1": np.expm1,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "cs": np.asarray,
}


_CS_ONES = "qvrCSInterceptOnes"


def _normalize_cs_intercept(formula: str) -> str:
    """Give category-specific literal intercepts an explicit dataframe row axis.

    Parameters
    ----------
    formula : str
        Formula accepted by formulae, with optional cs(1) calls.

    Returns
    -------
    str
        Formula with literal-one cs arguments replaced by a reserved column.
    """
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(formula).readline)
        if token.type
        not in {
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
        }
    ]
    lines = formula.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    spans: list[tuple[int, int]] = []
    for index in range(len(tokens) - 3):
        name, opening, value, closing = tokens[index : index + 4]
        if (
            name.type == tokenize.NAME
            and name.string == "cs"
            and opening.string == "("
            and closing.string == ")"
            and value.type == tokenize.NUMBER
        ):
            try:
                is_one = float(value.string) == 1.0
            except ValueError:
                is_one = False
            if is_one:
                spans.append(
                    (
                        starts[value.start[0] - 1] + value.start[1],
                        starts[value.end[0] - 1] + value.end[1],
                    )
                )
    for start, end in reversed(spans):
        formula = formula[:start] + _CS_ONES + formula[end:]
    return formula


def _contains_cs(expression: object) -> bool:
    """Recognize cs calls structurally in formulae's lazy-expression tree.

    Parameters
    ----------
    expression : object
        Lazy formulae expression.

    Returns
    -------
    bool
        Whether this expression contains a category-specific modifier.
    """
    return bool(_modifier_calls(expression, frozenset({"cs"})))


def _exposure(values: object) -> np.ndarray:
    """Convert positive observation exposures to log offsets.

    Parameters
    ----------
    values : object
        Numeric exposure values, one per observation.

    Returns
    -------
    np.ndarray
        Natural logarithms of the exposures.

    Raises
    ------
    ValueError
        If an exposure is nonpositive or nonfinite.
    """
    exposure = np.asarray(values, dtype=np.float64)
    if not np.isfinite(exposure).all() or np.any(exposure <= 0):
        raise ValueError("exposure terms require strictly positive finite values")
    return np.log(exposure)


def _modifier_calls(
    expression: object, names: frozenset[str] = frozenset({"offset", "exposure"})
) -> set[str]:
    """Collect selected modifiers within a formulae lazy expression.

    Parameters
    ----------
    expression : object
        Parsed call, operator, variable, or literal.
    names : frozenset[str]
        Modifier names to collect through the same expression traversal.

    Returns
    -------
    set[str]
        Selected modifier names found in the expression tree.
    """
    callee = getattr(expression, "callee", None)
    found = {callee} if callee in names else set()
    for argument in getattr(expression, "args", ()):
        found.update(_modifier_calls(argument, names))
    for argument in getattr(expression, "kwargs", {}).values():
        found.update(_modifier_calls(argument, names))
    return found


def _read_only(values: np.ndarray) -> np.ndarray:
    """A C-contiguous read-only copy of an array.

    A model compares its arrays by content and caches its hash, so it
    keeps a copy no caller can write through.

    Parameters
    ----------
    values : np.ndarray
        The array, or anything ``np.array`` reads as one.

    Returns
    -------
    np.ndarray
        The copy, with its writeable flag cleared.
    """
    frozen = np.array(values, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen


def _read_only_columns(columns: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Read-only copies of each column of a mapping.

    Parameters
    ----------
    columns : Mapping[str, np.ndarray]
        The columns, by name.

    Returns
    -------
    dict[str, np.ndarray]
        A read-only copy of each column, by name.
    """
    return {name: _read_only(column) for name, column in columns.items()}


class FixedColumn(ValueModel):
    """One evaluated numeric column retained by a regression formula.

    Fixed-design columns receive coefficients. Offset and response-weight
    columns reuse the same immutable data record without coefficient sites.

    Columns compare by value: ``data`` is stored as a read-only copy
    made at construction and compared exactly (dtype, shape, and
    bytes), so two columns are equal only when their labels and data
    agree.

    Attributes
    ----------
    term : str
        Originating term name (e.g. ``"poly(x, 2)"`` or ``"x"``).
    name : str
        Per-column label, equal to ``term`` for single-column terms
        and ``f"{term}_{k+1}"`` (1-indexed, matching R's display) for
        multi-column terms like ``poly(x, 2)``.
    qvr_name : str
        QVR-legal identifier derived from `name` (alnum / ``_``
        only). Fixed-design inputs use this identifier; offsets and
        response weights use compiler-owned input names.
    is_intercept : bool
        ``True`` for the constant-1 proportional predictor column.
    category_specific : bool
        Whether cs(expr) gives this column one coefficient per ordinal transition.
    data : np.ndarray
        The column's values, shape ``(N,)``; read-only.
    """

    term: str
    name: str
    qvr_name: str
    is_intercept: bool = False
    category_specific: bool = False
    data: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_read_only))


class RandomTerm(ValueModel):
    """One random-effect group, e.g. ``(1 | g)`` or ``(x | g)``.

    Attributes
    ----------
    slope : str
        ``"Intercept"`` for ``(1 | g)``; otherwise the slope
        variable name.
    group : str
        Grouping factor name.
    correlation_id : str or None
        Correlation-block identity; ``None`` requests independent effects.
    shared : bool
        Whether the identity was explicitly named and can join blocks across
        location and auxiliary formulas.
    column : FixedColumn or None
        Evaluated slope column, including categorical contrasts and numeric
        transforms. The intercept uses a constant column.
    """

    slope: str
    group: str
    correlation_id: str | None = None
    shared: bool = False
    column: FixedColumn | SymbolicColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


class SplineBasis(ValueModel):
    """Immutable structural state of an evaluated B-spline basis.

    Attributes
    ----------
    knots : tuple[float, ...]
        Complete nondecreasing knot sequence, including repeated boundaries.
    degree : int
        Nonnegative polynomial degree. The source basis width is the knot
        count minus degree minus one.
    """

    knots: tuple[float, ...]
    degree: int

    @dx.model_validator()
    def _validate(self) -> None:
        """Validate immutable knot geometry.

        Raises
        ------
        ValueError
            If knots or polynomial degree cannot define the retained basis.
        """
        if (
            type(self.degree) is not int
            or self.degree < 0
            or len(self.knots) <= self.degree + 1
            or not np.isfinite(self.knots).all()
            or any(left > right for left, right in zip(self.knots, self.knots[1:]))
            or self.knots[0] >= self.knots[-1]
        ):
            raise ValueError(
                "spline structure requires finite nondecreasing knots and a valid polynomial degree"
            )


def _distinct_columns(columns: Sequence[FixedColumn]) -> list[FixedColumn]:
    """Give every design column an identifier of its own.

    Parameters
    ----------
    columns : Sequence[FixedColumn]
        Columns whose labels may normalize to one identifier.

    Returns
    -------
    list[FixedColumn]
        The columns. A later column whose identifier is taken has the next
        free number after it.
    """
    taken = {column.qvr_name for column in columns}
    used: set[str] = set()
    result = []
    for column in columns:
        name, number = column.qvr_name, 2
        while name in used or name != column.qvr_name and name in taken:
            name, number = f"{column.qvr_name}_{number}", number + 1
        used.add(name)
        result.append(column.with_(qvr_name=name))
    return result


class GaussianDesignTerm(ValueModel):
    """One coupled Gaussian coefficient block evaluated on response rows.

    Attributes
    ----------
    term, qvr_name : str
        Authored term label and normalized runtime design identifier.
    column_indices : tuple[int, ...]
        Columns of the retained formula evaluator carrying penalized coordinates.
    data : np.ndarray
        Read-only response-row by Gaussian-coordinate design matrix. Coordinates
        multiply standard-Normal innovations and one independently sampled scale.
        Unpenalized null-space columns retain ordinary population priors.
    basis : SplineBasis or None
        Retained structural basis state when generated from a spline.
    coefficient_map : tuple[tuple[float, ...], ...]
        Original basis coordinates to evaluated null and Gaussian coordinates,
        in their retained evaluator-column order; empty for a supplied design.
        For one level of a factor ``by`` smooth, the map applies before the
        rows of other levels are set to zero.
    constraint : tuple[float, ...]
        Original training-basis mean defining the identification constraint.
    scale_group : str
        Name of a smoothing scale shared with other blocks (an mgcv ``id``),
        or empty when this block samples its own scale.
    gp : GaussianProcessBlock or None
        For an approximate ``gp()`` term, the block whose coordinates take
        per-coordinate scales ``sqrt(spd)`` of sampled ``sdgp`` and
        ``lscale`` in place of one smoothing scale. For an exact ``gp()``
        term, the block whose coordinates are its distinct inputs: ``data``
        maps rows to inputs with their ``by`` contrast values, and
        ``column_indices`` name the ``1 + D`` evaluated columns it reads.
    """

    term: str
    qvr_name: str
    column_indices: tuple[int, ...]
    data: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_read_only))
    basis: SplineBasis | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    coefficient_map: tuple[tuple[float, ...], ...] = ()
    constraint: tuple[float, ...] = ()
    scale_group: str = ""
    gp: GaussianProcessBlock | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate(self) -> None:
        """Validate retained row and coefficient coordinate geometry.

        Raises
        ------
        ValueError
            If design, basis map, or centering dimensions disagree.
        """
        if (
            self.data.ndim != 2
            or self.data.dtype.kind not in "iuf"
            or not np.isfinite(self.data).all()
        ):
            raise ValueError(
                "Gaussian design terms require a finite real row-by-coordinate matrix"
            )
        # An exact gp() block reads 1 + D evaluated columns into one design
        # column per distinct input.
        exact_gp = self.gp is not None and self.gp.exact
        if (
            self.data.shape[1] < 1
            or (not exact_gp and len(self.column_indices) != self.data.shape[1])
            or tuple(sorted(set(self.column_indices))) != self.column_indices
            or any(type(index) is not int or index < 0 for index in self.column_indices)
        ):
            raise ValueError(
                "Gaussian design coordinate indices must align with the retained basis width"
            )
        if self.coefficient_map:
            matrix = np.asarray(self.coefficient_map)
            if (
                matrix.ndim != 2
                or not np.isfinite(matrix).all()
                or matrix.shape[1] <= max(self.column_indices)
                or len(self.constraint) not in {0, matrix.shape[0]}
            ):
                raise ValueError(
                    "retained Gaussian coordinate maps must align with source basis and centering"
                )
            if (
                self.basis is not None
                and matrix.shape[0] != len(self.basis.knots) - self.basis.degree - 1
            ):
                raise ValueError(
                    "retained Gaussian coordinate map disagrees with its spline basis width"
                )


class CensoringSpec(ValueModel):
    """Retain recorded event codes and interval endpoints on response rows.

    Parameters
    ----------
    status : FixedColumn
        Codes minus one left, zero exact, one right, and two interval.
    upper : FixedColumn or None
        Interval upper endpoints with safe placeholders on inactive rows.
    """

    status: FixedColumn = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    upper: FixedColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


class TruncationSpec(ValueModel):
    """Retain inclusive sampling bounds without changing predictor columns.

    Parameters
    ----------
    lower, upper : FixedColumn or None
        Constant or rowwise inclusive bounds; None leaves that side unbounded.
    """

    lower: FixedColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    upper: FixedColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


class StandardErrorSpec(ValueModel):
    """Retain known measurement errors and the residual-scale policy.

    Parameters
    ----------
    column : FixedColumn
        Finite nonnegative standard errors in response units.
    sigma : bool
        Whether to include the estimated residual scale in quadrature. The
        default False matches brms: known errors alone scale the likelihood.
    """

    column: FixedColumn = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    sigma: bool = False


class RowSelection(ValueModel):
    """Map included response rows to their original dataframe positions.

    Parameters
    ----------
    size : int
        Original row count before selection.
    positions : tuple[int, ...]
        Included positions in original order.
    """

    size: int
    positions: tuple[int, ...]


class ThresholdGroups(ValueModel):
    """Group-specific ordinal threshold vectors of brms ``thres(n, gr)``.

    Parameters
    ----------
    column : FixedColumn
        Zero-based code of each row's threshold group, an index into
        ``levels``.
    levels : tuple[str, ...]
        Group labels in brms's sorted factor order.
    counts : tuple[int, ...]
        Number of thresholds of each group, one fewer than its number of
        categories. Empty until resolved from the response when the formula
        declares no count.
    extra_category : bool
        Whether response code zero is a category outside the thresholds (the
        hurdle of ``hurdle_cumulative``), so that a group with ``n``
        thresholds has the codes ``0, ..., n + 1``.
    """

    column: FixedColumn = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    levels: tuple[str, ...]
    counts: tuple[int, ...] = ()
    extra_category: bool = False

    @dx.model_validator()
    def _validate(self) -> None:
        """Require one positive count per labeled group.

        Raises
        ------
        ValueError
            If the levels are empty or repeated, or the counts do not give
            one positive integer per level.
        """
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError("threshold groups require distinct labels")
        if self.counts and (
            len(self.counts) != len(self.levels)
            or any(count < 1 for count in self.counts)
        ):
            raise ValueError(
                "threshold groups require one positive threshold count per group"
            )


class ResponseModifiers(ValueModel):
    """Retain typed response-row inputs independently of predictor coefficients.

    Attributes
    ----------
    weights : FixedColumn or None
        Finite nonnegative observation weights, optionally scaled after selection.
    decision : FixedColumn or None
        Binary boundary decisions, zero for lower and one for upper. Numeric
        0/1 and lower/upper labels normalize to the same integer carrier.
    censoring : CensoringSpec or None
        Recorded exact or censored outcome events.
    truncation : TruncationSpec or None
        Inclusive bounds on the response sampling law.
    standard_error : StandardErrorSpec or None
        Known measurement errors and optional residual scale.
    subset : RowSelection or None
        Structural row selection performed before evaluating the design: the
        rows kept by ``subset`` and by removing rows with missing model
        values, as original positions.
    trials : FixedColumn or None
        Exact finite trial counts supplied by a response modifier.
    rate : FixedColumn or None
        Positive response denominators of brms ``rate()``, multiplying the
        count mean (and the negative binomial shape) of each row.
    thresholds : int or None
        Number of ordinal thresholds declared by brms ``thres()`` or ``cat()``
        (one fewer than the number of categories), which may exceed the
        number of observed categories.
    threshold_groups : ThresholdGroups or None
        Row codes, labels, and threshold counts of brms ``thres(n, gr)``,
        whose groups each own one threshold vector.
    weights_scale : bool
        Whether weights were normalized to the number of included rows.
    sources : tuple[tuple[str, str or int or float or bool], ...]
        Authored column references and scalar literals used to bind fresh
        sampling-law inputs without requiring recorded outcomes.
    """

    weights: FixedColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    decision: FixedColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    censoring: CensoringSpec | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    truncation: TruncationSpec | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    standard_error: StandardErrorSpec | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    subset: RowSelection | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    trials: FixedColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    # mixtures package (WP2b): brms rate() denominators.
    rate: FixedColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    thresholds: int | None = None
    threshold_groups: ThresholdGroups | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    weights_scale: bool = False
    sources: tuple[tuple[str, str | int | float | bool], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def columns(self) -> tuple[tuple[str, FixedColumn], ...]:
        """Return supplied role-column pairs in canonical input order.

        Returns
        -------
        tuple of role and FixedColumn
            Checked modifier inputs with no fitted coefficient.
        """
        columns = [("weights", self.weights), ("decision", self.decision)]
        if self.censoring is not None:
            columns.extend(
                (
                    ("censoring", self.censoring.status),
                    ("censoring_upper", self.censoring.upper),
                )
            )
        if self.truncation is not None:
            columns.extend(
                (
                    ("trunc_lower", self.truncation.lower),
                    ("trunc_upper", self.truncation.upper),
                )
            )
        if self.standard_error is not None:
            columns.append(("standard_error", self.standard_error.column))
        columns.append(("trials", self.trials))
        columns.append(("rate", self.rate))
        if self.threshold_groups is not None:
            columns.append(("threshold_group", self.threshold_groups.column))
        return tuple((role, column) for role, column in columns if column is not None)


class ResponseLayout(ValueModel):
    """Retain event coordinates independently of the observation-row axis.

    Parameters
    ----------
    component_names : tuple[str, ...]
        Ordered labels of a vector event; empty for a scalar response.
    carrier : Literal["real", "integer"]
        Numeric carrier of the evaluated response, before family validation.
    category_levels : tuple[str, ...]
        Scalar factor labels in their evaluated contrast order; distinct
        from vector event component names.
    ordered : bool
        Whether the original factor declares an ordering.
    category_values : tuple[int, ...]
        Integer response value of each category, for a response whose
        categories are integers read as brms reads them (ordinal codes
        ``1, ..., K``). Empty for a factor response and for the zero-based
        codes ``0, ..., K - 1``.
    """

    component_names: tuple[str, ...] = ()
    carrier: Literal["real", "integer"] = "real"
    category_levels: tuple[str, ...] = ()
    ordered: bool = False
    category_values: tuple[int, ...] = ()

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Return event dimensions, excluding observation rows.

        Returns
        -------
        tuple[int, ...]
            Empty for a scalar response, or one extent for vector components.
        """
        return (len(self.component_names),) if self.component_names else ()


class Formula(ValueModel):
    """A parsed regression formula plus the data it was parsed
    against.

    Formulas compare by value over every field: two formulas are equal
    only when their formula strings, response names, fixed columns
    (labels and data), random terms, offsets, response weights, response
    values, group levels, and group indices all agree. The arrays are stored as read-only
    copies made at construction.

    Attributes
    ----------
    formula : str
        Original formula string.
    response_name : str
        Name of the response column.
    fixed_columns : tuple[FixedColumn, ...]
        One entry per design-matrix column (matches R/brms's
        one-coefficient-per-column convention).
    design_terms : tuple[GaussianDesignTerm, ...]
        Coupled Gaussian basis contributions with separate penalty-scale priors.
    random_terms : tuple[RandomTerm, ...]
        Random-effect group specifications.
    response_values : np.ndarray
        Response values with shape ``(N, *response_layout.event_shape)``; read-only.
    response_layout : ResponseLayout
        Event coordinates and evaluated numeric carrier.
    group_levels : dict[str, tuple[str, ...]]
        Canonical level ordering per grouping factor, used to derive
        deterministic plate-index tensors.
    modifiers : ResponseModifiers
        Checked response-row weights and decision columns, independent of
        predictor coefficients.
    offsets : tuple[FixedColumn, ...]
        Evaluated ``offset(expr)`` and ``exposure(expr)`` contributions on
        the linear-predictor scale. These columns receive no coefficients.
    group_indices : dict[str, tuple[int, ...]]
        Per-group integer index array, shape ``(N,)``.
    latent_terms : tuple[LatentTerm, ...]
        Structured latent vectors added to the location predictor, such as
        brms ``car`` terms, with their retained row-to-coordinate coding.
    monotonic_terms : tuple[MonotonicTerm, ...]
        brms ``mo()`` special-effects columns: cumulative category indicators
        of each monotonic factor and the covariate of the other components.
    grouping : GroupingPlan
        brms interaction (``g1:g2``) and multi-membership (``mm()``) grouping
        factors, multi-membership codes and weights, member covariates, and
        ``gr(cov = )`` matrix names.
    """

    formula: str
    response_name: str
    fixed_columns: tuple[FixedColumn | SymbolicColumn, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    design_terms: tuple[GaussianDesignTerm, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    random_terms: tuple[RandomTerm, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    response_layout: ResponseLayout = dx.field(
        default_factory=ResponseLayout, opaque=True, extras=OPAQUE_BY_VALUE
    )
    response_values: np.ndarray = dx.field(
        opaque=True, extras=opaque_by_value(_read_only)
    )
    group_levels: dict[str, tuple[str, ...]] = dx.field(default_factory=dict)
    group_indices: dict[str, tuple[int, ...]] = dx.field(default_factory=dict)
    offsets: tuple[FixedColumn, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    modifiers: ResponseModifiers = dx.field(
        default_factory=ResponseModifiers, opaque=True, extras=OPAQUE_BY_VALUE
    )
    latent_terms: tuple[LatentTerm, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    monotonic_terms: tuple[MonotonicTerm, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    grouping: GroupingPlan = dx.field(
        default_factory=GroupingPlan, opaque=True, extras=OPAQUE_BY_VALUE
    )

    def without(self, parts: Collection[FormulaPart]) -> Formula:
        """Return the formula without some of its elements.

        Parameters
        ----------
        parts : Collection[FormulaPart]
            Elements to leave out.

        Returns
        -------
        Formula
            The formula with every other element, in authored order.
        """

        def absent(field: FormulaField, key: str) -> bool:
            return FormulaPart(field=field, key=key) in parts

        return self.with_(
            fixed_columns=tuple(
                column
                for column in self.fixed_columns
                if not absent("fixed_columns", column.name)
            ),
            design_terms=tuple(
                term
                for term in self.design_terms
                if not absent("design_terms", term.qvr_name)
            ),
            random_terms=tuple(
                term
                for term in self.random_terms
                if not absent("random_terms", f"{term.group}:{term.slope}")
            ),
            latent_terms=tuple(
                term
                for term in self.latent_terms
                if not absent("latent_terms", term.name)
            ),
            monotonic_terms=tuple(
                term
                for term in self.monotonic_terms
                if not absent("monotonic_terms", term.name)
            ),
            offsets=tuple(
                column
                for index, column in enumerate(self.offsets)
                if not absent("offsets", str(index))
            ),
        )


class Witness(ValueModel):
    """The program binding that shows one element of a formula.

    Parameters
    ----------
    part : FormulaPart
        Element of the formula.
    binding : str
        Name in the program.
    """

    part: FormulaPart
    binding: str


class FormulaData(ValueModel):
    """The complement of a `Formula` under its translation to a program.

    The program states which terms a model has. The complement holds what a
    program cannot state: the data, the authored names, and the binding by
    which each authored element is recognized. ``backward(module, complement)``
    reads the terms from the module and their content from here.

    Parameters
    ----------
    source : Formula
        The translated formula.
    witnesses : tuple[Witness, ...]
        Binding of each element of the formula in the program.
    group_laws : tuple[GroupBlockLaw, ...]
        Laws of the group blocks of the program.
    level_sites : tuple[LevelSite, ...]
        Sites holding values for every level of a grouping factor.
    prior_targets : tuple[PriorTarget, ...]
        Parameter roles, authored labels, and nominal coefficient axes of the
        program's sites.
    """

    source: Formula = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    witnesses: tuple[Witness, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    group_laws: tuple[GroupBlockLaw, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    level_sites: tuple[LevelSite, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    prior_targets: tuple[PriorTarget, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @property
    def group_layouts(self) -> tuple[GroupBlockLayout, ...]:
        """Return the coordinate layout of each group block.

        Returns
        -------
        tuple[GroupBlockLayout, ...]
            Raw sites and coefficient slices derived from the block laws.
        """
        return tuple(law.layout for law in self.group_laws)


def _qvr_name(raw: str) -> str:
    """Normalize a label into a legal QVR identifier."""
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in raw)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


_SMOOTH_NAMES = frozenset({"s", "t2", "te", "ti"})
#: R's logical literals and NA, bound for brms term options such as ``iso = FALSE``.
R_LITERALS: dict[str, bool | None] = {"TRUE": True, "FALSE": False, "NA": None}
_SMOOTH_CALLS = frozenset({"s", "t2"})


def _r_vector(*values: int | float | str) -> tuple[int | float | str, ...]:
    """Evaluate R's ``c()`` for literal smooth options such as ``m = c(3, 1)``.

    Parameters
    ----------
    *values : int, float, or str
        Literal entries.

    Returns
    -------
    tuple[int | float | str, ...]
        The entries in order.

    Raises
    ------
    ValueError
        If an entry is not a scalar literal.
    """
    if any(not isinstance(value, (int, float, str)) for value in values):
        raise ValueError(
            "c() in a formula builds vectors of literal smooth options only"
        )
    return values


def _smooth_design(
    term_name: str, state: SmoothState, columns: list[FixedColumn]
) -> tuple[list[FixedColumn], list[GaussianDesignTerm]]:
    """Split an evaluated smooth into population columns and Gaussian blocks.

    Parameters
    ----------
    term_name : str
        Authored formulae term.
    state : SmoothState
        Retained smooth with its block layout.
    columns : list[FixedColumn]
        Exploded evaluated columns of the term.

    Returns
    -------
    tuple[list[FixedColumn], list[GaussianDesignTerm]]
        Unpenalized population columns and one design block per penalized
        group.
    """
    layouts = state.layout
    width = len(columns)
    basis = (
        SplineBasis(knots=state.basis.knots, degree=state.basis.order - 1)
        if isinstance(state.basis, BSplineBasis)
        else None
    )
    fixed = [columns[index] for layout in layouts for index in layout.fixed]
    terms = []
    for position, (block, layout) in enumerate(zip(state.blocks, layouts, strict=True)):
        mapping = np.asarray(block.coefficient_map)
        padded = np.zeros((mapping.shape[0], width))
        padded[:, list(layout.fixed)] = mapping[:, : block.fixed_width]
        offset = block.fixed_width
        for group_index, group in enumerate(layout.random):
            padded[:, list(group)] = mapping[:, offset : offset + len(group)]
            offset += len(group)
        for group_index, group in enumerate(layout.random):
            suffix = f"_by{position + 1}" if block.level is not None else ""
            if len(layout.random) > 1:
                suffix += f"_{group_index + 1}"
            terms.append(
                GaussianDesignTerm(
                    term=term_name,
                    qvr_name="smooth_" + _qvr_name(term_name) + suffix,
                    column_indices=group,
                    data=np.column_stack([columns[index].data for index in group]),
                    basis=basis,
                    coefficient_map=tuple(map(tuple, padded.tolist())),
                    constraint=state.constraint,
                    scale_group=(
                        f"smooth_id_{_qvr_name(state.identity)}_{group_index + 1}"
                        if state.identity is not None
                        else ""
                    ),
                )
            )
    return fixed, terms


def _gp_design(
    term_name: str, transform: GaussianProcess, columns: list[FixedColumn]
) -> list[GaussianDesignTerm]:
    """Split an evaluated ``gp()`` term into one Gaussian block per sub-GP.

    Parameters
    ----------
    term_name : str
        Authored formulae term.
    transform : GaussianProcess
        Retained transform with its sub-GPs.
    columns : list[FixedColumn]
        Exploded evaluated basis columns, sub-GPs in order.

    Returns
    -------
    list[GaussianDesignTerm]
        One block per sub-GP.
    """
    assert transform.state is not None
    terms = []
    start = 0
    for subterm in transform.state.subterms:
        # An exact gp() sub-GP evaluates its contrast values and covariates.
        width = (
            subterm.orders.shape[1] + 1
            if subterm.block.exact
            else subterm.orders.shape[0]
        )
        indices = tuple(range(start, start + width))
        start += width
        data = np.column_stack([columns[index].data for index in indices])
        if subterm.block.exact:
            data = gp_rows(data, subterm.block)[1]
        terms.append(
            GaussianDesignTerm(
                term=term_name,
                qvr_name="gp_" + _qvr_name(subterm.block.label),
                column_indices=indices,
                data=data,
                gp=subterm.block,
            )
        )
    return terms


def _explode_term(term_name: str, term, n_obs: int) -> list[FixedColumn]:
    """Explode a single formulae Term into one or more FixedColumn
    entries, matching R/brms's one-coefficient-per-column convention.
    """
    data = np.asarray(term.data)
    category_specific = any(
        _contains_cs(getattr(component, "call", None))
        for component in getattr(term, "components", ())
    )
    display_name = display_label(term_name.replace(_CS_ONES, "1"))
    if data.ndim == 1:
        # Single column; treat as one coefficient.
        is_intercept = (
            getattr(term, "kind", None) == "intercept" or term_name == "Intercept"
        )
        return [
            FixedColumn(
                term=term_name,
                name=display_name,
                qvr_name=("intercept" if is_intercept else _qvr_name(display_name)),
                is_intercept=is_intercept,
                category_specific=category_specific,
                data=data.astype(np.float64),
            )
        ]
    if data.shape[0] != n_obs:
        raise ValueError(
            f"formula_from_data: term {term_name!r} produced "
            f"{data.shape[0]} rows, expected {n_obs}"
        )
    n_cols = data.shape[1]
    if n_cols == 1:
        is_intercept = (
            getattr(term, "kind", None) == "intercept" or term_name == "Intercept"
        )
        return [
            FixedColumn(
                term=term_name,
                name=display_name,
                qvr_name=("intercept" if is_intercept else _qvr_name(display_name)),
                is_intercept=is_intercept,
                category_specific=category_specific,
                data=data[:, 0].astype(np.float64),
            )
        ]
    return [
        FixedColumn(
            term=term_name,
            name=f"{display_name}_{k + 1}",
            qvr_name=_qvr_name(f"{display_name}_{k + 1}"),
            is_intercept=False,
            category_specific=category_specific,
            data=data[:, k].astype(np.float64),
        )
        for k in range(n_cols)
    ]


def _random_blocks(
    formula: str,
) -> tuple[str, dict[tuple[str, str], tuple[str | None, bool]]]:
    """Normalize bar notation and retain each random-effect correlation block."""
    replacements: list[tuple[int, int, str]] = []
    metadata: dict[tuple[str, str], tuple[str | None, bool]] = {}
    starts: list[int] = []
    quote: str | None = None
    escaped = False
    block_counts: dict[str, int] = {}
    for position, char in enumerate(formula):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            starts.append(position)
        elif char == ")" and starts:
            start = starts.pop()
            body = formula[start + 1 : position]
            depth = 0
            bars: list[int] = []
            body_quote: str | None = None
            body_escaped = False
            for offset, token in enumerate(body):
                if body_quote is not None:
                    if body_escaped:
                        body_escaped = False
                    elif token == "\\":
                        body_escaped = True
                    elif token == body_quote:
                        body_quote = None
                    continue
                if token in {"'", '"'}:
                    body_quote = token
                elif token == "(":
                    depth += 1
                elif token == ")":
                    depth -= 1
                elif token == "|" and depth == 0:
                    bars.append(offset)
            if not bars:
                continue
            boundaries = [-1, *bars, len(body)]
            pieces = [
                body[left + 1 : right].strip()
                for left, right in zip(boundaries, boundaries[1:])
            ]
            if len(pieces) == 2:
                lhs, group = pieces
                block_counts[group] = block_counts.get(group, 0) + 1
                ordinal = block_counts[group]
                identity = group if ordinal == 1 else f"{group}_{ordinal}"
                shared = False
            elif len(pieces) == 3:
                lhs, identity, group = pieces
                shared = bool(identity)
                if not shared:
                    identity = None
            else:
                raise ValueError(f"invalid random-effect block ({body})")
            normalized = f"({lhs} | {group})"
            replacements.append((start, position + 1, normalized))
            model = fo.model_description(normalized)
            for term in model.group_terms:
                key = (term.expr.name, term.factor.name)
                if key in metadata:
                    raise ValueError(
                        f"random coefficient {key[0]!r} for {key[1]!r} appears in multiple blocks"
                    )
                metadata[key] = (identity, shared)
    for start, end, replacement in reversed(replacements):
        formula = formula[:start] + replacement + formula[end:]
    return formula, metadata


def _group_coding(
    frame: nw.DataFrame, group: str
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Encode a grouping factor identically for effects and structural parameters.

    Parameters
    ----------
    frame : narwhals.DataFrame
        Included training rows after response-row selection.
    group : str
        Grouping column name.

    Returns
    -------
    tuple of tuple of str and tuple of int
        Sorted fitted labels and one zero-based code per included row.

    Raises
    ------
    ValueError
        If the grouping column is absent, empty, or has missing identities.
    """
    if group not in frame.columns:
        raise ValueError(f"grouping factor {group!r} names no data column")
    column = frame[group]
    if len(column.drop_nulls()) != len(column):
        raise ValueError(f"grouping factor {group!r} has missing training identities")
    levels = tuple(str(value) for value in column.unique().sort().to_list())
    # brms drop_unused_levels = FALSE: declared levels no row uses keep their
    # group effects.
    native = column.to_native()
    if not dropping_unused_levels() and isinstance(
        getattr(native, "dtype", None), pd.CategoricalDtype
    ):
        levels = tuple(str(value) for value in sorted(native.cat.categories))
    if not levels or len(set(levels)) != len(levels):
        raise ValueError(f"grouping factor {group!r} has empty or ambiguous labels")
    positions = {value: index for index, value in enumerate(levels)}
    return levels, tuple(positions[str(value)] for value in column.to_list())


def _located_design_matrices(
    authored: str,
    normalized: str,
    frame: pd.DataFrame,
    namespace: Mapping[str, object],
) -> fo.matrices.DesignMatrices:
    """Evaluate the design matrices, locating an unknown name in the formula.

    Parameters
    ----------
    authored : str
        The formula as written, quoted in the diagnostic.
    normalized : str
        The formula text handed to the formula parser.
    frame : pandas.DataFrame
        Model rows.
    namespace : Mapping[str, object]
        Functions and constants a formula call may name.

    Returns
    -------
    formulae.matrices.DesignMatrices
        Evaluated response, population, and group designs.

    Raises
    ------
    ValueError
        If the formula names a variable that is not a data column, or calls
        a function that is not a formula function; the error names it and
        lists the data columns.
    """
    try:
        return fo.design_matrices(normalized, data=frame, extra_namespace=namespace)
    except KeyError as error:
        name = error.args[0] if error.args else None
        if (
            not isinstance(name, str)
            or re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.])", normalized) is None
        ):
            raise
        called = re.search(rf"(?<![\w.]){re.escape(name)}\s*\(", normalized)
        columns = sorted(
            str(column)
            for column in frame.columns
            if str(column) not in {BRMS_INTERCEPT, _CS_ONES}
        )
        kind = "names the variable" if called is None else "calls the unknown function"
        detail = ", which is not a column of the data" if called is None else ""
        raise ValueError(
            f"formula {authored!r} {kind} {name!r}{detail}; data columns are {columns}"
        ) from error


def formula_from_data(
    formula: str,
    data: IntoDataFrame,
    *,
    weights: str | None = None,
    decision: str | None = None,
    groups: tuple[str, ...] = (),
    extra_namespace: Mapping[str, object] | None = None,
) -> Formula:
    """Build a typed `Formula` IR by lifting
    `formulae.design_matrices` over a dataframe.

    This is an adapter, not a parser: the brms-style formula syntax
    is parsed by the [`formulae`](https://bambinos.github.io/formulae/)
    library; we lift its `formulae.matrices.DesignMatrices`
    result into a typed didactic record, augmented with deterministic
    per-group level orderings and integer-code arrays derived from
    the dataframe.

    The R-style numeric transforms (``log``, ``exp``, ``sqrt``,
    ``abs``, ``sin``, ``cos``, ``tan``, ``log10``, ``log2``,
    ``log1p``, ``expm1``, ``asin``, ``acos``, ``atan``, ``sinh``,
    ``cosh``, ``tanh``) are pre-loaded into the formulae evaluation
    namespace so users coming from R / brms get the expected base
    R behaviour without explicit registration.
    ``offset(expr)`` adds an evaluated expression with coefficient one;
    ``exposure(expr)`` adds its logarithm and requires positive finite values.
    These terms work in location and auxiliary formulas. Ordinal ``cs(expr)``
    terms retain a category-specific transition axis; ``cs(1)`` uses an internal
    constant column so fresh-data evaluation preserves its row axis.
    Polynomial terms via
    ``poly(x, k)`` are orthogonal by default, matching R's
    ``stats::poly``.
    As brms's ``na_omit``, a row with a missing value in any variable the
    formula uses (response, predictors, grouping factors, and role columns)
    is removed with a warning before anything is evaluated, and
    ``modifiers.subset`` records the original positions of the rows kept.

    Parameters
    ----------
    formula : str
        Formula string in brms / lme4 syntax.
    data : IntoDataFrame
        Pandas, polars, or any other Narwhals-compatible dataframe.
    weights : str or None
        Name of a numeric, finite, nonnegative likelihood-weight column.
        Its values are retained without normalization and must align with
        every response row after formula evaluation.
    decision : str or None
        Binary decision column, numeric 0/1 or lower/upper labels. Equivalent
        to ``response | dec(column)``; repeated declarations must agree.
    groups : tuple[str, ...]
        Additional grouping columns retained for structural parameters without
        adding varying coefficients to the predictor.
    extra_namespace : Mapping[str, object], optional
        Additional names visible inside the formula's expression
        evaluation, merged on top of the R-style transforms.

    Returns
    -------
    Formula
        The formula IR: response values, one fixed column per design
        column, the random terms, and each grouping factor's levels
        and per-row level indices.

    Raises
    ------
    ValueError
        If the formula has no response on the left of ``~``, or a
        random term is invalid, an offset modifier is used inside an
        interaction or group, or weights are nonnumeric, negative, nonfinite,
        or misaligned with the evaluated response rows.
    """
    return _formula_and_design(
        formula,
        data,
        weights=weights,
        decision=decision,
        groups=groups,
        extra_namespace=extra_namespace,
    )[0]


def _formula_and_design(
    formula: str,
    data: IntoDataFrame,
    *,
    weights: str | None = None,
    decision: str | None = None,
    groups: tuple[str, ...] = (),
    extra_namespace: Mapping[str, object] | None = None,
    cmc: bool = True,
    extra_category: bool = False,
    category_coding: CategoryCoding | None = None,
) -> tuple[Formula, DesignMatrices]:
    """Evaluate a formula once and retain its fitted transformation state.

    Parameters
    ----------
    formula : str
        Regression formula.
    data : IntoDataFrame
        Training rows.
    weights : str or None
        Optional likelihood-weight column.
    decision : str or None
        Optional lower/upper decision column.
    groups : tuple[str, ...]
        Additional structural grouping columns, using the same fitted coding.
    extra_namespace : Mapping[str, object] or None
        Additional expression bindings.
    cmc : bool
        brms ``cmc``: with False, a formula without intercept is evaluated
        with treatment coding and its intercept column is dropped.
    extra_category : bool
        Whether response code zero is a category outside the ordinal
        thresholds (the hurdle of ``hurdle_cumulative``), so that ``thres(n)``
        declares the codes ``0, ..., n + 1``.
    category_coding : {"ordinal", "hurdle", "nominal"} or None
        How the family reads an integer response as categories, following
        brms (ordinal codes ``1, ..., max(y)``). None keeps the response
        values as they are.

    Returns
    -------
    tuple[Formula, DesignMatrices]
        The immutable source IR and the evaluated formulae design.

    Raises
    ------
    ValueError
        If modifiers, weights, or evaluated response rows are invalid.
    """
    nw_df = nw.from_native(data, eager_only=True)
    pandas_df = nw_df.to_pandas()
    from quivers.formulas._response_roles import (
        bind_roles,
        declared_categories,
        parse_roles,
        resolved_threshold_groups,
        validate_event_response,
    )

    normalized_formula, names = parse_roles(
        formula, {"weights": weights, "decision": decision}
    )
    # R operators: ^ in I(), (a + b)^k, %in%, and R's scale.
    normalized_formula = normalize_r_syntax(normalized_formula)
    # brms na_omit: a row missing any model variable is removed before
    # anything is built, and its original position is retained.
    original_rows = len(pandas_df)
    retained = complete_rows(
        pandas_df,
        (normalized_formula,),
        (*role_columns(names), *groups),
    )
    if retained is not None:
        # A column holding missing values recovers its observed type.
        pandas_df = pandas_df.iloc[list(retained)].infer_objects()
    pandas_df, bound_roles = bind_roles(pandas_df, names)
    if retained is not None:
        selected = bound_roles.get("subset")
        positions = compose_positions(
            retained,
            selected.positions if selected is not None else None,
            len(pandas_df),
        )
        assert positions is not None
        bound_roles["subset"] = RowSelection(size=original_rows, positions=positions)
    response_modifiers = ResponseModifiers(**bound_roles)
    nw_df = nw.from_native(pandas_df, eager_only=True)
    namespace: dict[str, object] = dict(_R_TRANSFORMS)
    if extra_namespace:
        if {
            "offset",
            "exposure",
            "cs",
            "cbind",
            "c",
            "list",
            "mo",
            "mmc",
            "gp",
            *_SMOOTH_NAMES,
        } & extra_namespace.keys():
            raise ValueError(
                "offset, exposure, cs, cbind, c, list, mo, mmc, gp, s, t2, te, and ti are reserved formula modifiers"
            )
        namespace.update(extra_namespace)
    namespace["exposure"] = _exposure
    namespace.update(smooth_namespace())
    namespace.update(r_transform_namespace())
    namespace.update(gp_namespace())
    # R's logical literals and NA for brms term options such as iso = FALSE.
    namespace.update(R_LITERALS)
    namespace["c"] = _r_vector
    # R's list() for a smooth's xt argument, whose entries name data2 objects
    # (brms evaluates them in the data environment).
    namespace["list"] = r_list
    for data2_name, data2_object in formula_data2().items():
        if data2_name.isidentifier():
            namespace.setdefault(data2_name, data2_object)
    namespace["mo"] = Monotonic
    namespace["mmc"] = lambda *columns: np.column_stack(columns)
    namespace["cbind"] = lambda *columns: np.column_stack(columns)
    cs_normalized = _normalize_cs_intercept(normalized_formula)
    if _CS_ONES in pandas_df.columns:
        raise ValueError(f"{_CS_ONES!r} is reserved for category-specific intercepts")
    if cs_normalized != normalized_formula:
        pandas_df = pandas_df.copy()
        pandas_df[_CS_ONES] = 1.0
    # brms 0 + Intercept: the Intercept variable is a column of ones.
    intercept_normalized = normalize_intercept_term(cs_normalized)
    if intercept_normalized != cs_normalized:
        if BRMS_INTERCEPT in pandas_df.columns:
            raise ValueError(f"{BRMS_INTERCEPT!r} is reserved for the Intercept term")
        pandas_df = pandas_df.copy()
        pandas_df[BRMS_INTERCEPT] = 1.0
        cs_normalized = intercept_normalized
    # brms gr(), mm(), g1:g2, and g1/g2 become plain bars over derived factors.
    regrouped = rewrite_grouping(cs_normalized)
    pandas_df = add_grouping_columns(pandas_df, regrouped.specs)
    nw_df = nw.from_native(pandas_df, eager_only=True)
    member_covariates = []
    memberships = []
    normalized, random_blocks = _random_blocks(regrouped.formula)
    # Inspect only formulas containing modifiers, so ordinary formulas retain
    # their single design-matrix parse. Inspect before evaluation because a
    # nested offset object can otherwise fail inside formulae arithmetic.
    if re.search(r"\b(?:offset|exposure)\s*\(", normalized):
        description = fo.model_description(normalized)
        for term in description.common_terms:
            components = getattr(term, "components", ())
            modifiers = set().union(
                *(
                    _modifier_calls(getattr(component, "call", None))
                    for component in components
                )
            )
            root = (
                getattr(components[0], "call", None) if len(components) == 1 else None
            )
            if modifiers and not (
                root is not None
                and getattr(root, "callee", None) in {"offset", "exposure"}
                and not any(_modifier_calls(argument) for argument in root.args)
            ):
                raise ValueError(
                    "offset and exposure terms must be standalone additive terms"
                )
        for term in description.group_terms:
            components = getattr(term.expr, "components", ())
            if any(
                _modifier_calls(getattr(component, "call", None))
                for component in components
            ):
                raise ValueError(
                    "offset and exposure terms cannot be random coefficients"
                )
    dm = _located_design_matrices(formula, normalized, pandas_df, namespace)
    # brms cmc = FALSE: no intercept without cell-mean coding.
    drop_intercept = not cmc and (
        dm.common is None or "Intercept" not in dm.common.terms
    )
    if drop_intercept:
        dm = _located_design_matrices(
            formula, without_zero_intercept(normalized), pandas_df, namespace
        )
    # brms drop_unused_levels = FALSE keeps declared levels.
    if not dropping_unused_levels():
        dm = keep_unused_levels(dm, pandas_df)
    # R codes ordered factors by polynomial contrasts (contr.poly).
    dm = apply_ordered_contrasts(dm, pandas_df)
    if dm.response is None:
        raise ValueError(
            f"formula_from_data: formula {formula!r} has no response "
            f"variable on the left of `~`"
        )
    response_name = dm.response.name
    n_obs = int(pandas_df.shape[0])

    fixed_columns: list[FixedColumn] = []
    design_terms: list[GaussianDesignTerm] = []
    offsets: list[FixedColumn] = []
    smooth_ids: dict[str, tuple[tuple[str, ...], int]] = {}
    monotonic_terms: list[MonotonicTerm] = []
    if dm.common is not None:
        for term_name, term in dm.common.terms.items():
            components = getattr(term, "components", ())
            # brms mo() terms and their interactions are special-effects columns.
            if monotonic_positions(components):
                monotonic_terms.extend(monotonic_design(term_name, components))
                continue
            columns = _explode_term(term_name, term, n_obs)
            call = (
                getattr(components[0], "call", None) if len(components) == 1 else None
            )
            modifier = getattr(call, "callee", None)
            if modifier not in _SMOOTH_CALLS and any(
                _modifier_calls(getattr(component, "call", None), _SMOOTH_CALLS)
                for component in components
            ):
                raise ValueError(
                    "smooths must be standalone additive terms; use by= or t2() for smooth interactions"
                )
            # brms gp(): one Gaussian design block per by contrast.
            if modifier != "gp" and any(
                _modifier_calls(getattr(component, "call", None), frozenset({"gp"}))
                for component in components
            ):
                raise ValueError(
                    "gp() terms must be standalone additive terms; use by= for interactions"
                )
            if modifier == "gp":
                transform = getattr(call, "stateful_transform", None)
                if (
                    not isinstance(transform, GaussianProcess)
                    or transform.state is None
                ):
                    raise ValueError("gp() terms require the retained basis evaluator")
                design_terms.extend(_gp_design(term_name, transform, columns))
            elif modifier in _SMOOTH_CALLS:
                transform = getattr(call, "stateful_transform", None)
                if (
                    not isinstance(transform, SmoothTransform)
                    or transform.state is None
                ):
                    raise ValueError(
                        "smooth terms require the retained spline evaluator"
                    )
                identity = transform.state.identity
                if identity is not None:
                    shape = (transform.state.spec, len(transform.state.blocks))
                    if smooth_ids.setdefault(identity, shape) != shape:
                        raise ValueError(
                            f"smooths sharing id={identity!r} must use the same basis, dimension, order, and by structure"
                        )
                smooth_fixed, smooth_terms = _smooth_design(
                    term_name, transform.state, columns
                )
                fixed_columns.extend(smooth_fixed)
                design_terms.extend(smooth_terms)
            elif modifier in {"offset", "exposure"}:
                if len(columns) != 1 or columns[0].data.shape != (n_obs,):
                    raise ValueError(
                        "offset and exposure terms must produce one value per response row"
                    )
                if not np.isfinite(columns[0].data).all():
                    raise ValueError("offset terms require finite numeric values")
                offsets.extend(columns)
            else:
                fixed_columns.extend(columns)

    if drop_intercept:
        fixed_columns = [column for column in fixed_columns if not column.is_intercept]
    fixed_columns = _distinct_columns(fixed_columns)
    random_terms: list[RandomTerm] = []
    group_levels: dict[str, tuple[str, ...]] = {}
    group_indices: dict[str, tuple[int, ...]] = {}
    if dm.group is not None:
        for term in dm.group.terms.values():
            if any(
                _modifier_calls(getattr(component, "call", None), _SMOOTH_CALLS)
                for component in getattr(term.expr, "components", ())
            ):
                raise ValueError(
                    "random smooths require a structured smooth/group design, not ordinary random coefficients"
                )
            if any(
                _modifier_calls(getattr(component, "call", None), frozenset({"gp"}))
                for component in getattr(term.expr, "components", ())
            ):
                raise ValueError("gp() terms cannot be group-level coefficients")
            group = term.factor.name
            identity, shared = random_blocks.get((term.expr.name, group), (None, False))
            # A group-level mo() slope varies the matching population column.
            slopes = (
                monotonic_slopes(monotonic_terms, term.expr.name, group)
                if monotonic_positions(getattr(term.expr, "components", ()))
                else ()
            )
            random_terms.extend(
                RandomTerm(
                    slope=slope, group=group, correlation_id=identity, shared=shared
                )
                for slope in slopes
            )
            # mm(): the coefficient enters through every member's covariate.
            spec = next((item for item in regrouped.specs if item.name == group), None)
            membership = (
                spec if spec is not None and spec.kind == "membership" else None
            )
            per_member = is_member_call(getattr(term.expr, "components", ()))
            if per_member and membership is None:
                raise ValueError("mmc() requires an mm() grouping factor")
            if membership is not None:
                # A monotonic slope's value is a program binding every member
                # shares, so its member covariate is the weight alone.
                member_covariates.extend(
                    member_covariate(
                        group, membership, slope, "", np.ones(n_obs), per_member=False
                    )
                    for slope in slopes
                )
            if per_member and membership is not None:
                random_terms.append(
                    RandomTerm(
                        slope=term.expr.name,
                        group=group,
                        correlation_id=identity,
                        shared=shared,
                    )
                )
                member_covariates.append(
                    member_covariate(
                        group,
                        membership,
                        term.expr.name,
                        term.expr.name,
                        term.expr.data,
                        per_member=True,
                    )
                )
            for column in (
                ()
                if slopes or per_member
                else _explode_term(term.expr.name, term.expr, n_obs)
            ):
                random_terms.append(
                    RandomTerm(
                        slope="Intercept" if column.is_intercept else column.name,
                        group=group,
                        correlation_id=identity,
                        shared=shared,
                        column=column,
                    )
                )
                if membership is not None:
                    member_covariates.append(
                        member_covariate(
                            group,
                            membership,
                            "Intercept" if column.is_intercept else column.name,
                            "" if column.is_intercept else column.term,
                            np.ones(n_obs) if column.is_intercept else column.data,
                            per_member=False,
                        )
                    )
            if group not in group_levels and membership is not None:
                group_levels[group], fitted = fit_membership(pandas_df, membership)
                group_indices[group] = fitted.codes[0]
                memberships.append(fitted)
            if group not in group_levels:
                group_levels[group], group_indices[group] = _group_coding(nw_df, group)
    if not isinstance(groups, tuple) or any(
        not isinstance(group, str) or not group for group in groups
    ):
        raise ValueError("groups must be a tuple of nonempty grouping column names")
    for group in groups:
        if group not in group_levels:
            group_levels[group], group_indices[group] = _group_coding(nw_df, group)

    response_values = np.asarray(dm.response.design_matrix)
    components = dm.response.term.term.components
    response_call = (
        getattr(components[0], "call", None) if len(components) == 1 else None
    )
    vector_response = getattr(response_call, "callee", None) == "cbind"
    category_levels: tuple[str, ...] = ()
    ordered_response = False
    if not vector_response and getattr(dm.response, "kind", None) == "categoric":
        levels = dm.response.levels
        if levels is None:
            raise ValueError("categorical responses require retained level labels")
        category_levels = tuple(str(level) for level in levels)
        original = getattr(components[0], "_intermediate_data", None)
        ordered_response = bool(
            getattr(getattr(original, "dtype", None), "ordered", False)
        )
    if vector_response:
        assert response_call is not None
        arguments = response_call.args
        if arguments is None:
            raise ValueError("cbind responses require named numeric components")
        labels = tuple(str(argument) for argument in arguments)
    else:
        labels = ()
    from quivers.formulas._response import normalize_response_values

    response_values = normalize_response_values(
        response_values, component_names=labels, category_levels=category_levels
    )
    # brms integer categories: ordinal codes 1, ..., max(y) become positions.
    category_values: tuple[int, ...] = ()
    if (
        category_coding is not None
        and not category_levels
        and not labels
        # A refit reads the response as its original fit did.
        and not (category_coding != "hurdle" and is_zero_coded(response_name))
    ):
        category_values = integer_category_values(
            response_values,
            category_coding,
            None
            if response_modifiers.threshold_groups is not None
            else response_modifiers.thresholds,
        )
    if category_values:
        response_values = category_positions(response_values, category_values)
        if (
            response_modifiers.thresholds is None
            and response_modifiers.threshold_groups is None
        ):
            category_levels = tuple(str(value) for value in category_values)
            ordered_response = category_coding != "nominal"
    integer_carrier = response_values.dtype.kind in "iu"
    response_layout = ResponseLayout(
        component_names=labels,
        carrier="integer" if integer_carrier else "real",
        category_levels=category_levels,
        ordered=ordered_response,
    )
    # brms thres(n, gr): each group owns its threshold count.
    if response_modifiers.threshold_groups is not None:
        resolved = resolved_threshold_groups(
            response_values,
            response_modifiers.threshold_groups.with_(extra_category=extra_category),
        )
        response_modifiers = response_modifiers.with_(
            threshold_groups=resolved, thresholds=max(resolved.counts)
        )
    # brms thres()/cat(): declared ordinal categories beyond those observed.
    if response_modifiers.thresholds is not None:
        category_levels, ordered_response = declared_categories(
            response_values,
            category_levels,
            response_modifiers.thresholds + int(extra_category),
        )
        response_layout = response_layout.with_(
            category_levels=category_levels, ordered=ordered_response
        )
    if category_values:
        # Categories carry their authored integer values as labels.
        first = category_values[0]
        category_values = tuple(
            range(first, first + len(response_layout.category_levels))
            if category_coding != "nominal"
            else category_values
        )
        response_layout = response_layout.with_(
            category_levels=tuple(str(value) for value in category_values),
            category_values=category_values,
        )
    validate_event_response(
        response_values, response_modifiers.censoring, response_modifiers.truncation
    )
    if any(
        column.data.shape != response_values.shape[:1]
        for _, column in response_modifiers.columns()
    ):
        raise ValueError(
            "response modifiers must align with every evaluated response row; missing model values cannot drop rows with weights or decisions"
        )

    return Formula(
        formula=formula,
        response_name=response_name,
        fixed_columns=tuple(fixed_columns),
        design_terms=tuple(design_terms),
        offsets=tuple(offsets),
        modifiers=response_modifiers,
        random_terms=tuple(random_terms),
        response_values=response_values,
        response_layout=response_layout,
        group_levels=group_levels,
        group_indices=group_indices,
        monotonic_terms=tuple(monotonic_terms),
        grouping=GroupingPlan(
            specs=regrouped.specs,
            covariances=regrouped.covariances,
            memberships=tuple(memberships),
            covariates=tuple(member_covariates),
            options=fit_group_options(
                regrouped.options,
                pandas_df,
                group_levels,
                group_indices,
                memberships,
            ),
        ),
    ), dm


__all__ = [
    "FixedColumn",
    "GaussianDesignTerm",
    "SplineBasis",
    "RandomTerm",
    "Formula",
    "ResponseLayout",
    "ResponseModifiers",
    "CensoringSpec",
    "TruncationSpec",
    "StandardErrorSpec",
    "RowSelection",
    "FormulaData",
    "formula_from_data",
]
