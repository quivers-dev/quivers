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

from typing import Literal, Mapping
import io
import re
import tokenize

import didactic.api as dx
import formulae as fo
from formulae.matrices import DesignMatrices
import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_VALUE, ValueModel, opaque_by_value
from quivers.formulas._design import SymbolicColumn
from quivers.formulas._group_layout import GroupBlockLayout
from quivers.formulas._priors import PriorTarget
from quivers.formulas._row_retention import (
    complete_rows,
    compose_positions,
    role_columns,
)
from quivers.formulas._smooth import PSpline


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
    constraint : tuple[float, ...]
        Original training-basis mean defining the identification constraint.
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
        if (
            self.data.shape[1] < 1
            or len(self.column_indices) != self.data.shape[1]
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
        Whether to include the estimated residual scale in quadrature.
    """

    column: FixedColumn = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    sigma: bool = True


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
    """

    component_names: tuple[str, ...] = ()
    carrier: Literal["real", "integer"] = "real"
    category_levels: tuple[str, ...] = ()
    ordered: bool = False

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


class FormulaData(ValueModel):
    """Store the nonstructural complement of a `Formula` lens mapping.

    The emitted QVR module records formula structure under normalized
    identifiers. This complement retains the source formula, original names,
    presentation labels, per-row arrays, group levels, and group indices so
    ``backward(module, complement)`` can reconstruct the original `Formula`.
    Complements compare by value over every field, the arrays (stored
    as read-only copies made at construction) included.

    Attributes
    ----------
    formula : str
        Original formula string.
    response_name : str
        Original response column name.
    response_values : np.ndarray
        Response values with shape ``(N, *response_layout.event_shape)``; read-only.
    response_layout : ResponseLayout
        Event coordinates and evaluated numeric carrier.
    fixed_column_names : Mapping[str, tuple[str, str]]
        ``(term, name)`` pairs keyed by normalized fixed-column name; a
        read-only mapping.
    fixed_column_data : Mapping[str, np.ndarray]
        Predictor values keyed by normalized fixed-column name; a
        read-only mapping of read-only arrays.
    category_specific_columns : tuple[str, ...]
        Normalized fixed-column names carrying ordinal transition coefficients.
    fixed_column_order : tuple[str, ...]
        Authored fixed-column order, retained across staged predictor emission.
    design_terms : tuple[GaussianDesignTerm, ...]
        Retained coupled Gaussian designs, preserving source-lens value equality.
    group_original_names : dict[str, str]
        Original group names keyed by normalized group name.
    group_levels : dict[str, tuple[str, ...]]
        Canonical level order for each group.
    modifiers : ResponseModifiers
        Checked response-row weights and decision columns, independent of
        predictor coefficients.
    offsets : tuple[FixedColumn, ...]
        Evaluated ``offset(expr)`` and ``exposure(expr)`` contributions on
        the linear-predictor scale. These columns receive no coefficients.
    group_indices : dict[str, tuple[int, ...]]
        Per-row integer codes for each group.
    group_layouts : tuple[GroupBlockLayout, ...]
        Immutable coordinate layouts from this forward pass, independent of
        subsequent uses of the configured lens.
    prior_targets : tuple[PriorTarget, ...]
        Immutable parameter roles, authored labels, and nominal coefficient
        axes used to resolve the matching forward pass's prior rules.
    random_terms : tuple[RandomTerm, ...]
        Original slope-column data and correlation identities, matched to
        the group/slope structure decoded from the emitted module.
    """

    formula: str = ""
    response_name: str = ""
    response_layout: ResponseLayout = dx.field(
        default_factory=ResponseLayout, opaque=True, extras=OPAQUE_BY_VALUE
    )
    response_values: np.ndarray = dx.field(
        opaque=True, extras=opaque_by_value(_read_only)
    )
    fixed_column_names: Mapping[str, tuple[str, str]] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    fixed_column_data: Mapping[str, np.ndarray] = dx.field(
        default_factory=dict,
        opaque=True,
        extras=opaque_by_value(_read_only_columns),
    )
    symbolic_columns: Mapping[str, SymbolicColumn] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    category_specific_columns: tuple[str, ...] = ()
    fixed_column_order: tuple[str, ...] = ()
    design_terms: tuple[GaussianDesignTerm, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    group_original_names: dict[str, str] = dx.field(default_factory=dict)
    group_levels: dict[str, tuple[str, ...]] = dx.field(default_factory=dict)
    group_indices: dict[str, tuple[int, ...]] = dx.field(default_factory=dict)
    group_layouts: tuple[GroupBlockLayout, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    prior_targets: tuple[PriorTarget, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    offsets: tuple[FixedColumn, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    modifiers: ResponseModifiers = dx.field(
        default_factory=ResponseModifiers, opaque=True, extras=OPAQUE_BY_VALUE
    )
    random_terms: tuple[RandomTerm, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )


def _qvr_name(raw: str) -> str:
    """Normalize a label into a legal QVR identifier."""
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in raw)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def _explode_term(term_name: str, term, n_obs: int) -> list[FixedColumn]:
    """Explode a single formulae Term into one or more FixedColumn
    entries, matching R/brms's one-coefficient-per-column convention.
    """
    data = np.asarray(term.data)
    category_specific = any(
        _contains_cs(getattr(component, "call", None))
        for component in getattr(term, "components", ())
    )
    display_name = term_name.replace(_CS_ONES, "1")
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
    if not levels or len(set(levels)) != len(levels):
        raise ValueError(f"grouping factor {group!r} has empty or ambiguous labels")
    positions = {value: index for index, value in enumerate(levels)}
    return levels, tuple(positions[str(value)] for value in column.to_list())


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
        parse_roles,
        validate_event_response,
    )

    normalized_formula, names = parse_roles(
        formula, {"weights": weights, "decision": decision}
    )
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
        if {"offset", "exposure", "cs", "cbind", "s"} & extra_namespace.keys():
            raise ValueError(
                "offset, exposure, cs, and cbind are reserved formula modifiers"
            )
        namespace.update(extra_namespace)
    namespace["exposure"] = _exposure
    namespace["s"] = PSpline
    namespace["cbind"] = lambda *columns: np.column_stack(columns)
    cs_normalized = _normalize_cs_intercept(normalized_formula)
    if _CS_ONES in pandas_df.columns:
        raise ValueError(f"{_CS_ONES!r} is reserved for category-specific intercepts")
    if cs_normalized != normalized_formula:
        pandas_df = pandas_df.copy()
        pandas_df[_CS_ONES] = 1.0
    normalized, random_blocks = _random_blocks(cs_normalized)
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
    dm = fo.design_matrices(normalized, data=pandas_df, extra_namespace=namespace)
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
    if dm.common is not None:
        for term_name, term in dm.common.terms.items():
            columns = _explode_term(term_name, term, n_obs)
            components = getattr(term, "components", ())
            modifier = (
                getattr(getattr(components[0], "call", None), "callee", None)
                if len(components) == 1
                else None
            )
            if modifier != "s" and any(
                _modifier_calls(getattr(component, "call", None), frozenset({"s"}))
                for component in components
            ):
                raise ValueError(
                    "smooths must be standalone additive terms; use a retained tensor/by design for smooth interactions"
                )
            if modifier == "s":
                assert components
                transform = components[0].call.stateful_transform
                if not isinstance(transform, PSpline):
                    raise ValueError(
                        "smooth terms require the retained spline evaluator"
                    )
                fixed_columns.extend(columns[: transform.null_width])
                selected = tuple(range(transform.null_width, len(columns)))
                assert transform.penalty is not None
                coefficient_map = tuple(
                    (*fixed, *random)
                    for fixed, random in zip(
                        transform.penalty.null_factor,
                        transform.penalty.random_factor,
                        strict=True,
                    )
                )
                design_terms.append(
                    GaussianDesignTerm(
                        term=term_name,
                        qvr_name="smooth_" + _qvr_name(term_name),
                        column_indices=selected,
                        data=np.column_stack(
                            [columns[index].data for index in selected]
                        ),
                        basis=SplineBasis(
                            knots=tuple(transform.basis._knots.tolist()),
                            degree=transform.basis._degree,
                        ),
                        coefficient_map=coefficient_map,
                        constraint=transform.penalty.constraint,
                    )
                )
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

    random_terms: list[RandomTerm] = []
    group_levels: dict[str, tuple[str, ...]] = {}
    group_indices: dict[str, tuple[int, ...]] = {}
    if dm.group is not None:
        for term in dm.group.terms.values():
            if any(
                _modifier_calls(getattr(component, "call", None), frozenset({"s"}))
                for component in getattr(term.expr, "components", ())
            ):
                raise ValueError(
                    "random smooths require a structured smooth/group design, not ordinary random coefficients"
                )
            group = term.factor.name
            identity, shared = random_blocks.get((term.expr.name, group), (None, False))
            for column in _explode_term(term.expr.name, term.expr, n_obs):
                random_terms.append(
                    RandomTerm(
                        slope="Intercept" if column.is_intercept else column.name,
                        group=group,
                        correlation_id=identity,
                        shared=shared,
                        column=column,
                    )
                )
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
    integer_carrier = response_values.dtype.kind in "iu"
    response_layout = ResponseLayout(
        component_names=labels,
        carrier="integer" if integer_carrier else "real",
        category_levels=category_levels,
        ordered=ordered_response,
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
