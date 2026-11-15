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

from typing import Mapping
import re

import didactic.api as dx
import formulae as fo
from formulae.matrices import DesignMatrices
import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_VALUE, ValueModel, opaque_by_value


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
}


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


def _offset_calls(expression: object) -> set[str]:
    """Find offset modifiers within a formulae lazy expression.

    Parameters
    ----------
    expression : object
        Parsed call, operator, variable, or literal.

    Returns
    -------
    set[str]
        Names of offset and exposure calls in the expression tree.
    """
    callee = getattr(expression, "callee", None)
    found = {callee} if callee in {"offset", "exposure"} else set()
    for argument in getattr(expression, "args", ()):
        found.update(_offset_calls(argument))
    for argument in getattr(expression, "kwargs", {}).values():
        found.update(_offset_calls(argument))
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
        ``True`` for the constant-1 column.
    data : np.ndarray
        The column's values, shape ``(N,)``; read-only.
    """

    term: str
    name: str
    qvr_name: str
    is_intercept: bool = False
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
    column: FixedColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


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
    random_terms : tuple[RandomTerm, ...]
        Random-effect group specifications.
    response_values : np.ndarray
        Response column values, shape ``(N,)``; read-only.
    group_levels : dict[str, tuple[str, ...]]
        Canonical level ordering per grouping factor, used to derive
        deterministic plate-index tensors.
    response_weights : FixedColumn or None
        Retained nonnegative finite likelihood weights, one per response row.
        Weights multiply response log densities and receive no coefficient.
    offsets : tuple[FixedColumn, ...]
        Evaluated ``offset(expr)`` and ``exposure(expr)`` contributions on
        the linear-predictor scale. These columns receive no coefficients.
    group_indices : dict[str, tuple[int, ...]]
        Per-group integer index array, shape ``(N,)``.
    """

    formula: str
    response_name: str
    fixed_columns: tuple[FixedColumn, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    random_terms: tuple[RandomTerm, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    response_values: np.ndarray = dx.field(
        opaque=True, extras=opaque_by_value(_read_only)
    )
    group_levels: dict[str, tuple[str, ...]] = dx.field(default_factory=dict)
    group_indices: dict[str, tuple[int, ...]] = dx.field(default_factory=dict)
    offsets: tuple[FixedColumn, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    response_weights: FixedColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
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
        Response values with shape ``(N,)``; read-only.
    fixed_column_names : Mapping[str, tuple[str, str]]
        ``(term, name)`` pairs keyed by normalized fixed-column name; a
        read-only mapping.
    fixed_column_data : Mapping[str, np.ndarray]
        Predictor values keyed by normalized fixed-column name; a
        read-only mapping of read-only arrays.
    group_original_names : dict[str, str]
        Original group names keyed by normalized group name.
    group_levels : dict[str, tuple[str, ...]]
        Canonical level order for each group.
    response_weights : FixedColumn or None
        Retained nonnegative finite likelihood weights, one per response row.
        Weights multiply response log densities and receive no coefficient.
    offsets : tuple[FixedColumn, ...]
        Evaluated ``offset(expr)`` and ``exposure(expr)`` contributions on
        the linear-predictor scale. These columns receive no coefficients.
    group_indices : dict[str, tuple[int, ...]]
        Per-row integer codes for each group.
    random_terms : tuple[RandomTerm, ...]
        Original slope-column data and correlation identities, matched to
        the group/slope structure decoded from the emitted module.
    """

    formula: str = ""
    response_name: str = ""
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
    group_original_names: dict[str, str] = dx.field(default_factory=dict)
    group_levels: dict[str, tuple[str, ...]] = dx.field(default_factory=dict)
    group_indices: dict[str, tuple[int, ...]] = dx.field(default_factory=dict)
    offsets: tuple[FixedColumn, ...] = dx.field(
        default_factory=tuple, opaque=True, extras=OPAQUE_BY_VALUE
    )
    response_weights: FixedColumn | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
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
    if data.ndim == 1:
        # Single column; treat as one coefficient.
        is_intercept = (
            getattr(term, "kind", None) == "intercept" or term_name == "Intercept"
        )
        return [
            FixedColumn(
                term=term_name,
                name=term_name,
                qvr_name=("intercept" if is_intercept else _qvr_name(term_name)),
                is_intercept=is_intercept,
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
                name=term_name,
                qvr_name=("intercept" if is_intercept else _qvr_name(term_name)),
                is_intercept=is_intercept,
                data=data[:, 0].astype(np.float64),
            )
        ]
    return [
        FixedColumn(
            term=term_name,
            name=f"{term_name}_{k + 1}",
            qvr_name=_qvr_name(f"{term_name}_{k + 1}"),
            is_intercept=False,
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


def formula_from_data(
    formula: str,
    data: IntoDataFrame,
    *,
    weights: str | None = None,
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
    These terms work in location and auxiliary formulas. Polynomial terms via
    ``poly(x, k)`` are orthogonal by default, matching R's
    ``stats::poly``.

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
        formula, data, weights=weights, extra_namespace=extra_namespace
    )[0]


def _formula_and_design(
    formula: str,
    data: IntoDataFrame,
    *,
    weights: str | None = None,
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
    response_weights = None
    if weights is not None:
        if not isinstance(weights, str) or weights not in pandas_df.columns:
            raise ValueError("weights must name an existing dataframe column")
        values = np.asarray(pandas_df[weights])
        if values.ndim != 1 or values.dtype.kind not in "iuf":
            raise ValueError("weights require a real numeric column")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("weights must be finite and nonnegative")
        response_weights = FixedColumn(
            term=weights,
            name=weights,
            qvr_name="formula_weights",
            data=values.astype(np.float64),
        )
    namespace: dict[str, object] = dict(_R_TRANSFORMS)
    if extra_namespace:
        if {"offset", "exposure"} & extra_namespace.keys():
            raise ValueError("offset and exposure are reserved formula modifiers")
        namespace.update(extra_namespace)
    namespace["exposure"] = _exposure
    normalized, random_blocks = _random_blocks(formula)
    # Inspect only formulas containing modifiers, so ordinary formulas retain
    # their single design-matrix parse. Inspect before evaluation because a
    # nested offset object can otherwise fail inside formulae arithmetic.
    if re.search(r"\b(?:offset|exposure)\s*\(", normalized):
        description = fo.model_description(normalized)
        for term in description.common_terms:
            components = getattr(term, "components", ())
            modifiers = set().union(
                *(
                    _offset_calls(getattr(component, "call", None))
                    for component in components
                )
            )
            root = (
                getattr(components[0], "call", None) if len(components) == 1 else None
            )
            if modifiers and not (
                root is not None
                and getattr(root, "callee", None) in {"offset", "exposure"}
                and not any(_offset_calls(argument) for argument in root.args)
            ):
                raise ValueError(
                    "offset and exposure terms must be standalone additive terms"
                )
        for term in description.group_terms:
            components = getattr(term.expr, "components", ())
            if any(
                _offset_calls(getattr(component, "call", None))
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
            if modifier in {"offset", "exposure"}:
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
                levels = tuple(
                    str(v) for v in nw_df[group].drop_nulls().unique().sort().to_list()
                )
                group_levels[group] = levels
                level_index = {v: i for i, v in enumerate(levels)}
                codes = tuple(level_index[str(v)] for v in nw_df[group].to_list())
                group_indices[group] = codes

    response_values = (
        np.asarray(dm.response.design_matrix).reshape(-1).astype(np.float64)
    )

    if (
        response_weights is not None
        and response_weights.data.shape != response_values.shape
    ):
        raise ValueError(
            "weights must align with every evaluated response row; missing model values cannot drop weighted rows"
        )

    return Formula(
        formula=formula,
        response_name=response_name,
        fixed_columns=tuple(fixed_columns),
        offsets=tuple(offsets),
        response_weights=response_weights,
        random_terms=tuple(random_terms),
        response_values=response_values,
        group_levels=group_levels,
        group_indices=group_indices,
    ), dm


__all__ = [
    "FixedColumn",
    "RandomTerm",
    "Formula",
    "FormulaData",
    "formula_from_data",
]
