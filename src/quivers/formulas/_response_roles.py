"""Checked response-role syntax and one retained dataframe binding boundary."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd
import torch
from narwhals.typing import IntoDataFrame

from quivers._observation import EVENT_OPTIONS

from quivers.formulas.formula import (
    FixedColumn,
    CensoringSpec,
    RowSelection,
    StandardErrorSpec,
    TruncationSpec,
    Formula,
    ResponseModifiers,
)

if TYPE_CHECKING:
    from quivers.formulas.family import Family

type RoleSource = str | int | float | bool


class BoundRoles(TypedDict, total=False):
    """Checked keyword fields for the single retained ResponseModifiers record."""

    weights: FixedColumn
    decision: FixedColumn
    censoring: CensoringSpec
    truncation: TruncationSpec
    standard_error: StandardErrorSpec
    subset: RowSelection
    trials: FixedColumn
    weights_scale: bool
    sources: tuple[tuple[str, RoleSource], ...]


# Positional roles and accepted keyword roles share one call signature.
_CALLS = {
    "weights": (("weights",), {"scale": "weights_scale"}),
    "dec": (("decision",), {}),
    "cens": (("censoring", "censoring_upper"), {"y2": "censoring_upper"}),
    "trunc": (
        ("trunc_lower", "trunc_upper"),
        {"lb": "trunc_lower", "ub": "trunc_upper"},
    ),
    "se": (("standard_error",), {"sigma": "se_sigma"}),
    "subset": (("subset",), {}),
    "trials": (("trials",), {}),
}
_ALIASES = {
    "weights": "formula_weights",
    "decision": "formula_decision",
    "censoring": "formula_censoring",
    "censoring_upper": "formula_censoring_upper",
    "trunc_lower": "formula_trunc_lower",
    "trunc_upper": "formula_trunc_upper",
    "standard_error": "formula_standard_error",
    "trials": "formula_trials",
}
_ROLES = frozenset(
    role
    for positional, keywords in _CALLS.values()
    for role in (*positional, *keywords.values())
)


def _source(node: ast.expr) -> RoleSource:
    """Read a column name or scalar literal without evaluating user code.

    Parameters
    ----------
    node : ast.expr
        One modifier argument.

    Returns
    -------
    str or int or float or bool
        Named column or literal scalar.

    Raises
    ------
    ValueError
        If the expression is not a column or a scalar literal.
    """
    if isinstance(node, ast.Name):
        if node.id in {"true", "false", "TRUE", "FALSE"}:
            return node.id in {"true", "TRUE"}
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float | bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        value = _source(node.operand)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return -value if isinstance(node.op, ast.USub) else value
    raise ValueError(
        "response modifier arguments require dataframe columns or scalar literals"
    )


def parse_roles(
    formula: str, explicit: Mapping[str, RoleSource | None] | None = None
) -> tuple[str, dict[str, RoleSource]]:
    """Resolve modifier calls against shared role signatures without evaluation.

    Parameters
    ----------
    formula : str
        Response and predictor formula, optionally with a response-side bar.
    explicit : Mapping or None
        Equivalent keyword declarations keyed by semantic role.

    Returns
    -------
    tuple[str, dict]
        Design formula with its original response, and resolved role sources.

    Raises
    ------
    ValueError
        If a call, argument, keyword, or duplicate role is invalid.
    """
    roles = {key: value for key, value in (explicit or {}).items() if value is not None}
    if roles.keys() - _ROLES:
        raise ValueError(
            f"unknown response modifier roles {sorted(roles.keys() - _ROLES)}"
        )
    lhs, separator, rhs = formula.partition("~")
    if not separator or "|" not in lhs:
        return formula, roles
    try:
        expression = ast.parse(lhs.strip(), mode="eval").body
    except SyntaxError as error:
        raise ValueError(
            "response modifiers require named calls after a response-side bar"
        ) from error
    if not isinstance(expression, ast.BinOp) or not isinstance(
        expression.op, ast.BitOr
    ):
        raise ValueError("response modifiers require response | modifier(...) syntax")
    pending = [expression.right]
    while pending:
        call = pending.pop()
        if isinstance(call, ast.BinOp) and isinstance(call.op, ast.Add):
            pending.extend((call.right, call.left))
            continue
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id not in _CALLS
        ):
            raise ValueError("unknown response modifier call")
        positional, keywords = _CALLS[call.func.id]
        if len(call.args) > len(positional):
            raise ValueError(
                f"{call.func.id} modifier has too many positional arguments"
            )
        arguments = dict(zip(positional, call.args, strict=False))
        for keyword in call.keywords:
            if keyword.arg not in keywords:
                raise ValueError(
                    f"{call.func.id} has an unknown keyword {keyword.arg!r}"
                )
            role = keywords[keyword.arg]
            if role in arguments:
                raise ValueError(f"{call.func.id} gives {role} twice")
            arguments[role] = keyword.value
        required = set() if call.func.id == "trunc" else {positional[0]}
        if not arguments or required - arguments.keys():
            raise ValueError(f"{call.func.id} requires its response role argument")
        for role, node in arguments.items():
            value = _source(node)
            if role in roles and (
                type(value) is not type(roles[role]) or value != roles[role]
            ):
                raise ValueError(f"conflicting {role} response modifier declarations")
            roles[role] = value
    response = ast.get_source_segment(lhs.strip(), expression.left)
    assert response is not None
    return f"{response} ~ {rhs}", roles


def _values(frame: pd.DataFrame, source: RoleSource, role: str) -> np.ndarray:
    """Read one role column or broadcast a scalar literal.

    Parameters
    ----------
    frame : pandas.DataFrame
        Included observation rows.
    source : str or scalar
        Column name or constant.
    role : str
        Semantic role used in diagnostics.

    Returns
    -------
    numpy.ndarray
        One value per row.

    Raises
    ------
    ValueError
        If a named column is absent or not one dimensional.
    """
    if isinstance(source, str):
        if source not in frame.columns:
            raise ValueError(f"{role} must name an existing dataframe column")
        values = np.asarray(frame[source])
    else:
        values = np.full(len(frame), source)
    if values.shape != (len(frame),):
        raise ValueError(f"{role} requires one value per response row")
    return values


def _numeric(values: np.ndarray, role: str, *, finite: bool = True) -> np.ndarray:
    """Validate numeric carriers before converting a retained real column.

    Parameters
    ----------
    values : numpy.ndarray
        Input values, potentially an object array with inactive missing data.
    role : str
        Named role for diagnostics.
    finite : bool
        Whether infinities are refused; NaN is always refused.

    Returns
    -------
    numpy.ndarray
        Float64 numeric values.

    Raises
    ------
    ValueError
        If a value is boolean, nonnumeric, NaN, or forbidden infinity.
    """
    if values.dtype.kind not in "iuf" and not all(
        isinstance(value, Real) and not isinstance(value, bool | np.bool_)
        for value in values
    ):
        raise ValueError(f"{role} requires real numeric values")
    try:
        result = values.astype(np.float64)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"{role} requires representable real numeric values"
        ) from error
    if np.isnan(result).any() or finite and not np.isfinite(result).all():
        raise ValueError(
            f"{role} requires {'finite ' if finite else ''}real numeric values"
        )
    return result


def bind_roles(
    frame: pd.DataFrame, sources: Mapping[str, RoleSource]
) -> tuple[pd.DataFrame, BoundRoles]:
    """Select included rows, then bind every retained observation role once.

    Parameters
    ----------
    frame : pandas.DataFrame
        Original source rows; excluded rows are never inspected for other roles.
    sources : Mapping
        Checked modifier roles from parse_roles or equivalent API keywords.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Included frame in original order and immutable typed modifier fields.

    Raises
    ------
    ValueError
        If a role carrier, domain, alignment, or role dependency is invalid.
    """
    bindings: BoundRoles = {}
    unknown = set(sources) - _ROLES
    if unknown:
        raise ValueError(f"unknown response modifier roles {sorted(unknown)}")
    if "subset" in sources:
        mask = _values(frame, sources["subset"], "subset")
        if mask.dtype.kind != "b":
            raise ValueError("subset requires Boolean values")
        positions = tuple(int(index) for index in np.flatnonzero(mask))
        if not positions:
            raise ValueError("subset selects no response rows")
        bindings["subset"] = RowSelection(size=len(frame), positions=positions)
        frame = frame.iloc[list(positions)].copy(deep=False)

    def column(role: str, values: np.ndarray | None = None) -> FixedColumn:
        """Retain a role column under its compiler-owned input alias.

        Parameters
        ----------
        role : str
            Canonical data role.
        values : numpy.ndarray or None
            Validated values, or the named input before role validation.

        Returns
        -------
        FixedColumn
            Immutable copied column with its original source label.
        """
        source = sources[role]
        return FixedColumn(
            term=str(source),
            name=str(source),
            qvr_name=_ALIASES[role],
            data=_values(frame, source, role) if values is None else values,
        )

    for role in ("weights", "standard_error"):
        if role not in sources:
            continue
        values = _numeric(_values(frame, sources[role], role), role)
        if np.any(values < 0):
            raise ValueError(f"{role} values must be nonnegative")
        if role == "weights":
            scale = sources.get("weights_scale", False)
            if type(scale) is not bool:
                raise ValueError("weights scale must be a Boolean literal")
            if scale:
                maximum = values.max(initial=0.0)
                if maximum == 0:
                    raise ValueError("scaled weights require positive total weight")
                relative = values / maximum
                values = len(values) * relative / relative.sum()
                bindings["weights_scale"] = True
            bindings["weights"] = column(role, values)
        else:
            sigma = sources.get("se_sigma", True)
            if type(sigma) is not bool:
                raise ValueError("se sigma must be a Boolean literal")
            if not sigma and np.any(values == 0):
                raise ValueError(
                    "se-only observations require positive standard errors"
                )
            bindings["standard_error"] = StandardErrorSpec(
                column=column(role, values), sigma=sigma
            )
    if "se_sigma" in sources and "standard_error" not in sources:
        raise ValueError("se sigma requires standard errors")
    if "weights_scale" in sources and "weights" not in sources:
        raise ValueError("weights scale requires weights")
    if "decision" in sources:
        values = _values(frame, sources["decision"], "decision")
        if values.dtype.kind in "iuf":
            if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
                raise ValueError("decisions must be exactly zero/lower or one/upper")
            values = values.astype(np.int64)
        elif all(
            isinstance(value, str) and value in {"lower", "upper"} for value in values
        ):
            values = np.asarray(
                [int(value == "upper") for value in values], dtype=np.int64
            )
        else:
            raise ValueError("decisions require numeric 0/1 or lower/upper labels")
        bindings["decision"] = column("decision", values)
    if "censoring" in sources:
        values = _values(frame, sources["censoring"], "censoring")
        if values.dtype.kind == "b":
            status = values.astype(np.int64)
        elif values.dtype.kind in "iuf":
            if (
                not np.isfinite(values).all()
                or not np.isin(values, (-1, 0, 1, 2)).all()
            ):
                raise ValueError("censoring requires codes -1, 0, 1, or 2")
            status = values.astype(np.int64)
        else:
            labels = {"left": -1, "none": 0, "exact": 0, "right": 1, "interval": 2}
            if not all(isinstance(value, str) and value in labels for value in values):
                raise ValueError(
                    "censoring requires left/none/right/interval labels or integer codes"
                )
            status = np.asarray([labels[value] for value in values], dtype=np.int64)
        interval = status == 2
        upper = None
        if interval.any() and "censoring_upper" not in sources:
            raise ValueError("interval censoring requires an upper endpoint column")
        if "censoring_upper" in sources:
            raw = _values(frame, sources["censoring_upper"], "censoring_upper")
            values = np.zeros(len(frame), dtype=np.float64)
            values[interval] = _numeric(
                raw[interval], "interval censoring upper endpoint"
            )
            upper = column("censoring_upper", values)
        bindings["censoring"] = CensoringSpec(
            status=column("censoring", status), upper=upper
        )
    elif "censoring_upper" in sources:
        raise ValueError("censoring upper endpoint requires a censoring status")
    bounds: dict[str, FixedColumn] = {}
    for role in ("trunc_lower", "trunc_upper"):
        if role in sources:
            bounds[role] = column(
                role, _numeric(_values(frame, sources[role], role), role, finite=False)
            )
    if bounds:
        lower = bounds["trunc_lower"].data if "trunc_lower" in bounds else -np.inf
        upper = bounds["trunc_upper"].data if "trunc_upper" in bounds else np.inf
        if np.any(lower > upper):
            raise ValueError("truncation lower bounds must not exceed upper bounds")
        bindings["truncation"] = TruncationSpec(
            lower=bounds.get("trunc_lower"), upper=bounds.get("trunc_upper")
        )
    if "trials" in sources:
        values = _values(frame, sources["trials"], "trials")
        if (
            values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values > 2**53)
            or np.any(values != np.floor(values))
        ):
            raise ValueError("trials require nonnegative integers at most 2**53")
        bindings["trials"] = column("trials", values.astype(np.int64))
    bindings["sources"] = tuple(sorted(sources.items()))
    return frame, bindings


def validate_event_response(
    response: np.ndarray,
    censoring: CensoringSpec | None,
    truncation: TruncationSpec | None = None,
) -> None:
    """Validate recorded endpoints and exact outcomes after response evaluation.

    Parameters
    ----------
    response : numpy.ndarray
        Evaluated scalar response or first censoring endpoint.
    censoring : CensoringSpec or None
        Retained status and second endpoints on the included rows.
    truncation : TruncationSpec or None
        Inclusive retained sampling bounds. Exact outcomes must lie inside;
        censoring endpoints can lie outside the sampling support.

    Raises
    ------
    ValueError
        If censoring metadata is misaligned, the response is not scalar, or an
        active interval endpoint does not exceed its first endpoint, or an
        exact outcome lies outside its truncation interval.
    """
    if censoring is None and truncation is None:
        return
    values = np.asarray(response)
    if values.ndim != 1:
        raise ValueError("recorded events require one scalar response per included row")
    if truncation is not None:
        exact = (
            np.ones(values.shape, dtype=bool)
            if censoring is None
            else censoring.status.data == 0
        )
        for bound, lower in ((truncation.lower, True), (truncation.upper, False)):
            if bound is None:
                continue
            if bound.data.shape != values.shape:
                raise ValueError(
                    "truncation bounds must align with included response rows"
                )
            outside = values < bound.data if lower else values > bound.data
            if np.any(exact & outside):
                raise ValueError(
                    "exact responses must lie within their truncation bounds"
                )
    if censoring is None:
        return
    if censoring.status.data.shape != values.shape:
        raise ValueError("censoring requires one scalar response per included row")
    interval = censoring.status.data == 2
    if interval.any():
        if censoring.upper is None or censoring.upper.data.shape != values.shape:
            raise ValueError("interval censoring requires an aligned upper endpoint")
        if not np.all(censoring.upper.data[interval] > values[interval]):
            raise ValueError(
                "interval censoring upper endpoints must exceed response endpoints"
            )


def included_data(formula: Formula, data: IntoDataFrame) -> IntoDataFrame:
    """Apply retained row positions to auxiliary designs and other row inputs.

    Parameters
    ----------
    formula : Formula
        Primary formula containing the retained selection.
    data : IntoDataFrame
        Original dataframe, before row selection.

    Returns
    -------
    IntoDataFrame
        Included rows in their original order, or the unchanged input.

    Raises
    ------
    ValueError
        If input rows no longer align with the retained selection.
    """
    import narwhals as nw

    selection = formula.modifiers.subset
    if selection is None:
        return data
    frame = nw.from_native(data, eager_only=True).to_pandas()
    if len(frame) != selection.size:
        raise ValueError("subset row positions must align with the original dataframe")
    return frame.iloc[list(selection.positions)].copy(deep=False)


def event_option_columns(
    modifiers: ResponseModifiers,
) -> tuple[tuple[str, FixedColumn], ...]:
    """Return retained event roles under their existing source option names.

    Parameters
    ----------
    modifiers : ResponseModifiers
        Checked response inputs.

    Returns
    -------
    tuple[tuple[str, FixedColumn], ...]
        Supplied columns in the common event vocabulary's canonical order.
    """
    columns = dict(modifiers.columns())
    roles = {
        "status": "censoring",
        "upper": "censoring_upper",
        "lower_bound": "trunc_lower",
        "upper_bound": "trunc_upper",
    }
    return tuple(
        (option, columns[roles[field]])
        for option, (field, _) in EVENT_OPTIONS.items()
        if roles[field] in columns
    )


def prediction_role_bindings(
    modifiers: ResponseModifiers, frame: pd.DataFrame
) -> dict[str, torch.Tensor]:
    """Bind sampling-law roles on fresh rows without requiring recorded data.

    Parameters
    ----------
    modifiers : ResponseModifiers
        Retained authored input sources and policies.
    frame : pandas.DataFrame
        Fresh predictor rows; recorded responses and censoring are optional.

    Returns
    -------
    dict[str, torch.Tensor]
        Checked SE and truncation bindings, with exact-event placeholders.

    Raises
    ------
    ValueError
        If a required fresh sampling-law input is missing or invalid.
    """
    required = {
        role: source
        for role, source in modifiers.sources
        if role in {"trunc_lower", "trunc_upper", "standard_error", "se_sigma"}
    }
    _, bound = bind_roles(frame, required)
    columns = []
    if isinstance(error := bound.get("standard_error"), StandardErrorSpec):
        columns.append(error.column)
    if isinstance(truncation := bound.get("truncation"), TruncationSpec):
        columns.extend(
            column
            for column in (truncation.lower, truncation.upper)
            if column is not None
        )
    bindings = {
        column.qvr_name: torch.as_tensor(column.data.copy()) for column in columns
    }
    if modifiers.censoring is not None:
        bindings[modifiers.censoring.status.qvr_name] = torch.zeros(
            len(frame), dtype=torch.long
        )
        if modifiers.censoring.upper is not None:
            bindings[modifiers.censoring.upper.qvr_name] = torch.zeros(
                len(frame), dtype=torch.float64
            )
    return bindings


def retained_trials(
    formula: Formula,
    family: Family,
    data: IntoDataFrame,
    specification: int | str | None,
) -> int | str | None:
    """Resolve a trials modifier against the existing ergonomic keyword API.

    Parameters
    ----------
    formula : Formula
        Parsed formula and checked trial rows.
    family : Family
        Response metadata declaring whether trials are meaningful.
    data : IntoDataFrame
        Included rows used to resolve an optional keyword declaration.
    specification : int or str or None
        Existing constant or dataframe-column keyword API.

    Returns
    -------
    int or str or None
        Existing resolved specification, or the common retained trials alias.

    Raises
    ------
    ValueError
        If the family has no trials or declarations disagree.
    """
    from quivers.formulas._response import resolve_trials

    if formula.modifiers.trials is None:
        return resolve_trials(family, specification)
    if family.trials_parameter is None:
        raise ValueError("trials requires a response family with finite trial counts")
    if specification is not None:
        from quivers.formulas._fit import _binomial_trial_tensor

        expected = _binomial_trial_tensor(
            data, specification, n_obs=len(formula.response_values)
        ).numpy()
        if not np.array_equal(expected, formula.modifiers.trials.data):
            raise ValueError(
                "conflicting trials modifier and binomial_trials declarations"
            )
    return "formula_trials"
