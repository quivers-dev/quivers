"""brms autocorrelation terms: parsing, row coding, and latent residuals.

A formula such as ``y ~ x + ar(time = t, gr = g, p = 2)`` keeps its ordinary
predictor; the term leaves the formula text and is parsed here. For a family
with natural residuals the term binds the response's residual law
(`quivers.formulas._residual_law`); for any other family it is a latent
residual term of the location predictor, and a ``car`` term is a latent
spatial field. Supported terms are ``ar``, ``ma``, ``arma`` (with ``cov``),
``cosy``, ``unstr``, ``sar``, ``fcor``, and ``car``.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
import io
import tokenize
from typing import Literal

from formulae.matrices import DesignMatrices
import narwhals as nw
from narwhals.typing import IntoDataFrame
import numpy as np
import pandas as pd

from quivers._values import ValueModel
from quivers.formulas._latent_effects import (
    LatentTerm,
    latent_car_term,
)
from quivers.formulas._residual_dependence import ResidualKind, ResidualLayout
from quivers.formulas.formula import Formula, _formula_and_design

type AutocorMatrix = np.ndarray | pd.DataFrame
"""A ``data2`` matrix: SAR weights or an FCOR covariance over response rows."""

AUTOCOR_CALLS = frozenset({"ar", "ma", "arma", "cosy", "unstr", "sar", "car", "fcor"})
"""brms autocorrelation term constructors recognized in a formula."""

_SIGNATURES: Mapping[str, tuple[str, ...]] = {
    "arma": ("time", "gr", "p", "q", "cov"),
    "ar": ("time", "gr", "p", "cov"),
    "ma": ("time", "gr", "q", "cov"),
    "cosy": ("time", "gr"),
    "unstr": ("time", "gr"),
    "sar": ("M", "type"),
    "car": ("M", "gr", "type"),
    "fcor": ("M",),
}

_R_CONSTANTS: Mapping[str, bool | None] = {
    "TRUE": True,
    "FALSE": False,
    "T": True,
    "F": False,
    "True": True,
    "False": False,
    "NA": None,
}


class AutocorTerm(ValueModel):
    """One parsed brms autocorrelation term.

    Parameters
    ----------
    kind : {"arma", "cosy", "unstr", "sar", "car", "fcor"}
        Term class; ``ar`` and ``ma`` are ARMA terms with one zero order.
    time, group : str or None
        Time-ordering and grouping columns; None uses data order and one series.
    p, q : int
        Autoregressive and moving-average orders.
    cov : bool
        Covariance formulation (brms ``cov = TRUE``).
    sar_type : {"lag", "error"} or None
        Simultaneous autoregression type.
    car_type : {"escar", "esicar", "icar", "bym2"} or None
        Conditional autoregression type.
    matrix : str or None
        Name of the ``data2`` matrix for ``sar``, ``car``, and ``fcor``.
    label : str
        Authored call text.
    """

    kind: Literal["arma", "cosy", "unstr", "sar", "car", "fcor"]
    time: str | None = None
    group: str | None = None
    p: int = 0
    q: int = 0
    cov: bool = False
    sar_type: Literal["lag", "error"] | None = None
    car_type: Literal["escar", "esicar", "icar", "bym2"] | None = None
    matrix: str | None = None
    label: str = ""

    @property
    def dimension(self) -> Literal["time", "space", "fixed"]:
        """Name brms's dimension of this term.

        Returns
        -------
        {"time", "space", "fixed"}
            Time series, spatial, or fixed-covariance term.
        """
        if self.kind in {"arma", "cosy", "unstr"}:
            return "time"
        return "space" if self.kind in {"sar", "car"} else "fixed"

    @property
    def residual_kind(self) -> ResidualKind:
        """Name the `ResidualLayout` kind scoring this term.

        Returns
        -------
        ResidualKind
            Residual layout kind.

        Raises
        ------
        ValueError
            For a CAR term, which is not a residual structure.
        """
        if self.kind == "sar":
            return "sar_lag" if self.sar_type == "lag" else "sar_error"
        if self.kind == "car":
            raise ValueError("a CAR term is a latent effect, not a residual structure")
        return self.kind


def _call_text(rhs: str, start: int) -> int:
    """Find the closing parenthesis of a call whose ``(`` is at ``start``.

    Parameters
    ----------
    rhs : str
        Formula text.
    start : int
        Index of the opening parenthesis.

    Returns
    -------
    int
        Index one past the matching closing parenthesis.

    Raises
    ------
    ValueError
        If the parentheses do not balance.
    """
    depth, quote = 0, None
    for position in range(start, len(rhs)):
        char = rhs[position]
        if quote is not None:
            quote = None if char == quote else quote
        elif char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if not depth:
                return position + 1
    raise ValueError("unbalanced parentheses in an autocorrelation term")


def _argument(node: ast.expr, name: str, label: str) -> str | int | bool | None:
    """Read one literal or bare-name argument of an autocorrelation call.

    Parameters
    ----------
    node : ast.expr
        Parsed argument.
    name : str
        Formal parameter name.
    label : str
        Authored call, for errors.

    Returns
    -------
    str or int or bool or None
        Variable or matrix name, order, flag, or type string.

    Raises
    ------
    ValueError
        If the argument is neither a name nor a literal.
    """
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return _argument(ast.Constant(value=-node.operand.value), name, label)
    if isinstance(node, ast.Name):
        if node.id in _R_CONSTANTS:
            return _R_CONSTANTS[node.id]
        return node.id
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, int, bool, float)
    ):
        value = node.value
        if isinstance(value, float) and value.is_integer() and name in {"p", "q"}:
            return int(value)
        if isinstance(value, float):
            raise ValueError(f"{label}: argument {name!r} must be a whole number")
        return value
    raise ValueError(
        f"{label}: argument {name!r} must be a column name or a literal value"
    )


def parse_autocor_call(text: str) -> AutocorTerm:
    """Parse one brms autocorrelation call with brms's argument validation.

    Parameters
    ----------
    text : str
        Call text such as ``arma(time = t, gr = g, p = 1, q = 1)``.

    Returns
    -------
    AutocorTerm
        The validated term.

    Raises
    ------
    ValueError
        If arguments are unknown, duplicated, of the wrong kind, or violate
        brms's order and covariance restrictions.
    """
    try:
        call = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as error:
        raise ValueError(f"invalid autocorrelation term {text!r}") from error
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError(f"invalid autocorrelation term {text!r}")
    callee = call.func.id
    label = text.strip()
    signature = _SIGNATURES[callee]
    if len(call.args) > len(signature):
        raise ValueError(f"{label}: too many positional arguments")
    values: dict[str, str | int | bool | None] = {}
    for name, node in zip(signature, call.args, strict=False):
        values[name] = _argument(node, name, label)
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in signature:
            raise ValueError(f"{label}: unknown argument {keyword.arg!r}")
        if keyword.arg in values:
            raise ValueError(f"{label}: argument {keyword.arg!r} given twice")
        values[keyword.arg] = _argument(keyword.value, keyword.arg, label)

    def text_value(name: str, *, required: bool = False) -> str | None:
        value = values.get(name)
        if value is None:
            if required:
                raise ValueError(f"{label}: argument {name!r} is missing")
            return None
        if not isinstance(value, str) or isinstance(value, bool):
            raise ValueError(f"{label}: argument {name!r} must name one variable")
        return value

    def order(name: str, default: int) -> int:
        value = values.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "Autoregressive order must be a non-negative integer."
                if name == "p"
                else "Moving-average order must be a non-negative integer."
            )
        return value

    if callee in {"arma", "ar", "ma"}:
        p = order("p", 1) if callee != "ma" else 0
        q = order("q", 1) if callee != "ar" else 0
        if not p + q:
            raise ValueError("At least one of 'p' and 'q' should be greater zero.")
        cov = values.get("cov", False)
        if not isinstance(cov, bool):
            raise ValueError(f"{label}: argument 'cov' must be TRUE or FALSE")
        if cov and (p > 1 or q > 1):
            raise ValueError(
                "Covariance formulation of ARMA structures is only possible for "
                "effects of maximal order one."
            )
        return AutocorTerm(
            kind="arma",
            time=text_value("time"),
            group=text_value("gr"),
            p=p,
            q=q,
            cov=cov,
            label=label,
        )
    if callee in {"cosy", "unstr"}:
        required = callee == "unstr"
        return AutocorTerm(
            kind="cosy" if callee == "cosy" else "unstr",
            time=text_value("time", required=required),
            group=text_value("gr", required=required),
            cov=True,
            label=label,
        )
    matrix = text_value("M")
    if matrix is None:
        raise ValueError(f"Argument 'M' is missing in {callee}().")
    if callee == "sar":
        kind = values.get("type", "lag")
        if kind not in {"lag", "error"}:
            raise ValueError(f"{label}: type must be 'lag' or 'error'")
        return AutocorTerm(
            kind="sar",
            matrix=matrix,
            sar_type="lag" if kind == "lag" else "error",
            cov=True,
            label=label,
        )
    if callee == "car":
        kind = values.get("type", "escar")
        car_types: Mapping[str, Literal["escar", "esicar", "icar", "bym2"]] = {
            "escar": "escar",
            "esicar": "esicar",
            "icar": "icar",
            "bym2": "bym2",
        }
        if not isinstance(kind, str) or kind not in car_types:
            raise ValueError(
                f"{label}: type must be 'escar', 'esicar', 'icar', or 'bym2'"
            )
        return AutocorTerm(
            kind="car",
            matrix=matrix,
            group=text_value("gr"),
            car_type=car_types[kind],
            label=label,
        )
    return AutocorTerm(kind="fcor", matrix=matrix, cov=True, label=label)


def split_autocor_terms(formula: str) -> tuple[str, tuple[AutocorTerm, ...]]:
    """Remove standalone autocorrelation terms from a formula's right side.

    Parameters
    ----------
    formula : str
        Authored formula.

    Returns
    -------
    tuple[str, tuple[AutocorTerm, ...]]
        The formula without autocorrelation terms (``1`` when none remain)
        and the parsed terms in authored order.

    Raises
    ------
    ValueError
        If a term is not a standalone additive term, or the combination
        violates brms's one-term-per-class rules.
    """
    if not any(name in formula for name in AUTOCOR_CALLS):
        return formula, ()
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(formula).readline)
        if token.type not in {tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER}
    ]
    tilde = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.type == tokenize.OP and token.string == "~"
        ),
        None,
    )
    if tilde is None:
        return formula, ()
    offset = len(formula.splitlines(keepends=True)[0])
    if "\n" in formula:
        raise ValueError("autocorrelation formulas must be written on one line")
    del offset
    calls: list[tuple[int, int]] = []
    depth = 0
    for index, token in enumerate(tokens):
        if token.type == tokenize.OP and token.string in "([":
            depth += 1
        elif token.type == tokenize.OP and token.string in ")]":
            depth -= 1
        if (
            index > tilde
            and token.type == tokenize.NAME
            and token.string in AUTOCOR_CALLS
            and index + 1 < len(tokens)
            and tokens[index + 1].string == "("
        ):
            previous = tokens[index - 1]
            end = _call_text(formula, tokens[index + 1].start[1])
            following = next((item for item in tokens if item.start[1] >= end), None)
            standalone = (
                depth == 0
                and (index - 1 == tilde or previous.string == "+")
                and (following is None or following.string == "+")
            )
            if not standalone:
                raise ValueError(
                    f"autocorrelation term {token.string}(...) must be a standalone "
                    "additive term of the response formula"
                )
            calls.append((token.start[1], end))
    if not calls:
        return formula, ()
    terms = tuple(parse_autocor_call(formula[start:end]) for start, end in calls)
    kinds = [term.kind for term in terms]
    if len(set(kinds)) != len(kinds):
        raise ValueError("Can only model one term per autocorrelation class.")
    if sum(term.dimension == "time" for term in terms) > 1:
        raise ValueError("Can only model one time-series term.")
    if sum(term.dimension == "space" for term in terms) > 1:
        raise ValueError("Can only model one spatial term.")
    if sum(term.cov for term in terms) > 1:
        raise ValueError("Can only model one covariance matrix of natural residuals.")
    stripped = formula
    for start, end in reversed(calls):
        left = stripped[:start].rstrip()
        right = stripped[end:].lstrip()
        if left.endswith("+"):
            left = left[:-1].rstrip()
        elif right.startswith("+"):
            right = right[1:].lstrip()
        stripped = f"{left} {right}".strip()
    lhs, _, rhs = stripped.partition("~")
    if not rhs.strip():
        stripped = f"{lhs.rstrip()} ~ 1"
    return stripped, terms


def _time_level(value: float | int | str) -> str:
    """Canonicalize one time value so integer and float spellings agree.

    Parameters
    ----------
    value : float or int or str
        Time value.

    Returns
    -------
    str
        ``repr`` of the float value for numbers, the text otherwise.
    """
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        return repr(float(value))
    return str(value)


def series_layout(
    term: AutocorTerm,
    frame: pd.DataFrame,
    *,
    student: bool,
    history: np.ndarray | None,
    time_levels: tuple[str, ...] | None,
    caller: str,
) -> tuple[ResidualLayout, tuple[str, ...]]:
    """Sort rows by group and time and code each series as brms does.

    Parameters
    ----------
    term : AutocorTerm
        Time-series term.
    frame : pandas.DataFrame
        Included rows in original order.
    student : bool
        Student innovations.
    history : np.ndarray or None
        Response history for the conditional ARMA predictor.
    time_levels : tuple[str, ...] or None
        Retained unstructured time levels, or None to derive them.
    caller : str
        Entry point named in errors.

    Returns
    -------
    tuple[ResidualLayout, tuple[str, ...]]
        Layout and the unstructured time levels.

    Raises
    ------
    ValueError
        If a time or group column is absent or missing values, time points
        repeat within a group, or prediction rows use new UNSTR time points.
    """
    rows = len(frame)
    for role, column in (("time", term.time), ("gr", term.group)):
        if column is not None:
            if column not in frame.columns:
                raise ValueError(
                    f"{caller}: {term.label} names {role} column {column!r}, which is absent"
                )
            if bool(frame[column].isna().to_numpy().any()):
                raise ValueError(
                    f"{caller}: {term.label} {role} column {column!r} has missing values"
                )
    group = (
        frame[term.group].to_numpy()
        if term.group is not None
        else np.zeros(rows, dtype=np.int64)
    )
    time = (
        frame[term.time].to_numpy()
        if term.time is not None
        else np.arange(rows, dtype=np.int64)
    )
    keys = pd.DataFrame({"group": group, "time": time})
    if keys.duplicated().any():
        raise ValueError("Time points within groups must be unique.")
    order = tuple(
        int(index)
        for index in keys.sort_values(
            ["group", "time"], kind="mergesort"
        ).index.to_numpy()
    )
    codes = pd.factorize(keys["group"].to_numpy()[list(order)])[0]
    starts = tuple(int(i) for i in np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]))
    lengths = tuple(int(b - a) for a, b in zip(starts, (*starts[1:], rows)))
    levels: tuple[str, ...] = ()
    time_index: tuple[int, ...] = ()
    if term.kind == "unstr":
        labels = [_time_level(value) for value in keys["time"]]
        derived = tuple(
            _time_level(value) for value in sorted(keys["time"].unique().tolist())
        )
        levels = derived if time_levels is None else time_levels
        if set(derived) - set(levels):
            raise ValueError("Cannot handle new time points in UNSTR models.")
        position = {level: index for index, level in enumerate(levels)}
        time_index = tuple(position[labels[row]] for row in order)
    layout = ResidualLayout(
        kind=term.residual_kind,
        rows=rows,
        order=order,
        starts=starts,
        lengths=lengths,
        time_index=time_index,
        times=len(levels),
        p=term.p,
        q=term.q,
        stationary=term.cov,
        student=student,
        history=history if term.kind == "arma" and not term.cov else None,
    )
    return layout, levels


def latent_residual_term(
    term: AutocorTerm, *, data: IntoDataFrame, caller: str
) -> LatentTerm:
    """Bind brms's latent residuals for a family without natural residuals.

    brms adds ``err`` to the linear predictor with ``err = sderr * zerr``
    and the term's structure: a conditional AR term adds
    ``sum ar_k err_(n-k)`` (MA terms need ``cov = TRUE``), and a covariance
    term correlates ``err`` within each series. The ARMA coefficients are
    bounded by ``[-1, 1]`` and ``cosy`` by ``[0, 1]``, as brms bounds them
    with latent residuals.

    Parameters
    ----------
    term : AutocorTerm
        Residual term.
    data : IntoDataFrame
        Included training rows.
    caller : str
        Entry point named in errors.

    Returns
    -------
    LatentTerm
        The latent residual term.

    Raises
    ------
    ValueError
        With brms's messages for MA terms without ``cov = TRUE``, and SAR or
        FCOR terms.
    """
    if term.kind == "sar":
        raise ValueError("SAR terms are not implemented for this family.")
    if term.kind == "fcor":
        raise ValueError("FCOR terms are not implemented for this family.")
    if term.kind == "arma" and not term.cov and term.q > 0:
        raise ValueError(
            "Please set cov = TRUE when modeling MA structures for this family."
        )
    frame = nw.from_native(data, eager_only=True).to_pandas().reset_index(drop=True)
    layout, levels = series_layout(
        term, frame, student=False, history=None, time_levels=None, caller=caller
    )
    coefficients: list[tuple[str, str, float, float]] = []
    if term.kind == "arma":
        coefficients += [(f"ar_{lag}", "ar", -1.0, 1.0) for lag in range(1, term.p + 1)]
        coefficients += [(f"ma_{lag}", "ma", -1.0, 1.0) for lag in range(1, term.q + 1)]
        if not term.cov:
            # u = (I + sum ar_k L^k) err: the moving average of iid innovations.
            layout = layout.with_(p=0, q=term.p)
    elif term.kind == "cosy":
        coefficients.append(("cosy", "cosy", 0.0, 1.0))
    return LatentTerm(
        name="err",
        kind="residual",
        group=term.group,
        layout=layout,
        coefficients=tuple(coefficients),
        codes=tuple(range(len(frame))),
        time_levels=levels,
        label=term.label,
    )


def rebound_residual(
    term: LatentTerm, binding: AutocorTerm, frame: pd.DataFrame, *, include: bool
) -> LatentTerm:
    """Bind latent residuals to fresh rows, as brms predicts them.

    Fresh rows draw new residuals from the fitted structure at each posterior
    draw (brms ``cov = TRUE``); the conditional formulation cannot, as brms
    states.

    Parameters
    ----------
    term : LatentTerm
        Fitted residuals.
    binding : AutocorTerm
        Their authored structure.
    frame : pandas.DataFrame
        Fresh rows.
    include : bool
        Include autocorrelation terms (brms ``incl_autocor``).

    Returns
    -------
    LatentTerm
        Residuals of the fresh rows.

    Raises
    ------
    ValueError
        For a conditional structure, or rows its series cannot hold.
    """
    assert term.layout is not None
    if term.layout.kind == "arma" and not term.layout.stationary and include:
        raise ValueError(
            "Cannot predict new latent residuals when using cov = FALSE in "
            "autocor terms."
        )
    rows = frame.reset_index(drop=True)
    if not include and binding.kind != "unstr" and binding.time is not None:
        # An excluded term adds nothing to the predictor (brms
        # ``incl_autocor = FALSE``), so its rows need no time order:
        # prediction grids repeat time points within a group.
        rows = rows.assign(**{binding.time: range(len(rows))})
    layout, _ = series_layout(
        binding,
        rows,
        student=False,
        history=None,
        time_levels=_unstr_levels(term),
        caller="new_data",
    )
    if binding.kind == "arma" and not binding.cov:
        layout = layout.with_(p=0, q=binding.p)
    return term.with_(layout=layout, codes=tuple(range(len(frame))))


def _unstr_levels(term: LatentTerm) -> tuple[str, ...] | None:
    """Return a latent residual term's retained UNSTR time levels.

    Parameters
    ----------
    term : LatentTerm
        Latent residual term.

    Returns
    -------
    tuple[str, ...] or None
        The levels, or None for other structures.
    """
    assert term.layout is not None
    if term.layout.kind != "unstr":
        return None
    return term.time_levels


def bind_car_terms(
    terms: tuple[AutocorTerm, ...],
    formula: Formula,
    *,
    data: IntoDataFrame,
    data2: Mapping[str, AutocorMatrix] | None,
    caller: str,
) -> Formula:
    """Add the spatial fields of one predictor to its formula.

    Parameters
    ----------
    terms : tuple[AutocorTerm, ...]
        Parsed terms of the predictor.
    formula : Formula
        Evaluated predictor.
    data : IntoDataFrame
        Included training rows.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices.
    caller : str
        Entry point named in errors.

    Returns
    -------
    Formula
        The formula with one latent term per ``car`` term.

    Raises
    ------
    ValueError
        If a matrix is missing from ``data2``.
    """
    for term in terms:
        if term.kind != "car":
            continue
        assert term.matrix is not None and term.car_type is not None
        if data2 is None or term.matrix not in data2:
            raise ValueError(f"{caller}: {term.label} requires data2[{term.matrix!r}]")
        frame = nw.from_native(data, eager_only=True).to_pandas().reset_index(drop=True)
        formula = formula.with_(
            latent_terms=(
                *formula.latent_terms,
                latent_car_term(
                    term.label, term.car_type, term.group, data2[term.matrix], frame
                ),
            )
        )
    return formula


def fields_and_design(
    text: str,
    *,
    data: IntoDataFrame,
    data2: Mapping[str, AutocorMatrix] | None,
    caller: str,
) -> tuple[Formula, DesignMatrices]:
    """Evaluate one auxiliary predictor with its structured fields.

    Parameters
    ----------
    text : str
        Formula of the predictor.
    data : IntoDataFrame
        Included training rows.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices.
    caller : str
        Entry point named in errors.

    Returns
    -------
    tuple[Formula, DesignMatrices]
        The evaluated predictor and its retained design. Residual structures
        of a predictor other than the location are latent residuals.
    """
    stripped, terms = split_autocor_terms(text)
    formula, design = _formula_and_design(stripped, data)
    residual = sorted(
        (term for term in terms if term.kind != "car"),
        key=lambda term: term.dimension == "time",
    )
    formula = formula.with_(
        formula=text,
        latent_terms=tuple(
            latent_residual_term(term, data=data, caller=caller) for term in residual
        ),
    )
    return (
        bind_car_terms(terms, formula, data=data, data2=data2, caller=caller),
        design,
    )


__all__ = ["AutocorMatrix", "AutocorTerm"]
