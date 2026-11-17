"""One decision about which data rows a formula model uses.

brms evaluates every variable a model uses in one ``model.frame`` and removes
each row with a missing value in any of them (``na_omit`` in
``R/data-helpers.R``): the response, every predictor of every distributional
and nonlinear parameter, grouping factors, and the variables of addition
terms such as ``weights``, ``trials``, ``cens``, ``subset``, and of special
terms. Only variables handled by ``mi()`` keep their missing values. All of
these are removed together, before any design matrix, group coding, or
response array is built, so every part of the model sees the same rows.

`complete_rows` makes that decision once from the authored formula texts and
role columns. The retained positions index the caller's original frame; the
formula records them as its row selection, so predictions, pointwise
likelihoods, and refit partitions map back to the input rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Protocol
import warnings

import numpy as np
import pandas as pd

__all__: list[str] = []


_NAME = re.compile(r"`[^`]*`|(?:[A-Za-z]|\.(?![0-9]))[A-Za-z0-9._]*")


#: Arguments whose values apply only to some rows: the upper endpoint of
#: ``cens(status, y2)`` is read on interval-censored rows only.
_CONDITIONAL = {"cens": (1, "y2")}


def _scan(text: str) -> tuple[list[str], list[str]]:
    """Collect variable names that every row needs and names that may be missing.

    Names inside ``mi()`` may be missing, and so may a conditionally
    applicable argument such as the interval upper endpoint of ``cens()``,
    whose rows the response roles validate themselves.

    Parameters
    ----------
    text : str
        Formula text.

    Returns
    -------
    tuple[list[str], list[str]]
        Required names and names that may be missing, in order.
    """
    required: list[str] = []
    optional: list[str] = []
    # One frame per open call: its name, argument position, and keyword.
    stack: list[list[str | int]] = []
    pending = ""
    position = 0
    quote: str | None = None
    while position < len(text):
        char = text[position]
        if quote is not None:
            quote = None if char == quote else quote
            position += 1
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            stack.append([pending, 0, ""])
        elif char == ")" and stack:
            stack.pop()
        elif char == "," and stack:
            stack[-1][1] = int(stack[-1][1]) + 1
            stack[-1][2] = ""
        match = _NAME.match(text, position)
        if match is None or char in {"(", ")", ","}:
            pending = ""
            position += 1
            continue
        name = match.group(0).strip("`")
        position = match.end()
        rest = text[position:].lstrip()
        pending = ""
        if rest.startswith("(") and not match.group(0).startswith("`"):
            pending = name
        elif rest.startswith("=") and not rest.startswith("=="):
            if stack:
                stack[-1][2] = name
            continue
        elif any(frame[0] == "mi" for frame in stack) or _conditional(stack):
            optional.append(name)
        else:
            required.append(name)
    return required, optional


def _conditional(stack: list[list[str | int]]) -> bool:
    """Report whether the innermost call argument applies to some rows only.

    Parameters
    ----------
    stack : list[list[str | int]]
        Open calls with their argument position and keyword.

    Returns
    -------
    bool
        Whether the current argument is conditionally applicable.
    """
    if not stack:
        return False
    name, position, keyword = stack[-1]
    rule = _CONDITIONAL.get(str(name))
    if rule is None:
        return False
    return keyword == rule[1] or (keyword == "" and position == rule[0])


def formula_variables(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Find the variable names an R-style formula uses, as R's ``all.vars``.

    A name followed by ``(`` is a function and a name followed by a single
    ``=`` is an argument name, so neither is a variable. Quoted strings are
    literals. A name inside a ``mi()`` call may be missing, and so may the
    response of a formula whose left side carries ``mi()``.

    Parameters
    ----------
    text : str
        One formula or right-hand side.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        Names that must be observed, and names ``mi()`` allows to be missing.
    """
    left, tilde, right = text.partition("~")
    if not tilde:
        left, right = "", text
    required, optional = _scan(right)
    left_required, left_optional = _scan(left)
    optional += left_optional
    if left_required and re.search(r"\bmi\s*\(", left):
        optional.append(left_required.pop(0))
    required = left_required + required
    return tuple(dict.fromkeys(required)), tuple(dict.fromkeys(optional))


def _variables(
    frame: pd.DataFrame, texts: Iterable[str], columns: Iterable[str]
) -> list[str]:
    """List the frame columns a model part requires to be observed.

    Parameters
    ----------
    frame : pandas.DataFrame
        The caller's rows.
    texts : Iterable[str]
        Formula texts of the part.
    columns : Iterable[str]
        Further role columns of the part.

    Returns
    -------
    list[str]
        Required frame columns, without those ``mi()`` handles.
    """
    required: dict[str, None] = {}
    optional: set[str] = set()
    for text in texts:
        found, missing_allowed = formula_variables(text)
        required.update(dict.fromkeys(found))
        optional.update(missing_allowed)
    required.update(dict.fromkeys(columns))
    return [name for name in required if name in frame.columns and name not in optional]


def _missing(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Flag the rows where one column is missing.

    Parameters
    ----------
    frame : pandas.DataFrame
        The caller's rows.
    name : str
        Column name.

    Returns
    -------
    numpy.ndarray
        One Boolean per row.
    """
    flags = np.asarray(pd.isna(frame[name]), dtype=bool).reshape(len(frame), -1)
    return np.asarray(flags.any(axis=1), dtype=bool)


def _decide(
    frame: pd.DataFrame, omit: np.ndarray, names: Iterable[str]
) -> tuple[int, ...] | None:
    """Turn omitted rows into retained positions, warning as brms does.

    Parameters
    ----------
    frame : pandas.DataFrame
        The caller's rows.
    omit : numpy.ndarray
        Rows to remove.
    names : Iterable[str]
        Variables checked, named in the warning.

    Returns
    -------
    tuple[int, ...] or None
        Retained positions, or None when no row is removed.

    Raises
    ------
    ValueError
        If every row is removed.
    """
    if not omit.any():
        return None
    checked = sorted(set(names))
    if omit.all():
        raise ValueError(
            "All observations in the data were removed because of missing values "
            f"in {checked}"
        )
    dropped = [name for name in checked if bool((_missing(frame, name) & omit).any())]
    warnings.warn(
        f"Rows containing missing values in {dropped} were excluded from the model "
        f"({int(omit.sum())} of {len(frame)} rows); the row selection "
        "(BayesianFit.included_rows) records the rows used",
        UserWarning,
        stacklevel=4,
    )
    return tuple(int(position) for position in np.flatnonzero(~omit))


def complete_rows(
    frame: pd.DataFrame, texts: Iterable[str], columns: Iterable[str] = ()
) -> tuple[int, ...] | None:
    """Decide which rows have every model variable observed.

    Parameters
    ----------
    frame : pandas.DataFrame
        The caller's rows, in their original order.
    texts : Iterable[str]
        Every formula of the model: the response formula and the formulas of
        its distributional and nonlinear parameters.
    columns : Iterable[str]
        Further variables named outside the formulas, such as a weights,
        decision, trials, or grouping column.

    Returns
    -------
    tuple[int, ...] or None
        Original positions of the retained rows, or None when every row is
        complete.

    Raises
    ------
    ValueError
        If no row is complete, as brms refuses.
    """
    names = _variables(frame, texts, columns)
    omit = np.zeros(len(frame), dtype=bool)
    for name in names:
        omit |= _missing(frame, name)
    return _decide(frame, omit, names)


_SUBSET = re.compile(r"\bsubset\s*\(\s*`?([^()`]+?)`?\s*\)")


def joint_complete_rows(
    frame: pd.DataFrame,
    parts: Iterable[tuple[tuple[str, ...], tuple[str, ...]]],
) -> tuple[int, ...] | None:
    """Decide the rows of a multivariate model once for all responses.

    As brms's ``na_omit`` for a multivariate model, a row with a missing value
    in a variable is removed unless every response using that variable
    excludes the row through its ``subset`` addition term.

    Parameters
    ----------
    frame : pandas.DataFrame
        The caller's rows.
    parts : Iterable[tuple[tuple[str, ...], tuple[str, ...]]]
        For each response, its formula texts (response formula first) and
        role columns.

    Returns
    -------
    tuple[int, ...] or None
        Original positions of the rows every response may use, or None.

    Raises
    ------
    ValueError
        If no row remains.
    """
    used: dict[str, list[np.ndarray]] = {}
    for texts, columns in parts:
        include = np.ones(len(frame), dtype=bool)
        match = _SUBSET.search(texts[0].partition("~")[0]) if texts else None
        if match is not None and match.group(1).strip() in frame.columns:
            values = frame[match.group(1).strip()]
            include = np.asarray(values.fillna(True), dtype=bool)
        for name in _variables(frame, texts, columns):
            used.setdefault(name, []).append(include)
    omit = np.zeros(len(frame), dtype=bool)
    for name, includes in used.items():
        relevant = np.logical_or.reduce(includes)
        omit |= _missing(frame, name) & relevant
    return _decide(frame, omit, used)


def compose_positions(
    outer: tuple[int, ...] | None, inner: tuple[int, ...] | None, size: int
) -> tuple[int, ...] | None:
    """Map positions within retained rows back to the original rows.

    Parameters
    ----------
    outer : tuple[int, ...] or None
        Original positions of the rows a later selection indexes, or None for
        all rows.
    inner : tuple[int, ...] or None
        Positions within those rows, or None for all of them.
    size : int
        Number of rows ``inner`` indexes when ``outer`` is None.

    Returns
    -------
    tuple[int, ...] or None
        Original positions of the finally selected rows, or None for all.
    """
    if outer is None:
        return inner
    if inner is None:
        return outer
    return tuple(outer[position] for position in inner)


class _Expression(Protocol):
    """A nonlinear parameter's value expression."""

    @property
    def expression(self) -> str | Mapping[str | int, str]:
        """Return the scalar expression or its component expressions."""
        ...


def model_texts(
    formula: str,
    parameters: Iterable[Mapping[str, str | _Expression] | None],
) -> tuple[str, ...]:
    """Collect the response formula and every parameter formula or expression.

    Parameters
    ----------
    formula : str
        Authored response formula.
    parameters : Iterable[Mapping or None]
        Distributional and nonlinear parameter definitions: formula texts or
        value expressions.

    Returns
    -------
    tuple[str, ...]
        Texts whose variables the model uses.
    """
    texts = [formula]
    for mapping in parameters:
        for value in (mapping or {}).values():
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value.expression, str):
                texts.append(value.expression)
            else:
                texts.extend(value.expression.values())
    return tuple(texts)


def model_columns(*columns: str | int | None) -> tuple[str, ...]:
    """Keep the role arguments that name data columns.

    Parameters
    ----------
    *columns : str, int, or None
        Role arguments: a column name, a constant, or None.

    Returns
    -------
    tuple[str, ...]
        The column names.
    """
    return tuple(column for column in columns if isinstance(column, str))


#: Response roles read on some rows only. The interval upper endpoint of
#: ``cens()`` applies to interval-censored rows, which the role binding
#: checks itself, so a missing value elsewhere does not remove a row.
CONDITIONAL_ROLES = frozenset({"censoring_upper"})


def role_columns(sources: Mapping[str, str | int | float | bool]) -> tuple[str, ...]:
    """Keep the response-role columns every row needs.

    Parameters
    ----------
    sources : Mapping[str, str, int, float, or bool]
        Response roles and their column names or constants.

    Returns
    -------
    tuple[str, ...]
        Column names of the roles that apply to every row.
    """
    return tuple(
        source
        for role, source in sources.items()
        if isinstance(source, str) and role not in CONDITIONAL_ROLES
    )
