"""brms grouping-term syntax: ``gr()``, ``mm()``, ``mmc()``, nesting, and interactions.

brms reads the right side of a group-level bar as a grouping term
(``R/formula-re.R``):

- a column ``g``;
- ``gr(g, by, cor, id, cov, dist)``, the same column with options;
- ``g1:g2``, the interaction of several columns, whose levels brms labels by
  pasting the source levels with ``_``;
- ``g1/g2``, nesting, which brms expands to ``g1`` and ``g1:g2``;
- ``mm(g1, ..., gK, weights, scale, ...)``, a multi-membership term whose
  levels are the union of the member columns' levels.

`rewrite_grouping` normalizes these to the bar notation the formula frontend
already evaluates: ``cor = FALSE`` becomes ``||``, ``id`` becomes ``|id|``,
nesting becomes one block per level of the hierarchy, and every interaction
or multi-membership term becomes one grouping factor named by its brms label.
`GroupingSpec` retains how to build that factor from the source columns, for
training and fresh rows alike.

A multi-membership group-level coefficient ``r`` enters row ``n`` as

    sum_k W_k[n] * r[J_k[n]] * Z_k[n]

over the members ``k``, where ``J_k`` codes the ``k``-th member column, ``W_k``
is its weight (equal, or the ``weights`` columns, normalized to sum to one
per row when ``scale = TRUE``), and ``Z_k`` is the group-level covariate:
the same column for every member, or the ``k``-th column of ``mmc()``.
`IndexedDesign` computes the codes and normalized weights.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal

import didactic.api as dx
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from formulae.terms.call import Call
from formulae.terms.variable import Variable

from quivers._values import OPAQUE_BY_VALUE, ValueModel, opaque_by_value
from quivers.formulas._group_covariance import GroupCovariance
from quivers.formulas._latent_design import IndexedDesign

__all__: list[str] = []


def _frozen(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a read-only C-contiguous float copy of an array.

    Parameters
    ----------
    values : array_like
        Numeric values.

    Returns
    -------
    numpy.ndarray
        The copy, with its writeable flag cleared.
    """
    frozen = np.array(values, dtype=np.float64, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen


class GroupingSpec(ValueModel):
    """A grouping factor built from several source columns.

    Attributes
    ----------
    name : str
        Grouping-factor name used by the formula, the brms group label:
        ``g1:g2`` for an interaction and ``mmg1g2`` for ``mm(g1, g2)``.
    kind : {"interaction", "membership"}
        Interaction of the sources, or multi-membership over them.
    sources : tuple[str, ...]
        Source columns in authored order.
    weights : tuple[str, ...]
        Multi-membership weight columns, one per member; empty for equal
        weights.
    scale : bool
        Whether multi-membership weights are normalized to sum to one per row.
    """

    name: str
    kind: Literal["interaction", "membership"]
    sources: tuple[str, ...]
    weights: tuple[str, ...] = ()
    scale: bool = True

    def labels(self, frame: pd.DataFrame) -> pd.Series:
        """Build the factor column the formula frontend evaluates.

        An interaction pastes the source labels with ``_`` as brms does. A
        multi-membership factor carries its first member, which only fixes the
        row set: the coefficient enters through every member.

        Parameters
        ----------
        frame : pandas.DataFrame
            Rows holding the source columns.

        Returns
        -------
        pandas.Series
            One label per row, missing where a source is missing.

        Raises
        ------
        ValueError
            If a source column is absent.
        """
        missing = [column for column in self.sources if column not in frame.columns]
        if missing:
            raise ValueError(
                f"grouping term {self.name!r} names absent columns {missing!r}"
            )
        if self.kind == "membership":
            return pd.Series(
                frame[self.sources[0]].to_numpy(dtype=object),
                index=frame.index,
                dtype=object,
            )
        parts = [frame[column] for column in self.sources]
        absent = np.zeros(len(frame), dtype=bool)
        for part in parts:
            absent |= np.asarray(part.isna())
        labels = [
            None if absent[row] else "_".join(str(part.iloc[row]) for part in parts)
            for row in range(len(frame))
        ]
        return pd.Series(labels, index=frame.index, dtype=object)


class GroupOptions(ValueModel):
    """The ``gr()`` or ``mm()`` options of one grouping factor.

    Attributes
    ----------
    group : str
        Grouping-factor name.
    by : str or None
        Label of the variable whose levels each get their own group-level
        standard deviations and correlations; every level of the grouping
        factor must lie within one of its levels.
    by_columns : tuple[str, ...]
        Columns of the ``by`` variable, one per source column of the factor:
        the ``by`` level of the level that source names in a row.
    student : bool
        Whether group-level effects follow a multivariate Student-t law
        (brms ``dist = "student"``).
    prior_weight : str or None
        Column of per-level weights on the standard-normal prior of the
        group-level innovations (brms ``pw``).
    """

    group: str
    by: str | None = None
    by_columns: tuple[str, ...] = ()
    student: bool = False
    prior_weight: str | None = None


class GroupLevelOptions(ValueModel):
    """The ``gr()`` options of one grouping factor, coded on its levels.

    Attributes
    ----------
    options : GroupOptions
        Authored options.
    by_levels : tuple[str, ...]
        Sorted levels of the ``by`` factor; empty without ``by``.
    by_codes : tuple[int, ...]
        The ``by`` level of each grouping level, in level order.
    prior_weights : tuple[float, ...]
        The prior weight of each grouping level; empty without ``pw``.
    """

    options: GroupOptions = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    by_levels: tuple[str, ...] = ()
    by_codes: tuple[int, ...] = ()
    prior_weights: tuple[float, ...] = ()

    @property
    def group(self) -> str:
        """Return the grouping-factor name.

        Returns
        -------
        str
            Grouping-factor name.
        """
        return self.options.group


def _per_level(
    frame: pd.DataFrame,
    columns: Sequence[str],
    group: str,
    levels: tuple[str, ...],
    codes: Sequence[Sequence[int]],
) -> list[str | int | float | bool]:
    """Read one value per grouping level, which must be unique.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows aligned with the codes.
    columns : Sequence[str]
        Columns whose values must be constant within each level, one per
        source column of the factor.
    group : str
        Grouping-factor name, for messages.
    levels : tuple[str, ...]
        Grouping levels.
    codes : Sequence[Sequence[int]]
        Level code of each row, for every source column.

    Returns
    -------
    list
        One value per level.

    Raises
    ------
    ValueError
        If a column is absent or a level has several values, as brms refuses.
    """
    found: dict[int, set[str]] = {}
    values: dict[int, str | int | float | bool] = {}
    for column, coded in zip(columns, codes, strict=True):
        if column not in frame.columns:
            raise ValueError(
                f"grouping factor {group!r}: column {column!r} names no data column"
            )
        for code, value in zip(coded, frame[column].tolist(), strict=True):
            found.setdefault(code, set()).add(str(value))
            values.setdefault(code, value)
    if any(len(labels) > 1 for labels in found.values()):
        named = repr(columns[0]) if len(columns) == 1 else repr(list(columns))
        raise ValueError(
            f"Some levels of {group!r} correspond to multiple levels of {named}."
        )
    return [values[code] for code in range(len(levels))]


def fit_group_options(
    options: Sequence[GroupOptions],
    frame: pd.DataFrame,
    group_levels: Mapping[str, tuple[str, ...]],
    group_indices: Mapping[str, Sequence[int]],
    memberships: Sequence[Membership] = (),
) -> tuple[GroupLevelOptions, ...]:
    """Code the ``by``, ``dist`` and ``pw`` options on the training levels.

    Parameters
    ----------
    options : Sequence[GroupOptions]
        Authored options per grouping factor.
    frame : pandas.DataFrame
        Included training rows.
    group_levels, group_indices : Mapping
        Training coding of each grouping factor.
    memberships : Sequence[Membership]
        Training codes of the multi-membership factors.

    Returns
    -------
    tuple[GroupLevelOptions, ...]
        Level-coded options.

    Raises
    ------
    ValueError
        If ``by`` or ``pw`` varies within a grouping level, or weights are
        not finite and nonnegative.
    """
    members = {membership.group: membership.codes for membership in memberships}
    fitted = []
    for option in options:
        levels = group_levels[option.group]
        codes = members.get(option.group, (group_indices[option.group],))
        by_levels: tuple[str, ...] = ()
        by_codes: tuple[int, ...] = ()
        weights: tuple[float, ...] = ()
        if option.by_columns:
            per_level = [
                str(value)
                for value in _per_level(
                    frame, option.by_columns, option.group, levels, codes
                )
            ]
            by_levels = tuple(
                sorted(
                    {
                        str(value)
                        for column in option.by_columns
                        for value in frame[column].tolist()
                    }
                )
            )
            position = {label: index for index, label in enumerate(by_levels)}
            by_codes = tuple(position[value] for value in per_level)
        if option.prior_weight is not None:
            raw = np.asarray(
                _per_level(frame, (option.prior_weight,), option.group, levels, codes),
                dtype=np.float64,
            )
            if not np.isfinite(raw).all() or bool((raw < 0).any()):
                raise ValueError(
                    f"gr({option.group}): prior weights must be finite and nonnegative"
                )
            weights = tuple(float(value) for value in raw)
        fitted.append(
            GroupLevelOptions(
                options=option,
                by_levels=by_levels,
                by_codes=by_codes,
                prior_weights=weights,
            )
        )
    return tuple(fitted)


class Membership(ValueModel):
    """Training codes, weights, and member covariates of one ``mm()`` factor.

    Attributes
    ----------
    group : str
        Grouping-factor name.
    spec : GroupingSpec
        Its construction from member and weight columns.
    codes : tuple[tuple[int, ...], ...]
        One code per row for each member, into the factor's sorted levels.
    weights : numpy.ndarray
        Membership weights, shape ``(rows, members)``; read-only.
    """

    group: str
    spec: GroupingSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    codes: tuple[tuple[int, ...], ...]
    weights: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))

    @property
    def size(self) -> int:
        """Return the number of members.

        Returns
        -------
        int
            Member count.
        """
        return len(self.spec.sources)


class MemberCovariate(ValueModel):
    """The member-specific covariate of one multi-membership coefficient.

    Attributes
    ----------
    group : str
        Grouping-factor name.
    slope : str
        Group-level coefficient label, ``Intercept`` for the intercept.
    term : str
        formulae term evaluating the covariate; empty for the intercept.
    per_member : bool
        Whether the term evaluates one column per member (``mmc()``).
    values : numpy.ndarray
        Training values ``Z_k``, shape ``(rows, members)``; read-only.
    """

    group: str
    slope: str
    term: str = ""
    per_member: bool = False
    values: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))


_GROUPING_CALLS = frozenset({"gr", "mm"})


def _blocks(formula: str) -> list[tuple[int, int, str]]:
    """Find parenthesized group-level blocks containing a top-level bar.

    Parameters
    ----------
    formula : str
        Formula text.

    Returns
    -------
    list[tuple[int, int, str]]
        Start, end (exclusive), and body of each block, innermost blocks of
        nested parentheses excluded.
    """
    found = []
    starts: list[int] = []
    quote: str | None = None
    for position, char in enumerate(formula):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            starts.append(position)
        elif char == ")" and starts:
            start = starts.pop()
            body = formula[start + 1 : position]
            if _top_level(body, "|"):
                found.append((start, position + 1, body))
    return found


def _top_level(text: str, separator: str) -> list[int]:
    """Find separator positions outside parentheses and quotes.

    Parameters
    ----------
    text : str
        Text to scan.
    separator : str
        One character.

    Returns
    -------
    list[int]
        Positions of top-level separators.
    """
    depth = 0
    quote: str | None = None
    positions = []
    for offset, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == separator and depth == 0:
            positions.append(offset)
    return positions


def _split(text: str, separator: str) -> list[str]:
    """Split text at top-level separators.

    Parameters
    ----------
    text : str
        Text to split.
    separator : str
        One character.

    Returns
    -------
    list[str]
        Stripped pieces.
    """
    bounds = [-1, *_top_level(text, separator), len(text)]
    return [text[left + 1 : right].strip() for left, right in zip(bounds, bounds[1:])]


_R_NAME = re.compile(r"(?:[A-Za-z]|\.(?![0-9]))[A-Za-z0-9._]*")


def _r_value(text: str, call: str) -> str | bool | None:
    """Read one literal or name argument of a grouping call.

    Parameters
    ----------
    text : str
        Authored argument.
    call : str
        Calling function, for messages.

    Returns
    -------
    str, bool, or None
        A quoted string or column name, ``TRUE``/``FALSE``, or ``NULL``/``NA``.

    Raises
    ------
    ValueError
        If the argument is another expression.
    """
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    if text in {"TRUE", "T"}:
        return True
    if text in {"FALSE", "F"}:
        return False
    if text in {"NULL", "NA"}:
        return None
    if _R_NAME.fullmatch(text):
        return text
    raise ValueError(
        f"{call}(): arguments must be column names or literals, got {text!r}"
    )


def _r_call(text: str) -> tuple[str, list[str], dict[str, str]]:
    """Split an R call into its function, positional, and named arguments.

    Parameters
    ----------
    text : str
        Call text, such as ``gr(g, cor = FALSE)``.

    Returns
    -------
    tuple[str, list[str], dict[str, str]]
        Function name, positional argument texts, and named argument texts.

    Raises
    ------
    ValueError
        If the text is not one call or repeats an argument name.
    """
    head, _, rest = text.strip().partition("(")
    if not rest.endswith(")") or _top_level(rest[:-1], ")"):
        raise ValueError(f"invalid grouping term {text!r}")
    positional: list[str] = []
    named: dict[str, str] = {}
    body = rest[:-1]
    if not body.strip():
        return head.strip(), positional, named
    for piece in _split(body, ","):
        equals = [
            position
            for position in _top_level(piece, "=")
            if piece[position + 1 : position + 2] != "="
            and piece[position - 1 : position] not in {"=", "!", "<", ">"}
        ]
        if equals:
            key = piece[: equals[0]].strip()
            if key in named:
                raise ValueError(f"{head.strip()}(): argument {key!r} is repeated")
            named[key] = piece[equals[0] + 1 :].strip()
        else:
            positional.append(piece)
    return head.strip(), positional, named


class _Grouping:
    """One parsed grouping term and its options."""

    def __init__(
        self,
        names: tuple[str, ...],
        spec: GroupingSpec | None,
        cor: bool,
        identity: str | None,
        covariance: str | None,
    ) -> None:
        """Retain a parsed grouping term.

        Parameters
        ----------
        names : tuple[str, ...]
            Grouping factors after nesting is expanded, outermost first.
        spec : GroupingSpec or None
            Construction of a derived factor, for the last name.
        cor : bool
            Whether coefficients of a block are correlated.
        identity : str or None
            Shared correlation identity from ``id``.
        covariance : str or None
            ``data2`` name of a known covariance between levels.
        """
        self.names = names
        self.spec = spec
        self.cor = cor
        self.identity = identity
        self.covariance = covariance
        self.by: str | None = None
        self.by_columns: tuple[str, ...] = ()
        self.student = False
        self.prior_weight: str | None = None


def _parse_grouping(text: str) -> tuple[_Grouping, tuple[GroupingSpec, ...]]:
    """Parse the grouping side of one bar.

    Parameters
    ----------
    text : str
        Text right of the last bar.

    Returns
    -------
    tuple[_Grouping, tuple[GroupingSpec, ...]]
        The grouping and every derived factor it needs.

    Raises
    ------
    ValueError
        If the term is malformed or uses an unsupported option.
    """
    text = text.strip()
    head = text.split("(", 1)[0].strip()
    if head in _GROUPING_CALLS and text.endswith(")"):
        return _parse_call(text)
    if _top_level(text, "/"):
        levels = _split(text, "/")
        names: list[str] = []
        specs: list[GroupingSpec] = []
        for depth in range(1, len(levels) + 1):
            sources = tuple(
                part for level in levels[:depth] for part in _split(level, ":")
            )
            name, spec = _interaction(sources)
            names.append(name)
            if spec is not None:
                specs.append(spec)
        return _Grouping(tuple(names), None, True, None, None), tuple(specs)
    name, spec = _interaction(tuple(_split(text, ":")))
    return _Grouping((name,), spec, True, None, None), (
        (spec,) if spec is not None else ()
    )


def _interaction(sources: tuple[str, ...]) -> tuple[str, GroupingSpec | None]:
    """Name an interaction of grouping columns.

    Parameters
    ----------
    sources : tuple[str, ...]
        Column names.

    Returns
    -------
    tuple[str, GroupingSpec or None]
        The factor name and, for several columns, its construction.

    Raises
    ------
    ValueError
        If a source is not a plain column name.
    """
    for source in sources:
        if not _R_NAME.fullmatch(source):
            raise ValueError(
                f"grouping factor {source!r} must be a column name; brms refuses "
                "expressions in grouping terms"
            )
    if len(sources) == 1:
        return sources[0], None
    name = ":".join(sources)
    return name, GroupingSpec(name=name, kind="interaction", sources=sources)


def _member_columns(argument: str, text: str, members: int) -> tuple[str, ...]:
    """Parse an ``mm()`` argument holding one column per member.

    Parameters
    ----------
    argument : str
        Argument name, for messages.
    text : str
        Authored ``cbind()`` call.
    members : int
        Number of members.

    Returns
    -------
    tuple[str, ...]
        Column names in member order.

    Raises
    ------
    ValueError
        If the argument is not ``cbind()`` of one column per member.
    """
    function, arguments, extra = (
        _r_call(text) if text.rstrip().endswith(")") else ("", [], {})
    )
    columns = [_r_value(item, "mm") for item in arguments] if not extra else []
    if function != "cbind" or any(not isinstance(column, str) for column in columns):
        raise ValueError(
            f"mm(): {argument} must be cbind() of one data column per member"
        )
    if len(columns) != members:
        raise ValueError(
            f"mm(): {argument} need as many columns as grouping terms ({members})"
        )
    return tuple(str(column) for column in columns)


def _parse_call(text: str) -> tuple[_Grouping, tuple[GroupingSpec, ...]]:
    """Parse ``gr()`` or ``mm()``.

    Parameters
    ----------
    text : str
        Authored call.

    Returns
    -------
    tuple[_Grouping, tuple[GroupingSpec, ...]]
        The grouping and its derived factor.

    Raises
    ------
    ValueError
        If arguments are malformed or unknown.
    """
    head, arguments, options = _r_call(text)
    groups = []
    for argument in arguments:
        value = _r_value(argument, head)
        if not isinstance(value, str):
            raise ValueError(f"{head}(): grouping factors must be column names")
        groups.append(value)
    allowed = {"by", "cor", "id", "cov", "dist"}
    allowed |= {"weights", "scale"} if head == "mm" else {"pw"}
    unknown = sorted(options.keys() - allowed)
    if unknown:
        raise ValueError(f"{head}(): unknown arguments {unknown!r}")
    cor = _r_value(options["cor"], head) if "cor" in options else True
    if not isinstance(cor, bool):
        raise ValueError(f"{head}(): cor must be TRUE or FALSE")
    identity = _r_value(options["id"], head) if "id" in options else None
    if identity is not None and not isinstance(identity, str):
        raise ValueError(f"{head}(): id must be a character string")
    covariance = _r_value(options["cov"], head) if "cov" in options else None
    if covariance is not None and not isinstance(covariance, str):
        raise ValueError(f"{head}(): cov must name one data2 matrix")
    dist = _r_value(options["dist"], head) if "dist" in options else "gaussian"
    if dist not in {"gaussian", "student"}:
        raise ValueError(f"{head}(): dist must be 'gaussian' or 'student'")
    if head == "gr":
        columns: dict[str, str | None] = {}
        for option in ("by", "pw"):
            value = _r_value(options[option], head) if option in options else None
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{head}(): {option} must name a data column")
            columns[option] = value
        if len(groups) != 1:
            raise ValueError(
                "Grouping structure 'gr' expects only a single grouping term"
            )
        name, spec = _interaction(tuple(_split(groups[0], ":")))
        grouping = _Grouping((name,), spec, cor, identity, covariance)
        by = columns["by"]
        grouping.by, grouping.by_columns = by, (by,) if by is not None else ()
        grouping.prior_weight = columns["pw"]
        grouping.student = dist == "student"
        return grouping, ((spec,) if spec is not None else ())
    if len(groups) < 2:
        raise ValueError(
            "Multi-membership terms require at least two grouping variables."
        )
    for group in groups:
        _interaction((group,))
    weights = (
        _member_columns("weights", options["weights"], len(groups))
        if "weights" in options
        else ()
    )
    by_columns = (
        _member_columns("by", options["by"], len(groups)) if "by" in options else ()
    )
    scale = _r_value(options["scale"], "mm") if "scale" in options else True
    if not isinstance(scale, bool):
        raise ValueError("mm(): scale must be TRUE or FALSE")
    spec = GroupingSpec(
        name="mm" + "".join(groups),
        kind="membership",
        sources=tuple(groups),
        weights=weights,
        scale=scale,
    )
    grouping = _Grouping((spec.name,), spec, cor, identity, covariance)
    grouping.student = dist == "student"
    grouping.by_columns = by_columns
    if by_columns:
        grouping.by = f"cbind({', '.join(by_columns)})"
    return grouping, (spec,)


class GroupingRewrite(ValueModel):
    """A formula with brms grouping terms normalized to bar notation.

    Attributes
    ----------
    formula : str
        Rewritten formula text.
    specs : tuple[GroupingSpec, ...]
        Derived grouping factors to materialize as columns.
    covariances : tuple[tuple[str, str], ...]
        Grouping factors and the ``data2`` names of their ``cov`` matrices.
    """

    formula: str
    specs: tuple[GroupingSpec, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    covariances: tuple[tuple[str, str], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    options: tuple[GroupOptions, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


def rewrite_grouping(formula: str) -> GroupingRewrite:
    """Normalize ``gr()``, ``mm()``, nesting, and interactions to bars.

    Parameters
    ----------
    formula : str
        Authored formula.

    Returns
    -------
    GroupingRewrite
        Formula over plain grouping factors and the constructions they need.

    Raises
    ------
    ValueError
        If a grouping term is malformed or unsupported, or two blocks
        disagree on a derived factor.
    """
    specs: dict[str, GroupingSpec] = {}
    covariances: dict[str, str] = {}
    options: dict[str, GroupOptions] = {}
    replacements = []
    for start, end, body in _blocks(formula):
        bars = _top_level(body, "|")
        lhs = body[: bars[0]].strip()
        middle = body[bars[0] + 1 : bars[-1]] if len(bars) > 1 else None
        if len(bars) > 2 and not (len(bars) == 3 and bars[1] == bars[0] + 1):
            raise ValueError(f"invalid random-effect block ({body})")
        group_text = body[bars[-1] + 1 :]
        if len(bars) == 2 and bars[1] == bars[0] + 1:
            middle = ""
        grouping, derived = _parse_grouping(group_text)
        for spec in derived:
            if specs.setdefault(spec.name, spec) != spec:
                raise ValueError(
                    f"grouping factor {spec.name!r} is constructed inconsistently"
                )
        if grouping.covariance is not None:
            for name in grouping.names:
                if (
                    covariances.setdefault(name, grouping.covariance)
                    != grouping.covariance
                ):
                    raise ValueError(
                        f"grouping factor {name!r} has conflicting cov matrices"
                    )
        for name in grouping.names:
            option = GroupOptions(
                group=name,
                by=grouping.by,
                by_columns=grouping.by_columns,
                student=grouping.student,
                prior_weight=grouping.prior_weight,
            )
            if options.setdefault(name, option) != option:
                raise ValueError(
                    f"every group-level term of {name!r} must use the same gr() "
                    "options by, dist, and pw"
                )
        if not grouping.cor:
            bar = "||"
        elif grouping.identity is not None:
            if middle is not None and middle.strip() not in {"", grouping.identity}:
                raise ValueError(f"conflicting correlation identities in ({body})")
            bar = f"|{grouping.identity}|"
        elif middle is None:
            bar = "|"
        elif middle == "":
            bar = "||"
        else:
            bar = f"|{middle.strip()}|"
        text = " + ".join(f"({lhs} {bar} {name})" for name in grouping.names)
        replacements.append((start, end, text))
    for start, end, text in reversed(replacements):
        formula = formula[:start] + text + formula[end:]
    return GroupingRewrite(
        formula=formula,
        specs=tuple(specs.values()),
        covariances=tuple(covariances.items()),
        options=tuple(
            option
            for option in options.values()
            if option.by_columns or option.student or option.prior_weight
        ),
    )


def add_grouping_columns(
    frame: pd.DataFrame, specs: Sequence[GroupingSpec], *, require: bool = True
) -> pd.DataFrame:
    """Add the derived grouping factors to a frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows holding the source columns.
    specs : Sequence[GroupingSpec]
        Derived factors.
    require : bool
        Whether every source column must be present. Prediction rows may omit
        the sources of factors they do not use; the group-level validation
        then reports a missing factor that is needed.

    Returns
    -------
    pandas.DataFrame
        A copy with one column per derived factor, or the frame itself when
        there are none.

    Raises
    ------
    ValueError
        If a derived name is already a data column with other values.
    """
    if not specs:
        return frame
    frame = frame.copy()
    for spec in specs:
        if not require and not set(spec.sources) <= set(frame.columns):
            continue
        labels = spec.labels(frame)
        if spec.name in frame.columns and not frame[spec.name].astype(object).equals(
            labels
        ):
            raise ValueError(
                f"grouping factor name {spec.name!r} is already a data column"
            )
        frame[spec.name] = labels
    return frame


def _member_labels(frame: pd.DataFrame, spec: GroupingSpec) -> list[list[str]]:
    """Read every row's member labels.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows holding the member columns.
    spec : GroupingSpec
        Multi-membership construction.

    Returns
    -------
    list[list[str]]
        Member labels per row.

    Raises
    ------
    ValueError
        If a member column is absent or has missing values.
    """
    for column in spec.sources:
        if column not in frame.columns:
            raise ValueError(f"new_data is missing predictors {[column]}")
        if bool(frame[column].isna().to_numpy().any()):
            raise ValueError(
                f"multi-membership column {column!r} of {spec.name!r} has missing values"
            )
    return [
        [str(value) for value in values]
        for values in zip(*(frame[column].tolist() for column in spec.sources))
    ]


def member_weights(frame: pd.DataFrame, spec: GroupingSpec) -> npt.NDArray[np.float64]:
    """Evaluate membership weights as brms does.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows holding member and weight columns.
    spec : GroupingSpec
        Multi-membership construction.

    Returns
    -------
    numpy.ndarray
        Shape ``(rows, members)``: equal weights ``1 / K`` by default, or the
        weight columns, normalized per row when ``scale`` is true.

    Raises
    ------
    ValueError
        If weights are missing, nonfinite, or invalid for the scaling.
    """
    labels = _member_labels(frame, spec)
    if spec.weights:
        missing = [column for column in spec.weights if column not in frame.columns]
        if missing:
            raise ValueError(f"new_data is missing predictors {missing}")
        values = frame[list(spec.weights)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"multi-membership weights of {spec.name!r} must be finite"
            )
        design = IndexedDesign.fit(
            labels, torch.tensor(values.copy()), scale=spec.scale
        )
    else:
        design = IndexedDesign.fit(labels, None, scale=True)
    return np.asarray(design.weights, dtype=np.float64).reshape(len(labels), -1)


def fit_membership(
    frame: pd.DataFrame, spec: GroupingSpec
) -> tuple[tuple[str, ...], Membership]:
    """Code the training members of one ``mm()`` factor.

    Parameters
    ----------
    frame : pandas.DataFrame
        Included training rows.
    spec : GroupingSpec
        Multi-membership construction.

    Returns
    -------
    tuple[tuple[str, ...], Membership]
        Sorted union of the member levels and the retained membership.

    Raises
    ------
    ValueError
        If members or weights are invalid.
    """
    labels = _member_labels(frame, spec)
    design = IndexedDesign.fit(labels, None, scale=True)
    codes = tuple(
        tuple(row[member] for row in design.indices)
        for member in range(len(spec.sources))
    )
    return design.levels, Membership(
        group=spec.name,
        spec=spec,
        codes=codes,
        weights=member_weights(frame, spec),
    )


def member_codes(
    frame: pd.DataFrame, spec: GroupingSpec, positions: Mapping[str, int]
) -> tuple[tuple[int, ...], ...]:
    """Code fresh members against an explicit level map.

    Parameters
    ----------
    frame : pandas.DataFrame
        Fresh rows.
    spec : GroupingSpec
        Multi-membership construction.
    positions : Mapping[str, int]
        Code of every admissible label.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        One code per row for each member.

    Raises
    ------
    ValueError
        If a member label has no code.
    """
    labels = _member_labels(frame, spec)
    unknown = sorted({label for row in labels for label in row} - positions.keys())
    if unknown:
        raise ValueError(
            f"new_data contains unknown groups {unknown} for {spec.name!r}; "
            "only fitted levels are supported"
        )
    return tuple(
        tuple(positions[row[member]] for row in labels)
        for member in range(len(spec.sources))
    )


def member_label_sets(frame: pd.DataFrame, spec: GroupingSpec) -> list[list[str]]:
    """Read every fresh row's member labels.

    Parameters
    ----------
    frame : pandas.DataFrame
        Fresh rows.
    spec : GroupingSpec
        Multi-membership construction.

    Returns
    -------
    list[list[str]]
        Member labels per row, in member order.
    """
    return _member_labels(frame, spec)


class GroupingPlan(ValueModel):
    """Derived grouping factors, multi-membership codes, and member covariates.

    Attributes
    ----------
    specs : tuple[GroupingSpec, ...]
        Interaction and multi-membership factors built from source columns.
    covariances : tuple[tuple[str, str], ...]
        Grouping factors and the ``data2`` names of their ``cov`` matrices.
    memberships : tuple[Membership, ...]
        Training codes and weights of each ``mm()`` factor.
    covariates : tuple[MemberCovariate, ...]
        Member covariates of every multi-membership coefficient.
    options : tuple[GroupLevelOptions, ...]
        ``gr(by, dist, pw)`` options coded on the training levels.
    """

    specs: tuple[GroupingSpec, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    covariances: tuple[tuple[str, str], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    memberships: tuple[Membership, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    covariates: tuple[MemberCovariate, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    options: tuple[GroupLevelOptions, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def option(self, group: str) -> GroupLevelOptions | None:
        """Find the ``gr()`` options of a grouping factor.

        Parameters
        ----------
        group : str
            Grouping-factor name.

        Returns
        -------
        GroupLevelOptions or None
            Its level-coded options, or None without options.
        """
        return next((option for option in self.options if option.group == group), None)

    def spec(self, group: str) -> GroupingSpec | None:
        """Find the construction of a derived grouping factor.

        Parameters
        ----------
        group : str
            Grouping-factor name.

        Returns
        -------
        GroupingSpec or None
            Its construction, or None for a data column.
        """
        return next((spec for spec in self.specs if spec.name == group), None)

    def membership(self, group: str) -> Membership | None:
        """Find the retained membership of an ``mm()`` factor.

        Parameters
        ----------
        group : str
            Grouping-factor name.

        Returns
        -------
        Membership or None
            Its codes and weights, or None for an ordinary factor.
        """
        return next(
            (
                membership
                for membership in self.memberships
                if membership.group == group
            ),
            None,
        )

    def covariate(self, group: str, slope: str) -> MemberCovariate | None:
        """Find the member covariate of one multi-membership coefficient.

        Parameters
        ----------
        group : str
            Grouping-factor name.
        slope : str
            Coefficient label.

        Returns
        -------
        MemberCovariate or None
            Its member values, or None for an ordinary coefficient.
        """
        return next(
            (
                covariate
                for covariate in self.covariates
                if covariate.group == group and covariate.slope == slope
            ),
            None,
        )


def is_member_call(components: Sequence[Variable | Call]) -> bool:
    """Report whether a group-level term is a whole ``mmc()`` call.

    Parameters
    ----------
    components : Sequence[Variable | Call]
        formulae components of the group-level term.

    Returns
    -------
    bool
        Whether the term is exactly one ``mmc()`` call.

    Raises
    ------
    ValueError
        If ``mmc()`` appears inside another expression.
    """
    callees = [
        getattr(getattr(component, "call", None), "callee", None)
        for component in components
    ]
    if "mmc" in callees and len(components) != 1:
        raise ValueError(
            "mmc() must be the whole group-level term, as in (1 + mmc(x1, x2) | mm(g1, g2))"
        )
    return callees == ["mmc"]


def member_covariate(
    group: str,
    spec: GroupingSpec,
    slope: str,
    term: str,
    values: npt.ArrayLike,
    *,
    per_member: bool,
) -> MemberCovariate:
    """Retain the member values of one multi-membership coefficient.

    Parameters
    ----------
    group : str
        Grouping-factor name.
    spec : GroupingSpec
        Its construction.
    slope : str
        Coefficient label.
    term : str
        formulae term evaluating the covariate; empty for the intercept.
    values : array_like
        One column shared by every member, or one column per member.
    per_member : bool
        Whether ``values`` has one column per member (``mmc()``).

    Returns
    -------
    MemberCovariate
        Values of shape ``(rows, members)``.

    Raises
    ------
    ValueError
        If ``mmc()`` has another number of columns than members.
    """
    array = np.asarray(values, dtype=np.float64)
    size = len(spec.sources)
    if per_member:
        array = array.reshape(array.shape[0], -1)
        if array.shape[1] != size:
            raise ValueError(
                f"mmc() for {group!r} needs one covariate per member ({size}), "
                f"got {array.shape[1]}"
            )
    else:
        array = np.repeat(array.reshape(-1, 1), size, axis=1)
    return MemberCovariate(
        group=group, slope=slope, term=term, per_member=per_member, values=array
    )


def resolve_group_covariances(
    formula: str | None,
    data2: Mapping[str, np.ndarray | pd.DataFrame] | None,
    group_covariances: Mapping[str, GroupCovariance] | None,
) -> dict[str, GroupCovariance] | None:
    """Turn ``gr(g, cov = M)`` into a known covariance between levels of ``g``.

    brms reads ``M`` from ``data2`` and requires its row names to cover the
    levels of ``g``. The result joins the explicitly given covariances and
    feeds the shared Gaussian group layout, which scales the Cholesky factor
    of ``M`` by the coefficient's standard deviation, as brms's ``Lcov``.

    Parameters
    ----------
    formula : str or None
        Authored formula text, or None for a formula that cannot carry
        ``cov`` (a joint formula).
    data2 : Mapping or None
        Named matrices.
    group_covariances : Mapping[str, GroupCovariance] or None
        Covariances given directly.

    Returns
    -------
    dict[str, GroupCovariance] or None
        All covariances by grouping factor, or the given mapping unchanged
        when the formula names none.

    Raises
    ------
    ValueError
        If a named matrix is absent, unlabeled, not square with matching row
        and column labels, or a factor receives two covariances.
    """
    if formula is None:
        return None if group_covariances is None else dict(group_covariances)
    covariances = rewrite_grouping(formula).covariances
    if not covariances:
        return None if group_covariances is None else dict(group_covariances)
    resolved = dict(group_covariances or {})
    for group, name in covariances:
        if data2 is None or name not in data2:
            raise ValueError(f"gr({group}, cov = {name}) requires data2[{name!r}]")
        matrix = data2[name]
        if not isinstance(matrix, pd.DataFrame):
            raise ValueError(
                f"data2[{name!r}] needs row and column names matching the levels "
                f"of {group!r}; pass a labeled pandas DataFrame"
            )
        labels = tuple(str(label) for label in matrix.index)
        if tuple(str(label) for label in matrix.columns) != labels:
            raise ValueError(f"data2[{name!r}] must have matching row and column names")
        covariance = GroupCovariance(
            levels=labels, matrix=torch.tensor(matrix.to_numpy(dtype=np.float64))
        )
        if resolved.setdefault(group, covariance) != covariance:
            raise ValueError(
                f"grouping factor {group!r} has both gr(cov = {name}) and another "
                "group_covariances entry"
            )
    return resolved
