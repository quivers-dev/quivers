"""Monotonic effects of ordinal predictors: the brms ``mo()`` term.

Bürkner and Charpentier (2020) model the effect of an ordinal predictor
``x`` with categories coded ``0, ..., D`` as

    b * mo(x, zeta) = b * D * (zeta_1 + ... + zeta_x),

where ``zeta`` lies on the probability simplex of dimension ``D`` and
``mo(0, zeta) = 0``. The scale ``b`` is the expected average difference
between adjacent categories and ``zeta_i`` is the share of the total
effect ``b * D`` attributed to the step from category ``i - 1`` to ``i``.

`Monotonic` is the formulae stateful transform behind ``mo(x)``. It codes
an ordered factor by its declared level positions and an integer predictor
by its distance from the training minimum, as brms's ``get_mo_values``
does, and returns the cumulative indicator matrix whose row ``n`` has ones
in its first ``x_n`` columns. The matrix product of that matrix with
``zeta`` is the partial sum above, at cost O(rows times D).

`MonotonicTerm` retains one evaluated brms special-effects column: the
cumulative indicators of each ``mo()`` factor in an interaction, the
product of the remaining components (brms's ``Csp`` covariate), and the
brms coefficient and simplex labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import reduce
import itertools
from typing import Literal

import didactic.api as dx
from formulae.terms.call import Call
from formulae.terms.call_resolver import LazyCall, LazyOperator, LazyValue, LazyVariable
from formulae.terms.variable import Variable
from formulae.utils import get_interaction_matrix
import numpy as np
import numpy.typing as npt
import pandas as pd

from quivers._values import OPAQUE_BY_VALUE, ValueModel, opaque_by_value

__all__: list[str] = []

type _LazyNode = LazyOperator | LazyVariable | LazyValue | LazyCall
"""A node of formulae's parsed call expression."""


#: brms's ``rename`` substitutions, applied in order (``R/misc.R``).
_BRMS_RENAME: tuple[tuple[str, str], ...] = (
    (" ", ""),
    ("(", ""),
    (")", ""),
    ("[", ""),
    ("]", ""),
    (",", ""),
    ('"', ""),
    ("'", ""),
    ("?", ""),
    ("+", "P"),
    ("-", "M"),
    ("*", "MU"),
    ("/", "D"),
    ("^", "E"),
    ("=", "EQ"),
)


def brms_label(text: str) -> str:
    """Apply brms's parameter-name normalization to a term label.

    Parameters
    ----------
    text : str
        R-style column label, such as ``mo(x, id = "i"):z``.

    Returns
    -------
    str
        The label brms prints as the coefficient name, such as ``moxidEQi:z``.
    """
    for pattern, replacement in _BRMS_RENAME:
        text = text.replace(pattern, replacement)
    return text


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


class MonotonicState(ValueModel):
    """The training coding of one ``mo()`` predictor.

    Attributes
    ----------
    kind : {"ordered", "integer"}
        An ordered factor coded by level position, or whole numbers coded by
        their distance from the training minimum.
    levels : tuple[str, ...]
        Observed levels of an ordered factor, in declared order; empty for
        integers.
    minimum, maximum : int
        Training range of an integer predictor; zero for an ordered factor.
    identity : str or None
        The brms ``id`` naming a simplex shared by several ``mo()`` terms.
    """

    kind: Literal["ordered", "integer"]
    levels: tuple[str, ...] = ()
    minimum: int = 0
    maximum: int = 0
    identity: str | None = None

    @property
    def width(self) -> int:
        """Return the simplex dimension ``D``, one fewer than the category count.

        Returns
        -------
        int
            Number of steps between adjacent categories.
        """
        if self.kind == "ordered":
            return len(self.levels) - 1
        return self.maximum - self.minimum


def _variable_name(values: npt.ArrayLike | pd.Series) -> str:
    """Name an evaluated predictor for error messages.

    Parameters
    ----------
    values : array_like or pandas.Series
        Evaluated ``mo()`` argument.

    Returns
    -------
    str
        The series name, or a generic description.
    """
    name = getattr(values, "name", None)
    return repr(str(name)) if name is not None else "of a mo() term"


def _fit_state(
    values: npt.ArrayLike | pd.Series, identity: str | None
) -> MonotonicState:
    """Determine the coding of a ``mo()`` predictor from its training rows.

    Parameters
    ----------
    values : array_like or pandas.Series
        Training values of the predictor.
    identity : str or None
        Shared simplex identity.

    Returns
    -------
    MonotonicState
        The coding retained for fresh rows.

    Raises
    ------
    ValueError
        If the predictor is neither an ordered factor nor whole numbers, has
        missing values, or has fewer than two categories.
    """
    name = _variable_name(values)
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if bool(series.isna().to_numpy().any()):
        raise ValueError(f"mo(): monotonic predictor {name} has missing values")
    if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
        # brms drops unused factor levels before coding, so the categories
        # are the observed levels in their declared order.
        observed = set(series.tolist())
        state = MonotonicState(
            kind="ordered",
            levels=tuple(
                str(level) for level in series.dtype.categories if level in observed
            ),
            identity=identity,
        )
    else:
        numbers = series.to_numpy()
        if (
            numbers.dtype.kind not in "iuf"
            or not np.isfinite(numbers).all()
            or not np.equal(numbers, np.round(numbers)).all()
        ):
            raise ValueError(
                "Monotonic predictors must be integers or ordered factors. "
                f"Error occurred for variable {name}."
            )
        state = MonotonicState(
            kind="integer",
            minimum=int(numbers.min()),
            maximum=int(numbers.max()),
            identity=identity,
        )
    if state.width < 1:
        raise ValueError(
            f"mo(): monotonic predictor {name} needs at least two categories; "
            "a single category leaves the simplex without a coordinate"
        )
    return state


def _codes(state: MonotonicState, values: npt.ArrayLike | pd.Series) -> np.ndarray:
    """Code rows by their category position under the training coding.

    Parameters
    ----------
    state : MonotonicState
        Training coding.
    values : array_like or pandas.Series
        Predictor values of the evaluated rows.

    Returns
    -------
    numpy.ndarray
        Integer positions ``0, ..., D``.

    Raises
    ------
    ValueError
        If a value is missing, outside the training range, not a whole number,
        or an unknown level, which brms's ``validate_newdata`` also refuses.
    """
    name = _variable_name(values)
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if bool(series.isna().to_numpy().any()):
        raise ValueError(f"mo(): monotonic predictor {name} has missing values")
    if state.kind == "ordered":
        if (
            not isinstance(series.dtype, pd.CategoricalDtype)
            and series.dtype.kind in "iufb"
        ):
            raise ValueError(
                f"mo(): monotonic predictor {name} was an ordered factor in training"
            )
        position = {level: index for index, level in enumerate(state.levels)}
        labels = [str(value) for value in series.tolist()]
        unknown = sorted(set(labels) - position.keys())
        if unknown:
            raise ValueError(
                f"Invalid values in variable {name}: {', '.join(unknown)}; "
                f"levels allowed are {', '.join(state.levels)}"
            )
        return np.array([position[label] for label in labels], dtype=np.int64)
    if isinstance(series.dtype, pd.CategoricalDtype):
        raise ValueError(f"mo(): monotonic predictor {name} was numeric in training")
    numbers = series.to_numpy()
    if numbers.dtype.kind not in "iuf":
        raise ValueError(f"mo(): monotonic predictor {name} was numeric in training")
    invalid = (
        ~np.isfinite(numbers)
        | (numbers < state.minimum)
        | (numbers > state.maximum)
        | ~np.equal(numbers, np.round(numbers))
    )
    if invalid.any():
        shown = ", ".join(str(value) for value in numbers[invalid].tolist())
        raise ValueError(f"Invalid values in variable {name}: {shown}")
    return numbers.astype(np.int64) - state.minimum


def cumulative_indicators(codes: np.ndarray, width: int) -> npt.NDArray[np.float64]:
    """Build the matrix whose product with a simplex gives its partial sums.

    Parameters
    ----------
    codes : numpy.ndarray
        Category positions ``0, ..., width``.
    width : int
        Simplex dimension ``D``.

    Returns
    -------
    numpy.ndarray
        Shape ``(rows, width)``; entry ``(n, i)`` is one when ``i < codes[n]``.
    """
    return (np.arange(width)[None, :] < np.asarray(codes)[:, None]).astype(np.float64)


class Monotonic:
    """The brms ``mo(x, id)`` term as a formulae stateful transform.

    The first call fixes the coding from the training rows; later calls code
    fresh rows the same way and refuse values brms refuses.
    """

    __stateful_transform__ = True

    def __init__(self) -> None:
        """Start without a training coding."""
        self.state: MonotonicState | None = None

    def __call__(
        self,
        x: npt.ArrayLike | pd.Series,
        id: str | None = None,  # noqa: A002
    ) -> npt.NDArray[np.float64]:
        """Return the cumulative indicators of each row's category.

        Parameters
        ----------
        x : array_like or pandas.Series
            Integer predictor or ordered factor.
        id : str or None
            Name of a simplex shared by every ``mo()`` term with the same ``id``.

        Returns
        -------
        numpy.ndarray
            Shape ``(rows, D)`` cumulative indicators.

        Raises
        ------
        ValueError
            If ``id`` is not text, or the values are invalid for the coding.
        """
        if id is not None and (not isinstance(id, str) or not id):
            raise ValueError("mo(): id must be a nonempty character string")
        if self.state is None:
            self.state = _fit_state(x, id)
        state = self.state
        return cumulative_indicators(_codes(state, x), state.width)


class MonotonicFactor(ValueModel):
    """One ``mo()`` factor of a special-effects column.

    Attributes
    ----------
    call : str
        Authored component, such as ``mo(x)``.
    simplex : str
        brms ``simo`` coefficient label: the ``id``, or the column's
        coefficient label followed by the factor's position in the term.
    shared : bool
        Whether the simplex is named by an ``id`` and may be shared.
    design : numpy.ndarray
        Training cumulative indicators, shape ``(rows, D)``; read-only.
    """

    call: str
    simplex: str
    shared: bool = False
    design: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))

    @property
    def width(self) -> int:
        """Return the simplex dimension ``D``.

        Returns
        -------
        int
            Number of columns of the indicator matrix.
        """
        return int(self.design.shape[1])


class MonotonicTerm(ValueModel):
    """One evaluated brms special-effects column built from ``mo()`` factors.

    Its contribution to the linear predictor is

        bsp * prod_j (D_j * design_j @ simo_j) * covariate,

    with one scalar ``bsp`` and one simplex per factor.

    Attributes
    ----------
    term : str
        Authored formulae term, such as ``mo(x):z``.
    name : str
        Column label: the term, or ``{term}_{k}`` for the ``k``-th column of a
        term whose other components evaluate to several columns.
    coef : str
        brms coefficient label of ``bsp``, such as ``mox:z``.
    column : int
        Position of the covariate among the evaluated columns of the term's
        other components, or ``-1`` when the term has none.
    category_specific : bool
        Whether the column has one coefficient and one simplex per ordinal
        transition.
    factors : tuple[MonotonicFactor, ...]
        The ``mo()`` factors in term order.
    covariate : numpy.ndarray
        Training covariate (brms ``Csp``), shape ``(rows,)``, or shape ``(0,)``
        when the term has no other components; read-only.
    """

    term: str
    name: str
    coef: str
    column: int = -1
    category_specific: bool = False
    factors: tuple[MonotonicFactor, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    covariate: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))

    @property
    def linear_values(self) -> npt.NDArray[np.float64]:
        """Return the column the term equals when every simplex is uniform.

        Returns
        -------
        numpy.ndarray
            The product of the category positions and the covariate; the
            column a default coefficient prior is scaled by.
        """
        values = np.ones(self.factors[0].design.shape[0])
        for factor in self.factors:
            values = values * factor.design.sum(axis=1)
        if self.column >= 0:
            values = values * self.covariate
        return values


def modified_call(call: _LazyNode | None) -> _LazyNode | None:
    """Read the special term a ``cs()`` call modifies.

    Parameters
    ----------
    call : LazyCall or other lazy expression, or None
        Parsed component of a term.

    Returns
    -------
    LazyCall or other lazy expression, or None
        The modified ``s()``, ``t2()``, ``gp()`` or ``mo()`` call of a
        ``cs()`` call; the component itself otherwise.
    """
    arguments = getattr(call, "args", ())
    if getattr(call, "callee", None) != "cs" or len(arguments) != 1:
        return call
    inner = arguments[0]
    special = getattr(inner, "callee", None) in {"mo", "s", "t2", "gp"}
    return inner if special else call


def monotonic_positions(components: Sequence[Variable | Call]) -> tuple[int, ...]:
    """Find the ``mo()`` components of a formulae term.

    Parameters
    ----------
    components : Sequence[Variable | Call]
        The term's formulae ``Variable`` and ``Call`` components.

    Returns
    -------
    tuple[int, ...]
        Positions of components that are ``mo()`` calls.

    Raises
    ------
    ValueError
        If ``mo()`` appears inside another call, which brms refuses.
    """
    positions = []
    for index, component in enumerate(components):
        call = modified_call(getattr(component, "call", None))
        if getattr(call, "callee", None) == "mo":
            if any(_nested_mo(argument) for argument in getattr(call, "args", ())):
                raise ValueError(f"The monotonic term '{component}' is invalid.")
            positions.append(index)
        elif call is not None and _nested_mo(call):
            raise ValueError(
                f"The monotonic term in '{getattr(component, 'name', component)}' "
                "is invalid: mo() must be a whole term or an interaction factor"
            )
    return tuple(positions)


def _nested_mo(expression: _LazyNode | None) -> bool:
    """Report whether a formulae lazy expression calls ``mo`` anywhere.

    Parameters
    ----------
    expression : LazyOperator, LazyVariable, LazyValue, LazyCall, or None
        Parsed call, operator, variable, or literal.

    Returns
    -------
    bool
        Whether a ``mo`` call occurs in the expression tree.
    """
    if getattr(expression, "callee", None) == "mo":
        return True
    return any(
        _nested_mo(argument) for argument in getattr(expression, "args", ())
    ) or any(
        _nested_mo(argument) for argument in getattr(expression, "kwargs", {}).values()
    )


def monotonic_design(
    term_name: str, components: Sequence[Variable | Call]
) -> tuple[MonotonicTerm, ...]:
    """Split an evaluated term with ``mo()`` factors into special-effects columns.

    Parameters
    ----------
    term_name : str
        Authored formulae term.
    components : Sequence[Variable | Call]
        Its evaluated formulae components.

    Returns
    -------
    tuple[MonotonicTerm, ...]
        One column per evaluated column of the other components' product,
        as brms's ``Csp``; one column when there are none.

    Raises
    ------
    ValueError
        If a ``mo()`` component lacks its retained coding.
    """
    positions = monotonic_positions(components)
    states = {}
    for position in positions:
        component = components[position]
        assert isinstance(component, Call)
        transform = getattr(modified_call(component.call), "stateful_transform", None)
        if not isinstance(transform, Monotonic) or transform.state is None:
            raise ValueError("mo() terms require the retained monotonic coding")
        states[position] = transform.state
    others = [
        component for index, component in enumerate(components) if index not in states
    ]
    if others:
        product = reduce(
            get_interaction_matrix,
            [
                np.asarray(getattr(component, "value"), dtype=np.float64)
                for component in others
            ],
        )
        product = product.reshape(product.shape[0], -1)
        label_sets = [tuple(getattr(component, "labels")) for component in others]
        other_labels = [tuple(labels) for labels in itertools.product(*label_sets)]
    else:
        product = None
        other_labels = [()]
    terms = []
    for column, labels in enumerate(other_labels):
        remaining = iter(labels)
        parts = [
            brms_label(str(getattr(component, "name")))
            if index in states
            else brms_label(next(remaining))
            for index, component in enumerate(components)
        ]
        coef = ":".join(parts)
        factors = tuple(
            MonotonicFactor(
                call=str(getattr(components[position], "name")),
                simplex=states[position].identity or f"{coef}{order + 1}",
                shared=states[position].identity is not None,
                design=np.asarray(getattr(components[position], "value")),
            )
            for order, position in enumerate(positions)
        )
        terms.append(
            MonotonicTerm(
                term=term_name,
                name=term_name
                if len(other_labels) == 1
                else f"{term_name}_{column + 1}",
                coef=coef,
                column=column if product is not None else -1,
                category_specific=any(
                    getattr(getattr(component, "call", None), "callee", None) == "cs"
                    for component in components
                ),
                factors=factors,
                covariate=product[:, column] if product is not None else np.zeros(0),
            )
        )
    return tuple(terms)


def evaluate_monotonic_term(
    term: MonotonicTerm, components: Sequence[Variable | Call], frame: pd.DataFrame
) -> tuple[tuple[npt.NDArray[np.float64], ...], npt.NDArray[np.float64] | None]:
    """Evaluate a fitted special-effects column on fresh rows.

    Parameters
    ----------
    term : MonotonicTerm
        Fitted column.
    components : Sequence[Variable | Call]
        The retained formulae components of its term.
    frame : pandas.DataFrame
        Fresh rows.

    Returns
    -------
    tuple
        Cumulative indicators of each ``mo()`` factor, and the covariate or
        None when the term has no other components.

    Raises
    ------
    ValueError
        If a fresh value is refused by the training coding or a design width
        changes.
    """
    positions = monotonic_positions(components)
    designs = tuple(
        np.asarray(
            getattr(components[position], "eval_new_data")(frame), dtype=np.float64
        )
        for position in positions
    )
    for design, factor in zip(designs, term.factors, strict=True):
        if design.shape != (len(frame), factor.width):
            raise ValueError(f"new_data changed the monotonic design of {term.term!r}")
    others = [
        component
        for index, component in enumerate(components)
        if index not in positions
    ]
    if not others:
        return designs, None
    product = reduce(
        get_interaction_matrix,
        [
            np.asarray(getattr(component, "eval_new_data")(frame), dtype=np.float64)
            for component in others
        ],
    )
    product = product.reshape(product.shape[0], -1)
    if term.column >= product.shape[1]:
        raise ValueError(f"new_data changed the design width of {term.term!r}")
    return designs, product[:, term.column]


def monotonic_slopes(
    terms: Iterable[MonotonicTerm], expression: str, group: str
) -> tuple[str, ...]:
    """Find the population columns a group-level ``mo()`` slope varies.

    Parameters
    ----------
    terms : Iterable[MonotonicTerm]
        Population-level special-effects columns of the predictor.
    expression : str
        Authored group-level term, such as ``mo(x)``.
    group : str
        Grouping factor.

    Returns
    -------
    tuple[str, ...]
        Column labels whose value multiplies the group coefficient.

    Raises
    ------
    ValueError
        If the term has no population-level counterpart. brms reads the
        group slope as a deviation from that term's ``bsp`` and its simplex.
    """
    names = tuple(term.name for term in terms if term.term == expression)
    if not names:
        raise ValueError(
            f"group-level monotonic term {expression!r} for {group!r} requires the "
            f"same population-level term; add {expression} to the formula"
        )
    return names
