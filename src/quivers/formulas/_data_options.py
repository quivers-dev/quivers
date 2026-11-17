"""brms ``brm`` data options read while a formula is evaluated on its data.

brms passes ``knots`` (a list of knot vectors named by covariate) to mgcv's
smooth constructors, and with ``drop_unused_levels = FALSE`` keeps the
factor levels that no row uses. The formula layer evaluates terms inside the
formula evaluator, so the options of one preparation are scoped by
`formula_data_options` around that evaluation: smooth transforms read the
knots when they construct their bases, and `keep_unused_levels` widens the
evaluated factor codings. Each basis and coding is retained, so prediction
on new rows needs no scope.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import math

from formulae.categorical import Treatment
from formulae.matrices import DesignMatrices
from formulae.terms.variable import Variable
import didactic.api as dx
import numpy as np
import numpy.typing as npt
import pandas as pd

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel

type Knots = Mapping[str, Sequence[float]]
"""brms ``knots``: knot values keyed by smooth covariate."""

_KNOTS: ContextVar[Mapping[str, tuple[float, ...]]] = ContextVar(
    "quivers_smooth_knots", default={}
)
_DROP_UNUSED: ContextVar[bool] = ContextVar("quivers_drop_unused_levels", default=True)
type Data2Object = np.ndarray | pd.DataFrame
"""One brms ``data2`` entry: a matrix, labeled when it is a data frame."""
_DATA2: ContextVar[Mapping[str, Data2Object]] = ContextVar(
    "quivers_formula_data2", default={}
)


class Data2Entry(ValueModel):
    """One ``data2`` entry as a formula term reads it by name.

    Attributes
    ----------
    name : str
        Name of the entry in ``data2`` and in the formula.
    value : numpy.ndarray or pandas.DataFrame
        The entry, compared by identity as the caller's own object.
    """

    name: str
    value: Data2Object = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


def checked_knots(knots: Knots | None) -> dict[str, tuple[float, ...]]:
    """Validate brms ``knots``.

    Parameters
    ----------
    knots : Mapping[str, Sequence[float]] or None
        Knot values keyed by covariate.

    Returns
    -------
    dict[str, tuple[float, ...]]
        Finite float knots keyed by covariate.

    Raises
    ------
    ValueError
        If a key is not a string, or a knot vector is empty or not finite.
    """
    result: dict[str, tuple[float, ...]] = {}
    for name, values in (knots or {}).items():
        if not isinstance(name, str):
            raise ValueError("knots must be keyed by covariate name")
        if isinstance(values, (str, bytes)):
            raise ValueError(f"knots for {name!r} must be a sequence of numbers")
        vector = tuple(float(value) for value in values)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"knots for {name!r} must be finite and nonempty")
        result[name] = vector
    return result


@contextmanager
def formula_data_options(
    knots: Knots | None = None,
    drop_unused_levels: bool | None = None,
    data2: Mapping[str, Data2Object] | None = None,
) -> Iterator[None]:
    """Scope brms data options around the evaluation of a formula.

    Parameters
    ----------
    knots : Mapping[str, Sequence[float]] or None
        Knot values keyed by covariate; None keeps the enclosing scope's.
    drop_unused_levels : bool or None
        brms ``drop_unused_levels``; None keeps the enclosing scope's (True
        outside every scope).
    data2 : Mapping[str, numpy.ndarray or pandas.DataFrame] or None
        brms ``data2``, whose entries a smooth's ``xt`` argument names, as
        in ``xt = list(penalty = P)``; None keeps the enclosing scope's.

    Yields
    ------
    None
        Control, with the options readable by `smooth_knots`,
        `dropping_unused_levels` and `formula_data2`.

    Raises
    ------
    ValueError
        If the knots are invalid or ``drop_unused_levels`` is not Boolean.
    """
    if drop_unused_levels is not None and type(drop_unused_levels) is not bool:
        raise ValueError("drop_unused_levels must be True or False")
    knots_token = _KNOTS.set(checked_knots(knots)) if knots is not None else None
    drop_token = (
        _DROP_UNUSED.set(drop_unused_levels) if drop_unused_levels is not None else None
    )
    data2_token = _DATA2.set(dict(data2)) if data2 is not None else None
    try:
        yield
    finally:
        if data2_token is not None:
            _DATA2.reset(data2_token)
        if drop_token is not None:
            _DROP_UNUSED.reset(drop_token)
        if knots_token is not None:
            _KNOTS.reset(knots_token)


def formula_data2() -> dict[str, Data2Entry]:
    """Return the brms ``data2`` entries of the current scope.

    Returns
    -------
    dict[str, Data2Entry]
        Named entries, empty outside every scope.
    """
    return {
        name: Data2Entry(name=name, value=value) for name, value in _DATA2.get().items()
    }


def dropping_unused_levels() -> bool:
    """Report whether unused factor levels are dropped in the current scope.

    Returns
    -------
    bool
        brms ``drop_unused_levels``.
    """
    return _DROP_UNUSED.get()


def _categories(frame: pd.DataFrame, name: str) -> list[str | int | float] | None:
    """Return a column's declared categories, or None for other columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        Evaluated rows.
    name : str
        Column name.

    Returns
    -------
    list or None
        The pandas categories in their declared order.
    """
    if name not in frame.columns or not isinstance(
        frame[name].dtype, pd.CategoricalDtype
    ):
        return None
    return list(frame[name].cat.categories)


def keep_unused_levels(design: DesignMatrices, frame: pd.DataFrame) -> DesignMatrices:
    """Widen evaluated factor codings to every declared level (brms ``FALSE``).

    Each population-level factor component whose pandas column declares
    categories that no row uses takes all categories, in their declared
    order as R's factor levels, with treatment coding; the population design
    is then evaluated again on the same rows through the retained terms, so
    the unused levels add columns of zeros. Cost O(n p) for ``n`` rows and
    ``p`` columns.

    Parameters
    ----------
    design : formulae.matrices.DesignMatrices
        Design evaluated on ``frame``.
    frame : pandas.DataFrame
        The rows it was evaluated on.

    Returns
    -------
    formulae.matrices.DesignMatrices
        The same object, with a widened population design when a level was
        unused.

    Raises
    ------
    ValueError
        If a factor of a group-level term has unused levels, which brms keeps
        and this layer does not.
    """
    if design.group is not None:
        for name, term in design.group.terms.items():
            components = getattr(getattr(term, "expr", None), "components", ())
            for component in components:
                categories = (
                    _categories(frame, component.name)
                    if isinstance(component, Variable) and component.kind == "categoric"
                    else None
                )
                if categories is not None and set(categories) - set(
                    component.levels or ()
                ):
                    raise ValueError(
                        f"drop_unused_levels=False: factor {component.name!r} of the "
                        f"group-level term {name!r} has unused levels, which are "
                        "not supported there"
                    )
    if design.common is None:
        return design
    widened = False
    for term in design.common.terms.values():
        for component in getattr(term, "components", ()):
            if not isinstance(component, Variable) or component.kind != "categoric":
                continue
            categories = _categories(frame, component.name)
            if categories is None or not set(categories) - set(component.levels or ()):
                continue
            component.levels = categories
            component.contrast_matrix = (
                Treatment().code_with_intercept(categories)
                if component.spans_intercept
                else Treatment().code_without_intercept(categories)
            )
            widened = True
    if widened:
        # Every term is evaluated again through its retained state, and the
        # column slices are rebuilt for the widened codings.
        common = design.common
        rows = len(frame)
        blocks, slices, start = [], {}, 0
        for name, term in common.terms.items():
            block = np.asarray(term.eval_new_data(frame), dtype=np.float64)
            term.data = block
            columns = block.reshape(rows, -1)
            slices[name] = slice(start, start + columns.shape[1])
            start += columns.shape[1]
            blocks.append(columns)
        common.slices = slices
        common.design_matrix = np.column_stack(blocks)
    return design


def smooth_knots(covariate: str) -> npt.NDArray[np.float64] | None:
    """Return the user knots of a smooth covariate in the current scope.

    Parameters
    ----------
    covariate : str
        Covariate name.

    Returns
    -------
    numpy.ndarray or None
        The knots, or None when none were given for the covariate.
    """
    values = _KNOTS.get().get(covariate)
    return None if values is None else np.asarray(values, dtype=np.float64)


__all__: list[str] = []
