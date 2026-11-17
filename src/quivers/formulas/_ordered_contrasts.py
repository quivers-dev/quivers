"""R's polynomial contrasts for ordered factors in formula designs.

R codes an ordered factor by orthogonal polynomial contrasts
(``contr.poly``, the default of ``options("contrasts")`` for ordered
factors), and brms builds its design matrices with R's ``model.matrix``.
A pandas ordered categorical is R's ordered factor, so its contrast-coded
columns are the linear, quadratic, cubic and higher polynomial contrasts,
labelled ``.L``, ``.Q``, ``.C``, ``^4``, ... as R labels them. Where R codes
the factor by indicator columns instead (the first factor of a design
without an intercept), the indicator coding is kept.
"""

from __future__ import annotations

from formulae.categorical import ContrastMatrix
from formulae.matrices import DesignMatrices
from formulae.terms.variable import Variable
import numpy as np
import numpy.typing as npt
import pandas as pd

#: Largest number of levels R's ``contr.poly`` represents.
MAX_LEVELS = 95


def contr_poly(levels: int) -> npt.NDArray[np.float64]:
    """Compute R's ``contr.poly(n)`` for equally spaced levels.

    Parameters
    ----------
    levels : int
        Number of factor levels ``n``, at least two.

    Returns
    -------
    numpy.ndarray
        ``(n, n - 1)`` matrix whose column ``j`` is the orthonormal
        polynomial of degree ``j + 1`` in the scores ``1, ..., n``.

    Raises
    ------
    ValueError
        If ``levels`` is below two or above 95, the limit R states for the
        accuracy of the representation.

    Notes
    -----
    R takes the QR decomposition of the Vandermonde matrix of the centered
    scores and scales each column of ``Q diag(R)`` to unit length
    (``stats:::make.poly``). A column of ``Q diag(R)`` does not depend on
    the sign convention of the factorization, so the result equals R's.
    Cost is O(n^3).
    """
    if levels < 2:
        raise ValueError("contr.poly requires at least two levels")
    if levels > MAX_LEVELS:
        raise ValueError(
            "an ordered factor with more than 95 levels has no polynomial "
            "contrasts: R states that orthogonal polynomials cannot be "
            "represented accurately enough for 95 degrees of freedom"
        )
    scores = np.arange(1, levels + 1, dtype=np.float64)
    centered = scores - scores.mean()
    vandermonde = np.vander(centered, levels, increasing=True)
    q, r = np.linalg.qr(vandermonde)
    raw = q * np.diag(r)
    raw /= np.sqrt((raw**2).sum(axis=0))
    return raw[:, 1:]


def poly_labels(levels: int) -> list[str]:
    """Return R's column labels of ``contr.poly(n)``.

    Parameters
    ----------
    levels : int
        Number of factor levels.

    Returns
    -------
    list[str]
        ``.L``, ``.Q``, ``.C``, then ``^4``, ``^5``, ..., one per contrast.
    """
    named = [".L", ".Q", ".C"]
    return [
        named[degree - 1] if degree <= 3 else f"^{degree}"
        for degree in range(1, levels)
    ]


def _ordered(frame: pd.DataFrame, name: str) -> bool:
    """Report whether a column is a pandas ordered categorical.

    Parameters
    ----------
    frame : pandas.DataFrame
        Evaluated rows.
    name : str
        Column name.

    Returns
    -------
    bool
        True for an ordered categorical column.
    """
    if name not in frame.columns:
        return False
    dtype = frame[name].dtype
    return isinstance(dtype, pd.CategoricalDtype) and bool(dtype.ordered)


def _recode(components: tuple[object, ...], frame: pd.DataFrame) -> bool:
    """Give the contrast-coded ordered factors of one term polynomial contrasts.

    Parameters
    ----------
    components : tuple[object, ...]
        Evaluated components of a formulae term.
    frame : pandas.DataFrame
        The rows the term was evaluated on.

    Returns
    -------
    bool
        Whether a component changed its coding.

    Raises
    ------
    ValueError
        If an ordered factor has more than 95 levels.
    """
    changed = False
    for component in components:
        if (
            not isinstance(component, Variable)
            or component.kind != "categoric"
            or component.spans_intercept
            or not _ordered(frame, component.name)
            or len(component.levels or ()) < 2
        ):
            continue
        count = len(component.levels)
        component.contrast_matrix = ContrastMatrix(
            contr_poly(count), poly_labels(count)
        )
        changed = True
    return changed


def apply_ordered_contrasts(
    design: DesignMatrices, frame: pd.DataFrame
) -> DesignMatrices:
    """Code ordered factors by R's polynomial contrasts in an evaluated design.

    Every population-level and group-level term component that is a pandas
    ordered categorical coded by contrasts (not by indicator columns) takes
    ``contr.poly``. The affected terms are evaluated again on the same rows
    through their retained state, so new rows are coded by the same
    contrasts. Cost O(n p) for ``n`` rows and ``p`` columns.

    Parameters
    ----------
    design : formulae.matrices.DesignMatrices
        Design evaluated on ``frame``.
    frame : pandas.DataFrame
        The rows it was evaluated on.

    Returns
    -------
    formulae.matrices.DesignMatrices
        The same object with the polynomial codings.

    Raises
    ------
    ValueError
        If an ordered factor has more than 95 levels.
    """
    rows = len(frame)
    if design.common is not None:
        common = design.common
        changed = [
            _recode(tuple(getattr(term, "components", ())), frame)
            for term in common.terms.values()
        ]
        if any(changed):
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
    if design.group is not None:
        for term in design.group.terms.values():
            expression = getattr(term, "expr", None)
            if _recode(tuple(getattr(expression, "components", ())), frame):
                expression.data = np.asarray(
                    expression.eval_new_data(frame), dtype=np.float64
                )
    return design


__all__: list[str] = []
