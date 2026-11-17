"""R's ``scale``, ``splines::ns``, ``splines::bs`` and ``factor`` as stateful transforms.

Each retains its training state, so a fitted design evaluates new rows with
the training mean and deviation, the training knots, or the training levels,
as R's ``predict`` methods do.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from quivers.formulas._formula_options import R_BS, R_SCALE


class RScale:
    """R's ``scale(x)``: subtract the mean and divide by the n - 1 deviation.

    The formula parser's own ``scale`` divides by the n deviation; brms reads
    formulas with R's.
    """

    __stateful_transform__ = True

    def __init__(self) -> None:
        """Start without a retained mean and deviation."""
        self.mean: float | None = None
        self.deviation: float | None = None

    def __call__(
        self, x: np.ndarray, center: bool = True, scale: bool = True
    ) -> np.ndarray:
        """Standardize ``x`` with the training mean and deviation.

        Parameters
        ----------
        x : np.ndarray
            Values.
        center : bool
            Subtract the training mean, as R's ``center = TRUE``.
        scale : bool
            Divide by the training deviation (root mean square when not
            centered), as R's ``scale = TRUE``.

        Returns
        -------
        np.ndarray
            Transformed values.

        Raises
        ------
        ValueError
            If the training deviation is zero or not finite.
        """
        values = np.asarray(x, dtype=np.float64)
        if self.mean is None or self.deviation is None:
            self.mean = float(values.mean()) if center else 0.0
            shifted = values - self.mean
            # R: sd for centered columns, root mean square (n - 1) otherwise.
            self.deviation = (
                float(np.sqrt(np.sum(shifted**2) / (len(values) - 1))) if scale else 1.0
            )
            if not np.isfinite(self.deviation) or self.deviation == 0:
                raise ValueError("scale() needs a column with nonzero deviation")
        return (values - self.mean) / self.deviation


def _householder_qr(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factor a full-rank matrix as LINPACK ``dqrdc2`` does, without pivoting.

    Parameters
    ----------
    matrix : np.ndarray
        ``(n, p)`` matrix of full column rank.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The compact factor (Householder vectors below the diagonal, R on and
        above it) and ``qraux``, as R's ``qr()`` returns them.
    """
    x = np.array(matrix, dtype=np.float64)
    n, p = x.shape
    qraux = np.zeros(p)
    for column in range(min(n - 1, p)):
        norm = float(np.linalg.norm(x[column:, column]))
        if norm == 0.0:
            continue
        if x[column, column] != 0.0:
            norm = float(np.copysign(norm, x[column, column]))
        x[column:, column] /= norm
        x[column, column] += 1.0
        for other in range(column + 1, p):
            t = -float(x[column:, column] @ x[column:, other]) / x[column, column]
            x[column:, other] += t * x[column:, column]
        qraux[column] = x[column, column]
        x[column, column] = -norm
    return x, qraux


def _apply_qt(factor: np.ndarray, qraux: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute ``Q.T @ y`` as LINPACK ``dqrsl`` (R's ``qr.qty``) does.

    Parameters
    ----------
    factor : np.ndarray
        Compact factor from `_householder_qr`.
    qraux : np.ndarray
        Its auxiliary vector.
    y : np.ndarray
        ``(n, m)`` right-hand sides.

    Returns
    -------
    np.ndarray
        ``Q.T @ y`` with the full ``n`` by ``n`` orthogonal ``Q``, applying
        the Householder reflections in factorization order.
    """
    result = np.array(y, dtype=np.float64)
    n, p = factor.shape
    for column in range(min(p, n - 1)):
        if qraux[column] == 0.0:
            continue
        vector = factor[column:, column].copy()
        vector[0] = qraux[column]
        t = -(vector @ result[column:]) / vector[0]
        result[column:] += np.outer(vector, t)
    return result


def _spline_design(
    knots: np.ndarray, x: np.ndarray, derivatives: int = 0, degree: int = 3
) -> np.ndarray:
    """Evaluate B-splines or their derivatives, as R's ``splineDesign``.

    Parameters
    ----------
    knots : np.ndarray
        Full knot vector with ``degree + 1`` repeated boundary knots on each
        side.
    x : np.ndarray
        Points inside the boundary knots.
    derivatives : int
        Derivative order.
    degree : int
        Polynomial degree of the pieces, cubic by default.

    Returns
    -------
    np.ndarray
        ``(len(x), len(knots) - degree - 1)`` basis values.
    """
    count = len(knots) - degree - 1
    spline = BSpline(knots, np.eye(count), degree, extrapolate=True)
    if derivatives:
        spline = spline.derivative(derivatives)
    return np.atleast_2d(spline(np.asarray(x, dtype=np.float64)))


class NaturalSpline:
    """``splines::ns``: the natural cubic spline basis of R.

    The basis is the cubic B-spline basis on the boundary and interior knots,
    projected onto the subspace with zero second derivative at both boundary
    knots through R's Householder QR of the constraint (``qr.qty``), dropping the first
    column unless ``intercept``. Points outside the boundary knots continue
    linearly, as in ``predict.ns``. Evaluation costs O(N K) for N rows and
    K basis columns.
    """

    __stateful_transform__ = True

    def __init__(self) -> None:
        """Start without retained knots."""
        self.knots: np.ndarray | None = None
        self.boundary: tuple[float, float] | None = None
        self.intercept = False

    def __call__(
        self,
        x: np.ndarray,
        df: int | None = None,
        knots: tuple[float, ...] | list[float] | None = None,
        intercept: bool = False,
        Boundary_knots: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Evaluate the basis with the training knots.

        Parameters
        ----------
        x : np.ndarray
            Values.
        df : int or None
            Degrees of freedom; ``df - 1 - intercept`` interior knots at
            equally spaced quantiles of the training values.
        knots : sequence of float or None
            Explicit interior knots.
        intercept : bool
            Keep the column R drops for the intercept.
        Boundary_knots : tuple[float, float] or None
            Boundary knots, the training range by default.

        Returns
        -------
        np.ndarray
            ``(N, df)`` basis.

        Raises
        ------
        ValueError
            If neither or both ``df`` and ``knots`` imply no basis, or ``df``
            is too small.
        """
        values = np.asarray(x, dtype=np.float64)
        if self.knots is None:
            boundary = (
                (float(values.min()), float(values.max()))
                if Boundary_knots is None
                else (float(Boundary_knots[0]), float(Boundary_knots[1]))
            )
            if knots is not None:
                interior = np.asarray(knots, dtype=np.float64)
            elif df is not None:
                count = int(df) - 1 - int(bool(intercept))
                if count < 0:
                    raise ValueError(f"ns(): df = {df} is too small")
                inside = values[(values >= boundary[0]) & (values <= boundary[1])]
                interior = (
                    np.quantile(inside, np.linspace(0.0, 1.0, count + 2)[1:-1])
                    if count
                    else np.zeros(0)
                )
            else:
                interior = np.zeros(0)
            self.knots = np.sort(np.concatenate([np.repeat(boundary, 4), interior]))
            self.boundary = boundary
            self.intercept = bool(intercept)
        return self._basis(values)

    def _basis(self, values: np.ndarray) -> np.ndarray:
        """Evaluate the projected basis at training or new values.

        Parameters
        ----------
        values : np.ndarray
            Points.

        Returns
        -------
        np.ndarray
            Natural spline basis.
        """
        assert self.knots is not None and self.boundary is not None
        lower, upper = self.boundary
        count = len(self.knots) - 4
        basis = np.zeros((len(values), count))
        inside = (values >= lower) & (values <= upper)
        if inside.any():
            basis[inside] = _spline_design(self.knots, values[inside])
        for pivot, mask in ((lower, values < lower), (upper, values > upper)):
            if mask.any():
                tangent = np.vstack(
                    [
                        _spline_design(self.knots, np.array([pivot])),
                        _spline_design(self.knots, np.array([pivot]), 1),
                    ]
                )
                basis[mask] = (
                    np.column_stack([np.ones(mask.sum()), values[mask] - pivot])
                    @ tangent
                )
        constraint = _spline_design(self.knots, np.array([lower, upper]), 2)
        if not self.intercept:
            constraint = constraint[:, 1:]
            basis = basis[:, 1:]
        factor, qraux = _householder_qr(constraint.T)
        return _apply_qt(factor, qraux, basis.T).T[:, 2:]


class BSplineBasis:
    """``splines::bs``: the B-spline basis of R for a polynomial spline.

    The basis is the B-spline basis of the given degree on the boundary and
    interior knots, without its first column unless ``intercept``. Points
    outside the boundary knots continue the boundary polynomial pieces through
    their Taylor expansion about a pivot one quarter of the way from the
    boundary knot to the nearest interior knot, as in ``predict.bs``.
    Evaluation costs O(N K) for N rows and K basis columns.
    """

    __stateful_transform__ = True

    def __init__(self) -> None:
        """Start without retained knots."""
        self.knots: np.ndarray | None = None
        self.boundary: tuple[float, float] | None = None
        self.degree = 3
        self.intercept = False

    def __call__(
        self,
        x: np.ndarray,
        df: int | None = None,
        knots: tuple[float, ...] | list[float] | None = None,
        degree: int = 3,
        intercept: bool = False,
        Boundary_knots: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Evaluate the basis with the training knots.

        Parameters
        ----------
        x : np.ndarray
            Values.
        df : int or None
            Degrees of freedom; ``df - degree - intercept`` interior knots at
            equally spaced quantiles of the training values inside the
            boundary knots. An interior knot that falls on a boundary knot
            moves inside by an eighth of the distance to the nearest distinct
            interior knot, as R moves it.
        knots : sequence of float or None
            Explicit interior knots, which take precedence over ``df``.
        degree : int
            Polynomial degree of the pieces.
        intercept : bool
            Keep the column R drops for the intercept.
        Boundary_knots : tuple[float, float] or None
            Boundary knots, the training range by default.

        Returns
        -------
        np.ndarray
            ``(N, K)`` basis with ``K`` the number of interior knots plus
            ``degree + intercept``.

        Raises
        ------
        ValueError
            If ``degree`` is not a positive integer, ``df`` is too small, a
            knot is not finite, or every interior knot chosen from ``df``
            falls on one boundary knot.
        """
        values = np.asarray(x, dtype=np.float64)
        if self.knots is None:
            if int(degree) != degree or int(degree) < 1:
                raise ValueError(f"bs(): degree = {degree} must be an integer >= 1")
            order = int(degree) + 1
            if Boundary_knots is None:
                boundary = (float(values.min()), float(values.max()))
            else:
                first, second = sorted(float(knot) for knot in Boundary_knots)
                boundary = (first, second)
            if knots is not None:
                interior = np.asarray(knots, dtype=np.float64).reshape(-1)
                if not np.isfinite(interior).all():
                    raise ValueError("bs(): knots must be finite")
            elif df is not None:
                count = int(df) - order + 1 - int(bool(intercept))
                if count < 0:
                    raise ValueError(
                        f"bs(): df = {df} is too small for degree = {degree}"
                    )
                inside = values[(values >= boundary[0]) & (values <= boundary[1])]
                interior = _inside_boundary(
                    np.quantile(inside, np.linspace(0.0, 1.0, count + 2)[1:-1])
                    if count
                    else np.zeros(0),
                    boundary,
                )
            else:
                interior = np.zeros(0)
            self.knots = np.sort(np.concatenate([np.repeat(boundary, order), interior]))
            self.boundary = boundary
            self.degree = int(degree)
            self.intercept = bool(intercept)
        return self._basis(values)

    def _basis(self, values: np.ndarray) -> np.ndarray:
        """Evaluate the basis at training or new values.

        Parameters
        ----------
        values : np.ndarray
            Points.

        Returns
        -------
        np.ndarray
            B-spline basis.
        """
        assert self.knots is not None and self.boundary is not None
        lower, upper = self.boundary
        order = self.degree + 1
        basis = np.zeros((len(values), len(self.knots) - order))
        inside = (values >= lower) & (values <= upper)
        if inside.any():
            basis[inside] = _spline_design(
                self.knots, values[inside], degree=self.degree
            )
        scale = np.array([math.factorial(power) for power in range(order)])
        for boundary, neighbor, mask in (
            (lower, self.knots[order], values < lower),
            (upper, self.knots[len(self.knots) - order - 1], values > upper),
        ):
            if mask.any():
                pivot = 0.75 * boundary + 0.25 * neighbor
                taylor = np.vstack(
                    [
                        _spline_design(
                            self.knots, np.array([pivot]), power, self.degree
                        )
                        for power in range(order)
                    ]
                )
                powers = np.power.outer(values[mask] - pivot, np.arange(order))
                basis[mask] = powers @ (taylor / scale[:, None])
        return basis if self.intercept else basis[:, 1:]


def _inside_boundary(interior: np.ndarray, boundary: tuple[float, float]) -> np.ndarray:
    """Move quantile knots that fall on a boundary knot inside, as R's ``bs``.

    Parameters
    ----------
    interior : np.ndarray
        Sorted interior knots chosen at quantiles.
    boundary : tuple[float, float]
        Lower and upper boundary knots.

    Returns
    -------
    np.ndarray
        The knots, with each one equal to the lower boundary knot raised by an
        eighth of the distance to the smallest knot above it, and each one
        equal to the upper boundary knot lowered by an eighth of the distance
        to the largest knot below it.

    Raises
    ------
    ValueError
        If every interior knot equals one boundary knot, which leaves no
        distinct knot to move towards.
    """
    result = np.array(interior, dtype=np.float64)
    if not result.size:
        return result
    lower, upper = boundary
    if result.min() == lower:
        on_boundary = result == lower
        if on_boundary.all():
            raise ValueError("bs(): all interior knots match the left boundary knot")
        result[on_boundary] += (result[~on_boundary].min() - lower) / 8.0
    if result.max() == upper:
        on_boundary = result == upper
        if on_boundary.all():
            raise ValueError("bs(): all interior knots match the right boundary knot")
        result[on_boundary] -= (upper - result[~on_boundary].max()) / 8.0
    return result


class RFactor:
    """R's ``factor(x)``: read a column as an unordered factor.

    The levels are the sorted distinct training values, as R sorts them, or
    the ``levels`` argument in the order given. They are retained, so new
    rows are coded against the training levels and a value outside them is
    refused, as R's ``predict`` refuses a new factor level.
    """

    __stateful_transform__ = True

    def __init__(self) -> None:
        """Start without retained levels."""
        self.levels: tuple[str | float | int | bool, ...] | None = None

    def __call__(
        self,
        x: np.ndarray | pd.Series | pd.Categorical,
        levels: Sequence[str | float | int | bool] | np.ndarray | None = None,
        ordered: bool = False,
    ) -> pd.Categorical:
        """Code ``x`` as a categorical variable over the training levels.

        Parameters
        ----------
        x : np.ndarray or pd.Series or pd.Categorical
            Values of one column.
        levels : Sequence[str or float or int or bool] or np.ndarray or None
            Levels in order, as R's ``levels``; None sorts the distinct
            values. The first level is the reference level.
        ordered : bool
            R's ``ordered``. Only False is accepted.

        Returns
        -------
        pd.Categorical
            The values as a categorical whose categories are the retained
            levels in their order.

        Raises
        ------
        ValueError
            If ``ordered`` is true, a value is missing, ``levels`` repeats a
            level or omits an observed value, or a new row holds a value
            that is not a training level.
        """
        if ordered:
            raise ValueError(
                "factor(..., ordered = TRUE) is not supported in a formula; "
                "store the column as an ordered pandas Categorical, which is "
                "coded by polynomial contrasts"
            )
        values = pd.Series(
            x.astype(object) if isinstance(x, pd.Categorical) else np.asarray(x)
        )
        if bool(values.isna().any()):
            raise ValueError("factor() received a missing value")
        if self.levels is None:
            if levels is None:
                chosen = sorted(values.unique().tolist())
            else:
                chosen = np.asarray(levels).tolist()
                if len(set(chosen)) != len(chosen):
                    raise ValueError(f"factor() levels {chosen!r} repeat a level")
            self.levels = tuple(chosen)
        unknown = sorted(set(values.unique().tolist()) - set(self.levels))
        if unknown:
            raise ValueError(
                f"factor() received values {unknown!r} outside its levels "
                f"{list(self.levels)!r}"
            )
        # The formula parser sorts the categories of an unordered categorical;
        # declaring the order keeps the levels as retained. The term is still
        # coded by treatment contrasts against the first level.
        return pd.Categorical(values, categories=list(self.levels), ordered=True)


def r_transform_namespace() -> Mapping[str, type]:
    """Return the R transforms the formula namespace adds.

    Returns
    -------
    Mapping[str, type]
        R's ``scale`` and ``bs`` under their reserved names, ``ns`` and
        ``factor``.
    """
    return {
        R_SCALE: RScale,
        R_BS: BSplineBasis,
        "ns": NaturalSpline,
        "factor": RFactor,
    }


__all__ = [
    "BSplineBasis",
    "NaturalSpline",
    "RFactor",
    "RScale",
    "r_transform_namespace",
]
