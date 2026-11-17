"""R's ``scale`` and ``splines::ns`` as stateful formula transforms.

Both retain their training state, so a fitted design evaluates new rows with
the training mean and deviation, or the training knots, as R's
``predict`` methods do.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.interpolate import BSpline

from quivers.formulas._formula_options import R_SCALE


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
    knots: np.ndarray, x: np.ndarray, derivatives: int = 0
) -> np.ndarray:
    """Evaluate cubic B-splines or their derivatives, as R's ``splineDesign``.

    Parameters
    ----------
    knots : np.ndarray
        Full knot vector with four repeated boundary knots on each side.
    x : np.ndarray
        Points inside the boundary knots.
    derivatives : int
        Derivative order.

    Returns
    -------
    np.ndarray
        ``(len(x), len(knots) - 4)`` basis values.
    """
    count = len(knots) - 4
    spline = BSpline(knots, np.eye(count), 3, extrapolate=True)
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


def r_transform_namespace() -> Mapping[str, type]:
    """Return the R transforms the formula namespace adds.

    Returns
    -------
    Mapping[str, type]
        R's ``scale`` under its reserved name and ``ns``.
    """
    return {R_SCALE: RScale, "ns": NaturalSpline}


__all__ = ["NaturalSpline", "RScale", "r_transform_namespace"]
