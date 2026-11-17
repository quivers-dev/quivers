"""mgcv smooth bases, absorbed constraints, and mixed-model coordinates.

This module ports the parts of mgcv [@wood2017generalized] that brms runs to
turn ``s()`` and ``t2()`` terms into Stan data. For every smooth, brms calls
``smoothCon(..., absorb.cons = TRUE, diagonal.penalty = TRUE)`` and then
``smooth2random(..., type = 2)``. The result is an unpenalized design with
ordinary population coefficients and one or more penalized designs whose
coefficients are independent standard-Normal innovations multiplied by a
smoothing standard deviation [@wood2017generalized].

Each basis constructor retains exactly the state mgcv's prediction method
reads (knots, eigenvector maps, marginal reparameterizations), so fresh rows
are evaluated through the training basis and never through a refitted one.
The composite map from the raw basis to the mixed-model coordinates is one
fixed matrix, recorded per smooth block.

Where mgcv's result depends on the orthonormal basis LAPACK returns for a
repeated eigenvalue (a penalty null space of dimension at least two), this
module returns that basis where the solver determines it and otherwise the
principal axes of the null-space design columns (`_canonical_null`). Every
eigenvector sign is chosen so that its largest-magnitude entry is positive.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import didactic.api as dx
import numpy as np
import numpy.typing as npt
from scipy.interpolate import BSpline
from scipy.linalg import eigh_tridiagonal

from quivers._values import OPAQUE_BY_VALUE, ValueModel, opaque_by_value

type FloatArray = npt.NDArray[np.float64]

_EPS = float(np.finfo(np.float64).eps)
_TINY = float(np.finfo(np.float64).tiny)
_MAX_KNOTS = 2000
#: Largest entry of the base polynomials outside the computed penalty null
#: space of a factor-smooth base, relative to their largest entry. The null
#: vectors come from a symmetric eigendecomposition and are accurate to a
#: small multiple of machine epsilon times the penalty's condition number.
_FS_NULL_TOLERANCE = 1e-7
#: Relative gap below which two eigenvalues of a Markov random field penalty
#: count as repeated when a low-rank basis is cut between them.
_MRF_GAP = 1e-8
_BASES = (
    "tp",
    "ts",
    "cr",
    "cs",
    "cc",
    "ps",
    "bs",
    "re",
    "fs",
    "gp",
    "mrf",
    "sos",
)


def _frozen(values: FloatArray) -> FloatArray:
    """Copy an array into an immutable C-contiguous Double buffer.

    Parameters
    ----------
    values : numpy.ndarray
        Real values of any shape.

    Returns
    -------
    numpy.ndarray
        Read-only copy.
    """
    frozen = np.array(values, dtype=np.float64, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen


def _array_field() -> FloatArray:
    """Declare an immutable array field compared by content.

    Returns
    -------
    numpy.ndarray
        The didactic field descriptor, typed as its held value.
    """
    return dx.field(opaque=True, extras=opaque_by_value(_frozen))


def _signed_eigh(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Eigendecompose a symmetric matrix in descending order with fixed signs.

    Parameters
    ----------
    matrix : numpy.ndarray
        Real symmetric matrix.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Eigenvalues in descending order and eigenvectors whose
        largest-magnitude entry is positive.
    """
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    values, vectors = values[::-1], vectors[:, ::-1]
    return values, _fix_signs(vectors)


def _fix_signs(vectors: FloatArray) -> FloatArray:
    """Choose each column's sign so its largest-magnitude entry is positive.

    Parameters
    ----------
    vectors : numpy.ndarray
        Matrix whose columns have arbitrary sign.

    Returns
    -------
    numpy.ndarray
        Sign-normalized copy.
    """
    if not vectors.size:
        return vectors.copy()
    pivot = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[pivot, np.arange(vectors.shape[1])])
    signs[signs == 0] = 1.0
    return vectors * signs


def _canonical_null(design: FloatArray, vectors: FloatArray, rank: int) -> FloatArray:
    """Fix the orthonormal basis of a penalty's repeated zero eigenvalue.

    LAPACK's symmetric eigensolver, as R's ``eigen`` calls it, first reduces
    the matrix to tridiagonal form by reflections that fix the first
    coordinate axis. Rows and columns that vanish stay decoupled, so a null
    space spanned by coordinate axes is returned as those axes. Otherwise the
    tridiagonal matrix splits where the Krylov space of the first axis is
    exhausted, so for a two-dimensional null space one returned null vector
    is the projection of the first axis onto the null space and the other is
    the null vector orthogonal to that axis; this function returns that pair,
    the orthogonal vector first. Any other null space splits at
    rounding-dependent points, so mgcv's basis is unspecified there; this
    function then uses the principal axes of the null-space design columns.

    Parameters
    ----------
    design : numpy.ndarray
        Rows-by-coordinates design evaluated in the original coordinates.
    vectors : numpy.ndarray
        Eigenvectors in descending eigenvalue order.
    rank : int
        Number of leading eigenvectors spanning the penalized range.

    Returns
    -------
    numpy.ndarray
        The same eigenvectors with the trailing null block fixed.
    """
    null = vectors[:, rank:]
    if null.shape[1] < 2:
        return vectors
    rotated = vectors.copy()
    axes_rows = np.flatnonzero(np.linalg.norm(null, axis=1) > 1 - np.sqrt(_EPS))
    if axes_rows.size == null.shape[1]:
        rotated[:, rank:] = np.eye(vectors.shape[0])[:, axes_rows[::-1]]
        return rotated
    first = null[0]
    if null.shape[1] == 2 and float(np.linalg.norm(first)) > _EPS:
        lead = null @ (first / np.linalg.norm(first))
        trail = null @ (np.array([-first[1], first[0]]) / np.linalg.norm(first))
        rotated[:, rank:] = _fix_signs(np.column_stack((trail, lead)))
        return rotated
    projected = design @ null
    _, axes = _signed_eigh(projected.T @ projected)
    rotated[:, rank:] = _fix_signs(null @ axes)
    return rotated


# ---------------------------------------------------------------------------
# Thin-plate regression splines (Wood 2003; mgcv tprs.c).


def _eta_constant(m: int, d: int) -> float:
    """Return the thin-plate radial basis constant of mgcv ``eta_const``.

    Parameters
    ----------
    m, d : int
        Penalty order and covariate dimension, with ``2 m > d``.

    Returns
    -------
    float
        Constant multiplying the radial basis function.
    """
    pi = math.pi
    gamma_half = math.sqrt(pi)
    d2 = d // 2
    if d % 2 == 0:
        f = -1.0 if (m + 1 + d2) % 2 else 1.0
        for _ in range(2 * m - 1):
            f /= 2
        for _ in range(d2):
            f /= pi
        for i in range(2, m):
            f /= i
        for i in range(2, m - d2 + 1):
            f /= i
        return f
    f = gamma_half
    for i in range(m - (d - 1) // 2):
        f /= -0.5 - i
    for _ in range(m):
        f /= 4
    for _ in range(d2):
        f /= pi
    f /= gamma_half
    for i in range(2, m):
        f /= i
    return f


def _eta(squared: FloatArray, m: int, d: int) -> FloatArray:
    """Evaluate the thin-plate radial basis at squared distances.

    Parameters
    ----------
    squared : numpy.ndarray
        Nonnegative squared Euclidean distances.
    m, d : int
        Penalty order and covariate dimension.

    Returns
    -------
    numpy.ndarray
        Radial basis values, zero at zero distance.
    """
    constant = _eta_constant(m, d)
    d2 = d // 2
    positive = squared > 0
    safe = np.where(positive, squared, 1.0)
    if d % 2 == 0:
        values = constant * np.log(safe) * 0.5 * safe ** (m - d2)
    else:
        values = constant * safe ** (m - d2 - 1) * np.sqrt(safe)
    return np.where(positive, values, 0.0)


def _effective_order(m: int, d: int) -> int:
    """Raise a thin-plate penalty order to mgcv's smallest valid order.

    Parameters
    ----------
    m, d : int
        Requested order (zero for the default) and dimension.

    Returns
    -------
    int
        ``m`` itself when ``2 m > d``, otherwise the smallest order with
        ``2 m > d + 1``.
    """
    if 2 * m <= d:
        m = 0
        while 2 * m < d + 2:
            m += 1
    return m


def _poly_powers(m: int, d: int) -> tuple[tuple[int, ...], ...]:
    """List the monomial powers spanning the thin-plate penalty null space.

    Parameters
    ----------
    m, d : int
        Effective penalty order and dimension.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        One power vector per null-space polynomial, in mgcv's order.
    """
    count = math.comb(m + d - 1, d)
    index = [0] * d
    powers = []
    for _ in range(count):
        powers.append(tuple(index))
        total = sum(index)
        if total < m - 1:
            index[0] += 1
        else:
            total -= index[0]
            index[0] = 0
            for j in range(1, d):
                index[j] += 1
                total += 1
                if total == m:
                    total -= index[j]
                    index[j] = 0
                else:
                    break
    return tuple(powers)


def _poly(x: FloatArray, m: int) -> FloatArray:
    """Evaluate the thin-plate null-space polynomials at rows of ``x``.

    Parameters
    ----------
    x : numpy.ndarray
        Centered covariate rows, shape rows by dimension.
    m : int
        Effective penalty order.

    Returns
    -------
    numpy.ndarray
        Rows by null-space dimension.
    """
    powers = _poly_powers(m, x.shape[1])
    return np.column_stack(
        [np.prod(x ** np.asarray(power, dtype=np.float64), axis=1) for power in powers]
    )


def _squared_distances(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pairwise squared Euclidean distances between two row sets.

    Parameters
    ----------
    a, b : numpy.ndarray
        Rows by dimension.

    Returns
    -------
    numpy.ndarray
        ``len(a)`` by ``len(b)`` squared distances.
    """
    difference = a[:, None, :] - b[None, :, :]
    return np.einsum("ijk,ijk->ij", difference, difference)


def _lanczos(
    matrix: FloatArray, count: int, tolerance: float = _EPS**0.7
) -> tuple[FloatArray, FloatArray]:
    """Port mgcv's ``Rlanczos`` for the largest-magnitude eigenpairs.

    mgcv's thin-plate basis uses the Ritz vectors at which this iteration
    stops, so the port keeps its deterministic start vector, full
    reorthogonalization, check schedule, and stopping test exactly. Costs
    O(u squared) per iteration for a ``u`` by ``u`` matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Symmetric matrix.
    count : int
        Number of eigenpairs.
    tolerance : float
        Relative residual at which the iteration stops: mgcv's thin-plate
        code uses ``eps ** 0.7`` and R's ``slanczos`` default is
        ``eps ** 0.5``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Ritz values and vectors in mgcv's order: the selected largest values
        in descending order, then the selected smallest in descending order.
    """
    n = matrix.shape[0]
    f_check = max(10, count // 2)
    f_check = min(f_check, max(1, n // 10))
    start = np.empty(n)
    jran = 1
    for i in range(n):
        jran = (jran * 106 + 1283) % 6075
        start[i] = jran / 6075 - 0.5
    q = [start / np.linalg.norm(start)]
    alpha = np.zeros(n)
    beta = np.zeros(n)
    top, bottom = count, 0
    ritz_values = np.zeros(0)
    ritz_vectors = np.zeros((0, 0))
    j = 0
    while j < n:
        z = matrix @ q[j]
        alpha[j] = q[j] @ z
        if j == 0:
            z = z - alpha[j] * q[j]
        else:
            z = z - (alpha[j] * q[j] + beta[j - 1] * q[j - 1])
            basis = np.array(q[: j + 1])
            for _ in range(2):
                for row in basis:
                    z = z - (z @ row) * row
        beta[j] = float(np.sqrt(z @ z))
        if j < n - 1:
            q.append(z / beta[j])
        if (j >= count and j % f_check == 0) or j == n - 1:
            ritz_values, ritz_vectors = eigh_tridiagonal(alpha[: j + 1], beta[:j])
            ritz_values, ritz_vectors = ritz_values[::-1], ritz_vectors[:, ::-1]
            norm_t = max(abs(ritz_values[0]), abs(ritz_values[j]))
            errors = np.abs(beta[j] * ritz_vectors[j, :])
            if j >= count:
                limit = norm_t * tolerance
                upper = lower = 0
                converged = True
                while upper + lower < count:
                    # mgcv tests the lower end's error at index ``lower``.
                    if abs(ritz_values[upper]) >= abs(ritz_values[j - lower]):
                        if errors[upper] > limit:
                            converged = False
                            break
                        upper += 1
                    else:
                        if errors[lower] > limit:
                            converged = False
                            break
                        lower += 1
                if converged:
                    top, bottom = upper, lower
                    j += 1
                    break
        j += 1
    krylov = np.array(q[:j]).T
    chosen = list(range(top)) + [
        j - (bottom + top - k) for k in range(top, top + bottom)
    ]
    values = ritz_values[chosen]
    vectors = krylov @ ritz_vectors[:j, chosen]
    return values, vectors


def _qt(constraint: FloatArray) -> FloatArray:
    """Form mgcv's Householder ``QT`` factor ``A Q = [0, T]`` explicitly.

    Parameters
    ----------
    constraint : numpy.ndarray
        Rows-by-columns matrix with no more rows than columns.

    Returns
    -------
    numpy.ndarray
        Orthogonal ``Q``, whose leading ``columns - rows`` columns span the
        constraint null space.
    """
    a = constraint.astype(np.float64, copy=True)
    rows, columns = a.shape
    q = np.eye(columns)
    for i in range(rows):
        width = columns - i
        p = a[i, :width].copy()
        scale = float(np.max(np.abs(p))) if width else 0.0
        if scale:
            p /= scale
        length = float(np.sqrt(p @ p))
        if p[width - 1] < 0.0:
            length = -length
        p[width - 1] += length
        g = 1.0 / (length * p[width - 1]) if length else 0.0
        for j in range(i + 1, rows):
            a[j, :width] -= g * (a[j, :width] @ p) * p
        q[:, :width] -= g * np.outer(q[:, :width] @ p, p)
        a[i, : width - 1] = 0.0
        a[i, width - 1] = -length * scale
    return q


class ThinPlateBasis(ValueModel):
    """Retained thin-plate regression spline basis of mgcv ``tprs.smooth``.

    Attributes
    ----------
    shift : tuple[float, ...]
        Training covariate means subtracted before evaluation.
    knots : numpy.ndarray
        Unique centered training covariate rows.
    transform : numpy.ndarray
        Map from radial and polynomial evaluations to basis coordinates.
    order : int
        Penalty order as authored, zero for the default.
    drop_null : int
        Number of dropped null-space columns, zero when kept.
    column_means : tuple[float, ...]
        Training column means removed when the null space is dropped.
    """

    kind: Literal["tp"] = "tp"
    shift: tuple[float, ...]
    knots: FloatArray = _array_field()
    transform: FloatArray = _array_field()
    order: int
    drop_null: int = 0
    column_means: tuple[float, ...] = ()

    @property
    def width(self) -> int:
        """Number of raw basis columns.

        Returns
        -------
        int
            Basis dimension after any dropped null space.
        """
        return self.transform.shape[1] - self.drop_null

    def evaluate(self, x: FloatArray) -> FloatArray:
        """Evaluate the retained basis at fresh covariate rows.

        Costs O(rows times knots times dimension) for the radial basis and
        O(rows times knots times width) for the retained map.

        Parameters
        ----------
        x : numpy.ndarray
            Covariate rows, shape rows by dimension.

        Returns
        -------
        numpy.ndarray
            Rows by basis dimension.
        """
        centered = x - np.asarray(self.shift)
        d = centered.shape[1]
        m = _effective_order(self.order, d)
        radial = _eta(_squared_distances(centered, self.knots), m, d)
        values = np.column_stack((radial, _poly(centered, m))) @ self.transform
        if self.drop_null:
            values = values[:, : self.width] - np.asarray(self.column_means)
        return values


def thin_plate(
    x: FloatArray, k: int, m: int, *, shrink: float | None, drop_null: bool = False
) -> RawSmooth:
    """Construct mgcv's thin-plate regression spline [@wood2003thin].

    The basis is the eigen-truncation of the full thin-plate spline at the
    unique covariate rows: the ``k`` largest-magnitude eigenvectors of the
    radial basis matrix, constrained orthogonal to the null-space
    polynomials, followed by those polynomials, with columns rescaled to unit
    root mean square. Costs O(u cubed) for ``u`` unique rows.

    Parameters
    ----------
    x : numpy.ndarray
        Training covariate rows, shape rows by dimension.
    k : int
        Basis dimension, or a negative value for mgcv's default.
    m : int
        Penalty order, zero for the default.
    shrink : float or None
        Null-space shrinkage factor of ``bs = 'ts'``, or None.
    drop_null : bool
        Drop the null-space polynomials, as ``m = c(m, 0)`` requests.

    Returns
    -------
    RawSmooth
        Basis, training design, and scaled-free raw penalty.

    Raises
    ------
    ValueError
        If ``k`` is too small, or the covariate rows have too few unique
        combinations, or more than mgcv's exact-eigenproblem limit.
    """
    n, d = x.shape
    shift = x.mean(axis=0)
    centered = x - shift
    m_eff = _effective_order(m, d)
    null_dim = math.comb(m_eff + d - 1, d)
    if k < 0:
        k = null_dim + (8, 27, 100)[min(d, 3) - 1]
    if k < null_dim + 1:
        raise ValueError(
            f"thin-plate smooth k={k} must exceed its penalty null-space dimension {null_dim}"
        )
    knots, inverse = np.unique(centered, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    unique = knots.shape[0]
    if unique > _MAX_KNOTS and n > _MAX_KNOTS:
        raise ValueError(
            f"thin-plate smooth has {unique} unique covariate combinations; mgcv "
            f"subsamples knots above {_MAX_KNOTS} with R's random number generator, "
            "which QVR does not reproduce: use bs='cr' or bs='ps' for this term"
        )
    if unique < k:
        raise ValueError(
            "a thin-plate smooth has fewer unique covariate combinations than its basis dimension k"
        )
    radial = _eta(_squared_distances(knots, knots), m_eff, d)
    polynomial = _poly(knots, m_eff)
    if unique == k:
        q = _qt(polynomial.T)
        transform = np.zeros((unique + null_dim, k))
        transform[:unique, : k - null_dim] = q[:, : k - null_dim]
        transform[unique:, k - null_dim :] = np.eye(null_dim)
        rows = np.column_stack(
            (
                _eta(_squared_distances(centered, knots), m_eff, d),
                _poly(centered, m_eff),
            )
        )
        design = rows @ transform
        penalty = q.T @ radial @ q
    else:
        values, vectors = _lanczos(radial, k)
        q = _qt(polynomial.T @ vectors)
        transform = np.zeros((unique + null_dim, k))
        transform[:unique, : k - null_dim] = (vectors @ q)[:, : k - null_dim]
        transform[unique:, k - null_dim :] = np.eye(null_dim)
        reduced = (vectors * values) @ q
        reduced[:, k - null_dim :] = polynomial
        design = reduced[inverse]
        penalty = q.T @ np.diag(values) @ q
    penalty[k - null_dim :, :] = 0.0
    penalty[:, k - null_dim :] = 0.0
    scale = np.sqrt(np.mean(design * design, axis=0))
    design = design / scale
    transform = transform / scale
    penalty = penalty / np.outer(scale, scale)
    penalty = (penalty + penalty.T) * 0.5
    rank = k - null_dim
    if shrink is not None:
        values, vectors = _signed_eigh(penalty)
        values[k - null_dim :] = values[k - null_dim - 1] * shrink
        penalty = (vectors * values) @ vectors.T
        rank, null_dim = k, 0
    if drop_null:
        width = k - null_dim
        design = design[:, :width]
        penalty = penalty[:width, :width]
        means = design.mean(axis=0)
        basis = ThinPlateBasis(
            shift=tuple(shift.tolist()),
            knots=knots,
            transform=transform,
            order=m,
            drop_null=null_dim,
            column_means=tuple(means.tolist()),
        )
        return RawSmooth(
            basis=basis,
            design=design - means,
            penalties=(penalty,),
            rank=(width,),
            null_dim=0,
            constraint="none",
        )
    basis = ThinPlateBasis(
        shift=tuple(shift.tolist()), knots=knots, transform=transform, order=m
    )
    return RawSmooth(
        basis=basis,
        design=design,
        penalties=(penalty,),
        rank=(rank,),
        null_dim=null_dim,
    )


# ---------------------------------------------------------------------------
# Gaussian process smooths (Kammann and Wand 2003; mgcv gp.smooth).


def _gp_correlation(
    distance: FloatArray, kind: int, rho: float, power: float
) -> FloatArray:
    """Evaluate mgcv's ``gpE`` correlation function of a scaled distance.

    Parameters
    ----------
    distance : numpy.ndarray
        Euclidean distances.
    kind : int
        1 spherical, 2 power exponential, 3 to 5 Matern with smoothness
        1.5, 2.5 and 3.5.
    rho : float
        Range dividing the distances.
    power : float
        Exponent of the power exponential.

    Returns
    -------
    numpy.ndarray
        Correlations with the shape of ``distance``.
    """
    e = distance / rho
    if kind == 1:
        return (1 - 1.5 * e + 0.5 * e**3) * (e <= 1)
    if kind == 2:
        return np.exp(-(e**power))
    decay = np.exp(-e)
    if kind == 3:
        return (1 + e) * decay
    if kind == 4:
        return decay + (e * decay) * (1 + e / 3)
    return decay + (e * decay) * (1 + 0.4 * e + e**2 / 15)


class GaussianProcessSmoothBasis(ValueModel):
    """Retained state of mgcv's Gaussian process smooth ``bs = "gp"``.

    Attributes
    ----------
    shift : tuple[float, ...]
        Training covariate means subtracted before evaluation.
    knots : numpy.ndarray
        Unique centered training covariate rows.
    transform : numpy.ndarray
        Knots-by-rank eigenvector map of the knot correlation matrix.
    kind : int
        Correlation function, 1 to 5 as in mgcv's ``m[1]``.
    stationary : bool
        Whether the null space is the constant alone (a negative ``m[1]``)
        and not the constant and linear functions.
    rho : float
        Correlation range.
    power : float
        Exponent of the power exponential correlation.
    """

    basis: Literal["gp"] = "gp"
    shift: tuple[float, ...]
    knots: FloatArray = _array_field()
    transform: FloatArray = _array_field()
    kind: int
    stationary: bool
    rho: float
    power: float

    def evaluate(self, x: FloatArray) -> FloatArray:
        """Evaluate the retained basis at fresh covariate rows.

        Costs O(rows times knots times (dimension plus rank)).

        Parameters
        ----------
        x : numpy.ndarray
            Covariate rows, shape rows by dimension.

        Returns
        -------
        numpy.ndarray
            Rows by basis dimension: the reduced correlation columns, the
            constant, and for a nonstationary smooth the centered covariates.
        """
        centered = x - np.asarray(self.shift)
        correlation = _gp_correlation(
            np.sqrt(_squared_distances(centered, self.knots)),
            self.kind,
            self.rho,
            self.power,
        )
        null = [np.ones((centered.shape[0], 1))]
        if not self.stationary:
            null.append(centered)
        return np.column_stack((correlation @ self.transform, *null))


def gp_smooth(x: FloatArray, k: int, definition: tuple[float, ...]) -> RawSmooth:
    """Construct mgcv's low-rank Gaussian process smooth [@kammann2003geoadditive].

    The knots are the unique covariate rows. The basis is the correlation
    function between rows and knots reduced to the ``k`` leading
    eigenvectors of the knot correlation matrix, followed by the constant
    and, unless stationary, the linear functions. The penalty is the
    diagonal matrix of those eigenvalues. Costs O(u squared) per Lanczos
    iteration for ``u`` unique rows.

    Parameters
    ----------
    x : numpy.ndarray
        Training covariate rows, shape rows by dimension.
    k : int
        Basis dimension, or a negative value for mgcv's default.
    definition : tuple[float, ...]
        mgcv's ``m``: the correlation type (negative for a stationary
        smooth), then optionally the range and the power; empty for the
        default Matern 1.5 with range equal to the largest knot distance.

    Returns
    -------
    RawSmooth
        Basis, design, penalty and null-space dimension.

    Raises
    ------
    ValueError
        If the definition is invalid, the covariate has too few unique rows,
        or it has more than 2000 unique rows, where mgcv subsamples knots
        with R's random number generator.
    """
    kind_value = definition[0] if definition else 3.0
    kind = abs(round(kind_value))
    stationary = bool(definition) and kind_value < 0
    rho = definition[1] if len(definition) > 1 else -1.0
    power = definition[2] if len(definition) > 2 else 1.0
    if (
        len(definition) > 3
        or kind_value != round(kind_value)
        or kind not in {1, 2, 3, 4, 5}
        or not 0 < power <= 2
    ):
        raise ValueError(
            "bs='gp' takes m = c(type, range, power) with type in 1..5 "
            "(negative for a stationary smooth) and power in (0, 2]"
        )
    dimension = x.shape[1]
    if dimension > 3:
        raise ValueError(
            "bs='gp' has a default basis dimension for up to three covariates"
        )
    shift = x.mean(axis=0)
    centered = x - shift
    knots = np.unique(centered, axis=0)
    count = knots.shape[0]
    if x.shape[0] > _MAX_KNOTS and count > _MAX_KNOTS:
        raise ValueError(
            f"a bs='gp' smooth over more than {_MAX_KNOTS} unique covariate rows "
            "subsamples knots with R's random number generator in mgcv and is "
            "not supported"
        )
    null_dim = 1 if stationary else dimension + 1
    width = dimension + 1 + (10, 30, 100)[dimension - 1] if k < 0 else k
    width = max(width, dimension + 2)
    if count < width:
        raise ValueError(
            "a smooth term has fewer unique covariate combinations than its "
            "basis dimension"
        )
    distance = np.sqrt(_squared_distances(knots, knots))
    if rho <= 0:
        rho = float(np.max(distance))
    correlation = _gp_correlation(distance, kind, rho, power)
    rank = width - null_dim
    penalty = np.zeros((width, width))
    if rank < count:
        values, vectors = _lanczos(correlation, rank, _EPS**0.5)
        penalty[:rank, :rank] = np.diag(values)
    else:
        vectors = np.eye(rank)
        penalty[:rank, :rank] = correlation
    basis = GaussianProcessSmoothBasis(
        shift=tuple(shift.tolist()),
        knots=knots,
        transform=vectors,
        kind=kind,
        stationary=stationary,
        rho=rho,
        power=power,
    )
    return RawSmooth(
        basis=basis,
        design=basis.evaluate(x),
        penalties=(penalty,),
        rank=(rank,),
        null_dim=null_dim,
    )


# ---------------------------------------------------------------------------
# Splines on the sphere (Wahba 1981; mgcv sos.smooth).


def _sphere_dilogarithm_kernel(cosine: FloatArray) -> FloatArray:
    """Port mgcv's ``rksos`` kernel of the second-order spline on the sphere.

    The value is the series over the Legendre polynomials ``P_v`` with
    coefficients ``(2 v + 1) / (v (v + 1)) ** 2`` from ``v = 1``
    [@wahba1981spline, equation 2.12], which equals
    ``1 - pi ** 2 / 6 + Li2((1 + x) / 2)`` with ``Li2`` the dilogarithm.
    mgcv sums the dilogarithm's power series in an argument of at most one
    half, by reflection for a positive cosine, and this port keeps its
    summation order and stopping tests. Costs O(size) per series term, with
    at most 55 terms in double precision.

    Parameters
    ----------
    cosine : numpy.ndarray
        Cosines of the angles between points, of any shape.

    Returns
    -------
    numpy.ndarray
        Kernel values with the shape of ``cosine``.
    """
    x = np.clip(np.asarray(cosine, dtype=np.float64), -1.0, 1.0)
    result = np.empty_like(x)
    dilog_one = (math.acos(0.0) * 2.0) ** 2 / 6.0
    lower = x <= 0
    ratio = x[lower] / 2 + 0.5
    total = np.full(ratio.shape, 1.0 - dilog_one)
    power = ratio.copy()
    active = np.ones(ratio.shape, dtype=bool)
    for k in range(1, 1000):
        term = power / (k * k)
        total[active] += term[active]
        power *= ratio
        active &= ~(term < _EPS)
        if not active.any():
            break
    result[lower] = total
    upper = ~lower
    value = x[upper]
    ratio = 0.5 - value / 2
    at_one = value / 2 >= 0.5
    safe = np.where(at_one, 0.0, value)
    total = np.where(at_one, 1.0, 1.0 - np.log(0.5 + safe / 2) * np.log(0.5 - safe / 2))
    power = ratio.copy()
    active = np.ones(ratio.shape, dtype=bool)
    for k in range(1, 1000):
        term = power / (k * k)
        total[active] -= term[active]
        power *= ratio
        active &= ~(power < _EPS)
        if not active.any():
            break
    result[upper] = total
    return result


def _sphere_null(points: FloatArray, order: int) -> FloatArray:
    """Evaluate the penalty null space of a spline on the sphere.

    Parameters
    ----------
    points : numpy.ndarray
        Latitude and longitude in degrees, shape rows by two.
    order : int
        mgcv's ``m``, from -2 to 4.

    Returns
    -------
    numpy.ndarray
        The constant column, followed for ``m = -1`` by the three
        coordinates of the unit sphere in its Euclidean embedding.
    """
    ones = np.ones((points.shape[0], 1))
    if order != -1:
        return ones
    latitude = np.deg2rad(points[:, 0])
    longitude = np.deg2rad(points[:, 1])
    return np.column_stack(
        (
            ones,
            np.cos(latitude) * np.sin(longitude),
            np.cos(latitude) * np.cos(longitude),
            np.sin(latitude),
        )
    )


def _sphere_kernel(points: FloatArray, knots: FloatArray, order: int) -> FloatArray:
    """Evaluate mgcv's ``makeR`` kernel between points and knots on the sphere.

    With ``z`` the cosine of the angle between two points, ``m = 0`` is the
    reproducing kernel of the second-order spline on the sphere divided by
    ``4 pi``, and ``m`` from 1 to 4 is the pseudospline kernel
    ``(q_m(z) / m! - 1 / (m + 1)!) / (2 pi)`` with
    ``q_m(z)`` the integral over ``h`` in the unit interval of
    ``(1 - h) ** m / sqrt(1 - 2 h z + h ** 2)``, in closed form
    [@wahba1981spline, equation 3.3 and table 1]. ``m = -1`` and ``m = -2``
    are the semi-kernels ``r ** 2 log(r) / (8 pi)`` and ``-r`` of the chord
    length ``r`` between the points in the Euclidean embedding of the
    sphere [@duchon1977splines]. Costs O(rows times knots).

    Parameters
    ----------
    points : numpy.ndarray
        Latitude and longitude in degrees, shape rows by two.
    knots : numpy.ndarray
        Knot latitude and longitude in degrees, shape knots by two.
    order : int
        mgcv's ``m``, from -2 to 4.

    Returns
    -------
    numpy.ndarray
        Rows by knots kernel values.
    """
    latitude = np.deg2rad(points[:, 0])[:, None]
    longitude = np.deg2rad(points[:, 1])[:, None]
    knot_latitude = np.deg2rad(knots[:, 0])[None, :]
    knot_longitude = np.deg2rad(knots[:, 1])[None, :]
    cosine = np.sin(latitude) * np.sin(knot_latitude) + np.cos(latitude) * np.cos(
        knot_latitude
    ) * np.cos(longitude - knot_longitude)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    if order < 0:
        chord = np.maximum(2 * np.sin(angle / 2), _TINY * 10)
        if order == -2:
            return -chord
        return chord * chord * np.log(chord) / (8 * math.pi)
    if order == 0:
        return _sphere_dilogarithm_kernel(np.cos(angle)) / (4 * math.pi)
    w = np.maximum(1 - np.cos(angle), _EPS * 1e-4) / 2
    c = np.sqrt(w)
    a = np.log(1 + 1 / c)
    c = c * 2
    if order == 1:
        q = 2 * a * w - c + 1
        return (q - 1 / 2) / (2 * math.pi)
    w2 = w * w
    if order == 2:
        q = a * (6 * w2 - 2 * w) - 3 * c * w + 3 * w + 1 / 2
        return (q / 2 - 1 / 6) / (2 * math.pi)
    w3 = w2 * w
    if order == 3:
        q = (a * (60 * w3 - 36 * w2) + 30 * w2 + c * (8 * w - 30 * w2) - 3 * w + 1) / 3
        return (q / 6 - 1 / 24) / (2 * math.pi)
    w4 = w3 * w
    q = (
        a * (70 * w4 - 60 * w3 + 6 * w2)
        + 35 * w3 * (1 - c)
        + c * 55 * w2 / 3
        - 12.5 * w2
        - w / 3
        + 1 / 4
    )
    return (q / 24 - 1 / 120) / (2 * math.pi)


def _check_sphere(points: FloatArray, role: str) -> None:
    """Refuse coordinates that are not latitudes and longitudes in degrees.

    Parameters
    ----------
    points : numpy.ndarray
        Latitude and longitude, shape rows by two.
    role : str
        Description of the rows used in the error.

    Raises
    ------
    ValueError
        If a value is not finite, a latitude lies outside -90 to 90, or a
        longitude lies outside -180 to 360.
    """
    if not np.all(np.isfinite(points)):
        raise ValueError(f"bs='sos' {role} must be finite")
    if np.any(np.abs(points[:, 0]) > 90):
        raise ValueError(
            f"bs='sos' takes latitude in degrees as its first covariate; {role} "
            "hold a latitude outside [-90, 90]"
        )
    if np.any((points[:, 1] < -180) | (points[:, 1] > 360)):
        raise ValueError(
            f"bs='sos' takes longitude in degrees as its second covariate; {role} "
            "hold a longitude outside [-180, 360]"
        )


def _r_number_text(value: float) -> str:
    """Format a finite Double as R's ``as.character`` does.

    R writes the value rounded to at most 15 significant digits without
    trailing zeros, in fixed notation unless scientific notation is
    narrower.

    Parameters
    ----------
    value : float
        Finite number.

    Returns
    -------
    str
        R's text of the number.
    """
    if value == 0:
        return "0"
    mantissa, exponent_text = f"{abs(value):.14e}".split("e")
    digits = mantissa.replace(".", "").rstrip("0")
    exponent = int(exponent_text)
    count = len(digits)
    right = max(0, count - exponent - 1)
    left = exponent + 1 if exponent >= 0 else 1
    fixed_width = left + right + (1 if right else 0)
    scientific_width = (count + 1 if count > 1 else 1) + (
        4 if abs(exponent) < 100 else 5
    )
    sign = "-" if value < 0 else ""
    if fixed_width <= scientific_width:
        if exponent >= 0:
            whole = digits[: exponent + 1].ljust(exponent + 1, "0")
            fraction = digits[exponent + 1 :]
        else:
            whole = "0"
            fraction = "0" * (-exponent - 1) + digits
        return sign + whole + ("." + fraction if fraction else "")
    body = digits[0] + ("." + digits[1:] if count > 1 else "")
    return f"{sign}{body}e{'-' if exponent < 0 else '+'}{abs(exponent):02d}"


def _unique_rows_as_text(x: FloatArray) -> FloatArray:
    """Return the unique rows in the order of mgcv's ``uniquecombs(x, TRUE)``.

    mgcv pastes each row's numbers as text separated by ``*``, keeps the
    first row of each distinct text, and sorts the rows by that text in the
    C collation. Rows that agree to 15 significant digits are thus one row,
    and the order is the byte order of the decimal text, not numeric order.

    Parameters
    ----------
    x : numpy.ndarray
        Finite rows by columns.

    Returns
    -------
    numpy.ndarray
        Unique rows in mgcv's order.
    """
    first: dict[bytes, int] = {}
    for index, row in enumerate(x.tolist()):
        key = "*".join(_r_number_text(value) for value in row).encode("ascii")
        first.setdefault(key, index)
    return x[[first[key] for key in sorted(first)]]


class SphereBasis(ValueModel):
    """Retained state of mgcv's spline on the sphere ``bs = "sos"``.

    Attributes
    ----------
    knots : numpy.ndarray
        Knot latitude and longitude in degrees, shape knots by two.
    transform : numpy.ndarray
        mgcv's ``UZ``: the map from kernel evaluations at the knots to the
        penalized coordinates, orthogonal to the null space.
    order : int
        mgcv's ``m`` after rounding and clamping, from -2 to 4.
    scale : numpy.ndarray
        mgcv's ``xc.scale`` column multipliers.
    """

    basis: Literal["sos"] = "sos"
    knots: FloatArray = _array_field()
    transform: FloatArray = _array_field()
    order: int
    scale: FloatArray = _array_field()

    def unscaled(self, x: FloatArray) -> FloatArray:
        """Evaluate the basis columns before mgcv's column scaling.

        Parameters
        ----------
        x : numpy.ndarray
            Latitude and longitude in degrees, shape rows by two.

        Returns
        -------
        numpy.ndarray
            Rows by basis dimension: the reduced kernel columns, then the
            null space.

        Raises
        ------
        ValueError
            If a row is not a latitude and longitude in degrees.
        """
        _check_sphere(x, "the evaluated rows")
        kernel = _sphere_kernel(x, self.knots, self.order)
        return np.column_stack((kernel @ self.transform, _sphere_null(x, self.order)))

    def evaluate(self, x: FloatArray) -> FloatArray:
        """Evaluate the retained basis at fresh latitudes and longitudes.

        Costs O(rows times knots times rank).

        Parameters
        ----------
        x : numpy.ndarray
            Latitude and longitude in degrees, shape rows by two.

        Returns
        -------
        numpy.ndarray
            Rows by basis dimension.

        Raises
        ------
        ValueError
            If a row is not a latitude and longitude in degrees.
        """
        return self.unscaled(x) * self.scale


def sphere_order(m: float) -> int:
    """Round and clamp the ``m`` of a spline on the sphere as mgcv does.

    Parameters
    ----------
    m : float
        Authored ``m``.

    Returns
    -------
    int
        ``m`` rounded half to even, with a value below -2 read as -1 and a
        value above 4 read as 4.

    Raises
    ------
    ValueError
        If ``m`` is not finite.
    """
    if not math.isfinite(m):
        raise ValueError("bs='sos' takes one finite m from -2 to 4")
    order = round(m)
    if order < -2:
        return -1
    return min(order, 4)


def sphere_spline(
    x: FloatArray, k: int, order: int, *, knots: FloatArray | None = None
) -> RawSmooth:
    """Construct mgcv's low-rank spline on the sphere [@wahba1981spline].

    The knots are the unique latitude and longitude rows in the order of
    mgcv's ``uniquecombs``, or the given knots. The basis is the kernel between rows and knots reduced to the
    ``k`` largest-magnitude eigenvectors of the knot kernel matrix, as for
    the thin-plate regression spline [@wood2003thin], and constrained
    orthogonal to the penalty null space, followed by the null space. Every
    column except the constant is scaled to unit standard deviation. Costs
    O(u squared) per Lanczos iteration for ``u`` knots.

    Parameters
    ----------
    x : numpy.ndarray
        Training latitude and longitude in degrees, shape rows by two.
    k : int
        Basis dimension, or a negative value for mgcv's default of 50.
    order : int
        mgcv's ``m`` from -2 to 4: 0 for the second-order spline on the
        sphere, 1 to 4 for the pseudospline whose penalty has order
        ``(m + 2) / 2``, and -1 or -2 for the second or first derivative
        penalty of the sphere embedded in three dimensions.
    knots : numpy.ndarray or None
        Knot latitude and longitude in degrees, shape knots by two.

    Returns
    -------
    RawSmooth
        Basis, design, penalty and null-space dimension.

    Raises
    ------
    ValueError
        If the covariates are not one latitude and one longitude in
        degrees, the basis dimension is below the null-space dimension plus
        two or above the knot count, there are more knots than rows, the
        null space is not identified by the knots, or the rows have more
        than 2000 unique combinations, where mgcv subsamples knots with R's
        random number generator.
    """
    if x.shape[1] != 2:
        raise ValueError(
            "bs='sos' takes exactly two covariates, latitude and longitude in "
            "degrees, as in s(lat, lon, bs='sos')"
        )
    if order not in range(-2, 5):
        raise ValueError("bs='sos' takes one m from -2 to 4")
    _check_sphere(x, "the covariates")
    if knots is None:
        knots = _unique_rows_as_text(x)
        if x.shape[0] > _MAX_KNOTS and knots.shape[0] > _MAX_KNOTS:
            raise ValueError(
                f"a bs='sos' smooth over more than {_MAX_KNOTS} unique covariate "
                "rows subsamples knots with R's random number generator in mgcv "
                "and is not supported: give knots for both covariates"
            )
    else:
        _check_sphere(knots, "the knots")
        if knots.shape[0] > x.shape[0]:
            raise ValueError(
                "a bs='sos' smooth has more knots than rows; mgcv ignores such knots"
            )
    count = knots.shape[0]
    null = _sphere_null(knots, order)
    null_dim = null.shape[1]
    width = 50 if k < 0 else k
    if width < null_dim + 2:
        raise ValueError(
            f"bs='sos' with m={order} requires a basis dimension k of at least "
            f"{null_dim + 2}"
        )
    if count < width:
        raise ValueError(
            "a smooth term has fewer unique covariate combinations than its "
            f"basis dimension: bs='sos' has {count} knots and k={width}"
        )
    kernel = _sphere_kernel(knots, knots, order)
    if width < count:
        values, vectors = _lanczos(kernel, width, _EPS**0.5)
        constraint = vectors.T @ null
        reduced = np.diag(values)
    else:
        vectors = np.eye(width)
        constraint = null
        reduced = kernel
    if np.linalg.matrix_rank(constraint) < null_dim:
        raise ValueError(
            "bs='sos' with m=-1 has the null space of the constant and the three "
            "coordinates of the embedded sphere, which the knots do not "
            "identify: they lie on one circle"
        )
    q = np.linalg.qr(constraint, mode="complete")[0]
    rank = width - null_dim
    penalty = np.zeros((width, width))
    penalty[:rank, :rank] = (q.T @ reduced @ q)[null_dim:, null_dim:]
    basis = SphereBasis(
        knots=knots,
        transform=(vectors @ q)[:, null_dim:],
        order=order,
        scale=np.ones(width),
    )
    design = basis.unscaled(x)
    spread = np.std(design, axis=0, ddof=1)
    spread[spread == spread.min()] = 1.0
    scale = 1.0 / spread
    penalty = penalty * np.outer(scale, scale)
    return RawSmooth(
        basis=basis.with_(scale=scale),
        design=design * scale,
        penalties=((penalty + penalty.T) * 0.5,),
        rank=(rank,),
        null_dim=null_dim,
    )


# ---------------------------------------------------------------------------
# Cubic regression and cyclic cubic splines (mgcv crspl, cyclic.smooth).


class CubicRegressionBasis(ValueModel):
    """Retained knots and second-derivative map of mgcv ``cr.smooth``.

    Attributes
    ----------
    knots : tuple[float, ...]
        Increasing knot locations.
    second : numpy.ndarray
        Knots-by-knots map from knot values to second derivatives.
    """

    kind: Literal["cr"] = "cr"
    knots: tuple[float, ...]
    second: FloatArray = _array_field()

    @property
    def width(self) -> int:
        """Number of raw basis columns.

        Returns
        -------
        int
            Knot count.
        """
        return len(self.knots)

    def evaluate(self, x: FloatArray) -> FloatArray:
        """Evaluate the natural cubic spline basis, extrapolating linearly.

        Costs O(rows times knots).

        Parameters
        ----------
        x : numpy.ndarray
            One covariate column, shape rows by one.

        Returns
        -------
        numpy.ndarray
            Rows by knot count.
        """
        values = x[:, 0]
        knots = np.asarray(self.knots)
        g = self.second
        nk = knots.size
        out = np.zeros((values.size, nk))
        rows = np.arange(values.size)
        lower, upper = values < knots[0], values > knots[-1]
        inside = ~(lower | upper)
        j = np.clip(np.searchsorted(knots, values, side="left") - 1, 0, nk - 2)
        xi, jj = values[inside], j[inside]
        h = knots[jj + 1] - knots[jj]
        am, ap = knots[jj + 1] - xi, xi - knots[jj]
        cm = am * (am * am / h - h) / 6
        cp = ap * (ap * ap / h - h) / 6
        r = rows[inside]
        out[r] = cm[:, None] * g[jj] + cp[:, None] * g[jj + 1]
        out[r, jj] += am / h
        out[r, jj + 1] += ap / h
        if lower.any():
            r = rows[lower]
            h0 = knots[1] - knots[0]
            xik = values[lower] - knots[0]
            out[r] = (-xik * h0 / 3)[:, None] * g[0] + (-xik * h0 / 6)[:, None] * g[1]
            out[r, 0] += 1 - xik / h0
            out[r, 1] += xik / h0
        if upper.any():
            r = rows[upper]
            h1 = knots[-1] - knots[-2]
            xik = values[upper] - knots[-1]
            out[r] = (xik * h1 / 6)[:, None] * g[-2] + (xik * h1 / 3)[:, None] * g[-1]
            out[r, -2] += -xik / h1
            out[r, -1] += 1 + xik / h1
        return out


def _r_quantile(values: FloatArray, count: int) -> FloatArray:
    """Evaluate R's default type-7 quantiles at evenly spaced probabilities.

    Parameters
    ----------
    values : numpy.ndarray
        Finite values.
    count : int
        Number of probabilities from zero to one inclusive.

    Returns
    -------
    numpy.ndarray
        Quantiles in increasing order.
    """
    ordered = np.sort(values)
    probabilities = np.linspace(0.0, 1.0, count)
    index = (ordered.size - 1) * probabilities
    lower = np.floor(index + 4 * _EPS).astype(int)
    lower = np.clip(lower, 0, ordered.size - 1)
    upper = np.minimum(lower + 1, ordered.size - 1)
    fraction = index - lower
    fraction = np.where(np.abs(fraction) < 4 * _EPS, 0.0, fraction)
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _user_knots(knots: FloatArray | None, role: str) -> FloatArray | None:
    """Check user knots (brms ``knots``) as finite and strictly increasing.

    Parameters
    ----------
    knots : numpy.ndarray or None
        Knots as given.
    role : str
        Basis description for errors.

    Returns
    -------
    numpy.ndarray or None
        Float64 knots, or None when none were given.

    Raises
    ------
    ValueError
        If a knot is not finite or the knots do not increase strictly.
    """
    if knots is None:
        return None
    values = np.asarray(knots, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or (np.diff(values) <= 0).any():
        raise ValueError(f"{role} knots must be finite and strictly increasing")
    return values


def cubic_regression(
    x: FloatArray,
    k: int,
    *,
    shrink: float | None,
    knots: FloatArray | None = None,
) -> RawSmooth:
    """Construct mgcv's cubic regression spline with knots at quantiles.

    Parameters
    ----------
    x : numpy.ndarray
        One covariate column, shape rows by one.
    k : int
        Knot count, or a negative value for mgcv's default of ten.
    shrink : float or None
        Null-space shrinkage factor of ``bs = 'cs'``, or None.
    knots : numpy.ndarray or None
        User knots (brms ``knots``), exactly ``k`` of them, in place of the
        quantiles of the unique values, as mgcv's ``cr.smooth``.

    Returns
    -------
    RawSmooth
        Basis, training design, and raw penalty.

    Raises
    ------
    ValueError
        If ``k`` is below three or exceeds the unique covariate count, or the
        user knots are not ``k`` strictly increasing values.
    """
    nk = 10 if k < 0 else k
    if nk < 3:
        raise ValueError("cubic regression smooths require k >= 3")
    unique = np.unique(x[:, 0])
    if unique.size < nk:
        raise ValueError(
            f"cubic regression smooth has insufficient unique values to support {nk} knots"
        )
    supplied = _user_knots(knots, "cubic regression smooth")
    if supplied is not None and supplied.size != nk:
        raise ValueError(
            f"cubic regression smooth takes k = {nk} knots, {supplied.size} given"
        )
    knots = _r_quantile(unique, nk) if supplied is None else supplied
    h = np.diff(knots)
    n2 = nk - 2
    d = np.zeros((n2, nk))
    for i in range(n2):
        d[i, i] = 1 / h[i]
        d[i, i + 2] = 1 / h[i + 1]
        d[i, i + 1] = -d[i, i] - d[i, i + 2]
    b = np.diag((h[:-1] + h[1:]) / 3)
    for i in range(1, n2):
        b[i - 1, i] = b[i, i - 1] = h[i] / 6
    bd = np.linalg.solve(b, d)
    second = np.zeros((nk, nk))
    second[1:-1] = bd
    penalty = d.T @ bd
    penalty = (penalty + penalty.T) * 0.5
    basis = CubicRegressionBasis(knots=tuple(knots.tolist()), second=second)
    design = basis.evaluate(x)
    rank, null_dim = nk - 2, 2
    if shrink is not None:
        values, vectors = _signed_eigh(penalty)
        vectors = _canonical_null(design, vectors, nk - 2)
        values[nk - 2] = values[nk - 3] * shrink
        values[nk - 1] = values[nk - 2] * shrink
        penalty = (vectors * values) @ vectors.T
        rank, null_dim = nk, 0
    return RawSmooth(
        basis=basis,
        design=design,
        penalties=(penalty,),
        rank=(rank,),
        null_dim=null_dim,
    )


class CyclicBasis(ValueModel):
    """Retained knots and second-derivative map of mgcv ``cyclic.smooth``.

    Attributes
    ----------
    knots : tuple[float, ...]
        Increasing knots whose end points are identified.
    second : numpy.ndarray
        Map from the cyclic knot values to second derivatives.
    """

    kind: Literal["cc"] = "cc"
    knots: tuple[float, ...]
    second: FloatArray = _array_field()

    @property
    def width(self) -> int:
        """Number of raw basis columns.

        Returns
        -------
        int
            Knot count minus one.
        """
        return len(self.knots) - 1

    def evaluate(self, x: FloatArray) -> FloatArray:
        """Evaluate the cyclic cubic basis, wrapping values into the period.

        Costs O(rows times knots).

        Parameters
        ----------
        x : numpy.ndarray
            One covariate column, shape rows by one.

        Returns
        -------
        numpy.ndarray
            Rows by knot count minus one.
        """
        knots = np.asarray(self.knots)
        values = x[:, 0].copy()
        low, high = knots[0], knots[-1]
        period = high - low
        above, below = values > high, values < low
        values[above] = low + np.mod(values[above] - high, period)
        values[below] = high - np.mod(low - values[below], period)
        n = knots.size
        h = np.diff(knots)
        j = np.full(values.size, 0)
        for i in range(n, 1, -1):
            j[values <= knots[i - 1]] = i
        j1 = j - 1
        jw = j.copy()
        jw[jw == n] = 1
        bd = self.second
        eye = np.eye(n - 1)
        hj = h[j1 - 1]
        left, right = knots[j1] - values, values - knots[j1 - 1]
        return (
            bd[j1 - 1] * (left**3 / (6 * hj))[:, None]
            + bd[jw - 1] * (right**3 / (6 * hj))[:, None]
            - bd[j1 - 1] * (hj * left / 6)[:, None]
            - bd[jw - 1] * (hj * right / 6)[:, None]
            + eye[j1 - 1] * (left / hj)[:, None]
            + eye[jw - 1] * (right / hj)[:, None]
        )


def _place_knots(values: FloatArray, count: int) -> FloatArray:
    """Place knots evenly through sorted unique values (mgcv ``place.knots``).

    Parameters
    ----------
    values : numpy.ndarray
        Covariate values.
    count : int
        Number of knots.

    Returns
    -------
    numpy.ndarray
        Increasing knots spanning the data range.

    Raises
    ------
    ValueError
        If more knots than unique values are requested.
    """
    x = np.unique(values)
    n = x.size
    if count > n:
        raise ValueError("cyclic smooth requests more knots than unique data values")
    if count == 2:
        return np.array([x[0], x[-1]])
    delta = (n - 1) / (count - 1)
    steps = delta * np.arange(1, count - 1)
    lbi = np.floor(steps).astype(int) + 1
    frac = steps + 1 - lbi
    knots = np.empty(count)
    knots[0], knots[-1] = x[0], x[-1]
    knots[1:-1] = x[lbi - 1] * (1 - frac) + x[lbi] * frac
    return knots


def cyclic_cubic(
    x: FloatArray, k: int, *, knots: FloatArray | None = None
) -> RawSmooth:
    """Construct mgcv's cyclic cubic regression spline.

    Parameters
    ----------
    x : numpy.ndarray
        One covariate column, shape rows by one.
    k : int
        Knot count, or a negative value for mgcv's default of ten.
    knots : numpy.ndarray or None
        User knots (brms ``knots``): ``k`` knots, or two end points between
        which mgcv's ``place.knots`` spreads ``k`` knots over the end points
        and the data, as mgcv's ``cc.smooth``.

    Returns
    -------
    RawSmooth
        Basis, training design, and raw penalty.

    Raises
    ------
    ValueError
        If ``k`` is below four or exceeds the unique covariate count, or the
        user knots are neither two nor ``k`` strictly increasing values.
    """
    nk = 10 if k < 0 else k
    if nk < 4:
        raise ValueError("cyclic cubic smooths require k >= 4")
    supplied = _user_knots(knots, "cyclic cubic smooth")
    if supplied is None:
        knots = _place_knots(x[:, 0], nk)
    elif supplied.size == 2:
        knots = _place_knots(np.concatenate([supplied, x[:, 0]]), nk)
    elif supplied.size == nk:
        knots = supplied
    else:
        raise ValueError(
            f"cyclic cubic smooth takes k = {nk} knots or two end points, "
            f"{supplied.size} given"
        )
    h = np.diff(knots)
    n = nk - 1
    b = np.zeros((n, n))
    d = np.zeros((n, n))
    b[0, 0] = (h[n - 1] + h[0]) / 3
    b[0, 1] = h[0] / 6
    b[0, n - 1] = h[n - 1] / 6
    d[0, 0] = -(1 / h[0] + 1 / h[n - 1])
    d[0, 1] = 1 / h[0]
    d[0, n - 1] = 1 / h[n - 1]
    for i in range(1, n - 1):
        b[i, i - 1] = h[i - 1] / 6
        b[i, i] = (h[i - 1] + h[i]) / 3
        b[i, i + 1] = h[i] / 6
        d[i, i - 1] = 1 / h[i - 1]
        d[i, i] = -(1 / h[i - 1] + 1 / h[i])
        d[i, i + 1] = 1 / h[i]
    b[n - 1, n - 2] = h[n - 2] / 6
    b[n - 1, n - 1] = (h[n - 2] + h[n - 1]) / 3
    b[n - 1, 0] = h[n - 1] / 6
    d[n - 1, n - 2] = 1 / h[n - 2]
    d[n - 1, n - 1] = -(1 / h[n - 2] + 1 / h[n - 1])
    d[n - 1, 0] = 1 / h[n - 1]
    bd = np.asarray(np.linalg.solve(b, d), dtype=np.float64)
    penalty = d.T @ bd
    penalty = (penalty + penalty.T) * 0.5
    basis = CyclicBasis(knots=tuple(knots.tolist()), second=bd)
    return RawSmooth(
        basis=basis,
        design=basis.evaluate(x),
        penalties=(penalty,),
        rank=(n - 1,),
        null_dim=1,
    )


# ---------------------------------------------------------------------------
# B-spline bases: P-splines and derivative-penalized B-splines.


def _bspline_design(
    knots: FloatArray, x: FloatArray, order: int, derivative: int = 0
) -> FloatArray:
    """Evaluate a B-spline basis or its derivative (R ``splines::spline.des``).

    Parameters
    ----------
    knots : numpy.ndarray
        Full nondecreasing knot vector.
    x : numpy.ndarray
        Evaluation points inside the knot support.
    order : int
        Spline order, polynomial degree plus one.
    derivative : int
        Derivative order.

    Returns
    -------
    numpy.ndarray
        Points by basis functions.
    """
    width = knots.size - order
    spline = BSpline(knots, np.eye(width), order - 1, extrapolate=False)
    values = spline(x, nu=derivative) if derivative else spline(x)
    return np.nan_to_num(np.asarray(values, dtype=np.float64))


class BSplineBasis(ValueModel):
    """Retained knots of mgcv's ``pspline.smooth`` and ``Bspline.smooth``.

    Attributes
    ----------
    knots : tuple[float, ...]
        Full evenly spaced knot vector.
    order : int
        Spline order, polynomial degree plus one.
    """

    kind: Literal["bspline"] = "bspline"
    knots: tuple[float, ...]
    order: int

    @property
    def width(self) -> int:
        """Number of raw basis columns.

        Returns
        -------
        int
            Knot count minus spline order.
        """
        return len(self.knots) - self.order

    def evaluate(self, x: FloatArray, derivative: int = 0) -> FloatArray:
        """Evaluate the basis, extrapolating linearly beyond its support.

        Costs O(rows times basis width).

        Parameters
        ----------
        x : numpy.ndarray
            One covariate column, shape rows by one.
        derivative : int
            Derivative order of the evaluated basis.

        Returns
        -------
        numpy.ndarray
            Rows by basis width.
        """
        knots = np.asarray(self.knots)
        values = x[:, 0]
        lower = knots[self.order - 1]
        upper = knots[knots.size - self.order]
        inside = (values >= lower) & (values <= upper)
        out = np.zeros((values.size, self.width))
        if inside.any():
            out[inside] = _bspline_design(knots, values[inside], self.order, derivative)
        if derivative < 2 and not inside.all():
            ends = np.array([lower, lower, upper, upper])
            boundary = np.vstack(
                [
                    _bspline_design(knots, ends[i : i + 1], self.order, i % 2)
                    for i in range(4)
                ]
            )
            for mask, edge, rows in (
                (values < lower, lower, boundary[0:2]),
                (values > upper, upper, boundary[2:4]),
            ):
                if mask.any():
                    out[mask] = (
                        np.column_stack((np.ones(mask.sum()), values[mask] - edge))
                        @ rows
                        if derivative == 0
                        else np.tile(rows[1], (int(mask.sum()), 1))
                    )
        return out


def _padded_range(x: FloatArray) -> tuple[float, float]:
    """Pad the covariate range by mgcv's 0.1 percent on each side.

    Parameters
    ----------
    x : numpy.ndarray
        Covariate values.

    Returns
    -------
    tuple[float, float]
        Padded lower and upper limits.

    Raises
    ------
    ValueError
        If the covariate takes a single value.
    """
    low, high = float(x.min()), float(x.max())
    if low == high:
        raise ValueError(
            "smooth predictors require at least two distinct training values"
        )
    width = high - low
    return low - width * 0.001, high + width * 0.001


def _knot_range(x: FloatArray, supplied: FloatArray) -> tuple[float, float]:
    """Pad two user end points as mgcv pads the data range.

    Parameters
    ----------
    x : numpy.ndarray
        Covariate values.
    supplied : numpy.ndarray
        Two increasing end points.

    Returns
    -------
    tuple[float, float]
        End points widened by 0.1 percent of their distance on each side.

    Raises
    ------
    ValueError
        If the end points do not include the data.
    """
    low, high = float(supplied[0]), float(supplied[1])
    if low > float(x.min()) or high < float(x.max()):
        raise ValueError("smooth knot range does not include the data")
    width = high - low
    return low - width * 0.001, high + width * 0.001


def p_spline(
    x: FloatArray, k: int, m: tuple[int, int], *, knots: FloatArray | None = None
) -> RawSmooth:
    """Construct mgcv's P-spline with a coefficient-difference penalty.

    Parameters
    ----------
    x : numpy.ndarray
        One covariate column, shape rows by one.
    k : int
        Basis dimension, or a negative value for mgcv's default.
    m : tuple[int, int]
        Spline order minus two and difference order.
    knots : numpy.ndarray or None
        User knots (brms ``knots``): two end points that replace the data
        range, or the full sequence of ``k - m[0] + 2 m[0] + 2`` knots, as
        mgcv's ``ps.smooth``.

    Returns
    -------
    RawSmooth
        Basis, training design, and raw penalty.

    Raises
    ------
    ValueError
        If the basis dimension is too small for the spline order, or the user
        knots have another length or do not include the data.
    """
    width = max(10, m[0] + 1) if k < 0 else k
    nk = width - m[0]
    if nk <= 0 or m[1] < 0 or m[1] >= width:
        raise ValueError("P-spline basis dimension too small for its order and penalty")
    supplied = _user_knots(knots, "P-spline")
    if supplied is not None and supplied.size not in (2, nk + 2 * m[0] + 2):
        raise ValueError(
            f"P-spline takes two end points or {nk + 2 * m[0] + 2} knots, "
            f"{supplied.size} given"
        )
    if supplied is not None and supplied.size != 2:
        knots = supplied
    else:
        low, high = (
            _padded_range(x[:, 0])
            if supplied is None
            else _knot_range(x[:, 0], supplied)
        )
        dx_ = (high - low) / (nk - 1)
        knots = np.linspace(
            low - dx_ * (m[0] + 1), high + dx_ * (m[0] + 1), nk + 2 * m[0] + 2
        )
    basis = BSplineBasis(knots=tuple(knots.tolist()), order=m[0] + 2)
    difference = np.diff(np.eye(width), n=m[1], axis=0) if m[1] else np.eye(width)
    return RawSmooth(
        basis=basis,
        design=_bspline_design(knots, x[:, 0], m[0] + 2),
        penalties=(difference.T @ difference,),
        rank=(width - m[1],),
        null_dim=m[1],
    )


def b_spline(
    x: FloatArray, k: int, m: tuple[int, int], *, knots: FloatArray | None = None
) -> RawSmooth:
    """Construct mgcv's B-spline with an integrated squared derivative penalty.

    Parameters
    ----------
    x : numpy.ndarray
        One covariate column, shape rows by one.
    k : int
        Basis dimension, or a negative value for mgcv's default.
    m : tuple[int, int]
        Spline order minus one and penalized derivative order.
    knots : numpy.ndarray or None
        User knots (brms ``knots``), as mgcv's ``bs.smooth``: two end points
        that replace the data range; four knots, the outer two bounding the
        evaluation range and the inner two the interior knots; or the full
        sequence of ``k - m[0] + 1 + 2 m[0]`` knots.

    Returns
    -------
    RawSmooth
        Basis, training design, and raw penalty.

    Raises
    ------
    ValueError
        If the basis dimension or derivative is invalid, or the user knots
        have another length or do not include the data.
    """
    order, deriv = m
    width = max(10, order) if k < 0 else k
    nk = width - order + 1
    pord = order - deriv
    if nk <= 1 or pord < 0:
        raise ValueError("B-spline basis dimension or penalty derivative is invalid")
    supplied = _user_knots(knots, "B-spline")
    full = nk + 2 * order
    if supplied is not None and supplied.size == 4 and 4 < full:
        dx_ = (supplied[3] - supplied[0]) / (nk - 1)
        knots = np.concatenate(
            [
                np.linspace(supplied[0] - dx_ * order, supplied[0], order + 1),
                np.linspace(supplied[1], supplied[2], max(0, nk - 2)),
                np.linspace(supplied[3], supplied[3] + dx_ * order, order + 1),
            ]
        )
    elif supplied is None or supplied.size == 2:
        low, high = (
            _padded_range(x[:, 0])
            if supplied is None
            else _knot_range(x[:, 0], supplied)
        )
        dx_ = (high - low) / (nk - 1)
        knots = np.linspace(low - dx_ * order, high + dx_ * order, full)
    elif supplied.size == full:
        knots = supplied
    else:
        raise ValueError(
            f"B-spline takes two end points, four knots, or {full} knots, "
            f"{supplied.size} given"
        )
    basis = BSplineBasis(knots=tuple(knots.tolist()), order=order + 1)
    k0 = knots[order : order + nk]
    h = np.diff(k0)
    if pord == 0:
        k1 = (k0[1:] + k0[:-1]) / 2
        d = basis.evaluate(k1[:, None], deriv)
        root = np.sqrt(h)[:, None] * d
    else:
        k1 = np.cumsum(np.concatenate(([k0[0]], np.repeat(h / pord, pord))))
        d = basis.evaluate(k1[:, None], deriv)
        nodes = np.linspace(-1, 1, pord + 1)
        p = np.linalg.inv(nodes[:, None] ** np.arange(pord + 1)[None, :])
        i1 = np.add.outer(np.arange(pord + 1), np.arange(pord + 1))
        hmat = (1 + (-1.0) ** i1) / (i1 + 1)
        w1 = p.T @ hmat @ p
        half = h / 2
        segments = half.size
        ld0 = np.tile(np.diag(w1), segments) * np.repeat(half, pord + 1)
        picks = np.concatenate(
            (
                (
                    np.tile(np.arange(1, pord + 1), segments)
                    + np.repeat(np.arange(segments) * (pord + 1), pord)
                ),
                [ld0.size],
            )
        )
        ld = ld0[picks - 1]
        i0 = np.arange(1, segments) * pord + 1
        i2 = np.arange(1, segments) * (pord + 1)
        ld[i0 - 1] = ld[i0 - 1] + ld0[i2 - 1]
        band = np.zeros((pord + 1, ld.size))
        band[0] = ld
        for kk in range(1, pord + 1):
            diwk = np.diagonal(w1, offset=kk)
            filled = np.tile(
                np.concatenate((diwk, np.zeros(kk - 1))), segments
            ) * np.repeat(half, pord)
            band[kk, : ld.size - kk] = filled[: ld.size - kk]
        dense = np.diag(band[0])
        for kk in range(1, pord + 1):
            dense += np.diag(band[kk, : ld.size - kk], kk) + np.diag(
                band[kk, : ld.size - kk], -kk
            )
        upper = np.linalg.cholesky(dense).T
        root = upper @ d
    return RawSmooth(
        basis=basis,
        design=_bspline_design(knots, x[:, 0], order + 1),
        penalties=(root.T @ root,),
        rank=(width - deriv,),
        null_dim=deriv,
    )


# ---------------------------------------------------------------------------
# Random-effect smooths.


class RandomEffectBasis(ValueModel):
    """Retained factor levels of mgcv's ``random.effect`` smooth.

    Attributes
    ----------
    levels : tuple[tuple[str, ...], ...]
        Factor levels per covariate, empty for a numeric covariate.
    """

    kind: Literal["re"] = "re"
    levels: tuple[tuple[str, ...], ...]

    @property
    def width(self) -> int:
        """Number of raw basis columns.

        Returns
        -------
        int
            Product of the factor level counts.
        """
        return math.prod(len(levels) for levels in self.levels if levels) or 1

    def evaluate(self, columns: Sequence[npt.ArrayLike]) -> FloatArray:
        """Evaluate R's ``model.matrix(~ a:b - 1)`` for the retained levels.

        Costs O(rows times levels).

        Parameters
        ----------
        columns : Sequence[numpy.ndarray]
            One column per covariate, factor columns holding labels.

        Returns
        -------
        numpy.ndarray
            Rows by product of level counts.

        Raises
        ------
        ValueError
            If a factor column holds a level absent at training time.
        """
        rows = np.asarray(columns[0]).shape[0]
        design = np.ones((rows, 1))
        for column, levels in zip(columns, self.levels, strict=True):
            if levels:
                labels = np.asarray(column).astype(str)
                unknown = set(labels.tolist()) - set(levels)
                if unknown:
                    raise ValueError(
                        f"random-effect smooth has unseen levels {sorted(unknown)!r}"
                    )
                indicator = (labels[:, None] == np.asarray(levels)[None, :]).astype(
                    float
                )
                design = (indicator[:, None, :] * design[:, :, None]).reshape(rows, -1)
            else:
                design = design * np.asarray(column, dtype=np.float64)[:, None]
        return design


def random_effect(
    columns: Sequence[npt.ArrayLike], levels: tuple[tuple[str, ...], ...]
) -> RawSmooth:
    """Construct mgcv's random-effect smooth with an identity penalty.

    Parameters
    ----------
    columns : Sequence[numpy.ndarray]
        Covariate columns.
    levels : tuple[tuple[str, ...], ...]
        Factor levels per covariate, empty for numeric covariates.

    Returns
    -------
    RawSmooth
        Basis, indicator design, and identity penalty.
    """
    basis = RandomEffectBasis(levels=levels)
    design = basis.evaluate(columns)
    width = design.shape[1]
    return RawSmooth(
        basis=basis,
        design=design,
        penalties=(np.eye(width),),
        rank=(width,),
        null_dim=0,
        constraint="none",
    )


# ---------------------------------------------------------------------------
# Factor-smooth interactions (mgcv smooth.construct.fs.smooth.spec).


# ---------------------------------------------------------------------------
# Markov random field smooths.


class MarkovRandomFieldBasis(ValueModel):
    """Retained state of mgcv's Markov random field smooth ``bs = "mrf"``.

    Attributes
    ----------
    levels : tuple[str, ...]
        Regions, one indicator column each, in factor-level order.
    transform : numpy.ndarray
        Map from the region indicators to the retained coordinates: the
        identity for a full-rank basis, and the low-rank eigenvector map
        ``P`` when ``k`` is below the number of regions.
    """

    kind: Literal["mrf"] = "mrf"
    levels: tuple[str, ...]
    transform: FloatArray = _array_field()

    def evaluate(self, columns: Sequence[npt.ArrayLike]) -> FloatArray:
        """Evaluate the region indicators in the retained coordinates.

        Costs O(rows times regions times retained width).

        Parameters
        ----------
        columns : Sequence[numpy.ndarray]
            The single column of region labels.

        Returns
        -------
        numpy.ndarray
            Rows by retained width.

        Raises
        ------
        ValueError
            If a row holds a region that is not in the fitted structure.
        """
        labels = np.asarray(columns[0]).astype(str)
        unknown = set(labels.tolist()) - set(self.levels)
        if unknown:
            raise ValueError(
                "new regions are not allowed in a Markov random field smooth: "
                f"found {sorted(unknown)!r}, fitted {list(self.levels)!r}"
            )
        indicator = (labels[:, None] == np.asarray(self.levels)[None, :]).astype(float)
        return indicator @ self.transform


def neighbourhood_penalty(adjacency: FloatArray) -> FloatArray:
    """Build mgcv's Markov random field penalty from listed neighbours.

    mgcv sets the diagonal entry of a region to the number of neighbours
    listed for it and the entry of each listed neighbour to ``-1``, which is
    the [graph Laplacian](https://en.wikipedia.org/wiki/Laplacian_matrix)
    when no region lists itself.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Square matrix whose row ``i`` is nonzero at the neighbours listed
        for region ``i``.

    Returns
    -------
    numpy.ndarray
        Penalty matrix.

    Raises
    ------
    ValueError
        If the neighbour relation is not symmetric.
    """
    listed = adjacency != 0
    penalty = np.diag(listed.sum(axis=1).astype(np.float64))
    off = listed & ~np.eye(listed.shape[0], dtype=bool)
    penalty[off] = -1.0
    if not np.array_equal(penalty, penalty.T):
        raise ValueError(
            "the neighbours of a Markov random field must be listed symmetrically"
        )
    return penalty


def markov_random_field(
    labels: npt.NDArray[np.str_],
    levels: tuple[str, ...],
    penalty: FloatArray,
    k: int,
) -> RawSmooth:
    """Construct mgcv's Markov random field smooth over a factor of regions.

    The full-rank basis is one indicator column per region with the given
    penalty, a Gaussian Markov random field prior on the region effects
    [@rue2005gaussian; @wood2017generalized]. With ``k`` below the number
    of regions mgcv reparameterizes by ``nat.param(type = 0)``: for the
    thin QR factorization ``X = Q R`` of the indicators (with one extra row
    for every region without observations) and the eigenpairs
    ``(d_j, u_j)`` of ``R^-T S R^-1`` in decreasing order, it keeps the
    ``k`` columns ``Q u_j`` of smallest eigenvalue, the smoothest ones.
    Costs O(regions cubed) once and O(rows times regions) per evaluation.

    Parameters
    ----------
    labels : numpy.ndarray
        Region label of every training row.
    levels : tuple[str, ...]
        Regions in factor-level order.
    penalty : numpy.ndarray
        Symmetric positive semidefinite penalty in that order.
    k : int
        Basis dimension, negative for the number of regions.

    Returns
    -------
    RawSmooth
        Raw smooth with the sum-to-zero constraint still to be absorbed.

    Raises
    ------
    ValueError
        If the penalty is invalid, ``k`` exceeds the number of regions, or a
        low-rank basis would cut through a repeated eigenvalue or the
        penalty null space, where mgcv's basis is fixed only by rounding.
    """
    count = len(levels)
    if penalty.shape != (count, count):
        raise ValueError(
            f"the Markov random field penalty has shape {penalty.shape}; "
            f"{count} regions need a {count} by {count} matrix"
        )
    if not np.isfinite(penalty).all() or not np.allclose(
        penalty, penalty.T, rtol=0, atol=_EPS**0.8 * float(np.max(np.abs(penalty)))
    ):
        raise ValueError("the Markov random field penalty must be finite and symmetric")
    penalty = (penalty + penalty.T) * 0.5
    spectrum = np.linalg.eigvalsh(penalty)
    largest = float(spectrum[-1])
    threshold = _EPS**0.8 * largest
    if largest <= 0 or float(spectrum[0]) < -threshold:
        raise ValueError(
            "the Markov random field penalty must be positive semidefinite and nonzero"
        )
    width = count if k < 0 else k
    if width > count:
        raise ValueError("MRF basis dimension set too high")
    if width < 1:
        raise ValueError("MRF basis dimension must be positive")
    basis = MarkovRandomFieldBasis(levels=levels, transform=np.eye(count))
    design = basis.evaluate([labels])
    if width == count:
        rank = int(np.sum(spectrum > threshold))
        return RawSmooth(
            basis=basis,
            design=design,
            penalties=(penalty,),
            rank=(rank,),
            null_dim=count - rank,
        )
    # nat.param(type = 0) on the indicators, with one row added for every
    # region without observations so that the design has full column rank.
    counts = np.maximum(design.sum(axis=0), 1.0)
    root = np.sqrt(counts)
    values, vectors = _signed_eigh(penalty / np.outer(root, root))
    rank = int(np.sum(values > float(values[0]) * _EPS**0.8))
    cut = count - width
    if cut >= rank:
        raise ValueError(
            f"k = {width} keeps no penalized coordinate of the Markov random "
            f"field: its penalty null space has dimension {count - rank}"
        )
    if abs(float(values[cut - 1] - values[cut])) <= _MRF_GAP * float(values[0]):
        raise ValueError(
            f"k = {width} cuts through a repeated eigenvalue of the Markov random "
            "field penalty, where mgcv's low-rank basis is fixed only by rounding; "
            "choose another k"
        )
    rows = design.shape[0] + int(np.sum(design.sum(axis=0) == 0))
    transform = vectors / root[:, None]
    scale = math.sqrt(rows)
    transform = transform * scale
    kept = np.arange(cut, count)
    retained = min(rank, count) - min(cut, rank)
    diagonal = np.concatenate(
        (values[kept[kept < rank]] * scale**2, np.zeros(width - retained))
    )
    basis = MarkovRandomFieldBasis(levels=levels, transform=transform[:, kept])
    return RawSmooth(
        basis=basis,
        design=basis.evaluate([labels]),
        penalties=(np.diag(diagonal),),
        rank=(retained,),
        null_dim=width - retained,
    )


type FactorSmoothBase = (
    ThinPlateBasis
    | GaussianProcessSmoothBasis
    | CubicRegressionBasis
    | CyclicBasis
    | BSplineBasis
)
"""Singly penalized bases of numeric covariates that ``bs = "fs"`` can copy."""


class FactorSmoothBasis(ValueModel):
    """Retained state of mgcv's factor-smooth interaction ``bs = "fs"``.

    Attributes
    ----------
    base : FactorSmoothBase
        Singly penalized basis of the numeric covariates, shared by every
        level: thin-plate by default, or the basis named by ``xt``.
    transform : numpy.ndarray
        Map from the base coordinates to the diagonal-penalty coordinates:
        the penalized range columns, then the null-space columns.
    levels : tuple[str, ...]
        Training factor levels, one copy of the smooth each.
    factor : int
        Position of the factor among the term's covariates.
    """

    kind: Literal["fs"] = "fs"
    base: FactorSmoothBase = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    transform: FloatArray = _array_field()
    levels: tuple[str, ...]
    factor: int

    def evaluate(self, columns: Sequence[npt.ArrayLike]) -> FloatArray:
        """Evaluate one copy of the reparameterized base smooth per level.

        Costs O(rows times base width squared).

        Parameters
        ----------
        columns : Sequence[numpy.ndarray]
            One column per covariate in authored order, the factor column
            holding labels.

        Returns
        -------
        numpy.ndarray
            Rows by levels times base width, level-major.

        Raises
        ------
        ValueError
            If the factor column holds a level absent at training time, as
            brms refuses new levels of a smooth's factor.
        """
        labels = np.asarray(columns[self.factor]).astype(str)
        unknown = set(labels.tolist()) - set(self.levels)
        if unknown:
            raise ValueError(
                "new factor levels are not allowed in a factor-smooth "
                f"interaction: found {sorted(unknown)!r}, fitted {list(self.levels)!r}"
            )
        x = np.column_stack(
            [
                np.asarray(column, dtype=np.float64)
                for position, column in enumerate(columns)
                if position != self.factor
            ]
        )
        base = self.base.evaluate(x) @ self.transform
        indicator = (labels[:, None] == np.asarray(self.levels)[None, :]).astype(float)
        return (indicator[:, :, None] * base[:, None, :]).reshape(base.shape[0], -1)


def factor_smooth(
    raw: RawSmooth,
    columns: Sequence[npt.ArrayLike],
    factor: int,
    levels: tuple[str, ...],
) -> RawSmooth:
    """Construct mgcv's factor-smooth interaction over a singly penalized base.

    mgcv reparameterizes the base smooth by ``nat.param(type = 1)`` so that
    its penalty is diagonal, copies it once per factor level, and penalizes
    the range space and every null-space column separately, each with one
    smoothing parameter shared by the levels [@wood2017generalized]. The
    range columns are ``Q u_j / sqrt(d_j)`` for the thin QR factorization
    ``X = Q R`` and the eigenpairs ``(d_j, u_j)`` of ``R^-T S R^-1``. mgcv
    takes the null-space columns from the same eigendecomposition, where they
    belong to a repeated zero eigenvalue and are fixed only up to rotation.
    The penalty null space of every admitted base is the polynomials below
    the penalty order, so this function instead orthonormalizes those
    polynomials of the centered covariates in increasing degree (constant
    first), which spans the same functions. Costs O(rows times width
    squared) beyond the base construction.

    Parameters
    ----------
    raw : RawSmooth
        Base smooth of the numeric covariates, before any constraint.
    columns : Sequence[numpy.ndarray]
        Covariate columns in authored order.
    factor : int
        Position of the factor column.
    levels : tuple[str, ...]
        Factor levels.

    Returns
    -------
    RawSmooth
        Level-expanded design with one diagonal penalty per smoothing
        parameter and no identifiability constraint.

    Raises
    ------
    ValueError
        If the base has several penalties or a full-rank penalty, its design
        is rank deficient, or its penalty null space is not a space of
        polynomials.
    """
    base = raw.basis
    if isinstance(
        base,
        (RandomEffectBasis, FactorSmoothBasis, MarkovRandomFieldBasis, TensorBasis),
    ):
        raise ValueError(
            "the base of a factor-smooth interaction must be a smooth of "
            "numeric covariates"
        )
    if len(raw.penalties) != 1:
        raise ValueError(
            "a factor-smooth interaction cannot use a multiply penalized base "
            "(mgcv refuses the same)"
        )
    x = np.column_stack(
        [
            np.asarray(column, dtype=np.float64)
            for position, column in enumerate(columns)
            if position != factor
        ]
    )
    design, penalty, rank = raw.design, raw.penalties[0], raw.rank[0]
    width = design.shape[1]
    null_dim = width - rank
    if null_dim < 1:
        raise ValueError(
            "the base penalty of a factor-smooth interaction has full rank; "
            "mgcv's construction requires a penalty null space and fails for "
            "such a base (as does brms)"
        )
    q, r = np.linalg.qr(design)
    if (
        float(np.min(np.abs(np.diag(r))))
        <= float(np.max(np.abs(np.diag(r)))) * _EPS**0.8
    ):
        raise ValueError(
            "the base smooth of a factor-smooth interaction is rank deficient"
        )
    inverse = np.linalg.solve(r, np.eye(width))
    values, vectors = _signed_eigh(inverse.T @ penalty @ inverse)
    root = np.sqrt(values[:rank])
    range_design = q @ vectors[:, :rank] / root
    transform = np.zeros((width, width))
    transform[:, :rank] = inverse @ vectors[:, :rank] / root
    scale = 1 / math.sqrt(float(np.mean(range_design**2)))
    transform[:, :rank] *= scale
    dimension = x.shape[1]
    order = next(
        (
            m
            for m in range(1, null_dim + 1)
            if math.comb(m + dimension - 1, dimension) == null_dim
        ),
        0,
    )
    if order == 0:
        raise ValueError(
            f"the base penalty null space has dimension {null_dim}, which is "
            f"no space of polynomials of {dimension} covariates"
        )
    polynomials = _poly(x - x.mean(axis=0), order)
    null_space = q @ vectors[:, rank:]
    outside = polynomials - null_space @ (null_space.T @ polynomials)
    if float(np.max(np.abs(outside))) > _FS_NULL_TOLERANCE * float(
        np.max(np.abs(polynomials))
    ):
        raise ValueError(
            "the base penalty null space is not the polynomials below its penalty order"
        )
    null_q, null_r = np.linalg.qr(polynomials)
    signs = np.sign(np.diag(null_r))
    null_design = null_q * signs
    transform[:, rank:] = inverse @ (q.T @ null_design)
    transform[:, rank:] /= math.sqrt(float(np.mean(null_design**2)))
    basis = FactorSmoothBasis(
        base=base, transform=transform, levels=levels, factor=factor
    )
    count = len(levels)
    penalties = [
        np.diag(
            np.tile(
                np.concatenate((np.full(rank, scale**2), np.zeros(null_dim))), count
            )
        )
    ]
    for column in range(rank, width):
        unit = np.zeros(width)
        unit[column] = 1.0
        penalties.append(np.diag(np.tile(unit, count)))
    return RawSmooth(
        basis=basis,
        design=basis.evaluate(columns),
        penalties=tuple(penalties),
        rank=(rank * count, *([count] * null_dim)),
        null_dim=0,
        constraint="none",
    )


# ---------------------------------------------------------------------------
# Tensor products with separate penalties (mgcv t2.smooth; Wood et al. 2013).


def natural_parameterization(
    design: FloatArray,
    penalty: FloatArray,
    rank: int,
    *,
    kind: Literal[2, 3],
    unit: bool,
) -> tuple[FloatArray, FloatArray]:
    """Reparameterize to a diagonal penalty (mgcv ``nat.param``, types 2 and 3).

    Parameters
    ----------
    design : numpy.ndarray
        Training design.
    penalty : numpy.ndarray
        Penalty matrix.
    rank : int
        Penalty rank.
    kind : {2, 3}
        mgcv type: 3 additionally rotates a null space of dimension at least
        two to the principal axes of its centered design columns.
    unit : bool
        Rescale to unit mean squared design entries.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Reparameterized design and the coefficient map ``P``.
    """
    values, vectors = _signed_eigh(penalty)
    vectors = _canonical_null(design, vectors, rank)
    p = design.shape[1]
    scale_e = np.ones(p)
    scale_e[:rank] = np.sqrt(values[:rank])
    x = design @ vectors
    norms = np.sum(x * x, axis=0) / scale_e**2
    average = float(np.mean(norms[:rank]))
    if rank < p:
        scale_e[rank:] = np.sqrt(norms[rank:] / average)
    transform = vectors / scale_e
    x = x / scale_e
    if kind == 3 and rank < p - 1:
        null = x[:, rank:]
        centered = null - null.mean(axis=0)
        _, axes = _signed_eigh(centered.T @ centered)
        x[:, rank:] = (null @ axes)[:, ::-1]
        transform[:, rank:] = (transform[:, rank:] @ axes)[:, ::-1]
    if unit:
        scale = 1 / np.sqrt(np.mean(x[:, :rank] ** 2))
        x[:, :rank] *= scale
        transform[:rank] *= scale
        if rank < p:
            scale_f = 1 / np.sqrt(np.mean(x[:, rank:] ** 2))
            x[:, rank:] *= scale_f
            transform[rank:] *= scale_f
    return x, transform


class TensorBasis(ValueModel):
    """Retained margins and natural reparameterizations of an mgcv ``t2``.

    Attributes
    ----------
    margins : tuple[SmoothBasis, ...]
        Marginal bases.
    dims : tuple[int, ...]
        Covariates per margin.
    maps : tuple[numpy.ndarray, ...]
        Marginal natural-parameterization maps.
    ranks : tuple[int, ...]
        Marginal penalty ranks.
    full : bool
        mgcv's ``full`` option.
    orders : tuple[int, ...]
        Retained interaction orders, empty for all.
    """

    kind: Literal["t2"] = "t2"
    margins: tuple[SmoothBasis, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    dims: tuple[int, ...]
    maps: tuple[FloatArray, ...] = dx.field(
        opaque=True, extras=opaque_by_value(lambda values: tuple(map(_frozen, values)))
    )
    ranks: tuple[int, ...]
    full: bool = False
    orders: tuple[int, ...] = ()

    @property
    def width(self) -> int:
        """Number of raw basis columns.

        Returns
        -------
        int
            Product of the marginal widths.
        """
        return math.prod(m.shape[1] for m in self.maps)

    def evaluate(self, x: FloatArray) -> FloatArray:
        """Evaluate the retained tensor product at fresh rows.

        Costs O(rows times basis width) after the marginal evaluations.

        Parameters
        ----------
        x : numpy.ndarray
            Covariate rows, shape rows by total dimension.

        Returns
        -------
        numpy.ndarray
            Rows by basis width.
        """
        marginal = []
        start = 0
        for basis, dim, transform in zip(
            self.margins, self.dims, self.maps, strict=True
        ):
            marginal.append(
                _evaluate_basis(basis, x[:, start : start + dim]) @ transform
            )
            start += dim
        design, _ = t2_model_matrix(
            marginal, self.ranks, full=self.full, orders=self.orders
        )
        return design


def t2_model_matrix(
    margins: Sequence[FloatArray],
    ranks: Sequence[int],
    *,
    full: bool,
    orders: Sequence[int],
) -> tuple[FloatArray, tuple[int, ...]]:
    """Form mgcv's ``t2.model.matrix`` and its penalized block widths.

    Parameters
    ----------
    margins : Sequence[numpy.ndarray]
        Naturally parameterized marginal designs, penalized columns first.
    ranks : Sequence[int]
        Marginal penalty ranks.
    full : bool
        Penalize each null-space column product separately.
    orders : Sequence[int]
        Retained interaction orders, empty for all.

    Returns
    -------
    tuple[numpy.ndarray, tuple[int, ...]]
        Tensor design and the width of each penalized block, in order; the
        trailing unpenalized block is not listed.
    """
    first = margins[0]
    blocks = [first[:, : ranks[0]]]
    order = [1]
    penalized = [True]
    no_null = ranks[0] >= first.shape[1]
    if not no_null:
        blocks.append(first[:, ranks[0] :])
        order.append(0)
        penalized.append(False)
    rows = first.shape[0]
    for margin, rank in zip(margins[1:], ranks[1:], strict=True):
        z = margin[:, :rank]
        null_exists = rank < margin.shape[1]
        xi = margin[:, rank:] if null_exists else np.zeros((rows, 0))
        if not null_exists:
            no_null = True
        previous, previous_order, previous_pen = blocks, order, penalized
        blocks, order, penalized = [], [], []
        for block, block_order, pen in zip(
            previous, previous_order, previous_pen, strict=True
        ):
            if not full or pen:
                blocks.append((block[:, :, None] * z[:, None, :]).reshape(rows, -1))
                order.append(block_order + 1)
                penalized.append(True)
            else:
                for j in range(block.shape[1]):
                    blocks.append(block[:, j : j + 1] * z)
                    order.append(block_order + 1)
                    penalized.append(True)
        if null_exists:
            for block, block_order, pen in zip(
                previous, previous_order, previous_pen, strict=True
            ):
                if not full or not pen:
                    blocks.append(
                        (block[:, :, None] * xi[:, None, :]).reshape(rows, -1)
                    )
                    order.append(block_order)
                    penalized.append(False if full else True)
                else:
                    for j in range(xi.shape[1]):
                        blocks.append(block * xi[:, j : j + 1])
                        order.append(block_order)
                        penalized.append(True)
    if orders:
        keep = [o in orders for o in order]
        blocks = [b for b, k in zip(blocks, keep, strict=True) if k]
        if 0 not in orders:
            no_null = True
    widths = [b.shape[1] for b in blocks]
    design = np.column_stack(blocks)
    if not no_null:
        widths = widths[:-1]
    return design, tuple(widths)


def tensor_t2(
    margins: Sequence[RawSmooth],
    dims: Sequence[int],
    *,
    full: bool,
    orders: Sequence[int],
) -> RawSmooth:
    """Construct mgcv's ``t2`` tensor product with separate penalties.

    Each margin is reparameterized so its penalty is the identity on its
    range; products of marginal range and null columns form blocks, each
    with its own identity penalty [@wood2013straightforward].

    Parameters
    ----------
    margins : Sequence[RawSmooth]
        Constructed marginal smooths, without constraints.
    dims : Sequence[int]
        Covariates per margin.
    full : bool
        mgcv's ``full`` option.
    orders : Sequence[int]
        Retained interaction orders.

    Returns
    -------
    RawSmooth
        Tensor basis, design, and block penalties.

    Raises
    ------
    ValueError
        If a margin carries more than one penalty.
    """
    maps, natural, ranks = [], [], []
    for margin in margins:
        if len(margin.penalties) != 1:
            raise ValueError("t2 margins must carry exactly one penalty")
        x, transform = natural_parameterization(
            margin.design, margin.penalties[0], margin.rank[0], kind=3, unit=True
        )
        natural.append(x)
        maps.append(transform)
        ranks.append(margin.rank[0])
    design, widths = t2_model_matrix(natural, ranks, full=full, orders=orders)
    p = design.shape[1]
    penalties = []
    start = 0
    for width in widths:
        diagonal = np.zeros(p)
        diagonal[start : start + width] = 1.0
        penalties.append(np.diag(diagonal))
        start += width
    basis = TensorBasis(
        margins=tuple(margin.basis for margin in margins),
        dims=tuple(dims),
        maps=tuple(maps),
        ranks=tuple(ranks),
        full=full,
        orders=tuple(orders),
    )
    return RawSmooth(
        basis=basis,
        design=design,
        penalties=tuple(penalties),
        rank=widths,
        null_dim=p - sum(widths),
        constraint="t2",
    )


type SmoothBasis = (
    ThinPlateBasis
    | CubicRegressionBasis
    | CyclicBasis
    | BSplineBasis
    | RandomEffectBasis
    | GaussianProcessSmoothBasis
    | SphereBasis
    | FactorSmoothBasis
    | MarkovRandomFieldBasis
    | TensorBasis
)


class RawSmooth(ValueModel):
    """One constructed smooth before mgcv's constraint absorption.

    Attributes
    ----------
    basis : SmoothBasis
        Retained evaluator.
    design : numpy.ndarray
        Training rows evaluated in the raw basis.
    penalties : tuple[numpy.ndarray, ...]
        Raw penalty matrices.
    rank : tuple[int, ...]
        Penalty ranks as mgcv records them.
    null_dim : int
        Dimension of the joint penalty null space.
    constraint : Literal["mean", "none", "t2"]
        Identifiability constraint family mgcv applies to this basis.
    fixed : bool
        Whether the smooth is an unpenalized regression spline.
    """

    basis: SmoothBasis = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    design: FloatArray = _array_field()
    penalties: tuple[FloatArray, ...] = dx.field(
        opaque=True, extras=opaque_by_value(lambda values: tuple(map(_frozen, values)))
    )
    rank: tuple[int, ...]
    null_dim: int
    constraint: Literal["mean", "none", "t2"] = "mean"
    fixed: bool = False


def _evaluate_basis(basis: SmoothBasis, x: FloatArray) -> FloatArray:
    """Evaluate any retained numeric-covariate basis.

    Parameters
    ----------
    basis : SmoothBasis
        Retained basis state.
    x : numpy.ndarray
        Covariate rows, shape rows by dimension.

    Returns
    -------
    numpy.ndarray
        Rows by basis width.

    Raises
    ------
    ValueError
        If a random-effect basis is evaluated as a numeric margin.
    """
    if isinstance(
        basis, (RandomEffectBasis, FactorSmoothBasis, MarkovRandomFieldBasis)
    ):
        raise ValueError(
            "random-effect, factor-smooth, and Markov random field bases cannot "
            "be tensor product margins"
        )
    return basis.evaluate(x)


# ---------------------------------------------------------------------------
# smoothCon post-processing and smooth2random.


def _householder(vector: FloatArray) -> FloatArray:
    """LINPACK ``dqrdc2`` reflector mapping a vector onto its first axis.

    Parameters
    ----------
    vector : numpy.ndarray
        Nonzero constraint vector.

    Returns
    -------
    numpy.ndarray
        Symmetric orthogonal reflector ``H`` with ``H v`` a multiple of the
        first coordinate axis.
    """
    norm = float(np.linalg.norm(vector))
    signed = math.copysign(norm, vector[0]) if vector[0] != 0 else norm
    u = vector / signed
    u[0] += 1.0
    return np.eye(vector.size) - np.outer(u, u) / u[0]


def constraint_map(raw: RawSmooth, *, centered: bool) -> FloatArray:
    """Return mgcv's absorbed-constraint coefficient map ``Z``.

    Parameters
    ----------
    raw : RawSmooth
        Constructed smooth.
    centered : bool
        Whether the sum-to-zero constraint applies (a non-constant numeric
        ``by`` variable removes it).

    Returns
    -------
    numpy.ndarray
        Raw width by constrained width, with ``X Z`` the absorbed design.
    """
    x = raw.design
    p = x.shape[1]
    if raw.constraint == "none" or not centered:
        return np.eye(p)
    if raw.constraint == "t2":
        penalized = sum(raw.rank)
        if raw.null_dim == 0:
            return np.eye(p)
        if raw.null_dim == 1:
            return np.delete(np.eye(p), p - 1, axis=1)
        c = np.concatenate((np.zeros(penalized), x[:, penalized:].sum(axis=0)))
    else:
        c = x.mean(axis=0)
    active = np.flatnonzero(c != 0)
    reflector = _householder(c[active].copy())
    z = np.eye(p)
    if active.size < p:
        kept = active[:-1]
        z[np.ix_(active, kept)] = reflector[:, 1:]
        return np.delete(z, active[-1], axis=1)
    return reflector[:, 1:]


class SmoothBlock(ValueModel):
    """One mixed-model block of an mgcv smooth after ``smooth2random``.

    Attributes
    ----------
    label : str
        mgcv label, including any factor ``by`` level.
    level : str or None
        Factor ``by`` level selecting this block's rows, or None.
    coefficient_map : numpy.ndarray
        Raw basis to evaluated coordinates: unpenalized columns first, then
        each penalized group in order.
    fixed_width : int
        Number of unpenalized columns.
    random_widths : tuple[int, ...]
        Widths of the penalized groups, each with its own scale.
    """

    label: str
    level: str | None = None
    coefficient_map: FloatArray = _array_field()
    fixed_width: int
    random_widths: tuple[int, ...]


def mixed_model_blocks(
    raw: RawSmooth,
    *,
    label: str,
    by_values: FloatArray | None,
    by_levels: tuple[str, ...],
    by_labels: npt.NDArray[np.str_] | None,
    by_name: str,
) -> tuple[FloatArray, tuple[SmoothBlock, ...]]:
    """Run brms's ``smoothCon`` and ``smooth2random`` steps on one smooth.

    Parameters
    ----------
    raw : RawSmooth
        Constructed smooth.
    label : str
        mgcv smooth label.
    by_values : numpy.ndarray or None
        Numeric ``by`` multipliers, or None.
    by_levels : tuple[str, ...]
        Factor ``by`` levels producing one block each, or empty.
    by_labels : numpy.ndarray or None
        Factor ``by`` labels of the training rows.
    by_name : str
        ``by`` variable name used in block labels.

    Returns
    -------
    tuple[numpy.ndarray, tuple[SmoothBlock, ...]]
        Training raw-basis constraint vector and the mixed-model blocks.

    Raises
    ------
    ValueError
        If the smooth carries several penalties outside a tensor product.
    """
    x = raw.design
    penalties = list(raw.penalties)
    if not raw.fixed and penalties:
        norm_x = float(np.max(np.sum(np.abs(x), axis=1))) ** 2
        penalties = [
            s / (float(np.max(np.sum(np.abs(s), axis=0))) / norm_x) for s in penalties
        ]
    centered = True
    if by_values is not None:
        spread = float(np.std(by_values, ddof=1)) if by_values.size > 1 else 0.0
        centered = not spread > float(np.mean(by_values)) * _EPS * 1000
    z = constraint_map(raw, centered=centered)
    constrained = [z.T @ s @ z for s in penalties]
    rank = tuple(min(r, z.shape[1]) for r in raw.rank)
    if by_levels:
        assert by_labels is not None
        designs = [
            (by_labels == level).astype(float)[:, None] * x for level in by_levels
        ]
        labels = [f"{label}:{by_name}{level}" for level in by_levels]
    else:
        multiplier = 1.0 if by_values is None else by_values[:, None]
        designs = [multiplier * x]
        labels = [label if by_values is None else f"{label}:{by_name}"]
    maps = [z for _ in designs]
    if raw.fixed or not constrained:
        return x.mean(axis=0), tuple(
            SmoothBlock(
                label=name,
                level=level,
                coefficient_map=transform,
                fixed_width=transform.shape[1],
                random_widths=(),
            )
            for name, level, transform in zip(
                labels, by_levels or (None,), maps, strict=True
            )
        )
    if len(constrained) == 1:
        s = constrained[0]
        diagonal = s[0, 0]
        width = s.shape[0]
        target = np.diag(
            np.concatenate((np.full(rank[0], diagonal), np.zeros(width - rank[0])))
        )
        if np.max(np.abs(s - target)) > abs(diagonal) * _EPS**0.8:
            _, natural = natural_parameterization(
                designs[0] @ z, s, rank[0], kind=2, unit=False
            )
            maps = [transform @ natural for transform in maps]
            s = np.diag(np.concatenate((np.ones(rank[0]), np.zeros(width - rank[0]))))
        values, vectors = _signed_eigh(s)
        scale = 1 / np.sqrt(
            np.concatenate((values[: rank[0]], np.ones(width - rank[0])))
        )
        maps = [transform @ (vectors * scale) for transform in maps]
        random_widths: tuple[int, ...] = (rank[0],)
        order = np.concatenate((np.arange(rank[0], width), np.arange(rank[0])))
        fixed_width = width - rank[0]
    else:
        width = constrained[0].shape[0]
        scale = np.ones(width)
        penalized = np.zeros(width, dtype=bool)
        groups = []
        for s in constrained:
            diagonal = np.diag(s)
            index = np.flatnonzero(diagonal != 0)
            scale[index] = 1 / np.sqrt(diagonal[index])
            penalized[index] = True
            groups.append(index)
        maps = [transform * scale for transform in maps]
        fixed = np.flatnonzero(~penalized)
        order = np.concatenate((fixed, *groups))
        fixed_width = fixed.size
        random_widths = tuple(group.size for group in groups)
    blocks = tuple(
        SmoothBlock(
            label=name,
            level=level,
            coefficient_map=transform[:, order],
            fixed_width=fixed_width,
            random_widths=random_widths,
        )
        for name, level, transform in zip(
            labels, by_levels or (None,), maps, strict=True
        )
    )
    return x.mean(axis=0), blocks


__all__: list[str] = []
