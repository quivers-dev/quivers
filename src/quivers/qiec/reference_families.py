"""The reference backend's samplers and densities beyond the scalar core.

Every family of the semantic registry has a plain-Python sampler and log
density here or in :mod:`quivers.qiec.distributions`, so the reference
machine runs any model end to end and can be held to an independent
oracle. The structured families work on nested tuples: a vector is a
tuple of floats and a matrix a tuple of rows. The small linear algebra
they need, Cholesky factors, triangular solves, and determinants, is
written out here rather than taken from a host library, which is the
point of a reference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
import random

from quivers.qiec.families import FAMILIES
from quivers.qiec.reference_support import (
    Density,
    DistributionError,
    Sampler,
    finite_value,
    log_choose,
    normal_log_prob,
    probability,
    real_parameter,
    simplex,
    vector_parameter,
)

type Vector = tuple[float, ...]
type Matrix = tuple[tuple[float, ...], ...]

#: The relaxation temperature of the concrete Bernoulli and one-hot
#: categorical families when a construction leaves it out.
RELAXATION_TEMPERATURE = 0.5


def relaxation_temperature(a: Mapping[str, object], family: str) -> float:
    """The temperature a relaxed family is constructed at.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    float
        The ``temperature`` parameter, or the default when absent.

    Raises
    ------
    DistributionError
        If the temperature is not a positive number.
    """
    if "temperature" not in a:
        return RELAXATION_TEMPERATURE
    temperature = real_parameter(a, "temperature", family)
    if not temperature > 0.0:
        raise DistributionError(f"{family} needs a positive temperature")
    return temperature


#: Standard Gauss-Legendre nodes and weights on ``[-1, 1]`` for the
#: horseshoe marginal, mapped to ``[0, 1]`` where they are used.
_GAUSS_LEGENDRE_16: tuple[tuple[float, float], ...] = (
    (-0.9894009349916499, 0.0271524594117541),
    (-0.9445750230732326, 0.0622535239386479),
    (-0.8656312023878318, 0.0951585116824928),
    (-0.7554044083550030, 0.1246289712555339),
    (-0.6178762444026438, 0.1495959888165767),
    (-0.4580167776572274, 0.1691565193950025),
    (-0.2816035507792589, 0.1826034150449236),
    (-0.0950125098376374, 0.1894506104550685),
    (0.0950125098376374, 0.1894506104550685),
    (0.2816035507792589, 0.1826034150449236),
    (0.4580167776572274, 0.1691565193950025),
    (0.6178762444026438, 0.1495959888165767),
    (0.7554044083550030, 0.1246289712555339),
    (0.8656312023878318, 0.0951585116824928),
    (0.9445750230732326, 0.0622535239386479),
    (0.9894009349916499, 0.0271524594117541),
)


# ---------------------------------------------------------------------------
# scalar helpers


def _logsumexp(values: Sequence[float]) -> float:
    """The log of a sum of exponentials, stable for large magnitudes.

    Parameters
    ----------
    values : Sequence[float]
        The log terms.

    Returns
    -------
    float
        ``log(sum(exp(v)))``, ``-inf`` for no finite term.
    """
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    peak = max(finite)
    return peak + math.log(sum(math.exp(value - peak) for value in finite))


def _sigmoid(value: float) -> float:
    """The logistic function.

    Parameters
    ----------
    value : float
        The logit.

    Returns
    -------
    float
        ``1 / (1 + exp(-value))``, computed without overflow.
    """
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _log_sigmoid(value: float) -> float:
    """The log of the logistic function.

    Parameters
    ----------
    value : float
        The logit.

    Returns
    -------
    float
        ``log(sigmoid(value))``, computed without overflow.
    """
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def _normal_cdf(value: float) -> float:
    """The standard normal distribution function.

    Parameters
    ----------
    value : float
        The point.

    Returns
    -------
    float
        ``P(Z <= value)``.
    """
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _normal_icdf(probability_: float) -> float:
    """The standard normal quantile function.

    Parameters
    ----------
    probability_ : float
        A probability strictly between zero and one.

    Returns
    -------
    float
        The point ``z`` with ``P(Z <= z)`` equal to the probability,
        from a rational approximation refined by Newton steps on
        :func:`_normal_cdf`.

    Raises
    ------
    DistributionError
        If the probability is not strictly between zero and one.
    """
    if not 0.0 < probability_ < 1.0:
        raise DistributionError("a normal quantile needs a probability in (0, 1)")
    # Acklam's rational approximation, then Newton refinement.
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    low = 0.02425
    if probability_ < low:
        q = math.sqrt(-2 * math.log(probability_))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    elif probability_ > 1 - low:
        q = math.sqrt(-2 * math.log(1 - probability_))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    else:
        q = probability_ - 0.5
        r = q * q
        z = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    for _ in range(3):
        error = _normal_cdf(z) - probability_
        density = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        if density == 0.0:
            break
        z -= error / density
    return z


def _log_beta(a: float, b: float) -> float:
    """The log of the beta function.

    Parameters
    ----------
    a : float
        The first argument.
    b : float
        The second argument.

    Returns
    -------
    float
        ``lgamma(a) + lgamma(b) - lgamma(a + b)``.
    """
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _log_multivariate_gamma(a: float, dimension: int) -> float:
    """The log of the multivariate gamma function.

    Parameters
    ----------
    a : float
        The argument.
    dimension : int
        The dimension.

    Returns
    -------
    float
        ``log Gamma_d(a)``.
    """
    return dimension * (dimension - 1) / 4 * math.log(math.pi) + sum(
        math.lgamma(a - index / 2) for index in range(dimension)
    )


def _bessel_i0(value: float) -> float:
    """The modified Bessel function of the first kind and order zero.

    Parameters
    ----------
    value : float
        The argument.

    Returns
    -------
    float
        ``I_0(value)``, from its power series, which converges for every
        argument and is summed until a term no longer changes the sum.
    """
    total = 1.0
    term = 1.0
    quarter = value * value / 4.0
    index = 1
    while True:
        term *= quarter / (index * index)
        total += term
        if term < total * 1e-17:
            return total
        index += 1


def _log_bessel_i0(value: float) -> float:
    """The log of the modified Bessel function of order zero.

    Parameters
    ----------
    value : float
        The argument.

    Returns
    -------
    float
        ``log I_0(value)``; the asymptotic expansion carries large
        arguments the series would overflow on.
    """
    magnitude = abs(value)
    if magnitude < 500.0:
        return math.log(_bessel_i0(magnitude))
    # I_0(x) ~ exp(x) / sqrt(2 pi x) * (1 + 1/(8x) + 9/(128 x^2) + ...)
    inverse = 1.0 / magnitude
    correction = 1.0 + inverse / 8.0 + 9.0 * inverse * inverse / 128.0
    return magnitude - 0.5 * math.log(2 * math.pi * magnitude) + math.log(correction)


# ---------------------------------------------------------------------------
# vectors and matrices


def _matrix_parameter(
    arguments: Mapping[str, object], name: str, family: str
) -> Matrix:
    """Read one matrix-valued parameter.

    Parameters
    ----------
    arguments : Mapping[str, object]
        The named parameters.
    name : str
        The parameter wanted.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    Matrix
        The parameter's rows.

    Raises
    ------
    DistributionError
        If the parameter is absent, not a sequence of equal-length rows,
        or holds a non-number.
    """
    try:
        value = arguments[name]
    except KeyError as error:
        raise DistributionError(
            f"reference backend needs parameter {name!r} of {family}"
        ) from error
    return _as_matrix(value, f"parameter {name!r} of {family}")


def _as_vector(value: object, subject: str) -> Vector:
    """Read a host value as a vector.

    Parameters
    ----------
    value : object
        The value.
    subject : str
        What the value is, for the diagnostic.

    Returns
    -------
    Vector
        The components.

    Raises
    ------
    DistributionError
        If the value is not a sequence of numbers.
    """
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise DistributionError(f"{subject} is not a vector")
    components: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise DistributionError(f"{subject} is not a vector of numbers")
        components.append(float(item))
    return tuple(components)


def _as_matrix(value: object, subject: str) -> Matrix:
    """Read a host value as a matrix.

    Parameters
    ----------
    value : object
        The value.
    subject : str
        What the value is, for the diagnostic.

    Returns
    -------
    Matrix
        The rows.

    Raises
    ------
    DistributionError
        If the value is not a sequence of equal-length vectors.
    """
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise DistributionError(f"{subject} is not a matrix")
    rows = tuple(_as_vector(row, subject) for row in value)
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise DistributionError(f"{subject} has rows of different lengths")
    return rows


def _transpose(matrix: Matrix) -> Matrix:
    """Transpose a matrix.

    Parameters
    ----------
    matrix : Matrix
        The rows.

    Returns
    -------
    Matrix
        The columns as rows.
    """
    if not matrix:
        return ()
    return tuple(
        tuple(matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    )


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    """Multiply two matrices.

    Parameters
    ----------
    left : Matrix
        The left factor.
    right : Matrix
        The right factor.

    Returns
    -------
    Matrix
        The product.

    Raises
    ------
    DistributionError
        If the inner dimensions differ.
    """
    if left and right and len(left[0]) != len(right):
        raise DistributionError("matrix product of incompatible shapes")
    columns = _transpose(right)
    return tuple(
        tuple(
            math.fsum(a * b for a, b in zip(row, column, strict=True))
            for column in columns
        )
        for row in left
    )


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    """Apply a matrix to a vector.

    Parameters
    ----------
    matrix : Matrix
        The rows.
    vector : Vector
        The vector.

    Returns
    -------
    Vector
        The product.

    Raises
    ------
    DistributionError
        If the widths differ.
    """
    if matrix and len(matrix[0]) != len(vector):
        raise DistributionError("matrix applied to a vector of the wrong length")
    return tuple(
        math.fsum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix
    )


def _cholesky(matrix: Matrix, subject: str) -> Matrix:
    """The lower Cholesky factor of a symmetric positive definite matrix.

    Parameters
    ----------
    matrix : Matrix
        The matrix.
    subject : str
        What the matrix is, for the diagnostic.

    Returns
    -------
    Matrix
        The lower triangular ``L`` with ``L L^T`` the matrix.

    Raises
    ------
    DistributionError
        If the matrix is not square or not positive definite.
    """
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise DistributionError(f"{subject} is not square")
    factor = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            total = matrix[i][j] - math.fsum(
                factor[i][k] * factor[j][k] for k in range(j)
            )
            if i == j:
                if total <= 0.0:
                    raise DistributionError(f"{subject} is not positive definite")
                factor[i][j] = math.sqrt(total)
            else:
                factor[i][j] = total / factor[j][j]
    return tuple(tuple(row) for row in factor)


def _solve_lower(factor: Matrix, vector: Vector) -> Vector:
    """Solve ``L x = b`` for lower triangular ``L``.

    Parameters
    ----------
    factor : Matrix
        The lower triangular matrix.
    vector : Vector
        The right-hand side.

    Returns
    -------
    Vector
        The solution.
    """
    solution: list[float] = []
    for i, row in enumerate(factor):
        total = vector[i] - math.fsum(row[k] * solution[k] for k in range(i))
        solution.append(total / row[i])
    return tuple(solution)


def _solve_upper(factor: Matrix, vector: Vector) -> Vector:
    """Solve ``U x = b`` for upper triangular ``U``.

    Parameters
    ----------
    factor : Matrix
        The upper triangular matrix.
    vector : Vector
        The right-hand side.

    Returns
    -------
    Vector
        The solution.
    """
    size = len(factor)
    solution = [0.0] * size
    for i in range(size - 1, -1, -1):
        total = vector[i] - math.fsum(
            factor[i][k] * solution[k] for k in range(i + 1, size)
        )
        solution[i] = total / factor[i][i]
    return tuple(solution)


def _spd_solve(factor: Matrix, vector: Vector) -> Vector:
    """Solve ``A x = b`` given the lower Cholesky factor of ``A``.

    Parameters
    ----------
    factor : Matrix
        The lower Cholesky factor.
    vector : Vector
        The right-hand side.

    Returns
    -------
    Vector
        The solution.
    """
    return _solve_upper(_transpose(factor), _solve_lower(factor, vector))


def _spd_inverse(factor: Matrix) -> Matrix:
    """Invert a symmetric positive definite matrix from its Cholesky factor.

    Parameters
    ----------
    factor : Matrix
        The lower Cholesky factor.

    Returns
    -------
    Matrix
        The inverse.
    """
    size = len(factor)
    columns = [
        _spd_solve(factor, tuple(1.0 if i == j else 0.0 for i in range(size)))
        for j in range(size)
    ]
    return _transpose(tuple(columns))


def _log_det_from_cholesky(factor: Matrix) -> float:
    """The log determinant of ``L L^T`` from its Cholesky factor.

    Parameters
    ----------
    factor : Matrix
        The lower Cholesky factor.

    Returns
    -------
    float
        ``2 * sum(log(diag(L)))``.
    """
    return 2.0 * math.fsum(math.log(factor[i][i]) for i in range(len(factor)))


def _trace(matrix: Matrix) -> float:
    """The trace of a square matrix.

    Parameters
    ----------
    matrix : Matrix
        The matrix.

    Returns
    -------
    float
        The sum of the diagonal.
    """
    return math.fsum(matrix[i][i] for i in range(len(matrix)))


def _mvn_log_prob(point: Vector, loc: Vector, factor: Matrix) -> float:
    """The multivariate normal log density from a Cholesky factor.

    Parameters
    ----------
    point : Vector
        The point.
    loc : Vector
        The mean.
    factor : Matrix
        The lower Cholesky factor of the covariance.

    Returns
    -------
    float
        The log density.

    Raises
    ------
    DistributionError
        If the point's length differs from the mean's.
    """
    if len(point) != len(loc):
        raise DistributionError("a multivariate normal is scored at the wrong length")
    residual = tuple(a - b for a, b in zip(point, loc, strict=True))
    whitened = _solve_lower(factor, residual)
    return (
        -0.5 * math.fsum(item * item for item in whitened)
        - 0.5 * _log_det_from_cholesky(factor)
        - 0.5 * len(loc) * math.log(2 * math.pi)
    )


def _draw_mvn(loc: Vector, factor: Matrix, rng: random.Random) -> Vector:
    """Draw from a multivariate normal through its Cholesky factor.

    Parameters
    ----------
    loc : Vector
        The mean.
    factor : Matrix
        The lower Cholesky factor of the covariance.
    rng : random.Random
        The generator.

    Returns
    -------
    Vector
        The draw.
    """
    noise = tuple(rng.gauss(0.0, 1.0) for _ in loc)
    return tuple(a + b for a, b in zip(loc, _matvec(factor, noise), strict=True))


def _covariance_factor(arguments: Mapping[str, object], family: str) -> Matrix:
    """The Cholesky factor of a covariance given in any of three forms.

    Parameters
    ----------
    arguments : Mapping[str, object]
        The named parameters, holding ``covariance_matrix``,
        ``precision_matrix``, or ``scale_tril``.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    Matrix
        The lower Cholesky factor of the covariance.

    Raises
    ------
    DistributionError
        If none of the three is supplied.
    """
    if "scale_tril" in arguments:
        return _matrix_parameter(arguments, "scale_tril", family)
    if "covariance_matrix" in arguments:
        return _cholesky(
            _matrix_parameter(arguments, "covariance_matrix", family),
            f"covariance of {family}",
        )
    if "precision_matrix" in arguments:
        precision = _matrix_parameter(arguments, "precision_matrix", family)
        covariance = _spd_inverse(_cholesky(precision, f"precision of {family}"))
        return _cholesky(covariance, f"covariance of {family}")
    raise DistributionError(
        f"{family} needs covariance_matrix, precision_matrix, or scale_tril"
    )


def _integrate(density: Callable[[float], float], low: float, high: float) -> float:
    """Integrate a density over an interval, either end possibly infinite.

    Parameters
    ----------
    density : Callable[[float], float]
        The integrand.
    low : float
        The lower end.
    high : float
        The upper end.

    Returns
    -------
    float
        The integral, by adaptive Simpson quadrature; an infinite end is
        mapped onto a finite one by ``x = c + t / (1 - t)``.
    """
    if low >= high:
        return 0.0
    if math.isinf(low) and math.isinf(high):
        return _integrate(density, low, 0.0) + _integrate(density, 0.0, high)
    if math.isinf(high):

        def mapped_high(t: float) -> float:
            """The integrand over ``[0, 1)`` standing for ``[low, inf)``.

            Parameters
            ----------
            t : float
                The mapped coordinate.

            Returns
            -------
            float
                The density at ``low + t / (1 - t)`` times the map's
                derivative, zero at the far end.
            """
            if t >= 1.0:
                return 0.0
            return density(low + t / (1.0 - t)) / (1.0 - t) ** 2

        return _adaptive_simpson(mapped_high, 0.0, 1.0)
    if math.isinf(low):

        def mapped_low(t: float) -> float:
            """The integrand over ``[0, 1)`` standing for ``(-inf, high]``.

            Parameters
            ----------
            t : float
                The mapped coordinate.

            Returns
            -------
            float
                The density at ``high - t / (1 - t)`` times the map's
                derivative, zero at the far end.
            """
            if t >= 1.0:
                return 0.0
            return density(high - t / (1.0 - t)) / (1.0 - t) ** 2

        return _adaptive_simpson(mapped_low, 0.0, 1.0)
    return _adaptive_simpson(density, low, high)


def _adaptive_simpson(
    function: Callable[[float], float], low: float, high: float
) -> float:
    """Integrate over a finite interval by adaptive Simpson quadrature.

    Parameters
    ----------
    function : Callable[[float], float]
        The integrand.
    low : float
        The lower end.
    high : float
        The upper end.

    Returns
    -------
    float
        The integral to a relative tolerance near machine precision on
        smooth integrands, with subdivision bounded in depth.
    """
    # Seed with a few panels so a peaked integrand is not missed by the
    # first coarse estimate.
    panels = 16
    width = (high - low) / panels
    total = 0.0
    for index in range(panels):
        a = low + index * width
        b = a + width
        fa, fm, fb = function(a), function((a + b) / 2), function(b)
        whole = (b - a) / 6 * (fa + 4 * fm + fb)
        total += _simpson_refine(function, a, b, fa, fm, fb, whole, 1e-12, 40)
    return total


def _simpson_refine(
    function: Callable[[float], float],
    a: float,
    b: float,
    fa: float,
    fm: float,
    fb: float,
    whole: float,
    tolerance: float,
    depth: int,
) -> float:
    """One step of adaptive Simpson refinement.

    Parameters
    ----------
    function : Callable[[float], float]
        The integrand.
    a : float
        The panel's lower end.
    b : float
        The panel's upper end.
    fa : float
        The integrand at ``a``.
    fm : float
        The integrand at the midpoint.
    fb : float
        The integrand at ``b``.
    whole : float
        The Simpson estimate over the panel.
    tolerance : float
        The absolute tolerance for this panel.
    depth : int
        The subdivisions still allowed.

    Returns
    -------
    float
        The refined estimate.
    """
    m = (a + b) / 2
    lm, rm = (a + m) / 2, (m + b) / 2
    flm, frm = function(lm), function(rm)
    left = (m - a) / 6 * (fa + 4 * flm + fm)
    right = (b - m) / 6 * (fm + 4 * frm + fb)
    if depth <= 0 or abs(left + right - whole) <= 15 * tolerance:
        return left + right + (left + right - whole) / 15
    return _simpson_refine(
        function, a, m, fa, flm, fm, left, tolerance / 2, depth - 1
    ) + _simpson_refine(function, m, b, fm, frm, fb, right, tolerance / 2, depth - 1)


# ---------------------------------------------------------------------------
# nested sampleables


def _sampleable(arguments: Mapping[str, object], name: str, family: str) -> object:
    """Read a nested distribution parameter.

    Parameters
    ----------
    arguments : Mapping[str, object]
        The named parameters.
    name : str
        The parameter wanted.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    object
        The nested distribution, which offers ``log_prob`` and ``sample``.

    Raises
    ------
    DistributionError
        If the parameter is absent or offers no density.
    """
    try:
        value = arguments[name]
    except KeyError as error:
        raise DistributionError(
            f"reference backend needs parameter {name!r} of {family}"
        ) from error
    if not callable(getattr(value, "log_prob", None)) or not callable(
        getattr(value, "sample", None)
    ):
        raise DistributionError(f"parameter {name!r} of {family} is not a distribution")
    return value


def _components(arguments: Mapping[str, object], family: str) -> tuple[object, ...]:
    """The components of a mixture, one distribution each.

    Parameters
    ----------
    arguments : Mapping[str, object]
        The family's named parameters, whose ``component`` is either a
        distribution plated over the mixture axis or a tuple of
        distributions.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    tuple[object, ...]
        The components, each offering ``log_prob``, ``sample``, and
        ``log_mass``.

    Raises
    ------
    DistributionError
        If the parameter is absent, a plated distribution has no batch
        axis to index, or an entry of the tuple is not a distribution.
    """
    try:
        component = arguments["component"]
    except KeyError as error:
        raise DistributionError(
            f"reference backend needs parameter 'component' of {family}"
        ) from error
    if isinstance(component, tuple):
        for item in component:
            if not callable(getattr(item, "log_prob", None)):
                raise DistributionError(
                    f"a component of {family} is not a distribution"
                )
        return component
    batch = getattr(component, "batch", ())
    at = getattr(component, "at", None)
    if not batch or not callable(at):
        raise DistributionError(
            f"the component of {family} must be plated over the mixture axis or "
            "be a vector of distributions"
        )
    return tuple(at((index,)) for index in range(int(batch[0])))


def _log_mass_of(distribution: object) -> float:
    """The log mass of a nested distribution.

    Parameters
    ----------
    distribution : object
        The distribution.

    Returns
    -------
    float
        Its ``log_mass``, zero for a distribution that offers none, which
        is a probability measure.
    """
    log_mass = getattr(distribution, "log_mass", None)
    if callable(log_mass):
        return float(log_mass())
    return 0.0


# ---------------------------------------------------------------------------
# scalar continuous families


def _logitnormal_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``LogitNormal``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return _sigmoid(
        rng.gauss(
            real_parameter(a, "loc", "LogitNormal"),
            real_parameter(a, "scale", "LogitNormal"),
        )
    )


def _logitnormal_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``LogitNormal`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the open unit interval.
    """
    point = finite_value(value, "LogitNormal")
    if not 0.0 < point < 1.0:
        return -math.inf
    logit = math.log(point) - math.log1p(-point)
    return (
        normal_log_prob(
            logit,
            real_parameter(a, "loc", "LogitNormal"),
            real_parameter(a, "scale", "LogitNormal"),
        )
        - math.log(point)
        - math.log1p(-point)
    )


def _truncatednormal_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``TruncatedNormal`` by inverting its distribution function.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    loc = real_parameter(a, "loc", "TruncatedNormal")
    scale = real_parameter(a, "scale", "TruncatedNormal")
    low = real_parameter(a, "low", "TruncatedNormal")
    high = real_parameter(a, "high", "TruncatedNormal")
    lower = _normal_cdf((low - loc) / scale)
    upper = _normal_cdf((high - loc) / scale)
    probability_ = lower + rng.random() * (upper - lower)
    probability_ = min(max(probability_, 1e-300), 1.0 - 1e-16)
    return loc + scale * _normal_icdf(probability_)


def _truncatednormal_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``TruncatedNormal`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside ``[low, high]``.
    """
    point = finite_value(value, "TruncatedNormal")
    loc = real_parameter(a, "loc", "TruncatedNormal")
    scale = real_parameter(a, "scale", "TruncatedNormal")
    low = real_parameter(a, "low", "TruncatedNormal")
    high = real_parameter(a, "high", "TruncatedNormal")
    if not low <= point <= high:
        return -math.inf
    mass = _normal_cdf((high - loc) / scale) - _normal_cdf((low - loc) / scale)
    return normal_log_prob(point, loc, scale) - math.log(mass)


def _gumbel_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Gumbel``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    uniform = rng.random()
    while uniform <= 0.0:
        uniform = rng.random()
    return real_parameter(a, "loc", "Gumbel") - real_parameter(
        a, "scale", "Gumbel"
    ) * math.log(-math.log(uniform))


def _gumbel_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Gumbel`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density.
    """
    scale = real_parameter(a, "scale", "Gumbel")
    z = (finite_value(value, "Gumbel") - real_parameter(a, "loc", "Gumbel")) / scale
    return -(z + math.exp(-z)) - math.log(scale)


def _chi2_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Chi2``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return rng.gammavariate(real_parameter(a, "df", "Chi2") / 2.0, 2.0)


def _chi2_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Chi2`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the positive reals.
    """
    point = finite_value(value, "Chi2")
    half = real_parameter(a, "df", "Chi2") / 2.0
    if point <= 0.0:
        return -math.inf
    return (
        (half - 1) * math.log(point)
        - point / 2
        - half * math.log(2)
        - math.lgamma(half)
    )


def _halfcauchy_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``HalfCauchy``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return abs(
        real_parameter(a, "scale", "HalfCauchy")
        * math.tan(math.pi * (rng.random() - 0.5))
    )


def _halfcauchy_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``HalfCauchy`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` below zero.
    """
    point = finite_value(value, "HalfCauchy")
    scale = real_parameter(a, "scale", "HalfCauchy")
    if point < 0.0:
        return -math.inf
    return math.log(2.0 / (math.pi * scale)) - math.log1p((point / scale) ** 2)


def _inversegamma_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``InverseGamma``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return 1.0 / rng.gammavariate(
        real_parameter(a, "concentration", "InverseGamma"),
        1.0 / real_parameter(a, "rate", "InverseGamma"),
    )


def _inversegamma_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``InverseGamma`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the positive reals.
    """
    point = finite_value(value, "InverseGamma")
    concentration = real_parameter(a, "concentration", "InverseGamma")
    rate = real_parameter(a, "rate", "InverseGamma")
    if point <= 0.0:
        return -math.inf
    return (
        concentration * math.log(rate)
        - math.lgamma(concentration)
        - (concentration + 1) * math.log(point)
        - rate / point
    )


def _weibull_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Weibull``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return rng.weibullvariate(
        real_parameter(a, "scale", "Weibull"),
        real_parameter(a, "concentration", "Weibull"),
    )


def _weibull_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Weibull`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the positive reals.
    """
    point = finite_value(value, "Weibull")
    scale = real_parameter(a, "scale", "Weibull")
    shape = real_parameter(a, "concentration", "Weibull")
    if point <= 0.0:
        return -math.inf
    return (
        math.log(shape)
        - math.log(scale)
        + (shape - 1) * (math.log(point) - math.log(scale))
        - (point / scale) ** shape
    )


def _pareto_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Pareto``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    return real_parameter(a, "scale", "Pareto") * rng.paretovariate(
        real_parameter(a, "alpha", "Pareto")
    )


def _pareto_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Pareto`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` below the scale.
    """
    point = finite_value(value, "Pareto")
    alpha = real_parameter(a, "alpha", "Pareto")
    scale = real_parameter(a, "scale", "Pareto")
    if point < scale:
        return -math.inf
    return math.log(alpha) + alpha * math.log(scale) - (alpha + 1) * math.log(point)


def _kumaraswamy_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Kumaraswamy`` by inverting its distribution function.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    first = real_parameter(a, "concentration1", "Kumaraswamy")
    second = real_parameter(a, "concentration0", "Kumaraswamy")
    return (1.0 - (1.0 - rng.random()) ** (1.0 / second)) ** (1.0 / first)


def _kumaraswamy_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Kumaraswamy`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the open unit interval.
    """
    point = finite_value(value, "Kumaraswamy")
    first = real_parameter(a, "concentration1", "Kumaraswamy")
    second = real_parameter(a, "concentration0", "Kumaraswamy")
    if not 0.0 < point < 1.0:
        return -math.inf
    return (
        math.log(first)
        + math.log(second)
        + (first - 1) * math.log(point)
        + (second - 1) * math.log1p(-(point**first))
    )


def _continuousbernoulli_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``ContinuousBernoulli`` by inverting its distribution function.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    lam = probability(a, "ContinuousBernoulli")
    uniform = rng.random()
    if abs(lam - 0.5) < 1e-3:
        return uniform
    return math.log1p((2 * lam - 1) * uniform / (1 - lam)) / (
        math.log(lam) - math.log1p(-lam)
    )


def _continuousbernoulli_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``ContinuousBernoulli`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the closed unit interval. The
        normalizer is ``2 atanh(1 - 2 lambda) / (1 - 2 lambda)`` away from
        one half, where its limit is two; near one half a Taylor
        expansion of the same function replaces the quotient.
    """
    point = finite_value(value, "ContinuousBernoulli")
    lam = probability(a, "ContinuousBernoulli")
    if not 0.0 <= point <= 1.0:
        return -math.inf
    shift = 1.0 - 2.0 * lam
    if abs(shift) < 1e-3:
        log_normalizer = math.log(2.0) + math.log1p(shift * shift / 3 + shift**4 / 5)
    else:
        log_normalizer = math.log(2.0 * math.atanh(shift) / shift)
    return point * math.log(lam) + (1.0 - point) * math.log1p(-lam) + log_normalizer


def _fishersnedecor_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``FisherSnedecor``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    df1 = real_parameter(a, "df1", "FisherSnedecor")
    df2 = real_parameter(a, "df2", "FisherSnedecor")
    first = rng.gammavariate(df1 / 2.0, 2.0) / df1
    second = rng.gammavariate(df2 / 2.0, 2.0) / df2
    return first / second


def _fishersnedecor_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``FisherSnedecor`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the positive reals.
    """
    point = finite_value(value, "FisherSnedecor")
    df1 = real_parameter(a, "df1", "FisherSnedecor")
    df2 = real_parameter(a, "df2", "FisherSnedecor")
    if point <= 0.0:
        return -math.inf
    return (
        0.5 * (df1 * math.log(df1) + df2 * math.log(df2))
        + (df1 / 2 - 1) * math.log(point)
        - (df1 + df2) / 2 * math.log(df2 + df1 * point)
        - _log_beta(df1 / 2, df2 / 2)
    )


def _relaxedbernoulli_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``RelaxedBernoulli``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value: a logistic draw shifted by the logit and divided
        by the temperature, pushed through the logistic function.
    """
    lam = probability(a, "RelaxedBernoulli")
    uniform = rng.random()
    while uniform <= 0.0 or uniform >= 1.0:
        uniform = rng.random()
    logit = math.log(lam) - math.log1p(-lam)
    noise = math.log(uniform) - math.log1p(-uniform)
    return _sigmoid((logit + noise) / relaxation_temperature(a, "RelaxedBernoulli"))


def _relaxedbernoulli_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``RelaxedBernoulli`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` outside the open unit interval: the
        logistic-relaxed density at the point's logit, less the log
        derivative of the logistic map.
    """
    point = finite_value(value, "RelaxedBernoulli")
    lam = probability(a, "RelaxedBernoulli")
    if not 0.0 < point < 1.0:
        return -math.inf
    temperature = relaxation_temperature(a, "RelaxedBernoulli")
    logit = math.log(lam) - math.log1p(-lam)
    y = math.log(point) - math.log1p(-point)
    difference = logit - temperature * y
    log_logistic = (
        math.log(temperature) + difference - 2.0 * math.log1p(math.exp(difference))
        if difference < 0
        else math.log(temperature)
        - difference
        - 2.0 * math.log1p(math.exp(-difference))
    )
    return log_logistic - math.log(point) - math.log1p(-point)


def _logistic_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Logistic``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    uniform = rng.random()
    while uniform <= 0.0 or uniform >= 1.0:
        uniform = rng.random()
    return real_parameter(a, "loc", "Logistic") + real_parameter(
        a, "scale", "Logistic"
    ) * (math.log(uniform) - math.log1p(-uniform))


def _logistic_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Logistic`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density.
    """
    scale = real_parameter(a, "scale", "Logistic")
    z = (finite_value(value, "Logistic") - real_parameter(a, "loc", "Logistic")) / scale
    return -math.log(scale) + _log_sigmoid(z) + _log_sigmoid(-z)


def _halfstudentt_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``HalfStudentT``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    df = real_parameter(a, "df", "HalfStudentT")
    scale = real_parameter(a, "scale", "HalfStudentT")
    chi = rng.gammavariate(df / 2.0, 2.0)
    return abs(scale * rng.gauss(0.0, 1.0) / math.sqrt(chi / df))


def _halfstudentt_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``HalfStudentT`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density, ``-inf`` below zero: twice the Student t
        density folded at zero.
    """
    point = finite_value(value, "HalfStudentT")
    df = real_parameter(a, "df", "HalfStudentT")
    scale = real_parameter(a, "scale", "HalfStudentT")
    if point < 0.0:
        return -math.inf
    standardized = point / scale
    return (
        math.log(2.0)
        + math.lgamma((df + 1) / 2)
        - math.lgamma(df / 2)
        - 0.5 * math.log(df * math.pi)
        - math.log(scale)
        - (df + 1) / 2 * math.log1p(standardized**2 / df)
    )


def _vonmises_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``VonMises``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn angle in ``[-pi, pi)``.
    """
    loc = real_parameter(a, "loc", "VonMises")
    concentration = real_parameter(a, "concentration", "VonMises")
    draw = rng.vonmisesvariate(loc, concentration)
    return (draw + math.pi) % (2 * math.pi) - math.pi


def _vonmises_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``VonMises`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density.
    """
    point = finite_value(value, "VonMises")
    loc = real_parameter(a, "loc", "VonMises")
    concentration = real_parameter(a, "concentration", "VonMises")
    return (
        concentration * math.cos(point - loc)
        - math.log(2 * math.pi)
        - _log_bessel_i0(concentration)
    )


def _horseshoe_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Horseshoe``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value: a normal draw whose scale is the family's global
        scale times a half-Cauchy local scale.
    """
    local = abs(math.tan(math.pi * (rng.random() - 0.5)))
    return rng.gauss(0.0, real_parameter(a, "scale", "Horseshoe") * local)


def _horseshoe_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Horseshoe`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The marginal log density over the half-Cauchy local scale, by
        the sixteen-point Gauss-Legendre rule on ``lambda = tan(pi t /
        2)``, which is the rule the torch runtime applies.
    """
    point = finite_value(value, "Horseshoe")
    tau = real_parameter(a, "scale", "Horseshoe")
    terms: list[float] = []
    for node, weight in _GAUSS_LEGENDRE_16:
        t = (1.0 + node) * 0.5
        half_pi_t = 0.5 * math.pi * t
        lam = math.tan(half_pi_t)
        jacobian = 0.5 * math.pi / math.cos(half_pi_t) ** 2
        sigma = tau * lam
        terms.append(
            normal_log_prob(point, 0.0, sigma)
            + math.log(2.0 / math.pi)
            - math.log1p(lam * lam)
            + math.log(jacobian)
            + math.log(weight * 0.5)
        )
    return _logsumexp(terms)


# ---------------------------------------------------------------------------
# scalar discrete families


def _negativebinomial_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``NegativeBinomial``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn count: a Poisson draw whose rate is a gamma draw, which
        is the gamma-Poisson mixture the family is.
    """
    total = real_parameter(a, "total_count", "NegativeBinomial")
    success = probability(a, "NegativeBinomial")
    rate = rng.gammavariate(total, success / (1.0 - success))
    limit = math.exp(-rate)
    count = 0
    product = rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def _negativebinomial_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``NegativeBinomial`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log probability, ``-inf`` below zero, with ``probs`` the
        success probability of each of the ``total_count`` failures the
        count is waited on, as the torch convention has it.
    """
    count = int(finite_value(value, "NegativeBinomial"))
    total = real_parameter(a, "total_count", "NegativeBinomial")
    success = probability(a, "NegativeBinomial")
    if count < 0:
        return -math.inf
    return (
        math.lgamma(total + count)
        - math.lgamma(count + 1)
        - math.lgamma(total)
        + total * math.log1p(-success)
        + count * math.log(success)
    )


def _betabinomial_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``BetaBinomial``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn count, from a binomial whose probability is a beta draw.
    """
    total = int(real_parameter(a, "total_count", "BetaBinomial"))
    success = rng.betavariate(
        real_parameter(a, "concentration1", "BetaBinomial"),
        real_parameter(a, "concentration0", "BetaBinomial"),
    )
    return sum(1 for _ in range(total) if rng.random() < success)


def _betabinomial_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``BetaBinomial`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log probability, ``-inf`` outside ``0..total_count``.
    """
    count = int(finite_value(value, "BetaBinomial"))
    total = real_parameter(a, "total_count", "BetaBinomial")
    first = real_parameter(a, "concentration1", "BetaBinomial")
    second = real_parameter(a, "concentration0", "BetaBinomial")
    if not 0 <= count <= total:
        return -math.inf
    return (
        log_choose(total, count)
        + _log_beta(first + count, second + total - count)
        - _log_beta(first, second)
    )


def _ordered_probabilities(
    a: Mapping[str, object], family: str, link: Callable[[float], float]
) -> tuple[float, ...]:
    """The category probabilities of an ordered regression family.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters: ``eta`` and the sorted ``cutpoints``.
    family : str
        The family, for the diagnostic.
    link : Callable[[float], float]
        The distribution function of the latent noise.

    Returns
    -------
    tuple[float, ...]
        One probability per category, ``len(cutpoints) + 1`` of them:
        ``F(c_0 - eta)``, the differences of consecutive ``F(c_k - eta)``,
        and ``1 - F(c_last - eta)``.

    Raises
    ------
    DistributionError
        If no cutpoint is given.
    """
    eta = real_parameter(a, "eta", family)
    cutpoints = vector_parameter(a, "cutpoints", family)
    if not cutpoints:
        raise DistributionError(f"{family} needs at least one cutpoint")
    cumulative = [link(cut - eta) for cut in cutpoints]
    probabilities = [cumulative[0]]
    probabilities.extend(
        cumulative[index] - cumulative[index - 1] for index in range(1, len(cumulative))
    )
    probabilities.append(1.0 - cumulative[-1])
    return tuple(probabilities)


def _ordered_sample(
    a: Mapping[str, object],
    rng: random.Random,
    family: str,
    link: Callable[[float], float],
) -> int:
    """Draw one category from an ordered regression family.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.
    family : str
        The family, for the diagnostic.
    link : Callable[[float], float]
        The distribution function of the latent noise.

    Returns
    -------
    int
        The drawn category.
    """
    probabilities = _ordered_probabilities(a, family, link)
    uniform = rng.random()
    total = 0.0
    for index, item in enumerate(probabilities):
        total += item
        if uniform < total:
            return index
    return len(probabilities) - 1


def _ordered_density(
    a: Mapping[str, object], value: object, family: str, link: Callable[[float], float]
) -> float:
    """The log probability of one category of an ordered regression family.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The category.
    family : str
        The family, for the diagnostic.
    link : Callable[[float], float]
        The distribution function of the latent noise.

    Returns
    -------
    float
        The log probability, ``-inf`` outside the categories.
    """
    category = int(finite_value(value, family))
    probabilities = _ordered_probabilities(a, family, link)
    if not 0 <= category < len(probabilities):
        return -math.inf
    mass = probabilities[category]
    return math.log(mass) if mass > 0.0 else -math.inf


def _orderedlogistic_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``OrderedLogistic``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn category.
    """
    return _ordered_sample(a, rng, "OrderedLogistic", _sigmoid)


def _orderedlogistic_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``OrderedLogistic`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log probability, ``-inf`` outside the categories.
    """
    return _ordered_density(a, value, "OrderedLogistic", _sigmoid)


def _orderedprobit_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``OrderedProbit``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn category.
    """
    return _ordered_sample(a, rng, "OrderedProbit", _normal_cdf)


def _orderedprobit_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``OrderedProbit`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log probability, ``-inf`` outside the categories.
    """
    return _ordered_density(a, value, "OrderedProbit", _normal_cdf)


# ---------------------------------------------------------------------------
# vector families


def _onehotcategorical_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``OneHotCategorical``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn indicator vector.
    """
    probabilities = simplex(a, "OneHotCategorical")
    uniform = rng.random()
    total = 0.0
    chosen = len(probabilities) - 1
    for index, item in enumerate(probabilities):
        total += item
        if uniform < total:
            chosen = index
            break
    return tuple(1.0 if index == chosen else 0.0 for index in range(len(probabilities)))


def _onehotcategorical_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``OneHotCategorical`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log probability of the indicated category, ``-inf`` for a
        vector that is not an indicator.
    """
    probabilities = simplex(a, "OneHotCategorical")
    point = _as_vector(value, "OneHotCategorical point")
    if (
        len(point) != len(probabilities)
        or sum(point) != 1.0
        or any(item not in (0.0, 1.0) for item in point)
    ):
        return -math.inf
    mass = probabilities[point.index(1.0)]
    return math.log(mass) if mass > 0.0 else -math.inf


def _relaxedonehot_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``RelaxedOneHotCategorical``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn point of the simplex: Gumbel-perturbed logits divided
        by the temperature, through the softmax.
    """
    probabilities = simplex(a, "RelaxedOneHotCategorical")
    temperature = relaxation_temperature(a, "RelaxedOneHotCategorical")
    perturbed: list[float] = []
    for item in probabilities:
        uniform = rng.random()
        while uniform <= 0.0:
            uniform = rng.random()
        perturbed.append((math.log(item) - math.log(-math.log(uniform))) / temperature)
    peak = max(perturbed)
    weights = [math.exp(item - peak) for item in perturbed]
    total = sum(weights)
    return tuple(item / total for item in weights)


def _relaxedonehot_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``RelaxedOneHotCategorical`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The concrete-distribution log density of Maddison, Mnih, and Teh
        on the interior of the simplex, ``-inf`` elsewhere.
    """
    probabilities = simplex(a, "RelaxedOneHotCategorical")
    point = _as_vector(value, "RelaxedOneHotCategorical point")
    k = len(probabilities)
    if len(point) != k or any(item <= 0.0 for item in point):
        return -math.inf
    temperature = relaxation_temperature(a, "RelaxedOneHotCategorical")
    log_scale = math.lgamma(k) + (k - 1) * math.log(temperature)
    scores = [
        math.log(probability_) - (temperature + 1) * math.log(item)
        for probability_, item in zip(probabilities, point, strict=True)
    ]
    log_sum = _logsumexp(
        [
            math.log(probability_) - temperature * math.log(item)
            for probability_, item in zip(probabilities, point, strict=True)
        ]
    )
    return log_scale + math.fsum(scores) - k * log_sum


def _logisticnormal_parameters(
    a: Mapping[str, object], count: int
) -> tuple[Vector, Vector]:
    """The location and scale of ``LogisticNormal`` over a simplex of a size.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters, scalars or vectors of the
        transformed length.
    count : int
        The simplex's length, one more than the transformed length.

    Returns
    -------
    tuple[Vector, Vector]
        The location and scale, each of length ``count - 1``.

    Raises
    ------
    DistributionError
        If a vector parameter has the wrong length.
    """
    read: list[Vector] = []
    for name in ("loc", "scale"):
        raw = a.get(name)
        if isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
            vector = vector_parameter(a, name, "LogisticNormal")
            if len(vector) != count - 1:
                raise DistributionError(
                    f"LogisticNormal {name} has length {len(vector)}; the simplex "
                    f"of length {count} needs {count - 1}"
                )
            read.append(vector)
        else:
            read.append(
                tuple(
                    real_parameter(a, name, "LogisticNormal") for _ in range(count - 1)
                )
            )
    return read[0], read[1]


def _stick_breaking(reals: Vector) -> Vector:
    """Map reals to the simplex by stick breaking.

    Parameters
    ----------
    reals : Vector
        The unconstrained coordinates, one fewer than the simplex's.

    Returns
    -------
    Vector
        The simplex point: each coordinate takes the logistic fraction of
        the stick that remains, with an offset ``log(K - 1 - i)`` on the
        logit so that zero maps to the uniform point, as torch's stick
        breaking transform has it.
    """
    count = len(reals)
    remaining = 1.0
    point: list[float] = []
    for index, value in enumerate(reals):
        fraction = _sigmoid(value - math.log(count - index))
        point.append(fraction * remaining)
        remaining *= 1.0 - fraction
    point.append(remaining)
    return tuple(point)


def _stick_breaking_inverse(point: Vector) -> tuple[Vector, float]:
    """Map a simplex point back to the reals, with the transform's log Jacobian.

    Parameters
    ----------
    point : Vector
        The simplex point.

    Returns
    -------
    tuple[Vector, float]
        The unconstrained coordinates and the log absolute determinant of
        the stick-breaking map at them, as torch computes it.
    """
    count = len(point) - 1
    reals: list[float] = []
    log_jacobian = 0.0
    consumed = 0.0
    for index in range(count):
        consumed += point[index]
        remaining = max(1.0 - consumed, 5e-324)
        shifted = math.log(point[index]) - math.log(remaining)
        reals.append(shifted + math.log(count - index))
        log_jacobian += -shifted + _log_sigmoid(shifted) + math.log(point[index])
    return tuple(reals), log_jacobian


def _logisticnormal_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``LogisticNormal``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters; a vector ``loc`` or ``scale``
        fixes the simplex's length, and scalars give a two-point simplex.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn point of the simplex.
    """
    count = 2
    for name in ("loc", "scale"):
        raw = a.get(name)
        if isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
            count = len(raw) + 1
    loc, scale = _logisticnormal_parameters(a, count)
    return _stick_breaking(
        tuple(rng.gauss(m, s) for m, s in zip(loc, scale, strict=True))
    )


def _logisticnormal_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``LogisticNormal`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated, a point of the simplex.

    Returns
    -------
    float
        The normal log density of the stick-breaking coordinates less
        the transform's log Jacobian, ``-inf`` off the open simplex.
    """
    point = _as_vector(value, "LogisticNormal point")
    if len(point) < 2 or any(item <= 0.0 for item in point):
        return -math.inf
    loc, scale = _logisticnormal_parameters(a, len(point))
    reals, log_jacobian = _stick_breaking_inverse(point)
    total = math.fsum(
        normal_log_prob(real, m, s)
        for real, m, s in zip(reals, loc, scale, strict=True)
    )
    return total - log_jacobian


def _mixturenormal_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``MixtureNormal``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn value.
    """
    weights = vector_parameter(a, "weights", "MixtureNormal")
    loc = vector_parameter(a, "loc", "MixtureNormal")
    scale = vector_parameter(a, "scale", "MixtureNormal")
    uniform = rng.random() * sum(weights)
    total = 0.0
    chosen = len(weights) - 1
    for index, weight in enumerate(weights):
        total += weight
        if uniform < total:
            chosen = index
            break
    return rng.gauss(loc[chosen], scale[chosen])


def _mixturenormal_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``MixtureNormal`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log of the weighted sum of component densities.

    Raises
    ------
    DistributionError
        If the weights, locations, and scales differ in length.
    """
    point = finite_value(value, "MixtureNormal")
    weights = vector_parameter(a, "weights", "MixtureNormal")
    loc = vector_parameter(a, "loc", "MixtureNormal")
    scale = vector_parameter(a, "scale", "MixtureNormal")
    if not len(weights) == len(loc) == len(scale):
        raise DistributionError("MixtureNormal parameters have different lengths")
    total = sum(weights)
    return _logsumexp(
        [
            math.log(weight / total) + normal_log_prob(point, m, s)
            for weight, m, s in zip(weights, loc, scale, strict=True)
            if weight > 0.0
        ]
    )


def _mvn_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``MultivariateNormal``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn vector.
    """
    return _draw_mvn(
        vector_parameter(a, "loc", "MultivariateNormal"),
        _covariance_factor(a, "MultivariateNormal"),
        rng,
    )


def _mvn_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``MultivariateNormal`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density.
    """
    return _mvn_log_prob(
        _as_vector(value, "MultivariateNormal point"),
        vector_parameter(a, "loc", "MultivariateNormal"),
        _covariance_factor(a, "MultivariateNormal"),
    )


def _lowrank_factor(a: Mapping[str, object]) -> tuple[Vector, Matrix]:
    """The mean and covariance Cholesky factor of ``LowRankMVN``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters: ``loc``, the ``cov_factor`` with
        one row per coordinate, and the ``cov_diag`` added to the
        diagonal.

    Returns
    -------
    tuple[Vector, Matrix]
        The mean and the lower Cholesky factor of ``W W^T + diag(D)``.

    Raises
    ------
    DistributionError
        If the parameters differ in length or the covariance is not
        positive definite.
    """
    loc = vector_parameter(a, "loc", "LowRankMVN")
    factor = _matrix_parameter(a, "cov_factor", "LowRankMVN")
    diagonal = vector_parameter(a, "cov_diag", "LowRankMVN")
    if len(factor) != len(loc) or len(diagonal) != len(loc):
        raise DistributionError("LowRankMVN parameters have different lengths")
    covariance = _matmul(factor, _transpose(factor))
    full = tuple(
        tuple(
            covariance[i][j] + (diagonal[i] if i == j else 0.0) for j in range(len(loc))
        )
        for i in range(len(loc))
    )
    return loc, _cholesky(full, "covariance of LowRankMVN")


def _lowrankmvn_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``LowRankMVN``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn vector.
    """
    loc, factor = _lowrank_factor(a)
    return _draw_mvn(loc, factor, rng)


def _lowrankmvn_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``LowRankMVN`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log density.
    """
    loc, factor = _lowrank_factor(a)
    return _mvn_log_prob(_as_vector(value, "LowRankMVN point"), loc, factor)


def _gp_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``GP``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters: the ``mean`` at the input
        locations and the ``kernel`` matrix over them.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn function values.
    """
    return _draw_mvn(
        vector_parameter(a, "mean", "GP"),
        _cholesky(_matrix_parameter(a, "kernel", "GP"), "kernel of GP"),
        rng,
    )


def _gp_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``GP`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The function values evaluated.

    Returns
    -------
    float
        The multivariate normal log density under the kernel.
    """
    return _mvn_log_prob(
        _as_vector(value, "GP point"),
        vector_parameter(a, "mean", "GP"),
        _cholesky(_matrix_parameter(a, "kernel", "GP"), "kernel of GP"),
    )


# ---------------------------------------------------------------------------
# matrix families


def _wishart_sample_at(df: float, factor: Matrix, rng: random.Random) -> Matrix:
    """Draw a Wishart matrix by the Bartlett decomposition.

    Parameters
    ----------
    df : float
        The degrees of freedom.
    factor : Matrix
        The lower Cholesky factor of the scale matrix.
    rng : random.Random
        The generator.

    Returns
    -------
    Matrix
        The draw ``L A A^T L^T`` with ``A`` the Bartlett factor.
    """
    size = len(factor)
    bartlett = [[0.0] * size for _ in range(size)]
    for i in range(size):
        bartlett[i][i] = math.sqrt(rng.gammavariate((df - i) / 2.0, 2.0))
        for j in range(i):
            bartlett[i][j] = rng.gauss(0.0, 1.0)
    product = _matmul(factor, tuple(tuple(row) for row in bartlett))
    return _matmul(product, _transpose(product))


def _wishart_log_prob(df: float, factor: Matrix, point: Matrix) -> float:
    """The Wishart log density from the scale's Cholesky factor.

    Parameters
    ----------
    df : float
        The degrees of freedom.
    factor : Matrix
        The lower Cholesky factor of the scale matrix ``V``.
    point : Matrix
        The positive definite matrix scored.

    Returns
    -------
    float
        The log density, ``-inf`` for a matrix that is not positive
        definite.

    Raises
    ------
    DistributionError
        If the matrix scored is not of the scale's size.
    """
    size = len(factor)
    if len(point) != size or any(len(row) != size for row in point):
        raise DistributionError("a Wishart is scored at a matrix of the wrong size")
    try:
        point_factor = _cholesky(point, "Wishart point")
    except DistributionError:
        return -math.inf
    inverse_scale = _spd_inverse(factor)
    return (
        (df - size - 1) / 2 * _log_det_from_cholesky(point_factor)
        - 0.5 * _trace(_matmul(inverse_scale, point))
        - df * size / 2 * math.log(2)
        - df / 2 * _log_det_from_cholesky(factor)
        - _log_multivariate_gamma(df / 2, size)
    )


def _wishart_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Wishart``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn matrix.
    """
    return _wishart_sample_at(
        real_parameter(a, "df", "Wishart"),
        _cholesky(
            _matrix_parameter(a, "covariance_matrix", "Wishart"), "scale of Wishart"
        ),
        rng,
    )


def _wishart_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Wishart`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The matrix evaluated.

    Returns
    -------
    float
        The log density.
    """
    return _wishart_log_prob(
        real_parameter(a, "df", "Wishart"),
        _cholesky(
            _matrix_parameter(a, "covariance_matrix", "Wishart"), "scale of Wishart"
        ),
        _as_matrix(value, "Wishart point"),
    )


def _inversewishart_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``InverseWishart``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters: ``df`` and the ``scale_tril`` of
        the Wishart whose inverse the draw is.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn matrix, the inverse of a Wishart draw.
    """
    draw = _wishart_sample_at(
        real_parameter(a, "df", "InverseWishart"),
        _matrix_parameter(a, "scale_tril", "InverseWishart"),
        rng,
    )
    return _spd_inverse(_cholesky(draw, "InverseWishart draw"))


def _inversewishart_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``InverseWishart`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The matrix evaluated.

    Returns
    -------
    float
        The Wishart log density of the inverse matrix, less ``(d + 1)``
        times the log determinant for the inversion's Jacobian, which is
        the torch runtime's convention.
    """
    point = _as_matrix(value, "InverseWishart point")
    try:
        point_factor = _cholesky(point, "InverseWishart point")
    except DistributionError:
        return -math.inf
    inverse = _spd_inverse(point_factor)
    return _wishart_log_prob(
        real_parameter(a, "df", "InverseWishart"),
        _matrix_parameter(a, "scale_tril", "InverseWishart"),
        inverse,
    ) - (len(point) + 1) * _log_det_from_cholesky(point_factor)


def _matrixnormal_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``MatrixNormal``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn matrix ``M + A Z B^T`` for standard normal ``Z`` and
        ``A``, ``B`` the Cholesky factors of the row and column covariances.
    """
    loc = _matrix_parameter(a, "loc", "MatrixNormal")
    rows = _cholesky(
        _matrix_parameter(a, "row_covariance", "MatrixNormal"),
        "row covariance of MatrixNormal",
    )
    columns = _cholesky(
        _matrix_parameter(a, "col_covariance", "MatrixNormal"),
        "column covariance of MatrixNormal",
    )
    noise = tuple(tuple(rng.gauss(0.0, 1.0) for _ in loc[0]) for _ in loc)
    shaped = _matmul(_matmul(rows, noise), _transpose(columns))
    return tuple(
        tuple(m + z for m, z in zip(row, noise_row, strict=True))
        for row, noise_row in zip(loc, shaped, strict=True)
    )


def _matrixnormal_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``MatrixNormal`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The matrix evaluated.

    Returns
    -------
    float
        The log density ``-0.5 tr(V^-1 (X - M)^T U^-1 (X - M))`` with the
        determinant terms, for row covariance ``U`` and column
        covariance ``V``.

    Raises
    ------
    DistributionError
        If the matrix scored is not of the location's size.
    """
    point = _as_matrix(value, "MatrixNormal point")
    loc = _matrix_parameter(a, "loc", "MatrixNormal")
    n, p = len(loc), len(loc[0]) if loc else 0
    if len(point) != n or any(len(row) != p for row in point):
        raise DistributionError("MatrixNormal is scored at a matrix of the wrong size")
    row_factor = _cholesky(
        _matrix_parameter(a, "row_covariance", "MatrixNormal"),
        "row covariance of MatrixNormal",
    )
    column_factor = _cholesky(
        _matrix_parameter(a, "col_covariance", "MatrixNormal"),
        "column covariance of MatrixNormal",
    )
    residual = tuple(
        tuple(x - m for x, m in zip(row, loc_row, strict=True))
        for row, loc_row in zip(point, loc, strict=True)
    )
    # With U = A A^T and V = B B^T the quadratic form is the squared
    # Frobenius norm of Z = A^-1 R B^-T: whiten each column of R by A,
    # then each row of the result by B.
    by_rows = _transpose(
        tuple(_solve_lower(row_factor, column) for column in _transpose(residual))
    )
    whitened = tuple(_solve_lower(column_factor, row) for row in by_rows)
    quadratic = math.fsum(item * item for row in whitened for item in row)
    return (
        -0.5 * quadratic
        - n * p / 2 * math.log(2 * math.pi)
        - p / 2 * _log_det_from_cholesky(row_factor)
        - n / 2 * _log_det_from_cholesky(column_factor)
    )


def _lkj_unnormalized(point: Matrix, concentration: float) -> float:
    """The LKJ log density of a correlation Cholesky factor, up to its constant.

    Parameters
    ----------
    point : Matrix
        The lower Cholesky factor of a correlation matrix.
    concentration : float
        The shape ``eta``.

    Returns
    -------
    float
        ``sum_k (d - 1 - k + 2 (eta - 1)) log L_kk`` over the rows after the
        first, ``-inf`` for a factor with a nonpositive diagonal.
    """
    size = len(point)
    total = 0.0
    for k in range(1, size):
        diagonal = point[k][k]
        if diagonal <= 0.0:
            return -math.inf
        total += (size - 1 - k + 2.0 * (concentration - 1.0)) * math.log(diagonal)
    return total


def _lkj_log_normalizer(size: int, concentration: float) -> float:
    """The log normalizer of the LKJ density over Cholesky factors.

    Parameters
    ----------
    size : int
        The matrix dimension.
    concentration : float
        The shape ``eta``.

    Returns
    -------
    float
        The constant torch's ``LKJCholesky`` subtracts, from the
        multivariate gamma function of Lewandowski, Kurowicka, and Joe:
        ``(d - 1) / 2 log pi + log Gamma_{d-1}(alpha - 1/2) - (d - 1)
        lgamma(alpha)`` for ``alpha = eta + (d - 1) / 2``.
    """
    lesser = size - 1
    alpha = concentration + 0.5 * lesser
    return (
        0.5 * lesser * math.log(math.pi)
        + _log_multivariate_gamma(alpha - 0.5, lesser)
        - lesser * math.lgamma(alpha)
    )


def _lkj_sample(size: int, concentration: float, rng: random.Random) -> Matrix:
    """Draw a correlation Cholesky factor by the onion method.

    Parameters
    ----------
    size : int
        The matrix dimension.
    concentration : float
        The shape ``eta``.
    rng : random.Random
        The generator.

    Returns
    -------
    Matrix
        The lower Cholesky factor of an LKJ correlation matrix.
    """
    factor = [[0.0] * size for _ in range(size)]
    factor[0][0] = 1.0
    for k in range(1, size):
        beta = concentration + (size - 1 - k) / 2.0
        radius_squared = rng.betavariate(k / 2.0, beta)
        direction = [rng.gauss(0.0, 1.0) for _ in range(k)]
        norm = math.sqrt(math.fsum(item * item for item in direction))
        for j in range(k):
            factor[k][j] = math.sqrt(radius_squared) * direction[j] / norm
        factor[k][k] = math.sqrt(1.0 - radius_squared)
    return tuple(tuple(row) for row in factor)


def _lkj_dimension(value: object, family: str) -> tuple[Matrix, int]:
    """Read a square matrix point and its dimension.

    Parameters
    ----------
    value : object
        The matrix.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    tuple[Matrix, int]
        The matrix and its size.

    Raises
    ------
    DistributionError
        If the matrix is not square.
    """
    point = _as_matrix(value, f"{family} point")
    if any(len(row) != len(point) for row in point):
        raise DistributionError(f"{family} is scored at a non-square matrix")
    return point, len(point)


def _lkjcholesky_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``LKJCholesky``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters, with the matrix ``dimension`` the
        construction fixes beside the ``concentration``.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn Cholesky factor.
    """
    return _lkj_sample(
        int(real_parameter(a, "dimension", "LKJCholesky")),
        real_parameter(a, "concentration", "LKJCholesky"),
        rng,
    )


def _lkjcholesky_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``LKJCholesky`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The Cholesky factor evaluated.

    Returns
    -------
    float
        The normalized LKJ log density, as torch's ``LKJCholesky`` scores it.
    """
    point, size = _lkj_dimension(value, "LKJCholesky")
    concentration = real_parameter(a, "concentration", "LKJCholesky")
    unnormalized = _lkj_unnormalized(point, concentration)
    if unnormalized == -math.inf:
        return unnormalized
    return unnormalized - _lkj_log_normalizer(size, concentration)


def _lkjfactor_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``LKJCorrelationFactor``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters, with the matrix ``dimension`` the
        construction fixes beside the ``concentration``.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The drawn Cholesky factor.
    """
    return _lkj_sample(
        int(real_parameter(a, "dimension", "LKJCorrelationFactor")),
        real_parameter(a, "concentration", "LKJCorrelationFactor"),
        rng,
    )


def _lkjfactor_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``LKJCorrelationFactor`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The Cholesky factor evaluated.

    Returns
    -------
    float
        The LKJ log density up to its constant, which is what the torch
        runtime's ``LKJCorrelationFactor`` scores.
    """
    point, _ = _lkj_dimension(value, "LKJCorrelationFactor")
    return _lkj_unnormalized(
        point, real_parameter(a, "concentration", "LKJCorrelationFactor")
    )


# ---------------------------------------------------------------------------
# compositional families


def _mixture_weights(a: Mapping[str, object]) -> Vector:
    """The mixture weights, which must be nonnegative and not all zero.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.

    Returns
    -------
    Vector
        The raw weights; a mixture is a weighted sum of measures and its
        mass is the weights' sum of the components' masses.

    Raises
    ------
    DistributionError
        If a weight is negative or every weight is zero.
    """
    weights = vector_parameter(a, "weights", "Mixture")
    if any(weight < 0.0 for weight in weights) or not any(weights):
        raise DistributionError("Mixture weights must be nonnegative and not all zero")
    return weights


def _mixture_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Mixture``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters: the ``weights`` and the
        ``component`` distributions.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        A draw from a component chosen with probability proportional to
        its weight times its mass.

    Raises
    ------
    DistributionError
        If the weights and the components differ in number.
    """
    weights = _mixture_weights(a)
    components = _components(a, "Mixture")
    if len(components) != len(weights):
        raise DistributionError("Mixture weights and components differ in number")
    masses = [
        weight * math.exp(_log_mass_of(component))
        for weight, component in zip(weights, components, strict=True)
    ]
    uniform = rng.random() * sum(masses)
    total = 0.0
    chosen = len(weights) - 1
    for index, mass in enumerate(masses):
        total += mass
        if uniform < total:
            chosen = index
            break
    return components[chosen].sample(rng)  # type: ignore[attr-defined]


def _mixture_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Mixture`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The log of the weighted sum of the component densities, the
        density of the mixture as a measure; ``Normalize`` divides it by
        the mixture's mass.

    Raises
    ------
    DistributionError
        If the weights and the components differ in number.
    """
    weights = _mixture_weights(a)
    components = _components(a, "Mixture")
    if len(components) != len(weights):
        raise DistributionError("Mixture weights and components differ in number")
    return _logsumexp(
        [
            math.log(weight) + float(component.log_prob(value))  # type: ignore[attr-defined]
            for weight, component in zip(weights, components, strict=True)
            if weight > 0.0
        ]
    )


def _mixture_log_mass(a: Mapping[str, object]) -> float:
    """The log mass of ``Mixture``: the weights' sum of the components' masses.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.

    Returns
    -------
    float
        ``log(sum(w_k * m_k))``.

    Raises
    ------
    DistributionError
        If the weights and the components differ in number.
    """
    weights = _mixture_weights(a)
    components = _components(a, "Mixture")
    if len(components) != len(weights):
        raise DistributionError("Mixture weights and components differ in number")
    return _logsumexp(
        [
            math.log(weight) + _log_mass_of(component)
            for weight, component in zip(weights, components, strict=True)
            if weight > 0.0
        ]
    )


def _independent_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Independent``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters: the plated ``base`` and the count
        of its batch axes read as one event.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The base's draw over its whole plate.
    """
    base = _sampleable(a, "base", "Independent")
    _reinterpreted(a, base)
    return base.sample(rng)  # type: ignore[attr-defined]


def _reinterpreted(a: Mapping[str, object], base: object) -> int:
    """Check the reinterpreted batch count against the base's plate.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    base : object
        The base distribution.

    Returns
    -------
    int
        The count of batch axes joined into the event.

    Raises
    ------
    DistributionError
        If the count is not the base's batch rank: the reference machine
        scores a plated draw as one event or as independent batch
        positions, and a base with more batch axes than are reinterpreted
        would leave batch positions no single point can be scored at.
    """
    count = int(real_parameter(a, "reinterpreted_batch_ndims", "Independent"))
    batch = getattr(base, "batch", ())
    if count != len(batch):
        raise DistributionError(
            f"Independent reinterprets {count} batch axes of a base with "
            f"{len(batch)}; the reference machine joins a base's whole plate"
        )
    return count


def _independent_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Independent`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated, shaped like the base's plate.

    Returns
    -------
    float
        The base's log density totaled over its plate.
    """
    base = _sampleable(a, "base", "Independent")
    _reinterpreted(a, base)
    return float(base.log_prob(value))  # type: ignore[attr-defined]


type _Bijector = tuple[
    Callable[[float], float], Callable[[float], float], Callable[[float], float]
]
"""A named transform: its forward map, its inverse, and the log of the
absolute derivative of the forward map at a point of its domain."""


def _softplus(value: float) -> float:
    """The softplus function.

    Parameters
    ----------
    value : float
        The argument.

    Returns
    -------
    float
        ``log(1 + exp(value))``, computed without overflow.
    """
    return (
        value + math.log1p(math.exp(-value))
        if value > 0
        else math.log1p(math.exp(value))
    )


def _softplus_inverse(value: float) -> float:
    """The inverse of the softplus function.

    Parameters
    ----------
    value : float
        A positive argument.

    Returns
    -------
    float
        ``log(exp(value) - 1)``.
    """
    return value + math.log(-math.expm1(-value))


#: The transforms a ``Transformed`` chain may name, applied left to right
#: in the order written.
TRANSFORMS: Mapping[str, _Bijector] = {
    "exp": (math.exp, math.log, lambda x: x),
    "log": (math.log, math.exp, lambda x: -math.log(x)),
    "sigmoid": (
        _sigmoid,
        lambda y: math.log(y) - math.log1p(-y),
        lambda x: _log_sigmoid(x) + _log_sigmoid(-x),
    ),
    "logit": (
        lambda x: math.log(x) - math.log1p(-x),
        _sigmoid,
        lambda x: -math.log(x) - math.log1p(-x),
    ),
    "softplus": (
        _softplus,
        _softplus_inverse,
        lambda x: _log_sigmoid(x),
    ),
    "tanh": (
        math.tanh,
        math.atanh,
        lambda x: 2.0 * (math.log(2.0) - x - _softplus(-2.0 * x)),
    ),
    "neg": (lambda x: -x, lambda y: -y, lambda x: 0.0),
}


def _transform_chain(a: Mapping[str, object]) -> tuple[_Bijector, ...]:
    """Read the transform chain of ``Transformed``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters, with ``transforms`` naming the
        chain as comma-separated names.

    Returns
    -------
    tuple[_Bijector, ...]
        The transforms in application order.

    Raises
    ------
    DistributionError
        If the chain is not a string or names a transform not in
        :data:`TRANSFORMS`.
    """
    raw = a.get("transforms")
    if not isinstance(raw, str):
        raise DistributionError("Transformed needs a transform chain named as a string")
    chain: list[_Bijector] = []
    for name in (item.strip() for item in raw.split(",") if item.strip()):
        bijector = TRANSFORMS.get(name)
        if bijector is None:
            raise DistributionError(
                f"Transformed names an unknown transform {name!r}; the reference "
                f"backend knows {', '.join(sorted(TRANSFORMS))}"
            )
        chain.append(bijector)
    return tuple(chain)


def _transformed_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Transformed``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters: the scalar ``base`` and its
        ``transforms``.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        The base's draw pushed through the chain.
    """
    base = _sampleable(a, "base", "Transformed")
    point = finite_value(base.sample(rng), "Transformed base")  # type: ignore[attr-defined]
    for forward, _, _ in _transform_chain(a):
        point = forward(point)
    return point


def _transformed_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Transformed`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The base's log density at the chain's inverse image, less the
        log derivative of the chain there; ``-inf`` where the inverse
        leaves every transform's range.
    """
    base = _sampleable(a, "base", "Transformed")
    chain = _transform_chain(a)
    point = finite_value(value, "Transformed")
    log_derivative = 0.0
    try:
        for _, inverse, _ in reversed(chain):
            point = inverse(point)
        current = point
        for forward, _, log_abs_derivative in chain:
            log_derivative += log_abs_derivative(current)
            current = forward(current)
    except ValueError, OverflowError, ZeroDivisionError:
        return -math.inf
    return float(base.log_prob(point)) - log_derivative  # type: ignore[attr-defined]


def _truncated_mass(base: object, low: float, high: float) -> float:
    """The probability a scalar base places on an interval.

    Parameters
    ----------
    base : object
        The base distribution.
    low : float
        The lower end.
    high : float
        The upper end.

    Returns
    -------
    float
        The integral of the base density over the interval.
    """

    def density(point: float) -> float:
        """The base density at a point, zero where it underflows.

        Parameters
        ----------
        point : float
            The point.

        Returns
        -------
        float
            The density.
        """
        log_density = float(base.log_prob(point))  # type: ignore[attr-defined]
        return math.exp(log_density) if log_density > -700.0 else 0.0

    return _integrate(density, low, high)


def _truncated_bounds(a: Mapping[str, object]) -> tuple[float, float]:
    """Read the bounds of ``Truncated``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.

    Returns
    -------
    tuple[float, float]
        The lower and upper bounds; a non-finite bound leaves that side
        open.

    Raises
    ------
    DistributionError
        If the bounds are not ordered.
    """
    low = real_parameter(a, "low", "Truncated")
    high = real_parameter(a, "high", "Truncated")
    if low >= high:
        raise DistributionError("Truncated needs low < high")
    return low, high


def _truncated_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Truncated`` by rejection from the base.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        A base draw inside the interval.

    Raises
    ------
    DistributionError
        If the base places so little mass on the interval that no draw
        lands in it within the attempt budget.
    """
    base = _sampleable(a, "base", "Truncated")
    low, high = _truncated_bounds(a)
    for _ in range(100_000):
        draw = finite_value(base.sample(rng), "Truncated base")  # type: ignore[attr-defined]
        if low <= draw <= high:
            return draw
    raise DistributionError("Truncated found no base draw inside its interval")


def _truncated_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Truncated`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The base's log density less the log mass of the interval,
        ``-inf`` outside it.
    """
    base = _sampleable(a, "base", "Truncated")
    low, high = _truncated_bounds(a)
    point = finite_value(value, "Truncated")
    if not low <= point <= high:
        return -math.inf
    mass = _truncated_mass(base, low, high)
    if mass <= 0.0:
        return -math.inf
    return float(base.log_prob(point)) - math.log(mass)  # type: ignore[attr-defined]


def _zip_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``ZeroInflatedPoisson``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        Zero with the inflation probability, else a Poisson draw.
    """
    if rng.random() < real_parameter(a, "zero_prob", "ZeroInflatedPoisson"):
        return 0
    rate = real_parameter(a, "rate", "ZeroInflatedPoisson")
    limit = math.exp(-rate)
    count = 0
    product = rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def _poisson_log_mass(count: int, rate: float) -> float:
    """The Poisson log mass at a count.

    Parameters
    ----------
    count : int
        The count.
    rate : float
        The rate.

    Returns
    -------
    float
        ``count log rate - rate - lgamma(count + 1)``.
    """
    return count * math.log(rate) - rate - math.lgamma(count + 1)


def _zip_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``ZeroInflatedPoisson`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        ``log(pi + (1 - pi) e^-rate)`` at zero, ``log(1 - pi)`` plus the
        Poisson log mass elsewhere, ``-inf`` below zero.
    """
    count = int(finite_value(value, "ZeroInflatedPoisson"))
    zero_prob = real_parameter(a, "zero_prob", "ZeroInflatedPoisson")
    rate = real_parameter(a, "rate", "ZeroInflatedPoisson")
    if count < 0:
        return -math.inf
    if count == 0:
        return math.log(zero_prob + (1.0 - zero_prob) * math.exp(-rate))
    return math.log1p(-zero_prob) + _poisson_log_mass(count, rate)


def _hurdle_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``HurdlePoisson``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        Zero with the hurdle probability, else a zero-truncated Poisson
        draw by rejection.
    """
    if rng.random() < real_parameter(a, "zero_prob", "HurdlePoisson"):
        return 0
    rate = real_parameter(a, "rate", "HurdlePoisson")
    limit = math.exp(-rate)
    while True:
        count = 0
        product = rng.random()
        while product > limit:
            count += 1
            product *= rng.random()
        if count > 0:
            return count


def _hurdle_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``HurdlePoisson`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        ``log pi`` at zero, ``log(1 - pi)`` plus the zero-truncated
        Poisson log mass elsewhere, ``-inf`` below zero.
    """
    count = int(finite_value(value, "HurdlePoisson"))
    zero_prob = real_parameter(a, "zero_prob", "HurdlePoisson")
    rate = real_parameter(a, "rate", "HurdlePoisson")
    if count < 0:
        return -math.inf
    if count == 0:
        return math.log(zero_prob)
    return (
        math.log1p(-zero_prob)
        + _poisson_log_mass(count, rate)
        - math.log(-math.expm1(-rate))
    )


def _zoib_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``ZeroOneInflatedBeta``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        An endpoint with the inflation probabilities, else a beta draw in
        the mean-precision parameterization.
    """
    zoi = real_parameter(a, "zoi", "ZeroOneInflatedBeta")
    coi = real_parameter(a, "coi", "ZeroOneInflatedBeta")
    if rng.random() < zoi:
        return 1.0 if rng.random() < coi else 0.0
    mu = real_parameter(a, "mu", "ZeroOneInflatedBeta")
    phi = real_parameter(a, "phi", "ZeroOneInflatedBeta")
    return rng.betavariate(mu * phi, (1.0 - mu) * phi)


def _zoib_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``ZeroOneInflatedBeta`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        ``log(zoi (1 - coi))`` at zero, ``log(zoi coi)`` at one, and
        ``log(1 - zoi)`` plus the beta log density between, ``-inf``
        outside the closed unit interval.
    """
    point = finite_value(value, "ZeroOneInflatedBeta")
    zoi = real_parameter(a, "zoi", "ZeroOneInflatedBeta")
    coi = real_parameter(a, "coi", "ZeroOneInflatedBeta")
    if point < 0.0 or point > 1.0:
        return -math.inf
    if point == 0.0:
        return math.log(zoi) + math.log1p(-coi)
    if point == 1.0:
        return math.log(zoi) + math.log(coi)
    mu = real_parameter(a, "mu", "ZeroOneInflatedBeta")
    phi = real_parameter(a, "phi", "ZeroOneInflatedBeta")
    first = mu * phi
    second = (1.0 - mu) * phi
    return (
        math.log1p(-zoi)
        + (first - 1.0) * math.log(point)
        + (second - 1.0) * math.log1p(-point)
        - _log_beta(first, second)
    )


def _optional_bound(a: Mapping[str, object], name: str, family: str) -> float | None:
    """Read an optional interval bound.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    name : str
        ``low`` or ``high``.
    family : str
        The family, for the diagnostic.

    Returns
    -------
    float | None
        The bound, or ``None`` when the parameter is absent or not
        finite, which leaves that side of the interval open.
    """
    if name not in a:
        return None
    bound = real_parameter(a, name, family)
    return None if math.isinf(bound) else bound


def _within(point: float, low: float | None, high: float | None) -> bool:
    """Whether a point lies in a closed interval with optional ends.

    Parameters
    ----------
    point : float
        The point.
    low : float | None
        The lower end, or ``None`` for none.
    high : float | None
        The upper end, or ``None`` for none.

    Returns
    -------
    bool
        True when the point is inside.
    """
    return (low is None or point >= low) and (high is None or point <= high)


def _base_is_discrete(base: object) -> bool:
    """Whether a nested distribution is discrete.

    Parameters
    ----------
    base : object
        The distribution, which names its family.

    Returns
    -------
    bool
        The registry's discreteness of the family, false for a
        distribution naming none.
    """
    name = getattr(base, "family", None)
    if not isinstance(name, str) or name not in FAMILIES:
        return False
    return FAMILIES[name].discrete


def _restrict_log_mass_of(base: object, low: float | None, high: float | None) -> float:
    """The log mass a base places on an interval.

    Parameters
    ----------
    base : object
        The base distribution.
    low : float | None
        The lower end, or ``None`` for none.
    high : float | None
        The upper end, or ``None`` for none.

    Returns
    -------
    float
        For a continuous base the log of the integral of its density
        over the interval; for a discrete base the log of the sum of its
        mass over the integers in the interval, an open upper end summed
        until the tail is negligible.
    """
    if _base_is_discrete(base):
        start = 0 if low is None else int(math.ceil(low))
        stop = high if high is not None else None
        total = 0.0
        count = start
        tail = 0
        while stop is None or count <= stop:
            mass = math.exp(float(base.log_prob(count)))  # type: ignore[attr-defined]
            total += mass
            tail = tail + 1 if mass < 1e-16 * max(total, 1e-300) else 0
            if stop is None and tail > 50 and count > 50:
                break
            count += 1
        return math.log(total) if total > 0.0 else -math.inf
    mass = _truncated_mass(
        base, -math.inf if low is None else low, math.inf if high is None else high
    )
    return math.log(mass) if mass > 0.0 else -math.inf


def _restrict_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Restrict`` by rejection from the base.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        A base draw inside the interval.

    Raises
    ------
    DistributionError
        If the base places so little mass on the interval that no draw
        lands in it within the attempt budget.
    """
    base = _sampleable(a, "base", "Restrict")
    low = _optional_bound(a, "low", "Restrict")
    high = _optional_bound(a, "high", "Restrict")
    for _ in range(100_000):
        draw = finite_value(base.sample(rng), "Restrict base")  # type: ignore[attr-defined]
        if _within(draw, low, high):
            return draw
    raise DistributionError("Restrict found no base draw inside its interval")


def _restrict_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Restrict`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The base's log density inside the interval, ``-inf`` outside;
        the restriction is a sub-probability measure, and ``Normalize``
        divides by its mass.
    """
    base = _sampleable(a, "base", "Restrict")
    low = _optional_bound(a, "low", "Restrict")
    high = _optional_bound(a, "high", "Restrict")
    point = finite_value(value, "Restrict")
    if not _within(point, low, high):
        return -math.inf
    return float(base.log_prob(value))  # type: ignore[attr-defined]


def _restrict_log_mass(a: Mapping[str, object]) -> float:
    """The log mass of ``Restrict``: the base's mass on the interval.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.

    Returns
    -------
    float
        The log mass.
    """
    base = _sampleable(a, "base", "Restrict")
    return _log_mass_of(base) + _restrict_log_mass_of(
        base,
        _optional_bound(a, "low", "Restrict"),
        _optional_bound(a, "high", "Restrict"),
    )


def _normalize_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``Normalize``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator to draw from.

    Returns
    -------
    object
        A draw of the base, whose sampler already draws from the
        normalized measure.
    """
    return _sampleable(a, "base", "Normalize").sample(rng)  # type: ignore[attr-defined]


def _normalize_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``Normalize`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        The base's log density less its log mass.
    """
    base = _sampleable(a, "base", "Normalize")
    log_density = float(base.log_prob(value))  # type: ignore[attr-defined]
    if log_density == -math.inf:
        return log_density
    return log_density - _log_mass_of(base)


def _pointmass_sample(a: Mapping[str, object], rng: random.Random) -> object:
    """Draw one value from ``PointMass``.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    rng : random.Random
        The generator, unused.

    Returns
    -------
    object
        The point.
    """
    del rng
    return real_parameter(a, "value", "PointMass")


def _pointmass_density(a: Mapping[str, object], value: object) -> float:
    """The log density of ``PointMass`` at a point.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.
    value : object
        The point evaluated.

    Returns
    -------
    float
        Zero at the point, ``-inf`` elsewhere: the Dirac measure's
        density against counting measure at its point.
    """
    point = finite_value(value, "PointMass")
    atom = real_parameter(a, "value", "PointMass")
    return 0.0 if math.isclose(point, atom, rel_tol=1e-12, abs_tol=1e-12) else -math.inf


def _transformed_log_mass(a: Mapping[str, object]) -> float:
    """The log mass of ``Transformed``, which is its base's.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.

    Returns
    -------
    float
        The base's log mass; a bijection moves mass without changing it.
    """
    return _log_mass_of(_sampleable(a, "base", "Transformed"))


def _independent_log_mass(a: Mapping[str, object]) -> float:
    """The log mass of ``Independent``, which is its base's over the plate.

    Parameters
    ----------
    a : Mapping[str, object]
        The family's named parameters.

    Returns
    -------
    float
        The base's log mass, which a plated base totals over its plate.
    """
    return _log_mass_of(_sampleable(a, "base", "Independent"))


#: Log masses of the families that are not probability measures, or are
#: built from ones that may not be, by family name; a family absent here
#: has mass one.
LOG_MASSES: Mapping[str, Callable[[Mapping[str, object]], float]] = {
    "Restrict": _restrict_log_mass,
    "Mixture": _mixture_log_mass,
    "Transformed": _transformed_log_mass,
    "Independent": _independent_log_mass,
}


#: Samplers of the families beyond the scalar core, by family name.
SAMPLERS: Mapping[str, Sampler] = {
    "LogitNormal": _logitnormal_sample,
    "TruncatedNormal": _truncatednormal_sample,
    "Gumbel": _gumbel_sample,
    "Chi2": _chi2_sample,
    "HalfCauchy": _halfcauchy_sample,
    "InverseGamma": _inversegamma_sample,
    "Weibull": _weibull_sample,
    "Pareto": _pareto_sample,
    "Kumaraswamy": _kumaraswamy_sample,
    "ContinuousBernoulli": _continuousbernoulli_sample,
    "FisherSnedecor": _fishersnedecor_sample,
    "RelaxedBernoulli": _relaxedbernoulli_sample,
    "Logistic": _logistic_sample,
    "HalfStudentT": _halfstudentt_sample,
    "VonMises": _vonmises_sample,
    "Horseshoe": _horseshoe_sample,
    "NegativeBinomial": _negativebinomial_sample,
    "BetaBinomial": _betabinomial_sample,
    "OrderedLogistic": _orderedlogistic_sample,
    "OrderedProbit": _orderedprobit_sample,
    "OneHotCategorical": _onehotcategorical_sample,
    "RelaxedOneHotCategorical": _relaxedonehot_sample,
    "LogisticNormal": _logisticnormal_sample,
    "MixtureNormal": _mixturenormal_sample,
    "MultivariateNormal": _mvn_sample,
    "LowRankMVN": _lowrankmvn_sample,
    "GP": _gp_sample,
    "Wishart": _wishart_sample,
    "InverseWishart": _inversewishart_sample,
    "MatrixNormal": _matrixnormal_sample,
    "LKJCholesky": _lkjcholesky_sample,
    "LKJCorrelationFactor": _lkjfactor_sample,
    "Mixture": _mixture_sample,
    "Independent": _independent_sample,
    "Transformed": _transformed_sample,
    "Truncated": _truncated_sample,
    "Restrict": _restrict_sample,
    "Normalize": _normalize_sample,
    "PointMass": _pointmass_sample,
    "ZeroInflatedPoisson": _zip_sample,
    "HurdlePoisson": _hurdle_sample,
    "ZeroOneInflatedBeta": _zoib_sample,
}

#: Log densities of the families beyond the scalar core, by family name.
DENSITIES: Mapping[str, Density] = {
    "LogitNormal": _logitnormal_density,
    "TruncatedNormal": _truncatednormal_density,
    "Gumbel": _gumbel_density,
    "Chi2": _chi2_density,
    "HalfCauchy": _halfcauchy_density,
    "InverseGamma": _inversegamma_density,
    "Weibull": _weibull_density,
    "Pareto": _pareto_density,
    "Kumaraswamy": _kumaraswamy_density,
    "ContinuousBernoulli": _continuousbernoulli_density,
    "FisherSnedecor": _fishersnedecor_density,
    "RelaxedBernoulli": _relaxedbernoulli_density,
    "Logistic": _logistic_density,
    "HalfStudentT": _halfstudentt_density,
    "VonMises": _vonmises_density,
    "Horseshoe": _horseshoe_density,
    "NegativeBinomial": _negativebinomial_density,
    "BetaBinomial": _betabinomial_density,
    "OrderedLogistic": _orderedlogistic_density,
    "OrderedProbit": _orderedprobit_density,
    "OneHotCategorical": _onehotcategorical_density,
    "RelaxedOneHotCategorical": _relaxedonehot_density,
    "LogisticNormal": _logisticnormal_density,
    "MixtureNormal": _mixturenormal_density,
    "MultivariateNormal": _mvn_density,
    "LowRankMVN": _lowrankmvn_density,
    "GP": _gp_density,
    "Wishart": _wishart_density,
    "InverseWishart": _inversewishart_density,
    "MatrixNormal": _matrixnormal_density,
    "LKJCholesky": _lkjcholesky_density,
    "LKJCorrelationFactor": _lkjfactor_density,
    "Mixture": _mixture_density,
    "Independent": _independent_density,
    "Transformed": _transformed_density,
    "Truncated": _truncated_density,
    "Restrict": _restrict_density,
    "Normalize": _normalize_density,
    "PointMass": _pointmass_density,
    "ZeroInflatedPoisson": _zip_density,
    "HurdlePoisson": _hurdle_density,
    "ZeroOneInflatedBeta": _zoib_density,
}


__all__: list[str] = []
