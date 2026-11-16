"""Immutable normalized M/I-spline bases with an explicit hazard continuation.

This module has no numerical-library dependency. Bernstein coefficients retain
nonnegative M-spline pieces and their integrated I-spline pieces without the
cancellation of a power-basis expansion near a knot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real


def _bernstein_value(coefficients: tuple[float, ...], u: float) -> float:
    """Evaluate one Bernstein polynomial by de Casteljau's recurrence.

    Parameters
    ----------
    coefficients : tuple[float, ...]
        One polynomial piece in Bernstein form.
    u : float
        Evaluation coordinate on the unit interval.

    Returns
    -------
    float
        Polynomial value.
    """
    values = list(coefficients)
    for width in range(len(values) - 1, 0, -1):
        values = [(1 - u) * values[i] + u * values[i + 1] for i in range(width)]
    return values[0]


def _linear_product(
    coefficients: list[float], left: float, right: float
) -> list[float]:
    """Multiply a Bernstein polynomial by an affine factor.

    Parameters
    ----------
    coefficients : list[float]
        Bernstein coefficients of the polynomial.
    left, right : float
        Affine-factor values at zero and one.

    Returns
    -------
    list[float]
        Coefficients of the product, with its degree increased by one.
    """
    degree = len(coefficients)
    return [
        (i / degree * right * coefficients[i - 1] if i else 0.0)
        + ((1 - i / degree) * left * coefficients[i] if i < degree else 0.0)
        for i in range(degree + 1)
    ]


@dataclass(frozen=True)
class MSplineBasis:
    """Normalized M-splines, integrated I-splines, and a constant hazard tail.

    Parameters
    ----------
    boundary_knots : tuple[float, float]
        Finite increasing endpoints of the fitted spline interval.
    interior_knots : tuple[float, ...]
        Strictly increasing knots inside the boundary interval.
    degree : int
        Nonnegative polynomial degree of the M-splines.
    tail : str
        Hazard continuation; currently ``"constant"`` retains each M-spline's
        value at the upper boundary and integrates it linearly thereafter.

    Notes
    -----
    Every M-spline integrates to one over the fitted interval. Below its lower
    boundary both bases are zero. The retained knots and tail policy are part
    of value equality, so prediction cannot silently rebuild a training basis.
    """

    boundary_knots: tuple[float, float] = (0.0, 1.0)
    interior_knots: tuple[float, ...] = ()
    degree: int = 3
    tail: str = "constant"
    breaks: tuple[float, ...] = field(init=False, repr=False)
    m_coefficients: tuple[tuple[tuple[float, ...], ...], ...] = field(
        init=False, repr=False
    )
    i_coefficients: tuple[tuple[tuple[float, ...], ...], ...] = field(
        init=False, repr=False
    )

    def __post_init__(self):
        """Validate immutable knots and construct normalized basis tables.

        Raises
        ------
        ValueError
            If knots, degree, tail policy, or basis normalization are invalid.
        NotImplementedError
            If the retained table exceeds the explicit coefficient budget.
        """
        if (
            isinstance(self.degree, bool)
            or not isinstance(self.degree, int)
            or self.degree < 0
        ):
            raise ValueError("M-spline degree must be a nonnegative integer")
        if self.tail != "constant":
            raise ValueError("M-spline tail must be 'constant'")
        raw = (*self.boundary_knots, *self.interior_knots)
        if any(
            isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
            for v in raw
        ):
            raise ValueError("spline knots must be finite real numbers")
        if len(self.boundary_knots) != 2:
            raise ValueError("exactly two boundary knots are required")
        lower, upper = map(float, self.boundary_knots)
        interior = tuple(map(float, self.interior_knots))
        breaks = (lower, *interior, upper)
        if any(a >= b for a, b in zip(breaks, breaks[1:])):
            raise ValueError(
                "spline knots must increase strictly within their boundaries"
            )
        if not math.isfinite(upper - lower):
            raise ValueError("spline interval width must be finite")
        object.__setattr__(self, "boundary_knots", (lower, upper))
        object.__setattr__(self, "interior_knots", interior)
        object.__setattr__(self, "breaks", breaks)
        p = self.degree
        size = len(interior) + p + 1
        if (len(interior) + 1) * size * (2 * p + 3) > 1_048_576:
            raise NotImplementedError(
                "retained spline table exceeds 1,048,576 coefficients"
            )
        knots = (lower,) * (p + 1) + interior + (upper,) * (p + 1)
        integrated = [0.0] * size
        m_pieces, i_pieces = [], []
        for a, b in zip(breaks, breaks[1:]):
            pieces = [
                [float(t <= a and b <= s and t < s)] for t, s in zip(knots, knots[1:])
            ]
            for degree in range(1, p + 1):
                following = []
                for i in range(len(pieces) - 1):
                    left_width = knots[i + degree] - knots[i]
                    right_width = knots[i + degree + 1] - knots[i + 1]
                    left = (
                        _linear_product(
                            pieces[i],
                            (a - knots[i]) / left_width,
                            (b - knots[i]) / left_width,
                        )
                        if left_width
                        else [0.0] * (degree + 1)
                    )
                    right = (
                        _linear_product(
                            pieces[i + 1],
                            (knots[i + degree + 1] - a) / right_width,
                            (knots[i + degree + 1] - b) / right_width,
                        )
                        if right_width
                        else [0.0] * (degree + 1)
                    )
                    following.append([x + y for x, y in zip(left, right)])
                pieces = following
            m_piece, i_piece = [], []
            for i, polynomial in enumerate(pieces):
                scale = (p + 1) / (knots[i + p + 1] - knots[i])
                coefficients = tuple(scale * value for value in polynomial)
                integral = [integrated[i]]
                for value in coefficients:
                    integral.append(integral[-1] + (b - a) * value / (p + 1))
                integrated[i] = integral[-1]
                m_piece.append(coefficients)
                i_piece.append(tuple(integral))
            m_pieces.append(tuple(m_piece))
            i_pieces.append(tuple(i_piece))
        if any(
            not math.isclose(value, 1.0, rel_tol=2e-12, abs_tol=2e-12)
            for value in integrated
        ):
            raise ValueError("spline basis cannot be normalized at these knots")
        object.__setattr__(self, "m_coefficients", tuple(m_pieces))
        object.__setattr__(self, "i_coefficients", tuple(i_pieces))

    @property
    def size(self) -> int:
        """Return the number of retained spline components.

        Returns
        -------
        int
            Number of retained normalized spline components.
        """
        return len(self.interior_knots) + self.degree + 1

    def evaluate(self, value: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Evaluate the hazard basis and its integral, including the tail.

        Parameters
        ----------
        value : float
            Evaluation time, including support endpoints and infinities.

        Returns
        -------
        tuple[tuple[float, ...], tuple[float, ...]]
            M-spline values and integrated I-spline values in component order.

        Raises
        ------
        ValueError
            If the time is NaN.
        """
        if math.isnan(value):
            raise ValueError("spline argument must not be NaN")
        lower, upper = self.boundary_knots
        if value < lower:
            return (0.0,) * self.size, (0.0,) * self.size
        if value >= upper:
            m = tuple(piece[-1] for piece in self.m_coefficients[-1])
            i = tuple(piece[-1] for piece in self.i_coefficients[-1])
            if value > upper:
                i = tuple(y + x * (value - upper) if x else y for x, y in zip(m, i))
            return m, i
        import bisect

        index = bisect.bisect_right(self.breaks, value) - 1
        u = (value - self.breaks[index]) / (self.breaks[index + 1] - self.breaks[index])
        return (
            tuple(_bernstein_value(piece, u) for piece in self.m_coefficients[index]),
            tuple(_bernstein_value(piece, u) for piece in self.i_coefficients[index]),
        )
