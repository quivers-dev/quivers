"""Standard-library M-spline adapter for a normalized Bernstein hazard."""

from __future__ import annotations

import math

from quivers._splines import MSplineBasis
from quivers.qiec._bernstein_hazard import BernsteinHazard


class CoxPH(BernsteinHazard):
    """Scalar proportional hazards retaining an immutable fitted spline basis.

    Parameters
    ----------
    basis : MSplineBasis
        Normalized M-splines and constant endpoint-hazard continuation.
    weights : Sequence[float]
        Strictly positive simplex weights over retained components.
    rate : float or None
        Positive hazard multiplier, exclusive with log_rate.
    log_rate : float or None
        Finite log hazard multiplier.
    """

    def __init__(self, basis: MSplineBasis, weights, rate=None, *, log_rate=None):
        if basis.boundary_knots[0] != 0:
            raise ValueError("Cox baseline must start at time zero")
        self.basis = basis
        self.weights = tuple(map(float, weights))
        if len(self.weights) != basis.size or any(
            not math.isfinite(w) or w <= 0 for w in self.weights
        ):
            raise ValueError("Cox weights must be a strictly positive spline simplex")
        if not math.isclose(sum(self.weights), 1.0, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Cox spline weights must sum to one")
        controls = tuple(
            tuple(
                math.fsum(w * piece[k] for w, piece in zip(self.weights, interval))
                for k in range(basis.degree + 1)
            )
            for interval in basis.m_coefficients
        )
        super().__init__(basis.breaks, controls, rate, log_rate=log_rate)
