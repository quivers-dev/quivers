"""Stateful penalized spline evaluation through formulae's existing B-splines."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from formulae.transforms import BSpline

from quivers.formulas._latent_design import PenaltyBasis


class PSpline:
    """Retain a P-spline basis and its identified Gaussian penalty transform."""

    __stateful_transform__ = True

    def __init__(self) -> None:
        """Start with an unfitted B-spline basis and no retained penalty."""
        self.basis = BSpline()
        self.penalty: PenaltyBasis | None = None
        self.null_width = 0

    def __call__(
        self,
        x: npt.ArrayLike,
        *,
        bs: str | None = None,
        k: int = 10,
        m: int = 2,
        degree: int = 3,
    ) -> npt.NDArray[np.float64]:
        """Evaluate a retained B-spline penalty decomposition.

        Parameters
        ----------
        x : array-like
            Finite predictor rows.
        bs : str or None
            Explicitly ``ps``; other basis constructors require their own law.
        k, m, degree : int
            Basis width, coefficient-difference order, and polynomial degree.

        Returns
        -------
        numpy.ndarray
            Unpenalized columns followed by standardized Gaussian columns.

        Raises
        ------
        ValueError
            If the basis constructor, fixed predictor, or penalty is invalid.
        """
        if bs != "ps":
            raise ValueError(
                "this smooth checkpoint requires explicit bs='ps'; default thin-plate smooths are not B-splines"
            )
        if (
            type(k) is not int
            or type(m) is not int
            or type(degree) is not int
            or not 0 <= degree < k
            or not 0 < m < k
        ):
            raise ValueError("P-splines require integer k>degree>=0 and 0<m<k")
        values = np.asarray(x)
        if (
            values.ndim != 1
            or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()
        ):
            raise ValueError("smooth predictors must be finite real row vectors")
        if self.penalty is None:
            lower, upper = float(values.min()), float(values.max())
            if lower == upper:
                raise ValueError(
                    "smooth predictors require at least two distinct training values"
                )
            interior = np.linspace(lower, upper, k - degree + 1)[1:-1]
            raw = self.basis(
                values,
                df=k,
                knots=interior,
                degree=degree,
                intercept=True,
                lower_bound=lower,
                upper_bound=upper,
            )
            difference = torch.diff(torch.eye(k, dtype=torch.float64), n=m, dim=0)
            self.penalty = PenaltyBasis.fit(
                torch.as_tensor(raw), difference.T @ difference
            )
            self.null_width = len(self.penalty.null_factor[0])
        else:
            raw = self.basis.eval(values)
        fixed, random = self.penalty.evaluate(torch.as_tensor(raw))
        return np.column_stack(
            (
                np.asarray(fixed.values).reshape(len(values), fixed.width),
                np.asarray(random.values).reshape(len(values), random.width),
            )
        )


__all__ = []
