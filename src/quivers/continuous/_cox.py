"""M/I-spline adapters for normalized Bernstein baseline hazards."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import constraints
from quivers._splines import MSplineBasis
from quivers.continuous._bernstein_hazard import BernsteinHazard, _bernstein


def spline_values(basis: MSplineBasis, value: Tensor) -> tuple[Tensor, Tensor]:
    """Lift the retained M/I-spline basis to differentiable tensor arguments.

    Parameters
    ----------
    basis : MSplineBasis
        Retained normalized M/I-spline structure.
    value : Tensor
        Evaluation times.

    Returns
    -------
    tuple[Tensor, Tensor]
        M-spline hazards and integrated I-spline values, including the tail.
    """
    breaks = value.new_tensor(basis.breaks)
    lower, upper = basis.boundary_knots
    finite = torch.where(torch.isfinite(value), value, value.new_tensor(upper))
    x = finite.clamp(lower, upper)
    index = torch.bucketize(x.contiguous(), breaks[1:-1], right=True)
    u = (x - breaks[index]) / (breaks[index + 1] - breaks[index])
    m = _bernstein(value.new_tensor(basis.m_coefficients)[index], u[..., None])
    integral = _bernstein(value.new_tensor(basis.i_coefficients)[index], u[..., None])
    tail_m = value.new_tensor([piece[-1] for piece in basis.m_coefficients[-1]])
    excess = torch.where(finite > upper, finite - upper, 0.0)
    integral = integral + excess[..., None] * tail_m
    m = torch.where((finite > upper)[..., None], tail_m, m)
    m = torch.where((value < lower)[..., None], 0.0, m)
    integral = torch.where((value < lower)[..., None], 0.0, integral)
    # The zero tail components remain finite rather than forming 0 * infinity.
    integral = torch.where(
        (value == torch.inf)[..., None] & (tail_m > 0), torch.inf, integral
    )
    return m, integral


class CoxPH(BernsteinHazard):
    """Proportional hazards with a retained normalized M-spline basis.

    Parameters
    ----------
    basis : MSplineBasis
        Immutable fitted knots, degree, and constant endpoint tail.
    weights : Tensor or Sequence[float]
        Strictly positive simplex weights over basis components.
    rate : Tensor or float, optional
        Positive proportional multiplier, exclusive with log_rate.
    log_rate : Tensor or float, optional
        Finite log multiplier.
    validate_args : bool or None
        Torch parameter-validation policy.
    """

    arg_constraints = {"weights": constraints.simplex, "log_rate": constraints.real}

    def __init__(self, basis, weights, rate=None, *, log_rate=None, validate_args=None):
        if basis.boundary_knots[0] != 0:
            raise ValueError("Cox baseline must start at time zero")
        self.basis = basis
        device = next(
            (v.device for v in (weights, rate, log_rate) if isinstance(v, Tensor)), None
        )
        weights = torch.as_tensor(weights, dtype=torch.float64, device=device)
        if weights.ndim < 1 or weights.shape[-1] != basis.size:
            raise ValueError(f"Cox weights need {basis.size} spline components")
        if validate_args is not False and not bool(
            torch.isfinite(weights).all()
            & (weights > 0).all()
            & torch.isclose(
                weights.sum(-1), weights.new_tensor(1.0), rtol=1e-6, atol=1e-8
            ).all()
        ):
            raise ValueError("Cox spline weights must be a strictly positive simplex")
        self.weights = weights
        controls = torch.einsum(
            "...j,ijk->...ik", weights, weights.new_tensor(basis.m_coefficients)
        )
        super().__init__(
            weights.new_tensor(basis.breaks),
            controls,
            rate,
            log_rate=log_rate,
            validate_args=validate_args,
        )
        self.weights = weights.expand(*self.batch_shape, basis.size)

    def expand(self, batch_shape, _instance=None):
        """Expand distribution views while retaining the original fitted basis.

        Parameters
        ----------
        batch_shape : torch.Size
            Requested leading batch dimensions.
        _instance : CoxPH or None
            Optional instance populated by the Torch expansion protocol.

        Returns
        -------
        CoxPH
            Expanded view retaining the fitted basis and weights.
        """
        new = super().expand(batch_shape, _instance)
        new.basis = self.basis
        new.weights = self.weights.expand(*batch_shape, self.basis.size)
        return new
