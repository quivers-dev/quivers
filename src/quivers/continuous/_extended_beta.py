"""Symmetrically extended Beta responses censored to the unit interval."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor
from torch.distributions import constraints

from quivers.continuous._affine_beta import _AffineBeta
from quivers.continuous._beta_tails import log_beta_tails
from quivers.continuous._bounded_moments import _positive_unit_moments
from quivers.continuous._censored import Censored
from quivers.continuous._zip_hurdle import _OpenUnitInterval


class ExtendedBeta(Censored):
    """Extend a mean/precision Beta law and censor its tails at zero and one.

    Parameters
    ----------
    mu : Tensor or float
        Underlying Beta mean, strictly between zero and one.
    phi : Tensor or float
        Strictly positive Beta precision.
    kappa : Tensor or float
        Nonnegative extension: ``Y=clamp((1+2*kappa)*X-kappa,0,1)``.
    moment_atol, moment_rtol : float
        Deterministic bounded-moment quadrature error budgets.
    moment_max_panels : int
        Per-row adaptive panel limit; work is chunked over at most64 rows.
    validate_args : bool or None
        Torch validation policy; parameter validity is always checked.

    Raises
    ------
    ValueError
        If parameters or numerical budgets are invalid.
    RuntimeError
        If the symmetric extension cannot retain its censoring bounds in Float64.
    """

    support = constraints.unit_interval
    has_rsample = True
    arg_constraints = {
        "mu": _OpenUnitInterval(),
        "phi": constraints.positive,
        "kappa": constraints.nonnegative,
    }

    def __init__(
        self,
        mu,
        phi,
        kappa,
        *,
        moment_atol=1e-12,
        moment_rtol=1e-10,
        moment_max_panels=4096,
        validate_args=None,
    ):
        device = next(
            (x.device for x in (mu, phi, kappa) if isinstance(x, Tensor)), None
        )
        self.mu, self.phi, self.kappa = torch.broadcast_tensors(
            *(
                torch.as_tensor(x, dtype=torch.float64, device=device)
                for x in (mu, phi, kappa)
            )
        )
        valid = (
            (self.mu > 0)
            & (self.mu < 1)
            & (self.phi > 0)
            & (self.kappa >= 0)
            & torch.isfinite(self.mu + self.phi + self.kappa)
        )
        if not bool(valid.all()):
            raise ValueError(
                "ExtendedBeta requires mu in(0,1), finite phi>0 and kappa>=0"
            )
        if (
            not isinstance(moment_max_panels, int)
            or isinstance(moment_max_panels, bool)
            or moment_max_panels < 1
            or not math.isfinite(moment_atol)
            or not math.isfinite(moment_rtol)
            or moment_atol <= 0
            or moment_rtol <= 0
        ):
            raise ValueError("ExtendedBeta moment budgets must be finite and positive")
        self.moment_atol, self.moment_rtol = moment_atol, moment_rtol
        self.moment_max_panels = moment_max_panels
        width = 1 + 2 * self.kappa
        if not bool(torch.isfinite(width).all()):
            raise RuntimeError("ExtendedBeta extension exceeds Float64 affine capacity")
        low = self.kappa / width
        high = 0.5 + 0.5 / width
        if not bool(((low < 0.5) & (high > 0.5)).all()):
            raise RuntimeError(
                "ExtendedBeta censoring bounds exceed Float64 interval resolution"
            )
        beta = _AffineBeta(
            self.mu * self.phi, (1 - self.mu) * self.phi, self.kappa, self.kappa
        )
        super().__init__(beta, 0.0, 1.0)
        self._validate_args = bool(validate_args)

    def expand(self, batch_shape, _instance=None):
        """Broadcast the same law and deterministic moment budgets.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Requested parameter batch shape.
        _instance : ExtendedBeta or None
            Optional Torch expansion instance.

        Returns
        -------
        ExtendedBeta
            Expanded distribution, retaining the same numerical policy.
        """
        return type(self)(
            self.mu.expand(batch_shape),
            self.phi.expand(batch_shape),
            self.kappa.expand(batch_shape),
            moment_atol=self.moment_atol,
            moment_rtol=self.moment_rtol,
            moment_max_panels=self.moment_max_panels,
            validate_args=self._validate_args,
        )

    def sample(
        self, sample_shape: torch.Size | tuple[int, ...] | list[int] = torch.Size()
    ):
        """Draw censoring atoms before rounding complementary Beta weights.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading independent sample axes.

        Returns
        -------
        Tensor
            Distribution result on the requested sample and batch axes.

        Raises
        ------
        RuntimeError
            If the evaluation or sample exceeds the documented numerical capacity.
        """
        with torch.no_grad():
            return cast(_AffineBeta, self.base)._sample_unit_censored(
                sample_shape, reparameterized=False
            )

    def rsample(
        self, sample_shape: torch.Size | tuple[int, ...] | list[int] = torch.Size()
    ):
        """Draw pathwise responses, refusing unrepresentable interior samples.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading independent sample axes.

        Returns
        -------
        Tensor
            Distribution result on the requested sample and batch axes.
        """
        return cast(_AffineBeta, self.base)._sample_unit_censored(
            sample_shape, reparameterized=True
        )

    def _moments(self) -> tuple[Tensor, Tensor]:
        """Compute positive bounded moments without extensive subtraction.

        Returns
        -------
        tuple[Tensor, Tensor]
            Response mean and variance on the parameter batch axes.

        Raises
        ------
        RuntimeError
            If quadrature error cannot be certified within its panel budget.
        """
        shape = self.batch_shape
        mu, phi, kappa = (p.reshape(-1) for p in (self.mu, self.phi, self.kappa))
        if not mu.numel():
            return mu.reshape(shape), mu.reshape(shape)
        means, variances = [], []
        for start in range(0, len(mu), 64):
            m, p, k = (
                mu[start : start + 64],
                phi[start : start + 64],
                kappa[start : start + 64],
            )
            small = k <= self.moment_atol / 8
            # Clipping moves a unit-interval value by at most kappa.
            # The linear Beta limit has mean error at most kappa and
            # variance error at most5*kappa on these bounded responses.
            ordinary_mean = (m + (2 * m - 1) * k + 0 * p).clamp(0, 1)
            ordinary_variance = m * (1 - m) / (p + 1) * (1 + 4 * k)
            tiny = small & (k > 0)
            if bool(tiny.any()):
                # Endpoint clipping changes values only by O(kappa), but
                # its kappa derivative need not be close to the zero-limit
                # derivative for small Beta shapes. Retain partial tails
                # in these O(kappa) corrections before adding the baseline.
                tm, tp, tk = m[tiny], p[tiny], k[tiny]
                a, b, scale = tm * tp, (1 - tm) * tp, 1 + 2 * tk
                cutoff = tk / scale
                lo = [log_beta_tails(a + j, b, cutoff)[0].exp() for j in range(3)]
                hi = [log_beta_tails(b + j, a, cutoff)[0].exp() for j in range(3)]
                first_x = tm
                second_x = tm * (a + 1) / (tp + 1)
                first_q = 1 - tm
                second_q = (1 - tm) * (b + 1) / (tp + 1)
                lower = tk * lo[0] - scale * first_x * lo[1]
                upper = tk * hi[0] - scale * first_q * hi[1]
                lower_square = (
                    tk.square() * lo[0]
                    - 2 * tk * scale * first_x * lo[1]
                    + scale.square() * second_x * lo[2]
                )
                upper_square = (
                    tk.square() * hi[0]
                    - 2 * tk * scale * first_q * hi[1]
                    + scale.square() * second_q * hi[2]
                )
                baseline = tm + (2 * tm - 1) * tk
                mean = baseline + lower - upper
                variance = (
                    scale.square() * tm * (1 - tm) / (tp + 1)
                    - lower_square
                    - 2 * upper
                    - upper_square
                    - (lower - upper) * (mean + baseline)
                )
                ordinary_mean = ordinary_mean.masked_scatter(tiny, mean)
                ordinary_variance = ordinary_variance.masked_scatter(tiny, variance)
            if bool(small.all()):
                means.append(ordinary_mean)
                variances.append(ordinary_variance)
                continue
            active = ~small
            aa, bb = (m * p)[active], ((1 - m) * p)[active]
            kk = k[active]

            def log_density(point: Tensor, rows: Tensor) -> Tensor:
                """Evaluate affine log densities on selected quadrature rows.

                Parameters
                ----------
                point : Tensor
                    Quadrature response coordinates on the final node axis.
                rows : Tensor
                    Flattened parameter-row indices.

                Returns
                -------
                Tensor
                    Distribution result on the requested sample and batch axes.
                """
                return _AffineBeta(
                    aa[rows, None], bb[rows, None], kk[rows, None], kk[rows, None]
                ).log_prob(point)

            def log_error(left: Tensor, right: Tensor, rows: Tensor) -> Tensor:
                """Bound the absolute Gauss8 error using density derivative envelopes.

                Parameters
                ----------
                left : Tensor
                    Lower quadrature-panel boundaries.
                right : Tensor
                    Upper quadrature-panel boundaries.
                rows : Tensor
                    Flattened parameter-row indices.

                Returns
                -------
                Tensor
                    Distribution result on the requested sample and batch axes.
                """
                a, b, extension = aa[rows], bb[rows], kk[rows]
                denominator = a + b - 2
                peak = 0.5 + (extension + 0.5) * (a - b) / torch.where(
                    denominator != 0, denominator, 1.0
                )
                peak = peak.clamp(left, right)
                candidates = torch.stack((left, right, peak), -1)
                maximum = log_density(candidates, rows).amax(-1)
                u, v = left + extension, 1 + extension - right
                endpoint = torch.stack((left, right), -1)
                first = (
                    (
                        (
                            (a - b)[:, None] * (extension + 0.5)[:, None]
                            + (a + b - 2)[:, None] * (0.5 - endpoint)
                        )
                        / (
                            (endpoint + extension[:, None])
                            * (1 + extension[:, None] - endpoint)
                        )
                    )
                    .abs()
                    .amax(-1)
                )
                log_derivatives = [torch.zeros_like(a), first.log()]
                log_a, log_b = (a - 1).abs().log(), (b - 1).abs().log()
                for order in range(2, 17):
                    log_derivatives.append(
                        math.lgamma(order)
                        + torch.logaddexp(
                            log_a - order * u.log(), log_b - order * v.log()
                        )
                    )
                bell = [torch.zeros_like(a)]
                for order in range(1, 17):
                    bell.append(
                        torch.logsumexp(
                            torch.stack(
                                [
                                    math.log(math.comb(order - 1, part - 1))
                                    + log_derivatives[part]
                                    + bell[order - part]
                                    for part in range(1, order + 1)
                                ]
                            ),
                            0,
                        )
                    )
                polynomial = torch.logsumexp(
                    torch.stack(
                        (bell[16], math.log(32) + bell[15], math.log(240) + bell[14])
                    ),
                    0,
                )
                coefficient = 4 * math.lgamma(9) - math.log(17) - 3 * math.lgamma(17)
                return maximum + polynomial + 17 * (right - left).log() + coefficient

            distribution = Censored(_AffineBeta(aa, bb, kk, kk), 0.0, 1.0)
            lower = distribution.log_prob(torch.zeros_like(kk)).exp()
            upper = distribution.log_prob(torch.ones_like(kk)).exp()
            location, variance = _positive_unit_moments(
                log_density,
                log_error,
                lower,
                upper,
                atol=self.moment_atol / 16,
                rtol=self.moment_rtol / 16,
                max_panels=self.moment_max_panels,
            )
            means.append(ordinary_mean.masked_scatter(active, location))
            variances.append(ordinary_variance.masked_scatter(active, variance))
        return torch.cat(means).reshape(shape), torch.cat(variances).reshape(shape)

    @property
    def mean(self) -> Tensor:
        """True censored response mean, computed with deterministic error bounds.

        Returns
        -------
        Tensor
            Mean on the parameter batch axes, generally different from mu.
        """
        return self._moments()[0]

    @property
    def variance(self) -> Tensor:
        """Positive within/between response variance without raw-moment cancellation.

        Returns
        -------
        Tensor
            Variance on the parameter batch axes.
        """
        return self._moments()[1]


__all__ = ["ExtendedBeta"]
