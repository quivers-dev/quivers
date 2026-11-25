"""Conway--Maxwell--Poisson counts with certified sums and exact rejection."""

import math

import torch
from quivers._positions import demand, entry_count, everywhere, reject_unless
from quivers.continuous._closed_probability import _check_count_observation_source
from torch import Tensor
from torch.distributions import Distribution, constraints

from quivers.continuous._closed_probability import (
    ClosedPoisson,
    _require_sample_capacity,
)
from quivers.continuous._zero_inflated_count import _broadcast_count_parameters


def _relative_weight(k: Tensor, log_rate: Tensor, nu: Tensor, mode: Tensor) -> Tensor:
    """Evaluate log weights relative to a modal count.

    Parameters
    ----------
    k, log_rate, nu, mode : Tensor
        Counts, log rates, dispersions, and detached modal counts.

    Returns
    -------
    Tensor
        Stable relative log weights.
    """
    log_rho = log_rate / nu
    ordinary = log_rho > -700
    safe_log = torch.where(ordinary, log_rho, 0.0)
    rho = safe_log.exp()
    poisson = ClosedPoisson(rho, validate_args=False)
    centered = nu * (poisson.log_prob(k) - poisson.log_prob(mode))
    centered = centered + (k - mode) * (log_rate - nu * safe_log)
    direct = k * log_rate - nu * torch.lgamma(k + 1)
    return torch.where(ordinary, centered, direct)


def _geometry(log_ratio: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Bound geometric tails weighted by powers through degree four.

    Parameters
    ----------
    log_ratio : Tensor
        Strictly negative logarithms of geometric ratios.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        Sums of j**d * ratio**j for j>=1 and degrees zero through four.
    """
    r = log_ratio.exp()
    d = -torch.expm1(log_ratio)
    return (
        r / d,
        r / d.square(),
        r * (1 + r) / d.pow(3),
        r * (1 + 4 * r + r.square()) / d.pow(4),
        r * (1 + 11 * r + 11 * r.square() + r.pow(3)) / d.pow(5),
    )


class COMPoisson(Distribution):
    """Counts proportional to ``rate**k / (k!)**dispersion``.

    Parameters
    ----------
    rate : Tensor or float or None
        Nonnegative rate. Supply exactly one of rate and log_rate.
    dispersion : Tensor or float
        Strictly positive dispersion exponent.
    log_rate : Tensor or float or None
        Logarithm of the rate, permitting rates beyond floating-point range.
    validate_args : bool or None
        Whether to validate parameters and samples.
    normalization_rtol : float
        Certified relative tail tolerance for normalization and moments.
    max_terms : int
        Maximum centered count cells per row before an explicit refusal.
    max_proposals : int
        Maximum rejection rounds per requested sample.
    """

    arg_constraints = {
        "rate": constraints.nonnegative,
        "dispersion": constraints.positive,
        "log_rate": constraints.real,
    }
    support = constraints.nonnegative_integer
    has_rsample = False

    def __init__(
        self,
        rate=None,
        dispersion=1.0,
        *,
        log_rate=None,
        validate_args=None,
        normalization_rtol=1e-12,
        max_terms=262144,
        max_proposals=10000,
    ):
        if (rate is None) == (log_rate is None):
            raise ValueError("COMPoisson requires exactly one of rate and log_rate")
        if (
            not 0 < normalization_rtol < 1
            or not isinstance(max_terms, int)
            or not isinstance(max_proposals, int)
            or max_terms < 1
            or max_proposals < 1
        ):
            raise ValueError(
                "COMPoisson tolerances and resource budgets must be positive"
            )
        first, self.dispersion = _broadcast_count_parameters(
            log_rate if rate is None else rate, dispersion
        )
        if rate is None:
            self.log_rate = first
            self.rate = first.exp()
            self._boundary_rate = torch.zeros_like(first)
        else:
            self.rate = first
            self._boundary_rate = first
            demand(
                torch.isfinite(first) & (first >= 0),
                lambda: ValueError("COMPoisson rate must be finite and nonnegative"),
            )
            positive = first > 0
            self.log_rate = torch.where(
                positive, torch.where(positive, first, 1.0).log(), -torch.inf
            )
        demand(
            torch.isfinite(self.dispersion) & (self.dispersion > 0),
            lambda: ValueError("COMPoisson dispersion must be finite and positive"),
        )
        demand(
            torch.isfinite(self.log_rate) | (self.log_rate == -torch.inf),
            lambda: ValueError(
                "COMPoisson log_rate must be finite or negative infinity"
            ),
        )
        self.normalization_rtol = normalization_rtol
        self.max_terms = max_terms
        self.max_proposals = max_proposals
        # The negative-infinite log rate is the admitted zero-rate limit.
        super().__init__(self.dispersion.shape, validate_args=False)
        self._validate_args = bool(validate_args)

    def expand(self, batch_shape, _instance=None):
        """Expand parameter views while retaining the numerical policy.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Requested batch shape.
        _instance : COMPoisson or None
            Unused torch expansion hook.

        Returns
        -------
        COMPoisson
            Expanded distribution.
        """
        result = COMPoisson(
            log_rate=self.log_rate.expand(batch_shape),
            dispersion=self.dispersion.expand(batch_shape),
            validate_args=self._validate_args,
            normalization_rtol=self.normalization_rtol,
            max_terms=self.max_terms,
            max_proposals=self.max_proposals,
        )
        result._boundary_rate = self._boundary_rate.expand(batch_shape)
        return result

    def _parameters(self):
        """Resolve safe positive-rate parameters and a representable mode.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Safe log rates, dispersions, and detached modal counts.

        Raises
        ------
        NotImplementedError
            If the modal count exceeds exact Float64 capacity.
        """
        log_rate = torch.where(self.log_rate == -torch.inf, -1000.0, self.log_rate)
        log_rho = log_rate / self.dispersion
        beyond = log_rho > math.log(2**53)
        reject_unless(
            ~beyond,
            lambda: NotImplementedError(
                "COMPoisson modal count exceeds exact Float64 capacity 2**53"
            ),
        )
        log_rate = torch.where(beyond, 0.0, log_rate)
        log_rho = torch.where(beyond, 0.0, log_rho)
        return log_rate, self.dispersion, log_rho.exp().floor().detach()

    def _statistics(self):
        """Accumulate centered mass and moments with geometric tail certificates.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Modal counts, centered log normalizers, means, and variances.

        Raises
        ------
        NotImplementedError
            If the requested tail tolerance exceeds the count-cell budget.
        """
        log_rate, nu, mode = self._parameters()
        total = torch.zeros_like(mode)
        first = torch.zeros_like(mode)
        second = torch.zeros_like(mode)
        fourth = torch.zeros_like(mode)
        centered_fourth = torch.zeros_like(mode)
        offset = 0
        # Bounded tiles avoid a batch-by-entire-support allocation.
        width = max(1, min(64, 131072 // max(1, entry_count(mode))))
        radius_budget = (self.max_terms + 1) // 2
        done = torch.zeros_like(mode, dtype=torch.bool)
        while offset < radius_budget and not everywhere(done):
            end = min(offset + width, radius_budget)
            j = torch.arange(offset, end, device=mode.device, dtype=mode.dtype)
            signed = torch.cat((j, -j[j > 0]))
            k = mode.unsqueeze(-1) + signed
            valid = k >= 0
            safe_k = k.clamp_min(0)
            w = _relative_weight(
                safe_k, log_rate.unsqueeze(-1), nu.unsqueeze(-1), mode.unsqueeze(-1)
            ).exp()
            w = torch.where(valid, w, 0.0)
            total = total + w.sum(-1)
            first = first + (w * signed).sum(-1)
            second = second + (w * signed.square()).sum(-1)
            fourth = fourth + (w * safe_k.pow(4)).sum(-1)
            centered_fourth = centered_fourth + (w * signed.pow(4)).sum(-1)
            # A float radius keeps radius**4 representable beyond int64.
            radius = float(end - 1)
            with torch.no_grad():
                upper = mode + radius
                lower = (mode - radius).clamp_min(0)
                ur = log_rate - nu * torch.log(upper + 1)
                lr = nu * torch.log(lower.clamp_min(1)) - log_rate
                ratios_valid = (ur < 0) & ((lower == 0) | (lr < 0))
                safe_ur = torch.where(ur < 0, ur, -1.0)
                safe_lr = torch.where(lr < 0, lr, -1.0)
                us = _geometry(safe_ur)
                ls = _geometry(safe_lr)
                uw = _relative_weight(upper, log_rate, nu, mode).exp()
                lw = _relative_weight(lower, log_rate, nu, mode).exp()
                left = torch.where(lower > 0, lw * ls[0], 0.0)
                bound0 = uw * us[0] + left
                bound4 = uw * sum(
                    math.comb(4, d) * upper.pow(4 - d) * us[d] for d in range(5)
                ) + left * lower.pow(4)
                centered_tail2 = sum(
                    math.comb(2, d)
                    * radius ** (2 - d)
                    * (uw * us[d] + torch.where(lower > 0, lw * ls[d], 0.0))
                    for d in range(3)
                )
                centered_tail4 = sum(
                    math.comb(4, d)
                    * radius ** (4 - d)
                    * (uw * us[d] + torch.where(lower > 0, lw * ls[d], 0.0))
                    for d in range(5)
                )
                done = (
                    ratios_valid
                    & (bound0 <= self.normalization_rtol * total)
                    & (bound4 <= self.normalization_rtol * (1 + fourth))
                    & (centered_tail2 <= self.normalization_rtol * (1 + second))
                    & (
                        centered_tail4
                        <= self.normalization_rtol * (1 + centered_fourth)
                    )
                )
            offset = end
        reject_unless(
            done,
            lambda: NotImplementedError(
                "COMPoisson centered sum could not certify tails within max_terms"
            ),
        )
        center = first / total
        mean = mode + center
        variance = (second / total - center.square()).clamp_min(0)
        zero = self.log_rate == -torch.inf
        return (
            mode,
            torch.where(zero, 0.0, total.log()),
            torch.where(zero, self._boundary_rate, mean),
            torch.where(zero, self._boundary_rate, variance),
        )

    def log_prob(self, value):
        """Score counts with a certified centered normalizer.

        Parameters
        ----------
        value : Tensor
            Candidate count values.

        Returns
        -------
        Tensor
            Float64 log masses, negative infinity off support.
        """
        _check_count_observation_source(value)
        value = torch.as_tensor(
            value, device=self.dispersion.device, dtype=torch.float64
        )
        if self._validate_args:
            self._validate_sample(value)
        mode, log_sum, _, _ = self._statistics()
        log_rate, nu, _ = self._parameters()
        valid = torch.isfinite(value) & (value >= 0) & (value == value.floor())
        result = (
            _relative_weight(torch.where(valid, value, 0.0), log_rate, nu, mode)
            - log_sum
        )
        result = torch.where(
            self.log_rate == -torch.inf,
            torch.where(value == 0, -self._boundary_rate, -torch.inf),
            result,
        )
        return torch.where(valid, result, -torch.inf)

    @property
    def mean(self):
        """Return a certified response mean.

        Returns
        -------
        Tensor
            Float64 response means.
        """
        return self._statistics()[2]

    @property
    def variance(self):
        """Return a certified response variance.

        Returns
        -------
        Tensor
            Float64 response variances.
        """
        return self._statistics()[3]

    def sample(self, sample_shape=torch.Size()):
        """Sample from a log-concave envelope without truncating the law.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading sample dimensions.

        Returns
        -------
        Tensor
            Int64 count samples.

        Raises
        ------
        RuntimeError
            If a proposal exceeds exact count capacity or the round budget.
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            log_rate = self.log_rate.expand(shape).reshape(-1)
            nu = self.dispersion.expand(shape).reshape(-1)
            output = torch.zeros_like(log_rate)
            active = torch.nonzero(torch.isfinite(log_rate), as_tuple=True)[0]
            for _ in range(self.max_proposals):
                if active.numel() == 0:
                    return output.reshape(shape).to(torch.int64)
                lr, power = log_rate[active], nu[active]
                rho = _require_sample_capacity((lr / power).exp())
                mode = rho.floor()
                width = (rho / power).sqrt().clamp_min(1)
                left = (mode - width.ceil()).clamp_min(0)
                right = _require_sample_capacity(mode + width.ceil())
                log_lratio = power * left.clamp_min(1).log() - lr
                log_rratio = lr - power * (right + 1).log()
                lweight = _relative_weight(left, lr, power, mode)
                rweight = _relative_weight(right, lr, power, mode)
                lmass = torch.where(
                    left > 0,
                    (lweight + log_lratio).exp() / -torch.expm1(log_lratio),
                    0.0,
                )
                rmass = (rweight + log_rratio).exp() / -torch.expm1(log_rratio)
                middle = right - left + 1
                pick = torch.rand_like(lr) * (middle + lmass + rmass)
                in_left = pick < lmass
                in_right = pick >= lmass + middle
                u = torch.rand_like(lr).clamp_min(torch.finfo(torch.float64).tiny)
                safe_lratio = torch.where(left > 0, log_lratio, -1.0)
                lj = (u.log() / safe_lratio).floor() + 1
                rj = (u.log() / log_rratio).floor() + 1
                center = left + (torch.rand_like(lr) * middle).floor()
                candidate = torch.where(
                    in_left, left - lj, torch.where(in_right, right + rj, center)
                )
                nonnegative = candidate >= 0
                _require_sample_capacity(candidate.clamp_min(0))
                envelope = torch.where(
                    in_left,
                    lweight + lj * log_lratio,
                    torch.where(in_right, rweight + rj * log_rratio, 0.0),
                )
                actual = _relative_weight(candidate.clamp_min(0), lr, power, mode)
                accept = nonnegative & (torch.rand_like(lr).log() <= actual - envelope)
                output[active[accept]] = candidate[accept]
                active = active[~accept]
        raise RuntimeError("COMPoisson rejection sampling exhausted max_proposals")


__all__ = ["COMPoisson"]
