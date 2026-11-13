"""Normal distributions conditioned on an interval, evaluated in log space."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints

from quivers.continuous._ordered import _log1mexp


def _log_ndtr(value: Tensor) -> Tensor:
    """Evaluate the normal log CDF with finite tail derivatives.

    Parameters
    ----------
    value : Tensor
        Standard normal quantiles, including infinite endpoints.

    Returns
    -------
    Tensor
        Log cumulative probabilities.
    """
    finite = torch.isfinite(value)
    tail = finite & (value < -1)
    safe_tail = torch.where(tail, value, -1.0)
    tail_result = (
        torch.special.erfcx(-safe_tail / math.sqrt(2)).log()
        - 0.5 * safe_tail.square()
        - math.log(2)
    )
    safe_center = torch.where(finite & ~tail, value, 0.0)
    result = torch.where(tail, tail_result, torch.special.log_ndtr(safe_center))
    endpoint = torch.where(value < 0, -torch.inf, 0.0)
    return torch.where(finite, result, endpoint)


def _log_mass(low: Tensor, high: Tensor) -> Tensor:
    """Compute the log normal mass between standardized bounds.

    Parameters
    ----------
    low : Tensor
        Standardized lower bounds.
    high : Tensor
        Standardized upper bounds, greater than low.

    Returns
    -------
    Tensor
        Log interval probabilities.
    """
    central = (low < 0) & (high > 0)
    safe_low = torch.where(central, -1.0, low)
    safe_high = torch.where(central, 0.0, high)
    reflect = safe_low > 0
    left = torch.where(reflect, -safe_high, safe_low)
    right = torch.where(reflect, -safe_low, safe_high)
    upper = _log_ndtr(right)
    lower = _log_ndtr(left)
    tail_mass = upper + _log1mexp(lower - upper)
    central_low = torch.where(central, low, -1.0)
    central_high = torch.where(central, high, 1.0)
    # erf preserves narrow intervals through zero and saturates with
    # a finite zero derivative for intervals many scales wide.
    central_mass = torch.log(
        0.5
        * (
            torch.erf(central_high / math.sqrt(2))
            - torch.erf(central_low / math.sqrt(2))
        )
    )
    return torch.where(central, central_mass, tail_mass)


def _inverse_log_ndtr(log_probability: Tensor) -> Tensor:
    """Invert the normal log CDF without exponentiating extreme tails.

    Newton iterations solve log Phi(x) = log_probability in the left
    tail, initialized by its leading asymptotic term. Reflection maps
    probabilities above one half to the left tail. Moderate probabilities
    use torch.special.ndtri directly.

    Parameters
    ----------
    log_probability : Tensor
        Finite negative log probabilities.

    Returns
    -------
    Tensor
        Standard normal quantiles.
    """
    reflect = log_probability > -math.log(2)
    left_probability = torch.where(reflect, _log1mexp(log_probability), log_probability)
    tail = left_probability < -32
    safe_tail = torch.where(tail, left_probability, -32.0)
    quantile = -torch.sqrt(-2 * safe_tail)
    for _ in range(8):
        log_cdf = _log_ndtr(quantile)
        inverse_derivative = math.sqrt(math.pi / 2) * torch.special.erfcx(
            -quantile / math.sqrt(2)
        )
        quantile = quantile - (log_cdf - safe_tail) * inverse_derivative
    safe_center = torch.where(tail, -1.0, left_probability)
    central_quantile = torch.special.ndtri(safe_center.exp())
    result = torch.where(tail, quantile, central_quantile)
    return torch.where(reflect, -result, result)


class TruncatedNormal(Distribution):
    """A Normal distribution conditioned on low <= value <= high.

    Normalizing masses and inverse CDF inputs are evaluated in log space.
    Both bounds broadcast with the location and scale.

    Parameters
    ----------
    loc : Tensor
        Untruncated normal location.
    scale : Tensor
        Strictly positive untruncated normal standard deviation.
    low : Tensor
        Lower truncation bound.
    high : Tensor
        Upper truncation bound, strictly greater than low.
    validate_args : bool or None
        Whether to validate parameter and observation constraints.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "low": constraints.real,
        "high": constraints.real,
    }
    has_rsample = True

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        low: Tensor,
        high: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale, self.low, self.high = torch.broadcast_tensors(
            loc, scale, low, high
        )
        if not bool((self.low < self.high).all()):
            raise ValueError("TruncatedNormal requires low < high")
        super().__init__(self.loc.shape, torch.Size(), validate_args=validate_args)

    @property
    def support(self) -> constraints.Constraint:
        return constraints.interval(self.low, self.high)

    def _bounds(self) -> tuple[Tensor, Tensor]:
        """Standardize finite bounds and retain infinite endpoints.

        Returns
        -------
        tuple[Tensor, Tensor]
            Lower and upper bounds in standard normal units.
        """
        finite_low = torch.isfinite(self.low)
        finite_high = torch.isfinite(self.high)
        safe_low = torch.where(finite_low, self.low, self.loc)
        safe_high = torch.where(finite_high, self.high, self.loc)
        low = torch.where(finite_low, (safe_low - self.loc) / self.scale, self.low)
        high = torch.where(finite_high, (safe_high - self.loc) / self.scale, self.high)
        return low, high

    def log_normalizer(self) -> Tensor:
        """Return the log probability of the truncation interval.

        Returns
        -------
        Tensor
            Interval mass under the untruncated Normal, on the log scale.
        """
        return _log_mass(*self._bounds())

    def log_prob(self, value: Tensor) -> Tensor:
        """Score observations under the normalized density.

        Parameters
        ----------
        value : Tensor
            Observations in the truncation interval.

        Returns
        -------
        Tensor
            Log densities, with the broadcast observation and batch shape.
        """
        if self._validate_args:
            self._validate_sample(value)
        low, high = self._bounds()
        density = (
            -0.5 * ((value - self.loc) / self.scale).square()
            - self.scale.log()
            - 0.5 * math.log(2 * math.pi)
            - _log_mass(low, high)
        )
        return torch.where(self.support.check(value), density, -torch.inf)

    def icdf(self, value: Tensor) -> Tensor:
        """Invert cumulative probabilities in log space.

        Parameters
        ----------
        value : Tensor
            Cumulative probabilities in [0, 1].

        Returns
        -------
        Tensor
            Quantiles, including the bounds at probabilities zero and one.
        """
        if not bool(((value >= 0) & (value <= 1)).all()):
            raise ValueError(
                "TruncatedNormal quantiles require probabilities in [0, 1]"
            )
        low, high = self._bounds()
        reflect = low > 0
        left = torch.where(reflect, -high, low)
        right = torch.where(reflect, -low, high)
        interior = (value > 0) & (value < 1)
        safe_value = torch.where(interior, value, 0.5)
        log_weight = torch.where(reflect, torch.log1p(-safe_value), safe_value.log())
        log_complement = torch.where(
            reflect, safe_value.log(), torch.log1p(-safe_value)
        )
        log_probability = torch.logaddexp(
            _log_ndtr(left) + log_complement,
            _log_ndtr(right) + log_weight,
        )
        standardized = _inverse_log_ndtr(log_probability)
        standardized = torch.where(reflect, -standardized, standardized)
        result = self.loc + self.scale * standardized
        return torch.where(
            value == 0, self.low, torch.where(value == 1, self.high, result)
        )

    def cdf(self, value: Tensor) -> Tensor:
        """Evaluate cumulative probabilities on the whole real line.

        Parameters
        ----------
        value : Tensor
            Observation values.

        Returns
        -------
        Tensor
            Cumulative probabilities, zero below low and one above high.
        """
        low, high = self._bounds()
        interior = (value > self.low) & (value < self.high)
        midpoint = self.icdf(torch.full_like(value, 0.5))
        safe_value = torch.where(interior, value, midpoint)
        standardized = (safe_value - self.loc) / self.scale
        log_cdf = _log_mass(low, standardized) - _log_mass(low, high)
        result = log_cdf.exp()
        return torch.where(
            value <= self.low, 0.0, torch.where(value >= self.high, 1.0, result)
        )

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw reparameterized samples by inverse CDF.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample dimensions.

        Returns
        -------
        Tensor
            Draws with shape sample_shape + batch_shape.
        """
        uniforms = torch.rand(
            sample_shape + self.batch_shape,
            dtype=self.loc.dtype,
            device=self.loc.device,
        )
        return self.icdf(uniforms)

    def expand(
        self, batch_shape: torch.Size, _instance: TruncatedNormal | None = None
    ) -> TruncatedNormal:
        """Broadcast the distribution to a batch shape.

        Parameters
        ----------
        batch_shape : torch.Size
            Target batch dimensions.
        _instance : TruncatedNormal or None
            Optional instance for the Distribution expansion protocol.

        Returns
        -------
        TruncatedNormal
            A distribution sharing the expanded parameter tensors.
        """
        new = self._get_checked_instance(TruncatedNormal, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        super(TruncatedNormal, new).__init__(
            batch_shape, torch.Size(), validate_args=False
        )
        new._validate_args = self._validate_args
        return new


__all__: list[str] = []
