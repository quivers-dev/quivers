"""Shifted lognormal, Frechet, and generalized extreme-value response laws."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from quivers.continuous._ordinal import _log1mexp


_EULER = 0.57721566490153286061
_ZETA = (
    1.6449340668482264,
    1.2020569031595943,
    1.0823232337111382,
    1.03692775514337,
    1.0173430619844492,
    1.0083492773819228,
    1.0040773561979444,
    1.0020083928260822,
    1.000994575127818,
)


def _expm1_ratio(x: Tensor) -> Tensor:
    """Evaluate expm1(x)/x with its smooth zero limit.

    Parameters
    ----------
    x : Tensor
        Real arguments.

    Returns
    -------
    Tensor
        Ratio with value one and derivative one half at zero.
    """
    small = x.abs() < 0.001
    z = torch.where(small, x, torch.zeros_like(x))
    polynomial = 1 + z * (
        1 / 2 + z * (1 / 6 + z * (1 / 24 + z * (1 / 120 + z * (1 / 720 + z / 5040))))
    )
    regular = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, polynomial, torch.expm1(regular) / regular)


def _log1p_ratio(x: Tensor) -> Tensor:
    """Evaluate log1p(x)/x with its smooth zero limit.

    Parameters
    ----------
    x : Tensor
        Arguments greater than minus one.

    Returns
    -------
    Tensor
        Ratio including its differentiable zero limit.
    """
    small = x.abs() < 0.001
    z = torch.where(small, x, torch.zeros_like(x))
    polynomial = 1 + z * (
        -1 / 2 + z * (1 / 3 + z * (-1 / 4 + z * (1 / 5 + z * (-1 / 6 + z / 7))))
    )
    regular = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, polynomial, torch.log1p(regular) / regular)


def _log_gamma1m_ratio(x: Tensor) -> Tensor:
    """Evaluate log(Gamma(1-x))/x including its zero limit.

    Parameters
    ----------
    x : Tensor
        Real values below one.

    Returns
    -------
    Tensor
        The ratio, with Euler's constant at zero.
    """
    small = x.abs() < 0.01
    z = torch.where(small, x, torch.zeros_like(x))
    polynomial = _ZETA[-1] / 10
    for n in range(9, 1, -1):
        polynomial = _ZETA[n - 2] / n + z * polynomial
    polynomial = _EULER + z * polynomial
    regular = torch.where(small, torch.full_like(x, 0.1), x)
    return torch.where(small, polynomial, torch.lgamma(1 - regular) / regular)


def _gamma_variance_ratio(x: Tensor) -> Tensor:
    """Evaluate the squared-shape ratio in Frechet and GEV variances.

    Parameters
    ----------
    x : Tensor
        Shape or reciprocal shape, below one half.

    Returns
    -------
    Tensor
        ``(Gamma(1-2*x)-Gamma(1-x)**2)/x**2`` with its zero limit.
    """
    small = x.abs() < 0.01
    z = torch.where(small, x, torch.zeros_like(x))
    polynomial = _ZETA[-1] * (2**10 - 2) / 10
    for n in range(9, 1, -1):
        polynomial = _ZETA[n - 2] * (2**n - 2) / n + z * polynomial
    log_gamma = z * _log_gamma1m_ratio(z)
    series = (2 * log_gamma).exp() * polynomial * _expm1_ratio(z.square() * polynomial)
    regular = torch.where(small, torch.full_like(x, 0.1), x)
    log_first = torch.lgamma(1 - regular)
    difference = torch.lgamma(1 - 2 * regular) - 2 * log_first
    result = (2 * log_first).exp() * torch.expm1(difference) / regular.square()
    return torch.where(small, series, result)


def _log1mexp_neg_exp(log_t: Tensor) -> Tensor:
    """Evaluate log(1-exp(-exp(log_t))) throughout both exponential tails.

    Parameters
    ----------
    log_t : Tensor
        Real log intensities.

    Returns
    -------
    Tensor
        Log survival without dropping the small-intensity correction.
    """
    small = log_t < -2
    large = log_t > math.log(750)
    small_t = torch.exp(torch.where(small, log_t, torch.zeros_like(log_t)))
    left = log_t + _expm1_ratio(-small_t).log()
    regular = torch.where(small | large, torch.zeros_like(log_t), log_t)
    right = _log1mexp(-regular.exp())
    return torch.where(large, torch.zeros_like(log_t), torch.where(small, left, right))


def _check_probability(value: Tensor) -> None:
    """Validate a quantile probability.

    Parameters
    ----------
    value : Tensor
        Candidate probabilities.

    Raises
    ------
    ValueError
        If a value is outside the closed unit interval.
    """
    if not bool(constraints.unit_interval.check(value).all()):
        raise ValueError("icdf probabilities must be in [0,1]")


class ShiftedLogNormal(Distribution):
    """A lognormal response plus a nonnegative location shift.

    Parameters
    ----------
    loc : Tensor or float
        Mean of the underlying normal log response.
    scale : Tensor or float
        Positive standard deviation of the underlying normal.
    shift : Tensor or float
        Nonnegative lower support endpoint.
    validate_args : bool or None
        Whether to validate parameters.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "shift": constraints.nonnegative,
    }
    has_rsample = True

    def __init__(self, loc, scale, shift, validate_args=None):
        self.loc, self.scale, self.shift = broadcast_all(loc, scale, shift)
        super().__init__(self.loc.shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        """Return the parameter-dependent response support.

        Returns
        -------
        constraints.Constraint
            Broadcast result with the parameter device and dtype.
        """
        return constraints.greater_than(self.shift)

    def expand(self, batch_shape, _instance=None):
        """Expand parameter views without copying their underlying storage.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Requested broadcast batch dimensions.
        _instance : Distribution or None
            Optional instance supplied by PyTorch.

        Returns
        -------
        Distribution
            Broadcast result with the parameter device and dtype.
        """
        new = self._get_checked_instance(type(self), _instance)
        shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(shape))
        super(ShiftedLogNormal, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _standardized(self, value):
        """Standardize supported shifted responses while masking excluded values.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Broadcast result with the parameter device and dtype.
        """
        difference = value - self.shift
        inside = (difference > 0) & torch.isfinite(difference)
        safe = torch.where(inside, difference, torch.ones_like(difference))
        return difference, inside, (safe.log() - self.loc) / self.scale, safe

    def log_prob(self, value):
        """Evaluate log densities, returning negative infinity outside the support.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        _, inside, z, safe = self._standardized(value)
        result = (
            -0.5 * z.square() - self.scale.log() - safe.log() - 0.91893853320467274178
        )
        return torch.where(inside, result, -torch.inf)

    def log_cdf(self, value):
        """Evaluate cumulative probabilities directly in log space.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        difference, inside, z, _ = self._standardized(value)
        return torch.where(
            inside,
            torch.special.log_ndtr(z),
            torch.where(difference > 0, 0.0, -torch.inf),
        )

    def log_survival_function(self, value):
        """Evaluate complementary probabilities directly in log space.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        difference, inside, z, _ = self._standardized(value)
        return torch.where(
            inside,
            torch.special.log_ndtr(-z),
            torch.where(difference > 0, -torch.inf, 0.0),
        )

    def cdf(self, value):
        """Evaluate cumulative probabilities.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        return self.log_cdf(value).exp()

    def icdf(self, value):
        """Transform probabilities to response quantiles.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        if self._validate_args:
            _check_probability(value)
        return self.shift + torch.exp(
            self.loc + self.scale * torch.special.ndtri(value)
        )

    def rsample(self, sample_shape=torch.Size()):
        """Draw responses with pathwise gradients through the distribution parameters.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading sample dimensions.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        z = torch.randn(
            self._extended_shape(sample_shape),
            dtype=self.loc.dtype,
            device=self.loc.device,
        )
        return self.shift + torch.exp(self.loc + self.scale * z)

    @property
    def mean(self):
        """Return the response expectation, or infinity when its first moment does not exist.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        return self.shift + torch.exp(self.loc + self.scale.square() / 2)

    @property
    def variance(self):
        """Return the response variance, or infinity when its second moment does not exist.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        return torch.expm1(self.scale.square()) * torch.exp(
            2 * self.loc + self.scale.square()
        )


class Frechet(Distribution):
    """Frechet distribution on positive responses.

    Parameters
    ----------
    scale : Tensor or float
        Positive response scale.
    concentration : Tensor or float
        Positive tail shape. The first moment exists above one and the
        second moment above two.
    validate_args : bool or None
        Whether to validate parameters.
    """

    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.positive,
    }
    support = constraints.positive
    has_rsample = True

    def __init__(self, scale, concentration, validate_args=None):
        self.scale, self.concentration = broadcast_all(scale, concentration)
        super().__init__(self.scale.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        """Expand parameter views without copying their underlying storage.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Requested broadcast batch dimensions.
        _instance : Distribution or None
            Optional instance supplied by PyTorch.

        Returns
        -------
        Distribution
            Broadcast result with the parameter device and dtype.
        """
        new = self._get_checked_instance(type(self), _instance)
        shape = torch.Size(batch_shape)
        new.scale, new.concentration = (
            self.scale.expand(shape),
            self.concentration.expand(shape),
        )
        super(Frechet, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _log_ratio(self, value):
        """Compute safe log response-to-scale ratios.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        tuple[Tensor, Tensor]
            Broadcast result with the parameter device and dtype.
        """
        inside = (value > 0) & torch.isfinite(value)
        point = torch.where(inside, value, self.scale)
        return inside, point.log() - self.scale.log()

    def log_prob(self, value):
        """Evaluate log densities, returning negative infinity outside the support.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        inside, ratio = self._log_ratio(value)
        a = self.concentration
        result = a.log() - self.scale.log() - (a + 1) * ratio - torch.exp(-a * ratio)
        return torch.where(inside, result, -torch.inf)

    def log_cdf(self, value):
        """Evaluate cumulative probabilities directly in log space.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        inside, ratio = self._log_ratio(value)
        result = -torch.exp(-self.concentration * ratio)
        return torch.where(inside, result, torch.where(value > 0, 0.0, -torch.inf))

    def log_survival_function(self, value):
        """Evaluate complementary probabilities directly in log space.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        inside, ratio = self._log_ratio(value)
        log_t = -self.concentration * ratio
        result = _log1mexp_neg_exp(log_t)
        return torch.where(inside, result, torch.where(value > 0, -torch.inf, 0.0))

    def cdf(self, value):
        """Evaluate cumulative probabilities.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        return self.log_cdf(value).exp()

    def icdf(self, value):
        """Transform probabilities to response quantiles.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        if self._validate_args:
            _check_probability(value)
        return self.scale * torch.exp(-torch.log(-value.log()) / self.concentration)

    def rsample(self, sample_shape=torch.Size()):
        """Draw responses with pathwise gradients through the distribution parameters.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading sample dimensions.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        exponential = torch.empty(
            self._extended_shape(sample_shape),
            dtype=self.scale.dtype,
            device=self.scale.device,
        ).exponential_()
        return self.scale * torch.exp(-exponential.log() / self.concentration)

    @property
    def mean(self):
        """Return the response expectation, or infinity when its first moment does not exist.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        valid = self.concentration > 1
        a = torch.where(
            valid, self.concentration, torch.full_like(self.concentration, 2)
        )
        x = a.reciprocal()
        result = self.scale * torch.exp(x * _log_gamma1m_ratio(x))
        return torch.where(valid, result, torch.inf)

    @property
    def variance(self):
        """Return the response variance, or infinity when its second moment does not exist.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        valid = self.concentration > 2
        a = torch.where(
            valid, self.concentration, torch.full_like(self.concentration, 3)
        )
        x = a.reciprocal()
        result = self.scale.square() * x.square() * _gamma_variance_ratio(x)
        return torch.where(valid, result, torch.inf)


class GeneralizedExtremeValue(Distribution):
    """Generalized extreme-value law with a differentiable Gumbel limit.

    Parameters
    ----------
    loc : Tensor or float
        Response location, distinct from the expectation.
    scale : Tensor or float
        Positive scale.
    concentration : Tensor or float
        Real extreme-value shape; zero gives Gumbel. SciPy's genextreme
        uses the opposite sign for its shape parameter.
    validate_args : bool or None
        Whether to validate parameters.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    has_rsample = True

    def __init__(self, loc, scale, concentration, validate_args=None):
        self.loc, self.scale, self.concentration = broadcast_all(
            loc, scale, concentration
        )
        super().__init__(self.loc.shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        """Return the parameter-dependent response support.

        Returns
        -------
        constraints.Constraint
            Broadcast result with the parameter device and dtype.
        """
        c = self.concentration
        safe = torch.where(c != 0, c, torch.ones_like(c))
        endpoint = self.loc - self.scale / safe
        return constraints.interval(
            torch.where(c > 0, endpoint, -torch.inf),
            torch.where(c < 0, endpoint, torch.inf),
        )

    def expand(self, batch_shape, _instance=None):
        """Expand parameter views without copying their underlying storage.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Requested broadcast batch dimensions.
        _instance : Distribution or None
            Optional instance supplied by PyTorch.

        Returns
        -------
        Distribution
            Broadcast result with the parameter device and dtype.
        """
        new = self._get_checked_instance(type(self), _instance)
        shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(shape))
        super(GeneralizedExtremeValue, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _transformed(self, value):
        """Evaluate the support-aware generalized extreme-value transformation.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Broadcast result with the parameter device and dtype.
        """
        finite = torch.isfinite(value)
        point = torch.where(finite, value, self.loc)
        z = (point - self.loc) / self.scale
        inside = finite & (1 + self.concentration * z > 0)
        safe_z = torch.where(inside, z, torch.zeros_like(z))
        endpoint_z = torch.where(finite, z, value)
        return endpoint_z, inside, safe_z * _log1p_ratio(self.concentration * safe_z)

    def log_prob(self, value):
        """Evaluate log densities, returning negative infinity outside the support.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        z, inside, r = self._transformed(value)
        result = -self.scale.log() - (1 + self.concentration) * r - torch.exp(-r)
        endpoint = (self.concentration < 0) & (1 + self.concentration * z == 0)
        limit = torch.where(
            self.concentration == -1,
            -self.scale.log(),
            torch.where(self.concentration < -1, torch.inf, -torch.inf),
        )
        return torch.where(inside, result, torch.where(endpoint, limit, -torch.inf))

    def log_cdf(self, value):
        """Evaluate cumulative probabilities directly in log space.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        z, inside, r = self._transformed(value)
        result = -torch.exp(-r)
        upper = (self.concentration < 0) & (1 + self.concentration * z <= 0) | (
            z == torch.inf
        )
        return torch.where(inside, result, torch.where(upper, 0.0, -torch.inf))

    def log_survival_function(self, value):
        """Evaluate complementary probabilities directly in log space.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        z, inside, r = self._transformed(value)
        result = _log1mexp_neg_exp(-r)
        upper = (self.concentration < 0) & (1 + self.concentration * z <= 0) | (
            z == torch.inf
        )
        return torch.where(inside, result, torch.where(upper, -torch.inf, 0.0))

    def cdf(self, value):
        """Evaluate cumulative probabilities.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        return self.log_cdf(value).exp()

    def icdf(self, value):
        """Transform probabilities to response quantiles.

        Parameters
        ----------
        value : Tensor
            Response values; for icdf, probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        if self._validate_args:
            _check_probability(value)
        interior = (value > 0) & (value < 1)
        p = torch.where(interior, value, torch.full_like(value, 0.5))
        gumbel = -torch.log(-p.log())
        result = self.loc + self.scale * gumbel * _expm1_ratio(
            self.concentration * gumbel
        )
        safe = torch.where(
            self.concentration != 0,
            self.concentration,
            torch.ones_like(self.concentration),
        )
        endpoint = self.loc - self.scale / safe
        lower = torch.where(self.concentration > 0, endpoint, -torch.inf)
        upper = torch.where(self.concentration < 0, endpoint, torch.inf)
        return torch.where(interior, result, torch.where(value == 0, lower, upper))

    def rsample(self, sample_shape=torch.Size()):
        """Draw responses with pathwise gradients through the distribution parameters.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading sample dimensions.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        exponential = torch.empty(
            self._extended_shape(sample_shape),
            dtype=self.loc.dtype,
            device=self.loc.device,
        ).exponential_()
        gumbel = -exponential.log()
        return self.loc + self.scale * gumbel * _expm1_ratio(
            self.concentration * gumbel
        )

    @property
    def mean(self):
        """Return the response expectation, or infinity when its first moment does not exist.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        valid = self.concentration < 1
        c = torch.where(valid, self.concentration, torch.zeros_like(self.concentration))
        ratio = _log_gamma1m_ratio(c)
        result = self.loc + self.scale * ratio * _expm1_ratio(c * ratio)
        return torch.where(valid, result, torch.inf)

    @property
    def variance(self):
        """Return the response variance, or infinity when its second moment does not exist.

        Returns
        -------
        Tensor
            Broadcast result with the parameter device and dtype.
        """
        valid = self.concentration < 0.5
        c = torch.where(valid, self.concentration, torch.zeros_like(self.concentration))
        return torch.where(
            valid, self.scale.square() * _gamma_variance_ratio(c), torch.inf
        )


__all__ = ["ShiftedLogNormal", "Frechet", "GeneralizedExtremeValue"]
