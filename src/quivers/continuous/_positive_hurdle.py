"""Exact zero atoms combined with strictly positive response distributions."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, Gamma, LogNormal, constraints
from torch.distributions.utils import broadcast_all

from quivers._positions import demand
from quivers.continuous._interval import log_interval_mass
from quivers.continuous._recorded_event_tails import (
    _log_add,
    _log_mass,
    log_hurdle_tails,
)


class _PositiveHurdle(Distribution):
    """Shared scoring and sampling under the atom-plus-positive reference measure."""

    support = constraints.nonnegative
    has_rsample = False
    zero_prob: Tensor

    def _base(self) -> Distribution:
        """Return the strictly positive component distribution.

        Returns
        -------
        Distribution
            The subclass's Gamma or LogNormal component.

        Raises
        ------
        NotImplementedError
            If a subclass does not supply its positive component.
        """
        raise NotImplementedError

    @property
    def mean(self) -> Tensor:
        """Mean including the structural zero probability.

        Returns
        -------
        Tensor
            Positive-component mean multiplied by its mixture probability.
        """
        return (1 - self.zero_prob) * self._base().mean

    @property
    def variance(self) -> Tensor:
        """Variance including within-component and between-component variation.

        Returns
        -------
        Tensor
            Variance of the full response distribution.
        """
        base = self._base()
        positive = 1 - self.zero_prob
        return positive * base.variance + self.zero_prob * positive * base.mean.square()

    def expand(self, batch_shape: torch.Size, _instance=None) -> _PositiveHurdle:
        """Expand parameter views to a broadcast batch shape.

        Parameters
        ----------
        batch_shape : torch.Size
            Requested batch shape.
        _instance : _PositiveHurdle or None
            Optional distribution instance supplied by PyTorch internals.

        Returns
        -------
        _PositiveHurdle
            A distribution sharing expanded parameter tensors.
        """
        new = self._get_checked_instance(type(self), _instance)
        shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(shape))
        super(_PositiveHurdle, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value: Tensor) -> Tensor:
        """Score the zero atom or the positive component with safe inactive inputs.

        Parameters
        ----------
        value : Tensor
            Responses broadcastable with the distribution's batch shape.

        Returns
        -------
        Tensor
            Log mass at zero or log density on the positive half-line.
        """
        if self._validate_args:
            self._validate_sample(value)
        at_zero = value == 0
        positive = (value > 0) & torch.isfinite(value)
        zero_probability = torch.where(at_zero, self.zero_prob, 1.0)
        positive_probability = torch.where(positive, self.zero_prob, 0.0)
        # Gamma can have an infinite density at zero. The continuous
        # density must not contribute to the atom, even in an inactive
        # autograd branch, so evaluate that branch at a safe positive point.
        point = torch.where(positive, value, 1.0)
        atom = zero_probability.log()
        interior = torch.log1p(-positive_probability) + self._base().log_prob(point)
        return torch.where(at_zero, atom, torch.where(positive, interior, -torch.inf))

    def cdf(self, value: Tensor) -> Tensor:
        """Evaluate the CDF, including its jump at zero.

        Parameters
        ----------
        value : Tensor
            Real-valued query points, including negative points and infinities.

        Returns
        -------
        Tensor
            Cumulative probabilities, differentiable in every parameter
            including the Gamma shape.
        """
        result = self._log_tails(value)[0].exp().to(self.zero_prob.dtype)
        return torch.where(torch.isnan(value), value, result)

    def _log_tails(self, value: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the log CDF and log survival, including the zero atom.

        Parameters
        ----------
        value : Tensor
            Real query points.

        Returns
        -------
        tuple[Tensor, Tensor]
            Float64 ``log P(Y <= value)`` and ``log P(Y > value)``.
        """
        point = torch.where(torch.isnan(value), 0.0, value)
        return log_hurdle_tails(self.zero_prob, self._base(), point, left=False)

    def log_interval_mass(
        self,
        low: Tensor | float | None,
        high: Tensor | float | None,
        *,
        lower_closed: bool | Tensor = True,
        upper_closed: bool | Tensor = True,
    ) -> Tensor:
        """Log mass of an interval, adding the atom and the positive part apart.

        The mass is ``p0 [0 in I] + (1 - p0) G(I)`` for the positive
        component ``G``. Scoring ``G(I)`` on its own keeps an interval far in
        a tail of the component finite, where the difference of two composed
        tails would cancel against the atom. Cost is one component interval
        mass per query.

        Parameters
        ----------
        low : Tensor | float | None
            Lower event bound; None leaves that side unbounded.
        high : Tensor | float | None
            Upper event bound; None leaves that side unbounded.
        lower_closed : bool | Tensor
            Whether the lower endpoint belongs to the event.
        upper_closed : bool | Tensor
            Whether the upper endpoint belongs to the event.

        Returns
        -------
        Tensor
            Float64 log masses, negative infinity for an empty event.

        Raises
        ------
        ValueError
            If an endpoint is NaN.
        """
        device = self.zero_prob.device
        lower = torch.as_tensor(
            -torch.inf if low is None else low, device=device, dtype=torch.float64
        )
        upper = torch.as_tensor(
            torch.inf if high is None else high, device=device, dtype=torch.float64
        )
        demand(~lower.isnan(), lambda: ValueError("interval endpoints must not be NaN"))
        demand(~upper.isnan(), lambda: ValueError("interval endpoints must not be NaN"))
        lower_flag = torch.as_tensor(lower_closed, device=device)
        upper_flag = torch.as_tensor(upper_closed, device=device)
        holds_atom = ((lower < 0) | ((lower == 0) & lower_flag)) & (
            (upper > 0) | ((upper == 0) & upper_flag)
        )
        start = lower.clamp_min(0.0)
        occupied = upper > start
        component = log_interval_mass(
            self._base(),
            torch.where(occupied, start, 0.0),
            torch.where(occupied, upper, 1.0),
        ).double()
        probability = self.zero_prob.double()
        return _log_add(
            torch.where(holds_atom, _log_mass(probability), -torch.inf),
            torch.where(occupied, _log_mass(1 - probability) + component, -torch.inf),
        )

    def log_cdf_left(self, value: Tensor) -> Tensor:
        """Evaluate ``log P(Y < value)``, excluding an atom at the query.

        Parameters
        ----------
        value : Tensor
            Real query points.

        Returns
        -------
        Tensor
            Float64 log left limits of the CDF.
        """
        return log_hurdle_tails(self.zero_prob, self._base(), value, left=True)[0]

    def log_survival_function_left(self, value: Tensor) -> Tensor:
        """Evaluate ``log P(Y >= value)``, including an atom at the query.

        Parameters
        ----------
        value : Tensor
            Real query points.

        Returns
        -------
        Tensor
            Float64 log survival including the query point.
        """
        return log_hurdle_tails(self.zero_prob, self._base(), value, left=True)[1]

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw independent atom indicators and positive component values.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading independent sample dimensions.

        Returns
        -------
        Tensor
            Zero or positive responses with sample, batch, and event axes.
        """
        with torch.no_grad():
            shape = sample_shape + self.batch_shape
            positive = (
                torch.rand(
                    shape, dtype=self.zero_prob.dtype, device=self.zero_prob.device
                )
                >= self.zero_prob
            )
            return torch.where(positive, self._base().sample(sample_shape), 0.0)


class HurdleGamma(_PositiveHurdle):
    """A zero atom and Gamma concentration/rate response component.

    Parameters
    ----------
    zero_prob : Tensor or float
        Structural zero probability, including zero and one.
    concentration : Tensor or float
        Positive Gamma shape.
    rate : Tensor or float
        Positive Gamma rate.
    validate_args : bool or None
        Whether to validate parameters and scored observations.
    """

    arg_constraints = {
        "zero_prob": constraints.unit_interval,
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }

    def __init__(
        self,
        zero_prob: Tensor | float,
        concentration: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.zero_prob, self.concentration, self.rate = broadcast_all(
            zero_prob, concentration, rate
        )
        super().__init__(self.zero_prob.shape, validate_args=validate_args)

    def _base(self) -> Gamma:
        """Build the positive Gamma component.

        Returns
        -------
        Gamma
            Gamma distribution with the retained shape and rate.
        """
        return Gamma(self.concentration, self.rate, validate_args=self._validate_args)


class HurdleLogNormal(_PositiveHurdle):
    """A zero atom and LogNormal log-location/log-scale response component.

    Parameters
    ----------
    zero_prob : Tensor or float
        Structural zero probability, including zero and one.
    loc : Tensor or float
        Mean of the underlying Normal log response.
    scale : Tensor or float
        Positive standard deviation of the underlying Normal log response.
    validate_args : bool or None
        Whether to validate parameters and scored observations.
    """

    arg_constraints = {
        "zero_prob": constraints.unit_interval,
        "loc": constraints.real,
        "scale": constraints.positive,
    }

    def __init__(
        self,
        zero_prob: Tensor | float,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.zero_prob, self.loc, self.scale = broadcast_all(zero_prob, loc, scale)
        super().__init__(self.zero_prob.shape, validate_args=validate_args)

    def _base(self) -> LogNormal:
        """Build the positive LogNormal component.

        Returns
        -------
        LogNormal
            LogNormal distribution with the retained log location and scale.
        """
        return LogNormal(self.loc, self.scale, validate_args=self._validate_args)


__all__ = ["HurdleGamma", "HurdleLogNormal"]
