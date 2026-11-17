"""Dtype-safe generalized Pareto tails with differentiable zero-tail limits."""

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


class GeneralizedPareto(Distribution):
    """Generalized Pareto law on a location and tail-dependent upper endpoint.

    Parameters
    ----------
    loc : torch.Tensor or float
        Lower endpoint.
    scale : torch.Tensor or float
        Strictly positive scale.
    concentration : torch.Tensor or float
        Real tail index; zero gives the exponential limit.
    validate_args : bool, optional
        Whether to validate parameters and observations.
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
        """Return the location and tail-dependent interval.

        Returns
        -------
        constraints.Constraint
            Closed lower bound and finite upper bound for negative tail indices.
        """
        negative = self.concentration < 0
        safe = torch.where(
            negative, self.concentration, -torch.ones_like(self.concentration)
        )
        upper = torch.where(negative, self.loc - self.scale / safe, torch.inf)
        return constraints.interval(self.loc, upper)

    def expand(self, batch_shape, _instance=None):
        """Expand parameter views without copying their storage.

        Parameters
        ----------
        batch_shape : torch.Size or Sequence[int]
            Requested batch dimensions.
        _instance : GeneralizedPareto or None
            Optional instance supplied by PyTorch.

        Returns
        -------
        GeneralizedPareto
            Expanded distribution.
        """
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(batch_shape))
        super(GeneralizedPareto, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _log_survival(self, value):
        """Compute the exponential-limit survival transformation.

        Parameters
        ----------
        value : torch.Tensor
            Observations on the parameter device.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Standardized values, interior mask, and interior log survival.
        """
        z = (value - self.loc) / self.scale
        inside = (z >= 0) & (1 + self.concentration * z > 0)
        safe_z = torch.where(inside, z, torch.zeros_like(z))
        t = self.concentration * safe_z
        near = t.abs() < 1e-4
        safe_t = torch.where(near, torch.ones_like(t), t)
        ratio = torch.where(
            near,
            1 - t / 2 + t.square() / 3 - t.pow(3) / 4 + t.pow(4) / 5 - t.pow(5) / 6,
            torch.log1p(safe_t) / safe_t,
        )
        result = -safe_z * ratio
        return z, inside, result

    def log_prob(self, value):
        """Evaluate the density and finite-endpoint limits.

        Parameters
        ----------
        value : torch.Tensor
            Real observations.

        Returns
        -------
        torch.Tensor
            Broadcast log densities.
        """
        if self._validate_args:
            self._validate_sample(value)
        z, inside, log_survival = self._log_survival(value)
        result = -self.scale.log() + (1 + self.concentration) * log_survival
        endpoint = (
            (self.concentration < 0) & (1 + self.concentration * z == 0) & (z >= 0)
        )
        limit = torch.where(
            self.concentration == -1,
            -self.scale.log(),
            torch.where(self.concentration < -1, torch.inf, -torch.inf),
        )
        return torch.where(inside, result, torch.where(endpoint, limit, -torch.inf))

    def log_survival_function(self, value):
        """Evaluate log survival over the entire real line.

        Parameters
        ----------
        value : torch.Tensor
            Real observations.

        Returns
        -------
        torch.Tensor
            Zero below the lower endpoint and negative infinity above the upper endpoint.
        """
        z, inside, result = self._log_survival(value)
        return torch.where(z < 0, 0.0, torch.where(inside, result, -torch.inf))

    def cdf(self, value):
        """Evaluate cumulative probability.

        Parameters
        ----------
        value : torch.Tensor
            Real observations.

        Returns
        -------
        torch.Tensor
            Broadcast cumulative probabilities.
        """
        return -torch.expm1(self.log_survival_function(value))

    def icdf(self, value):
        """Transform probabilities through the tail quantile.

        Parameters
        ----------
        value : torch.Tensor
            Probabilities in the closed unit interval.

        Returns
        -------
        torch.Tensor
            Quantiles, including both support endpoints.
        """
        upper = value == 1
        safe_value = torch.where(upper, torch.zeros_like(value), value)
        t = -torch.log1p(-safe_value)
        x = self.concentration * t
        near = x.abs() < 1e-4
        safe = torch.where(near, torch.ones_like(x), x)
        ratio = torch.where(
            near,
            1
            + x / 2
            + x.square() / 6
            + x.pow(3) / 24
            + x.pow(4) / 120
            + x.pow(5) / 720,
            torch.expm1(safe) / safe,
        )
        result = self.loc + self.scale * t * ratio
        negative = self.concentration < 0
        safe_c = torch.where(
            negative, self.concentration, -torch.ones_like(self.concentration)
        )
        endpoint = torch.where(negative, self.loc - self.scale / safe_c, torch.inf)
        return torch.where(upper, endpoint, result)

    def rsample(self, sample_shape=torch.Size()):
        """Draw observations with pathwise gradients through every parameter.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples with the parameter dtype and device.
        """
        u = torch.rand(
            self._extended_shape(sample_shape),
            dtype=self.loc.dtype,
            device=self.loc.device,
        )
        return self.icdf(u)

    @property
    def mean(self):
        """Return the expectation where the first moment exists.

        Returns
        -------
        torch.Tensor
            Mean for concentration below one; NaN otherwise.
        """
        valid = self.concentration < 1
        safe = torch.where(valid, self.concentration, 0.0)
        return torch.where(valid, self.loc + self.scale / (1 - safe), torch.nan)

    @property
    def variance(self):
        """Return the variance where the second moment exists.

        Returns
        -------
        torch.Tensor
            Variance for concentration below one half; NaN otherwise.
        """
        valid = self.concentration < 0.5
        safe = torch.where(valid, self.concentration, 0.0)
        return torch.where(
            valid,
            self.scale.square() / ((1 - safe).square() * (1 - 2 * safe)),
            torch.nan,
        )


__all__: list[str] = []
