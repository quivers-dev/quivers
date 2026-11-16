"""An exact zero atom combined with an asymmetric Laplace component."""

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from quivers.continuous._asymmetric_laplace import AsymmetricLaplace
from quivers.continuous._shape_families import _check_probability
from quivers.continuous._zero_inflated_count import _zero_atom_log_mass
from quivers.continuous._zip_hurdle import _OpenUnitInterval


class ZeroInflatedAsymmetricLaplace(Distribution):
    """Structural zeros and a continuous asymmetric Laplace response.

    Parameters
    ----------
    zero_prob : Tensor or float
        Probability of the atom at zero, including zero and one.
    loc : Tensor or float
        Quantile location of the continuous component.
    scale : Tensor or float
        Positive component scale in the check-loss parameterization.
    quantile : Tensor or float
        Component quantile strictly between zero and one.
    validate_args : bool or None
        Whether to validate distribution parameters.

    Notes
    -----
    Density is relative to the sum of a zero atom and Lebesgue measure:
    the continuous density does not contribute to the mass at zero.
    Sampling selects the mixture branch before sampling its component.
    """

    arg_constraints = {
        "zero_prob": constraints.unit_interval,
        "loc": constraints.real,
        "scale": constraints.positive,
        "quantile": _OpenUnitInterval(),
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, zero_prob, loc, scale, quantile, validate_args=None):
        self.zero_prob, self.loc, self.scale, self.quantile = broadcast_all(
            zero_prob, loc, scale, quantile
        )
        super().__init__(self.loc.shape, validate_args=validate_args)

    def _component(self):
        """Construct the broadcast continuous component.

        Returns
        -------
        AsymmetricLaplace
            Component law sharing the parameter tensors.
        """
        return AsymmetricLaplace(
            self.loc, self.scale, self.quantile, validate_args=False
        )

    def expand(self, batch_shape, _instance=None):
        """Expand parameter views over a new batch.

        Parameters
        ----------
        batch_shape : torch.Size
            Target broadcast batch dimensions.
        _instance : Distribution or None
            Optional reusable distribution instance.

        Returns
        -------
        ZeroInflatedAsymmetricLaplace
            Distribution with expanded parameter views.
        """
        new = self._get_checked_instance(ZeroInflatedAsymmetricLaplace, _instance)
        shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(shape))
        super(ZeroInflatedAsymmetricLaplace, new).__init__(shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self):
        """Expected mixed response.

        Returns
        -------
        Tensor
            Component mean multiplied by its mixture probability.
        """
        return (1 - self.zero_prob) * self._component().mean

    @property
    def variance(self):
        """Variance including uncertainty about the mixture branch.

        Returns
        -------
        Tensor
            Component variance and between-component mean contribution.
        """
        base = self._component()
        return (1 - self.zero_prob) * (
            base.variance + self.zero_prob * base.mean.square()
        )

    def log_prob(self, value):
        """Evaluate the zero atom or continuous component density.

        Parameters
        ----------
        value : Tensor
            Real responses, broadcastable with the parameter batch.

        Returns
        -------
        Tensor
            Log atom mass at zero and log Lebesgue density elsewhere.
        """
        finite = torch.isfinite(value)
        atom = value == 0
        point = torch.where(finite, value, 0.0)
        probability = torch.where(atom & finite, self.zero_prob, 1.0)
        at_zero = probability.log()
        ordinary_probability = torch.where(~atom & finite, self.zero_prob, 0.0)
        ordinary = torch.log1p(-ordinary_probability) + self._component().log_prob(
            point
        )
        return torch.where(finite, torch.where(atom, at_zero, ordinary), -torch.inf)

    def _log_tails(self, value):
        """Evaluate complementary tails with the correct zero-atom boundary.

        Parameters
        ----------
        value : Tensor
            Real evaluation points, including infinite bounds.

        Returns
        -------
        tuple[Tensor, Tensor]
            Log cumulative and strict survival probabilities.
        """
        base = self._component()
        base_cdf = base.log_cdf(value)
        base_sf = base.log_survival_function(value)
        below = value < 0
        left_probability = torch.where(below, self.zero_prob, 0.0)
        right_probability = torch.where(~below, self.zero_prob, 0.0)
        log_cdf = torch.where(
            below,
            torch.log1p(-left_probability) + base_cdf,
            _zero_atom_log_mass(self.zero_prob, base_cdf),
        )
        log_sf = torch.where(
            below,
            _zero_atom_log_mass(self.zero_prob, base_sf),
            torch.log1p(-right_probability) + base_sf,
        )
        log_cdf = torch.where(value == -torch.inf, -torch.inf, log_cdf)
        log_cdf = torch.where(value == torch.inf, 0.0, log_cdf)
        log_sf = torch.where(value == -torch.inf, 0.0, log_sf)
        log_sf = torch.where(value == torch.inf, -torch.inf, log_sf)
        return log_cdf, log_sf

    def log_cdf(self, value):
        """Evaluate log probability at or below a response.

        Parameters
        ----------
        value : Tensor
            Real response bounds.

        Returns
        -------
        Tensor
            Log cumulative probabilities, including the atom at zero.
        """
        return self._log_tails(value)[0]

    def log_survival_function(self, value):
        """Evaluate log probability strictly above a response.

        Parameters
        ----------
        value : Tensor
            Real response bounds.

        Returns
        -------
        Tensor
            Log survival probabilities, excluding the atom at zero.
        """
        return self._log_tails(value)[1]

    def cdf(self, value):
        """Evaluate cumulative probabilities.

        Parameters
        ----------
        value : Tensor
            Real response bounds.

        Returns
        -------
        Tensor
            Cumulative probabilities.
        """
        return (1 - self.zero_prob) * self._component().cdf(value) + self.zero_prob * (
            value >= 0
        ).to(self.loc.dtype)

    def icdf(self, value):
        """Invert the continuous tails and the jump at zero.

        Parameters
        ----------
        value : Tensor
            Probabilities in the closed unit interval.

        Returns
        -------
        Tensor
            Generalized quantiles, with zero throughout the atom's jump.
        """
        if self._validate_args:
            _check_probability(value)
        value, zero_prob = torch.broadcast_tensors(value, self.zero_prob)
        base = self._component()
        component_mass = 1 - zero_prob
        left = component_mass * base.cdf(torch.zeros_like(value))
        right = left + zero_prob
        atom = (value >= left) & (value <= right)
        degenerate = zero_prob == 1
        denominator = torch.where(degenerate, 1.0, component_mass)
        transformed = torch.where(value <= left, value, value - zero_prob) / denominator
        endpoint = (value == 0) | (value == 1)
        transformed = torch.where(atom | degenerate | endpoint, 0.5, transformed)
        result = torch.where(atom | degenerate, 0.0, base.icdf(transformed))
        result = torch.where((value == 0) & ~degenerate, -torch.inf, result)
        return torch.where((value == 1) & ~degenerate, torch.inf, result)

    def sample(self, sample_shape=torch.Size()):
        """Select branches before sampling active continuous responses.

        Parameters
        ----------
        sample_shape : torch.Size
            Dimensions prepended to the parameter batch.

        Returns
        -------
        Tensor
            Floating responses, including exact structural zeros.
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            active = torch.rand(
                shape, dtype=self.loc.dtype, device=self.loc.device
            ) >= self.zero_prob.expand(shape)
            result = torch.zeros(shape, dtype=self.loc.dtype, device=self.loc.device)
            if bool(active.any()):
                component = AsymmetricLaplace(
                    self.loc.expand(shape)[active],
                    self.scale.expand(shape)[active],
                    self.quantile.expand(shape)[active],
                    validate_args=False,
                )
                result[active] = component.sample()
            return result


__all__ = ["ZeroInflatedAsymmetricLaplace"]
