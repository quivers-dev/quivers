"""Shared probability and log-tail interface for scalar counting laws."""

import torch
from torch.distributions import Distribution


def probability_tails(log_cdf, log_sf):
    """Project log tails while retaining finite derivatives at exact zero mass.

    Parameters
    ----------
    log_cdf, log_sf : Tensor
        Complementary logarithmic tail probabilities.

    Returns
    -------
    tuple[Tensor, Tensor]
        Inclusive cumulative and strict survival probabilities. Exact zero
        masses use their complementary finite log coordinate, preserving
        finite one-sided derivatives when log zero itself has no derivative.
    """
    cdf = torch.where(torch.isneginf(log_cdf), -torch.expm1(log_sf), log_cdf.exp())
    sf = torch.where(torch.isneginf(log_sf), -torch.expm1(log_cdf), log_sf.exp())
    return cdf, sf


class _CountCDF(Distribution):
    """Supply one common scalar count CDF interface from a paired log kernel."""

    def expand(self, batch_shape, _instance=None):
        """Expand all count laws through one declared constructor-rank path.

        Parameters
        ----------
        batch_shape : torch.Size
            Requested native batch axes.
        _instance : Distribution or None
            Optional native expansion instance; reconstruction preserves type.

        Returns
        -------
        Distribution
            Law retaining original probability coordinates and resource policies.
        """
        from quivers.continuous._batch_selection import expand_registered_batch

        return expand_registered_batch(self, batch_shape)

    def _count_log_tails(self, value):
        """Return paired inclusive/strict log tails from the native family.

        Parameters
        ----------
        value : Tensor
            Real-valued thresholds.

        Returns
        -------
        tuple[Tensor, Tensor]
            Inclusive cumulative and strict survival logarithms.

        Raises
        ------
        NotImplementedError
            If a subclass does not provide its numerical tail kernel.
        """
        raise NotImplementedError

    def _count_probability_tails(self, value):
        """Project the shared log-tail kernel to ordinary probabilities.

        Parameters
        ----------
        value : Tensor
            Real-valued thresholds.

        Returns
        -------
        tuple[Tensor, Tensor]
            Inclusive cumulative and strict survival probabilities.
        """
        return probability_tails(*self._count_log_tails(value))

    def log_cdf(self, value):
        """Evaluate log P(Y<=value), using the floor of noninteger thresholds.

        Parameters
        ----------
        value : Tensor
            Real-valued thresholds, including infinities.

        Returns
        -------
        Tensor
            Inclusive log cumulative probability.
        """
        return self._count_log_tails(value)[0]

    def log_survival_function(self, value):
        """Evaluate log P(Y>value), retaining small complementary tails.

        Parameters
        ----------
        value : Tensor
            Real-valued thresholds, including infinities.

        Returns
        -------
        Tensor
            Strict log survival probability.
        """
        return self._count_log_tails(value)[1]

    def cdf(self, value):
        """Evaluate inclusive cumulative probability from the paired kernel.

        Parameters
        ----------
        value : Tensor
            Real-valued thresholds, including infinities.

        Returns
        -------
        Tensor
            Inclusive cumulative probability.
        """
        return self._count_probability_tails(value)[0]

    def survival_function(self, value):
        """Evaluate strict survival probability from the paired kernel.

        Parameters
        ----------
        value : Tensor
            Real-valued thresholds, including infinities.

        Returns
        -------
        Tensor
            Strict survival probability.
        """
        return self._count_probability_tails(value)[1]
