"""Closed-probability Bernoulli scores shared with standalone targets."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _qvr_bernoulli_log_score(
    probs: "torch.Tensor", value: "torch.Tensor"
) -> "torch.Tensor":
    """Score the selected atom with its original probability.

    Parameters
    ----------
    probs : torch.Tensor
        Success probabilities in the closed unit interval.
    value : torch.Tensor
        Outcomes broadcastable against the probabilities.

    Returns
    -------
    torch.Tensor
        Log mass in the probability tensor's dtype. An impossible or
        out-of-support outcome has negative infinite log mass and zero
        local score slope. A possible endpoint retains its slope.
    """
    import torch

    probs, value = torch.broadcast_tensors(probs, value)
    success = (value == 1) & (probs > 0)
    failure = (value == 0) & (probs < 1)
    log_p = torch.where(success, probs, torch.ones_like(probs)).log()
    log_q = (-torch.where(failure, probs, torch.zeros_like(probs))).log1p()
    return torch.where(success | failure, log_p + log_q, -torch.inf)
