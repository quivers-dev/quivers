"""Shared underflow-safe Gamma weights for compound count sampling."""

import torch
from torch import Tensor


def _dirichlet_weights(concentration: Tensor) -> Tensor:
    """Draw normalized log-gamma weights without small-shape gamma underflow.

    Parameters
    ----------
    concentration : Tensor
        Positive Float64 Dirichlet concentrations with trailing event axes.

    Returns
    -------
    Tensor
        Probability vectors on the input device.

    Raises
    ------
    RuntimeError
        If the underlying augmented gamma sampler returns invalid values.
    """
    small = concentration < 1
    augmented = torch.where(small, concentration + 1, concentration)
    gamma = torch.distributions.Gamma(augmented, torch.ones_like(augmented)).sample()
    if not bool((torch.isfinite(gamma) & (gamma > 0)).all()):
        raise RuntimeError("Dirichlet count sampling exceeded finite gamma capacity")
    exponential = torch.empty_like(concentration).exponential_()
    log_gamma = gamma.log()
    logs = log_gamma - torch.where(small, exponential / concentration, 0.0)
    missing = ~torch.isfinite(logs).any(-1, keepdim=True)
    # If every log gamma overflowed negatively, all shapes are subnormal.
    # Compare their finite scaled logs first; ordinary rows retain full
    # precision even when they also contain a tiny concentration.
    scale = concentration.amin(-1, keepdim=True).clamp_max(1)
    scaled = scale * log_gamma - torch.where(
        small, exponential * (scale / concentration), 0.0
    )
    centered = (scaled - scaled.amax(-1, keepdim=True)) / scale
    safe_logs = torch.where(missing, centered, logs)
    return torch.softmax(safe_logs, -1)


__all__: list[str] = []
