"""Shared underflow-safe Gamma weights for compound count sampling."""

import torch
from torch import Tensor


def _dirichlet_log_weights(concentration: Tensor, *, reparameterized=False) -> Tensor:
    """Draw normalized log-Gamma weights before probability underflow.

    Parameters
    ----------
    concentration : Tensor
        Positive Float64 Dirichlet concentrations with trailing event axes.
    reparameterized : bool
        Use pathwise augmented-Gamma draws when true.

    Returns
    -------
    Tensor
        Log probability vectors on the input device.

    Raises
    ------
    RuntimeError
        If the augmented Gamma sampler returns invalid values or pathwise
        shape-gradient denominators exceed floating capacity.
    """
    if reparameterized and bool(
        (concentration < torch.finfo(concentration.dtype).tiny ** 0.5).any()
    ):
        raise RuntimeError(
            "pathwise Dirichlet sampling exceeded Float64 shape-gradient capacity"
        )
    small = concentration < 1
    augmented = torch.where(small, concentration + 1, concentration)
    distribution = torch.distributions.Gamma(augmented, torch.ones_like(augmented))
    gamma = distribution.rsample() if reparameterized else distribution.sample()
    if not bool((torch.isfinite(gamma) & (gamma > 0)).all()):
        raise RuntimeError("Dirichlet count sampling exceeded finite gamma capacity")
    exponential = torch.empty_like(concentration).exponential_()
    log_gamma = gamma.log()
    logs = log_gamma - torch.where(small, exponential / concentration, 0.0)
    missing = ~torch.isfinite(logs).any(-1, keepdim=True)
    if reparameterized and bool(missing.any()):
        raise RuntimeError(
            "pathwise Dirichlet sampling exceeded finite log-Gamma capacity"
        )
    # If every log gamma overflowed negatively, all shapes are subnormal.
    # Compare their finite scaled logs first; ordinary rows retain full
    # precision even when they also contain a tiny concentration.
    scale = concentration.amin(-1, keepdim=True).clamp_max(1)
    scaled = scale * log_gamma - torch.where(
        small, exponential * (scale / concentration), 0.0
    )
    safe_scale = torch.where(missing, scale, 1.0)
    difference = torch.where(missing, scaled - scaled.amax(-1, keepdim=True), 0.0)
    centered = difference / safe_scale
    safe_logs = torch.where(missing, centered, logs)
    return torch.log_softmax(safe_logs, -1)


def _dirichlet_weights(concentration: Tensor) -> Tensor:
    """Draw normalized Gamma probabilities through the shared log sampler.

    Parameters
    ----------
    concentration : Tensor
        Positive concentrations with a trailing category axis.

    Returns
    -------
    Tensor
        Probability vectors retaining the established Gamma/uniform RNG order.
    """
    return _dirichlet_log_weights(concentration).exp()


__all__: list[str] = []
