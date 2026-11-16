"""Shared probability primitives for standalone pyro runtime generation."""

# ruff: noqa: F821
# pyright: reportUndefinedVariable=false

import torch


def _qvr_log_probability(value):
    value = torch.as_tensor(value)
    if not value.is_floating_point() or not bool(
        (torch.isfinite(value) & (value >= 0) & (value <= 1)).all()
    ):
        raise ValueError("log_probability requires finite real values in [0, 1]")
    positive = value > 0
    safe = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, safe.log(), -torch.inf)


def _qvr_log1mexp(value):
    value = torch.as_tensor(value)
    if not value.is_floating_point() or not bool(
        (~torch.isnan(value) & (value <= 0)).all()
    ):
        raise ValueError("log1mexp requires nonpositive real values")
    interior = torch.isfinite(value) & (value < 0)
    safe = torch.where(interior, value, -torch.ones_like(value))
    near = safe > -0.6931471805599453
    near_arg = torch.where(near, safe, -torch.ones_like(safe))
    far_arg = torch.where(near, -torch.ones_like(safe), safe)
    result = torch.where(
        near, torch.log(-torch.expm1(near_arg)), torch.log1p(-far_arg.exp())
    )
    return torch.where(
        value == 0, -torch.inf, torch.where(value.isneginf(), 0.0, result)
    )


def _qvr_log_ndtr(value):
    value = torch.as_tensor(value)
    if not value.is_floating_point() or bool(torch.isnan(value).any()):
        raise ValueError("log_ndtr requires real values other than NaN")
    safe = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    result = torch.special.log_ndtr(safe)
    return torch.where(
        value.isneginf(), -torch.inf, torch.where(value.isposinf(), 0.0, result)
    )


def _qvr_positive_parameter(value):
    value = torch.as_tensor(value)
    if not value.is_floating_point() or not bool(
        (torch.isfinite(value) & (value > 0)).all()
    ):
        raise ValueError(
            "positive_parameter requires finite strictly positive real values"
        )
    return value


def _qvr_observation_status(value):
    """Validate integer recorded-event codes without changing their carrier.

    Parameters
    ----------
    value
        Scalar or lifted integer code tensor.

    Returns
    -------
    integer tensor
        Codes minus one through two, unchanged.

    Raises
    ------
    ValueError
        If the carrier is not integer or an active code is invalid.
    """
    value = torch.as_tensor(value)
    if (
        value.dtype == torch.bool
        or value.is_floating_point()
        or value.is_complex()
        or not bool(((value >= -1) & (value <= 2)).all())
    ):
        raise ValueError("observation_status requires integer codes -1, 0, 1, or 2")
    return value


def _qvr_interval_upper(lower, upper):
    """Check an active interval and retain its upper-endpoint derivatives.

    Parameters
    ----------
    lower, upper
        Real scalar or lifted endpoints; lower may be infinite.

    Returns
    -------
    real tensor
        Finite upper endpoint, strictly above lower.

    Raises
    ------
    ValueError
        If the endpoint carrier or active interval is invalid.
    """
    lower, upper = torch.as_tensor(lower), torch.as_tensor(upper)
    if (
        not lower.is_floating_point()
        or not upper.is_floating_point()
        or not bool((~lower.isnan() & upper.isfinite() & (upper > lower)).all())
    ):
        raise ValueError(
            "interval_upper requires a finite upper endpoint greater than its lower endpoint"
        )
    return upper
