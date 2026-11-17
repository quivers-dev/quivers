"""Regression diagnostics from retained conditional response moments."""

from __future__ import annotations

import torch

from quivers.diagnostics._numeric import real_samples


def bayesian_r2(
    expected: torch.Tensor,
    *,
    observed: torch.Tensor | None = None,
    residual_variance: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Compute a variance ratio for each retained posterior draw and event.

    Parameters
    ----------
    expected : torch.Tensor
        Conditional response means, shaped chain x draw x row x event axes.
    observed : torch.Tensor or None
        Responses shaped row x event axes. Their centered residual variance
        supplies the denominator's unexplained component.
    residual_variance : torch.Tensor or float or None
        Conditional response variances broadcastable to expected. Their mean
        over rows instead supplies the unexplained component. Exactly one
        of observed and residual_variance is required.

    Returns
    -------
    torch.Tensor
        Float64 ratios on chain, draw, and event axes. A constant prediction
        has ratio zero when unexplained variance is positive; both components
        being zero gives NaN because no response variation is defined.
        Values are detached, as with the other retained-sample diagnostics.

    Raises
    ------
    TypeError
        If moments or observations are not real tensors.
    ValueError
        If axes, finiteness, devices, integer capacity, or variances are invalid.

    Notes
    -----
    Prediction variance uses the unbiased row variance. The residual method
    uses the same row variance of observed minus predicted values; the model
    method uses the average conditional variance. A common scale prevents
    overflow in either variance and cancels from their ratio.
    """
    if (observed is None) == (residual_variance is None):
        raise ValueError("provide exactly one of observed and residual_variance")
    means = real_samples(expected, label="R-squared expected responses")
    if means.ndim < 3 or not all(means.shape) or means.shape[2] < 2:
        raise ValueError(
            "R-squared requires chain, draw, and at least two response rows"
        )
    scale = means.abs().amax(2, keepdim=True)
    response = None
    variances = None
    if observed is not None:
        response = real_samples(observed, label="R-squared observed responses")
        if response.shape != means.shape[2:] or response.device != means.device:
            raise ValueError(
                "R-squared responses must match the row/event axes and device"
            )
        scale = torch.maximum(scale, response.abs().amax(0, keepdim=True))
    else:
        assert residual_variance is not None
        raw = (
            residual_variance
            if isinstance(residual_variance, torch.Tensor)
            else torch.tensor(
                residual_variance, device=means.device, dtype=torch.float64
            )
        )
        variances = real_samples(raw, label="R-squared conditional variances")
        if variances.device != means.device or bool((variances < 0).any()):
            raise ValueError(
                "R-squared conditional variances must be nonnegative on the same device"
            )
        try:
            variances = variances.broadcast_to(means.shape)
        except RuntimeError as error:
            raise ValueError(
                "R-squared conditional variances must broadcast to response moments"
            ) from error
        scale = torch.maximum(scale, variances.sqrt().amax(2, keepdim=True))
    scale = torch.where(scale > 0, scale, 1.0)
    normalized = means / scale
    predicted_variance = (normalized - normalized.mean(2, keepdim=True)).square().sum(
        2
    ) / (means.shape[2] - 1)
    if response is not None:
        residuals = response / scale - normalized
        unexplained = (residuals - residuals.mean(2, keepdim=True)).square().sum(2) / (
            means.shape[2] - 1
        )
    else:
        assert variances is not None
        unexplained = (variances.sqrt() / scale).square().mean(2)
    total = predicted_variance + unexplained
    return torch.where(
        total > 0,
        predicted_variance / torch.where(total > 0, total, 1.0),
        torch.nan,
    )


__all__ = ["bayesian_r2"]
