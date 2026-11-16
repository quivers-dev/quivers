"""Stable raw mixture sums with closed-boundary weight derivatives."""

from __future__ import annotations

import torch
from torch import Tensor


def _partials(weights, scores, result):
    """Evaluate analytical partial derivatives, including zero-weight cells.

    Parameters
    ----------
    weights, scores : Tensor
        Broadcast component weights and log scores.
    result : Tensor
        Reduced logarithmic weighted sum.

    Returns
    -------
    tuple[Tensor, Tensor]
        Weight and score partial derivatives, respectively.
    """
    finite = torch.isfinite(result).unsqueeze(-1)
    center = torch.where(finite, result.unsqueeze(-1), 0.0)
    exponent = torch.where(finite, scores - center, 0.0)
    ratio = torch.where(finite, exponent.exp(), 0.0)
    positive = weights > 0
    log_weight = torch.where(positive, weights, 1.0).log()
    selected = finite & positive
    posterior = torch.where(
        selected, torch.where(selected, log_weight + scores - center, 0.0).exp(), 0.0
    )
    # Ordinary multiplication preserves the mixed weight/score derivative at
    # weight zero. Only genuinely unrepresentable ratios need a safe zero;
    # the weight derivative itself remains infinite in that case.
    zero_ratio = torch.where(torch.isfinite(ratio), ratio, 0.0)
    score_partial = torch.where(positive, posterior, weights * zero_ratio)
    return ratio, score_partial


def _multiply(tangent, partial):
    """Multiply derivative terms without producing zero times infinity.

    Parameters
    ----------
    tangent, partial : Tensor
        Tangent or upstream derivative and an analytical partial.

    Returns
    -------
    Tensor
        Product, exactly zero where the tangent is zero.
    """
    return tangent * torch.where(tangent == 0, 0.0, partial)


class _WeightedLogSumExp(torch.autograd.Function):
    """Analytical reverse and forward AD for a weighted log-sum."""

    generate_vmap_rule = True

    @staticmethod
    def forward(weights, scores):
        """Evaluate the sum using positive component weights as the pivot.

        Parameters
        ----------
        weights, scores : Tensor
            Broadcast weights and component log scores.

        Returns
        -------
        Tensor
            Logarithm of the raw weighted sum.
        """
        positive = weights > 0
        terms = torch.where(
            positive, torch.where(positive, weights, 1.0).log() + scores, -torch.inf
        )
        return torch.logsumexp(terms, -1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Retain original coordinates and output for higher-order derivatives.

        Parameters
        ----------
        ctx : object
            Torch automatic-differentiation context.
        inputs : tuple[Tensor, Tensor]
            Original component coordinates.
        output : Tensor
            Reduced logarithmic weighted sum.
        """
        ctx.save_for_backward(*inputs, output)
        ctx.save_for_forward(*inputs, output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Return analytical partials without differentiating log zero.

        Parameters
        ----------
        ctx : object
            Torch automatic-differentiation context.
        grad_outputs : tuple[Tensor]
            Upstream reduced derivative.

        Returns
        -------
        tuple[Tensor, Tensor]
            Weight and log-score derivatives.
        """
        weights, scores, result = ctx.saved_tensors
        weight_partial, score_partial = _partials(weights, scores, result)
        (grad_output,) = grad_outputs
        upstream = grad_output.unsqueeze(-1)
        return _multiply(upstream, weight_partial), _multiply(upstream, score_partial)

    @staticmethod
    def jvp(ctx, *grad_inputs):
        """Apply the same analytical partials to forward-mode tangents.

        Parameters
        ----------
        ctx : object
            Torch automatic-differentiation context.
        grad_inputs : tuple[Tensor, Tensor]
            Weight and log-score input coordinate tangents.

        Returns
        -------
        Tensor
            Reduced directional derivative.
        """
        weights, scores, result = ctx.saved_tensors
        weight_partial, score_partial = _partials(weights, scores, result)
        grad_weights, grad_scores = grad_inputs
        return (
            _multiply(grad_weights, weight_partial)
            + _multiply(grad_scores, score_partial)
        ).sum(-1)


def weighted_logsumexp(weights: Tensor, scores: Tensor, dim: int = -1) -> Tensor:
    """Reduce raw nonnegative weights while preserving zero-boundary AD.

    Parameters
    ----------
    weights : Tensor
        Finite nonnegative raw weights, broadcastable against scores. Callers
        validate this parameter contract when constructing the mixture.
    scores : Tensor
        Component log scores. Negative infinity denotes a zero component mass.
    dim : int
        Component axis in the broadcast inputs.

    Returns
    -------
    Tensor
        Logarithm of sum(weights*exp(scores)). Finite results preserve one-sided
        weight derivatives at zero and reverse/forward higher-order AD. A
        zero total returns minus infinity with inactive derivatives set to zero.
        Derivatives exceeding the floating dtype's range remain infinite.
    """
    weights, scores = torch.broadcast_tensors(weights, scores)
    return _WeightedLogSumExp.apply(weights.movedim(dim, -1), scores.movedim(dim, -1))
