"""Compatibility constructors for cumulative ordinal logit and probit distributions."""

from __future__ import annotations

from torch import Tensor

from quivers.continuous._ordinal import OrdinalDistribution, _log1mexp as _log1mexp


class OrderedLogistic(OrdinalDistribution):
    """Ordered-logit distribution over ``K = cutpoints.shape[-1] + 1``
    ordered categories indexed ``0 .. K - 1``.

    With a real predictor ``eta`` and a sorted cutpoint vector
    ``c = (c_0 < c_1 < ... < c_{K-2})``:

    * ``P(Y = 0)       = sigmoid(c_0 - eta)``
    * ``P(Y = k)       = sigmoid(c_k - eta) - sigmoid(c_{k-1} - eta)``
                          for ``0 < k < K - 1``
    * ``P(Y = K - 1)   = 1 - sigmoid(c_{K-2} - eta)``

    Cutpoint broadcasting:

    * Shared cutpoints ``c`` of shape ``(K - 1,)`` apply uniformly to
      every row.
    * Per-row cutpoints of shape ``(batch, K - 1)`` apply distinct
      thresholds per observation (the ordinal-mixed-model case).
    * Arbitrary leading batch shape is supported; the last axis is
      always the cutpoint axis.

    Log probabilities are computed without forming the difference of
    sigmoids: with ``a < b`` the two cutpoint offsets bounding a
    category, ``sigmoid(b) - sigmoid(a)`` equals
    ``sigmoid(b) * sigmoid(-a) * (1 - exp(a - b))``, so its log is a sum
    of two log-sigmoids and a ``log(1 - exp(.))`` term, each accurate in
    both tails. A category far from the predictor thus keeps its exact
    log probability rather than one floored at the smallest normal
    float.

    Parameters
    ----------
    predictor : Tensor
        Finite real predictor with arbitrary batch dimensions.
    cutpoints : Tensor
        Finite, strictly increasing thresholds with final dimension K-1.
    validate_args : bool or None
        Whether to validate arguments and response codes.

    References
    ----------
    [McCullagh 1980](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x).
    """

    _batch_parameter_ranks = {"predictor": 0, "cutpoints": 1}

    def __init__(
        self,
        predictor: Tensor,
        cutpoints: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        super().__init__(
            predictor, cutpoints, link="logit", validate_args=validate_args
        )


class OrderedProbit(OrderedLogistic):
    """Ordered-probit distribution over ``K = cutpoints.shape[-1] + 1``
    ordered categories indexed ``0 .. K - 1``.

    The ordered-logistic construction with standard normal latent noise:
    ``P(Y <= k) = Phi(c_k - eta)`` for the standard normal distribution
    function ``Phi``, so ``P(Y = k) = Phi(c_k - eta) - Phi(c_{k-1} -
    eta)``. Predictor and cutpoint broadcasting follow `OrderedLogistic`.

    Log probabilities are computed without forming the difference of
    distribution functions: with ``a < b`` the offsets bounding a
    category, ``log(Phi(b) - Phi(a)) = log Phi(b) + log(1 - exp(log
    Phi(a) - log Phi(b)))``, and for a category above the predictor
    (``a > 0``) the same identity on the mirrored offsets ``-b < -a``,
    since ``Phi(b) - Phi(a) = Phi(-a) - Phi(-b)``. Each term is accurate in
    both tails, so a category far from the predictor keeps its log
    probability.

    Parameters
    ----------
    predictor : Tensor
        The real predictor ``eta``.
    cutpoints : Tensor
        The sorted cutpoints, ``(..., K - 1)``.
    validate_args : bool | None
        Whether to validate arguments and values.
    """

    def __init__(
        self,
        predictor: Tensor,
        cutpoints: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        OrdinalDistribution.__init__(
            self, predictor, cutpoints, link="probit", validate_args=validate_args
        )


__all__ = ["OrderedLogistic", "OrderedProbit"]
