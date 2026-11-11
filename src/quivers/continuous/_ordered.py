"""Ordered-categorical distribution families that PyTorch does not
ship natively (`OrderedLogistic`, `OrderedProbit`).

The DSL surfaces these as inline call sites:

    observe y : Resp <- OrderedLogistic(eta, cutpoints)

with ``eta`` a real predictor (one row per observation) and
``cutpoints`` either a globally shared sorted vector of length
``K - 1`` or a per-row sorted matrix of shape ``(batch, K - 1)``.
Per-row cutpoints land in the ordinal-mixed-model setting where
every participant has distinct thresholds; the distribution's
broadcast rules handle both shapes uniformly.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import constraints as _constraints
from torch.distributions.distribution import Distribution


class OrderedLogistic(Distribution):
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

    Reference: [McCullagh 1980](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x).
    """

    arg_constraints = {
        "predictor": _constraints.real,
        "cutpoints": _constraints.real_vector,
    }
    has_rsample = False

    def __init__(
        self,
        predictor: Tensor,
        cutpoints: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        if cutpoints.dim() == 0:
            raise ValueError(
                "OrderedLogistic: `cutpoints` must have at least one "
                f"dimension carrying the K-1 thresholds, got "
                f"shape={tuple(cutpoints.shape)}"
            )
        if cutpoints.shape[-1] < 1:
            raise ValueError(
                "OrderedLogistic: `cutpoints` last dimension must "
                f"have size >= 1 (K >= 2 categories), got "
                f"shape={tuple(cutpoints.shape)}"
            )
        self.predictor = predictor
        self.cutpoints = cutpoints
        self._num_categories = int(cutpoints.shape[-1]) + 1
        batch_shape = torch.broadcast_shapes(predictor.shape, cutpoints.shape[:-1])
        super().__init__(
            batch_shape=batch_shape,
            event_shape=torch.Size(()),
            validate_args=validate_args,
        )

    @property
    def num_categories(self) -> int:
        return self._num_categories

    @_constraints.dependent_property
    def support(self):
        return _constraints.integer_interval(0, self._num_categories - 1)

    @property
    def mean(self) -> Tensor:
        weights = torch.arange(
            self._num_categories,
            device=self.predictor.device,
            dtype=self.predictor.dtype,
        )
        probs = self._category_probs()
        return (probs * weights).sum(dim=-1)

    @property
    def mode(self) -> Tensor:
        return self._category_probs().argmax(dim=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        log_probs = self._category_log_probs()
        idx = value.long()
        # The value and the distribution's batch broadcast against each
        # other: a predictor shared by every row scores a vector of
        # responses, and a per-row predictor scores one response each.
        shape = torch.broadcast_shapes(idx.shape, log_probs.shape[:-1])
        log_probs = log_probs.expand(*shape, self._num_categories)
        idx = idx.expand(shape).unsqueeze(-1)
        return log_probs.gather(-1, idx).squeeze(-1)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        # `torch.distributions.Distribution.sample` accepts any
        # `Sequence[int]` (tuple, list, or `torch.Size`); coerce
        # so callers writing ``.sample((200,))`` work uniformly.
        sample_shape = torch.Size(sample_shape)
        probs = self._category_probs()
        flat = probs.reshape(-1, self._num_categories)
        draws = torch.multinomial(
            flat,
            num_samples=max(1, sample_shape.numel()) if sample_shape else 1,
            replacement=True,
        )
        if not sample_shape:
            return draws[..., 0].reshape(self.batch_shape)
        return draws.t().reshape(*sample_shape, *self.batch_shape)

    def _category_log_probs(self) -> Tensor:
        """Log probability of every category.

        Returns
        -------
        Tensor
            Shape ``(*batch_shape, num_categories)``.
        """
        offsets = self.cutpoints - self.predictor.unsqueeze(-1)
        below = torch.full_like(offsets[..., :1], -torch.inf)
        above = torch.full_like(offsets[..., :1], torch.inf)
        bounds = torch.cat([below, offsets, above], dim=-1)
        lower = bounds[..., :-1]
        upper = bounds[..., 1:]
        # Out-of-order cutpoints give a category no mass.
        gap = (lower - upper).clamp(max=0.0)
        return (
            torch.nn.functional.logsigmoid(upper)
            + torch.nn.functional.logsigmoid(-lower)
            + _log1mexp(gap)
        )

    def _category_probs(self) -> Tensor:
        """Probability of every category.

        Returns
        -------
        Tensor
            Shape ``(*batch_shape, num_categories)``.
        """
        return self._category_log_probs().exp()


def _log1mexp(x: Tensor) -> Tensor:
    """``log(1 - exp(x))`` for ``x <= 0``, accurate across its range.

    Uses ``log(-expm1(x))`` near zero and ``log1p(-exp(x))`` below
    ``-log 2``, following [Maechler 2012](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf).

    Parameters
    ----------
    x : Tensor
        Non-positive values.

    Returns
    -------
    Tensor
        ``log(1 - exp(x))``, ``-inf`` at zero and zero at ``-inf``.
    """
    near_zero = x > -0.6931471805599453
    safe_near = torch.where(near_zero, x, torch.full_like(x, -1.0))
    safe_far = torch.where(near_zero, torch.full_like(x, -1.0), x)
    return torch.where(
        near_zero,
        torch.log(-torch.expm1(safe_near)),
        torch.log1p(-torch.exp(safe_far)),
    )


__all__ = ["OrderedLogistic"]
