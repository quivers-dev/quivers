"""Cutpoint-parameterised ordered-categorical morphisms.

Ordered-categorical distributions assign probability mass to a
discrete set of ordered categories ``{0, 1, ..., K-1}`` via a latent
real-valued ``eta`` and ``K - 1`` cutpoints
``c_0 < c_1 < ... < c_{K-2}``. The cumulative probability of
landing at or below category ``k`` is ``CDF(c_k - eta)``, where
``CDF`` is either the logistic or the standard normal cumulative
distribution function:

    P(Y <= k | eta, c) = CDF(c_k - eta)         for 0 <= k < K - 1
    P(Y = k | eta, c) = CDF(c_k - eta) - CDF(c_{k-1} - eta)
    P(Y = 0 | eta, c) = CDF(c_0 - eta)
    P(Y = K - 1 | eta, c) = 1 - CDF(c_{K-2} - eta)

The ``cutpoints`` argument is supplied at call time as a 1-D tensor
of length ``K - 1``, allowing the same morphism to be reused with
different cutpoint vectors across observations or models.

Sampling is non-reparameterisable: the support is discrete.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.distributions import constraints as _constraints

from quivers.continuous._ordinal import OrdinalDistribution, OrdinalLink
from quivers.continuous.morphisms import AnySpace, ContinuousMorphism
from quivers.continuous.param_source import ParamSource, _make_source


class _ConditionalOrdered(ContinuousMorphism):
    """Shared machinery for cutpoint-parameterised ordered families."""

    mappable: bool = True

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        num_categories: int,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        if num_categories < 2:
            raise ValueError(
                f"ordered family requires num_categories >= 2; got {num_categories}"
            )
        super().__init__(domain, codomain)
        self._k = int(num_categories)
        self.param_source = _make_source(
            domain,
            1,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.integer_interval(0, self._k - 1)

    def _get_eta(self, x: torch.Tensor) -> torch.Tensor:
        return self.param_source(x).squeeze(-1)

    def _validate_cutpoints(self, cutpoints: torch.Tensor) -> torch.Tensor:
        if cutpoints.dim() != 1:
            raise ValueError(
                f"ordered family: cutpoints must be 1-D; got shape "
                f"{tuple(cutpoints.shape)}"
            )
        if cutpoints.shape[0] != self._k - 1:
            raise ValueError(
                f"ordered family: cutpoints length {cutpoints.shape[0]} != "
                f"num_categories - 1 = {self._k - 1}"
            )
        return cutpoints

    _ordinal_link: OrdinalLink = "logit"

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cutpoints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Log-probability of ordered-category ``y`` given input ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape ``(batch,)`` for discrete domain or
            ``(batch, dim)`` for continuous domain.
        y : torch.Tensor
            Observed category indices in ``{0, ..., K - 1}``. Shape
            ``(batch,)``.
        cutpoints : torch.Tensor, optional
            Ordered cutpoints of shape ``(K - 1,)``. When omitted, an
            equally spaced default centred at zero is used so the
            morphism remains callable with the same interface as
            other discrete families.

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape ``(batch,)``.
        """
        eta = self._get_eta(x)
        cps = (
            self._default_cutpoints(eta)
            if cutpoints is None
            else self._validate_cutpoints(cutpoints)
        )
        return OrdinalDistribution(eta, cps, link=self._ordinal_link).log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__}.rsample is not supported: "
            "ordered-categorical sampling is not reparameterisable; "
            "use .sample() instead."
        )

    def sample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
        cutpoints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Non-reparameterised samples from the ordered-categorical.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sample_shape : torch.Size
            Additional leading sample dimensions.
        cutpoints : torch.Tensor, optional
            Ordered cutpoints of shape ``(K - 1,)``; defaults as in
            ``log_prob``.

        Returns
        -------
        torch.Tensor
            Sampled category indices in ``{0, ..., K - 1}``. Shape
            ``(*sample_shape, batch)``.
        """
        with torch.no_grad():
            eta = self._get_eta(x)
            cps = (
                self._default_cutpoints(eta)
                if cutpoints is None
                else self._validate_cutpoints(cutpoints)
            )
            return OrdinalDistribution(eta, cps, link=self._ordinal_link).sample(
                sample_shape
            )

    def _default_cutpoints(self, eta: torch.Tensor) -> torch.Tensor:
        # Equally spaced cutpoints around zero on the unit-scale
        # latent. ``K - 1`` cutpoints at ``(k + 1 - K / 2)`` for k in
        # 0..K-2 produces a symmetric layout.
        ks = torch.arange(self._k - 1, device=eta.device, dtype=eta.dtype)
        return ks - (self._k - 2) / 2.0


class ConditionalOrderedLogistic(_ConditionalOrdered):
    """Conditional ordered-logistic over ``num_categories`` ordered classes.

    The latent ``eta(x)`` is computed by the parameter source. Given
    a vector of ``num_categories - 1`` ordered cutpoints supplied at
    call time, the probability of category ``k`` is

        P(Y = k) = sigmoid(c_k - eta) - sigmoid(c_{k-1} - eta)

    with the conventions ``sigmoid(c_{-1} - eta) = 0`` and
    ``sigmoid(c_{K-1} - eta) = 1``.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : SetObject or ContinuousSpace
        Target space; semantically the set of ordered categories.
    num_categories : int
        Number of ordered categories ``K >= 2``.
    hidden_dim : int
        Hidden layer width for the neural parameter source.
    """

    _ordinal_link: OrdinalLink = "logit"


class ConditionalOrderedProbit(_ConditionalOrdered):
    """Conditional ordered-probit over ``num_categories`` ordered classes.

    Identical to :class:`ConditionalOrderedLogistic` but uses the
    standard normal cumulative distribution function in place of the
    logistic CDF.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : SetObject or ContinuousSpace
        Target space; semantically the set of ordered categories.
    num_categories : int
        Number of ordered categories ``K >= 2``.
    hidden_dim : int
        Hidden layer width for the neural parameter source.
    """

    _ordinal_link: OrdinalLink = "probit"


__all__ = [
    "ConditionalOrderedLogistic",
    "ConditionalOrderedProbit",
]
