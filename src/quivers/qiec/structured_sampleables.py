"""Finite sampleables whose points are closed discrete QIEC values."""

from __future__ import annotations

from collections.abc import Hashable
from functools import cached_property
from types import MappingProxyType

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.core.algebras import LOG_PROB
from quivers.qiec.checking import KernelError, KernelRegistry
from quivers.qiec.deduction_outcomes import _typed_key, validate_outcome_key
from quivers.qiec.distributions import (
    DistributionBackend,
    DistributionError,
    RuntimeDistribution,
    distribution_backend,
)
from quivers.qiec.evaluator import InvalidHandlerError, RuntimeConstructor
from quivers.qiec.numbers import NumberDomain, decide, host_index
from quivers.qiec.types import TypeApplication, TypeExpr


class FiniteSampleable(ValueModel):
    """A normalized distribution on explicit closed discrete QIEC values.

    Equal points share one mass, obtained by adding the masses of their
    occurrences. Zero-mass points are absent from enumeration and a
    typed point outside the positive support has log probability
    negative infinity. Generation draws once from the finite law.
    Enumeration and scoring share the same normalization. The log
    weights retain their gradient and are read afresh at every request.

    Parameters
    ----------
    element
        The closed discrete QIEC type of each point. Regular recursive
        families with ordinary constructors are admitted.
    values
        The candidate points as checked runtime values. Treat these
        discrete values as immutable for the lifetime of the sampleable.
    log_weights
        One unnormalized log mass for each candidate, in order. The
        tensor is retained by identity. No batch or plate axes are used.
    registry
        The checked declaration registry for the nominal types and constructors.
    backend
        The ordinary distribution backend used to draw a categorical index.
    """

    element: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    values: tuple[object, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_weights: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    registry: KernelRegistry = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    backend: DistributionBackend = dx.field(
        default_factory=distribution_backend, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @dx.model_validator()
    def _validate_points(self) -> None:
        """Check the finite carrier, support, and number of weights.

        Raises
        ------
        DistributionError
            If the type or candidate values are outside the admitted domain,
            or the weights have the wrong shape or no positive mass.
        """
        try:
            validate_outcome_key(self.element, self.registry)
        except KernelError as error:
            raise DistributionError(
                "finite sampleables require a closed discrete QIEC carrier"
            ) from error
        if (
            not isinstance(self.log_weights, torch.Tensor)
            or not self.log_weights.is_floating_point()
            or self.log_weights.shape != (len(self.values),)
            or not self.values
        ):
            raise DistributionError(
                "a finite sampleable needs one floating log weight per point"
            )
        self._validate_weights()
        # Building the support's equality classes also checks the live points.
        _ = self._groups

    def _validate_weights(self) -> None:
        """Check the current tensor without keeping a differentiable cache.

        Raises
        ------
        DistributionError
            If a log weight is invalid or the total mass is zero.
        """
        weights = self.log_weights.detach()
        if decide(
            (torch.isnan(weights) | torch.isposinf(weights)).any(), "finite log weights"
        ):
            raise DistributionError(
                "finite sampleable log weights contain NaN or +infinity"
            )
        if not decide(torch.isfinite(weights).any(), "finite sampleable mass"):
            raise DistributionError(
                "cannot normalize a finite sampleable without positive mass"
            )

    def _centered_weights(self) -> torch.Tensor:
        """Subtract a common offset before computing normalized masses.

        Returns
        -------
        torch.Tensor
            Current differentiable log weights, with the maximum weight
            fixed at zero for stable normalization.

        Raises
        ------
        DistributionError
            If the weights are invalid or their finite differences overflow.
        """
        self._validate_weights()
        weights = self.log_weights
        shifted = weights - weights.detach().amax()
        out_of_range = torch.isfinite(weights) & torch.isneginf(shifted)
        if decide(out_of_range.any(), "finite sampleable weight range"):
            raise DistributionError(
                "finite sampleable log weight difference is out of range"
            )
        return shifted

    @cached_property
    def _groups(self) -> MappingProxyType[Hashable, tuple[int, ...]]:
        """Index the immutable support by exact checked point equality.

        Returns
        -------
        Mapping
            The input positions for each distinct point, in first-occurrence order.
        """
        groups: dict[Hashable, list[int]] = {}
        for position, value in enumerate(self.values):
            groups.setdefault(self._key(value), []).append(position)
        return MappingProxyType(
            {key: tuple(indices) for key, indices in groups.items()}
        )

    def _key(self, value: object) -> Hashable:
        """Check the fields of one finite runtime value and obtain its key.

        Parameters
        ----------
        value
            A point represented in the machine's checked value layout.

        Returns
        -------
        Hashable
            Its exact value key.

        Raises
        ------
        DistributionError
            If a point violates the declarations of its checked type.
        """
        pending = [(value, self.element)]
        while pending:
            item, type_ = pending.pop()
            if not isinstance(item, RuntimeConstructor):
                try:
                    _typed_key(item, type_)
                except InvalidHandlerError as error:
                    raise DistributionError(
                        "invalid finite sampleable point"
                    ) from error
                continue
            constructor = self.registry.constructors.get(item.constructor)
            family = (
                self.registry.families.get(constructor.family)
                if constructor is not None
                else None
            )
            if (
                constructor is None
                or family is None
                or not isinstance(type_, TypeApplication)
                or family.type_constructor != type_.constructor
                or item.result_type != type_
                or item.static_arguments
                or len(constructor.fields) != len(item.fields)
            ):
                raise DistributionError("invalid finite sampleable constructor point")
            pending.extend(
                (child, field.type)
                for field, child in zip(constructor.fields, item.fields, strict=True)
            )
        return _typed_key(value, self.element)

    @property
    def numbers(self) -> NumberDomain:
        """The dtype and device shared by the law's log probabilities.

        Returns
        -------
        NumberDomain
            The number domain of the supplied log weights.
        """
        return NumberDomain.of(self.log_weights)

    def log_mass(self, batch_axes: int = 0) -> torch.Tensor:
        """Return the log total mass of the normalized law.

        Parameters
        ----------
        batch_axes
            Retained batch rank, which must be zero for this unbatched law.

        Returns
        -------
        torch.Tensor
            Scalar zero in the law's number domain.

        Raises
        ------
        DistributionError
            If a retained batch axis is requested or the live mass is invalid.
        """
        if isinstance(batch_axes, bool) or batch_axes != 0:
            raise DistributionError("finite sampleables do not have batch axes")
        self._centered_weights()
        return self.numbers.zero()

    def log_prob(self, value: object, keep_batch: bool = False) -> torch.Tensor:
        """Score one typed value under the current normalized law.

        Parameters
        ----------
        value
            A typed value of this law's element type.
        keep_batch
            Whether to keep independent batch axes. This law has no such axes.

        Returns
        -------
        torch.Tensor
            The scalar log probability, or negative infinity outside the support.
        """
        del keep_batch
        weights = self._centered_weights()
        key = self._key(value)
        group = self._groups.get(key)
        if group is None:
            return self.log_weights.new_full((), -torch.inf)
        return LOG_PROB.join(weights[list(group)], 0) - LOG_PROB.join(weights, 0)

    def weighted_support(self) -> tuple[tuple[object, torch.Tensor], ...]:
        """Return every positive-support value with normalized log probability.

        Returns
        -------
        tuple
            Distinct points and their scalar log probabilities, in
            first-occurrence order.
        """
        weights = self._centered_weights()
        normalizer = LOG_PROB.join(weights, 0)
        result = []
        for indices in self._groups.values():
            mass = LOG_PROB.join(weights[list(indices)], 0)
            if decide(torch.isfinite(mass), "finite sampleable support"):
                result.append((self.values[indices[0]], mass - normalizer))
        return tuple(result)

    def sample(self) -> object:
        """Draw one value with the ordinary categorical distribution backend.

        Returns
        -------
        object
            One point of the checked element type.
        """
        with torch.no_grad():
            support = self.weighted_support()
            distribution = RuntimeDistribution(
                backend=self.backend,
                numbers=self.numbers,
                family="Categorical",
                arguments={"logits": torch.stack([weight for _, weight in support])},
                ranks={"logits": 1},
            )
            index = host_index(distribution.sample(), "finite categorical draw")
        return support[index][0]


__all__ = ["FiniteSampleable"]
