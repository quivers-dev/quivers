"""Power-likelihood scoring over an observation's own rows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import torch

from quivers._positions import demand
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.plate import VectorisedObserve


class _RowDensity(Protocol):
    """Row-scoring interface implemented by observation plate wrappers."""

    def log_prob_pointwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

    def log_prob_pointwise_rows(
        self, x: torch.Tensor, y: torch.Tensor, *rows: torch.Tensor
    ) -> torch.Tensor: ...

    def log_prob_rows(
        self, x: torch.Tensor, y: torch.Tensor, *rows: torch.Tensor
    ) -> torch.Tensor: ...


def checked_weights(
    value: object, input: torch.Tensor, rows: int | None
) -> torch.Tensor:
    """Validate a scalar weight or one real weight per observation row."""
    weights = torch.as_tensor(value, device=input.device)
    if not weights.is_floating_point() or weights.is_complex():
        raise TypeError("observation weights require real-valued floating data")
    demand(
        torch.isfinite(weights) & (weights >= 0),
        lambda: ValueError("observation weights must be finite and nonnegative"),
    )
    if weights.numel() == 1:
        return weights.reshape(())
    if rows is None or tuple(weights.shape) != (rows,):
        expected = "a scalar" if rows is None else f"a scalar or shape ({rows},)"
        raise ValueError(
            f"observation weights require {expected}, got {tuple(weights.shape)}"
        )
    return weights


class WeightedObservation(ContinuousMorphism):
    """Score active rows of an observation while preserving its sampling law."""

    def __init__(
        self,
        inner: ContinuousMorphism,
        declared: ContinuousMorphism,
        weights: object,
        input: torch.Tensor,
    ) -> None:
        super().__init__(inner.domain, inner.codomain)
        self._inner = inner
        self._plated = isinstance(declared, VectorisedObserve)
        self._rows: int | None = None
        self._array_shape: tuple[int, ...] = ()
        if isinstance(declared, VectorisedObserve):
            self._rows = int(declared.response.shape[0])
            self._array_shape = declared.array_shape
        self._weights = checked_weights(weights, input, self._rows)

    @property
    def _row_density(self) -> _RowDensity:
        """View the inner morphism through its plate-scoring interface."""
        return cast(_RowDensity, self._inner)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Draw from the unweighted response distribution."""
        return self._inner.rsample(x, sample_shape)

    def _zero(self, x: torch.Tensor) -> torch.Tensor:
        """Keep a zero score connected to its differentiable inputs."""
        value = (x * 0.0).sum() + (self._weights * 0.0).sum()
        for parameter in self._inner.parameters():
            value = value + (parameter * 0.0).sum()
        dependencies = getattr(self._inner, "score_dependencies", None)
        if callable(dependencies):
            for parameter in cast(
                Callable[[], tuple[torch.Tensor, ...]], dependencies
            )():
                value = value + (parameter * 0.0).sum()
        return value

    def _score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        kept: torch.Tensor | None,
        pointwise: bool = False,
    ) -> torch.Tensor:
        weights = self._weights
        if weights.ndim == 0:
            if not bool(weights > 0):
                zero = self._zero(x)
                if not pointwise:
                    return zero
                assert self._rows is not None
                return zero.expand(self._rows if kept is None else kept.numel())
            if kept is None:
                method = (
                    self._row_density.log_prob_pointwise
                    if pointwise
                    else self._inner.log_prob
                )
                return weights * method(x, y)
            arrays = tuple(
                torch.arange(n, device=kept.device) for n in self._array_shape
            )
            method = (
                self._row_density.log_prob_pointwise_rows
                if pointwise
                else self._row_density.log_prob_rows
            )
            return weights * method(x, y, kept, *arrays)
        full_rows = kept is None
        if kept is None:
            kept = torch.arange(weights.numel(), device=weights.device)
        selected = weights.index_select(0, kept.to(weights.device))
        active = torch.nonzero(selected > 0, as_tuple=False).flatten()
        if active.numel() == 0:
            zero = self._zero(x)
            return zero.expand(kept.numel()) if pointwise else zero
        if active.numel() == selected.numel():
            arrays = tuple(
                torch.arange(n, device=kept.device) for n in self._array_shape
            )
            density = (
                self._row_density.log_prob_pointwise(x, y)
                if full_rows
                else self._row_density.log_prob_pointwise_rows(x, y, kept, *arrays)
            )
            weighted = selected.to(density.device) * density
            return weighted if pointwise else weighted.sum()
        active_rows = kept.index_select(0, active.to(kept.device))
        arrays = tuple(torch.arange(n, device=kept.device) for n in self._array_shape)
        # Full-row and already-subsampled responses share the same row map.
        # Once zero-weight rows are removed, already-kept data must follow it.
        if kept.numel() != self._rows:
            if int(y.shape[0]) == kept.numel():
                y = y.index_select(0, active.to(y.device))
            if x.ndim and int(x.shape[0]) == kept.numel() and x.shape[0] != 1:
                x = x.index_select(0, active.to(x.device))
        density = self._row_density.log_prob_pointwise_rows(x, y, active_rows, *arrays)
        weighted = selected.index_select(0, active).to(density.device) * density
        if pointwise:
            return weighted.new_zeros(kept.numel()).index_copy(0, active, weighted)
        return weighted.sum()

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Weight each active row before reducing the observation plate."""
        return self._score(x, y, None)

    def log_prob_pointwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Keep each weighted row separate for an enclosing marginalization."""
        return self._score(x, y, None, pointwise=self._plated)

    def log_prob_rows(
        self, x: torch.Tensor, y: torch.Tensor, *rows: torch.Tensor
    ) -> torch.Tensor:
        """Score the selected rows before an outer subsampling correction."""
        if not self._plated or len(rows) != 1 + len(self._array_shape):
            raise ValueError("weighted observations require one index per plate axis")
        for index, extent in zip(rows[1:], self._array_shape, strict=True):
            if not torch.equal(index.cpu(), torch.arange(extent)):
                raise ValueError("subsample observation rows, not their array entries")
        return self._score(x, y, rows[0])
