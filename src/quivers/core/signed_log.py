"""A packed signed logarithmic carrier for first-order weight derivatives."""

from __future__ import annotations

from math import prod

import torch

from quivers.core.carriers import _Packed
from quivers.core.laws import Law, LawEvidence, Proved

__all__ = ["SignedLogDualSemiring"]


def _canonical(value: torch.Tensor) -> torch.Tensor:
    sign, log_abs = value.unbind(-1)
    return torch.where(
        sign == 0, torch.isneginf(log_abs), (sign.abs() == 1) & torch.isfinite(log_abs)
    )


def _check_pair(value: torch.Tensor) -> None:
    if value.dtype != torch.float64 or value.ndim == 0 or value.shape[-1] != 2:
        raise TypeError(
            "signed log pairs require a trailing axis of two float64 values"
        )
    if value.requires_grad:
        raise TypeError("pack the derivative as an explicit signed log tangent")
    if not bool(_canonical(value).all()):
        raise ValueError("signed log pairs require a sign and canonical log magnitude")


def _pair(sign: torch.Tensor, log_abs: torch.Tensor) -> torch.Tensor:
    if bool(((sign != 0) & ~torch.isfinite(log_abs)).any()):
        raise FloatingPointError("signed log magnitude exceeds float64 precision")
    return torch.stack((sign, torch.where(sign == 0, -torch.inf, log_abs)), dim=-1)


def _encode(value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
        raise ValueError("signed log embedding requires finite float64 values")
    return torch.stack((value.sign(), value.abs().log()), dim=-1)


def _decode(value: torch.Tensor) -> torch.Tensor:
    sign, log_abs = value.unbind(-1)
    magnitude = log_abs.exp()
    if bool((~torch.isfinite(magnitude) | ((magnitude == 0) & (sign != 0))).any()):
        raise FloatingPointError(
            "signed log value cannot be represented in linear float64"
        )
    return sign * magnitude


def _multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_sign, left_log = left.unbind(-1)
    right_sign, right_log = right.unbind(-1)
    return _pair(left_sign * right_sign, left_log + right_log)


def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_sign, left_log = left.unbind(-1)
    right_sign, right_log = right.unbind(-1)
    greater = left_log >= right_log
    high = torch.maximum(left_log, right_log)
    low = torch.minimum(left_log, right_log)
    opposite = left_sign * right_sign < 0
    cancels = opposite & (left_log == right_log)
    difference = torch.where(opposite & ~cancels, low - high, -1.0)
    log_abs = torch.where(
        opposite,
        high + (-difference.expm1()).log(),
        torch.logaddexp(left_log, right_log),
    )
    sign = torch.where(cancels, 0.0, torch.where(greater, left_sign, right_sign))
    lost = (left_sign != 0) & (right_sign != 0) & ~cancels & (log_abs == high)
    if bool(lost.any()):
        raise FloatingPointError("a nonzero signed log summand rounds away")
    return _pair(sign, log_abs)


def _sum(values: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    if not axes:
        return values
    last = values.ndim - 1
    keep = tuple(i for i in range(last) if i not in axes)
    width = prod(values.shape[i] for i in axes)
    row = values.permute((*keep, *axes, last)).reshape(
        (*(values.shape[i] for i in keep), width, 2)
    )
    if width == 1:
        return row.squeeze(-2)
    sign, log_abs = row.unbind(-1)
    log_abs, indices = log_abs.sort(dim=-1, descending=True, stable=True)
    sign = sign.gather(-1, indices)
    start = torch.cat(
        (
            torch.ones_like(sign[..., :1], dtype=torch.bool),
            log_abs[..., 1:] != log_abs[..., :-1],
        ),
        dim=-1,
    )
    groups = start.cumsum(dim=-1) - 1
    net = torch.zeros_like(sign).scatter_add(-1, groups, sign)
    magnitude = torch.full_like(log_abs, -torch.inf).scatter_reduce(
        -1, groups, log_abs, reduce="amax"
    )
    scaled = magnitude + net.abs().log()
    if bool(((net.abs() > 1) & (scaled == magnitude)).any()):
        raise FloatingPointError("signed log multiplicity rounds away")
    row = _pair(net.sign(), scaled)
    while row.shape[-2] > 1:
        size = row.shape[-2]
        even = size - size % 2
        merged = _add(row[..., :even:2, :], row[..., 1:even:2, :])
        row = torch.cat((merged, row[..., even:, :]), dim=-2)
    return row.squeeze(-2)


class SignedLogDualSemiring(_Packed):
    """A real dual algebra with independently signed log primal and tangent.

    Each element stores ``(sign, log_abs, tangent_sign, tangent_log_abs)``.
    Zero in either component is ``(0, -inf)``. A zero primal can have a
    nonzero tangent. The operations carry one supplied direction without
    differentiating the sign or logarithm of the primal representation.

    The carrier uses float64 arithmetic. Equal represented magnitudes
    cancel in a multiway sum before other additions. Detectable disappearance
    of a nonzero summand raises a numerical error. Algebraic laws concern
    the decoded ideal dual algebra.
    """

    @property
    def name(self) -> str:
        """The name of the signed log dual algebra."""
        return "SignedLogDual"

    def _packed_size(self) -> int:
        """Return the two signed log pairs' total width."""
        return 4

    def evidence(self, law: Law) -> LawEvidence:
        """Give the ideal carrier evidence for the requested law."""
        if law.scope == "carrier" and law in self.laws():
            return Proved(argument="api/core/carriers.md#proof-signed-log-dual")
        return super().evidence(law)

    def pack(self, primal: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """Combine two broadcastable canonical signed log pairs."""
        _check_pair(primal)
        _check_pair(tangent)
        return torch.cat(torch.broadcast_tensors(primal, tangent), dim=-1)

    def parts(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the primal and tangent pairs without changing their encoding."""
        self._check(value)
        primal, tangent = value[..., :2], value[..., 2:]
        _check_pair(primal)
        _check_pair(tangent)
        return primal, tangent

    def from_linear(self, primal: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """Embed two finite float64 components as a dual number."""
        return self.pack(_encode(primal), _encode(tangent))

    def to_linear(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode both components, refusing an unrepresentable nonzero value."""
        primal, tangent = self.parts(value)
        return _decode(primal), _decode(tangent)

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Return a Boolean tensor for the canonical primal and tangent pairs."""
        self._check(t)
        self._check_dtype(t)
        return _canonical(t[..., :2]) & _canonical(t[..., 2:])

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply the primal coefficients and apply the product rule."""
        x, dx = self.parts(a)
        y, dy = self.parts(b)
        return torch.cat(
            (_multiply(x, y), _add(_multiply(x, dy), _multiply(y, dx))), dim=-1
        )

    def _join_parts(self, t: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        """Sum the two coefficients independently."""
        primal, tangent = self.parts(t)
        return torch.cat((_sum(primal, axes), _sum(tangent, axes)), dim=-1)

    def zero_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """Broadcast the canonical primal and tangent zero to the index shape."""
        self._check_dtype(like)
        return like.new_tensor((0, -torch.inf, 0, -torch.inf)).expand((*shape, 4))

    def unit_like(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """Broadcast the canonical primal one and tangent zero."""
        self._check_dtype(like)
        return like.new_tensor((1, 0, 0, -torch.inf)).expand((*shape, 4))

    def _check_dtype(self, value: torch.Tensor) -> None:
        if value.dtype != torch.float64:
            raise TypeError("the signed log dual algebra requires float64")
