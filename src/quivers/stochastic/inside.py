"""CKY inside algorithm for probabilistic context-free grammars.

Implements the inside algorithm as a differentiable PyTorch module,
enabling end-to-end gradient-based learning of PCFG parameters.

A PCFG is specified by two stochastic morphisms:

- ``binary : N -> N * N``: binary production probabilities.
  For each nonterminal A, ``binary[A, B, C]`` is the probability
  of the rule A -> B C.
- ``lexical : N -> T``: terminal production probabilities.
  For each nonterminal A, ``lexical[A, t]`` is the probability
  of the rule A -> t.

The inside algorithm computes, for each nonterminal A and span
(i, j) of the input sentence:

    beta(A, i, j) = P(w_i ... w_{j-1} | A)

The sentence log-probability is ``log beta(start, 0, L)`` where
``start`` is the start symbol index (default 0).

All computation is done in log-space for numerical stability,
using logsumexp for marginalization. This preserves gradient
flow for learning rule probabilities end-to-end.

Unary rules
-----------
An optional unary morphism adds rules ``A -> B``. A chain of them
derives a category over a span from another over the same span, so
each chart cell is closed once under the closure matrix
``U* = (I - U)^{-1}`` of Stolcke (1995), section 4.5.2, which sums the
probability of every chain exactly once.

What it computes
----------------
The algorithm computes a function from nonempty terminal strings to
their inside weights under the start symbol: the sum, over the parse
trees of the string, of the product of the weights of the rules in the
tree. When the rule weights of each nonterminal form a probability
distribution over its productions, this is the string's probability.

Examples
--------
>>> from quivers.core.objects import FinSet, ProductSet
>>> from quivers.stochastic.morphisms import StochasticMorphism
>>> N = FinSet(name="N", cardinality=5)
>>> T = FinSet(name="T", cardinality=10)
>>> binary = StochasticMorphism(N, ProductSet(N, N))
>>> lexical = StochasticMorphism(N, T)
>>> cky = InsideAlgorithm(binary, lexical, start=0)
>>> tokens = torch.randint(0, 10, (4, 6))  # batch=4, length=6
>>> log_probs = cky(tokens)  # (4,)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quivers.core.morphisms import Morphism
from quivers.core.objects import ProductSet
from quivers.core.wiring import EinsumWiring
from quivers.stochastic._kleene import kleene_star
from quivers.stochastic.semiring import LOG_PROB


class InsideAlgorithm(nn.Module):
    """CKY inside algorithm for differentiable PCFG parsing.

    Computes sentence log-probabilities under a PCFG defined by
    binary and lexical production rules, both expressed as
    stochastic morphisms.

    Parameters
    ----------
    binary : Morphism
        Binary production rules. Must be a morphism ``N -> N * N``
        where N is a finite set of nonterminals. The tensor has
        shape ``(|N|, |N|, |N|)`` with ``binary[A, B, C]`` =
        P(A -> B C).
    lexical : Morphism
        Lexical (terminal) production rules. Must be a morphism
        ``N -> T`` where T is a finite set of terminals. The
        tensor has shape ``(|N|, |T|)`` with ``lexical[A, t]`` =
        P(A -> t).
    start : int
        Index of the start symbol in N (default 0).
    unary : Morphism or None
        Optional unary production rules, a morphism ``N -> N`` read in
        the derivation direction: ``unary[B, A]`` is the probability of
        the rule ``A -> B``, which derives ``A`` over a span from ``B``
        over the same span. Each chart cell is closed under these rules
        once, by the matrix ``U* = I + U + U² + ... = (I - U)⁻¹``, so a
        chain of unary rules is counted once
        [Stolcke (1995)](<https://aclanthology.org/J95-2002/>),
        section 4.5.2.

    Raises
    ------
    TypeError
        If the morphisms have incompatible types.

    Notes
    -----
    The closure exists only when the spectral radius of ``U`` is below
    one; a row-stochastic ``U`` (every category rewrites by some unary
    rule with probability one) has spectral radius one, and parsing
    with it raises `DivergentClosureError`.
    """

    def __init__(
        self,
        binary: Morphism,
        lexical: Morphism,
        start: int = 0,
        unary: Morphism | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(binary.codomain, ProductSet):
            raise TypeError(
                f"binary morphism codomain must be a ProductSet, got {binary.codomain!r}"
            )
        if binary.domain != lexical.domain:
            raise TypeError(
                f"binary and lexical must share the same domain (nonterminals), got {binary.domain!r} and {lexical.domain!r}"
            )
        if unary is not None:
            if unary.domain != binary.domain or unary.codomain != binary.domain:
                raise TypeError(
                    f"unary morphism must have shape N -> N matching binary "
                    f"domain {binary.domain!r}; got {unary.domain!r} -> {unary.codomain!r}"
                )
        self._binary = binary
        self._lexical = lexical
        self._unary = unary
        self._start = start
        self._n_nonterm = binary.domain.size
        self._n_term = lexical.codomain.size
        self._binary_mod = binary.module()
        self._lexical_mod = lexical.module()
        if unary is not None:
            self._unary_mod = unary.module()
        else:
            self._unary_mod = None

    @property
    def n_nonterminals(self) -> int:
        """Number of nonterminal symbols.

        Returns
        -------
        int
            The count.
        """
        return self._n_nonterm

    @property
    def n_terminals(self) -> int:
        """Number of terminal symbols.

        Returns
        -------
        int
            The count.
        """
        return self._n_term

    @property
    def start(self) -> int:
        """Index of the start symbol.

        Returns
        -------
        int
            The index.
        """
        return self._start

    def _tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """The rule probabilities.

        Returns
        -------
        tuple of torch.Tensor
            The binary, lexical and (or ``None``) unary tensors.
        """
        unary = self._unary.tensor if self._unary is not None else None
        return self._binary.tensor, self._lexical.tensor, unary

    def _fill_chart(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """The log inside chart by span length, without a gradient.

        Parameters
        ----------
        tokens : torch.Tensor
            Shape ``(batch, seq_len)``.

        Returns
        -------
        list of torch.Tensor
            Entry ``l`` is ``(batch, seq_len - l + 1, N)``: the log inside
            weight of each category over the span of length ``l`` at each
            start; entry ``0`` is unused.

        Raises
        ------
        DivergentClosureError
            If the unary rules' closure diverges.
        """
        binary, lexical, unary = self._tensors()
        with torch.no_grad():
            return _inside(*_logarithms(binary, lexical, unary), tokens)[0]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute sentence log-probabilities via the inside algorithm.

        Parameters
        ----------
        tokens : torch.Tensor
            Integer tensor of terminal indices. Shape
            ``(batch, seq_len)`` or ``(seq_len,)`` for a single
            sentence.

        Returns
        -------
        torch.Tensor
            Log-probability of each sentence under the grammar.
            Shape ``(batch,)`` or scalar for a single sentence.

        Raises
        ------
        ValueError
            If the sentences are empty.
        DivergentClosureError
            If the unary rules' closure diverges.
        """
        squeeze = False
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            squeeze = True
        if tokens.shape[1] == 0:
            raise ValueError("cannot parse empty sentences")
        binary, lexical, unary = self._tensors()
        empty = binary.new_zeros((0,))
        result = _InsideOutside.apply(
            binary, lexical, unary if unary is not None else empty, tokens, self._start
        )
        if squeeze:
            return result.squeeze(0)
        return result

    def inside_chart(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute the full inside chart (for analysis/debugging).

        Parameters
        ----------
        tokens : torch.Tensor
            Integer tensor of terminal indices. Shape
            ``(batch, seq_len)`` or ``(seq_len,)``.

        Returns
        -------
        torch.Tensor
            The full inside chart in log-space. Shape
            ``(batch, N, seq_len, seq_len+1)`` where entry
            ``[b, A, i, j]`` is ``log P(w_i..w_{j-1} | A)``.

        Raises
        ------
        ValueError
            If the sentences are empty.
        """
        squeeze = False
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            squeeze = True
        if tokens.shape[1] == 0:
            raise ValueError("cannot parse empty sentences")
        cells = self._fill_chart(tokens)
        batch, length = tokens.shape
        chart = torch.full(
            (batch, self._n_nonterm, length, length + 1),
            -torch.inf,
            dtype=cells[1].dtype,
            device=cells[1].device,
        )
        for width in range(1, length + 1):
            for start in range(length - width + 1):
                chart[:, :, start, start + width] = cells[width][:, start]
        if squeeze:
            return chart.squeeze(0)
        return chart

    def __repr__(self) -> str:
        """The sizes of the grammar.

        Returns
        -------
        str
            The representation.
        """
        return f"InsideAlgorithm(N={self._n_nonterm}, T={self._n_term}, start={self._start})"


_PAIRS = EinsumWiring(LOG_PROB, "bmkB,bmkC->bmBC")
"""The sum over split points of each pair of premise categories."""

_RULES = EinsumWiring(LOG_PROB, "ABC,bmBC->bmA")
"""The binary rules applied to the pair sums."""

_CLOSE = EinsumWiring(LOG_PROB, "bmB,BA->bmA")
"""A cell closed under the unary rules."""

_OUTSIDE_PRE = EinsumWiring(LOG_PROB, "BA,bmA->bmB")
"""The outside weight of a cell before its unary closure."""

_OUTSIDE_PAIRS = EinsumWiring(LOG_PROB, "bmA,ABC->bmBC")
"""The outside weight of each premise pair of a cell."""

_OUTSIDE_LEFT = EinsumWiring(LOG_PROB, "bmBC,bmkC->bmkB")
"""The outside weight of each left premise."""

_OUTSIDE_RIGHT = EinsumWiring(LOG_PROB, "bmBC,bmkB->bmkC")
"""The outside weight of each right premise."""

_BINARY_USES = EinsumWiring(LOG_PROB, "bmA,bmBC->bABC")
"""The derivative of a sentence's weight in each binary rule."""

_UNARY_USES = EinsumWiring(LOG_PROB, "bmB,bmA->bBA")
"""The derivative of a sentence's weight in each unary rule."""

_LEXICAL_USES = EinsumWiring(LOG_PROB, "bmA,bmt->bAt")
"""The derivative of a sentence's weight in each lexical rule."""


def _logarithms(
    binary: torch.Tensor, lexical: torch.Tensor, unary: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The log weights of the rules and the log closure of the unary ones.

    A probability of zero has the log weight ``-inf`` exactly.

    Parameters
    ----------
    binary : torch.Tensor
        ``(N, N, N)`` probabilities.
    lexical : torch.Tensor
        ``(N, T)`` probabilities.
    unary : torch.Tensor or None
        ``(N, N)`` probabilities, ``unary[B, A]`` deriving ``A`` from ``B``.

    Returns
    -------
    tuple of torch.Tensor
        The binary and lexical log weights, and the log of ``U*`` or
        ``None`` without unary rules.

    Raises
    ------
    DivergentClosureError
        If the closure diverges.
    """
    star = None if unary is None else kleene_star(torch.log(unary), LOG_PROB)
    return torch.log(binary), torch.log(lexical), star


def _spans(chart: list[torch.Tensor], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The left and right premises of every cell of one length.

    Parameters
    ----------
    chart : list of torch.Tensor
        The cells of the shorter lengths.
    width : int
        The length, at least two.

    Returns
    -------
    tuple of torch.Tensor
        ``(batch, m, width - 1, N)`` each: the cell over ``(i, i + k)`` and
        over ``(i + k, i + width)`` for each start ``i`` and split ``k``.
    """
    rows = chart[1].shape[1] - width + 1
    left = torch.stack([chart[k][:, :rows] for k in range(1, width)], dim=2)
    right = torch.stack(
        [chart[width - k][:, k : k + rows] for k in range(1, width)], dim=2
    )
    return left, right


def _inside(
    binary: torch.Tensor,
    lexical: torch.Tensor,
    star: torch.Tensor | None,
    tokens: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """The log inside chart, before and after each cell's unary closure.

    Parameters
    ----------
    binary : torch.Tensor
        ``(N, N, N)`` log weights.
    lexical : torch.Tensor
        ``(N, T)`` log weights.
    star : torch.Tensor or None
        The log closure of the unary rules.
    tokens : torch.Tensor
        ``(batch, n)`` terminals.

    Returns
    -------
    tuple of list of torch.Tensor
        The closed cells and the cells before closure, by length.
    """
    length = tokens.shape[1]
    closed: list[torch.Tensor] = [lexical.new_zeros((0,))]
    opened: list[torch.Tensor] = [lexical.new_zeros((0,))]
    for width in range(1, length + 1):
        if width == 1:
            cell = lexical.T[tokens]
        else:
            left, right = _spans(closed, width)
            cell = _RULES.apply(binary, _PAIRS.apply(left, right))
        opened.append(cell)
        closed.append(cell if star is None else _CLOSE.apply(cell, star))
    return closed, opened


def _cotangent(
    uses: torch.Tensor, level: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """The cotangent of the probabilities from log derivatives of ``P``.

    Parameters
    ----------
    uses : torch.Tensor
        ``(batch, ...)``: ``log dP / dp`` per sentence.
    level : torch.Tensor
        ``(batch,)``: ``log P``, zero for a sentence without a derivation.
    scale : torch.Tensor
        ``(batch,)``: the cotangent of ``log P``, zero for such a sentence.

    Returns
    -------
    torch.Tensor
        ``sum_b scale_b (dP_b / dp) / P_b``.
    """
    shape = (uses.shape[0],) + (1,) * (uses.dim() - 1)
    ratio = torch.exp(uses - level.reshape(shape))
    weighted = torch.where(scale.reshape(shape) != 0, scale.reshape(shape) * ratio, 0.0)
    return weighted.sum(0)


def _outside(
    binary: torch.Tensor,
    lexical: torch.Tensor,
    tokens: torch.Tensor,
    cells: list[torch.Tensor],
    star: torch.Tensor | None,
    start: int,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The derivatives in the rule probabilities, by the outside pass.

    Parameters
    ----------
    binary : torch.Tensor
        ``(N, N, N)`` log weights.
    lexical : torch.Tensor
        ``(N, T)`` log weights.
    tokens : torch.Tensor
        ``(batch, n)`` terminals.
    cells : list of torch.Tensor
        The closed inside cells of lengths one to ``n``.
    star : torch.Tensor or None
        The log closure of the unary rules.
    start : int
        The start symbol.
    upstream : torch.Tensor
        ``(batch,)`` cotangents of the log weights.

    Returns
    -------
    tuple of torch.Tensor
        The cotangents of the binary, lexical and unary probabilities; the
        last is meaningless without unary rules.
    """
    batch, length = tokens.shape
    size, terminals = lexical.shape
    closed = [lexical.new_zeros((0,)), *cells]
    impossible = -torch.inf
    outside = [lexical.new_zeros((0,))] + [
        torch.full_like(cells[width - 1], impossible) for width in range(1, length + 1)
    ]
    goal = outside[length].clone()
    goal[:, 0, start] = 0.0
    outside[length] = goal
    uses_binary = torch.full((batch, size, size, size), impossible).to(binary)
    uses_unary = torch.full((batch, size, size), impossible).to(binary)
    uses_lexical = torch.full((batch, size, terminals), impossible).to(binary)
    for width in range(length, 0, -1):
        before = (
            outside[width] if star is None else _OUTSIDE_PRE.apply(star, outside[width])
        )
        if star is not None:
            uses_unary = LOG_PROB.join_pair(
                uses_unary, _UNARY_USES.apply(closed[width], before)
            )
        if width == 1:
            marks = torch.where(
                torch.nn.functional.one_hot(tokens, terminals).bool(),
                0.0,
                impossible,
            ).to(binary)
            uses_lexical = _LEXICAL_USES.apply(before, marks)
            continue
        left, right = _spans(closed, width)
        uses_binary = LOG_PROB.join_pair(
            uses_binary, _BINARY_USES.apply(before, _PAIRS.apply(left, right))
        )
        pairs = _OUTSIDE_PAIRS.apply(before, binary)
        to_left = _OUTSIDE_LEFT.apply(pairs, right)
        to_right = _OUTSIDE_RIGHT.apply(pairs, left)
        rows = to_left.shape[1]
        for split in range(1, width):
            head = outside[split].clone()
            head[:, :rows] = LOG_PROB.join_pair(
                head[:, :rows], to_left[:, :, split - 1]
            )
            outside[split] = head
            tail = outside[width - split].clone()
            tail[:, split : split + rows] = LOG_PROB.join_pair(
                tail[:, split : split + rows], to_right[:, :, split - 1]
            )
            outside[width - split] = tail
    total = closed[length][:, 0, start]
    possible = torch.isfinite(total)
    scale = torch.where(possible, upstream, torch.zeros_like(upstream))
    level = torch.where(possible, total, torch.zeros_like(total))

    return (
        _cotangent(uses_binary, level, scale),
        _cotangent(uses_lexical, level, scale),
        _cotangent(uses_unary, level, scale),
    )


class _InsideOutside(torch.autograd.Function):
    """Sentence log weights, differentiated by the outside pass.

    The derivative of ``log P`` in a rule probability ``p`` is
    ``(dP / dp) / P``, and ``dP / dp`` is a sum of products of outside and
    inside weights that contains no factor ``p``. Computing it in the log
    semiring by the outside pass gives it exactly where ``p`` is zero, where
    the derivative of ``log p`` is not finite and differentiating the log
    weights would give ``0 * inf``. A sentence without a derivation has the
    log weight ``-inf`` and derivative zero, the convention of
    `quivers.core.algebras.LogProbAlgebra`. Only the chart is kept for the
    backward pass.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        binary: torch.Tensor,
        lexical: torch.Tensor,
        unary: torch.Tensor,
        tokens: torch.Tensor,
        start: int,
    ) -> torch.Tensor:
        """The log weight of each sentence under the start symbol.

        Parameters
        ----------
        ctx : FunctionCtx
            The context.
        binary : torch.Tensor
            ``(N, N, N)`` probabilities.
        lexical : torch.Tensor
            ``(N, T)`` probabilities.
        unary : torch.Tensor
            ``(N, N)`` probabilities, or an empty tensor without unary rules.
        tokens : torch.Tensor
            ``(batch, n)`` terminals.
        start : int
            The start symbol.

        Returns
        -------
        torch.Tensor
            ``(batch,)`` log weights.
        """
        logs = _logarithms(binary, lexical, unary if unary.numel() else None)
        closed, opened = _inside(*logs, tokens)
        ctx.save_for_backward(logs[0], logs[1], tokens, *closed[1:])
        ctx.star = logs[2]  # type: ignore[attr-defined]
        ctx.start = start  # type: ignore[attr-defined]
        ctx.unary = unary.numel() > 0  # type: ignore[attr-defined]
        ctx.shape = unary.shape  # type: ignore[attr-defined]
        del opened
        return closed[tokens.shape[1]][:, 0, start]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, upstream: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """The derivatives in the rule probabilities, by the outside pass.

        Parameters
        ----------
        ctx : FunctionCtx
            The context.
        upstream : torch.Tensor
            ``(batch,)`` cotangents.

        Returns
        -------
        tuple
            The derivatives in the binary, lexical and unary probabilities,
            and none for the terminals and the start symbol.
        """
        binary, lexical, tokens, *cells = ctx.saved_tensors  # type: ignore[attr-defined]
        grads = _outside(
            binary,
            lexical,
            tokens,
            cells,
            ctx.star,  # type: ignore[attr-defined]
            ctx.start,  # type: ignore[attr-defined]
            upstream,
        )
        unary = (
            grads[2]
            if ctx.unary  # type: ignore[attr-defined]
            else binary.new_zeros(ctx.shape)  # type: ignore[attr-defined]
        )
        return grads[0], grads[1], unary, None, None


__all__ = [
    "InsideAlgorithm",
]
