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
        """Number of nonterminal symbols."""
        return self._n_nonterm

    @property
    def n_terminals(self) -> int:
        """Number of terminal symbols."""
        return self._n_term

    @property
    def start(self) -> int:
        """Index of the start symbol."""
        return self._start

    def _fill_chart(self, tokens: torch.Tensor) -> torch.Tensor:
        """Fill the inside chart without in-place tensor modifications.

        Uses a cell-list approach to preserve autograd graph.

        Parameters
        ----------
        tokens : torch.Tensor
            Shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Chart of shape ``(batch, N, seq_len, seq_len+1)``.

        Raises
        ------
        DivergentClosureError
            If the unary rules' closure diverges.
        """
        batch, seq_len = tokens.shape
        log_binary = torch.log(self._binary.tensor.clamp(min=1e-30))
        log_lexical = torch.log(self._lexical.tensor.clamp(min=1e-30))
        log_star = _log_star(self._unary.tensor) if self._unary is not None else None
        N = self._n_nonterm
        cells: list[list[torch.Tensor | None]] = [
            [None for _ in range(seq_len + 1)] for _ in range(seq_len)
        ]
        for i in range(seq_len):
            tok_i = tokens[:, i]
            cell = log_lexical[:, tok_i].T
            if log_star is not None:
                cell = _apply_unary_closure(cell, log_star)
            cells[i][i + 1] = cell
        for span_len in range(2, seq_len + 1):
            for i in range(seq_len - span_len + 1):
                j = i + span_len
                parts = []
                for k in range(i + 1, j):
                    left = cells[i][k]
                    right = cells[k][j]
                    assert left is not None and right is not None
                    combined = (
                        log_binary.unsqueeze(0)
                        + left.unsqueeze(1).unsqueeze(3)
                        + right.unsqueeze(1).unsqueeze(2)
                    )
                    split_score = _masked_logsumexp(
                        combined.reshape(batch, N, -1), dim=-1
                    )
                    parts.append(split_score)
                stacked = torch.stack(parts, dim=0)
                cell = _masked_logsumexp(stacked, dim=0)
                if log_star is not None:
                    cell = _apply_unary_closure(cell, log_star)
                cells[i][j] = cell
        chart = torch.full(
            (batch, N, seq_len, seq_len + 1), float("-inf"), device=tokens.device
        )
        for i in range(seq_len):
            for j in range(i + 1, seq_len + 1):
                cell = cells[i][j]
                if cell is not None:
                    chart[:, :, i, j] = cell
        return chart

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
        chart = self._fill_chart(tokens)
        result = chart[:, self._start, 0, tokens.shape[1]]
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
        chart = self._fill_chart(tokens)
        if squeeze:
            return chart.squeeze(0)
        return chart

    def __repr__(self) -> str:
        return f"InsideAlgorithm(N={self._n_nonterm}, T={self._n_term}, start={self._start})"


def _masked_logsumexp(scores: torch.Tensor, dim: int) -> torch.Tensor:
    """``logsumexp`` whose empty rows carry no gradient.

    Replace negative infinities with the dtype floor during reduction,
    then restore all-empty rows to negative infinity. This avoids NaN
    gradients for unreachable chart entries. Existing NaN inputs are
    unchanged.

    Parameters
    ----------
    scores : torch.Tensor
        Log-space scores to reduce.
    dim : int
        Axis to reduce.
    """
    empty = torch.isneginf(scores)
    floor = torch.finfo(scores.dtype).min
    bounded = torch.where(empty, torch.full_like(scores, floor), scores)
    reduced = torch.logsumexp(bounded, dim=dim)
    all_empty = empty.all(dim=dim)
    return torch.where(all_empty, torch.full_like(reduced, float("-inf")), reduced)


def _log_star(unary: torch.Tensor) -> torch.Tensor:
    """The log of the unary rules' closure ``U* = (I - U)⁻¹``.

    Parameters
    ----------
    unary : torch.Tensor
        Shape ``(N, N)``: ``unary[B, A]`` is the probability of deriving
        ``A`` from ``B`` over one span.

    Returns
    -------
    torch.Tensor
        Shape ``(N, N)``: the log of the total probability of the chains
        of unary rules deriving ``A`` from ``B`` (``0`` on the diagonal
        when no cycle returns there), ``-inf`` where there is none.

    Raises
    ------
    DivergentClosureError
        If the spectral radius of ``U`` is at least one.
    """
    log_unary = torch.where(
        unary > 0, unary.clamp(min=torch.finfo(unary.dtype).tiny).log(), -torch.inf
    )
    return kleene_star(log_unary, LOG_PROB)


def _apply_unary_closure(
    log_cell: torch.Tensor, log_star: torch.Tensor
) -> torch.Tensor:
    """Close a chart cell under the unary rules: ``cell · U*`` in log space.

    Parameters
    ----------
    log_cell : torch.Tensor
        Shape ``(batch, N)``: the cell's log inside weights from its
        lexical or binary derivations.
    log_star : torch.Tensor
        Shape ``(N, N)``: the log closure from `_log_star`.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, N)``: ``log Σ_B exp(cell[B]) U*[B, A]``.
    """
    return _masked_logsumexp(log_cell.unsqueeze(2) + log_star.unsqueeze(0), dim=1)


__all__ = [
    "InsideAlgorithm",
]
