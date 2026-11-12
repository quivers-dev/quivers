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

Categorical perspective
-----------------------
The inside algorithm implements a morphism

    inside(binary, lexical) : FreeMonoid(T) -> 1

that maps variable-length terminal strings to their probability
under the grammar. This is the counit of the adjunction between
the free monad on the polynomial functor induced by the grammar
and the forgetful functor to strings.

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
        Optional unary production rules, a morphism ``N -> N``
        closed over at every chart cell.

    Raises
    ------
    TypeError
        If the morphisms have incompatible types.
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
        """
        batch, seq_len = tokens.shape
        log_binary = torch.log(self._binary.tensor.clamp(min=1e-30))
        log_lexical = torch.log(self._lexical.tensor.clamp(min=1e-30))
        log_unary = (
            _unary_star(torch.log(self._unary.tensor.clamp(min=1e-30)))
            if self._unary is not None
            else None
        )
        N = self._n_nonterm
        cells: list[list[torch.Tensor | None]] = [
            [None for _ in range(seq_len + 1)] for _ in range(seq_len)
        ]
        for i in range(seq_len):
            tok_i = tokens[:, i]
            cell = log_lexical[:, tok_i].T
            if log_unary is not None:
                cell = _apply_unary_closure(cell, log_unary)
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
                if log_unary is not None:
                    cell = _apply_unary_closure(cell, log_unary)
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


def _unary_star(log_unary: torch.Tensor) -> torch.Tensor:
    """The log of the reflexive-transitive closure of the unary rules.

    With ``U`` the matrix of unary rule probabilities, a cell's weights
    after any number of unary steps, zero included, are ``c U*`` for the
    star ``U* = sum_{k >= 0} U^k = (I - U)^{-1}``, the closure in the
    probability semiring (Goodman 1999, "Semiring Parsing",
    https://aclanthology.org/J99-4004/). The series converges, and
    the inverse is that sum, exactly when the spectral radius of ``U``
    is below one.

    Parameters
    ----------
    log_unary : torch.Tensor
        Shape ``(N, N)``: the log-probability of each unary rule, the
        entry at ``[B, A]`` weighing ``B`` into ``A``.

    Returns
    -------
    torch.Tensor
        Shape ``(N, N)``: ``log U*``, ``-inf`` where the closure is zero.

    Raises
    ------
    ValueError
        If the spectral radius of ``U`` is at least one, so the unary
        rules derive infinite weight and the star does not exist.
    """
    unary = log_unary.exp()
    radius = float(torch.linalg.eigvals(unary.detach().double()).abs().max())
    if radius >= 1.0:
        raise ValueError(
            f"the unary rules have spectral radius {radius:.6g}, at least one, "
            "so chains of unary steps carry unbounded weight and their closure "
            "does not exist; the rules out of each category must sum to "
            "less than one along every cycle"
        )
    eye = torch.eye(unary.shape[0], dtype=unary.dtype, device=unary.device)
    star = torch.linalg.inv(eye - unary)
    positive = star > 0
    safe = torch.where(positive, star, torch.ones_like(star))
    return torch.where(positive, torch.log(safe), torch.full_like(star, float("-inf")))


def _apply_unary_closure(
    log_cell: torch.Tensor, log_star: torch.Tensor
) -> torch.Tensor:
    """A cell's log weights after any number of unary steps.

    ``cell[A] <- logsumexp_B(cell[B] + log_star[B, A])``, the cell times
    the closure ``U*`` of the unary rules: the fixed point of
    ``x = c + x U``, which every derivation that ends in a chain of unary
    rules contributes to once.

    Parameters
    ----------
    log_cell : torch.Tensor
        Shape ``(batch, N)``: the cell's log weights before unary steps.
    log_star : torch.Tensor
        Shape ``(N, N)``: the log closure from `_unary_star`.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, N)``.
    """
    return _masked_logsumexp(log_cell.unsqueeze(2) + log_star.unsqueeze(0), dim=1)


__all__ = [
    "InsideAlgorithm",
]
