"""Span-based CKY components for the deductive system framework.

This module instantiates the abstract deduction primitives
(``Axiom``, ``Deduction``, ``Goal``, ``Schedule``) for span-indexed
chart parsing (CKY). Items are indexed by ``(i, j, A)`` where
``[i, j)`` is a string span and ``A`` is a category index.

The chart tensor has shape ``(batch, n_categories, seq_len, seq_len+1)``
(categories-first for efficient indexing of the start symbol).

Components
----------
LexicalAxiom
    Populates length-1 spans from a learnable lexicon.
BinarySpanDeduction
    Applies binary rules: chart[i,k,L] ⊗ chart[k,j,R] → chart[i,j,A].
UnarySpanDeduction
    Closes a cell under the unary rules by their Kleene star.
SpanGoal
    Extracts chart[start_cat, 0, seq_len].
CKYSchedule
    Bottom-up evaluation by span length.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from quivers.stochastic._kleene import apply_star, kleene_star, semiring_sum
from quivers.stochastic._rule_system import RuleSystem
from quivers.stochastic.deduction import Axiom, Deduction, Goal, Schedule
from quivers.stochastic.semiring import ChartSemiring


# ================================================================
# axiom: lexical entries
# ================================================================


class LexicalAxiom(Axiom):
    """Populate length-1 spans from a learnable lexicon.

    Creates the chart tensor and fills
    ``chart[b, :, i, i+1] = log_softmax(lexicon[tokens[b, i]])``.

    Parameters
    ----------
    n_terminals : int
        Vocabulary size.
    n_categories : int
        Number of category types.
    """

    def __init__(self, n_terminals: int, n_categories: int) -> None:
        super().__init__()
        self._n_cat = n_categories
        self.lexicon_logits = nn.Parameter(torch.randn(n_terminals, n_categories) * 0.1)

    @property
    def log_lexicon(self) -> torch.Tensor:
        """Log-probability lexicon, shape (n_terminals, n_categories)."""
        return torch.log_softmax(self.lexicon_logits, dim=-1)

    def forward(self, tokens: torch.Tensor, semiring: ChartSemiring) -> SpanChart:  # type: ignore[override]
        """Fill the length-1 spans from the lexicon.

        Parameters
        ----------
        tokens : torch.Tensor
            Integer terminal indices of shape ``(batch, seq_len)``.
        semiring : ChartSemiring
            The scoring semiring, recorded on the chart.

        Returns
        -------
        SpanChart
            A chart whose cells ``(i, i + 1)`` hold the lexical
            log-probabilities of each token.
        """
        batch, seq_len = tokens.shape
        log_lex = self.log_lexicon[tokens]  # (batch, seq_len, C)

        # build chart as list-of-lists (functional, no in-place writes)
        # stored as flat dict keyed by (i, j)
        chart_cells: dict[tuple[int, int], torch.Tensor] = {}

        for i in range(seq_len):
            chart_cells[(i, i + 1)] = log_lex[:, i, :]  # (batch, C)

        # pack into metadata for the schedule
        return SpanChart(
            cells=chart_cells,
            batch=batch,
            seq_len=seq_len,
            n_categories=self._n_cat,
            device=tokens.device,
            semiring=semiring,
        )


class SpanChart:
    """A span-indexed chart held as functional (non-in-place) cells.

    This wraps a dict of ``(i, j) -> (batch, C)`` tensors. The
    schedule fills cells bottom-up, and ``to_tensor()`` assembles
    them into a dense ``(batch, C, seq_len, seq_len+1)`` tensor.

    Parameters
    ----------
    cells : dict of (int, int) to torch.Tensor
        The filled cells, keyed by span ``(i, j)``, each of shape
        ``(batch, n_categories)``.
    batch : int
        The batch size.
    seq_len : int
        The sentence length.
    n_categories : int
        The number of categories.
    device : torch.device
        The device the dense chart is built on.
    semiring : ChartSemiring
        The scoring semiring; its ``zero`` fills unfilled cells.
    """

    __slots__ = ("cells", "batch", "seq_len", "n_categories", "device", "semiring")

    def __init__(
        self,
        cells: dict[tuple[int, int], torch.Tensor],
        batch: int,
        seq_len: int,
        n_categories: int,
        device: torch.device,
        semiring: ChartSemiring,
    ) -> None:
        self.cells = cells
        self.batch = batch
        self.seq_len = seq_len
        self.n_categories = n_categories
        self.device = device
        self.semiring = semiring

    def to_tensor(self) -> torch.Tensor:
        """Assemble cells into a dense chart tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_categories, seq_len, seq_len+1)``.
        """
        chart = torch.full(
            (self.batch, self.n_categories, self.seq_len, self.seq_len + 1),
            self.semiring.zero,
            device=self.device,
        )

        for (i, j), cell in self.cells.items():
            chart[:, :, i, j] = cell

        return chart


# ================================================================
# deductions
# ================================================================


def _scatter_semiring(
    scores: torch.Tensor,
    indices: torch.Tensor,
    n_categories: int,
    batch: int,
    device: torch.device,
    semiring: ChartSemiring,
) -> torch.Tensor:
    """Scatter rule scores into category bins via semiring plus."""
    result = torch.full(
        (batch, n_categories),
        semiring.zero,
        device=device,
    )

    for cat_idx in range(n_categories):
        mask = indices == cat_idx

        if mask.any():
            relevant = scores[:, mask]
            aggregated = semiring.plus(relevant, dim=-1)
            result = torch.cat(
                [
                    result[:, :cat_idx],
                    aggregated.unsqueeze(-1),
                    result[:, cat_idx + 1 :],
                ],
                dim=-1,
            )

    return result


class BinarySpanDeduction(Deduction):
    """Binary rule application over span items.

    For each split point k in [i+1, j), combines
    ``chart[i, k, lefts]`` and ``chart[k, j, rights]``
    to produce ``chart[i, j, results]``.

    Parameters
    ----------
    rule_system : RuleSystem
        The structural rules.
    learnable : bool
        Whether rule weights are learnable.
    """

    def __init__(
        self,
        rule_system: RuleSystem,
        learnable: bool = True,
    ) -> None:
        super().__init__()

        results, lefts, rights = rule_system.binary_tensors()
        self.register_buffer("_results", results)
        self.register_buffer("_lefts", lefts)
        self.register_buffer("_rights", rights)
        self._n_rules = rule_system.n_binary

        weights = rule_system.binary_weight_tensor()

        if learnable and self._n_rules > 0:
            self.weights = nn.Parameter(weights)

        else:
            self.register_buffer("weights", weights)

    def forward(
        self,
        chart: torch.Tensor | None,
        semiring: ChartSemiring,
        **context,
    ) -> torch.Tensor:
        """Apply binary rules for one span cell (i, j).

        Parameters
        ----------
        chart : torch.Tensor or None
            Unused; the cells are read from ``chart_cells``.
        semiring : ChartSemiring
            The scoring semiring.
        **context
            ``i`` and ``j`` (the span bounds), ``chart_cells`` (the
            filled cells keyed by span), and ``n_categories``.

        Returns
        -------
        torch.Tensor
            The ``(batch, C)`` scores for cell ``(i, j)``, aggregated
            over every split point.
        """
        i = context["i"]
        j = context["j"]
        chart_cells = context["chart_cells"]
        n_categories = context["n_categories"]

        _results = cast(torch.Tensor, self._results)
        _lefts = cast(torch.Tensor, self._lefts)
        _rights = cast(torch.Tensor, self._rights)

        if self._n_rules == 0:
            return torch.full(
                (chart_cells[(i, i + 1)].shape[0], n_categories),
                semiring.zero,
                device=_results.device,
            )

        parts = []

        for k in range(i + 1, j):
            left_cell = chart_cells[(i, k)]
            right_cell = chart_cells[(k, j)]

            left_scores = left_cell[:, _lefts]
            right_scores = right_cell[:, _rights]
            combined = semiring.times(left_scores, right_scores)

            if self._n_rules > 0:
                combined = semiring.times(
                    combined,
                    self.weights.unsqueeze(0),
                )

            parts.append(combined)

        stacked = torch.stack(parts, dim=0)
        split_scores = semiring.plus(stacked, dim=0)

        return _scatter_semiring(
            split_scores,
            _results,
            n_categories,
            split_scores.shape[0],
            device=_results.device,
            semiring=semiring,
        )


class UnarySpanDeduction(Deduction):
    """Unary rules over one span, closed under their Kleene star.

    A unary rule ``(a, b)`` with weight ``w`` derives category ``a``
    over a span from category ``b`` over the same span. Chains of such
    rules never consume input and may cycle, so a cell is closed once by
    the closure ``U* = I ⊕ U ⊕ U² ⊕ ...`` of the rule matrix
    ``U[b, a] = ⊕ w``, which sums every chain exactly once
    [Stolcke (1995)](<https://aclanthology.org/J95-2002/>), section
    4.5.2, generalized to a semiring as in
    [Goodman (1999)](<https://aclanthology.org/J99-4004/>).

    Parameters
    ----------
    rule_system : RuleSystem
        The structural rules.
    learnable : bool
        Whether rule weights are learnable.
    """

    def __init__(
        self,
        rule_system: RuleSystem,
        learnable: bool = True,
    ) -> None:
        super().__init__()

        unary = rule_system.unary_tensors()

        if unary is not None:
            self.register_buffer("_results", unary[0])
            self.register_buffer("_inputs", unary[1])

        else:
            self.register_buffer(
                "_results",
                torch.zeros(0, dtype=torch.long),
            )
            self.register_buffer(
                "_inputs",
                torch.zeros(0, dtype=torch.long),
            )

        self._n_rules = rule_system.n_unary

        weights = rule_system.unary_weight_tensor()

        if learnable and self._n_rules > 0:
            self.weights = nn.Parameter(weights)

        else:
            self.register_buffer("weights", weights)

    @property
    def has_rules(self) -> bool:
        """Whether this deduction has any unary rules."""
        return self._n_rules > 0

    def rule_matrix(self, n_categories: int, semiring: ChartSemiring) -> torch.Tensor:
        """The unary rules as a matrix over the semiring.

        Parameters
        ----------
        n_categories : int
            The number of categories ``C``.
        semiring : ChartSemiring
            The scoring semiring.

        Returns
        -------
        torch.Tensor
            ``(C, C)``: entry ``[b, a]`` is the semiring sum of the
            weights of the rules deriving ``a`` from ``b``, and the
            semiring's zero where there is none.
        """
        _results = cast(torch.Tensor, self._results)
        _inputs = cast(torch.Tensor, self._inputs)
        weights = cast(torch.Tensor, self.weights)
        flat = _inputs * n_categories + _results
        one_hot = torch.full(
            (self._n_rules, n_categories * n_categories),
            float(semiring.zero),
            dtype=weights.dtype,
            device=weights.device,
        )
        placed = one_hot.scatter(1, flat.unsqueeze(1), weights.unsqueeze(1))
        return semiring_sum(placed, dim=0, semiring=semiring).reshape(
            n_categories, n_categories
        )

    def closure(self, n_categories: int, semiring: ChartSemiring) -> torch.Tensor:
        """The Kleene star ``U*`` of the unary-rule matrix.

        Parameters
        ----------
        n_categories : int
            The number of categories ``C``.
        semiring : ChartSemiring
            The scoring semiring.

        Returns
        -------
        torch.Tensor
            ``(C, C)``: entry ``[b, a]`` is the semiring sum, over the
            chains of unary rules deriving ``a`` from ``b`` (the empty
            chain included), of the product of their weights.

        Raises
        ------
        DivergentClosureError
            If the closure's series diverges in the semiring.
        """
        return kleene_star(self.rule_matrix(n_categories, semiring), semiring)

    def forward(
        self,
        chart: torch.Tensor,
        semiring: ChartSemiring,
        **context,
    ) -> torch.Tensor:
        """Close a cell under the unary rules.

        Parameters
        ----------
        chart : torch.Tensor
            A ``(batch, C)`` cell tensor.
        semiring : ChartSemiring
            The scoring semiring.
        **context
            ``closure``: a precomputed `closure` to reuse across the
            cells of one parse; computed here when absent.

        Returns
        -------
        torch.Tensor
            The ``(batch, C)`` cell ``chart ⊗ U*``.

        Raises
        ------
        DivergentClosureError
            If the closure's series diverges in the semiring.
        """
        if not self.has_rules:
            return chart
        star = context.get("closure")
        if not isinstance(star, torch.Tensor):
            star = self.closure(chart.shape[1], semiring)
        return apply_star(chart, star, semiring)


# ================================================================
# goal
# ================================================================


class SpanGoal(Goal):
    """Extract chart[start_cat, 0, seq_len].

    Parameters
    ----------
    start_idx : int
        Index of the start category.
    """

    def __init__(self, start_idx: int) -> None:
        super().__init__()
        self._start_idx = start_idx

    def forward(self, chart: torch.Tensor) -> torch.Tensor:
        """Read the start category's score over the whole sentence.

        Parameters
        ----------
        chart : torch.Tensor
            The dense chart of shape ``(batch, C, seq_len, seq_len+1)``.

        Returns
        -------
        torch.Tensor
            The ``(batch,)`` scores at ``chart[:, start, 0, seq_len]``.
        """
        seq_len = chart.shape[2]
        return chart[:, self._start_idx, 0, seq_len]


# ================================================================
# schedule
# ================================================================


class CKYSchedule(Schedule):
    """Bottom-up CKY: process spans in increasing length order.

    For each span length 2..n:
      1. Apply binary deductions for each cell (i, i+span_len).
      2. Apply unary deductions to the resulting cell.
    """

    def run(  # type: ignore[override]
        self,
        chart: SpanChart,
        deductions: nn.ModuleList,
        semiring: ChartSemiring,
    ) -> torch.Tensor:
        """Fill the chart bottom-up and return it as a dense tensor.

        Parameters
        ----------
        chart : SpanChart
            The chart `LexicalAxiom` produced, with its length-1
            cells filled.
        deductions : nn.ModuleList
            The `BinarySpanDeduction` and `UnarySpanDeduction`
            steps; other deductions are ignored.
        semiring : ChartSemiring
            The scoring semiring.

        Returns
        -------
        torch.Tensor
            The dense chart of shape ``(batch, C, seq_len, seq_len+1)``.
        """
        span_chart = chart
        cells = span_chart.cells
        seq_len = span_chart.seq_len
        n_categories = span_chart.n_categories

        # separate binary and unary deductions
        binary_deds = [d for d in deductions if isinstance(d, BinarySpanDeduction)]
        unary_deds = [d for d in deductions if isinstance(d, UnarySpanDeduction)]

        # each unary deduction's closure is computed once per parse
        closures = {
            id(ud): ud.closure(n_categories, semiring)
            for ud in unary_deds
            if ud.has_rules
        }

        # apply unary rules to lexical cells first
        for i in range(seq_len):
            cell = cells[(i, i + 1)]

            for ud in unary_deds:
                if ud.has_rules:
                    cell = ud(cell, semiring, closure=closures[id(ud)])

            cells[(i, i + 1)] = cell

        # fill chart bottom-up
        for span_len in range(2, seq_len + 1):
            for i in range(seq_len - span_len + 1):
                j = i + span_len

                # binary step: accumulate across all binary deductions
                cell = None

                for bd in binary_deds:
                    contrib = bd(
                        None,
                        semiring,
                        i=i,
                        j=j,
                        chart_cells=cells,
                        n_categories=n_categories,
                    )

                    if cell is None:
                        cell = contrib

                    else:
                        cell = semiring.plus_pair(cell, contrib)

                if cell is None:
                    cell = torch.full(
                        (span_chart.batch, n_categories),
                        semiring.zero,
                        device=span_chart.device,
                    )

                # unary step
                for ud in unary_deds:
                    if ud.has_rules:
                        cell = ud(cell, semiring, closure=closures[id(ud)])

                cells[(i, j)] = cell

        # assemble into dense tensor
        return span_chart.to_tensor()


__all__ = [
    "LexicalAxiom",
    "SpanChart",
    "BinarySpanDeduction",
    "UnarySpanDeduction",
    "SpanGoal",
    "CKYSchedule",
]
