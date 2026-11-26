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

Span plans
----------
`span_grammar` reads the span form of a rule presentation, and
`span_inside` and `span_recognize` are the executors that
`quivers.stochastic.planning.plan` selects for it: values by length,
with the split points of each premise pair summed before the rules, and
support by Boolean matrix products over the split points.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Literal, cast

import didactic.api as dx
import torch
import torch.nn as nn

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.core.algebras import OrderPreservingSemiring, Star
from quivers.core.elimination import fiber_join
from quivers.qiec.identifiers import FamilyId
from quivers.qiec.kinds import NAT, UserIndexSort
from quivers.qiec.rules import (
    Ground,
    Item,
    Judgment,
    Rule,
    RulePresentation,
    evaluate_index,
)
from quivers.qiec.types import IndexLiteral, IndexVariable
from quivers.stochastic.indexed import side_condition_holds

from quivers.stochastic._kleene import apply_star, kleene_star
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
        """The log-probability lexicon.

        Returns
        -------
        torch.Tensor
            Shape ``(n_terminals, n_categories)``.
        """
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
        like = next(iter(self.cells.values()), torch.empty(0, device=self.device))
        chart = self.semiring.zero_like(
            (self.batch, self.n_categories, self.seq_len, self.seq_len + 1), like
        )

        for (i, j), cell in self.cells.items():
            chart[:, :, i, j] = cell

        return chart


# ================================================================
# deductions
# ================================================================


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

        fibers: list[list[int]] = [[] for _ in range(n_categories)]
        for position, result in enumerate(_results.tolist()):
            fibers[result].append(position)
        return fiber_join(
            semiring,
            split_scores,
            fibers,
            semiring.zero_like((split_scores.shape[0], 1), split_scores),
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
        """Whether this deduction has any unary rules.

        Returns
        -------
        bool
            True when it has one.
        """
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
        return semiring.plus(placed, dim=0).reshape(n_categories, n_categories)

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
    "SpanGrammar",
    "SpanRefused",
    "SpanRule",
    "SpanWork",
    "span_grammar",
    "span_inside",
    "span_recognize",
]


# ================================================================
# span plans
# ================================================================


class SpanRefused(ValueError):
    """A presentation is not in span form, or a run does not fit its plan.

    Parameters
    ----------
    reason : str
        Why.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SpanRule(ValueModel):
    """One instance of a rule over categories, as the span kernel reads it.

    Parameters
    ----------
    rule : Rule
        The rule.
    binding : dict of str to Ground
        The values of its category variables.
    result : int
        The conclusion's category.
    left : int
        The first premise's category; ``-1`` for a rule without premises.
    right : int
        The second premise's category; ``-1`` for a rule with fewer than two.
    reads : str
        What positions the side condition reads: ``"none"``, ``"span"``
        (the conclusion's ends only), or ``"split"`` (the split point too).
    positions : tuple of str
        The rule's position variables: start and end, with the split
        point between them for a binary rule.
    """

    rule: Rule = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    binding: dict[str, Ground] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result: int
    left: int = -1
    right: int = -1
    reads: Literal["none", "span", "split"] = "none"
    positions: tuple[str, ...] = ()


class SpanGrammar(ValueModel):
    """A presentation in span form: categories over spans of positions.

    Every judgment is of one family whose indices are some indices of
    closed sorts of nullary constructors, the *category*, and two natural
    numbers, the *start* and *end* of a span. Every rule is binary,
    ``A(i, j) <- B(i, k), C(k, j)``, unary, ``A(i, j) <- B(i, j)``, or
    has no premises and a ground conclusion; its position arguments are
    variables, its category arguments literals or variables of their
    sorts (a variable stands for each constructor in turn), and its side
    condition may read any of them. The length ``j - i`` grows strictly
    along a binary rule, so the spans of each length are final once the
    shorter ones are; that is the schedule.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    family : FamilyId
        The family.
    category_slots : tuple of int
        The argument positions of the category.
    start : int
        The argument position of the start.
    end : int
        The argument position of the end.
    categories : tuple of tuple of str
        Every category, its constructors at the category slots in order.
    binary : tuple of SpanRule
        The binary rule instances.
    unary : tuple of SpanRule
        The unary rule instances.
    constants : tuple of SpanRule
        The rules without premises.
    unary_cyclic : bool
        Whether the unary rules' graph of categories has a cycle.
    """

    presentation: RulePresentation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    family: FamilyId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    category_slots: tuple[int, ...] = ()
    start: int = 0
    end: int = 1
    categories: tuple[tuple[str, ...], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    binary: tuple[SpanRule, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    unary: tuple[SpanRule, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    constants: tuple[SpanRule, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    unary_cyclic: bool = False

    @property
    def pairs(self) -> tuple[tuple[int, int, int], ...]:
        """The distinct premise pairs of the binary rules.

        A rule whose side condition reads the split point is a pair of its
        own, since its sum over split points is its own.

        Returns
        -------
        tuple of tuple of int
            ``(left, right, own)``: the categories and the rule's position
            in `binary` when it is a pair of its own, else ``-1``.
        """
        found: dict[tuple[int, int, int], None] = {}
        for index, rule in enumerate(self.binary):
            own = index if rule.reads == "split" else -1
            found.setdefault((rule.left, rule.right, own), None)
        return tuple(found)

    def category(self, item: Item) -> int:
        """The category of an item of the family.

        Parameters
        ----------
        item : Item
            The item.

        Returns
        -------
        int
            Its category's position in `categories`.
        """
        key = tuple(str(item.arguments[slot]) for slot in self.category_slots)
        return self.categories.index(key)

    def positions(self, item: Item) -> tuple[int, int]:
        """The start and end of an item of the family.

        Parameters
        ----------
        item : Item
            The item.

        Returns
        -------
        tuple of int
            ``(i, j)``.
        """
        start, end = item.arguments[self.start], item.arguments[self.end]
        assert isinstance(start, int) and isinstance(end, int)
        return start, end

    def item(self, category: int, start: int, end: int) -> Item:
        """The item of a category over a span.

        Parameters
        ----------
        category : int
            The category's position.
        start : int
            The start.
        end : int
            The end.

        Returns
        -------
        Item
            The item.
        """
        width = len(self.category_slots) + 2
        arguments: list[Ground] = [0] * width
        for slot, value in zip(
            self.category_slots, self.categories[category], strict=True
        ):
            arguments[slot] = value
        arguments[self.start] = start
        arguments[self.end] = end
        return Item(family=self.family, arguments=tuple(arguments))


def _positions_of(judgment: Judgment, start: int, end: int) -> tuple[str, str] | None:
    """The position variables of a judgment pattern.

    Parameters
    ----------
    judgment : Judgment
        The pattern.
    start : int
        The argument position of the start.
    end : int
        The argument position of the end.

    Returns
    -------
    tuple of str or None
        The two variables, or ``None`` when either is not a variable.
    """
    first, second = judgment.arguments[start], judgment.arguments[end]
    if isinstance(first, IndexVariable) and isinstance(second, IndexVariable):
        return first.name, second.name
    return None


def span_grammar(presentation: RulePresentation) -> SpanGrammar:
    """The span form of a presentation.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.

    Returns
    -------
    SpanGrammar
        Its span form.

    Raises
    ------
    SpanRefused
        If the presentation is not in span form, with the first reason.
    """
    if len(presentation.judgments) != 1:
        raise SpanRefused("span form has one judgment family")
    (declaration,) = presentation.judgments
    naturals = [
        index for index, binder in enumerate(declaration.indices) if binder.sort == NAT
    ]
    if len(naturals) != 2:
        raise SpanRefused(
            "span form has two natural-number indices, a start and an end"
        )
    sorts: list[UserIndexSort] = []
    slots: list[int] = []
    for index, binder in enumerate(declaration.indices):
        if index in naturals:
            continue
        sort = binder.sort
        if not isinstance(sort, UserIndexSort) or any(sort.arities):
            raise SpanRefused(
                f"index {binder.name!r} is not of a closed sort of nullary constructors"
            )
        sorts.append(sort)
        slots.append(index)
    start, end = naturals
    categories = tuple(itertools.product(*(sort.constructors for sort in sorts)))
    family = declaration.id
    binary: list[SpanRule] = []
    unary: list[SpanRule] = []
    constants: list[SpanRule] = []

    def assignments(rule: Rule) -> list[dict[str, Ground]]:
        """Every assignment of the rule's category variables.

        Parameters
        ----------
        rule : Rule
            The rule.

        Returns
        -------
        list of dict of str to Ground
            The assignments.

        Raises
        ------
        SpanRefused
            If a category is neither a constant nor a variable.
        """
        names: dict[str, UserIndexSort] = {}
        for judgment in (rule.conclusion, *(p.judgment for p in rule.premises)):
            for slot, sort in zip(slots, sorts, strict=True):
                argument = judgment.arguments[slot]
                if isinstance(argument, IndexVariable):
                    names.setdefault(argument.name, sort)
                elif not isinstance(argument, IndexLiteral):
                    raise SpanRefused(
                        f"{rule.name}: a category is neither a constant nor a variable"
                    )
        keys = list(names)
        return [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*(names[key].constructors for key in keys))
        ]

    def category_of(judgment: Judgment, binding: Mapping[str, Ground]) -> int:
        """The category of a pattern under an assignment.

        Parameters
        ----------
        judgment : Judgment
            The pattern.
        binding : Mapping of str to Ground
            The category variables' values.

        Returns
        -------
        int
            The category's position.
        """
        key = tuple(
            str(evaluate_index(judgment.arguments[slot], binding)) for slot in slots
        )
        return categories.index(key)

    for rule in presentation.rules:
        judgments = (rule.conclusion, *(premise.judgment for premise in rule.premises))
        if any(judgment.family != family for judgment in judgments):
            raise SpanRefused(f"{rule.name}: a judgment of another family")
        if any(premise.hypotheses for premise in rule.premises):
            raise SpanRefused(f"{rule.name}: a premise has hypotheses")
        reads_positions: set[str] = set()
        if not rule.premises:
            conclusion = rule.conclusion
            if not all(
                isinstance(conclusion.arguments[slot], IndexLiteral)
                for slot in (start, end)
            ):
                raise SpanRefused(f"{rule.name}: a rule without premises is not ground")
            for binding in assignments(rule):
                constants.append(
                    SpanRule(
                        rule=rule,
                        binding=binding,
                        result=category_of(conclusion, binding),
                    )
                )
            continue
        spans = [_positions_of(judgment, start, end) for judgment in judgments]
        if any(span is None for span in spans):
            raise SpanRefused(f"{rule.name}: a position is not a variable")
        conclusion_span, *premise_spans = [span for span in spans if span is not None]
        condition = rule.side_condition
        arguments = set() if condition is None else set(condition.arguments)
        if len(premise_spans) == 1:
            if (
                premise_spans[0] != conclusion_span
                or conclusion_span[0] == conclusion_span[1]
            ):
                raise SpanRefused(f"{rule.name}: a unary rule changes its span")
            reads_positions = arguments & set(conclusion_span)
            kind = "span" if reads_positions else "none"
            for binding in assignments(rule):
                unary.append(
                    SpanRule(
                        rule=rule,
                        binding=binding,
                        result=category_of(rule.conclusion, binding),
                        left=category_of(rule.premises[0].judgment, binding),
                        reads=kind,
                        positions=conclusion_span,
                    )
                )
            continue
        if len(premise_spans) != 2:
            raise SpanRefused(f"{rule.name}: span form has at most two premises")
        (i, k), (k2, j) = premise_spans
        if k != k2 or conclusion_span != (i, j) or len({i, k, j}) != 3:
            raise SpanRefused(
                f"{rule.name}: a binary rule is not A(i, j) <- B(i, k), C(k, j)"
            )
        kind = "split" if k in arguments else "span" if arguments & {i, j} else "none"
        for binding in assignments(rule):
            binary.append(
                SpanRule(
                    rule=rule,
                    binding=binding,
                    result=category_of(rule.conclusion, binding),
                    left=category_of(rule.premises[0].judgment, binding),
                    right=category_of(rule.premises[1].judgment, binding),
                    reads=kind,
                    positions=(i, k, j),
                )
            )
    for slot in presentation.axioms:
        if slot.judgment.family != family:
            raise SpanRefused(f"input {slot.name} is of another family")
    edges: dict[int, set[int]] = {}
    for rule in unary:
        edges.setdefault(rule.left, set()).add(rule.result)
    return SpanGrammar(
        presentation=presentation,
        family=family,
        category_slots=tuple(slots),
        start=start,
        end=end,
        categories=categories,
        binary=tuple(binary),
        unary=tuple(unary),
        constants=tuple(constants),
        unary_cyclic=_cyclic(edges),
    )


def _cyclic(edges: Mapping[int, set[int]]) -> bool:
    """Whether a directed graph has a cycle.

    Parameters
    ----------
    edges : Mapping of int to set of int
        The successors of each node.

    Returns
    -------
    bool
        True when some node reaches itself.
    """
    state: dict[int, int] = {}
    for root in edges:
        if root in state:
            continue
        stack: list[tuple[int, list[int]]] = [(root, sorted(edges.get(root, ())))]
        state[root] = 1
        while stack:
            node, rest = stack[-1]
            if not rest:
                state[node] = 2
                stack.pop()
                continue
            nxt = rest.pop()
            if state.get(nxt) == 1:
                return True
            if nxt not in state:
                state[nxt] = 1
                stack.append((nxt, sorted(edges.get(nxt, ()))))
    return False


class SpanWork(ValueModel):
    """The work of a span run, in elements of the algebra.

    Parameters
    ----------
    products : int
        Binary products.
    sums : int
        Binary sums.
    pivots : int
        Closures of one element inside the unary closures.
    exact : int
        Operations of the closures' exact decisions.
    conditions : int
        Side conditions evaluated.
    allocated : int
        Elements of the tensors formed.
    peak : int
        Elements formed and held at once: the chart so far and one
        length's temporaries.
    """

    products: int = 0
    sums: int = 0
    pivots: int = 0
    exact: int = 0
    conditions: int = 0
    allocated: int = 0
    peak: int = 0


class _Count:
    """Running counts of a span run.

    Parameters
    ----------
    rank : int
        The number of element axes of the algebra.
    """

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.products = 0
        self.sums = 0
        self.pivots = 0
        self.exact = 0
        self.conditions = 0
        self.allocated = 0
        self.held = 0
        self.temporary = 0
        self.peak = 0

    def elements(self, tensor: torch.Tensor) -> int:
        """The elements of the algebra a tensor holds.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor, its element axes last.

        Returns
        -------
        int
            The product of its index extents.
        """
        return math.prod(tensor.shape[: tensor.dim() - self.rank])

    def made(self, tensor: torch.Tensor, *, kept: bool = False) -> torch.Tensor:
        """Record a tensor formed.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor.
        kept : bool
            Whether it is part of the chart, held to the end.

        Returns
        -------
        torch.Tensor
            The tensor.
        """
        size = self.elements(tensor)
        self.allocated += size
        if kept:
            self.held += size
        else:
            self.temporary += size
        self.peak = max(self.peak, self.held + self.temporary)
        return tensor

    def times(
        self, algebra: OrderPreservingSemiring, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """One counted product.

        Parameters
        ----------
        algebra : OrderPreservingSemiring
            The algebra.
        a, b : torch.Tensor
            The factors, ``a`` first.

        Returns
        -------
        torch.Tensor
            ``a ⊗ b``.
        """
        result = self.made(algebra.tensor_op(a, b))
        self.products += self.elements(result)
        return result

    def plus(
        self, algebra: OrderPreservingSemiring, terms: torch.Tensor, axis: int
    ) -> torch.Tensor:
        """One counted sum over an index axis.

        Parameters
        ----------
        algebra : OrderPreservingSemiring
            The algebra.
        terms : torch.Tensor
            The terms.
        axis : int
            The axis.

        Returns
        -------
        torch.Tensor
            The sum.
        """
        result = self.made(algebra.join(terms, axis))
        self.sums += max(self.elements(terms) - self.elements(result), 0)
        return result

    def report(self) -> SpanWork:
        """The counts.

        Returns
        -------
        SpanWork
            The counts.
        """
        return SpanWork(
            products=self.products,
            sums=self.sums,
            pivots=self.pivots,
            exact=self.exact,
            conditions=self.conditions,
            allocated=self.allocated,
            peak=self.peak,
        )


def _holds(
    grammar: SpanGrammar, rule: SpanRule, positions: tuple[int, ...], count: _Count
) -> bool:
    """Whether a rule instance's side condition holds at positions.

    Parameters
    ----------
    grammar : SpanGrammar
        The grammar.
    rule : SpanRule
        The instance.
    positions : tuple of int
        The values of its position variables, in order, or of its start
        and end alone when the condition reads no split point.
    count : _Count
        The counts.

    Returns
    -------
    bool
        Whether it holds.
    """
    count.conditions += 1
    binding: dict[str, Ground] = dict(rule.binding)
    names = rule.positions
    if len(positions) < len(names):
        names = (names[0], names[-1])
    binding.update(zip(names, positions, strict=True))
    return side_condition_holds(grammar.presentation, rule.rule, binding)


def _closure(
    grammar: SpanGrammar,
    algebra: OrderPreservingSemiring,
    rules: list[SpanRule],
    weights: list[torch.Tensor],
    like: torch.Tensor,
    count: _Count,
) -> torch.Tensor:
    """The closure ``U*`` of unary rules over the categories.

    ``U[b, a]`` is the sum of the weights of the rules deriving ``a`` from
    ``b``. With a cycle the closure is the algebra's star; without one,
    each row is ``U*[b] = e_b ⊕ U[b] U*`` for the categories in an order
    that puts every category after those it derives, which needs no star
    and costs ``N^3``.

    Parameters
    ----------
    grammar : SpanGrammar
        The grammar.
    algebra : OrderPreservingSemiring
        The algebra.
    rules : list of SpanRule
        The unary instances that apply.
    weights : list of torch.Tensor
        Their weights.
    like : torch.Tensor
        The dtype and device.
    count : _Count
        The counts.

    Returns
    -------
    torch.Tensor
        ``(N, N, *element_shape)``.

    Raises
    ------
    SpanRefused
        If the rules have a cycle and the algebra has no star.
    """
    size = len(grammar.categories)
    identity = algebra.identity_tensor((size,), dtype=like.dtype, device=like.device)
    if not rules:
        return identity
    stacked = torch.stack(weights)
    fibers: list[list[int]] = [[] for _ in range(size * size)]
    for position, rule in enumerate(rules):
        fibers[rule.left * size + rule.result].append(position)
    matrix = fiber_join(
        algebra, stacked, fibers, algebra.zero_like((1,), like), axis=0
    ).reshape(size, size, *stacked.shape[1:])
    if grammar.unary_cyclic:
        if not isinstance(algebra, Star):
            raise SpanRefused(
                "the unary rules have a cycle and the algebra has no star to sum it"
            )
        closure, work = algebra.star_with_work(matrix)
        count.pivots += work.pivots
        count.exact += work.exact
        count.products += work.products
        count.sums += work.sums
        return count.made(closure)
    successors: dict[int, set[int]] = {}
    for rule in rules:
        successors.setdefault(rule.left, set()).add(rule.result)
    finished: list[int] = []
    seen: set[int] = set()
    for root in range(size):
        stack: list[tuple[int, list[int]]] = []
        if root not in seen:
            seen.add(root)
            stack.append((root, sorted(successors.get(root, ()))))
        while stack:
            node, rest = stack[-1]
            if rest:
                nxt = rest.pop()
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, sorted(successors.get(nxt, ()))))
                continue
            stack.pop()
            finished.append(node)
    rows: dict[int, torch.Tensor] = {}
    zero_row = algebra.zero_like((size,), like)
    for node in finished:
        table = torch.stack([rows.get(other, zero_row) for other in range(size)])
        through = algebra.compose(matrix[node].unsqueeze(0), table, 1)[0]
        count.products += size * size
        count.sums += size * (size - 1)
        rows[node] = algebra.join_pair(identity[node], through)
        count.sums += size
    return count.made(torch.stack([rows[node] for node in range(size)]))


def _unary_closure(
    grammar: SpanGrammar,
    algebra: OrderPreservingSemiring,
    cells: torch.Tensor,
    length: int,
    unary: list[torch.Tensor],
    like: torch.Tensor,
    count: _Count,
    closures: dict[tuple[bool, ...], torch.Tensor],
) -> torch.Tensor:
    """Close every cell of one length under the unary rules: ``cell ⊗ U*``.

    Cells whose conditioned unary rules hold alike share one closure.

    Parameters
    ----------
    grammar : SpanGrammar
        The grammar.
    algebra : OrderPreservingSemiring
        The algebra.
    cells : torch.Tensor
        ``(m, N, *element_shape)``: the cells starting at ``0 .. m - 1``.
    length : int
        Their length.
    unary : list of torch.Tensor
        The unary instances' weights.
    like : torch.Tensor
        The dtype and device.
    count : _Count
        The counts.
    closures : dict
        The closures made so far, by which conditioned rules hold.

    Returns
    -------
    torch.Tensor
        The closed cells.
    """
    if not grammar.unary:
        return cells
    conditioned = [
        position for position, rule in enumerate(grammar.unary) if rule.reads != "none"
    ]
    groups: dict[tuple[bool, ...], list[int]] = {}
    for start in range(cells.shape[0]):
        key = tuple(
            _holds(grammar, grammar.unary[position], (start, start + length), count)
            for position in conditioned
        )
        groups.setdefault(key, []).append(start)
    results: dict[int, torch.Tensor] = {}
    for key, starts in groups.items():
        if key not in closures:
            holding = dict(zip(conditioned, key, strict=True))
            chosen = [
                position
                for position in range(len(grammar.unary))
                if holding.get(position, True)
            ]
            closures[key] = _closure(
                grammar,
                algebra,
                [grammar.unary[position] for position in chosen],
                [unary[position] for position in chosen],
                like,
                count,
            )
        index = torch.tensor(starts, dtype=torch.long, device=cells.device)
        chosen_cells = cells.index_select(0, index)
        closed = algebra.compose(chosen_cells, closures[key], 1)
        size = len(grammar.categories)
        count.products += len(starts) * size * size
        count.sums += len(starts) * size * (size - 1)
        for row, start in enumerate(starts):
            results[start] = closed[row]
    return count.made(torch.stack([results[start] for start in range(cells.shape[0])]))


def span_inside(
    grammar: SpanGrammar,
    algebra: OrderPreservingSemiring,
    seeds: Mapping[int, torch.Tensor],
    length: int,
    binary: list[torch.Tensor],
    unary: list[torch.Tensor],
    *,
    like: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], SpanWork]:
    """The inside values of every span, by span length.

    For each length ``l`` from one to the input's, the cells of length
    ``l`` are computed at once: for each distinct premise pair ``(B, C)``
    the products ``x(B, i, k) ⊗ x(C, k, i + l)`` are summed over the split
    points ``k`` first, then each binary rule multiplies its pair's sum by
    its weight, the rules are summed into their conclusion's category,
    the inputs of the length are added, and the cells are closed under the
    unary rules. Summing the split points before the rules is right
    distributivity, so the products keep the premises' order and the
    rule weight last, and no product over categories, rules and split
    points together is formed.

    Parameters
    ----------
    grammar : SpanGrammar
        The grammar.
    algebra : OrderPreservingSemiring
        The algebra.
    seeds : Mapping of int to torch.Tensor
        For each length, ``(n - l + 1, N, *element_shape)``: the inputs and
        constants over each span of that length, the zero elsewhere; a
        length without inputs is absent.
    length : int
        ``n``, the last position.
    binary : list of torch.Tensor
        The weight of each instance of `SpanGrammar.binary`.
    unary : list of torch.Tensor
        The weight of each instance of `SpanGrammar.unary`.
    like : torch.Tensor
        The dtype and device.

    Returns
    -------
    tuple of dict and SpanWork
        For each length ``l``, the cells ``(n - l + 1, N, *element_shape)``
        starting at ``0 .. n - l``; and the work.

    Raises
    ------
    SpanRefused
        If the unary rules have a cycle and the algebra has no star.
    """
    rank = len(algebra.element_shape)
    count = _Count(rank)
    size = len(grammar.categories)
    pairs = grammar.pairs
    pair_of = {pair: position for position, pair in enumerate(pairs)}
    rule_pair = [
        pair_of[(rule.left, rule.right, position if rule.reads == "split" else -1)]
        for position, rule in enumerate(grammar.binary)
    ]
    fibers: list[list[int]] = [[] for _ in range(size)]
    for position, rule in enumerate(grammar.binary):
        fibers[rule.result].append(position)
    lefts = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    rights = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    weights = torch.stack(binary) if binary else None
    closures: dict[tuple[bool, ...], torch.Tensor] = {}
    chart: dict[int, torch.Tensor] = {}
    for width in range(1, length + 1):
        rows = length - width + 1
        cells = seeds.get(width)
        if cells is None:
            cells = count.made(algebra.zero_like((rows, size), like))
        if width > 1 and pairs and weights is not None:
            left = count.made(
                torch.stack([chart[k][:rows] for k in range(1, width)], dim=1)
            )
            right = count.made(
                torch.stack(
                    [chart[width - k][k : k + rows] for k in range(1, width)], dim=1
                )
            )
            products = count.times(
                algebra,
                left.index_select(2, lefts.to(left.device)),
                right.index_select(2, rights.to(right.device)),
            )
            own = [position for position, pair in enumerate(pairs) if pair[2] >= 0]
            if own:
                mask = torch.ones(
                    (rows, width - 1, len(pairs)), dtype=torch.bool, device=like.device
                )
                for position in own:
                    rule = grammar.binary[pairs[position][2]]
                    for start in range(rows):
                        for split in range(1, width):
                            mask[start, split - 1, position] = _holds(
                                grammar,
                                rule,
                                (start, start + split, start + width),
                                count,
                            )
                shaped = mask.reshape(*mask.shape, *([1] * rank))
                products = torch.where(
                    shaped, products, algebra.zero_like(tuple(products.shape[:3]), like)
                )
            summed = count.plus(algebra, products, 1)
            gathered = summed.index_select(
                1, torch.tensor(rule_pair, dtype=torch.long, device=like.device)
            )
            weighted = count.times(algebra, gathered, weights.unsqueeze(0))
            spans = [
                position
                for position, rule in enumerate(grammar.binary)
                if rule.reads == "span"
            ]
            if spans:
                mask = torch.ones(
                    (rows, len(grammar.binary)), dtype=torch.bool, device=like.device
                )
                for position in spans:
                    for start in range(rows):
                        mask[start, position] = _holds(
                            grammar,
                            grammar.binary[position],
                            (start, start + width),
                            count,
                        )
                shaped = mask.reshape(*mask.shape, *([1] * rank))
                weighted = torch.where(
                    shaped, weighted, algebra.zero_like(tuple(weighted.shape[:2]), like)
                )
            derived = count.made(
                fiber_join(
                    algebra,
                    weighted,
                    fibers,
                    algebra.zero_like((rows, 1), like),
                    axis=1,
                )
            )
            widest = max((len(fiber) for fiber in fibers), default=0)
            count.sums += rows * size * max(widest - 1, 0)
            cells = count.made(algebra.join_pair(cells, derived))
            count.sums += rows * size
            count.temporary = 0
        closed = _unary_closure(
            grammar, algebra, cells, width, unary, like, count, closures
        )
        chart[width] = closed
        count.held += count.elements(closed)
        count.temporary = 0
    return chart, count.report()


def _boolean_closure(
    size: int, rules: list[SpanRule], device: torch.device
) -> torch.Tensor:
    """The reflexive transitive closure of unary rules, by squaring.

    Parameters
    ----------
    size : int
        The number of categories.
    rules : list of SpanRule
        The unary instances present.
    device : torch.device
        The device.

    Returns
    -------
    torch.Tensor
        ``(N, N)`` of 0 and 1 in float64: whether a chain of the rules
        derives the second category from the first, the empty one
        included.
    """
    closure = torch.eye(size, dtype=torch.float64, device=device)
    for rule in rules:
        closure[rule.left, rule.result] = 1.0
    steps = max(size - 1, 1).bit_length()
    for _ in range(steps):
        closure = (closure @ closure > 0).to(torch.float64)
    return closure


def span_recognize(
    grammar: SpanGrammar,
    seeds: Mapping[int, torch.Tensor],
    length: int,
    binary: list[bool],
    unary: list[bool],
    *,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], SpanWork]:
    """Which category is derivable over which span, by Boolean matrix products.

    For each length the pair matrix ``M[i][B, C] = OR_k x(B, i, k) AND
    x(C, k, i + l)`` is the product of two 0-1 matrices over the split
    points, computed as one batched matrix product and read as ``> 0``;
    each present binary rule reads its pair, and the cells are closed
    under the present unary rules by one product with their closure.
    The products are of counts below ``2^53``, so float64 holds them
    exactly.

    Parameters
    ----------
    grammar : SpanGrammar
        The grammar.
    seeds : Mapping of int to torch.Tensor
        For each length, ``(n - l + 1, N)`` Booleans: the inputs present.
    length : int
        ``n``.
    binary : list of bool
        Whether each binary instance is present.
    unary : list of bool
        Whether each unary instance is present.
    device : torch.device
        The device.

    Returns
    -------
    tuple of dict and SpanWork
        For each length, ``(n - l + 1, N)`` Booleans; and the work, a
        matrix product of sizes ``a b c`` counted as ``a b c`` products.
    """
    count = _Count(0)
    size = len(grammar.categories)
    chart: dict[int, torch.Tensor] = {}
    closures: dict[tuple[bool, ...], torch.Tensor] = {}
    conditioned = [
        position for position, rule in enumerate(grammar.unary) if rule.reads != "none"
    ]
    present = [rule for rule, kept in zip(grammar.binary, binary, strict=True) if kept]
    for width in range(1, length + 1):
        rows = length - width + 1
        cells = seeds.get(width)
        cells = (
            torch.zeros((rows, size), dtype=torch.float64, device=device)
            if cells is None
            else cells.to(torch.float64)
        )
        if width > 1 and present:
            left = torch.stack([chart[k][:rows] for k in range(1, width)], dim=1)
            right = torch.stack(
                [chart[width - k][k : k + rows] for k in range(1, width)], dim=1
            )
            pairs = torch.bmm(left.transpose(1, 2), right) > 0
            count.products += rows * size * size * (width - 1)
            derived = torch.zeros((rows, size), dtype=torch.float64, device=device)
            for rule in present:
                if rule.reads == "split":
                    hit = torch.tensor(
                        [
                            any(
                                bool(left[start, split - 1, rule.left])
                                and bool(right[start, split - 1, rule.right])
                                and _holds(
                                    grammar,
                                    rule,
                                    (start, start + split, start + width),
                                    count,
                                )
                                for split in range(1, width)
                            )
                            for start in range(rows)
                        ],
                        dtype=torch.bool,
                        device=device,
                    )
                else:
                    hit = pairs[:, rule.left, rule.right]
                if rule.reads == "span":
                    hit = hit & torch.tensor(
                        [
                            _holds(grammar, rule, (start, start + width), count)
                            for start in range(rows)
                        ],
                        dtype=torch.bool,
                        device=device,
                    )
                derived[:, rule.result] = torch.maximum(
                    derived[:, rule.result], hit.to(torch.float64)
                )
                count.sums += rows
            cells = torch.maximum(cells, derived)
        if grammar.unary:
            groups: dict[tuple[bool, ...], list[int]] = {}
            for start in range(rows):
                key = tuple(
                    _holds(grammar, grammar.unary[p], (start, start + width), count)
                    for p in conditioned
                )
                groups.setdefault(key, []).append(start)
            closed = cells.clone()
            for key, starts in groups.items():
                if key not in closures:
                    holding = dict(zip(conditioned, key, strict=True))
                    closures[key] = _boolean_closure(
                        size,
                        [
                            rule
                            for position, rule in enumerate(grammar.unary)
                            if unary[position] and holding.get(position, True)
                        ],
                        device,
                    )
                    count.products += size**3 * max(size - 1, 1).bit_length()
                index = torch.tensor(starts, dtype=torch.long, device=device)
                closed[index] = (cells[index] @ closures[key] > 0).to(torch.float64)
                count.products += len(starts) * size * size
            cells = closed
        chart[width] = cells
    return {width: cells > 0 for width, cells in chart.items()}, count.report()
