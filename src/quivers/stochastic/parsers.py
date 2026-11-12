"""Chart parsers as weighted deductive systems.

A ``ChartParser`` is a ``DeductiveSystem`` that composes:

- ``LexicalAxiom``: populates length-1 spans from a learnable lexicon.
- ``BinarySpanDeduction``: applies binary structural rules.
- ``UnarySpanDeduction``: closes each cell under the unary rules.
- ``SpanGoal``: extracts the start-symbol score for the full span.
- ``CKYSchedule``: bottom-up evaluation by span length.

Concrete grammar formalisms are specific choices of ``RuleSchema``:

- ``CCG = EVALUATION | HARMONIC_COMPOSITION | CROSSED_COMPOSITION``
- ``LAMBEK = EVALUATION | ADJUNCTION_UNITS | TENSOR_INTRODUCTION | TENSOR_PROJECTION``
- ``NL = EVALUATION``

The ``ChartParser`` can also be constructed directly from a ``RuleSystem``
or from a ``RuleSchema`` + ``CategorySystem``.

Categorical perspective
-----------------------
A chart parser implements a morphism

    parse : FreeMonoid(T) -> 1

in the category of measurable spaces, mapping variable-length terminal
strings to their log-probability under the grammar. The ``RuleSchema``
specifies the structural rules (the generating arrows of the free
category), while the lexicon provides the interpretation functor
mapping terminals to weighted category distributions.

The parsing algorithm is parameterized by a ``ChartSemiring``
(Goodman, 1999), making the parser agnostic to the scoring algebra.
"""

from __future__ import annotations

from typing import cast

import torch

from quivers.stochastic.categories import (
    AtomicCategory,
    Category,
    CategorySystem,
)
from quivers.stochastic._rule_system import RuleSystem
from quivers.stochastic.deduction import DeductiveSystem
from quivers.stochastic.schema import RuleSchema
from quivers.stochastic.semiring import ChartSemiring, LOG_PROB
from quivers.stochastic.span import (
    BinarySpanDeduction,
    CKYSchedule,
    LexicalAxiom,
    SpanGoal,
    UnarySpanDeduction,
)


class ChartParser(DeductiveSystem):
    """Differentiable CKY chart parser as a weighted deductive system.

    Composes ``LexicalAxiom``, ``BinarySpanDeduction``,
    ``UnarySpanDeduction``, ``SpanGoal``, and ``CKYSchedule``
    into a single ``DeductiveSystem`` (nn.Module).

    Parameters
    ----------
    rule_system : RuleSystem
        The structural rules governing the grammar.
    n_terminals : int
        Vocabulary size.
    start_idx : int
        Index of the start category in the rule system.
    category_system : CategorySystem or None
        Optional reference to the category system (for introspection).
    semiring : ChartSemiring or None
        Scoring algebra. Defaults to log-probability semiring.
    learnable_rule_weights : bool
        Whether rule weights are learnable parameters.
    """

    def __init__(
        self,
        rule_system: RuleSystem,
        n_terminals: int,
        start_idx: int,
        category_system: CategorySystem | None = None,
        semiring: ChartSemiring | None = None,
        learnable_rule_weights: bool = True,
    ) -> None:
        axiom = LexicalAxiom(n_terminals, rule_system.n_categories)
        binary_ded = BinarySpanDeduction(
            rule_system,
            learnable=learnable_rule_weights,
        )
        unary_ded = UnarySpanDeduction(
            rule_system,
            learnable=learnable_rule_weights,
        )
        goal = SpanGoal(start_idx)
        schedule = CKYSchedule()

        super().__init__(
            axiom=axiom,
            deductions=[binary_ded, unary_ded],
            goal=goal,
            schedule=schedule,
            semiring=semiring,
        )

        self._rule_system = rule_system
        self._system = category_system
        self._n_term = n_terminals
        self._n_cat = rule_system.n_categories
        self._start_idx = start_idx

    @classmethod
    def from_schema(
        cls,
        schema: RuleSchema,
        category_system: CategorySystem,
        n_terminals: int,
        start: str | Category = "S",
        semiring: ChartSemiring | None = None,
        learnable_rule_weights: bool = True,
    ) -> ChartParser:
        """Create a chart parser from a rule schema and category system.

        Parameters
        ----------
        schema : RuleSchema
            The composable rule schema (e.g. ``CCG``, ``LAMBEK``).
        category_system : CategorySystem
            The category inventory.
        n_terminals : int
            Vocabulary size.
        start : str or Category
            The start category.
        semiring : ChartSemiring or None
            Scoring algebra.
        learnable_rule_weights : bool
            Whether rule weights are learnable.

        Returns
        -------
        ChartParser
            A new parser instance.
        """
        rule_system = schema(category_system)

        return cls.from_category_system(
            category_system,
            rule_system,
            n_terminals,
            start,
            semiring,
            learnable_rule_weights,
        )

    @classmethod
    def from_category_system(
        cls,
        category_system: CategorySystem,
        rule_system: RuleSystem,
        n_terminals: int,
        start: str | Category = "S",
        semiring: ChartSemiring | None = None,
        learnable_rule_weights: bool = True,
    ) -> ChartParser:
        """Create a chart parser from a category system and rules.

        Parameters
        ----------
        category_system : CategorySystem
            The category inventory.
        rule_system : RuleSystem
            The structural rules.
        n_terminals : int
            Vocabulary size.
        start : str or Category
            The start category.
        semiring : ChartSemiring or None
            Scoring algebra.
        learnable_rule_weights : bool
            Whether rule weights are learnable.

        Returns
        -------
        ChartParser
            A new parser instance.

        Raises
        ------
        ValueError
            If ``start`` is not in ``category_system``.
        """
        if isinstance(start, str):
            start = AtomicCategory(name=start)

        if start not in category_system:
            raise ValueError(f"start category {start!r} not in category system")

        return cls(
            rule_system=rule_system,
            n_terminals=n_terminals,
            start_idx=category_system.index(start),
            category_system=category_system,
            semiring=semiring,
            learnable_rule_weights=learnable_rule_weights,
        )

    @property
    def rule_system(self) -> RuleSystem:
        """The structural rule system."""
        return self._rule_system

    @property
    def category_system(self) -> CategorySystem | None:
        """The category inventory (if available)."""
        return self._system

    @property
    def n_rules(self) -> int:
        """Number of binary combination rules."""
        return self._rule_system.n_binary

    @property
    def n_unary_rules(self) -> int:
        """Number of unary rules."""
        return self._rule_system.n_unary

    @property
    def log_lexicon(self) -> torch.Tensor:
        """Log-probabilities of lexical category assignments.

        Returns
        -------
        torch.Tensor
            Shape ``(n_terminals, n_categories)``.
        """
        return cast(LexicalAxiom, self.axiom).log_lexicon

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute sentence scores via CKY chart parsing.

        Parameters
        ----------
        input : torch.Tensor
            Integer tensor of terminal indices. Shape
            ``(batch, seq_len)`` or ``(seq_len,)`` for a single
            sentence.

        Returns
        -------
        torch.Tensor
            Score of each sentence under the grammar.

        Raises
        ------
        ValueError
            If the sentences are empty.
        """
        squeeze = False

        if input.dim() == 1:
            input = input.unsqueeze(0)
            squeeze = True

        if input.shape[1] == 0:
            raise ValueError("cannot parse empty sentences")

        result = super().forward(input)

        if squeeze:
            return result.squeeze(0)

        return result

    def inside_chart(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute the full inside chart (for analysis/debugging).

        Parameters
        ----------
        tokens : torch.Tensor
            Integer tensor of terminal indices.

        Returns
        -------
        torch.Tensor
            Full chart of shape ``(batch, N, seq_len, seq_len+1)``.

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

        # run axiom + schedule but skip goal extraction
        chart = self.axiom(tokens, self._semiring)
        chart = self._schedule.run(chart, self.deductions, self._semiring)

        if squeeze:
            return chart.squeeze(0)

        return chart

    def __repr__(self) -> str:
        semiring_desc = (
            f", semiring={self._semiring!r}"
            if not isinstance(self._semiring, type(LOG_PROB))
            else ""
        )
        rule_desc = (
            f" [{self._rule_system.description}]"
            if self._rule_system.description
            else ""
        )

        return (
            f"ChartParser(categories={self._n_cat}, "
            f"terminals={self._n_term}, "
            f"binary_rules={self.n_rules}, "
            f"unary_rules={self.n_unary_rules}"
            f"{semiring_desc}{rule_desc})"
        )


__all__ = [
    "ChartParser",
]
