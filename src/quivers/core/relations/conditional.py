"""Globally normalized fixed-input queries of a rational relation."""

from __future__ import annotations

from collections.abc import Callable

import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import RationalRelation
from quivers.core.relations._centered import (
    CenteredStringLogScore,
    _CenteredBinding,
    _CenteredLogRatio,
)
from quivers.core.relations.paths import _conditional_log_ratio
from quivers.core.relations.queries import _validate_limits
from quivers.core.relations.topology import Words
from quivers.core.relations.transforms import condition


class ConditionalStringQuery:
    """A sparse conditional distribution over all output word tuples.

    The input words are fixed in tape order. The denominator sums every
    accepting path that agrees with them. An output query then sums
    only paths that also agree with its output words. Both sums use the
    same current weights and the same limits on productive closure.
    """

    def __init__(
        self,
        relation: RationalRelation,
        input_words: Words,
        *,
        max_states: int,
        max_work: int,
        max_closure_size: int | None = None,
    ) -> None:
        if relation.algebra is not MARKOV and relation.algebra is not LOG_PROB:
            raise TypeError("conditional string queries require Markov or LogProb")
        if max_states is None or max_work is None:
            raise ValueError(
                "conditional string queries require explicit state and work limits"
            )
        _validate_limits(max_states, max_work, max_closure_size)
        topology = relation.topology
        if not 0 < topology.input_tapes < len(topology.tapes):
            raise ValueError(
                "conditional string queries require input and output tapes"
            )
        if len(input_words) != topology.input_tapes:
            raise ValueError(
                "a conditional string query requires a word for each input tape"
            )
        for tape, word in reversed(tuple(enumerate(input_words))):
            relation = condition(
                relation, tape, word, max_states=max_states, max_work=max_work
            )
        self._relation = relation
        self._max_states = max_states
        self._max_work = max_work
        self._max_closure_size = max_closure_size
        self._mass = relation._total_query(max_states, max_work, max_closure_size)

    def normalizer(self) -> torch.Tensor:
        """Return the total output mass in the relation's algebra."""
        return self._mass.weight()

    def log_prob(self, output_words: Words) -> torch.Tensor:
        """Return the log probability of the supplied output word tuple."""
        query = self._relation.query(
            output_words,
            max_states=self._max_states,
            max_work=self._max_work,
            max_closure_size=self._max_closure_size,
        )
        values = self._relation._weights()
        if not values.is_floating_point():
            raise TypeError("conditional string queries require floating-point weights")
        algebra = self._relation.algebra
        mass = self._mass._plan.evaluate(values, algebra)
        score = query._plan.evaluate(values, algebra)
        return _conditional_log_ratio(score, mass, algebra)

    def centered_query(
        self, output_words: Words, *, max_center_occurrences: int, max_dyadic_work: int
    ) -> Callable[[], CenteredStringLogScore]:
        """Compile an opt-in centered binary64 LogProb likelihood.

        Both the fixed-output and all-output productive observations
        must be acyclic. Each call reads one live weight vector and
        uses its exact binary64 factors to remove the large absolute
        log masses before packed summation. The limits bound occurrence
        storage and a worst-case count of exact pass visits.
        """
        plan, weights = self._centered_binding(
            output_words,
            max_center_occurrences=max_center_occurrences,
            max_dyadic_work=max_dyadic_work,
        )

        def evaluate() -> CenteredStringLogScore:
            return plan.evaluate(weights())

        return evaluate

    def _centered_binding(
        self, output_words: Words, *, max_center_occurrences: int, max_dyadic_work: int
    ) -> _CenteredBinding:
        if self._relation.algebra is not LOG_PROB:
            raise TypeError("centered log ratios require LogProb relations")
        plan = _CenteredLogRatio.compile(
            self._relation.topology,
            self._mass._relation.topology,
            output_words,
            max_states=self._max_states,
            max_work=self._max_work,
            max_center_occurrences=max_center_occurrences,
            max_dyadic_work=max_dyadic_work,
        )

        return plan, self._relation._weights


__all__ = ["ConditionalStringQuery"]
