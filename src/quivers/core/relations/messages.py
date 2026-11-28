"""Finite candidate views of unary weighted string messages."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.objects import StringSet
from quivers.core.relations import RationalRelation, _weight_snapshot
from quivers.core.relations.graph import _IndicatorWeights
from quivers.core.relations.paths import _require_normalizer
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import Arc, Boundary, RelationTopology
from quivers.core.relations.transforms import intersect


type _Word = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StringCandidateMessage:
    """A restricted weighted message and its exact full-language scale.

    The selected words and log weights use the order of the supplied
    candidates. The omitted mass includes every other word. Status
    records whether that omitted mass is zero. A positive-weight tie
    across the selection boundary is reported separately.
    """

    tape: StringSet
    words: tuple[_Word, ...]
    log_weights: torch.Tensor
    log_total_mass: torch.Tensor
    log_omitted_mass: torch.Tensor
    status: Literal["exact", "truncated"]
    boundary_tie: bool

    def normalized_log_weights(self) -> torch.Tensor:
        """Normalize the retained candidates, requiring positive retained mass."""
        scale = LOG_PROB.join(self.log_weights, dim=0)
        _require_normalizer(scale, LOG_PROB)
        return self.log_weights - scale


def _complement(
    relation: RationalRelation, words: tuple[_Word, ...], max_states: int, max_work: int
) -> RationalRelation:
    prefixes: list[dict[int, int]] = [{}]
    excluded: set[int] = set()
    if sum(len(word) + 1 for word in words) > max_work:
        raise RelationLimitError("candidate indexing exceeds max_work")
    for word in words:
        relation.topology._validate_words((word,))
        state = 0
        for symbol in word:
            outgoing = prefixes[state]
            if symbol not in outgoing:
                outgoing[symbol] = len(prefixes)
                prefixes.append({})
                if len(prefixes) + 1 > max_states:
                    raise RelationLimitError("candidate complement exceeds max_states")
            state = outgoing[symbol]
        excluded.add(state)

    sink = len(prefixes)
    prefixes.append({})
    tape = relation.topology.tapes[0]
    alphabet_size = tape.alphabet.cardinality
    if len(prefixes) > max_states:
        raise RelationLimitError("candidate complement exceeds max_states")
    if len(prefixes) * alphabet_size > max_work:
        raise RelationLimitError("candidate complement exceeds max_work")
    topology = RelationTopology(
        (tape,),
        relation.topology.input_tapes,
        len(prefixes),
        tuple(
            Arc(state, outgoing.get(symbol, sink), (symbol,), 0)
            for state, outgoing in enumerate(prefixes)
            for symbol in range(alphabet_size)
        ),
        (Boundary(0, 0),),
        tuple(
            Boundary(state, 0)
            for state in range(len(prefixes))
            if state not in excluded
        ),
        ("unit",),
    )
    return RationalRelation(topology, _IndicatorWeights(relation), algebra=LOG_PROB)


class StringCandidatePlan:
    """Compile a finite candidate view of a unary LogProb relation.

    A deterministic complement of the candidate language sums the mass
    of all other words. An optional beam ranks the supplied candidates
    by their marginal word weights, with lexicographic ties. The beam
    selection is discrete. Differentiation follows the scores and the
    full and omitted masses for the selected support.

    Parameters
    ----------
    relation : RationalRelation
        A unary weighted relation supplying the original message.
    words : tuple of words
        Distinct candidate words on that relation's tape.
    max_states : int
        State limit for each construction and query.
    max_work : int
        Work limit for each construction and query.
    max_closure_size : int or None
        Maximum productive local closure size, if admitted.
    """

    def __init__(
        self,
        relation: RationalRelation,
        words: tuple[_Word, ...],
        *,
        max_states: int,
        max_work: int,
        max_closure_size: int | None = None,
    ) -> None:
        if relation.algebra is not LOG_PROB or len(relation.topology.tapes) != 1:
            raise TypeError("a string candidate plan requires one LogProb tape")
        if max_states is None or max_work is None:
            raise ValueError("string candidates require explicit state and work limits")
        _validate_limits(max_states, max_work, max_closure_size)
        if len(set(words)) != len(words):
            raise ValueError("string message candidates must be distinct")
        self._words = tuple(words)
        self._tape = relation.topology.tapes[0]
        complementary = intersect(
            relation,
            _complement(relation, self._words, max_states, max_work),
            max_states=max_states,
            max_work=max_work,
        )
        self._complement = complementary._total_query(
            max_states, max_work, max_closure_size
        )
        self._scores = tuple(
            relation.query(
                (word,),
                max_states=max_states,
                max_work=max_work,
                max_closure_size=max_closure_size,
            )
            for word in self._words
        )

    def evaluate(self, *, beam: int | None = None) -> StringCandidateMessage:
        """Return a typed candidate message, including the full and omitted masses.

        Parameters
        ----------
        beam : int or None
            The number of highest-weight words to retain from the
            supplied list. None retains the complete list.

        Returns
        -------
        StringCandidateMessage
            The scores, scale, and truncation diagnostics for live weights.

        Raises
        ------
        ValueError
            If the beam is invalid or the full message has no finite
            positive normalizer.
        """
        if beam is not None and (type(beam) is not int or beam < 0):
            raise ValueError("the string message beam must be a nonnegative integer")
        with _weight_snapshot():
            outside = self._complement.weight()
            scores = (
                torch.stack([query.weight() for query in self._scores])
                if self._scores
                else outside.new_empty((0,))
            )
        return _restrict_candidates(self._tape, self._words, scores, outside, beam)[1]


def _restrict_candidates(
    tape: StringSet,
    words: tuple[_Word, ...],
    scores: torch.Tensor,
    outside: torch.Tensor,
    beam: int | None,
) -> tuple[list[int], StringCandidateMessage]:
    # The complement and within-candidate discards have disjoint support.
    log_total = LOG_PROB.join(torch.cat((outside.reshape(1), scores)), dim=0)
    _require_normalizer(log_total, LOG_PROB)
    selected = list(range(len(words)))
    boundary_tie = False
    omitted = outside
    if beam is not None and beam < len(words):
        values: list[float] = scores.detach().tolist()
        order = sorted(selected, key=lambda i: (-values[i], words[i]))
        used = sorted(order[:beam])
        excluded = order[beam:]
        omitted = LOG_PROB.join(
            torch.cat((outside.reshape(1), scores[excluded])), dim=0
        )
        if beam:
            last = values[order[beam - 1]]
            boundary_tie = isfinite(last) and last == values[order[beam]]
        selected = used
    return selected, StringCandidateMessage(
        tape=tape,
        words=tuple(words[i] for i in selected),
        log_weights=scores[selected],
        log_total_mass=log_total,
        log_omitted_mass=omitted,
        status="exact" if bool(torch.isneginf(omitted)) else "truncated",
        boundary_tie=boundary_tie,
    )


__all__ = ["StringCandidateMessage", "StringCandidatePlan"]
