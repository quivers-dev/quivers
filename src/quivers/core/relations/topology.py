"""Topology for finite-state relations with ordered, typed tapes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from quivers.core.objects import StringSet

type Label = int | None
"""A symbol index or ``None`` for epsilon."""

type Words = tuple[tuple[int, ...], ...]
"""One word on each tape in tape order."""

EPSILON = None
"""An arc with epsilon on a tape consumes no symbol on that tape."""


@dataclass(frozen=True, slots=True)
class Arc:
    """One arc occurrence, identified by its position in the arc table."""

    source: int
    target: int
    labels: tuple[Label, ...]
    weight: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))


@dataclass(frozen=True, slots=True)
class Boundary:
    """An initial or final state with a weight slot."""

    state: int
    weight: int


@dataclass(frozen=True, slots=True)
class RelationTopology:
    """Finite directed multigraph independent of its weight values.

    State and symbol IDs start at zero. Tape IDs are positions in
    ``tapes``. The first ``input_tapes`` tapes make up the domain
    and the other tapes make up the codomain. Arc IDs are positions in
    ``arcs``, which can include multiple identical arc records.
    The positions in ``weight_names`` are weight slots. Arc and boundary
    occurrences can refer to the same slot.
    """

    tapes: tuple[StringSet, ...]
    input_tapes: int
    n_states: int
    arcs: tuple[Arc, ...]
    initial: tuple[Boundary, ...]
    final: tuple[Boundary, ...]
    weight_names: tuple[str, ...]
    _final_slots: Mapping[int, int] = field(init=False, repr=False, compare=False)
    _by_label: Mapping[tuple[int, int, int], tuple[int, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("tapes", "arcs", "initial", "final", "weight_names"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.tapes or not all(
            isinstance(tape, StringSet) for tape in self.tapes
        ):
            raise TypeError("a relation must have one or more StringSet tapes")
        if type(self.input_tapes) is not int or not 0 <= self.input_tapes <= len(
            self.tapes
        ):
            raise ValueError("input_tapes must be between zero and the tape count")
        if type(self.n_states) is not int or self.n_states < 0:
            raise ValueError("n_states must be a nonnegative integer")
        if any(not isinstance(name, str) or not name for name in self.weight_names):
            raise ValueError("weight slot names must be nonempty strings")
        if len(set(self.weight_names)) != len(self.weight_names):
            raise ValueError("weight slot names must be distinct")
        for boundary in (self.initial, self.final):
            if len({mark.state for mark in boundary}) != len(boundary):
                raise ValueError(
                    "a state may occur at most once in each boundary table"
                )
            for mark in boundary:
                self._check_state(mark.state)
                self._check_weight(mark.weight)
        object.__setattr__(
            self,
            "_final_slots",
            MappingProxyType({mark.state: mark.weight for mark in self.final}),
        )
        alphabet_sizes = tuple(tape.alphabet.cardinality for tape in self.tapes)
        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for arc_id, arc in enumerate(self.arcs):
            self._check_state(arc.source)
            self._check_state(arc.target)
            self._check_weight(arc.weight)
            if len(arc.labels) != len(self.tapes):
                raise ValueError("an arc must have exactly one label per tape")
            anchor = (-1, -1)
            for tape, (label, size) in enumerate(zip(arc.labels, alphabet_sizes)):
                if label is EPSILON:
                    continue
                if type(label) is not int or not 0 <= label < size:
                    raise ValueError(f"invalid symbol {label!r} on tape {tape}")
                if anchor == (-1, -1):
                    anchor = (tape, label)
            groups[arc.source, *anchor].append(arc_id)
        object.__setattr__(
            self,
            "_by_label",
            MappingProxyType(
                {
                    key: tuple(sorted(ids, key=self._encoded_labels))
                    for key, ids in groups.items()
                }
            ),
        )

    def _check_state(self, state: int) -> None:
        if type(state) is not int or not 0 <= state < self.n_states:
            raise ValueError(f"invalid state ID {state!r}")

    def _check_weight(self, weight: int) -> None:
        if type(weight) is not int or not 0 <= weight < len(self.weight_names):
            raise ValueError(f"invalid weight slot {weight!r}")

    def _encoded_labels(self, arc_id: int) -> tuple[int, ...]:
        return tuple(
            -1 if label is None else label for label in self.arcs[arc_id].labels
        )

    def _validate_words(self, words: Words) -> None:
        if len(words) != len(self.tapes):
            raise ValueError("a query must supply exactly one word per tape")
        for index, (word, tape) in enumerate(zip(words, self.tapes)):
            size = tape.alphabet.cardinality
            if any(
                type(symbol) is not int or not 0 <= symbol < size for symbol in word
            ):
                raise ValueError(
                    f"query word contains an invalid symbol on tape {index}"
                )

    def _candidate_arcs(
        self, state: int, words: Words, offsets: tuple[int, ...]
    ) -> Iterator[int]:
        # Index each arc once by its first non-epsilon label. This includes
        # all compatible arcs without double-counting arcs with several labels.
        yield from self._by_label.get((state, -1, -1), ())
        for tape, (word, offset) in enumerate(zip(words, offsets)):
            if offset < len(word):
                yield from self._by_label.get((state, tape, word[offset]), ())


__all__ = [
    "Arc",
    "Boundary",
    "EPSILON",
    "Label",
    "RelationTopology",
    "Words",
]
