"""Sparse finite-state n-gram representation with original weight ownership."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import torch
from torch import nn

from quivers.core.algebras import LOG_PROB
from quivers.core.relations import RationalRelation, _read_weight_source, _TensorWeights
from quivers.core.relations._ngram import _NgramAutomaton
from quivers.core.relations._packed import _take
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import Arc, Boundary, RelationTopology

type _Feature = tuple[int, ...]
type _Slot = tuple[int, tuple[int, ...]]


class _NgramMessageWeights(nn.Module):
    def __init__(
        self,
        base: RationalRelation,
        coefficients: nn.Module,
        feature_count: int,
        slots: tuple[_Slot, ...],
    ) -> None:
        super().__init__()
        self._base = base
        self.source = base.module()
        self.coefficients = coefficients
        self._width = feature_count
        self._original = tuple(slot for slot, _ in slots)
        terms = [
            (slot, feature)
            for slot, (_, features) in enumerate(slots)
            for feature in features
        ]
        self._indices = tuple(slot for slot, _ in terms)
        self._features = tuple(feature for _, feature in terms)

    def forward(self) -> torch.Tensor:
        original = self._base._weights()
        coefficients = _read_weight_source(self.coefficients)
        if (
            not isinstance(coefficients, torch.Tensor)
            or coefficients.shape != (self._width,)
            or not coefficients.is_floating_point()
            or coefficients.device != original.device
            or coefficients.dtype != original.dtype
            or not bool(torch.isfinite(coefficients).all())
        ):
            raise ValueError(
                "n-gram coefficients must be finite and match the base dtype, device and feature count"
            )
        result = _take(original, self._original)
        indices = torch.tensor(self._indices, dtype=torch.long, device=original.device)
        bonuses = result.new_zeros(len(self._original)).index_add(
            0, indices, _take(coefficients, self._features)
        )
        if not bool(torch.isfinite(bonuses).all()):
            raise OverflowError("n-gram log bonuses are outside the input precision")
        tilted = result + bonuses
        if not bool((torch.isfinite(tilted) | torch.isneginf(result)).all()):
            raise OverflowError(
                "n-gram reweighted coefficients are outside the input precision"
            )
        return tilted


class NgramMessagePlan:
    """A deterministic n-gram score on a weighted regular language.

    The message multiplies the weight of each word by the exponential
    of its linear n-gram score. Every original path emits that same
    word score, including paths with epsilon arcs or shared slots.
    The resulting relation retains the original alphabet and support.

    Parameters
    ----------
    base : RationalRelation
        A one-tape relation with LogProb weights. The topology and
        original weight source are retained.
    features : tuple[tuple[int, ...], ...]
        Distinct, nonempty n-grams in coefficient order. Symbols are
        integers from the base tape's alphabet.
    max_states : int
        Maximum product states reached from initial marks.
    max_work : int
        Maximum trie and failure-link work, original transition
        examinations, and feature emissions in the reached product.
    """

    def __init__(
        self,
        base: RationalRelation,
        features: tuple[tuple[int, ...], ...],
        *,
        max_states: int,
        max_work: int,
    ) -> None:
        topology = base.topology
        if len(topology.tapes) != 1 or base.algebra is not LOG_PROB:
            raise TypeError("an n-gram message requires a one-tape LogProb relation")
        if max_states is None or max_work is None:
            raise ValueError("an n-gram message requires explicit construction limits")
        _validate_limits(max_states, max_work)
        size = topology.tapes[0].alphabet.cardinality
        features = tuple(map(tuple, features))
        if len(features) != len(set(features)) or any(
            not gram
            or any(type(symbol) is not int or not 0 <= symbol < size for symbol in gram)
            for gram in features
        ):
            raise ValueError(
                "n-gram message features must be distinct nonempty words in the base alphabet"
            )
        self._base = base
        self._features = features
        self._topology, self._slots = _compile(topology, features, max_states, max_work)

    @property
    def features(self) -> tuple[tuple[int, ...], ...]:
        """Active feature identities in coefficient order."""
        return self._features

    @property
    def n_states(self) -> int:
        """Reached product states in the compiled representation."""
        return self._topology.n_states

    def bind(self, coefficients: torch.Tensor | nn.Module) -> RationalRelation:
        """Expose the message relation with the original live coefficient source.

        A leaf tensor can be supplied directly. A module must return a
        vector on the base weight device and dtype. Query normalization
        applies the ordinary finite-state closure contract.
        """
        if isinstance(coefficients, torch.Tensor):
            coefficients = _TensorWeights(coefficients)
        if not isinstance(coefficients, nn.Module):
            raise TypeError("n-gram coefficients require a tensor or weight module")
        source = _NgramMessageWeights(
            self._base, coefficients, len(self.features), self._slots
        )
        return RationalRelation(self._topology, source, algebra=LOG_PROB)


def _compile(
    topology: RelationTopology,
    features: tuple[_Feature, ...],
    max_states: int,
    max_work: int,
) -> tuple[RelationTopology, tuple[_Slot, ...]]:
    if len(topology.initial) > max_states:
        raise RelationLimitError("n-gram message exceeds max_states")
    remaining = max_work

    def charge(amount: int) -> None:
        nonlocal remaining
        remaining -= amount
        if remaining < 0:
            raise RelationLimitError("n-gram message exceeds max_work")

    scorer = _NgramAutomaton(features, charge)
    outgoing: dict[int, list[Arc]] = defaultdict(list)
    for arc in topology.arcs:
        outgoing[arc.source].append(arc)
    states: list[tuple[int, int]] = [(mark.state, 0) for mark in topology.initial]
    identifiers = {state: i for i, state in enumerate(states)}
    slots: dict[_Slot, int] = {}

    def slot(base: int, matching: tuple[int, ...]) -> int:
        return slots.setdefault((base, matching), len(slots))

    initial = tuple(
        Boundary(i, slot(mark.weight, ())) for i, mark in enumerate(topology.initial)
    )
    arcs: list[Arc] = []
    final: list[Boundary] = []
    for source, (state, suffix) in enumerate(states):
        if (weight := topology._final_slots.get(state)) is not None:
            final.append(Boundary(source, slot(weight, ())))
        for arc in outgoing.get(state, ()):
            symbol = arc.labels[0]
            next_suffix = (
                suffix if symbol is None else scorer.step(suffix, symbol, charge)
            )
            matching = () if symbol is None else scorer.emissions[next_suffix]
            charge(1 + len(matching))
            config = arc.target, next_suffix
            target = identifiers.get(config)
            if target is None:
                target = len(states)
                if target == max_states:
                    raise RelationLimitError("n-gram message exceeds max_states")
                identifiers[config] = target
                states.append(config)
            arcs.append(Arc(source, target, arc.labels, slot(arc.weight, matching)))
    compiled = replace(
        topology,
        n_states=len(states),
        arcs=tuple(arcs),
        initial=initial,
        final=tuple(final),
        weight_names=tuple(f"feature-weight:{i}" for i in range(len(slots))),
    )
    return compiled, tuple(slots)


__all__ = ["NgramMessagePlan"]
