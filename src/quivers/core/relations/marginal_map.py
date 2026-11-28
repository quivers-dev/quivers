"""Exact alignment-marginal ranking over declared output candidates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import RationalRelation
from quivers.core.relations._packed import _pack_paths
from quivers.core.relations.paths import _markov_log_weights
from quivers.core.relations.queries import (
    ProductiveEpsilonCycleError,
    _Observation,
    _ProductiveGraph,
    _observe,
    _productive_graph,
    _validate_limits,
)
from quivers.core.relations.topology import Words


@dataclass(frozen=True, slots=True)
class MarginalOutputChoice:
    """One positive candidate ranked by its exact alignment sum.

    The log pair weight remains differentiable for this candidate.
    Selection itself is a discrete operation.
    """

    output: Words
    exact_weight: Fraction
    log_pair_weight: torch.Tensor


def _weight(
    graph: _ProductiveGraph,
    final: list[tuple[int, int]],
    values: tuple[Fraction, ...],
) -> Fraction:
    mass = dict.fromkeys(graph.states, Fraction(0))
    for state, slot in graph.initial:
        mass[state] += values[slot]
    for state in graph.order:
        origin = mass[state]
        for target, slot in graph.arcs[state]:
            mass[target] += origin * values[slot]
    return sum((mass[state] * values[slot] for state, slot in final), Fraction(0))


def best_marginal_output(
    relation: RationalRelation,
    inputs: Words,
    candidates: tuple[Words, ...],
    *,
    max_states: int,
    max_work: int,
) -> MarginalOutputChoice | None:
    """Maximize the total alignment weight over explicit output candidates.

    With inputs held fixed, maximizing pair weight is equivalent to
    maximizing conditional probability whenever its normalizer exists
    and is positive. This operation returns an unnormalized pair score.

    Parameters
    ----------
    relation : RationalRelation
        A Markov relation with at least one input and one output tape.
    inputs : Words
        One fixed word per input tape in tape order.
    candidates : tuple[Words, ...]
        Word tuples for the output tapes. The first maximum wins ties.
    max_states : int
        Total reached configurations across the candidate queries.
    max_work : int
        Total examined transitions across the candidate queries.

    Returns
    -------
    MarginalOutputChoice or None
        The candidate with greatest positive alignment sum, or None if
        all candidates have zero mass. Exact rational comparison uses
        the original binary coefficients. The selected pair's log sum
        uses a packed log-space evaluation with live gradients.

    Raises
    ------
    TypeError
        If the relation is outside the supported algebra, tape, or
        weight domains.
    ValueError
        If the words or limits are invalid.
    RelationLimitError
        If cumulative candidate observation exceeds a construction
        limit.
    ProductiveEpsilonCycleError
        If a candidate has a productive cyclic observation graph.
    """
    topology = relation.topology
    if relation.algebra is not MARKOV:
        raise TypeError("exact marginal output ranking requires Markov weights")
    if not 0 < topology.input_tapes < len(topology.tapes):
        raise TypeError("marginal output ranking requires input and output tapes")
    if max_states is None or max_work is None:
        raise ValueError(
            "marginal output ranking requires explicit state and work limits"
        )
    _validate_limits(max_states, max_work)
    if len(inputs) != topology.input_tapes:
        raise ValueError("marginal output ranking requires one word per input tape")
    topology._validate_words((*inputs, *((),) * (len(topology.tapes) - len(inputs))))
    if not candidates:
        return None

    weights = relation._weights()
    if not weights.is_floating_point():
        raise TypeError("marginal output ranking requires floating-point weights")
    values = tuple(map(Fraction, weights.detach().tolist()))
    best: tuple[Words, _Observation, _ProductiveGraph] | None = None
    best_weight = Fraction(0)
    for output in candidates:
        words = (*inputs, *output)
        topology._validate_words(words)
        observation = _observe(
            topology,
            words,
            max_states=max_states,
            max_work=max_work,
        )
        max_states -= len(observation.states)
        max_work -= observation.examined
        graph = _productive_graph(
            observation.arcs, observation.initial, observation.final
        )
        if len(graph.order) != len(graph.states):
            raise ProductiveEpsilonCycleError(
                "exact marginal output ranking requires acyclic productive observations"
            )
        total = _weight(graph, observation.final, values)
        if total > best_weight:
            best = tuple(map(tuple, output)), observation, graph
            best_weight = total
    if best is None:
        return None

    output, observation, graph = best
    plan = _pack_paths(graph.order, graph.arcs, graph.initial, observation.final)
    log_score = plan.evaluate(_markov_log_weights(weights), LOG_PROB)
    if not bool(torch.isfinite(log_score)):
        raise OverflowError(
            "selected log pair weight is outside the evaluation precision"
        )
    return MarginalOutputChoice(output, best_weight, log_score)


__all__ = ["MarginalOutputChoice", "best_marginal_output"]
