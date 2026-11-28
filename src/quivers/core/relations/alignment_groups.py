"""Bounded exact one-tape pairing for string alignment dual decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from quivers.core.relations.alignment_dual import (
    StringAlignment,
    StringAlignmentMapResult,
    _AlignmentSettings,
    _DualAlignmentRun,
    _LocalAlignments,
    _alignment_inputs,
    _run_alignment_map,
)
from quivers.core.relations.calculus import (
    _ArcOrigin,
    _JoinLayout,
    _materialize_product,
)
from quivers.core.relations.graph import StringFactorGraph
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import Words

type _Groups = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class StringAlignmentTrace:
    """One original factor's path and exact log score before dual shifts."""

    words: Words
    start: int
    arc_ids: tuple[int, ...]
    stop: int
    score: Fraction


@dataclass(frozen=True, slots=True)
class PairedAlignmentMapResult:
    """Grouped search and a feasible walk in each original factor.

    ``search`` carries the verified local certificate in group order,
    with multipliers on the group's retained tape positions. An available
    ``factor_traces`` tuple follows original factor order, including
    on an uncertified returned primal. ``product_shapes`` gives the
    constructed state and arc counts of each pair in group order.
    Singleton groups have no product shape.
    """

    groups: _Groups
    search: StringAlignmentMapResult
    factor_traces: tuple[StringAlignmentTrace, ...] | None
    product_shapes: tuple[tuple[int, int] | None, ...]


@dataclass(frozen=True, slots=True)
class _PairOrigins:
    arcs: tuple[_ArcOrigin, ...]
    boundaries: dict[int, tuple[int, int]]


def _validate_groups(groups: _Groups) -> tuple[int, ...]:
    if (
        not isinstance(groups, tuple)
        or not groups
        or any(
            not isinstance(group, tuple) or len(group) not in (1, 2) for group in groups
        )
    ):
        raise ValueError("alignment decomposition requires singleton or paired groups")
    factors = tuple(index for group in groups for index in group)
    if any(type(index) is not int or index < 0 for index in factors) or len(
        set(factors)
    ) != len(factors):
        raise ValueError(
            "alignment decomposition requires distinct nonnegative factor indices"
        )
    return factors


def _factor_pair(
    left: _LocalAlignments,
    right: _LocalAlignments,
    left_scope: tuple[str, ...],
    right_scope: tuple[str, ...],
    *,
    max_states: int,
    max_work: int,
) -> tuple[_LocalAlignments, tuple[str, ...], _PairOrigins]:
    shared = set(left_scope) & set(right_scope)
    if len(shared) != 1:
        raise ValueError("an alignment pair must share exactly one tape variable")
    (variable,) = shared
    left_tape, right_tape = left_scope.index(variable), right_scope.index(variable)
    layout = _JoinLayout.build(
        left._topology, right._topology, left_tape, right_tape, keep_shared=True
    )
    boundaries: dict[int, tuple[int, int]] = {}
    origins: list[_ArcOrigin] = []
    topology, pairs = _materialize_product(
        left._topology,
        right._topology,
        layout,
        max_states,
        max_work,
        arc_origins=origins,
        boundary_origins=boundaries,
    )
    # Product arc macros already use individual original slots. Boundary
    # pairs are added as exact rationals, before any float rounding.
    paired = []
    for left_slot, right_slot in pairs:
        first, second = left._weights[left_slot], right._weights[right_slot]
        paired.append(None if first is None or second is None else first + second)
    values = (*left._weights, *right._weights, *paired)
    scopes = left_scope, right_scope
    scope = tuple(scopes[side][tape] for side, tape in layout.visible)
    return (
        _LocalAlignments(topology, values),
        scope,
        _PairOrigins(tuple(origins), boundaries),
    )


def _factor_trace(
    decoder: _LocalAlignments,
    words: Words,
    start: int,
    ids: tuple[int, ...],
    stop: int,
) -> StringAlignmentTrace:
    start_slot = decoder._initial.get(start)
    stop_slot = decoder._final.get(stop)
    if start_slot is None or stop_slot is None:
        raise ArithmeticError(
            "expanded alignment boundaries are outside original support"
        )
    if any(not 0 <= index < len(decoder._topology.arcs) for index in ids):
        raise ArithmeticError("expanded alignment arc is outside original support")
    slots = (start_slot, *(decoder._topology.arcs[i].weight for i in ids), stop_slot)
    weights = tuple(decoder._weights[slot] for slot in slots)
    if None in weights:
        raise ArithmeticError("expanded alignment contains a zero original weight")
    score = sum((weight for weight in weights if weight is not None), Fraction(0))
    path = StringAlignment(words, start, ids, stop, score, score)
    decoder.check_trace(path, words, tuple({} for _ in words))
    return StringAlignmentTrace(words, start, ids, stop, score)


def _expand_paths(
    groups: _Groups,
    paths: tuple[StringAlignment, ...],
    origins: tuple[_PairOrigins | None, ...],
    decoders: tuple[_LocalAlignments, ...],
    scopes: tuple[tuple[str, ...], ...],
    assignment: tuple[tuple[str, tuple[int, ...]], ...],
) -> tuple[StringAlignmentTrace, ...]:
    words = dict(assignment)
    traces: dict[int, StringAlignmentTrace] = {}
    for group, path, origin in zip(groups, paths, origins, strict=True):
        if origin is None:
            (factor,) = group
            traces[factor] = StringAlignmentTrace(
                path.words, path.start, path.arc_ids, path.stop, path.base_score
            )
            continue
        ids: tuple[list[int], list[int]] = [], []
        for edge in path.arc_ids:
            side, index = origin.arcs[edge]
            if side not in (0, 1):
                raise ArithmeticError("grouped alignment arc has an unknown origin")
            ids[side].append(index)
        start = origin.boundaries.get(path.start)
        stop = origin.boundaries.get(path.stop)
        if start is None or stop is None:
            raise ArithmeticError("grouped alignment boundary has no original pair")
        score = Fraction(0)
        for side, factor in enumerate(group):
            observed = tuple(words[name] for name in scopes[factor])
            trace = _factor_trace(
                decoders[factor], observed, start[side], tuple(ids[side]), stop[side]
            )
            traces[factor] = trace
            score += trace.score
        if score != path.base_score:
            raise ArithmeticError(
                "grouped alignment score differs from its original factor paths"
            )
    return tuple(traces[index] for index in range(len(decoders)))


def paired_dual_alignment_map(
    graph: StringFactorGraph,
    *,
    groups: _Groups,
    max_group_states: int,
    max_group_work: int,
    max_iterations: int,
    max_ngram_order: int,
    max_features: int,
    max_states: int,
    max_work: int,
    step_size: Fraction | int = Fraction(1),
    absolute_gap: Fraction | int | None = None,
    relative_gap: Fraction | int | None = None,
) -> PairedAlignmentMapResult:
    """Run exact local alignment MAP with a declared singleton/pair partition.

    A pair shares exactly one nominal tape variable. An epsilon-filtered
    join enforces that variable's equality inside the pair. Other copies
    coordinate across groups using the same centered n-gram multipliers.
    Local search and independent verification use exact rational scores.

    Parameters
    ----------
    graph : StringFactorGraph
        The unchanged typed LogProb graph with positive productive DAG factors.
    groups : tuple of tuples of int
        Ordered singleton or paired groups partitioning original factor positions.
    max_group_states, max_group_work : int
        State and construction work limits for each filtered pair join.
    max_iterations, max_ngram_order, max_features : int
        Iteration and adaptive-feature limits for the common dual search.
    max_states, max_work : int
        Local state and per-pass arc limits for each decode and verification.
    step_size : Fraction or int
        Positive multiplier step divided by the current iteration.
    absolute_gap, relative_gap : Fraction or int or None
        Optional exact objective gap targets with the ordinary solver's meaning.

    Returns
    -------
    PairedAlignmentMapResult
        Verified grouped search with feasible original factor paths when found.

    Raises
    ------
    TypeError
        The graph does not have the supported LogProb capability.
    ValueError
        A group or limit is invalid, a pair has a different number of shared
        tapes, or a factor has cyclic positive support.
    RelationLimitError
        Either grouped construction or local decoding exceeds its budget.
    """
    indices = _validate_groups(groups)
    if set(indices) != set(range(len(graph._factors))):
        raise ValueError("alignment decomposition must partition all original factors")
    if max_group_states is None or max_group_work is None:
        raise ValueError(
            "alignment pair construction requires explicit state and work limits"
        )
    _validate_limits(max_group_states, max_group_work)
    settings = _AlignmentSettings(
        max_iterations,
        max_ngram_order,
        max_features,
        max_states,
        max_work,
        step_size,
        absolute_gap,
        relative_gap,
    )
    decoders, scopes, observations = _alignment_inputs(graph)
    grouped = []
    layouts = []
    origins = []
    shapes = []
    for group in groups:
        left = group[0]
        if len(group) == 1:
            grouped.append(decoders[left])
            layouts.append(scopes[left])
            origins.append(None)
            shapes.append(None)
            continue
        right = group[1]
        try:
            local, scope, origin = _factor_pair(
                decoders[left],
                decoders[right],
                scopes[left],
                scopes[right],
                max_states=max_group_states,
                max_work=max_group_work,
            )
        except RelationLimitError as error:
            raise RelationLimitError(
                f"alignment pair {group} product: {error}"
            ) from error
        grouped.append(local)
        layouts.append(scope)
        origins.append(origin)
        shapes.append((local._topology.n_states, len(local._topology.arcs)))
    run = _DualAlignmentRun(tuple(grouped), tuple(layouts), max_states, max_work)
    search, paths = _run_alignment_map(run, observations, settings)
    original = None
    if paths is not None:
        assert search.assignment is not None
        original = _expand_paths(
            groups, paths, tuple(origins), decoders, scopes, search.assignment
        )
    return PairedAlignmentMapResult(groups, search, original, tuple(shapes))


__all__ = [
    "PairedAlignmentMapResult",
    "StringAlignmentTrace",
    "paired_dual_alignment_map",
]
