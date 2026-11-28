"""Compare exact joint alignment MAP with a materialized weighted intersection.

Run `python -m tests.benchmarks.string_alignment_map` from the repository root.
The timed calls use CPU float64 weights and fresh search on the same
unchanged fixture. Enumeration, certificate checks, and parameter updates
are separate from timing.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from graphlib import TopologicalSorter
from itertools import product
from statistics import median
from time import perf_counter
from typing import Literal
from unittest.mock import patch

import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    StringAlignment,
    StringAlignmentMapResult,
    StringFactor,
    StringFactorGraph,
    StringVariable,
    dual_alignment_map,
    intersect,
)
from quivers.core.relations import alignment_dual, calculus

TAPE = StringSet(alphabet=FinSet(name="AlignmentBenchmarkAlphabet", cardinality=2))
WORDS = ((0, 1), (1, 0))
MAX_STATES = 100_000
MAX_WORK = 1_000_000
DEFAULT_WIDTHS = (2, 8, 24, 48)
SETTINGS = {
    "max_iterations": 8,
    "max_ngram_order": 2,
    "max_features": 4,
    "max_states": MAX_STATES,
    "max_work": MAX_WORK,
    "step_size": Fraction(3, 2),
}

type Word = tuple[int, ...]
type Mode = Literal["agree", "adaptive"]
type PathKey = tuple[Word, int, tuple[int, ...], int, Fraction]
type Shifts = tuple[tuple[Mapping[Word, Fraction], ...], ...]


@dataclass(frozen=True)
class Fixture:
    left: RationalRelation
    right: RationalRelation
    graph: StringFactorGraph
    right_scores: nn.Parameter


def _factor(
    width: int, scores: tuple[float, float], *, leading_epsilon: bool
) -> tuple[RationalRelation, nn.Parameter]:
    weights = nn.Parameter(
        torch.tensor(
            [0.0, *scores, *(-float(rank // 2) / 64 for rank in range(width))],
            dtype=torch.float64,
        )
    )
    arcs = []
    final = []
    state = 1
    for word_index, word in enumerate(WORDS, 1):
        labels = (None, *word) if leading_epsilon else (*word, None)
        for rank in range(width):
            start = 0
            for label, slot in zip(labels, (word_index, 3 + rank, 0), strict=True):
                arcs.append(Arc(start, state, (label,), slot))
                start = state
                state += 1
            final.append(Boundary(start, 0))
    topology = RelationTopology(
        (TAPE,),
        0,
        state,
        tuple(arcs),
        (Boundary(0, 0),),
        tuple(final),
        (
            "unit",
            "first_word",
            "second_word",
            *(f"alignment_{i}" for i in range(width)),
        ),
    )
    return RationalRelation(topology, weights, algebra=LOG_PROB), weights


def _graph(left: RationalRelation, right: RationalRelation) -> StringFactorGraph:
    return StringFactorGraph(
        (StringVariable("x", TAPE),),
        (StringFactor(("x",), left), StringFactor(("x",), right)),
    )


def fixture(mode: Mode, width: int) -> Fixture:
    """Build matched or competing local word preferences at a fixed width."""
    left, _ = _factor(width, (1.0, 0.0), leading_epsilon=True)
    right, weights = _factor(
        width, (0.5, 0.0) if mode == "agree" else (0.0, 0.5), leading_epsilon=False
    )
    return Fixture(left, right, _graph(left, right), weights)


def _weights(relation: RationalRelation) -> tuple[Fraction | None, ...]:
    return tuple(
        None if value == -torch.inf else Fraction(value)
        for value in relation.module()().detach().tolist()
    )


def _outgoing(topology: RelationTopology) -> dict[int, list[tuple[int, Arc]]]:
    outgoing: dict[int, list[tuple[int, Arc]]] = defaultdict(list)
    for index, arc in enumerate(topology.arcs):
        outgoing[arc.source].append((index, arc))
    return outgoing


def _paths(relation: RationalRelation) -> tuple[PathKey, ...]:
    topology = relation.topology
    values = _weights(relation)
    outgoing = _outgoing(topology)
    final = {mark.state: values[mark.weight] for mark in topology.final}
    pending: list[PathKey] = []
    for mark in topology.initial:
        if (weight := values[mark.weight]) is not None:
            pending.append(((), mark.state, (), mark.state, weight))
    accepted: list[PathKey] = []
    while pending:
        word, initial, ids, state, score = pending.pop()
        if (weight := final.get(state)) is not None:
            accepted.append((word, initial, ids, state, score + weight))
        for index, arc in outgoing.get(state, ()):
            if (weight := values[arc.weight]) is not None:
                label = arc.labels[0]
                emitted = (*word, label) if label is not None else word
                pending.append(
                    (emitted, initial, (*ids, index), arc.target, score + weight)
                )
    return tuple(accepted)


@dataclass(frozen=True)
class ExplicitDecode:
    optimum: Fraction | None
    reached_states: int
    peak_pending_states: int
    examined_arcs: int


def _explicit_decode(relation: RationalRelation) -> ExplicitDecode:
    topology = relation.topology
    values = _weights(relation)
    outgoing = _outgoing(topology)
    order = TopologicalSorter(
        {
            source: tuple(arc.target for _, arc in arcs)
            for source, arcs in outgoing.items()
        }
    )
    final = {mark.state: values[mark.weight] for mark in topology.final}
    pending: dict[int, Fraction] = {}
    for mark in topology.initial:
        if (weight := values[mark.weight]) is not None:
            pending[mark.state] = weight
        order.add(mark.state)
    peak = len(pending)
    result = None
    visited = examined = 0
    for state in reversed(tuple(order.static_order())):
        score = pending.pop(state, None)
        if score is None:
            continue
        visited += 1
        if (weight := final.get(state)) is not None:
            finish = score + weight
            result = finish if result is None else max(finish, result)
        for _, arc in outgoing.get(state, ()):
            examined += 1
            if (weight := values[arc.weight]) is not None:
                candidate = score + weight
                previous = pending.get(arc.target)
                if previous is None or previous < candidate:
                    pending[arc.target] = candidate
        peak = max(peak, len(pending))
    return ExplicitDecode(result, visited, peak, examined)


def _explicit(case: Fixture) -> RationalRelation:
    return intersect(case.left, case.right, max_states=MAX_STATES, max_work=MAX_WORK)


def _materialization_metrics(case: Fixture) -> tuple[RationalRelation, int, int]:
    counters = []
    original = calculus._CompositionWork

    def track(limit: int | None) -> calculus._CompositionWork:
        work = original(limit)
        counters.append(work)
        return work

    with patch.object(calculus, "_CompositionWork", track):
        relation = _explicit(case)
    (work,) = counters
    assert len(relation.topology.arcs) == work.arcs
    return relation, work.probes, work.arcs


def _adjusted(path: PathKey, shifts: Mapping[Word, Fraction]) -> Fraction:
    word, _, _, _, score = path
    padded = (-1, *word, -2)
    return score + sum(
        coefficient
        for feature, coefficient in shifts.items()
        for start in range(len(padded) - len(feature) + 1)
        if padded[start : start + len(feature)] == feature
    )


def _check_local(
    originals: tuple[tuple[PathKey, ...], ...],
    paths: tuple[StringAlignment, ...] | None,
    fixed: Word | None,
    shifts: Shifts,
) -> None:
    candidates = [
        tuple(path for path in accepted if fixed is None or path[0] == fixed)
        for accepted in originals
    ]
    if paths is None:
        assert any(not choices for choices in candidates)
        return
    for choices, path, (potential,) in zip(candidates, paths, shifts, strict=True):
        (word,) = path.words
        selected = word, path.start, path.arc_ids, path.stop, path.base_score
        assert selected in choices
        assert path.adjusted_score == _adjusted(selected, potential)
        assert path.adjusted_score == max(
            _adjusted(candidate, potential) for candidate in choices
        )


def _check_implicit(
    case: Fixture,
    originals: tuple[tuple[PathKey, ...], ...],
    feasible: Counter[tuple[Word, Fraction]],
    *,
    unigram_only: bool = False,
) -> tuple[StringAlignmentMapResult, int]:
    optimum = max((score for _, score in feasible), default=None)
    bounds = []
    primal_scores = []
    original_decode = alignment_dual._DualAlignmentRun.decode
    original_best = alignment_dual._LocalAlignments.best
    local_costs: list[tuple[int, int]] = []

    def count(
        decoder: alignment_dual._LocalAlignments,
        observed: tuple[Word | None, ...],
        adjustments: alignment_dual._Adjustments,
        max_states: int,
        max_work: int,
    ) -> tuple[StringAlignment | None, int, int]:
        result = original_best(decoder, observed, adjustments, max_states, max_work)
        local_costs.append(result[1:])
        return result

    def track(
        run: alignment_dual._DualAlignmentRun,
        observed: Mapping[str, Word | None],
        adjustments: alignment_dual._Multipliers,
    ) -> tuple[StringAlignment, ...] | None:
        paths = original_decode(run, observed, adjustments)
        fixed = observed["x"]
        _check_local(originals, paths, fixed, adjustments)
        if paths is None:
            return None
        if fixed is None:
            total: dict[Word, Fraction] = defaultdict(Fraction)
            for (shift,) in adjustments:
                for feature, value in shift.items():
                    total[feature] += value
            assert not any(total.values())
            bound = sum((path.adjusted_score for path in paths), Fraction(0))
            assert optimum is None or optimum <= bound
            bounds.append(bound)
        if all(path.words == paths[0].words for path in paths):
            primal_scores.append(sum((path.base_score for path in paths), Fraction(0)))
        return paths

    settings = SETTINGS | ({"max_ngram_order": 1} if unigram_only else {})
    with (
        patch.object(alignment_dual._DualAlignmentRun, "decode", track),
        patch.object(alignment_dual._LocalAlignments, "best", count),
    ):
        result = dual_alignment_map(case.graph, **settings)
    assert result.max_local_states == max(states for states, _ in local_costs)
    assert result.local_arc_examinations == sum(arcs for _, arcs in local_costs)
    assert result.upper_bound == min(bounds, default=None)
    assert result.score == max(primal_scores, default=None)
    if result.assignment is not None:
        ((name, word),) = result.assignment
        assert name == "x" and (word, result.score) in feasible
    if result.certificate is not None:
        assert result.status == "optimal"
        assert result.score == result.upper_bound == optimum
        certificate = result.certificate
        shifts = tuple(
            tuple(dict(rows) for rows in factor) for factor in certificate.multipliers
        )
        _check_local(originals, certificate.alignments, None, shifts)
        assert result.assignment is not None
        ((_, word),) = result.assignment
        assert all(path.words == (word,) for path in certificate.alignments)
        assert sum(path.adjusted_score for path in certificate.alignments) == optimum
    return result, len(local_costs)


def compare(case: Fixture, *, unigram_only: bool = False) -> dict[str, object]:
    """Check both decoders and collect exact structural and search counts."""
    originals = _paths(case.left), _paths(case.right)
    feasible = Counter(
        (left[0], left[-1] + right[-1])
        for left, right in product(*originals)
        if left[0] == right[0]
    )
    relation, probes, arcs = _materialization_metrics(case)
    assert Counter((path[0], path[-1]) for path in _paths(relation)) == feasible
    implicit, local_calls = _check_implicit(
        case, originals, feasible, unigram_only=unigram_only
    )
    explicit = _explicit_decode(relation)
    assert explicit.optimum == max((score for _, score in feasible), default=None)
    return {
        "input_states": sum(
            operand.topology.n_states for operand in (case.left, case.right)
        ),
        "explicit_peak_product_states": relation.topology.n_states,
        "explicit_product_arcs": arcs,
        "explicit_label_probes": probes,
        "explicit_decode": {
            **asdict(explicit),
            "optimum": str(explicit.optimum),
        },
        "enumerated_local_paths": [len(accepted) for accepted in originals],
        "enumerated_compatible_pairs": sum(feasible.values()),
        "implicit_largest_decode_configurations": implicit.max_local_states,
        "implicit_total_examined_arcs": implicit.local_arc_examinations,
        "implicit_iterations": implicit.iterations,
        "implicit_local_decode_calls": local_calls,
        "implicit_peak_active_features": sum(
            len(grams) for _, grams in implicit.active_features
        ),
        "assignment": implicit.assignment,
        "primal": str(implicit.score),
        "upper_bound": str(implicit.upper_bound),
        "status": implicit.status,
        "certificate": implicit.certificate is not None,
    }


def _measure(case: Fixture, repeats: int) -> dict[str, float]:
    dual_times = []
    explicit_times = []
    build_times = []

    def dual() -> None:
        start = perf_counter()
        dual_alignment_map(case.graph, **SETTINGS)
        dual_times.append(perf_counter() - start)

    def explicit() -> None:
        start = perf_counter()
        relation = _explicit(case)
        before_decode = perf_counter()
        _explicit_decode(relation)
        explicit_times.append(perf_counter() - start)
        build_times.append(before_decode - start)

    dual()
    explicit()
    dual_times.clear()
    explicit_times.clear()
    build_times.clear()
    for i in range(repeats):
        methods = (dual, explicit) if i % 2 == 0 else (explicit, dual)
        for method in methods:
            method()
    return {
        "implicit_seconds_median": median(dual_times),
        "explicit_build_and_decode_seconds_median": median(explicit_times),
        "explicit_build_seconds_median": median(build_times),
    }


def _expect(error: type[Exception], wording: str, invoke: Callable[[], object]) -> None:
    try:
        invoke()
    except error as found:
        assert wording in str(found), str(found)
    else:
        raise AssertionError(f"expected {error.__name__}: {wording}")


def controls() -> dict[str, object]:
    """Check unchanged settings across live updates and explicit refusal cases."""
    case = fixture("adaptive", 8)
    retained_product = _explicit(case)
    before = compare(case)
    before_product_score = _explicit_decode(retained_product).optimum
    assert before["assignment"] == (("x", WORDS[0]),) and before_product_score == 1
    limited = compare(case, unigram_only=True)
    assert (limited["status"], limited["primal"], limited["upper_bound"]) == (
        "limit",
        "1",
        "3/2",
    )
    assert limited["certificate"] is False
    with torch.no_grad():
        case.right_scores[2] = 2.0
    after = compare(case)
    assert after["assignment"] == (("x", WORDS[1]),)
    assert _explicit_decode(retained_product).optimum == 2
    with torch.no_grad():
        case.right_scores[1:3] = -torch.inf
    empty = compare(case)
    assert empty["status"] == "infeasible" and empty["certificate"] is False
    assert _explicit_decode(retained_product).optimum is None

    _expect(
        RelationLimitError,
        "max_states",
        lambda: dual_alignment_map(case.graph, **(SETTINGS | {"max_states": 1})),
    )
    _expect(
        RelationLimitError,
        "max_work",
        lambda: dual_alignment_map(case.graph, **(SETTINGS | {"max_work": 0})),
    )
    _expect(
        RelationLimitError,
        "max_work",
        lambda: intersect(case.left, case.right, max_states=MAX_STATES, max_work=0),
    )
    cyclic = RationalRelation(
        replace(
            case.left.topology, arcs=(*case.left.topology.arcs, Arc(0, 0, (None,), 0))
        ),
        case.left.module(),
        algebra=LOG_PROB,
    )
    _expect(
        ValueError,
        "acyclic positive",
        lambda: dual_alignment_map(_graph(cyclic, case.right), **SETTINGS),
    )
    wrong = RationalRelation(
        case.left.topology,
        torch.ones(len(case.left.topology.weight_names)),
        algebra=MARKOV,
    )
    _expect(
        TypeError,
        "requires LogProb",
        lambda: dual_alignment_map(_graph(wrong, wrong), **SETTINGS),
    )
    return {
        "before_update": before,
        "after_update": after,
        "zero_weight_right_support": empty,
        "unigram_only": limited,
        "checked_refusals": [
            "local states",
            "local work",
            "explicit work",
            "positive cycle",
            "algebra",
        ],
    }


def main() -> None:
    """Run the fixed scaling grid and print the complete JSON evidence."""
    parser = argparse.ArgumentParser(description="Measure exact string alignment MAP.")
    parser.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or any(not 2 <= width <= 48 for width in args.widths):
        parser.error("require positive repeats and widths between 2 and 48")
    torch.set_num_threads(1)
    rows = []
    for mode in ("agree", "adaptive"):
        for width in args.widths:
            case = fixture(mode, width)
            row = compare(case)
            assert row["status"] == "optimal" and row["certificate"] is True
            rows.append(
                {"family": mode, "width": width, **row, **_measure(case, args.repeats)}
            )
    print(
        json.dumps(
            {
                "device": "cpu",
                "dtype": "float64",
                "machine": platform.machine(),
                "platform": platform.platform(),
                "torch_version": str(torch.__version__),
                "torch_threads": torch.get_num_threads(),
                "timed_repeats": args.repeats,
                "warmup_calls_per_method": 1,
                "settings": SETTINGS | {"step_size": str(SETTINGS["step_size"])},
                "rows": rows,
                "untimed_controls": controls(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
