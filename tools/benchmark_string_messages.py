"""Compare projected, exact, and beam messages on a finite string-factor tree."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from statistics import median
from time import perf_counter
import tracemalloc

import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.objects import FinSet, StringSet
from quivers.core.relations import (
    Arc,
    Boundary,
    NgramMessagePlan,
    RationalRelation,
    RelationTopology,
    StringFactor,
    StringFactorGraph,
    StringVariable,
    regular_string_ep,
)


_WORDS = tuple(product(range(2), repeat=2))
_TAPE = StringSet(alphabet=FinSet(name="Letter", cardinality=2))
_NAMES = "x", "y"
_LIMITS = dict(max_states=2048, max_work=32_768)


@dataclass(frozen=True)
class _Messages:
    probabilities: torch.Tensor
    marginal_states: int
    site_states: int | None
    features: int | None
    status: str


def _table(scores: torch.Tensor) -> RationalRelation:
    arity = scores.ndim
    entries = tuple(product(_WORDS, repeat=arity))
    arcs = []
    for slot, words in enumerate(entries, 1):
        middle = slot + 1
        arcs.extend(
            (
                Arc(0, middle, tuple(word[0] for word in words), 0),
                Arc(middle, 1, tuple(word[1] for word in words), slot),
            )
        )
    topology = RelationTopology(
        (_TAPE,) * arity,
        0,
        len(entries) + 2,
        tuple(arcs),
        (Boundary(0, 0),),
        (Boundary(1, 0),),
        ("unit", *(f"entry-{i}" for i in range(len(entries)))),
    )
    return RationalRelation(
        topology, torch.cat((scores.new_zeros(1), scores.flatten())), algebra=LOG_PROB
    )


def _base() -> RationalRelation:
    arcs = tuple(
        Arc(state, state + 1, (label,), 0) for state in range(2) for label in range(2)
    )
    topology = RelationTopology(
        (_TAPE,), 0, 3, arcs, (Boundary(0, 0),), (Boundary(2, 0),), ("unit",)
    )
    return RationalRelation(
        topology, torch.zeros(1, dtype=torch.float64), algebra=LOG_PROB
    )


def _exact(graph: StringFactorGraph) -> _Messages:
    queries = tuple(graph.tree_marginal(name, **_LIMITS) for name in _NAMES)
    values = torch.stack(
        tuple(torch.stack(tuple(q.log_prob(w).exp() for w in _WORDS)) for q in queries)
    )
    return _Messages(
        values, sum(q.relation.topology.n_states for q in queries), None, None, "exact"
    )


def _projected(
    graph: StringFactorGraph, bases: dict[str, RationalRelation], penalty: float
) -> _Messages:
    features = {name: ((0,), (1,), *_WORDS) for name in _NAMES}
    result = regular_string_ep(
        graph,
        bases,
        features,
        penalty=penalty,
        max_sweeps=12,
        projection_tolerance=1e-8,
        **_LIMITS,
    )
    values = torch.stack(
        tuple(
            torch.stack(tuple(result.marginals[name].log_prob(w).exp() for w in _WORDS))
            for name in _NAMES
        )
    )
    site_states = sum(
        NgramMessagePlan(bases[s.variable], s.features, **_LIMITS).n_states
        for s in result.sites
    )
    return _Messages(
        values,
        sum(belief.relation.topology.n_states for belief in result.marginals.values()),
        site_states,
        sum(len(s.features) for s in result.sites),
        result.status,
    )


def _beam(
    scores: tuple[torch.Tensor, torch.Tensor, torch.Tensor], width: int
) -> _Messages:
    def prune(row: torch.Tensor) -> torch.Tensor:
        selected = torch.argsort(row, descending=True, stable=True)[:width]
        return row.new_full(row.shape, -torch.inf).scatter(0, selected, row[selected])

    def prefix_states(row: torch.Tensor) -> int:
        support = (_WORDS[i] for i in torch.where(row.isfinite())[0].tolist())
        return len(
            {word[:length] for word in support for length in range(len(word) + 1)}
        )

    pair, left, right = scores
    incoming_left = prune(left)
    incoming_right = prune(right)
    outgoing_left = prune(torch.logsumexp(pair + incoming_right[None, :], dim=1))
    outgoing_right = prune(torch.logsumexp(pair + incoming_left[:, None], dim=0))
    beliefs = torch.stack(
        (incoming_left + outgoing_left, incoming_right + outgoing_right)
    )
    return _Messages(
        beliefs.softmax(1),
        sum(map(prefix_states, beliefs)),
        sum(
            map(
                prefix_states,
                (incoming_left, incoming_right, outgoing_left, outgoing_right),
            )
        ),
        None,
        "truncated",
    )


def _measure(
    run: Callable[[], _Messages], repeats: int
) -> tuple[_Messages, float, float]:
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = run()
        samples.append(perf_counter() - start)
    tracemalloc.start()
    try:
        run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, 1000 * median(samples), peak / 1024


def main() -> None:
    """Report finite-domain accuracy, representation size, and measured costs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(1)
    pair = torch.tensor(
        [
            [1.6, 0.0, 0.4, 0.2],
            [0.2, 1.1, 0.0, 0.4],
            [0.4, 0.3, 0.9, 0.1],
            [0.2, 0.5, 0.1, 1.3],
        ],
        dtype=torch.float64,
    )
    left = pair.new_tensor([0.8, -0.2, 0.4, -0.5])
    right = pair.new_tensor([-0.3, 0.7, -0.4, 0.5])
    graph = StringFactorGraph(
        tuple(StringVariable(name, _TAPE) for name in _NAMES),
        (
            StringFactor(_NAMES, _table(pair)),
            StringFactor(("x",), _table(left)),
            StringFactor(("y",), _table(right)),
        ),
    )
    base = _base()
    bases = dict.fromkeys(_NAMES, base)
    joint = (
        (pair + left[:, None] + right[None, :])
        .reshape(-1)
        .softmax(0)
        .reshape(pair.shape)
    )
    oracle = torch.stack((joint.sum(1), joint.sum(0)))
    torch.testing.assert_close(_exact(graph).probabilities, oracle)
    print(
        "Method | Status | Mean TV error | Site features | Site states | Marginal states | Median ms | Peak Python KiB"
    )
    print("--- | --- | --- | --- | --- | --- | --- | ---")

    def report(name: str, run: Callable[[], _Messages]) -> None:
        result, milliseconds, peak = _measure(run, args.repeats)
        error = (result.probabilities - oracle).abs().sum(dim=1).mean().item() / 2
        features = result.features if result.features is not None else "—"
        states = result.site_states if result.site_states is not None else "—"
        print(
            f"{name} | {result.status} | {error:.6f} | {features} | {states} | {result.marginal_states} | {milliseconds:.1f} | {peak:.1f}"
        )

    report("Exact sparse tree", lambda: _exact(graph))
    report("Beam 3", lambda: _beam((pair, left, right), 3))
    for penalty in (0.0, 0.025, 0.1, 0.3, 1.0):
        report(f"Regular EP ({penalty:g})", lambda: _projected(graph, bases, penalty))


if __name__ == "__main__":
    main()
