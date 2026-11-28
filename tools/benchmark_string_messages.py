"""Compare projected, exact, and beam messages on a finite string-factor tree."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from math import prod
from statistics import median
from time import perf_counter
import tracemalloc

import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.objects import FinSet, StringSet
from quivers.core.relations import (
    Arc,
    Boundary,
    FiniteStringBeliefPlan,
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
_NAMES: tuple[str, ...] = ("x", "y")
_MAX_STATES = 2048
_MAX_WORK = 32_768


@dataclass(frozen=True)
class _Messages:
    probabilities: torch.Tensor
    status: str
    marginal_size: tuple[int, int]
    site_size: tuple[int, int] | None = None
    features: int | None = None
    sweeps: int | None = None
    fit_attempts: int | None = None
    moment_evaluations: int | None = None
    beam_reduction_cells: int | None = None


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


def _size(relations: tuple[RationalRelation, ...]) -> tuple[int, int]:
    return (
        sum(relation.topology.n_states for relation in relations),
        sum(len(relation.topology.arcs) for relation in relations),
    )


def _exact(graph: StringFactorGraph) -> _Messages:
    queries = tuple(
        graph.tree_marginal(name, max_states=_MAX_STATES, max_work=_MAX_WORK)
        for name in _NAMES
    )
    values = torch.stack(
        tuple(torch.stack(tuple(q.log_prob(w).exp() for w in _WORDS)) for q in queries)
    )
    return _Messages(values, "exact", _size(tuple(q.relation for q in queries)))


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
        max_states=_MAX_STATES,
        max_work=_MAX_WORK,
    )
    values = torch.stack(
        tuple(
            torch.stack(tuple(result.marginals[name].log_prob(w).exp() for w in _WORDS))
            for name in _NAMES
        )
    )
    sites = tuple(
        NgramMessagePlan(
            bases[site.variable],
            site.features,
            max_states=_MAX_STATES,
            max_work=_MAX_WORK,
        ).bind(site.coefficients)
        for site in result.sites
    )
    # Every accepted sweep visited each factor occurrence by tape.
    # The final pass can stop early on a failed fit.
    attempts_per_sweep = sum(len(factor.variables) for factor in graph._factors)
    return _Messages(
        values,
        result.status,
        _size(tuple(belief.relation for belief in result.marginals.values())),
        site_size=_size(sites),
        features=sum(len(site.features) for site in result.sites),
        sweeps=result.sweeps,
        fit_attempts=attempts_per_sweep * result.sweeps + len(result.updates),
        moment_evaluations=result.moment_evaluations,
    )


def _beam(graph: StringFactorGraph, width: int) -> _Messages:
    supports = dict.fromkeys(_NAMES, _WORDS)
    entry_counts = tuple(
        prod(len(supports[name]) for name in factor.variables)
        for factor in graph._factors
    )
    plan = FiniteStringBeliefPlan(
        graph,
        supports,
        max_entries=sum(entry_counts),
        max_states=_MAX_STATES,
        max_work=_MAX_WORK,
    )
    result = plan.run_pruned(beam=width, max_sweeps=12)
    if result.log_marginals is None:
        raise ValueError("benchmark beam has no normalized beliefs")

    def support_size(rows: tuple[torch.Tensor, ...]) -> tuple[int, int]:
        roots = tuple(
            len(
                {
                    _WORDS[index][:length]
                    for index in torch.where(row.isfinite())[0].tolist()
                    for length in range(len(_WORDS[index]) + 1)
                }
            )
            for row in rows
        )
        return sum(roots), sum(max(0, states - 1) for states in roots)

    beliefs = tuple(result.log_marginals[name] for name in _NAMES)
    messages = tuple(row for group in result.log_messages for row in group)
    # A successful proposal pass visits each outgoing axis. Charge the
    # input table extent to every such computation, including unary axes.
    cells_per_pass = sum(
        len(factor.variables) * entries
        for factor, entries in zip(graph._factors, entry_counts, strict=True)
    )
    status = f"{result.status} (truncated)" if result.truncated else result.status
    return _Messages(
        torch.stack(tuple(row.exp() for row in beliefs)),
        status,
        support_size(beliefs),
        site_size=support_size(messages),
        sweeps=result.sweeps,
        beam_reduction_cells=(result.sweeps + 1) * cells_per_pass,
    )


def _measure(
    run: Callable[[], _Messages], repeats: int
) -> tuple[_Messages, float, float]:
    result = run()
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


def _fixture() -> tuple[StringFactorGraph, dict[str, RationalRelation], torch.Tensor]:
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
    return graph, bases, torch.stack((joint.sum(1), joint.sum(0)))


def main() -> None:
    """Report finite-domain accuracy, representation size, and measured costs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(1)
    graph, bases, oracle = _fixture()
    torch.testing.assert_close(_exact(graph).probabilities, oracle, rtol=0, atol=1e-14)
    torch.testing.assert_close(
        _beam(graph, len(_WORDS)).probabilities, oracle, rtol=0, atol=1e-14
    )
    methods = [
        ("Exact sparse tree", lambda: _exact(graph)),
        ("Production beam 3", lambda: _beam(graph, 3)),
    ]
    methods.extend(
        (
            f"Regular EP ({penalty:g})",
            lambda penalty=penalty: _projected(graph, bases, penalty),
        )
        for penalty in (0.0, 0.025, 0.1, 0.3, 1.0)
    )
    results = [(name, *_measure(run, args.repeats)) for name, run in methods]

    def cell(value: object) -> str:
        return "—" if value is None else str(value)

    def size_cell(value: tuple[int, int] | None) -> str:
        return "—" if value is None else f"{value[0]}/{value[1]}"

    print(
        "Method | Status | Mean TV error | Site features | Site states/arcs | Marginal states/arcs"
    )
    print("--- | --- | --- | --- | --- | ---")
    for name, result, _, _ in results:
        error = (result.probabilities - oracle).abs().sum(dim=1).mean().item() / 2
        print(
            f"{name} | {result.status} | {error:.6f} | {cell(result.features)} | "
            f"{size_cell(result.site_size)} | {size_cell(result.marginal_size)}"
        )
    print()
    print(
        "Method | Sweeps | Site fits | Moment evaluations | Beam input cells | Median ms | Peak Python KiB"
    )
    print("--- | --- | --- | --- | --- | --- | ---")
    for name, result, milliseconds, peak in results:
        print(
            f"{name} | {cell(result.sweeps)} | {cell(result.fit_attempts)} | "
            f"{cell(result.moment_evaluations)} | {cell(result.beam_reduction_cells)} | "
            f"{milliseconds:.1f} | {peak:.1f}"
        )


if __name__ == "__main__":
    main()
