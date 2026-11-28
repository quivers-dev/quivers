"""Compare proof search and exact weight plans over repeated parameter updates."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from itertools import product
from statistics import median
from time import perf_counter
import tracemalloc
from typing import Literal

import torch

from quivers.core.algebras import REAL
from quivers.qiec.contracts import (
    Acyclic,
    JudgmentPattern,
    NoRetention,
    OccurrenceId,
    RequestedEvidence,
    Retention,
)
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.rules import AxiomOccurrence, AxiomSlot
from quivers.stochastic import Plan, plan
from tests.evidence import bodies, graphs


type Engine = Callable[[torch.Tensor], tuple[torch.Tensor, int]]


def _engines(size: int) -> dict[str, Engine]:
    slots = tuple(
        AxiomSlot(
            id=ConstructorId.derive("deduction-plan-benchmark", name),
            name=name,
            telescope=tuple(IndexBinder(name=axis, sort=NAT) for axis in ("x", "y")),
            judgment=bodies.judgment(name, "x", "y"),
        )
        for name in ("r1", "r2", "r3")
    )
    presentation = bodies.PRESENTATION.with_(rules=(bodies.STAR,), axioms=slots)
    queries = graphs.query().with_(presentation=presentation.id)
    occurrences = tuple(
        AxiomOccurrence(
            slot=slot.id,
            occurrence=OccurrenceId.derive(
                "deduction-plan-benchmark", slot.name, str(i)
            ),
            binding=(("x", 0), ("y", i)),
        )
        for slot in slots
        for i in range(size)
    )
    (goal,) = bodies.items("hub", [(0,)])
    without = NoRetention()
    retained = RequestedEvidence(patterns=(JudgmentPattern(family=goal.family),))

    def selected(engine: Plan) -> Engine:
        def run(weights: torch.Tensor) -> tuple[torch.Tensor, int]:
            inputs = {
                axiom.occurrence: weight
                for axiom, weight in zip(
                    occurrences, weights.flatten().unbind(), strict=True
                )
            }
            result = engine.run(occurrences, inputs, like=weights.new_zeros(()))
            return result.value(goal), result.report().provenance_records

        return run

    def search(weights: torch.Tensor) -> tuple[torch.Tensor, int]:
        proofs = tuple(product(range(size), repeat=3))
        indices = torch.tensor(proofs, device=weights.device)
        rows = torch.arange(3, device=weights.device)
        terms = weights[rows, indices].prod(dim=1)
        return terms.sum(), len(proofs)

    def contract(weights: torch.Tensor) -> tuple[torch.Tensor, int]:
        return torch.einsum("i,j,k->", *weights.unbind()), 0

    def checked(
        retention: Retention, *, executor: Literal["auto", "eliminate"] = "auto"
    ) -> Engine:
        return selected(
            plan(
                queries.with_(retention=retention),
                presentation,
                REAL,
                Acyclic(),
                graphs.contract(),
                retention,
                executor=executor,
            )
        )

    return {
        "Explicit proofs": search,
        "Indexed": checked(without),
        "Indexed with goal evidence": checked(retained),
        "Dense contraction": contract,
        "Rule variable elimination": checked(without, executor="eliminate"),
    }


def _fit(
    engine: Engine,
    initial: torch.Tensor,
    steps: int,
    *,
    record: bool = False,
    verify: bool = False,
) -> tuple[int, float]:
    logits = initial.detach().clone().requires_grad_()
    provenance = 0
    largest_tape = 0
    saved: dict[int, int] = {}

    def pack(value: torch.Tensor) -> torch.Tensor:
        storage = value.untyped_storage()
        saved[storage.data_ptr()] = storage.nbytes()
        return value.detach()

    for _ in range(steps):
        saved.clear()
        if record:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                partition, kept = engine(logits.exp())
        else:
            partition, kept = engine(logits.exp())
        if record:
            largest_tape = max(largest_tape, sum(saved.values()))
        (gradient,) = torch.autograd.grad(partition, logits)
        provenance = max(provenance, kept)
        with torch.no_grad():
            if verify:
                weights = logits.exp()
                sums = weights.sum(dim=1)
                expected = sums.prod()
                # Each occurrence in group i participates exactly once with
                # every choice in the other two groups.
                oracle = weights * (expected / sums).unsqueeze(1)
                torch.testing.assert_close(partition, expected)
                torch.testing.assert_close(gradient, oracle)
            logits.sub_(0.2 * gradient / partition)
    return provenance, largest_tape / 1024


def main() -> None:
    """Print reproducible per-arm timing, provenance, and allocation measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=(4, 8, 12))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(*args.sizes, args.steps, args.repeats) < 1:
        parser.error("sizes, steps, and repeats must be positive")
    torch.set_num_threads(1)
    print(
        "Width | Engine | Refit ms | Provenance records | Peak Python KiB | Saved tape KiB"
    )
    print("--- | --- | --- | --- | --- | ---")
    for size in args.sizes:
        initial = torch.linspace(-0.8, 0.7, 3 * size, dtype=torch.float64).reshape(
            3, size
        )
        for name, engine in _engines(size).items():
            _fit(engine, initial, args.steps, verify=True)
            times = []
            for _ in range(args.repeats):
                start = perf_counter()
                records, _ = _fit(engine, initial, args.steps)
                times.append(1000 * (perf_counter() - start) / args.steps)
            tracemalloc.start()
            try:
                _fit(engine, initial, args.steps)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            _, saved = _fit(engine, initial, args.steps, record=True)
            print(
                f"{size} | {name} | {median(times):.2f} | {records} | {peak / 1024:.1f} | {saved:.1f}"
            )


if __name__ == "__main__":
    main()
