"""Measure repeated parameter updates on the acyclic a b a fixture.

Run `python -m tests.benchmarks.deduction_partition` from the repository
root. Each executor runs the same CPU float64 forward, reverse, and
gradient-ascent steps from the same initial point. The timed runs exclude
executor construction and warmup. A separate run reads peak live CPU tensor bytes
above the profiling scope's initial balance. Torch allocator events
include kernel temporaries and backward buffers. Python memory and
memory already live before profiling are outside that measure.

The report counts the maximum retained provenance returned by one
update, in the executor's own units. Search runs with trace recording
disabled. The agenda keeps ground applications with their premises and
lexical occurrences. Wiring retains no proof records. Timings and
allocator samples are descriptive.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from statistics import median
from time import perf_counter

import torch

from tests.evidence.checks.partition import (
    NAMES,
    POINTS,
    SENTENCE,
    Reader,
    agenda,
    contraction,
    machine,
)
from tests.evidence.counters import memory
from tests.evidence.problems import logarithm

STEP_SIZE = 0.01


def updates(
    evaluate: Reader, initial: Mapping[str, float], steps: int
) -> tuple[float, int]:
    """Run gradient ascent on log partition and return the last score and peak records."""
    weights = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in initial.items()
    }
    retained = 0
    for _ in range(steps):
        score, leaves, records = evaluate(weights)
        gradients = torch.autograd.grad(score, tuple(leaves.values()))
        with torch.no_grad():
            for name, gradient in zip(leaves, gradients, strict=True):
                weights[name].add_(gradient, alpha=STEP_SIZE)
        retained = max(retained, records)
    return float(score.detach()), retained


def measure(
    evaluate: Reader,
    initial: Mapping[str, float],
    *,
    steps: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    """Run independent timed and profiled repetitions from the initial point."""
    if warmup:
        updates(evaluate, initial, warmup)
    durations: list[float] = []
    peak_records = 0
    for _ in range(repeats):
        start = perf_counter()
        last, records = updates(evaluate, initial, steps)
        durations.append(perf_counter() - start)
        peak_records = max(peak_records, records)

    def profiled() -> None:
        updates(evaluate, initial, steps)

    allocations = memory(profiled)
    return {
        "elapsed_seconds": durations,
        "median_seconds": median(durations),
        "peak_live_cpu_tensor_bytes_above_baseline": allocations.peak_bytes,
        "total_cpu_tensor_allocated_bytes": allocations.allocated_bytes,
        "peak_retained_provenance_per_update": peak_records,
        "last_pre_update_log_partition": last,
    }


def main() -> None:
    """Print one JSON benchmark report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args()
    if options.steps < 1 or options.repeats < 1 or options.warmup < 0:
        parser.error("steps and repeats must be positive, and warmup nonnegative")
    torch.set_num_threads(1)
    initial = dict(zip(NAMES, map(logarithm, dict(POINTS)["positive"]), strict=True))
    executors: dict[str, dict[str, object]] = {}
    report = {
        "sentence": SENTENCE,
        "initial_log_weights": initial,
        "step_size": STEP_SIZE,
        "steps": options.steps,
        "warmup_steps": options.warmup,
        "timed_repeats": options.repeats,
        "dtype": "float64",
        "device": "cpu",
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "executors": executors,
    }
    for name, make, provenance in (
        ("machine-search", machine, "trace events with recording disabled"),
        (
            "indexed-agenda",
            agenda,
            "axiom, application, and premise-occurrence records",
        ),
        ("einsum-wiring", contraction, "proof records"),
    ):
        executors[name] = {
            **measure(
                make(),
                initial,
                steps=options.steps,
                warmup=options.warmup,
                repeats=options.repeats,
            ),
            "provenance_unit": provenance,
        }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
