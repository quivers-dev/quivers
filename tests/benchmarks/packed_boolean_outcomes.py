"""Measure the checked packed Boolean fold over ambiguity and repeated fitting.

From the repository root, run `python -m tests.benchmarks.packed_boolean_outcomes`.
The workload uses the shipped parity grammar, CPU float64, one thread and the
ordinary checked program entry. The constructor records include every interval.
The logical fold visits only goal ancestors, where two lanes per unary edge and
four per binary edge suffice. There are 2**n proofs for a sentence of length n.

Arithmetic counts cover dispatched tensor operations across the program, with
the forward and reverse measured separately at the initial parameters. Saved
elements are logical tensor elements counted per autograd save, including
duplicates. Separate refitting trials measure wall time and CPU allocator
events. Peak live tensor bytes are above the profiler's opening balance.
A separate run measures peak traced Python bytes. Both exclude the
precompiled program. Timing and allocation measurements vary across
Torch builds and machines.

Let L be the number of recorded meaning lanes. The fold does one
addition per premise in each rule lane and a linear reduction over all L
contributions. Its reverse also takes O(L) scalar operations when each
scalar kernel's partials cost constant work. The saved-tensor bound
models the CPU autograd rules for the actual masked logsumexp reductions.
For a nonempty group of m entries, logsumexp saves m inputs and one
output. The two where operations save one scalar condition each. The
fold has six nonempty groups per word, so it saves at most L + 18n
elements. The three final reductions save at most 13 elements. The
score's exp and log save one each, and the abs operations in the range
and finiteness checks save up to seven fixed-size elements altogether.
Stacking, selecting or splitting a tensor, and adding log weights need
metadata but no saved tensor values here. This bounds logical saves.
Full forest construction retains quadratic provenance on this fixture.
The benchmark keeps that cost separate from the goal-restricted fold.
"""

from __future__ import annotations

import argparse
import json
from math import log
from pathlib import Path
from statistics import median
from time import perf_counter
import tracemalloc

import torch

from quivers.dsl import loads
from quivers.qiec import ProgramRun, QiecModule, run_program
from tests.evidence.counters import measure, memory

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "docs/examples/qiec/packed-boolean-proof-outcomes.qvr"
)
INITIAL = (-2.5, -0.15)
STEP_SIZE = 0.02


def _evaluate(
    module: QiecModule, sentence: tuple[str, ...], logits: torch.Tensor
) -> ProgramRun:
    return run_program(
        module,
        "fit",
        data={"sentence": sentence},
        sites={},
        parameters=dict(
            zip(("Parity.lex.0", "Parity.lex.1"), logits.unbind(), strict=True)
        ),
    )


def _oracle(logits: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Each word independently contributes either exp(u) or exp(v) to the
    # product. The odd total is ((exp(u)+exp(v))**n-(exp(u)-exp(v))**n)/2.
    # Divide by the full total and differentiate this expression directly.
    u, v = logits.detach().unbind()
    ratio = ((u - v) / 2).tanh()
    complement = 1 - ratio.pow(length)
    derivative = (
        -length * ratio.pow(length - 1) * (1 - ratio.square()) / (2 * complement)
    )
    return (-ratio.pow(length)).log1p() - log(2), torch.stack((derivative, -derivative))


def _fit(
    module: QiecModule, sentence: tuple[str, ...], steps: int, *, verify: bool = False
) -> float:
    logits = torch.tensor(INITIAL, dtype=torch.float64, requires_grad=True)
    for _ in range(steps):
        score = _evaluate(module, sentence, logits).log_joint
        (gradient,) = torch.autograd.grad(score, logits)
        if verify:
            expected, expected_gradient = _oracle(logits, len(sentence))
            torch.testing.assert_close(score, expected, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(
                gradient, expected_gradient, atol=1e-12, rtol=1e-12
            )
        with torch.no_grad():
            logits.add_(gradient, alpha=STEP_SIZE)
    return float(score.detach())


def _work(module: QiecModule, sentence: tuple[str, ...]) -> dict[str, object]:
    logits = torch.tensor(INITIAL, dtype=torch.float64, requires_grad=True)
    results: list[ProgramRun] = []

    def forward() -> None:
        results.append(_evaluate(module, sentence, logits))

    forward_work = measure(forward)
    (result,) = results
    backward_work = measure(lambda: torch.autograd.grad(result.log_joint, logits)[0])
    (trace,) = (
        event.detail
        for event in result.result.trace
        if event.event == "deduction.boolean_outcomes"
    )
    assert trace["algorithm"] == "indexed-packed-forest-boolean-fold"
    construction = dict(trace["construction_work"])
    actual = {
        "evaluated_items": trace["evaluated_items"],
        "edge_meaning_lanes": trace["edge_meaning_lanes"],
        "weight_products": trace["weight_products"],
        "reduced_entries": trace["reduced_entries"],
        "construction_provenance_records": construction["provenance_records"],
    }
    n = len(sentence)
    # The goal ancestors are n each of A, B, T, and S. There are 2n
    # axioms, 2n+1 unary edges and n-1 binary prefix edges. Construction
    # makes the S edges on all intervals before reachability restriction.
    planned = {
        "evaluated_items": 4 * n,
        "edge_meaning_lanes": 10 * n - 2,
        "weight_products": 12 * n - 6,
        "reduced_entries": 10 * n - 2,
        "construction_provenance_records": 8 * n + 3 * n * (n - 1) // 2,
    }
    assert actual == planned, (planned, actual)
    saved_bound = planned["edge_meaning_lanes"] + 18 * n + 22
    assert forward_work.saved <= saved_bound, (forward_work.saved, saved_bound)
    return {
        "algorithm": trace["algorithm"],
        "observer": trace["observer"],
        "observed_schedule": actual,
        "derived_schedule": planned,
        "forward_tensor_scalar_ops_counted": forward_work.scalar,
        "backward_tensor_scalar_ops_counted": backward_work.scalar,
        "forward_dispatched_tensor_ops": forward_work.kernels,
        "backward_dispatched_tensor_ops": backward_work.kernels,
        "logical_elements_saved_for_backward": forward_work.saved,
        "derived_logical_saved_element_bound": saved_bound,
    }


def _measure(
    module: QiecModule, length: int, *, steps: int, warmup: int, repeats: int
) -> dict[str, object]:
    sentence = ("x",) * length
    last = _fit(module, sentence, steps, verify=True)
    work = _work(module, sentence)
    if warmup:
        _fit(module, sentence, warmup)
    durations: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        _fit(module, sentence, steps)
        durations.append(perf_counter() - start)

    def refit() -> None:
        _fit(module, sentence, steps)

    allocations = memory(refit)
    tracemalloc.start()
    try:
        refit()
        _, python_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "sentence_length": length,
        "derivations": 2**length,
        **work,
        "elapsed_seconds": durations,
        "median_seconds": median(durations),
        "peak_live_cpu_tensor_bytes_above_baseline": allocations.peak_bytes,
        "total_cpu_tensor_allocated_bytes": allocations.allocated_bytes,
        "peak_traced_python_bytes": python_peak,
        "last_pre_update_log_likelihood": last,
    }


def main() -> None:
    """Fit the same checked program at each length and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--lengths", type=int, nargs="+", default=(1, 4, 8, 16, 20))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args()
    if options.steps < 1 or options.repeats < 1 or options.warmup < 0:
        parser.error("steps and repeats must be positive, and warmup nonnegative")
    if min(options.lengths) < 1:
        parser.error("sentence lengths must be positive")
    torch.set_num_threads(1)
    module = loads(SOURCE.read_text()).qiec
    assert isinstance(module, QiecModule)
    report = {
        "initial_log_weights": INITIAL,
        "step_size": STEP_SIZE,
        "steps": options.steps,
        "warmup_steps": options.warmup,
        "timed_repeats": options.repeats,
        "dtype": "float64",
        "device": "cpu",
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "results": [
            _measure(
                module,
                length,
                steps=options.steps,
                warmup=options.warmup,
                repeats=options.repeats,
            )
            for length in options.lengths
        ],
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
