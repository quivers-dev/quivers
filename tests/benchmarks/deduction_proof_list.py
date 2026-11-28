"""Measure checked proof enumeration, typed reification, and the proof cap.

Run `python -m tests.benchmarks.deduction_proof_list` from the repository
root. In a case N:D, each of N lexical entries has a distinct category
and one rule into the common category Step0. Each of D further unary
rules advances that category one step. Exactly one choice is made, at
the smallest premise. Thus the goal has N proofs with D+2 nodes each.
The packed forest keeps 3N+2D edge and premise-reference records. A
successful positive-weight query returns N(D+2) proof nodes and N rows.

The cap uses structural proof counts, which include zero-weight proofs.
The counting pass follows the packed forest, using saturated integers
with bit width bounded by the cap, and uses O(N+D) work and state on
this acyclic unary fixture. A failed cap still builds the forest and
counts. It reaches neither proof-node construction nor reification.
The paired probe observes these constructors directly.

Each unary proof node adds one LogProb product, hence N(D+1) scalar
log additions after the cap succeeds. The output-size calculation
bounds the number of reification nodes. The measured checked query also
performs index/type checks and retains its execution trace. Each lookup
of a constructor's arity can scan an index sort's S declared symbols, so
node counts alone give no linear end-to-end time bound as S grows. For
positive weights, the finite log range check saves N tensor elements,
the singleton group reductions save at most 5N, and the normalizer
and remaining finite checks save at most 2N+4. The resulting forward
tape bound is 8N+4 logical elements, independent of proof size. This
counts logical saves, including repeats. It excludes proof records,
Python allocations, and tensor allocator overhead.

Forward counts end at the typed proof rows. Reverse counts and refit
timings include a scalar host likelihood over those rows. Compilation
runs before profiling. Peak live CPU tensor bytes use the profiler's
initial balance as baseline. Reported timings and allocator samples
are descriptive.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from statistics import median
import sys
from time import perf_counter
import tracemalloc
from types import CodeType

import torch

from quivers.core.algebras import LOG_PROB
from quivers.dsl import loads as load_source
from quivers.qiec import ProgramRun, QiecModule, run_program
from quivers.qiec import deduction_outcomes
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.execution import ExecutionFailure, TraceObserver
from quivers.qiec.foreign import DeductionProofs
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.kinds import UserIndexSort
from quivers.qiec.rules import Derivation
from quivers.qiec.terms import ConstructorValue
from tests.evidence.counters import CallCounter, measure, memory

STEP_SIZE = 0.03
type Case = tuple[int, int]


def _source(choices: int, depth: int) -> str:
    categories = [f"Cat{i}" for i in range(choices)]
    steps = [f"Step{j}" for j in range(depth + 1)]
    atoms = ", ".join(("span", "lex", *categories, *steps))
    rules = [
        f"    rule choose{i} : span(I, J, {category}) |- span(I, J, Step0)"
        for i, category in enumerate(categories)
    ] + [
        f"    rule wrap{j} : span(I, J, {premise}) |- span(I, J, {result})"
        for j, (premise, result) in enumerate(zip(steps, steps[1:]))
    ]
    lexicon = "\n".join(
        f'        "x" : {category} = lex #[learnable]' for category in categories
    )
    rule_lines = "\n".join(rules)
    return f"""\
object Token : FinSet 1
object Result : Real 1

deduction Paths : Token -> Token [semiring=LogProb, start={steps[-1]}]
    atoms {atoms}
{rule_lines}
    lexicon
{lexicon}

program listed(limit : Int) : Token -> Result
    let proofs = parse_normalized_proofs(Paths, sentence, limit)
    return proofs

export listed
"""


@dataclass(frozen=True)
class _Fixture:
    module: QiecModule
    descriptor: DeductionProofs
    proof_ids: dict[str, str]
    initial: torch.Tensor
    rates: torch.Tensor
    depth: int

    @property
    def choices(self) -> int:
        return self.initial.numel()

    def run(
        self,
        weights: torch.Tensor,
        limit: int,
        *,
        observer: TraceObserver | None = None,
    ) -> ProgramRun:
        return run_program(
            self.module,
            "listed",
            data={"sentence": ("x",), "limit": limit},
            sites={},
            parameters={
                f"Paths.lex.{i}": weight for i, weight in enumerate(weights.unbind())
            },
            observer=observer,
        )

    def read(self, value: object) -> tuple[tuple[int, ...], torch.Tensor]:
        labels: list[int] = []
        weights: list[torch.Tensor] = []
        while (
            isinstance(value, RuntimeConstructor)
            and value.constructor == self.descriptor.chart.entry
        ):
            proof, weight, value = value.fields
            assert isinstance(weight, torch.Tensor) and weight.ndim == 0
            labels.append(self.lexical_choice(proof))
            weights.append(weight)
        assert isinstance(value, RuntimeConstructor)
        assert value.constructor == self.descriptor.chart.empty and value.fields == ()
        return tuple(labels), torch.stack(weights)

    def lexical_choice(self, proof: object) -> int:
        for j in reversed(range(self.depth)):
            assert isinstance(proof, RuntimeConstructor)
            assert self.proof_ids[proof.constructor.digest] == f"Paths__rule_wrap{j}"
            (proof,) = proof.fields
        assert isinstance(proof, RuntimeConstructor)
        choice = self.proof_ids[proof.constructor.digest]
        assert choice.startswith("Paths__rule_choose")
        (axiom,) = proof.fields
        assert isinstance(axiom, RuntimeConstructor)
        assert self.proof_ids[axiom.constructor.digest] == "Paths__axiom_span"
        assert axiom.fields == ("x",)
        return int(choice.removeprefix("Paths__rule_choose"))


def _fixture(case: Case) -> _Fixture:
    choices, depth = case
    module = load_source(_source(choices, depth)).qiec
    assert isinstance(module, QiecModule)
    (descriptor,) = (
        item.foreign
        for item in module.handlers
        if isinstance(item.foreign, DeductionProofs)
    )
    _, constructors = descriptor.proof_presentation.declarations()
    return _Fixture(
        module,
        descriptor,
        {item.id.digest: item.name for item in constructors},
        torch.linspace(-1.0, 0.8, choices, dtype=torch.float64),
        torch.arange(1, choices + 1, dtype=torch.float64) / (choices + 1),
        depth,
    )


class _ProofCalls(CallCounter):
    """Count proof-only node allocation at its actual constructor calls."""

    _model = Derivation.__init__

    def __init__(self, fixture: _Fixture) -> None:
        super().__init__(
            {
                "reified_proofs": deduction_outcomes.closed_derivation_value,
                "weight_products": type(LOG_PROB).tensor_op,
                "model_init": Derivation.__init__,
            }
        )
        del self.counts["model_init"]
        self.counts.update(
            derivation_nodes=0, proof_value_nodes=0, runtime_proof_nodes=0
        )
        self._proof_ids = fixture.proof_ids

    def _count(self, code: CodeType, offset: int) -> None:
        if code is not type(self)._model.__code__:
            super()._count(code, offset)
            return
        arguments = sys._getframe(1).f_locals
        candidate = arguments["self"]
        if isinstance(candidate, Derivation):
            self.counts["derivation_nodes"] += 1
        elif isinstance(candidate, (ConstructorValue, RuntimeConstructor)):
            constructor = arguments["kwargs"].get("constructor")
            if (
                isinstance(constructor, ConstructorId)
                and constructor.digest in self._proof_ids
            ):
                key = (
                    "proof_value_nodes"
                    if isinstance(candidate, ConstructorValue)
                    else "runtime_proof_nodes"
                )
                self.counts[key] += 1


def _expected_counts(fixture: _Fixture, positive: int) -> dict[str, int]:
    length = fixture.depth + 2
    return {
        "derivation_nodes": fixture.choices * length,
        "proof_value_nodes": positive * length,
        "runtime_proof_nodes": positive * length,
        "reified_proofs": positive,
        "weight_products": fixture.choices * (length - 1),
    }


def _refused(fixture: _Fixture) -> dict[str, int]:
    assert fixture.choices > 1
    logs = fixture.initial.clone()
    logs[0] = -torch.inf
    # Exactly one zero-weight proof survives the structural count but has
    # no output row when the cap allows it. Let the paired run demonstrate
    # that the allocation counters are active on the same checked route.
    accepted_work = _ProofCalls(fixture)
    with accepted_work:
        accepted = fixture.run(logs, fixture.choices)
    assert accepted_work.counts == _expected_counts(fixture, fixture.choices - 1)
    labels, _ = fixture.read(accepted.value)
    assert labels == tuple(range(1, fixture.choices))
    observed = []
    refused_work = _ProofCalls(fixture)
    with refused_work:
        try:
            fixture.run(logs, fixture.choices - 1, observer=observed.append)
        except ExecutionFailure as error:
            assert "too-many-derivations" in str(error), error
        else:
            raise AssertionError("the structural proof cap accepted too many proofs")
    (refusal,) = (
        event.detail for event in observed if event.event == "deduction.proofs.refused"
    )
    assert refusal["reason"] == "too-many-derivations"
    assert refusal["proof_budget"] == fixture.choices - 1 and refusal["goal_items"] == 1
    assert not any(event.event == "deduction.proofs" for event in observed)
    assert not any(refused_work.counts.values()), refused_work.counts
    return refused_work.counts


def _score(fixture: _Fixture, run: ProgramRun) -> tuple[torch.Tensor, torch.Tensor]:
    labels, masses = fixture.read(run.value)
    assert labels == tuple(range(fixture.choices))
    return masses, (masses + fixture.rates.log()).logsumexp(0)


def _refit(fixture: _Fixture, steps: int, *, verify: bool = False) -> None:
    weights = fixture.initial.clone().requires_grad_()
    for _ in range(steps):
        masses, response = _score(fixture, fixture.run(weights, fixture.choices))
        (gradient,) = torch.autograd.grad(response, weights)
        if verify:
            target = weights.detach()
            numerator = target + fixture.rates.log()
            torch.testing.assert_close(masses, target.log_softmax(0))
            torch.testing.assert_close(
                response, numerator.logsumexp(0) - target.logsumexp(0)
            )
            torch.testing.assert_close(
                gradient,
                numerator.softmax(0) - target.softmax(0),
                atol=1e-12,
                rtol=1e-12,
            )
        with torch.no_grad():
            weights.add_(gradient, alpha=STEP_SIZE)


def _work(fixture: _Fixture) -> dict[str, object]:
    weights = fixture.initial.clone().requires_grad_()
    runs: list[ProgramRun] = []

    def forward() -> None:
        runs.append(fixture.run(weights, fixture.choices))

    construction = _ProofCalls(fixture)
    with construction:
        forward_work = measure(
            forward, {"sort_arity_lookups": UserIndexSort.constructor_arity}
        )
    (run,) = runs
    _, response = _score(fixture, run)
    reverse_work = measure(lambda: torch.autograd.grad(response, weights)[0])
    (trace,) = (
        event.detail for event in run.result.trace if event.event == "deduction.proofs"
    )
    n, depth = fixture.choices, fixture.depth
    assert trace["algorithm"] == "indexed-packed-forest-budgeted-enumeration"
    assert trace["structural_goal_proofs"] == trace["positive_rows"] == n
    provenance = dict(trace["construction_work"])["provenance_records"]
    assert provenance == 3 * n + 2 * depth
    assert construction.counts == _expected_counts(fixture, n)
    saved_bound = 8 * n + 4
    assert forward_work.saved <= saved_bound, (forward_work.saved, saved_bound)
    (sort,) = fixture.descriptor.proof_presentation.index_sorts
    return {
        **construction.counts,
        "declared_index_symbols": len(sort.constructors),
        "forward_sort_arity_lookups": forward_work.calls["sort_arity_lookups"],
        "forest_edge_and_reference_records": provenance,
        "returned_proof_rows": trace["positive_rows"],
        "forward_scalar_operations": forward_work.scalar,
        "reverse_scalar_operations_including_likelihood": reverse_work.scalar,
        "forward_tensor_dispatches": forward_work.kernels,
        "reverse_tensor_dispatches_including_likelihood": reverse_work.kernels,
        "saved_logical_elements": forward_work.saved,
        "derived_saved_element_bound": saved_bound,
    }


def _benchmark(case: Case, steps: int, repeats: int) -> dict[str, object]:
    fixture = _fixture(case)
    _refit(fixture, steps, verify=True)

    def trial() -> None:
        _refit(fixture, steps)

    durations = []
    for _ in range(repeats):
        started = perf_counter()
        trial()
        durations.append(perf_counter() - started)
    allocation = memory(trial)
    tracemalloc.start()
    try:
        trial()
        _, peak_python = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "choices": case[0],
        "unary_suffix_depth": case[1],
        **_work(fixture),
        "below_cap": _refused(fixture),
        "elapsed_seconds_for_refits": durations,
        "median_seconds_for_refits": median(durations),
        "peak_live_cpu_tensor_bytes": allocation.peak_bytes,
        "total_cpu_tensor_allocated_bytes": allocation.allocated_bytes,
        "peak_traced_python_bytes": peak_python,
    }


def _parse_case(text: str) -> Case:
    if len(pieces := text.split(":")) != 2:
        raise argparse.ArgumentTypeError("expected CHOICES:DEPTH")
    try:
        choices, depth = map(int, pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError("case values must be integers") from error
    if choices < 2 or depth < 0:
        raise argparse.ArgumentTypeError(
            "require at least two choices and nonnegative depth"
        )
    return choices, depth


def main() -> None:
    """Report proof costs with an independent value, gradient, and cap oracle."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--cases",
        nargs="+",
        type=_parse_case,
        default=((4, 0), (4, 4), (4, 12), (8, 4), (16, 4), (16, 12)),
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    options = parser.parse_args()
    if options.steps < 2 or options.repeats < 1:
        parser.error("need at least two refit steps and one timed repetition")
    torch.set_num_threads(1)
    report = {
        "dtype": "float64",
        "device": "cpu",
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "steps": options.steps,
        "cases": [
            _benchmark(case, options.steps, options.repeats) for case in options.cases
        ],
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
