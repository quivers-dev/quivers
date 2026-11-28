"""Measure typed-key pushforward through a checked lexical QVR program.

Run `python -m tests.benchmarks.deduction_key_grouping` from the repository
root. A case N:K:D has N one-axiom items, K outcome keys, and D unary
constructors above each integer leaf. Item i maps to label i mod K.
Every occurrence constructs its own key. Source preparation runs before
any timed or profiled work.

The whole query includes chart preparation, projection, and grouping,
with execution traces enabled. The isolated checked invocation installs
the same generated provider on fresh preprojected rows with independent
weight leaves. Its measurements include native entry preparation and
provider attachment and exclude chart and key preparation. Forward
dispatch and tape counts end at the returned rows. Reverse counts and
refit timings include a scalar host likelihood over those rows. Peak live CPU tensor bytes use
the profiler's initial allocation balance as their baseline.

The selector checks the span arm first, then scans the K literal label
arms. It takes O(N(K+D+1)) work on this fixture, including N(D+1) key
constructors. For L = D+1, cold key construction and first hashing take
O(NL) work and space. Later hashes of those objects reuse cached results.
In a collision-free dictionary model the two setdefault calls compare
each of N-K duplicate keys with its representative, using 2(N-K)L
structural visits. Colliding distinct keys can require O(NKL) work.
The script checks exact key events when the observed distinct labels'
hashes differ. These are interpreter-level counts on this fixture.

The reducer reads N weights in K separate reductions, then K masses
in one normalizer. Its arithmetic and backward state are O(N+K). A
masked logsumexp on m elements saves at most m+3 elements through the
input, result, and two Boolean masks. The K initial reductions and their
finiteness checks save N+4K. The normalizer and final finite checks save
2K+4, giving N+6K+4 for the standalone reducer. Input rows can retain
O(NL) key nodes. The output retains one representative per key. Allocator
peaks are separate measurements from the logical tape bound.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
from statistics import median
import sys
from time import perf_counter
import tracemalloc
from types import CodeType

import torch

from quivers._values import ValueModel
from quivers.core.algebras import LOG_PROB
from quivers.dsl import loads as load_source
from quivers.qiec import (
    ExecutionResult,
    ProgramRun,
    QiecModule,
    RuntimeConfiguration,
    RuntimeSelection,
    run_named,
    run_program,
)
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.foreign import DeductionOutcomeGrouping
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.program_runtime import foreign_handler_configuration
from tests.evidence.counters import CallCounter, measure, memory

STEP_SIZE = 0.03
type Case = tuple[int, int, int]
type Row = tuple[RuntimeConstructor, torch.Tensor]
type ReadRows = Callable[[torch.Tensor], object]


def _source(items: int, keys: int, depth: int) -> str:
    categories = [f"C{i}" for i in range(items)]
    labels = [f"Tag{k}" for k in range(keys)]
    shapes = ["span", "Missing", "Unused", *categories]
    atoms = shapes + labels
    lexicon = "\n".join(
        f'        "x" : {category} = {labels[i % keys]} #[learnable]'
        for i, category in enumerate(categories)
    )
    # The unreachable rule declares the form field on a lexical span.
    return f"""\
object Token : FinSet 1
object Result : Real 1

deduction Groups : Token -> Token [semiring=LogProb]
    atoms {", ".join(atoms)}
    rule witness : span(I, J, Missing, Tag0) |- span(I, J, Unused, Tag0)
    lexicon
{lexicon}

family Key : Type
    constructor Leaf : Int -> Key
    constructor Box : Key -> Key

define ordinal(form : Groups__Item) : Int !{{}} =
    case form motive => Int
{_branches(labels + shapes, lambda name: name[3:] if name in labels else "-1")}

define wrap(depth : Int, key : Key) : Key !{{}} =
    if depth == 0 then
        return key
    else
        let next = construct Box(key) as Key
        let result <- wrap(depth - 1, next)
        return result

define classify(item : Groups__Item) : Key !{{}} =
    case item motive => Key
        Groups__span(i, j, cat, lf) =>
            let label <- ordinal(lf)
            let leaf = construct Leaf(label) as Key
            let result <- wrap({depth}, leaf)
            return result
{_branches(atoms[1:], lambda _name: "construct Leaf(-1) as Key")}

define isolated(rows : Groups__classify__Outcomes) : Groups__classify__Outcomes !{{}} =
    handle Groups__classify__outcome_grouping with Groups__classify__group_outcomes in
        perform Groups__classify__outcome_grouping.group(rows, true)

program grouped : Token -> Result
    let rows = parse_normalized_outcomes(Groups, sentence, classify)
    return rows

program raw : Token -> Result
    let rows = parse_outcomes(Groups, sentence, "item")
    return rows

export grouped
"""


def _branches(names: list[str], answer: Callable[[str], str]) -> str:
    return "\n".join(
        f"        Groups__{name}{'(i, j, cat, lf)' if name == 'span' else ''} =>\n"
        f"            return {answer(name)}"
        for name in names
    )


def _rows(value: object, entry: ConstructorId, empty: ConstructorId) -> list[Row]:
    rows: list[Row] = []
    while isinstance(value, RuntimeConstructor) and value.constructor == entry:
        key, weight, value = value.fields
        assert isinstance(key, RuntimeConstructor)
        assert isinstance(weight, torch.Tensor) and weight.ndim == 0
        rows.append((key, weight))
    assert isinstance(value, RuntimeConstructor) and value.constructor == empty
    return rows


@dataclass(frozen=True)
class _Fixture:
    module: QiecModule
    layout: DeductionOutcomeGrouping
    runtime: RuntimeConfiguration
    raw: RuntimeConstructor
    box: ConstructorId
    leaf: ConstructorId
    initial: torch.Tensor
    labels: torch.Tensor
    rates: torch.Tensor
    depth: int

    def read(self, value: object) -> list[Row]:
        return _rows(value, self.layout.entry, self.layout.empty)

    def labels_of(self, rows: list[Row]) -> list[int]:
        labels = []
        for key, _ in rows:
            for _ in range(self.depth):
                assert key.constructor == self.box, "unexpected key depth"
                (child,) = key.fields
                assert isinstance(child, RuntimeConstructor)
                key = child
            assert key.constructor == self.leaf, "unexpected key depth"
            (label,) = key.fields
            assert isinstance(label, torch.Tensor)
            labels.append(int(label.item()))
        return labels

    def weights(self, returned: object, *, check_keys: bool = True) -> torch.Tensor:
        rows = self.read(returned)
        if check_keys:
            assert self.labels_of(rows) == list(range(self.rates.numel()))
        return torch.stack([weight for _, weight in rows])

    def project(self) -> RuntimeConstructor:
        result = run_named(
            self.module,
            "Groups__classify__map_outcomes",
            (self.raw,),
            record_trace=False,
        )
        assert isinstance(result.value, RuntimeConstructor)
        return result.value

    def bind_rows(self, log_weights: torch.Tensor) -> RuntimeConstructor:
        projected = self.project()
        rows = self.read(projected)
        tail = RuntimeConstructor(
            constructor=self.layout.empty,
            static_arguments=(),
            fields=(),
            result_type=projected.result_type,
        )
        for (key, _), log_weight in reversed(
            tuple(zip(rows, log_weights.unbind(), strict=True))
        ):
            tail = RuntimeConstructor(
                constructor=self.layout.entry,
                static_arguments=(),
                fields=(key, log_weight, tail),
                result_type=projected.result_type,
            )
        return tail


def _run(module: QiecModule, entry: str, log_weights: torch.Tensor) -> ProgramRun:
    return run_program(
        module,
        entry,
        data={"sentence": ("x",)},
        sites={},
        parameters={
            f"Groups.lex.{i}": weight for i, weight in enumerate(log_weights.unbind())
        },
    )


def _fixture(case: Case) -> _Fixture:
    n, k, d = case
    module = load_source(_source(n, k, d)).qiec
    assert isinstance(module, QiecModule)
    (handler,) = (
        item
        for item in module.handlers
        if isinstance(item.foreign, DeductionOutcomeGrouping)
    )
    layout = handler.foreign
    assert isinstance(layout, DeductionOutcomeGrouping)
    runtime = RuntimeConfiguration(
        selections=(
            RuntimeSelection(
                provider="core",
                options={
                    "handlers": foreign_handler_configuration(
                        module, reached=frozenset({handler.id})
                    )
                },
            ),
        )
    )
    initial = torch.linspace(-1.1, 0.7, n, dtype=torch.float64)
    raw = _run(module, "raw", initial).value
    assert isinstance(raw, RuntimeConstructor)
    constructor_ids = {item.name: item.id for item in module.constructors}
    fixture = _Fixture(
        module,
        layout,
        runtime,
        raw,
        constructor_ids["Box"],
        constructor_ids["Leaf"],
        initial,
        torch.arange(n) % k,
        torch.arange(1, k + 1, dtype=torch.float64) / (k + 1),
        d,
    )
    # Check each source item separately, before assigning fresh inputs to
    # the reducer. The distinct initial logits identify the source slots.
    projected = fixture.read(fixture.project())
    assert fixture.labels_of(projected) == fixture.labels.tolist()
    torch.testing.assert_close(
        torch.stack([weight for _, weight in projected]), initial
    )
    return fixture


def _score(log_masses: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(log_masses + rates.log(), 0)


def _oracle(
    fixture: _Fixture, log_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_weights = log_weights.detach()
    rates = fixture.rates[fixture.labels]
    log_total = log_weights.logsumexp(0)
    log_masses = torch.stack(
        [
            log_weights[fixture.labels == label].logsumexp(0) - log_total
            for label in range(fixture.rates.numel())
        ]
    )
    log_response = log_weights + rates.log()
    return (
        log_masses,
        log_response.logsumexp(0) - log_total,
        log_response.softmax(0) - log_weights.softmax(0),
    )


def _assert_oracle(
    fixture: _Fixture,
    log_weights: torch.Tensor,
    masses: torch.Tensor,
    grad: torch.Tensor,
) -> None:
    expected_masses, expected_score, expected_grad = _oracle(fixture, log_weights)
    torch.testing.assert_close(masses, expected_masses, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        _score(masses, fixture.rates), expected_score, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(grad, expected_grad, atol=1e-12, rtol=1e-12)


class _StageCalls(CallCounter):
    """Attribute the shared counters to key constructors and LogProb reductions."""

    _join = type(LOG_PROB).join

    def __init__(self, fixture: _Fixture) -> None:
        super().__init__(
            {
                "cold_key_nodes": RuntimeConstructor._local_key,
                "key_comparisons": ValueModel.__eq__,
                "key_local_field_visits": ValueModel._comparison,
                "log_reductions": type(LOG_PROB).join,
            }
        )
        self._constructors = frozenset((fixture.leaf.digest, fixture.box.digest))
        self.reduction_lanes = 0

    def _count(self, code: CodeType, offset: int) -> None:
        # PY_START observes the actual receiver. Inspecting its immutable
        # constructor digest does not hash or compare the key itself.
        arguments = sys._getframe(1).f_locals
        if code is type(self)._join.__code__:
            values = arguments["t"]
            assert isinstance(values, torch.Tensor)
            self.reduction_lanes += values.numel()
            super()._count(code, offset)
            return
        receiver = arguments.get("self")
        if (
            isinstance(receiver, RuntimeConstructor)
            and receiver.constructor.digest in self._constructors
        ):
            super()._count(code, offset)


def _isolate(
    fixture: _Fixture, rows: RuntimeConstructor, *, traced: bool = False
) -> ExecutionResult:
    return run_named(
        fixture.module,
        "isolated",
        (rows,),
        runtime=fixture.runtime,
        record_trace=traced,
    )


def _schedule(
    result: ExecutionResult, items: int, keys: int, length: int, *, whole: bool
) -> dict[str, object]:
    (trace,) = (
        event.detail
        for event in result.trace
        if event.event == "deduction.grouped_outcomes"
    )
    expected = (items, keys, keys)
    actual = tuple(
        trace[field] for field in ("input_rows", "distinct_keys", "positive_rows")
    )
    assert actual == expected, (actual, expected)
    report = {
        "algorithm": trace["algorithm"],
        **dict(
            zip(("input_rows", "distinct_keys", "positive_rows"), actual, strict=True)
        ),
    }
    if whole:
        (chart,) = (
            event.detail
            for event in result.trace
            if event.event == "deduction.outcomes"
        )
        provenance = dict(chart["construction_work"])["provenance_records"]
        assert (chart["goal_items"], chart["positive_rows"], provenance) == (items,) * 3
        report["chart_provenance_records"] = provenance
        projection_calls = sum(
            event.event == "call.entered" and event.detail["computation"] == "classify"
            for event in result.trace
        )
        wrapper_calls = sum(
            event.event == "call.entered" and event.detail["computation"] == "wrap"
            for event in result.trace
        )
        assert (projection_calls, wrapper_calls) == (items, items * length)
        report["source_projection_calls"] = projection_calls
        report["source_key_constructor_steps"] = wrapper_calls
    return report


def _forward_work(fixture: _Fixture, *, isolated: bool) -> dict[str, object]:
    weights = fixture.initial.clone().requires_grad_()
    rows = fixture.bind_rows(weights) if isolated else None
    runs: list[tuple[object, ExecutionResult]] = []

    def forward() -> None:
        if rows is None:
            run = _run(fixture.module, "grouped", weights)
            runs.append((run.value, run.result))
        else:
            run = _isolate(fixture, rows, traced=True)
            runs.append((run.value, run))

    key_work = _StageCalls(fixture)
    with key_work:
        forward_work = measure(forward)
    ((value, result),) = runs
    masses = fixture.weights(value)
    score = _score(masses, fixture.rates)
    reverse_work = measure(lambda: torch.autograd.grad(score, weights)[0])
    n, k, length = weights.numel(), fixture.rates.numel(), fixture.depth + 1
    schedule = _schedule(result, n, k, length, whole=not isolated)
    assert key_work.counts["cold_key_nodes"] == n * length, key_work.counts
    hashes = {hash(key) for key, _ in fixture.read(value)}
    if len(hashes) == k:
        assert key_work.counts["key_comparisons"] == 2 * (n - k), key_work.counts
        assert key_work.counts["key_local_field_visits"] == (
            2 * n * length + 4 * (n - k) * length
        ), key_work.counts
    bounds: dict[str, object] = {}
    if isolated:
        bound = n + 6 * k + 4
        assert forward_work.saved <= bound, (forward_work.saved, bound)
        assert key_work.counts["log_reductions"] == k + 1
        assert key_work.reduction_lanes == n + k
        bounds["derived_reduction_saved_element_bound"] = bound
    return {
        **schedule,
        **key_work.counts,
        **bounds,
        "observed_distinct_output_hashes": len(hashes),
        "log_reduction_lanes": key_work.reduction_lanes,
        "output_key_nodes": k * length,
        "forward_tensor_scalar_operations": forward_work.scalar,
        "reverse_tensor_scalar_operations_including_likelihood": reverse_work.scalar,
        "forward_tensor_dispatches": forward_work.kernels,
        "reverse_tensor_dispatches_including_likelihood": reverse_work.kernels,
        "forward_saved_logical_elements": forward_work.saved,
    }


def _fit(
    fixture: _Fixture, read: ReadRows, steps: int, *, verify: bool = False
) -> None:
    weights = fixture.initial.clone().requires_grad_()
    for _ in range(steps):
        masses = fixture.weights(read(weights), check_keys=verify)
        (gradient,) = torch.autograd.grad(_score(masses, fixture.rates), weights)
        if verify:
            _assert_oracle(fixture, weights, masses, gradient)
        with torch.no_grad():
            weights.add_(gradient, alpha=STEP_SIZE)


def _peak_python(body: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        body()
        _, peak = tracemalloc.get_traced_memory()
        return peak
    finally:
        tracemalloc.stop()


def _benchmark(case: Case, *, steps: int, repeats: int) -> dict[str, object]:
    fixture = _fixture(case)

    def whole(weights: torch.Tensor) -> object:
        return _run(fixture.module, "grouped", weights).value

    def reduce(weights: torch.Tensor) -> object:
        return _isolate(fixture, fixture.bind_rows(weights)).value

    # Both routes check each live refit point. In isolation, fixed weights
    # and cold projected keys are prepared outside the profiled operation.
    _fit(fixture, whole, steps, verify=True)
    _fit(fixture, reduce, steps, verify=True)

    def full_trial() -> None:
        _fit(fixture, whole, steps)

    durations = []
    for _ in range(repeats):
        started = perf_counter()
        full_trial()
        durations.append(perf_counter() - started)
    full_mem = memory(full_trial)

    def isolated_gradient(
        rows: RuntimeConstructor, weights: torch.Tensor
    ) -> torch.Tensor:
        masses = fixture.weights(_isolate(fixture, rows).value, check_keys=False)
        return torch.autograd.grad(_score(masses, fixture.rates), weights)[0]

    isolated_durations = []
    for _ in range(repeats):
        weights = fixture.initial.clone().requires_grad_()
        elapsed = 0.0
        for _ in range(steps):
            rows = fixture.bind_rows(weights)
            started = perf_counter()
            gradient = isolated_gradient(rows, weights)
            with torch.no_grad():
                weights.add_(gradient, alpha=STEP_SIZE)
            elapsed += perf_counter() - started
        isolated_durations.append(elapsed)

    isolated_weights = fixture.initial.clone().requires_grad_()
    prepared_rows = fixture.bind_rows(isolated_weights)
    isolated_mem = memory(lambda: isolated_gradient(prepared_rows, isolated_weights))
    prepared_rows = fixture.bind_rows(isolated_weights)
    isolated_python = _peak_python(
        lambda: isolated_gradient(prepared_rows, isolated_weights)
    )
    return {
        "items": case[0],
        "keys": case[1],
        "wrapper_depth": case[2],
        "whole_query": {
            **_forward_work(fixture, isolated=False),
            "elapsed_seconds_for_refits": durations,
            "median_seconds_for_refits": median(durations),
            "peak_live_cpu_tensor_bytes": full_mem.peak_bytes,
            "total_cpu_tensor_allocated_bytes": full_mem.allocated_bytes,
            "peak_traced_python_bytes": _peak_python(full_trial),
        },
        "isolated_reduction": {
            **_forward_work(fixture, isolated=True),
            "elapsed_seconds_for_refits": isolated_durations,
            "median_seconds_for_refits": median(isolated_durations),
            "peak_live_cpu_tensor_bytes_one_update": isolated_mem.peak_bytes,
            "total_cpu_tensor_allocated_bytes_one_update": isolated_mem.allocated_bytes,
            "peak_traced_python_bytes_one_update": isolated_python,
        },
    }


def _parse_case(text: str) -> Case:
    if len(pieces := text.split(":")) != 3:
        raise argparse.ArgumentTypeError("expected ITEMS:KEYS:DEPTH")
    try:
        items, keys, depth = map(int, pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError("case counts must be integers") from error
    if not (1 <= keys <= items and depth >= 0):
        raise argparse.ArgumentTypeError("require 1 <= keys <= items and depth >= 0")
    return items, keys, depth


def main() -> None:
    """Run the checked grouping benchmark and print its measurements."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--cases",
        type=_parse_case,
        nargs="+",
        default=(
            (8, 4, 4),
            (16, 4, 4),
            (16, 1, 4),
            (16, 16, 4),
            (16, 4, 0),
            (16, 4, 12),
        ),
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    options = parser.parse_args()
    if options.steps < 2 or options.repeats < 1:
        parser.error("need at least two refit steps and one timed repetition")
    torch.set_num_threads(1)
    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "dtype": "float64",
                "device": "cpu",
                "cpu_threads": torch.get_num_threads(),
                "steps": options.steps,
                "step_size": STEP_SIZE,
                "cases": [
                    _benchmark(case, steps=options.steps, repeats=options.repeats)
                    for case in options.cases
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
