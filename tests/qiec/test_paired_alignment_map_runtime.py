"""Checked paired MAP preserves an exact original trace for each factor."""

from fractions import Fraction
from hashlib import sha256
from itertools import product

import didactic.api as dx
import pytest
import torch

from quivers.core.relations import alignment_groups
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    ExecutionFailure,
    JointAlignmentMapGapConfig,
    PairedAlignmentMapRequest,
    PairedAlignmentMapRuntimeProvider,
    QiecModule,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    paired_alignment_map_runtime,
    run_named,
)
from quivers.transpile import project_for_target
from tests.qiec.test_alignment_map_decomposition import (
    _MODULE,
    _SOURCE,
    _fixture as _portable_fixture,
    _owner,
    _primitive_oracle,
)
from tests.qiec.test_alignment_map_runtime import _fixture as _singleton_fixture
from tests.qiec.test_alignment_map_runtime import _read_ratios

_GRAPH = (
    _SOURCE.replace(
        "string_graph lexical",
        """relation silent [algebra=logprob, states=1]
    output Outer
    initial 0 [weight=quiet_mass]
    final 0 [weight=unit]

string_graph lexical
    variable quiet : Outer""",
    ).rstrip()
    + "\n    factor silent(quiet)\n"
)


def _fixture():
    _, parameters, policy = _portable_fixture()
    parameters[_owner("quiet_mass")] = torch.nn.Parameter(
        torch.zeros((), dtype=torch.float64)
    )
    graph = Compiler(parse(_GRAPH), module_name=_MODULE).string_graph_specs["lexical"]
    request = PairedAlignmentMapRequest(
        graph=graph,
        solver=policy.with_(groups=((1, 0), (2,), (3,))),
        word_capacity=2,
        max_assignment_cells=12,
        max_ratio_bits=4096,
        arc_capacity=3,
        max_trace_cells=30,
        max_trace_ratio_bits=4096,
    )
    return request, parameters


def _module(request):
    v, k = len(request.graph.variables), request.word_capacity
    f, a = len(request.graph.factors), request.arc_capacity
    g = len(request.solver.groups)
    common = (
        f"String * String * Bool * Tensor[Int]([{v}, {k}]) "
        f"* Tensor[Int]([{v}]) * Tensor[String]([4, 2]) * Bool * Bool * Tensor[Int]([5])"
    )
    report = (
        f"({common} * Tensor[Int]([{f}, 3]) * Tensor[Int]([{f}, {a}]) "
        f"* Tensor[String]([{f}, 2]) * Tensor[Int]([{g}, 2]))"
    )
    name = request.handler_name
    program = (
        _GRAPH
        + f"""
instance alignment : Compute[Unit, {report}]
handler {name} for Compute[Unit, {report}] : {report} -> {report} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define best : {report} !{{}} =
    handle alignment with {name} in
        let result <- perform alignment.apply(unit)
        return result
"""
    )
    module = Compiler(
        parse(module_to_source(parse(program))), module_name=_MODULE
    ).qiec_module
    decoded = loads(dumps(module))
    assert isinstance(decoded, QiecModule)
    return decoded


def _run(module, provider):
    return run_named(
        module, "best", runtime=RuntimeConfiguration(providers=(provider,))
    )


def _group_upper(graph, paths, groups):
    # Enumerate original paths inside each declared group only. The
    # fixture executes one iteration, so all multipliers are zero.
    upper = Fraction(0)
    observed = {name: word for name, _, _, word in graph.variables if word is not None}
    for group in groups:
        choices = []
        for local in product(*(paths[index] for index in group)):
            words = dict(observed)
            for index, (emitted, *_) in zip(group, local, strict=True):
                scope = graph.factors[index][1]
                if any(
                    words.setdefault(name, word) != word
                    for name, word in zip(scope, emitted, strict=True)
                ):
                    break
            else:
                choices.append(sum((path[-1] for path in local), Fraction(0)))
        upper += max(choices)
    return upper


def _check_evidence(run, request, parameters, status):
    assert run.result_type == request.result_type
    common = run.value[:9]
    metadata, arcs, scores, products = run.value[9:]
    actual, _, assigned, padded, lengths, ratios, certified, gap, work = common
    assert actual == status and assigned.item()
    assert certified.item() is (status == "optimal")
    assert gap.item() is (status == "gap")
    assert work[0].item() == 1
    assignment = tuple(
        (name, tuple(row[:size]))
        for (name, *_), row, size in zip(
            request.graph.variables, padded.tolist(), lengths.tolist(), strict=True
        )
    )
    words = dict(assignment)
    assert words["quiet"] == ()
    exact, original = _primitive_oracle(request.graph, parameters)
    lower, upper, absolute, relative = _read_ratios(ratios)
    assert lower == exact[assignment]
    assert upper == _group_upper(request.graph, original, request.solver.groups)
    assert lower <= max(exact.values()) <= upper
    assert absolute == upper - lower
    assert relative == absolute / abs(upper)
    assert products.tolist() == [[7, 8], [-1, -1], [-1, -1]]

    trace_scores = _read_ratios(scores)
    for (_, scope), (start, stop, length), row, score, paths in zip(
        request.graph.factors,
        metadata.tolist(),
        arcs.tolist(),
        trace_scores,
        original,
        strict=True,
    ):
        assert all(index == -1 for index in row[length:])
        assert (
            tuple(words[name] for name in scope),
            start,
            tuple(row[:length]),
            stop,
            score,
        ) in paths
    assert metadata.tolist() == [[1, 5, 2], [2, 4, 2], [0, 1, 1], [0, 0, 0]]
    assert arcs[-1].tolist() == [-1, -1, -1]
    assert sum(trace_scores) == lower
    return words, trace_scores, (lower, upper, absolute)


def test_authored_pair_retains_primitive_order_exact_boundaries_and_actual_products(
    monkeypatch,
):
    request, parameters = _fixture()
    wire = dumps(request)
    request = loads(wire)
    assert isinstance(request, PairedAlignmentMapRequest) and dumps(request) == wire
    assert request.objective == "joint_alignment_map"
    module = _module(request)
    provider = PairedAlignmentMapRuntimeProvider(request, parameters)
    products = []
    materialize = alignment_groups._materialize_product

    def observed(*args, **kwargs):
        topology, slots = materialize(*args, **kwargs)
        products.append((topology.n_states, len(topology.arcs)))
        return topology, slots

    monkeypatch.setattr(alignment_groups, "_materialize_product", observed)
    words, scores, bounds = _check_evidence(
        _run(module, provider), request, parameters, "optimal"
    )
    assert words["x"] == (0,)
    assert scores == (1, 1, Fraction(1, 4), 0)
    assert bounds == (Fraction(9, 4), Fraction(9, 4), 0)
    assert products == [(7, 8)]
    assert (parameters[_owner("large")] + parameters[_owner("small")]).item() == 1e16

    with torch.no_grad():
        parameters[_owner("l1")].fill_(2)
    _, _, bounds = _check_evidence(_run(module, provider), request, parameters, "limit")
    assert bounds == (Fraction(7, 2), Fraction(15, 4), Fraction(1, 4))
    gap = JointAlignmentMapGapConfig(
        solver=request.solver.solver, absolute_gap_ratio=(1, 4)
    )
    qualified = request.with_(solver=request.solver.with_(solver=gap))
    qualified = loads(dumps(qualified))
    assert isinstance(qualified, PairedAlignmentMapRequest)
    words, _, bounds = _check_evidence(
        _run(
            _module(qualified), PairedAlignmentMapRuntimeProvider(qualified, parameters)
        ),
        qualified,
        parameters,
        "gap",
    )
    assert words["x"] == (1,) and bounds[-1] == Fraction(1, 4)


def test_wrong_shared_word_provenance_fails_through_native_compute(monkeypatch):
    request, parameters = _fixture()
    module = _module(request)
    materialize = alignment_groups._materialize_product

    def wrong_shared_word(*args, **kwargs):
        result = materialize(*args, **kwargs)
        # Group order is (right, left). Replace right's shared-word zero
        # occurrence with its word-one occurrence at the same endpoints.
        origins = kwargs["arc_origins"]
        for index, origin in enumerate(origins):
            if origin == (0, 0):
                origins[index] = (0, 1)
        return result

    monkeypatch.setattr(alignment_groups, "_materialize_product", wrong_shared_word)
    with pytest.raises(ExecutionFailure, match="valid original trace"):
        _run(module, PairedAlignmentMapRuntimeProvider(request, parameters))


def _check_absent(run, code, diagnostic):
    common = run.value[:9]
    metadata, arcs, scores, products = run.value[9:]
    status, reason, assigned, words, lengths, ratios, certified, gap, counts = common
    assert status == code and diagnostic in reason
    assert not any(flag.item() for flag in (assigned, certified, gap))
    assert (metadata == -1).all() and (arcs == -1).all()
    assert (words == -1).all() and (lengths == -1).all()
    assert scores == (("", ""),) * 4 and ratios == (("", ""),) * 4
    return counts.tolist(), products.tolist()


def test_pair_resource_and_encoding_limits_and_absent_traces(monkeypatch):
    request, parameters = _fixture()
    with pytest.raises(dx.ValidationError, match="max_assignment_cells"):
        request.with_(max_assignment_cells=11)
    with pytest.raises(dx.ValidationError, match="max_trace_cells"):
        request.with_(max_trace_cells=29)
    for key in ("max_group_states", "max_group_work"):
        bounded = request.with_(solver=request.solver.with_(**{key: 0}))
        report = _run(
            _module(bounded), PairedAlignmentMapRuntimeProvider(bounded, parameters)
        )
        assert _check_absent(report, "resource_limit", "alignment pair") == (
            [0] * 5,
            [[-1, -1]] * 3,
        )
    short = request.with_(arc_capacity=1)
    report = _run(_module(short), PairedAlignmentMapRuntimeProvider(short, parameters))
    assert _check_absent(report, "report_limit", "arc_capacity")[0][0] == 1

    module = _module(request)
    with torch.no_grad():
        parameters[_owner("quiet_mass")].fill_(-torch.inf)
    unavailable = _run(module, PairedAlignmentMapRuntimeProvider(request, parameters))
    _, products = _check_absent(unavailable, "infeasible", "")
    assert products == [[7, 8], [-1, -1], [-1, -1]]

    with torch.no_grad():
        parameters[_owner("quiet_mass")].zero_()
    bits = request.with_(max_trace_ratio_bits=0)
    assert bits.handler_name == request.handler_name

    def must_preflight(_value):
        raise AssertionError("trace scores were formatted before their bit preflight")

    monkeypatch.setattr(paired_alignment_map_runtime, "_encode", must_preflight)
    report = _run(module, PairedAlignmentMapRuntimeProvider(bits, parameters))
    assert _check_absent(report, "report_limit", "max_trace_ratio_bits")[0][0] == 1


def test_same_shape_group_order_substitution_and_unsupported_target():
    request, parameters = _fixture()
    module = _module(request)
    changed = request.with_(solver=request.solver.with_(groups=((0, 1), (2,), (3,))))
    assert (
        changed.result_type == request.result_type
        and changed.handler_name != request.handler_name
    )
    with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
        _run(module, PairedAlignmentMapRuntimeProvider(changed, parameters))
    target = project_for_target(module, discharge_module(module), "stan")
    assert any(
        obstacle.computation == "best" and request.handler_name in obstacle.reason
        for obstacle in target.obstacles
    )


def test_original_singleton_gap_and_paired_policy_wire_remain_frozen():
    ordinary, _ = _singleton_fixture()
    gap = ordinary.with_(
        solver=JointAlignmentMapGapConfig(
            solver=ordinary.solver, absolute_gap_ratio=(2, 1)
        )
    )
    for request, digest, handler in (
        (
            ordinary,
            "3419653740a3aaee45584101e4a6606d7ed3ede41f1362d011197c5df09c5685",
            "__joint_alignment_map_be130ca5a4caf92909dfb73e88bcc4a004b97a4e74d15793ea0c5fa60e0ad4dc",
        ),
        (
            gap,
            "47c7a73c67979faad79ebc8d9875bc92c510994ddc0736d46cf810fa541369c9",
            "__joint_alignment_map_441a4a9399eea3691f109dbd9cd93a221018aa05c487388a35f510e9593c35a8",
        ),
    ):
        assert sha256(dumps(request).encode()).hexdigest() == digest
        assert request.handler_name == handler
    _, _, portable = _portable_fixture()
    assert sha256(dumps(portable).encode()).hexdigest() == (
        "fb218dd5c219d8d455be092fc1760f0d7a3a662a23949604d697303be75986cd"
    )
