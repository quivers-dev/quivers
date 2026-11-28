"""Run exact joint MAP through authored Compute against a rational path oracle."""

from fractions import Fraction
from hashlib import sha256
from itertools import product

import didactic.api as dx
import pytest
import torch

from quivers.core.relations.alignment_dual import _DualAlignmentRun
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    ExecutionFailure,
    JointAlignmentMapConfig,
    JointAlignmentMapGapConfig,
    JointAlignmentMapRequest,
    JointAlignmentMapRuntimeProvider,
    ParameterId,
    QiecModule,
    RuntimeConfiguration,
    alignment_map_runtime,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target

_MODULE = "checked_alignment_map"
_GRAPH = """object Letter : FinSet 2

relation left [algebra=logprob, states=2]
    output Letter
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0), weight=left_zero]
    arc 0 -> 1 [labels=(0), weight=left_zero]
    arc 0 -> 1 [labels=(1), weight=left_one]
    final 1 [weight=unit]

relation right [algebra=logprob, states=2]
    output Letter
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0), weight=right_zero]
    arc 0 -> 1 [labels=(1), weight=right_one]
    final 1 [weight=unit]

relation blank [algebra=logprob, states=1]
    output Letter
    initial 0 [weight=unit]
    final 0 [weight=unit]

string_graph choice
    variable y : Letter
    variable x : Letter
    factor left(x)
    factor right(x)
    factor blank(y)
"""
_EMPTY = """object Letter : FinSet 2

relation blank [algebra=logprob, states=1]
    output Letter
    initial 0 [weight=unit]
    final 0 [weight=unit]

string_graph choice
    variable y : Letter
    factor blank(y)
"""
_TINY = Fraction(1, 2**80)


def _owner(name):
    return ParameterId.derive(_MODULE, "relation_weights", name)


def _fixture(source=_GRAPH):
    compiler = Compiler(parse(source), module_name=_MODULE)
    graph = compiler.string_graph_specs["choice"]
    solver = JointAlignmentMapConfig(
        max_iterations=1, max_ngram_order=1, max_features=0, max_states=16, max_work=32
    )
    request = JointAlignmentMapRequest(
        graph=graph,
        solver=solver,
        word_capacity=2,
        max_assignment_cells=6,
        max_ratio_bits=2048,
    )
    parameters = {
        _owner(name): torch.nn.Parameter(
            torch.tensor(float(value), dtype=torch.float64)
        )
        for name, value in (
            ("unit", 0),
            ("left_zero", -_TINY),
            ("left_one", -2),
            ("right_zero", -3),
            ("right_one", -1),
        )
    }
    return request, parameters


def _module(request, source=_GRAPH):
    n = len(request.graph.variables)
    report = (
        f"(String * String * Bool * Tensor[Int]([{n}, {request.word_capacity}]) "
        f"* Tensor[Int]([{n}]) * Tensor[String]([4, 2]) * Bool * Bool * Tensor[Int]([5]))"
    )
    handler = request.handler_name
    program = (
        source
        + f"""
instance alignment : Compute[Unit, {report}]
handler {handler} for Compute[Unit, {report}] : {report} -> {report} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define best : {report} !{{}} =
    handle alignment with {handler} in
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
    runtime = RuntimeConfiguration(providers=(provider,))
    return run_named(module, "best", runtime=runtime)


def _read_ratios(rows):
    return tuple(
        None if not numerator else Fraction(int(numerator, 16), int(denominator, 16))
        for numerator, denominator in rows
    )


def _oracle(parameters, observed=None):
    def weight(name):
        return Fraction(parameters[_owner(name)].item())

    left = ((0, weight("left_zero")), (0, weight("left_zero")), (1, weight("left_one")))
    right = ((0, weight("right_zero")), (1, weight("right_one")))
    left, right = tuple(
        tuple(
            (word, score)
            for word, score in factor
            if observed is None or word == observed
        )
        for factor in (left, right)
    )
    scores = {
        (word, score + other)
        for (word, score), (opposite, other) in product(left, right)
        if word == opposite
    }
    optimum = max(value for _, value in scores)
    separate = max(value for _, value in left) + max(value for _, value in right)
    return dict(scores), optimum, separate


def _check_report(
    run, request, parameters, expected_status, expected_word, observed=None
):
    assert run.result_type == request.result_type
    status, reason, present, words, lengths, ratios, certificate, gap, counts = (
        run.value
    )
    assert status == expected_status and not reason and present.item()
    assert words.tolist() == [[-1, -1], [expected_word, -1]]
    assert lengths.tolist() == [0, 1]
    chosen, optimum, upper = _oracle(parameters, observed)
    lower = chosen[expected_word]
    expected = (lower, upper, upper - lower, (upper - lower) / abs(upper))
    assert _read_ratios(ratios) == expected, "reported exact scores were narrowed"
    assert lower <= optimum <= upper
    assert certificate.item() is (status == "optimal")
    assert gap.item() is (status == "gap")
    assert all(value > 0 for value in counts.tolist())
    assert not any(field.requires_grad for field in (words, lengths, counts))


def _set_weight(parameters, name, value):
    with torch.no_grad():
        parameters[_owner(name)].fill_(float(value))


def test_live_native_exact_bound_gap_and_agreement_on_original_graph():
    request, parameters = _fixture()
    wire = dumps(request)
    request = loads(wire)
    assert isinstance(request, JointAlignmentMapRequest) and dumps(request) == wire
    assert request.objective == "joint_alignment_map"
    module = _module(request)
    provider = JointAlignmentMapRuntimeProvider(request, parameters)
    _check_report(_run(module, provider), request, parameters, "limit", 0)

    gap = request.with_(
        solver=JointAlignmentMapGapConfig(
            solver=request.solver, absolute_gap_ratio=(2, 1)
        )
    )
    decoded = loads(dumps(gap))
    assert isinstance(decoded, JointAlignmentMapRequest) and decoded == gap
    _check_report(
        _run(_module(decoded), JointAlignmentMapRuntimeProvider(decoded, parameters)),
        decoded,
        parameters,
        "gap",
        0,
    )

    _set_weight(parameters, "right_zero", Fraction(-1, 2))
    result = _run(module, provider)
    _check_report(result, request, parameters, "optimal", 0)
    assert _read_ratios(result.value[5])[0] == -Fraction(1, 2) - _TINY
    assert parameters[_owner("left_zero")].grad is None


def test_exact_native_oracle_rejects_float64_report_narrowing(monkeypatch):
    request, parameters = _fixture()
    _set_weight(parameters, "right_zero", Fraction(-1, 2))
    module = _module(request)
    encode = alignment_map_runtime._encode

    def narrow(value):
        return encode(None if value is None else Fraction(float(value)))

    monkeypatch.setattr(alignment_map_runtime, "_encode", narrow)
    with pytest.raises(AssertionError, match="scores were narrowed"):
        _check_report(
            _run(module, JointAlignmentMapRuntimeProvider(request, parameters)),
            request,
            parameters,
            "optimal",
            0,
        )


def test_authored_agreement_always_reaches_the_independent_local_verifier(monkeypatch):
    request, parameters = _fixture()
    _set_weight(parameters, "right_zero", Fraction(-1, 2))
    module = _module(request)
    decode = _DualAlignmentRun.decode

    def force_inferior_word(run, observations, multipliers):
        fixed = (
            observations
            if observations["x"] is not None
            else observations | {"x": (1,)}
        )
        return decode(run, fixed, multipliers)

    monkeypatch.setattr(_DualAlignmentRun, "decode", force_inferior_word)
    with pytest.raises(
        ExecutionFailure, match="does not attain the independent local optimum"
    ):
        _run(module, JointAlignmentMapRuntimeProvider(request, parameters))


def _check_unavailable(run, request, status, diagnostic):
    assert run.result_type == request.result_type
    code, reason, assigned, words, lengths, ratios, certificate, gap, counts = run.value
    assert code == status and diagnostic in reason
    assert not any(flag.item() for flag in (assigned, certificate, gap))
    assert words.shape == (len(request.graph.variables), request.word_capacity)
    assert (words == -1).all() and (lengths == -1).all()
    assert ratios == (("", ""),) * 4
    return counts.tolist()


def test_static_shape_and_runtime_solver_and_payload_limits_are_distinct(monkeypatch):
    request, parameters = _fixture()
    with pytest.raises(dx.ValidationError, match="max_assignment_cells"):
        request.with_(max_assignment_cells=5)
    for setting in ("max_states", "max_work"):
        restricted = request.with_(solver=request.solver.with_(**{setting: 0}))
        report = _run(
            _module(restricted),
            JointAlignmentMapRuntimeProvider(restricted, parameters),
        )
        assert (
            _check_unavailable(report, restricted, "resource_limit", setting) == [0] * 5
        )
    too_short = request.with_(word_capacity=0)
    report = _run(
        _module(too_short), JointAlignmentMapRuntimeProvider(too_short, parameters)
    )
    assert (
        _check_unavailable(report, too_short, "report_limit", "word_capacity")[0] == 1
    )

    too_wide = request.with_(max_ratio_bits=32)
    assert too_wide.handler_name == request.handler_name

    def must_preflight(_value):
        raise AssertionError(
            "a rejected integer was formatted before the bit preflight"
        )

    monkeypatch.setattr(alignment_map_runtime, "_encode", must_preflight)
    report = _run(
        _module(too_wide), JointAlignmentMapRuntimeProvider(too_wide, parameters)
    )
    assert (
        _check_unavailable(report, too_wide, "report_limit", "max_ratio_bits")[0] == 1
    )


def test_absent_assignment_and_feasible_empty_word_have_distinct_encodings():
    empty, parameters = _fixture(_EMPTY)
    empty = empty.with_(word_capacity=0)
    run = _run(
        _module(empty, _EMPTY), JointAlignmentMapRuntimeProvider(empty, parameters)
    )
    status, _, assigned, words, lengths, ratios, certified, gap, _ = run.value
    assert (
        status == "optimal" and assigned.item() and certified.item() and not gap.item()
    )
    assert words.shape == (1, 0) and lengths.tolist() == [0]
    assert _read_ratios(ratios) == (0, 0, 0, 0)

    request, parameters = _fixture()
    module = _module(request)
    provider = JointAlignmentMapRuntimeProvider(request, parameters)
    _set_weight(parameters, "left_one", -torch.inf)
    _set_weight(parameters, "right_zero", -torch.inf)
    run = _run(module, provider)
    status, reason, assigned, words, lengths, ratios, certified, gap, _ = run.value
    assert (status, reason, assigned.item(), certified.item(), gap.item()) == (
        "limit",
        "",
        False,
        False,
        False,
    )
    assert words.tolist() == [[-1, -1], [-1, -1]]
    assert lengths.tolist() == [-1, -1]
    assert _read_ratios(ratios) == (None, -1 - _TINY, None, None)

    _set_weight(parameters, "right_one", -torch.inf)
    _check_unavailable(_run(module, provider), request, "infeasible", "")


def test_same_shape_observation_owner_and_solver_substitutions_refuse():
    request, parameters = _fixture()
    module = _module(request)
    source = _GRAPH.replace("variable x : Letter", "variable x : Letter [observed=(1)]")
    observed, _ = _fixture(source)

    alias = _owner("different_left_zero")
    old = _owner("left_zero")
    relations = tuple(
        (
            name,
            spec.with_(
                weight_parameters=tuple(
                    alias if owner == old else owner for owner in spec.weight_parameters
                )
            ),
        )
        if name == "left"
        else (name, spec)
        for name, spec in request.graph.relations
    )
    substitute = request.with_(graph=request.graph.with_(relations=relations))
    alias_weights = {
        alias if key == old else key: value for key, value in parameters.items()
    }
    for other, weights in (
        (observed, parameters),
        (substitute, alias_weights),
        (request.with_(solver=request.solver.with_(max_iterations=2)), parameters),
    ):
        assert (
            other.result_type == request.result_type
            and other.handler_name != request.handler_name
        )
        with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
            _run(module, JointAlignmentMapRuntimeProvider(other, weights))

    _check_report(
        _run(
            _module(observed, source),
            JointAlignmentMapRuntimeProvider(observed, parameters),
        ),
        observed,
        parameters,
        "optimal",
        1,
        observed=1,
    )
    target = project_for_target(module, discharge_module(module), "stan")
    assert any(
        item.computation == "best" and request.handler_name in item.reason
        for item in target.obstacles
    )


def test_original_graph_and_solver_wires_are_frozen():
    request, _ = _fixture()
    assert (
        sha256(dumps(request.graph).encode()).hexdigest()
        == "4a6e4d2b6c83bb5be3e4b5c50104df6422f5e4303438b2903ef2328cd11c4f01"
    )
    assert (
        sha256(dumps(request.solver).encode()).hexdigest()
        == "dd0701367dab7d4f46f14a6ab4c13c1539008ebcfd896a19eca7b4aa4bb10ca1"
    )
