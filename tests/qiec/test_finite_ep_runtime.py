"""Independent finite-update EP scores and derivatives through checked QVR."""

from math import exp, log

import didactic.api as dx
import pytest
import torch

from quivers.core.relations import FiniteStringBeliefPlan, NgramFamily
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    ExecutionFailure,
    FiniteStringEPRequest,
    FiniteStringEPRuntimeProvider,
    ParameterId,
    QiecModule,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target

_MODULE = "finite_ep_program"
_GRAPH = """object Segments : FinSet 2

relation lexicon [algebra=logprob, states=2]
    output Segments
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0), weight=shared]
    arc 0 -> 1 [labels=(0), weight=shared]
    arc 0 -> 1 [labels=(1), weight=unit]
    final 1 [weight=unit]

string_graph latent
    variable word : Segments
    factor lexicon(word)
"""


def _fixture(source=_GRAPH):
    compiler = Compiler(parse(source), module_name=_MODULE)
    request = FiniteStringEPRequest(
        graph=compiler.string_graph_specs["latent"],
        supports=(("word", ((0,), (1,))),),
        variable="word",
        words=((1,), (), (0,)),
        sweeps=2,
        damping=0.5,
        projection_tolerance=0.002,
        projection_max_iterations=2,
        projection_max_backtracks=1,
        max_entries=2,
        max_states=16,
        max_work=32,
        max_feature_cells=4,
        max_site_fits=3,
        initial_sites=((0, "word", ((0,),), (0.2,)),),
    )
    owner = torch.nn.Parameter(torch.tensor(log(0.75), dtype=torch.float64))
    parameters = {
        ParameterId.derive(_MODULE, "relation_weights", name): value
        for name, value in (
            ("unit", owner.new_zeros(())),
            ("shared", owner),
            ("loop", owner.new_tensor(-1.0)),
        )
    }
    return request, parameters, owner


def _module(request, source=_GRAPH):
    name = request.handler_name
    report = "(String * String * Int * Bool * Real * Bool * Tensor[LogWeight]([3]) * Tensor[Int]([3]))"
    program = (
        source
        + f"""
instance projection : Compute[Unit, {report}]
handler {name} for Compute[Unit, {report}] : {report} -> {report} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define belief : {report} !{{}} =
    handle projection with {name} in
        let result <- perform projection.apply(unit)
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
        module, "belief", runtime=RuntimeConfiguration(providers=(provider,))
    )


def _sigmoid(value):
    return 1 / (1 + exp(-value))


def _oracle(log_weight, initial, damping, sweeps):
    # Let z be the exact normalized target of two shared parallel arcs
    # against one unit competitor. With z in (0.57, 0.61), the initial
    # residual exceeds 0.07. The full Newton displacement a=4(z-1/2)
    # has curvature log cosh(a/2) <= a^2/8 and tangent descent a^2/4.
    # With u=a/2 the next residual is (u-tanh(u))/2 <= u^3/6 < 0.002.
    # Thus the schedule takes exactly one Newton step per diagnostic fit
    # on an open neighborhood, independently of implementation choices.
    z = _sigmoid(log_weight + log(2))
    assert 0.57 < z < 0.61
    old = damping**sweeps
    odds = old * initial + (1 - old) * 4 * (z - 0.5)
    probability = _sigmoid(odds)
    derivative = (1 - probability) * (1 - old) * 4 * z * (1 - z)
    return probability, derivative, 1 - z


def _check_report(run, request, owner):
    assert run.result_type == request.result_type
    status, reason, sweeps, have_residual, residual, available, scores, work = run.value
    assert (status, reason, sweeps.item(), have_residual.item(), available.item()) == (
        "completed",
        "",
        request.sweeps,
        True,
        True,
    )
    assert work.tolist() == [request.sweeps + 1] * 3
    assert residual.item() > 0
    initial = 0 if request.initial_sites is None else request.initial_sites[0][3][0]
    probability, derivative, exact_optimum = _oracle(
        owner.item(), initial, request.damping, request.sweeps
    )
    assert scores.exp().tolist() == pytest.approx(
        [1 - probability, 0, probability], abs=2e-15
    )
    assert scores.requires_grad, "finite EP lost its live owner derivative"
    gradient = torch.autograd.grad(scores[-1], owner)[0].item()
    assert gradient == pytest.approx(derivative, abs=2e-15)
    assert gradient != pytest.approx(exact_optimum, abs=1e-3)


@pytest.mark.parametrize("warm", (False, True))
def test_native_ep_differentiates_exactly_the_executed_damped_program(
    warm, monkeypatch
):
    request, parameters, owner = _fixture()
    if not warm:
        request = request.with_(initial_sites=None)
    wire = dumps(request)
    request = loads(wire)
    assert isinstance(request, FiniteStringEPRequest)
    assert dumps(request) == wire and request.gradient_policy == "finite_unrolled"
    module = _module(request)
    provider = FiniteStringEPRuntimeProvider(request, parameters)
    compiles = []
    init = FiniteStringBeliefPlan.__init__
    basis_constructions = []
    build_basis = NgramFamily.__post_init__

    def counted(plan, *args, **kwargs):
        compiles.append(1)
        init(plan, *args, **kwargs)

    def counted_basis(family):
        basis_constructions.append(1)
        build_basis(family)

    monkeypatch.setattr(FiniteStringBeliefPlan, "__init__", counted)
    monkeypatch.setattr(NgramFamily, "__post_init__", counted_basis)
    for mass in (0.75, 2 / 3):
        with torch.no_grad():
            owner.fill_(log(mass))
        _check_report(_run(module, provider), request, owner)
    assert compiles == [1] and basis_constructions == [1] and owner.grad is None


def test_independent_oracle_detects_detaching_a_live_factor(monkeypatch):
    request, parameters, owner = _fixture()
    module = _module(request)
    start = FiniteStringBeliefPlan._start

    def detached(plan):
        tables, messages = start(plan)
        return tuple(table.detach() for table in tables), messages

    monkeypatch.setattr(FiniteStringBeliefPlan, "_start", detached)
    provider = FiniteStringEPRuntimeProvider(request, parameters)
    with pytest.raises(AssertionError, match="lost its live owner derivative"):
        _check_report(_run(module, provider), request, owner)


def test_native_work_limits_and_failed_local_projection_have_distinct_reports():
    request, parameters, _ = _fixture()
    module = _module(request)
    for name, value in (
        ("max_entries", 1),
        ("max_states", 1),
        ("max_work", 1),
        ("max_feature_cells", 3),
        ("max_site_fits", 2),
    ):
        limited = request.with_(**{name: value})
        assert limited.handler_name == request.handler_name
        status, reason, seen, have_residual, _, available, scores, work = _run(
            module, FiniteStringEPRuntimeProvider(limited, parameters)
        ).value
        assert status == "resource_limit" and name in reason
        assert not have_residual.item() and not available.item()
        assert (
            seen.item() == 0 and scores.isneginf().all() and work.tolist() == [0, 0, 0]
        )

    failed = request.with_(projection_max_iterations=0)
    status, reason, sweeps, measured, _, available, scores, counts = _run(
        _module(failed), FiniteStringEPRuntimeProvider(failed, parameters)
    ).value
    assert status == "projection_failed" and reason == "local projection limit"
    assert sweeps.item() == 0 and counts.tolist() == [1, 0, 0]
    assert not measured.item() and not available.item() and scores.isneginf().all()

    source = _GRAPH.replace(
        "    final", "    arc 0 -> 0 [labels=(eps), weight=loop]\n    final"
    )
    cyclic, parameters, _ = _fixture(source)
    status, reason, _, _, _, available, *_ = _run(
        _module(cyclic, source), FiniteStringEPRuntimeProvider(cyclic, parameters)
    ).value
    assert status == "unsupported_closure" and "productive epsilon cycle" in reason
    assert not available.item()


def test_native_fixed_schedule_and_warm_site_match_exact_nominal_declarations():
    request, parameters, _ = _fixture()
    module = _module(request)
    with pytest.raises(dx.ValidationError, match="graph order"):
        request.with_(initial_sites=((0, "different", ((0,),), (0.2,)),))
    invalid = request.with_(initial_sites=((0, "word", ((1,),), (0.2,)),))
    with pytest.raises(ExecutionFailure, match="affine-basis order"):
        _run(_module(invalid), FiniteStringEPRuntimeProvider(invalid, parameters))
    for altered in (
        request.with_(sweeps=1),
        request.with_(words=tuple(reversed(request.words))),
        request.with_(initial_sites=None),
    ):
        assert altered.result_type == request.result_type
        assert altered.handler_name != request.handler_name
        with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
            _run(module, FiniteStringEPRuntimeProvider(altered, parameters))
    target = project_for_target(module, discharge_module(module), "stan")
    assert any(
        item.computation == "belief" and request.handler_name in item.reason
        for item in target.obstacles
    )
