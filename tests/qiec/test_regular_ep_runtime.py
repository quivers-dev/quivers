"""An authored regular string EP query runs through the ordinary QIEC provider."""

from hashlib import sha256
from math import log

import didactic.api as dx
import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ExecutionFailure,
    ParameterId,
    QiecModule,
    RegularStringEPConfig,
    RegularStringEPPolicy,
    RegularStringEPRequest,
    RegularStringEPRuntimeProvider,
    RuntimeConfiguration,
    StringFactorGraphSpec,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target

_MODULE = "regular_ep_query"
_GRAPH = """object Segments : FinSet 2
object Wrong : FinSet 2

relation lexicon [algebra=logprob, states=2]
    output Segments
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0), weight=zero_word]
    arc 0 -> 1 [labels=(1), weight=one_word]
    final 1 [weight=one]

relation reference [algebra=logprob, states=2]
    output Segments
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0), weight=reference_mass]
    arc 0 -> 1 [labels=(1), weight=reference_mass]
    final 1 [weight=one]

relation wrong [algebra=logprob, states=2]
    output Wrong
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0), weight=one]
    arc 0 -> 1 [labels=(1), weight=one]
    final 1 [weight=one]

string_graph latent
    variable word : Segments
    factor lexicon(word)
"""


def _fixture():
    compiler = Compiler(parse(_GRAPH), module_name=_MODULE)
    config = RegularStringEPConfig(
        bases=(("word", compiler.relation_specs["reference"]),),
        features=(("word", ((0,), (1,))),),
        penalty=0.0,
        max_sweeps=2,
        max_states=128,
        max_work=256,
        tolerance=1e-9,
        projection_tolerance=1e-12,
    )
    request = RegularStringEPRequest(
        graph=compiler.string_graph_specs["latent"],
        config=config,
        variable="word",
        words=((1,), (), (0,)),
    )
    result_type = (
        "(String * String * Int * Bool * Real * Bool * Tensor[LogWeight]([3]) * Int)"
    )
    name = request.handler_name
    source = (
        _GRAPH
        + f"""
instance projection : Compute[Unit, {result_type}]
handler {name} for Compute[Unit, {result_type}] : {result_type} -> {result_type} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define belief : {result_type} !{{}} =
    handle projection with {name} in
        let report <- perform projection.apply(unit)
        return report
"""
    )
    module = Compiler(parse(source), module_name=_MODULE).qiec_module
    wire_module = loads(dumps(module))
    assert isinstance(wire_module, QiecModule)
    owner = torch.nn.Parameter(torch.tensor(log(3), dtype=torch.float64))
    parameters = {
        ParameterId.derive(_MODULE, "relation_weights", name): value
        for name, value in (
            ("zero_word", owner),
            ("one_word", owner.new_zeros(())),
            ("one", owner.new_zeros(())),
            ("reference_mass", owner.new_zeros(())),
        )
    }
    return compiler, wire_module, request, parameters, owner


def _run(module, provider):
    return run_named(
        module, "belief", runtime=RuntimeConfiguration(providers=(provider,))
    )


def test_checked_regular_belief_matches_current_normalized_weights_and_nominal_word_order():
    _, module, request, parameters, owner = _fixture()
    wire = dumps(request)
    restored = loads(wire)
    assert isinstance(restored, RegularStringEPRequest) and dumps(restored) == wire
    assert restored == request
    provider = RegularStringEPRuntimeProvider(restored, parameters)
    for mass in (3.0, 0.25):
        with torch.no_grad():
            owner.fill_(log(mass))
        run = _run(module, provider)
        assert run.result_type == request.result_type
        status, reason, sweeps, measured, residual, available, scores, attempted = (
            run.value
        )
        assert (
            status,
            reason,
            sweeps.item(),
            measured.item(),
            available.item(),
            attempted.item(),
        ) == (
            "converged",
            "",
            1,
            True,
            True,
            1,
        )
        assert residual.item() <= request.config.tolerance
        assert scores[1].item() == -torch.inf
        assert scores.exp().tolist() == pytest.approx(
            [1 / (1 + mass), 0.0, mass / (1 + mass)], abs=1e-10
        )
        assert not scores.requires_grad


def test_checked_regular_belief_reports_distinct_outer_inner_and_resource_limits():
    _, module, request, parameters, _ = _fixture()
    for options, status, available in (
        ({"max_sweeps": 1, "tolerance": None}, "limit", True),
        ({"projection_max_iterations": 1}, "projection_failed", True),
        ({"max_states": 1}, "resource_limit", False),
    ):
        changed = request.with_(config=request.config.with_(**options))
        assert changed.handler_name == request.handler_name
        provider = RegularStringEPRuntimeProvider(changed, parameters)
        actual, reason, sweeps, measured, _, usable, scores, attempted = _run(
            module, provider
        ).value
        assert actual == status and usable.item() is available
        assert measured.item() is (status == "limit")
        if status == "resource_limit":
            assert "max_states" in reason
            assert attempted.item() == sweeps.item() == 0
            assert scores.isneginf().all()
        elif status == "projection_failed":
            assert reason == "limit" and sweeps.item() == 0 and attempted.item() == 1
            assert scores.exp().tolist() == [0.5, 0.0, 0.5]


@pytest.mark.parametrize(
    ("slots", "distribution"),
    [
        (("zero_word", "one_word"), "tilted distribution"),
        (("reference_mass",), "regular belief or cavity"),
    ],
)
def test_log_zero_normalizer_reports_unavailable_and_recovers(slots, distribution):
    _, module, request, parameters, _ = _fixture()
    provider = RegularStringEPRuntimeProvider(request, parameters)
    with torch.no_grad():
        for name in slots:
            parameters[ParameterId.derive(_MODULE, "relation_weights", name)].fill_(
                -torch.inf
            )
    report = _run(module, provider)
    assert report.result_type == request.result_type
    status, reason, sweeps, measured, _, available, scores, attempted = report.value
    assert status == "zero_mass" and distribution in reason
    assert sweeps.item() == attempted.item() == 0
    assert not measured.item() and not available.item() and scores.isneginf().all()

    with torch.no_grad():
        for name in slots:
            parameters[ParameterId.derive(_MODULE, "relation_weights", name)].zero_()
    assert _run(module, provider).value[0] == "converged"


def test_invalid_weights_are_not_reported_as_zero_normalizers():
    _, module, request, parameters, owner = _fixture()
    with torch.no_grad():
        owner.fill_(torch.inf)
    with pytest.raises(ExecutionFailure, match="carrier"):
        _run(module, RegularStringEPRuntimeProvider(request, parameters))


def test_portable_policy_keeps_old_bytes_and_initializes_native_ep_from_warm_sites():
    _, module, request, parameters, owner = _fixture()
    assert sha256(dumps(request).encode()).hexdigest() == (
        "a4953d7519d41ddf15bd64027e1c39ac25c23a3963a9545372e00dbbc80c4812"
    )
    assert sha256(dumps(request.config).encode()).hexdigest() == (
        "433b83e73f43aadc2a4ccec7baac37eb83d290191f4bc72d8547275600173164"
    )
    limited = request.with_(config=request.config.with_(max_sweeps=0, tolerance=None))
    policy = RegularStringEPPolicy(
        config=limited.config,
        projection_optimizer="fixed_lipschitz",
        initial_sites=((0, "word", ((0,), (1,)), (log(4), 0.0), 7.0),),
    )
    wire = dumps(policy)
    actual = loads(wire)
    assert isinstance(actual, RegularStringEPPolicy) and dumps(actual) == wire
    assert actual.objective == "regular_string_ep"
    assert actual.gradient_policy == "detached" and actual.error_guarantee == "none"
    result = actual.infer(request.graph, parameters)
    assert result.status == "limit" and result.sweeps == 0
    assert result.sites[0].log_scale.item() == 7.0
    assert result.sites[0].features == ((0,),)
    assert result.updates[0].fit.optimizer == "fixed_lipschitz"
    assert result.moment_evaluations == 2 + result.updates[0].fit.iterations
    assert not result.marginals["word"].log_prob((0,)).requires_grad

    limited = loads(dumps(limited))
    assert isinstance(limited, RegularStringEPRequest)
    provider = RegularStringEPRuntimeProvider(limited, parameters, policy=actual)
    report = _run(module, provider).value
    assert report[0] == "limit" and report[2].item() == 0 and report[5].item()
    assert report[6].exp().tolist() == pytest.approx([0.2, 0.0, 0.8])
    with pytest.raises(ValueError, match="exact request config"):
        RegularStringEPRuntimeProvider(request, parameters, policy=actual)

    repeated = policy.with_(config=request.config)
    result = repeated.infer(request.graph, parameters)
    assert result.status == "converged" and result.sweeps == 1
    assert result.moment_evaluations > result.updates[0].fit.moment_evaluations
    assert result.marginals["word"].log_prob((0,)).exp().item() == pytest.approx(0.75)
    assert owner.requires_grad and owner.grad is None


def test_regular_ep_policy_refuses_unsupported_guarantees_and_bad_initials(monkeypatch):
    compiler, _, request, parameters, _ = _fixture()
    site = (0, "word", ((0,), (1,)), (0.0, 0.0), 0.0)
    policy = RegularStringEPPolicy(config=request.config, initial_sites=(site,))
    for gradient in ("finite_unrolled", "converged"):
        with pytest.raises(dx.ValidationError, match="detached gradient policy"):
            policy.with_(gradient_policy=gradient)
    with pytest.raises(dx.ValidationError, match="global posterior error"):
        policy.with_(error_guarantee="certified_global")
    for bad in (
        (*site[:3], (0.0, float("inf")), 0.0),
        (*site[:3], (False, 0.0), 0.0),
        (*site[:3], (0.0, 0.0), float("nan")),
    ):
        with pytest.raises(dx.ValidationError, match="finite floats"):
            policy.with_(initial_sites=(bad,))

    def unexpected_bind(*_args, **_kwargs):
        raise AssertionError("invalid initial sites must refuse before graph binding")

    monkeypatch.setattr(StringFactorGraphSpec, "bind", unexpected_bind)
    wrong = policy.with_(
        config=policy.config.with_(bases=(("word", compiler.relation_specs["wrong"]),))
    )
    with pytest.raises(ValueError, match="matching nominal tapes"):
        wrong.infer(request.graph, parameters)
    with pytest.raises(ValueError, match="match graph and feature order"):
        policy.with_(initial_sites=((0, "word", ((1,), (0,)), (0.0, 0.0), 0.0),)).infer(
            request.graph, parameters
        )
    duplicate = request.graph.with_(
        factors=(*request.graph.factors, *request.graph.factors)
    )
    with pytest.raises(ValueError, match="match graph and feature order"):
        policy.with_(initial_sites=((1, *site[1:]), site)).infer(duplicate, parameters)


def test_regular_ep_provider_caches_static_identity_only(monkeypatch):
    _, module, request, parameters, owner = _fixture()
    provider = RegularStringEPRuntimeProvider(request, parameters)

    def unexpected_lookup(_request):
        raise AssertionError("the attached provider must reuse its immutable identity")

    for attribute in ("handler_name", "result_type"):
        monkeypatch.setattr(
            RegularStringEPRequest, attribute, property(unexpected_lookup)
        )
    for weight, expected in ((log(3), 0.75), (log(0.25), 0.2)):
        with torch.no_grad():
            owner.fill_(weight)
        run = _run(module, provider)
        assert run.value[0] == "converged"
        assert run.value[6][-1].exp().item() == pytest.approx(expected)


def test_regular_belief_refuses_same_shape_substitution_and_reports_foreign_target_boundary():
    compiler, module, request, parameters, _ = _fixture()
    with pytest.raises(dx.ValidationError, match="nominal tapes"):
        request.with_(
            config=request.config.with_(
                bases=(("word", compiler.relation_specs["wrong"]),)
            )
        )
    with pytest.raises(dx.ValidationError, match="distinct words"):
        request.with_(words=((1,), (1,)))
    with pytest.raises(dx.ValidationError, match="query alphabet"):
        request.with_(words=((2,),))
    changed = request.with_(words=tuple(reversed(request.words)))
    assert (
        changed.result_type == request.result_type
        and changed.handler_name != request.handler_name
    )
    with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
        _run(module, RegularStringEPRuntimeProvider(changed, parameters))
    report = project_for_target(module, discharge_module(module), "stan")
    assert any(
        item.computation == "belief" and request.handler_name in item.reason
        for item in report.obstacles
    )
