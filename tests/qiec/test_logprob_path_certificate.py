"""Native checked LogProb certificates preserve evaluated snapshot identity."""

from fractions import Fraction
from hashlib import sha256

import didactic.api as dx
import pytest
import torch

from quivers.core.relations import RationalRelation, log_accuracy
from quivers.core.relations.queries import _PackedPaths
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    ExecutionFailure,
    FixedLogProbPathCertificateRequest,
    FixedLogProbPathCertificateRuntimeProvider,
    FixedRelationRequest,
    ParameterId,
    QiecModule,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target
from tests.test_logprob_path_accuracy import _log_two_interval

_MODULE = "checked_logprob_path_certificate"
_SOURCE = """object Letters : FinSet 2

relation original [algebra=logprob, states=2]
    output Letters
    initial 0 [weight=initial]
    arc 0 -> 1 [labels=(0), weight=parallel]
    arc 0 -> 1 [labels=(0), weight=parallel]
    final 1 [weight=final]
"""


def _observation(source=_SOURCE):
    compiler = Compiler(parse(source), module_name=_MODULE)
    return FixedRelationRequest(
        relation=compiler.relation_specs["original"],
        operation="pair_weight",
        words=((0,),),
        max_states=16,
        max_work=32,
    )


def _fixture(dtype=torch.float64):
    request = FixedLogProbPathCertificateRequest(
        observation=_observation(),
        absolute_log_tolerance=(1, 10**12),
        decimal_precision=42,
        max_decimal_exponent=2000,
        max_decimal_work=64,
        max_exact_bits=1024,
        max_exact_work=16,
    )
    spec = request.observation.relation
    values = {"initial": -3, "parallel": -1000, "final": -5}
    parameters = {
        owner: torch.nn.Parameter(torch.tensor(values[name], dtype=dtype))
        for name, owner in zip(
            spec.topology.weight_names, spec.weight_parameters, strict=True
        )
    }
    return (
        request,
        parameters,
        spec.weight_parameters[spec.topology.weight_names.index("parallel")],
    )


def _module(request, source=_SOURCE):
    report = "(String * String * Bool * LogWeight * Tensor[String]([3, 2]) * Bool * Tensor[Int]([3]))"
    name = request.handler_name
    program = (
        source
        + f"""
instance verification : Compute[Unit, {report}]
handler {name} for Compute[Unit, {report}] : {report} -> {report} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define certificate : {report} !{{}} =
    handle verification with {name} in
        let result <- perform verification.apply(unit)
        return result
"""
    )
    module = Compiler(
        parse(module_to_source(parse(program))), module_name=_MODULE
    ).qiec_module
    result = loads(dumps(module))
    assert isinstance(result, QiecModule)
    return result


def _run(module, provider):
    return run_named(
        module, "certificate", runtime=RuntimeConfiguration(providers=(provider,))
    )


def _decode(value):
    numerator, denominator = value
    return Fraction(int(numerator, 16), int(denominator, 16))


def _assert_positive(run, request, log_arc):
    assert run.result_type == request.result_type
    status, reason, available, weight, witness, endpoints, work = run.value
    assert (status, reason, available.item(), endpoints.item()) == (
        "certified",
        "",
        True,
        True,
    )
    assert not weight.requires_grad
    lower, upper, error = map(_decode, witness)
    ln_two_lower, ln_two_upper = _log_two_interval()
    assert lower <= log_arc - 8 + ln_two_lower <= log_arc - 8 + ln_two_upper <= upper
    assert (
        Fraction(weight.item()) - error
        <= lower
        <= upper
        <= Fraction(weight.item()) + error
    )
    assert upper - lower < Fraction(1, 10**34)
    assert error <= Fraction(*request.absolute_log_tolerance)
    assert work[0].item() == 22 and work[1].item() == 8 and work[2].item() > 63


def _assert_unavailable(run, status, reason):
    code, message, available, weight, witness, endpoints, work = run.value
    assert code == status and reason in message
    assert not available.item() and not endpoints.item()
    assert weight.item() == 0 and not weight.requires_grad
    assert witness == (("", ""),) * 3
    assert work.tolist() == [0, 0, 0]


def test_native_score_and_rational_bounds_use_one_snapshot(monkeypatch):
    request, parameters, shared = _fixture()
    wire = dumps(request)
    request = loads(wire)
    assert isinstance(request, FixedLogProbPathCertificateRequest)
    assert dumps(request) == wire
    module = _module(request)
    provider = FixedLogProbPathCertificateRuntimeProvider(request, parameters)
    evaluations = []
    compilations = []
    read_weights = RationalRelation._weights
    compile_paths = log_accuracy._compile_paths
    mass_bounds = log_accuracy._mass_bounds

    def counted_reads(relation):
        evaluations.append(1)
        return read_weights(relation)

    def counted_compiles(*args, **kwargs):
        compilations.append(1)
        return compile_paths(*args, **kwargs)

    def change_live_owner_after_snapshot(*args):
        with torch.no_grad():
            parameters[shared].fill_(-1200)
        return mass_bounds(*args)

    monkeypatch.setattr(RationalRelation, "_weights", counted_reads)
    monkeypatch.setattr(log_accuracy, "_compile_paths", counted_compiles)
    with monkeypatch.context() as patch:
        patch.setattr(log_accuracy, "_mass_bounds", change_live_owner_after_snapshot)
        _assert_positive(_run(module, provider), request, -1000)
    _assert_positive(_run(module, provider), request, -1200)
    assert evaluations == [1, 1] and compilations == [1]
    assert parameters[shared].requires_grad and parameters[shared].grad is None


def test_zero_and_typed_failures_do_not_fabricate_evidence(monkeypatch):
    request, parameters, shared = _fixture()
    module = _module(request)
    cases = (
        (
            request.with_(max_decimal_exponent=499),
            parameters,
            "resource_limit",
            "max_decimal_exponent",
        ),
        (
            request.with_(max_decimal_work=21),
            parameters,
            "resource_limit",
            "max_decimal_work",
        ),
        (
            request.with_(max_exact_bits=128),
            parameters,
            "resource_limit",
            "max_exact_bits",
        ),
        (
            request.with_(max_exact_work=7),
            parameters,
            "resource_limit",
            "max_exact_work",
        ),
        (
            request.with_(decimal_precision=5),
            parameters,
            "unverified_accuracy",
            "absolute log path error",
        ),
        (
            request,
            _fixture(torch.float32)[1],
            "unsupported_precision",
            "requires float64",
        ),
    )
    for changed, bindings, status, reason in cases:
        assert changed.handler_name == request.handler_name
        run = _run(
            module, FixedLogProbPathCertificateRuntimeProvider(changed, bindings)
        )
        assert run.result_type == request.result_type
        _assert_unavailable(run, status, reason)

    source = _SOURCE.replace(
        "    final", "    arc 0 -> 0 [labels=(eps), weight=parallel]\n    final"
    )
    cyclic = request.with_(observation=_observation(source))
    provider = FixedLogProbPathCertificateRuntimeProvider(cyclic, parameters)
    _assert_unavailable(
        _run(_module(cyclic, source), provider), "unsupported_closure", "epsilon cycle"
    )

    provider = FixedLogProbPathCertificateRuntimeProvider(request, parameters)
    evaluate = _PackedPaths.evaluate

    def omit_parallel_mass(plan, *args, **kwargs):
        return evaluate(plan, *args, **kwargs) - 0.6931471805599453

    with monkeypatch.context() as patch:
        patch.setattr(_PackedPaths, "evaluate", omit_parallel_mass)
        _assert_unavailable(
            _run(module, provider), "unverified_accuracy", "absolute log path error"
        )

    with torch.no_grad():
        parameters[shared].fill_(-torch.inf)
    status, diagnostic, available, log_weight, witness, has_endpoints, work = _run(
        module, provider
    ).value
    assert (
        status,
        diagnostic,
        available.item(),
        log_weight.item(),
        has_endpoints.item(),
    ) == ("certified", "", True, -torch.inf, False)
    assert witness == (("", ""), ("", ""), ("0x0", "0x1"))
    assert work.tolist() == [0, 1, 40]


def test_checked_objective_owner_and_native_target_refusal():
    request, parameters, shared = _fixture()
    module = _module(request)
    with pytest.raises(dx.ValidationError, match="acyclic checked LogProb pair"):
        request.with_(observation=request.observation.with_(max_closure_size=4))
    with pytest.raises(dx.ValidationError, match="acyclic checked LogProb pair"):
        request.with_(observation=_observation(_SOURCE.replace("logprob", "markov")))
    with pytest.raises(dx.ValidationError, match="nonnegative rational pair"):
        request.with_(absolute_log_tolerance=(1, 0))
    assert (
        request.with_(absolute_log_tolerance=(2, 2 * 10**12)).handler_name
        == request.handler_name
    )
    alias = ParameterId.derive(_MODULE, "relation_weights", "same_shape_other_owner")
    relation = request.observation.relation
    spec = relation.with_(
        weight_parameters=tuple(
            alias if owner == shared else owner for owner in relation.weight_parameters
        )
    )
    other_owner = request.with_(observation=request.observation.with_(relation=spec))
    other_bindings = {
        alias if owner == shared else owner: weight
        for owner, weight in parameters.items()
    }
    for other, bindings in (
        (other_owner, other_bindings),
        (request.with_(absolute_log_tolerance=(1, 10**11)), parameters),
        (
            request.with_(observation=request.observation.with_(words=((1,),))),
            parameters,
        ),
    ):
        assert (
            other.result_type == request.result_type
            and other.handler_name != request.handler_name
        )
        with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
            _run(module, FixedLogProbPathCertificateRuntimeProvider(other, bindings))
    target = project_for_target(module, discharge_module(module), "stan")
    assert any(
        item.computation == "certificate" and request.handler_name in item.reason
        for item in target.obstacles
    )


def test_existing_fixed_pair_wire_remains_byte_stable():
    encoded = dumps(_observation())
    assert (
        sha256(encoded.encode()).hexdigest()
        == "6b7ed2c3ff856fe8b3b0f92ef0be5ee7eeee9c7b76c2fb2931f6f3b63664ed0e"
    )
    assert dumps(loads(encoded)) == encoded
