"""Native value-only Markov certificates preserve exact rational evidence."""

from fractions import Fraction
from hashlib import sha256

import didactic.api as dx
import pytest
import torch

from quivers.core.relations import accuracy
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    ExecutionFailure,
    FixedMarkovPathCertificateRequest,
    FixedMarkovPathCertificateRuntimeProvider,
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
from tests.test_markov_path_accuracy import _independent_solution

_MODULE = "checked_markov_path_certificate"
_SOURCE = """object Letters : FinSet 2

relation original [algebra=markov, states=3]
    output Letters
    initial 0 [weight=unit]
    arc 0 -> 0 [labels=(eps), weight=stay]
    arc 0 -> 1 [labels=(eps), weight=bridge]
    arc 0 -> 1 [labels=(eps), weight=bridge]
    arc 1 -> 0 [labels=(eps), weight=back]
    arc 1 -> 1 [labels=(eps), weight=stay]
    arc 1 -> 2 [labels=(0), weight=finish]
    final 2 [weight=unit]
"""


def _observation(source=_SOURCE):
    compiler = Compiler(parse(source), module_name=_MODULE)
    return FixedRelationRequest(
        relation=compiler.relation_specs["original"],
        operation="pair_weight",
        words=((0,),),
        max_states=32,
        max_work=64,
        max_closure_size=6,
    )


def _fixture(dtype=torch.float64):
    observation = _observation()
    request = FixedMarkovPathCertificateRequest(
        observation=observation,
        absolute_tolerance=(1, 10**12),
        relative_tolerance=(1, 10**12),
        max_exact_bits=4096,
        max_exact_work=4096,
        max_closure_cells=9,
    )
    spec = observation.relation
    names = spec.topology.weight_names
    values = dict(unit=1, stay=0.125, bridge=0.75, back=0.125, finish=0.0625)
    parameters = {
        owner: torch.nn.Parameter(torch.tensor(values[name], dtype=dtype))
        for name, owner in zip(names, spec.weight_parameters, strict=True)
    }
    return request, parameters, spec.weight_parameters[names.index("bridge")]


def _module(request):
    report = (
        "(String * String * Bool * Real * Tensor[String]([4, 2]) "
        "* Bool * Tensor[Int]([3]))"
    )
    name = request.handler_name
    source = (
        _SOURCE
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
        parse(module_to_source(parse(source))), module_name=_MODULE
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


def test_existing_fixed_markov_pair_bytes_are_unchanged():
    encoded = dumps(_observation())
    assert sha256(encoded.encode()).hexdigest() == (
        "5a2a10c60786a337ce54f9f05500bd4be52af4e6edea80ea4f8ea8f28d3a6ffc"
    )
    assert dumps(loads(encoded)) == encoded


def test_native_certificate_keeps_exact_witness_and_uses_live_snapshot(monkeypatch):
    request, parameters, bridge = _fixture()
    wire = dumps(request)
    request = loads(wire)
    assert (
        isinstance(request, FixedMarkovPathCertificateRequest)
        and dumps(request) == wire
    )
    module = _module(request)
    provider = FixedMarkovPathCertificateRuntimeProvider(request, parameters)
    compile_calls = []
    compile_paths = accuracy._compile_paths

    def counted(*args, **kwargs):
        compile_calls.append(1)
        return compile_paths(*args, **kwargs)

    monkeypatch.setattr(accuracy, "_compile_paths", counted)
    for p in (Fraction(3, 4), Fraction(5, 8)):
        with torch.no_grad():
            parameters[bridge].fill_(float(p))
        exact, _ = _independent_solution(p)
        run = _run(module, provider)
        assert run.result_type == request.result_type
        status, reason, available, weight, witness, relative, work = run.value
        assert (status, reason, available.item(), relative.item()) == (
            "certified",
            "",
            True,
            True,
        )
        assert not weight.requires_grad
        lower, upper, error, relative_error = map(_decode, witness)
        actual = Fraction(weight.item())
        assert lower <= exact <= upper
        assert abs(actual - exact) <= error <= Fraction(1, 10**12)
        assert abs(actual - exact) / exact <= relative_error <= Fraction(1, 10**12)
        assert (
            max(
                item.denominator.bit_length()
                for item in (lower, upper, error, relative_error)
            )
            > 63
        )
        assert work[0].item() > 0 and work[1].item() > 63 and work[2].item() == 9
    assert compile_calls == [1]


def _check_unavailable(run, status, reason):
    code, message, available, score, witness, relative, work = run.value
    assert code == status and reason in message
    assert not available.item() and not relative.item()
    assert score.item() == 0 and not score.requires_grad
    assert witness == (("", ""),) * 4
    assert work.tolist() == [0, 0, 0]


def test_typed_resource_precision_accuracy_and_domain_refusals():
    request, parameters, bridge = _fixture()
    module = _module(request)
    for changed, current, status, reason in (
        (
            request.with_(max_closure_cells=8),
            parameters,
            "resource_limit",
            "max_closure_cells",
        ),
        (
            request.with_(max_exact_work=2),
            parameters,
            "resource_limit",
            "max_exact_work",
        ),
        (
            request.with_(observation=request.observation.with_(max_closure_size=None)),
            parameters,
            "unsupported_closure",
            "productive epsilon cycle",
        ),
        (request, _fixture(torch.float32)[1], "unsupported_precision", "float64"),
    ):
        assert changed.handler_name == request.handler_name
        run = _run(module, FixedMarkovPathCertificateRuntimeProvider(changed, current))
        assert run.result_type == request.result_type
        _check_unavailable(run, status, reason)

    zero_tolerance = request.with_(absolute_tolerance=(0, 1), relative_tolerance=None)
    run = _run(
        _module(zero_tolerance),
        FixedMarkovPathCertificateRuntimeProvider(zero_tolerance, parameters),
    )
    _check_unavailable(run, "unverified_accuracy", "requested absolute path error")

    provider = FixedMarkovPathCertificateRuntimeProvider(request, parameters)
    with torch.no_grad():
        parameters[bridge].fill_(4)
    _check_unavailable(_run(module, provider), "divergent", "cycle")
    with torch.no_grad():
        parameters[bridge].fill_(0.75)
    assert _run(module, provider).value[0] == "certified"


def test_absolute_only_and_exact_zero_have_distinct_relative_evidence():
    request, parameters, _ = _fixture()
    absolute = request.with_(relative_tolerance=None)
    result = _run(
        _module(absolute),
        FixedMarkovPathCertificateRuntimeProvider(absolute, parameters),
    )
    assert result.value[0] == "certified" and not result.value[5].item()
    assert result.value[4][-1] == ("", "")

    zero = request.with_(observation=request.observation.with_(words=((1,),)))
    result = _run(
        _module(zero), FixedMarkovPathCertificateRuntimeProvider(zero, parameters)
    )
    assert result.value[0] == "certified" and result.value[5].item()
    assert tuple(map(_decode, result.value[4])) == (Fraction(0),) * 4


def test_nominal_owner_tolerances_and_targets_are_checked():
    request, parameters, bridge = _fixture()
    with pytest.raises(dx.ValidationError, match="Markov pair weight"):
        request.with_(
            observation=_observation(
                _SOURCE.replace("algebra=markov", "algebra=logprob")
            )
        )
    with pytest.raises(dx.ValidationError, match="absolute or relative tolerance"):
        request.with_(absolute_tolerance=None, relative_tolerance=None)
    with pytest.raises(dx.ValidationError, match="nonnegative rational pairs"):
        request.with_(absolute_tolerance=(1, 0))
    assert (
        request.with_(absolute_tolerance=(2, 2 * 10**12)).handler_name
        == request.handler_name
    )

    module = _module(request)
    tolerance = request.with_(absolute_tolerance=(1, 10**11))
    alias = ParameterId.derive(_MODULE, "relation_weights", "same_shape_other_owner")
    spec = request.observation.relation
    other = spec.with_(
        weight_parameters=tuple(
            alias if owner == bridge else owner for owner in spec.weight_parameters
        )
    )
    ownership = request.with_(observation=request.observation.with_(relation=other))
    other_weights = {
        alias if owner == bridge else owner: value
        for owner, value in parameters.items()
    }
    for substitute, bindings in ((tolerance, parameters), (ownership, other_weights)):
        assert (
            substitute.result_type == request.result_type
            and substitute.handler_name != request.handler_name
        )
        with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
            _run(
                module, FixedMarkovPathCertificateRuntimeProvider(substitute, bindings)
            )
    target = project_for_target(module, discharge_module(module), "stan")
    assert any(
        item.computation == "certificate" and request.handler_name in item.reason
        for item in target.obstacles
    )
