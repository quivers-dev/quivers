"""Check exact cyclic QVR reports, runtime ownership, and closure diagnostics."""

from fractions import Fraction
from hashlib import sha256

import didactic.api as dx
import pytest
import torch

from quivers.core.relations import RationalRelation, log_accuracy
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ExecutionFailure,
    FixedCyclicLogProbPathCertificateRequest,
    FixedCyclicLogProbPathCertificateRuntimeProvider,
    FixedLogProbPathCertificateRequest,
    FixedLogProbPathCertificateRuntimeProvider,
    FixedRelationRequest,
    ParameterId,
    discharge_module,
    dumps,
    loads,
)
from quivers.transpile import project_for_target
from tests.qiec.test_logprob_path_certificate import (
    _MODULE,
    _assert_unavailable,
    _decode,
    _fixture as _acyclic_fixture,
    _module,
    _run,
)
from tests.test_cyclic_logprob_path_accuracy import _near_critical_reference

_SOURCE = """object Letters : FinSet 2

relation original [algebra=logprob, states=2]
    output Letters
    initial 0 [weight=unit]
    arc 0 -> 0 [labels=(eps), weight=repeat]
    arc 0 -> 1 [labels=(0), weight=output]
    final 1 [weight=unit]
"""


def _fixture():
    compiler = Compiler(parse(_SOURCE), module_name=_MODULE)
    spec = compiler.relation_specs["original"]
    request = FixedCyclicLogProbPathCertificateRequest(
        observation=FixedRelationRequest(
            relation=spec,
            operation="pair_weight",
            words=((0,),),
            max_states=16,
            max_work=32,
            max_closure_size=1,
        ),
        absolute_log_tolerance=(1, 10**12),
        max_closure_cells=1,
        decimal_precision=52,
        max_decimal_exponent=2048,
        max_decimal_work=128,
        max_exact_bits=4096,
        max_exact_work=128,
    )
    slots = dict(zip(spec.topology.weight_names, spec.weight_parameters, strict=True))
    values = {"unit": 0.0, "repeat": -(2.0**-40), "output": -1000.0}
    parameters = {
        owner: torch.nn.Parameter(torch.tensor(values[name], dtype=torch.float64))
        for name, owner in slots.items()
    }
    return request, parameters, slots


def _log_report(run, request, exit_weight):
    assert run.result_type == request.result_type
    status, message, available, score, witness, endpoints, work = run.value
    assert (status, message, available.item(), endpoints.item()) == (
        "certified",
        "",
        True,
        True,
    )
    assert not score.requires_grad
    lower, upper, error = map(_decode, witness)
    expected_low, expected_high = (
        bound + exit_weight + 1000 for bound in _near_critical_reference()
    )
    assert lower <= expected_low <= expected_high <= upper
    returned = Fraction(score.item())
    assert (
        max(abs(returned - expected_low), abs(returned - expected_high))
        <= error
        == max(abs(returned - lower), abs(returned - upper))
    )
    assert error <= Fraction(*request.absolute_log_tolerance)
    assert work[0].item() > 0 and work[1].item() > 0 and work[2].item() > 1500


def test_live_native_cycle_uses_the_same_binary64_snapshot(monkeypatch):
    request, parameters, slots = _fixture()
    wire = dumps(request)
    request = loads(wire)
    assert isinstance(request, FixedCyclicLogProbPathCertificateRequest)
    assert request.observation.max_closure_size == 1 and dumps(request) == wire
    module = _module(request, _SOURCE)
    provider = FixedCyclicLogProbPathCertificateRuntimeProvider(request, parameters)
    reads, plans = [], []
    original_weights = RationalRelation._weights
    compile_paths = log_accuracy._compile_paths
    cyclic_bounds = log_accuracy._cyclic_mass_bounds

    def counted_weights(relation):
        reads.append(1)
        return original_weights(relation)

    def counted_plan(*args, **kwargs):
        plans.append(1)
        return compile_paths(*args, **kwargs)

    def change_live_slot(*args):
        with torch.no_grad():
            parameters[slots["output"]].fill_(-1200)
        return cyclic_bounds(*args)

    monkeypatch.setattr(RationalRelation, "_weights", counted_weights)
    monkeypatch.setattr(log_accuracy, "_compile_paths", counted_plan)
    with monkeypatch.context() as patch:
        patch.setattr(log_accuracy, "_cyclic_mass_bounds", change_live_slot)
        _log_report(_run(module, provider), request, -1000)
    _log_report(_run(module, provider), request, -1200)
    assert reads == [1, 1] and plans == [1]
    assert parameters[slots["output"]].requires_grad
    assert parameters[slots["output"]].grad is None


def test_native_lower_divergence_and_unverified_upper_star_are_distinct():
    request, parameters, slots = _fixture()
    module = _module(request, _SOURCE)
    cases = (
        (request.with_(max_closure_cells=0), "resource_limit", "max_closure_cells"),
        (
            request.with_(observation=request.observation.with_(max_closure_size=0)),
            "resource_limit",
            "max_closure_size",
        ),
        (request.with_(max_exact_work=2), "resource_limit", "max_exact_work"),
        (
            request.with_(decimal_precision=8),
            "unverified_accuracy",
            "finite upper star",
        ),
    )
    for changed, status, expected in cases:
        assert changed.handler_name == request.handler_name
        result = _run(
            module,
            FixedCyclicLogProbPathCertificateRuntimeProvider(changed, parameters),
        )
        assert result.result_type == request.result_type
        _assert_unavailable(result, status, expected)

    provider = FixedCyclicLogProbPathCertificateRuntimeProvider(request, parameters)
    with torch.no_grad():
        parameters[slots["repeat"]].fill_(0)
    for exit_weight in (-1000, -torch.inf):
        with torch.no_grad():
            parameters[slots["output"]].fill_(exit_weight)
        _assert_unavailable(
            _run(module, provider),
            "divergent_closure",
            "lower return mass is at least one",
        )
    with torch.no_grad():
        parameters[slots["repeat"]].fill_(-(2.0**-40))
        parameters[slots["output"]].fill_(-1000)
    _log_report(_run(module, provider), request, -1000)

    zero_tolerance = request.with_(absolute_log_tolerance=(0, 1))
    _assert_unavailable(
        _run(
            _module(zero_tolerance, _SOURCE),
            FixedCyclicLogProbPathCertificateRuntimeProvider(
                zero_tolerance, parameters
            ),
        ),
        "unverified_accuracy",
        "absolute log path error",
    )
    with torch.no_grad():
        parameters[slots["output"]].fill_(-torch.inf)
    code, message, available, score, witness, endpoints, _ = _run(
        module, provider
    ).value
    assert (code, message, available.item(), score.item(), endpoints.item()) == (
        "certified",
        "",
        True,
        -torch.inf,
        False,
    )
    assert witness == (("", ""), ("", ""), ("0x0", "0x1"))


def test_native_closure_identity_ownership_and_target_refusal():
    request, parameters, slots = _fixture()
    module = _module(request, _SOURCE)
    with pytest.raises(dx.ValidationError, match="closure order limit"):
        request.with_(observation=request.observation.with_(max_closure_size=None))
    with pytest.raises(dx.ValidationError, match="max_closure_cells"):
        request.with_(max_closure_cells=-1)
    acyclic = FixedLogProbPathCertificateRequest(
        observation=request.observation.with_(max_closure_size=None),
        absolute_log_tolerance=request.absolute_log_tolerance,
        decimal_precision=request.decimal_precision,
        max_decimal_exponent=request.max_decimal_exponent,
        max_decimal_work=request.max_decimal_work,
        max_exact_bits=request.max_exact_bits,
        max_exact_work=request.max_exact_work,
    )
    # Keep the mathematical input identity and tolerance fixed so only the
    # closure mode distinguishes the two names.
    assert acyclic.result_type == request.result_type
    assert (
        acyclic.handler_name.rsplit("_", 1)[1] == request.handler_name.rsplit("_", 1)[1]
    )
    assert acyclic.handler_name != request.handler_name
    with pytest.raises(TypeError, match="acyclic provider"):
        FixedLogProbPathCertificateRuntimeProvider(request, parameters)
    with pytest.raises(TypeError, match="bounded cyclic request"):
        FixedCyclicLogProbPathCertificateRuntimeProvider(acyclic, parameters)
    with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
        _run(module, FixedLogProbPathCertificateRuntimeProvider(acyclic, parameters))

    old = slots["repeat"]
    alias = ParameterId.derive(_MODULE, "relation_weights", "different_owner")
    relation = request.observation.relation
    substitute = relation.with_(
        weight_parameters=tuple(
            alias if owner == old else owner for owner in relation.weight_parameters
        )
    )
    changed = request.with_(observation=request.observation.with_(relation=substitute))
    assert (
        changed.result_type == request.result_type
        and changed.handler_name != request.handler_name
    )
    bindings = {
        alias if owner == old else owner: value for owner, value in parameters.items()
    }
    with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
        _run(
            module, FixedCyclicLogProbPathCertificateRuntimeProvider(changed, bindings)
        )

    target = project_for_target(module, discharge_module(module), "stan")
    assert any(
        item.computation == "certificate" and request.handler_name in item.reason
        for item in target.obstacles
    )


def test_acyclic_certificate_wire_and_handler_remain_verbatim():
    acyclic, _, _ = _acyclic_fixture()
    assert sha256(dumps(acyclic).encode()).hexdigest() == (
        "39d472896fbb9841741b39389609eff55a7e392e0fd98fdf5060ca7abf971f51"
    )
    assert acyclic.handler_name == (
        "__logprob_path_certificate_fbc458d59de013066ea495281bd83ec90e827a1a08063df7e8c744fe522ac5b0"
    )
