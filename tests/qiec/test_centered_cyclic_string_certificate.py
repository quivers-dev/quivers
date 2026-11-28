"""Independent cyclic likelihood and derivative checks through checked QVR."""

from fractions import Fraction
from math import ceil, exp, floor, inf

import torch

from quivers.core.algebras import DivergentClosureError
from quivers.core.relations import centered_cyclic, certify_centered_cyclic_log_ratio
from quivers.core.relations.centered_cyclic import _CyclicCenteredChart
from quivers.core.relations.queries import _PathPlan
from quivers.qiec import (
    CenteredCyclicStringLogProbCertificateRequest,
    CenteredStringLogProbCertificateRuntimeProvider,
    FixedRelationRequest,
    discharge_module,
    dumps,
    loads,
)
from quivers.transpile import project_for_target
from tests.qiec.test_centered_string_certificate import (
    _check_certificate,
    _compiler,
    _decode,
    _module,
    _run,
)

_SOURCE = """object Letters : FinSet 2

relation channel [algebra=logprob, states=4]
    input Letters
    output Letters
    initial 0 [weight=common]
    initial 3 [weight=common]
    arc 0 -> 1 [labels=(eps, eps), weight=forward]
    arc 0 -> 1 [labels=(eps, eps), weight=forward]
    arc 1 -> 0 [labels=(eps, eps), weight=return_weight]
    arc 0 -> 2 [labels=(0, 0), weight=unit]
    arc 1 -> 2 [labels=(0, 1), weight=unit]
    arc 3 -> 2 [labels=(0, 1), weight=unit]
    final 2 [weight=unit]
"""
_MIDPOINT = """object Letters : FinSet 2

relation channel [algebra=logprob, states=4]
    input Letters
    output Letters
    initial 0 [weight=common]
    initial 3 [weight=common]
    arc 0 -> 0 [labels=(eps, eps), weight=forward]
    arc 0 -> 2 [labels=(0, 0), weight=unit]
    arc 3 -> 1 [labels=(0, 1), weight=forward]
    arc 1 -> 2 [labels=(eps, eps), weight=return_weight]
    final 2 [weight=unit]
"""

_SEARCH = {"max_states": 48, "max_work": 96, "max_closure_size": 3}
_BUDGET = dict(
    max_center_occurrences=64,
    max_dyadic_work=500,
    max_closure_cells=18,
    decimal_precision=34,
    max_decimal_exponent=800,
    max_decimal_work=480,
    max_exact_bits=8192,
    max_exact_work=4096,
)
_TOLERANCE = (1, 10**12)


def _case(word=(0,), *, source=_SOURCE):
    spec = _compiler(source).relation_specs["channel"]
    observation = FixedRelationRequest(
        relation=spec, operation="conditional_log_prob", words=((0,), word), **_SEARCH
    )
    request = CenteredCyclicStringLogProbCertificateRequest(
        observation=observation, absolute_log_tolerance=_TOLERANCE, **_BUDGET
    )
    owners = {
        name: torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
        for name, value in (
            ("common", -1e20),
            ("forward", -1),
            ("return_weight", -1),
            ("unit", 0),
        )
    }
    parameters = {
        identity: owners[name]
        for name, identity in zip(
            spec.weight_names, spec.weight_parameters, strict=True
        )
    }
    return request, parameters, owners


def _exp_bounds(log_weight):
    term = total = Fraction(1)
    for i in range(1, 57):
        term *= log_weight / i
        total += term
    lower = total + term * log_weight / 57
    # Rational outward compression keeps the subsequent logarithm
    # series small without losing the interval guarantee.
    scale = 10**42
    return Fraction(floor(lower * scale), scale), Fraction(ceil(total * scale), scale)


def _log_bounds(value):
    z = (value - 1) / (value + 1)
    power = z
    total = Fraction(0)
    for i in range(61):
        total += 2 * power / (2 * i + 1)
        power *= z * z
    return total, total + 2 * power / (123 * (1 - z * z))


def _oracle(forward, reverse):
    # Two primitive forward choices give a = 2*exp(forward).
    # Returning uses b = exp(reverse). The SCC exits carry output 0
    # at state 0 and output 1 at state 1. A disjoint bypass adds one
    # output-1 path. Cancel the common initial scale and then solve
    # the geometric repeat explicitly. The ratio is 1/(2+a*(1-b)).
    a_lo, a_hi = (2 * value for value in _exp_bounds(Fraction(forward)))
    b_lo, b_hi = _exp_bounds(Fraction(reverse))
    low = 2 + a_lo * (1 - b_hi)
    high = 2 + a_hi * (1 - b_lo)
    return -_log_bounds(high)[1], -_log_bounds(low)[0]


def test_checked_cyclic_ratio_and_live_host_gradients_match_geometric_oracle():
    request, parameters, owners = _case()
    wire = dumps(request)
    decoded = loads(wire)
    assert decoded == request and dumps(decoded) == wire
    module = _module(decoded, _SOURCE)
    provider = CenteredStringLogProbCertificateRuntimeProvider(decoded, parameters)
    relation = request.observation.relation.bind(parameters)
    ordinary = relation.conditional(((0,),), **_SEARCH)
    assert ordinary.log_prob(((0,),)).item() == 0.0
    for reverse in (-1, -2):
        with torch.no_grad():
            owners["return_weight"].fill_(reverse)
        oracle = _oracle(-1, reverse)
        native = _check_certificate(_run(module, provider), oracle)
        host = certify_centered_cyclic_log_ratio(
            relation,
            ((0,),),
            ((0,),),
            absolute_log_tolerance=Fraction(*_TOLERANCE),
            **_SEARCH,
            **_BUDGET,
        )
        assert Fraction(host.weight.item()) == native
        assert host.lower <= oracle[0] <= oracle[1] <= host.upper
        live = tuple(owners[name] for name in ("common", "forward", "return_weight"))
        result = tuple(x.item() for x in torch.autograd.grad(host.weight, live))
        a, b = 2 * exp(-1), exp(reverse)
        total = 2 + a * (1 - b)
        torch.testing.assert_close(
            torch.tensor(result, dtype=torch.float64),
            torch.tensor([0, -a * (1 - b) / total, a * b / total], dtype=torch.float64),
            atol=1e-14,
            rtol=0,
        )
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        assert any(
            item.computation == "check_ratio" and request.handler_name in item.reason
            for item in project_for_target(module, discharged, target).obstacles
        )


def test_cyclic_ratio_distinguishes_divergence_unverified_finiteness_and_cells():
    request, parameters, owners = _case()
    module = _module(request, _SOURCE)

    def run(current):
        return _run(
            module, CenteredStringLogProbCertificateRuntimeProvider(current, parameters)
        )

    assert run(request)[0] == "certified"
    report = run(request.with_(max_closure_cells=17))
    assert report[0] == "resource_limit" and "max_closure_cells" in report[1]
    report = run(
        request.with_(observation=request.observation.with_(max_closure_size=2))
    )
    assert report[0] == "resource_limit" and "max_closure_size" in report[1]
    with torch.no_grad():
        owners["return_weight"].fill_(2)
    report = run(request)
    assert report[0] == "divergent_closure" and not report[2].item()
    with torch.no_grad():
        owners["forward"].fill_(-0.36)
        owners["return_weight"].fill_(-0.36)
    report = run(request.with_(decimal_precision=2))
    assert report[0] == "unverified_closure" and "upper star" in report[1]
    assert not report[2].item() and run(request)[0] == "certified"
    with torch.no_grad():
        owners["forward"].fill_(-1000)
    report = run(request.with_(max_decimal_exponent=100))
    assert report[0] == "resource_limit" and "max_decimal_exponent" in report[1]


def test_cyclic_zero_numerator_and_undefined_denominator():
    request, parameters, owners = _case(word=())
    provider = CenteredStringLogProbCertificateRuntimeProvider(request, parameters)
    module = _module(request, _SOURCE)
    result = _run(module, provider)
    assert result[0] == "certified" and result[2].item() and result[3].item() == -inf
    assert not result[5].item() and _decode(result[4][2]) == 0
    with torch.no_grad():
        owners["common"].fill_(-inf)
    result = _run(module, provider)
    assert result[0] == "undefined_normalizer" and not result[2].item()


def test_cyclic_proof_rejects_a_mutated_internal_occurrence(monkeypatch):
    request, parameters, _ = _case()
    original = _CyclicCenteredChart._gauge

    def changed(self, weights, budget):
        offset, factors = original(self, weights, budget)
        first = self.internal[0]
        return offset, [
            (slot, exact - 1, candidate - 1)
            if slot == first
            else (slot, exact, candidate)
            for slot, exact, candidate in factors
        ]

    monkeypatch.setattr(_CyclicCenteredChart, "_gauge", changed)
    provider = CenteredStringLogProbCertificateRuntimeProvider(request, parameters)
    report = _run(_module(request, _SOURCE), provider)
    assert report[0] == "unverified_accuracy"
    assert "changed an original internal occurrence" in report[1]


def test_cyclic_ratio_proof_corrects_rounding_on_a_condensation_boundary(monkeypatch):
    request, parameters, owners = _case(source=_MIDPOINT)
    with torch.no_grad():
        owners["return_weight"].fill_(-(2**-53))
    # A single productive loop contributes 1/(1-c) to the numerator.
    # The bypass contributes t to the denominator in the common gauge.
    # The exact log probability is -log(1+(1-c)t).
    c_lo, c_hi = _exp_bounds(Fraction(-1))
    t_lo, t_hi = _exp_bounds(-1 - Fraction(1, 2**53))
    lo, hi = 1 + (1 - c_hi) * t_lo, 1 + (1 - c_lo) * t_hi
    oracle = -_log_bounds(hi)[1], -_log_bounds(lo)[0]
    module = _module(request, _MIDPOINT)
    provider = CenteredStringLogProbCertificateRuntimeProvider(request, parameters)
    _check_certificate(_run(module, provider), oracle)

    exact_extrema = centered_cyclic._path_error_extrema

    def forget_boundary_errors(graph, errors, budget):
        return exact_extrema(graph, dict.fromkeys(errors, Fraction(0)), budget)

    monkeypatch.setattr(centered_cyclic, "_path_error_extrema", forget_boundary_errors)
    report = _run(module, provider)
    assert report[0] == "certified"
    assert _decode(report[4][1]) < oracle[0]


def test_finite_upper_star_does_not_report_a_numerical_refusal_as_divergence(
    monkeypatch,
):
    request, parameters, _ = _case()

    def unavailable(*_args, **_kwargs):
        raise DivergentClosureError("mutated numerical star")

    monkeypatch.setattr(_PathPlan, "evaluate", unavailable)
    provider = CenteredStringLogProbCertificateRuntimeProvider(request, parameters)
    report = _run(_module(request, _SOURCE), provider)
    assert report[0] == "undecided_closure"
    assert "verified finite upper star" in report[1] and not report[2].item()
