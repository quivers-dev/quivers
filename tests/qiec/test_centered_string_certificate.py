"""Native centered ratio certificates against exact rational-series oracles."""

from fractions import Fraction
from hashlib import sha256
from math import inf

import pytest
import torch

from quivers.core.relations import _centered_accuracy
from quivers.core.relations.accuracy import _RationalBudget
from quivers.core.relations.queries import _ActiveObservation
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    CenteredStringLogProbCertificateRequest,
    CenteredStringLogProbCertificateRuntimeProvider,
    CenteredStringLogProbRequest,
    ExecutionFailure,
    FixedRelationRequest,
    QiecModule,
    RuntimeConfiguration,
    TreeStringGraphRequest,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target

_MODULE = "centered_certificates"
_HALF_ULP = Fraction(1, 2**53)
_LIMITS = {"max_states": 30, "max_work": 64}
_CENTER_LIMITS = {"max_center_occurrences": 50, "max_dyadic_work": 300}
_HEADER = "object Letters : FinSet 2\n\n"
_GRAPH = """
string_graph model
    variable input_word : Letters [observed=(0)]
    variable output_word : Letters
    factor channel(input_word, output_word)
"""
_SOURCE = (
    _HEADER
    + """relation channel [algebra=logprob, states=3]
    input Letters
    output Letters
    initial 0 [weight=common]
    arc 0 -> 2 [labels=(0, 0), weight=unit]
    arc 0 -> 1 [labels=(0, 1), weight=minus_one]
    arc 1 -> 2 [labels=(eps, eps), weight=half_ulp]
    final 2 [weight=unit]
"""
    + _GRAPH
)
_BOTH = (
    _HEADER
    + """relation channel [algebra=logprob, states=7]
    input Letters
    output Letters
    initial 0 [weight=common]
    arc 0 -> 1 [labels=(0, 0), weight=minus_one]
    arc 1 -> 6 [labels=(eps, eps), weight=half_ulp]
    arc 0 -> 2 [labels=(0, 0), weight=minus_one]
    arc 2 -> 3 [labels=(eps, eps), weight=half_ulp]
    arc 3 -> 4 [labels=(eps, eps), weight=minus_one]
    arc 4 -> 6 [labels=(eps, eps), weight=half_ulp]
    arc 0 -> 6 [labels=(0, 1), weight=unit]
    arc 0 -> 5 [labels=(0, 1), weight=minus_one]
    arc 5 -> 6 [labels=(eps, eps), weight=half_ulp]
    final 6 [weight=unit]
"""
    + _GRAPH
)


def _compiler(source):
    compiler = Compiler(parse(source), module_name=_MODULE)
    compiler.compile()
    return compiler


def _observation(source, mode, word):
    compiler = _compiler(source)
    relation = compiler.relation_specs["channel"]
    query = (
        FixedRelationRequest(
            relation=relation,
            operation="conditional_log_prob",
            words=((0,), word),
            **_LIMITS,
        )
        if mode == "conditional"
        else TreeStringGraphRequest(
            graph=compiler.string_graph_specs["model"],
            variable="output_word",
            word=word,
            **_LIMITS,
        )
    )
    return CenteredStringLogProbRequest(observation=query, **_CENTER_LIMITS), relation


def _fixture(source=_SOURCE, mode="conditional", word=(0,), tolerance=(1, 10**15)):
    centered, relation = _observation(source, mode, word)
    values = {
        "common": -1e20,
        "unit": 0.0,
        "minus_one": -1.0,
        "half_ulp": -float(_HALF_ULP),
    }
    owners = {
        name: torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
        for name, value in values.items()
    }
    parameters = {
        identity: owners[name]
        for name, identity in zip(
            relation.weight_names, relation.weight_parameters, strict=True
        )
    }
    request = CenteredStringLogProbCertificateRequest(
        observation=centered,
        absolute_log_tolerance=tolerance,
        decimal_precision=30,
        max_decimal_exponent=400,
        max_decimal_work=192,
        max_exact_bits=4096,
        max_exact_work=1280,
    )
    return request, parameters, owners


def _module(request, source=_SOURCE):
    report = "(String * String * Bool * LogWeight * Tensor[String]([3, 2]) * Bool * Tensor[Int]([3]))"
    name = request.handler_name
    program = (
        source
        + f"""
instance audit : Compute[Unit, {report}]
handler {name} for Compute[Unit, {report}] : {report} -> {report} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define check_ratio : {report} !{{}} =
    handle audit with {name} in
        let result <- perform audit.apply(unit)
        return result
"""
    )
    module = loads(dumps(_compiler(module_to_source(parse(program))).qiec_module))
    assert isinstance(module, QiecModule)
    return module


def _run(module, provider):
    return run_named(
        module, "check_ratio", runtime=RuntimeConfiguration(providers=(provider,))
    ).value


def _evaluate(request, parameters, source=_SOURCE):
    return _run(
        _module(request, source),
        CenteredStringLogProbCertificateRuntimeProvider(request, parameters),
    )


def _oracle(rare_log: Fraction) -> tuple[Fraction, Fraction]:
    # Alternating exponential terms decrease from k=1 for these inputs.
    term = total = Fraction(1)
    for k in range(1, 39):
        term *= rare_log / k
        total += term
    masses = (total + term * rare_log / 39, total)

    def log1p(x):
        z = x / (2 + x)
        power = z
        estimate = Fraction(0)
        for i in range(21):
            estimate += 2 * power / (2 * i + 1)
            power *= z * z
        return estimate, estimate + 2 * power / (43 * (1 - z * z))

    return -log1p(masses[1])[1], -log1p(masses[0])[0]


def _decode(value):
    numerator, denominator = value
    return Fraction(int(numerator, 16), int(denominator, 16))


def _check_certificate(report, oracle):
    status, reason, available, score, witness, endpoints, work = report
    assert (status, reason, available.item(), endpoints.item()) == (
        "certified",
        "",
        True,
        True,
    )
    assert not score.requires_grad and all(count > 0 for count in work.tolist())
    low, high, error = map(_decode, witness)
    assert low <= oracle[0] <= oracle[1] <= high
    candidate = Fraction(score.item())
    assert error == max(abs(candidate - low), abs(candidate - high))
    return candidate


def test_historical_rounded_factor_mutant_falsely_certifies_at_the_midpoint(
    monkeypatch,
):
    request, parameters, _ = _fixture(tolerance=(1, 25_000_000_000_000_000))
    module = _module(request)
    provider = CenteredStringLogProbCertificateRuntimeProvider(request, parameters)
    report = _run(module, provider)
    assert report[0] == "unverified_accuracy" and not report[2].item()
    assert "requested absolute centered log ratio error" in report[1]
    original_extrema = _centered_accuracy._path_error_extrema

    def rounded_only(graph, errors, budget):
        return original_extrema(graph, dict.fromkeys(errors, Fraction(0)), budget)

    monkeypatch.setattr(_centered_accuracy, "_path_error_extrema", rounded_only)
    mutant = _run(module, provider)
    observed = _check_certificate(mutant, _oracle(Fraction(-1)))
    correct_lo, _ = _oracle(-1 - _HALF_ULP)
    assert _decode(mutant[4][1]) < correct_lo
    assert _decode(mutant[4][2]) < Fraction(*request.absolute_log_tolerance)
    assert correct_lo - observed > Fraction(*request.absolute_log_tolerance)


@pytest.mark.parametrize("mode", ("conditional", "tree"))
def test_live_checked_certificate_of_both_residuals_and_exact_offset(mode, monkeypatch):
    request, parameters, owners = _fixture(_BOTH, mode)
    wire = dumps(request)
    decoded = loads(wire)
    assert decoded == request and dumps(decoded) == wire
    original_binding = CenteredStringLogProbRequest._binding
    reads = []
    bindings = []
    nominal_plan, nominal_weights = original_binding(request.observation, parameters)

    def observed_binding(self, values):
        bindings.append(1)
        plan, read = original_binding(self, values)

        def once():
            reads.append(1)
            return read()

        return plan, once

    monkeypatch.setattr(CenteredStringLogProbRequest, "_binding", observed_binding)
    module = _module(decoded, _BOTH)
    provider = CenteredStringLogProbCertificateRuntimeProvider(decoded, parameters)
    scores = []
    for invocation, tail in enumerate((-_HALF_ULP, Fraction(-1, 2)), 1):
        with torch.no_grad():
            owners["half_ulp"].fill_(float(tail))
        # With t = exp(x), enumerate the paths. The observed weight
        # is t + t^2 and the total is 1 + 2t + t^2 in the common gauge.
        x = -1 + tail
        exact = tuple(bound + x for bound in _oracle(x))
        scores.append(_check_certificate(_run(module, provider), exact))
        assert scores[-1] == Fraction(
            nominal_plan.evaluate(nominal_weights()).log_prob.item()
        )
        assert len(reads) == invocation and len(bindings) == 1
    assert scores[0] != scores[1]


def test_certified_log_zero_and_undefined_normalizer_have_distinct_reports():
    request, parameters, owners = _fixture(word=())
    module = _module(request)
    provider = CenteredStringLogProbCertificateRuntimeProvider(request, parameters)
    report = _run(module, provider)
    assert report[0] == "certified" and report[2].item()
    assert report[3].item() == -inf and not report[5].item()
    assert _decode(report[4][2]) == 0
    with torch.no_grad():
        owners["common"].fill_(-inf)
    report = _run(module, provider)
    assert report[0] == "undefined_normalizer"
    assert not report[2].item() and not report[5].item()


def test_resource_exponent_and_output_range_refusals_remain_distinct():
    request, parameters, owners = _fixture()
    for setting in ("max_decimal_work", "max_exact_work"):
        result = _evaluate(request.with_(**{setting: 0}), parameters)
        assert result[0] == "resource_limit" and setting in result[1]
        assert not result[2].item()
    bit_limit = _evaluate(request.with_(max_exact_bits=64), parameters)
    assert bit_limit[0] == "resource_limit" and "max_exact_bits" in bit_limit[1]
    float32 = {
        owner: value.detach().to(torch.float32) for owner, value in parameters.items()
    }
    assert _evaluate(request, float32)[0] == "unsupported_precision"
    with torch.no_grad():
        owners["minus_one"].fill_(-1000)
    report = _evaluate(request.with_(max_decimal_exponent=100), parameters)
    assert report[0] == "resource_limit" and "max_decimal_exponent" in report[1]
    with torch.no_grad():
        owners["minus_one"].fill_(-9e307)
        owners["half_ulp"].fill_(-9e307)
    report = _evaluate(request, parameters)
    assert report[0] == "numerical_range" and "reduced path factor" in report[1]


def test_frozen_legacy_identity_and_exact_native_observation_and_target():
    request, parameters, _ = _fixture()
    old = request.observation
    assert (
        sha256(dumps(old).encode()).hexdigest()
        == "5b0d6ed40608a77d574799060d311c884b92a6b4558be8a4833624b14d403a9f"
    )
    assert (
        old.handler_name
        == "__centered_string_log_prob_ca1c61be690611b556791a3b8bc56778f0924322790be71c0d97539a43aad95c"
    )
    module = _module(request)
    changed = request.with_(absolute_log_tolerance=(1, 10**14))
    assert changed.result_type == request.result_type
    with pytest.raises(ExecutionFailure, match="centered_string_logprob_certificate"):
        _run(
            module, CenteredStringLogProbCertificateRuntimeProvider(changed, parameters)
        )
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        obstacles = project_for_target(module, discharged, target).obstacles
        assert any(
            item.computation == "check_ratio" and request.handler_name in item.reason
            for item in obstacles
        )


def test_rounding_path_extrema_include_parallel_arcs_and_both_boundary_marks():
    graph = _ActiveObservation(
        (((2, 2), (2, 3)), ((3, 4),), (), ()),
        ((0, 0), (1, 1)),
        ((1, 5), (2, 6), (3, 7)),
        ((0,), (1,), (2,), (3,)),
    )
    errors = {
        slot: Fraction(raw, 128)
        for slot, raw in enumerate((3, -2, -5, 7, -4, 19, 11, -9))
    }
    # The independent path table includes an empty path at initial state 1.
    paths = ((0, 2, 6), (0, 3, 6), (1, 4, 7), (1, 5))
    sums = tuple(sum(errors[slot] for slot in path) for path in paths)
    found = _centered_accuracy._path_error_extrema(
        graph, errors, _RationalBudget(128, 50)
    )
    assert found == (min(sums), max(sums)) == (Fraction(-15, 128), Fraction(21, 128))


def test_centered_proof_shares_arithmetic_budgets_across_both_charts():
    request, parameters, _ = _fixture(_BOTH)
    module = _module(request, _BOTH)
    report = _run(
        module, CenteredStringLogProbCertificateRuntimeProvider(request, parameters)
    )
    assert report[0] == "certified"
    decimal_work, exact_work, _ = report[6].tolist()
    for field, limit in (
        ("max_decimal_work", decimal_work),
        ("max_exact_work", exact_work),
    ):
        request_at_limit = request.with_(**{field: limit})
        report = _run(
            module,
            CenteredStringLogProbCertificateRuntimeProvider(
                request_at_limit, parameters
            ),
        )
        assert report[0] == "certified"
        below = request.with_(**{field: limit - 1})
        refused = _run(
            module, CenteredStringLogProbCertificateRuntimeProvider(below, parameters)
        )
        assert refused[0] == "resource_limit" and field in refused[1]
