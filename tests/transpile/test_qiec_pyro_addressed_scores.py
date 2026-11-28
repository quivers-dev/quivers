"""Compare an authored two-shot measured program and its Pyro trace."""

from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from itertools import product

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import Compiler, parse
from quivers.dsl.qiec_lowering import QVR_SOURCE_VERSION
from quivers.qiec import RANDOM_SAMPLE, SCORE_ADD, NumberDomain, SiteId, run_program
from quivers.qiec import execution
from quivers.transpile import transpile

_SOURCE = """\
object UnitInput : FinSet 1
object Output : Real 1

effect Fork
    copy : Real -> Real

effect Relay
    pass : Real -> Real

instance fork : Fork
instance relay : Relay
instance random : Random
instance score : Score

handler twice for Fork : Real -> Real [coverage=total, forwards=none, implementation=authored]
    copy(x : Real) resumes omega =>
        let left <- resume(x)
        let right <- resume(1.0 - x)
        return left + right

handler identity for Relay : Real -> Real [coverage=total, forwards=none, implementation=authored]
    pass(x : Real) resumes 1 =>
        resume(x)

define draw(p : Real) : Bool !{random} =
    perform random.sample[Bool](site("b"), Bernoulli(p))

define subject(p : Real) : Real !{random, score} =
    handle relay with identity in
        handle fork with twice in
            let q <- perform fork.copy(p)
            let r <- perform relay.pass(q)
            let b <- draw(r)
            perform score.add(weight(r * r))
            if b then
                return r + r
            else
                return r

define work(p : Real) : Real !{random, score} =
    let result <- subject(p)
    perform score.add(weight(3.0 * p))
    return result

program measured(p : Real) : UnitInput -> Output
    let result <- work(p)
    return result

export measured
"""
_ROOT = ("measured#1", "work#2")
_DRAW_PATH = ("computations", "draw", "body")
_INNER_PATH = (
    "computations",
    "subject",
    "body",
    "handled",
    "handled",
    "then",
    "then",
    "then",
    "first",
)
_OUTER_PATH = ("computations", "work", "body", "then", "first")
_BERNOULLI_NAMES = ("b", "b@1")
_SITE_NAMES = ("b", "score", "b@1", "score@1", "score@2")
_ZERO_SOURCE = """\
object UnitInput : FinSet 1
object Output : Real 1

effect Stop
    stop : Real -> Real

instance stop : Stop
instance score : Score

handler abort for Stop : Real -> Real [coverage=total, forwards=none, implementation=authored]
    stop(x : Real) resumes 0 =>
        return x

define subject(p : Real) : Real !{score} =
    handle stop with abort in
        perform score.add(weight(3.0 * p))
        let after <- perform stop.stop(p)
        perform score.add(weight(after * after))
        return 2.0 * after

program measured(p : Real) : UnitInput -> Output
    let result <- subject(p)
    return result

export measured
"""


def _static(path, role="effect-request"):
    return str(SiteId.derive(QVR_SOURCE_VERSION, "source", path, role))


def _address(static, calls, shots=()):
    return (static, tuple(("call", name) for name in calls), shots)


def _word():
    draw = _static(_DRAW_PATH)
    replay = _static((*_DRAW_PATH, "generated", "replay-score"), "replay-score")
    inner = _static(_INNER_PATH)
    outer = _static(_OUTER_PATH)
    rows = []
    for shot, leaf in enumerate(("draw#4", "draw#5")):
        helper = (*_ROOT, "subject#3")
        rows.extend(
            (
                (str(RANDOM_SAMPLE), _address(draw, (*helper, leaf), (shot,))),
                (str(SCORE_ADD), _address(replay, (*helper, leaf), (shot,))),
                (str(SCORE_ADD), _address(inner, helper, (shot,))),
            )
        )
    rows.append((str(SCORE_ADD), _address(outer, _ROOT)))
    return tuple(rows)


def _program_address(address):
    # IRCall enters the one helper directly. Native execution enters its
    # ProgramEntry and then that helper through two calls before the
    # first helper-internal call. These two frames also advance its serial.
    static, frames, shots = address
    shifted = []
    for scope, key in frames:
        assert scope == "call"
        name, serial = key.rsplit("#", 1)
        shifted.append(f"{name}#{int(serial) + 2}")
    return _address(static, (*_ROOT, *shifted), shots)


def _oracle(p, bits):
    value = (1 + bits[0]) * p + (1 + bits[1]) * (1 - p)
    slope = bits[0] - bits[1]
    with localcontext() as context:
        context.prec = 70
        x = Decimal(p.numerator) / Decimal(p.denominator)

        def components(t):
            a, b = t, 1 - t
            return (
                (a if bits[0] else b).ln(),
                a * a,
                (b if bits[1] else a).ln(),
                b * b,
                3 * t,
            )

        scores = components(x)
        h = Decimal("1e-25")
        difference = tuple(
            (upper - lower) / (2 * h)
            for upper, lower in zip(components(x + h), components(x - h), strict=True)
        )
    first = 1 / p if bits[0] else -1 / (1 - p)
    second = -1 / (1 - p) if bits[1] else 1 / p
    derivatives = first, 2 * p, second, -2 * (1 - p), 3
    assert tuple(map(float, difference)) == pytest.approx(
        tuple(map(float, derivatives)), rel=1e-13, abs=0
    )
    return (value, slope), (scores, derivatives)


@pytest.fixture(scope="module", params=("1", "aff"), ids=("linear", "affine"))
def checked_route(request):
    pytest.importorskip("pyro")
    parsed = parse(_SOURCE.replace("resumes 1", f"resumes {request.param}"))
    plan = Compiler(parsed).effect_plan("measured")
    assert plan.discharge.obstacles == ()
    (subject,) = (
        item for item in plan.bundle.module.computations if item.name == "subject"
    )
    assert any(
        rewrite.computation == subject.id and rewrite.method == "handler-discharge"
        for rewrite in plan.discharge.rewrites
    )
    code = transpile(parsed, target="pyro")
    receipt = plan.target_evidence("pyro")
    assert receipt.artifact_digest == sha256(code).hexdigest()
    assert (
        receipt.projection == "program-plan" and receipt.helper_projection == "source"
    )
    assert receipt.external_derivative.missing == ("external derivative unverified",)
    return plan, code


def _pyro_trace(code, p, bits):
    pyro = pytest.importorskip("pyro")
    namespace = {"pyro": pyro}
    exec(compile(code, "<emitted Pyro model>", "exec"), namespace)
    point = torch.tensor(float(p), dtype=torch.float64, requires_grad=True)
    data = {
        name: point.new_tensor(float(bit))
        for name, bit in zip(_BERNOULLI_NAMES, bits, strict=True)
    }
    conditioned = pyro.poutine.condition(namespace["model"], data=data)
    trace = pyro.poutine.trace(conditioned).get_trace(point)
    trace.compute_log_prob()
    sites = {
        name: node for name, node in trace.nodes.items() if node["type"] == "sample"
    }
    return point, trace, sites


def _pair(value, point):
    (slope,) = torch.autograd.grad(value, point, retain_graph=True)
    return value.item(), slope.item()


def _check_rows(contributions, point, wanted):
    values, derivatives = wanted
    actual = tuple(_pair(item, point) for item in contributions)
    assert len(actual) == len(values)
    for pair, value, derivative in zip(actual, values, derivatives, strict=True):
        assert pair == pytest.approx((float(value), float(derivative)), abs=1e-13)


@pytest.mark.parametrize("p", (Fraction(1, 4), Fraction(3, 8)))
@pytest.mark.parametrize("bits", tuple(product((False, True), repeat=2)))
def test_replayed_value_complete_score_and_address_agree_across_stages(
    checked_route, monkeypatch, p, bits
):
    plan, code = checked_route
    word = _word()
    expected_value, expected_scores = _oracle(p, bits)
    score_handler = execution.score_handler
    rows = []

    def monitored_handler(**options):
        def observe(request, contribution):
            rows.append((request.address, request.core.origin.parents, contribution))

        return score_handler(**options, observer=observe)

    monkeypatch.setattr(execution, "score_handler", monitored_handler)
    for module in (plan.bundle.module, plan.discharge.module):
        rows.clear()
        point = torch.tensor(float(p), dtype=torch.float64, requires_grad=True)
        run = run_program(
            module,
            "measured",
            data={"p": point},
            sites={"b@0": bits[0], "b@1": bits[1]},
            numbers=NumberDomain.of(point),
            distributions=TensorBackend(),
        )
        requests = tuple(
            (item.detail["operation"], item.detail["address"])
            for item in run.result.trace
            if item.event == "operation.requested"
            and item.detail["operation"] in (str(RANDOM_SAMPLE), str(SCORE_ADD))
        )
        assert requests == word
        assert tuple(address for address, _, _ in rows) == tuple(
            word[i][1] for i in (1, 2, 4, 5, 6)
        )
        assert [tuple(map(str, lineage)) for _, lineage, _ in rows] == [
            (_static(_DRAW_PATH),),
            (),
            (_static(_DRAW_PATH),),
            (),
            (),
        ]
        _check_rows(
            (contribution for _, _, contribution in rows), point, expected_scores
        )
        assert _pair(run.value, point) == expected_value
        assert _pair(run.log_joint, point) == pytest.approx(
            tuple(float(sum(items)) for items in expected_scores), abs=1e-13
        )

    point, trace, sites = _pyro_trace(code, p, bits)
    assert tuple(sites) == _SITE_NAMES
    assert (
        tuple(
            name
            for name, site in sites.items()
            if not site["infer"].get("is_auxiliary")
        )
        == _BERNOULLI_NAMES
    )
    assert tuple(sites[name]["value"].item() for name in _BERNOULLI_NAMES) == bits
    assert tuple(
        _program_address(site["infer"]["qiec_address"]) for site in sites.values()
    ) == tuple(word[i][1] for i in (0, 2, 3, 5, 6))
    _check_rows(
        (site["log_prob_sum"] for site in sites.values()), point, expected_scores
    )
    assert _pair(trace.nodes["_RETURN"]["value"], point) == expected_value
    assert _pair(trace.log_prob_sum(), point) == pytest.approx(
        tuple(float(sum(items)) for items in expected_scores), abs=1e-13
    )


@pytest.mark.parametrize(
    "mutation", ("extra-linear-shot", "reverse-shots", "omit-score", "double-score")
)
def test_emitted_runtime_mutants_separate_address_and_score(checked_route, mutation):
    _, code = checked_route
    p, bits = Fraction(1, 4), (True, False)
    if mutation == "extra-linear-shot":
        before = b'if grade == "omega":\n                    continuation ='
        after = b'if grade != "0":\n                    continuation ='
    elif mutation == "reverse-shots":
        before = b"static, dynamic, (*resumptions, shot)"
        after = b"static, dynamic, (*resumptions, 1 - shot)"
    else:
        count = b"0" if mutation == "omit-score" else b"2"
        before = b'pyro.factor(name("score"), _qvr_qiec_array(weight))'
        after = b'pyro.factor(name("score"), ' + count + b" * _qvr_qiec_array(weight))"
    assert code.count(before) == 1
    point, trace, sites = _pyro_trace(code.replace(before, after), p, bits)
    assert tuple(sites) == _SITE_NAMES
    (value, slope), scores = _oracle(p, bits)
    total = tuple(float(sum(items)) for items in scores)
    assert _pair(trace.nodes["_RETURN"]["value"], point) == (value, slope)
    addresses = tuple(
        _program_address(site["infer"]["qiec_address"]) for site in sites.values()
    )
    expected = tuple(_word()[i][1] for i in (0, 2, 3, 5, 6))
    if mutation in ("extra-linear-shot", "reverse-shots"):
        assert addresses != expected
        assert _pair(trace.log_prob_sum(), point) == pytest.approx(total, abs=1e-13)
        wanted = ((0, 0), (0, 0), (1, 0, 0), (1, 0, 0), ())
        if mutation == "reverse-shots":
            wanted = ((1,), (1,), (0,), (0,), ())
        assert tuple(address[2] for address in addresses) == wanted
    else:
        assert addresses == expected
        score, derivative = _pair(trace.log_prob_sum(), point)
        offset = -1 if mutation == "omit-score" else 1
        authored = p * p + (1 - p) ** 2 + 3 * p
        authored_slope = 2 * p - 2 * (1 - p) + 3
        assert (score, derivative) == pytest.approx(
            (total[0] + offset * authored, total[1] + offset * authored_slope),
            abs=1e-13,
        )
        assert derivative != pytest.approx(total[1], abs=1e-13)


def test_grade_zero_drops_the_later_score_and_retains_the_prefix_address():
    pyro = pytest.importorskip("pyro")
    parsed = parse(_ZERO_SOURCE)
    plan = Compiler(parsed).effect_plan("measured")
    assert plan.discharge.obstacles == ()
    static = _static(("computations", "subject", "body", "handled", "first"))
    for module in (plan.bundle.module, plan.discharge.module):
        point = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
        run = run_program(module, "measured", data={"p": point}, sites={})
        addresses = tuple(
            item.detail["address"]
            for item in run.result.trace
            if item.event == "operation.requested"
            and item.detail["operation"] == str(SCORE_ADD)
        )
        assert addresses == (_address(static, ("measured#1", "subject#2")),)
        assert _pair(run.value, point) == (0.25, 1)
        assert _pair(run.log_joint, point) == (0.75, 3)
    namespace = {"pyro": pyro}
    exec(transpile(parsed, target="pyro"), namespace)
    point = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    trace = pyro.poutine.trace(namespace["model"]).get_trace(point)
    sites = tuple(
        name for name, node in trace.nodes.items() if node["type"] == "sample"
    )
    assert sites == ("score",)
    assert trace.nodes["score"]["infer"]["qiec_address"] == _address(static, ())
    assert _pair(trace.nodes["_RETURN"]["value"], point) == (0.25, 1)
    assert _pair(trace.log_prob_sum(), point) == (0.75, 3)
