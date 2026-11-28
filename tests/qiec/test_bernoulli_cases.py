"""Check addressed Bernoulli laws across runtime constructor selections."""

from argparse import Namespace
from dataclasses import replace
from fractions import Fraction
from itertools import product
import json
from math import log

import pytest
import torch

from quivers.cli.run import main as run_cli
from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    RANDOM_SAMPLE,
    BernoulliTensorAxis,
    Bind,
    Case,
    JvpBoundaryError,
    JvpUnsupported,
    Perform,
    SiteEstimator,
    plan_bernoulli_batch,
    plan_bernoulli_gradient,
    sample_program,
)
from quivers.transpile import UnsupportedConstruct, transpile

_SOURCE = """\
object UnitInput : FinSet 1
object Output : Real 1

family Mode : Type
    constructor Left : Real -> Mode
    constructor Right : Real -> Mode

effect Gate
    tune : Real -> Real

instance gate : Gate
instance random : Random

handler adjust for Gate : Real -> Real [coverage=total, forwards=none, implementation=authored]
    tune(v : Real) resumes 1 =>
        resume(0.25 + 0.5 * v)

define mode(s : Real) : Mode !{} =
    if s > 0.0 then
        return construct Left(s) as Mode
    else
        return construct Right(s) as Mode

define sample_once(prob : Real) : Bool !{random} =
    perform random.sample[Bool](site("shared"), Bernoulli(prob))

define choose(route : Mode, p : Real, q : Real) : Real !{gate, random} =
    case route motive => Real
        Left(offset) =>
            let chance <- perform gate.tune(p)
            let draw <- sample_once(chance)
            if draw then
                return offset + p + 1.0
            else
                return offset + p
        Right(offset) =>
            let chance <- perform gate.tune(q)
            let draw <- sample_once(chance)
            if draw then
                return offset + q + 2.0
            else
                return offset + q

define selected(s : Real, p : Real, q : Real) : Real !{random} =
    let route <- mode(s)
    handle gate with adjust in
        choose(route, p, q)

program picked(s : Real, p : Real, q : Real) : UnitInput -> Output
    let picked_value <- selected(s, p, q)
    return picked_value

export picked
"""
_DISJOINT = _SOURCE.replace(
    "let draw <- sample_once(chance)\n            if draw then\n                return offset + p",
    'let draw <- perform random.sample[Bool](site("left"), Bernoulli(chance))\n'
    "            if draw then\n                return offset + p",
).replace(
    "let draw <- sample_once(chance)\n            if draw then\n                return offset + q",
    'let draw <- perform random.sample[Bool](site("right"), Bernoulli(chance))\n'
    "            if draw then\n                return offset + q",
)
_POINTS = (
    (Fraction(1, 2), Fraction(1, 4), Fraction(3, 4)),
    (Fraction(-1, 2), Fraction(1, 4), Fraction(3, 4)),
)
_DIRECTIONS = ((2, -1, 3), (-3, 2, -2))
_COTANGENTS = (2, -3)


def _entry(module, name):
    return next(item for item in module.computations if item.name == name)


def _policy(perform):
    assert isinstance(perform, Perform)
    assert perform.request.origin.static_site is not None
    return SiteEstimator(site=perform.request.origin.static_site, estimator="enumerate")


def _source_policies(module, disjoint):
    if not disjoint:
        return (_policy(_entry(module, "sample_once").body),)
    case = _entry(module, "choose").body
    assert isinstance(case, Case)
    policies = []
    for branch in case.branches:
        assert isinstance(branch.body, Bind)
        assert isinstance(branch.body.then, Bind)
        policies.append(_policy(branch.body.then.first))
    return tuple(policies)


def _compile(source):
    compiler = Compiler(parse(source), module_name="addressed-bernoulli-case")
    module = compiler.qiec_module
    assert module is not None
    plan = compiler.effect_plan("selected")
    assert plan.discharge.obstacles == ()
    owner = _entry(module, "selected")
    assert owner.id in {rewrite.computation for rewrite in plan.discharge.rewrites}
    return compiler, module, plan.discharge.module, owner.id


def _oracle(point, bit):
    s, p, q = point
    branch = 1 if s > 0 else 2
    parameter = p if branch == 1 else q
    probability = Fraction(1, 4) + parameter / 2
    mass = probability if bit else 1 - probability
    value = s + parameter + branch * bit
    slope = (1, int(branch == 1), int(branch == 2))
    score = (1 if bit else -1) / (2 * mass)
    score_gradient = (0, score if branch == 1 else 0, score if branch == 2 else 0)
    return mass, value, slope, score_gradient


def _dot(left, right):
    return sum(x * y for x, y in zip(left, right, strict=True))


def _gradient(output, inputs):
    return tuple(
        0 if value is None else value.item()
        for value in torch.autograd.grad(output, inputs, allow_unused=True)
    )


def _inputs(point):
    return tuple(
        torch.tensor(float(value), dtype=torch.float64, requires_grad=True)
        for value in point
    )


def _requests(trace):
    return [
        event.detail["address"]
        for event in trace
        if event.event == "operation.requested"
        and event.detail["operation"] == str(RANDOM_SAMPLE)
    ]


class _ReplayScores(TensorBackend):
    """Record the actual conditioned Boolean seen by differentiable scoring."""

    def __init__(self):
        self.values = []

    def sample(self, distribution, rng):
        pytest.fail("the inactive branch requested an unconditioned draw")

    def log_prob(self, distribution, value, keep_batch):
        self.values.append(bool(value))
        return super().log_prob(distribution, value, keep_batch)


@pytest.mark.parametrize("disjoint", (False, True))
def test_native_discharge_and_public_pyro_keep_selected_site_and_live_score(disjoint):
    pyro = pytest.importorskip("pyro")
    source = _DISJOINT if disjoint else _SOURCE
    compiler, raw, discharged, _ = _compile(source)
    report = compiler.effect_plan("picked").check_target("pyro")
    assert report.projection == "program-plan" and report.helper_projection == "source"
    assert report.refusals == ()
    assert report.external_derivative.missing == ("external derivative unverified",)
    with pytest.raises(
        UnsupportedConstruct, match="requires the `case` QIEC capability"
    ):
        transpile(parse(source), target="stan")
    namespace = {"pyro": pyro}
    exec(transpile(parse(source), target="pyro"), namespace)
    policies = _source_policies(raw, disjoint)
    for row, point in enumerate(_POINTS):
        label = ("left", "right")[row] if disjoint else "shared"
        static = str(policies[row if disjoint else 0].site)
        calls = (("call", "picked#1"), ("call", "selected#2"), ("call", "choose#4"))
        if not disjoint:
            calls += (("call", "sample_once#5"),)
        for bit in (False, True):
            mass, value, slope, score = _oracle(point, bit)
            for module in raw, discharged:
                arguments = _inputs(point)
                backend = _ReplayScores()
                run = sample_program(
                    module,
                    "picked",
                    data=dict(zip(("s", "p", "q"), arguments, strict=True)),
                    sites={label: bit},
                    distributions=backend,
                )
                assert _requests(run.result.trace) == [(static, calls, ())]
                assert backend.values == [bit]
                assert run.value.item() == value
                assert run.log_joint.item() == pytest.approx(log(mass))
                assert _gradient(run.value, arguments) == slope
                assert _gradient(run.log_joint, arguments) == pytest.approx(score)
            arguments = _inputs(point)
            observed = torch.tensor(float(bit), dtype=torch.float64)
            model = pyro.poutine.condition(namespace["model"], data={label: observed})
            trace = pyro.poutine.trace(model).get_trace(*arguments)
            sites = {
                name: node
                for name, node in trace.nodes.items()
                if node["type"] == "sample"
            }
            assert tuple(sites) == (label,)
            assert sites[label]["value"].item() == bit
            target_value = trace.nodes["_RETURN"]["value"]
            assert target_value.item() == value
            target_score = trace.log_prob_sum()
            assert target_score.item() == pytest.approx(log(mass))
            assert _gradient(target_value, arguments) == slope
            assert _gradient(target_score, arguments) == pytest.approx(score)


def test_scalar_enumeration_and_fixed_batch_joint_jvp_vjp():
    _, raw, discharged, owner = _compile(_SOURCE)
    policies = _source_policies(raw, False)
    local = (
        str(policies[0].site),
        (("call", "choose#2"), ("call", "sample_once#3")),
        (),
    )
    batch = plan_bernoulli_batch(discharged, owner, "Items", (), _POINTS, policies)
    assert batch.shape == (local,)
    reverse_rows = plan_bernoulli_batch(
        discharged, owner, "Items", (), _POINTS[::-1], policies
    )
    assert reverse_rows.shape == batch.shape
    assert reverse_rows.expected.values == batch.expected.values[::-1]
    with pytest.raises(JvpUnsupported, match="WithTraceAddresses"):
        BernoulliTensorAxis(discharged, owner, "Items", 0)
    with pytest.raises(JvpBoundaryError, match="input-dependent equality boundary"):
        plan_bernoulli_gradient(
            discharged, owner, (0, *_POINTS[0][1:]), (1, 0, 0), policies
        )

    expected = []
    slopes = []
    for point, direction in zip(_POINTS, _DIRECTIONS, strict=True):
        atoms = [_oracle(point, bit) for bit in (False, True)]
        expected.append(sum(mass * value for mass, value, _, _ in atoms))
        slopes.append(
            tuple(
                sum(
                    mass * (slope[i] + value * score[i])
                    for mass, value, slope, score in atoms
                )
                for i in range(len(point))
            )
        )
        scalar = plan_bernoulli_gradient(discharged, owner, point, direction, policies)
        assert scalar.paths == 2 and len(scalar.visits) == 1
        assert scalar.visits[0].address == local
        assert scalar.visits[0].probability == atoms[1][0]
        assert (scalar.exact.value, scalar.exact.tangent) == (
            expected[-1],
            _dot(direction, slopes[-1]),
        )
    assert batch.expected.values == tuple(expected)
    assert batch.expected.jvp((), _DIRECTIONS) == tuple(
        _dot(gradient, direction)
        for gradient, direction in zip(slopes, _DIRECTIONS, strict=True)
    )
    assert batch.expected.vjp(_COTANGENTS).rows == tuple(
        tuple(c * slope for slope in gradient)
        for c, gradient in zip(_COTANGENTS, slopes, strict=True)
    )

    total_mass = 0
    for bits in product((False, True), repeat=2):
        tapes = tuple(
            next(path.tape for path in paths if path.tape.choices == ((local, bit),))
            for bit, paths in zip(bits, batch.paths, strict=True)
        )
        assert tuple(tape.addresses for tape in tapes) == (
            (("Items", 0, local),),
            (("Items", 1, local),),
        )
        result = batch.replay(tapes)
        first, second = (
            _oracle(point, bit) for point, bit in zip(_POINTS, bits, strict=True)
        )
        assert result.probability == first[0] * second[0]
        assert result.linearization.values == (first[1], second[1])
        assert result.linearization.jvp((), _DIRECTIONS) == (
            _dot(first[2], _DIRECTIONS[0]),
            _dot(second[2], _DIRECTIONS[1]),
        )
        assert result.linearization.vjp(_COTANGENTS).rows == (
            tuple(_COTANGENTS[0] * x for x in first[2]),
            tuple(_COTANGENTS[1] * x for x in second[2]),
        )
        total_mass += result.probability
        wrong = replace(
            tapes[1], choices=(((local[0], local[1][1:], local[2]), bits[1]),)
        )
        with pytest.raises(ValueError, match="original addresses"):
            batch.replay((tapes[0], wrong))
    assert total_mass == 1


def test_disjoint_arms_are_scalar_but_refuse_a_fixed_batch():
    _, raw, discharged, owner = _compile(_DISJOINT)
    policies = _source_policies(raw, True)
    for point, policy in zip(_POINTS, policies, strict=True):
        plan = plan_bernoulli_gradient(discharged, owner, point, (1, 0, 0), (policy,))
        assert plan.visits[0].address == (str(policy.site), (("call", "choose#2"),), ())
        assert [visit.probability for visit in plan.visits] == [_oracle(point, True)[0]]
    with pytest.raises(JvpUnsupported, match="unreached sites"):
        plan_bernoulli_batch(discharged, owner, "Items", (), _POINTS, policies)


@pytest.mark.parametrize(
    "mutation", ("shift_address", "duplicate_draw", "select_wrong_case")
)
def test_reached_source_mutations_separate_addresses_and_measure(mutation):
    source = _SOURCE
    if mutation == "shift_address":
        source = source.replace(
            "define choose(",
            "define identity(x : Real) : Real !{} =\n    return x\n\ndefine choose(",
        ).replace(
            "let chance <- perform gate.tune(q)\n            let draw <- sample_once(chance)",
            "let chance <- perform gate.tune(q)\n"
            "            let same <- identity(chance)\n"
            "            let draw <- sample_once(same)",
        )
    elif mutation == "duplicate_draw":
        source = source.replace(
            "let chance <- perform gate.tune(q)\n            let draw <- sample_once(chance)",
            "let chance <- perform gate.tune(q)\n"
            "            let ignored <- sample_once(chance)\n"
            "            let draw <- sample_once(chance)",
        )
    else:
        source = source.replace("return construct Right(s)", "return construct Left(s)")
    _, raw, discharged, owner = _compile(source)
    policies = _source_policies(raw, False)
    right = _POINTS[1]
    chosen = plan_bernoulli_gradient(discharged, owner, right, (1, 0, 0), policies)
    if mutation == "select_wrong_case":
        assert chosen.paths == 2
        assert chosen.exact.value != sum(
            mass * value
            for mass, value, _, _ in (_oracle(right, bit) for bit in (False, True))
        )
        assert chosen.visits[0].probability == Fraction(3, 8)
        return
    if mutation == "shift_address":
        assert chosen.paths == 2
        assert chosen.visits[0].address[1][-1] == ("call", "sample_once#4")
        assert chosen.visits[0].probability == Fraction(5, 8)
    else:
        assert chosen.paths == 4
        assert {visit.address[1][-1] for visit in chosen.visits} == {
            ("call", "sample_once#3"),
            ("call", "sample_once#4"),
        }
        assert {visit.probability for visit in chosen.visits} == {Fraction(5, 8)}
    with pytest.raises(JvpUnsupported, match="same ordered draw addresses"):
        plan_bernoulli_batch(discharged, owner, "Items", (), _POINTS, policies)


def test_public_cli_replays_selected_constructor_site(tmp_path, capsys):
    path = tmp_path / "selected.qvr"
    path.write_text(_DISJOINT)
    args = Namespace(
        file=path,
        computation="picked",
        arguments=("-0.5", "0.25", "0.75"),
        site=("right=true",),
        json=True,
    )
    assert run_cli(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["value"] == 2.25 and result["log_joint"] == pytest.approx(
        log(Fraction(5, 8))
    )
    requests = [
        event["detail"]["address"]
        for event in result["trace"]
        if event["event"] == "operation.requested"
        and event["detail"]["operation"] == str(RANDOM_SAMPLE)
    ]
    assert len(requests) == 1 and requests[0][1][-1] == ["call", "choose#4"]
    args.site = ("right=true", "left=false")
    assert run_cli(args) == 1
    assert "left" in capsys.readouterr().out
