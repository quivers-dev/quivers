"""Check emitted Pyro's Bernoulli support, fixed-atom scores and gradients."""

import ast
import math

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import Compiler, parse
from quivers.qiec import NumberDomain, sample_program
from quivers.transpile import transpile

_SOURCE = """\
object UnitInput : FinSet 1
object Output : Real 1
instance random : Random

define choose(p : Real) : Bool !{random} =
    perform random.sample[Bool](site("outcome"), Bernoulli(p))

program flipped(p : Real) : UnitInput -> Output
    let outcome <- choose(p)
    return p

export flipped
"""
_DIRECT = _SOURCE.replace(
    "define choose(p : Real) : Bool !{random} =\n"
    '    perform random.sample[Bool](site("outcome"), Bernoulli(p))\n\n',
    "",
).replace("    let outcome <- choose(p)", "    sample outcome <- Bernoulli(probs=p)")


def _emitted(source):
    pyro = pytest.importorskip("pyro")
    parsed = parse(source)
    code = transpile(parsed, target="pyro")
    tree = ast.parse(code)
    assert (
        sum(
            isinstance(node, ast.FunctionDef)
            and node.name == "_qvr_bernoulli_log_score"
            for node in tree.body
        )
        == 1
    )
    assert all(
        not (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("quivers")
        )
        for node in ast.walk(tree)
    )
    namespace = {"pyro": pyro}
    exec(compile(tree, "<standalone Pyro>", "exec"), namespace)
    module = Compiler(parsed, module_name="pyro-bernoulli-score").qiec_module
    assert module is not None
    return pyro, namespace, module


def _trace_score(pyro, namespace, p, outcome):
    data = {"outcome": p.new_tensor(float(outcome))}
    model = pyro.poutine.condition(namespace["model"], data=data)
    trace = pyro.poutine.trace(model).get_trace(p)
    assert tuple(
        name for name, node in trace.nodes.items() if node["type"] == "sample"
    ) == ("outcome",)
    assert trace.nodes["outcome"]["value"].item() == outcome
    torch.testing.assert_close(trace.nodes["_RETURN"]["value"], p)
    score = trace.log_prob_sum()
    (gradient,) = torch.autograd.grad(score, p)
    return score, gradient


@pytest.mark.parametrize(
    "source", (_SOURCE, _DIRECT), ids=("qiec_helper", "program_plan")
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_emitted_probability_score_matches_native_at_closed_endpoints(source, dtype):
    pyro, namespace, module = _emitted(source)
    epsilon = torch.finfo(dtype).eps
    cases = (
        (0.0, False),
        (0.0, True),
        (epsilon / 256, False),
        (epsilon / 256, True),
        (1 - epsilon / 2, False),
        (1 - epsilon / 2, True),
        (1.0, False),
        (1.0, True),
    )
    for probability, outcome in cases:
        p = torch.tensor(probability, dtype=dtype, requires_grad=True)
        score, slope = _trace_score(pyro, namespace, p, outcome)
        if outcome:
            expected = (
                (math.log(probability), 1 / probability)
                if probability
                else (-math.inf, 0)
            )
        else:
            expected = (
                (math.log1p(-probability), -1 / (1 - probability))
                if probability < 1
                else (-math.inf, 0)
            )
        assert (score.item(), slope.item()) == pytest.approx(expected, rel=1e-6, abs=0)
        native = sample_program(
            module,
            "flipped",
            data={"p": p},
            sites={"outcome": outcome},
            distributions=TensorBackend(),
            numbers=NumberDomain.of(p),
        )
        (native_slope,) = torch.autograd.grad(native.log_joint, p)
        torch.testing.assert_close(score, native.log_joint)
        torch.testing.assert_close(slope, native_slope)

    p = torch.tensor(epsilon / 256, dtype=dtype, requires_grad=True)
    namespace["Bernoulli"] = pyro.distributions.Bernoulli
    clamped_score, clamped_slope = _trace_score(pyro, namespace, p, True)
    assert clamped_score.item() != pytest.approx(math.log(p.item()))
    assert clamped_slope.item() == 0


def test_emitted_bernoulli_keeps_parameterization_and_shape_after_expansion():
    pyro, namespace, _ = _emitted(_DIRECT)
    constructor = namespace["Bernoulli"]
    eps = torch.finfo(torch.float64).eps
    probs = torch.tensor([0.0, eps / 256, 1.0], dtype=torch.float64, requires_grad=True)
    initial = constructor(probs=probs)
    # Torch materializes both members lazily. Expansion must retain the
    # original coordinates even after this opposite projection is read.
    _ = initial.logits
    expanded = initial.expand((2, 3))
    assert expanded.batch_shape == (2, 3) and expanded.event_shape == ()
    sites = torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.float64)
    per_site = expanded.log_prob(sites)
    wanted = torch.tensor(
        [[0, math.log(eps / 256), 0], [-math.inf, math.log1p(-eps / 256), -math.inf]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(per_site, wanted)
    torch.testing.assert_close(expanded.to_event(1).log_prob(sites), wanted.sum(dim=1))
    slopes = torch.autograd.grad(per_site[0].sum(), probs)[0]
    torch.testing.assert_close(slopes, probs.new_tensor([-1, 256 / eps, 1]))
    draws = expanded.sample(torch.Size((4,)))
    assert draws.shape == (4, 2, 3)
    assert bool((draws[..., 0] == 0).all() and (draws[..., 2] == 1).all())
    assert bool(expanded.support.check(draws).all())

    logits = torch.tensor([-40.0, 0.9, 40.0], dtype=torch.float64, requires_grad=True)
    actual = constructor(logits=logits)
    _ = actual.probs
    got = actual.expand((2, 3)).log_prob(sites)
    want = pyro.distributions.Bernoulli(logits=logits.expand((2, 3))).log_prob(sites)
    torch.testing.assert_close(got, want)
    (left,) = torch.autograd.grad(got.sum(), logits)
    (right,) = torch.autograd.grad(want.sum(), logits)
    torch.testing.assert_close(left, right)


def test_emitted_logits_route_keeps_score_and_gradient():
    pyro, namespace, _ = _emitted(
        _DIRECT.replace("Bernoulli(probs=p)", "Bernoulli(logits=p)")
    )
    for value, outcome in ((-43.0, True), (43.0, False), (2.1, True)):
        coordinate = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        score, slope = _trace_score(pyro, namespace, coordinate, outcome)
        expected = torch.distributions.Bernoulli(logits=coordinate).log_prob(
            coordinate.new_tensor(float(outcome))
        )
        torch.testing.assert_close(score, expected)
        (expected_slope,) = torch.autograd.grad(expected, coordinate)
        torch.testing.assert_close(slope, expected_slope)
