"""Exact nested finite draws through the checked Random handler."""

from math import log, pi

import didactic.api as dx
import pytest
import torch
from torch import nn

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.qiec import (
    Enumeration,
    ExecutionFailure,
    QiecModule,
    RuntimeConfiguration,
    RuntimeSelection,
    dumps,
    loads,
    run_named,
)
from tests.qiec.test_structured_sampleables import SOURCE as STRUCTURED_SOURCE
from tests.qiec.test_structured_sampleables import _LawProvider, _law


_RUNTIME = RuntimeConfiguration(
    selections=(
        RuntimeSelection(
            provider="core", options={"handlers": {"exact": {"kind": "declared"}}}
        ),
    ),
)


_SOURCE = """\
instance random : Random
handler exact for Random : LogWeight -> LogWeight [coverage=total, implementation=foreign]
    sample resumes omega

define site_probability(x : Bool, q : Real) : Real !{} =
    if x then
        return q
    else
        return 1.0 - q

define joint(p : Real, q : Real) : LogWeight !{} =
    handle random with exact in
        let x <- perform random.sample[Bool](site("x"), Bernoulli(p))
        let probability <- site_probability(x, q)
        let y <- perform random.sample[Bool](site("y"), Bernoulli(probability))
        if x then
            return log_prob(Bernoulli(0.875), y)
        else
            return log_prob(Bernoulli(0.25), y)

define impossible(p : Real) : LogWeight !{} =
    handle random with exact in
        let x <- perform random.sample[Bool](site("x"), Bernoulli(p))
        return log_prob(Bernoulli(0.0), true)

define unreachable() : LogWeight !{} =
    handle random with exact in
        let x <- perform random.sample[Bool](site("x"), Bernoulli(0.0))
        if x then
            let y <- perform random.sample[Real](site("unsupported"), Normal(0.0, 1.0))
            return log_prob(Normal(0.0, 1.0), y)
        else
            return log_prob(Bernoulli(0.25), false)

define continuous() : LogWeight !{} =
    handle random with exact in
        let y <- perform random.sample[Real](site("unsupported"), Normal(0.0, 1.0))
        return log_prob(Normal(0.0, 1.0), y)
"""


def _module(source=_SOURCE):
    original = load_source(source).qiec
    assert isinstance(original, QiecModule)
    checked = original.with_(
        handlers=(
            original.handlers[0].with_(
                foreign=Enumeration(reduction="log_expectation")
            ),
        )
    )
    restored = loads(dumps(checked))
    assert isinstance(restored, QiecModule)
    assert restored == checked
    return restored


def test_nested_finite_draws_compute_joint_log_expectation_and_gradients():
    module = _module()
    p = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
    q = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    for probability in (0.3, 0.45, 0.8):
        with torch.no_grad():
            p.fill_(probability)
        result = run_named(
            module, "joint", (p, q), runtime=_RUNTIME, distributions=TensorBackend()
        ).value
        assert isinstance(result, torch.Tensor)

        branches = (
            (1 - p) * q * 0.75,
            (1 - p) * (1 - q) * 0.25,
            p * (1 - q) * 0.125,
            p * q * 0.875,
        )
        expected = torch.stack(branches).sum().log()
        torch.testing.assert_close(result, expected)
        gradients = torch.autograd.grad(result, (p, q))
        oracle = torch.autograd.grad(expected, (p, q))
        for actual, intended in zip(gradients, oracle, strict=True):
            torch.testing.assert_close(actual, intended)


def test_nested_finite_draws_share_the_execution_fuel_budget():
    with pytest.raises(ExecutionFailure) as error:
        run_named(_module(), "joint", (0.3, 0.7), runtime=_RUNTIME, fuel=12)
    assert error.value.diagnostic.code == "qiec-run-fuel"


def test_zero_scores_zero_support_and_nonfinite_draws_have_distinct_meanings():
    module = _module()
    p = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
    zero = run_named(
        module, "impossible", (p,), runtime=_RUNTIME, distributions=TensorBackend()
    ).value
    assert isinstance(zero, torch.Tensor) and torch.isneginf(zero)
    (gradient,) = torch.autograd.grad(zero, p)
    torch.testing.assert_close(gradient, p.new_zeros(()))

    alive = run_named(module, "unreachable", runtime=_RUNTIME).value
    assert isinstance(alive, torch.Tensor) and alive.exp().item() == pytest.approx(0.75)
    with pytest.raises(ExecutionFailure, match="finite support|enumerat"):
        run_named(module, "continuous", runtime=_RUNTIME)


def test_checked_log_expectation_requires_the_same_unbatched_scalar_answer():
    with pytest.raises(dx.ValidationError, match="ungrouped"):
        Enumeration(grouped=True, reduction="log_expectation")
    module = load_source(
        """
instance random : Random
handler exact for Random : (Bool * LogWeight) -> LogWeight [coverage=total, implementation=foreign]
    sample resumes omega
"""
    ).qiec
    assert isinstance(module, QiecModule)
    with pytest.raises(dx.ValidationError, match="requires a LogWeight -> LogWeight"):
        module.handlers[0].with_(foreign=Enumeration(reduction="log_expectation"))
    assert loads(dumps(Enumeration(grouped=True, reduction="table"))) == Enumeration(
        grouped=True, reduction="table"
    )


def test_nested_structured_draws_keep_nominal_values_and_weight_multiplicities():
    prelude = STRUCTURED_SOURCE.split("\nprogram inspect", 1)[0]
    module = _module(
        prelude
        + """
handler exact for Random : LogWeight -> LogWeight [coverage=total, implementation=foreign]
    sample resumes omega

define joint_structured(law : Sampleable[Meaning], response : Real) : LogWeight !{} =
    handle random with exact in
        let first <- encode_sample(law)
        let second <- encode_sample(law)
        return log_prob(Normal(first + second, 1.0), response)
"""
    )
    weights = nn.Parameter(
        torch.tensor([0.2, -0.3, 1.2, -torch.inf], dtype=torch.float64)
    )
    response = nn.Parameter(weights.new_tensor(2.5))
    law, _, _ = _law(module, weights)
    runtime = _RUNTIME.with_(providers=(_LawProvider(law),))
    actual = run_named(
        module,
        "joint_structured",
        (law, response),
        runtime=runtime,
        distributions=TensorBackend(),
    ).value
    assert isinstance(actual, torch.Tensor)
    masses = weights[:3].log_softmax(0)
    points = weights.new_tensor([0, 2, 2])
    means = points[:, None] + points[None, :]
    target = masses[:, None] + masses[None, :] - (response - means).square() / 2
    expected = torch.logsumexp(target.flatten(), dim=0) - log(2 * pi) / 2
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual, (weights, response))
    oracle = torch.autograd.grad(expected, (weights, response))
    for estimate, intended in zip(gradients, oracle, strict=True):
        torch.testing.assert_close(estimate, intended)
