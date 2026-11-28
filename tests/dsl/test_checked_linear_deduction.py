"""The checked lexical query runs certified closure instead of settlement."""

from __future__ import annotations

from math import log
from pathlib import Path

import didactic.api as dx
import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.dsl.qiec_tooling import target_diagnostics
from quivers.qiec.checking import KernelError
from quivers.qiec.contracts import (
    SPECTRAL_RADIUS_BELOW_ONE,
    Control,
    DerivativeSemantics,
    Exact,
    LinearCyclic,
    PresentationId,
    Rational,
)
from quivers.qiec.execution import ExecutionFailure, run_named
from quivers.qiec.foreign import DeductionLinearGoal
from quivers.qiec.identifiers import HandlerId
from quivers.qiec.kinds import NAT
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.numbers import NumberDomain
from quivers.qiec.program_runtime import run_program
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.types import IndexLiteral
from quivers.stochastic.planning import SPAN_ALGORITHM
from quivers.transpile import UnsupportedConstruct, transpile


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "docs/examples/qiec/checked-cyclic-deduction.qvr"
).read_text()
CALL = """\

program exact : Token -> Token
    let value = parse_linear(Linear, sentence, 64)
    return value
"""
PARALLEL = SOURCE.replace(
    "    lexicon",
    "    rule second : span(I, J, S) |- span(I, J, S) #[learnable]\n    lexicon",
    1,
)


def _module(source: str) -> QiecModule:
    compiled = load_source(source)
    encoded = dumps(compiled.qiec)
    module = loads(encoded)
    assert isinstance(module, QiecModule)
    assert module == compiled.qiec and dumps(module) == encoded
    return module


def _parameters(weights: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"Linear.lex.0": weights[0], "Linear.rule.step:": weights[1]}


def _query(
    module: QiecModule, sentence: tuple[str, ...], weights: torch.Tensor
) -> torch.Tensor:
    result = run_program(
        module,
        "exact",
        data={"sentence": sentence},
        sites={},
        parameters=_parameters(weights),
        numbers=NumberDomain.of(weights),
    ).value
    assert isinstance(result, torch.Tensor)
    return result


def test_nominal_plan_runs_to_infinite_sum_and_differentiates_finite_closure(
    monkeypatch,
) -> None:
    module = _module(SOURCE + CALL)
    (semantics,) = tuple(
        handler.foreign
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionLinearGoal)
    )
    selected = semantics.plan
    assert selected.presentation.id == PresentationId.derive("Linear")
    assert selected.record.algorithm == SPAN_ALGORITHM
    assert selected.record.recursion == LinearCyclic()
    assert selected.record.accuracy.guarantee == Exact()
    execute = type(selected).run

    def observe(plan, axioms, inputs, *, like):
        assert plan.record.accuracy.controls == (
            Control(name="max_closure_size", value=Rational.of(64)),
        )
        assert plan.record.accuracy.gradient == DerivativeSemantics.implicit(
            SPECTRAL_RADIUS_BELOW_ONE
        )
        assert plan.record.query.differentiate_with_respect_to == tuple(inputs)
        assert len(inputs) == 2
        return execute(plan, axioms, inputs, like=like)

    monkeypatch.setattr(type(selected), "run", observe)
    weights = torch.tensor([0.3, 0.4], dtype=torch.float64).log().requires_grad_()
    value = _query(module, ("seed",), weights)
    lexical, loop = weights.exp().unbind()
    geometric = (lexical / (1 - loop)).log()
    # Derivation h applies the only unary rule h times to the lexical axiom.
    unfolded = sum(lexical * loop**h for h in range(80)).log()
    stopped = sum(lexical * loop**h for h in range(3)).log()
    torch.testing.assert_close(value, geometric)
    torch.testing.assert_close(value, unfolded)
    assert not torch.isclose(value, stopped)
    gradient = torch.autograd.grad(value, weights)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(unfolded, weights)[0])
    torch.testing.assert_close(gradient, torch.tensor([1, 2 / 3], dtype=weights.dtype))


@pytest.mark.parametrize("tokens", [(), ("unknown",), ("seed", "unknown")])
def test_absent_goal_is_structural_zero_with_zero_gradient(tokens) -> None:
    module = _module(SOURCE + CALL)
    weights = torch.tensor([0.3, 0.4], dtype=torch.float64).log().requires_grad_()
    value = _query(module, tokens, weights)
    assert torch.isneginf(value)
    torch.testing.assert_close(
        torch.autograd.grad(value, weights)[0], torch.zeros_like(weights)
    )


def test_zero_axiom_mass_stays_zero_under_unary_closure() -> None:
    module = _module(SOURCE + CALL)
    weights = torch.tensor(
        [-float("inf"), log(0.4)], dtype=torch.float64, requires_grad=True
    )
    value = _query(module, ("seed",), weights)
    assert torch.isneginf(value)
    torch.testing.assert_close(
        torch.autograd.grad(value, weights)[0], torch.zeros_like(weights)
    )


def test_absent_start_category_sums_every_full_span_mass() -> None:
    module = _module(
        """\
object Token : FinSet 1

deduction Both : Token -> Token [semiring=LogProb]
    atoms span, A, B, seed
    rule to_b : span(I, J, A) |- span(I, J, B) #[learnable]
    rule to_a : span(I, J, B) |- span(I, J, A) #[learnable]
    lexicon
        "seed" : A = seed #[learnable]

program exact : Token -> Token
    let value = parse_linear(Both, sentence, 64)
    return value
"""
    )
    weights = torch.tensor([0.3, 0.25, 0.4], dtype=torch.float64).log().requires_grad_()
    names = ("Both.lex.0", "Both.rule.to_b:", "Both.rule.to_a:")
    value = run_program(
        module,
        "exact",
        data={"sentence": ("seed",)},
        sites={},
        parameters=dict(zip(names, weights.unbind(), strict=True)),
    ).value
    a, p, q = weights.exp().unbind()
    # Even paths end at A and odd paths end at B. Each complete turn
    # contributes p q, and each odd path also contributes one p.
    expected = (a * (1 + p) / (1 - p * q)).log()
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(
        torch.autograd.grad(value, weights)[0],
        torch.autograd.grad(expected, weights)[0],
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_binary_schedule_uses_two_independent_lexical_and_unary_choices(dtype) -> None:
    module = _module(
        """\
object Token : FinSet 1

deduction Width : Token -> Token [semiring=LogProb, start=S]
    atoms span, S, A, first, second
    rule spin : span(I, J, A) |- span(I, J, A) #[learnable]
    rule pair : span(I, K, A), span(K, J, A) |- span(I, J, S) #[learnable]
    lexicon
        "seed" : A = first #[learnable]
        "seed" : A = second #[learnable]

program exact : Token -> Token
    let value = parse_linear(Width, sentence, 64)
    return value
"""
    )
    weights = torch.tensor([0.2, 0.3, 0.4, 0.75], dtype=dtype).log().requires_grad_()
    names = ("Width.lex.0", "Width.lex.1", "Width.rule.spin:", "Width.rule.pair:")
    value = run_program(
        module,
        "exact",
        data={"sentence": ("seed", "seed")},
        sites={},
        parameters=dict(zip(names, weights.unbind(), strict=True)),
        numbers=NumberDomain.of(weights),
    ).value
    assert isinstance(value, torch.Tensor) and value.dtype == dtype
    a, b, r, rule = weights.exp().unbind()
    # One binary root. Its left and right premises independently choose
    # one of two lexical leaves and one of all possible unary path lengths.
    expected = (rule * ((a + b) / (1 - r)) ** 2).log()
    torch.testing.assert_close(value, expected)
    gradient = torch.autograd.grad(value, weights)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(expected, weights)[0])
    torch.testing.assert_close(
        gradient, torch.tensor([0.8, 1.2, 4 / 3, 1], dtype=dtype)
    )
    assert not torch.isclose(value, (rule * (max(a, b) / (1 - r)) ** 2).log())


def test_ordinary_entry_keeps_existing_bounded_rule_reader() -> None:
    source = SOURCE.replace("#[learnable]", "#[learnable, bounded]", 1)
    source += """\

define tokens() : Tensor[String]([1]) !{} =
    return ["seed"]
"""
    source += CALL.replace(
        "    let value",
        "    let sentence <- tokens()\n    let maximum = 64\n    let value",
    ).replace("sentence, 64", "sentence, maximum")
    result = load_source(source).run("exact")
    assert isinstance(result.value, torch.Tensor)
    torch.testing.assert_close(result.value, result.value.new_tensor(log(2.0)))


def test_declared_divergence_and_unestablished_convergence_are_distinct() -> None:
    module = _module(SOURCE + CALL)
    critical = torch.tensor([log(0.3), 0.0], dtype=torch.float64)
    with pytest.raises(ExecutionFailure, match="linear deduction goal diverges"):
        _query(module, ("seed",), critical)

    module = _module(PARALLEL + CALL)
    with pytest.raises(
        ExecutionFailure, match="cannot decide whether the closure converges"
    ):
        run_program(
            module,
            "exact",
            data={"sentence": ("seed",)},
            sites={},
            parameters={
                "Linear.lex.0": log(0.3),
                "Linear.rule.step:": -log(2),
                "Linear.rule.second:": -log(2),
            },
        )
    low_precision = torch.tensor(
        [log(0.3), -0.6931478977203369, -0.6931464672088623], dtype=torch.float32
    )
    with pytest.raises(ExecutionFailure, match="working precision cannot evaluate"):
        run_program(
            module,
            "exact",
            data={"sentence": ("seed",)},
            sites={},
            parameters=dict(
                zip(
                    ("Linear.lex.0", "Linear.rule.step:", "Linear.rule.second:"),
                    low_precision.unbind(),
                    strict=True,
                )
            ),
            numbers=NumberDomain.of(low_precision),
        )


def test_declared_cap_applies_to_augmented_order_before_star() -> None:
    nominal = _module(PARALLEL + CALL)
    (descriptor,) = tuple(
        handler.foreign
        for handler in nominal.handlers
        if isinstance(handler.foreign, DeductionLinearGoal)
    )
    assert descriptor.plan.span is not None
    states = len(descriptor.plan.span.categories)
    parameters = {
        "Linear.lex.0": log(0.3),
        "Linear.rule.step:": log(0.16),
        "Linear.rule.second:": log(0.24),
    }
    for limit, message in (
        (0, "must be positive"),
        (states, f"requires order {states + 1}, above the declared maximum {states}"),
    ):
        module = _module(PARALLEL + CALL.replace(", 64)", f", {limit})"))
        with pytest.raises(ExecutionFailure, match=message):
            run_program(
                module,
                "exact",
                data={"sentence": ("seed",)},
                sites={},
                parameters=parameters,
            )
    module = _module(PARALLEL + CALL.replace(", 64)", f", {states + 1})"))
    value = run_program(
        module, "exact", data={"sentence": ("seed",)}, sites={}, parameters=parameters
    ).value
    torch.testing.assert_close(value, torch.tensor(log(0.5), dtype=torch.float64))


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (torch.tensor([0.0, float("inf")], dtype=torch.float64), "finite chart range"),
        (torch.tensor([0.0, float("nan")], dtype=torch.float64), "finite chart range"),
        (
            torch.tensor([torch.finfo(torch.float64).max, -1.0], dtype=torch.float64),
            "finite chart range",
        ),
        (torch.tensor([0.0, -1.0], dtype=torch.float16), "float64 or float32"),
    ],
)
def test_closure_refuses_unsupported_numeric_domains(weights, message) -> None:
    module = _module(SOURCE + CALL)
    with pytest.raises(ExecutionFailure, match=message):
        _query(module, ("seed",), weights)


def test_codec_rechecks_weight_order_and_nominal_attachment() -> None:
    module = _module(SOURCE + CALL)
    (handler,) = tuple(
        handler
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionLinearGoal)
    )
    semantics = handler.foreign
    assert isinstance(semantics, DeductionLinearGoal)
    with pytest.raises((ValueError, dx.ValidationError), match="weight addresses"):
        semantics.with_(parameters=())
    with pytest.raises((ValueError, dx.ValidationError), match="lexical categories"):
        semantics.with_(lexicon=(("seed", "undeclared"),))
    forged = handler.with_(name="Different__query_linear_goal")
    handlers = tuple(forged if entry == handler else entry for entry in module.handlers)
    with pytest.raises(KernelError, match="different nominal recurrence"):
        validate_module(module.with_(handlers=handlers))


def test_raw_entry_requires_the_declared_runtime_attachment() -> None:
    module = _module(SOURCE.replace(" #[learnable]", "") + CALL)
    with pytest.raises(
        ExecutionFailure, match="no runtime clauses are attached"
    ) as refusal:
        run_named(
            module,
            "Linear__linear_goal",
            (("seed",), 64),
            static_arguments=(IndexLiteral(value=1, sort=NAT),),
        )
    assert str(
        HandlerId.derive(module.module, "handler", "Linear__query_linear_goal")
    ) in str(refusal.value)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            SOURCE + CALL.replace("Linear, sentence", "Branching, sentence"),
            "concatenate two adjacent spans",
        ),
        (
            SOURCE.replace("span(I, J, S)", "span(I, J, X)", 2) + CALL,
            "binding-independent",
        ),
        (
            SOURCE + CALL.replace("Linear, sentence", 'Linear, "seed"'),
            "token tensor binding",
        ),
        (
            SOURCE + CALL.replace("Linear, sentence", "Unknown, sentence"),
            "declared lexical deduction",
        ),
        (
            SOURCE
            + CALL.replace("    let value", "    let sentence = 1\n    let value"),
            "rank-one Tensor",
        ),
        (SOURCE + CALL.replace(", 64)", ")"), "maximum closure size of type Int"),
        (
            SOURCE + CALL.replace(", 64)", ", 2.5)"),
            "maximum closure size must have type Int",
        ),
    ],
)
def test_source_refuses_queries_outside_the_linear_fragment(source, message) -> None:
    with pytest.raises(QiecDiagnosticError, match=message):
        load_source(source)


def test_target_refusal_is_located_and_unrelated_export_remains_available() -> None:
    source = (
        SOURCE
        + CALL
        + """\

program independent : Token -> Token
    sample z <- Normal(loc=0.0, scale=1.0)
    return z
"""
    )
    active_source = source + "\nexport exact\n"
    active = parse(active_source, file_path="linear_goal.qvr")
    checked = lower_qvr_to_qiec(active, file_path="linear_goal.qvr")
    (diagnostic,) = target_diagnostics(checked, active, "stan")
    assert diagnostic.code == "qiec:capability:deduction-linear-goal:exact"
    assert "parse_linear(" in active_source.splitlines()[diagnostic.line - 1]
    with pytest.raises(UnsupportedConstruct, match="deduction-linear-goal"):
        transpile(active, target="stan")

    independent = parse(source + "\nexport independent\n", file_path="linear_goal.qvr")
    checked = lower_qvr_to_qiec(independent, file_path="linear_goal.qvr")
    assert not target_diagnostics(checked, independent, "stan")
    assert b"normal" in transpile(independent, target="stan")
