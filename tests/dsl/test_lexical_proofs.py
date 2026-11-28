"""A QVR program receives exact, typed proof alternatives from one chart."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.dsl.qiec_tooling import target_diagnostics
from quivers.qiec import deduction_outcomes
from quivers.qiec.checking import KernelError
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.foreign import DeductionProofs
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.program_runtime import run_program
from quivers.qiec.serialization import dumps, loads
from quivers.transpile import UnsupportedConstruct, transpile


GRAMMAR = """\
object Token : FinSet 2
object Result : Real 1

deduction Mix : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, C, D, S, same, other, lex
    rule first : span(I, K, A, lex), span(K, J, D, lex) |- span(I, J, S, same)
    rule second : span(I, K, B, lex), span(K, J, D, lex) |- span(I, J, S, same)
    rule third : span(I, K, C, lex), span(K, J, D, lex) |- span(I, J, S, other)
    lexicon
        "x" : A = lex #[learnable]
        "x" : B = lex #[learnable]
        "x" : C = lex #[learnable]
        "y" : D = lex
"""

QUERY = """\
define pack(first : String, last : String) : Tensor[String]([2]) !{} =
    return [first, last]

define rate[i : Nat, j : Nat, c : Mix__Term, f : Mix__Term](proof : Mix__SpanProof(i, j, c, f), a : Real, b : Real, d : Real) : Real !{} =
    case proof motive (x : Nat, y : Nat, z : Mix__Term, m : Mix__Term) => Real
        Mix__rule_first[l, k, r](left, right) =>
            return a * a
        Mix__rule_second[l, k, r](left, right) =>
            return b
        Mix__rule_third[l, k, r](left, right) =>
            return d
        Mix__axiom_span[l, r, category, form](word) =>
            return d

define response(rows : Mix__Proofs, a : Real, b : Real, d : Real) : Real !{} =
    case rows motive => Real
        Mix__proof_end =>
            return 0.0
        Mix__proof_entry[i, j, c, f](proof, weight, tail) =>
            let value <- rate[i, j, c, f](proof, a, b, d)
            let rest <- response(tail, a, b, d)
            return exp(weight) * value + rest

program assess(first : String, last : String, limit : Int) : Token -> Result
    let sentence <- pack(first, last)
    let proofs = parse_normalized_proofs(Mix, sentence, limit)
    let answer <- response(proofs, a, b, d)
    score likelihood = log(answer)
    return likelihood

program raw : Token -> Result
    let proofs = parse_proofs(Mix, sentence, 3)
    let answer <- response(proofs, 1.0, 1.0, 1.0)
    return answer

export assess
export raw
"""


def test_qvr_scores_distinct_proofs_of_the_same_goal_and_preserves_gradients() -> None:
    compiled = load_source(GRAMMAR + "\n" + QUERY)
    data = {"first": "x", "last": "y", "limit": 3, "a": 0.8, "b": 0.3, "d": 0.1}
    result = compiled.run("assess", data=data)
    assert result.log_joint == pytest.approx(math.log((0.8**2 + 0.3 + 0.1) / 3))
    names = {"pack", "Mix__proofs", "rate"}
    call_order = [
        event.detail["computation"]
        for event in result.result.trace
        if event.event == "call.entered" and event.detail["computation"] in names
    ]
    assert call_order == ["pack", "Mix__proofs", "rate", "rate", "rate"]

    module = compiled.qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule) and restored == module
    logs = torch.tensor([-0.1, -1.3, -0.4], dtype=torch.float64, requires_grad=True)
    rates = torch.tensor([0.8, 0.3, 0.1], dtype=logs.dtype, requires_grad=True)
    parameters = {
        f"Mix.lex.{index}": weight for index, weight in enumerate(logs.unbind())
    }
    scored = run_program(
        restored,
        "assess",
        data={**data, **dict(zip(("a", "b", "d"), rates.unbind(), strict=True))},
        sites={},
        parameters=parameters,
    )
    # The first two rows prove exactly the same span and logical form but
    # their root constructors, and thus their response likelihoods, differ.
    proof_rates = torch.stack((rates[0].square(), rates[1], rates[2]))
    expected = (logs + proof_rates.log()).logsumexp(0) - logs.logsumexp(0)
    torch.testing.assert_close(scored.log_joint, expected, atol=1e-12, rtol=1e-12)
    found = torch.autograd.grad(scored.log_joint, (logs, rates))
    reference = torch.autograd.grad(expected, (logs, rates))
    for actual, expected_gradient in zip(found, reference, strict=True):
        torch.testing.assert_close(actual, expected_gradient, atol=1e-12, rtol=1e-12)
    raw = run_program(
        restored, "raw", data={"sentence": ("x", "y")}, sites={}, parameters=parameters
    )
    assert isinstance(raw.value, torch.Tensor)
    torch.testing.assert_close(raw.value, logs.exp().sum(), atol=1e-12, rtol=1e-12)
    for run, normalized in ((scored, True), (raw, False)):
        (event,) = [
            event for event in run.result.trace if event.event == "deduction.proofs"
        ]
        assert event.detail["observer"] == "weighted-proof-rows"
        assert event.detail["normalized"] is normalized
        assert event.detail["algorithm"] == "indexed-packed-forest-budgeted-enumeration"
        assert event.detail["proof_budget"] == 3
        assert event.detail["goal_items"] == 2
        assert (
            event.detail["structural_goal_proofs"] == event.detail["positive_rows"] == 3
        )
        assert dict(event.detail["construction_work"])["provenance_records"] == 13
        assert "reified_proof_nodes" in event.detail["unmeasured"]


def test_proof_report_distinguishes_structural_budget_from_positive_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_source(GRAMMAR + "\n" + QUERY).qiec
    assert isinstance(module, QiecModule)
    logs = torch.tensor(
        [-0.1, -torch.inf, -0.4], dtype=torch.float64, requires_grad=True
    )
    parameters = {
        f"Mix.lex.{index}": weight for index, weight in enumerate(logs.unbind())
    }
    data = {"first": "x", "last": "y", "limit": 3, "a": 0.8, "b": 0.3, "d": 0.1}
    result = run_program(module, "assess", data=data, sites={}, parameters=parameters)
    expected = (logs.softmax(0) @ logs.new_tensor([0.8**2, 0.3, 0.1])).log()
    torch.testing.assert_close(result.log_joint, expected)
    torch.testing.assert_close(
        torch.autograd.grad(result.log_joint, logs)[0],
        torch.autograd.grad(expected, logs)[0],
    )
    (accepted,) = [
        event.detail
        for event in result.result.trace
        if event.event == "deduction.proofs"
    ]
    assert accepted["structural_goal_proofs"] == 3 and accepted["positive_rows"] == 2

    def forbidden(*args, **kwargs):
        pytest.fail("the rejected proof query reified a row")

    monkeypatch.setattr(deduction_outcomes, "closed_derivation_value", forbidden)
    observed = []
    with pytest.raises(ExecutionFailure, match="too-many-derivations"):
        run_program(
            module,
            "assess",
            data={**data, "limit": 2},
            sites={},
            parameters=parameters,
            observer=observed.append,
        )
    assert all(event.event != "deduction.proofs" for event in observed)
    (refused,) = [
        event.detail for event in observed if event.event == "deduction.proofs.refused"
    ]
    assert refused["reason"] == "too-many-derivations" and refused["proof_budget"] == 2
    assert refused["goal_items"] == accepted["goal_items"]
    assert "structural_goal_proofs" not in refused


def test_proof_query_refuses_over_budget_before_reifying_any_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = load_source(GRAMMAR + "\n" + QUERY)
    data = {"first": "x", "last": "y", "limit": 2, "a": 0.8, "b": 0.3, "d": 0.1}

    def forbidden(*args, **kwargs):
        pytest.fail("a proof was reified before the query's budget check")

    monkeypatch.setattr(deduction_outcomes, "closed_derivation_value", forbidden)
    with pytest.raises(ExecutionFailure, match="too-many-derivations"):
        compiled.run("assess", data=data)
    with pytest.raises(ExecutionFailure, match="proof budgets must be positive"):
        compiled.run("assess", data={**data, "limit": 0})
    assert compiled.run("raw", data={"sentence": ("missing", "y")}).value == 0.0
    with pytest.raises(ExecutionFailure, match="without positive goal mass"):
        compiled.run("assess", data={**data, "first": "missing"})


def test_proof_query_keeps_gradients_to_program_supplied_production_weights() -> None:
    normalized = """\
object Token : FinSet 2
object Result : Real 1

deduction Mix : Token -> Token [semiring=LogProb, normalize=lhs]
    atoms span, A, B, C, D, S, T, lex
    rule first : span(I, K, A), span(K, J, D) |- span(I, J, S)
    rule second : span(I, K, B), span(K, J, D) |- span(I, J, S)
    rule third : span(I, K, C), span(K, J, D) |- span(I, J, T)
    lexicon
        "x" : A = lex
        "x" : B = lex
        "x" : C = lex
        "y" : D = lex
"""
    query = (
        QUERY.replace("c : Mix__Term, f : Mix__Term", "c : Mix__Term")
        .replace("i, j, c, f", "i, j, c")
        .replace("z : Mix__Term, m : Mix__Term", "z : Mix__Term")
        .replace("l, r, category, form", "l, r, category")
        .replace("limit : Int) : Token", "limit : Int, p : Tensor[Real]([2])) : Token")
        .replace(
            "parse_normalized_proofs(Mix, sentence, limit)",
            'parse_normalized_proofs(Mix, sentence, limit, ("S", p))',
        )
    )
    compiled = load_source(normalized + "\n" + query)
    p = torch.tensor([0.2, 0.4], dtype=torch.float64, requires_grad=True)
    data = {"first": "x", "last": "y", "limit": 3, "a": 0.8, "b": 0.3, "d": 0.1, "p": p}
    actual = compiled.run("assess", data=data).log_joint
    # The S-family has total weight 1 after its production normalization.
    # The sole T rule keeps weight 1, making the goal partition 2.
    numerator = p @ p.new_tensor([0.8**2, 0.3]) / p.sum() + 0.1
    expected = (numerator / 2).log()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual, p)[0],
        torch.autograd.grad(expected, p)[0],
        atol=1e-12,
        rtol=1e-12,
    )


def test_proof_query_requires_an_integer_budget_and_its_public_proof_type() -> None:
    wrong_input = GRAMMAR + QUERY.replace("limit : Int", "limit : Real")
    with pytest.raises(QiecDiagnosticError, match="proof budgets must have type Int"):
        load_source(wrong_input)
    wrong_literal = GRAMMAR + QUERY.replace("sentence, 3)", "sentence, 3.0)")
    with pytest.raises(QiecDiagnosticError, match="proof budgets must have type Int"):
        load_source(wrong_literal)

    module = load_source(GRAMMAR + "\n" + QUERY).qiec
    assert isinstance(module, QiecModule)
    descriptor = next(
        handler.foreign
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionProofs)
    )
    assert isinstance(descriptor, DeductionProofs)
    (proof_family,), _ = descriptor.proof_presentation.declarations()
    families = tuple(
        family.with_(name="IncorrectProof") if family.id == proof_family.id else family
        for family in module.families
    )
    with pytest.raises(KernelError, match="must expose its checked proof declarations"):
        validate_module(module.with_(families=families))


def test_two_lexical_proof_queries_use_distinct_visible_index_sorts() -> None:
    other = (
        GRAMMAR[GRAMMAR.index("deduction Mix") :]
        .replace("Mix", "Other")
        .replace("same", "another")
    )
    source = (
        GRAMMAR
        + other
        + QUERY
        + """
index Term = Plain

program two : Token -> Result
    let first = parse_proofs(Mix, sentence, 3)
    let second = parse_proofs(Other, sentence, 3)
    let grouped = parse_outcomes(Mix, sentence, "form")
    let value = 1.0
    return value

export two
"""
    )
    compiled = load_source(source)
    assert compiled.run("two", data={"sentence": ("x", "y")}).value == 1.0
    assert isinstance(compiled.qiec, QiecModule)
    sorts = {sort.name: sort for sort in compiled.qiec.index_sorts}
    assert sorts["Term"].constructors == ("Plain",)
    assert "same" in sorts["Mix__Term"].constructors
    assert "another" in sorts["Other__Term"].constructors


def test_proof_target_refusal_is_reached_only_by_its_own_entry() -> None:
    source = (
        GRAMMAR
        + QUERY[: QUERY.index("export assess")]
        + """
program independent : Token -> Result
    sample x <- Normal(loc=0.0, scale=1.0)
    return x
"""
    )
    active_source = source + "\nexport assess\n"
    active = parse(active_source)
    lowered = lower_qvr_to_qiec(active)
    diagnostic = next(
        item
        for item in target_diagnostics(lowered, active, "stan")
        if item.code == "qiec:capability:deduction-outcomes:assess"
    )
    assert "parse_normalized_proofs(" in active_source.splitlines()[diagnostic.line - 1]
    with pytest.raises(UnsupportedConstruct, match="deduction-outcomes"):
        transpile(active, target="stan")
    independent = parse(source + "\nexport independent\n")
    assert b"normal" in transpile(independent, target="stan")
