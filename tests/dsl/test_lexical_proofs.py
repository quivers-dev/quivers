"""A QVR program receives exact, typed proof alternatives from one chart."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.laws import DIVISION
from quivers.dsl import loads as load_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.dsl.qiec_tooling import target_diagnostics
from quivers.qiec import deduction_outcomes
from quivers.qiec.checking import KernelError
from quivers.qiec.contracts import (
    Acyclic,
    Control,
    DerivativeSemantics,
    Exact,
    FailurePolicy,
    InadmissibleContract,
    NoRetention,
    Rational,
    ResourceBudget,
    WeightedProofRows,
    check_admissible,
    check_complexity,
)
from quivers.qiec.evaluator import Evaluator, RuntimeConstructor
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.foreign import DeductionProofs
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.program_runtime import run_program
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.types import BOOL, INT, IndexLiteral, TypeApplication
from quivers.stochastic import forest
from quivers.stochastic.planning import PlanRecord, PlanRefusal, PlanRefused, plan
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
    torch.testing.assert_close(
        torch.autograd.grad(raw.value, logs)[0], logs.exp(), atol=1e-12, rtol=1e-12
    )
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

        (report,) = [
            item for item in run.result.trace if item.event == "deduction.proofs.plan"
        ]
        assert event.sequence < report.sequence
        (handler,) = (
            item
            for item in restored.handlers
            if isinstance(item.foreign, DeductionProofs)
        )
        chart = handler.foreign.chart
        interface = next(
            effect for effect in restored.effects if effect.ref == handler.effect
        )
        (operation,) = interface.operations
        assert report.detail["handler"] == str(handler.id)
        assert report.detail["operation"] == str(operation.id)
        assert report.detail["presentation"] == str(chart.presentation.id)
        assert report.detail["proof_presentation"] == str(
            handler.foreign.proof_presentation.id
        )
        assert loads(report.detail["output_type"]) == handler.output_type
        assert (
            report.detail["temporary_provenance"]
            == "packed-forest-and-shared-derivations"
        )
        assert loads(report.detail["derivative"]) == DerivativeSemantics()
        record = loads(report.detail["plan"])
        assert isinstance(record, PlanRecord)
        assert dumps(record) == report.to_data()["detail"]["plan"]
        assert record.query.presentation == chart.presentation.id
        assert record.query.observable == WeightedProofRows(normalized=normalized)
        assert record.query.retention == record.retention == NoRetention()
        assert record.accuracy.controls == (
            Control(name="max_proofs", value=Rational(numerator=3)),
        )
        assert record.accuracy.budget == ResourceBudget()
        assert record.accuracy.guarantee == Exact()
        assert record.accuracy.on_failure == FailurePolicy.REFUSE
        assert record.accuracy.numerics.dtype == "float64"
        assert record.accuracy.numerics.domain == "log"
        assert record.accuracy.numerics.rounding is None
        assert record.query.input_types[-2:] == (INT, BOOL)
        assert record.algorithm == event.detail["algorithm"]
        assert record.claims == LOG_PROB.claims()
        assert record.recursion == Acyclic() and record.backward == "autograd"
        laws = {
            requirement.law
            for requirement in record.query.observable.requirements(record.algebra)
        }
        assert (DIVISION in laws) is normalized
        check_admissible(
            record.query,
            record.recursion,
            record.accuracy,
            algebra=record.algebra,
            claims=record.claims,
        )
        check_complexity(record.complexity)
        if normalized:
            with pytest.raises(InadmissibleContract, match="division"):
                check_admissible(
                    record.query,
                    record.recursion,
                    record.accuracy,
                    algebra=record.algebra,
                    claims=tuple(
                        claim for claim in record.claims if claim.law != DIVISION
                    ),
                )
        with pytest.raises(PlanRefused) as unsupported:
            plan(
                record.query,
                chart.presentation,
                LOG_PROB,
                record.recursion,
                record.accuracy,
                record.retention,
            )
        assert unsupported.value.code == PlanRefusal.OBSERVABLE_WITHOUT_EXECUTOR


@pytest.mark.parametrize("length", (3, 1024))
def test_qvr_materializes_one_budgeted_proof_with_exact_indices(length: int) -> None:
    module = load_source(
        """\
object Token : FinSet 2
object Result : Real 1

deduction Spine : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, S, word
    rule seed : span(0, J, A) |- span(0, J, S)
    rule join : span(0, K, S), span(K, J, A) |- span(0, J, S)
    lexicon
        "x" : A = word #[learnable]

program listed : Token -> Result
    let proofs = parse_proofs(Spine, sentence, 1)
    return proofs

export listed
"""
    ).qiec
    assert isinstance(module, QiecModule)
    descriptor = next(
        handler.foreign
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionProofs)
    )
    assert isinstance(descriptor, DeductionProofs)
    (proof_family,), constructors = descriptor.proof_presentation.declarations()
    by_name = {declaration.name: declaration.id for declaration in constructors}
    term = proof_family.indices[2]
    assert isinstance(term, IndexBinder)
    atom = IndexLiteral(value="A", sort=term.sort)
    start = IndexLiteral(value="S", sort=term.sort)
    boundaries = tuple(IndexLiteral(value=n, sort=NAT) for n in range(length + 1))

    log_weight = torch.tensor(-0.125, dtype=torch.float64, requires_grad=True)
    result = run_program(
        module,
        "listed",
        data={"sentence": ("x",) * length},
        sites={},
        parameters={"Spine.lex.0": log_weight},
    )
    rows = result.value
    assert isinstance(rows, RuntimeConstructor)
    assert rows.constructor == descriptor.chart.entry
    assert rows.static_arguments == (boundaries[0], boundaries[length], start)
    proof, mass, tail = rows.fields
    assert isinstance(tail, RuntimeConstructor)
    assert tail.constructor == descriptor.chart.empty
    assert tail.fields == ()

    # The seed is the first word. Each join appends exactly the next word
    # to that prefix. The bounded walk checks 2n proof nodes by occurrence
    # without comparing, printing or recursing over a whole subtree.
    nodes = 0
    for right in range(length, 0, -1):
        assert isinstance(proof, RuntimeConstructor)
        assert isinstance(proof.result_type, TypeApplication)
        assert proof.result_type.constructor == proof_family.type_constructor
        assert proof.result_type.arguments == (boundaries[0], boundaries[right], start)
        if right == 1:
            assert proof.constructor == by_name["Spine__rule_seed"]
            assert proof.static_arguments == (boundaries[1],)
            (word,) = proof.fields
        else:
            assert proof.constructor == by_name["Spine__rule_join"]
            assert proof.static_arguments == (boundaries[right - 1], boundaries[right])
            proof, word = proof.fields
        assert isinstance(word, RuntimeConstructor)
        assert word.constructor == by_name["Spine__axiom_span"]
        assert isinstance(word.result_type, TypeApplication)
        assert word.result_type.constructor == proof_family.type_constructor
        assert (
            word.result_type.arguments
            == word.static_arguments
            == (boundaries[right - 1], boundaries[right], atom)
        )
        assert word.fields == ("x",)
        nodes += 2
    assert nodes == 2 * length <= 2048

    # Run the same live parameter through every lexical occurrence.
    assert isinstance(mass, torch.Tensor)
    torch.testing.assert_close(mass, length * log_weight, rtol=0, atol=0)
    (gradient,) = torch.autograd.grad(mass, log_weight)
    torch.testing.assert_close(gradient, log_weight.new_tensor(length), rtol=0, atol=0)
    (event,) = [
        item for item in result.result.trace if item.event == "deduction.proofs"
    ]
    assert event.detail["proof_budget"] == 1
    assert event.detail["structural_goal_proofs"] == event.detail["positive_rows"] == 1
    assert (
        dict(event.detail["construction_work"])["provenance_records"] == 4 * length - 1
    )


def _branching_proofs(depth: int, cap_option: str) -> str:
    """Shared empty proofs, two full-span goals, and an impossible alternative."""
    categories = ", ".join(f"C{index}" for index in range(depth + 1))
    forks = "\n".join(
        f"    rule fork{index} : span(0, 0, C{index - 1}, lex), "
        f"span(0, 0, C{index - 1}, lex) |- span(0, 0, C{index}, lex)"
        for index in range(1, depth + 1)
    )
    return f"""\
object Token : FinSet 1
object Result : Real 1

deduction Forks : Token -> Token [semiring=LogProb, start=S{cap_option}]
    atoms span, {categories}, A, B, M, S, lex, on, off
    rule base : |- span(0, 0, C0, lex)
{forks}
    rule left : span(0, 0, C{depth}, lex), span(0, 1, A, lex) |- span(0, 1, S, on)
    rule right : span(0, 1, B, lex) |- span(0, 1, S, off)
    rule dead : span(0, 0, C{depth}, lex), span(0, 1, M, lex) |- span(0, 1, S, on)
    lexicon
        "x" : A = lex #[learnable]
        "x" : B = lex #[learnable]

program listed : Token -> Result
    let proofs = parse_normalized_proofs(Forks, sentence, 2)
    return proofs

export listed
"""


def test_authored_proof_node_cap_admits_exactly_bounded_rows_and_gradients() -> None:
    depth = 8
    # The on proof extends a branching empty proof by a rule and a word.
    # The off proof contains one rule and one word.
    on_nodes = (1 << (depth + 1)) + 1
    node_budget = on_nodes + 2
    checked = load_source(_branching_proofs(depth, f", proof_nodes={node_budget}"))
    module = loads(dumps(checked.qiec))
    assert isinstance(module, QiecModule) and module == checked.qiec
    parameters = torch.tensor([-0.125, -0.25], dtype=torch.float64, requires_grad=True)
    for _ in range(2):
        result = run_program(
            module,
            "listed",
            data={"sentence": ("x",)},
            sites={},
            parameters={
                f"Forks.lex.{index}": value
                for index, value in enumerate(parameters.unbind())
            },
        )
        rows = result.value
        measured = {}
        for _ in range(2):
            assert isinstance(rows, RuntimeConstructor)
            form = rows.static_arguments[-1]
            assert isinstance(form, IndexLiteral)
            proof, weight, rows = rows.fields
            assert isinstance(weight, torch.Tensor)
            todo = [proof]
            seen = set()
            while todo:
                node = todo.pop()
                assert isinstance(node, RuntimeConstructor) and id(node) not in seen
                seen.add(id(node))
                assert len(seen) <= node_budget
                todo.extend(
                    field
                    for field in node.fields
                    if isinstance(field, RuntimeConstructor)
                )
            assert form.value not in measured
            measured[form.value] = len(seen), weight
        assert isinstance(rows, RuntimeConstructor) and rows.fields == ()
        assert {form: count for form, (count, _) in measured.items()} == {
            "on": on_nodes,
            "off": 2,
        }
        actual = (0.8 * measured["on"][1].exp() + 0.2 * measured["off"][1].exp()).log()
        expected = (parameters.softmax(0) @ parameters.new_tensor([0.8, 0.2])).log()
        torch.testing.assert_close(actual, expected)
        (gradient,) = torch.autograd.grad(actual, parameters)
        torch.testing.assert_close(
            gradient, torch.autograd.grad(expected, parameters)[0]
        )
        with torch.no_grad():
            parameters.add_(0.25 * gradient)
        (event,) = [
            item for item in result.result.trace if item.event == "deduction.proofs"
        ]
        assert event.detail["proof_node_budget"] == node_budget
        assert (
            event.detail["proof_budget"] == event.detail["structural_goal_proofs"] == 2
        )
        (plan_event,) = [
            item
            for item in result.result.trace
            if item.event == "deduction.proofs.plan"
        ]
        record = loads(plan_event.detail["plan"])
        assert record.accuracy.controls == (
            Control(name="max_proofs", value=Rational(numerator=2)),
            Control(name="proof_nodes", value=Rational(numerator=on_nodes + 2)),
        )
        assert record.accuracy.budget == ResourceBudget()
        assert record.query.observable == WeightedProofRows(normalized=True)
        assert dict(plan_event.detail["span_order"])["nullable"] is True


@pytest.mark.parametrize("zero_weight", (False, True))
def test_authored_proof_node_refusal_creates_no_proof_or_weight_work(
    monkeypatch: pytest.MonkeyPatch, zero_weight: bool
) -> None:
    cap = 514
    module = load_source(_branching_proofs(8, f", proof_nodes={cap}")).qiec
    assert isinstance(module, QiecModule)
    descriptor = next(
        handler.foreign
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionProofs)
    )
    proof_ids = descriptor.proof_presentation.constructors_by_id.keys()

    def forbidden(*args, **kwargs):
        pytest.fail("proof construction or weight evaluation preceded the node check")

    materialize = Evaluator._constructor_value

    def check_materialization(self, value, environment):
        assert value.constructor not in proof_ids
        return materialize(self, value, environment)

    monkeypatch.setattr(forest, "Derivation", forbidden)
    monkeypatch.setattr(forest, "edge_weight", forbidden)
    monkeypatch.setattr(deduction_outcomes, "closed_derivation_value", forbidden)
    monkeypatch.setattr(Evaluator, "_constructor_value", check_materialization)
    observed = []
    with pytest.raises(
        ExecutionFailure, match="too-many-proof-nodes.*515.*node limit of 514"
    ):
        run_program(
            module,
            "listed",
            data={"sentence": ("x",)},
            sites={},
            parameters={"Forks.lex.0": -torch.inf if zero_weight else -0.125},
            observer=observed.append,
        )
    assert all(
        item.event not in {"deduction.proofs", "deduction.proofs.plan"}
        for item in observed
    )
    (event,) = [item for item in observed if item.event == "deduction.proofs.refused"]
    assert event.detail["reason"] == "too-many-proof-nodes"
    assert event.detail["proof_node_budget"] == cap
    assert event.detail["minimum_structural_proof_nodes"] == 515


@pytest.mark.parametrize(
    "option", ("0", "1.5", "inf", "9007199254740992", "2, proof_nodes=3")
)
def test_proof_node_cap_refuses_invalid_options(option: str) -> None:
    with pytest.raises(
        QiecDiagnosticError, match="proof_nodes must be one positive integer"
    ):
        lower_qvr_to_qiec(parse(_branching_proofs(1, f", proof_nodes={option}")))


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
    assert all(
        event.event not in {"deduction.proofs", "deduction.proofs.plan"}
        for event in observed
    )
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


@pytest.mark.parametrize("cap_option", ("", ", proof_nodes=10"))
def test_proof_target_refusal_is_reached_only_by_its_own_entry(cap_option: str) -> None:
    source = (
        GRAMMAR.replace("start=S]", f"start=S{cap_option}]")
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
