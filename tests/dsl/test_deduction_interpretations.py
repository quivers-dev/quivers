"""Authored rule maps elaborate to occurrence-sensitive checked proof folds."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_tooling import qiec_binding_map
from quivers.modeling import Model
from quivers.qiec import (
    BOOL,
    ClosedInterpretationBinding,
    ComputationId,
    EffectRow,
    LiteralValue,
    ParameterId,
    PresentationId,
    Return,
    RuleError,
    validate_module,
)
from quivers.qiec.execution import ExecutionFailure, run_named
from quivers.qiec.foreign import DeductionOutcomes, DeductionProofs
from quivers.qiec.rules import RulePresentation
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import run_program
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.terms import Case
from quivers.qiec.types import IndexLiteral, TypeApplication


GRAMMAR = """\
object Token : FinSet 2
object Result : Real 1

deduction Mix : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, D, S, same, lex
    rule first : span(I, K, A, lex), span(K, J, D, lex) |- span(I, J, S, same)
    rule second : span(I, K, B, lex), span(K, J, D, lex) |- span(I, J, S, same)
    lexicon
        "x" : A = lex #[learnable]
        "x" : B = lex #[learnable]
        "y" : D = lex

#! An occurrence-sensitive likelihood.
interpret rating : Mix -> Real
    rule first = square_left
    rule second = square_right
    axiom span = read_word

interpret evidence : Mix -> Bool
    rule first = left_truth
    rule second = right_truth
    axiom span = is_x

effect Inspection
    advance : Real -> Real

instance audit : Inspection

handler shift for Inspection : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    advance(value : Real) resumes 1 =>
        resume(value + 1.0)

define read_word(word : String) : Real !{audit} =
    if word == "x" then
        perform audit.advance(1.0)
    else
        perform audit.advance(2.0)

define square_left(left : Real, right : Real) : Real !{} =
    return (left * left + right) / 100.0

define square_right(left : Real, right : Real) : Real !{} =
    return (right * right + left) / 100.0

define is_x(word : String) : Bool !{} =
    return word == "x"

define left_truth(left : Bool, right : Bool) : Bool !{} =
    return left

define right_truth(left : Bool, right : Bool) : Bool !{} =
    return right

define elicit(rows : Mix__Proofs) : Real !{audit} =
    case rows motive => Real
        Mix__proof_end =>
            return 0.0
        Mix__proof_entry[i, j, category, form](proof, weight, tail) =>
            let likelihood <- rating[i, j, category, form](proof)
            let accepted <- evidence[i, j, category, form](proof)
            let rest <- elicit(tail)
            if accepted then
                return rest + exp(weight) * likelihood
            else
                return rest + exp(weight) * likelihood / 2.0

define handle_audit(rows : Mix__Proofs) : Real !{} =
    handle audit with shift in
        elicit(rows)

program assess : Token -> Result
    let proofs = parse_normalized_proofs(Mix, sentence, 2)
    let response <- handle_audit(proofs)
    score likelihood = log(response)
    return likelihood

export assess
"""


INDEXED = """\
object Token : FinSet 2
object Result : Real 1

deduction Indexed : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, S, lex
    rule pair : span(I, K, A), span(K, J, B) |- span(I, J, S)
    lexicon
        "x" : A = lex
        "y" : B = lex

family Measurement(position : Nat) : Type
    constructor measured[n : Nat] : Real -> Measurement(n)

interpret measure : Indexed -> Measurement(x1)
    rule pair = combine
    axiom span = reading

define reading[left : Nat, right : Nat, tag : Indexed__Term](word : String) : Measurement(right) !{} =
    if word == "x" then
        return construct measured[right](2.0) as Measurement(right)
    else
        return construct measured[right](3.0) as Measurement(right)

define extract[n : Nat](reading : Measurement(n)) : Real !{} =
    case reading motive (p : Nat) => Real
        measured[k](value) =>
            return value

define combine[i : Nat, k : Nat, j : Nat](left : Measurement(k), right : Measurement(j)) : Measurement(j) !{} =
    let x <- extract[k](left)
    let y <- extract[j](right)
    return construct measured[j](x * x - y) as Measurement(j)

define source_proof() : Indexed__SpanProof(0, 2, S) !{} =
    let left = construct Indexed__axiom_span[0, 1, A]("x") as Indexed__SpanProof(0, 1, A)
    let right = construct Indexed__axiom_span[1, 2, B]("y") as Indexed__SpanProof(1, 2, B)
    return construct Indexed__rule_pair[0, 1, 2](left, right) as Indexed__SpanProof(0, 2, S)

define reading_of_proof() : Measurement(2) !{} =
    let proof <- source_proof()
    measure[0, 2, S](proof)

define result() : Real !{} =
    let reading <- reading_of_proof()
    extract[2](reading)

program assess_indexed : Token -> Result
    let value <- result()
    return value

export assess_indexed
"""


def test_indexed_proof_carrier_substitutes_each_premise_and_conclusion() -> None:
    compiled = load_source(INDEXED)
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule) and restored == module
    fold = next(item for item in restored.computations if item.name == "measure")
    assert isinstance(fold.body, Case)
    assert {binder.name for binder in fold.telescope}.isdisjoint(
        binder.name for binder in fold.body.motive.indices
    )
    measured = run_named(restored, "reading_of_proof").value
    assert isinstance(measured, RuntimeConstructor)
    assert isinstance(measured.result_type, TypeApplication)
    (position,) = measured.result_type.arguments
    assert isinstance(position, IndexLiteral) and position.value == 2
    (value,) = measured.fields
    assert float(value) == 1.0
    assert float(compiled.run("assess_indexed", data={}).value) == 1.0


def test_contextual_lift_in_an_indexed_binary_interpretation() -> None:
    source = INDEXED.replace(
        "    let x <- extract[k](left)\n    let y <- extract[j](right)\n",
        "    let kept <- lift_first[k](extract, left, right)\n"
        "    let x = kept[0]\n"
        "    let y <- extract[j](kept[1])\n",
    )
    compiled = load_source(source)
    module = loads(dumps(compiled.qiec))
    assert isinstance(module, QiecModule)
    # The proof's left and right child denote 2 and 3 independently.
    assert float(run_named(module, "result").value) == 2**2 - 3
    assert float(compiled.run("assess_indexed", data={}).value) == 2**2 - 3

    with pytest.raises(
        QiecDiagnosticError, match=r"expected Measurement\[k\], got Measurement\[j\]"
    ):
        load_source(source.replace("extract[j](kept[1])", "extract[k](kept[1])"))


@pytest.mark.parametrize(
    "old,new,message",
    [
        (
            "right : Measurement(j)) : Measurement(j)",
            "right : Measurement(k)) : Measurement(j)",
            "operation 'combine' does not match the closed interpretation",
        ),
        (
            "let y <- extract[j](right)",
            "let y <- extract[2](right)",
            r"expected Measurement\[2\], got Measurement\[j\]",
        ),
        (
            "reading_of_proof() : Measurement(2)",
            "reading_of_proof() : Measurement(1)",
            r"returns Measurement\[2\], expected Measurement\[1\]",
        ),
    ],
)
def test_indexed_proof_carrier_refuses_wrong_indices(old, new, message) -> None:
    with pytest.raises(QiecDiagnosticError, match=message):
        load_source(INDEXED.replace(old, new))


def test_authored_folds_score_one_chart_and_preserve_effect_occurrences() -> None:
    syntax = parse(GRAMMAR)
    printed = module_to_source(syntax)
    assert "interpret evidence : Mix -> Bool" in printed
    assert module_to_source(parse(printed)) == printed
    assert qiec_binding_map(syntax)["rating"].kind == "computation"

    compiled = load_source(printed)
    data = {"sentence": ("x", "y")}
    public = compiled.run("assess", data=data)
    chart = compiled.qiec
    assert isinstance(chart, QiecModule)
    restored = loads(dumps(chart))
    assert isinstance(restored, QiecModule) and restored == chart
    folds = {
        computation.name: computation
        for computation in restored.computations
        if computation.name in ("rating", "evidence")
    }
    assert isinstance(folds["rating"].body, Case)
    assert isinstance(folds["evidence"].body, Case)
    (audit,) = folds["rating"].type.effects.entries
    assert audit == next(
        item.entry for item in restored.instances if item.name == "audit"
    )
    assert not folds["evidence"].type.effects.entries

    logs = torch.tensor([0.3, 0.1], dtype=torch.float64, requires_grad=True)
    result = run_program(
        restored,
        "assess",
        data=data,
        sites={},
        parameters=dict(zip(("Mix.lex.0", "Mix.lex.1"), logs.unbind(), strict=True)),
    )
    # Each proof visits x then y under the shift handler. The left-selected
    # Boolean accepts the first proof and the right-selected Boolean rejects
    # the second. Swapping children or using one item-level readout changes
    # the likelihood before or after normalization.
    response = logs.new_tensor([(2.0**2 + 3.0) / 100, (3.0**2 + 2.0) / 200])
    expected = (logs + response.log()).logsumexp(0) - logs.logsumexp(0)
    assert public.log_joint == pytest.approx(math.log((0.07 + 0.055) / 2))
    torch.testing.assert_close(result.log_joint, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(result.log_joint, logs)[0],
        torch.autograd.grad(expected, logs)[0],
        atol=1e-12,
        rtol=1e-12,
    )
    calls = [
        event.detail["computation"]
        for event in result.result.trace
        if event.event == "call.entered"
        and event.detail["computation"] in {"read_word", "square_left", "square_right"}
    ]
    assert calls == [
        "read_word",
        "read_word",
        "square_left",
        "read_word",
        "read_word",
        "square_right",
    ]

    addresses = [
        event.detail["address"]
        for event in result.result.trace
        if event.event == "operation.requested"
        and event.detail["instance"] == str(audit.instance)
    ]
    assert len(addresses) == len(set(addresses)) == 4


def test_given_finite_proof_can_be_interpreted_without_a_finite_chart() -> None:
    source = """\
object Token : FinSet 1
object Height : Real 1

deduction Cyclic : Token -> Token [semiring=LogProb, start=A]
    atoms span, A, lex
    rule loop : span(I, J, A) |- span(I, J, A)
    lexicon
        "x" : A = lex

interpret height : Cyclic -> Real
    rule loop = successor
    axiom span = one

define successor(child : Real) : Real !{} =
    return child + 1.0

define one(word : String) : Real !{} =
    return 1.0

define example() : Real !{} =
    let leaf = construct Cyclic__axiom_span[0, 1, A]("x") as Cyclic__SpanProof(0, 1, A)
    let proof = construct Cyclic__rule_loop[0, 1](leaf) as Cyclic__SpanProof(0, 1, A)
    height[0, 1, A](proof)

program inspect : Token -> Height
    let depth <- example()
    return depth

export inspect
"""
    assert load_source(source).run("inspect").value == 2.0


def test_closed_schema_checks_assigned_actions_and_weighted_proof_identity() -> None:
    source = (
        GRAMMAR.replace(
            "instance audit : Inspection",
            "instance audit : Inspection\ninstance spare : Inspection",
        )
        + """
program unrelated : Token -> Result
    let value = 0.0
    return value
"""
    )
    model = Model(source, entry="assess")
    rating, evidence = model.schema.closed_interpretations
    module = model.checked_module
    wire = dumps(module)
    assert {item.name for item in rating.presentation.rules} == {
        "Mix__rule_first",
        "Mix__rule_second",
    }
    assert rating.presentation == evidence.presentation
    assert rating.carrier != evidence.carrier == BOOL
    assert rating.interpretation.preserves == evidence.interpretation.preserves == ()
    (audit,) = rating.instances
    assert audit.name == "audit" and rating.effects == EffectRow(entries=(audit.entry,))
    assert evidence.effects == EffectRow() and evidence.instances == ()
    assert model.bind({"sentence": ("x", "y")}).schema.closed_interpretations == (
        rating,
        evidence,
    )
    assert Model(source, entry="unrelated").schema.closed_interpretations == ()

    restored_module = loads(wire)
    assert isinstance(restored_module, QiecModule)
    for binding in (rating, evidence):
        restored = loads(dumps(binding))
        assert isinstance(restored, ClosedInterpretationBinding) and restored == binding
        restored.validate(restored_module)
    assert dumps(restored_module) == wire

    first, second = rating.presentation.rules
    changed_weight = rating.presentation.with_(
        rules=(first.with_(weight=ParameterId.derive("stale")), second)
    )
    ((a, f), (b, g), leaf) = rating.interpretation.operations
    reassigned = rating.interpretation.with_(operations=((a, g), (b, f), leaf))
    spare = next(item for item in module.instances if item.name == "spare")
    variants = (
        (
            rating.with_(
                presentation=rating.presentation.with_(
                    id=PresentationId.derive("stale")
                )
            ),
            "nominal closed interpretation presentation",
        ),
        (
            rating.with_(computation=ComputationId.derive("stale")),
            "unknown closed interpretation root",
        ),
        (rating.with_(presentation=changed_weight), "differs from its packed query"),
        (rating.with_(effects=EffectRow()), "effect row does not match"),
        (rating.with_(instances=(spare,)), "lexical effect requirements are stale"),
        (rating.with_(interpretation=reassigned), "differs from its checked fold"),
    )
    for changed, diagnostic in variants:
        restored = loads(dumps(changed))
        assert isinstance(restored, ClosedInterpretationBinding)
        with pytest.raises(RuleError, match=diagnostic):
            restored.validate(restored_module)

    replaced = restored_module.with_(
        computations=tuple(
            computation.with_(body=Return(value=LiteralValue(value=False, type=BOOL)))
            if computation.id == evidence.computation
            else computation
            for computation in restored_module.computations
        )
    )
    validate_module(replaced)
    with pytest.raises(RuleError, match="differs from its checked fold"):
        evidence.validate(replaced)


def test_closed_schema_keeps_index_template_reached_through_an_authored_handler() -> (
    None
):
    source = (
        INDEXED
        + """
effect Lookup
    get : Unit -> Real

instance lookup : Lookup

handler use_proof for Lookup : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        let value <- result()
        resume(value)

define prompted() : Real !{} =
    handle lookup with use_proof in
        perform lookup.get()

program through_handler : Token -> Result
    let value <- prompted()
    return value
"""
    )
    model = Model(source, entry="through_handler")
    (binding,) = model.bind({}).schema.closed_interpretations
    assert binding == model.schema.closed_interpretations[0]
    fold = next(
        item
        for item in model.checked_module.computations
        if item.id == binding.computation
    )
    assert isinstance(fold.body, Case)
    assert binding.carrier == fold.body.motive.result_type
    assert binding.carrier != fold.type.result
    (judgment,) = binding.presentation.judgments
    assert fold.body.motive.indices == judgment.indices
    restored = loads(dumps(binding))
    assert isinstance(restored, ClosedInterpretationBinding)
    restored.validate(model.checked_module)
    assert float(load_source(source).run("through_handler").value) == 1.0
    with pytest.raises(RuleError, match="does not match the closed interpretation"):
        restored.with_(carrier=BOOL).validate(model.checked_module)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("rule first = square_left", "rule bad = square_left", "unknown rule 'bad'"),
        (
            "rule first = square_left",
            "rule first = missing",
            "unknown interpretation operation",
        ),
        (
            "rule first = square_left",
            "rule first = is_x",
            "does not match the closed interpretation",
        ),
        (
            "rule first = square_left",
            "rule second = square_left",
            "duplicate interpretation",
        ),
        (
            "    rule first = square_left\n",
            "",
            "missing interpretation for rule 'first'",
        ),
        ("axiom span = read_word", "axiom leaf = read_word", "unknown axiom 'leaf'"),
        (
            "interpret rating : Mix",
            "interpret rating : Missing",
            "unknown deduction 'Missing'",
        ),
        (
            "interpret rating : Mix",
            "interpret read_word : Mix",
            "duplicate computation 'read_word'",
        ),
    ),
)
def test_interpretation_rejects_invalid_signatures_and_source_maps(
    old: str, new: str, message: str
) -> None:
    with pytest.raises(QiecDiagnosticError, match=message) as raised:
        load_source(GRAMMAR.replace(old, new))
    assert raised.value.line > 0


PROOF_OUTCOMES = (
    GRAMMAR[: GRAMMAR.index("define elicit")]
    .replace(
        "    lexicon\n",
        "    rule third : span(I, K, A, lex), span(K, J, D, lex) |- span(I, J, S, same)\n"
        "    lexicon\n",
    )
    .replace(
        "    axiom span = read_word",
        "    rule third = square_left\n    axiom span = read_word",
    )
    .replace(
        "    axiom span = is_x", "    rule third = left_truth\n    axiom span = is_x"
    )
    + """\
define likelihood(rows : Mix__evidence__Outcomes) : Real !{} =
    case rows motive => Real
        Mix__evidence__outcome_end =>
            return 0.0
        Mix__evidence__outcome_entry(key, weight, tail) =>
            let rest <- likelihood(tail)
            if key then
                return rest + 0.8 * exp(weight)
            else
                return rest + 0.3 * exp(weight)

define cardinality(rows : Mix__evidence__Outcomes) : Real !{} =
    case rows motive => Real
        Mix__evidence__outcome_end =>
            return 0.0
        Mix__evidence__outcome_entry(key, weight, tail) =>
            let rest <- cardinality(tail)
            return 1.0 + rest

program observed(limit : Int) : Token -> Result
    let readings = parse_normalized_proof_outcomes(Mix, sentence, limit, evidence)
    let answer <- likelihood(readings)
    score response = log(answer)
    return response

program raw : Token -> Result
    let readings = parse_proof_outcomes(Mix, sentence, 3, evidence)
    let answer <- likelihood(readings)
    return answer

program cardinal : Token -> Result
    let readings = parse_proof_outcomes(Mix, sentence, 3, evidence)
    let count <- cardinality(readings)
    return count

export observed
export raw
export cardinal
"""
)


def test_proof_keys_group_equal_meanings_within_one_identical_chart_item() -> None:
    compiled = load_source(PROOF_OUTCOMES)
    data = {"sentence": ("x", "y"), "limit": 3}
    assert compiled.run("observed", data=data).log_joint == pytest.approx(
        math.log((0.8 + 0.3 + 0.8) / 3)
    )
    assert compiled.run("cardinal", data={"sentence": data["sentence"]}).value == 2
    with pytest.raises(ExecutionFailure, match="too-many-derivations"):
        compiled.run("observed", data={**data, "limit": 2})
    with pytest.raises(ExecutionFailure, match="without positive goal mass"):
        compiled.run("observed", data={**data, "sentence": ("missing", "y")})
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule) and restored == module
    weights = torch.tensor([-0.33, 0.45], dtype=torch.float64, requires_grad=True)
    parameters = dict(zip(("Mix.lex.0", "Mix.lex.1"), weights.unbind(), strict=True))
    actual = run_program(
        restored, "observed", data=data, sites={}, parameters=parameters
    )
    # Proofs first and third share both one lexical choice and one Boolean
    # meaning. Proof second has the same goal and a different meaning.
    coefficients = weights.new_tensor([0.8, 0.3, 0.8])
    derivations = weights[torch.tensor([0, 1, 0])]
    expected = (derivations + coefficients.log()).logsumexp(0) - derivations.logsumexp(
        0
    )
    torch.testing.assert_close(actual.log_joint, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual.log_joint, weights)[0],
        torch.autograd.grad(expected, weights)[0],
        atol=1e-12,
        rtol=1e-12,
    )
    raw = run_program(
        restored,
        "raw",
        data={"sentence": data["sentence"]},
        sites={},
        parameters=parameters,
    )
    torch.testing.assert_close(raw.value, (coefficients * derivations.exp()).sum())
    calls = tuple(
        event.detail["computation"]
        for event in actual.result.trace
        if event.event == "call.entered"
        and event.detail["computation"]
        in {"Mix__proofs", "Mix__evidence__key_outcomes"}
    )
    assert calls == ("Mix__proofs", "Mix__evidence__key_outcomes")
    (proofs, grouped) = [
        event.detail
        for event in actual.result.trace
        if event.event in {"deduction.proofs", "deduction.grouped_outcomes"}
    ]
    assert proofs["normalized"] is False and proofs["structural_goal_proofs"] == 3
    assert grouped["normalized"] is True and grouped["input_rows"] == 3
    assert grouped["key_type"] == "Bool" and grouped["positive_rows"] == 2


def test_proof_keys_reject_swapped_proof_indices() -> None:
    wrong_key = """\
define bad[i : Nat, j : Nat, c : Mix__Term, f : Mix__Term](proof : Mix__SpanProof(j, i, c, f)) : Bool !{} =
    return true

"""
    source = PROOF_OUTCOMES.replace("program observed", wrong_key + "program observed")
    source = source.replace("limit, evidence)", "limit, bad)").replace(
        "3, evidence)", "3, bad)"
    )
    source = source.replace("Mix__evidence__", "Mix__bad__")
    with pytest.raises(QiecDiagnosticError, match="taking one Mix__SpanProof argument"):
        load_source(source)


def test_proof_keys_refuse_effects_and_locally_handled_effects() -> None:
    effectful = PROOF_OUTCOMES.replace(
        'define is_x(word : String) : Bool !{} =\n    return word == "x"',
        """define is_x(word : String) : Bool !{audit} =
    let value <- read_word(word)
    return value == 2.0""",
    )
    with pytest.raises(QiecDiagnosticError, match="must be a pure computation"):
        load_source(effectful)
    masked = effectful.replace(
        "define is_x(word : String) : Bool !{audit}",
        "define inspect_word(word : String) : Bool !{audit}",
    )
    masked += """
handler hide_effect for Inspection : Bool -> Bool [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    advance(value : Real) resumes 1 =>
        resume(value + 1.0)

define is_x(word : String) : Bool !{} =
    handle audit with hide_effect in
        inspect_word(word)
"""
    with pytest.raises(
        QiecDiagnosticError, match="must be deterministic.*installs an effect handler"
    ):
        load_source(masked)


def _normalized_lexical_source() -> str:
    return (
        GRAMMAR.replace("start=S]", "start=S, normalize=lhs]")
        .replace(", lex)", ")")
        .replace(", same)", ")")
        .replace("[i, j, category, form]", "[i, j, category]")
    )


@pytest.mark.parametrize("learnable", (False, True))
def test_ordinary_proof_companion_accepts_normalized_production_weights(
    learnable,
) -> None:
    source = _normalized_lexical_source()
    if learnable:
        source = source.replace(
            "span(I, J, S)\n    rule second",
            "span(I, J, S) #[learnable]\n    rule second",
            1,
        )
    model = Model(source, entry="assess")
    assert len(model.schema.closed_interpretations) == 2
    for binding in model.schema.closed_interpretations:
        restored = loads(dumps(binding))
        assert isinstance(restored, ClosedInterpretationBinding)
        restored.validate(model.checked_module)


def test_normalized_companion_checks_both_sources_of_weight_identity() -> None:
    model = Model(_normalized_lexical_source(), entry="assess")
    module = model.checked_module
    original = model.schema.closed_interpretations[0]
    first, second = original.presentation.rules
    forged = original.with_(
        presentation=original.presentation.with_(
            rules=(
                first.with_(weight=ParameterId.derive("unrelated", "rule", "first")),
                second,
            )
        )
    )
    with pytest.raises(RuleError, match="differs from its packed query"):
        forged.validate(module)

    (handler,) = [
        item for item in module.handlers if isinstance(item.foreign, DeductionProofs)
    ]
    assert isinstance(handler.foreign, DeductionProofs)
    chart = handler.foreign.chart
    assert len(chart.parameters) == 2
    forged_chart = chart.with_(parameters=chart.parameters[::-1])
    changed = module.with_(
        handlers=tuple(
            item.with_(foreign=handler.foreign.with_(chart=forged_chart))
            if item.id == handler.id
            else item
            for item in module.handlers
        )
    )
    validate_module(changed)
    with pytest.raises(RuleError, match="differs from its packed query"):
        original.validate(changed)


def test_companion_refuses_a_different_checked_chart_proof_shape() -> None:
    source = GRAMMAR.replace(
        "    let proofs = parse_normalized_proofs",
        '    let items = parse_outcomes(Mix, sentence, "item")\n'
        "    let proofs = parse_normalized_proofs",
        1,
    )
    model = Model(source, entry="assess")
    module = model.checked_module
    (handler,) = [
        item for item in module.handlers if isinstance(item.foreign, DeductionOutcomes)
    ]
    chart = handler.foreign
    assert isinstance(chart, DeductionOutcomes)
    presentation = chart.presentation
    assert isinstance(presentation, RulePresentation)
    rule, *others = presentation.rules
    first, *rest = rule.premises
    i, k, category, form = first.judgment.arguments
    assert isinstance(category, IndexLiteral) and category.value == "A"
    different = first.judgment.with_(arguments=(i, k, category.with_(value="B"), form))
    changed = presentation.with_(
        rules=(rule.with_(premises=(first.with_(judgment=different), *rest)), *others)
    )
    forged = module.with_(
        handlers=tuple(
            item.with_(foreign=chart.with_(presentation=changed))
            if item.id == handler.id
            else item
            for item in module.handlers
        )
    )
    validate_module(forged)
    with pytest.raises(RuleError, match="differs from its packed query"):
        model.schema.closed_interpretations[0].validate(forged)
