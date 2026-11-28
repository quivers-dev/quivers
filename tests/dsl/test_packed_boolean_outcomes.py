"""Packed Boolean proof algebras preserve proof mass and lexical gradients."""

from itertools import product
from math import log
from pathlib import Path

import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec import (
    DeductionBooleanOutcomes,
    QiecModule,
    dumps,
    loads,
    validate_module,
)
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.program_runtime import run_program
from quivers.transpile import UnsupportedConstruct, transpile


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "docs/examples/qiec/packed-boolean-proof-outcomes.qvr"
).read_text()


EXPLICIT = """
define explicit_mass(rows : Parity__parity__Outcomes) : Real !{} =
    case rows motive => Real
        Parity__parity__outcome_end =>
            return 0.0
        Parity__parity__outcome_entry(key, weight, tail) =>
            if key then
                return exp(weight)
            else
                explicit_mass(tail)

program explicit : Token -> Result
    let rows = parse_normalized_proof_outcomes(Parity, sentence, 16, parity)
    let mass <- explicit_mass(rows)
    score observed = log(mass)
    return observed

export explicit
"""

RAW = """
define score_true(law : Sampleable[Bool]) : Real !{} =
    return weight_value(log_prob(law, true))

program raw : Token -> Result
    let rows = parse_packed_boolean_outcomes(Parity, sentence, parity)
    let mass <- truth_mass(rows)
    return mass

program finite_law : Token -> Result
    let tokens = ["x", "x", "x"]
    let rows = parse_packed_boolean_outcomes(Parity, tokens, parity)
    let law = outcome_law(rows)
    let density <- score_true(law)
    score observed = density
    return observed

export raw
export finite_law
"""


@pytest.fixture(scope="module")
def compiled():
    return load_source(SOURCE + EXPLICIT + RAW)


def _weights(logs):
    return dict(zip(("Parity.lex.0", "Parity.lex.1"), logs.unbind(), strict=True))


def _native(module, entry, length, logs):
    return run_program(
        module,
        entry,
        data={"sentence": ("x",) * length},
        sites={},
        parameters=_weights(logs),
    )


def _enumerated(logs, length):
    # Each word has exactly one A and one B axiom. The only binary tree
    # successively adjoins one more word to the goal's left prefix.
    # Thus there is exactly one proof per bit string, weighted by one
    # lexical factor for each digit. The reading is true at odd parity.
    words = tuple(product((0, 1), repeat=length))
    terms = torch.stack(tuple(logs[list(bits)].sum() for bits in words))
    keep = [index for index, bits in enumerate(words) if sum(bits) % 2]
    return terms[keep].logsumexp(0), terms.logsumexp(0)


def test_checked_packed_proof_mass_agrees_with_explicit_proofs_and_autograd(compiled):
    assert compiled.run(
        "fit", data={"sentence": ("x",) * 3}
    ).log_joint == pytest.approx(log(0.5))
    assert compiled.run("finite_law").log_joint == pytest.approx(log(0.5))
    module = loads(dumps(compiled.qiec))
    assert isinstance(module, QiecModule) and module == compiled.qiec
    validate_module(module)
    (descriptor,) = [
        handler.foreign
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionBooleanOutcomes)
    ]
    assert loads(dumps(descriptor)) == descriptor

    logs = torch.tensor([-0.91, 0.27], dtype=torch.float64, requires_grad=True)
    raw, normalizer = _enumerated(logs, 3)
    expected = raw - normalizer
    gradient = torch.autograd.grad(expected, logs, retain_graph=True)[0]
    results = tuple(_native(module, which, 3, logs) for which in ("fit", "explicit"))
    law = run_program(
        module, "finite_law", data={}, sites={}, parameters=_weights(logs)
    )
    for result in (*results, law):
        torch.testing.assert_close(result.log_joint, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            torch.autograd.grad(result.log_joint, logs)[0],
            gradient,
            atol=1e-12,
            rtol=1e-12,
        )
    unnormalized = _native(module, "raw", 3, logs)
    torch.testing.assert_close(unnormalized.value, raw.exp(), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(unnormalized.value, logs)[0],
        torch.autograd.grad(raw.exp(), logs)[0],
        atol=1e-12,
        rtol=1e-12,
    )
    calls = [
        event.detail["computation"]
        for event in results[0].result.trace
        if event.event == "call.entered"
        and event.detail["computation"]
        in {"lexical", "reject", "accept", "copy", "xor"}
    ]
    assert calls == [
        "lexical",
        "reject",
        "reject",
        "accept",
        "accept",
        "copy",
        "copy",
    ] + ["xor", "xor", "xor", "xor"]

    previous = results[0].log_joint.detach()
    with torch.no_grad():
        logs.add_(0.4 * gradient)
    refit = _native(module, "fit", 3, logs)
    new_raw, new_total = _enumerated(logs, 3)
    new_expected = new_raw - new_total
    assert not torch.isclose(refit.log_joint, previous)
    torch.testing.assert_close(refit.log_joint, new_expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(refit.log_joint, logs)[0],
        torch.autograd.grad(new_expected, logs)[0],
        atol=1e-12,
        rtol=1e-12,
    )


def test_proof_interpretation_work_tracks_packed_items_instead_of_proofs(compiled):
    module = compiled.qiec
    logs = torch.tensor([-3.0, -0.1], dtype=torch.float64, requires_grad=True)
    for length in (4, 20):
        packed = _native(module, "fit", length, logs)
        a, b = logs.exp().unbind()
        # The sum over all bit strings is (a+b)^n. Subtracting the
        # signed odd terms from the even ones gives (a-b)^n.
        expected = ((1 - ((a - b) / (a + b)) ** length) / 2).log()
        torch.testing.assert_close(packed.log_joint, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            torch.autograd.grad(packed.log_joint, logs)[0],
            torch.autograd.grad(expected, logs)[0],
            atol=1e-12,
            rtol=1e-12,
        )
        (event,) = [
            event.detail
            for event in packed.result.trace
            if event.event == "deduction.boolean_outcomes"
        ]
        assert event["observer"] == "weighted-boolean-proof-meanings"
        assert event["algorithm"] == "indexed-packed-forest-boolean-fold"
        assert event["positive_rows"] == 2
        assert event["truth_table_entries"] == 11
        assert event["evaluated_items"] == 4 * length
        assert event["edge_meaning_lanes"] == 10 * length - 2
        assert event["weight_products"] == 12 * length - 6
        assert event["reduced_entries"] == event["edge_meaning_lanes"]
        # All intervals enter the construction, but only prefixes reach
        # the final result. The prior measurement counts original edges
        # and their premise references, including the discarded intervals.
        assert dict(event["construction_work"])["provenance_records"] == (
            8 * length + 3 * length * (length - 1) // 2
        )
        assert (
            any(event.event == "deduction.proofs" for event in packed.result.trace)
            is False
        )
    assert 2**20 > 1_000_000
    with pytest.raises(ExecutionFailure, match="too-many-derivations"):
        _native(module, "explicit", 20, logs)


def test_same_signature_semantic_substitution_changes_the_likelihood(compiled):
    changed = (
        SOURCE.replace("rule join = xor", "rule join = first")
        + """
define first(left : Bool, right : Bool) : Bool !{} =
    return left
"""
    )
    alternative = load_source(changed)
    logs = torch.tensor([-1.0, 0.3], dtype=torch.float64, requires_grad=True)
    left_choice = _native(alternative.qiec, "fit", 3, logs)
    original = _native(compiled.qiec, "fit", 3, logs)
    # This operation always copies the first lexical node through both
    # binary applications, so its meaning law is just the first draw.
    expected = logs.log_softmax(0)[1]
    assert abs(original.log_joint.item() - expected.item()) > 0.1
    torch.testing.assert_close(left_choice.log_joint, expected)
    torch.testing.assert_close(
        torch.autograd.grad(left_choice.log_joint, logs)[0],
        torch.autograd.grad(expected, logs)[0],
    )


def test_zero_semantic_mass_preserves_positive_neighbor_and_refuses_empty_normalizer(
    compiled,
):
    logs = torch.tensor([0.0, -torch.inf], dtype=torch.float64, requires_grad=True)
    packed = _native(compiled.qiec, "fit", 2, logs)
    assert torch.isneginf(packed.log_joint)
    (event,) = [
        event.detail
        for event in packed.result.trace
        if event.event == "deduction.boolean_outcomes"
    ]
    assert event["positive_rows"] == 1
    with pytest.raises(ExecutionFailure, match="without positive goal mass"):
        run_program(compiled.qiec, "fit", data={"sentence": ("missing",)}, sites={})


def test_packed_truth_table_refuses_a_partial_operation_before_any_folding():
    # A one-token proof never uses the join, so its explicit evaluation
    # can succeed even when the function's false-left input is undefined.
    source = SOURCE.replace(
        "    if left == right then\n        return false\n    else\n        return true",
        "    if left then\n        return right\n    else\n        return log(-1.0) <= 0.0",
    )
    compiled = load_source(source + EXPLICIT)
    assert compiled.run(
        "explicit", data={"sentence": ("x",)}
    ).log_joint == pytest.approx(log(0.5))
    with pytest.raises(ExecutionFailure, match="log is applied outside its domain"):
        compiled.run("fit", data={"sentence": ("x",)})


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (
            SOURCE.replace("sentence, parity", "sentence, xor"),
            "closed Bool interpretation",
        ),
        (
            SOURCE.replace("define xor(", "define xor[i : Nat, k : Nat, j : Nat]("),
            "no static parameters or effects",
        ),
        (
            SOURCE.replace(
                "rule join : span(I, K, S, word), span(K, J, T, word)",
                "rule join : span(I, K, S, word), span(K, L, T, word), span(L, J, T, word)",
            ),
            "arity at most two",
        ),
    ],
    ids=("unbound-interpretation", "indexed-operation", "ternary-rule"),
)
def test_inapplicable_packed_interpretations_have_stable_refusals(source, reason):
    with pytest.raises(QiecDiagnosticError, match=reason) as error:
        load_source(source)
    assert error.value.code == "qiec-deduction-outcomes"


def test_packed_proof_map_refuses_effects_hidden_under_a_handler():
    attached = """
effect Probe
    inspect : Bool -> Bool

instance probe : Probe

handler identity for Probe : Bool -> Bool [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    inspect(value : Bool) resumes 1 =>
        resume(value)

define intercepted(value : Bool) : Bool !{} =
    handle probe with identity in
        let inspected <- perform probe.inspect(value)
        return inspected

"""
    source = SOURCE.replace("define xor", attached + "define xor").replace(
        "if left == right then\n        return false",
        "if left == right then\n        intercepted(false)",
    )
    with pytest.raises(
        QiecDiagnosticError, match="'intercepted' installs an effect handler"
    ):
        load_source(source)


def test_native_target_refuses_packed_query_and_can_render_unrelated_export():
    with pytest.raises(UnsupportedConstruct, match="deduction-outcomes") as error:
        transpile(parse(SOURCE), target="stan")
    assert "qiec:capability:deduction-outcomes:fit" in error.value.kinds

    independent = SOURCE.replace(
        "export fit",
        """
program independent : Token -> Result
    sample value <- Normal(loc=0.0, scale=1.0)
    return value

export independent
""",
    )
    assert b"normal" in transpile(parse(independent), target="stan")


def test_ordered_truth_operands_distinguish_left_from_right():
    source = (
        SOURCE.replace(
            '        "x" : B = word #[learnable]',
            '        "x" : B = word #[learnable]\n        "y" : A = word',
        ).replace("rule join = xor", "rule join = select")
        + """
define select(left : Bool, right : Bool) : Bool !{} =
    return left
"""
    )
    first = load_source(source).run("fit", data={"sentence": ("x", "y")})
    second = load_source(source.replace("return left", "return right")).run(
        "fit", data={"sentence": ("x", "y")}
    )
    # The left span can mean either truth value. The right span can only
    # mean false, so sorting or swapping their inputs would change the law.
    assert first.log_joint == pytest.approx(log(0.5))
    assert torch.isneginf(second.log_joint)


def test_repeated_nullable_premise_selects_two_independent_proofs():
    start = SOURCE.index("    rule no :")
    end = SOURCE.index("interpret parity")
    source = (
        SOURCE[:start]
        + """\
    rule no : |- span(0, 0, T, word)
    rule yes : |- span(0, 0, T, word)
    rule join : span(I, K, T, word), span(K, J, T, word) |- span(I, J, S, word)
    lexicon
        "x" : A = word

"""
        + SOURCE[end:]
    )
    source = (
        source.replace("    rule base = copy\n", "")
        .replace("rule no = reject", "rule no = ground_no")
        .replace("rule yes = accept", "rule yes = ground_yes")
        + """
define ground_no() : Bool !{} =
    return false

define ground_yes() : Bool !{} =
    return true
"""
    )
    compiled = load_source(source + EXPLICIT)
    packed = compiled.run("fit", data={"sentence": ()})
    explicit = compiled.run("explicit", data={"sentence": ()})
    # Both premise slots use the same chart item. Its two distinct trees
    # give four ordered tuples, two with odd parity. Reusing one tree in
    # both slots would give zero true mass.
    assert packed.log_joint == explicit.log_joint == pytest.approx(log(0.5))
    (trace,) = [
        item.detail
        for item in packed.result.trace
        if item.event == "deduction.boolean_outcomes"
    ]
    assert trace["evaluated_items"] == 2
    assert trace["edge_meaning_lanes"] == 6


def test_zero_truth_branch_keeps_live_gradient_through_raw_mass():
    source = SOURCE.replace("if key then", "if key == false then").replace(
        "parse_normalized_packed_boolean_outcomes(", "parse_packed_boolean_outcomes("
    )
    compiled = load_source(source)
    logs = torch.tensor([-0.2, -torch.inf], dtype=torch.float64, requires_grad=True)
    run = _native(compiled.qiec, "fit", 2, logs)
    torch.testing.assert_close(run.log_joint, 2 * logs[0])
    torch.testing.assert_close(
        torch.autograd.grad(run.log_joint, logs)[0], logs.new_tensor([2, 0])
    )


def test_same_carrier_interpretation_from_another_deduction_is_refused():
    start = SOURCE.index("deduction Parity")
    end = SOURCE.index("define lexical")
    nominally_other = (
        SOURCE[start:end]
        .replace("deduction Parity :", "deduction Other :")
        .replace("interpret parity : Parity", "interpret other : Other")
    )
    source = (
        SOURCE[:end]
        + nominally_other
        + SOURCE[end:].replace("sentence, parity", "sentence, other")
    )
    with pytest.raises(
        QiecDiagnosticError, match="closed Bool interpretation of 'Parity'"
    ):
        load_source(source)
