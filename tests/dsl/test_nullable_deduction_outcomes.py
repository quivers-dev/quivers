"""Finite nullable checked presentations keep each empty proof occurrence."""

from __future__ import annotations

import didactic.api as dx
import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.qiec.foreign import DeductionOutcomes
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.kinds import NAT
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.program_runtime import run_program
from quivers.qiec.rules import Judgment, Rule, RulePresentation
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.types import IndexLiteral


GRAMMAR = """\
object Token : FinSet 1
object Result : Real 1

deduction Mix : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, E, F, S, lex, yes, no
    rule pair : span(I, K, E, lex), span(K, J, E, lex) |- span(I, J, F, lex)
    rule agree : span(I, K, F, lex), span(K, J, A, lex) |- span(I, J, S, yes)
    rule disagree : span(I, K, E, lex), span(K, J, B, lex) |- span(I, J, S, no)
    rule accept_empty : span(I, J, F, lex) |- span(I, J, S, yes)
    lexicon
        "x" : A = lex #[learnable]
        "x" : B = lex #[learnable]
"""

PROGRAM = """\
define mass(rows : Mix__Outcomes, yes_rate : Real, no_rate : Real) : Real !{} =
    case rows motive => Real
        Mix__outcome_end =>
            return 0.0
        Mix__outcome_entry(key, weight, tail) =>
            let label <- Mix__show(key)
            let rest <- mass(tail, yes_rate, no_rate)
            if label == "yes" then
                return exp(weight) * yes_rate + rest
            else
                return exp(weight) * no_rate + rest

program scored : Token -> Result
    let rows = parse_normalized_outcomes(Mix, sentence, "form")
    let answer <- mass(rows, yes_rate, no_rate)
    score response = log(answer)
    return response

program raw : Token -> Result
    let rows = parse_outcomes(Mix, sentence, "form")
    let answer <- mass(rows, yes_rate, no_rate)
    return answer

export scored
export raw
"""


def _with_ground_rules(
    grammar: str, *rows: tuple[str, int, int, str, str]
) -> tuple[QiecModule, DeductionOutcomes]:
    """Install checked ground QIEC rules in a normal authored query module."""
    module = load_source(grammar + "\n" + PROGRAM).qiec
    assert isinstance(module, QiecModule)
    handler = next(
        item for item in module.handlers if isinstance(item.foreign, DeductionOutcomes)
    )
    semantics = handler.foreign
    assert isinstance(semantics, DeductionOutcomes)
    presentation = semantics.presentation
    assert isinstance(presentation, RulePresentation)
    (span,) = presentation.judgments
    category_sort, form_sort = (binder.sort for binder in span.indices[2:])
    axioms = tuple(
        Rule(
            id=ConstructorId.derive(presentation.id, "test-ground", name),
            name=name,
            conclusion=Judgment(
                family=span.id,
                arguments=(
                    IndexLiteral(value=left, sort=NAT),
                    IndexLiteral(value=right, sort=NAT),
                    IndexLiteral(value=category, sort=category_sort),
                    IndexLiteral(value=form, sort=form_sort),
                ),
            ),
        )
        for name, left, right, category, form in rows
    )
    semantics = semantics.with_(
        presentation=presentation.with_(rules=(*presentation.rules, *axioms))
    )
    module = module.with_(
        handlers=tuple(
            item.with_(foreign=semantics) if item.id == handler.id else item
            for item in module.handlers
        )
    )
    validate_module(module)
    return module, semantics


def test_nullable_proofs_repeat_premise_occurrences_and_preserve_gradients() -> None:
    module, semantics = _with_ground_rules(
        GRAMMAR,
        ("epsilon_one", 0, 0, "E", "lex"),
        ("epsilon_two", 0, 0, "E", "lex"),
        ("span_axiom", 0, 1, "S", "yes"),
        ("out_of_goal", 2, 2, "E", "lex"),
    )
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    assert restored == module
    assert semantics.unary_height == 1
    assert semantics.span_order.height == 2
    assert semantics.span_order.nullable

    logs = torch.tensor([-0.5, -0.9], dtype=torch.float64, requires_grad=True)
    rates = torch.tensor([0.8, 0.3], dtype=logs.dtype, requires_grad=True)
    parameters = {f"Mix.lex.{i}": weight for i, weight in enumerate(logs.unbind())}
    data = {"sentence": ("x",), "yes_rate": rates[0], "no_rate": rates[1]}

    # Independent proof list: four ordered pairs of epsilon choices
    # feed A, two individual epsilon choices feed B, and one ground rule
    # proves yes directly. The chart may share the empty premises.
    yes_mass = sum(logs[0].exp() for _ in range(4)) + 1.0
    no_mass = sum(logs[1].exp() for _ in range(2))
    expected = ((rates[0] * yes_mass + rates[1] * no_mass) / (yes_mass + no_mass)).log()
    result = run_program(restored, "scored", data=data, sites={}, parameters=parameters)
    torch.testing.assert_close(result.log_joint, expected, atol=1e-12, rtol=1e-12)
    for actual, reference in zip(
        torch.autograd.grad(result.log_joint, (logs, rates)),
        torch.autograd.grad(expected, (logs, rates)),
        strict=True,
    ):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)

    empty = {"sentence": (), "yes_rate": 1.0, "no_rate": 0.0}
    raw = run_program(restored, "raw", data=empty, sites={}, parameters=parameters)
    assert raw.value == pytest.approx(4.0)
    normalized = run_program(
        restored, "scored", data=empty, sites={}, parameters=parameters
    )
    assert normalized.log_joint == pytest.approx(0.0)


def test_ground_empty_span_exposes_a_cycle_in_a_binary_rule() -> None:
    original = GRAMMAR.replace(
        "    lexicon\n",
        "    rule cycle : span(I, K, S, yes), span(K, J, E, lex) |- span(I, J, S, yes)\n"
        "    lexicon\n",
    )
    # Before E can be empty, this is strictly width-decreasing on each
    # premise. Afterwards S(0,0) has an actual self-dependency.
    _, checked = _with_ground_rules(original)
    assert not checked.span_order.nullable
    with pytest.raises(
        dx.ValidationError, match="equal-width category heads must be acyclic"
    ):
        _with_ground_rules(original, ("epsilon", 0, 0, "E", "lex"))


def test_ground_span_refuses_reversed_endpoints() -> None:
    with pytest.raises(
        dx.ValidationError, match="requires ordered literal span endpoints"
    ):
        _with_ground_rules(GRAMMAR, ("reversed", 2, 1, "E", "lex"))
