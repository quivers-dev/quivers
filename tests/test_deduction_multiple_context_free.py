"""The checked search agrees with the agenda on multiple context-free grammars.

A lexical deduction whose items carry several spans (a multiple
context-free grammar) elaborates to a search that derives an item from
its head and the positions of its yield: a rule binds its conclusion's
components, chooses where its premises' components meet inside them,
and derives each premise over its own components. Its goal spans cover
the whole sentence. These tests compare that search with the agenda
chart on discontinuous items, premises in any order, rules with three
premises, and every short string of a copy language.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest
import torch

from quivers.dsl import load, loads, parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.qiec.program_runtime import run_deduction

_EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples" / "source"

DISCONTINUOUS = """
object Term : FinSet 4
deduction G : Term -> Term [semiring=LogProb, start=S, depth=2]
    atoms S, A, B, p, span, a, b
    rule pair : span(I1, J1, A), span(I2, J2, B) |- p(I1, J1, I2, J2)
    rule top : span(I, J, A), p(J, K, L, M), span(K, L, B) |- span(I, M, S)
    lexicon
        "a" : A = a
        "b" : B = b
"""

# The copy language { w w : w in {a, b}+ }: ``p`` pairs the two copies of
# a prefix, one component each, and ``grow_a`` / ``grow_b`` extend both
# components at once (their premises listed in different orders).
COPY = """
object Term : FinSet 8
deduction Copy : Term -> Term [semiring=SEMIRING, start=S]
    atoms S, A, B, p, span, a, b
    rule pair_a : span(I1, J1, A), span(I2, J2, A) |- p(I1, J1, I2, J2) #[learnable]
    rule pair_b : span(I2, J2, B), span(I1, J1, B) |- p(I1, J1, I2, J2) #[learnable]
    rule grow_a : span(J1, K1, A), p(I1, J1, I2, J2), span(J2, K2, A) |- p(I1, K1, I2, K2) #[learnable]
    rule grow_b : p(I1, J1, I2, J2), span(J2, K2, B), span(J1, K1, B) |- p(I1, K1, I2, K2) #[learnable]
    rule copy : p(I, J, J, K) |- span(I, K, S) #[learnable]
    lexicon
        "a" : A = a
        "b" : B = b
"""


def _both(source: str, name: str, tokens: list[str]) -> tuple[float, float]:
    """Score a sentence with the agenda chart and with the checked search.

    Parameters
    ----------
    source : str
        The QVR source.
    name : str
        The deduction.
    tokens : list[str]
        The sentence.

    Returns
    -------
    tuple[float, float]
        The agenda's goal weight and ``run_deduction``'s.
    """
    program = loads(source)
    agenda = float(program.deductions[name](tokens).goal_weight().detach())
    checked = run_deduction(program.qiec, name, tokens=tokens).weight
    return agenda, float(checked)


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("a a b b", 0.0),
        ("a b", -math.inf),
        ("a a b b a", -math.inf),
        ("a b a b", -math.inf),
    ],
)
def test_a_discontinuous_item_is_derived_like_the_agenda_derives_it(
    sentence: str, expected: float
) -> None:
    agenda, checked = _both(DISCONTINUOUS, "G", sentence.split())
    assert agenda == expected
    assert checked == expected


def test_every_short_string_of_the_copy_language_agrees() -> None:
    source = COPY.replace("SEMIRING", "LogProb")
    program = loads(source)
    system = program.deductions["Copy"]
    accepted = []
    for length in range(1, 4):
        for letters in itertools.product("ab", repeat=length):
            tokens = list(letters)
            agenda = float(system(tokens).goal_weight().detach())
            checked = run_deduction(program.qiec, "Copy", tokens=tokens).weight
            assert checked == agenda, tokens
            if agenda > -math.inf:
                accepted.append("".join(tokens))
    assert accepted == ["aa", "bb"]


@pytest.mark.parametrize(
    ("word", "derivations"), [("abab", 1), ("baba", 1), ("abba", 0), ("aabb", 0)]
)
def test_copy_language_counts_agree_with_interleaved_components(
    word: str, derivations: int
) -> None:
    source = COPY.replace("SEMIRING", "Counting")
    program = loads(source)
    tokens = list(word)
    agenda = int(program.deductions["Copy"](tokens).goal_weight())
    checked = run_deduction(program.qiec, "Copy", tokens=tokens).weight
    assert agenda == derivations
    assert checked == derivations


def test_copy_language_weights_agree() -> None:
    source = COPY.replace("SEMIRING", "LogProb")
    program = loads(source)
    system = program.deductions["Copy"]
    values = {
        "pair_a": -0.4,
        "pair_b": -1.1,
        "grow_a": 0.3,
        "grow_b": -0.6,
        "copy": -0.2,
    }
    with torch.no_grad():
        for key, parameter in system.named_parameters():
            rule = key.split(".")[1].removeprefix("rule_")
            parameter.fill_(values[rule])
    tokens = list("abab")
    agenda = float(system(tokens).goal_weight().detach())
    # abab = (ab)(ab): pair_a over the two a's, grow_b over the two b's,
    # then copy.
    assert agenda == pytest.approx(-0.4 - 0.6 - 0.2)
    checked = run_deduction(
        program.qiec,
        "Copy",
        tokens=tokens,
        parameters={f"Copy.rule.{rule}:": value for rule, value in values.items()},
    ).weight
    assert checked == pytest.approx(agenda)


@pytest.mark.slow
def test_the_cross_serial_grammar_scores_one_derivation_on_both_evaluators() -> None:
    program = load(_EXAMPLES / "wmcfg.qvr")
    # The shortest sentence the grammar licenses: one accusative pair
    # inside the finite cluster's own pair.
    tokens = "mer d'chind es huus lönd aastriiche".split()
    view = program.deductions["WMCFG"](tokens)
    assert [item for item, _ in view.goal_items] == [("span", 0, 6, ("atom", "S"))]
    agenda = float(view.goal_weight().detach())
    checked = run_deduction(program.qiec, "WMCFG", tokens=tokens).weight
    assert agenda == 0.0
    assert checked == 0.0


@pytest.mark.parametrize(
    ("rule", "fragment"),
    [
        (
            "rule both : span(I, J, A), span(I, K, B) |- p(I, J, J, K)",
            "two premise components start at 'I'",
        ),
        (
            "rule drop : span(I, J, A), span(J, K, B) |- p(I, J, J, J)",
            "premise component (J, K) is not part of the conclusion's yield",
        ),
    ],
)
def test_a_rule_that_is_not_linear_is_refused_at_the_rule(
    rule: str, fragment: str
) -> None:
    source = DISCONTINUOUS.replace(
        "    rule pair : span(I1, J1, A), span(I2, J2, B) |- p(I1, J1, I2, J2)",
        f"    {rule}",
    )
    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parse(source), file_path="nonlinear.qvr")
    assert captured.value.line == 5
    assert fragment in captured.value.message


ERASING = """
object Term : FinSet 4
deduction E : Term -> Term [semiring=Counting, start=seen]
    atoms seen, A, B, span, a, b
    rule found : span(I, J, B) |- seen
    lexicon
        "a" : A = a
        "b" : B = b
"""


@pytest.mark.parametrize(("sentence", "count"), [("a b b", 2), ("a a", 0), ("b", 1)])
def test_an_erasing_rule_counts_its_premises_anywhere_in_the_sentence(
    sentence: str, count: int
) -> None:
    program = loads(ERASING)
    tokens = sentence.split()
    agenda = int(program.deductions["E"](tokens).goal_weight())
    checked = run_deduction(program.qiec, "E", tokens=tokens).weight
    assert agenda == count
    assert checked == count
