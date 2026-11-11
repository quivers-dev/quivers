"""A deduction's ``depth`` bounds category depth on both evaluators.

The ``depth`` option bounds the constructor-tree depth of each rule's
conclusion category, in the agenda chart and in the checked QIEC search
alike; it never bounds how many rules a derivation applies. The search
recurs on smaller spans, applies unary rules in chains over one span,
and converges on a cycle of unary rules by raising how often a chain may
revisit an item until the goal's weight settles within the declared
``tolerance``.
"""

from __future__ import annotations

import math

import pytest

from quivers.dsl import loads
from quivers.qiec.program_runtime import run_deduction

CHAIN = """
object Term : FinSet 4
deduction G : Term -> Term [semiring=LogProb, start=S, depth=%d]
    atoms S, A, span, a
    rule grow : span(I, K, S), span(K, J, A) |- span(I, J, S)
    rule start : span(I, J, A) |- span(I, J, S)
    lexicon
        "a" : A = a
"""

RAISING = """
object Term : FinSet 8
deduction G : Term -> Term [semiring=Counting, start=S, depth=%d]
    atoms S, NP, Fwd, Bwd, span, john, sleeps
    rule fwd : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X)
    rule bwd : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X)
    rule lift : span(I, J, NP) |- span(I, J, Fwd(S, Bwd(S, NP)))
    lexicon
        "john"   : NP         = john
        "sleeps" : Bwd(S, NP) = sleeps
"""

ATTACH = """
object Term : FinSet 8
deduction Attach : Term -> Term [semiring=SEMIRING, start=S, depth=1]
    atoms S, NP, VP, PP, V, P, span, saw, man, telescope, with, a
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S)
    rule vp_v_np : span(I, K, V), span(K, J, NP) |- span(I, J, VP)
    rule vp_vp_pp : span(I, K, VP), span(K, J, PP) |- span(I, J, VP)
    rule np_np_pp : span(I, K, NP), span(K, J, PP) |- span(I, J, NP)
    rule pp_p_np : span(I, K, P), span(K, J, NP) |- span(I, J, PP)
    rule vp_v : span(I, J, V) |- span(I, J, VP)
    lexicon
        "I"         : NP = a
        "saw"       : V  = saw
        "man"       : NP = man
        "telescope" : NP = telescope
        "with"      : P  = with
"""

MODAL = """
object Term : FinSet 8
deduction M : Term -> Term [semiring=LogProb, start=S, depth=1, tolerance=1e-5]
    atoms S, NP, VP, Dia, span, kim, runs
    rule s : span(I, K, NP), span(K, J, VP) |- span(I, J, S)
    rule dia_intro : span(I, J, A) |- span(I, J, Dia(A)) #[learnable, bounded]
    rule dia_elim : span(I, J, Dia(A)) |- span(I, J, A) #[learnable, bounded]
    lexicon
        "kim"  : NP = kim
        "runs" : VP = runs
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


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_depth_never_cuts_a_derivation_whose_categories_are_atoms(
    depth: int, length: int
) -> None:
    agenda, checked = _both(CHAIN % depth, "G", ["a"] * length)
    assert agenda == 0.0
    assert checked == 0.0


@pytest.mark.parametrize(("depth", "derivations"), [(1, 1), (2, 2), (3, 2)])
def test_depth_cuts_the_same_deep_categories_on_both_evaluators(
    depth: int, derivations: int
) -> None:
    # Lifting NP to Fwd(S, Bwd(S, NP)) builds a category of depth 2, so
    # the second derivation of "john sleeps" exists from depth 2 on.
    agenda, checked = _both(RAISING % depth, "G", ["john", "sleeps"])
    assert agenda == derivations
    assert checked == derivations


@pytest.mark.parametrize(
    ("semiring", "expected"), [("Counting", 2), ("LogProb", math.log(2.0))]
)
def test_an_ambiguous_sentence_longer_than_the_depth_counts_every_derivation(
    semiring: str, expected: float
) -> None:
    tokens = ["I", "saw", "man", "with", "telescope"]
    agenda, checked = _both(ATTACH.replace("SEMIRING", semiring), "Attach", tokens)
    assert agenda == pytest.approx(expected)
    assert checked == pytest.approx(expected)


def test_a_cycle_of_unary_rules_converges_to_the_charts_fixed_point() -> None:
    agenda, checked = _both(MODAL, "M", ["kim", "runs"])
    # Each bounded rule fires with weight 1/4, so an item and its Dia
    # each carry the series sum_k (1/16)^k = 16/15 around the cycle: once
    # for each word and once for the sentence.
    expected = 3 * math.log(16 / 15)
    assert agenda == pytest.approx(expected, abs=1e-4)
    assert checked == pytest.approx(expected, abs=1e-4)
