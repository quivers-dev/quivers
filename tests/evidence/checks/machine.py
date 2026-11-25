"""Checks of the reference machine's deductions against enumeration."""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache, partial

from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import deduction_item, run_deduction
from tests.evidence.checks._registry import add
from tests.evidence.compare import ELEMENTARY, Accuracy, Mismatch, compare_number
from tests.evidence.oracles import derivations
from tests.evidence.oracles.derivations import RuleRow, Tree, Variable
from tests.evidence.problems import logarithm
from tests.evidence.sources import program

JOIN = Fraction(2, 5)
"""The weight of the binary rule."""

LIFT = Fraction(3, 4)
"""The weight of the unary rule."""

SENTENCES = (("a",), ("b",), ("a", "b"), ("a", "b", "a"))
"""Sentences of one to three words."""

SEMIRINGS = ("LogProb", "Viterbi", "Counting", "Boolean")
"""The semirings of a deduction."""

#: A path through one derivation multiplies at most this many factors, and
#: the search adds one alternative per derivation.
ROUNDINGS = 4 * ELEMENTARY + 8


def module_of(name: str, **values: str) -> QiecModule:
    """The checked module of a source.

    Parameters
    ----------
    name : str
        The source's name.
    **values : str
        Replacements in the source.

    Returns
    -------
    QiecModule
        The module.
    """
    module = program(name, **values).qiec
    assert module is not None
    return module


@lru_cache(maxsize=None)
def parses(sentence: tuple[str, ...]) -> tuple[Tree, ...]:
    """Every derivation of ``S`` over a sentence.

    Parameters
    ----------
    sentence : tuple of str
        The words.

    Returns
    -------
    tuple of Tree
        The derivations.
    """
    lexicon = (("a", "S"), ("b", "V"), ("a", "V"))
    axioms = [
        (f"{i}:{category}", ("span", i, i + 1, category))
        for i, word in enumerate(sentence)
        for entry, category in lexicon
        if entry == word
    ]
    i, k, j = Variable("i"), Variable("k"), Variable("j")
    rules: list[RuleRow] = [
        ("join", (("span", i, k, "S"), ("span", k, j, "S")), ("span", i, j, "S"), "join"),
        ("lift", (("span", i, j, "V"),), ("span", i, j, "S"), "lift"),
    ]
    return tuple(
        derivations.derivations(axioms, rules, ("span", 0, len(sentence), "S"))
    )


def lexical_reference(sentence: tuple[str, ...], semiring: str) -> float:
    """The weight of a sentence from its derivations.

    Parameters
    ----------
    sentence : tuple of str
        The words.
    semiring : str
        The semiring's name.

    Returns
    -------
    float
        The weight.
    """
    trees = parses(sentence)
    if semiring in ("Counting", "Boolean"):
        return float(len(trees) if semiring == "Counting" else bool(trees))
    weights = {"join": JOIN, "lift": LIFT}
    weights |= {f"{i}:{c}": Fraction(1) for i in range(len(sentence)) for c in "SV"}
    if semiring == "Viterbi":
        return logarithm(derivations.best(trees, weights))
    return logarithm(derivations.total(trees, weights, Fraction(0), Fraction(1)))


def check_lexical(semiring: str) -> None:
    """Compare the machine's weight of a sentence with its derivations.

    Parameters
    ----------
    semiring : str
        The semiring's name.
    """
    module = module_of("lexical", SEMIRING=semiring)
    for sentence in SENTENCES:
        value = run_deduction(
            module,
            "G",
            tokens=sentence,
            parameters={
                "G.rule.join:": logarithm(JOIN),
                "G.rule.lift:": logarithm(LIFT),
            },
        ).weight
        assert isinstance(value, (bool, int, float))
        compare_number(
            float(value),
            lexical_reference(sentence, semiring),
            Accuracy(
                domain="log",
                operations=ROUNDINGS * max(1, len(parses(sentence))),
                magnitude=8.0,
            ),
            f"machine {semiring} {sentence}",
        )


@lru_cache(maxsize=None)
def nonlexical_trees(occurrences: int) -> tuple[Tree, ...]:
    """Every derivation of ``C`` from copies of ``A``.

    Parameters
    ----------
    occurrences : int
        The number of copies.

    Returns
    -------
    tuple of Tree
        The derivations.
    """
    rules: list[RuleRow] = [
        ("first", (("A",),), ("B",), "first"),
        ("second", (("B",), ("A",)), ("C",), "second"),
        ("direct", (("A",),), ("C",), "direct"),
        ("twice", (("B",), ("B",)), ("C",), "twice"),
    ]
    axioms = [(f"axiom{i}", ("A",)) for i in range(occurrences)]
    return tuple(derivations.derivations(axioms, rules, ("C",)))


#: The rule weights of the deduction without a lexicon.
NONLEXICAL_WEIGHTS = {
    "first": Fraction(1, 3),
    "second": Fraction(5, 7),
    "direct": Fraction(1, 8),
    "twice": Fraction(9, 10),
}
#: The weights of the copies of the axiom.
AXIOM_WEIGHTS = (Fraction(2, 3), Fraction(1, 6))


def check_nonlexical(semiring: str) -> None:
    """Compare the machine's chart with the derivations from two equal axioms.

    Parameters
    ----------
    semiring : str
        The semiring's name.
    """
    module = module_of("nonlexical", SEMIRING=semiring)
    trees = nonlexical_trees(2)
    weights = {
        **NONLEXICAL_WEIGHTS,
        **{f"axiom{i}": weight for i, weight in enumerate(AXIOM_WEIGHTS)},
    }
    entries: list[bool | int | float]
    if semiring == "Boolean":
        entries, wanted = [True, True], 1.0
    elif semiring == "Counting":
        entries, wanted = [1, 1], float(len(trees))
    else:
        entries = [logarithm(weight) for weight in AXIOM_WEIGHTS]
        wanted = logarithm(
            derivations.best(trees, weights)
            if semiring == "Viterbi"
            else derivations.total(trees, weights, Fraction(0), Fraction(1))
        )
    value = run_deduction(
        module,
        "W",
        axioms=[deduction_item(module, "W", "A")] * 2,
        axiom_weights=entries,
        parameters={
            f"W.rule.{name}:": logarithm(weight)
            for name, weight in NONLEXICAL_WEIGHTS.items()
        },
        fuel=200_000,
    ).weight
    assert isinstance(value, (bool, int, float))
    compare_number(
        float(value),
        wanted,
        Accuracy(domain="log", operations=ROUNDINGS * len(trees), magnitude=8.0),
        f"machine chart {semiring}",
    )


def check_boolean_cycle(name: str) -> None:
    """Require a Boolean cycle to settle on true.

    Parameters
    ----------
    name : str
        ``"cycle"`` or ``"chained"``.
    """
    module = module_of(name, SEMIRING="Boolean", TOLERANCE="")
    if name == "cycle":
        value = run_deduction(
            module,
            "Cycle",
            axioms=[deduction_item(module, "Cycle", "A")],
            axiom_weights=[True],
            fuel=10_000,
        ).weight
    else:
        value = run_deduction(module, "M", tokens=["a"]).weight
    assert isinstance(value, bool)
    compare_number(
        float(value), 1, Accuracy(domain="linear", operations=1), f"machine {name}"
    )


def check_budget(name: str) -> None:
    """Require a refusal when a count does not settle within its budget.

    Parameters
    ----------
    name : str
        ``"cycle"`` or ``"chained"``.

    Raises
    ------
    Mismatch
        If the run returns a number or the diagnostic differs.
    """
    module = module_of(name, SEMIRING="Counting", TOLERANCE="")
    wanted = "did not finish within 2000 steps"
    try:
        if name == "cycle":
            value = run_deduction(
                module,
                "Cycle",
                axioms=[deduction_item(module, "Cycle", "A")],
                axiom_weights=[1],
                fuel=2000,
            ).weight
        else:
            value = run_deduction(module, "M", tokens=["a"], fuel=2000).weight
    except ExecutionFailure as refusal:
        if wanted in str(refusal):
            return
        raise Mismatch("refusal", math.inf, str(refusal)) from refusal
    raise Mismatch("refusal", math.inf, f"a budget of 2000 steps returned {value!r}")


def check_wrong_input(name: str) -> None:
    """Require an input of the wrong kind to be refused by name.

    Parameters
    ----------
    name : str
        ``"lexical"`` or ``"nonlexical"``.

    Raises
    ------
    Mismatch
        If the run returns or the diagnostic differs.
    """
    module = module_of(name, SEMIRING="Counting")
    try:
        if name == "lexical":
            run_deduction(module, "G", axioms=[], axiom_weights=[])
            wanted = ""
        else:
            run_deduction(module, "W", tokens=["a"])
            wanted = ""
    except ValueError as refusal:
        wanted = (
            "deduction 'G' takes a sentence of tokens"
            if name == "lexical"
            else "deduction 'W' takes its axioms and their weights"
        )
        if str(refusal) == wanted:
            return
        raise Mismatch("refusal", math.inf, str(refusal)) from refusal
    raise Mismatch("refusal", math.inf, f"{name} accepted the other kind of input")


def check_normalized() -> None:
    """Require the productions of each left-hand side to sum to one."""
    table = program("normalized").deductions["G"].production_log_probs()
    if not table:
        raise Mismatch("support", math.inf, "no production group")
    for label, group in table.items():
        mass = math.fsum(float(value.detach().exp()) for value in group.values())
        compare_number(
            mass,
            1,
            # The weights are stored in float32.
            Accuracy(
                domain="linear",
                operations=ELEMENTARY * (2 + len(group)),
                magnitude=1.0,
                dtype="float32",
            ),
            f"mass of {label}",
        )


def _register() -> None:
    """Register every check of this module."""
    for semiring in SEMIRINGS:
        question = {"Boolean": "Q2", "Viterbi": "Q4", "Counting": "Q10"}.get(
            semiring, "Q3"
        )
        add(
            f"machine/lexical/{semiring}",
            "machine-lexical",
            question,
            partial(check_lexical, semiring),
        )
        add(
            f"machine/nonlexical/{semiring}",
            "machine-nonlexical",
            "Q1" if semiring == "LogProb" else question,
            partial(check_nonlexical, semiring),
        )
    for name, row in (("cycle", "machine-nonlexical"), ("chained", "machine-chained")):
        add(f"machine/boolean_cycle/{name}", row, "Q2", partial(check_boolean_cycle, name))
        add(f"machine/budget/{name}", row, "Q8", partial(check_budget, name))
    for name, row in (("lexical", "machine-lexical"), ("nonlexical", "machine-nonlexical")):
        add(f"machine/wrong_input/{name}", row, "Q24", partial(check_wrong_input, name))
    add("machine/normalized", "machine-lexical", "Q14", check_normalized)


_register()
