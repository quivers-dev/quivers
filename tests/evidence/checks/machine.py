"""Checks of the reference machine's deductions against enumeration."""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache, partial

import torch
from quivers.dsl.compiler.deductions import BASE_WEIGHT_KEY

from quivers.qiec.contracts import (
    ContractField,
    ContractNotMet,
    ControlReached,
    FailurePolicy,
    Unestablished,
)
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import DeductionRun, deduction_item, run_deduction
from tests.evidence.checks._registry import add
from tests.evidence.compare import ELEMENTARY, Accuracy, Mismatch, compare_number
from tests.evidence.oracles import closed_forms, derivations
from tests.evidence.oracles.derivations import RuleRow, Tree, Variable
from tests.evidence.problems import logarithm
from quivers.dsl import loads
from tests.evidence.sources import SOURCES, deduction, program

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
        (
            "join",
            (("span", i, k, "S"), ("span", k, j, "S")),
            ("span", i, j, "S"),
            "join",
        ),
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
        entries, wanted = list[bool | int | float]([True, True]), 1.0
    elif semiring == "Counting":
        entries, wanted = list[bool | int | float]([1, 1]), float(len(trees))
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


#: The turn of the cycles whose rounded iterates stop changing within the
#: budget of a fast check.
ROUNDED_TURN = Fraction(1, 16)
#: The tolerances the residual checks declare, by source; each stops the
#: iteration before its weight reaches the closed form.
RESIDUAL_TOLERANCES = {"cycle": "0.01", "chained": "0.1"}


def residual_run(
    name: str, tolerance: str, policy: FailurePolicy, turn: Fraction = Fraction(1, 2)
) -> DeductionRun:
    """Run a cycle under a declared tolerance.

    Parameters
    ----------
    name : str
        ``"cycle"`` (a chart) or ``"chained"`` (a search).
    tolerance : str
        The declared tolerance, or empty for none.
    policy : FailurePolicy
        What a stop by a control does.
    turn : Fraction
        The weight of one turn of the cycle.

    Returns
    -------
    DeductionRun
        The run.
    """
    module = module_of(
        name,
        SEMIRING="LogProb",
        TOLERANCE=f", tolerance={tolerance}" if tolerance else "",
    )
    if name == "cycle":
        return run_deduction(
            module,
            "Cycle",
            axioms=[deduction_item(module, "Cycle", "A")],
            axiom_weights=[0.0],
            parameters={"Cycle.rule.loop:": logarithm(turn)},
            fuel=300_000,
            on_failure=policy,
        )
    return run_deduction(
        module,
        "M",
        tokens=["a"],
        parameters={"M.rule.there:": logarithm(turn), "M.rule.back:": 0.0},
        fuel=300_000,
        on_failure=policy,
    )


def _claimed(value: object, what: str) -> Mismatch:
    """The mismatch of a stop by a control that returned a weight.

    Parameters
    ----------
    value : object
        What the run returned.
    what : str
        The run.

    Returns
    -------
    Mismatch
        A value mismatch whose deviation is the distance from the closed
        form, infinite for a value that is not a number.
    """
    exact = logarithm(closed_forms.geometric(Fraction(1, 2)))
    deviation = abs(value - exact) if isinstance(value, float) else math.inf
    return Mismatch("value", deviation, f"{what} returned {value!r} as its weight")


def check_residual(name: str) -> None:
    """Require a stop at a tolerance to be refused or held as unestablished.

    Parameters
    ----------
    name : str
        ``"cycle"`` or ``"chained"``.

    Raises
    ------
    Mismatch
        If either policy returns the stopped iterate as a weight, or the
        report does not name the control and the unmet guarantee.
    """
    tolerance = RESIDUAL_TOLERANCES[name]
    try:
        refused = residual_run(name, tolerance, FailurePolicy.REFUSE).weight
    except ContractNotMet as refusal:
        if refusal.report.unmet != (ContractField.GUARANTEE,):
            raise Mismatch("refusal", math.inf, str(refusal)) from refusal
    else:
        raise _claimed(refused, f"machine {name} refusing")
    held = residual_run(name, tolerance, FailurePolicy.RETURN_UNESTABLISHED).weight
    if not isinstance(held, Unestablished):
        raise _claimed(held, f"machine {name} returning unestablished values")
    stop = held.report.stop
    if not isinstance(stop, ControlReached) or stop.value.fraction > Fraction(
        tolerance
    ):
        raise Mismatch("refusal", math.inf, f"machine {name} reported {stop!r}")
    assert isinstance(held.value, float)
    compare_number(
        held.value,
        # The stopped iterate is below the least fixed point.
        min(held.value, logarithm(closed_forms.geometric(Fraction(1, 2)))),
        Accuracy(domain="log", operations=1),
        f"machine {name} iterate below the closed form",
    )


def check_rounded_agreement(name: str) -> None:
    """Require two iterates agreeing in rounded arithmetic to certify nothing.

    Without a tolerance the cycle stops when the rounded weights stop
    changing while derivations keep growing; the value is within rounding
    of the closed form but no argument shows it, so it is unestablished.

    Parameters
    ----------
    name : str
        ``"cycle"`` or ``"chained"``.

    Raises
    ------
    Mismatch
        If the run returns a weight, or the held value is not the closed
        form within rounding.
    """
    held = residual_run(
        name, "", FailurePolicy.RETURN_UNESTABLISHED, ROUNDED_TURN
    ).weight
    if not isinstance(held, Unestablished):
        raise _claimed(held, f"machine {name} without a tolerance")
    assert isinstance(held.value, float)
    compare_number(
        held.value,
        logarithm(closed_forms.geometric(ROUNDED_TURN)),
        Accuracy(domain="log", operations=ROUNDINGS, magnitude=1.0),
        f"machine {name} stagnated iterate",
    )


def check_renamed_handlers() -> None:
    """Require a deduction's handlers to keep their meaning under any names.

    Raises
    ------
    Mismatch
        If renaming the handlers changes or loses the weight.
    """
    for semiring in SEMIRINGS:
        module = module_of("lexical", SEMIRING=semiring)
        renamed = module.with_(
            handlers=tuple(
                handler.with_(name=f"renamed{index}")
                for index, handler in enumerate(module.handlers)
            )
        )
        weights = [
            run_deduction(
                candidate,
                "G",
                tokens=("a", "b"),
                parameters={
                    "G.rule.join:": logarithm(JOIN),
                    "G.rule.lift:": logarithm(LIFT),
                },
            ).weight
            for candidate in (module, renamed)
        ]
        if weights[0] != weights[1]:
            raise Mismatch(
                "value", math.inf, f"renaming {semiring} handlers gave {weights}"
            )


#: The weights of the rules of the deduction with a judgment beside spans.
JUDGMENT_WEIGHTS = {
    "found": Fraction(1, 2),
    "after": Fraction(2, 3),
    "lift": Fraction(1, 5),
    "join": Fraction(3, 4),
}

#: Sentences of that deduction, with and without the ``b`` its spans rest on.
JUDGMENT_SENTENCES = (("a", "b"), ("b", "a", "a"), ("a", "a", "b", "b"), ("a",), ("b",))


def judgment_trees(sentence: tuple[str, ...]) -> tuple[Tree, ...]:
    """Every derivation of the sentence's ``S`` span.

    Parameters
    ----------
    sentence : tuple of str
        The words.

    Returns
    -------
    tuple of Tree
        The derivations.
    """
    categories = {"a": "A", "b": "B"}
    axioms = [
        (f"{i}:{categories[word]}", ("span", i, i + 1, categories[word]))
        for i, word in enumerate(sentence)
    ]
    i, k, j = Variable("I"), Variable("K"), Variable("J")
    rules: list[RuleRow] = [
        ("found", (("span", i, j, "B"),), "seen", "found"),
        ("after", ("seen", ("span", i, j, "A")), ("span", i, j, "S"), "after"),
        ("lift", (("span", i, j, "B"),), ("span", i, j, "S"), "lift"),
        (
            "join",
            (("span", i, k, "S"), ("span", k, j, "S")),
            ("span", i, j, "S"),
            "join",
        ),
    ]
    return tuple(
        derivations.derivations(axioms, rules, ("span", 0, len(sentence), "S"))
    )


def check_judgments(semiring: str, fresh: bool = False) -> None:
    """Compare both engines with the derivations of a deduction beyond spans.

    The item ``seen`` holds no position, so no search by spans or by
    yields bounds the deduction; its chart computes it.

    Parameters
    ----------
    semiring : str
        The semiring's name.
    fresh : bool
        Whether to compile the source again rather than reuse its compiled
        program, so that a change of the elaboration is seen.

    Raises
    ------
    Mismatch
        If the machine refuses, or the machine or the agenda differs from
        the derivations.
    """
    compiled = (
        loads(SOURCES["judgments"].replace("SEMIRING", semiring))
        if fresh
        else program("judgments", SEMIRING=semiring)
    )
    module = compiled.qiec
    assert module is not None
    system = deduction(compiled, "J")
    parameters = {
        f"J.rule.{name}:": logarithm(weight)
        for name, weight in JUDGMENT_WEIGHTS.items()
    }
    for name, weight in JUDGMENT_WEIGHTS.items():
        for key, value in system.modules["_rule_module"].named_parameters():
            if key == f"rule_{name}.{BASE_WEIGHT_KEY}":
                with torch.no_grad():
                    value.fill_(logarithm(weight))
    for sentence in JUDGMENT_SENTENCES:
        trees = judgment_trees(sentence)
        weights = {**JUDGMENT_WEIGHTS}
        weights |= {f"{i}:{c}": Fraction(1) for i in range(len(sentence)) for c in "AB"}
        if semiring == "Counting":
            wanted = float(len(trees))
        elif semiring == "Boolean":
            wanted = float(bool(trees))
        elif semiring == "Viterbi":
            wanted = logarithm(derivations.best(trees, weights)) if trees else -math.inf
        else:
            wanted = logarithm(
                derivations.total(trees, weights, Fraction(0), Fraction(1))
            )
        try:
            machine = run_deduction(
                module, "J", tokens=sentence, parameters=parameters, fuel=2_000_000
            ).weight
        except ContractNotMet as refusal:
            raise Mismatch(
                "refusal", math.inf, f"machine {semiring} {sentence}: {refusal}"
            ) from refusal
        assert isinstance(machine, (bool, int, float))
        agenda = float(system(list(sentence)).goal_weight().detach())
        accuracy = Accuracy(
            domain="log", operations=ROUNDINGS * max(1, len(trees)), magnitude=8.0
        )
        compare_number(
            float(machine), wanted, accuracy, f"machine {semiring} {sentence}"
        )
        compare_number(
            agenda,
            wanted,
            # The agenda's rule weights are stored in float32.
            Accuracy(
                domain="log",
                operations=ROUNDINGS * max(1, len(trees)),
                magnitude=8.0,
                dtype="float32",
            ),
            f"agenda {semiring} {sentence}",
        )


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
    """Require the productions of each left-hand side to sum to one.

    Raises
    ------
    Mismatch
        If the result differs from its reference.
    """
    table = deduction(program("normalized"), "G").production_log_probs()
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
        add(
            f"machine/boolean_cycle/{name}",
            row,
            "Q2",
            partial(check_boolean_cycle, name),
        )
        add(f"machine/budget/{name}", row, "Q8", partial(check_budget, name))
        add(f"machine/residual/{name}", row, "Q8", partial(check_residual, name))
        add(
            f"machine/rounded_agreement/{name}",
            row,
            "Q8",
            partial(check_rounded_agreement, name),
        )
    for name, row in (
        ("lexical", "machine-lexical"),
        ("nonlexical", "machine-nonlexical"),
    ):
        add(f"machine/wrong_input/{name}", row, "Q24", partial(check_wrong_input, name))
    add("machine/normalized", "machine-lexical", "Q14", check_normalized)
    for semiring in SEMIRINGS:
        add(
            f"machine/judgments/{semiring}",
            "machine-lexical",
            "Q1",
            partial(check_judgments, semiring),
        )
    add(
        "machine/judgments/compiled",
        "machine-lexical",
        "Q4",
        partial(check_judgments, "Viterbi", fresh=True),
    )
    add("machine/renamed_handlers", "machine-lexical", "Q3", check_renamed_handlers)


_register()
