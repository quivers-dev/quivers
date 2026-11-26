"""Checks of the span parsers against enumerated parse trees."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from fractions import Fraction
from functools import lru_cache, partial

import torch

from quivers.core.morphisms import observed
from quivers.core.objects import FinSet, ProductSet
from quivers.stochastic._kleene import DivergentClosureError
from quivers.stochastic._rule_system import RuleSystem
from quivers.stochastic.inside import InsideAlgorithm
from quivers.stochastic.parsers import ChartParser
from quivers.stochastic.semiring import LOG_PROB, VITERBI, ChartSemiring
from quivers.stochastic.span import (
    BinarySpanDeduction,
    LexicalAxiom,
    UnarySpanDeduction,
)
from tests.evidence.checks._registry import add
from tests.evidence.counters import saved_elements
from tests.evidence.compare import (
    differentiate,
    ELEMENTARY,
    Accuracy,
    Mismatch,
    compare_gradients,
    compare_tensor,
)
from tests.evidence.oracles import derivations, exact
from tests.evidence.oracles.derivations import Tree, Variable
from tests.evidence.problems import logarithm

CATEGORIES = 3
"""Categories of the grammars."""

TERMINALS = 2
"""Terminals of the grammars."""

#: Weight of ``a -> b c`` at ``(a, b, c)``.
BINARY = {
    (0, 1, 2): "1/2",
    (0, 0, 0): "1/5",
    (1, 2, 1): "3/8",
    (2, 1, 1): "1/4",
    (1, 0, 2): "1/3",
}
#: Weight of ``a -> t`` at ``(a, t)``; the weights of one terminal sum to one.
LEXICAL = {
    (0, 0): "1/4",
    (1, 0): "1/2",
    (2, 0): "1/4",
    (0, 1): "1/8",
    (1, 1): "1/8",
    (2, 1): "3/4",
}
#: Weight of ``a -> b`` at ``(a, b)``, without a cycle.
UNARY = {(0, 1): "1/3", (1, 2): "1/6", (0, 2): "1/7"}
#: Weight of ``a -> b`` at ``(a, b)``, with a cycle.
CYCLIC = {(0, 1): "1/3", (1, 0): "1/4", (2, 2): "1/5"}

SENTENCES = ((0,), (0, 1), (1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 0, 1))
"""Sentences of one to five words."""

#: The longest sentence whose trees with unary rules are listed; a longer
#: one is summed by the rational recurrence, which the listed ones check.
ENUMERATED = 3

PARSERS: dict[str, ChartSemiring] = {"logprob": LOG_PROB, "viterbi": VITERBI}
"""The semirings a chart parser is checked in."""


@contextmanager
def double_precision() -> Iterator[None]:
    """Make float64 the default type.

    Yields
    ------
    None
        Inside, new tensors are float64.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def fractions[K: tuple[int, ...]](weights: Mapping[K, str]) -> dict[K, Fraction]:
    """Read rational strings.

    Parameters
    ----------
    weights : Mapping[K, str]
        The strings.

    Returns
    -------
    dict of K to Fraction
        The rationals.
    """
    return {key: Fraction(value) for key, value in weights.items()}


@lru_cache(maxsize=None)
def parses(sentence: tuple[int, ...], unary: bool) -> tuple[Tree, ...]:
    """Every parse tree of a sentence.

    Parameters
    ----------
    sentence : tuple of int
        The terminals.
    unary : bool
        Whether the acyclic unary rules are used.

    Returns
    -------
    tuple of Tree
        The trees of category zero over the sentence.
    """
    axioms = [
        (f"{i}:{a}", ("span", a, i, i + 1))
        for i, token in enumerate(sentence)
        for a, t in LEXICAL
        if t == token
    ]
    rules: list[derivations.RuleRow] = [
        (
            f"binary{a}{b}{c}",
            (
                ("span", b, Variable("i"), Variable("k")),
                ("span", c, Variable("k"), Variable("j")),
            ),
            ("span", a, Variable("i"), Variable("j")),
            f"b{a}{b}{c}",
        )
        for a, b, c in BINARY
    ]
    if unary:
        rules += [
            (
                f"unary{a}{b}",
                (("span", b, Variable("i"), Variable("j")),),
                ("span", a, Variable("i"), Variable("j")),
                f"u{a}{b}",
            )
            for a, b in UNARY
        ]
    return tuple(derivations.derivations(axioms, rules, ("span", 0, 0, len(sentence))))


def weights_of(sentence: tuple[int, ...]) -> dict[str, Fraction]:
    """The weight of every name a parse tree uses.

    Parameters
    ----------
    sentence : tuple of int
        The terminals.

    Returns
    -------
    dict of str to Fraction
        The weights.
    """
    answer = {f"b{a}{b}{c}": Fraction(w) for (a, b, c), w in BINARY.items()}
    answer |= {f"u{a}{b}": Fraction(w) for (a, b), w in UNARY.items()}
    for i, token in enumerate(sentence):
        for (a, t), weight in LEXICAL.items():
            if t == token:
                answer[f"{i}:{a}"] = Fraction(weight)
    return answer


def closure_condition(unary: Mapping[tuple[int, int], str]) -> float:
    """The condition number of a unary closure.

    Parameters
    ----------
    unary : Mapping[tuple of (int, int), str]
        The unary rules.

    Returns
    -------
    float
        The largest row sum of ``(I - U)^-1``.
    """
    weights = fractions(unary)
    star = exact.linear_star(
        [
            [weights.get((b, a), Fraction(0)) for b in range(CATEGORIES)]
            for a in range(CATEGORIES)
        ]
    )
    return max(float(sum(row)) for row in star)


def span_accuracy(length: int, unary: Mapping[tuple[int, int], str]) -> Accuracy:
    """The accuracy of a span recurrence.

    A path from a word to the sentence passes at most ``2 n`` cells. Each
    cell multiplies three factors, sums over splits and rules with one
    maximum, one exponential per term, one logarithm, and the additions,
    and closes under at most ``C`` categories.

    Parameters
    ----------
    length : int
        The sentence length.
    unary : Mapping[tuple of (int, int), str]
        The unary rules, whose closure's condition number multiplies the
        error of every cell it closes.

    Returns
    -------
    Accuracy
        The accuracy.
    """
    terms = length * len(BINARY) + CATEGORIES
    per_cell = 3 + 4 * ELEMENTARY + 2 * terms + CATEGORIES * (5 * ELEMENTARY + 5)
    return Accuracy(
        domain="log",
        operations=2 * length * per_cell,
        magnitude=4.0 * length,
        condition=closure_condition(unary),
    )


def chart_parser(
    semiring: ChartSemiring, unary: Mapping[tuple[int, int], str]
) -> ChartParser:
    """A chart parser with the module's weights, in float64.

    Parameters
    ----------
    semiring : ChartSemiring
        The semiring.
    unary : Mapping[tuple of (int, int), str]
        The unary rules.

    Returns
    -------
    ChartParser
        The parser.
    """
    parser = ChartParser(
        RuleSystem(
            binary_rules=tuple(BINARY),
            unary_rules=tuple(unary),
            n_categories=CATEGORIES,
        ),
        n_terminals=TERMINALS,
        start_idx=0,
        semiring=semiring,
    ).double()
    axiom, (binary, chains) = parser.axiom, parser.deductions
    assert isinstance(axiom, LexicalAxiom)
    assert isinstance(binary, BinarySpanDeduction)
    assert isinstance(chains, UnarySpanDeduction)
    with torch.no_grad():
        for (a, t), weight in LEXICAL.items():
            axiom.lexicon_logits[t, a] = logarithm(Fraction(weight))
        binary.weights.copy_(
            torch.tensor([logarithm(Fraction(w)) for w in BINARY.values()])
        )
        if unary:
            chains.weights.copy_(
                torch.tensor([logarithm(Fraction(w)) for w in unary.values()])
            )
    return parser


def inside_algorithm(
    unary: Mapping[tuple[int, int], str], *, requires_grad: bool = False
) -> tuple[InsideAlgorithm, dict[str, torch.Tensor]]:
    """An inside algorithm with the module's weights, in float64.

    Parameters
    ----------
    unary : Mapping[tuple of (int, int), str]
        The unary rules.
    requires_grad : bool
        Whether the rule tensors are differentiated.

    Returns
    -------
    tuple of (InsideAlgorithm, dict)
        The algorithm and its rule tensors.
    """
    categories = FinSet(name="N", cardinality=CATEGORIES)
    terminals = FinSet(name="T", cardinality=TERMINALS)
    tensors = {
        "binary": torch.zeros(CATEGORIES, CATEGORIES, CATEGORIES, dtype=torch.float64),
        "lexical": torch.zeros(CATEGORIES, TERMINALS, dtype=torch.float64),
        "unary": torch.zeros(CATEGORIES, CATEGORIES, dtype=torch.float64),
    }
    for key, weight in BINARY.items():
        tensors["binary"][key] = float(Fraction(weight))
    for key, weight in LEXICAL.items():
        tensors["lexical"][key] = float(Fraction(weight))
    for (a, b), weight in unary.items():
        tensors["unary"][b, a] = float(Fraction(weight))
    if requires_grad:
        for tensor in tensors.values():
            tensor.requires_grad_()
    pair = ProductSet(components=(categories, categories))
    return (
        InsideAlgorithm(
            observed(categories, pair, tensors["binary"]),
            observed(categories, terminals, tensors["lexical"]),
            start=0,
            unary=observed(categories, categories, tensors["unary"]) if unary else None,
        ),
        tensors,
    )


def expected(sentence: tuple[int, ...], semiring: str, unary: str) -> float:
    """The log weight of a sentence.

    Parameters
    ----------
    sentence : tuple of int
        The terminals.
    semiring : str
        ``"logprob"`` or ``"viterbi"``.
    unary : str
        ``"none"``, ``"acyclic"``, or ``"cyclic"``.

    Returns
    -------
    float
        The log weight.
    """
    if unary == "cyclic" or (
        len(sentence) > ENUMERATED and (unary, semiring) == ("acyclic", "logprob")
    ):
        return logarithm(
            exact.inside(
                fractions(BINARY),
                fractions(LEXICAL),
                fractions(rules_named(unary)),
                sentence,
                CATEGORIES,
            )[0]
        )
    trees = parses(sentence, unary == "acyclic")
    weights = weights_of(sentence)
    if semiring == "viterbi":
        return logarithm(derivations.best(trees, weights))
    return logarithm(derivations.total(trees, weights, Fraction(0), Fraction(1)))


def rules_named(unary: str) -> Mapping[tuple[int, int], str]:
    """The unary rules under a name.

    Parameters
    ----------
    unary : str
        ``"none"``, ``"acyclic"``, or ``"cyclic"``.

    Returns
    -------
    Mapping[tuple of (int, int), str]
        The rules.
    """
    return {"none": {}, "acyclic": UNARY, "cyclic": CYCLIC}[unary]


def check_parser(semiring: str, unary: str) -> None:
    """Compare a chart parser's scores with the sums over parse trees.

    Parameters
    ----------
    semiring : str
        The semiring's name.
    unary : str
        The unary rules' name.
    """
    with double_precision():
        parser = chart_parser(PARSERS[semiring], rules_named(unary))
        for sentence in SENTENCES:
            if len(sentence) > ENUMERATED and semiring == "viterbi" and unary != "none":
                continue
            compare_tensor(
                parser(torch.tensor(sentence)),
                torch.tensor(expected(sentence, semiring, unary)),
                span_accuracy(len(sentence), rules_named(unary)),
                f"chart parser {semiring} {unary} {sentence}",
            )


def check_parser_batch() -> None:
    """Compare a batch of sentences with one sentence at a time."""
    with double_precision():
        parser = chart_parser(LOG_PROB, UNARY)
        rows = ((0, 1, 1), (1, 0, 1), (1, 1, 1))
        compare_tensor(
            parser(torch.tensor(rows)),
            torch.tensor([expected(row, "logprob", "acyclic") for row in rows]),
            span_accuracy(3, UNARY),
            "chart parser batch",
        )


def check_parser_gradient() -> None:
    """Compare a chart parser's gradient with expected rule counts.

    A lexical logit enters through a normalized exponential, so its
    derivative is the expected count of the entry minus its probability
    times the expected count of the terminal.
    """
    sentence = (0, 1, 1)
    trees = parses(sentence, True)
    weights = weights_of(sentence)

    def count(name: str) -> Fraction:
        """The expected count of one name.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        Fraction
            The count.
        """
        return derivations.logarithmic_gradient(trees, weights, name)

    with double_precision():
        parser = chart_parser(LOG_PROB, UNARY)
        axiom, (binary, chains) = parser.axiom, parser.deductions
        assert isinstance(axiom, LexicalAxiom)
        assert isinstance(binary, BinarySpanDeduction)
        assert isinstance(chains, UnarySpanDeduction)
        found = differentiate(
            parser(torch.tensor(sentence)),
            [binary.weights, chains.weights, axiom.lexicon_logits],
            "chart parser",
        )
    gradients: dict[str, torch.Tensor | None] = {}
    reference: dict[str, Fraction] = {}
    for position, (a, b, c) in enumerate(BINARY):
        gradients[f"b{a}{b}{c}"] = found[0][position]
        reference[f"b{a}{b}{c}"] = count(f"b{a}{b}{c}")
    for position, (a, b) in enumerate(UNARY):
        gradients[f"u{a}{b}"] = found[1][position]
        reference[f"u{a}{b}"] = count(f"u{a}{b}")
    size = Fraction(0)
    for t in range(TERMINALS):
        uses = {
            a: sum(
                (count(f"{i}:{a}") for i, token in enumerate(sentence) if token == t),
                Fraction(0),
            )
            for a in range(CATEGORIES)
        }
        for a in range(CATEGORIES):
            gradients[f"l{t}{a}"] = found[2][t, a]
            whole = sum(uses.values(), Fraction(0))
            reference[f"l{t}{a}"] = uses[a] - Fraction(LEXICAL[(a, t)]) * whole
            size = max(size, uses[a] + whole)
    smallest = min(abs(value) for value in reference.values() if value != 0)
    forward = span_accuracy(len(sentence), UNARY)
    compare_gradients(
        gradients,
        reference,
        Accuracy(
            domain="linear",
            operations=5 * forward.operations,
            condition=forward.condition * forward.magnitude * float(size / smallest),
        ),
        "chart parser gradient",
    )


def check_inside(unary: str) -> None:
    """Compare the inside algorithm with the sums over parse trees.

    Parameters
    ----------
    unary : str
        The unary rules' name.
    """
    with double_precision():
        algorithm, _ = inside_algorithm(rules_named(unary))
        for sentence in SENTENCES:
            compare_tensor(
                algorithm(torch.tensor(sentence)),
                torch.tensor(expected(sentence, "logprob", unary)),
                span_accuracy(len(sentence), rules_named(unary)),
                f"inside {unary} {sentence}",
            )


def check_inside_gradient() -> None:
    """Compare the inside algorithm's gradient with expected rule counts."""
    sentence = (0, 1, 1)
    trees = parses(sentence, True)
    weights = weights_of(sentence)
    with double_precision():
        algorithm, tensors = inside_algorithm(UNARY, requires_grad=True)
        value = algorithm(torch.tensor(sentence))
        found = differentiate(value, list(tensors.values()), "inside")
    gradients: dict[str, torch.Tensor | None] = {}
    reference: dict[str, Fraction] = {}
    for (a, b, c), weight in BINARY.items():
        name = f"b{a}{b}{c}"
        gradients[name] = found[0][a, b, c]
        reference[name] = derivations.logarithmic_gradient(
            trees, weights, name
        ) / Fraction(weight)
    for (a, t), weight in LEXICAL.items():
        name = f"l{a}{t}"
        gradients[name] = found[1][a, t]
        reference[name] = sum(
            (
                derivations.logarithmic_gradient(trees, weights, f"{i}:{a}")
                for i, token in enumerate(sentence)
                if token == t
            ),
            Fraction(0),
        ) / Fraction(weight)
    for (a, b), weight in UNARY.items():
        name = f"u{a}{b}"
        gradients[name] = found[2][b, a]
        reference[name] = derivations.logarithmic_gradient(
            trees, weights, name
        ) / Fraction(weight)
    forward = span_accuracy(len(sentence), UNARY)
    compare_gradients(
        gradients,
        reference,
        Accuracy(
            domain="linear",
            operations=5 * forward.operations,
            condition=forward.condition * forward.magnitude,
        ),
        "inside gradient",
    )


def check_impossible_cell_gradient() -> None:
    """Compare a gradient where some cells of the chart are impossible.

    Category two has no binary rule, so it is impossible over every span
    longer than one word. The derivative in the weight of a rule is the
    expected number of its uses over the weight. A rule of weight zero is
    left to the ledger.
    """
    sentence = (0, 1, 0)
    with double_precision():
        algorithm, tensors = inside_algorithm({}, requires_grad=True)
        with torch.no_grad():
            tensors["binary"].zero_()
            tensors["binary"][0, 0, 1] = 0.5
            tensors["binary"][0, 1, 0] = 0.5
        value = algorithm(torch.tensor(sentence))
        (found,) = differentiate(value, [tensors["binary"]], "inside")
    axioms = [
        (f"{i}:{a}", ("span", a, i, i + 1))
        for i, token in enumerate(sentence)
        for a, t in LEXICAL
        if t == token
    ]
    i, k, j = Variable("i"), Variable("k"), Variable("j")
    rules: list[derivations.RuleRow] = [
        (
            f"b{a}{b}{c}",
            (("span", b, i, k), ("span", c, k, j)),
            ("span", a, i, j),
            f"b{a}{b}{c}",
        )
        for a, b, c in ((0, 0, 1), (0, 1, 0))
    ]
    trees = derivations.derivations(axioms, rules, ("span", 0, 0, 3))
    weights = {
        "b001": Fraction(1, 2),
        "b010": Fraction(1, 2),
        **{
            name: Fraction(
                LEXICAL[(int(name.split(":")[1]), sentence[int(name.split(":")[0])])]
            )
            for name, _ in axioms
        },
    }
    reference = {
        name: derivations.logarithmic_gradient(trees, weights, name) / weights[name]
        for name in ("b001", "b010")
    }
    gradients: dict[str, torch.Tensor | None] = {
        "b001": found[0, 0, 1],
        "b010": found[0, 1, 0],
    }
    forward = span_accuracy(len(sentence), {})
    compare_gradients(
        gradients,
        reference,
        Accuracy(
            domain="linear",
            operations=5 * forward.operations,
            condition=forward.condition * forward.magnitude,
        ),
        "inside gradient with impossible cells",
    )


def check_underivable_category_gradient() -> None:
    """Require every derivative of a parse to be a number.

    Category two of the grammar is derived by no rule, so every chart
    cell of it over a span longer than one word is a sum of no
    derivations. The sentence parses with positive weight, so its score
    is finite and its derivative in every parameter is a number.
    """
    with double_precision():
        torch.manual_seed(0)
        parser = ChartParser(
            RuleSystem(binary_rules=((0, 1, 2), (0, 0, 0)), n_categories=3),
            n_terminals=2,
            start_idx=0,
        )
        score = parser(torch.tensor((0, 1, 0)))
        if not math.isfinite(float(score)):
            raise Mismatch(
                "infinite", math.inf, f"chart parser score {float(score)} is not finite"
            )
        found = differentiate(score, list(parser.parameters()), "chart parser")
    for position, gradient in enumerate(found):
        if bool(torch.isnan(gradient).any()):
            raise Mismatch(
                "not-a-number",
                math.inf,
                f"chart parser: the derivative in parameter {position} is not a number",
            )


def check_inside_precision() -> None:
    """Require a float64 grammar to score and chart in float64.

    Raises
    ------
    Mismatch
        If the score or the chart has another dtype.
    """
    with double_precision():
        algorithm, _ = inside_algorithm({})
        tokens = torch.tensor((0, 1))
        score = algorithm(tokens)
        chart = algorithm.inside_chart(tokens)
    for what, value in (("score", score), ("chart", chart)):
        if value.dtype != torch.float64:
            raise Mismatch(
                "dtype", math.inf, f"inside {what} in {value.dtype} for torch.float64"
            )


def check_divergent_unary(executor: str) -> None:
    """Require a refusal for unary rules whose chains do not sum.

    Parameters
    ----------
    executor : str
        ``"chart-parser"`` or ``"inside"``.


    Raises
    ------
    Mismatch
        If a number is returned.
    """
    cycle = {(0, 1): "1", (1, 0): "1"}
    with double_precision():
        try:
            if executor == "inside":
                value = inside_algorithm(cycle)[0](torch.tensor([0]))
            else:
                value = chart_parser(LOG_PROB, cycle)(torch.tensor([0]))
        except DivergentClosureError:
            return
    raise Mismatch("refusal", math.inf, f"{executor} returned {float(value)}")


def check_empty_sentence(executor: str) -> None:
    """Require a refusal of a sentence without words, by its exact message.

    Parameters
    ----------
    executor : str
        ``"chart-parser"`` or ``"inside"``.


    Raises
    ------
    Mismatch
        If a number is returned or the message differs.
    """
    with double_precision():
        parse = (
            inside_algorithm({})[0]
            if executor == "inside"
            else chart_parser(LOG_PROB, {})
        )
        try:
            value = parse(torch.zeros(0, dtype=torch.long))
        except ValueError as refusal:
            if str(refusal) == "cannot parse empty sentences":
                return
            raise Mismatch("refusal", math.inf, str(refusal)) from refusal
    raise Mismatch("refusal", math.inf, f"{executor} returned {float(value)}")


def check_update() -> None:
    """Compare a chart parser's score before and after a parameter update."""
    sentence = (0, 1, 1)
    trees = parses(sentence, True)
    with double_precision():
        parser = chart_parser(LOG_PROB, UNARY)
        _, chains = parser.deductions
        assert isinstance(chains, UnarySpanDeduction)
        weights = weights_of(sentence)
        for factor in (Fraction(1), Fraction(3, 7)):
            with torch.no_grad():
                chains.weights.add_(logarithm(factor))
            for a, b in UNARY:
                weights[f"u{a}{b}"] *= factor
            compare_tensor(
                parser(torch.tensor(sentence)),
                torch.tensor(
                    logarithm(
                        derivations.total(trees, weights, Fraction(0), Fraction(1))
                    )
                ),
                span_accuracy(len(sentence), UNARY),
                f"chart parser after an update by {factor}",
            )


def _two_categories(
    binary: Mapping[tuple[int, int, int], float],
    lexical: Sequence[Sequence[float]],
    *,
    requires_grad: bool = False,
) -> tuple[InsideAlgorithm, torch.Tensor]:
    """An inside algorithm over two categories and two terminals.

    Parameters
    ----------
    binary : Mapping
        The probability of each binary rule.
    lexical : Sequence of Sequence of float
        ``lexical[a][t]``.
    requires_grad : bool
        Whether the binary tensor is differentiated.

    Returns
    -------
    tuple of InsideAlgorithm and torch.Tensor
        The algorithm and its binary tensor.
    """
    categories = FinSet(name="N", cardinality=2)
    terminals = FinSet(name="T", cardinality=2)
    rules = torch.zeros(2, 2, 2, dtype=torch.float64)
    for key, weight in binary.items():
        rules[key] = weight
    rules.requires_grad_(requires_grad)
    words = torch.tensor(lexical, dtype=torch.float64)
    return (
        InsideAlgorithm(
            observed(
                categories, ProductSet(components=(categories, categories)), rules
            ),
            observed(categories, terminals, words),
        ),
        rules,
    )


def check_impossible_sentence() -> None:
    """Require a sentence without a derivation to score ``-inf`` with derivative zero.

    The grammar derives only ``0 -> 1 1`` over words of category one, so
    a sentence starting with the second terminal has no derivation; its
    log weight is the log of zero, and as a constant of the parameters its
    derivative is zero, not a number of no meaning.

    Raises
    ------
    Mismatch
        If the score is finite or its derivative is not zero.
    """
    algorithm, rules = _two_categories(
        {(0, 1, 1): 1.0}, [[0.0, 0.0], [1.0, 0.0]], requires_grad=True
    )
    wanted = logarithm(
        exact.inside({(0, 1, 1): Fraction(1)}, {(1, 0): Fraction(1)}, {}, (0, 1), 2)[0]
    )
    score = algorithm(torch.tensor([0, 1]))
    if float(score.detach()) != wanted:
        raise Mismatch(
            "value", math.inf, f"impossible sentence scores {float(score.detach())}"
        )
    (found,) = torch.autograd.grad(score, rules)
    if not bool((found == 0).all()):
        raise Mismatch(
            "value", math.inf, f"its derivative is {found.flatten().tolist()}"
        )


def check_zero_rule_derivative() -> None:
    """Compare the derivative of a log score in a rule of weight zero with its exact value.

    The rule ``1 -> 1 1`` has weight zero and derives the second category
    over two of three words, so the sentence's weight is a polynomial of
    degree at most two in it; three values of the closed recurrence give
    its linear coefficient, and the derivative of the log weight is that
    over the weight at zero.

    Raises
    ------
    Mismatch
        If the derivative differs.
    """
    algorithm, rules = _two_categories(
        {(0, 1, 1): 1.0, (0, 0, 1): 0.5}, [[0.5, 0.0], [0.5, 0.0]], requires_grad=True
    )
    (found,) = torch.autograd.grad(algorithm(torch.tensor([0, 0, 0])), rules)
    values = [
        exact.inside(
            {
                (0, 1, 1): Fraction(1),
                (0, 0, 1): Fraction(1, 2),
                (1, 1, 1): Fraction(level),
            },
            {(0, 0): Fraction(1, 2), (1, 0): Fraction(1, 2)},
            {},
            (0, 0, 0),
            2,
        )[0]
        for level in (0, 1, 2)
    ]
    slope = 2 * (values[1] - values[0]) - (values[2] - values[0]) / 2
    compare_tensor(
        found[1, 1, 1],
        torch.tensor(float(slope / values[0]), dtype=torch.float64),
        Accuracy(domain="linear", operations=200),
        "derivative in a rule of weight zero",
    )


def inside_saved(length: int, batch: int = 2) -> int:
    """Elements the inside algorithm keeps for its backward pass.

    Parameters
    ----------
    length : int
        The sentence length.
    batch : int
        The sentences.

    Returns
    -------
    int
        Elements saved.
    """
    with double_precision():
        algorithm, tensors = inside_algorithm(UNARY, requires_grad=True)
        tokens = torch.tensor(
            [[(i + b) % TERMINALS for i in range(length)] for b in range(batch)]
        )
        with saved_elements() as sizes:
            score = algorithm(tokens).sum()
        torch.autograd.grad(score, list(tensors.values()))
    return sum(sizes)


def check_inside_memory() -> None:
    """Require the inside algorithm to keep only its chart and weights for backward.

    The chart has ``batch N n (n + 1) / 2`` entries; the weights ``N^3``,
    ``N T`` and the terminals ``batch n``. No product over categories and
    split points is kept.

    Raises
    ------
    Mismatch
        If more is saved.
    """
    for length in (3, 5):
        found = inside_saved(length)
        bound = (
            2 * CATEGORIES * length * (length + 1) // 2
            + CATEGORIES**3
            + CATEGORIES * TERMINALS
            + 2 * length
        )
        if found > bound:
            raise Mismatch(
                "work", found / bound, f"length {length}: {found} elements saved"
            )


def _register() -> None:
    """Register every check of this module."""
    for semiring in PARSERS:
        for unary in (
            ("none", "acyclic")
            if semiring == "viterbi"
            else (
                "none",
                "acyclic",
                "cyclic",
            )
        ):
            add(
                f"span/chart-parser/{semiring}/{unary}",
                "chart-parser",
                "Q4" if semiring == "viterbi" else "Q3",
                partial(check_parser, semiring, unary),
            )
    add("span/chart-parser/batch", "chart-parser", "Q3", check_parser_batch)
    add("span/chart-parser/gradient", "chart-parser", "Q1g", check_parser_gradient)
    add("span/chart-parser/counts", "chart-parser", "Q11", check_parser_gradient)
    add(
        "span/chart-parser/underivable_category_gradient",
        "chart-parser",
        "Q1g",
        check_underivable_category_gradient,
    )
    add("span/chart-parser/update", "chart-parser", "Q24", check_update)
    for unary in ("none", "acyclic", "cyclic"):
        add(f"span/inside/{unary}", "inside", "Q3", partial(check_inside, unary))
    add("span/inside/gradient", "inside", "Q1g", check_inside_gradient)
    add(
        "span/inside/impossible_cell_gradient",
        "inside",
        "Q1g",
        check_impossible_cell_gradient,
    )
    add("span/inside/counts", "inside", "Q11", check_inside_gradient)
    add("span/inside/precision", "inside", "Q3", check_inside_precision)
    add("span/inside/impossible_sentence", "inside", "Q2", check_impossible_sentence)
    add("span/inside/impossible_score", "inside", "Q3", check_impossible_sentence)
    add("span/inside/zero_rule_derivative", "inside", "Q1g", check_zero_rule_derivative)
    add("span/inside/memory", "inside", "Q3", check_inside_memory)
    for executor in ("chart-parser", "inside"):
        add(
            f"span/{executor}/divergent_unary",
            executor,
            "Q8",
            partial(check_divergent_unary, executor),
        )
        add(
            f"span/{executor}/empty_sentence",
            executor,
            "Q24",
            partial(check_empty_sentence, executor),
        )


_register()
