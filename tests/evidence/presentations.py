"""Rule presentations of the harness, each with its plain-data twin.

The prepositional attachment grammar below is stated twice: once as a
`RulePresentation` and once as the rows the derivation oracle reads. The
oracle never sees the presentation, so agreement of the two is evidence
about the presentation and its reference interpretation.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

from quivers.qiec.contracts import OccurrenceId, ParameterId, PresentationId
from quivers.qiec.declarations import FieldDef
from quivers.qiec.effects import EMPTY_ROW, ComputationType
from quivers.qiec.identifiers import (
    ComputationId,
    ConstructorId,
    FamilyId,
    SourceOrigin,
)
from quivers.qiec.kinds import NAT, IndexBinder, UserIndexSort
from quivers.qiec.module import NamedComputation
from quivers.qiec.primitives import primitive
from quivers.qiec.rules import (
    PRESENTATION_PROTOCOL,
    AxiomOccurrence,
    AxiomSlot,
    Derivation,
    Judgment,
    JudgmentDecl,
    Premise,
    Rule,
    RulePresentation,
    SideCondition,
)
from quivers.qiec.terms import Local, PrimitiveApplication, Return, Var
from quivers.qiec.types import BOOL, INT, STRING, IndexLiteral, IndexVariable
from tests.evidence.oracles import derivations as tree_oracle
from tests.evidence.oracles.derivations import (
    AxiomRow,
    RuleRow,
    Term,
    Tree,
    Variable,
)

NAME = "attachment"
"""The presentation's name."""

CATEGORIES = ("S", "NP", "VP", "V", "P", "PP")
"""The categories."""

CAT = UserIndexSort(name="Cat", constructors=CATEGORIES)
"""The closed sort of categories."""

SPAN = FamilyId.derive(NAME, "span")
"""The one judgment form: a category over a span."""

ORIGIN = SourceOrigin(
    module=NAME,
    structural_path=("conditions", "before"),
    role="computation",
    source_protocol=PRESENTATION_PROTOCOL,
)

LEFT = Local(name="left", type=INT)
RIGHT = Local(name="right", type=INT)

_LT = primitive("lt_int")

BEFORE = NamedComputation(
    id=ComputationId.derive(NAME, "before"),
    name="before",
    telescope=(),
    parameters=(LEFT, RIGHT),
    body=Return(
        value=PrimitiveApplication(
            primitive=_LT.id,
            name=_LT.name,
            arguments=(Var(local=LEFT), Var(local=RIGHT)),
            result_type=BOOL,
            origin=ORIGIN,
        )
    ),
    type=ComputationType(effects=EMPTY_ROW, result=BOOL),
    origin=ORIGIN,
)
"""The side condition ``left < right``."""


def position(name: str) -> IndexVariable:
    """A position variable.

    Parameters
    ----------
    name : str
        Its name.

    Returns
    -------
    IndexVariable
        The variable.
    """
    return IndexVariable(name=name, sort=NAT)


def category(name: str) -> IndexLiteral:
    """A category constant.

    Parameters
    ----------
    name : str
        The category.

    Returns
    -------
    IndexLiteral
        The constant, a nullary constructor of the category sort.
    """
    return IndexLiteral(value=name, sort=CAT)


def span(label: str, start: str, end: str) -> Judgment:
    """A span judgment pattern.

    Parameters
    ----------
    label : str
        The category.
    start, end : str
        The position variables.

    Returns
    -------
    Judgment
        The pattern.
    """
    return Judgment(
        family=SPAN, arguments=(category(label), position(start), position(end))
    )


BINARY = (
    ("s", "NP", "VP", "S"),
    ("vp", "V", "NP", "VP"),
    ("vp-pp", "VP", "PP", "VP"),
    ("np-pp", "NP", "PP", "NP"),
    ("pp", "P", "NP", "PP"),
)
"""The rules: name, left, right, conclusion."""

WEIGHTS = {
    "s": Fraction(1, 2),
    "vp": Fraction(2, 3),
    "vp-pp": Fraction(1, 5),
    "np-pp": Fraction(3, 7),
    "pp": Fraction(5, 6),
}
"""The rule weights."""

TELESCOPE = tuple(IndexBinder(name=name, sort=NAT) for name in ("i", "k", "j"))


def _rule(name: str, left: str, right: str, conclusion: str) -> Rule:
    """One binary rule with the side condition ``i < j``.

    Parameters
    ----------
    name : str
        The rule.
    left, right, conclusion : str
        Its categories.

    Returns
    -------
    Rule
        The rule.
    """
    return Rule(
        id=ConstructorId.derive(NAME, name),
        name=name,
        telescope=TELESCOPE,
        premises=(
            Premise(judgment=span(left, "i", "k")),
            Premise(judgment=span(right, "k", "j")),
        ),
        conclusion=span(conclusion, "i", "j"),
        side_condition=SideCondition(computation=BEFORE.id, arguments=("i", "j")),
        weight=ParameterId.derive(NAME, name),
    )


WORD = AxiomSlot(
    id=ConstructorId.derive(NAME, "word"),
    name="word",
    telescope=(
        IndexBinder(name="c", sort=CAT),
        IndexBinder(name="i", sort=NAT),
        IndexBinder(name="j", sort=NAT),
    ),
    judgment=Judgment(
        family=SPAN,
        arguments=(
            IndexVariable(name="c", sort=CAT),
            position("i"),
            position("j"),
        ),
    ),
    values=(FieldDef(name="word", type=STRING),),
)
"""An input word with its category."""

PRESENTATION = RulePresentation(
    id=PresentationId.derive(NAME),
    name=NAME,
    index_sorts=(CAT,),
    judgments=(
        JudgmentDecl(
            id=SPAN,
            name="span",
            indices=(
                IndexBinder(name="c", sort=CAT, refinable=True),
                IndexBinder(name="i", sort=NAT, refinable=True),
                IndexBinder(name="j", sort=NAT, refinable=True),
            ),
        ),
    ),
    rules=tuple(_rule(*row) for row in BINARY),
    axioms=(WORD,),
    conditions=(BEFORE,),
)
"""The attachment grammar."""

SENTENCE = (
    ("w0", "NP", 0, "she"),
    ("w1", "V", 1, "saw"),
    ("w2", "NP", 2, "stars"),
    ("w3", "P", 3, "with"),
    ("w4", "NP", 4, "telescopes"),
    ("w5", "NP", 4, "telescopes"),
)
"""The input words: occurrence, category, start, word. The last two are
two occurrences of one item."""

LEAF_WEIGHTS = {
    "w0": Fraction(1, 3),
    "w1": Fraction(1),
    "w2": Fraction(2, 5),
    "w3": Fraction(1),
    "w4": Fraction(1, 4),
    "w5": Fraction(3, 4),
}
"""The weight of each occurrence."""

GOAL = ("S", 0, 5)
"""The goal item's arguments."""


def occurrence_id(name: str) -> OccurrenceId:
    """The identity of a named occurrence.

    Parameters
    ----------
    name : str
        The occurrence.

    Returns
    -------
    OccurrenceId
        Its identity.
    """
    return OccurrenceId.derive(NAME, name)


OCCURRENCES = tuple(
    AxiomOccurrence(
        slot=WORD.id,
        occurrence=occurrence_id(name),
        binding=(("c", label), ("i", start), ("j", start + 1)),
        values=(word,),
    )
    for name, label, start, word in SENTENCE
)
"""The input occurrences."""

NAMES: Mapping[OccurrenceId, str] = {occurrence_id(name): name for name, *_ in SENTENCE}
"""Occurrence names by identity."""

RULE_NAMES: Mapping[ConstructorId, str] = {
    ConstructorId.derive(NAME, name): name for name, *_ in BINARY
}
"""Rule names by identity."""


def oracle_rules() -> tuple[RuleRow, ...]:
    """The rules as oracle rows.

    Returns
    -------
    tuple of RuleRow
        Name, premise patterns, conclusion pattern, weight name.
    """
    i, k, j = Variable("i"), Variable("k"), Variable("j")
    return tuple(
        (name, ((left, i, k), (right, k, j)), (conclusion, i, j), name)
        for name, left, right, conclusion in BINARY
    )


def oracle_axioms() -> tuple[AxiomRow, ...]:
    """The input occurrences as oracle rows.

    Returns
    -------
    tuple of AxiomRow
        Occurrence name and item.
    """
    return tuple(
        (name, (label, start, start + 1)) for name, label, start, _ in SENTENCE
    )


def oracle_sides() -> dict[str, tree_oracle.SideCondition]:
    """The side conditions as oracle predicates.

    Returns
    -------
    dict of str to SideCondition
        The predicate ``i < j`` by rule name.
    """
    return {name: _before for name, *_ in BINARY}


def _before(binding: Mapping[str, Term]) -> bool:
    """Whether ``i < j``.

    Parameters
    ----------
    binding : Mapping[str, Term]
        The variables.

    Returns
    -------
    bool
        The comparison.
    """
    left, right = binding["i"], binding["j"]
    assert isinstance(left, int) and isinstance(right, int)
    return left < right


def oracle_weights() -> dict[str, Fraction]:
    """Every weight by name.

    Returns
    -------
    dict of str to Fraction
        Rule and occurrence weights.
    """
    return {**WEIGHTS, **LEAF_WEIGHTS}


def shape(derivation: Derivation) -> Tree:
    """A derivation in the oracle's tree form.

    Parameters
    ----------
    derivation : Derivation
        The derivation.

    Returns
    -------
    Tree
        Name, item, weight name, children.
    """
    if derivation.occurrence is not None:
        name = NAMES[derivation.occurrence]
        return (name, derivation.item.arguments, name, ())
    rule = RULE_NAMES[derivation.constructor]
    return (
        rule,
        derivation.item.arguments,
        rule,
        tuple(shape(child) for child in derivation.premises),
    )


def node_weight(derivation: Derivation) -> Fraction:
    """The weight of one node of a derivation.

    Parameters
    ----------
    derivation : Derivation
        The node.

    Returns
    -------
    Fraction
        Its rule's or occurrence's weight.
    """
    if derivation.occurrence is not None:
        return LEAF_WEIGHTS[NAMES[derivation.occurrence]]
    return WEIGHTS[RULE_NAMES[derivation.constructor]]
