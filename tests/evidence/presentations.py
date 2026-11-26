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
    Ground,
    Item,
    Judgment,
    JudgmentDecl,
    Premise,
    Rule,
    RulePresentation,
    SideCondition,
)
from quivers.qiec.terms import Local, PrimitiveApplication, Return, Var
from quivers.qiec.types import (
    BOOL,
    INT,
    STRING,
    IndexConstructor,
    IndexLiteral,
    IndexVariable,
)
from quivers.structural.signature import (
    Binder,
    BinderArgSpec,
    BinderVarSpec,
    Constructor,
    Signature,
    Sort,
    SortVocabEntry,
    bound_var,
)
from quivers.structural.signature import Term as StructuralTerm
from tests.evidence.oracles import binding as binding_oracle
from tests.evidence.oracles import lambek as lambek_oracle
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

LEFT = Local(name="left", type=INT)
RIGHT = Local(name="right", type=INT)

_LT = primitive("lt_int")


def before(module: str) -> NamedComputation:
    """The side condition ``left < right`` of one module.

    Parameters
    ----------
    module : str
        The module that holds it.

    Returns
    -------
    NamedComputation
        The computation.
    """
    origin = SourceOrigin(
        module=module,
        structural_path=("conditions", "before"),
        role="computation",
        source_protocol=PRESENTATION_PROTOCOL,
    )
    return NamedComputation(
        id=ComputationId.derive(module, "before"),
        name="before",
        telescope=(),
        parameters=(LEFT, RIGHT),
        body=Return(
            value=PrimitiveApplication(
                primitive=_LT.id,
                name=_LT.name,
                arguments=(Var(local=LEFT), Var(local=RIGHT)),
                result_type=BOOL,
                origin=origin,
            )
        ),
        type=ComputationType(effects=EMPTY_ROW, result=BOOL),
        origin=origin,
    )


BEFORE = before(NAME)
"""The side condition ``left < right`` of the attachment grammar."""


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
        tuple(
            shape(child)
            for child in derivation.premises
            if isinstance(child, Derivation)
        ),
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


# -- binding ------------------------------------------------------------


LAMBDA = Signature(
    name="Lambda",
    sorts_t=(
        Sort(name="T", kind="object"),
        Sort(
            name="Name",
            kind="data",
            vocab=tuple(
                SortVocabEntry(kind="string", value=v) for v in ("a", "b", "c")
            ),
        ),
    ),
    constructors_t=(
        Constructor(name="Const", domain=("Name",), codomain="T"),
        Constructor(name="App", domain=("T", "T"), codomain="T"),
    ),
    binders_t=(
        Binder(
            name="Abs",
            binds=(BinderVarSpec(var="x", sort="T"),),
            scoped=(BinderArgSpec(arg="body", sort="T"),),
            codomain="T",
        ),
    ),
)
"""Constants, applications and abstractions."""

BINDERS = {"Abs": (1, frozenset({0}))}
"""The binder of `LAMBDA` as the binding oracle reads it."""


def var(index: int) -> StructuralTerm:
    """A bound variable.

    Parameters
    ----------
    index : int
        Its de Bruijn index.

    Returns
    -------
    Term
        The variable.
    """
    return bound_var(index)


def const(name: str) -> StructuralTerm:
    """A constant.

    Parameters
    ----------
    name : str
        Its name.

    Returns
    -------
    Term
        The constant.
    """
    return StructuralTerm(op="Const", args=(name,))


def app(function: StructuralTerm, argument: StructuralTerm) -> StructuralTerm:
    """An application.

    Parameters
    ----------
    function, argument : Term
        The parts.

    Returns
    -------
    Term
        The application.
    """
    return StructuralTerm(op="App", args=(function, argument))


def abs_(body: StructuralTerm) -> StructuralTerm:
    """An abstraction.

    Parameters
    ----------
    body : Term
        The body, with the bound variable at index 0.

    Returns
    -------
    Term
        The abstraction.
    """
    return StructuralTerm(op="Abs", args=(body,))


def from_tuple(data: binding_oracle.Indexed) -> StructuralTerm:
    """A term from the binding oracle's tuples.

    Parameters
    ----------
    data : Indexed
        ``(op, *arguments)``.

    Returns
    -------
    Term
        The term.
    """
    op = data[0]
    assert isinstance(op, str)
    arguments: list[StructuralTerm | str | int | float | bool | bytes] = []
    for item in data[1:]:
        arguments.append(from_tuple(item) if isinstance(item, tuple) else item)
    return StructuralTerm(op=op, args=tuple(arguments))


TERMS = (
    var(0),
    var(2),
    const("a"),
    app(var(1), var(0)),
    abs_(var(0)),
    abs_(var(1)),
    abs_(app(var(0), var(2))),
    app(abs_(abs_(app(var(1), var(3)))), var(1)),
    abs_(abs_(app(app(var(0), var(1)), app(var(2), var(3))))),
)
"""Terms in a context of three variables, under and across binders."""

REPLACEMENTS = (
    (abs_(app(var(0), var(1))), var(0), const("b")),
    (var(1), var(0), var(2)),
    (app(var(0), abs_(var(1))), const("c"), var(0)),
)
"""Replacements of the three variables, each in a context of three; a
substitution by them keeps three variables, so its shift is three."""


# -- ordered resources --------------------------------------------------


FIRST, SECOND, PAIR = (FamilyId.derive("pairs", name) for name in ("a", "b", "pair"))

ORDERED_PAIRS = RulePresentation(
    id=PresentationId.derive("pairs"),
    name="pairs",
    judgments=(
        JudgmentDecl(id=FIRST, name="a"),
        JudgmentDecl(id=SECOND, name="b"),
        JudgmentDecl(id=PAIR, name="pair"),
    ),
    rules=(
        Rule(
            id=ConstructorId.derive("pairs", "pair"),
            name="pair",
            premises=(
                Premise(judgment=Judgment(family=FIRST)),
                Premise(judgment=Judgment(family=SECOND)),
            ),
            conclusion=Judgment(family=PAIR),
        ),
        Rule(
            id=ConstructorId.derive("pairs", "twice"),
            name="twice",
            premises=(
                Premise(judgment=Judgment(family=FIRST)),
                Premise(judgment=Judgment(family=FIRST)),
            ),
            conclusion=Judgment(family=PAIR),
        ),
    ),
)
"""A product of two resources in a context with no structural rule."""


# -- the Lambek calculus ------------------------------------------------


TYPE = UserIndexSort(
    name="LambekType",
    constructors=("np", "s", "n", "over", "under"),
    arities=(0, 0, 0, 2, 2),
)
"""Atoms and the two slashes: ``over(a, b)`` is ``a / b`` and
``under(b, a)`` is ``b \\\\ a``."""

DER = FamilyId.derive("lambek", "der")
"""The judgment that a type is derived."""


def _type_variable(name: str) -> IndexVariable:
    """A type variable.

    Parameters
    ----------
    name : str
        Its name.

    Returns
    -------
    IndexVariable
        The variable.
    """
    return IndexVariable(name=name, sort=TYPE)


def _der(argument: IndexVariable | IndexConstructor) -> Judgment:
    """The judgment of a type pattern.

    Parameters
    ----------
    argument : IndexVariable or IndexConstructor
        The type.

    Returns
    -------
    Judgment
        The judgment.
    """
    return Judgment(family=DER, arguments=(argument,))


def _slash(name: str, first: str, second: str) -> IndexConstructor:
    """A slash type pattern.

    Parameters
    ----------
    name : str
        ``"over"`` or ``"under"``.
    first, second : str
        Its argument variables, in constructor order.

    Returns
    -------
    IndexConstructor
        The pattern.
    """
    return IndexConstructor(
        name=name,
        arguments=(_type_variable(first), _type_variable(second)),
        sort=TYPE,
    )


_TYPES = (IndexBinder(name="a", sort=TYPE), IndexBinder(name="b", sort=TYPE))

LAMBEK = RulePresentation(
    id=PresentationId.derive("lambek"),
    name="lambek",
    index_sorts=(TYPE,),
    judgments=(
        JudgmentDecl(
            id=DER,
            name="der",
            indices=(IndexBinder(name="t", sort=TYPE, refinable=True),),
        ),
    ),
    rules=(
        Rule(
            id=ConstructorId.derive("lambek", "over-elim"),
            name="over-elim",
            telescope=_TYPES,
            premises=(
                Premise(judgment=_der(_slash("over", "a", "b"))),
                Premise(judgment=_der(_type_variable("b"))),
            ),
            conclusion=_der(_type_variable("a")),
        ),
        Rule(
            id=ConstructorId.derive("lambek", "under-elim"),
            name="under-elim",
            telescope=_TYPES,
            premises=(
                Premise(judgment=_der(_type_variable("b"))),
                Premise(judgment=_der(_slash("under", "b", "a"))),
            ),
            conclusion=_der(_type_variable("a")),
        ),
        Rule(
            id=ConstructorId.derive("lambek", "over-intro"),
            name="over-intro",
            telescope=_TYPES,
            premises=(
                Premise(
                    judgment=_der(_type_variable("a")),
                    after=(_der(_type_variable("b")),),
                ),
            ),
            conclusion=_der(_slash("over", "a", "b")),
        ),
        Rule(
            id=ConstructorId.derive("lambek", "under-intro"),
            name="under-intro",
            telescope=_TYPES,
            premises=(
                Premise(
                    judgment=_der(_type_variable("a")),
                    before=(_der(_type_variable("b")),),
                ),
            ),
            conclusion=_der(_slash("under", "b", "a")),
        ),
    ),
)
"""The product-free Lambek calculus in natural deduction, ordered contexts."""


def lambek_item(type_: lambek_oracle.Type) -> Item:
    """The item of a derived type.

    Parameters
    ----------
    type_ : Type
        The type in the oracle's form.

    Returns
    -------
    Item
        Its ``der`` item.
    """
    return Item(family=DER, arguments=(_ground(type_),))


def _ground(type_: lambek_oracle.Type) -> Ground:
    """A type in the ground form of the presentation.

    Parameters
    ----------
    type_ : Type
        The type in the oracle's form.

    Returns
    -------
    Ground
        A name for an atom, ``("over", a, b)`` or ``("under", b, a)``.
    """
    if len(type_) == 1:
        return type_[0]
    connective, first, second = type_
    name = "over" if connective == "/" else "under"
    return (name, _ground(first), _ground(second))


def lambek_ground(type_: lambek_oracle.Type) -> Ground:
    """A type in the ground form of the presentation.

    Parameters
    ----------
    type_ : Type
        The type in the oracle's form.

    Returns
    -------
    Ground
        Its ground value.
    """
    return _ground(type_)


def lambek_type(item: Item) -> lambek_oracle.Type:
    """The oracle's type of a ``der`` item.

    Parameters
    ----------
    item : Item
        The item.

    Returns
    -------
    Type
        Its type.
    """
    return _type(item.arguments[0])


def _type(value: Ground) -> lambek_oracle.Type:
    """The oracle's type of a ground type.

    Parameters
    ----------
    value : Ground
        The ground type.

    Returns
    -------
    Type
        The same type.

    Raises
    ------
    ValueError
        If the value is not a type.
    """
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{value!r} is not a Lambek type")
    name, first, second = value
    return ("/" if name == "over" else "\\", _type(first), _type(second))
