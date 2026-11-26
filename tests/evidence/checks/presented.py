"""Checks that one rule presentation drives every reading of a deduction.

The lexical grammar of the harness is read once into its rule
presentation. The agenda's rules and the reference machine's computations
are both derived from it, so the number of parses each gives is checked
against the derivation oracle; the presentation's own reference
interpretation lists the derivations, a host interpretation folds each
into its tree, and the trees are checked against the oracle's; and every
derivation is encoded as a QIEC value the checker accepts. The oracle
reads the grammar from plain rows and never sees the presentation.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping

import torch

from quivers.dsl.ast_nodes import DeductionDecl
from quivers.dsl.compiler._deduction_presentation import (
    TERM_ITEM,
    DeductionPresentation,
    deduction_presentation,
)
from quivers.qiec.rules import Ground
from quivers.dsl.parser import parse
from quivers.qiec.contracts import OccurrenceId, ParameterId
from quivers.qiec.identifiers import StableId
from quivers.qiec.program_runtime import deduction_item, run_deduction
from quivers.qiec.rules import (
    AxiomOccurrence,
    Derivation,
    HostInterpretation,
    Item,
    OpenDerivation,
    PremiseValue,
    ReferenceInterpretation,
    RuleError,
    RulePresentation,
    check_derivation,
    closed_values,
)
from quivers.dsl.compiler._deduction_terms import atom_term
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.stochastic.agenda import InferenceRule, run_agenda
from quivers.stochastic.semiring import BOOLEAN
from quivers.qiec.module import QiecModule
from quivers.qiec.serialization import SerializationError, dumps, loads
from quivers.stochastic.categories import CategorySystem
from quivers.stochastic.schema import EVALUATION, schema_presentation
from tests.evidence.checks._registry import add
from tests.evidence.checks.machine import SENTENCES, module_of, parses
from tests.evidence.compare import Mismatch
from tests.evidence.oracles import derivations
from tests.evidence.oracles.derivations import (
    AxiomRow,
    RuleRow,
    Term,
    Tree,
    Variable,
)
from tests.evidence.sources import SOURCES, deduction, program

LEXICON = (("a", "S"), ("b", "V"), ("a", "V"))
"""The lexicon of the grammar: word and category."""


def declaration() -> DeductionDecl:
    """The grammar's declaration, parsed afresh.

    Returns
    -------
    DeductionDecl
        The deduction ``G`` of the lexical source.
    """
    text = SOURCES["lexical"].replace("SEMIRING", "Counting")
    return next(
        statement
        for statement in parse(text).statements
        if isinstance(statement, DeductionDecl)
    )


def occurrences(
    presented: DeductionPresentation, sentence: tuple[str, ...]
) -> list[AxiomOccurrence]:
    """The lexical axioms of a sentence, one occurrence per entry.

    Parameters
    ----------
    presented : DeductionPresentation
        The grammar's presentation.
    sentence : tuple of str
        The words.

    Returns
    -------
    list of AxiomOccurrence
        An occurrence of the span slot per word and entry of the word.
    """
    (slot,) = (
        item
        for item in presented.presentation.axioms
        if presented.families[item.judgment.family].name == "span"
    )
    names = [binder.name for binder in slot.telescope]
    return [
        AxiomOccurrence(
            slot=slot.id,
            occurrence=OccurrenceId.derive("lexical", f"{index}:{category}"),
            binding=tuple(zip(names, (index, index + 1, category), strict=True)),
            values=(word,),
        )
        for index, word in enumerate(sentence)
        for entry, category in LEXICON
        if entry == word
    ]


def _tree_interpretation(presentation: RulePresentation) -> HostInterpretation[Tree]:
    """The interpretation folding a derivation into the oracle's tree.

    Parameters
    ----------
    presentation : RulePresentation
        The grammar's presentation.

    Returns
    -------
    HostInterpretation of Tree
        A rule node is its name, its item, its weight name and its
        children; an axiom is its position and category.
    """

    def item(node: Derivation) -> Term:
        """The oracle's spelling of a node's item.

        Parameters
        ----------
        node : Derivation
            The node.

        Returns
        -------
        Term
            ``("span", i, j, category)``.
        """
        return ("span", *node.item.arguments)

    def rule(node: Derivation, children: tuple[PremiseValue[Tree], ...]) -> Tree:
        """A rule node.

        Parameters
        ----------
        node : Derivation
            The node.
        children : tuple of PremiseValue of Tree
            The trees of its premises.

        Returns
        -------
        Tree
            The tree.
        """
        name = presentation.constructors_by_id[node.constructor].name
        return (name, item(node), name, closed_values(children))

    def axiom(node: Derivation, _children: tuple[PremiseValue[Tree], ...]) -> Tree:
        """An axiom leaf.

        Parameters
        ----------
        node : Derivation
            The leaf.
        _children : tuple of PremiseValue of Tree
            None.

        Returns
        -------
        Tree
            The leaf, named by its position and category.
        """
        start, _end, category = node.item.arguments
        name = f"{start}:{category}"
        return (name, item(node), name, ())

    operations = {
        identifier: axiom
        if identifier in {slot.id for slot in presentation.axioms}
        else rule
        for identifier in presentation.constructors_by_id
    }
    return HostInterpretation(presentation=presentation, operations=operations)


def check_one_presentation() -> None:
    """Require the parser, the meaning and the encoder of one presentation to agree.

    Raises
    ------
    Mismatch
        If the agenda or the machine counts other parses than the oracle,
        the compiled presentation is not the one read from the source, the
        folded derivations are not the oracle's trees, or the checker
        refuses a derivation.
    """
    presented = deduction_presentation(declaration())
    compiled = program("lexical", SEMIRING="Counting")
    system = deduction(compiled, "G")
    if system.presentation != presented.presentation:
        raise Mismatch(
            "value", math.inf, "the agenda's rules come from another presentation"
        )
    module = module_of("lexical", SEMIRING="Counting")
    interpretation = _tree_interpretation(presented.presentation)
    for sentence in SENTENCES:
        expected = parses(sentence)
        found = int(system(list(sentence)).goal_weight().detach())
        machine = run_deduction(module, "G", tokens=sentence).weight
        assert isinstance(machine, int)
        reference = ReferenceInterpretation(
            presented.presentation, occurrences(presented, sentence)
        )
        span = next(
            identity
            for identity, form in presented.families.items()
            if form.name == "span"
        )
        goal = reference.item(span, (0, len(sentence), "S"))
        derived = reference.derivations(goal)
        for count, what in ((found, "agenda"), (machine, "machine")):
            if count != len(expected):
                raise Mismatch(
                    "value",
                    float(abs(count - len(expected))),
                    f"{what} parses {sentence} {count} times, not {len(expected)}",
                )
        trees = Counter(interpretation.fold(item) for item in derived)
        if trees != Counter(expected):
            missing = (Counter(expected) - trees).total() + (
                trees - Counter(expected)
            ).total()
            raise Mismatch(
                "value", float(missing), f"folded trees of {sentence} differ"
            )
        for item in derived:
            try:
                check_derivation(presented.presentation, OpenDerivation(body=item))
            except RuleError as refusal:
                raise Mismatch(
                    "value", math.inf, f"a derivation is refused: {refusal}"
                ) from refusal


def _identities(presentation: RulePresentation) -> list[StableId | None]:
    """The identities of a presentation's rules and their weights.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.

    Returns
    -------
    list of StableId or None
        The presentation's, then each rule's and its weight's.
    """
    held: list[StableId | None] = [presentation.id]
    for rule in presentation.rules:
        held.extend((rule.id, rule.weight))
    return held


def check_presentation_codec() -> None:
    """Require a deduction's presentation to keep every rule's parameter.

    Each learnable rule has a parameter identity of its own, named by the
    deduction and the rule, and a round trip through the codec keeps every
    identity.

    Raises
    ------
    Mismatch
        If a learnable rule has no parameter, shares one, has another
        name, or the round trip changes an identity, by the number of
        identities lost.
    """
    presented = deduction_presentation(declaration())
    weights = [item.rule.weight for item in presented.rules if item.learnable]
    wanted = [
        ParameterId.derive("G", "rule", item.rule.name)
        for item in presented.rules
        if item.learnable
    ]
    wrong = sum(1 for have, want in zip(weights, wanted, strict=True) if have != want)
    if wrong:
        raise Mismatch("value", float(wrong), f"{wrong} rule parameters are lost")
    try:
        decoded = loads(dumps(presented.presentation))
    except SerializationError as error:
        raise Mismatch(
            "value", math.inf, f"a round trip was refused: {error}"
        ) from error
    if not isinstance(decoded, RulePresentation):
        raise Mismatch("value", math.inf, "the presentation decoded to another class")
    before, after = _identities(presented.presentation), _identities(decoded)
    lost = sum(1 for a, b in zip(before, after, strict=True) if a != b)
    if lost or decoded != presented.presentation:
        raise Mismatch("value", float(max(lost, 1)), f"{lost} identities changed")


SCHEMA_SENTENCES: tuple[tuple[tuple[str, ...], int], ...] = (
    (("NP", "S\\NP"), 1),
    (("S/NP", "NP"), 1),
    (("NP/NP", "NP", "S\\NP"), 1),
    (("S/S", "S/NP", "NP"), 1),
    (("NP", "S\\NP", "S\\S"), 1),
    (("S/S", "NP", "S\\NP"), 1),
    (("NP", "NP"), 0),
)
"""Category sequences and their numbers of application derivations of ``S``,
counted by hand: forward and backward application over atoms ``NP`` and
``S`` with slashes of depth one. ``S/S NP S\\NP`` has one derivation,
since ``S/S`` needs an ``S`` on its right."""


def check_schema_presentation() -> None:
    """Require the presentation of the application schema to count derivations.

    Raises
    ------
    Mismatch
        If the reference interpretation of the presentation lists another
        number of derivations of ``S`` than the hand count.
    """
    system = CategorySystem.from_atoms_and_slash_depth(["NP", "S"], 1)
    presentation = schema_presentation(EVALUATION, system, "application")
    (slot,) = presentation.axioms
    (span,) = presentation.judgments
    names = [binder.name for binder in slot.telescope]
    for categories, wanted in SCHEMA_SENTENCES:
        axioms = [
            AxiomOccurrence(
                slot=slot.id,
                occurrence=OccurrenceId.derive("application", str(index)),
                binding=tuple(zip(names, (category, index, index + 1), strict=True)),
                values=(category,),
            )
            for index, category in enumerate(categories)
        ]
        reference = ReferenceInterpretation(presentation, axioms)
        goal = reference.item(span.id, ("S", 0, len(categories)))
        found = len(reference.derivations(goal))
        if found != wanted:
            raise Mismatch(
                "value",
                float(abs(found - wanted)),
                f"{categories} has {found} derivations, not {wanted}",
            )


BARE_RULES: tuple[RuleRow, ...] = (
    ("ab", (("span", Variable("I"), Variable("J"), "A"),), "B", "ab"),
    ("tag", (("span", Variable("I"), Variable("J"), "K"),), "K", "tag"),
    ("wrap", ("K",), ("p", "K"), "wrap"),
    ("keep", (Variable("X"), "K"), Variable("X"), "keep"),
    ("goal", ("B", ("p", "K")), "G", "goal"),
)
"""The rules of the deduction with a bare variable, as plain rows."""

BARE_AXIOMS: tuple[tuple[str, Term], ...] = (
    ("0:A", ("span", 0, 1, "A")),
    ("1:K", ("span", 1, 2, "K")),
)
"""Its lexical axioms for the sentence ``a k``: occurrence and span item."""

BARE_WORDS = {"A": "a", "K": "k"}
"""The word of each category of the lexicon."""

BARE_HEIGHT = 3
"""The height the derivations are listed to: ``keep`` makes every item cyclic."""


def _ground(term: Term) -> Ground:
    """An oracle term without variables, as a ground index value.

    Parameters
    ----------
    term : Term
        A number, a name, or a tuple of them.

    Returns
    -------
    Ground
        The same value.
    """
    if isinstance(term, tuple):
        return tuple(_ground(part) for part in term)
    assert isinstance(term, (int, str)) and not isinstance(term, Variable)
    return term


def _oracle_item(presented: DeductionPresentation, item: Item) -> Term:
    """The oracle's spelling of an item of the presentation.

    Parameters
    ----------
    presented : DeductionPresentation
        The presentation.
    item : Item
        The item.

    Returns
    -------
    Term
        The term of a term item, or the head applied to the arguments.
    """
    head = presented.families[item.family].name
    if head == TERM_ITEM:
        (term,) = item.arguments
        return term
    return (head, *item.arguments)


def _bare_interpretation(presented: DeductionPresentation) -> HostInterpretation[Tree]:
    """Fold a derivation of the bare deduction into the oracle's tree.

    Parameters
    ----------
    presented : DeductionPresentation
        The presentation.

    Returns
    -------
    HostInterpretation of Tree
        A rule node is named by the sequent it instantiates.
    """
    sources = {item.rule.id: item.source.name for item in presented.rules}
    leaves = {OccurrenceId.derive("bare", name): name for name, _item in BARE_AXIOMS}

    def node(derivation: Derivation, children: tuple[PremiseValue[Tree], ...]) -> Tree:
        """One node.

        Parameters
        ----------
        derivation : Derivation
            The node.
        children : tuple of PremiseValue of Tree
            The trees of its premises.

        Returns
        -------
        Tree
            The tree.
        """
        item = _oracle_item(presented, derivation.item)
        if derivation.occurrence is not None:
            name = leaves[derivation.occurrence]
            return (name, item, name, ())
        name = sources[derivation.constructor]
        return (name, item, name, closed_values(children))

    return HostInterpretation(
        presentation=presented.presentation,
        operations={
            identifier: node for identifier in presented.presentation.constructors_by_id
        },
    )


def check_bare_items() -> None:
    """Require a rule over every judgment form to be the family of its instances.

    Raises
    ------
    Mismatch
        If the derivations of an item to the height differ from the
        oracle's, by the number of trees that differ, or the agenda run
        on the rules spelled from the presentation does not derive the
        items the oracle derives.
    """
    text = SOURCES["bare"].replace("SEMIRING", "Boolean")
    decl = next(
        statement
        for statement in parse(text).statements
        if isinstance(statement, DeductionDecl)
    )
    presented = deduction_presentation(decl)
    (slot,) = (
        item
        for item in presented.presentation.axioms
        if presented.families[item.judgment.family].name == "span"
    )
    names = [binder.name for binder in slot.telescope]
    axioms = [
        AxiomOccurrence(
            slot=slot.id,
            occurrence=OccurrenceId.derive("bare", name),
            binding=tuple(
                zip(names, (_ground(part) for part in item[1:]), strict=True)
            ),
            values=(BARE_WORDS[str(item[3])],),
        )
        for name, item in BARE_AXIOMS
        if isinstance(item, tuple)
    ]
    reference = ReferenceInterpretation(presented.presentation, axioms)
    interpretation = _bare_interpretation(presented)
    family = {form.name: identity for identity, form in presented.families.items()}
    goals: tuple[Term, ...] = (
        "B",
        "K",
        "G",
        ("p", "K"),
        ("span", 0, 1, "A"),
        ("span", 1, 2, "K"),
    )
    for goal in goals:
        item = (
            reference.item(family["span"], goal[1:])
            if isinstance(goal, tuple) and goal[0] == "span"
            else reference.item(family[TERM_ITEM], (goal,))
        )
        found = Counter(
            interpretation.fold(tree)
            for tree in reference.derivations(item, height=BARE_HEIGHT)
        )
        wanted = Counter(
            derivations.derivations(BARE_AXIOMS, BARE_RULES, goal, height=BARE_HEIGHT)
        )
        if found != wanted:
            differ = (found - wanted).total() + (wanted - found).total()
            raise Mismatch(
                "value", float(differ), f"{differ} derivations of {goal} differ"
            )
    rules = [
        InferenceRule(
            name=item.rule.name,
            premises=tuple(
                presented.chart_pattern(premise.judgment)
                for premise in item.rule.premises
            ),
            conclusion=presented.chart_pattern(item.rule.conclusion),
        )
        for item in presented.rules
    ]
    spelled = {"A": atom_term("A"), "K": atom_term("K")}
    chart = run_agenda(
        [
            (
                ("span", item[1], item[2], spelled[str(item[3])]),
                torch.tensor(1.0, dtype=torch.float64),
            )
            for _name, item in BARE_AXIOMS
            if isinstance(item, tuple)
        ],
        rules,
        semiring=BOOLEAN,
    ).chart
    for goal in ("B", "K", "G"):
        found = chart.get(atom_term(goal))
        if found is None or not bool(found):
            raise Mismatch("value", math.inf, f"the agenda does not derive {goal}")
    wrapped = chart.get(("p", atom_term("K")))
    if wrapped is None or not bool(wrapped):
        raise Mismatch("value", math.inf, "the agenda does not derive p(K)")


CATEGORIAL_SENTENCES: tuple[tuple[Term, ...], ...] = (
    ("NP", ("bwd_slash", "NP", "S")),
    (("fwd_slash", "S", "NP"), "NP"),
    ("NP", ("bwd_slash", "NP", ("fwd_slash", "S", "NP")), "NP"),
    (("fwd_slash", "S", "VP"), ("fwd_slash", "VP", "NP"), "NP"),
    ("NP", "NP"),
)
"""Inputs of the categorial deduction, each a sequence of category terms,
none of which derives an item from itself."""

CATEGORIAL_RULES: tuple[RuleRow, ...] = (
    (
        "fwd_app",
        (("fwd_slash", Variable("X"), Variable("Y")), Variable("Y")),
        Variable("X"),
        "fwd_app",
    ),
    (
        "bwd_app",
        (Variable("Y"), ("bwd_slash", Variable("Y"), Variable("X"))),
        Variable("X"),
        "bwd_app",
    ),
)
"""Application in both directions, as written: ``X/Y, Y |- X`` and
``Y, Y\\X |- X``."""


def _category_item(module: QiecModule, category: Term) -> RuntimeConstructor:
    """The machine's item of a category term.

    Parameters
    ----------
    module : QiecModule
        The module.
    category : Term
        An atom's name, or a slash applied to two categories.

    Returns
    -------
    RuntimeConstructor
        The item.
    """
    if isinstance(category, str):
        return deduction_item(module, "CG", category)
    assert isinstance(category, tuple)
    head, *arguments = category
    assert isinstance(head, str)
    return deduction_item(
        module, "CG", head, *(_category_item(module, item) for item in arguments)
    )


def check_categorial() -> None:
    """Require a deduction over the terms of its signature to count derivations.

    Every item is a category term, a bare variable is a category, and a
    variable that is an item is the variable inside a slash.

    Raises
    ------
    Mismatch
        If the presentation's derivations of ``S`` are not the oracle's
        trees, or the agenda or the machine counts another number.
    """
    text = SOURCES["categorial"].replace("SEMIRING", "Counting")
    decl = next(
        statement
        for statement in parse(text).statements
        if isinstance(statement, DeductionDecl)
    )
    presented = deduction_presentation(decl)
    (slot,) = presented.presentation.axioms
    (form,) = presented.presentation.judgments
    module = module_of("categorial", SEMIRING="Counting")
    for sentence in CATEGORIAL_SENTENCES:
        rows: list[AxiomRow] = [
            (f"w{index}", item) for index, item in enumerate(sentence)
        ]
        leaves = {OccurrenceId.derive("categorial", name): name for name, _ in rows}
        reference = ReferenceInterpretation(
            presented.presentation,
            [
                AxiomOccurrence(
                    slot=slot.id,
                    occurrence=identity,
                    binding=(("x0", _ground(item)),),
                )
                for identity, (_, item) in zip(leaves, rows, strict=True)
            ],
        )
        goal = reference.item(form.id, ("S",))
        found = Counter(
            _categorial_tree(presented, leaves, tree)
            for tree in reference.derivations(goal)
        )
        wanted = Counter(derivations.derivations(rows, CATEGORIAL_RULES, "S"))
        if found != wanted:
            differ = (found - wanted).total() + (wanted - found).total()
            raise Mismatch(
                "value", float(differ), f"{differ} derivations of {sentence} differ"
            )
        counted = run_deduction(
            module,
            "CG",
            axioms=[_category_item(module, item) for item in sentence],
            axiom_weights=[1] * len(sentence),
            fuel=500_000,
        ).weight
        assert isinstance(counted, int)
        if counted != wanted.total():
            raise Mismatch(
                "value",
                float(abs(counted - wanted.total())),
                f"the machine counts {counted} derivations of {sentence}",
            )


def _categorial_tree(
    presented: DeductionPresentation,
    leaves: Mapping[OccurrenceId, str],
    node: Derivation,
) -> Tree:
    """A derivation of the categorial deduction in the oracle's tree form.

    Parameters
    ----------
    presented : DeductionPresentation
        The presentation.
    leaves : Mapping of OccurrenceId to str
        The name of each input occurrence.
    node : Derivation
        The derivation.

    Returns
    -------
    Tree
        Rule or occurrence name, item, weight name, children.
    """
    item = _oracle_item(presented, node.item)
    if node.occurrence is not None:
        name = leaves[node.occurrence]
        return (name, item, name, ())
    name = presented.presentation.constructors_by_id[node.constructor].name
    return (
        name,
        item,
        name,
        tuple(
            _categorial_tree(presented, leaves, child)
            for child in node.premises
            if isinstance(child, Derivation)
        ),
    )


def _register() -> None:
    """Register every check of this module."""
    add("presented/one_presentation", "machine-lexical", "Q24", check_one_presentation)
    add("presented/codec", "machine-lexical", "Q24", check_presentation_codec)
    add("presented/schema", "chart-parser", "Q24", check_schema_presentation)
    add("presented/bare_items", "agenda-fifo", "Q24", check_bare_items)
    add("presented/categorial", "machine-nonlexical", "Q24", check_categorial)


_register()
