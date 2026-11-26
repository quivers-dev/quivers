"""Checks and deliberately wrong versions of rule presentations.

A presentation is a description, not an executor, so like the contract
records its checks stand beside the grid. Each check compares the
presentation's reference interpretation with the independent derivation
oracle, or counts work, or sends records through the codec; each
mutation installs one wrong version of a library function, and the
checks it names reject it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from fractions import Fraction

from quivers.qiec import rules as rules_module
from quivers.qiec.identifiers import StableId
from quivers.qiec.rules import (
    LINEAR,
    ORDERED,
    Derivation,
    Hole,
    HostInterpretation,
    Item,
    OpenDerivation,
    PremiseValue,
    ReferenceInterpretation,
    RuleError,
    RulePresentation,
    StructureError,
    check_derivation,
    check_structure,
    compose,
    identity,
    presentation_of,
    rename,
    required_structure,
    substitute_hypotheses,
    term_derivation,
)
from quivers.qiec.identifiers import ConstructorId
from quivers.structural.signature import Substitution, substitute
from tests.evidence.oracles import binding
from quivers.qiec.serialization import SerializationError, dumps, loads
from tests.evidence import presentations as fixture
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Mismatch
from tests.evidence.counters import CallCounter
from tests.evidence.mutations import Mutation, Rewrite
from tests.evidence.oracles import derivations as oracle

RULES = "quivers.qiec.rules"
SERIALIZATION = "quivers.qiec.serialization"


def _reference() -> ReferenceInterpretation:
    """The reference interpretation of the attachment sentence.

    Returns
    -------
    ReferenceInterpretation
        The oracle of the presentation.
    """
    return ReferenceInterpretation(fixture.PRESENTATION, fixture.OCCURRENCES)


def _trees() -> list[oracle.Tree]:
    """The goal's trees by the independent oracle.

    Returns
    -------
    list of Tree
        Every tree.
    """
    return oracle.derivations(
        fixture.oracle_axioms(),
        fixture.oracle_rules(),
        fixture.GOAL,
        fixture.oracle_sides(),
    )


def _algebra(name: str, children: tuple[Fraction, ...]) -> Fraction:
    """A fold whose rules are not symmetric in their premises.

    Parameters
    ----------
    name : str
        The rule or occurrence.
    children : tuple of Fraction
        The premise values.

    Returns
    -------
    Fraction
        ``2 left - right + 1`` for a rule, the leaf weight for an
        occurrence.
    """
    if not children:
        return fixture.LEAF_WEIGHTS[name]
    left, right = children
    return 2 * left - right + 1


def _operation(node: Derivation, values: tuple[Fraction, ...]) -> Fraction:
    """The same fold, as an operation of the presentation.

    Parameters
    ----------
    node : Derivation
        The node.
    values : tuple of Fraction
        The premise values.

    Returns
    -------
    Fraction
        The value.
    """
    if node.occurrence is not None:
        return fixture.LEAF_WEIGHTS[fixture.NAMES[node.occurrence]]
    return 2 * values[0] - values[1] + 1


def _interpretation() -> HostInterpretation[Fraction]:
    """The asymmetric fold of the attachment grammar.

    Returns
    -------
    HostInterpretation of Fraction
        The interpretation.
    """
    constructors = (*fixture.PRESENTATION.rules, *fixture.PRESENTATION.axioms)
    return HostInterpretation[Fraction](
        presentation=fixture.PRESENTATION,
        operations={item.id: _operation for item in constructors},
    )


def check_derivations() -> None:
    """Require the reference's trees to be the oracle's.

    Raises
    ------
    Mismatch
        If the multisets of trees differ, by the number of trees apart.
    """
    table = _reference()
    found = Counter(
        repr(fixture.shape(item))
        for item in table.derivations(table.item(fixture.SPAN, fixture.GOAL))
    )
    expected = Counter(repr(tree) for tree in _trees())
    apart = sum(((found - expected) + (expected - found)).values())
    if apart:
        raise Mismatch("value", float(apart), f"{apart} trees differ from the oracle")


def check_occurrences() -> None:
    """Require equal inputs to be distinct occurrences.

    Raises
    ------
    Mismatch
        If an item with two input occurrences has another number of leaves.
    """
    table = _reference()
    leaves = table.derivations(table.item(fixture.SPAN, ("NP", 4, 5)))
    if len(leaves) != 2:
        raise Mismatch(
            "value", float(abs(len(leaves) - 2)), f"{len(leaves)} leaves for two inputs"
        )


def check_premise_order() -> None:
    """Require a fold that is not symmetric to see the premises in order.

    Raises
    ------
    Mismatch
        If the sum over trees of the fold differs from the oracle's.
    """
    table = _reference()
    goal = table.item(fixture.SPAN, fixture.GOAL)
    found = table.observe(
        goal,
        fixture.node_weight,
        _interpretation().fold,
        lambda value: value,
        Fraction(0),
    )
    expected = oracle.nonlinear_fold(
        _trees(), fixture.oracle_weights(), _algebra, lambda value: value
    )
    if found != expected:
        raise Mismatch(
            "value",
            float(abs(found - expected)),
            f"fold sum {found}, oracle {expected}",
        )


def check_nonlinear_observer() -> None:
    """Require a nonlinear observer to be applied per derivation.

    Raises
    ------
    Mismatch
        If the weighted sum of squared folds differs from the oracle's.
    """
    table = _reference()
    goal = table.item(fixture.SPAN, fixture.GOAL)
    found = table.observe(
        goal,
        fixture.node_weight,
        _interpretation().fold,
        lambda value: value * value,
        Fraction(0),
    )
    expected = oracle.nonlinear_fold(
        _trees(), fixture.oracle_weights(), _algebra, lambda value: value * value
    )
    if found != expected:
        raise Mismatch(
            "value",
            float(abs(found - expected)),
            f"observed {found}, oracle {expected}",
        )


def check_codec() -> None:
    """Require a presentation and a derivation to keep every identity.

    Raises
    ------
    Mismatch
        If a decoded record differs, by the number of identities changed.
    """
    table = _reference()
    derivation = table.derivations(table.item(fixture.SPAN, fixture.GOAL))[0]
    before = _identities(fixture.PRESENTATION, derivation)
    try:
        presentation = loads(dumps(fixture.PRESENTATION))
        decoded = loads(dumps(derivation))
    except SerializationError as error:
        raise Mismatch(
            "value", math.inf, f"a round trip was refused: {error}"
        ) from error
    if not isinstance(presentation, RulePresentation) or not isinstance(
        decoded, Derivation
    ):
        raise Mismatch("value", math.inf, "a record decoded to another class")
    after = _identities(presentation, decoded)
    lost = sum(1 for a, b in zip(before, after, strict=True) if a != b)
    if lost or presentation != fixture.PRESENTATION or decoded != derivation:
        raise Mismatch("value", float(max(lost, 1)), f"{lost} identities changed")


def _identities(presentation: RulePresentation, derivation: Derivation) -> list[str]:
    """The identities of a presentation and a derivation, in order.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    derivation : Derivation
        A derivation.

    Returns
    -------
    list of str
        Rule, weight and occurrence identities with their classes.
    """
    held: list[StableId | None] = [presentation.id]
    for rule in presentation.rules:
        held.extend((rule.id, rule.weight))
    stack = [derivation]
    while stack:
        node = stack.pop()
        held.extend((node.constructor, node.occurrence))
        stack.extend(node.premises)
    return [f"{type(item).__name__}:{item}" for item in held]


def check_validated_once() -> None:
    """Require repeated runs of one presentation not to validate it again.

    Raises
    ------
    Mismatch
        If any run validates or analyses the presentation again.
    """
    fixture.PRESENTATION.analysis
    watched = {
        "validation": rules_module._validate_presentation,
        "analysis": rules_module._analyse,
    }
    with CallCounter(watched) as calls:
        for _ in range(3):
            table = _reference()
            table.derivations(table.item(fixture.SPAN, fixture.GOAL))
            fixture.PRESENTATION.analysis
    again = sum(calls.counts.values())
    if again:
        raise Mismatch(
            "work", float(again), f"presentation work repeated: {calls.counts}"
        )


def check_substitution() -> None:
    """Require de Bruijn substitution to agree with named substitution.

    Every term and replacement of the fixture is substituted by the
    library and by the binding oracle, which renames bound variables by
    name; with and without a shift of the remaining variables.

    Raises
    ------
    Mismatch
        If any result differs, by the number of results apart.
    """
    apart = 0
    for replacements in fixture.REPLACEMENTS:
        for shift in (3, 5):
            sigma = Substitution(replacements=replacements, shift=shift)
            for term in fixture.TERMS:
                found = substitute(term, sigma, fixture.LAMBDA)
                expected = binding.substitute_indexed(
                    term.to_tuple(),
                    tuple(item.to_tuple() for item in replacements),
                    shift,
                    3,
                    fixture.BINDERS,
                )
                if found.to_tuple() != expected:
                    apart += 1
    if apart:
        raise Mismatch("value", float(apart), f"{apart} substitutions differ")


def check_substitution_laws() -> None:
    """Require identity, composition and renaming laws of substitution.

    Raises
    ------
    Mismatch
        If a law fails, by the number of failures.
    """
    failures = 0
    identity_substitution = Substitution()
    for term in fixture.TERMS:
        if substitute(term, identity_substitution, fixture.LAMBDA) != term:
            failures += 1
        for first in fixture.REPLACEMENTS:
            for second in fixture.REPLACEMENTS:
                sigma = Substitution(replacements=first, shift=3)
                tau = Substitution(replacements=second, shift=3)
                twice = substitute(
                    substitute(term, sigma, fixture.LAMBDA), tau, fixture.LAMBDA
                )
                once = substitute(term, sigma.then(tau, fixture.LAMBDA), fixture.LAMBDA)
                if twice != once:
                    failures += 1
    if failures:
        raise Mismatch("value", float(failures), f"{failures} substitution laws fail")


def _pairs_items() -> tuple[Item, Item, Item]:
    """The items of the ordered pairs presentation.

    Returns
    -------
    tuple of Item
        The two resources and their pair.
    """
    return (
        Item(family=fixture.FIRST),
        Item(family=fixture.SECOND),
        Item(family=fixture.PAIR),
    )


def _pair(first: int, second: int, rule: str = "pair") -> OpenDerivation:
    """A pair of two hypotheses of a context ``(a, b)`` or ``(a, a)``.

    Parameters
    ----------
    first, second : int
        The de Bruijn indices of the two premises.
    rule : str
        ``"pair"`` or ``"twice"``.

    Returns
    -------
    OpenDerivation
        The derivation.
    """
    a, b, pair = _pairs_items()
    context = (a, b) if rule == "pair" else (a, a)
    return OpenDerivation(
        context=context,
        body=Derivation(
            constructor=ConstructorId.derive("pairs", rule),
            item=pair,
            premises=(
                Hole(index=first, item=context[1 - first]),
                Hole(index=second, item=context[1 - second]),
            ),
        ),
    )


def check_ordered_structure() -> None:
    """Require an ordered context to admit uses in order and refuse a permutation.

    Raises
    ------
    Mismatch
        If the ordered use is refused or a permuted one admitted.
    """
    presentation = fixture.ORDERED_PAIRS
    in_order = _pair(1, 0, "twice")
    check_structure(presentation, in_order)
    a, _, _ = _pairs_items()
    swapped = rename(presentation, in_order, (1, 0), (a, a))
    required = required_structure(presentation, swapped)
    if not required.exchange:
        raise Mismatch("refusal", math.inf, "a permuted use needs no exchange")
    try:
        check_structure(presentation, swapped)
    except StructureError:
        pass
    else:
        raise Mismatch("refusal", math.inf, "an ordered context admitted a permutation")
    if not LINEAR.admits(required) or ORDERED.admits(required):
        raise Mismatch(
            "refusal", math.inf, "exchange is admitted by the wrong structures"
        )


def check_multicategory() -> None:
    """Require the unit and associativity laws of composition, and its functoriality.

    Raises
    ------
    Mismatch
        If a law fails, or folding a composite differs from composing folds.
    """
    presentation = fixture.ORDERED_PAIRS
    a, b, pair = _pairs_items()
    f = _pair(1, 0)
    failures = 0
    try:
        if compose(presentation, f, (identity(a), identity(b))) != f:
            failures += 1
        if compose(presentation, identity(pair), (f,)) != f:
            failures += 1
        nested = compose(
            presentation, identity(pair), (compose(presentation, identity(pair), (f,)),)
        )
        flat = compose(
            presentation, compose(presentation, identity(pair), (identity(pair),)), (f,)
        )
        if nested != flat:
            failures += 1
        lam = presentation_of(fixture.LAMBDA)
        body = term_derivation(
            fixture.LAMBDA,
            fixture.app(fixture.var(0), fixture.abs_(fixture.var(2))),
            ("T", "T"),
        )
        arguments = (
            term_derivation(fixture.LAMBDA, fixture.abs_(fixture.var(1)), ("T",)),
            term_derivation(fixture.LAMBDA, fixture.const("a"), ()),
        )
        composite = compose(lam, body, arguments)
        planar = Substitution(
            replacements=(fixture.const("a"), fixture.abs_(fixture.var(1))), shift=1
        )
        expected = term_derivation(
            fixture.LAMBDA,
            substitute(
                fixture.app(fixture.var(0), fixture.abs_(fixture.var(2))),
                planar,
                fixture.LAMBDA,
            ),
            ("T",),
        )
        if composite != expected:
            failures += 1
        interpretation = _strings(lam)
        whole = interpretation.fold_open(composite, ("x",))
        parts = interpretation.fold_open(
            body,
            tuple(
                interpretation.fold_open(argument, env)
                for argument, env in zip(arguments, (("x",), ()), strict=True)
            ),
        )
        if whole != parts:
            failures += 1
    except RuleError as error:
        raise Mismatch("value", math.inf, f"composition refused: {error}") from error
    if failures:
        raise Mismatch("value", float(failures), f"{failures} multicategory laws fail")


def check_naturality() -> None:
    """Require derivations of terms to commute with substitution.

    Raises
    ------
    Mismatch
        If the derivation of a substituted term differs from substituting
        the derivations, by the number of terms apart.
    """
    lam = presentation_of(fixture.LAMBDA)
    context = ("T", "T", "T")
    apart = 0
    try:
        for replacements in fixture.REPLACEMENTS:
            sigma = Substitution(replacements=replacements, shift=3)
            arguments = tuple(
                term_derivation(fixture.LAMBDA, item, context)
                for item in reversed(replacements)
            )
            for term in fixture.TERMS:
                left = term_derivation(
                    fixture.LAMBDA, substitute(term, sigma, fixture.LAMBDA), context
                )
                right = substitute_hypotheses(
                    lam,
                    term_derivation(fixture.LAMBDA, term, context),
                    arguments,
                    left.context,
                )
                check_derivation(lam, left)
                if left != right:
                    apart += 1
    except RuleError as error:
        raise Mismatch("value", math.inf, f"substitution refused: {error}") from error
    if apart:
        raise Mismatch("value", float(apart), f"{apart} terms do not commute")


def _strings(presentation: RulePresentation) -> HostInterpretation[str]:
    """Print a lambda term, naming each bound variable by its value.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation of the lambda signature.

    Returns
    -------
    HostInterpretation of str
        The printer.
    """
    names = {rule.name: rule.id for rule in presentation.rules}

    def constant(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
        """A constant's name.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple
            No premise values.

        Returns
        -------
        str
            The name.
        """
        del values
        return str(node.values[0])

    def application(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
        """An application.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple
            The function and the argument.

        Returns
        -------
        str
            ``(f a)``.
        """
        del node
        return f"({values[0]} {values[1]})"

    def abstraction(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
        """An abstraction, its variable named by its depth.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple
            The body, a function of the variable's value.

        Returns
        -------
        str
            ``(\\v. body)``.
        """
        del node
        body = values[0]
        assert callable(body)
        name = "v"
        return f"(\\{name}. {body((name,))})"

    return HostInterpretation[str](
        presentation=presentation,
        operations={
            names["Const"]: constant,
            names["App"]: application,
            names["Abs"]: abstraction,
        },
    )


CHECKS: dict[str, Check] = {}
"""Every check of rule presentations by name."""


def _add(name: str, run: Callable[[], None]) -> None:
    """Register one check.

    Parameters
    ----------
    name : str
        Its name.
    run : Callable[[], None]
        Its body.

    Raises
    ------
    ValueError
        If the name is taken.
    """
    if name in CHECKS:
        raise ValueError(f"the check {name!r} is registered twice")
    CHECKS[name] = Check(name=name, executor="presentation", question="Q24", run=run)


for _name, _run in (
    ("presentations/derivations", check_derivations),
    ("presentations/occurrences", check_occurrences),
    ("presentations/premise-order", check_premise_order),
    ("presentations/nonlinear-observer", check_nonlinear_observer),
    ("presentations/codec", check_codec),
    ("presentations/validated-once", check_validated_once),
    ("binding/substitution", check_substitution),
    ("binding/substitution-laws", check_substitution_laws),
    ("binding/ordered-structure", check_ordered_structure),
    ("binding/multicategory", check_multicategory),
    ("binding/naturality", check_naturality),
):
    _add(_name, _run)


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="premises-folded-in-reverse",
        defect_class="order",
        provenance="a fold reads the premises of a node from the last to the first",
        target=f"{RULES}:HostInterpretation._evaluate",
        rewrites=(
            Rewrite(
                old="self.operations[node.constructor](node, tuple(done))",
                new="self.operations[node.constructor](node, tuple(reversed(done)))",
            ),
        ),
        rejected_by=("presentations/premise-order", "presentations/nonlinear-observer"),
        min_effect=0.01,
    ),
    Mutation(
        name="occurrences-of-one-item-merged",
        defect_class="provenance",
        provenance="two inputs with one item keep one leaf, as a chart keyed by item does",
        target=f"{RULES}:ReferenceInterpretation._ground",
        rewrites=(
            Rewrite(
                old="self._leaves.setdefault(leaf.item, []).append(leaf)",
                new="self._leaves[leaf.item] = [leaf]",
            ),
        ),
        rejected_by=("presentations/occurrences", "presentations/derivations"),
        min_effect=1.0,
    ),
    Mutation(
        name="observer-applied-to-the-aggregate",
        defect_class="observer",
        provenance="a nonlinear observer is applied to the weighted sum of values",
        target=f"{RULES}:ReferenceInterpretation.observe",
        rewrites=(
            Rewrite(
                old="total = total + product * observer(value(derivation))",
                new="total = total + product * value(derivation)",
            ),
            Rewrite(old="    return total", new="    return observer(total)"),
        ),
        rejected_by=("presentations/nonlinear-observer",),
        min_effect=0.01,
    ),
    Mutation(
        name="presentation-copied-per-run",
        defect_class="graph work",
        provenance="each run copies the immutable presentation, which validates it again",
        target=f"{RULES}:ReferenceInterpretation.__init__",
        rewrites=(
            Rewrite(
                old="self.presentation = presentation",
                new="self.presentation = presentation.with_()",
            ),
        ),
        rejected_by=("presentations/validated-once",),
        min_effect=2.0,
    ),
    Mutation(
        name="replacement-not-shifted-under-a-binder",
        defect_class="binding",
        provenance="a substitution goes under a binder without weakening its replacements",
        target="quivers.structural.signature:Substitution.lift",
        rewrites=(
            Rewrite(
                old="*(substitute(item, weaken, signature) for item in self.replacements),",
                new="*self.replacements,",
            ),
        ),
        rejected_by=("binding/substitution", "binding/naturality"),
        min_effect=1.0,
    ),
    Mutation(
        name="hypothesis-index-not-shifted",
        defect_class="binding",
        provenance="a derivation moved under new hypotheses keeps its hole indices",
        target=f"{RULES}:_weaken",
        rewrites=(
            Rewrite(
                old="return Hole(index=depth + by + outer - 1 - level, item=items[level])",
                new="return Hole(index=depth + outer - 1 - level, item=items[level])",
            ),
        ),
        rejected_by=("binding/naturality", "binding/multicategory"),
        min_effect=1.0,
    ),
    Mutation(
        name="order-of-uses-ignored",
        defect_class="order",
        provenance="a check of structure counts uses but never compares their order",
        target=f"{RULES}:required_structure",
        rewrites=(
            Rewrite(
                old="if any(later < earlier for earlier, later in itertools.pairwise(uses)):",
                new="if False:",
            ),
        ),
        rejected_by=("binding/ordered-structure",),
        min_effect=1.0,
    ),
    Mutation(
        name="presentation-tuples-sorted-on-the-wire",
        defect_class="serialization",
        provenance="the encoder writes tuples in a sorted order, so rule order is lost",
        target=f"{SERIALIZATION}:_encode",
        rewrites=(
            Rewrite(
                old='return {"$tuple": [_encode(item) for item in value]}',
                new='return {"$tuple": [_encode(item) for item in sorted(value, key=str)]}',
            ),
        ),
        rejected_by=("presentations/codec",),
        min_effect=1.0,
    ),
    Mutation(
        name="presentation-identities-derived-again",
        defect_class="serialization",
        provenance="the decoder derives identifiers from their text, so rules get new ones",
        target=f"{SERIALIZATION}:_decode",
        rewrites=(
            Rewrite(
                old="return _IDS[tag].parse(encoded)",
                new="return _IDS[tag].derive(encoded)",
            ),
        ),
        rejected_by=("presentations/codec",),
        min_effect=2.0,
    ),
)
"""Every wrong version of rule presentations and the checks that reject it."""
