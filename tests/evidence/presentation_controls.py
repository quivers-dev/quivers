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
    Operation,
    Derivation,
    Hole,
    HostInterpretation,
    Item,
    OpenDerivation,
    PremiseValue,
    ReferenceInterpretation,
    Scoped,
    closed_values,
    RuleError,
    RulePresentation,
    StructureError,
    check_derivation,
    check_structure,
    compose_derivations,
    identity_derivation,
    presentation_of,
    rename_hypotheses,
    required_structure,
    substitute_hypotheses,
    term_derivation,
)
from quivers.qiec.identifiers import ConstructorId
from quivers.structural.signature import Substitution, substitute
from tests.evidence.oracles import binding, lambek
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


def _operation(
    node: Derivation, values: tuple[PremiseValue[Fraction], ...]
) -> Fraction:
    """The same fold, as an operation of the presentation.

    Parameters
    ----------
    node : Derivation
        The node.
    values : tuple of PremiseValue
        The premise values.

    Returns
    -------
    Fraction
        The value.
    """
    if node.occurrence is not None:
        return fixture.LEAF_WEIGHTS[fixture.NAMES[node.occurrence]]
    left, right = closed_values(values)
    return 2 * left - right + 1


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
        stack.extend(child for child in node.premises if isinstance(child, Derivation))
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
    swapped = rename_hypotheses(presentation, in_order, (1, 0), (a, a))
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
        if (
            compose_derivations(
                presentation, f, (identity_derivation(a), identity_derivation(b))
            )
            != f
        ):
            failures += 1
        if compose_derivations(presentation, identity_derivation(pair), (f,)) != f:
            failures += 1
        nested = compose_derivations(
            presentation,
            identity_derivation(pair),
            (compose_derivations(presentation, identity_derivation(pair), (f,)),),
        )
        flat = compose_derivations(
            presentation,
            compose_derivations(
                presentation, identity_derivation(pair), (identity_derivation(pair),)
            ),
            (f,),
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
        composite = compose_derivations(lam, body, arguments)
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
        assert isinstance(body, Scoped)
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


NP, S, N = ("np",), ("s",), ("n",)


def _over(a: lambek.Type, b: lambek.Type) -> lambek.Type:
    """The type ``a / b``.

    Parameters
    ----------
    a, b : Type
        The parts.

    Returns
    -------
    Type
        The type.
    """
    return ("/", a, b)


def _under(b: lambek.Type, a: lambek.Type) -> lambek.Type:
    """The type ``b \\ a``.

    Parameters
    ----------
    b, a : Type
        The parts.

    Returns
    -------
    Type
        The type.
    """
    return ("\\", b, a)


def _hole(index: int, type_: lambek.Type) -> Hole:
    """A hypothesis of a derived type.

    Parameters
    ----------
    index : int
        Its de Bruijn index.
    type_ : Type
        Its type.

    Returns
    -------
    Hole
        The hole.
    """
    return Hole(index=index, item=fixture.lambek_item(type_))


def _rule(
    name: str,
    conclusion: lambek.Type,
    a: lambek.Type,
    b: lambek.Type,
    premises: tuple[Derivation | Hole, ...],
) -> Derivation:
    """An application of a Lambek rule.

    Parameters
    ----------
    name : str
        The rule.
    conclusion : Type
        The type it derives.
    a, b : Type
        The rule's type variables.
    premises : tuple of Derivation or Hole
        Its premises.

    Returns
    -------
    Derivation
        The node.
    """
    return Derivation(
        constructor=ConstructorId.derive("lambek", name),
        item=fixture.lambek_item(conclusion),
        binding=(
            ("a", fixture.lambek_ground(a)),
            ("b", fixture.lambek_ground(b)),
        ),
        premises=premises,
    )


def _open(context: tuple[lambek.Type, ...], body: Derivation) -> OpenDerivation:
    """An open Lambek derivation.

    Parameters
    ----------
    context : tuple of Type
        The hypotheses.
    body : Derivation
        The derivation.

    Returns
    -------
    OpenDerivation
        The derivation in that context.
    """
    return OpenDerivation(
        context=tuple(fixture.lambek_item(t) for t in context), body=body
    )


def raised_right() -> OpenDerivation:
    """``np |- s / (np \\ s)``: the hypothesis ``np \\ s`` is on the right.

    Returns
    -------
    OpenDerivation
        The derivation.
    """
    verb = _under(NP, S)
    inner = _rule("under-elim", S, S, NP, (_hole(1, NP), _hole(0, verb)))
    return _open((NP,), _rule("over-intro", _over(S, verb), S, verb, (inner,)))


def raised_left() -> OpenDerivation:
    """``np |- (s / np) \\ s``: the hypothesis ``s / np`` is on the left.

    Returns
    -------
    OpenDerivation
        The derivation.
    """
    verb = _over(S, NP)
    inner = _rule("over-elim", S, S, NP, (_hole(1, verb), _hole(0, NP)))
    return _open((NP,), _rule("under-intro", _under(verb, S), S, verb, (inner,)))


def composed() -> OpenDerivation:
    """``s / np, np / n |- s / n``: composition, by a right hypothesis.

    Returns
    -------
    OpenDerivation
        The derivation.
    """
    inner = _rule(
        "over-elim",
        S,
        S,
        NP,
        (
            _hole(2, _over(S, NP)),
            _rule("over-elim", NP, NP, N, (_hole(1, _over(NP, N)), _hole(0, N))),
        ),
    )
    return _open(
        (_over(S, NP), _over(NP, N)), _rule("over-intro", _over(S, N), S, N, (inner,))
    )


def crossed_right() -> OpenDerivation:
    """``np |- s / (s / np)``, which uses its hypotheses out of order.

    Returns
    -------
    OpenDerivation
        The derivation.
    """
    verb = _over(S, NP)
    inner = _rule("over-elim", S, S, NP, (_hole(0, verb), _hole(1, NP)))
    return _open((NP,), _rule("over-intro", _over(S, verb), S, verb, (inner,)))


def crossed_left() -> OpenDerivation:
    """``np |- (np \\ s) \\ s``, which uses its hypotheses out of order.

    Returns
    -------
    OpenDerivation
        The derivation.
    """
    verb = _under(NP, S)
    inner = _rule("under-elim", S, S, NP, (_hole(0, NP), _hole(1, verb)))
    return _open((NP,), _rule("under-intro", _under(verb, S), S, verb, (inner,)))


def _sequent(derivation: OpenDerivation) -> tuple[tuple[lambek.Type, ...], lambek.Type]:
    """The sequent an open Lambek derivation proves.

    Parameters
    ----------
    derivation : OpenDerivation
        The derivation.

    Returns
    -------
    tuple of tuple of Type and Type
        The antecedent and the succedent, in the oracle's form.
    """
    return (
        tuple(fixture.lambek_type(item) for item in derivation.context),
        fixture.lambek_type(derivation.item),
    )


def check_lambek_planar() -> None:
    """Require the Lambek slash rules to derive exactly the planar sequents.

    The planar derivations pass the QIEC checker and an ordered context,
    and their sequents are provable in the Lambek calculus. The crossed
    ones pass the checker, need exchange, are refused in an ordered
    context and admitted in a linear one, and their sequents are provable
    only with permutation.

    Raises
    ------
    Mismatch
        If any of these fails.
    """
    presentation = fixture.LAMBEK
    try:
        for derivation in (raised_right(), raised_left(), composed()):
            check_derivation(presentation, derivation)
            check_structure(presentation, derivation)
            if not lambek.provable(*_sequent(derivation)):
                raise Mismatch(
                    "value", math.inf, "a planar derivation proves no sequent"
                )
        for derivation in (crossed_right(), crossed_left()):
            check_derivation(presentation, derivation)
            antecedent, succedent = _sequent(derivation)
            if lambek.provable(
                antecedent, succedent
            ) or not lambek.provable_with_exchange(antecedent, succedent):
                raise Mismatch(
                    "value", math.inf, "the oracle disagrees on a crossed sequent"
                )
            if not required_structure(presentation, derivation).exchange:
                raise Mismatch("refusal", math.inf, "a crossed use needs no exchange")
            try:
                check_structure(presentation, derivation)
            except StructureError:
                pass
            else:
                raise Mismatch(
                    "refusal", math.inf, "an ordered context admits a crossing"
                )
            check_structure(presentation.with_(structure=LINEAR), derivation)
    except RuleError as error:
        raise Mismatch(
            "refusal", math.inf, f"a Lambek derivation refused: {error}"
        ) from error


def check_lambek_sides() -> None:
    """Require the two slashes not to be interchangeable.

    The left type raising with its introduction replaced by the right one
    keeps its premise and claims ``np |- s / (s / np)``; the checker has
    to refuse it, since its hypothesis is then on the other side.

    Raises
    ------
    Mismatch
        If the swapped derivation is admitted.
    """
    body = raised_left().body
    assert isinstance(body, Derivation)
    verb = _over(S, NP)
    swapped = _open((NP,), _rule("over-intro", _over(S, verb), S, verb, body.premises))
    try:
        check_derivation(fixture.LAMBEK, swapped)
        check_structure(fixture.LAMBEK, swapped)
    except RuleError:
        return
    raise Mismatch("refusal", math.inf, "a hypothesis was accepted on the other side")


def check_lambek_composition() -> None:
    """Require the multicategory laws and functoriality with left hypotheses.

    Raises
    ------
    Mismatch
        If composing into a left-scoped hypothesis differs from the
        derivation built by hand, a law fails, or folding does not commute.
    """
    presentation = fixture.LAMBEK
    f = raised_left()
    noun = _over(NP, N)
    g = _open((noun, N), _rule("over-elim", NP, NP, N, (_hole(1, noun), _hole(0, N))))
    failures = 0
    try:
        composite = compose_derivations(presentation, f, (g,))
        verb = _over(S, NP)
        inner = _rule(
            "over-elim",
            S,
            S,
            NP,
            (
                _hole(2, verb),
                _rule("over-elim", NP, NP, N, (_hole(1, noun), _hole(0, N))),
            ),
        )
        expected = _open(
            (noun, N), _rule("under-intro", _under(verb, S), S, verb, (inner,))
        )
        if composite != expected:
            failures += 1
        check_derivation(presentation, composite)
        check_structure(presentation, composite)
        if not lambek.provable(*_sequent(composite)):
            failures += 1
        if (
            compose_derivations(presentation, f, (identity_derivation(f.context[0]),))
            != f
        ):
            failures += 1
        unit = tuple(identity_derivation(item) for item in g.context)
        if compose_derivations(presentation, composite, unit) != compose_derivations(
            presentation, f, (compose_derivations(presentation, g, unit),)
        ):
            failures += 1
        printer = lambek_printer(presentation)
        whole = printer.fold_open(composite, ("d", "x"))
        parts = printer.fold_open(f, (printer.fold_open(g, ("d", "x")),))
        if whole != parts:
            failures += 1
    except RuleError as error:
        raise Mismatch("value", math.inf, f"composition refused: {error}") from error
    if failures:
        raise Mismatch(
            "value", float(failures), f"{failures} laws fail with left hypotheses"
        )


def lambek_printer(presentation: RulePresentation) -> HostInterpretation[str]:
    """Read a Lambek derivation as a lambda term with directed abstractions.

    Parameters
    ----------
    presentation : RulePresentation
        The Lambek presentation.

    Returns
    -------
    HostInterpretation of str
        The reading.
    """
    ids = {rule.name: rule.id for rule in presentation.rules}

    def apply_right(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
        """``f x`` for ``a / b``.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple
            Function and argument.

        Returns
        -------
        str
            The application.
        """
        function, argument = closed_values(values)
        return f"({function} {argument})"

    def apply_left(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
        """``f x`` for ``b \\ a``.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple
            Argument and function.

        Returns
        -------
        str
            The application.
        """
        argument, function = closed_values(values)
        return f"({function} {argument})"

    def abstract(side: str) -> Operation[str]:
        """An abstraction over a hypothesis on one side.

        Parameters
        ----------
        side : str
            ``"/"`` or ``"\\"``.

        Returns
        -------
        Operation of str
            The operation.
        """

        def operation(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
            """The abstraction.

            Parameters
            ----------
            node : Derivation
                The node.
            values : tuple
                The body, scoped by the hypothesis.

            Returns
            -------
            str
                ``(side v. body)``.
            """
            body = values[0]
            assert isinstance(body, Scoped)
            return f"({side}v. {body(('v',))})"

        return operation

    return HostInterpretation[str](
        presentation=presentation,
        operations={
            ids["over-elim"]: apply_right,
            ids["under-elim"]: apply_left,
            ids["over-intro"]: abstract("/"),
            ids["under-intro"]: abstract("\\"),
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
    ("lambek/planar", check_lambek_planar),
    ("lambek/sides", check_lambek_sides),
    ("lambek/composition", check_lambek_composition),
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
                old="index=inner_right + total - 1 - (left + level), item=items[level]",
                new="index=inner_right + outer - 1 - level, item=items[level]",
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
        name="side-of-hypotheses-ignored",
        defect_class="order",
        provenance="the left hypotheses of a premise are placed on its right",
        target=f"{RULES}:extend_context",
        rewrites=(
            Rewrite(
                old="return (*before, *context, *after)",
                new="return (*context, *before, *after)",
            ),
        ),
        rejected_by=("lambek/planar", "lambek/composition"),
        min_effect=1.0,
    ),
    Mutation(
        name="left-hypotheses-counted-on-the-right",
        defect_class="binding",
        provenance="the hole arithmetic treats every hypothesis as a right one",
        target=f"{RULES}:_counts",
        rewrites=(
            Rewrite(
                old="return (len(before), len(after))",
                new="return (0, len(before) + len(after))",
            ),
        ),
        rejected_by=("lambek/composition",),
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
