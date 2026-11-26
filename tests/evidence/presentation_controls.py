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
    Derivation,
    HostInterpretation,
    ReferenceInterpretation,
    RulePresentation,
)
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
):
    _add(_name, _run)


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="premises-folded-in-reverse",
        defect_class="order",
        provenance="a fold reads the premises of a node from the last to the first",
        target=f"{RULES}:HostInterpretation.fold",
        rewrites=(
            Rewrite(
                old="tuple(values.pop(id(child)) for child in node.premises)",
                new="tuple(values.pop(id(child)) for child in reversed(node.premises))",
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
