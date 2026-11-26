"""Rule bodies over binary relations, for joins and their work.

The triangle ``r(a, b), s(b, c), t(a, c)`` has an AGM bound of ``N^{3/2}``
and inputs on which every plan that joins two atoms first builds ``N^2``
intermediate tuples: a hub value of degree ``N`` shared by two atoms, and
a skewed input whose triangles lie on a diagonal. The star
``r1(x, y1), r2(x, y2), r3(x, y3)`` grouped by ``x`` has a cubic number of
instances and an aggregate computed in linear time by summing out each
``y`` on its own.
"""

from __future__ import annotations

from collections.abc import Sequence

from quivers.qiec.contracts import ParameterId, PresentationId
from quivers.qiec.identifiers import ConstructorId, FamilyId
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.rules import (
    Item,
    Judgment,
    JudgmentDecl,
    Premise,
    Rule,
    RulePresentation,
    SideCondition,
)
from quivers.qiec.types import IndexSum, IndexVariable
from tests.evidence import presentations as attachment

NAME = "bodies"

BEFORE = attachment.before(NAME)
"""The side condition ``left < right``."""


def _v(name: str) -> IndexVariable:
    """A natural-number variable.

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


def _form(name: str, arity: int) -> JudgmentDecl:
    """A form of natural-number arguments.

    Parameters
    ----------
    name : str
        Its name.
    arity : int
        Its number of arguments.

    Returns
    -------
    JudgmentDecl
        The form.
    """
    return JudgmentDecl(
        id=FamilyId.derive(NAME, name),
        name=name,
        indices=tuple(
            IndexBinder(name=f"x{index}", sort=NAT, refinable=True)
            for index in range(arity)
        ),
    )


FORMS = {
    name: _form(name, arity)
    for name, arity in (
        ("r", 2),
        ("s", 2),
        ("t", 2),
        ("r1", 2),
        ("r2", 2),
        ("r3", 2),
        ("u", 1),
        ("w", 1),
        ("tri", 3),
        ("total", 0),
        ("hub", 1),
        ("first", 1),
        ("summed", 1),
    )
}


def judgment(name: str, *variables: str) -> Judgment:
    """A judgment of one form over variables.

    Parameters
    ----------
    name : str
        The form.
    *variables : str
        Its arguments.

    Returns
    -------
    Judgment
        The judgment.
    """
    return Judgment(
        family=FORMS[name].id, arguments=tuple(_v(variable) for variable in variables)
    )


def _rule(
    name: str,
    conclusion: Judgment,
    premises: Sequence[Judgment],
    *,
    condition: SideCondition | None = None,
) -> Rule:
    """A rule whose telescope declares every variable it uses.

    Parameters
    ----------
    name : str
        The rule.
    conclusion : Judgment
        Its conclusion.
    premises : Sequence of Judgment
        Its premises.
    condition : SideCondition or None
        Its side condition.

    Returns
    -------
    Rule
        The rule.
    """
    names: list[str] = []
    for item in (*premises, conclusion):
        for argument in item.arguments:
            terms = argument.terms if isinstance(argument, IndexSum) else (argument,)
            for term in terms:
                assert isinstance(term, IndexVariable)
                if term.name not in names:
                    names.append(term.name)
    return Rule(
        id=ConstructorId.derive(NAME, name),
        name=name,
        telescope=tuple(IndexBinder(name=n, sort=NAT) for n in names),
        premises=tuple(Premise(judgment=item) for item in premises),
        conclusion=conclusion,
        side_condition=condition,
        weight=ParameterId.derive(NAME, name),
    )


TRIANGLE_BODY = (
    judgment("r", "a", "b"),
    judgment("s", "b", "c"),
    judgment("t", "a", "c"),
)

TRIANGLE = _rule("triangle", judgment("tri", "a", "b", "c"), TRIANGLE_BODY)
"""``tri(a, b, c) <- r(a, b), s(b, c), t(a, c)``: every instance is output."""

TRIANGLES = _rule("triangles", judgment("total"), TRIANGLE_BODY)
"""``total() <- r(a, b), s(b, c), t(a, c)``: the sum over every triangle."""

STAR = _rule(
    "star",
    judgment("hub", "x"),
    (judgment("r1", "x", "y1"), judgment("r2", "x", "y2"), judgment("r3", "x", "y3")),
)
"""``hub(x) <- r1(x, y1), r2(x, y2), r3(x, y3)``."""

FIRST = _rule(
    "first",
    judgment("first", "a"),
    (judgment("r", "a", "b"), judgment("s", "b", "c")),
    condition=SideCondition(computation=BEFORE.id, arguments=("a", "c")),
)
"""``first(a) <- r(a, b), s(b, c)`` when ``a < c``."""

SEPARATE = _rule(
    "separate", judgment("total"), (judgment("u", "a"), judgment("w", "b"))
)
"""``total() <- u(a), w(b)``: two variables summed out on their own."""

SUMMED = Rule(
    id=ConstructorId.derive(NAME, "summed"),
    name="summed",
    telescope=(IndexBinder(name="i", sort=NAT), IndexBinder(name="k", sort=NAT)),
    premises=(
        Premise(judgment=judgment("u", "i")),
        Premise(judgment=judgment("w", "k")),
        Premise(
            judgment=Judgment(
                family=FORMS["u"].id,
                arguments=(IndexSum(terms=(_v("i"), _v("k")), constant=0),),
            )
        ),
    ),
    conclusion=judgment("summed", "i"),
)
"""``summed(i) <- u(i), w(k), u(i + k)``: a premise that is a sum of two variables."""

PRESENTATION = RulePresentation(
    id=PresentationId.derive(NAME),
    name=NAME,
    judgments=tuple(FORMS.values()),
    rules=(TRIANGLE, TRIANGLES, STAR, FIRST, SEPARATE, SUMMED),
    conditions=(BEFORE,),
)
"""Every body."""


def items(name: str, rows: Sequence[tuple[int, ...]]) -> list[Item]:
    """Items of one form.

    Parameters
    ----------
    name : str
        The form.
    rows : Sequence of tuple of int
        Their arguments.

    Returns
    -------
    list of Item
        The items.
    """
    return [Item(family=FORMS[name].id, arguments=row) for row in rows]


def hub(size: int) -> list[Item]:
    """The hub input: ``0`` is joined to everything in each relation.

    Each relation is ``{(0, j)} | {(i, 0)} | {(0, 0)}`` for ``1 <= i, j <=
    size``, so ``r`` and ``s`` share ``size + 1`` tuples at ``b = 0`` on each
    side: joining them first gives ``(size + 1)^2`` tuples, while the
    triangles are the ``3 size + 1`` that pass through the hub,
    ``(0, j, 0)``, ``(i, 0, 0)``, ``(0, 0, j)`` and ``(0, 0, 0)``.

    Parameters
    ----------
    size : int
        The degree of the hub.

    Returns
    -------
    list of Item
        The items of ``r``, ``s`` and ``t``.
    """
    rows = [(0, j) for j in range(1, size + 1)]
    rows += [(i, 0) for i in range(1, size + 1)]
    rows.append((0, 0))
    return [item for name in ("r", "s", "t") for item in items(name, rows)]


def skew(size: int) -> list[Item]:
    """The skewed input: ``r = {(a, 0)}``, ``s = {(0, c)}``, ``t = {(a, a)}``.

    Joining ``r`` and ``s`` first gives ``size^2`` tuples; the triangles
    are the ``size`` tuples ``(a, 0, a)``.

    Parameters
    ----------
    size : int
        The size of each relation.

    Returns
    -------
    list of Item
        The items of ``r``, ``s`` and ``t``.
    """
    return (
        items("r", [(a, 0) for a in range(size)])
        + items("s", [(0, c) for c in range(size)])
        + items("t", [(a, a) for a in range(size)])
    )


def star(size: int) -> list[Item]:
    """The star input: ``x = 0`` has ``size`` partners in each relation.

    Each ``r_i`` is ``{(0, y)} | {(x, 0)}`` for ``0 <= y < size`` and
    ``1 <= x < size``, so the star has ``size^3 + size - 1`` instances.

    Parameters
    ----------
    size : int
        The degree of ``x = 0``.

    Returns
    -------
    list of Item
        The items of ``r1``, ``r2`` and ``r3``.
    """
    rows = [(0, y) for y in range(size)] + [(x, 0) for x in range(1, size)]
    return [item for name in ("r1", "r2", "r3") for item in items(name, rows)]
