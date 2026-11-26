"""Reachability presentations over input graphs, for plans and their work.

``reach(j) <- reach(i), edge(i, j)`` with the input ``reach(0)`` sums, for
each node, the weights of the paths to it. Chains, diamonds and skip
graphs vary the number of paths while the items and instances stay
linear in the edges. A second rule fills three premises with one item,
and a graph with a cycle derives an item from itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from quivers.qiec.contracts import (
    AbsoluteLogError,
    AccuracyContract,
    InterpretationId,
    Numerics,
    OccurrenceId,
    ParameterId,
    PresentationId,
    QuerySpec,
    ResourceBudget,
    Target,
    Weight,
)
from quivers.qiec.identifiers import ConstructorId, FamilyId
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.rules import (
    AxiomOccurrence,
    AxiomSlot,
    Item,
    Judgment,
    JudgmentDecl,
    Premise,
    Rule,
    RulePresentation,
)
from quivers.qiec.types import IndexLiteral, IndexVariable

NAME = "graphs"
REACH = FamilyId.derive(NAME, "reach")
EDGE = FamilyId.derive(NAME, "edge")
TRIPLE = FamilyId.derive(NAME, "triple")


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


def _binder(name: str, *, refinable: bool = False) -> IndexBinder:
    """A natural-number binder.

    Parameters
    ----------
    name : str
        Its name.
    refinable : bool
        Whether it is an index of a form.

    Returns
    -------
    IndexBinder
        The binder.
    """
    return IndexBinder(name=name, sort=NAT, refinable=refinable)


STEP = Rule(
    id=ConstructorId.derive(NAME, "step"),
    name="step",
    telescope=(_binder("i"), _binder("j")),
    premises=(
        Premise(judgment=Judgment(family=REACH, arguments=(_v("i"),))),
        Premise(judgment=Judgment(family=EDGE, arguments=(_v("i"), _v("j")))),
    ),
    conclusion=Judgment(family=REACH, arguments=(_v("j"),)),
    weight=ParameterId.derive(NAME, "step"),
)
"""``reach(j) <- reach(i), edge(i, j)``, premises in that order."""

START = AxiomSlot(
    id=ConstructorId.derive(NAME, "start"),
    name="start",
    telescope=(_binder("i"),),
    judgment=Judgment(family=REACH, arguments=(_v("i"),)),
)
EDGE_SLOT = AxiomSlot(
    id=ConstructorId.derive(NAME, "edge"),
    name="edge",
    telescope=(_binder("i"), _binder("j")),
    judgment=Judgment(family=EDGE, arguments=(_v("i"), _v("j"))),
)

TRIPLED = Rule(
    id=ConstructorId.derive(NAME, "tripled"),
    name="tripled",
    telescope=(_binder("i"),),
    premises=tuple(
        Premise(judgment=Judgment(family=REACH, arguments=(_v("i"),))) for _ in range(3)
    ),
    conclusion=Judgment(family=TRIPLE, arguments=(_v("i"),)),
    weight=ParameterId.derive(NAME, "tripled"),
)
"""``triple(i) <- reach(i), reach(i), reach(i)``: one item fills every premise."""

FORMS = (
    JudgmentDecl(id=REACH, name="reach", indices=(_binder("n", refinable=True),)),
    JudgmentDecl(
        id=EDGE,
        name="edge",
        indices=(_binder("a", refinable=True), _binder("b", refinable=True)),
    ),
    JudgmentDecl(id=TRIPLE, name="triple", indices=(_binder("n", refinable=True),)),
)

PRESENTATION = RulePresentation(
    id=PresentationId.derive(NAME),
    name=NAME,
    judgments=FORMS,
    rules=(STEP, TRIPLED),
    axioms=(START, EDGE_SLOT),
)
"""Reachability, and the tripled rule."""


def occurrences(edges: Sequence[tuple[int, int]]) -> list[AxiomOccurrence]:
    """The inputs: ``reach(0)`` and one occurrence per edge.

    Parameters
    ----------
    edges : Sequence of tuple of int and int
        The edges.

    Returns
    -------
    list of AxiomOccurrence
        The occurrences.
    """
    found = [
        AxiomOccurrence(
            slot=START.id,
            occurrence=OccurrenceId.derive(NAME, "start"),
            binding=(("i", 0),),
        )
    ]
    for index, (source, target) in enumerate(edges):
        found.append(
            AxiomOccurrence(
                slot=EDGE_SLOT.id,
                occurrence=edge_id(index),
                binding=(("i", source), ("j", target)),
            )
        )
    return found


def edge_id(index: int) -> OccurrenceId:
    """The identity of the input occurrence of one edge.

    Parameters
    ----------
    index : int
        The edge's position.

    Returns
    -------
    OccurrenceId
        Its identity.
    """
    return OccurrenceId.derive(NAME, "edge", index)


def chain(length: int) -> list[tuple[int, int]]:
    """``0 -> 1 -> ... -> length``.

    Parameters
    ----------
    length : int
        The number of edges.

    Returns
    -------
    list of tuple of int and int
        The edges.
    """
    return [(i, i + 1) for i in range(length)]


def diamonds(count: int) -> list[tuple[int, int]]:
    """``count`` diamonds in a row: ``2^count`` paths to the last node.

    Parameters
    ----------
    count : int
        The number of diamonds.

    Returns
    -------
    list of tuple of int and int
        The edges.
    """
    edges = []
    for k in range(count):
        top, left, right, bottom = 3 * k, 3 * k + 1, 3 * k + 2, 3 * k + 3
        edges += [(top, left), (top, right), (left, bottom), (right, bottom)]
    return edges


def skips(length: int) -> list[tuple[int, int]]:
    """Edges of length one and two: a Fibonacci number of paths.

    Parameters
    ----------
    length : int
        The last node.

    Returns
    -------
    list of tuple of int and int
        The edges.
    """
    return [(i, i + 1) for i in range(length)] + [(i, i + 2) for i in range(length - 1)]


def query() -> QuerySpec:
    """The weight query of the graphs.

    Returns
    -------
    QuerySpec
        The query.
    """
    return QuerySpec(
        presentation=PRESENTATION.id,
        interpretation=InterpretationId.derive(NAME, "weight"),
        observable=Weight(),
    )


def contract(*, recompute: bool = False) -> AccuracyContract:
    """An exact contract; a memory budget selects the recomputing backward pass.

    Parameters
    ----------
    recompute : bool
        Whether to set a memory budget.

    Returns
    -------
    AccuracyContract
        The contract.
    """
    return AccuracyContract(
        numerics=Numerics(dtype="float64", domain="log"),
        target=Target(quantity="path weight", metric=AbsoluteLogError()),
        budget=ResourceBudget(elements=10**9) if recompute else ResourceBudget(),
    )


def node(index: int) -> Item:
    """The item ``reach(index)``.

    Parameters
    ----------
    index : int
        The node.

    Returns
    -------
    Item
        The item.
    """
    return Item(family=REACH, arguments=(index,))


def edge_weights(
    edges: Sequence[tuple[int, int]], values: Sequence[float]
) -> dict[OccurrenceId, torch.Tensor]:
    """Input weights of the edges.

    Parameters
    ----------
    edges : Sequence of tuple of int and int
        The edges.
    values : Sequence of float
        One weight per edge.

    Returns
    -------
    dict of OccurrenceId to torch.Tensor
        The weights.
    """
    return {
        edge_id(index): torch.tensor(value, dtype=torch.float64)
        for index, value in zip(range(len(edges)), values, strict=True)
    }


ORIGIN = Rule(
    id=ConstructorId.derive(NAME, "origin"),
    name="origin",
    conclusion=Judgment(family=REACH, arguments=(IndexLiteral(value=0, sort=NAT),)),
    weight=ParameterId.derive(NAME, "origin"),
)
"""``reach(0) <-``: a rule without premises."""

SEEDED = PRESENTATION.with_(rules=(STEP, TRIPLED, ORIGIN))
"""Reachability whose start is a rule without premises as well as an input."""
