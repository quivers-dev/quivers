"""Linear cyclic systems, solved component by component through the algebra's closure.

The derived hypergraph of a run is split into strongly connected
components, visited so that every premise outside a component is final
before the component is. Inside a component every edge has at most one
premise in the component, so with the products in premise order an edge
concluding ``x`` from ``y`` contributes ``a ⊗ v(y) ⊗ c``, where ``a`` is
the product of the premises before ``y`` and ``c`` that of the premises
after it, then the rule weight. When every such edge has ``c`` the unit
the component is right linear, ``v = b ⊕ M v``, and its least solution is
``M* b``; when every one has ``a`` the unit it is left linear,
``v = b ⊕ v N``, with least solution ``b N*`` (Lehmann 1977). Neither
needs the product to commute. Only a component that mixes the sides, or
has an edge with both coefficients, does: with a commutative product
``a ⊗ v ⊗ c = (a ⊗ c) ⊗ v``, and the component is right linear.

Whether a coefficient is the unit is decided by the rule, not by the
numbers: ``a`` is the unit when the premise is the first, ``c`` when it
is the last and the rule has no weight.

References
----------
Lehmann, D. J. (1977). Algebraic structures for transitive closure.
*Theoretical Computer Science*, 4(1), 59-76.
[doi:10.1016/0304-3975(77)90056-1](https://doi.org/10.1016/0304-3975(77)90056-1)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import (
    DivergentClosureError,
    OrderPreservingSemiring,
    Star,
    UndecidedClosureError,
)
from quivers.core.laws import TIMES_COMMUTATIVE
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import OccurrenceId, ParameterId
from quivers.qiec.rules import Item, Rule, RulePresentation
from quivers.stochastic.forest import DerivationForest, ForestEdge, edge_weight

__all__ = [
    "Component",
    "CyclicRefusal",
    "CyclicRefused",
    "CyclicWork",
    "Side",
    "components",
    "solve",
]


type InputId = ParameterId | OccurrenceId
"""A rule's parameter, or an input occurrence."""

type Side = Literal["none", "right", "left", "commuted"]
"""How a component is solved: no cycle, ``M* b``, ``b N*``, or ``M* b``
after commuting both coefficients to the left."""


class CyclicRefusal(str, Enum):
    """Why a linear cyclic system has no solution here."""

    NONLINEAR = "nonlinear"
    SIDES = "sides"
    DIVERGENT = "divergent"
    UNDECIDED = "undecided"


class CyclicRefused(ValueError):
    """A cyclic system is refused, with the reason.

    Parameters
    ----------
    code : CyclicRefusal
        The kind of reason.
    reason : str
        The reason.
    """

    def __init__(self, code: CyclicRefusal, reason: str) -> None:
        super().__init__(f"{code.value}: {reason}")
        self.code = code
        self.reason = reason


class _Work:
    """Counts of the algebra's operations, in elements of the algebra."""

    def __init__(self, algebra: OrderPreservingSemiring) -> None:
        self.size = math.prod(algebra.element_shape)
        self.sums = 0
        self.products = 0
        self.pivots = 0

    def count(self, kind: Literal["sums", "products"], result: torch.Tensor) -> None:
        """Count one operation over a batch of elements.

        Parameters
        ----------
        kind : {"sums", "products"}
            The operation.
        result : torch.Tensor
            Its result, one element per operation.
        """
        setattr(self, kind, getattr(self, kind) + result.numel() // self.size)


class CyclicWork(QiecModel):
    """The algebra's operations a cyclic solve performed.

    Parameters
    ----------
    sums : int
        Sums, one per element summed into another.
    products : int
        Products, one per element.
    pivots : int
        Closures of one element, one per state eliminated.
    """

    sums: int = 0
    products: int = 0
    pivots: int = 0


class Component(QiecModel):
    """One strongly connected component of a run and how it was solved.

    Parameters
    ----------
    items : tuple of Item
        Its items.
    side : Side
        How it was solved.
    """

    items: tuple[Item, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    side: Side = "none"


def components(forest: DerivationForest) -> list[list[Item]]:
    """The strongly connected components, premises' components first.

    Tarjan's algorithm on the graph from each premise to its conclusion,
    without recursion. It emits a component after every component it
    reaches, so the reverse of its order puts premises first.

    Parameters
    ----------
    forest : DerivationForest
        The derived hypergraph.

    Returns
    -------
    list of list of Item
        The components, each in discovery order, premises' components
        before their conclusions'.
    """
    users: dict[Item, list[Item]] = {}
    for item in forest.items:
        for edge in forest.edges.get(item, ()):
            for premise in edge.premises:
                users.setdefault(premise, []).append(item)
    index: dict[Item, int] = {}
    low: dict[Item, int] = {}
    on_stack: set[Item] = set()
    stack: list[Item] = []
    found: list[list[Item]] = []
    counter = 0
    for root in forest.items:
        if root in index:
            continue
        work: list[tuple[Item, int]] = [(root, 0)]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            item, position = work[-1]
            successors = users.get(item, [])
            if position < len(successors):
                work[-1] = (item, position + 1)
                successor = successors[position]
                if successor not in index:
                    index[successor] = low[successor] = counter
                    counter += 1
                    stack.append(successor)
                    on_stack.add(successor)
                    work.append((successor, 0))
                elif successor in on_stack:
                    low[item] = min(low[item], index[successor])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[item])
            if low[item] == index[item]:
                component: list[Item] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == item:
                        break
                order = {member: forest.items.index(member) for member in component}
                found.append(sorted(component, key=order.__getitem__))
    found.reverse()
    return found


def _product(
    algebra: OrderPreservingSemiring,
    factors: Sequence[torch.Tensor],
    work: _Work,
) -> torch.Tensor | None:
    """The product of factors in order, ``None`` for none.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    factors : Sequence of torch.Tensor
        The factors.
    work : _Work
        The counts.

    Returns
    -------
    torch.Tensor or None
        The product, or ``None`` standing for the unit.
    """
    if not factors:
        return None
    total = factors[0]
    for factor in factors[1:]:
        total = algebra.tensor_op(total, factor)
        work.count("products", total)
    return total


def _times(
    algebra: OrderPreservingSemiring, a: torch.Tensor, b: torch.Tensor, work: _Work
) -> torch.Tensor:
    """One counted product.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    a, b : torch.Tensor
        The factors, broadcast together.
    work : _Work
        The counts.

    Returns
    -------
    torch.Tensor
        ``a ⊗ b``.
    """
    result = algebra.tensor_op(a, b)
    work.count("products", result)
    return result


def _add(
    algebra: OrderPreservingSemiring,
    current: torch.Tensor | None,
    term: torch.Tensor,
    work: _Work,
) -> torch.Tensor:
    """Add a term to a running sum.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    current : torch.Tensor or None
        The sum so far.
    term : torch.Tensor
        The term.
    work : _Work
        The counts.

    Returns
    -------
    torch.Tensor
        ``current ⊕ term``.
    """
    if current is None:
        return term
    result = algebra.join(torch.stack(torch.broadcast_tensors(current, term)), 0)
    work.count("sums", result)
    return result


def _closure(
    algebra: Star, matrix: torch.Tensor, like: torch.Tensor, work: _Work
) -> torch.Tensor:
    """The closure ``A*`` of a square matrix of elements, by eliminating states.

    For each state ``k`` in turn, with ``s = (A_kk)*`` the closure of one
    element, every entry becomes ``A_ij ⊕ A_ik ⊗ s ⊗ A_kj`` from the
    matrix before the step; after the last step the matrix is ``A⁺``,
    the sum of the paths of positive length, and ``A* = I ⊕ A⁺``
    (Lehmann 1977). Each step closes one element and does ``n + n²``
    products and ``n²`` sums, and ``I ⊕ A⁺`` ``n²`` sums more, so the
    closure is ``n`` closures of one element, ``n³ + n²`` products and
    ``n³ + n²`` sums. The closure of one
    element is the algebra's own, which refuses outside its domain.

    Parameters
    ----------
    algebra : Star
        The algebra.
    matrix : torch.Tensor
        Shape ``(n, n, *element_shape)``.
    like : torch.Tensor
        The dtype and device of the unit.
    work : _Work
        The counts.

    Returns
    -------
    torch.Tensor
        The closure.

    Raises
    ------
    DivergentClosureError
        If an element's closure has no value.
    UndecidedClosureError
        If rounding cannot decide whether it has.
    """
    n = matrix.shape[0]
    current = matrix
    for k in range(n):
        star = algebra.star(current[k : k + 1, k : k + 1])[0, 0]
        work.pivots += 1
        column = _times(algebra, current[:, k], star.unsqueeze(0), work)
        update = _times(algebra, column.unsqueeze(1), current[k].unsqueeze(0), work)
        current = _add(algebra, current, update, work)
    zero = algebra.zero_like((n, n), like)
    unit = algebra.unit_like((n,), like)
    identity = zero.clone()
    for i in range(n):
        identity[i, i] = unit[i]
    return _add(algebra, identity, current, work)


def _unit_or(
    algebra: OrderPreservingSemiring, value: torch.Tensor | None, like: torch.Tensor
) -> torch.Tensor:
    """A product, or the unit for ``None``.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    value : torch.Tensor or None
        The product.
    like : torch.Tensor
        The dtype and device of the unit.

    Returns
    -------
    torch.Tensor
        The value.
    """
    return algebra.unit_like((), like) if value is None else value


def solve(
    forest: DerivationForest,
    algebra: OrderPreservingSemiring,
    inputs: Mapping[InputId, torch.Tensor],
    *,
    like: torch.Tensor,
) -> tuple[dict[Item, torch.Tensor], tuple[Component, ...], CyclicWork]:
    """The least solution of a linear system over a derived hypergraph.

    Parameters
    ----------
    forest : DerivationForest
        The derived hypergraph, every instance and input an edge.
    algebra : OrderPreservingSemiring
        The algebra; it is a `Star` whose closure is the least solution.
    inputs : Mapping of InputId to torch.Tensor
        The weights.
    like : torch.Tensor
        The dtype and device of units and zeros.

    Returns
    -------
    tuple of dict, tuple of Component and CyclicWork
        The value of every item, each component with its side, and the
        algebra's operations: building each component's system, its
        closure, and the product of the closure with the constants.

    Raises
    ------
    CyclicRefused
        If an edge has two premises in its component, a component mixes
        the sides without a commutative product, or a closure diverges or
        cannot be decided.
    TypeError
        If the algebra has no closure.
    """
    if not isinstance(algebra, Star):
        raise TypeError(f"{algebra.subject.name} has no closure")
    commutes = any(
        claim.law.name == TIMES_COMMUTATIVE.name and claim.established
        for claim in algebra.claims()
    )
    presentation = forest.presentation
    work = _Work(algebra)
    values: dict[Item, torch.Tensor] = {}
    solved: list[Component] = []
    for members in components(forest):
        inside = set(members)
        position = {item: index for index, item in enumerate(members)}
        constants: dict[Item, torch.Tensor] = {}
        loops: list[
            tuple[Item, Item, torch.Tensor | None, torch.Tensor | None, ForestEdge]
        ] = []
        for item in members:
            for edge in forest.edges.get(item, ()):
                weight = edge_weight(forest, edge, algebra, inputs, like=like)
                if edge.occurrence is not None:
                    constants[item] = _add(algebra, constants.get(item), weight, work)
                    continue
                places = [k for k, p in enumerate(edge.premises) if p in inside]
                if not places:
                    term = _product(
                        algebra, [*(values[p] for p in edge.premises), weight], work
                    )
                    assert term is not None
                    constants[item] = _add(algebra, constants.get(item), term, work)
                    continue
                if len(places) > 1:
                    raise CyclicRefused(
                        CyclicRefusal.NONLINEAR,
                        f"an instance of {_name(presentation, edge)} concluding "
                        f"{item.arguments} has {len(places)} premises in its own "
                        "component",
                    )
                (k,) = places
                rule = presentation.constructors_by_id[edge.constructor]
                assert isinstance(rule, Rule)
                before = _product(algebra, [values[p] for p in edge.premises[:k]], work)
                after_factors = [values[p] for p in edge.premises[k + 1 :]]
                if rule.weight is not None:
                    after_factors.append(weight)
                after = _product(algebra, after_factors, work)
                loops.append((item, edge.premises[k], before, after, edge))
        if not loops:
            for item in members:
                values[item] = constants.get(item, algebra.zero_like((), like))
            solved.append(Component(items=tuple(members), side="none"))
            continue
        side = _side(loops, commutes, presentation)
        n = len(members)
        matrix = algebra.zero_like((n, n), like)
        entries: dict[tuple[int, int], torch.Tensor] = {}
        for conclusion, premise, before, after, _ in loops:
            x, y = position[conclusion], position[premise]
            if side == "right":
                key, term = (x, y), _unit_or(algebra, before, like)
            elif side == "left":
                key, term = (y, x), _unit_or(algebra, after, like)
            else:
                key = (x, y)
                term = _times(
                    algebra,
                    _unit_or(algebra, before, like),
                    _unit_or(algebra, after, like),
                    work,
                )
            entries[key] = _add(algebra, entries.get(key), term, work)
        rows = [
            torch.stack([entries.get((i, j), matrix[i, j]) for j in range(n)])
            for i in range(n)
        ]
        filled = torch.stack(rows)
        try:
            closure = _closure(algebra, filled, like, work)
        except DivergentClosureError as error:
            raise CyclicRefused(CyclicRefusal.DIVERGENT, str(error)) from error
        except UndecidedClosureError as error:
            raise CyclicRefused(CyclicRefusal.UNDECIDED, str(error)) from error
        zero = algebra.zero_like((), like)
        constant = [constants.get(item, zero) for item in members]
        for i, item in enumerate(members):
            total: torch.Tensor | None = None
            for j in range(n):
                if side == "left":
                    term = _times(algebra, constant[j], closure[j, i], work)
                else:
                    term = _times(algebra, closure[i, j], constant[j], work)
                total = _add(algebra, total, term, work)
            assert total is not None
            values[item] = total
        solved.append(Component(items=tuple(members), side=side))
    return (
        values,
        tuple(solved),
        CyclicWork(sums=work.sums, products=work.products, pivots=work.pivots),
    )


def _name(presentation: RulePresentation, edge: ForestEdge) -> str:
    """The name of an edge's rule or slot.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    edge : ForestEdge
        The edge.

    Returns
    -------
    str
        The name.
    """
    return presentation.constructors_by_id[edge.constructor].name


def _side(
    loops: Sequence[
        tuple[Item, Item, torch.Tensor | None, torch.Tensor | None, ForestEdge]
    ],
    commutes: bool,
    presentation: RulePresentation,
) -> Side:
    """The side a component is solved on.

    Parameters
    ----------
    loops : Sequence of tuple
        Each edge inside the component: its conclusion, its premise in the
        component, the coefficients before and after it (``None`` for the
        unit), and the edge.
    commutes : bool
        Whether the product is established to commute.
    presentation : RulePresentation
        The presentation, for the reason of a refusal.

    Returns
    -------
    Side
        ``"right"`` when every coefficient after is the unit, else
        ``"left"`` when every coefficient before is, else ``"commuted"``.

    Raises
    ------
    CyclicRefused
        If neither side holds and the product is not established to
        commute.
    """
    if all(after is None for _, _, _, after, _ in loops):
        return "right"
    if all(before is None for _, _, before, _, _ in loops):
        return "left"
    if commutes:
        return "commuted"
    left = next(edge for _, _, _, after, edge in loops if after is not None)
    right = next(edge for _, _, before, _, edge in loops if before is not None)
    raise CyclicRefused(
        CyclicRefusal.SIDES,
        f"{_name(presentation, left)} multiplies its premise in the component on "
        f"the right and {_name(presentation, right)} on the left, and the product "
        "is not established to commute",
    )
