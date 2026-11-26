"""Cyclic systems, solved component by component: linear ones by the
algebra's closure, monotone polynomial ones by decomposed Newton.

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

A monotone polynomial system, whose components may have edges with
several premises inside them, is solved in the reals of the algebra's
`quivers.core.algebras.RealEmbedding` by `solve_polynomial`: Newton's
method from zero on each component keeps a certified lower bound, a
pre-fixed point bounds it from above, both in outward rounded arithmetic,
and the value carries the implicit derivative of the fixed point.

References
----------
Etessami, K., and Yannakakis, M. (2009). Recursive Markov chains,
stochastic grammars, and monotone systems of nonlinear equations.
*Journal of the ACM* 56(1), 1-66.
[doi:10.1145/1462153.1462154](https://doi.org/10.1145/1462153.1462154)

Esparza, J., Kiefer, S., and Luttenberger, M. (2010). Computing the least
fixed point of positive polynomial systems. *SIAM Journal on Computing*
39(6), 2282-2335.
[doi:10.1137/090749591](https://doi.org/10.1137/090749591)

Lehmann, D. J. (1977). Algebraic structures for transitive closure.
*Theoretical Computer Science*, 4(1), 59-76.
[doi:10.1016/0304-3975(77)90056-1](https://doi.org/10.1016/0304-3975(77)90056-1)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from enum import Enum
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.core.algebras import (
    DivergentClosureError,
    OrderPreservingSemiring,
    RealEmbedding,
    Star,
    UndecidedClosureError,
)
from quivers.core.laws import TIMES_COMMUTATIVE
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import OccurrenceId, ParameterId
from quivers.qiec.rules import Item, Rule, RulePresentation
from quivers.stochastic.forest import DerivationForest, ForestEdge, edge_weight

__all__ = [
    "Divergent",
    "NewtonComponent",
    "PolynomialSolution",
    "solve_polynomial",
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
        self.exact = 0

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
        States the closures eliminated, each a closure of one element.
    exact : int
        Operations of the closures' exact or enclosed decisions.
    """

    sums: int = 0
    products: int = 0
    pivots: int = 0
    exact: int = 0


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
        algebra's operations of building each component's system, closing
        it (as the closure reports its own work) and applying the closure
        to the constants.

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
            closure, closing = algebra.star_with_work(filled)
            work.sums += closing.sums
            work.products += closing.products
            work.pivots += closing.pivots
            work.exact += closing.exact
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
        CyclicWork(
            sums=work.sums,
            products=work.products,
            pivots=work.pivots,
            exact=work.exact,
        ),
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


# ---------------------------------------------------------------------------
# monotone polynomial systems


class Divergent(ValueError):
    """A monotone polynomial system whose least fixed point is infinite.

    Parameters
    ----------
    reason : str
        The certificate of divergence.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class NewtonComponent(QiecModel):
    """One strongly connected component solved by Newton's method.

    Parameters
    ----------
    items : tuple of Item
        Its items.
    iterations : int
        Newton steps taken.
    spectral : float or None
        An estimate of the spectral radius of the Jacobian at the solution,
        ``max_x (J s)_x / s_x`` for ``s = (I - J)^-1 1``; ``None`` when it
        has no inverse there.
    condition : float or None
        ``||(I - J)^-1||`` in the maximum norm at the solution, ``max_x s_x``.
    certified : bool
        Whether both ends of every item's interval were certified.
    implicit : bool
        Whether the spectral radius at the upper end is certified below
        one, so that the implicit derivative exists.
    """

    items: tuple[Item, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    iterations: int = 0
    spectral: float | None = None
    condition: float | None = None
    certified: bool = False
    implicit: bool = False


class PolynomialSolution(ValueModel):
    """The least fixed point of a monotone polynomial system, with its enclosure.

    Parameters
    ----------
    values : dict of Item to torch.Tensor
        The value of every item in the algebra, with the implicit
        derivative in its inputs.
    lower : dict of Item to float
        A certified lower bound of every item's real value.
    upper : dict of Item to float
        A certified upper bound, ``inf`` where none was certified.
    components : tuple of NewtonComponent
        The components, premises' components first.
    iterations : int
        Newton steps over all components.
    sums : int
        Sums of reals in point arithmetic.
    products : int
        Products of reals in point arithmetic.
    exact : int
        Operations in outward rounded arithmetic.
    stopped : str
        ``"completed"`` when every component's iteration stopped making
        progress, ``"budget"`` when the iteration budget ended it.
    """

    values: dict[Item, torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    lower: dict[Item, float] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    upper: dict[Item, float] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    components: tuple[NewtonComponent, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    iterations: int = 0
    sums: int = 0
    products: int = 0
    exact: int = 0
    stopped: Literal["completed", "budget"] = "completed"


def _down(value: float) -> float:
    """A lower bound of the exact result of the operation that rounded to ``value``.

    Parameters
    ----------
    value : float
        A result of one rounded operation on nonnegative operands.

    Returns
    -------
    float
        The float below it, and never below zero.

    """
    return max(math.nextafter(value, -math.inf), 0.0)


def _up(value: float) -> float:
    """An upper bound of the exact result of the operation that rounded to ``value``.

    Parameters
    ----------
    value : float
        A result of one rounded operation.

    Returns
    -------
    float
        The float above it.
    """
    return math.nextafter(value, math.inf)


class _Term:
    """One monomial: a coefficient and the members it multiplies.

    Parameters
    ----------
    coefficient : torch.Tensor
        The real coefficient, on the autograd graph.
    lower : float
        A certified lower bound of it.
    upper : float
        A certified upper bound of it.
    inside : tuple of int
        The positions of the members it multiplies, with repeats.
    """

    def __init__(
        self,
        coefficient: torch.Tensor,
        lower: float,
        upper: float,
        inside: tuple[int, ...],
    ) -> None:
        self.coefficient = coefficient
        self.point = float(coefficient.detach())
        self.lower = lower
        self.upper = upper
        self.inside = inside


class _Count:
    """Counts of the Newton solver."""

    def __init__(self) -> None:
        self.sums = 0
        self.products = 0
        self.exact = 0
        self.iterations = 0


def _evaluate(
    terms: Sequence[Sequence[_Term]], point: Sequence[float], count: _Count
) -> list[float]:
    """The polynomials at a point, in floating point.

    Parameters
    ----------
    terms : Sequence of Sequence of _Term
        The monomials of each member.
    point : Sequence of float
        The members' values.
    count : _Count
        The counts.

    Returns
    -------
    list of float
        Each member's polynomial.
    """
    found: list[float] = []
    for monomials in terms:
        total = 0.0
        for term in monomials:
            value = term.point
            for position in term.inside:
                value *= point[position]
                count.products += 1
            total += value
            count.sums += 1
        found.append(total)
    return found


def _bound(
    terms: Sequence[Sequence[_Term]],
    point: Sequence[float],
    count: _Count,
    *,
    upper: bool,
) -> list[float]:
    """The polynomials at a point, rounded outward.

    Parameters
    ----------
    terms : Sequence of Sequence of _Term
        The monomials.
    point : Sequence of float
        The members' values, nonnegative and exact.
    count : _Count
        The counts.
    upper : bool
        Whether upper bounds are computed, from the coefficients' upper
        bounds; else lower ones from their lower bounds.

    Returns
    -------
    list of float
        A bound of each member's polynomial.
    """
    step = _up if upper else _down
    found: list[float] = []
    for monomials in terms:
        total = 0.0
        for term in monomials:
            value = term.upper if upper else term.lower
            for position in term.inside:
                value = step(value * point[position])
                count.exact += 1
            total = step(total + value)
            count.exact += 1
        found.append(total)
    return found


def _jacobian(
    terms: Sequence[Sequence[_Term]],
    point: Sequence[float],
    count: _Count,
    *,
    bound: Literal["point", "lower", "upper"],
) -> list[list[float]]:
    """The Jacobian of the polynomials at a point.

    Parameters
    ----------
    terms : Sequence of Sequence of _Term
        The monomials.
    point : Sequence of float
        The members' values.
    count : _Count
        The counts.
    bound : {"point", "lower", "upper"}
        Floating point, or outward rounded from the coefficients' lower or
        upper bounds.

    Returns
    -------
    list of list of float
        ``J[x][y]``, the derivative of member ``x``'s polynomial in member
        ``y``.
    """
    size = len(point)
    matrix = [[0.0] * size for _ in range(size)]
    step = _up if bound == "upper" else _down
    for row, monomials in enumerate(terms):
        for term in monomials:
            for place, position in enumerate(term.inside):
                value = (
                    term.point
                    if bound == "point"
                    else term.upper
                    if bound == "upper"
                    else term.lower
                )
                for other, factor in enumerate(term.inside):
                    if other == place:
                        continue
                    value = value * point[factor]
                    if bound != "point":
                        value = step(value)
                        count.exact += 1
                    else:
                        count.products += 1
                total = matrix[row][position] + value
                matrix[row][position] = total if bound == "point" else step(total)
                count.exact += bound != "point"
                count.sums += bound == "point"
    return matrix


def _star_ones(matrix: Sequence[Sequence[float]], count: _Count) -> list[float] | None:
    """``(I - J)^-1 1`` in floating point, when it is positive.

    Parameters
    ----------
    matrix : Sequence of Sequence of float
        ``J``.
    count : _Count
        The counts: a dense solve of order ``S`` as ``S^3`` products and
        ``S^3`` sums.

    Returns
    -------
    list of float or None
        The vector, or ``None`` when ``I - J`` is singular or the solution
        is not positive.
    """
    size = len(matrix)
    count.products += size**3
    count.sums += size**3
    system = torch.eye(size, dtype=torch.float64) - torch.tensor(
        matrix, dtype=torch.float64
    ).reshape(size, size)
    try:
        solution = torch.linalg.solve(system, torch.ones(size, dtype=torch.float64))
    except RuntimeError:
        return None
    if not bool(torch.isfinite(solution).all()) or not bool((solution > 0).all()):
        return None
    return [float(value) for value in solution]


def _contracts(
    matrix: Sequence[Sequence[float]], vector: Sequence[float], count: _Count
) -> bool:
    """Whether ``J v < v`` holds in outward rounded arithmetic, ``v`` positive.

    For a nonnegative matrix it shows the spectral radius below one
    (Collatz and Wielandt: ``rho(J) <= max_x (J v)_x / v_x``).

    Parameters
    ----------
    matrix : Sequence of Sequence of float
        Upper bounds of ``J``.
    vector : Sequence of float
        ``v``, positive.
    count : _Count
        The counts.

    Returns
    -------
    bool
        Whether every row's upper bound is below ``v``'s entry.
    """
    for row, value in zip(matrix, vector, strict=True):
        total = 0.0
        for entry, factor in zip(row, vector, strict=True):
            total = _up(total + _up(entry * factor))
            count.exact += 2
        if not total < value:
            return False
    return True


def _expands(
    matrix: Sequence[Sequence[float]], vector: Sequence[float], count: _Count
) -> bool:
    """Whether ``J v >= v`` holds in outward rounded arithmetic, ``v`` positive.

    For a nonnegative matrix it shows the spectral radius at least one
    (Collatz and Wielandt: ``rho(J) >= min_x (J v)_x / v_x``).

    Parameters
    ----------
    matrix : Sequence of Sequence of float
        Lower bounds of ``J``.
    vector : Sequence of float
        ``v``, positive.
    count : _Count
        The counts.

    Returns
    -------
    bool
        Whether every row's lower bound is at least ``v``'s entry.
    """
    for row, value in zip(matrix, vector, strict=True):
        total = 0.0
        for entry, factor in zip(row, vector, strict=True):
            total = _down(total + _down(entry * factor))
            count.exact += 2
        if total < value:
            return False
    return True


def _perron(matrix: Sequence[Sequence[float]]) -> list[float] | None:
    """A positive estimate of the Perron vector of a nonnegative matrix.

    Parameters
    ----------
    matrix : Sequence of Sequence of float
        The matrix.

    Returns
    -------
    list of float or None
        The eigenvector of the eigenvalue of largest real part, scaled to
        be positive, or ``None`` when it has a nonpositive entry.
    """
    size = len(matrix)
    values, vectors = torch.linalg.eig(
        torch.tensor(matrix, dtype=torch.float64).reshape(size, size)
    )
    vector = vectors[:, int(torch.argmax(values.real))].real
    vector = vector if float(vector.sum()) >= 0 else -vector
    if not bool((vector > 0).all()):
        return None
    return [float(value) for value in vector]


_SHRINK = (2.0**-40, 2.0**-20, 2.0**-10, 0.5)
"""The fractions a Newton step is shortened by until it certifies."""


def _newton(
    terms: Sequence[Sequence[_Term]],
    count: _Count,
    budget: int | None,
) -> tuple[list[float], list[float], int, bool]:
    """Certified Newton iteration from zero on one component.

    From a certified lower bound ``l`` of the least fixed point ``m`` and
    the Jacobian ``J`` at ``l``: Taylor's expansion of a polynomial with
    nonnegative coefficients gives ``(I - J)(m - l) >= f(l) - l``; when
    ``rho(J) < 1``, ``(I - J)^-1`` is nonnegative, so any ``d`` with
    ``(I - J) d <= f(l) - l`` has ``l + d <= m``. The floating Newton step
    is shortened until both hypotheses hold in outward rounded arithmetic,
    and ``l`` moves to ``l + d`` rounded down.

    Parameters
    ----------
    terms : Sequence of Sequence of _Term
        The component's polynomials.
    count : _Count
        The counts.
    budget : int or None
        The most steps.

    Returns
    -------
    tuple
        The certified lower bound, the last floating iterate, the steps
        taken, and whether the budget ended the iteration.

    Raises
    ------
    Divergent
        If the spectral radius at a lower bound strictly below every
        component's value is certified at least one, so the least fixed
        point is infinite.
    """
    size = len(terms)
    lower = [0.0] * size
    steps = 0
    while True:
        if budget is not None and steps >= budget:
            return lower, _estimate(terms, lower, count, 1), steps, True
        lowest = _bound(terms, lower, count, upper=False)
        lower_jacobian = _jacobian(terms, lower, count, bound="lower")
        direction = _solve(
            lower_jacobian,
            [low - here for low, here in zip(lowest, lower, strict=True)],
            count,
        )
        ones = _star_ones(_jacobian(terms, lower, count, bound="point"), count)
        upper_jacobian = _jacobian(terms, lower, count, bound="upper")
        if ones is None or not _contracts(upper_jacobian, ones, count):
            _diverges(terms, lower, count)
            return (
                lower,
                _estimate(terms, lower, count, _remaining(budget, steps)),
                steps,
                False,
            )
        if direction is None:
            return (
                lower,
                _estimate(terms, lower, count, _remaining(budget, steps)),
                steps,
                False,
            )
        moved = False
        for shrink in _SHRINK:
            candidate = _valid_step(
                lower_jacobian,
                [max(value, 0.0) * (1.0 - shrink) for value in direction],
                lowest,
                lower,
                count,
            )
            advanced = [
                max(here, _down(here + step))
                for here, step in zip(lower, candidate, strict=True)
            ]
            if advanced != lower:
                moved = True
                lower = advanced
                break
        steps += 1
        count.iterations += 1
        if not moved:
            return (
                lower,
                _estimate(terms, lower, count, _remaining(budget, steps)),
                steps,
                False,
            )


def _remaining(budget: int | None, steps: int) -> int:
    """The floating steps left for the estimate.

    Parameters
    ----------
    budget : int or None
        The most steps.
    steps : int
        The steps taken.

    Returns
    -------
    int
        At most sixty-four, and no more than the budget leaves.
    """
    return 64 if budget is None else max(min(64, budget - steps), 0)


def _estimate(
    terms: Sequence[Sequence[_Term]],
    start: Sequence[float],
    count: _Count,
    limit: int,
) -> list[float]:
    """Floating Newton steps from a certified lower bound until they stall.

    Parameters
    ----------
    terms : Sequence of Sequence of _Term
        The polynomials.
    start : Sequence of float
        The lower bound.
    count : _Count
        The counts.
    limit : int
        The most steps.

    Returns
    -------
    list of float
        The last iterate.
    """
    point = list(start)
    for _ in range(limit):
        following = _step(terms, point, count)
        count.iterations += 1
        if following is None or following == point:
            break
        point = following
    return point


def _valid_step(
    lower_jacobian: Sequence[Sequence[float]],
    step: Sequence[float],
    lowest: Sequence[float],
    lower: Sequence[float],
    count: _Count,
) -> list[float]:
    """Lower a step until ``(I - J) d <= f(l) - l`` holds exactly in each row.

    Each row is required as ``d_x <= (f(l)_x - l_x) + (J d)_x`` with the
    right side a lower bound computed in outward rounded arithmetic and
    compared in exact rational arithmetic, and ``d_x`` is lowered to the
    largest float that satisfies it. Lowering one entry can only lower
    the right side of the others, so the sweeps repeat until none lowers
    anything; when ``2 S + 2`` sweeps do not settle, the step is zero,
    which always holds, and the caller shortens the proposal instead.

    Parameters
    ----------
    lower_jacobian : Sequence of Sequence of float
        Lower bounds of the Jacobian of the lower system at ``l``.
    step : Sequence of float
        The proposed step, nonnegative.
    lowest : Sequence of float
        Lower bounds of the lower system at ``l``.
    lower : Sequence of float
        ``l``.
    count : _Count
        The counts.

    Returns
    -------
    list of float
        A step every row of which holds, zero when the sweeps do not
        settle.
    """
    current = list(step)
    for _ in range(2 * len(current) + 2):
        changed = False
        for row, (entries, low, here) in enumerate(
            zip(lower_jacobian, lowest, lower, strict=True)
        ):
            total = 0.0
            for entry, factor in zip(entries, current, strict=True):
                total = _down(total + _down(entry * factor))
                count.exact += 2
            allowed = Fraction(low) - Fraction(here) + Fraction(total)
            count.exact += 1
            if Fraction(current[row]) <= allowed:
                continue
            value = max(float(allowed), 0.0)
            while value > 0 and Fraction(value) > allowed:
                value = math.nextafter(value, -math.inf)
            current[row] = max(value, 0.0) if allowed >= 0 else 0.0
            changed = True
        if not changed:
            return current
    return [0.0] * len(current)


def _step(
    terms: Sequence[Sequence[_Term]], point: Sequence[float], count: _Count
) -> list[float] | None:
    """One floating Newton step from a point.

    Parameters
    ----------
    terms : Sequence of Sequence of _Term
        The polynomials.
    point : Sequence of float
        The point.
    count : _Count
        The counts.

    Returns
    -------
    list of float or None
        The next iterate, or ``None`` when ``I - J`` is singular.
    """
    values = _evaluate(terms, point, count)
    direction = _solve(
        _jacobian(terms, point, count, bound="point"),
        [value - here for value, here in zip(values, point, strict=True)],
        count,
    )
    if direction is None:
        return None
    return [here + max(step, 0.0) for here, step in zip(point, direction, strict=True)]


def _solve(
    matrix: Sequence[Sequence[float]], right: Sequence[float], count: _Count
) -> list[float] | None:
    """``(I - J)^-1 r`` in floating point.

    Parameters
    ----------
    matrix : Sequence of Sequence of float
        ``J``.
    right : Sequence of float
        ``r``.
    count : _Count
        The counts: a dense solve of order ``S`` as ``S^3`` products and
        ``S^3`` sums.

    Returns
    -------
    list of float or None
        The solution, or ``None`` when it is singular or not finite.
    """
    size = len(matrix)
    count.products += size**3
    count.sums += size**3
    system = torch.eye(size, dtype=torch.float64) - torch.tensor(
        matrix, dtype=torch.float64
    ).reshape(size, size)
    try:
        solution = torch.linalg.solve(
            system, torch.tensor(list(right), dtype=torch.float64)
        )
    except RuntimeError:
        return None
    if not bool(torch.isfinite(solution).all()):
        return None
    return [float(value) for value in solution]


def _diverges(
    terms: Sequence[Sequence[_Term]], lower: Sequence[float], count: _Count
) -> None:
    """Refuse a component whose least fixed point is certified infinite.

    If ``f(l) > l`` in every member, then ``l`` lies strictly below the
    least fixed point ``m``, as ``m = f(m) >= f(l) > l``. Below a finite
    least fixed point of a strongly connected clean system the spectral
    radius of the Jacobian is below one (Esparza, Kiefer and Luttenberger
    2010, section 4). So a spectral radius certified at least one there
    shows ``m`` infinite.

    Parameters
    ----------
    terms : Sequence of Sequence of _Term
        The polynomials.
    lower : Sequence of float
        A certified lower bound.
    count : _Count
        The counts.

    Raises
    ------
    Divergent
        If divergence is certified.
    """
    gaps = [
        math.nextafter(low - here, -math.inf)
        for low, here in zip(
            _bound(terms, lower, count, upper=False), lower, strict=True
        )
    ]
    if min(gaps, default=0.0) <= 0:
        return
    jacobian = _jacobian(terms, lower, count, bound="lower")
    vector = _perron(_jacobian(terms, lower, count, bound="point"))
    if vector is not None and _expands(jacobian, vector, count):
        raise Divergent(
            "the Jacobian's spectral radius is certified at least one below the "
            "least fixed point, so the system's least fixed point is infinite"
        )


def _upper(
    terms: Sequence[Sequence[_Term]],
    estimate: Sequence[float],
    lower: Sequence[float],
    count: _Count,
) -> tuple[list[float] | None, float | None, float | None, bool]:
    """A certified pre-fixed point above an estimate, and the conditioning there.

    ``u = e + t s`` for ``s = (I - J(e))^-1 1`` and growing ``t``: when
    ``f(u) <= u`` in outward rounded arithmetic, ``u`` is a pre-fixed
    point and bounds the least fixed point from above.

    Parameters
    ----------
    terms : Sequence of Sequence of _Term
        The polynomials.
    estimate : Sequence of float
        The estimate ``e``.
    lower : Sequence of float
        The certified lower bound.
    count : _Count
        The counts.

    Returns
    -------
    tuple
        The upper bound or ``None``; the spectral radius estimate and the
        condition number at the estimate, or ``None`` when ``I - J`` has no
        positive inverse there; and whether the spectral radius at the
        upper bound is certified below one.
    """
    base = [max(value, low) for value, low in zip(estimate, lower, strict=True)]
    point = _jacobian(terms, base, count, bound="point")
    ones = _star_ones(point, count)
    if ones is None:
        return None, None, None, False
    spectral = max(
        sum(entry * factor for entry, factor in zip(row, ones, strict=True)) / own
        for row, own in zip(point, ones, strict=True)
    )
    condition = max(ones)
    values = _evaluate(terms, base, count)
    residual = max(
        (abs(value - here) for value, here in zip(values, base, strict=True)),
        default=0.0,
    )
    scale = max(max(base, default=0.0), 1e-300)
    width = max(2.0 * residual, scale * 2.0**-45)
    for _ in range(96):
        candidate = [
            _up(here + width * own) for here, own in zip(base, ones, strict=True)
        ]
        images = _bound(terms, candidate, count, upper=True)
        if all(image <= own for image, own in zip(images, candidate, strict=True)):
            implicit = _contracts(
                _jacobian(terms, candidate, count, bound="upper"), ones, count
            )
            return candidate, spectral, condition, implicit
        width *= 2.0
    return None, spectral, condition, False


def solve_polynomial(
    forest: DerivationForest,
    algebra: OrderPreservingSemiring,
    inputs: Mapping[InputId, torch.Tensor],
    *,
    like: torch.Tensor,
    budget: int | None = None,
) -> PolynomialSolution:
    """The least fixed point of a monotone polynomial system, by decomposed Newton.

    The derived hypergraph's strongly connected components are solved one
    after another, premises' components first (Etessami and Yannakakis
    2009, decomposed Newton's method; Esparza, Kiefer and Luttenberger
    2010 for its convergence). In the reals of the algebra's embedding,
    each item's value is the sum over its edges of the edge's weight times
    its premises' values: a polynomial with nonnegative coefficients in
    the component's items, whose coefficients carry the values of the
    components below. Each component's Newton iteration starts at zero
    and keeps a certified lower bound (`_newton`); a certified pre-fixed
    point bounds it from above (`_upper`). The bounds of a component's
    coefficients come from the bounds of the components below, so every
    interval holds the exact least fixed point. The value returned is the
    last iterate, inside its interval, and carries the implicit
    derivative ``(I - J)^-1 dF`` of the fixed point in the inputs.

    Parameters
    ----------
    forest : DerivationForest
        The derived hypergraph, every instance and input an edge.
    algebra : OrderPreservingSemiring
        The algebra: a `RealEmbedding` whose product commutes.
    inputs : Mapping of InputId to torch.Tensor
        The weights.
    like : torch.Tensor
        The dtype and device of units and zeros.
    budget : int or None
        The most Newton steps over all components; ``None`` for no limit.

    Returns
    -------
    PolynomialSolution
        The values, their intervals, and the work.

    Raises
    ------
    CyclicRefused
        If the algebra has no real embedding or its product is not
        established to commute, or a component's least fixed point is
        certified infinite.
    """
    if not isinstance(algebra, RealEmbedding):
        raise CyclicRefused(
            CyclicRefusal.NONLINEAR,
            f"{algebra.subject.name} has no embedding in the nonnegative reals",
        )
    if not algebra.establishes(TIMES_COMMUTATIVE):
        raise CyclicRefused(
            CyclicRefusal.SIDES,
            "a polynomial's monomials multiply premises in any order, and the "
            "product is not established to commute",
        )
    rounding = algebra.embedding_rounding
    count = _Count()
    reals: dict[Item, torch.Tensor] = {}
    lower: dict[Item, float] = {}
    upper: dict[Item, float] = {}
    solved: list[NewtonComponent] = []
    stopped: Literal["completed", "budget"] = "completed"
    for members in components(forest):
        position = {item: index for index, item in enumerate(members)}
        terms: list[list[_Term]] = []
        for item in members:
            monomials: list[_Term] = []
            for edge in forest.edges.get(item, ()):
                weight = algebra.to_reals(
                    edge_weight(forest, edge, algebra, inputs, like=like)
                )
                low = high = float(weight.detach())
                for _ in range(rounding):
                    low, high = _down(low), _up(high)
                coefficient = weight
                inside: list[int] = []
                for premise in edge.premises:
                    if premise in position:
                        inside.append(position[premise])
                        continue
                    coefficient = coefficient * reals[premise]
                    low = _down(low * lower[premise])
                    high = _up(high * upper[premise])
                monomials.append(_Term(coefficient, low, high, tuple(inside)))
            terms.append(monomials)
        remaining = None if budget is None else max(budget - count.iterations, 0)
        try:
            floor, estimate, steps, exhausted = _newton(terms, count, remaining)
        except Divergent as error:
            raise CyclicRefused(CyclicRefusal.DIVERGENT, error.reason) from error
        if exhausted:
            stopped = "budget"
        ceiling, spectral, condition, implicit = _upper(terms, estimate, floor, count)
        top = ceiling if ceiling is not None else [math.inf] * len(members)
        point = [
            min(max(value, low), high)
            for value, low, high in zip(estimate, floor, top, strict=True)
        ]
        values = _implicit(terms, point)
        for index, item in enumerate(members):
            reals[item] = values[index]
            lower[item] = floor[index]
            upper[item] = top[index]
        solved.append(
            NewtonComponent(
                items=tuple(members),
                iterations=steps,
                spectral=spectral,
                condition=condition,
                certified=ceiling is not None,
                implicit=implicit,
            )
        )
    found: dict[Item, torch.Tensor] = {}
    for item, value in reals.items():
        absent = value.detach() == 0
        safe = torch.where(absent, torch.ones_like(value), value)
        found[item] = torch.where(
            absent, algebra.zero_like((), like), algebra.from_reals(safe).to(like.dtype)
        )
    return PolynomialSolution(
        values=found,
        lower=lower,
        upper=upper,
        components=tuple(solved),
        iterations=count.iterations,
        sums=count.sums,
        products=count.products,
        exact=count.exact,
        stopped=stopped,
    )


def _implicit(
    terms: Sequence[Sequence[_Term]], point: Sequence[float]
) -> list[torch.Tensor]:
    """The fixed point's values, carrying the implicit derivative.

    At the fixed point ``x = F(x, w)``, ``dx = (I - J)^-1 dF``. The value
    ``x + (I - J)^-1 (F - F_detached)`` is ``x`` and has that derivative.

    Parameters
    ----------
    terms : Sequence of Sequence of _Term
        The polynomials, with coefficients on the autograd graph.
    point : Sequence of float
        The fixed point.

    Returns
    -------
    list of torch.Tensor
        One real value per member; without a derivative when no input is
        differentiated or ``I - J`` has no inverse at the point, where the
        implicit derivative does not exist.
    """
    size = len(point)
    fixed = torch.tensor(list(point), dtype=torch.float64)
    images: list[torch.Tensor] = []
    for monomials in terms:
        total = torch.zeros((), dtype=torch.float64)
        for term in monomials:
            factor = term.coefficient.to(torch.float64)
            for position in term.inside:
                factor = factor * fixed[position]
            total = total + factor
        images.append(total)
    image = torch.stack(images) if images else fixed
    if not image.requires_grad:
        return [fixed[index] for index in range(size)]
    jacobian = torch.zeros((size, size), dtype=torch.float64)
    for row, monomials in enumerate(terms):
        for term in monomials:
            for place, column in enumerate(term.inside):
                value = term.point
                for other, factor in enumerate(term.inside):
                    if other != place:
                        value *= point[factor]
                jacobian[row, column] += value
    try:
        inverse = torch.linalg.inv(torch.eye(size, dtype=torch.float64) - jacobian)
    except RuntimeError:
        return [fixed[index] for index in range(size)]
    if not bool(torch.isfinite(inverse).all()):
        return [fixed[index] for index in range(size)]
    values = fixed + inverse @ (image - image.detach())
    return [values[index] for index in range(size)]
