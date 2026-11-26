"""Joins of rule bodies: worst-case optimal for full output, elimination for aggregates.

A rule body is a conjunctive query. Each premise is an atom whose
attributes are the variables of its pattern; an item of the premise's
form that matches the pattern is a tuple of the atom's relation, so a
nested constructor pattern is flattened into the variables of its slots.
The side condition is a filter on the variables it reads.

`generic_join` lists every instance of the body, one variable at a time:
the values of a variable are the intersection, over the atoms that
contain it, of the values their tuples allow given the variables already
bound, found by iterating the smallest and probing the others. Its work
is at most the number of atoms and variables times the AGM bound of the
query (Ngo, Porat, Re and Rudra 2018, with the Generic Join of Ngo, Re
and Rudra 2014), and `agm_bound` computes that bound for a fractional
edge cover. The bound is tight in the worst case over inputs of the same
sizes (Atserias, Grohe and Marx 2013), so no algorithm listing the
output is better in data complexity; it is not instance optimal.

`eliminate` computes the sum, grouped by the conclusion's variables, of
the products of the premise weights, without listing the body's
instances: each variable that is summed out is removed by joining only
the factors that contain it and summing it away (InsideOut, Abo Khamis,
Ngo and Rudra 2016). Reordering products and sums this way needs a
commutative product that distributes over a commutative sum, so it is
refused for weights that keep their order. The order of elimination is
chosen by a greedy rule and recorded with its widths; it is not claimed
to be optimal, which is NP-hard to find.

References
----------
Ngo, H. Q., Porat, E., Re, C., and Rudra, A. (2018). Worst-case optimal
join algorithms. *Journal of the ACM* 65(3), 1-40.
[doi:10.1145/3180143](https://doi.org/10.1145/3180143)

Ngo, H. Q., Re, C., and Rudra, A. (2014). Skew strikes back: new
developments in the theory of join algorithms. *SIGMOD Record* 42(4), 5-16.
[doi:10.1145/2590989.2590991](https://doi.org/10.1145/2590989.2590991)

Atserias, A., Grohe, M., and Marx, D. (2013). Size bounds and query plans
for relational joins. *SIAM Journal on Computing* 42(4), 1737-1767.
[doi:10.1137/110859440](https://doi.org/10.1137/110859440)

Abo Khamis, M., Ngo, H. Q., and Rudra, A. (2016). FAQ: questions asked
frequently. *PODS*, 13-28.
[doi:10.1145/2902251.2902280](https://doi.org/10.1145/2902251.2902280)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from enum import Enum
from fractions import Fraction
from typing import Literal

import didactic.api as dx
import numpy
import torch
from scipy.optimize import linprog

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.core.algebras import OrderPreservingSemiring
from quivers.core.laws import (
    DISTRIBUTIVE,
    PLUS_ASSOCIATIVE,
    PLUS_COMMUTATIVE,
    TIMES_ASSOCIATIVE,
    TIMES_COMMUTATIVE,
    TIMES_UNIT,
    Citation,
    Law,
)
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import (
    ALGEBRA_OPERATIONS,
    ComplexityBound,
    ComplexityClaim,
    ComplexityParameter,
    Conditional,
    Phase,
    PhaseInapplicable,
    ProvedOptimal,
    Rational,
    WorkField,
    WorkReport,
    measuring,
)
from quivers.qiec.identifiers import FamilyId
from quivers.qiec.rules import (
    Binding,
    Ground,
    Item,
    Rule,
    RulePresentation,
    evaluate_index,
    match_index,
)
from quivers.qiec.types import IndexTerm
from quivers.stochastic.tries import Trie, TrieCounter, insert, intersect
from quivers.stochastic.indexed import (
    flat_pattern,
    index_variables,
    side_condition_holds,
)

__all__ = [
    "AGMBound",
    "Atom",
    "BodyQuery",
    "ELIMINATION_LAWS",
    "ELIMINATION_WORK",
    "ENUMERATION_LAWS",
    "Elimination",
    "EliminationStep",
    "GENERIC_JOIN_WORK",
    "JoinCounter",
    "JoinRefusal",
    "JoinRefused",
    "JoinWork",
    "Relation",
    "agm_bound",
    "body_query",
    "eliminate",
    "elimination_order",
    "enumerate_aggregate",
    "generic_join",
    "relations",
]


class JoinRefusal(str, Enum):
    """Why a join is refused."""

    NOT_FLAT = "not-flat"
    LAWS = "laws"
    ORDER = "order"


class JoinRefused(ValueError):
    """A join is refused, with the reason.

    Parameters
    ----------
    code : JoinRefusal
        The kind of reason.
    reason : str
        The reason.
    """

    def __init__(self, code: JoinRefusal, reason: str) -> None:
        super().__init__(f"{code.value}: {reason}")
        self.code = code
        self.reason = reason


# ---------------------------------------------------------------------------
# rule bodies as queries


class Atom(ValueModel):
    """One premise as a relation over its pattern's variables.

    Parameters
    ----------
    premise : int
        The premise's position.
    family : FamilyId
        Its form.
    arguments : tuple of IndexTerm
        Its pattern.
    variables : tuple of str
        The pattern's variables, in the query's order.
    """

    premise: int
    family: FamilyId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    arguments: tuple[IndexTerm, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    variables: tuple[str, ...] = ()


class BodyQuery(ValueModel):
    """A rule body as a conjunctive query.

    Parameters
    ----------
    rule : Rule
        The rule.
    atoms : tuple of Atom
        One per premise.
    variables : tuple of str
        Every variable of the premises, in order of first occurrence.
    free : tuple of str
        The variables of the conclusion, which an aggregate groups by.
    filtered : tuple of str
        The variables the side condition reads.
    """

    rule: Rule = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    atoms: tuple[Atom, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    variables: tuple[str, ...] = ()
    free: tuple[str, ...] = ()
    filtered: tuple[str, ...] = ()


def body_query(rule: Rule) -> BodyQuery:
    """The conjunctive query of a rule body.

    Parameters
    ----------
    rule : Rule
        The rule.

    Returns
    -------
    BodyQuery
        Its query.

    Raises
    ------
    JoinRefused
        If a premise has a sum of two variables, whose matches are not
        a relation of the premise's items alone; or a premise has
        hypotheses.
    """
    order: list[str] = []
    atoms: list[Atom] = []
    for position, premise in enumerate(rule.premises):
        if premise.hypotheses:
            raise JoinRefused(
                JoinRefusal.NOT_FLAT,
                f"premise {position} of {rule.name} has hypotheses",
            )
        names: list[str] = []
        for argument in premise.judgment.arguments:
            if not flat_pattern(argument):
                raise JoinRefused(
                    JoinRefusal.NOT_FLAT,
                    f"premise {position} of {rule.name} has a sum of two variables",
                )
            for name in sorted(index_variables(argument)):
                if name not in names:
                    names.append(name)
                if name not in order:
                    order.append(name)
        atoms.append(
            Atom(
                premise=position,
                family=premise.judgment.family,
                arguments=premise.judgment.arguments,
                variables=tuple(names),
            )
        )
    free: list[str] = []
    for argument in rule.conclusion.arguments:
        for name in sorted(index_variables(argument)):
            if name not in free:
                free.append(name)
    condition = rule.side_condition
    return BodyQuery(
        rule=rule,
        atoms=tuple(atoms),
        variables=tuple(order),
        free=tuple(free),
        filtered=() if condition is None else tuple(condition.arguments),
    )


class Relation(ValueModel):
    """The tuples of one atom: each item matching it, by its variables' values.

    Matching determines an item from its tuple, since the item's
    arguments are the pattern evaluated at the tuple.

    Parameters
    ----------
    atom : Atom
        The atom.
    tuples : dict of tuple of Ground to Item
        Each tuple, over the atom's variables in order, and its item.
    """

    atom: Atom = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tuples: dict[tuple[Ground, ...], Item] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )


def relations(query: BodyQuery, items: Iterable[Item]) -> tuple[Relation, ...]:
    """The relation of every atom over a set of items.

    Parameters
    ----------
    query : BodyQuery
        The query.
    items : Iterable of Item
        The items; repeats count once.

    Returns
    -------
    tuple of Relation
        One per atom.
    """
    by_family: dict[FamilyId, list[Item]] = {}
    for item in dict.fromkeys(items):
        by_family.setdefault(item.family, []).append(item)
    found: list[Relation] = []
    for atom in query.atoms:
        tuples: dict[tuple[Ground, ...], Item] = {}
        for item in by_family.get(atom.family, []):
            binding: Binding = {}
            if all(
                match_index(pattern, value, binding)
                for pattern, value in zip(atom.arguments, item.arguments, strict=True)
            ):
                tuples[tuple(binding[name] for name in atom.variables)] = item
        found.append(Relation(atom=atom, tuples=tuples))
    return tuple(found)


# ---------------------------------------------------------------------------
# the AGM bound


class AGMBound(QiecModel):
    """A fractional edge cover and the size bound it gives.

    For relations of sizes ``N_e`` and weights ``x_e >= 0`` that cover
    every variable (the weights of the atoms containing it sum to one or
    more), the join has at most ``prod_e N_e^{x_e}`` tuples (Atserias,
    Grohe and Marx 2013). The cover is exact rationals checked to cover;
    it comes from a linear program minimizing the bound, and the bound
    holds whether or not it is the minimum.

    Parameters
    ----------
    variables : tuple of str
        The variables covered.
    sizes : tuple of int
        The relation sizes.
    cover : tuple of Rational
        The weight of each relation.
    """

    variables: tuple[str, ...] = ()
    sizes: tuple[int, ...] = ()
    cover: tuple[Rational, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @property
    def value(self) -> float:
        """The bound ``prod_e N_e^{x_e}``, zero when a covering relation is empty.

        Returns
        -------
        float
            The bound.
        """
        total = 0.0
        for size, weight in zip(self.sizes, self.cover, strict=True):
            if weight.numerator == 0:
                continue
            if size == 0:
                return 0.0
            total += float(weight.fraction) * math.log(size)
        return math.exp(total)

    @property
    def rho(self) -> Fraction:
        """The cover's total weight, the bound's exponent when every size is ``N``.

        Returns
        -------
        Fraction
            ``sum_e x_e``.
        """
        return sum((weight.fraction for weight in self.cover), Fraction(0))


def agm_bound(
    variables: Sequence[str],
    edges: Sequence[Sequence[str]],
    sizes: Sequence[int],
) -> AGMBound:
    """A fractional edge cover of variables by relations, and its bound.

    Parameters
    ----------
    variables : Sequence of str
        The variables to cover.
    edges : Sequence of Sequence of str
        The variables of each relation.
    sizes : Sequence of int
        The size of each relation.

    Returns
    -------
    AGMBound
        A cover; with every size at least two, of least bound up to the
        rounding of the program's solution, which is then repaired to
        cover exactly.

    Raises
    ------
    JoinRefused
        If a variable is in no relation.
    """
    names = list(variables)
    for name in names:
        if not any(name in edge for edge in edges):
            raise JoinRefused(JoinRefusal.NOT_FLAT, f"{name!r} is in no relation")
    if not names:
        return AGMBound(
            variables=(),
            sizes=tuple(sizes),
            cover=tuple(Rational.of(0) for _ in sizes),
        )
    costs = numpy.array([math.log(max(size, 2)) for size in sizes])
    matrix = numpy.array(
        [[-1.0 if name in edge else 0.0 for edge in edges] for name in names]
    )
    solution = linprog(
        costs,
        A_ub=matrix,
        b_ub=-numpy.ones(len(names)),
        bounds=[(0, None)] * len(edges),
        method="highs",
    )
    weights = [
        Fraction(max(float(value), 0.0)).limit_denominator(1024) for value in solution.x
    ]
    for name in names:
        covered = sum(
            (
                weight
                for weight, edge in zip(weights, edges, strict=True)
                if name in edge
            ),
            Fraction(0),
        )
        if covered < 1:
            index = min(
                (i for i, edge in enumerate(edges) if name in edge),
                key=lambda i: sizes[i],
            )
            weights[index] += 1 - covered
    return AGMBound(
        variables=tuple(names),
        sizes=tuple(sizes),
        cover=tuple(Rational.of(weight) for weight in weights),
    )


# ---------------------------------------------------------------------------
# generic join


class JoinWork(ValueModel):
    """Counters of a join.

    Parameters
    ----------
    iterated : int
        Values iterated from the smallest set of each intersection.
    probes : int
        Membership tests in the other sets.
    tuples : int
        Partial and complete bindings extended.
    outputs : int
        Complete bindings produced.
    conditions : int
        Side conditions evaluated.
    products : int
        Products of the algebra computed.
    sums : int
        Sums of the algebra computed.
    """

    iterated: int = 0
    probes: int = 0
    tuples: int = 0
    outputs: int = 0
    conditions: int = 0
    products: int = 0
    sums: int = 0

    def report(self, phase: Phase) -> WorkReport:
        """The counters as a work report.

        Parameters
        ----------
        phase : Phase
            The phase they counted.

        Returns
        -------
        WorkReport
            Values iterated and probes as key comparisons, the algebra's
            sums and products, each measured; a join divides nothing.
        """
        return WorkReport(
            phase=phase,
            key_comparisons=self.iterated + self.probes,
            scalar_sums=self.sums,
            scalar_products=self.products,
            measured=measuring(
                {
                    WorkField.KEY_COMPARISONS,
                    WorkField.SCALAR_SUMS,
                    WorkField.SCALAR_PRODUCTS,
                    WorkField.SCALAR_DIVISIONS,
                }
            ),
        )


class JoinCounter(TrieCounter):
    """Mutable counters of a join, read out as a `JoinWork`."""

    def __init__(self) -> None:
        super().__init__()
        self.outputs = 0
        self.conditions = 0
        self.products = 0
        self.sums = 0

    def report(self) -> JoinWork:
        """The counts so far.

        Returns
        -------
        JoinWork
            The counters.
        """
        return JoinWork(
            iterated=self.iterated,
            probes=self.probes,
            tuples=self.tuples,
            outputs=self.outputs,
            conditions=self.conditions,
            products=self.products,
            sums=self.sums,
        )


def _trie(
    relation: Relation, position: Mapping[str, int]
) -> tuple[tuple[str, ...], Trie[Ground, Item] | Item]:
    """The trie of a relation, keyed in the global order of its variables.

    Parameters
    ----------
    relation : Relation
        The relation.
    position : Mapping of str to int
        The place of each variable in the global order.

    Returns
    -------
    tuple of tuple of str and Trie
        The relation's variables in global order, and the trie; a
        relation of no variables is its one item, or an empty dict.
    """
    atom = relation.atom
    ordered = tuple(sorted(atom.variables, key=lambda name: position[name]))
    permutation = [atom.variables.index(name) for name in ordered]
    if not ordered:
        return ordered, next(iter(relation.tuples.values()), {})
    root: Trie[Ground, Item] = {}
    for values, item in relation.tuples.items():
        insert(root, [values[index] for index in permutation], item)
    return ordered, root


def generic_join(
    query: BodyQuery,
    tables: Sequence[Relation],
    presentation: RulePresentation,
    *,
    order: Sequence[str] | None = None,
    work: JoinCounter | None = None,
) -> Iterator[tuple[Binding, tuple[Item, ...]]]:
    """Every instance of a rule body over the relations, by Generic Join.

    Variables are bound one at a time in ``order``; each takes the
    values in the intersection, over the atoms containing it, of the
    values their tries allow below the variables already bound. The side
    condition filters each complete binding.

    Parameters
    ----------
    query : BodyQuery
        The query.
    tables : Sequence of Relation
        One relation per atom, in atom order.
    presentation : RulePresentation
        The presentation, for the side condition.
    order : Sequence of str or None
        The order of the variables; the query's by default.
    work : JoinCounter or None
        Counters to add to.

    Yields
    ------
    tuple of Binding and tuple of Item
        Each instance's binding and its premise items in premise order.

    Raises
    ------
    JoinRefused
        If ``order`` is not an order of the query's variables.
    """
    chosen = tuple(query.variables if order is None else order)
    if sorted(chosen) != sorted(query.variables):
        raise JoinRefused(JoinRefusal.ORDER, f"{chosen} orders other variables")
    counters = JoinCounter() if work is None else work
    position = {name: index for index, name in enumerate(chosen)}
    tries = [_trie(table, position) for table in tables]
    if any(not variables and isinstance(node, dict) for variables, node in tries):
        return
    containing = [
        [index for index, (variables, _) in enumerate(tries) if name in variables]
        for name in chosen
    ]
    yield from _descend(
        query, presentation, chosen, containing, [node for _, node in tries], counters
    )


def _descend(
    query: BodyQuery,
    presentation: RulePresentation,
    order: tuple[str, ...],
    containing: list[list[int]],
    nodes: list[Trie[Ground, Item] | Item],
    work: JoinCounter,
) -> Iterator[tuple[Binding, tuple[Item, ...]]]:
    """The instances of the body, by the intersection step, filtered by the side condition.

    Parameters
    ----------
    query : BodyQuery
        The query.
    presentation : RulePresentation
        The presentation, for the side condition.
    order : tuple of str
        The variable order.
    containing : list of list of int
        The atoms containing each variable.
    nodes : list
        Each atom's trie.
    work : JoinCounter
        The counters.

    Yields
    ------
    tuple of Binding and tuple of Item
        Complete instances.
    """
    for binding, leaves in intersect(order, containing, nodes, {}, work):
        if query.rule.side_condition is not None:
            work.conditions += 1
            if not side_condition_holds(presentation, query.rule, binding):
                continue
        premises: list[Item] = []
        for leaf in leaves:
            assert isinstance(leaf, Item)
            premises.append(leaf)
        work.outputs += 1
        yield binding, tuple(premises)


# ---------------------------------------------------------------------------
# aggregates


ELIMINATION_LAWS: tuple[Law, ...] = (
    PLUS_ASSOCIATIVE,
    PLUS_COMMUTATIVE,
    TIMES_ASSOCIATIVE,
    TIMES_COMMUTATIVE,
    TIMES_UNIT,
    DISTRIBUTIVE,
)
"""The laws elimination reorders by: a commutative semiring on one domain."""


class EliminationStep(QiecModel):
    """One variable summed out.

    Parameters
    ----------
    variable : str
        The variable.
    bag : tuple of str
        The variables of the factors joined to remove it.
    cover : AGMBound
        A fractional cover of the bag by the body's atoms.
    """

    variable: str
    bag: tuple[str, ...] = ()
    cover: AGMBound = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class Elimination(ValueModel):
    """The aggregate of a rule body, and how it was computed.

    Parameters
    ----------
    values : dict of Item to torch.Tensor
        The sum, for each conclusion, of the rule weight times the
        product of the premise weights over the body's instances.
    algorithm : str
        ``"variable elimination"`` or ``"enumeration in premise order"``.
    order : tuple of str
        The variables summed out, in order.
    steps : tuple of EliminationStep
        Each step, with its bag and cover.
    width : int
        The largest bag less one, the induced width of the order.
    fractional_width : Rational
        The largest cover weight of a bag.
    work : JoinCounter
        The counters.
    """

    values: dict[Item, torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    algorithm: Literal["variable elimination", "enumeration in premise order"]
    order: tuple[str, ...] = ()
    steps: tuple[EliminationStep, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    width: int = 0
    fractional_width: Rational = dx.field(
        default=Rational(numerator=0), opaque=True, extras=OPAQUE_BY_VALUE
    )
    work: JoinWork = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


def _require_laws(algebra: OrderPreservingSemiring, laws: Sequence[Law]) -> None:
    """Refuse an algebra that does not establish the laws.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    laws : Sequence of Law
        The laws needed.

    Raises
    ------
    JoinRefused
        If a law has no proved or decided claim.
    """
    established = {claim.law.name for claim in algebra.claims() if claim.established}
    missing = [law.name for law in laws if law.name not in established]
    if missing:
        raise JoinRefused(
            JoinRefusal.LAWS,
            f"{algebra.subject.name} does not establish {missing}",
        )


def elimination_order(query: BodyQuery) -> tuple[str, ...]:
    """A greedy order of the variables to sum out: fewest neighbours first.

    At each step the variable whose bag (the variables sharing an atom
    or the side condition with it, among those left) is smallest is
    removed, ties in the query's order. Finding an order of least width
    is NP-hard, so this order's widths are recorded, not claimed least.

    Parameters
    ----------
    query : BodyQuery
        The query.

    Returns
    -------
    tuple of str
        The variables not in the conclusion, in elimination order.
    """
    edges = [set(atom.variables) for atom in query.atoms]
    if query.filtered:
        edges.append(set(query.filtered))
    left = [name for name in query.variables if name not in query.free]
    chosen: list[str] = []
    while left:

        def bag(name: str) -> int:
            """The size of a variable's bag among the remaining edges.

            Parameters
            ----------
            name : str
                The variable.

            Returns
            -------
            int
                The bag's size.
            """
            return len(set().union(*(edge for edge in edges if name in edge)))

        best = min(left, key=lambda name: (bag(name), query.variables.index(name)))
        joined = set().union(*(edge for edge in edges if best in edge))
        edges = [edge for edge in edges if best not in edge] + [joined - {best}]
        left.remove(best)
        chosen.append(best)
    return tuple(chosen)


class _Factor(ValueModel):
    """A table of weights over some variables.

    Parameters
    ----------
    variables : tuple of str
        Its variables.
    table : dict of tuple of Ground to torch.Tensor
        A weight per tuple of their values.
    """

    variables: tuple[str, ...] = ()
    table: dict[tuple[Ground, ...], torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )


def _join_factors(
    factors: Sequence[_Factor],
    variables: tuple[str, ...],
    algebra: OrderPreservingSemiring,
    work: JoinCounter,
) -> Iterator[tuple[Binding, torch.Tensor]]:
    """Every binding of ``variables`` in the support of all factors, with the product.

    The supports are joined by Generic Join in the order ``variables``.

    Parameters
    ----------
    factors : Sequence of _Factor
        The factors; together they contain every variable.
    variables : tuple of str
        The variables, in join order.
    algebra : OrderPreservingSemiring
        The algebra.
    work : JoinCounter
        The counters.

    Yields
    ------
    tuple of Binding and torch.Tensor
        A binding and the product of the factors' weights at it.
    """
    position = {name: index for index, name in enumerate(variables)}
    built: list[tuple[tuple[str, ...], Trie[Ground, torch.Tensor] | torch.Tensor]] = []
    for factor in factors:
        ordered = tuple(sorted(factor.variables, key=lambda name: position[name]))
        permutation = [factor.variables.index(name) for name in ordered]
        if not ordered:
            if () not in factor.table:
                return
            built.append(((), factor.table[()]))
            continue
        root: Trie[Ground, torch.Tensor] = {}
        for values, weight in factor.table.items():
            insert(root, [values[index] for index in permutation], weight)
        built.append((ordered, root))
    containing = [
        [index for index, (names, _) in enumerate(built) if name in names]
        for name in variables
    ]
    for binding, leaves in intersect(
        variables, containing, [node for _, node in built], {}, work
    ):
        weights: list[torch.Tensor] = []
        for leaf in leaves:
            assert isinstance(leaf, torch.Tensor)
            weights.append(leaf)
        total = weights[0]
        for weight in weights[1:]:
            work.products += 1
            total = algebra.tensor_op(total, weight)
        yield binding, total


def _unit(algebra: OrderPreservingSemiring, like: torch.Tensor) -> torch.Tensor:
    """The algebra's unit.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    like : torch.Tensor
        The dtype and device.

    Returns
    -------
    torch.Tensor
        The unit.
    """
    return algebra.unit_like((), like)


def _add(
    algebra: OrderPreservingSemiring,
    table: dict[tuple[Ground, ...], torch.Tensor],
    key: tuple[Ground, ...],
    term: torch.Tensor,
    work: JoinCounter,
) -> None:
    """Add a term to one entry of a table.

    Parameters
    ----------
    algebra : OrderPreservingSemiring
        The algebra.
    table : dict of tuple of Ground to torch.Tensor
        The table.
    key : tuple of Ground
        The entry.
    term : torch.Tensor
        The term.
    work : JoinCounter
        The counters.
    """
    current = table.get(key)
    if current is None:
        table[key] = term
        return
    work.sums += 1
    table[key] = algebra.join(torch.stack(torch.broadcast_tensors(current, term)), 0)


def _conclusions(
    query: BodyQuery,
    table: Mapping[tuple[Ground, ...], torch.Tensor],
) -> dict[Item, torch.Tensor]:
    """The conclusion items of a table over the free variables.

    Parameters
    ----------
    query : BodyQuery
        The query.
    table : Mapping of tuple of Ground to torch.Tensor
        Weights by the values of the free variables, in their order.

    Returns
    -------
    dict of Item to torch.Tensor
        Weights by conclusion.
    """
    conclusion = query.rule.conclusion
    found: dict[Item, torch.Tensor] = {}
    for values, weight in table.items():
        binding = dict(zip(query.free, values, strict=True))
        arguments = tuple(evaluate_index(a, binding) for a in conclusion.arguments)
        found[Item(family=conclusion.family, arguments=arguments)] = weight
    return found


def eliminate(
    query: BodyQuery,
    tables: Sequence[Relation],
    presentation: RulePresentation,
    algebra: OrderPreservingSemiring,
    weights: Mapping[Item, torch.Tensor],
    *,
    like: torch.Tensor,
    rule_weight: torch.Tensor | None = None,
    order: Sequence[str] | None = None,
) -> Elimination:
    """The aggregate of a rule body by eliminating its summed variables.

    Each step joins the factors containing the variable (and, while the
    side condition is pending, the factors covering its variables, so
    that it filters the bag), sums the variable out, and replaces those
    factors by the result. What is left, over the conclusion's
    variables, is joined and multiplied by the rule weight.

    Parameters
    ----------
    query : BodyQuery
        The query.
    tables : Sequence of Relation
        One relation per atom.
    presentation : RulePresentation
        The presentation, for the side condition.
    algebra : OrderPreservingSemiring
        The algebra.
    weights : Mapping of Item to torch.Tensor
        The weight of each premise item; a missing one is the unit.
    like : torch.Tensor
        The dtype and device of the unit.
    rule_weight : torch.Tensor or None
        The rule's weight; the unit by default.
    order : Sequence of str or None
        The variables to sum out, in order; `elimination_order` by
        default.

    Returns
    -------
    Elimination
        The weight of each conclusion, the order, and its widths.

    Raises
    ------
    JoinRefused
        If the algebra does not establish `ELIMINATION_LAWS`, or
        ``order`` is not an order of the summed variables.
    """
    _require_laws(algebra, ELIMINATION_LAWS)
    summed = [name for name in query.variables if name not in query.free]
    chosen = elimination_order(query) if order is None else tuple(order)
    if sorted(chosen) != sorted(summed):
        raise JoinRefused(JoinRefusal.ORDER, f"{chosen} does not order {summed}")
    work = JoinCounter()
    factors = [
        _Factor(
            variables=table.atom.variables,
            table={
                values: weights.get(item, _unit(algebra, like))
                for values, item in table.tuples.items()
            },
        )
        for table in tables
    ]
    sizes = [len(table.tuples) for table in tables]
    pending = query.rule.side_condition is not None
    steps: list[EliminationStep] = []
    for name in chosen:
        joined = [factor for factor in factors if name in factor.variables]
        filtering = pending and name in query.filtered
        needed = set().union(*(factor.variables for factor in joined))
        if filtering:
            needed |= set(query.filtered)
            for factor in factors:
                if factor not in joined and set(factor.variables) & (
                    needed - set().union(*(f.variables for f in joined))
                ):
                    joined.append(factor)
                    needed |= set(factor.variables)
        bag = tuple(variable for variable in query.variables if variable in needed)
        rest = tuple(variable for variable in bag if variable != name)
        table: dict[tuple[Ground, ...], torch.Tensor] = {}
        for binding, product in _join_factors(joined, bag, algebra, work):
            if filtering:
                work.conditions += 1
                if not side_condition_holds(presentation, query.rule, binding):
                    continue
            _add(algebra, table, tuple(binding[v] for v in rest), product, work)
        factors = [factor for factor in factors if factor not in joined]
        factors.append(_Factor(variables=rest, table=table))
        if filtering:
            pending = False
        edges = [tuple(v for v in atom.variables if v in bag) for atom in query.atoms]
        steps.append(
            EliminationStep(variable=name, bag=bag, cover=agm_bound(bag, edges, sizes))
        )
    output: dict[tuple[Ground, ...], torch.Tensor] = {}
    weight = _unit(algebra, like) if rule_weight is None else rule_weight
    for binding, product in _join_factors(factors, query.free, algebra, work):
        if pending:
            work.conditions += 1
            if not side_condition_holds(presentation, query.rule, binding):
                continue
        work.products += 1
        work.outputs += 1
        _add(
            algebra,
            output,
            tuple(binding[v] for v in query.free),
            algebra.tensor_op(weight, product),
            work,
        )
    return Elimination(
        values=_conclusions(query, output),
        algorithm="variable elimination",
        order=chosen,
        steps=tuple(steps),
        width=max((len(step.bag) - 1 for step in steps), default=0),
        fractional_width=Rational.of(
            max((step.cover.rho for step in steps), default=Fraction(0))
        ),
        work=work.report(),
    )


ENUMERATION_LAWS: tuple[Law, ...] = (
    PLUS_ASSOCIATIVE,
    PLUS_COMMUTATIVE,
    TIMES_ASSOCIATIVE,
)
"""The laws a sum of products in premise order uses."""


def enumerate_aggregate(
    query: BodyQuery,
    tables: Sequence[Relation],
    presentation: RulePresentation,
    algebra: OrderPreservingSemiring,
    weights: Mapping[Item, torch.Tensor],
    *,
    like: torch.Tensor,
    rule_weight: torch.Tensor | None = None,
    order: Sequence[str] | None = None,
) -> Elimination:
    """The aggregate of a rule body by listing its instances.

    Each instance's product is taken in premise order, then the rule
    weight, as a derivation's factors are read, so the product need not
    commute; the cost is that of
    `generic_join`, the number of instances at worst.

    Parameters
    ----------
    query : BodyQuery
        The query.
    tables : Sequence of Relation
        One relation per atom.
    presentation : RulePresentation
        The presentation, for the side condition.
    algebra : OrderPreservingSemiring
        The algebra.
    weights : Mapping of Item to torch.Tensor
        The weight of each premise item; a missing one is the unit.
    like : torch.Tensor
        The dtype and device of the unit.
    rule_weight : torch.Tensor or None
        The rule's weight; the unit by default.
    order : Sequence of str or None
        The join's variable order.

    Returns
    -------
    Elimination
        The weight of each conclusion.

    Raises
    ------
    JoinRefused
        If the algebra does not establish `ENUMERATION_LAWS`.
    """
    _require_laws(algebra, ENUMERATION_LAWS)
    work = JoinCounter()
    weight = _unit(algebra, like) if rule_weight is None else rule_weight
    output: dict[tuple[Ground, ...], torch.Tensor] = {}
    for binding, premises in generic_join(
        query, tables, presentation, order=order, work=work
    ):
        total = weights.get(premises[0], _unit(algebra, like)) if premises else weight
        for item in premises[1:]:
            work.products += 1
            total = algebra.tensor_op(total, weights.get(item, _unit(algebra, like)))
        if premises:
            work.products += 1
            total = algebra.tensor_op(total, weight)
        _add(algebra, output, tuple(binding[v] for v in query.free), total, work)
    return Elimination(
        values=_conclusions(query, output),
        algorithm="enumeration in premise order",
        order=tuple(query.variables if order is None else order),
        work=work.report(),
    )


# ---------------------------------------------------------------------------
# cost claims


_AGM = Citation(
    reference="Atserias, Grohe and Marx 2013, Size Bounds and Query Plans for Relational Joins",
    locator="the bound is tight for every query and sizes",
    doi="10.1137/110859440",
)
_NPRR = Citation(
    reference="Ngo, Porat, Re and Rudra 2018, Worst-case Optimal Join Algorithms",
    locator="theorem 5.1; section 7 shows no instance optimality",
    doi="10.1145/3180143",
)
_GENERIC = Citation(
    reference="Ngo, Re and Rudra 2014, Skew strikes back",
    locator="Generic Join",
    doi="10.1145/2590989.2590991",
)
_FAQ = Citation(
    reference="Abo Khamis, Ngo and Rudra 2016, FAQ: Questions Asked Frequently",
    locator="InsideOut and the width of an order",
    doi="10.1145/2902251.2902280",
)
_MARX = Citation(
    reference="Marx 2010, Can you beat treewidth?",
    locator="lower bound under the exponential time hypothesis",
    doi="10.4086/toc.2010.v006a005",
)
_READ = Citation(
    reference="every input tuple is read, by an adversary argument",
    locator="complexity claims",
)

GENERIC_JOIN_WORK = ComplexityClaim(
    query="every instance of one rule body",
    parameters=(
        ComplexityParameter(name="N", meaning="tuples of the atoms' relations"),
        ComplexityParameter(
            name="AGM", meaning="the bound of a fractional edge cover of the body"
        ),
        ComplexityParameter(name="n", meaning="variables of the body"),
        ComplexityParameter(name="m", meaning="atoms of the body"),
    ),
    model=ALGEBRA_OPERATIONS,
    bounds=(
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="time",
            expression="N + m n AGM",
            status=ProvedOptimal(
                lower_bound="AGM in the worst case over relations of the same sizes",
                model=ALGEBRA_OPERATIONS,
                sources=(_AGM, _NPRR, _GENERIC),
            ),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="space",
            expression="N + n",
            status=ProvedOptimal(
                lower_bound="N", model=ALGEBRA_OPERATIONS, sources=(_READ,)
            ),
        ),
    ),
    inapplicable=(
        PhaseInapplicable(
            phase=Phase.CONSTRUCTION,
            reason="the query is read off the rule in its size",
        ),
        PhaseInapplicable(
            phase=Phase.DIFFERENTIATION,
            reason="instances are listed, not weighed",
        ),
        PhaseInapplicable(
            phase=Phase.REFIT,
            reason="instances depend on the items only, not on weights",
        ),
    ),
    conditions=(
        "data complexity: the rule is fixed and the relations grow",
        "a natural join over flat attributes; a nested constructor pattern is "
        "flattened by matching, and a sum of two variables is refused",
        "the side condition filters the output and is outside the bound",
        "expected constant-time hashing",
        "worst-case optimal, not instance optimal",
    ),
)
"""The cost of listing a rule body's instances by Generic Join."""

ELIMINATION_WORK = ComplexityClaim(
    query="sum of products of one rule body, grouped by its conclusion",
    parameters=(
        ComplexityParameter(name="N", meaning="tuples of the atoms' relations"),
        ComplexityParameter(
            name="W",
            meaning="the largest bound of a bag's cover over the chosen order",
        ),
        ComplexityParameter(name="F", meaning="conclusions"),
        ComplexityParameter(name="n", meaning="variables of the body"),
        ComplexityParameter(name="m", meaning="atoms of the body"),
    ),
    model=ALGEBRA_OPERATIONS,
    bounds=(
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="time",
            expression="N + n m W + F",
            status=Conditional(
                hypothesis=(
                    "the exponential time hypothesis: for classes of bounded "
                    "arity no algorithm beats treewidth by more than a "
                    "logarithmic factor in the exponent"
                ),
                sources=(_FAQ, _MARX),
            ),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="space",
            expression="N + W + F",
            status=Conditional(
                hypothesis="as for time",
                sources=(_FAQ,),
            ),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="time",
            expression="N + n m W + F",
            status=Conditional(hypothesis="as for execution", sources=(_FAQ,)),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="space",
            expression="N + n m W + F",
            status=Conditional(
                hypothesis="the autograd tape holds every product and sum",
                sources=(_FAQ,),
            ),
        ),
    ),
    inapplicable=(
        PhaseInapplicable(
            phase=Phase.CONSTRUCTION,
            reason="the order is chosen from the rule in its size",
        ),
        PhaseInapplicable(
            phase=Phase.REFIT,
            reason="a refit is one more elimination at the new weights",
        ),
    ),
    conditions=(
        "a commutative semiring on one domain sharing zero and one: the laws "
        "of ELIMINATION_LAWS, each proved or decided",
        "not for weights that keep their order, effects, or algebras that are "
        "not semirings; those are summed by enumeration in premise order",
        "the order is greedy and its widths are recorded; an order of least "
        "width is NP-hard to find and is not claimed",
        "the side condition joins the factors covering its variables into one bag",
    ),
)
"""The cost of aggregating a rule body by variable elimination."""
