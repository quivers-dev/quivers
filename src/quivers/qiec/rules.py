"""Rule presentations: typed judgments, rules, and their derivations.

A rule presentation (`RulePresentation`) states a deduction system as
QIEC declarations. Each judgment is an indexed family, and a judgment
instance is an application of the family to index arguments. A rule is
a constructor of its conclusion's family whose fields are its premise
judgments in order, followed by the values it carries; its telescope
holds its typed schematic variables. An axiom slot is a constructor with
no premises, and each input axiom is an occurrence of a slot with its own
identity. Proofs are thus an inductive family, and the QIEC checker that
checks every other declaration checks the presentation: `declarations`
lowers it, and construction runs `validate_module` on the result.

For the presentation with input occurrences ``A_j`` the derivations of a
judgment ``j`` are the initial algebra of the polynomial functor

    (F_A X)_j = A_j + sum over rules r : (j_1, ..., j_n) -> j of
                X_{j_1} x ... x X_{j_n},

and a fold of a derivation by an interpretation is the unique map out of
it. That is a description of the semantics; nothing here materializes
every derivation except `ReferenceInterpretation`, which is an oracle
for small systems and not an executor.

Items are interned by `ItemTable`, so an item is built and hashed once.
A presentation is validated once when it is built, and its indexes (rules
by conclusion family, rules by premise family and position, the family
dependency graph and its components) are built once on first use and
shared by every later use.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction
from functools import cached_property
from types import MappingProxyType
from typing import Literal
from weakref import WeakKeyDictionary

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.qiec._models import QiecModel
from quivers.core.laws import (
    Citation,
    ClaimSubject,
    Law,
    LawClaim,
    Tested,
)
from quivers.qiec.contracts import (
    WORD_RAM_EXPECTED_HASHING,
    Acyclic,
    BestKnown,
    ComplexityBound,
    ComplexityClaim,
    ComplexityParameter,
    Phase,
    ProvedOptimal,
    Unresolved,
    InterpretationId,
    LinearCyclic,
    MonotonePolynomial,
    OccurrenceId,
    ParameterId,
    PresentationId,
    Recursion,
)
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl, FieldDef
from quivers.qiec.effects import EMPTY_ROW
from quivers.qiec.evaluator import Evaluator
from quivers.qiec.identifiers import ComputationId, ConstructorId, FamilyId, TypeId
from quivers.qiec.kinds import (
    IndexBinder,
    IndexSort,
    NatSort,
    Telescope,
    UserIndexSort,
    validate_telescope,
)
from quivers.qiec.checking import CheckContext, KernelError, KernelRegistry, infer_value
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.terms import ConstructorValue, LiteralValue, Local, Value, Var
from quivers.structural.signature import Signature, Term
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    TypeConstructorRef,
    TypeExpr,
    IndexConstructor,
    IndexLiteral,
    IndexSum,
    IndexTerm,
    IndexVariable,
    ShapeIndex,
    TypeApplication,
)

type Ground = int | str | tuple[Ground, ...]
"""A ground index value: a natural number, a nullary constructor of a
closed sort by name, or a constructor application as ``(name, *arguments)``.
"""

type Binding = dict[str, Ground]
"""Values of the schematic variables of a rule."""


class RuleError(ValueError):
    """A rule presentation is ill-formed."""


# ---------------------------------------------------------------------------
# declarations


class JudgmentDecl(QiecModel):
    """A judgment form: a family indexed by its arguments.

    Parameters
    ----------
    id : FamilyId
        The family's identity.
    name : str
        Its name.
    indices : Telescope
        Its arguments, each an index binder.
    """

    id: FamilyId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    indices: Telescope = dx.field(default=(), opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require index binders that may be refined.

        Raises
        ------
        ValueError
            If the telescope is malformed, holds a binder that is not an
            index binder, or a binder is not refinable.
        """
        validate_telescope(self.indices)
        for binder in self.indices:
            if not isinstance(binder, IndexBinder):
                raise ValueError(f"judgment {self.name!r} takes index arguments only")
            if not binder.refinable:
                raise ValueError(
                    f"argument {binder.name!r} of judgment {self.name!r} must be "
                    "refinable: a rule's conclusion fixes it"
                )


class Judgment(QiecModel):
    """A judgment instance: a family applied to index arguments.

    Parameters
    ----------
    family : FamilyId
        The judgment form.
    arguments : tuple of IndexTerm
        One per index of the form; they may mention the schematic
        variables of the enclosing rule.
    """

    family: FamilyId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    arguments: tuple[IndexTerm, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


class Premise(QiecModel):
    """One premise of a rule, in its position.

    The premise is derived in the context ``before + context + after``.
    In an ordered context the two sides differ: in the Lambek calculus a
    hypothesis on the right introduces ``A / B`` and one on the left
    introduces ``B \\ A``.

    Parameters
    ----------
    judgment : Judgment
        What the premise has to derive.
    before : tuple of Judgment
        Hypotheses the premise may use that are inserted at the left of
        the rule's context, outermost first.
    after : tuple of Judgment
        Hypotheses inserted at its right, outermost first.
    """

    judgment: Judgment = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    before: tuple[Judgment, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    after: tuple[Judgment, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @property
    def hypotheses(self) -> tuple[Judgment, ...]:
        """Every hypothesis of the premise, left block then right block.

        Returns
        -------
        tuple of Judgment
            ``before + after``.
        """
        return (*self.before, *self.after)


class SideCondition(QiecModel):
    """A checked QIEC computation a rule instance has to satisfy.

    Parameters
    ----------
    computation : ComputationId
        A pure computation of the presentation's conditions with no
        telescope that returns ``Bool``.
    arguments : tuple of str
        The rule's schematic variables or value fields passed to it, one
        per parameter. A natural-number variable is passed as an
        ``Int``, a nullary constructor of a closed sort as a ``String``.
    """

    computation: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    arguments: tuple[str, ...] = ()


class ComputedRuleWeight(QiecModel):
    """A rule weight supplied by a named checked computation.

    The presentation records the requirement. Checked source lowers each
    application to a call on the rule telescope's bound fields. An executor
    without that lowering cannot give the rule a numerical weight.

    Parameters
    ----------
    operation : str
        The name of the computation in the owning source module.
    """

    operation: str


class Rule(QiecModel):
    """An inference rule: a constructor of its conclusion's proof family.

    The premise order is part of the rule: an interpretation receives
    the values of the premises in this order, and no executor may
    permute them unless the interpretation declares that it may.

    Parameters
    ----------
    id : ConstructorId
        The rule's identity, which is its constructor's.
    name : str
        Its name.
    telescope : Telescope
        Its typed schematic variables.
    premises : tuple of Premise
        Its premises, in order.
    conclusion : Judgment
        Its conclusion.
    values : tuple of FieldDef
        Values the rule carries beside its premises, such as the label of
        a leaf; their types are value types.
    side_condition : SideCondition or None
        A checked computation each instance has to satisfy.
    weight : ParameterId, ComputedRuleWeight, or None
        The parameter or checked computation that weights each application,
        or ``None`` for the unit weight.
    """

    id: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    telescope: Telescope = dx.field(default=(), opaque=True, extras=OPAQUE_BY_VALUE)
    premises: tuple[Premise, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    conclusion: Judgment = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    values: tuple[FieldDef, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    side_condition: SideCondition | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    weight: ParameterId | ComputedRuleWeight | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @property
    def arity(self) -> int:
        """The number of premises.

        Returns
        -------
        int
            The arity.
        """
        return len(self.premises)


class AxiomSlot(QiecModel):
    """A kind of input axiom: a rule with no premises filled at run time.

    Each input is an `AxiomOccurrence` of a slot with its own identity, so
    two equal inputs are two axioms.

    Parameters
    ----------
    id : ConstructorId
        The slot's identity, which is its constructor's.
    name : str
        Its name.
    telescope : Telescope
        Its schematic variables.
    judgment : Judgment
        The judgment an occurrence derives.
    values : tuple of FieldDef
        Values an occurrence carries.
    weight : ParameterId or None
        The parameter that weights every occurrence, or ``None`` when each
        occurrence brings its own weight.
    """

    id: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    telescope: Telescope = dx.field(default=(), opaque=True, extras=OPAQUE_BY_VALUE)
    judgment: Judgment = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    values: tuple[FieldDef, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    weight: ParameterId | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


# ---------------------------------------------------------------------------
# items


class Item(QiecModel):
    """A ground judgment instance: a family with ground arguments.

    An item is a conclusion, not a proof: one item may have many
    derivations. `ItemTable` interns items so that equal items are one
    object.

    Parameters
    ----------
    family : FamilyId
        The judgment form.
    arguments : tuple of Ground
        Its ground arguments.
    """

    family: FamilyId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    arguments: tuple[Ground, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


class ItemTable:
    """Interns items, so an item is built and hashed once.

    Interning costs the size of the arguments once per distinct item;
    after that an item compares and hashes in constant time by identity.
    """

    def __init__(self) -> None:
        self._items: dict[tuple[FamilyId, tuple[Ground, ...]], Item] = {}

    def intern(self, family: FamilyId, arguments: tuple[Ground, ...]) -> Item:
        """The one item of a family with these arguments.

        Parameters
        ----------
        family : FamilyId
            The judgment form.
        arguments : tuple of Ground
            The ground arguments.

        Returns
        -------
        Item
            The same object for equal family and arguments.
        """
        key = (family, arguments)
        item = self._items.get(key)
        if item is None:
            item = Item(family=family, arguments=arguments)
            self._items[key] = item
        return item

    def __len__(self) -> int:
        """The number of distinct items.

        Returns
        -------
        int
            The count.
        """
        return len(self._items)


class AxiomOccurrence(QiecModel):
    """One input axiom: an occurrence of a slot at ground arguments.

    Parameters
    ----------
    slot : ConstructorId
        The slot.
    occurrence : OccurrenceId
        The identity of this occurrence.
    binding : tuple of tuple of str and Ground
        Values of the slot's schematic variables, by name.
    values : tuple of str or int or float or bool
        Values of the slot's value fields, in order.
    """

    slot: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    occurrence: OccurrenceId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    binding: tuple[tuple[str, Ground], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    values: tuple[str | int | float | bool, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


class Hole(QiecModel):
    """An open hypothesis in a derivation: a variable of its context.

    Parameters
    ----------
    index : int
        Its de Bruijn index in the context where it occurs: 0 is the
        innermost hypothesis.
    item : Item
        The item it assumes.
    tag : str
        Always ``"hole"``.
    """

    index: int
    item: Item = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["hole"] = "hole"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a nonnegative index.

        Raises
        ------
        ValueError
            If it is negative.
        """
        if self.index < 0:
            raise ValueError("a hole's index is at least zero")


class Derivation(QiecModel):
    """A derivation tree: a proof of an item, not the item.

    Parameters
    ----------
    constructor : ConstructorId
        The rule or axiom slot at the root.
    item : Item
        The item it proves.
    binding : tuple of tuple of str and Ground
        Values of the rule's schematic variables.
    values : tuple of str or int or float or bool
        Values of the rule's value fields.
    occurrence : OccurrenceId or None
        The input occurrence, for an axiom.
    premises : tuple of Derivation or Hole
        Derivations of the premises, in order; a hole is an open
        hypothesis.
    """

    constructor: ConstructorId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    item: Item = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    binding: tuple[tuple[str, Ground], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    values: tuple[str | int | float | bool, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    occurrence: OccurrenceId | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    premises: tuple[Derivation | Hole, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def size(self) -> int:
        """The number of nodes, holes included.

        Returns
        -------
        int
            The nodes of the tree, counted without recursion.
        """
        count = 0
        stack: list[Derivation | Hole] = [self]
        while stack:
            node = stack.pop()
            count += 1
            if isinstance(node, Derivation):
                stack.extend(node.premises)
        return count


type ProofNode = Derivation | Hole
"""A derivation or an open hypothesis."""


# ---------------------------------------------------------------------------
# matching


class MatchError(RuleError):
    """A pattern cannot be matched or evaluated with the variables bound."""


def _ground_literal(term: IndexLiteral) -> Ground:
    """The ground value of a literal.

    Parameters
    ----------
    term : IndexLiteral
        The literal.

    Returns
    -------
    Ground
        Its value.
    """
    return term.value


def evaluate_index(term: IndexTerm, binding: Mapping[str, Ground]) -> Ground:
    """The ground value of an index term whose variables are bound.

    Parameters
    ----------
    term : IndexTerm
        The term.
    binding : Mapping[str, Ground]
        Values of its variables.

    Returns
    -------
    Ground
        Its value.

    Raises
    ------
    MatchError
        If a variable is unbound, a sum has a summand that is not a
        number, or the term is a shape.
    """
    if isinstance(term, IndexVariable):
        if term.name not in binding:
            raise MatchError(f"index variable {term.name!r} is unbound")
        return binding[term.name]
    if isinstance(term, IndexLiteral):
        return _ground_literal(term)
    if isinstance(term, IndexConstructor):
        return (term.name, *(evaluate_index(item, binding) for item in term.arguments))
    if isinstance(term, IndexSum):
        total = term.constant
        for summand in term.terms:
            value = evaluate_index(summand, binding)
            if not isinstance(value, int):
                raise MatchError(f"summand {summand.name!r} is not a number")
            total += value
        return total
    raise MatchError("a shape is not a judgment argument")


def match_index(term: IndexTerm, value: Ground, binding: Binding) -> bool:
    """Extend a binding so that an index term equals a ground value.

    The cost is the size of the term.

    Parameters
    ----------
    term : IndexTerm
        The pattern.
    value : Ground
        The ground value.
    binding : Binding
        The variables bound so far; extended in place on success.

    Returns
    -------
    bool
        Whether an extension exists. On failure the binding may hold
        variables bound by this call.

    Raises
    ------
    MatchError
        If a sum has more than one unbound variable, which the
        presentation's validation excludes.
    """
    if isinstance(term, IndexVariable):
        if term.name in binding:
            return binding[term.name] == value
        binding[term.name] = value
        return True
    if isinstance(term, IndexLiteral):
        return _ground_literal(term) == value
    if isinstance(term, IndexConstructor):
        if not isinstance(value, tuple) or len(value) != len(term.arguments) + 1:
            return False
        if value[0] != term.name:
            return False
        return all(
            match_index(item, part, binding)
            for item, part in zip(term.arguments, value[1:], strict=True)
        )
    if isinstance(term, IndexSum):
        if not isinstance(value, int):
            return False
        unbound = [item for item in term.terms if item.name not in binding]
        known = term.constant
        for item in term.terms:
            if item.name in binding:
                bound = binding[item.name]
                if not isinstance(bound, int):
                    return False
                known += bound
        if not unbound:
            return known == value
        if len(unbound) > 1:
            raise MatchError("an index sum with two unbound summands is not a pattern")
        rest = value - known
        if rest < 0:
            return False
        binding[unbound[0].name] = rest
        return True
    if isinstance(term, ShapeIndex):
        raise MatchError("a shape is not a judgment argument")
    raise MatchError("unknown index term")


def _variables(term: IndexTerm) -> Iterator[IndexVariable]:
    """The variables of an index term, with repeats.

    Parameters
    ----------
    term : IndexTerm
        The term.

    Yields
    ------
    IndexVariable
        Each variable occurrence.
    """
    if isinstance(term, IndexVariable):
        yield term
    elif isinstance(term, IndexConstructor):
        for item in term.arguments:
            yield from _variables(item)
    elif isinstance(term, IndexSum):
        yield from term.terms


def _constructors(term: IndexTerm) -> Iterator[IndexConstructor]:
    """The constructor applications in an index term.

    Parameters
    ----------
    term : IndexTerm
        The term.

    Yields
    ------
    IndexConstructor
        Each application, outermost first.
    """
    if isinstance(term, IndexConstructor):
        yield term
        for item in term.arguments:
            yield from _constructors(item)


def _bindable(term: IndexTerm, bound: set[str]) -> bool:
    """Whether matching a term binds its variables deterministically.

    Parameters
    ----------
    term : IndexTerm
        The pattern.
    bound : set of str
        The variables bound before it.

    Returns
    -------
    bool
        False when a sum, here or nested, has two summands unbound.
    """
    if isinstance(term, IndexSum):
        unbound = {item.name for item in term.terms if item.name not in bound}
        repeated = [item.name for item in term.terms if item.name in unbound]
        return len(unbound) <= 1 and len(repeated) == len(unbound)
    if isinstance(term, IndexConstructor):
        seen = set(bound)
        for item in term.arguments:
            if not _bindable(item, seen):
                return False
            seen |= {variable.name for variable in _variables(item)}
        return True
    return True


# ---------------------------------------------------------------------------
# analysis


class FamilyComponent(QiecModel):
    """A strongly connected component of the family dependency graph.

    Parameters
    ----------
    families : tuple of FamilyId
        Its families.
    cyclic : bool
        Whether some rule concludes a family of the component from a
        premise in it.
    linear : bool
        Whether every such rule has at most one premise in the component.
    """

    families: tuple[FamilyId, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    cyclic: bool
    linear: bool


class PresentationAnalysis(QiecModel):
    """What an executor needs to know of a presentation to plan for it.

    Parameters
    ----------
    rules : int
        The number of rules.
    axiom_slots : int
        The number of axiom slots.
    families : int
        The number of judgment forms.
    max_arity : int
        The largest number of premises of a rule.
    size : int
        The number of index-term nodes over all judgments of all rules
        and slots, which bounds the cost of matching every rule once.
    components : tuple of FamilyComponent
        The components of the dependency graph, premises before
        conclusions.
    join_variables : tuple of tuple of ConstructorId and tuple of str
        For each rule, the variables shared by two or more premises,
        which are the attributes its join is on.
    """

    rules: int
    axiom_slots: int
    families: int
    max_arity: int
    size: int
    components: tuple[FamilyComponent, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    join_variables: tuple[tuple[ConstructorId, tuple[str, ...]], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )

    def recursion(self) -> Recursion:
        """The weakest recursion contract the family structure guarantees.

        The family graph over-approximates the derived hypergraph: if no
        family depends on itself no item does.

        Returns
        -------
        Recursion
            `Acyclic` when no component is cyclic; `LinearCyclic` when
            every cyclic component is linear; `MonotonePolynomial`
            otherwise.
        """
        cyclic = [component for component in self.components if component.cyclic]
        if not cyclic:
            return Acyclic()
        if all(component.linear for component in cyclic):
            return LinearCyclic()
        return MonotonePolynomial()


def _term_size(term: IndexTerm) -> int:
    """The number of nodes of an index term.

    Parameters
    ----------
    term : IndexTerm
        The term.

    Returns
    -------
    int
        Its nodes.
    """
    if isinstance(term, IndexConstructor):
        return 1 + sum(_term_size(item) for item in term.arguments)
    if isinstance(term, IndexSum):
        return 1 + len(term.terms)
    return 1


def _components(
    nodes: Sequence[FamilyId], edges: Mapping[FamilyId, Sequence[FamilyId]]
) -> list[list[FamilyId]]:
    """Strongly connected components in dependency order (Tarjan).

    Parameters
    ----------
    nodes : Sequence of FamilyId
        The families.
    edges : Mapping[FamilyId, Sequence of FamilyId]
        For each family, the families its rules conclude from it.

    Returns
    -------
    list of list of FamilyId
        The components, each before every component that depends on it.
    """
    index: dict[FamilyId, int] = {}
    low: dict[FamilyId, int] = {}
    stack: list[FamilyId] = []
    on_stack: set[FamilyId] = set()
    found: list[list[FamilyId]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work: list[tuple[FamilyId, Iterator[FamilyId]]] = [
            (root, iter(edges.get(root, ())))
        ]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, successors = work[-1]
            advanced = False
            for successor in successors:
                if successor not in index:
                    index[successor] = low[successor] = counter
                    counter += 1
                    stack.append(successor)
                    on_stack.add(successor)
                    work.append((successor, iter(edges.get(successor, ()))))
                    advanced = True
                    break
                if successor in on_stack:
                    low[node] = min(low[node], index[successor])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[FamilyId] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                found.append(component)
    found.reverse()
    return found


# ---------------------------------------------------------------------------
# context structure


class ContextStructure(QiecModel):
    """The structural rules a context admits, as declared data.

    A use of the variables of a context, read from left to right, is
    licensed when it arises from the context in order by the declared
    rules: exchange permutes uses, weakening drops a variable, and
    contraction uses one twice. Nothing is inferred from how a context is
    stored.

    Parameters
    ----------
    exchange : bool
        Whether variables may be used out of order.
    weakening : bool
        Whether a variable may go unused.
    contraction : bool
        Whether a variable may be used more than once.
    """

    exchange: bool = False
    weakening: bool = False
    contraction: bool = False

    def admits(self, required: ContextStructure) -> bool:
        """Whether every rule another structure needs is admitted here.

        Parameters
        ----------
        required : ContextStructure
            The rules a derivation uses.

        Returns
        -------
        bool
            True when each rule used is admitted.
        """
        return (
            (self.exchange or not required.exchange)
            and (self.weakening or not required.weakening)
            and (self.contraction or not required.contraction)
        )


ORDERED = ContextStructure()
"""No structural rule: every variable used once, in order (Lambek)."""

LINEAR = ContextStructure(exchange=True)
"""Exchange only: every variable used exactly once."""

AFFINE = ContextStructure(exchange=True, weakening=True)
"""Exchange and weakening: every variable used at most once."""

RELEVANT = ContextStructure(exchange=True, contraction=True)
"""Exchange and contraction: every variable used at least once."""

CARTESIAN = ContextStructure(exchange=True, weakening=True, contraction=True)
"""Every structural rule."""


# ---------------------------------------------------------------------------
# the presentation


PRESENTATION_PROTOCOL = "quivers.rules"
"""The source protocol of the module a presentation lowers to."""


class RulePresentation(QiecModel):
    """A deduction system stated as QIEC declarations.

    Construction validates the presentation once: the rules are well
    scoped and range restricted, and the lowered families, constructors
    and side conditions pass `validate_module`. The indexes an executor
    reads are built once, on first use.

    Parameters
    ----------
    id : PresentationId
        The presentation's identity.
    name : str
        Its name, which is the name of the module it lowers to.
    index_sorts : tuple of UserIndexSort
        The closed sorts its judgments are indexed by.
    judgments : tuple of JudgmentDecl
        Its judgment forms.
    rules : tuple of Rule
        Its rules, in order.
    axioms : tuple of AxiomSlot
        Its axiom slots.
    conditions : tuple of NamedComputation
        The computations its side conditions name.
    structure : ContextStructure
        The structural rules its contexts admit. The default admits none:
        exchange, weakening and contraction are never inferred.
    equations : tuple of Equation
        The equations between open derivations it intends an
        interpretation to preserve.

    Raises
    ------
    didactic.api.ValidationError
        If an identity or name repeats, a judgment names an unknown form
        or has the wrong number of arguments, a variable of a conclusion
        or side condition is not bound by the premises or the telescope,
        a pattern cannot be matched deterministically, a side condition
        is not a pure ``Bool`` computation of matching parameters, or the
        lowered module fails the QIEC checker.
    """

    id: PresentationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    index_sorts: tuple[UserIndexSort, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    judgments: tuple[JudgmentDecl, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    rules: tuple[Rule, ...] = dx.field(default=(), opaque=True, extras=OPAQUE_BY_VALUE)
    axioms: tuple[AxiomSlot, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    conditions: tuple[NamedComputation, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    structure: ContextStructure = dx.field(
        default=ORDERED, opaque=True, extras=OPAQUE_BY_VALUE
    )
    equations: tuple[Equation, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Validate the presentation once, by its own rules and by QIEC.

        Raises
        ------
        ValueError
            If the presentation is ill-formed.
        """
        _validate_presentation(self)

    # ---- lowering -------------------------------------------------------

    @cached_property
    def computed_weight_rules(self) -> tuple[Rule, ...]:
        """Rules requiring an executable, telescope-dependent weight.

        Returns
        -------
        tuple of Rule
            The rules in presentation order.
        """
        return tuple(
            rule for rule in self.rules if isinstance(rule.weight, ComputedRuleWeight)
        )

    @cached_property
    def constructors_by_id(self) -> Mapping[ConstructorId, Rule | AxiomSlot]:
        """Every rule and slot by its identity.

        Returns
        -------
        Mapping[ConstructorId, Rule or AxiomSlot]
            Read-only; built once.
        """
        table: dict[ConstructorId, Rule | AxiomSlot] = {}
        for item in (*self.rules, *self.axioms):
            table[item.id] = item
        return MappingProxyType(table)

    @cached_property
    def judgments_by_id(self) -> Mapping[FamilyId, JudgmentDecl]:
        """Every judgment form by its identity.

        Returns
        -------
        Mapping[FamilyId, JudgmentDecl]
            Read-only; built once.
        """
        return MappingProxyType({judgment.id: judgment for judgment in self.judgments})

    @cached_property
    def conditions_by_id(self) -> Mapping[ComputationId, NamedComputation]:
        """Every side condition's computation by its identity.

        Returns
        -------
        Mapping[ComputationId, NamedComputation]
            Read-only; built once.
        """
        return MappingProxyType({item.id: item for item in self.conditions})

    def declarations(
        self,
    ) -> tuple[tuple[FamilyDecl, ...], tuple[ConstructorDecl, ...]]:
        """The QIEC families and constructors of the presentation.

        Each judgment form is a closed family whose constructors are the
        rules and slots that conclude it. A rule's fields are its premises,
        named ``premise0``, ``premise1``, ..., then its value fields; a
        premise's field type is its judgment's family applied to the
        premise arguments.

        Returns
        -------
        tuple of tuple of FamilyDecl and tuple of ConstructorDecl
            The families and the constructors.
        """
        return self._declarations

    @cached_property
    def _declarations(
        self,
    ) -> tuple[tuple[FamilyDecl, ...], tuple[ConstructorDecl, ...]]:
        """Return the lowered declarations for this immutable presentation.

        Returns
        -------
        tuple of tuple of FamilyDecl and tuple of ConstructorDecl
            The declarations, reused for subsequent folds and checks.
        """
        return _lower(self)

    def to_module(self) -> QiecModule:
        """The QIEC module of the presentation.

        Returns
        -------
        QiecModule
            Its index sorts, families, constructors and conditions.
        """
        families, constructors = self.declarations()
        return QiecModule(
            module=self.name,
            source_protocol=PRESENTATION_PROTOCOL,
            index_sorts=self.index_sorts,
            families=families,
            constructors=constructors,
            computations=self.conditions,
        )

    @cached_property
    def registry(self) -> KernelRegistry:
        """The QIEC registry of the lowered module, resolved once.

        Returns
        -------
        KernelRegistry
            The registry derivations are checked against.
        """
        return validate_module(self.to_module())

    # ---- indexes --------------------------------------------------------

    @cached_property
    def rules_by_conclusion(self) -> Mapping[FamilyId, tuple[Rule, ...]]:
        """The rules that conclude each judgment form.

        Returns
        -------
        Mapping[FamilyId, tuple of Rule]
            Read-only; built once.
        """
        table: dict[FamilyId, list[Rule]] = {}
        for rule in self.rules:
            table.setdefault(rule.conclusion.family, []).append(rule)
        return MappingProxyType({key: tuple(value) for key, value in table.items()})

    @cached_property
    def rules_by_premise(self) -> Mapping[FamilyId, tuple[tuple[Rule, int], ...]]:
        """The rules and premise positions that each judgment form can fill.

        Returns
        -------
        Mapping[FamilyId, tuple of tuple of Rule and int]
            Read-only; built once.
        """
        table: dict[FamilyId, list[tuple[Rule, int]]] = {}
        for rule in self.rules:
            for position, premise in enumerate(rule.premises):
                table.setdefault(premise.judgment.family, []).append((rule, position))
        return MappingProxyType({key: tuple(value) for key, value in table.items()})

    @cached_property
    def analysis(self) -> PresentationAnalysis:
        """Sizes, components and join variables, for planning.

        Returns
        -------
        PresentationAnalysis
            Built once in time linear in the presentation.
        """
        return _analyse(self)


def _lower(
    presentation: RulePresentation,
) -> tuple[tuple[FamilyDecl, ...], tuple[ConstructorDecl, ...]]:
    """The QIEC declarations of a presentation.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.

    Returns
    -------
    tuple of tuple of FamilyDecl and tuple of ConstructorDecl
        The families and constructors.
    """
    forms = {judgment.id: judgment for judgment in presentation.judgments}
    families = {judgment.id: judgment for judgment in presentation.judgments}
    owned: dict[FamilyId, list[ConstructorId]] = {key: [] for key in families}
    constructors: list[ConstructorDecl] = []
    for rule in presentation.rules:
        fields = (
            tuple(
                FieldDef(
                    name=f"premise{position}",
                    type=_judgment_type(premise.judgment, forms),
                )
                for position, premise in enumerate(rule.premises)
            )
            + rule.values
        )
        owned[rule.conclusion.family].append(rule.id)
        constructors.append(
            ConstructorDecl(
                id=rule.id,
                family=rule.conclusion.family,
                name=rule.name,
                telescope=rule.telescope,
                fields=fields,
                result_indices=rule.conclusion.arguments,
            )
        )
    for slot in presentation.axioms:
        owned[slot.judgment.family].append(slot.id)
        constructors.append(
            ConstructorDecl(
                id=slot.id,
                family=slot.judgment.family,
                name=slot.name,
                telescope=slot.telescope,
                fields=slot.values,
                result_indices=slot.judgment.arguments,
            )
        )
    declared = tuple(
        FamilyDecl(
            id=judgment.id,
            name=judgment.name,
            parameters=(),
            indices=_family_indices(judgment),
            constructors=tuple(owned[judgment.id]),
        )
        for judgment in presentation.judgments
    )
    return declared, tuple(constructors)


def _family_indices(form: JudgmentDecl) -> Telescope:
    """The index binders of a judgment form's family.

    They are named ``<form>.<argument>`` so that no rule's schematic
    variable shadows them, which the QIEC checker forbids.

    Parameters
    ----------
    form : JudgmentDecl
        The judgment form.

    Returns
    -------
    Telescope
        The renamed binders.
    """
    return tuple(
        binder.with_(name=f"{form.name}.{binder.name}") for binder in form.indices
    )


def _judgment_type(
    judgment: Judgment, forms: Mapping[FamilyId, JudgmentDecl]
) -> TypeApplication:
    """The type of a proof of a judgment.

    Parameters
    ----------
    judgment : Judgment
        The judgment.
    forms : Mapping[FamilyId, JudgmentDecl]
        The judgment forms.

    Returns
    -------
    TypeApplication
        The family's type constructor applied to the arguments.
    """
    form = forms[judgment.family]
    family = FamilyDecl(
        id=form.id,
        name=form.name,
        parameters=(),
        indices=_family_indices(form),
        constructors=(),
    )
    return TypeApplication(
        constructor=family.type_constructor, arguments=judgment.arguments
    )


def _validate_presentation(presentation: RulePresentation) -> None:
    """Check a presentation by its own rules and by the QIEC checker.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.

    Raises
    ------
    RuleError
        If it is ill-formed.
    """
    forms = {judgment.id: judgment for judgment in presentation.judgments}
    if len(forms) != len(presentation.judgments):
        raise RuleError("a judgment form repeats")
    identities = [item.id for item in (*presentation.rules, *presentation.axioms)]
    if len(set(identities)) != len(identities):
        raise RuleError("a rule or slot identity repeats")
    names = [item.name for item in (*presentation.rules, *presentation.axioms)]
    if len(set(names)) != len(names):
        raise RuleError("a rule or slot name repeats")
    conditions = {item.id: item for item in presentation.conditions}

    def check_judgment(judgment: Judgment, where: str) -> None:
        """Require a known form and its number of arguments.

        Parameters
        ----------
        judgment : Judgment
            The judgment.
        where : str
            Where it occurs, for the diagnostic.

        Raises
        ------
        RuleError
            If the form is unknown or the arity differs.
        """
        form = forms.get(judgment.family)
        if form is None:
            raise RuleError(f"{where} names an unknown judgment form")
        for argument in judgment.arguments:
            if any(not item.arguments for item in _constructors(argument)):
                raise RuleError(
                    f"{where} applies a nullary index constructor; write it as an "
                    "IndexLiteral, its canonical form"
                )
        if len(judgment.arguments) != len(form.indices):
            raise RuleError(
                f"{where} gives {len(judgment.arguments)} arguments to "
                f"{form.name!r}, which takes {len(form.indices)}"
            )

    for rule in presentation.rules:
        if rule.weight is not None and not isinstance(
            rule.weight, (ParameterId, ComputedRuleWeight)
        ):
            raise RuleError(f"rule {rule.name!r} has an invalid weight descriptor")
        bound: set[str] = set()
        for position, premise in enumerate(rule.premises):
            for hypothesis in premise.hypotheses:
                check_judgment(
                    hypothesis, f"hypothesis of premise {position} of {rule.name!r}"
                )
            check_judgment(premise.judgment, f"premise {position} of {rule.name!r}")
            for argument in premise.judgment.arguments:
                if not _bindable(argument, bound):
                    raise RuleError(
                        f"premise {position} of {rule.name!r} has a sum with two "
                        "unbound summands, so it does not match deterministically"
                    )
                bound |= {variable.name for variable in _variables(argument)}
        check_judgment(rule.conclusion, f"the conclusion of {rule.name!r}")
        telescope = {binder.name for binder in rule.telescope}
        used = {
            variable.name
            for judgment in (
                rule.conclusion,
                *(premise.judgment for premise in rule.premises),
                *(h for premise in rule.premises for h in premise.hypotheses),
            )
            for argument in judgment.arguments
            for variable in _variables(argument)
        }
        if not used <= telescope:
            raise RuleError(
                f"{rule.name!r} uses variables {sorted(used - telescope)} that its "
                "telescope does not declare"
            )
        conclusion = {
            variable.name
            for argument in rule.conclusion.arguments
            for variable in _variables(argument)
        }
        # A variable of a hypothesis is fixed by the derivation that
        # discharges it, as in an introduction rule; grounding forward
        # from closed items cannot find it, and the reference
        # interpretation refuses such rules.
        bound |= {
            variable.name
            for premise in rule.premises
            for hypothesis in premise.hypotheses
            for argument in hypothesis.arguments
            for variable in _variables(argument)
        }
        if not conclusion <= bound:
            raise RuleError(
                f"the conclusion of {rule.name!r} uses {sorted(conclusion - bound)}, "
                "which no premise binds"
            )
        if rule.side_condition is not None:
            _check_side_condition(rule, bound, conditions)
    for slot in presentation.axioms:
        check_judgment(slot.judgment, f"slot {slot.name!r}")
        used = {
            variable.name
            for argument in slot.judgment.arguments
            for variable in _variables(argument)
        }
        if not used <= {binder.name for binder in slot.telescope}:
            raise RuleError(f"slot {slot.name!r} uses variables it does not declare")
    validate_module(presentation.to_module())


def named_conditions(presentation: RulePresentation) -> tuple[SideCondition, ...]:
    """The side conditions of a presentation's rules.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.

    Returns
    -------
    tuple of SideCondition
        One per rule that has one, in rule order.
    """
    return tuple(
        rule.side_condition for rule in presentation.rules if rule.side_condition
    )


def _check_side_condition(
    rule: Rule, bound: set[str], conditions: Mapping[ComputationId, NamedComputation]
) -> None:
    """Require a side condition to be a pure ``Bool`` computation of the rule.

    Parameters
    ----------
    rule : Rule
        The rule.
    bound : set of str
        The variables its premises bind.
    conditions : Mapping[ComputationId, NamedComputation]
        The presentation's conditions.

    Raises
    ------
    RuleError
        If the computation is unknown, has a telescope, is not pure, does
        not return ``Bool``, or its parameters do not match the arguments.
    """
    condition = rule.side_condition
    assert condition is not None
    computation = conditions.get(condition.computation)
    if computation is None:
        raise RuleError(
            f"the side condition of {rule.name!r} names an unknown computation"
        )
    if computation.telescope:
        raise RuleError(f"the side condition of {rule.name!r} has a telescope")
    if computation.type.effects != EMPTY_ROW or computation.type.result != BOOL:
        raise RuleError(f"the side condition of {rule.name!r} is not a pure Bool")
    if len(computation.parameters) != len(condition.arguments):
        raise RuleError(f"the side condition of {rule.name!r} takes another arity")
    binders = {binder.name: binder for binder in rule.telescope}
    fields = {field.name: field for field in rule.values}
    for name, parameter in zip(
        condition.arguments, computation.parameters, strict=True
    ):
        if name in fields:
            if fields[name].type != parameter.type:
                raise RuleError(
                    f"the side condition of {rule.name!r} takes {name!r} at another type"
                )
            continue
        binder = binders.get(name)
        if binder is None or name not in bound:
            raise RuleError(
                f"the side condition of {rule.name!r} reads {name!r}, which the "
                "premises do not bind"
            )
        assert isinstance(binder, IndexBinder)
        expected = INT if isinstance(binder.sort, NatSort) else STRING
        if parameter.type != expected:
            raise RuleError(
                f"the side condition of {rule.name!r} takes {name!r} at another type"
            )


def _analyse(presentation: RulePresentation) -> PresentationAnalysis:
    """Sizes, components and join variables of a presentation.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.

    Returns
    -------
    PresentationAnalysis
        The analysis.
    """
    edges: dict[FamilyId, list[FamilyId]] = {}
    for rule in presentation.rules:
        for premise in rule.premises:
            edges.setdefault(premise.judgment.family, []).append(rule.conclusion.family)
    order = _components([judgment.id for judgment in presentation.judgments], edges)
    member = {family: index for index, group in enumerate(order) for family in group}
    cyclic = [False] * len(order)
    linear = [True] * len(order)
    for rule in presentation.rules:
        component = member[rule.conclusion.family]
        inside = sum(
            1
            for premise in rule.premises
            if member[premise.judgment.family] == component
        )
        if inside:
            cyclic[component] = True
        if inside > 1:
            linear[component] = False
    joins: list[tuple[ConstructorId, tuple[str, ...]]] = []
    for rule in presentation.rules:
        counts: dict[str, int] = {}
        for premise in rule.premises:
            names = {
                variable.name
                for argument in premise.judgment.arguments
                for variable in _variables(argument)
            }
            for name in names:
                counts[name] = counts.get(name, 0) + 1
        joins.append(
            (rule.id, tuple(sorted(name for name, n in counts.items() if n > 1)))
        )
    size = sum(
        _term_size(argument)
        for rule in presentation.rules
        for judgment in (rule.conclusion, *(p.judgment for p in rule.premises))
        for argument in judgment.arguments
    ) + sum(
        _term_size(argument)
        for slot in presentation.axioms
        for argument in slot.judgment.arguments
    )
    return PresentationAnalysis(
        rules=len(presentation.rules),
        axiom_slots=len(presentation.axioms),
        families=len(presentation.judgments),
        max_arity=max((rule.arity for rule in presentation.rules), default=0),
        size=size,
        components=tuple(
            FamilyComponent(
                families=tuple(group), cyclic=cyclic[index], linear=linear[index]
            )
            for index, group in enumerate(order)
        ),
        join_variables=tuple(joins),
    )


# ---------------------------------------------------------------------------
# the cost of presentation work


_READING = Citation(
    reference="every symbol of the input is read, by an adversary argument",
    locator="complexity claims, Q24",
)

PRESENTATION_WORK = ComplexityClaim(
    query="presentation work: construction, matching, hashing, folding",
    parameters=(
        ComplexityParameter(name="P", meaning="index-term nodes over all judgments"),
        ComplexityParameter(name="R", meaning="rules and axiom slots"),
        ComplexityParameter(name="N", meaning="nodes of the derivation folded"),
    ),
    model=WORD_RAM_EXPECTED_HASHING,
    bounds=(
        ComplexityBound(
            phase=Phase.CONSTRUCTION,
            resource="time",
            expression="P + R",
            status=ProvedOptimal(
                lower_bound="P + R",
                model=WORD_RAM_EXPECTED_HASHING,
                sources=(_READING,),
            ),
        ),
        ComplexityBound(
            phase=Phase.CONSTRUCTION,
            resource="space",
            expression="P + R",
            status=ProvedOptimal(
                lower_bound="P + R",
                model=WORD_RAM_EXPECTED_HASHING,
                sources=(_READING,),
            ),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="time",
            expression="N",
            status=ProvedOptimal(
                lower_bound="N", model=WORD_RAM_EXPECTED_HASHING, sources=(_READING,)
            ),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="space",
            expression="N",
            status=BestKnown(sources=(_READING,)),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="time",
            expression="N",
            status=BestKnown(
                sources=(
                    Citation(
                        reference=(
                            "Baur and Strassen 1983, The complexity of partial "
                            "derivatives"
                        ),
                        locator="the cheap gradient theorem",
                        doi="10.1016/0304-3975(83)90110-X",
                    ),
                )
            ),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="space",
            expression="N",
            status=Unresolved(
                obligation=(
                    "whether a fold can be differentiated in less than linear "
                    "space without recomputation"
                )
            ),
        ),
        ComplexityBound(
            phase=Phase.REFIT,
            resource="time",
            expression="N",
            status=ProvedOptimal(
                lower_bound="N", model=WORD_RAM_EXPECTED_HASHING, sources=(_READING,)
            ),
        ),
        ComplexityBound(
            phase=Phase.REFIT,
            resource="space",
            expression="N",
            status=BestKnown(sources=(_READING,)),
        ),
    ),
    conditions=(
        "a presentation is validated and indexed once, when it is built and on "
        "first use, and never again per firing, fold or refit",
        "matching a premise costs the size of its pattern",
        "an interned item hashes in constant time",
        "operations of the interpretation cost one unit",
    ),
)
"""The cost of building, matching and folding a presentation."""


# ---------------------------------------------------------------------------
# structural signatures


_SIGNATURE_PRESENTATIONS: WeakKeyDictionary[Signature, RulePresentation] = (
    WeakKeyDictionary()
)
"""The presentation of each signature, built once."""


def _sort_family(signature: Signature, sort: str) -> FamilyId:
    """The judgment form of an object sort of a signature.

    Parameters
    ----------
    signature : Signature
        The signature.
    sort : str
        The sort.

    Returns
    -------
    FamilyId
        Its identity.
    """
    return FamilyId.derive("signature", signature.name, "sort", sort)


def _value_type(signature: Signature, sort: str) -> TypeExpr:
    """The value type of a data or index position.

    Parameters
    ----------
    signature : Signature
        The signature.
    sort : str
        A data or index sort.

    Returns
    -------
    TypeExpr
        ``Int`` for an index sort; for a data sort, ``String``, ``Int`` or
        ``Real`` when its vocabulary is of one kind, and otherwise a type
        named after the sort.
    """
    declared = signature.sorts[sort]
    if declared.kind == "index":
        return INT
    kinds = {entry.kind for entry in declared.vocab}
    by_kind: dict[str, TypeExpr] = {"string": STRING, "integer": INT, "float": REAL}
    if len(kinds) == 1:
        return by_kind[next(iter(kinds))]
    return TypeApplication(
        constructor=TypeConstructorRef(
            id=TypeId.derive("signature", signature.name, "data", sort), name=sort
        )
    )


def presentation_of(signature: Signature) -> RulePresentation:
    """The rule presentation of a structural signature.

    Each object sort is a judgment form without arguments. A constructor
    is a rule with no side condition: its object arguments are premises
    in order and its data and index arguments are value fields. A binder
    is a rule whose annotation arguments are premises and whose scoped
    arguments are premises with the bound variables as hypotheses. The
    presentation is built once per signature.

    Parameters
    ----------
    signature : Signature
        A term signature.

    Returns
    -------
    RulePresentation
        Its presentation.

    Raises
    ------
    RuleError
        If the signature is graph shaped, or a binder binds a variable of
        a sort that is not an object sort.
    """
    known = _SIGNATURE_PRESENTATIONS.get(signature)
    if known is not None:
        return known
    if signature.vertex_kinds_t or signature.edge_kinds_t:
        raise RuleError(
            f"signature {signature.name!r} is graph shaped; a graph is not a term "
            "algebra and has no rule presentation here"
        )
    objects = [sort.name for sort in signature.sorts_t if sort.kind == "object"]

    def judgment(sort: str) -> Judgment:
        """The judgment of an object sort.

        Parameters
        ----------
        sort : str
            The sort.

        Returns
        -------
        Judgment
            Its judgment.
        """
        return Judgment(family=_sort_family(signature, sort))

    def split(domain: Sequence[str]) -> tuple[list[Premise], list[FieldDef]]:
        """Premises and value fields of a positional domain.

        Parameters
        ----------
        domain : Sequence of str
            The sorts of the positions.

        Returns
        -------
        tuple of list of Premise and list of FieldDef
            The premises of object positions and the fields of the others.
        """
        premises: list[Premise] = []
        values: list[FieldDef] = []
        for position, sort in enumerate(domain):
            if signature.sorts[sort].kind == "object":
                premises.append(Premise(judgment=judgment(sort)))
            else:
                values.append(
                    FieldDef(name=f"value{position}", type=_value_type(signature, sort))
                )
        return premises, values

    rules: list[Rule] = []
    for constructor in signature.constructors_t:
        premises, values = split(constructor.domain)
        rules.append(
            Rule(
                id=ConstructorId.derive("signature", signature.name, constructor.name),
                name=constructor.name,
                premises=tuple(premises),
                conclusion=judgment(constructor.codomain),
                values=tuple(values),
            )
        )
    for binder in signature.binders_t:
        hypotheses = []
        for variable in binder.binds:
            if variable.sort not in objects:
                raise RuleError(
                    f"binder {binder.name!r} binds {variable.var!r} of sort "
                    f"{variable.sort!r}, which is not an object sort"
                )
            hypotheses.append(judgment(variable.sort))
        annotations = [b.annot_sort for b in binder.binds if b.annot_sort is not None]
        premises, values = split(annotations)
        for argument in binder.scoped:
            premises.append(
                Premise(judgment=judgment(argument.sort), after=tuple(hypotheses))
            )
        rules.append(
            Rule(
                id=ConstructorId.derive("signature", signature.name, binder.name),
                name=binder.name,
                premises=tuple(premises),
                conclusion=judgment(binder.codomain),
                values=tuple(values),
            )
        )
    presentation = RulePresentation(
        id=PresentationId.derive("signature", signature.name),
        name=signature.name,
        judgments=tuple(
            JudgmentDecl(id=_sort_family(signature, sort), name=sort)
            for sort in objects
        ),
        rules=tuple(rules),
    )
    _SIGNATURE_PRESENTATIONS[signature] = presentation
    return presentation


# ---------------------------------------------------------------------------
# open derivations


class StructureError(RuleError):
    """A derivation uses its context in a way the structure does not admit."""


class OpenDerivation(QiecModel):
    """A derivation with open hypotheses: an arrow of the free multicategory.

    It derives its body's item from the items of its context, in order.
    Its holes refer to the context by de Bruijn index, counted from the
    innermost (last) item, and inside a premise with hypotheses the
    context is extended by them.

    Parameters
    ----------
    context : tuple of Item
        The hypotheses, outermost first.
    body : Derivation or Hole
        The derivation.
    """

    context: tuple[Item, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    body: Derivation | Hole = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @property
    def item(self) -> Item:
        """The item derived.

        Returns
        -------
        Item
            The body's item.
        """
        return self.body.item


class Equation(QiecModel):
    """An equation between two open derivations an interpretation may preserve.

    Parameters
    ----------
    law : Law
        The law the equation is an instance of.
    left : OpenDerivation
        One side.
    right : OpenDerivation
        The other side, with the same context and item.

    Raises
    ------
    didactic.api.ValidationError
        If the sides differ in context or item.
    """

    law: Law = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    left: OpenDerivation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    right: OpenDerivation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require one context and one item on both sides.

        Raises
        ------
        ValueError
            If they differ.
        """
        if self.left.context != self.right.context or self.left.item != self.right.item:
            raise ValueError("the two sides of an equation share context and item")


def identity_derivation(item: Item) -> OpenDerivation:
    """The identity arrow on an item: its own hypothesis.

    Parameters
    ----------
    item : Item
        The item.

    Returns
    -------
    OpenDerivation
        The derivation of the item from itself.
    """
    return OpenDerivation(context=(item,), body=Hole(index=0, item=item))


def extend_context[T](
    before: tuple[T, ...], context: tuple[T, ...], after: tuple[T, ...]
) -> tuple[T, ...]:
    """The context of a premise: its left hypotheses, the context, its right ones.

    Every operation on open derivations places a premise's hypotheses by
    this function or by the counts it implies, so the side of a
    hypothesis is decided in one place.

    Parameters
    ----------
    before : tuple
        The left block.
    context : tuple
        The enclosing context.
    after : tuple
        The right block.

    Returns
    -------
    tuple
        ``before + context + after``.
    """
    return (*before, *context, *after)


type Blocks = tuple[tuple[Item, ...], tuple[Item, ...]]
"""The left and right hypotheses of one premise."""


def _hypotheses(presentation: RulePresentation, node: Derivation) -> tuple[Blocks, ...]:
    """The hypotheses each premise of a node is derived under.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    node : Derivation
        The node.

    Returns
    -------
    tuple of Blocks
        One pair of left and right blocks per premise, each outermost
        first; empty for an axiom.

    Raises
    ------
    RuleError
        If the node's constructor is not in the presentation, or its
        number of premises is not the rule's.
    """
    rule = presentation.constructors_by_id.get(node.constructor)
    if rule is None:
        raise RuleError("a derivation names a rule of another presentation")
    if isinstance(rule, AxiomSlot):
        if node.premises:
            raise RuleError(f"axiom {rule.name!r} has no premises")
        return ()
    if len(node.premises) != len(rule.premises):
        raise RuleError(
            f"{rule.name!r} takes {len(rule.premises)} premises, got {len(node.premises)}"
        )
    binding = dict(node.binding)

    def items(block: tuple[Judgment, ...]) -> tuple[Item, ...]:
        """The items of a block of hypotheses at the node's binding.

        Parameters
        ----------
        block : tuple of Judgment
            The block.

        Returns
        -------
        tuple of Item
            Its items.
        """
        return tuple(
            Item(
                family=hypothesis.family,
                arguments=tuple(
                    evaluate_index(a, binding) for a in hypothesis.arguments
                ),
            )
            for hypothesis in block
        )

    return tuple(
        (items(premise.before), items(premise.after)) for premise in rule.premises
    )


def _counts(blocks: tuple[Blocks, ...], position: int) -> tuple[int, int]:
    """The numbers of left and right hypotheses of one premise.

    Parameters
    ----------
    blocks : tuple of Blocks
        The blocks of a node's premises, empty for an axiom.
    position : int
        The premise.

    Returns
    -------
    tuple of int and int
        The sizes of its left and right blocks.
    """
    if not blocks:
        return (0, 0)
    before, after = blocks[position]
    return (len(before), len(after))


def _map_holes(
    presentation: RulePresentation,
    body: ProofNode,
    outer: int,
    new_outer: int,
    replace: Callable[[int, int, int], ProofNode],
) -> ProofNode:
    """Replace the holes that refer to the outer context of a derivation.

    Inside the derivation the context is the outer one with ``left``
    hypotheses inserted at its left and ``right`` at its right, summed
    over the premises above. Holes count from the right, so a hole with
    index ``i`` is bound on the right when ``i < right``, refers to outer
    level ``outer - 1 - (i - right)`` when ``i < right + outer``, and is
    bound on the left otherwise. A right-bound hole is kept; an outer one
    becomes ``replace(level, left, right)``; a left-bound one moves past a
    new outer context of ``new_outer`` hypotheses.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation, for the hypotheses of each premise.
    body : Derivation or Hole
        The derivation.
    outer : int
        The length of its outer context.
    new_outer : int
        The length of the outer context after the replacement.
    replace : Callable[[int, int, int], Derivation or Hole]
        The replacement of an outer level, given the left and right
        hypotheses around it; it must be valid in a context of
        ``left + new_outer + right`` hypotheses.

    Returns
    -------
    Derivation or Hole
        The derivation with the outer holes replaced.

    Raises
    ------
    RuleError
        If a hole refers outside its context.
    """

    def mapped(hole: Hole, left: int, right: int) -> ProofNode:
        """One hole after the replacement.

        Parameters
        ----------
        hole : Hole
            The hole.
        left, right : int
            The hypotheses around the outer context at it.

        Returns
        -------
        Derivation or Hole
            The kept, replaced or moved hole.

        Raises
        ------
        RuleError
            If it lies outside its context.
        """
        if hole.index < right:
            return hole
        if hole.index < right + outer:
            return replace(outer - 1 - (hole.index - right), left, right)
        if hole.index < right + outer + left:
            return hole.with_(index=hole.index - outer + new_outer)
        raise RuleError(
            f"hole {hole.index} lies outside a context of {left + outer + right}"
        )

    if isinstance(body, Hole):
        return mapped(body, 0, 0)
    results: list[list[ProofNode]] = []
    frames: list[tuple[Derivation, int, int, tuple[Blocks, ...]]] = []
    frames.append((body, 0, 0, _hypotheses(presentation, body)))
    results.append([])
    finished: ProofNode | None = None
    while frames:
        node, left, right, blocks = frames[-1]
        done = results[-1]
        if len(done) == len(node.premises):
            frames.pop()
            results.pop()
            rebuilt = node.with_(premises=tuple(done))
            if results:
                results[-1].append(rebuilt)
            else:
                finished = rebuilt
            continue
        position = len(done)
        child = node.premises[position]
        before, after = _counts(blocks, position)
        inner_left, inner_right = left + before, right + after
        if isinstance(child, Hole):
            done.append(mapped(child, inner_left, inner_right))
            continue
        frames.append(
            (child, inner_left, inner_right, _hypotheses(presentation, child))
        )
        results.append([])
    assert finished is not None
    return finished


def _weaken(
    presentation: RulePresentation,
    body: ProofNode,
    outer: int,
    left: int,
    right: int,
) -> ProofNode:
    """A derivation moved into a context with hypotheses on both sides.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    body : Derivation or Hole
        The derivation.
    outer : int
        The length of its context.
    left, right : int
        The hypotheses added at the left and at the right.

    Returns
    -------
    Derivation or Hole
        The derivation in the context ``left + outer + right``.
    """
    if left == 0 and right == 0:
        return body
    items = _hole_items(presentation, body, outer)
    total = left + outer + right

    def shifted(level: int, inner_left: int, inner_right: int) -> Hole:
        """The hole of an outer level after the move.

        Parameters
        ----------
        level : int
            The level in the old context.
        inner_left, inner_right : int
            The hypotheses around it inside the derivation.

        Returns
        -------
        Hole
            The same variable, indexed in the larger context.
        """
        del inner_left
        return Hole(index=inner_right + total - 1 - (left + level), item=items[level])

    return _map_holes(presentation, body, outer, total, shifted)


def _hole_items(
    presentation: RulePresentation, body: ProofNode, outer: int
) -> dict[int, Item]:
    """The item each outer hole of a derivation assumes.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    body : Derivation or Hole
        The derivation.
    outer : int
        The length of its context.

    Returns
    -------
    dict of int to Item
        The item of each used outer level.

    Raises
    ------
    RuleError
        If two holes of one level assume different items.
    """
    items: dict[int, Item] = {}
    stack: list[tuple[ProofNode, int]] = [(body, 0)]
    while stack:
        node, right = stack.pop()
        if isinstance(node, Hole):
            if right <= node.index < right + outer:
                level = outer - 1 - (node.index - right)
                known = items.setdefault(level, node.item)
                if known != node.item:
                    raise RuleError(f"level {level} is assumed at two items")
            continue
        blocks = _hypotheses(presentation, node)
        for position, child in enumerate(node.premises):
            stack.append((child, right + _counts(blocks, position)[1]))
    return items


def compose_derivations(
    presentation: RulePresentation,
    outer: OpenDerivation,
    arguments: Sequence[OpenDerivation],
) -> OpenDerivation:
    """Plug derivations into the hypotheses of a derivation, in order.

    This is composition in the free colored nonsymmetric multicategory:
    hypothesis ``k`` of ``outer`` is replaced by ``arguments[k]``, and the
    contexts of the arguments are concatenated in order. It needs no
    structural rule.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    outer : OpenDerivation
        The derivation whose hypotheses are filled.
    arguments : Sequence of OpenDerivation
        One derivation per hypothesis, of its item.

    Returns
    -------
    OpenDerivation
        The composite.

    Raises
    ------
    RuleError
        If the number of arguments or the item of one differs.
    """
    if len(arguments) != len(outer.context):
        raise RuleError(
            f"{len(outer.context)} hypotheses take as many derivations, got "
            f"{len(arguments)}"
        )
    for position, (argument, item) in enumerate(
        zip(arguments, outer.context, strict=True)
    ):
        if argument.item != item:
            raise RuleError(f"argument {position} derives another item")
    context = tuple(item for argument in arguments for item in argument.context)
    offsets: list[int] = []
    start = 0
    for argument in arguments:
        offsets.append(start)
        start += len(argument.context)
    total = len(context)
    placed: list[ProofNode] = []
    for argument, offset in zip(arguments, offsets, strict=True):
        width = len(argument.context)
        items = argument.context

        def moved(
            level: int,
            left: int,
            right: int,
            offset: int = offset,
            items: tuple[Item, ...] = items,
        ) -> Hole:
            """A hole of an argument, placed in the concatenated context.

            Parameters
            ----------
            level : int
                Its level in the argument's context.
            left, right : int
                The argument's own hypotheses around it.
            offset : int
                Where the argument's context starts.
            items : tuple of Item
                The argument's context.

            Returns
            -------
            Hole
                The same hypothesis in the new context.
            """
            del left
            return Hole(index=right + total - 1 - (offset + level), item=items[level])

        placed.append(_map_holes(presentation, argument.body, width, total, moved))

    def filled(level: int, left: int, right: int) -> ProofNode:
        """The derivation that fills one hypothesis of ``outer``.

        Parameters
        ----------
        level : int
            The hypothesis.
        left, right : int
            The hypotheses of ``outer`` around it.

        Returns
        -------
        Derivation or Hole
            The argument, moved between those hypotheses.
        """
        return _weaken(presentation, placed[level], total, left, right)

    return OpenDerivation(
        context=context,
        body=_map_holes(presentation, outer.body, len(outer.context), total, filled),
    )


def substitute_hypotheses(
    presentation: RulePresentation,
    outer: OpenDerivation,
    arguments: Sequence[OpenDerivation],
    context: tuple[Item, ...],
) -> OpenDerivation:
    """Replace every hypothesis of a derivation by a derivation in one context.

    Unlike `compose_derivations`, every argument is in the same context, so a
    hypothesis used twice duplicates its argument and one unused drops
    it. Whether that is licensed is a question for `required_structure`.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    outer : OpenDerivation
        The derivation.
    arguments : Sequence of OpenDerivation
        One derivation per hypothesis, each in ``context``.
    context : tuple of Item
        The shared context.

    Returns
    -------
    OpenDerivation
        The substituted derivation.

    Raises
    ------
    RuleError
        If the number, items or contexts of the arguments differ.
    """
    if len(arguments) != len(outer.context):
        raise RuleError("one derivation per hypothesis")
    for position, (argument, item) in enumerate(
        zip(arguments, outer.context, strict=True)
    ):
        if argument.item != item or argument.context != context:
            raise RuleError(f"argument {position} has another item or context")
    width = len(context)

    def filled(level: int, left: int, right: int) -> ProofNode:
        """The argument of one hypothesis, between the inner hypotheses.

        Parameters
        ----------
        level : int
            The hypothesis.
        left, right : int
            The hypotheses around it.

        Returns
        -------
        Derivation or Hole
            The argument.
        """
        return _weaken(presentation, arguments[level].body, width, left, right)

    return OpenDerivation(
        context=context,
        body=_map_holes(presentation, outer.body, len(outer.context), width, filled),
    )


def rename_hypotheses(
    presentation: RulePresentation,
    derivation: OpenDerivation,
    targets: Sequence[int],
    context: tuple[Item, ...],
) -> OpenDerivation:
    """Move each hypothesis of a derivation to a position of a new context.

    An injective, order-keeping renaming is a weakening; a permutation is
    an exchange; a renaming that sends two hypotheses to one is a
    contraction. The result says which rules it uses through
    `required_structure`.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    derivation : OpenDerivation
        The derivation.
    targets : Sequence of int
        The new level of each old level.
    context : tuple of Item
        The new context.

    Returns
    -------
    OpenDerivation
        The renamed derivation.

    Raises
    ------
    RuleError
        If a target is outside the new context or holds another item.
    """
    if len(targets) != len(derivation.context):
        raise RuleError("one target per hypothesis")
    for level, target in enumerate(targets):
        if (
            not 0 <= target < len(context)
            or context[target] != derivation.context[level]
        ):
            raise RuleError(f"hypothesis {level} goes to a position of another item")
    width = len(context)

    def moved(level: int, left: int, right: int) -> Hole:
        """The hole of an old level in the new context.

        Parameters
        ----------
        level : int
            The old level.
        left, right : int
            The hypotheses around it.

        Returns
        -------
        Hole
            The renamed hole.
        """
        del left
        return Hole(
            index=right + width - 1 - targets[level], item=context[targets[level]]
        )

    return OpenDerivation(
        context=context,
        body=_map_holes(
            presentation, derivation.body, len(derivation.context), width, moved
        ),
    )


def required_structure(
    presentation: RulePresentation, derivation: OpenDerivation
) -> ContextStructure:
    """The structural rules a derivation uses.

    In every scope (the derivation's context, and each premise with
    hypotheses) the uses of visible hypotheses, read left to right, are
    compared with the scope's context in order: a use out of order needs
    exchange, a hypothesis of the scope not used needs weakening, and one
    used twice needs contraction.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    derivation : OpenDerivation
        The derivation.

    Returns
    -------
    ContextStructure
        The least structure that admits it.
    """
    exchange = weakening = contraction = False

    def scope(
        body: ProofNode, width: int, own: set[int], start: int, outer: int
    ) -> list[int]:
        """Check one scope and return the uses of its enclosing context.

        Parameters
        ----------
        body : Derivation or Hole
            The derivation of the scope.
        width : int
            The length of the scope's context.
        own : set of int
            The positions this scope binds.
        start : int
            Where the enclosing context begins in the scope's.
        outer : int
            The length of the enclosing context.

        Returns
        -------
        list of int
            The uses of positions of the enclosing context, in order.
        """
        nonlocal exchange, weakening, contraction
        uses = _uses(body, width)
        if any(later < earlier for earlier, later in itertools.pairwise(uses)):
            exchange = True
        counts = Counter(position for position in uses if position in own)
        if any(counts[position] == 0 for position in own):
            weakening = True
        if any(count > 1 for count in counts.values()):
            contraction = True
        return [p - start for p in uses if start <= p < start + outer]

    def _uses(body: ProofNode, width: int) -> list[int]:
        """The positions used in a scope, in order, after checking inner scopes.

        Parameters
        ----------
        body : Derivation or Hole
            The derivation.
        width : int
            The length of its context.

        Returns
        -------
        list of int
            The positions of the scope's context used, left to right.
        """
        if isinstance(body, Hole):
            return [width - 1 - body.index]
        found: list[int] = []
        blocks = _hypotheses(presentation, body)
        for position, child in enumerate(body.premises):
            before, after = _counts(blocks, position)
            if before or after:
                inner = before + width + after
                own = set(range(before)) | set(range(before + width, inner))
                found.extend(scope(child, inner, own, before, width))
            else:
                found.extend(_uses(child, width))
        return found

    width = len(derivation.context)
    scope(derivation.body, width, set(range(width)), 0, width)
    return ContextStructure(
        exchange=exchange, weakening=weakening, contraction=contraction
    )


def check_structure(presentation: RulePresentation, derivation: OpenDerivation) -> None:
    """Refuse a derivation that uses a structural rule the presentation lacks.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    derivation : OpenDerivation
        The derivation.

    Raises
    ------
    StructureError
        Naming each rule used and not admitted.
    """
    required = required_structure(presentation, derivation)
    missing = [
        name
        for name in ("exchange", "weakening", "contraction")
        if getattr(required, name) and not getattr(presentation.structure, name)
    ]
    if missing:
        raise StructureError(
            f"the derivation uses {', '.join(missing)}, which "
            f"{presentation.name!r} does not admit"
        )


def _static(value: Ground, sort: IndexSort) -> IndexTerm:
    """The static index term of a ground value.

    Parameters
    ----------
    value : Ground
        The value.
    sort : IndexSort
        Its sort.

    Returns
    -------
    IndexTerm
        A literal for a number or a nullary constructor, an application
        otherwise.

    Raises
    ------
    RuleError
        If the value does not fit the sort.
    """
    if isinstance(value, tuple):
        if not isinstance(sort, UserIndexSort) or not value:
            raise RuleError(f"{value!r} is not a constructor application")
        name = value[0]
        assert isinstance(name, str)
        # Constructor arguments of a closed sort range over the same sort.
        return IndexConstructor(
            name=name,
            arguments=tuple(_static(item, sort) for item in value[1:]),
            sort=sort,
        )
    return IndexLiteral(value=value, sort=sort)


def _item_type(presentation: RulePresentation, item: Item) -> TypeApplication:
    """The QIEC type of the proofs of an item.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    item : Item
        The item.

    Returns
    -------
    TypeApplication
        The family applied to the item's arguments as static terms.

    Raises
    ------
    RuleError
        If the item's form is unknown or its arguments do not fit.
    """
    form = presentation.judgments_by_id.get(item.family)
    if form is None or len(form.indices) != len(item.arguments):
        raise RuleError("an item does not fit its judgment form")
    arguments: list[IndexTerm] = []
    for binder, value in zip(form.indices, item.arguments, strict=True):
        assert isinstance(binder, IndexBinder)
        arguments.append(_static(value, binder.sort))
    return _judgment_type(
        Judgment(family=item.family, arguments=tuple(arguments)),
        presentation.judgments_by_id,
    )


def check_derivation(
    presentation: RulePresentation, derivation: OpenDerivation
) -> None:
    """Check a derivation with the QIEC checker.

    Each node becomes a QIEC constructor value of its rule, with the
    rule's telescope instantiated at the node's binding, its premises as
    fields, and its values as literals; each hypothesis becomes a local of
    its item's type. `quivers.qiec.checking.infer_value` then checks that
    every field has the type the rule gives it and that the node has its
    item's type. A premise with hypotheses is checked as its own value in
    the context extended by them, and stands for a local of its type in
    its parent, since QIEC has no function values.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    derivation : OpenDerivation
        The derivation.

    Raises
    ------
    RuleError
        If a node names another presentation's rule, a hole assumes an
        item other than its context's, or the checker refuses it.
    """
    hypotheses: list[Local] = []

    def local(item: Item) -> Local:
        """A fresh local of an item's proof type.

        Parameters
        ----------
        item : Item
            The item.

        Returns
        -------
        Local
            The local, named by its number, and recorded.
        """
        made = Local(
            name=f"hypothesis{len(hypotheses)}", type=_item_type(presentation, item)
        )
        hypotheses.append(made)
        return made

    context_locals = tuple(local(item) for item in derivation.context)
    pending: list[tuple[ProofNode, tuple[Local, ...], Item]] = [
        (derivation.body, context_locals, derivation.item)
    ]
    while pending:
        body, scope, item = pending.pop()
        placeholders: list[Local] = []
        value = _lower_value(
            presentation, body, scope, pending, local, hypotheses, placeholders
        )
        _check_lowered_proof(presentation, value, item, (*scope, *placeholders))


def closed_derivation_value(
    presentation: RulePresentation, derivation: Derivation
) -> ConstructorValue:
    """Give a closed derivation's checked proof value to an ordinary QIEC call.

    The value has exactly the source rule constructors, index arguments,
    ordered premises, and data fields. Axiom occurrence identities and
    weights remain in the derivation or forest that supplied the value.
    Encoding several derivations does not combine their weights.

    Parameters
    ----------
    presentation
        The unconditional rule presentation containing the proof declarations.
    derivation
        A closed proof in that presentation.

    Returns
    -------
    ConstructorValue
        A value of the proved judgment's exact indexed family type. Its
        consuming module needs the declarations of the same presentation.

    Raises
    ------
    RuleError
        If the presentation requires side conditions or proof equations,
        the proof uses holes or scoped premises, or a proof field is ill typed.
    """
    if presentation.conditions or presentation.equations:
        raise RuleError(
            "closed proof values require unconditional syntax without equations"
        )

    frames: list[
        tuple[
            Derivation,
            Rule | AxiomSlot,
            tuple[IndexTerm, ...],
            tuple[Blocks, ...],
            list[Value],
        ]
    ] = []
    node: ProofNode = derivation
    while True:
        if isinstance(node, Hole):
            raise RuleError(f"hole {node.index} lies outside its context")
        rule, _, static = _proof_arguments(presentation, node)
        frames.append((node, rule, static, _hypotheses(presentation, node), []))
        while frames:
            current, rule, static, blocks, fields = frames[-1]
            position = len(fields)
            if position < len(current.premises):
                if blocks and any(blocks[position]):
                    raise RuleError(
                        "a closed proof value cannot encode a premise under hypotheses"
                    )
                node = current.premises[position]
                break
            value = _proof_constructor(presentation, current, rule, static, fields)
            frames.pop()
            if not frames:
                _check_lowered_proof(presentation, value, derivation.item, ())
                return value
            frames[-1][-1].append(value)


def _check_lowered_proof(
    presentation: RulePresentation, value: Value, item: Item, locals_: tuple[Local, ...]
) -> None:
    """Check the ordinary constructor value against the claimed judgment.

    Parameters
    ----------
    presentation
        The rule declarations against which to check the value.
    value
        The proof or hypothesis as a QIEC value.
    item
        The item's claimed judgment family and indices.
    locals_
        The hypotheses and checked placeholders in scope.

    Raises
    ------
    RuleError
        If the kernel refuses the value or it has a different judgment type.
    """
    context = CheckContext()
    for entry in locals_:
        context = context.extend(entry)
    try:
        found = infer_value(value, presentation.registry, context)
    except KernelError as error:
        raise RuleError(f"the QIEC checker refuses the derivation: {error}") from error
    if found != _item_type(presentation, item):
        raise RuleError("a derivation does not have its item's type")


def _lower_value(
    presentation: RulePresentation,
    body: ProofNode,
    scope: tuple[Local, ...],
    pending: list[tuple[ProofNode, tuple[Local, ...], Item]],
    local: Callable[[Item], Local],
    hypotheses: list[Local],
    placeholders: list[Local],
) -> Value:
    """A derivation as a QIEC value in a scope of hypothesis locals.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    body : Derivation or Hole
        The derivation.
    scope : tuple of Local
        The locals of its context, outermost first.
    pending : list
        Scoped premises to check later, appended to.
    local : Callable[[Item], Local]
        Makes the local of a new hypothesis.
    hypotheses : list of Local
        Every local made so far, appended to.
    placeholders : list of Local
        The locals that stand for scoped premises, appended to; they are
        in scope of the value but not of its holes.

    Returns
    -------
    Value
        The QIEC value.

    Raises
    ------
    RuleError
        If a hole is out of scope or assumes another item.
    """
    if isinstance(body, Hole):
        if body.index >= len(scope):
            raise RuleError(f"hole {body.index} lies outside its context")
        entry = scope[len(scope) - 1 - body.index]
        if entry.type != _item_type(presentation, body.item):
            raise RuleError("a hole assumes another item than its context holds")
        return Var(local=entry)
    rule, binding, static = _proof_arguments(presentation, body)
    fields: list[Value] = []
    blocks = _hypotheses(presentation, body)
    for position, child in enumerate(body.premises):
        before, after = blocks[position] if blocks else ((), ())
        if not before and not after:
            fields.append(
                _lower_value(
                    presentation, child, scope, pending, local, hypotheses, placeholders
                )
            )
            continue
        premise = rule.premises[position] if isinstance(rule, Rule) else None
        assert premise is not None
        target = Item(
            family=premise.judgment.family,
            arguments=tuple(
                evaluate_index(a, binding) for a in premise.judgment.arguments
            ),
        )
        left = tuple(local(item) for item in before)
        right = tuple(local(item) for item in after)
        pending.append((child, extend_context(left, scope, right), target))
        placeholder = local(target)
        placeholders.append(placeholder)
        fields.append(Var(local=placeholder))
    return _proof_constructor(presentation, body, rule, static, fields)


def _proof_arguments(
    presentation: RulePresentation, body: Derivation
) -> tuple[Rule | AxiomSlot, dict[str, Ground], tuple[IndexTerm, ...]]:
    """Resolve the derivation's exact constructor and telescope arguments.

    Parameters
    ----------
    presentation
        The proof declarations in scope.
    body
        The derivation occurrence.

    Returns
    -------
    tuple
        The constructor declaration, ground binding, and typed arguments in
        telescope order.

    Raises
    ------
    RuleError
        If the constructor belongs to another presentation or a telescope
        variable is unbound or has the wrong sort.
    """
    rule = presentation.constructors_by_id.get(body.constructor)
    if rule is None:
        raise RuleError("a derivation names a rule of another presentation")
    binding = dict(body.binding)
    static: list[IndexTerm] = []
    for binder in rule.telescope:
        assert isinstance(binder, IndexBinder)
        if binder.name not in binding:
            raise RuleError(f"{rule.name!r} leaves {binder.name!r} unbound")
        static.append(_static(binding[binder.name], binder.sort))
    return rule, binding, tuple(static)


def _proof_constructor(
    presentation: RulePresentation,
    body: Derivation,
    rule: Rule | AxiomSlot,
    static: tuple[IndexTerm, ...],
    premises: list[Value],
) -> ConstructorValue:
    """Finish the proof constructor after its ordered premises are lowered.

    Parameters
    ----------
    presentation
        The proof declarations in scope.
    body
        The derivation occurrence.
    rule
        Its resolved rule or axiom slot.
    static
        Its typed telescope arguments.
    premises
        Its lowered premise values, in rule order.

    Returns
    -------
    ConstructorValue
        The ordinary constructor value with its claimed indexed result type.

    Raises
    ------
    RuleError
        If the occurrence carries the wrong number of data fields.
    """
    if len(body.values) != len(rule.values):
        raise RuleError(f"{rule.name!r} carries {len(rule.values)} values")
    return ConstructorValue(
        constructor=rule.id,
        static_arguments=static,
        fields=(
            *premises,
            *(
                LiteralValue(value=value, type=field.type)
                for value, field in zip(body.values, rule.values, strict=True)
            ),
        ),
        result_type=_item_type(presentation, body.item),
    )


def term_derivation(
    signature: Signature, term: Term, context: tuple[str, ...] = ()
) -> OpenDerivation:
    """The derivation of a structural term in its signature's presentation.

    A constructor or binder node is a rule application, its data and
    index arguments are values, and a bound variable is a hole.

    Parameters
    ----------
    signature : Signature
        The signature.
    term : Term
        The term.
    context : tuple of str
        The sorts of its free variables, outermost first.

    Returns
    -------
    OpenDerivation
        The derivation, in the context of the free variables.

    Raises
    ------
    RuleError
        If the term does not fit its signature.
    """
    rules = {rule.name: rule.id for rule in presentation_of(signature).rules}

    def item(sort: str) -> Item:
        """The item of an object sort.

        Parameters
        ----------
        sort : str
            The sort.

        Returns
        -------
        Item
            The item.
        """
        return Item(family=_sort_family(signature, sort))

    frames: list[
        tuple[
            Term, tuple[str, ...], str, list[ProofNode], list[str | int | float | bool]
        ]
    ] = []
    finished: ProofNode | None = None

    def deliver(node: ProofNode) -> None:
        """Hand a finished node to its parent.

        Parameters
        ----------
        node : Derivation or Hole
            The node.
        """
        nonlocal finished
        if frames:
            frames[-1][3].append(node)
        else:
            finished = node

    def enter(node: Term, scope: tuple[str, ...], sort: str) -> None:
        """Start a node, or finish a variable at once.

        Parameters
        ----------
        node : Term
            The node.
        scope : tuple of str
            The sorts of the variables in scope, outermost first.
        sort : str
            The sort the node must have.

        Raises
        ------
        RuleError
            If a variable is out of scope or of another sort.
        """
        if node.op == "BoundVar":
            index = node.args[0]
            assert isinstance(index, int)
            if index >= len(scope) or scope[len(scope) - 1 - index] != sort:
                raise RuleError(f"variable {index} is out of scope or of another sort")
            deliver(Hole(index=index, item=item(sort)))
            return
        if signature.codomain_of(node.op) != sort:
            raise RuleError(f"{node.op!r} does not produce sort {sort!r}")
        frames.append((node, scope, sort, [], []))

    enter(term, context, _sort_of_root(signature, term, context))
    while frames:
        node, scope, sort, premises, values = frames[-1]
        domain = signature.domain_of(node.op)
        position = len(premises) + len(values)
        if position == len(node.args):
            frames.pop()
            deliver(
                Derivation(
                    constructor=rules[node.op],
                    item=item(sort),
                    values=tuple(values),
                    premises=tuple(premises),
                )
            )
            continue
        argument = node.args[position]
        argument_sort = domain[position]
        inner = scope
        binder = signature.binders.get(node.op)
        if binder is not None:
            annotations = sum(1 for b in binder.binds if b.annot_sort is not None)
            if position >= annotations:
                inner = scope + tuple(b.sort for b in binder.binds)
        if signature.sorts[argument_sort].kind == "object":
            if not isinstance(argument, Term):
                raise RuleError(f"{node.op!r} takes a term at position {position}")
            enter(argument, inner, argument_sort)
        else:
            if isinstance(argument, Term) or isinstance(argument, bytes):
                raise RuleError(f"{node.op!r} takes a value at position {position}")
            values.append(argument)
    assert finished is not None
    return OpenDerivation(context=tuple(item(sort) for sort in context), body=finished)


def _sort_of_root(signature: Signature, term: Term, context: tuple[str, ...]) -> str:
    """The sort of a term's root.

    Parameters
    ----------
    signature : Signature
        The signature.
    term : Term
        The term.
    context : tuple of str
        The sorts of its free variables.

    Returns
    -------
    str
        The sort.

    Raises
    ------
    RuleError
        If a root variable is out of scope.
    """
    if term.op == "BoundVar":
        index = term.args[0]
        assert isinstance(index, int)
        if index >= len(context):
            raise RuleError(f"variable {index} is out of scope")
        return context[len(context) - 1 - index]
    return signature.codomain_of(term.op)


# ---------------------------------------------------------------------------
# interpretations


class InterpretationDecl(QiecModel):
    """A serializable interpretation: a QIEC computation per rule and slot.

    Parameters
    ----------
    id : InterpretationId
        Its identity.
    presentation : PresentationId
        The presentation it interprets.
    operations : tuple of tuple of ConstructorId and ComputationId
        The computation of each rule and slot.
    preserves : tuple of LawClaim
        The equations of the presentation it claims to preserve, each with
        its evidence.
    """

    id: InterpretationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    presentation: PresentationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    operations: tuple[tuple[ConstructorId, ComputationId], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    preserves: tuple[LawClaim, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


class Scoped[B](ValueModel):
    """The value of a premise derived under hypotheses.

    It is a function of the hypotheses' values, the premise's left block
    then its right block, each outermost first: the context-indexed
    reading of a premise. Calling it folds the premise
    with them bound.

    Parameters
    ----------
    apply : Callable[[tuple of B], B]
        The function.
    """

    apply: Callable[[tuple[B, ...]], B] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    def __call__[V](self: Scoped[V], hypotheses: tuple[V, ...]) -> V:
        """Fold the premise with values for its hypotheses.

        Parameters
        ----------
        hypotheses : tuple of V
            Their values, outermost first.

        Returns
        -------
        V
            The premise's value.
        """
        return self.apply(hypotheses)


type PremiseValue[B] = B | Scoped[B]
"""The value an operation receives for one premise."""

type Operation[B] = Callable[[Derivation, tuple[PremiseValue[B], ...]], B]
"""The value of a rule application from its derivation node and the
values of its premises, in premise order."""


class HostInterpretation[B](ValueModel):
    """An interpretation by host functions, one per rule and slot.

    A premise without hypotheses reaches its rule's operation as a value;
    a premise with hypotheses reaches it as a function from the values of
    the hypotheses, so a binder decides what its variables stand for.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    operations : Mapping[ConstructorId, Operation]
        The operation of each rule and axiom slot. An axiom's operation
        receives no premise values.
    """

    presentation: RulePresentation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    operations: Mapping[ConstructorId, Operation[B]] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require exactly one operation per rule and slot.

        Raises
        ------
        ValueError
            If one is missing or an extra one names nothing.
        """
        expected = set(self.presentation.constructors_by_id)
        if set(self.operations) != expected:
            raise ValueError("an interpretation has one operation per rule and slot")

    def fold[V](self: HostInterpretation[V], derivation: Derivation) -> V:
        """The value of a closed derivation: the unique map out of the initial algebra.

        Parameters
        ----------
        derivation : Derivation
            The derivation.

        Returns
        -------
        V
            Its value.
        """
        return self.fold_open(OpenDerivation(body=derivation), ())

    def fold_open[V](
        self: HostInterpretation[V],
        derivation: OpenDerivation,
        environment: tuple[V, ...],
    ) -> V:
        """The value of an open derivation at values of its hypotheses.

        The cost is one operation per node; the traversal uses an explicit
        stack, and a premise under hypotheses is folded when its rule's
        operation applies it.

        Parameters
        ----------
        derivation : OpenDerivation
            The derivation.
        environment : tuple of V
            A value per hypothesis of its context, outermost first.

        Returns
        -------
        V
            Its value.

        Raises
        ------
        RuleError
            If the environment does not match the context, or a hole lies
            outside it.
        """
        if len(environment) != len(derivation.context):
            raise RuleError("one value per hypothesis")
        return self._evaluate(derivation.body, environment)

    def _evaluate[V](
        self: HostInterpretation[V], body: ProofNode, environment: tuple[V, ...]
    ) -> V:
        """Fold a derivation in an environment.

        Parameters
        ----------
        body : Derivation or Hole
            The derivation.
        environment : tuple of V
            The values of its context, outermost first.

        Returns
        -------
        V
            Its value.

        Raises
        ------
        RuleError
            If a hole lies outside the environment.
        """
        if isinstance(body, Hole):
            return _lookup(environment, body.index)
        presentation = self.presentation
        frames: list[tuple[Derivation, tuple[Blocks, ...]]] = []
        built: list[list[PremiseValue[V]]] = []
        result: V | None = None
        frames.append((body, _hypotheses(presentation, body)))
        built.append([])
        while frames:
            node, widths = frames[-1]
            done = built[-1]
            if len(done) == len(node.premises):
                frames.pop()
                built.pop()
                value = self.operations[node.constructor](node, tuple(done))
                if built:
                    built[-1].append(value)
                else:
                    result = value
                continue
            child = node.premises[len(done)]
            before, after = _counts(widths, len(done))
            if before or after:
                done.append(_scoped(self, child, environment, before))
            elif isinstance(child, Hole):
                done.append(_lookup(environment, child.index))
            else:
                frames.append((child, _hypotheses(presentation, child)))
                built.append([])
        assert result is not None
        return result


def closed_values[V](values: tuple[PremiseValue[V], ...]) -> tuple[V, ...]:
    """The values of premises that have no hypotheses.

    Parameters
    ----------
    values : tuple of PremiseValue
        What an operation received.

    Returns
    -------
    tuple of V
        The same values.

    Raises
    ------
    RuleError
        If a premise is scoped under hypotheses.
    """
    found: list[V] = []
    for value in values:
        if isinstance(value, Scoped):
            raise RuleError("a premise under hypotheses is a function, not a value")
        found.append(value)
    return tuple(found)


def _lookup[V](environment: tuple[V, ...], index: int) -> V:
    """The value of a hypothesis by de Bruijn index.

    Parameters
    ----------
    environment : tuple of V
        The values, outermost first.
    index : int
        The index, from the innermost.

    Returns
    -------
    V
        The value.

    Raises
    ------
    RuleError
        If the index lies outside the environment.
    """
    if index >= len(environment):
        raise RuleError(f"hole {index} lies outside a context of {len(environment)}")
    return environment[len(environment) - 1 - index]


def _scoped[V](
    interpretation: HostInterpretation[V],
    child: ProofNode,
    environment: tuple[V, ...],
    before: int,
) -> Scoped[V]:
    """The value of a premise under hypotheses, as a function of theirs.

    Parameters
    ----------
    interpretation : HostInterpretation
        The interpretation.
    child : Derivation or Hole
        The premise's derivation.
    environment : tuple of V
        The values of the enclosing context.
    before : int
        How many of the hypotheses are on the left.

    Returns
    -------
    Scoped
        The function from the hypotheses' values, the left block then the
        right block, to the premise's value.
    """

    def apply(hypotheses: tuple[V, ...]) -> V:
        """Fold the premise with its hypotheses bound.

        Parameters
        ----------
        hypotheses : tuple of V
            Their values: the left block, then the right block, each
            outermost first.

        Returns
        -------
        V
            The premise's value.
        """
        context = extend_context(hypotheses[:before], environment, hypotheses[before:])
        return interpretation._evaluate(child, context)

    return Scoped(apply=apply)


class EquationFails(RuleError):
    """An interpretation gives the two sides of an equation different values."""


def check_equation[V](
    interpretation: HostInterpretation[V],
    equation: Equation,
    environments: Sequence[tuple[V, ...]],
    equal: Callable[[V, V], bool],
    *,
    subject: ClaimSubject,
    check: str,
    domain: str,
) -> LawClaim:
    """Test whether an interpretation preserves an equation.

    Both sides are folded at every environment. A test on finitely many
    environments can refute preservation and cannot establish it, so the
    claim it returns is tested, never proved.

    Parameters
    ----------
    interpretation : HostInterpretation
        The interpretation.
    equation : Equation
        The equation.
    environments : Sequence of tuple of V
        Values of the context to try.
    equal : Callable[[V, V], bool]
        Equality of values.
    subject : ClaimSubject
        The interpretation, as the claim names it.
    check : str
        The test that runs this.
    domain : str
        Where the environments come from.

    Returns
    -------
    LawClaim
        The tested claim that the interpretation preserves the equation's law.

    Raises
    ------
    EquationFails
        With the first environment at which the sides differ.
    ValueError
        If no environment is given.
    """
    if not environments:
        raise ValueError("a test of an equation needs an environment")
    for environment in environments:
        left = interpretation.fold_open(equation.left, environment)
        right = interpretation.fold_open(equation.right, environment)
        if not equal(left, right):
            raise EquationFails(
                f"{equation.law.name!r} fails at {environment!r}: {left!r} and {right!r}"
            )
    return LawClaim(
        law=equation.law,
        subject=subject,
        evidence=Tested(check=check, cases=len(environments), domain=domain),
    )


# ---------------------------------------------------------------------------
# the finite reference interpretation


class CyclicDerivations(RuleError):
    """A goal has infinitely many derivations and no height bound was given."""


class ReferenceInterpretation:
    """Every derivation of a small system, listed explicitly.

    This is an oracle: it grounds every rule against every item, lists
    every derivation of a goal, and folds each one. Its cost is
    exponential in general. Executors are planned from the presentation's
    analysis and checked against this.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    axioms : Sequence of AxiomOccurrence
        The input occurrences.
    item_limit : int
        The most items it may derive before refusing.
    """

    def __init__(
        self,
        presentation: RulePresentation,
        axioms: Sequence[AxiomOccurrence],
        *,
        item_limit: int = 10_000,
    ) -> None:
        self.presentation = presentation
        self.axioms = tuple(axioms)
        self.items = ItemTable()
        self.item_limit = item_limit
        self._leaves: dict[Item, list[Derivation]] = {}
        self._instances: dict[Item, list[tuple[Rule, Binding, tuple[Item, ...]]]] = {}
        if any(
            premise.hypotheses
            for rule in presentation.rules
            for premise in rule.premises
        ):
            raise RuleError(
                "the reference interpretation lists closed derivations; a premise "
                "with hypotheses is derived open, by an open derivation"
            )
        self._ground()

    def _axiom(self, occurrence: AxiomOccurrence) -> Derivation:
        """The derivation of one input occurrence.

        Parameters
        ----------
        occurrence : AxiomOccurrence
            The occurrence.

        Returns
        -------
        Derivation
            A leaf.

        Raises
        ------
        RuleError
            If the occurrence names no slot of the presentation.
        """
        slot = self.presentation.constructors_by_id.get(occurrence.slot)
        if not isinstance(slot, AxiomSlot):
            raise RuleError("an axiom occurrence names no slot of the presentation")
        binding = dict(occurrence.binding)
        arguments = tuple(
            evaluate_index(item, binding) for item in slot.judgment.arguments
        )
        item = self.items.intern(slot.judgment.family, arguments)
        return Derivation(
            constructor=slot.id,
            item=item,
            binding=occurrence.binding,
            values=occurrence.values,
            occurrence=occurrence.occurrence,
        )

    def _ground(self) -> None:
        """Derive every item and every grounded rule instance.

        Raises
        ------
        RuleError
            If more items than the limit are derived.
        """
        known: list[Item] = []
        seen: set[Item] = set()
        for occurrence in self.axioms:
            leaf = self._axiom(occurrence)
            self._leaves.setdefault(leaf.item, []).append(leaf)
            if leaf.item not in seen:
                seen.add(leaf.item)
                known.append(leaf.item)
        instances: set[tuple[ConstructorId, tuple[Item, ...]]] = set()
        changed = True
        while changed:
            changed = False
            by_family: dict[FamilyId, list[Item]] = {}
            for item in known:
                by_family.setdefault(item.family, []).append(item)
            for rule in self.presentation.rules:
                pools = [by_family.get(p.judgment.family, []) for p in rule.premises]
                for premises in itertools.product(*pools):
                    key = (rule.id, tuple(premises))
                    if key in instances:
                        continue
                    binding = self._bind(rule, premises)
                    if binding is None:
                        continue
                    instances.add(key)
                    arguments = tuple(
                        evaluate_index(item, binding)
                        for item in rule.conclusion.arguments
                    )
                    conclusion = self.items.intern(rule.conclusion.family, arguments)
                    self._instances.setdefault(conclusion, []).append(
                        (rule, binding, tuple(premises))
                    )
                    if conclusion not in seen:
                        seen.add(conclusion)
                        known.append(conclusion)
                        changed = True
                        if len(known) > self.item_limit:
                            raise RuleError(
                                f"the system derives more than {self.item_limit} items"
                            )

    def _bind(self, rule: Rule, premises: Sequence[Item]) -> Binding | None:
        """Match a rule's premises against items, in order, and check it.

        Parameters
        ----------
        rule : Rule
            The rule.
        premises : Sequence of Item
            One item per premise.

        Returns
        -------
        Binding or None
            The binding, or ``None`` when the items do not match or the
            side condition fails.
        """
        binding: Binding = {}
        for premise, item in zip(rule.premises, premises, strict=True):
            for pattern, value in zip(
                premise.judgment.arguments, item.arguments, strict=True
            ):
                if not match_index(pattern, value, binding):
                    return None
        if rule.side_condition is not None and not self._holds(rule, binding):
            return None
        return binding

    def _holds(self, rule: Rule, binding: Binding) -> bool:
        """Evaluate a rule's side condition by the QIEC evaluator.

        Parameters
        ----------
        rule : Rule
            The rule.
        binding : Binding
            The values of its variables.

        Returns
        -------
        bool
            Whether the condition holds.

        Raises
        ------
        RuleError
            If the condition reads a value field, which a rule instance
            found by grounding does not have.
        """
        condition = rule.side_condition
        assert condition is not None
        computation = next(
            item
            for item in self.presentation.conditions
            if item.id == condition.computation
        )
        environment = {}
        for name, parameter in zip(
            condition.arguments, computation.parameters, strict=True
        ):
            if name not in binding:
                raise RuleError(
                    f"the side condition of {rule.name!r} reads the value {name!r}, "
                    "which grounding does not give"
                )
            environment[parameter] = binding[name]
        return bool(Evaluator().evaluate(computation.body, environment))

    def item(self, family: FamilyId, arguments: tuple[Ground, ...]) -> Item:
        """The interned item of a family with arguments.

        Parameters
        ----------
        family : FamilyId
            The judgment form.
        arguments : tuple of Ground
            The arguments.

        Returns
        -------
        Item
            The item.
        """
        return self.items.intern(family, arguments)

    def observe[N: (Fraction, float)](
        self,
        goal: Item,
        weight: Callable[[Derivation], N],
        value: Callable[[Derivation], N],
        observer: Callable[[N], N],
        zero: N,
        *,
        height: int | None = None,
    ) -> N:
        """The weighted sum over derivations of an observer of their values.

        This is the observable ``sum_d w(d) h(fold(d))`` computed by its
        definition: the observer is applied to each derivation's value and
        never to an aggregate, so it may be nonlinear.

        Parameters
        ----------
        goal : Item
            The item.
        weight : Callable[[Derivation], N]
            The weight of one node of a derivation; a derivation's weight
            is the product over its nodes.
        value : Callable[[Derivation], N]
            The value of a whole derivation, usually a fold.
        observer : Callable[[N], N]
            Any function of one value.
        zero : N
            The empty sum.
        height : int or None
            The greatest height, as for `derivations`.

        Returns
        -------
        N
            The sum, ``zero`` when the item has no derivation.

        Raises
        ------
        RuleError
            If a derivation has an open hypothesis.
        """
        total = zero
        for derivation in self.derivations(goal, height=height):
            product = weight(derivation)
            stack: list[ProofNode] = list(derivation.premises)
            while stack:
                node = stack.pop()
                if isinstance(node, Hole):
                    raise RuleError("a reference derivation has no open hypothesis")
                product = product * weight(node)
                stack.extend(node.premises)
            total = total + product * observer(value(derivation))
        return total

    def derivations(self, goal: Item, *, height: int | None = None) -> list[Derivation]:
        """Every derivation of an item.

        Parameters
        ----------
        goal : Item
            The item.
        height : int or None
            The greatest height, a restriction of the model that is
            required when the item lies above a cycle.

        Returns
        -------
        list of Derivation
            Every derivation, each once.

        Raises
        ------
        CyclicDerivations
            If the item has infinitely many derivations and no height was
            given.
        """
        if height is None:
            self._refuse_cycles(goal)

        def trees(item: Item, budget: int | None) -> Iterator[Derivation]:
            """The derivations of one item within a height.

            Parameters
            ----------
            item : Item
                The item.
            budget : int or None
                The height left.

            Yields
            ------
            Derivation
                A derivation.
            """
            yield from self._leaves.get(item, ())
            if budget == 0:
                return
            lower = None if budget is None else budget - 1
            for rule, binding, premises in self._instances.get(item, ()):
                branches = [list(trees(premise, lower)) for premise in premises]
                for children in itertools.product(*branches):
                    yield Derivation(
                        constructor=rule.id,
                        item=item,
                        binding=tuple(sorted(binding.items())),
                        premises=tuple(children),
                    )

        return list(trees(goal, height))

    def _refuse_cycles(self, goal: Item) -> None:
        """Raise when an item below the goal derives itself.

        Parameters
        ----------
        goal : Item
            The goal.

        Raises
        ------
        CyclicDerivations
            If there is a cycle.
        """
        state: dict[Item, int] = {}
        stack: list[tuple[Item, Iterator[Item]]] = []

        def successors(item: Item) -> Iterator[Item]:
            """The premises of every instance concluding an item.

            Parameters
            ----------
            item : Item
                The item.

            Yields
            ------
            Item
                A premise.
            """
            for _, _, premises in self._instances.get(item, ()):
                yield from premises

        state[goal] = 1
        stack.append((goal, successors(goal)))
        while stack:
            item, rest = stack[-1]
            for child in rest:
                if state.get(child) == 1:
                    raise CyclicDerivations(
                        "the goal lies above a cycle; give a height"
                    )
                if child not in state:
                    state[child] = 1
                    stack.append((child, successors(child)))
                    break
            else:
                state[item] = 2
                stack.pop()


__all__ = [
    "AFFINE",
    "AxiomOccurrence",
    "AxiomSlot",
    "Blocks",
    "Binding",
    "CARTESIAN",
    "ComputedRuleWeight",
    "ContextStructure",
    "CyclicDerivations",
    "Derivation",
    "Equation",
    "EquationFails",
    "FamilyComponent",
    "Ground",
    "Hole",
    "HostInterpretation",
    "InterpretationDecl",
    "Item",
    "ItemTable",
    "Judgment",
    "JudgmentDecl",
    "LINEAR",
    "MatchError",
    "ORDERED",
    "OpenDerivation",
    "Operation",
    "PRESENTATION_PROTOCOL",
    "PRESENTATION_WORK",
    "Premise",
    "PremiseValue",
    "PresentationAnalysis",
    "ProofNode",
    "RELEVANT",
    "ReferenceInterpretation",
    "Rule",
    "RuleError",
    "RulePresentation",
    "Scoped",
    "SideCondition",
    "StructureError",
    "check_derivation",
    "closed_derivation_value",
    "closed_values",
    "check_equation",
    "check_structure",
    "compose_derivations",
    "evaluate_index",
    "extend_context",
    "identity_derivation",
    "match_index",
    "named_conditions",
    "presentation_of",
    "rename_hypotheses",
    "required_structure",
    "substitute_hypotheses",
    "term_derivation",
]
