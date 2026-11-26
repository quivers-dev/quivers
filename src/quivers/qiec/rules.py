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
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import cached_property
from types import MappingProxyType

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import (
    Acyclic,
    InterpretationId,
    LawClaim,
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
from quivers.qiec.identifiers import ComputationId, ConstructorId, FamilyId
from quivers.qiec.kinds import (
    IndexBinder,
    NatSort,
    Telescope,
    UserIndexSort,
    validate_telescope,
)
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.types import (
    BOOL,
    INT,
    STRING,
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

    Parameters
    ----------
    judgment : Judgment
        What the premise has to derive.
    hypotheses : tuple of Judgment
        Judgments the premise may use as open hypotheses: the context it
        is derived in is the rule's context extended by these, innermost
        last.
    """

    judgment: Judgment = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    hypotheses: tuple[Judgment, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


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
    weight : ParameterId or None
        The parameter that weights each application, or ``None`` for the
        unit weight.
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
    weight: ParameterId | None = dx.field(
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


class Item(ValueModel):
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


class Derivation(ValueModel):
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
    premises : tuple of Derivation
        Derivations of the premises, in order.
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
    premises: tuple[Derivation, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def size(self) -> int:
        """The number of nodes.

        Returns
        -------
        int
            The nodes of the tree, counted without recursion.
        """
        count = 0
        stack: list[Derivation] = [self]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.premises)
        return count


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
        if not term.arguments:
            return term.name
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
        if not term.arguments:
            return value == term.name
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
        if len(judgment.arguments) != len(form.indices):
            raise RuleError(
                f"{where} gives {len(judgment.arguments)} arguments to "
                f"{form.name!r}, which takes {len(form.indices)}"
            )

    for rule in presentation.rules:
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


type Operation[B] = Callable[[Derivation, tuple[B, ...]], B]
"""The value of a rule application from its derivation node and the
values of its premises, in premise order."""


class HostInterpretation[B](ValueModel):
    """An interpretation by host functions, one per rule and slot.

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

    def fold(self, derivation: Derivation) -> B:
        """The value of a derivation: the unique map out of the initial algebra.

        The cost is one operation per node; the traversal uses an explicit
        stack.

        Parameters
        ----------
        derivation : Derivation
            The derivation.

        Returns
        -------
        B
            Its value.
        """
        values: dict[int, B] = {}
        stack: list[tuple[Derivation, bool]] = [(derivation, False)]
        while stack:
            node, ready = stack.pop()
            if ready:
                children = tuple(values.pop(id(child)) for child in node.premises)
                values[id(node)] = self.operations[node.constructor](node, children)
                continue
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(node.premises))
        return values[id(derivation)]


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
    "AxiomOccurrence",
    "AxiomSlot",
    "Binding",
    "CyclicDerivations",
    "Derivation",
    "FamilyComponent",
    "Ground",
    "HostInterpretation",
    "InterpretationDecl",
    "Item",
    "ItemTable",
    "Judgment",
    "JudgmentDecl",
    "MatchError",
    "Operation",
    "PRESENTATION_PROTOCOL",
    "Premise",
    "PresentationAnalysis",
    "ReferenceInterpretation",
    "Rule",
    "RuleError",
    "RulePresentation",
    "SideCondition",
    "evaluate_index",
    "match_index",
    "named_conditions",
]
