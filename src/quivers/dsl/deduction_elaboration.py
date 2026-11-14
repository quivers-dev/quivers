"""Elaborate ``deduction`` declarations into kernel computations.

A weighted deductive system is a closed data family of items, a
computation enumerating its derivations, and a pair of handlers
aggregating them. The family declares one nullary constructor per atom
and one constructor per applied symbol the rules and lexicon mention,
with a slot typed ``Int`` where the system uses it as a position and
``Item`` otherwise. Rules become branches of a recursive computation
that performs ``choose`` on the deduction's ``Search`` instance to pick
an axiom or a rule, derives each premise by recursion, matches the
derived item against the premise pattern by ``case`` analysis (a shared
pattern variable is checked by the family's structural equality
computation), instantiates the conclusion, and adds the rule's weight
on the deduction's ``Weight[K]`` instance, where ``K`` is the carrier of
the declared semiring. A failed match performs an empty choice, which
the search handler answers by resuming nobody.

The entry computation handles the ``Weight`` instance with a collecting
handler and the ``Search`` instance with a handler that resumes once per
alternative and combines the shots by the semiring's addition, so the
answer is the semiring sum over derivations of the goal of the product
of the weights along each: the inside weight of the goal. Learned
weights are read through the module's ``params`` instance of ``Param``,
under names the classic engine keys its parameters by, so a program
that calls the deduction and scores its answer leaves ``params`` in its
row for the run to serve.

A system whose rules all relate ``span`` items enumerates by span: the
derivation of ``span(i, j, ...)`` chooses split positions for the
premise positions the conclusion leaves free and derives each premise
over its own smaller span, so the search is bounded by the sentence.
Unary rules, which keep the span, extend a derived item in chains; when
a chain could revisit an item, the chain carries the items it visited,
a bound limits how often it may revisit one, and the entry raises the
bound until the goal's weight settles within the declared
``tolerance``, as the chart's Kleene-star aggregation settles. The
``depth`` option bounds the constructor depth of every rule's
conclusion category, as the agenda engine's side condition does, and
never the number of rules a derivation applies. A system with a lexicon whose items are not all
spans, such as a multiple context-free grammar, derives an item from its
head symbol and the positions of its yield, read in pairs as the
components the item covers: its rules must be linear, each conclusion
component a concatenation of premise components, so a rule with several
premises derives each over fewer tokens, and its unary rules must form
no cycle. Any other system takes its axioms and their weights as input and
iterates a finite weighted chart. Each round adds the axioms to the rule
contributions from the preceding chart, until every item weight converges.
Constructor depth does not impose a proof-height limit.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import TYPE_CHECKING, Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.qiec._models import QiecModel

from quivers.dsl._deduction_chart import declare_chart, elaborate_chart

from quivers.dsl.ast_nodes import (
    BundleDecl,
    DeductionDecl,
    DefineDecl,
    ExprParser,
    LetExprCall,
    LetExprLiteral,
    LetExprNode,
    LetExprTuple,
    LetExprVar,
    LexiconEntry,
    LexiconCategoryFixed,
    LexiconCategoryRestricted,
    LexiconCategoryWildcard,
    ObjectDecl,
    ObjectEffectApply,
    ObjectExpr,
    ObjectProduct,
    ObjectSlash,
    OptionEntry,
    OptionFlag,
    OptionName,
    OptionNumber,
    ProgramDecl,
    SchemaDecl,
    SequentRule,
    TypeEnumSet,
    TypeFreeResiduated,
    TypeName,
)
from quivers.dsl.compiler._options import (
    find_option,
    get_option_flag,
    get_option_float,
    get_option_name,
)
from quivers.dsl.compiler._deduction_positions import (
    is_pattern_variable,
    misplaced_positions,
    position_slots,
)
from quivers.dsl.compiler._deduction_productions import (
    normalization_mode,
    production_groups,
)
from quivers.dsl.compiler.deductions import load_lexicon_tsv
from quivers.stochastic.categories import (
    AtomicCategory,
    Category,
    CategorySystem,
    ModalCategory,
    ProductCategory,
    SlashCategory,
    UnitCategory,
)
from quivers.stochastic.productions import Production, ProductionGroup, ProductionKey
from quivers.stochastic.schema import (
    SCHEMA_REGISTRY,
    PatternBinarySchema,
    PatternUnarySchema,
    RuleSchema,
)
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl, FieldDef
from quivers.qiec.effects import (
    ArgumentDef,
    EffectDef,
    EffectRef,
    EffectRequest,
    EffectRow,
    HandlerClauseDef,
    HandlerDef,
    OperationDef,
    ResumptionGrade,
    SiteProvenance,
    instantiate_effect,
)
from quivers.qiec.identifiers import (
    ComputationId,
    ConstructorId,
    EffectId,
    FamilyId,
    HandlerId,
    OperationId,
    StaticScopeId,
)
from quivers.qiec.kinds import NAT
from quivers.qiec.module import NamedComputation, NamedEffectInstance
from quivers.qiec.checking import ComputationSignature
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    CaseBranch,
    CaseMotive,
    Computation,
    Comprehension,
    ConstructorValue,
    Gather,
    Handle,
    If,
    LiteralValue,
    Local,
    Perform,
    Reduction,
    Return,
    TensorValue,
    Value,
    Var,
)
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    UNIT,
    IndexBinder,
    IndexLiteral,
    IndexVariable,
    TypeApplication,
    TypeBinder,
    TypeExpr,
    TypeVariable,
    product_type,
)

if TYPE_CHECKING:
    from quivers.dsl.qiec_lowering import _Elaborator

#: The module effect a deduction's search performs on: a choice among
#: the entries of a tensor, resumed once per entry by the search handler.
SEARCH_EFFECT = "Search"
#: The operation of that effect.
SEARCH_CHOOSE = "choose"
#: The module instance of ``Param`` every deduction reads its learned
#: weights through.
PARAMS_INSTANCE = "params"
#: The semirings a deduction may declare, each with the kernel type of
#: its carrier.
SEMIRING_CARRIERS: dict[str, TypeExpr] = {
    "LogProb": LOG_WEIGHT,
    "Viterbi": LOG_WEIGHT,
    "Boolean": BOOL,
    "Counting": INT,
}
#: The constructor name of the ``span`` items a lexicon injects.
SPAN = "span"
#: The computations of a deduction that read production weights, which
#: its weighted variant elaborates again over a table argument.
THREADED: tuple[str, ...] = ("axiom", "derive", "close", "inside", "converge", "run")
#: The constructor holding a canonically numbered bound variable of a
#: lexicon's logical forms.
BOUND = "bound"

type Sort = Literal["int", "item"]


class _Atom(dx.Model):
    """A declared atom in a pattern.

    Parameters
    ----------
    name
        The atom's name.
    """

    name: str


class _Wild(dx.Model):
    """A pattern variable.

    Parameters
    ----------
    name
        The variable's name.
    """

    name: str


class _Lit(dx.Model):
    """An integer literal in a pattern.

    Parameters
    ----------
    value
        The integer.
    """

    value: int


class _Bound(dx.Model):
    """A bound variable of a logical form, canonically numbered.

    Parameters
    ----------
    index
        The variable's number within its term, from one.
    """

    index: int


class _App(ValueModel):
    """An applied constructor symbol in a pattern.

    Parameters
    ----------
    symbol
        The symbol.
    args
        The argument patterns.
    """

    symbol: str
    args: tuple[Pattern, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _check_arguments(self) -> None:
        """Require immutable, recursively typed pattern arguments.

        Raises
        ------
        TypeError
            If the arguments are not a tuple of pattern records.
        """
        if not isinstance(self.args, tuple) or not all(
            isinstance(arg, (_Atom, _Wild, _Lit, _Bound, _App)) for arg in self.args
        ):
            raise TypeError("pattern arguments must be a tuple of pattern records")


type Pattern = _Atom | _Wild | _Lit | _Bound | _App


class _Rule(QiecModel):
    """One rule of a deduction, in pattern form.

    Parameters
    ----------
    source
        The declaration.
    premises
        The premise patterns.
    conclusion
        The conclusion pattern.
    learnable
        Whether the rule reads a learned weight keyed by its bindings.
    bounded
        Whether that weight is bounded below one per firing.
    parent
        The rule whose weight on the same bindings is added, if any.
    """

    source: SequentRule = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    premises: tuple[Pattern, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    conclusion: Pattern = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    learnable: bool
    bounded: bool
    parent: str | None


class _Entry(QiecModel):
    """One expanded lexicon entry.

    Parameters
    ----------
    word
        The token the entry fires on.
    category
        The category pattern, ground.
    form
        The logical form pattern, ground, or ``None`` when the system
        carries no forms.
    learnable
        Whether the entry reads a learned weight.
    """

    word: str
    category: Pattern = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    form: Pattern | None = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    learnable: bool


class _Group(QiecModel):
    """The rules whose premises share one shape.

    Parameters
    ----------
    arity
        The number of premises.
    spans
        For a span-shaped system, the span each premise covers as a
        pair of normalized position names.
    splits
        The normalized position names beyond the conclusion's, which a
        derivation chooses.
    rules
        The rules with their indices.
    names
        For each rule index, its position variables' normalized names.
    """

    arity: int
    spans: tuple[tuple[str, str], ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    splits: tuple[str, ...]
    rules: list[tuple[int, _Rule]] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    names: dict[int, dict[str, str]] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


class _System(QiecModel):
    """Everything the elaboration reads off one deduction.

    Parameters
    ----------
    decl
        The declaration.
    semiring
        The declared semiring's name.
    start
        The start symbol, if any.
    depth_cell
        Owned one-element cell.
        The declared derivation depth bound, if any.
    rules
        The rules in pattern form.
    entries
        The expanded lexicon, empty for a system that takes its axioms
        as input.
    uses_forms_cell
        Owned one-element cell.
        Whether span items carry a logical form.
    symbols
        Every constructor symbol with its slot sorts, atoms included as
        symbols of no slots.
    bare
        The atoms used as items on their own, which cannot also be
        applied.
    span_shaped_cell
        Owned one-element cell.
        Whether every rule relates span items, so derivations may be
        enumerated by span.
    span_fuel_factor_cell
        Owned one-element cell.
        For a schema parser, the units of derivation fuel per sentence
        position.
    schema_categories_cell
        Owned one-element cell.
        For a schema parser, its finite category algebra.
    schema_words_cell
        Owned one-element cell.
        For a schema parser, its vocabulary.
    tolerance_cell
        Owned one-element cell.
        The declared convergence tolerance, zero without one.
    normalized_cell
        Owned one-element cell.
        Whether the weights of the productions sharing a left-hand side
        are normalized (``normalize=lhs``).
    groups_cell
        Owned one-element cell.
        For a normalized system, its production groups.
    table
        For a normalized system, the position of each production of a
        group of several in the table a program supplies.
    handlers_cell
        Owned one-element cell.
        The search and collection handlers, once declared.
    heads
        For a yield-directed system, the head symbols of its items, in
        the order its derivation numbers them.
    live
        For a yield-directed system, the indices of the rules that can
        fire: those without a premise that has no positions.
    closure
        The indices of the unary rules, which a span-shaped system
        applies in chains over one derived item.
    cyclic_cell
        Owned one-element cell.
        Whether a chain of unary rules may revisit an item, so a
        span-shaped search bounds how often a chain repeats an item and
        iterates that bound to convergence.
    family_cell
        Owned one-element cell.
        The item family, once declared.
    constructors
        The item family's constructors, by symbol.
    chain_family_cell
        Owned one-element cell.
        For a cyclic system, the family of the item chains the unary
        closure carries.
    chain_constructors
        The chain family's ``nil`` and ``cons`` constructors.
    choice_cell
        Owned one-element cell.
        The ``Search`` instance.
    weight_cell
        Owned one-element cell.
        The ``Weight`` instance.
    fresh_cell
        Owned one-element cell.
        The count of fresh locals minted so far.
    """

    decl: DeductionDecl = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    semiring: str = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    start: str | None = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    depth_cell: list[int | None] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    rules: list[_Rule] = dx.field(
        default_factory=list, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    entries: list[_Entry] = dx.field(
        default_factory=list, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    uses_forms_cell: list[bool] = dx.field(
        default_factory=lambda: [False], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    symbols: dict[str, list[Sort]] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    bare: set[str] = dx.field(
        default_factory=set, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    span_shaped_cell: list[bool] = dx.field(
        default_factory=lambda: [False], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    span_fuel_factor_cell: list[int] = dx.field(
        default_factory=lambda: [0], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    schema_categories_cell: list[tuple[Pattern, ...]] = dx.field(
        default_factory=lambda: [()], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    schema_words_cell: list[tuple[str, ...]] = dx.field(
        default_factory=lambda: [()], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    tolerance_cell: list[float] = dx.field(
        default_factory=lambda: [0.0], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    normalized_cell: list[bool] = dx.field(
        default_factory=lambda: [False], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    groups_cell: list[tuple[ProductionGroup, ...]] = dx.field(
        default_factory=lambda: [()], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    table: dict[ProductionKey, int] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    handlers_cell: list[tuple[HandlerDef, HandlerDef] | None] = dx.field(
        default_factory=lambda: [None], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    heads: list[str] = dx.field(
        default_factory=list, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    live: list[int] = dx.field(
        default_factory=list, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    closure: list[int] = dx.field(
        default_factory=list, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    cyclic_cell: list[bool] = dx.field(
        default_factory=lambda: [False], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    family_cell: list[FamilyDecl | None] = dx.field(
        default_factory=lambda: [None], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    constructors: dict[str, ConstructorDecl] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    chain_family_cell: list[FamilyDecl | None] = dx.field(
        default_factory=lambda: [None], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    chain_constructors: dict[str, ConstructorDecl] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    choice_cell: list[NamedEffectInstance | None] = dx.field(
        default_factory=lambda: [None], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    weight_cell: list[NamedEffectInstance | None] = dx.field(
        default_factory=lambda: [None], opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    fresh_cell: list[int] = dx.field(
        default_factory=lambda: [0], opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @property
    def by_yield(self) -> bool:
        """Whether items are derived from their heads and yield positions.

        Returns
        -------
        bool
            ``True`` for a system with a lexicon whose rules do not all
            relate spans, such as a multiple context-free grammar.
        """
        return bool(self.entries) and not self.span_shaped_cell[0]

    @property
    def closes(self) -> bool:
        """Whether derivations apply unary rules as chains over a derived item.

        Returns
        -------
        bool
            ``True`` for a span-shaped lexical system with a unary rule.
        """
        return (
            self.span_shaped_cell[0]
            and not self.schema_categories_cell[0]
            and bool(self.closure)
        )

    @property
    def chained(self) -> bool:
        """Whether the unary closure carries the chain of items it visited.

        Returns
        -------
        bool
            ``True`` when the closure may revisit an item.
        """
        return self.closes and self.cyclic_cell[0]

    @property
    def chain(self) -> TypeApplication:
        """The chain family applied.

        Returns
        -------
        TypeApplication
            The type of the item chains a cyclic closure carries.
        """
        assert self.chain_family_cell[0] is not None
        return TypeApplication(
            constructor=self.chain_family_cell[0].type_constructor, arguments=()
        )

    @property
    def carrier(self) -> TypeExpr:
        """The kernel type of the semiring's weights.

        Returns
        -------
        TypeExpr
            ``LogWeight`` for the log semirings, ``Bool`` for the Boolean
            one, ``Int`` for the counting one.
        """
        return SEMIRING_CARRIERS[self.semiring]

    @property
    def item(self) -> TypeApplication:
        """The item family applied.

        Returns
        -------
        TypeApplication
            The type of every item.
        """
        assert self.family_cell[0] is not None
        return TypeApplication(
            constructor=self.family_cell[0].type_constructor, arguments=()
        )

    def name(self, suffix: str) -> str:
        """A module-level name derived from the deduction's.

        Parameters
        ----------
        suffix : str
            What the name is for.

        Returns
        -------
        str
            ``<deduction>__<suffix>``.
        """
        return f"{self.decl.name}__{suffix}"

    def local(self, hint: str, type_: TypeExpr) -> Local:
        """A fresh local.

        Parameters
        ----------
        hint : str
            The name's stem.
        type_ : TypeExpr
            The local's type.

        Returns
        -------
        Local
            A local named uniquely within the deduction's computations.
        """
        self.fresh_cell[0] += 1
        return Local(name=f"{hint}_{self.fresh_cell[0]}", type=type_)


def _int(value: int) -> LiteralValue:
    """An ``Int`` literal.

    Parameters
    ----------
    value : int
        The integer.

    Returns
    -------
    LiteralValue
        The literal.
    """
    return LiteralValue(value=value, type=INT)


def _string(value: str) -> LiteralValue:
    """A ``String`` literal.

    Parameters
    ----------
    value : str
        The text.

    Returns
    -------
    LiteralValue
        The literal.
    """
    return LiteralValue(value=value, type=STRING)


def _pattern_variables(pattern: Pattern) -> list[str]:
    """The variables of a pattern in order of first occurrence.

    Parameters
    ----------
    pattern : Pattern
        The pattern.

    Returns
    -------
    list[str]
        The variable names, each once.
    """
    found: list[str] = []

    def visit(node: Pattern) -> None:
        """Collect one pattern's variables.

        Parameters
        ----------
        node : Pattern
            The pattern.
        """
        if isinstance(node, _Wild):
            if node.name not in found:
                found.append(node.name)
        elif isinstance(node, _App):
            for argument in node.args:
                visit(argument)

    visit(pattern)
    return found


def _is_span(pattern: Pattern) -> bool:
    """Whether a pattern is a span with variable positions.

    Parameters
    ----------
    pattern : Pattern
        The pattern.

    Returns
    -------
    bool
        ``True`` for ``span(I, J, ...)`` with ``I`` and ``J`` variables.
    """
    return (
        isinstance(pattern, _App)
        and pattern.symbol == SPAN
        and len(pattern.args) >= 3
        and isinstance(pattern.args[0], _Wild)
        and isinstance(pattern.args[1], _Wild)
    )


def _rule_positions(system: _System, rule: _Rule) -> list[str]:
    """The variables a rule uses in integer slots, in first-occurrence order.

    Parameters
    ----------
    system : _System
        The system, whose slot sorts are inferred.
    rule : _Rule
        The rule.

    Returns
    -------
    list[str]
        The names.
    """
    found: list[str] = []

    def walk(pattern: Pattern) -> None:
        """Collect the integer-slot variables of one pattern.

        Parameters
        ----------
        pattern : Pattern
            The pattern.
        """
        if not isinstance(pattern, _App):
            return
        sorts = system.symbols[pattern.symbol]
        for position, argument in enumerate(pattern.args):
            if isinstance(argument, _Wild):
                if sorts[position] == "int" and argument.name not in found:
                    found.append(argument.name)
            else:
                walk(argument)

    for pattern in (*rule.premises, rule.conclusion):
        walk(pattern)
    return found


#: A position in a rule's yield: a position variable's name or a literal.
type _Node = str | int
#: How a rule's conclusion yield is built from its premises' components:
#: for each conclusion component, the positions from its start to its end
#: through the premise components it concatenates; and the position
#: variables only an empty premise component mentions.
type _YieldPlan = tuple[tuple[tuple[_Node, ...], ...], tuple[str, ...]]


def _components(system: _System, pattern: Pattern) -> list[tuple[_Node, _Node]] | str:
    """The yield components of an item pattern: its position slots in pairs.

    Parameters
    ----------
    system : _System
        The system, whose slot sorts are inferred.
    pattern : Pattern
        A rule's premise or conclusion.

    Returns
    -------
    list[tuple[_Node, _Node]] | str
        Each component's start and end, or the reason the pattern has no
        such components.
    """
    if not isinstance(pattern, _App):
        return "is not an item with a head symbol and positions"
    slots = [
        index
        for index, sort in enumerate(system.symbols[pattern.symbol])
        if sort == "int"
    ]
    if not slots or len(slots) % 2:
        return (
            f"is headed by {pattern.symbol!r}, whose {len(slots)} position slots "
            "do not pair into components"
        )
    nodes: list[_Node] = []
    for slot in slots:
        argument = pattern.args[slot]
        if isinstance(argument, _Wild):
            nodes.append(argument.name)
        elif isinstance(argument, _Lit):
            nodes.append(argument.value)
        else:
            return f"fills a position slot of {pattern.symbol!r} with a term"
    return [(nodes[index], nodes[index + 1]) for index in range(0, len(nodes), 2)]


def _has_positions(system: _System, pattern: Pattern) -> bool:
    """Whether an item pattern has a position slot.

    Parameters
    ----------
    system : _System
        The system, whose slot sorts are inferred.
    pattern : Pattern
        A rule's premise or conclusion.

    Returns
    -------
    bool
        ``True`` for an applied symbol with an ``Int`` slot.
    """
    return isinstance(pattern, _App) and "int" in system.symbols[pattern.symbol]


def _head(pattern: Pattern) -> str:
    """The head symbol of an item pattern.

    Parameters
    ----------
    pattern : Pattern
        An applied symbol or an atom.

    Returns
    -------
    str
        Its symbol or the atom's name.
    """
    if isinstance(pattern, _App):
        return pattern.symbol
    assert isinstance(pattern, _Atom)
    return pattern.name


def _yield_plan(system: _System, rule: _Rule) -> _YieldPlan | str:
    """Read how a rule concatenates its premises' components into its conclusion's.

    A rule is linear when each component of its conclusion is the
    concatenation of premise components, start to end, and every
    non-empty premise component is used exactly once: the multiple
    context-free reading, under which an item's yield is the tokens its
    components cover and a rule rearranges its premises' yields without
    copying or dropping any.

    Parameters
    ----------
    system : _System
        The system, whose slot sorts are inferred.
    rule : _Rule
        The rule.

    Returns
    -------
    _YieldPlan | str
        The plan, or why the rule is not linear.
    """
    conclusion = _components(system, rule.conclusion)
    if isinstance(conclusion, str):
        return f"its conclusion {conclusion}"
    edges: list[tuple[_Node, _Node]] = []
    mentioned: list[str] = []
    for position, premise in enumerate(rule.premises):
        components = _components(system, premise)
        if isinstance(components, str):
            return f"its premise {position + 1} {components}"
        for start, end in components:
            for node in (start, end):
                if isinstance(node, str) and node not in mentioned:
                    mentioned.append(node)
            if start != end:
                edges.append((start, end))
    leaving: dict[_Node, list[int]] = {}
    for index, (start, _end) in enumerate(edges):
        leaving.setdefault(start, []).append(index)
    for node, indices in leaving.items():
        if len(indices) > 1:
            return f"two premise components start at {node!r}, so they overlap"
    used: set[int] = set()
    paths: list[tuple[_Node, ...]] = []
    for start, end in conclusion:
        path: list[_Node] = [start]
        current = start
        while current != end:
            following = [
                index for index in leaving.get(current, []) if index not in used
            ]
            if not following:
                return (
                    f"its conclusion's component ({start}, {end}) is not a "
                    "concatenation of premise components"
                )
            used.add(following[0])
            current = edges[following[0]][1]
            path.append(current)
        paths.append(tuple(path))
    if len(used) != len(edges):
        start, end = next(edge for index, edge in enumerate(edges) if index not in used)
        return (
            f"its premise component ({start}, {end}) is not part of the "
            "conclusion's yield"
        )
    on_paths = {node for path in paths for node in path}
    loose = tuple(name for name in mentioned if name not in on_paths)
    return tuple(paths), loose


def _unifiable(left: Pattern, right: Pattern) -> bool:
    """Whether two patterns, their variables apart, have a common instance.

    Parameters
    ----------
    left : Pattern
        A pattern whose variables are its own.
    right : Pattern
        A pattern whose variables are its own.

    Returns
    -------
    bool
        ``True`` when some substitution makes the two equal.
    """
    bindings: dict[tuple[int, str], tuple[int, Pattern]] = {}

    def resolve(side: int, pattern: Pattern) -> tuple[int, Pattern]:
        """Follow a variable's binding to its value.

        Parameters
        ----------
        side : int
            Which pattern the variable belongs to.
        pattern : Pattern
            The pattern.

        Returns
        -------
        tuple[int, Pattern]
            The side and pattern the variable stands for.
        """
        while isinstance(pattern, _Wild) and (side, pattern.name) in bindings:
            side, pattern = bindings[(side, pattern.name)]
        return side, pattern

    def unify(one: tuple[int, Pattern], two: tuple[int, Pattern]) -> bool:
        """Unify two sided patterns, extending the bindings.

        Parameters
        ----------
        one : tuple[int, Pattern]
            A side and its pattern.
        two : tuple[int, Pattern]
            A side and its pattern.

        Returns
        -------
        bool
            Whether they unify.
        """
        side_a, a = resolve(*one)
        side_b, b = resolve(*two)
        if isinstance(a, _Wild):
            if not (isinstance(b, _Wild) and (side_a, a.name) == (side_b, b.name)):
                bindings[(side_a, a.name)] = (side_b, b)
            return True
        if isinstance(b, _Wild):
            bindings[(side_b, b.name)] = (side_a, a)
            return True
        if isinstance(a, _App) and isinstance(b, _App):
            return (
                a.symbol == b.symbol
                and len(a.args) == len(b.args)
                and all(
                    unify((side_a, x), (side_b, y))
                    for x, y in zip(a.args, b.args, strict=True)
                )
            )
        return a == b

    return unify((0, left), (1, right))


def _may_cycle(system: _System) -> bool:
    """Whether a chain of the system's unary rules may revisit an item.

    One unary rule can follow another when the first's conclusion and
    the second's premise have a common instance; a chain revisits an
    item only if this follows-graph over the unary rules has a cycle.
    The test is conservative: a cycle in the graph is necessary for a
    revisit, not sufficient.

    Parameters
    ----------
    system : _System
        The system.

    Returns
    -------
    bool
        ``False`` only when no chain of unary rules can revisit an item.
    """
    unary = [system.rules[index] for index in system.closure]
    follows = {
        position: [
            target
            for target, after in enumerate(unary)
            if _unifiable(rule.conclusion, after.premises[0])
        ]
        for position, rule in enumerate(unary)
    }
    finished: set[int] = set()
    active: set[int] = set()

    def reaches_cycle(node: int) -> bool:
        """Whether a cycle is reachable from a rule.

        Parameters
        ----------
        node : int
            The rule's position among the unary rules.

        Returns
        -------
        bool
            ``True`` when the depth-first walk meets a rule on its path.
        """
        if node in active:
            return True
        if node in finished:
            return False
        active.add(node)
        found = any(reaches_cycle(target) for target in follows[node])
        active.discard(node)
        finished.add(node)
        return found

    return any(reaches_cycle(node) for node in follows)


def _weighted_deductions(statements: tuple[object, ...]) -> frozenset[str]:
    """The deductions a program parses with production probabilities.

    Parameters
    ----------
    statements : tuple[object, ...]
        The module's statements.

    Returns
    -------
    frozenset[str]
        The name of every deduction some program's
        ``let x = parse(D, input, ..., ("A", p), ...)`` names.
    """
    names: set[str] = set()
    for statement in statements:
        if not isinstance(statement, ProgramDecl):
            continue
        for step in statement.draws:
            value = getattr(step, "value", None)
            if (
                isinstance(value, LetExprCall)
                and value.func == "parse"
                and value.args
                and isinstance(value.args[0], LetExprVar)
                and any(isinstance(argument, LetExprTuple) for argument in value.args)
            ):
                names.add(value.args[0].name)
    return frozenset(names)


def _longest_unary_chain(
    unary_rules: tuple[tuple[int, ...], ...],
    n_categories: int,
) -> int | None:
    """The number of rules in the longest chain of unary rules.

    The reference machine enumerates a schema parser's derivation
    trees one by one, so every tree must be finite: the unary rules,
    each deriving one category from another over the same span, must
    not cycle. The classic chart sums a cycle's series in closed form,
    which tree enumeration cannot reproduce.

    Parameters
    ----------
    unary_rules : tuple[tuple[int, ...], ...]
        The unary rules as ``(result, premise)`` category indices.
    n_categories : int
        The number of categories.

    Returns
    -------
    int | None
        The longest chain's length in rules (``0`` without unary
        rules), or ``None`` when the unary rules cycle.
    """
    successors: list[list[int]] = [[] for _ in range(n_categories)]
    for result, premise in unary_rules:
        successors[premise].append(result)
    longest: dict[int, int] = {}
    # Depth-first search with an explicit stack: grey nodes are on
    # the current path, so meeting one again closes a cycle.
    grey: set[int] = set()
    for root in range(n_categories):
        if root in longest:
            continue
        stack: list[tuple[int, int]] = [(root, 0)]
        grey.add(root)
        while stack:
            node, position = stack[-1]
            if position < len(successors[node]):
                stack[-1] = (node, position + 1)
                successor = successors[node][position]
                if successor in grey:
                    return None
                if successor not in longest:
                    grey.add(successor)
                    stack.append((successor, 0))
                continue
            stack.pop()
            grey.discard(node)
            longest[node] = max(
                (1 + longest[successor] for successor in successors[node]),
                default=0,
            )
    return max(longest.values(), default=0)


class _DeductionElaboration:
    """Deduction elaboration, mixed into the QIEC elaborator.

    The declaration pass runs once the module's own families, effects,
    and instances are declared and registered, so a deduction's item
    family and instances join the registry; the computation pass runs
    after the module's handlers are registered and before programs are
    elaborated, so a program body can call a deduction's entry.
    """

    def _declare_deduction_declarations(self: _Elaborator) -> None:
        """Declare each deduction's item family, effect, and instances.

        Raises
        ------
        QiecDiagnosticError
            If a deduction declares an unknown semiring, an option
            outside the elaboration, or a rule the pattern language does
            not cover.
        """
        self._deductions: dict[str, _System] = {}
        declarations = [
            statement
            for statement in self.source.syntax.statements
            if isinstance(statement, DeductionDecl)
        ]
        self._schema_unary_chain: dict[str, int] = {}
        schema_parsers = self._schema_parser_declarations()
        declarations.extend(declaration for declaration, _ in schema_parsers)
        if not declarations:
            return
        self._declare_search_effect()
        self._program_instance(PARAMS_INSTANCE, "Param")
        for declaration in declarations:
            system = self._read_deduction(declaration)
            if any(item[0] is declaration for item in schema_parsers):
                # A schema parser's ``depth`` bounds its category algebra,
                # not its derivation. A root-to-leaf path of a tree over a
                # sentence of length L visits at most L spans, at most L - 1
                # binary rules, and at most one chain of unary rules per
                # span, so L times one more than the longest chain units of
                # fuel reach every tree.
                system.depth_cell[0] = None
                system.span_fuel_factor_cell[0] = (
                    1 + self._schema_unary_chain[declaration.name]
                )
                first_word = system.entries[0].word
                category_count = next(
                    (
                        index
                        for index, entry in enumerate(system.entries)
                        if entry.word != first_word
                    ),
                    len(system.entries),
                )
                system.schema_categories_cell[0] = tuple(
                    entry.category for entry in system.entries[:category_count]
                )
                system.schema_words_cell[0] = tuple(
                    dict.fromkeys(entry.word for entry in system.entries)
                )
            self._declare_item_family(system)
            if not system.entries:
                declare_chart(self, system)
            if system.chained:
                self._declare_chain_family(system)
            self._declare_deduction_instances(system)
            self._deductions[declaration.name] = system

    def _schema_parser_declarations(
        self: _Elaborator,
    ) -> list[tuple[DeductionDecl, DefineDecl]]:
        """Translate schema-backed parser definitions into deductions.

        Returns
        -------
        list[tuple[DeductionDecl, DefineDecl]]
            One synthetic deduction and its source definition for every
            ``define name = parser(...)`` over declared schemas.
        """
        statements = self.source.syntax.statements
        schemas = {
            statement.name: statement
            for statement in statements
            if isinstance(statement, SchemaDecl)
        }
        bundles = {
            statement.name: statement
            for statement in statements
            if isinstance(statement, BundleDecl)
        }
        objects = [
            statement for statement in statements if isinstance(statement, ObjectDecl)
        ]
        translated: list[tuple[DeductionDecl, DefineDecl]] = []
        for statement in statements:
            if not (
                isinstance(statement, DefineDecl)
                and isinstance(statement.expr, ExprParser)
            ):
                continue
            declaration = self._schema_parser_declaration(
                statement, statement.expr, schemas, bundles, objects
            )
            if declaration is not None:
                translated.append((declaration, statement))
        return translated

    def _schema_parser_declaration(
        self: _Elaborator,
        definition: DefineDecl,
        parser: ExprParser,
        schemas: dict[str, SchemaDecl],
        bundles: dict[str, BundleDecl],
        objects: list[ObjectDecl],
    ) -> DeductionDecl | None:
        """Build the deduction denoted by one schema parser definition.

        Returns
        -------
        DeductionDecl | None
            The deduction, or ``None`` for a parser whose unary rules
            cycle when the module is lowered without its programs.

        Raises
        ------
        QiecDiagnosticError
            If the parser's rules do not elaborate, with code
            ``qiec-program-gap`` for unary rules that cycle.
        """

        def expand(name: str, seen: frozenset[str]) -> list[str]:
            bundle = bundles.get(name)
            if bundle is None:
                return [name]
            if name in seen:
                self._fail(
                    definition,
                    f"bundle cycle through {name!r}",
                    code="qiec-program",
                )
            expanded: list[str] = []
            for member in bundle.rules:
                expanded.extend(expand(member, seen | {name}))
            return expanded

        rule_names: list[str] = []
        for name in parser.rules:
            rule_names.extend(expand(name, frozenset()))
        selected: list[tuple[SchemaDecl | None, RuleSchema | None]] = []
        for name in rule_names:
            schema = schemas.get(name)
            builtin = SCHEMA_REGISTRY.get(name)
            if schema is None and builtin is None:
                self._fail(
                    definition,
                    f"schema parser {definition.name!r} uses {name!r}, which is "
                    "neither a declared schema nor a built-in schema primitive; "
                    "morphism-backed parser rules require chart_fold and do not "
                    "have a finite schema projection",
                    code="qiec-program-gap",
                )
            selected.append((schema, builtin))
        if not selected:
            self._fail(
                definition,
                f"schema parser {definition.name!r} has no rules",
                code="qiec-program",
            )
        atoms = self._schema_parser_atoms(definition, parser, objects)
        categories = (
            CategorySystem.from_generators(
                list(atoms), list(parser.constructors), parser.depth
            )
            if parser.constructors is not None
            else CategorySystem.from_atoms_and_slash_depth(list(atoms), parser.depth)
        )
        category_patterns = tuple(
            self._category_object(category) for category in categories
        )
        words = self._schema_parser_terminals(definition, parser, objects)
        learnable = (OptionEntry(key="learnable", value=OptionFlag()),)
        schema_runtime = None
        for schema, builtin in selected:
            if builtin is not None:
                piece = builtin
                schema_runtime = (
                    piece if schema_runtime is None else schema_runtime | piece
                )
                continue
            assert schema is not None
            variables = frozenset(
                name for parameter in schema.parameters for name in parameter.names
            )
            if not (
                isinstance(schema.domain, ObjectProduct)
                and len(schema.domain.components) == 2
            ):
                piece = PatternUnarySchema(
                    schema.domain, schema.codomain, variables, schema.name
                )
            else:
                left, right = schema.domain.components
                piece = PatternBinarySchema(
                    left,
                    right,
                    schema.codomain,
                    variables,
                    schema.name,
                )
            schema_runtime = piece if schema_runtime is None else schema_runtime | piece
        assert schema_runtime is not None
        rule_system = schema_runtime(categories)
        rules: list[SequentRule] = []
        for index, (result, left, right) in enumerate(rule_system.binary_rules):
            rules.append(
                SequentRule(
                    name=f"binary_{index}",
                    premises=(
                        self._span("I", "K", category_patterns[left]),
                        self._span("K", "J", category_patterns[right]),
                    ),
                    conclusion=self._span("I", "J", category_patterns[result]),
                    options=learnable,
                    line=definition.line,
                    col=definition.col,
                )
            )
        for index, (result, premise) in enumerate(rule_system.unary_rules):
            rules.append(
                SequentRule(
                    name=f"unary_{index}",
                    premises=(self._span("I", "J", category_patterns[premise]),),
                    conclusion=self._span("I", "J", category_patterns[result]),
                    options=learnable,
                    line=definition.line,
                    col=definition.col,
                )
            )
        chain = _longest_unary_chain(rule_system.unary_rules, len(category_patterns))
        if chain is None:
            if not self.source.elaborate_programs:
                # Lowered again after the gap below: the module keeps its
                # other declarations and records the gap.
                return None
            self._fail(
                definition,
                f"schema parser {definition.name!r} has unary rules that "
                "cycle; its derivation trees are unbounded, and the reference "
                "machine, which enumerates them, has no elaboration of the "
                "closed-form sum the chart parser computes",
                code="qiec-program-gap",
            )
        self._schema_unary_chain[definition.name] = chain
        lexicon = tuple(
            LexiconEntry(
                words=(word,),
                category=LexiconCategoryFixed(category=category),
                lf=LetExprLiteral(value=0.0),
                options=learnable,
                line=definition.line,
                col=definition.col,
            )
            for word in words
            for category in category_patterns
        )
        start = parser.start if isinstance(parser.start, str) else "S"
        options = (
            OptionEntry(key="semiring", value=OptionName(value="LogProb")),
            OptionEntry(key="start", value=OptionName(value=start)),
            # `_read_deduction` requires a bound for unary rules. It is
            # replaced by dynamic span fuel after validation.
            OptionEntry(key="depth", value=OptionNumber(value=1.0)),
        )
        terminal = TypeName(name=parser.terminal or "Token")
        return DeductionDecl(
            name=definition.name,
            domain=terminal,
            codomain=terminal,
            options=options,
            atoms=atoms,
            rules=tuple(rules),
            lexicon=lexicon,
            docs=definition.docs,
            line=definition.line,
            col=definition.col,
        )

    def _schema_parser_atoms(
        self: _Elaborator,
        definition: DefineDecl,
        parser: ExprParser,
        objects: list[ObjectDecl],
    ) -> tuple[str, ...]:
        """Return the atom names of a schema parser's category algebra."""
        if parser.categories:
            return parser.categories
        residuated = [
            declaration
            for declaration in objects
            if isinstance(declaration.init, TypeFreeResiduated)
        ]
        if len(residuated) != 1:
            self._fail(
                definition,
                f"schema parser {definition.name!r} needs one FreeResiduated "
                "object or an explicit categories list",
                code="qiec-program",
            )
        generator = residuated[0].init.generators
        for declaration in objects:
            if generator in declaration.names and isinstance(
                declaration.init, TypeEnumSet
            ):
                return declaration.init.elements
        self._fail(
            definition,
            f"schema parser {definition.name!r} names the undeclared atom "
            f"inventory {generator!r}",
            code="qiec-program",
        )

    def _schema_parser_terminals(
        self: _Elaborator,
        definition: DefineDecl,
        parser: ExprParser,
        objects: list[ObjectDecl],
    ) -> tuple[str, ...]:
        """Return the terminal vocabulary of a schema parser."""
        if parser.terminal is None:
            self._fail(
                definition,
                f"schema parser {definition.name!r} requires terminal=<object>",
                code="qiec-program",
            )
        for declaration in objects:
            if parser.terminal in declaration.names and isinstance(
                declaration.init, TypeEnumSet
            ):
                return declaration.init.elements
        self._fail(
            definition,
            f"schema parser {definition.name!r} terminal {parser.terminal!r} "
            "must be a declared enum object",
            code="qiec-program-gap",
        )

    @staticmethod
    def _span(start: str, stop: str, category: ObjectExpr) -> ObjectEffectApply:
        """Return a chart item pattern over one span and category."""
        return ObjectEffectApply(
            effect=SPAN,
            args=(TypeName(name=start), TypeName(name=stop), category),
        )

    def _category_object(self: _Elaborator, category: Category) -> ObjectExpr:
        """Translate a finite category term back to the QVR object AST."""
        if isinstance(category, AtomicCategory):
            return TypeName(name=category.name)
        if isinstance(category, SlashCategory):
            return ObjectSlash(
                result=self._category_object(category.result),
                argument=self._category_object(category.argument),
                direction=category.direction,
            )
        if isinstance(category, ProductCategory):
            return ObjectProduct(
                components=(
                    self._category_object(category.left),
                    self._category_object(category.right),
                )
            )
        if isinstance(category, UnitCategory):
            return TypeName(name="I")
        if isinstance(category, ModalCategory):
            return ObjectEffectApply(
                effect=category.modality,
                args=(self._category_object(category.inner),),
            )
        self._fail(
            self.source.syntax.statements[0],
            f"category term {category!r} has no QIEC chart encoding",
            code="qiec-program-gap",
        )

    def _declare_search_effect(self: _Elaborator) -> None:
        """Declare the ``Search`` effect and register it.

        Its one operation, ``choose[a : Type, n : Nat] : Tensor[a]([n])
        -> a``, is answered by resuming once per entry of the tensor.

        Raises
        ------
        QiecDiagnosticError
            If the module declares an effect of that name itself.
        """
        if SEARCH_EFFECT in self.effects:
            self._fail(
                self.source.syntax.statements[0],
                f"effect {SEARCH_EFFECT!r} is reserved for deductions",
                code="qiec-program",
            )
        effect_id = EffectId.derive(self.source.module_name, "effect", SEARCH_EFFECT)
        ref = EffectRef(id=effect_id, name=SEARCH_EFFECT)
        element = TypeVariable(name="a")
        extent = IndexVariable(name="n", sort=NAT)
        operation = OperationDef(
            id=OperationId.derive(effect_id, "operation", SEARCH_CHOOSE),
            name=SEARCH_CHOOSE,
            telescope=(TypeBinder(name="a"), IndexBinder(name="n", sort=NAT)),
            arguments=(
                ArgumentDef(name="alternatives", type=tensor_type(element, (extent,))),
            ),
            result_type=element,
        )
        effect = EffectDef(ref=ref, telescope=(), operations=(operation,))
        self.effects[SEARCH_EFFECT] = effect
        self.registry.register_effect(effect)

    def _read_deduction(self: _Elaborator, declaration: DeductionDecl) -> _System:
        """Read a declaration's options, rules, and lexicon into patterns.

        Parameters
        ----------
        declaration : DeductionDecl
            The declaration.

        Returns
        -------
        _System
            The system, with its symbols' slot sorts inferred.

        Raises
        ------
        QiecDiagnosticError
            If the semiring is unknown, an option has no elaboration, a
            rule fills a position slot with an atom or a structured
            term, or a rule's conclusion mentions a variable no premise
            binds.
        """
        semiring = get_option_name(
            declaration.options, "semiring", line=declaration.line, col=declaration.col
        )
        if semiring is None:
            semiring = "LogProb"
        if semiring not in SEMIRING_CARRIERS:
            self._fail(
                declaration,
                f"deduction {declaration.name!r} names the semiring {semiring!r}; "
                f"the semirings are {', '.join(SEMIRING_CARRIERS)}",
                code="qiec-program",
            )
        # An ``axioms`` morphism produces the axioms the agenda engine
        # starts from; on the reference machine the caller supplies them,
        # as for any system without a lexicon. A ``signature`` or an
        # ``encoder`` attaches item embeddings for structural losses and
        # leaves the weights alone, so neither concerns the elaboration.
        depth_entry = find_option(declaration.options, "depth")
        depth: int | None = None
        if depth_entry is not None:
            if not isinstance(depth_entry.value, OptionNumber):
                self._fail(
                    declaration,
                    f"deduction {declaration.name!r} option depth must be a number",
                    code="qiec-program",
                )
            depth = int(depth_entry.value.value)
        start = get_option_name(
            declaration.options, "start", line=declaration.line, col=declaration.col
        )
        system = _System(
            decl=declaration, semiring=semiring, start=start, depth_cell=[depth]
        )
        atoms = set(declaration.atoms)
        for atom in declaration.atoms:
            system.symbols.setdefault(atom, [])
        system.uses_forms_cell[0] = bool(declaration.binders) or any(
            isinstance(pattern, ObjectEffectApply)
            and pattern.effect == SPAN
            and len(pattern.args) >= 4
            for rule in declaration.rules
            for pattern in (*rule.premises, rule.conclusion)
        )
        for misplaced in misplaced_positions(declaration, position_slots(declaration)):
            rule_node = next(
                rule for rule in declaration.rules if rule.name == misplaced.rule
            )
            self._fail(
                rule_node, misplaced.message(declaration.name), code="qiec-program"
            )
        counter = [0]
        for rule in declaration.rules:
            premises = tuple(
                self._span_form(
                    self._object_pattern(item, atoms, system), system, counter
                )
                for item in rule.premises
            )
            conclusion = self._span_form(
                self._object_pattern(rule.conclusion, atoms, system), system, counter
            )
            bound = {
                name for premise in premises for name in _pattern_variables(premise)
            }
            unbound = [
                name for name in _pattern_variables(conclusion) if name not in bound
            ]
            if unbound:
                self._fail(
                    rule,
                    f"rule {rule.name!r} concludes with {', '.join(unbound)!r}, "
                    "which no premise binds",
                    code="qiec-program",
                )
            parent = find_option(rule.options, "parent")
            parent_name: str | None = None
            if parent is not None:
                if not isinstance(parent.value, OptionName):
                    self._fail(
                        rule,
                        f"rule {rule.name!r} option parent must name a rule",
                        code="qiec-program",
                    )
                parent_name = parent.value.value
            if find_option(rule.options, "weight") is not None:
                self._fail(
                    rule,
                    f"rule {rule.name!r} option weight is an expression over its "
                    "bindings; a computed rule weight has no elaboration yet",
                    code="qiec-program-gap",
                )
            system.rules.append(
                _Rule(
                    source=rule,
                    premises=premises,
                    conclusion=conclusion,
                    learnable=get_option_flag(rule.options, "learnable"),
                    bounded=get_option_flag(rule.options, "bounded"),
                    parent=parent_name,
                )
            )
        for rule in system.rules:
            if rule.parent is not None and rule.parent not in {
                item.source.name for item in system.rules
            }:
                self._fail(
                    rule.source,
                    f"rule {rule.source.name!r} names the parent {rule.parent!r}, "
                    "which the deduction does not declare",
                    code="qiec-program",
                )
        self._read_lexicon(declaration, atoms, system)
        mode, mode_refusal = normalization_mode(declaration)
        if mode_refusal is not None:
            self._fail(mode_refusal, mode_refusal.message, code="qiec-program")
        if mode == "lhs":
            file_rows = (
                tuple(
                    (word, category)
                    for word, category, _form in load_lexicon_tsv(
                        declaration.lexicon_from_file, declaration
                    )
                )
                if declaration.lexicon_from_file is not None
                else ()
            )
            analysis = production_groups(
                declaration, position_slots(declaration), file_rows
            )
            if analysis.refusals:
                first = analysis.refusals[0]
                self._fail(
                    first,
                    f"{first.message} (required by normalize=lhs)",
                    code="qiec-program",
                )
            system.normalized_cell[0] = True
            system.groups_cell[0] = analysis.groups
            system.table.update(
                {
                    production.key: position
                    for position, production in enumerate(
                        production
                        for group in analysis.groups
                        if len(group.productions) > 1
                        for production in group.productions
                    )
                }
            )
        self._infer_sorts(system)
        system.span_shaped_cell[0] = bool(system.entries) and all(
            _is_span(pattern)
            for rule in system.rules
            for pattern in (*rule.premises, rule.conclusion)
        )
        tolerance = get_option_float(
            declaration.options, "tolerance", line=declaration.line, col=declaration.col
        )
        system.tolerance_cell[0] = float(tolerance) if tolerance is not None else 0.0
        system.closure[:] = [
            index for index, rule in enumerate(system.rules) if len(rule.premises) == 1
        ]
        system.cyclic_cell[0] = _may_cycle(system)
        if system.by_yield:
            self._read_heads(system)
        if system.span_shaped_cell[0]:
            for rule in system.rules:
                conclusion = rule.conclusion
                assert isinstance(conclusion, _App)
                whole = (conclusion.args[0], conclusion.args[1])
                if len(rule.premises) > 1 and any(
                    isinstance(premise, _App)
                    and (premise.args[0], premise.args[1]) == whole
                    for premise in rule.premises
                ):
                    self._fail(
                        rule.source,
                        f"rule {rule.source.name!r} has a premise over its "
                        "conclusion's whole span beside other premises; the "
                        "search derives such a premise over the same span it "
                        "is deriving, which nothing bounds",
                        code="qiec-program",
                    )
        return system

    def _read_heads(self: _Elaborator, system: _System) -> None:
        """Number the heads of a yield-directed system and check its rules.

        A rule with a positional conclusion must be linear. A rule whose
        conclusion has no positions erases its premises' positions, as a
        test that some item exists anywhere in the sentence does; its
        premises' positions range over the whole sentence. Items without
        positions are derived only by erasing rules, so a positional rule
        with such a premise fires only if the deduction has an erasing
        rule, and then it is refused, since its premise could rest on its
        own conclusion.

        Parameters
        ----------
        system : _System
            A system with a lexicon whose rules do not all relate spans.

        Raises
        ------
        QiecDiagnosticError
            If a rule is not linear in the multiple context-free sense,
            a positional rule has a premise without positions that an
            erasing rule could derive, the rules among items without
            positions may form a cycle, or the unary rules may form a
            cycle, none of which the derivation by heads and components
            bounds.
        """
        heads: list[str] = [SPAN]
        erasing = any(
            not _has_positions(system, rule.conclusion) for rule in system.rules
        )
        unpositioned: dict[str, set[str]] = {}
        for index, rule in enumerate(system.rules):
            for pattern in (*rule.premises, rule.conclusion):
                if not isinstance(pattern, (_Atom, _App)):
                    self._fail(
                        rule.source,
                        f"rule {rule.source.name!r} has a premise or conclusion "
                        "that is not an item with a head symbol; a lexical "
                        "deduction whose items are not all spans derives each "
                        "item by its head",
                        code="qiec-program",
                    )
            free = [
                premise
                for premise in rule.premises
                if not _has_positions(system, premise)
            ]
            if _has_positions(system, rule.conclusion):
                if free and not erasing:
                    # Every item an erasing rule does not derive covers a
                    # token: an axiom covers one, and a linear rule's
                    # conclusion covers its premises'. Without erasing
                    # rules a premise without positions is never derived,
                    # and the rule never fires, in either evaluator.
                    continue
                if free:
                    self._fail(
                        rule.source,
                        f"rule {rule.source.name!r} derives an item with "
                        "positions from an item without them, which an erasing "
                        "rule derives from items with positions; the search by "
                        "yields cannot bound that recursion",
                        code="qiec-program",
                    )
                plan = _yield_plan(system, rule)
                if isinstance(plan, str):
                    self._fail(
                        rule.source,
                        f"rule {rule.source.name!r}: {plan}; a lexical deduction "
                        "whose items are not all spans derives items by their "
                        "yields, which needs every rule to concatenate its "
                        "premises' position pairs into its conclusion's",
                        code="qiec-program",
                    )
            else:
                unpositioned.setdefault(_head(rule.conclusion), set()).update(
                    _head(premise) for premise in free
                )
            system.live.append(index)
            for pattern in (*rule.premises, rule.conclusion):
                if _head(pattern) not in heads:
                    heads.append(_head(pattern))
        finished: set[str] = set()
        active: set[str] = set()

        def cycles(node: str) -> bool:
            """Whether a cycle among heads without positions is reachable.

            Parameters
            ----------
            node : str
                A head without positions.

            Returns
            -------
            bool
                ``True`` when the walk meets a head on its path.
            """
            if node in active:
                return True
            if node in finished:
                return False
            active.add(node)
            found = any(cycles(target) for target in unpositioned.get(node, ()))
            active.discard(node)
            finished.add(node)
            return found

        if any(cycles(node) for node in list(unpositioned)):
            self._fail(
                system.decl,
                f"deduction {system.decl.name!r} has rules among items without "
                "positions that may form a cycle, which the search by yields "
                "cannot bound",
                code="qiec-program",
            )
        if system.cyclic_cell[0]:
            self._fail(
                system.decl,
                f"deduction {system.decl.name!r} has unary rules that may form a "
                "cycle; a lexical deduction whose items are not all spans "
                "derives items by their yields, which a cycle of unary rules "
                "does not shrink",
                code="qiec-program",
            )
        system.heads[:] = heads

    def _object_pattern(
        self: _Elaborator, expr: ObjectExpr, atoms: set[str], system: _System
    ) -> Pattern:
        """Convert a rule's object expression into a pattern.

        Parameters
        ----------
        expr : ObjectExpr
            The expression.
        atoms : set[str]
            The declared atoms, whose names are ground; every other
            identifier is a variable.
        system : _System
            The system, whose symbol table the applied heads join.

        Returns
        -------
        Pattern
            The pattern.

        Raises
        ------
        QiecDiagnosticError
            If the expression has a shape the pattern language does not
            cover.
        """
        if isinstance(expr, TypeName):
            if is_pattern_variable(expr, frozenset(atoms)):
                return _Wild(name=expr.name)
            if expr.name.isdigit():
                return _Lit(value=int(expr.name))
            self._note_atom(system, expr.name)
            return _Atom(name=expr.name)
        if isinstance(expr, ObjectEffectApply):
            args = tuple(
                self._object_pattern(item, atoms, system) for item in expr.args
            )
            self._note_symbol(system, expr.effect, len(args))
            return _App(symbol=expr.effect, args=args)
        if isinstance(expr, ObjectSlash):
            symbol = "fwd_slash" if expr.direction == "/" else "bwd_slash"
            args = (
                self._object_pattern(expr.result, atoms, system),
                self._object_pattern(expr.argument, atoms, system),
            )
            self._note_symbol(system, symbol, 2)
            return _App(symbol=symbol, args=args)
        if isinstance(expr, ObjectProduct):
            args = tuple(
                self._object_pattern(item, atoms, system) for item in expr.components
            )
            symbol = f"product{len(args)}"
            self._note_symbol(system, symbol, len(args))
            return _App(symbol=symbol, args=args)
        self._fail(
            system.decl,
            f"deduction {system.decl.name!r} has a rule pattern of kind "
            f"{expr.kind!r}, which the pattern language does not cover",
            code="qiec-program",
        )

    def _note_atom(self: _Elaborator, system: _System, atom: str) -> None:
        """Record an atom used as an item on its own.

        Parameters
        ----------
        system : _System
            The system.
        atom : str
            The atom.

        Raises
        ------
        QiecDiagnosticError
            If the atom is also applied to arguments.
        """
        if system.symbols.get(atom):
            self._fail(
                system.decl,
                f"deduction {system.decl.name!r} uses {atom!r} both bare and applied",
                code="qiec-program",
            )
        system.bare.add(atom)

    def _note_symbol(
        self: _Elaborator, system: _System, symbol: str, arity: int
    ) -> None:
        """Record an applied symbol and its arity.

        A declared atom applied to arguments is a constructor of that
        arity, as the classic engine reads it, unless it is also used
        bare.

        Parameters
        ----------
        system : _System
            The system.
        symbol : str
            The symbol.
        arity : int
            How many arguments it takes.

        Raises
        ------
        QiecDiagnosticError
            If the symbol is applied at two arities, or is an atom also
            used bare.
        """
        existing = system.symbols.get(symbol)
        if existing is None or (not existing and symbol not in system.bare):
            system.symbols[symbol] = ["item"] * arity
        elif len(existing) != arity:
            self._fail(
                system.decl,
                f"deduction {system.decl.name!r} applies {symbol!r} to both "
                f"{len(existing)} and {arity} arguments",
                code="qiec-program",
            )

    def _span_form(
        self: _Elaborator, pattern: Pattern, system: _System, counter: list[int]
    ) -> Pattern:
        """Give a three-slot span pattern its form slot when the system has one.

        Parameters
        ----------
        pattern : Pattern
            The pattern.
        system : _System
            The system.
        counter : list[int]
            The count of fresh form variables minted so far.

        Returns
        -------
        Pattern
            The pattern, every ``span(I, J, C)`` extended with a fresh
            variable in the form slot when the system carries forms.
        """
        if not system.uses_forms_cell[0]:
            return pattern
        if isinstance(pattern, _App):
            args = tuple(
                self._span_form(item, system, counter) for item in pattern.args
            )
            if pattern.symbol == SPAN and len(args) == 3:
                counter[0] += 1
                args = (*args, _Wild(name=f"_lf{counter[0]}"))
                system.symbols[SPAN] = ["item"] * 4
            return _App(symbol=pattern.symbol, args=args)
        return pattern

    def _read_lexicon(
        self: _Elaborator, declaration: DeductionDecl, atoms: set[str], system: _System
    ) -> None:
        """Expand the lexicon into entries with pattern-form categories and forms.

        Parameters
        ----------
        declaration : DeductionDecl
            The declaration.
        atoms : set[str]
            The declared atoms.
        system : _System
            The system the entries join.

        Raises
        ------
        QiecDiagnosticError
            If a restricted category names an undeclared atom, or a
            logical form uses an expression outside the constructor
            language.
        """
        raw: list[
            tuple[
                str,
                LexiconCategoryFixed | None,
                LetExprNode | None,
                bool,
                tuple[str, ...],
            ]
        ] = []
        for entry in declaration.lexicon:
            learnable = get_option_flag(entry.options, "learnable")
            if isinstance(entry.category, LexiconCategoryFixed):
                candidates: tuple[str, ...] = ()
            elif isinstance(entry.category, LexiconCategoryWildcard):
                candidates = tuple(declaration.atoms)
            elif isinstance(entry.category, LexiconCategoryRestricted):
                unknown = [name for name in entry.category.atoms if name not in atoms]
                if unknown:
                    self._fail(
                        declaration,
                        f"deduction {declaration.name!r}: lexicon entry restricts its "
                        f"category to {', '.join(unknown)!r}, not declared atoms",
                        code="qiec-program",
                    )
                candidates = tuple(entry.category.atoms)
            else:
                self._fail(
                    declaration,
                    f"deduction {declaration.name!r}: lexicon entry has an unknown "
                    "category kind",
                    code="qiec-program",
                )
            fixed = (
                entry.category
                if isinstance(entry.category, LexiconCategoryFixed)
                else None
            )
            for word in entry.words:
                raw.append((word, fixed, entry.lf, learnable, candidates))
        if declaration.lexicon_from_file is not None:
            learnable = get_option_flag(
                declaration.lexicon_from_file_options, "learnable"
            )
            for word, category, form in load_lexicon_tsv(
                declaration.lexicon_from_file, declaration
            ):
                raw.append(
                    (word, LexiconCategoryFixed(category=category), form, learnable, ())
                )
        binders = frozenset(declaration.binders)
        for word, fixed, form_expr, learnable, candidates in raw:
            form = (
                self._form_pattern(form_expr, atoms, binders, system, declaration)
                if system.uses_forms_cell[0] and form_expr is not None
                else None
            )
            if fixed is not None:
                category = self._object_pattern(fixed.category, atoms, system)
                system.entries.append(
                    _Entry(word=word, category=category, form=form, learnable=learnable)
                )
            else:
                for atom in candidates:
                    system.entries.append(
                        _Entry(
                            word=word,
                            category=_Atom(name=atom),
                            form=form,
                            learnable=True,
                        )
                    )
        if system.entries:
            sorts: list[Sort] = ["int", "int", "item", "item"]
            system.symbols[SPAN] = sorts[: 4 if system.uses_forms_cell[0] else 3]

    def _form_pattern(
        self: _Elaborator,
        expr: LetExprNode,
        atoms: set[str],
        binders: frozenset[str],
        system: _System,
        declaration: DeductionDecl,
    ) -> Pattern:
        """Convert a logical form into a ground pattern with numbered binders.

        Parameters
        ----------
        expr : LetExprNode
            The form.
        atoms : set[str]
            The declared atoms.
        binders : frozenset[str]
            The constructors whose first argument binds a variable.
        system : _System
            The system, whose symbol table the heads join.
        declaration : DeductionDecl
            The declaration, for diagnostics.

        Returns
        -------
        Pattern
            The pattern, with bound variables numbered from one in
            binding order so alpha-equivalent forms are equal.

        Raises
        ------
        QiecDiagnosticError
            If the form uses an expression that is not a constructor
            application, an atom, a bound variable, or an integer.
        """
        counter = [0]

        def visit(node: LetExprNode, scope: dict[str, int]) -> Pattern:
            """Convert one subterm.

            Parameters
            ----------
            node : LetExprNode
                The subterm.
            scope : dict[str, int]
                The bound variables in scope, by number.

            Returns
            -------
            Pattern
                The pattern.
            """
            if isinstance(node, LetExprVar):
                if node.name in scope:
                    self._note_symbol(system, BOUND, 1)
                    return _Bound(index=scope[node.name])
                if node.name in atoms:
                    self._note_atom(system, node.name)
                    return _Atom(name=node.name)
                self._fail(
                    declaration,
                    f"deduction {declaration.name!r}: logical form names "
                    f"{node.name!r}, which is neither an atom nor bound",
                    code="qiec-program",
                )
            if isinstance(node, LetExprLiteral):
                if float(node.value).is_integer():
                    return _Lit(value=int(node.value))
                self._fail(
                    declaration,
                    f"deduction {declaration.name!r}: logical form holds the real "
                    f"{node.value!r}; items carry integers",
                    code="qiec-program",
                )
            if isinstance(node, LetExprCall):
                if node.func in binders and node.args:
                    first = node.args[0]
                    if not isinstance(first, LetExprVar):
                        self._fail(
                            declaration,
                            f"deduction {declaration.name!r}: binder {node.func!r} "
                            "binds a name in its first argument",
                            code="qiec-program",
                        )
                    counter[0] += 1
                    inner = {**scope, first.name: counter[0]}
                    self._note_symbol(system, BOUND, 1)
                    args = (
                        _Bound(index=counter[0]),
                        *(visit(item, inner) for item in node.args[1:]),
                    )
                else:
                    args = tuple(visit(item, scope) for item in node.args)
                self._note_symbol(system, node.func, len(args))
                return _App(symbol=node.func, args=args)
            self._fail(
                declaration,
                f"deduction {declaration.name!r}: logical form uses a "
                f"{type(node).__name__}, which is not a constructor term",
                code="qiec-program",
            )

        return visit(expr, {})

    def _infer_sorts(self: _Elaborator, system: _System) -> None:
        """Fix each constructor slot's sort from how the system uses it.

        A slot is an integer exactly when it holds a chart position, as
        `position_slots` infers it for the agenda compiler too: a
        position of ``span`` under a lexicon, a slot a literal fills, or
        a slot a variable shared with a position slot fills. Every other
        slot holds an item.

        Parameters
        ----------
        system : _System
            The system, whose symbol table is updated in place.
        """
        if BOUND in system.symbols:
            system.symbols[BOUND] = ["int"]
        for symbol, flags in position_slots(system.decl).items():
            sorts = system.symbols.get(symbol)
            if sorts is None:
                continue
            for index, positional in enumerate(flags[: len(sorts)]):
                if positional:
                    sorts[index] = "int"

    def _declare_item_family(self: _Elaborator, system: _System) -> None:
        """Declare the deduction's item family and register it.

        Parameters
        ----------
        system : _System
            The system.

        Raises
        ------
        QiecDiagnosticError
            If the kernel rejects a declaration.
        """
        family_name = system.name("Item")
        family_id = FamilyId.derive(self.source.module_name, "family", family_name)
        names = tuple(system.symbols)
        constructor_ids = tuple(
            ConstructorId.derive(family_id, "constructor", system.name(symbol))
            for symbol in names
        )
        family = FamilyDecl(
            id=family_id,
            name=family_name,
            parameters=(),
            indices=(),
            constructors=constructor_ids,
        )
        system.family_cell[0] = family
        item = TypeApplication(constructor=family.type_constructor, arguments=())
        self.families[family_name] = family
        self.registry.register_family(family)
        for symbol, identity in zip(names, constructor_ids, strict=True):
            fields = tuple(
                FieldDef(name=f"arg{position}", type=INT if sort == "int" else item)
                for position, sort in enumerate(system.symbols[symbol])
            )
            constructor = ConstructorDecl(
                id=identity,
                family=family_id,
                name=system.name(symbol),
                telescope=(),
                fields=fields,
                result_indices=(),
            )
            system.constructors[symbol] = constructor
            self.constructors[constructor.name] = constructor
            self.registry.register_constructor(constructor)

    def _frame_width(self: _Elaborator, system: _System) -> int:
        """The most position slots any head of a yield-directed system has.

        Parameters
        ----------
        system : _System
            The system.

        Returns
        -------
        int
            The number of frame arguments its derivation takes.
        """
        return max(
            sum(1 for sort in system.symbols[head] if sort == "int")
            for head in system.heads
        )

    def _declare_chain_family(self: _Elaborator, system: _System) -> None:
        """Declare the family of item chains a cyclic closure carries.

        A chain is ``nil`` or ``cons(item, chain)``: the items a chain of
        unary rules has visited, most recent first, against which each
        new conclusion is counted.

        Parameters
        ----------
        system : _System
            The system, whose item family is declared.
        """
        family_name = system.name("Chain")
        family_id = FamilyId.derive(self.source.module_name, "family", family_name)
        symbols = ("nil", "cons")
        constructor_ids = tuple(
            ConstructorId.derive(
                family_id, "constructor", system.name(f"chain_{symbol}")
            )
            for symbol in symbols
        )
        family = FamilyDecl(
            id=family_id,
            name=family_name,
            parameters=(),
            indices=(),
            constructors=constructor_ids,
        )
        system.chain_family_cell[0] = family
        chain = TypeApplication(constructor=family.type_constructor, arguments=())
        self.families[family_name] = family
        self.registry.register_family(family)
        fields_of = {
            "nil": (),
            "cons": (
                FieldDef(name="head", type=system.item),
                FieldDef(name="tail", type=chain),
            ),
        }
        for symbol, identity in zip(symbols, constructor_ids, strict=True):
            constructor = ConstructorDecl(
                id=identity,
                family=family_id,
                name=system.name(f"chain_{symbol}"),
                telescope=(),
                fields=fields_of[symbol],
                result_indices=(),
            )
            system.chain_constructors[symbol] = constructor
            self.constructors[constructor.name] = constructor
            self.registry.register_constructor(constructor)

    def _declare_deduction_instances(self: _Elaborator, system: _System) -> None:
        """Allocate the deduction's search and weight instances.

        Parameters
        ----------
        system : _System
            The system.
        """
        search = self.effects[SEARCH_EFFECT]
        choice_name = system.name("choice")
        entry = instantiate_effect(
            search.ref,
            module=self.source.module_name,
            lexical_path=("instances", choice_name),
        )
        system.choice_cell[0] = NamedEffectInstance(
            name=choice_name,
            entry=entry,
            origin=self._origin(
                system.decl, ("instances", choice_name), "effect-instance"
            ),
        )
        self.instances[choice_name] = system.choice_cell[0]
        weight_effect = self._effect("Weight")
        assert weight_effect is not None
        weight_name = system.name("weight")
        entry = instantiate_effect(
            weight_effect.apply((system.carrier,)),
            module=self.source.module_name,
            lexical_path=("instances", weight_name),
        )
        system.weight_cell[0] = NamedEffectInstance(
            name=weight_name,
            entry=entry,
            origin=self._origin(
                system.decl, ("instances", weight_name), "effect-instance"
            ),
        )
        self.instances[weight_name] = system.weight_cell[0]

    # ------------------------------------------------------------------
    # computations

    def _declare_deductions(self: _Elaborator) -> tuple[NamedComputation, ...]:
        """Elaborate every deduction into its handlers and computations.

        Returns
        -------
        tuple[NamedComputation, ...]
            The deductions' computations in source order: for each, the
            equality, rendering, and goal computations, the axiom and
            derivation computations, and the entry.

        Raises
        ------
        QiecDiagnosticError
            If a body fails to check.
        """
        computations: list[NamedComputation] = []
        weighted = _weighted_deductions(self.source.syntax.statements)
        for name, system in getattr(self, "_deductions", {}).items():
            computations.extend(self._elaborate_deduction(system))
            if name in weighted and system.normalized_cell[0]:
                computations.extend(self._elaborate_deduction(system, weighted=True))
        return tuple(computations)

    def _elaborate_deduction(
        self: _Elaborator, system: _System, weighted: bool = False
    ) -> list[NamedComputation]:
        """Elaborate one deduction.

        A normalized grammar whose production probabilities a program
        supplies is elaborated a second time, ``weighted``: the
        computations that read weights (the axiom, the derivation and its
        closure, and the entry with its convergence) take one more
        argument, the table of the log-probabilities of every production
        in a group of several, and read their weights from it instead of
        the ``Param`` store. They are named ``<D>__weighted_<suffix>``
        and share the item family, the instances, the handlers, and the
        pure computations of the deduction.

        Parameters
        ----------
        system : _System
            The system.
        weighted : bool
            Whether to elaborate the weighted variant.

        Returns
        -------
        list[NamedComputation]
            Its computations.
        """
        params = self.instances[PARAMS_INSTANCE]
        assert system.choice_cell[0] is not None and system.weight_cell[0] is not None
        # Only a log semiring reads learned weights, and only the
        # computations whose entries or rules declare one perform on
        # the parameter store.
        logged = system.semiring in ("LogProb", "Viterbi")
        lexical_learned = logged and any(entry.learnable for entry in system.entries)
        firing = (
            [system.rules[index] for index in system.live]
            if system.by_yield
            else system.rules
        )
        learned = lexical_learned or (
            logged and any(rule.learnable or rule.parent is not None for rule in firing)
        )
        if system.normalized_cell[0]:
            # A normalized production's weight reads the learnable free
            # weights of its group when the group has several members.
            reading = {
                production.key
                for group in system.groups_cell[0]
                if len(group.productions) > 1
                and any(member.learnable for member in group.productions)
                for production in group.productions
            }
            lexical_learned = any(
                ("lexicon", index) in reading for index in range(len(system.entries))
            )
            learned = lexical_learned or any(
                ("rule", index) in reading
                for index in (
                    system.live if system.by_yield else range(len(system.rules))
                )
            )
        if weighted:
            lexical_learned = learned = False
        base = (system.choice_cell[0].entry, system.weight_cell[0].entry)
        axiom_row = EffectRow(
            entries=(*base, params.entry) if lexical_learned else base
        )
        row = EffectRow(entries=(*base, params.entry) if learned else base)
        outer = EffectRow(entries=(params.entry,) if learned else ())
        pure = EffectRow()
        extent = IndexBinder(name="n", sort=NAT)
        n = IndexVariable(name="n", sort=NAT)
        item = system.item
        carrier = system.carrier
        signatures: dict[str, ComputationSignature] = {}
        table = (
            Local(
                name="productions",
                type=tensor_type(
                    REAL, (IndexLiteral(value=len(system.table), sort=NAT),)
                ),
            )
            if weighted
            else None
        )
        threaded = (
            {
                system.name(suffix): system.name(f"weighted_{suffix}")
                for suffix in THREADED
            }
            if weighted
            else {}
        )

        if weighted and not system.entries:
            for suffix in (
                "chart_step",
                *(
                    f"chart_rule{i}_{p}"
                    for i, rule in enumerate(system.rules)
                    for p in range(len(rule.premises) + 1)
                ),
            ):
                threaded[system.name(suffix)] = system.name(f"weighted_{suffix}")

        def declare(
            name: str,
            telescope: tuple[IndexBinder, ...],
            parameters: tuple[TypeExpr, ...],
            result: TypeExpr,
            effects: EffectRow,
        ) -> ComputationSignature:
            """Register one computation's signature before any body is built.

            Parameters
            ----------
            name : str
                The computation's name.
            telescope : tuple[IndexBinder, ...]
                Its static binders.
            parameters : tuple[TypeExpr, ...]
                Its parameter types.
            result : TypeExpr
                Its result type.
            effects : EffectRow
                Its row.

            Returns
            -------
            ComputationSignature
                The registered signature: in the weighted variant, the
                one declared by the deduction for a computation that
                reads no weight, and the weighted one, taking the table
                last, for one that does.
            """
            if weighted and name not in threaded:
                signature = self.computation_signatures[name]
                signatures[name] = signature
                return signature
            if table is not None:
                parameters = (*parameters, table.type)
                signatures[name] = signatures[threaded[name]] = declare_named(
                    threaded[name], telescope, parameters, result, effects
                )
                return signatures[name]
            return declare_named(name, telescope, parameters, result, effects)

        def declare_named(
            name: str,
            telescope: tuple[IndexBinder, ...],
            parameters: tuple[TypeExpr, ...],
            result: TypeExpr,
            effects: EffectRow,
        ) -> ComputationSignature:
            """Register a signature under its own name.

            Parameters
            ----------
            name : str
                The computation's name.
            telescope : tuple[IndexBinder, ...]
                Its static binders.
            parameters : tuple[TypeExpr, ...]
                Its parameter types.
            result : TypeExpr
                Its result type.
            effects : EffectRow
                Its row.

            Returns
            -------
            ComputationSignature
                The registered signature.
            """
            identity = ComputationId.derive(
                self.source.module_name, "computation", name
            )
            signature = ComputationSignature(
                id=identity,
                name=name,
                telescope=telescope,
                parameters=parameters,
                result=result,
                effects=effects,
            )
            self.registry.register_computation(signature)
            self.computation_signatures[name] = signature
            signatures[name] = signature
            return signature

        eq = declare(system.name("eq"), (), (item, item), BOOL, pure)
        show = declare(system.name("show"), (), (item,), STRING, pure)
        goal = declare(system.name("goal"), (), (item,), BOOL, pure)
        failing = declare(
            system.name("fail"),
            (),
            (),
            item,
            EffectRow(entries=(system.choice_cell[0].entry,)),
        )
        if system.entries:
            tokens_type = tensor_type(STRING, (n,))
            axiom = declare(
                system.name("axiom"),
                (extent,),
                (INT, item, tokens_type)
                if system.schema_categories_cell[0]
                else (INT, tokens_type),
                item,
                axiom_row,
            )
            repeats: tuple[TypeExpr, ...] = (INT,) if system.chained else ()
            derive = declare(
                system.name("derive"),
                (extent,),
                (INT, INT, INT, item, tokens_type)
                if system.schema_categories_cell[0]
                else (*repeats, INT, INT, tokens_type)
                if system.span_shaped_cell[0]
                else (INT, *([INT] * self._frame_width(system)), tokens_type),
                item,
                row,
            )
            if system.closes:
                declare(
                    system.name("close"),
                    (),
                    (INT, system.chain, item) if system.chained else (item,),
                    item,
                    row,
                )
            if system.chained:
                declare(system.name("count"), (), (item, system.chain), INT, pure)
                declare(
                    system.name("inside"),
                    (extent,),
                    (INT, tokens_type),
                    carrier,
                    outer,
                )
                declare(
                    system.name("converge"),
                    (extent,),
                    (INT, carrier, tokens_type),
                    carrier,
                    outer,
                )
            entry = declare(
                system.name("run"), (extent,), (tokens_type,), carrier, outer
            )
        else:
            axioms_type = tensor_type(item, (n,))
            weights_type = tensor_type(carrier, (n,))
            axiom = None
            chart = TypeApplication(
                constructor=self.families[system.name("Chart")].type_constructor,
                arguments=(),
            )
            derive = declare(
                system.name("derive"),
                (extent,),
                (chart, axioms_type, weights_type),
                chart,
                outer,
            )
            for suffix, parameters, result, effects in (
                ("chart_add", (chart, item, carrier), chart, pure),
                ("chart_zero", (chart,), chart, pure),
                ("chart_equal", (chart, chart), BOOL, pure),
                ("chart_goal", (chart,), carrier, pure),
                ("chart_axioms", (INT, chart, axioms_type, weights_type), chart, pure),
                ("chart_step", (chart, axioms_type, weights_type), chart, outer),
                ("chart_plus", (carrier, carrier), carrier, pure),
            ):
                declare(system.name(suffix), (extent,), parameters, result, effects)
            for rule_index, rule in enumerate(system.rules):
                for position in range(len(rule.premises) + 1):
                    declare(
                        system.name(f"chart_rule{rule_index}_{position}"),
                        (extent,),
                        (chart, chart, chart, carrier, *((item,) * position)),
                        chart,
                        outer,
                    )
            entry = declare(
                system.name("run"),
                (extent,),
                (axioms_type, weights_type),
                carrier,
                outer,
            )
        if system.depth_cell[0] is not None:
            declare(system.name("depth"), (), (item,), INT, pure)
        normalizers: list[tuple[ComputationSignature, ProductionGroup]] = []
        if system.normalized_cell[0] and not weighted:
            for index, group in enumerate(system.groups_cell[0]):
                if len(group.productions) > 1 and any(
                    member.learnable for member in group.productions
                ):
                    normalizers.append(
                        (
                            declare(
                                system.name(f"lhs{index}"),
                                (),
                                (),
                                REAL,
                                EffectRow(entries=(params.entry,)),
                            ),
                            group,
                        )
                    )
        context = _Context(self, system, signatures, params, n, table, threaded)
        computations: list[NamedComputation] = [
            context.normalizer(signature, group) for signature, group in normalizers
        ]
        if not weighted:
            computations.extend(
                (
                    context.equality(eq),
                    context.rendering(show),
                    context.goal(goal),
                    context.failure(failing),
                )
            )
            if system.depth_cell[0] is not None:
                computations.append(
                    context.depth_measure(signatures[system.name("depth")])
                )
            if system.chained:
                computations.append(
                    context.chain_count(signatures[system.name("count")])
                )
        if axiom is not None:
            computations.append(context.axiom(axiom))
        if system.entries:
            computations.append(context.derivation(derive))
        if system.closes:
            computations.append(context.closure(signatures[system.name("close")]))
        if system.handlers_cell[0] is None:
            system.handlers_cell[0] = self._deduction_handlers(system)
        handlers = system.handlers_cell[0]
        if system.chained:
            computations.append(
                context.inside(signatures[system.name("inside")], handlers)
            )
            computations.append(
                context.convergence(signatures[system.name("converge")])
            )
        if system.entries:
            computations.append(context.entry(entry, handlers))
        else:
            computations.extend(elaborate_chart(context))
        return computations

    def _deduction_handlers(
        self: _Elaborator, system: _System
    ) -> tuple[HandlerDef, HandlerDef]:
        """Declare the search and collection handlers of one deduction.

        Parameters
        ----------
        system : _System
            The system.

        Returns
        -------
        tuple[HandlerDef, HandlerDef]
            The ``Search`` handler, answering the semiring sum of the
            collected weights of its shots, and the ``Weight`` handler,
            answering the scope's unit paired with its product.
        """
        assert system.choice_cell[0] is not None and system.weight_cell[0] is not None
        search = self.effects[SEARCH_EFFECT]
        weight_effect = self._effect("Weight")
        assert weight_effect is not None
        carrier = system.carrier
        shot = product_type(UNIT, carrier)
        search_name = system.name(f"search_{system.semiring.lower()}")
        search_handler = HandlerDef(
            id=HandlerId.derive(self.source.module_name, "handler", search_name),
            name=search_name,
            effect=search.ref,
            clauses=(
                HandlerClauseDef(
                    operation=search.operations[0].id,
                    grade=ResumptionGrade.UNRESTRICTED,
                ),
            ),
            input_type=shot,
            output_type=carrier,
            introduced=EffectRow(),
            total=True,
            implementation="foreign",
        )
        collect_name = system.name(f"collect_{system.semiring.lower()}")
        add = next(item for item in weight_effect.operations if item.name == "add")
        collect_handler = HandlerDef(
            id=HandlerId.derive(self.source.module_name, "handler", collect_name),
            name=collect_name,
            effect=weight_effect.apply((carrier,)),
            clauses=(HandlerClauseDef(operation=add.id, grade=ResumptionGrade.LINEAR),),
            input_type=UNIT,
            output_type=shot,
            introduced=EffectRow(),
            total=True,
            implementation="foreign",
        )
        for handler in (search_handler, collect_handler):
            self.handlers[handler.name] = handler
            self.registry.register_handler(handler)
        return search_handler, collect_handler


type _Continuation = Callable[[dict[str, Local]], Computation]


class _Context:
    """The term builders of one deduction's computations.

    Parameters
    ----------
    elaborator : _Elaborator
        The elaborator, for primitives, origins, and diagnostics.
    system : _System
        The system.
    signatures : dict[str, ComputationSignature]
        The deduction's computations by name.
    params : NamedEffectInstance
        The module's ``Param`` instance.
    extent : IndexVariable
        The static extent of the entry's tensor argument.
    table : Local | None
        In the weighted variant, the table of production log-weights
        every weight-reading computation takes last.
    threaded : dict[str, str]
        In the weighted variant, the weighted name of each computation
        that takes the table.
    """

    def __init__(
        self,
        elaborator: _Elaborator,
        system: _System,
        signatures: dict[str, ComputationSignature],
        params: NamedEffectInstance,
        extent: IndexVariable,
        table: Local | None = None,
        threaded: dict[str, str] | None = None,
    ) -> None:
        self.elaborator = elaborator
        self.system = system
        self.signatures = signatures
        self.params = params
        self.extent = extent
        self.table = table
        self.threaded = dict(threaded or {})
        self.module = elaborator.source.module_name
        self.path: tuple[str | int, ...] = (
            ("deductions", system.decl.name)
            if table is None
            else ("deductions", system.decl.name, "weighted")
        )

    # -- term helpers ---------------------------------------------------

    def origin(self, *path: str | int) -> object:
        """A source origin beneath the deduction's path.

        Parameters
        ----------
        *path : str | int
            The path beneath it.

        Returns
        -------
        object
            The origin.
        """
        return self.elaborator._origin(
            self.system.decl, (*self.path, *path), "computation"
        )

    def primitive(self, name: str, *arguments: Value) -> Value:
        """Apply a registry primitive to scalar arguments.

        Parameters
        ----------
        name : str
            The primitive.
        *arguments : Value
            Its arguments.

        Returns
        -------
        Value
            The application.
        """
        return self.elaborator._primitive(
            name, tuple(arguments), self.system.decl, (*self.path, "primitives", name)
        )

    def call(self, name: str, arguments: tuple[Value, ...]) -> Call:
        """Call one of the deduction's computations.

        Parameters
        ----------
        name : str
            The computation's name.
        arguments : tuple[Value, ...]
            The arguments.

        Returns
        -------
        Call
            The call, with the extent passed along when the callee
            binds one.
        """
        signature = self.signatures[name]
        if name in self.threaded or name in self.threaded.values():
            assert self.table is not None
            name = self.threaded.get(name, name)
            arguments = (*arguments, Var(local=self.table))
        statics = tuple(
            IndexVariable(name=binder.name, sort=binder.sort)
            for binder in signature.telescope
            if isinstance(binder, IndexBinder)
        )
        return Call(
            callee=signature.id,
            name=name,
            static_arguments=statics,
            arguments=arguments,
            result_type=signature.result,
            effects=signature.effects,
            origin=self.elaborator._origin(
                self.system.decl, (*self.path, "calls", name), "call"
            ),
        )

    def request(
        self,
        instance: NamedEffectInstance,
        operation: str,
        statics: tuple[TypeExpr | IndexLiteral | IndexVariable, ...],
        arguments: tuple[Value, ...],
        result: TypeExpr,
        *path: str | int,
    ) -> Perform:
        """Perform an operation on one of the deduction's instances.

        Parameters
        ----------
        instance : NamedEffectInstance
            The instance.
        operation : str
            The operation's name.
        statics : tuple[TypeExpr | IndexLiteral | IndexVariable, ...]
            The operation's static arguments.
        arguments : tuple[Value, ...]
            Its value arguments.
        result : TypeExpr
            The request's result type.
        *path : str | int
            The request's structural path beneath the deduction's.

        Returns
        -------
        Perform
            The request.
        """
        effect = self.elaborator.registry.effects[instance.entry.effect.id]
        definition = next(item for item in effect.operations if item.name == operation)
        return Perform(
            request=EffectRequest(
                instance=instance.entry.instance,
                effect=instance.entry.effect,
                operation=definition.id,
                static_arguments=statics,
                arguments=arguments,
                result_type=result,
                origin=SiteProvenance(
                    origin=self.elaborator._origin(
                        self.system.decl, (*self.path, *path), "effect-request"
                    )
                ),
            )
        )

    def choose(
        self,
        alternatives: Value,
        element: TypeExpr,
        count: IndexLiteral | IndexVariable,
        *path: str | int,
    ) -> Perform:
        """Choose among the entries of a tensor.

        Parameters
        ----------
        alternatives : Value
            A ``Tensor[element]([count])``.
        element : TypeExpr
            The entry type.
        count : IndexLiteral | IndexVariable
            The tensor's extent.
        *path : str | int
            The request's path.

        Returns
        -------
        Perform
            The request, answered with one entry per resumption.
        """
        assert self.system.choice_cell[0] is not None
        return self.request(
            self.system.choice_cell[0],
            SEARCH_CHOOSE,
            (element, count),
            (alternatives,),
            element,
            *path,
        )

    def fail(self, result: TypeExpr, *path: str | int) -> Computation:
        """An empty choice, which no derivation continues from.

        Parameters
        ----------
        result : TypeExpr
            The type the failed computation would have produced.
        *path : str | int
            The request's path.

        Returns
        -------
        Computation
            A choice among no alternatives: a call of the deduction's
            failing computation when the type is its item type, which
            the matches reach at every constructor a pattern refuses,
            else the request itself.
        """
        if result == self.system.item and self.system.name("fail") in self.signatures:
            return self.call(self.system.name("fail"), ())
        empty = TensorValue(
            items=(),
            result_type=tensor_type(result, (IndexLiteral(value=0, sort=NAT),)),
        )
        return self.choose(empty, result, IndexLiteral(value=0, sort=NAT), *path)

    def failure(self, signature: ComputationSignature) -> NamedComputation:
        """The failing computation: an empty choice at the item type.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation.
        """
        empty = TensorValue(
            items=(),
            result_type=tensor_type(
                self.system.item, (IndexLiteral(value=0, sort=NAT),)
            ),
        )
        body = self.choose(
            empty, self.system.item, IndexLiteral(value=0, sort=NAT), "fail"
        )
        return self.computation(signature, (), body)

    def add_weight(self, weight: Value, *path: str | int) -> Perform:
        """Add a weight on the deduction's ``Weight`` instance.

        Parameters
        ----------
        weight : Value
            A value of the semiring's carrier.
        *path : str | int
            The request's path.

        Returns
        -------
        Perform
            The request.
        """
        assert self.system.weight_cell[0] is not None
        return self.request(
            self.system.weight_cell[0], "add", (), (weight,), UNIT, *path
        )

    def parameter(self, name: Value, *path: str | int) -> Perform:
        """Read a learned real parameter by name.

        Parameters
        ----------
        name : Value
            The parameter's name, a ``String``.
        *path : str | int
            The request's path.

        Returns
        -------
        Perform
            The request, answered with a ``Real``.
        """
        return self.request(self.params, "get", (REAL,), (name,), REAL, *path)

    def constructor(self, symbol: str, fields: tuple[Value, ...]) -> ConstructorValue:
        """Apply one of the item family's constructors.

        Parameters
        ----------
        symbol : str
            The symbol.
        fields : tuple[Value, ...]
            The field values.

        Returns
        -------
        ConstructorValue
            The item.
        """
        return ConstructorValue(
            constructor=self.system.constructors[symbol].id,
            static_arguments=(),
            fields=fields,
            result_type=self.system.item,
        )

    def case(
        self,
        scrutinee: Value,
        result: TypeExpr,
        branches: dict[str, Callable[[tuple[Local, ...]], Computation]],
        otherwise: Callable[[], Computation],
        *path: str | int,
    ) -> Case:
        """Analyze an item by constructor.

        Parameters
        ----------
        scrutinee : Value
            The item.
        result : TypeExpr
            The type every branch produces.
        branches : dict[str, Callable[[tuple[Local, ...]], Computation]]
            The bodies of the constructors matched, each given the
            locals bound to the constructor's fields.
        otherwise : Callable[[], Computation]
            The body of every other constructor.
        *path : str | int
            The case's path, which the branch scopes derive from.

        Returns
        -------
        Case
            The complete case.
        """
        built: list[CaseBranch] = []
        for position, (symbol, constructor) in enumerate(
            self.system.constructors.items()
        ):
            fields = tuple(
                self.system.local(f"f{index}", definition.type)
                for index, definition in enumerate(constructor.fields)
            )
            body = branches[symbol](fields) if symbol in branches else otherwise()
            built.append(
                CaseBranch(
                    constructor=constructor.id,
                    static_arguments=(),
                    fields=fields,
                    body=body,
                    scope=StaticScopeId.derive(
                        self.module,
                        (*self.path, *path),
                        "branch",
                        position,
                        constructor.id,
                    ),
                )
            )
        return Case(
            scrutinee=scrutinee,
            motive=CaseMotive(indices=(), result_type=result),
            branches=tuple(built),
        )

    def bind(
        self,
        hint: str,
        type_: TypeExpr,
        first: Computation,
        then: Callable[[Local], Computation],
    ) -> Bind:
        """Bind a computation's result to a fresh local.

        Parameters
        ----------
        hint : str
            The local's name stem.
        type_ : TypeExpr
            The result's type.
        first : Computation
            The computation.
        then : Callable[[Local], Computation]
            The continuation, given the local.

        Returns
        -------
        Bind
            The bind.
        """
        local = self.system.local(hint, type_)
        return Bind(binder=local, first=first, then=then(local))

    def weight_of(self, real: Value | None) -> Value:
        """A weight of the semiring's carrier from a real log weight.

        Parameters
        ----------
        real : Value | None
            The log weight as a ``Real``, or ``None`` for the semiring's
            multiplicative identity.

        Returns
        -------
        Value
            The carrier value: the log weight itself for the log
            semirings, ``true`` for the Boolean one, ``1`` for the
            counting one, which ignore learned weights.
        """
        semiring = self.system.semiring
        if semiring in ("LogProb", "Viterbi"):
            return self.primitive(
                "as_weight",
                real if real is not None else LiteralValue(value=0.0, type=REAL),
            )
        if semiring == "Boolean":
            return LiteralValue(value=True, type=BOOL)
        return _int(1)

    # -- equality, rendering, goal --------------------------------------

    def equality(self, signature: ComputationSignature) -> NamedComputation:
        """Structural equality of two items.

        Two items are equal exactly when their renderings are: the
        rendering writes each constructor's name and its fields in
        order, integers in decimal, so it is injective on items, and
        comparing the two strings costs one case analysis per item
        rather than one per pair of constructors.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation.
        """
        item = self.system.item
        left = Local(name="left", type=item)
        right = Local(name="right", type=item)
        show = self.system.name("show")
        body = self.bind(
            "left_text",
            STRING,
            self.call(show, (Var(local=left),)),
            lambda left_text: self.bind(
                "right_text",
                STRING,
                self.call(show, (Var(local=right),)),
                lambda right_text: Return(
                    value=self.primitive(
                        "eq_string", Var(local=left_text), Var(local=right_text)
                    )
                ),
            ),
        )
        return self.computation(signature, (left, right), body)

    def rendering(self, signature: ComputationSignature) -> NamedComputation:
        """Render an item as the text the classic engine keys weights by.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation.
        """
        item = Local(name="item", type=self.system.item)

        def render(symbol: str) -> Callable[[tuple[Local, ...]], Computation]:
            """The branch rendering one constructor.

            Parameters
            ----------
            symbol : str
                The constructor's symbol.

            Returns
            -------
            Callable[[tuple[Local, ...]], Computation]
                The branch body builder.
            """

            def body(fields: tuple[Local, ...]) -> Computation:
                """Render the constructor applied to its fields.

                Parameters
                ----------
                fields : tuple[Local, ...]
                    The fields.

                Returns
                -------
                Computation
                    The text.
                """
                if not fields:
                    return Return(value=_string(symbol))

                def piece(position: int, rendered: list[Value]) -> Computation:
                    """Render the fields from a position on, then join.

                    Parameters
                    ----------
                    position : int
                        The next field.
                    rendered : list[Value]
                        The fields rendered so far.

                    Returns
                    -------
                    Computation
                        The text.
                    """
                    if position == len(fields):
                        text: Value = _string(f"{symbol}(")
                        for index, part in enumerate(rendered):
                            if index:
                                text = self.primitive("concat", text, _string(","))
                            text = self.primitive("concat", text, part)
                        return Return(
                            value=self.primitive("concat", text, _string(")"))
                        )
                    local = fields[position]
                    if local.type == INT:
                        return piece(
                            position + 1,
                            [
                                *rendered,
                                self.primitive("int_to_string", Var(local=local)),
                            ],
                        )
                    return self.bind(
                        "text",
                        STRING,
                        self.call(signature.name, (Var(local=local),)),
                        lambda text: piece(position + 1, [*rendered, Var(local=text)]),
                    )

                return piece(0, [])

            return body

        body = self.case(
            Var(local=item),
            STRING,
            {symbol: render(symbol) for symbol in self.system.constructors},
            lambda: Return(value=_string("")),
            "show",
        )
        return self.computation(signature, (item,), body)

    def goal(self, signature: ComputationSignature) -> NamedComputation:
        """Whether an item is a goal: the start symbol as the engine reads it.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation: every item is a goal without a start
            symbol; with one, the atom itself, an item headed by it, or
            a span whose category is that atom.
        """
        item = Local(name="item", type=self.system.item)
        start = self.system.start

        def yes() -> Computation:
            """Answer that the item is a goal.

            Returns
            -------
            Computation
                ``true``.
            """
            return Return(value=LiteralValue(value=True, type=BOOL))

        def no() -> Computation:
            """Answer that the item is not a goal.

            Returns
            -------
            Computation
                ``false``.
            """
            return Return(value=LiteralValue(value=False, type=BOOL))

        if start is None:
            return self.computation(signature, (item,), yes())
        branches: dict[str, Callable[[tuple[Local, ...]], Computation]] = {}
        if start in self.system.constructors:
            branches[start] = lambda fields: yes()
        if (
            SPAN in self.system.constructors
            and start in self.system.constructors
            and not self.system.constructors[start].fields
        ):
            branches[SPAN] = lambda fields: self.case(
                Var(local=fields[2]),
                BOOL,
                {start: lambda inner: yes()},
                no,
                "goal",
                "category",
            )
        body = self.case(Var(local=item), BOOL, branches, no, "goal")
        return self.computation(signature, (item,), body)

    # -- axioms ---------------------------------------------------------

    def axiom(self, signature: ComputationSignature) -> NamedComputation:
        """The lexical items a token yields at a position.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation, choosing among the entries whose word is
            the token and adding the entry's weight.
        """
        if self.system.schema_categories_cell[0]:
            return self.schema_axiom(signature)
        position = Local(name="position", type=INT)
        tokens = Local(name="tokens", type=signature.parameters[1])
        entries = self.system.entries
        token = self.system.local("token", STRING)
        alternatives = TensorValue(
            items=tuple(_int(index) for index in range(len(entries))),
            result_type=tensor_type(INT, (IndexLiteral(value=len(entries), sort=NAT),)),
        )

        def entry_body(index: int, entry: _Entry) -> Computation:
            """The item one entry yields, once its word matches.

            Parameters
            ----------
            index : int
                The entry's index.
            entry : _Entry
                The entry.

            Returns
            -------
            Computation
                Add the entry's weight and return its span.
            """
            fields: list[Value] = [
                Var(local=position),
                self.primitive("add_int", Var(local=position), _int(1)),
                self.ground(entry.category),
            ]
            if self.system.uses_forms_cell[0]:
                assert entry.form is not None
                fields.append(self.ground(entry.form))
            span = self.constructor(SPAN, tuple(fields))
            result = Return(value=span)

            def weighted(weight: Value) -> Computation:
                """Add a weight, then answer the span.

                Parameters
                ----------
                weight : Value
                    The weight.

                Returns
                -------
                Computation
                    The sequence.
                """
                return Bind(
                    binder=self.system.local("added", UNIT),
                    first=self.add_weight(weight, "lexicon", index, "weight"),
                    then=result,
                )

            if self.system.normalized_cell[0]:
                return self.production_weight(("lexicon", index), weighted)
            if entry.learnable and self.system.semiring in ("LogProb", "Viterbi"):
                return self.bind(
                    "weight",
                    REAL,
                    self.parameter(
                        _string(f"{self.system.decl.name}.lex.{index}"),
                        "lexicon",
                        index,
                        "parameter",
                    ),
                    lambda real: weighted(self.weight_of(Var(local=real))),
                )
            return weighted(self.weight_of(None))

        def dispatch(chosen: Local, index: int) -> Computation:
            """Select the chosen entry by index.

            Parameters
            ----------
            chosen : Local
                The chosen index.
            index : int
                The entry tested here.

            Returns
            -------
            Computation
                The chain of tests from this entry on.
            """
            if index == len(entries):
                return self.fail(self.system.item, "lexicon", "none")
            entry = entries[index]
            return If(
                condition=self.primitive("eq_int", Var(local=chosen), _int(index)),
                then=If(
                    condition=self.primitive(
                        "eq_string", Var(local=token), _string(entry.word)
                    ),
                    then=entry_body(index, entry),
                    otherwise=self.fail(self.system.item, "lexicon", index, "mismatch"),
                ),
                otherwise=dispatch(chosen, index + 1),
            )

        body = Bind(
            binder=token,
            first=Return(
                value=Gather(
                    value=Var(local=tokens),
                    index=Var(local=position),
                    result_type=STRING,
                )
            ),
            then=self.bind(
                "entry",
                INT,
                self.choose(
                    alternatives,
                    INT,
                    IndexLiteral(value=len(entries), sort=NAT),
                    "lexicon",
                    "choose",
                ),
                lambda chosen: dispatch(chosen, 0),
            ),
        )
        return self.computation(signature, (position, tokens), body)

    def schema_axiom(self, signature: ComputationSignature) -> NamedComputation:
        """Build a schema parser's lexical item at an expected category.

        Parameters
        ----------
        signature : ComputationSignature
            The axiom computation's registered signature.

        Returns
        -------
        NamedComputation
            The lexical axiom computation.
        """
        system = self.system
        position = Local(name="position", type=INT)
        expected = Local(name="expected", type=system.item)
        tokens = Local(name="tokens", type=signature.parameters[2])
        token = system.local("token", STRING)
        count = len(system.schema_categories_cell[0])

        def token_body(token_index: int, category_index: int) -> Computation:
            """Add one lexical-table cell and return its span."""
            span = self.constructor(
                SPAN,
                (
                    Var(local=position),
                    self.primitive("add_int", Var(local=position), _int(1)),
                    Var(local=expected),
                ),
            )
            flat_index = token_index * count + category_index

            def weighted(weight: Value) -> Computation:
                """Add the lexical weight and return its span."""
                return Bind(
                    binder=system.local("added", UNIT),
                    first=self.add_weight(
                        weight,
                        "lexicon",
                        flat_index,
                        "weight",
                    ),
                    then=Return(value=span),
                )

            return self.bind(
                "weight",
                REAL,
                self.parameter(
                    _string(f"{system.decl.name}.lex.{flat_index}"),
                    "lexicon",
                    flat_index,
                    "parameter",
                ),
                lambda real: weighted(self.weight_of(Var(local=real))),
            )

        def word(token_index: int, category_index: int) -> Computation:
            """Dispatch the observed token for one expected category."""
            if token_index == len(system.schema_words_cell[0]):
                return self.fail(system.item, "lexicon", "token", "none")
            return If(
                condition=self.primitive(
                    "eq_string",
                    Var(local=token),
                    _string(system.schema_words_cell[0][token_index]),
                ),
                then=token_body(token_index, category_index),
                otherwise=word(token_index + 1, category_index),
            )

        def category(index: int) -> Computation:
            """Dispatch the expected category against the finite algebra."""
            if index == count:
                return self.fail(system.item, "lexicon", "category", "none")
            return self.bind(
                "same",
                BOOL,
                self.call(
                    system.name("eq"),
                    (
                        Var(local=expected),
                        self.ground(system.schema_categories_cell[0][index]),
                    ),
                ),
                lambda same: If(
                    condition=Var(local=same),
                    then=Bind(
                        binder=token,
                        first=Return(
                            value=Gather(
                                value=Var(local=tokens),
                                index=Var(local=position),
                                result_type=STRING,
                            )
                        ),
                        then=word(0, index),
                    ),
                    otherwise=category(index + 1),
                ),
            )

        return self.computation(
            signature,
            (position, expected, tokens),
            category(0),
        )

    def ground(self, pattern: Pattern) -> Value:
        """The item a ground pattern denotes.

        Parameters
        ----------
        pattern : Pattern
            A pattern without variables.

        Returns
        -------
        Value
            The item, or the integer for a literal.
        """
        return self.instantiate(pattern, {})

    def instantiate(self, pattern: Pattern, bindings: dict[str, Local]) -> Value:
        """The value a pattern denotes under bindings.

        Parameters
        ----------
        pattern : Pattern
            The pattern.
        bindings : dict[str, Local]
            The variables' values.

        Returns
        -------
        Value
            The item, or the integer for a literal or an integer
            variable.
        """
        if isinstance(pattern, _Atom):
            return self.constructor(pattern.name, ())
        if isinstance(pattern, _Wild):
            return Var(local=bindings[pattern.name])
        if isinstance(pattern, _Lit):
            return _int(pattern.value)
        if isinstance(pattern, _Bound):
            return self.constructor(BOUND, (_int(pattern.index),))
        return self.constructor(
            pattern.symbol,
            tuple(self.instantiate(argument, bindings) for argument in pattern.args),
        )

    # -- matching -------------------------------------------------------

    def match(
        self,
        value: Local,
        pattern: Pattern,
        bindings: dict[str, Local],
        result: TypeExpr,
        continue_with: _Continuation,
        *path: str | int,
        failed: Callable[[], Computation] | None = None,
    ) -> Computation:
        """Match a value against a pattern, continuing with the bindings.

        Parameters
        ----------
        value : Local
            The value, an item or an integer as the pattern's slot says.
        pattern : Pattern
            The pattern.
        bindings : dict[str, Local]
            The variables bound so far.
        result : TypeExpr
            The type the continuation produces, which a failed match
            produces by an empty choice.
        continue_with : _Continuation
            What to run once the pattern matches, given the bindings.
        *path : str | int
            The match's path.

        failed : Callable[[], Computation] | None
            The deterministic failure continuation, or an empty search choice.

        Returns
        -------
        Computation
            The match.
        """
        failure = failed or (lambda: self.fail(result, *path, "match"))
        if isinstance(pattern, _Wild):
            bound = bindings.get(pattern.name)
            if bound is None:
                return continue_with({**bindings, pattern.name: value})
            if value.type == INT:
                condition = self.primitive("eq_int", Var(local=bound), Var(local=value))
                return If(
                    condition=condition,
                    then=continue_with(bindings),
                    otherwise=failure(),
                )
            return self.bind(
                "same",
                BOOL,
                self.call(self.system.name("eq"), (Var(local=bound), Var(local=value))),
                lambda same: If(
                    condition=Var(local=same),
                    then=continue_with(bindings),
                    otherwise=failure(),
                ),
            )
        if isinstance(pattern, _Lit):
            return If(
                condition=self.primitive(
                    "eq_int", Var(local=value), _int(pattern.value)
                ),
                then=continue_with(bindings),
                otherwise=failure(),
            )
        if isinstance(pattern, _Bound):
            return self.case(
                Var(local=value),
                result,
                {
                    BOUND: lambda fields: If(
                        condition=self.primitive(
                            "eq_int", Var(local=fields[0]), _int(pattern.index)
                        ),
                        then=continue_with(bindings),
                        otherwise=failure(),
                    )
                },
                lambda: failure(),
                *path,
                "bound",
            )
        symbol = pattern.name if isinstance(pattern, _Atom) else pattern.symbol
        arguments = () if isinstance(pattern, _Atom) else pattern.args

        def fields_matched(fields: tuple[Local, ...]) -> Computation:
            """Match the constructor's fields in order.

            Parameters
            ----------
            fields : tuple[Local, ...]
                The fields.

            Returns
            -------
            Computation
                The matches, innermost the continuation.
            """

            def step(position: int, current: dict[str, Local]) -> Computation:
                """Match the fields from a position on.

                Parameters
                ----------
                position : int
                    The next field.
                current : dict[str, Local]
                    The bindings so far.

                Returns
                -------
                Computation
                    The match from that field on.
                """
                if position == len(fields):
                    return continue_with(current)
                return self.match(
                    fields[position],
                    arguments[position],
                    current,
                    result,
                    lambda extended: step(position + 1, extended),
                    *path,
                    position,
                    failed=failed,
                )

            return step(0, bindings)

        return self.case(
            Var(local=value),
            result,
            {symbol: fields_matched},
            lambda: failure(),
            *path,
            symbol,
        )

    # -- derivation -----------------------------------------------------

    def derivation(self, signature: ComputationSignature) -> NamedComputation:
        """Derive an item by one rule application or one axiom.

        A span-shaped system derives the items over a span: an axiom
        over a one-token span, or a rule with several premises whose
        spans split the conclusion's at chosen positions, each premise
        derived over its own smaller span; the unary rules then extend
        the item by the closure. The recursion thus follows the
        sentence, and only the closure, over one span, needs its own
        bound. Nonlexical systems use the weighted chart computations instead.

        Rules are grouped by the shape of their premises: the spans the
        premises cover relative to the conclusion's, or, without spans,
        their number. A derivation derives the group's premises once,
        then chooses a rule of the group and matches the premises
        against its patterns, so the sub-derivations are shared by every
        rule of the same shape.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation, which answers the conclusion of the chosen
            rule having added its weight.
        """
        if self.system.schema_categories_cell[0]:
            return self.schema_derivation(signature)
        if self.system.by_yield:
            return self.yield_derivation(signature)
        system = self.system
        item = system.item
        repeats: Local | None = None
        start = Local(name="start", type=INT)
        stop = Local(name="stop", type=INT)
        assert system.span_shaped_cell[0]
        repeats = Local(name="repeats", type=INT) if system.chained else None
        tokens = Local(name="tokens", type=self.inputs(signature)[-1])
        parameters: tuple[Local, ...] = (
            *((repeats,) if repeats is not None else ()),
            start,
            stop,
            tokens,
        )

        def recursion(low: Value, high: Value) -> Call:
            """Derive an item over a sub-span.

            Parameters
            ----------
            low : Value
                The sub-span's start.
            high : Value
                Its end.

            Returns
            -------
            Call
                The recursive call.
            """
            prefix = (Var(local=repeats),) if repeats is not None else ()
            return self.call(signature.name, (*prefix, low, high, Var(local=tokens)))

        axiom_alternative: Computation = If(
            condition=self.primitive(
                "eq_int",
                Var(local=stop),
                self.primitive("add_int", Var(local=start), _int(1)),
            ),
            then=self.call(system.name("axiom"), (Var(local=start), Var(local=tokens))),
            otherwise=self.fail(item, "derive", "lexical", "span"),
        )
        groups = [group for group in self.rule_groups() if group.arity > 1]
        alternatives = TensorValue(
            items=tuple(_int(index) for index in range(-1, len(groups))),
            result_type=tensor_type(
                INT, (IndexLiteral(value=len(groups) + 1, sort=NAT),)
            ),
        )

        def group_body(group_index: int, group: _Group) -> Computation:
            """Apply one rule of a group.

            Parameters
            ----------
            group_index : int
                The group's index.
            group : _Group
                The group.

            Returns
            -------
            Computation
                The application.
            """
            path: tuple[str | int, ...] = ("derive", "group", group_index)

            def choose_positions(
                remaining: list[str], chosen: dict[str, Local]
            ) -> Computation:
                """Choose a value for each split the group's premises need.

                Parameters
                ----------
                remaining : list[str]
                    The normalized position names still to choose.
                chosen : dict[str, Local]
                    The positions chosen so far, by normalized name.

                Returns
                -------
                Computation
                    The choices, innermost the premises.
                """
                if not remaining:
                    return premises(0, chosen, [])
                name = remaining[0]
                return self.bind(
                    "split",
                    INT,
                    self.choose_position(*path, "position", name),
                    lambda split: If(
                        condition=self.primitive(
                            "and",
                            self.primitive(
                                "lt_int", Var(local=start), Var(local=split)
                            ),
                            self.primitive("lt_int", Var(local=split), Var(local=stop)),
                        ),
                        then=choose_positions(remaining[1:], {**chosen, name: split}),
                        otherwise=self.fail(item, *path, "position", name),
                    ),
                )

            def positions(index: int, chosen: dict[str, Local]) -> dict[str, Local]:
                """The bindings a rule's position variables start from.

                Parameters
                ----------
                index : int
                    The rule's index.
                chosen : dict[str, Local]
                    The positions, by normalized name.

                Returns
                -------
                dict[str, Local]
                    Each position variable bound to its position.
                """
                return {
                    name: chosen[normalized]
                    for name, normalized in group.names[index].items()
                }

            def premises(
                position: int, chosen: dict[str, Local], derived: list[Local]
            ) -> Computation:
                """Derive the group's premises from a position on.

                Parameters
                ----------
                position : int
                    The next premise.
                chosen : dict[str, Local]
                    The positions, by normalized name.
                derived : list[Local]
                    The premises derived so far.

                Returns
                -------
                Computation
                    The derivations, innermost the rule choice.
                """
                if position == group.arity:
                    return self.apply_group(
                        group,
                        derived,
                        lambda index: positions(index, chosen),
                        lambda value: Return(value=value),
                        *path,
                    )
                low, high = group.spans[position]
                call = recursion(Var(local=chosen[low]), Var(local=chosen[high]))
                return self.bind(
                    "premise",
                    item,
                    call,
                    lambda value: premises(position + 1, chosen, [*derived, value]),
                )

            return choose_positions(list(group.splits), {"L": start, "R": stop})

        def dispatch(chosen: Local, index: int) -> Computation:
            """Select the chosen alternative by index.

            Parameters
            ----------
            chosen : Local
                The chosen index, ``-1`` for an axiom.
            index : int
                The group tested here.

            Returns
            -------
            Computation
                The chain of tests from this group on.
            """
            if index == len(groups):
                return self.fail(item, "derive", "none")
            return If(
                condition=self.primitive("eq_int", Var(local=chosen), _int(index)),
                then=group_body(index, groups[index]),
                otherwise=dispatch(chosen, index + 1),
            )

        base = self.bind(
            "alternative",
            INT,
            self.choose(
                alternatives,
                INT,
                IndexLiteral(value=len(groups) + 1, sort=NAT),
                "derive",
                "choose",
            ),
            lambda chosen: If(
                condition=self.primitive("eq_int", Var(local=chosen), _int(-1)),
                then=axiom_alternative,
                otherwise=dispatch(chosen, 0),
            ),
        )
        body: Computation = If(
            condition=self.primitive("lt_int", Var(local=start), Var(local=stop)),
            then=self.bind(
                "base", item, base, lambda found: self.closed(repeats, found)
            ),
            otherwise=self.fail(item, "derive", "span"),
        )
        return self.computation(signature, parameters, body)

    def schema_derivation(self, signature: ComputationSignature) -> NamedComputation:
        """Derive one expected category of a schema chart.

        The ordinary deduction computation enumerates an arbitrary item
        before matching it against a rule. A schema chart has a finite
        category algebra, so it can run goal-directed instead: every
        recursive call carries the ground category its parent rule needs.
        This is the recursive counterpart of the classic chart's sparse
        rule tensors and avoids enumerating every category tuple only to
        reject nearly all of them.

        Parameters
        ----------
        signature : ComputationSignature
            The goal-directed derivation signature.

        Returns
        -------
        NamedComputation
            The checked derivation computation.
        """
        system = self.system
        item = system.item
        fuel = Local(name="fuel", type=INT)
        start = Local(name="start", type=INT)
        stop = Local(name="stop", type=INT)
        expected = Local(name="expected", type=item)
        tokens = Local(name="tokens", type=signature.parameters[4])
        parameters = (fuel, start, stop, expected, tokens)
        spent = self.primitive("sub_int", Var(local=fuel), _int(1))

        def recursion(low: Value, high: Value, category: Pattern) -> Call:
            """Call the derivation on a ground premise category."""
            return self.call(
                signature.name,
                (spent, low, high, self.ground(category), Var(local=tokens)),
            )

        def answer() -> Return:
            """Return the expected category over the current span."""
            return Return(
                value=self.constructor(
                    SPAN,
                    (Var(local=start), Var(local=stop), Var(local=expected)),
                )
            )

        def weighted(index: int, rule: _Rule) -> Computation:
            """Add one concrete rule's learned log weight."""

            def added(weight: Value) -> Computation:
                """Add the carrier weight and return the chart item."""
                return Bind(
                    binder=system.local("added", UNIT),
                    first=self.add_weight(weight, "derive", "rule", index, "weight"),
                    then=answer(),
                )

            if not rule.learnable:
                return added(self.weight_of(None))
            return self.bind(
                "raw",
                REAL,
                self.parameter(
                    _string(f"{system.decl.name}.rule.{rule.source.name}:"),
                    "derive",
                    "rule",
                    index,
                    "parameter",
                ),
                lambda raw: added(self.weight_of(Var(local=raw))),
            )

        def apply_rule(index: int) -> Computation:
            """Apply one ground unary or binary schema rule."""
            rule = system.rules[index]

            def premises() -> Computation:
                """Derive the selected rule's ground premises."""
                if len(rule.premises) == 1:
                    premise = rule.premises[0]
                    assert isinstance(premise, _App) and premise.symbol == SPAN
                    return self.bind(
                        "premise",
                        item,
                        recursion(Var(local=start), Var(local=stop), premise.args[2]),
                        lambda _: weighted(index, rule),
                    )
                left, right = rule.premises
                assert isinstance(left, _App) and left.symbol == SPAN
                assert isinstance(right, _App) and right.symbol == SPAN
                split_index = system.local("k", INT)
                return self.bind(
                    "split",
                    INT,
                    self.choose(
                        Comprehension(
                            binder=split_index,
                            extent=self.extent,
                            body=Var(local=split_index),
                            result_type=tensor_type(INT, (self.extent,)),
                        ),
                        INT,
                        self.extent,
                        "derive",
                        "rule",
                        index,
                        "split",
                    ),
                    lambda split: If(
                        condition=self.primitive(
                            "and",
                            self.primitive(
                                "lt_int", Var(local=start), Var(local=split)
                            ),
                            self.primitive("lt_int", Var(local=split), Var(local=stop)),
                        ),
                        then=self.bind(
                            "left",
                            item,
                            recursion(Var(local=start), Var(local=split), left.args[2]),
                            lambda _: self.bind(
                                "right",
                                item,
                                recursion(
                                    Var(local=split), Var(local=stop), right.args[2]
                                ),
                                lambda _: weighted(index, rule),
                            ),
                        ),
                        otherwise=self.fail(item, "derive", "rule", index, "split"),
                    ),
                )

            return premises()

        lexical = If(
            condition=self.primitive(
                "eq_int",
                Var(local=stop),
                self.primitive("add_int", Var(local=start), _int(1)),
            ),
            then=self.call(
                system.name("axiom"),
                (Var(local=start), Var(local=expected), Var(local=tokens)),
            ),
            otherwise=self.fail(item, "derive", "lexical", "span"),
        )
        rules_by_category: list[list[int]] = [
            [] for _ in system.schema_categories_cell[0]
        ]
        for rule_index, rule in enumerate(system.rules):
            conclusion = rule.conclusion
            assert isinstance(conclusion, _App) and conclusion.symbol == SPAN
            category = conclusion.args[2]
            category_index = system.schema_categories_cell[0].index(category)
            rules_by_category[category_index].append(rule_index)

        def category_dispatch(category_index: int) -> Computation:
            """Choose only rules whose result is the expected category."""
            if category_index == len(system.schema_categories_cell[0]):
                return self.fail(item, "derive", "category")
            rule_indices = rules_by_category[category_index]
            alternatives = TensorValue(
                items=(_int(-1), *(_int(index) for index in rule_indices)),
                result_type=tensor_type(
                    INT, (IndexLiteral(value=len(rule_indices) + 1, sort=NAT),)
                ),
            )

            def rule_dispatch(chosen: Local, offset: int) -> Computation:
                """Dispatch one rule from the category-specific choice."""
                if offset == len(rule_indices):
                    return self.fail(item, "derive", "none")
                rule_index = rule_indices[offset]
                return If(
                    condition=self.primitive(
                        "eq_int", Var(local=chosen), _int(rule_index)
                    ),
                    then=apply_rule(rule_index),
                    otherwise=rule_dispatch(chosen, offset + 1),
                )

            category = system.schema_categories_cell[0][category_index]
            return self.bind(
                "same",
                BOOL,
                self.call(
                    system.name("eq"),
                    (Var(local=expected), self.ground(category)),
                ),
                lambda same: If(
                    condition=Var(local=same),
                    then=self.bind(
                        "alternative",
                        INT,
                        self.choose(
                            alternatives,
                            INT,
                            IndexLiteral(value=len(rule_indices) + 1, sort=NAT),
                            "derive",
                            "choose",
                            category_index,
                        ),
                        lambda chosen: If(
                            condition=self.primitive(
                                "eq_int", Var(local=chosen), _int(-1)
                            ),
                            then=lexical,
                            otherwise=rule_dispatch(chosen, 0),
                        ),
                    ),
                    otherwise=category_dispatch(category_index + 1),
                ),
            )

        body = If(
            condition=self.primitive("le_int", Var(local=fuel), _int(0)),
            then=self.fail(item, "derive", "fuel"),
            otherwise=category_dispatch(0),
        )
        return self.computation(signature, parameters, body)

    def conclude(
        self,
        index: int,
        rule: _Rule,
        bindings: dict[str, Local],
        then: Callable[[Value], Computation],
    ) -> Computation:
        """Instantiate a rule's conclusion, add its weight, and continue.

        Parameters
        ----------
        index : int
            The rule's index.
        rule : _Rule
            The rule.
        bindings : dict[str, Local]
            The complete bindings.
        then : Callable[[Value], Computation]
            What to run with the conclusion once its weight is added.

        Returns
        -------
        Computation
            The weighted conclusion.
        """
        system = self.system
        conclusion = self.instantiate(rule.conclusion, bindings)
        if system.depth_cell[0] is not None:
            # The declared depth bounds the conclusion's category, as the
            # agenda engine's side condition does; a deeper conclusion is
            # not derived.
            bound = system.depth_cell[0]
            return self.bind(
                "measured",
                INT,
                self.call(system.name("depth"), (conclusion,)),
                lambda measured: If(
                    condition=self.primitive(
                        "le_int", Var(local=measured), _int(bound)
                    ),
                    then=self.weighted_conclusion(
                        index, rule, bindings, conclusion, then
                    ),
                    otherwise=self.fail(system.item, "derive", "rule", index, "depth"),
                ),
            )
        return self.weighted_conclusion(index, rule, bindings, conclusion, then)

    def weighted_conclusion(
        self,
        index: int,
        rule: _Rule,
        bindings: dict[str, Local],
        conclusion: Value,
        then: Callable[[Value], Computation],
    ) -> Computation:
        """Add a rule's weight, then continue with its conclusion.

        Parameters
        ----------
        index : int
            The rule's index.
        rule : _Rule
            The rule.
        bindings : dict[str, Local]
            The complete bindings.
        conclusion : Value
            The instantiated conclusion.
        then : Callable[[Value], Computation]
            What to run with the conclusion once its weight is added.

        Returns
        -------
        Computation
            The weighted conclusion.
        """
        system = self.system

        def weighted(weight: Value) -> Computation:
            """Add the weight, then continue.

            Parameters
            ----------
            weight : Value
                The weight.

            Returns
            -------
            Computation
                The sequence.
            """
            return Bind(
                binder=system.local("added", UNIT),
                first=self.add_weight(weight, "derive", "rule", index, "weight"),
                then=then(conclusion),
            )

        if system.normalized_cell[0]:
            return self.production_weight(("rule", index), weighted)
        if system.semiring not in ("LogProb", "Viterbi") or not (
            rule.learnable or rule.parent is not None
        ):
            return weighted(self.weight_of(None))
        return self.rule_weight(rule, index, bindings, weighted)

    def depth_measure(self, signature: ComputationSignature) -> NamedComputation:
        """The constructor-tree depth of an item's category.

        Atoms, bound variables, and positions have depth zero; a ``span``
        has its category's depth; any other constructor adds one to the
        deepest of its item fields. This is the agenda engine's measure,
        which its ``depth`` side condition compares against the bound.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation.
        """
        system = self.system
        item = Local(name="item", type=system.item)

        def branch(symbol: str) -> Callable[[tuple[Local, ...]], Computation]:
            """The depth of one constructor applied to its fields.

            Parameters
            ----------
            symbol : str
                The constructor's symbol.

            Returns
            -------
            Callable[[tuple[Local, ...]], Computation]
                The branch body builder.
            """

            def body(fields: tuple[Local, ...]) -> Computation:
                """Measure the constructor's fields.

                Parameters
                ----------
                fields : tuple[Local, ...]
                    The fields.

                Returns
                -------
                Computation
                    The depth.
                """
                if symbol == SPAN:
                    return self.call(signature.name, (Var(local=fields[2]),))
                if symbol == BOUND:
                    return Return(value=_int(0))
                nested = [field for field in fields if field.type != INT]
                if not fields:
                    return Return(value=_int(0))

                def deepest(position: int, most: Value) -> Computation:
                    """Fold the item fields' depths into their maximum.

                    Parameters
                    ----------
                    position : int
                        The next item field.
                    most : Value
                        The deepest so far.

                    Returns
                    -------
                    Computation
                        One more than the maximum.
                    """
                    if position == len(nested):
                        return Return(value=self.primitive("add_int", most, _int(1)))
                    return self.bind(
                        "inner",
                        INT,
                        self.call(signature.name, (Var(local=nested[position]),)),
                        lambda inner: deepest(
                            position + 1,
                            self.primitive("max_int", most, Var(local=inner)),
                        ),
                    )

                return deepest(0, _int(0))

            return body

        body = self.case(
            Var(local=item),
            INT,
            {symbol: branch(symbol) for symbol in system.constructors},
            lambda: Return(value=_int(0)),
            "depth",
        )
        return self.computation(signature, (item,), body)

    def apply_group(
        self,
        group: _Group,
        derived: list[Local],
        positions: Callable[[int], dict[str, Local]],
        then: Callable[[Value], Computation],
        *path: str | int,
    ) -> Computation:
        """Choose a rule of a group and match its premises against derived items.

        Parameters
        ----------
        group : _Group
            The group, whose premises share one shape.
        derived : list[Local]
            The derived premise items, in premise order.
        positions : Callable[[int], dict[str, Local]]
            For a rule index, the bindings its position variables start
            from.
        then : Callable[[Value], Computation]
            What to run with the chosen rule's weighted conclusion.
        *path : str | int
            The group's path.

        Returns
        -------
        Computation
            The rule choice, each rule's matches innermost.
        """
        item = self.system.item

        def match_rule(entry: tuple[int, _Rule]) -> Computation:
            """Match the derived premises against one rule's patterns.

            Parameters
            ----------
            entry : tuple[int, _Rule]
                The rule and its index.

            Returns
            -------
            Computation
                The matches, innermost the conclusion.
            """
            index, rule = entry

            def step(position: int, current: dict[str, Local]) -> Computation:
                """Match the premises from a position on.

                Parameters
                ----------
                position : int
                    The next premise.
                current : dict[str, Local]
                    The bindings so far.

                Returns
                -------
                Computation
                    The match from that premise on.
                """
                if position == len(rule.premises):
                    return self.conclude(index, rule, current, then)
                return self.match(
                    derived[position],
                    rule.premises[position],
                    current,
                    item,
                    lambda extended: step(position + 1, extended),
                    *path,
                    "rule",
                    index,
                    "premise",
                    position,
                )

            return step(0, positions(index))

        if len(group.rules) == 1:
            return match_rule(group.rules[0])
        rule_alternatives = TensorValue(
            items=tuple(_int(index) for index in range(len(group.rules))),
            result_type=tensor_type(
                INT, (IndexLiteral(value=len(group.rules), sort=NAT),)
            ),
        )

        def dispatch(selected: Local, position: int) -> Computation:
            """Select the chosen rule by index.

            Parameters
            ----------
            selected : Local
                The chosen index.
            position : int
                The rule tested here.

            Returns
            -------
            Computation
                The chain of tests from this rule on.
            """
            if position == len(group.rules):
                return self.fail(item, *path, "rule", "none")
            return If(
                condition=self.primitive("eq_int", Var(local=selected), _int(position)),
                then=match_rule(group.rules[position]),
                otherwise=dispatch(selected, position + 1),
            )

        return self.bind(
            "rule",
            INT,
            self.choose(
                rule_alternatives,
                INT,
                IndexLiteral(value=len(group.rules), sort=NAT),
                *path,
                "rule",
            ),
            lambda selected: dispatch(selected, 0),
        )

    def choose_position(self, *path: str | int) -> Perform:
        """Choose an integer below the extent.

        Parameters
        ----------
        *path : str | int
            The request's path.

        Returns
        -------
        Perform
            The request, resumed once per integer ``0 .. n - 1``.
        """
        index = self.system.local("k", INT)
        return self.choose(
            Comprehension(
                binder=index,
                extent=self.extent,
                body=Var(local=index),
                result_type=tensor_type(INT, (self.extent,)),
            ),
            INT,
            self.extent,
            *path,
        )

    def closed(self, repeats: Local | None, base: Local) -> Computation:
        """Answer a derived item extended by a chain of unary rules.

        Parameters
        ----------
        repeats : Local | None
            How often a chain may revisit an item, for a cyclic system.
        base : Local
            The item derived without a trailing unary rule.

        Returns
        -------
        Computation
            The item itself without unary rules, else the closure's
            answer starting from it.
        """
        system = self.system
        if not system.closes:
            return Return(value=Var(local=base))
        if not system.chained:
            return self.call(system.name("close"), (Var(local=base),))
        assert repeats is not None
        nil = ConstructorValue(
            constructor=system.chain_constructors["nil"].id,
            static_arguments=(),
            fields=(),
            result_type=system.chain,
        )
        chain = ConstructorValue(
            constructor=system.chain_constructors["cons"].id,
            static_arguments=(),
            fields=(Var(local=base), nil),
            result_type=system.chain,
        )
        return self.call(
            system.name("close"), (Var(local=repeats), chain, Var(local=base))
        )

    def closure(self, signature: ComputationSignature) -> NamedComputation:
        """Extend an item by a chain of unary rules.

        The computation answers the item itself, or applies a unary rule
        to it and recurs on the conclusion. In a cyclic system it
        carries the chain of items visited and refuses a conclusion that
        the chain already holds more often than the repeat bound allows,
        so every chain is finite and the bound counts how often a chain
        may traverse a cycle.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation.
        """
        system = self.system
        item = system.item
        current = Local(name="current", type=item)
        if system.chained:
            repeats: Local | None = Local(name="repeats", type=INT)
            chain: Local | None = Local(name="chain", type=system.chain)
            parameters: tuple[Local, ...] = (repeats, chain, current)
        else:
            repeats = chain = None
            parameters = (current,)
        rules = system.closure
        alternatives = TensorValue(
            items=tuple(_int(index) for index in range(-1, len(rules))),
            result_type=tensor_type(
                INT, (IndexLiteral(value=len(rules) + 1, sort=NAT),)
            ),
        )

        def onward(conclusion: Value) -> Computation:
            """Recur on a unary rule's conclusion.

            Parameters
            ----------
            conclusion : Value
                The conclusion.

            Returns
            -------
            Computation
                The closure from the conclusion on.
            """
            if repeats is None or chain is None:
                return self.call(signature.name, (conclusion,))

            def counted(following: Local) -> Computation:
                """Count the conclusion in the chain, then recur or refuse.

                Parameters
                ----------
                following : Local
                    The conclusion.

                Returns
                -------
                Computation
                    The bounded recursion.
                """
                assert repeats is not None and chain is not None
                bound, visited = repeats, chain
                extended = ConstructorValue(
                    constructor=system.chain_constructors["cons"].id,
                    static_arguments=(),
                    fields=(Var(local=following), Var(local=visited)),
                    result_type=system.chain,
                )
                return self.bind(
                    "seen",
                    INT,
                    self.call(
                        system.name("count"), (Var(local=following), Var(local=visited))
                    ),
                    lambda seen: If(
                        condition=self.primitive(
                            "le_int", Var(local=seen), Var(local=bound)
                        ),
                        then=self.call(
                            signature.name,
                            (Var(local=bound), extended, Var(local=following)),
                        ),
                        otherwise=self.fail(item, "close", "repeat"),
                    ),
                )

            return self.bind("following", item, Return(value=conclusion), counted)

        def apply(position: int) -> Computation:
            """Apply one unary rule to the current item.

            Parameters
            ----------
            position : int
                The rule's position among the unary rules.

            Returns
            -------
            Computation
                The match and the recursion on the conclusion.
            """
            index = rules[position]
            rule = system.rules[index]
            return self.match(
                current,
                rule.premises[0],
                {},
                item,
                lambda bindings: self.conclude(index, rule, bindings, onward),
                "close",
                position,
            )

        def dispatch(chosen: Local, position: int) -> Computation:
            """Select the chosen unary rule by position.

            Parameters
            ----------
            chosen : Local
                The chosen position, ``-1`` to stop.
            position : int
                The rule tested here.

            Returns
            -------
            Computation
                The chain of tests from this rule on.
            """
            if position == len(rules):
                return self.fail(item, "close", "none")
            return If(
                condition=self.primitive("eq_int", Var(local=chosen), _int(position)),
                then=apply(position),
                otherwise=dispatch(chosen, position + 1),
            )

        body = self.bind(
            "step",
            INT,
            self.choose(
                alternatives,
                INT,
                IndexLiteral(value=len(rules) + 1, sort=NAT),
                "close",
                "choose",
            ),
            lambda chosen: If(
                condition=self.primitive("eq_int", Var(local=chosen), _int(-1)),
                then=Return(value=Var(local=current)),
                otherwise=dispatch(chosen, 0),
            ),
        )
        return self.computation(signature, parameters, body)

    def chain_count(self, signature: ComputationSignature) -> NamedComputation:
        """How often an item occurs in a chain.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation, comparing by the family's structural
            equality.
        """
        system = self.system
        item = Local(name="item", type=system.item)
        chain = Local(name="chain", type=system.chain)
        nil = system.chain_constructors["nil"]
        cons = system.chain_constructors["cons"]
        head = system.local("head", system.item)
        tail = system.local("tail", system.chain)
        cons_body = self.bind(
            "same",
            BOOL,
            self.call(system.name("eq"), (Var(local=item), Var(local=head))),
            lambda same: self.bind(
                "rest",
                INT,
                self.call(signature.name, (Var(local=item), Var(local=tail))),
                lambda rest: If(
                    condition=Var(local=same),
                    then=Return(
                        value=self.primitive("add_int", Var(local=rest), _int(1))
                    ),
                    otherwise=Return(value=Var(local=rest)),
                ),
            ),
        )
        branches = tuple(
            CaseBranch(
                constructor=constructor.id,
                static_arguments=(),
                fields=fields,
                body=body,
                scope=StaticScopeId.derive(
                    self.module,
                    (*self.path, "count"),
                    "branch",
                    position,
                    constructor.id,
                ),
            )
            for position, (constructor, fields, body) in enumerate(
                ((nil, (), Return(value=_int(0))), (cons, (head, tail), cons_body))
            )
        )
        body = Case(
            scrutinee=Var(local=chain),
            motive=CaseMotive(indices=(), result_type=INT),
            branches=branches,
        )
        return self.computation(signature, (item, chain), body)

    def frame_slots(self, head: str) -> list[int]:
        """The position slots of a head symbol, in slot order.

        Parameters
        ----------
        head : str
            The symbol.

        Returns
        -------
        list[int]
            The indices of its ``Int`` slots.
        """
        return [
            index
            for index, sort in enumerate(self.system.symbols[head])
            if sort == "int"
        ]

    def frame_width(self) -> int:
        """The most position slots any head symbol has.

        Returns
        -------
        int
            The number of frame arguments the derivation takes.
        """
        return max(len(self.frame_slots(head)) for head in self.system.heads)

    def choose_boundary(
        self, then: Callable[[Local], Computation], *path: str | int
    ) -> Computation:
        """Choose a sentence boundary, an integer from zero to the extent.

        Parameters
        ----------
        then : Callable[[Local], Computation]
            What to run with the chosen boundary.
        *path : str | int
            The choice's path.

        Returns
        -------
        Computation
            The choice: zero, or one more than an integer below the
            extent.
        """
        return self.bind("boundary", INT, self.boundary(*path), then)

    def boundary(self, *path: str | int) -> Computation:
        """Choose any sentence boundary, an integer from zero to the extent.

        Parameters
        ----------
        *path : str | int
            The choice's path.

        Returns
        -------
        Computation
            Zero, or one more than an integer below the extent.
        """
        sides = TensorValue(
            items=(_int(0), _int(1)),
            result_type=tensor_type(INT, (IndexLiteral(value=2, sort=NAT),)),
        )
        return self.bind(
            "side",
            INT,
            self.choose(sides, INT, IndexLiteral(value=2, sort=NAT), *path, "side"),
            lambda side: If(
                condition=self.primitive("eq_int", Var(local=side), _int(0)),
                then=Return(value=_int(0)),
                otherwise=self.bind(
                    "offset",
                    INT,
                    self.choose_position(*path, "offset"),
                    lambda offset: Return(
                        value=self.primitive("add_int", Var(local=offset), _int(1))
                    ),
                ),
            ),
        )

    def choose_between(
        self,
        low: Value,
        high: Value,
        then: Callable[[Local], Computation],
        *path: str | int,
    ) -> Computation:
        """Choose a sentence boundary that may lie between two others.

        The choice offers ``low + k`` for every ``k`` below the extent,
        which covers every boundary from ``low`` to ``high`` unless the
        two are the sentence's own ends; only then does it offer every
        boundary. Candidates outside the range are left for the caller's
        order check to refuse.

        Parameters
        ----------
        low : Value
            The least boundary the caller accepts.
        high : Value
            The greatest.
        then : Callable[[Local], Computation]
            What to run with the chosen boundary.
        *path : str | int
            The choice's path.

        Returns
        -------
        Computation
            The choice.
        """
        offset = self.bind(
            "offset",
            INT,
            self.choose_position(*path, "offset"),
            lambda chosen: Return(
                value=self.primitive("add_int", low, Var(local=chosen))
            ),
        )
        boundary = If(
            condition=self.primitive(
                "lt_int", self.primitive("sub_int", high, low), self.length()
            ),
            then=offset,
            otherwise=self.bind(
                "any",
                INT,
                self.boundary(*path, "any"),
                lambda any_: Return(value=Var(local=any_)),
            ),
        )
        return self.bind("boundary", INT, boundary, then)

    def yield_derivation(self, signature: ComputationSignature) -> NamedComputation:
        """Derive an item from its head and the positions of its yield.

        A lexical system whose items are not all spans, such as a
        multiple context-free grammar, derives an item given its head
        symbol and its frame, the values of its position slots, read in
        pairs as the components of its yield. An axiom is the item over
        a one-token span; a rule whose conclusion has the head binds its
        conclusion's positions to the frame, chooses the positions inside
        each component where its premises' components meet, in order,
        and derives each premise from its own head and frame. Every
        derivable item covers at least one token with components that run
        forward, so a frame that does not is refused at once; the rules
        are linear, so a rule with several premises derives each over
        fewer tokens than its conclusion's, and the unary rules form no
        cycle, so the recursion is finite.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation.
        """
        system = self.system
        item = system.item
        width = self.frame_width()
        head = Local(name="head", type=INT)
        frame = tuple(Local(name=f"p{index}", type=INT) for index in range(width))
        tokens = Local(name="tokens", type=self.inputs(signature)[-1])
        parameters = (head, *frame, tokens)

        def recursion(symbol: str, values: list[Value]) -> Call:
            """Derive an item with a head and a frame.

            Parameters
            ----------
            symbol : str
                The head symbol.
            values : list[Value]
                The frame's values, in slot order.

            Returns
            -------
            Call
                The recursive call.
            """
            padded = [*values, *([_int(0)] * (width - len(values)))]
            return self.call(
                signature.name,
                (_int(system.heads.index(symbol)), *padded, Var(local=tokens)),
            )

        def value_of(node: _Node, bindings: dict[str, Local]) -> Value:
            """The value of a yield position under bindings.

            Parameters
            ----------
            node : _Node
                A variable's name or a literal.
            bindings : dict[str, Local]
                The variables' values.

            Returns
            -------
            Value
                The position.
            """
            return _int(node) if isinstance(node, int) else Var(local=bindings[node])

        def rule_body(index: int, rule: _Rule, symbol: str) -> Computation:
            """Derive the head's item by one rule.

            Parameters
            ----------
            index : int
                The rule's index.
            rule : _Rule
                The rule, whose conclusion has the head.
            symbol : str
                The head symbol.

            Returns
            -------
            Computation
                The frame binding, the choice of the inner positions, the
                premises' derivations, and the rule's match.
            """
            path: tuple[str | int, ...] = ("derive", "rule", index)
            if _has_positions(system, rule.conclusion):
                plan = _yield_plan(system, rule)
                assert not isinstance(plan, str)
                paths, loose = plan
                conclusion = _components(system, rule.conclusion)
                assert not isinstance(conclusion, str)
            else:
                # An erasing rule: every premise position ranges over
                # the sentence.
                paths, loose = (), tuple(self.position_variables(rule))
                conclusion = []
            checks: list[Value] = []
            bound: dict[str, Local] = {}
            for position, (start, end) in enumerate(conclusion):
                for offset, node in ((0, start), (1, end)):
                    given = frame[2 * position + offset]
                    if isinstance(node, int):
                        checks.append(
                            self.primitive("eq_int", Var(local=given), _int(node))
                        )
                    elif node in bound:
                        checks.append(
                            self.primitive(
                                "eq_int", Var(local=bound[node]), Var(local=given)
                            )
                        )
                    else:
                        bound[node] = given
            low: Value = Var(local=frame[0]) if conclusion else _int(0)
            high: Value = Var(local=frame[0]) if conclusion else self.length()
            for position in range(1, 2 * len(conclusion)):
                low = self.primitive("min_int", low, Var(local=frame[position]))
                high = self.primitive("max_int", high, Var(local=frame[position]))
            steps: list[tuple[_Node, _Node, _Node]] = [
                (component[step - 1], component[step], component[-1])
                for component in paths
                for step in range(1, len(component) - 1)
            ]
            group = _Group(
                arity=len(rule.premises),
                spans=(),
                splits=(),
                rules=[(index, rule)],
                names={},
            )

            def inner(position: int, current: dict[str, Local]) -> Computation:
                """Place the positions inside each component, in order.

                Parameters
                ----------
                position : int
                    The next inner position.
                current : dict[str, Local]
                    The position bindings so far.

                Returns
                -------
                Computation
                    The choices, innermost the loose positions.
                """
                if position == len(steps):
                    return spread(0, current)
                previous, node, end = steps[position]
                after = value_of(previous, current)
                before = value_of(end, current)

                def placed(chosen: Value, extended: dict[str, Local]) -> Computation:
                    """Check that a position lies in order, then continue.

                    Parameters
                    ----------
                    chosen : Value
                        The position.
                    extended : dict[str, Local]
                        The bindings with it.

                    Returns
                    -------
                    Computation
                        The check and the rest.
                    """
                    return If(
                        condition=self.primitive(
                            "and",
                            self.primitive("le_int", after, chosen),
                            self.primitive("le_int", chosen, before),
                        ),
                        then=inner(position + 1, extended),
                        otherwise=self.fail(item, *path, "order", position),
                    )

                if isinstance(node, int) or node in current:
                    return placed(value_of(node, current), current)
                return self.choose_between(
                    after,
                    before,
                    lambda chosen: placed(Var(local=chosen), {**current, node: chosen}),
                    *path,
                    "position",
                    node,
                )

            def spread(position: int, current: dict[str, Local]) -> Computation:
                """Choose the positions only an empty premise component has.

                Parameters
                ----------
                position : int
                    The next such position.
                current : dict[str, Local]
                    The position bindings so far.

                Returns
                -------
                Computation
                    The choices, innermost the premises.
                """
                if position == len(loose):
                    return If(
                        condition=nonempty(current),
                        then=premises(0, [], current),
                        otherwise=self.fail(item, *path, "premises"),
                    )
                name = loose[position]
                if name in current:
                    return spread(position + 1, current)
                return self.choose_boundary(
                    lambda chosen: If(
                        condition=self.primitive(
                            "and",
                            self.primitive("le_int", low, Var(local=chosen)),
                            self.primitive("le_int", Var(local=chosen), high),
                        ),
                        then=spread(position + 1, {**current, name: chosen}),
                        otherwise=self.fail(item, *path, "loose", name),
                    ),
                    *path,
                    "position",
                    name,
                )

            def premises(
                position: int, derived: list[Local], current: dict[str, Local]
            ) -> Computation:
                """Derive the premises from a position on.

                Parameters
                ----------
                position : int
                    The next premise.
                derived : list[Local]
                    The premises derived so far.
                current : dict[str, Local]
                    The position bindings.

                Returns
                -------
                Computation
                    The derivations, innermost the rule's match.
                """
                if position == len(rule.premises):
                    return self.apply_group(
                        group,
                        derived,
                        lambda _: dict(current),
                        lambda value: Return(value=value),
                        *path,
                    )
                pattern = rule.premises[position]
                values: list[Value] = []
                if _has_positions(system, pattern):
                    components = _components(system, pattern)
                    assert not isinstance(components, str)
                    values = [
                        value_of(node, current)
                        for component in components
                        for node in component
                    ]
                return self.bind(
                    "premise",
                    item,
                    recursion(_head(pattern), values),
                    lambda value: premises(position + 1, [*derived, value], current),
                )

            def nonempty(current: dict[str, Local]) -> Value:
                """Whether every premise's frame runs forward over a token.

                Checked before any premise is derived: every derivable item
                covers a token, and the premises' tokens add up to the
                conclusion's, so each premise covers fewer than it does.

                Parameters
                ----------
                current : dict[str, Local]
                    The position bindings.

                Returns
                -------
                Value
                    The test.
                """
                condition: Value = LiteralValue(value=True, type=BOOL)
                for pattern in rule.premises:
                    if not _has_positions(system, pattern):
                        continue
                    components = _components(system, pattern)
                    assert not isinstance(components, str)
                    covered: Value = _int(0)
                    for start, end in components:
                        low_end = value_of(start, current)
                        high_end = value_of(end, current)
                        condition = self.primitive(
                            "and",
                            condition,
                            self.primitive("le_int", low_end, high_end),
                        )
                        covered = self.primitive(
                            "add_int",
                            covered,
                            self.primitive("sub_int", high_end, low_end),
                        )
                    condition = self.primitive(
                        "and", condition, self.primitive("lt_int", _int(0), covered)
                    )
                return condition

            body = inner(0, dict(bound))
            if not checks:
                return body
            condition = checks[0]
            for check in checks[1:]:
                condition = self.primitive("and", condition, check)
            return If(
                condition=condition,
                then=body,
                otherwise=self.fail(item, *path, "frame"),
            )

        def head_body(symbol: str) -> Computation:
            """Derive an item with one head: an axiom or a rule concluding it.

            Parameters
            ----------
            symbol : str
                The head symbol.

            Returns
            -------
            Computation
                The frame's check and the choice among the head's
                derivations.
            """
            components = len(self.frame_slots(symbol)) // 2
            concluding = [
                (index, system.rules[index])
                for index in system.live
                if _head(system.rules[index].conclusion) == symbol
            ]
            options: list[Callable[[], Computation]] = []
            if symbol == SPAN:
                options.append(
                    lambda: If(
                        condition=self.primitive(
                            "eq_int",
                            Var(local=frame[1]),
                            self.primitive("add_int", Var(local=frame[0]), _int(1)),
                        ),
                        then=self.call(
                            system.name("axiom"),
                            (Var(local=frame[0]), Var(local=tokens)),
                        ),
                        otherwise=self.fail(item, "derive", "axiom", "span"),
                    )
                )
            for index, rule in concluding:
                options.append(
                    lambda index=index, rule=rule: rule_body(index, rule, symbol)
                )
            if not options:
                return self.fail(item, "derive", "head", symbol)

            def dispatch(chosen: Local, position: int) -> Computation:
                """Select the chosen derivation by position.

                Parameters
                ----------
                chosen : Local
                    The chosen position.
                position : int
                    The derivation tested here.

                Returns
                -------
                Computation
                    The chain of tests from this derivation on.
                """
                if position == len(options) - 1:
                    return options[position]()
                return If(
                    condition=self.primitive(
                        "eq_int", Var(local=chosen), _int(position)
                    ),
                    then=options[position](),
                    otherwise=dispatch(chosen, position + 1),
                )

            if len(options) == 1:
                derivations = options[0]()
            else:
                derivations = self.bind(
                    "derivation",
                    INT,
                    self.choose(
                        TensorValue(
                            items=tuple(_int(index) for index in range(len(options))),
                            result_type=tensor_type(
                                INT, (IndexLiteral(value=len(options), sort=NAT),)
                            ),
                        ),
                        INT,
                        IndexLiteral(value=len(options), sort=NAT),
                        "derive",
                        "head",
                        symbol,
                    ),
                    lambda chosen: dispatch(chosen, 0),
                )
            if not components:
                return derivations
            covered: Value = _int(0)
            forward: Value = LiteralValue(value=True, type=BOOL)
            for position in range(components):
                start, end = (
                    Var(local=frame[2 * position]),
                    Var(local=frame[2 * position + 1]),
                )
                forward = self.primitive(
                    "and", forward, self.primitive("le_int", start, end)
                )
                covered = self.primitive(
                    "add_int", covered, self.primitive("sub_int", end, start)
                )
            return If(
                condition=self.primitive(
                    "and", forward, self.primitive("lt_int", _int(0), covered)
                ),
                then=derivations,
                otherwise=self.fail(item, "derive", "head", symbol, "frame"),
            )

        def heads(position: int) -> Computation:
            """Dispatch on the requested head.

            Parameters
            ----------
            position : int
                The head tested here.

            Returns
            -------
            Computation
                The chain of tests from this head on.
            """
            if position == len(system.heads):
                return self.fail(item, "derive", "head", "none")
            return If(
                condition=self.primitive("eq_int", Var(local=head), _int(position)),
                then=head_body(system.heads[position]),
                otherwise=heads(position + 1),
            )

        return self.computation(signature, parameters, heads(0))

    def length(self) -> Value:
        """The sentence's length, the extent counted as a sum of ones.

        Returns
        -------
        Value
            The length.
        """
        return Reduction(
            operator="sum",
            value=Comprehension(
                binder=self.system.local("k", INT),
                extent=self.extent,
                body=_int(1),
                result_type=tensor_type(INT, (self.extent,)),
            ),
            result_type=INT,
        )

    def yield_root(self, tokens: Local) -> Computation:
        """Derive a goal item of a yield-directed system.

        A goal span covers the whole sentence, so its frame is the
        sentence's bounds, as the agenda engine's goal spans are. A goal
        item headed by the start symbol has any frame whose components
        lie in the sentence; without a start symbol every head is a goal
        head.

        Parameters
        ----------
        tokens : Local
            The sentence.

        Returns
        -------
        Computation
            The derivation of a candidate goal item.
        """
        system = self.system
        item = system.item
        width = self.frame_width()
        start = system.start
        roots: list[str] = []
        if start is None:
            roots = list(system.heads)
        else:
            if start in system.constructors and not system.constructors[start].fields:
                roots.append(SPAN)
            if start in system.heads and start not in roots:
                roots.append(start)

        def derive(symbol: str, values: list[Value]) -> Call:
            """Call the derivation on a head and a frame.

            Parameters
            ----------
            symbol : str
                The head.
            values : list[Value]
                The frame.

            Returns
            -------
            Call
                The call.
            """
            padded = [*values, *([_int(0)] * (width - len(values)))]
            return self.call(
                system.name("derive"),
                (_int(system.heads.index(symbol)), *padded, Var(local=tokens)),
            )

        def root(symbol: str) -> Computation:
            """Derive a goal candidate with one head.

            Parameters
            ----------
            symbol : str
                The head.

            Returns
            -------
            Computation
                The derivation.
            """
            if symbol == SPAN:
                return derive(SPAN, [_int(0), self.length()])
            slots = self.frame_slots(symbol)

            def frame(values: list[Value]) -> Computation:
                """Choose the frame's values, then derive.

                Parameters
                ----------
                values : list[Value]
                    The values chosen so far.

                Returns
                -------
                Computation
                    The choices, innermost the derivation.
                """
                if len(values) == len(slots):
                    return derive(symbol, values)
                return self.choose_boundary(
                    lambda chosen: frame([*values, Var(local=chosen)]),
                    "entry",
                    "frame",
                    symbol,
                    len(values),
                )

            return frame([])

        if not roots:
            return self.fail(item, "entry", "root")
        if len(roots) == 1:
            return root(roots[0])

        def dispatch(chosen: Local, position: int) -> Computation:
            """Select the chosen root head.

            Parameters
            ----------
            chosen : Local
                The chosen position.
            position : int
                The head tested here.

            Returns
            -------
            Computation
                The chain of tests from this head on.
            """
            if position == len(roots) - 1:
                return root(roots[position])
            return If(
                condition=self.primitive("eq_int", Var(local=chosen), _int(position)),
                then=root(roots[position]),
                otherwise=dispatch(chosen, position + 1),
            )

        return self.bind(
            "root",
            INT,
            self.choose(
                TensorValue(
                    items=tuple(_int(index) for index in range(len(roots))),
                    result_type=tensor_type(
                        INT, (IndexLiteral(value=len(roots), sort=NAT),)
                    ),
                ),
                INT,
                IndexLiteral(value=len(roots), sort=NAT),
                "entry",
                "root",
            ),
            lambda chosen: dispatch(chosen, 0),
        )

    def rule_groups(self) -> list[_Group]:
        """Group the rules by the shape of their premises.

        Returns
        -------
        list[_Group]
            For a span-shaped system, one group per pattern of premise
            spans relative to the conclusion's, the conclusion's start
            and end normalized to ``L`` and ``R`` and each further
            position to ``k0``, ``k1``, and so on in order of first
            occurrence; otherwise one group per premise count.
        """
        groups: dict[tuple[object, ...], _Group] = {}
        for index, rule in enumerate(self.system.rules):
            if not self.system.span_shaped_cell[0]:
                key: tuple[object, ...] = (len(rule.premises),)
                spans: tuple[tuple[str, str], ...] = ()
                names: dict[str, str] = {}
            else:
                conclusion = rule.conclusion
                assert isinstance(conclusion, _App)
                first, second = conclusion.args[0], conclusion.args[1]
                assert isinstance(first, _Wild) and isinstance(second, _Wild)
                names = {first.name: "L", second.name: "R"}
                for name in self.position_variables(rule):
                    if name not in names:
                        names[name] = f"k{len(names) - 2}"
                spans_list: list[tuple[str, str]] = []
                for premise in rule.premises:
                    assert isinstance(premise, _App)
                    low, high = premise.args[0], premise.args[1]
                    assert isinstance(low, _Wild) and isinstance(high, _Wild)
                    spans_list.append((names[low.name], names[high.name]))
                spans = tuple(spans_list)
                key = spans
            group = groups.get(key)
            if group is None:
                splits = tuple(
                    name
                    for name in dict.fromkeys(
                        normalized for span in spans for normalized in span
                    )
                    if name not in ("L", "R")
                )
                group = _Group(
                    arity=len(rule.premises),
                    spans=spans,
                    splits=splits,
                    rules=[],
                    names={},
                )
                groups[key] = group
            group.rules.append((index, rule))
            group.names[index] = names
        return list(groups.values())

    def position_variables(self, rule: _Rule) -> list[str]:
        """The variables a rule uses in integer slots, in first-occurrence order.

        Parameters
        ----------
        rule : _Rule
            The rule.

        Returns
        -------
        list[str]
            The names.
        """
        return _rule_positions(self.system, rule)

    def rule_weight(
        self,
        rule: _Rule,
        index: int,
        bindings: dict[str, Local],
        continue_with: Callable[[Value], Computation],
    ) -> Computation:
        """Read a rule's learned weight on its bindings, then continue.

        The weight is keyed by the rule's name and its non-position
        bindings (categories and logical forms) rendered in name order,
        which is the keying the agenda engine uses: chart positions
        never enter the key, so one weight serves the rule at every
        position and sentence length. A bounded rule's raw parameter is
        mapped below zero, and a parented rule adds its parent's weight
        on the same bindings.

        Parameters
        ----------
        rule : _Rule
            The rule.
        index : int
            The rule's index.
        bindings : dict[str, Local]
            The complete bindings.
        continue_with : Callable[[Value], Computation]
            What to run with the weight, a carrier value.

        Returns
        -------
        Computation
            The reads, innermost the continuation.
        """
        positions = set(self.position_variables(rule))
        names = sorted(name for name in bindings if name not in positions)

        def render(
            position: int, rendered: list[Value], then: Callable[[Value], Computation]
        ) -> Computation:
            """Render the bindings from a position on into one key suffix.

            Parameters
            ----------
            position : int
                The next binding.
            rendered : list[Value]
                The bindings rendered so far.
            then : Callable[[Value], Computation]
                What to run with the suffix.

            Returns
            -------
            Computation
                The renders, innermost the continuation.
            """
            if position == len(names):
                suffix: Value = _string("")
                for part_index, part in enumerate(rendered):
                    if part_index:
                        suffix = self.primitive("concat", suffix, _string("|"))
                    suffix = self.primitive("concat", suffix, part)
                return then(suffix)
            local = bindings[names[position]]
            return self.bind(
                "text",
                STRING,
                self.call(self.system.name("show"), (Var(local=local),)),
                lambda text: render(position + 1, [*rendered, Var(local=text)], then),
            )

        bounded = [
            item
            for item in self.system.rules
            if item.bounded and (item.learnable or item.parent is not None)
        ]
        cap = math.log(len(bounded)) if len(bounded) > 1 else 0.0

        def own(
            chain: _Rule, suffix: Value, then: Callable[[Value], Computation]
        ) -> Computation:
            """Read one rule's own weight on the bindings, then continue.

            Parameters
            ----------
            chain : _Rule
                The rule.
            suffix : Value
                The rendered bindings.
            then : Callable[[Value], Computation]
                What to run with the weight, a ``Real``.

            Returns
            -------
            Computation
                The read, or the identity when the rule learns nothing.
            """
            if not chain.learnable:
                return then(LiteralValue(value=0.0, type=REAL))
            key = self.primitive(
                "concat",
                _string(f"{self.system.decl.name}.rule.{chain.source.name}:"),
                suffix,
            )

            def bounded_or_raw(raw: Local) -> Value:
                """The rule's weight from its raw parameter.

                Parameters
                ----------
                raw : Local
                    The parameter.

                Returns
                -------
                Value
                    ``-softplus(raw) - log(n)`` for a bounded rule, the
                    parameter itself otherwise.
                """
                if not chain.bounded:
                    return Var(local=raw)
                return self.primitive(
                    "sub_real",
                    self.primitive(
                        "neg_real", self.primitive("softplus", Var(local=raw))
                    ),
                    LiteralValue(value=cap, type=REAL),
                )

            return self.bind(
                "raw",
                REAL,
                self.parameter(
                    key, "derive", "rule", index, "parameter", chain.source.name
                ),
                lambda raw: then(bounded_or_raw(raw)),
            )

        by_name = {item.source.name: item for item in self.system.rules}

        def chain(
            current: _Rule, suffix: Value, total: Value, seen: set[str]
        ) -> Computation:
            """Add the parent chain's weights to a running total.

            Parameters
            ----------
            current : _Rule
                The rule whose parent is read next.
            suffix : Value
                The rendered bindings.
            total : Value
                The weight so far.
            seen : set[str]
                The rules read, to refuse a cycle.

            Returns
            -------
            Computation
                The reads, innermost the continuation with the total.
            """
            if current.parent is None:
                return continue_with(self.weight_of(total))
            if current.parent in seen:
                self.elaborator._fail(
                    rule.source,
                    f"rule {rule.source.name!r}: parent chain cycles at "
                    f"{current.parent!r}",
                    code="qiec-program",
                )
            parent = by_name[current.parent]
            return own(
                parent,
                suffix,
                lambda weight: chain(
                    parent,
                    suffix,
                    self.primitive("add_real", total, weight),
                    seen | {parent.source.name},
                ),
            )

        return render(
            0,
            [],
            lambda suffix: own(
                rule,
                suffix,
                lambda weight: chain(rule, suffix, weight, {rule.source.name}),
            ),
        )

    def production_weight(
        self,
        key: ProductionKey,
        continue_with: Callable[[Value], Computation],
    ) -> Computation:
        """Read a production's log-probability in its group, then continue.

        The production's group holds every rule and lexicon entry that
        rewrites its left-hand side. In the weighted variant the
        production's log-probability is its entry of the table the
        program supplies. Otherwise each member's free log-weight is its
        learned parameter, read under the name the agenda engine keys it
        by, or zero for a member that is not learnable, and the
        production's weight is its own free log-weight minus the group's
        log-sum-exp, which the group's normalizer computation answers:
        the entry of the group's ``log_softmax`` the agenda engine reads.

        Parameters
        ----------
        key : ProductionKey
            The production's address.
        continue_with : Callable[[Value], Computation]
            What to run with the weight, a carrier value.

        Returns
        -------
        Computation
            The reads, innermost the continuation.
        """
        system = self.system
        index, group = next(
            (index, group)
            for index, group in enumerate(system.groups_cell[0])
            if any(production.key == key for production in group.productions)
        )
        if len(group.productions) == 1:
            # A production alone in its group has probability one.
            return continue_with(self.weight_of(LiteralValue(value=0.0, type=REAL)))
        if self.table is not None:
            entry = Gather(
                value=Var(local=self.table),
                index=_int(system.table[key]),
                result_type=REAL,
            )
            return continue_with(self.weight_of(entry))
        production = next(item for item in group.productions if item.key == key)
        if not any(member.learnable for member in group.productions):
            # Every member has free log-weight zero: a uniform group.
            return continue_with(
                self.weight_of(
                    LiteralValue(value=-math.log(len(group.productions)), type=REAL)
                )
            )

        def normalized(own: Value) -> Computation:
            """Subtract the group's log-sum-exp from the own free weight.

            Parameters
            ----------
            own : Value
                The production's free log-weight.

            Returns
            -------
            Computation
                The normalizer's call, then the continuation.
            """
            return self.bind(
                "normalizer",
                REAL,
                self.call(system.name(f"lhs{index}"), ()),
                lambda normalizer: continue_with(
                    self.weight_of(
                        self.primitive("sub_real", own, Var(local=normalizer))
                    )
                ),
            )

        if not production.learnable:
            return normalized(LiteralValue(value=0.0, type=REAL))
        return self.bind(
            "free",
            REAL,
            self.parameter(
                _string(self.parameter_name(production)),
                "productions",
                group.label,
                "parameter",
            ),
            lambda free: normalized(Var(local=free)),
        )

    def parameter_name(self, production: Production) -> str:
        """The ``Param`` name of a learnable production's free weight.

        Parameters
        ----------
        production : Production
            The production.

        Returns
        -------
        str
            ``<D>.rule.<r>:`` for a rule, ``<D>.lex.<i>`` for an entry.
        """
        system = self.system
        if production.kind == "rule":
            return (
                f"{system.decl.name}.rule.{system.rules[production.index].source.name}:"
            )
        return f"{system.decl.name}.lex.{production.index}"

    def normalizer(
        self, signature: ComputationSignature, group: ProductionGroup
    ) -> NamedComputation:
        """The log-sum-exp of one group's free log-weights.

        The free weights are read from ``Param`` (zero for a member that
        is not learnable) and summed stably about their maximum.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.
        group : ProductionGroup
            The group, of several productions.

        Returns
        -------
        NamedComputation
            The computation.
        """

        def read(position: int, values: list[Value]) -> Computation:
            """Read the free log-weights from a member on.

            Parameters
            ----------
            position : int
                The next member.
            values : list[Value]
                The free log-weights read so far.

            Returns
            -------
            Computation
                The reads, innermost the log-sum-exp.
            """
            if position == len(group.productions):
                return Return(value=self.log_sum_exp(values))
            production = group.productions[position]
            if not production.learnable:
                return read(position + 1, [*values, LiteralValue(value=0.0, type=REAL)])
            return self.bind(
                "free",
                REAL,
                self.parameter(
                    _string(self.parameter_name(production)),
                    "normalizers",
                    group.label,
                    position,
                ),
                lambda free: read(position + 1, [*values, Var(local=free)]),
            )

        return self.computation(signature, (), read(0, []))

    def log_sum_exp(self, values: list[Value]) -> Value:
        """``log(sum(exp(v)))`` computed about the largest value.

        Parameters
        ----------
        values : list[Value]
            Reals, at least one.

        Returns
        -------
        Value
            The log-sum-exp, a ``Real``.
        """
        peak = values[0]
        for value in values[1:]:
            peak = self.primitive("max_real", peak, value)
        total: Value | None = None
        for value in values:
            term = self.primitive("exp", self.primitive("sub_real", value, peak))
            total = term if total is None else self.primitive("add_real", total, term)
        assert total is not None
        return self.primitive("add_real", peak, self.primitive("log", total))

    # -- entry ------------------------------------------------------------

    def entry(
        self, signature: ComputationSignature, handlers: tuple[HandlerDef, HandlerDef]
    ) -> NamedComputation:
        """The deduction's entry: the inside weight of the goal.

        A cyclic system's entry iterates the closure's repeat bound from
        zero until two successive inside weights agree within the
        declared tolerance (exactly, without one), which is how the
        chart's Kleene-star aggregation converges on the same cycles.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.
        handlers : tuple[HandlerDef, HandlerDef]
            The search and collection handlers.

        Returns
        -------
        NamedComputation
            The computation, which handles the weight instance with the
            collector and the search instance with the searcher around a
            derivation of the goal.
        """
        system = self.system
        parameters = tuple(
            Local(name=name, type=type_)
            for name, type_ in zip(
                ("tokens",) if system.entries else ("axioms", "weights"),
                self.inputs(signature),
                strict=True,
            )
        )
        if not system.chained:
            body = self.searched(parameters, None, handlers)
            return self.computation(signature, parameters, body)
        inputs = tuple(Var(local=parameter) for parameter in parameters)
        body = self.bind(
            "first",
            system.carrier,
            self.call(system.name("inside"), (_int(0), *inputs)),
            lambda first: self.call(
                system.name("converge"), (_int(1), Var(local=first), *inputs)
            ),
        )
        return self.computation(signature, parameters, body)

    def inside(
        self, signature: ComputationSignature, handlers: tuple[HandlerDef, HandlerDef]
    ) -> NamedComputation:
        """The inside weight of the goal under a repeat bound on chains.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.
        handlers : tuple[HandlerDef, HandlerDef]
            The search and collection handlers.

        Returns
        -------
        NamedComputation
            The computation.
        """
        repeats = Local(name="repeats", type=INT)
        tokens = Local(name="tokens", type=signature.parameters[1])
        body = self.searched((tokens,), Var(local=repeats), handlers)
        return self.computation(signature, (repeats, tokens), body)

    def convergence(self, signature: ComputationSignature) -> NamedComputation:
        """Raise the repeat bound until the inside weight converges.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's registered signature.

        Returns
        -------
        NamedComputation
            The computation, answering the first inside weight within
            the tolerance of the one under the bound below it.
        """
        system = self.system
        repeats = Local(name="repeats", type=INT)
        previous = Local(name="previous", type=system.carrier)
        tokens = Local(name="tokens", type=signature.parameters[2])

        def converged(current: Local) -> Value:
            """Whether a weight is within the tolerance of the previous one.

            Parameters
            ----------
            current : Local
                The weight under the current bound.

            Returns
            -------
            Value
                The test.
            """
            if system.semiring == "Boolean":
                return self.primitive(
                    "eq_bool", Var(local=current), Var(local=previous)
                )
            if system.semiring == "Counting":
                return self.primitive("eq_int", Var(local=current), Var(local=previous))
            now = self.primitive("weight_value", Var(local=current))
            before = self.primitive("weight_value", Var(local=previous))
            return self.primitive(
                "or",
                self.primitive("eq_real", now, before),
                self.primitive(
                    "le_real",
                    self.primitive("abs_real", self.primitive("sub_real", now, before)),
                    LiteralValue(value=system.tolerance_cell[0], type=REAL),
                ),
            )

        body = self.bind(
            "current",
            system.carrier,
            self.call(system.name("inside"), (Var(local=repeats), Var(local=tokens))),
            lambda current: If(
                condition=converged(current),
                then=Return(value=Var(local=current)),
                otherwise=self.call(
                    signature.name,
                    (
                        self.primitive("add_int", Var(local=repeats), _int(1)),
                        Var(local=current),
                        Var(local=tokens),
                    ),
                ),
            ),
        )
        return self.computation(signature, (repeats, previous, tokens), body)

    def searched(
        self,
        parameters: tuple[Local, ...],
        repeats: Value | None,
        handlers: tuple[HandlerDef, HandlerDef],
    ) -> Computation:
        """Handle the search and the weights around a derivation of the goal.

        Parameters
        ----------
        parameters : tuple[Local, ...]
            The entry's inputs: the tokens, or the axioms and weights.
        repeats : Value | None
            The repeat bound on unary chains, for a cyclic system.
        handlers : tuple[HandlerDef, HandlerDef]
            The search and collection handlers.

        Returns
        -------
        Computation
            The handled derivation, answering the goal's inside weight.
        """
        system = self.system
        assert system.choice_cell[0] is not None and system.weight_cell[0] is not None
        search_handler, collect_handler = handlers
        item = system.item
        assert system.entries
        length: Value = Reduction(
            operator="sum",
            value=Comprehension(
                binder=system.local("k", INT),
                extent=self.extent,
                body=_int(1),
                result_type=tensor_type(INT, (self.extent,)),
            ),
            result_type=INT,
        )
        if system.by_yield:
            derived = self.yield_root(parameters[0])
        elif system.schema_categories_cell[0]:
            # A schema parser's acyclic unary rules bound every tree's
            # depth by the fuel factor per sentence position.
            fuel_value = length
            for _ in range(system.span_fuel_factor_cell[0] - 1):
                fuel_value = self.primitive("add_int", fuel_value, length)
            assert system.start is not None
            expected = self.constructor(system.start, ())
            derived = self.call(
                system.name("derive"),
                (fuel_value, _int(0), length, expected, Var(local=parameters[0])),
            )
        else:
            prefix = (repeats,) if repeats is not None else ()
            derived = self.call(
                system.name("derive"),
                (*prefix, _int(0), length, Var(local=parameters[0])),
            )

        def accept() -> Computation:
            """Answer the scope's unit, accepting the derivation.

            Returns
            -------
            Computation
                ``()``.
            """
            return Return(value=LiteralValue(value=None, type=UNIT))

        def whole(found: Local) -> Computation:
            """Accept a goal item, requiring a span to cover the sentence.

            Parameters
            ----------
            found : Local
                The goal item.

            Returns
            -------
            Computation
                Acceptance, or an empty choice for a span over part of
                the sentence.
            """
            if not system.by_yield:
                return accept()
            return self.case(
                Var(local=found),
                UNIT,
                {
                    SPAN: lambda fields: If(
                        condition=self.primitive(
                            "and",
                            self.primitive("eq_int", Var(local=fields[0]), _int(0)),
                            self.primitive("eq_int", Var(local=fields[1]), length),
                        ),
                        then=accept(),
                        otherwise=self.fail(UNIT, "entry", "extent"),
                    )
                },
                accept,
                "entry",
                "extent",
            )

        goal_check = self.bind(
            "item",
            item,
            derived,
            lambda found: self.bind(
                "is_goal",
                BOOL,
                self.call(system.name("goal"), (Var(local=found),)),
                lambda is_goal: If(
                    condition=Var(local=is_goal),
                    then=whole(found),
                    otherwise=self.fail(UNIT, "entry", "goal"),
                ),
            ),
        )
        return Handle(
            instance=system.choice_cell[0].entry.instance,
            handler=search_handler.id,
            computation=Handle(
                instance=system.weight_cell[0].entry.instance,
                handler=collect_handler.id,
                computation=goal_check,
                static_arguments=(),
            ),
            static_arguments=(),
        )

    def inputs(self, signature: ComputationSignature) -> tuple[TypeExpr, ...]:
        """A computation's parameter types before any production table.

        Parameters
        ----------
        signature : ComputationSignature
            The computation's signature.

        Returns
        -------
        tuple[TypeExpr, ...]
            Its parameters, less the table the weighted variant appends.
        """
        if self.table is not None and signature.name in self.threaded.values():
            return signature.parameters[:-1]
        return signature.parameters

    def computation(
        self,
        signature: ComputationSignature,
        parameters: tuple[Local, ...],
        body: Computation,
    ) -> NamedComputation:
        """Check a body against its signature and wrap it as a computation.

        Parameters
        ----------
        signature : ComputationSignature
            The registered signature.
        parameters : tuple[Local, ...]
            The parameter locals, in order.
        body : Computation
            The body.

        Returns
        -------
        NamedComputation
            The computation.

        Raises
        ------
        QiecDiagnosticError
            If the body fails to check or its type differs from the
            signature's.
        """
        if self.table is not None and signature.name in self.threaded.values():
            parameters = (*parameters, self.table)
        return self.elaborator._checked_computation(
            signature, parameters, body, self.system.decl, self.path
        )


__all__ = []
