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
over its own span, so the search is bounded by the sentence, and only a
unary rule, which keeps the span, needs the declaration's ``depth``
option to bound it. A system with a lexicon whose items are not all
spans, such as a multiple context-free grammar, derives an item from its
head symbol and the positions of its yield, read in pairs as the
components the item covers: its rules must be linear, each conclusion
component a concatenation of premise components, so a rule with several
premises derives each over fewer tokens, and its unary rules must form
no cycle. Any other system takes its axioms and their weights as input
and enumerates derivations of any item to the depth the option
declares, or to the number of axioms and rules together without one.
The agenda engine tabulates the same derivations; the two agree on
every system whose derivations of the goal fit the bound.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Literal

from quivers.dsl.ast_nodes import (
    BundleDecl,
    DeductionDecl,
    DefineDecl,
    ExprParser,
    LetExprCall,
    LetExprLiteral,
    LetExprNode,
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
    SchemaDecl,
    SequentRule,
    TypeEnumSet,
    TypeFreeResiduated,
    TypeName,
)
from quivers.dsl.compiler._options import (
    find_option,
    get_option_flag,
    get_option_name,
)
from quivers.dsl.compiler._deduction_positions import (
    is_pattern_variable,
    misplaced_positions,
    position_slots,
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
#: The constructor holding a canonically numbered bound variable of a
#: lexicon's logical forms.
BOUND = "bound"

type Sort = Literal["int", "item"]


@dataclass(frozen=True, slots=True)
class _Atom:
    """A declared atom in a pattern.

    Parameters
    ----------
    name
        The atom's name.
    """

    name: str


@dataclass(frozen=True, slots=True)
class _Wild:
    """A pattern variable.

    Parameters
    ----------
    name
        The variable's name.
    """

    name: str


@dataclass(frozen=True, slots=True)
class _Lit:
    """An integer literal in a pattern.

    Parameters
    ----------
    value
        The integer.
    """

    value: int


@dataclass(frozen=True, slots=True)
class _Bound:
    """A bound variable of a logical form, canonically numbered.

    Parameters
    ----------
    index
        The variable's number within its term, from one.
    """

    index: int


@dataclass(frozen=True, slots=True)
class _App:
    """An applied constructor symbol in a pattern.

    Parameters
    ----------
    symbol
        The symbol.
    args
        The argument patterns.
    """

    symbol: str
    args: tuple[Pattern, ...]


type Pattern = _Atom | _Wild | _Lit | _Bound | _App


@dataclass(slots=True)
class _Rule:
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

    source: SequentRule
    premises: tuple[Pattern, ...]
    conclusion: Pattern
    learnable: bool
    bounded: bool
    parent: str | None


@dataclass(slots=True)
class _Entry:
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
    category: Pattern
    form: Pattern | None
    learnable: bool


@dataclass(slots=True)
class _Group:
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
    spans: tuple[tuple[str, str], ...]
    splits: tuple[str, ...]
    rules: list[tuple[int, _Rule]]
    names: dict[int, dict[str, str]]


@dataclass(slots=True)
class _System:
    """Everything the elaboration reads off one deduction.

    Parameters
    ----------
    decl
        The declaration.
    semiring
        The declared semiring's name.
    start
        The start symbol, if any.
    depth
        The declared derivation depth bound, if any.
    rules
        The rules in pattern form.
    entries
        The expanded lexicon, empty for a system that takes its axioms
        as input.
    uses_forms
        Whether span items carry a logical form.
    symbols
        Every constructor symbol with its slot sorts, atoms included as
        symbols of no slots.
    bare
        The atoms used as items on their own, which cannot also be
        applied.
    span_shaped
        Whether every rule relates span items, so derivations may be
        enumerated by span.
    span_fuel_factor
        For a schema parser, the units of derivation fuel per sentence
        position.
    schema_categories
        For a schema parser, its finite category algebra.
    schema_words
        For a schema parser, its vocabulary.
    heads
        For a yield-directed system, the head symbols of its items, in
        the order its derivation numbers them.
    live
        For a yield-directed system, the indices of the rules that can
        fire: those without a premise that has no positions.
    closure
        The indices of the unary rules.
    cyclic
        Whether a chain of unary rules may revisit an item.
    family
        The item family, once declared.
    constructors
        The item family's constructors, by symbol.
    choice
        The ``Search`` instance.
    weight
        The ``Weight`` instance.
    fresh
        The count of fresh locals minted so far.
    """

    decl: DeductionDecl
    semiring: str
    start: str | None
    depth: int | None
    rules: list[_Rule] = field(default_factory=list)
    entries: list[_Entry] = field(default_factory=list)
    uses_forms: bool = False
    symbols: dict[str, list[Sort]] = field(default_factory=dict)
    bare: set[str] = field(default_factory=set)
    span_shaped: bool = False
    span_fuel_factor: int = 0
    schema_categories: tuple[Pattern, ...] = ()
    schema_words: tuple[str, ...] = ()
    heads: list[str] = field(default_factory=list)
    live: list[int] = field(default_factory=list)
    closure: list[int] = field(default_factory=list)
    cyclic: bool = False
    family: FamilyDecl | None = None
    constructors: dict[str, ConstructorDecl] = field(default_factory=dict)
    choice: NamedEffectInstance | None = None
    weight: NamedEffectInstance | None = None
    fresh: int = 0

    @property
    def by_yield(self) -> bool:
        """Whether items are derived from their heads and yield positions.

        Returns
        -------
        bool
            ``True`` for a system with a lexicon whose rules do not all
            relate spans, such as a multiple context-free grammar.
        """
        return bool(self.entries) and not self.span_shaped

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
        assert self.family is not None
        return TypeApplication(self.family.type_constructor, ())

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
        self.fresh += 1
        return Local(f"{hint}_{self.fresh}", type_)


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
    return LiteralValue(value, INT)


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
    return LiteralValue(value, STRING)


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
                # not its derivation. The classic chart applies unary closure
                # three times per cell, so four units of fuel per sentence
                # position cover those rounds and the binary spine.
                system.depth = None
                system.span_fuel_factor = 4
                first_word = system.entries[0].word
                category_count = next(
                    (
                        index
                        for index, entry in enumerate(system.entries)
                        if entry.word != first_word
                    ),
                    len(system.entries),
                )
                system.schema_categories = tuple(
                    entry.category for entry in system.entries[:category_count]
                )
                system.schema_words = tuple(
                    dict.fromkeys(entry.word for entry in system.entries)
                )
            self._declare_item_family(system)
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
            translated.append(
                (
                    self._schema_parser_declaration(
                        statement, statement.expr, schemas, bundles, objects
                    ),
                    statement,
                )
            )
        return translated

    def _schema_parser_declaration(
        self: _Elaborator,
        definition: DefineDecl,
        parser: ExprParser,
        schemas: dict[str, SchemaDecl],
        bundles: dict[str, BundleDecl],
        objects: list[ObjectDecl],
    ) -> DeductionDecl:
        """Build the deduction denoted by one schema parser definition."""

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
            # `UnarySpanDeduction` applies its closure update three times.
            # Three same-named branches reproduce that multiplicity while
            # sharing the one learned rule weight.
            for _ in range(3):
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
        ref = EffectRef(effect_id, SEARCH_EFFECT)
        element = TypeVariable("a")
        extent = IndexVariable("n", NAT)
        operation = OperationDef(
            OperationId.derive(effect_id, "operation", SEARCH_CHOOSE),
            SEARCH_CHOOSE,
            (TypeBinder("a"), IndexBinder("n", NAT)),
            (ArgumentDef("alternatives", tensor_type(element, (extent,))),),
            element,
        )
        effect = EffectDef(ref, (), (operation,))
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
        system = _System(declaration, semiring, start, depth)
        atoms = set(declaration.atoms)
        for atom in declaration.atoms:
            system.symbols.setdefault(atom, [])
        system.uses_forms = bool(declaration.binders) or any(
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
                    rule,
                    premises,
                    conclusion,
                    get_option_flag(rule.options, "learnable"),
                    get_option_flag(rule.options, "bounded"),
                    parent_name,
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
        self._infer_sorts(system)
        system.span_shaped = bool(system.entries) and all(
            _is_span(pattern)
            for rule in system.rules
            for pattern in (*rule.premises, rule.conclusion)
        )
        system.closure = [
            index for index, rule in enumerate(system.rules) if len(rule.premises) == 1
        ]
        system.cyclic = _may_cycle(system)
        if system.by_yield:
            self._read_heads(system)
        unary = any(len(rule.premises) == 1 for rule in system.rules)
        if system.depth is None and unary and system.span_shaped:
            self._fail(
                declaration,
                f"deduction {declaration.name!r} has a unary rule, so its "
                "derivations over one span need a depth option to bound them",
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
        if system.cyclic:
            self._fail(
                system.decl,
                f"deduction {system.decl.name!r} has unary rules that may form a "
                "cycle; a lexical deduction whose items are not all spans "
                "derives items by their yields, which a cycle of unary rules "
                "does not shrink",
                code="qiec-program",
            )
        system.heads = heads

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
                return _Wild(expr.name)
            if expr.name.isdigit():
                return _Lit(int(expr.name))
            self._note_atom(system, expr.name)
            return _Atom(expr.name)
        if isinstance(expr, ObjectEffectApply):
            args = tuple(
                self._object_pattern(item, atoms, system) for item in expr.args
            )
            self._note_symbol(system, expr.effect, len(args))
            return _App(expr.effect, args)
        if isinstance(expr, ObjectSlash):
            symbol = "fwd_slash" if expr.direction == "/" else "bwd_slash"
            args = (
                self._object_pattern(expr.result, atoms, system),
                self._object_pattern(expr.argument, atoms, system),
            )
            self._note_symbol(system, symbol, 2)
            return _App(symbol, args)
        if isinstance(expr, ObjectProduct):
            args = tuple(
                self._object_pattern(item, atoms, system) for item in expr.components
            )
            symbol = f"product{len(args)}"
            self._note_symbol(system, symbol, len(args))
            return _App(symbol, args)
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
        if not system.uses_forms:
            return pattern
        if isinstance(pattern, _App):
            args = tuple(
                self._span_form(item, system, counter) for item in pattern.args
            )
            if pattern.symbol == SPAN and len(args) == 3:
                counter[0] += 1
                args = (*args, _Wild(f"_lf{counter[0]}"))
                system.symbols[SPAN] = ["item"] * 4
            return _App(pattern.symbol, args)
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
                if system.uses_forms and form_expr is not None
                else None
            )
            if fixed is not None:
                category = self._object_pattern(fixed.category, atoms, system)
                system.entries.append(_Entry(word, category, form, learnable))
            else:
                for atom in candidates:
                    system.entries.append(_Entry(word, _Atom(atom), form, True))
        if system.entries:
            sorts: list[Sort] = ["int", "int", "item", "item"]
            system.symbols[SPAN] = sorts[: 4 if system.uses_forms else 3]

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
                    return _Bound(scope[node.name])
                if node.name in atoms:
                    self._note_atom(system, node.name)
                    return _Atom(node.name)
                self._fail(
                    declaration,
                    f"deduction {declaration.name!r}: logical form names "
                    f"{node.name!r}, which is neither an atom nor bound",
                    code="qiec-program",
                )
            if isinstance(node, LetExprLiteral):
                if float(node.value).is_integer():
                    return _Lit(int(node.value))
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
                        _Bound(counter[0]),
                        *(visit(item, inner) for item in node.args[1:]),
                    )
                else:
                    args = tuple(visit(item, scope) for item in node.args)
                self._note_symbol(system, node.func, len(args))
                return _App(node.func, args)
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
        family = FamilyDecl(family_id, family_name, (), (), constructor_ids)
        system.family = family
        item = TypeApplication(family.type_constructor, ())
        self.families[family_name] = family
        self.registry.register_family(family)
        for symbol, identity in zip(names, constructor_ids, strict=True):
            fields = tuple(
                FieldDef(f"arg{position}", INT if sort == "int" else item)
                for position, sort in enumerate(system.symbols[symbol])
            )
            constructor = ConstructorDecl(
                identity, family_id, system.name(symbol), (), fields, ()
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
        system.choice = NamedEffectInstance(
            choice_name,
            entry,
            self._origin(system.decl, ("instances", choice_name), "effect-instance"),
        )
        self.instances[choice_name] = system.choice
        weight_effect = self._effect("Weight")
        assert weight_effect is not None
        weight_name = system.name("weight")
        entry = instantiate_effect(
            weight_effect.apply((system.carrier,)),
            module=self.source.module_name,
            lexical_path=("instances", weight_name),
        )
        system.weight = NamedEffectInstance(
            weight_name,
            entry,
            self._origin(system.decl, ("instances", weight_name), "effect-instance"),
        )
        self.instances[weight_name] = system.weight

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
        for system in getattr(self, "_deductions", {}).values():
            computations.extend(self._elaborate_deduction(system))
        return tuple(computations)

    def _elaborate_deduction(
        self: _Elaborator, system: _System
    ) -> list[NamedComputation]:
        """Elaborate one deduction.

        Parameters
        ----------
        system : _System
            The system.

        Returns
        -------
        list[NamedComputation]
            Its computations.
        """
        params = self.instances[PARAMS_INSTANCE]
        assert system.choice is not None and system.weight is not None
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
        base = (system.choice.entry, system.weight.entry)
        axiom_row = EffectRow((*base, params.entry) if lexical_learned else base)
        row = EffectRow((*base, params.entry) if learned else base)
        outer = EffectRow((params.entry,) if learned else ())
        pure = EffectRow()
        extent = IndexBinder("n", NAT)
        n = IndexVariable("n", NAT)
        item = system.item
        carrier = system.carrier
        signatures: dict[str, ComputationSignature] = {}

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
                The registered signature.
            """
            identity = ComputationId.derive(
                self.source.module_name, "computation", name
            )
            signature = ComputationSignature(
                identity, name, telescope, parameters, result, effects
            )
            self.registry.register_computation(signature)
            self.computation_signatures[name] = signature
            signatures[name] = signature
            return signature

        eq = declare(system.name("eq"), (), (item, item), BOOL, pure)
        show = declare(system.name("show"), (), (item,), STRING, pure)
        goal = declare(system.name("goal"), (), (item,), BOOL, pure)
        failing = declare(
            system.name("fail"), (), (), item, EffectRow((system.choice.entry,))
        )
        if system.entries:
            tokens_type = tensor_type(STRING, (n,))
            axiom = declare(
                system.name("axiom"),
                (extent,),
                (INT, item, tokens_type)
                if system.schema_categories
                else (INT, tokens_type),
                item,
                axiom_row,
            )
            derive = declare(
                system.name("derive"),
                (extent,),
                (INT, INT, INT, item, tokens_type)
                if system.schema_categories
                else (INT, INT, INT, tokens_type)
                if system.span_shaped
                else (INT, *([INT] * self._frame_width(system)), tokens_type),
                item,
                row,
            )
            entry = declare(
                system.name("run"), (extent,), (tokens_type,), carrier, outer
            )
        else:
            axioms_type = tensor_type(item, (n,))
            weights_type = tensor_type(carrier, (n,))
            axiom = None
            derive = declare(
                system.name("derive"),
                (extent,),
                (INT, axioms_type, weights_type),
                item,
                row,
            )
            entry = declare(
                system.name("run"),
                (extent,),
                (axioms_type, weights_type),
                carrier,
                outer,
            )
        context = _Context(self, system, signatures, params, n)
        computations = [
            context.equality(eq),
            context.rendering(show),
            context.goal(goal),
            context.failure(failing),
        ]
        if axiom is not None:
            computations.append(context.axiom(axiom))
        computations.append(context.derivation(derive))
        computations.append(context.entry(entry, self._deduction_handlers(system)))
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
        assert system.choice is not None and system.weight is not None
        search = self.effects[SEARCH_EFFECT]
        weight_effect = self._effect("Weight")
        assert weight_effect is not None
        carrier = system.carrier
        shot = product_type(UNIT, carrier)
        search_name = system.name(f"search_{system.semiring.lower()}")
        search_handler = HandlerDef(
            HandlerId.derive(self.source.module_name, "handler", search_name),
            search_name,
            search.ref,
            (HandlerClauseDef(search.operations[0].id, ResumptionGrade.UNRESTRICTED),),
            shot,
            carrier,
            EffectRow(),
            total=True,
            implementation="foreign",
        )
        collect_name = system.name(f"collect_{system.semiring.lower()}")
        add = next(item for item in weight_effect.operations if item.name == "add")
        collect_handler = HandlerDef(
            HandlerId.derive(self.source.module_name, "handler", collect_name),
            collect_name,
            weight_effect.apply((carrier,)),
            (HandlerClauseDef(add.id, ResumptionGrade.LINEAR),),
            UNIT,
            shot,
            EffectRow(),
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
    """

    def __init__(
        self,
        elaborator: _Elaborator,
        system: _System,
        signatures: dict[str, ComputationSignature],
        params: NamedEffectInstance,
        extent: IndexVariable,
    ) -> None:
        self.elaborator = elaborator
        self.system = system
        self.signatures = signatures
        self.params = params
        self.extent = extent
        self.module = elaborator.source.module_name
        self.path: tuple[str | int, ...] = ("deductions", system.decl.name)

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
        statics = tuple(
            IndexVariable(binder.name, binder.sort)
            for binder in signature.telescope
            if isinstance(binder, IndexBinder)
        )
        return Call(
            signature.id,
            name,
            statics,
            arguments,
            signature.result,
            signature.effects,
            self.elaborator._origin(
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
            EffectRequest(
                instance.entry.instance,
                instance.entry.effect,
                definition.id,
                statics,
                arguments,
                result,
                SiteProvenance(
                    self.elaborator._origin(
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
        assert self.system.choice is not None
        return self.request(
            self.system.choice,
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
        empty = TensorValue((), tensor_type(result, (IndexLiteral(0, NAT),)))
        return self.choose(empty, result, IndexLiteral(0, NAT), *path)

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
        empty = TensorValue((), tensor_type(self.system.item, (IndexLiteral(0, NAT),)))
        body = self.choose(empty, self.system.item, IndexLiteral(0, NAT), "fail")
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
        assert self.system.weight is not None
        return self.request(self.system.weight, "add", (), (weight,), UNIT, *path)

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
            self.system.constructors[symbol].id, (), fields, self.system.item
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
                    constructor.id,
                    (),
                    fields,
                    body,
                    StaticScopeId.derive(
                        self.module,
                        (*self.path, *path),
                        "branch",
                        position,
                        constructor.id,
                    ),
                )
            )
        return Case(scrutinee, CaseMotive((), result), tuple(built))

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
        return Bind(local, first, then(local))

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
                "as_weight", real if real is not None else LiteralValue(0.0, REAL)
            )
        if semiring == "Boolean":
            return LiteralValue(True, BOOL)
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
        left = Local("left", item)
        right = Local("right", item)
        show = self.system.name("show")
        body = self.bind(
            "left_text",
            STRING,
            self.call(show, (Var(left),)),
            lambda left_text: self.bind(
                "right_text",
                STRING,
                self.call(show, (Var(right),)),
                lambda right_text: Return(
                    self.primitive("eq_string", Var(left_text), Var(right_text))
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
        item = Local("item", self.system.item)

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
                    return Return(_string(symbol))

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
                        return Return(self.primitive("concat", text, _string(")")))
                    local = fields[position]
                    if local.type == INT:
                        return piece(
                            position + 1,
                            [*rendered, self.primitive("int_to_string", Var(local))],
                        )
                    return self.bind(
                        "text",
                        STRING,
                        self.call(signature.name, (Var(local),)),
                        lambda text: piece(position + 1, [*rendered, Var(text)]),
                    )

                return piece(0, [])

            return body

        body = self.case(
            Var(item),
            STRING,
            {symbol: render(symbol) for symbol in self.system.constructors},
            lambda: Return(_string("")),
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
        item = Local("item", self.system.item)
        start = self.system.start

        def yes() -> Computation:
            """Answer that the item is a goal.

            Returns
            -------
            Computation
                ``true``.
            """
            return Return(LiteralValue(True, BOOL))

        def no() -> Computation:
            """Answer that the item is not a goal.

            Returns
            -------
            Computation
                ``false``.
            """
            return Return(LiteralValue(False, BOOL))

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
                Var(fields[2]),
                BOOL,
                {start: lambda inner: yes()},
                no,
                "goal",
                "category",
            )
        body = self.case(Var(item), BOOL, branches, no, "goal")
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
        if self.system.schema_categories:
            return self.schema_axiom(signature)
        position = Local("position", INT)
        tokens = Local("tokens", signature.parameters[1])
        entries = self.system.entries
        token = self.system.local("token", STRING)
        alternatives = TensorValue(
            tuple(_int(index) for index in range(len(entries))),
            tensor_type(INT, (IndexLiteral(len(entries), NAT),)),
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
                Var(position),
                self.primitive("add_int", Var(position), _int(1)),
                self.ground(entry.category),
            ]
            if self.system.uses_forms:
                assert entry.form is not None
                fields.append(self.ground(entry.form))
            span = self.constructor(SPAN, tuple(fields))
            result = Return(span)

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
                    self.system.local("added", UNIT),
                    self.add_weight(weight, "lexicon", index, "weight"),
                    result,
                )

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
                    lambda real: weighted(self.weight_of(Var(real))),
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
                self.primitive("eq_int", Var(chosen), _int(index)),
                If(
                    self.primitive("eq_string", Var(token), _string(entry.word)),
                    entry_body(index, entry),
                    self.fail(self.system.item, "lexicon", index, "mismatch"),
                ),
                dispatch(chosen, index + 1),
            )

        body = Bind(
            token,
            Return(Gather(Var(tokens), Var(position), STRING)),
            self.bind(
                "entry",
                INT,
                self.choose(
                    alternatives,
                    INT,
                    IndexLiteral(len(entries), NAT),
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
        position = Local("position", INT)
        expected = Local("expected", system.item)
        tokens = Local("tokens", signature.parameters[2])
        token = system.local("token", STRING)
        count = len(system.schema_categories)

        def token_body(token_index: int, category_index: int) -> Computation:
            """Add one lexical-table cell and return its span."""
            span = self.constructor(
                SPAN,
                (
                    Var(position),
                    self.primitive("add_int", Var(position), _int(1)),
                    Var(expected),
                ),
            )
            flat_index = token_index * count + category_index

            def weighted(weight: Value) -> Computation:
                """Add the lexical weight and return its span."""
                return Bind(
                    system.local("added", UNIT),
                    self.add_weight(
                        weight,
                        "lexicon",
                        flat_index,
                        "weight",
                    ),
                    Return(span),
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
                lambda real: weighted(self.weight_of(Var(real))),
            )

        def word(token_index: int, category_index: int) -> Computation:
            """Dispatch the observed token for one expected category."""
            if token_index == len(system.schema_words):
                return self.fail(system.item, "lexicon", "token", "none")
            return If(
                self.primitive(
                    "eq_string",
                    Var(token),
                    _string(system.schema_words[token_index]),
                ),
                token_body(token_index, category_index),
                word(token_index + 1, category_index),
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
                    (Var(expected), self.ground(system.schema_categories[index])),
                ),
                lambda same: If(
                    Var(same),
                    Bind(
                        token,
                        Return(Gather(Var(tokens), Var(position), STRING)),
                        word(0, index),
                    ),
                    category(index + 1),
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
            return Var(bindings[pattern.name])
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

        Returns
        -------
        Computation
            The match.
        """
        if isinstance(pattern, _Wild):
            bound = bindings.get(pattern.name)
            if bound is None:
                return continue_with({**bindings, pattern.name: value})
            if value.type == INT:
                condition = self.primitive("eq_int", Var(bound), Var(value))
                return If(
                    condition,
                    continue_with(bindings),
                    self.fail(result, *path, "shared"),
                )
            return self.bind(
                "same",
                BOOL,
                self.call(self.system.name("eq"), (Var(bound), Var(value))),
                lambda same: If(
                    Var(same),
                    continue_with(bindings),
                    self.fail(result, *path, "shared"),
                ),
            )
        if isinstance(pattern, _Lit):
            return If(
                self.primitive("eq_int", Var(value), _int(pattern.value)),
                continue_with(bindings),
                self.fail(result, *path, "literal"),
            )
        if isinstance(pattern, _Bound):
            return self.case(
                Var(value),
                result,
                {
                    BOUND: lambda fields: If(
                        self.primitive("eq_int", Var(fields[0]), _int(pattern.index)),
                        continue_with(bindings),
                        self.fail(result, *path, "bound"),
                    )
                },
                lambda: self.fail(result, *path, "bound"),
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
                )

            return step(0, bindings)

        return self.case(
            Var(value),
            result,
            {symbol: fields_matched},
            lambda: self.fail(result, *path, "constructor"),
            *path,
            symbol,
        )

    # -- derivation -----------------------------------------------------

    def derivation(self, signature: ComputationSignature) -> NamedComputation:
        """Derive an item by one rule application or one axiom.

        Rules are grouped by the shape of their premises: the spans the
        premises cover relative to the conclusion's, or, without spans,
        their number. A derivation chooses an axiom or a group, derives
        the group's premises once, then chooses a rule of the group and
        matches the premises against its patterns, so the sub-derivations
        are shared by every rule of the same shape.

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
        if self.system.schema_categories:
            return self.schema_derivation(signature)
        if self.system.by_yield:
            return self.yield_derivation(signature)
        system = self.system
        item = system.item
        fuel = Local("fuel", INT)
        start = Local("start", INT)
        stop = Local("stop", INT)
        spent = self.primitive("sub_int", Var(fuel), _int(1))
        if system.span_shaped:
            tokens = Local("tokens", signature.parameters[3])
            parameters: tuple[Local, ...] = (fuel, start, stop, tokens)

            def recursion(low: Value, high: Value) -> Call:
                """Derive an item over a sub-span with one less fuel.

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
                return self.call(signature.name, (spent, low, high, Var(tokens)))

            axiom_alternative: Computation = If(
                self.primitive(
                    "eq_int", Var(stop), self.primitive("add_int", Var(start), _int(1))
                ),
                self.call(system.name("axiom"), (Var(start), Var(tokens))),
                self.fail(item, "derive", "lexical", "span"),
            )
        else:
            axioms = Local("axioms", signature.parameters[1])
            weights = Local("weights", signature.parameters[2])
            parameters = (fuel, axioms, weights)

            def recursion(low: Value, high: Value) -> Call:
                """Derive any item with one less fuel.

                Parameters
                ----------
                low : Value
                    Unused; the system has no spans.
                high : Value
                    Unused; the system has no spans.

                Returns
                -------
                Call
                    The recursive call.
                """
                del low, high
                return self.call(signature.name, (spent, Var(axioms), Var(weights)))

            positions = self.system.local("k", INT)
            axiom_alternative = self.bind(
                "chosen",
                INT,
                self.choose(
                    Comprehension(
                        positions,
                        self.extent,
                        Var(positions),
                        tensor_type(INT, (self.extent,)),
                    ),
                    INT,
                    self.extent,
                    "derive",
                    "axiom",
                    "choose",
                ),
                lambda chosen: Bind(
                    system.local("added", UNIT),
                    self.add_weight(
                        Gather(Var(weights), Var(chosen), system.carrier),
                        "derive",
                        "axiom",
                        "weight",
                    ),
                    Return(Gather(Var(axioms), Var(chosen), item)),
                ),
            )
        groups = self.rule_groups()
        alternatives = TensorValue(
            tuple(_int(index) for index in range(-1, len(groups))),
            tensor_type(INT, (IndexLiteral(len(groups) + 1, NAT),)),
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
                index_local = system.local("k", INT)
                return self.bind(
                    "split",
                    INT,
                    self.choose(
                        Comprehension(
                            index_local,
                            self.extent,
                            Var(index_local),
                            tensor_type(INT, (self.extent,)),
                        ),
                        INT,
                        self.extent,
                        *path,
                        "position",
                        name,
                    ),
                    lambda split: If(
                        self.primitive(
                            "and",
                            self.primitive("lt_int", Var(start), Var(split)),
                            self.primitive("lt_int", Var(split), Var(stop)),
                        ),
                        choose_positions(remaining[1:], {**chosen, name: split}),
                        self.fail(item, *path, "position", name),
                    ),
                )

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
                    return choose_rule(chosen, derived)
                if system.span_shaped:
                    low, high = group.spans[position]
                    call = recursion(Var(chosen[low]), Var(chosen[high]))
                else:
                    call = recursion(_int(0), _int(0))
                return self.bind(
                    "premise",
                    item,
                    call,
                    lambda value: premises(position + 1, chosen, [*derived, value]),
                )

            def choose_rule(
                chosen: dict[str, Local], derived: list[Local]
            ) -> Computation:
                """Choose a rule of the group and match the premises against it.

                Parameters
                ----------
                chosen : dict[str, Local]
                    The positions, by normalized name.
                derived : list[Local]
                    The derived premises.

                Returns
                -------
                Computation
                    The choice, each rule's match innermost.
                """
                if len(group.rules) == 1:
                    return match_rule(group.rules[0], chosen, derived)
                rule_alternatives = TensorValue(
                    tuple(_int(index) for index in range(len(group.rules))),
                    tensor_type(INT, (IndexLiteral(len(group.rules), NAT),)),
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
                        self.primitive("eq_int", Var(selected), _int(position)),
                        match_rule(group.rules[position], chosen, derived),
                        dispatch(selected, position + 1),
                    )

                return self.bind(
                    "rule",
                    INT,
                    self.choose(
                        rule_alternatives,
                        INT,
                        IndexLiteral(len(group.rules), NAT),
                        *path,
                        "rule",
                    ),
                    lambda selected: dispatch(selected, 0),
                )

            def match_rule(
                entry: tuple[int, _Rule], chosen: dict[str, Local], derived: list[Local]
            ) -> Computation:
                """Match the derived premises against one rule's patterns.

                Parameters
                ----------
                entry : tuple[int, _Rule]
                    The rule and its index.
                chosen : dict[str, Local]
                    The positions, by normalized name.
                derived : list[Local]
                    The derived premises.

                Returns
                -------
                Computation
                    The matches, innermost the conclusion.
                """
                index, rule = entry
                bindings: dict[str, Local] = {}
                if system.span_shaped:
                    for name, normalized in group.names[index].items():
                        bindings[name] = (
                            start
                            if normalized == "L"
                            else stop
                            if normalized == "R"
                            else chosen[normalized]
                        )

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
                        return conclude(index, rule, current)
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

                return step(0, bindings)

            return choose_positions(list(group.splits), {"L": start, "R": stop})

        def conclude(index: int, rule: _Rule, current: dict[str, Local]) -> Computation:
            """Instantiate a rule's conclusion, add its weight, answer.

            Parameters
            ----------
            index : int
                The rule's index.
            rule : _Rule
                The rule.
            current : dict[str, Local]
                The complete bindings.

            Returns
            -------
            Computation
                The conclusion.
            """
            answer = Return(self.instantiate(rule.conclusion, current))

            def weighted(weight: Value) -> Computation:
                """Add the weight, then answer.

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
                    system.local("added", UNIT),
                    self.add_weight(weight, "derive", "rule", index, "weight"),
                    answer,
                )

            if system.semiring not in ("LogProb", "Viterbi") or not (
                rule.learnable or rule.parent is not None
            ):
                return weighted(self.weight_of(None))
            return self.rule_weight(rule, index, current, weighted)

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
                self.primitive("eq_int", Var(chosen), _int(index)),
                group_body(index, groups[index]),
                dispatch(chosen, index + 1),
            )

        body = If(
            self.primitive("le_int", Var(fuel), _int(0)),
            self.fail(item, "derive", "fuel"),
            self.bind(
                "alternative",
                INT,
                self.choose(
                    alternatives,
                    INT,
                    IndexLiteral(len(groups) + 1, NAT),
                    "derive",
                    "choose",
                ),
                lambda chosen: If(
                    self.primitive("eq_int", Var(chosen), _int(-1)),
                    axiom_alternative,
                    dispatch(chosen, 0),
                ),
            ),
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
        fuel = Local("fuel", INT)
        start = Local("start", INT)
        stop = Local("stop", INT)
        expected = Local("expected", item)
        tokens = Local("tokens", signature.parameters[4])
        parameters = (fuel, start, stop, expected, tokens)
        spent = self.primitive("sub_int", Var(fuel), _int(1))

        def recursion(low: Value, high: Value, category: Pattern) -> Call:
            """Call the derivation on a ground premise category."""
            return self.call(
                signature.name,
                (spent, low, high, self.ground(category), Var(tokens)),
            )

        def answer() -> Return:
            """Return the expected category over the current span."""
            return Return(
                self.constructor(
                    SPAN,
                    (Var(start), Var(stop), Var(expected)),
                )
            )

        def weighted(index: int, rule: _Rule) -> Computation:
            """Add one concrete rule's learned log weight."""

            def added(weight: Value) -> Computation:
                """Add the carrier weight and return the chart item."""
                return Bind(
                    system.local("added", UNIT),
                    self.add_weight(weight, "derive", "rule", index, "weight"),
                    answer(),
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
                lambda raw: added(self.weight_of(Var(raw))),
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
                        recursion(Var(start), Var(stop), premise.args[2]),
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
                            split_index,
                            self.extent,
                            Var(split_index),
                            tensor_type(INT, (self.extent,)),
                        ),
                        INT,
                        self.extent,
                        "derive",
                        "rule",
                        index,
                        "split",
                    ),
                    lambda split: If(
                        self.primitive(
                            "and",
                            self.primitive("lt_int", Var(start), Var(split)),
                            self.primitive("lt_int", Var(split), Var(stop)),
                        ),
                        self.bind(
                            "left",
                            item,
                            recursion(Var(start), Var(split), left.args[2]),
                            lambda _: self.bind(
                                "right",
                                item,
                                recursion(Var(split), Var(stop), right.args[2]),
                                lambda _: weighted(index, rule),
                            ),
                        ),
                        self.fail(item, "derive", "rule", index, "split"),
                    ),
                )

            return premises()

        lexical = If(
            self.primitive(
                "eq_int", Var(stop), self.primitive("add_int", Var(start), _int(1))
            ),
            self.call(
                system.name("axiom"),
                (Var(start), Var(expected), Var(tokens)),
            ),
            self.fail(item, "derive", "lexical", "span"),
        )
        rules_by_category: list[list[int]] = [[] for _ in system.schema_categories]
        for rule_index, rule in enumerate(system.rules):
            conclusion = rule.conclusion
            assert isinstance(conclusion, _App) and conclusion.symbol == SPAN
            category = conclusion.args[2]
            category_index = system.schema_categories.index(category)
            rules_by_category[category_index].append(rule_index)

        def category_dispatch(category_index: int) -> Computation:
            """Choose only rules whose result is the expected category."""
            if category_index == len(system.schema_categories):
                return self.fail(item, "derive", "category")
            rule_indices = rules_by_category[category_index]
            alternatives = TensorValue(
                (_int(-1), *(_int(index) for index in rule_indices)),
                tensor_type(INT, (IndexLiteral(len(rule_indices) + 1, NAT),)),
            )

            def rule_dispatch(chosen: Local, offset: int) -> Computation:
                """Dispatch one rule from the category-specific choice."""
                if offset == len(rule_indices):
                    return self.fail(item, "derive", "none")
                rule_index = rule_indices[offset]
                return If(
                    self.primitive("eq_int", Var(chosen), _int(rule_index)),
                    apply_rule(rule_index),
                    rule_dispatch(chosen, offset + 1),
                )

            category = system.schema_categories[category_index]
            return self.bind(
                "same",
                BOOL,
                self.call(
                    system.name("eq"),
                    (Var(expected), self.ground(category)),
                ),
                lambda same: If(
                    Var(same),
                    self.bind(
                        "alternative",
                        INT,
                        self.choose(
                            alternatives,
                            INT,
                            IndexLiteral(len(rule_indices) + 1, NAT),
                            "derive",
                            "choose",
                            category_index,
                        ),
                        lambda chosen: If(
                            self.primitive("eq_int", Var(chosen), _int(-1)),
                            lexical,
                            rule_dispatch(chosen, 0),
                        ),
                    ),
                    category_dispatch(category_index + 1),
                ),
            )

        body = If(
            self.primitive("le_int", Var(fuel), _int(0)),
            self.fail(item, "derive", "fuel"),
            category_dispatch(0),
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
                system.local("added", UNIT),
                self.add_weight(weight, "derive", "rule", index, "weight"),
                then(conclusion),
            )

        if system.semiring not in ("LogProb", "Viterbi") or not (
            rule.learnable or rule.parent is not None
        ):
            return weighted(self.weight_of(None))
        return self.rule_weight(rule, index, bindings, weighted)

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
            tuple(_int(index) for index in range(len(group.rules))),
            tensor_type(INT, (IndexLiteral(len(group.rules), NAT),)),
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
                self.primitive("eq_int", Var(selected), _int(position)),
                match_rule(group.rules[position]),
                dispatch(selected, position + 1),
            )

        return self.bind(
            "rule",
            INT,
            self.choose(
                rule_alternatives,
                INT,
                IndexLiteral(len(group.rules), NAT),
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
                index, self.extent, Var(index), tensor_type(INT, (self.extent,))
            ),
            INT,
            self.extent,
            *path,
        )

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
            (_int(0), _int(1)), tensor_type(INT, (IndexLiteral(2, NAT),))
        )
        return self.bind(
            "side",
            INT,
            self.choose(sides, INT, IndexLiteral(2, NAT), *path, "side"),
            lambda side: If(
                self.primitive("eq_int", Var(side), _int(0)),
                Return(_int(0)),
                self.bind(
                    "offset",
                    INT,
                    self.choose_position(*path, "offset"),
                    lambda offset: Return(
                        self.primitive("add_int", Var(offset), _int(1))
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
            lambda chosen: Return(self.primitive("add_int", low, Var(chosen))),
        )
        boundary = If(
            self.primitive(
                "lt_int", self.primitive("sub_int", high, low), self.length()
            ),
            offset,
            self.bind(
                "any", INT, self.boundary(*path, "any"), lambda any_: Return(Var(any_))
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
        head = Local("head", INT)
        frame = tuple(Local(f"p{index}", INT) for index in range(width))
        tokens = Local("tokens", signature.parameters[-1])
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
                (_int(system.heads.index(symbol)), *padded, Var(tokens)),
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
            return _int(node) if isinstance(node, int) else Var(bindings[node])

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
                        checks.append(self.primitive("eq_int", Var(given), _int(node)))
                    elif node in bound:
                        checks.append(
                            self.primitive("eq_int", Var(bound[node]), Var(given))
                        )
                    else:
                        bound[node] = given
            low: Value = Var(frame[0]) if conclusion else _int(0)
            high: Value = Var(frame[0]) if conclusion else self.length()
            for position in range(1, 2 * len(conclusion)):
                low = self.primitive("min_int", low, Var(frame[position]))
                high = self.primitive("max_int", high, Var(frame[position]))
            steps: list[tuple[_Node, _Node, _Node]] = [
                (component[step - 1], component[step], component[-1])
                for component in paths
                for step in range(1, len(component) - 1)
            ]
            group = _Group(len(rule.premises), (), (), [(index, rule)], {})

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
                        self.primitive(
                            "and",
                            self.primitive("le_int", after, chosen),
                            self.primitive("le_int", chosen, before),
                        ),
                        inner(position + 1, extended),
                        self.fail(item, *path, "order", position),
                    )

                if isinstance(node, int) or node in current:
                    return placed(value_of(node, current), current)
                return self.choose_between(
                    after,
                    before,
                    lambda chosen: placed(Var(chosen), {**current, node: chosen}),
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
                        nonempty(current),
                        premises(0, [], current),
                        self.fail(item, *path, "premises"),
                    )
                name = loose[position]
                if name in current:
                    return spread(position + 1, current)
                return self.choose_boundary(
                    lambda chosen: If(
                        self.primitive(
                            "and",
                            self.primitive("le_int", low, Var(chosen)),
                            self.primitive("le_int", Var(chosen), high),
                        ),
                        spread(position + 1, {**current, name: chosen}),
                        self.fail(item, *path, "loose", name),
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
                        group, derived, lambda _: dict(current), Return, *path
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
                condition: Value = LiteralValue(True, BOOL)
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
            return If(condition, body, self.fail(item, *path, "frame"))

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
                        self.primitive(
                            "eq_int",
                            Var(frame[1]),
                            self.primitive("add_int", Var(frame[0]), _int(1)),
                        ),
                        self.call(system.name("axiom"), (Var(frame[0]), Var(tokens))),
                        self.fail(item, "derive", "axiom", "span"),
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
                    self.primitive("eq_int", Var(chosen), _int(position)),
                    options[position](),
                    dispatch(chosen, position + 1),
                )

            if len(options) == 1:
                derivations = options[0]()
            else:
                derivations = self.bind(
                    "derivation",
                    INT,
                    self.choose(
                        TensorValue(
                            tuple(_int(index) for index in range(len(options))),
                            tensor_type(INT, (IndexLiteral(len(options), NAT),)),
                        ),
                        INT,
                        IndexLiteral(len(options), NAT),
                        "derive",
                        "head",
                        symbol,
                    ),
                    lambda chosen: dispatch(chosen, 0),
                )
            if not components:
                return derivations
            covered: Value = _int(0)
            forward: Value = LiteralValue(True, BOOL)
            for position in range(components):
                start, end = Var(frame[2 * position]), Var(frame[2 * position + 1])
                forward = self.primitive(
                    "and", forward, self.primitive("le_int", start, end)
                )
                covered = self.primitive(
                    "add_int", covered, self.primitive("sub_int", end, start)
                )
            return If(
                self.primitive(
                    "and", forward, self.primitive("lt_int", _int(0), covered)
                ),
                derivations,
                self.fail(item, "derive", "head", symbol, "frame"),
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
                self.primitive("eq_int", Var(head), _int(position)),
                head_body(system.heads[position]),
                heads(position + 1),
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
            "sum",
            Comprehension(
                self.system.local("k", INT),
                self.extent,
                _int(1),
                tensor_type(INT, (self.extent,)),
            ),
            INT,
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
                (_int(system.heads.index(symbol)), *padded, Var(tokens)),
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
                    lambda chosen: frame([*values, Var(chosen)]),
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
                self.primitive("eq_int", Var(chosen), _int(position)),
                root(roots[position]),
                dispatch(chosen, position + 1),
            )

        return self.bind(
            "root",
            INT,
            self.choose(
                TensorValue(
                    tuple(_int(index) for index in range(len(roots))),
                    tensor_type(INT, (IndexLiteral(len(roots), NAT),)),
                ),
                INT,
                IndexLiteral(len(roots), NAT),
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
            if not self.system.span_shaped:
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
                group = _Group(len(rule.premises), spans, splits, [], {})
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
                self.call(self.system.name("show"), (Var(local),)),
                lambda text: render(position + 1, [*rendered, Var(text)], then),
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
                return then(LiteralValue(0.0, REAL))
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
                    return Var(raw)
                return self.primitive(
                    "sub_real",
                    self.primitive("neg_real", self.primitive("softplus", Var(raw))),
                    LiteralValue(cap, REAL),
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

    # -- entry ------------------------------------------------------------

    def entry(
        self, signature: ComputationSignature, handlers: tuple[HandlerDef, HandlerDef]
    ) -> NamedComputation:
        """The deduction's entry: the inside weight of the goal.

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
            Local(name, type_)
            for name, type_ in zip(
                ("tokens",) if system.entries else ("axioms", "weights"),
                signature.parameters,
                strict=True,
            )
        )
        body = self.searched(parameters, handlers)
        return self.computation(signature, parameters, body)

    def searched(
        self,
        parameters: tuple[Local, ...],
        handlers: tuple[HandlerDef, HandlerDef],
    ) -> Computation:
        """Handle the search and the weights around a derivation of the goal.

        Parameters
        ----------
        parameters : tuple[Local, ...]
            The entry's inputs: the tokens, or the axioms and weights.
        handlers : tuple[HandlerDef, HandlerDef]
            The search and collection handlers.

        Returns
        -------
        Computation
            The handled derivation, answering the goal's inside weight.
        """
        system = self.system
        assert system.choice is not None and system.weight is not None
        search_handler, collect_handler = handlers
        item = system.item
        length: Value = Reduction(
            "sum",
            Comprehension(
                system.local("k", INT),
                self.extent,
                _int(1),
                tensor_type(INT, (self.extent,)),
            ),
            INT,
        )
        if system.by_yield:
            derived = self.yield_root(parameters[0])
        elif system.span_shaped:
            # The sentence's length is the sum of a tensor of ones over
            # the extent; without a declared depth it bounds the depth of
            # a binary system's derivations of the whole sentence.
            if system.span_fuel_factor:
                fuel_value = length
                for _ in range(system.span_fuel_factor - 1):
                    fuel_value = self.primitive("add_int", fuel_value, length)
            else:
                fuel_value = _int(system.depth) if system.depth is not None else length
            if system.schema_categories:
                assert system.start is not None
                expected = self.constructor(system.start, ())
                derived = self.call(
                    system.name("derive"),
                    (
                        fuel_value,
                        _int(0),
                        length,
                        expected,
                        Var(parameters[0]),
                    ),
                )
            else:
                derived = self.call(
                    system.name("derive"),
                    (fuel_value, _int(0), length, Var(parameters[0])),
                )
        else:
            # Without a declared depth, a system deriving without spans
            # enumerates derivations at most as deep as it has tokens or
            # axioms and rules together, which every derivation without a
            # repeated rule application along a path fits within.
            if system.depth is not None:
                fuel_value = _int(system.depth)
            else:
                fuel_value = self.primitive("add_int", length, _int(len(system.rules)))
            derived = self.call(
                system.name("derive"),
                (fuel_value, *(Var(parameter) for parameter in parameters)),
            )

        def accept() -> Computation:
            """Answer the scope's unit, accepting the derivation.

            Returns
            -------
            Computation
                ``()``.
            """
            return Return(LiteralValue(None, UNIT))

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
                Var(found),
                UNIT,
                {
                    SPAN: lambda fields: If(
                        self.primitive(
                            "and",
                            self.primitive("eq_int", Var(fields[0]), _int(0)),
                            self.primitive("eq_int", Var(fields[1]), length),
                        ),
                        accept(),
                        self.fail(UNIT, "entry", "extent"),
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
                self.call(system.name("goal"), (Var(found),)),
                lambda is_goal: If(
                    Var(is_goal),
                    whole(found),
                    self.fail(UNIT, "entry", "goal"),
                ),
            ),
        )
        return Handle(
            system.choice.entry.instance,
            search_handler.id,
            Handle(system.weight.entry.instance, collect_handler.id, goal_check, ()),
            (),
        )

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
        return self.elaborator._checked_computation(
            signature, parameters, body, self.system.decl, self.path
        )


__all__ = []
