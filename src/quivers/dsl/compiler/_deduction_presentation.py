"""The rule presentation of a ``deduction`` declaration.

A deduction is read once into a `RulePresentation`: one judgment form
per item head, one rule per sequent, and an axiom slot per kind of input.
Everything else reads the presentation: the agenda compiler derives its
`InferenceRule` patterns from the rules' judgments, and the QIEC
elaboration derives its pattern records from them, so a pattern is
converted from the source in exactly one place.

Items and their arguments are index terms. A slot that holds a chart
position (`position_slots`) is a natural number; every other argument is
a term of one closed sort, ``Term``, whose constructors are the atoms and
applied symbols the declaration mentions. Each head symbol, of a
judgment form or of a ``Term`` constructor, keeps the `TermForm` it was
written in, which says how the chart representation of
`quivers.dsl.compiler._deduction_terms` spells it: an atom, a numeral, a
slash, a product, or an application.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE
from quivers.dsl.ast_nodes import (
    DeductionDecl,
    LexiconCategoryFixed,
    ObjectEffectApply,
    ObjectExpr,
    ObjectProduct,
    ObjectSlash,
    SequentRule,
    TypeName,
)
from quivers.dsl.compiler._deduction_positions import (
    SPAN_SYMBOL,
    PositionSlots,
    has_lexicon,
    is_pattern_variable,
    pattern_arguments,
    pattern_symbol,
    position_slots,
)
from quivers.dsl.compiler._deduction_terms import atom_term
from quivers.dsl.compiler._lexicon_files import load_lexicon_tsv
from quivers.dsl.compiler._options import find_option, get_option_flag
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import ParameterId, PresentationId
from quivers.qiec.declarations import FieldDef
from quivers.qiec.identifiers import ConstructorId, FamilyId
from quivers.qiec.kinds import NAT, IndexBinder, IndexSort, UserIndexSort
from quivers.qiec.rules import (
    AxiomSlot,
    Judgment,
    JudgmentDecl,
    Premise,
    Rule,
    RulePresentation,
)
from quivers.qiec.types import (
    STRING,
    IndexConstructor,
    IndexLiteral,
    IndexTerm,
    IndexVariable,
)
from quivers.stochastic.agenda import Pattern, Wildcard

TERM_SORT = "Term"
"""The name of the closed sort of a deduction's non-position arguments."""

FORM_VARIABLE = "_form"
"""The prefix of the variables a span pattern gains for its logical form."""

type FormKind = Literal["atom", "numeral", "slash", "product", "application"]
"""How a head symbol is written."""


class DeductionPresentationError(ValueError):
    """A deduction has no rule presentation.

    Parameters
    ----------
    message : str
        What is wrong.
    line : int
        The source line.
    col : int
        The source column.
    """

    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(message)
        self.line = line
        self.col = col


class TermForm(QiecModel):
    """How one head symbol is written in the source.

    Parameters
    ----------
    kind : FormKind
        An atom, a numeral, a slash, a product, or an application.
    text : str
        The atom's name, the numeral's digits, the slash (``"/"`` or
        ``"\\\\"``), or the applied symbol; empty for a product.
    """

    kind: FormKind
    text: str = ""


class DeductionRule(QiecModel):
    """One sequent of a deduction: its presentation rule and its options.

    Parameters
    ----------
    rule : Rule
        The rule of the presentation.
    source : SequentRule
        The declaration.
    learnable : bool
        Whether the rule reads a learned weight keyed by its bindings.
    bounded : bool
        Whether that weight is bounded below one per firing.
    parent : str or None
        The rule whose weight on the same bindings is added, if any.
    """

    rule: Rule = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    source: SequentRule = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    learnable: bool
    bounded: bool
    parent: str | None

    @property
    def weighted(self) -> bool:
        """Whether the rule reads a learned weight, its own or a parent's.

        Returns
        -------
        bool
            ``learnable`` or a parent.
        """
        return self.learnable or self.parent is not None


class DeductionPresentation(QiecModel):
    """A deduction read into its rule presentation.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    rules : tuple of DeductionRule
        Its rules with their options, in source order.
    forms : Mapping of str to TermForm
        How each head symbol, of a judgment form or a ``Term``
        constructor, is written.
    slots : PositionSlots
        Which slot of each applied symbol holds a position.
    uses_forms : bool
        Whether span items carry a logical form slot.
    """

    presentation: RulePresentation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    rules: tuple[DeductionRule, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    forms: Mapping[str, TermForm] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    slots: PositionSlots = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    uses_forms: bool

    @property
    def families(self) -> Mapping[FamilyId, JudgmentDecl]:
        """The judgment forms by identity.

        Returns
        -------
        Mapping of FamilyId to JudgmentDecl
            The presentation's table.
        """
        return self.presentation.judgments_by_id

    def chart_pattern(self, judgment: Judgment) -> Pattern:
        """The agenda pattern of a judgment.

        Parameters
        ----------
        judgment : Judgment
            A judgment of the presentation.

        Returns
        -------
        Pattern
            The item in the chart representation, with a `Wildcard` per
            variable.
        """
        head = self.families[judgment.family].name
        return self._spelled(
            head, tuple(self.chart_term(term) for term in judgment.arguments)
        )

    def chart_term(self, term: IndexTerm) -> Pattern:
        """The chart representation of an index term.

        Parameters
        ----------
        term : IndexTerm
            A rule's argument.

        Returns
        -------
        Pattern
            A position, a `Wildcard`, or a spelled term.

        Raises
        ------
        TypeError
            If the term is not of a form a deduction writes.
        """
        if isinstance(term, IndexVariable):
            return Wildcard(name=term.name)
        if isinstance(term, IndexLiteral):
            if isinstance(term.value, int):
                return term.value
            return self._spelled(term.value, ())
        if isinstance(term, IndexConstructor):
            return self._spelled(
                term.name, tuple(self.chart_term(item) for item in term.arguments)
            )
        raise TypeError(f"a deduction writes no index term {term!r}")

    def _spelled(self, head: str, arguments: tuple[Pattern, ...]) -> Pattern:
        """Spell a head symbol applied to arguments as it was written.

        Parameters
        ----------
        head : str
            The head symbol.
        arguments : tuple of Pattern
            Its spelled arguments.

        Returns
        -------
        Pattern
            The spelling the chart representation uses for its form.
        """
        form = self.forms[head]
        if form.kind == "atom":
            return atom_term(form.text)
        if form.kind == "numeral":
            return ("literal", int(form.text))
        if form.kind == "slash":
            return (form.text, *arguments)
        if form.kind == "product":
            return ("product", arguments)
        return (form.text, *arguments)


class _Reader:
    """Reads one declaration's patterns into index terms.

    Parameters
    ----------
    decl : DeductionDecl
        The declaration.
    """

    def __init__(self, decl: DeductionDecl) -> None:
        self.decl = decl
        self.atoms = frozenset(decl.atoms)
        self.slots = position_slots(decl)
        self.forms: dict[str, TermForm] = {}
        self.term_arities: dict[str, int] = {}
        self.families: dict[str, list[bool]] = {}
        self.uses_forms = bool(decl.binders) or any(
            isinstance(pattern, ObjectEffectApply)
            and pattern.effect == SPAN_SYMBOL
            and len(pattern.args) >= 4
            for rule in decl.rules
            for pattern in (*rule.premises, rule.conclusion)
        )
        self.fresh = 0

    def fail(self, message: str, node: SequentRule | DeductionDecl) -> None:
        """Refuse the declaration.

        Parameters
        ----------
        message : str
            What is wrong.
        node : SequentRule or DeductionDecl
            Where.

        Raises
        ------
        DeductionPresentationError
            Always.
        """
        raise DeductionPresentationError(
            f"deduction {self.decl.name!r}: {message}", node.line, node.col
        )

    def form(
        self, head: str, form: TermForm, node: SequentRule | DeductionDecl
    ) -> None:
        """Record how a head symbol is written.

        Parameters
        ----------
        head : str
            The symbol.
        form : TermForm
            Its form.
        node : SequentRule
            The rule that writes it.

        Raises
        ------
        DeductionPresentationError
            If the symbol is written in two forms.
        """
        known = self.forms.setdefault(head, form)
        if known != form:
            self.fail(f"{head!r} is written both as {known.kind} and {form.kind}", node)

    def term(
        self,
        expr: ObjectExpr,
        position: bool,
        variables: dict[str, bool],
        node: SequentRule | DeductionDecl,
    ) -> _Read:
        """The argument as read.

        Parameters
        ----------
        expr : ObjectExpr
            The argument as written.
        position : bool
            Whether its slot holds a position.
        variables : dict of str to bool
            Whether each of the rule's variables is a position, extended in place.
        node : SequentRule or DeductionDecl
            The rule, or the declaration for a lexicon category.

        Returns
        -------
        _Read
            The argument.

        Raises
        ------
        DeductionPresentationError
            If a variable is both a position and a term, or the argument
            has a form the pattern language does not cover.
        """
        if is_pattern_variable(expr, self.atoms):
            assert isinstance(expr, TypeName)
            known = variables.setdefault(expr.name, position)
            if known != position:
                self.fail(
                    f"rule {node.name!r} uses {expr.name!r} both as a position and "
                    "as a term",
                    node,
                )
            return _Read(kind="variable", name=expr.name, position=position)
        if isinstance(expr, TypeName):
            if expr.name.isdigit() and position:
                return _Read(kind="numeral", name=expr.name, position=True)
            kind: FormKind = "numeral" if expr.name.isdigit() else "atom"
            self.form(expr.name, TermForm(kind=kind, text=expr.name), node)
            self.term_arities[expr.name] = 0
            return _Read(kind="constant", name=expr.name)
        symbol = pattern_symbol(expr)
        if symbol is None:
            self.fail(
                f"rule {node.name!r} has a pattern of kind {expr.kind!r}, which the "
                "pattern language does not cover",
                node,
            )
        assert symbol is not None
        self.form(symbol, _form_of(expr), node)
        arguments = self.arguments(symbol, expr, variables, node)
        self.term_arities[symbol] = len(arguments)
        return _Read(kind="application", name=symbol, arguments=arguments)

    def arguments(
        self,
        symbol: str,
        expr: ObjectExpr,
        variables: dict[str, bool],
        node: SequentRule | DeductionDecl,
    ) -> tuple[_Read, ...]:
        """The arguments of an applied symbol, as read.

        Parameters
        ----------
        symbol : str
            The symbol.
        expr : ObjectExpr
            The application.
        variables : dict of str to bool
            Whether each of the rule's variables is a position.
        node : SequentRule or DeductionDecl
            The rule.

        Returns
        -------
        tuple of _Read
            One per slot.
        """
        flags = self.slots.get(symbol, [])
        return tuple(
            self.term(argument, index < len(flags) and flags[index], variables, node)
            for index, argument in enumerate(pattern_arguments(expr))
        )

    def judgment(
        self, expr: ObjectExpr, variables: dict[str, bool], node: SequentRule
    ) -> tuple[str, tuple[_Read, ...]]:
        """The head and arguments of an item pattern.

        A span pattern without its logical form slot gains a fresh
        variable there when the deduction's spans carry one.

        Parameters
        ----------
        expr : ObjectExpr
            The item as written.
        variables : dict of str to bool
            Whether each of the rule's variables is a position.
        node : SequentRule
            The rule.

        Returns
        -------
        tuple of str and tuple of _Read
            The judgment form's head and the arguments.

        Raises
        ------
        DeductionPresentationError
            If the item is a variable or a numeral, which names no
            judgment form, or a head is applied at two arities.
        """
        if isinstance(expr, TypeName):
            if is_pattern_variable(expr, self.atoms) or expr.name.isdigit():
                self.fail(
                    f"rule {node.name!r} has the item {expr.name!r}, which names no "
                    "judgment; an item is an atom or an applied symbol",
                    node,
                )
            self.form(expr.name, TermForm(kind="atom", text=expr.name), node)
            head, arguments = expr.name, ()
        else:
            head = pattern_symbol(expr) or ""
            if not head:
                self.fail(
                    f"rule {node.name!r} has an item of kind {expr.kind!r}, which "
                    "the pattern language does not cover",
                    node,
                )
            self.form(head, _form_of(expr), node)
            arguments = self.arguments(head, expr, variables, node)
        if head == SPAN_SYMBOL and self.uses_forms and len(arguments) == 3:
            self.fresh += 1
            name = f"{FORM_VARIABLE}{self.fresh}"
            variables[name] = False
            arguments = (*arguments, _Read(kind="variable", name=name))
        flags = self.slots.get(head, [])
        positions = [
            index < len(flags) and flags[index] for index in range(len(arguments))
        ]
        known = self.families.setdefault(head, positions)
        if len(known) != len(positions):
            self.fail(
                f"{head!r} is applied to both {len(known)} and {len(positions)} "
                "arguments",
                node,
            )
        return head, arguments


class _Read(QiecModel):
    """An argument as read, before the ``Term`` sort is known.

    Parameters
    ----------
    kind : str
        ``"variable"``, ``"numeral"`` (a position), ``"constant"`` or
        ``"application"``.
    name : str
        The variable, the digits, the constant or the applied symbol.
    position : bool
        Whether a variable is a position.
    arguments : tuple of _Read
        An application's arguments.
    """

    kind: Literal["variable", "numeral", "constant", "application"]
    name: str
    position: bool = False
    arguments: tuple[_Read, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


def _form_of(expr: ObjectExpr) -> TermForm:
    """The form of an applied pattern.

    Parameters
    ----------
    expr : ObjectExpr
        A slash, a product, or an application.

    Returns
    -------
    TermForm
        Its form.
    """
    if isinstance(expr, ObjectSlash):
        return TermForm(kind="slash", text=expr.direction)
    if isinstance(expr, ObjectProduct):
        return TermForm(kind="product")
    assert isinstance(expr, ObjectEffectApply)
    return TermForm(kind="application", text=expr.effect)


def _index_term(read: _Read, sort: UserIndexSort | None) -> IndexTerm:
    """The index term of an argument as read.

    Parameters
    ----------
    read : _Read
        The argument.
    sort : UserIndexSort or None
        The ``Term`` sort, ``None`` when the deduction has no terms.

    Returns
    -------
    IndexTerm
        The term: a position is a natural number, anything else a
        ``Term``.
    """
    if read.kind == "numeral":
        return IndexLiteral(value=int(read.name), sort=NAT)
    if read.kind == "variable" and read.position:
        return IndexVariable(name=read.name, sort=NAT)
    assert sort is not None
    if read.kind == "variable":
        return IndexVariable(name=read.name, sort=sort)
    if read.kind == "constant":
        return IndexLiteral(value=read.name, sort=sort)
    return IndexConstructor(
        name=read.name,
        arguments=tuple(_index_term(item, sort) for item in read.arguments),
        sort=sort,
    )


def deduction_presentation(decl: DeductionDecl) -> DeductionPresentation:
    """Read a deduction into its rule presentation.

    Parameters
    ----------
    decl : DeductionDecl
        The declaration.

    Returns
    -------
    DeductionPresentation
        The presentation, its rules' options, and the forms of its
        symbols.

    Raises
    ------
    DeductionPresentationError
        If a pattern names no judgment, a symbol is written in two forms
        or at two arities, a variable is both a position and a term, or a
        rule's parent is not declared.
    """
    reader = _Reader(decl)
    read: list[tuple[SequentRule, list, tuple, dict[str, IndexSort]]] = []
    for rule in decl.rules:
        variables: dict[str, bool] = {}
        premises = [reader.judgment(item, variables, rule) for item in rule.premises]
        conclusion = reader.judgment(rule.conclusion, variables, rule)
        read.append((rule, premises, conclusion, variables))
    names = {rule.name for rule in decl.rules}
    for rule in decl.rules:
        parent = find_option(rule.options, "parent")
        if parent is not None and getattr(parent.value, "value", None) not in names:
            reader.fail(
                f"rule {rule.name!r} names a parent the deduction does not declare",
                rule,
            )
    for entry in decl.lexicon:
        if isinstance(entry.category, LexiconCategoryFixed):
            reader.term(entry.category.category, False, {}, decl)
    if decl.lexicon_from_file is not None:
        for _word, category, _form in load_lexicon_tsv(decl.lexicon_from_file, decl):
            reader.term(category, False, {}, decl)
    for atom in decl.atoms:
        reader.forms.setdefault(atom, TermForm(kind="atom", text=atom))
        reader.term_arities.setdefault(atom, 0)
    if has_lexicon(decl):
        reader.families.setdefault(
            SPAN_SYMBOL, [True, True, False, False][: 4 if reader.uses_forms else 3]
        )
        reader.forms.setdefault(
            SPAN_SYMBOL, TermForm(kind="application", text=SPAN_SYMBOL)
        )
    constructors = tuple(sorted(reader.term_arities))
    term_sort = (
        UserIndexSort(
            name=TERM_SORT,
            constructors=constructors,
            arities=tuple(reader.term_arities[name] for name in constructors),
        )
        if constructors
        else None
    )

    def sort_of(position: bool) -> IndexSort:
        """The sort of a slot.

        Parameters
        ----------
        position : bool
            Whether it holds a position.

        Returns
        -------
        IndexSort
            ``Nat`` or ``Term``.
        """
        if position:
            return NAT
        assert term_sort is not None
        return term_sort

    family_ids = {
        head: FamilyId.derive(decl.name, "judgment", head) for head in reader.families
    }
    judgments = tuple(
        JudgmentDecl(
            id=family_ids[head],
            name=head,
            indices=tuple(
                IndexBinder(name=f"x{index}", sort=sort_of(flag), refinable=True)
                for index, flag in enumerate(flags)
            ),
        )
        for head, flags in reader.families.items()
    )

    def judgment(head: str, arguments: tuple[_Read, ...]) -> Judgment:
        """A judgment over the final sorts.

        Parameters
        ----------
        head : str
            The judgment form's head.
        arguments : tuple of _Read
            Its arguments.

        Returns
        -------
        Judgment
            The judgment.
        """
        return Judgment(
            family=family_ids[head],
            arguments=tuple(_index_term(item, term_sort) for item in arguments),
        )

    rules: list[DeductionRule] = []
    for rule, premises, conclusion, variables in read:
        learnable = get_option_flag(rule.options, "learnable")
        parent = find_option(rule.options, "parent")
        parent_name = (
            str(getattr(parent.value, "value")) if parent is not None else None
        )
        presented = Rule(
            id=ConstructorId.derive(decl.name, "rule", rule.name),
            name=rule.name,
            telescope=tuple(
                IndexBinder(name=name, sort=sort_of(position))
                for name, position in variables.items()
            ),
            premises=tuple(
                Premise(judgment=judgment(head, arguments))
                for head, arguments in premises
            ),
            conclusion=judgment(*conclusion),
            weight=(
                ParameterId.derive(decl.name, "rule", rule.name)
                if learnable or parent_name is not None
                else None
            ),
        )
        rules.append(
            DeductionRule(
                rule=presented,
                source=rule,
                learnable=learnable,
                bounded=get_option_flag(rule.options, "bounded"),
                parent=parent_name,
            )
        )
    axioms: list[AxiomSlot] = []
    for decl_judgment in judgments:
        binders = tuple(
            IndexBinder(name=binder.name, sort=binder.sort)
            for binder in decl_judgment.indices
            if isinstance(binder, IndexBinder)
        )
        lexical = has_lexicon(decl) and decl_judgment.name == SPAN_SYMBOL
        axioms.append(
            AxiomSlot(
                id=ConstructorId.derive(decl.name, "axiom", decl_judgment.name),
                name=f"axiom_{decl_judgment.name}",
                telescope=binders,
                judgment=Judgment(
                    family=decl_judgment.id,
                    arguments=tuple(
                        IndexVariable(name=binder.name, sort=binder.sort)
                        for binder in binders
                    ),
                ),
                values=(FieldDef(name="word", type=STRING),) if lexical else (),
            )
        )
    presentation = RulePresentation(
        id=PresentationId.derive(decl.name),
        name=decl.name,
        index_sorts=(term_sort,) if term_sort is not None else (),
        judgments=judgments,
        rules=tuple(item.rule for item in rules),
        axioms=tuple(axioms),
    )
    return DeductionPresentation(
        presentation=presentation,
        rules=tuple(rules),
        forms=dict(reader.forms),
        slots=reader.slots,
        uses_forms=reader.uses_forms,
    )


def category_term(presentation: DeductionPresentation, expr: ObjectExpr) -> IndexTerm:
    """The ``Term`` of a ground category the lexicon names.

    Parameters
    ----------
    presentation : DeductionPresentation
        The deduction's presentation, whose ``Term`` sort holds every
        category the lexicon names.
    expr : ObjectExpr
        The category as written.

    Returns
    -------
    IndexTerm
        The category.
    """
    (sort,) = presentation.presentation.index_sorts
    if isinstance(expr, TypeName):
        return IndexLiteral(value=expr.name, sort=sort)
    symbol = pattern_symbol(expr)
    assert symbol is not None
    return IndexConstructor(
        name=symbol,
        arguments=tuple(
            category_term(presentation, item) for item in pattern_arguments(expr)
        ),
        sort=sort,
    )


def chart_category(presentation: DeductionPresentation, expr: ObjectExpr) -> Pattern:
    """The chart representation of a ground category the lexicon names.

    Parameters
    ----------
    presentation : DeductionPresentation
        The deduction's presentation, whose forms spell the category.
    expr : ObjectExpr
        The category as written.

    Returns
    -------
    Pattern
        The category in the chart representation.
    """
    return presentation.chart_term(category_term(presentation, expr))


__all__: list[str] = []
