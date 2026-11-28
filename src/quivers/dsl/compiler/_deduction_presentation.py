"""The rule presentation of a ``deduction`` declaration.

A deduction is read once into a `RulePresentation`: judgment forms for
its items, one rule per sequent, and an axiom slot per kind of input. A
deduction ``T -> T`` derives terms of ``T``, so every item whose head holds
no chart position, such as ``X/Y`` or an atom, is an instance of one form,
`TERM_ITEM`, whose argument is the term; a variable that is an item is
then the same variable as anywhere inside a term. An item whose head holds
positions, such as a lexicon's ``span(i, j, c)``, is a form of its own, and
a bare variable item ranges over the term form and those forms.
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

Reading costs time linear in the size of the declaration, after the
fixed point of `position_slots`, and the presentation is then validated
once by `RulePresentation`; spelling a judgment costs the size of its
arguments.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import itertools

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE
from quivers.dsl.ast_nodes import (
    DeductionDecl,
    LexiconCategoryFixed,
    ObjectEffectApply,
    OptionName,
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
    misplaced_positions,
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
    ComputedRuleWeight,
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

BARE = ""
"""The head of an item that is a bare variable or numeral as read, before
its rule is expanded over the forms it ranges over."""

TERM_ITEM = "_term"
"""The judgment form of the items that are terms: an item whose head holds
no position is the term itself, as the deduction's signature declares."""

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
    """One sequent of a deduction, or one instance of it: its rule and options.

    A sequent whose premise or conclusion is a bare variable ranges over
    every judgment form, so it is the family of its instances, one per
    choice of a form for each such variable; each instance is a rule of
    the presentation with the sequent's weight parameter.

    Parameters
    ----------
    rule : Rule
        The rule of the presentation.
    source : SequentRule
        The declaration.
    index : int
        The declaration's position among the deduction's rules; the
        instances of a rule whose item is a bare variable share it.
    learnable : bool
        Whether the rule reads a learned weight keyed by its bindings.
    bounded : bool
        Whether that weight is bounded below one per firing.
    parent : str or None
        The rule whose weight on the same bindings is added, if any.
    """

    rule: Rule = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    source: SequentRule = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    index: int
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
    atoms : frozenset of str
        The declared atoms; every other identifier is a variable.
    """

    presentation: RulePresentation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    rules: tuple[DeductionRule, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    forms: Mapping[str, TermForm] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    slots: PositionSlots = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    uses_forms: bool
    atoms: frozenset[str] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

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
            variable: a term item is its term.
        """
        head = self.families[judgment.family].name
        if head == TERM_ITEM:
            (term,) = judgment.arguments
            spelled = self.chart_term(term)
            assert not isinstance(spelled, int)
            return spelled
        return self._spelled(
            head, tuple(self.chart_term(term) for term in judgment.arguments)
        )

    def chart_term(self, term: IndexTerm) -> Pattern | int:
        """The chart representation of an index term.

        Parameters
        ----------
        term : IndexTerm
            A rule's argument.

        Returns
        -------
        Pattern or int
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

    def _spelled(self, head: str, arguments: tuple[Pattern | int, ...]) -> Pattern:
        """Spell a head symbol applied to arguments as it was written.

        Parameters
        ----------
        head : str
            The head symbol.
        arguments : tuple of Pattern or int
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
            The judgment form's head and the arguments: `TERM_ITEM` and the
            term for an item whose head holds no position; the head and
            its arguments for one that holds positions; `BARE` and the
            variable or numeral for a bare one, which ranges over the
            forms.

        Raises
        ------
        DeductionPresentationError
            If a head is applied at two arities.
        """
        if isinstance(expr, TypeName):
            if is_pattern_variable(expr, self.atoms):
                self.families.setdefault(TERM_ITEM, [False])
                return BARE, (_Read(kind="variable", name=expr.name),)
            if expr.name.isdigit():
                self.families.setdefault(TERM_ITEM, [False])
                self.form(expr.name, TermForm(kind="numeral", text=expr.name), node)
                self.term_arities[expr.name] = 0
                return BARE, (_Read(kind="numeral", name=expr.name, position=True),)
        symbol = expr.name if isinstance(expr, TypeName) else pattern_symbol(expr)
        if symbol is not None and not any(self.slots.get(symbol, [])):
            self.families.setdefault(TERM_ITEM, [False])
            return TERM_ITEM, (self.term(expr, False, variables, node),)
        if isinstance(expr, TypeName):
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


def _bare_names(items: list[tuple[str, tuple[_Read, ...]]]) -> set[str]:
    """The bare variables among items.

    Parameters
    ----------
    items : list of tuple of str and tuple of _Read
        Items as read.

    Returns
    -------
    set of str
        The names of the items that are bare variables.
    """
    return {
        arguments[0].name
        for head, arguments in items
        if head == BARE and arguments[0].kind == "variable"
    }


def _instances(
    reader: _Reader,
    items: list[tuple[str, tuple[_Read, ...]]],
    forms: list[str],
    variables: dict[str, bool],
) -> list[dict[str, str]]:
    """Every choice of a judgment form for the bare items of a rule.

    A bare variable names one item, so its occurrences take one form. It
    is the term itself (`TERM_ITEM`) when it also occurs inside a term of
    the rule; otherwise it ranges over every form, the term form and the
    forms that hold positions. A bare numeral is the numeral as a term, or
    the position of a form of one position slot.

    Parameters
    ----------
    reader : _Reader
        The declaration's reader, whose families are complete.
    items : list of tuple of str and tuple of _Read
        The rule's premises and conclusion as read.
    forms : list of str
        The judgment forms in declaration order.
    variables : dict of str to bool
        The variables the rule's terms bind.

    Returns
    -------
    list of dict of str to str
        One assignment of a form to each bare variable or numeral, keyed
        by its text; a single empty assignment when there is none.
    """
    positioned = [form for form in forms if form != TERM_ITEM]
    choices: dict[str, list[str]] = {}
    for head, arguments in items:
        if head != BARE:
            continue
        (bare,) = arguments
        if bare.kind == "variable":
            choices[bare.name] = (
                [TERM_ITEM] if bare.name in variables else [TERM_ITEM, *positioned]
            )
            continue
        choices[bare.name] = [
            TERM_ITEM,
            *(form for form in positioned if reader.families[form] == [True]),
        ]
    names = sorted(choices)
    return [
        dict(zip(names, picked, strict=True))
        for picked in itertools.product(*(choices[name] for name in names))
    ]


def _instantiated(
    reader: _Reader,
    item: tuple[str, tuple[_Read, ...]],
    choice: dict[str, str],
    telescope: dict[str, bool],
) -> tuple[str, tuple[_Read, ...]]:
    """An item with its bare variable or numeral given a judgment form.

    A bare variable of the term form is the term variable itself. Of a
    form ``f`` that holds positions it becomes ``f`` applied to fresh
    variables, one per slot, a position where ``f`` has one; the same
    variable gets the same fresh variables wherever it occurs.

    Parameters
    ----------
    reader : _Reader
        The declaration's reader.
    item : tuple of str and tuple of _Read
        The item as read.
    choice : dict of str to str
        The form of each bare variable or numeral.
    telescope : dict of str to bool
        The instance's variables, extended in place with the fresh ones.

    Returns
    -------
    tuple of str and tuple of _Read
        The item's head and arguments.
    """
    head, arguments = item
    if head != BARE:
        return item
    (bare,) = arguments
    form = choice[bare.name]
    if form == TERM_ITEM:
        if bare.kind == "numeral":
            return form, (_Read(kind="constant", name=bare.name),)
        telescope[bare.name] = False
        return form, (_Read(kind="variable", name=bare.name),)
    if bare.kind == "numeral":
        return form, (bare,)
    fresh: list[_Read] = []
    for slot, position in enumerate(reader.families[form]):
        name = f"{bare.name}.{slot}"
        telescope[name] = position
        fresh.append(_Read(kind="variable", name=name, position=position))
    return form, tuple(fresh)


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
    for misplaced in misplaced_positions(decl, reader.slots):
        raise DeductionPresentationError(
            misplaced.message(decl.name), misplaced.line, misplaced.col
        )
    read: list[
        tuple[
            SequentRule,
            list[tuple[str, tuple[_Read, ...]]],
            tuple[str, tuple[_Read, ...]],
            dict[str, bool],
        ]
    ] = []
    for rule in decl.rules:
        variables: dict[str, bool] = {}
        premises = [reader.judgment(item, variables, rule) for item in rule.premises]
        bound = set(variables) | _bare_names(premises)
        conclusion = reader.judgment(rule.conclusion, variables, rule)
        bare = _bare_names([*premises, conclusion])
        positions = sorted(name for name in bare if variables.get(name, False))
        if positions:
            reader.fail(
                f"rule {rule.name!r} uses {', '.join(positions)!r} both as an item "
                "and as a position; a position is not an item",
                rule,
            )
        unbound = [
            name
            for name in (*variables, *_bare_names([conclusion]))
            if name not in bound
        ]
        if unbound:
            reader.fail(
                f"rule {rule.name!r} concludes with {', '.join(unbound)!r}, which "
                "no premise binds",
                rule,
            )
        read.append((rule, premises, conclusion, variables))
    names = {rule.name for rule in decl.rules}
    for rule in decl.rules:
        parent = find_option(rule.options, "parent")
        if parent is not None and (
            not isinstance(parent.value, OptionName) or parent.value.value not in names
        ):
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
    forms = list(reader.families)
    computed_names = {
        rule.name
        for rule in decl.rules
        if find_option(rule.options, "weight") is not None
    }
    normalize = find_option(decl.options, "normalize")
    normalizes = (
        normalize is not None
        and isinstance(normalize.value, OptionName)
        and normalize.value.value == "lhs"
    )
    for index, (rule, premises, conclusion, variables) in enumerate(read):
        learnable = get_option_flag(rule.options, "learnable")
        parent = find_option(rule.options, "parent")
        parent_name = (
            parent.value.value
            if parent is not None and isinstance(parent.value, OptionName)
            else None
        )
        weights = tuple(option for option in rule.options if option.key == "weight")
        weight: ParameterId | ComputedRuleWeight | None = None
        if weights:
            if len(weights) != 1 or not isinstance(weights[0].value, OptionName):
                raise DeductionPresentationError(
                    f"rule {rule.name!r} weight must name one checked computation",
                    rule.line,
                    rule.col,
                )
            if (
                any(
                    option.key in {"learnable", "bounded", "parent"}
                    for option in rule.options
                )
                or normalizes
            ):
                raise DeductionPresentationError(
                    f"rule {rule.name!r} computed weight cannot be combined with "
                    "learnable, bounded, parent, or normalize=lhs",
                    rule.line,
                    rule.col,
                )
            weight = ComputedRuleWeight(operation=weights[0].value.value)
        elif parent_name in computed_names:
            raise DeductionPresentationError(
                f"rule {rule.name!r} cannot inherit a computed weight from {parent_name!r}",
                rule.line,
                rule.col,
            )
        elif learnable or parent_name is not None:
            weight = ParameterId.derive(decl.name, "rule", rule.name)
        choices = _instances(reader, [*premises, conclusion], forms, variables)
        for choice in choices:
            telescope = dict(variables)
            items = [
                _instantiated(reader, item, choice, telescope)
                for item in (*premises, conclusion)
            ]
            suffix = (
                tuple(choice[key] for key in sorted(choice)) if len(choices) > 1 else ()
            )
            presented = Rule(
                id=ConstructorId.derive(decl.name, "rule", rule.name, *suffix),
                name="@".join((rule.name, *suffix)),
                telescope=tuple(
                    IndexBinder(name=name, sort=sort_of(position))
                    for name, position in telescope.items()
                ),
                premises=tuple(
                    Premise(judgment=judgment(head, arguments))
                    for head, arguments in items[:-1]
                ),
                conclusion=judgment(*items[-1]),
                weight=weight,
            )
            rules.append(
                DeductionRule(
                    rule=presented,
                    source=rule,
                    index=index,
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
        atoms=reader.atoms,
    )


def category_term(presentation: DeductionPresentation, expr: ObjectExpr) -> IndexTerm:
    """The ``Term`` of a category the lexicon names.

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
        The category; an identifier that is not a declared atom is a
        variable over categories.
    """
    (sort,) = presentation.presentation.index_sorts
    if is_pattern_variable(expr, presentation.atoms):
        assert isinstance(expr, TypeName)
        return IndexVariable(name=expr.name, sort=sort)
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
    """The chart representation of a category the lexicon names.

    Parameters
    ----------
    presentation : DeductionPresentation
        The deduction's presentation, whose forms spell the category.
    expr : ObjectExpr
        The category as written.

    Returns
    -------
    Pattern
        The category in the chart representation; a ``Term`` is never a
        position.
    """
    spelled = presentation.chart_term(category_term(presentation, expr))
    assert not isinstance(spelled, int)
    return spelled


__all__: list[str] = []
