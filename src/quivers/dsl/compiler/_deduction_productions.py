"""Production groups of a deduction, shared by both evaluators.

A deduction with a lexicon is a grammar whose productions are its rules
and its expanded lexicon entries. The ``normalize=lhs`` option makes the
productions that share a left-hand side a probability distribution. The
left-hand side of a production is defined by its form:

* a rule whose conclusion is a ``span`` item rewrites the category in
  the item's third slot, so ``span(I, J, NP)`` has left-hand side
  ``NP`` and ``span(I, J, Fwd(NP, N))`` has ``Fwd(NP, N)``;
* a lexicon entry ``"the" : Det`` rewrites ``Det``, the category of the
  ``span(i, i + 1, Det)`` axiom it injects, so it shares a group with
  the rules concluding ``span(I, J, Det)``; a wildcard or restricted
  entry expands to one entry per candidate atom, each in its atom's
  group;
* a rule whose conclusion is any other item rewrites that item with its
  chart positions erased: ``np(I, J)`` has left-hand side ``np(_, _)``,
  and a conclusion with no position slot is its own left-hand side.

Unary and binary rules are treated alike: the premises are the
right-hand side and never enter the grouping. A group is fixed when the
deduction compiles only if no production's left-hand side, and no
production's weight key, depends on how a rule's variables are bound.
So a rule that binds a variable outside the chart positions (a category
variable such as the ``X`` of ``span(I, K, Fwd(X, Y)), span(K, J, Y) |-
span(I, J, X)``, whose instances are one weight per binding, or the
logical-form variable every span of a system with forms carries) is
refused, as is a lexicon whose categories are not ground. The analysis
here is read by the agenda compiler and by the QIEC elaboration, so the
two evaluators normalize over the same groups in the same order.
"""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    DeductionDecl,
    LexiconCategoryFixed,
    LexiconCategoryRestricted,
    LexiconCategoryWildcard,
    ObjectEffectApply,
    ObjectExpr,
    ObjectProduct,
    ObjectSlash,
    OptionName,
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
    rule_position_variables,
)
from quivers.dsl.compiler._options import find_option, get_option_flag
from quivers.stochastic.productions import Production, ProductionGroup


#: The option that selects per-left-hand-side normalization.
NORMALIZE_OPTION = "normalize"

#: The values ``normalize`` accepts.
NORMALIZE_MODES: tuple[str, ...] = ("lhs", "none")

#: The semirings whose chart reads production weights.
WEIGHTED_SEMIRINGS: tuple[str, ...] = ("LogProb", "Viterbi")

type NormalizeMode = Literal["lhs", "none"]


class ProductionRefusal(dx.Model):
    """A reason a deduction's production groups are not fixed.

    Parameters
    ----------
    message : str
        The diagnostic, naming the deduction and the offending part.
    line : int
        The source line of the offending part.
    col : int
        The source column of the offending part.
    """

    message: str
    line: int
    col: int


class ProductionAnalysis(dx.Model):
    """A deduction's production groups, or why it has none.

    Parameters
    ----------
    groups : tuple[ProductionGroup, ...]
        The groups, in order of each left-hand side's first production;
        empty when refused.
    refusals : tuple[ProductionRefusal, ...]
        Every reason the groups are not fixed, in source order.
    """

    groups: tuple[ProductionGroup, ...] = ()
    refusals: tuple[ProductionRefusal, ...] = ()


def normalization_mode(
    decl: DeductionDecl,
) -> tuple[NormalizeMode, ProductionRefusal | None]:
    """Read a deduction's ``normalize`` option.

    Parameters
    ----------
    decl : DeductionDecl
        The deduction.

    Returns
    -------
    tuple[NormalizeMode, ProductionRefusal | None]
        The mode, ``"none"`` when the option is absent, and a refusal
        when the option's value is not one of `NORMALIZE_MODES`.
    """
    entry = find_option(decl.options, NORMALIZE_OPTION)
    if entry is None:
        return "none", None
    value = entry.value
    if isinstance(value, OptionName) and value.value == "lhs":
        return "lhs", None
    if isinstance(value, OptionName) and value.value == "none":
        return "none", None
    shown = value.value if isinstance(value, OptionName) else type(value).__name__
    return "none", ProductionRefusal(
        message=(
            f"deduction {decl.name!r}: option normalize={shown} is not one of "
            f"{', '.join(NORMALIZE_MODES)}; normalize=lhs makes the productions "
            "sharing a left-hand side a probability distribution, and "
            "normalize=none keeps free weights"
        ),
        line=entry.line or decl.line,
        col=entry.col or decl.col,
    )


def render_pattern(expr: ObjectExpr, *, nested: bool = False) -> str:
    """Render a ground pattern as QVR source writes it.

    Parameters
    ----------
    expr : ObjectExpr
        The pattern: an identifier, an application, a slash, or a
        product.
    nested : bool
        Whether the pattern is an argument of a slash or product, which
        then takes parentheses.

    Returns
    -------
    str
        The rendering.

    Raises
    ------
    ValueError
        If the pattern is of another kind, which no rule or lexicon
        category takes.
    """
    if isinstance(expr, TypeName):
        return expr.name
    if isinstance(expr, ObjectEffectApply):
        inner = ", ".join(render_pattern(argument) for argument in expr.args)
        return f"{expr.effect}({inner})"
    if isinstance(expr, ObjectSlash):
        text = (
            f"{render_pattern(expr.result, nested=True)}{expr.direction}"
            f"{render_pattern(expr.argument, nested=True)}"
        )
        return f"({text})" if nested else text
    if isinstance(expr, ObjectProduct):
        text = " * ".join(
            render_pattern(component, nested=True) for component in expr.components
        )
        return f"({text})" if nested else text
    raise ValueError(f"a {type(expr).__name__} pattern has no rendering")


def _erased(expr: ObjectExpr, slots: PositionSlots) -> str:
    """Render an item with its position slots written ``_``.

    Parameters
    ----------
    expr : ObjectExpr
        The item pattern.
    slots : PositionSlots
        The deduction's position slots.

    Returns
    -------
    str
        The rendering.
    """
    symbol = pattern_symbol(expr)
    if symbol is None or not isinstance(expr, ObjectEffectApply):
        return render_pattern(expr)
    flags = slots.get(symbol, [])
    parts = [
        "_" if index < len(flags) and flags[index] else render_pattern(argument)
        for index, argument in enumerate(expr.args)
    ]
    return f"{expr.effect}({', '.join(parts)})"


def _left_hand_side(expr: ObjectExpr, slots: PositionSlots) -> tuple[str, str]:
    """The grouping key and label of a conclusion's left-hand side.

    Parameters
    ----------
    expr : ObjectExpr
        The conclusion.
    slots : PositionSlots
        The deduction's position slots.

    Returns
    -------
    tuple[str, str]
        ``("span", category)`` for a span item, ``("item", erased)``
        otherwise; the second component is the group's label.
    """
    if (
        isinstance(expr, ObjectEffectApply)
        and expr.effect == SPAN_SYMBOL
        and len(expr.args) == 3
    ):
        return "span", render_pattern(expr.args[2])
    return "item", _erased(expr, slots)


def _variables(expr: ObjectExpr, atoms: frozenset[str]) -> list[str]:
    """The variables a pattern mentions, in first-occurrence order.

    Parameters
    ----------
    expr : ObjectExpr
        The pattern.
    atoms : frozenset[str]
        The deduction's atoms.

    Returns
    -------
    list[str]
        The names.
    """
    if is_pattern_variable(expr, atoms):
        assert isinstance(expr, TypeName)
        return [expr.name]
    found: list[str] = []
    for argument in pattern_arguments(expr):
        for name in _variables(argument, atoms):
            if name not in found:
                found.append(name)
    return found


def _uses_forms(decl: DeductionDecl) -> bool:
    """Whether a deduction's span items carry a logical form.

    Parameters
    ----------
    decl : DeductionDecl
        The deduction.

    Returns
    -------
    bool
        ``True`` with a ``binders`` block or a four-slot span pattern.
    """
    return bool(decl.binders) or any(
        isinstance(pattern, ObjectEffectApply)
        and pattern.effect == SPAN_SYMBOL
        and len(pattern.args) >= 4
        for rule in decl.rules
        for pattern in (*rule.premises, rule.conclusion)
    )


def _semiring(decl: DeductionDecl) -> str:
    """The deduction's semiring name, ``LogProb`` by default.

    Parameters
    ----------
    decl : DeductionDecl
        The deduction.

    Returns
    -------
    str
        The name as written.
    """
    entry = find_option(decl.options, "semiring")
    if entry is not None and isinstance(entry.value, OptionName):
        return entry.value.value
    return "LogProb"


def _rule_refusal(
    decl: DeductionDecl, rule: SequentRule, slots: PositionSlots
) -> ProductionRefusal | None:
    """Why one rule's group or weight is not fixed, if it is not.

    Parameters
    ----------
    decl : DeductionDecl
        The deduction.
    rule : SequentRule
        The rule.
    slots : PositionSlots
        The deduction's position slots.

    Returns
    -------
    ProductionRefusal | None
        The refusal, or ``None`` for a rule that binds only positions
        and carries neither ``bounded`` nor ``parent``.
    """
    atoms = frozenset(decl.atoms)
    for flag in ("bounded", "parent"):
        if get_option_flag(rule.options, flag) or (
            flag == "parent" and find_option(rule.options, flag) is not None
        ):
            return ProductionRefusal(
                message=(
                    f"deduction {decl.name!r}: rule {rule.name!r} declares "
                    f"{flag!r}, which reparameterizes its weight; a grammar "
                    "whose weights are production probabilities gives each "
                    "production one weight from its left-hand side's "
                    f"distribution, so remove {flag!r}"
                ),
                line=rule.line,
                col=rule.col,
            )
    positions = rule_position_variables(decl, slots, rule)
    bound: list[str] = []
    for pattern in (*rule.premises, rule.conclusion):
        for name in _variables(pattern, atoms):
            if name not in positions and name not in bound:
                bound.append(name)
    if bound:
        names = ", ".join(repr(name) for name in bound)
        return ProductionRefusal(
            message=(
                f"deduction {decl.name!r}: rule {rule.name!r} binds {names} "
                "outside the chart positions, so it stands for one production "
                "per binding and its left-hand side's group is not fixed when "
                "the deduction compiles; write one rule per production, with "
                "ground categories"
            ),
            line=rule.line,
            col=rule.col,
        )
    try:
        _left_hand_side(rule.conclusion, slots)
    except ValueError as error:
        return ProductionRefusal(
            message=f"deduction {decl.name!r}: rule {rule.name!r}: {error}",
            line=rule.line,
            col=rule.col,
        )
    return None


def production_groups(
    decl: DeductionDecl,
    slots: PositionSlots,
    file_rows: tuple[tuple[str, TypeName], ...] = (),
) -> ProductionAnalysis:
    """Group a deduction's productions by left-hand side.

    Parameters
    ----------
    decl : DeductionDecl
        The deduction.
    slots : PositionSlots
        Its position slots, from `position_slots`.
    file_rows : tuple[tuple[str, TypeName], ...]
        The word and category of each row of the deduction's lexicon
        file, as its loader read them; empty without a file.

    Returns
    -------
    ProductionAnalysis
        The groups when every production's left-hand side and weight
        are fixed at compile time, else every reason they are not.
    """
    refusals: list[ProductionRefusal] = []
    semiring = _semiring(decl)
    if semiring not in WEIGHTED_SEMIRINGS:
        refusals.append(
            ProductionRefusal(
                message=(
                    f"deduction {decl.name!r}: the {semiring} semiring reads no "
                    "production weights; per-left-hand-side weights need "
                    f"{' or '.join(WEIGHTED_SEMIRINGS)}"
                ),
                line=decl.line,
                col=decl.col,
            )
        )
    if find_option(decl.options, "axioms") is not None or not has_lexicon(decl):
        refusals.append(
            ProductionRefusal(
                message=(
                    f"deduction {decl.name!r} takes its axioms as input rather "
                    "than from a lexicon, so its axioms are not productions of "
                    "a grammar and no left-hand side groups them"
                ),
                line=decl.line,
                col=decl.col,
            )
        )
    if _uses_forms(decl):
        refusals.append(
            ProductionRefusal(
                message=(
                    f"deduction {decl.name!r}: its span items carry logical "
                    "forms, so every rule binds a form outside the chart "
                    "positions and stands for one production per form"
                ),
                line=decl.line,
                col=decl.col,
            )
        )
    for rule in decl.rules:
        refusal = _rule_refusal(decl, rule, slots)
        if refusal is not None:
            refusals.append(refusal)
    atoms = frozenset(decl.atoms)
    lexical: list[tuple[str, ObjectExpr | str, bool, int, int]] = []
    for entry in decl.lexicon:
        learnable = get_option_flag(entry.options, "learnable")
        for word in entry.words:
            category = entry.category
            if isinstance(category, LexiconCategoryFixed):
                lexical.append(
                    (word, category.category, learnable, entry.line, entry.col)
                )
            elif isinstance(category, LexiconCategoryWildcard):
                lexical.extend(
                    (word, atom, True, entry.line, entry.col) for atom in decl.atoms
                )
            elif isinstance(category, LexiconCategoryRestricted):
                lexical.extend(
                    (word, atom, True, entry.line, entry.col)
                    for atom in category.atoms
                )
    if decl.lexicon_from_file is not None:
        learnable = get_option_flag(decl.lexicon_from_file_options, "learnable")
        lexical.extend(
            (word, category, learnable, decl.line, decl.col)
            for word, category in file_rows
        )
    for word, category, _learnable, line, col in lexical:
        if isinstance(category, str):
            continue
        try:
            render_pattern(category)
        except ValueError as error:
            refusals.append(
                ProductionRefusal(
                    message=(
                        f"deduction {decl.name!r}: lexicon entry {word!r}: {error}"
                    ),
                    line=line,
                    col=col,
                )
            )
            continue
        variables = _variables(category, atoms)
        if variables:
            refusals.append(
                ProductionRefusal(
                    message=(
                        f"deduction {decl.name!r}: lexicon entry {word!r} has "
                        f"category {render_pattern(category)}, which mentions "
                        f"{', '.join(repr(name) for name in variables)}, not a "
                        "declared atom, so its left-hand side is not ground"
                    ),
                    line=line,
                    col=col,
                )
            )
    if refusals:
        return ProductionAnalysis(refusals=tuple(refusals))

    keys: dict[tuple[str, str], list[Production]] = {}
    for index, rule in enumerate(decl.rules):
        key = _left_hand_side(rule.conclusion, slots)
        keys.setdefault(key, []).append(
            Production(
                kind="rule",
                index=index,
                label=rule.name,
                learnable=get_option_flag(rule.options, "learnable"),
            )
        )
    for index, (word, category, learnable, line, col) in enumerate(lexical):
        label = category if isinstance(category, str) else render_pattern(category)
        members = keys.setdefault(("span", label), [])
        shown = f'"{word}"'
        if any(member.label == shown for member in members):
            refusals.append(
                ProductionRefusal(
                    message=(
                        f"deduction {decl.name!r}: the lexicon gives {word!r} "
                        f"category {label} twice, so the two entries are one "
                        "production"
                    ),
                    line=line,
                    col=col,
                )
            )
        members.append(
            Production(kind="lexicon", index=index, label=shown, learnable=learnable)
        )
    labels: dict[str, tuple[str, str]] = {}
    for key in keys:
        other = labels.setdefault(key[1], key)
        if other != key:
            refusals.append(
                ProductionRefusal(
                    message=(
                        f"deduction {decl.name!r}: the left-hand side {key[1]} "
                        "names both a span category and an item, two different "
                        "left-hand sides; rename one"
                    ),
                    line=decl.line,
                    col=decl.col,
                )
            )
    if refusals:
        return ProductionAnalysis(refusals=tuple(refusals))
    return ProductionAnalysis(
        groups=tuple(
            ProductionGroup(label=key[1], productions=tuple(members))
            for key, members in keys.items()
        )
    )


__all__: list[str] = []
