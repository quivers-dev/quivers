"""Position sorts of a deduction's item slots, shared by both evaluators.

A deduction's items mix two kinds of argument: chart positions, which
say where in the input an item sits, and everything else (categories,
logical forms, arbitrary structured terms). The kinds are not marked
in the surface syntax; they are inferred from how the rules use each
constructor slot. A slot holds a position when

* it is one of the first two slots of ``span`` and the deduction has a
  lexicon, whose axioms are ``span(i, i + 1, category)`` items;
* an integer literal fills it in some rule; or
* a variable that fills a position slot somewhere in the same rule
  fills it.

The closure of these clauses is the least assignment the rules
support. The agenda compiler reads it to keep positions out of
learnable rule-weight keys, and the QIEC elaboration reads the same
assignment to type each slot ``Int`` or ``Item``, so the two
evaluators agree on which variables are positions.
"""

from __future__ import annotations

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    DeductionDecl,
    ObjectEffectApply,
    ObjectExpr,
    ObjectProduct,
    ObjectSlash,
    SequentRule,
    TypeName,
)

#: The constructor of the items a lexicon injects.
SPAN_SYMBOL = "span"

#: For each applied symbol, which of its slots hold positions.
type PositionSlots = dict[str, list[bool]]


def pattern_symbol(expr: ObjectExpr) -> str | None:
    """The constructor symbol an applied pattern names.

    Parameters
    ----------
    expr : ObjectExpr
        A rule pattern.

    Returns
    -------
    str | None
        The applied symbol (``fwd_slash`` and ``bwd_slash`` for the two
        slashes, ``product<n>`` for an ``n``-fold product), or ``None``
        for an identifier, which applies nothing.
    """
    if isinstance(expr, ObjectEffectApply):
        return expr.effect
    if isinstance(expr, ObjectSlash):
        return "fwd_slash" if expr.direction == "/" else "bwd_slash"
    if isinstance(expr, ObjectProduct):
        return f"product{len(expr.components)}"
    return None


def pattern_arguments(expr: ObjectExpr) -> tuple[ObjectExpr, ...]:
    """The argument patterns of an applied pattern, in slot order.

    Parameters
    ----------
    expr : ObjectExpr
        A rule pattern.

    Returns
    -------
    tuple[ObjectExpr, ...]
        The arguments; empty for an identifier.
    """
    if isinstance(expr, ObjectEffectApply):
        return tuple(expr.args)
    if isinstance(expr, ObjectSlash):
        return (expr.result, expr.argument)
    if isinstance(expr, ObjectProduct):
        return tuple(expr.components)
    return ()


def has_lexicon(decl: DeductionDecl) -> bool:
    """Whether a deduction takes its axioms from a lexicon.

    Parameters
    ----------
    decl : DeductionDecl
        The deduction.

    Returns
    -------
    bool
        ``True`` for an inline lexicon or a lexicon file.
    """
    return bool(decl.lexicon) or decl.lexicon_from_file is not None


def is_pattern_variable(expr: ObjectExpr, atoms: frozenset[str]) -> bool:
    """Whether a pattern is a variable: an identifier that is no atom or numeral.

    Parameters
    ----------
    expr : ObjectExpr
        A rule pattern.
    atoms : frozenset[str]
        The deduction's declared atoms.

    Returns
    -------
    bool
        ``True`` for a variable.
    """
    return (
        isinstance(expr, TypeName)
        and expr.name not in atoms
        and not expr.name.isdigit()
    )


def position_slots(decl: DeductionDecl) -> PositionSlots:
    """Infer which slot of each applied symbol holds a chart position.

    Parameters
    ----------
    decl : DeductionDecl
        The deduction.

    Returns
    -------
    PositionSlots
        For every symbol the rules apply, one flag per slot (up to the
        largest arity the symbol is applied at), ``True`` where the
        slot holds a position.
    """
    atoms = frozenset(decl.atoms)
    slots: PositionSlots = {}

    def ensure(symbol: str, arity: int) -> list[bool]:
        """The flags of a symbol, grown to an arity.

        Parameters
        ----------
        symbol : str
            The symbol.
        arity : int
            The arity it is applied at.

        Returns
        -------
        list[bool]
            Its flags.
        """
        flags = slots.setdefault(symbol, [])
        flags.extend([False] * (arity - len(flags)))
        return flags

    def register(expr: ObjectExpr) -> None:
        """Record every symbol a pattern applies, and its literal slots.

        Parameters
        ----------
        expr : ObjectExpr
            The pattern.
        """
        symbol = pattern_symbol(expr)
        if symbol is None:
            return
        arguments = pattern_arguments(expr)
        flags = ensure(symbol, len(arguments))
        for index, argument in enumerate(arguments):
            if isinstance(argument, TypeName) and argument.name.isdigit():
                flags[index] = True
            register(argument)

    for rule in decl.rules:
        for pattern in (*rule.premises, rule.conclusion):
            register(pattern)
    if has_lexicon(decl):
        flags = ensure(SPAN_SYMBOL, 2)
        flags[0] = flags[1] = True

    changed = True
    while changed:
        changed = False
        for rule in decl.rules:
            occurrences: dict[str, list[tuple[str, int]]] = {}
            positional: set[str] = set()

            def walk(expr: ObjectExpr) -> None:
                """Collect the slots each variable of a pattern fills.

                Parameters
                ----------
                expr : ObjectExpr
                    The pattern.
                """
                symbol = pattern_symbol(expr)
                if symbol is None:
                    return
                for index, argument in enumerate(pattern_arguments(expr)):
                    if is_pattern_variable(argument, atoms):
                        assert isinstance(argument, TypeName)
                        occurrences.setdefault(argument.name, []).append(
                            (symbol, index)
                        )
                        if slots[symbol][index]:
                            positional.add(argument.name)
                    else:
                        walk(argument)

            for pattern in (*rule.premises, rule.conclusion):
                walk(pattern)
            for name in positional:
                for symbol, index in occurrences[name]:
                    if not slots[symbol][index]:
                        slots[symbol][index] = True
                        changed = True
    return slots


class MisplacedPosition(dx.Model):
    """A position slot a rule fills with something other than a position.

    Parameters
    ----------
    rule : str
        The rule.
    symbol : str
        The applied symbol whose slot it is.
    slot : int
        The slot's index.
    text : str
        The offending argument as written: a declared atom's name, or
        the symbol of a structured term.
    is_atom : bool
        Whether the argument is a declared atom.
    line : int
        The rule's source line.
    col : int
        The rule's source column.
    """

    rule: str
    symbol: str
    slot: int
    text: str
    is_atom: bool
    line: int
    col: int

    def message(self, deduction: str) -> str:
        """The diagnostic both evaluators report.

        Parameters
        ----------
        deduction : str
            The deduction's name.

        Returns
        -------
        str
            The message, naming the rule, the identifier, and the slot.
        """
        where = f"slot {self.slot} of {self.symbol!r}"
        if self.is_atom:
            return (
                f"deduction {deduction!r}: rule {self.rule!r} uses {self.text!r} "
                f"at {where}, which holds a chart position, but {self.text!r} is "
                "a declared atom; a position slot takes a position variable or "
                "an integer, so rename the variable"
            )
        return (
            f"deduction {deduction!r}: rule {self.rule!r} puts a {self.text!r} "
            f"term at {where}, which holds a chart position; a position slot "
            "takes a position variable or an integer"
        )


def misplaced_positions(
    decl: DeductionDecl, slots: PositionSlots
) -> list[MisplacedPosition]:
    """The position slots the rules fill with an atom or a structured term.

    An identifier spelled like a declared atom is that atom wherever it
    occurs, in every evaluator; in a category slot that is its meaning,
    but a position slot cannot hold it, so the rule is rejected rather
    than read as a pattern that never matches.

    Parameters
    ----------
    decl : DeductionDecl
        The deduction.
    slots : PositionSlots
        Its position slots, from `position_slots`.

    Returns
    -------
    list[MisplacedPosition]
        Every offending slot, in rule and source order.
    """
    atoms = frozenset(decl.atoms)
    found: list[MisplacedPosition] = []
    for rule in decl.rules:

        def walk(expr: ObjectExpr, rule: SequentRule = rule) -> None:
            """Collect a pattern's misplaced positions.

            Parameters
            ----------
            expr : ObjectExpr
                The pattern.
            rule : SequentRule
                The rule it belongs to.
            """
            symbol = pattern_symbol(expr)
            if symbol is None:
                return
            for index, argument in enumerate(pattern_arguments(expr)):
                inner = pattern_symbol(argument)
                if slots[symbol][index]:
                    if isinstance(argument, TypeName) and argument.name in atoms:
                        text, is_atom = argument.name, True
                    elif inner is not None:
                        text, is_atom = inner, False
                    else:
                        continue
                    found.append(
                        MisplacedPosition(
                            rule=rule.name,
                            symbol=symbol,
                            slot=index,
                            text=text,
                            is_atom=is_atom,
                            line=rule.line,
                            col=rule.col,
                        )
                    )
                walk(argument)

        for pattern in (*rule.premises, rule.conclusion):
            walk(pattern)
    return found


def rule_position_variables(
    decl: DeductionDecl, slots: PositionSlots, rule: SequentRule
) -> frozenset[str]:
    """The variables a rule uses as chart positions.

    Parameters
    ----------
    decl : DeductionDecl
        The deduction declaring the rule.
    slots : PositionSlots
        The deduction's position slots, from `position_slots`.
    rule : SequentRule
        The rule.

    Returns
    -------
    frozenset[str]
        The names of the variables that fill a position slot.
    """
    atoms = frozenset(decl.atoms)
    found: set[str] = set()

    def walk(expr: ObjectExpr) -> None:
        """Collect a pattern's position variables.

        Parameters
        ----------
        expr : ObjectExpr
            The pattern.
        """
        symbol = pattern_symbol(expr)
        if symbol is None:
            return
        for index, argument in enumerate(pattern_arguments(expr)):
            if is_pattern_variable(argument, atoms):
                assert isinstance(argument, TypeName)
                if slots[symbol][index]:
                    found.add(argument.name)
            else:
                walk(argument)

    for pattern in (*rule.premises, rule.conclusion):
        walk(pattern)
    return frozenset(found)


__all__ = [
    "SPAN_SYMBOL",
    "MisplacedPosition",
    "PositionSlots",
    "has_lexicon",
    "is_pattern_variable",
    "misplaced_positions",
    "pattern_arguments",
    "pattern_symbol",
    "position_slots",
    "rule_position_variables",
]
