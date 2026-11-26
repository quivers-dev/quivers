"""Named terms with capture-avoiding substitution, compared modulo alpha.

A term is plain data: ``("var", name)`` for a variable, or a tuple of an
operation name and its arguments. A binder is described by the number of
variables it binds and the positions of its arguments that are in their
scope; a bound occurrence in a binder node is written ``(op, names,
*arguments)`` with the tuple of bound names first. Nothing here reads de
Bruijn indices except `from_indices` and `to_indices`, which translate
the library's terms in and out, so substitution itself is the textbook
named algorithm: rename a bound variable when it would capture a free
variable of what is substituted.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

type Leaf = str | int | float | bool | bytes
"""A data value inside a term."""

type Named = tuple[Leaf | Named | tuple[str, ...], ...]
"""A named term as nested tuples."""

type Indexed = tuple[Leaf | Indexed, ...]
"""A de Bruijn term as nested tuples: ``("BoundVar", i)`` for a variable."""

type Binders = Mapping[str, tuple[int, frozenset[int]]]
"""For each binder: how many variables it binds, and its scoped positions."""


_FRESH = itertools.count()


def fresh(avoid: set[str]) -> str:
    """A variable name not in a set.

    Parameters
    ----------
    avoid : set of str
        Names in use.

    Returns
    -------
    str
        A new name.
    """
    while True:
        name = f"v{next(_FRESH)}"
        if name not in avoid:
            return name


def free(term: Named, binders: Binders) -> set[str]:
    """The free variables of a named term.

    Parameters
    ----------
    term : Named
        The term.
    binders : Binders
        The binders.

    Returns
    -------
    set of str
        Its free variables.
    """
    if term[0] == "var":
        return {str(term[1])}
    op = str(term[0])
    if op in binders:
        _, scoped = binders[op]
        names = term[1]
        assert isinstance(names, tuple)
        found: set[str] = set()
        for position, argument in enumerate(term[2:]):
            assert isinstance(argument, tuple)
            inner = free(argument, binders)
            found |= inner - set(names) if position in scoped else inner
        return found
    found = set()
    for argument in term[1:]:
        if isinstance(argument, tuple):
            found |= free(argument, binders)
    return found


def substitute(term: Named, mapping: Mapping[str, Named], binders: Binders) -> Named:
    """Replace free variables, renaming bound ones that would capture.

    Parameters
    ----------
    term : Named
        The term.
    mapping : Mapping[str, Named]
        The replacement of each free variable it names.
    binders : Binders
        The binders.

    Returns
    -------
    Named
        The substituted term.
    """
    if term[0] == "var":
        return mapping.get(str(term[1]), term)
    op = str(term[0])
    if op in binders:
        _, scoped = binders[op]
        names = term[1]
        assert isinstance(names, tuple)
        incoming: set[str] = set()
        for replacement in mapping.values():
            incoming |= free(replacement, binders)
        renamed: list[str] = []
        inner_mapping = {
            key: value for key, value in mapping.items() if key not in names
        }
        for name in names:
            assert isinstance(name, str)
            if name in incoming:
                avoid = incoming | {str(item) for item in names} | free(term, binders)
                new = fresh(avoid)
                inner_mapping[name] = ("var", new)
                renamed.append(new)
            else:
                renamed.append(name)
        arguments = []
        for position, argument in enumerate(term[2:]):
            assert isinstance(argument, tuple)
            if position in scoped:
                arguments.append(substitute(argument, inner_mapping, binders))
            else:
                arguments.append(substitute(argument, mapping, binders))
        return (op, tuple(renamed), *arguments)
    return (
        op,
        *(
            substitute(argument, mapping, binders)
            if isinstance(argument, tuple)
            else argument
            for argument in term[1:]
        ),
    )


def from_indices(term: Indexed, context: tuple[str, ...], binders: Binders) -> Named:
    """Read a de Bruijn term as a named term.

    Parameters
    ----------
    term : Indexed
        The de Bruijn term; ``("BoundVar", i)`` names the ``i``-th
        innermost variable of the context.
    context : tuple of str
        The names of the free variables, innermost last.
    binders : Binders
        The binders.

    Returns
    -------
    Named
        The named term.
    """
    op = str(term[0])
    if op == "BoundVar":
        index = term[1]
        assert isinstance(index, int)
        return ("var", context[len(context) - 1 - index])
    if op in binders:
        count, scoped = binders[op]
        names = tuple(fresh(set(context)) for _ in range(count))
        inner = context + names
        arguments = []
        for position, argument in enumerate(term[1:]):
            assert isinstance(argument, tuple)
            arguments.append(
                from_indices(
                    argument, inner if position in scoped else context, binders
                )
            )
        return (op, names, *arguments)
    return (
        op,
        *(
            from_indices(argument, context, binders)
            if isinstance(argument, tuple)
            else argument
            for argument in term[1:]
        ),
    )


def to_indices(term: Named, context: tuple[str, ...], binders: Binders) -> Indexed:
    """Write a named term with de Bruijn indices, which identifies alpha-variants.

    Parameters
    ----------
    term : Named
        The term.
    context : tuple of str
        The names of the free variables, innermost last.
    binders : Binders
        The binders.

    Returns
    -------
    Indexed
        The de Bruijn term.

    Raises
    ------
    KeyError
        If a free variable is not in the context.
    """
    if term[0] == "var":
        name = str(term[1])
        for depth, known in enumerate(reversed(context)):
            if known == name:
                return ("BoundVar", depth)
        raise KeyError(name)
    op = str(term[0])
    if op in binders:
        _, scoped = binders[op]
        names = term[1]
        assert isinstance(names, tuple)
        inner = context + tuple(str(name) for name in names)
        arguments = []
        for position, argument in enumerate(term[2:]):
            assert isinstance(argument, tuple)
            arguments.append(
                to_indices(argument, inner if position in scoped else context, binders)
            )
        return (op, *arguments)
    return (
        op,
        *(
            to_indices(argument, context, binders)
            if isinstance(argument, tuple)
            else argument
            for argument in term[1:]
        ),
    )


def substitute_indexed(
    term: Indexed,
    replacements: tuple[Indexed, ...],
    shift: int,
    depth: int,
    binders: Binders,
) -> Indexed:
    """Substitute in a de Bruijn term by way of named terms.

    The variable with index ``i`` becomes ``replacements[i]`` for ``i``
    below their number, and the variable ``i - n + shift`` otherwise.
    Every term lives in a context of ``depth`` variables; replacements and
    the result live in a context of ``depth - n + shift`` variables.

    Parameters
    ----------
    term : Indexed
        The term.
    replacements : tuple of Indexed
        The replacements of the innermost variables.
    shift : int
        Where the other variables go.
    depth : int
        The number of free variables of the term's context.
    binders : Binders
        The binders.

    Returns
    -------
    Indexed
        The result, by named capture-avoiding substitution.
    """
    count = len(replacements)
    target = depth - count + shift
    source_names = tuple(fresh(set()) for _ in range(depth))
    target_names = tuple(fresh(set(source_names)) for _ in range(target))
    mapping: dict[str, Named] = {}
    for index in range(depth):
        name = source_names[depth - 1 - index]
        if index < count:
            mapping[name] = from_indices(replacements[index], target_names, binders)
        else:
            mapping[name] = ("var", target_names[target - 1 - (index - count + shift)])
    named = from_indices(term, source_names, binders)
    return to_indices(substitute(named, mapping, binders), target_names, binders)
