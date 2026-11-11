"""Reading adapter over the DSL `DrawArg` AST for the transpile.

The DSL models a draw-step argument with a recursive `DrawArg`
tagged union: an atom is a `DrawArgScalar` (number), a `DrawArgName`
(identifier), or a `DrawArgIndex` (bracket-indexed reference); a
vector literal ``[a, b]`` is a `DrawArgList` whose ``items`` are
atoms; a matrix literal ``[[a, b], [c, d]]`` is a `DrawArgList`
whose ``items`` are themselves `DrawArgList`s.

The transpile pipeline reads those arguments as flat forms: atoms
(numbers, identifiers, bracket-encoded references), vectors of
atoms, and matrices of atoms. This module is the single boundary
that translates the recursive AST shape into those flat forms, so
the lowering pass, the resolver, and the backend renderers never
match `DrawArg` variants or reach into ``items`` themselves. They
call `atom_value`, `list_items`, `list_atoms`, `is_matrix`, and
`matrix_rows` instead.

A bracket-indexed reference re-serialises to the repeated-bracket
surface form ``theta[i]`` / ``w[i][j]`` that the backends' own
index regex consumes, via `encode_index`.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
)
from quivers.dsl.family_schemas import (
    family_parameterizations,
    render_parameterizations,
)

Atom = str | float


def argument_parameter(arg: DrawArg) -> str | None:
    """Return the keyword carried by a structural named argument."""
    return arg.parameter if isinstance(arg, DrawArgNamed) else None


def argument_value(arg: DrawArg) -> DrawArg:
    """Return a named argument's value, or ``arg`` when it is positional."""
    return arg.value if isinstance(arg, DrawArgNamed) else arg


def bind_family_arguments(
    family: str,
    args: tuple[DrawArg, ...],
    canonical: tuple[str, ...] = (),
) -> tuple[tuple[str, DrawArg], ...]:
    """Validate and order one keyword-only family construction.

    The AST retains the names authors wrote. This boundary matches those
    names against one complete schema and returns values in that schema's
    canonical order, so positional target APIs never depend on source order.
    """
    if not args:
        return ()
    schemas = family_parameterizations(family, canonical)
    rendered = render_parameterizations(schemas)
    if not schemas:
        raise TypeError(f"family {family!r} does not take named arguments")
    bound: dict[str, DrawArg] = {}
    for argument in args:
        if not isinstance(argument, DrawArgNamed):
            raise TypeError(
                f"family {family!r} arguments are keyword-only; "
                f"use one complete schema: {rendered}"
            )
        if argument.parameter in bound:
            raise TypeError(
                f"family {family!r} is given parameter {argument.parameter!r} twice"
            )
        bound[argument.parameter] = argument.value
    selected = next((schema for schema in schemas if set(schema) == set(bound)), None)
    if selected is None:
        raise TypeError(
            f"family {family!r} requires exactly one complete parameter schema "
            f"{rendered}; got ({', '.join(bound)})"
        )
    return tuple((name, bound[name]) for name in selected)


def encode_index(arg: DrawArgIndex) -> str:
    """Serialise a bracket-indexed reference to its repeated-bracket
    surface form: ``theta[i]`` for a single index, ``w[i][j]`` for
    several. This is the wire form the backends' index parser reads.
    """
    return arg.name + "".join(f"[{index}]" for index in arg.indices)


def is_atom(arg: DrawArg) -> bool:
    """True when `arg` is an atomic variant: a scalar, a bare name,
    or a bracket-indexed reference."""
    return isinstance(arg, (DrawArgScalar, DrawArgName, DrawArgIndex))


def atom_value(arg: DrawArg) -> Atom:
    """Return the flat wire value of an atomic `DrawArg`.

    A `DrawArgScalar` yields its float, a `DrawArgName` its
    identifier text, and a `DrawArgIndex` its bracket-encoded
    surface string. Raises for compound variants (`DrawArgList`,
    `DrawArgDist`), which carry no single atomic value.
    """
    if isinstance(arg, DrawArgScalar):
        return arg.value
    if isinstance(arg, DrawArgName):
        return arg.text
    if isinstance(arg, DrawArgIndex):
        return encode_index(arg)
    raise TypeError(f"atom_value: not an atomic DrawArg: {type(arg).__name__}")


def list_items(arg: DrawArgList) -> tuple[DrawArg, ...]:
    """Return the recursive child `DrawArg`s of a list literal.

    Use this when the children may themselves be compound (a matrix
    row, a nested distribution); use `list_atoms` when every child
    is known to be atomic.
    """
    return arg.items


def list_atoms(arg: DrawArgList) -> tuple[Atom, ...]:
    """Return the atomic wire values of a vector-literal
    `DrawArgList`. Raises when any child is compound (guard matrix
    inputs with `is_matrix`)."""
    return tuple(atom_value(item) for item in arg.items)


def is_matrix(arg: DrawArg) -> bool:
    """True when `arg` is a matrix literal: a non-empty `DrawArgList`
    whose every item is itself a `DrawArgList`."""
    return (
        isinstance(arg, DrawArgList)
        and len(arg.items) > 0
        and all(isinstance(item, DrawArgList) for item in arg.items)
    )


def matrix_rows(arg: DrawArgList) -> tuple[tuple[Atom, ...], ...]:
    """Return the row-major atom values of a matrix-literal
    `DrawArgList` (each item is a `DrawArgList` of atoms).

    Callers should gate this with `is_matrix`.
    """
    rows: list[tuple[Atom, ...]] = []
    for item in arg.items:
        if not isinstance(item, DrawArgList):
            raise TypeError(
                f"matrix_rows: row is not a DrawArgList: {type(item).__name__}"
            )
        rows.append(list_atoms(item))
    return tuple(rows)


def is_literal(arg: DrawArg) -> bool:
    """Whether an argument is a number or a nested list of numbers.

    Parameters
    ----------
    arg : DrawArg
        The argument, named or positional.

    Returns
    -------
    bool
        True for a numeric literal and for a vector or matrix literal
        whose every entry is one; False for a name, an indexed
        reference, or a nested family.
    """
    value = argument_value(arg)
    if isinstance(value, DrawArgScalar):
        return True
    if isinstance(value, DrawArgList):
        return all(is_literal(item) for item in value.items)
    return False


def argument_text(arg: DrawArg) -> str:
    """Render a draw argument as the source spells it.

    Parameters
    ----------
    arg : DrawArg
        The argument.

    Returns
    -------
    str
        The surface text: ``scale=q``, ``theta[i]``, ``[0.0, 1.0]``,
        ``Normal(loc=0.0, scale=1.0)``.

    Raises
    ------
    TypeError
        If ``arg`` is not a `DrawArg` variant this module knows.
    """
    if isinstance(arg, DrawArgNamed):
        return f"{arg.parameter}={argument_text(arg.value)}"
    if isinstance(arg, DrawArgScalar):
        return repr(arg.value)
    if isinstance(arg, DrawArgName):
        return arg.text
    if isinstance(arg, DrawArgIndex):
        return f"{arg.name}[{', '.join(arg.indices)}]"
    if isinstance(arg, DrawArgList):
        return "[" + ", ".join(argument_text(item) for item in arg.items) + "]"
    if isinstance(arg, DrawArgDist):
        inner = ", ".join(argument_text(item) for item in arg.args)
        return f"{arg.family}({inner})"
    raise TypeError(f"argument_text: unknown DrawArg {type(arg).__name__}")


def kernel_argument_refusal(
    morphism: str, family: str, args: tuple[DrawArg, ...]
) -> tuple[DrawArg, str] | None:
    """Explain why a kernel declaration's family arguments are refused.

    A kernel declared ``~ Family(...)`` either fixes its family with
    literal arguments, one complete schema of numbers, or leaves the
    argument list empty and takes every parameter from its learned
    parameter source. An argument that names a value (``scale=q``)
    has nothing to read in a declaration, which binds no program
    value, so every execution path refuses it with this message.

    Parameters
    ----------
    morphism : str
        The declared morphism's name.
    family : str
        The family its initializer names.
    args : tuple[DrawArg, ...]
        The initializer's arguments.

    Returns
    -------
    tuple[DrawArg, str] | None
        The first argument that is not a literal and the refusal
        naming it, or ``None`` when every argument is a literal.
    """
    offending = next((arg for arg in args if not is_literal(arg)), None)
    if offending is None:
        return None
    text = argument_text(offending)
    return offending, (
        f"kernel morphism {morphism!r}: family argument `{text}` of "
        f"`~ {family}(...)` is not a literal, and a morphism declaration "
        f"binds no value for it to read. A kernel's family arguments are "
        f"literals that fix every parameter, or absent, when every parameter "
        f"comes from the kernel's learned parameter source. To draw with "
        f"`{text}` as a parameter, draw from the family in a program step, "
        f"where the value is bound (`sample s <- {family}(..., {text})`)."
    )


__all__ = []
