"""Shared encoding constants for classic deduction terms.

The agenda engine represents both categories and logical forms as nested
tuples. Nullary constants and bound variables therefore need explicit leaf
tags: a bare one-tuple cannot distinguish a constant from a variable.
"""

from __future__ import annotations


TERM_ATOM_TAG = "atom"
TERM_VARIABLE_TAG = "var"
TERM_LEAF_TAGS = frozenset({TERM_ATOM_TAG, TERM_VARIABLE_TAG})
RESERVED_TERM_TAGS = TERM_LEAF_TAGS

LET_CONSTRUCTORS_KEY = "__constructors__"
LET_BOUND_VARIABLES_KEY = "__bound_vars__"
LET_FAMILIES_KEY = "__family_catalog__"
LET_INTERNAL_KEYS = frozenset(
    {LET_CONSTRUCTORS_KEY, LET_BOUND_VARIABLES_KEY, LET_FAMILIES_KEY}
)


def atom_term(name: str) -> tuple[str, str]:
    """Encode a nullary constant in the public chart representation."""
    return (TERM_ATOM_TAG, name)


def variable_term(name: str) -> tuple[str, str]:
    """Encode a bound variable in the public chart representation."""
    return (TERM_VARIABLE_TAG, name)


def variable_name(value: object) -> str | None:
    """Return a tagged variable's name, or ``None`` for another term."""
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and value[0] == TERM_VARIABLE_TAG
        and isinstance(value[1], str)
    ):
        return value[1]
    return None


__all__ = [
    "LET_BOUND_VARIABLES_KEY",
    "LET_CONSTRUCTORS_KEY",
    "LET_INTERNAL_KEYS",
    "LET_FAMILIES_KEY",
    "RESERVED_TERM_TAGS",
    "TERM_ATOM_TAG",
    "TERM_LEAF_TAGS",
    "TERM_VARIABLE_TAG",
    "atom_term",
    "variable_name",
    "variable_term",
]
