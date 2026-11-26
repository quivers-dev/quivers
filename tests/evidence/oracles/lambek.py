"""Provability in the Lambek calculus by cut-free sequent search.

A type is plain data: ``("np",)`` or another atom, ``("/", a, b)`` for
``a / b`` (wants a ``b`` on its right), and ``("\\\\", b, a)`` for
``b \\\\ a`` (wants a ``b`` on its left). A sequent has a nonempty,
ordered antecedent. The rules are those of Lambek (1958) without cut,
which is admissible, so the search decides provability: every rule
removes one connective, so it terminates. With ``exchange`` the
antecedent may be permuted first, which is the Lambek calculus with
permutation (LP).
"""

from __future__ import annotations

import itertools
from functools import cache

type Type = tuple[str] | tuple[str, Type, Type]
"""A Lambek type."""


def _parts(type_: Type) -> tuple[Type, Type]:
    """The two arguments of a slash type, in their written order.

    Parameters
    ----------
    type_ : Type
        A slash type.

    Returns
    -------
    tuple of Type and Type
        ``(a, b)`` for ``a / b`` and ``(b, a)`` for ``b \\ a``.

    Raises
    ------
    ValueError
        If the type is an atom.
    """
    if len(type_) != 3:
        raise ValueError(f"{type_!r} is an atom")
    return (type_[1], type_[2])


@cache
def provable(antecedent: tuple[Type, ...], succedent: Type) -> bool:
    """Whether a sequent is provable in the Lambek calculus.

    Parameters
    ----------
    antecedent : tuple of Type
        The hypotheses, in order; nonempty.
    succedent : Type
        The conclusion.

    Returns
    -------
    bool
        Whether a cut-free proof exists.
    """
    if not antecedent:
        return False
    if len(antecedent) == 1 and antecedent[0] == succedent:
        return True
    if succedent[0] == "/":
        a, b = _parts(succedent)
        if provable((*antecedent, b), a):
            return True
    if succedent[0] == "\\":
        b, a = _parts(succedent)
        if provable((b, *antecedent), a):
            return True
    for index, hypothesis in enumerate(antecedent):
        if hypothesis[0] == "/":
            a, b = _parts(hypothesis)
            left = antecedent[:index]
            for end in range(index + 2, len(antecedent) + 1):
                middle = antecedent[index + 1 : end]
                right = antecedent[end:]
                if provable(middle, b) and provable((*left, a, *right), succedent):
                    return True
        if hypothesis[0] == "\\":
            b, a = _parts(hypothesis)
            right = antecedent[index + 1 :]
            for start in range(index):
                middle = antecedent[start:index]
                left = antecedent[:start]
                if provable(middle, b) and provable((*left, a, *right), succedent):
                    return True
    return False


@cache
def provable_with_exchange(antecedent: tuple[Type, ...], succedent: Type) -> bool:
    """Whether a sequent is provable in the Lambek calculus with permutation.

    The antecedent is a multiset at every step of the search, so the two
    slashes coincide: this is LP, cut free.

    Parameters
    ----------
    antecedent : tuple of Type
        The hypotheses, in any order; nonempty.
    succedent : Type
        The conclusion.

    Returns
    -------
    bool
        Whether a cut-free LP proof exists.
    """
    hypotheses = tuple(sorted(antecedent, key=repr))
    if not hypotheses:
        return False
    if len(hypotheses) == 1 and hypotheses[0] == succedent:
        return True
    if succedent[0] == "/":
        a, b = _parts(succedent)
        if provable_with_exchange((*hypotheses, b), a):
            return True
    if succedent[0] == "\\":
        b, a = _parts(succedent)
        if provable_with_exchange((*hypotheses, b), a):
            return True
    for index, hypothesis in enumerate(hypotheses):
        if hypothesis[0] not in {"/", "\\"}:
            continue
        if hypothesis[0] == "/":
            a, b = _parts(hypothesis)
        else:
            b, a = _parts(hypothesis)
        rest = hypotheses[:index] + hypotheses[index + 1 :]
        for size in range(1, len(rest) + 1):
            for chosen in set(itertools.combinations(range(len(rest)), size)):
                middle = tuple(rest[k] for k in chosen)
                others = tuple(rest[k] for k in range(len(rest)) if k not in chosen)
                if provable_with_exchange(middle, b) and provable_with_exchange(
                    (*others, a), succedent
                ):
                    return True
    return False
