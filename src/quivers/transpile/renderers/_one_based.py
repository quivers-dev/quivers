"""The boundary between QVR's zero-based indices and a one-based host.

A QVR index value counts from zero: a factor binder over an object of
size ``n`` takes the values ``0, ..., n - 1``, a marginalized latent
takes the same values, and a subscript ``y[2 - t]`` reads the entry at
that zero-based position. Stan, BUGS, JAGS, and Julia count from one.
The renderers for those hosts keep every value in QVR's origin and
translate at exactly one place, the subscript: the index expression of
``a[i]`` becomes ``i + 1``. A host loop counter, which runs over
``1, ..., n``, enters an expression as its QVR value ``counter - 1``,
so it reads correctly where it is used as a number, and the subscript
translation folds ``(counter - 1) + 1`` back to ``counter`` where it is
used as an index.

Every value the program holds is in QVR's origin, data included: an
index covariate (``group_idx``) and a class code (a Categorical
outcome, or a drawn class) count from zero, and a subscript that reads
one reads it one higher. A one-based target's categorical family, which
counts classes from one, reads a class code one higher too
([`is_class_code_family`][quivers.transpile.renderers._one_based.is_class_code_family]
names the families whose values are codes), so a caller passes every
target the same zero-based data.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprLiteral,
    LetExprNode,
    LetExprVar,
)
from quivers.transpile.family_meta import FAMILY_META, class_index_outcome


def is_class_code_family(family: str) -> bool:
    """Whether a family's values are class codes of an alphabet.

    QVR codes a Categorical, OrderedLogistic, or OrderedProbit value
    from zero, and a one-based target's categorical families count
    classes from one, so such a value crosses the same boundary a
    subscript does. Bernoulli is not one: its value is a bit.

    Parameters
    ----------
    family : str
        The QVR family name.

    Returns
    -------
    bool
        True for a class-index family.
    """
    meta = FAMILY_META.get(family)
    return meta is not None and class_index_outcome(meta) is not None


def counter_value(name: str) -> LetExprNode:
    """The QVR value of a one-based host loop counter.

    Parameters
    ----------
    name : str
        The counter's name in the host program.

    Returns
    -------
    LetExprNode
        ``name - 1``, the zero-based position the counter stands for.
    """
    return LetExprBinOp(
        op="-",
        left=LetExprVar(name=name),
        right=LetExprLiteral(value=1.0, integral=True),
    )


def _integer(expr: LetExprNode) -> int | None:
    """The value of an integer literal, or `None` for anything else.

    Parameters
    ----------
    expr : LetExprNode
        The expression.

    Returns
    -------
    int | None
        The literal's value when it is a whole number.
    """
    if isinstance(expr, LetExprLiteral) and float(expr.value).is_integer():
        return int(expr.value)
    return None


def _literal(value: int) -> LetExprLiteral:
    """An integer literal.

    Parameters
    ----------
    value : int
        The value.

    Returns
    -------
    LetExprLiteral
        The literal, marked integral.
    """
    return LetExprLiteral(value=float(value), integral=True)


def fold_integers(expr: LetExprNode) -> LetExprNode:
    """Evaluate integer-literal arithmetic in an index expression.

    Parameters
    ----------
    expr : LetExprNode
        An index expression.

    Returns
    -------
    LetExprNode
        The expression with every ``+``, ``-``, or ``*`` of two integer
        literals replaced by its value.
    """
    if not isinstance(expr, LetExprBinOp):
        return expr
    left = fold_integers(expr.left)
    right = fold_integers(expr.right)
    a, b = _integer(left), _integer(right)
    if a is not None and b is not None and expr.op in ("+", "-", "*"):
        return _literal(a + b if expr.op == "+" else a - b if expr.op == "-" else a * b)
    return LetExprBinOp(op=expr.op, left=left, right=right)


def one_based_subscript(index: LetExprNode) -> LetExprNode:
    """Translate a zero-based QVR index into the host's one-based one.

    Parameters
    ----------
    index : LetExprNode
        The index expression, valued in QVR's origin.

    Returns
    -------
    LetExprNode
        An expression for ``index + 1``, with the shift folded into an
        integer literal or a trailing ``- 1`` where the expression has
        one, so a counter's value ``k - 1`` subscripts as ``k`` and a
        literal ``0`` as ``1``.
    """
    folded = fold_integers(index)
    value = _integer(folded)
    if value is not None:
        return _literal(value + 1)
    if isinstance(folded, LetExprBinOp):
        right = _integer(folded.right)
        left = _integer(folded.left)
        if folded.op == "-" and right is not None and right >= 1:
            return (
                folded.left
                if right == 1
                else LetExprBinOp(op="-", left=folded.left, right=_literal(right - 1))
            )
        if folded.op == "+" and right is not None and right >= 0:
            return LetExprBinOp(op="+", left=folded.left, right=_literal(right + 1))
        if folded.op == "+" and left is not None and left >= 0:
            return LetExprBinOp(op="+", left=_literal(left + 1), right=folded.right)
        if folded.op == "-" and left is not None:
            return LetExprBinOp(op="-", left=_literal(left + 1), right=folded.right)
    return LetExprBinOp(op="+", left=folded, right=_literal(1))


__all__ = [
    "counter_value",
    "fold_integers",
    "is_class_code_family",
    "one_based_subscript",
]
