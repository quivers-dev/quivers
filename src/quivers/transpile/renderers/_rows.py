"""One row of a let expression's value.

A renderer that emits a plated binding as a loop computes the binding
one row at a time, and a scope that runs per row (a marginalize body
under its observation plate) reads a plated binding at its own row.
Both need the expression for the value's entry at one position of its
leading axis. QVR broadcasts by aligning trailing axes, so an operand
carries the leading axis exactly when it has the expression's full
rank; an operand of lower rank is the same at every row and stays as
it is. [`take_row`][quivers.transpile.renderers._rows.take_row] pushes
the row selection down to the operands that carry the axis, so the
emitted expression reads one entry of each plated operand, whatever
the expression's shape.
"""

from __future__ import annotations

from collections.abc import Mapping

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprList,
    LetExprLiteral,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.renderers._ranks import LAST_AXIS_REDUCTIONS, let_expr_rank
from quivers.transpile.renderers._stan_helpers import _substitute_let_expr

#: The let builtins that act along the last axis of their argument and
#: keep its rank.
_LAST_AXIS_TRANSFORMS: frozenset[str] = frozenset(
    {"softmax", "log_softmax", "cumsum", "sort", "normalize"}
)


def take_row(
    expr: LetExprNode,
    row: LetExprNode,
    ranks: Mapping[str, int],
    target: str,
) -> LetExprNode:
    """The entry of a value at one position of its leading axis.

    Parameters
    ----------
    expr : LetExprNode
        The expression, of rank one or more.
    row : LetExprNode
        The position, valued in QVR's zero-based origin.
    ranks : Mapping[str, int]
        The rank of each name the expression may read.
    target : str
        The target being rendered, for diagnostics.

    Returns
    -------
    LetExprNode
        An expression for ``expr[row]`` in which the selection is
        applied to each operand that carries the leading axis.

    Raises
    ------
    UnsupportedConstruct
        If the expression's rank is zero, or the leading axis is the
        axis a rowwise transform acts along, so no operand selection
        expresses the row.
    """
    rank = let_expr_rank(expr, ranks)
    if rank == 0:
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [f"let-expr:row:{type(expr).__name__}: a number has no rows to select"],
        )
    return _take(expr, row, rank, ranks, target)


def _take(
    expr: LetExprNode,
    row: LetExprNode,
    rank: int,
    ranks: Mapping[str, int],
    target: str,
) -> LetExprNode:
    """Select one row of ``expr``, whose rank is ``rank``.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    row : LetExprNode
        The position.
    rank : int
        The rank of the value the row is selected from; an operand of
        lower rank does not carry the axis.
    ranks : Mapping[str, int]
        The rank of each name.
    target : str
        The target, for diagnostics.

    Returns
    -------
    LetExprNode
        The selected row.

    Raises
    ------
    UnsupportedConstruct
        If the row cannot be pushed through the expression.
    """
    if let_expr_rank(expr, ranks) < rank:
        return expr
    if isinstance(expr, LetExprVar):
        return LetExprIndex(array=expr, indices=(row,))
    if isinstance(expr, LetExprIndex):
        index_ranks = [let_expr_rank(index, ranks) for index in expr.indices]
        if not any(index_ranks):
            return LetExprIndex(array=expr.array, indices=(*expr.indices, row))
        gathers = [position for position, r in enumerate(index_ranks) if r > 0]
        if gathers[0] == 0 and len(gathers) == 1:
            # A gather by an index array: the leading axis is the
            # index array's own, so the row selects one index.
            first = _take(expr.indices[0], row, index_ranks[0], ranks, target)
            return LetExprIndex(array=expr.array, indices=(first, *expr.indices[1:]))
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [
                "let-expr:row:LetExprIndex: a subscript gathering along "
                "several axes has no single row to select"
            ],
        )
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_take(expr.left, row, rank, ranks, target),
            right=_take(expr.right, row, rank, ranks, target),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            op=expr.op, operand=_take(expr.operand, row, rank, ranks, target)
        )
    if isinstance(expr, LetExprCall):
        if expr.func in LAST_AXIS_REDUCTIONS and len(expr.args) == 1:
            return LetExprCall(
                func=expr.func,
                args=(_take(expr.args[0], row, rank + 1, ranks, target),),
            )
        if expr.func in _LAST_AXIS_TRANSFORMS and rank == 1:
            raise UnsupportedConstruct(
                f"qvr-{target}",
                [
                    f"let-expr:row:{expr.func}: `{expr.func}` acts along the "
                    f"axis whose row is selected, so its entries are not "
                    f"computed row by row"
                ],
            )
        return LetExprCall(
            func=expr.func,
            args=tuple(_take(arg, row, rank, ranks, target) for arg in expr.args),
        )
    if isinstance(expr, LetExprFactor) and expr.body is not None and not expr.cases:
        first, *rest = expr.binders
        body = _substitute_let_expr(
            expr.body, first.var, index_value=row, scalar_value=row
        )
        if rest:
            return LetExprFactor(binders=tuple(rest), body=body)
        return body
    if (
        isinstance(expr, LetExprFactor)
        and expr.cases
        and isinstance(row, LetExprLiteral)
    ):
        for case in expr.cases:
            if case.label == int(row.value):
                return case.value
    if isinstance(expr, LetExprList) and isinstance(row, LetExprLiteral):
        return expr.items[int(row.value)]
    return LetExprIndex(array=expr, indices=(row,))


__all__ = ["take_row"]
