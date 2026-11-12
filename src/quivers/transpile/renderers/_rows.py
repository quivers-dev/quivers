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
from quivers.dsl.pure_builtins import BUILTIN_REGISTRY
from quivers.qiec.families import FAMILIES
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.ir import Plate
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


def pointwise_let(expr: LetExprNode) -> bool:
    """Whether a let expression computes each entry from the same entries.

    An expression of names, literals, arithmetic, and the scalar builtins
    the checked lowering lifts entry by entry (``exp``, ``tanh``, a
    two-argument ``max``) has, at each position, the value of the same
    expression over its operands' entries at that position. A reduction,
    a last-axis transform, an index, or any other form reads more than
    one entry.

    Parameters
    ----------
    expr : LetExprNode
        The expression.

    Returns
    -------
    bool
        True when the expression is pointwise.
    """
    if isinstance(expr, LetExprVar | LetExprLiteral):
        return True
    if isinstance(expr, LetExprBinOp):
        return pointwise_let(expr.left) and pointwise_let(expr.right)
    if isinstance(expr, LetExprUnaryOp):
        return pointwise_let(expr.operand)
    if isinstance(expr, LetExprCall):
        spec = BUILTIN_REGISTRY.get(expr.func)
        if (
            spec is None
            or spec.shape_rule != "scalar-lift"
            or expr.func in LAST_AXIS_REDUCTIONS
            and len(expr.args) == 1
            or not all(
                len(signature) == len(expr.args) for signature in spec.qiec_overloads
            )
        ):
            return False
        return all(pointwise_let(argument) for argument in expr.args)
    return False


def scalar_parameter(family: str, parameter: str) -> bool:
    """Whether a family parameter takes one number per drawn element.

    A rank-0 parameter (`Normal`'s `loc`, `Poisson`'s `rate`) is read
    once per element of the draw, so a value filling it that varies
    along the axes the draw is replicated over is read at each
    element's position. A parameter of higher rank (`Categorical`'s
    `probs`) reads a whole event and keeps its own axes.

    Parameters
    ----------
    family : str
        The family's QVR name.
    parameter : str
        The parameter's name.

    Returns
    -------
    bool
        True when the family declares the parameter with rank 0.
    """
    record = FAMILIES.get(family)
    if record is None or parameter not in record.parameter_names:
        return False
    return record.parameter(parameter).rank == 0


def visited_position(plate: Plate | None, loops: Mapping[str, str]) -> tuple[str, ...]:
    """The loop variables at which a value over the visited axes is read.

    A renderer that visits a site one position at a time, a loop per
    axis of the site's plate, hands each draw one entry of every
    parameter that varies along those axes. A value whose every axis,
    batch then event, is one the loops visit holds one entry per
    position: a vector ``z`` over ``Feat`` filling a scalar family
    replicated over ``Feat`` is read as ``z[m_Feat]``, where the bare
    ``z`` would hand the scalar slot the whole vector.

    Parameters
    ----------
    plate : Plate | None
        The value's declared plate, if the program declares it.
    loops : Mapping[str, str]
        Each visited axis name mapped to its loop variable.

    Returns
    -------
    tuple[str, ...]
        One loop variable per axis of the value, in the value's axis
        order; empty when the value is undeclared, has no axes, repeats
        an axis, or has an axis the loops do not visit.
    """
    if plate is None:
        return ()
    names = tuple(str(dim.name) for dim in (*plate.batch_dims, *plate.event_dims))
    if not names or len(set(names)) != len(names):
        return ()
    if any(name not in loops for name in names):
        return ()
    return tuple(loops[name] for name in names)


__all__ = ["pointwise_let", "scalar_parameter", "take_row", "visited_position"]
