"""The array rank of a let expression in a rendered program.

A let builtin that reduces (``sum``, ``mean``, ``prod``, ``max``, ``min``,
``logsumexp``) collapses the last axis of its argument, and a rowwise one
(``softmax``, ``log_softmax``, ``cumsum``, ``sort``, ``normalize``) acts
along it; the checked elaboration, the reference machine, and the
compiled runtime all read them so. A renderer has to know how many axes
the argument carries to spell that reduction in the target: a vector
reduces to a number under any target's whole-array reduction, while an
argument of two or more axes needs the target's reduction along its last
axis. This module computes the argument's rank from the plates the plan
records for every bound name, which the plan reads off the checked types.
"""

from __future__ import annotations

from collections.abc import Mapping

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprList,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.transpile.ir import (
    Dim,
    IRArg,
    IRArgRef,
    IRDeterministic,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRSample,
    IRScore,
    LetExprAffineMap,
    Plate,
)

#: The let builtins that collapse their argument's last axis.
LAST_AXIS_REDUCTIONS: frozenset[str] = frozenset(
    {"sum", "mean", "prod", "max", "min", "logsumexp"}
)


def _plate_rank(plate: Plate) -> int:
    """The number of axes a value bound over a plate carries.

    Parameters
    ----------
    plate : Plate
        The binding's plate.

    Returns
    -------
    int
        Its batch axes and its event axes together.
    """
    return len(plate.batch_dims) + len(plate.event_dims)


def name_rank_map(ir: IRProgram) -> dict[str, int]:
    """Map every name the plan binds to the rank of its value.

    Parameters
    ----------
    ir : IRProgram
        The plan.

    Returns
    -------
    dict[str, int]
        Each input's, site's, and deterministic's batch and event axes
        together; a score's value is a number.
    """
    ranks: dict[str, int] = {item.name: _plate_rank(item.plate) for item in ir.inputs}
    _walk_ranks(ir.body, ranks)
    return ranks


def name_plate_map(ir: IRProgram) -> dict[str, Plate]:
    """Map every IR-bound name to the [`Plate`][quivers.transpile.ir.Plate]
    its binding site declares.

    The marginalize lowering reads both halves: the batch dims decide
    whether a weight tensor is grouped over the marginalize grouping
    axis (and so needs the observe's `via` index to reach the
    observation axis), and the trailing event extent supplies the
    class count of a `class_index` atom set.

    Parameters
    ----------
    ir : IRProgram
        The plan.

    Returns
    -------
    dict[str, Plate]
        Each input's, site's, deterministic's and marginalized latent's
        plate, by name.
    """
    out: dict[str, Plate] = {}
    for inp in ir.inputs:
        out[inp.name] = inp.plate
    _walk_for_name_plates(ir.body, out)
    return out


def _walk_for_name_plates(body: tuple[IRNode, ...], out: dict[str, Plate]) -> None:
    for node in body:
        if isinstance(node, (IRSample, IRObserve, IRDeterministic)):
            out[node.name] = node.plate
        elif isinstance(node, IRMarginalize):
            out[node.latent] = node.plate
            _walk_for_name_plates(node.scope, out)


def carried_dims(arg: IRArg, name_plates: Mapping[str, Plate]) -> tuple[Dim, ...]:
    """The axes a family argument's value carries, outermost first.

    A reference carries every axis of the name it reads, batch then
    event. A gather ``mu[z]`` reads ``mu`` at one index per indexed
    axis, so each gathered element carries the axes of ``mu`` after
    those: none for an entry of a vector, the columns for a row of a
    matrix. The index's own axes lead the gathered value and are the
    plate the site iterates, so they are not among the carried axes.
    Any other argument (a literal, a list, an expression) carries none
    a plate states.

    Parameters
    ----------
    arg : IRArg
        The argument.
    name_plates : Mapping[str, Plate]
        The plate of each name the plan binds.

    Returns
    -------
    tuple[Dim, ...]
        The axes the argument's value carries beyond the plate the site
        iterates.
    """
    if not isinstance(arg, IRArgRef):
        return ()
    plate = name_plates.get(arg.name)
    if plate is None:
        return ()
    axes = (*plate.batch_dims, *plate.event_dims)
    return axes[len(arg.indices) :] if arg.indices else axes


def carries_event_dims(
    args: tuple[IRArg, ...],
    event_dims: tuple[Dim, ...],
    name_plates: Mapping[str, Plate],
) -> bool:
    """Whether some argument's value already ends in a site's event axes.

    A family whose argument carries the site's event axes draws those
    axes from the argument itself, so the site lifts them into its event
    with no broadcast of the family to their extents.

    Parameters
    ----------
    args : tuple[IRArg, ...]
        The family's arguments.
    event_dims : tuple[Dim, ...]
        The event axes the site lifts.
    name_plates : Mapping[str, Plate]
        The plate of each name the plan binds.

    Returns
    -------
    bool
        Whether an argument's carried axes end in ``event_dims``.
    """
    if not event_dims:
        return False
    names = tuple(str(dim.name) for dim in event_dims)
    for arg in args:
        carried = tuple(str(dim.name) for dim in carried_dims(arg, name_plates))
        if len(carried) >= len(names) and carried[len(carried) - len(names) :] == names:
            return True
    return False


def _walk_ranks(body: tuple[IRNode, ...], ranks: dict[str, int]) -> None:
    """Record the ranks of the names a body binds.

    Parameters
    ----------
    body : tuple[IRNode, ...]
        The nodes.
    ranks : dict[str, int]
        The map, extended in place.
    """
    for node in body:
        if isinstance(node, IRSample | IRObserve | IRDeterministic):
            ranks[node.name] = _plate_rank(node.plate)
        elif isinstance(node, IRMarginalize):
            ranks[node.latent] = 0
            _walk_ranks(node.scope, ranks)
        elif isinstance(node, IRScore):
            ranks[node.name] = 0


def let_expr_rank(expr: LetExprNode, ranks: Mapping[str, int]) -> int:
    """The number of axes a let expression's value carries.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    ranks : Mapping[str, int]
        The rank of each name it may read; a name absent from it (a
        factor binder, a loop index, a number) is rank zero.

    Returns
    -------
    int
        The rank: a name's own; the wider operand's under a binary
        operator; one fewer than the argument's under a one-argument
        reduction; the argument's under any other call; the array's
        less one per index, plus each index's own rank, under a
        subscript; one more than the widest item for a list; and one
        axis per binder over the body's for a factor.
    """
    if isinstance(expr, LetExprVar):
        return ranks.get(expr.name, 0)
    if isinstance(expr, LetExprBinOp):
        return max(let_expr_rank(expr.left, ranks), let_expr_rank(expr.right, ranks))
    if isinstance(expr, LetExprUnaryOp):
        return let_expr_rank(expr.operand, ranks)
    if isinstance(expr, LetExprCall):
        inner = max((let_expr_rank(arg, ranks) for arg in expr.args), default=0)
        if expr.func in LAST_AXIS_REDUCTIONS and len(expr.args) == 1:
            return max(0, inner - 1)
        return inner
    if isinstance(expr, LetExprIndex):
        rank = let_expr_rank(expr.array, ranks)
        for index in expr.indices:
            rank = max(0, rank - 1) + let_expr_rank(index, ranks)
        return rank
    if isinstance(expr, LetExprList):
        return 1 + max((let_expr_rank(item, ranks) for item in expr.items), default=0)
    if isinstance(expr, LetExprFactor):
        if expr.body is not None:
            return len(expr.binders) + let_expr_rank(expr.body, ranks)
        return len(expr.binders) + max(
            (let_expr_rank(case.value, ranks) for case in expr.cases), default=0
        )
    if isinstance(expr, LetExprAffineMap):
        return 1
    return 0


def reduces_last_of_several_axes(expr: LetExprCall, ranks: Mapping[str, int]) -> bool:
    """Whether a reduction must name the last axis of its argument.

    Parameters
    ----------
    expr : LetExprCall
        The call.
    ranks : Mapping[str, int]
        The rank of each name the argument may read.

    Returns
    -------
    bool
        True for a one-argument reduction whose argument has two or
        more axes, where a target's whole-array reduction would collapse
        more than the last one.
    """
    return (
        expr.func in LAST_AXIS_REDUCTIONS
        and len(expr.args) == 1
        and let_expr_rank(expr.args[0], ranks) >= 2
    )


def several_axis_reductions(
    expr: LetExprNode, ranks: Mapping[str, int]
) -> tuple[LetExprCall, ...]:
    """Every reduction in an expression whose argument has several axes.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    ranks : Mapping[str, int]
        The rank of each name it may read.

    Returns
    -------
    tuple[LetExprCall, ...]
        The reductions that must name their argument's last axis, in
        preorder.
    """
    found: list[LetExprCall] = []
    if isinstance(expr, LetExprCall):
        if reduces_last_of_several_axes(expr, ranks):
            found.append(expr)
        children: tuple[LetExprNode, ...] = expr.args
    elif isinstance(expr, LetExprBinOp):
        children = (expr.left, expr.right)
    elif isinstance(expr, LetExprUnaryOp):
        children = (expr.operand,)
    elif isinstance(expr, LetExprIndex):
        children = (expr.array, *expr.indices)
    elif isinstance(expr, LetExprList):
        children = expr.items
    elif isinstance(expr, LetExprFactor):
        children = (
            (expr.body,)
            if expr.body is not None
            else tuple(case.value for case in expr.cases)
        )
    else:
        children = ()
    for child in children:
        found.extend(several_axis_reductions(child, ranks))
    return tuple(found)


__all__ = [
    "LAST_AXIS_REDUCTIONS",
    "let_expr_rank",
    "name_rank_map",
    "reduces_last_of_several_axes",
    "several_axis_reductions",
]
