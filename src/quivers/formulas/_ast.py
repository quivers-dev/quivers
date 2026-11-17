"""Shared surface-expression constructors for formula lowering."""

from __future__ import annotations

from functools import lru_cache

from quivers.dsl._value_expressions import _parse_value_expression
from quivers.dsl.let_expr_traversal import substitute_let_expr
from quivers.dsl.ast_nodes import (
    LetStep,
    Module,
    ObjectDecl,
    TypeFromExpr,
    DiscreteConstructor,
    LetExprNode,
    LetExprVar,
    LetExprLiteral,
    LetExprList,
    LetExprBinOp,
    LetExprCall,
    LetExprIndex,
    LetExprFactor,
    LetFactorBinder,
    LetFactorCase,
    TypeName,
)


@lru_cache(maxsize=None)
def _expression(text: str) -> LetExprNode:
    """Parse one declared QVR expression, once per distinct text.

    Parameters
    ----------
    text : str
        A deterministic QVR expression over named quantities.

    Returns
    -------
    LetExprNode
        The parsed expression.

    Raises
    ------
    ValueError
        If the text is not exactly one expression.
    """
    return _parse_value_expression(text)


def _bind(text: str, **values: LetExprNode) -> LetExprNode:
    """Instantiate a declared expression at the given named quantities.

    Parameters
    ----------
    text : str
        A deterministic QVR expression over named quantities.
    **values : LetExprNode
        Replacement for each named quantity; other names are kept.

    Returns
    -------
    LetExprNode
        The expression with every named quantity replaced.
    """
    return (
        substitute_let_expr(_expression(text), values) if values else _expression(text)
    )


def _let(name: str, value: LetExprNode) -> LetStep:
    """Bind one deterministic surface expression.

    Parameters
    ----------
    name : str
        Result binder.
    value : LetExprNode
        Deterministic value.

    Returns
    -------
    LetStep
        Source binding without an invented source origin.
    """
    return LetStep(name=name, value=value)


def _var(name: str) -> LetExprVar:
    """Refer to one surface binder.

    Parameters
    ----------
    name : str
        Binder name.

    Returns
    -------
    LetExprVar
        Identifier expression.
    """
    return LetExprVar(name=name)


def _add(*terms: LetExprNode) -> LetExprNode:
    """Add expressions in authored order, using zero for an empty sum.

    Parameters
    ----------
    *terms : LetExprNode
        Summands in evaluation order.

    Returns
    -------
    LetExprNode
        Left-associated sum or its zero identity.
    """
    if not terms:
        return LetExprLiteral(value=0.0)
    out = terms[0]
    for t in terms[1:]:
        out = LetExprBinOp(op="+", left=out, right=t)
    return out


def _mul(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    """Multiply two expressions.

    Parameters
    ----------
    left : LetExprNode
        Left operand.
    right : LetExprNode
        Right operand.

    Returns
    -------
    LetExprBinOp
        Multiplication expression.
    """
    return LetExprBinOp(op="*", left=left, right=right)


def _sub(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    """Subtract one expression from another.

    Parameters
    ----------
    left : LetExprNode
        Minuend.
    right : LetExprNode
        Subtrahend.

    Returns
    -------
    LetExprBinOp
        Subtraction expression.
    """
    return LetExprBinOp(op="-", left=left, right=right)


def _div(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    """Divide two expressions without changing checked domain semantics.

    Parameters
    ----------
    left : LetExprNode
        Numerator.
    right : LetExprNode
        Denominator.

    Returns
    -------
    LetExprBinOp
        Division expression.
    """
    return LetExprBinOp(op="/", left=left, right=right)


def _call(name: str, *args: LetExprNode) -> LetExprCall:
    """Apply one source function to ordered arguments.

    Parameters
    ----------
    name : str
        Function name.
    *args : LetExprNode
        Actual value arguments.

    Returns
    -------
    LetExprCall
        Application with default source metadata.
    """
    return LetExprCall(func=name, args=tuple(args))


def _index(name: str, *indices: LetExprNode) -> LetExprIndex:
    """Select coordinates from a named surface tensor.

    Parameters
    ----------
    name : str
        Tensor binder.
    *indices : LetExprNode
        Ordered coordinate expressions.

    Returns
    -------
    LetExprIndex
        Indexed expression retaining its coordinate order.
    """
    return LetExprIndex(array=_var(name), indices=tuple(indices))


def _factor(
    binders: tuple[tuple[str, str], ...],
    *,
    body: LetExprNode | None = None,
    cases: tuple[tuple[int, LetExprNode], ...] = (),
) -> LetExprFactor:
    """Construct a finite tensor from scoped coordinates or explicit cases.

    Parameters
    ----------
    binders : tuple[tuple[str, str], ...]
        Coordinate names and finite axis names.
    body : LetExprNode or None
        Expression evaluated at each coordinate.
    cases : tuple[tuple[int, LetExprNode], ...]
        Explicit labeled branches when no common body is supplied.

    Returns
    -------
    LetExprFactor
        Surface factor with the original binder and branch order.
    """
    return LetExprFactor(
        binders=tuple(
            LetFactorBinder(var=name, index=TypeName(name=index))
            for name, index in binders
        ),
        body=body,
        cases=tuple(LetFactorCase(label=label, value=value) for label, value in cases),
    )


def _real_tensor_literal(values):
    """Retain a nested immutable numeric table as a Real source tensor.

    Parameters
    ----------
    values : tuple
        Nested numeric tuple, preserving structural axes.

    Returns
    -------
    LetExprList
        Nested Real literals without observation-row replication.
    """
    return LetExprList(
        items=tuple(
            _real_tensor_literal(value)
            if isinstance(value, tuple)
            else LetExprLiteral(value=float(value))
            for value in values
        )
    )


__all__ = []


def finite_object_sizes(module: Module) -> dict[str, int]:
    """Read explicit finite extents without identifying distinct nominal axes.

    Parameters
    ----------
    module : Module
        Formula-emitted literal finite object declarations.

    Returns
    -------
    dict[str, int]
        Original object names and their declared cardinalities.
    """
    sizes = {}
    for statement in module.statements:
        if not isinstance(statement, ObjectDecl) or not isinstance(
            statement.init, TypeFromExpr
        ):
            continue
        expression = statement.init.expr
        if (
            isinstance(expression, DiscreteConstructor)
            and expression.constructor == "FinSet"
            and len(expression.args) == 1
        ):
            value = expression.args[0]
        elif isinstance(expression, TypeName):
            value = expression.name
        else:
            continue
        try:
            extent = int(value)
        except TypeError, ValueError:
            continue
        sizes.update((name, extent) for name in statement.names)
    return sizes
