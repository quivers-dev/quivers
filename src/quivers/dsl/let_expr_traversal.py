"""Exhaustive, binder-aware operations over the shared let-expression AST.

The native compiler, QIEC elaborator, and transpilation preflight all consume
the same expression nodes.  Keeping their recursive walks here makes adding a
new node an explicit cross-surface change: an unrecognised variant raises
instead of being silently treated as a leaf.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

from quivers.dsl import ast_nodes as surface


def let_expr_children(expr: surface.LetExprNode) -> tuple[surface.LetExprNode, ...]:
    """Return the immediate expression children of ``expr`` in source order.

    Parameters
    ----------
    expr : LetExprNode
        The expression to take apart.

    Returns
    -------
    tuple[LetExprNode, ...]
        The direct subexpressions, empty for a leaf.

    Raises
    ------
    TypeError
        If ``expr`` is a let-expression variant this function does not
        cover.
    """
    if isinstance(
        expr,
        surface.LetExprLiteral
        | surface.LetExprBool
        | surface.LetExprUnit
        | surface.LetExprVar
        | surface.LetExprString,
    ):
        return ()
    if isinstance(expr, surface.LetExprBinOp):
        return (expr.left, expr.right)
    if isinstance(expr, surface.LetExprUnaryOp):
        return (expr.operand,)
    if isinstance(expr, surface.LetExprCall):
        return expr.args
    if isinstance(expr, surface.LetExprIndex):
        return (expr.array, *expr.indices)
    if isinstance(expr, surface.LetExprList | surface.LetExprTuple):
        return expr.items
    if isinstance(expr, surface.LetExprLambda):
        return (expr.body,)
    if isinstance(expr, surface.LetExprFactor):
        body = () if expr.body is None else (expr.body,)
        return (*body, *(case.value for case in expr.cases))
    if isinstance(expr, surface.LetExprMethodCall):
        return (expr.receiver, *expr.args)
    if isinstance(expr, surface.QiecConstructorValue):
        return expr.fields
    raise TypeError(
        f"let_expr_children: unsupported let-expression variant {type(expr).__name__}"
    )


def walk_let_expr(expr: surface.LetExprNode) -> Iterator[surface.LetExprNode]:
    """Yield ``expr`` and all descendants in preorder.

    Parameters
    ----------
    expr : LetExprNode
        The root of the walk.

    Yields
    ------
    LetExprNode
        ``expr`` first, then each subexpression before its own children.

    Raises
    ------
    TypeError
        If the walk reaches an unsupported let-expression variant.
    """
    yield expr
    for child in let_expr_children(expr):
        yield from walk_let_expr(child)


def map_let_expr(
    expr: surface.LetExprNode,
    transform: Callable[[surface.LetExprNode], surface.LetExprNode],
) -> surface.LetExprNode:
    """Rebuild ``expr`` bottom-up, applying ``transform`` at every node.

    Each node is first rebuilt over its rewritten children and then
    passed to ``transform``, so ``transform`` sees a node whose
    subexpressions are already rewritten. The rebuild is structural and
    does not track binders, so a rewrite of variable names belongs in
    `substitute_let_expr`.

    Parameters
    ----------
    expr : LetExprNode
        The root of the rewrite.
    transform : Callable[[LetExprNode], LetExprNode]
        The rewrite applied to each rebuilt node.

    Returns
    -------
    LetExprNode
        The rewritten expression.

    Raises
    ------
    TypeError
        If the rewrite reaches an unsupported let-expression variant.
    """

    def rebuild(node: surface.LetExprNode) -> surface.LetExprNode:
        """Rebuild an expression with the rewrite applied below it.

        Parameters
        ----------
        node : surface.LetExprNode
            The expression.

        Returns
        -------
        surface.LetExprNode
            The rebuilt expression.
        """
        if isinstance(
            node,
            surface.LetExprLiteral
            | surface.LetExprBool
            | surface.LetExprUnit
            | surface.LetExprVar
            | surface.LetExprString,
        ):
            return transform(node)
        if isinstance(node, surface.LetExprBinOp):
            return transform(
                node.with_(left=rebuild(node.left), right=rebuild(node.right))
            )
        if isinstance(node, surface.LetExprUnaryOp):
            return transform(node.with_(operand=rebuild(node.operand)))
        if isinstance(node, surface.LetExprCall):
            return transform(node.with_(args=tuple(map(rebuild, node.args))))
        if isinstance(node, surface.LetExprIndex):
            return transform(
                node.with_(
                    array=rebuild(node.array),
                    indices=tuple(map(rebuild, node.indices)),
                )
            )
        if isinstance(node, surface.LetExprList | surface.LetExprTuple):
            return transform(node.with_(items=tuple(map(rebuild, node.items))))
        if isinstance(node, surface.LetExprLambda):
            return transform(node.with_(body=rebuild(node.body)))
        if isinstance(node, surface.LetExprFactor):
            return transform(
                node.with_(
                    body=None if node.body is None else rebuild(node.body),
                    cases=tuple(
                        case.with_(value=rebuild(case.value)) for case in node.cases
                    ),
                )
            )
        if isinstance(node, surface.LetExprMethodCall):
            return transform(
                node.with_(
                    receiver=rebuild(node.receiver),
                    args=tuple(map(rebuild, node.args)),
                )
            )
        if isinstance(node, surface.QiecConstructorValue):
            return transform(node.with_(fields=tuple(map(rebuild, node.fields))))
        raise TypeError(
            f"map_let_expr: unsupported let-expression variant {type(node).__name__}"
        )

    return rebuild(expr)


def free_let_names(
    expr: surface.LetExprNode,
    *,
    bound: frozenset[str] = frozenset(),
    include_callees: bool = False,
) -> tuple[str, ...]:
    """Return free value names in first-use order.

    Lambda parameters and factor indices bind in their bodies. Call names are
    declarations by default; callers that model a callee as a host value can
    include them explicitly.

    Parameters
    ----------
    expr : LetExprNode
        The expression to scan.
    bound : frozenset[str]
        Names already bound around ``expr``, which are not free.
    include_callees : bool
        Whether the function name of a call counts as a free name.

    Returns
    -------
    tuple[str, ...]
        Each free name once, in the order of its first occurrence.

    Raises
    ------
    TypeError
        If the scan reaches an unsupported let-expression variant.
    """
    found: list[str] = []

    def visit(node: surface.LetExprNode, local_bound: frozenset[str]) -> None:
        if isinstance(node, surface.LetExprVar):
            if node.name not in local_bound and node.name not in found:
                found.append(node.name)
            return
        if isinstance(node, surface.LetExprCall):
            if (
                include_callees
                and node.func not in local_bound
                and node.func not in found
            ):
                found.append(node.func)
            for argument in node.args:
                visit(argument, local_bound)
            return
        if isinstance(node, surface.LetExprLambda):
            visit(node.body, local_bound | {node.param})
            return
        if isinstance(node, surface.LetExprFactor):
            inner = local_bound | {binder.var for binder in node.binders}
            if node.body is not None:
                visit(node.body, inner)
            for case in node.cases:
                visit(case.value, inner)
            return
        for child in let_expr_children(node):
            visit(child, local_bound)

    visit(expr, bound)
    return tuple(found)


def free_let_callees(
    expr: surface.LetExprNode,
    *,
    bound: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Return the free callee names of calls in first-use order.

    A call ``f(...)`` whose ``f`` is not bound by an enclosing lambda
    parameter or factor index contributes ``f``.

    Parameters
    ----------
    expr : LetExprNode
        The expression to scan.
    bound : frozenset[str]
        Names already bound around ``expr``, which are not free.

    Returns
    -------
    tuple[str, ...]
        Each free callee once, in the order of its first occurrence.

    Raises
    ------
    TypeError
        If the scan reaches an unsupported let-expression variant.
    """
    found: list[str] = []

    def visit(node: surface.LetExprNode, local_bound: frozenset[str]) -> None:
        if isinstance(node, surface.LetExprCall):
            if node.func not in local_bound and node.func not in found:
                found.append(node.func)
        elif isinstance(node, surface.LetExprLambda):
            visit(node.body, local_bound | {node.param})
            return
        elif isinstance(node, surface.LetExprFactor):
            inner = local_bound | {binder.var for binder in node.binders}
            for child in let_expr_children(node):
                visit(child, inner)
            return
        for child in let_expr_children(node):
            visit(child, local_bound)

    visit(expr, bound)
    return tuple(found)


def substitute_let_expr(
    expr: surface.LetExprNode,
    substitution: Mapping[str, surface.LetExprNode],
    *,
    callees: Mapping[str, str] | None = None,
) -> surface.LetExprNode:
    """Substitute expressions for free variables in ``expr`` without capture.

    A lambda parameter or factor binder that would capture a free name of a
    replacement is renamed to a fresh name first.

    Parameters
    ----------
    expr : LetExprNode
        The expression to rewrite.
    substitution : Mapping[str, LetExprNode]
        The replacement for each free variable name.
    callees : Mapping[str, str] or None
        The replacement name for each free callee of a call
        ``f(...)``. A callee bound by an enclosing lambda or factor is
        not free and keeps its name; a binder that would capture a
        replacement callee name is renamed like any other capture.

    Returns
    -------
    LetExprNode
        ``expr`` with every free occurrence of a key replaced.

    Raises
    ------
    TypeError
        If the rewrite reaches an unsupported let-expression variant.
    """
    callee_map: Mapping[str, str] = {} if callees is None else callees
    occupied = set(free_let_names(expr, include_callees=True))
    for replacement in substitution.values():
        occupied.update(free_let_names(replacement, include_callees=True))
    occupied.update(callee_map.values())
    for node in walk_let_expr(expr):
        if isinstance(node, surface.LetExprLambda):
            occupied.add(node.param)
        elif isinstance(node, surface.LetExprFactor):
            occupied.update(binder.var for binder in node.binders)

    def fresh(base: str) -> str:
        suffix = 1
        candidate = f"{base}_{suffix}"
        while candidate in occupied:
            suffix += 1
            candidate = f"{base}_{suffix}"
        occupied.add(candidate)
        return candidate

    def replacement_names(
        active: Mapping[str, surface.LetExprNode],
        active_callees: Mapping[str, str],
        body: surface.LetExprNode,
    ) -> set[str]:
        used_keys = set(free_let_names(body)) & active.keys()
        names = {name for key in used_keys for name in free_let_names(active[key])}
        used_callees = set(free_let_names(body, include_callees=True))
        names.update(
            active_callees[key] for key in used_callees & active_callees.keys()
        )
        return names

    def replace(
        node: surface.LetExprNode,
        active: Mapping[str, surface.LetExprNode],
        active_callees: Mapping[str, str],
    ) -> surface.LetExprNode:
        if isinstance(node, surface.LetExprVar):
            return active.get(node.name, node)
        if isinstance(
            node,
            surface.LetExprLiteral
            | surface.LetExprBool
            | surface.LetExprUnit
            | surface.LetExprString,
        ):
            return node
        if isinstance(node, surface.LetExprBinOp):
            return node.with_(
                left=replace(node.left, active, active_callees),
                right=replace(node.right, active, active_callees),
            )
        if isinstance(node, surface.LetExprUnaryOp):
            return node.with_(operand=replace(node.operand, active, active_callees))
        if isinstance(node, surface.LetExprCall):
            return node.with_(
                func=active_callees.get(node.func, node.func),
                args=tuple(replace(item, active, active_callees) for item in node.args),
            )
        if isinstance(node, surface.LetExprIndex):
            return node.with_(
                array=replace(node.array, active, active_callees),
                indices=tuple(
                    replace(item, active, active_callees) for item in node.indices
                ),
            )
        if isinstance(node, surface.LetExprList | surface.LetExprTuple):
            return node.with_(
                items=tuple(
                    replace(item, active, active_callees) for item in node.items
                )
            )
        if isinstance(node, surface.LetExprLambda):
            inner = {key: value for key, value in active.items() if key != node.param}
            inner_callees = {
                key: value for key, value in active_callees.items() if key != node.param
            }
            param = node.param
            body = node.body
            if param in replacement_names(inner, inner_callees, body):
                renamed = fresh(param)
                body = replace(
                    body,
                    {
                        param: surface.LetExprVar(
                            name=renamed,
                            line=node.line,
                            col=node.col,
                        )
                    },
                    {param: renamed},
                )
                param = renamed
            return node.with_(param=param, body=replace(body, inner, inner_callees))
        if isinstance(node, surface.LetExprFactor):
            bound = {binder.var for binder in node.binders}
            inner = {key: value for key, value in active.items() if key not in bound}
            inner_callees = {
                key: value for key, value in active_callees.items() if key not in bound
            }
            bodies = (() if node.body is None else (node.body,)) + tuple(
                case.value for case in node.cases
            )
            captures = {
                name
                for body in bodies
                for name in replacement_names(inner, inner_callees, body)
            }
            binders = list(node.binders)
            body = node.body
            cases = node.cases
            for position, binder in enumerate(binders):
                if binder.var not in captures:
                    continue
                renamed = fresh(binder.var)
                alpha = {
                    binder.var: surface.LetExprVar(
                        name=renamed,
                        line=binder.line,
                        col=binder.col,
                    )
                }
                alpha_callees = {binder.var: renamed}
                body = None if body is None else replace(body, alpha, alpha_callees)
                cases = tuple(
                    case.with_(value=replace(case.value, alpha, alpha_callees))
                    for case in cases
                )
                binders[position] = binder.with_(var=renamed)
            return node.with_(
                binders=tuple(binders),
                body=None if body is None else replace(body, inner, inner_callees),
                cases=tuple(
                    case.with_(value=replace(case.value, inner, inner_callees))
                    for case in cases
                ),
            )
        if isinstance(node, surface.LetExprMethodCall):
            return node.with_(
                receiver=replace(node.receiver, active, active_callees),
                args=tuple(replace(item, active, active_callees) for item in node.args),
            )
        if isinstance(node, surface.QiecConstructorValue):
            return node.with_(
                fields=tuple(
                    replace(field, active, active_callees) for field in node.fields
                )
            )
        raise TypeError(
            f"substitute_let_expr: unsupported let-expression variant "
            f"{type(node).__name__}"
        )

    return replace(expr, substitution, callee_map)


__all__ = [
    "free_let_callees",
    "free_let_names",
    "let_expr_children",
    "map_let_expr",
    "substitute_let_expr",
    "walk_let_expr",
]
