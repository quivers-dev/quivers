"""Walkers for type expressions, morphism expressions, and let-arith
expressions ().

The ``_walk_type`` handles every type expression, including
discrete_constructor (``FinSet(N)``) and continuous_constructor
(``Real(64)``, ``Simplex(K)``, ...).
"""

from __future__ import annotations

from typing import Literal

from quivers.dsl.ast_nodes import (
    ContinuousConstructor,
    DiscreteConstructor,
    Expr,
    ExprCap,
    ExprChangeBase,
    ExprChartFold,
    ExprCompose,
    ExprCup,
    ExprCurry,
    ExprDagger,
    ExprFan,
    ExprFreeze,
    ExprFromData,
    ExprIdent,
    ExprIdentity,
    ExprMarginalize,
    ExprMorphismCall,
    ExprParser,
    ExprRepeat,
    ExprScan,
    ExprStack,
    ExprTensorProduct,
    ExprTrace,
    ExprTransCompose,
    LetExprBinOp,
    LetExprBool,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLambda,
    LetExprList,
    LetExprLiteral,
    LetExprMethodCall,
    LetExprNode,
    LetExprString,
    LetExprTuple,
    LetExprUnaryOp,
    LetExprUnit,
    LetExprVar,
    LetFactorBinder,
    LetFactorCase,
    ObjectCoproduct,
    ObjectEffectApply,
    ObjectExpr,
    TypeName,
    ObjectProduct,
    ObjectSlash,
)
from quivers.dsl.parser._helpers import _required_field, _required_text
from quivers.dsl.parser._registry import ParseError, _Tree

# ---------------------------------------------------------------------------
# type expressions
# ---------------------------------------------------------------------------

_CONTINUOUS_CTORS = frozenset(
    {
        "Real",
        "Simplex",
        "Sphere",
        "Ball",
        "CholeskyFactor",
        "Covariance",
        "Correlation",
        "Orthogonal",
        "Stiefel",
        "LowerTriangular",
        "Diagonal",
    }
)


def _walk_type(t: _Tree, vid: str) -> ObjectExpr:
    """Walk a tree-sitter type-expr vertex into the unified ObjectExpr AST."""
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "object_atom":
        kids = t.positional(vid)
        if not kids:
            raise ParseError(f"object_atom {vid} has no child")
        return TypeName(name=t.text(kids[0]), line=line, col=col)
    if k == "object_paren":
        return _walk_type(t, t.positional(vid)[0])
    if k == "object_product":
        return ObjectProduct(
            components=tuple(_flatten_type(t, vid, "object_product")),
            line=line,
            col=col,
        )
    if k == "object_coproduct":
        return ObjectCoproduct(
            components=tuple(_flatten_type(t, vid, "object_coproduct")),
            line=line,
            col=col,
        )
    if k == "object_slash":
        result_vid = t.field(vid, "result")
        argument_vid = t.field(vid, "argument")
        if result_vid is None or argument_vid is None:
            raise ParseError(f"object_slash missing result/argument at {vid}")
        raw_direction = t.consts(vid).get("field:direction")
        if raw_direction not in ("/", "\\"):
            raise ParseError(
                f"object_slash has no direction field at {vid}",
            )
        direction: Literal["/", "\\"] = raw_direction  # type: ignore[assignment]
        return ObjectSlash(
            result=_walk_type(t, result_vid),
            argument=_walk_type(t, argument_vid),
            direction=direction,
            line=line,
            col=col,
        )
    if k == "object_effect_apply":
        effect_vid = t.field(vid, "effect")
        if effect_vid is None:
            raise ParseError(f"object_effect_apply missing effect at {vid}")
        arg_vids = t.fields(vid, "args")
        return ObjectEffectApply(
            effect=t.text(effect_vid),
            args=tuple(_walk_type(t, av) for av in arg_vids),
            line=line,
            col=col,
        )
    if k == "discrete_constructor":
        ctor = _constructor_name(t, vid)
        args, kwargs = _walk_constructor_args(t, vid)
        return DiscreteConstructor(
            constructor=ctor,  # type: ignore[arg-type]
            args=tuple(args),
            kwargs=kwargs,
            line=line,
            col=col,
        )
    if k == "continuous_constructor":
        ctor = _constructor_name(t, vid)
        if ctor not in _CONTINUOUS_CTORS:
            raise ParseError(f"unknown continuous constructor {ctor!r} at {vid}")
        args, kwargs = _walk_constructor_args(t, vid)
        return ContinuousConstructor(
            constructor=ctor,  # type: ignore[arg-type]
            args=tuple(args),
            kwargs=kwargs,
            line=line,
            col=col,
        )
    raise ParseError(f"unexpected type-expression kind: {k}")


def _constructor_name(t: _Tree, vid: str) -> str:
    """Return the constructor name from a discrete/continuous
    constructor vertex.

    Tree-sitter stores ``field('constructor', choice('Real',
    'Simplex', ...))`` as a string constant on the parent vertex
    (the literal-string alternatives don't materialise as their
    own child vertices). The panproto schema lifts that constant
    onto the vertex as ``field:constructor``; we read it back
    from there rather than chasing a child reference that does
    not exist.
    """
    name = t.consts(vid).get("field:constructor")
    if name is None:
        raise ParseError(
            f"constructor vertex at {vid} is missing the "
            "``field:constructor`` constant (malformed parse)",
        )
    return name


def _walk_constructor_args(
    t: _Tree, vid: str
) -> tuple[list[str], dict[str, float | int | str]]:
    """Return a constructor vertex's positional args (from
    ``field('args', ...)`` edges) and keyword args (from its trailing
    brace-delimited ``constructor_options``).
    ``Real 28 28 {low=-1.0, high=1.0}`` yields
    ``(["28", "28"], {"low": -1.0, "high": 1.0})``.
    """
    args: list[str] = []
    # ``cardinality`` covers FinSet's single-arg shape; ``args``
    # covers the multi-positional continuous-constructor shape.
    cardinality_vid = t.field(vid, "cardinality")
    if cardinality_vid is not None:
        args.append(t.text(cardinality_vid))
    for arg_vid in t.fields(vid, "args"):
        if t.kind(arg_vid) in ("integer", "float", "identifier"):
            args.append(t.text(arg_vid))
    kwargs: dict[str, float | int | str] = {}
    options_vid = t.field(vid, "options")
    if options_vid is not None:
        for kwarg_vid in t.positional(options_vid):
            if t.kind(kwarg_vid) != "constructor_kwarg":
                continue
            key_vid = t.field(kwarg_vid, "key")
            val_vid = t.field(kwarg_vid, "value")
            if key_vid is None or val_vid is None:
                raise ParseError(f"constructor_kwarg missing key/value at {kwarg_vid}")
            kwargs[t.text(key_vid)] = _constructor_kwarg_value(t, val_vid)
    return args, kwargs


def _constructor_kwarg_value(t: _Tree, vid: str) -> float | int | str:
    """Type a constructor kwarg's value.

    Identifiers stay strings; a ``signed_number`` becomes an ``int``
    when its inner literal is an integer (a leading ``-`` keeps the
    value integral) and a ``float`` when the inner literal carries a
    decimal point or exponent.
    """
    k = t.kind(vid)
    if k == "identifier":
        return t.text(vid)
    if k == "signed_number":
        kids = t.positional(vid)
        if not kids:
            raise ParseError(f"signed_number without a literal child at {vid}")
        inner_kind = t.kind(kids[0])
        text = t.text(vid)
        if inner_kind == "integer":
            return int(text)
        if inner_kind == "float":
            return float(text)
        raise ParseError(f"unexpected signed_number child kind {inner_kind!r} at {vid}")
    raise ParseError(f"unexpected constructor kwarg value kind: {k}")


def _flatten_type(t: _Tree, vid: str, op_kind: str) -> list[ObjectExpr]:
    """Flatten a left-associative binary type operator into a tuple."""
    out: list[ObjectExpr] = []
    left_vid = t.field(vid, "left")
    right_vid = t.field(vid, "right")
    if left_vid is None or right_vid is None:
        raise ParseError(f"{op_kind} missing left/right at {vid}")
    if t.kind(left_vid) == op_kind:
        out.extend(_flatten_type(t, left_vid, op_kind))
    else:
        out.append(_walk_type(t, left_vid))
    if t.kind(right_vid) == op_kind:
        out.extend(_flatten_type(t, right_vid, op_kind))
    else:
        out.append(_walk_type(t, right_vid))
    return out


# ---------------------------------------------------------------------------
# morphism expressions
# ---------------------------------------------------------------------------


def _walk_expr(t: _Tree, vid: str) -> Expr:
    """Walk a tree-sitter expr vertex into the Expr AST family."""
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "expr_paren":
        return _walk_expr(t, t.positional(vid)[0])
    if k == "expr_ident":
        kids = t.positional(vid)
        if not kids:
            raise ParseError(f"expr_ident has no child at {vid}")
        return ExprIdent(name=t.text(kids[0]), line=line, col=col)
    if k == "identity_expr":
        obj_vid = t.field(vid, "object")
        return ExprIdentity(
            object_name=_required_text(t, obj_vid, vid, "object"),
            line=line,
            col=col,
        )
    if k == "cup_expr":
        return ExprCup(
            object_name=_required_text(t, t.field(vid, "object"), vid, "object"),
            line=line,
            col=col,
        )
    if k == "cap_expr":
        return ExprCap(
            object_name=_required_text(t, t.field(vid, "object"), vid, "object"),
            line=line,
            col=col,
        )
    if k == "from_data_expr":
        key_vid = t.field(vid, "key")
        if key_vid is None:
            raise ParseError(f"from_data missing key at {vid}")
        key_text = t.text(key_vid)
        # Strip surrounding quotes if present.
        if key_text.startswith('"') and key_text.endswith('"'):
            key_text = key_text[1:-1]
        return ExprFromData(key=key_text, line=line, col=col)
    if k == "fan_expr":
        arg_vids = t.fields(vid, "args")
        return ExprFan(
            exprs=tuple(_walk_expr(t, av) for av in arg_vids),
            line=line,
            col=col,
        )
    if k == "repeat_expr":
        inner_vid = t.field(vid, "inner")
        count_vid = t.field(vid, "count")
        return ExprRepeat(
            expr=_walk_expr(t, inner_vid) if inner_vid else _err_expr(t, vid, "inner"),
            count=int(t.text(count_vid)) if count_vid else None,
            line=line,
            col=col,
        )
    if k == "stack_expr":
        inner_vid = t.field(vid, "inner")
        count_vid = t.field(vid, "count")
        if inner_vid is None or count_vid is None:
            raise ParseError(f"stack_expr missing inner/count at {vid}")
        return ExprStack(
            expr=_walk_expr(t, inner_vid),
            count=int(t.text(count_vid)),
            line=line,
            col=col,
        )
    if k == "scan_expr":
        return _walk_scan_expr(t, vid, line, col)
    if k == "parser_expr":
        return _walk_parser_expr(t, vid, line, col)
    if k == "chart_fold_expr":
        return _walk_chart_fold_expr(t, vid, line, col)
    if k == "morphism_call":
        callee_vid = t.field(vid, "callee")
        if callee_vid is None:
            raise ParseError(f"morphism_call missing callee at {vid}")
        return ExprMorphismCall(
            callee=t.text(callee_vid),
            args=tuple(t.text(av) for av in t.fields(vid, "args")),
            line=line,
            col=col,
        )
    if k == "trans_compose":
        left_vid = t.field(vid, "left")
        right_vid = t.field(vid, "right")
        if left_vid is None or right_vid is None:
            raise ParseError(f"trans_compose missing operands at {vid}")
        return ExprTransCompose(
            left=_walk_expr(t, left_vid),
            right=_walk_expr(t, right_vid),
            line=line,
            col=col,
        )
    if k == "compose_expr":
        left_vid = t.field(vid, "left")
        right_vid = t.field(vid, "right")
        if left_vid is None or right_vid is None:
            raise ParseError(f"compose_expr missing operands at {vid}")
        # ``field('op', choice('>>', '<<'))`` binds an anonymous token,
        # so the operator rides the vertex as a ``field:op`` constant
        # rather than a child vertex (same mechanism as
        # ``field:constructor``).
        op_text = t.consts(vid).get("field:op")
        if op_text is None:
            raise ParseError(f"compose_expr missing field:op constant at {vid}")
        if op_text not in (">>", "<<"):
            raise ParseError(f"unexpected compose operator {op_text!r} at {vid}")
        return ExprCompose(
            left=_walk_expr(t, left_vid),
            right=_walk_expr(t, right_vid),
            op=op_text,
            line=line,
            col=col,
        )
    if k == "tensor_expr":
        left_vid = t.field(vid, "left")
        right_vid = t.field(vid, "right")
        if left_vid is None or right_vid is None:
            raise ParseError(f"tensor_expr missing operands at {vid}")
        return ExprTensorProduct(
            left=_walk_expr(t, left_vid),
            right=_walk_expr(t, right_vid),
            line=line,
            col=col,
        )
    if k == "postfix_expr":
        inner_vid = t.field(vid, "inner")
        method_vid = t.field(vid, "method")
        if inner_vid is None or method_vid is None:
            raise ParseError(f"postfix_expr missing inner/method at {vid}")
        inner = _walk_expr(t, inner_vid)
        method_name_vid = t.field(method_vid, "name")
        if method_name_vid is not None:
            method_name = t.text(method_name_vid)
        else:
            method_name = t.consts(method_vid).get("field:name", "")
        if method_name == "marginalize":
            names = tuple(t.text(av) for av in t.fields(method_vid, "args"))
            return ExprMarginalize(inner=inner, names=names, line=line, col=col)
        if method_name == "freeze":
            return ExprFreeze(inner=inner, line=line, col=col)
        if method_name == "dagger":
            return ExprDagger(inner=inner, line=line, col=col)
        if method_name == "trace":
            args_vid = t.field(method_vid, "args")
            if args_vid is None:
                raise ParseError(f"trace() missing arg at {method_vid}")
            return ExprTrace(
                inner=inner,
                object_name=t.text(args_vid),
                line=line,
                col=col,
            )
        if method_name == "change_base":
            phi_vid = t.field(method_vid, "arg")
            if phi_vid is None:
                raise ParseError(f"change_base() missing arg at {method_vid}")
            return ExprChangeBase(
                inner=inner,
                phi=_walk_expr(t, phi_vid),
                line=line,
                col=col,
            )
        if method_name in ("curry_right", "curry_left"):
            return ExprCurry(
                inner=inner,
                direction="right" if method_name == "curry_right" else "left",
                line=line,
                col=col,
            )
        raise ParseError(f"unknown postfix method {method_name!r} at {vid}")
    raise ParseError(f"unexpected expression kind: {k}")


def _err_expr(t: _Tree, vid: str, field: str) -> Expr:
    raise ParseError(f"expression node {vid} missing {field}")


def _walk_scan_expr(t: _Tree, vid: str, line: int, col: int) -> ExprScan:
    """Walk ``scan(cell, ...)`` and its options.

    Parameters
    ----------
    t : _Tree
        The parsed tree.
    vid : str
        The ``scan_expr`` vertex.
    line : int
        The expression's 1-based line.
    col : int
        The expression's column.

    Returns
    -------
    ExprScan
        The scan, with ``init`` defaulting to ``zeros``, ``outputs`` to
        ``last``, and ``reverse`` to false.

    Raises
    ------
    ParseError
        If the cell is missing, an option is given more than once, or
        ``outputs`` names neither ``all`` nor ``last``.
    """
    inner_vid = t.field(vid, "inner")
    if inner_vid is None:
        raise ParseError(f"scan_expr missing inner at {vid}")
    values: dict[str, str] = {}
    for option in ("init", "outputs", "reverse"):
        given = t.fields(vid, option)
        if len(given) > 1:
            option_line, option_col = t.line_col(given[1])
            raise ParseError(
                f"scan option {option!r} given twice at line {option_line}, "
                f"col {option_col}; a scan takes each option at most once"
            )
        if given:
            values[option] = t.text(given[0])
    outputs = values.get("outputs", "last")
    if outputs not in ("last", "all"):
        (outputs_vid,) = t.fields(vid, "outputs")
        option_line, option_col = t.line_col(outputs_vid)
        raise ParseError(
            f"scan option outputs={outputs} at line {option_line}, col "
            f"{option_col}: a scan's outputs are 'last' (its final state) or "
            f"'all' (its state after every position)"
        )
    return ExprScan(
        expr=_walk_expr(t, inner_vid),
        init=values.get("init", "zeros"),
        outputs="all" if outputs == "all" else "last",
        reverse=values.get("reverse", "false") == "true",
        line=line,
        col=col,
    )


def _walk_parser_expr(t: _Tree, vid: str, line: int, col: int) -> ExprParser:
    args = t.fields(vid, "args")
    rules: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    terminal: str | None = None
    start: str | int = "S"
    depth = 1
    constructors: tuple[str, ...] | None = None
    for arg_vid in args:
        # The grammar binds each argument's keyword as a field on an
        # anonymous token; panproto surfaces those as ``field:<name>``
        # constraints on the argument vertex rather than as edges.
        key = t.consts(arg_vid).get("field:key")
        val_vid = t.field(arg_vid, "value")
        if key is None or val_vid is None:
            raise ParseError(f"parser_arg missing key/value at {arg_vid}")
        if key == "rules":
            rules = _ident_list_to_tuple(t, val_vid)
        elif key == "categories":
            categories = _ident_list_to_tuple(t, val_vid)
        elif key == "constructors":
            constructors = _ident_list_to_tuple(t, val_vid)
        elif key == "terminal":
            terminal = t.text(val_vid)
        elif key == "start":
            start_text = t.text(val_vid)
            try:
                start = int(start_text)
            except ValueError:
                start = start_text
        elif key == "depth":
            depth = int(t.text(val_vid))
    return ExprParser(
        rules=rules,
        categories=categories,
        terminal=terminal,
        start=start,
        depth=depth,
        constructors=constructors,
        line=line,
        col=col,
    )


def _walk_chart_fold_expr(t: _Tree, vid: str, line: int, col: int) -> ExprChartFold:
    args = t.fields(vid, "args")
    lex: Expr | None = None
    binary: Expr | None = None
    unary: Expr | None = None
    start: str | int = "S"
    depth = 1
    effect_depth = 0
    for arg_vid in args:
        # As in ``_walk_parser_expr``: the keyword rides a
        # ``field:key`` constraint, not an edge.
        key = t.consts(arg_vid).get("field:key")
        val_vid = t.field(arg_vid, "value")
        if key is None or val_vid is None:
            raise ParseError(f"chart_fold_arg missing key/value at {arg_vid}")
        if key == "lex":
            lex = _walk_expr(t, val_vid)
        elif key == "binary":
            binary = _walk_expr(t, val_vid)
        elif key == "unary":
            unary = _walk_expr(t, val_vid)
        elif key == "start":
            start_text = t.text(val_vid)
            try:
                start = int(start_text)
            except ValueError:
                start = start_text
        elif key == "depth":
            depth = int(t.text(val_vid))
        elif key == "effect_depth":
            effect_depth = int(t.text(val_vid))
    if lex is None:
        raise ParseError(f"chart_fold missing lex= at {vid}")
    return ExprChartFold(
        lex=lex,
        binary=binary,
        unary=unary,
        start=start,
        depth=depth,
        effect_depth=effect_depth,
        line=line,
        col=col,
    )


def _ident_list_to_tuple(t: _Tree, vid: str) -> tuple[str, ...]:
    if t.kind(vid) == "ident_list":
        return tuple(t.text(c) for c in t.positional(vid))
    return (t.text(vid),)


# ---------------------------------------------------------------------------
# let-arithmetic
# ---------------------------------------------------------------------------


_BINARY_OPERATORS = frozenset(
    ("+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||")
)


def _walk_let_arith(t: _Tree, vid: str) -> LetExprNode:
    """Walk a pure-expression vertex into the shared expression tree.

    Parameters
    ----------
    t : _Tree
        The parse tree.
    vid : str
        The expression vertex.

    Returns
    -------
    LetExprNode
        The expression node, carrying the vertex's line and column.

    Raises
    ------
    ParseError
        If a required field is absent, an operator is outside the
        grammar's set, or the vertex is not an expression form.
    """
    k = t.kind(vid)
    line, col = t.line_col(vid)
    if k == "let_paren":
        return _walk_let_arith(t, t.positional(vid)[0])
    if k == "let_var":
        return LetExprVar(name=t.text(vid), line=line, col=col)
    if k in ("integer", "float", "signed_number", "let_literal"):
        text = t.text(vid)
        integral = not any(character in text for character in ".eE")
        return LetExprLiteral(value=float(text), integral=integral, line=line, col=col)
    if k == "let_bool":
        return LetExprBool(value=t.text(vid) == "true", line=line, col=col)
    if k == "let_unit":
        return LetExprUnit(line=line, col=col)
    if k == "let_tuple":
        return LetExprTuple(
            items=tuple(_walk_let_arith(t, iv) for iv in t.fields(vid, "items")),
            line=line,
            col=col,
        )
    if k in ("let_string", "string"):
        text = t.text(vid)
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return LetExprString(value=text, line=line, col=col)
    if k == "let_unary":
        operand_vid = t.field(vid, "operand")
        if operand_vid is None:
            raise ParseError(f"let_unary missing operand at {vid}")
        op = t.consts(vid).get("field:op", "-")
        if op not in ("-", "not"):
            raise ParseError(f"unexpected let_unary op {op!r} at {vid}")
        return LetExprUnaryOp(
            operand=_walk_let_arith(t, operand_vid),
            op=op,  # type: ignore[arg-type]
            line=line,
            col=col,
        )
    if k == "let_binop":
        left_vid = t.field(vid, "left")
        right_vid = t.field(vid, "right")
        if left_vid is None or right_vid is None:
            raise ParseError(f"let_binop missing field at {vid}")
        op = t.consts(vid).get("field:op")
        if op is None:
            raise ParseError(f"let_binop at {vid} missing field:op constant")
        if op not in _BINARY_OPERATORS:
            raise ParseError(f"unexpected let_binop op {op!r} at {vid}")
        return LetExprBinOp(
            op=op,  # type: ignore[arg-type]
            left=_walk_let_arith(t, left_vid),
            right=_walk_let_arith(t, right_vid),
            line=line,
            col=col,
        )
    if k == "let_call":
        func_vid = t.field(vid, "func")
        if func_vid is None:
            raise ParseError(f"let_call missing func at {vid}")
        return LetExprCall(
            func=t.text(func_vid),
            args=tuple(_walk_let_arith(t, av) for av in t.fields(vid, "args")),
            line=line,
            col=col,
        )
    if k == "let_index":
        array_vid = t.field(vid, "array")
        if array_vid is None:
            raise ParseError(f"let_index missing array at {vid}")
        return LetExprIndex(
            array=_walk_let_arith(t, array_vid),
            indices=tuple(_walk_let_arith(t, iv) for iv in t.fields(vid, "indices")),
            line=line,
            col=col,
        )
    if k == "let_list":
        kids = t.positional(vid)
        return LetExprList(
            items=tuple(_walk_let_arith(t, c) for c in kids), line=line, col=col
        )
    if k == "let_lambda":
        param_vid = t.field(vid, "param")
        body_vid = t.field(vid, "body")
        if param_vid is None or body_vid is None:
            raise ParseError(f"let_lambda missing field at {vid}")
        return LetExprLambda(
            param=t.text(param_vid),
            body=_walk_let_arith(t, body_vid),
            line=line,
            col=col,
        )
    if k == "let_method_call":
        receiver_vid = t.field(vid, "receiver")
        method_vid = t.field(vid, "method")
        if receiver_vid is None or method_vid is None:
            raise ParseError(f"let_method_call missing field at {vid}")
        return LetExprMethodCall(
            receiver=_walk_let_arith(t, receiver_vid),
            method=t.text(method_vid),
            args=tuple(_walk_let_arith(t, av) for av in t.fields(vid, "args")),
            line=line,
            col=col,
        )
    if k == "let_factor":
        binders = tuple(
            LetFactorBinder(
                var=_required_text(t, t.field(bv, "var"), bv, "var"),
                index=_walk_type(t, _required_field(t, bv, "index")),
            )
            for bv in t.fields(vid, "binders")
        )
        cases = tuple(
            LetFactorCase(
                label=int(_required_text(t, t.field(cv, "label"), cv, "label")),
                value=_walk_let_arith(t, _required_field(t, cv, "value")),
            )
            for cv in t.fields(vid, "cases")
        )
        body_vid = t.field(vid, "body")
        body = _walk_let_arith(t, body_vid) if body_vid else None
        return LetExprFactor(
            binders=binders, body=body, cases=cases, line=line, col=col
        )
    raise ParseError(f"unexpected let-expression kind: {k}")


__all__ = []
