"""Let-step arithmetic expression AST nodes."""

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes.objects import ObjectExpr


class LetExprNode(dx.TaggedUnion, discriminator="kind"):
    """Sum of pure expression nodes.

    One expression tree serves both surfaces: a ``program`` let step and a
    QIEC value position parse to the same nodes. Every node carries the
    1-based source line and column of its first token, which the QIEC
    lowerer needs for diagnostics and stable provenance; a program step
    ignores them.
    """


type LetBinaryOperator = Literal[
    "+",
    "-",
    "*",
    "/",
    "%",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "&&",
    "||",
]
"""The operator of a binary let expression: arithmetic, comparison, or boolean."""


class LetExprBinOp(LetExprNode):
    """Binary operation in an expression.

    Parameters
    ----------
    op
        The operator: arithmetic, comparison, or boolean.
    left
        The left operand.
    right
        The right operand.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"let_expr_binop"``.
    """

    op: LetBinaryOperator
    left: LetExprNode
    right: LetExprNode
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_binop"] = "let_expr_binop"


class LetExprUnaryOp(LetExprNode):
    """Unary negation or boolean negation in an expression.

    Parameters
    ----------
    operand
        The operand.
    op
        ``"-"`` for arithmetic negation or ``"not"`` for boolean negation.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"let_expr_unary"``.
    """

    operand: LetExprNode
    op: Literal["-", "not"] = "-"
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_unary"] = "let_expr_unary"


class LetExprCall(LetExprNode):
    """Built-in function application in an expression.

    Parameters
    ----------
    func
        The builtin's name.
    args
        The arguments, in order.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"let_expr_call"``.
    """

    func: str
    args: tuple[LetExprNode, ...]
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_call"] = "let_expr_call"


class LetExprLiteral(LetExprNode):
    """Numeric literal in an expression.

    Parameters
    ----------
    value
        The literal's value.
    integral
        Whether the source spelled an integer, with no fraction or
        exponent. A program step treats every number as real; the QIEC
        lowerer types an integral literal as ``Int``.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"let_expr_literal"``.
    """

    value: float
    integral: bool = False
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_literal"] = "let_expr_literal"


class LetExprBool(LetExprNode):
    """Boolean literal ``true`` or ``false``.

    Parameters
    ----------
    value
        The literal's value.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"let_expr_bool"``.
    """

    value: bool
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_bool"] = "let_expr_bool"


class LetExprUnit(LetExprNode):
    """The unit literal ``unit``.

    Parameters
    ----------
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"let_expr_unit"``.
    """

    line: int = 0
    col: int = 0
    kind: Literal["let_expr_unit"] = "let_expr_unit"


class LetExprTuple(LetExprNode):
    """Tuple construction ``(a, b, ...)`` with at least two components.

    Parameters
    ----------
    items
        The components, in order.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"let_expr_tuple"``.
    """

    items: tuple[LetExprNode, ...]
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_tuple"] = "let_expr_tuple"


class LetExprVar(LetExprNode):
    """Variable reference in an expression.

    Parameters
    ----------
    name
        The variable's name.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"let_expr_var"``.
    """

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_var"] = "let_expr_var"


class LetExprIndex(LetExprNode):
    """Indexed access into a finite-domain-indexed family ``a[i]``.

    Categorically the *pullback* morphism: given a finite-index-map
    ``index : N → A`` and a per-A morphism ``arr : A → B``, the
    indexed expression ``arr[index[n]]`` denotes
    ``arr ∘ index : N → B``, the natural Kleisli pullback of
    ``arr`` along ``index``.

    Attributes
    ----------
    array : LetExprNode
        The indexed-family expression (typically a `LetExprVar`
        naming a previously-drawn plate variable).
    indices : tuple of LetExprNode
        The index expressions; supports multi-dim indexing for
        nested plates (``coefs[subj[n], k]``).
    """

    array: LetExprNode
    indices: tuple[LetExprNode, ...]
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_index"] = "let_expr_index"


class LetExprString(LetExprNode):
    """String literal in a let expression.

    Used for tokenisation, lexicon keys, and as ground-atom names
    in LF constructors like ``pred("dog")`` and
    ``forall("x", body)``. The runtime represents these as plain
    Python strings.
    """

    value: str
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_string"] = "let_expr_string"


class LetExprList(LetExprNode):
    """List literal in a let expression: ``[a, b, c]``.

    Categorically a free-monoid element over the value sublanguage;
    the runtime represents it as a Python list (with autograd
    flowing through tensor-valued items).
    """

    items: tuple[LetExprNode, ...]
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_list"] = "let_expr_list"


class LetExprLambda(LetExprNode):
    """Lambda expression ``param -> body`` in a let expression.

    Closes over the surrounding let-environment at instantiation
    time. Categorically a curried function in the Kleisli
    setting; used as the argument to fold / map / filter / reduce
    combinators.
    """

    param: str
    body: LetExprNode
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_lambda"] = "let_expr_lambda"


class LetFactorBinder(dx.Model):
    """One ``<var> : <Index>`` binder in a multi-axis factor expression.

    The variable name binds to integer index values 0, 1, ...,
    |Index|-1 in the surrounding factor body.  The index type
    expression resolves to a finite-set object whose cardinality
    is the corresponding axis size of the constructed tensor.
    """

    var: str
    index: ObjectExpr
    line: int = 0
    col: int = 0


class LetFactorCase(dx.Model):
    """One ``<integer> -> <body>`` case in a factor pattern-match.

    The label is the integer index this case populates; the body
    is the value at that index.  The compiler verifies that the
    union of labels across all cases covers ``{0, ..., |Index|-1}``
    exactly.
    """

    label: int
    value: LetExprNode
    line: int = 0
    col: int = 0


class LetExprFactor(LetExprNode):
    """Multi-axis factor expression: assemble an indexed tensor.

    Surface forms:

    ``factor v1 : I1, v2 : I2, ..., vn : In in <body>`` denotes the
    tensor of shape ``(|I1|, ..., |In|, *body_shape)`` whose value
    at position ``(i1, ..., in)`` is ``body[v_k := i_k]``.

    ``factor v : I in { 0 -> e0, 1 -> e1, ... }`` denotes the
    single-axis case-structured form: the body at index `k` is the
    expression labelled `k`, and the labels must cover
    ``{0, ..., |I|-1}`` exactly.  Multi-axis case form is not
    accepted; the uniform body form (which can itself contain
    conditionals on the binders) is the general construction.

    Categorically the left adjoint of indexing.  Single-axis is a
    section of the trivial bundle ``I -> body_type``; multi-axis is
    a section over the product ``I1 x ... x In``.  The dual
    operation is the index pullback ``arr[i1, ..., in]``
    (`LetExprIndex`); together they realize the indexed-family
    colim / lim pair in the slice category over ``FinSet``.
    """

    binders: tuple[LetFactorBinder, ...]
    body: LetExprNode | None = None
    cases: tuple[LetFactorCase, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_factor"] = "let_expr_factor"


class LetExprMethodCall(LetExprNode):
    """Method call ``receiver.method(args)`` in a let expression.

    The receiver is itself a let-expression (typically a variable
    reference to a let-bound chart-valued, list-valued, or other
    object-valued value); the method is dispatched at runtime
    against the receiver's type. Used primarily for chart-view
    queries (``chart.weight(item)``, ``chart.enumerate(pattern)``,
    ``chart.goal_weight()``).
    """

    receiver: LetExprNode
    method: str
    args: tuple[LetExprNode, ...]
    line: int = 0
    col: int = 0
    kind: Literal["let_expr_method_call"] = "let_expr_method_call"


__all__ = [
    "LetBinaryOperator",
    "LetExprNode",
    "LetExprBinOp",
    "LetExprBool",
    "LetExprUnit",
    "LetExprTuple",
    "LetExprUnaryOp",
    "LetExprCall",
    "LetExprLiteral",
    "LetExprVar",
    "LetExprIndex",
    "LetExprString",
    "LetExprList",
    "LetExprLambda",
    "LetFactorBinder",
    "LetFactorCase",
    "LetExprFactor",
    "LetExprMethodCall",
]
