"""Value-expression AST nodes (morphism computations)."""

from typing import Literal

import didactic.api as dx


class Expr(dx.TaggedUnion, discriminator="kind"):
    """Sum of value-expression node kinds (morphism computations)."""


class ExprIdent(Expr):
    """Reference to a named morphism or binding."""

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_ident"] = "expr_ident"


class ExprIdentity(Expr):
    """Identity morphism: ``identity(A)``."""

    object_name: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_identity"] = "expr_identity"


class ExprFromData(Expr):
    """Data-derived initializer ``from_data("KEY")``.

    The string key is resolved against the runtime data dictionary
    at fit time; the morphism's tensor is the looked-up value.
    The resulting morphism is `ObservedMorphism`: its
    entries are frozen / structural inputs, not learnable
    parameters.
    """

    key: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_from_data"] = "expr_from_data"


class ExprFreeze(Expr):
    """Detach gradients: ``inner.freeze`` materialises ``inner``'s
    tensor with ``detach()`` and wraps the result as a frozen
    `ObservedMorphism`. Used to pin a learned composition
    as a structural input that the downstream model treats as
    constant."""

    inner: Expr
    line: int = 0
    col: int = 0
    kind: Literal["expr_freeze"] = "expr_freeze"


class ExprDagger(Expr):
    """Compact-closed dagger / transpose of an expression."""

    inner: Expr
    line: int = 0
    col: int = 0
    kind: Literal["expr_dagger"] = "expr_dagger"


class ExprTrace(Expr):
    """Compact-closed trace of an expression along a named object."""

    inner: Expr
    object_name: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_trace"] = "expr_trace"


class ExprChangeBase(Expr):
    """Change-of-base: apply a transformation (a carrier map, such
    as an algebra homomorphism, or a `MorphismTransformation`) to a
    morphism.

    The transformation is a first-class value: ``phi`` is any
    expression whose compile-time value is a
    `MorphismTransformation` or
    `CarrierMap`.  Concretely:

    * A bare identifier resolving a registered singleton
      (``f.change_base(expectation)``) or a let-bound trans value
      (``f.change_base(t)``).
    * A constructor call (``f.change_base(softmax(B))``).
    * A composition (``f.change_base(t1 >>> t2)``).
    """

    inner: Expr
    phi: Expr
    line: int = 0
    col: int = 0
    kind: Literal["expr_change_base"] = "expr_change_base"


class ExprTransCompose(Expr):
    """Composition of two transformations: ``t1 >>> t2`` denotes
    sequential application, first apply ``t1``, then ``t2``.

    Required: ``t1.target == t2.source`` (typed at compile time;
    a mismatch raises `CompileError`).  The result behaves
    as a transformation with ``source = t1.source`` and
    ``target = t2.target``.
    """

    left: Expr
    right: Expr
    line: int = 0
    col: int = 0
    kind: Literal["expr_trans_compose"] = "expr_trans_compose"


class ExprCup(Expr):
    """Compact-closed unit ``η_A : I → A ⊗ A`` for a named object."""

    object_name: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_cup"] = "expr_cup"


class ExprCap(Expr):
    """Compact-closed counit ``ε_A : A ⊗ A → I`` for a named object."""

    object_name: str
    line: int = 0
    col: int = 0
    kind: Literal["expr_cap"] = "expr_cap"


class ExprCompose(Expr):
    """Algebra-typed sequential composition.

    The ``op`` field selects which enrichment algebra's monoidal
    structure to use for the V-Cat composition:

    * ``">>"`` for ProductFuzzyAlgebra noisy-OR (the default).
    * ``"<<"`` for reverse ProductFuzzyAlgebra.
    * ``">=>"`` for Kleisli composition (operands' shared algebra).
    * ``"*>"`` for Markov sum-product.
    * ``"~>"`` for LogProb (log-space sum-product).
    * ``"||>"`` for Gödel (lattice min/max with Heyting implication).
    * ``"?>"`` for Viterbi (max-plus tropical, best path).
    * ``"&&>"`` for Boolean (∧/∨).
    * ``"+>"`` for Łukasiewicz (probabilistic sum bounded by 1).

    Each operator carries its own algebra; cross-operator
    composition in one chain requires explicit ``.change_base(φ)``
    between segments.
    """

    left: Expr
    right: Expr
    op: str = ">>"
    line: int = 0
    col: int = 0
    kind: Literal["expr_compose"] = "expr_compose"


class ExprTensorProduct(Expr):
    """Parallel product: ``left @ right``."""

    left: Expr
    right: Expr
    line: int = 0
    col: int = 0
    kind: Literal["expr_tensor_product"] = "expr_tensor_product"


class ExprMarginalize(Expr):
    """Marginalization: ``expr.marginalize(A, B, ...)``."""

    inner: Expr
    names: tuple[str, ...]
    line: int = 0
    col: int = 0
    kind: Literal["expr_marginalize"] = "expr_marginalize"


class ExprFan(Expr):
    """Fan-out (diagonal) composition: ``fan(f, g, ...)``."""

    exprs: tuple[Expr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["expr_fan"] = "expr_fan"


class ExprRepeat(Expr):
    """Iterated composition: ``repeat(f, n)`` or ``repeat(f)``."""

    expr: Expr
    count: int | None
    line: int = 0
    col: int = 0
    kind: Literal["expr_repeat"] = "expr_repeat"


class ExprStack(Expr):
    """Independent multi-layer composition: ``stack(f, n)``."""

    expr: Expr
    count: int
    line: int = 0
    col: int = 0
    kind: Literal["expr_stack"] = "expr_stack"


class ExprScan(Expr):
    """Temporal scan: ``scan(cell)`` with its options.

    The options are ``init`` (``zeros``, ``learned``, or a binding of
    the scanning program), ``outputs`` (``last`` for the final state,
    ``all`` for the state after every position, in input order), and
    ``reverse`` (visit the positions from the last to the first).
    """

    expr: Expr
    init: str = "zeros"
    outputs: Literal["last", "all"] = "last"
    reverse: bool = False
    line: int = 0
    col: int = 0
    kind: Literal["expr_scan"] = "expr_scan"


class ExprParser(Expr):
    """Deductive parser assembled from rules."""

    rules: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    terminal: str | None = None
    start: str | int = "S"
    depth: int = 1
    constructors: tuple[str, ...] | None = None
    line: int = 0
    col: int = 0
    kind: Literal["expr_parser"] = "expr_parser"


class ExprChartFold(Expr):
    """Desugared parser-construction primitive.

    Surface form: ``chart_fold(lex=, binary=, unary=, start=, depth=,
    effect_depth=)``.

    Constructs a chart parser from morphism-valued arguments rather
    than from a list of named rule schemas. ``lex`` is a morphism
    Token -> Cat; ``binary`` is a Cat * Cat -> Cat morphism (the
    union of all binary rule schemas); ``unary`` is an optional
    Cat -> Cat morphism (the union of all unary rule schemas);
    ``start`` is the goal category name (or integer index);
    ``depth`` is the maximum category nesting depth; ``effect_depth``
    bounds effect-stack nesting (defaults to 0).
    """

    lex: Expr
    binary: Expr | None = None
    unary: Expr | None = None
    start: str | int = "S"
    depth: int = 1
    effect_depth: int = 0
    line: int = 0
    col: int = 0
    kind: Literal["expr_chart_fold"] = "expr_chart_fold"


class ExprCurry(Expr):
    """Residuation-witness curry combinator.

    For an inner morphism ``f : X * Y -> Z`` whose codomain ``Z``
    inhabits a residuated universe, ``f.curry_right`` denotes the
    morphism ``X -> Z/Y`` and ``f.curry_left`` denotes ``Y -> X\\Z``.

    The categorical interpretation is the right (resp. left) component
    of the residuation-adjunction unit/counit triangle. Validity of the
    construction is checked at compile time: domain must factor as a
    non-commutative product and codomain must inhabit a residuated
    universe (a `FreeResiduated` object in scope).
    """

    inner: Expr
    direction: Literal["right", "left"]
    line: int = 0
    col: int = 0
    kind: Literal["expr_curry"] = "expr_curry"


class ExprMorphismCall(Expr):
    """Call expression ``callee(arg1, arg2, ...)`` resolving to a
    morphism-level operation.

    Used by `ContractionDecl` declarations: when the user
    writes ``let out = op_apply(arg1, arg2, kernel)``, the
    ``op_apply`` identifier resolves to a registered contraction
    and the arguments are looked up in the morphism scope.
    """

    callee: str
    args: tuple[str, ...]
    line: int = 0
    col: int = 0
    kind: Literal["expr_morphism_call"] = "expr_morphism_call"


__all__ = [
    "Expr",
    "ExprIdent",
    "ExprIdentity",
    "ExprFromData",
    "ExprFreeze",
    "ExprDagger",
    "ExprTrace",
    "ExprChangeBase",
    "ExprTransCompose",
    "ExprCup",
    "ExprCap",
    "ExprCompose",
    "ExprTensorProduct",
    "ExprMarginalize",
    "ExprFan",
    "ExprRepeat",
    "ExprStack",
    "ExprScan",
    "ExprParser",
    "ExprChartFold",
    "ExprCurry",
    "ExprMorphismCall",
]
