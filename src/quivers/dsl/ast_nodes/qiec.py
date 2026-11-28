"""Typed source AST for the Quivers Indexed Effect Core surface.

These nodes describe authored QVR syntax, not stable QIEC identities.  Stable
nominal identifiers, lexical instance identifiers, static scopes, and source
provenance are derived by the QVR-to-QIEC lowerer from names and the source
locations carried by every node here.

The QIEC surface uses distinct nodes alongside the categorical-object and
structural-signature ASTs.  In particular, :class:`QiecEffectRef` is not
an ``ObjectEffectApply`` and :class:`QiecFamilyConstructor` is not the
structural-signature ``ConstructorDecl``.
"""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes._shared import Statement
from quivers.dsl.ast_nodes.let_expressions import LetExprNode


# ---------------------------------------------------------------------------
# Kinds, index terms, and heterogeneous telescopes
# ---------------------------------------------------------------------------


class QiecKindExpr(dx.TaggedUnion, discriminator="kind"):
    """A QIEC kind appearing in source."""


class QiecTypeKind(QiecKindExpr):
    """The kind ``Type`` of value types.

    Parameters
    ----------
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_type_kind"``.
    """

    line: int = 0
    col: int = 0
    kind: Literal["qiec_type_kind"] = "qiec_type_kind"


class QiecEffectKind(QiecKindExpr):
    """The kind ``Effect`` of effect interfaces.

    Parameters
    ----------
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_effect_kind"``.
    """

    line: int = 0
    col: int = 0
    kind: Literal["qiec_effect_kind"] = "qiec_effect_kind"


class QiecIndexExpr(dx.TaggedUnion, discriminator="kind"):
    """A source-level term in a closed index sort."""


class QiecIndexName(QiecIndexExpr):
    """A reference to an index binder in scope.

    Parameters
    ----------
    name
        The binder's name.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_index_name"``.
    """

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_index_name"] = "qiec_index_name"


class QiecIndexLiteral(QiecIndexExpr):
    """A natural-number index literal.

    Parameters
    ----------
    value
        The literal's value.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_index_literal"``.
    """

    value: int
    line: int = 0
    col: int = 0
    kind: Literal["qiec_index_literal"] = "qiec_index_literal"


class QiecIndexApplication(QiecIndexExpr):
    """A user index-sort constructor applied to index arguments.

    Parameters
    ----------
    constructor
        The constructor's name.
    arguments
        The index arguments, in order.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_index_application"``.
    """

    constructor: str
    arguments: tuple[QiecIndexExpr, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["qiec_index_application"] = "qiec_index_application"


class QiecShapeIndex(QiecIndexExpr):
    """A shape written as a bracketed list of dimensions.

    Parameters
    ----------
    dimensions
        The dimension indices, outermost first.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_shape_index"``.
    """

    dimensions: tuple[QiecIndexExpr, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["qiec_shape_index"] = "qiec_shape_index"


class QiecIndexSort(dx.TaggedUnion, discriminator="kind"):
    """A closed index sort annotation."""


class QiecNatSort(QiecIndexSort):
    """The sort ``Nat`` of natural-number indices.

    Parameters
    ----------
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_nat_sort"``.
    """

    line: int = 0
    col: int = 0
    kind: Literal["qiec_nat_sort"] = "qiec_nat_sort"


class QiecShapeSort(QiecIndexSort):
    """The sort of shapes, of a fixed rank or of any rank.

    Parameters
    ----------
    rank
        The rank a shape must have, or ``None`` for any rank.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_shape_sort"``.
    """

    rank: int | None = None
    line: int = 0
    col: int = 0
    kind: Literal["qiec_shape_sort"] = "qiec_shape_sort"


class QiecContextSort(QiecIndexSort):
    """The sort of contexts over a structural signature.

    Parameters
    ----------
    signature
        The signature's name.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_context_sort"``.
    """

    signature: str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_context_sort"] = "qiec_context_sort"


class QiecUserIndexSort(QiecIndexSort):
    """A reference to a user-declared closed index sort.

    Parameters
    ----------
    name
        The sort's name.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_user_index_sort"``.
    """

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_user_index_sort"] = "qiec_user_index_sort"


class QiecBinder(dx.TaggedUnion, discriminator="kind"):
    """One entry in a heterogeneous static telescope."""


class QiecTypeBinder(QiecBinder):
    """A telescope binder of kind ``Type`` or ``Effect``.

    Parameters
    ----------
    name
        The binder's name.
    binder_kind
        The kind the binder ranges over.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_type_binder"``.
    """

    name: str
    binder_kind: QiecKindExpr
    line: int = 0
    col: int = 0
    kind: Literal["qiec_type_binder"] = "qiec_type_binder"


class QiecIndexBinder(QiecBinder):
    """A telescope binder over a closed index sort.

    Parameters
    ----------
    name
        The binder's name.
    sort
        The sort the binder ranges over.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_index_binder"``.
    """

    name: str
    sort: QiecIndexSort
    line: int = 0
    col: int = 0
    kind: Literal["qiec_index_binder"] = "qiec_index_binder"


class QiecEffectBinder(QiecBinder):
    """A telescope binder over effect interfaces.

    Parameters
    ----------
    name
        The binder's name.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_effect_binder"``.
    """

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_effect_binder"] = "qiec_effect_binder"


class QiecIndexConstructor(dx.Model):
    """One constructor of a closed user-defined index sort."""

    name: str
    arguments: tuple[QiecIndexSort, ...] = ()
    line: int = 0
    col: int = 0


# ---------------------------------------------------------------------------
# Static type and effect terms
# ---------------------------------------------------------------------------


class QiecTypeExpr(dx.TaggedUnion, discriminator="kind"):
    """A value type in the QIEC source stratum."""


type QiecStaticArgument = QiecTypeExpr | QiecIndexLiteral
"""A static argument: a type or effect written as type syntax, or an index literal."""


class QiecTypeName(QiecTypeExpr):
    """A reference to a type by name: a primitive, a binder, or a declared family without arguments.

    Parameters
    ----------
    name
        The type's name.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_type_name"``.
    """

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["qiec_type_name"] = "qiec_type_name"


class QiecTypeApplication(QiecTypeExpr):
    """A type constructor applied to static arguments and indices.

    Parameters
    ----------
    constructor
        The constructor's name.
    static_arguments
        The uniform type arguments, in order.
    indices
        The refinable index arguments, in order.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_type_application"``.
    """

    constructor: str
    static_arguments: tuple[QiecStaticArgument, ...] = ()
    indices: tuple[QiecIndexExpr, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["qiec_type_application"] = "qiec_type_application"


class QiecProductType(QiecTypeExpr):
    """A finite product of value types.

    Parameters
    ----------
    components
        The component types, in order.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_product_type"``.
    """

    components: tuple[QiecTypeExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["qiec_product_type"] = "qiec_product_type"


class QiecFunctionType(QiecTypeExpr):
    """A function type between value types.

    Parameters
    ----------
    parameter
        The parameter type.
    result
        The result type.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_function_type"``.
    """

    parameter: QiecTypeExpr
    result: QiecTypeExpr
    line: int = 0
    col: int = 0
    kind: Literal["qiec_function_type"] = "qiec_function_type"


class QiecEffectRef(dx.Model):
    """An applied effect interface such as ``State[Int]``."""

    name: str
    arguments: tuple[QiecStaticArgument, ...] = ()
    line: int = 0
    col: int = 0


# ---------------------------------------------------------------------------
# Effect rows and signature members
# ---------------------------------------------------------------------------


class QiecRowEntry(dx.Model):
    """A named lexical effect instance in an effect row."""

    instance: str
    line: int = 0
    col: int = 0


class QiecEffectRow(dx.Model):
    """Explicit row entries plus an optional constrained open tail."""

    entries: tuple[QiecRowEntry, ...] = ()
    tail: str | None = None
    lacks: tuple[str, ...] = ()
    line: int = 0
    col: int = 0


class QiecValueParameter(dx.Model):
    """One typed value parameter of a computation declaration.

    Parameters
    ----------
    name
        The parameter's name.
    type_expr
        The parameter's type.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    """

    name: str
    type_expr: QiecTypeExpr
    line: int = 0
    col: int = 0


class QiecFamilyConstructor(dx.Model):
    """A GADT constructor with a local telescope and refined result."""

    name: str
    binders: tuple[QiecBinder, ...] = ()
    arguments: tuple[QiecTypeExpr, ...] = ()
    result: QiecTypeExpr
    line: int = 0
    col: int = 0


class QiecOperationDecl(dx.Model):
    """One operation of an effect interface declaration.

    Parameters
    ----------
    name
        The operation's name.
    binders
        The operation's own static telescope.
    arguments
        The argument types, in order.
    result
        The result type.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    """

    name: str
    binders: tuple[QiecBinder, ...] = ()
    arguments: tuple[QiecTypeExpr, ...] = ()
    result: QiecTypeExpr
    line: int = 0
    col: int = 0


type QiecResumptionGrade = Literal["0", "aff", "1", "omega"]
"""How many times an operation clause may resume its continuation.

``"0"`` never resumes, ``"aff"`` resumes at most once, ``"1"`` resumes
exactly once, and ``"omega"`` resumes any number of times.
"""


class QiecHandlerClause(dx.TaggedUnion, discriminator="kind"):
    """One clause of a handler declaration.

    A handler answers a returning computation and each operation it
    covers, and those are different shapes: the return clause binds a
    value, an operation clause binds the operation's arguments and may
    resume.
    """


# ---------------------------------------------------------------------------
# Values and computations
# ---------------------------------------------------------------------------


QiecValue = LetExprNode
"""A QIEC value, which is the shared pure-expression tree.

Variables, literals, operators, tuples, and builtin applications are
the same nodes a ``program`` let step uses; constructor application
joins that tree, so there is one value language rather than two
isomorphic ones.
"""


class QiecConstructorValue(LetExprNode):
    """Application of a family constructor, ``construct C[..](..) as T``.

    Parameters
    ----------
    constructor
        The constructor's name.
    static_arguments
        The family parameters and constructor-local static arguments.
    fields
        The field values, in declaration order.
    result_type
        The family type the value inhabits.
    line
        The 1-based source line.
    col
        The 0-based source column.
    kind
        The discriminator; always ``"qiec_constructor_value"``.
    """

    constructor: str
    static_arguments: tuple[QiecStaticArgument, ...] = ()
    fields: tuple[QiecValue, ...] = ()
    result_type: QiecTypeExpr
    line: int = 0
    col: int = 0
    kind: Literal["qiec_constructor_value"] = "qiec_constructor_value"


class QiecLocalBinding(dx.Model):
    """A local name bound by a computation, with an optional type annotation.

    Parameters
    ----------
    name
        The local's name.
    type_expr
        The annotated type, or ``None`` when inferred.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    """

    name: str
    type_expr: QiecTypeExpr | None = None
    line: int = 0
    col: int = 0


class QiecEffectRequest(dx.Model):
    """A request of one operation on a lexical effect instance.

    Parameters
    ----------
    instance
        The instance's name.
    operation
        The operation's name.
    static_arguments
        The static arguments the operation's telescope takes.
    arguments
        The value arguments, in order.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    """

    instance: str
    operation: str
    static_arguments: tuple[QiecStaticArgument, ...] = ()
    arguments: tuple[QiecValue, ...] = ()
    line: int = 0
    col: int = 0


class QiecHandlerApplication(dx.Model):
    """A handler at its static arguments and its parameters' first values.

    Parameters
    ----------
    name
        The handler's name.
    static_arguments
        The static arguments, in order.
    values
        The values the handler's parameters start from, in order.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    """

    name: str
    static_arguments: tuple[QiecStaticArgument, ...] = ()
    values: tuple[QiecValue, ...] = ()
    line: int = 0
    col: int = 0


class QiecComputation(dx.TaggedUnion, discriminator="kind"):
    """A computation in the QIEC fine-grain call-by-value surface."""


class QiecCaseMotive(dx.Model):
    """The motive of a case analysis: the indices it abstracts and the result type they determine.

    Parameters
    ----------
    indices
        The index binders the motive abstracts over.
    result_type
        The result type under those binders.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    """

    indices: tuple[QiecIndexBinder, ...]
    result_type: QiecTypeExpr
    line: int = 0
    col: int = 0


class QiecCaseBranch(dx.Model):
    """One branch of a case analysis over an indexed family value.

    Parameters
    ----------
    constructor
        The constructor the branch matches.
    static_arguments
        The constructor-local static arguments the branch binds.
    fields
        The field binders, in declaration order.
    body
        The branch body.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    """

    constructor: str
    static_arguments: tuple[QiecStaticArgument, ...] = ()
    fields: tuple[QiecLocalBinding, ...] = ()
    body: QiecComputation
    line: int = 0
    col: int = 0


class QiecReturnComputation(QiecComputation):
    """``return VALUE``, the computation that performs nothing.

    Parameters
    ----------
    value
        The value returned.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_return_computation"``.
    """

    value: QiecValue
    line: int = 0
    col: int = 0
    kind: Literal["qiec_return_computation"] = "qiec_return_computation"


class QiecBindComputation(QiecComputation):
    """``let x <- FIRST`` followed by a computation reading ``x``.

    Parameters
    ----------
    binder
        The local the first computation's result is bound to.
    first
        The computation run first.
    then
        The computation the binding scopes over.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_bind_computation"``.
    """

    binder: QiecLocalBinding
    first: QiecComputation
    then: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_bind_computation"] = "qiec_bind_computation"


class QiecSequenceComputation(QiecComputation):
    """Two computations run in order, the first result discarded.

    Parameters
    ----------
    first
        The computation run first.
    then
        The computation run after it.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_sequence_computation"``.
    """

    first: QiecComputation
    then: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_sequence_computation"] = "qiec_sequence_computation"


class QiecPerformComputation(QiecComputation):
    """``perform`` of one effect request.

    Parameters
    ----------
    request
        The request performed.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_perform_computation"``.
    """

    request: QiecEffectRequest
    line: int = 0
    col: int = 0
    kind: Literal["qiec_perform_computation"] = "qiec_perform_computation"


class QiecHandleComputation(QiecComputation):
    """A computation run under a handler installed for one instance.

    Parameters
    ----------
    instance
        The instance the handler is installed for.
    handler
        The handler applied.
    body
        The computation run under it.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_handle_computation"``.
    """

    instance: str
    handler: QiecHandlerApplication
    body: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_handle_computation"] = "qiec_handle_computation"


class QiecCaseComputation(QiecComputation):
    """Case analysis over an indexed family value.

    Parameters
    ----------
    scrutinee
        The value analysed.
    motive
        The motive the branches are checked against.
    branches
        The branches, one per constructor covered.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_case_computation"``.
    """

    scrutinee: QiecValue
    motive: QiecCaseMotive
    branches: tuple[QiecCaseBranch, ...]
    line: int = 0
    col: int = 0
    kind: Literal["qiec_case_computation"] = "qiec_case_computation"


class QiecIfComputation(QiecComputation):
    """``if COND then ... else ...``, branching on a Boolean value.

    Parameters
    ----------
    condition
        The Boolean value branched on.
    then
        The computation run when the condition holds.
    otherwise
        The computation run when it does not.
    line
        The 1-based source line.
    col
        The 0-based source column.
    kind
        The discriminator; always ``"qiec_if_computation"``.
    """

    condition: QiecValue
    then: QiecComputation
    otherwise: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_if_computation"] = "qiec_if_computation"


class QiecPureBinding(QiecComputation):
    """`let x = VALUE`, binding a pure expression.

    Separate from
    [`QiecBindComputation`][quivers.dsl.ast_nodes.QiecBindComputation],
    which binds a computation's result. The two have different typing
    rules and neither is sugar for the other.

    Parameters
    ----------
    binder
        The local the value is bound to.
    value
        The pure expression.
    then
        The computation the binding scopes over.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"qiec_pure_binding"``.
    """

    binder: QiecLocalBinding
    value: QiecValue
    then: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_pure_binding"] = "qiec_pure_binding"


class QiecCallComputation(QiecComputation):
    """Application of a named computation.

    The callee is resolved against the module's signature table rather
    than against a local binding, so a call may precede the declaration
    it names and mutual recursion needs no forward declaration.
    """

    callee: str
    static_arguments: tuple[QiecStaticArgument, ...] = ()
    arguments: tuple[QiecValue, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["qiec_call_computation"] = "qiec_call_computation"


class QiecResumeComputation(QiecComputation):
    """Invocation of the enclosing handler clause's continuation.

    The resumption is not a named local, so there is exactly one way to
    invoke it. A `value` of None resumes with unit. The `parameters` are
    the values the handler's parameters have while the continuation runs;
    an empty tuple keeps the clause's own.
    """

    value: QiecValue | None = None
    parameters: tuple[QiecValue, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["qiec_resume_computation"] = "qiec_resume_computation"


class QiecInstanceComputation(QiecComputation):
    """`with instance x : E in`, allocating a lexically scoped instance.

    The binder is live only in `body`. Its identity is derived rather
    than generated, so the same allocation site yields the same instance
    on every run.
    """

    name: str
    effect: QiecEffectRef
    body: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_instance_computation"] = "qiec_instance_computation"


class QiecHandlerReturnClause(QiecHandlerClause):
    """What a handler does with the value its computation returns."""

    binder: QiecLocalBinding
    body: QiecComputation
    line: int = 0
    col: int = 0
    kind: Literal["qiec_handler_return_clause"] = "qiec_handler_return_clause"


class QiecHandlerOperationClause(QiecHandlerClause):
    """One operation a handler covers.

    A `body` of None is a signature rather than an implementation, and is
    legal only where the declaration says `implementation=foreign`.
    """

    operation: str
    grade: QiecResumptionGrade
    binders: tuple[QiecBinder, ...] = ()
    parameters: tuple[QiecLocalBinding, ...] = ()
    body: QiecComputation | None = None
    line: int = 0
    col: int = 0
    kind: Literal["qiec_handler_operation_clause"] = "qiec_handler_operation_clause"


# ---------------------------------------------------------------------------
# Top-level declarations
# ---------------------------------------------------------------------------


class QiecIndexDecl(Statement):
    """``index NAME = C1 | C2(...) | ...``, a closed user index sort.

    Parameters
    ----------
    name
        The sort's name.
    constructors
        The sort's constructors.
    docs
        The doc comment lines attached to the declaration.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"index_decl"``.
    """

    name: str
    constructors: tuple[QiecIndexConstructor, ...]
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["index_decl"] = "index_decl"


class QiecFamilyDecl(Statement):
    """``family NAME[params](indices) : KIND`` with its GADT constructors.

    Parameters
    ----------
    name
        The family's name.
    parameters
        The uniform static parameters.
    indices
        The refinable index binders.
    result_kind
        The kind of the family's applications.
    constructors
        The family's constructors.
    docs
        The doc comment lines attached to the declaration.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"indexed_family_decl"``.
    """

    name: str
    parameters: tuple[QiecBinder, ...] = ()
    indices: tuple[QiecIndexBinder, ...] = ()
    result_kind: QiecKindExpr
    constructors: tuple[QiecFamilyConstructor, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["indexed_family_decl"] = "indexed_family_decl"


class QiecEffectDecl(Statement):
    """``effect NAME[binders]`` declaring an operation interface.

    Parameters
    ----------
    name
        The interface's name.
    binders
        The interface's static telescope.
    operations
        The operations declared.
    docs
        The doc comment lines attached to the declaration.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"effect_decl"``.
    """

    name: str
    binders: tuple[QiecBinder, ...] = ()
    operations: tuple[QiecOperationDecl, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["effect_decl"] = "effect_decl"


class QiecEffectInstanceDecl(Statement):
    """``instance NAME : EFFECT``, a module-level lexical effect instance.

    Parameters
    ----------
    name
        The instance's name.
    effect
        The applied interface the instance carries.
    docs
        The doc comment lines attached to the declaration.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"effect_instance_decl"``.
    """

    name: str
    effect: QiecEffectRef
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["effect_instance_decl"] = "effect_instance_decl"


type QiecHandlerImplementation = Literal["authored", "foreign"]
"""Whether a handler's clause bodies are written in source.

An ``"authored"`` handler carries a body for every clause; a
``"foreign"`` handler may give clause signatures without bodies, which
the target runtime supplies.
"""


class QiecHandlerDecl(Statement):
    """``handler NAME[binders] for EFFECT : IN -> OUT with (params)``, with its clauses.

    Parameters
    ----------
    name
        The handler's name.
    binders
        The handler's static telescope.
    effect
        The applied interface the handler handles.
    input_type
        The type of the computation handled.
    output_type
        The type the handled computation yields.
    introduced
        The effect row the handler's clauses may perform.
    coverage
        Whether the clauses cover every operation (``total``) or forward the rest (``partial``).
    forwards_unknown
        Whether requests of operations without a clause are forwarded outward.
    implementation
        Whether the clause bodies are authored here or supplied by a runtime provider.
    parameters
        The handler's typed parameters, in order.
    clauses
        The return and operation clauses.
    duplicate_options
        The option names the source repeated, kept so the checker can report them.
    docs
        The doc comment lines attached to the declaration.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"handler_decl"``.
    """

    name: str
    binders: tuple[QiecBinder, ...] = ()
    effect: QiecEffectRef
    input_type: QiecTypeExpr
    output_type: QiecTypeExpr
    introduced: QiecEffectRow = QiecEffectRow()
    coverage: Literal["total", "partial"] = "total"
    forwards_unknown: bool = False
    implementation: QiecHandlerImplementation = "authored"
    parameters: tuple[QiecValueParameter, ...] = ()
    clauses: tuple[QiecHandlerClause, ...] = ()
    duplicate_options: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["handler_decl"] = "handler_decl"


class QiecComputationDecl(Statement):
    """``define NAME[binders](params) : TYPE !ROW = BODY``, a named computation.

    Parameters
    ----------
    name
        The computation's name.
    binders
        The static telescope.
    parameters
        The value parameters.
    result_type
        The declared result type.
    effects
        The declared effect row.
    body
        The body.
    docs
        The doc comment lines attached to the declaration.
    line
        The 1-based source line, or 0 when unknown.
    col
        The 0-based source column, or 0 when unknown.
    kind
        The discriminator; always ``"computation_decl"``.
    """

    name: str
    binders: tuple[QiecBinder, ...] = ()
    parameters: tuple[QiecValueParameter, ...] = ()
    result_type: QiecTypeExpr
    effects: QiecEffectRow
    body: QiecComputation
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["computation_decl"] = "computation_decl"


class QiecInterpretationClause(dx.Model):
    """A computation assigned to one rule or axiom slot.

    Parameters
    ----------
    constructor_kind
        Whether the constructor is a rule or an input axiom slot.
    constructor
        The constructor's name in its deduction.
    operation
        The name of the checked computation that interprets it.
    line, col
        The source position.
    """

    constructor_kind: Literal["rule", "axiom"]
    constructor: str
    operation: str
    line: int = 0
    col: int = 0


class QiecInterpretationDecl(Statement):
    """A named computation that folds a deduction's typed proof.

    Parameters
    ----------
    name
        The computation's name.
    deduction
        The deduction whose proof family it consumes.
    carrier
        The closed result type of the fold.
    clauses
        One computation assignment per rule or axiom slot.
    docs
        The attached doc comments.
    line, col
        The source position.
    kind
        The syntax discriminator.
    """

    name: str
    deduction: str
    carrier: QiecTypeExpr
    clauses: tuple[QiecInterpretationClause, ...]
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["interpretation_decl"] = "interpretation_decl"


__all__ = [
    "QiecKindExpr",
    "QiecTypeKind",
    "QiecEffectKind",
    "QiecIndexExpr",
    "QiecIndexName",
    "QiecIndexLiteral",
    "QiecIndexApplication",
    "QiecShapeIndex",
    "QiecIndexSort",
    "QiecNatSort",
    "QiecShapeSort",
    "QiecContextSort",
    "QiecUserIndexSort",
    "QiecBinder",
    "QiecTypeBinder",
    "QiecIndexBinder",
    "QiecEffectBinder",
    "QiecIndexConstructor",
    "QiecTypeExpr",
    "QiecStaticArgument",
    "QiecTypeName",
    "QiecTypeApplication",
    "QiecProductType",
    "QiecFunctionType",
    "QiecEffectRef",
    "QiecRowEntry",
    "QiecEffectRow",
    "QiecValueParameter",
    "QiecFamilyConstructor",
    "QiecOperationDecl",
    "QiecResumptionGrade",
    "QiecHandlerClause",
    "QiecValue",
    "QiecConstructorValue",
    "QiecLocalBinding",
    "QiecEffectRequest",
    "QiecHandlerApplication",
    "QiecComputation",
    "QiecCaseMotive",
    "QiecCaseBranch",
    "QiecReturnComputation",
    "QiecBindComputation",
    "QiecSequenceComputation",
    "QiecPerformComputation",
    "QiecHandleComputation",
    "QiecCaseComputation",
    "QiecIfComputation",
    "QiecPureBinding",
    "QiecCallComputation",
    "QiecResumeComputation",
    "QiecInstanceComputation",
    "QiecHandlerReturnClause",
    "QiecHandlerOperationClause",
    "QiecIndexDecl",
    "QiecFamilyDecl",
    "QiecEffectDecl",
    "QiecEffectInstanceDecl",
    "QiecHandlerImplementation",
    "QiecHandlerDecl",
    "QiecComputationDecl",
    "QiecInterpretationClause",
    "QiecInterpretationDecl",
]
