"""Exact QVR to QIEC lowering.

Quivers owns this adapter. Didactic orders and negotiates its stages and checks
the first-order indexed-family projection. Name resolution, stable identity,
effect elaboration, runtime checking, and the QIEC ABI remain source-language
responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from didactic.extensions import LoweringRoute, lower_checked
from didactic.gadt import (
    GADT,
    Family as DidacticFamily,
    GADTDeclarationError,
    Operation as DidacticOperation,
    SortExpr as DidacticSortExpr,
    Term as DidacticTerm,
    Var as DidacticVar,
)
from panproto import GatError

from quivers.dsl import ast_nodes as surface
from quivers.qiec.canonical import (
    BUILTIN_TYPE_CONSTRUCTORS,
    LOG_WEIGHT,
    SITE_CONSTRUCTOR,
    sampleable_type,
    sampled_element,
    tensor_shape,
    tensor_type,
)
from quivers.qiec.builtins import BUILTIN_EFFECTS
from quivers.qiec.families import FAMILIES, DistributionFamily
from quivers.qiec.programs import ProgramEntry
from quivers.qiec.effects import render_row
from quivers.qiec.types import render_static
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.transpile._pure_calls import (
    replace_locals as _replace_locals,
)
from quivers.dsl.let_expr_traversal import let_expr_children, substitute_let_expr
from quivers.dsl.pure_builtins import (
    HOST_ONLY_BUILTINS,
    _BINARY_PRIMITIVES,
    _BUILTIN_PRIMITIVES,
    _COLLECTION_BUILTINS,
    _REDUCTIONS,
    _ROWWISE,
)
from quivers.dsl.deduction_elaboration import _DeductionElaboration
from quivers.dsl.structural_elaboration import _StructuralElaboration
from quivers.dsl.latent_weights import LatentWeightsError, expand_latent_weights
from quivers.dsl.program_elaboration import (
    LATENT_WEIGHT_HEADS,
    ObjectInfo,
    _extent,
    _fixed_extent,
    _object_expr_info,
    _ProgramElaboration,
    _ProgramState,
)
from quivers.qiec import (
    ComputationSignature,
    ResumptionType,
    DistributionValue,
    LogDensity,
    SiteValue,
    If,
    PrimitiveApplication,
    Projection,
    TensorValue,
    TupleValue,
    Value,
    Gather,
    Reduction,
    ReductionOperator,
    Rowwise,
    Comprehension,
    LetBinding,
    LetValue,
    primitive,
    substitute_row,
    Call,
    NewInstance,
    Resume,
    HandlerReturnClauseDef,
    ComputationId,
    BOOL,
    EFFECT,
    INT,
    NAT,
    QIEC_ABI,
    REAL,
    STRING,
    TYPE,
    UNIT,
    ArgumentDef,
    Bind,
    Case,
    CaseBranch,
    CaseMotive,
    CheckContext,
    Computation,
    ComputationType,
    computation_type_conforms,
    ConstructorDecl,
    ConstructorId,
    ConstructorValue,
    ContextSort,
    EffectBinder,
    EffectDef,
    EffectId,
    EffectInstanceId,
    EffectRef,
    EffectRequest,
    EffectRow,
    EffectVariable,
    EqualityType,
    FamilyDecl,
    FamilyId,
    FieldDef,
    FunctionType,
    Handle,
    HandlerClauseDef,
    HandlerDef,
    HandlerId,
    IndexBinder,
    IndexConstructor,
    IndexLiteral,
    IndexTerm,
    IndexSort,
    IndexVariable,
    Kind,
    KernelError,
    KernelRegistry,
    LiteralValue,
    Local,
    NamedComputation,
    NamedEffectInstance,
    NatSort,
    OperationDef,
    OperationId,
    Perform,
    QiecModule,
    ResumptionGrade,
    Return,
    RowEntry,
    RowVariable,
    RowVariableId,
    ShapeIndex,
    ShapeSort,
    SiteProvenance,
    SourceOrigin,
    StaticArgument,
    StaticScopeId,
    StaticSubstitution,
    Telescope,
    TelescopeBinder,
    TypeApplication,
    TypeConstructorRef,
    TypeBinder,
    TypeExpr,
    TypeVariable,
    UserIndexSort,
    Var,
    constructor_skolems,
    dumps,
    infer_computation,
    infer_value,
    index_sort,
    instantiate_effect,
    instantiate_operation,
    instantiate_telescope,
    loads,
    product_type,
    substitute_type,
    validate_module,
)


QVR_SOURCE_VERSION = "qvr-source/v0.20"
"""The source protocol a `QvrQiecLowerer` route accepts.

`lower_qvr_to_qiec` checks its ``source_version`` argument against this
identifier, which names the QVR surface the elaboration reads.
"""

QIEC_STATEMENT_TYPES = (
    surface.QiecIndexDecl,
    surface.QiecFamilyDecl,
    surface.QiecEffectDecl,
    surface.QiecEffectInstanceDecl,
    surface.QiecHandlerDecl,
    surface.QiecComputationDecl,
)
"""The declaration classes that make up the QIEC surface of a module.

`qiec_projection` keeps these statements and `non_qiec_projection`
drops them, so the tuple fixes where the two routes divide a module.
"""

#: Declarations the program elaboration reads beside the QIEC statements:
#: the programs and deductions themselves and the objects, morphisms, and
#: let aliases their steps refer to. They stay in the ordinary compiler's
#: projection as well, since it builds the runtime program from them.
PROGRAM_CONTEXT_TYPES = (
    surface.ProgramDecl,
    surface.DeductionDecl,
    surface.ObjectDecl,
    surface.MorphismDecl,
    surface.DefineDecl,
    surface.SchemaDecl,
    surface.BundleDecl,
    surface.SignatureDecl,
    surface.EncoderDecl,
    surface.DecoderDecl,
    surface.LossDecl,
)


@dataclass(frozen=True, slots=True)
class QvrQiecSource:
    """A parsed QVR projection plus the identity absent from the AST root.

    Parameters
    ----------
    syntax
        The projected module.
    module_name
        The name every stable identity derives from.
    file_path
        The path recorded on diagnostics.
    elaborate_programs
        Whether ``program`` declarations become computations. A caller
        that has found a program using a construct the elaboration does
        not yet cover lowers the rest of the module without it.
    """

    syntax: surface.Module
    module_name: str
    file_path: str = "<source>"
    elaborate_programs: bool = True


@dataclass(frozen=True, slots=True)
class CheckedQvrQiec:
    """The checked result passed between Didactic's ordered stages.

    Parameters
    ----------
    module : QiecModule
        The kernel module the stages produced.
    """

    module: QiecModule


def has_qiec_surface(module: surface.Module) -> bool:
    """Return whether a parsed module has anything to elaborate through QIEC.

    Parameters
    ----------
    module : surface.Module
        The parsed module to inspect.

    Returns
    -------
    bool
        True when at least one statement belongs to the QIEC surface or
        is a program or deduction the elaboration turns into
        computations. Callers
        use this to decide whether the QIEC route runs at all, so a
        module of other declarations pays nothing for it.
    """

    return any(
        isinstance(
            statement,
            (
                *QIEC_STATEMENT_TYPES,
                surface.DeductionDecl,
                surface.SignatureDecl,
                surface.EncoderDecl,
                surface.DecoderDecl,
                surface.LossDecl,
            ),
        )
        or (
            isinstance(statement, surface.DefineDecl)
            and isinstance(statement.expr, surface.ExprParser)
        )
        or (
            isinstance(statement, surface.ProgramDecl)
            and (
                statement.type_params is None
                or all(
                    isinstance(parameter, surface.ScalarParam)
                    for parameter in statement.type_params
                )
            )
        )
        for statement in module.statements
    )


def qiec_projection(module: surface.Module) -> surface.Module:
    """Project QIEC declarations from a module with multiple neighborhoods.

    Parameters
    ----------
    module : surface.Module
        The parsed module to project.

    Returns
    -------
    surface.Module
        A module of the QIEC statements alone, in source order. The
        complement is `non_qiec_projection`, and the two partition the
        input, so nothing is dropped between the two routes.
    """

    return surface.Module(
        statements=tuple(
            statement
            for statement in module.statements
            if isinstance(statement, (*QIEC_STATEMENT_TYPES, *PROGRAM_CONTEXT_TYPES))
        )
    )


def non_qiec_projection(module: surface.Module) -> surface.Module:
    """Project declarations consumed by the categorical/probabilistic compiler.

    Parameters
    ----------
    module : surface.Module
        The parsed module to project.

    Returns
    -------
    surface.Module
        A module of everything outside the QIEC surface, in source
        order.
    """

    return surface.Module(
        statements=tuple(
            statement
            for statement in module.statements
            if not isinstance(statement, QIEC_STATEMENT_TYPES)
        )
    )


#: The prelude's effect interfaces, resolvable from any module by name.
_PRELUDE_EFFECTS: Mapping[str, EffectDef] = {
    effect.ref.name: effect for effect in BUILTIN_EFFECTS
}


def _substitute_let(
    expr: surface.LetExprNode, substitution: Mapping[str, surface.LetExprNode]
) -> surface.LetExprNode:
    """Replace variables in a let expression.

    Parameters
    ----------
    expr : surface.LetExprNode
        The expression.
    substitution : Mapping[str, surface.LetExprNode]
        Variable names to the expressions replacing them.

    Returns
    -------
    surface.LetExprNode
        The expression with every free occurrence replaced; a lambda or
        factor binder shadows the substitution inside its body.
    """
    return substitute_let_expr(expr, substitution)


#: The let expressions a substitution may copy freely: reading one costs
#: nothing, so duplicating it neither grows a term nor repeats work.
_ATOMIC_LET_EXPRESSIONS = (
    surface.LetExprVar,
    surface.LetExprLiteral,
    surface.LetExprBool,
    surface.LetExprUnit,
    surface.LetExprString,
)


def _atomic(expr: surface.LetExprNode) -> bool:
    """Whether substituting ``expr`` for a variable can copy it freely.

    Parameters
    ----------
    expr : surface.LetExprNode
        The expression a binder would be replaced by.

    Returns
    -------
    bool
        ``True`` for a variable or a scalar literal.
    """
    return isinstance(expr, _ATOMIC_LET_EXPRESSIONS)


def _reads_repeatedly(name: str, body: surface.LetExprNode) -> bool:
    """Whether ``body`` would evaluate a value substituted for ``name`` twice.

    A substituted expression is copied once per free occurrence, and an
    occurrence under a lambda or ``factor`` binder sits in a body that
    is evaluated once per item. Either way a substitution would repeat
    the expression's work, so the value must be bound once and shared.

    Parameters
    ----------
    name : str
        The variable a value would replace.
    body : surface.LetExprNode
        The expression ``name`` is free in.

    Returns
    -------
    bool
        ``True`` if ``name`` occurs free more than once, or once under
        a lambda or ``factor`` binder.
    """
    reads = 0

    def visit(node: surface.LetExprNode, nested: bool) -> bool:
        """Whether the name is read more than once, or under a binder.

        Parameters
        ----------
        node : surface.LetExprNode
            The expression.
        nested : bool
            Whether the expression sits under a binder.

        Returns
        -------
        bool
            Whether a read found so far needs the value shared.
        """
        nonlocal reads
        if isinstance(node, surface.LetExprVar):
            if node.name == name:
                reads += 1
                return nested or reads > 1
            return False
        if isinstance(node, surface.LetExprLambda):
            return node.param != name and visit(node.body, True)
        if isinstance(node, surface.LetExprFactor):
            if any(binder.var == name for binder in node.binders):
                return False
            return any(visit(child, True) for child in let_expr_children(node))
        return any(visit(child, nested) for child in let_expr_children(node))

    return visit(body, False)


_EXPRESSION_FORMS = {
    "LetExprList": "a list literal",
    "LetExprLambda": "a lambda",
    "LetExprFactor": "a `factor` expression",
    "LetExprMethodCall": "a method call",
}


class QvrQiecLowerer:
    """Quivers-owned implementation of Didactic's extension-lowering protocol."""

    route = LoweringRoute(QVR_SOURCE_VERSION, QIEC_ABI)

    def check(self, source: QvrQiecSource, /) -> CheckedQvrQiec:
        """Elaborate a source module and recheck the result.

        The recheck is deliberate rather than redundant: it rebuilds a
        registry from the lowered module alone, so a bug in elaboration
        surfaces here instead of reaching a backend.

        Parameters
        ----------
        source : QvrQiecSource
            The QIEC projection, its module name, and its file path.

        Returns
        -------
        CheckedQvrQiec
            The checked module.

        Raises
        ------
        QiecDiagnosticError
            If elaboration rejects the source, carrying a stable code and
            a source position.
        KernelError
            If the lowered module fails its independent recheck.
        """
        module = _Elaborator(source).elaborate()
        validate_module(module)
        return CheckedQvrQiec(module)

    def lower(self, checked: CheckedQvrQiec, /) -> QiecModule:
        """Return the checked module.

        Checking and lowering are one step here, because the elaborator
        produces the kernel module directly, so this unwraps rather than
        transforms.

        Parameters
        ----------
        checked : CheckedQvrQiec
            The result of `check`.

        Returns
        -------
        QiecModule
            The lowered module.
        """
        return checked.module

    def validate(self, target: QiecModule, /) -> None:
        """Recheck a module and confirm it survives serialization.

        Parameters
        ----------
        target : QiecModule
            The module to validate.

        Raises
        ------
        KernelError
            If the module fails its recheck, or if a dump and load round
            trip does not return an equal module. The second is checked
            because identities downstream are content addressed: a module
            that changed under serialization would hash differently on
            the other side of the boundary.
        """
        validate_module(target)
        decoded = loads(dumps(target))
        if decoded != target:
            raise KernelError("QIEC module changed during canonical serialization")


def _last_axis_reduction(
    operator: ReductionOperator,
    value: Value,
    shape: tuple[TypeExpr, tuple[IndexTerm, ...]],
    path: tuple[str | int, ...],
    occupied: frozenset[str],
) -> Value:
    """Reduce a tensor along its last axis.

    A reduction builtin summarizes each row of its argument, as the
    torch runtime reduces along ``dim=-1``: a vector reduces to one
    number, and a tensor of higher rank to a tensor of the leading
    axes, each entry the reduction of one row.

    Parameters
    ----------
    operator : ReductionOperator
        The reduction.
    value : Value
        The tensor reduced.
    shape : tuple[TypeExpr, tuple[IndexTerm, ...]]
        The tensor's element type and dimensions.
    path : tuple[str | int, ...]
        The structural path of the reduction, positions included, which
        names the comprehension binders: a reduction nested in another's
        argument sits at a longer path, so the two never share a binder.
    occupied : frozenset[str]
        The names of the locals in scope, which no binder repeats.

    Returns
    -------
    Value
        The reduction, under one comprehension per leading axis.
    """
    element, dimensions = shape
    if len(dimensions) <= 1:
        return Reduction(operator, value, element)
    stem = "_".join(str(item) for item in path)
    name = f"__{stem}_row{len(dimensions)}"
    suffix = 0
    while name in occupied:
        suffix += 1
        name = f"__{stem}_row{len(dimensions)}_{suffix}"
    binder = Local(name, INT)
    row = Gather(value, Var(binder), tensor_type(element, dimensions[1:]))
    inner = _last_axis_reduction(
        operator, row, (element, dimensions[1:]), path, occupied | {name}
    )
    inner_shape = tensor_shape(_reduced_type(element, dimensions[1:]))
    result = (
        tensor_type(element, (dimensions[0],))
        if inner_shape is None
        else tensor_type(inner_shape[0], (dimensions[0], *inner_shape[1]))
    )
    return Comprehension(binder, dimensions[0], inner, result)


class _PointwiseGatherError(ValueError):
    """A subscript with index tensors that does not fit the tensor it reads."""


def _pointwise_selection(
    source: Value,
    source_type: TypeExpr,
    indices: tuple[Value, ...],
    index_types: tuple[TypeExpr, ...],
    path: tuple[str | int, ...],
    occupied: frozenset[str],
) -> Value:
    """Select one entry per position of a subscript's index tensors.

    ``t[e_1, ..., e_k]`` with two or more indices, at least one of them
    a ``Tensor[Int]``, reads the indices together, as advanced indexing
    does in NumPy and ``t[r[n], c[n]]`` does in a Stan loop: every index
    tensor has one shape ``S``, an ``Int`` index stands for the same
    value at every position, and the result has shape ``S`` followed by
    the axes of ``t`` after the ``k`` indexed ones, its entry at a
    position ``n`` being ``t[e_1[n], ..., e_k[n]]``. The selection is a
    comprehension over each axis of ``S`` whose body gathers ``t`` one
    axis at a time.

    Parameters
    ----------
    source : Value
        The tensor selected from.
    source_type : TypeExpr
        Its type.
    indices : tuple[Value, ...]
        The indices, outermost axis first.
    index_types : tuple[TypeExpr, ...]
        Their types: ``Int`` or ``Tensor[Int]``.
    path : tuple[str | int, ...]
        Structural path of the subscript, which names the binders.
    occupied : frozenset[str]
        The names of the locals in scope, which no binder repeats.

    Returns
    -------
    Value
        The selection.

    Raises
    ------
    _PointwiseGatherError
        If the tensor has fewer axes than indices, an index is neither
        an ``Int`` nor a ``Tensor[Int]``, or the index tensors differ in
        shape.
    """
    split = tensor_shape(source_type)
    if split is None or len(split[1]) < len(indices):
        raise _PointwiseGatherError(
            f"a subscript of {len(indices)} indices selects from a value with "
            f"fewer axes"
        )
    element, dimensions = split
    shape: tuple[IndexTerm, ...] | None = None
    for position, index_type in enumerate(index_types):
        if index_type == INT:
            continue
        index_shape = tensor_shape(index_type)
        if index_shape is None or index_shape[0] != INT:
            raise _PointwiseGatherError(
                f"index {position} of the subscript is neither an Int nor a Tensor[Int]"
            )
        if shape is None:
            shape = index_shape[1]
        elif index_shape[1] != shape:
            raise _PointwiseGatherError(
                "the index tensors of one subscript select one entry per "
                "position, so they share one shape; index "
                f"{position} has another shape than the first index tensor"
            )
    assert shape is not None
    stem = "_".join(str(item) for item in path)
    binders: list[Local] = []
    taken = set(occupied)
    for axis in range(len(shape)):
        name = f"__{stem}_pick{axis}"
        suffix = 0
        while name in taken:
            suffix += 1
            name = f"__{stem}_pick{axis}_{suffix}"
        taken.add(name)
        binders.append(Local(name, INT))
    entries: list[Value] = []
    for index, index_type in zip(indices, index_types, strict=True):
        if index_type == INT:
            entries.append(index)
            continue
        entry = index
        for axis, binder in enumerate(binders):
            remaining = shape[axis + 1 :]
            entry = Gather(
                entry,
                Var(binder),
                INT if not remaining else tensor_type(INT, remaining),
            )
        entries.append(entry)
    body: Value = source
    for axis, entry in enumerate(entries):
        rest = dimensions[axis + 1 :]
        body = Gather(body, entry, element if not rest else tensor_type(element, rest))
    rest = dimensions[len(indices) :]
    for axis in reversed(range(len(shape))):
        outer = (*shape[axis:], *rest)
        body = Comprehension(
            binders[axis], shape[axis], body, tensor_type(element, outer)
        )
    return body


def _reduced_type(element: TypeExpr, dimensions: tuple[IndexTerm, ...]) -> TypeExpr:
    """The type a last-axis reduction of a tensor has.

    Parameters
    ----------
    element : TypeExpr
        The tensor's element type.
    dimensions : tuple[IndexTerm, ...]
        The tensor's dimensions.

    Returns
    -------
    TypeExpr
        The element for a vector, else the tensor of the leading axes.
    """
    if len(dimensions) <= 1:
        return element
    return tensor_type(element, dimensions[:-1])


def lower_qvr_to_qiec(
    module: surface.Module,
    *,
    module_name: str | None = None,
    file_path: str = "<source>",
    source_version: str = QVR_SOURCE_VERSION,
    target_version: str = QIEC_ABI,
    elaborate_programs: bool = True,
) -> QiecModule:
    """Check and lower the QIEC projection of one parsed QVR module.

    Parameters
    ----------
    module : surface.Module
        The parsed module, which may mix QIEC and ordinary declarations.
    module_name : str or None
        Name the stable identities derive from. None takes it from
        `file_path`, so two files of the same stem produce the same
        identities and a rename is a deliberate act.
    file_path : str
        Path recorded on diagnostics.
    source_version : str
        The source protocol the module is written against.
    target_version : str
        The kernel ABI to lower to.
    elaborate_programs : bool
        Whether ``program`` declarations become computations.

    Returns
    -------
    QiecModule
        The checked, lowered module.

    Raises
    ------
    QiecDiagnosticError
        If the source is rejected, with a stable code and a position. The
        code ``qiec-program-gap`` names a program construct, such as a
        parsing chart or a network-parameterized morphism, whose
        elaboration is not yet defined; a caller may lower the module
        again without its programs.
    KernelError
        If the lowered module fails its independent recheck.
    """

    if module_name is None:
        module_name = _module_name(file_path)
    try:
        module = expand_latent_weights(module, LATENT_WEIGHT_HEADS)
    except LatentWeightsError as error:
        raise QiecDiagnosticError(
            str(error),
            code="qiec-program",
            file=file_path,
            line=error.line,
            column=error.col,
        ) from error
    source = QvrQiecSource(
        qiec_projection(module), module_name, file_path, elaborate_programs
    )
    return lower_checked(
        source,
        QvrQiecLowerer(),
        source_version=source_version,
        target_version=target_version,
    )


def _module_name(file_path: str) -> str:
    """Derive a module name from a source path.

    Parameters
    ----------
    file_path : str
        The path, or the placeholder used for input with no file.

    Returns
    -------
    str
        The path's stem, or ``"source"`` when there is no usable one.
        The name enters every derived identity, so it depends on the path
        alone and not on how the path was spelled.
    """
    if file_path == "<source>":
        return "source"
    return Path(file_path).stem or "source"


def _didactic_name(category: str, identity: object) -> str:
    """Return an injective identifier for Didactic's global GAT namespace.

    Parameters
    ----------
    category : str
        What kind of declaration is being named.
    identity : object
        Its stable identity.

    Returns
    -------
    str
        A name unique to that category and identity. The identity is hex
        encoded rather than interpolated, so two identities differing
        only by a character the namespace treats specially cannot collide.
    """

    encoded = str(identity).encode().hex()
    return f"qiec_{category}_{encoded}"


class _DidacticGadtProjection:
    """Project QIEC indexed declarations into Didactic's checked GADT API.

    Didactic and Panproto use a first-order GAT signature, while QIEC admits
    static parameters of kind ``Type``.  The projection therefore uses the
    standard Tarski encoding: type parameters inhabit a code sort and values
    of a coded type inhabit an indexed carrier.  QIEC indexed families remain
    genuine Didactic families, so constructor result refinements and dependent
    field telescopes are checked without flattening their indices.

    Parameters
    ----------
    module_name : str
        The module the declarations come from; it prefixes Didactic names.
    index_sorts : tuple[UserIndexSort, ...]
        The module's user index sorts.
    families : tuple[FamilyDecl, ...]
        The indexed families to project.
    constructors : tuple[ConstructorDecl, ...]
        The constructors of those families.
    """

    def __init__(
        self,
        module_name: str,
        index_sorts: tuple[UserIndexSort, ...],
        families: tuple[FamilyDecl, ...],
        constructors: tuple[ConstructorDecl, ...],
    ) -> None:
        self.language = GADT(_didactic_name("module", module_name))
        self.type_codes = self.language.sort(_didactic_name("sort", "Type"))
        self.effect_codes = self.language.sort(_didactic_name("sort", "Effect"))
        self.values = self.language.family(
            _didactic_name("family", "El"),
            **{_didactic_name("parameter", "code"): self.type_codes()},
        )
        self._sorts: dict[IndexSort, DidacticFamily] = {}
        self._families: dict[FamilyId, DidacticFamily] = {}
        self._qiec_families = {family.id: family for family in families}
        self._family_by_type_id = {
            family.type_constructor.id: family for family in families
        }
        self._index_constructors: dict[
            tuple[UserIndexSort, str], DidacticOperation
        ] = {}
        self._code_operations: dict[tuple[str, str], DidacticOperation] = {}

        for sort in index_sorts:
            self._sort_for(sort)
        for family in families:
            self._declare_family(family)
        for constructor in constructors:
            self._declare_constructor(constructor)

    def compile(self) -> object:
        """Compile and check the indexed signature with Panproto.

        Returns
        -------
        object
            The compiled signature. Panproto checks the projection here,
            so a family or constructor this route cannot express is
            rejected before the QIEC checker sees it.
        """

        return self.language.compile()

    def _declare_family(self, family: FamilyDecl) -> None:
        """Project one indexed family into the Didactic signature.

        Parameters
        ----------
        family : FamilyDecl
            The family to project.

        Raises
        ------
        GADTDeclarationError
            If a binder uses a kind the projection cannot express.
        """
        parameters = {
            _didactic_name(
                "family_parameter", f"{position}:{binder.name}"
            ): self._binder_sort(binder)
            for position, binder in enumerate((*family.parameters, *family.indices))
        }
        self._families[family.id] = self.language.family(
            _didactic_name("indexed_family", family.id),
            closed=family.closed,
            **parameters,
        )

    def _declare_constructor(self, constructor: ConstructorDecl) -> None:
        """Project one constructor into the Didactic signature.

        Parameters
        ----------
        constructor : ConstructorDecl
            The constructor to project.

        Raises
        ------
        GADTDeclarationError
            If a field type or result index cannot be expressed.
        """
        family = self._families[constructor.family]
        qiec_family = self._qiec_families[constructor.family]
        scope: dict[str, DidacticTerm] = {}
        inputs: dict[str, DidacticSortExpr] = {}
        for position, binder in enumerate(qiec_family.parameters):
            name = _didactic_name("constructor_parameter", f"{position}:{binder.name}")
            inputs[name] = self._binder_sort(binder)
            scope[binder.name] = DidacticVar(name)
        for position, binder in enumerate(constructor.telescope):
            name = _didactic_name("constructor_static", f"{position}:{binder.name}")
            inputs[name] = self._binder_sort(binder)
            scope[binder.name] = DidacticVar(name)
        for position, field in enumerate(constructor.fields):
            inputs[_didactic_name("constructor_field", position)] = self._value_sort(
                field.type, scope
            )
        result_arguments = (
            *(scope[binder.name] for binder in qiec_family.parameters),
            *(self._static_term(index, scope) for index in constructor.result_indices),
        )
        self.language.constructor(
            _didactic_name("constructor", constructor.id),
            returns=family(*result_arguments),
            **inputs,
        )

    def _binder_sort(self, binder: TelescopeBinder) -> DidacticSortExpr:
        """The Didactic sort a telescope binder ranges over.

        Parameters
        ----------
        binder : TelescopeBinder
            The binder to classify.

        Returns
        -------
        DidacticSortExpr
            The sort expression for the binder's namespace and kind.

        Raises
        ------
        GADTDeclarationError
            If the binder's kind has no projection.
        """
        if isinstance(binder, TypeBinder):
            if binder.kind == TYPE:
                return self.type_codes()
            if binder.kind == EFFECT:
                return self.effect_codes()
            raise GADTDeclarationError(
                f"unsupported QIEC binder kind in GADT projection: {binder.kind!r}"
            )
        if isinstance(binder, EffectBinder):
            return self.effect_codes()
        return self._sort_for(binder.sort)()

    def _sort_for(self, sort: IndexSort) -> DidacticFamily:
        """The Didactic family standing for one index sort, memoised.

        Parameters
        ----------
        sort : IndexSort
            The index sort to project.

        Returns
        -------
        DidacticFamily
            The family, created on first request and reused after, so one sort projects to one family.
        """
        existing = self._sorts.get(sort)
        if existing is not None:
            return existing
        family = self.language.sort(
            _didactic_name("index_sort", sort),
            closed=isinstance(sort, UserIndexSort),
        )
        self._sorts[sort] = family
        if isinstance(sort, UserIndexSort):
            for constructor_name, arity in zip(
                sort.constructors, sort.arities, strict=True
            ):
                operation = self.language.constructor(
                    _didactic_name(
                        "index_constructor", f"{sort.name}:{constructor_name}"
                    ),
                    returns=family(),
                    **{
                        _didactic_name("index_argument", position): family()
                        for position in range(arity)
                    },
                )
                self._index_constructors[(sort, constructor_name)] = operation
        return family

    def _value_sort(
        self, type_: TypeExpr, scope: Mapping[str, DidacticTerm]
    ) -> DidacticSortExpr:
        """The Didactic sort of the values one type ranges over.

        Parameters
        ----------
        type_ : TypeExpr
            The type whose value space is wanted.
        scope : Mapping[str, DidacticTerm]
            Terms already projected for the binders in scope.

        Returns
        -------
        DidacticSortExpr
            The sort expression for those values.
        """
        if isinstance(type_, TypeApplication):
            family = self._family_by_type_id.get(type_.constructor.id)
            if family is not None:
                target = self._families[family.id]
                return target(
                    *(
                        self._static_term(argument, scope)
                        for argument in type_.arguments
                    )
                )
        return self.values(self._type_term(type_, scope))

    def _type_term(
        self, type_: TypeExpr, scope: Mapping[str, DidacticTerm]
    ) -> DidacticTerm:
        """Project a QIEC type into a Didactic term.

        Parameters
        ----------
        type_ : TypeExpr
            The type to project.

        Returns
        -------
        DidacticTerm
            The projected term.

        Raises
        ------
        GADTDeclarationError
            If the type uses a construct the projection cannot express.
        """
        if isinstance(type_, TypeVariable):
            return scope[type_.name]
        if isinstance(type_, TypeApplication):
            operation = self._code_operation(
                "type_constructor",
                type_.constructor.id,
                tuple(
                    self._binder_sort(binder) for binder in type_.constructor.telescope
                ),
                self.type_codes(),
            )
            return operation(
                *(self._static_term(argument, scope) for argument in type_.arguments)
            )
        if isinstance(type_, FunctionType):
            operation = self._code_operation(
                "function_type",
                "function",
                (self.type_codes(), self.type_codes()),
                self.type_codes(),
            )
            return operation(
                self._type_term(type_.parameter, scope),
                self._type_term(type_.result, scope),
            )
        if isinstance(type_, EqualityType):
            operand_sort = self._static_sort(type_.kind)
            operation = self._code_operation(
                "equality_type",
                type_.kind,
                (operand_sort, operand_sort),
                self.type_codes(),
            )
            return operation(
                self._static_term(type_.left, scope),
                self._static_term(type_.right, scope),
            )
        raise GADTDeclarationError(f"unsupported QIEC type projection: {type_!r}")

    def _static_sort(self, kind: Kind | IndexSort) -> DidacticSortExpr:
        """The Didactic sort of a static argument.

        Parameters
        ----------
        kind : Kind
            The kind to classify.

        Returns
        -------
        DidacticSortExpr
            The sort expression for that kind's namespace.

        Raises
        ------
        GADTDeclarationError
            If the argument is of a class with no projection.
        """
        if kind == TYPE:
            return self.type_codes()
        if kind == EFFECT:
            return self.effect_codes()
        if isinstance(kind, (NatSort, ShapeSort, ContextSort, UserIndexSort)):
            return self._sort_for(kind)()
        raise GADTDeclarationError(f"unsupported QIEC static sort: {kind!r}")

    def _static_term(
        self, term: StaticArgument, scope: Mapping[str, DidacticTerm]
    ) -> DidacticTerm:
        """Project a static argument into a Didactic term.

        Parameters
        ----------
        term : StaticArgument
            The static argument to project.

        Returns
        -------
        DidacticTerm
            The projected term.

        Raises
        ------
        GADTDeclarationError
            If the argument cannot be expressed.
        """
        if isinstance(
            term, (TypeVariable, TypeApplication, FunctionType, EqualityType)
        ):
            return self._type_term(term, scope)
        if isinstance(term, IndexVariable):
            return scope[term.name]
        if isinstance(term, IndexLiteral):
            if isinstance(term.sort, UserIndexSort):
                return self._index_constructors[(term.sort, str(term.value))]()
            operation = self._code_operation(
                "index_literal",
                f"{term.sort}:{term.value}",
                (),
                self._sort_for(term.sort)(),
            )
            return operation()
        if isinstance(term, IndexConstructor):
            constructor_sort = cast(UserIndexSort, term.sort)
            operation = self._index_constructors[(constructor_sort, term.name)]
            return operation(
                *(self._static_term(argument, scope) for argument in term.arguments)
            )
        if isinstance(term, ShapeIndex):
            sort = index_sort(term)
            operation = self._code_operation(
                "shape_index",
                len(term.dimensions),
                tuple(self._sort_for(NAT)() for _ in term.dimensions),
                self._sort_for(sort)(),
            )
            return operation(
                *(self._static_term(dimension, scope) for dimension in term.dimensions)
            )
        if isinstance(term, EffectVariable):
            return scope[term.name]
        if isinstance(term, EffectRef):
            operation = self._code_operation(
                "effect_interface",
                term.id,
                tuple(self._term_sort(argument) for argument in term.arguments),
                self.effect_codes(),
            )
            return operation(
                *(self._static_term(argument, scope) for argument in term.arguments)
            )
        raise GADTDeclarationError(f"unsupported QIEC static projection: {term!r}")

    def _term_sort(self, term: StaticArgument) -> DidacticSortExpr:
        """The Didactic sort of an index term.

        Parameters
        ----------
        term : IndexTerm
            The index term to classify.

        Returns
        -------
        DidacticSortExpr
            The sort expression for the term's own sort.
        """
        if isinstance(
            term, (TypeVariable, TypeApplication, FunctionType, EqualityType)
        ):
            return self.type_codes()
        if isinstance(term, (EffectVariable, EffectRef)):
            return self.effect_codes()
        return self._sort_for(index_sort(term))()

    def _code_operation(
        self,
        category: str,
        identity: object,
        inputs: tuple[DidacticSortExpr, ...],
        output: DidacticSortExpr,
    ) -> DidacticOperation:
        """Declare or reuse a Didactic operation encoding one QIEC construct.

        Parameters
        ----------
        category : str
            What kind of construct is being encoded, which enters the
            operation's derived name.
        identity : object
            The construct's stable identity, which also enters the name,
            so two constructs never share one operation.
        inputs : tuple[DidacticSortExpr, ...]
            The sorts the operation takes.
        output : DidacticSortExpr
            The sort it produces.

        Returns
        -------
        DidacticOperation
            The operation, created once per name and reused after.
        """
        key = (category, str(identity))
        operation = self._code_operations.get(key)
        if operation is not None:
            return operation
        operation = self.language.operation(
            _didactic_name(category, identity),
            returns=output,
            **{
                _didactic_name("code_argument", position): sort
                for position, sort in enumerate(inputs)
            },
        )
        self._code_operations[key] = operation
        return operation


class _Elaborator(
    _ProgramElaboration,
    _DeductionElaboration,
    _StructuralElaboration,
):
    """Lower one parsed source to a checked kernel module.

    The elaborator keeps the declarations it has processed in dictionaries
    keyed by source name and lowers bodies against them, so every pass of
    :meth:`elaborate` sees the results of the passes before it. The program
    and deduction passes it inherits from :class:`_ProgramElaboration` and
    :class:`_DeductionElaboration` share the same state.

    Parameters
    ----------
    source : QvrQiecSource
        The parsed source with its projection and options.
    """

    def __init__(self, source: QvrQiecSource) -> None:
        self.source = source
        self.statements = source.syntax.statements
        self.index_sorts: dict[str, UserIndexSort] = {}
        self.families: dict[str, FamilyDecl] = {}
        self.constructors: dict[str, ConstructorDecl] = {}
        self.effects: dict[str, EffectDef] = {}
        self.instances: dict[str, NamedEffectInstance] = {}
        self.handlers: dict[str, HandlerDef] = {}
        self.computation_signatures: dict[str, ComputationSignature] = {}
        self.entries: list[ProgramEntry] = []
        self._lambda_macros: dict[str, surface.LetExprLambda] = {}
        # Numbers the locals a lowering introduces to share a value, so
        # each such binder is unique across the module.
        self._shared_serial = 0
        self._program_state: _ProgramState | None = None
        self._program_objects: dict[str, ObjectInfo] = {}
        self._program_morphisms: dict[str, surface.MorphismDecl] = {}
        self._program_lets: dict[str, surface.Expr] = {}
        self._program_declarations: dict[str, surface.ProgramDecl] = {}
        # Types fixed for free names of the program being elaborated: a
        # scan cell specialized to its scanning program reads each name
        # that program binds before the scan at that binding's type.
        self._pinned_types: dict[str, TypeExpr] = {}
        self.registry = KernelRegistry()
        # The type a `return` in the body being lowered should produce. It
        # is a hint for positions that cannot state their type, such as a
        # site literal or an integral literal standing for a real; the
        # kernel still checks the body against its declared type.
        self._expected_result: TypeExpr | None = None

    def elaborate(self) -> QiecModule:
        """Run every pass and return the checked kernel module.

        Returns
        -------
        QiecModule
            The lowered module.

        Notes
        -----
        The order is load bearing. Declarations come first, then the
        registry, then computation signatures, then handler bodies, then
        computation bodies. A body may call a computation declared after
        it, so every signature has to exist before the first body is
        read.

        Raises
        ------
        QiecDiagnosticError
            If any pass rejects the source.
        """
        self._declare_index_sorts()
        self._declare_family_headers()
        self._declare_effect_headers()
        self._declare_effect_operations()
        self._declare_constructors()
        self._declare_structural_declarations()
        self._validate_indexed_language()
        self._declare_instances()
        self._declare_program_instances()
        self._build_registry()
        self._declare_deduction_declarations()
        self._declare_structural_signatures()
        self._declare_computation_signatures()
        self._declare_handlers()
        self._register_handlers()
        deductions = self._declare_deductions()
        structural = self._declare_structural_computations()
        programs = self._declare_programs()
        computations = self._declare_computations()
        return QiecModule(
            self.source.module_name,
            QVR_SOURCE_VERSION,
            tuple(self.index_sorts.values()),
            tuple(self.families.values()),
            tuple(self.constructors.values()),
            tuple(self.effects.values()),
            tuple(self.instances.values()),
            tuple(self.handlers.values()),
            (*deductions, *structural, *programs, *computations),
            tuple(self.entries),
        )

    def _validate_indexed_language(self) -> None:
        """Check the QVR GADT projection through Didactic's public language API."""

        if not self.index_sorts and not self.families:
            return
        try:
            _DidacticGadtProjection(
                self.source.module_name,
                tuple(self.index_sorts.values()),
                tuple(self.families.values()),
                tuple(self.constructors.values()),
            ).compile()
        except (GADTDeclarationError, GatError, TypeError, ValueError) as error:
            declarations = self._items(surface.QiecFamilyDecl) or self._items(
                surface.QiecIndexDecl
            )
            self._fail(
                declarations[0],
                f"Didactic rejected the indexed-family declaration: {error}",
                code="qiec-index",
            )

    def _items(self, type_: type[object]) -> tuple[object, ...]:
        """Every statement of one class, in source order.

        Parameters
        ----------
        type_ : type
            The statement class to select.

        Returns
        -------
        tuple
            The matching statements, in source order.
        """
        return tuple(item for item in self.statements if isinstance(item, type_))

    def _declare_index_sorts(self) -> None:
        """Collect every user index sort declared in the module.

        Raises
        ------
        QiecDiagnosticError
            If two sorts share a name, or a sort is malformed. A sort's constructors are part of its identity, so a duplicate name with different constructors would make coverage answer against the wrong datatype.
        """
        declarations = cast(
            tuple[surface.QiecIndexDecl, ...], self._items(surface.QiecIndexDecl)
        )
        for declaration in declarations:
            if declaration.name in self.index_sorts:
                self._fail(declaration, f"duplicate index sort {declaration.name!r}")
            constructor_names: set[str] = set()
            for constructor in declaration.constructors:
                if constructor.name in constructor_names:
                    self._fail(
                        constructor,
                        f"duplicate constructor {constructor.name!r} in index sort "
                        f"{declaration.name!r}",
                        code="qiec-index",
                    )
                constructor_names.add(constructor.name)
            try:
                self.index_sorts[declaration.name] = UserIndexSort(
                    declaration.name,
                    tuple(item.name for item in declaration.constructors),
                    tuple(len(item.arguments) for item in declaration.constructors),
                )
            except (TypeError, ValueError) as error:
                self._fail_kernel(declaration, error, fallback="qiec-index")
        for declaration in declarations:
            current = self.index_sorts[declaration.name]
            for constructor in declaration.constructors:
                for argument in constructor.arguments:
                    actual = self._lower_sort(argument)
                    if actual != current:
                        self._fail(
                            constructor,
                            "QIEC index constructors must be self-recursive; "
                            f"expected {current.name!r}, got {actual!r}",
                            code="qiec-index",
                        )

    def _declare_family_headers(self) -> None:
        """Collect each indexed family's header, before its constructors.

        Raises
        ------
        QiecDiagnosticError
            If two families share a name, or a header's binders are malformed. Headers come first because a constructor's result indices mention the family's own binders.
        """
        declarations = cast(
            tuple[surface.QiecFamilyDecl, ...], self._items(surface.QiecFamilyDecl)
        )
        for declaration in declarations:
            if declaration.name in self.families:
                self._fail(declaration, f"duplicate family {declaration.name!r}")
            if not isinstance(declaration.result_kind, surface.QiecTypeKind):
                self._fail(
                    declaration.result_kind,
                    "indexed families must return Type",
                    code="qiec-kind",
                )
            parameters = self._lower_telescope(declaration.parameters)
            indices = self._lower_telescope(declaration.indices, refinable=True)
            parameter_names = tuple(binder.name for binder in declaration.parameters)
            for binder in declaration.indices:
                if binder.name in parameter_names:
                    self._fail(
                        binder,
                        f"duplicate binder {binder.name!r} in family "
                        f"{declaration.name!r}",
                        code="qiec-kind",
                    )
            family_id = FamilyId.derive(
                self.source.module_name, "family", declaration.name
            )
            constructor_ids = tuple(
                ConstructorId.derive(family_id, "constructor", item.name)
                for item in declaration.constructors
            )
            self.families[declaration.name] = FamilyDecl(
                family_id,
                declaration.name,
                parameters,
                indices,
                constructor_ids,
            )

    def _declare_constructors(self) -> None:
        """Lower every constructor against the family that declares it.

        Raises
        ------
        QiecDiagnosticError
            If a constructor is duplicated, names an unknown family, returns the wrong number of indices, or shadows a family binder.
        """
        declarations = cast(
            tuple[surface.QiecFamilyDecl, ...], self._items(surface.QiecFamilyDecl)
        )
        for declaration in declarations:
            family = self.families[declaration.name]
            family_scope = family.parameters
            family_binder_names = {
                binder.name for binder in (*family.parameters, *family.indices)
            }
            for position, authored in enumerate(declaration.constructors):
                if authored.name in self.constructors:
                    self._fail(
                        authored,
                        f"duplicate constructor name {authored.name!r}",
                        code="qiec-index",
                    )
                for binder in authored.binders:
                    if binder.name in family_binder_names:
                        self._fail(
                            binder,
                            f"constructor {authored.name!r} shadows family binder "
                            f"{binder.name!r}",
                            code="qiec-index",
                        )
                telescope = self._lower_telescope(authored.binders)
                scope = (*family_scope, *telescope)
                fields = tuple(
                    FieldDef(f"arg{field_position}", self._lower_type(item, scope))
                    for field_position, item in enumerate(authored.arguments)
                )
                result = self._lower_type(authored.result, scope)
                if not isinstance(result, TypeApplication) or (
                    result.constructor != family.type_constructor
                ):
                    self._fail(
                        authored.result,
                        f"constructor {authored.name!r} must return {family.name!r}",
                        code="qiec-index",
                    )
                parameter_count = len(family.parameters)
                expected_parameters = tuple(
                    self._binder_variable(binder) for binder in family.parameters
                )
                if result.arguments[:parameter_count] != expected_parameters:
                    self._fail(
                        authored.result,
                        "GADT constructor results must preserve family parameters",
                        code="qiec-index",
                    )
                constructor = ConstructorDecl(
                    family.constructors[position],
                    family.id,
                    authored.name,
                    telescope,
                    fields,
                    result.arguments[parameter_count:],
                )
                self.constructors[authored.name] = constructor

    def _declare_effect_headers(self) -> None:
        """Collect each effect's name and static telescope.

        Raises
        ------
        QiecDiagnosticError
            If two effects share a name, or a telescope is malformed.
        """
        declarations = cast(
            tuple[surface.QiecEffectDecl, ...], self._items(surface.QiecEffectDecl)
        )
        for declaration in declarations:
            if declaration.name in self.effects:
                self._fail(declaration, f"duplicate effect {declaration.name!r}")
            effect_id = EffectId.derive(
                self.source.module_name,
                "effect",
                declaration.name,
            )
            telescope = self._lower_telescope(declaration.binders)
            ref = EffectRef(effect_id, declaration.name)
            try:
                self.effects[declaration.name] = EffectDef(ref, telescope, ())
            except (TypeError, ValueError) as error:
                self._fail_kernel(declaration, error, fallback="qiec-handler")

    def _declare_effect_operations(self) -> None:
        """Lower each effect's operations into its declaration.

        Raises
        ------
        QiecDiagnosticError
            If an operation is duplicated within its effect, shadows an interface binder, or has an ill-formed signature.
        """
        declarations = cast(
            tuple[surface.QiecEffectDecl, ...], self._items(surface.QiecEffectDecl)
        )
        for declaration in declarations:
            header = self.effects[declaration.name]
            effect_id = header.ref.id
            telescope = header.telescope
            lowered_operations: list[OperationDef] = []
            for operation in declaration.operations:
                operation_telescope = self._lower_telescope(operation.binders)
                operation_scope = (*telescope, *operation_telescope)
                authored_arguments = operation.arguments
                # ``Unit -> A`` is the conventional nullary-operation surface.
                if (
                    len(authored_arguments) == 1
                    and isinstance(authored_arguments[0], surface.QiecTypeName)
                    and authored_arguments[0].name == "Unit"
                ):
                    authored_arguments = ()
                lowered_operations.append(
                    OperationDef(
                        OperationId.derive(
                            str(effect_id),
                            operation.name,
                        ),
                        operation.name,
                        operation_telescope,
                        tuple(
                            ArgumentDef(
                                f"arg{position}",
                                self._lower_type(argument, operation_scope),
                            )
                            for position, argument in enumerate(authored_arguments)
                        ),
                        self._lower_type(operation.result, operation_scope),
                    )
                )
            operations = tuple(lowered_operations)
            try:
                self.effects[declaration.name] = EffectDef(
                    header.ref,
                    telescope,
                    operations,
                )
            except (TypeError, ValueError) as error:
                self._fail_kernel(declaration, error, fallback="qiec-handler")

    def _declare_instances(self) -> None:
        """Allocate the module-level lexical effect instances.

        Raises
        ------
        QiecDiagnosticError
            If two instances share a name, or an instance names an unknown or wrongly applied interface.
        """
        declarations = cast(
            tuple[surface.QiecEffectInstanceDecl, ...],
            self._items(surface.QiecEffectInstanceDecl),
        )
        for declaration in declarations:
            if declaration.name in self.instances:
                self._fail(declaration, f"duplicate instance {declaration.name!r}")
            effect = self._lower_effect_ref(declaration.effect, ())
            entry = instantiate_effect(
                effect,
                module=self.source.module_name,
                lexical_path=("instances", declaration.name),
            )
            self.instances[declaration.name] = NamedEffectInstance(
                declaration.name,
                entry,
                self._origin(
                    declaration,
                    ("instances", declaration.name),
                    "effect-instance",
                ),
            )
            self.registry.instance_names[entry.instance] = declaration.name

    def _declare_handlers(self) -> None:
        """Lower every handler, including its authored clause bodies.

        Raises
        ------
        QiecDiagnosticError
            If a handler is duplicated, claims both totality and forwarding, covers an unknown operation, binds the wrong number of arguments in a clause, or fails the kernel's own validation.
        """
        declarations = cast(
            tuple[surface.QiecHandlerDecl, ...], self._items(surface.QiecHandlerDecl)
        )
        for declaration in declarations:
            if declaration.name in self.handlers:
                self._fail(declaration, f"duplicate handler {declaration.name!r}")
            if declaration.duplicate_options:
                self._fail(
                    declaration,
                    "duplicate handler option(s): "
                    f"{', '.join(declaration.duplicate_options)}",
                    code="qiec-handler",
                )
            if declaration.coverage == "total" and declaration.forwards_unknown:
                self._fail(
                    declaration,
                    "a handler cannot be both total and forwarding",
                    code="qiec-handler",
                )
            telescope = self._lower_telescope(declaration.binders)
            effect = self._lower_effect_ref(declaration.effect, telescope)
            definition = self._effect(declaration.effect.name)
            assert definition is not None
            operations = {
                operation.name: operation for operation in definition.operations
            }
            input_type = self._lower_type(declaration.input_type, telescope)
            output_type = self._lower_type(declaration.output_type, telescope)
            introduced = self._lower_row(
                declaration.introduced,
                ("handlers", declaration.name, "introduced"),
            )
            # The interface's own binders are instantiated at the
            # application this handler names, so an operation signature
            # mentioning them reads at the right types inside a clause.
            try:
                interface_substitution = instantiate_telescope(
                    definition.telescope, effect.arguments
                )
            except (TypeError, ValueError) as error:
                self._fail_kernel(declaration, error, fallback="qiec-handler")
            clauses: list[HandlerClauseDef] = []
            return_clause: HandlerReturnClauseDef | None = None
            seen_clauses: set[str] = set()
            for position, clause in enumerate(declaration.clauses):
                base = ("handlers", declaration.name, "clauses", position)
                if isinstance(clause, surface.QiecHandlerReturnClause):
                    if return_clause is not None:
                        self._fail(
                            clause,
                            f"handler {declaration.name!r} has more than one "
                            f"return clause",
                            code="qiec-handler",
                        )
                    binder = Local(clause.binder.name, input_type)
                    self._expected_result = output_type
                    return_clause = HandlerReturnClauseDef(
                        binder,
                        self._lower_computation(
                            clause.body,
                            telescope,
                            CheckContext((binder,)),
                            (*base, "body"),
                        ),
                    )
                    self._expected_result = None
                    continue
                if clause.operation in seen_clauses:
                    self._fail(
                        clause,
                        f"duplicate clause {clause.operation!r} in handler "
                        f"{declaration.name!r}",
                        code="qiec-handler",
                    )
                seen_clauses.add(clause.operation)
                operation = operations.get(clause.operation)
                if operation is None:
                    self._fail(
                        clause,
                        f"effect {effect.name!r} has no operation {clause.operation!r}",
                        code="qiec-handler",
                    )
                clause_binders = self._lower_telescope(clause.binders)
                # A signature-only clause has no body for a static binder
                # to be in scope of, so it need not name the operation's
                # telescope; an authored clause has to bind all of it.
                if clause.body is None and not clause_binders:
                    clause_binders = operation.telescope
                if len(clause_binders) != len(operation.telescope):
                    self._fail(
                        clause,
                        f"clause {clause.operation!r} binds "
                        f"{len(clause_binders)} static argument(s); the "
                        f"operation declares {len(operation.telescope)}",
                        code="qiec-handler",
                    )
                clause_scope = (*telescope, *clause_binders)
                try:
                    parameter_types, resumed = instantiate_operation(
                        operation,
                        tuple(
                            self._binder_variable(binder) for binder in clause_binders
                        ),
                        interface_substitution,
                    )
                except (TypeError, ValueError) as error:
                    self._fail_kernel(clause, error, fallback="qiec-handler")
                # A signature-only clause binds nothing, because it has
                # no body for a binder to be in scope of. Only an
                # authored clause has to name every argument.
                if clause.body is not None and len(clause.parameters) != len(
                    parameter_types
                ):
                    self._fail(
                        clause,
                        f"clause {clause.operation!r} binds "
                        f"{len(clause.parameters)} argument(s); the operation "
                        f"takes {len(parameter_types)}",
                        code="qiec-handler",
                    )
                parameters = tuple(
                    Local(authored.name, declared)
                    for authored, declared in zip(
                        clause.parameters, parameter_types, strict=False
                    )
                )
                # Inside the body, ``resume`` carries what the operation
                # supplies and answers the handler's output, performing the
                # handler's introduced row; the context has to know all
                # three before a bound ``resume`` can be typed.
                self._expected_result = output_type
                body = (
                    None
                    if clause.body is None
                    else self._lower_computation(
                        clause.body,
                        clause_scope,
                        CheckContext(parameters).with_resumption(
                            ResumptionType(resumed, output_type, introduced)
                        ),
                        (*base, "body"),
                    )
                )
                self._expected_result = None
                clauses.append(
                    HandlerClauseDef(
                        operation.id,
                        ResumptionGrade(clause.grade),
                        parameters,
                        body,
                    )
                )
            try:
                handler = HandlerDef(
                    HandlerId.derive(
                        self.source.module_name, "handler", declaration.name
                    ),
                    declaration.name,
                    effect,
                    tuple(clauses),
                    input_type,
                    output_type,
                    introduced,
                    total=declaration.coverage == "total",
                    forwards_unknown=declaration.forwards_unknown,
                    telescope=telescope,
                    return_clause=return_clause,
                    implementation=declaration.implementation,
                )
            except (TypeError, ValueError) as error:
                self._fail_kernel(declaration, error, fallback="qiec-handler")
            self.handlers[declaration.name] = handler

    def _build_registry(self) -> None:
        """Register the families, constructors, and effects with the kernel.

        Raises
        ------
        QiecDiagnosticError
            If any declaration fails the kernel's validation, reported at the declaration's own source position.
        """
        family_nodes = {
            item.name: item
            for item in cast(
                tuple[surface.QiecFamilyDecl, ...],
                self._items(surface.QiecFamilyDecl),
            )
        }
        effect_nodes = {
            item.name: item
            for item in cast(
                tuple[surface.QiecEffectDecl, ...],
                self._items(surface.QiecEffectDecl),
            )
        }
        for family in self.families.values():
            try:
                self.registry.register_family(family)
            except (KernelError, TypeError, ValueError) as error:
                self._fail_kernel(
                    family_nodes[family.name], error, fallback="qiec-index"
                )
        for constructor in self.constructors.values():
            try:
                self.registry.register_constructor(constructor)
            except (KernelError, TypeError, ValueError) as error:
                self._fail_kernel(
                    family_nodes[self.registry.families[constructor.family].name],
                    error,
                    fallback="qiec-index",
                )
        for effect in self.effects.values():
            if effect.ref.id in self.registry.effects:
                # A prelude interface resolved by an earlier pass is already
                # registered; the registry rejects a second registration.
                continue
            try:
                self.registry.register_effect(effect)
            except (KernelError, TypeError, ValueError) as error:
                self._fail_kernel(
                    effect_nodes[effect.ref.name], error, fallback="qiec-kind"
                )

    def _register_handlers(self) -> None:
        """Register built handlers, after their bodies have been lowered.

        Separate from the rest of the registry build because a clause
        body may call a computation or perform an effect, so it can only
        be lowered once the families, effects, and computation
        signatures are all present.

        Raises
        ------
        QiecDiagnosticError
            If a handler fails the kernel's own validation, reported at
            the declaration's source position.
        """
        handler_nodes = {
            item.name: item
            for item in cast(
                tuple[surface.QiecHandlerDecl, ...],
                self._items(surface.QiecHandlerDecl),
            )
        }
        for handler in self.handlers.values():
            try:
                self.registry.register_handler(handler)
            except (KernelError, TypeError, ValueError) as error:
                node = handler_nodes.get(handler.name)
                if node is None:
                    node = next(
                        (
                            component.declaration
                            for component in getattr(
                                self, "_structural_components", {}
                            ).values()
                            if component.handler.id == handler.id
                        ),
                        None,
                    )
                if node is None:
                    raise
                self._fail_kernel(node, error, fallback="qiec-handler")

    def _declare_computation_signatures(self) -> None:
        """Collect every computation signature before any body is lowered.

        This is what makes a forward call ordinary. A body may call a
        computation declared below it, and two computations may call each
        other, because the table is complete before the first body is
        read.

        Raises
        ------
        QiecDiagnosticError
            If two computations share a name, if a parameter name
            repeats, or if a declared type or row is ill-formed.
        """
        seen: set[str] = set()
        for declaration in cast(
            tuple[surface.QiecComputationDecl, ...],
            self._items(surface.QiecComputationDecl),
        ):
            if declaration.name in seen:
                self._fail(declaration, f"duplicate computation {declaration.name!r}")
            seen.add(declaration.name)
            telescope = self._lower_telescope(declaration.binders)
            parameters = tuple(
                self._lower_type(parameter.type_expr, telescope)
                for parameter in declaration.parameters
            )
            signature = ComputationSignature(
                ComputationId.derive(
                    self.source.module_name,
                    "computation",
                    declaration.name,
                ),
                declaration.name,
                telescope,
                parameters,
                self._lower_type(declaration.result_type, telescope),
                self._lower_row(
                    declaration.effects,
                    ("computations", declaration.name, "effects"),
                ),
            )
            try:
                self.registry.register_computation(signature)
            except (KernelError, TypeError, ValueError) as error:
                self._fail_kernel(declaration, error, fallback="qiec-route")
            self.computation_signatures[declaration.name] = signature

    def _declare_computations(self) -> tuple[NamedComputation, ...]:
        """Lower every computation body against its declared signature.

        Returns
        -------
        tuple[NamedComputation, ...]
            The lowered computations, in source order. Their signatures
            were registered in an earlier pass, so a body calling one
            declared later resolves here.

        Raises
        ------
        QiecDiagnosticError
            If a body fails to check, or its inferred type does not
            conform to the declared one.
        """
        declarations = cast(
            tuple[surface.QiecComputationDecl, ...],
            self._items(surface.QiecComputationDecl),
        )
        seen: set[str] = set()
        computations: list[NamedComputation] = []
        for declaration in declarations:
            if declaration.name in seen:
                self._fail(declaration, f"duplicate computation {declaration.name!r}")
            seen.add(declaration.name)
            telescope = self._lower_telescope(declaration.binders)
            parameter_names: set[str] = set()
            static_names = {binder.name for binder in telescope}
            for parameter in declaration.parameters:
                if parameter.name in parameter_names:
                    self._fail(
                        parameter,
                        f"duplicate parameter {parameter.name!r} in computation "
                        f"{declaration.name!r}",
                    )
                if parameter.name in static_names:
                    self._fail(
                        parameter,
                        f"computation {declaration.name!r} reuses static name "
                        f"{parameter.name!r} as a parameter",
                    )
                parameter_names.add(parameter.name)
            parameters = tuple(
                Local(parameter.name, self._lower_type(parameter.type_expr, telescope))
                for parameter in declaration.parameters
            )
            context = CheckContext(parameters)
            path = ("computations", declaration.name, "body")
            self._expected_result = self._lower_type(declaration.result_type, telescope)
            body = self._lower_computation(declaration.body, telescope, context, path)
            self._expected_result = None
            computation = NamedComputation(
                ComputationId.derive(
                    self.source.module_name,
                    "computation",
                    declaration.name,
                ),
                declaration.name,
                telescope,
                parameters,
                body,
                ComputationType(
                    self._lower_row(
                        declaration.effects,
                        ("computations", declaration.name, "effects"),
                    ),
                    self._lower_type(declaration.result_type, telescope),
                ),
                self._origin(
                    declaration,
                    ("computations", declaration.name),
                    "computation",
                ),
            )
            try:
                actual = infer_computation(body, self.registry, context)
            except (KernelError, TypeError, ValueError) as error:
                self._fail_kernel(declaration.body, error, fallback="qiec-kind")
            if actual.result != computation.type.result:
                self._fail(
                    declaration,
                    f"computation body returns {render_static(actual.result)}, "
                    f"expected {render_static(computation.type.result)}",
                    code="qiec-kind",
                )
            if not computation_type_conforms(actual, computation.type):
                self._fail(
                    declaration,
                    f"computation body has effect row "
                    f"{render_row(actual.effects, self.registry.instance_names)}, "
                    f"not declared row "
                    f"{render_row(computation.type.effects, self.registry.instance_names)}",
                    code="qiec-row",
                )
            computations.append(computation)
        return tuple(computations)

    def _lower_telescope(
        self,
        binders: tuple[surface.QiecBinder, ...],
        *,
        refinable: bool = False,
    ) -> Telescope:
        """Lower a source telescope into kernel binders.

        Parameters
        ----------
        binders : tuple
            The authored binders.

        Returns
        -------
        Telescope
            The lowered binders, in order.

        Raises
        ------
        QiecDiagnosticError
            If a binder is malformed or its sort is unknown.
        """
        lowered: list[TelescopeBinder] = []
        seen: set[str] = set()
        for binder in binders:
            name = cast(str, binder.name)
            if name in seen:
                self._fail(
                    binder,
                    f"duplicate telescope binder {name!r}",
                    code="qiec-kind",
                )
            seen.add(name)
            if isinstance(binder, surface.QiecTypeBinder):
                kind = (
                    TYPE
                    if isinstance(binder.binder_kind, surface.QiecTypeKind)
                    else EFFECT
                )
                lowered.append(TypeBinder(name, kind, refinable))
            elif isinstance(binder, surface.QiecIndexBinder):
                lowered.append(
                    IndexBinder(name, self._lower_sort(binder.sort), refinable)
                )
            elif isinstance(binder, surface.QiecEffectBinder):
                lowered.append(EffectBinder(name, refinable))
            else:  # pragma: no cover - closed source union
                self._fail(binder, "unknown telescope binder", code="qiec-kind")
        return tuple(lowered)

    def _lower_sort(self, sort: surface.QiecIndexSort) -> IndexSort:
        """Lower an authored index sort.

        Parameters
        ----------
        sort : object
            The authored sort.

        Returns
        -------
        IndexSort
            The lowered sort.

        Raises
        ------
        QiecDiagnosticError
            If the sort is unknown.
        """
        if isinstance(sort, surface.QiecNatSort):
            return self.index_sorts.get("Nat", NAT)
        if isinstance(sort, surface.QiecShapeSort):
            return ShapeSort(sort.rank)
        if isinstance(sort, surface.QiecContextSort):
            return ContextSort(sort.signature)
        if isinstance(sort, surface.QiecUserIndexSort):
            resolved = self.index_sorts.get(sort.name)
            if resolved is None:
                self._fail(sort, f"unknown index sort {sort.name!r}", code="qiec-index")
            return resolved
        self._fail(sort, "unknown index sort", code="qiec-index")

    def _lower_type(
        self,
        authored: surface.QiecTypeExpr,
        scope: Telescope,
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> TypeExpr:
        """Lower an authored type expression.

        Parameters
        ----------
        authored : object
            The authored type expression.
        scope : Telescope
            Static binders in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.

        Returns
        -------
        TypeExpr
            The lowered type.

        Raises
        ------
        QiecDiagnosticError
            If the type is malformed, or names an unknown constructor or an unbound variable.
        """
        if isinstance(authored, surface.QiecTypeName):
            bound = (static_bindings or {}).get(authored.name)
            if bound is not None:
                if isinstance(bound, TypeVariable):
                    return bound
                self._fail(
                    authored,
                    f"static binder {authored.name!r} is not a type",
                    code="qiec-kind",
                )
            binder = next(
                (
                    item
                    for item in reversed(scope)
                    if isinstance(item, TypeBinder) and item.name == authored.name
                ),
                None,
            )
            if binder is not None:
                return TypeVariable(binder.name, binder.kind)
            primitive = {
                "Unit": UNIT,
                "Bool": BOOL,
                "Int": INT,
                "Real": REAL,
                "String": STRING,
                "LogWeight": LOG_WEIGHT,
            }.get(authored.name)
            if primitive is not None:
                return primitive
            family = self.families.get(authored.name)
            if family is not None and not (*family.parameters, *family.indices):
                return TypeApplication(family.type_constructor)
            if authored.name in BUILTIN_TYPE_CONSTRUCTORS:
                self._fail(
                    authored,
                    f"builtin type {authored.name!r} takes arguments",
                    code="qiec-kind",
                )
            self._fail(authored, f"unknown or unsaturated type {authored.name!r}")
        if isinstance(authored, surface.QiecTypeApplication):
            builtin = BUILTIN_TYPE_CONSTRUCTORS.get(authored.constructor)
            if builtin is not None and authored.constructor not in self.families:
                return self._lower_builtin_type(
                    authored, builtin, scope, static_bindings
                )
            family = self.families.get(authored.constructor)
            if family is None:
                self._fail(authored, f"unknown indexed family {authored.constructor!r}")
            if len(authored.static_arguments) != len(family.parameters):
                self._fail(
                    authored,
                    f"family {family.name!r} expects {len(family.parameters)} "
                    "static parameters",
                    code="qiec-kind",
                )
            if len(authored.indices) != len(family.indices):
                self._fail(
                    authored,
                    f"family {family.name!r} expects {len(family.indices)} indices",
                    code="qiec-index",
                )
            parameters = tuple(
                self._lower_static_argument(argument, binder, scope, static_bindings)
                for argument, binder in zip(
                    authored.static_arguments, family.parameters, strict=True
                )
            )
            indices = tuple(
                self._lower_index(argument, binder.sort, scope, static_bindings)
                for argument, binder in zip(
                    authored.indices, family.indices, strict=True
                )
                if isinstance(binder, IndexBinder)
            )
            if len(indices) != len(family.indices):
                self._fail(
                    authored,
                    "family indices must use index binders",
                    code="qiec-index",
                )
            return TypeApplication(family.type_constructor, (*parameters, *indices))
        if isinstance(authored, surface.QiecProductType):
            return product_type(
                *(
                    self._lower_type(item, scope, static_bindings)
                    for item in authored.components
                )
            )
        if isinstance(authored, surface.QiecFunctionType):
            return FunctionType(
                self._lower_type(authored.parameter, scope, static_bindings),
                self._lower_type(authored.result, scope, static_bindings),
            )
        self._fail(authored, "unknown type expression", code="qiec-kind")

    def _effect(self, name: str) -> EffectDef | None:
        """Resolve an effect interface by name, admitting the prelude's.

        A module refers to ``Random``, ``Score``, ``State``, ``Abort``,
        ``Choose``, or ``Weight`` without declaring it; the prelude's
        interface then joins the module's effects under the prelude's own
        identity, which is what lets the prelude handlers serve it. A
        module declaration of the same name takes precedence.

        Parameters
        ----------
        name : str
            The interface's source name.

        Returns
        -------
        EffectDef or None
            The module's or the prelude's interface, or ``None`` when
            neither declares the name.
        """
        definition = self.effects.get(name)
        if definition is not None:
            return definition
        prelude = _PRELUDE_EFFECTS.get(name)
        if prelude is None:
            return None
        self.effects[name] = prelude
        if prelude.ref.id not in self.registry.effects:
            self.registry.register_effect(prelude)
        return prelude

    def _lower_builtin_type(
        self,
        authored: surface.QiecTypeApplication,
        constructor: TypeConstructorRef,
        scope: Telescope,
        static_bindings: Mapping[str, StaticArgument] | None,
    ) -> TypeApplication:
        """Lower an application of a canonical builtin type constructor.

        Parameters
        ----------
        authored : surface.QiecTypeApplication
            The application, such as ``Sampleable[Real]`` or
            ``Tensor[Real]([3])``.
        constructor : TypeConstructorRef
            The builtin constructor named.
        scope : Telescope
            Static binders in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.

        Returns
        -------
        TypeApplication
            The application at the constructor's telescope.

        Raises
        ------
        QiecDiagnosticError
            If the argument counts do not match the constructor's
            telescope.
        """
        type_binders = tuple(
            binder for binder in constructor.telescope if isinstance(binder, TypeBinder)
        )
        index_binders = tuple(
            binder
            for binder in constructor.telescope
            if isinstance(binder, IndexBinder)
        )
        if len(authored.static_arguments) != len(type_binders):
            self._fail(
                authored,
                f"builtin type {constructor.name!r} expects {len(type_binders)} "
                "static arguments",
                code="qiec-kind",
            )
        if len(authored.indices) != len(index_binders):
            self._fail(
                authored,
                f"builtin type {constructor.name!r} expects {len(index_binders)} "
                "indices",
                code="qiec-index",
            )
        parameters = tuple(
            self._lower_static_argument(argument, binder, scope, static_bindings)
            for argument, binder in zip(
                authored.static_arguments, type_binders, strict=True
            )
        )
        indices = tuple(
            self._lower_index(argument, binder.sort, scope, static_bindings)
            for argument, binder in zip(authored.indices, index_binders, strict=True)
        )
        return TypeApplication(constructor, (*parameters, *indices))

    def _lower_static_argument(
        self,
        authored: surface.QiecStaticArgument,
        binder: TelescopeBinder,
        scope: Telescope,
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> StaticArgument:
        """Lower one static argument against the binder it instantiates.

        Parameters
        ----------
        authored : surface.QiecStaticArgument
            The authored argument: type syntax, or an index literal.
        binder : TelescopeBinder
            The binder it instantiates, which fixes the namespace expected.
        scope : Telescope
            Static binders in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.

        Returns
        -------
        StaticArgument
            The lowered argument.

        Raises
        ------
        QiecDiagnosticError
            If the argument is of the wrong namespace for the binder, or is itself malformed.
        """
        if isinstance(authored, surface.QiecIndexLiteral):
            if not isinstance(binder, IndexBinder):
                self._fail(
                    authored,
                    f"index literal {authored.value} fills the {binder.name!r} "
                    "binder, which takes a type or an effect",
                    code="qiec-kind",
                )
            return self._lower_index(authored, binder.sort, scope, static_bindings)
        if isinstance(binder, TypeBinder):
            return self._lower_type(authored, scope, static_bindings)
        if isinstance(binder, IndexBinder):
            return self._lower_index(
                self._type_syntax_as_index(authored),
                binder.sort,
                scope,
                static_bindings,
            )
        if isinstance(authored, surface.QiecTypeApplication):
            if authored.indices:
                self._fail(
                    authored,
                    "effect applications cannot carry family indices",
                    code="qiec-kind",
                )
            return self._lower_effect_ref(
                surface.QiecEffectRef(
                    name=authored.constructor,
                    arguments=authored.static_arguments,
                    line=authored.line,
                    col=authored.col,
                ),
                scope,
                static_bindings,
            )
        if isinstance(authored, surface.QiecTypeName):
            bound = (static_bindings or {}).get(authored.name)
            if bound is not None:
                if isinstance(bound, EffectVariable):
                    return bound
                self._fail(
                    authored,
                    f"static binder {authored.name!r} is not an effect",
                    code="qiec-kind",
                )
            if any(
                isinstance(item, EffectBinder) and item.name == authored.name
                for item in scope
            ):
                return EffectVariable(authored.name)
            definition = self._effect(authored.name)
            if definition is not None and not definition.telescope:
                return definition.ref
        self._fail(
            authored,
            f"argument for effect binder {binder.name!r} is not an effect",
            code="qiec-kind",
        )

    def _type_syntax_as_index(
        self, authored: surface.QiecTypeExpr
    ) -> surface.QiecIndexExpr:
        """Read type syntax in an index position.

        The surface writes a type application and an index application
        alike, so a name in an index position arrives as type syntax and
        is reinterpreted here rather than being rejected.

        Parameters
        ----------
        authored : object
            Type syntax to reinterpret.

        Returns
        -------
        IndexTerm or None
            The index, or None when the syntax cannot be read as one.
        """
        if isinstance(authored, surface.QiecTypeName):
            return surface.QiecIndexName(
                name=authored.name, line=authored.line, col=authored.col
            )
        if isinstance(authored, surface.QiecTypeApplication):
            arguments = tuple(
                self._type_syntax_as_index(argument)
                for argument in authored.static_arguments
            )
            return surface.QiecIndexApplication(
                constructor=authored.constructor,
                arguments=(*arguments, *authored.indices),
                line=authored.line,
                col=authored.col,
            )
        self._fail(
            authored,
            "this static argument is not valid index syntax",
            code="qiec-index",
        )

    def _lower_index(
        self,
        authored: surface.QiecIndexExpr,
        expected: IndexSort,
        scope: Telescope,
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> IndexVariable | IndexLiteral | IndexConstructor | ShapeIndex:
        """Lower an authored index term.

        Parameters
        ----------
        authored : object
            The authored index.
        expected : IndexSort
            The sort the position requires.
        scope : Telescope
            Static binders in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.

        Returns
        -------
        IndexTerm
            The lowered index.

        Raises
        ------
        QiecDiagnosticError
            If the index is malformed, of the wrong sort, or names a constructor the sort does not declare.
        """
        if isinstance(authored, surface.QiecIndexName):
            bound = (static_bindings or {}).get(authored.name)
            if bound is not None:
                if isinstance(bound, IndexVariable):
                    if bound.sort != expected:
                        self._fail(
                            authored,
                            f"index {authored.name!r} has the wrong sort",
                            code="qiec-index",
                        )
                    return bound
                self._fail(
                    authored,
                    f"static binder {authored.name!r} is not an index",
                    code="qiec-index",
                )
            binder = next(
                (
                    item
                    for item in reversed(scope)
                    if isinstance(item, IndexBinder) and item.name == authored.name
                ),
                None,
            )
            if binder is not None:
                if binder.sort != expected:
                    self._fail(
                        authored,
                        f"index {authored.name!r} has the wrong sort",
                        code="qiec-index",
                    )
                return IndexVariable(authored.name, expected)
            if isinstance(expected, UserIndexSort):
                try:
                    if expected.constructor_arity(authored.name) == 0:
                        return IndexLiteral(authored.name, expected)
                except ValueError:
                    pass
            self._fail(authored, f"unknown index {authored.name!r}", code="qiec-index")
        if isinstance(authored, surface.QiecIndexLiteral):
            try:
                return IndexLiteral(authored.value, expected)
            except (TypeError, ValueError) as error:
                self._fail_kernel(authored, error, fallback="qiec-index")
        if isinstance(authored, surface.QiecIndexApplication):
            if not isinstance(expected, UserIndexSort):
                self._fail(
                    authored,
                    "named index constructors require a user-defined index sort",
                    code="qiec-index",
                )
            arguments = tuple(
                self._lower_index(argument, expected, scope, static_bindings)
                for argument in authored.arguments
            )
            try:
                return IndexConstructor(
                    authored.constructor,
                    arguments,
                    expected,
                )
            except (TypeError, ValueError) as error:
                self._fail_kernel(authored, error, fallback="qiec-index")
        if isinstance(authored, surface.QiecShapeIndex):
            return ShapeIndex(
                tuple(
                    self._lower_index(item, NAT, scope, static_bindings)
                    for item in authored.dimensions
                )
            )
        self._fail(authored, "unknown index expression", code="qiec-index")

    def _lower_effect_ref(
        self,
        authored: surface.QiecEffectRef,
        scope: Telescope,
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> EffectRef:
        """Lower an authored effect interface application.

        Parameters
        ----------
        authored : object
            The authored interface application.
        scope : Telescope
            Static binders in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.

        Returns
        -------
        EffectRef
            The lowered application.

        Raises
        ------
        QiecDiagnosticError
            If the interface is unknown, or its arguments do not saturate the telescope.
        """
        definition = self._effect(authored.name)
        if definition is None:
            self._fail(authored, f"unknown effect interface {authored.name!r}")
        if len(authored.arguments) != len(definition.telescope):
            self._fail(
                authored,
                f"effect {authored.name!r} expects {len(definition.telescope)} arguments",
                code="qiec-kind",
            )
        arguments = tuple(
            self._lower_static_argument(argument, binder, scope, static_bindings)
            for argument, binder in zip(
                authored.arguments, definition.telescope, strict=True
            )
        )
        try:
            return definition.apply(arguments)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(authored, error, fallback="qiec-kind")

    def _lower_row(
        self,
        authored: surface.QiecEffectRow,
        path: tuple[str | int, ...],
    ) -> EffectRow:
        """Lower an authored effect row.

        Parameters
        ----------
        authored : object
            The authored row.
        path : tuple[str | int, ...]
            Structural path, for the row variable's derived identity.

        Returns
        -------
        EffectRow
            The lowered row, with its open tail when it has one.

        Raises
        ------
        QiecDiagnosticError
            If an entry names an unknown instance, or the row repeats one.
        """
        entries: list[RowEntry] = []
        for item in authored.entries:
            instance = self.instances.get(item.instance)
            if instance is None:
                self._fail(
                    item,
                    f"unknown effect instance {item.instance!r}",
                    code="qiec-row",
                )
            entries.append(instance.entry)
        tail: RowVariable | None = None
        if authored.tail is not None:
            if len(set(authored.lacks)) != len(authored.lacks):
                self._fail(
                    authored,
                    "a row lacks constraint cannot repeat an instance",
                    code="qiec-row",
                )
            lacks_names = set(authored.lacks)
            explicit_names = {item.instance for item in authored.entries}
            missing = explicit_names - lacks_names
            if missing:
                self._fail(
                    authored,
                    "an open row tail must explicitly lack its entries: "
                    f"{sorted(missing)!r}",
                    code="qiec-row",
                )
            lacks: list[EffectInstanceId] = []
            for name in authored.lacks:
                instance = self.instances.get(name)
                if instance is None:
                    self._fail(
                        authored,
                        f"row lacks unknown instance {name!r}",
                        code="qiec-row",
                    )
                lacks.append(instance.entry.instance)
            try:
                tail = RowVariable(
                    authored.tail,
                    RowVariableId.derive(self.source.module_name, path, authored.tail),
                    tuple(lacks),
                )
            except ValueError as error:
                self._fail(authored, str(error), code="qiec-row")
        try:
            return EffectRow(tuple(entries), tail)
        except ValueError as error:
            self._fail(authored, str(error), code="qiec-row")

    def _lower_value(
        self,
        authored: surface.QiecValue,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None = None,
        path: tuple[str | int, ...] = (),
        expected: TypeExpr | None = None,
    ) -> Value:
        """Lower an authored value term.

        Parameters
        ----------
        authored : surface.QiecValue
            The authored value: a constructor application or a node of
            the shared pure-expression tree.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.
        path : tuple[str | int, ...]
            Structural path of the position, entering the provenance of
            primitive applications.
        expected : TypeExpr or None
            The type the position calls for, when the surrounding term
            fixes one. It types a ``site`` literal, whose element type
            is not written at the site, and lets an integral literal
            stand where a ``Real`` is expected.

        Returns
        -------
        Value
            The lowered value.

        Raises
        ------
        QiecDiagnosticError
            If the value names an unbound local, applies a constructor
            wrongly, applies an operator or builtin to operands of the
            wrong types, projects from a non-tuple, or uses an expression
            form QIEC values do not admit.
        """
        if isinstance(authored, surface.QiecConstructorValue):
            constructor = self.constructors.get(authored.constructor)
            if constructor is None:
                self._fail(
                    authored,
                    f"unknown constructor {authored.constructor!r}",
                    code="qiec-index",
                )
            family = self.registry.families[constructor.family]
            binders = (*family.parameters, *constructor.telescope)
            if len(authored.static_arguments) != len(binders):
                self._fail(
                    authored,
                    f"constructor {constructor.name!r} expects {len(binders)} "
                    "static arguments",
                    code="qiec-kind",
                )
            static_arguments = tuple(
                self._lower_static_argument(argument, binder, scope, static_bindings)
                for argument, binder in zip(
                    authored.static_arguments, binders, strict=True
                )
            )
            return ConstructorValue(
                constructor.id,
                static_arguments,
                tuple(
                    self._lower_value(
                        value, scope, context, static_bindings, (*path, position)
                    )
                    for position, value in enumerate(authored.fields)
                ),
                self._lower_type(authored.result_type, scope, static_bindings),
            )
        if isinstance(authored, surface.LetExprVar):
            local = next(
                (
                    item
                    for item in reversed(context.locals)
                    if item.name == authored.name
                ),
                None,
            )
            if local is None:
                self._fail(authored, f"unbound local {authored.name!r}")
            return Var(local)
        if isinstance(authored, surface.LetExprLiteral):
            if authored.integral and expected != REAL:
                return LiteralValue(int(authored.value), INT)
            return LiteralValue(authored.value, REAL)
        if isinstance(authored, surface.LetExprBool):
            return LiteralValue(authored.value, BOOL)
        if isinstance(authored, surface.LetExprUnit):
            return LiteralValue(None, UNIT)
        if isinstance(authored, surface.LetExprString):
            return LiteralValue(authored.value, STRING)
        if isinstance(authored, surface.LetExprUnaryOp):
            operand = self._lower_value(
                authored.operand, scope, context, static_bindings, (*path, "operand")
            )
            if authored.op == "-" and isinstance(operand, LiteralValue):
                if operand.type == INT:
                    return LiteralValue(-cast(int, operand.value), INT)
                if operand.type == REAL:
                    return LiteralValue(-cast(float, operand.value), REAL)
            operand_type = self._value_type(operand, context, authored)
            element, shape = self._broadcast_element(
                authored, authored.op, (operand_type,)
            )
            if authored.op == "-":
                name = self._numeric_primitive(
                    authored, "-", element, {INT: "neg_int", REAL: "neg_real"}
                )
            else:
                if element != BOOL:
                    self._fail(
                        authored,
                        f"operand of `not` has type {self._render(operand_type)}; "
                        "`not` takes Bool",
                        code="qiec-primitive",
                    )
                name = "not"
            return self._primitive(name, (operand,), authored, path, shape)
        if isinstance(authored, surface.LetExprBinOp):
            left = self._lower_value(
                authored.left, scope, context, static_bindings, (*path, "left")
            )
            right = self._lower_value(
                authored.right, scope, context, static_bindings, (*path, "right")
            )
            left_type = self._value_type(left, context, authored.left)
            right_type = self._value_type(right, context, authored.right)
            left, left_type = self._promoted(
                left, left_type, right_type, authored, (*path, "left")
            )
            right, right_type = self._promoted(
                right, right_type, left_type, authored, (*path, "right")
            )
            element, shape = self._broadcast_element(
                authored, authored.op, (left_type, right_type)
            )
            table = _BINARY_PRIMITIVES.get(authored.op)
            if table is None:
                self._fail(
                    authored,
                    f"unknown operator `{authored.op}`",
                    code="qiec-primitive",
                )
            name = self._numeric_primitive(authored, authored.op, element, table)
            return self._primitive(name, (left, right), authored, path, shape)
        if isinstance(authored, surface.LetExprTuple):
            items = tuple(
                self._lower_value(
                    item, scope, context, static_bindings, (*path, position)
                )
                for position, item in enumerate(authored.items)
            )
            types = tuple(
                self._value_type(item, context, source)
                for item, source in zip(items, authored.items, strict=True)
            )
            return TupleValue(items, product_type(*types))
        if isinstance(authored, surface.LetExprIndex):
            source = self._lower_value(
                authored.array, scope, context, static_bindings, (*path, "array")
            )
            source_type = self._value_type(source, context, authored.array)
            if tensor_shape(source_type) is not None:
                return self._lower_gather(
                    source, source_type, authored, scope, context, static_bindings, path
                )
            if not (
                isinstance(source_type, TypeApplication)
                and source_type.constructor.name.startswith("Product")
            ):
                self._fail(
                    authored,
                    "indexing in a QIEC value selects a tuple component or a "
                    f"tensor slice, but the value has type {self._render(source_type)}",
                    code="qiec-primitive",
                )
            if len(authored.indices) != 1 or not (
                isinstance(authored.indices[0], surface.LetExprLiteral)
                and authored.indices[0].integral
            ):
                self._fail(
                    authored,
                    "a tuple component is selected by one integer literal, as "
                    "in `pair[0]`",
                    code="qiec-primitive",
                )
            position = int(authored.indices[0].value)
            components = source_type.arguments
            if not 0 <= position < len(components):
                self._fail(
                    authored,
                    f"tuple component {position} is outside a product of "
                    f"{len(components)} components",
                    code="qiec-primitive",
                )
            return Projection(source, position, cast(TypeExpr, components[position]))
        if isinstance(authored, surface.LetExprCall):
            if authored.func in FAMILIES and authored.func not in _BUILTIN_PRIMITIVES:
                return self._lower_family_application(
                    authored, scope, context, static_bindings, path, expected
                )
            if authored.func == "site":
                return self._lower_site(authored, expected)
            if authored.func == "log_prob":
                return self._lower_log_density(
                    authored, scope, context, static_bindings, path
                )
            macro = self._lambda_macros.get(authored.func)
            if macro is not None:
                return self._lower_macro_application(
                    macro, authored, scope, context, static_bindings, path, expected
                )
            if authored.func in _COLLECTION_BUILTINS:
                return self._lower_collection_value(
                    authored,
                    scope,
                    context,
                    static_bindings,
                    path,
                    expected,
                )
            arguments = tuple(
                self._lower_value(
                    argument, scope, context, static_bindings, (*path, position)
                )
                for position, argument in enumerate(authored.args)
            )
            types = tuple(
                self._value_type(argument, context, source)
                for argument, source in zip(arguments, authored.args, strict=True)
            )
            if authored.func in _REDUCTIONS and len(arguments) == 1:
                shape = tensor_shape(types[0])
                if shape is not None:
                    return _last_axis_reduction(
                        _REDUCTIONS[authored.func],
                        arguments[0],
                        shape,
                        path,
                        frozenset(local.name for local in context.locals),
                    )
            if authored.func in _ROWWISE:
                if len(arguments) != 1 or tensor_shape(types[0]) is None:
                    self._fail(
                        authored,
                        f"builtin {authored.func!r} takes one Tensor[Real]",
                        code="qiec-primitive",
                    )
                return Rowwise(_ROWWISE[authored.func], arguments[0], types[0])
            elements: list[TypeExpr] = []
            shape = None
            for item_type in types:
                split = tensor_shape(item_type)
                if split is None:
                    elements.append(item_type)
                    continue
                elements.append(split[0])
                if shape is not None and tuple(split[1]) != shape:
                    self._fail(
                        authored,
                        f"builtin {authored.func!r} is applied to tensors of "
                        "differing shapes",
                        code="qiec-primitive",
                    )
                shape = tuple(split[1])
            name = self._builtin_primitive(authored, tuple(elements))
            return self._primitive(name, arguments, authored, path, shape)
        if isinstance(authored, surface.LetExprList):
            return self._lower_tensor_literal(
                authored, scope, context, static_bindings, path, expected
            )
        if isinstance(authored, surface.LetExprFactor):
            return self._lower_factor(authored, scope, context, static_bindings, path)
        if isinstance(authored, surface.LetExprMethodCall):
            goal = self._goal_weight_value(authored, context, path)
            if goal is not None:
                return goal
        if isinstance(
            authored,
            surface.LetExprLambda | surface.LetExprMethodCall,
        ):
            self._fail(
                authored,
                f"{_EXPRESSION_FORMS[type(authored).__name__]} is not a QIEC "
                "value; QIEC values are literals, locals, operators, tuples, "
                "builtin applications, and constructor applications",
                code="qiec-primitive",
            )
        self._fail(authored, "unknown QIEC value")

    def _collection_lambda(
        self,
        authored: surface.LetExprCall,
        position: int,
    ) -> surface.LetExprLambda:
        """Resolve one inline or locally named lambda argument."""
        if position >= len(authored.args):
            self._fail(
                authored,
                f"{authored.func} expects a lambda at argument {position + 1}",
                code="qiec-primitive",
            )
        argument = authored.args[position]
        if isinstance(argument, surface.LetExprLambda):
            return argument
        if isinstance(argument, surface.LetExprVar):
            macro = self._lambda_macros.get(argument.name)
            if macro is not None:
                return macro
        self._fail(
            argument,
            f"argument {position + 1} of {authored.func!r} must be a lambda",
            code="qiec-primitive",
        )

    def _indexed_collection_item(
        self,
        source: surface.LetExprNode,
        index: surface.LetExprNode,
    ) -> surface.LetExprIndex:
        """Build the source expression selecting one leading-axis item."""
        return surface.LetExprIndex(
            array=source,
            indices=(index,),
            line=getattr(source, "line", 0),
            col=getattr(source, "col", 0),
        )

    def _fresh_collection_index(
        self,
        context: CheckContext,
        path: tuple[str | int, ...],
    ) -> str:
        """Choose a deterministic comprehension index absent from ``context``."""
        stem = "_".join(str(item) for item in path if isinstance(item, str))
        base = f"__{stem or 'collection'}_index"
        occupied = {local.name for local in context.locals}
        name = base
        suffix = 0
        while name in occupied:
            suffix += 1
            name = f"{base}_{suffix}"
        return name

    def _shared_local(self, stem: str, type_: TypeExpr, context: CheckContext) -> Local:
        """A fresh local that shares one lowered value by name.

        Parameters
        ----------
        stem : str
            What the local holds, as part of its name.
        type_ : TypeExpr
            The type of the value it binds.
        context : CheckContext
            Value bindings in scope, none of which the name may repeat.

        Returns
        -------
        Local
            A local named ``__<stem>_<n>``, unique across the module.
        """
        occupied = {local.name for local in context.locals}
        while True:
            self._shared_serial += 1
            name = f"__{stem}_{self._shared_serial}"
            if name not in occupied:
                return Local(name, type_)

    def _let(
        self,
        bindings: Sequence[LetBinding],
        body: Value,
        context: CheckContext,
        authored: object,
    ) -> Value:
        """Wrap a lowered body in the bindings it reads.

        Parameters
        ----------
        bindings : Sequence[LetBinding]
            The bindings in evaluation order, each in scope in the later
            ones and in ``body``.
        body : Value
            The lowered body.
        context : CheckContext
            Value bindings in scope around the whole term.
        authored : object
            The source node to blame.

        Returns
        -------
        Value
            ``body`` itself when there are no bindings, else one
            :class:`LetValue` holding these bindings followed by those
            of a body that is itself a ``LetValue``.

        Raises
        ------
        QiecDiagnosticError
            If the kernel rejects the bound term.
        """
        if not bindings:
            return body
        collected = tuple(bindings)
        if isinstance(body, LetValue):
            collected = (*collected, *body.bindings)
            body = body.body
        inner = context
        for binding in collected:
            inner = inner.extend(binding.binder)
        return LetValue(collected, body, self._value_type(body, inner, authored))

    def _lower_map(
        self,
        authored: surface.LetExprCall,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None,
        path: tuple[str | int, ...],
    ) -> Value:
        """Lower a pure leading-axis ``map`` to a QIEC comprehension.

        The body reads its item as the source at the comprehension's
        index. A source other than a variable or literal is lowered once
        and shared through a :class:`LetValue`, so the comprehension body
        indexes a binding instead of holding, and evaluating once per
        item, a copy of the source for every read of the item.

        Parameters
        ----------
        authored : surface.LetExprCall
            The ``map(xs, x -> body)`` application.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.
        path : tuple[str | int, ...]
            Structural path of the application.

        Returns
        -------
        Value
            The comprehension, wrapped in the source's binding when the
            source is shared.

        Raises
        ------
        QiecDiagnosticError
            If the application does not take a non-scalar tensor and a
            lambda.
        """
        if len(authored.args) != 2:
            self._fail(
                authored,
                f"map takes 2 arguments, got {len(authored.args)}",
                code="qiec-primitive",
            )
        source_expr = authored.args[0]
        source = self._lower_value(
            source_expr, scope, context, static_bindings, (*path, "source")
        )
        source_type = self._value_type(source, context, source_expr)
        source_shape = tensor_shape(source_type)
        if source_shape is None or not source_shape[1]:
            self._fail(
                source_expr,
                f"map expects a non-scalar tensor, got {self._render(source_type)}",
                code="qiec-primitive",
            )
        function = self._collection_lambda(authored, 1)
        bindings: list[LetBinding] = []
        outer = context
        if not _atomic(source_expr):
            shared = self._shared_local("map_source", source_type, context)
            bindings.append(LetBinding(shared, source))
            outer = context.extend(shared)
            source_expr = surface.LetExprVar(
                name=shared.name, line=authored.line, col=authored.col
            )
        index = Local(self._fresh_collection_index(outer, path), INT)
        indexed = self._indexed_collection_item(
            source_expr,
            surface.LetExprVar(
                name=index.name,
                line=authored.line,
                col=authored.col,
            ),
        )
        body_expr = substitute_let_expr(function.body, {function.param: indexed})
        body_context = outer.extend(index)
        body = self._lower_value(
            body_expr,
            scope,
            body_context,
            static_bindings,
            (*path, "body"),
        )
        body_type = self._value_type(body, body_context, body_expr)
        inner = tensor_shape(body_type)
        result = (
            tensor_type(body_type, (source_shape[1][0],))
            if inner is None
            else tensor_type(inner[0], (source_shape[1][0], *inner[1]))
        )
        return self._let(
            bindings,
            Comprehension(index, source_shape[1][0], body, result),
            context,
            authored,
        )

    def _lower_collection_value(
        self,
        authored: surface.LetExprCall,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None,
        path: tuple[str | int, ...],
        expected: TypeExpr | None,
    ) -> Value:
        """Lower the checked pure collection operations."""
        if authored.func == "map":
            return self._lower_map(authored, scope, context, static_bindings, path)
        if authored.func == "length":
            if len(authored.args) != 1:
                self._fail(
                    authored,
                    f"length takes 1 argument, got {len(authored.args)}",
                    code="qiec-primitive",
                )
            source = self._lower_value(
                authored.args[0],
                scope,
                context,
                static_bindings,
                (*path, "source"),
            )
            source_type = self._value_type(source, context, authored.args[0])
            shape = tensor_shape(source_type)
            if shape is None or not shape[1]:
                self._fail(
                    authored.args[0],
                    f"length expects a non-scalar tensor, got "
                    f"{self._render(source_type)}",
                    code="qiec-primitive",
                )
            extent = shape[1][0]
            if not isinstance(extent, IndexLiteral):
                self._fail(
                    authored,
                    "length of a statically polymorphic tensor is an index term, "
                    "not an Int value; specialize the extent before using length",
                    code="qiec-primitive",
                )
            return LiteralValue(int(extent.value), INT)
        if authored.func == "fold":
            return self._lower_fold(
                authored, scope, context, static_bindings, path, expected
            )
        if authored.func == "logsumexp_over":
            if len(authored.args) != 2:
                self._fail(
                    authored,
                    f"logsumexp_over takes 2 arguments, got {len(authored.args)}",
                    code="qiec-primitive",
                )
            mapped = self._lower_map(
                authored.with_(func="map"),
                scope,
                context,
                static_bindings,
                (*path, "map"),
            )
            mapped_type = self._value_type(mapped, context, authored)
            shape = tensor_shape(mapped_type)
            if shape is None or len(shape[1]) != 1 or shape[0] != REAL:
                self._fail(
                    authored,
                    "logsumexp_over's lambda must return one Real per item",
                    code="qiec-primitive",
                )
            return Reduction("logsumexp", mapped, REAL)
        if authored.func == "filter":
            self._fail(
                authored,
                "filter has data-dependent output length, which fixed-shape QIEC "
                "tensors cannot represent; use a Boolean mask and a reduction",
                code="qiec-collection:filter:dynamic-shape",
            )
        self._fail(authored, f"unknown collection builtin {authored.func!r}")

    def _lower_fold(
        self,
        authored: surface.LetExprCall,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None,
        path: tuple[str | int, ...],
        expected: TypeExpr | None,
    ) -> Value:
        """Unroll a ``fold`` over a statically sized leading axis.

        ``fold(xs, init, acc -> x -> body)`` applies the body at each
        item in order, the accumulator reading the previous step's
        result. The body is lowered once per accumulator type, over a
        local for the accumulator and one for the item; each step
        instantiates that lowered body at the previous accumulator and
        the step's item, and every intermediate accumulator is bound once
        by a :class:`LetValue` that each of the body's reads refers to.
        The lowered term and the lowering's work thus grow linearly in
        the axis's extent however often the body reads the accumulator.
        A source or initial value other than a variable or literal is
        bound once the same way.

        Parameters
        ----------
        authored : surface.LetExprCall
            The ``fold`` application.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.
        path : tuple[str | int, ...]
            Structural path of the application.
        expected : TypeExpr or None
            The type the position calls for, passed to the final step.

        Returns
        -------
        Value
            The final accumulator, wrapped in the bindings it reads.

        Raises
        ------
        QiecDiagnosticError
            If the application is not ``fold`` of a statically sized
            tensor, an initial value, and a curried lambda.
        """
        if len(authored.args) != 3:
            self._fail(
                authored,
                f"fold takes 3 arguments, got {len(authored.args)}",
                code="qiec-primitive",
            )
        source_expr, initial_expr, _function = authored.args
        source = self._lower_value(
            source_expr,
            scope,
            context,
            static_bindings,
            (*path, "source"),
        )
        source_type = self._value_type(source, context, source_expr)
        shape = tensor_shape(source_type)
        if shape is None or not shape[1]:
            self._fail(
                source_expr,
                f"fold expects a non-scalar tensor, got {self._render(source_type)}",
                code="qiec-primitive",
            )
        extent = shape[1][0]
        if not isinstance(extent, IndexLiteral):
            self._fail(
                authored,
                "fold requires a statically known leading extent",
                code="qiec-primitive",
            )
        accumulator = self._collection_lambda(authored, 2)
        if not isinstance(accumulator.body, surface.LetExprLambda):
            self._fail(
                accumulator.body,
                "fold's lambda is curried as `acc -> item -> body`",
                code="qiec-primitive",
            )
        item_lambda = accumulator.body
        count = int(extent.value)
        if count == 0:
            return self._lower_value(
                initial_expr, scope, context, static_bindings, (*path, "init"), expected
            )
        bindings: list[LetBinding] = []
        inner = context

        def share(value: Value, value_type: TypeExpr) -> Var:
            """Bind a value to a fresh local and read it through that local.

            Parameters
            ----------
            value : Value
                The value.
            value_type : TypeExpr
                Its type.

            Returns
            -------
            Var
                The reference to the new local.
            """
            nonlocal inner
            local = self._shared_local("fold_acc", value_type, inner)
            bindings.append(LetBinding(local, value))
            inner = inner.extend(local)
            return Var(local)

        if not _atomic(source_expr):
            source_local = self._shared_local("fold_source", source_type, inner)
            bindings.append(LetBinding(source_local, source))
            inner = inner.extend(source_local)
            source_expr = surface.LetExprVar(
                name=source_local.name, line=authored.line, col=authored.col
            )
        base = inner
        initial = self._lower_value(
            initial_expr, scope, context, static_bindings, (*path, "init")
        )
        current_type = self._value_type(initial, context, initial_expr)
        current: Value = (
            initial
            if isinstance(initial, Var | LiteralValue)
            else share(initial, current_type)
        )
        # The body lowered over an accumulator of each type it meets, and
        # separately for the last step, which the position's expected
        # type may refine.
        bodies: dict[tuple[TypeExpr, bool], tuple[Local, Local, Value, TypeExpr]] = {}
        item_type: TypeExpr | None = None
        result: Value | None = None
        for position in range(count):
            item_expr = self._indexed_collection_item(
                source_expr,
                surface.LetExprLiteral(
                    value=float(position),
                    integral=True,
                    line=authored.line,
                    col=authored.col,
                ),
            )
            item = self._lower_value(
                item_expr, scope, base, static_bindings, (*path, "items", position)
            )
            if item_type is None:
                item_type = self._value_type(item, base, item_expr)
            last = position == count - 1
            entry = bodies.get((current_type, last))
            if entry is None:
                state = self._shared_local("fold_state", current_type, base)
                element = self._shared_local("fold_item", item_type, base.extend(state))
                body_context = base.extend(state).extend(element)
                renamed: dict[str, surface.LetExprNode] = {
                    accumulator.param: surface.LetExprVar(
                        name=state.name, line=authored.line, col=authored.col
                    )
                }
                renamed[item_lambda.param] = surface.LetExprVar(
                    name=element.name, line=authored.line, col=authored.col
                )
                body_expr = substitute_let_expr(item_lambda.body, renamed)
                body = self._lower_value(
                    body_expr,
                    scope,
                    body_context,
                    static_bindings,
                    (*path, "body"),
                    expected if last else None,
                )
                entry = (
                    state,
                    element,
                    body,
                    self._value_type(body, body_context, body_expr),
                )
                bodies[(current_type, last)] = entry
            state, element, body, body_type = entry
            step = _replace_locals(body, {state: current, element: item})
            if last:
                result = step
            else:
                current = share(step, body_type)
                current_type = body_type
        assert result is not None
        return self._let(bindings, result, context, authored)

    def _goal_weight_value(
        self,
        authored: surface.LetExprMethodCall,
        context: CheckContext,
        path: tuple[str | int, ...],
    ) -> Value | None:
        """Read ``chart.goal_weight()`` on a deduction's answer as a real.

        Parameters
        ----------
        authored : surface.LetExprMethodCall
            The method call.
        context : CheckContext
            The bindings in scope, where the chart's answer is a local.
        path : tuple[str | int, ...]
            The structural path of the value.

        Returns
        -------
        Value | None
            The answer as a ``Real`` when the call is ``goal_weight()``
            on a bound deduction answer: a log weight's value, or a
            count's; ``None`` for any other method call.

        Raises
        ------
        QiecDiagnosticError
            If the answer is a Boolean, which has no real value.
        """
        if (
            authored.method != "goal_weight"
            or authored.args
            or not isinstance(authored.receiver, surface.LetExprVar)
        ):
            return None
        name = authored.receiver.name
        local = next(
            (item for item in reversed(context.locals) if item.name == name), None
        )
        if local is None or local.type not in (LOG_WEIGHT, INT, BOOL):
            return None
        if local.type == BOOL:
            self._fail(
                authored,
                f"{name!r} is the answer of a Boolean deduction, which has no "
                "real value to score",
                code="qiec-program",
            )
        conversion = "weight_value" if local.type == LOG_WEIGHT else "int_to_real"
        return self._primitive(
            conversion, (Var(local),), authored, (*path, "goal_weight", name)
        )

    def _lower_tensor_literal(
        self,
        authored: surface.LetExprList,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None,
        path: tuple[str | int, ...],
        expected: TypeExpr | None,
        element_hint: TypeExpr | None = None,
    ) -> TensorValue:
        """Lower a list literal to a tensor whose leading dimension it fixes.

        Parameters
        ----------
        authored : surface.LetExprList
            The literal.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.
        path : tuple[str | int, ...]
            Structural path of the literal.
        expected : TypeExpr or None
            The ``Tensor`` type the position calls for, when fixed. Its
            element type and inner dimensions type the entries.
        element_hint : TypeExpr or None
            The element type the entries should take when no type is
            expected, so integral literals lower as ``Real`` where a
            family parameter is real.

        Returns
        -------
        TensorValue
            The tensor, typed ``Tensor[E]([n, ...])`` with ``n`` the
            entry count.

        Raises
        ------
        QiecDiagnosticError
            If the literal is empty with no expected type to fix its
            element, the entries do not all have one type, or the
            expected type is not a tensor.
        """
        expected_split = tensor_shape(expected) if expected is not None else None
        if expected is not None and expected_split is None:
            self._fail(
                authored,
                f"a list literal is a Tensor, but {self._render(expected)} is "
                "expected here",
                code="qiec-primitive",
            )
        if expected_split is not None:
            element, dimensions = expected_split
            leading = dimensions[0]
            if isinstance(leading, IndexLiteral) and leading.value != len(
                authored.items
            ):
                self._fail(
                    authored,
                    f"list literal has {len(authored.items)} entries where "
                    f"{self._render(expected)} is expected",
                    code="qiec-primitive",
                )
            entry_hint: TypeExpr = (
                element
                if len(dimensions) == 1
                else tensor_type(element, dimensions[1:])
            )
        elif element_hint is not None:
            entry_hint = element_hint
        elif not authored.items:
            self._fail(
                authored,
                "an empty list literal needs an annotation or a position fixing "
                "its Tensor type",
                code="qiec-primitive",
            )
        else:
            entry_hint = self._tensor_entry_hint(
                authored, scope, context, static_bindings
            )
        items = tuple(
            self._lower_tensor_literal(
                item,
                scope,
                context,
                static_bindings,
                (*path, position),
                None,
                entry_hint,
            )
            if isinstance(item, surface.LetExprList)
            and tensor_shape(entry_hint) is None
            else self._lower_value(
                item, scope, context, static_bindings, (*path, position), entry_hint
            )
            for position, item in enumerate(authored.items)
        )
        types = tuple(
            self._value_type(item, context, source)
            for item, source in zip(items, authored.items, strict=True)
        )
        if any(item_type != types[0] for item_type in types[1:]):
            self._fail(
                authored,
                "list literal entries have types "
                + ", ".join(self._render(item_type) for item_type in types)
                + "; a Tensor's entries share one type",
                code="qiec-primitive",
            )
        entry_type = types[0] if types else entry_hint
        inner = tensor_shape(entry_type)
        if inner is None:
            result_type = tensor_type(entry_type, (IndexLiteral(len(items), NAT),))
        else:
            result_type = tensor_type(
                inner[0], (IndexLiteral(len(items), NAT), *inner[1])
            )
        value = TensorValue(items, result_type)
        self._value_type(value, context, authored)
        return value

    def _tensor_entry_hint(
        self,
        authored: surface.LetExprList,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None,
    ) -> TypeExpr:
        """Choose the entry type of an unannotated list literal.

        Parameters
        ----------
        authored : surface.LetExprList
            The literal, with at least one entry.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.

        Returns
        -------
        TypeExpr
            ``Real`` when the entries are numeric literals of which any
            is fractional, so ``[1, 2.5]`` is a real vector; otherwise
            the type of the first entry lowered on its own.
        """
        literals = [
            item for item in authored.items if isinstance(item, surface.LetExprLiteral)
        ]
        if literals and any(not item.integral for item in literals):
            return REAL
        first = authored.items[0]
        if isinstance(first, surface.LetExprList):
            return self._lower_tensor_literal(
                first, scope, context, static_bindings, (), None
            ).result_type
        return self._value_type(
            self._lower_value(first, scope, context, static_bindings), context, first
        )

    def _lower_family_application(
        self,
        authored: surface.LetExprCall,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None,
        path: tuple[str | int, ...],
        expected: TypeExpr | None = None,
    ) -> DistributionValue:
        """Lower ``Family(args...)`` to a distribution construction.

        Positional arguments fill the family's parameters in registry
        order, so ``Normal(0.0, 1.0)`` supplies ``loc`` and ``scale``.

        Parameters
        ----------
        authored : surface.LetExprCall
            The application.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.
        path : tuple[str | int, ...]
            Structural path of the application.
        expected : TypeExpr or None
            The ``Sampleable`` type the position calls for, when fixed;
            it supplies the event shape of a family no parameter fixes.

        Returns
        -------
        DistributionValue
            The construction, typed by the family's sample type.

        Raises
        ------
        QiecDiagnosticError
            If more arguments are given than the family has parameters,
            or the construction fails to check.
        """
        record = FAMILIES[authored.func]
        if len(authored.args) > len(record.parameters):
            self._fail(
                authored,
                f"family {record.name!r} takes at most {len(record.parameters)} "
                f"parameters ({', '.join(record.parameter_names)})",
                code="qiec-distribution",
            )
        arguments: list[tuple[str, Value]] = []
        for position, (parameter, argument) in enumerate(
            zip(record.parameters, authored.args, strict=False)
        ):
            numeric = (
                parameter.constraint not in ("boolean", "sampleable", "transform")
                and "integer" not in parameter.constraint
            )
            if parameter.rank > 0 and isinstance(argument, surface.LetExprList):
                value: Value = self._lower_tensor_literal(
                    argument,
                    scope,
                    context,
                    static_bindings,
                    (*path, position),
                    None,
                    REAL if numeric else None,
                )
            else:
                value = self._lower_value(
                    argument,
                    scope,
                    context,
                    static_bindings,
                    (*path, position),
                    REAL if numeric and parameter.rank == 0 else None,
                )
            arguments.append((parameter.name, value))
        if record.event_rank == 0:
            result_type: TypeExpr = sampleable_type(record.element)
        else:
            result_type = sampleable_type(
                self._event_type(record, arguments, context, authored, expected)
            )
        value = DistributionValue(
            record.id,
            record.name,
            tuple(arguments),
            result_type,
            self._origin(authored, path, "distribution"),
        )
        self._value_type(value, context, authored)
        return value

    def _event_type(
        self,
        record: DistributionFamily,
        arguments: Sequence[tuple[str, Value]],
        context: CheckContext,
        authored: surface.LetExprCall,
        expected: TypeExpr | None,
    ) -> TypeExpr:
        """The sample type of a family whose events are tensors.

        Parameters
        ----------
        record : DistributionFamily
            The family, with a positive event rank.
        arguments : Sequence[tuple[str, Value]]
            The lowered parameters.
        context : CheckContext
            Value bindings in scope.
        authored : surface.LetExprCall
            The application, for diagnostics.
        expected : TypeExpr or None
            The ``Sampleable`` type the position calls for, when fixed.

        Returns
        -------
        TypeExpr
            ``Tensor[E](event)``: the event shape is the trailing
            dimensions of the parameter the registry names as its source,
            or, for a family whose event shape no parameter carries, the
            shape the expected type spells.

        Raises
        ------
        QiecDiagnosticError
            If neither a source parameter nor the expected type fixes
            the event shape.
        """
        supplied = record.event_argument(arguments)
        if supplied is not None:
            shape = tensor_shape(self._value_type(supplied[1], context, authored))
            if shape is not None and len(shape[1]) >= record.event_rank:
                event = shape[1][len(shape[1]) - record.event_rank :]
                return tensor_type(record.element, event)
        sampled = sampled_element(expected) if expected is not None else None
        expected_shape = tensor_shape(sampled) if sampled is not None else None
        if (
            expected_shape is not None
            and expected_shape[0] == record.element
            and len(expected_shape[1]) == record.event_rank
        ):
            return tensor_type(record.element, expected_shape[1])
        if record.event_sources:
            self._fail(
                authored,
                f"family {record.name!r} needs parameter "
                f"{' or '.join(repr(name) for name in record.event_sources)} as a "
                "Tensor to fix its event shape",
                code="qiec-distribution",
            )
        self._fail(
            authored,
            f"family {record.name!r} samples a rank-{record.event_rank} tensor "
            "whose shape no parameter carries; annotate the position with its "
            "Sampleable[Tensor[...]] type",
            code="qiec-distribution",
        )

    def _lower_site(
        self, authored: surface.LetExprCall, expected: TypeExpr | None
    ) -> SiteValue:
        """Lower ``site("label")`` at the type its position calls for.

        Parameters
        ----------
        authored : surface.LetExprCall
            The application.
        expected : TypeExpr or None
            The ``Site`` type the surrounding term expects.

        Returns
        -------
        SiteValue
            The site.

        Raises
        ------
        QiecDiagnosticError
            If the argument is not one string literal, or the position
            fixes no ``Site`` type to give the site.
        """
        if len(authored.args) != 1 or not isinstance(
            authored.args[0], surface.LetExprString
        ):
            self._fail(
                authored,
                'site takes one string literal, as in site("x")',
                code="qiec-distribution",
            )
        if not (
            isinstance(expected, TypeApplication)
            and expected.constructor == SITE_CONSTRUCTOR
        ):
            self._fail(
                authored,
                "site needs a position that fixes its type, such as a request "
                "argument or a binding annotated `: Site[T]`",
                code="qiec-distribution",
            )
        return SiteValue(authored.args[0].value, expected)

    def _lower_log_density(
        self,
        authored: surface.LetExprCall,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None,
        path: tuple[str | int, ...],
    ) -> LogDensity:
        """Lower ``log_prob(d, x)`` to a log-density evaluation.

        Parameters
        ----------
        authored : surface.LetExprCall
            The application.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.
        path : tuple[str | int, ...]
            Structural path of the application.

        Returns
        -------
        LogDensity
            The evaluation, of type ``LogWeight``.

        Raises
        ------
        QiecDiagnosticError
            If the arity is wrong or the evaluation fails to check.
        """
        if len(authored.args) != 2:
            self._fail(
                authored,
                "log_prob takes a distribution and a value",
                code="qiec-distribution",
            )
        sampleable = self._lower_value(
            authored.args[0], scope, context, static_bindings, (*path, 0)
        )
        element = sampled_element(
            self._value_type(sampleable, context, authored.args[0])
        )
        point = self._lower_value(
            authored.args[1], scope, context, static_bindings, (*path, 1), element
        )
        value = LogDensity(
            sampleable, point, self._origin(authored, path, "log-density")
        )
        self._value_type(value, context, authored)
        return value

    def _value_type(
        self,
        value: Value,
        context: CheckContext,
        authored: object,
    ) -> TypeExpr:
        """Infer a lowered value's type, blaming its source on failure.

        Parameters
        ----------
        value : Value
            The lowered value.
        context : CheckContext
            Value bindings in scope.
        authored : object
            The source node to blame.

        Returns
        -------
        TypeExpr
            The kernel's inferred type.

        Raises
        ------
        QiecDiagnosticError
            If the kernel rejects the value.
        """
        try:
            return infer_value(value, self.registry, context)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(authored, error, fallback="qiec-primitive")

    def _numeric_primitive(
        self,
        authored: object,
        operator: str,
        type_: TypeExpr,
        table: Mapping[TypeExpr, str],
    ) -> str:
        """Choose the primitive an operator denotes at an operand type.

        Parameters
        ----------
        authored : object
            The source node to blame.
        operator : str
            The operator, for the diagnostic.
        type_ : TypeExpr
            The operand type both sides share.
        table : Mapping[TypeExpr, str]
            Operand type to primitive name.

        Returns
        -------
        str
            The primitive's nominal name.

        Raises
        ------
        QiecDiagnosticError
            If the operator is not defined at the type.
        """
        name = table.get(type_)
        if name is None:
            admitted = ", ".join(self._render(item) for item in table)
            self._fail(
                authored,
                f"`{operator}` is not defined at {self._render(type_)}; it takes "
                f"{admitted}",
                code="qiec-primitive",
            )
        return name

    def _builtin_primitive(
        self,
        authored: surface.LetExprCall,
        types: tuple[TypeExpr, ...],
    ) -> str:
        """Resolve a builtin application to a primitive by its argument types.

        Parameters
        ----------
        authored : surface.LetExprCall
            The application, read for its name and blamed on failure.
        types : tuple[TypeExpr, ...]
            The argument types, in order.

        Returns
        -------
        str
            The primitive's nominal name.

        Raises
        ------
        QiecDiagnosticError
            If the name is a computation rather than a builtin, is no
            builtin at all, or is applied to the wrong number or types of
            arguments.
        """
        overloads = _BUILTIN_PRIMITIVES.get(authored.func)
        if overloads is None:
            if authored.func in HOST_ONLY_BUILTINS:
                self._fail(
                    authored,
                    f"builtin {authored.func!r} is available only to native eager "
                    "execution and has no checked QIEC lowering",
                    code="qiec-builtin:host-only",
                )
            if authored.func in self.computation_signatures:
                self._fail(
                    authored,
                    f"{authored.func!r} is a computation, not a builtin; call it "
                    f"with `let x <- {authored.func}(...)`",
                    code="qiec-primitive",
                )
            self._fail(
                authored,
                f"unknown builtin {authored.func!r}",
                code="qiec-primitive",
            )
        name = overloads.get(types)
        if name is None:
            rendered = ", ".join(self._render(item) for item in types)
            self._fail(
                authored,
                f"builtin {authored.func!r} is not defined at ({rendered})",
                code="qiec-primitive",
            )
        return name

    def _primitive(
        self,
        name: str,
        arguments: tuple[Value, ...],
        authored: object,
        path: tuple[str | int, ...],
        shape: tuple[IndexTerm, ...] | None = None,
    ) -> PrimitiveApplication:
        """Build a primitive application from the registry's signature.

        Parameters
        ----------
        name : str
            The primitive's nominal name.
        arguments : tuple[Value, ...]
            The lowered arguments.
        authored : object
            The source node, for provenance.
        path : tuple[str | int, ...]
            Structural path of the application.
        shape : tuple[IndexTerm, ...] or None
            The shape the application broadcasts over, when an argument
            is a tensor.

        Returns
        -------
        PrimitiveApplication
            The application at the registry's result type, under the
            broadcast shape when there is one.
        """
        signature = primitive(name)
        result: TypeExpr = (
            signature.result if shape is None else tensor_type(signature.result, shape)
        )
        return PrimitiveApplication(
            signature.id,
            name,
            arguments,
            result,
            self._origin(authored, path, "primitive"),
        )

    def _promoted(
        self,
        value: Value,
        type_: TypeExpr,
        other: TypeExpr,
        authored: object,
        path: tuple[str | int, ...],
    ) -> tuple[Value, TypeExpr]:
        """Convert an integer operand to a real beside a real operand.

        An operator over an ``Int`` and a ``Real`` applies at ``Real``,
        the integer converted first; the conversion is explicit in the
        kernel term as ``int_to_real``, applied over the operand's shape
        when it is a tensor.

        Parameters
        ----------
        value : Value
            The operand.
        type_ : TypeExpr
            Its type.
        other : TypeExpr
            The other operand's type.
        authored : object
            The source node.
        path : tuple[str | int, ...]
            The operand's path.

        Returns
        -------
        tuple[Value, TypeExpr]
            The operand and its type, converted when it is integral and
            the other operand is real.
        """
        own = tensor_shape(type_)
        other_split = tensor_shape(other)
        own_element = own[0] if own is not None else type_
        other_element = other_split[0] if other_split is not None else other
        if own_element != INT or other_element != REAL:
            return value, type_
        shape = tuple(own[1]) if own is not None else None
        converted = self._primitive("int_to_real", (value,), authored, path, shape)
        return converted, (REAL if shape is None else tensor_type(REAL, shape))

    def _broadcast_element(
        self,
        authored: object,
        operator: str,
        types: tuple[TypeExpr, ...],
    ) -> tuple[TypeExpr, tuple[IndexTerm, ...] | None]:
        """The element type an operator applies at, over a shared shape.

        Parameters
        ----------
        authored : object
            The source node to blame.
        operator : str
            The operator, for the diagnostic.
        types : tuple[TypeExpr, ...]
            The operand types, scalars or tensors.

        Returns
        -------
        tuple[TypeExpr, tuple[IndexTerm, ...] | None]
            The element type every operand shares, and the shape of the
            tensor operands, ``None`` when all are scalars.

        Raises
        ------
        QiecDiagnosticError
            If the operands' element types or tensor shapes disagree.
        """
        elements: list[TypeExpr] = []
        shape: tuple[IndexTerm, ...] | None = None
        for item in types:
            split = tensor_shape(item)
            if split is None:
                elements.append(item)
                continue
            elements.append(split[0])
            if shape is not None and tuple(split[1]) != shape:
                self._fail(
                    authored,
                    f"operands of `{operator}` are tensors of differing shapes",
                    code="qiec-primitive",
                )
            shape = tuple(split[1])
        if any(item != elements[0] for item in elements[1:]):
            rendered = " and ".join(self._render(item) for item in types)
            self._fail(
                authored,
                f"operands of `{operator}` have types {rendered}; both sides must "
                "agree",
                code="qiec-primitive",
            )
        return elements[0], shape

    def _lower_gather(
        self,
        source: Value,
        source_type: TypeExpr,
        authored: surface.LetExprIndex,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None,
        path: tuple[str | int, ...],
    ) -> Value:
        """Lower ``t[i][j]`` on a tensor to nested selections.

        Parameters
        ----------
        source : Value
            The tensor.
        source_type : TypeExpr
            Its type.
        authored : surface.LetExprIndex
            The indexing expression.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.
        path : tuple[str | int, ...]
            Structural path of the expression.

        Returns
        -------
        Value
            The selection, one gather per index.

        Raises
        ------
        QiecDiagnosticError
            If an index is not an ``Int`` or a tensor of them, or more
            indices are given than the tensor has axes.
        """
        indices = tuple(
            self._lower_value(
                index, scope, context, static_bindings, (*path, "index", position), INT
            )
            for position, index in enumerate(authored.indices)
        )
        index_types = tuple(
            self._value_type(index_value, context, index)
            for index_value, index in zip(indices, authored.indices, strict=True)
        )
        if len(indices) > 1 and any(item != INT for item in index_types):
            return self._pointwise_gather(
                source,
                source_type,
                indices,
                index_types,
                authored,
                path,
                frozenset(local.name for local in context.locals),
            )
        value = source
        value_type = source_type
        for position, index in enumerate(authored.indices):
            split = tensor_shape(value_type)
            if split is None:
                self._fail(
                    authored,
                    f"index {position} selects from a value that is not a Tensor",
                    code="qiec-primitive",
                )
            element, dimensions = split
            index_value = indices[position]
            index_type = index_types[position]
            rest = dimensions[1:]
            if index_type == INT:
                result: TypeExpr = element if not rest else tensor_type(element, rest)
            else:
                index_shape = tensor_shape(index_type)
                if index_shape is None or index_shape[0] != INT:
                    self._fail(
                        index,
                        f"a tensor is indexed by an Int or a Tensor[Int], not "
                        f"{self._render(index_type)}",
                        code="qiec-primitive",
                    )
                result = tensor_type(element, (*index_shape[1], *rest))
            value = Gather(value, index_value, result)
            value_type = result
        return value

    def _pointwise_gather(
        self,
        source: Value,
        source_type: TypeExpr,
        indices: tuple[Value, ...],
        index_types: tuple[TypeExpr, ...],
        authored: surface.LetExprIndex,
        path: tuple[str | int, ...],
        occupied: frozenset[str],
    ) -> Value:
        """Lower ``t[e_1, ..., e_k]`` with index tensors to one entry per position.

        The subscript reads its indices together, one entry per position
        of the index tensors, as `_pointwise_selection` describes; this
        method reports a misfit at the subscript.

        Parameters
        ----------
        source : Value
            The tensor selected from.
        source_type : TypeExpr
            Its type.
        indices : tuple[Value, ...]
            The lowered indices, outermost axis first.
        index_types : tuple[TypeExpr, ...]
            Their types.
        authored : surface.LetExprIndex
            The subscript, for diagnostics.
        path : tuple[str | int, ...]
            Structural path of the subscript, which names the binders.
        occupied : frozenset[str]
            The names of the locals in scope.

        Returns
        -------
        Value
            The selection.

        Raises
        ------
        QiecDiagnosticError
            If the tensor has fewer axes than indices, an index is
            neither an ``Int`` nor a ``Tensor[Int]``, or the index
            tensors differ in shape.
        """
        try:
            return _pointwise_selection(
                source, source_type, indices, index_types, path, occupied
            )
        except _PointwiseGatherError as error:
            self._fail(authored, str(error), code="qiec-primitive")

    def _lower_factor(
        self,
        authored: surface.LetExprFactor,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None,
        path: tuple[str | int, ...],
    ) -> Value:
        """Lower a ``factor`` expression to nested comprehensions.

        Parameters
        ----------
        authored : surface.LetExprFactor
            The expression.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.
        path : tuple[str | int, ...]
            Structural path of the expression.

        Returns
        -------
        Value
            A comprehension per binder, innermost last; the case form is
            a comprehension whose body branches on the index.

        Raises
        ------
        QiecDiagnosticError
            If a binder's axis has no static extent, or the case labels
            do not cover the axis exactly.
        """
        binders = list(authored.binders)
        if not binders:
            self._fail(
                authored, "a factor names at least one binder", code="qiec-primitive"
            )
        if authored.cases:
            if len(binders) != 1:
                self._fail(
                    authored,
                    "a case-structured factor has one binder",
                    code="qiec-primitive",
                )
            binder = binders[0]
            extent = _fixed_extent(self._factor_extent(binder, authored))
            if extent is None:
                self._fail(
                    authored,
                    f"a case-structured factor needs a fixed extent, but binder "
                    f"{binder.var!r} ranges over the object parameter "
                    f"{getattr(binder.index, 'name', binder.index.kind)!r}",
                    code="qiec-primitive",
                )
            labels = sorted(case.label for case in authored.cases)
            if labels != list(range(extent)):
                self._fail(
                    authored,
                    f"factor cases label {labels}, not every index below {extent}",
                    code="qiec-primitive",
                )
            local = Local(binder.var, INT)
            inner = context.extend(local)
            ordered = sorted(authored.cases, key=lambda case: case.label)
            entries = tuple(
                self._lower_value(
                    case.value,
                    scope,
                    inner,
                    static_bindings,
                    (*path, "cases", case.label),
                )
                for case in ordered
            )
            types = [
                self._value_type(entry, inner, case.value)
                for entry, case in zip(entries, ordered, strict=True)
            ]
            if any(item != types[0] for item in types[1:]):
                self._fail(
                    authored, "factor cases differ in type", code="qiec-primitive"
                )
            inner_shape = tensor_shape(types[0])
            result = (
                tensor_type(types[0], (IndexLiteral(extent, NAT),))
                if inner_shape is None
                else tensor_type(
                    inner_shape[0], (IndexLiteral(extent, NAT), *inner_shape[1])
                )
            )
            return TensorValue(entries, result)
        body_context = context
        locals_: list[Local] = []
        for binder in binders:
            local = Local(binder.var, INT)
            locals_.append(local)
            body_context = body_context.extend(local)
        assert authored.body is not None
        value = self._lower_value(
            authored.body, scope, body_context, static_bindings, (*path, "body")
        )
        value_type = self._value_type(value, body_context, authored.body)
        for binder, local in reversed(list(zip(binders, locals_, strict=True))):
            extent = _extent(self._factor_extent(binder, authored))
            inner_shape = tensor_shape(value_type)
            result = (
                tensor_type(value_type, (extent,))
                if inner_shape is None
                else tensor_type(inner_shape[0], (extent, *inner_shape[1]))
            )
            value = Comprehension(local, extent, value, result)
            value_type = result
        return value

    def _factor_extent(
        self, binder: surface.LetFactorBinder, authored: object
    ) -> int | IndexVariable:
        """The extent of a factor binder's axis.

        Parameters
        ----------
        binder : surface.LetFactorBinder
            The binder.
        authored : object
            The source node to blame.

        Returns
        -------
        int | IndexVariable
            The axis's cardinality, or the index variable a program
            template binds for an object parameter's size.

        Raises
        ------
        QiecDiagnosticError
            If the axis names no object with a known extent.
        """
        index = binder.index
        if isinstance(index, surface.TypeName) and index.name.isdigit():
            return int(index.name)
        info = _object_expr_info(index, self._program_objects)
        if info is None or info.extent is None:
            self._fail(
                authored,
                f"factor binder {binder.var!r} ranges over {getattr(index, 'name', index.kind)!r}, "
                "which has no static extent",
                code="qiec-primitive",
            )
        return info.extent

    def _lower_macro_application(
        self,
        macro: surface.LetExprLambda,
        authored: surface.LetExprCall,
        scope: Telescope,
        context: CheckContext,
        static_bindings: Mapping[str, StaticArgument] | None,
        path: tuple[str | int, ...],
        expected: TypeExpr | None,
    ) -> Value:
        """Lower the application of a lambda-bound name to its argument.

        An argument that is a variable or a literal, or one the body
        reads once outside any lambda or ``factor``, is substituted for
        the parameter. Any other argument is lowered once and bound by a
        :class:`LetValue` that every read of the parameter refers to, so
        nested applications of a body that reads its parameter several
        times lower to a term linear in the nesting depth.

        Parameters
        ----------
        macro : surface.LetExprLambda
            The lambda a `let` bound.
        authored : surface.LetExprCall
            The application.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.
        path : tuple[str | int, ...]
            Structural path of the application.
        expected : TypeExpr or None
            The type the position calls for, passed to the body.

        Returns
        -------
        Value
            The lowered body at the argument.

        Raises
        ------
        QiecDiagnosticError
            If the argument count differs from the parameter count.
        """
        if len(authored.args) != 1:
            self._fail(
                authored,
                f"{authored.func!r} takes 1 argument, got {len(authored.args)}",
                code="qiec-primitive",
            )
        argument = authored.args[0]
        if _atomic(argument) or not _reads_repeatedly(macro.param, macro.body):
            return self._lower_value(
                _substitute_let(macro.body, {macro.param: argument}),
                scope,
                context,
                static_bindings,
                path,
                expected,
            )
        value = self._lower_value(
            argument, scope, context, static_bindings, (*path, "argument")
        )
        shared = self._shared_local(
            "argument", self._value_type(value, context, argument), context
        )
        body = self._lower_value(
            _substitute_let(
                macro.body,
                {
                    macro.param: surface.LetExprVar(
                        name=shared.name, line=authored.line, col=authored.col
                    )
                },
            ),
            scope,
            context.extend(shared),
            static_bindings,
            path,
            expected,
        )
        return self._let((LetBinding(shared, value),), body, context, authored)

    @staticmethod
    def _render(type_: TypeExpr) -> str:
        """Render a type for a diagnostic.

        Parameters
        ----------
        type_ : TypeExpr
            The type.

        Returns
        -------
        str
            The surface spelling, as :func:`render_static` gives it.
        """
        return render_static(type_)

    def _lower_computation(
        self,
        authored: surface.QiecComputation,
        scope: Telescope,
        context: CheckContext,
        path: tuple[str | int, ...],
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> Computation:
        """Lower an authored computation.

        Parameters
        ----------
        authored : surface.QiecComputation
            The authored computation.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        path : tuple[str | int, ...]
            Structural path, entering derived identities and provenance.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.

        Returns
        -------
        Computation
            The lowered computation.

        Raises
        ------
        QiecDiagnosticError
            If the computation is malformed, names something undeclared, or fails to check as it is built.
        """
        if isinstance(authored, surface.QiecReturnComputation):
            return Return(
                self._lower_value(
                    authored.value,
                    scope,
                    context,
                    static_bindings,
                    (*path, "value"),
                    self._expected_result,
                )
            )
        if isinstance(authored, surface.QiecBindComputation):
            first = self._lower_computation(
                authored.first,
                scope,
                context,
                (*path, "first"),
                static_bindings,
            )
            try:
                inferred = infer_computation(first, self.registry, context)
            except (KernelError, TypeError, ValueError) as error:
                self._fail_kernel(authored.first, error)
            binder_type = (
                inferred.result
                if authored.binder.type_expr is None
                else self._lower_type(authored.binder.type_expr, scope, static_bindings)
            )
            binder = Local(authored.binder.name, binder_type)
            then_context = context.extend(binder)
            return Bind(
                binder,
                first,
                self._lower_computation(
                    authored.then,
                    scope,
                    then_context,
                    (*path, "then"),
                    static_bindings,
                ),
            )
        if isinstance(authored, surface.QiecSequenceComputation):
            first = self._lower_computation(
                authored.first,
                scope,
                context,
                (*path, "first"),
                static_bindings,
            )
            try:
                inferred = infer_computation(first, self.registry, context)
            except (KernelError, TypeError, ValueError) as error:
                self._fail_kernel(authored.first, error)
            name = self._fresh_sequence_name(context, path)
            binder = Local(name, inferred.result)
            return Bind(
                binder,
                first,
                self._lower_computation(
                    authored.then,
                    scope,
                    context.extend(binder),
                    (*path, "then"),
                    static_bindings,
                ),
            )
        if isinstance(authored, surface.QiecPerformComputation):
            request = authored.request
            instance = self.instances.get(request.instance)
            if instance is None:
                self._fail(
                    request,
                    f"unknown effect instance {request.instance!r}",
                    code="qiec-unhandled-effect",
                )
            definition = self.effects[instance.entry.effect.name]
            operation = next(
                (
                    item
                    for item in definition.operations
                    if item.name == request.operation
                ),
                None,
            )
            if operation is None:
                self._fail(
                    request,
                    f"effect {definition.ref.name!r} has no operation "
                    f"{request.operation!r}",
                    code="qiec-handler",
                )
            if len(request.static_arguments) != len(operation.telescope):
                self._fail(
                    request,
                    f"operation {operation.name!r} expects "
                    f"{len(operation.telescope)} static arguments",
                    code="qiec-kind",
                )
            static_arguments = tuple(
                self._lower_static_argument(argument, binder, scope, static_bindings)
                for argument, binder in zip(
                    request.static_arguments, operation.telescope, strict=True
                )
            )
            outer = instantiate_telescope(
                definition.telescope, instance.entry.effect.arguments
            )
            parameter_types, result_type = instantiate_operation(
                operation, static_arguments, outer
            )
            if len(request.arguments) != len(parameter_types):
                self._fail(
                    request,
                    f"operation {operation.name!r} takes {len(parameter_types)} "
                    f"argument(s), got {len(request.arguments)}",
                    code="qiec-kind",
                )
            return Perform(
                EffectRequest(
                    instance.entry.instance,
                    instance.entry.effect,
                    operation.id,
                    static_arguments,
                    tuple(
                        self._lower_value(
                            value,
                            scope,
                            context,
                            static_bindings,
                            (*path, position),
                            parameter_type,
                        )
                        for position, (value, parameter_type) in enumerate(
                            zip(request.arguments, parameter_types, strict=True)
                        )
                    ),
                    result_type,
                    SiteProvenance(self._origin(request, path, "effect-request")),
                )
            )
        if isinstance(authored, surface.QiecHandleComputation):
            instance = self.instances.get(authored.instance)
            if instance is None:
                self._fail(
                    authored,
                    f"unknown effect instance {authored.instance!r}",
                    code="qiec-handler",
                )
            handler = self.handlers.get(authored.handler.name)
            if handler is None:
                self._fail(
                    authored.handler,
                    f"unknown handler {authored.handler.name!r}",
                    code="qiec-handler",
                )
            if len(authored.handler.static_arguments) != len(handler.telescope):
                self._fail(
                    authored.handler,
                    f"handler {handler.name!r} expects {len(handler.telescope)} "
                    "static arguments",
                    code="qiec-kind",
                )
            static_arguments = tuple(
                self._lower_static_argument(argument, binder, scope, static_bindings)
                for argument, binder in zip(
                    authored.handler.static_arguments, handler.telescope, strict=True
                )
            )
            outer_expectation = self._expected_result
            self._expected_result = substitute_type(
                handler.input_type,
                instantiate_telescope(handler.telescope, static_arguments),
            )
            handled = self._lower_computation(
                authored.body,
                scope,
                context,
                (*path, "handled"),
                static_bindings,
            )
            self._expected_result = outer_expectation
            return Handle(
                instance.entry.instance,
                handler.id,
                handled,
                static_arguments,
            )
        if isinstance(authored, surface.QiecCaseComputation):
            return self._lower_case(authored, scope, context, path, static_bindings)
        if isinstance(authored, surface.QiecIfComputation):
            condition = self._lower_value(
                authored.condition,
                scope,
                context,
                static_bindings,
                (*path, "condition"),
            )
            condition_type = self._value_type(condition, context, authored.condition)
            if condition_type != BOOL:
                self._fail(
                    authored.condition,
                    f"if condition has type {self._render(condition_type)}; it "
                    "must be Bool",
                    code="qiec-primitive",
                )
            then = self._lower_computation(
                authored.then, scope, context, (*path, "then"), static_bindings
            )
            otherwise = self._lower_computation(
                authored.otherwise,
                scope,
                context,
                (*path, "otherwise"),
                static_bindings,
            )
            try:
                then_type = infer_computation(then, self.registry, context)
                otherwise_type = infer_computation(otherwise, self.registry, context)
            except (KernelError, TypeError, ValueError) as error:
                self._fail_kernel(authored, error)
            if then_type.result != otherwise_type.result:
                self._fail(
                    authored,
                    "if branches return "
                    f"{self._render(then_type.result)} and "
                    f"{self._render(otherwise_type.result)}; both must agree",
                    code="qiec-primitive",
                )
            return If(condition, then, otherwise)
        if isinstance(authored, surface.QiecPureBinding):
            if isinstance(authored.value, surface.LetExprLambda):
                if authored.binder.type_expr is not None:
                    self._fail(
                        authored.binder,
                        "a lambda macro cannot carry a first-order value type",
                        code="qiec-primitive",
                    )
                previous = self._lambda_macros.get(authored.binder.name)
                self._lambda_macros[authored.binder.name] = authored.value
                try:
                    return self._lower_computation(
                        authored.then,
                        scope,
                        context,
                        (*path, "then"),
                        static_bindings,
                    )
                finally:
                    if previous is None:
                        del self._lambda_macros[authored.binder.name]
                    else:
                        self._lambda_macros[authored.binder.name] = previous
            # A pure binding is a bind of a returned value. The core has
            # one sequencing form, and keeping it that way means every
            # later pass sees one shape rather than two that behave the
            # same.
            annotated = (
                None
                if authored.binder.type_expr is None
                else self._lower_type(authored.binder.type_expr, scope, static_bindings)
            )
            value = self._lower_value(
                authored.value,
                scope,
                context,
                static_bindings,
                (*path, "value"),
                annotated,
            )
            try:
                inferred = infer_value(value, self.registry, context)
            except (KernelError, TypeError, ValueError) as error:
                self._fail_kernel(authored.value, error)
            binder_type = (
                inferred
                if authored.binder.type_expr is None
                else self._lower_type(authored.binder.type_expr, scope, static_bindings)
            )
            binder = Local(authored.binder.name, binder_type)
            return Bind(
                binder,
                Return(value),
                self._lower_computation(
                    authored.then,
                    scope,
                    context.extend(binder),
                    (*path, "then"),
                    static_bindings,
                ),
            )
        if isinstance(authored, surface.QiecResumeComputation):
            return Resume(
                (
                    LiteralValue(None, UNIT)
                    if authored.value is None
                    else self._lower_value(
                        authored.value,
                        scope,
                        context,
                        static_bindings,
                        (*path, "value"),
                    )
                ),
                self._origin(authored, path, "resume"),
            )
        if isinstance(authored, surface.QiecInstanceComputation):
            return self._lower_local_instance(
                authored, scope, context, path, static_bindings
            )
        if isinstance(authored, surface.QiecCallComputation):
            return self._lower_call(authored, scope, context, path, static_bindings)
        self._fail(authored, "unknown QIEC computation")

    def _lower_local_instance(
        self,
        authored: surface.QiecInstanceComputation,
        scope: Telescope,
        context: CheckContext,
        path: tuple[str | int, ...],
        static_bindings: Mapping[str, StaticArgument] | None,
    ) -> NewInstance:
        """Lower a scoped allocation and its body.

        The instance identity is derived from the module and the lexical
        path, so the same allocation site yields the same instance on
        every run while two sites of one interface stay distinct. The
        binder is registered only while the body is lowered, which is
        what keeps a reference outside the body from resolving.

        Parameters
        ----------
        authored : surface.QiecInstanceComputation
            The source allocation.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        path : tuple[str | int, ...]
            Structural path of this allocation, entering its identity.
        static_bindings : Mapping[str, StaticArgument] or None
            Static bindings from an enclosing case refinement.

        Returns
        -------
        NewInstance
            The lowered allocation.

        Raises
        ------
        QiecDiagnosticError
            If the interface is unknown, wrongly applied, or the binder
            shadows an instance already in scope. Shadowing is rejected
            because a qualified operation would otherwise resolve to
            whichever instance happened to be innermost.
        """
        effect = self._lower_effect_ref(authored.effect, scope, static_bindings)
        if authored.name in self.instances:
            self._fail(
                authored,
                f"local instance {authored.name!r} shadows an instance already "
                f"in scope",
                code="qiec-handler",
            )
        entry = instantiate_effect(
            effect,
            module=self.source.module_name,
            lexical_path=path,
        )
        self.instances[authored.name] = NamedEffectInstance(
            authored.name,
            entry,
            self._origin(authored, path, "effect-instance"),
        )
        self.registry.instance_names[entry.instance] = authored.name
        try:
            body = self._lower_computation(
                authored.body,
                scope,
                context,
                (*path, "body"),
                static_bindings,
            )
        finally:
            # The binder is lexical, so it leaves scope with the body
            # whether lowering succeeded or failed.
            del self.instances[authored.name]
        return NewInstance(
            entry.instance,
            effect,
            body,
            self._origin(authored, path, "local-instance"),
        )

    def _lower_call(
        self,
        authored: surface.QiecCallComputation,
        scope: Telescope,
        context: CheckContext,
        path: tuple[str | int, ...],
        static_bindings: Mapping[str, StaticArgument] | None,
    ) -> Computation:
        """Lower an application of a named computation.

        The callee is resolved against the signature table, which is
        complete before any body is lowered, so a call to a computation
        declared later in the module resolves like any other.

        Parameters
        ----------
        authored : surface.QiecCallComputation
            The source call.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        path : tuple[str | int, ...]
            Structural path of this call site.
        static_bindings : Mapping[str, StaticArgument] or None
            Static bindings from an enclosing case refinement.

        Returns
        -------
        Call
            The lowered call, carrying the instantiated result type and
            effect row so a later check can compare them against the
            callee's signature.

        Raises
        ------
        QiecDiagnosticError
            If the callee is undeclared, or the static arguments do not
            instantiate its telescope.
        """
        if authored.callee == "traverse":
            return self._lower_traverse(authored, scope, context, path, static_bindings)
        signature = self.computation_signatures.get(authored.callee)
        if signature is None:
            self._fail(
                authored,
                f"unknown computation {authored.callee!r}",
                code="qiec-route",
            )
        if len(authored.static_arguments) != len(signature.telescope):
            self._fail(
                authored,
                f"call to {authored.callee!r} supplies "
                f"{len(authored.static_arguments)} static argument(s); the "
                f"declaration binds {len(signature.telescope)}",
                code="qiec-kind",
            )
        static_arguments = tuple(
            self._lower_static_argument(argument, binder, scope, static_bindings)
            for argument, binder in zip(
                authored.static_arguments, signature.telescope, strict=True
            )
        )
        try:
            substitution = instantiate_telescope(signature.telescope, static_arguments)
        except (TypeError, ValueError) as error:
            self._fail_kernel(authored, error, fallback="qiec-kind")
        return Call(
            signature.id,
            authored.callee,
            static_arguments,
            tuple(
                self._lower_value(
                    value, scope, context, static_bindings, (*path, position)
                )
                for position, value in enumerate(authored.arguments)
            ),
            substitute_type(signature.result, substitution),
            substitute_row(signature.effects, substitution),
            self._origin(authored, path, "call"),
        )

    def _lower_traverse(
        self,
        authored: surface.QiecCallComputation,
        scope: Telescope,
        context: CheckContext,
        path: tuple[str | int, ...],
        static_bindings: Mapping[str, StaticArgument] | None,
    ) -> Computation:
        """Sequence one computation call per item of a fixed tensor.

        ``traverse(xs, x -> f(x, ...))`` is computation syntax, unlike
        pure ``map``.  Its finite leading extent is expanded into ordinary
        QIEC calls and binds, so effect rows, site provenance, interpreters,
        and every transpiler continue to consume the existing checked core.
        A source other than a variable or literal is bound once ahead of
        the calls, and each call reads its item from that binding.
        """
        if authored.static_arguments:
            self._fail(
                authored,
                "traverse takes no static arguments",
                code="qiec-kind",
            )
        if len(authored.arguments) != 2:
            self._fail(
                authored,
                f"traverse takes 2 arguments, got {len(authored.arguments)}",
                code="qiec-primitive",
            )
        source_expr = authored.arguments[0]
        source = self._lower_value(
            source_expr, scope, context, static_bindings, (*path, "source")
        )
        source_type = self._value_type(source, context, source_expr)
        shape = tensor_shape(source_type)
        if shape is None or not shape[1]:
            self._fail(
                source_expr,
                f"traverse expects a non-scalar tensor, got "
                f"{self._render(source_type)}",
                code="qiec-primitive",
            )
        extent = shape[1][0]
        if not isinstance(extent, IndexLiteral):
            self._fail(
                authored,
                "traverse requires a statically known leading extent",
                code="qiec-primitive",
            )
        count = int(extent.value)
        if count == 0:
            self._fail(
                authored,
                "traverse cannot infer the result type of an empty tensor",
                code="qiec-primitive",
            )
        function_arg = authored.arguments[1]
        if isinstance(function_arg, surface.LetExprLambda):
            function = function_arg
        elif isinstance(function_arg, surface.LetExprVar):
            function = self._lambda_macros.get(function_arg.name)
            if function is None:
                self._fail(
                    function_arg,
                    f"unknown lambda macro {function_arg.name!r}",
                    code="qiec-primitive",
                )
        else:
            self._fail(
                function_arg,
                "traverse's second argument must be a lambda",
                code="qiec-primitive",
            )
        if not isinstance(function.body, surface.LetExprCall):
            self._fail(
                function.body,
                "traverse's lambda body must call a named computation",
                code="qiec-primitive",
            )

        calls: list[tuple[Local, Computation]] = []
        result_type: TypeExpr | None = None
        # A source other than a variable is bound once by the computation,
        # so each call reads an item of the binding rather than a copy of
        # the source.
        shared: Local | None = None
        if not _atomic(source_expr):
            shared = self._shared_local("traverse_source", source_type, context)
            context = context.extend(shared)
            source_expr = surface.LetExprVar(
                name=shared.name, line=authored.line, col=authored.col
            )
        occupied = {local.name for local in context.locals}
        for position in range(count):
            item = self._indexed_collection_item(
                source_expr,
                surface.LetExprLiteral(
                    value=float(position),
                    integral=True,
                    line=authored.line,
                    col=authored.col,
                ),
            )
            applied = substitute_let_expr(function.body, {function.param: item})
            assert isinstance(applied, surface.LetExprCall)
            call_surface = surface.QiecCallComputation(
                callee=applied.func,
                arguments=applied.args,
                line=applied.line,
                col=applied.col,
            )
            call = self._lower_call(
                call_surface,
                scope,
                context,
                (*path, "items", position),
                static_bindings,
            )
            try:
                inferred = infer_computation(call, self.registry, context)
            except (KernelError, TypeError, ValueError) as error:
                self._fail_kernel(applied, error)
            if result_type is None:
                result_type = inferred.result
            elif inferred.result != result_type:
                self._fail(
                    applied,
                    "traverse calls return inconsistent types",
                    code="qiec-primitive",
                )
            base = f"__traverse_{position}"
            name = base
            suffix = 0
            while name in occupied:
                suffix += 1
                name = f"{base}_{suffix}"
            occupied.add(name)
            calls.append((Local(name, inferred.result), call))

        assert result_type is not None
        inner = tensor_shape(result_type)
        tensor_result = (
            tensor_type(result_type, (extent,))
            if inner is None
            else tensor_type(inner[0], (extent, *inner[1]))
        )
        body: Computation = Return(
            TensorValue(tuple(Var(local) for local, _call in calls), tensor_result)
        )
        for local, call in reversed(calls):
            body = Bind(local, call, body)
        if shared is not None:
            body = Bind(shared, Return(source), body)
        return body

    def _lower_case(
        self,
        authored: surface.QiecCaseComputation,
        scope: Telescope,
        context: CheckContext,
        path: tuple[str | int, ...],
        static_bindings: Mapping[str, StaticArgument] | None = None,
    ) -> Case:
        """Lower a case analysis, refining each branch.

        Each branch gets its own static scope, whose identity derives
        from the path, so two branches' skolems can never be confused and
        a type mentioning one cannot escape its branch.

        Parameters
        ----------
        authored : surface.QiecCaseComputation
            The authored case.
        scope : Telescope
            Static binders in scope.
        context : CheckContext
            Value bindings in scope.
        path : tuple[str | int, ...]
            Structural path, entering each branch's scope identity.
        static_bindings : Mapping[str, StaticArgument] or None
            Bindings from an enclosing case refinement.

        Returns
        -------
        Case
            The lowered case.

        Raises
        ------
        QiecDiagnosticError
            If the scrutinee is not an indexed family, a branch names an unknown constructor, coverage is incomplete, or a branch body fails to check.
        """
        scrutinee = self._lower_value(
            authored.scrutinee, scope, context, static_bindings, (*path, "scrutinee")
        )
        try:
            scrutinee_type = infer_value(scrutinee, self.registry, context)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(authored.scrutinee, error, fallback="qiec-index")
        if not isinstance(scrutinee_type, TypeApplication):
            self._fail(
                authored.scrutinee,
                "case scrutinee is not an indexed-family value",
                code="qiec-index",
            )
        try:
            family = self.registry.family_for_type(scrutinee_type)
        except KernelError as error:
            self._fail(authored.scrutinee, str(error), code="qiec-index")
        motive_indices = self._lower_telescope(authored.motive.indices, refinable=True)
        motive_bindings = dict(static_bindings or {})
        for binder in motive_indices:
            motive_bindings.pop(binder.name, None)
        motive = CaseMotive(
            motive_indices,
            self._lower_type(
                authored.motive.result_type,
                (*scope, *motive_indices),
                motive_bindings,
            ),
        )
        parameter_count = len(family.parameters)
        actual_parameters = scrutinee_type.arguments[:parameter_count]
        parameter_substitution = instantiate_telescope(
            family.parameters, actual_parameters
        )
        branches: list[CaseBranch] = []
        for position, authored_branch in enumerate(authored.branches):
            constructor = self.constructors.get(authored_branch.constructor)
            if constructor is None or constructor.family != family.id:
                self._fail(
                    authored_branch,
                    f"constructor {authored_branch.constructor!r} does not belong "
                    f"to {family.name!r}",
                    code="qiec-coverage",
                )
            if len(authored_branch.static_arguments) != len(constructor.telescope):
                self._fail(
                    authored_branch,
                    f"branch {constructor.name!r} expects "
                    f"{len(constructor.telescope)} static binders",
                    code="qiec-index",
                )
            branch_scope = StaticScopeId.derive(
                self.source.module_name, path, "branch", position, constructor.id
            )
            skolems = constructor_skolems(constructor, branch_scope)
            authored_binders: list[TelescopeBinder] = []
            branch_bindings = dict(static_bindings or {})
            branch_names: set[str] = set()
            for authored_binder, declared_binder, skolem in zip(
                authored_branch.static_arguments,
                constructor.telescope,
                skolems,
                strict=True,
            ):
                if not isinstance(authored_binder, surface.QiecTypeName):
                    self._fail(
                        authored_binder,
                        "case branch static binders must be plain identifiers",
                        code="qiec-index",
                    )
                if authored_binder.name in branch_names:
                    self._fail(
                        authored_binder,
                        f"duplicate case branch static binder {authored_binder.name!r}",
                        code="qiec-index",
                    )
                branch_names.add(authored_binder.name)
                authored_binders.append(
                    self._rename_binder(declared_binder, authored_binder.name)
                )
                branch_bindings[authored_binder.name] = skolem
            local_substitution = instantiate_telescope(constructor.telescope, skolems)
            substitution = StaticSubstitution(
                (*parameter_substitution.types, *local_substitution.types),
                (*parameter_substitution.indices, *local_substitution.indices),
                (*parameter_substitution.effects, *local_substitution.effects),
            )
            if len(authored_branch.fields) != len(constructor.fields):
                self._fail(
                    authored_branch,
                    f"branch {constructor.name!r} binds "
                    f"{len(authored_branch.fields)} fields; expected "
                    f"{len(constructor.fields)}",
                    code="qiec-coverage",
                )
            branch_context = context.with_static_scope(
                branch_scope,
                tuple(
                    argument.identity
                    for argument in skolems
                    if isinstance(
                        argument, TypeVariable | IndexVariable | EffectVariable
                    )
                    and argument.identity is not None
                ),
            )
            fields: list[Local] = []
            seen_fields: set[str] = set()
            for authored_field, definition in zip(
                authored_branch.fields, constructor.fields, strict=True
            ):
                if authored_field.name in seen_fields:
                    self._fail(
                        authored_field,
                        f"duplicate case branch field {authored_field.name!r}",
                        code="qiec-kind",
                    )
                seen_fields.add(authored_field.name)
                expected = substitute_type(definition.type, substitution)
                if authored_field.type_expr is not None:
                    annotated = self._lower_type(
                        authored_field.type_expr,
                        (*scope, *authored_binders),
                        branch_bindings,
                    )
                    if annotated != expected:
                        self._fail(
                            authored_field,
                            f"branch field has type {annotated!r}, expected {expected!r}",
                            code="qiec-index",
                        )
                local = Local(authored_field.name, expected)
                fields.append(local)
                branch_context = branch_context.extend(local)
            branches.append(
                CaseBranch(
                    constructor.id,
                    skolems,
                    tuple(fields),
                    self._lower_computation(
                        authored_branch.body,
                        (*scope, *authored_binders),
                        branch_context,
                        (*path, "branch", position, "body"),
                        branch_bindings,
                    ),
                    branch_scope,
                )
            )
        return Case(scrutinee, motive, tuple(branches))

    def _rename_binder(
        self,
        binder: TelescopeBinder,
        name: str,
    ) -> TelescopeBinder:
        """Give an authored branch name the constructor binder's exact kind.

        A case branch names the constructor's static binders itself, so
        the authored name is substituted into the declared binder rather
        than a new binder being invented. That keeps the kind or sort the
        constructor declared, which is what the branch body is checked
        against.

        Parameters
        ----------
        binder : TelescopeBinder
            The constructor's own binder, supplying the kind or sort.
        name : str
            The name the branch gives it.

        Returns
        -------
        TelescopeBinder
            A binder of the same class and classifier, under the new
            name.
        """

        if isinstance(binder, TypeBinder):
            return TypeBinder(name, binder.kind, binder.refinable)
        if isinstance(binder, IndexBinder):
            return IndexBinder(name, binder.sort, binder.refinable)
        return EffectBinder(name, binder.refinable)

    def _binder_variable(self, binder: TelescopeBinder) -> StaticArgument:
        """The variable a binder introduces.

        Parameters
        ----------
        binder : TelescopeBinder
            The binder to reflect.

        Returns
        -------
        StaticArgument
            A variable of the binder's own name and namespace, carrying no identity, so substitution can reach it.
        """
        if isinstance(binder, TypeBinder):
            return TypeVariable(binder.name, binder.kind)
        if isinstance(binder, IndexBinder):
            return IndexVariable(binder.name, binder.sort)
        return EffectVariable(binder.name)

    def _fresh_sequence_name(
        self, context: CheckContext, path: tuple[str | int, ...]
    ) -> str:
        """A local name for a sequenced computation's discarded result.

        A sequence binds nothing in the source, but the core has one
        sequencing form, so a name is needed. It is chosen not to collide
        with anything the user wrote.

        Parameters
        ----------
        context : CheckContext
            Bindings already in scope, which the new name must avoid.
        path : tuple[str | int, ...]
            Structural path, making the stem readable in a diagnostic.

        Returns
        -------
        str
            The fresh name.
        """
        stem = "__qiec_sequence_" + "_".join(str(item) for item in path[-3:])
        name = stem
        suffix = 0
        existing = {local.name for local in context.locals}
        while name in existing:
            suffix += 1
            name = f"{stem}_{suffix}"
        return name

    def _checked_computation(
        self,
        signature: ComputationSignature,
        parameters: tuple[Local, ...],
        body: Computation,
        node: object,
        path: tuple[str | int, ...],
    ) -> NamedComputation:
        """Check a synthesized body against its registered signature.

        Parameters
        ----------
        signature : ComputationSignature
            The signature, already registered.
        parameters : tuple[Local, ...]
            The parameter locals, in the signature's order.
        body : Computation
            The body.
        node : object
            The source node diagnostics are reported at.
        path : tuple[str | int, ...]
            The computation's structural path.

        Returns
        -------
        NamedComputation
            The computation.

        Raises
        ------
        QiecDiagnosticError
            If the body fails to check, or its result or row differs
            from the signature's.
        """
        context = CheckContext(parameters)
        try:
            actual = infer_computation(body, self.registry, context)
        except (KernelError, TypeError, ValueError) as error:
            self._fail_kernel(node, error, fallback="qiec-program")
        if actual.result != signature.result:
            self._fail(
                node,
                f"computation {signature.name!r} produces "
                f"{render_static(actual.result)}, not "
                f"{render_static(signature.result)}",
                code="qiec-program",
            )
        declared = {entry.instance for entry in signature.effects.entries}
        performed = {entry.instance for entry in actual.effects.entries}
        if not performed <= declared:
            self._fail(
                node,
                f"computation {signature.name!r} performs on instances its "
                "signature does not declare",
                code="qiec-program",
            )
        return NamedComputation(
            signature.id,
            signature.name,
            signature.telescope,
            parameters,
            body,
            ComputationType(signature.effects, signature.result),
            self._origin(node, path, "computation"),
        )

    def _origin(
        self,
        node: object,
        path: tuple[str | int, ...],
        role: str,
    ) -> SourceOrigin:
        """The source origin for one lowered construct.

        Parameters
        ----------
        node : object
            The source node, read for its line and column.
        path : tuple[str | int, ...]
            Structural path, which is what the identity derives from.
        role : str
            What this position is.

        Returns
        -------
        SourceOrigin
            The origin. Line and column are diagnostic only: the identity
            comes from the path, so reformatting a file does not change
            what the sites in it are.
        """
        return SourceOrigin(
            self.source.module_name,
            path,
            role,
            QVR_SOURCE_VERSION,
            self.source.file_path,
            getattr(node, "line", None),
            getattr(node, "col", None),
        )

    def _fail_kernel(
        self,
        node: object,
        error: Exception,
        *,
        fallback: str = "qiec-kind",
    ) -> NoReturn:
        """Re-raise a kernel rejection at a source position.

        The kernel classifies its own rejections, so its code is kept
        rather than replaced. Overriding it would collapse a precise
        condition, such as a call arity mismatch, into whichever code the
        surrounding lowering pass happens to use.

        Parameters
        ----------
        node : object
            The source node to blame.
        error : Exception
            The rejection to report.
        fallback : str
            The code to use when the error carries none, as a plain
            `TypeError` or `ValueError` does.

        Raises
        ------
        QiecDiagnosticError
            Always.
        """
        code = getattr(error, "code", None)
        self._fail(node, str(error), code=code if isinstance(code, str) else fallback)

    def _fail(
        self,
        node: object,
        message: str,
        *,
        code: str = "qiec-kind",
    ) -> NoReturn:
        """Raise a source-located diagnostic and stop lowering.

        Parameters
        ----------
        node : object
            The source node to blame, read for its line and column.
        message : str
            What went wrong, in prose.
        code : str
            The stable diagnostic code.

        Raises
        ------
        QiecDiagnosticError
            Always. The return type is `NoReturn` so callers need no
            unreachable branch after calling it.
        """
        raise QiecDiagnosticError(
            message,
            code=code,
            file=self.source.file_path,
            line=getattr(node, "line", 0),
            column=getattr(node, "col", 0),
        )


__all__ = [
    "CheckedQvrQiec",
    "QIEC_STATEMENT_TYPES",
    "QVR_SOURCE_VERSION",
    "QvrQiecLowerer",
    "QvrQiecSource",
    "has_qiec_surface",
    "lower_qvr_to_qiec",
    "non_qiec_projection",
    "qiec_projection",
]
