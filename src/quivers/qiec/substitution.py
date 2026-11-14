"""Capture-free substitution and telescope instantiation for QIEC."""

from __future__ import annotations

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
import didactic.api as dx


from quivers.qiec.effects import (
    EffectRequest,
    EffectRow,
    OperationDef,
    RowEntry,
)
from quivers.qiec.evidence import BranchGiven, EqualityEvidence, Reflexivity
from quivers.qiec.terms import (
    AttachmentRef,
    Bind,
    Call,
    Case,
    CaseBranch,
    CaseMotive,
    Computation,
    ConstructorValue,
    DistributionValue,
    EvidenceValue,
    Handle,
    If,
    LiteralValue,
    Local,
    LogDensity,
    NewInstance,
    Perform,
    PrimitiveApplication,
    Projection,
    Resume,
    Return,
    SiteValue,
    TransportValue,
    TensorValue,
    TupleValue,
    PlateAxis,
    PlateShape,
    Gather,
    WeightSum,
    SegmentSum,
    KernelMatrix,
    AffineMap,
    TableMap,
    Reduction,
    Rowwise,
    Comprehension,
    LetBinding,
    LetValue,
    Value,
    Var,
)
from quivers.qiec.kinds import (
    Telescope,
)
from quivers.qiec.types import (
    EffectRef,
    EffectVariable,
    EqualityType,
    FunctionType,
    IndexConstructor,
    IndexTerm,
    IndexVariable,
    ShapeIndex,
    StaticArgument,
    TypeApplication,
    TypeExpr,
    TypeVariable,
    check_static_arguments,
)


class StaticSubstitution(QiecModel):
    """A deterministic substitution split by static namespace.

    Parameters
    ----------
    types
        Bindings from type-variable names to types.
    indices
        Bindings from index-variable names to index terms.
    effects
        Bindings from effect-variable names to interface applications or
        other effect variables.
    """

    types: tuple[tuple[str, TypeExpr], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    indices: tuple[tuple[str, IndexTerm], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    effects: tuple[tuple[str, EffectRef | EffectVariable], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Reject a substitution that binds one name twice.

        Raises
        ------
        ValueError
            If a name repeats within a namespace. Two bindings for one
            name would make the result depend on lookup order.
        """
        for namespace in (self.types, self.indices, self.effects):
            names = [name for name, _ in namespace]
            if len(set(names)) != len(names):
                raise ValueError("duplicate substitution binding")

    def type(self, name: str) -> TypeExpr | None:
        """What a type binder of this name is bound to.

        Parameters
        ----------
        name : str
            The binder name.

        Returns
        -------
        TypeExpr or None
            The bound type, or None when this substitution says nothing
            about the name, in which case the variable stands.
        """
        return next((value for key, value in self.types if key == name), None)

    def index(self, name: str) -> IndexTerm | None:
        """What an index binder of this name is bound to.

        Parameters
        ----------
        name : str
            The binder name.

        Returns
        -------
        IndexTerm or None
            The bound index, or None when the name is unbound here.
        """
        return next((value for key, value in self.indices if key == name), None)

    def effect(self, name: str) -> EffectRef | EffectVariable | None:
        """What an effect binder of this name is bound to.

        Parameters
        ----------
        name : str
            The binder name.

        Returns
        -------
        EffectRef or EffectVariable or None
            The bound interface application or effect variable, or None
            when the name is unbound here.
        """
        return next((value for key, value in self.effects if key == name), None)


def instantiate_telescope(
    telescope: Telescope,
    arguments: tuple[StaticArgument, ...],
) -> StaticSubstitution:
    """Kind-check static arguments and construct their substitution.

    Parameters
    ----------
    telescope : Telescope
        The binders to instantiate, in order. Later binders may depend on
        earlier ones, so the arguments are checked left to right with the
        substitution built so far already applied.
    arguments : tuple[StaticArgument, ...]
        One argument per binder.

    Returns
    -------
    StaticSubstitution
        The substitution carrying each binder to its argument.

    Raises
    ------
    TypeError
        If an argument is of the wrong static class for its binder.
    ValueError
        If the counts differ, or an argument is of the wrong kind or
        index sort.
    """
    types, indices, effects = check_static_arguments(telescope, arguments)
    return StaticSubstitution(types=types, indices=indices, effects=effects)


def substitute_index(term: IndexTerm, substitution: StaticSubstitution) -> IndexTerm:
    """Substitute into an index term.

    Parameters
    ----------
    term : IndexTerm
        The index to rewrite.
    substitution : StaticSubstitution
        The bindings to apply.

    Returns
    -------
    IndexTerm
        The rewritten index. A variable carrying an identity is rigid and
        is returned unchanged, which is what keeps a branch's skolem from
        being captured by a like-named declaration binder.
    """
    if isinstance(term, IndexVariable) and term.identity is None:
        return substitution.index(term.name) or term
    if isinstance(term, IndexConstructor):
        return IndexConstructor(
            name=term.name,
            arguments=tuple(
                substitute_index(arg, substitution) for arg in term.arguments
            ),
            sort=term.sort,
        )
    if isinstance(term, ShapeIndex):
        return ShapeIndex(
            dimensions=tuple(
                substitute_index(dimension, substitution)
                for dimension in term.dimensions
            )
        )
    return term


def substitute_effect(effect: EffectRef, substitution: StaticSubstitution) -> EffectRef:
    """Substitute into an interface application's static arguments.

    Parameters
    ----------
    effect : EffectRef
        The application to rewrite.
    substitution : StaticSubstitution
        The bindings to apply.

    Returns
    -------
    EffectRef
        The application with its arguments rewritten. The declaration's
        identity and name are unchanged, since substitution instantiates
        an interface rather than selecting a different one.
    """
    return EffectRef(
        id=effect.id,
        name=effect.name,
        arguments=tuple(
            substitute_static(arg, substitution) for arg in effect.arguments
        ),
    )


def substitute_plate(plate: PlateShape, substitution: StaticSubstitution) -> PlateShape:
    """Substitute static variables in a plate's axis sizes.

    Parameters
    ----------
    plate : PlateShape
        The plate.
    substitution : StaticSubstitution
        The static substitution.

    Returns
    -------
    PlateShape
        The plate with every axis size substituted.
    """
    return PlateShape(
        batch=tuple(
            PlateAxis(name=axis.name, size=substitute_index(axis.size, substitution))
            for axis in plate.batch
        ),
        event=tuple(
            PlateAxis(name=axis.name, size=substitute_index(axis.size, substitution))
            for axis in plate.event
        ),
    )


def substitute_type(type_: TypeExpr, substitution: StaticSubstitution) -> TypeExpr:
    """Substitute into a type expression.

    Parameters
    ----------
    type_ : TypeExpr
        The type to rewrite.
    substitution : StaticSubstitution
        The bindings to apply.

    Returns
    -------
    TypeExpr
        The rewritten type. As with indices, a variable carrying an
        identity is rigid and survives unchanged.

    Raises
    ------
    TypeError
        If the expression is of an unknown class.
    """
    if isinstance(type_, TypeVariable) and type_.identity is None:
        return substitution.type(type_.name) or type_
    if isinstance(type_, TypeVariable):
        return type_
    if isinstance(type_, TypeApplication):
        return TypeApplication(
            constructor=type_.constructor,
            arguments=tuple(
                substitute_static(arg, substitution) for arg in type_.arguments
            ),
        )
    if isinstance(type_, FunctionType):
        return FunctionType(
            parameter=substitute_type(type_.parameter, substitution),
            result=substitute_type(type_.result, substitution),
        )
    if isinstance(type_, EqualityType):
        return EqualityType(
            kind=type_.kind,
            left=substitute_static(type_.left, substitution),
            right=substitute_static(type_.right, substitution),
        )
    raise TypeError(f"unknown type term {type_!r}")


def substitute_static(
    term: StaticArgument,
    substitution: StaticSubstitution,
) -> StaticArgument:
    """Substitute into a static argument of any namespace.

    Parameters
    ----------
    term : StaticArgument
        A type, index, or effect argument.
    substitution : StaticSubstitution
        The bindings to apply.

    Returns
    -------
    StaticArgument
        The rewritten argument, dispatched to the namespace it belongs
        to.

    Raises
    ------
    TypeError
        If the term is of an unknown class.
    """
    if isinstance(term, EffectVariable) and term.identity is None:
        return substitution.effect(term.name) or term
    if isinstance(term, EffectVariable):
        return term
    if isinstance(term, EffectRef):
        return substitute_effect(term, substitution)
    if isinstance(term, (IndexVariable, IndexConstructor, ShapeIndex)) or hasattr(
        term, "sort"
    ):
        return substitute_index(term, substitution)  # type: ignore[arg-type]
    return substitute_type(term, substitution)  # type: ignore[arg-type]


def substitute_row(row: EffectRow, substitution: StaticSubstitution) -> EffectRow:
    """Substitute into every interface application in an effect row.

    Parameters
    ----------
    row : EffectRow
        The row to rewrite.
    substitution : StaticSubstitution
        The bindings to apply.

    Returns
    -------
    EffectRow
        The row with each entry's interface rewritten. Instance
        identities and the open tail are untouched: this substitutes
        static arguments, and a row variable is solved by unification
        rather than by this substitution.
    """
    return EffectRow(
        entries=tuple(
            RowEntry(
                instance=entry.instance,
                effect=substitute_effect(entry.effect, substitution),
            )
            for entry in row.entries
        ),
        tail=row.tail,
    )


def instantiate_operation(
    operation: OperationDef,
    arguments: tuple[StaticArgument, ...],
    outer: StaticSubstitution = StaticSubstitution(),
) -> tuple[tuple[TypeExpr, ...], TypeExpr]:
    """Instantiate an operation's signature at a request site.

    Parameters
    ----------
    operation : OperationDef
        The declared operation.
    arguments : tuple[StaticArgument, ...]
        Static arguments for the operation's own telescope.
    outer : StaticSubstitution
        Bindings already in force from the interface's telescope. These
        come first, so an operation binder of the same name shadows the
        interface's, which is the scoping a reader expects.

    Returns
    -------
    tuple[tuple[TypeExpr, ...], TypeExpr]
        The instantiated parameter types and result type.

    Raises
    ------
    TypeError
        If an argument is of the wrong static class for its binder.
    ValueError
        If the argument count is wrong, or an argument is ill-kinded.
    """
    local = instantiate_telescope(operation.telescope, arguments)
    substitution = StaticSubstitution(
        types=(*outer.types, *local.types),
        indices=(*outer.indices, *local.indices),
        effects=(*outer.effects, *local.effects),
    )
    parameter_types = tuple(
        substitute_type(argument.type, substitution) for argument in operation.arguments
    )
    return parameter_types, substitute_type(operation.result_type, substitution)


def substitute_evidence(
    evidence: EqualityEvidence,
    substitution: StaticSubstitution,
) -> EqualityEvidence:
    """Substitute into the equality an evidence term states.

    Parameters
    ----------
    evidence : EqualityEvidence
        The evidence to rewrite.
    substitution : StaticSubstitution
        The static bindings to apply.

    Returns
    -------
    EqualityEvidence
        Evidence of the same class stating the substituted equality. A
        branch given keeps its identity, since substituting into what it
        proves does not make it a different piece of evidence.

    Raises
    ------
    TypeError
        If the evidence is of an unknown class.
    """
    equality = EqualityType(
        kind=evidence.equality.kind,
        left=substitute_static(evidence.equality.left, substitution),
        right=substitute_static(evidence.equality.right, substitution),
    )
    if isinstance(evidence, Reflexivity):
        return Reflexivity(equality=equality)
    if isinstance(evidence, BranchGiven):
        return BranchGiven(id=evidence.id, equality=equality)
    raise TypeError(f"unknown equality evidence {evidence!r}")


def substitute_value(value: Value, substitution: StaticSubstitution) -> Value:
    """Substitute static arguments throughout a value term.

    Parameters
    ----------
    value : Value
        The value to rewrite.
    substitution : StaticSubstitution
        The static bindings to apply.

    Returns
    -------
    Value
        The value with every type, index, and effect position
        substituted. Value-level structure is untouched: this rewrites
        the static layer, not the data.

    Raises
    ------
    TypeError
        If the value is of an unknown class.
    """
    if isinstance(value, Var):
        return Var(
            local=Local(
                name=value.local.name,
                type=substitute_type(value.local.type, substitution),
            )
        )
    if isinstance(value, LiteralValue):
        return LiteralValue(
            value=value.value, type=substitute_type(value.type, substitution)
        )
    if isinstance(value, ConstructorValue):
        return ConstructorValue(
            constructor=value.constructor,
            static_arguments=tuple(
                substitute_static(argument, substitution)
                for argument in value.static_arguments
            ),
            fields=tuple(
                substitute_value(field, substitution) for field in value.fields
            ),
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, EvidenceValue):
        return EvidenceValue(evidence=substitute_evidence(value.evidence, substitution))
    if isinstance(value, AttachmentRef):
        return AttachmentRef(
            attachment=value.attachment, type=substitute_type(value.type, substitution)
        )
    if isinstance(value, TransportValue):
        return TransportValue(
            evidence=substitute_evidence(value.evidence, substitution),
            value=substitute_value(value.value, substitution),
            target_type=substitute_type(value.target_type, substitution),
        )
    if isinstance(value, PrimitiveApplication):
        return PrimitiveApplication(
            primitive=value.primitive,
            name=value.name,
            arguments=tuple(
                substitute_value(item, substitution) for item in value.arguments
            ),
            result_type=substitute_type(value.result_type, substitution),
            origin=value.origin,
        )
    if isinstance(value, TupleValue):
        return TupleValue(
            items=tuple(substitute_value(item, substitution) for item in value.items),
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, TensorValue):
        return TensorValue(
            items=tuple(substitute_value(item, substitution) for item in value.items),
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, Projection):
        return Projection(
            value=substitute_value(value.value, substitution),
            position=value.position,
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, DistributionValue):
        return DistributionValue(
            family=value.family,
            name=value.name,
            arguments=tuple(
                (name, substitute_value(argument, substitution))
                for name, argument in value.arguments
            ),
            result_type=substitute_type(value.result_type, substitution),
            origin=value.origin,
            plate=substitute_plate(value.plate, substitution),
        )
    if isinstance(value, Gather):
        return Gather(
            value=substitute_value(value.value, substitution),
            index=substitute_value(value.index, substitution),
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, WeightSum):
        return WeightSum(value=substitute_value(value.value, substitution))
    if isinstance(value, SegmentSum):
        return SegmentSum(
            value=substitute_value(value.value, substitution),
            index=substitute_value(value.index, substitution),
            groups=substitute_index(value.groups, substitution),
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, KernelMatrix):
        return KernelMatrix(
            inputs=substitute_value(value.inputs, substitution),
            kernel=value.kernel,
            length_scale=value.length_scale,
            jitter=value.jitter,
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, AffineMap):
        return AffineMap(
            weight=substitute_value(value.weight, substitution),
            bias=substitute_value(value.bias, substitution),
            sources=tuple(
                substitute_value(item, substitution) for item in value.sources
            ),
            row_offset=value.row_offset,
            rows=value.rows,
            transform=value.transform,
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, TableMap):
        return TableMap(
            table=substitute_value(value.table, substitution),
            index=substitute_value(value.index, substitution),
            row_offset=value.row_offset,
            rows=value.rows,
            transform=value.transform,
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, Reduction):
        return Reduction(
            operator=value.operator,
            value=substitute_value(value.value, substitution),
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, Rowwise):
        return Rowwise(
            operator=value.operator,
            value=substitute_value(value.value, substitution),
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, Comprehension):
        return Comprehension(
            binder=Local(
                name=value.binder.name,
                type=substitute_type(value.binder.type, substitution),
            ),
            extent=substitute_index(value.extent, substitution),
            body=substitute_value(value.body, substitution),
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, LetValue):
        return LetValue(
            bindings=tuple(
                LetBinding(
                    binder=Local(
                        name=binding.binder.name,
                        type=substitute_type(binding.binder.type, substitution),
                    ),
                    value=substitute_value(binding.value, substitution),
                )
                for binding in value.bindings
            ),
            body=substitute_value(value.body, substitution),
            result_type=substitute_type(value.result_type, substitution),
        )
    if isinstance(value, LogDensity):
        return LogDensity(
            sampleable=substitute_value(value.sampleable, substitution),
            value=substitute_value(value.value, substitution),
            origin=value.origin,
            batch=tuple(
                PlateAxis(
                    name=axis.name, size=substitute_index(axis.size, substitution)
                )
                for axis in value.batch
            ),
        )
    if isinstance(value, SiteValue):
        return SiteValue(
            label=value.label,
            result_type=substitute_type(value.result_type, substitution),
        )
    raise TypeError(f"unknown value term {value!r}")


def substitute_request(
    request: EffectRequest,
    substitution: StaticSubstitution,
) -> EffectRequest:
    """Substitute static arguments throughout an effect request.

    Parameters
    ----------
    request : EffectRequest
        The request to rewrite.
    substitution : StaticSubstitution
        The static bindings to apply.

    Returns
    -------
    EffectRequest
        The request against the substituted interface application, with
        its own static and value arguments and result type rewritten. The
        instance, the operation, and the site are identities and do not
        move.
    """
    return EffectRequest(
        instance=request.instance,
        effect=substitute_effect(request.effect, substitution),
        operation=request.operation,
        static_arguments=tuple(
            substitute_static(argument, substitution)
            for argument in request.static_arguments
        ),
        arguments=tuple(
            substitute_value(argument, substitution) for argument in request.arguments
        ),
        result_type=substitute_type(request.result_type, substitution),
        origin=request.origin,
    )


def substitute_computation(
    computation: Computation,
    substitution: StaticSubstitution,
) -> Computation:
    """Substitute static arguments throughout a computation.

    Capture cannot occur, and the reason is structural rather than a
    freshening pass. A substitution binds declaration binders, which
    carry no identity and are matched by name. Every binder a computation
    introduces is rigid: a case branch's skolems are built with derived
    `StaticVariableId`s, and `substitute_type` and `substitute_static`
    return a variable carrying an identity unchanged. So a branch's own
    binders are invisible to substitution while references to an
    enclosing declaration binder inside that branch are rewritten, which
    is exactly the capture-avoiding behavior.

    Parameters
    ----------
    computation : Computation
        The computation to rewrite.
    substitution : StaticSubstitution
        The static bindings to apply.

    Returns
    -------
    Computation
        The computation with every static position substituted.
        Identities are preserved throughout: a call still names the same
        callee, a handle the same handler, and an allocation the same
        instance.

    Raises
    ------
    TypeError
        If the computation, or a term nested in it, is of an unknown
        class.
    """
    if isinstance(computation, Return):
        return Return(value=substitute_value(computation.value, substitution))
    if isinstance(computation, Bind):
        return Bind(
            binder=Local(
                name=computation.binder.name,
                type=substitute_type(computation.binder.type, substitution),
            ),
            first=substitute_computation(computation.first, substitution),
            then=substitute_computation(computation.then, substitution),
        )
    if isinstance(computation, Perform):
        return Perform(request=substitute_request(computation.request, substitution))
    if isinstance(computation, Handle):
        return Handle(
            instance=computation.instance,
            handler=computation.handler,
            computation=substitute_computation(computation.computation, substitution),
            static_arguments=tuple(
                substitute_static(argument, substitution)
                for argument in computation.static_arguments
            ),
        )
    if isinstance(computation, Case):
        return Case(
            scrutinee=substitute_value(computation.scrutinee, substitution),
            motive=CaseMotive(
                indices=computation.motive.indices,
                result_type=substitute_type(
                    computation.motive.result_type, substitution
                ),
            ),
            branches=tuple(
                CaseBranch(
                    constructor=branch.constructor,
                    static_arguments=tuple(
                        substitute_static(argument, substitution)
                        for argument in branch.static_arguments
                    ),
                    fields=tuple(
                        Local(
                            name=field.name,
                            type=substitute_type(field.type, substitution),
                        )
                        for field in branch.fields
                    ),
                    body=substitute_computation(branch.body, substitution),
                    scope=branch.scope,
                )
                for branch in computation.branches
            ),
        )
    if isinstance(computation, If):
        return If(
            condition=substitute_value(computation.condition, substitution),
            then=substitute_computation(computation.then, substitution),
            otherwise=substitute_computation(computation.otherwise, substitution),
        )
    if isinstance(computation, Call):
        return Call(
            callee=computation.callee,
            name=computation.name,
            static_arguments=tuple(
                substitute_static(argument, substitution)
                for argument in computation.static_arguments
            ),
            arguments=tuple(
                substitute_value(argument, substitution)
                for argument in computation.arguments
            ),
            result_type=substitute_type(computation.result_type, substitution),
            effects=substitute_row(computation.effects, substitution),
            origin=computation.origin,
        )
    if isinstance(computation, Resume):
        return Resume(
            value=substitute_value(computation.value, substitution),
            origin=computation.origin,
        )
    if isinstance(computation, NewInstance):
        return NewInstance(
            instance=computation.instance,
            effect=substitute_effect(computation.effect, substitution),
            body=substitute_computation(computation.body, substitution),
            origin=computation.origin,
        )
    raise TypeError(f"unknown computation term {computation!r}")


__all__ = [
    "StaticSubstitution",
    "instantiate_operation",
    "instantiate_telescope",
    "substitute_computation",
    "substitute_effect",
    "substitute_evidence",
    "substitute_index",
    "substitute_plate",
    "substitute_request",
    "substitute_row",
    "substitute_static",
    "substitute_type",
    "substitute_value",
]
