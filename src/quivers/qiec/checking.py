"""Validation and type/effect inference for the QIEC reference kernel."""

from __future__ import annotations

from collections.abc import Mapping


import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel

from quivers.qiec.declarations import ConstructorDecl, FamilyDecl
from quivers.qiec.effects import (
    ResumptionGrade,
    ComputationType,
    EMPTY_ROW,
    EffectDef,
    EffectRequest,
    EffectRow,
    HandlerDef,
    OperationDef,
    RowEntry,
    render_instance,
    render_row,
)
from quivers.qiec.coverage import (
    BranchPattern,
    CoverageStatus,
    Reachability,
    check_indexed_coverage,
    constructor_skolems,
    join_branch_rows,
    refine_branch,
)
from quivers.qiec.evidence import BranchGiven, EqualityEvidence, Reflexivity
from quivers.qiec.identifiers import (
    ComputationId,
    ConstructorId,
    EffectId,
    EffectInstanceId,
    EqualityId,
    FamilyId,
    HandlerId,
    OperationId,
    StaticScopeId,
    StaticVariableId,
    TypeId,
)
from quivers.qiec.substitution import (
    StaticSubstitution,
    instantiate_operation,
    instantiate_telescope,
    substitute_effect,
    substitute_row,
    substitute_static,
    substitute_type,
)
from quivers.qiec.kinds import (
    NAT,
    TYPE,
    ContextSort,
    EffectBinder,
    IndexBinder,
    IndexSort,
    Kind,
    NatSort,
    ShapeSort,
    Telescope,
    TypeBinder,
    UserIndexSort,
)
from quivers.qiec.terms import (
    Resume,
    NewInstance,
    Call,
    AttachmentRef,
    Bind,
    Case,
    Computation,
    ConstructorValue,
    DistributionValue,
    EvidenceValue,
    Handle,
    If,
    LiteralValue,
    Local,
    LogDensity,
    Perform,
    PrimitiveApplication,
    Projection,
    Return,
    SiteValue,
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
    LetValue,
    TransportValue,
    TensorValue,
    TupleValue,
    Value,
    Var,
)
from quivers.qiec.canonical import (
    LOG_WEIGHT,
    SITE_CONSTRUCTOR,
    sampleable_type,
    sampled_element,
    tensor_shape,
    tensor_type,
)
from quivers.qiec.families import FAMILIES, DistributionFamily, FamilyParameter
from quivers.qiec.primitives import PRIMITIVES
from quivers.qiec.types import (
    IndexTerm,
    BOOL,
    product_type,
    INT,
    REAL,
    STRING,
    UNIT,
    EffectRef,
    EffectVariable,
    EqualityType,
    FunctionType,
    IndexConstructor,
    IndexLiteral,
    IndexVariable,
    ShapeIndex,
    StaticArgument,
    TypeApplication,
    TypeConstructorRef,
    TypeExpr,
    TypeVariable,
    index_sort,
    render_static,
    static_kind,
)


class KernelError(TypeError):
    """A static QIEC kernel rejection.

    Carries a stable diagnostic code alongside its message. The code is
    attached where the rejection is raised, because only that site knows
    which condition fired; classifying by message text afterwards is a
    guess that silently changes whenever the prose is reworded.

    Parameters
    ----------
    message : str
        What was rejected, in prose, for a human reader.
    code : str
        A member of the published code set. Defaults to ``"qiec-kind"``,
        which is where a rejection with no more specific classification
        belongs.
    """

    def __init__(self, message: str, code: str = "qiec-kind") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class StaticScope(QiecModel):
    """The three disjoint namespaces visible in a declaration body.

    Parameters
    ----------
    types
        Type binders in scope, each with its kind.
    indices
        Index binders in scope, each with its sort.
    effects
        Effect binder names in scope.
    """

    types: tuple[tuple[str, Kind], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    indices: tuple[tuple[str, IndexSort], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    effects: tuple[str, ...] = ()

    @property
    def names(self) -> frozenset[str]:
        """Every static name bound here, across all three namespaces.

        Returns
        -------
        frozenset[str]
            The union of the type, index, and effect binder names. The
            namespaces are disjoint by construction, so this is what
            shadowing is tested against.
        """
        return frozenset(
            (
                *[name for name, _ in self.types],
                *[name for name, _ in self.indices],
                *self.effects,
            )
        )

    def extend(self, telescope: Telescope, *, subject: str) -> StaticScope:
        """Return this scope widened by a telescope's binders.

        Parameters
        ----------
        telescope : Telescope
            Binders to add. Each must be a type, index, or effect binder.
        subject : str
            What is being extended, named in any diagnostic so the error
            points at the declaration rather than at the scope.

        Returns
        -------
        StaticScope
            A scope carrying the original binders and the new ones.

        Raises
        ------
        KernelError
            If a binder is of an unknown class, or if its name is already
            bound. Shadowing is rejected rather than resolved so a static
            name always denotes one thing within a declaration.
        """
        types = list(self.types)
        indices = list(self.indices)
        effects = list(self.effects)
        seen = set(self.names)
        for binder in telescope:
            name = getattr(binder, "name", None)
            if not isinstance(name, str):  # pragma: no cover - closed binder union
                raise KernelError(f"unknown binder in {subject}: {binder!r}")
            if name in seen:
                raise KernelError(f"static binder {name!r} is shadowed in {subject}")
            seen.add(name)
            if isinstance(binder, TypeBinder):
                types.append((binder.name, binder.kind))
            elif isinstance(binder, IndexBinder):
                indices.append((binder.name, binder.sort))
            elif isinstance(binder, EffectBinder):
                effects.append(binder.name)
            else:  # pragma: no cover - closed binder union
                raise KernelError(f"unknown binder in {subject}: {binder!r}")
        return StaticScope(
            types=tuple(types), indices=tuple(indices), effects=tuple(effects)
        )

    def type_kind(self, name: str) -> Kind | None:
        """The kind of a type binder in scope.

        Parameters
        ----------
        name : str
            The binder name to resolve.

        Returns
        -------
        Kind or None
            The binder's kind, or None when `name` is not a type binder
            here. None does not mean unbound: the name may be an index or
            effect binder instead.
        """
        return next((kind for candidate, kind in self.types if candidate == name), None)

    def index_sort(self, name: str) -> IndexSort | None:
        """The sort of an index binder in scope.

        Parameters
        ----------
        name : str
            The binder name to resolve.

        Returns
        -------
        IndexSort or None
            The binder's sort, or None when `name` is not an index binder
            here.
        """
        return next(
            (sort for candidate, sort in self.indices if candidate == name), None
        )


class ComputationSignature(QiecModel):
    """What a call needs to know about a computation it invokes.

    A signature rather than a declaration, and deliberately so: a call
    resolves against this, not against the callee's body. That is what
    lets every signature be collected before any body is checked, which
    in turn makes a forward call and mutual recursion ordinary rather
    than requiring a forward declaration.

    Parameters
    ----------
    id : ComputationId
        The declaration's stable identity, which the call carries.
    name : str
        The authored name, used in diagnostics.
    telescope : Telescope
        Static binders the computation takes.
    parameters : tuple[TypeExpr, ...]
        Value parameter types, in order, before instantiation.
    result : TypeExpr
        The result type before instantiation.
    effects : EffectRow
        The row the computation performs, before instantiation.
    """

    id: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    telescope: Telescope = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    parameters: tuple[TypeExpr, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    effects: EffectRow = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class KernelRegistry(QiecModel):
    """Resolved declarations used by the reference checker.

    Every table is keyed by stable identity, so a lookup never depends on a
    display name.

    Parameters
    ----------
    families
        Indexed data families by identity.
    constructors
        Family constructors by identity.
    effects
        Effect interfaces by identity.
    operations
        Each operation by identity, paired with the interface declaring it.
    handlers
        Handler declarations by identity.
    type_constructors
        Type constructors by identity, recording the telescope each is
        known with.
    computations
        Named computation signatures by identity.
    instance_names
        The source name of each lexical effect instance, for rendering
        rows in diagnostics.
    """

    families: dict[FamilyId, FamilyDecl] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    constructors: dict[ConstructorId, ConstructorDecl] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    effects: dict[EffectId, EffectDef] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    operations: dict[OperationId, tuple[EffectDef, OperationDef]] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    handlers: dict[HandlerId, HandlerDef] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    type_constructors: dict[TypeId, TypeConstructorRef] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    computations: dict[ComputationId, ComputationSignature] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    instance_names: dict[EffectInstanceId, str] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    def _record_type_constructor(self, constructor: TypeConstructorRef) -> None:
        """Remember one type constructor's telescope, or confirm it.

        Parameters
        ----------
        constructor : TypeConstructorRef
            The constructor whose telescope metadata to record. A
            constructor already recorded is checked for agreement rather
            than overwritten.

        Raises
        ------
        KernelError
            If the constructor is already known with a different
            telescope, which would make one identity mean two arities.
        """
        existing = self.type_constructors.get(constructor.id)
        if existing is not None and existing.telescope != constructor.telescope:
            raise KernelError(
                f"conflicting telescope metadata for type constructor {constructor.id}"
            )
        if existing is None:
            self.type_constructors[constructor.id] = constructor

    def _record_type_constructors(self, term: StaticArgument) -> None:
        """Record every type constructor reachable from a static term.

        Parameters
        ----------
        term : StaticArgument
            The term to walk. Structures that carry no constructor are
            ignored rather than rejected, since this collects metadata
            and does not validate.

        Raises
        ------
        KernelError
            If a constructor found here disagrees with one already
            recorded under the same identity.
        """
        if isinstance(term, TypeApplication):
            self._record_type_constructor(term.constructor)
            for argument in term.arguments:
                self._record_type_constructors(argument)
            return
        if isinstance(term, FunctionType):
            self._record_type_constructors(term.parameter)
            self._record_type_constructors(term.result)
            return
        if isinstance(term, EqualityType):
            self._record_type_constructors(term.left)
            self._record_type_constructors(term.right)
            return
        if isinstance(term, EffectRef | IndexConstructor):
            for argument in term.arguments:
                self._record_type_constructors(argument)
            return
        if isinstance(term, ShapeIndex):
            for dimension in term.dimensions:
                self._record_type_constructors(dimension)

    def _validate_effect_references(
        self,
        term: StaticArgument,
        additional: tuple[EffectDef, ...] = (),
    ) -> None:
        """Check every interface application reachable from a static term.

        Parameters
        ----------
        term : StaticArgument
            The term to walk.
        additional : tuple[EffectDef, ...]
            Declarations not yet in the registry to consider alongside it.
            This is what lets a declaration's own signature mention the
            interface being declared. An application naming a declaration
            found in neither is left alone rather than rejected, since an
            unregistered interface is resolved elsewhere.

        Raises
        ------
        KernelError
            If an application names a known declaration but does not
            saturate its telescope, or supplies an ill-kinded argument.
        """
        if isinstance(term, EffectRef):
            for argument in term.arguments:
                self._validate_effect_references(argument, additional)
            definition = self.effects.get(term.id) or next(
                (candidate for candidate in additional if candidate.ref.id == term.id),
                None,
            )
            if definition is not None and not definition.matches(term):
                raise KernelError(
                    f"invalid application of effect interface {term.name!r}"
                )
            return
        if isinstance(term, TypeApplication | IndexConstructor):
            for argument in term.arguments:
                self._validate_effect_references(argument, additional)
            return
        if isinstance(term, FunctionType):
            self._validate_effect_references(term.parameter, additional)
            self._validate_effect_references(term.result, additional)
            return
        if isinstance(term, EqualityType):
            self._validate_effect_references(term.left, additional)
            self._validate_effect_references(term.right, additional)
            return
        if isinstance(term, ShapeIndex):
            for dimension in term.dimensions:
                self._validate_effect_references(dimension, additional)

    def validate_static(
        self,
        term: StaticArgument,
        scope: StaticScope | None = None,
    ) -> None:
        """Validate a static term against every declaration known here.

        Parameters
        ----------
        term : StaticArgument
            The static term to check.
        scope : StaticScope or None
            Binders in scope at the term's position. None means the empty
            scope, so any variable reference is out of scope.

        Raises
        ------
        KernelError
            If the term is ill-kinded, references an unbound static
            variable, or applies a known declaration wrongly.
        """
        check_static(term, scope, registry=self)
        self._record_type_constructors(term)

    def validate_type(
        self,
        type_: TypeExpr,
        scope: StaticScope | None = None,
    ) -> None:
        """Validate a value type against every declaration known here.

        Parameters
        ----------
        type_ : TypeExpr
            The value type to check.
        scope : StaticScope or None
            Binders in scope at the type's position. None means the empty
            scope.

        Raises
        ------
        KernelError
            If the type is ill-kinded or references an unbound variable.
        """
        check_type(type_, scope, registry=self)
        self._record_type_constructors(type_)

    def validate_effect_row(self, row: EffectRow) -> None:
        """Validate all concrete interface applications in an effect row.

        Parameters
        ----------
        row : EffectRow
            The row whose entries to check. An open tail carries no
            application and so is not checked here.

        Raises
        ------
        KernelError
            If an entry applies its interface wrongly.
        """
        for entry in row.entries:
            self.validate_static(entry.effect)

    def validate_computation_type(self, type_: ComputationType) -> None:
        """Validate both strata of a computation type.

        Parameters
        ----------
        type_ : ComputationType
            The computation type whose effect row and result type to
            check.

        Raises
        ------
        KernelError
            If either stratum is ill-formed.
        """
        self.validate_effect_row(type_.effects)
        self.validate_type(type_.result)

    def register_family(self, family: FamilyDecl) -> None:
        """Record an indexed family declaration.

        Parameters
        ----------
        family : FamilyDecl
            The family to register, with its uniform parameters and its
            refinable indices.

        Raises
        ------
        KernelError
            If a family with this identity is already registered, if two
            of its binders share a name, or if its type constructor
            disagrees with one already recorded.
        """
        if family.id in self.families:
            raise KernelError(f"family already registered: {family.name!r}")
        StaticScope().extend(
            (*family.parameters, *family.indices),
            subject=f"family {family.name!r}",
        )
        self._record_type_constructor(family.type_constructor)
        self.families[family.id] = family

    def register_constructor(self, constructor: ConstructorDecl) -> None:
        """Record a constructor against the family that declares it.

        Parameters
        ----------
        constructor : ConstructorDecl
            The constructor to register. Its family must already be
            registered and must list this constructor.

        Raises
        ------
        KernelError
            If the constructor is already registered, if its family is
            unknown or does not declare it, if it returns the wrong
            number of result indices, if its telescope shadows a family
            binder, or if a field type or result index is ill-formed.
        """
        if constructor.id in self.constructors:
            raise KernelError(f"constructor already registered: {constructor.name!r}")
        family = self.families.get(constructor.family)
        if family is None:
            raise KernelError(f"unknown family for constructor {constructor.name!r}")
        if constructor.id not in family.constructors:
            raise KernelError(
                f"constructor {constructor.name!r} is not declared by family {family.name!r}"
            )
        if len(constructor.result_indices) != len(family.indices):
            raise KernelError(
                f"constructor {constructor.name!r} returns "
                f"{len(constructor.result_indices)} indices; expected {len(family.indices)}"
            )
        family_names = {binder.name for binder in (*family.parameters, *family.indices)}
        constructor_names = {binder.name for binder in constructor.telescope}
        shadowed = family_names & constructor_names
        if shadowed:
            raise KernelError(
                f"constructor {constructor.name!r} shadows family binders: {shadowed!r}"
            )
        scope = (
            StaticScope()
            .extend(
                family.parameters,
                subject=f"family parameters of {family.name!r}",
            )
            .extend(
                constructor.telescope,
                subject=f"constructor {constructor.name!r}",
            )
        )
        for field_definition in constructor.fields:
            self.validate_type(field_definition.type, scope)
        for result, binder in zip(
            constructor.result_indices,
            family.indices,
            strict=True,
        ):
            self.validate_static(result, scope)
            _check_binder_argument(
                binder,
                result,
                subject=f"result index of constructor {constructor.name!r}",
            )
        self.constructors[constructor.id] = constructor

    def register_effect(self, effect: EffectDef) -> None:
        """Record an effect interface and index its operations.

        Each operation's signature is checked in a scope carrying the
        interface binders and then its own, and may mention the interface
        being declared.

        Parameters
        ----------
        effect : EffectDef
            The interface to register.

        Raises
        ------
        KernelError
            If the interface or one of its operations is already
            registered, if a binder is shadowed, or if an argument or
            result type is ill-formed.
        """
        if effect.ref.id in self.effects:
            raise KernelError(f"effect already registered: {effect.ref.name!r}")
        interface_scope = StaticScope().extend(
            effect.telescope,
            subject=f"effect {effect.ref.name!r}",
        )
        for operation in effect.operations:
            if operation.id in self.operations:
                raise KernelError(f"operation already registered: {operation.name!r}")
            operation_scope = interface_scope.extend(
                operation.telescope,
                subject=f"operation {operation.name!r}",
            )
            for argument in operation.arguments:
                self.validate_type(argument.type, operation_scope)
                self._validate_effect_references(argument.type, (effect,))
            self.validate_type(operation.result_type, operation_scope)
            self._validate_effect_references(operation.result_type, (effect,))
        self.effects[effect.ref.id] = effect
        for operation in effect.operations:
            self.operations[operation.id] = (effect, operation)

    def register_handler(self, handler: HandlerDef) -> None:
        """Record a handler against the interface it handles.

        Coverage is checked here rather than at construction, because it
        is a claim about the interface's operations and only the registry
        knows those.

        Parameters
        ----------
        handler : HandlerDef
            The handler signature to register.

        Raises
        ------
        KernelError
            If the handler is already registered, if its telescope
            shadows a binder, if its input, output, or introduced row is
            ill-formed, if the interface it names is unknown or wrongly
            applied, if a clause covers an operation the interface does
            not declare, or if the handler claims totality while leaving
            an operation uncovered, or if an authored clause body fails
            to check, answers the wrong type, or resumes more often than
            its grade allows.
        """
        if handler.id in self.handlers:
            raise KernelError(f"handler already registered: {handler.name!r}")
        scope = StaticScope().extend(
            handler.telescope,
            subject=f"handler {handler.name!r}",
        )
        self.validate_static(handler.effect, scope)
        self.validate_type(handler.input_type, scope)
        self.validate_type(handler.output_type, scope)
        for entry in handler.introduced.entries:
            self.validate_static(entry.effect, scope)
            introduced = self.effects.get(entry.effect.id)
            if introduced is None or not introduced.matches(entry.effect):
                raise KernelError(
                    f"handler {handler.name!r} introduces an unknown effect interface"
                )
        effect = self.effects.get(handler.effect.id)
        if effect is None or not effect.matches(handler.effect):
            raise KernelError(f"unknown effect interface for handler {handler.name!r}")
        declared = {operation.id for operation in effect.operations}
        clauses = {clause.operation for clause in handler.clauses}
        unknown = clauses - declared
        if unknown:
            raise KernelError(
                f"handler has clauses for unknown operations: {unknown!r}"
            )
        if handler.total and clauses != declared:
            missing = declared - clauses
            raise KernelError(f"total handler is missing operations: {missing!r}")
        self._check_handler_bodies(
            handler,
            effect,
            substitution=instantiate_telescope(
                effect.telescope, handler.effect.arguments
            ),
        )
        self.handlers[handler.id] = handler

    def _check_handler_bodies(
        self,
        handler: HandlerDef,
        effect: EffectDef,
        *,
        substitution: StaticSubstitution,
    ) -> None:
        """Type every authored clause body and hold it to its grade.

        A clause body answers the handler's output type, and inside it
        `resume` carries what the handled operation supplies. Those two
        facts are what make the body checkable at all, and they come from
        the interface rather than from the clause, so they are
        established here rather than at the clause.

        Parameters
        ----------
        handler : HandlerDef
            The handler being registered.
        effect : EffectDef
            The interface it handles, already matched against its
            application.
        substitution : StaticSubstitution
            The interface telescope instantiated at that application.

        Raises
        ------
        KernelError
            If a clause body fails to check, answers a type other than
            the handler's output, or resumes more often than its grade
            allows.
        """
        output = substitute_type(handler.output_type, substitution)
        if handler.return_clause is not None:
            clause = handler.return_clause
            context = CheckContext().extend(clause.binder)
            actual = infer_computation(clause.body, self, context)
            if actual.result != output:
                raise KernelError(
                    f"return clause of handler {handler.name!r} answers "
                    f"{actual.result!r}, not the declared {output!r}",
                    "qiec-handler-body",
                )
        for clause in handler.clauses:
            if clause.body is None:
                continue
            _, operation = self.operation(clause.operation)
            parameter_types, resumed = instantiate_operation(
                operation,
                tuple(_binder_variable(binder) for binder in operation.telescope),
                substitution,
            )
            if len(clause.parameters) != len(parameter_types):
                raise KernelError(
                    f"clause {operation.name!r} of handler {handler.name!r} "
                    f"binds {len(clause.parameters)} argument(s); the "
                    f"operation takes {len(parameter_types)}",
                    "qiec-handler-body",
                )
            context = CheckContext().with_resumption(
                ResumptionType(
                    result=resumed, answer=output, effects=handler.introduced
                )
            )
            for parameter, declared in zip(
                clause.parameters, parameter_types, strict=True
            ):
                if parameter.type != declared:
                    raise KernelError(
                        f"clause {operation.name!r} of handler "
                        f"{handler.name!r} binds {parameter.name!r} at "
                        f"{render_static(parameter.type)}, not the declared {render_static(declared)}",
                        "qiec-handler-body",
                    )
                context = context.extend(parameter)
            actual = infer_computation(clause.body, self, context)
            if actual.result != output:
                raise KernelError(
                    f"clause {operation.name!r} of handler {handler.name!r} "
                    f"answers {actual.result!r}, not the declared {output!r}",
                    "qiec-handler-body",
                )
            check_resumption_grade(
                clause.grade,
                resumption_use(clause.body),
                subject=f"clause {operation.name!r} of handler {handler.name!r}",
            )

    def register_computation(self, signature: ComputationSignature) -> None:
        """Record a computation's signature so calls can resolve to it.

        Signatures are registered before any body is checked. A call
        therefore resolves whether the callee is declared above or below
        it, and mutual recursion needs no forward declaration.

        Parameters
        ----------
        signature : ComputationSignature
            The signature to record.

        Raises
        ------
        KernelError
            If a computation with this identity is already registered, if
            its telescope shadows a binder, or if a parameter type, the
            result type, or the declared row is ill-formed. A duplicate
            identity is rejected rather than overwritten, since a call
            carrying that identity would otherwise resolve to whichever
            declaration happened to register last.
        """
        if signature.id in self.computations:
            raise KernelError(f"computation already registered: {signature.name!r}")
        scope = StaticScope().extend(
            signature.telescope,
            subject=f"computation {signature.name!r}",
        )
        for parameter in signature.parameters:
            self.validate_type(parameter, scope)
        self.validate_type(signature.result, scope)
        self.validate_effect_row(signature.effects)
        self.computations[signature.id] = signature

    def computation(self, computation: ComputationId) -> ComputationSignature:
        """The registered signature with a given identity.

        Parameters
        ----------
        computation : ComputationId
            The identity to resolve.

        Returns
        -------
        ComputationSignature
            The registered signature.

        Raises
        ------
        KernelError
            If no computation is registered under that identity, which is
            how a call to an undeclared name is caught.
        """
        try:
            return self.computations[computation]
        except KeyError as exc:
            raise KernelError(
                f"unknown computation {computation}", "qiec-call"
            ) from exc

    def constructor(self, constructor: ConstructorId) -> ConstructorDecl:
        """The registered constructor with a given identity.

        Parameters
        ----------
        constructor : ConstructorId
            The identity to resolve.

        Returns
        -------
        ConstructorDecl
            The registered declaration.

        Raises
        ------
        KernelError
            If no constructor is registered under that identity. The
            underlying `KeyError` is wrapped so callers handle one
            kernel error class rather than two.
        """
        try:
            return self.constructors[constructor]
        except KeyError as exc:
            raise KernelError(f"unknown constructor {constructor}") from exc

    def handler(self, handler: HandlerId) -> HandlerDef:
        """The registered handler with a given identity.

        Parameters
        ----------
        handler : HandlerId
            The identity to resolve.

        Returns
        -------
        HandlerDef
            The registered signature.

        Raises
        ------
        KernelError
            If no handler is registered under that identity.
        """
        try:
            return self.handlers[handler]
        except KeyError as exc:
            raise KernelError(f"unknown handler {handler}") from exc

    def operation(self, operation: OperationId) -> tuple[EffectDef, OperationDef]:
        """The operation with a given identity, and its owning interface.

        Parameters
        ----------
        operation : OperationId
            The identity to resolve.

        Returns
        -------
        tuple[EffectDef, OperationDef]
            The interface that declares the operation, and the operation
            itself. The interface comes back too because checking a
            request needs both.

        Raises
        ------
        KernelError
            If no operation is registered under that identity.
        """
        try:
            return self.operations[operation]
        except KeyError as exc:
            raise KernelError(f"unknown operation {operation}") from exc

    def family_for_type(self, type_: TypeApplication) -> FamilyDecl:
        """The indexed family a type application is an instance of.

        Parameters
        ----------
        type_ : TypeApplication
            The applied type whose family to find.

        Returns
        -------
        FamilyDecl
            The family whose type constructor the application names.

        Raises
        ------
        KernelError
            If the application is ill-formed, or names a constructor no
            registered family declares, which is how a case analysis on
            a non-family type is rejected.
        """
        self.validate_static(type_)
        for family in self.families.values():
            if family.type_constructor == type_.constructor:
                return family
        raise KernelError(f"unknown indexed family type {type_.constructor.name!r}")


class ResumptionType(QiecModel):
    """The continuation available inside one handler clause body.

    A resumption is not a value and cannot be stored, so it is not a
    local binding. It is a capability of the position instead: available
    inside a clause body and nowhere else.

    Parameters
    ----------
    result : TypeExpr
        What the resumed operation supplies, so a `resume` must carry a
        value of this type.
    answer : TypeExpr
        What resuming produces, which is the clause's own answer type.
    effects : EffectRow
        The row resuming performs.
    """

    result: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    answer: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    effects: EffectRow = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class CheckContext(QiecModel):
    """What is in scope at one point in a term being checked.

    Parameters
    ----------
    locals : tuple[Local, ...]
        Value bindings, innermost last.
    givens : tuple[BranchGiven, ...]
        Equality evidence introduced by enclosing case branches, which is
        what makes an index refinement usable in the branch body.
    static_scopes : tuple[StaticScopeId, ...]
        Static scopes already entered. Tracked so a branch cannot reuse
        an enclosing scope's identity and silently capture its variables.
    static_variables : tuple[StaticVariableId, ...]
        Rigid variables currently active, tracked for the same reason.
    """

    locals: tuple[Local, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    givens: tuple[BranchGiven, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    static_scopes: tuple[StaticScopeId, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    static_variables: tuple[StaticVariableId, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    resumption: ResumptionType | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    """The continuation, when checking a handler clause body.

    ``None`` everywhere else, which is what makes `resume` outside a
    clause a checker error rather than a term with no meaning.
    """

    def extend(self, local: Local) -> CheckContext:
        """Return this context with one more value binding.

        Parameters
        ----------
        local : Local
            The binding to add.

        Returns
        -------
        CheckContext
            A context carrying the new binding innermost.

        Raises
        ------
        KernelError
            If the name is already bound. Rebinding is rejected rather
            than shadowed, so a name denotes one value throughout a body.
        """
        if any(existing.name == local.name for existing in self.locals):
            raise KernelError(f"local already bound: {local.name!r}")
        return CheckContext(
            locals=(*self.locals, local),
            givens=self.givens,
            static_scopes=self.static_scopes,
            static_variables=self.static_variables,
            resumption=self.resumption,
        )

    def with_givens(self, givens: tuple[BranchGiven, ...]) -> CheckContext:
        """Return this context with further equality evidence in scope.

        Parameters
        ----------
        givens : tuple[BranchGiven, ...]
            Evidence a case branch introduces by matching a constructor.

        Returns
        -------
        CheckContext
            A context in which that evidence may be appealed to.
        """
        return CheckContext(
            locals=self.locals,
            givens=(*self.givens, *givens),
            static_scopes=self.static_scopes,
            static_variables=self.static_variables,
            resumption=self.resumption,
        )

    def with_static_scope(
        self,
        scope: StaticScopeId,
        variables: tuple[StaticVariableId, ...] = (),
    ) -> CheckContext:
        """Return this context inside a fresh static scope.

        Parameters
        ----------
        scope : StaticScopeId
            Identity of the scope being entered.
        variables : tuple[StaticVariableId, ...]
            Rigid variables the scope introduces.

        Returns
        -------
        CheckContext
            A context recording the scope and its variables as active.

        Raises
        ------
        KernelError
            If the scope or one of its variables is already active. Reuse
            would let a branch's rigid variable be confused with an
            enclosing one, which is what keeps branch refinements from
            leaking out of the branch.
        """
        if scope in self.static_scopes:
            raise KernelError("a case branch reused an enclosing static scope")
        if set(variables) & set(self.static_variables):
            raise KernelError("a case branch reused an active static variable")
        return CheckContext(
            locals=self.locals,
            givens=self.givens,
            static_scopes=(*self.static_scopes, scope),
            static_variables=(*self.static_variables, *variables),
            resumption=self.resumption,
        )

    def with_resumption(self, resumption: ResumptionType) -> CheckContext:
        """Return this context inside a handler clause body.

        Parameters
        ----------
        resumption : ResumptionType
            The continuation the clause body may invoke.

        Returns
        -------
        CheckContext
            A context in which `resume` is available. A clause body
            nested inside another handler's clause replaces the outer
            resumption rather than stacking, since `resume` names the
            innermost clause's continuation and there is no syntax for
            reaching past it.
        """
        return CheckContext(
            locals=self.locals,
            givens=self.givens,
            static_scopes=self.static_scopes,
            static_variables=self.static_variables,
            resumption=resumption,
        )

    def local_type(self, name: str) -> TypeExpr | None:
        """The type of a value binding in scope.

        Parameters
        ----------
        name : str
            The bound name to resolve.

        Returns
        -------
        TypeExpr or None
            The binding's type, or None when the name is not bound here.
            The innermost binding wins, though `extend` rejects rebinding,
            so at most one can match.
        """
        return next(
            (local.type for local in reversed(self.locals) if local.name == name),
            None,
        )

    def given(self, equality: EqualityId) -> BranchGiven | None:
        """The equality evidence with a given identity, if in scope.

        Parameters
        ----------
        equality : EqualityId
            The identity to resolve.

        Returns
        -------
        BranchGiven or None
            The evidence, or None when it is not in scope, which is how
            an appeal to a refinement outside its branch is caught.
        """
        return next((given for given in self.givens if given.id == equality), None)


def _binder_variable(binder: TypeBinder | IndexBinder | EffectBinder) -> StaticArgument:
    """The variable a binder introduces, for instantiating it by itself.

    An operation's own telescope is instantiated at the variables it
    binds when a clause body is checked, because the clause is checked
    once for every instantiation rather than at a particular one.

    Parameters
    ----------
    binder : TypeBinder or IndexBinder or EffectBinder
        The binder to reflect.

    Returns
    -------
    StaticArgument
        A variable of the binder's own name, kind, and namespace. It
        carries no identity, so it is a declaration binder rather than a
        rigid one and substitution can reach it.
    """
    if isinstance(binder, TypeBinder):
        return TypeVariable(name=binder.name, kind=binder.kind)
    if isinstance(binder, IndexBinder):
        return IndexVariable(name=binder.name, sort=binder.sort)
    return EffectVariable(name=binder.name)


def _sort_matches(expected: IndexSort, actual: IndexSort) -> bool:
    """Whether an index of one sort is acceptable where another is wanted.

    Parameters
    ----------
    expected : IndexSort
        The sort the position requires.
    actual : IndexSort
        The sort the supplied index carries.

    Returns
    -------
    bool
        True when the index is acceptable. Sorts match exactly, except
        that a shape sort of unspecified rank accepts any rank, which is
        what lets a rank-polymorphic binder take a concrete shape.
    """
    if isinstance(expected, ShapeSort) and isinstance(actual, ShapeSort):
        return expected.rank is None or expected.rank == actual.rank
    return expected == actual


def check_static(
    term: StaticArgument,
    scope: StaticScope | None = None,
    *,
    registry: KernelRegistry | None = None,
) -> None:
    """Check the intrinsic kind/sort structure of one static term.

    Parameters
    ----------
    term : StaticArgument
        The static term to check.
    scope : StaticScope or None
        Binders in scope. None checks the term's intrinsic structure
        alone and permits no variable reference, which is the right mode
        for a closed declaration.
    registry : KernelRegistry or None
        Registry to resolve interface applications against. None skips
        that check, leaving only the structural one.

    Raises
    ------
    KernelError
        If the term is ill-kinded, references an unbound or wrongly
        kinded static variable, carries a scoped variable where a
        declaration is expected, or applies a known interface wrongly.
    """
    if registry is not None:
        registry._validate_effect_references(term)
    if isinstance(term, TypeVariable):
        if scope is not None:
            if term.identity is not None:
                raise KernelError(
                    "a scoped type variable cannot occur in a declaration"
                )
            bound_kind = scope.type_kind(term.name)
            if bound_kind is None:
                raise KernelError(f"unbound type variable {term.name!r}")
            if bound_kind != term.kind:
                raise KernelError(f"type variable {term.name!r} has the wrong kind")
        return
    if isinstance(term, TypeApplication):
        if len(term.constructor.telescope) != len(term.arguments):
            raise KernelError(
                f"type constructor {term.constructor.name!r} expects "
                f"{len(term.constructor.telescope)} static arguments"
            )
        for binder, argument in zip(
            term.constructor.telescope,
            term.arguments,
            strict=True,
        ):
            _check_binder_argument(
                binder,
                argument,
                subject=f"argument of type constructor {term.constructor.name!r}",
            )
            check_static(argument, scope)
        return
    if isinstance(term, FunctionType):
        check_type(term.parameter, scope)
        check_type(term.result, scope)
        return
    if isinstance(term, EqualityType):
        check_static(term.left, scope)
        check_static(term.right, scope)
        if isinstance(term.kind, (NatSort, ShapeSort, ContextSort, UserIndexSort)):
            index_nodes = (IndexVariable, IndexLiteral, IndexConstructor, ShapeIndex)
            if not isinstance(term.left, index_nodes) or not isinstance(
                term.right, index_nodes
            ):
                raise KernelError("index equality endpoints must be index terms")
            if not _sort_matches(term.kind, index_sort(term.left)):  # type: ignore[arg-type]
                raise KernelError("left equality endpoint has the wrong index sort")
            if not _sort_matches(term.kind, index_sort(term.right)):  # type: ignore[arg-type]
                raise KernelError("right equality endpoint has the wrong index sort")
        else:
            if (
                static_kind(term.left) != term.kind
                or static_kind(term.right) != term.kind
            ):
                raise KernelError("equality endpoints have the wrong kind")
        return
    if isinstance(term, EffectRef):
        for argument in term.arguments:
            check_static(argument, scope)
        return
    if isinstance(term, EffectVariable):
        if scope is not None:
            if term.identity is not None:
                raise KernelError(
                    "a scoped effect variable cannot occur in a declaration"
                )
            if term.name not in scope.effects:
                raise KernelError(f"unbound effect variable {term.name!r}")
        return
    if isinstance(term, IndexConstructor):
        for argument in term.arguments:
            check_static(argument, scope)
        return
    if isinstance(term, ShapeIndex):
        index_sort(term)
        for dimension in term.dimensions:
            check_static(dimension, scope)
        return
    if isinstance(term, IndexVariable):
        if scope is not None:
            if term.identity is not None:
                raise KernelError(
                    "a scoped index variable cannot occur in a declaration"
                )
            bound_sort = scope.index_sort(term.name)
            if bound_sort is None:
                raise KernelError(f"unbound index variable {term.name!r}")
            if not _sort_matches(bound_sort, term.sort):
                raise KernelError(f"index variable {term.name!r} has the wrong sort")
        return
    if isinstance(term, IndexLiteral):
        return
    raise KernelError(f"unknown static term {term!r}")


def check_type(
    type_: TypeExpr,
    scope: StaticScope | None = None,
    *,
    registry: KernelRegistry | None = None,
) -> None:
    """Check that a type expression is intrinsically well-kinded.

    Parameters
    ----------
    type_ : TypeExpr
        The type expression to check.
    scope : StaticScope or None
        Binders in scope, or None for a closed type.
    registry : KernelRegistry or None
        Registry to resolve interface applications against, or None.

    Raises
    ------
    KernelError
        If the expression is ill-kinded, or is a well-formed static term
        that is not a type, such as an index where a type is required.
    """
    check_static(type_, scope, registry=registry)
    if static_kind(type_) != TYPE:
        raise KernelError(f"expected a type, got {render_static(type_)}")


def _check_binder_argument(
    binder: TypeBinder | IndexBinder | EffectBinder,
    argument: StaticArgument,
    *,
    subject: str,
) -> None:
    """Check one static argument against the binder it instantiates.

    Parameters
    ----------
    binder : TypeBinder or IndexBinder or EffectBinder
        The binder being instantiated, which fixes the class of argument
        allowed and the kind or sort it must carry.
    argument : StaticArgument
        The supplied argument.
    subject : str
        What is being checked, named in any diagnostic so the error
        points at the source position rather than at the binder.

    Raises
    ------
    KernelError
        If the argument is of the wrong static class for the binder, or
        of the right class but the wrong kind or index sort.
    """
    if isinstance(binder, TypeBinder):
        if not isinstance(
            argument,
            (TypeVariable, TypeApplication, FunctionType, EqualityType),
        ):
            raise KernelError(f"{subject} must be a type argument")
        try:
            actual = static_kind(argument)
        except TypeError as error:
            raise KernelError(f"{subject} must be a type argument") from error
        if actual != binder.kind:
            raise KernelError(f"{subject} has the wrong kind")
        return
    if isinstance(binder, IndexBinder):
        if not isinstance(
            argument,
            (IndexVariable, IndexLiteral, IndexConstructor, ShapeIndex),
        ):
            raise KernelError(f"{subject} must be an index argument")
        if not _sort_matches(binder.sort, index_sort(argument)):
            raise KernelError(f"{subject} has the wrong index sort")
        return
    if not isinstance(argument, (EffectRef, EffectVariable)):
        raise KernelError(f"{subject} must be an effect argument")


def check_evidence(evidence: EqualityEvidence, context: CheckContext) -> EqualityType:
    """Check an appeal to equality evidence and return what it proves.

    Parameters
    ----------
    evidence : EqualityEvidence
        The appeal being made, either reflexivity or an appeal to a
        branch given.
    context : CheckContext
        Scope the appeal is made in, which decides whether a branch given
        is still available.

    Returns
    -------
    EqualityType
        The equality the evidence establishes.

    Raises
    ------
    KernelError
        If the stated equality is ill-kinded, if reflexivity is claimed
        between unequal endpoints, if a branch given is appealed to
        outside its branch or with an equality other than the one it
        carries, or if the evidence is of an unknown class.
    """
    check_type(evidence.equality)
    if isinstance(evidence, Reflexivity):
        if evidence.equality.left != evidence.equality.right:
            raise KernelError("reflexivity evidence has unequal endpoints")
        return evidence.equality
    if isinstance(evidence, BranchGiven):
        available = context.given(evidence.id)
        if available is None or available.equality != evidence.equality:
            raise KernelError("branch equality evidence escaped or was forged")
        return evidence.equality
    raise KernelError(f"unknown equality evidence {evidence!r}")


def _check_literal(value: object, type_: TypeExpr) -> None:
    """Check a literal's runtime value against its declared type.

    Parameters
    ----------
    value : object
        The literal's value.
    type_ : TypeExpr
        The type it claims.

    Raises
    ------
    KernelError
        If the value does not inhabit the type, or the type is outside
        the literal-supporting set. Booleans are rejected where `Int` or
        `Real` is claimed even though Python treats them as numbers,
        since the kernel keeps the two apart.
    """
    if type_ == UNIT:
        if value is not None:
            raise KernelError("Unit literals must be None")
        return
    if type_ == BOOL:
        if not isinstance(value, bool):
            raise KernelError("Bool literal has a non-boolean value")
        return
    if type_ == INT:
        if not isinstance(value, int) or isinstance(value, bool):
            raise KernelError("Int literal has a non-integer value")
        return
    if type_ == REAL:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise KernelError("Real literal has a non-numeric value")
        return
    if type_ == STRING:
        if not isinstance(value, str):
            raise KernelError("String literal has a non-string value")
        return
    raise KernelError(
        "literal values are restricted to Unit, Bool, Int, Real, and String"
    )


def _combine_substitutions(
    first: StaticSubstitution,
    second: StaticSubstitution,
) -> StaticSubstitution:
    """Concatenate two static substitutions.

    Parameters
    ----------
    first : StaticSubstitution
        The outer substitution.
    second : StaticSubstitution
        The inner one, applied in the scope the first has already opened.

    Returns
    -------
    StaticSubstitution
        One substitution carrying both sets of bindings across all three
        static namespaces.
    """
    return StaticSubstitution(
        types=(*first.types, *second.types),
        indices=(*first.indices, *second.indices),
        effects=(*first.effects, *second.effects),
    )


def _computation_static_scopes(computation: Computation) -> tuple[StaticScopeId, ...]:
    """Every static scope a computation opens, including nested ones.

    Parameters
    ----------
    computation : Computation
        The term to walk.

    Returns
    -------
    tuple[StaticScopeId, ...]
        The scopes in traversal order. Only case branches open one, so a
        computation without case analysis contributes nothing.

    Raises
    ------
    KernelError
        If the term is of an unknown computation class.
    """
    scopes: list[StaticScopeId] = []
    pending: list[Computation] = [computation]
    while pending:
        term = pending.pop()
        if isinstance(term, Return | Perform | Call | Resume):
            continue
        if isinstance(term, NewInstance):
            pending.append(term.body)
        elif isinstance(term, Bind):
            pending.append(term.then)
            pending.append(term.first)
        elif isinstance(term, Handle):
            pending.append(term.computation)
        elif isinstance(term, If):
            pending.append(term.otherwise)
            pending.append(term.then)
        elif isinstance(term, Case):
            for branch in reversed(term.branches):
                pending.append(branch.body)
                scopes.append(branch.scope)
        else:
            raise KernelError(f"unknown computation term {term!r}")
    return tuple(scopes)


def _static_variable_identities(
    term: StaticArgument,
) -> tuple[StaticVariableId, ...]:
    """Every scoped static variable occurring in a term.

    Parameters
    ----------
    term : StaticArgument
        The term to walk.

    Returns
    -------
    tuple[StaticVariableId, ...]
        Identities of the scoped variables found. A variable with no
        identity is a declaration binder rather than a rigid branch
        variable and is not reported, since only the latter can escape.
    """
    if isinstance(term, TypeVariable | IndexVariable | EffectVariable):
        return (term.identity,) if term.identity is not None else ()
    if isinstance(term, TypeApplication | EffectRef | IndexConstructor):
        return tuple(
            identity
            for argument in term.arguments
            for identity in _static_variable_identities(argument)
        )
    if isinstance(term, FunctionType):
        return (
            *_static_variable_identities(term.parameter),
            *_static_variable_identities(term.result),
        )
    if isinstance(term, EqualityType):
        return (
            *_static_variable_identities(term.left),
            *_static_variable_identities(term.right),
        )
    if isinstance(term, ShapeIndex):
        return tuple(
            identity
            for dimension in term.dimensions
            for identity in _static_variable_identities(dimension)
        )
    return ()


def _check_static_variable_scope(
    term: StaticArgument,
    context: CheckContext,
    *,
    subject: str,
) -> None:
    """Reject a term mentioning a rigid variable that is out of scope.

    This is what confines a case branch's index refinement to the branch:
    a variable introduced by matching a constructor may be used inside
    the branch body, and a type escaping the branch may not mention it.

    Parameters
    ----------
    term : StaticArgument
        The term to check, typically a branch's result type or effect
        row.
    context : CheckContext
        Scope the term must be valid in, carrying the active variables.
    subject : str
        What is being checked, named in any diagnostic.

    Raises
    ------
    KernelError
        If the term mentions a rigid variable not active in `context`.
    """
    escaped = set(_static_variable_identities(term)) - set(context.static_variables)
    if escaped:
        raise KernelError(
            f"{subject} contains a static variable that escapes its case branch"
        )


def _checked_computation_type(
    effects: EffectRow,
    result: TypeExpr,
    registry: KernelRegistry,
    context: CheckContext,
) -> ComputationType:
    """Assemble a computation type and check it is legal where it stands.

    Parameters
    ----------
    effects : EffectRow
        The row the computation performs.
    result : TypeExpr
        The value it returns.
    registry : KernelRegistry
        Registry the two strata are validated against.
    context : CheckContext
        Scope the type must be valid in.

    Returns
    -------
    ComputationType
        The assembled type.

    Raises
    ------
    KernelError
        If either stratum is ill-formed, or if the result type or an
        effect mentions a rigid variable that does not survive out to
        this point.
    """
    type_ = ComputationType(effects=effects, result=result)
    registry.validate_computation_type(type_)
    _check_static_variable_scope(result, context, subject="computation result type")
    for entry in effects.entries:
        _check_static_variable_scope(
            entry.effect,
            context,
            subject="computation effect row",
        )
    return type_


def _parameter_expectation(
    parameter: FamilyParameter, actual: TypeExpr, plate: PlateShape = PlateShape()
) -> str | None:
    """Explain why a value's type does not fit a family parameter.

    Parameters
    ----------
    parameter : FamilyParameter
        The parameter, whose constraint and rank fix the admissible
        types.
    actual : TypeExpr
        The supplied value's type.
    plate : PlateShape
        The plate the family is constructed over. A parameter may carry
        leading dimensions following a suffix of the plate's axes and
        broadcast over the rest.

    Returns
    -------
    str | None
        ``None`` when the type fits, else a description of what the
        parameter takes. A ``"sampleable"`` parameter takes any
        ``Sampleable``; a ``"transform"`` parameter takes a ``String``;
        an integer constraint takes ``Int`` and a Boolean one ``Bool``,
        otherwise ``Real``; a parameter of rank ``r`` takes that element
        under ``r`` dimensions, or, over a plate, a scalar broadcast to
        every position or a tensor whose leading dimensions are a
        suffix of the plate's axes.
    """
    if parameter.constraint == "sampleable":
        if sampled_element(actual) is not None:
            return None
        shape = tensor_shape(actual)
        if (
            shape is not None
            and len(shape[1]) == 1
            and sampled_element(shape[0]) is not None
        ):
            return None
        return "a Sampleable, or a vector of them"
    if parameter.constraint == "transform":
        return None if actual == STRING else "a String naming the transforms"
    if parameter.constraint == "numeric":
        if actual in (INT, REAL):
            return None
        shape = tensor_shape(actual)
        if shape is not None and shape[0] in (INT, REAL):
            return None
        return "an Int or a Real"
    if parameter.constraint == "boolean":
        element: TypeApplication = BOOL
    elif "integer" in parameter.constraint:
        element = INT
    else:
        element = REAL
    axes = tuple(axis.size for axis in (*plate.batch, *plate.event))
    if actual == element and (parameter.rank == 0 or axes):
        return None
    shape = tensor_shape(actual)
    if shape is not None and shape[0] == element:
        leading = len(shape[1]) - parameter.rank
        if leading == 0:
            return None
        if (
            0 < leading <= len(axes)
            and tuple(shape[1][:leading]) == axes[len(axes) - leading :]
        ):
            return None
    if parameter.rank == 0 and not axes:
        return f"a {element.constructor.name}"
    return f"a Tensor[{element.constructor.name}] of rank {parameter.rank}" + (
        ", or one whose leading dimensions follow the plate" if axes else ""
    )


def _plated_sample_type(
    value: DistributionValue,
    family: DistributionFamily,
    argument_types: Mapping[str, TypeExpr],
) -> TypeExpr:
    """The sample type of a construction, from its plate and arguments.

    Parameters
    ----------
    value : DistributionValue
        The construction.
    family : DistributionFamily
        Its family.
    argument_types : Mapping[str, TypeExpr]
        The inferred types of the supplied parameters.

    Returns
    -------
    TypeExpr
        ``Sampleable[E]`` where ``E`` is the family's element under the
        plate's batch axes, its event axes, and the family's own event
        dimensions; the family's own dimensions come from the trailing
        event axes when the plate names enough of them, else from the
        parameter the registry names as their source.

    Raises
    ------
    KernelError
        If the event shape cannot be read off the plate or the source
        parameter, or the two disagree.
    """
    element = family.element
    numeric = next(
        (item.name for item in family.parameters if item.constraint == "numeric"),
        None,
    )
    if numeric is not None and numeric in argument_types:
        # An atom samples what its value is: an integer atom is a mass on
        # the integers, a real one on the reals.
        atom_type = argument_types[numeric]
        shape = tensor_shape(atom_type)
        element = shape[0] if shape is not None else atom_type
    if family.compositional:
        # A measure built from another samples what that one samples: the
        # base of a restriction or normalization, the components of a
        # mixture, whichever the family names.
        for name in ("base", "component"):
            source_type = argument_types.get(name)
            if source_type is None:
                continue
            inner = sampled_element(source_type)
            if inner is None:
                shape = tensor_shape(source_type)
                inner = sampled_element(shape[0]) if shape is not None else None
            if inner is not None:
                element = inner
            break
    element_name = (
        element.constructor.name
        if isinstance(element, TypeApplication)
        else repr(element)
    )
    plate = value.plate
    natural: tuple[IndexTerm, ...] | None = None
    supplied = family.event_argument(argument_types.items())
    source_name = supplied[0] if supplied is not None else None
    if supplied is not None and family.event_rank > 0:
        source_shape = tensor_shape(supplied[1])
        if source_shape is not None and len(source_shape[1]) >= family.event_rank:
            natural = tuple(source_shape[1][len(source_shape[1]) - family.event_rank :])
    event_axes = tuple(axis.size for axis in plate.event)
    if family.event_rank > 0:
        if len(event_axes) >= family.event_rank:
            trailing = event_axes[len(event_axes) - family.event_rank :]
            if natural is not None and tuple(natural) != tuple(trailing):
                raise KernelError(
                    f"family {value.name!r} samples the trailing shape of "
                    f"{source_name!r}, {natural!r}, but its plate's "
                    f"event axes end in {trailing!r}",
                    "qiec-distribution",
                )
            event: tuple[IndexTerm, ...] = event_axes
        elif natural is not None and not event_axes:
            event = natural
        else:
            claimed = sampled_element(value.result_type)
            shape = tensor_shape(claimed) if claimed is not None else None
            if (
                shape is None
                or shape[0] != element
                or len(shape[1]) != len(plate.batch) + family.event_rank
                or family.event_sources
            ):
                raise KernelError(
                    f"family {value.name!r} samples a Tensor[{element_name}] of "
                    f"rank {family.event_rank}, not {render_static(value.result_type)}",
                    "qiec-distribution",
                )
            event = tuple(shape[1][len(plate.batch) :])
    else:
        event = event_axes
    batch = tuple(axis.size for axis in plate.batch)
    dimensions = (*batch, *event)
    if not dimensions:
        return sampleable_type(element)
    return sampleable_type(tensor_type(element, dimensions))


def _infer_tensor(
    value: TensorValue, registry: KernelRegistry, context: CheckContext
) -> TypeExpr:
    """Type a tensor construction against its claimed ``Tensor`` type.

    Parameters
    ----------
    value : TensorValue
        The construction.
    registry : KernelRegistry
        The module's declarations.
    context : CheckContext
        Value bindings in scope.

    Returns
    -------
    TypeExpr
        The claimed type, once every entry has the slice type it fixes.

    Raises
    ------
    KernelError
        If the claimed type is not a literal-shaped ``Tensor``, the entry
        count differs from the leading dimension, or an entry's type is
        not the slice type (the element type at rank one, the tensor one
        rank lower otherwise).
    """
    split = tensor_shape(value.result_type)
    if split is None:
        raise KernelError(
            f"tensor construction claims non-tensor type {render_static(value.result_type)}",
            "qiec-primitive",
        )
    element, dimensions = split
    if not dimensions:
        raise KernelError(
            "tensor construction claims a rank-zero tensor type", "qiec-primitive"
        )
    leading = dimensions[0]
    if not (isinstance(leading, IndexLiteral) and isinstance(leading.value, int)):
        raise KernelError(
            f"tensor construction claims a leading dimension {leading!r} that is "
            "not a literal",
            "qiec-primitive",
        )
    if leading.value != len(value.items):
        raise KernelError(
            f"tensor construction has {len(value.items)} entries along a leading "
            f"dimension of {leading.value}",
            "qiec-primitive",
        )
    slice_type: TypeExpr = (
        element if len(dimensions) == 1 else tensor_type(element, dimensions[1:])
    )
    for position, item in enumerate(value.items):
        actual = infer_value(item, registry, context)
        if actual != slice_type:
            raise KernelError(
                f"tensor entry {position} has type {render_static(actual)}, not the slice type "
                f"{render_static(slice_type)}",
                "qiec-primitive",
            )
    registry.validate_type(value.result_type)
    _check_static_variable_scope(value.result_type, context, subject="tensor type")
    return value.result_type


def _infer_gather(
    value: Gather, registry: KernelRegistry, context: CheckContext
) -> TypeExpr:
    """Type a selection along a tensor's outermost axis.

    Parameters
    ----------
    value : Gather
        The selection.
    registry : KernelRegistry
        The module's declarations.
    context : CheckContext
        Value bindings in scope.

    Returns
    -------
    TypeExpr
        The slice type for an ``Int`` index, or a tensor of slices shaped
        like a ``Tensor[Int]`` index.

    Raises
    ------
    KernelError
        If the value is not a tensor, the index is neither an ``Int`` nor
        a tensor of them, or the claimed type is not the selection's.
    """
    source = tensor_shape(infer_value(value.value, registry, context))
    if source is None:
        raise KernelError("gather selects from a Tensor", "qiec-primitive")
    element, dimensions = source
    rest = dimensions[1:]
    index = infer_value(value.index, registry, context)
    if index == INT:
        expected: TypeExpr = element if not rest else tensor_type(element, rest)
    else:
        index_shape = tensor_shape(index)
        if index_shape is None or index_shape[0] != INT:
            raise KernelError(
                f"gather indexes with an Int or a Tensor[Int], not {index!r}",
                "qiec-primitive",
            )
        expected = tensor_type(element, (*index_shape[1], *rest))
    if value.result_type != expected:
        raise KernelError(
            f"gather has type {render_static(expected)}, not the claimed {render_static(value.result_type)}",
            "qiec-primitive",
        )
    registry.validate_type(expected)
    _check_static_variable_scope(expected, context, subject="gather type")
    return expected


def _infer_segment_sum(
    value: SegmentSum, registry: KernelRegistry, context: CheckContext
) -> TypeExpr:
    """Type per-group totals of a weight vector.

    Parameters
    ----------
    value : SegmentSum
        The totals.
    registry : KernelRegistry
        The module's declarations.
    context : CheckContext
        Value bindings in scope.

    Returns
    -------
    TypeExpr
        ``Tensor[LogWeight]([groups])``.

    Raises
    ------
    KernelError
        If the weights and the index are not vectors of one length, or
        the claimed type is not a weight vector over the groups.
    """
    weights = tensor_shape(infer_value(value.value, registry, context))
    index = tensor_shape(infer_value(value.index, registry, context))
    if (
        weights is None
        or index is None
        or weights[0] != LOG_WEIGHT
        or index[0] != INT
        or len(weights[1]) != 1
        or len(index[1]) != 1
        or weights[1] != index[1]
    ):
        raise KernelError(
            "a segment sum takes a Tensor[LogWeight] and a Tensor[Int] of one length",
            "qiec-distribution",
        )
    expected = tensor_type(LOG_WEIGHT, (value.groups,))
    if value.result_type != expected:
        raise KernelError(
            f"segment sum has type {render_static(expected)}, not the claimed {render_static(value.result_type)}",
            "qiec-distribution",
        )
    registry.validate_type(expected)
    _check_static_variable_scope(expected, context, subject="segment sum type")
    return expected


def _infer_table_map(
    value: TableMap, registry: KernelRegistry, context: CheckContext
) -> TypeExpr:
    """Type one head of a table-indexed parameter map.

    Parameters
    ----------
    value : TableMap
        The head.
    registry : KernelRegistry
        The module's declarations.
    context : CheckContext
        Value bindings in scope.

    Returns
    -------
    TypeExpr
        ``Tensor[Real]([rows])``, or ``Real`` for a one-column head.

    Raises
    ------
    KernelError
        If the table is not a real matrix, the index not an ``Int``,
        the block lies outside the table's columns, or the claimed type
        is not the head's.
    """
    table = tensor_shape(infer_value(value.table, registry, context))
    if table is None or table[0] != REAL or len(table[1]) != 2:
        raise KernelError(
            "a table map's table is a Tensor[Real] of rank 2", "qiec-primitive"
        )
    if infer_value(value.index, registry, context) != INT:
        raise KernelError("a table map's index is an Int", "qiec-primitive")
    if value.rows <= 0 or value.row_offset < 0:
        raise KernelError(
            "a table map's block needs a nonnegative offset and positive width",
            "qiec-primitive",
        )
    columns = table[1][1]
    if isinstance(columns, IndexLiteral) and isinstance(columns.value, int):
        if value.row_offset + value.rows > columns.value:
            raise KernelError(
                "a table map's block lies outside its table", "qiec-primitive"
            )
    expected: TypeExpr = (
        REAL
        if value.rows == 1
        else tensor_type(REAL, (IndexLiteral(value=value.rows, sort=NAT),))
    )
    if value.result_type != expected:
        raise KernelError(
            f"table map has type {render_static(expected)}, not the claimed "
            f"{render_static(value.result_type)}",
            "qiec-primitive",
        )
    return expected


def _infer_affine_map(
    value: AffineMap, registry: KernelRegistry, context: CheckContext
) -> TypeExpr:
    """Type one head of an affine parameter map.

    Parameters
    ----------
    value : AffineMap
        The head.
    registry : KernelRegistry
        The module's declarations.
    context : CheckContext
        Value bindings in scope.

    Returns
    -------
    TypeExpr
        ``Tensor[Real]([rows])``, or ``Real`` for a one-row head.

    Raises
    ------
    KernelError
        If the weight is not a real matrix, the bias not a real vector,
        a source not a real scalar or vector, the block lies outside the
        weight, the sources' widths do not fill the weight's columns, or
        the claimed type is not the head's.
    """
    weight = tensor_shape(infer_value(value.weight, registry, context))
    bias = tensor_shape(infer_value(value.bias, registry, context))
    if weight is None or weight[0] != REAL or len(weight[1]) != 2:
        raise KernelError(
            "an affine map's weight is a Tensor[Real] of rank 2", "qiec-primitive"
        )
    if bias is None or bias[0] != REAL or len(bias[1]) != 1:
        raise KernelError(
            "an affine map's bias is a Tensor[Real] of rank 1", "qiec-primitive"
        )
    if value.rows <= 0 or value.row_offset < 0:
        raise KernelError(
            "an affine map's block needs a nonnegative offset and positive height",
            "qiec-primitive",
        )
    total_rows, columns = weight[1]
    if isinstance(total_rows, IndexLiteral) and isinstance(total_rows.value, int):
        if value.row_offset + value.rows > total_rows.value:
            raise KernelError(
                "an affine map's block lies outside its weight", "qiec-primitive"
            )
    if bias[1][0] != total_rows:
        raise KernelError(
            "an affine map's bias has one entry per weight row", "qiec-primitive"
        )
    width = 0
    for source in value.sources:
        actual = infer_value(source, registry, context)
        if actual == REAL:
            width += 1
            continue
        shape = tensor_shape(actual)
        if (
            shape is None
            or shape[0] != REAL
            or len(shape[1]) != 1
            or not isinstance(shape[1][0], IndexLiteral)
            or not isinstance(shape[1][0].value, int)
        ):
            raise KernelError(
                "an affine map's sources are Reals and literal-width real vectors",
                "qiec-primitive",
            )
        width += shape[1][0].value
    if isinstance(columns, IndexLiteral) and columns.value != width:
        raise KernelError(
            f"an affine map's sources fill {width} columns of a {columns.value}-wide "
            "weight",
            "qiec-primitive",
        )
    expected: TypeExpr = (
        REAL
        if value.rows == 1
        else tensor_type(REAL, (IndexLiteral(value=value.rows, sort=NAT),))
    )
    if value.result_type != expected:
        raise KernelError(
            f"affine map has type {render_static(expected)}, not the claimed {render_static(value.result_type)}",
            "qiec-primitive",
        )
    return expected


def _infer_distribution(
    value: DistributionValue,
    registry: KernelRegistry,
    context: CheckContext,
) -> TypeExpr:
    """Check a distribution construction against the family registry.

    Parameters
    ----------
    value : DistributionValue
        The construction.
    registry : KernelRegistry
        Declarations the argument values are checked against.
    context : CheckContext
        Bindings in scope.

    Returns
    -------
    TypeExpr
        The ``Sampleable`` type the construction claims, once verified.

    Raises
    ------
    KernelError
        If the family is unknown or applied under another family's
        identity, an argument names no parameter or is repeated, an
        argument's type does not fit its parameter, the family's event
        shape cannot be read off the argument that determines it, or the
        claimed result type disagrees with the family's sample type.
    """
    family = FAMILIES.get(value.name)
    if family is None:
        raise KernelError(
            f"unknown distribution family {value.name!r}", "qiec-distribution"
        )
    if family.id != value.family:
        raise KernelError(
            f"family {value.name!r} is applied under the identity of another family",
            "qiec-distribution",
        )
    seen: set[str] = set()
    argument_types: dict[str, TypeExpr] = {}
    for name, argument in value.arguments:
        if name in seen:
            raise KernelError(
                f"family {value.name!r} is given parameter {name!r} twice",
                "qiec-distribution",
            )
        seen.add(name)
        try:
            parameter = family.parameter(name)
        except KeyError as error:
            raise KernelError(
                f"family {value.name!r} has no parameter {name!r}; it takes "
                f"{', '.join(family.parameter_names)}",
                "qiec-distribution",
            ) from error
        actual = infer_value(argument, registry, context)
        expectation = _parameter_expectation(parameter, actual, value.plate)
        if expectation is not None:
            raise KernelError(
                f"parameter {name!r} of {value.name!r} has type {render_static(actual)}; it "
                f"takes {expectation}",
                "qiec-distribution",
            )
        argument_types[name] = actual
    if not value.arguments:
        raise KernelError(
            f"family {value.name!r} is constructed with no parameters",
            "qiec-distribution",
        )
    expected = _plated_sample_type(value, family, argument_types)
    if value.result_type != expected:
        raise KernelError(
            f"family {value.name!r} produces {render_static(expected)}, not the claimed "
            f"{render_static(value.result_type)}",
            "qiec-distribution",
        )
    registry.validate_type(expected)
    _check_static_variable_scope(expected, context, subject="distribution type")
    return expected


def infer_value(
    value: Value,
    registry: KernelRegistry,
    context: CheckContext = CheckContext(),
) -> TypeExpr:
    """Infer the type of a pure value term.

    Parameters
    ----------
    value : Value
        The value to check.
    registry : KernelRegistry
        Declarations the value's types are resolved against.
    context : CheckContext
        Bindings and evidence in scope. The default empty context suits a
        closed value.

    Returns
    -------
    TypeExpr
        The inferred type.

    Raises
    ------
    KernelError
        If a local is unbound or carries a type other than the one the
        term claims, if a literal's value does not inhabit its type, if
        a constructor is applied wrongly, if a let binds a value of a
        type other than its binder's or rebinds a name in scope, if
        equality evidence is forged or out of scope, or if an inferred
        type mentions a rigid variable that escapes its branch.
    """
    if isinstance(value, Var):
        actual = context.local_type(value.local.name)
        if actual is None or actual != value.local.type:
            raise KernelError(f"unbound or mistyped local {value.local.name!r}")
        registry.validate_type(actual)
        _check_static_variable_scope(actual, context, subject="local type")
        return actual
    if isinstance(value, LiteralValue):
        registry.validate_type(value.type)
        _check_static_variable_scope(value.type, context, subject="literal type")
        _check_literal(value.value, value.type)
        return value.type
    if isinstance(value, AttachmentRef):
        registry.validate_type(value.type)
        _check_static_variable_scope(value.type, context, subject="attachment type")
        return value.type
    if isinstance(value, EvidenceValue):
        equality = check_evidence(value.evidence, context)
        registry.validate_type(equality)
        _check_static_variable_scope(equality, context, subject="equality evidence")
        return equality
    if isinstance(value, TransportValue):
        equality = check_evidence(value.evidence, context)
        registry.validate_type(equality)
        _check_static_variable_scope(equality, context, subject="transport equality")
        source = infer_value(value.value, registry, context)
        registry.validate_type(value.target_type)
        _check_static_variable_scope(
            value.target_type,
            context,
            subject="transport target",
        )
        if equality.left != source or equality.right != value.target_type:
            raise KernelError("transport endpoints do not match value and target types")
        return value.target_type
    if isinstance(value, PrimitiveApplication):
        signature = PRIMITIVES.get(value.name)
        if signature is None:
            raise KernelError(f"unknown primitive {value.name!r}", "qiec-primitive")
        if signature.id != value.primitive:
            raise KernelError(
                f"primitive {value.name!r} is applied under the identity of "
                "another primitive",
                "qiec-primitive",
            )
        if len(value.arguments) != len(signature.parameters):
            raise KernelError(
                f"primitive {value.name!r} takes {len(signature.parameters)} "
                f"argument(s), got {len(value.arguments)}",
                "qiec-primitive",
            )
        broadcast: tuple[IndexTerm, ...] | None = None
        for position, (argument, expected) in enumerate(
            zip(value.arguments, signature.parameters, strict=True)
        ):
            actual = infer_value(argument, registry, context)
            if actual == expected:
                continue
            shape = tensor_shape(actual)
            if shape is None or shape[0] != expected:
                raise KernelError(
                    f"argument {position} of primitive {value.name!r} has type "
                    f"{actual!r}; the primitive takes {expected!r} or a Tensor "
                    "of it",
                    "qiec-primitive",
                )
            if broadcast is not None and tuple(shape[1]) != broadcast:
                raise KernelError(
                    f"primitive {value.name!r} is applied to tensors of shapes "
                    f"{broadcast!r} and {tuple(shape[1])!r}, which do not agree",
                    "qiec-primitive",
                )
            broadcast = tuple(shape[1])
        expected_result: TypeExpr = (
            signature.result
            if broadcast is None
            else tensor_type(signature.result, broadcast)
        )
        if value.result_type != expected_result:
            raise KernelError(
                f"primitive {value.name!r} produces {render_static(expected_result)}, not "
                f"the claimed {render_static(value.result_type)}",
                "qiec-primitive",
            )
        return expected_result
    if isinstance(value, DistributionValue):
        return _infer_distribution(value, registry, context)
    if isinstance(value, SiteValue):
        if not value.label:
            raise KernelError("a site label cannot be empty", "qiec-distribution")
        if not (
            isinstance(value.result_type, TypeApplication)
            and value.result_type.constructor == SITE_CONSTRUCTOR
            and len(value.result_type.arguments) == 1
        ):
            raise KernelError(
                f"site {value.label!r} claims {render_static(value.result_type)}, not a Site",
                "qiec-distribution",
            )
        registry.validate_type(value.result_type)
        _check_static_variable_scope(value.result_type, context, subject="site type")
        return value.result_type
    if isinstance(value, LogDensity):
        sampleable = infer_value(value.sampleable, registry, context)
        element = sampled_element(sampleable)
        if element is None:
            raise KernelError(
                f"log_prob takes a Sampleable, not {sampleable!r}",
                "qiec-distribution",
            )
        actual = infer_value(value.value, registry, context)
        if actual != element:
            raise KernelError(
                f"log_prob evaluates a distribution over {render_static(element)} at a value "
                f"of type {render_static(actual)}",
                "qiec-distribution",
            )
        if not value.batch:
            return LOG_WEIGHT
        if not (
            isinstance(value.sampleable, DistributionValue)
            and value.sampleable.plate.batch == value.batch
        ):
            raise KernelError(
                "log_prob keeps its weights apart over batch axes only for a "
                "construction plated over those axes",
                "qiec-distribution",
            )
        return tensor_type(LOG_WEIGHT, tuple(axis.size for axis in value.batch))
    if isinstance(value, Gather):
        return _infer_gather(value, registry, context)
    if isinstance(value, WeightSum):
        actual = infer_value(value.value, registry, context)
        shape = tensor_shape(actual)
        if actual != LOG_WEIGHT and (shape is None or shape[0] != LOG_WEIGHT):
            raise KernelError(
                f"a weight sum totals log weights, not {actual!r}",
                "qiec-distribution",
            )
        return LOG_WEIGHT
    if isinstance(value, SegmentSum):
        return _infer_segment_sum(value, registry, context)
    if isinstance(value, KernelMatrix):
        actual = infer_value(value.inputs, registry, context)
        shape = tensor_shape(actual)
        if shape is None or shape[0] != REAL or len(shape[1]) != 1:
            raise KernelError(
                f"a kernel matrix takes a Tensor[Real] of rank 1, not {actual!r}",
                "qiec-distribution",
            )
        if value.kernel != "rbf":
            raise KernelError(
                f"unknown kernel {value.kernel!r}; the kernels are rbf",
                "qiec-distribution",
            )
        if not value.length_scale > 0 or value.jitter < 0:
            raise KernelError(
                "a kernel matrix needs a positive length scale and a "
                "nonnegative jitter",
                "qiec-distribution",
            )
        expected = tensor_type(REAL, (shape[1][0], shape[1][0]))
        if value.result_type != expected:
            raise KernelError(
                f"kernel matrix has type {render_static(expected)}, not the claimed "
                f"{render_static(value.result_type)}",
                "qiec-distribution",
            )
        return expected
    if isinstance(value, AffineMap):
        return _infer_affine_map(value, registry, context)
    if isinstance(value, TableMap):
        return _infer_table_map(value, registry, context)
    if isinstance(value, Reduction):
        shape = tensor_shape(infer_value(value.value, registry, context))
        if shape is None:
            raise KernelError(
                f"a {value.operator} reduction takes a Tensor", "qiec-primitive"
            )
        element = shape[0]
        integral = value.operator in ("sum", "max", "min", "prod")
        if element != REAL and not (integral and element == INT):
            raise KernelError(
                f"a {value.operator} reduction takes a Tensor[Real]"
                + (" or a Tensor[Int]" if integral else "")
                + f", not one of {element!r}",
                "qiec-primitive",
            )
        if value.result_type != element:
            raise KernelError(
                f"reduction has type {render_static(element)}, not the claimed "
                f"{render_static(value.result_type)}",
                "qiec-primitive",
            )
        return element
    if isinstance(value, Rowwise):
        actual = infer_value(value.value, registry, context)
        shape = tensor_shape(actual)
        if shape is None or shape[0] != REAL:
            raise KernelError(
                f"a {value.operator} takes a Tensor[Real], not {actual!r}",
                "qiec-primitive",
            )
        if value.result_type != actual:
            raise KernelError(
                f"{value.operator} keeps its argument's type {render_static(actual)}, not the "
                f"claimed {render_static(value.result_type)}",
                "qiec-primitive",
            )
        return actual
    if isinstance(value, Comprehension):
        if value.binder.type != INT:
            raise KernelError(
                "a comprehension's binder is an Int index", "qiec-primitive"
            )
        registry.validate_type(value.result_type)
        body = infer_value(value.body, registry, context.extend(value.binder))
        inner = tensor_shape(body)
        expected = (
            tensor_type(body, (value.extent,))
            if inner is None
            else tensor_type(inner[0], (value.extent, *inner[1]))
        )
        if value.result_type != expected:
            raise KernelError(
                f"comprehension has type {render_static(expected)}, not the claimed "
                f"{render_static(value.result_type)}",
                "qiec-primitive",
            )
        _check_static_variable_scope(expected, context, subject="comprehension type")
        return expected
    if isinstance(value, LetValue):
        if not value.bindings:
            raise KernelError("a let binds at least one value", "qiec-primitive")
        inner = context
        for binding in value.bindings:
            bound = infer_value(binding.value, registry, inner)
            if bound != binding.binder.type:
                raise KernelError(
                    f"let binds {binding.binder.name!r} at "
                    f"{render_static(binding.binder.type)} a value of type "
                    f"{render_static(bound)}",
                    "qiec-primitive",
                )
            inner = inner.extend(binding.binder)
        body = infer_value(value.body, registry, inner)
        if value.result_type != body:
            raise KernelError(
                f"let has type {render_static(body)}, not the claimed "
                f"{render_static(value.result_type)}",
                "qiec-primitive",
            )
        _check_static_variable_scope(body, context, subject="let type")
        return body
    if isinstance(value, TupleValue):
        components = tuple(infer_value(item, registry, context) for item in value.items)
        expected = product_type(*components)
        if value.result_type != expected:
            raise KernelError(
                f"tuple has type {render_static(expected)}, not the claimed {render_static(value.result_type)}",
                "qiec-primitive",
            )
        registry.validate_type(expected)
        _check_static_variable_scope(expected, context, subject="tuple type")
        return expected
    if isinstance(value, TensorValue):
        return _infer_tensor(value, registry, context)
    if isinstance(value, Projection):
        source = infer_value(value.value, registry, context)
        if not (
            isinstance(source, TypeApplication)
            and source.constructor.name.startswith("Product")
            and source.constructor.id == product_type(*source.arguments).constructor.id  # type: ignore[arg-type]
        ):
            raise KernelError(
                f"projection from a value of non-product type {render_static(source)}",
                "qiec-primitive",
            )
        if not 0 <= value.position < len(source.arguments):
            raise KernelError(
                f"projection position {value.position} is outside a product of "
                f"{len(source.arguments)} components",
                "qiec-primitive",
            )
        component = source.arguments[value.position]
        if value.result_type != component:
            raise KernelError(
                f"projection {value.position} has type {render_static(component)}, not the "
                f"claimed {render_static(value.result_type)}",
                "qiec-primitive",
            )
        return component  # type: ignore[return-value]
    if isinstance(value, ConstructorValue):
        constructor = registry.constructor(value.constructor)
        family = registry.families[constructor.family]
        for argument in value.static_arguments:
            registry.validate_static(argument)
            _check_static_variable_scope(
                argument,
                context,
                subject="constructor static argument",
            )
        registry.validate_type(value.result_type)
        _check_static_variable_scope(
            value.result_type,
            context,
            subject="constructor result type",
        )
        parameter_count = len(family.parameters)
        local_count = len(constructor.telescope)
        if len(value.static_arguments) != parameter_count + local_count:
            raise KernelError(
                f"constructor {constructor.name!r} expects "
                f"{parameter_count + local_count} static arguments"
            )
        parameters = instantiate_telescope(
            family.parameters,
            value.static_arguments[:parameter_count],
        )
        locals_ = instantiate_telescope(
            constructor.telescope,
            value.static_arguments[parameter_count:],
        )
        substitution = _combine_substitutions(parameters, locals_)
        if len(value.fields) != len(constructor.fields):
            raise KernelError(
                f"constructor {constructor.name!r} expects {len(constructor.fields)} fields"
            )
        for field_value, field in zip(value.fields, constructor.fields, strict=True):
            actual = infer_value(field_value, registry, context)
            expected = substitute_type(field.type, substitution)
            if actual != expected:
                raise KernelError(
                    f"field {field.name!r} has type {render_static(actual)}, expected {render_static(expected)}"
                )
        parameter_arguments = value.static_arguments[:parameter_count]
        result_indices = tuple(
            substitute_static(index, substitution)
            for index in constructor.result_indices
        )
        expected_result = TypeApplication(
            constructor=family.type_constructor,
            arguments=(*parameter_arguments, *result_indices),
        )
        if value.result_type != expected_result:
            raise KernelError(
                f"constructor result is {render_static(value.result_type)}, expected {render_static(expected_result)}"
            )
        return expected_result
    raise KernelError(f"unknown value term {value!r}")


def check_request(
    request: EffectRequest,
    registry: KernelRegistry,
    context: CheckContext,
) -> tuple[EffectRef, TypeExpr]:
    """Check one effect request and report what performing it yields.

    Parameters
    ----------
    request : EffectRequest
        The request to check, naming an instance, an interface
        application, an operation, and the arguments it carries.
    registry : KernelRegistry
        Declarations the request is resolved against.
    context : CheckContext
        Scope the request is made in.

    Returns
    -------
    tuple[EffectRef, TypeExpr]
        The interface application the request performs, and the type
        resuming it supplies. The interface comes back because the caller
        needs it to build the row this request adds to.

    Raises
    ------
    KernelError
        If the operation is unknown, if the named interface does not own
        it, if the static or value arguments do not match the operation's
        signature, if the declared result type disagrees with the
        instantiated one, or if any part mentions a rigid variable out of
        scope.
    """
    registry.validate_static(request.effect)
    _check_static_variable_scope(request.effect, context, subject="request effect")
    for argument in request.static_arguments:
        registry.validate_static(argument)
        _check_static_variable_scope(
            argument,
            context,
            subject="request static argument",
        )
    registry.validate_type(request.result_type)
    _check_static_variable_scope(
        request.result_type,
        context,
        subject="request result type",
    )
    effect, operation_object = registry.operation(request.operation)
    if not effect.matches(request.effect):
        raise KernelError("request effect interface does not own the operation")
    interface_substitution = instantiate_telescope(
        effect.telescope,
        request.effect.arguments,
    )
    argument_types, result_type = instantiate_operation(
        operation_object,
        request.static_arguments,
        interface_substitution,
    )
    if len(request.arguments) != len(argument_types):
        raise KernelError(
            f"operation {operation_object.name!r} expects {len(argument_types)} arguments"
        )
    for argument, expected in zip(request.arguments, argument_types, strict=True):
        actual = infer_value(argument, registry, context)
        if actual != expected:
            raise KernelError(
                f"operation argument has type {render_static(actual)}, expected {render_static(expected)}"
            )
    if request.result_type != result_type:
        raise KernelError(
            f"request result is {render_static(request.result_type)}, expected {render_static(result_type)}"
        )
    return request.effect, result_type


def infer_computation(
    computation: Computation,
    registry: KernelRegistry,
    context: CheckContext = CheckContext(),
) -> ComputationType:
    """Infer the result type and effect row of a computation.

    This is the kernel's entry point for checking effectful code. The
    row it returns is what remains unhandled: a `Handle` discharges its
    instance, so a fully handled computation comes back with the empty
    row.

    Parameters
    ----------
    computation : Computation
        The term to check.
    registry : KernelRegistry
        Declarations the term is resolved against.
    context : CheckContext
        Bindings and evidence in scope. The default empty context suits a
        closed computation.

    Returns
    -------
    ComputationType
        The result type paired with the effects still outstanding.

    Raises
    ------
    KernelError
        If two case branches share a static scope, if a request, value,
        or branch fails to check, if a handler's interface does not match
        the instance it discharges, if branch types or rows fail to
        unify, or if a rigid variable escapes the branch that introduced
        it.
    """
    static_scopes = _computation_static_scopes(computation)
    if len(set(static_scopes)) != len(static_scopes):
        raise KernelError("case branch static scopes must be globally fresh")
    return _infer_computation(computation, registry, context)


def _infer_computation(
    computation: Computation,
    registry: KernelRegistry,
    context: CheckContext,
) -> ComputationType:
    """Infer a computation whose static scopes are known to be fresh.

    Parameters
    ----------
    computation : Computation
        The term to check.
    registry : KernelRegistry
        Declarations the term is resolved against.
    context : CheckContext
        Bindings and evidence in scope.

    Returns
    -------
    ComputationType
        The result type paired with the effects still outstanding.

    Raises
    ------
    KernelError
        As :func:`infer_computation`, except for the scope freshness it
        checks once at the root rather than at every subterm.
    """
    if isinstance(computation, Return):
        return _checked_computation_type(
            EMPTY_ROW,
            infer_value(computation.value, registry, context),
            registry,
            context,
        )
    if isinstance(computation, Perform):
        effect, result = check_request(computation.request, registry, context)
        return _checked_computation_type(
            EffectRow(
                entries=(
                    RowEntry(instance=computation.request.instance, effect=effect),
                )
            ),
            result,
            registry,
            context,
        )
    if isinstance(computation, Bind):
        # A chain of binds is walked in a loop rather than by recursion on
        # the continuation, so a long straight-line body does not recurse
        # once per step; each first computation is still checked in the
        # context its binder extends.
        effects = EMPTY_ROW
        current: Computation = computation
        chain_context = context
        while isinstance(current, Bind):
            first = _infer_computation(current.first, registry, chain_context)
            _check_static_variable_scope(
                current.binder.type,
                chain_context,
                subject="bind type",
            )
            if first.result != current.binder.type:
                raise KernelError(
                    f"bind expects {render_static(current.binder.type)}, got {render_static(first.result)}"
                )
            effects = effects.union(first.effects)
            chain_context = chain_context.extend(current.binder)
            current = current.then
        then = _infer_computation(current, registry, chain_context)
        return _checked_computation_type(
            effects.union(then.effects),
            then.result,
            registry,
            context,
        )
    if isinstance(computation, If):
        condition = infer_value(computation.condition, registry, context)
        if condition != BOOL:
            raise KernelError(
                f"if condition has type {render_static(condition)}, not Bool",
                "qiec-primitive",
            )
        then = _infer_computation(computation.then, registry, context)
        otherwise = _infer_computation(computation.otherwise, registry, context)
        if then.result != otherwise.result:
            raise KernelError(
                f"if branches return {render_static(then.result)} and {render_static(otherwise.result)}; "
                "both must agree",
                "qiec-primitive",
            )
        return _checked_computation_type(
            then.effects.union(otherwise.effects),
            then.result,
            registry,
            context,
        )
    if isinstance(computation, Case):
        scrutinee_type = infer_value(computation.scrutinee, registry, context)
        if not isinstance(scrutinee_type, TypeApplication):
            raise KernelError("case scrutinee is not an indexed family application")
        family = registry.family_for_type(scrutinee_type)
        registry.validate_type(computation.motive.result_type)
        _check_static_variable_scope(
            computation.motive.result_type,
            context,
            subject="case motive",
        )
        if len(computation.motive.indices) != len(family.indices):
            raise KernelError("case motive has the wrong number of index binders")
        for motive_index, family_index in zip(
            computation.motive.indices,
            family.indices,
            strict=True,
        ):
            same_index = (
                (
                    isinstance(motive_index, TypeBinder)
                    and isinstance(family_index, TypeBinder)
                    and motive_index.kind == family_index.kind
                )
                or (
                    isinstance(motive_index, IndexBinder)
                    and isinstance(family_index, IndexBinder)
                    and motive_index.sort == family_index.sort
                )
                or (
                    isinstance(motive_index, EffectBinder)
                    and isinstance(family_index, EffectBinder)
                )
            )
            if not same_index:
                raise KernelError("case motive index has the wrong sort")

        constructors = tuple(
            registry.constructor(constructor) for constructor in family.constructors
        )
        coverage = check_indexed_coverage(
            family,
            constructors,
            scrutinee_type,
            tuple(
                BranchPattern(constructor=branch.constructor)
                for branch in computation.branches
            ),
        )
        if coverage.status is not CoverageStatus.COMPLETE:
            raise KernelError(
                f"case coverage is {coverage.status.value}; missing={coverage.missing!r}, "
                f"duplicates={coverage.duplicates!r}"
            )

        parameter_count = len(family.parameters)
        actual_parameters = scrutinee_type.arguments[:parameter_count]
        actual_indices = scrutinee_type.arguments[parameter_count:]
        motive_at_scrutinee = instantiate_telescope(
            computation.motive.indices,
            actual_indices,
        )
        result_type = substitute_type(
            computation.motive.result_type,
            motive_at_scrutinee,
        )
        _check_static_variable_scope(
            result_type,
            context,
            subject="case result type",
        )

        branch_scopes = [branch.scope for branch in computation.branches]
        if len(set(branch_scopes)) != len(branch_scopes):
            raise KernelError("case branches must have distinct static scopes")
        branch_rows: list[tuple[Reachability, EffectRow]] = []
        for branch in computation.branches:
            constructor = registry.constructor(branch.constructor)
            static_arguments = constructor_skolems(constructor, branch.scope)
            static_variables = tuple(
                identity
                for argument in static_arguments
                for identity in _static_variable_identities(argument)
            )
            branch_context = context.with_static_scope(
                branch.scope,
                static_variables,
            )
            if branch.static_arguments != static_arguments:
                raise KernelError(
                    f"branch {constructor.name!r} must use its canonical rigid skolems"
                )
            refinement = refine_branch(
                family,
                constructor,
                scrutinee_type,
                static_arguments,
            )
            if refinement.reachability is Reachability.IMPOSSIBLE:
                continue

            parameter_substitution = instantiate_telescope(
                family.parameters,
                actual_parameters,
            )
            local_substitution = instantiate_telescope(
                constructor.telescope,
                static_arguments,
            )
            substitution = _combine_substitutions(
                parameter_substitution,
                local_substitution,
            )
            if len(branch.fields) != len(constructor.fields):
                raise KernelError(
                    f"branch {constructor.name!r} binds the wrong number of fields"
                )
            branch_context = branch_context.with_givens(refinement.givens)
            for local, field in zip(branch.fields, constructor.fields, strict=True):
                expected = substitute_type(field.type, substitution)
                if local.type != expected:
                    raise KernelError(
                        f"branch field {local.name!r} has type {render_static(local.type)}, "
                        f"expected {render_static(expected)}"
                    )
                branch_context = branch_context.extend(local)
            body = _infer_computation(branch.body, registry, branch_context)
            for entry in body.effects.entries:
                _check_static_variable_scope(
                    entry.effect,
                    context,
                    subject="case branch effect row",
                )
            constructor_indices = tuple(
                substitute_static(index, substitution)
                for index in constructor.result_indices
            )
            motive_at_constructor = instantiate_telescope(
                computation.motive.indices,
                constructor_indices,
            )
            expected_result = substitute_type(
                computation.motive.result_type,
                motive_at_constructor,
            )
            if body.result != expected_result:
                raise KernelError(
                    f"branch {constructor.name!r} returns {render_static(body.result)}, "
                    f"expected {expected_result!r}"
                )
            branch_rows.append((refinement.reachability, body.effects))
        return _checked_computation_type(
            join_branch_rows(tuple(branch_rows)),
            result_type,
            registry,
            context,
        )
    if isinstance(computation, Handle):
        handler = registry.handler(computation.handler)
        for argument in computation.static_arguments:
            registry.validate_static(argument)
            _check_static_variable_scope(
                argument,
                context,
                subject="handler static argument",
            )
        substitution = instantiate_telescope(
            handler.telescope,
            computation.static_arguments,
        )
        handled_effect = substitute_effect(handler.effect, substitution)
        input_type = substitute_type(handler.input_type, substitution)
        output_type = substitute_type(handler.output_type, substitution)
        introduced = substitute_row(handler.introduced, substitution)
        inner = _infer_computation(computation.computation, registry, context)
        if inner.result != input_type:
            raise KernelError(
                f"handler {handler.name!r} expects {render_static(input_type)}, got {render_static(inner.result)}"
            )
        interface = inner.effects.lookup(computation.instance)
        if interface != handled_effect:
            raise KernelError(
                f"handler {handler.name!r} does not match the lexical effect instance"
            )
        residual = (
            inner.effects.without(computation.instance)
            if handler.total
            else inner.effects
        )
        return _checked_computation_type(
            residual.union(introduced),
            output_type,
            registry,
            context,
        )
    if isinstance(computation, Call):
        return _infer_call(computation, registry, context)
    if isinstance(computation, NewInstance):
        return _infer_new_instance(computation, registry, context)
    if isinstance(computation, Resume):
        return _infer_resume(computation, registry, context)
    raise KernelError(f"unknown computation term {computation!r}")


def _infer_resume(
    resumption: Resume,
    registry: KernelRegistry,
    context: CheckContext,
) -> ComputationType:
    """Check an invocation of the enclosing clause's continuation.

    Parameters
    ----------
    resumption : Resume
        The invocation to check.
    registry : KernelRegistry
        Registry the carried value is checked against.
    context : CheckContext
        Scope the invocation occurs in, which carries the continuation
        when there is one.

    Returns
    -------
    ComputationType
        The clause's answer type and the row resuming performs.

    Raises
    ------
    KernelError
        If there is no continuation in scope, which is `resume` written
        outside a handler clause body, or if the value carried does not
        have the type the resumed operation supplies.
    """
    available = context.resumption
    if available is None:
        raise KernelError(
            "resume outside a handler clause body: there is no continuation "
            "to invoke here",
            "qiec-resumption",
        )
    actual = infer_value(resumption.value, registry, context)
    if actual != available.result:
        raise KernelError(
            f"resume carries {actual!r} but the operation it resumes supplies "
            f"{available.result!r}",
            "qiec-resumption",
        )
    return _checked_computation_type(
        available.effects,
        available.answer,
        registry,
        context,
    )


def _unmapped_hint(expected: TypeExpr, actual: TypeExpr) -> str:
    """Advice for a tensor passed where its entries are expected.

    A computation's parameter keeps the type it declares: a call does
    not map a computation over the leading axes of a tensor argument.

    Parameters
    ----------
    expected : TypeExpr
        The parameter's type.
    actual : TypeExpr
        The argument's type.

    Returns
    -------
    str
        A clause saying to declare the parameter as the tensor and map
        over it inside the computation, when ``actual`` is a tensor
        whose entries (or trailing slices) have type ``expected``; else
        empty.
    """
    split = tensor_shape(actual)
    if split is None:
        return ""
    element, dimensions = split
    inner = tensor_shape(expected)
    if inner is None:
        matches = element == expected
    else:
        matches = inner[0] == element and (
            len(inner[1]) < len(dimensions)
            and dimensions[len(dimensions) - len(inner[1]) :] == inner[1]
        )
    if not matches:
        return ""
    return (
        f"; a call does not map a computation over a tensor's entries: "
        f"declare the parameter as {render_static(actual)} and `map` over it "
        f"inside the computation"
    )


def _infer_call(
    call: Call,
    registry: KernelRegistry,
    context: CheckContext,
) -> ComputationType:
    """Check one call against the callee's registered signature.

    The callee is resolved by identity, not by name, so a rename recorded
    in a migration keeps calls resolving while two like-named
    declarations in different modules stay apart.

    Parameters
    ----------
    call : Call
        The call to check.
    registry : KernelRegistry
        Registry holding the callee's signature.
    context : CheckContext
        Scope the call is made in.

    Returns
    -------
    ComputationType
        The callee's result type and row, instantiated at this call's
        static arguments. The caller unions that row into its own, which
        is what propagates a callee's effects outward.

    Raises
    ------
    KernelError
        If the callee is unregistered, if the static or value arity is
        wrong, if a static argument is ill-kinded, if an argument's type
        does not match the instantiated parameter type, if the call's
        recorded result type or row disagrees with the instantiated
        signature, or if any part mentions a rigid variable out of scope.
    """
    signature = registry.computation(call.callee)
    for argument in call.static_arguments:
        registry.validate_static(argument)
        _check_static_variable_scope(argument, context, subject="call static argument")
    if len(call.static_arguments) != len(signature.telescope):
        raise KernelError(
            f"call to {signature.name!r} supplies {len(call.static_arguments)} "
            f"static argument(s); the declaration binds {len(signature.telescope)}",
            "qiec-call-arity",
        )
    substitution = instantiate_telescope(signature.telescope, call.static_arguments)
    if len(call.arguments) != len(signature.parameters):
        raise KernelError(
            f"call to {signature.name!r} supplies {len(call.arguments)} "
            f"argument(s); the declaration takes {len(signature.parameters)}",
            "qiec-call-arity",
        )
    for position, (argument, declared) in enumerate(
        zip(call.arguments, signature.parameters, strict=True)
    ):
        actual = infer_value(argument, registry, context)
        expected = substitute_type(declared, substitution)
        if actual != expected:
            raise KernelError(
                f"call to {signature.name!r} argument {position}: expected "
                f"{render_static(expected)}, got {render_static(actual)}"
                f"{_unmapped_hint(expected, actual)}",
                "qiec-call",
            )
    result = substitute_type(signature.result, substitution)
    effects = substitute_row(signature.effects, substitution)
    # The call records what it expects, and the signature says what it
    # gets. Comparing them here means a stale call site is a checker
    # error rather than a silently wrong type flowing onward.
    if call.result_type != result:
        raise KernelError(
            f"call to {signature.name!r} records result {call.result_type!r} "
            f"but the instantiated signature gives {result!r}",
            "qiec-call",
        )
    if call.effects != effects:
        raise KernelError(
            f"call to {signature.name!r} records row {render_row(call.effects, registry.instance_names)} "
            f"but the instantiated signature gives {render_row(effects, registry.instance_names)}",
            "qiec-call",
        )
    return _checked_computation_type(effects, result, registry, context)


def _infer_new_instance(
    allocation: NewInstance,
    registry: KernelRegistry,
    context: CheckContext,
) -> ComputationType:
    """Check a scoped allocation and confirm the instance does not escape.

    Parameters
    ----------
    allocation : NewInstance
        The allocation to check.
    registry : KernelRegistry
        Registry the interface is resolved against.
    context : CheckContext
        Scope the allocation occurs in.

    Returns
    -------
    ComputationType
        The body's type with the allocated instance discharged from its
        row. The instance exists only inside the body, so a row still
        mentioning it outside would name something no handler can reach.

    Raises
    ------
    KernelError
        If the interface application is unknown or ill-formed, or if the
        allocated instance survives in the body's residual row, which is
        the escape this rule exists to reject.
    """
    registry.validate_static(allocation.effect)
    definition = registry.effects.get(allocation.effect.id)
    if definition is None or not definition.matches(allocation.effect):
        raise KernelError(
            "unknown effect interface for local instance "
            f"{render_instance(allocation.instance, registry.instance_names)}"
        )
    inner = _infer_computation(allocation.body, registry, context)
    if inner.effects.contains(allocation.instance):
        raise KernelError(
            "local instance "
            f"{render_instance(allocation.instance, registry.instance_names)} "
            f"escapes its scope: the body still performs it, so nothing "
            f"outside can discharge it",
            "qiec-instance-escape",
        )
    return _checked_computation_type(
        inner.effects,
        inner.result,
        registry,
        context,
    )


class ResumptionUse(dx.Model):
    """How many times a clause body may invoke its continuation.

    An interval rather than a count, because a `case` splits the paths
    through a body and different branches may resume different numbers of
    times. `minimum` is what every path does at least, `maximum` what
    some path does at most.

    Parameters
    ----------
    minimum : int
        The fewest invocations on any path.
    maximum : int or None
        The most on any path, or None for unbounded. None is the top of
        the lattice and is used wherever the analysis cannot bound the
        count, so an unbounded use is never mistaken for a small one.
    """

    minimum: int
    maximum: int | None

    def then(self, other: ResumptionUse) -> ResumptionUse:
        """Sequential composition: both run, so the counts add.

        Parameters
        ----------
        other : ResumptionUse
            What runs after this.

        Returns
        -------
        ResumptionUse
            The combined use. Unbounded on either side stays unbounded.
        """
        maximum = (
            None
            if self.maximum is None or other.maximum is None
            else self.maximum + other.maximum
        )
        return ResumptionUse(minimum=self.minimum + other.minimum, maximum=maximum)

    def join(self, other: ResumptionUse) -> ResumptionUse:
        """Alternation: one path or the other, so the interval widens.

        Parameters
        ----------
        other : ResumptionUse
            The alternative path's use.

        Returns
        -------
        ResumptionUse
            The interval covering both. The minimum drops to the smaller
            because some path now does that few, and the maximum rises to
            the larger for the same reason.
        """
        maximum = (
            None
            if self.maximum is None or other.maximum is None
            else max(self.maximum, other.maximum)
        )
        return ResumptionUse(minimum=min(self.minimum, other.minimum), maximum=maximum)


NEVER = ResumptionUse(minimum=0, maximum=0)
"""A computation that cannot invoke the continuation."""

ONCE = ResumptionUse(minimum=1, maximum=1)
"""A computation that invokes the continuation exactly once."""


def resumption_use(computation: Computation) -> ResumptionUse:
    """Count the continuation invocations along the paths of a body.

    A `resume` is lexical to the clause that encloses it, so a call
    contributes nothing: the callee has no access to this clause's
    continuation and cannot invoke it however it is written. That is what
    keeps the analysis exact rather than conservative across recursion.

    Parameters
    ----------
    computation : Computation
        The clause body to analyse.

    Returns
    -------
    ResumptionUse
        The interval of invocation counts over the body's paths.

    Raises
    ------
    KernelError
        If the term is of an unknown computation class.
    """
    if isinstance(computation, Resume):
        return ONCE
    if isinstance(computation, Return | Perform | Call):
        return NEVER
    if isinstance(computation, Bind):
        return resumption_use(computation.first).then(resumption_use(computation.then))
    if isinstance(computation, NewInstance):
        return resumption_use(computation.body)
    if isinstance(computation, Handle):
        # A `resume` inside a handled computation still names this
        # clause's continuation: handling an instance does not introduce
        # a new one, and the nested handler's own clauses are separate
        # declarations analysed on their own.
        return resumption_use(computation.computation)
    if isinstance(computation, If):
        return resumption_use(computation.then).join(
            resumption_use(computation.otherwise)
        )
    if isinstance(computation, Case):
        if not computation.branches:
            return NEVER
        uses = [resumption_use(branch.body) for branch in computation.branches]
        result = uses[0]
        for use in uses[1:]:
            result = result.join(use)
        return result
    raise KernelError(f"unknown computation term {computation!r}")


def check_resumption_grade(
    grade: ResumptionGrade,
    use: ResumptionUse,
    *,
    subject: str,
) -> None:
    """Hold a clause body to the grade its declaration promises.

    Parameters
    ----------
    grade : ResumptionGrade
        The declared grade.
    use : ResumptionUse
        What the body actually does, from `resumption_use`.
    subject : str
        What is being checked, named in any diagnostic.

    Raises
    ------
    KernelError
        If the body can resume more often than the grade allows, or, for
        a linear grade, if some path fails to resume at all. A grade is a
        promise other code relies on: a handler declared `0` may be
        compiled without keeping the continuation alive, so resuming
        anyway is not a stylistic matter.
    """
    maximum = use.maximum
    if grade is ResumptionGrade.ZERO:
        if maximum != 0:
            raise KernelError(
                f"{subject} declares grade 0 but can resume "
                f"{'any number of' if maximum is None else maximum} time(s)",
                "qiec-resumption",
            )
        return
    if grade is ResumptionGrade.AFFINE:
        if maximum is None or maximum > 1:
            raise KernelError(
                f"{subject} declares an affine grade but can resume "
                f"{'any number of' if maximum is None else maximum} times",
                "qiec-resumption",
            )
        return
    if grade is ResumptionGrade.LINEAR:
        if maximum is None or maximum > 1:
            raise KernelError(
                f"{subject} declares a linear grade but can resume "
                f"{'any number of' if maximum is None else maximum} times",
                "qiec-resumption",
            )
        if use.minimum < 1:
            raise KernelError(
                f"{subject} declares a linear grade but some path through "
                f"its body does not resume at all",
                "qiec-resumption",
            )
        return


__all__ = [
    "CheckContext",
    "ComputationSignature",
    "KernelError",
    "KernelRegistry",
    "ResumptionType",
    "StaticScope",
    "check_evidence",
    "check_request",
    "check_static",
    "check_type",
    "infer_computation",
    "infer_value",
]
