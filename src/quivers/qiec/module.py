"""Serializable, whole-module containers for the QIEC kernel.

The small records in this module are the boundary returned by source-language
extension lowerers.  They contain declarations and checked computation graphs,
but never parser nodes, compiler objects, callbacks, or backend-specific data.
"""

from __future__ import annotations

from quivers._values import HeldValue, OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
import didactic.api as dx

from collections.abc import Mapping

from quivers.qiec.checking import (
    ComputationSignature,
    CheckContext,
    KernelError,
    KernelRegistry,
    infer_computation,
)
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl
from quivers.qiec.effects import (
    ComputationType,
    EffectDef,
    EffectRequest,
    EffectRow,
    HandlerDef,
    RowEntry,
    render_row,
)
from quivers.qiec.programs import ProgramEntry
from quivers.qiec.identifiers import (
    ComputationId,
    EffectInstanceId,
    QIEC_ABI,
    SiteProvenance,
    SourceOrigin,
)
from quivers.qiec.kinds import (
    EffectBinder,
    IndexBinder,
    Telescope,
    TypeBinder,
    UserIndexSort,
    validate_telescope,
)
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    CaseMotive,
    Computation,
    Handle,
    If,
    Local,
    NewInstance,
)
from quivers.qiec.types import (
    EffectRef,
    EffectVariable,
    IndexVariable,
    TypeVariable,
    render_static,
)


type _BinderMaps = tuple[
    dict[str, TypeBinder], dict[str, IndexBinder], dict[str, EffectBinder]
]
"""The disjoint type, index, and effect namespaces of a telescope."""


class NamedEffectInstance(QiecModel):
    """A source name for one deterministically allocated lexical instance.

    Parameters
    ----------
    name
        The name requests resolve through; must be non-empty.
    entry
        The allocated instance and its interface application.
    origin
        Where the instance was declared.
    """

    name: str
    entry: RowEntry = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    origin: SourceOrigin = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Reject an unnamed instance.

        Raises
        ------
        ValueError
            If the source name is empty. The name is how a request
            resolves to this instance, so an empty one names nothing.
        """
        if not self.name:
            raise ValueError("an effect instance name cannot be empty")


class NamedComputation(QiecModel):
    """One named QVR computation and its checked QIEC type.

    Parameters
    ----------
    id
        The computation's stable identity, which calls refer to.
    name
        The display name; must be non-empty and unique within the module.
    telescope
        The static parameters the computation abstracts over, in order.
    parameters
        The value parameters, in order; their names must be distinct and
        must not reuse a telescope binder's name.
    body
        The computation's body, with the parameters in scope.
    type
        The checked effect row and result type of ``body``.
    origin
        Where the computation was declared.
    """

    id: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    telescope: Telescope = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    parameters: tuple[Local, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    body: Computation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    type: ComputationType = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    origin: SourceOrigin = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Reject a computation whose names collide or are absent.

        Raises
        ------
        ValueError
            If the name is empty, the telescope is malformed, two
            parameters share a name, or a parameter shadows a static
            binder. Shadowing is rejected so a name inside the body
            denotes one thing.
        """
        if not self.name:
            raise ValueError("a computation name cannot be empty")
        validate_telescope(self.telescope)
        static_names = {binder.name for binder in self.telescope}
        local_names = [parameter.name for parameter in self.parameters]
        if len(set(local_names)) != len(local_names):
            raise ValueError(f"duplicate parameter in computation {self.name!r}")
        overlap = static_names & set(local_names)
        if overlap:
            raise ValueError(
                f"computation {self.name!r} reuses static names as parameters: "
                f"{sorted(overlap)!r}"
            )


class QiecModule(QiecModel):
    """A complete, serializable QIEC compilation unit.

    ``source_protocol`` records the exact source route that produced the
    module.  The wire envelope separately records :data:`QIEC_ABI`, so neither
    endpoint can be guessed during deserialization or compiler integration.

    Parameters
    ----------
    module
        The module's name, from which every stable identity in it derives.
    source_protocol
        The source route that produced the module.
    index_sorts
        The user-defined closed index sorts.
    families
        The indexed data families.
    constructors
        The family constructors.
    effects
        The effect interfaces.
    instances
        The module-scoped named effect instances.
    handlers
        The handler declarations.
    computations
        The named computations.
    entries
        The program entry points, each naming one of the computations.
    gap
        The diagnostic of a program the module lowered without, because
        it uses a construct whose elaboration is not yet defined; empty
        when every program elaborated.
    abi
        The kernel ABI the module is expressed against; must equal
        :data:`QIEC_ABI`.
    """

    module: str
    source_protocol: str
    index_sorts: tuple[UserIndexSort, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    families: tuple[FamilyDecl, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    constructors: tuple[ConstructorDecl, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    effects: tuple[EffectDef, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    instances: tuple[NamedEffectInstance, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    handlers: tuple[HandlerDef, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    computations: tuple[NamedComputation, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    entries: tuple[ProgramEntry, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    gap: str = ""
    abi: str = QIEC_ABI

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Reject a module whose identity or ABI is missing or wrong.

        Raises
        ------
        ValueError
            If the module name or source protocol is empty, or the ABI is
            not the one this kernel speaks. The ABI is checked here as
            well as at the wire envelope so a module built in process
            cannot claim a version it was not built for.
        """
        if not isinstance(self.module, str) or not self.module:
            raise ValueError("a QIEC module name cannot be empty")
        if not isinstance(self.source_protocol, str) or not self.source_protocol:
            raise ValueError("a QIEC source protocol cannot be empty")
        if self.abi != QIEC_ABI:
            raise ValueError(f"QIEC module ABI must be {QIEC_ABI!r}")
        _require_unique(
            (sort.name for sort in self.index_sorts),
            subject="index sort name",
        )
        _require_unique((family.id for family in self.families), subject="family ID")
        _require_unique(
            (family.name for family in self.families), subject="family name"
        )
        _require_unique(
            (constructor.id for constructor in self.constructors),
            subject="constructor ID",
        )
        _require_unique((effect.ref.id for effect in self.effects), subject="effect ID")
        _require_unique(
            (effect.ref.name for effect in self.effects), subject="effect name"
        )
        _require_unique(
            (instance.entry.instance for instance in self.instances),
            subject="effect-instance ID",
        )
        _require_unique(
            (instance.name for instance in self.instances),
            subject="effect-instance name",
        )
        _require_unique((handler.id for handler in self.handlers), subject="handler ID")
        _require_unique(
            (handler.name for handler in self.handlers), subject="handler name"
        )
        _require_unique(
            (computation.name for computation in self.computations),
            subject="computation name",
        )

        declared_constructors = {
            constructor.id: constructor for constructor in self.constructors
        }
        family_ids = {family.id for family in self.families}
        expected_constructor_ids = {
            constructor
            for family in self.families
            for constructor in family.constructors
        }
        if set(declared_constructors) != expected_constructor_ids:
            raise ValueError(
                "module constructors must exactly match the constructors named by families"
            )
        if any(
            constructor.family not in family_ids
            for constructor in declared_constructors.values()
        ):
            raise ValueError("a module constructor names an unknown family")

        effect_ids = {effect.ref.id for effect in self.effects}
        if any(
            instance.entry.effect.id not in effect_ids for instance in self.instances
        ):
            raise ValueError("an effect instance names an unknown interface")
        if any(handler.effect.id not in effect_ids for handler in self.handlers):
            raise ValueError("a handler names an unknown effect interface")
        for item in (*self.instances, *self.computations):
            if item.origin.module != self.module:
                raise ValueError("source origin belongs to another QIEC module")
            if item.origin.source_protocol != self.source_protocol:
                raise ValueError("source origin uses another source protocol")


def _children(node: object) -> tuple[object, ...]:
    """The parts of a node a structural walk descends into.

    Parameters
    ----------
    node : object
        A kernel node, a tuple of nodes, or a leaf.

    Returns
    -------
    tuple[object, ...]
        A tuple's items, a record's field values, or nothing for a leaf.
    """
    if isinstance(node, tuple):
        return node
    if isinstance(node, QiecModel):
        return tuple(getattr(node, name) for name in type(node).__field_specs__)
    return ()


def called_computations(body: Computation) -> frozenset[ComputationId]:
    """The computations a body calls, directly.

    Parameters
    ----------
    body : Computation
        The body to walk.

    Returns
    -------
    frozenset[ComputationId]
        The callee of every ``Call`` in the body, at any depth.
    """
    found: set[ComputationId] = set()
    pending: list[Computation] = [body]
    while pending:
        term = pending.pop()
        if isinstance(term, Call):
            found.add(term.callee)
        elif isinstance(term, Bind):
            pending.append(term.then)
            pending.append(term.first)
        elif isinstance(term, Handle):
            pending.append(term.computation)
        elif isinstance(term, Case):
            pending.extend(branch.body for branch in term.branches)
        elif isinstance(term, If):
            pending.append(term.otherwise)
            pending.append(term.then)
        elif isinstance(term, NewInstance):
            pending.append(term.body)
    return frozenset(found)


def program_computations(module: QiecModule) -> frozenset[ComputationId]:
    """The computations that hold programs and their own helpers.

    Parameters
    ----------
    module : QiecModule
        The module.

    Returns
    -------
    frozenset[ComputationId]
        The identity of every entry point's computation and of every
        computation the elaboration made for a program, such as a
        marginalization helper or a scan's recurrence: what a target
        renders from the program's plan rather than as a computation
        of its own.
    """
    entries = {entry.computation for entry in module.entries}
    return frozenset(
        computation.id
        for computation in module.computations
        if computation.id in entries
        or computation.origin.structural_path[:1] == ("programs",)
    )


def reachable_computations(module: QiecModule) -> frozenset[ComputationId]:
    """The computations a program reaches.

    Parameters
    ----------
    module : QiecModule
        The module.

    Returns
    -------
    frozenset[ComputationId]
        The identity of every entry point's computation and of every
        computation it calls, transitively.
    """
    by_id = {computation.id: computation for computation in module.computations}
    found: set[ComputationId] = set()
    pending = [entry.computation for entry in module.entries]
    while pending:
        identity = pending.pop()
        if identity in found or identity not in by_id:
            continue
        found.add(identity)
        pending.extend(called_computations(by_id[identity].body))
    return frozenset(found)


def _require_unique(values: object, *, subject: str) -> None:
    """Reject a repeated identity in a module-level collection.

    Parameters
    ----------
    values : object
        An iterable of hashable identities.
    subject : str
        What is being checked, named in the diagnostic.

    Raises
    ------
    ValueError
        If any value repeats. Two declarations sharing an identity would
        make a reference to it ambiguous.
    """
    materialized = tuple(values)  # type: ignore[arg-type]
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"duplicate {subject} in QIEC module")


def validate_module(module: QiecModule) -> KernelRegistry:
    """Recheck a whole lowered module and return its resolved registry.

    This is deliberately independent of the lowerer. It rebuilds the
    registry from the module's own declarations and rechecks every body
    against it, so a module that arrived from elsewhere, or from a
    lowerer with a bug, is held to the same standard as one just built.

    Parameters
    ----------
    module : QiecModule
        The lowered module to recheck.

    Returns
    -------
    KernelRegistry
        The registry resolved from the module's declarations, for a
        caller that wants to check further terms against it.

    Raises
    ------
    KernelError
        If the ABI does not match, a declaration is ill-formed, an
        instance or index sort is undeclared, a body fails to check, or
        an inferred type does not conform to the declared one.
    """

    if module.abi != QIEC_ABI:
        raise KernelError(f"unsupported QIEC ABI {module.abi!r}")
    _validate_index_sorts(module)
    registry = KernelRegistry()
    for family in module.families:
        registry.register_family(family)
    for constructor in module.constructors:
        registry.register_constructor(constructor)
    for effect in module.effects:
        registry.register_effect(effect)
    for instance in module.instances:
        _validate_closed_effect_instance(instance)
        registry.validate_static(instance.entry.effect)
        declared = registry.effects.get(instance.entry.effect.id)
        if declared is None or not declared.matches(instance.entry.effect):
            raise KernelError(
                f"unknown effect interface for instance {instance.name!r}"
            )
        registry.instance_names[instance.entry.instance] = instance.name
    # Every computation signature is registered before any body is
    # rechecked, for the same reason the lowerer collects them first: a
    # body may call a computation declared after it, and two may call
    # each other.
    for computation in module.computations:
        registry.register_computation(
            ComputationSignature(
                id=computation.id,
                name=computation.name,
                telescope=computation.telescope,
                parameters=tuple(
                    parameter.type for parameter in computation.parameters
                ),
                result=computation.type.result,
                effects=computation.type.effects,
            )
        )
    for handler in module.handlers:
        registry.register_handler(handler)
    instance_interfaces = {
        instance.entry.instance: instance.entry.effect for instance in module.instances
    }
    for handler in module.handlers:
        _validate_effect_instances(
            handler.introduced,
            instance_interfaces,
            subject=f"handler {handler.name!r}",
            module=module.module,
            source_protocol=module.source_protocol,
        )
    for computation in module.computations:
        _validate_named_static_scope(computation, computation.telescope)
        registry.validate_computation_type(computation.type)
        _validate_effect_instances(
            computation.type.effects,
            instance_interfaces,
            subject=f"declared row of computation {computation.name!r}",
            module=module.module,
            source_protocol=module.source_protocol,
        )
        _validate_effect_instances(
            computation.body,
            instance_interfaces,
            subject=f"body of computation {computation.name!r}",
            module=module.module,
            source_protocol=module.source_protocol,
        )
        context = CheckContext()
        for parameter in computation.parameters:
            registry.validate_type(parameter.type)
            context = context.extend(parameter)
        actual = infer_computation(computation.body, registry, context)
        if not computation_type_conforms(actual, computation.type):
            raise KernelError(
                f"computation {computation.name!r} has type "
                f"{render_computation_type(actual, registry.instance_names)}, "
                f"not declared type "
                f"{render_computation_type(computation.type, registry.instance_names)}"
            )
    return registry


def render_computation_type(
    type_: ComputationType, names: Mapping[EffectInstanceId, str] | None = None
) -> str:
    """Render a computation type as its result and row.

    Parameters
    ----------
    type_ : ComputationType
        The type.
    names : Mapping[EffectInstanceId, str] | None
        The source name of each lexical instance, for the row.

    Returns
    -------
    str
        ``Result !{row}``.
    """
    return f"{render_static(type_.result)} {render_row(type_.effects, names)}"


def inferred_computation_type(
    module: QiecModule, registry: KernelRegistry, name: str
) -> ComputationType:
    """The type a named computation's body has, as the checker infers it.

    Parameters
    ----------
    module : QiecModule
        The module, already validated.
    registry : KernelRegistry
        The registry `validate_module` resolved for it.
    name : str
        The computation's name.

    Returns
    -------
    ComputationType
        The inferred result type and the row the body performs, which the
        declared row may widen.

    Raises
    ------
    KeyError
        If the module has no computation of that name.
    """
    computation = next(
        (item for item in module.computations if item.name == name), None
    )
    if computation is None:
        raise KeyError(f"module {module.module!r} has no computation {name!r}")
    context = CheckContext()
    for parameter in computation.parameters:
        context = context.extend(parameter)
    return infer_computation(computation.body, registry, context)


def computation_type_conforms(
    actual: ComputationType,
    declared: ComputationType,
) -> bool:
    """Check exact rows or sound weakening into an explicitly open row.

    An inferred open tail can inhabit a declared open row only when it proves
    every absence guaranteed by the declaration. Row-variable names and IDs
    are lexical identities, so conformance compares their constraints rather
    than treating unrelated tails as identical.

    Parameters
    ----------
    actual : ComputationType
        The type inferred from the body.
    declared : ComputationType
        The type the declaration states.

    Returns
    -------
    bool
        True when the body's type inhabits the declaration. The result
        types must be equal; the rows may differ only by the inferred row
        being weakened into a declared open one that it proves every
        absence of.
    """

    if actual.result != declared.result:
        return False
    if declared.effects.tail is None:
        return actual.effects == declared.effects
    actual_entries = {entry.instance: entry.effect for entry in actual.effects.entries}
    declared_entries = {
        entry.instance: entry.effect for entry in declared.effects.entries
    }
    for instance, effect in actual_entries.items():
        expected = declared_entries.get(instance)
        if expected is not None:
            if expected != effect:
                return False
            continue
        if declared.effects.tail.proves_lacks(instance):
            return False
    if actual.effects.tail is None:
        return True
    forbidden = set(declared.effects.tail.lacks) - set(declared_entries)
    if any(instance in actual_entries for instance in forbidden):
        return False
    if any(not actual.effects.tail.proves_lacks(instance) for instance in forbidden):
        return False
    return True


def _validate_effect_instances(
    value: object,
    interfaces: dict[EffectInstanceId, EffectRef],
    *,
    subject: str,
    module: str,
    source_protocol: str,
) -> None:
    """Validate every concrete lexical instance in a module-owned object.

    Parameters
    ----------
    value : object
        The module-owned object to walk.
    interfaces : dict[EffectInstanceId, EffectRef]
        Module-level instances and the interfaces they implement.
    subject : str
        What is being validated, named in any diagnostic.
    module : str
        The module's name, which request provenance must match.
    source_protocol : str
        The source protocol, which request provenance must match.

    Raises
    ------
    KernelError
        If an instance is undeclared, if an entry assigns it the wrong
        interface, or if a request carries provenance from another module
        or another source protocol.

    Notes
    -----
    Instances come from two places. A module-level `instance` declaration
    is visible throughout, and a `NewInstance` allocation is visible only
    inside its own body. The walk therefore carries a scope: entering an
    allocation adds its instance, and leaving it drops the instance
    again, so a reference outside is the undeclared-instance error it
    should be.
    """

    def require(
        instance: EffectInstanceId,
        scope: dict[EffectInstanceId, EffectRef],
        effect: EffectRef | None = None,
    ) -> None:
        """Require one instance to be in scope, and to match its interface.

        Parameters
        ----------
        instance : EffectInstanceId
            The instance referred to.
        scope : dict[EffectInstanceId, EffectRef]
            Locally allocated instances currently visible.
        effect : EffectRef or None
            The interface the reference claims, when it carries one.

        Raises
        ------
        KernelError
            If the instance is neither locally allocated nor declared at
            module level, or the claimed interface disagrees.
        """
        declared = scope.get(instance, interfaces.get(instance))
        if declared is None:
            raise KernelError(f"{subject} uses an undeclared lexical effect instance")
        if effect is not None and declared != effect:
            raise KernelError(
                f"{subject} assigns the wrong interface to a lexical effect instance"
            )

    pending: list[tuple[object, dict[EffectInstanceId, EffectRef]]] = [(value, {})]
    while pending:
        node, scope = pending.pop()
        if isinstance(node, NewInstance):
            pending.append((node.body, {**scope, node.instance: node.effect}))
            pending.append((node.effect, scope))
            continue
        if isinstance(node, EffectRow):
            for entry in node.entries:
                require(entry.instance, scope, entry.effect)
            if node.tail is not None:
                for instance in node.tail.lacks:
                    require(instance, scope)
            continue
        if isinstance(node, EffectRequest):
            require(node.instance, scope, node.effect)
            if not isinstance(node.origin, SiteProvenance):
                raise KernelError(f"{subject} has malformed request provenance")
            request_origin = node.origin.origin
            if not isinstance(request_origin, SourceOrigin):
                raise KernelError(f"{subject} has malformed request provenance")
            if request_origin.module != module:
                raise KernelError(
                    f"{subject} has request provenance from another QIEC module"
                )
            if request_origin.source_protocol != source_protocol:
                raise KernelError(
                    f"{subject} has request provenance from another source protocol"
                )
        elif isinstance(node, Handle):
            require(node.instance, scope)
        for child in reversed(_children(node)):
            pending.append((child, scope))


def _validate_closed_effect_instance(instance: NamedEffectInstance) -> None:
    """Require module-level effect applications to be fully closed.

    A module-level instance is visible everywhere, so its interface
    application cannot mention a variable: there is no binder in scope at
    module level for one to refer to.

    Parameters
    ----------
    instance : NamedEffectInstance
        The declared instance to check.

    Raises
    ------
    KernelError
        If the application contains any static variable, whether a free
        named one or a rigid branch skolem that escaped.
    """

    pending: list[object] = [instance.entry.effect.arguments]
    while pending:
        node = pending.pop()
        if isinstance(node, (TypeVariable, IndexVariable, EffectVariable)):
            identity = "identity-bearing" if node.identity is not None else "free named"
            raise KernelError(
                f"module-level effect instance {instance.name!r} contains an "
                f"{identity} static variable {node.name!r}"
            )
        pending.extend(reversed(_children(node)))


def _validate_index_sorts(module: QiecModule) -> None:
    """Require every embedded user index sort to name its module declaration.

    A sort's constructors are part of its identity, so an embedded copy
    that disagrees with the declaration would make coverage checking
    answer against a different datatype than the one declared.

    Parameters
    ----------
    module : QiecModule
        The module to walk.

    Raises
    ------
    KernelError
        If an embedded sort is undeclared, or disagrees with the
        declaration of the same name.
    """

    declared = {sort.name: sort for sort in module.index_sorts}

    pending: list[object] = [
        module.families,
        module.constructors,
        module.effects,
        module.instances,
        module.handlers,
        module.computations,
    ]
    while pending:
        node = pending.pop()
        if isinstance(node, UserIndexSort):
            expected = declared.get(node.name)
            if expected is None:
                raise KernelError(f"undeclared user index sort {node.name!r}")
            if node != expected:
                raise KernelError(
                    f"embedded index sort {node.name!r} disagrees with its declaration"
                )
            continue
        pending.extend(reversed(_children(node)))


def _validate_named_static_scope(value: object, telescope: Telescope) -> None:
    """Reject source variables not introduced by a computation telescope.

    Parameters
    ----------
    value : object
        The declaration to walk.
    telescope : Telescope
        The binders the declaration introduces.

    Raises
    ------
    KernelError
        If a free named variable is not bound by the telescope, or is
        bound at a different kind or sort than it is used at.
    """

    def bindings(
        scope: Telescope,
    ) -> tuple[
        dict[str, TypeBinder],
        dict[str, IndexBinder],
        dict[str, EffectBinder],
    ]:
        """Split a telescope into its three namespaces.

        Parameters
        ----------
        scope : Telescope
            The binders to index.

        Returns
        -------
        tuple[dict[str, TypeBinder], dict[str, IndexBinder], dict[str, EffectBinder]]
            Name-keyed maps of the type, index, and effect binders. The
            namespaces are disjoint, so a name appears in at most one.
        """
        return (
            {binder.name: binder for binder in scope if isinstance(binder, TypeBinder)},
            {
                binder.name: binder
                for binder in scope
                if isinstance(binder, IndexBinder)
            },
            {
                binder.name: binder
                for binder in scope
                if isinstance(binder, EffectBinder)
            },
        )

    pending: list[tuple[HeldValue, _BinderMaps]] = [(value, bindings(telescope))]
    while pending:
        node, (type_binders, index_binders, effect_binders) = pending.pop()
        if isinstance(node, TypeVariable) and node.identity is None:
            binder = type_binders.get(node.name)
            if binder is None or binder.kind != node.kind:
                raise KernelError(f"unbound or mistyped type variable {node.name!r}")
            continue
        if isinstance(node, IndexVariable) and node.identity is None:
            index_binder = index_binders.get(node.name)
            if index_binder is None or index_binder.sort != node.sort:
                raise KernelError(f"unbound or mistyped index variable {node.name!r}")
            continue
        if isinstance(node, EffectVariable) and node.identity is None:
            if node.name not in effect_binders:
                raise KernelError(f"unbound effect variable {node.name!r}")
            continue
        if isinstance(node, CaseMotive):
            nested_types = dict(type_binders)
            nested_indices = dict(index_binders)
            nested_effects = dict(effect_binders)
            for motive_binder in node.indices:
                if isinstance(motive_binder, TypeBinder):
                    nested_types[motive_binder.name] = motive_binder
                elif isinstance(motive_binder, IndexBinder):
                    nested_indices[motive_binder.name] = motive_binder
                else:
                    nested_effects[motive_binder.name] = motive_binder
            pending.append(
                (node.result_type, (nested_types, nested_indices, nested_effects))
            )
            continue
        scope = (type_binders, index_binders, effect_binders)
        for child in reversed(_children(node)):
            pending.append((child, scope))


__all__ = [
    "NamedComputation",
    "NamedEffectInstance",
    "QiecModule",
    "called_computations",
    "computation_type_conforms",
    "inferred_computation_type",
    "program_computations",
    "reachable_computations",
    "render_computation_type",
    "validate_module",
]
