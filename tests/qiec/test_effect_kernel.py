import didactic.api as dx
import pytest

from quivers.qiec import (
    INT,
    NAT,
    STRING,
    ArgumentDef,
    AttachmentId,
    AttachmentRef,
    ComputationType,
    EffectBinder,
    EffectDef,
    EffectId,
    EffectRef,
    EffectRequest,
    EffectRow,
    Handle,
    HandlerClauseDef,
    HandlerDef,
    HandlerId,
    IndexBinder,
    KernelError,
    KernelRegistry,
    LiteralValue,
    OperationDef,
    OperationId,
    Perform,
    ResumptionGrade,
    RowEntry,
    SiteProvenance,
    SourceOrigin,
    TypeBinder,
    TypeApplication,
    TypeConstructorRef,
    TypeId,
    TypeVariable,
    infer_computation,
    infer_value,
    instantiate_effect,
)


def _reader() -> tuple[EffectDef, OperationDef]:
    effect_id = EffectId.derive("tests", "Reader")
    operation = OperationDef(
        id=OperationId.derive(str(effect_id), "ask"),
        name="ask",
        telescope=(TypeBinder(name="a"),),
        arguments=(ArgumentDef(name="key", type=STRING),),
        result_type=TypeVariable(name="a"),
    )
    effect = EffectDef(
        ref=EffectRef(id=effect_id, name="Reader"),
        telescope=(),
        operations=(operation,),
    )
    return effect, operation


def _origin() -> SiteProvenance:
    return SiteProvenance(
        origin=SourceOrigin(
            module="tests",
            structural_path=("body", 0),
            role="reader.ask",
            source_protocol="qvr-source/v0.19",
        )
    )


def test_typed_request_and_total_handler_remove_only_the_matched_instance() -> None:
    effect, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(effect)
    entry = instantiate_effect(effect.ref, module="tests", lexical_path=("reader",))
    request = EffectRequest(
        instance=entry.instance,
        effect=effect.ref,
        operation=operation.id,
        static_arguments=(INT,),
        arguments=(LiteralValue(value="answer", type=STRING),),
        result_type=INT,
        origin=_origin(),
    )
    performed = infer_computation(Perform(request=request), registry)
    assert performed.result == INT
    assert performed.effects == EffectRow(entries=(entry,))

    handler = HandlerDef(
        id=HandlerId.derive("tests", "run_reader"),
        name="run_reader",
        effect=effect.ref,
        clauses=(
            HandlerClauseDef(operation=operation.id, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
    )
    registry.register_handler(handler)
    handled = infer_computation(
        Handle(
            instance=entry.instance,
            handler=handler.id,
            computation=Perform(request=request),
        ),
        registry,
    )
    assert handled.result == INT
    assert handled.effects.entries == ()


def test_request_rejects_an_operation_another_effect_does_not_own() -> None:
    """An operation belongs to the effect that declared it.

    Effect identity is nominal, derived from the module and the effect's
    own name, so a request naming a different effect cannot carry this
    operation however alike the two interfaces look.
    """
    effect, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(effect)
    wrong = EffectRef(id=EffectId.derive("tests", "OtherReader"), name="OtherReader")
    request = EffectRequest(
        instance=instantiate_effect(wrong, module="tests", lexical_path=(0,)).instance,
        effect=wrong,
        operation=operation.id,
        static_arguments=(INT,),
        arguments=(LiteralValue(value="answer", type=STRING),),
        result_type=INT,
        origin=_origin(),
    )

    with pytest.raises(KernelError, match="does not own the operation"):
        infer_computation(Perform(request=request), registry)


def test_a_total_handler_must_cover_every_declared_operation() -> None:
    """Totality is a claim about coverage, and the registry checks it."""
    effect, _operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(effect)
    missing = HandlerDef(
        id=HandlerId.derive("tests", "missing"),
        name="missing",
        effect=effect.ref,
        clauses=(),
        input_type=INT,
        output_type=INT,
    )
    with pytest.raises(KernelError, match="missing operations"):
        registry.register_handler(missing)


def test_a_partial_handler_may_forward_operations_it_does_not_cover() -> None:
    """Forwarding is a property of the handler, not of the interface.

    An effect declares operations and nothing more, so whether unknown
    requests pass through is settled by the handler that admits them.
    A partial handler covering one operation and forwarding the rest
    registers on the same interface a total handler would have to cover
    completely.
    """
    effect, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(effect)
    forwarding = HandlerDef(
        id=HandlerId.derive("tests", "forward"),
        name="forward",
        effect=effect.ref,
        clauses=(
            HandlerClauseDef(operation=operation.id, grade=ResumptionGrade.AFFINE),
        ),
        input_type=INT,
        output_type=INT,
        total=False,
        forwards_unknown=True,
    )
    registry.register_handler(forwarding)
    assert registry.handlers[forwarding.id].forwards_unknown


def test_partial_handler_retains_the_instance_row() -> None:
    effect, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(effect)
    entry = instantiate_effect(effect.ref, module="tests", lexical_path=("partial",))
    request = EffectRequest(
        instance=entry.instance,
        effect=effect.ref,
        operation=operation.id,
        static_arguments=(INT,),
        arguments=(LiteralValue(value="answer", type=STRING),),
        result_type=INT,
        origin=_origin(),
    )
    handler = HandlerDef(
        id=HandlerId.derive("tests", "partial"),
        name="partial",
        effect=effect.ref,
        clauses=(HandlerClauseDef(operation=operation.id, grade=ResumptionGrade.ZERO),),
        input_type=INT,
        output_type=INT,
        total=False,
    )
    registry.register_handler(handler)

    inferred = infer_computation(
        Handle(
            instance=entry.instance,
            handler=handler.id,
            computation=Perform(request=request),
        ),
        registry,
    )
    assert inferred.effects == EffectRow(
        entries=(RowEntry(instance=entry.instance, effect=effect.ref),)
    )


def test_parameterized_effect_applications_are_distinct_and_checked() -> None:
    effect_id = EffectId.derive("tests", "State")
    get = OperationDef(
        id=OperationId.derive(str(effect_id), "get"),
        name="get",
        telescope=(),
        arguments=(),
        result_type=TypeVariable(name="state"),
    )
    state = EffectDef(
        ref=EffectRef(id=effect_id, name="State"),
        telescope=(TypeBinder(name="state"),),
        operations=(get,),
    )
    int_state = state.apply((INT,))
    string_state = state.apply((STRING,))
    int_entry = instantiate_effect(
        int_state,
        module="tests",
        lexical_path=("state",),
    )
    string_entry = instantiate_effect(
        string_state,
        module="tests",
        lexical_path=("state",),
    )
    assert int_state != string_state
    assert int_entry.instance != string_entry.instance
    assert len(EffectRow(entries=(int_entry, string_entry)).entries) == 2

    registry = KernelRegistry()
    registry.register_effect(state)
    mismatched = EffectRequest(
        instance=string_entry.instance,
        effect=string_state,
        operation=get.id,
        static_arguments=(),
        arguments=(),
        result_type=INT,
        origin=_origin(),
    )
    with pytest.raises(KernelError, match="request result"):
        infer_computation(Perform(request=mismatched), registry)

    valid = EffectRequest(
        instance=string_entry.instance,
        effect=string_state,
        operation=get.id,
        static_arguments=(),
        arguments=(),
        result_type=STRING,
        origin=_origin(),
    )
    int_handler = HandlerDef(
        id=HandlerId.derive("tests", "int_state"),
        name="int_state",
        effect=int_state,
        clauses=(HandlerClauseDef(operation=get.id, grade=ResumptionGrade.LINEAR),),
        input_type=STRING,
        output_type=STRING,
    )
    registry.register_handler(int_handler)
    with pytest.raises(KernelError, match="does not match"):
        infer_computation(
            Handle(
                instance=string_entry.instance,
                handler=int_handler.id,
                computation=Perform(request=valid),
            ),
            registry,
        )


def test_effect_registration_checks_interface_and_operation_scope() -> None:
    effect_id = EffectId.derive("tests", "ScopedEffect")
    valid_operation = OperationDef(
        id=OperationId.derive(str(effect_id), "valid"),
        name="valid",
        telescope=(TypeBinder(name="result"),),
        arguments=(ArgumentDef(name="state", type=TypeVariable(name="state")),),
        result_type=TypeVariable(name="result"),
    )
    registry = KernelRegistry()
    registry.register_effect(
        EffectDef(
            ref=EffectRef(id=effect_id, name="ScopedEffect"),
            telescope=(TypeBinder(name="state"),),
            operations=(valid_operation,),
        )
    )

    bad_id = EffectId.derive("tests", "BadScope")
    unbound_operation = OperationDef(
        id=OperationId.derive(str(bad_id), "bad"),
        name="bad",
        telescope=(),
        arguments=(),
        result_type=TypeVariable(name="missing"),
    )
    with pytest.raises(KernelError, match="unbound type variable"):
        KernelRegistry().register_effect(
            EffectDef(
                ref=EffectRef(id=bad_id, name="BadScope"),
                telescope=(TypeBinder(name="state"),),
                operations=(unbound_operation,),
            )
        )

    with pytest.raises(dx.ValidationError, match="shadows interface binders"):
        EffectDef(
            ref=EffectRef(id=EffectId.derive("tests", "Shadow"), name="Shadow"),
            telescope=(TypeBinder(name="a"),),
            operations=(
                OperationDef(
                    id=OperationId.derive("tests", "shadow"),
                    name="shadow",
                    telescope=(TypeBinder(name="a"),),
                    arguments=(),
                    result_type=INT,
                ),
            ),
        )


def test_handler_telescope_is_scoped_and_instantiated_at_handle_site() -> None:
    effect, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(effect)
    entry = instantiate_effect(effect.ref, module="tests", lexical_path=("generic",))
    request = EffectRequest(
        instance=entry.instance,
        effect=effect.ref,
        operation=operation.id,
        static_arguments=(INT,),
        arguments=(LiteralValue(value="answer", type=STRING),),
        result_type=INT,
        origin=_origin(),
    )
    answer = TypeVariable(name="answer")
    generic = HandlerDef(
        id=HandlerId.derive("tests", "generic-reader"),
        name="generic-reader",
        effect=effect.ref,
        clauses=(
            HandlerClauseDef(operation=operation.id, grade=ResumptionGrade.LINEAR),
        ),
        input_type=answer,
        output_type=answer,
        telescope=(TypeBinder(name="answer"),),
    )
    registry.register_handler(generic)
    inferred = infer_computation(
        Handle(
            instance=entry.instance,
            handler=generic.id,
            computation=Perform(request=request),
            static_arguments=(INT,),
        ),
        registry,
    )
    assert inferred.result == INT

    unsaturated = HandlerDef(
        id=HandlerId.derive("tests", "unsaturated-reader"),
        name="unsaturated-reader",
        effect=effect.ref,
        clauses=(
            HandlerClauseDef(operation=operation.id, grade=ResumptionGrade.LINEAR),
        ),
        input_type=TypeVariable(name="free"),
        output_type=TypeVariable(name="free"),
    )
    with pytest.raises(KernelError, match="unbound type variable"):
        registry.register_handler(unsaturated)


def test_effect_application_deeply_validates_nested_static_arguments() -> None:
    malformed_inner = TypeApplication(
        constructor=TypeConstructorRef(
            id=TypeId.derive("tests", "Indexed"),
            name="Indexed",
            telescope=(IndexBinder(name="index", sort=NAT),),
        ),
        arguments=(INT,),
    )
    effect = EffectDef(
        ref=EffectRef(id=EffectId.derive("tests", "Deep"), name="Deep"),
        telescope=(TypeBinder(name="value"),),
        operations=(),
    )

    with pytest.raises(TypeError, match="index argument"):
        effect.apply((malformed_inner,))

    reader, operation = _reader()
    registry = KernelRegistry()
    registry.register_effect(reader)
    entry = instantiate_effect(reader.ref, module="tests", lexical_path=("deep",))
    malformed_request = EffectRequest(
        instance=entry.instance,
        effect=reader.ref,
        operation=operation.id,
        static_arguments=(malformed_inner,),
        arguments=(LiteralValue(value="key", type=STRING),),
        result_type=INT,
        origin=_origin(),
    )
    with pytest.raises(TypeError, match="index argument"):
        infer_computation(Perform(request=malformed_request), registry)


def test_type_constructor_metadata_is_canonical_per_stable_id() -> None:
    type_id = TypeId.derive("tests", "Canonical")

    def effect_with_argument(
        label: str,
        constructor: TypeConstructorRef,
        arguments: tuple[object, ...] = (),
    ) -> EffectDef:
        effect_id = EffectId.derive("tests", "metadata", label)
        operation = OperationDef(
            id=OperationId.derive(effect_id, "use"),
            name="use",
            telescope=(),
            arguments=(
                ArgumentDef(
                    name="value",
                    type=TypeApplication(constructor=constructor, arguments=arguments),  # type: ignore[arg-type]
                ),
            ),
            result_type=INT,
        )
        return EffectDef(
            ref=EffectRef(id=effect_id, name=label),
            telescope=(),
            operations=(operation,),
        )

    registry = KernelRegistry()
    registry.register_effect(
        effect_with_argument("first", TypeConstructorRef(id=type_id, name="Before"))
    )
    registry.register_effect(
        effect_with_argument("renamed", TypeConstructorRef(id=type_id, name="After"))
    )

    conflicting = TypeConstructorRef(
        id=type_id,
        name="After",
        telescope=(TypeBinder(name="argument"),),
    )
    with pytest.raises(KernelError, match="conflicting telescope metadata"):
        registry.register_effect(effect_with_argument("conflict", conflicting, (INT,)))


def test_handle_rejects_an_unsaturated_known_effect_static_argument() -> None:
    reader, operation = _reader()
    unary_id = EffectId.derive("tests", "UnaryEffect")
    unary = EffectDef(
        ref=EffectRef(id=unary_id, name="UnaryEffect"),
        telescope=(TypeBinder(name="value"),),
        operations=(),
    )
    registry = KernelRegistry()
    registry.register_effect(reader)
    registry.register_effect(unary)
    entry = instantiate_effect(reader.ref, module="tests", lexical_path=("effect-arg",))
    request = EffectRequest(
        instance=entry.instance,
        effect=reader.ref,
        operation=operation.id,
        static_arguments=(INT,),
        arguments=(LiteralValue(value="key", type=STRING),),
        result_type=INT,
        origin=_origin(),
    )
    handler = HandlerDef(
        id=HandlerId.derive("tests", "effect-polymorphic"),
        name="effect-polymorphic",
        effect=reader.ref,
        clauses=(
            HandlerClauseDef(operation=operation.id, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
        telescope=(EffectBinder(name="nested"),),
    )
    registry.register_handler(handler)

    with pytest.raises(KernelError, match="invalid application"):
        infer_computation(
            Handle(
                instance=entry.instance,
                handler=handler.id,
                computation=Perform(request=request),
                static_arguments=(unary.ref,),
            ),
            registry,
        )


def test_registry_validates_effect_references_on_every_static_surface() -> None:
    reader, reader_operation = _reader()
    state_id = EffectId.derive("tests", "SurfaceState")
    state = EffectDef(
        ref=EffectRef(id=state_id, name="SurfaceState"),
        telescope=(TypeBinder(name="state"),),
        operations=(),
    )
    registry = KernelRegistry()
    registry.register_effect(reader)
    registry.register_effect(state)
    carrier = TypeConstructorRef(
        id=TypeId.derive("tests", "EffectCarrierSurface"),
        name="EffectCarrierSurface",
        telescope=(EffectBinder(name="effect"),),
    )

    invalid_references = (
        # Unsaturated: the declaration binds one type and none is given.
        state.ref,
        # Saturated but ill-kinded: an effect where a type is required.
        EffectRef(id=state_id, name="SurfaceState", arguments=(reader.ref,)),
    )
    for position, invalid in enumerate(invalid_references):
        invalid_type = TypeApplication(constructor=carrier, arguments=(invalid,))
        with pytest.raises(KernelError, match="invalid application"):
            infer_value(
                AttachmentRef(
                    attachment=AttachmentId.derive("tests", "invalid-effect", position),
                    type=invalid_type,
                ),
                registry,
            )

    # And the check accepts a well-kinded application, so the rejections
    # above are the rule discriminating rather than refusing everything
    # that reaches this surface.
    well_kinded = TypeApplication(
        constructor=carrier,
        arguments=(EffectRef(id=state_id, name="SurfaceState", arguments=(INT,)),),
    )
    infer_value(
        AttachmentRef(
            attachment=AttachmentId.derive("tests", "valid-effect"),
            type=well_kinded,
        ),
        registry,
    )

    unsaturated_type = TypeApplication(constructor=carrier, arguments=(state.ref,))

    def malformed_declaration(label: str, *, in_result: bool) -> EffectDef:
        effect_id = EffectId.derive("tests", "surface-declaration", label)
        operation = OperationDef(
            id=OperationId.derive(effect_id, "operation"),
            name="operation",
            telescope=(),
            arguments=()
            if in_result
            else (ArgumentDef(name="bad", type=unsaturated_type),),
            result_type=unsaturated_type if in_result else INT,
        )
        return EffectDef(
            ref=EffectRef(id=effect_id, name=label),
            telescope=(),
            operations=(operation,),
        )

    for label, in_result in (("argument", False), ("result", True)):
        with pytest.raises(KernelError, match="invalid application"):
            registry.register_effect(malformed_declaration(label, in_result=in_result))

    entry = instantiate_effect(
        reader.ref,
        module="tests",
        lexical_path=("surface-request",),
    )
    bad_result_request = EffectRequest(
        instance=entry.instance,
        effect=reader.ref,
        operation=reader_operation.id,
        static_arguments=(INT,),
        arguments=(LiteralValue(value="key", type=STRING),),
        result_type=unsaturated_type,
        origin=_origin(),
    )
    with pytest.raises(KernelError, match="invalid application"):
        infer_computation(Perform(request=bad_result_request), registry)

    static_operation_id = OperationId.derive("tests", "surface-static-operation")
    static_operation = OperationDef(
        id=static_operation_id,
        name="static",
        telescope=(EffectBinder(name="nested"),),
        arguments=(),
        result_type=INT,
    )
    static_effect = EffectDef(
        ref=EffectRef(
            id=EffectId.derive("tests", "StaticSurface"), name="StaticSurface"
        ),
        telescope=(),
        operations=(static_operation,),
    )
    registry.register_effect(static_effect)
    static_entry = instantiate_effect(
        static_effect.ref,
        module="tests",
        lexical_path=("surface-static-request",),
    )
    bad_static_request = EffectRequest(
        instance=static_entry.instance,
        effect=static_effect.ref,
        operation=static_operation.id,
        static_arguments=(state.ref,),
        arguments=(),
        result_type=INT,
        origin=_origin(),
    )
    with pytest.raises(KernelError, match="invalid application"):
        infer_computation(Perform(request=bad_static_request), registry)

    clause = (
        HandlerClauseDef(operation=reader_operation.id, grade=ResumptionGrade.LINEAR),
    )
    handler_surfaces = (
        HandlerDef(
            id=HandlerId.derive("tests", "bad-handler-input"),
            name="bad-handler-input",
            effect=reader.ref,
            clauses=clause,
            input_type=unsaturated_type,
            output_type=INT,
        ),
        HandlerDef(
            id=HandlerId.derive("tests", "bad-handler-output"),
            name="bad-handler-output",
            effect=reader.ref,
            clauses=clause,
            input_type=INT,
            output_type=unsaturated_type,
        ),
        HandlerDef(
            id=HandlerId.derive("tests", "bad-handler-row"),
            name="bad-handler-row",
            effect=reader.ref,
            clauses=clause,
            input_type=INT,
            output_type=INT,
            introduced=EffectRow(
                entries=(
                    RowEntry(
                        instance=instantiate_effect(
                            state.ref,
                            module="tests",
                            lexical_path=("bad-handler-row",),
                        ).instance,
                        effect=state.ref,
                    ),
                )
            ),
        ),
    )
    for handler in handler_surfaces:
        with pytest.raises(KernelError, match="invalid application"):
            registry.register_handler(handler)

    invalid_row = handler_surfaces[-1].introduced
    with pytest.raises(KernelError, match="invalid application"):
        registry.validate_effect_row(invalid_row)
    with pytest.raises(KernelError, match="invalid application"):
        registry.validate_computation_type(
            ComputationType(effects=EffectRow(), result=unsaturated_type)
        )

    unknown = EffectRef(id=EffectId.derive("tests", "External"), name="External")
    external_type = TypeApplication(constructor=carrier, arguments=(unknown,))
    attachment = AttachmentRef(
        attachment=AttachmentId.derive("tests", "external"), type=external_type
    )
    assert infer_value(attachment, registry) == external_type
