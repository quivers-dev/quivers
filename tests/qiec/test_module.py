from __future__ import annotations

import didactic.api as dx
import pytest

from quivers.qiec import (
    ComputationId,
    EMPTY_ROW,
    INT,
    QIEC_ABI,
    ComputationType,
    EffectDef,
    EffectId,
    EffectInstanceId,
    EffectRef,
    EffectRequest,
    EffectRow,
    Handle,
    HandlerClauseDef,
    HandlerDef,
    HandlerId,
    LiteralValue,
    NamedComputation,
    NamedEffectInstance,
    OperationDef,
    OperationId,
    Perform,
    QiecModule,
    ResumptionGrade,
    RowEntry,
    RowVariable,
    RowVariableId,
    Return,
    SiteProvenance,
    SourceOrigin,
    TypeBinder,
    UserIndexSort,
    TypeVariable,
    StaticVariableId,
    dumps,
    loads,
    validate_module,
)


SOURCE_PROTOCOL = "qvr-source/v0.19"


def _origin() -> SourceOrigin:
    return SourceOrigin(
        module="example",
        structural_path=("computations", "answer"),
        role="computation",
        source_protocol=SOURCE_PROTOCOL,
        file="example.qvr",
        line=1,
        column=1,
    )


def _request_provenance(
    name: str,
    *,
    module: str = "example",
    source_protocol: str = SOURCE_PROTOCOL,
) -> SiteProvenance:
    return SiteProvenance(
        origin=SourceOrigin(
            module=module,
            structural_path=("computations", name, "request"),
            role="perform",
            source_protocol=source_protocol,
        )
    )


def test_module_is_serializable_and_recheckable() -> None:
    computation = NamedComputation(
        id=ComputationId.derive("tests", "answer"),
        name="answer",
        telescope=(),
        parameters=(),
        body=Return(value=LiteralValue(value=42, type=INT)),
        type=ComputationType(effects=EMPTY_ROW, result=INT),
        origin=_origin(),
    )
    module = QiecModule(
        module="example", source_protocol=SOURCE_PROTOCOL, computations=(computation,)
    )

    assert loads(dumps(module)) == module
    assert validate_module(module).families == {}


def test_module_rejects_abi_guessing() -> None:
    with pytest.raises(dx.ValidationError, match="module ABI"):
        QiecModule(
            module="example", source_protocol=SOURCE_PROTOCOL, abi="qiec-core/v99"
        )

    assert QIEC_ABI == "qiec-core/v1alpha1"


def test_module_rejects_origin_from_another_route() -> None:
    computation = NamedComputation(
        id=ComputationId.derive("tests", "answer"),
        name="answer",
        telescope=(),
        parameters=(),
        body=Return(value=LiteralValue(value=42, type=INT)),
        type=ComputationType(effects=EMPTY_ROW, result=INT),
        origin=SourceOrigin(
            module="example",
            structural_path=("computations", "answer"),
            role="computation",
            source_protocol="qvr-source/v0.18",
        ),
    )

    with pytest.raises(dx.ValidationError, match="another source protocol"):
        QiecModule(
            module="example",
            source_protocol=SOURCE_PROTOCOL,
            computations=(computation,),
        )


def test_module_checks_computation_static_scope() -> None:
    a = TypeVariable(name="a")
    computation = NamedComputation(
        id=ComputationId.derive("tests", "identity"),
        name="identity",
        telescope=(TypeBinder(name="a"),),
        parameters=(),
        body=Return(value=LiteralValue(value=42, type=INT)),
        type=ComputationType(effects=EMPTY_ROW, result=a),
        origin=_origin(),
    )

    with pytest.raises(Exception, match="not declared type"):
        validate_module(
            QiecModule(
                module="example",
                source_protocol=SOURCE_PROTOCOL,
                computations=(computation,),
            )
        )


def test_module_rejects_undeclared_static_variable() -> None:
    """A result type mentioning a variable no telescope binds is rejected.

    The rejection now happens while the computation's signature is
    registered, before any body is rechecked, and names the offending
    variable. Registering signatures first is what lets a call resolve
    to a computation declared later, so this check moved earlier with
    them.
    """
    a = TypeVariable(name="a")
    computation = NamedComputation(
        id=ComputationId.derive("tests", "bad"),
        name="bad",
        telescope=(),
        parameters=(),
        body=Return(value=LiteralValue(value=42, type=INT)),
        type=ComputationType(effects=EMPTY_ROW, result=a),
        origin=_origin(),
    )

    with pytest.raises(Exception, match="unbound type variable 'a'"):
        validate_module(
            QiecModule(
                module="example",
                source_protocol=SOURCE_PROTOCOL,
                computations=(computation,),
            )
        )


def _effect_fixture() -> tuple[EffectDef, NamedEffectInstance, OperationDef]:
    ref = EffectRef(id=EffectId.derive("example", "Reader"), name="Reader")
    operation = OperationDef(
        id=OperationId.derive(ref.id, "read"),
        name="read",
        telescope=(),
        arguments=(),
        result_type=INT,
    )
    effect = EffectDef(ref=ref, telescope=(), operations=(operation,))
    entry = RowEntry(instance=EffectInstanceId.derive("example", "reader"), effect=ref)
    instance = NamedEffectInstance(name="reader", entry=entry, origin=_origin())
    return effect, instance, operation


def test_module_rejects_request_for_undeclared_lexical_instance() -> None:
    effect, _instance, operation = _effect_fixture()
    request = EffectRequest(
        instance=EffectInstanceId.derive("example", "forged"),
        effect=effect.ref,
        operation=operation.id,
        static_arguments=(),
        arguments=(),
        result_type=INT,
        origin=_request_provenance("read"),
    )
    computation = NamedComputation(
        id=ComputationId.derive("tests", "read"),
        name="read",
        telescope=(),
        parameters=(),
        body=Perform(request=request),
        type=ComputationType(
            effects=EffectRow(
                entries=(RowEntry(instance=request.instance, effect=effect.ref),)
            ),
            result=INT,
        ),
        origin=_origin(),
    )

    with pytest.raises(Exception, match="undeclared lexical effect instance"):
        validate_module(
            QiecModule(
                module="example",
                source_protocol=SOURCE_PROTOCOL,
                effects=(effect,),
                computations=(computation,),
            )
        )


def test_module_rejects_open_tail_that_does_not_prove_declared_lacks() -> None:
    effect, instance, operation = _effect_fixture()
    actual_tail = RowVariable(
        name="sigma",
        identity=RowVariableId.derive("example", "actual"),
    )
    declared_tail = RowVariable(
        name="rho",
        identity=RowVariableId.derive("example", "declared"),
        lacks=(instance.entry.instance,),
    )
    handler = HandlerDef(
        id=HandlerId.derive("example", "open-handler"),
        name="open-handler",
        effect=effect.ref,
        clauses=(
            HandlerClauseDef(operation=operation.id, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
        introduced=EffectRow(entries=(), tail=actual_tail),
    )
    request = EffectRequest(
        instance=instance.entry.instance,
        effect=effect.ref,
        operation=operation.id,
        static_arguments=(),
        arguments=(),
        result_type=INT,
        origin=_request_provenance("open"),
    )
    computation = NamedComputation(
        id=ComputationId.derive("tests", "open"),
        name="open",
        telescope=(),
        parameters=(),
        body=Handle(
            instance=instance.entry.instance,
            handler=handler.id,
            computation=Perform(request=request),
        ),
        type=ComputationType(
            effects=EffectRow(entries=(), tail=declared_tail), result=INT
        ),
        origin=_origin(),
    )

    with pytest.raises(Exception, match="not declared type"):
        validate_module(
            QiecModule(
                module="example",
                source_protocol=SOURCE_PROTOCOL,
                effects=(effect,),
                instances=(instance,),
                handlers=(handler,),
                computations=(computation,),
            )
        )


def test_module_validates_unused_entries_in_declared_effect_row() -> None:
    effect, instance, _operation = _effect_fixture()
    external = EffectRef(id=EffectId.derive("external", "Other"), name="Other")
    bad_entry = RowEntry(instance=instance.entry.instance, effect=external)
    computation = NamedComputation(
        id=ComputationId.derive("tests", "pure"),
        name="pure",
        telescope=(),
        parameters=(),
        body=Return(value=LiteralValue(value=1, type=INT)),
        type=ComputationType(
            effects=EffectRow(
                entries=(bad_entry,),
                tail=RowVariable(
                    name="rho",
                    identity=RowVariableId.derive("example", "declared-row"),
                    lacks=(bad_entry.instance,),
                ),
            ),
            result=INT,
        ),
        origin=_origin(),
    )

    with pytest.raises(Exception, match="wrong interface"):
        validate_module(
            QiecModule(
                module="example",
                source_protocol=SOURCE_PROTOCOL,
                effects=(effect,),
                instances=(instance,),
                computations=(computation,),
            )
        )


def test_module_rejects_embedded_index_sort_that_disagrees_with_declaration() -> None:
    from quivers.qiec import FamilyDecl, FamilyId, IndexBinder

    declared = UserIndexSort(name="Nat", constructors=("Z", "S"), arities=(0, 1))
    conflicting = UserIndexSort(name="Nat", constructors=("Only",), arities=(0,))
    family = FamilyDecl(
        id=FamilyId.derive("example", "Vector"),
        name="Vector",
        parameters=(),
        indices=(IndexBinder(name="n", sort=conflicting, refinable=True),),
        constructors=(),
    )

    with pytest.raises(Exception, match="disagrees with its declaration"):
        validate_module(
            QiecModule(
                module="example",
                source_protocol=SOURCE_PROTOCOL,
                index_sorts=(declared,),
                families=(family,),
            )
        )


@pytest.mark.parametrize(
    ("request_origin", "message"),
    [
        (_request_provenance("read", module="elsewhere"), "another QIEC module"),
        (
            _request_provenance("read", source_protocol="qvr-source/v0.20"),
            "another source protocol",
        ),
    ],
)
def test_module_rejects_request_provenance_from_another_route(
    request_origin: SiteProvenance,
    message: str,
) -> None:
    effect, instance, operation = _effect_fixture()
    request = EffectRequest(
        instance=instance.entry.instance,
        effect=effect.ref,
        operation=operation.id,
        static_arguments=(),
        arguments=(),
        result_type=INT,
        origin=request_origin,
    )
    computation = NamedComputation(
        id=ComputationId.derive("tests", "read"),
        name="read",
        telescope=(),
        parameters=(),
        body=Perform(request=request),
        type=ComputationType(effects=EffectRow(entries=(instance.entry,)), result=INT),
        origin=_origin(),
    )

    with pytest.raises(Exception, match=message):
        validate_module(
            QiecModule(
                module="example",
                source_protocol=SOURCE_PROTOCOL,
                effects=(effect,),
                instances=(instance,),
                computations=(computation,),
            )
        )


def test_module_rejects_non_string_identity_fields() -> None:
    with pytest.raises(dx.ValidationError, match="module: expected str"):
        QiecModule(module=1, source_protocol="qvr-source/v0.19")  # type: ignore[arg-type]
    with pytest.raises(dx.ValidationError, match="source_protocol: expected str"):
        QiecModule(module="example", source_protocol=2)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "argument",
    [
        TypeVariable(name="state"),
        TypeVariable(
            name="state",
            identity=StaticVariableId.derive("example", "instance", "state"),
        ),
    ],
)
def test_module_rejects_static_variables_in_module_effect_instances(
    argument: TypeVariable,
) -> None:
    ref = EffectRef(id=EffectId.derive("example", "State"), name="State")
    operation = OperationDef(
        id=OperationId.derive(ref.id, "get"),
        name="get",
        telescope=(),
        arguments=(),
        result_type=TypeVariable(name="state"),
    )
    effect = EffectDef(
        ref=ref, telescope=(TypeBinder(name="state"),), operations=(operation,)
    )
    concrete = effect.apply((argument,))
    instance = NamedEffectInstance(
        name="state",
        entry=RowEntry(
            instance=EffectInstanceId.derive("example", "state"), effect=concrete
        ),
        origin=_origin(),
    )

    with pytest.raises(
        Exception, match="module-level effect instance.*static variable"
    ):
        validate_module(
            QiecModule(
                module="example",
                source_protocol=SOURCE_PROTOCOL,
                effects=(effect,),
                instances=(instance,),
            )
        )
