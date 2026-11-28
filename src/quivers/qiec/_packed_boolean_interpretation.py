"""Check the association between a Boolean proof fold and its packed use."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._closed_interpretation_binding import ClosedInterpretationBinding
from quivers.qiec._lexical_proof_types import export_lexical_proofs
from quivers.qiec._models import QiecModel
from quivers.qiec._proof_interpretation import bind_operands
from quivers.qiec.canonical import tensor_shape, tensor_type
from quivers.qiec.checking import ComputationSignature, KernelRegistry
from quivers.qiec.contracts import InterpretationId
from quivers.qiec.effects import EMPTY_ROW, ComputationType, HandlerDef
from quivers.qiec.foreign import DeductionBooleanOutcomes
from quivers.qiec.identifiers import ComputationId, HandlerId, SourceOrigin
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.module import (
    NamedComputation,
    QiecModule,
    _first_handled_call,
    validate_module,
)
from quivers.qiec.rules import RuleError, RulePresentation
from quivers.qiec.terms import (
    Bind,
    Call,
    Handle,
    LiteralValue,
    Local,
    Perform,
    TensorValue,
    Value,
    Var,
)
from quivers.qiec.types import BOOL, REAL, STRING, IndexLiteral, IndexVariable


class PackedBooleanInterpretationBinding(QiecModel):
    """A portable association between a proof interpretation and its packed use.

    Parameters
    ----------
    interpretation : InterpretationId
        The closed interpretation whose operation table is computed.
    computation : ComputationId
        The wrapper that supplies that table to the native query.
    query : ComputationId
        The checked lexical query receiving the table.
    """

    interpretation: InterpretationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    computation: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    query: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    def validate(
        self, module: QiecModule, interpretation: ClosedInterpretationBinding
    ) -> None:
        """Recheck the ordinary fold and its exact packed application.

        Parameters
        ----------
        module
            The executable containing both interpretations.
        interpretation
            The independently checked companion of the ordinary proof fold.

        Raises
        ------
        RuleError
            If the association is stale or the query changes the table.
        KernelError
            If a checked module or an operation is ill typed.
        """
        check_packed_boolean_bindings(
            module, validate_module(module), (interpretation,), (self,)
        )


def prepare_packed_boolean_computation(
    registry: KernelRegistry,
    semantics: DeductionBooleanOutcomes,
    interpretation: ClosedInterpretationBinding,
    signature: ComputationSignature,
    query: ComputationSignature,
    origin: SourceOrigin,
) -> NamedComputation:
    """Build the exact ordered table of this closed proof interpretation.

    Parameters
    ----------
    registry
        The checked operation signatures.
    semantics
        The lexical chart and its table layout.
    interpretation
        The nominal assignment of operations to proof constructors.
    signature, query
        The wrapper and the native operation adapter, respectively.
    origin
        The generated wrapper's source origin.

    Returns
    -------
    NamedComputation
        Ordinary calls followed by one application of the checked adapter.

    Raises
    ------
    RuleError
        If the carrier, operations, or adapter signature is incompatible.
    """
    presentation = semantics.chart.presentation
    assert isinstance(presentation, RulePresentation)
    if interpretation.carrier != BOOL or interpretation.effects != EMPTY_ROW:
        raise RuleError(
            "packed Boolean proof outcomes require a closed Bool interpretation with an empty effect row"
        )
    if interpretation.presentation.id != presentation.id:
        raise RuleError(
            "packed Boolean proof outcomes require the same rule presentation"
        )
    assignments = dict(interpretation.interpretation.operations)
    (slot,) = presentation.axioms
    constructors = (slot, *presentation.rules)
    if len(assignments) != len(interpretation.interpretation.operations) or set(
        assignments
    ) != {constructor.id for constructor in constructors}:
        raise RuleError(
            "packed Boolean proof outcomes require one operation per proof constructor"
        )
    operations = tuple(
        registry.computation(assignments[constructor.id])
        for constructor in constructors
    )
    if any(
        operation.telescope or operation.effects != EMPTY_ROW
        for operation in operations
    ):
        raise RuleError(
            "packed Boolean proof outcomes require rule and axiom operations with no static parameters or effects"
        )
    if any(
        operation.result != BOOL or operation.parameters != parameters
        for operation, parameters in zip(
            operations,
            ((STRING,), *((BOOL,) * rule.arity for rule in presentation.rules)),
            strict=True,
        )
    ):
        raise RuleError(
            "packed Boolean proof operations have incompatible argument or result types"
        )
    if len(query.telescope) != 1 or len(query.parameters) not in (3, 4):
        raise RuleError("packed Boolean proof outcomes have the wrong query arity")
    (extent,) = query.telescope
    if not isinstance(extent, IndexBinder) or extent.sort != NAT:
        raise RuleError("packed Boolean proof outcomes require one token extent")
    extent_term = IndexVariable(name=extent.name, sort=extent.sort)
    tokens_type = tensor_type(STRING, (extent_term,))
    table_type = tensor_type(
        BOOL, (IndexLiteral(value=semantics.table_size, sort=NAT),)
    )
    productions = query.parameters[3:]
    if productions:
        layout = tensor_shape(productions[0])
        if (
            layout is None
            or layout[0] != REAL
            or len(layout[1]) != 1
            or not isinstance(layout[1][0], IndexLiteral)
        ):
            raise RuleError("packed Boolean query has an invalid production table")
    if (
        query.parameters[:3] != (tokens_type, table_type, BOOL)
        or signature.parameters != (tokens_type, BOOL, *productions)
        or signature.telescope != query.telescope
        or signature.result != query.result
        or signature.effects != query.effects
    ):
        raise RuleError(
            "packed Boolean proof outcomes have an incompatible query signature"
        )

    tokens = Local(name="tokens", type=tokens_type)
    mode = Local(name="normalize", type=BOOL)
    table_parameter = tuple(
        Local(name="productions", type=type_) for type_ in productions
    )
    inputs: list[tuple[ComputationSignature, tuple[Value, ...]]] = [
        (operations[0], (LiteralValue(value=word, type=STRING),))
        for word in semantics.lexical_words
    ]
    inputs.extend(
        (
            operation,
            tuple(LiteralValue(value=truth, type=BOOL) for truth in truths),
        )
        for rule, operation in zip(presentation.rules, operations[1:], strict=True)
        for truths in product((False, True), repeat=rule.arity)
    )
    table = tuple(
        Local(name=f"truth{index}", type=BOOL) for index in range(len(inputs))
    )
    path = origin.structural_path
    body = bind_operands(
        tuple(
            (
                local,
                Call(
                    callee=operation.id,
                    name=operation.name,
                    static_arguments=(),
                    arguments=arguments,
                    result_type=BOOL,
                    effects=EMPTY_ROW,
                    origin=origin.with_(
                        structural_path=(*path, "truth", index), role="call"
                    ),
                ),
            )
            for index, (local, (operation, arguments)) in enumerate(
                zip(table, inputs, strict=True)
            )
        ),
        Call(
            callee=query.id,
            name=query.name,
            static_arguments=(extent_term,),
            arguments=(
                Var(local=tokens),
                TensorValue(
                    items=tuple(Var(local=value) for value in table),
                    result_type=table_type,
                ),
                Var(local=mode),
                *(Var(local=value) for value in table_parameter),
            ),
            result_type=query.result,
            effects=query.effects,
            origin=origin.with_(structural_path=(*path, "query"), role="call"),
        ),
    )
    return NamedComputation(
        id=signature.id,
        name=signature.name,
        telescope=signature.telescope,
        parameters=(tokens, mode, *table_parameter),
        body=body,
        type=ComputationType(result=signature.result, effects=signature.effects),
        origin=origin,
    )


def _forwarded_table(
    query: NamedComputation, handlers: Mapping[HandlerId, HandlerDef]
) -> DeductionBooleanOutcomes:
    """Locate the native request and check that its arguments are forwarded.

    Parameters
    ----------
    query
        The checked lexical query with its weight-reading binds.
    handlers
        The executable handler declarations by identity.

    Returns
    -------
    DeductionBooleanOutcomes
        The exact native Boolean payload reached by the query.

    Raises
    ------
    RuleError
        If the original checked arguments are shadowed or replaced.
    """
    if len(query.parameters) not in (3, 4):
        raise RuleError("packed Boolean query has the wrong parameter layout")
    tokens, table, normalizer = query.parameters[:3]
    protected = {parameter.name for parameter in query.parameters}
    tail = query.body
    while isinstance(tail, Bind):
        if tail.binder.name in protected:
            raise RuleError("packed Boolean query shadows an original argument")
        tail = tail.then
    if not isinstance(tail, Handle) or not isinstance(tail.computation, Perform):
        raise RuleError("packed Boolean query does not reach its native operation")
    handler = handlers.get(tail.handler)
    request = tail.computation.request
    if (
        handler is None
        or not isinstance(handler.foreign, DeductionBooleanOutcomes)
        or tail.instance != request.instance
        or len(request.arguments) != 5
        or tuple(request.arguments[index] for index in (0, 3, 4))
        != tuple(Var(local=value) for value in (tokens, table, normalizer))
    ):
        raise RuleError("packed Boolean query does not forward its checked truth table")
    return handler.foreign


def check_packed_boolean_bindings(
    module: QiecModule,
    registry: KernelRegistry,
    interpretations: Sequence[ClosedInterpretationBinding],
    bindings: Sequence[PackedBooleanInterpretationBinding],
) -> None:
    """Verify compiled or restored packed uses against an already checked kernel.

    Parameters
    ----------
    module, registry
        The executable and the registry resolved from that exact executable.
    interpretations
        The resolved ordinary proof interpretations.
    bindings
        The packed uses attached by source elaboration.

    Raises
    ------
    RuleError
        If a use has stale declarations, changed calls, or a hidden handler.
    """
    if not bindings:
        return
    ordinary = {item.interpretation.id: item for item in interpretations}
    computations = {item.id: item for item in module.computations}
    handlers = {item.id: item for item in module.handlers}
    instances = {item.entry.instance: item for item in module.instances}
    checked: set[InterpretationId] = set()
    visited: set[ComputationId] = set()
    for packed in bindings:
        binding = ordinary.get(packed.interpretation)
        wrapper = computations.get(packed.computation)
        query = computations.get(packed.query)
        if binding is None or wrapper is None or query is None:
            raise RuleError(
                "packed Boolean interpretation has stale executable identities"
            )
        root = computations.get(binding.computation)
        if packed.interpretation not in checked:
            binding._validate_root(module.module, registry, root, instances)
            checked.add(packed.interpretation)
        assert root is not None
        name = binding.presentation.name
        weighted = len(query.parameters) == 4
        prefix = "weighted_" if weighted else ""
        wrapper_name = f"{name}__{prefix}{root.name}__packed_boolean_outcomes"
        query_name = f"{name}__{prefix}boolean_outcomes"
        path = (
            "deductions",
            name,
            "packed_boolean_outcomes",
            root.name,
            *(("weighted",) if weighted else ()),
        )
        if (
            wrapper.name != wrapper_name
            or wrapper.id
            != ComputationId.derive(module.module, "computation", wrapper_name)
            or query.name != query_name
            or query.id
            != ComputationId.derive(module.module, "computation", query_name)
            or wrapper.origin.structural_path != path
        ):
            raise RuleError("packed Boolean interpretation has a stale nominal use")
        chart = _forwarded_table(query, handlers)
        presentation = chart.chart.presentation
        assert isinstance(presentation, RulePresentation)
        # Weight addresses can change when a normalized chart reads a
        # production table. The proof constructors preserve their nominal
        # identities, ordered types, and static arguments across that change.
        proof = export_lexical_proofs(presentation, binding.presentation.name)
        if proof.declarations() != binding.presentation.declarations():
            raise RuleError(
                "packed Boolean interpretation uses another proof presentation"
            )
        expected = prepare_packed_boolean_computation(
            registry,
            chart,
            binding,
            registry.computation(wrapper.id),
            registry.computation(query.id),
            wrapper.origin,
        )
        if expected != wrapper:
            raise RuleError(
                "packed Boolean interpretation does not compute its checked truth table "
                "or forward the production table"
            )
        for _, operation in binding.interpretation.operations:
            owner = _first_handled_call(operation, computations, visited)
            if owner is not None:
                raise RuleError(
                    f"packed Boolean interpretation {owner.name!r} installs an effect handler"
                )


__all__ = ["PackedBooleanInterpretationBinding"]
