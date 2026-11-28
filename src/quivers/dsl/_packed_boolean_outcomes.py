"""Lower a finite Boolean proof algebra to an ordinary checked packed query."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from quivers.dsl.ast_nodes import LetExprCall, LetExprVar
from quivers.dsl._deduction_outcomes import (
    OUTCOME_CODE,
    PACKED_BOOLEAN_CALLS,
    _first_handled_call,
    outcome_call_sites,
)
from quivers.qiec._proof_interpretation import bind_operands
from quivers.qiec.checking import ComputationSignature, KernelError
from quivers.qiec.effects import EMPTY_ROW, ComputationType
from quivers.qiec.foreign import DeductionBooleanOutcomes
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.kinds import IndexBinder
from quivers.qiec.module import NamedComputation
from quivers.qiec.rules import RulePresentation
from quivers.qiec.terms import (
    Call,
    Computation,
    LiteralValue,
    Local,
    TensorValue,
    Value,
    Var,
)
from quivers.qiec.types import BOOL, STRING, IndexVariable

if TYPE_CHECKING:
    from quivers.dsl.qiec_lowering import _Elaborator

type _Roots = tuple[tuple[LetExprCall, tuple[ComputationId, ...]], ...]


def declare_packed_boolean_computations(
    elaborator: _Elaborator,
) -> tuple[tuple[NamedComputation, ...], _Roots]:
    """Use the checked interpretation's exact operations at every truth tuple."""
    closed = {
        binding.computation: binding for binding in elaborator.closed_interpretations
    }
    signatures = {
        signature.id: signature
        for signature in elaborator.computation_signatures.values()
    }
    wrappers: dict[tuple[str, str], NamedComputation] = {}
    roots: list[tuple[LetExprCall, tuple[ComputationId, ...]]] = []
    for call in outcome_call_sites(elaborator):
        if call.func not in PACKED_BOOLEAN_CALLS or len(call.args) != 3:
            continue
        deduction, tokens, selector = call.args
        if (
            not isinstance(deduction, LetExprVar)
            or not isinstance(tokens, LetExprVar)
            or not isinstance(selector, LetExprVar)
        ):
            continue
        query = elaborator._boolean_outcome_queries.get(deduction.name)
        if query is None or (deduction.name, selector.name) in wrappers:
            continue
        semantics = query.handler.foreign
        assert isinstance(semantics, DeductionBooleanOutcomes)
        presentation = semantics.chart.presentation
        assert isinstance(presentation, RulePresentation)
        source = elaborator.computation_signatures.get(selector.name)
        binding = closed.get(source.id) if source is not None else None
        if (
            binding is None
            or binding.presentation != elaborator._lexical_proofs.get(deduction.name)
            or binding.carrier != BOOL
            or binding.effects != EMPTY_ROW
        ):
            elaborator._fail(
                selector,
                "packed Boolean proof outcomes require a closed Bool interpretation "
                f"of {deduction.name!r} with an empty effect row",
                code=OUTCOME_CODE,
            )
        assignments = dict(binding.interpretation.operations)
        (slot,) = presentation.axioms
        operations = tuple(
            signatures[assignments[constructor.id]]
            for constructor in (slot, *presentation.rules)
        )
        if any(op.telescope or op.effects != EMPTY_ROW for op in operations):
            elaborator._fail(
                selector,
                "packed Boolean proof outcomes require rule and axiom operations "
                "with no static parameters or effects",
                code=OUTCOME_CODE,
            )
        table_type = query.boolean_table
        assert table_type is not None
        low_level = elaborator.computation_signatures[
            f"{deduction.name}__boolean_outcomes"
        ]
        name = f"{deduction.name}__{selector.name}__packed_boolean_outcomes"
        if name in elaborator.computation_signatures:
            elaborator._fail(
                selector,
                f"generated Boolean outcome computation {name!r} already exists",
                code=OUTCOME_CODE,
            )
        (extent,) = low_level.telescope
        assert isinstance(extent, IndexBinder)
        extent_term = IndexVariable(name=extent.name, sort=extent.sort)
        token_type, _, mode_type = low_level.parameters
        token_local = Local(name="tokens", type=token_type)
        mode = Local(name="normalize", type=mode_type)
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
        path = ("deductions", deduction.name, "packed_boolean_outcomes", selector.name)
        body: Computation = Call(
            callee=low_level.id,
            name=low_level.name,
            static_arguments=(extent_term,),
            arguments=(
                Var(local=token_local),
                TensorValue(
                    items=tuple(Var(local=value) for value in table),
                    result_type=table_type,
                ),
                Var(local=mode),
            ),
            result_type=low_level.result,
            effects=low_level.effects,
            origin=elaborator._origin(selector, (*path, "query"), "call"),
        )
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
                        origin=elaborator._origin(
                            selector, (*path, "truth", index), "call"
                        ),
                    ),
                )
                for index, (local, (operation, arguments)) in enumerate(
                    zip(table, inputs, strict=True)
                )
            ),
            body,
        )
        signature = ComputationSignature(
            id=ComputationId.derive(elaborator.source.module_name, "computation", name),
            name=name,
            telescope=low_level.telescope,
            parameters=(token_type, mode_type),
            result=low_level.result,
            effects=low_level.effects,
        )
        try:
            elaborator.registry.register_computation(signature)
        except KernelError as error:
            elaborator._fail_kernel(selector, error, fallback=OUTCOME_CODE)
        elaborator.computation_signatures[name] = signature
        wrappers[deduction.name, selector.name] = NamedComputation(
            id=signature.id,
            name=signature.name,
            telescope=signature.telescope,
            parameters=(token_local, mode),
            body=body,
            type=ComputationType(result=signature.result, effects=signature.effects),
            origin=elaborator._origin(selector, path, "computation"),
        )
        roots.append((call, tuple(op.id for op in operations)))
    return tuple(wrappers.values()), tuple(roots)


def check_packed_boolean_bodies(
    elaborator: _Elaborator, computations: tuple[NamedComputation, ...], roots: _Roots
) -> None:
    """Refuse hidden handlers before replacing repeated evaluations by a table."""
    if not roots:
        return
    by_id = {computation.id: computation for computation in computations}
    visited: set[ComputationId] = set()
    for call, operations in roots:
        for operation in operations:
            local_handler = _first_handled_call(operation, by_id, visited)
            if local_handler is not None:
                elaborator._fail(
                    call,
                    "packed Boolean proof operations must be deterministic; "
                    f"{local_handler.name!r} installs an effect handler",
                    code=OUTCOME_CODE,
                )


__all__: list[str] = []
