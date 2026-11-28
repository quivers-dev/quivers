"""Lower a finite Boolean proof algebra to an ordinary checked packed query."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quivers.dsl.ast_nodes import LetExprCall, LetExprVar
from quivers.dsl._deduction_outcomes import (
    OUTCOME_CODE,
    PACKED_BOOLEAN_CALLS,
    outcome_call_sites,
)
from quivers.qiec._packed_boolean_interpretation import (
    PackedBooleanInterpretationBinding,
    prepare_packed_boolean_computation,
)
from quivers.qiec.checking import ComputationSignature, KernelError
from quivers.qiec.effects import EMPTY_ROW
from quivers.qiec.foreign import DeductionBooleanOutcomes
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.module import NamedComputation, _first_handled_call
from quivers.qiec.rules import RuleError
from quivers.qiec.types import BOOL

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
        token_type, _, mode = low_level.parameters
        signature = ComputationSignature(
            id=ComputationId.derive(elaborator.source.module_name, "computation", name),
            name=name,
            telescope=low_level.telescope,
            parameters=(token_type, mode),
            result=low_level.result,
            effects=low_level.effects,
        )
        path = ("deductions", deduction.name, "packed_boolean_outcomes", selector.name)
        try:
            wrapper = prepare_packed_boolean_computation(
                elaborator.registry,
                semantics,
                binding,
                signature,
                low_level,
                elaborator._origin(selector, path, "computation"),
            )
            elaborator.registry.register_computation(signature)
        except (KernelError, RuleError) as error:
            elaborator._fail_kernel(selector, error, fallback=OUTCOME_CODE)
        elaborator.computation_signatures[name] = signature
        wrappers[deduction.name, selector.name] = wrapper
        elaborator.packed_interpretations.append(
            PackedBooleanInterpretationBinding(
                interpretation=binding.interpretation.id,
                computation=wrapper.id,
                query=low_level.id,
            )
        )
        roots.append((call, tuple(op for _, op in binding.interpretation.operations)))
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
