"""Source declarations of closed checked folds over deduction proofs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import didactic.api as dx

from quivers.dsl.ast_nodes import QiecInterpretationDecl
from quivers.qiec._proof_interpretation import prepare_closed_interpretation
from quivers.qiec.checking import ComputationSignature, KernelError
from quivers.qiec.contracts import InterpretationId
from quivers.qiec.identifiers import ComputationId, ConstructorId
from quivers.qiec.module import NamedComputation
from quivers.qiec.rules import InterpretationDecl

if TYPE_CHECKING:
    from quivers.dsl.qiec_lowering import _Elaborator


def declare_interpretations(elaborator: _Elaborator) -> tuple[NamedComputation, ...]:
    """Register ordinary fold signatures before lowering their callers."""
    generated: list[NamedComputation] = []
    seen: set[str] = set()
    for declaration in elaborator.source.syntax.statements:
        if not isinstance(declaration, QiecInterpretationDecl):
            continue
        if (
            declaration.name in seen
            or declaration.name in elaborator.computation_signatures
        ):
            elaborator._fail(
                declaration,
                f"duplicate computation {declaration.name!r}",
                code="qiec-interpretation",
            )
        seen.add(declaration.name)
        fold = _interpretation(elaborator, declaration)
        generated.append(fold)

    for fold in generated:
        signature = ComputationSignature(
            id=fold.id,
            name=fold.name,
            telescope=fold.telescope,
            parameters=tuple(parameter.type for parameter in fold.parameters),
            result=fold.type.result,
            effects=fold.type.effects,
        )
        elaborator.registry.register_computation(signature)
        elaborator.computation_signatures[signature.name] = signature
    return tuple(generated)


def _interpretation(
    elaborator: _Elaborator, declaration: QiecInterpretationDecl
) -> NamedComputation:
    """Resolve source clauses to checked constructor and computation identities."""
    presentation = elaborator._lexical_proofs.get(declaration.deduction)
    if presentation is None:
        elaborator._fail(
            declaration,
            f"unknown deduction {declaration.deduction!r}",
            code="qiec-interpretation",
        )
    system = elaborator._deductions[declaration.deduction]
    presented = system.presentation_cell[0]
    assert presented is not None
    source = presented.presentation
    judgment_names = {judgment.id: judgment.name for judgment in source.judgments}
    slots: dict[tuple[str, str], list[ConstructorId]] = {}
    for rule in presented.rules:
        slots.setdefault(("rule", rule.source.name), []).append(rule.rule.id)
    for axiom in source.axioms:
        name = judgment_names[axiom.judgment.family]
        slots.setdefault(("axiom", name), []).append(axiom.id)

    selected: list[tuple[ConstructorId, ComputationId]] = []
    used: set[tuple[str, str]] = set()
    for clause in declaration.clauses:
        slot = clause.constructor_kind, clause.constructor
        if slot not in slots:
            elaborator._fail(
                clause,
                f"unknown {slot[0]} {slot[1]!r} in {declaration.deduction!r}",
                code="qiec-interpretation",
            )
        if slot in used:
            elaborator._fail(
                clause,
                f"duplicate interpretation for {slot[0]} {slot[1]!r}",
                code="qiec-interpretation",
            )
        used.add(slot)
        operation = elaborator.computation_signatures.get(clause.operation)
        if operation is None:
            elaborator._fail(
                clause,
                f"unknown interpretation operation {clause.operation!r}",
                code="qiec-interpretation",
            )
        selected.extend((id_, operation.id) for id_ in slots[slot])
    missing = tuple(
        f"{kind} {name!r}" for kind, name in slots if (kind, name) not in used
    )
    if missing:
        elaborator._fail(
            declaration,
            f"missing interpretation for {', '.join(missing)}",
            code="qiec-interpretation",
        )

    (judgment,) = presentation.judgments
    origin = elaborator._origin(declaration, ("proof_interpretations",), "computation")
    interpretation = InterpretationDecl(
        id=InterpretationId.derive(
            origin.module, *origin.structural_path, declaration.name
        ),
        presentation=presentation.id,
        operations=tuple(selected),
    )
    try:
        (fold,) = prepare_closed_interpretation(
            elaborator.registry,
            presentation,
            interpretation,
            names={judgment.id: declaration.name},
            carriers={judgment.id: elaborator._lower_type(declaration.carrier, ())},
            origin=origin,
        )
    except (KernelError, ValueError, dx.ValidationError) as error:
        elaborator._fail_kernel(declaration, error, fallback="qiec-interpretation")
    return fold
