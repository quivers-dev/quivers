"""Elaborate source-declared fixed relation score computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quivers.dsl.ast_nodes import QiecHandlerDecl, QiecRelationScoreDecl
from quivers.qiec.builtins import COMPUTE_APPLY, COMPUTE_EFFECT
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.checking import ComputationSignature
from quivers.qiec.effects import (
    EMPTY_ROW,
    EffectRequest,
    HandlerClauseDef,
    HandlerDef,
    ResumptionGrade,
    SiteProvenance,
    instantiate_effect,
)
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.module import NamedComputation, NamedEffectInstance
from quivers.qiec.relation_runtime import (
    _relation_score_handler_id,
    _relation_score_handler_name,
)
from quivers.qiec.terms import Handle, LiteralValue, Local, Perform
from quivers.qiec.types import REAL, UNIT

if TYPE_CHECKING:
    from quivers.dsl.qiec_lowering import _Elaborator


@dataclass(frozen=True, slots=True)
class _RelationScore:
    declaration: QiecRelationScoreDecl
    signature: ComputationSignature
    instance: NamedEffectInstance
    handler: HandlerDef


def declare_relation_scores(elaborator: _Elaborator) -> tuple[_RelationScore, ...]:
    """Register score signatures and their checked foreign declarations."""
    routes: list[_RelationScore] = []
    declared_handlers = {
        statement.name
        for statement in elaborator.source.syntax.statements
        if isinstance(statement, QiecHandlerDecl)
    }
    for declaration in elaborator.source.syntax.statements:
        if not isinstance(declaration, QiecRelationScoreDecl):
            continue
        name = declaration.name
        path = ("relation_scores", name, declaration.operation)
        if name in elaborator.computation_signatures:
            elaborator._fail(
                declaration,
                f"duplicate computation {name!r}",
                code="qiec-relation-score",
            )
        result = elaborator._lower_type(declaration.result_type, ())
        if result not in (REAL, LOG_WEIGHT):
            elaborator._fail(
                declaration,
                "a relation score requires a Real or LogWeight result",
                code="qiec-relation-score",
            )
        if declaration.operation == "conditional_log_prob" and result != LOG_WEIGHT:
            elaborator._fail(
                declaration,
                "a conditional relation score requires a LogWeight result",
                code="qiec-relation-score",
            )
        handler_name = _relation_score_handler_name(name)
        instance_name = f"{handler_name}__compute"
        if (
            handler_name in declared_handlers
            or handler_name in elaborator.handlers
            or instance_name in elaborator.instances
        ):
            elaborator._fail(
                declaration,
                f"relation score {name!r} conflicts with an existing attachment",
                code="qiec-relation-score",
            )
        effect = COMPUTE_EFFECT.apply((UNIT, result))
        if elaborator.effects.setdefault("Compute", COMPUTE_EFFECT) != COMPUTE_EFFECT:
            elaborator._fail(
                declaration,
                "relation scores require the prelude Compute interface",
                code="qiec-relation-score",
            )
        if COMPUTE_EFFECT.ref.id not in elaborator.registry.effects:
            elaborator.registry.register_effect(COMPUTE_EFFECT)
        module = elaborator.source.module_name
        instance = NamedEffectInstance(
            name=instance_name,
            entry=instantiate_effect(
                effect, module=module, lexical_path=(*path, "instance")
            ),
            origin=elaborator._origin(
                declaration, (*path, "instance"), "effect-instance"
            ),
        )
        handler = HandlerDef(
            name=handler_name,
            id=_relation_score_handler_id(module, name, declaration.operation),
            effect=effect,
            clauses=(
                HandlerClauseDef(
                    operation=COMPUTE_APPLY,
                    grade=ResumptionGrade.LINEAR,
                    parameters=(Local(name="argument", type=UNIT),),
                ),
            ),
            input_type=result,
            output_type=result,
        )
        signature = ComputationSignature(
            id=ComputationId.derive(module, "computation", name),
            name=name,
            telescope=(),
            parameters=(),
            result=result,
            effects=EMPTY_ROW,
        )
        elaborator.instances[instance_name] = instance
        elaborator.handlers[handler_name] = handler
        elaborator.registry.register_computation(signature)
        elaborator.computation_signatures[name] = signature
        routes.append(_RelationScore(declaration, signature, instance, handler))
    return tuple(routes)


def lower_relation_scores(
    elaborator: _Elaborator, routes: tuple[_RelationScore, ...]
) -> tuple[NamedComputation, ...]:
    """Build checked unit-argument calls to the declared runtime providers."""
    results: list[NamedComputation] = []
    for route in routes:
        declaration = route.declaration
        path = ("relation_scores", declaration.name, declaration.operation)
        request = EffectRequest(
            instance=route.instance.entry.instance,
            effect=route.instance.entry.effect,
            operation=COMPUTE_APPLY,
            static_arguments=(),
            arguments=(LiteralValue(value=None, type=UNIT),),
            result_type=route.signature.result,
            origin=SiteProvenance(
                origin=elaborator._origin(
                    declaration, (*path, "apply"), "effect-request"
                )
            ),
        )
        results.append(
            elaborator._checked_computation(
                route.signature,
                (),
                Handle(
                    instance=route.instance.entry.instance,
                    handler=route.handler.id,
                    computation=Perform(request=request),
                ),
                declaration,
                path,
            )
        )
    return tuple(results)
