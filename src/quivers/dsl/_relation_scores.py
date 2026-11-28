"""Elaborate source-declared relation and string graph scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quivers.core.algebras import LOG_PROB
from quivers.dsl._relation_specs import AuthoredRelationSpec, authored_relation_specs
from quivers.dsl._string_graph_specs import authored_string_graph_specs
from quivers.dsl.ast_nodes import (
    QiecHandlerDecl,
    QiecRelationScoreDecl,
    QiecStringGraphScoreDecl,
)
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
from quivers.qiec.string_graph_runtime import (
    _string_graph_score_handler_name,
    _tree_string_graph_handler,
)
from quivers.qiec.relation_composition import ComposedRelationSpec
from quivers.qiec.relation_transforms import (
    BinaryRelationTransformSpec,
    UnaryRelationTransformSpec,
)
from quivers.qiec.terms import Handle, LiteralValue, Local, Perform
from quivers.qiec.types import REAL, UNIT, TypeExpr

if TYPE_CHECKING:
    from quivers.dsl.qiec_lowering import _Elaborator


@dataclass(frozen=True, slots=True)
class _RelationScore:
    declaration: QiecRelationScoreDecl | QiecStringGraphScoreDecl
    path: tuple[str, ...]
    signature: ComputationSignature
    instance: NamedEffectInstance
    handler: HandlerDef


def _score_specification(
    elaborator: _Elaborator,
    declaration: QiecRelationScoreDecl,
    specs: dict[str, AuthoredRelationSpec],
    result: TypeExpr,
) -> AuthoredRelationSpec | None:
    operation = declaration.operation
    if operation in ("path_entropy", "expected_path_cost") and result != REAL:
        elaborator._fail(
            declaration,
            "path statistics require a Real result",
            code="qiec-relation-score",
        )
    if declaration.relation is None:
        return None
    relation = specs.get(declaration.relation)
    if relation is None:
        elaborator._fail(
            declaration,
            f"unknown authored relation {declaration.relation!r}",
            code="qiec-relation-score",
        )
    if isinstance(
        relation,
        ComposedRelationSpec | UnaryRelationTransformSpec | BinaryRelationTransformSpec,
    ):
        if operation != "pair_weight":
            kind = (
                "compositions"
                if isinstance(relation, ComposedRelationSpec)
                else "transformations"
            )
            elaborator._fail(
                declaration,
                f"authored {kind} currently support pair_weight scores",
                code="qiec-relation-score",
            )
        algebra = (
            relation.source.algebra
            if isinstance(relation, UnaryRelationTransformSpec)
            else relation.left.algebra
        )
    else:
        algebra = relation.algebra
        if operation == "conditional_log_prob" and not 0 < relation.input_tapes < len(
            relation.tapes
        ):
            elaborator._fail(
                declaration,
                "an authored conditional score requires input and output tapes",
                code="qiec-relation-score",
            )
    if operation == "pair_weight":
        expected = LOG_WEIGHT if algebra == LOG_PROB.subject else REAL
        if result != expected:
            elaborator._fail(
                declaration,
                f"the authored relation's pair weight requires {expected.constructor.name}",
                code="qiec-relation-score",
            )
    return relation


def declare_relation_scores(elaborator: _Elaborator) -> tuple[_RelationScore, ...]:
    """Register score signatures and their checked foreign declarations."""
    routes: list[_RelationScore] = []
    specs = authored_relation_specs(
        elaborator.source.syntax,
        elaborator.source.module_name,
        elaborator.source.file_path,
    )
    graphs = authored_string_graph_specs(
        elaborator.source.syntax, specs, elaborator.source.file_path
    )
    declared_handlers = {
        statement.name
        for statement in elaborator.source.syntax.statements
        if isinstance(statement, QiecHandlerDecl)
    }
    for declaration in elaborator.source.syntax.statements:
        if not isinstance(
            declaration, QiecRelationScoreDecl | QiecStringGraphScoreDecl
        ):
            continue
        name = declaration.name
        is_graph = isinstance(declaration, QiecStringGraphScoreDecl)
        path = (
            "string_graph_scores" if is_graph else "relation_scores",
            name,
            declaration.operation,
        )
        description = "string graph score" if is_graph else "relation score"
        diagnostic = "qiec-string-graph-score" if is_graph else "qiec-relation-score"
        if name in elaborator.computation_signatures:
            elaborator._fail(
                declaration,
                f"duplicate computation {name!r}",
                code=diagnostic,
            )
        result = elaborator._lower_type(declaration.result_type, ())
        if result not in (REAL, LOG_WEIGHT):
            elaborator._fail(
                declaration,
                f"a {description} requires a Real or LogWeight result",
                code=diagnostic,
            )
        if (
            declaration.operation in ("conditional_log_prob", "tree_log_prob")
            and result != LOG_WEIGHT
        ):
            elaborator._fail(
                declaration,
                (
                    "a tree word score requires a LogWeight result"
                    if is_graph
                    else "a conditional relation score requires a LogWeight result"
                ),
                code=diagnostic,
            )
        handler_name = (
            _string_graph_score_handler_name(name)
            if is_graph
            else _relation_score_handler_name(name)
        )
        instance_name = f"{handler_name}__compute"
        if (
            handler_name in declared_handlers
            or handler_name in elaborator.handlers
            or instance_name in elaborator.instances
        ):
            elaborator._fail(
                declaration,
                f"{description} {name!r} conflicts with an existing attachment",
                code=diagnostic,
            )
        effect = COMPUTE_EFFECT.apply((UNIT, result))
        if elaborator.effects.setdefault("Compute", COMPUTE_EFFECT) != COMPUTE_EFFECT:
            elaborator._fail(
                declaration,
                f"{description}s require the prelude Compute interface",
                code=diagnostic,
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
        if is_graph:
            assert isinstance(declaration, QiecStringGraphScoreDecl)
            graph = None
            if declaration.graph is not None:
                graph = graphs.get(declaration.graph)
                if graph is None:
                    elaborator._fail(
                        declaration,
                        f"unknown authored string graph {declaration.graph!r}",
                        code=diagnostic,
                    )
                if declaration.operation == "tree_partition":
                    expected = LOG_WEIGHT if graph.algebra == LOG_PROB.subject else REAL
                    if result != expected:
                        elaborator._fail(
                            declaration,
                            f"the authored string graph partition requires {expected.constructor.name}",
                            code=diagnostic,
                        )
            handler = _tree_string_graph_handler(
                module,
                handler_name,
                declaration.operation.removeprefix("tree_"),
                result,
                graph=graph,
            )
        else:
            assert isinstance(declaration, QiecRelationScoreDecl)
            relation = _score_specification(elaborator, declaration, specs, result)
            handler = HandlerDef(
                name=handler_name,
                id=_relation_score_handler_id(
                    module, name, declaration.operation, relation
                ),
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
        routes.append(_RelationScore(declaration, path, signature, instance, handler))
    return tuple(routes)


def lower_relation_scores(
    elaborator: _Elaborator, routes: tuple[_RelationScore, ...]
) -> tuple[NamedComputation, ...]:
    """Build checked unit-argument calls to the declared runtime providers."""
    results: list[NamedComputation] = []
    for route in routes:
        declaration = route.declaration
        path = route.path
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
