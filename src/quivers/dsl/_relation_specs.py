"""Elaborate authored sparse topology without binding runtime weights."""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from typing import Literal, cast

import didactic.api as dx

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.objects import FinSet, StringSet
from quivers.core.relations import Arc, Boundary, RelationTopology
from quivers.dsl.ast_nodes import (
    Module,
    QiecRelationComposeDecl,
    QiecRelationDecl,
    QiecRelationTransformDecl,
)
from quivers.dsl.ast_nodes.declarations import ObjectDecl, TypeFromExpr
from quivers.dsl.ast_nodes.objects import DiscreteConstructor, TypeName
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec.contracts import ParameterId
from quivers.qiec.relation_cascades import RelationCascadeSpec, RelationCascadeStep
from quivers.qiec.relation_composition import ComposedRelationSpec
from quivers.qiec.relation_transforms import (
    BinaryRelationTransformSpec,
    UnaryRelationTransformSpec,
)
from quivers.qiec.relations import RationalRelationSpec


type AuthoredRelationSpec = (
    RationalRelationSpec
    | ComposedRelationSpec
    | UnaryRelationTransformSpec
    | BinaryRelationTransformSpec
    | RelationCascadeSpec
)
type _Declaration = (
    QiecRelationDecl | QiecRelationComposeDecl | QiecRelationTransformDecl
)


def _alphabet(name: str, objects: dict[str, ObjectDecl]) -> FinSet:
    visited: set[str] = set()
    original = name
    while name in objects and name not in visited:
        visited.add(name)
        init = objects[name].init
        if not isinstance(init, TypeFromExpr):
            break
        expr = init.expr
        if isinstance(expr, TypeName):
            if expr.name.isdecimal():
                return FinSet(name=name, cardinality=int(expr.name))
            name = expr.name
            continue
        if (
            isinstance(expr, DiscreteConstructor)
            and expr.constructor == "FinSet"
            and not expr.kwargs
            and len(expr.args) == 1
            and expr.args[0].isdecimal()
        ):
            return FinSet(name=name, cardinality=int(expr.args[0]))
        break
    raise ValueError(f"relation tape {original!r} requires a declared finite alphabet")


def _specification(
    declaration: QiecRelationDecl,
    objects: dict[str, ObjectDecl],
    module_name: str,
) -> RationalRelationSpec:
    count = sum(tape.direction == "input" for tape in declaration.tapes)
    order = tuple(tape.direction for tape in declaration.tapes)
    if order != ("input",) * count + ("output",) * (len(order) - count):
        raise ValueError("input relation tapes must precede output tapes")
    tapes = tuple(
        StringSet(alphabet=_alphabet(tape.alphabet, objects))
        for tape in declaration.tapes
    )
    slots = tuple(
        dict.fromkeys(
            (
                *(mark.weight for mark in declaration.initial),
                *(arc.weight for arc in declaration.arcs),
                *(mark.weight for mark in declaration.final),
            )
        )
    )
    positions = {name: index for index, name in enumerate(slots)}
    topology = RelationTopology(
        tapes=tapes,
        input_tapes=count,
        n_states=declaration.n_states,
        arcs=tuple(
            Arc(arc.source, arc.target, arc.labels, positions[arc.weight])
            for arc in declaration.arcs
        ),
        initial=tuple(
            Boundary(mark.state, positions[mark.weight]) for mark in declaration.initial
        ),
        final=tuple(
            Boundary(mark.state, positions[mark.weight]) for mark in declaration.final
        ),
        weight_names=slots,
    )
    algebra = MARKOV if declaration.algebra == "markov" else LOG_PROB
    identities = tuple(
        ParameterId.derive(module_name, "relation_weights", name) for name in slots
    )
    return RationalRelationSpec.of(topology, algebra, identities)


def _transform_spec(
    declaration: QiecRelationTransformDecl,
    source: RationalRelationSpec,
    other: RationalRelationSpec | None,
) -> UnaryRelationTransformSpec | BinaryRelationTransformSpec:
    if other is None:
        return UnaryRelationTransformSpec(
            source=source,
            operation=cast(
                Literal["project", "condition", "invert", "trim", "closure"],
                declaration.operation,
            ),
            positions=declaration.positions,
            word=declaration.word,
        )
    return BinaryRelationTransformSpec(
        left=source,
        right=other,
        operation=cast(
            Literal["union", "concatenate", "intersect", "join"], declaration.operation
        ),
        shared_tapes=(
            cast(tuple[int, int], declaration.positions)
            if declaration.operation == "join"
            else None
        ),
        keep_shared=declaration.keep_shared,
    )


def _operands(declaration: _Declaration) -> tuple[str, ...]:
    if isinstance(declaration, QiecRelationDecl):
        return ()
    if isinstance(declaration, QiecRelationComposeDecl):
        return declaration.left, declaration.right
    if declaration.other is None:
        return (declaration.source,)
    return declaration.source, declaration.other


def _cascade(
    root: str,
    declarations: dict[str, _Declaration],
    primitive: dict[str, RationalRelationSpec],
    dependencies: dict[str, tuple[str, ...]],
) -> RelationCascadeSpec:
    # Postorder following ordered operands makes the reachable wire independent
    # of unrelated declarations and of source order for forward references.
    pending = [(root, False)]
    indices: dict[str, int] = {}
    steps = []
    while pending:
        name, expanded = pending.pop()
        if name in indices:
            continue
        declaration = declarations[name]
        if not expanded:
            pending.append((name, True))
            pending.extend((item, False) for item in reversed(dependencies[name]))
            continue
        indices[name] = len(steps)
        operands = tuple(indices[item] for item in dependencies[name])
        if isinstance(declaration, QiecRelationDecl):
            step = RelationCascadeStep(name=name, primitive=primitive[name])
        elif isinstance(declaration, QiecRelationComposeDecl):
            step = RelationCascadeStep(
                name=name, operation="compose", operands=operands
            )
        else:
            step = RelationCascadeStep(
                name=name,
                operation=declaration.operation,
                operands=operands,
                positions=declaration.positions,
                word=declaration.word,
                keep_shared=declaration.keep_shared,
            )
        steps.append(step)
    return RelationCascadeSpec(steps=tuple(steps))


def authored_relation_specs(
    syntax: Module, module_name: str, file_path: str
) -> dict[str, AuthoredRelationSpec]:
    """Build checked primitive and derived relation specifications from source."""
    objects = {
        name: statement
        for statement in syntax.statements
        if isinstance(statement, ObjectDecl)
        for name in statement.names
    }

    def diagnostic(declaration: _Declaration, message: str) -> QiecDiagnosticError:
        return QiecDiagnosticError(
            message,
            code="qiec-relation-declaration",
            file=file_path,
            line=declaration.line,
            column=declaration.col,
        )

    declarations: dict[str, _Declaration] = {}
    primitive: dict[str, RationalRelationSpec] = {}
    for declaration in syntax.statements:
        if isinstance(
            declaration,
            QiecRelationDecl | QiecRelationComposeDecl | QiecRelationTransformDecl,
        ):
            if declaration.name in declarations:
                raise diagnostic(
                    declaration, f"duplicate authored relation {declaration.name!r}"
                )
            declarations[declaration.name] = declaration
        if isinstance(declaration, QiecRelationDecl):
            try:
                primitive[declaration.name] = _specification(
                    declaration, objects, module_name
                )
            except (TypeError, ValueError, dx.ValidationError) as error:
                raise diagnostic(declaration, str(error)) from error

    dependencies = {
        name: _operands(declaration) for name, declaration in declarations.items()
    }
    for name, operands in dependencies.items():
        for operand in operands:
            if operand not in declarations:
                raise diagnostic(
                    declarations[name], f"unknown authored relation {operand!r}"
                )
    try:
        order = tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        names = tuple(cast(list[str], error.args[1]))
        raise diagnostic(
            declarations[names[0]], f"cyclic authored relation: {' -> '.join(names)}"
        ) from error

    result: dict[str, AuthoredRelationSpec] = {}
    for name in order:
        declaration = declarations[name]
        if isinstance(declaration, QiecRelationDecl):
            result[name] = primitive[name]
            continue
        operands = dependencies[name]
        try:
            if any(operand not in primitive for operand in operands):
                result[name] = _cascade(name, declarations, primitive, dependencies)
                continue
            source = primitive[operands[0]]
            if isinstance(declaration, QiecRelationComposeDecl):
                result[name] = ComposedRelationSpec(
                    left=source, right=primitive[operands[1]]
                )
            else:
                other = primitive[operands[1]] if len(operands) == 2 else None
                result[name] = _transform_spec(declaration, source, other)
        except (TypeError, ValueError, dx.ValidationError) as error:
            raise diagnostic(declaration, str(error)) from error
    return {name: result[name] for name in declarations}
