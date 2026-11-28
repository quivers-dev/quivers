"""Elaborate authored sparse topology without binding runtime weights."""

from __future__ import annotations

import didactic.api as dx

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.objects import FinSet, StringSet
from quivers.core.relations import Arc, Boundary, RelationTopology
from quivers.dsl.ast_nodes import Module, QiecRelationComposeDecl, QiecRelationDecl
from quivers.dsl.ast_nodes.declarations import ObjectDecl, TypeFromExpr
from quivers.dsl.ast_nodes.objects import DiscreteConstructor, TypeName
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec.contracts import ParameterId
from quivers.qiec.relation_composition import ComposedRelationSpec
from quivers.qiec.relations import RationalRelationSpec


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


def authored_relation_specs(
    syntax: Module, module_name: str, file_path: str
) -> dict[str, RationalRelationSpec | ComposedRelationSpec]:
    """Build checked primitive and binary relation specifications from source."""
    objects = {
        name: statement
        for statement in syntax.statements
        if isinstance(statement, ObjectDecl)
        for name in statement.names
    }

    def diagnostic(
        declaration: QiecRelationDecl | QiecRelationComposeDecl, message: str
    ) -> QiecDiagnosticError:
        return QiecDiagnosticError(
            message,
            code="qiec-relation-declaration",
            file=file_path,
            line=declaration.line,
            column=declaration.col,
        )

    declarations: dict[str, QiecRelationDecl | QiecRelationComposeDecl] = {}
    primitive: dict[str, RationalRelationSpec] = {}
    for declaration in syntax.statements:
        if isinstance(declaration, QiecRelationDecl | QiecRelationComposeDecl):
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

    result: dict[str, RationalRelationSpec | ComposedRelationSpec] = {}
    for name, declaration in declarations.items():
        if isinstance(declaration, QiecRelationDecl):
            result[name] = primitive[name]
            continue
        for operand in (declaration.left, declaration.right):
            if operand not in declarations:
                raise diagnostic(declaration, f"unknown authored relation {operand!r}")
            if operand not in primitive:
                raise diagnostic(
                    declaration,
                    f"composition requires primitive relation operands, got {operand!r}",
                )
        try:
            result[name] = ComposedRelationSpec(
                left=primitive[declaration.left], right=primitive[declaration.right]
            )
        except (TypeError, ValueError, dx.ValidationError) as error:
            raise diagnostic(declaration, str(error)) from error
    return result
