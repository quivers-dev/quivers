"""Elaborate checked graph topology declarations from named relation specs."""

from __future__ import annotations

from collections.abc import Mapping

import didactic.api as dx

from quivers.dsl._relation_specs import (
    AuthoredRelationSpec,
    _alphabet,
    _cascade,
    _operands,
)
from quivers.dsl.ast_nodes import (
    Module,
    QiecRelationComposeDecl,
    QiecRelationDecl,
    QiecRelationTransformDecl,
    QiecStringGraphDecl,
)
from quivers.dsl.ast_nodes.declarations import ObjectDecl
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec.relation_cascades import RelationCascadeSpec
from quivers.qiec.relations import RationalRelationSpec
from quivers.qiec.string_graphs import StringFactorGraphSpec


def authored_string_graph_specs(
    syntax: Module,
    relations: Mapping[str, AuthoredRelationSpec],
    file_path: str,
) -> dict[str, StringFactorGraphSpec]:
    """Build checked graph specifications in source declaration order."""
    objects = {
        name: statement
        for statement in syntax.statements
        if isinstance(statement, ObjectDecl)
        for name in statement.names
    }
    declarations = {
        statement.name: statement
        for statement in syntax.statements
        if isinstance(
            statement,
            QiecRelationDecl | QiecRelationComposeDecl | QiecRelationTransformDecl,
        )
    }
    operands = {name: _operands(statement) for name, statement in declarations.items()}
    primitive = {
        name: relation
        for name, relation in relations.items()
        if isinstance(relation, RationalRelationSpec)
    }
    graphs: dict[str, StringFactorGraphSpec] = {}
    for declaration in syntax.statements:
        if not isinstance(declaration, QiecStringGraphDecl):
            continue
        try:
            if declaration.name in graphs:
                raise ValueError(
                    f"duplicate authored string graph {declaration.name!r}"
                )
            variables = []
            for variable in declaration.variables:
                alphabet = _alphabet(variable.alphabet, objects)
                variables.append(
                    (
                        variable.name,
                        alphabet.name,
                        alphabet.cardinality,
                        variable.observed,
                    )
                )
            used: dict[str, RationalRelationSpec | RelationCascadeSpec] = {}
            for factor in declaration.factors:
                if factor.relation in used:
                    continue
                relation = relations.get(factor.relation)
                if relation is None:
                    raise ValueError(
                        f"unknown authored factor relation {factor.relation!r}"
                    )
                if not isinstance(relation, RationalRelationSpec | RelationCascadeSpec):
                    relation = _cascade(
                        factor.relation, declarations, primitive, operands
                    )
                used[factor.relation] = relation
            graphs[declaration.name] = StringFactorGraphSpec(
                variables=tuple(variables),
                relations=tuple(used.items()),
                factors=tuple(
                    (factor.relation, factor.scope) for factor in declaration.factors
                ),
            )
        except (TypeError, ValueError, dx.ValidationError) as error:
            raise QiecDiagnosticError(
                str(error),
                code="qiec-string-graph-declaration",
                file=file_path,
                line=declaration.line,
                column=declaration.col,
            ) from error
    return graphs
