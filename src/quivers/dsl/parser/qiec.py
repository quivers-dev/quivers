"""Tree walkers for the typed QIEC surface declarations and terms."""

from __future__ import annotations

from typing import Literal, cast

from quivers.dsl.ast_nodes.qiec import (
    QiecBindComputation,
    QiecBinder,
    QiecCaseBranch,
    QiecCaseComputation,
    QiecIfComputation,
    QiecCaseMotive,
    QiecComputation,
    QiecComputationDecl,
    QiecInterpretationClause,
    QiecInterpretationDecl,
    QiecRelationScoreDecl,
    QiecConstructorValue,
    QiecContextSort,
    QiecEffectBinder,
    QiecEffectDecl,
    QiecEffectInstanceDecl,
    QiecEffectRef,
    QiecEffectRequest,
    QiecEffectRow,
    QiecFamilyConstructor,
    QiecFamilyDecl,
    QiecFunctionType,
    QiecHandleComputation,
    QiecHandlerApplication,
    QiecCallComputation,
    QiecHandlerClause,
    QiecHandlerOperationClause,
    QiecHandlerReturnClause,
    QiecInstanceComputation,
    QiecPureBinding,
    QiecResumeComputation,
    QiecHandlerDecl,
    QiecIndexApplication,
    QiecIndexBinder,
    QiecIndexConstructor,
    QiecIndexDecl,
    QiecIndexExpr,
    QiecIndexLiteral,
    QiecIndexName,
    QiecIndexSort,
    QiecLocalBinding,
    QiecNatSort,
    QiecOperationDecl,
    QiecPerformComputation,
    QiecProductType,
    QiecResumptionGrade,
    QiecReturnComputation,
    QiecRowEntry,
    QiecSequenceComputation,
    QiecShapeIndex,
    QiecShapeSort,
    QiecStaticArgument,
    QiecTypeApplication,
    QiecTypeBinder,
    QiecTypeExpr,
    QiecTypeKind,
    QiecTypeName,
    QiecUserIndexSort,
    QiecValue,
    QiecValueParameter,
)
from quivers.dsl.parser._helpers import _field_text, _required_field
from quivers.dsl.parser.expressions import _walk_let_arith
from quivers.dsl.parser._registry import ParseError, _Tree


_QIEC_STATEMENT_KINDS = frozenset(
    {
        "index_decl",
        "indexed_family_decl",
        "effect_decl",
        "effect_instance_decl",
        "handler_decl",
        "computation_decl",
        "interpretation_decl",
        "relation_score_decl",
    }
)


def _walk_qiec_statement(t: _Tree, vid: str):
    """Walk one QIEC declaration selected by the statement dispatcher."""
    kind = t.kind(vid)
    line, col = t.line_col(vid)
    docs = _docs(t, vid)
    if kind == "index_decl":
        return QiecIndexDecl(
            name=_field_text(t, vid, "name"),
            constructors=tuple(
                _walk_index_constructor(t, child)
                for child in t.fields(vid, "constructors")
            ),
            docs=docs,
            line=line,
            col=col,
        )
    if kind == "indexed_family_decl":
        parameters = _walk_telescope_field(t, vid, "parameters")
        indices = _walk_index_telescope_field(t, vid, "indices")
        return QiecFamilyDecl(
            name=_field_text(t, vid, "name"),
            parameters=parameters,
            indices=indices,
            result_kind=_walk_kind(t, _required_field(t, vid, "kind")),
            constructors=tuple(
                _walk_family_constructor(t, child)
                for child in t.fields(vid, "constructors")
            ),
            docs=docs,
            line=line,
            col=col,
        )
    if kind == "effect_decl":
        return QiecEffectDecl(
            name=_field_text(t, vid, "name"),
            binders=_walk_telescope_field(t, vid, "binders"),
            operations=tuple(
                _walk_operation(t, child) for child in t.fields(vid, "operations")
            ),
            docs=docs,
            line=line,
            col=col,
        )
    if kind == "effect_instance_decl":
        return QiecEffectInstanceDecl(
            name=_field_text(t, vid, "name"),
            effect=_walk_effect_ref(t, _required_field(t, vid, "effect")),
            docs=docs,
            line=line,
            col=col,
        )
    if kind == "handler_decl":
        (
            coverage,
            forwards,
            introduced,
            implementation,
            duplicate_options,
        ) = _walk_handler_options(t, t.field(vid, "options"))
        return QiecHandlerDecl(
            name=_field_text(t, vid, "name"),
            binders=_walk_telescope_field(t, vid, "binders"),
            effect=_walk_effect_ref(t, _required_field(t, vid, "effect")),
            input_type=_walk_type(t, _required_field(t, vid, "input")),
            output_type=_walk_type(t, _required_field(t, vid, "output")),
            introduced=introduced,
            coverage=coverage,
            forwards_unknown=forwards,
            implementation=implementation,
            parameters=tuple(
                _walk_value_parameter(t, child) for child in t.fields(vid, "parameters")
            ),
            clauses=tuple(
                _walk_handler_clause(t, child) for child in t.fields(vid, "clauses")
            ),
            duplicate_options=duplicate_options,
            docs=docs,
            line=line,
            col=col,
        )
    if kind == "interpretation_decl":
        clauses = []
        for child in t.fields(vid, "clauses"):
            clause_line, clause_col = t.line_col(child)
            clauses.append(
                QiecInterpretationClause(
                    constructor_kind=cast(
                        Literal["rule", "axiom"],
                        _field_text(t, child, "constructor_kind"),
                    ),
                    constructor=_field_text(t, child, "constructor"),
                    operation=_field_text(t, child, "operation"),
                    line=clause_line,
                    col=clause_col,
                )
            )
        return QiecInterpretationDecl(
            name=_field_text(t, vid, "name"),
            deduction=_field_text(t, vid, "deduction"),
            carrier=_walk_type(t, _required_field(t, vid, "carrier")),
            clauses=tuple(clauses),
            docs=docs,
            line=line,
            col=col,
        )
    if kind == "relation_score_decl":
        return QiecRelationScoreDecl(
            name=_field_text(t, vid, "name"),
            result_type=_walk_type(t, _required_field(t, vid, "result")),
            operation=cast(
                Literal["pair_weight", "conditional_log_prob"],
                _field_text(t, vid, "operation"),
            ),
            docs=docs,
            line=line,
            col=col,
        )
    if kind == "computation_decl":
        return QiecComputationDecl(
            name=_field_text(t, vid, "name"),
            binders=_walk_telescope_field(t, vid, "binders"),
            parameters=tuple(
                _walk_value_parameter(t, child) for child in t.fields(vid, "parameters")
            ),
            result_type=_walk_type(t, _required_field(t, vid, "result")),
            effects=_walk_effect_row(t, _required_field(t, vid, "row")),
            body=_walk_computation(t, _required_field(t, vid, "body")),
            docs=docs,
            line=line,
            col=col,
        )
    raise ParseError(f"unexpected QIEC statement kind {kind!r} at {vid}")


def _docs(t: _Tree, vid: str) -> tuple[str, ...]:
    group = t.field(vid, "docs")
    if group is None:
        return ()
    result: list[str] = []
    for child in t.positional(group):
        if t.kind(child) != "doc_comment":
            continue
        text = t.text(child)
        result.append((text[2:] if text.startswith("#!") else text).strip())
    return tuple(result)


def _walk_kind(t: _Tree, vid: str):
    line, col = t.line_col(vid)
    kind = t.kind(vid)
    if kind == "qiec_type_kind":
        return QiecTypeKind(line=line, col=col)
    raise ParseError(f"unexpected QIEC kind {kind!r} at {vid}")


def _walk_index_sort(t: _Tree, vid: str) -> QiecIndexSort:
    if t.kind(vid) == "qiec_index_sort":
        children = t.positional(vid)
        if len(children) != 1:
            raise ParseError(f"malformed QIEC index sort at {vid}")
        return _walk_index_sort(t, children[0])
    line, col = t.line_col(vid)
    kind = t.kind(vid)
    if kind == "qiec_nat_sort":
        return QiecNatSort(line=line, col=col)
    if kind == "qiec_shape_sort":
        rank_vid = t.field(vid, "rank")
        return QiecShapeSort(
            rank=int(t.text(rank_vid)) if rank_vid is not None else None,
            line=line,
            col=col,
        )
    if kind == "qiec_context_sort":
        return QiecContextSort(
            signature=_field_text(t, vid, "signature"), line=line, col=col
        )
    if kind == "qiec_user_index_sort":
        return QiecUserIndexSort(name=_field_text(t, vid, "name"), line=line, col=col)
    raise ParseError(f"unexpected QIEC index sort {kind!r} at {vid}")


def _walk_index(t: _Tree, vid: str) -> QiecIndexExpr:
    line, col = t.line_col(vid)
    kind = t.kind(vid)
    if kind == "qiec_index_name":
        return QiecIndexName(name=_field_text(t, vid, "name"), line=line, col=col)
    if kind == "qiec_index_literal":
        return QiecIndexLiteral(
            value=int(_field_text(t, vid, "value")), line=line, col=col
        )
    if kind == "qiec_index_application":
        return QiecIndexApplication(
            constructor=_field_text(t, vid, "constructor"),
            arguments=tuple(
                _walk_index(t, child) for child in t.fields(vid, "arguments")
            ),
            line=line,
            col=col,
        )
    if kind == "qiec_shape_index":
        return QiecShapeIndex(
            dimensions=tuple(
                _walk_index(t, child) for child in t.fields(vid, "dimensions")
            ),
            line=line,
            col=col,
        )
    raise ParseError(f"unexpected QIEC index term {kind!r} at {vid}")


def _walk_binder(t: _Tree, vid: str) -> QiecBinder:
    line, col = t.line_col(vid)
    kind = t.kind(vid)
    name = _field_text(t, vid, "name")
    if kind == "qiec_type_binder":
        return QiecTypeBinder(
            name=name,
            binder_kind=_walk_kind(t, _required_field(t, vid, "kind")),
            line=line,
            col=col,
        )
    if kind == "qiec_index_binder":
        return QiecIndexBinder(
            name=name,
            sort=_walk_index_sort(t, _required_field(t, vid, "sort")),
            line=line,
            col=col,
        )
    if kind == "qiec_effect_binder":
        return QiecEffectBinder(name=name, line=line, col=col)
    raise ParseError(f"unexpected QIEC binder {kind!r} at {vid}")


def _walk_telescope_field(t: _Tree, parent: str, field: str) -> tuple[QiecBinder, ...]:
    telescope = t.field(parent, field)
    if telescope is None:
        return ()
    return tuple(_walk_binder(t, child) for child in t.fields(telescope, "binders"))


def _walk_index_telescope_field(
    t: _Tree, parent: str, field: str
) -> tuple[QiecIndexBinder, ...]:
    telescope = t.field(parent, field)
    if telescope is None:
        return ()
    return tuple(
        cast(QiecIndexBinder, _walk_binder(t, child))
        for child in t.fields(telescope, "binders")
    )


def _walk_static_arguments(
    t: _Tree, parent: str, field: str
) -> tuple[QiecStaticArgument, ...]:
    """Walk the static arguments under a field of a node.

    Parameters
    ----------
    t : _Tree
        The parse tree.
    parent : str
        The node holding the arguments.
    field : str
        The field they sit under.

    Returns
    -------
    tuple[QiecStaticArgument, ...]
        The arguments in order: type syntax, or an index literal.

    Raises
    ------
    ParseError
        If a child under the field is not a static argument wrapper.
    """
    result: list[QiecStaticArgument] = []
    for wrapper in t.fields(parent, field):
        if t.kind(wrapper) != "qiec_static_argument":
            raise ParseError(f"unexpected static argument wrapper at {wrapper}")
        value = _required_field(t, wrapper, "value")
        if t.kind(value) == "qiec_index_literal":
            result.append(cast(QiecIndexLiteral, _walk_index(t, value)))
        else:
            result.append(_walk_type(t, value))
    return tuple(result)


def _walk_type(t: _Tree, vid: str) -> QiecTypeExpr:
    line, col = t.line_col(vid)
    kind = t.kind(vid)
    if kind == "qiec_type_paren":
        return _walk_type(t, _required_field(t, vid, "type"))
    if kind == "qiec_type_name":
        return QiecTypeName(name=_field_text(t, vid, "name"), line=line, col=col)
    if kind == "qiec_type_application":
        return QiecTypeApplication(
            constructor=_field_text(t, vid, "constructor"),
            static_arguments=_walk_static_arguments(t, vid, "arguments"),
            indices=tuple(_walk_index(t, child) for child in t.fields(vid, "indices")),
            line=line,
            col=col,
        )
    if kind == "qiec_product_type":
        left = _walk_type(t, _required_field(t, vid, "left"))
        right = _walk_type(t, _required_field(t, vid, "right"))
        components: list[QiecTypeExpr] = []
        components.extend(
            left.components if isinstance(left, QiecProductType) else (left,)
        )
        components.extend(
            right.components if isinstance(right, QiecProductType) else (right,)
        )
        return QiecProductType(components=tuple(components), line=line, col=col)
    if kind == "qiec_function_type":
        return QiecFunctionType(
            parameter=_walk_type(t, _required_field(t, vid, "parameter")),
            result=_walk_type(t, _required_field(t, vid, "result")),
            line=line,
            col=col,
        )
    raise ParseError(f"unexpected QIEC type expression {kind!r} at {vid}")


def _walk_effect_ref(t: _Tree, vid: str) -> QiecEffectRef:
    line, col = t.line_col(vid)
    return QiecEffectRef(
        name=_field_text(t, vid, "name"),
        arguments=_walk_static_arguments(t, vid, "arguments"),
        line=line,
        col=col,
    )


def _walk_effect_row(t: _Tree, vid: str) -> QiecEffectRow:
    if t.kind(vid) == "qiec_effect_row":
        return _walk_effect_row(t, _required_field(t, vid, "row"))
    if t.kind(vid) != "qiec_effect_row_literal":
        raise ParseError(f"unexpected QIEC effect row at {vid}")
    line, col = t.line_col(vid)
    entries = tuple(
        QiecRowEntry(
            instance=_field_text(t, child, "name"),
            line=t.line_col(child)[0],
            col=t.line_col(child)[1],
        )
        for child in t.fields(vid, "entries")
    )
    tail_vid = t.field(vid, "tail")
    return QiecEffectRow(
        entries=entries,
        tail=t.text(tail_vid) if tail_vid is not None else None,
        lacks=tuple(t.text(child) for child in t.fields(vid, "lacks")),
        line=line,
        col=col,
    )


def _walk_index_constructor(t: _Tree, vid: str) -> QiecIndexConstructor:
    line, col = t.line_col(vid)
    return QiecIndexConstructor(
        name=_field_text(t, vid, "name"),
        arguments=tuple(
            _walk_index_sort(t, child) for child in t.fields(vid, "arguments")
        ),
        line=line,
        col=col,
    )


def _walk_family_constructor(t: _Tree, vid: str) -> QiecFamilyConstructor:
    line, col = t.line_col(vid)
    return QiecFamilyConstructor(
        name=_field_text(t, vid, "name"),
        binders=_walk_telescope_field(t, vid, "binders"),
        arguments=tuple(_walk_type(t, child) for child in t.fields(vid, "arguments")),
        result=_walk_type(t, _required_field(t, vid, "result")),
        line=line,
        col=col,
    )


def _walk_operation(t: _Tree, vid: str) -> QiecOperationDecl:
    line, col = t.line_col(vid)
    return QiecOperationDecl(
        name=_field_text(t, vid, "name"),
        binders=_walk_telescope_field(t, vid, "binders"),
        arguments=tuple(_walk_type(t, child) for child in t.fields(vid, "arguments")),
        result=_walk_type(t, _required_field(t, vid, "result")),
        line=line,
        col=col,
    )


def _walk_handler_options(
    t: _Tree, vid: str | None
) -> tuple[str, bool, QiecEffectRow, str, tuple[str, ...]]:
    """Read a handler's bracketed option list.

    Parameters
    ----------
    t : _Tree
        The parsed tree.
    vid : str or None
        Vertex of the option list, or None when the declaration carries
        none, in which case every option takes its default.

    Returns
    -------
    tuple[str, bool, QiecEffectRow, str, tuple[str, ...]]
        Coverage, whether unknown operations are forwarded, the
        introduced row, the implementation, and the names of any options
        given more than once. Duplicates are reported rather than
        rejected here, so the caller can raise one diagnostic carrying
        the declaration's position.
    """
    coverage = "total"
    forwards_unknown = False
    introduced = QiecEffectRow()
    implementation = "authored"
    if vid is None:
        return coverage, forwards_unknown, introduced, implementation, ()
    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in t.fields(vid, "entries"):
        constants = t.consts(entry)
        if value := constants.get("field:coverage"):
            if "coverage" in seen and "coverage" not in duplicates:
                duplicates.append("coverage")
            seen.add("coverage")
            coverage = value
        if value := constants.get("field:forwards"):
            if "forwards" in seen and "forwards" not in duplicates:
                duplicates.append("forwards")
            seen.add("forwards")
            forwards_unknown = value == "unknown"
        introduced_vid = t.field(entry, "introduced")
        if introduced_vid is not None:
            if "introduces" in seen and "introduces" not in duplicates:
                duplicates.append("introduces")
            seen.add("introduces")
            introduced = _walk_effect_row(t, introduced_vid)
        if value := constants.get("field:implementation"):
            if "implementation" in seen and "implementation" not in duplicates:
                duplicates.append("implementation")
            seen.add("implementation")
            implementation = value
    return coverage, forwards_unknown, introduced, implementation, tuple(duplicates)


def _walk_handler_clause(t: _Tree, vid: str) -> QiecHandlerClause:
    """Walk one handler clause, of either shape.

    Parameters
    ----------
    t : _Tree
        The parsed tree.
    vid : str
        Vertex of the clause.

    Returns
    -------
    QiecHandlerClause
        A return clause or an operation clause. An operation clause
        without a body is a signature, which the checker admits only for
        a handler declared `implementation=foreign`.

    Raises
    ------
    ParseError
        If the clause is of an unknown kind, or names a resumption grade
        outside the four the grammar allows.
    """
    line, col = t.line_col(vid)
    kind = t.kind(vid)
    if kind == "qiec_handler_return_clause":
        return QiecHandlerReturnClause(
            binder=_walk_local(t, _required_field(t, vid, "binder")),
            body=_walk_computation(t, _required_field(t, vid, "body")),
            line=line,
            col=col,
        )
    if kind != "qiec_handler_operation_clause":
        raise ParseError(f"unexpected QIEC handler clause {kind!r} at {vid}")
    grade = _field_text(t, vid, "grade")
    if grade not in {"0", "aff", "1", "omega"}:
        raise ParseError(f"unknown resumption grade {grade!r} at {vid}")
    body_vid = t.field(vid, "body")
    return QiecHandlerOperationClause(
        operation=_field_text(t, vid, "operation"),
        grade=cast(QiecResumptionGrade, grade),
        binders=_walk_telescope_field(t, vid, "binders"),
        parameters=tuple(
            _walk_local(t, child) for child in t.fields(vid, "parameters")
        ),
        body=None if body_vid is None else _walk_computation(t, body_vid),
        line=line,
        col=col,
    )


def _walk_value_parameter(t: _Tree, vid: str) -> QiecValueParameter:
    line, col = t.line_col(vid)
    return QiecValueParameter(
        name=_field_text(t, vid, "name"),
        type_expr=_walk_type(t, _required_field(t, vid, "type")),
        line=line,
        col=col,
    )


def _walk_local(t: _Tree, vid: str) -> QiecLocalBinding:
    line, col = t.line_col(vid)
    type_vid = t.field(vid, "type")
    return QiecLocalBinding(
        name=_field_text(t, vid, "name"),
        type_expr=_walk_type(t, type_vid) if type_vid is not None else None,
        line=line,
        col=col,
    )


def _walk_value(t: _Tree, vid: str) -> QiecValue:
    """Walk a QIEC value: a constructor application or a pure expression.

    Parameters
    ----------
    t : _Tree
        The parse tree.
    vid : str
        The value vertex.

    Returns
    -------
    QiecValue
        The constructor value, or the shared expression node the pure
        expression walker produces.

    Raises
    ------
    ParseError
        If the vertex is not a value form.
    """
    line, col = t.line_col(vid)
    kind = t.kind(vid)
    if kind == "qiec_constructor_value":
        return QiecConstructorValue(
            constructor=_field_text(t, vid, "constructor"),
            static_arguments=_walk_static_arguments(t, vid, "static_arguments"),
            fields=tuple(_walk_value(t, child) for child in t.fields(vid, "fields")),
            result_type=_walk_type(t, _required_field(t, vid, "result")),
            line=line,
            col=col,
        )
    if kind.startswith("let_") or kind in (
        "integer",
        "float",
        "signed_number",
        "string",
    ):
        return _walk_let_arith(t, vid)
    raise ParseError(f"unexpected QIEC value {kind!r} at {vid}")


def _walk_request(t: _Tree, vid: str) -> QiecEffectRequest:
    line, col = t.line_col(vid)
    return QiecEffectRequest(
        instance=_field_text(t, vid, "instance"),
        operation=_field_text(t, vid, "operation"),
        static_arguments=_walk_static_arguments(t, vid, "static_arguments"),
        arguments=tuple(_walk_value(t, child) for child in t.fields(vid, "arguments")),
        line=line,
        col=col,
    )


def _walk_computation(t: _Tree, vid: str) -> QiecComputation:
    line, col = t.line_col(vid)
    kind = t.kind(vid)
    if kind == "qiec_return_computation":
        return QiecReturnComputation(
            value=_walk_value(t, _required_field(t, vid, "value")),
            line=line,
            col=col,
        )
    if kind == "qiec_perform_computation":
        return QiecPerformComputation(
            request=_walk_request(t, _required_field(t, vid, "request")),
            line=line,
            col=col,
        )
    if kind == "qiec_bind_computation":
        return QiecBindComputation(
            binder=_walk_local(t, _required_field(t, vid, "binder")),
            first=_walk_computation(t, _required_field(t, vid, "first")),
            then=_walk_computation(t, _required_field(t, vid, "then")),
            line=line,
            col=col,
        )
    if kind == "qiec_sequence_computation":
        return QiecSequenceComputation(
            first=_walk_computation(t, _required_field(t, vid, "first")),
            then=_walk_computation(t, _required_field(t, vid, "then")),
            line=line,
            col=col,
        )
    if kind == "qiec_handle_computation":
        handler_vid = _required_field(t, vid, "handler")
        handler_line, handler_col = t.line_col(handler_vid)
        handler = QiecHandlerApplication(
            name=_field_text(t, handler_vid, "name"),
            static_arguments=_walk_static_arguments(t, handler_vid, "arguments"),
            values=tuple(
                _walk_value(t, child) for child in t.fields(handler_vid, "values")
            ),
            line=handler_line,
            col=handler_col,
        )
        return QiecHandleComputation(
            instance=_field_text(t, vid, "instance"),
            handler=handler,
            body=_walk_computation(t, _required_field(t, vid, "body")),
            line=line,
            col=col,
        )
    if kind == "qiec_case_computation":
        return QiecCaseComputation(
            scrutinee=_walk_value(t, _required_field(t, vid, "scrutinee")),
            motive=_walk_case_motive(t, _required_field(t, vid, "motive")),
            branches=tuple(
                _walk_case_branch(t, child) for child in t.fields(vid, "branches")
            ),
            line=line,
            col=col,
        )
    if kind == "qiec_if_computation":
        return QiecIfComputation(
            condition=_walk_value(t, _required_field(t, vid, "condition")),
            then=_walk_computation(t, _required_field(t, vid, "then")),
            otherwise=_walk_computation(t, _required_field(t, vid, "otherwise")),
            line=line,
            col=col,
        )
    if kind == "qiec_pure_binding":
        return QiecPureBinding(
            binder=_walk_local(t, _required_field(t, vid, "binder")),
            value=_walk_value(t, _required_field(t, vid, "value")),
            then=_walk_computation(t, _required_field(t, vid, "then")),
            line=line,
            col=col,
        )
    if kind == "qiec_call_computation":
        return QiecCallComputation(
            callee=_field_text(t, vid, "callee"),
            static_arguments=_walk_static_arguments(t, vid, "static_arguments"),
            arguments=tuple(
                _walk_value(t, child) for child in t.fields(vid, "arguments")
            ),
            line=line,
            col=col,
        )
    if kind == "qiec_resume_computation":
        value_vid = t.field(vid, "value")
        return QiecResumeComputation(
            value=None if value_vid is None else _walk_value(t, value_vid),
            parameters=tuple(
                _walk_value(t, child) for child in t.fields(vid, "parameters")
            ),
            line=line,
            col=col,
        )
    if kind == "qiec_instance_computation":
        return QiecInstanceComputation(
            name=_field_text(t, vid, "name"),
            effect=_walk_effect_ref(t, _required_field(t, vid, "effect")),
            body=_walk_computation(t, _required_field(t, vid, "body")),
            line=line,
            col=col,
        )
    raise ParseError(f"unexpected QIEC computation {kind!r} at {vid}")


def _walk_case_motive(t: _Tree, vid: str) -> QiecCaseMotive:
    line, col = t.line_col(vid)
    return QiecCaseMotive(
        indices=_walk_index_telescope_field(t, vid, "indices"),
        result_type=_walk_type(t, _required_field(t, vid, "result")),
        line=line,
        col=col,
    )


def _walk_case_branch(t: _Tree, vid: str) -> QiecCaseBranch:
    line, col = t.line_col(vid)
    return QiecCaseBranch(
        constructor=_field_text(t, vid, "constructor"),
        static_arguments=tuple(
            QiecTypeName(
                name=_field_text(t, child, "name"),
                line=t.line_col(child)[0],
                col=t.line_col(child)[1],
            )
            for child in t.fields(vid, "binders")
        ),
        fields=tuple(_walk_local(t, child) for child in t.fields(vid, "fields")),
        body=_walk_computation(t, _required_field(t, vid, "body")),
        line=line,
        col=col,
    )


__all__ = []
