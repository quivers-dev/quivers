"""Specialize inferred objects and validate indices from checked QIEC types."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import didactic.api as dx
import torch

from quivers.dsl.ast_nodes import DiscreteConstructor, Module, ObjectDecl, TypeFromExpr
from quivers.dsl.compiler._prelude import CompileError
from quivers.qiec import (
    Call,
    Gather,
    IndexTerm,
    IndexVariable,
    ProgramEntry,
    QiecModule,
    Var,
    tensor_shape,
)


def _shape(value: object, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape)
    if isinstance(value, str):
        return ()
    if isinstance(value, Sequence):
        children = [_shape(item, name) for item in value]
        if children and any(child != children[0] for child in children):
            raise ValueError(
                f"datum {name!r} is ragged; provide a rectangular tensor or an explicitly structured type"
            )
        return (len(value), *(children[0] if children else ()))
    return ()


def _walk(value: object) -> Iterator[dx.Model]:
    pending = [value]
    seen = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        if isinstance(node, dx.Model):
            seen.add(id(node))
            yield node
            pending.extend(getattr(node, name) for name in type(node).__field_specs__)
        elif isinstance(node, tuple):
            pending.extend(node)


def external_index_bounds(
    module: QiecModule, entry: ProgramEntry
) -> dict[str, tuple[IndexTerm, ...]]:
    """Read outer-axis bounds of external gather indices from checked values."""
    external = {
        p.name
        for p in entry.parameters
        if p.role in {"data", "index_map", "observation"}
    }
    computations = {c.id: c for c in module.computations}
    pending = [entry.computation]
    seen = set()
    bounds: dict[str, list[IndexTerm]] = {}
    while pending:
        identity = pending.pop()
        if identity in seen or identity not in computations:
            continue
        seen.add(identity)
        for node in _walk(computations[identity].body):
            if isinstance(node, Call):
                pending.append(node.callee)
            if not isinstance(node, Gather):
                continue
            value = node.value
            type_ = (
                value.local.type
                if isinstance(value, Var)
                else getattr(value, "result_type", None)
            )
            shape = tensor_shape(type_) if type_ is not None else None
            if shape is None or not shape[1]:
                continue
            index = node.index
            while isinstance(index, Gather):
                index = index.value
            if isinstance(index, Var) and index.local.name in external:
                bounds.setdefault(index.local.name, []).append(shape[1][0])
    return {name: tuple(values) for name, values in bounds.items()}


def _inferred_objects(module: Module) -> dict[str, ObjectDecl]:
    return {
        name: statement
        for statement in module.statements
        if isinstance(statement, ObjectDecl)
        and isinstance(statement.init, TypeFromExpr)
        and isinstance(statement.init.expr, DiscreteConstructor)
        and statement.init.expr.args == ("_",)
        for name in statement.names
    }


def has_inferred_objects(module: Module) -> bool:
    """Whether binding must determine at least one finite object size."""
    return bool(_inferred_objects(module))


def infer_data_objects(module: Module, data: Mapping[str, object]) -> Module:
    """Replace ``FinSet _`` when the checked inputs provide consistent evidence."""
    inferred = _inferred_objects(module)
    if not inferred:
        return module
    from quivers.dsl.qiec_lowering import lower_qvr_to_qiec

    checked = lower_qvr_to_qiec(module)
    evidence: dict[str, tuple[int, str]] = {}
    lower_bounds: dict[str, tuple[int, str]] = {}
    for entry in checked.entries:
        for parameter in entry.parameters:
            if parameter.name not in data:
                continue
            shape = tensor_shape(parameter.type)
            supplied = _shape(data[parameter.name], parameter.name)
            if shape is None or len(shape[1]) != len(supplied):
                continue
            for dimension, size in zip(shape[1], supplied, strict=True):
                if (
                    not isinstance(dimension, IndexVariable)
                    or dimension.name not in inferred
                ):
                    continue
                name = dimension.name
                prior = evidence.setdefault(name, (size, parameter.name))
                if prior[0] != size:
                    declaration = inferred[name]
                    raise CompileError(
                        f"cannot infer {name!r}: data {prior[1]!r} requires {prior[0]}, but {parameter.name!r} requires {size}",
                        declaration.line,
                        declaration.col,
                    )
        for name, dimensions in external_index_bounds(checked, entry).items():
            if name not in data:
                continue
            relevant = [
                d.name
                for d in dimensions
                if isinstance(d, IndexVariable) and d.name in inferred
            ]
            if not relevant:
                continue
            value = torch.as_tensor(data[name])
            if (
                value.is_floating_point()
                or value.is_complex()
                or value.dtype == torch.bool
            ):
                raise TypeError(f"index datum {name!r} must contain integers")
            if value.numel() == 0:
                continue
            if bool((value < 0).any()):
                raise ValueError(f"index datum {name!r} contains a negative index")
            for axis in relevant:
                bound = (int(value.max()) + 1, name)
                if axis not in lower_bounds or lower_bounds[axis][0] < bound[0]:
                    lower_bounds[axis] = bound
    for axis, (minimum, source) in lower_bounds.items():
        if axis in evidence and evidence[axis][0] < minimum:
            declaration = inferred[axis]
            raise CompileError(
                f"inferred size of {axis!r} is {evidence[axis][0]}, but indices {source!r} require at least {minimum}",
                declaration.line,
                declaration.col,
            )
        evidence.setdefault(axis, (minimum, source))
    if missing := sorted(set(inferred) - set(evidence)):
        name = missing[0]
        declaration = inferred[name]
        raise CompileError(
            f"cannot infer the size of {name!r}; supply data whose checked shape or index range determines it, or replace `FinSet _` with an explicit size",
            declaration.line,
            declaration.col,
        )
    statements = []
    for statement in module.statements:
        if isinstance(statement, ObjectDecl) and any(
            name in inferred for name in statement.names
        ):
            for name in statement.names:
                statements.append(
                    statement.with_(
                        names=(name,),
                        init=TypeFromExpr(
                            expr=DiscreteConstructor(
                                constructor="FinSet", args=(str(evidence[name][0]),)
                            )
                        ),
                    )
                )
        else:
            statements.append(statement)
    return module.with_(statements=tuple(statements))


__all__ = []
