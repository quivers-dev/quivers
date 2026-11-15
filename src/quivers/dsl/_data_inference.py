"""Specialize inferred objects and validate indices from checked QIEC types."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from numbers import Integral

import didactic.api as dx
import torch

from quivers.dsl.ast_nodes import (
    AxisSpec,
    ContinuousConstructor,
    DiscreteConstructor,
    Module,
    ObjectDecl,
    ObjectParam,
    ProgramDecl,
    QiecIndexBinder,
    QiecIndexName,
    QiecIndexLiteral,
    QiecNatSort,
    TypeFromExpr,
    TypeName,
    CallStep,
    ExprMorphismCall,
    MarginalizeStep,
    ObserveStep,
    SampleStep,
)
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
        return _specialize_program_parameters(module, data)
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
    return _specialize_program_parameters(
        module.with_(statements=tuple(statements)), data
    )


def _specialize_program_parameters(
    module: Module, data: Mapping[str, object]
) -> Module:
    """Solve scoped finite binders from explicit arguments and checked shapes.

    Each entry has its own binder namespace. Substitution changes only syntax
    carrying static indices or object references; external values stay shared.
    """
    candidates = {
        declaration.name: {
            parameter.name: parameter
            for parameter in (*declaration.binders, *(declaration.type_params or ()))
            if isinstance(parameter, ObjectParam)
            and parameter.universe == "FinSet"
            or isinstance(parameter, QiecIndexBinder)
            and isinstance(parameter.sort, QiecNatSort)
        }
        for declaration in module.statements
        if isinstance(declaration, ProgramDecl)
    }
    candidates = {name: binders for name, binders in candidates.items() if binders}
    if not candidates or not data:
        return module
    from quivers.dsl.qiec_lowering import lower_qvr_to_qiec

    checked = lower_qvr_to_qiec(module)
    solutions: dict[str, dict[str, int]] = {}
    for entry in checked.entries:
        binders = candidates.get(entry.name, {})
        evidence: dict[str, tuple[int, str]] = {}

        def constrain(name: str, size: int, source: str) -> None:
            previous = evidence.setdefault(name, (size, source))
            if size != previous[0]:
                binder = binders[name]
                raise CompileError(
                    f"cannot infer {name!r}: {previous[1]!r} requires {previous[0]}, but {source!r} requires {size}",
                    binder.line,
                    binder.col,
                )

        for name, binder in binders.items():
            if name not in data:
                continue
            value = data[name]
            if isinstance(binder, ObjectParam) and isinstance(value, str):
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise TypeError(
                    f"static extent {name!r} requires a nonnegative integer"
                )
            constrain(name, int(value), name)
        for parameter in entry.parameters:
            if parameter.name not in data:
                continue
            shape = tensor_shape(parameter.type)
            supplied = _shape(data[parameter.name], parameter.name)
            if shape is None or len(shape[1]) != len(supplied):
                continue
            for dimension, size in zip(shape[1], supplied, strict=True):
                if isinstance(dimension, IndexVariable) and dimension.name in binders:
                    constrain(dimension.name, size, parameter.name)
        if evidence:
            solutions[entry.name] = {name: size for name, (size, _) in evidence.items()}
    if not solutions:
        return module
    # Binding an exported entry must not consume the static arguments of its
    # callers. Keep referenced declarations as generic templates under fresh
    # internal names; the original entry can then specialize independently.
    called = {
        node.morphism
        if isinstance(node, SampleStep | ObserveStep | MarginalizeStep)
        else node.callee
        if isinstance(node, ExprMorphismCall)
        else node.call.callee
        for node in _walk(module)
        if isinstance(
            node,
            SampleStep | ObserveStep | MarginalizeStep | ExprMorphismCall | CallStep,
        )
    }
    names = {
        name
        for statement in module.statements
        for name in (
            *getattr(statement, "names", ()),
            getattr(statement, "name", None),
        )
        if isinstance(name, str)
    }
    generic_names: dict[str, str] = {}
    for name in sorted(called.intersection(solutions)):
        generic = f"__qvr_template_{name}"
        while generic in names:
            generic += "_"
        names.add(generic)
        generic_names[name] = generic

    def retain_generic_calls(
        value: object, shadowed: frozenset[str] = frozenset()
    ) -> object:
        if isinstance(value, ProgramDecl):
            shadowed = frozenset(p.name for p in value.type_params or ())
        if isinstance(value, tuple):
            items = tuple(retain_generic_calls(item, shadowed) for item in value)
            return (
                value
                if all(a is b for a, b in zip(value, items, strict=True))
                else items
            )
        if not isinstance(value, dx.Model):
            return value
        changed = {}
        for field in type(value).__field_specs__:
            original = getattr(value, field)
            rewritten = retain_generic_calls(original, shadowed)
            if rewritten is not original:
                changed[field] = rewritten
        if isinstance(value, SampleStep | ObserveStep | MarginalizeStep):
            if value.morphism not in shadowed:
                if value.morphism in generic_names:
                    changed["morphism"] = generic_names[value.morphism]
        elif isinstance(value, ExprMorphismCall):
            if value.callee not in shadowed:
                if value.callee in generic_names:
                    changed["callee"] = generic_names[value.callee]
        elif isinstance(value, CallStep):
            if value.call.callee not in shadowed and value.call.callee in generic_names:
                changed["call"] = changed.get("call", value.call).with_(
                    callee=generic_names[value.call.callee]
                )
        return value.with_(**changed) if changed else value

    if generic_names:
        declarations = {
            s.name: s for s in module.statements if isinstance(s, ProgramDecl)
        }
        clones = tuple(
            declarations[name].with_(name=generic)
            for name, generic in generic_names.items()
        )
        module = retain_generic_calls(
            module.with_(statements=(*clones, *module.statements))
        )
    occupied = set(names)
    statements = []
    for declaration in module.statements:
        if (
            not isinstance(declaration, ProgramDecl)
            or declaration.name not in solutions
        ):
            statements.append(declaration)
            continue
        sizes = solutions[declaration.name]
        objects = {}
        for name, size in sizes.items():
            object_name = f"__qvr_{declaration.name}_{name}"
            while object_name in occupied:
                object_name += "_"
            occupied.add(object_name)
            objects[name] = object_name
            statements.append(
                ObjectDecl(
                    names=(object_name,),
                    init=TypeFromExpr(
                        expr=DiscreteConstructor(
                            constructor="FinSet", args=(str(size),)
                        )
                    ),
                )
            )

        def substitute(value: object) -> object:
            if isinstance(value, TypeName) and value.name in objects:
                return value.with_(name=objects[value.name])
            if isinstance(value, QiecIndexName) and value.name in sizes:
                return QiecIndexLiteral(
                    value=sizes[value.name], line=value.line, col=value.col
                )
            if isinstance(value, AxisSpec):
                return value.with_(
                    over=tuple(objects.get(name, name) for name in value.over),
                    iid_over=tuple(objects.get(name, name) for name in value.iid_over),
                )
            if isinstance(value, ContinuousConstructor | DiscreteConstructor):
                return value.with_(
                    args=tuple(
                        str(sizes[name]) if name in sizes else name
                        for name in value.args
                    )
                )
            if isinstance(value, tuple):
                return tuple(substitute(item) for item in value)
            if isinstance(value, dx.Model):
                return value.with_(
                    **{
                        field: substitute(getattr(value, field))
                        for field in type(value).__field_specs__
                        if field not in {"binders", "type_params"}
                    }
                )
            return value

        specialized = substitute(declaration)
        remaining = tuple(
            p for p in declaration.type_params or () if p.name not in sizes
        )
        statements.append(
            specialized.with_(
                binders=tuple(p for p in declaration.binders if p.name not in sizes),
                type_params=remaining or None,
            )
        )
    return module.with_(statements=tuple(statements))


__all__ = []
