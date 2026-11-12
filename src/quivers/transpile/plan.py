"""The target plan of a program, derived from its checked kernel module.

A program is one named computation of the checked
[`QiecModule`][quivers.qiec.QiecModule], performing ``Random.sample`` and
``Score.add`` on the module's canonical instances, binding pure values,
and calling its marginalization helpers. A probabilistic-programming
target has statements for exactly those things: a draw, an observation,
a deterministic binding, a log-density increment, and an integration
scope. The plan pass reads the computation and recognizes each of them,
producing the [`IRProgram`][quivers.transpile.ir.IRProgram] body the
renderers consume: one [`IRSample`][quivers.transpile.ir.IRSample] per
``Random.sample`` request, one [`IRObserve`][quivers.transpile.ir.IRObserve]
per scored density at an observation, one
[`IRDeterministic`][quivers.transpile.ir.IRDeterministic] per pure binding,
one [`IRScore`][quivers.transpile.ir.IRScore] per other weight, and one
[`IRMarginalize`][quivers.transpile.ir.IRMarginalize] per helper call whose
answer is scored. The plan's inputs are the entry point's parameters,
typed by the module and constrained by how the body reads them.

The pass reads the source declarations only for what the module does
not carry: object cardinalities and bounds, the morphism a site draws
through (which fixes a class-index draw's alphabet and a structured
family's wire form), and the option blocks of structured families. A
construct the target vocabulary has no statement for, a call of a
deduction or a recurrence, is reported with the kind of the request or
call the plan cannot place, so a renderer never receives a body with a
hole in it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import fields, replace
from typing import cast

from torch.distributions import Distribution

from quivers.dsl.ast_nodes import (
    Expr,
    LetStep,
    MarginalizeStep,
    Module,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    SampleStep,
    TypeName,
)
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprFactor,
    LetExprNode,
    LetExprString,
    LetExprVar,
    LetFactorBinder,
    LetFactorCase,
)
from quivers.transpile._resolve import build_let_table, build_morphism_table
from quivers.qiec import QiecModule, validate_module
from quivers.qiec.effects import EffectRequest
from quivers.qiec.module import NamedComputation
from quivers.qiec.programs import ProgramEntry, ProgramParameter
from quivers.qiec.terms import (
    AffineMap,
    Bind,
    Call,
    Comprehension,
    Computation,
    DistributionValue,
    Gather,
    Handle,
    KernelMatrix,
    LetBinding,
    LiteralValue,
    Local,
    LogDensity,
    NewInstance,
    Perform,
    PlateAxis,
    PlateShape,
    PrimitiveApplication,
    Projection,
    Return,
    SegmentSum,
    SiteValue,
    TableMap,
    TensorValue,
    TupleValue,
    Value,
    Var,
    WeightSum,
)
from quivers.dsl.program_elaboration import GAP_CODE
from quivers.dsl.qiec_lowering import (
    QiecDiagnosticError,
    has_qiec_surface,
    lower_qvr_to_qiec,
)
from quivers.qiec.canonical import sampled_element, tensor_shape
from quivers.qiec.families import FAMILIES
from quivers.qiec.types import (
    INT,
    IndexLiteral,
    IndexTerm,
    IndexVariable,
    StaticArgument,
    TypeExpr,
)
import didactic.api as dx

from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.qiec_ir import host_runtime_capability_diagnostics
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile.family_meta import (
    FAMILY_META,
    FamilyMeta,
    class_index_outcome,
    marginalize_support,
)
from quivers.transpile.ir import (
    CSIntegerInterval,
    CSNonnegativeInteger,
    CSPositive,
    CSReal,
    ConstraintSpec,
    Dim,
    DimDynamic,
    DimStatic,
    IRArg,
    IRArgBroadcast,
    IRArgKernel,
    IRArgList,
    IRArgMatrix,
    IRArgNumber,
    IRArgRef,
    IRCall,
    IRDataInput,
    IRDeterministic,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRReturn,
    IRSample,
    IRScore,
    LetAffineSource,
    LetExprAffineMap,
    Plate,
    StructuredDataArg,
    StructuredZeroVectorArg,
    from_constraint,
)
from quivers.transpile.lower import (
    _DATA_CONSTRAINT_FACTORY,
    _LowerCtx,
    _ParamHead,
    _apply_declared_bounds,
    _arg_names_for,
    _codomain_alphabet,
    _wrap_for_constraint,
    axis_shape,
    _floor_expr,
    _head_binding_name,
    _head_raw_binding_name,
    _resolve_arg_constraints,
    _resolve_support,
    _retarget_alphabet_arg,
    object_shapes,
    pick_program,
)
from quivers.transpile.qiec_ir import IRQiecStatic, _convert, lower_qiec_ir
from quivers.transpile._pure_calls import (
    _VALUE_TYPES,
    _TRANSPARENT,
    _LetFloater,
    _expression,
    _local_names,
    pure_call_bindings,
)

#: The registry's name for a structured family's constructor argument
#: where the two differ.
_REGISTRY_NAMES: dict[str, dict[str, str]] = {"GP": {"covariance_matrix": "kernel"}}


class _Planner:
    """Derive one program's plan from the checked module.

    Parameters
    ----------
    module : QiecModule
        The checked module.
    source : Module
        The expanded source, read for object cardinalities and bounds,
        morphism declarations, and the morphism each site draws
        through.
    target : str
        The transpile target the plan serves, named in diagnostics.
    """

    def __init__(self, module: QiecModule, source: Module, target: str) -> None:
        self.module = module
        self.source = source
        self.target = target
        self.computations = {item.name: item for item in module.computations}
        self.entries = {entry.name: entry for entry in module.entries}
        shapes = object_shapes(source)
        self.cards = {
            name: shape.extent
            for name, shape in shapes.items()
            if shape.extent is not None
        }
        self.morphisms = build_morphism_table(source)
        self.lets: dict[str, Expr] = build_let_table(source)
        self.shapes = shapes
        self.sentinel_cache: dict[tuple[str, tuple[str, ...]], Distribution] = {}
        self.marginal_names = {
            name
            for name in self.computations
            if name.endswith("_marginal") and "__" in name
        }

    # ------------------------------------------------------------------
    # entry

    def plan(self, program: ProgramDecl) -> IRProgram:
        """The plan of one program.

        Parameters
        ----------
        program : ProgramDecl
            The program's source declaration, which the expanded
            module's steps and the plan's name are read from.

        Returns
        -------
        IRProgram
            The plan: the entry point's inputs, the body derived from
            its computation, the object cardinalities, and the module.

        Raises
        ------
        UnsupportedConstruct
            If the module records no entry for the program, or the
            body performs a request or call the target vocabulary has
            no statement for.
        """
        entry = self.entries.get(program.name)
        computation = self.computations.get(program.name)
        if entry is None or computation is None:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"program:unelaborated:{program.name}"],
            )
        ctx = _LowerCtx(
            morphisms=self.morphisms,
            lets=self.lets,
            cards=dict(self.cards),
            family_set=frozenset(FAMILY_META),
            sentinel_cache=self.sentinel_cache,
            program=program,
            real_widths={
                name: shape.real_width
                for name, shape in self.shapes.items()
                if shape.real_width is not None
            },
            bounds={
                name: shape.bounds
                for name, shape in self.shapes.items()
                if shape.bounds.is_bounded
            },
            shapes=self.shapes,
        )
        walk = _ProgramWalk(self, entry, computation, program, ctx)
        body = walk.body()
        inputs = walk.inputs(body)
        return IRProgram(
            name=program.name,
            inputs=tuple(_spelled(item) for item in inputs),
            body=tuple(_spelled(item) for item in body),
            module=lower_qiec_ir(self.module),
            cards=dict(self.cards),
        )


class _ProgramWalk:
    """The derivation of one program's body and inputs.

    Parameters
    ----------
    planner : _Planner
        The planner, holding the module and the declaration tables.
    entry : ProgramEntry
        The program's entry point.
    computation : NamedComputation
        The program's computation.
    program : ProgramDecl
        The program's expanded source declaration.
    ctx : _LowerCtx
        The support tables the constraints are resolved against.
    """

    def __init__(
        self,
        planner: _Planner,
        entry: ProgramEntry,
        computation: NamedComputation,
        program: ProgramDecl,
        ctx: _LowerCtx,
    ) -> None:
        self.planner = planner
        self.entry = entry
        self.computation = computation
        self.program = program
        self.ctx = ctx
        self.site_steps: dict[str, SampleStep | ObserveStep | MarginalizeStep] = {}
        _collect_site_steps(program.draws, self.site_steps)
        self.factor_axes: dict[str, tuple[str, ...]] = {}
        _collect_factor_axes(program.draws, self.factor_axes)
        self.observed: dict[str, IRObserve] = {}
        self.structured_inputs: dict[str, tuple[ConstraintSpec, Plate]] = {}
        self.param_map_inputs: dict[str, Plate] = {}
        self.bound_plates = ctx.bound_plates
        self.head_nodes: list[IRNode] = []
        self.axes: dict[str, tuple[str, ...]] = {
            parameter.name: parameter.axes for parameter in entry.parameters
        }
        self.data_parameters: dict[str, ProgramParameter] = {
            parameter.name: parameter
            for parameter in entry.parameters
            if parameter.role == "data"
        }
        self.local_types: dict[str, TypeExpr] = {}
        self.via: str | None = None
        # The components of every local bound to a tuple of bindings,
        # which the plan holds as the bindings themselves.
        self.tuples: dict[str, tuple[Value, ...]] = {}
        self.floater = _LetFloater(
            _local_names(computation.body)
            | {parameter.name for parameter in computation.parameters}
            | set(self.site_steps)
        )

    # ------------------------------------------------------------------
    # body

    def body(self) -> tuple[IRNode, ...]:
        """The plan's body.

        Returns
        -------
        tuple[IRNode, ...]
            One node per statement the computation performs, in order.
        """
        return tuple(self._nodes(self.computation.body))

    def _nodes(self, computation: Computation) -> Iterator[IRNode]:
        """The nodes of a computation, one per bound statement.

        Parameters
        ----------
        computation : Computation
            A chain of binds ending in a return.

        Yields
        ------
        IRNode
            The statements in order.

        Raises
        ------
        UnsupportedConstruct
            If a computation form is not a statement of the target
            vocabulary.
        """
        node: Computation = computation
        pending_marginal: dict[str, IRMarginalize] = {}
        emitted: list[IRNode] = []
        while isinstance(node, Bind):
            first = node.first
            if isinstance(first, Bind):
                # ``traverse`` and other finite sequencing forms elaborate to
                # a computation in the first position of the source binding.
                # Associate that nested bind into the surrounding chain before
                # recognizing target statements. This is the monad
                # associativity law; the inner local cannot occur in the
                # outer continuation, so extending its lexical scope here is
                # capture-free.
                node = Bind(
                    first.binder,
                    first.first,
                    Bind(node.binder, first.then, node.then),
                )
                continue
            binder = node.binder
            shared, first = self.floater.computation(first)
            for binding in shared:
                self.local_types[binding.binder.name] = binding.binder.type
                emitted.extend(self._deterministic(binding.binder, binding.value))
            self.local_types[binder.name] = binder.type
            if isinstance(first, Perform):
                request = first.request
                if request.effect.name == "Random":
                    emitted.extend(self._sample(request, binder))
                elif request.effect.name == "Score":
                    emitted.extend(
                        self._score(request, binder, pending_marginal, emitted)
                    )
                else:
                    raise UnsupportedConstruct(
                        f"qvr-{self.planner.target}",
                        [f"qiec:effect:{request.effect.name}:{self.program.name}"],
                    )
            elif isinstance(first, Return):
                emitted.extend(self._deterministic(binder, first.value))
            elif isinstance(first, Call):
                if first.name in self.planner.marginal_names:
                    pending_marginal[binder.name] = self._marginalize(first)
                    emitted.extend(self._flush_heads())
                else:
                    emitted.extend(self._call(binder, first))
            else:
                raise UnsupportedConstruct(
                    f"qvr-{self.planner.target}",
                    [f"qiec:computation:{type(first).__name__}:{self.program.name}"],
                )
            node = node.then
        if isinstance(node, Return):
            names = _returned_names(node.value)
            if names is not None:
                emitted.append(IRReturn(names=names))
                yield from emitted
                return
        raise UnsupportedConstruct(
            f"qvr-{self.planner.target}",
            [f"qiec:computation:{type(node).__name__}:{self.program.name}"],
        )

    # ------------------------------------------------------------------
    # draws

    def _sample(self, request: EffectRequest, binder: Local) -> Iterator[IRNode]:
        """The node of a ``Random.sample`` request.

        Parameters
        ----------
        request : EffectRequest
            The request, carrying the site and its sampleable.
        binder : Local
            The local the draw binds.

        Yields
        ------
        IRNode
            The head bindings of a parameter map, then the sample.
        """
        site, distribution = _site_and_distribution(request, self.planner.target)
        name = site.label
        family, args, arg_names, plate, constraint = self._distribution(
            distribution, name
        )
        yield from self._flush_heads()
        node = IRSample(
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            constraint=constraint,
            plate=plate,
        )
        self.bound_plates[name] = plate
        self.ctx.bound_kinds[name] = "sample"
        self.axes[name] = _dim_names(plate)
        if binder.name != name:
            self.bound_plates[binder.name] = plate
            self.axes[binder.name] = self.axes[name]
        yield node

    def _score(
        self,
        request: EffectRequest,
        binder: Local,
        pending_marginal: dict[str, IRMarginalize],
        emitted: list[IRNode],
    ) -> Iterator[IRNode]:
        """The node of a ``Score.add`` request.

        Parameters
        ----------
        request : EffectRequest
            The request, carrying the weight added.
        binder : Local
            The local the unit answer binds.
        pending_marginal : dict[str, IRMarginalize]
            The marginalization helpers called so far, by the local
            their answer is bound to.
        emitted : list[IRNode]
            The nodes emitted so far, whose last binding a score of
            that binding's value absorbs: ``score p = e`` is one
            statement binding ``p`` and adding it.

        Yields
        ------
        IRNode
            An observation, a marginalization scope, or a score.
        """
        (weight,) = request.arguments
        scored = weight
        if isinstance(scored, PrimitiveApplication) and scored.name in _TRANSPARENT:
            scored = scored.arguments[0]
        if (
            isinstance(scored, Var)
            and emitted
            and isinstance(emitted[-1], IRDeterministic)
            and emitted[-1].name == scored.local.name
        ):
            bound = emitted.pop()
            assert isinstance(bound, IRDeterministic)
            yield IRScore(name=bound.name, expr=bound.expr)
            return
        via: str | None = None
        density = weight
        if isinstance(density, WeightSum):
            density = density.value
        if isinstance(density, SegmentSum):
            index = density.index
            if not isinstance(index, Var):
                raise UnsupportedConstruct(
                    f"qvr-{self.planner.target}",
                    [f"qiec:fibration:{self.program.name}"],
                )
            via = index.local.name
            density = density.value
        if isinstance(density, LogDensity) and isinstance(density.value, Var):
            observation = density.value.local.name
            if isinstance(density.sampleable, DistributionValue):
                family, args, arg_names, plate, constraint = self._distribution(
                    density.sampleable, observation, via=via
                )
                yield from self._flush_heads()
                node = IRObserve(
                    name=observation,
                    family=family,
                    args=args,
                    arg_names=arg_names,
                    constraint=constraint,
                    plate=plate,
                    via=via,
                )
                self.observed[observation] = node
                self.bound_plates[observation] = plate
                yield node
                return
        if isinstance(weight, Var) and weight.local.name in pending_marginal:
            yield pending_marginal.pop(weight.local.name)
            return
        yield IRScore(name=binder.name, expr=self._expr(weight))

    def _deterministic(self, binder: Local, value: Value) -> Iterator[IRNode]:
        """The node of a pure binding.

        Parameters
        ----------
        binder : Local
            The bound local.
        value : Value
            The value bound.

        Yields
        ------
        IRNode
            The deterministic binding.
        """
        value = self._resolved(value)
        if isinstance(value, TupleValue):
            # A pair of bindings has no expression on any target; its
            # components are read back where the pair is projected.
            self.tuples[binder.name] = value.items
            return
        axes = self._factor_axes(binder.name, binder.type) or self._inferred_axes(
            binder.type, value
        )
        self.axes[binder.name] = axes
        plate = self._inferred_plate(binder.type, axes, value)
        self.bound_plates[binder.name] = plate
        self.ctx.bound_kinds[binder.name] = "deterministic"
        yield IRDeterministic(
            name=binder.name,
            expr=self._expr(value),
            constraint=CSReal(),
            plate=plate,
        )

    def _resolved(self, value: Value) -> Value:
        """A value with every projection of a held pair read back.

        Parameters
        ----------
        value : Value
            The value.

        Returns
        -------
        Value
            The value with ``t[i]`` replaced by the ``i``-th component
            of every local ``t`` the plan holds as a tuple of bindings.
        """
        if isinstance(value, Projection):
            inner = self._resolved(value.value)
            if isinstance(inner, Var) and inner.local.name in self.tuples:
                return self.tuples[inner.local.name][value.position]
            return replace(value, value=inner)
        if isinstance(value, (Var, LiteralValue)):
            return value
        changes: dict[str, Value | tuple[Value, ...]] = {}
        for field in fields(value):
            current = getattr(value, field.name)
            if isinstance(current, _VALUE_TYPES):
                resolved = self._resolved(current)
                if resolved is not current:
                    changes[field.name] = resolved
            elif isinstance(current, tuple) and any(
                isinstance(item, _VALUE_TYPES) for item in current
            ):
                items = tuple(
                    self._resolved(item) if isinstance(item, _VALUE_TYPES) else item
                    for item in current
                )
                if any(a is not b for a, b in zip(items, current, strict=True)):
                    changes[field.name] = items
        return replace(value, **changes) if changes else value

    def _factor_axes(self, name: str, type_: TypeExpr) -> tuple[str, ...]:
        """The axis names a source ``factor`` binding states.

        ``let after = factor t : Step in ...`` names its axis ``Step``,
        which the checked comprehension keeps only as an extent. Two
        objects of one cardinality leave that extent ambiguous, so the
        source binder's object names the axis instead, and a value read
        off the binding inherits it.

        Parameters
        ----------
        name : str
            The bound name.
        type_ : TypeExpr
            The checked type of the bound value.

        Returns
        -------
        tuple[str, ...]
            One name per tensor axis when the binding is a source
            factor whose binder objects have the axes' extents, the
            trailing axes named as `_inferred_axes` names them; empty
            otherwise.
        """
        stated = self.factor_axes.get(name)
        shape = tensor_shape(type_)
        if stated is None or shape is None:
            return ()
        dimensions = tuple(shape[1])
        if len(stated) > len(dimensions):
            return ()
        for axis, size in zip(stated, dimensions, strict=False):
            if not isinstance(size, IndexLiteral) or self.ctx.cards.get(axis) != (
                size.value
            ):
                return ()
        rest = _typed_dims(type_, ())[len(stated) :]
        return (*stated, *(str(dim.name) for dim in rest))

    def _inferred_axes(self, type_: TypeExpr, value: Value) -> tuple[str, ...]:
        """The axis names of a bound value.

        Parameters
        ----------
        type_ : TypeExpr
            The value's type.
        value : Value
            The value, whose bindings' axes name its own when their
            extents agree.

        Returns
        -------
        tuple[str, ...]
            Per dimension: the name a referenced binding of the same
            extents gives it, else the one declared object of that
            cardinality, else the position.
        """
        shape = tensor_shape(type_)
        if shape is None:
            return ()
        wanted = tuple(shape[1])
        for name in _referenced(value):
            known = self.axes.get(name)
            if known is None or len(known) < len(wanted):
                continue
            local_type = self._type_of(name)
            local_shape = tensor_shape(local_type) if local_type is not None else None
            if local_shape is None:
                continue
            offset = len(local_shape[1]) - len(wanted)
            if offset >= 0 and tuple(local_shape[1][offset:]) == wanted:
                return tuple(known[offset:])
        names: list[str] = []
        for index, size in enumerate(wanted):
            if isinstance(size, IndexVariable):
                names.append(size.name)
                continue
            matching = [
                object_name
                for object_name, card in self.ctx.cards.items()
                if isinstance(size, IndexLiteral) and card == size.value
            ]
            names.append(matching[0] if len(matching) == 1 else f"axis{index}")
        return tuple(names)

    def _inferred_plate(
        self, type_: TypeExpr, axes: tuple[str, ...], value: Value
    ) -> Plate:
        """The plate of a bound value.

        A value read off a binding with event dimensions keeps them
        as its own trailing event, ``Z[i]`` being one row of the
        event-shaped ``Z``; every other dimension is a batch.

        Parameters
        ----------
        type_ : TypeExpr
            The value's type.
        axes : tuple[str, ...]
            The dimensions' names.
        value : Value
            The value.

        Returns
        -------
        Plate
            The dimensions split into batch and event.
        """
        dims = _typed_dims(type_, axes)
        for name in _referenced(value):
            known = self.bound_plates.get(name)
            if known is None or not known.event_dims:
                continue
            count = len(known.event_dims)
            if len(dims) >= count and tuple(dims[len(dims) - count :]) == tuple(
                known.event_dims
            ):
                return Plate(
                    event_dims=tuple(dims[len(dims) - count :]),
                    batch_dims=tuple(dims[: len(dims) - count]),
                )
        return Plate(event_dims=(), batch_dims=dims)

    def _type_of(self, name: str) -> TypeExpr | None:
        """The type of a bound name.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        TypeExpr | None
            The entry parameter's or the computation local's type.
        """
        for parameter in self.entry.parameters:
            if parameter.name == name:
                return parameter.type
        return self.local_types.get(name)

    def _call(self, binder: Local, call: Call) -> Iterator[IRNode]:
        """The node of a call of one of the module's computations.

        A pure computation whose body is a chain of pure bindings is
        inlined: its result reads back as one let expression over the
        arguments, which every target's arithmetic carries. Any other
        computation is called as a host function, under the target's
        own draw and score primitives for the program's instances.

        Parameters
        ----------
        binder : Local
            The local the call's result binds.
        call : Call
            The call.

        Yields
        ------
        IRNode
            A deterministic binding for an inlined pure computation,
            else a call node.

        Raises
        ------
        UnsupportedConstruct
            If the callee performs on an instance other than the
            program's ``random`` and ``score``, or is not a computation
            of the module.
        """
        callee = self.planner.computations.get(call.name)
        if callee is None:
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [f"qiec:call:{call.name}:{self.program.name}"],
            )
        if call.name in self.planner.entries:
            # A program called as a computation keeps its own sites
            # under the callee's names, which no target's plan holds; a
            # draw from the program runs it in place under this
            # program's names instead.
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [
                    f"qiec:call:program:{call.name}:{self.program.name}: a "
                    f"program is called as a computation; draw from it with "
                    f"`sample {binder.name} <- {call.name}` to run it in place"
                ],
            )
        inlined = self._inlined(callee, call, binder)
        arguments_value = TupleValue(call.arguments, binder.type)
        axes = self._inferred_axes(binder.type, arguments_value)
        self.axes[binder.name] = axes
        plate = self._inferred_plate(binder.type, axes, arguments_value)
        self.bound_plates[binder.name] = plate
        if inlined is not None:
            shared, result = inlined
            nodes: list[IRNode] = []
            for binding in shared:
                self.local_types[binding.binder.name] = binding.binder.type
                nodes.extend(self._deterministic(binding.binder, binding.value))
            self.axes[binder.name] = axes
            self.bound_plates[binder.name] = plate
            self.ctx.bound_kinds[binder.name] = "deterministic"
            nodes.append(
                IRDeterministic(
                    name=binder.name, expr=result, constraint=CSReal(), plate=plate
                )
            )
            yield from nodes
            return
        arguments = tuple(self._expr(argument) for argument in call.arguments)
        allowed = {self.entry.random_instance, self.entry.score_instance}
        for row_entry in call.effects.entries:
            if row_entry.instance not in allowed:
                raise UnsupportedConstruct(
                    f"qvr-{self.planner.target}",
                    [f"qiec:effect:{row_entry.effect.name}:{self.program.name}"],
                )
        self.ctx.bound_kinds[binder.name] = "call"
        yield IRCall(
            name=binder.name,
            callee=call.name,
            static_arguments=tuple(
                _lower_static(argument) for argument in call.static_arguments
            ),
            arguments=arguments,
            random_instance=str(self.entry.random_instance),
            score_instance=str(self.entry.score_instance),
            plate=plate,
        )

    def _inlined(
        self, callee: NamedComputation, call: Call, binder: Local
    ) -> tuple[tuple[LetBinding, ...], LetExprNode] | None:
        """A pure computation's result at the call, as shared bindings.

        A pure computation whose body is a chain of pure bindings is
        inlined: its parameters are bound to the call's arguments and
        each of its bindings to its value, as one
        [`LetValue`][quivers.qiec.LetValue] that the plan's floater
        hoists to statements of their own under fresh names. Each
        binding is thus stated once, however many times the body reads
        it, so the inlined code grows linearly in the computation's
        size.

        Parameters
        ----------
        callee : NamedComputation
            The computation called.
        call : Call
            The call.
        binder : Local
            The local the call's result binds.

        Returns
        -------
        tuple[tuple[LetBinding, ...], LetExprNode] | None
            The hoisted bindings and the result's expression, or
            ``None`` when the computation performs an effect, takes
            static arguments, calls anything, or binds a value the
            expression language has no form for; such a call is made
            through the target's host runtime.
        """
        found = pure_call_bindings(callee, call.arguments, binder.type, self.floater)
        if found is None:
            return None
        hoisted, result = found
        try:
            for binding in hoisted:
                self._expr(binding.value)
            return hoisted, self._expr(result)
        except UnsupportedConstruct:
            return None

    def _marginalize(self, call: Call) -> IRMarginalize:
        """The scope a marginalization helper's call denotes.

        Parameters
        ----------
        call : Call
            The call of the helper.

        Returns
        -------
        IRMarginalize
            The latent's draw and the scope's statements.

        Raises
        ------
        UnsupportedConstruct
            If the helper's body is not one enumerated draw over a
            collected scope, or reduces other than by log-sum-exp.
        """
        helper = self.planner.computations[call.name]
        handler_names = {
            handler.id: handler.name for handler in self.planner.module.handlers
        }
        body = helper.body
        # The helper allocates a Random instance for the latent, handles
        # it with the enumeration handler, draws the latent, allocates
        # a Weight instance, handles it with the collecting handler, and
        # scores the scope inside.
        if not isinstance(body, NewInstance):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}", [f"marginalize:shape:{call.name}"]
            )
        handled = body.body
        if not isinstance(handled, Handle):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}", [f"marginalize:shape:{call.name}"]
            )
        enumeration = handler_names.get(handled.handler, "")
        reduction = _marginal_reduction(enumeration)
        if reduction != "logsumexp":
            raise UnsupportedConstruct(
                "qvr-lower",
                [f"marginalize:reduction:{reduction}"],
            )
        draw = handled.computation
        if not (
            isinstance(draw, Bind)
            and isinstance(draw.first, Perform)
            and draw.first.request.effect.name == "Random"
        ):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}", [f"marginalize:shape:{call.name}"]
            )
        site, distribution = _site_and_distribution(
            draw.first.request, self.planner.target
        )
        latent = site.label
        family, args, arg_names, plate, constraint = self._distribution(
            distribution, latent, marginalized=True
        )
        constraint = self._enumerated_support(
            latent, family, args, arg_names, constraint
        )
        self.bound_plates[latent] = plate
        self.ctx.bound_kinds[latent] = "marginalize"
        self.axes[latent] = _dim_names(plate)
        self.local_types[latent] = draw.binder.type
        collected = draw.then
        if not isinstance(collected, NewInstance) or not isinstance(
            collected.body, Handle
        ):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}", [f"marginalize:shape:{call.name}"]
            )
        scope = tuple(self._scope_nodes(collected.body.computation))
        node = IRMarginalize(
            latent=latent,
            family=family,
            args=args,
            arg_names=arg_names,
            constraint=constraint,
            plate=plate,
            reduction="logsumexp",
            scope=scope,
        )
        return node

    def _enumerated_support(
        self,
        latent: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        constraint: ConstraintSpec,
    ) -> ConstraintSpec:
        """The support a marginalized latent is enumerated over.

        A family whose support is an integer range sized by its
        arguments (Categorical's ``0 .. K - 1``) has that range read
        off a placeholder instance, which is exact for literal
        arguments but not for a reference to a bound name, whose
        placeholder has no size of its own. The latent's atoms are
        the elements of its declared index object, so a declared
        index sizes the range; without one, the width of the
        probability argument does, read off the referenced binding's
        plate. A range neither determines is refused rather than
        enumerated at a placeholder size.

        Parameters
        ----------
        latent : str
            The latent.
        family : str
            Its family.
        args : tuple[IRArg, ...]
            The family's arguments.
        arg_names : tuple[str, ...]
            Their names.
        constraint : ConstraintSpec
            The support read off the placeholder instance.

        Returns
        -------
        ConstraintSpec
            The support, with an integer range's bounds taken from
            the latent's index object or its probability argument.

        Raises
        ------
        UnsupportedConstruct
            If the range depends on a referenced argument and neither
            the declared index nor that argument's plate sizes it.
        """
        if not isinstance(constraint, CSIntegerInterval):
            return constraint
        if all(isinstance(arg, IRArgNumber | IRArgList) for arg in args):
            return constraint
        step = self.site_steps.get(latent)
        if isinstance(step, MarginalizeStep) and step.index is not None:
            size = axis_shape(step.index, dict(self.planner.cards))
            if size is not None:
                return CSIntegerInterval(lower=0, upper=size - 1)
        named = dict(zip(arg_names, args, strict=True))
        width_arg = named.get("probs", named.get("logits"))
        width: int | None = None
        if isinstance(width_arg, IRArgList):
            width = len(width_arg.elements)
        elif isinstance(width_arg, IRArgBroadcast) and width_arg.target_shape:
            width = width_arg.target_shape[-1]
        elif isinstance(width_arg, IRArgRef):
            bound = self.bound_plates.get(width_arg.name)
            if bound is not None:
                axes = (*bound.batch_dims, *bound.event_dims)
                if axes and isinstance(axes[-1], DimStatic):
                    width = axes[-1].size
        if width is None:
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [
                    f"marginalize:unknown-cardinality:{family}: the support "
                    f"of `{latent}` depends on an argument whose size is not "
                    f"static, and the latent declares no index object"
                ],
            )
        return CSIntegerInterval(lower=0, upper=width - 1)

    def _scope_nodes(self, computation: Computation) -> Iterator[IRNode]:
        """The statements of a marginalization scope.

        Parameters
        ----------
        computation : Computation
            The scope's computation, whose weights are added on the
            scope's ``Weight`` instance.

        Yields
        ------
        IRNode
            The scope's statements, without its unit return.
        """
        node: Computation = computation
        pending: dict[str, IRMarginalize] = {}
        emitted: list[IRNode] = []
        while isinstance(node, Bind):
            first = node.first
            if isinstance(first, Bind):
                node = Bind(
                    first.binder,
                    first.first,
                    Bind(node.binder, first.then, node.then),
                )
                continue
            self.local_types[node.binder.name] = node.binder.type
            if isinstance(first, Perform):
                if first.request.effect.name == "Random":
                    emitted.extend(self._sample(first.request, node.binder))
                elif first.request.effect.name in ("Weight", "Score"):
                    emitted.extend(
                        self._score(first.request, node.binder, pending, emitted)
                    )
                else:
                    raise UnsupportedConstruct(
                        f"qvr-{self.planner.target}",
                        [
                            f"qiec:effect:{first.request.effect.name}:{self.program.name}"
                        ],
                    )
            elif isinstance(first, Return):
                emitted.extend(self._deterministic(node.binder, first.value))
            elif isinstance(first, Call) and first.name in self.planner.marginal_names:
                pending[node.binder.name] = self._marginalize(first)
                emitted.extend(self._flush_heads())
            elif isinstance(first, Call):
                emitted.extend(self._call(node.binder, first))
            else:
                raise UnsupportedConstruct(
                    f"qvr-{self.planner.target}",
                    [f"qiec:computation:{type(first).__name__}:{self.program.name}"],
                )
            node = node.then
        if not isinstance(node, Return):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [f"qiec:computation:{type(node).__name__}:{self.program.name}"],
            )
        yield from emitted

    def _flush_heads(self) -> Iterator[IRNode]:
        """The head bindings queued by the last distribution read.

        Yields
        ------
        IRNode
            The deterministic bindings, in order.
        """
        nodes = tuple(self.head_nodes)
        self.head_nodes.clear()
        yield from nodes

    # ------------------------------------------------------------------
    # distributions

    def _distribution(
        self,
        distribution: DistributionValue,
        site: str,
        *,
        marginalized: bool = False,
        via: str | None = None,
    ) -> tuple[str, tuple[IRArg, ...], tuple[str, ...], Plate, ConstraintSpec]:
        """Read a construction as a family call.

        Parameters
        ----------
        distribution : DistributionValue
            The construction.
        site : str
            The site drawn or scored, which names the head bindings of
            a mapped kernel.
        marginalized : bool
            Whether the site is a marginalized latent, whose plate
            keeps only the grouping axes.
        via : str | None
            The fibration an observation reads its arguments through;
            the plan states it on the observation rather than on the
            arguments, so a gather by it is stripped.

        Returns
        -------
        tuple[str, tuple[IRArg, ...], tuple[str, ...], Plate, ConstraintSpec]
            The family, its arguments and their names, the plate, and
            the support.

        Raises
        ------
        UnsupportedConstruct
            If the family is outside the target vocabulary or an
            argument has no wire form.
        """
        family = distribution.name
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                "qvr-lower",
                [f"family:{family}: not in FAMILY_META registry"],
            )
        step = self.site_steps.get(site)
        morphism = step.morphism if step is not None else ""
        plate = self._broadcast_axes(
            _plate_of(distribution.plate, marginalized, meta, distribution, self),
            distribution,
        )
        named = {name: value for name, value in distribution.arguments}
        self.via = via
        if meta.structured_lowering is not None and _structured(distribution, meta):
            args, arg_names = self._structured_args(distribution, site, plate)
        else:
            args, arg_names = self._arguments(distribution, site, plate, meta, morphism)
        self.via = None
        if (
            family == "Categorical"
            and arg_names == ("logits",)
            and self.planner.target
            not in {"ir", "pyro", "numpyro", "pymc", "edward2", "stan"}
        ):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [
                    "family:Categorical:logits: the target only exposes the "
                    "probability parameterization; apply softmax in QVR and "
                    "pass the result as probs"
                ],
            )
        args, arg_names = self._parameterization(
            family, args, arg_names, marginalized=marginalized
        )
        constraint = _apply_declared_bounds(
            from_constraint(_resolve_support(meta, args, self.ctx, arg_names)),
            plate,
            self.ctx,
            site,
        )
        args, constraint = self._class_index(
            morphism, meta, args, arg_names, constraint, site
        )
        del named
        return family, args, arg_names, plate, constraint

    def _broadcast_axes(self, plate: Plate, distribution: DistributionValue) -> Plate:
        """Name a broadcast plate's axes after the argument it is read off.

        An unplated scalar family whose parameter is filled by a tensor
        binding, ``Normal(loc=z, scale=0.5)`` with ``z`` a vector, is
        replicated once per entry of ``z``; the checked program names
        those batch axes after the binding, ``z``, ``z_1``, and so on.
        The entry at each position is the entry of ``z`` there, so the
        plate's axes are ``z``'s own axes, which the plan already names
        (``Feat`` for a ``z`` read off a draw over ``Feat``). Naming them
        so lets a target that visits the plate one position at a time
        read the binding at the position it visits.

        Parameters
        ----------
        plate : Plate
            The site's plate as the checked program states it.
        distribution : DistributionValue
            The construction, whose arguments name the broadcast axes.

        Returns
        -------
        Plate
            The plate with every broadcast axis renamed to the axis of
            the binding it is read off, or `plate` unchanged when its
            axes are not a binding's broadcast or the binding's axes
            are unnamed or repeat a name.
        """
        if plate.event_dims or not plate.batch_dims:
            return plate
        names = tuple(dim.name for dim in plate.batch_dims)
        for _, value in distribution.arguments:
            if not isinstance(value, Var):
                continue
            stem = value.local.name
            stated = tuple(
                stem if index == 0 else f"{stem}_{index}" for index in range(len(names))
            )
            known = self.axes.get(stem)
            if (
                names != stated
                or known is None
                or len(known) != len(names)
                or len(set(known)) != len(known)
                or any(name in self.ctx.cards for name in stated)
            ):
                continue
            return Plate(
                event_dims=(),
                batch_dims=tuple(
                    _renamed_dim(dim, axis)
                    for dim, axis in zip(plate.batch_dims, known, strict=True)
                ),
            )
        return plate

    def _arguments(
        self,
        distribution: DistributionValue,
        site: str,
        plate: Plate,
        meta: FamilyMeta,
        morphism: str,
    ) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
        """The wire arguments of a family call.

        Parameters
        ----------
        distribution : DistributionValue
            The construction.
        site : str
            The site the call is at.
        plate : Plate
            The site's plate.
        meta : FamilyMeta
            The family's transpile metadata.
        morphism : str
            The morphism the site draws through, if any.

        Returns
        -------
        tuple[tuple[IRArg, ...], tuple[str, ...]]
            The arguments in the family's constructor order and their
            names.
        """
        pre: list[IRArg] = []
        names: list[str] = []
        for name, value in distribution.arguments:
            pre.append(self._arg(value, site, name, plate, morphism))
            names.append(name)
            self._tensor_parameter_input(distribution.name, name, value)
        if distribution.name in _FACTOR_FAMILIES and "scale_tril" in names:
            self._factor_scale(distribution.name, pre, names)
        if distribution.name == "InverseWishart" and "scale_tril" in names:
            self._inverse_wishart_scale(pre, names)
        ordered = _arg_names_for(meta, tuple(pre), self.ctx)
        if set(ordered) >= set(names) and len(ordered) >= len(names):
            order = {name: index for index, name in enumerate(ordered)}
            pairs = sorted(
                zip(names, pre, strict=True), key=lambda item: order[item[0]]
            )
            names = [name for name, _ in pairs]
            pre = [arg for _, arg in pairs]
        constraints = _resolve_arg_constraints(meta, tuple(pre), self.ctx)
        event_axes = tuple(
            dim.name for dim in plate.event_dims if isinstance(dim, DimStatic)
        )
        step = self.site_steps.get(site)
        axes_index = step.index if step is not None else None
        out = tuple(
            _wrap_for_constraint(
                arg, constraints.get(name), event_axes, axes_index, self.ctx
            )
            for arg, name in zip(pre, names, strict=True)
        )
        return out, tuple(names)

    def _parameterization(
        self,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        *,
        marginalized: bool = False,
    ) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
        """Hand an alternative parameterization to the target as it reads it.

        A target whose library has the parameterization the source wrote
        receives it unchanged, under the keyword the renderer spells it
        with. A literal precision matrix reaches a target without a
        precision parameterization inverted, as the covariance it is the
        inverse of. A marginalized latent given on the logit scale its
        family's marginalize support weighs its atoms from reaches every
        target unchanged: no target draws it, and every renderer weighs
        the atoms from the logits in log space. Any other alternative a
        target has no spelling for is refused by name.

        Parameters
        ----------
        family : str
            The family.
        args : tuple[IRArg, ...]
            The call's wire arguments.
        arg_names : tuple[str, ...]
            The parameter each argument fills.
        marginalized : bool
            Whether the call is a marginalized latent's prior.

        Returns
        -------
        tuple[tuple[IRArg, ...], tuple[str, ...]]
            The arguments and names the target receives.

        Raises
        ------
        UnsupportedConstruct
            If the target has no spelling for the parameterization.
        """
        target = self.planner.target
        meta = FAMILY_META.get(family)
        support = marginalize_support(meta) if meta is not None else None
        for position, name in enumerate(arg_names):
            natives = _NATIVE_ALTERNATIVES.get((family, name))
            if natives is None or target in natives:
                continue
            if marginalized and support is not None and support.logit_arg == name:
                continue
            argument = args[position]
            if name == "precision_matrix" and isinstance(argument, IRArgMatrix):
                inverse = _literal_inverse(argument)
                if inverse is not None:
                    return (
                        tuple(
                            inverse if index == position else item
                            for index, item in enumerate(args)
                        ),
                        tuple(
                            "covariance_matrix" if index == position else item
                            for index, item in enumerate(arg_names)
                        ),
                    )
            raise UnsupportedConstruct(
                f"qvr-{target}",
                [
                    f"family:{family}:{name}: {target} has no {family} "
                    f"parameterized by its {name}"
                    + (
                        "; write the matrix as a literal, which is inverted "
                        "into the covariance"
                        if name == "precision_matrix"
                        else "; write the probability the logits name as probs"
                    )
                ],
            )
        return args, arg_names

    def _tensor_parameter_input(
        self, family: str, parameter: str, value: Value
    ) -> None:
        """Declare a host tensor filling a family's whole tensor parameter.

        A datum that is one value of a vector or matrix parameter, such as
        a multivariate normal's host mean or Cholesky factor, is declared
        over the parameter's own axes as event axes, so a target reads it
        as one vector or matrix rather than an array of scalars.

        Parameters
        ----------
        family : str
            The family.
        parameter : str
            The family parameter the value fills.
        value : Value
            The value.
        """
        record = FAMILIES.get(family)
        if not isinstance(value, Var) or record is None:
            return
        datum = self.data_parameters.get(value.local.name)
        if (
            datum is None
            or datum.per_row
            or parameter not in record.parameter_names
            or record.parameter(parameter).rank == 0
        ):
            return
        dims = _typed_dims(datum.type, datum.axes)
        if len(dims) != record.parameter(parameter).rank:
            return
        self.structured_inputs.setdefault(
            datum.name, (CSReal(), Plate(event_dims=dims, batch_dims=()))
        )

    def _inverse_wishart_scale(self, pre: list[IRArg], names: list[str]) -> None:
        """Hand an inverse Wishart's scale to the target as it reads it.

        A source's ``InverseWishart(df, scale_tril=L)`` draws the inverse
        of a ``Wishart(df, scale_tril=L)`` draw: in the conventional
        ``IW(df, Psi)`` the scale is ``Psi = (L L^T)^-1``. Stan, NumPyro,
        and Turing take ``Psi`` itself, under ``scale_matrix``; Pyro's
        grafted class takes the Cholesky factor of ``Psi``, under
        ``scale_tril``. A literal factor is converted here, in place.

        Parameters
        ----------
        pre : list[IRArg]
            The call's wire arguments, rewritten in place.
        names : list[str]
            The parameter each argument fills, rewritten in place.

        Raises
        ------
        UnsupportedConstruct
            If the factor is not a literal, or is singular.
        """
        position = names.index("scale_tril")
        factor = pre[position]
        scale = None
        if isinstance(factor, IRArgMatrix) and all(
            isinstance(entry, IRArgNumber)
            for row in factor.rows
            for entry in row.elements
        ):
            rows = [
                [cast(IRArgNumber, entry).value for entry in row.elements]
                for row in factor.rows
            ]
            product = [
                [sum(a * b for a, b in zip(left, right, strict=True)) for right in rows]
                for left in rows
            ]
            scale = _literal_inverse(_matrix_literal(product))
        if scale is None:
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [
                    "family:InverseWishart:scale_tril: the InverseWishart's "
                    "scale_tril is a Cholesky factor known only at run time; "
                    "write it as a matrix literal"
                ],
            )
        if self.planner.target == "pyro":
            cholesky = _literal_cholesky(scale)
            if cholesky is None:
                raise UnsupportedConstruct(
                    "qvr-pyro",
                    [
                        "family:InverseWishart:scale_tril: the scale the factor "
                        "names is not positive definite"
                    ],
                )
            pre[position] = cholesky
            return
        pre[position] = scale
        names[position] = "scale_matrix"

    def _factor_scale(self, family: str, pre: list[IRArg], names: list[str]) -> None:
        """Hand a Cholesky factor to the target as it reads it.

        A source writes the scale of a Wishart or the covariance of a
        multivariate normal as its lower Cholesky factor ``L``; most
        targets take the matrix ``L L^T`` itself, and no target's plan
        can size a factor known only at run time. A literal factor is
        multiplied out here, in place, so every target receives the
        matrix under ``covariance_matrix``.

        Parameters
        ----------
        family : str
            The family, one of `_FACTOR_FAMILIES`.
        pre : list[IRArg]
            The call's wire arguments, rewritten in place.
        names : list[str]
            The parameter each argument fills, rewritten in place.

        Raises
        ------
        UnsupportedConstruct
            If the factor is not a literal.
        """
        position = names.index("scale_tril")
        factor = pre[position]
        if isinstance(factor, IRArgMatrix) and all(
            isinstance(entry, IRArgNumber)
            for row in factor.rows
            for entry in row.elements
        ):
            rows = [
                [cast(IRArgNumber, entry).value for entry in row.elements]
                for row in factor.rows
            ]
            scale = [
                [sum(a * b for a, b in zip(left, right, strict=True)) for right in rows]
                for left in rows
            ]
            pre[position] = IRArgMatrix(
                rows=tuple(
                    IRArgList(elements=tuple(IRArgNumber(value=v) for v in row))
                    for row in scale
                )
            )
            names[position] = "covariance_matrix"
            return
        if self.planner.target in _RUNTIME_FACTOR_TARGETS:
            # The target's library takes the factor itself.
            return
        raise UnsupportedConstruct(
            f"qvr-{self.planner.target}",
            [
                f"family:{family}:scale_tril: the {family}'s scale_tril is a "
                f"Cholesky factor known only at run time, which "
                f"{self.planner.target} has no Cholesky parameterization of "
                f"{family} to take; write it as a matrix literal"
            ],
        )

    def _arg(
        self, value: Value, site: str, head: str, plate: Plate, morphism: str
    ) -> IRArg:
        """The wire form of one family argument.

        Parameters
        ----------
        value : Value
            The argument.
        site : str
            The site the call is at.
        head : str
            The family parameter the argument fills.
        plate : Plate
            The site's plate.
        morphism : str
            The morphism the site draws through, if any.

        Returns
        -------
        IRArg
            A literal, a reference, a list, a matrix, a broadcast, a
            kernel, or a reference to a head binding the argument is
            computed by.

        Raises
        ------
        UnsupportedConstruct
            If the argument has no wire form.
        """
        if isinstance(value, PrimitiveApplication) and value.name in _TRANSPARENT:
            return self._arg(value.arguments[0], site, head, plate, morphism)
        if isinstance(value, Var):
            return IRArgRef(name=value.local.name, indices=())
        if isinstance(value, LiteralValue) and isinstance(value.value, int | float):
            return IRArgNumber(value=float(value.value))
        if isinstance(value, Comprehension):
            spread = _spread_literal(value)
            if spread is not None:
                literal, shape = spread
                return IRArgBroadcast(
                    value=IRArgNumber(value=float(literal)), target_shape=shape
                )
        if isinstance(value, TensorValue):
            if value.items and all(
                isinstance(item, TensorValue) for item in value.items
            ):
                return IRArgMatrix(
                    rows=tuple(
                        IRArgList(
                            elements=tuple(
                                self._arg(entry, site, head, plate, morphism)
                                for entry in row.items
                            )
                        )
                        for row in value.items
                        if isinstance(row, TensorValue)
                    )
                )
            return IRArgList(
                elements=tuple(
                    self._arg(item, site, head, plate, morphism) for item in value.items
                )
            )
        if isinstance(value, Gather):
            if (
                self.via is not None
                and isinstance(value.index, Var)
                and value.index.local.name == self.via
            ):
                return self._arg(value.value, site, head, plate, morphism)
            base = self._arg(value.value, site, head, plate, morphism)
            index = self._arg(value.index, site, head, plate, morphism)
            if isinstance(base, IRArgRef):
                return IRArgRef(name=base.name, indices=(*base.indices, index))
        if isinstance(value, KernelMatrix):
            inputs = value.inputs
            if isinstance(inputs, Var):
                shape = tensor_shape(inputs.local.type)
                size = (
                    shape[1][0].value
                    if shape is not None
                    and shape[1]
                    and isinstance(shape[1][0], IndexLiteral)
                    and isinstance(shape[1][0].value, int)
                    else 0
                )
                return IRArgKernel(
                    kernel=value.kernel,
                    length_scale=value.length_scale,
                    x_name=inputs.local.name,
                    grid_size=size,
                    jitter=value.jitter,
                )
        if isinstance(value, AffineMap):
            return self._affine_head(value, site, head, plate, morphism)
        if isinstance(value, TableMap):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [f"param-source:table:{morphism}"],
            )
        if isinstance(value, DistributionValue):
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"nested-distribution-arg:{value.name}: a "
                    "distribution-valued draw argument is not "
                    "representable in this backend's IR"
                ],
            )
        raise UnsupportedConstruct(
            f"qvr-{self.planner.target}",
            [f"qiec:argument:{type(value).__name__}:{site}:{head}"],
        )

    def _affine_head(
        self, value: AffineMap, site: str, head: str, plate: Plate, morphism: str
    ) -> IRArg:
        """Bind one head of a parameter map and refer to it.

        Parameters
        ----------
        value : AffineMap
            The head.
        site : str
            The site whose argument the head fills.
        head : str
            The family parameter the head fills.
        plate : Plate
            The site's plate.
        morphism : str
            The morphism carrying the map.

        Returns
        -------
        IRArg
            A reference to the head's binding.

        Raises
        ------
        UnsupportedConstruct
            If the map's weight or bias is not a program input, or a
            source is not a binding.
        """
        if not isinstance(value.weight, Var) or not isinstance(value.bias, Var):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [f"param-source:linear:map:{morphism}"],
            )
        sources: list[LetAffineSource] = []
        for source in value.sources:
            if not isinstance(source, Var):
                raise UnsupportedConstruct(
                    f"qvr-{self.planner.target}",
                    [f"param-source:linear:argument:{morphism}"],
                )
            shape = tensor_shape(source.local.type)
            width = (
                int(shape[1][0].value)
                if shape is not None
                and len(shape[1]) == 1
                and isinstance(shape[1][0], IndexLiteral)
                else 1
            )
            sources.append(
                LetAffineSource(value=LetExprVar(name=source.local.name), width=width)
            )
        weight_shape = tensor_shape(value.weight.local.type)
        bias_shape = tensor_shape(value.bias.local.type)
        if weight_shape is not None and bias_shape is not None:
            self.param_map_inputs[value.weight.local.name] = Plate(
                event_dims=(),
                batch_dims=tuple(
                    _dim(size, f"{morphism}_param_{axis}")
                    for size, axis in zip(weight_shape[1], ("row", "col"), strict=False)
                ),
            )
            self.param_map_inputs[value.bias.local.name] = Plate(
                event_dims=(),
                batch_dims=(_dim(bias_shape[1][0], f"{morphism}_param_row"),),
            )
        transform = value.transform
        param_head = _ParamHead(
            arg_name=head,
            transform="exp_floor" if transform == "exp_floor" else "identity",
        )
        name = _head_binding_name(site, param_head)
        if transform == "identity":
            self.head_nodes.append(
                IRDeterministic(
                    name=name,
                    expr=LetExprAffineMap(
                        weight=LetExprVar(name=value.weight.local.name),
                        bias=LetExprVar(name=value.bias.local.name),
                        sources=tuple(sources),
                        row_offset=value.row_offset,
                        rows=value.rows,
                        transform="identity",
                    ),
                    constraint=CSReal(),
                    plate=plate,
                )
            )
            return IRArgRef(name=name, indices=())
        if transform != "exp_floor":
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [f"param-source:linear:transform:{transform}:{morphism}"],
            )
        raw = _head_raw_binding_name(site, param_head)
        axis = plate.batch_dims[-1] if plate.batch_dims else None
        axis_name = axis.name if isinstance(axis, DimStatic) else morphism
        self.head_nodes.append(
            IRDeterministic(
                name=raw,
                expr=LetExprAffineMap(
                    weight=LetExprVar(name=value.weight.local.name),
                    bias=LetExprVar(name=value.bias.local.name),
                    sources=tuple(sources),
                    row_offset=value.row_offset,
                    rows=value.rows,
                    transform="exp",
                ),
                constraint=CSReal(),
                plate=plate,
            )
        )
        self.head_nodes.append(
            IRDeterministic(
                name=name,
                expr=LetExprFactor(
                    binders=(
                        LetFactorBinder(
                            var=f"_{axis_name.lower()}_i",
                            index=TypeName(name=axis_name),
                        ),
                    ),
                    cases=tuple(
                        LetFactorCase(label=index, value=_floor_expr(raw, index))
                        for index in range(value.rows)
                    ),
                ),
                constraint=CSPositive(),
                plate=plate,
            )
        )
        return IRArgRef(name=name, indices=())

    def _structured_args(
        self, distribution: DistributionValue, site: str, plate: Plate
    ) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
        """The wire arguments of a structured family.

        A Gaussian process reads its mean as one literal and its
        covariance as a kernel argument; a multivariate or matrix
        normal reads its location and its scale factors as references
        to the inputs the construction names.

        Parameters
        ----------
        distribution : DistributionValue
            The construction.
        site : str
            The site the call is at.
        plate : Plate
            The site's plate.

        Returns
        -------
        tuple[tuple[IRArg, ...], tuple[str, ...]]
            The arguments and the constructor names they fill.
        """
        meta = FAMILY_META[distribution.name]
        assert meta.structured_lowering is not None
        named = dict(distribution.arguments)
        registry = _REGISTRY_NAMES.get(distribution.name, {})
        args: list[IRArg] = []
        names: list[str] = []
        for spec in meta.structured_lowering.args:
            arg_name = str(spec.arg_name)
            value = named.get(registry.get(arg_name, arg_name))
            if value is None:
                raise UnsupportedConstruct(
                    f"qvr-{self.planner.target}",
                    [f"family:{distribution.name}:structured_arg:{arg_name}"],
                )
            if isinstance(spec, StructuredZeroVectorArg):
                args.append(IRArgNumber(value=0.0))
                names.append(arg_name)
                continue
            arg = self._arg(value, site, arg_name, plate, "")
            if isinstance(spec, StructuredDataArg) and isinstance(arg, IRArgRef):
                dims = tuple(plate.event_dims[i] for i in spec.axis_indices)
                self.structured_inputs[arg.name] = (
                    _DATA_CONSTRAINT_FACTORY[spec.constraint_kind](),
                    Plate(event_dims=dims, batch_dims=()),
                )
            args.append(arg)
            names.append(arg_name)
        return tuple(args), tuple(names)

    def _class_index(
        self,
        morphism: str,
        meta: FamilyMeta,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        constraint: ConstraintSpec,
        site: str,
    ) -> tuple[tuple[IRArg, ...], ConstraintSpec]:
        """Restate a class-index draw's alphabet from its codomain.

        Parameters
        ----------
        morphism : str
            The morphism the site draws through, if any.
        meta : FamilyMeta
            The family's metadata.
        args : tuple[IRArg, ...]
            The arguments.
        arg_names : tuple[str, ...]
            Their names.
        constraint : ConstraintSpec
            The support read off the family.
        site : str
            The site.

        Returns
        -------
        tuple[tuple[IRArg, ...], ConstraintSpec]
            The arguments with the alphabet widened and the support
            over the codomain's elements.
        """
        outcome = class_index_outcome(meta)
        if outcome is None or not morphism:
            return args, constraint
        alphabet = _codomain_alphabet(morphism, self.ctx)
        if alphabet is None:
            return args, constraint
        width = alphabet.size
        if width < 2:
            raise UnsupportedConstruct(
                "qvr-lower",
                [
                    f"class-index:{site}:codomain-width:{width}: a "
                    f"{meta.qvr_name} draw through {morphism!r} "
                    f"needs a codomain naming at least two classes"
                ],
            )
        arg_dim = DimStatic(size=width - outcome.extent_offset, name=alphabet.name)
        out = list(args)
        for index, name in enumerate(arg_names):
            if name in outcome.alphabet_args:
                out[index] = _retarget_alphabet_arg(out[index], arg_dim, site, self.ctx)
        return tuple(out), CSIntegerInterval(lower=0, upper=width - 1)

    # ------------------------------------------------------------------
    # pure expressions

    def _expr(self, value: Value) -> LetExprNode:
        """A pure value as a let expression.

        Parameters
        ----------
        value : Value
            The value.

        Returns
        -------
        LetExprNode
            The expression the renderers translate.
        """
        return _expression(
            value, self.ctx.cards, self.planner.target, self.program.name
        )

    # ------------------------------------------------------------------
    # inputs

    def inputs(self, body: tuple[IRNode, ...]) -> tuple[IRDataInput, ...]:
        """The plan's inputs.

        Parameters
        ----------
        body : tuple[IRNode, ...]
            The derived body, whose observations fix their inputs'
            supports and plates.

        Returns
        -------
        tuple[IRDataInput, ...]
            One input per entry-point parameter, grouped by role as
            the targets declare them: the program's domain, its
            scalars, fibrations, observations, parameter maps, and
            data.
        """
        groups: dict[str, list[IRDataInput]] = {
            "domain": [],
            "scalar": [],
            "fibration": [],
            "observation": [],
            "map": [],
            "data": [],
        }
        for parameter in self.entry.parameters:
            node = self._input(parameter, body)
            if parameter.role in ("weight", "bias", "table"):
                groups["map"].append(node)
            elif parameter.role in ("kernel-input",):
                groups["data"].append(node)
            else:
                groups[parameter.role].append(node)
        return tuple(
            node
            for role in ("domain", "scalar", "fibration", "observation", "map", "data")
            for node in groups[role]
        )

    def _input(
        self, parameter: ProgramParameter, body: tuple[IRNode, ...]
    ) -> IRDataInput:
        """One input.

        Parameters
        ----------
        parameter : ProgramParameter
            The entry-point parameter.
        body : tuple[IRNode, ...]
            The derived body.

        Returns
        -------
        IRDataInput
            The input, typed by the parameter and constrained by its
            role and use.
        """
        name = parameter.name
        if parameter.role == "observation" and name in self.observed:
            node = self.observed[name]
            return IRDataInput(name=name, constraint=node.constraint, plate=node.plate)
        if parameter.role == "fibration":
            upper = None
            batch: tuple[Dim, ...] = ()
            for node in _walk(body):
                if isinstance(node, IRObserve) and node.via == name:
                    batch = node.plate.batch_dims
                    first = batch[0] if batch else None
                    upper = first.size if isinstance(first, DimStatic) else None
                    break
            constraint: ConstraintSpec = (
                CSNonnegativeInteger()
                if upper is None
                else CSIntegerInterval(lower=0, upper=max(upper - 1, 0))
            )
            return IRDataInput(
                name=name,
                constraint=constraint,
                plate=Plate(event_dims=(), batch_dims=batch),
            )
        if name in self.structured_inputs:
            constraint, plate = self.structured_inputs[name]
            return IRDataInput(name=name, constraint=constraint, plate=plate)
        if name in self.param_map_inputs:
            return IRDataInput(
                name=name, constraint=CSReal(), plate=self.param_map_inputs[name]
            )
        dims = _typed_dims(parameter.type, parameter.axes)
        if (
            parameter.per_row
            and len(dims) > 1
            and self.planner.target not in _PER_ROW_VECTOR_TARGETS
        ):
            raise UnsupportedConstruct(
                f"qvr-{self.planner.target}",
                [
                    f"data:per-row-vector-parameter:{name}: the input fills a "
                    f"vector-valued family parameter with one vector per row of "
                    f"a plate, which {self.planner.target} has no rendering for"
                ],
            )
        if parameter.role == "domain":
            # The program's input is one value of its domain: the
            # factor's width is the input's event.
            plate = Plate(event_dims=dims, batch_dims=())
            return IRDataInput(
                name=name,
                constraint=_apply_declared_bounds(CSReal(), plate, self.ctx, name),
                plate=plate,
            )
        plate = Plate(event_dims=(), batch_dims=dims)
        element = _element_type(parameter.type)
        constraint = CSNonnegativeInteger() if element == INT else CSReal()
        return IRDataInput(name=name, constraint=constraint, plate=plate)


# ---------------------------------------------------------------------------
# module-level helpers


def _collect_site_steps(
    steps: Sequence[ProgramStep],
    out: dict[str, SampleStep | ObserveStep | MarginalizeStep],
) -> None:
    """Index a program's draws by the site each binds.

    Parameters
    ----------
    steps : Sequence[ProgramStep]
        The steps, searched through marginalization scopes.
    out : dict[str, SampleStep | ObserveStep | MarginalizeStep]
        The index, extended in place.
    """
    for step in steps:
        if isinstance(step, SampleStep):
            for name in step.vars:
                out[name] = step
        elif isinstance(step, ObserveStep):
            for name in step.vars:
                out[name] = step
        elif isinstance(step, MarginalizeStep):
            out[step.var] = step
            _collect_site_steps(step.scope, out)


def _collect_factor_axes(
    steps: Sequence[ProgramStep], out: dict[str, tuple[str, ...]]
) -> None:
    """Index a program's ``factor`` lets by the objects their binders range over.

    Parameters
    ----------
    steps : Sequence[ProgramStep]
        The steps, searched through marginalization scopes.
    out : dict[str, tuple[str, ...]]
        The index, extended in place: each let whose value is a
        ``factor`` over named objects, mapped to those objects in
        binder order.
    """
    for step in steps:
        if isinstance(step, LetStep) and isinstance(step.value, LetExprFactor):
            indices = tuple(binder.index for binder in step.value.binders)
            if all(isinstance(index, TypeName) for index in indices):
                out[step.name] = tuple(cast(TypeName, index).name for index in indices)
        elif isinstance(step, MarginalizeStep):
            _collect_factor_axes(step.scope, out)


def _site_and_distribution(
    request: EffectRequest, target: str
) -> tuple[SiteValue, DistributionValue]:
    """The site and construction of a sample request.

    Parameters
    ----------
    request : EffectRequest
        The request.
    target : str
        The transpile target, for the diagnostic.

    Returns
    -------
    tuple[SiteValue, DistributionValue]
        The site and the family construction drawn from.

    Raises
    ------
    UnsupportedConstruct
        If the request's arguments are not a site and a construction.
    """
    if (
        len(request.arguments) == 2
        and isinstance(request.arguments[0], SiteValue)
        and isinstance(request.arguments[1], DistributionValue)
    ):
        return request.arguments[0], request.arguments[1]
    raise UnsupportedConstruct(f"qvr-{target}", ["qiec:sample:unstructured"])


def _returned_names(value: Value) -> tuple[str, ...] | None:
    """The names a return value lists.

    Parameters
    ----------
    value : Value
        The returned value.

    Returns
    -------
    tuple[str, ...] | None
        The bound names, or ``None`` when the value is not a binding
        or a tuple of bindings.
    """
    if isinstance(value, Var):
        return (value.local.name,)
    if isinstance(value, TupleValue) and all(
        isinstance(item, Var) for item in value.items
    ):
        return tuple(item.local.name for item in value.items if isinstance(item, Var))
    return None


def _marginal_reduction(handler: str) -> str:
    """The reduction an enumeration handler's name records.

    Parameters
    ----------
    handler : str
        The handler's name.

    Returns
    -------
    str
        ``logsumexp``, ``sum``, or ``mean``.
    """
    if handler.endswith("_sum"):
        return "sum"
    if handler.endswith("_mean"):
        return "mean"
    return "logsumexp"


def _dim_names(plate: Plate) -> tuple[str, ...]:
    """The axis names of a plate, batch dimensions first.

    Parameters
    ----------
    plate : Plate
        The plate.

    Returns
    -------
    tuple[str, ...]
        One name per dimension.
    """
    return tuple(str(dim.name) for dim in (*plate.batch_dims, *plate.event_dims))


#: The targets that render a family parameter filled by host data holding
#: one vector per row of the draw's plate: their array libraries read the
#: per-row matrix and broadcast it against the plate as the checked
#: program does. Every other target refuses such a program.
_PER_ROW_VECTOR_TARGETS: frozenset[str] = frozenset({"edward2", "numpyro", "pyro"})


def _dim(size: IndexTerm, name: str) -> Dim:
    """A plate dimension from an index term.

    Parameters
    ----------
    size : IndexTerm
        The extent.
    name : str
        The axis's name.

    Returns
    -------
    Dim
        A static dimension for a literal extent, a dynamic one for a
        variable.
    """
    if isinstance(size, IndexLiteral) and isinstance(size.value, int):
        return DimStatic(size=size.value, name=name)
    if isinstance(size, IndexVariable):
        return DimDynamic(size_name=size.name, name=name)
    raise UnsupportedConstruct("qvr-lower", [f"axis:unknown-cardinality:{name}"])


def _renamed_dim(dim: Dim, name: str) -> Dim:
    """A plate dimension of the same extent under another name.

    Parameters
    ----------
    dim : Dim
        The dimension.
    name : str
        The new name.

    Returns
    -------
    Dim
        The dimension, named ``name``.

    Raises
    ------
    TypeError
        If ``dim`` is neither a static nor a dynamic dimension.
    """
    if isinstance(dim, DimStatic):
        return DimStatic(size=dim.size, name=name)
    if isinstance(dim, DimDynamic):
        return DimDynamic(size_name=dim.size_name, name=name)
    raise TypeError(f"unknown plate dimension {type(dim).__name__}")


def _axis_dim(axis: PlateAxis) -> Dim:
    """A plate dimension from a plate axis.

    Parameters
    ----------
    axis : PlateAxis
        The axis.

    Returns
    -------
    Dim
        The dimension, named after the axis.
    """
    return _dim(axis.size, axis.name)


def _plate_of(
    plate: PlateShape,
    marginalized: bool,
    meta: FamilyMeta,
    distribution: DistributionValue,
    walk: _ProgramWalk,
) -> Plate:
    """The IR plate of a construction.

    Parameters
    ----------
    plate : PlateShape
        The construction's plate.
    marginalized : bool
        Whether the construction is a marginalized latent's.
    meta : FamilyMeta
        The family's metadata.
    distribution : DistributionValue
        The construction.
    walk : _ProgramWalk
        The walk, for the support tables.

    Returns
    -------
    Plate
        Batch and event dimensions named after the axes.
    """
    del marginalized, walk
    event = tuple(_axis_dim(axis) for axis in plate.event)
    record = FAMILIES.get(distribution.name)
    if not plate.event and record is not None and record.name in _SAMPLE_EVENT_FAMILIES:
        # A multivariate normal's or a Wishart's own event axes are those
        # of its sample, which an unplated construction's type states.
        sampled = sampled_element(distribution.result_type)
        dims = _typed_dims(sampled, ()) if sampled is not None else ()
        if len(dims) >= record.event_rank:
            event = dims[len(dims) - record.event_rank :]
    if meta.qvr_name == "LKJCholesky" and len(event) == 2 and event[0] == event[1]:
        # A correlation factor is square over one axis, which the
        # plan states once.
        event = event[:1]
    return Plate(
        event_dims=event,
        batch_dims=tuple(_axis_dim(axis) for axis in plate.batch),
    )


def _typed_dims(type_: TypeExpr, axes: tuple[str, ...]) -> tuple[Dim, ...]:
    """The dimensions a tensor type states.

    Parameters
    ----------
    type_ : TypeExpr
        The type.
    axes : tuple[str, ...]
        The dimensions' names, when known.

    Returns
    -------
    tuple[Dim, ...]
        One dimension per tensor axis, named by ``axes`` or by
        position; none for a scalar.
    """
    shape = tensor_shape(type_)
    if shape is None:
        return ()
    return tuple(
        _dim(size, axes[index] if index < len(axes) else f"axis{index}")
        for index, size in enumerate(shape[1])
    )


def _element_type(type_: TypeExpr) -> TypeExpr:
    """The element of a tensor type, or the type itself.

    Parameters
    ----------
    type_ : TypeExpr
        The type.

    Returns
    -------
    TypeExpr
        The scalar type.
    """
    shape = tensor_shape(type_)
    return shape[0] if shape is not None else type_


#: The families whose ``scale_tril`` factor a target receives multiplied
#: out, as the matrix it names the factor of.
_FACTOR_FAMILIES: frozenset[str] = frozenset({"MultivariateNormal", "Wishart"})

#: The targets whose libraries take a multivariate normal's and a
#: Wishart's Cholesky factor itself, so a factor known only at run time
#: reaches them as written: Stan's `multi_normal_cholesky` and a Wishart
#: scale `multiply_lower_tri_self_transpose(L)`, the `scale_tril=` of
#: NumPyro and Pyro, and PyMC's `chol=` with a Wishart scale `L @ L.T`.
_RUNTIME_FACTOR_TARGETS: frozenset[str] = frozenset(
    {"ir", "numpyro", "pymc", "pyro", "stan"}
)

_EVERY_TARGET: frozenset[str] = frozenset(
    {
        "ir",
        "bugs",
        "church",
        "edward2",
        "gen",
        "jags",
        "numpyro",
        "pymc",
        "pyro",
        "stan",
        "turing",
        "webppl",
    }
)

#: The targets whose library takes each alternative parameterization as
#: written, keyed by family and parameter. Every other target receives a
#: literal precision matrix inverted and refuses anything else.
_NATIVE_ALTERNATIVES: dict[tuple[str, str], frozenset[str]] = {
    ("Bernoulli", "logits"): frozenset(
        {"ir", "edward2", "numpyro", "pymc", "pyro", "stan", "turing"}
    ),
    ("Binomial", "logits"): frozenset(
        {"ir", "edward2", "numpyro", "pymc", "pyro", "stan", "turing"}
    ),
    ("Geometric", "logits"): frozenset({"ir", "edward2", "numpyro", "pyro"}),
    ("ContinuousBernoulli", "logits"): frozenset({"ir"}),
    ("NegativeBinomial", "logits"): frozenset({"ir", "edward2", "pyro"}),
    ("MultivariateNormal", "covariance_matrix"): _EVERY_TARGET,
    ("MultivariateNormal", "precision_matrix"): frozenset(
        {"ir", "bugs", "jags", "numpyro", "pymc", "pyro", "stan"}
    ),
    ("Wishart", "covariance_matrix"): _EVERY_TARGET,
    ("Wishart", "precision_matrix"): frozenset({"ir", "numpyro", "pyro"}),
}


def _matrix_literal(rows: list[list[float]]) -> IRArgMatrix:
    """A literal matrix argument from its rows of numbers.

    Parameters
    ----------
    rows : list[list[float]]
        The rows.

    Returns
    -------
    IRArgMatrix
        The literal.
    """
    return IRArgMatrix(
        rows=tuple(
            IRArgList(elements=tuple(IRArgNumber(value=value) for value in row))
            for row in rows
        )
    )


def _literal_cholesky(matrix: IRArgMatrix) -> IRArgMatrix | None:
    """The lower Cholesky factor of a literal positive-definite matrix.

    Parameters
    ----------
    matrix : IRArgMatrix
        The matrix, square and symmetric with numeric entries.

    Returns
    -------
    IRArgMatrix | None
        Its lower Cholesky factor in double precision, or ``None`` when
        an entry is not a number or the matrix is not positive definite.
    """
    if not all(
        isinstance(entry, IRArgNumber) for row in matrix.rows for entry in row.elements
    ):
        return None
    rows = [
        [cast(IRArgNumber, entry).value for entry in row.elements]
        for row in matrix.rows
    ]
    size = len(rows)
    factor = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            partial = rows[i][j] - sum(factor[i][k] * factor[j][k] for k in range(j))
            if i == j:
                if partial <= 0.0:
                    return None
                factor[i][j] = partial**0.5
            else:
                factor[i][j] = partial / factor[j][j]
    return _matrix_literal(factor)


def _literal_inverse(matrix: IRArgMatrix) -> IRArgMatrix | None:
    """The inverse of a literal matrix, by Gauss-Jordan elimination.

    Parameters
    ----------
    matrix : IRArgMatrix
        The matrix, square with numeric entries.

    Returns
    -------
    IRArgMatrix | None
        Its inverse in double precision, or ``None`` when an entry is not
        a number, the matrix is not square, or it is singular.
    """
    if not all(
        isinstance(entry, IRArgNumber) for row in matrix.rows for entry in row.elements
    ):
        return None
    rows = [
        [cast(IRArgNumber, entry).value for entry in row.elements]
        for row in matrix.rows
    ]
    size = len(rows)
    if any(len(row) != size for row in rows):
        return None
    work = [
        [*row, *(1.0 if column == index else 0.0 for column in range(size))]
        for index, row in enumerate(rows)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if work[pivot][column] == 0.0:
            return None
        work[column], work[pivot] = work[pivot], work[column]
        lead = work[column][column]
        work[column] = [value / lead for value in work[column]]
        for row in range(size):
            if row != column and work[row][column] != 0.0:
                factor = work[row][column]
                work[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(work[row], work[column], strict=True)
                ]
    return IRArgMatrix(
        rows=tuple(
            IRArgList(elements=tuple(IRArgNumber(value=value) for value in row[size:]))
            for row in work
        )
    )


#: The families an unplated construction of which is planned over its
#: sample's own event axes, as an ``[over=...]`` clause would name them.
_SAMPLE_EVENT_FAMILIES: frozenset[str] = frozenset(
    {"MultivariateNormal", "LowRankMVN", "Wishart"}
)


def _structured(distribution: DistributionValue, meta: FamilyMeta) -> bool:
    """Whether a construction is a structured family's.

    A Gaussian process reads its covariance as a kernel argument; a
    multivariate or matrix normal drawn through a morphism reads its
    location and scale from inputs the elaboration names after the
    family's structured arguments. A multivariate normal whose source
    writes its own arguments under another parameterization, such as a
    ``scale_tril`` factor, is an ordinary family call.

    Parameters
    ----------
    distribution : DistributionValue
        The construction.
    meta : FamilyMeta
        The family's transpile metadata.

    Returns
    -------
    bool
        ``True`` when an argument is a kernel matrix, or the family
        reads its factors structurally and the construction supplies
        every structured argument by name.
    """
    if any(isinstance(value, KernelMatrix) for _, value in distribution.arguments):
        return True
    if distribution.name not in ("GP", "MultivariateNormal", "MatrixNormal"):
        return False
    lowering = meta.structured_lowering
    if lowering is None:
        return False
    registry = _REGISTRY_NAMES.get(distribution.name, {})
    supplied = {name for name, _ in distribution.arguments}
    return all(
        registry.get(str(spec.arg_name), str(spec.arg_name)) in supplied
        for spec in lowering.args
        if isinstance(spec, StructuredDataArg)
    )


def _spread_literal(value: Comprehension) -> tuple[float, tuple[int, ...]] | None:
    """A literal spread over a vector, as the literal and the shape.

    A single literal a construction spreads over a vector parameter,
    ``Dirichlet(1.0)`` over an axis, is a comprehension of the literal,
    which is a broadcast on the wire.

    Parameters
    ----------
    value : Comprehension
        The comprehension.

    Returns
    -------
    tuple[float, tuple[int, ...]] | None
        The literal and the shape, or ``None`` when the body is not a
        literal or a spread itself, or an extent is open.
    """
    if not isinstance(value.extent, IndexLiteral) or not isinstance(
        value.extent.value, int
    ):
        return None
    body = value.body
    if isinstance(body, LiteralValue) and isinstance(body.value, int | float):
        return float(body.value), (value.extent.value,)
    if isinstance(body, Comprehension):
        inner = _spread_literal(body)
        if inner is not None:
            literal, shape = inner
            return literal, (value.extent.value, *shape)
    return None


def _lower_static(argument: StaticArgument) -> IRQiecStatic:
    """A static argument as structural IR.

    Parameters
    ----------
    argument : StaticArgument
        The argument.

    Returns
    -------
    IRQiecStatic
        Its IR form.
    """
    lowered = _convert(argument)
    assert isinstance(lowered, IRQiecStatic)
    return lowered


def _referenced(value: Value) -> Iterator[str]:
    """The bindings a value reads, in order of appearance.

    Parameters
    ----------
    value : Value
        The value.

    Yields
    ------
    str
        Each referenced local's name.
    """
    if isinstance(value, Var):
        yield value.local.name
        return
    for name in getattr(value, "__dataclass_fields__", {}):
        field = getattr(value, name)
        if isinstance(field, tuple):
            for item in field:
                if hasattr(item, "__dataclass_fields__"):
                    yield from _referenced(item)  # type: ignore[arg-type]
        elif hasattr(field, "__dataclass_fields__"):
            yield from _referenced(field)  # type: ignore[arg-type]


def _walk(body: tuple[IRNode, ...]) -> Iterator[IRNode]:
    """Every node of a body, descending into scopes.

    Parameters
    ----------
    body : tuple[IRNode, ...]
        The body.

    Yields
    ------
    IRNode
        The nodes in order.
    """
    for node in body:
        yield node
        if isinstance(node, IRMarginalize):
            yield from _walk(node.scope)


def checked_module(
    module: Module, *, target: str, require_target_runtime: bool = True
) -> QiecModule | None:
    """Elaborate and check a module at the lowering boundary.

    A source with nothing to elaborate takes no checking path. Otherwise
    the elaboration constructs the complete
    :class:`~quivers.qiec.QiecModule` and ``validate_module`` independently
    checks its nominal declarations, static scopes, effect rows, and every
    computation type. Capability analysis is deliberately separate: the
    lowering always retains the complete typed module, and a selected
    renderer uses
    :func:`quivers.transpile.qiec_ir.analyze_qiec_capabilities` to determine
    whether it can preserve each feature.

    Parameters
    ----------
    module : Module
        The parsed module.
    target : str
        The transpile target, for diagnostics.
    require_target_runtime : bool
        Whether to reject host-only runtime capabilities before deriving a
        target plan. Disable this only when retaining the checked QIEC graph
        for reference execution without a target-oriented plan.

    Returns
    -------
    QiecModule | None
        The checked module, or ``None`` when the source has nothing to
        elaborate.

    Raises
    ------
    UnsupportedConstruct
        If a program uses a form the elaboration does not admit; the
        construct is reported the way a renderer reports one it cannot
        emit.
    QiecDiagnosticError
        If a QIEC declaration is rejected.
    """
    if not has_qiec_surface(module):
        return None
    try:
        qiec_module: QiecModule = lower_qvr_to_qiec(module)
    except QiecDiagnosticError as error:
        if error.code == GAP_CODE:
            # A program using a chart or a network morphism has no
            # elaboration yet. The module's other declarations still
            # lower; the program itself reaches the renderer through its
            # plan alone, and the gap is recorded on the plan.
            qiec_module = lower_qvr_to_qiec(module, elaborate_programs=False)
            validate_module(qiec_module)
            return replace(qiec_module, gap=error.message)
        if error.code != "qiec-program" and error.program is None:
            raise
        raise UnsupportedConstruct(
            f"qvr-{target}", [_elaboration_kind(error.code, error.message)]
        ) from error
    validate_module(qiec_module)
    if require_target_runtime:
        _refuse_host_runtime_calls(qiec_module, target)
    return qiec_module


#: A diagnostic that leads with a structured kind, `family:X: ...` or
#: `let:call:unknown:x; ...`, spells its own kind.
_STRUCTURED_LEAD = re.compile(r"^[a-z][a-z-]*:[^\s]")


def _elaboration_kind(code: str, message: str) -> str:
    """The refusal kind an elaboration diagnostic reports as.

    Parameters
    ----------
    code : str
        The diagnostic's code.
    message : str
        Its message.

    Returns
    -------
    str
        The message itself when it leads with a structured kind, else
        ``program:elaboration:<code>: <message>``, whose explanation
        is the message.
    """
    if _STRUCTURED_LEAD.match(message):
        return message
    return f"program:elaboration:{code}: {message}"


def _refuse_host_runtime_calls(qiec_module: QiecModule, target: str) -> None:
    """Refuse computations that require an unavailable host runtime.

    A deduction needs the search runtime. A structural encoder, decoder,
    or loss needs its neural attachment. Neither belongs to a target's
    probabilistic runtime, so the boundary reports the exact computation
    and capability before target-specific lowering.

    Parameters
    ----------
    qiec_module : QiecModule
        The checked module.
    target : str
        The transpile target, for the diagnostic.

    Raises
    ------
    UnsupportedConstruct
        If a reachable computation needs search or a neural attachment.
    """
    kinds = [
        diagnostic.kind
        for diagnostic in host_runtime_capability_diagnostics(qiec_module, target)
    ]
    if kinds:
        raise UnsupportedConstruct(f"qvr-{target}", kinds)


class Lower(dx.Mapping[Module, IRProgram]):
    """Map a parsed [`Module`][quivers.dsl.ast_nodes.Module] to an
    [`IRProgram`][quivers.transpile.ir.IRProgram].

    The lowering elaborates the module through the QIEC route first;
    the program's plan is then derived from its checked computation.
    A module without a program lowers to its checked computations
    alone. Stateless: every table is derived from the input module on
    each call.
    """

    def forward(
        self, module: Module, *, target: str = "ir", optimize: bool = True
    ) -> IRProgram:
        """Lower a parsed module to the transpile IR.

        Parameters
        ----------
        module : Module
            The parsed module, with composite lets already expanded for
            the target.
        target : str
            The transpile target the lowering serves, named in the
            diagnostics the QIEC boundary raises.
        optimize : bool
            Whether to derive the target-oriented probabilistic plan. When
            false, retain only the checked QIEC graph so reference execution
            can test semantics before target-plan recognition or rewriting.

        Returns
        -------
        IRProgram
            The lowered program with its checked kernel module.

        Raises
        ------
        UnsupportedConstruct
            If the module has no program and no QIEC surface, uses a
            construct the elaboration does not admit, or falls in a
            gap of the elaboration.
        QiecDiagnosticError
            If a QIEC declaration is rejected.
        """
        qiec_module = checked_module(
            module,
            target=target,
            require_target_runtime=optimize,
        )
        if qiec_module is None:
            # A source with nothing to elaborate has no program either;
            # report the established precise ``program:absent`` error.
            pick_program(module)
            raise UnsupportedConstruct(f"qvr-{target}", ["program:absent"])
        if not any(
            isinstance(statement, ProgramDecl) for statement in module.statements
        ):
            qiec_ir = lower_qiec_ir(qiec_module)
            return IRProgram(
                name=qiec_ir.module, inputs=(), body=(), module=qiec_ir, cards={}
            )
        if qiec_module.gap:
            kind = qiec_module.gap.partition("; ")[0]
            raise UnsupportedConstruct(f"qvr-{target}", [kind])
        if not optimize:
            expanded = expand_composite_lets(module, target="stan")
            program = pick_program(expanded)
            return IRProgram(
                name=program.name,
                inputs=(),
                body=(),
                module=lower_qiec_ir(qiec_module),
                cards={},
            )
        plan = checked_program_plan(qiec_module, module, target)
        assert plan is not None
        return plan


TARGET_NAME_SEPARATOR = "__"
"""How a target spells the ``$`` of a name a program draw renamed.

``theta$z`` is the site of the local ``z`` of a program drawn under
``theta``, and no target language admits ``$`` in an identifier.
"""


def target_name(name: str) -> str:
    """A source name as every target spells it.

    Parameters
    ----------
    name : str
        The name, as the reference machine labels it.

    Returns
    -------
    str
        The name with each ``$`` a program draw introduced spelled as
        ``TARGET_NAME_SEPARATOR``, two underscores.
    """
    return name.replace("$", TARGET_NAME_SEPARATOR)


def _spelled[NodeT: dx.Model](node: NodeT) -> NodeT:
    """A plan node with every name spelled as the targets spell it.

    Parameters
    ----------
    node : NodeT
        The node.

    Returns
    -------
    NodeT
        The node with ``$`` rewritten in every string field and every
        nested node, a string literal's text excepted.
    """
    if isinstance(node, LetExprString):
        return node
    changes: dict[str, str | dx.Model | tuple[str | dx.Model, ...]] = {}
    for field in type(node).__field_specs__:
        current = getattr(node, field)
        if isinstance(current, str):
            if "$" in current:
                changes[field] = target_name(current)
        elif isinstance(current, dx.Model):
            spelled = _spelled(current)
            if spelled is not current:
                changes[field] = spelled
        elif isinstance(current, tuple) and current:
            items = tuple(
                target_name(item)
                if isinstance(item, str)
                else _spelled(item)
                if isinstance(item, dx.Model)
                else item
                for item in current
            )
            if any(a is not b for a, b in zip(items, current, strict=True)):
                changes[field] = items
    if not changes:
        return node
    return node.with_(**changes)


def program_plan(
    module: QiecModule, source: Module, program: ProgramDecl, target: str
) -> IRProgram:
    """Derive a program's plan from its checked module.

    Parameters
    ----------
    module : QiecModule
        The checked module.
    source : Module
        The expanded source module.
    program : ProgramDecl
        The program.
    target : str
        The transpile target.

    Returns
    -------
    IRProgram
        The plan.
    """
    return _Planner(module, source, target).plan(program)


def checked_program_plan(
    module: QiecModule, source: Module, target: str
) -> IRProgram | None:
    """Derive a target plan from an existing checked module.

    This is the target-planning half of [`Lower.forward`][quivers.transpile.plan.Lower.forward]
    for callers such as the CLI and language server that already hold the checked
    QIEC module. Reusing it avoids parsing and elaborating the source a second time.

    Parameters
    ----------
    module : QiecModule
        The checked module.
    source : Module
        The parsed source module.
    target : str
        The transpile target.

    Returns
    -------
    IRProgram | None
        The target plan, or ``None`` when the source declares no program.

    Raises
    ------
    UnsupportedConstruct
        If the program falls in an elaboration gap or its checked computation
        cannot be represented by the target plan.
    """
    if not any(isinstance(statement, ProgramDecl) for statement in source.statements):
        return None
    expanded = expand_composite_lets(source, target="stan")
    program = pick_program(expanded)
    if module.gap:
        kind = module.gap.partition("; ")[0]
        raise UnsupportedConstruct(f"qvr-{target}", [kind])
    return program_plan(module, expanded, program, target)


__all__ = [
    "Lower",
    "TARGET_NAME_SEPARATOR",
    "checked_program_plan",
    "checked_module",
    "program_plan",
    "target_name",
]
