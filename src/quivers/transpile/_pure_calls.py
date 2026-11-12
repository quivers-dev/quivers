"""Inlining pure computation calls, shared by the planner and the runtime.

A call of a pure computation (one that performs no effect, takes no
static arguments, and is a chain of pure bindings ending in a return)
can be replaced by the callee's body at the call's arguments. The
transpile planner states such a call as deterministic statements, and
the runtime compiler compiles it to let steps, so a program that calls
one is a `MonadicProgram` with gradients. Both read the call the same
way, through `pure_call_bindings`; `inline_pure_call` further spells
the result as let expressions in the compiled runtime's vocabulary.

This module holds the analysis and the value-to-expression reading it
needs, apart from the planner, so that the runtime compiler can use it
without importing the planner's elaboration machinery.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Literal, cast

from quivers.dsl.ast_nodes import TypeName
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprList,
    LetExprLiteral,
    LetExprNode,
    LetExprString,
    LetExprUnaryOp,
    LetExprVar,
    LetFactorBinder,
)
from quivers.dsl.pure_builtins import PRIMITIVE_BUILTINS, PRIMITIVE_OPERATORS
from quivers.qiec.canonical import tensor_shape, tensor_type
from quivers.qiec.identifiers import SourceOrigin
from quivers.qiec.module import NamedComputation
from quivers.qiec.primitives import primitive
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    Comprehension,
    Computation,
    ConstructorValue,
    DistributionValue,
    Gather,
    Handle,
    If,
    LetBinding,
    LetValue,
    LiteralValue,
    Local,
    LogDensity,
    NewInstance,
    Perform,
    PrimitiveApplication,
    Projection,
    Reduction,
    Resume,
    Return,
    Rowwise,
    TensorValue,
    Value,
    Var,
    WeightSum,
)
from quivers.qiec.types import (
    INT,
    REAL,
    IndexLiteral,
    IndexTerm,
    IndexVariable,
    TypeExpr,
    render_static,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._builtin_spellings import RUNTIME_TARGET, spell_builtin
from quivers.transpile.ir import IRQiecValueExpr
from quivers.transpile.qiec_ir import IRQiecValue, _convert


#: The records of a value term a local's reads can sit in: every value
#: variant and the bindings of a ``LetValue``.
_VALUE_RECORDS: tuple[type, ...] = (
    *(item for item in Value.__value__.__args__ if isinstance(item, type)),
    LetBinding,
)

type _ValueTerm = (
    Value
    | LetBinding
    | str
    | int
    | float
    | bool
    | bytes
    | None
    | tuple[_ValueTerm, ...]
)
"""A value term or one of the parts a rewrite of its locals descends into."""


def replace_locals(value: Value, replacements: Mapping[Local, Value]) -> Value:
    """Replace the reads of some locals in a lowered value.

    The replaced locals are fresh binders the lowering introduced, so
    no binder inside ``value`` can capture a replacement's free locals.

    Parameters
    ----------
    value : Value
        The lowered value.
    replacements : Mapping[Local, Value]
        The value each local's reads become.

    Returns
    -------
    Value
        ``value`` with every ``Var`` of a replaced local substituted.
    """

    def rewrite(term: _ValueTerm) -> _ValueTerm:
        """Replace each variable the replacements name, sharing unchanged terms.

        Parameters
        ----------
        term : _ValueTerm
            The term.

        Returns
        -------
        _ValueTerm
            The rewritten term, the input itself where nothing changed.
        """
        if isinstance(term, Var):
            return replacements.get(term.local, term)
        if isinstance(term, tuple):
            items = tuple(rewrite(item) for item in term)
            if any(new is not old for new, old in zip(items, term, strict=True)):
                return items
            return term
        if isinstance(term, _VALUE_RECORDS) and is_dataclass(term):
            changes: dict[str, _ValueTerm] = {}
            for field in fields(term):
                current = getattr(term, field.name)
                if isinstance(current, (tuple, *_VALUE_RECORDS)):
                    rewritten = rewrite(current)
                    if rewritten is not current:
                        changes[field.name] = rewritten
            return replace(term, **changes) if changes else term
        return term

    return cast(Value, rewrite(value))


#: The internal let function the compiled runtime evaluates a family's
#: log density with: ``$log_prob("Family", value, "parameter", argument,
#: ...)``. The ``$`` keeps it apart from every function a source can
#: call.
LOG_PROB_CALL = "$log_prob"


#: Primitives that are conversions the target's arithmetic performs on
#: its own, so they read back as their operand.
_TRANSPARENT: frozenset[str] = frozenset({"int_to_real", "as_weight", "weight_value"})

#: The conversions of a Boolean to a number, as the values ``select``
#: takes where the Boolean holds and where it does not.
_BOOL_CONVERSIONS: dict[str, tuple[LetExprLiteral, LetExprLiteral]] = {
    "bool_to_real": (LetExprLiteral(value=1.0), LetExprLiteral(value=0.0)),
    "bool_to_int": (
        LetExprLiteral(value=1.0, integral=True),
        LetExprLiteral(value=0.0, integral=True),
    ),
}

#: Targets whose arithmetic keeps two integers integral: Stan's `/`
#: truncates `int / int`. On them a conversion to real is not
#: transparent, so `int_to_real(x)` reads back as `1.0 * x`, which is
#: the same number and a real.
_INTEGRAL_ARITHMETIC_TARGETS: frozenset[str] = frozenset({"stan", RUNTIME_TARGET})

#: Targets whose arithmetic does not promote an integer operand to a
#: real one: TensorFlow refuses a product of a float and an int tensor.
#: On them a conversion to real reads back as an explicit `real(x)`.
_STRICT_DTYPE_TARGETS: frozenset[str] = frozenset({"edward2"})

#: Integer division and remainder, which truncate toward zero. A target
#: whose `/` and `%` on two integers do so reads them as the operators;
#: every other target reads them as these calls, which each renderer
#: spells with its own truncating division and remainder.
_TRUNCATING_CALLS: dict[str, str] = {"div_int": "quot", "mod_int": "rem"}

#: Unary primitives the expression language writes as operators.
_UNARY_OPERATORS: dict[str, Literal["-", "not"]] = {
    "neg_int": "-",
    "neg_real": "-",
    "not": "not",
}


#: The value terms a plan walks into when reading a pair's components
#: back: every variant of `Value`, as one runtime tuple.
_VALUE_TYPES: tuple[type, ...] = tuple(
    item for item in Value.__value__.__args__ if isinstance(item, type)
)


class _LetFloater:
    """Hoist the bindings of every :class:`LetValue` in a value to the front.

    A plan states a program as a sequence of target statements, and the
    target expression vocabulary has no binding form, so a value's
    shared intermediates become statements of their own ahead of the
    statement that reads them. Each binding is hoisted once, under a
    fresh name the targets accept, so a term that reads a binding many
    times keeps its size on every target. A binding inside a
    comprehension reads the comprehension's index, so it is hoisted as
    the comprehension of its value over the same axis and each read
    becomes that tensor at the index.

    Parameters
    ----------
    occupied : set[str]
        Every name the program already uses, which no hoisted binding
        may take.
    """

    def __init__(self, occupied: set[str]) -> None:
        self.occupied = occupied

    def fresh(self, name: str) -> str:
        """A name no statement of the program uses yet.

        Parameters
        ----------
        name : str
            The binder's name in the checked term.

        Returns
        -------
        str
            The name without its leading underscores, which several
            targets reject, suffixed until it is unused.
        """
        base = name.lstrip("_") or "shared"
        candidate = base
        suffix = 0
        while candidate in self.occupied:
            suffix += 1
            candidate = f"{base}_{suffix}"
        self.occupied.add(candidate)
        return candidate

    def computation(
        self, first: Computation
    ) -> tuple[tuple[LetBinding, ...], Computation]:
        """Hoist the bindings out of the values a statement reads.

        Parameters
        ----------
        first : Computation
            The computation one program statement performs.

        Returns
        -------
        tuple[tuple[LetBinding, ...], Computation]
            The hoisted bindings in evaluation order, and the
            computation reading them by name. A computation other than
            a return, a request, or a call is returned unchanged.
        """
        if isinstance(first, Return):
            bindings, value = self.value(first.value, {})
            return bindings, Return(value)
        if isinstance(first, Perform):
            bindings, arguments = self.values(first.request.arguments, {})
            return bindings, Perform(replace(first.request, arguments=arguments))
        if isinstance(first, Call):
            bindings, arguments = self.values(first.arguments, {})
            return bindings, replace(first, arguments=arguments)
        return (), first

    def values(
        self, items: tuple[Value, ...], renamed: dict[Local, Value]
    ) -> tuple[tuple[LetBinding, ...], tuple[Value, ...]]:
        """Hoist the bindings out of several values in order.

        Parameters
        ----------
        items : tuple[Value, ...]
            The values.
        renamed : dict[Local, Value]
            The term each local bound by an enclosing hoisted binding is
            read through once hoisted.

        Returns
        -------
        tuple[tuple[LetBinding, ...], tuple[Value, ...]]
            The hoisted bindings, and the values reading them.
        """
        hoisted: list[LetBinding] = []
        results: list[Value] = []
        for item in items:
            bindings, result = self.value(item, renamed)
            hoisted.extend(bindings)
            results.append(result)
        return tuple(hoisted), tuple(results)

    def value(
        self, value: Value, renamed: dict[Local, Value]
    ) -> tuple[tuple[LetBinding, ...], Value]:
        """Hoist the bindings out of one value.

        Parameters
        ----------
        value : Value
            The value.
        renamed : dict[Local, Value]
            The term each local bound by an enclosing hoisted binding is
            read through once hoisted.

        Returns
        -------
        tuple[tuple[LetBinding, ...], Value]
            The hoisted bindings in evaluation order, each reading only
            program names and earlier hoisted bindings, and the value
            with no ``LetValue`` left in it.
        """
        if isinstance(value, Var):
            return (), renamed.get(value.local, value)
        if isinstance(value, LetValue):
            hoisted: list[LetBinding] = []
            scope = dict(renamed)
            for binding in value.bindings:
                inner, bound = self.value(binding.value, scope)
                hoisted.extend(inner)
                local = Local(self.fresh(binding.binder.name), binding.binder.type)
                hoisted.append(LetBinding(local, bound))
                scope[binding.binder] = Var(local)
            inner, body = self.value(value.body, scope)
            return (*hoisted, *inner), body
        if isinstance(value, Comprehension):
            inner, body = self.value(value.body, renamed)
            if not inner:
                return (), replace(value, body=body)
            return self._stacked(value, inner, body)
        changes: dict[
            str, Value | tuple[Value, ...] | tuple[tuple[str, Value], ...]
        ] = {}
        hoisted = []
        for field in fields(value):
            current = getattr(value, field.name)
            if isinstance(current, _VALUE_TYPES):
                inner, result = self.value(cast(Value, current), renamed)
                hoisted.extend(inner)
                changes[field.name] = result
            elif isinstance(current, tuple) and current:
                if all(isinstance(item, _VALUE_TYPES) for item in current):
                    inner, results = self.values(
                        cast(tuple[Value, ...], current), renamed
                    )
                    hoisted.extend(inner)
                    changes[field.name] = results
                elif all(
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[1], _VALUE_TYPES)
                    for item in current
                ):
                    pairs: list[tuple[str, Value]] = []
                    for name, item in cast(tuple[tuple[str, Value], ...], current):
                        inner, result = self.value(item, renamed)
                        hoisted.extend(inner)
                        pairs.append((name, result))
                    changes[field.name] = tuple(pairs)
        return tuple(hoisted), replace(value, **changes) if changes else value

    def _stacked(
        self,
        value: Comprehension,
        hoisted: tuple[LetBinding, ...],
        body: Value,
    ) -> tuple[tuple[LetBinding, ...], Value]:
        """Hoist a comprehension body's bindings over the comprehension's axis.

        Parameters
        ----------
        value : Comprehension
            The comprehension.
        hoisted : tuple[LetBinding, ...]
            The bindings hoisted out of its body, which may read its
            index.
        body : Value
            The body reading them.

        Returns
        -------
        tuple[tuple[LetBinding, ...], Value]
            One binding per hoisted binding, holding the comprehension of
            its value over the axis, and the comprehension whose body
            reads each at the index.
        """
        index = Var(value.binder)
        stacked: dict[Local, Value] = {}
        lifted: list[LetBinding] = []
        for binding in hoisted:
            _, bound = self.value(binding.value, stacked)
            element = binding.binder.type
            split = tensor_shape(element)
            tensor = (
                tensor_type(element, (value.extent,))
                if split is None
                else tensor_type(split[0], (value.extent, *split[1]))
            )
            local = Local(binding.binder.name, tensor)
            lifted.append(
                LetBinding(
                    local, Comprehension(value.binder, value.extent, bound, tensor)
                )
            )
            stacked[binding.binder] = Gather(Var(local), index, element)
        _, body = self.value(body, stacked)
        return tuple(lifted), replace(value, body=body)


def _local_names(term: Computation) -> set[str]:
    """Every local name a checked term binds or reads.

    Parameters
    ----------
    term : Computation
        The computation, whose values and records are walked in turn.

    Returns
    -------
    set[str]
        The names.
    """
    found: set[str] = set()
    pending: list[Hashable] = [term]
    while pending:
        item = pending.pop()
        if isinstance(item, Local):
            found.add(item.name)
        elif isinstance(item, tuple):
            pending.extend(item)
        elif is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, field.name) for field in fields(item))
    return found


def _axis_name_of(extent: IndexTerm, cards: dict[str, int]) -> str:
    """The object an extent names.

    Parameters
    ----------
    extent : IndexTerm
        The extent.
    cards : dict[str, int]
        The object cardinalities.

    Returns
    -------
    str
        The first object with that cardinality, or the extent rendered.
    """
    if isinstance(extent, IndexLiteral):
        for name, size in cards.items():
            if size == extent.value:
                return name
    if isinstance(extent, IndexVariable):
        return extent.name
    return render_static(extent)


def _row_reduction(value: Value) -> tuple[str, Value] | None:
    """A reduction along a tensor's last axis, as the operator and tensor.

    The elaboration writes ``sum(x)`` over a tensor of rank two or
    more as a comprehension over the leading axes of the reduction of
    each row; the wire form is the call itself.

    Parameters
    ----------
    value : Value
        The value.

    Returns
    -------
    tuple[str, Value] | None
        The operator and the tensor reduced, or ``None`` when the
        value is not such a comprehension.
    """
    if not isinstance(value, Comprehension):
        return None
    body = value.body
    inner = _row_reduction(body)
    if inner is not None:
        operator, reduced = inner
    elif isinstance(body, Reduction):
        operator, reduced = body.operator, body.value
    else:
        return None
    if (
        isinstance(reduced, Gather)
        and isinstance(reduced.index, Var)
        and reduced.index.local == value.binder
    ):
        return operator, reduced.value
    return None


def _log_prob_call(
    density: LogDensity, cards: dict[str, int], program: str
) -> LetExprNode:
    """A family's log density at a value, as the compiled runtime reads it.

    Parameters
    ----------
    density : LogDensity
        The log density.
    cards : dict[str, int]
        The object cardinalities.
    program : str
        The program the value belongs to, for diagnostics.

    Returns
    -------
    LetExprNode
        ``$log_prob("Family", value, "parameter", argument, ...)``,
        summed over the family's plate when it has one.

    Raises
    ------
    UnsupportedConstruct
        If the density is not of a family construction.
    """
    sampleable = density.sampleable
    if not isinstance(sampleable, DistributionValue):
        raise UnsupportedConstruct(
            f"qvr-{RUNTIME_TARGET}", [f"qiec:expression:log_density:{program}"]
        )
    arguments: list[LetExprNode] = [
        LetExprString(value=sampleable.name),
        _expression(density.value, cards, RUNTIME_TARGET, program),
    ]
    for parameter, value in sampleable.arguments:
        arguments.append(LetExprString(value=parameter))
        arguments.append(_expression(value, cards, RUNTIME_TARGET, program))
    call = LetExprCall(func=LOG_PROB_CALL, args=tuple(arguments))
    if sampleable.plate.batch or sampleable.plate.event:
        return LetExprCall(func="sum", args=(call,))
    return call


def _expression(
    value: Value, cards: dict[str, int], target: str, program: str
) -> LetExprNode:
    """A pure value as a let expression.

    Parameters
    ----------
    value : Value
        The value.
    cards : dict[str, int]
        The object cardinalities, which name a comprehension's axis.
    target : str
        The transpile target, for diagnostics.
    program : str
        The program the value belongs to, for diagnostics.

    Returns
    -------
    LetExprNode
        The expression the renderers translate.

    Raises
    ------
    UnsupportedConstruct
        If the value has no expression form.
    """
    if isinstance(value, Var):
        return LetExprVar(name=value.local.name)
    if isinstance(value, LiteralValue):
        if isinstance(value.value, bool):
            return LetExprLiteral(value=float(value.value), integral=True)
        if isinstance(value.value, int | float):
            return LetExprLiteral(
                value=float(value.value), integral=isinstance(value.value, int)
            )
        if isinstance(value.value, str):
            return LetExprString(value=value.value)
    if isinstance(value, PrimitiveApplication):
        if value.name == "int_to_real" and target in _INTEGRAL_ARITHMETIC_TARGETS:
            return LetExprBinOp(
                op="*",
                left=LetExprLiteral(value=1.0),
                right=_expression(value.arguments[0], cards, target, program),
            )
        if value.name == "int_to_real" and target in _STRICT_DTYPE_TARGETS:
            return LetExprCall(
                func="real",
                args=(_expression(value.arguments[0], cards, target, program),),
            )
        if value.name in _TRANSPARENT:
            return _expression(value.arguments[0], cards, target, program)
        truncating = _TRUNCATING_CALLS.get(value.name)
        if truncating is not None and target not in _INTEGRAL_ARITHMETIC_TARGETS:
            return LetExprCall(
                func=truncating,
                args=tuple(
                    _expression(argument, cards, target, program)
                    for argument in value.arguments
                ),
            )
        operator = PRIMITIVE_OPERATORS.get(value.name)
        if operator is not None and len(value.arguments) == 2:
            left, right = value.arguments
            return LetExprBinOp(
                op=operator,
                left=_expression(left, cards, target, program),
                right=_expression(right, cards, target, program),
            )
        unary = _UNARY_OPERATORS.get(value.name)
        if unary is not None and len(value.arguments) == 1:
            return LetExprUnaryOp(
                op=unary,
                operand=_expression(value.arguments[0], cards, target, program),
            )
        if value.name in _BOOL_CONVERSIONS:
            # A Boolean read as a number is one where it holds and zero
            # elsewhere: the conditional value every target spells.
            one, zero = _BOOL_CONVERSIONS[value.name]
            return spell_builtin(
                target,
                "select",
                (_expression(value.arguments[0], cards, target, program), one, zero),
            )
        builtin = PRIMITIVE_BUILTINS.get(value.name)
        if builtin is not None:
            return spell_builtin(
                target,
                builtin,
                tuple(
                    _expression(argument, cards, target, program)
                    for argument in value.arguments
                ),
            )
    if isinstance(value, Gather):
        array = _expression(value.value, cards, target, program)
        index = _expression(value.index, cards, target, program)
        if isinstance(array, LetExprIndex):
            # Nested gathers read back as one multi-index subscript.
            return LetExprIndex(array=array.array, indices=(*array.indices, index))
        return LetExprIndex(array=array, indices=(index,))
    if isinstance(value, TensorValue):
        return LetExprList(
            items=tuple(
                _expression(item, cards, target, program) for item in value.items
            )
        )
    if isinstance(value, Reduction):
        return LetExprCall(
            func=value.operator,
            args=(_expression(value.value, cards, target, program),),
        )
    rowwise = _row_reduction(value)
    if rowwise is not None:
        operator, reduced = rowwise
        return LetExprCall(
            func=operator, args=(_expression(reduced, cards, target, program),)
        )
    if isinstance(value, Rowwise):
        return LetExprCall(
            func=value.operator,
            args=(_expression(value.value, cards, target, program),),
        )
    if isinstance(value, Comprehension):
        axis = _axis_name_of(value.extent, cards)
        binder = LetFactorBinder(var=value.binder.name, index=TypeName(name=axis))
        body = _expression(value.body, cards, target, program)
        if isinstance(body, LetExprFactor) and body.body is not None:
            # Nested comprehensions read back as one factor over
            # every binder.
            return LetExprFactor(binders=(binder, *body.binders), body=body.body)
        return LetExprFactor(binders=(binder,), body=body)
    if isinstance(value, WeightSum):
        return LetExprCall(
            func="sum", args=(_expression(value.value, cards, target, program),)
        )
    if isinstance(value, LogDensity) and target == RUNTIME_TARGET:
        return _log_prob_call(value, cards, program)
    if isinstance(value, LogDensity):
        raise UnsupportedConstruct(
            f"qvr-{target}",
            [f"qiec:expression:log_density:{program}"],
        )
    if isinstance(value, ConstructorValue):
        # A constructor value reaches a callee through the target's QIEC
        # runtime; its fields are values of the model body.
        return IRQiecValueExpr(value=cast(IRQiecValue, _convert(value)))
    if isinstance(value, Projection):
        # A component of a product, as the source subscripts a tuple.
        return LetExprIndex(
            array=_expression(value.value, cards, target, program),
            indices=(LetExprLiteral(value=float(value.position)),),
        )
    raise UnsupportedConstruct(
        f"qvr-{target}",
        [f"qiec:expression:{type(value).__name__}:{program}"],
    )


class CallNotLowered(Exception):
    """A call the analysis does not lower to bindings, and why.

    Parameters
    ----------
    reason : str
        What in the callee stops the lowering, as a clause completing
        "the call ...".
    effects : bool
        Whether the obstacle is that the callee makes requests, which
        the compiled runtime may still run as steps of the program.
    """

    def __init__(self, reason: str, *, effects: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.effects = effects


def origin_line(term: object) -> str:
    """The source line a term records, as a parenthetical.

    Parameters
    ----------
    term : object
        A checked term, whose fields are searched for a
        [`SourceOrigin`][quivers.qiec.SourceOrigin] in order.

    Returns
    -------
    str
        `` (line N)`` for the first origin with a line, else empty.
    """
    found = _first_origin(term)
    if found is None or not found.line:
        return ""
    return f" (line {found.line})"


def _first_origin(term: object) -> SourceOrigin | None:
    """The first source origin with a line that a term records.

    Parameters
    ----------
    term : object
        A checked term, whose fields are searched breadth first.

    Returns
    -------
    SourceOrigin | None
        The origin, or ``None`` when the term records none with a line.
    """
    pending: list[object] = [term]
    while pending:
        item = pending.pop(0)
        if isinstance(item, SourceOrigin):
            if item.line:
                return item
            continue
        if isinstance(item, tuple):
            pending.extend(item)
        elif is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, field.name) for field in fields(item))
    return None


def free_local_names(value: Value) -> set[str]:
    """The names of the locals a value reads.

    Parameters
    ----------
    value : Value
        The value.

    Returns
    -------
    set[str]
        The name of every variable in it, binders' own reads included.
    """
    found: set[str] = set()
    pending: list[object] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, Var):
            found.add(item.local.name)
        elif isinstance(item, tuple):
            pending.extend(item)
        elif is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, field.name) for field in fields(item))
    return found


def _bound_names(term: Computation | Value) -> set[str]:
    """The names a term binds: its sequencing, let, and comprehension binders.

    Parameters
    ----------
    term : Computation | Value
        The term.

    Returns
    -------
    set[str]
        The names.
    """
    found: set[str] = set()
    pending: list[object] = [term]
    while pending:
        item = pending.pop()
        if isinstance(item, Bind | LetBinding | Comprehension):
            found.add(item.binder.name)
        if isinstance(item, tuple):
            pending.extend(item)
        elif is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, field.name) for field in fields(item))
    return found


def _select(
    condition: Value,
    then: Value,
    otherwise: Value,
    result_type: TypeExpr,
    owner: str,
    term: If,
) -> Value:
    """A conditional between two pure values, as ``select``.

    Parameters
    ----------
    condition : Value
        The condition.
    then : Value
        The value where it holds.
    otherwise : Value
        The value where it does not.
    result_type : TypeExpr
        The type both branches return.
    owner : str
        The computation holding the conditional, for the diagnostic.
    term : If
        The conditional, for the diagnostic's line.

    Returns
    -------
    Value
        ``select(condition, then, otherwise)``, entrywise over a tensor.

    Raises
    ------
    CallNotLowered
        If the branches return neither real nor integer values.
    """
    split = tensor_shape(result_type)
    element = result_type if split is None else split[0]
    if element == REAL:
        name = "select_real"
    elif element == INT:
        name = "select_int"
    else:
        raise CallNotLowered(
            f"runs an `if` in `{owner}`{origin_line(term)} whose branches return "
            f"{render_static(result_type)}, which `select` cannot choose "
            f"between: it chooses between real or integer values"
        )
    signature = primitive(name)
    return PrimitiveApplication(
        signature.id,
        name,
        (condition, then, otherwise),
        result_type,
        _origin_of(condition),
    )


def _origin_of(value: Value) -> SourceOrigin:
    """The origin a lowered conditional records: its condition's.

    Parameters
    ----------
    value : Value
        The condition.

    Returns
    -------
    SourceOrigin
        The condition's own origin when it records one, else an origin
        naming the lowering.
    """
    found = _first_origin(value)
    if found is not None:
        return found
    return SourceOrigin("lowered", ("if",), "primitive", "qiec-lowering")


def inline_call(
    callee: NamedComputation,
    arguments: tuple[Value, ...],
    result_type: TypeExpr,
    computations: Mapping[str, NamedComputation],
    active: tuple[str, ...] = (),
) -> Value:
    """A pure computation's call as one value, its calls inlined in turn.

    The body is read at the arguments: a parameter whose argument is a
    variable or a literal is read in place, unless a binder of the body
    has a name the argument reads, and any other argument is bound
    first. A pure computation the body calls is inlined the same way,
    and an ``if`` between pure branches is ``select`` of its branches.

    Parameters
    ----------
    callee : NamedComputation
        The computation called.
    arguments : tuple[Value, ...]
        The call's arguments, one per parameter.
    result_type : TypeExpr
        The type of the call's result.
    computations : Mapping[str, NamedComputation]
        The module's computations, by name.
    active : tuple[str, ...]
        The computations whose bodies enclose this call, outermost
        first, which the call may not reenter.

    Returns
    -------
    Value
        The body's value: a [`LetValue`][quivers.qiec.LetValue] of the
        bound arguments and the body's bindings around the returned
        value, or the returned value alone when nothing is bound.

    Raises
    ------
    CallNotLowered
        If the callee takes static arguments, is reentered, makes a
        request (``effects`` set), handles an effect, branches with
        ``case``, or chooses with ``if`` between values ``select``
        cannot.
    """
    if callee.name in active:
        raise CallNotLowered(recursion_reason((*active, callee.name)))
    if callee.telescope:
        raise CallNotLowered(
            f"passes static arguments to `{callee.name}`, which the compiled "
            f"runtime does not instantiate in a call"
        )
    if len(arguments) != len(callee.parameters):
        raise CallNotLowered(
            f"passes {len(arguments)} argument(s) to `{callee.name}`, which "
            f"takes {len(callee.parameters)}"
        )
    if callee.type.effects.entries:
        raise CallNotLowered(
            f"calls `{callee.name}`, which makes requests", effects=True
        )
    captured = _bound_names(callee.body)
    replaced: dict[Local, Value] = {}
    bindings: list[LetBinding] = []
    for parameter, argument in zip(callee.parameters, arguments, strict=True):
        if isinstance(argument, Var | LiteralValue) and not (
            free_local_names(argument) & captured
        ):
            replaced[parameter] = argument
        else:
            bindings.append(LetBinding(parameter, argument))
    body = computation_value(
        callee.body, replaced, result_type, callee, computations, (*active, callee.name)
    )
    if not bindings:
        return body
    return LetValue(tuple(bindings), body, result_type)


def computation_value(
    term: Computation,
    replaced: Mapping[Local, Value],
    result_type: TypeExpr,
    owner: NamedComputation,
    computations: Mapping[str, NamedComputation],
    active: tuple[str, ...],
) -> Value:
    """A pure computation term as one value.

    Parameters
    ----------
    term : Computation
        A chain of binds whose steps return pure values, call pure
        computations, or branch with ``if`` between pure computations.
    replaced : Mapping[Local, Value]
        The value each of the owner's locals read so far stands for.
    result_type : TypeExpr
        The type of the term's result.
    owner : NamedComputation
        The computation whose body holds the term, for diagnostics.
    computations : Mapping[str, NamedComputation]
        The module's computations, by name.
    active : tuple[str, ...]
        The computations whose bodies enclose the term, the owner last.

    Returns
    -------
    Value
        The term's value, a [`LetValue`][quivers.qiec.LetValue] when it
        binds anything.

    Raises
    ------
    CallNotLowered
        If the term makes a request (``effects`` set), reenters a
        computation, handles an effect, branches with ``case``, or
        chooses between values ``select`` cannot.
    """
    bindings: list[LetBinding] = []
    node = term
    while isinstance(node, Bind):
        first = node.first
        if isinstance(first, Bind):
            # A nested sequence associates into the enclosing one; its
            # binder is fresh to the continuation.
            node = Bind(
                first.binder, first.first, Bind(node.binder, first.then, node.then)
            )
            continue
        bindings.append(
            LetBinding(
                node.binder,
                _step_value(
                    first, replaced, node.binder.type, owner, computations, active
                ),
            )
        )
        node = node.then
    result = _step_value(node, replaced, result_type, owner, computations, active)
    if not bindings:
        return result
    return LetValue(tuple(bindings), result, result_type)


def _step_value(
    term: Computation,
    replaced: Mapping[Local, Value],
    result_type: TypeExpr,
    owner: NamedComputation,
    computations: Mapping[str, NamedComputation],
    active: tuple[str, ...],
) -> Value:
    """One pure step of a body as a value.

    Parameters
    ----------
    term : Computation
        The step.
    replaced : Mapping[Local, Value]
        The value each of the owner's locals read so far stands for.
    result_type : TypeExpr
        The type of the step's result.
    owner : NamedComputation
        The computation whose body holds the step.
    computations : Mapping[str, NamedComputation]
        The module's computations, by name.
    active : tuple[str, ...]
        The computations whose bodies enclose the step, the owner last.

    Returns
    -------
    Value
        The step's value.

    Raises
    ------
    CallNotLowered
        As `computation_value` does.
    """
    if isinstance(term, Return):
        return replace_locals(term.value, replaced)
    if isinstance(term, Call):
        callee = computations.get(term.name)
        if callee is None:
            raise CallNotLowered(
                f"calls `{term.name}` from `{owner.name}`, which is not a "
                f"computation of the module"
            )
        return inline_call(
            callee,
            tuple(replace_locals(item, replaced) for item in term.arguments),
            result_type,
            computations,
            active,
        )
    if isinstance(term, If):
        condition = replace_locals(term.condition, replaced)
        then = computation_value(
            term.then, replaced, result_type, owner, computations, active
        )
        otherwise = computation_value(
            term.otherwise, replaced, result_type, owner, computations, active
        )
        return _select(condition, then, otherwise, result_type, owner.name, term)
    if isinstance(term, Perform):
        raise CallNotLowered(
            f"makes a `{term.request.effect.name}` request in `{owner.name}`"
            f"{origin_line(term)}",
            effects=True,
        )
    if isinstance(term, Case):
        raise CallNotLowered(
            f"branches on a family value with `case` in `{owner.name}`"
            f"{origin_line(term)}; the compiled runtime lowers only an `if` "
            f"whose branches are pure, as `select`"
        )
    if isinstance(term, Handle | NewInstance | Resume):
        raise CallNotLowered(handled_reason(owner, term))
    raise CallNotLowered(
        f"runs a {type(term).__name__} in `{owner.name}`, which the compiled "
        f"runtime has no step for"
    )


def recursion_reason(chain: tuple[str, ...]) -> str:
    """Why a recursive call is refused, and what to write instead.

    Parameters
    ----------
    chain : tuple[str, ...]
        The computations entered, ending with the one reentered.

    Returns
    -------
    str
        The reason clause.
    """
    start = chain.index(chain[-1])
    cycle = " -> ".join(f"`{name}`" for name in chain[start:])
    return (
        f"reaches a recursive call ({cycle}), so how many steps the call runs "
        f"depends on the values it is given, while the compiled runtime runs a "
        f"fixed chain of steps; write the repetition without recursion: a "
        f"reduction such as `sum` or `fold`, or `map`, over a tensor of fixed "
        f"length, or `scan` over a sequence for a recurrence"
    )


def handled_reason(owner: NamedComputation, term: Computation) -> str:
    """Why a callee that handles an effect is refused.

    Parameters
    ----------
    owner : NamedComputation
        The callee.
    term : Computation
        The handling, instance, or resumption term.

    Returns
    -------
    str
        The reason clause.
    """
    effect = (
        f"the effect `{term.effect.name}`"
        if isinstance(term, NewInstance)
        else "an effect"
    )
    return (
        f"handles {effect} inside `{owner.name}`{origin_line(term)}, and the "
        f"compiled runtime runs only the program's own `Random` and `Score` "
        f"requests, not a handler and its resumptions"
    )


def pure_call_bindings(
    callee: NamedComputation,
    arguments: tuple[Value, ...],
    result_type: TypeExpr,
    floater: _LetFloater,
    computations: Mapping[str, NamedComputation],
) -> tuple[tuple[LetBinding, ...], Value] | None:
    """A pure computation's call as hoisted bindings and a result.

    The analysis both the planner and the runtime compiler inline a
    pure call with: the callee's body at the arguments, with every pure
    computation it calls inlined in turn and every ``if`` between pure
    branches read as ``select`` (`inline_call`), its bindings hoisted to
    fresh names by ``floater``.

    Parameters
    ----------
    callee : NamedComputation
        The computation called.
    arguments : tuple[Value, ...]
        The call's arguments, one per parameter.
    result_type : TypeExpr
        The type of the call's result.
    floater : _LetFloater
        The floater that names the hoisted bindings.
    computations : Mapping[str, NamedComputation]
        The module's computations, by name.

    Returns
    -------
    tuple[tuple[LetBinding, ...], Value] | None
        The bindings in evaluation order and the result reading them,
        or ``None`` when `inline_call` does not lower the call.
    """
    try:
        shared = inline_call(callee, arguments, result_type, computations)
    except CallNotLowered:
        return None
    return floater.value(shared, {})


__all__: list[str] = []
