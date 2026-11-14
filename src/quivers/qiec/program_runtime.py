"""Running program entry points on the reference machine.

An elaborated program is a named computation whose row names the module's
canonical ``random`` and ``score`` instances. Running it means choosing
handlers for the two: this module wraps the entry point in a scoring
replay of ``Random`` and an accumulating ``Score``, so a program applied to
data and to a value for each of its sites yields the value it returns and
the log joint density the sites and observations accumulate. The wrapper is
an ordinary computation added to a copy of the module, so the kernel checks
it like any other before it runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import didactic.api as dx

from quivers._values import HeldValue, OPAQUE_BY_IDENTITY
from quivers.qiec._models import QiecModel

from quivers.qiec.builtins import PARAM_GET, RANDOM_SAMPLE, SCORE_ADD
from quivers.qiec.canonical import LOG_WEIGHT, tensor_shape
from quivers.qiec.effects import (
    ComputationType,
    EffectRow,
    HandlerClauseDef,
    HandlerDef,
    ResumptionGrade,
    RowEntry,
)
from quivers.qiec.execution import (
    ExecutionDiagnostic,
    ExecutionFailure,
    ExecutionResult,
    RuntimeConfiguration,
    RuntimeProvider,
    RuntimeSelection,
    TraceObserver,
    run_named,
)
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.identifiers import ComputationId, HandlerId, SourceOrigin
from quivers.qiec.module import NamedComputation, QiecModule
from quivers.qiec.programs import ProgramEntry
from quivers.qiec.kinds import IndexBinder
from quivers.qiec.primitives import primitive
from quivers.qiec.terms import (
    Bind,
    Call,
    Computation,
    Handle,
    LiteralValue,
    Local,
    PrimitiveApplication,
    Return,
    TupleValue,
    Var,
)
from quivers.qiec.types import (
    REAL,
    IndexLiteral,
    IndexVariable,
    ShapeIndex,
    StaticArgument,
    TypeApplication,
    TypeExpr,
    product_type,
    render_static,
)

#: The names of the wrapper's handlers, which no source may declare.
REPLAY_HANDLER = "__program_replay"
CONDITION_HANDLER = "__program_condition"
DRAW_HANDLER = "__program_draw"
ACCUMULATE_HANDLER = "__program_accumulate"
PARAMS_HANDLER = "__program_params"


class ProgramRun(QiecModel):
    """What running a program on the reference machine produced.

    Parameters
    ----------
    value
        The value the program returned.
    log_joint
        The log joint density: every replayed site's log density under
        its distribution plus every observation's and score's weight.
    result
        The underlying execution result, with its trace.
    """

    value: HeldValue = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_joint: float
    result: ExecutionResult = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


def program_entry(module: QiecModule, name: str) -> ProgramEntry:
    """Find a program's entry point in a module.

    Parameters
    ----------
    module : QiecModule
        The module.
    name : str
        The program's name.

    Returns
    -------
    ProgramEntry
        The entry.

    Raises
    ------
    KeyError
        If the module has no program of that name.
    """
    for entry in module.entries:
        if entry.name == name:
            return entry
    raise KeyError(f"module {module.module!r} has no program {name!r}")


def _computation(module: QiecModule, identity: ComputationId) -> NamedComputation:
    """Find a computation by identity.

    Parameters
    ----------
    module : QiecModule
        The module.
    identity : ComputationId
        The computation's identity.

    Returns
    -------
    NamedComputation
        The computation.

    Raises
    ------
    KeyError
        If no computation has the identity.
    """
    for computation in module.computations:
        if computation.id == identity:
            return computation
    raise KeyError(f"module {module.module!r} has no computation {identity!r}")


def _marginal_handler(name: str) -> tuple[bool, str] | None:
    """Read a marginalization handler's kind off its name.

    The elaborator names its enumeration handlers ``enumerate_marginal``
    and ``enumerate_grouped_marginal``, with ``_sum`` or ``_mean``
    appended for a reduction other than ``logsumexp``, and ``_table``
    for the handler answering one weight per support value.

    Parameters
    ----------
    name : str
        The handler's name.

    Returns
    -------
    tuple[bool, str] | None
        Whether the handler answers per group position, and its
        reduction; ``None`` for a handler of another kind.
    """
    for reduction in ("sum", "mean", "table"):
        suffix = f"_{reduction}"
        if name.endswith(suffix):
            base = _marginal_handler(name[: -len(suffix)])
            return None if base is None else (base[0], reduction)
    if name.endswith("enumerate_grouped_marginal"):
        return (True, "logsumexp")
    if name.endswith("enumerate_marginal"):
        return (False, "logsumexp")
    return None


def joint_module(module: QiecModule, name: str) -> tuple[QiecModule, str]:
    """Extend a module with a computation scoring one program's log joint.

    Parameters
    ----------
    module : QiecModule
        The module holding the program.
    name : str
        The program's name.

    Returns
    -------
    tuple[QiecModule, str]
        The extended module and the wrapper computation's name. The
        wrapper takes the program's parameters, handles its ``score``
        instance with an accumulating handler and its ``random``
        instance with a replaying one, and returns the program's value
        paired with the accumulated log weight.

    Raises
    ------
    KeyError
        If the module has no such program.
    """
    return _wrapped_module(module, name, sampling=False)


def sample_module(module: QiecModule, name: str) -> tuple[QiecModule, str]:
    """Extend a module with a computation running one program forward.

    Parameters
    ----------
    module : QiecModule
        The module holding the program.
    name : str
        The program's name.

    Returns
    -------
    tuple[QiecModule, str]
        The extended module and the wrapper computation's name. The
        wrapper takes the program's parameters, handles its ``random``
        instance with a conditioning replay of the given sites inside a
        scoring draw of every other site, its ``score`` instance with an
        accumulating handler, and returns the program's value paired
        with the accumulated log weight: the log joint of the draws it
        made and the sites it was given.

    Raises
    ------
    KeyError
        If the module has no such program.
    """
    return _wrapped_module(module, name, sampling=True)


def _wrapped_module(
    module: QiecModule, name: str, *, sampling: bool
) -> tuple[QiecModule, str]:
    """Extend a module with a wrapper handling one program's instances.

    Parameters
    ----------
    module : QiecModule
        The module holding the program.
    name : str
        The program's name.
    sampling : bool
        Whether the wrapper draws the sites it is not given (a forward
        run) or requires every site (a replay scoring the log joint).

    Returns
    -------
    tuple[QiecModule, str]
        The extended module and the wrapper computation's name.

    Raises
    ------
    KeyError
        If the module has no such program.
    """
    entry = program_entry(module, name)
    computation = _computation(module, entry.computation)
    answer: TypeExpr = computation.type.result
    statics: tuple[StaticArgument, ...] = tuple(
        IndexVariable(name=binder.name, sort=binder.sort)
        for binder in computation.telescope
        if isinstance(binder, IndexBinder)
    )
    random = next(
        instance
        for instance in module.instances
        if instance.entry.instance == entry.random_instance
    )
    score = next(
        instance
        for instance in module.instances
        if instance.entry.instance == entry.score_instance
    )
    role = "sample" if sampling else "joint"

    def random_handler(handler_name: str, total: bool) -> HandlerDef:
        """A handler of the program's ``random`` instance.

        Parameters
        ----------
        handler_name : str
            The wrapper handler's name.
        total : bool
            Whether the handler answers every sample request.

        Returns
        -------
        HandlerDef
            The declaration, scoring into the ``score`` instance.
        """
        return HandlerDef(
            id=HandlerId.derive(module.module, "handler", handler_name),
            name=handler_name,
            effect=random.entry.effect,
            clauses=(
                HandlerClauseDef(operation=RANDOM_SAMPLE, grade=ResumptionGrade.LINEAR),
            ),
            input_type=answer,
            output_type=answer,
            introduced=EffectRow(entries=(score.entry,)),
            total=total,
            implementation="foreign",
            telescope=computation.telescope,
        )

    accumulate = HandlerDef(
        id=HandlerId.derive(module.module, "handler", ACCUMULATE_HANDLER),
        name=ACCUMULATE_HANDLER,
        effect=score.entry.effect,
        clauses=(HandlerClauseDef(operation=SCORE_ADD, grade=ResumptionGrade.LINEAR),),
        input_type=answer,
        output_type=product_type(answer, LOG_WEIGHT),
        introduced=EffectRow(),
        total=True,
        implementation="foreign",
        telescope=computation.telescope,
    )
    result_type = answer
    parameters = tuple(
        Local(name=parameter.name, type=parameter.type)
        for parameter in computation.parameters
    )
    call = Call(
        callee=computation.id,
        name=computation.name,
        static_arguments=statics,
        arguments=tuple(Var(local=parameter) for parameter in parameters),
        result_type=result_type,
        effects=computation.type.effects,
        origin=SourceOrigin(
            module=module.module,
            structural_path=("programs", name, role, "call"),
            role="call",
            source_protocol=module.source_protocol,
        ),
    )
    performed = {entry.instance for entry in computation.type.effects.entries}
    handlers: list[HandlerDef] = []
    body: Computation = call
    if random.entry.instance in performed:
        # Replaying or drawing a site scores its density, so a program
        # with a site always has scores to accumulate.
        if sampling:
            condition = random_handler(CONDITION_HANDLER, False)
            draw = random_handler(DRAW_HANDLER, True)
            body = Handle(
                instance=random.entry.instance,
                handler=condition.id,
                computation=body,
                static_arguments=statics,
            )
            body = Handle(
                instance=random.entry.instance,
                handler=draw.id,
                computation=body,
                static_arguments=statics,
            )
            handlers.extend((condition, draw))
        else:
            replay = random_handler(REPLAY_HANDLER, True)
            body = Handle(
                instance=random.entry.instance,
                handler=replay.id,
                computation=body,
                static_arguments=statics,
            )
            handlers.append(replay)
    if random.entry.instance in performed or score.entry.instance in performed:
        body = Handle(
            instance=score.entry.instance,
            handler=accumulate.id,
            computation=body,
            static_arguments=statics,
        )
        handlers.append(accumulate)
    else:
        # A program scoring nothing has the zero log joint.
        answered = Local(name="__answer", type=answer)
        body = Bind(
            binder=answered,
            first=body,
            then=Return(
                value=TupleValue(
                    items=(
                        Var(local=answered),
                        PrimitiveApplication(
                            primitive=primitive("as_weight").id,
                            name="as_weight",
                            arguments=(LiteralValue(value=0.0, type=REAL),),
                            result_type=LOG_WEIGHT,
                            origin=SourceOrigin(
                                module=module.module,
                                structural_path=("programs", name, role, "zero"),
                                role="primitive",
                                source_protocol=module.source_protocol,
                            ),
                        ),
                    ),
                    result_type=product_type(answer, LOG_WEIGHT),
                )
            ),
        )
    params = _params_entry(module, computation.type.effects)
    if params is not None:
        # A program calling a deduction reads learned weights through
        # the module's parameter store, which the wrapper serves.
        store = HandlerDef(
            id=HandlerId.derive(module.module, "handler", PARAMS_HANDLER),
            name=PARAMS_HANDLER,
            effect=params.effect,
            clauses=(
                HandlerClauseDef(operation=PARAM_GET, grade=ResumptionGrade.LINEAR),
            ),
            input_type=product_type(answer, LOG_WEIGHT),
            output_type=product_type(answer, LOG_WEIGHT),
            introduced=EffectRow(),
            total=True,
            implementation="foreign",
            telescope=computation.telescope,
        )
        body = Handle(
            instance=params.instance,
            handler=store.id,
            computation=body,
            static_arguments=statics,
        )
        handlers.append(store)
    wrapper_name = f"__{role}_{name}"
    wrapper = NamedComputation(
        id=ComputationId.derive(module.module, "computation", wrapper_name),
        name=wrapper_name,
        telescope=computation.telescope,
        parameters=parameters,
        body=body,
        type=ComputationType(
            effects=EffectRow(), result=product_type(result_type, LOG_WEIGHT)
        ),
        origin=SourceOrigin(
            module=module.module,
            structural_path=("programs", name, role),
            role="computation",
            source_protocol=module.source_protocol,
        ),
    )
    extended = module.with_(
        handlers=(*module.handlers, *handlers),
        computations=(*module.computations, wrapper),
    )
    return extended, wrapper_name


def _params_entry(module: QiecModule, row: EffectRow) -> RowEntry | None:
    """The ``Param`` instance a computation's row names, if any.

    Parameters
    ----------
    module : QiecModule
        The module.
    row : EffectRow
        The computation's row.

    Returns
    -------
    RowEntry | None
        The entry of the module's ``Param`` instance in the row.
    """
    for entry in row.entries:
        if entry.effect.name == "Param":
            return entry
    return None


def _extent_of(value: object, depth: int) -> int:
    """Read the extent of a nested host tensor at one axis.

    Parameters
    ----------
    value : object
        A host tensor: nested tuples or lists, or an object with a
        ``shape``.
    depth : int
        The axis, outermost first.

    Returns
    -------
    int
        The extent.

    Raises
    ------
    KeyError
        If the value has no such axis.
    """
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return int(shape[depth])
        except IndexError as error:
            raise KeyError(f"host value has no axis {depth}") from error
    current = value
    for _ in range(depth):
        if not isinstance(current, (tuple, list)) or not current:
            raise KeyError(f"host value has no axis {depth}")
        current = current[0]
    if not isinstance(current, (tuple, list)):
        raise KeyError(f"host value has no axis {depth}")
    return len(current)


def _host_rank(value: object) -> int:
    """The number of axes of a host value.

    Parameters
    ----------
    value : object
        A number, nested tuples or lists of numbers, or an object with a
        ``shape``.

    Returns
    -------
    int
        Its rank, reading a nested sequence along its first entries.
    """
    shape = getattr(value, "shape", None)
    if shape is not None:
        return len(shape)
    rank = 0
    current = value
    while isinstance(current, (tuple, list)):
        rank += 1
        if not current:
            break
        current = current[0]
    return rank


def broadcast_shared_data(
    entry: ProgramEntry, data: Mapping[str, object]
) -> dict[str, object]:
    """Spread each shared datum across the plate of the draw it fills.

    A parameter that fills a family parameter of a plated draw directly
    (`ProgramParameter.per_row`) is typed with the draw's plate as its
    leading axis. A datum supplied without that axis is one value every
    row shares, and is replicated once per row so that it inhabits the
    checked type.

    Parameters
    ----------
    entry : ProgramEntry
        The program's entry point.
    data : Mapping[str, object]
        The host value of each parameter, by name.

    Returns
    -------
    dict[str, object]
        The data, each shared datum replicated across its plate.
    """
    spread = dict(data)
    for parameter in entry.parameters:
        if not parameter.per_row or parameter.name not in spread:
            continue
        shape = tensor_shape(parameter.type)
        if shape is None or not shape[1]:
            continue
        plate = shape[1][0]
        value = spread[parameter.name]
        if (
            isinstance(plate, IndexLiteral)
            and isinstance(plate.value, int)
            and _host_rank(value) == len(shape[1]) - 1
        ):
            spread[parameter.name] = tuple(value for _ in range(plate.value))
    return spread


def static_arguments_for(
    entry: ProgramEntry,
    data: Mapping[str, object],
    given: Sequence[StaticArgument] = (),
) -> tuple[StaticArgument, ...]:
    """Read a program's static extents off the data it is applied to.

    Parameters
    ----------
    entry : ProgramEntry
        The program's entry point, whose telescope names the extents.
    data : Mapping[str, object]
        The host value of each parameter, by name.
    given : Sequence[StaticArgument]
        Static arguments supplied explicitly, one per telescope binder
        in telescope order, or none. A template's object parameter is
        an extent no data fixes, so a run of the template supplies it;
        a binder the data does fix must agree with what is supplied.

    Returns
    -------
    tuple[StaticArgument, ...]
        One index literal per telescope binder, in telescope order: the
        length of the axis of the first parameter whose tensor type is
        shaped by the binder, else the supplied argument.

    Raises
    ------
    KeyError
        If a binder shapes no parameter and none is supplied, or the
        parameter is not given or has too few axes, or a supplied
        argument disagrees with the data, or the supplied arguments do
        not number the telescope.
    """
    if given and len(given) != len(entry.telescope):
        raise KeyError(
            f"program {entry.name!r} binds {len(entry.telescope)} extent(s); "
            f"{len(given)} static argument(s) were given"
        )
    arguments: list[StaticArgument] = []
    for position, binder in enumerate(entry.telescope):
        supplied = given[position] if given else None
        found: int | None = None
        for parameter in entry.parameters:
            type_ = parameter.type
            if not isinstance(type_, TypeApplication) or len(type_.arguments) != 2:
                continue
            shape = type_.arguments[1]
            if not isinstance(shape, ShapeIndex):
                continue
            for depth, dimension in enumerate(shape.dimensions):
                if (
                    isinstance(dimension, IndexVariable)
                    and dimension.name == binder.name
                ):
                    try:
                        found = _extent_of(data[parameter.name], depth)
                    except KeyError as error:
                        raise KeyError(
                            f"parameter {parameter.name!r} of program "
                            f"{entry.name!r} does not fix its extent "
                            f"{binder.name!r}: {error.args[0]}"
                        ) from error
                    break
            if found is not None:
                break
        if found is None:
            if supplied is None:
                raise KeyError(
                    f"program {entry.name!r} has an extent {binder.name!r} that "
                    "no parameter fixes; supply it as a static argument"
                )
            arguments.append(supplied)
            continue
        literal = IndexLiteral(value=found, sort=binder.sort)  # type: ignore[arg-type]
        if supplied is not None and supplied != literal:
            raise KeyError(
                f"static argument {binder.name!r} of program {entry.name!r} is "
                f"given as {render_static(supplied)}, but its data fix it at "
                f"{found}"
            )
        arguments.append(literal)
    return tuple(arguments)


def run_program(
    module: QiecModule,
    name: str,
    *,
    data: Mapping[str, object],
    sites: Mapping[str, object],
    parameters: Mapping[str, object] | None = None,
    static_arguments: Sequence[StaticArgument] = (),
    fuel: int | None = None,
    observer: TraceObserver | None = None,
    providers: tuple[RuntimeProvider, ...] = (),
) -> ProgramRun:
    """Run a program with every site replayed and score its log joint.

    Parameters
    ----------
    module : QiecModule
        The module holding the program.
    name : str
        The program's name.
    data : Mapping[str, object]
        A host value for each of the program's parameters, by name:
        its data, observations, index maps, and map numbers.
    sites : Mapping[str, object]
        A host value for each sampled site, by label, or by label and
        occurrence as ``"<label>@<n>"`` for a site a recurrence reaches
        at every position. Every site the program reaches must be
        given, and every given site reached.
    parameters : Mapping[str, object] | None
        The learned weights a deduction the program calls reads, by
        name; an absent weight reads as zero.
    static_arguments : Sequence[StaticArgument]
        The program's static extents, one per telescope binder, for an
        extent no data fixes (a template's object parameter); empty
        when the data fix every extent.
    fuel : int | None
        A step budget for the run.
    observer : TraceObserver | None
        A callback receiving every trace event as it is emitted.
    providers : tuple[RuntimeProvider, ...]
        Additional typed providers used by computations the program calls.

    Returns
    -------
    ProgramRun
        The returned value and the log joint.

    Raises
    ------
    KeyError
        If the module has no such program, a parameter is not given, or
        an open extent cannot be read off the data.
    ExecutionFailure
        If the run fails, a site is missing, or a value does not inhabit
        its type.
    """
    entry = program_entry(module, name)
    extended, wrapper_name = joint_module(module, name)
    configured: dict[str, object] = {
        REPLAY_HANDLER: {
            "kind": "replay",
            "values": dict(sites),
            "score_instance": entry.score_instance,
        },
        ACCUMULATE_HANDLER: {"kind": "score"},
    }
    return _run_wrapped(
        module,
        extended,
        wrapper_name,
        entry,
        configured,
        data=data,
        parameters=parameters,
        static_arguments=static_arguments,
        fuel=fuel,
        observer=observer,
        providers=providers,
    )


def sample_program(
    module: QiecModule,
    name: str,
    *,
    data: Mapping[str, object],
    sites: Mapping[str, object] | None = None,
    parameters: Mapping[str, object] | None = None,
    static_arguments: Sequence[StaticArgument] = (),
    fuel: int | None = None,
    observer: TraceObserver | None = None,
    providers: tuple[RuntimeProvider, ...] = (),
) -> ProgramRun:
    """Run a program forward, drawing every site it is not given.

    Parameters
    ----------
    module : QiecModule
        The module holding the program.
    name : str
        The program's name.
    data : Mapping[str, object]
        A host value for each of the program's parameters, by name:
        its data, observations, index maps, and map numbers.
    sites : Mapping[str, object] | None
        A host value for each conditioned site, by label, or by label
        and occurrence as ``"<label>@<n>"``; every other site is drawn
        from its distribution. A given site the program never reaches
        is an error.
    parameters : Mapping[str, object] | None
        The learned weights a deduction the program calls reads, by
        name; an absent weight reads as zero.
    static_arguments : Sequence[StaticArgument]
        The program's static extents, one per telescope binder, for an
        extent no data fixes; empty when the data fix every extent.
    fuel : int | None
        A step budget for the run.
    observer : TraceObserver | None
        A callback receiving every trace event as it is emitted.
    providers : tuple[RuntimeProvider, ...]
        Additional typed providers used by computations the program calls.

    Returns
    -------
    ProgramRun
        The returned value and the log joint of the drawn and given
        sites together with the program's observations and scores.

    Raises
    ------
    KeyError
        If the module has no such program, a parameter is not given, or
        an open extent cannot be read off the data.
    ExecutionFailure
        If the run fails or a value does not inhabit its type.
    """
    entry = program_entry(module, name)
    extended, wrapper_name = sample_module(module, name)
    configured: dict[str, object] = {
        CONDITION_HANDLER: {
            "kind": "replay",
            "values": dict(sites or {}),
            "score_instance": entry.score_instance,
        },
        DRAW_HANDLER: {"kind": "draw-scoring", "score_instance": entry.score_instance},
        ACCUMULATE_HANDLER: {"kind": "score"},
    }
    return _run_wrapped(
        module,
        extended,
        wrapper_name,
        entry,
        configured,
        data=data,
        parameters=parameters,
        static_arguments=static_arguments,
        fuel=fuel,
        observer=observer,
        providers=providers,
    )


def passed_data_notes(module: QiecModule, name: str) -> dict[str, str]:
    """Where each datum a program passes straight to a computation goes.

    A free name passed only as a call's argument takes the type of the
    parameter it fills; a call does not map the computation over the
    entries of a tensor. When a datum of another shape is supplied, the
    refusal says which call fixed the datum's type and what to write
    instead.

    Parameters
    ----------
    module : QiecModule
        The module holding the program.
    name : str
        The program's name.

    Returns
    -------
    dict[str, str]
        For each data parameter some call reads as a whole argument, a
        located note naming the call, the parameter it fills, and the
        remedy: declare the parameter as a tensor and ``map`` over it
        inside the computation.
    """
    entry = program_entry(module, name)
    data = {
        parameter.name
        for parameter in entry.parameters
        if parameter.role in ("data", "index_map")
    }
    body = _computation(module, entry.computation).body
    notes: dict[str, str] = {}
    pending: list[Computation] = [body]
    while pending:
        node = pending.pop()
        if not isinstance(node, Bind):
            continue
        pending.append(node.then)
        first = node.first
        if isinstance(first, Bind):
            pending.append(first)
            continue
        if not isinstance(first, Call):
            continue
        callee = _computation(module, first.callee)
        for argument, parameter in zip(first.arguments, callee.parameters):
            if not isinstance(argument, Var) or argument.local.name not in data:
                continue
            datum = argument.local.name
            where = (
                f"line {first.origin.line}, col {first.origin.column}: "
                if first.origin.line
                else ""
            )
            notes.setdefault(
                datum,
                f"{where}`let {node.binder.name} <- {first.name}(...)` passes "
                f"{datum!r} to the parameter `{parameter.name} : "
                f"{render_static(parameter.type)}` of `{first.name}`, which "
                f"fixes its type; a call does not map a computation over a "
                f"tensor's entries: declare the parameter as a tensor and "
                f"`map` over it inside the computation",
            )
    return notes


def _refuse_unmapped_data(
    module: QiecModule, entry: ProgramEntry, data: Mapping[str, object]
) -> None:
    """Refuse a datum whose shape is not that of the parameter it fills.

    Parameters
    ----------
    module : QiecModule
        The module holding the program.
    entry : ProgramEntry
        The program's entry point.
    data : Mapping[str, object]
        The supplied data, by name.

    Raises
    ------
    ExecutionFailure
        Located at the call that fixes a datum's type, when the datum
        has more or fewer axes than that type.
    """
    notes = passed_data_notes(module, entry.name)
    for parameter in entry.parameters:
        note = notes.get(parameter.name)
        if note is None or parameter.name not in data:
            continue
        split = tensor_shape(parameter.type)
        expected = 0 if split is None else len(split[1])
        supplied = _host_rank(data[parameter.name])
        if supplied == expected:
            continue
        raise ExecutionFailure(
            ExecutionDiagnostic(
                code="qiec-run-type",
                message=f"datum {parameter.name!r} has {supplied} axes, but the program "
                f"reads it at type {render_static(parameter.type)}: {note}",
                computation=entry.name,
            )
        )


def _run_wrapped(
    module: QiecModule,
    extended: QiecModule,
    wrapper_name: str,
    entry: ProgramEntry,
    configured: dict[str, object],
    *,
    data: Mapping[str, object],
    parameters: Mapping[str, object] | None,
    static_arguments: Sequence[StaticArgument],
    fuel: int | None,
    observer: TraceObserver | None,
    providers: tuple[RuntimeProvider, ...],
) -> ProgramRun:
    """Run a program's wrapper under the core provider.

    Parameters
    ----------
    module : QiecModule
        The module holding the program.
    extended : QiecModule
        The module extended with the wrapper.
    wrapper_name : str
        The wrapper computation's name.
    entry : ProgramEntry
        The program's entry point.
    configured : dict[str, object]
        The core configuration of the wrapper's handlers, by name; a
        handler the wrapper did not add is dropped.
    data : Mapping[str, object]
        A host value for each of the program's parameters, by name.
    parameters : Mapping[str, object] | None
        The learned weights the program's parameter store answers.
    static_arguments : Sequence[StaticArgument]
        The program's static extents supplied explicitly, one per
        telescope binder, or none.
    fuel : int | None
        A step budget for the run.
    observer : TraceObserver | None
        A callback receiving every trace event as it is emitted.
    providers : tuple[RuntimeProvider, ...]
        Additional typed providers used by computations the program calls.

    Returns
    -------
    ProgramRun
        The returned value and the accumulated log weight.

    Raises
    ------
    KeyError
        If a parameter is not given or an open extent cannot be read
        off the data.
    ExecutionFailure
        If the run fails.
    """
    data = broadcast_shared_data(entry, data)
    _refuse_unmapped_data(module, entry, data)
    arguments = tuple(data[parameter.name] for parameter in entry.parameters)
    statics = static_arguments_for(entry, data, static_arguments)
    configured = {
        handler_name: options
        for handler_name, options in configured.items()
        if any(handler.name == handler_name for handler in extended.handlers)
    }
    if any(handler.name == PARAMS_HANDLER for handler in extended.handlers):
        configured[PARAMS_HANDLER] = {"kind": "param", "values": dict(parameters or {})}
    configured.update(foreign_handler_configuration(module))
    runtime = RuntimeConfiguration(
        selections=(
            RuntimeSelection(provider="core", options={"handlers": configured}),
        ),
        providers=providers,
    )
    result = run_named(
        extended,
        wrapper_name,
        arguments,
        static_arguments=statics,
        runtime=runtime,
        fuel=fuel,
        observer=observer,
    )
    value, weight = result.value  # type: ignore[misc]
    return ProgramRun(value=value, log_joint=float(weight), result=result)  # type: ignore[arg-type]


def foreign_handler_configuration(module: QiecModule) -> dict[str, object]:
    """The runtime configuration of a module's elaborated foreign handlers.

    Parameters
    ----------
    module : QiecModule
        The module.

    Returns
    -------
    dict[str, object]
        By handler name: the enumeration handlers of marginalization
        blocks, the collecting handlers of those blocks and of
        deductions, and the search handlers of deductions, each with
        the options its name records.
    """
    configured: dict[str, object] = {}
    for handler in module.handlers:
        if handler.implementation != "foreign":
            continue
        marginal = _marginal_handler(handler.name)
        deduction = _deduction_handler(handler.name)
        if handler.effect.name == "Random" and marginal is not None:
            grouped, reduction = marginal
            configured[handler.name] = {
                "kind": "enumerate",
                "grouped": grouped,
                "reduction": reduction,
            }
        elif handler.effect.name == "Weight" and handler.name.endswith(
            "collect_marginal"
        ):
            configured[handler.name] = {"kind": "collect"}
        elif deduction is not None:
            kind, semiring = deduction
            if kind == "search":
                configured[handler.name] = {
                    "kind": "search",
                    "reduction": SEARCH_REDUCTION_OF[semiring],
                }
            else:
                configured[handler.name] = {
                    "kind": "collect",
                    "combine": COLLECT_COMBINE_OF[semiring],
                    "forkable": True,
                }
    return configured


#: The search handler's reduction per deduction semiring.
SEARCH_REDUCTION_OF: dict[str, str] = {
    "logprob": "logsumexp",
    "viterbi": "max",
    "boolean": "or",
    "counting": "sum",
}
#: The collecting handler's multiplication per deduction semiring.
COLLECT_COMBINE_OF: dict[str, str] = {
    "logprob": "add",
    "viterbi": "add",
    "boolean": "and",
    "counting": "mul",
}


def _deduction_handler(name: str) -> tuple[str, str] | None:
    """Read a deduction handler's kind and semiring off its name.

    The elaborator names a deduction's handlers
    ``<deduction>__search_<semiring>`` and
    ``<deduction>__collect_<semiring>``.

    Parameters
    ----------
    name : str
        The handler's name.

    Returns
    -------
    tuple[str, str] | None
        ``("search", semiring)`` or ``("collect", semiring)``, or
        ``None`` for a handler of another kind.
    """
    for kind in ("search", "collect"):
        marker = f"__{kind}_"
        if marker in name:
            semiring = name.rsplit(marker, 1)[1]
            if semiring in SEARCH_REDUCTION_OF:
                return kind, semiring
    return None


class DeductionRun(QiecModel):
    """The outcome of running a deduction on the reference machine.

    Parameters
    ----------
    weight
        The inside weight of the goal in the deduction's semiring: a
        log weight, a Boolean, or a count.
    result
        The machine's full result, with its trace.
    """

    weight: HeldValue = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    result: ExecutionResult = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


def run_deduction(
    module: QiecModule,
    name: str,
    *,
    tokens: Sequence[str] | None = None,
    axioms: Sequence[object] | None = None,
    axiom_weights: Sequence[object] | None = None,
    parameters: Mapping[str, object] | None = None,
    fuel: int | None = None,
) -> DeductionRun:
    """Run a deduction's entry on the reference machine.

    Parameters
    ----------
    module : QiecModule
        The module holding the deduction.
    name : str
        The deduction's name.
    tokens : Sequence[str] | None
        The sentence, for a deduction with a lexicon.
    axioms : Sequence[object] | None
        The axiom items, for a deduction without one; runtime
        constructor values of its item family.
    axiom_weights : Sequence[object] | None
        One weight per axiom in the semiring's carrier.
    parameters : Mapping[str, object] | None
        The learned weights by name; an absent weight reads as zero.
    fuel : int | None
        A step budget for the run.

    Returns
    -------
    DeductionRun
        The goal's inside weight.

    Raises
    ------
    KeyError
        If the module has no such deduction.
    ValueError
        If the input does not fit the deduction: tokens for one that
        takes axioms, or the reverse.
    ExecutionFailure
        If the run fails.
    """
    entry_name = f"{name}__derive"
    if not any(item.name == entry_name for item in module.computations):
        raise KeyError(f"module {module.module!r} has no deduction {name!r}")
    computation = next(
        item for item in module.computations if item.name == f"{name}__run"
    )
    takes_tokens = len(computation.parameters) == 1
    if takes_tokens and (tokens is None or axioms is not None):
        raise ValueError(f"deduction {name!r} takes a sentence of tokens")
    if not takes_tokens and (
        axioms is None or axiom_weights is None or tokens is not None
    ):
        raise ValueError(f"deduction {name!r} takes its axioms and their weights")
    extended, wrapper_name = _deduction_wrapper(module, computation)
    if takes_tokens:
        assert tokens is not None
        arguments: tuple[object, ...] = (tuple(tokens),)
        extent = len(tokens)
    else:
        assert axioms is not None and axiom_weights is not None
        arguments = (tuple(axioms), tuple(axiom_weights))
        extent = len(axioms)
    binder = computation.telescope[0]
    assert isinstance(binder, IndexBinder)
    configured: dict[str, object] = {}
    if wrapper_name != computation.name:
        configured[PARAMS_HANDLER] = {"kind": "param", "values": dict(parameters or {})}
    configured.update(foreign_handler_configuration(module))
    result = run_named(
        extended,
        wrapper_name,
        arguments,
        static_arguments=(IndexLiteral(value=extent, sort=binder.sort),),  # type: ignore[arg-type]
        runtime=RuntimeConfiguration(
            selections=(
                RuntimeSelection(provider="core", options={"handlers": configured}),
            )
        ),
        fuel=fuel,
    )
    return DeductionRun(weight=result.value, result=result)


def deduction_item(
    module: QiecModule, name: str, symbol: str, *fields: object
) -> RuntimeConstructor:
    """Build an item of a deduction's family as a host value.

    Parameters
    ----------
    module : QiecModule
        The module holding the deduction.
    name : str
        The deduction's name.
    symbol : str
        The atom or constructor symbol, as the deduction spells it.
    *fields : object
        The constructor's fields: integers for position slots, items for
        the rest.

    Returns
    -------
    RuntimeConstructor
        The item, which ``run_deduction`` takes among the axioms of a
        deduction without a lexicon.

    Raises
    ------
    KeyError
        If the deduction declares no such symbol.
    ValueError
        If the field count differs from the constructor's arity.
    """
    constructor_name = f"{name}__{symbol}"
    constructor = next(
        (item for item in module.constructors if item.name == constructor_name), None
    )
    if constructor is None:
        raise KeyError(f"deduction {name!r} declares no symbol {symbol!r}")
    if len(fields) != len(constructor.fields):
        raise ValueError(
            f"{symbol!r} of deduction {name!r} takes {len(constructor.fields)} "
            f"field(s), not {len(fields)}"
        )
    family = next(item for item in module.families if item.id == constructor.family)
    return RuntimeConstructor(
        constructor=constructor.id,
        static_arguments=(),
        fields=tuple(fields),
        result_type=TypeApplication(constructor=family.type_constructor, arguments=()),
    )


def _deduction_wrapper(
    module: QiecModule, computation: NamedComputation
) -> tuple[QiecModule, str]:
    """Extend a module with a computation serving a deduction's parameters.

    Parameters
    ----------
    module : QiecModule
        The module.
    computation : NamedComputation
        The deduction's entry.

    Returns
    -------
    tuple[QiecModule, str]
        The extended module and the wrapper's name; the module itself
        and the entry's name when the deduction reads no learned weight.
    """
    params = _params_entry(module, computation.type.effects)
    statics: tuple[StaticArgument, ...] = tuple(
        IndexVariable(name=binder.name, sort=binder.sort)
        for binder in computation.telescope
        if isinstance(binder, IndexBinder)
    )
    if params is None:
        return module, computation.name
    store = HandlerDef(
        id=HandlerId.derive(module.module, "handler", PARAMS_HANDLER),
        name=PARAMS_HANDLER,
        effect=params.effect,
        clauses=(HandlerClauseDef(operation=PARAM_GET, grade=ResumptionGrade.LINEAR),),
        input_type=computation.type.result,
        output_type=computation.type.result,
        introduced=EffectRow(),
        total=True,
        implementation="foreign",
        telescope=computation.telescope,
    )
    parameters = tuple(
        Local(name=parameter.name, type=parameter.type)
        for parameter in computation.parameters
    )
    call = Call(
        callee=computation.id,
        name=computation.name,
        static_arguments=statics,
        arguments=tuple(Var(local=parameter) for parameter in parameters),
        result_type=computation.type.result,
        effects=computation.type.effects,
        origin=SourceOrigin(
            module=module.module,
            structural_path=("deductions", computation.name, "wrapper", "call"),
            role="call",
            source_protocol=module.source_protocol,
        ),
    )
    wrapper_name = f"__deduction_{computation.name}"
    wrapper = NamedComputation(
        id=ComputationId.derive(module.module, "computation", wrapper_name),
        name=wrapper_name,
        telescope=computation.telescope,
        parameters=parameters,
        body=Handle(
            instance=params.instance,
            handler=store.id,
            computation=call,
            static_arguments=statics,
        ),
        type=ComputationType(effects=EffectRow(), result=computation.type.result),
        origin=SourceOrigin(
            module=module.module,
            structural_path=("deductions", computation.name, "wrapper"),
            role="computation",
            source_protocol=module.source_protocol,
        ),
    )
    extended = module.with_(
        handlers=(*module.handlers, store),
        computations=(*module.computations, wrapper),
    )
    return extended, wrapper_name


__all__ = [
    "DeductionRun",
    "ProgramRun",
    "broadcast_shared_data",
    "deduction_item",
    "foreign_handler_configuration",
    "joint_module",
    "program_entry",
    "run_deduction",
    "run_program",
    "sample_module",
    "sample_program",
    "static_arguments_for",
]
