"""Entry points of a checked module and one way to invoke them.

A checked module has two kinds of executable entry point: a named
computation declared with ``define``, run by
[`run_named`][quivers.qiec.execution.run_named] under a runtime
configuration, and a ``program``, elaborated to a computation over the
module's canonical ``random`` and ``score`` instances and run by
[`sample_program`][quivers.qiec.program_runtime.sample_program] with its
data and any conditioned sites. The command line, the REPL, and a Python
caller invoke either through [`invoke_entry`][quivers.qiec.entries.invoke_entry],
so an entry validates its arguments, selects its providers, traces, and
fails with the same codes however it is reached.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal, NoReturn, cast

import didactic.api as dx
import torch

from quivers._values import HeldValue, OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec.execution import (
    ExecutionDiagnostic,
    ExecutionFailure,
    ExecutionResult,
    RuntimeConfiguration,
    TraceObserver,
    parse_static_arguments,
    run_named,
)
from quivers.qiec.distributions import seed_reference_rng
from quivers.qiec.module import NamedComputation, QiecModule
from quivers.qiec.kinds import IndexBinder
from quivers.qiec.program_runtime import (
    broadcast_shared_data,
    program_entry,
    sample_program,
    static_arguments_for,
)
from quivers.qiec.programs import ProgramEntry
from quivers.qiec.serialization import JsonValue
from quivers.qiec.substitution import StaticSubstitution, substitute_type
from quivers.qiec.types import IndexTerm, render_static

type EntryKind = Literal["computation", "program"]
"""How an entry point is declared: with ``define`` or with ``program``."""

type HostValue = (
    bool
    | int
    | float
    | str
    | torch.Tensor
    | tuple["HostValue", ...]
    | Mapping[str, "HostValue"]
)
"""A value the reference machine takes or returns: a scalar, a tensor, a
tuple, or a record of such values."""


class EntryParameter(dx.Model):
    """One value parameter of an entry point.

    Parameters
    ----------
    name
        The parameter's name.
    type
        Its type, rendered in the surface spelling.
    role
        How a program parameter is supplied (its data, an observation, a
        index map, a scalar, or the program's domain); ``"value"`` for a
        computation's parameter.
    """

    name: str
    type: str
    role: str


class EntryPoint(dx.Model):
    """An executable entry point of a checked module.

    Parameters
    ----------
    name
        The entry's source name.
    kind
        Whether it is a ``define`` computation or a ``program``.
    parameters
        Its value parameters, in order.
    statics
        The names of the static binders its telescope takes, in order.
    effects
        The effect instances its body performs, rendered as
        ``"<instance> : <Effect>"`` in name order; empty for a pure
        computation and for a program, whose canonical instances the run
        handles.
    result
        Its result type, rendered.
    sites
        The labels of a program's sample sites, which a run may condition
        on; empty for a computation.
    """

    name: str
    kind: EntryKind
    parameters: tuple[EntryParameter, ...]
    statics: tuple[str, ...]
    effects: tuple[str, ...]
    result: str
    sites: tuple[str, ...]


class EntryRun(QiecModel):
    """What invoking an entry point produced.

    Parameters
    ----------
    entry
        The entry invoked.
    value
        The validated host value the entry returned.
    log_joint
        The log joint density a program run accumulated, or ``None`` for
        a computation.
    result
        The underlying execution result, with its trace.
    """

    entry: EntryPoint = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    value: HeldValue = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_joint: float | None
    result: ExecutionResult = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def to_data(self) -> dict[str, object]:
        """Render the run as JSON-compatible data.

        Returns
        -------
        dict[str, object]
            The execution result's data, with the entry's kind and, for a
            program, the log joint.
        """
        data = self.result.to_data()
        data["kind"] = self.entry.kind
        if self.log_joint is not None:
            data["log_joint"] = self.log_joint
        return data


def _instance_names(module: QiecModule) -> dict[object, str]:
    """The source name of each lexical instance of a module.

    Parameters
    ----------
    module : QiecModule
        The module.

    Returns
    -------
    dict[object, str]
        Instance identity to its declared name.
    """
    return {instance.entry.instance: instance.name for instance in module.instances}


def _computation_entry(module: QiecModule, computation: NamedComputation) -> EntryPoint:
    """Describe a ``define`` computation.

    Parameters
    ----------
    module : QiecModule
        The module declaring it.
    computation : NamedComputation
        The computation.

    Returns
    -------
    EntryPoint
        Its description.
    """
    names = _instance_names(module)
    return EntryPoint(
        name=computation.name,
        kind="computation",
        parameters=tuple(
            EntryParameter(
                name=parameter.name, type=render_static(parameter.type), role="value"
            )
            for parameter in computation.parameters
        ),
        statics=tuple(binder.name for binder in computation.telescope),
        effects=tuple(
            sorted(
                f"{names.get(entry.instance, str(entry.instance))} : {entry.effect.name}"
                for entry in computation.type.effects.entries
            )
        ),
        result=render_static(computation.type.result),
        sites=(),
    )


def _program_entry_point(
    entry: ProgramEntry, computation: NamedComputation
) -> EntryPoint:
    """Describe a program.

    Parameters
    ----------
    entry : ProgramEntry
        The program's elaborated entry.
    computation : NamedComputation
        The computation holding its body.

    Returns
    -------
    EntryPoint
        Its description.
    """
    return EntryPoint(
        name=entry.name,
        kind="program",
        parameters=tuple(
            EntryParameter(
                name=parameter.name,
                type=render_static(parameter.type),
                role=str(parameter.role),
            )
            for parameter in entry.parameters
        ),
        statics=tuple(binder.name for binder in entry.telescope),
        effects=(),
        result=render_static(computation.type.result),
        sites=tuple(site.name for site in entry.sites if site.kind == "sample"),
    )


def entry_points(module: QiecModule) -> tuple[EntryPoint, ...]:
    """Every executable entry point of a module.

    Parameters
    ----------
    module : QiecModule
        The checked module.

    Returns
    -------
    tuple[EntryPoint, ...]
        The programs and the computations, in declaration order; a
        computation a program elaborated to appears once, as the program,
        and a marginalization helper not at all.
    """
    programs = {entry.computation: entry for entry in module.entries}
    helpers = {
        computation.id
        for computation in module.computations
        if computation.name.startswith("__")
    }
    points: list[EntryPoint] = []
    for computation in module.computations:
        program = programs.get(computation.id)
        if program is not None:
            points.append(_program_entry_point(program, computation))
        elif computation.id not in helpers:
            points.append(_computation_entry(module, computation))
    return tuple(points)


def entry_point(module: QiecModule, name: str) -> EntryPoint:
    """One entry point of a module, by name.

    Parameters
    ----------
    module : QiecModule
        The checked module.
    name : str
        The entry's source name.

    Returns
    -------
    EntryPoint
        The entry.

    Raises
    ------
    ExecutionFailure
        With code ``qiec-run-computation`` if the module has no such
        entry.
    """
    for point in entry_points(module):
        if point.name == name:
            return point
    available = ", ".join(point.name for point in entry_points(module)) or "(none)"
    raise ExecutionFailure(
        ExecutionDiagnostic(
            code="qiec-run-computation",
            message=f"unknown entry point {name!r}; available: {available}",
            computation=name,
        )
    )


def invoke_entry(
    module: QiecModule,
    name: str,
    arguments: Sequence[HostValue] = (),
    *,
    data: Mapping[str, HostValue] | None = None,
    sites: Mapping[str, HostValue] | None = None,
    static_arguments: Sequence[str] = (),
    runtime: RuntimeConfiguration | None = None,
    fuel: int | None = None,
    seed: int | None = None,
    observer: TraceObserver | None = None,
) -> EntryRun:
    """Invoke an entry point.

    Parameters
    ----------
    module : QiecModule
        The checked module.
    name : str
        The entry's source name.
    arguments : Sequence[HostValue]
        Host values for the entry's value parameters, in order. A
        program's parameters may instead, or additionally, be given by
        name through ``data``.
    data : Mapping[str, HostValue] | None
        A program's parameters by name: its data, observations,
        index maps, and scalars. A name given both positionally and here
        is an error.
    sites : Mapping[str, HostValue] | None
        The values a program run conditions its sample sites on, by
        label; every other site is drawn. Not accepted for a computation.
    static_arguments : Sequence[str]
        Closed terms instantiating a computation's telescope, as
        ``NAME=TERM`` text in any order. A program's extents are read
        off its data where the data fix them, so a program takes a
        static argument only for an extent no data fixes, a template's
        object parameter.
    runtime : RuntimeConfiguration | None
        The providers a computation runs under; the core provider alone
        by default. A program's handlers are fixed by the run, so only
        the default configuration is accepted for a program.
    fuel : int | None
        A step budget for the run.
    seed : int | None
        A seed for the reference generator every draw of the run uses;
        two runs seeded alike draw alike. ``None`` continues the
        generator's current state.
    observer : TraceObserver | None
        A callback receiving every trace event as it is emitted.

    Returns
    -------
    EntryRun
        The value, the log joint for a program, and the execution result.

    Raises
    ------
    ExecutionFailure
        With code ``qiec-run-computation`` for an unknown entry,
        ``qiec-run-config`` for an option the entry's kind does not
        accept or a parameter given twice, ``qiec-run-arity`` for a
        program parameter not given, and the codes of
        [`run_named`][quivers.qiec.execution.run_named] for the run.
    """
    entry = entry_point(module, name)
    if seed is not None:
        seed_reference_rng(seed)
    if entry.kind == "computation":
        if data:
            _refuse(name, "a computation takes its arguments positionally")
        if sites:
            _refuse(name, "a computation has no sites to condition on")
        statics = parse_static_arguments(module, name, tuple(static_arguments))
        result = run_named(
            module,
            name,
            tuple(arguments),
            static_arguments=statics,
            runtime=runtime,
            observer=observer,
            fuel=fuel,
        )
        return EntryRun(
            entry=entry,
            value=result.value,
            log_joint=None,
            result=result,
        )
    statics = (
        parse_static_arguments(module, name, tuple(static_arguments))
        if static_arguments
        else ()
    )
    if runtime is not None and runtime.selections not in (
        RuntimeConfiguration().selections,
        (),
    ):
        _refuse(
            name,
            "a program's handlers are fixed by the run; runtime selections "
            "must be the default core provider or empty",
        )
    program = program_entry(module, name)
    if len(arguments) > len(program.parameters):
        raise ExecutionFailure(
            ExecutionDiagnostic(
                code="qiec-run-arity",
                message=f"program {name!r} takes {len(program.parameters)} parameters, "
                f"got {len(arguments)} positional arguments",
                computation=name,
            )
        )
    supplied: dict[str, HostValue] = {}
    for parameter, value in zip(program.parameters, arguments):
        supplied[parameter.name] = value
    for parameter_name, value in (data or {}).items():
        if parameter_name in supplied:
            _refuse(name, f"parameter {parameter_name!r} is given twice")
        supplied[parameter_name] = value
    missing = [
        parameter.name
        for parameter in program.parameters
        if parameter.name not in supplied
    ]
    if missing:
        raise ExecutionFailure(
            ExecutionDiagnostic(
                code="qiec-run-arity",
                message=f"program {name!r} is missing parameter(s) {', '.join(missing)}",
                computation=name,
            )
        )
    unknown = [
        parameter_name
        for parameter_name in supplied
        if all(parameter.name != parameter_name for parameter in program.parameters)
    ]
    if unknown:
        _refuse(name, f"program {name!r} has no parameter(s) {', '.join(unknown)}")
    supplied = cast(dict[str, HostValue], broadcast_shared_data(program, supplied))
    try:
        run = sample_program(
            module,
            name,
            data=supplied,
            sites=sites,
            static_arguments=statics,
            fuel=fuel,
            observer=observer,
            providers=runtime.providers if runtime is not None else (),
        )
    except KeyError as error:
        _refuse(name, str(error.args[0]))
    computation = next(
        item for item in module.computations if item.id == program.computation
    )
    # The wrapper returns the value paired with the log weight; the
    # entry's result is the program's value at the program's type,
    # instantiated at the extents the run fixed.
    extents = static_arguments_for(program, supplied, statics)
    result = replace(
        run.result,
        computation=name,
        value=run.value,
        result_type=substitute_type(
            computation.type.result,
            StaticSubstitution(
                indices=tuple(
                    (binder.name, cast(IndexTerm, extent))
                    for binder, extent in zip(program.telescope, extents, strict=True)
                    if isinstance(binder, IndexBinder)
                )
            ),
        ),
    )
    return EntryRun(
        entry=entry,
        value=run.value,
        log_joint=run.log_joint,
        result=result,
    )


def _refuse(name: str, message: str) -> NoReturn:
    """Fail an invocation with a configuration diagnostic.

    Parameters
    ----------
    name : str
        The entry's name.
    message : str
        What was wrong.

    Raises
    ------
    ExecutionFailure
        Always, with code ``qiec-run-config``.
    """
    raise ExecutionFailure(
        ExecutionDiagnostic(code="qiec-run-config", message=message, computation=name)
    )


def parse_bindings(items: Sequence[str]) -> dict[str, HostValue]:
    """Read ``NAME=JSON`` bindings, as the command line and REPL take them.

    Parameters
    ----------
    items : Sequence[str]
        The bindings as typed.

    Returns
    -------
    dict[str, HostValue]
        Name to the parsed value, lists read as tuples.

    Raises
    ------
    ValueError
        If an item has no ``=``, names a binding twice, or carries a
        value that is not JSON.
    """
    bindings: dict[str, HostValue] = {}
    for item in items:
        binding_name, separator, text = item.partition("=")
        binding_name = binding_name.strip()
        if not separator or not binding_name:
            raise ValueError(f"binding {item!r} must have the form NAME=JSON")
        if binding_name in bindings:
            raise ValueError(f"binding {binding_name!r} is given twice")
        try:
            bindings[binding_name] = json_value(json.loads(text))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"binding {binding_name!r} has a value that is not JSON: {error.msg}"
            ) from error
    return bindings


def json_value(value: JsonValue) -> HostValue:
    """Read a parsed JSON value as a host value.

    Parameters
    ----------
    value : JsonValue
        The parsed JSON.

    Returns
    -------
    HostValue
        The value with every list read as a tuple, which is how the
        runtime carries a tensor.

    Raises
    ------
    ValueError
        If the value is ``null``, which no host type inhabits.
    """
    if value is None:
        raise ValueError("null is not a host value")
    if isinstance(value, list):
        return tuple(json_value(item) for item in value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    return value


def render_entry(entry: EntryPoint) -> str:
    """Render an entry point as one signature line.

    Parameters
    ----------
    entry : EntryPoint
        The entry.

    Returns
    -------
    str
        ``name(params) : result`` with the kind, the statics, the effects,
        and a program's sites where present.
    """
    parameters = ", ".join(
        f"{parameter.name} : {parameter.type}" for parameter in entry.parameters
    )
    statics = f"[{', '.join(entry.statics)}]" if entry.statics else ""
    line = f"{entry.kind} {entry.name}{statics}({parameters}) : {entry.result}"
    if entry.effects:
        line += " !{" + ", ".join(entry.effects) + "}"
    if entry.sites:
        line += " sites " + ", ".join(entry.sites)
    return line


__all__ = [
    "EntryKind",
    "EntryParameter",
    "EntryPoint",
    "EntryRun",
    "HostValue",
    "entry_point",
    "entry_points",
    "invoke_entry",
    "json_value",
    "parse_bindings",
    "render_entry",
]
