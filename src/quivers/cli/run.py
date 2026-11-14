"""``qvr run``: execute one entry point of a checked module.

An entry point is a ``define`` computation or a ``program``; both are
invoked through [`invoke_entry`][quivers.qiec.entries.invoke_entry], which
the REPL's ``:run`` and a Python caller share, so an entry validates its
arguments, selects its providers, traces, and fails with the same codes
however it is reached.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from quivers.dsl import Compiler, ParseError, parse
from quivers.qiec import (
    ExecutionDiagnostic,
    ExecutionFailure,
    RuntimeConfiguration,
    entry_points,
    invoke_entry,
    load_runtime_configuration,
    parse_bindings,
    render_entry,
)
from quivers.qiec.entries import json_value


def _argument(text: str) -> object:
    """Read one positional value argument.

    Parameters
    ----------
    text : str
        The argument as typed.

    Returns
    -------
    object
        The host value.

    Raises
    ------
    ValueError
        If the text is not JSON.
    """
    try:
        return json_value(json.loads(text))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON value argument {text!r}: {error.msg}"
        ) from error


def _emit_failure(
    diagnostic: ExecutionDiagnostic,
    *,
    json_output: bool,
) -> int:
    """Report a failed invocation.

    Parameters
    ----------
    diagnostic : ExecutionDiagnostic
        The failure.
    json_output : bool
        Whether to write JSON to standard output rather than a line to
        standard error.

    Returns
    -------
    int
        The exit status, always 1.
    """
    if json_output:
        sys.stdout.write(
            json.dumps({"ok": False, "diagnostics": [diagnostic.to_data()]}, indent=2)
            + "\n"
        )
    else:
        origin = diagnostic.origin
        location = ""
        if origin is not None and origin.file:
            location = origin.file
            if origin.line is not None:
                location += f":{origin.line}:{origin.column or 0}"
            location += ": "
        sys.stderr.write(
            f"{location}{diagnostic.severity}[{diagnostic.code}]: "
            f"{diagnostic.message}\n"
        )
    return 1


def main(args: object) -> int:
    """Execute parsed argparse ``run`` arguments.

    Parameters
    ----------
    args : object
        The parsed arguments: the file, the entry name, its positional
        arguments, and the options.

    Returns
    -------
    int
        The exit status: 0 on success, 1 on any failure.
    """

    path = Path(str(getattr(args, "file")))
    json_output = bool(getattr(args, "json", False))
    computation = getattr(args, "computation", None)
    name = str(computation) if computation is not None else ""
    try:
        source = path.read_bytes()
        parsed = parse(source, file_path=str(path))
        compiler = Compiler(parsed, module_name=path.stem, file_path=str(path))
        module = compiler.qiec_module
        if module is None:
            raise ExecutionFailure(
                ExecutionDiagnostic(
                    code="qiec-run-module",
                    message=f"{path} contains no executable entry points",
                    computation=name,
                )
            )
        if computation is None or bool(getattr(args, "list", False)):
            points = entry_points(module)
            if json_output:
                sys.stdout.write(
                    json.dumps(
                        {
                            "ok": True,
                            "entries": [
                                json.loads(point.model_dump_json()) for point in points
                            ],
                        },
                        indent=2,
                    )
                    + "\n"
                )
            else:
                for point in points:
                    sys.stdout.write(render_entry(point) + "\n")
            return 0
        runtime_path = getattr(args, "runtime", None)
        runtime = (
            load_runtime_configuration(runtime_path)
            if runtime_path
            else RuntimeConfiguration()
        )
        arguments = tuple(_argument(item) for item in getattr(args, "arguments", ()))
        data = parse_bindings(tuple(getattr(args, "data", ()) or ()))
        sites = parse_bindings(tuple(getattr(args, "site", ()) or ()))
        fuel = getattr(args, "fuel", None)
        if fuel is not None and fuel <= 0:
            raise ValueError("--fuel must be a positive number of steps")
        seed = getattr(args, "seed", None)
        run = invoke_entry(
            module,
            name,
            arguments,
            data=data,
            sites=sites,
            static_arguments=tuple(getattr(args, "static", ()) or ()),
            runtime=runtime,
            fuel=fuel,
            seed=int(seed) if seed is not None else None,
        )
    except ExecutionFailure as error:
        return _emit_failure(error.diagnostic, json_output=json_output)
    except ParseError as error:
        return _emit_failure(
            ExecutionDiagnostic(code="parse", message=str(error), computation=name),
            json_output=json_output,
        )
    except (OSError, ValueError, KeyError) as error:
        return _emit_failure(
            ExecutionDiagnostic(
                code="qiec-run-config", message=str(error), computation=name
            ),
            json_output=json_output,
        )
    except Exception as error:
        return _emit_failure(
            ExecutionDiagnostic(code="compile", message=str(error), computation=name),
            json_output=json_output,
        )

    payload = run.to_data()
    if json_output:
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        line = f"{run.entry.name} = {payload['value']!r} : {payload['result_type']}"
        if run.log_joint is not None:
            line += f"  log_joint = {run.log_joint!r}"
        sys.stdout.write(line + "\n")
        if bool(getattr(args, "trace", False)):
            for event in run.result.trace:
                sys.stderr.write(
                    f"trace[{event.sequence}] {event.event} "
                    f"{json.dumps(dict(event.detail), default=repr, sort_keys=True)}\n"
                )
    return 0


__all__ = []
