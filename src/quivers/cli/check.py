"""``qvr check``: parse and compile .qvr files and report diagnostics.

Implementation
--------------

For each input file:

1. ``parse(source)`` is invoked; ``ParseError`` produces a
   ``code="parse"`` diagnostic.
2. ``Compiler(module).compile()`` is invoked; ``CompileError``
   produces a ``code="compile"`` diagnostic.
3. The constraint solver in [`quivers.dsl.constraints`][quivers.dsl.constraints] walks the
   parsed AST for residuated-universe and effect-typed-application
   well-formedness violations; each emits a ``code="residuated_constraint"``
   or ``code="effect_constraint"`` diagnostic.

Successful files emit no diagnostics; on success the human-readable
mode prints ``"OK file.qvr"``.

Exit codes:

- ``0``: every file compiled without diagnostics,
- ``1``: at least one file produced an ``error`` diagnostic,
- ``2``: usage / IO error.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Literal

import didactic.api as dx

from quivers.dsl import (
    CompileError,
    Compiler,
    ParseError,
    non_qiec_projection,
    parse,
)
from quivers.dsl.compiler._validate import validate_family_arg_shapes
from quivers.dsl.constraints import check_constraints
from quivers.dsl.qiec_tooling import (
    has_qiec_surface,
    qiec_diagnostic,
    qiec_module_name,
)
from quivers.transpile import UnsupportedConstruct
from quivers.transpile.plan import checked_program_plan
from quivers.transpile.qiec_ir import analyze_qiec_capabilities

type CheckSeverity = Literal["error", "warning", "note"]
"""The severity of a `CheckDiagnostic`."""


_PARSE_POSITION = re.compile(r"line\s+(\d+),\s*col\s+(\d+)")


class CheckDiagnostic(dx.Model):
    """One structured diagnostic that ``qvr check`` reports for a file.

    Parameters
    ----------
    file : str
        The path of the checked file.
    line : int
        The 1-indexed source line, or ``0`` when the diagnostic has no
        position.
    col : int
        The 1-indexed source column, or ``0`` when the diagnostic has
        no position.
    severity : CheckSeverity
        ``"error"``, ``"warning"``, or ``"note"``.
    code : str
        The stable diagnostic code, such as ``"parse"`` or
        ``"compile"``.
    message : str
        The human-readable description.
    """

    file: str
    line: int
    col: int
    severity: CheckSeverity
    code: str
    message: str


def _normalize_severity(value: str) -> CheckSeverity:
    """Coerce a raw severity string into the typed `CheckSeverity` literal,
    raising on an unknown value."""
    if value == "error":
        return "error"
    if value == "warning":
        return "warning"
    if value == "note":
        return "note"
    raise ValueError(f"unknown severity {value!r}")


def check_file(path: Path, *, target: str | None = None) -> list[CheckDiagnostic]:
    """Parse, constraint-check, and compile one ``.qvr`` file.

    This is the per-file pipeline behind ``qvr check``. It reports
    failures as diagnostics rather than raising, so an unreadable file
    yields an ``io`` diagnostic and a syntax error a ``parse`` one.

    Parameters
    ----------
    path : Path
        The ``.qvr`` file to check.
    target : str or None
        A transpile target whose missing capabilities are also
        reported, as ``qvr check --target`` does; ``None`` for none.

    Returns
    -------
    list[CheckDiagnostic]
        Every diagnostic the file produces, empty when it checks
        cleanly.
    """
    try:
        source = path.read_bytes()
    except OSError as e:
        return [
            CheckDiagnostic(
                file=str(path),
                line=0,
                col=0,
                severity="error",
                code="io",
                message=f"could not read file: {e}",
            )
        ]

    diags: list[CheckDiagnostic] = []

    try:
        module = parse(source, file_path=str(path))
    except ParseError as e:
        match = _PARSE_POSITION.search(str(e))
        line = int(match.group(1)) if match is not None else 0
        col = int(match.group(2)) if match is not None else 0
        diags.append(
            CheckDiagnostic(
                file=str(path),
                line=line,
                col=col,
                severity="error",
                code="parse",
                message=str(e),
            )
        )
        return diags

    compiler_module = non_qiec_projection(module)
    try:
        compiler = Compiler(
            module,
            module_name=qiec_module_name(path),
            file_path=str(path),
        )
        qiec_module = compiler.qiec_module
        if qiec_module is not None and qiec_module.gap:
            diags.append(
                CheckDiagnostic(
                    file=str(path),
                    line=0,
                    col=0,
                    severity="error",
                    code="qiec-program-gap",
                    message=qiec_module.gap,
                )
            )
    except Exception as error:
        if not has_qiec_surface(module):
            raise
        diagnostic = qiec_diagnostic(error)
        diags.append(
            CheckDiagnostic(
                file=str(path),
                line=diagnostic.line,
                col=diagnostic.col,
                severity="error",
                code=diagnostic.code,
                message=diagnostic.message,
            )
        )
        compiler = Compiler(compiler_module, lower_qiec=False)

    # Constraint solver runs before compile so users see structural
    # diagnostics even when compilation would also fail.
    diags.extend(
        CheckDiagnostic(
            file=str(path),
            line=v.line,
            col=v.col,
            severity=_normalize_severity(v.severity),
            code=v.code,
            message=v.message,
        )
        for v in check_constraints(compiler_module)
    )

    # Family argument-shape pass: arity and elementwise compatibility
    # against the distribution's `arg_constraints`. Surfaces the
    # ``implicit-family-defaults`` warning and ``family-arg-shape``
    # errors / warnings.
    family_diags = [
        CheckDiagnostic(
            file=str(path),
            line=v.line,
            col=v.col,
            severity=_normalize_severity(v.severity),
            code=v.code,
            message=v.message,
        )
        for v in validate_family_arg_shapes(compiler_module)
    ]
    diags.extend(family_diags)
    has_family_shape_error = any(
        d.code == "family-arg-shape" and d.severity == "error" for d in family_diags
    )
    if has_family_shape_error:
        # Skip compile: the wider compiler would crash on shape-mismatched
        # args before we get a useful diagnostic.
        return diags

    try:
        compiler.compile()
    except CompileError as error:
        diags.append(
            CheckDiagnostic(
                file=str(path),
                line=getattr(error, "line", 0),
                col=getattr(error, "col", 0),
                severity="error",
                code="compile",
                message=str(error),
            )
        )

    if target is not None and compiler.qiec_module is not None:
        capabilities = analyze_qiec_capabilities(compiler.qiec_module, target)
        for capability in capabilities:
            origin = capability.origin
            diags.append(
                CheckDiagnostic(
                    file=str(path),
                    line=(origin.line or 0) if origin is not None else 0,
                    col=(origin.column or 0) if origin is not None else 0,
                    severity="error",
                    code=capability.kind,
                    message=capability.message,
                )
            )
        if not capabilities:
            try:
                checked_program_plan(compiler.qiec_module, module, target)
            except UnsupportedConstruct as error:
                diags.extend(
                    CheckDiagnostic(
                        file=str(path),
                        line=0,
                        col=0,
                        severity="error",
                        code=kind,
                        message=str(error),
                    )
                    for kind in error.kinds
                )

    return diags


def main(
    files: list[str],
    *,
    json_output: bool = False,
    target: str | None = None,
) -> int:
    """Run ``qvr check`` on a list of paths.

    Parameters
    ----------
    files : list of str
        Paths to ``.qvr`` files.
    json_output : bool
        When True, emit a single JSON document on stdout containing
        the full diagnostic list. When False (default), emit
        human-readable lines.

    Returns
    -------
    int
        Exit code. 0 on full success; 1 on any error diagnostic.
    """
    paths = [Path(f) for f in files]
    all_diags: list[CheckDiagnostic] = []
    for p in paths:
        all_diags.extend(check_file(p, target=target))

    has_error = any(d.severity == "error" for d in all_diags)

    if json_output:
        payload = {
            "files": [str(p) for p in paths],
            "diagnostics": [d.model_dump() for d in all_diags],
            "ok": not has_error,
        }
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
    else:
        files_with_diags: set[str] = set()
        for d in all_diags:
            files_with_diags.add(d.file)
            loc = f"{d.file}:{d.line}:{d.col}" if d.line else d.file
            sys.stderr.write(f"{loc}: {d.severity}[{d.code}]: {d.message}\n")
        for p in paths:
            if str(p) not in files_with_diags:
                sys.stdout.write(f"OK {p}\n")
        if has_error:
            sys.stderr.write(
                f"\n{sum(1 for d in all_diags if d.severity == 'error')} "
                f"error(s) across {len(paths)} file(s)\n"
            )

    return 1 if has_error else 0


__all__ = ["CheckDiagnostic", "CheckSeverity", "check_file"]
