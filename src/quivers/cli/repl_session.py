"""UI-agnostic REPL engine.

A `ReplSession` owns the in-memory environment that ``qvr repl``,
the Textual TUI, the prompt_toolkit fallback, and the Jupyter kernel
all drive. Every meta-command dispatches to a method on this class and
returns a `ReplResponse`; the frontends decide how to render it.

The session never imports any UI library and never reads from stdin or
writes to stdout. That keeps it fully testable from pytest.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import didactic.api as dx

from quivers.analysis.chain_shape import ChainShape
from quivers.analysis.plate_graph import build_plate_graph
from quivers.analysis.plate_render import (
    render_daft,
    render_dot,
    render_mermaid,
    render_table_plain,
    render_tikz,
)
from quivers.analysis.scope import (
    SCOPE_SEPARATOR,
    ScopedRef,
    find_all_references,
    resolve_scoped_path,
    scope_children,
)
from quivers.cli.repl_browse import (
    _children_for_bundle,
    _children_for_contraction,
    _children_for_decoder,
    _children_for_deduction,
    _children_for_encoder,
    _children_for_loss,
    _children_for_morphism,
    _children_for_object,
    _children_for_program,
    _children_for_rule,
    _children_for_signature,
    _children_for_space,
)
from quivers.core.objects import FinSet
from quivers.dsl import (
    CompileError,
    Compiler,
    ParseError,
    parse,
)
from quivers.dsl.ast_nodes import (
    ExportDecl,
    ExprCompose,
    ExprIdent,
    ExprTensorProduct,
    LetStep,
    MarginalizeStep,
    Module,
    ObjectDecl,
    ObjectExpr,
    ObserveStep,
    ProgramStep,
    SampleStep,
    Statement,
    TypeFromExpr,
    TypeName,
)
from quivers.dsl.ast_nodes.qiec import (
    QiecComputationDecl,
    QiecEffectDecl,
    QiecHandlerDecl,
)
from quivers.dsl.constraints import Violation, check_constraints
from quivers.dsl.emit import draw_call_to_source, module_to_source
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_tooling import (
    analyze_module,
    prelude_effect,
    qiec_binding_candidates,
    qiec_binding_map,
    qiec_bindings,
    qiec_env_kinds,
    qiec_module_name,
    qualified_operation_signature,
    render_effect,
)
from quivers.qiec import (
    EntryRun,
    ExecutionDiagnostic,
    ExecutionFailure,
    ExecutionResult,
    HostValue,
    QiecModule,
    RuntimeConfiguration,
    RuntimeSelection,
    entry_points,
    invoke_entry,
    load_runtime_configuration,
    parse_bindings,
    render_entry,
)
from quivers.qiec.checking import KernelRegistry
from quivers.qiec.effects import render_row
from quivers.qiec.entries import json_value
from quivers.qiec.module import inferred_computation_type, validate_module
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from quivers.transpile.plan import checked_program_plan
from quivers.transpile.qiec_ir import analyze_qiec_capabilities

Severity = Literal["error", "warning", "info", "ok"]


_PARSE_POSITION = re.compile(r"line\s+(\d+),\s*col\s+(\d+)")


class Diagnostic(dx.Model):
    """One structured diagnostic surfaced by the REPL or LSP."""

    message: str
    severity: Severity
    line: int = 0
    col: int = 0
    end_line: int = 0
    end_col: int = 0
    code: str = ""


class ReplResponse(dx.Model):
    """A single command's result, ready for any frontend to render.

    ``body`` is the primary payload, which a frontend renders by
    pairing it with ``body_kind`` (``text`` / ``qvr`` / ``json`` /
    ``markdown``); each frontend is responsible for choosing the
    concrete renderer (Rich syntax block, prompt_toolkit ANSI, etc.)
    from ``body_kind``. ``diagnostics`` carry structured errors and
    warnings.
    """

    body: str = ""
    diagnostics: tuple[Diagnostic, ...] = ()
    body_kind: Literal["text", "qvr", "json", "markdown"] = "text"

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)


class SessionOptions(dx.Model):
    """User-tunable knobs exposed via ``:set``.

    ``theme`` selects a Rich syntax theme for QVR bodies rendered
    via ``:info`` / ``:dump`` etc. Common values: ``ansi_dark``,
    ``ansi_light``, ``monokai``, ``nord``, ``solarized-dark``,
    ``solarized-light``, ``github-dark``.
    """

    highlight: bool = True
    unicode: bool = True
    show_axes: bool = True
    paranoid: bool = False
    autoload_on_save: bool = True
    theme: str = "ansi_dark"
    target: str = ""
    """A transpile target whose capability diagnostics ``:load`` and
    ``:reload`` report beside the checked module's own, as ``qvr check
    --target`` does; empty for none."""


class ReplSession:
    """Stateful evaluator for a single REPL session."""

    def _build_alias_map(self) -> dict[int, str]:
        """``id(obj) -> user-given name`` for every type-level binding.

        Lets the morphism renderer surface ``Item -> H_in`` instead
        of the compiler's resolved ``FinSet 200 -> FinSet 4``.
        """
        if self._compiler is None:
            return {}
        out: dict[int, str] = {}
        for name, obj in self._compiler.objects.items():
            out[id(obj)] = name
        for name, sp in self._compiler.spaces.items():
            out[id(sp)] = name
        return out

    def _pretty(self, obj: Any) -> str:
        return _pretty_object_with_aliases(obj, self._build_alias_map())

    def __init__(self) -> None:
        self._loaded_path: Path | None = None
        self._loaded_source: str = ""
        self._module: Module = Module(statements=())
        self._compiler: Compiler | None = None
        self._qiec_module: QiecModule | None = None
        self._qiec_registry: KernelRegistry | None = None
        self._env: dict[str, Any] = {}
        self._last_diags: tuple[Diagnostic, ...] = ()
        self._runtime = RuntimeConfiguration()
        self._runtime_source: str = "built-in core"
        self._last_run: ExecutionResult | None = None
        self._last_entry: EntryRun | None = None
        self._last_failure: ExecutionDiagnostic | None = None
        self.options = SessionOptions()
        # Track when the loaded file was last read so :reload can be
        # auto-fired on a modified mtime.
        self._loaded_mtime: float | None = None
        # Pinned `:watch EXPR` re-evaluations. The TUI renders these in
        # a dedicated panel; each install + reload re-runs every entry.
        self._watches: list[str] = []
        self._watch_results: dict[str, str] = {}

    # ----- introspection ------------------------------------------------

    @property
    def loaded_path(self) -> Path | None:
        return self._loaded_path

    @property
    def module(self) -> Module:
        return self._module

    @property
    def env(self) -> dict[str, Any]:
        """Live environment dict (every declared atom: objects, spaces,
        morphisms, rules, programs, deductions, signatures, encoders,
        decoders, losses, bundles, contractions)."""
        return dict(self._env)

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        return self._last_diags

    @property
    def runtime_label(self) -> str:
        return self._runtime.label

    @property
    def last_run(self) -> ExecutionResult | None:
        return self._last_run

    @property
    def last_entry(self) -> EntryRun | None:
        """The most recent entry-point run of this session, or ``None``."""
        return self._last_entry

    @property
    def last_failure(self) -> ExecutionDiagnostic | None:
        """The diagnostic of the most recent failed run, or ``None`` once a
        run succeeds after it."""
        return self._last_failure

    def watch_results(self) -> dict[str, str]:
        """Return the current ``expr -> rendered`` map for pinned watches."""
        return dict(self._watch_results)

    def env_kinds(self) -> dict[str, str]:
        """Return a name -> semantic-token-type map for the live env.

        Frontends pass this to the highlighter so an identifier renders
        in its env-known colour everywhere it appears, regardless of
        whether the surrounding grammar context parses cleanly.
        """
        kinds: dict[str, str] = {}
        if self._compiler is not None:
            for name in self._compiler.objects:
                kinds[name] = "type"
            for name in self._compiler.spaces:
                kinds[name] = "type"
            for name in self._compiler.morphisms:
                kinds[name] = "function"
            for name in self._compiler.rules:
                kinds[name] = "namespace"
            for name in self._compiler.programs:
                kinds[name] = "function"
            for name in self._compiler.deductions:
                kinds[name] = "namespace"
            for name in self._compiler.signatures:
                kinds[name] = "type"
            for name in self._compiler.encoders:
                kinds[name] = "function"
            for name in self._compiler.decoders:
                kinds[name] = "function"
            for name in self._compiler.losses:
                kinds[name] = "function"
            for name in self._compiler.bundles:
                kinds[name] = "namespace"
            for name in self._compiler.contractions:
                kinds[name] = "function"
        kinds.update(qiec_env_kinds(self._module))
        return kinds

    # ----- entry points -------------------------------------------------

    def dispatch(self, line: str) -> ReplResponse:
        """Route a raw input line to the right handler."""
        stripped = line.strip()
        if not stripped:
            return ReplResponse()
        if stripped.startswith(":"):
            cmd, _, rest = stripped[1:].partition(" ")
            return self._dispatch_meta(cmd, rest.strip())
        # Bare line: try statement first, then expression.
        return self._eval_source(stripped)

    def _dispatch_meta(self, cmd: str, arg: str) -> ReplResponse:
        method = _META_COMMANDS.get(cmd)
        if method is None:
            return _err(f"unknown command :{cmd}; try :help")
        return method(self, arg)

    # ----- :load / :reload ----------------------------------------------

    def load_file(self, path: str | Path) -> ReplResponse:
        p = Path(path).expanduser()
        if not p.exists():
            return _err(f"file not found: {p}")
        try:
            source = p.read_bytes()
            module = parse(source, file_path=str(p))
        except ParseError as e:
            match = _PARSE_POSITION.search(str(e))
            self._last_diags = (
                Diagnostic(
                    message=str(e),
                    severity="error",
                    code="parse",
                    line=int(match.group(1)) if match is not None else 0,
                    col=int(match.group(2)) if match is not None else 0,
                ),
            )
            return _resp("", self._last_diags)
        self._loaded_source = source.decode("utf-8", errors="replace")
        return self._install_module(module, source_path=p)

    def reload(self) -> ReplResponse:
        if self._loaded_path is None:
            return _err("no file loaded; use :load <FILE>")
        prior = set(self._env)
        result = self.load_file(self._loaded_path)
        if not result.ok:
            return result
        new = set(self._env) - prior
        removed = prior - set(self._env)
        body_lines = [f"reloaded {self._loaded_path}"]
        if new:
            body_lines.append("  added: " + ", ".join(sorted(new)))
        if removed:
            body_lines.append("  removed: " + ", ".join(sorted(removed)))
        return _resp("\n".join(body_lines), result.diagnostics)

    def _install_module(
        self, module: Module, *, source_path: Path | None = None
    ) -> ReplResponse:
        analysis = analyze_module(
            module,
            module_name=qiec_module_name(source_path),
            file_path=str(source_path) if source_path is not None else "<repl>",
        )
        diags: list[Diagnostic] = [
            _violation_to_diag(v) for v in check_constraints(analysis.non_qiec_module)
        ]
        for diagnostic in analysis.qiec_diagnostics:
            diags.append(
                Diagnostic(
                    message=diagnostic.message,
                    severity="error",
                    line=diagnostic.line,
                    col=diagnostic.col,
                    code=diagnostic.code,
                )
            )
        if analysis.compile_error is not None:
            error = analysis.compile_error
            diags.append(
                Diagnostic(
                    message=str(error),
                    severity="error",
                    line=getattr(error, "line", 0),
                    col=getattr(error, "col", 0),
                    code="compile",
                )
            )
        if self.options.target and analysis.qiec_module is not None:
            capabilities = analyze_qiec_capabilities(
                analysis.qiec_module, self.options.target
            )
            for capability in capabilities:
                origin = capability.origin
                diags.append(
                    Diagnostic(
                        message=capability.message,
                        severity="error",
                        line=(origin.line or 0) if origin is not None else 0,
                        col=(origin.column or 0) if origin is not None else 0,
                        code=capability.kind,
                    )
                )
            if not capabilities:
                try:
                    checked_program_plan(
                        analysis.qiec_module, module, self.options.target
                    )
                except UnsupportedConstruct as error:
                    diags.extend(
                        Diagnostic(
                            message=str(error),
                            severity="error",
                            line=0,
                            col=0,
                            code=kind,
                        )
                        for kind in error.kinds
                    )
        self._module = module
        self._compiler = analysis.compiler
        self._qiec_module = analysis.qiec_module
        self._qiec_registry = None
        self._last_run = None
        self._last_entry = None
        self._last_failure = None
        self._env = analysis.env
        self._last_diags = tuple(diags)
        if source_path is not None:
            self._loaded_path = source_path
            try:
                self._loaded_mtime = source_path.stat().st_mtime
            except OSError:
                self._loaded_mtime = None
        self._refresh_watches()
        body = (
            f"loaded {source_path}: {_env_counts(analysis.env)}"
            if source_path is not None
            else f"installed module: {_env_counts(analysis.env)}"
        )
        return _resp(body, tuple(diags))

    # ----- :type / :kind ------------------------------------------------
    #
    # GHCi semantics:
    #   :type EXPR   : print the type of a value-level expression
    #                  (morphism, program, sample site, etc.).
    #                  Refuses type-level names; suggests :kind.
    #   :kind T      : print the kind of a type-level name or type
    #                  expression (object, space, sort, FinSet 3, ...).
    #                  Refuses value-level names; suggests :type.
    #
    # The bare-expression fallback (a name typed without a colon
    # command) lands in ``_describe`` which tries :type first and
    # falls back to :kind, so users keep the "just type a name"
    # ergonomics.

    def type_of(self, expr_source: str) -> ReplResponse:
        """Print the type of a value-level expression.

        Mirrors GHCi's ``:type``: works on morphisms, programs,
        deductions, scoped sample / observe / let sites, and any
        expression that resolves to a morphism. For type-level names
        (objects, spaces, sorts), returns an error directing the user
        to ``:kind``.
        """
        if not self._compiler:
            return _err("no environment loaded; use :load <FILE> first")

        if SCOPE_SEPARATOR in expr_source:
            ref = resolve_scoped_path(self._compiler, expr_source.strip())
            if ref is None:
                return _err(f"unknown path: {expr_source}")
            if _ref_kind_class(ref.kind) == "type":
                return _err(
                    f"{expr_source} is a type, not an expression; use :kind {expr_source}"
                )
            return _resp(self._type_signature_for_ref(ref), body_kind="qvr")

        bare = expr_source.strip()
        qiec = qiec_binding_map(self._module).get(bare)
        if qiec is not None:
            if qiec.kind in {"index", "family", "effect"}:
                return _err(f"{bare} is a type, not an expression; use :kind {bare}")
            return _resp(
                render_qiec_signature(self._module, bare) or f"{bare} :: {qiec.kind}",
                body_kind="qvr",
            )
        qiec_candidates = qiec_binding_candidates(self._module, bare)
        if len(qiec_candidates) > 1:
            choices = ", ".join(item.qualified_name for item in qiec_candidates)
            return _err(f"ambiguous name {bare!r}; use one of: {choices}")
        if prelude_effect(bare) is not None:
            return _err(f"{bare} is a type, not an expression; use :kind {bare}")
        if "." in bare and all(part.isidentifier() for part in bare.split(".")):
            operation = qualified_operation_signature(self._module, bare)
            if operation is not None:
                return _resp(operation, body_kind="qvr")
            owner, _, member = bare.partition(".")
            return _err(
                f"unknown operation {member!r}: {owner!r} is neither an effect "
                "interface nor an instance declaring it"
            )
        if bare.isidentifier():
            line = self._value_line_for_name(bare)
            if line is not None:
                return _resp(line, body_kind="qvr")
            if self._is_type_level_name(bare):
                return _err(f"{bare} is a type, not an expression; use :kind {bare}")

        # Try to resolve as an expression (morphism reference / composition).
        # We drive the compiler's own ``_compile_expr`` inference rather
        # than re-running the full module compile: ``_compile_expr``
        # reads ``_morphisms`` / ``_objects`` / ``_spaces`` but does
        # not mutate them, so the session's already-compiled state is
        # the right environment to ask "what is the type of e?" in.
        try:
            mod = parse(f"export {expr_source}", file_path="<expr>")
        except ParseError as e:
            # If the user wrote a type expression like ``FinSet 3``,
            # parsing as an output binding fails; surface a hint.
            if self._parses_as_type(expr_source):
                return _err(
                    f"{expr_source} is a type expression; use :kind {expr_source}"
                )
            return _err(f"parse error: {e}")
        expr_ast = _extract_export_expr(mod)
        if expr_ast is None:
            return _err("expression did not resolve to a morphism")
        try:
            morph = self._compiler._compile_expr(expr_ast)
        except CompileError as e:
            return _err(f"type error: {e}")
        if morph is None:
            return _err("expression did not resolve to a morphism")
        return _resp(
            self._type_signature_for_morphism(expr_source, morph),
            body_kind="qvr",
        )

    def transpile_module(self, target: str) -> ReplResponse:
        """Emit the loaded module as source for the named transpile
        backend.

        Mirrors the ``qvr transpile`` CLI
        subcommand, dispatched in-session against the currently-loaded
        [`Module`][quivers.dsl.ast_nodes.Module]. Returns the
        transpiled bytes decoded as UTF-8; never mutates session state.

        Errors:

        - No module loaded yet (caller must run ``:load FILE`` first).
        - Empty or whitespace-only target string.
        - Target is not in
          [`available_targets`][quivers.transpile.available_targets].
        - The walker raises
          [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
          (the message carries the offending construct kinds).
        """
        if self._compiler is None:
            return _err("no environment loaded; use :load <FILE> first")
        target = target.strip()
        if not target:
            return _err("usage: :transpile <TARGET>")

        targets = available_targets()
        if target not in targets:
            return _err(f"unknown target {target!r}; available: {', '.join(targets)}")
        try:
            output = transpile(self._module, target=target)
        except UnsupportedConstruct as e:
            return _err(str(e))
        return _resp(output.decode("utf-8"))

    # ----- QIEC execution ----------------------------------------------

    def runtime(self, specification: str = "") -> ReplResponse:
        """Show or replace the explicit runtime-provider configuration."""

        specification = specification.strip()
        if not specification:
            payload = self._runtime.to_data()
            payload["source"] = self._runtime_source
            return _resp(json.dumps(payload, indent=2), body_kind="json")
        path = Path(specification).expanduser()
        try:
            if path.exists():
                self._runtime = load_runtime_configuration(path)
                self._runtime_source = str(path)
            elif path.suffix.lower() == ".json" or "/" in specification:
                return _err(
                    f"runtime configuration not found: {path}", code="qiec-run-config"
                )
            else:
                self._runtime = RuntimeConfiguration(
                    selections=(RuntimeSelection(provider=specification),)
                )
                # Instantiate now so a misspelled provider fails at :runtime,
                # not after the next computation has been prepared.
                self._runtime.instantiate()
                self._runtime_source = f"provider {specification}"
        except (ExecutionFailure, ValueError) as error:
            if isinstance(error, ExecutionFailure):
                return _execution_error(error)
            return _err(str(error), code="qiec-run-config")
        self._last_run = None
        return _resp(f"runtime attached: {self._runtime.label}")

    def detach_runtime(self) -> ReplResponse:
        """Remove every runtime provider from this session."""

        previous = self._runtime.label
        self._runtime = RuntimeConfiguration(selections=())
        self._runtime_source = "detached"
        self._last_run = None
        return _resp(f"runtime detached: {previous}")

    def run_computation(self, invocation: str) -> ReplResponse:
        """Execute ``NAME [JSON ...] [--data NAME=JSON] [--site NAME=JSON]
        [--static NAME=TERM] [--fuel STEPS] [--seed N]``.

        The entry is a ``define`` computation or a ``program``, invoked
        through the same [`invoke_entry`][quivers.qiec.entries.invoke_entry]
        as ``qvr run``, so its validation, providers, trace, and error
        codes are those of the command line. ``:run`` alone lists the
        module's entry points.

        Parameters
        ----------
        invocation : str
            The entry name and its arguments, as typed after ``:run``.

        Returns
        -------
        ReplResponse
            The value, its type, the runtime label, the trace length,
            and a program's log joint; or the diagnostic of a failure.
        """

        if self._qiec_module is None:
            return _err(
                "no checked module loaded; use :load <FILE> first",
                code="qiec-run-module",
            )
        try:
            parts = shlex.split(invocation)
        except ValueError as error:
            return _err(f"invalid :run invocation: {error}", code="qiec-run-config")
        if not parts:
            lines = [render_entry(point) for point in entry_points(self._qiec_module)]
            return _resp("\n".join(lines) if lines else "(no entry points)")
        name = parts.pop(0)
        statics: list[str] = []
        data: list[str] = []
        sites: list[str] = []
        values: list[HostValue] = []
        fuel: int | None = None
        seed: int | None = None
        index = 0
        try:
            while index < len(parts):
                option, separator, inline = parts[index].partition("=")
                if option in ("--static", "--data", "--site", "--fuel", "--seed"):
                    if separator:
                        text = inline
                    else:
                        index += 1
                        if index >= len(parts):
                            raise ValueError(f"{option} requires a value")
                        text = parts[index]
                    if option == "--static":
                        statics.append(text)
                    elif option == "--data":
                        data.append(text)
                    elif option == "--site":
                        sites.append(text)
                    elif option == "--fuel":
                        fuel = _fuel(text)
                    else:
                        seed = int(text)
                else:
                    values.append(json_value(json.loads(parts[index])))
                index += 1
            run = invoke_entry(
                self._qiec_module,
                name,
                tuple(values),
                data=parse_bindings(data),
                sites=parse_bindings(sites),
                static_arguments=tuple(statics),
                runtime=self._runtime,
                fuel=fuel,
                seed=seed,
            )
        except json.JSONDecodeError as error:
            return _err(
                f"value arguments must be JSON: {error.msg}",
                code="qiec-run-config",
            )
        except (ValueError, KeyError) as error:
            self._last_failure = ExecutionDiagnostic(
                code="qiec-run-config", message=str(error), computation=name
            )
            return _err(str(error), code="qiec-run-config")
        except ExecutionFailure as error:
            self._last_failure = error.diagnostic
            return _execution_error(error)
        self._last_run = run.result
        self._last_entry = run
        self._last_failure = None
        payload = run.to_data()
        summary: dict[str, object] = {
            "kind": payload["kind"],
            "value": payload["value"],
            "type": payload["result_type"],
            "runtime": payload["runtime"],
            "trace_events": len(run.result.trace),
        }
        if run.log_joint is not None:
            summary["log_joint"] = run.log_joint
        return _resp(json.dumps(summary, indent=2), body_kind="json")

    def _value_line_for_name(self, bare: str) -> str | None:
        """Return the value-level signature for ``bare``, or None.

        Bare-name fast path shared by ``:type`` and ``_describe``.
        A name is value-level when its declaration denotes a morphism,
        program, encoder/decoder/loss, contraction, transformation,
        rule (rule schema), or the rule-set of a bundle / deduction
        block. Returns a GHCi-shaped ``name :: type`` line.
        """
        if not self._compiler:
            return None
        c = self._compiler
        if bare in c.morphisms:
            return self._type_signature_for_morphism(bare, c.morphisms[bare])
        if bare in c.programs:
            return self._type_signature_for_program(bare, c.programs[bare])
        if bare in c.encoders:
            return f"{bare} :: encoder"
        if bare in c.decoders:
            return f"{bare} :: decoder"
        if bare in c.losses:
            return f"{bare} :: loss"
        if bare in c.contractions:
            return self._type_signature_for_contraction(bare, c.contractions[bare])
        if bare in c.transformations:
            return f"{bare} :: transformation"
        if bare in c.rules:
            return self._type_signature_for_rule(bare, c.rules[bare])
        if bare in c.bundles:
            members = c.bundles[bare]
            return f"{bare} :: {' | '.join(members)}"
        if bare in c.deductions:
            return f"{bare} :: deduction"
        return None

    def _type_line_for_name(self, bare: str) -> str | None:
        """Return the type-level declaration line for ``bare``, or None.

        Type-level names are those whose declaration introduces a
        universe / theory / signature: objects, spaces, signatures
        (generalised algebraic theories), and category atoms. Bundles
        and deductions are namespaces over rule-sets and are treated
        as value-level (they appear in :type, not :kind).
        """
        if not self._compiler:
            return None
        c = self._compiler
        if bare in c.objects:
            obj = c.objects[bare]
            if isinstance(obj, FinSet) and obj.name == bare:
                # A finite set declared by its size carries its own name;
                # its definition is the size.
                return f"object {bare} : FinSet {obj.cardinality}"
            return f"object {bare} : {_pretty_object(obj)}"
        if bare in c.spaces:
            return f"space {bare} : {_pretty_object(c.spaces[bare])}"
        if bare in c.signatures:
            return f"signature {bare}"
        if bare in c.categories:
            return f"category {bare}"
        return None

    def _is_type_level_name(self, bare: str) -> bool:
        if not self._compiler:
            return False
        c = self._compiler
        return (
            bare in c.objects
            or bare in c.spaces
            or bare in c.signatures
            or bare in c.categories
        )

    def _is_value_level_name(self, bare: str) -> bool:
        if not self._compiler:
            return False
        c = self._compiler
        return (
            bare in c.morphisms
            or bare in c.programs
            or bare in c.deductions
            or bare in c.encoders
            or bare in c.decoders
            or bare in c.losses
            or bare in c.contractions
            or bare in c.transformations
            or bare in c.rules
            or bare in c.bundles
        )

    def _type_signature_for_contraction(self, name: str, comp: Any) -> str:
        """GHCi-style signature for a registered contraction.

        Contractions denote operadic n-ary morphism builders
        ``(A_1, …, A_k) -> B``. The wiring's input objects and
        codomain are inspected through their declared AST node.
        """
        decl = getattr(comp, "decl", None) or comp
        inputs = getattr(decl, "input_domain", None) or getattr(decl, "domain", None)
        cod = getattr(decl, "input_codomain", None) or getattr(decl, "codomain", None)
        if inputs is None or cod is None:
            return f"{name} :: ?"
        if hasattr(inputs, "components"):
            input_strs = [
                (self._pretty_obj_expr(c) or "?")
                for c in getattr(inputs, "components", ())
            ]
            input_render = ", ".join(input_strs) if input_strs else "?"
        else:
            input_render = self._pretty_obj_expr(inputs) or "?"
        cod_render = self._pretty_obj_expr(cod) or "?"
        return f"{name} :: ({input_render}) -> {cod_render}"

    def _type_signature_for_rule(self, name: str, rule: Any) -> str:
        """GHCi-style signature for a deduction rule schema.

        A rule denotes a hyperedge ``premises |- conclusion`` in
        the deduction multicategory. Premises and conclusion are
        rendered as pattern tuples to mirror the schema's surface
        form.
        """
        del self
        premises = getattr(rule, "premises", ()) or ()
        conclusion = getattr(rule, "conclusion", None)
        prem_str = ", ".join(_pat_str(p) for p in premises)
        return f"{name} :: {prem_str} |- {_pat_str(conclusion)}"

    def _parses_as_type(self, expr_source: str) -> bool:
        try:
            mod = parse(f"object __probe__ : {expr_source}", file_path="<probe>")
        except ParseError:
            return False
        if not mod.statements:
            return False
        stmt = mod.statements[0]
        return isinstance(stmt, ObjectDecl) and isinstance(stmt.init, TypeFromExpr)

    def _describe(self, expr_source: str) -> ReplResponse:
        """Fallback inspector used by bare-expression evaluation
        and ``:watch``: try ``:type`` first, then ``:kind``.

        Lets the user type a bare name without caring whether it
        denotes an expression or a type.
        """
        response = self.type_of(expr_source)
        if response.ok:
            return response
        # If :type rejected the name as a type, retry as :kind.
        msg = response.diagnostics[0].message if response.diagnostics else ""
        if "use :kind" in msg or "is a type" in msg:
            kind_response = self.kind_of(expr_source)
            if kind_response.ok:
                return kind_response
        return response

    # ----- GHCi-style :type signature renderers ------------------------
    #
    # These produce ``name :: type`` lines (no decl-keyword prefix, no
    # ``[role=...]`` / option annotations). They drive ``:type``.
    # The ``_type_line_for_*`` family below produces decl-shaped lines
    # and drives ``:info`` / ``:browse`` instead.

    def _type_signature_for_morphism(self, name: str, morph: Any) -> str:
        dom = getattr(morph, "domain", getattr(morph, "dom", None))
        cod = getattr(morph, "codomain", getattr(morph, "cod", None))
        if dom is None or cod is None:
            return f"{name} :: ?"
        return f"{name} :: {self._pretty(dom)} -> {self._pretty(cod)}"

    def _type_signature_for_program(self, name: str, tmpl: Any) -> str:
        """GHCi-style signature for a program template.

        Per ``docs/semantics/programs.md``, a program's denotation
        depends on the parameter list:

        - Bare-identifier params (``P(q₁, …, qₖ) : τ₁ -> τ₂``) project
          components of the domain. The kernel's signature is
          ``τ₁ -> τ₂``; the q_i are syntactic conveniences, not
          additional arguments, so they do not appear in the type.
        - Typed params (``alpha : Real``, ``X : Object``,
          ``f : Mor[A, B]``) denote a dependent family
          ``∏ p_i:P_i. Kern(dom, cod)`` and are surfaced as a
          Haskell-style constraint context ``(p₁ : P₁, …) => dom -> cod``
          so the ∏-bound parameters never get conflated with the
          kernel's dom -> cod arrow.

        The dom / cod ``ObjectExpr`` AST nodes are routed through
        the compiler's ``_resolve_any_space`` so the rendered
        signature carries the *elaborated* SetObject / ContinuousSpace
        (with cardinalities and dims resolved) rather than the raw
        parse tree.
        """
        type_param_strs: list[str] = []
        for p in getattr(tmpl, "type_params", None) or ():
            kind = type(p).__name__
            if kind == "ScalarParam":
                type_param_strs.append(str(getattr(p, "scalar_kind", "?")))
            elif kind == "ObjectParam":
                type_param_strs.append(str(getattr(p, "universe", "?")))
            elif kind == "MorphismParam":
                dom_p = self._pretty_obj_expr(getattr(p, "domain", None))
                cod_p = self._pretty_obj_expr(getattr(p, "codomain", None))
                if dom_p is not None and cod_p is not None:
                    type_param_strs.append(f"Mor[{dom_p}, {cod_p}]")
                else:
                    type_param_strs.append("Mor")
            else:
                type_param_strs.append("?")
        dom = self._pretty_obj_expr(getattr(tmpl, "domain", None))
        cod = self._pretty_obj_expr(getattr(tmpl, "codomain", None))
        if dom is None or cod is None:
            return f"{name} :: ?"
        morph_sig = f"{dom} -> {cod}"
        if type_param_strs:
            ctx = ", ".join(type_param_strs)
            return f"{name} :: ({ctx}) => {morph_sig}"
        return f"{name} :: {morph_sig}"

    def _pretty_obj_expr(self, expr: Any) -> str | None:
        """Render an ``ObjectExpr`` for display, preferring the
        user-given alias for bare ``TypeName`` references.

        For bare names pointing at a module-level binding, returns
        the alias the user declared (``Word`` rather than the
        compiler's internal ``FinSet(name='_FinSet_200',
        cardinality=200)``). For compound type expressions
        (``FinSet 3``, ``A * B``) defers to the compiler's
        ``_resolve_any_space`` so the elaborated structure surfaces
        through ``_pretty_object``.
        """
        if expr is None:
            return None
        if not isinstance(expr, ObjectExpr):
            return _pretty_object(expr)
        if not self._compiler:
            return _pretty_object(expr)
        if isinstance(expr, TypeName):
            if (
                expr.name in self._compiler.objects
                or expr.name in self._compiler.spaces
            ):
                return expr.name
        try:
            return _pretty_object(self._compiler._resolve_any_space(expr))
        except CompileError:
            return _pretty_object(expr)
        except Exception:
            return _pretty_object(expr)

    def _type_signature_for_ref(self, ref: ScopedRef) -> str:
        """Render a GHCi-style ``name :: type`` line for a scoped ref.

        Strips the decl-style keyword prefix and any annotation
        suffixes, keeping only the underlying type information.
        """
        node = ref.node
        kind = ref.kind
        if kind == "program":
            return self._type_signature_for_program(ref.name, node)
        if kind == "morphism":
            return self._type_signature_for_morphism(ref.name, node)
        if kind == "deduction":
            dom = getattr(node, "domain", None)
            cod = getattr(node, "codomain", None)
            if dom is not None and cod is not None:
                return f"{ref.name} :: {_pretty_object(dom)} -> {_pretty_object(cod)}"
            return f"{ref.name} :: ?"
        if kind in ("sample-site", "observe-site", "marginalize-site"):
            return _site_signature(ref.name, node)
        if kind == "let-site":
            return f"{ref.name} :: ?"
        if kind == "score-site":
            return f"{ref.name} :: Real"
        if kind == "return-site":
            members = node if isinstance(node, tuple) else (str(node),)
            return f"return :: {', '.join(str(m) for m in members)}"
        if kind == "param":
            sk = getattr(node, "scalar_kind", None)
            universe = getattr(node, "universe", None)
            if sk is not None:
                return f"{ref.name} :: {sk}"
            if universe is not None:
                return f"{ref.name} :: {universe}"
            return f"{ref.name} :: ?"
        if kind == "deduction-rule":
            premises = getattr(node, "premises", ()) or ()
            conclusion = getattr(node, "conclusion", None)
            prem_str = ", ".join(_pat_str(p) for p in premises)
            return f"{ref.name} :: {prem_str} |- {_pat_str(conclusion)}"
        # Other value-level kinds (signature / encoder / decoder /
        # loss / bundle / rule / contraction / category / *-rule /
        # var-init / decoder-head / bundle-member / composition):
        # surface whatever the decl-line renderer produces with the
        # leading kind keyword stripped.
        line = self._type_line_for_ref(ref)
        stripped = _drop_leading_keyword(line, ref.name)
        return stripped if stripped is not None else line

    def _type_line_for_morphism(self, name: str, morph: Any) -> str:
        """Render a morphism's signature in valid-QVR notation.

        The unified ``morphism NAME : DOM -> COD [role=parameter]`` form
        is the only signature line that round-trips cleanly through
        the QVR grammar without a body block, so the TUI's tokenizer
        sees the dom and cod identifiers as type positions. The
        binding's true role is surfaced by ``:info``.
        """
        del self
        return f"morphism {name} : {_pretty_morphism(morph)} [role=parameter]"

    def _type_line_for_program(self, name: str, tmpl: Any) -> str:
        """Render a program template's signature.

        Reconstructs ``program NAME(params) : DOM -> COD`` from the
        AST node. Parameters come from two slots:
        ``params`` (bare-name list, no type annotation) and
        ``type_params`` (typed list of ``ScalarParam`` / ``ObjectParam``
        / ``MorphismParam`` records); a program declares at most one
        of the two, but the AST keeps them separate so the renderer
        merges whichever is populated.
        """
        del self
        param_strs: list[str] = []
        bare = getattr(tmpl, "params", None) or ()
        for n in bare:
            param_strs.append(str(n))
        typed = getattr(tmpl, "type_params", None) or ()
        for p in typed:
            pname = getattr(p, "name", "?")
            kind = type(p).__name__
            if kind == "ScalarParam":
                annot = getattr(p, "scalar_kind", "?")
            elif kind == "ObjectParam":
                annot = getattr(p, "universe", "?")
            elif kind == "MorphismParam":
                dom = getattr(p, "domain", None)
                cod = getattr(p, "codomain", None)
                annot = (
                    f"Mor[{_pretty_object(dom)}, {_pretty_object(cod)}]"
                    if dom is not None and cod is not None
                    else "Mor"
                )
            else:
                annot = ""
            if annot:
                param_strs.append(f"{pname} : {annot}")
            else:
                param_strs.append(str(pname))
        head = f"program {name}"
        if param_strs:
            head += f"({', '.join(param_strs)})"
        dom = getattr(tmpl, "domain", None)
        cod = getattr(tmpl, "codomain", None)
        if dom is not None and cod is not None:
            head += f" : {_pretty_object(dom)} -> {_pretty_object(cod)}"
        return head

    def _type_line_for_ref(self, ref: ScopedRef) -> str:
        """Render a single-line type signature for any scoped binding.

        Dispatches on ``ref.kind`` so a sample-site shows its
        ``sample NAME : INDEX <- FAMILY(args) [opts]`` line, a
        deduction rule shows its ``RULE : LHS |- RHS``, a sort
        shows ``sort NAME : kind [dim=...]``, and so on.
        """
        node = ref.node
        kind = ref.kind
        if kind == "program":
            return self._type_line_for_program(ref.name, node)
        if kind == "morphism":
            return self._type_line_for_morphism(ref.name, node)
        if kind == "object":
            return f"object {ref.name} : {_pretty_object(node)}"
        if kind == "space":
            return f"space {ref.name} : {_pretty_object(node)}"
        if kind == "deduction":
            dom = getattr(node, "domain", None)
            cod = getattr(node, "codomain", None)
            if dom is not None and cod is not None:
                return (
                    f"deduction {ref.name} : "
                    f"{_pretty_object(dom)} -> {_pretty_object(cod)}"
                )
            return f"deduction {ref.name}"
        if kind == "signature":
            return f"signature {ref.name}"
        if kind == "encoder":
            sig = getattr(node, "signature", "?")
            return f"encoder {ref.name} : {sig}"
        if kind == "decoder":
            sig = getattr(node, "signature", "?")
            return f"decoder {ref.name} : {sig}"
        if kind == "loss":
            return f"loss {ref.name}"
        if kind == "bundle":
            members = (
                getattr(node, "rules", ()) or node if isinstance(node, tuple) else ()
            )
            return f"bundle {ref.name} = {' | '.join(members)}"
        if kind == "rule":
            return f"rule {ref.name}"
        if kind == "contraction":
            return f"contraction {ref.name}"
        if kind == "param":
            sk = getattr(node, "scalar_kind", None)
            universe = getattr(node, "universe", None)
            if sk is not None:
                return f"param {ref.name} : {sk}"
            if universe is not None:
                return f"param {ref.name} : {universe}"
            return f"param {ref.name}"
        if kind == "sample-site":
            return _render_sample_line(node)
        if kind == "observe-site":
            return _render_observe_line(node)
        if kind == "marginalize-site":
            return _render_marginalize_line(node)
        if kind == "let-site":
            return f"let {ref.name} = …"
        if kind == "score-site":
            return f"score {ref.name} = …"
        if kind == "return-site":
            members = node if isinstance(node, tuple) else (str(node),)
            return f"return {', '.join(members)}"
        if kind == "deduction-rule":
            premises = getattr(node, "premises", ()) or ()
            conclusion = getattr(node, "conclusion", None)
            prem_str = ", ".join(_pat_str(p) for p in premises)
            return f"rule {ref.name} : {prem_str} |- {_pat_str(conclusion)}"
        if kind == "atom":
            return f"atom {ref.name}"
        if kind == "lexicon-entry":
            return f'lexicon "{ref.name}"'
        if kind == "sort":
            sk = getattr(node, "kind", "?")
            dim = getattr(node, "dim", None)
            tail = f" [dim={dim}]" if dim else ""
            return f"sort {ref.name} : {sk}{tail}"
        if kind == "constructor":
            args = getattr(node, "args", ()) or ()
            ret = getattr(node, "return_sort", None) or getattr(node, "result", "?")
            return (
                f"constructor {ref.name} : {', '.join(str(a) for a in args)} -> {ret}"
            )
        if kind == "binder":
            return f"binder {ref.name}"
        if kind in ("vertex-kind", "edge-kind"):
            return f"{kind.replace('-', ' ')} {ref.name}"
        if kind in (
            "op-rule",
            "init-rule",
            "message-rule",
            "update-rule",
            "var-init",
            "decoder-head",
        ):
            return f"{kind.replace('-', ' ')} {ref.name}"
        if kind == "composition":
            level = getattr(node, "level", None) or getattr(node, "rule", "?")
            return f"composition {ref.name} as {level}"
        if kind == "composition-entry":
            return f"composition entry {ref.name}"
        if kind == "bundle-member":
            return f"bundle member {ref.name}"
        if kind == "category":
            return f"category {ref.name}"
        return f"{kind} {ref.name}"

    def kind_of(self, expr_source: str) -> ReplResponse:
        """Print the kind of a type-level name or type expression.

        Mirrors GHCi's ``:kind``: works on objects, spaces, sorts,
        atoms, constructors, and bare type expressions like
        ``FinSet 3`` or ``A * B``. For value-level names (morphisms,
        programs, sites), returns an error directing the user to
        ``:type``.
        """
        if not self._compiler:
            return _err("no environment loaded; use :load <FILE> first")

        if SCOPE_SEPARATOR in expr_source:
            ref = resolve_scoped_path(self._compiler, expr_source.strip())
            if ref is None:
                return _err(f"unknown path: {expr_source}")
            if _ref_kind_class(ref.kind) == "value":
                return _err(
                    f"{expr_source} is an expression, not a type; use :type {expr_source}"
                )
            return _resp(self._type_line_for_ref(ref), body_kind="qvr")

        bare = expr_source.strip()
        qiec = qiec_binding_map(self._module).get(bare)
        if qiec is not None:
            if qiec.kind in {"index", "family", "effect"}:
                return _resp(
                    render_qiec_signature(self._module, bare)
                    or f"{bare} :: {qiec.kind}",
                    body_kind="qvr",
                )
            return _err(f"{bare} is an expression, not a type; use :type {bare}")
        qiec_candidates = qiec_binding_candidates(self._module, bare)
        if len(qiec_candidates) > 1:
            choices = ", ".join(item.qualified_name for item in qiec_candidates)
            return _err(f"ambiguous name {bare!r}; use one of: {choices}")
        prelude = prelude_effect(bare)
        if prelude is not None:
            return _resp(render_effect(prelude).splitlines()[0], body_kind="qvr")
        if bare.isidentifier():
            line = self._type_line_for_name(bare)
            if line is not None:
                return _resp(line, body_kind="qvr")
            if self._is_value_level_name(bare):
                return _err(f"{bare} is an expression, not a type; use :type {bare}")

        # Type-expression form: resolve through the compiler's space
        # resolver and pretty-print as a QVR object declaration.
        try:
            mod = parse(f"object __probe__ : {expr_source}", file_path="<kind>")
        except ParseError as e:
            return _err(f"parse error: {e}")
        if (
            not mod.statements
            or not isinstance(mod.statements[0], ObjectDecl)
            or not isinstance(mod.statements[0].init, TypeFromExpr)
        ):
            return _err("expected a type expression")
        texpr: ObjectExpr = mod.statements[0].init.expr
        probe = self._scratch_compiler()
        try:
            obj = probe._resolve_any_space(texpr)
        except CompileError as e:
            return _err(f"kind error: {e}")
        except Exception as e:
            return _err(f"kind error: {e}")
        return _resp(
            f"{expr_source} :: {_pretty_object(obj)}",
            body_kind="qvr",
        )

    # ----- :info / :doc -------------------------------------------------

    def info(self, name: str, *, python: bool = False) -> ReplResponse:
        # Scope-path form: ``lda::theta``, ``CCG::fwd_app``,
        # ``LF::sorts::Term``, etc. Resolves via the scope walker
        # rather than the module-level decl finder.
        if SCOPE_SEPARATOR in name:
            if self._compiler is None:
                return _err("no environment loaded; use :load <FILE> first")
            ref = resolve_scoped_path(self._compiler, name)
            if ref is None:
                return _err(f"unknown path: {name}")
            return self._info_for_scoped_ref(ref, python=python)

        ambiguity = self._qiec_ambiguity(name)
        if ambiguity is not None:
            return _err(ambiguity)
        decl = self._find_decl(name)
        if decl is None:
            if name in self._env:
                value = self._env[name]
                return _resp(
                    f"{name} :: {_pretty_runtime_value(value)}\n"
                    f"  (bound, no source declaration)"
                )
            return _err(f"unknown name: {name}")
        line = getattr(decl, "line", 0)
        col = getattr(decl, "col", 0)
        source = str(self._loaded_path) if self._loaded_path is not None else "<repl>"
        loc = f"{source}:{line}:{col}" if line else f"{source} (no position recorded)"
        if python:
            rendered = repr(decl)
            body_kind: Literal["text", "qvr", "json", "markdown"] = "text"
        else:
            rendered = self._render_decl_qvr(decl)
            body_kind = "qvr"
        docs = getattr(decl, "docs", ())
        doc_block = "\n".join(f"-- {d}" for d in docs)
        body = f"{rendered}\n-- declared at {loc}"
        if doc_block:
            body = f"{doc_block}\n{body}"
        return _resp(body, body_kind=body_kind)

    def _info_for_scoped_ref(
        self, ref: ScopedRef, *, python: bool = False
    ) -> ReplResponse:
        """Render an ``:info`` body for a scoped binding.

        For sub-scoped nodes (a sample / observe / marginalize
        step, a deduction rule, a sort, ...) the rendered body is
        the step's type line plus a source-location footer pointing
        at the enclosing declaration. For top-level scoped refs
        (``lda``, ``CCG``, ``Doc``) the body falls through to the
        same renderer ``:info`` uses for module-level decls.
        """
        node = ref.node
        if ref.parent_kind is None:
            # Top-level scoped ref: delegate to the existing
            # declaration-level renderer so verbatim source slicing
            # + doc-comment harvesting work the same way.
            return self.info(ref.name, python=python)
        line = getattr(node, "line", 0)
        col = getattr(node, "col", 0)
        if python:
            rendered = repr(node)
            body_kind: Literal["text", "qvr", "json", "markdown"] = "text"
        else:
            rendered = self._type_line_for_ref(ref)
            body_kind = "qvr"
        loc = (
            f"{self._loaded_path}:{line}:{col}"
            if self._loaded_path is not None and line
            else type(node).__name__
        )
        scope_chain = ref.path.rsplit(SCOPE_SEPARATOR, 1)[0]
        footer = f"-- {ref.kind} inside {scope_chain} at {loc}"
        docs = getattr(node, "docs", ())
        if docs:
            doc_block = "\n".join(f"-- {d}" for d in docs)
            body = f"{doc_block}\n{rendered}\n{footer}"
        else:
            body = f"{rendered}\n{footer}"
        return _resp(body, body_kind=body_kind)

    def _render_decl_qvr(self, decl: object) -> str:
        """Return the declaration as QVR source.

        Order of preference:
        1. Verbatim slice of the original source between this decl's
           start byte and the next decl's start byte (preserves comments
           and formatting).
        2. ``quivers.dsl.emit.module_to_source`` (canonical re-emission).
        3. ``repr(decl)`` as a last-ditch fallback for variants neither
           the slicer nor the emitter handles.
        """
        sliced = self._slice_source_for(decl)
        if sliced is not None:
            return sliced.rstrip() + "\n"
        return _render_decl(decl)

    def _slice_source_for(self, decl: object) -> str | None:
        """Return the original source lines that produced ``decl``."""
        if not self._loaded_source:
            return None
        start_line = getattr(decl, "line", 0)
        if not start_line:
            return None
        lines = self._loaded_source.splitlines()
        if start_line - 1 >= len(lines):
            return None
        # Find the next declaration's start line; everything between
        # `start_line` and that line belongs to this declaration.
        end_line = len(lines) + 1
        candidates: list[object] = list(self._module.statements)
        binding = next(
            (item for item in qiec_bindings(self._module) if item.declaration == decl),
            None,
        )
        if binding is not None and binding.declaration is not binding.owner:
            candidates.extend(
                item.declaration
                for item in qiec_bindings(self._module)
                if item.owner is binding.owner and item.declaration is not item.owner
            )
        for other in candidates:
            if other == decl:
                continue
            other_line = getattr(other, "line", 0)
            if other_line > start_line and other_line < end_line:
                end_line = other_line
        # Walk forward past trailing blank lines so the slice is tight.
        while end_line - 1 > start_line and not lines[end_line - 2].strip():
            end_line -= 1
        return "\n".join(lines[start_line - 1 : end_line - 1])

    def doc(self, name: str) -> ReplResponse:
        if SCOPE_SEPARATOR in name:
            if self._compiler is None:
                return _err("no environment loaded; use :load <FILE> first")
            ref = resolve_scoped_path(self._compiler, name)
            if ref is None:
                return _err(f"unknown path: {name}")
            docs = getattr(ref.node, "docs", ())
            if not docs:
                return _resp(f"{name}: (no doc comment)")
            return _resp("\n".join(docs), body_kind="markdown")
        ambiguity = self._qiec_ambiguity(name)
        if ambiguity is not None:
            return _err(ambiguity)
        decl = self._find_decl(name)
        if decl is None:
            return _err(f"unknown name: {name}")
        docs = getattr(decl, "docs", ())
        if not docs:
            return _resp(f"{name}: (no doc comment)")
        return _resp("\n".join(docs), body_kind="markdown")

    # ----- :browse ------------------------------------------------------

    # ----- :plate / :graph / :where / :effects / :shape -----------------

    def plate(self, name: str, *, fmt: str = "table") -> ReplResponse:
        """Render the plate diagram for ``name`` (a program).

        ``fmt`` selects the output format: ``"table"`` (default,
        in-TUI Rich table; also has a plain-text fallback),
        ``"mermaid"``, ``"dot"``, ``"tikz"``, ``"daft"``, or
        ``"open"`` (render via daft or graphviz, save to a temp
        PNG, and open with the system default opener).
        """
        if self._compiler is None:
            return _err("no environment loaded; use :load <FILE> first")
        graph = build_plate_graph(self._compiler, name)
        if graph is None:
            return _err(f"no program named {name!r} in the loaded module")
        if fmt == "mermaid":
            return _resp(render_mermaid(graph), body_kind="markdown")
        if fmt == "dot":
            return _resp(render_dot(graph))
        if fmt == "tikz":
            return _resp(render_tikz(graph))
        if fmt == "daft":
            return _resp(render_daft(graph))
        if fmt == "open":
            return self._plate_open(graph)
        # Default: plain-text table (TUI also accepts this and
        # wraps it through the Rich highlighter for colour).
        return _resp(render_table_plain(graph))

    def _plate_open(self, graph):  # type: ignore[no-untyped-def]
        """Render the plate graph to a temp PNG and open it.

        Strategy:
        1. If ``daft`` is importable, build the figure in-process
           and save to a temp file.
        2. Otherwise, if the ``dot`` binary is on PATH, pipe the
           DOT source through it to produce a PNG.
        3. Failing both, return Mermaid source with a hint to
           paste it into mermaid.live.

        After saving, opens the file with the system default app
        (``open`` / ``xdg-open`` / ``start``).
        """
        # Path 1: in-process daft.
        try:
            daft = importlib.import_module("daft")
        except ImportError:
            daft = None
        if daft is not None:
            script = render_daft(graph)
            # daft scripts produce a ``build_pgm()`` factory; exec
            # in a fresh namespace and call it.
            ns: dict[str, object] = {}
            try:
                exec(script, ns)  # noqa: S102
                pgm = ns["build_pgm"]()  # type: ignore[operator]
                tmp_path = self._temp_png()
                pgm.render()
                pgm.savefig(str(tmp_path))
                self._open_file(tmp_path)
                return _resp(
                    f"opened plate diagram at {tmp_path}",
                )
            except Exception as exc:
                # Fall through to the next backend on any failure.
                _ = exc

        # Path 2: dot binary.
        if _has_command("dot"):
            try:
                tmp_path = self._temp_png()
                _run_dot(render_dot(graph), tmp_path)
                self._open_file(tmp_path)
                return _resp(f"opened plate diagram at {tmp_path}")
            except Exception as exc:
                _ = exc

        # Path 3: Mermaid source with a hint.
        return _resp(
            "no plate-diagram renderer found (install ``daft`` or "
            "``graphviz``). Falling back to Mermaid source:\n\n"
            + render_mermaid(graph),
            body_kind="markdown",
        )

    def _temp_png(self) -> Path:
        return Path(tempfile.gettempdir()) / f"qvr_plate_{os.getpid()}.png"

    def _open_file(self, path: Path) -> None:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False)
        elif sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]

    def graph(self, name: str, *, fmt: str = "table") -> ReplResponse:
        """Render a step-flow view of ``name`` (a program).

        The step-flow view lists the program's body steps one per
        row with their dependency parents on the side, plus an
        indented sub-block for any marginalize body. Same ``fmt``
        flags as ``:plate``.
        """
        if self._compiler is None:
            return _err("no environment loaded; use :load <FILE> first")
        graph = build_plate_graph(self._compiler, name)
        if graph is None:
            return _err(f"no program named {name!r} in the loaded module")
        # The graph and plate diagrams share a model; ``:graph``
        # uses a step-oriented rendering vs ``:plate``'s
        # variable-oriented one.
        if fmt == "mermaid":
            return _resp(render_mermaid(graph), body_kind="markdown")
        if fmt == "dot":
            return _resp(render_dot(graph))
        if fmt == "open":
            return self._plate_open(graph)
        # The graph was built from this declaration, so it exists.
        lines = _scoped_step_lines(self._compiler.programs[name].draws, name)
        return _resp(_render_step_flow(graph, lines))

    def where(self, name: str) -> ReplResponse:
        """List every scope path whose final segment is ``name``."""
        if self._compiler is None:
            return _err("no environment loaded; use :load <FILE> first")
        refs = find_all_references(self._compiler, name)
        if not refs:
            return _err(f"no references to {name!r}")
        lines = [f"references to {name!r}:"]
        for ref in refs:
            lines.append(f"  {ref.kind:14} {ref.path}")
        return _resp("\n".join(lines))

    def effects(self, name: str) -> ReplResponse:
        """Compare declared and inferred effect sets for a program."""
        ambiguity = self._qiec_ambiguity(name)
        if ambiguity is not None:
            return _err(ambiguity)
        qiec = qiec_binding_map(self._module).get(name)
        if qiec is not None:
            declaration = qiec.declaration
            if isinstance(declaration, QiecEffectDecl):
                operations = ", ".join(op.name for op in declaration.operations)
                return _resp(
                    f"effect {declaration.name}\n  operations: {{{operations}}}"
                )
            if isinstance(declaration, QiecComputationDecl):
                row = declaration.effects
                entries = ", ".join(entry.instance for entry in row.entries)
                tail = row.tail or "(closed)"
                lacks = ", ".join(row.lacks) or "(none)"
                lines = [
                    f"computation {declaration.name}:",
                    f"  instances : {{{entries}}}",
                    f"  tail      : {tail}",
                    f"  lacks     : {{{lacks}}}",
                ]
                inferred = self._inferred_row(declaration.name)
                if inferred is not None:
                    lines.append(f"  inferred  : {inferred}")
                return _resp("\n".join(lines))
            if isinstance(declaration, QiecHandlerDecl):
                grades = ", ".join(
                    f"{clause.operation}:{clause.grade}"
                    for clause in declaration.clauses
                )
                return _resp(
                    f"handler {declaration.name} for {declaration.effect.name}:\n"
                    f"  coverage  : {declaration.coverage}\n"
                    f"  forwards  : {declaration.forwards_unknown}\n"
                    f"  resumptions: {{{grades}}}"
                )
            return _err(f"{name!r} does not declare an effect row")
        prelude = prelude_effect(name)
        if prelude is not None:
            operations = ", ".join(op.name for op in prelude.operations)
            return _resp(f"effect {name} (prelude)\n  operations: {{{operations}}}")

        if self._compiler is None:
            return _err("no environment loaded; use :load <FILE> first")
        graph = build_plate_graph(self._compiler, name)
        if graph is None:
            return _err(f"no program named {name!r} in the loaded module")
        declared = _declared_effects_for_program(self._compiler, name)
        inferred = _inferred_effects_from_graph(graph)
        lines = [
            f"program {name}:",
            f"  declared : {{{', '.join(sorted(declared)) or '(none)'}}}",
            f"  inferred : {{{', '.join(sorted(inferred))}}}",
        ]
        checked = self._inferred_row(name)
        if checked is not None:
            lines.append(f"  checked  : {checked}")
        leak = inferred - (declared or inferred)
        missing = declared - inferred if declared else set()
        if declared and leak:
            lines.append(
                f"  ! leak    : {{{', '.join(sorted(leak))}}}"
                " (body uses effects not declared)"
            )
        if declared and missing:
            lines.append(
                f"  ! unused  : {{{', '.join(sorted(missing))}}}"
                " (declared but not used)"
            )
        if not declared:
            lines.append("  (no [effects=[...]] declared)")
        return _resp("\n".join(lines))

    def shape(self, name: str) -> ReplResponse:
        """Render the ``ChainShape`` of the named program."""
        if self._compiler is None:
            return _err("no environment loaded; use :load <FILE> first")
        try:
            shape_obj = ChainShape.from_module(self._module)
        except Exception as exc:
            return _err(f"chain shape: {exc}")
        # ChainShape walks the module's first program decl
        # automatically; ``name`` is used only to disambiguate
        # output. Stay graceful when the module has multiple
        # programs and only one matches.
        steps = list(shape_obj.steps)
        if not steps:
            return _err(f"no program steps in module for {name!r}")
        lines = [
            f"chain shape ({shape_obj.algebra_name}):",
            "  #  depth  kind         name        size",
            "  -  -----  -----------  ----------  ----",
        ]
        for i, step in enumerate(steps, start=1):
            size = (
                str(step.intermediate_size)
                if step.intermediate_size is not None
                else "?"
            )
            lines.append(
                f"  {i:<3d} {step.depth:<5d}  {step.kind:<11s}  "
                f"{step.name:<10s}  {size:<4s}"
            )
        return _resp("\n".join(lines))

    def _browse_scope(self, ref: ScopedRef) -> ReplResponse:
        """Render ``:browse PATH`` for a scoped binding.

        Lists the binding's own type line then enumerates its
        children one per line. Children are addressable themselves
        via ``PATH::CHILD`` so the user can navigate deeper.
        """
        children = scope_children(ref)
        head = self._type_line_for_ref(ref)
        if not children:
            return _resp(f"{head}\n(no inner scope)")
        lines = [head]
        for cname, cref in children.items():
            lines.append(f"  {self._type_line_for_ref(cref)}")
        return _resp("\n".join(lines))

    def browse(self, namespace: str = "") -> ReplResponse:
        # Scoped browse: ``:browse lda`` shows the program's
        # children; ``:browse lda::z`` shows the marginalize's
        # inner scope. Falls through to the module-level view
        # below when ``namespace`` is empty or doesn't resolve.
        if namespace and self._compiler is not None:
            target = namespace.strip()
            if SCOPE_SEPARATOR in target or target.isidentifier():
                ref = resolve_scoped_path(self._compiler, target)
                if ref is not None:
                    return self._browse_scope(ref)

        groups: dict[str, list[tuple[str, object]]] = {
            "objects": [],
            "spaces": [],
            "morphisms": [],
            "rules": [],
            "programs": [],
            "deductions": [],
            "signatures": [],
            "encoders": [],
            "decoders": [],
            "losses": [],
            "bundles": [],
            "contractions": [],
            "indices": [],
            "families": [],
            "constructors": [],
            "effects": [],
            "operations": [],
            "instances": [],
            "handlers": [],
            "computations": [],
        }
        compiler = self._compiler
        if compiler is None:
            return _err("no environment loaded; use :load <FILE> first")
        sources: tuple[tuple[str, dict, object], ...] = (
            ("objects", compiler.objects, _children_for_object),
            ("spaces", compiler.spaces, _children_for_space),
            ("morphisms", compiler.morphisms, _children_for_morphism),
            ("rules", compiler.rules, _children_for_rule),
            ("programs", compiler.programs, _children_for_program),
            ("deductions", compiler.deductions, _children_for_deduction),
            ("signatures", compiler.signatures, _children_for_signature),
            ("encoders", compiler.encoders, _children_for_encoder),
            ("decoders", compiler.decoders, _children_for_decoder),
            ("losses", compiler.losses, _children_for_loss),
            ("bundles", compiler.bundles, _children_for_bundle),
            ("contractions", compiler.contractions, _children_for_contraction),
        )
        programs = compiler.programs
        for ns, mapping, builder in sources:
            entries: list[tuple[str, object]] = []
            for name in sorted(mapping):
                if ns == "morphisms" and name in programs:
                    # A program without typed parameters also compiles
                    # to a morphism; it is listed once, as the program.
                    continue
                head, children = builder(name, mapping[name])
                entries.append((head, children))
            groups[ns] = entries
        qiec_groups = {
            "index": "indices",
            "index-constructor": "constructors",
            "family": "families",
            "constructor": "constructors",
            "effect": "effects",
            "operation": "operations",
            "instance": "instances",
            "handler": "handlers",
            "computation": "computations",
        }
        for binding in qiec_bindings(self._module):
            namespace_name = qiec_groups[binding.kind]
            signature = render_qiec_signature(self._module, binding.qualified_name)
            groups[namespace_name].append((signature or binding.qualified_name, []))
        if namespace:
            ns = namespace if namespace in groups else namespace.rstrip("s") + "s"
            if ns not in groups:
                return _err(f"unknown namespace: {namespace}")
            groups = {ns: groups[ns]}
        lines: list[str] = []

        def _walk(label: str, children: object, depth: int) -> None:
            indent = "  " * depth
            lines.append(f"{indent}{label}")
            if not children:
                return
            assert isinstance(children, list)
            for child in children:
                if isinstance(child, str):
                    lines.append(f"{indent}  {child}")
                else:
                    sub_label, sub_children = child
                    _walk(sub_label, sub_children, depth + 1)

        for ns, entries in groups.items():
            if not entries:
                continue
            lines.append(f"{ns}:")
            for head, children in entries:
                _walk(head, children, 1)
        if not lines:
            return _resp("(empty environment)")
        return _resp("\n".join(lines))

    # ----- :dump --------------------------------------------------------

    def dump(self, name: str, *, as_json: bool = False) -> ReplResponse:
        ambiguity = self._qiec_ambiguity(name)
        if ambiguity is not None:
            return _err(ambiguity)
        decl = self._find_decl(name)
        if decl is None:
            return _err(f"unknown name: {name}")
        if as_json:
            if not isinstance(decl, dx.Model):
                return _err(f"{name!r} has no serializable AST node")
            return _resp(decl.model_dump_json(indent=2), body_kind="json")
        return _resp(repr(decl))

    # ----- :edit --------------------------------------------------------

    def edit(self, name: str, *, editor: str | None = None) -> ReplResponse:
        ambiguity = self._qiec_ambiguity(name)
        if ambiguity is not None:
            return _err(ambiguity)
        decl = self._find_decl(name)
        if decl is None:
            return _err(f"unknown name: {name}")
        text = _render_decl(decl)
        editor_cmd = editor or os.environ.get("EDITOR") or "vi"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".qvr", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(text + "\n")
            tmp_path = Path(tmp.name)
        try:
            subprocess.run([editor_cmd, str(tmp_path)], check=True)
            edited = tmp_path.read_text(encoding="utf-8")
        except subprocess.CalledProcessError as e:
            return _err(f"editor exited non-zero ({e.returncode})")
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        try:
            new_mod = parse(edited, file_path=f"<edit:{name}>")
        except ParseError as e:
            return _err(f"parse error in edited source: {e}")
        if not new_mod.statements:
            return _err("edited source contains no statements; aborting")
        replaced = _replace_decl_by_name(self._module, name, new_mod.statements)
        if replaced is None:
            return _err(f"could not locate {name} in current module")
        return self._install_module(replaced, source_path=self._loaded_path)

    # ----- :trace -------------------------------------------------------

    def trace(self, expr_source: str) -> ReplResponse:
        """Step through elaboration of a morphism expression."""
        if self._compiler is None:
            return _err("no environment loaded")
        try:
            mod = parse(f"export {expr_source}", file_path="<trace>")
        except ParseError as e:
            return _err(f"parse error: {e}")
        scratch = Compiler(_extend_module(self._module, mod.statements))
        # Re-run statement compilation so the env is fresh.
        try:
            scratch.compile()
        except (CompileError, QiecDiagnosticError) as e:
            return _err(f"compile error: {e}")
        # Walk the expression once and emit shape info as we go.
        lines: list[str] = []
        try:
            steps = _trace_expr(scratch, scratch._output_expr)
        except Exception as e:
            return _err(f"trace failed: {e}")
        for step in steps:
            lines.append(step)
        return _resp("\n".join(lines))

    # ----- :save / :watch / :unwatch -----------------------------------

    def save(self, path: str = "") -> ReplResponse:
        """Write the live module back to a ``.qvr`` file."""
        target = Path(path).expanduser() if path else self._loaded_path
        if target is None:
            return _err("usage: :save <FILE> (no file currently loaded)")
        try:
            source = module_to_source(self._module)
        except NotImplementedError as e:
            return _err(
                f"cannot emit: {e}. (the canonical emitter does not yet "
                "support every statement variant; edit the source file "
                "directly instead)"
            )
        try:
            target.write_text(source, encoding="utf-8")
        except OSError as e:
            return _err(f"write failed: {e}")
        return _resp(f"saved {target}")

    def watch(self, expr: str) -> ReplResponse:
        """Pin an expression for re-evaluation on every recompile."""
        expr = expr.strip()
        if not expr:
            return _err("usage: :watch <EXPR>")
        if expr not in self._watches:
            self._watches.append(expr)
        self._refresh_watches()
        rendered = self._watch_results.get(expr, "(unresolved)")
        return _resp(f"watch {expr} => {rendered}", body_kind="qvr")

    def unwatch(self, expr: str) -> ReplResponse:
        """Remove a previously-pinned watch expression."""
        expr = expr.strip()
        if not expr:
            # No argument: clear every watch.
            self._watches.clear()
            self._watch_results.clear()
            return _resp("cleared all watches")
        if expr not in self._watches:
            return _err(f"not watching: {expr}")
        self._watches.remove(expr)
        self._watch_results.pop(expr, None)
        return _resp(f"unwatched {expr}")

    def _refresh_watches(self) -> None:
        """Re-evaluate every pinned watch against the current env."""
        self._watch_results = {}
        for expr in self._watches:
            try:
                response = self._describe(expr)
                self._watch_results[expr] = response.body if response.ok else "(error)"
            except Exception:
                self._watch_results[expr] = "(error)"

    # ----- :set ---------------------------------------------------------

    def set_option(self, raw: str) -> ReplResponse:
        if "=" not in raw:
            return _err("usage: :set option=value")
        key, _, val = raw.partition("=")
        key = key.strip().replace("-", "_")
        if not hasattr(self.options, key):
            return _err(f"unknown option: {key}")
        current = getattr(self.options, key)
        v: bool | str
        if isinstance(current, bool):
            v = val.strip().lower() in ("1", "true", "yes", "on")
        else:
            v = val.strip()
        if key == "target" and v and v not in available_targets():
            return _err(
                f"unknown target {v!r}; available: {', '.join(available_targets())}"
            )
        self.options = self.options.with_(**{key: v})
        if key == "target" and self._loaded_path is not None:
            reloaded = self.reload()
            return _resp(f"{key} = {v}\n{reloaded.body}", reloaded.diagnostics)
        return _resp(f"{key} = {v}")

    # ----- :help --------------------------------------------------------

    def help(self, arg: str = "") -> ReplResponse:
        if arg:
            entry = _HELP.get(arg)
            if entry is None:
                return _err(f"no help for :{arg}")
            return _resp(entry)
        body = "\n".join(
            f":{name:<10} {summary}" for name, summary in _HELP_SUMMARIES.items()
        )
        return _resp(body)

    # ----- bare-line evaluation ----------------------------------------

    def _eval_source(self, src: str) -> ReplResponse:
        """Treat `src` first as statements appended to the current module."""
        try:
            new_mod = parse(src, file_path="<repl>")
            extended = _extend_module(self._module, new_mod.statements)
            return self._install_module(extended, source_path=self._loaded_path)
        except ParseError:
            pass
        # Fall back to expression: print its type or kind.
        return self._describe(src)

    # ----- helpers ------------------------------------------------------

    def _find_decl(self, name: str) -> object | None:
        qiec = qiec_binding_map(self._module).get(name)
        if qiec is not None:
            # Nested QIEC members are didactic models but not top-level
            # ``Statement`` instances. Interactive inspection treats both
            # uniformly; the narrow annotation is retained for compatibility
            # with current call sites.
            return qiec.declaration
        for stmt in self._module.statements:
            if getattr(stmt, "name", None) == name:
                return stmt
            if name in (getattr(stmt, "names", None) or ()):
                return stmt
        return None

    def _qiec_ambiguity(self, name: str) -> str | None:
        """Explain an ambiguous bare QIEC member name, if any."""

        if "." in name:
            return None
        candidates = qiec_binding_candidates(self._module, name)
        if len(candidates) <= 1:
            return None
        choices = ", ".join(item.qualified_name for item in candidates)
        return f"ambiguous name {name!r}; use one of: {choices}"

    def _inferred_row(self, name: str) -> str | None:
        """The row a computation's body performs, as the checker infers it.

        Parameters
        ----------
        name : str
            The computation's name.

        Returns
        -------
        str | None
            The inferred row with the declared instances' names, after
            every call in the body is expanded to its callee's signature;
            ``None`` when no checked module holds the computation.
        """
        module = self._qiec_module
        if module is None:
            return None
        if self._qiec_registry is None:
            self._qiec_registry = validate_module(module)
        try:
            inferred = inferred_computation_type(module, self._qiec_registry, name)
        except KeyError:
            return None
        return render_row(inferred.effects, self._qiec_registry.instance_names)

    def _scratch_compiler(self) -> Compiler:
        """A fresh Compiler with the current module already elaborated.

        Returns
        -------
        Compiler
            A compiler over the whole loaded module, its checked
            projection included, so a program bound to its checked
            computation resolves the computations it calls.
        """
        c = Compiler(
            self._module,
            module_name=qiec_module_name(self._loaded_path),
            file_path=str(self._loaded_path) if self._loaded_path else "<repl>",
        )
        try:
            c.compile_env()
        except CompileError, QiecDiagnosticError:
            pass
        return c

    def autoreload_if_stale(self) -> ReplResponse | None:
        """Re-run :reload if the loaded file's mtime advanced."""
        if not self.options.autoload_on_save:
            return None
        if self._loaded_path is None or self._loaded_mtime is None:
            return None
        try:
            current = self._loaded_path.stat().st_mtime
        except OSError:
            return None
        if current > self._loaded_mtime:
            return self.reload()
        return None


# ---------------------------------------------------------------------------
# meta-command dispatch table
# ---------------------------------------------------------------------------


def _cmd_load(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :load <FILE>")
    return s.load_file(arg)


def _cmd_reload(s: ReplSession, arg: str) -> ReplResponse:
    del arg
    return s.reload()


def _cmd_type(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :type <EXPR>")
    return s.type_of(arg)


def _cmd_kind(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :kind <TYPE-EXPR>")
    return s.kind_of(arg)


def _cmd_transpile(s: ReplSession, arg: str) -> ReplResponse:
    return s.transpile_module(arg)


def _cmd_info(s: ReplSession, arg: str) -> ReplResponse:
    parts = arg.split()
    if not parts:
        return _err("usage: :info <NAME> [--python]")
    name = parts[0]
    return s.info(name, python="--python" in parts[1:])


def _cmd_doc(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :doc <NAME>")
    return s.doc(arg)


def _cmd_plate(s: ReplSession, arg: str) -> ReplResponse:
    """``:plate PROGRAM [--mermaid|--dot|--tikz|--daft|--open]``.

    Render the plate diagram for ``PROGRAM``. Default is an
    in-TUI Rich table; flags emit alternate formats.
    """
    parts = arg.strip().split()
    if not parts:
        return _err("usage: :plate PROGRAM [--mermaid|--dot|--tikz|--daft|--open]")
    fmt = "table"
    program = ""
    for tok in parts:
        if tok in ("--mermaid", "--dot", "--tikz", "--daft", "--open"):
            fmt = tok[2:]
        elif tok.startswith("--"):
            return _err(f"unknown flag: {tok}")
        else:
            if program:
                return _err("usage: :plate PROGRAM [--FLAG]")
            program = tok
    if not program:
        return _err("usage: :plate PROGRAM [--FLAG]")
    return s.plate(program, fmt=fmt)


def _cmd_graph(s: ReplSession, arg: str) -> ReplResponse:
    """``:graph PROGRAM [--mermaid|--dot|--open]``."""
    parts = arg.strip().split()
    if not parts:
        return _err("usage: :graph PROGRAM [--mermaid|--dot|--open]")
    fmt = "table"
    program = ""
    for tok in parts:
        if tok in ("--mermaid", "--dot", "--open"):
            fmt = tok[2:]
        elif tok.startswith("--"):
            return _err(f"unknown flag: {tok}")
        else:
            if program:
                return _err("usage: :graph PROGRAM [--FLAG]")
            program = tok
    if not program:
        return _err("usage: :graph PROGRAM [--FLAG]")
    return s.graph(program, fmt=fmt)


def _cmd_where(s: ReplSession, arg: str) -> ReplResponse:
    name = arg.strip()
    if not name:
        return _err("usage: :where NAME")
    return s.where(name)


def _cmd_effects(s: ReplSession, arg: str) -> ReplResponse:
    name = arg.strip()
    if not name:
        return _err("usage: :effects PROGRAM")
    return s.effects(name)


def _cmd_shape(s: ReplSession, arg: str) -> ReplResponse:
    name = arg.strip()
    if not name:
        return _err("usage: :shape PROGRAM")
    return s.shape(name)


def _declared_effects_for_program(compiler, name: str) -> set[str]:  # type: ignore[no-untyped-def]
    """Read the ``[effects=[...]]`` option block of a program."""
    decl = compiler.programs.get(name)
    if decl is None:
        return set()
    out: set[str] = set()
    for opt in getattr(decl, "options", ()) or ():
        if getattr(opt, "key", None) != "effects":
            continue
        value = getattr(opt, "value", None)
        items = getattr(value, "items", None)
        if items is None:
            inner = getattr(value, "value", None)
            if isinstance(inner, str):
                out.add(inner)
            continue
        for item in items:
            v = getattr(item, "value", None)
            if isinstance(v, str):
                out.add(v)
    return out


def _inferred_effects_from_graph(graph) -> set[str]:  # type: ignore[no-untyped-def]
    inferred: set[str] = set()
    for n in graph.nodes:
        if n.kind == "latent":
            inferred.add("Sample")
        elif n.kind == "observed":
            inferred.add("Score")
        elif n.kind == "marginalized":
            inferred.add("Marginal")
    if not inferred:
        inferred.add("Pure")
    return inferred


def _scoped_step_lines(steps: Iterable[ProgramStep], prefix: str) -> dict[str, str]:
    """Each draw of a program body as written, keyed by its scope path.

    Parameters
    ----------
    steps : Iterable[ProgramStep]
        The body, searched through marginalization scopes.
    prefix : str
        The scope path of the body, the program's name at the top.

    Returns
    -------
    dict[str, str]
        ``<prefix>::<name>`` mapped to the step's source line, for
        every sample, observe, and marginalize site, and to
        ``let <name> = …`` for every let.
    """
    lines: dict[str, str] = {}
    for step in steps:
        if isinstance(step, LetStep):
            lines[f"{prefix}::{step.name}"] = f"let {step.name} = …"
        elif isinstance(step, SampleStep):
            for var in step.vars:
                lines[f"{prefix}::{var}"] = _render_sample_line(step)
        elif isinstance(step, ObserveStep):
            for var in step.vars:
                lines[f"{prefix}::{var}"] = _render_observe_line(step)
        elif isinstance(step, MarginalizeStep):
            lines[f"{prefix}::{step.var}"] = _render_marginalize_line(step)
            lines.update(_scoped_step_lines(step.scope, f"{prefix}::{step.var}"))
    return lines


def _render_step_flow(graph, lines: dict[str, str]) -> str:  # type: ignore[no-untyped-def]
    """Render a vertical step-flow table for ``:graph``.

    One row per step with columns: # / kind / step expression /
    parents. Differs from ``:plate``'s variable-oriented view in
    that the step expression carries the family + args + options,
    so the user can see the full source-level form for each step.
    ``lines`` holds each draw's source line by scope path, so a
    literal or keyword argument reads as written.
    """
    parents_by_dst: dict[str, list[str]] = {}
    for e in graph.edges:
        parents_by_dst.setdefault(e.dst, []).append(e.src)

    cols = ("#", "kind", "step", "parents")
    rows: list[tuple[str, ...]] = []
    for i, node in enumerate(graph.nodes, start=1):
        rows.append(
            (
                str(i),
                node.kind,
                lines[node.scope_path],
                ", ".join(parents_by_dst.get(node.name, ())) or "-",
            )
        )
    widths = [len(c) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out_lines = [
        f"program {graph.program_name}: {graph.domain} -> {graph.codomain}",
        "",
        _fmt(cols),
        "  ".join("-" * w for w in widths),
    ]
    out_lines.extend(_fmt(row) for row in rows)
    return "\n".join(out_lines)


def _has_command(name: str) -> bool:

    return shutil.which(name) is not None


def _run_dot(dot_source: str, out_png: Path) -> None:
    """Pipe DOT source through ``dot -Tpng`` to ``out_png``.

    Raises ``RuntimeError`` if ``dot`` exits non-zero.
    """
    proc = subprocess.run(
        ["dot", "-Tpng", "-o", str(out_png)],
        input=dot_source.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))


def _cmd_browse(s: ReplSession, arg: str) -> ReplResponse:
    return s.browse(arg)


def _cmd_dump(s: ReplSession, arg: str) -> ReplResponse:
    parts = arg.split()
    if not parts:
        return _err("usage: :dump <NAME> [--json]")
    name = parts[0]
    return s.dump(name, as_json=("--json" in parts[1:]))


def _cmd_edit(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :edit <NAME>")
    return s.edit(arg)


def _cmd_trace(s: ReplSession, arg: str) -> ReplResponse:
    if not arg:
        return _err("usage: :trace <EXPR>")
    return s.trace(arg)


def _cmd_save(s: ReplSession, arg: str) -> ReplResponse:
    return s.save(arg)


def _cmd_watch(s: ReplSession, arg: str) -> ReplResponse:
    return s.watch(arg)


def _cmd_unwatch(s: ReplSession, arg: str) -> ReplResponse:
    return s.unwatch(arg)


def _cmd_set(s: ReplSession, arg: str) -> ReplResponse:
    return s.set_option(arg)


def _cmd_runtime(s: ReplSession, arg: str) -> ReplResponse:
    return s.runtime(arg)


def _cmd_run(s: ReplSession, arg: str) -> ReplResponse:
    return s.run_computation(arg)


def _cmd_detach(s: ReplSession, arg: str) -> ReplResponse:
    if arg:
        return _err("usage: :detach")
    return s.detach_runtime()


def _cmd_help(s: ReplSession, arg: str) -> ReplResponse:
    return s.help(arg)


HELP_CATEGORIES: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "loading",
        (
            (":load FILE", "parse + elaborate; rebind the session env"),
            (":reload", "re-run :load on the last file; diff added / removed names"),
            (":save FILE", "write the live module to disk via module_to_source"),
        ),
    ),
    (
        "inspection",
        (
            (
                ":type EXPR",
                "print EXPR's type (morphisms, programs, sites; like GHCi :type)",
            ),
            (":kind T", "print T's kind (objects, spaces, sorts; like GHCi :kind)"),
            (":info NAME", "show NAME's declaration source + location"),
            (":doc NAME", "render the doc-comment above NAME's declaration"),
            (":browse [NS]", "list every binding in NS (or the module's scopes)"),
            (":dump NAME [--json]", "model_dump of NAME's AST node"),
        ),
    ),
    (
        "program exploration",
        (
            (":graph PROGRAM [--mermaid|--dot|--open]", "vertical step-flow diagram"),
            (":plate PROGRAM [--mermaid|--dot|--tikz|--daft|--open]", "plate notation"),
            (":where NAME", "list every scope path that mentions NAME"),
            (":effects PROGRAM", "declared vs inferred effect set"),
            (":shape PROGRAM", "per-step ChainShape (depth, intermediate sizes)"),
            (":trace EXPR", "step elaboration; surface each intermediate object/space"),
        ),
    ),
    (
        "live editing",
        (
            (":edit NAME", "open $EDITOR on NAME's decl; splice back on save"),
            (":watch EXPR", "pin EXPR; re-evaluate after every recompile"),
            (":unwatch [EXPR]", "remove EXPR (or all) from the watch list"),
            (":set KEY=VALUE", "toggle session options"),
        ),
    ),
    (
        "Execution",
        (
            (
                ":runtime [PROVIDER|FILE.json]",
                "show or attach explicit runtime providers",
            ),
            (
                ":run [NAME [JSON ...] [--data NAME=JSON] [--site NAME=JSON] "
                "[--static NAME=TERM] [--fuel STEPS] [--seed N]]",
                "run an entry point (a define or a program), or list them",
            ),
            (":detach", "detach every runtime provider"),
        ),
    ),
    (
        "code generation",
        (
            (
                ":transpile TARGET",
                "emit the loaded module as source for TARGET "
                "(stan, numpyro, pyro, pymc, edward2, church, webppl, "
                "turing, gen, bugs, jags)",
            ),
        ),
    ),
    (
        "control",
        (
            (":help", "show this dialog (Esc closes)"),
            (":quit", "exit the REPL"),
        ),
    ),
)


KEY_BINDINGS: tuple[tuple[str, str], ...] = (
    ("Ctrl-G / Ctrl-O / F8", "evaluate the current buffer"),
    ("Ctrl-Up / Ctrl-Down", "previous / next input from history"),
    ("Tab", "complete the identifier under the cursor"),
    ("Ctrl-P", "command palette"),
    ("Ctrl-L", "clear the output log"),
    ("Ctrl-R", "reload the last file"),
    ("Ctrl-Q", "quit"),
    ("F1", "open the help dialog"),
    ("Esc", "dismiss a modal overlay"),
)


def _cmd_quit(s: ReplSession, arg: str) -> ReplResponse:
    del s, arg
    return _resp("__quit__")


_META_COMMANDS = {
    "load": _cmd_load,
    "l": _cmd_load,
    "reload": _cmd_reload,
    "r": _cmd_reload,
    "type": _cmd_type,
    "t": _cmd_type,
    "kind": _cmd_kind,
    "k": _cmd_kind,
    "transpile": _cmd_transpile,
    "info": _cmd_info,
    "i": _cmd_info,
    "doc": _cmd_doc,
    "browse": _cmd_browse,
    "b": _cmd_browse,
    "plate": _cmd_plate,
    "p": _cmd_plate,
    "graph": _cmd_graph,
    "g": _cmd_graph,
    "where": _cmd_where,
    "effects": _cmd_effects,
    "shape": _cmd_shape,
    "dump": _cmd_dump,
    "edit": _cmd_edit,
    "trace": _cmd_trace,
    "save": _cmd_save,
    "s": _cmd_save,
    "watch": _cmd_watch,
    "w": _cmd_watch,
    "unwatch": _cmd_unwatch,
    "set": _cmd_set,
    "runtime": _cmd_runtime,
    "run": _cmd_run,
    "detach": _cmd_detach,
    "help": _cmd_help,
    "h": _cmd_help,
    "quit": _cmd_quit,
    "q": _cmd_quit,
    "exit": _cmd_quit,
}


_HELP_SUMMARIES = {
    "load FILE": "parse + compile a .qvr file into the session",
    "reload": "re-run :load on the last file, diffing the env",
    "type EXPR": "print the type of a value-level expression (GHCi-style)",
    "kind TYPE": "print the kind of a type (GHCi-style)",
    "transpile TARGET": "transpile the loaded module to TARGET (stan, numpyro, ...)",
    "info NAME": "show the declaration and source location of NAME (--python for AST repr)",
    "doc NAME": "render the doc comment attached to NAME",
    "browse [NS]": "list bound names, optionally per namespace",
    "dump NAME": "show the AST node for NAME (add --json for didactic dump)",
    "edit NAME": "open $EDITOR on NAME's source, splice back on save",
    "trace EXPR": "step through elaboration of a morphism expression",
    "save FILE": "write the live module back to FILE via the canonical emitter",
    "watch EXPR": "pin EXPR for re-eval after every recompile (watch panel)",
    "unwatch EXPR": "remove EXPR from the watch list (no arg clears all)",
    "set k=v": "toggle session options (highlight, unicode, theme, ...)",
    "help [CMD]": "list commands or detail one",
    "quit": "exit the REPL",
}


_HELP: dict[str, str] = {
    "load": "Parse and compile <FILE>; the environment is rebound to the new module.",
    "reload": "Re-parse the most recently loaded file and show which names changed.",
    "type": "Resolve <EXPR> as a value-level expression and print its type "
    "signature. Works on morphisms, programs, deductions, scoped sample "
    "/ observe / let sites, or any expression that resolves to a morphism. "
    "If <EXPR> names a type-level binding (object, space, sort), the "
    "command reports an error directing you to :kind.",
    "kind": "Resolve <TYPE> as a type-level expression and print its kind. "
    "Works on objects, spaces, sorts, atoms, constructors, and bare type "
    "expressions like ``FinSet 3`` or ``A * B``. If <TYPE> names a "
    "value-level binding (morphism, program), the command reports an "
    "error directing you to :type.",
    "transpile": "Emit the currently-loaded module as source for the named "
    "TARGET backend. TARGET must be one of the names returned by "
    "``quivers.transpile.available_targets()`` (stan, numpyro, pyro, pymc, "
    "edward2, church, webppl, turing, gen, bugs, jags). The output is the "
    "transpiled source bytes decoded as UTF-8.",
    "info": "Show NAME's declaration as verbatim .qvr source (sliced from the "
    "loaded file), plus the source location and any leading doc comment. "
    "Pass --python to see the didactic AST `repr()` instead.",
    "doc": "Render only the doc comment for NAME, useful for piping.",
    "browse": "List every bound name grouped by namespace. Pass a namespace name "
    "(objects/spaces/morphisms/rules) to restrict the listing.",
    "dump": "Pretty-print the AST node for NAME. Pass --json for didactic's "
    "model_dump_json output.",
    "edit": "Open $EDITOR on NAME's declaration; on save the edited text is "
    "spliced back into the module and recompiled.",
    "trace": "Step through morphism elaboration, surfacing each intermediate "
    "domain/codomain.",
    "set": "Toggle session options: highlight=true|false, unicode=true|false, "
    "show_axes=true|false, paranoid=true|false, autoload_on_save=true|false, "
    "target=TARGET (report that transpile target's capability diagnostics "
    "on :load and :reload; empty for none).",
    "help": "Without an argument, list every command. With one, print its help.",
    "quit": "Leave the REPL.",
    "runtime": "Show the active runtime, attach a registered provider by name, "
    "or load a non-executable JSON provider configuration.",
    "run": "Execute an entry point: a define with JSON value arguments, or a "
    "program with its data (positional or --data NAME=JSON) and any "
    "conditioned sites (--site NAME=JSON), every other site drawn. "
    "Polymorphic definitions require one --static NAME=TERM assignment per "
    "static binder; :run alone lists the entry points.",
    "detach": "Remove every runtime provider. Attach one again with :runtime.",
}


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def _resp(
    body: str,
    diagnostics: Iterable[Diagnostic] = (),
    *,
    body_kind: Literal["text", "qvr", "json", "markdown"] = "text",
) -> ReplResponse:
    return ReplResponse(body=body, diagnostics=tuple(diagnostics), body_kind=body_kind)


def _err(message: str, *, code: str = "repl") -> ReplResponse:
    return ReplResponse(
        diagnostics=(Diagnostic(message=message, severity="error", code=code),)
    )


def _execution_error(error: ExecutionFailure) -> ReplResponse:
    diagnostic = error.diagnostic
    origin = diagnostic.origin
    return ReplResponse(
        diagnostics=(
            Diagnostic(
                message=diagnostic.message,
                severity="error",
                code=diagnostic.code,
                line=origin.line
                if origin is not None and origin.line is not None
                else 0,
                col=(
                    origin.column
                    if origin is not None and origin.column is not None
                    else 0
                ),
            ),
        )
    )


def _fuel(text: str) -> int:
    """Read a ``--fuel`` step budget.

    Parameters
    ----------
    text : str
        The budget as typed.

    Returns
    -------
    int
        The budget.

    Raises
    ------
    ValueError
        If the text is not a positive integer.
    """
    try:
        steps = int(text)
    except ValueError as error:
        raise ValueError(f"--fuel must be a positive integer, got {text!r}") from error
    if steps <= 0:
        raise ValueError("--fuel must be a positive number of steps")
    return steps


def _env_counts(env: dict[str, Any]) -> str:
    items = sum(1 for k in env if not k.startswith("__"))
    return f"{items} binding(s)"


def _violation_to_diag(v: Violation) -> Diagnostic:
    return Diagnostic(
        message=v.message,
        severity="error",
        line=v.line,
        col=v.col,
        code=v.code,
    )


# ScopedRef.kind values that denote value-level bindings (have a
# DOM -> COD signature or evaluate to a morphism); used by :type.
_VALUE_REF_KINDS: frozenset[str] = frozenset(
    {
        "morphism",
        "program",
        "deduction",
        "sample-site",
        "observe-site",
        "marginalize-site",
        "let-site",
        "score-site",
        "return-site",
        "encoder",
        "decoder",
        "loss",
        "signature",
        "bundle",
        "rule",
        "contraction",
        "deduction-rule",
        "lexicon-entry",
        "composition",
        "composition-entry",
        "bundle-member",
        "category",
        "op-rule",
        "init-rule",
        "message-rule",
        "update-rule",
        "var-init",
        "decoder-head",
        "param",
    }
)

# ScopedRef.kind values that denote type-level bindings (describe a
# universe / shape / sort); used by :kind.
_TYPE_REF_KINDS: frozenset[str] = frozenset(
    {
        "object",
        "space",
        "sort",
        "atom",
        "constructor",
        "binder",
        "vertex-kind",
        "edge-kind",
    }
)


def _ref_kind_class(kind: str) -> Literal["value", "type", "other"]:
    """Classify a ``ScopedRef.kind`` for the :type / :kind split.

    Returns ``"value"`` for expression-shaped refs (rendered by
    :type), ``"type"`` for type-shaped refs (rendered by :kind),
    and ``"other"`` for refs that belong to neither (top-level
    namespaces, unclassified kinds). Callers default ``"other"`` to
    the permissive branch so unknown kinds still render somewhere.
    """
    if kind in _VALUE_REF_KINDS:
        return "value"
    if kind in _TYPE_REF_KINDS:
        return "type"
    return "other"


_AUTO_NAME_RE = re.compile(r"^_([A-Z][A-Za-z]*)_([0-9]+(?:_[0-9]+)*)$")


def _strip_auto_name(name: str) -> str | None:
    """If ``name`` is a compiler-generated placeholder like
    ``_FinSet_20`` or ``_Real_8`` or ``_Real_3_4``, return the
    surface form (``FinSet 20``, ``Real 8``, ``Real 3 4``).
    Otherwise return None so callers fall back to the raw name.
    """
    m = _AUTO_NAME_RE.match(name)
    if m is None:
        return None
    ctor, args = m.group(1), m.group(2)
    return f"{ctor} {args.replace('_', ' ')}"


def _pretty_object_with_aliases(obj: Any, alias_map: dict[int, str] | None) -> str:
    """Render an object preferring user-given aliases.

    When ``alias_map`` is provided, a resolved SetObject /
    ContinuousSpace whose ``id()`` is a key in the map renders as
    the user's declared name (``Item`` rather than ``FinSet 200``).
    Product / coproduct factors recurse so a mixed ``Doc * Topic``
    still surfaces each component's alias.
    """
    if obj is None:
        return "?"
    if alias_map is not None and id(obj) in alias_map:
        return alias_map[id(obj)]
    kind = type(obj).__name__
    if kind in ("ProductSet", "ProductSpace"):
        comps = getattr(obj, "components", ())
        if comps:
            return " * ".join(_pretty_object_with_aliases(c, alias_map) for c in comps)
    if kind == "CoproductSet":
        comps = getattr(obj, "components", ())
        if comps:
            return " + ".join(_pretty_object_with_aliases(c, alias_map) for c in comps)
    return _pretty_object(obj)


def _pretty_object(obj: Any) -> str:
    """Render a SetObject / ContinuousSpace in QVR-shaped notation.

    Examples:
        FinSet(name='X', cardinality=3)            -> "X"
        FinSet(name='', cardinality=3)              -> "FinSet 3"
        FinSet(name='_FinSet_20', cardinality=20)  -> "FinSet 20"
        ProductSet(components=(A, B))               -> "A * B"
        CoproductSet(components=(A, B))             -> "A + B"
        FreeMonoid(name='Words', alphabet=A)        -> "Words"
        EnumSet(name='Tags', members=('NP','S'))    -> "Tags"
        FreeResiduated(name='Cat', ...)             -> "Cat"
        Euclidean(name='_Real_8', dim=8)           -> "Real 8"
    """
    kind = type(obj).__name__
    name = getattr(obj, "name", "") or ""
    if kind in ("ProductSet", "ProductSpace"):
        comps = getattr(obj, "components", ())
        if comps:
            return " * ".join(_pretty_object(c) for c in comps)
    if kind == "CoproductSet":
        comps = getattr(obj, "components", ())
        if comps:
            return " + ".join(_pretty_object(c) for c in comps)
    if kind == "FinSet":
        if name:
            stripped = _strip_auto_name(name)
            if stripped is not None:
                return stripped
            return name
        cardinality = getattr(obj, "cardinality", None)
        return f"FinSet {cardinality}" if cardinality is not None else "FinSet"
    if name:
        stripped = _strip_auto_name(name)
        if stripped is not None:
            return stripped
        # The discrete-to-continuous embedding wraps a FinSet inside
        # an `Euclidean(name="idx(FinSet(name='Source', ...))", ...)`.
        # Strip the wrapper so users see the original object name.
        if (
            name.startswith("idx(FinSet(name='")
            and "'" in name[len("idx(FinSet(name='") :]
        ):
            inner = name[len("idx(FinSet(name='") :]
            return inner.split("'", 1)[0]
        return name
    # ContinuousSpace constructors keep their constructor + args
    # readable via repr, but we want to avoid raw didactic output.
    constructor = getattr(obj, "constructor", None)
    if constructor is not None:
        args = getattr(obj, "args", ()) or ()
        return f"{constructor}({', '.join(str(a) for a in args)})"
    return kind


def _pretty_runtime_value(value: Any) -> str:
    return _pretty_object(value)


def _pretty_morphism(m: Any) -> str:
    dom = getattr(m, "domain", getattr(m, "dom", None))
    cod = getattr(m, "codomain", getattr(m, "cod", None))
    if dom is None or cod is None:
        return f"{type(m).__name__}"
    return f"{_pretty_object(dom)} -> {_pretty_object(cod)}"


def _render_decl(decl: object) -> str:
    """Emit a declaration as canonical source, falling back to repr.

    `quivers.dsl.emit.module_to_source` covers the common declarations
    and raises NotImplementedError otherwise; we catch and fall back so
    :info / :edit never crash on a rare variant.
    """
    if not isinstance(decl, Statement):
        return repr(decl)
    try:
        return module_to_source(Module(statements=(decl,))).rstrip("\n")
    except NotImplementedError:
        return repr(decl)


def _extend_module(base: Module, additions: Iterable[Statement]) -> Module:
    return Module(statements=tuple(base.statements) + tuple(additions))


def render_signature(compiler: Compiler | None, name: str) -> str | None:
    """Return the GHCi-style ``name :: type`` (or ``object NAME : ...``
    for type-level bindings) for a top-level name, or ``None`` when
    the compiler is absent or the name is unknown.

    This is the shared entry point used by both the REPL (``:type``
    / ``:kind`` / bare-expression fallback) and the LSP server's
    hover panel: by routing every surface that wants a one-line
    signature through the same function, a binding looks identical
    regardless of whether the user is in the TUI or hovering in an
    editor.

    Parameters
    ----------
    compiler : Compiler or None
        The compiler holding the elaborated environment.
    name : str
        The top-level name to describe.

    Returns
    -------
    str or None
        The one-line signature, or ``None`` when there is no compiler
        or the name is unbound.
    """
    if compiler is None:
        return None
    s = ReplSession()
    s._compiler = compiler
    s._module = getattr(compiler, "_module", Module(statements=()))
    line = s._value_line_for_name(name)
    if line is not None:
        return line
    return s._type_line_for_name(name)


def render_qiec_signature(module: Module, name: str) -> str | None:
    """Return a compact source signature for one QIEC binding.

    This works from the authored AST rather than a backend compiler object, so
    it remains available when a document contains only indexed families or
    effect interfaces.

    Parameters
    ----------
    module : Module
        The parsed module holding the binding.
    name : str
        The QIEC binding to describe.

    Returns
    -------
    str or None
        The compact signature, or ``None`` when ``name`` is not a QIEC
        binding of ``module``.
    """

    binding = qiec_binding_map(module).get(name)
    if binding is None:
        return None
    if binding.kind == "index":
        return f"index {name} :: Index"

    rendered = _render_decl(binding.owner)
    lines = [
        line.strip()
        for line in rendered.splitlines()
        if line.strip() and not line.lstrip().startswith("#!")
    ]
    if binding.declaration is binding.owner:
        return lines[0] if lines else f"{name} :: {binding.kind}"

    pattern = re.compile(rf"\b{re.escape(binding.name)}\b")
    for line in lines[1:]:
        if pattern.search(line):
            if line.startswith("constructor "):
                line = line.removeprefix("constructor ")
            if name != binding.name:
                line = pattern.sub(name, line, count=1)
            return line
    return f"{name} :: {binding.kind}"


def _extract_export_expr(mod: Module):
    """Return the ``Expr`` AST node from a probe ``export <expr>``
    module, or ``None`` if the parse did not produce an
    `ExportDecl`. Used by :type to feed the user's expression
    directly into ``Compiler._compile_expr`` without re-running
    the full module compile pass.
    """
    for stmt in mod.statements:
        if isinstance(stmt, ExportDecl):
            return stmt.expr
    return None


# ---------------------------------------------------------------------------
# Scoped-step pretty printers (used by ``:type lda::theta``-style queries)
# ---------------------------------------------------------------------------


def _call_str(step: Any) -> str:
    head = getattr(step, "morphism", "?") or "?"
    return draw_call_to_source(str(head), getattr(step, "args", None))


def _index_suffix(idx: Any) -> str:
    if idx is None:
        return ""
    name = getattr(idx, "name", None) or repr(idx)
    return f" : {name}"


def _options_suffix(step: Any) -> str:
    """Render an option block ``[k=v, ...]`` for a step, or empty
    when the step has no options."""
    options = getattr(step, "options", ()) or ()
    if not options:
        return ""
    parts: list[str] = []
    for opt in options:
        key = getattr(opt, "key", "?")
        value = getattr(opt, "value", None)
        v = getattr(value, "value", None)
        parts.append(f"{key}={v if v is not None else value}")
    return f" [{', '.join(parts)}]"


def _site_value_space(step: Any) -> str | None:
    """Extract the ``over=`` value-space option from a sample /
    observe / marginalize step, returning its rendered name."""
    for opt in getattr(step, "options", ()) or ():
        if getattr(opt, "key", "") == "over":
            value = getattr(opt, "value", None)
            v = getattr(value, "value", None)
            return str(v) if v is not None else str(value)
    return None


def _site_signature(name: str, step: Any) -> str:
    """GHCi-style ``name :: type`` for a sample / observe / marginalize
    step. ``type`` is ``index -> value-space`` when both are known,
    or ``value-space`` when the step has no index.
    Falls back to the family call (``Dirichlet(alpha)``) when the
    value-space can't be read off the options.
    """
    idx_obj = getattr(step, "index", None)
    idx_name = (
        getattr(idx_obj, "name", None) or repr(idx_obj) if idx_obj is not None else None
    )
    value_space = _site_value_space(step)
    if value_space is None:
        value_space = _call_str(step)
    if idx_name is not None:
        return f"{name} :: {idx_name} -> {value_space}"
    return f"{name} :: {value_space}"


def _drop_leading_keyword(line: str, name: str) -> str | None:
    """If ``line`` starts with ``<keyword> <name> :`` (the decl-line
    shape used by :info / :browse), strip the keyword + name and
    return a ``<name> :: <rest>`` form. Otherwise return None.
    """
    head, sep, rest = line.partition(" : ")
    if not sep:
        return None
    head_tokens = head.split()
    if len(head_tokens) < 2 or head_tokens[-1] != name:
        return None
    return f"{name} :: {rest}"


def _render_sample_line(step: Any) -> str:
    vars_ = getattr(step, "vars", ()) or ()
    var = vars_[0] if vars_ else "?"
    idx = _index_suffix(getattr(step, "index", None))
    return f"sample {var}{idx} <- {_call_str(step)}{_options_suffix(step)}"


def _render_observe_line(step: Any) -> str:
    vars_ = getattr(step, "vars", ()) or ()
    var = ", ".join(vars_) if vars_ else "?"
    idx = _index_suffix(getattr(step, "index", None))
    return f"observe {var}{idx} <- {_call_str(step)}{_options_suffix(step)}"


def _render_marginalize_line(step: Any) -> str:
    var = getattr(step, "var", "?")
    idx = _index_suffix(getattr(step, "index", None))
    return f"marginalize {var}{idx} <- {_call_str(step)}{_options_suffix(step)}"


def _pat_str(pat: Any) -> str:
    if pat is None:
        return "?"
    if isinstance(pat, tuple):
        return "(" + ", ".join(_pat_str(p) for p in pat) + ")"
    return str(pat)


def _replace_decl_by_name(
    base: Module, name: str, replacements: Iterable[Statement]
) -> Module | None:
    out: list[Statement] = []
    replaced = False
    repls = list(replacements)
    for stmt in base.statements:
        bound = (getattr(stmt, "names", None) or ()) or (
            (getattr(stmt, "name", None),) if getattr(stmt, "name", None) else ()
        )
        if not replaced and name in bound:
            out.extend(repls)
            replaced = True
            continue
        out.append(stmt)
    if not replaced:
        return None
    return Module(statements=tuple(out))


def _trace_expr(compiler: Compiler, expr: Any) -> list[str]:
    """Yield human-readable lines describing each elaboration step."""
    out: list[str] = []
    seen: set[int] = set()

    def visit(e: Any, depth: int) -> None:
        if id(e) in seen:
            return
        seen.add(id(e))
        pad = "  " * depth
        if isinstance(e, ExprIdent):
            morph = compiler.morphisms.get(e.name)
            if morph is not None:
                out.append(f"{pad}{e.name} :: {_pretty_morphism(morph)}")
            else:
                out.append(f"{pad}{e.name} :: <unresolved>")
            return
        if isinstance(e, ExprCompose):
            out.append(f"{pad}compose {e.op}")
            visit(e.left, depth + 1)
            visit(e.right, depth + 1)
            try:
                morph = compiler._compile_expr(e)
                out.append(f"{pad}  => {_pretty_morphism(morph)}")
            except CompileError as ce:
                out.append(f"{pad}  => error: {ce}")
            return
        if isinstance(e, ExprTensorProduct):
            out.append(f"{pad}tensor")
            visit(e.left, depth + 1)
            visit(e.right, depth + 1)
            return
        try:
            morph = compiler._compile_expr(e)
            out.append(f"{pad}{type(e).__name__} :: {_pretty_morphism(morph)}")
        except Exception as ex:
            out.append(f"{pad}{type(e).__name__} :: <{ex}>")

    visit(expr, 0)
    return out


__all__ = [
    "Diagnostic",
    "ReplResponse",
    "ReplSession",
    "SessionOptions",
    "render_qiec_signature",
    "render_signature",
]
