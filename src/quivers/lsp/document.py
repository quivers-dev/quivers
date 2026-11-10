"""Per-document state for the QVR language server.

A `DocumentState` holds the latest source bytes, the parsed
`Module`, the elaborated environment, and the diagnostic list
for one open document. Re-analysis runs synchronously on every
``didOpen`` / ``didChange`` so subsequent capabilities (hover,
documentSymbol, completion) read from a coherent snapshot.
"""

from __future__ import annotations

import re
from typing import Protocol

from quivers.cli.repl_session import Diagnostic
from quivers.dsl import Compiler, ParseError, parse
from quivers.dsl.ast_nodes import DefineDecl, Module, Statement
from quivers.dsl.ast_nodes.qiec import (
    QiecBinder,
    QiecBindComputation,
    QiecCaseComputation,
    QiecIfComputation,
    QiecComputation,
    QiecComputationDecl,
    QiecEffectDecl,
    QiecEffectInstanceDecl,
    QiecFamilyDecl,
    QiecHandleComputation,
    QiecHandlerDecl,
    QiecHandlerOperationClause,
    QiecHandlerReturnClause,
    QiecLocalBinding,
    QiecSequenceComputation,
    QiecTypeName,
    QiecValueParameter,
)
from quivers.dsl.constraints import check_constraints
from quivers.qiec.checking import KernelRegistry
from quivers.qiec.module import validate_module
from quivers.transpile.qiec_ir import analyze_qiec_capabilities
from quivers.dsl.qiec_tooling import (
    QiecBinding,
    analyze_module,
    qiec_binding_candidates,
    qiec_binding_map,
    qiec_module_name,
)
from quivers.qiec import QiecModule


class EnvBinding(Protocol):
    """Any value the compiler binds in the environment.

    The language server never calls into a binding; it only reports
    the binding's class name in hover output, so the protocol is
    intentionally empty.
    """


type QiecLocalDecl = QiecBinder | QiecValueParameter | QiecLocalBinding | QiecTypeName
"""A source binder whose identity is lexical rather than module-level."""


class DocumentState:
    """Per-document mutable cache for the language server.

    Each LSP document open by an editor has one of these. ``update``
    re-parses and re-elaborates in place; the resulting snapshot is
    what subsequent hover / definition / diagnostic capabilities read.
    """

    uri: str
    target: str | None
    version: int
    source: str
    module: Module
    compiler: Compiler | None
    qiec_module: QiecModule | None
    env: dict[str, EnvBinding]
    diagnostics: list[Diagnostic]

    def __init__(self, uri: str, *, target: str | None = None) -> None:
        self.uri = uri
        self.target = target
        self.version = 0
        self.source = ""
        self.module = Module(statements=())
        self.compiler = None
        self.qiec_module = None
        self.env = {}
        self.diagnostics = []
        #: The diagnostics of the parse and the check, before any target's
        #: capability pass adds its own.
        self._base_diagnostics: list[Diagnostic] = []
        self._registry: KernelRegistry | None = None

    def update(self, *, source: str, version: int) -> None:
        """Re-parse + re-elaborate after a text change."""
        self.source = source
        self.version = version
        self.diagnostics = []
        self.module = Module(statements=())
        self.compiler = None
        self.qiec_module = None
        self._registry = None
        self.env = {}
        try:
            module = parse(source, file_path=self.uri)
        except ParseError as e:
            line, col = _extract_position(str(e))
            self.diagnostics.append(
                Diagnostic(
                    message=str(e),
                    severity="error",
                    code="parse",
                    line=line,
                    col=col,
                )
            )
            self._base_diagnostics = list(self.diagnostics)
            return
        self.module = module
        analysis = analyze_module(
            module,
            module_name=qiec_module_name(self.uri),
            file_path=self.uri,
        )
        for v in check_constraints(analysis.non_qiec_module):
            self.diagnostics.append(
                Diagnostic(
                    message=v.message,
                    severity="error",
                    line=v.line,
                    col=v.col,
                    code=v.code,
                )
            )
        for diagnostic in analysis.qiec_diagnostics:
            self.diagnostics.append(
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
            self.diagnostics.append(
                Diagnostic(
                    message=str(error),
                    severity="error",
                    line=getattr(error, "line", 0),
                    col=getattr(error, "col", 0),
                    code="compile",
                )
            )
        self.env = analysis.env
        self.compiler = analysis.compiler
        self.qiec_module = analysis.qiec_module
        self._base_diagnostics = list(self.diagnostics)
        self.retarget(self.target)

    def registry(self) -> KernelRegistry | None:
        """The registry the checked module validates against.

        Returns
        -------
        KernelRegistry | None
            The registry, resolved once per checked module and reused
            until the document changes; ``None`` when the document has
            no checked module.
        """
        if self.qiec_module is None:
            return None
        if self._registry is None:
            self._registry = validate_module(self.qiec_module)
        return self._registry

    def retarget(self, target: str | None) -> None:
        """Recompute the target-capability diagnostics against ``target``.

        The parsed module and its checked projection are kept: only the
        capability pass runs again, so a configuration change costs no
        reparse and cannot change the grammar the document was read
        with.

        Parameters
        ----------
        target : str | None
            The transpile target, or ``None`` for no capability
            diagnostics.
        """
        self.target = target
        self.diagnostics = list(self._base_diagnostics)
        if target is None or self.qiec_module is None:
            return
        capabilities = analyze_qiec_capabilities(self.qiec_module, target)
        for diagnostic in capabilities:
            origin = diagnostic.origin
            self.diagnostics.append(
                Diagnostic(
                    message=diagnostic.message,
                    severity="error",
                    line=(origin.line or 0) if origin is not None else 0,
                    col=(origin.column or 0) if origin is not None else 0,
                    code=diagnostic.kind,
                )
            )
        if not capabilities:
            from quivers.transpile import UnsupportedConstruct
            from quivers.transpile.plan import checked_program_plan

            try:
                checked_program_plan(self.qiec_module, self.module, target)
            except UnsupportedConstruct as error:
                self.diagnostics.extend(
                    Diagnostic(
                        message=str(error),
                        severity="error",
                        line=0,
                        col=0,
                        code=kind,
                    )
                    for kind in error.kinds
                )

    def find_decl(
        self,
        name: str,
        *,
        line: int | None = None,
        col: int | None = None,
    ) -> object | None:
        """Return the statement binding ``name``, or ``None``.

        A plural-name declaration (``morphism f, g : ...``) binds every
        name in its ``names`` tuple, so several names may resolve to
        the same statement. ``define`` where-blocks are searched after
        their enclosing statement.
        """
        local = self.find_qiec_local(name, line=line, col=col)
        if local is not None:
            return local
        binding = self.find_qiec_binding(name, line=line, col=col)
        if binding is not None:
            return binding.declaration
        return _find_non_qiec_decl_in(self.module.statements, name)

    def find_qiec_local(
        self,
        name: str,
        *,
        line: int | None,
        col: int | None = None,
    ) -> QiecLocalDecl | None:
        """Resolve the innermost QIEC binder visible at a source position."""

        if line is None:
            return None
        visible = self.visible_qiec_locals(line=line, col=col)
        return next(
            (
                local
                for local in reversed(visible)
                if getattr(local, "name", None) == name
            ),
            None,
        )

    def visible_qiec_locals(
        self,
        *,
        line: int,
        col: int | None = None,
    ) -> tuple[QiecLocalDecl, ...]:
        """Return lexical QIEC binders visible at an LSP source position.

        The tuple is ordered outermost-to-innermost. Shadowed spellings remain
        in the tuple so navigation can retain declaration identity; callers
        presenting completion should keep the last declaration for each name.
        """

        source_line = line + 1
        owner = _top_level_statement_at(self.module.statements, source_line)
        if owner is None:
            return ()
        source_lines = self.source.splitlines()
        if line >= len(source_lines):
            return ()
        source_col = col if col is not None else len(source_lines[line])
        visible: list[QiecLocalDecl] = []

        if isinstance(owner, QiecFamilyDecl):
            visible.extend(
                local
                for local in (*owner.parameters, *owner.indices)
                if _binder_has_entered_scope(local, source_line, source_col)
            )
            constructor = next(
                (
                    candidate
                    for candidate in owner.constructors
                    if candidate.line == source_line
                ),
                None,
            )
            if constructor is not None:
                visible.extend(
                    local
                    for local in constructor.binders
                    if _binder_has_entered_scope(local, source_line, source_col)
                )
        elif isinstance(owner, QiecEffectDecl):
            visible.extend(
                local
                for local in owner.binders
                if _binder_has_entered_scope(local, source_line, source_col)
            )
            operation = next(
                (
                    candidate
                    for candidate in owner.operations
                    if candidate.line == source_line
                ),
                None,
            )
            if operation is not None:
                visible.extend(
                    local
                    for local in operation.binders
                    if _binder_has_entered_scope(local, source_line, source_col)
                )
        elif isinstance(owner, QiecHandlerDecl):
            visible.extend(
                local
                for local in owner.binders
                if _binder_has_entered_scope(local, source_line, source_col)
            )
            # A clause's binders and parameters, and the locals of its
            # body, are visible from the clause's own line to the line
            # before the next clause.
            clauses = sorted(owner.clauses, key=lambda clause: clause.line)
            for index, clause in enumerate(clauses):
                following = (
                    clauses[index + 1].line if index + 1 < len(clauses) else None
                )
                if source_line < clause.line or (
                    following is not None and source_line >= following
                ):
                    continue
                if isinstance(clause, QiecHandlerOperationClause):
                    visible.extend(
                        local
                        for local in (*clause.binders, *clause.parameters)
                        if _binder_has_entered_scope(local, source_line, source_col)
                    )
                elif isinstance(clause, QiecHandlerReturnClause):
                    if _binder_has_entered_scope(
                        clause.binder, source_line, source_col
                    ):
                        visible.append(clause.binder)
                body = getattr(clause, "body", None)
                if body is not None:
                    visible.extend(
                        _visible_computation_locals(body, source_line, source_col)
                    )
        elif isinstance(owner, QiecComputationDecl):
            visible.extend(
                local
                for local in (*owner.binders, *owner.parameters)
                if _binder_has_entered_scope(local, source_line, source_col)
            )
            visible.extend(
                _visible_computation_locals(owner.body, source_line, source_col)
            )
        return tuple(visible)

    def find_qiec_binding(
        self,
        name: str,
        *,
        line: int | None = None,
        col: int | None = None,
    ) -> QiecBinding | None:
        """Resolve a QIEC identity, using source context when needed.

        Operation names are scoped by their effect interfaces. At a source
        position, a lexical ``instance.operation`` request, an operation
        declaration, or a handler clause supplies the missing owner. Without
        that context, ambiguous bare member names deliberately do not resolve.
        """

        candidates = qiec_binding_candidates(self.module, name)
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        if line is None:
            return qiec_binding_map(self.module).get(name)

        source_lines = self.source.splitlines()
        source_line = source_lines[line] if line < len(source_lines) else ""
        if col is not None:
            name_start = min(col, len(source_line))
            while name_start > 0 and _is_ident_char(source_line[name_start - 1]):
                name_start -= 1
            prefix = source_line[:name_start]
            qualifier = re.search(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*$",
                prefix,
            )
            if qualifier is not None:
                owner = qualifier.group(1)
                qualified = qiec_binding_map(self.module).get(f"{owner}.{name}")
                if qualified is not None:
                    return qualified
                instance = qiec_binding_map(self.module).get(owner)
                if instance is not None and isinstance(
                    instance.declaration, QiecEffectInstanceDecl
                ):
                    effect_name = instance.declaration.effect.name
                    qualified = qiec_binding_map(self.module).get(
                        f"{effect_name}.{name}"
                    )
                    if qualified is not None:
                        return qualified

        source_line_number = line + 1
        declared_here = [
            binding
            for binding in candidates
            if getattr(binding.declaration, "line", 0) == source_line_number
        ]
        if len(declared_here) == 1:
            return declared_here[0]

        owner = _top_level_statement_at(self.module.statements, source_line_number)
        if owner is not None:
            owner_name = getattr(owner, "name", None)
            if isinstance(owner, QiecHandlerDecl):
                owner_name = owner.effect.name
            if isinstance(owner_name, str):
                qualified = qiec_binding_map(self.module).get(f"{owner_name}.{name}")
                if qualified is not None:
                    return qualified
        return None

    def name_at_position(self, line: int, col: int) -> str | None:
        """Return the identifier covering ``(line, col)`` in the source."""
        lines = self.source.splitlines()
        if line >= len(lines):
            return None
        text = lines[line]
        if col > len(text):
            return None
        # Expand left and right to grab the identifier characters.
        i = col
        while i > 0 and _is_ident_char(text[i - 1]):
            i -= 1
        j = col
        while j < len(text) and _is_ident_char(text[j]):
            j += 1
        if j <= i:
            return None
        return text[i:j] or None

    def qualified_name_at_position(self, line: int, col: int) -> str | None:
        """The ``owner.member`` spelling covering a position, if any.

        Parameters
        ----------
        line : int
            The 0-based line.
        col : int
            The 0-based column.

        Returns
        -------
        str | None
            ``owner.member`` when the identifier at the position is the
            member of such a spelling, as the operation of a
            ``perform instance.operation(...)`` request is; ``None``
            when the identifier stands alone.
        """
        lines = self.source.splitlines()
        if line >= len(lines):
            return None
        text = lines[line]
        i = col
        while i > 0 and _is_ident_char(text[i - 1]):
            i -= 1
        j = col
        while j < len(text) and _is_ident_char(text[j]):
            j += 1
        if j <= i or i == 0 or text[i - 1] != ".":
            return None
        k = i - 1
        while k > 0 and _is_ident_char(text[k - 1]):
            k -= 1
        owner = text[k : i - 1]
        if not owner:
            return None
        return f"{owner}.{text[i:j]}"

    def scoped_path_at_position(self, line: int, col: int) -> str | None:
        """The ``::`` scope path covering a position, up to its segment.

        Parameters
        ----------
        line : int
            The 0-based line.
        col : int
            The 0-based column.

        Returns
        -------
        str | None
            ``prog::site`` when the identifier at the position is a
            segment of such a path, cut after that segment: the cursor
            on ``z`` in ``lda::z::w`` gives ``lda::z``. ``None`` when
            the identifier is not part of a ``::`` path.
        """
        lines = self.source.splitlines()
        if line >= len(lines):
            return None
        text = lines[line]
        if col > len(text):
            return None
        i = col
        while i > 0 and (_is_ident_char(text[i - 1]) or text[i - 1] == ":"):
            i -= 1
        j = col
        while j < len(text) and _is_ident_char(text[j]):
            j += 1
        spelled = text[i:j].lstrip(":")
        if "::" not in spelled:
            return None
        segments = spelled.split("::")
        if any(not segment or ":" in segment for segment in segments):
            return None
        return spelled


def decl_names(stmt: Statement) -> tuple[str, ...]:
    """Every name bound by ``stmt``.

    Plural-name declarations (``morphism f, g : ...``, ``object A, B :
    ...``, ``category NP, S``) carry a ``names`` tuple and bind one
    name per entry; single-name declarations carry ``name``;
    expression-only statements (``export``) bind nothing.
    """
    names = getattr(stmt, "names", None)
    if isinstance(names, tuple):
        return tuple(n for n in names if isinstance(n, str))
    single = getattr(stmt, "name", None)
    if isinstance(single, str):
        return (single,)
    return ()


def _find_non_qiec_decl_in(
    statements: tuple[Statement, ...], name: str
) -> object | None:
    for stmt in statements:
        if name in decl_names(stmt):
            return stmt
        if isinstance(stmt, DefineDecl):
            nested = _find_non_qiec_decl_in(stmt.where, name)
            if nested is not None:
                return nested
    return None


def _visible_computation_locals(
    computation: QiecComputation,
    source_line: int,
    source_col: int,
) -> tuple[QiecLocalDecl, ...]:
    """Return lexically visible local binders along one source-position path."""

    if isinstance(computation, QiecBindComputation):
        then_line = getattr(computation.then, "line", 0)
        if then_line and source_line >= then_line:
            return (
                computation.binder,
                *_visible_computation_locals(computation.then, source_line, source_col),
            )
        if _position_covers_name(computation.binder, source_line, source_col):
            return (computation.binder,)
        return _visible_computation_locals(computation.first, source_line, source_col)
    if isinstance(computation, QiecSequenceComputation):
        then_line = getattr(computation.then, "line", 0)
        child = (
            computation.then
            if then_line and source_line >= then_line
            else computation.first
        )
        return _visible_computation_locals(child, source_line, source_col)
    if isinstance(computation, QiecHandleComputation):
        return _visible_computation_locals(computation.body, source_line, source_col)
    if isinstance(computation, QiecIfComputation):
        otherwise_line = getattr(computation.otherwise, "line", 0)
        branch = (
            computation.otherwise
            if otherwise_line and source_line >= otherwise_line
            else computation.then
        )
        return _visible_computation_locals(branch, source_line, source_col)
    if isinstance(computation, QiecCaseComputation):
        # Motive indices scope only over the motive result on the case line;
        # branches receive fresh constructor-pattern statics instead.
        if source_line == computation.motive.line:
            return tuple(
                binder
                for binder in computation.motive.indices
                if _binder_has_entered_scope(binder, source_line, source_col)
            )
        eligible = [
            branch for branch in computation.branches if branch.line <= source_line
        ]
        if not eligible:
            return ()
        branch = max(eligible, key=lambda candidate: candidate.line)
        branch_locals: tuple[QiecLocalDecl, ...] = (
            *(
                binder
                for binder in branch.static_arguments
                if isinstance(binder, QiecTypeName)
                and _binder_has_entered_scope(binder, source_line, source_col)
            ),
            *(
                field
                for field in branch.fields
                if _binder_has_entered_scope(field, source_line, source_col)
            ),
        )
        return (
            *branch_locals,
            *_visible_computation_locals(branch.body, source_line, source_col),
        )
    return ()


def _binder_has_entered_scope(
    binder: QiecLocalDecl,
    source_line: int,
    source_col: int,
) -> bool:
    """Whether ``binder`` has been declared by the requested position."""

    binder_line = getattr(binder, "line", 0)
    binder_col = getattr(binder, "col", 0)
    return binder_line < source_line or (
        binder_line == source_line and binder_col <= source_col
    )


def _position_covers_name(
    binder: QiecLocalDecl,
    source_line: int,
    source_col: int,
) -> bool:
    """Whether the cursor covers a binder's own declaration token."""

    name = getattr(binder, "name", "")
    binder_line = getattr(binder, "line", 0)
    binder_col = getattr(binder, "col", 0)
    return (
        bool(name)
        and source_line == binder_line
        and binder_col <= source_col <= binder_col + len(name)
    )


def _top_level_statement_at(
    statements: tuple[Statement, ...], source_line: int
) -> Statement | None:
    """Return the top-level statement whose source span contains ``source_line``."""

    located = sorted(
        (statement for statement in statements if getattr(statement, "line", 0)),
        key=lambda statement: getattr(statement, "line", 0),
    )
    current: Statement | None = None
    for statement in located:
        if getattr(statement, "line", 0) > source_line:
            break
        current = statement
    return current


def _is_ident_char(c: str) -> bool:
    return c.isalnum() or c == "_"


_POSITION_PATTERN = re.compile(r"line\s+(\d+),\s*col\s+(\d+)")


def _extract_position(msg: str) -> tuple[int, int]:
    """``(line, col)`` extraction from a `ParseError` message.

    The parser's syntax-error messages carry ``line L, col C`` with a
    1-based line and a 0-based column; walker-invariant messages carry
    no position, in which case ``(0, 0)`` selects the whole-file span.
    """
    m = _POSITION_PATTERN.search(msg)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


__all__ = []
