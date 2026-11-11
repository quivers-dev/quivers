"""pygls-backed LSP server for QVR.

Capabilities (LSP 3.17):

- ``textDocument/didOpen``, ``didChange``, ``didSave``, ``didClose``
- ``workspace/didChangeConfiguration`` for live transpile-target diagnostics
- ``textDocument/publishDiagnostics``
- ``textDocument/semanticTokens/full``
- ``textDocument/hover``
- ``textDocument/definition``
- ``textDocument/references``
- ``textDocument/documentSymbol``
- ``textDocument/completion``
- ``textDocument/formatting``

The server holds per-URI `DocumentState` and re-analyses on
every change. All token classification, completion, and rendering
logic is shared with the REPL so the in-editor experience matches the
TUI exactly.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import cast

import didactic.api as dx
from pygls.lsp.server import LanguageServer
from lsprotocol import types as lsp

from quivers.analysis.scope import ScopedRef, resolve_scoped_path
from quivers.cli.repl_complete import all_completions
from quivers.cli.repl_highlight import (
    SEMANTIC_TOKEN_MODIFIERS,
    SEMANTIC_TOKEN_TYPES,
    to_semantic_token_data,
)
from quivers.cli.repl_session import (
    Diagnostic,
    ReplSession,
    render_qiec_signature,
    render_signature,
)
from quivers.dsl.ast_nodes import (
    ContinuousConstructor,
    DefineDecl,
    MorphismDecl,
    ObjectDecl,
    ProgramDecl,
    Statement,
    TypeFromExpr,
    TypeInitializer,
)
from quivers.dsl.ast_nodes.program_steps import CallStep
from quivers.dsl.ast_nodes.qiec import (
    QiecBinder,
    QiecComputationDecl,
    QiecEffectDecl,
    QiecEffectInstanceDecl,
    QiecFamilyDecl,
    QiecHandlerDecl,
    QiecHandlerReturnClause,
    QiecIndexDecl,
    QiecLocalBinding,
    QiecTypeName,
    QiecValueParameter,
)
from quivers.dsl.emit import _emit_qiec_binder, _emit_qiec_type, module_to_source
from quivers.dsl.qiec_tooling import (
    call_site_at,
    call_sites,
    instance_effect_name,
    instantiated_call_signature,
    prelude_effect,
    qiec_bindings,
    qiec_env_kinds,
    qiec_local_bindings,
    qualified_operation_signature,
)
from quivers.qiec.effects import render_row
from quivers.qiec.module import inferred_computation_type
from quivers.lsp.document import DocumentState, decl_names

SERVER_NAME = "qvr-lsp"
"""The server name reported to clients and attached to diagnostics."""

SERVER_VERSION = "0.3.0"
"""The server version reported to clients in the initialize response."""

_TARGET_UNCHANGED = object()


def _target_from_settings(settings: object) -> str | None | object:
    """Extract a target from a didChangeConfiguration payload."""

    if not isinstance(settings, dict):
        return _TARGET_UNCHANGED
    section = settings.get("qvr", settings)
    if not isinstance(section, dict):
        return _TARGET_UNCHANGED
    for key in ("transpileTarget", "transpile_target", "target"):
        if key not in section:
            continue
        value = section[key]
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() or None
        return _TARGET_UNCHANGED
    return _TARGET_UNCHANGED


def build_server(*, target: str | None = None) -> LanguageServer:
    """Build a QVR language server with every feature registered.

    Parameters
    ----------
    target : str or None
        A transpile target whose missing capabilities are published as
        ``qiec:capability:*`` diagnostics; ``None`` disables them. A
        client may change it later through
        ``workspace/didChangeConfiguration``.

    Returns
    -------
    LanguageServer
        The configured pygls server, ready for ``start_io`` or
        ``start_tcp``.
    """
    server = LanguageServer(name=SERVER_NAME, version=SERVER_VERSION)
    docs: dict[str, DocumentState] = {}
    selected_target = target

    # ----- lifecycle ----------------------------------------------------

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def _did_open(ls: LanguageServer, params: lsp.DidOpenTextDocumentParams) -> None:
        doc = DocumentState(uri=params.text_document.uri, target=selected_target)
        doc.update(
            source=params.text_document.text,
            version=params.text_document.version,
        )
        docs[doc.uri] = doc
        _publish(ls, doc)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def _did_change(
        ls: LanguageServer, params: lsp.DidChangeTextDocumentParams
    ) -> None:
        uri = params.text_document.uri
        doc = docs.get(uri)
        if doc is None:
            return
        # Apply incremental changes if present; full-replace otherwise.
        for change in params.content_changes:
            if isinstance(change, lsp.TextDocumentContentChangePartial):
                doc.source = _apply_partial(doc.source, change)
            else:
                doc.source = change.text
        doc.update(source=doc.source, version=params.text_document.version)
        _publish(ls, doc)

    @server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
    def _did_save(ls: LanguageServer, params: lsp.DidSaveTextDocumentParams) -> None:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return
        if params.text is not None:
            doc.update(source=params.text, version=doc.version)
        _publish(ls, doc)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
    def _did_close(_ls: LanguageServer, params: lsp.DidCloseTextDocumentParams) -> None:
        docs.pop(params.text_document.uri, None)

    @server.feature(lsp.WORKSPACE_DID_CHANGE_CONFIGURATION)
    def _did_change_configuration(
        ls: LanguageServer, params: lsp.DidChangeConfigurationParams
    ) -> None:
        """Refresh target-capability diagnostics for every open document.

        Clients may send either the VS Code-shaped
        ``{qvr: {transpileTarget: ...}}`` object or the section value
        ``{transpileTarget: ...}``. An empty string disables target-specific
        diagnostics. Settings unrelated to the target leave it unchanged.
        """

        nonlocal selected_target
        configured = _target_from_settings(params.settings)
        if configured is _TARGET_UNCHANGED:
            return
        selected_target = cast(str | None, configured)
        for doc in docs.values():
            doc.retarget(selected_target)
            _publish(ls, doc)

    # ----- semantic tokens ---------------------------------------------

    @server.feature(
        lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
        lsp.SemanticTokensLegend(
            token_types=list(SEMANTIC_TOKEN_TYPES),
            token_modifiers=list(SEMANTIC_TOKEN_MODIFIERS),
        ),
    )
    def _semantic_tokens(
        _ls: LanguageServer, params: lsp.SemanticTokensParams
    ) -> lsp.SemanticTokens:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return lsp.SemanticTokens(data=[])
        env_kinds = _env_kinds_for(doc)
        data = to_semantic_token_data(doc.source, env_kinds=env_kinds)
        return lsp.SemanticTokens(data=data)

    # ----- hover --------------------------------------------------------

    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def _hover(_ls: LanguageServer, params: lsp.HoverParams) -> lsp.Hover | None:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return None
        scoped = _scoped_ref_at(doc, params.position.line, params.position.character)
        if scoped is not None:
            return lsp.Hover(
                contents=lsp.MarkupContent(
                    kind=lsp.MarkupKind.Markdown,
                    value=_render_scoped_hover(doc, scoped),
                )
            )
        name = doc.name_at_position(params.position.line, params.position.character)
        if name is None:
            return None
        body = _render_hover(
            doc,
            name,
            line=params.position.line,
            col=params.position.character,
        )
        if body is None:
            return None
        return lsp.Hover(
            contents=lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value=body)
        )

    # ----- definition / references / documentSymbol --------------------

    @server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
    def _definition(
        _ls: LanguageServer, params: lsp.DefinitionParams
    ) -> list[lsp.Location] | None:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return None
        scoped = _scoped_ref_at(doc, params.position.line, params.position.character)
        if scoped is not None:
            target = _scoped_ref_position(doc, scoped)
            if target is None:
                return None
            line, col = target
            return [
                lsp.Location(
                    uri=doc.uri,
                    range=lsp.Range(
                        start=lsp.Position(line=line, character=col),
                        end=lsp.Position(line=line, character=col + len(scoped.name)),
                    ),
                )
            ]
        name = doc.name_at_position(params.position.line, params.position.character)
        if name is None:
            return None
        decl = doc.find_decl(
            name,
            line=params.position.line,
            col=params.position.character,
        )
        if decl is None:
            return None
        line, col = _name_position(doc, decl, name)
        return [
            lsp.Location(
                uri=doc.uri,
                range=lsp.Range(
                    start=lsp.Position(line=line, character=col),
                    end=lsp.Position(line=line, character=col + len(name)),
                ),
            )
        ]

    @server.feature(lsp.TEXT_DOCUMENT_REFERENCES)
    def _references(
        _ls: LanguageServer, params: lsp.ReferenceParams
    ) -> list[lsp.Location] | None:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return None
        name = doc.name_at_position(params.position.line, params.position.character)
        if name is None:
            return None
        return list(
            _find_references(
                doc,
                name,
                line=params.position.line,
                col=params.position.character,
            )
        )

    @server.feature(lsp.TEXT_DOCUMENT_PREPARE_RENAME)
    def _prepare_rename(
        _ls: LanguageServer, params: lsp.PrepareRenameParams
    ) -> lsp.PrepareRenameResult_Type1 | None:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return None
        name = doc.name_at_position(params.position.line, params.position.character)
        if name is None:
            return None
        if _renameable_declaration(doc, name, params.position) is None:
            return None
        line, col = params.position.line, params.position.character
        text = doc.source.splitlines()[line]
        start = text.rfind(name, 0, col + len(name))
        return lsp.PrepareRenameResult_Type1(
            range=lsp.Range(
                start=lsp.Position(line=line, character=start),
                end=lsp.Position(line=line, character=start + len(name)),
            ),
            placeholder=name,
        )

    @server.feature(lsp.TEXT_DOCUMENT_RENAME)
    def _rename(
        _ls: LanguageServer, params: lsp.RenameParams
    ) -> lsp.WorkspaceEdit | None:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return None
        name = doc.name_at_position(params.position.line, params.position.character)
        if name is None:
            return None
        if _renameable_declaration(doc, name, params.position) is None:
            return None
        if not params.new_name.isidentifier():
            return None
        edits = [
            lsp.TextEdit(range=location.range, new_text=params.new_name)
            for location in _find_references(
                doc,
                name,
                line=params.position.line,
                col=params.position.character,
            )
        ]
        if not edits:
            return None
        return lsp.WorkspaceEdit(changes={doc.uri: edits})

    @server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
    def _symbols(
        _ls: LanguageServer, params: lsp.DocumentSymbolParams
    ) -> list[lsp.DocumentSymbol]:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return []
        return _statement_symbols(doc, doc.module.statements)

    # ----- completion ---------------------------------------------------

    @server.feature(
        lsp.TEXT_DOCUMENT_COMPLETION,
        lsp.CompletionOptions(trigger_characters=[":", " ", "."]),
    )
    def _completion(
        _ls: LanguageServer, params: lsp.CompletionParams
    ) -> lsp.CompletionList:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return lsp.CompletionList(is_incomplete=False, items=[])
        session = ReplSession()
        session._module = doc.module  # noqa: SLF001
        session._compiler = doc.compiler  # noqa: SLF001
        session._qiec_module = doc.qiec_module  # noqa: SLF001
        session._env = doc.env  # noqa: SLF001
        prefix = _prefix_at(doc.source, params.position.line, params.position.character)
        items = _qiec_local_completion_items(
            doc,
            prefix,
            line=params.position.line,
            col=params.position.character,
        )
        local_names = {item.label for item in items}
        items.extend(
            lsp.CompletionItem(
                label=completion.text,
                kind=_completion_kind(completion.kind),
                detail=completion.detail,
            )
            for completion in all_completions(session, prefix)
            if completion.text not in local_names
        )
        return lsp.CompletionList(is_incomplete=False, items=items)

    # ----- formatting ---------------------------------------------------

    @server.feature(lsp.TEXT_DOCUMENT_FORMATTING)
    def _formatting(
        _ls: LanguageServer, params: lsp.DocumentFormattingParams
    ) -> list[lsp.TextEdit] | None:
        doc = docs.get(params.text_document.uri)
        return _format_document(doc) if doc is not None else None

    # The pygls feature decorator registers each handler with the
    # server, but the resulting name is never referenced from this
    # function's local scope, so the linter flags it as unused. The
    # tuple below names every handler explicitly to make the lifetime
    # contract legible to a reader and to the type checker.
    _registered_handlers = (
        _did_open,
        _did_change,
        _did_save,
        _did_close,
        _did_change_configuration,
        _semantic_tokens,
        _hover,
        _definition,
        _references,
        _prepare_rename,
        _rename,
        _symbols,
        _completion,
        _formatting,
    )
    del _registered_handlers

    return server


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _env_kinds_for(doc: DocumentState) -> dict[str, str]:
    """Build the name -> semantic-token-type map from a document's env."""
    compiler = doc.compiler
    kinds: dict[str, str] = qiec_env_kinds(doc.module)
    if compiler is None:
        return kinds
    for name in getattr(compiler, "objects", {}):
        kinds[name] = "type"
    for name in getattr(compiler, "spaces", {}):
        kinds[name] = "type"
    for name in getattr(compiler, "morphisms", {}):
        kinds[name] = "function"
    for name in getattr(compiler, "rules", {}):
        kinds[name] = "namespace"
    return kinds


def _publish(ls: LanguageServer, doc: DocumentState) -> None:
    # pygls 2.x renamed `publish_diagnostics` to follow the LSP
    # method name with snake_case prefix; the 1.x name no longer
    # exists.
    ls.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(
            uri=doc.uri,
            diagnostics=[_to_lsp_diag(d, doc) for d in doc.diagnostics],
        )
    )


def _to_lsp_diag(d: Diagnostic, doc: DocumentState) -> lsp.Diagnostic:
    line = max(0, d.line - 1) if d.line else 0
    col = max(0, d.col)
    end_line = max(0, (d.end_line or d.line or 1) - 1)
    end_col = max(col + 1, d.end_col or col + 1)
    if end_line == 0 and not d.line:
        # Whole-file diagnostic when position is unknown.
        end_line, end_col = _lsp_eof_position(doc.source)
    severity_map = {
        "error": lsp.DiagnosticSeverity.Error,
        "warning": lsp.DiagnosticSeverity.Warning,
        "info": lsp.DiagnosticSeverity.Information,
        "ok": lsp.DiagnosticSeverity.Information,
    }
    return lsp.Diagnostic(
        range=lsp.Range(
            start=lsp.Position(line=line, character=col),
            end=lsp.Position(line=end_line, character=end_col),
        ),
        message=d.message,
        severity=severity_map.get(d.severity, lsp.DiagnosticSeverity.Error),
        code=d.code or None,
        source=SERVER_NAME,
    )


def _format_document(doc: DocumentState) -> list[lsp.TextEdit] | None:
    """Return a safe whole-document canonical edit for ``doc``.

    QIEC lowering errors can leave a useful recovery AST whose normalized
    fields no longer contain every rejected token. Formatting such a document
    could erase the evidence for the diagnostic, so the formatter declines it.
    """

    if not doc.module.statements:
        return None
    if any(
        diagnostic.severity == "error"
        and (diagnostic.code == "parse" or diagnostic.code.startswith("qiec-"))
        for diagnostic in doc.diagnostics
    ):
        return None
    canonical = module_to_source(doc.module)
    if canonical == doc.source:
        return []
    end_line, end_col = _lsp_eof_position(doc.source)
    return [
        lsp.TextEdit(
            range=lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=end_line, character=end_col),
            ),
            new_text=canonical,
        )
    ]


def _lsp_eof_position(source: str) -> tuple[int, int]:
    """Return the UTF-16 LSP position immediately after ``source``."""

    lines = source.split("\n")
    final_line = lines[-1] if lines else ""
    return len(lines) - 1, len(final_line.encode("utf-16-le")) // 2


def _apply_partial(source: str, change: lsp.TextDocumentContentChangePartial) -> str:
    """Apply one incremental change to ``source``."""
    rng = change.range
    lines = source.split("\n")

    def offset(line: int, character: int) -> int:
        line = min(line, len(lines) - 1)
        return sum(len(line_text) + 1 for line_text in lines[:line]) + character

    start = offset(rng.start.line, rng.start.character)
    end = offset(rng.end.line, rng.end.character)
    return source[:start] + change.text + source[end:]


def _scoped_ref_at(doc: DocumentState, line: int, col: int) -> ScopedRef | None:
    """The binding a ``::`` scope path at a position resolves to.

    Parameters
    ----------
    doc : DocumentState
        The document.
    line : int
        The 0-based line.
    col : int
        The 0-based column.

    Returns
    -------
    ScopedRef | None
        The binding the path names up to the segment under the cursor,
        or ``None`` when no ``::`` path covers the position, the module
        did not compile, or the path names no binding.
    """
    path = doc.scoped_path_at_position(line, col)
    if path is None or doc.compiler is None:
        return None
    return resolve_scoped_path(doc.compiler, path)


def _scoped_ref_position(doc: DocumentState, ref: ScopedRef) -> tuple[int, int] | None:
    """The 0-based position of the token a scoped binding is bound at.

    Parameters
    ----------
    doc : DocumentState
        The document.
    ref : ScopedRef
        The binding.

    Returns
    -------
    tuple[int, int] | None
        The line and column of the binding's name on the line its node
        starts, at or after the node's column; ``None`` when the node
        carries no source position.
    """
    node_line = getattr(ref.node, "line", 0)
    if not isinstance(node_line, int) or node_line < 1:
        return None
    node_col = getattr(ref.node, "col", 0)
    start = node_col if isinstance(node_col, int) else 0
    lines = doc.source.splitlines()
    if node_line - 1 >= len(lines):
        return None
    text = lines[node_line - 1]
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(ref.name)}(?![A-Za-z0-9_])")
    match = pattern.search(text, start)
    return node_line - 1, match.start() if match is not None else start


def _render_scoped_hover(doc: DocumentState, ref: ScopedRef) -> str:
    """Hover markdown for a binding named by a ``::`` scope path.

    Parameters
    ----------
    doc : DocumentState
        The document.
    ref : ScopedRef
        The binding.

    Returns
    -------
    str
        The source line that binds it, when it has one, and its kind
        and full path.
    """
    parts: list[str] = []
    position = _scoped_ref_position(doc, ref)
    if position is not None:
        source = doc.source.splitlines()[position[0]].strip()
        parts.append(f"```qvr\n{source}\n```")
    parts.append(f"{ref.kind} `{ref.path}`")
    return "\n\n".join(parts)


def _render_hover(
    doc: DocumentState,
    name: str,
    *,
    line: int | None = None,
    col: int | None = None,
) -> str | None:
    """Hover Markdown: GHCi-style signature, doc comment, QVR source,
    and the didactic AST.

    Panes from top to bottom:

      1. A bolded **Type** (or **Kind**) header followed by the
         one-line GHCi signature rendered by the same code path the
         REPL's ``:type`` and ``:kind`` use, so a binding looks the
         same whether the user is hovering in the editor or asking
         the REPL.
      2. Doc comment lines harvested from the declaration.
      3. A bolded **QVR source** header.
      4. The verbatim source slice inside a fenced ```qvr block.
      5. A horizontal rule (``---``) drawing the divider.
      6. A bolded **AST (didactic)** header.
      7. The didactic ``repr()`` inside a fenced ```python block,
         collapsed behind ``<details>`` so it only takes vertical
         space when the user clicks to expand it.

    The QVR slice is taken from the original source so user
    formatting and comments survive verbatim; statements with no
    source range (appended in the REPL) render through the canonical
    emitter instead. Options render as written: a morphism with no
    ``role=`` shows no role, since the compiler infers one from
    program usage and the server does not run that inference.
    """
    if line is not None and col is not None:
        qualified = doc.qualified_name_at_position(line, col)
        if qualified is not None:
            operation = qualified_operation_signature(doc.module, qualified)
            if operation is not None:
                lines = [operation]
                owner, _, member = qualified.partition(".")
                interface = instance_effect_name(doc.module, owner)
                if interface is not None:
                    # An instance's operation, followed by the interface's
                    # own declaration of it.
                    declared = qualified_operation_signature(
                        doc.module, f"{interface}.{member}"
                    )
                    if declared is not None and declared != operation:
                        lines.append(declared)
                return "**Operation**\n\n```qvr\n" + "\n".join(lines) + "\n```"
    qiec_local = (
        doc.find_qiec_local(name, line=line, col=col) if line is not None else None
    )
    qiec_binding = (
        None
        if qiec_local is not None
        else doc.find_qiec_binding(name, line=line, col=col)
    )
    decl = (
        qiec_local
        if qiec_local is not None
        else (
            qiec_binding.declaration
            if qiec_binding is not None
            else doc.find_decl(name, line=line, col=col)
        )
    )
    if decl is None:
        if name in doc.env:
            return f"```\n{name} :: {type(doc.env[name]).__name__}\n```"
        return None
    qvr = _slice_source(doc, decl)
    if isinstance(decl, QiecBinder | QiecValueParameter | QiecLocalBinding) or (
        isinstance(decl, QiecTypeName) and qiec_local is decl
    ):
        source_lines = doc.source.splitlines()
        source_line = max(0, getattr(decl, "line", 1) - 1)
        qvr = (
            source_lines[source_line].strip()
            if source_line < len(source_lines)
            else name
        )
    if qvr is None:
        if isinstance(decl, Statement):
            qvr = module_to_source(type(doc.module)(statements=(decl,))).rstrip()
        else:
            qvr = render_qiec_signature(doc.module, name) or repr(decl)
    python_repr = _pretty_ast(decl)
    docs = getattr(decl, "docs", ())

    parts: list[str] = []
    site = (
        call_site_at(doc.module, line, col)
        if line is not None and col is not None
        else None
    )
    if site is not None and site.callee == name:
        instantiated = instantiated_call_signature(doc.module, site)
        if instantiated is not None:
            parts.append("**Call**")
            parts.append(f"```qvr\n{instantiated}\n```")
    inferred = (
        _inferred_row(doc, name) if isinstance(decl, QiecComputationDecl) else None
    )
    if inferred is not None:
        parts.append(f"**Inferred row**\n\n```qvr\n{inferred}\n```")
    lookup_name = qiec_binding.qualified_name if qiec_binding is not None else name
    if qiec_local is decl:
        signature = _qiec_local_signature(decl)
    else:
        signature = render_qiec_signature(doc.module, lookup_name) or render_signature(
            doc.compiler, name
        )
    if signature is not None:
        header = (
            "Kind"
            if signature.startswith(
                (
                    "object ",
                    "space ",
                    "signature ",
                    "category ",
                    "index ",
                    "family ",
                    "effect ",
                )
            )
            else "Type"
        )
        parts.append(f"**{header}**")
        parts.append(f"```qvr\n{signature}\n```")
    if docs:
        parts.append("\n".join(docs))
    parts.append("**QVR source**")
    parts.append(f"```qvr\n{qvr}\n```")
    parts.append("---")
    parts.append("**AST (didactic)**")
    parts.append(
        "<details><summary><i>click to expand</i></summary>\n\n"
        f"```python\n{python_repr}\n```\n\n"
        "</details>"
    )
    return "\n\n".join(parts)


def _inferred_row(doc: DocumentState, name: str) -> str | None:
    """The row the checker infers for a computation's body.

    Parameters
    ----------
    doc : DocumentState
        The document, whose checked module holds the computation.
    name : str
        The computation's name.

    Returns
    -------
    str | None
        The row after every call is expanded to its callee's signature,
        rendered with the instances' source names; ``None`` when the
        document has no checked module or no computation of that name.
    """
    module = doc.qiec_module
    if module is None:
        return None
    registry = doc.registry()
    if registry is None:
        return None
    try:
        inferred = inferred_computation_type(module, registry, name)
    except KeyError:
        return None
    return render_row(inferred.effects, registry.instance_names)


type _AstValue = (
    dx.Model
    | str
    | int
    | float
    | bool
    | None
    | tuple["_AstValue", ...]
    | list["_AstValue"]
    | dict[str, "_AstValue"]
)
"""Anything a didactic AST field can hold: a nested model, a scalar,
or a container of the same."""


def _pretty_ast(decl: object) -> str:
    """Pretty-print a didactic AST node, one field per line.

    Plain ``repr()`` puts the whole struct on one line; for deeply
    nested QVR declarations (kernels with product domains, programs
    with chains of binds) that produces a single ~200-column blob
    that hover panes truncate. Walk the didactic model tree manually
    so every field gets its own indented line, matching Python's
    standard ``pprint`` shape but preserving the keyword=value
    syntax that didactic uses.
    """
    return _ast_lines(cast(_AstValue, decl), indent=0)


def _ast_lines(value: _AstValue, indent: int) -> str:
    """Recursive renderer used by `_pretty_ast`."""
    pad = "    " * indent
    next_pad = "    " * (indent + 1)
    # didactic models expose __field_specs__; treat them as records.
    field_specs = getattr(type(value), "__field_specs__", None)
    if field_specs is not None:
        class_name = type(value).__name__
        fields = []
        for fname in field_specs:
            attr = getattr(value, fname, None)
            # Skip empty-tuple `docs` and the synthetic `kind`
            # discriminator that didactic stamps on every tagged
            # union; both add noise without information.
            if fname == "kind":
                continue
            if fname == "docs" and not attr:
                continue
            if attr is None:
                rendered = "None"
            else:
                rendered = _ast_lines(attr, indent + 1)
            fields.append(f"{next_pad}{fname}={rendered}")
        if not fields:
            return f"{class_name}()"
        body = ",\n".join(fields)
        return f"{class_name}(\n{body},\n{pad})"
    if isinstance(value, tuple):
        if not value:
            return "()"
        items = ",\n".join(f"{next_pad}{_ast_lines(v, indent + 1)}" for v in value)
        return f"(\n{items},\n{pad})"
    if isinstance(value, list):
        if not value:
            return "[]"
        items = ",\n".join(f"{next_pad}{_ast_lines(v, indent + 1)}" for v in value)
        return f"[\n{items},\n{pad}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = ",\n".join(
            f"{next_pad}{k!r}: {_ast_lines(v, indent + 1)}" for k, v in value.items()
        )
        return f"{{\n{items},\n{pad}}}"
    return repr(value)


def _decl_line_span(doc: DocumentState, decl: object) -> tuple[int, int] | None:
    """1-based ``[start, end)`` line span of ``decl``'s source slice.

    The span runs from the statement's recorded line to the line of
    the next top-level statement (or one past the last source line),
    or ``None`` when the statement carries no source position.
    """
    start_line = getattr(decl, "line", 0)
    if not start_line:
        return None
    lines = doc.source.splitlines()
    if start_line - 1 >= len(lines):
        return None
    end_line = len(lines) + 1
    candidates: list[object] = list(doc.module.statements)
    binding = next(
        (item for item in qiec_bindings(doc.module) if item.declaration == decl),
        None,
    )
    if binding is not None and binding.declaration is not binding.owner:
        candidates.extend(
            item.declaration
            for item in qiec_bindings(doc.module)
            if item.owner is binding.owner and item.declaration is not item.owner
        )
    for other in candidates:
        if other == decl:
            continue
        other_line = getattr(other, "line", 0)
        if other_line > start_line and other_line < end_line:
            end_line = other_line
    return start_line, end_line


def _slice_source(doc: DocumentState, decl: object) -> str | None:
    """Return the original source lines that produced ``decl``."""
    span = _decl_line_span(doc, decl)
    if span is None:
        return None
    start_line, end_line = span
    lines = doc.source.splitlines()
    while end_line - 1 > start_line and not lines[end_line - 2].strip():
        end_line -= 1
    return "\n".join(lines[start_line - 1 : end_line - 1])


def _name_position(doc: DocumentState, decl: object, name: str) -> tuple[int, int]:
    """0-based ``(line, col)`` of ``name``'s own binding token.

    A plural-name declaration binds several names in one statement, so
    each name's definition target is its own token, located by a
    word-boundary scan of the declaration's source slice. Comment
    lines are skipped since a doc comment above the declaration may
    mention the name. Statements with no source position (appended in
    the REPL) anchor at the declaration's recorded coordinates.
    """
    decl_line = getattr(decl, "line", 0) or 0
    decl_col = getattr(decl, "col", 0) or 0
    span = _decl_line_span(doc, decl)
    if span is None:
        return max(0, decl_line - 1), max(0, decl_col)
    start_line, end_line = span
    lines = doc.source.splitlines()
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
    for lineno in range(start_line - 1, min(end_line - 1, len(lines))):
        text = lines[lineno]
        if text.lstrip().startswith("#"):
            continue
        m = pattern.search(text)
        if m is not None:
            return lineno, m.start()
    return max(0, decl_line - 1), max(0, decl_col)


def _statement_symbols(
    doc: DocumentState, statements: tuple[Statement, ...]
) -> list[lsp.DocumentSymbol]:
    """Document symbols for ``statements``, one per bound name.

    A plural-name declaration yields one symbol per name, each
    anchored at that name's own token. ``define`` where-blocks nest
    as children of the enclosing define's symbol.
    """
    out: list[lsp.DocumentSymbol] = []
    for stmt in statements:
        children = _statement_symbol_children(doc, stmt)
        for name in decl_names(stmt):
            line, col = _name_position(doc, stmt, name)
            rng = lsp.Range(
                start=lsp.Position(line=line, character=col),
                end=lsp.Position(line=line, character=col + len(name)),
            )
            out.append(
                lsp.DocumentSymbol(
                    name=name,
                    kind=_symbol_kind(stmt),
                    range=rng,
                    selection_range=rng,
                    children=children or None,
                )
            )
    return out


def _statement_symbol_children(
    doc: DocumentState, statement: Statement
) -> list[lsp.DocumentSymbol]:
    """Return nested declarations exposed by one top-level symbol."""

    if isinstance(statement, DefineDecl):
        return _statement_symbols(doc, statement.where)
    if isinstance(statement, QiecComputationDecl):
        return _body_symbols(doc, statement.name)
    if isinstance(statement, ProgramDecl):
        return _program_step_symbols(doc, statement)
    members: tuple[object, ...] = ()
    kind = lsp.SymbolKind.Variable
    if isinstance(statement, QiecIndexDecl):
        members = statement.constructors
        kind = lsp.SymbolKind.EnumMember
    elif isinstance(statement, QiecFamilyDecl):
        members = statement.constructors
        kind = lsp.SymbolKind.Constructor
    elif isinstance(statement, QiecEffectDecl):
        members = statement.operations
        kind = lsp.SymbolKind.Method
    elif isinstance(statement, QiecHandlerDecl):
        members = statement.clauses
        kind = lsp.SymbolKind.Method

    out: list[lsp.DocumentSymbol] = []
    for member in members:
        name = getattr(member, "name", None) or getattr(member, "operation", None)
        if isinstance(member, QiecHandlerReturnClause):
            name = "return"
        if not isinstance(name, str):
            continue
        line, col = _name_position(doc, member, name)
        rng = lsp.Range(
            start=lsp.Position(line=line, character=col),
            end=lsp.Position(line=line, character=col + len(name)),
        )
        out.append(
            lsp.DocumentSymbol(
                name=name,
                kind=kind,
                range=rng,
                selection_range=rng,
            )
        )
    if isinstance(statement, QiecHandlerDecl):
        # A clause's binders and the calls its body makes nest under the
        # clause that introduces them.
        nested = _body_symbols(doc, statement.name)
        for clause_symbol, clause in zip(out, statement.clauses):
            first = clause_symbol.range.start.line
            following = [
                other.range.start.line
                for other in out
                if other.range.start.line > first
            ]
            last = min(following) if following else None
            children = [
                symbol
                for symbol in nested
                if symbol.range.start.line >= first
                and (last is None or symbol.range.start.line < last)
            ]
            clause_symbol.children = children or None
            del clause
    return out


def _symbol_at(
    name: str, line: int, col: int, kind: lsp.SymbolKind, detail: str | None = None
) -> lsp.DocumentSymbol:
    """A document symbol anchored at one identifier.

    Parameters
    ----------
    name : str
        The symbol's name.
    line : int
        The 0-based line of the identifier.
    col : int
        The 0-based column of the identifier.
    kind : lsp.SymbolKind
        The symbol kind.
    detail : str | None
        A detail string, shown beside the name.

    Returns
    -------
    lsp.DocumentSymbol
        The symbol, its range the identifier's own.
    """
    rng = lsp.Range(
        start=lsp.Position(line=line, character=col),
        end=lsp.Position(line=line, character=col + len(name)),
    )
    return lsp.DocumentSymbol(
        name=name, kind=kind, range=rng, selection_range=rng, detail=detail
    )


def _body_symbols(doc: DocumentState, owner: str) -> list[lsp.DocumentSymbol]:
    """The symbols nested in a computation: its binders and its calls.

    Parameters
    ----------
    doc : DocumentState
        The document.
    owner : str
        The computation's name.

    Returns
    -------
    list[lsp.DocumentSymbol]
        One symbol per lexical binder (a let, a scoped instance, a
        branch binder) and one per call site, in source order.
    """
    out: list[tuple[int, int, lsp.DocumentSymbol]] = []
    for binding in qiec_local_bindings(doc.module):
        if getattr(binding.owner, "name", None) != owner:
            continue
        line = max(0, getattr(binding.declaration, "line", 1) - 1)
        col = getattr(binding.declaration, "col", 0)
        kind = (
            lsp.SymbolKind.Object
            if binding.kind == "scoped-instance"
            else lsp.SymbolKind.Variable
        )
        out.append((line, col, _symbol_at(binding.name, line, col, kind, binding.kind)))
    for site in call_sites(doc.module):
        if site.owner != owner:
            continue
        line = max(0, site.line - 1)
        detail = "recursive call" if site.callee == owner else "call"
        out.append(
            (
                line,
                site.col,
                _symbol_at(
                    site.callee, line, site.col, lsp.SymbolKind.Function, detail
                ),
            )
        )
    return [symbol for _, _, symbol in sorted(out, key=lambda item: (item[0], item[1]))]


def _program_step_symbols(
    doc: DocumentState, program: ProgramDecl
) -> list[lsp.DocumentSymbol]:
    """The symbols nested in a program: the names its steps bind.

    Parameters
    ----------
    doc : DocumentState
        The document.
    program : ProgramDecl
        The program.

    Returns
    -------
    list[lsp.DocumentSymbol]
        One symbol per step, named by the variable it binds and
        detailed by its kind, a call step detailed by its callee; a
        marginalize block's scope nests under its own symbol.
    """

    def steps(items: tuple[object, ...]) -> list[lsp.DocumentSymbol]:
        """The symbols of one step sequence.

        Parameters
        ----------
        items : tuple[object, ...]
            The steps.

        Returns
        -------
        list[lsp.DocumentSymbol]
            Their symbols in order.
        """
        out: list[lsp.DocumentSymbol] = []
        for step in items:
            names = getattr(step, "vars", None) or (
                (getattr(step, "name", None),) if getattr(step, "name", None) else ()
            )
            names = tuple(name for name in names if isinstance(name, str))
            if not names:
                continue
            line = max(0, getattr(step, "line", 1) - 1)
            col = getattr(step, "col", 0)
            kind_name = type(step).__name__.removesuffix("Step").lower()
            detail = kind_name
            if isinstance(step, CallStep):
                detail = f"call {step.call.callee}"
            symbol = _symbol_at(
                ", ".join(names), line, col, lsp.SymbolKind.Variable, detail
            )
            scope = getattr(step, "scope", None)
            if isinstance(scope, tuple) and scope:
                symbol.children = steps(scope) or None
            out.append(symbol)
        return out

    return steps(program.draws)


def _renameable_declaration(
    doc: DocumentState, name: str, position: lsp.Position
) -> object | None:
    """The declaration a rename at a position would rename, if any.

    Parameters
    ----------
    doc : DocumentState
        The document.
    name : str
        The identifier at the position.
    position : lsp.Position
        The position.

    Returns
    -------
    object | None
        The declaration the identifier resolves to when it is declared
        in this document: a top-level declaration, a signature member,
        or a lexical binder. ``None`` for a keyword, a prelude name, or
        a name the document does not declare, none of which a rename
        may touch.
    """
    if prelude_effect(name) is not None:
        return None
    return doc.find_decl(name, line=position.line, col=position.character)


def _find_references(
    doc: DocumentState,
    name: str,
    *,
    line: int | None = None,
    col: int | None = None,
) -> Iterator[lsp.Location]:
    """Locate references that resolve to the selected declaration identity.

    Merely matching text conflates branch locals, static binders, and repeated
    operation names. Resolve every occurrence at its own lexical position and
    retain it only when it names the exact selected AST declaration.
    """

    selected = doc.find_decl(name, line=line, col=col)
    if selected is None:
        return
    selected_identity = _declaration_identity(selected, name)
    for lineno, source_text in enumerate(doc.source.splitlines()):
        start = 0
        while True:
            idx = source_text.find(name, start)
            if idx == -1:
                break
            # Word-boundary check.
            left_ok = idx == 0 or not (
                source_text[idx - 1].isalnum() or source_text[idx - 1] == "_"
            )
            right_ok = idx + len(name) == len(source_text) or not (
                source_text[idx + len(name)].isalnum()
                or source_text[idx + len(name)] == "_"
            )
            resolved = (
                doc.find_decl(name, line=lineno, col=idx)
                if left_ok and right_ok
                else None
            )
            if (
                resolved is not None
                and _declaration_identity(resolved, name) == selected_identity
            ):
                yield lsp.Location(
                    uri=doc.uri,
                    range=lsp.Range(
                        start=lsp.Position(line=lineno, character=idx),
                        end=lsp.Position(line=lineno, character=idx + len(name)),
                    ),
                )
            start = idx + len(name)


def _declaration_identity(declaration: object, name: str) -> tuple[object, ...]:
    """Return a stable source identity despite Didactic's defensive copies."""

    return (
        type(declaration),
        name,
        getattr(declaration, "line", 0),
        getattr(declaration, "col", 0),
    )


def _qiec_local_signature(local: object) -> str:
    """Render one lexical QIEC binder for hover and completion details."""

    name = str(getattr(local, "name", ""))
    if isinstance(local, QiecBinder):
        return _emit_qiec_binder(local)
    if isinstance(local, QiecValueParameter):
        return f"{name} : {_emit_qiec_type(local.type_expr)}"
    if isinstance(local, QiecLocalBinding):
        return (
            f"{name} : {_emit_qiec_type(local.type_expr)}"
            if local.type_expr is not None
            else f"{name} : (inferred)"
        )
    if isinstance(local, QiecTypeName):
        return f"{name} : (branch static)"
    raise TypeError(f"unsupported QIEC local {type(local).__name__}")


def _qiec_local_completion_items(
    doc: DocumentState,
    prefix: str,
    *,
    line: int,
    col: int,
) -> list[lsp.CompletionItem]:
    """Return position-visible QIEC locals, with shadowing de-duplicated."""

    if any(separator in prefix for separator in (".", ":")):
        return []
    locals_by_name: dict[str, object] = {}
    for local in doc.visible_qiec_locals(line=line, col=col):
        name = getattr(local, "name", None)
        if isinstance(name, str):
            locals_by_name[name] = local
    return [
        lsp.CompletionItem(
            label=name,
            kind=(
                lsp.CompletionItemKind.TypeParameter
                if isinstance(local, QiecBinder | QiecTypeName)
                else lsp.CompletionItemKind.Variable
            ),
            detail=_qiec_local_signature(local),
        )
        for name, local in locals_by_name.items()
        if name.startswith(prefix)
    ]


def _prefix_at(source: str, line: int, character: int) -> str:
    lines = source.splitlines()
    if line >= len(lines):
        return ""
    text = lines[line][:character]
    i = len(text)
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] in "_:."):
        i -= 1
    return text[i:]


def _symbol_kind(stmt: Statement) -> lsp.SymbolKind:
    """Classify a top-level decl as a Class / Struct / Function symbol.

    For a `ObjectDecl`, peek at the initializer's inner expression
    to distinguish discrete objects (``Class``) from continuous spaces
    (``Struct``). Unwrapped or aliased forms default to ``Class``.
    A `DefineDecl` binds a morphism-valued expression, so it
    classifies as ``Function`` alongside `MorphismDecl`.
    """
    if isinstance(stmt, ObjectDecl):
        init: TypeInitializer = stmt.init
        if isinstance(init, TypeFromExpr) and isinstance(
            init.expr,
            ContinuousConstructor,
        ):
            return lsp.SymbolKind.Struct
        return lsp.SymbolKind.Class
    if isinstance(stmt, (MorphismDecl, DefineDecl)):
        return lsp.SymbolKind.Function
    if isinstance(stmt, (QiecIndexDecl, QiecFamilyDecl)):
        return lsp.SymbolKind.Class
    if isinstance(stmt, QiecEffectDecl):
        return lsp.SymbolKind.Interface
    if isinstance(stmt, QiecEffectInstanceDecl):
        return lsp.SymbolKind.Object
    if isinstance(stmt, (QiecHandlerDecl, QiecComputationDecl)):
        return lsp.SymbolKind.Function
    return lsp.SymbolKind.Variable


def _completion_kind(kind: str) -> lsp.CompletionItemKind:
    return {
        "command": lsp.CompletionItemKind.Operator,
        "env": lsp.CompletionItemKind.Variable,
        "keyword": lsp.CompletionItemKind.Keyword,
        "type": lsp.CompletionItemKind.Class,
        "function": lsp.CompletionItemKind.Function,
        "namespace": lsp.CompletionItemKind.Module,
        "path": lsp.CompletionItemKind.File,
    }.get(kind, lsp.CompletionItemKind.Text)


__all__ = ["build_server", "SERVER_NAME", "SERVER_VERSION"]
