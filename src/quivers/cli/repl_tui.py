"""Textual TUI for ``qvr repl``.

Layout
------

::

    +--- Status bar -----------------------------------------+
    | Input (TextArea, multi-line)    | Env filter           |
    |                                 |----------------------|
    |                                 | Env tree             |
    +---------------------------------|                      |
    | Output log (rich text)          |                      |
    |                                 |                      |
    +---------------------------------+----------------------+
    | Watches (hidden if empty)                              |
    +--------------------------------------------------------+
    | Diagnostics (hidden if empty)                          |
    +--------------------------------------------------------+

Key bindings (chosen to work uniformly on macOS, Linux, and Windows):

- ``ctrl+g``      evaluate the current buffer (semantic: "go")
- ``ctrl+o``      evaluate (alternate)
- ``f8``          evaluate (Fn-row alternate)
- ``ctrl+up``     previous input from history
- ``ctrl+down``   next input from history
- ``tab``         complete the identifier under the cursor
- ``ctrl+p``      command palette
- ``ctrl+l``      clear the eval log
- ``ctrl+r``      reload the loaded file
- ``ctrl+q``      quit
- ``f1``          help

The widgets all draw from a single `ReplSession` instance.
Highlighting is driven by [`tokenize`][quivers.cli.tokenize], so the
TUI tracks the live grammar and the live env.
"""

from __future__ import annotations

import os
import subprocess
from importlib.resources import files
from pathlib import Path

from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Footer,
    Input,
    RichLog,
    Static,
    TextArea,
    Tree,
)

from quivers.analysis.scope import (
    SCOPE_SEPARATOR,
    ScopedRef,
    scope_children,
    top_level_bindings,
)
from quivers.cli.repl_highlight import to_rich_text, tokenize
from quivers.cli.repl_complete import all_completions, public_meta_commands
from quivers.cli.repl_session import (
    HELP_CATEGORIES,
    KEY_BINDINGS,
    ReplSession,
    render_qiec_signature,
)
from quivers.dsl.pygments_lexer import _load_parser
from quivers.dsl.qiec_tooling import qiec_bindings


_HISTORY_PATH = (
    Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config"))
    / "quivers"
    / "history"
)


def run_tui(session: ReplSession) -> int:
    """Run the Textual REPL front end on ``session``.

    Parameters
    ----------
    session : ReplSession
        The session that evaluates each input.

    Returns
    -------
    int
        The process exit status, ``0`` once the app exits.
    """
    app = _build_tui_app(session)
    app.run()  # type: ignore[attr-defined]
    return 0


def _build_tui_app(session: ReplSession) -> object:
    """Construct the real Textual app, with an in-process test seam."""

    META_COMMANDS = public_meta_commands()

    class _MetaCommandProvider(Provider):
        """Surface every meta-command in the Ctrl-P palette."""

        async def search(self, query: str) -> Hits:
            matcher = self.matcher(query)
            for name in META_COMMANDS:
                label = f":{name}"
                score = matcher.match(label)
                if score > 0:
                    yield Hit(
                        score,
                        matcher.highlight(label),
                        partial_run := _make_runner(self.app, label),
                        help=f"meta-command :{name}",
                    )
                    del partial_run

    def _make_runner(app, command: str):  # type: ignore[no-untyped-def]
        def _run() -> None:
            app.run_meta_command(command)

        return _run

    class _HelpScreen(ModalScreen):
        """Modal overlay listing every meta-command + key binding.

        Pushed by ``action_help`` (``F1`` or ``:help``). Dismissed
        with ``Esc`` or by pressing the close button. The filter
        input narrows the visible commands live as the user types.
        """

        BINDINGS = [
            Binding("escape", "dismiss", "Close", show=True, priority=True),
        ]

        CSS = """
        _HelpScreen {
            align: center middle;
        }
        _HelpScreen > #help-frame {
            background: $panel;
            border: thick $primary;
            padding: 1 2;
            width: 90%;
            max-width: 100;
            height: 80%;
        }
        _HelpScreen #help-title {
            text-style: bold;
            color: $accent;
            padding-bottom: 1;
        }
        _HelpScreen #help-filter {
            border: tall $primary;
            margin-bottom: 1;
        }
        _HelpScreen #help-body {
            height: 1fr;
        }
        """

        def __init__(self) -> None:
            super().__init__()
            self._filter = ""

        def compose(self) -> ComposeResult:
            self._HELP_CATEGORIES = HELP_CATEGORIES
            self._KEY_BINDINGS = KEY_BINDINGS
            with Vertical(id="help-frame"):
                yield Static("qvr repl: help (Esc to close)", id="help-title")
                yield Input(placeholder="filter… (substring match)", id="help-filter")
                yield RichLog(id="help-body", highlight=False, markup=False)

        def on_mount(self) -> None:
            self._render_help_body()
            try:
                self.query_one("#help-filter", Input).focus()
            except Exception:
                pass

        def on_input_changed(self, event) -> None:  # type: ignore[no-untyped-def]
            if event.input.id == "help-filter":
                self._filter = event.value.strip().lower()
                self._render_help_body()

        def _render_help_body(self) -> None:
            log = self.query_one("#help-body", RichLog)
            log.clear()
            needle = self._filter

            def matches(*fields: str) -> bool:
                if not needle:
                    return True
                return any(needle in f.lower() for f in fields)

            for cat, entries in self._HELP_CATEGORIES:
                visible = [
                    (cmd, summary) for cmd, summary in entries if matches(cmd, summary)
                ]
                if not visible:
                    continue
                log.write(Text(cat, style="bold cyan"))
                for cmd, summary in visible:
                    line = Text()
                    line.append("  ")
                    line.append(cmd, style="bold")
                    line.append("  ")
                    line.append(summary, style="dim")
                    log.write(line)
                log.write(Text(""))
            kb_visible = [(k, d) for k, d in self._KEY_BINDINGS if matches(k, d)]
            if kb_visible:
                log.write(Text("keybindings", style="bold cyan"))
                for k, d in kb_visible:
                    line = Text()
                    line.append("  ")
                    line.append(k, style="bold")
                    line.append("  ")
                    line.append(d, style="dim")
                    log.write(line)

        def action_dismiss(self) -> None:
            self.app.pop_screen()

    class _QvrTextArea(TextArea):
        """TextArea with brace-balancing and auto-indent on Enter.

        These are pure key handlers; no language hook required, so we
        avoid Textual's tree-sitter shipping list (which lacks QVR).
        """

        BRACE_PAIRS = {"(": ")", "[": "]", "{": "}"}

        async def _on_key(self, event) -> None:  # type: ignore[no-untyped-def]
            char = event.character
            if char in self.BRACE_PAIRS:
                self.insert(char + self.BRACE_PAIRS[char])
                row, col = self.cursor_location
                self.move_cursor((row, col - 1))
                event.prevent_default()
                event.stop()
                return
            if event.key == "enter":
                row, col = self.cursor_location
                current_line = self.document.get_line(row)
                indent = ""
                for c in current_line[:col]:
                    if c in (" ", "\t"):
                        indent += c
                    else:
                        break
                # Indent one level after a line ending in an open
                # brace / colon / arrow / `=` (declaration body).
                trailing = current_line[:col].rstrip()
                extra = ""
                if trailing.endswith(("(", "[", "{", ":", "->", "=", "<-")):
                    extra = "    "
                self.insert("\n" + indent + extra)
                event.prevent_default()
                event.stop()
                return
            await super()._on_key(event)

    class QvrRepl(App):
        CSS = """
        Screen { layout: vertical; }
        #status { dock: top; height: 1; background: $boost; padding: 0 1; }
        #top { height: 1fr; }
        #input-pane { width: 2fr; }
        #env-pane { width: 1fr; border-left: tall $primary; }
        #env-filter { height: 1; border: none; padding: 0 1; }
        #env { height: 1fr; }
        #input { height: 1fr; }
        #output { height: 1fr; border-top: tall $primary; }
        #watches { height: auto; max-height: 8; border-top: tall $accent;
                   padding: 0 1; display: none; }
        #watches.has-content { display: block; }
        #diagnostics { height: auto; max-height: 8; border-top: tall $error;
                       padding: 0 1; display: none; }
        #diagnostics.has-content { display: block; }
        """

        COMMANDS = {_MetaCommandProvider}

        BINDINGS = [
            Binding("ctrl+g", "submit", "Eval", show=True, priority=True),
            Binding("ctrl+o", "submit", "Eval", show=True, priority=True),
            Binding("f8", "submit", "Eval", show=True, priority=True),
            Binding("ctrl+enter", "submit", "Eval", show=False, priority=True),
            Binding("ctrl+j", "submit", "Eval", show=False, priority=True),
            Binding("ctrl+up", "history_prev", "Prev", show=True, priority=True),
            Binding("ctrl+down", "history_next", "Next", show=False, priority=True),
            Binding("tab", "complete", "Complete", show=True, priority=True),
            Binding("ctrl+l", "clear", "Clear", show=True),
            Binding("ctrl+r", "reload", "Reload", show=True),
            Binding("ctrl+q", "quit", "Quit", show=True),
            Binding("f1", "help", "Help", show=True),
        ]

        def __init__(self, session) -> None:  # type: ignore[no-untyped-def]
            super().__init__()
            self.session = session
            self._history: list[str] = _load_history()
            self._history_idx: int | None = None
            self._completion_cycle: list[str] = []
            self._completion_anchor: tuple[int, int] | None = None
            self._completion_prefix_len: int = 0
            self._completion_idx: int = 0

        def compose(self) -> ComposeResult:
            yield Static("", id="status")
            with Horizontal(id="top"):
                with Vertical(id="input-pane"):
                    yield _QvrTextArea(
                        id="input",
                        language=None,
                        show_line_numbers=True,
                    )
                    yield RichLog(
                        id="output",
                        wrap=True,
                        markup=False,
                        highlight=False,
                    )
                with Vertical(id="env-pane"):
                    yield Input(placeholder="filter env...", id="env-filter")
                    yield Tree("environment", id="env")
            yield Static("", id="watches")
            yield Static("", id="diagnostics")
            yield Footer()

        def on_mount(self) -> None:
            input_widget = self.query_one("#input", TextArea)
            _enable_qvr_highlighting(input_widget)
            input_widget.focus()
            self._refresh_status()
            self._refresh_env()
            if self.session.loaded_path is not None:
                self._log(Text(f"loaded {self.session.loaded_path}", style="dim"))
            # File watcher: poll the loaded file every 1s and fire :reload
            # automatically when its mtime advances.
            self.set_interval(1.0, self._poll_for_file_change)

        # --- actions ----------------------------------------------------

        def action_submit(self) -> None:
            input_widget: TextArea = self.query_one("#input", TextArea)
            text = input_widget.text
            if not text.strip():
                return
            self._log(Text("> ", style="bold cyan") + Text(text))
            self._push_history(text)
            response = self.session.dispatch(text)
            if response.body == "__quit__":
                self.exit(0)
                return
            self._render(response)
            self._refresh_env()
            self._refresh_status()
            self._refresh_watches()
            input_widget.clear()
            self._reset_completion()

        def action_clear(self) -> None:
            self.query_one("#output", RichLog).clear()

        def action_reload(self) -> None:
            self._render(self.session.reload())
            self._refresh_env()
            self._refresh_status()
            self._refresh_watches()

        def action_help(self) -> None:
            self.push_screen(_HelpScreen())

        def action_info(self, name: str) -> None:
            """Click handler for identifiers rendered with @click=info."""
            self._log(Text(f"> :info {name}", style="bold cyan"))
            self._render(self.session.info(name))

        def action_open_at(self, path: str, line: str) -> None:
            """Click handler for `path:line:col` footer entries.

            Opens the file at the given line in ``$EDITOR``. Defaults
            to ``vi`` when the env var is unset.
            """
            editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
            try:
                subprocess.Popen(
                    [editor, f"+{line}", path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as e:
                self._log(Text(f"could not launch editor: {e}", style="red"))

        def action_history_prev(self) -> None:
            if not self._history:
                return
            idx = (
                len(self._history) - 1
                if self._history_idx is None
                else max(0, self._history_idx - 1)
            )
            self._history_idx = idx
            self._set_input_text(self._history[idx])

        def action_history_next(self) -> None:
            if self._history_idx is None:
                return
            idx = self._history_idx + 1
            if idx >= len(self._history):
                self._history_idx = None
                self._set_input_text("")
                return
            self._history_idx = idx
            self._set_input_text(self._history[idx])

        def action_complete(self) -> None:
            input_widget: TextArea = self.query_one("#input", TextArea)
            cursor_row, cursor_col = input_widget.cursor_location
            line = input_widget.document.get_line(cursor_row)
            prefix = _word_prefix(line, cursor_col)
            anchor = (cursor_row, cursor_col)
            if self._completion_anchor == anchor and self._completion_cycle:
                # Cycle: rotate through the existing candidate list.
                self._completion_idx = (self._completion_idx + 1) % len(
                    self._completion_cycle
                )
                self._apply_completion(input_widget, prefix)
                return
            # Build a fresh candidate list at this cursor position.
            completions = all_completions(self.session, prefix)
            if not completions:
                return
            self._completion_cycle = [c.text for c in completions]
            self._completion_idx = 0
            self._completion_prefix_len = len(prefix)
            self._apply_completion(input_widget, prefix)
            self._completion_anchor = (
                cursor_row,
                cursor_col - len(prefix) + len(self._completion_cycle[0]),
            )

        def run_meta_command(self, command: str) -> None:
            """Entry point for command-palette execution."""
            self._set_input_text(command + " ")
            self.query_one("#input", TextArea).focus()

        # --- env / status / watches ------------------------------------

        def on_tree_node_selected(self, event) -> None:  # type: ignore[no-untyped-def]
            name = resolve_click_target(event.node)
            if name is None:
                return
            self._log(Text(f"> :info {name}", style="bold cyan"))
            self._render(self.session.info(name))

        def on_input_changed(self, event) -> None:  # type: ignore[no-untyped-def]
            if event.input.id == "env-filter":
                self._refresh_env(filter_text=event.value.strip())

        # --- file watcher ----------------------------------------------

        def _poll_for_file_change(self) -> None:
            response = self.session.autoreload_if_stale()
            if response is None:
                return
            self._log(Text("auto-reload", style="italic dim cyan"))
            self._render(response)
            self._refresh_env()
            self._refresh_status()
            self._refresh_watches()

        # --- helpers ----------------------------------------------------

        def _log(self, message) -> None:  # type: ignore[no-untyped-def]
            self.query_one("#output", RichLog).write(message)

        def _set_input_text(self, text: str) -> None:
            input_widget: TextArea = self.query_one("#input", TextArea)
            input_widget.text = text
            input_widget.move_cursor(input_widget.document.end)

        def _apply_completion(self, input_widget, prefix: str) -> None:  # type: ignore[no-untyped-def]
            candidate = self._completion_cycle[self._completion_idx]
            cursor_row, cursor_col = input_widget.cursor_location
            start = (cursor_row, cursor_col - self._completion_prefix_len)
            end = (cursor_row, cursor_col)
            input_widget.replace(candidate, start, end)
            # Reanchor at the new cursor position so subsequent Tab
            # presses cycle relative to the same prefix span.
            new_col = cursor_col - self._completion_prefix_len + len(candidate)
            self._completion_anchor = (cursor_row, new_col)
            self._completion_prefix_len = len(candidate)

        def _reset_completion(self) -> None:
            self._completion_cycle = []
            self._completion_anchor = None
            self._completion_prefix_len = 0
            self._completion_idx = 0

        def _push_history(self, line: str) -> None:
            line = line.strip()
            if not line:
                return
            if self._history and self._history[-1] == line:
                self._history_idx = None
                return
            self._history.append(line)
            self._history_idx = None
            _append_history(line)

        def _render(self, response) -> None:  # type: ignore[no-untyped-def]
            log: RichLog = self.query_one("#output", RichLog)
            if response.body:
                if response.body_kind == "qvr":
                    env_kinds = self.session.env_kinds()
                    for line in response.body.splitlines() or [""]:
                        stripped = line.lstrip()
                        if stripped.startswith("--") and not stripped.startswith("->"):
                            log.write(_decorate_comment_line(line))
                        else:
                            log.write(
                                _to_tui_rich_text(
                                    line,
                                    env_kinds=env_kinds,
                                    link_action="info",
                                )
                            )
                else:
                    log.write(response.body)
            self._update_diagnostics(response.diagnostics)

        def _update_diagnostics(self, diagnostics) -> None:  # type: ignore[no-untyped-def]
            panel: Static = self.query_one("#diagnostics", Static)
            if diagnostics:
                rendered = Text()
                first = True
                for d in diagnostics:
                    if not first:
                        rendered.append("\n")
                    first = False
                    loc = f":{d.line}:{d.col}" if d.line else ""
                    style = "bold red" if d.severity == "error" else "yellow"
                    rendered.append(f"[{d.severity}] ", style=style)
                    rendered.append(f"{d.code}{loc}: ", style="dim")
                    rendered.append(d.message)
                panel.update(rendered)
                panel.add_class("has-content")
                # Move the input cursor to the first diagnostic with a
                # known location so the user sees where to look.
                first_loc = next(
                    (d for d in diagnostics if d.line > 0 and d.severity == "error"),
                    None,
                )
                if first_loc is not None:
                    input_widget = self.query_one("#input", TextArea)
                    target_row = max(0, first_loc.line - 1)
                    target_col = max(0, first_loc.col)
                    try:
                        line_text = input_widget.document.get_line(target_row)
                        end_col = min(len(line_text), target_col + 1)
                        input_widget.selection = (  # type: ignore[assignment]
                            (target_row, target_col),
                            (target_row, end_col),
                        )
                    except Exception:
                        pass
            else:
                panel.update("")
                panel.remove_class("has-content")

        def _refresh_watches(self) -> None:
            panel: Static = self.query_one("#watches", Static)
            watches = self.session.watch_results()
            if not watches:
                panel.update("")
                panel.remove_class("has-content")
                return
            env_kinds = self.session.env_kinds()
            text = Text()
            first = True
            for expr, line in watches.items():
                if not first:
                    text.append("\n")
                first = False
                text.append("watch ", style="bold magenta")
                text.append(expr, style="bold")
                text.append(" => ", style="dim")
                text.append(_to_tui_rich_text(line, env_kinds=env_kinds))
            panel.update(text)
            panel.add_class("has-content")

        def _refresh_status(self) -> None:
            bar: Static = self.query_one("#status", Static)
            bar.update(self._status_text())

        def _status_text(self):  # type: ignore[no-untyped-def]
            compiler = self.session._compiler  # noqa: SLF001
            path = (
                str(self.session.loaded_path)
                if self.session.loaded_path is not None
                else "<no file>"
            )
            counts = "no env"
            algebra = ""
            if compiler is not None:
                # Show every populated bucket; suppress empty ones so a
                # tiny module's status line stays readable.
                parts: list[str] = []
                for label, mapping in (
                    ("obj", compiler.objects),
                    ("space", compiler.spaces),
                    ("morph", compiler.morphisms),
                    ("rule", compiler.rules),
                    ("prog", compiler.programs),
                    ("ded", compiler.deductions),
                    ("sig", compiler.signatures),
                    ("enc", compiler.encoders),
                    ("dec", compiler.decoders),
                    ("loss", compiler.losses),
                    ("bundle", compiler.bundles),
                    ("contr", compiler.contractions),
                ):
                    n = len(mapping)
                    if n:
                        parts.append(f"{n} {label}")
                qiec_count = len(qiec_bindings(self.session.module))
                if qiec_count:
                    parts.append(f"{qiec_count} qiec")
                counts = " · ".join(parts) if parts else "empty env"
                algebra = type(compiler.algebra).__name__
            text = Text()
            text.append("● ", style="bold green")
            text.append(path, style="bold")
            if algebra:
                text.append("  algebra:", style="dim")
                text.append(f" {algebra}", style="bold cyan")
            text.append("  ", style="dim")
            text.append(counts, style="dim")
            runtime_status = _runtime_status(self.session)
            text.append("  ", style="dim")
            text.append(
                runtime_status,
                style=(
                    "bold yellow"
                    if self.session.runtime_label == "detached"
                    else "bold green"
                ),
            )
            return text

        def _refresh_env(self, *, filter_text: str = "") -> None:
            tree: Tree = self.query_one("#env", Tree)
            tree.clear()
            compiler = self.session._compiler  # noqa: SLF001
            if compiler is None:
                tree.root.label = "(no module loaded)"
                return
            tree.root.label = (
                str(self.session.loaded_path)
                if self.session.loaded_path is not None
                else "<module>"
            )
            tree.root.expand()
            needle = filter_text.lower()

            def keep(name: str) -> bool:
                return not needle or needle in name.lower()

            # Walk the scope graph uniformly: for each top-level
            # bucket, add a category heading, then recursively expand
            # each binding's scope_children. Every node carries the
            # binding's full ``::`` path on ``data`` so the click
            # handler resolves through the same scope walker the
            # meta-commands use.
            _populate_scope_tree(tree.root, self.session, keep, filter_text=needle)

    return QvrRepl(session)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _enable_qvr_highlighting(text_area: object) -> bool:
    """Register the wheel-shipped QVR grammar with Textual's ``TextArea``."""

    try:
        _, language, _library = _load_parser()
        resource = files("quivers.dsl._grammar_data").joinpath("highlights.scm")
        if resource.is_file():
            query = resource.read_text()
        else:
            query = (
                Path(__file__).resolve().parents[3]
                / "grammars"
                / "qvr"
                / "queries"
                / "highlights.scm"
            ).read_text()
        text_area.register_language("qvr", language, query)  # type: ignore[attr-defined]
        text_area.language = "qvr"  # type: ignore[attr-defined]
    except ImportError, OSError, AttributeError, RuntimeError, TypeError, ValueError:
        return False
    return True


def _runtime_status(session: ReplSession) -> str:
    """The runtime fragment of the TUI status bar.

    Parameters
    ----------
    session : ReplSession
        The session.

    Returns
    -------
    str
        The active runtime, the transpile target whose capabilities
        ``:load`` reports (when one is set), and the most recent run: a
        successful entry point with its kind, value, type, and a
        program's log joint, or a failed one with its diagnostic code
        (``failed:spin[qiec-run-fuel]`` for a run out of fuel).
    """

    status = f"runtime:{session.runtime_label}"
    if session.options.target:
        status += f" target:{session.options.target}"
    failure = session.last_failure
    if failure is not None:
        name = failure.computation or "?"
        return f"{status} failed:{name}[{failure.code}]"
    entry = session.last_entry
    if entry is not None:
        data = entry.to_data()
        rendered = (
            f"{status} entry:{entry.entry.name}({entry.entry.kind}) "
            f"last:{entry.entry.name}={data['value']!r}:{data['result_type']}"
        )
        if entry.log_joint is not None:
            rendered += f" log_joint={entry.log_joint:.4g}"
        return rendered
    result = session.last_run
    if result is None:
        return status
    data = result.to_data()
    return f"{status} last:{result.computation}={data['value']!r}:{data['result_type']}"


_LOCATION_RE = __import__("re").compile(r"(\S+\.qvr):(\d+):(\d+)")


def _to_tui_rich_text(
    source: str,
    *,
    env_kinds: dict[str, str] | None = None,
    link_action: str | None = None,
):  # type: ignore[no-untyped-def]
    """Highlight TUI text, layering click metadata over colour styles.

    Rich supports overlapping spans. Applying click metadata as its own span
    keeps the highlighter's parsed true-colour style intact when Textual later
    resolves the combined style for a cell.
    """
    text = to_rich_text(source, env_kinds=env_kinds)
    if link_action is None:
        return text

    interesting = {"type", "function", "namespace"}
    offset = 0
    for span in tokenize(source, env_kinds=env_kinds):
        end = offset + len(span.text)
        if (
            span.token in interesting
            and span.text.replace("_", "").isalnum()
            and not span.text[0].isdigit()
        ):
            text.apply_meta(
                {"@click": f"{link_action}('{span.text}')"},
                start=offset,
                end=end,
            )
        offset = end
    return text


def _decorate_comment_line(line: str):  # type: ignore[no-untyped-def]
    """Render a ``-- ...`` comment line, linking any path:line:col span."""
    base = Style.parse("italic dim")
    out = Text(style=base)
    cursor = 0
    for m in _LOCATION_RE.finditer(line):
        if m.start() > cursor:
            out.append(line[cursor : m.start()])
        path, lineno, _col = m.group(1), m.group(2), m.group(3)
        link_style = base + Style(
            underline=True,
            meta={"@click": f"open_at('{path}', '{lineno}')"},
        )
        out.append(m.group(0), style=link_style)
        cursor = m.end()
    if cursor < len(line):
        out.append(line[cursor:])
    return out


def _word_prefix(line: str, col: int) -> str:
    """Return the identifier-ish substring ending at ``col``.

    A leading ``:`` is preserved so meta-command completion works.
    """
    i = col
    while i > 0 and (line[i - 1].isalnum() or line[i - 1] in "_:."):
        i -= 1
    return line[i:col]


def _load_history() -> list[str]:
    if not _HISTORY_PATH.exists():
        return []
    try:
        return [
            line.rstrip("\n")
            for line in _HISTORY_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError:
        return []


def _append_history(line: str) -> None:
    try:
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Click-handler logic (extracted so the unit suite can drive it
# without needing a full Textual App).
# ---------------------------------------------------------------------------


def resolve_click_target(node):  # type: ignore[no-untyped-def]
    """Return the ``::``-path to dispatch ``:info`` against, or
    ``None`` if the node is not bound to a single declaration.

    Each env-tree node that corresponds to a binding (top-level
    or scoped) carries its full ``::``-separated scope path on
    ``node.data`` (set at build time by ``_populate_scope_tree``).
    Category headings (``objects``, ``programs``, ...) carry
    ``data=None`` and produce no dispatch. Paths may contain any
    number of ``::`` segments; the click handler routes the whole
    string through the session's scope-aware ``info()``, which in
    turn resolves it via ``resolve_scoped_path``.
    """
    if getattr(node, "is_root", False):
        return None
    path = getattr(node, "data", None)
    if not isinstance(path, str) or not path:
        return None
    # Require every segment to be a valid identifier; rejects
    # spurious data such as a rich label that got into ``data``.
    segments = path.split(SCOPE_SEPARATOR)
    if not all(s.isidentifier() for s in segments):
        return None
    return path


# ---------------------------------------------------------------------------
# Env-tree population via the scope walker
# ---------------------------------------------------------------------------


# Display labels per top-level kind, in the order categories appear
# in the tree. Each entry is (category heading, compiler accessor
# attribute, the ScopeKind to tag the resulting refs with).
_TREE_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("objects", "objects", "object"),
    ("spaces", "spaces", "space"),
    ("morphisms", "morphisms", "morphism"),
    ("rules", "rules", "rule"),
    ("programs", "programs", "program"),
    ("deductions", "deductions", "deduction"),
    ("signatures", "signatures", "signature"),
    ("encoders", "encoders", "encoder"),
    ("decoders", "decoders", "decoder"),
    ("losses", "losses", "loss"),
    ("bundles", "bundles", "bundle"),
    ("contractions", "contractions", "contraction"),
)


def _populate_scope_tree(root, session, keep, *, filter_text):  # type: ignore[no-untyped-def]
    """Populate a Textual ``Tree`` from the session's compiler env.

    Walks the scope graph: each top-level binding is added under
    its category heading; each binding's ``scope_children`` are
    recursively added as nested nodes. Every binding node carries
    its full ``::`` path on ``node.data`` so the click handler
    dispatches the right scope-aware lookup.
    """
    assert isinstance(session, ReplSession)
    compiler = session._compiler  # noqa: SLF001
    if compiler is None:
        return
    # Each binding sits under its one top-level kind: a program without
    # typed parameters, which also compiles to a morphism, is listed as
    # the program whose sites it scopes.
    by_kind: dict[str, list[ScopedRef]] = {}
    for ref in top_level_bindings(compiler):
        if keep(ref.name):
            by_kind.setdefault(ref.kind, []).append(ref)
    for heading, _attr, kind in _TREE_CATEGORIES:
        refs = sorted(by_kind.get(kind, ()), key=lambda ref: ref.name)
        if not refs:
            continue
        cat_node = root.add(heading, expand=True)
        for ref in refs:
            _add_ref_node(cat_node, session, ref)

    grouped: dict[str, list] = {}
    for binding in qiec_bindings(session.module):
        if keep(binding.qualified_name):
            grouped.setdefault(binding.kind, []).append(binding)
    headings = {
        "index": "qiec indices",
        "index-constructor": "qiec index constructors",
        "family": "qiec families",
        "constructor": "qiec constructors",
        "effect": "qiec effects",
        "operation": "qiec operations",
        "instance": "qiec instances",
        "handler": "qiec handlers",
        "computation": "qiec computations",
    }
    for kind, bindings in grouped.items():
        cat_node = root.add(headings[kind], expand=True)
        for binding in bindings:
            label = render_qiec_signature(session.module, binding.qualified_name)
            cat_node.add_leaf(
                label or binding.qualified_name,
                data=binding.qualified_name,
            )


def _add_ref_node(parent, session, ref):  # type: ignore[no-untyped-def]
    """Add a tree node for ``ref`` under ``parent``; recursively
    expand its scope_children as nested nodes. The node's display
    label is the scope-aware type line; the node's ``data`` is the
    full path so the click handler can resolve it."""
    label = session._type_line_for_ref(ref)  # noqa: SLF001
    children = scope_children(ref)
    if not children:
        parent.add_leaf(label, data=ref.path)
        return
    node = parent.add(label, data=ref.path, expand=False)
    for child in children.values():
        _add_ref_node(node, session, child)


# ---------------------------------------------------------------------------
# Env-tree child builders
# ---------------------------------------------------------------------------
#
def _populate_children(node, items):  # type: ignore[no-untyped-def]
    for item in items:
        if isinstance(item, str):
            node.add_leaf(item)
            continue
        label, sub = item
        if sub:
            sub_node = node.add(label, expand=False)
            _populate_children(sub_node, sub)
        else:
            node.add_leaf(label)


__all__ = ["run_tui"]
