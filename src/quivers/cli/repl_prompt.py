"""prompt_toolkit single-line / multi-line REPL frontend.

Used when stdin/stdout is not a TTY or when the user passes
``--plain``. Honours the same `ReplSession` as the Textual TUI
and the Jupyter kernel; the only difference is how input is gathered
and output is rendered.

Two render paths:

- If ``rich`` is available we render `ReplResponse` bodies with
  syntax highlighting via [`to_rich_text`][quivers.cli.to_rich_text].
- Otherwise we print the body verbatim. Diagnostics always print to
  stderr.

History persists at ``~/.config/quivers/history``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from quivers.cli.repl_complete import all_completions
from quivers.cli.repl_session import Diagnostic, ReplResponse, ReplSession
from quivers.dsl.pygments_lexer import QvrLexer

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import CompleteEvent, Completer, Completion
    from prompt_toolkit.document import Document
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.lexers import PygmentsLexer

    _PROMPT_AVAILABLE = True
else:
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import CompleteEvent, Completer, Completion
        from prompt_toolkit.document import Document
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.lexers import PygmentsLexer
    except ModuleNotFoundError as error:
        if error.name != "prompt_toolkit":
            raise
        _PROMPT_AVAILABLE = False
    else:
        _PROMPT_AVAILABLE = True

if TYPE_CHECKING:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax

    _RICH_AVAILABLE = True
else:
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.syntax import Syntax
    except ModuleNotFoundError as error:
        if error.name != "rich":
            raise
        _RICH_AVAILABLE = False
    else:
        _RICH_AVAILABLE = True


def run_plain(session: ReplSession) -> int:
    """Run the single-line REPL front end on ``session``.

    The front end reads with prompt_toolkit when it is installed and
    with bare standard input otherwise.

    Parameters
    ----------
    session : ReplSession
        The session that evaluates each line.

    Returns
    -------
    int
        The process exit status, ``0`` when the user quits.
    """
    getline: Callable[[], str | None] = _bare_input
    if _PROMPT_AVAILABLE:
        history_path = _history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)

        class _Completer(Completer):
            def get_completions(
                self, document: Document, complete_event: CompleteEvent
            ) -> Iterable[Completion]:
                """Yield completions at the current source position.

                Parameters
                ----------
                document : Document
                    The editable input buffer.
                complete_event : CompleteEvent
                    The completion request.

                Returns
                -------
                Iterable[Completion]
                    Matching names with their replacement positions.
                """
                del complete_event
                word = document.get_word_before_cursor(WORD=True)
                for c in all_completions(session, word):
                    yield Completion(
                        text=c.text,
                        start_position=-len(word),
                        display_meta=c.detail,
                    )

        ps: PromptSession[str] = PromptSession(
            message="qvr> ",
            history=FileHistory(str(history_path)),
            completer=_Completer(),
            lexer=PygmentsLexer(QvrLexer),
            multiline=False,
        )

        getline = ps.prompt

    while True:
        try:
            line = getline()
        except EOFError, KeyboardInterrupt:
            sys.stdout.write("\n")
            return 0
        if line is None:
            return 0
        # Allow `:reload` to auto-fire if file changed on disk.
        auto = session.autoreload_if_stale()
        if auto is not None:
            _render(auto)
        response = session.dispatch(line)
        if response.body == "__quit__":
            return 0
        _render(response)


def _bare_input() -> str | None:
    """Read one line from standard input.

    Returns
    -------
    str or None
        The input line, or None at end of input.
    """
    try:
        return input("qvr> ")
    except EOFError:
        return None


def _render(response: ReplResponse) -> None:
    """Print one response with the available terminal renderer.

    Parameters
    ----------
    response : ReplResponse
        The body and diagnostics to display.
    """
    if _RICH_AVAILABLE:
        console = Console()
        if response.body:
            if response.body_kind == "qvr":
                console.print(Syntax(response.body, "qvr", theme="ansi_dark"))
            elif response.body_kind == "json":
                console.print(Syntax(response.body, "json", theme="ansi_dark"))
            elif response.body_kind == "markdown":
                console.print(Markdown(response.body))
            else:
                console.print(response.body)
        for diagnostic in response.diagnostics:
            console.print(
                _format_diag(diagnostic),
                style="red" if diagnostic.severity == "error" else "yellow",
            )
    else:
        if response.body:
            sys.stdout.write(response.body + "\n")
        for diagnostic in response.diagnostics:
            sys.stderr.write(_format_diag(diagnostic) + "\n")


def _format_diag(d: Diagnostic) -> str:
    """Format a diagnostic with its source position.

    Parameters
    ----------
    d : Diagnostic
        The diagnostic to display.

    Returns
    -------
    str
        Its severity, code, position, and message.
    """
    loc = f":{d.line}:{d.col}" if d.line else ""
    return f"{d.severity}[{d.code}]{loc}: {d.message}"


def _history_path() -> Path:
    """Return the configured REPL history file.

    Returns
    -------
    Path
        The history path under the user's configuration directory.
    """
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "quivers" / "history"


__all__ = ["run_plain"]
