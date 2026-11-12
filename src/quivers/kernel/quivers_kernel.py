"""ipykernel Kernel subclass for QVR.

A thin adapter over [`ReplSession`][quivers.cli.ReplSession].
Cells with a leading ``:`` are routed through the same meta-command
dispatcher the REPL uses; bare cells are sent through
`ReplSession.dispatch` and evaluated as either a statement
block or an expression.

Highlighting is delegated to the front end via the standard
``mimetype`` field (``text/x-qvr``); JupyterLab picks up the
``CodeMirror`` mode the kernelspec advertises.
"""

from __future__ import annotations

from typing import Any

from ipykernel.kernelbase import Kernel

from quivers.cli.repl_session import ReplSession


class QuiversKernel(Kernel):
    """A Jupyter kernel that evaluates QVR cells in a `ReplSession`.

    Each line of a cell is dispatched to the session as the REPL would
    dispatch it, so a leading ``:`` runs a meta-command and other input
    is elaborated against the session's module. Jupyter constructs the
    kernel; `install_kernelspec` registers it.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments forwarded to ``ipykernel.kernelbase.Kernel``.
    """

    implementation = "quivers"
    implementation_version = "0.1.0"
    language = "qvr"
    language_version = "0.20"
    language_info = {
        "name": "qvr",
        "mimetype": "text/x-qvr",
        "file_extension": ".qvr",
        "pygments_lexer": "qvr",
        "codemirror_mode": "qvr",
    }
    banner = "quivers REPL kernel"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.repl = ReplSession()

    def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: dict[str, Any] | None = None,
        allow_stdin: bool = False,
        *,
        cell_id: str | None = None,
    ) -> dict[str, Any]:
        """Dispatch each chunk of a cell to the session.

        A ``:quit`` chunk ends the cell: the chunks after it are not
        dispatched, and the kernel stays up, since a kernel is shut down
        by its front end. Unless ``silent``, a note on standard error
        says so.

        Parameters
        ----------
        code : str
            The cell.
        silent : bool
            Whether to suppress the cell's output streams.
        store_history : bool
            Unused; part of the ipykernel signature.
        user_expressions : dict[str, Any] | None
            Unused; part of the ipykernel signature.
        allow_stdin : bool
            Unused; part of the ipykernel signature.
        cell_id : str | None
            Unused; part of the ipykernel signature.

        Returns
        -------
        dict[str, Any]
            The execute reply: ``"error"`` status if any chunk reported
            an error diagnostic, ``"ok"`` otherwise.
        """
        del store_history, user_expressions, allow_stdin, cell_id
        had_error = False
        for raw_line in _split_cell(code):
            response = self.repl.dispatch(raw_line)
            if response.body == "__quit__":
                if not silent:
                    self.send_response(
                        self.iopub_socket,
                        "stream",
                        {
                            "name": "stderr",
                            "text": (
                                ":quit ends this cell; shut the kernel down "
                                "from the front end\n"
                            ),
                        },
                    )
                break
            if not silent:
                if response.body:
                    stream = "stderr" if not response.ok else "stdout"
                    self.send_response(
                        self.iopub_socket,
                        "stream",
                        {"name": stream, "text": response.body + "\n"},
                    )
                for diag in response.diagnostics:
                    had_error = had_error or diag.severity == "error"
                    self.send_response(
                        self.iopub_socket,
                        "stream",
                        {
                            "name": "stderr",
                            "text": f"{diag.severity}[{diag.code}]: {diag.message}\n",
                        },
                    )
        return {
            "status": "error" if had_error else "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }

    def do_complete(self, code: str, cursor_pos: int) -> dict[str, Any]:
        from quivers.cli.repl_complete import all_completions

        prefix = _prefix_at(code, cursor_pos)
        completions = all_completions(self.repl, prefix)
        return {
            "matches": [c.text for c in completions],
            "cursor_start": cursor_pos - len(prefix),
            "cursor_end": cursor_pos,
            "metadata": {},
            "status": "ok",
        }

    def do_inspect(
        self,
        code: str,
        cursor_pos: int,
        detail_level: int = 0,
        omit_sections: Any = None,
    ) -> dict[str, Any]:
        del detail_level, omit_sections
        word = _word_at(code, cursor_pos)
        if not word:
            return {"status": "ok", "found": False, "data": {}, "metadata": {}}
        response = self.repl.info(word)
        if not response.ok:
            return {"status": "ok", "found": False, "data": {}, "metadata": {}}
        return {
            "status": "ok",
            "found": True,
            "data": {"text/plain": response.body},
            "metadata": {},
        }


def _split_cell(code: str) -> list[str]:
    """Treat a cell as a series of independent lines.

    Meta-commands are line-oriented; statements may span multiple lines
    but the parser handles any whitespace inside a single chunk. We
    split on blank-line boundaries so the user can write either form.
    """
    out: list[str] = []
    block: list[str] = []
    for line in code.splitlines():
        if not line.strip():
            if block:
                out.append("\n".join(block))
                block = []
            continue
        if line.startswith(":") and not block:
            out.append(line)
        else:
            block.append(line)
    if block:
        out.append("\n".join(block))
    return out


def _prefix_at(code: str, cursor_pos: int) -> str:
    text = code[:cursor_pos]
    i = len(text)
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] in "_:."):
        i -= 1
    return text[i:]


def _word_at(code: str, cursor_pos: int) -> str:
    i = cursor_pos
    while i > 0 and (code[i - 1].isalnum() or code[i - 1] == "_"):
        i -= 1
    j = cursor_pos
    while j < len(code) and (code[j].isalnum() or code[j] == "_"):
        j += 1
    return code[i:j]


__all__ = ["QuiversKernel"]
