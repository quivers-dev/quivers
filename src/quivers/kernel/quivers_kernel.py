"""ipykernel Kernel subclass for QVR.

A thin adapter over [`ReplSession`][quivers.cli.ReplSession].
Cells with a leading ``:`` are routed through the same meta-command
dispatcher the REPL uses; bare cells are sent through
`ReplSession.dispatch` and evaluated as either a statement
block or an expression.

The kernel advertises the bundled Pygments lexer. Live cell highlighting
depends on the notebook frontend; no CodeMirror mode is bundled.
"""

from __future__ import annotations

from typing import Any

from ipykernel.kernelbase import Kernel

from quivers import __version__
from quivers.cli.repl_complete import _completion_prefix, all_completions
from quivers.cli.repl_session import ReplSession
from quivers.dsl import parse, ParseError


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
    implementation_version = __version__
    language = "qvr"
    language_version = __version__
    language_info = {
        "name": "qvr",
        "mimetype": "text/x-qvr",
        "file_extension": ".qvr",
        "pygments_lexer": "qvr",
        "version": __version__,
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
            an error diagnostic or failed response, ``"ok"`` otherwise.
            Execution stops at the first failed chunk, including in silent mode.
        """
        del store_history, user_expressions, allow_stdin, cell_id
        failure: dict[str, Any] | None = None
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
            errors = [d for d in response.diagnostics if d.severity == "error"]
            if not response.ok or errors:
                failure = {
                    "ename": errors[0].code or "QvrError" if errors else "QvrError",
                    "evalue": "\n".join(d.message for d in errors) or response.body,
                    "traceback": [
                        f"{d.code or 'QvrError'}: {d.message}" for d in errors
                    ]
                    or [response.body],
                }
                if not silent:
                    self.send_response(self.iopub_socket, "error", failure)
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
                    self.send_response(
                        self.iopub_socket,
                        "stream",
                        {
                            "name": "stderr",
                            "text": f"{diag.severity}[{diag.code}]: {diag.message}\n",
                        },
                    )
        if failure is not None:
            return {
                "status": "error",
                "execution_count": self.execution_count,
                **failure,
            }
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }

    def do_complete(self, code: str, cursor_pos: int) -> dict[str, Any]:

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
    """Keep declaration blocks intact and dispatch top-level meta-command lines."""
    out: list[str] = []
    block: list[str] = []
    for line in code.splitlines():
        if line.startswith(":"):
            out.extend(_source_chunks(block))
            block = []
            out.append(line)
        else:
            block.append(line)
    out.extend(_source_chunks(block))
    return out


def _source_chunks(lines: list[str]) -> list[str]:
    """Split blank-separated declarations only at canonical parser boundaries.

    Parameters
    ----------
    lines : list[str]
        A contiguous source region between notebook meta-commands.

    Returns
    -------
    list[str]
        Complete source chunks, retaining blank lines within hanging headers.
    """
    if not any(line.strip() for line in lines):
        return []
    try:
        module = parse("\n".join(lines), file_path="<notebook>")
        boundaries = {
            line
            for statement in module.statements
            if isinstance(line := getattr(statement, "line", None), int)
        }
    except ParseError:
        # Mixed statement/expression cells need the same statement-first,
        # expression-second probes as ReplSession. Valid declaration cells
        # take the single full-parse path above, irrespective of blank count.
        boundaries = None
    out: list[str] = []
    block: list[str] = []
    for index, line in enumerate(lines, 1):
        if block and not block[-1].strip() and line and not line[0].isspace():
            complete = (
                index in boundaries
                if boundaries is not None
                else _complete_source(block)
            )
            if complete:
                out.append("\n".join(block).rstrip())
                block = []
        block.append(line)
    if any(line.strip() for line in block):
        out.append("\n".join(block).rstrip())
    return out


def _complete_source(lines: list[str]) -> bool:
    """Use the session's canonical statement and expression syntax probes.

    Parameters
    ----------
    lines : list[str]
        A proposed complete source chunk.

    Returns
    -------
    bool
        Whether the chunk parses as declarations or as a value expression.
    """
    source = "\n".join(lines)
    for candidate in (source, f"export {source}"):
        try:
            parse(candidate, file_path="<notebook>")
        except ParseError:
            continue
        return True
    return False


def _prefix_at(code: str, cursor_pos: int) -> str:
    return _completion_prefix(code, cursor_pos)


def _word_at(code: str, cursor_pos: int) -> str:
    i = cursor_pos
    while i > 0 and (code[i - 1].isalnum() or code[i - 1] in "_:."):
        i -= 1
    j = cursor_pos
    while j < len(code) and (code[j].isalnum() or code[j] in "_:."):
        j += 1
    return code[i:j]


__all__ = ["QuiversKernel"]
