"""Plain tooling remains usable without the optional Rich renderers."""

from __future__ import annotations

import subprocess
import sys
import textwrap


_BLOCK_DEPENDENCY = """
import sys
from collections.abc import Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

class BlockDependency(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        \"\"\"Refuse the requested dependency in this isolated interpreter.

        Parameters
        ----------
        fullname : str
            The requested module.
        path : sequence of str or None
            Its parent package path.
        target : ModuleType or None
            The module being reloaded.

        Returns
        -------
        ModuleSpec or None
            None to let ordinary finders resolve permitted modules.

        Raises
        ------
        ModuleNotFoundError
            If the dependency is blocked.
        \"\"\"
        if fullname == blocked:
            raise ModuleNotFoundError("blocked test dependency", name=fullname)
        return None

sys.meta_path.insert(0, BlockDependency())
"""


def _run_isolated(blocked: str, checks: str) -> subprocess.CompletedProcess[str]:
    """Run import checks with one dependency absent.

    Parameters
    ----------
    blocked : str
        The module the import finder refuses.
    checks : str
        Assertions to execute after installing the finder.

    Returns
    -------
    subprocess.CompletedProcess[str]
        The child interpreter's status and diagnostic output.
    """
    return subprocess.run(
        [sys.executable, "-c", f"blocked = {blocked!r}\n" + _BLOCK_DEPENDENCY + checks],
        capture_output=True,
        text=True,
        check=False,
    )


def test_plain_cli_does_not_require_rich() -> None:
    """Raw tokens and plain plate tables work, while Rich views refuse."""
    checks = textwrap.dedent(
        """
        from quivers.cli import tokenize, to_pygments_pairs, to_rich_text
        from quivers.analysis import PlateGraph, render_table, render_table_plain
        from quivers.cli.repl_prompt import _render
        from quivers.cli.repl_session import Diagnostic, ReplResponse

        source = "object X : FinSet 3"
        assert tokenize(source)
        assert to_pygments_pairs(source)
        graph = PlateGraph(program_name="example")
        assert "example" in render_table_plain(graph)
        for renderer, value in ((to_rich_text, source), (render_table, graph)):
            try:
                renderer(value)
            except ModuleNotFoundError as error:
                assert error.name == "rich"
                assert "quivers[repl]" in str(error)
            else:
                raise AssertionError("Rich renderer accepted a missing dependency")
        _render(ReplResponse(
            body="plain response",
            diagnostics=(Diagnostic(severity="error", code="test", message="failure"),),
        ))
        """
    )
    result = _run_isolated("rich", checks)
    assert result.returncode == 0, result.stderr
    assert "plain response" in result.stdout
    assert "error[test]: failure" in result.stderr


def test_plain_repl_without_prompt_toolkit() -> None:
    """The standard-input frontend evaluates a command and handles exit."""
    checks = textwrap.dedent(
        """
        import io
        from quivers.cli import ReplSession, run_plain

        sys.stdin = io.StringIO(":help\\n:quit\\n")
        assert run_plain(ReplSession()) == 0
        sys.stdin = io.StringIO("")
        assert run_plain(ReplSession()) == 0
        """
    )
    result = _run_isolated("prompt_toolkit", checks)
    assert result.returncode == 0, result.stderr
    assert "qvr>" in result.stdout
    assert ":quit" in result.stdout


def test_broken_rich_installation_is_not_treated_as_absent() -> None:
    """An internal Rich import failure propagates with its module name."""
    checks = textwrap.dedent(
        """
        try:
            import quivers.cli
        except ModuleNotFoundError as error:
            assert error.name == "rich.style"
        else:
            raise AssertionError("internal Rich import failure was swallowed")
        """
    )
    result = _run_isolated("rich.style", checks)
    assert result.returncode == 0, result.stderr
