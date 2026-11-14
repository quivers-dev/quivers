"""Command-line entry points for quivers.

The `main` function is registered as the ``qvr`` console script
in ``pyproject.toml``.

Subcommands:

- ``qvr check FILES...``: parse and compile every supplied ``.qvr``
  file, emitting structured diagnostics. Exits 0 on full success,
  non-zero when any file produces an error.
- ``qvr migrate --from VER --to VER PATHS...``: lower ``.qvr``
  source files from one tagged grammar revision to another, via
  panproto migrations composed from the in-tree
  ``grammars/qvr/vcs`` chain.
- ``qvr run FILE COMPUTATION [ARGS...]``: execute one checked QIEC
  computation with JSON value arguments and an explicit runtime configuration.

Output format: human-readable by default, structured JSON when
``--json`` is supplied. Each diagnostic carries:

- ``file``: source path,
- ``line``, ``col``: 1-indexed source location,
- ``severity``: ``"error"``, ``"warning"``, or ``"note"``,
- ``code``: stable diagnostic code (``parse``, ``compile``,
  ``effect_constraint``, ``residuated_constraint``),
- ``message``: human-readable description.

The Python surface beside `main` is the tooling the subcommands share:
`check_file` and `CheckDiagnostic` for checking, and `ReplSession`
with its completion, highlighting, and front-end helpers for driving
a REPL session from Python. `run_tui` binds lazily because its
Textual front end is an optional dependency.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import TYPE_CHECKING, Callable

from quivers.cli.check import CheckDiagnostic, CheckSeverity, check_file
from quivers.cli.check import main as check_main
from quivers.cli.repl_complete import (
    Completion,
    all_completions,
    public_meta_commands,
)
from quivers.cli.repl_highlight import (
    SEMANTIC_TOKEN_MODIFIERS,
    SEMANTIC_TOKEN_TYPES,
    STYLE_TABLE,
    Span,
    to_pygments_pairs,
    to_rich_text,
    to_semantic_token_data,
    to_semantic_token_legend,
    tokenize,
)
from quivers.cli.repl_prompt import run_plain
from quivers.cli.repl_session import (
    Diagnostic,
    ReplResponse,
    ReplSession,
    SessionOptions,
    render_qiec_signature,
    render_signature,
)

if TYPE_CHECKING:
    from quivers.cli.repl_tui import run_tui


def __getattr__(name: str) -> Callable[[ReplSession], int]:
    """Bind the re-exports whose modules need optional dependencies.

    Importing `run_tui` loads Textual, which only the ``repl`` extra
    installs, so the package binds it on first access rather than at
    import time.

    Parameters
    ----------
    name : str
        The attribute being looked up.

    Returns
    -------
    Callable[[ReplSession], int]
        The lazily bound front end.

    Raises
    ------
    AttributeError
        When ``name`` is not a lazily bound re-export.
    """
    if name == "run_tui":
        repl_tui = importlib.import_module("quivers.cli.repl_tui")
        return repl_tui.run_tui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> int:
    """Run the ``qvr`` command line on ``sys.argv``.

    Returns
    -------
    int
        The process exit status of the selected subcommand.
    """

    parser = argparse.ArgumentParser(prog="qvr", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser(
        "check",
        help="Parse + compile .qvr files and report diagnostics.",
    )
    check.add_argument("files", nargs="+", help="Paths to .qvr files.")
    check.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON diagnostics on stdout.",
    )
    check.add_argument(
        "--target",
        default=None,
        help="Also diagnose the features this transpile target lacks.",
    )

    repl = sub.add_parser(
        "repl",
        help="Start an interactive type-exploration REPL.",
    )
    repl.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Optional .qvr file to :load on startup.",
    )
    repl.add_argument(
        "--plain",
        action="store_true",
        help="Use the prompt_toolkit single-line front end instead of the TUI.",
    )
    repl.add_argument(
        "--target",
        default=None,
        metavar="TARGET",
        help=(
            "Report this transpile target's capability diagnostics on :load "
            "and :reload, as `qvr check --target` does."
        ),
    )

    run = sub.add_parser(
        "run",
        help="Execute one entry point: a `define` computation or a `program`.",
    )
    run.add_argument("file", help="Path to the .qvr source file.")
    run.add_argument(
        "computation",
        nargs="?",
        default=None,
        help=(
            "Entry point to execute: a `define` computation or a `program`. "
            "Omit it, or pass --list, to list the file's entry points."
        ),
    )
    run.add_argument(
        "arguments",
        nargs="*",
        help="Value arguments as JSON literals, in declaration order.",
    )
    run.add_argument(
        "--list",
        action="store_true",
        help="List the file's entry points with their signatures and exit.",
    )
    run.add_argument(
        "--static",
        action="append",
        default=[],
        metavar="NAME=TERM",
        help="Closed static specialization; repeat once per telescope binder.",
    )
    run.add_argument(
        "--data",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help=(
            "A program parameter by name (its data, an observation, an "
            "index map, or a scalar); repeatable."
        ),
    )
    run.add_argument(
        "--site",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help=(
            "Condition a program's sample site on a value; every other site "
            "is drawn. Repeatable."
        ),
    )
    run.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Seed the reference generator a program run draws with.",
    )
    run.add_argument(
        "--runtime",
        default=None,
        metavar="FILE.json",
        help="Explicit JSON runtime-provider configuration.",
    )
    run.add_argument(
        "--fuel",
        type=int,
        default=None,
        metavar="STEPS",
        help=(
            "Stop after this many evaluation steps with a qiec-run-fuel "
            "diagnostic; recursion may otherwise run forever."
        ),
    )
    run.add_argument(
        "--trace",
        action="store_true",
        help="Write stable execution trace events to stderr.",
    )
    run.add_argument(
        "--json",
        action="store_true",
        help="Emit the result, diagnostics, and trace as JSON.",
    )

    lsp = sub.add_parser(
        "lsp",
        help="Run the QVR Language Server (LSP 3.17 over stdio).",
    )
    lsp.add_argument(
        "--tcp",
        type=int,
        default=None,
        metavar="PORT",
        help="Bind to TCP port instead of stdio.",
    )
    lsp.add_argument(
        "--target",
        default=None,
        help="Diagnose the features this transpile target lacks.",
    )

    migrate = sub.add_parser(
        "migrate",
        help="Migrate .qvr source files between tagged grammar revisions.",
    )
    migrate.add_argument(
        "--from",
        dest="from_ref",
        default=None,
        help="Source grammar revision (git tag or commit id). Required when "
        "migrating because QVR source does not yet carry a version marker.",
    )
    migrate.add_argument(
        "--to",
        dest="to_ref",
        default=None,
        help="Target grammar revision. Defaults to the upcoming "
        "release (last entry on the chain).",
    )
    migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing output.",
    )
    migrate.add_argument(
        "--output",
        default=None,
        help="Write migrated files under this directory rather than "
        "overwriting in place.",
    )
    migrate.add_argument(
        "--check",
        action="store_true",
        help="Validate the migration chain against the panproto VCS schema "
        "diff, converter coverage, and asserted identity hops. Non-zero "
        "exit on uncovered or unexpected drift. Does not migrate files and "
        "does not require --from.",
    )
    migrate.add_argument(
        "paths",
        nargs="*",
        help=".qvr files or directories to migrate.",
    )

    kernel = sub.add_parser(
        "kernel",
        help="Jupyter kernel: install kernelspec or run a kernel.",
    )
    kernel_sub = kernel.add_subparsers(dest="kernel_cmd", required=True)
    kernel_install = kernel_sub.add_parser(
        "install", help="Register the `quivers` Jupyter kernelspec."
    )
    kernel_install.add_argument(
        "--user",
        action="store_true",
        help="Install to the user kernel directory instead of system.",
    )
    kernel_install.add_argument(
        "--prefix",
        default=None,
        help="Install kernelspec under PREFIX/share/jupyter/kernels.",
    )
    kernel_sub.add_parser(
        "run", help="Run as a Jupyter kernel (invoked by Jupyter itself)."
    )
    kernel_run_f = kernel_sub.add_parser(
        "_run_f",
        help=argparse.SUPPRESS,
    )
    kernel_run_f.add_argument("connection_file")

    transpile_sub = sub.add_parser(
        "transpile",
        help="Transpile a .qvr file to other probabilistic-programming "
        "languages (Stan, NumPyro, Pyro, PyMC, ...).",
    )
    transpile_sub.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to the .qvr source file.",
    )
    transpile_sub.add_argument(
        "--to",
        default=None,
        help="Target backend name (e.g. 'stan', 'numpyro').",
    )
    transpile_sub.add_argument(
        "--to-all",
        action="store_true",
        help="Transpile to every registered backend.",
    )
    transpile_sub.add_argument(
        "--list-targets",
        action="store_true",
        help="List every registered backend and exit.",
    )
    transpile_sub.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the single-target output to this path instead of stdout.",
    )
    transpile_sub.add_argument(
        "-d",
        "--out-dir",
        default=None,
        help="With --to-all, write one file per backend under this "
        "directory (named <stem>.<extension>).",
    )

    args = parser.parse_args()
    if args.cmd == "check":
        return check_main(args.files, json_output=args.json, target=args.target)
    if args.cmd == "repl":
        from quivers.cli.repl import main as repl_main

        return repl_main(args)
    if args.cmd == "run":
        from quivers.cli.run import main as run_main

        return run_main(args)
    if args.cmd == "lsp":
        from quivers.cli.lsp import main as lsp_main

        return lsp_main(args)
    if args.cmd == "migrate":
        from quivers.cli.migrate import main as migrate_main

        return migrate_main(args)
    if args.cmd == "kernel":
        from quivers.kernel.install import main as kernel_main

        return kernel_main(args)
    if args.cmd == "transpile":
        from quivers.cli.transpile import main as transpile_main

        return transpile_main(args)
    parser.print_help(sys.stderr)
    return 2


__all__ = [
    "main",
    "CheckDiagnostic",
    "CheckSeverity",
    "check_file",
    "Diagnostic",
    "ReplResponse",
    "ReplSession",
    "SessionOptions",
    "render_qiec_signature",
    "render_signature",
    "Completion",
    "all_completions",
    "public_meta_commands",
    "SEMANTIC_TOKEN_MODIFIERS",
    "SEMANTIC_TOKEN_TYPES",
    "STYLE_TABLE",
    "Span",
    "to_pygments_pairs",
    "to_rich_text",
    "to_semantic_token_data",
    "to_semantic_token_legend",
    "tokenize",
    "run_plain",
    "run_tui",
]
