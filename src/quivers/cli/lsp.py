"""``qvr lsp`` subcommand: start the QVR language server.

By default runs over stdio (which is what every editor expects).
Pass ``--tcp PORT`` to bind a TCP listener for debugging or for
clients that prefer socket transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import argparse
import sys

if TYPE_CHECKING:
    from quivers.lsp import build_server

    _LSP_UNAVAILABLE = None
else:
    try:
        from quivers.lsp import build_server
    except ModuleNotFoundError as error:
        if error.name not in {"pygls", "lsprotocol"}:
            raise
        _LSP_UNAVAILABLE: ModuleNotFoundError | None = error
    else:
        _LSP_UNAVAILABLE = None


def main(args: argparse.Namespace) -> int:
    """Start the language server using the selected transport.

    Parameters
    ----------
    args : argparse.Namespace
        The TCP port and optional transpilation target.

    Returns
    -------
    int
        Zero after the server exits, or two when its dependencies are absent.
    """
    if _LSP_UNAVAILABLE is not None:
        sys.stderr.write(
            f"language server unavailable ({_LSP_UNAVAILABLE}); "
            "install with `pip install 'quivers[lsp]'`.\n"
        )
        return 2
    server = build_server(target=getattr(args, "target", None))
    if args.tcp is not None:
        server.start_tcp("127.0.0.1", args.tcp)
    else:
        server.start_io()
    return 0


def _entry() -> int:
    """Console-script entry point for ``qvr-lsp``."""
    parser = argparse.ArgumentParser(prog="qvr-lsp")
    parser.add_argument("--tcp", type=int, default=None, metavar="PORT")
    parser.add_argument(
        "--target",
        default=None,
        help="publish diagnostics for the features TARGET lacks",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="run over stdio (the default; accepted for compatibility).",
    )
    args = parser.parse_args()
    return main(args)


__all__ = []
