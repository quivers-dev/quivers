"""``python -m quivers.kernel`` entry point.

Dispatches to the ``qvr kernel`` handler in ``quivers.kernel.install`` so the kernelspec
``argv`` invocation route works for ``install``, ``run``, and the
internal ``_run_f`` variant Jupyter passes with a connection file.
"""

from __future__ import annotations

import argparse
import sys

from quivers.kernel.install import main as kmain


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m quivers.kernel")
    sub = parser.add_subparsers(dest="kernel_cmd", required=True)
    install = sub.add_parser("install")
    install.add_argument("--user", action="store_true")
    install.add_argument("--prefix", default=None)
    sub.add_parser("run")
    run_f = sub.add_parser("_run_f")
    run_f.add_argument("connection_file")
    args = parser.parse_args()

    return kmain(args)


if __name__ == "__main__":
    sys.exit(main())
